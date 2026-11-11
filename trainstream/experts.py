"""Mixture-of-experts layers whose frozen experts stay in host RAM: only routed experts are copied in.

A MoE decoder layer holds every expert's weights, but each token uses a few
of them (8 of 128 in Qwen3-30B-A3B). Paging a whole layer, as the block pager
does, copies all of them over PCIe every pass. With wrap(moe_paging=True) the
frozen expert tensors of Hugging Face's stacked-experts modules (gate_up_proj
[E, 2I, H] and down_proj [E, H, I]; Qwen3-MoE, Qwen2-MoE, Mixtral, ...) live
in pinned host memory, and the layer copies in only the experts some token was
routed to, one expert at a time, again in backward instead of keeping them.

It pays when a step routes to few experts: short sequences, small batches,
generation during RL. With thousands of tokens per step nearly every expert
is hit and the copy volume is the same as paging the layer. The expert
weights must be frozen (LoRA on the attention, the router, the dense layers).
"""

from __future__ import annotations

import types
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from . import _C

NAMES = ("gate_up_proj", "down_proj")


def is_stacked_experts(m: nn.Module) -> bool:
    """Hugging Face's stacked-experts module: 3-D gate_up_proj / down_proj, num_experts, act_fn."""
    return (all(isinstance(getattr(m, n, None), nn.Parameter) and getattr(m, n).dim() == 3 for n in NAMES)
            and isinstance(getattr(m, "num_experts", None), int) and callable(getattr(m, "act_fn", None))
            and not getattr(m, "has_bias", False))


class ExpertFn(torch.autograd.Function):
    """One expert, x [T, H] -> [T, H], with weights fetched from the host in forward and again in backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, pager, e):
        w1, w2 = pager.fetch(e, x.device)
        gate, up = F.linear(x, w1).chunk(2, dim=-1)
        y = F.linear(pager.module.act_fn(gate) * up, w2)
        ctx.save_for_backward(x, gate, up)
        ctx.pager, ctx.e = pager, e
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        x, gate, up = ctx.saved_tensors
        w1, w2 = ctx.pager.fetch(ctx.e, gy.device)
        da = gy.to(w2.dtype) @ w2  # y = a w2^T
        with torch.enable_grad():
            g = gate.detach().requires_grad_()
            u = up.detach().requires_grad_()
            a = ctx.pager.module.act_fn(g) * u
            dg, du = torch.autograd.grad(a, (g, u), da.to(a.dtype))
        dx = torch.cat((dg, du), dim=-1).to(w1.dtype) @ w1  # h = x w1^T
        return dx.to(x.dtype), None, None


def _forward(self, hidden_states, top_k_index, top_k_weights):
    """Hugging Face's loop over the experts some token was routed to, with each expert's
    weights copied in from the host for it."""
    pager = self._ts_expert_pager
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().view(-1).tolist()
    for e in hit:
        if e == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(mask[e])
        out = ExpertFn.apply(hidden_states[token_idx], pager, e)
        out = out * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, out.to(final.dtype))
    pager.calls += 1
    pager.experts_used += len([e for e in hit if e != self.num_experts])
    return final


class ExpertPager:
    """The host copies of one stacked-experts module and the copies in."""

    def __init__(self, module: nn.Module, device):
        self.module = module
        self.device = torch.device(device)
        self.host = {}
        self.calls = self.experts_used = self.bytes_in = 0
        for n in NAMES:
            p = getattr(module, n)
            src = p.detach()
            h = _C.pinned_empty(list(src.shape), src.dtype, "param.experts", False, False)
            h.copy_(src)
            self.host[n] = h
            with torch.no_grad():  # the module keeps its shapes; the values live on the host
                p.data = torch.empty(src.shape, dtype=src.dtype, device=self.device)
            p.untyped_storage().resize_(0)
            p._ts_expert = True
        module._ts_expert_pager = self
        module.forward = types.MethodType(_forward, module)

    def fetch(self, e: int, device):
        w1 = self.host["gate_up_proj"][e].to(device, non_blocking=True)
        w2 = self.host["down_proj"][e].to(device, non_blocking=True)
        self.bytes_in += w1.numel() * w1.element_size() + w2.numel() * w2.element_size()
        return w1, w2

    @property
    def nbytes(self) -> int:
        return sum(h.numel() * h.element_size() for h in self.host.values())

    def remove(self) -> None:
        """Put the experts back in VRAM, as they were."""
        m = self.module
        m.__dict__.pop("forward", None)
        m.__dict__.pop("_ts_expert_pager", None)
        for n in NAMES:
            p = getattr(m, n)
            with torch.no_grad():
                p.data = self.host[n].to(self.device)
            p.__dict__.pop("_ts_expert", None)
        self.host = {}


def install(model: nn.Module, device) -> List[ExpertPager]:
    """Page the frozen stacked experts of `model` from host RAM (see the module docstring)."""
    out = []
    for m in model.modules():
        if is_stacked_experts(m) and "forward" not in m.__dict__ and not any(getattr(m, n).requires_grad for n in NAMES):
            out.append(ExpertPager(m, device))
    return out
