"""One autograd function for a PEFT LoRA layer's adapter.

PEFT adds a LoRA adapter as ``result + lora_B(lora_A(dropout(x))) * scaling``.
Under bf16 autocast with fp32 adapters (PEFT's default), that is a cast of x
to fp32 and back, two matmuls, a multiply, an add and a cast of the sum, and
autograd keeps a bf16 copy of x for lora_A's weight gradient, per adapted
Linear. Across a 14B model's 336 adapted Linears these small kernels add up
to thousands of launches and a copy of every adapted input per step.

LoRAFn computes the same sum with two matmuls, the second accumulating into
the base layer's output (``addmm(result, x A^T, B^T, alpha=scaling)``), in
the dtype autocast would use. It keeps x itself (the tensor the base layer
read, shared by q / k / v) and the small x A^T for backward, where

    dB = s (x A^T)^T g      dA = s (g B)^T x      dx = s (g B) A

are three matmuls on [M, r] intermediates. Results match PEFT's up to the
rounding of its intermediate products, so wrap(fused_lora="auto") turns it on
only where the base is quantized (already not bit-exact).
"""

from __future__ import annotations

import types
from typing import List

import torch
from torch import nn


class LoRAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, result, x, A, B, scaling, dt):
        with torch.autocast("cuda", enabled=False):
            n = result.shape[-1]
            x2 = x.reshape(-1, x.shape[-1])
            xa = x2.to(dt) @ A.to(dt).t()  # [M, r]
            r2 = result.reshape(-1, n)
            if r2.dtype == dt:
                out = torch.addmm(r2, xa, B.to(dt).t(), alpha=scaling)
            else:  # PEFT adds in the adapter's dtype, then casts back to the result's
                out = torch.addmm(r2.to(dt), xa, B.to(dt).t(), alpha=scaling).to(r2.dtype)
        ctx.save_for_backward(x, xa, A, B)
        ctx.scaling, ctx.dt = scaling, dt
        return out.view(result.shape)

    @staticmethod
    def backward(ctx, g):
        x, xa, A, B = ctx.saved_tensors
        s, dt = ctx.scaling, ctx.dt
        need_x, need_a, need_b = ctx.needs_input_grad[1:4]
        dx = dA = dB = None
        with torch.autocast("cuda", enabled=False):
            g2 = g.reshape(-1, g.shape[-1]).to(dt)
            gb = g2 @ (B.to(dt) * s)  # s (g B): [M, r]
            if need_x:
                dx = (gb @ A.to(dt)).view(x.shape).to(x.dtype)
            if need_a:
                dA = (gb.t() @ x.reshape(-1, x.shape[-1]).to(dt)).to(A.dtype)
            if need_b:
                dB = (g2.t() @ (xa * s)).to(B.dtype)
        return g, dx, dA, dB, None, None


def _plain(layer, name) -> bool:
    """A vanilla adapter LoRAFn computes: no dropout, no DoRA / variant, no LoRA bias."""
    if name not in layer.lora_A or name in getattr(layer, "lora_variant", {}):
        return False
    a, b, drop = layer.lora_A[name], layer.lora_B[name], layer.lora_dropout[name]
    if not (type(a) is nn.Linear and type(b) is nn.Linear) or a.bias is not None or b.bias is not None:
        return False
    return isinstance(drop, nn.Identity) or (isinstance(drop, nn.Dropout) and (drop.p == 0 or not drop.training))


def _forward(self, x, *args, **kwargs):
    adapters = self.active_adapters
    if (args or kwargs or self.disable_adapters or self.merged or len(adapters) != 1 or not x.is_cuda
            or not _plain(self, adapters[0])):
        return type(self).forward(self, x, *args, **kwargs)
    name = adapters[0]
    A, B = self.lora_A[name].weight, self.lora_B[name].weight
    result = self.base_layer(x)
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else A.dtype
    return LoRAFn.apply(result, x, A, B, float(self.scaling[name]), dt)


def _is_lora_linear(m) -> bool:
    return (hasattr(m, "lora_A") and hasattr(m, "lora_B") and hasattr(m, "base_layer")
            and isinstance(m.base_layer, nn.Linear) and type(m).__name__ == "Linear"
            and "forward" not in m.__dict__)


def install(model: nn.Module) -> List[nn.Module]:
    """Route `model`'s PEFT LoRA Linears through LoRAFn (each call falls back to PEFT's own
    forward when its adapter isn't a plain one). Returns the patched layers."""
    patched = []
    for m in model.modules():
        if _is_lora_linear(m):
            m.forward = types.MethodType(_forward, m)
            patched.append(m)
    return patched


def uninstall(layers) -> None:
    for m in layers:
        m.__dict__.pop("forward", None)
