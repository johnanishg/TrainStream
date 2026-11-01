"""Memory-lean output layer + cross-entropy.

For a large vocabulary the logits are the biggest tensor of the step: Qwen's
151,936-token head turns a 4 x 512 batch into 1.16 GiB of fp32 logits, plus the
same again for their gradient. ``chunked_cross_entropy`` computes
``cross_entropy(hidden @ weight.T + bias, targets)`` a slice of rows at a time
and recomputes each slice in backward, so no more than one slice of logits
ever exists. Results match the dense computation (same fp32 log-sum-exp).

Preference training (DPO) needs per-sequence log-probabilities of the policy
and of the reference model for a chosen and a rejected answer: four full
logit tensors with a dense head. ``sequence_logprobs`` gives them from the
hidden states in the same chunked way, and ``dpo_loss`` combines them.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F


def _chunks(n: int, rows: int):
    for b in range(0, n, rows):
        yield b, min(n, b + rows)


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, bias, targets, rows, ignore_index, denom):
        n = hidden.shape[0]
        lse = torch.empty(n, device=hidden.device, dtype=torch.float32)
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        valid = targets != ignore_index
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))
        for b, e in _chunks(n, rows):
            logits = F.linear(hidden[b:e], weight, bias).float()
            l = torch.logsumexp(logits, dim=-1)
            lse[b:e] = l
            picked = logits.gather(1, safe_t[b:e, None]).squeeze(1)
            total += ((l - picked) * valid[b:e]).sum()
        ctx.save_for_backward(hidden, weight, bias if bias is not None else torch.empty(0), safe_t, valid, lse)
        ctx.rows = rows
        ctx.has_bias = bias is not None
        ctx.denom = denom
        return total / denom

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        hidden, weight, bias, safe_t, valid, lse = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        scale = grad_out.float() / ctx.denom
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_b = ctx.has_bias and ctx.needs_input_grad[2]
        gh = torch.empty_like(hidden) if need_h else None
        gw = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if need_w else None
        gb = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if need_b else None
        for b, e in _chunks(hidden.shape[0], ctx.rows):
            h = hidden[b:e]
            p = F.linear(h, weight, bias).float()
            p.sub_(lse[b:e, None]).exp_()                                     # softmax, in place
            p.scatter_add_(1, safe_t[b:e, None], -torch.ones_like(p[:, :1]))  # - one_hot(target)
            p.mul_(valid[b:e, None] * scale)
            g = p.to(h.dtype) if h.dtype != torch.float32 else p
            if need_h:
                gh[b:e] = g @ weight.to(g.dtype)
            if need_w:
                gw += p.t() @ h.float()
            if need_b:
                gb += p.sum(0)
        return (gh, gw.to(weight.dtype) if need_w else None, gb.to(bias.dtype) if need_b else None,
                None, None, None, None)


class _ChunkedLinearNLL(torch.autograd.Function):
    """Per-row -log p(target) of ``hidden @ weight.T + bias`` (0 where ignored)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, bias, targets, rows, ignore_index):
        n = hidden.shape[0]
        nll = torch.empty(n, device=hidden.device, dtype=torch.float32)
        lse = torch.empty(n, device=hidden.device, dtype=torch.float32)
        valid = targets != ignore_index
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))
        for b, e in _chunks(n, rows):
            logits = F.linear(hidden[b:e], weight, bias).float()
            l = torch.logsumexp(logits, dim=-1)
            lse[b:e] = l
            nll[b:e] = (l - logits.gather(1, safe_t[b:e, None]).squeeze(1)) * valid[b:e]
        ctx.save_for_backward(hidden, weight, bias if bias is not None else torch.empty(0), safe_t, valid, lse)
        ctx.rows = rows
        ctx.has_bias = bias is not None
        return nll

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_nll):
        hidden, weight, bias, safe_t, valid, lse = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        coef = valid * grad_nll.float()
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_b = ctx.has_bias and ctx.needs_input_grad[2]
        gh = torch.empty_like(hidden) if need_h else None
        gw = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if need_w else None
        gb = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if need_b else None
        for b, e in _chunks(hidden.shape[0], ctx.rows):
            h = hidden[b:e]
            p = F.linear(h, weight, bias).float()
            p.sub_(lse[b:e, None]).exp_()
            p.scatter_add_(1, safe_t[b:e, None], -torch.ones_like(p[:, :1]))
            p.mul_(coef[b:e, None])
            g = p.to(h.dtype) if h.dtype != torch.float32 else p
            if need_h:
                gh[b:e] = g @ weight.to(g.dtype)
            if need_w:
                gw += p.t() @ h.float()
            if need_b:
                gb += p.sum(0)
        return (gh, gw.to(weight.dtype) if need_w else None, gb.to(bias.dtype) if need_b else None,
                None, None, None)


def sequence_logprobs(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                      bias: Optional[torch.Tensor] = None, *, chunk_size: int = 1024, ignore_index: int = -100,
                      shift: bool = True, average: bool = False) -> torch.Tensor:
    """Log-probability of each sequence's labels, without materializing the logits.

    hidden  [B, T, D] final hidden states (after the last norm)
    weight  [V, D] output-layer weight; bias [V] optional
    labels  [B, T] token ids; ``ignore_index`` positions (the prompt, padding) do not count
    shift   labels are the input ids, as Hugging Face passes them: position t
            predicts label t + 1 (default). False: position t predicts label t.
    average the mean log-probability per counted token instead of the sum.
    Returns [B] float32, differentiable with respect to hidden, weight and bias.
    """
    if shift:
        hidden, labels = hidden[:, :-1, :], labels[:, 1:]
    B, T = labels.shape[0], labels.shape[1]
    h = hidden.reshape(-1, hidden.shape[-1])
    t = labels.reshape(-1).to(h.device)
    if t.shape[0] != h.shape[0]:
        raise ValueError(f"{h.shape[0]} hidden rows but {t.shape[0]} labels")
    nll = _ChunkedLinearNLL.apply(h, weight, bias, t, max(1, int(chunk_size)), ignore_index)
    out = -nll.view(B, T).sum(-1)
    if average:
        out = out / (t.view(B, T) != ignore_index).sum(-1).clamp_min(1)
    return out


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, reference_chosen: torch.Tensor,
             reference_rejected: torch.Tensor, *, beta: float = 0.1, label_smoothing: float = 0.0):
    """Direct Preference Optimization loss (Rafailov et al., 2023) from
    per-sequence log-probabilities ([B] each, e.g. from sequence_logprobs).

    Returns ``(loss, chosen_rewards, rejected_rewards)``: the mean loss and the
    implicit rewards beta * (policy - reference) of each answer (detached).
    ``label_smoothing`` > 0 is the conservative DPO variant for noisy preferences.
    """
    ref_c = reference_chosen.to(policy_chosen.device, torch.float32)
    ref_r = reference_rejected.to(policy_chosen.device, torch.float32)
    logits = beta * ((policy_chosen.float() - policy_rejected.float()) - (ref_c - ref_r))
    losses = -F.logsigmoid(logits) * (1.0 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    chosen_rewards = (beta * (policy_chosen.float() - ref_c)).detach()
    rejected_rewards = (beta * (policy_rejected.float() - ref_r)).detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                          bias: Optional[torch.Tensor] = None, *, chunk_size: int = 1024,
                          ignore_index: int = -100, reduction: str = "mean",
                          num_items: Optional[int] = None) -> torch.Tensor:
    """Cross-entropy of ``hidden @ weight.T (+ bias)`` without materializing the logits.

    hidden  [..., D] final hidden states (after the last norm)
    weight  [V, D] output-layer weight; bias [V] optional
    targets [...] class indices; ``ignore_index`` positions do not count
    reduction "mean" (over counted positions, or over ``num_items`` when given,
    as Hugging Face does for gradient accumulation) or "sum".
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1).to(h.device)
    if t.shape[0] != h.shape[0]:
        raise ValueError(f"{h.shape[0]} hidden rows but {t.shape[0]} targets")
    if reduction == "sum":
        denom = 1.0
    elif num_items is not None:
        denom = float(num_items.item() if torch.is_tensor(num_items) else num_items)
    else:
        denom = float(max(int((t != ignore_index).sum()), 1))
    return _ChunkedLinearCE.apply(h, weight, bias, t, max(1, int(chunk_size)), ignore_index, denom)


def causal_lm_loss(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                   bias: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
    """Next-token loss as Hugging Face computes it (labels shifted by one)."""
    return chunked_cross_entropy(hidden[..., :-1, :], weight, labels[..., 1:], bias, **kwargs)
