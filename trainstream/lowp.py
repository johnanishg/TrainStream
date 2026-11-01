"""Tensor-core matmuls with block-scaled frozen weights (NVFP4, MXFP8).

On GPUs whose tensor cores take block-scaled operands (Blackwell: sm_100,
sm_120), a frozen ``nn.Linear`` (also a PEFT ``base_layer``) whose weight was
quantized to "nvfp4" or "fp8" is not expanded to bf16 before it runs. Its
forward quantizes the activations on the fly with TrainStream's kernel (same
format as the weight) and multiplies the packed operands with
``torch.nn.functional.scaled_mm``, so the weight stays 4- or 8-bit end to end.

The backward pass needs only the input gradient (the weight is frozen). It is
computed against the stored weight, expanded to the gradient's dtype:
``grad_x = grad_y @ dequant(W)`` — exact for the weights training uses, the
straight-through estimate for the activation quantization (as in QLoRA-style
and FP8 training recipes).

Support is probed once per device and kind with a tiny matmul; elsewhere the
same weights are expanded before use like nf4 / int8, so results only differ
by the activation quantization.

Outlier channels (nvfp4, ``outliers`` > 0). A few hidden dimensions of an LLM's
activations are far larger than the rest; one of them sets the scale of its
16-value block and rounds the other 15 to zero. Per call, on the GPU and
without a host sync: the per-channel absmax picks the largest ``outliers``
fraction of input channels; they are quantized as 0 (so the remaining values,
and the tensor scale, follow the ordinary activations) and multiplied in
bf16 against the same weight columns instead:

    y = scaled_mm(nvfp4(x without them), W) + x[:, idx] @ W[:, idx].T

Both terms use the stored weight, so the input gradient is unchanged. The
weight itself stays 4-bit: this removes the activation part of NVFP4's error,
not the weight part.
"""

from __future__ import annotations

import types
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from . import _C
from .quant import BLOCKSCALED_KINDS, global_decode_scale, quantized_of

_SUPPORT: Dict[Tuple[str, int], bool] = {}
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _mm(kind: str, aq, as_, ag, bq, bs, bg, out_dtype):
    """aq [M, K'] @ bq[N, K'].T for packed operands with swizzled scales."""
    if kind == "nvfp4":
        fp4 = torch.float4_e2m1fn_x2
        recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
        swz = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
        return F.scaled_mm(aq.view(fp4), bq.view(fp4).t(), [as_.view(torch.float8_e4m3fn), ag], recipe,
                           [bs.view(torch.float8_e4m3fn), bg], recipe, swizzle_a=swz, swizzle_b=swz,
                           output_dtype=out_dtype)
    e8 = torch.float8_e8m0fnu
    return F.scaled_mm(aq.view(torch.float8_e4m3fn), bq.view(torch.float8_e4m3fn).t(), as_.view(e8),
                       F.ScalingType.BlockWise1x32, bs.view(e8), F.ScalingType.BlockWise1x32,
                       swizzle_a=F.SwizzleType.SWIZZLE_32_4_4, swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                       output_dtype=out_dtype)


def _quantize(x2: torch.Tensor, kind: str):
    g = global_decode_scale(x2) if kind == "nvfp4" else None
    q, s = _C.quantize_blockscaled(x2, kind, g)
    return q, s, g


def outlier_count(fraction: float, K: int) -> int:
    """Input channels kept in bf16 for a fraction (0 = none; at least one otherwise)."""
    if not fraction or fraction <= 0:
        return 0
    return min(K - 1, max(1, int(round(fraction * K))))


def _quantize_split(x2: torch.Tensor, k: int):
    """nvfp4 of x2 without its k largest channels: (q, scales, global_decode, idx)."""
    amax = _C.col_absmax(x2)
    vals, idx = torch.topk(amax, k + 1)
    idx = idx[:k]
    g = (vals[k:].clamp_min(1e-12) / (6.0 * 448.0)).contiguous()  # absmax of the channels that stay
    mask = torch.zeros(x2.shape[1], dtype=torch.uint8, device=x2.device).index_fill_(0, idx, 1)
    q, s = _C.quantize_blockscaled(x2, "nvfp4", g, mask)
    return q, s, g, idx


def native_supported(kind: str, device) -> bool:
    """Whether this GPU (and this PyTorch) multiplies `kind` on tensor cores."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (kind, idx)
    if key not in _SUPPORT:
        ok = False
        if kind in BLOCKSCALED_KINDS and hasattr(F, "scaled_mm") and hasattr(torch, "float4_e2m1fn_x2"):
            try:
                gen = torch.Generator(device=dev).manual_seed(0)
                x = torch.randn(32, 64, device=dev, generator=gen).to(torch.bfloat16)
                w = torch.randn(32, 64, device=dev, generator=gen).to(torch.bfloat16)
                xq, xs, xg = _quantize(x, kind)
                wq, ws, wg = _quantize(w, kind)
                one = torch.ones(1, device=dev)
                y = _mm(kind, xq, xs, xg if xg is not None else one, wq, ws, wg if wg is not None else one,
                        torch.float32)
                xd = torch.empty(32, 64, device=dev)
                wd = torch.empty(32, 64, device=dev)
                _C.dequantize_blockscaled(xd, xq, xs, kind, float(xg) if xg is not None else 1.0)
                _C.dequantize_blockscaled(wd, wq, ws, kind, float(wg) if wg is not None else 1.0)
                ref = xd @ wd.t()
                ok = bool(((y - ref).norm() / ref.norm().clamp_min(1e-12)) < 1e-2)
            except Exception:  # noqa: BLE001 - any failure means "not supported here"
                ok = False
        _SUPPORT[key] = ok
    return _SUPPORT[key]


def packed_on_device(qw, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The packed weight and its scales on the GPU: the near copy, the copy the
    pager staged for this layer, or (no pager) a synchronous upload."""
    if qw.dev_q is not None:
        return qw.dev_q, qw.dev_absmax
    staged = getattr(qw, "_staged", None)
    if staged is not None:
        return staged
    return qw.host_q.to(device, non_blocking=True), qw.host_absmax.to(device, non_blocking=True)


class QuantLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qw, bias, weight=None):
        K = x.shape[-1]
        N = qw.shape[0]
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        if out_dtype not in _FLOAT_DTYPES:
            out_dtype = torch.bfloat16
        x2 = x.reshape(-1, K)
        if x2.dtype not in _FLOAT_DTYPES:
            x2 = x2.float()
        wq, ws = packed_on_device(qw, x.device)
        k = outlier_count(qw.outliers, K) if qw.kind == "nvfp4" else 0
        if k:
            xq, xs, xg, idx = _quantize_split(x2.contiguous(), k)
        else:
            xq, xs, xg = _quantize(x2, qw.kind)
        one = None
        if qw.kind == "nvfp4":
            wg = qw.global_tensor(x.device)
        else:
            one = torch.ones(1, device=x.device)
            xg = wg = one
        y = _mm(qw.kind, xq, xs, xg, wq, ws, wg, out_dtype)
        if k:  # the outlier channels, in bf16 against the same (stored) weight columns
            w_out = _C.dequantize_blockscaled_cols(wq, ws, idx, N, K, qw.global_decode, out_dtype)
            y.addmm_(x2.index_select(1, idx).to(out_dtype), w_out.t())
        if bias is not None:
            y = y + bias.to(y.dtype)
        ctx.qw = qw
        ctx.weight = weight  # the parameter: the pager expands into it for backward
        ctx.x_dtype = x.dtype
        ctx.has_bias_grad = bias is not None and bias.requires_grad
        return y.view(*x.shape[:-1], N)

    @staticmethod
    def backward(ctx, gy):
        qw = ctx.qw
        N = qw.shape[0]
        K = qw.numel // N
        g2 = gy.reshape(-1, N)
        p = ctx.weight
        if p is not None and p.untyped_storage().nbytes() > 0:  # expanded ahead by the pager
            w = p.detach().view(N, K)
            dt = w.dtype
        else:
            wq, ws = packed_on_device(qw, gy.device)
            dt = g2.dtype if g2.dtype in _FLOAT_DTYPES else torch.float32
            w = torch.empty(N, K, dtype=dt, device=gy.device)
            _C.dequantize_blockscaled(w, wq, ws, qw.kind, qw.global_decode)
        gx = (g2.to(dt) @ w).view(*gy.shape[:-1], K).to(ctx.x_dtype)
        del w
        gb = g2.sum(0) if ctx.has_bias_grad else None
        return gx, None, gb, None


def _forward(self, x):
    qw = quantized_of(self.weight)
    if qw is None or not getattr(qw, "native", False):
        return nn.Linear.forward(self, x)
    return QuantLinearFn.apply(x, qw, self.bias, self.weight)


def install(blocks, device, outliers: float = 0.0) -> List[nn.Module]:
    """Route the frozen block-scaled Linears of `blocks` through tensor-core
    matmuls where this GPU supports them; nvfp4 ones keep the `outliers`
    fraction of input channels in bf16. Returns the patched modules."""
    patched = []
    for b in blocks:
        for m in b.modules():
            if not isinstance(m, nn.Linear):
                continue
            qw = quantized_of(m.weight)
            if qw is None or not qw.blockscaled or not native_supported(qw.kind, device):
                continue
            qw.native = True
            qw.outliers = float(outliers) if qw.kind == "nvfp4" else 0.0
            m.forward = types.MethodType(_forward, m)
            patched.append(m)
    return patched


def uninstall(modules) -> None:
    for m in modules:
        qw = quantized_of(m.weight)
        if qw is not None:
            qw.native = False
            qw.outliers = 0.0
        if "forward" in m.__dict__:
            del m.forward
