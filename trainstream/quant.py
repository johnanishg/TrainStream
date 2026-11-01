"""Quantized frozen weights (int8, NF4, FP8, NVFP4), with no dependencies.

Only frozen weights (requires_grad=False, e.g. a LoRA base) are quantized:
trained weights need full precision for their updates. Two families:

Expanded before use (any GPU). The pager expands the layer's weights to the
parameter's dtype with a CUDA kernel just before the layer runs and frees the
expanded copy after:

    nf4   QLoRA's 4-bit NormalFloat: 16 quantiles of N(0, 1) scaled to [-1, 1];
          two codes per byte (high nibble first), one fp32 absmax per block of
          `block_size` values. About 0.56 bytes per weight with 64-value blocks.
    int8  linear: round(x / absmax * 127), one fp32 absmax per block. About
          1.06 bytes per weight.

Block-scaled, multiplied directly by tensor cores (trainstream.lowp) on GPUs
that support it (Blackwell: sm_100 / sm_120), and expanded like the others
elsewhere. Activations are quantized on the fly for the forward matmul:

    nvfp4 NVIDIA's FP4: E2M1 codes, two per byte (first value in the low
          nibble), one e4m3 scale per 16 values plus one fp32 global scale.
          0.5625 bytes per weight.
    fp8   MXFP8: e4m3 values with one power-of-two (e8m0) scale per 32 values.
          1.03 bytes per weight.

Block-scaled scales are stored in the swizzled layout cuBLAS reads, so the
packed weight goes to the matmul as it is. `quantize_model` quantizes a model
in memory; the Hugging Face integration's `load_model` quantizes while reading
a checkpoint, so a base larger than RAM never exists in full precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn

EXPAND_KINDS = ("nf4", "int8")
BLOCKSCALED_KINDS = ("fp8", "nvfp4")
KINDS = EXPAND_KINDS + BLOCKSCALED_KINDS

NF4_CODES = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIN_NUMEL = 4096  # smaller frozen tensors (norms, biases) stay unquantized
_GROUP = {"nvfp4": 16, "fp8": 32}


def bytes_per_weight(kind: str, block_size: int = 64) -> float:
    if kind == "nf4":
        return 0.5 + 4.0 / block_size
    if kind == "int8":
        return 1.0 + 4.0 / block_size
    if kind == "nvfp4":
        return 0.5 + 1.0 / 16
    if kind == "fp8":
        return 1.0 + 1.0 / 32
    raise ValueError(f"unknown quantization kind {kind!r}; use one of {KINDS}")


def scale_bytes(kind: str, rows: int, cols: int) -> int:
    """Bytes of the swizzled block-scale buffer of a [rows, cols] block-scaled matrix."""
    sc = cols // _GROUP[kind]
    return -(-rows // 128) * 128 * (-(-sc // 4) * 4)


def packed_bytes(numel: int, kind: str, block_size: int = 64, shape: Optional[Sequence[int]] = None) -> int:
    if kind in BLOCKSCALED_KINDS:
        if shape is not None and len(shape) == 2:
            rows, cols = int(shape[0]), int(shape[1])
            return (numel // 2 if kind == "nvfp4" else numel) + scale_bytes(kind, rows, cols)
        return int(numel * bytes_per_weight(kind))
    nblocks = -(-numel // block_size)
    return (numel + 1) // 2 + 4 * nblocks if kind == "nf4" else numel + 4 * nblocks


def eligible(p: torch.Tensor, kind: Optional[str] = None) -> bool:
    """Frozen floating-point matrices worth quantizing (in `kind`, if given:
    the block-scaled formats need a 2-D [rows, cols] weight with cols % 32 == 0
    and rows % 16 == 0, the shapes tensor cores take)."""
    ok = (not p.requires_grad and p.is_floating_point() and p.dim() >= 2 and p.numel() >= MIN_NUMEL
          and p.numel() % 2 == 0)
    if ok and kind in BLOCKSCALED_KINDS:
        ok = p.dim() == 2 and p.shape[1] % 32 == 0 and p.shape[0] % 16 == 0
    return ok


def _check(kind: str, block_size: int) -> None:
    if kind not in KINDS:
        raise ValueError(f"quantize must be one of {KINDS} here, got {kind!r}")
    if kind in EXPAND_KINDS and (block_size < 2 or block_size % 2):
        raise ValueError(f"quant_block_size must be an even number >= 2, got {block_size}")


def _device(device) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def quantize_tensor(w: torch.Tensor, kind: str = "nf4", block_size: int = 64,
                    device: Optional[torch.device] = None):
    """(q, absmax) for `w` in an expanded format (nf4 / int8), computed on
    `device` (default: CUDA when available). Block-scaled formats: see
    quantize_blockscaled."""
    _check(kind, block_size)
    if kind in BLOCKSCALED_KINDS:
        raise ValueError(f"{kind} is block-scaled: use quantize_blockscaled")
    device = _device(device)
    x = w.detach().to(device=device, dtype=torch.float32).reshape(-1)
    n = x.numel()
    pad = (-n) % block_size
    if pad:
        x = torch.cat([x, x.new_zeros(pad)])
    xb = x.view(-1, block_size)
    absmax = xb.abs().amax(dim=1).clamp_min_(torch.finfo(torch.float32).tiny)
    scaled = xb / absmax[:, None]
    if kind == "int8":
        q = torch.round(scaled * 127.0).clamp_(-127, 127).to(torch.int8).reshape(-1)[:n]
        return q.contiguous(), absmax.contiguous()
    codes = torch.tensor(NF4_CODES, device=device)
    mids = (codes[1:] + codes[:-1]) / 2
    idx = torch.bucketize(scaled.reshape(-1), mids).to(torch.uint8)[: n + (n % 2)]
    if idx.numel() % 2:
        idx = torch.cat([idx, idx.new_zeros(1)])
    q = (idx[0::2] << 4) | idx[1::2]
    return q.contiguous(), absmax.contiguous()


@torch.no_grad()
def dequantize_reference(q: torch.Tensor, absmax: torch.Tensor, kind: str, block_size: int, numel: int,
                         dtype=torch.bfloat16) -> torch.Tensor:
    """Plain-PyTorch expansion of nf4 / int8 (tests, state_dict); the training path uses the CUDA kernel."""
    if kind == "int8":
        vals = q.to(torch.float32)[:numel]
        scale = absmax.repeat_interleave(block_size)[:numel] * (1.0 / 127.0)
        return (vals * scale).to(dtype)
    codes = torch.tensor(NF4_CODES, device=q.device)
    idx = torch.stack([(q >> 4) & 15, q & 15], dim=1).reshape(-1)[:numel].long()
    scale = absmax.repeat_interleave(block_size)[:numel]
    return (codes[idx] * scale).to(dtype)


def global_decode_scale(x: torch.Tensor) -> torch.Tensor:
    """NVFP4's fp32 tensor scale: makes the largest block scale fit e4m3 (448)
    with E2M1's largest value (6). Stays on the GPU (no host round trip)."""
    return (x.detach().abs().amax().float().clamp_min(1e-12) / (6.0 * 448.0)).reshape(1)


@torch.no_grad()
def quantize_blockscaled(w: torch.Tensor, kind: str, device=None):
    """(q, scales, global_decode) of a 2-D tensor in nvfp4 / fp8 with the CUDA
    kernel. global_decode is a 1-element CUDA tensor (1.0 for fp8)."""
    from . import _C
    if kind not in BLOCKSCALED_KINDS:
        raise ValueError(f"block-scaled kind must be one of {BLOCKSCALED_KINDS}, got {kind!r}")
    device = _device(device)
    if device.type != "cuda":
        raise RuntimeError("block-scaled quantization runs on a CUDA device")
    x = w.detach().to(device).reshape(w.shape[0], -1)
    g = global_decode_scale(x) if kind == "nvfp4" else torch.ones(1, device=device)
    q, s = _C.quantize_blockscaled(x, kind, g if kind == "nvfp4" else None)
    return q, s, g


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    p = torch.zeros(rb * 128, cb * 4, dtype=m.dtype, device=m.device)
    p[:rows, :cols] = m
    return p.view(rb, 128, cb, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@torch.no_grad()
def quantize_blockscaled_reference(x: torch.Tensor, kind: str, g: Optional[torch.Tensor] = None):
    """Plain-PyTorch (q, scales, dequantized) that the CUDA kernel reproduces bit for bit."""
    R, K = x.shape
    if kind == "nvfp4":
        xb = x.float().view(R, K // 16, 16)
        bs8 = (xb.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
        dec = bs8.float() * g
        dec = torch.where(dec == 0, torch.full_like(dec, 1e-30), dec)
        scaled = xb / dec[..., None]
        th = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
        code = (scaled.abs().clamp(max=6.0)[..., None] > th).sum(-1)
        code = (code | ((scaled < 0).long() << 3)).view(R, K)
        q = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        vals = torch.tensor(E2M1_VALUES, device=x.device)[code & 7] * torch.where((code & 8) > 0, -1.0, 1.0)
        deq = vals.view(R, K // 16, 16) * (bs8.float() * g)[..., None]
        return q, _to_blocked(bs8.view(torch.uint8)), deq.view(R, K)
    xb = x.float().view(R, K // 32, 32)
    amax = xb.abs().amax(-1)
    _, ex = torch.frexp(amax)
    e = torch.where(amax > 0, ex - 9, torch.full_like(ex, -127)).clamp(-127, 127)
    v = torch.ldexp(xb, -e[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return v.view(R, K).view(torch.uint8), _to_blocked((e + 127).to(torch.uint8)), torch.ldexp(v.float(), e[..., None]).view(R, K)


@dataclass
class QuantizedWeight:
    """A frozen weight stored quantized. Host copies are the authority; the
    planner may add a VRAM copy of the packed bytes (dev_q / dev_absmax).
    For block-scaled kinds `host_absmax` holds the swizzled scale bytes."""

    kind: str
    block_size: int
    shape: tuple
    dtype: torch.dtype
    host_q: torch.Tensor
    host_absmax: torch.Tensor
    pinned: bool = False
    dev_q: Optional[torch.Tensor] = None
    dev_absmax: Optional[torch.Tensor] = None
    global_decode: float = 1.0
    outliers: float = 0.0  # nvfp4 tensor-core path: input channels kept in bf16 (trainstream.lowp)
    _gdev: Optional[torch.Tensor] = None

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def blockscaled(self) -> bool:
        return self.kind in BLOCKSCALED_KINDS

    @property
    def nbytes(self) -> int:
        return (self.host_q.numel() * self.host_q.element_size()
                + self.host_absmax.numel() * self.host_absmax.element_size())

    def to_device(self, device) -> None:
        if self.dev_q is None:
            self.dev_q = self.host_q.to(device, non_blocking=False)
            self.dev_absmax = self.host_absmax.to(device, non_blocking=False)

    def drop_device(self) -> None:
        self.dev_q = None
        self.dev_absmax = None

    def global_tensor(self, device) -> torch.Tensor:
        """The global decode scale as a 1-element tensor on `device` (cached)."""
        if self._gdev is None or self._gdev.device != torch.device(device):
            self._gdev = torch.tensor([self.global_decode], dtype=torch.float32, device=device)
        return self._gdev

    def dequantize(self, device=None) -> torch.Tensor:
        q = self.dev_q if self.dev_q is not None else self.host_q
        a = self.dev_absmax if self.dev_absmax is not None else self.host_absmax
        if self.blockscaled:
            from . import _C
            dev = torch.device(device) if device is not None else (q.device if q.is_cuda else torch.device("cuda"))
            out = torch.empty(self.shape[0], self.numel // self.shape[0], dtype=self.dtype, device=dev)
            _C.dequantize_blockscaled(out, q.to(dev), a.to(dev), self.kind, self.global_decode)
            return out.view(self.shape)
        if device is not None:
            q, a = q.to(device), a.to(device)
        return dequantize_reference(q, a, self.kind, self.block_size, self.numel, self.dtype).view(self.shape)


def _host_copy(t: torch.Tensor, pin: bool) -> torch.Tensor:
    if pin:
        from . import _C
        h = _C.pinned_empty(list(t.shape), t.dtype, "param.quantized", False, False)
        h.copy_(t)
        return h
    return t.to("cpu", copy=True)


def make_quantized(w: torch.Tensor, kind: str, block_size: int, *, pin: bool) -> QuantizedWeight:
    if kind in BLOCKSCALED_KINDS:
        q, s, g = quantize_blockscaled(w, kind)
        return QuantizedWeight(kind=kind, block_size=_GROUP[kind], shape=tuple(w.shape), dtype=w.dtype,
                               host_q=_host_copy(q, pin), host_absmax=_host_copy(s, pin), pinned=pin,
                               global_decode=float(g))
    q, absmax = quantize_tensor(w, kind, block_size)
    return QuantizedWeight(kind=kind, block_size=block_size, shape=tuple(w.shape), dtype=w.dtype,
                           host_q=_host_copy(q, pin), host_absmax=_host_copy(absmax, pin), pinned=pin)


def attach(p: torch.Tensor, qw: QuantizedWeight) -> None:
    """Mark `p` as backed by `qw` and drop its full-precision storage."""
    p._ts_qweight = qw  # read by trainstream.wrap
    if p.untyped_storage().nbytes():
        try:
            p.untyped_storage().resize_(0)
        except RuntimeError:  # not resizable (e.g. memory-mapped from a .safetensors file)
            with torch.no_grad():
                p.data = torch.empty(p.shape, dtype=p.dtype, device=p.device)
            p.untyped_storage().resize_(0)


def quantized_of(p: torch.Tensor) -> Optional[QuantizedWeight]:
    return getattr(p, "_ts_qweight", None)


@torch.no_grad()
def quantize_model(model: nn.Module, quantize: str = "nf4", *, blocks: Optional[Sequence[nn.Module]] = None,
                   block_size: int = 64, pin_budget: Optional[int] = None) -> Dict[str, int]:
    """Quantize the frozen block weights of `model` in place (before wrap()).

    Each eligible weight is quantized one at a time, so peak memory grows by
    one tensor, never by a model. Returns counts and byte totals.
    """
    from .hooks import find_blocks
    from .optimizer import physical_ram_bytes
    from . import _C
    _check(quantize, block_size)
    stack = list(blocks) if blocks is not None else find_blocks(model)
    if pin_budget is None:
        ram = physical_ram_bytes()
        pin_budget = int(0.40 * ram) if ram else 1 << 62
    seen = set()
    count = before = after = 0
    for b in stack:
        for p in b.parameters():
            if id(p) in seen or quantized_of(p) is not None or not eligible(p, quantize):
                continue
            seen.add(id(p))
            nb = packed_bytes(p.numel(), quantize, block_size, p.shape)
            pin = _C.pinned_stats()["live_bytes"] + nb <= pin_budget
            qw = make_quantized(p, quantize, block_size, pin=pin)
            attach(p, qw)
            count += 1
            before += p.numel() * p.element_size()
            after += qw.nbytes
    return {"tensors": count, "bytes_before": before, "bytes_after": after}
