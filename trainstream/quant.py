"""Blockwise quantization of frozen weights (int8, NF4), with no dependencies.

Only frozen weights (requires_grad=False, e.g. a LoRA base) are quantized:
trained weights need full precision for their updates. A quantized weight
keeps

    q       packed codes: NF4 two per byte (high nibble first), int8 one per byte
    absmax  one fp32 scale per block of `block_size` values

in host RAM (and, when the memory planner has room, a copy in VRAM). Before
its layer runs, the pager expands it to the parameter's dtype with a CUDA
kernel (trainstream/csrc/quant_kernel.cu) and frees the expanded copy after.

    nf4   QLoRA's 4-bit NormalFloat: 16 quantiles of N(0, 1) scaled to [-1, 1];
          each value is replaced by the nearest code times its block's absmax.
          About 0.56 bytes per weight with 64-value blocks.
    int8  linear: round(x / absmax * 127). About 1.06 bytes per weight.

`quantize_model` does this in place for a model in memory; the Hugging Face
integration's `load_model` quantizes while reading a checkpoint, so a base
larger than RAM never exists in full precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

KINDS = ("nf4", "int8")

NF4_CODES = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)
MIN_NUMEL = 4096  # smaller frozen tensors (norms, biases) stay unquantized


def bytes_per_weight(kind: str, block_size: int = 64) -> float:
    if kind == "nf4":
        return 0.5 + 4.0 / block_size
    if kind == "int8":
        return 1.0 + 4.0 / block_size
    raise ValueError(f"unknown quantization kind {kind!r}; use one of {KINDS}")


def packed_bytes(numel: int, kind: str, block_size: int = 64) -> int:
    nblocks = -(-numel // block_size)
    return (numel + 1) // 2 + 4 * nblocks if kind == "nf4" else numel + 4 * nblocks


def eligible(p: torch.Tensor) -> bool:
    """Frozen floating-point matrices worth quantizing."""
    return (not p.requires_grad and p.is_floating_point() and p.dim() >= 2 and p.numel() >= MIN_NUMEL
            and p.numel() % 2 == 0)


def _check(kind: str, block_size: int) -> None:
    if kind not in KINDS:
        raise ValueError(f"quantize must be one of {KINDS} here, got {kind!r}")
    if block_size < 2 or block_size % 2:
        raise ValueError(f"quant_block_size must be an even number >= 2, got {block_size}")


@torch.no_grad()
def quantize_tensor(w: torch.Tensor, kind: str = "nf4", block_size: int = 64,
                    device: Optional[torch.device] = None):
    """(q, absmax) for `w`, computed on `device` (default: CUDA when available)."""
    _check(kind, block_size)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    """Plain-PyTorch expansion (tests, state_dict); the training path uses the CUDA kernel."""
    if kind == "int8":
        vals = q.to(torch.float32)[:numel]
        scale = absmax.repeat_interleave(block_size)[:numel] * (1.0 / 127.0)
        return (vals * scale).to(dtype)
    codes = torch.tensor(NF4_CODES, device=q.device)
    idx = torch.stack([(q >> 4) & 15, q & 15], dim=1).reshape(-1)[:numel].long()
    scale = absmax.repeat_interleave(block_size)[:numel]
    return (codes[idx] * scale).to(dtype)


@dataclass
class QuantizedWeight:
    """A frozen weight stored quantized. Host copies are the authority; the
    planner may add a VRAM copy of the packed bytes (dev_q / dev_absmax)."""

    kind: str
    block_size: int
    shape: tuple
    dtype: torch.dtype
    host_q: torch.Tensor
    host_absmax: torch.Tensor
    pinned: bool = False
    dev_q: Optional[torch.Tensor] = None
    dev_absmax: Optional[torch.Tensor] = None

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        return self.host_q.numel() * self.host_q.element_size() + self.host_absmax.numel() * 4

    def to_device(self, device) -> None:
        if self.dev_q is None:
            self.dev_q = self.host_q.to(device, non_blocking=False)
            self.dev_absmax = self.host_absmax.to(device, non_blocking=False)

    def drop_device(self) -> None:
        self.dev_q = None
        self.dev_absmax = None

    def dequantize(self, device=None) -> torch.Tensor:
        q = self.dev_q if self.dev_q is not None else self.host_q
        a = self.dev_absmax if self.dev_absmax is not None else self.host_absmax
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
            if id(p) in seen or quantized_of(p) is not None or not eligible(p):
                continue
            seen.add(id(p))
            nb = packed_bytes(p.numel(), quantize, block_size)
            pin = _C.pinned_stats()["live_bytes"] + nb <= pin_budget
            qw = make_quantized(p, quantize, block_size, pin=pin)
            attach(p, qw)
            count += 1
            before += p.numel() * p.element_size()
            after += qw.nbytes
    return {"tensors": count, "bytes_before": before, "bytes_after": after}
