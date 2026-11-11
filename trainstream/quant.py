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


def packed_bytes(numel: int, kind: str, block_size: int = 64, shape: Optional[Sequence[int]] = None,
                 double_quant: bool = False) -> int:
    if kind in BLOCKSCALED_KINDS:
        if shape is not None and len(shape) == 2:
            rows, cols = int(shape[0]), int(shape[1])
            return (numel // 2 if kind == "nvfp4" else numel) + scale_bytes(kind, rows, cols)
        return int(numel * bytes_per_weight(kind))
    nblocks = -(-numel // block_size)
    scales = dq_bytes(nblocks) if double_quant else 4 * nblocks
    return (numel + 1) // 2 + scales if kind == "nf4" else numel + scales


# ------------------------------------------------------------ double-quantized scales
# nf4 / int8 with double_quant=True: each block's fp32 absmax becomes a one-byte log code c,
# with one fp32 scale g per DQ_GROUP blocks: absmax = g * 2^(-c / 32) (2.2 % steps, eight
# octaves below the group's largest). The values are then quantized against that absmax.
# Packed in one uint8 tensor: the codes, padded to 4 bytes, then the scales.
DQ_GROUP = 256


def dq_bytes(nblocks: int) -> int:
    return -(-nblocks // 4) * 4 + 4 * -(-nblocks // DQ_GROUP)


def _dq_table(device) -> torch.Tensor:
    return torch.exp2(-torch.arange(256, dtype=torch.float64) / 32.0).float().to(device)


def _double_quant(absmax: torch.Tensor):
    """(packed uint8 scales, the absmax they decode to) for fp32 block scales."""
    nb = absmax.numel()
    ng = -(-nb // DQ_GROUP)
    a = torch.cat([absmax, absmax.new_full((ng * DQ_GROUP - nb,), torch.finfo(torch.float32).tiny)]).view(ng, -1)
    g = a.amax(dim=1)
    codes = torch.round(-torch.log2(a / g[:, None]) * 32.0).clamp_(0, 255).to(torch.uint8).reshape(-1)[:nb]
    decoded = g.repeat_interleave(DQ_GROUP)[:nb] * _dq_table(absmax.device)[codes.long()]
    coded = -(-nb // 4) * 4
    packed = torch.zeros(coded + 4 * ng, dtype=torch.uint8, device=absmax.device)
    packed[:nb] = codes
    packed[coded:] = g.contiguous().view(torch.uint8)
    return packed, decoded


def undouble_quant(packed: torch.Tensor, nblocks: int) -> torch.Tensor:
    """The fp32 absmax per block that packed double-quantized scales stand for."""
    coded = -(-nblocks // 4) * 4
    ng = -(-nblocks // DQ_GROUP)
    g = packed[coded:coded + 4 * ng].contiguous().view(torch.float32)
    codes = packed[:nblocks].long()
    return g.repeat_interleave(DQ_GROUP)[:nblocks] * _dq_table(packed.device)[codes]


def eligible(p: torch.Tensor, kind: Optional[str] = None) -> bool:
    """Frozen floating-point matrices worth quantizing (in `kind`, if given:
    the block-scaled formats need a 2-D [rows, cols] weight with cols % 32 == 0
    and rows % 16 == 0, the shapes tensor cores take)."""
    ok = (not p.requires_grad and p.is_floating_point() and p.dim() >= 2 and p.numel() >= MIN_NUMEL
          and p.numel() % 2 == 0 and not getattr(p, "_ts_expert", False))
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
                    device: Optional[torch.device] = None, double_quant: bool = False):
    """(q, absmax) for `w` in an expanded format (nf4 / int8), computed on
    `device` (default: CUDA when available). Block-scaled formats: see
    quantize_blockscaled. double_quant: absmax is packed uint8 (see _double_quant)."""
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
    stored = absmax
    if double_quant:
        stored, absmax = _double_quant(absmax)
    scaled = xb / absmax[:, None]
    if kind == "int8":
        q = torch.round(scaled * 127.0).clamp_(-127, 127).to(torch.int8).reshape(-1)[:n]
        return q.contiguous(), stored.contiguous()
    codes = torch.tensor(NF4_CODES, device=device)
    mids = (codes[1:] + codes[:-1]) / 2
    idx = torch.bucketize(scaled.reshape(-1), mids).to(torch.uint8)[: n + (n % 2)]
    if idx.numel() % 2:
        idx = torch.cat([idx, idx.new_zeros(1)])
    q = (idx[0::2] << 4) | idx[1::2]
    return q.contiguous(), stored.contiguous()


@torch.no_grad()
def dequantize_reference(q: torch.Tensor, absmax: torch.Tensor, kind: str, block_size: int, numel: int,
                         dtype=torch.bfloat16) -> torch.Tensor:
    """Plain-PyTorch expansion of nf4 / int8 (tests, state_dict); the training path uses the CUDA kernel.
    absmax may be packed double-quantized scales (uint8)."""
    if absmax.dtype == torch.uint8:
        absmax = undouble_quant(absmax, -(-numel // block_size))
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
    fp4_backward: bool = False  # nvfp4 tensor-core path: the input gradient in nvfp4 too
    # error_lowrank > 0: (A [r, K], B [N, r]) on the host, B @ A ~ the original weight minus this one's
    # expansion, for init_lora_from_error (LoftQ)
    lora_init: Optional[tuple] = None
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


def make_quantized(w: torch.Tensor, kind: str, block_size: int, *, pin: bool, error_rank: int = 0,
                   error_iters: int = 1, double_quant: bool = False) -> QuantizedWeight:
    """`w` stored in `kind`. error_rank > 0: LoftQ, see lowrank_error (the factors ride along as
    qw.lora_init for init_lora_from_error). double_quant: nf4 / int8 block scales stored in
    one byte each (see _double_quant)."""
    factors = None
    if error_rank and w.dim() == 2:
        w, a, b = lowrank_error(w, kind, block_size, error_rank, error_iters)
        factors = (a, b)
    if kind in BLOCKSCALED_KINDS:
        q, s, g = quantize_blockscaled(w, kind)
        return QuantizedWeight(kind=kind, block_size=_GROUP[kind], shape=tuple(w.shape), dtype=w.dtype,
                               host_q=_host_copy(q, pin), host_absmax=_host_copy(s, pin), pinned=pin,
                               global_decode=float(g), lora_init=factors)
    q, absmax = quantize_tensor(w, kind, block_size, double_quant=double_quant)
    return QuantizedWeight(kind=kind, block_size=block_size, shape=tuple(w.shape), dtype=w.dtype,
                           host_q=_host_copy(q, pin), host_absmax=_host_copy(absmax, pin), pinned=pin,
                           lora_init=factors)


@torch.no_grad()
def init_lora_from_error(model: nn.Module) -> int:
    """Start each PEFT LoRA adapter whose base weight carries LoftQ factors (error_rank) at the
    weight's quantization error, so base + adapter starts close to the original weight instead
    of the quantized one: lora_B = B / scaling, lora_A = A (their first r components). Returns
    how many adapters were set."""
    n = 0
    for m in model.modules():
        base = getattr(m, "base_layer", None)
        if base is None or not hasattr(m, "lora_A") or not isinstance(getattr(base, "weight", None), torch.Tensor):
            continue
        qw = quantized_of(base.weight)
        if qw is None or qw.lora_init is None:
            continue
        a, b = qw.lora_init
        for name in m.lora_A.keys():
            la, lb = m.lora_A[name].weight, m.lora_B[name].weight
            r = la.shape[0]
            if r > a.shape[0] or la.shape[1] != a.shape[1] or lb.shape[0] != b.shape[0]:
                continue  # the adapter is larger than the factors (or shaped differently): left as it is
            s = float(m.scaling[name])
            la.copy_(a[:r].to(la.device, la.dtype))
            lb.copy_((b[:, :r].float() / s).to(lb.device, lb.dtype))
            n += 1
    return n


# -------------------------------------------------------------- selective precision
KEEP_KIND = {"nf4": "int8", "nvfp4": "fp8"}  # one step more precise, on the same code path
KEEP_AUTO = 0.05


def keep_fraction(value) -> float:
    """keep_precise as the extra bytes allowed, as a fraction of the quantized weights' bytes:
    0 / None / False none; "auto" or True KEEP_AUTO; a float in (0, 0.5]."""
    if value in (None, False, 0, 0.0):
        return 0.0
    if value in ("auto", True):
        return KEEP_AUTO
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 0.5:
        return float(value)
    raise ValueError(f"keep_precise must be 'auto', True, False or a fraction up to 0.5, got {value!r}")


@torch.no_grad()
def roundtrip(x: torch.Tensor, kind: str, block_size: int = 64) -> torch.Tensor:
    """x stored in `kind` and expanded again, in fp32 on x's device (a CUDA device for block-scaled kinds)."""
    x = x.float()
    if kind in BLOCKSCALED_KINDS:
        from . import _C
        x2 = x.reshape(x.shape[0], -1)
        q, s, g = quantize_blockscaled(x2, kind, x.device)
        out = torch.empty_like(x2)
        _C.dequantize_blockscaled(out, q, s, kind, float(g))
        return out.view(x.shape)
    q, a = quantize_tensor(x, kind, block_size, x.device)
    return dequantize_reference(q, a, kind, block_size, x.numel(), torch.float32).view(x.shape)


@torch.no_grad()
def quant_error(w: torch.Tensor, kind: str, block_size: int = 64) -> float:
    """Relative error ||W - deq(q(W))|| / ||W|| of storing `w` in `kind` (on the GPU when there is one)."""
    x = w.detach().to(_device(None), torch.float32)
    return float((roundtrip(x, kind, block_size) - x).norm() / x.norm().clamp_min(1e-30))


@torch.no_grad()
def lowrank_error(w: torch.Tensor, kind: str, block_size: int, rank: int, iters: int = 1):
    """LoftQ (Li et al., 2023): (weight to store in `kind`, A [rank, K], B [N, rank]) such that
    deq(stored) + B @ A is close to w. Alternates `iters` times between quantizing w - B A and
    fitting B A to what that quantization lost (randomized SVD, on the GPU when there is one);
    with iters=1 the stored weight is w itself and B A its quantization error."""
    dev = _device(None)
    W = w.detach().to(dev, torch.float32).reshape(w.shape[0], -1)
    rank = max(1, min(rank, min(W.shape)))
    BA = None
    R = W
    for _ in range(max(1, iters)):
        R = W if BA is None else W - BA
        E = W - roundtrip(R, kind, block_size)
        U, S, V = torch.svd_lowrank(E, q=min(rank + 8, min(E.shape)), niter=2)
        sq = S[:rank].sqrt()
        B = U[:, :rank] * sq
        A = (V[:, :rank] * sq).t()
        BA = B @ A
    return R.to(w.dtype).view(w.shape), A.to(torch.bfloat16).cpu(), B.to(torch.bfloat16).cpu()


def choose_precise(errors, kind: str, fraction: float, block_size: int = 64) -> set:
    """Which weights to keep at KEEP_KIND[kind]: errors is [(key, relative error, numel, shape)];
    the largest errors first, as long as the extra bytes stay within `fraction` of what the
    weights take in `kind`."""
    keep_kind = KEEP_KIND.get(kind)
    if keep_kind is None or fraction <= 0 or not errors:
        return set()
    budget = fraction * sum(packed_bytes(n, kind, block_size, shape) for _, _, n, shape in errors)
    out, used = set(), 0
    for key, _, n, shape in sorted(errors, key=lambda e: -e[1]):
        extra = packed_bytes(n, keep_kind, block_size, shape) - packed_bytes(n, kind, block_size, shape)
        if used + extra <= budget:
            out.add(key)
            used += extra
    return out


def from_modelopt_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2, dtype,
                        pin: bool = False) -> QuantizedWeight:
    """A QuantizedWeight from an NVFP4 checkpoint exported by NVIDIA ModelOpt (the
    `quant_method: modelopt, quant_algo: NVFP4` Hugging Face format): weight uint8 [N, K/2],
    two E2M1 codes per byte, the first in the low nibble (TrainStream's own order); weight_scale
    e4m3 [N, K/16], one per 16 values; weight_scale_2 the fp32 tensor scale. Only the block
    scales change layout (cuBLAS's swizzled tiles)."""
    n, half = weight.shape
    k = 2 * half
    if k % 32 or n % 16:
        raise ValueError(f"NVFP4 weight [{n}, {k}]: tensor cores need rows % 16 == 0 and columns % 32 == 0")
    sc = weight_scale if weight_scale.dtype == torch.uint8 else weight_scale.to(torch.float8_e4m3fn).view(torch.uint8)
    swz = _to_blocked(sc.reshape(n, k // 16))
    g = float(weight_scale_2.reshape(-1)[0]) if isinstance(weight_scale_2, torch.Tensor) else float(weight_scale_2)
    return QuantizedWeight(kind="nvfp4", block_size=_GROUP["nvfp4"], shape=(n, k), dtype=dtype,
                           host_q=_host_copy(weight.to(torch.uint8).contiguous(), pin),
                           host_absmax=_host_copy(swz.contiguous(), pin), pinned=pin, global_decode=g)


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
                   block_size: int = 64, pin_budget: Optional[int] = None, keep_precise=0, error_rank: int = 0,
                   error_iters: int = 1, double_quant: bool = False) -> Dict[str, int]:
    """Quantize the frozen block weights of `model` in place (before wrap()).

    Each eligible weight is quantized one at a time, so peak memory grows by
    one tensor, never by a model. keep_precise (nf4 / nvfp4): keep the weights
    that lose the most in `quantize` one step more precise (int8 / fp8), within
    that fraction of extra bytes ("auto": 0.05). error_rank > 0 also keeps each weight's
    quantization error as a rank-`error_rank` approximation (LoftQ, `error_iters` rounds), from
    which init_lora_from_error / wrap() start LoRA adapters. double_quant (nf4 / int8): block
    scales in one byte each instead of four (about 0.37 bit per weight less at block size 64).
    Returns counts and byte totals.
    """
    from .hooks import find_blocks
    from .optimizer import physical_ram_bytes
    from . import _C
    _check(quantize, block_size)
    stack = list(blocks) if blocks is not None else find_blocks(model)
    if pin_budget is None:
        ram = physical_ram_bytes()
        pin_budget = int(0.40 * ram) if ram else 1 << 62
    todo, seen = [], set()
    for b in stack:
        for p in b.parameters():
            if id(p) in seen or quantized_of(p) is not None or not eligible(p, quantize):
                continue
            seen.add(id(p))
            todo.append(p)
    fraction = keep_fraction(keep_precise)
    keep = choose_precise([(id(p), quant_error(p, quantize, block_size), p.numel(), tuple(p.shape)) for p in todo],
                          quantize, fraction, block_size) if fraction and quantize in KEEP_KIND else set()
    count = before = after = 0
    for p in todo:
        kind = KEEP_KIND[quantize] if id(p) in keep else quantize
        nb = packed_bytes(p.numel(), kind, block_size, p.shape)
        pin = _C.pinned_stats()["live_bytes"] + nb <= pin_budget
        qw = make_quantized(p, kind, block_size, pin=pin, error_rank=error_rank, error_iters=error_iters,
                            double_quant=double_quant)
        attach(p, qw)
        count += 1
        before += p.numel() * p.element_size()
        after += qw.nbytes
    return {"tensors": count, "kept_precise": len(keep), "bytes_before": before, "bytes_after": after}
