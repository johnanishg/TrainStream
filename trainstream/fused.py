"""Fused RMSNorm, rotary embedding and gated MLP activation for Hugging Face decoder layers.

A decoder layer runs these as chains of small PyTorch ops, each a kernel that
reads and writes the whole activation. On a 14B LoRA step they take as long
as a third of the matrix multiplies, and their kernel launches keep the GPU
waiting on the host. trainstream/csrc/fused_kernel.cu does each in one pass,
computing in fp32 and rounding once (so results differ from PyTorch's in the
last bits, which is why wrap(kernels="auto") turns them on only for a
quantized base, already not bit-exact):

    RMSNorm   one kernel forward, one backward; saves the per-row rstd
    RoPE      q and k rotated in place of apply_rotary_pos_emb
    act(g)*u  one kernel; saves g and u (not act(g)), recomputed in backward

install() patches what it can verify: each RMSNorm class, MLP class and
apply_rotary_pos_emb is run once on random inputs against the fused version,
and patched only when they agree. Anything else is left as it is.

The MLP front without its two largest activations (GatedFrontFn). g = gate(x)
and u = up(x) are each [tokens, intermediate], about 2.7 times the hidden
state, and the activation's backward needs both: of everything a decoder
layer keeps for backward they are roughly a third. Under memory pressure
those bytes go to host RAM and come back over PCIe. GatedFrontFn keeps x
instead (which the LoRA adapters keep anyway) and runs the two projections
again in backward, through the model's own modules, so the result is the same
bits and every gradient (base, adapters) flows as before. On a tensor-core
base that is two FP4 matmuls, a few milliseconds against tens for the bytes'
round trip. FrontPolicy decides where: always, or ("auto") for tensor-core
MLPs in blocks whose activations are being paged to host RAM. Measured on
Qwen2.5-14B (NVFP4 base, LoRA, 2,048 tokens, RTX 5090 Laptop GPU) with every
block's activations paged: 1.68 s per step against 2.08 s, and 35 % fewer
bytes paged. Where activations stay in VRAM or whole blocks are recomputed
instead, it only costs the two matmuls (0.4 to 2 % of a step), so "auto"
leaves those blocks alone.

With fused_gemm the forward comes from TrainStream's own kernel
(csrc/gemm_fp4_kernel.cu: the two FP4 products, the LoRA terms and the
activation in one pass, g and u never in memory) where it applies: consumer
Blackwell, both projections nvfp4 without bias or outlier channels, plain
LoRA or none, bf16 output. It measured slower than cuBLAS's matmuls on an RTX
5090 Laptop GPU, so it is off unless asked for.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Dict, List, Optional

import torch
from torch import nn

from . import _C, lowp
from . import lora as _lora
from .hooks import in_block_recompute
from .quant import quantized_of

_FLOATS = (torch.float32, torch.float16, torch.bfloat16)
ACT_SILU, ACT_GELU_TANH = 0, 1
# fronts computed without keeping g / u; how many of them by the fused FP4 kernel; the bytes of
# g and u that were therefore not kept for backward (process-wide)
FRONT_STATS = dict(recomputed=0, kernel=0, bytes_not_kept=0)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps, gemma):
        y, rstd = _C.rms_norm_forward(x, w, eps, gemma)
        ctx.save_for_backward(x, w, rstd)
        ctx.gemma = gemma
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        dx, dw = _C.rms_norm_backward(dy.contiguous(), x, w, rstd, ctx.gemma, ctx.needs_input_grad[1])
        return dx.view(x.shape), (dw.to(w.dtype) if dw is not None else None), None, None


class RoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _C.rope(x, cos, sin, False)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        return _C.rope(dy, cos, sin, True), None, None


class GatedActFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u, act):
        ctx.save_for_backward(g, u)
        ctx.act = act
        return _C.gated_act_forward(g, u, act)

    @staticmethod
    def backward(ctx, dh):
        g, u = ctx.saved_tensors
        dg, du = _C.gated_act_backward(dh, g, u, ctx.act)
        return dg, du, None


# ------------------------------------------------------- the MLP front without g and u
def _front_params(m) -> Optional[list]:
    """The trainable parameters of m.gate_proj / m.up_proj when both give the same output for
    the same input every time (a Linear, or a PEFT LoRA layer with no dropout in effect), so
    that running them again in backward reproduces forward; None otherwise."""
    out = []
    for proj in (m.gate_proj, m.up_proj):
        if hasattr(proj, "lora_A") and hasattr(proj, "base_layer"):
            if not isinstance(proj.base_layer, nn.Linear):
                return None
            if not (getattr(proj, "disable_adapters", False) or getattr(proj, "merged", False)):
                for name in proj.active_adapters:
                    drop = proj.lora_dropout[name] if name in proj.lora_dropout else None
                    if isinstance(drop, nn.Dropout) and drop.p > 0 and drop.training:
                        return None
                    if drop is not None and not isinstance(drop, (nn.Dropout, nn.Identity)):
                        return None
        elif not isinstance(proj, nn.Linear):
            return None
        out.extend(p for p in proj.parameters() if p.requires_grad)
    return out


def _kernel_operands(proj):
    """(quantized weight, A, B, scaling) when `proj` is an nvfp4 tensor-core Linear without
    bias or outlier channels, alone or under exactly one plain PEFT LoRA adapter (A, B None
    without one): what the fused FP4 kernel computes. None for anything else."""
    base, A, B, s = proj, None, None, 0.0
    if hasattr(proj, "lora_A") and hasattr(proj, "base_layer"):
        if getattr(proj, "merged", False):
            return None
        if not getattr(proj, "disable_adapters", False):
            names = list(proj.active_adapters)
            if len(names) != 1 or not _lora._plain(proj, names[0]):
                return None
            A, B = proj.lora_A[names[0]].weight, proj.lora_B[names[0]].weight
            s = float(proj.scaling[names[0]])
        base = proj.base_layer
    if not isinstance(base, nn.Linear) or base.bias is not None:
        return None
    qw = quantized_of(base.weight)
    if qw is None or qw.kind != "nvfp4" or not getattr(qw, "native", False) or qw.outliers:
        return None
    return qw, A, B, s


def _kernel_front(mlp, x, act):
    """h = act(gate(x)) * up(x) from the single fused FP4 kernel, or None when this MLP, this
    input or this GPU is not one it serves (the caller then computes it the usual way)."""
    if not x.is_cuda or x.dtype not in _FLOATS:
        return None
    dev = x.device
    if not _C.gemm_fp4_supported(dev.index if dev.index is not None else -1):
        return None
    out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if out_dtype != torch.bfloat16:
        return None
    og, ou = _kernel_operands(mlp.gate_proj), _kernel_operands(mlp.up_proj)
    if og is None or ou is None:
        return None
    (qg, Ag, Bg, sg), (qu, Au, Bu, su) = og, ou
    K = x.shape[-1]
    N = qg.shape[0]
    if tuple(qg.shape) != (N, K) or tuple(qu.shape) != (N, K) or K % 64 or N % 16:
        return None
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if M == 0:
        return None
    hit = lowp._CACHE.get(x, "nvfp4", 0)
    if hit is None:
        hit = lowp._CACHE.put(x, "nvfp4", 0, lowp._quantize(x2, "nvfp4") + (None,))
    xq, xs, xg, _ = hit
    bf = torch.bfloat16
    ranks = [A.shape[0] for A in (Ag, Au) if A is not None]
    R = -(-max(ranks) // 16) * 16 if ranks else 0  # the kernel adds LoRA terms 16 ranks at a time
    empty = torch.empty(0, dtype=bf, device=dev)
    xa, sb, xb = [], [], None
    for A, B, s in ((Ag, Bg, sg), (Au, Bu, su)):
        if A is None:
            xa.append(empty)
            sb.append(empty)
            continue
        if xb is None:
            xb = x2.to(bf)
        left = xb @ A.to(bf).t()               # x A^T: [M, r]
        right = (B.float() * s).to(bf)         # s B: [N, r]
        pad = R - A.shape[0]
        if pad:
            left = nn.functional.pad(left, (0, pad))
            right = nn.functional.pad(right, (0, pad))
        xa.append(left.contiguous())
        sb.append(right.contiguous())
    wqg, wsg = lowp.packed_on_device(qg, dev)
    wqu, wsu = lowp.packed_on_device(qu, dev)
    (h,) = _C.gemm_fp4_front(xq, xs, xg, [wqg, wqu], [wsg, wsu], [float(qg.global_decode), float(qu.global_decode)],
                             M, N, K, xa, sb, int(act), True, False)
    return h.view(*x.shape[:-1], N)


class GatedFrontFn(torch.autograd.Function):
    """h = act(gate_proj(x)) * up_proj(x), keeping x for backward instead of the two projection
    outputs. Backward runs the projections again (the model's own modules, under the autocast
    setting of forward), takes the activation's gradient with the fused kernel, and sends it
    through the graph of that second run: the input gradient and the gradients of `params`
    (the projections' trainable parameters) are exactly what the usual path computes."""

    @staticmethod
    def forward(ctx, x, mlp, act, cast, kernel, *params):
        h = _kernel_front(mlp, x, act) if kernel else None
        FRONT_STATS["kernel"] += h is not None
        if h is None:
            g = mlp.gate_proj(x)
            u = mlp.up_proj(x)
            if g.dtype == u.dtype and g.dtype in _FLOATS and g.shape == u.shape:
                h = _C.gated_act_forward(g, u, act)
            else:
                h = mlp.act_fn(g) * u
        ctx.save_for_backward(x)
        ctx.mlp, ctx.act, ctx.cast, ctx.params = mlp, act, cast, params
        FRONT_STATS["recomputed"] += 1
        FRONT_STATS["bytes_not_kept"] += 2 * h.numel() * h.element_size()
        return h

    @staticmethod
    def backward(ctx, dh):
        (x,) = ctx.saved_tensors
        mlp, act = ctx.mlp, ctx.act
        on, dtype = ctx.cast
        needs = ctx.needs_input_grad
        with torch.enable_grad(), torch.autocast("cuda", dtype=dtype, enabled=on):
            xg = x.detach()
            if needs[0]:
                xg.requires_grad_()
            g = mlp.gate_proj(xg)
            u = mlp.up_proj(xg)
        gd, ud = g.detach(), u.detach()
        if gd.dtype == ud.dtype and gd.dtype in _FLOATS and gd.shape == ud.shape:
            dg, du = _C.gated_act_backward(dh.to(gd.dtype), gd, ud, act)
        else:  # an activation the kernel does not take: PyTorch's own derivative
            with torch.enable_grad():
                gl, ul = gd.requires_grad_(), ud.requires_grad_()
                a = mlp.act_fn(gl) * ul
                dg, du = torch.autograd.grad(a, (gl, ul), dh.to(a.dtype))
        inputs = ([xg] if needs[0] else []) + [p for p, need in zip(ctx.params, needs[5:]) if need]
        outs = [(t, d.view(t.shape).to(t.dtype)) for t, d in ((g, dg), (u, du)) if t.requires_grad]
        if inputs and outs:
            grads = list(torch.autograd.grad([t for t, _ in outs], inputs, [d for _, d in outs], allow_unused=True))
        else:
            grads = [None] * len(inputs)
        dx = grads.pop(0) if needs[0] else None
        dparams = [grads.pop(0) if need else None for need in needs[5:]]
        return (dx, None, None, None, None, *dparams)


class FrontPolicy:
    """Which gated MLPs run through GatedFrontFn (see the module docstring).

    mode: True (always, any MLP whose projections are deterministic), "auto" (MLPs whose two
    projections run on tensor cores, whenever `pressure()` is true: the session answers whether
    the block running now has its activations paged to host RAM; that does not change within a
    step, so a block's forward and anything that runs it again decide alike), False (only what
    `kernel` asks for). kernel: compute the forward with the fused FP4 kernel where it applies,
    which also means those MLPs keep no g / u. unkept(nbytes), when set, is told how many bytes
    a front left out."""

    def __init__(self, mode, kernel: bool, pressure=None, unkept=None):
        self.mode = mode
        self.kernel = bool(kernel)
        self.pressure = pressure
        self.unkept = unkept
        self.engaged = mode is True  # "auto": whether it has switched on at least once
        self.on_engage = None
        self._cheap: Dict[int, bool] = {}
        self._served: Dict[int, bool] = {}

    @staticmethod
    def _tensor_core(proj) -> bool:
        base = getattr(proj, "base_layer", proj)
        qw = quantized_of(base.weight) if isinstance(base, nn.Linear) else None
        return qw is not None and bool(getattr(qw, "native", False))

    def kernel_serves(self, mlp) -> bool:
        k = id(mlp)
        if k not in self._served:
            w = getattr(getattr(mlp.gate_proj, "base_layer", mlp.gate_proj), "weight", None)
            dev = w.device if isinstance(w, torch.Tensor) and w.is_cuda else None
            self._served[k] = (dev is not None and _C.gemm_fp4_supported(dev.index if dev.index is not None else -1)
                               and _kernel_operands(mlp.gate_proj) is not None
                               and _kernel_operands(mlp.up_proj) is not None)
        return self._served[k]

    def wanted(self, mlp) -> bool:
        if self.mode is True:
            return True
        if self.kernel and self.kernel_serves(mlp):
            return True
        if self.mode != "auto" or self.pressure is None:
            return False
        k = id(mlp)
        if k not in self._cheap:
            self._cheap[k] = self._tensor_core(mlp.gate_proj) and self._tensor_core(mlp.up_proj)
        if not self._cheap[k] or not self.pressure():
            return False
        if not self.engaged:
            self.engaged = True
            if self.on_engage is not None:
                self.on_engage()
        return True


# ------------------------------------------------------------------ patched forwards
def _norm_forward(self, x):
    eps, gemma = self._ts_fused_norm
    w = self.weight
    if (not x.is_cuda or x.dtype not in _FLOATS or w.dtype not in _FLOATS or w.device != x.device
            or (not gemma and w.dtype != x.dtype) or x.numel() == 0):
        return type(self).forward(self, x)  # e.g. an fp32 weight on bf16 input: PyTorch promotes the result
    return RMSNormFn.apply(x, w, eps, gemma)


def _mlp_forward(self, x):
    front = self.__dict__.get("_ts_front")
    # (a block TrainStream recomputes in backward keeps nothing anyway, and must take the same
    # path in its forward and in its recomputation: the usual one)
    if (front is not None and x.is_cuda and torch.is_grad_enabled() and not in_block_recompute()
            and front.wanted(self)):
        params = _front_params(self)
        if params is not None and (x.requires_grad or params):
            cast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
            h = GatedFrontFn.apply(x, self, self._ts_fused_act, cast, front.kernel, *params)
            if front.unkept is not None:
                front.unkept(2 * h.numel() * h.element_size())
            return self.down_proj(h)
    g = self.gate_proj(x)
    u = self.up_proj(x)
    if not g.is_cuda or g.dtype != u.dtype or g.dtype not in _FLOATS or g.shape != u.shape:
        return self.down_proj(self.act_fn(g) * u)
    return self.down_proj(GatedActFn.apply(g, u, self._ts_fused_act))


def _rope_ok(q, cos, unsqueeze_dim, position_ids) -> bool:
    return (unsqueeze_dim == 1 and position_ids is None and q.is_cuda and q.dim() == 4 and cos.dim() == 3
            and q.dtype in _FLOATS and cos.dtype == q.dtype and q.stride(-1) == 1 and q.shape[-1] % 2 == 0
            and cos.shape[-1] == q.shape[-1] and cos.shape[1] == q.shape[2] and cos.shape[0] in (1, q.shape[0]))


def _make_rope(orig):
    sig = inspect.signature(orig)

    def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        try:
            bound = sig.bind(q, k, cos, sin, *args, **kwargs)
        except TypeError:
            return orig(q, k, cos, sin, *args, **kwargs)
        a = bound.arguments
        unsq, pos = a.get("unsqueeze_dim", 1), a.get("position_ids")
        if not (_rope_ok(q, cos, unsq, pos) and _rope_ok(k, cos, unsq, pos) and sin.shape == cos.shape
                and sin.dtype == cos.dtype):
            return orig(q, k, cos, sin, *args, **kwargs)
        return RoPEFn.apply(q, cos, sin), RoPEFn.apply(k, cos, sin)

    apply_rotary_pos_emb._ts_original = orig
    return apply_rotary_pos_emb


# ---------------------------------------------------------------------- probing
def _bare(cls):
    """An instance of a module class without running its __init__ (no config needed)."""
    obj = object.__new__(cls)
    nn.Module.__init__(obj)
    return obj


def _close(a, b) -> bool:
    return a.shape == b.shape and torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-5)


def _probe_norm(cls, m, device) -> Optional[bool]:
    """Which order the class computes (False: Llama's w * n, True: Gemma's n * (1 + w)), or None when
    its forward matches neither. Run in fp32 on random data, so the two are told apart."""
    eps = getattr(m, "variance_epsilon", getattr(m, "eps", None))
    w = getattr(m, "weight", None)
    if not isinstance(eps, float) or not isinstance(w, nn.Parameter) or w.dim() != 1:
        return None
    c = w.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(4, c, generator=g).to(device)
    obj = _bare(cls)
    for k, v in vars(m).items():
        if not k.startswith("_") and not isinstance(v, (torch.Tensor, nn.Module)):
            setattr(obj, k, v)
    obj.weight = nn.Parameter((1 + 0.1 * torch.randn(c, generator=g)).to(device))
    try:
        with torch.no_grad():
            ref = obj(x)
            for gemma in (False, True):
                y, _ = _C.rms_norm_forward(x, obj.weight.detach(), float(eps), gemma)
                if _close(ref, y):
                    return gemma
    except Exception:  # noqa: BLE001 - a forward we can't run on a bare instance: leave it alone
        return None
    return None


def _act_code(act_fn) -> Optional[int]:
    x = torch.linspace(-6, 6, 97)
    try:
        with torch.no_grad():
            y = act_fn(x)
    except Exception:  # noqa: BLE001
        return None
    if torch.allclose(y, x * torch.sigmoid(x), atol=1e-5):
        return ACT_SILU
    if torch.allclose(y, nn.functional.gelu(x, approximate="tanh"), atol=1e-5):
        return ACT_GELU_TANH
    return None


def _probe_mlp(cls, m, device) -> Optional[int]:
    """The activation code when the class computes down(act(gate(x)) * up(x)) with these
    projections, checked on a bare instance with small random ones, else None."""
    if not all(isinstance(getattr(m, n, None), nn.Module) for n in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        return None
    act = _act_code(m.act_fn)
    if act is None:
        return None
    torch.manual_seed(0)
    obj = _bare(cls)
    for k, v in vars(m).items():
        if not k.startswith("_") and not isinstance(v, (torch.Tensor, nn.Module)):
            setattr(obj, k, v)
    obj.gate_proj, obj.up_proj, obj.down_proj = nn.Linear(16, 32, bias=False), nn.Linear(16, 32, bias=False), \
        nn.Linear(32, 16, bias=False)
    obj.act_fn = m.act_fn
    obj.to(device)
    obj._ts_fused_act = act
    x = torch.randn(3, 16, device=device)
    try:
        with torch.no_grad():
            ref = cls.forward(obj, x)
            mine = _mlp_forward(obj, x)
    except Exception:  # noqa: BLE001
        return None
    return act if _close(ref, mine) else None


def _probe_rope(fn, device) -> bool:
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(2, 3, 5, 8, generator=g).to(device)
    k = torch.randn(2, 1, 5, 8, generator=g).to(device)
    ang = torch.randn(2, 5, 4, generator=g).to(device)
    emb = torch.cat((ang, ang), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    try:
        with torch.no_grad():
            rq, rk = fn(q, k, cos, sin)
    except Exception:  # noqa: BLE001
        return False
    return _close(rq, _C.rope(q, cos, sin, False)) and _close(rk, _C.rope(k, cos, sin, False))


# ------------------------------------------------------------------ install / remove
_ROPE: Dict[str, list] = {}  # python module name -> [original apply_rotary_pos_emb, models using it]


class Fused:
    """What install() patched: undo with remove()."""

    def __init__(self):
        self.norms: List[nn.Module] = []
        self.mlps: List[nn.Module] = []
        self.rope: List[str] = []
        self.acts: Dict[str, int] = {}

    def describe(self) -> str:
        parts = []
        if self.norms:
            parts.append(f"{len(self.norms)} RMSNorms")
        if self.mlps:
            parts.append(f"{len(self.mlps)} gated MLPs ({', '.join(sorted(self.acts))})")
        if self.rope:
            parts.append("rotary embedding (" + ", ".join(n.rsplit(".", 1)[-1] for n in self.rope) + ")")
        return ", ".join(parts) or "nothing this model uses"

    def remove(self) -> None:
        for m in self.norms + self.mlps:
            m.__dict__.pop("forward", None)
            m.__dict__.pop("_ts_fused_norm", None)
            m.__dict__.pop("_ts_fused_act", None)
            m.__dict__.pop("_ts_front", None)
        for name in self.rope:
            entry = _ROPE.get(name)
            if entry is None:
                continue
            entry[1] -= 1
            if entry[1] <= 0:
                mod = sys.modules.get(name)
                if mod is not None:
                    mod.apply_rotary_pos_emb = entry[0]
                del _ROPE[name]
        self.norms, self.mlps, self.rope = [], [], []


def install(model: nn.Module, device) -> Fused:
    """Route `model`'s RMSNorms, gated MLPs and rotary embedding through the fused kernels,
    as far as each verifies against its original. Returns what was patched."""
    done = Fused()
    device = torch.device(device)
    norm_mode: Dict[type, Optional[bool]] = {}
    mlp_act: Dict[type, Optional[int]] = {}
    pymods = set()
    for m in model.modules():
        cls = type(m)
        if "forward" in m.__dict__:
            continue  # already patched by someone (or by us)
        pymods.add(cls.__module__)
        if cls.__name__.endswith("RMSNorm"):
            if cls not in norm_mode:
                norm_mode[cls] = _probe_norm(cls, m, device)
            if norm_mode[cls] is not None:
                eps = getattr(m, "variance_epsilon", getattr(m, "eps", None))
                m._ts_fused_norm = (float(eps), norm_mode[cls])
                m.forward = types.MethodType(_norm_forward, m)
                done.norms.append(m)
        elif hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj"):
            if cls not in mlp_act:
                mlp_act[cls] = _probe_mlp(cls, m, device)
            if mlp_act[cls] is not None:
                m._ts_fused_act = mlp_act[cls]
                m.forward = types.MethodType(_mlp_forward, m)
                done.mlps.append(m)
                done.acts["silu" if mlp_act[cls] == ACT_SILU else "gelu (tanh)"] = 1
    for name in sorted(pymods):
        mod = sys.modules.get(name)
        fn = getattr(mod, "apply_rotary_pos_emb", None) if mod is not None else None
        if fn is None or not callable(fn):
            continue
        if name in _ROPE:  # patched for another model already
            _ROPE[name][1] += 1
            done.rope.append(name)
            continue
        if getattr(fn, "_ts_original", None) is None and _probe_rope(fn, device):
            _ROPE[name] = [fn, 1]
            mod.apply_rotary_pos_emb = _make_rope(fn)
            done.rope.append(name)
    return done
