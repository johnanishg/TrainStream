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
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Dict, List, Optional

import torch
from torch import nn

from . import _C

_FLOATS = (torch.float32, torch.float16, torch.bfloat16)
ACT_SILU, ACT_GELU_TANH = 0, 1


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


# ------------------------------------------------------------------ patched forwards
def _norm_forward(self, x):
    eps, gemma = self._ts_fused_norm
    w = self.weight
    if (not x.is_cuda or x.dtype not in _FLOATS or w.dtype not in _FLOATS or w.device != x.device
            or (not gemma and w.dtype != x.dtype) or x.numel() == 0):
        return type(self).forward(self, x)  # e.g. an fp32 weight on bf16 input: PyTorch promotes the result
    return RMSNormFn.apply(x, w, eps, gemma)


def _mlp_forward(self, x):
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
