"""Attention-backend selection for ``scaled_dot_product_attention``.

PyTorch picks an SDPA kernel by a fixed priority list, which is not always the
fastest one on a given GPU: on an RTX 5090 (sm_120, torch 2.14) it chooses the
memory-efficient kernel, while cuDNN's runs forward + backward 2-3x faster.
With ``attention="auto"`` TrainStream times the available backends once, on
the model's own head shape and sequence length (taken from the first training
forward), and runs every later forward with the winner first in the priority
order; the other backends stay behind it, so inputs the winner cannot take (an
unusual mask, fp32) still run. A forward given an attention mask or position
ids (padding, packed documents) keeps PyTorch's own order instead: the timing
was without a mask, and cuDNN's backward rejects the expanded masks
transformers builds. The backward kernel follows from the forward one. Recomputed blocks (activation recomputation) use the same order, so the
recomputed forward matches the original.

Under ``torch.use_deterministic_algorithms(True)`` nondeterministic kernels
(cuDNN, flash) are left out. With ``warn_only=True`` the memory-efficient kernel
is left out too: PyTorch then keeps its nondeterministic backward (and only
warns), while ``warn_only=False`` switches it to a deterministic one. The order
is read at every training forward, so turning deterministic mode on or off
after ``wrap()`` takes effect, recomputed blocks included.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - very old PyTorch
    SDPBackend = None
    sdpa_kernel = None

CHOICES = ("auto", "cudnn", "efficient", "flash", "math")
_NONDETERMINISTIC = ("cudnn", "flash")
_TIMINGS: Dict[Tuple, Dict[str, float]] = {}


def _backend(name: str):
    return {"cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH}[name]


def _nondeterministic() -> Tuple[str, ...]:
    """The backends deterministic mode leaves out right now (none while it is off)."""
    if not torch.are_deterministic_algorithms_enabled():
        return ()
    if torch.is_deterministic_algorithms_warn_only_enabled():
        return _NONDETERMINISTIC + ("efficient",)  # warn-only keeps its nondeterministic backward
    return _NONDETERMINISTIC


def _order(first: str, measured: Optional[Dict[str, float]] = None) -> List:
    if measured:  # only backends that ran this shape, fastest first; math always works
        names = sorted(measured, key=measured.get) + ["math"]
    else:
        names = [first] + [n for n in ("cudnn", "flash", "efficient", "math") if n != first]
    skip = _nondeterministic()
    names = [n for n in names if n not in skip] or ["math"]
    return [_backend(n) for n in names]


def time_backends(batch: int, heads: int, seq: int, head_dim: int, dtype, device, *,
                  candidates: Sequence[str] = ("cudnn", "flash", "efficient")) -> Dict[str, float]:
    """Forward + backward milliseconds of each backend that runs this shape
    (causal attention); backends that cannot run it are left out."""
    key = (batch, heads, seq, head_dim, dtype, torch.device(device).index, tuple(candidates))
    if key in _TIMINGS:
        return _TIMINGS[key]
    out: Dict[str, float] = {}
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype, generator=gen).requires_grad_()
               for _ in range(3))
    go = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype, generator=gen)
    import warnings
    for name in candidates:
        try:
            with sdpa_kernel([_backend(name)]), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                def step():
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    o.backward(go)
                step()  # warm-up (and the check that this backend runs the shape at all)
                torch.cuda.synchronize(device)
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                for _ in range(3):
                    step()
                e.record()
                torch.cuda.synchronize(device)
                out[name] = s.elapsed_time(e) / 3
        except Exception:  # noqa: BLE001 - "No available kernel" etc.
            continue
        finally:
            q.grad = k.grad = v.grad = None
    _TIMINGS[key] = out
    return out


def _masked(args, kwargs) -> bool:
    """Whether a model call may attend through a mask: an attention_mask or position_ids passed."""
    if kwargs.get("attention_mask") is not None or kwargs.get("position_ids") is not None:
        return True
    return len(args) > 1 and isinstance(args[1], torch.Tensor)  # forward(input_ids, attention_mask, ...)


def _head_shape(model: nn.Module) -> Optional[Tuple[int, int]]:
    cfg = getattr(model, "config", None)
    heads = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
    if not heads:
        return None
    head_dim = getattr(cfg, "head_dim", None) or getattr(cfg, "hidden_size", 0) // heads
    return (int(heads), int(head_dim)) if head_dim else None


def _batch_seq(args, kwargs) -> Optional[Tuple[int, int]]:
    for t in list(args) + [kwargs.get("input_ids"), kwargs.get("inputs_embeds")]:
        if isinstance(t, torch.Tensor) and t.dim() >= 2:
            return int(t.shape[0]), int(t.shape[1])
    return None


class AttentionSelector:
    """Runs the model's forward passes with a chosen SDPA backend order."""

    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module], mode: str, device, dtype=None):
        if mode not in CHOICES:
            raise ValueError(f"attention must be one of {CHOICES} or None, got {mode!r}")
        self.model = model
        self.blocks = list(blocks)
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.choice: Optional[str] = None if mode == "auto" else mode
        self.timings: Dict[str, float] = {}
        self.events: List[str] = []
        self._ctx = None
        self._depth = 0
        self._masked_noted = False
        self._handles = [model.register_forward_pre_hook(self._pre, with_kwargs=True),
                         model.register_forward_hook(self._post, always_call=True)]
        if self.choice is not None:
            self._apply()

    def _decide(self, args, kwargs) -> None:
        shape = _head_shape(self.model)
        bs = _batch_seq(args, kwargs)
        if shape is None or bs is None:
            self.choice = "efficient"  # no attention shape to measure: keep PyTorch's usual pick
            self.events.append("attention: no head configuration found; PyTorch's default backend order kept")
            self._apply()
            return
        dtype = self.dtype
        if torch.is_autocast_enabled("cuda"):
            dtype = torch.get_autocast_dtype("cuda")
        dtype = dtype or next(self.model.parameters()).dtype
        cands = [n for n in ("cudnn", "flash", "efficient") if n not in _nondeterministic()]
        self.timings = time_backends(bs[0], shape[0], bs[1], shape[1], dtype, self.device, candidates=cands)
        self.choice = min(self.timings, key=self.timings.get) if self.timings else "math"
        parts = ", ".join(f"{k} {v:.2f} ms" for k, v in sorted(self.timings.items(), key=lambda kv: kv[1]))
        self.events.append(f"attention: {self.choice} backend first (forward+backward on {bs[0]}x{shape[0]}x"
                           f"{bs[1]}x{shape[1]} {str(dtype).replace('torch.', '')}: {parts})")
        self._apply()

    def _apply(self) -> List:
        order = _order(self.choice, self.timings)
        for b in self.blocks:
            b._ts_sdpa_order = order  # recomputed blocks re-enter the same order
        return order

    def order(self):
        return _order(self.choice, self.timings) if self.choice else None

    def _pre(self, module, args, kwargs):
        self._depth += 1
        if self._depth > 1:
            return None
        if self.choice is None and torch.is_grad_enabled():
            self._decide(args, kwargs)
        if self.choice is not None and _masked(args, kwargs):
            # The timing used causal attention without a mask. With one (padding, packed documents),
            # cuDNN's backward rejects the expanded masks transformers builds: keep PyTorch's own
            # order (the memory-efficient kernel), for recomputed blocks too.
            for b in self.blocks:
                b._ts_sdpa_order = None
            if not self._masked_noted:
                self._masked_noted = True
                self.events.append("attention: inputs with an attention mask or position ids (padding, packed "
                                   "documents) keep PyTorch's default backend order")
            return None
        if self.choice is not None:
            # read again every forward (deterministic mode may have changed), for the blocks too
            self._ctx = sdpa_kernel(self._apply(), set_priority=True)
            self._ctx.__enter__()
        return None

    def _post(self, module, args, output):
        self._depth -= 1
        if self._depth == 0 and self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    def remove(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for b in self.blocks:
            b.__dict__.pop("_ts_sdpa_order", None)
