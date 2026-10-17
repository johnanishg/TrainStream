"""OffloadAdamW — a torch.optim.Optimizer whose state never touches VRAM."""

from __future__ import annotations

import os
import queue
import sys
import threading
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from . import _C
from .hooks import GradientOffloader, ParamReadyHooks


def physical_ram_bytes() -> int:
    """Total physical memory, from the OS (0 if unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]

            st = _MS()
            st.len = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.total)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return 0


# Tensor attributes and methods that never read a parameter's values: the
# readiness guard lets them through without waiting for in-flight updates.
_META_ATTRS = frozenset({
    "grad", "shape", "dtype", "device", "requires_grad", "is_leaf", "grad_fn", "layout", "is_cuda", "names",
    "ndim", "_version", "is_sparse", "is_quantized", "is_meta", "itemsize", "nbytes", "_base", "output_nr",
})
_META_METHODS = frozenset({
    "size", "dim", "numel", "nelement", "ndimension", "stride", "is_contiguous", "data_ptr", "element_size",
    "untyped_storage", "storage_offset", "is_floating_point", "is_complex", "get_device", "__len__",
    "register_hook", "register_post_accumulate_grad_hook", "requires_grad_", "retain_grad", "__hash__",
    "is_pinned", "is_shared", "type", "__repr__", "__format__",
})


class _ParamGuard(TorchFunctionMode):
    """Makes any read of a parameter whose update is still in flight wait for it.

    Module pre-hooks cover the normal case (a module's own parameters, right
    before it runs). This catches everything else — a weight used functionally
    by another module, read by a loss or logging code, or touched between
    step() and the next forward. It only does work while updates are pending.
    """

    def __init__(self, optimizer: "OffloadAdamW"):
        super().__init__()
        self.opt = optimizer
        self.hits = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        opt = self.opt
        if opt._pending_ids:
            name = getattr(func, "__name__", "")
            if name in ("__get__", "__set__"):
                if getattr(getattr(func, "__self__", None), "__name__", "") in _META_ATTRS:
                    return func(*args, **kwargs)
            elif name in _META_METHODS:
                return func(*args, **kwargs)
            hit = opt._pending_in(args, kwargs)
            if hit:
                self.hits += 1
                opt._wait_slots(hit)
        return func(*args, **kwargs)


class OffloadAdamW(torch.optim.Optimizer):
    """AdamW with its moments (and fp32 master weights) in host RAM.

    mode="cpu"  Gradients stream to host as they are produced; a pool of C++
                worker threads accumulates them and runs AdamW on the host. The
                updated weights are uploaded on a dedicated H2D stream. step()
                returns immediately — the next forward pass waits per module.
                VRAM holds only the weights (+ transient gradients).
    mode="gpu"  Gradients stay in VRAM until step(); a fused CUDA kernel updates
                the weights in place, reading/writing moments through mapped
                pinned memory (zero-copy over PCIe).

    moment_dtype="bfloat16" stores exp_avg/exp_avg_sq in bf16 with stochastic
    rounding: half the host memory and RAM traffic for the moments, at a small
    numerical cost (master weights stay fp32).

    Eager updates (cpu mode, attached): with no gradient clipping, a tensor's
    update is queued the moment its last micro-batch gradient lands, so host
    AdamW overlaps the remainder of backward. Set ``accumulation_steps`` to the
    number of backward passes per step(); extra backward passes before step()
    raise instead of silently changing the math. ``eager=False`` disables it.

    distributed=True (cpu mode, torch.distributed initialized): data parallel.
    Parameters are broadcast from rank 0, and every gradient is averaged across
    ranks on the host (over a gloo group) before its update.

    Hyper-parameters and update math match torch.optim.AdamW (amsgrad and
    maximize are not supported). Works standalone (step() then blocks until the
    update has landed) or, after ``attach(model)`` / ``trainstream.wrap``, fully
    asynchronously.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        mode: str = "cpu",
        max_grad_norm: Optional[float] = None,
        compute_grad_norm: bool = False,
        threads: Optional[int] = None,
        chunk_elems: int = 1 << 20,
        accumulation_steps: int = 1,
        eager: Optional[bool] = None,
        pin_moments: Optional[bool] = None,
        moment_dtype: str = "float32",
        distributed: bool = False,
        _broadcast: bool = True,
        _paged: Optional[set] = None,
        _host_init: Optional[Dict[int, torch.Tensor]] = None,
    ):
        if mode not in ("cpu", "gpu"):
            raise ValueError(f"mode must be 'cpu' or 'gpu', got {mode!r}")
        if moment_dtype in (torch.bfloat16, "bf16"):
            moment_dtype = "bfloat16"
        if moment_dtype in (torch.float32, "fp32"):
            moment_dtype = "float32"
        if moment_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"moment_dtype must be 'float32' or 'bfloat16', got {moment_dtype!r}")
        if not 0.0 <= lr:
            raise ValueError(f"invalid learning rate {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if distributed and mode != "cpu":
            raise ValueError("distributed=True requires mode='cpu' (gradients are averaged on the host)")
        if _paged and mode != "cpu":
            raise ValueError("parameter paging requires mode='cpu'")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        self.engine = None
        super().__init__(params, defaults)

        # Frozen parameters (LoRA/PEFT base weights, quantized layers) get no
        # host state at all: they never receive gradients.
        all_params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        if not all_params:
            raise ValueError("OffloadAdamW got no trainable parameters (all have requires_grad=False)")
        device = all_params[0].device
        if device.type != "cuda":
            raise ValueError("OffloadAdamW requires parameters on a CUDA device")
        self.device = device
        self.mode = mode
        self.max_grad_norm = max_grad_norm
        self.compute_grad_norm = compute_grad_norm
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.eager = eager
        self.moment_dtype = moment_dtype
        self._threads = int(threads or 0)
        self._chunk = int(chunk_elems)
        self._pin_request = pin_moments
        self._paged_ids = set(_paged or ())
        self._host_init = dict(_host_init or {})

        self._pending_ids: Dict[int, int] = {}   # id(param) -> slot with an update in flight
        self._pending_ptrs: Dict[int, int] = {}  # data_ptr -> slot (aliases such as .data)
        self._guard: Optional[_ParamGuard] = None
        self._resident: Optional[torch.optim.AdamW] = None
        self._suspended: Optional[dict] = None
        self._post_step_cb = None
        self._last_norm: Optional[float] = None

        self._dp = None
        if distributed:
            self._init_distributed(broadcast=_broadcast)
        self._build_engine()

    # -- engine lifecycle ---------------------------------------------------------
    def _build_engine(self) -> None:
        params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        self.pin_moments = self._decide_pinning(params, self.mode, self._pin_request, self.moment_dtype)
        self.engine = _C.OffloadAdamWEngine(self.device.index or 0, self.mode, self._threads, self._chunk,
                                            self.moment_dtype)
        self._slot_of: Dict[int, int] = {}
        self._slot_params: List[torch.Tensor] = []
        self._slot_group: List[dict] = []
        self._micro: List[int] = []
        for group in self.param_groups:
            self._register_group(group)
        self._host_init.clear()  # CPU copies are now inside the engine
        self._forward_order: List[int] = []
        self._order: Optional[List[int]] = None
        self._ready_hooks: Optional[ParamReadyHooks] = None
        self._grad_hooks: Optional[GradientOffloader] = None
        self._fwd_handle = None
        self._pending_ids.clear()
        self._pending_ptrs.clear()

    def _register_group(self, group: dict) -> None:
        for p in group["params"]:
            if id(p) in self._slot_of or not p.requires_grad:
                continue
            if p.device != self.device:
                raise ValueError(f"all parameters must be on {self.device}, found {p.device}")
            host_init = self._host_init.pop(id(p), None)
            slot = self.engine.add_param(p, self.pin_moments, host_init)
            if id(p) in self._paged_ids:
                self.engine.set_paged(slot, True)
            self._slot_of[id(p)] = slot
            self._slot_params.append(p)
            self._slot_group.append(group)
            self._micro.append(0)

    @staticmethod
    def _decide_pinning(params, mode: str, requested: Optional[bool], moment_dtype: str = "float32") -> bool:
        """Page-lock the moments unless that would push the process past a
        safe share of physical RAM (the OS starts failing DMA mappings when too
        much memory is locked). The zero-copy GPU mode always pins."""
        if mode == "gpu" or requested is not None:
            return True if mode == "gpu" else bool(requested)
        numel = sum(p.numel() for p in params)
        pbytes = params[0].element_size()
        always_pinned = numel * (4 + (pbytes if pbytes < 4 else 0))  # master + rounded staging copy
        moments = numel * (4 if moment_dtype == "bfloat16" else 8)
        total_ram = physical_ram_bytes()
        budget = 0.40 * total_ram if total_ram else float("inf")
        live = _C.pinned_stats()["live_bytes"]
        return live + always_pinned + moments <= budget

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if self.engine is not None:  # groups added after construction
            self._register_group(self.param_groups[-1])
            self._order = None

    def _rebuild(self, device: Optional[torch.device] = None) -> None:
        """Re-create the engine for the parameters' current storage (after the
        model was moved or cast), carrying the optimizer state across."""
        state = self._suspended if self._suspended is not None else self.state_dict()
        self.detach()
        self.engine = None
        self._suspended = None
        params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        self.device = device or params[0].device
        self._build_engine()
        self.load_state_dict(state)

    def _suspend(self) -> None:
        """The model left the GPU: keep the state on the host, drop the engine."""
        self._suspended = self.state_dict()
        self.detach()
        self.engine = None

    # -- attachment --------------------------------------------------------------
    def attach(self, model: nn.Module, guard: bool = True) -> "OffloadAdamW":
        """Install per-module readiness hooks (and gradient streaming in cpu mode)."""
        self.detach()
        if self._resident is not None:
            return self
        self._ready_hooks = ParamReadyHooks(model, self._wait_slots, self._slot_of, self._forward_order)
        if self.mode == "cpu":
            self._grad_hooks = GradientOffloader(self._slot_params, self._on_grad, self._slot_of)
        # Any weight still in flight when a forward pass ends (used outside its
        # module, or not at all) is made ready before the caller sees outputs.
        self._fwd_handle = model.register_forward_hook(lambda m, a, o: self._wait_all_pending(), always_call=True)
        if guard:
            self._guard = _ParamGuard(self)
            self._guard.__enter__()
        return self

    def detach(self) -> None:
        self.synchronize()
        if self._ready_hooks is not None:
            self._ready_hooks.remove()
            self._ready_hooks = None
        if self._grad_hooks is not None:
            self._grad_hooks.remove()
            self._grad_hooks = None
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        self._pop_guard()
        self._pending_ids.clear()
        self._pending_ptrs.clear()

    def _pop_guard(self) -> None:
        g = self._guard
        if g is None:
            return
        self._guard = None
        g.opt = _NoPending  # inert even if it cannot be popped (another mode above it)
        try:
            from torch.overrides import _get_current_function_mode
            if _get_current_function_mode() is g:
                g.__exit__(None, None, None)
        except Exception:
            pass

    @property
    def attached(self) -> bool:
        return self._ready_hooks is not None

    @property
    def eager_active(self) -> bool:
        if self.mode != "cpu" or not self.attached or self._dp is not None:
            return False
        if self.eager is not None:
            return bool(self.eager) and not self.max_grad_norm
        return not self.max_grad_norm and not self.compute_grad_norm

    @property
    def resident(self) -> bool:
        return self._resident is not None

    # -- readiness ---------------------------------------------------------------
    def _mark_pending(self, slot: int) -> None:
        p = self._slot_params[slot]
        if id(p) in self._paged_ids:
            return  # the pager streams paged weights in itself
        self._pending_ids[id(p)] = slot
        self._pending_ptrs[p.data_ptr()] = slot

    def _wait_slots(self, slots) -> None:
        self.engine.wait_ready(slots)
        if self._pending_ids:
            for s in slots:
                p = self._slot_params[s]
                self._pending_ids.pop(id(p), None)
                self._pending_ptrs.pop(p.data_ptr(), None)

    def _wait_all_pending(self) -> None:
        if self._pending_ids:
            self._wait_slots(sorted(set(self._pending_ids.values())))

    def _pending_in(self, args, kwargs) -> List[int]:
        ids, ptrs = self._pending_ids, self._pending_ptrs
        hit: List[int] = []

        def check(a):
            s = ids.get(id(a))
            if s is None and isinstance(a, torch.Tensor) and a.is_cuda:
                try:
                    s = ptrs.get(a.data_ptr())
                except RuntimeError:
                    s = None
            if s is not None:
                hit.append(s)

        for a in args:
            if isinstance(a, (list, tuple)):
                for b in a:
                    check(b)
            else:
                check(a)
        for a in kwargs.values():
            check(a)
        return hit

    # -- gradient path -----------------------------------------------------------
    def _on_grad(self, slot: int, grad: torch.Tensor) -> None:
        """Called from the gradient hook (autograd thread)."""
        self._micro[slot] += 1
        n = self._micro[slot]
        if self.eager_active or self._dp is not None:
            if n > self.accumulation_steps:
                raise RuntimeError(
                    f"TrainStream: parameter received {n} backward passes before optimizer.step() but "
                    f"accumulation_steps={self.accumulation_steps}. Pass accumulation_steps=<backwards per step> "
                    "to trainstream.wrap()/OffloadAdamW (or eager=False).")
        if self._dp is not None:
            self.engine.offload_grad(slot, grad)
            if n == self.accumulation_steps:
                self._dp["queue"].put(slot)
            return
        if self.eager_active and n == self.accumulation_steps:
            g = self._slot_group[slot]
            self.engine.offload_grad(slot, grad, True, float(g["lr"]), float(g["betas"][0]),
                                     float(g["betas"][1]), float(g["eps"]), float(g["weight_decay"]))
            self._mark_pending(slot)
            return
        self.engine.offload_grad(slot, grad)

    def _update_order(self) -> List[int]:
        if self._order is None:
            seen = set(self._forward_order)
            rest = [i for i in range(len(self._slot_params)) if i not in seen]
            self._order = list(self._forward_order) + rest
            if self._ready_hooks is not None and self._forward_order:
                self._ready_hooks.stop_recording()
        return self._order

    # -- data parallel -------------------------------------------------------------
    def _init_distributed(self, broadcast: bool = True) -> None:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed=True needs torch.distributed.init_process_group() first")
        world = dist.get_world_size()
        # Gradients are reduced where they already are: in host RAM. gloo works on
        # CPU tensors under any default backend, on Windows and Linux alike.
        group = dist.new_group(backend="gloo")
        with torch.no_grad():
            for g in (self.param_groups if broadcast else ()):
                for p in g["params"]:
                    host = p.detach().to("cpu", copy=True)
                    dist.broadcast(host, src=0, group=group)
                    p.copy_(host)
        self._dp = {"group": group, "world": world, "queue": queue.Queue(), "error": None}
        t = threading.Thread(target=self._reduce_loop, name="trainstream-allreduce", daemon=True)
        self._dp["thread"] = t
        t.start()

    def _reduce_loop(self) -> None:
        import torch.distributed as dist
        dp = self._dp
        q = dp["queue"]
        while True:
            slot = q.get()
            if slot is None:
                q.task_done()
                return
            try:
                buf = self.engine.grad_buffer(slot)  # waits for this slot's accumulation
                dist.all_reduce(buf, group=dp["group"])  # same order on every rank (backward order)
                if not self.max_grad_norm and not self.compute_grad_norm:
                    g = self._slot_group[slot]
                    self.engine.update_slot(slot, float(g["lr"]), float(g["betas"][0]), float(g["betas"][1]),
                                            float(g["eps"]), float(g["weight_decay"]), 1.0 / dp["world"])
                    self._mark_pending(slot)
            except Exception as exc:  # surfaced on the training thread at step()
                dp["error"] = exc
            finally:
                q.task_done()

    # -- optimisation ----------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        if self._suspended is not None:
            raise RuntimeError("TrainStream: the model was moved off the GPU; move it back to a CUDA device "
                               "before calling optimizer.step()")
        if self._resident is not None:
            return self._resident_step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.mode == "cpu":
            # Gradients not already streamed by hooks (standalone use, or params
            # outside the attached model) are offloaded here.
            for slot, p in enumerate(self._slot_params):
                if p.grad is not None:
                    self.engine.offload_grad(slot, p.grad)
                    p.grad = None

        scale = 1.0
        if self._dp is not None:
            self._dp["queue"].join()  # every gradient averaged (and, without clipping, its update queued)
            if self._dp["error"] is not None:
                err, self._dp["error"] = self._dp["error"], None
                raise RuntimeError(f"TrainStream data-parallel gradient reduction failed: {err}") from err
            scale = 1.0 / self._dp["world"]

        ids, lrs, b1s, b2s, epss, wds, grads = [], [], [], [], [], [], []
        for slot in self._update_order():
            p = self._slot_params[slot]
            if self.mode == "gpu":
                if p.grad is None:
                    continue
                grads.append(p.grad)
            group = self._slot_group[slot]
            ids.append(slot)
            lrs.append(float(group["lr"]))
            b1s.append(float(group["betas"][0]))
            b2s.append(float(group["betas"][1]))
            epss.append(float(group["eps"]))
            wds.append(float(group["weight_decay"]))

        self._micro = [0] * len(self._slot_params)
        norm = self.engine.step(ids, lrs, b1s, b2s, epss, wds, grads,
                                float(self.max_grad_norm or 0.0), bool(self.compute_grad_norm), scale)
        if norm >= 0:
            self._last_norm = norm
        for slot in ids:
            self._mark_pending(slot)
        if self.mode == "gpu":
            for slot in ids:
                self._slot_params[slot].grad = None  # memory recycled once the kernel ran
        if not self.attached:
            self.engine.synchronize()
            self._pending_ids.clear()
            self._pending_ptrs.clear()
        if self._post_step_cb is not None:
            self._post_step_cb()
        return loss

    def _resident_step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        if self.max_grad_norm or self.compute_grad_norm:
            total = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm or float("inf"))
            self._last_norm = float(total)
        self._resident.step()
        return loss

    def _go_resident(self) -> None:
        """Switch to torch.optim.AdamW(fused=True) with all state in VRAM.

        Used when the measured footprint shows the model fits on the GPU with a
        stock optimizer: offloading would only cost time. The optimizer object
        and its param_group dicts stay the same, so LR schedulers keep working.
        """
        self.synchronize()
        moved = {}
        for slot, p in enumerate(self._slot_params):
            step = self.engine.get_step(slot)
            if step > 0:
                _, m, v = self.engine.state(slot)
                moved[id(p)] = (step, m.view(p.shape).to(p.device, p.dtype), v.view(p.shape).to(p.device, p.dtype))
        self.detach()
        self.engine = None
        inner = torch.optim.AdamW(self.param_groups, fused=True)  # shares our group dicts
        for p in self._slot_params:
            if id(p) in moved:
                step, m, v = moved[id(p)]
                inner.state[p] = {"step": torch.tensor(float(step), device=p.device, dtype=torch.float32),
                                  "exp_avg": m, "exp_avg_sq": v}
        self._resident = inner
        self.state = inner.state

    def _relink_resident(self) -> None:
        """torch's load_state_dict replaces the group dicts (and resets
        `fused`); point the inner optimizer back at ours."""
        inner = self._resident
        for ours, theirs in zip(self.param_groups, inner.param_groups):
            for k, v in theirs.items():
                if k != "params":
                    ours[k] = v
            ours["fused"] = True
        inner.param_groups = self.param_groups
        for st in inner.state.values():
            if torch.is_tensor(st.get("step")) and st["step"].device.type == "cpu":
                dev = st["exp_avg"].device
                st["step"] = st["step"].to(dev, torch.float32)
        self.state = inner.state

    def synchronize(self) -> None:
        """Block until every queued update has landed in the GPU parameters."""
        if self._dp is not None and self.engine is not None:
            self._dp["queue"].join()
        if self.engine is not None:
            self.engine.synchronize()
        self._pending_ids.clear()
        self._pending_ptrs.clear()

    @property
    def last_grad_norm(self) -> Optional[float]:
        if self._resident is None and self.engine is not None and (self.max_grad_norm or self.compute_grad_norm):
            n = self.engine.last_grad_norm()
            return n if n >= 0 else None
        return self._last_norm

    def stats(self) -> dict:
        if self.engine is None:
            return {"mode": "resident" if self._resident is not None else "suspended"}
        s = dict(self.engine.stats())
        s["mode"] = self.mode
        s["moment_dtype"] = self.moment_dtype
        if self._grad_hooks is not None:
            s["grad_hook_calls"] = self._grad_hooks.calls
        if self._guard is not None:
            s["guard_waits"] = self._guard.hits
        return s

    # -- (de)serialisation -------------------------------------------------------
    def state_dict(self) -> dict:
        """torch.optim.AdamW-compatible state dict (plus fp32 ``master`` weights)."""
        if self._resident is not None:
            return self._resident.state_dict()
        if self._suspended is not None:
            return self._suspended
        self.synchronize()
        state, groups, index = {}, [], 0
        for group in self.param_groups:
            packed = {k: v for k, v in group.items() if k != "params"}
            packed["params"] = list(range(index, index + len(group["params"])))
            for p in group["params"]:
                slot = self._slot_of.get(id(p))
                if slot is None:  # frozen parameter: no state, keep the index
                    index += 1
                    continue
                master, m, v = self.engine.state(slot)
                entry = {
                    "step": torch.tensor(float(self.engine.get_step(slot))),
                    "exp_avg": m.view(p.shape).to(torch.float32, copy=True),
                    "exp_avg_sq": v.view(p.shape).to(torch.float32, copy=True),
                }
                if master is not None:
                    entry["master"] = master.view(p.shape).clone()
                if self.engine.get_step(slot) > 0:
                    state[index] = entry
                index += 1
            groups.append(packed)
        return {"state": state, "param_groups": groups}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load an OffloadAdamW *or* a torch.optim.Adam/AdamW state dict."""
        if self._resident is not None:
            self._resident.load_state_dict(state_dict)
            self._relink_resident()
            return
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("state dict has a different number of parameter groups")
        by_index = {}
        for group, saved in zip(self.param_groups, saved_groups):
            if len(group["params"]) != len(saved["params"]):
                raise ValueError("state dict group sizes do not match the optimizer")
            for k, v in saved.items():
                if k != "params" and k in self.defaults:
                    group[k] = tuple(v) if k == "betas" else v
            for idx, p in zip(saved["params"], group["params"]):
                by_index[idx] = p
        self.synchronize()
        for idx, entry in state_dict["state"].items():
            p = by_index[int(idx)]
            if id(p) not in self._slot_of:
                continue  # frozen here
            step = entry.get("step", 0)
            step = int(step.item() if torch.is_tensor(step) else step)

            def host(t):
                return t.detach().to("cpu", torch.float32)

            master = entry.get("master")
            self.engine.load_state(
                self._slot_of[id(p)],
                host(entry["exp_avg"]),
                host(entry["exp_avg_sq"]),
                host(master) if master is not None else None,
                step,
            )

    def __repr__(self) -> str:
        if self._resident is not None:
            return f"OffloadAdamW(resident: torch.optim.AdamW(fused=True), params={len(self._slot_params)})"
        if self.engine is None:
            return "OffloadAdamW(suspended: model is not on a CUDA device)"
        n = sum(p.numel() for p in self._slot_params)
        return (f"OffloadAdamW(mode={self.mode}, params={len(self._slot_params)}, numel={n:,}, "
                f"moments={self.moment_dtype}, threads={self.engine.threads}, attached={self.attached})")


class _NoPendingType:
    _pending_ids: Dict[int, int] = {}


_NoPending = _NoPendingType()
