"""OffloadAdamW — a torch.optim.Optimizer whose state never touches VRAM."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
from torch import nn

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


class OffloadAdamW(torch.optim.Optimizer):
    """AdamW with fp32 moments (and master weights) in pinned host RAM.

    mode="cpu"  Gradients stream to host as they are produced; a pool of C++
                worker threads accumulates them and runs AdamW on the host. The
                updated weights are uploaded on a dedicated H2D stream. step()
                returns immediately — the next forward pass waits per module.
                VRAM holds only the weights (+ transient gradients).
    mode="gpu"  Gradients stay in VRAM until step(); a fused CUDA kernel updates
                the weights in place, reading/writing moments through mapped
                pinned memory (zero-copy over PCIe).

    Eager updates (cpu mode, attached): with no gradient clipping, a tensor's
    update is queued the moment its last micro-batch gradient lands, so host
    AdamW overlaps the remainder of backward. Set ``accumulation_steps`` to the
    number of backward passes per step(); extra backward passes before step()
    raise instead of silently changing the math. ``eager=False`` disables it.

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
    ):
        if mode not in ("cpu", "gpu"):
            raise ValueError(f"mode must be 'cpu' or 'gpu', got {mode!r}")
        if not 0.0 <= lr:
            raise ValueError(f"invalid learning rate {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
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
        self.pin_moments = self._decide_pinning(all_params, mode, pin_moments)
        self.engine = _C.OffloadAdamWEngine(device.index or 0, mode, int(threads or 0), int(chunk_elems))
        self._slot_of: Dict[int, int] = {}
        self._slot_params: List[torch.Tensor] = []
        self._slot_group: List[dict] = []
        for group in self.param_groups:
            self._register_group(group)

        self._forward_order: List[int] = []
        self._order: Optional[List[int]] = None
        self._ready_hooks: Optional[ParamReadyHooks] = None
        self._grad_hooks: Optional[GradientOffloader] = None
        self._last_norm: Optional[float] = None
        self._micro: List[int] = [0] * len(self._slot_params)

    # -- registration --------------------------------------------------------
    def _register_group(self, group: dict) -> None:
        for p in group["params"]:
            if id(p) in self._slot_of or not p.requires_grad:
                continue
            if p.device != self.device:
                raise ValueError(f"all parameters must be on {self.device}, found {p.device}")
            slot = self.engine.add_param(p, self.pin_moments)
            self._slot_of[id(p)] = slot
            self._slot_params.append(p)
            self._slot_group.append(group)
            if hasattr(self, "_micro"):
                self._micro.append(0)

    @staticmethod
    def _decide_pinning(params, mode: str, requested: Optional[bool]) -> bool:
        """Page-lock the moments unless that would push the process past a
        safe share of physical RAM (the OS starts failing DMA mappings when too
        much memory is locked). The zero-copy GPU mode always pins."""
        if mode == "gpu" or requested is not None:
            return True if mode == "gpu" else bool(requested)
        numel = sum(p.numel() for p in params)
        pbytes = params[0].element_size()
        always_pinned = numel * (4 + pbytes + (pbytes if pbytes < 4 else 0))  # master + grad/param staging
        moments = numel * 8
        total_ram = physical_ram_bytes()
        budget = 0.40 * total_ram if total_ram else float("inf")
        live = _C.pinned_stats()["live_bytes"]
        return live + always_pinned + moments <= budget

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if self.engine is not None:  # groups added after construction
            self._register_group(self.param_groups[-1])
            self._order = None

    def attach(self, model: nn.Module) -> "OffloadAdamW":
        """Install per-module readiness hooks (and gradient streaming in cpu mode)."""
        self.detach()
        self._ready_hooks = ParamReadyHooks(model, self.engine, self._slot_of, self._forward_order)
        if self.mode == "cpu":
            self._grad_hooks = GradientOffloader(self._slot_params, self._on_grad, self._slot_of)
        return self

    def detach(self) -> None:
        self.synchronize()
        if self._ready_hooks is not None:
            self._ready_hooks.remove()
            self._ready_hooks = None
        if self._grad_hooks is not None:
            self._grad_hooks.remove()
            self._grad_hooks = None

    @property
    def attached(self) -> bool:
        return self._ready_hooks is not None

    @property
    def eager_active(self) -> bool:
        if self.mode != "cpu" or not self.attached:
            return False
        if self.eager is not None:
            return bool(self.eager) and not self.max_grad_norm
        return not self.max_grad_norm and not self.compute_grad_norm

    def _on_grad(self, slot: int, grad: torch.Tensor) -> None:
        """Called from the gradient hook (autograd thread)."""
        self._micro[slot] += 1
        n = self._micro[slot]
        if self.eager_active:
            if n > self.accumulation_steps:
                raise RuntimeError(
                    f"TrainStream: parameter received {n} backward passes before optimizer.step() but "
                    f"accumulation_steps={self.accumulation_steps}. Pass accumulation_steps=<backwards per step> "
                    "to trainstream.wrap()/OffloadAdamW (or eager=False).")
            if n == self.accumulation_steps:
                g = self._slot_group[slot]
                self.engine.offload_grad(slot, grad, True, float(g["lr"]), float(g["betas"][0]),
                                         float(g["betas"][1]), float(g["eps"]), float(g["weight_decay"]))
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

    # -- optimisation ----------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
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
                                float(self.max_grad_norm or 0.0), bool(self.compute_grad_norm))
        if norm >= 0:
            self._last_norm = norm
        if self.mode == "gpu":
            for slot in ids:
                self._slot_params[slot].grad = None  # memory recycled once the kernel ran
        if not self.attached:
            self.engine.synchronize()
        return loss

    def synchronize(self) -> None:
        """Block until every queued update has landed in the GPU parameters."""
        if self.engine is not None:
            self.engine.synchronize()

    @property
    def last_grad_norm(self) -> Optional[float]:
        if self.max_grad_norm or self.compute_grad_norm:
            n = self.engine.last_grad_norm()
            return n if n >= 0 else None
        return self._last_norm

    def stats(self) -> dict:
        s = dict(self.engine.stats())
        s["mode"] = self.mode
        if self._grad_hooks is not None:
            s["grad_hook_calls"] = self._grad_hooks.calls
        return s

    # -- (de)serialisation -------------------------------------------------------
    def state_dict(self) -> dict:
        """torch.optim.AdamW-compatible state dict (plus fp32 ``master`` weights)."""
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
                    "exp_avg": m.view(p.shape).clone(),
                    "exp_avg_sq": v.view(p.shape).clone(),
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
        n = sum(p.numel() for p in self._slot_params)
        return (f"OffloadAdamW(mode={self.mode}, params={len(self._slot_params)}, "
                f"numel={n:,}, threads={self.engine.threads}, attached={self.attached})")
