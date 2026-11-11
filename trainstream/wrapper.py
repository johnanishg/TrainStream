"""One-line integration: ``model, optimizer = trainstream.wrap(model, optimizer)``."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn

from . import _C
from .hooks import ActivationOffloader, find_blocks
from .optimizer import OffloadAdamW
from .paging import ParamPager, pageable_params
from .planner import MemoryPlanner, explain_ram_shortfall, host_bytes_needed, measure_budget
from . import experts, fused, lora, lowp
from . import quant as Q
from .attention import CHOICES as ATTENTION_CHOICES, AttentionSelector
from .profiler import HardwareProfile, profile_hardware

_ATTR = "_trainstream_session"
PEER_SETUP_STEP = 10  # after the memory plan's refinement (at most 8 steps) has settled


class TrainStreamSession:
    """Everything wrap() installed on a model; reachable as ``model._trainstream_session``."""

    def __init__(self, model, optimizer, profile, config, blocks, paged_blocks: List[int]):
        self.model = model
        self.optimizer: OffloadAdamW = optimizer
        self.activations: Optional[ActivationOffloader] = None
        self.pager: Optional[ParamPager] = None
        self.profile: Optional[HardwareProfile] = profile
        self.config = config
        self.blocks = list(blocks)
        self.paged_blocks = list(paged_blocks)
        self.planner: Optional[MemoryPlanner] = None
        self.candidates = None
        self._act_prev_bytes = 0
        self._act_step_bytes = 0
        self._act_room_step: Optional[int] = None
        self._act_room_done = False
        self._resident_check_step: Optional[int] = None
        self._watch_level = 0
        self.steps = 0
        self.events: List[str] = []
        self._applying = False
        self._orig_apply: Dict[int, object] = {}
        self.lowp_modules: List[nn.Module] = []
        self.fused = None  # trainstream.fused.Fused: the fused norm / RoPE / MLP kernels installed
        self.lora_layers: List[nn.Module] = []  # PEFT LoRA Linears routed through trainstream.lora
        self.expert_pagers: list = []  # trainstream.experts.ExpertPager per paged stack of MoE experts
        self.attention: Optional[AttentionSelector] = None
        self.peer = None  # trainstream.peer.PeerPool once set up
        self._peer_step: Optional[int] = None

    # -- install / teardown ----------------------------------------------------------
    def install(self, frozen_init=None) -> None:
        cfg = self.config
        opt = self.optimizer
        opt.attach(self.model, guard=cfg["guard_params"])
        if cfg.get("kernels") and self.fused is None:
            self.fused = fused.install(self.model, opt.device)
        if cfg.get("fused_lora") and not self.lora_layers:
            self.lora_layers = lora.install(self.model)
        if opt.resident:
            return
        if self.config.get("quantize") in Q.BLOCKSCALED_KINDS:  # before the pager reads the flags
            self.lowp_modules = lowp.install(self.blocks, opt.device, cfg.get("outliers", 0.0),
                                             cfg.get("nvfp4_backward", False))
        if cfg.get("attention") is not None and self.blocks:
            self.attention = AttentionSelector(self.model, self.blocks, cfg["attention"], opt.device)
            self.attention.events = self.events
        if self.paged_blocks or self.config.get("quantize") or (self.planner is not None and not self.planner.done):
            if self.candidates is None:
                self.candidates = pageable_params(self.model, self.blocks, range(len(self.blocks)))
            self.pager = ParamPager(self.model, self.blocks, opt, self.paged_blocks, candidates=self.candidates,
                                    frozen_init=frozen_init, prefetch_depth=cfg["prefetch_depth"],
                                    disk=cfg.get("_disk"))
        rc = cfg.get("recompute", "auto")
        explicit_rc = rc not in ("auto", False, None, 0, 0.0)
        if (cfg["activation_offload"] not in (False, 0, 0.0, None) or explicit_rc) and self.blocks:
            act = cfg["activation_offload"] if not cfg.get("_act_fixed") else cfg["_act_fixed"]
            self.activations = ActivationOffloader(
                self.model, self.blocks, act if act not in (None,) else False, min_bytes=cfg["min_offload_bytes"],
                prefetch_depth=cfg["prefetch_depth"], profile=self.profile,
                max_slowdown=cfg["max_activation_slowdown"], reserved_link_bytes=cfg["_reserved"],
                max_inflight_bytes=cfg["max_inflight_bytes"], recompute=rc,
                memory_pressure=bool(cfg.get("_act_forced")), vram_target=cfg.get("_vram_target"))
            self.activations.events = self.events
            self.activations.weights_paged = lambda: self.pager is not None and bool(self.pager.remote)
            self._act_room_step, self._act_room_done = None, False

    def teardown(self, materialize: bool = True) -> None:
        if self.pager is not None:
            self.paged_blocks = sorted(self.pager.remote)
        self.optimizer.detach()
        if self.activations is not None:
            self.activations.remove()
            self.activations = None
        if self.pager is not None:
            self.pager.remove(materialize=materialize)  # still reads pooled weights from the peers
            self.pager = None
        if self.peer is not None:
            self.peer.close()
            self.peer = None
            self.events.append("peer GPUs: pools released")
        lowp.uninstall(self.lowp_modules)
        self.lowp_modules = []
        if self.fused is not None:
            self.fused.remove()
            self.fused = None
        lora.uninstall(self.lora_layers)
        self.lora_layers = []
        if self.attention is not None:
            self.attention.remove()
            self.attention = None

    def remove(self) -> None:
        self.teardown(materialize=True)
        for ep in self.expert_pagers:
            ep.remove()
        self.expert_pagers = []
        for m in self.model.modules():
            if "_apply" in m.__dict__:
                del m.__dict__["_apply"]

    @property
    def adapting(self) -> bool:
        """True while TrainStream is still choosing how to run (the memory plan, which
        blocks' activations are paged or recomputed, the switch to resident mode):
        until it turns False, step times are not yet steady."""
        act = self.activations
        if act is not None and (act.phase != "fixed" or (act.adjustable and not self._act_room_done)):
            return True
        if self.planner is not None and self.pager is not None and not self.planner.done:
            return True
        if self._resident_check_step is not None:
            return True
        return self.config.get("offload") == "auto" and not self.optimizer.resident and self.steps < 2

    # -- resident switch -------------------------------------------------------------
    def _after_step(self) -> None:
        self.steps += 1
        if self._peer_step is not None and self.steps == self._peer_step:
            self._peer_step = None
            self._setup_peer()
        if self.activations is not None:
            packed = self.activations.stats["packed_bytes"]
            self._act_step_bytes, self._act_prev_bytes = packed - self._act_prev_bytes, packed
        if self.planner is not None and self.pager is not None:
            if not self.planner.done:
                self._refine_plan()
                return
            self._watch_plan()
        if self.activations is not None and not self._act_room_done and self.activations.phase == "fixed":
            self._fill_activation_room()
        if self._resident_check_step is not None:
            if self.steps < self._resident_check_step:
                return
            self._resident_check_step = None
        elif self.steps != 2:
            return
        if self.config["offload"] != "auto":
            return
        opt = self.optimizer
        streaming = self.pager is not None and (self.pager.remote or opt._paged_ids)
        if streaming or opt._dp is not None or opt.resident:
            return
        act = self.activations
        if act is not None and (act.recompute or act.offload):
            return  # the measured peak relies on them; the resident switch would drop them
        if self._fits_resident():
            opt._post_step_cb = None
            if self.activations is not None:
                self.activations.remove()
                self.activations = None
            opt._go_resident()
            self.events.append("model fits in VRAM: switched to resident fused AdamW")
            if self.config.get("verbose"):
                print("[trainstream] the model fits in VRAM with a stock optimizer; offloading switched off")

    def _refine_plan(self) -> None:
        dev = self.optimizer.device
        peak = torch.cuda.max_memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)  # per-step peaks while the plan is being tuned
        pager, planner = self.pager, self.planner
        new = planner.refine(peak, set(pager.remote))
        if new is not None:
            for i in sorted(new - pager.remote):  # free VRAM first ...
                pager.page(i)
            for i in sorted(pager.remote - new):  # ... then fill it
                pager.unpage(i)
            self.paged_blocks = sorted(pager.remote)
            self.events.append(f"memory plan: step peak {peak / 2**30:.2f} GiB of {planner.target / 2**30:.2f} GiB "
                               f"target -> {pager.n - len(pager.remote)} of {pager.n} blocks resident in VRAM")
        if planner.done:
            resident = pager.n - len(pager.remote)
            self.events.append(f"memory plan settled: {resident} of {pager.n} blocks resident, {len(pager.remote)} "
                               f"streamed ({pager.bytes_paged / 2**30:.1f} GiB from host RAM per pass)")
            if not pager.paged and self.activations is not None and self.config.get("_act_forced"):
                # Every block fits: activations too, if the measured paged volume
                # fits in the remaining headroom.
                if planner.target - peak > 1.25 * self._act_step_bytes:
                    self.activations.remove()
                    self.activations = None
                    self.events.append(f"activations ({self._act_step_bytes / 2**30:.2f} GiB per step) fit in "
                                       "VRAM: activation paging switched off")
            self._resident_check_step = self.steps + 2  # measure a clean step, then consider resident mode

    def _fill_activation_room(self) -> None:
        """Once the activation policy is final (and the memory plan settled):
        measure one clean step of it, then keep as many of the blocks "auto"
        pages or recomputes in VRAM as the room left under the target holds."""
        act, dev = self.activations, self.optimizer.device
        if not act.adjustable or not (act.recompute or act.offload) or act.block_bytes <= 0:
            self._act_room_done = True
            return
        if self._act_room_step is None:
            torch.cuda.reset_peak_memory_stats(dev)  # the next step runs the final policy from start to end
            self._act_room_step = self.steps + 1
            return
        if self.steps < self._act_room_step:
            return
        self._act_room_step, self._act_room_done = None, True
        peak = torch.cuda.max_memory_allocated(dev)
        target = self.planner.target if self.planner is not None else self.config.get("_vram_target") or 0
        kept = act.keep_resident(int((target - peak) // (1.25 * act.block_bytes)))
        if kept:
            self.events.append(f"activations: step peak {peak / 2**30:.2f} GiB of {target / 2**30:.2f} GiB target "
                               f"leaves room for {len(kept)} more blocks' activations: kept in VRAM "
                               f"({len(act.recompute)} recomputed, {len(act.offload)} paged now)")
            if not (act.recompute or act.offload):
                self._resident_check_step = self.steps + 2  # nothing paged or recomputed: consider resident mode

    def _watch_plan(self) -> None:
        """After the plan settled: if a later step (longer sequences, a bigger
        batch) eats into the safety margin, page one more block. The peak is
        read, not reset, so the user's own memory statistics stay intact."""
        planner, pager = self.planner, self.pager
        peak = torch.cuda.max_memory_allocated(self.optimizer.device)
        alarm = planner.target + 0.5 * (planner.budget.vram_usable - planner.target)
        if peak <= alarm or peak <= self._watch_level + planner.max_block // 2:
            return
        resident = [i for i in range(pager.n) if i not in pager.remote and pager.block_bytes[i]]
        if not resident:
            return
        from .planner import spread
        pick = spread(1, resident)[0]
        pager.page(pick)
        self.paged_blocks = sorted(pager.remote)
        self._watch_level = peak
        self.events.append(f"memory watchdog: peak {peak / 2**30:.2f} GiB is inside the safety margin; block {pick} "
                           f"paged ({pager.n - len(pager.remote)} of {pager.n} resident)")

    def _setup_peer(self) -> None:
        """Collective, at the same step on every local rank: pool the frozen
        blocks every rank streams in the GPUs' VRAM (trainstream.peer)."""
        from . import peer
        local = self.optimizer._dp["local"]
        dev = self.optimizer.device
        host = self.profile.h2d_gbps if self.profile is not None else \
            _C.benchmark_pcie(dev.index or 0, 64 << 20, 4)["h2d_gbps"]
        on, why, _ = peer.probe(local, dev, host)
        if not on:
            self.events.append(f"peer GPUs: off ({why})")
            return
        setting = self.config.get("peer", "auto")
        if setting == "auto":  # half the headroom the memory plan leaves under its target
            target = self.planner.target if self.planner is not None else self.config["_vram_target"]
            budget = int(0.5 * max(0, target - torch.cuda.max_memory_allocated(dev)))
        else:
            budget = int(setting)
        pool = peer.PeerPool.build(local, self.pager, dev, budget)
        if self.pager is not None:
            self.pager.peer = pool
        self.peer = pool
        self.events.append(f"peer GPUs: {why}; {len(pool.pooled_blocks)} streamed blocks pooled across "
                           f"{local.world} GPUs ({pool.bytes_owned / 2**30:.2f} GiB held on this one)")

    def _fits_resident(self) -> bool:
        dev = self.optimizer.device
        trainable = self.optimizer._slot_params
        extra = sum(3 * p.numel() * p.element_size() for p in trainable)  # grads + two moments (param dtype)
        peak = torch.cuda.max_memory_allocated(dev)
        free, total = vram_budget(dev)
        others = max(0, total - free - torch.cuda.memory_reserved(dev))
        return peak + extra <= 0.85 * (total - others)

    # -- .to() / .cuda() / .half() after wrap ---------------------------------------
    def _apply(self, orig, fn, *args, **kwargs):
        if self._applying:
            return orig(fn, *args, **kwargs)
        opt = self.optimizer
        params = opt._slot_params
        p0 = params[0]
        probe = fn(torch.empty(0, device=p0.device, dtype=p0.dtype))
        moving = probe.device != p0.device or probe.dtype != p0.dtype
        self._applying = True
        try:
            if opt.resident:
                out = orig(fn, *args, **kwargs)
                if moving:
                    opt.load_state_dict(opt._resident.state_dict())  # moments follow the params
                return out
            if not moving:
                ptrs = [p.data_ptr() for p in params if id(p) not in opt._paged_ids]
                out = orig(fn, *args, **kwargs)
                after = [p.data_ptr() for p in params if id(p) not in opt._paged_ids]
                if ptrs != after:  # storage replaced (e.g. a memory-format change)
                    self.teardown(materialize=True)
                    opt._rebuild()
                    self.install()
                return out
            # A real move or cast: park the state on the host, let PyTorch convert,
            # then rebuild the engine for the new device / dtype.
            state = opt.state_dict() if opt._suspended is None else opt._suspended
            if self.pager is not None:
                values = {}
                for es in self.pager._entries.values():
                    for p, slot, _ in es:
                        values[id(p)] = (p, self.pager.host_value(p, slot))
                self.teardown(materialize=False)
                for p, host in values.values():
                    p.data = host
            else:
                self.teardown(materialize=False)
            out = orig(fn, *args, **kwargs)
            dev = params[0].device
            opt._suspended = state
            if dev.type == "cuda":
                opt._rebuild(dev)
                self.install()
                self.events.append(f"model moved to {dev}: engine rebuilt")
            else:
                opt.engine = None
                self.events.append(f"model moved to {dev}: optimizer suspended until it returns to a GPU")
            return out
        finally:
            self._applying = False

    def patch_apply(self) -> None:
        for m in self.model.modules():
            orig = m._apply

            def patched(fn, *args, _orig=orig, **kwargs):
                return self._apply(_orig, fn, *args, **kwargs)

            m.__dict__["_apply"] = patched

    # -- telemetry -------------------------------------------------------------------
    def stats(self) -> dict:
        dev = self.optimizer.device
        out = {
            "optimizer": self.optimizer.stats(),
            "transfers": dict(_C.transfer_stats(dev.index or 0)),
            "pinned": dict(_C.pinned_stats()),
            "pinned_by_tag": dict(_C.pinned_tag_bytes()),
            "gpu": {
                "allocated": torch.cuda.memory_allocated(dev),
                "reserved": torch.cuda.memory_reserved(dev),
                "peak_allocated": torch.cuda.max_memory_allocated(dev),
            },
            "resident": self.optimizer.resident,
            "events": list(self.events),
        }
        if self.expert_pagers:
            out["experts"] = dict(stacks=len(self.expert_pagers),
                                  host_bytes=sum(e.nbytes for e in self.expert_pagers),
                                  calls=sum(e.calls for e in self.expert_pagers),
                                  experts_copied=sum(e.experts_used for e in self.expert_pagers),
                                  bytes_copied=sum(e.bytes_in for e in self.expert_pagers))
        if self.attention is not None:
            out["attention"] = dict(mode=self.attention.mode, backend=self.attention.choice,
                                    timings_ms=dict(self.attention.timings))
        if self.activations is not None:
            out["activations"] = dict(self.activations.stats)
            out["activations"]["offloaded_blocks"] = sorted(self.activations.offload)
            out["activations"]["total_blocks"] = self.activations.n
            out["activations"]["calibrating"] = self.activations.calibrating
            out["activations"]["decision"] = dict(self.activations.decision)
            out["activations"]["policy"] = self.activations.policy
            out["activations"]["recompute_blocks"] = sorted(self.activations.recompute)
        if self.pager is not None:
            out["params"] = dict(self.pager.stats, paged_blocks=sorted(self.pager.remote),
                                 paged_bytes=self.pager.bytes_paged, total_blocks=self.pager.n,
                                 quantized_blocks=sorted(self.pager.quant_blocks),
                                 quantized_bytes=self.pager.quantized_bytes, quantize=self.config.get("quantize"),
                                 outliers=self.config.get("outliers", 0.0),
                                 frozen_pinned_bytes=self.pager.frozen_pinned_bytes,
                                 frozen_shared_bytes=self.pager.frozen_shared_bytes,
                                 frozen_pageable_bytes=self.pager.frozen_pageable_bytes,
                                 frozen_disk_bytes=self.pager.frozen_disk_bytes,
                                 peer_blocks=self.peer.pooled_blocks if self.peer is not None else [],
                                 peer_owned_bytes=self.peer.bytes_owned if self.peer is not None else 0)
        if self.planner is not None:
            out["plan"] = dict(target_vram=self.planner.target, settled=self.planner.done,
                               history=list(self.planner.history), budget=self.planner.budget.describe())
        return out


def vram_budget(device) -> tuple:
    """(free, total) bytes this process may use on `device`, honouring a
    torch.cuda.set_per_process_memory_fraction cap."""
    free, total = torch.cuda.mem_get_info(device)
    try:
        frac = torch.cuda.memory.get_per_process_memory_fraction(device)
    except (AttributeError, RuntimeError):
        frac = 1.0
    if frac < 1.0:
        cap = int(total * frac)
        free = min(free, max(0, cap - torch.cuda.memory_reserved(device)))
        total = cap
    return free, total


def _convert_optimizer(optimizer, model: nn.Module, mode: str, max_grad_norm, threads, lr, **extra) -> OffloadAdamW:
    if isinstance(optimizer, OffloadAdamW):
        return optimizer
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return OffloadAdamW(trainable, lr=lr, mode=mode, max_grad_norm=max_grad_norm, threads=threads, **extra)
    if not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        raise TypeError(f"trainstream.wrap supports Adam/AdamW optimizers, got {type(optimizer).__name__}")

    groups = []
    for g in optimizer.param_groups:
        if g.get("amsgrad"):
            raise ValueError("amsgrad=True is not supported by OffloadAdamW")
        if g.get("maximize"):
            raise ValueError("maximize=True is not supported by OffloadAdamW")
        groups.append({"params": list(g["params"]), "lr": float(g["lr"]) if not torch.is_tensor(g["lr"]) else g["lr"].item(),
                       "betas": tuple(g["betas"]), "eps": g["eps"], "weight_decay": g["weight_decay"]})
    if type(optimizer) is torch.optim.Adam and any(g["weight_decay"] for g in groups):
        warnings.warn("torch.optim.Adam applies L2 weight decay; OffloadAdamW uses decoupled (AdamW) decay.")

    new = OffloadAdamW(groups, mode=mode, max_grad_norm=max_grad_norm, threads=threads, **extra)
    if len(optimizer.state):
        new.load_state_dict(optimizer.state_dict())  # migrate moments to host
        optimizer.state.clear()  # release the VRAM they occupied
    return new


def _unique_params(model: nn.Module):
    seen = set()
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            yield p


def _paged_block_indices(param_offload, n_blocks: int, weight_bytes: int, on_cpu: bool, device) -> List[int]:
    """Fixed (non-planned) paging choices: True, False, a fraction, or indices."""
    from .planner import spread
    if n_blocks == 0 or param_offload in (False, 0, 0.0, None, "auto"):
        return []  # "auto" lands here only without a block stack or in gpu mode
    if param_offload is True:
        return list(range(n_blocks))
    if isinstance(param_offload, (list, tuple, set)):
        return sorted(int(i) for i in param_offload if 0 <= int(i) < n_blocks)
    frac = min(max(float(param_offload), 0.0), 1.0)
    return spread(int(round(frac * n_blocks)), list(range(n_blocks)))  # spread for transfer/compute overlap


def _outlier_fraction(value) -> float:
    if value is True or value == "auto":
        return 0.01
    if value is None or value is False:
        return 0.0
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 0.1:
        return float(value)
    raise ValueError(f"outliers must be 'auto', True, False or a fraction between 0 and 0.1, got {value!r}")


def _reserve_host_arena(host_arena, budget, session) -> dict:
    """Page-lock the host memory the run allocates while it trains (paged
    activations, gradient staging, bounce slots) now, instead of during steps."""
    streams = session.pager is not None or session.activations is not None
    if host_arena in (False, None, 0) or (host_arena == "auto" and (not streams or session.optimizer.resident)):
        return dict(reserved=0, cap=0)
    if host_arena == "auto":
        cap = max(0, budget.pin_budget - _C.pinned_stats()["live_bytes"])
        now = min(cap, 1 << 30)
    else:
        cap = now = int(host_arena)
    if cap <= 0:
        session.events.append("pinned host arena: none (the page-lock budget is already used)")
        return dict(reserved=0, cap=0)
    reserved = _C.pinned_reserve(now, 0, cap)
    session.events.append(f"pinned host arena: {reserved / 2**30:.2f} GiB page-locked up front, grows on demand "
                          f"to {cap / 2**30:.1f} GiB")
    return dict(reserved=int(reserved), cap=int(cap))


def _broadcast_from_rank0(model: nn.Module, group) -> None:
    import torch.distributed as dist
    with torch.no_grad():
        for t in list(_unique_params(model)) + [b for b in model.buffers()]:
            host = t.detach().to("cpu", copy=True)
            dist.broadcast(host, src=0, group=group)
            t.copy_(host)


def wrap(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    optimizer_mode: str = "cpu",
    offload: Union[str, bool] = "auto",
    activation_offload: Union[str, float, bool] = "auto",
    param_offload: Union[str, float, bool, Sequence[int]] = "auto",
    blocks: Optional[Sequence[nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
    moment_dtype: str = "auto",
    distributed: Optional[bool] = None,
    guard_params: bool = True,
    max_grad_norm: Optional[float] = None,
    threads: Optional[int] = None,
    min_offload_bytes: int = 1 << 20,
    prefetch_depth: int = 1,
    accumulation_steps: int = 1,
    max_activation_slowdown: float = 0.15,
    max_inflight_bytes: Optional[int] = None,
    profile: bool = True,
    lr: float = 1e-3,
    vram_headroom: float = 0.10,
    check_memory: bool = True,
    quantize: Optional[str] = "auto",
    quant_block_size: int = 64,
    keep_precise: Union[str, float, bool, None] = 0,
    error_rank: int = 0,
    error_iters: int = 1,
    double_quant: bool = False,
    prefer: str = "exact",
    recompute: Union[str, float, bool, Sequence[int]] = "auto",
    attention: Optional[str] = "auto",
    outliers: Union[str, float, bool, None] = 0.0,
    nvfp4_backward: bool = False,
    kernels: Union[str, bool, None] = "auto",
    fused_lora: Union[str, bool, None] = "auto",
    host_arena: Union[str, int, bool] = "auto",
    disk_offload: Optional[str] = None,
    moe_paging: bool = False,
    peer: Union[str, int, bool] = "auto",
    verbose: bool = False,
):
    """Enable TrainStream offloading for `model` and return ``(model, optimizer)``.

    Args:
        optimizer: torch.optim.AdamW / Adam (converted, state migrated), an
            OffloadAdamW, or None (a default OffloadAdamW is created with `lr`).
        optimizer_mode: "cpu", "gpu" or "auto" (decided from the profile).
        offload: "auto" measures the first two steps and, if a stock optimizer
            would fit in VRAM, switches to torch.optim.AdamW(fused=True) — so
            small models are never slowed down. True always offloads.
        activation_offload: "auto" (default) measures them in the first step:
            kept in VRAM if they fit under the VRAM target, otherwise paged to
            host RAM as far as it holds them (see recompute for the rest). A
            fraction in [0, 1], True (all but the last block) or False fix it.
        param_offload: page transformer-block weights to host RAM, streaming
            each block in just before it runs. "auto" (default) plans from the
            VRAM this process may use: as many blocks as fit stay resident, the
            rest are paged (spread through the stack), and the first steps'
            measured peaks move blocks in or out until it settles. True pages
            every block; False none; a fraction or block indices fix the split.
            Needs optimizer_mode="cpu".
        blocks: the repeated layer stack; auto-detected when None.
        device: target GPU when `model` is still on the CPU. With parameter
            paging, block weights then go straight to host RAM and never
            occupy VRAM all at once — models larger than the GPU can train.
        moment_dtype: "auto" (fp32; bf16 when fp32 moments would not fit in the
            available RAM; 8-bit when bf16 would not either and the optimizer
            runs on the host), "float32", "bfloat16" (stochastic rounding; half
            the host memory and traffic for the moments) or "int8" (one byte
            per value with an fp32 scale per 256, stochastic rounding; a
            quarter; cpu optimizer_mode only).
        distributed: data parallel over torch.distributed (default: on when a
            process group with more than one rank is initialized). Parameters
            are broadcast from rank 0; gradients are averaged on the host.
        guard_params: make reads of a weight whose update is still in flight
            wait for it, wherever they happen (default on).
        max_grad_norm: global-norm clipping applied inside the optimizer
            (torch.nn.utils.clip_grad_norm_ cannot see streamed gradients).
        accumulation_steps: backward passes per optimizer.step(). In cpu mode
            without clipping, updates run eagerly during the last backward.
        max_activation_slowdown: without a VRAM target to measure against (an
            ActivationOffloader used on its own), "auto" activation paging backs
            off until the measured step time is within this fraction of the
            un-paged baseline. wrap() always has a target: activations that fit
            under it stay in VRAM instead.
        quantize: frozen weights only (a LoRA base; trained weights are never
            quantized). "auto" (default) quantizes only when the run cannot fit
            otherwise, trying "int8" before "nf4"; "none" never; "int8" / "nf4"
            always. Weights quantized beforehand (quantize_model, load_model)
            are used as they are.
        quant_block_size: values per fp32 scale (even; default 64).
        keep_precise: with quantize="nf4" / "nvfp4", the weights that lose the
            most in that format (relative error, measured weight by weight) stay
            one step more precise, int8 / fp8, as long as the extra bytes stay
            within this fraction of the quantized weights' size. 0 (default)
            none; "auto" or True 0.05; a fraction up to 0.5.
        error_rank: LoftQ. When > 0, quantizing a weight also fits a rank-
            error_rank approximation B A of what quantization lost (randomized
            SVD; error_iters rounds alternate it with quantizing W - B A), and
            the model's PEFT LoRA adapters on those weights start at it instead
            of at zero, so base + adapter starts close to the original weights.
            Weights quantized beforehand with error_rank (load_model,
            quantize_model) start their adapters the same way. Default 0: off.
        double_quant: nf4 / int8 block scales stored as one-byte log codes with
            an fp32 scale per 256 blocks (bitsandbytes' double quantization),
            about 0.37 bit per weight less at block size 64 (0.65 GB on a 14B
            model); the values are quantized against the decoded scales.
        prefer: "exact" (default) or "speed". With "speed", quantize="auto" also
            picks NF4 when the bf16 weights would have to stream from RAM but the
            4-bit ones fit in VRAM — usually faster, no longer bit-exact.
        recompute: run a block's forward again in backward instead of keeping
            or paging its activations. "auto" (default) does it only when the
            activations cannot all stay in VRAM (weights paged, or too large),
            for as many blocks as the measured trials in the first steps show
            is fastest; blocks go back to VRAM afterwards as far as the room
            left under the target allows. True: every block but the last;
            False: never; a fraction or block indices fix it. Off when the
            model already uses gradient checkpointing (Hugging Face).
        attention: scaled_dot_product_attention backend. "auto" (default)
            times the backends once on the model's head shape and runs the
            fastest first (others stay as fallbacks); "cudnn", "efficient",
            "flash" or "math" put that one first; None leaves PyTorch's order.
        outliers: with quantize="nvfp4" on tensor cores, the fraction of input
            channels (the largest, picked per call) multiplied in bf16 instead
            of being quantized to 4 bits with the rest of the activation.
            0 (default) none; "auto" or True 0.01; a float up to 0.1.
        nvfp4_backward: with quantize="nvfp4" on tensor cores, the input
            gradient of each frozen matmul is computed in nvfp4 too (the
            gradient and the transposed weight quantized along the reduced
            dimension, round to nearest), instead of in bf16 against the
            expanded weight. Faster matmuls in backward, more rounding: try it
            with benchmarks/quality.py (format nvfp4-b). Default False.
        kernels: fused CUDA kernels for each block's RMSNorm, rotary embedding
            and gated MLP activation (Hugging Face models; each is checked
            against the model's own code once and used only where they agree).
            They compute in fp32 and round once, so results differ from
            PyTorch's in the last bits. "auto" (default): on when the frozen
            weights are quantized (already not bit-exact), off otherwise; True:
            always; False: never.
        fused_lora: PEFT LoRA adapters (plain ones: no dropout, DoRA or bias)
            computed by one autograd function: two matmuls, the second
            accumulating into the base layer's output, and three in backward,
            keeping the input the base layer read instead of a copy. Matches
            PEFT up to the rounding of its intermediate products. "auto"
            (default): on when the frozen weights are quantized; True / False.
        host_arena: page-locked host memory reserved up front, from which
            paged activations, gradient staging and other pinned buffers are
            carved, so training steps never page-lock memory (a stall on
            Windows). "auto" (default): when the run streams anything, reserve
            one slab now and grow on demand within the page-lock budget; an int
            reserves that many bytes now (and caps growth there); False: none.
        disk_offload: a directory (on a fast SSD). Frozen paged weights the
            page-lock budget can't hold go to memory-mapped files there instead
            of ordinary RAM: the OS keeps them in its page cache while RAM has
            room and reads them from the disk when it doesn't, and the RAM
            check no longer counts them. Weights load_model(disk_offload=...)
            already put on disk stay there. Default None: RAM only.
        moe_paging: the frozen experts of mixture-of-experts layers (Hugging
            Face's stacked experts: Qwen3-MoE, Qwen2-MoE, Mixtral...) stay in
            pinned host RAM, and each layer copies in only the experts its
            tokens were routed to, again in backward instead of keeping them.
            Pays when a step routes to few experts (short sequences, small
            batches); with thousands of tokens nearly all are hit. Default False.
        peer: data parallel on one machine whose GPUs reach each other's
            memory (NVLink / PCIe P2P, through CUDA IPC): the frozen blocks every
            rank streams are split among the GPUs' VRAM and copied GPU to GPU
            instead of from host RAM. Set up collectively at step 10. "auto"
            (default): when the GPUs allow it and peer copies are at least 0.9x
            the host link, each GPU contributes half the headroom under the
            memory plan's target; an int: that many bytes per GPU; False: off.
            With it active, unwrap() is collective too.
        vram_headroom: fraction of the usable VRAM the planner leaves free.
        check_memory: raise MemoryError at wrap() when the host state cannot
            fit in the available RAM, instead of failing later.
        max_inflight_bytes: cap on VRAM held by activations (and, separately,
            by gradients) still in flight to the host; the forward / backward
            pass waits for the link beyond it. Default: 1 GiB, or 10% of the
            GPU's memory budget if that is smaller.
    """
    if unwrap_session(model) is not None:
        raise RuntimeError("model is already wrapped; call trainstream.unwrap(model) first")
    if offload not in ("auto", True):
        raise ValueError("offload must be 'auto' or True (to train without TrainStream, don't wrap the model)")
    if quantize in (None, False):
        quantize = "none"
    if quantize not in ("auto", "none") + Q.KINDS:
        raise ValueError(f"quantize must be 'auto', 'none', 'int8', 'nf4', 'fp8' or 'nvfp4', got {quantize!r}")
    if prefer not in ("exact", "speed"):
        raise ValueError(f"prefer must be 'exact' or 'speed', got {prefer!r}")
    if not (recompute in ("auto", True, False, None) or isinstance(recompute, (float, int, list, tuple, set))):
        raise ValueError(f"recompute must be 'auto', True, False, a fraction or block indices, got {recompute!r}")
    if getattr(model, "is_gradient_checkpointing", False):
        recompute = False  # the model recomputes its own layers already
    if attention is not None and attention not in ATTENTION_CHOICES:
        raise ValueError(f"attention must be one of {ATTENTION_CHOICES} or None, got {attention!r}")
    outliers = _outlier_fraction(outliers)
    keep_frac = Q.keep_fraction(keep_precise)
    if kernels not in ("auto", True, False, None):
        raise ValueError(f"kernels must be 'auto', True or False, got {kernels!r}")
    if fused_lora not in ("auto", True, False, None):
        raise ValueError(f"fused_lora must be 'auto', True or False, got {fused_lora!r}")
    if not (host_arena in ("auto", False, None) or (isinstance(host_arena, int) and not isinstance(host_arena, bool)
                                                    and host_arena >= 0)):
        raise ValueError(f"host_arena must be 'auto', False or a number of bytes, got {host_arena!r}")
    if not (peer in ("auto", False, None) or (isinstance(peer, int) and not isinstance(peer, bool) and peer >= 0)):
        raise ValueError(f"peer must be 'auto', False or a number of bytes per GPU, got {peer!r}")
    params = list(_unique_params(model))
    if not params:
        raise ValueError("trainstream.wrap needs a model with parameters")
    on_cpu = any(p.device.type in ("cpu", "meta") for p in params)
    if on_cpu:
        target = torch.device(device or "cuda")
        if target.type != "cuda":
            raise ValueError("trainstream.wrap trains on a CUDA device; pass device='cuda[:N]'")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
    else:
        target = params[0].device
        if target.type != "cuda":
            raise ValueError("trainstream.wrap expects the model on a CUDA device (or on the CPU with device=...)")
        want = torch.device(device) if device is not None else target
        if want.type == "cuda" and want.index is None:
            want = torch.device("cuda", torch.cuda.current_device())
        if want != target:
            raise ValueError(f"model is on {target} but device={device!r}; move it first or leave device unset")
    expert_pagers = experts.install(model, target) if moe_paging else []  # before anything places or pages them
    if expert_pagers:
        params = list(_unique_params(model))

    if distributed is None:
        import torch.distributed as dist
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if distributed:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed=True needs torch.distributed.init_process_group() first")
        if optimizer_mode == "gpu":
            raise ValueError("distributed training averages gradients on the host: use optimizer_mode='cpu'")
        _broadcast_from_rank0(model, dist.new_group(backend="gloo"))

    stack = list(blocks) if blocks is not None else find_blocks(model)
    weight_bytes = sum(p.numel() * p.element_size() for p in params)
    candidates = pageable_params(model, stack, range(len(stack))) if stack else {}
    block_bytes = [sum(p.numel() * p.element_size() for p in candidates[i]) for i in range(len(stack))]
    budget = measure_budget(target)
    if max_inflight_bytes is None:
        max_inflight_bytes = int(min(1 << 30, 0.10 * budget.vram_total))

    # Quantization of frozen block weights: decided before the VRAM plan.
    qparams = [p for i in range(len(stack)) for p in candidates.get(i, [])
               if Q.quantized_of(p) is not None or Q.eligible(p)]
    pre = sorted({Q.quantized_of(p).kind for p in qparams if Q.quantized_of(p) is not None})
    trainable_params = [p for p in params if p.requires_grad]
    grad_acc = bool(max_grad_norm) or accumulation_steps > 1 or bool(distributed)
    have = budget.ram_available + (sum(p.untyped_storage().nbytes() for p in params) if on_cpu else 0)
    have = int(0.90 * have)  # room for paged activations and the rest of the process
    frozen_eligible = sum(p.numel() * p.element_size() for p in qparams if Q.quantized_of(p) is None)
    qwhy = "requested"
    if pre:
        qwhy = "already quantized"
        if quantize == "none":
            raise ValueError("the model already holds quantized weights; use quantize='auto' or their kind")
        qkind = pre[-1]
    elif quantize in Q.KINDS:
        if not qparams:
            raise ValueError("quantize applies to frozen weights (e.g. a LoRA base) inside the block stack, "
                             "and this model has none; trained weights are never quantized")
        qkind = quantize
    elif quantize == "auto" and qparams and optimizer_mode != "gpu":
        qkind = None
        bf16_need = host_bytes_needed(trainable_params, frozen_eligible, moment_bytes=2, grad_acc=grad_acc)
        if bf16_need > have:  # bf16 cannot be held even with bf16 moments: try int8, then nf4
            qwhy = "bf16 frozen weights do not fit in RAM"
            for kind in ("int8", "nf4"):
                packed = sum(Q.packed_bytes(p.numel(), kind, quant_block_size) for p in qparams)
                if host_bytes_needed(trainable_params, packed, moment_bytes=2, grad_acc=grad_acc) <= have:
                    qkind = kind
                    break
            if qkind is None:
                qkind = "nf4"  # the RAM check below reports the shortfall
        elif prefer == "speed" and param_offload == "auto" and stack:
            fixed0 = (weight_bytes - sum(block_bytes)) + max_inflight_bytes + max(96 << 20, int(0.02 * budget.vram_total))
            bf16_plan = MemoryPlanner(budget, block_bytes, fixed_vram=fixed0, inflight_bytes=max_inflight_bytes,
                                      margin=vram_headroom)
            if bf16_plan.initial_paged():
                # NF4: NVFP4 compute is faster on Blackwell but also quantizes activations
                # (perplexity +11% vs +6% for NF4 on Qwen2.5-1.5B, benchmarks/quality.py),
                # so it stays an explicit choice
                speed_kind = "nf4"
                qset = {id(p) for p in qparams if Q.eligible(p, speed_kind)}
                near = [sum(Q.packed_bytes(p.numel(), speed_kind, quant_block_size, p.shape) if id(p) in qset
                            else p.numel() * p.element_size() for p in candidates[i]) for i in range(len(stack))]
                q4_plan = MemoryPlanner(budget, near, fixed_vram=fixed0, inflight_bytes=max_inflight_bytes,
                                        margin=vram_headroom, window_block_bytes=max(block_bytes))
                if not q4_plan.initial_paged():
                    qkind = speed_kind
                    qwhy = "prefer='speed': 4-bit weights fit in VRAM, bf16 would stream from RAM"
    else:
        qkind = None
    kept_note = None
    if qkind is not None:
        from .optimizer import physical_ram_bytes
        pin_budget = int(0.40 * (physical_ram_bytes() or 1 << 40))
        # block-scaled formats take only tensor-core shapes; other weights stay as they are
        qparams = [p for p in qparams if Q.quantized_of(p) is not None or Q.eligible(p, qkind)]
        todo = [p for p in qparams if Q.quantized_of(p) is None]
        keep = set()
        if keep_frac and qkind in Q.KEEP_KIND and todo:  # measure every weight's loss first, then choose
            keep = Q.choose_precise([(id(p), Q.quant_error(p, qkind, quant_block_size), p.numel(), tuple(p.shape))
                                     for p in todo], qkind, keep_frac, quant_block_size)
        for p in todo:  # one tensor at a time: the model never exists twice
            kind = Q.KEEP_KIND[qkind] if id(p) in keep else qkind
            nb = Q.packed_bytes(p.numel(), kind, quant_block_size, p.shape)
            Q.attach(p, Q.make_quantized(p, kind, quant_block_size,
                                         pin=_C.pinned_stats()["live_bytes"] + nb <= pin_budget,
                                         error_rank=max(0, int(error_rank)), error_iters=max(1, int(error_iters)),
                                         double_quant=bool(double_quant)))
        if any(Q.quantized_of(p).lora_init is not None for p in qparams):
            n = Q.init_lora_from_error(model)  # before the optimizer copies the adapters to the host
            kept_note = ((kept_note + "; ") if kept_note else "") + (
                f"LoftQ: {n} LoRA adapters start at their base weights' quantization error")
        if keep:
            kept_note = (f"kept the {len(keep)} of {len(todo)} weights that lose the most in {qkind} as "
                         f"{Q.KEEP_KIND[qkind]} (keep_precise={keep_frac:g})")
    qset = {id(p) for p in qparams} if qkind is not None else set()
    near_bytes = [sum(Q.quantized_of(p).nbytes if id(p) in qset else p.numel() * p.element_size()
                      for p in candidates.get(i, [])) for i in range(len(stack))]

    planner = None
    if param_offload == "auto" and stack and optimizer_mode != "gpu":
        buffers = sum(b.numel() * b.element_size() for b in model.buffers())
        workspace = max(96 << 20, int(0.02 * budget.vram_total))  # cuBLAS/cuDNN workspaces, allocator slack
        fixed = (weight_bytes - sum(block_bytes)) + buffers + max_inflight_bytes + workspace
        planner = MemoryPlanner(budget, near_bytes, fixed_vram=fixed, inflight_bytes=max_inflight_bytes,
                                margin=vram_headroom, window_block_bytes=max(block_bytes) if block_bytes else None)
        paged_blocks = sorted(planner.initial_paged())
        if not paged_blocks:
            planner = None  # everything fits: no paging machinery at all (quantized blocks stay near)
        elif check_memory and not planner.fits_at_all():
            raise MemoryError(
                f"TrainStream: even with every block paged, this model needs about "
                f"{(fixed + 2 * planner.max_block) / 2**30:.1f} GiB of VRAM ({budget.describe()}). The parts "
                f"outside the block stack (embeddings, output layer) stay resident. Use a smaller batch or model, "
                f"or pass check_memory=False to try anyway.")
    else:
        paged_blocks = _paged_block_indices(param_offload, len(stack), weight_bytes, on_cpu, target)

    # Host RAM: decide the moment precision and refuse what cannot fit.
    paged_set = set(paged_blocks)
    frozen_paged = sum(p.numel() * p.element_size() for i in paged_set for p in candidates.get(i, [])
                       if not p.requires_grad and id(p) not in qset)
    frozen_paged += sum(Q.quantized_of(p).nbytes for p in qparams if id(p) in qset)  # packed host copies
    disk = None
    if disk_offload:
        from .disk import DiskStore
        from .optimizer import physical_ram_bytes
        disk = DiskStore(str(disk_offload))
        frozen_paged = min(frozen_paged, int(0.40 * (physical_ram_bytes() or 1 << 40)))  # the rest is on disk
    if distributed:
        from . import shared
        if shared.enabled():
            import os as _os
            import torch.distributed as _dist
            if int(_os.environ.get("LOCAL_WORLD_SIZE", _dist.get_world_size())) == _dist.get_world_size():
                # one shared copy for the machine: each rank's share of the RAM pays a part of it
                frozen_paged //= _dist.get_world_size()
    need32 = host_bytes_needed(trainable_params, frozen_paged, moment_bytes=4, grad_acc=grad_acc)
    need16 = host_bytes_needed(trainable_params, frozen_paged, moment_bytes=2, grad_acc=grad_acc)
    need8 = host_bytes_needed(trainable_params, frozen_paged, moment_bytes=1, grad_acc=grad_acc)
    if moment_dtype == "auto":  # the most precise moments that fit; 8-bit only for the host optimizer
        moment_dtype = ("float32" if need32 <= have else "bfloat16" if need16 <= have or optimizer_mode == "gpu"
                        else "int8")
    need = (need16 if moment_dtype in ("bfloat16", "bf16", torch.bfloat16)
            else need8 if moment_dtype in ("int8", "8bit") else need32)
    if check_memory and budget.ram_available and need > have:
        raise MemoryError(explain_ram_shortfall(need, have, budget, trainable=sum(p.numel() for p in trainable_params),
                                                frozen=frozen_paged // 2))
    if paged_blocks:
        if optimizer_mode == "gpu":
            raise ValueError("param_offload needs optimizer_mode='cpu' (paged weights are updated on the host)")
        optimizer_mode = "cpu"
    if moment_dtype in ("int8", "8bit"):
        if optimizer_mode == "gpu":
            raise ValueError("moment_dtype='int8' needs optimizer_mode='cpu' (8-bit moments are updated on the host)")
        optimizer_mode = "cpu"

    host_init: Dict[int, torch.Tensor] = {}
    frozen_init: Dict[int, torch.Tensor] = {}
    # Trained parameters of quantized blocks (adapters on a quantized base) stay
    # in VRAM; only the other paged blocks put their trained weights on the host.
    quant_idx = {i for i in range(len(stack)) if any(id(p) in qset for p in candidates.get(i, []))}
    paged_plain = [i for i in paged_blocks if i not in quant_idx]
    if on_cpu:
        # Move everything except the paged block weights; those go to host RAM
        # and get a VRAM tensor with empty storage, filled block by block later.
        paged_ids = {id(p) for ps in pageable_params(model, stack, paged_plain).values() for p in ps} | qset
        with torch.no_grad():
            for m in model.modules():
                for name, b in list(m._buffers.items()):
                    if b is not None:
                        m._buffers[name] = b.to(target)
            for p in params:
                if id(p) in qset:  # quantized: its packed host copy is the source
                    shape, dt = p.shape, p.dtype
                    p.data = torch.empty(shape, dtype=dt, device=target)
                    p.untyped_storage().resize_(0)
                elif id(p) in paged_ids:
                    cpu = p.data
                    (host_init if p.requires_grad else frozen_init)[id(p)] = cpu
                    p.data = torch.empty(cpu.shape, dtype=cpu.dtype, device=target)
                    p.untyped_storage().resize_(0)
                else:
                    p.data = p.data.to(target)
        params = list(_unique_params(model))

    need_profile = profile and (activation_offload == "auto" or optimizer_mode == "auto")
    hw = profile_hardware(target) if need_profile else None
    if optimizer_mode == "auto":
        numel = sum(p.numel() for p in params)
        optimizer_mode = hw.recommend_optimizer_mode(numel) if hw else "cpu"

    paged_trainable = set(host_init)
    if not on_cpu:
        paged_trainable = {id(p) for ps in pageable_params(model, stack, paged_plain).values()
                           for p in ps if p.requires_grad}
    opt = _convert_optimizer(optimizer, model, optimizer_mode, max_grad_norm, threads, lr,
                             moment_dtype=moment_dtype, distributed=bool(distributed), _broadcast=False,
                             _paged=paged_trainable, _host_init=host_init)
    host_init.clear()
    if max_grad_norm is not None:
        opt.max_grad_norm = max_grad_norm
    opt.accumulation_steps = max(1, int(accumulation_steps))
    if opt.engine is not None and opt.mode == "cpu":
        opt.engine.set_max_inflight_grad_bytes(int(max_inflight_bytes))
        # Pooled gradient staging: a quarter of all gradient bytes (at least
        # 1 GiB) in flight between the GPU and the host optimizer threads.
        grad_bytes = sum(p.numel() * p.element_size() for p in opt._slot_params)
        opt.engine.set_max_staging_bytes(int(max(1 << 30, grad_bytes // 4)))

    # PCIe bytes the optimizer itself moves per step, reserved when sizing
    # activation paging: cpu mode streams grads out and weights back in;
    # gpu mode reads+writes both fp32 moments through zero-copy.
    numel = sum(p.numel() for p in params)
    pbytes = params[0].element_size()
    reserved = (numel * pbytes, numel * pbytes) if optimizer_mode == "cpu" else (numel * 8, numel * 8)
    if paged_blocks:
        paged_bytes = sum(p.numel() * p.element_size()
                          for ps in pageable_params(model, stack, paged_blocks).values() for p in ps)
        reserved = (reserved[0], reserved[1] + paged_bytes)  # each paged weight is loaded twice per step
    act_forced = False
    if paged_blocks and activation_offload == "auto":
        # A model that needs its weights paged is memory-bound: page every
        # block's activations from the first step instead of calibrating with
        # all of them resident (which is exactly what would not fit). The
        # planner switches it off again if everything turns out to fit.
        activation_offload = True
        act_forced = True
    if activation_offload not in (False, 0, 0.0, None) and not stack and verbose:
        print("[trainstream] no repeated block stack found; activation offload disabled")

    config = dict(quantize=qkind, quant_block_size=quant_block_size, prefer=prefer, recompute=recompute,
                  attention=attention, outliers=outliers if qkind == "nvfp4" else 0.0,
                  nvfp4_backward=bool(nvfp4_backward) and qkind == "nvfp4",
                  kernels=kernels is True or (kernels == "auto" and qkind is not None),
                  fused_lora=fused_lora is True or (fused_lora == "auto" and qkind is not None),
                  optimizer_mode=optimizer_mode, offload=offload, activation_offload=activation_offload,
                  param_offload=param_offload, paged_blocks=len(paged_blocks), moment_dtype=opt.moment_dtype,
                  distributed=bool(distributed), accumulation_steps=opt.accumulation_steps,
                  eager_updates=opt.eager_active, max_grad_norm=max_grad_norm, prefetch_depth=prefetch_depth,
                  min_offload_bytes=min_offload_bytes, guard_params=guard_params,
                  max_activation_slowdown=max_activation_slowdown, max_inflight_bytes=max_inflight_bytes,
                  verbose=verbose, _reserved=reserved, _act_forced=act_forced, _disk=disk,
                  _vram_target=planner.target if planner is not None else int(budget.vram_usable * (1 - vram_headroom)),
                  plan=dict(budget=budget.describe(), host_bytes=need, blocks=len(stack),
                            initial_paged=len(paged_blocks), auto=planner is not None))
    session = TrainStreamSession(model, opt, hw, config, stack, paged_blocks)
    session.expert_pagers = expert_pagers
    if expert_pagers:
        session.events.append(f"MoE: {len(expert_pagers)} stacks of experts "
                              f"({sum(e.nbytes for e in expert_pagers) / 2**30:.2f} GiB) stay in host RAM; each layer "
                              f"copies in only the experts its tokens are routed to")
    session.planner = planner
    session.candidates = candidates
    session.install(frozen_init=frozen_init)
    if qkind is not None:
        qbytes = sum(Q.quantized_of(p).nbytes for p in qparams)
        session.events.append(f"quantized {len(qparams)} frozen weights to {qkind} ({qbytes / 2**30:.2f} GiB; "
                              f"{qwhy}); training uses their dequantized values and is no longer bit-exact "
                              f"with the bf16 model")
        if kept_note:
            session.events.append(kept_note)
        if qkind in Q.BLOCKSCALED_KINDS:
            cap = torch.cuda.get_device_capability(target)
            if session.lowp_modules:
                kept = (f", except the largest {config['outliers']:.1%} of input channels, multiplied in bf16"
                        if config["outliers"] else "")
                session.events.append(f"{len(session.lowp_modules)} {qkind} matmuls run on tensor cores "
                                      f"(sm_{cap[0]}{cap[1]}); activations are quantized to {qkind} on the fly"
                                      + kept)
            else:
                session.events.append(f"this GPU (sm_{cap[0]}{cap[1]}) has no {qkind} tensor-core matmul: "
                                      f"weights are expanded before use, like nf4 / int8")
    if session.fused is not None:
        session.events.append(f"fused kernels: {session.fused.describe()}")
    if session.lora_layers:
        session.events.append(f"fused LoRA: {len(session.lora_layers)} adapted Linears computed in one function "
                              f"each (two matmuls forward, three backward)")
    if planner is not None:
        session.events.append(f"memory plan: {budget.describe()}; starting with {len(stack) - len(paged_blocks)} "
                              f"of {len(stack)} blocks resident, {len(paged_blocks)} paged; "
                              f"moments {opt.moment_dtype}")
    config["host_arena"] = _reserve_host_arena(host_arena, budget, session)
    config["disk_offload"] = disk.dir if disk is not None else None
    if session.pager is not None and session.pager.frozen_disk_bytes:
        session.events.append(f"disk tier: {session.pager.frozen_disk_bytes / 2**30:.2f} GiB of frozen weights in "
                              f"memory-mapped files under {disk_offload}")
    config["peer"] = peer if peer not in (None, 0) else False
    if config["peer"] is not False and opt._dp is not None and opt._dp.get("local") is not None:
        session._peer_step = PEER_SETUP_STEP  # the same step on every rank: the setup is collective
    frozen_init.clear()
    config["eager_updates"] = opt.eager_active
    opt._post_step_cb = session._after_step
    session.patch_apply()
    object.__setattr__(model, _ATTR, session)
    torch.cuda.reset_peak_memory_stats(target)
    if verbose:
        if hw is not None:
            print(hw.summary())
        print(f"[trainstream] {opt!r}; activation blocks: {session.activations.n if session.activations else 0}; "
              f"paged weight blocks: {len(paged_blocks)}")
    return model, opt


def unwrap_session(model: nn.Module) -> Optional[TrainStreamSession]:
    return model.__dict__.get(_ATTR)


def unwrap(model: nn.Module) -> nn.Module:
    """Remove every hook wrap() installed. Parameters keep their trained values
    (paged weights are loaded back into VRAM)."""
    session = unwrap_session(model)
    if session is not None:
        session.remove()
        del model.__dict__[_ATTR]
    return model


def memory_report(model: Optional[nn.Module] = None) -> dict:
    """GPU + pinned-host memory snapshot (with TrainStream stats when wrapped)."""
    session = unwrap_session(model) if model is not None else None
    if session is not None:
        return session.stats()
    return {
        "pinned": dict(_C.pinned_stats()),
        "pinned_by_tag": dict(_C.pinned_tag_bytes()),
        "gpu": {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated()},
    }
