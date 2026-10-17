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
from .profiler import HardwareProfile, profile_hardware

_ATTR = "_trainstream_session"


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
        self.steps = 0
        self.events: List[str] = []
        self._applying = False
        self._orig_apply: Dict[int, object] = {}

    # -- install / teardown ----------------------------------------------------------
    def install(self, frozen_init=None) -> None:
        cfg = self.config
        opt = self.optimizer
        opt.attach(self.model, guard=cfg["guard_params"])
        if opt.resident:
            return
        if self.paged_blocks:
            paged = pageable_params(self.model, self.blocks, self.paged_blocks)
            for ps in paged.values():
                for p in ps:
                    slot = opt._slot_of.get(id(p))
                    if slot is not None:
                        opt._paged_ids.add(id(p))
                        opt.engine.set_paged(slot, True)
            self.pager = ParamPager(self.model, self.blocks, opt, paged, frozen_init=frozen_init,
                                    prefetch_depth=cfg["prefetch_depth"])
        if cfg["activation_offload"] not in (False, 0, 0.0, None) and self.blocks:
            act = cfg["activation_offload"] if not cfg.get("_act_fixed") else cfg["_act_fixed"]
            self.activations = ActivationOffloader(
                self.model, self.blocks, act, min_bytes=cfg["min_offload_bytes"],
                prefetch_depth=cfg["prefetch_depth"], profile=self.profile,
                max_slowdown=cfg["max_activation_slowdown"], reserved_link_bytes=cfg["_reserved"],
                max_inflight_bytes=cfg["max_inflight_bytes"])

    def teardown(self, materialize: bool = True) -> None:
        self.optimizer.detach()
        if self.activations is not None:
            self.activations.remove()
            self.activations = None
        if self.pager is not None:
            self.pager.remove(materialize=materialize)
            self.pager = None

    def remove(self) -> None:
        self.teardown(materialize=True)
        for m in self.model.modules():
            if "_apply" in m.__dict__:
                del m.__dict__["_apply"]

    # -- resident switch -------------------------------------------------------------
    def _after_step(self) -> None:
        self.steps += 1
        if self.steps != 2 or self.config["offload"] != "auto":
            return
        opt = self.optimizer
        if self.pager is not None or opt._dp is not None or opt.resident:
            return
        if self._fits_resident():
            opt._post_step_cb = None
            if self.activations is not None:
                self.activations.remove()
                self.activations = None
            opt._go_resident()
            self.events.append("model fits in VRAM: switched to resident fused AdamW")
            if self.config.get("verbose"):
                print("[trainstream] the model fits in VRAM with a stock optimizer; offloading switched off")

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
        if self.activations is not None:
            out["activations"] = dict(self.activations.stats)
            out["activations"]["offloaded_blocks"] = sorted(self.activations.offload)
            out["activations"]["total_blocks"] = self.activations.n
            out["activations"]["calibrating"] = self.activations.calibrating
            out["activations"]["decision"] = dict(self.activations.decision)
        if self.pager is not None:
            out["params"] = dict(self.pager.stats, paged_blocks=sorted(self.pager.paged),
                                 paged_bytes=self.pager.bytes_paged, total_blocks=self.pager.n,
                                 frozen_pinned_bytes=self.pager.frozen_pinned_bytes,
                                 frozen_pageable_bytes=self.pager.frozen_pageable_bytes)
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
    if n_blocks == 0 or param_offload in (False, 0, 0.0, None):
        return []
    if param_offload == "auto":
        free, total = vram_budget(device)
        # Page when the weights alone would take more than half of the GPU.
        need = weight_bytes > 0.5 * (free if on_cpu else total)
        return list(range(n_blocks)) if need else []
    if param_offload is True:
        return list(range(n_blocks))
    if isinstance(param_offload, (list, tuple, set)):
        return sorted(int(i) for i in param_offload if 0 <= int(i) < n_blocks)
    frac = min(max(float(param_offload), 0.0), 1.0)
    return list(range(int(round(frac * n_blocks))))


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
    moment_dtype: str = "float32",
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
        activation_offload: "auto" (calibrate, then size from bandwidth), a
            fraction in [0, 1], True (all but the last block) or False.
        param_offload: page transformer-block weights to host RAM, streaming
            each block in just before it runs. "auto" pages when the weights
            would take more than half of the GPU; True, False, a fraction of
            the blocks, or explicit block indices. Needs optimizer_mode="cpu".
        blocks: the repeated layer stack; auto-detected when None.
        device: target GPU when `model` is still on the CPU. With parameter
            paging, block weights then go straight to host RAM and never
            occupy VRAM all at once — models larger than the GPU can train.
        moment_dtype: "float32" or "bfloat16" (stochastic rounding; half the
            host memory and traffic for the moments).
        distributed: data parallel over torch.distributed (default: on when a
            process group with more than one rank is initialized). Parameters
            are broadcast from rank 0; gradients are averaged on the host.
        guard_params: make reads of a weight whose update is still in flight
            wait for it, wherever they happen (default on).
        max_grad_norm: global-norm clipping applied inside the optimizer
            (torch.nn.utils.clip_grad_norm_ cannot see streamed gradients).
        accumulation_steps: backward passes per optimizer.step(). In cpu mode
            without clipping, updates run eagerly during the last backward.
        max_activation_slowdown: "auto" activation paging backs off until the
            measured step time is within this fraction of the un-paged baseline.
        max_inflight_bytes: cap on VRAM held by activations (and, separately,
            by gradients) still in flight to the host; the forward / backward
            pass waits for the link beyond it. Default: 1 GiB, or 10% of the
            GPU's memory budget if that is smaller.
    """
    if unwrap_session(model) is not None:
        raise RuntimeError("model is already wrapped; call trainstream.unwrap(model) first")
    if offload not in ("auto", True):
        raise ValueError("offload must be 'auto' or True (to train without TrainStream, don't wrap the model)")
    params = list(_unique_params(model))
    if not params:
        raise ValueError("trainstream.wrap needs a model with parameters")
    on_cpu = params[0].device.type == "cpu"
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
    paged_blocks = _paged_block_indices(param_offload, len(stack), weight_bytes, on_cpu, target)
    if paged_blocks:
        if optimizer_mode == "gpu":
            raise ValueError("param_offload needs optimizer_mode='cpu' (paged weights are updated on the host)")
        optimizer_mode = "cpu"

    host_init: Dict[int, torch.Tensor] = {}
    frozen_init: Dict[int, torch.Tensor] = {}
    if on_cpu:
        # Move everything except the paged block weights; those go to host RAM
        # and get a VRAM tensor with empty storage, filled block by block later.
        paged_ids = {id(p) for ps in pageable_params(model, stack, paged_blocks).values() for p in ps}
        with torch.no_grad():
            for m in model.modules():
                for name, b in list(m._buffers.items()):
                    if b is not None:
                        m._buffers[name] = b.to(target)
            for p in params:
                if id(p) in paged_ids:
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
        paged_trainable = {id(p) for ps in pageable_params(model, stack, paged_blocks).values()
                           for p in ps if p.requires_grad}
    opt = _convert_optimizer(optimizer, model, optimizer_mode, max_grad_norm, threads, lr,
                             moment_dtype=moment_dtype, distributed=bool(distributed), _broadcast=False,
                             _paged=paged_trainable, _host_init=host_init)
    host_init.clear()
    if max_grad_norm is not None:
        opt.max_grad_norm = max_grad_norm
    opt.accumulation_steps = max(1, int(accumulation_steps))
    if max_inflight_bytes is None:
        max_inflight_bytes = int(min(1 << 30, 0.10 * vram_budget(target)[1]))
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
    if paged_blocks and activation_offload == "auto":
        # A model that needs its weights paged is memory-bound: page every
        # block's activations from the first step instead of calibrating with
        # all of them resident (which is exactly what would not fit).
        activation_offload = True
    if activation_offload not in (False, 0, 0.0, None) and not stack and verbose:
        print("[trainstream] no repeated block stack found; activation offload disabled")

    config = dict(optimizer_mode=optimizer_mode, offload=offload, activation_offload=activation_offload,
                  param_offload=param_offload, paged_blocks=len(paged_blocks), moment_dtype=opt.moment_dtype,
                  distributed=bool(distributed), accumulation_steps=opt.accumulation_steps,
                  eager_updates=opt.eager_active, max_grad_norm=max_grad_norm, prefetch_depth=prefetch_depth,
                  min_offload_bytes=min_offload_bytes, guard_params=guard_params,
                  max_activation_slowdown=max_activation_slowdown, max_inflight_bytes=max_inflight_bytes,
                  verbose=verbose, _reserved=reserved)
    session = TrainStreamSession(model, opt, hw, config, stack, paged_blocks)
    session.install(frozen_init=frozen_init)
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
