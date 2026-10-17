"""One-line integration: ``model, optimizer = trainstream.wrap(model, optimizer)``."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import torch
from torch import nn

from . import _C
from .hooks import ActivationOffloader, find_blocks
from .optimizer import OffloadAdamW
from .profiler import HardwareProfile, profile_hardware

_ATTR = "_trainstream_session"


class TrainStreamSession:
    """Everything wrap() installed on a model; reachable as ``model._trainstream_session``."""

    def __init__(self, model, optimizer, activations, profile, config):
        self.model = model
        self.optimizer: OffloadAdamW = optimizer
        self.activations: Optional[ActivationOffloader] = activations
        self.profile: Optional[HardwareProfile] = profile
        self.config = config

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
        }
        if self.activations is not None:
            out["activations"] = dict(self.activations.stats)
            out["activations"]["offloaded_blocks"] = sorted(self.activations.offload)
            out["activations"]["total_blocks"] = self.activations.n
            out["activations"]["calibrating"] = self.activations.calibrating
            out["activations"]["decision"] = dict(self.activations.decision)
        return out

    def remove(self) -> None:
        self.optimizer.detach()
        if self.activations is not None:
            self.activations.remove()


def _convert_optimizer(optimizer, model: nn.Module, mode: str, max_grad_norm, threads, lr) -> OffloadAdamW:
    if isinstance(optimizer, OffloadAdamW):
        return optimizer
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return OffloadAdamW(trainable, lr=lr, mode=mode, max_grad_norm=max_grad_norm, threads=threads)
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

    new = OffloadAdamW(groups, mode=mode, max_grad_norm=max_grad_norm, threads=threads)
    if len(optimizer.state):
        new.load_state_dict(optimizer.state_dict())  # migrate moments to host
        optimizer.state.clear()  # release the VRAM they occupied
    return new


def wrap(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    optimizer_mode: str = "cpu",
    activation_offload: Union[str, float, bool] = "auto",
    blocks: Optional[Sequence[nn.Module]] = None,
    max_grad_norm: Optional[float] = None,
    threads: Optional[int] = None,
    min_offload_bytes: int = 1 << 20,
    prefetch_depth: int = 1,
    accumulation_steps: int = 1,
    max_activation_slowdown: float = 0.15,
    max_inflight_bytes: int = 1 << 30,
    profile: bool = True,
    lr: float = 1e-3,
    verbose: bool = False,
):
    """Enable TrainStream offloading for `model` and return ``(model, optimizer)``.

    Args:
        optimizer: torch.optim.AdamW / Adam (converted, state migrated), an
            OffloadAdamW, or None (a default OffloadAdamW is created with `lr`).
        optimizer_mode: "cpu", "gpu" or "auto" (decided from the profile).
        activation_offload: "auto" (calibrate, then size from bandwidth), a
            fraction in [0, 1], True (all but the last block) or False.
        blocks: the repeated layer stack; auto-detected when None.
        max_grad_norm: global-norm clipping applied inside the optimizer
            (torch.nn.utils.clip_grad_norm_ cannot see streamed gradients).
        accumulation_steps: backward passes per optimizer.step(). In cpu mode
            without clipping, updates run eagerly during the last backward.
        max_activation_slowdown: "auto" activation paging backs off until the
            measured step time is within this fraction of the un-paged baseline.
        max_inflight_bytes: cap on VRAM held by activations still in flight to
            the host; the forward pass waits for the link beyond it.
    """
    if unwrap_session(model) is not None:
        raise RuntimeError("model is already wrapped; call trainstream.unwrap(model) first")
    params = list(model.parameters())
    if not params or params[0].device.type != "cuda":
        raise ValueError("trainstream.wrap expects a model whose parameters are already on a CUDA device")
    device = params[0].device

    need_profile = profile and (activation_offload == "auto" or optimizer_mode == "auto")
    hw = profile_hardware(device) if need_profile else None
    if optimizer_mode == "auto":
        numel = sum(p.numel() for p in params)
        optimizer_mode = hw.recommend_optimizer_mode(numel) if hw else "cpu"

    opt = _convert_optimizer(optimizer, model, optimizer_mode, max_grad_norm, threads, lr)
    if max_grad_norm is not None:
        opt.max_grad_norm = max_grad_norm
    opt.accumulation_steps = max(1, int(accumulation_steps))
    opt.attach(model)

    # PCIe bytes the optimizer itself moves per step, reserved when sizing
    # activation paging: cpu mode streams grads out and weights back in;
    # gpu mode reads+writes both fp32 moments through zero-copy.
    numel = sum(p.numel() for p in params)
    pbytes = params[0].element_size()
    reserved = (numel * pbytes, numel * pbytes) if optimizer_mode == "cpu" else (numel * 8, numel * 8)

    act = None
    if activation_offload not in (False, 0, 0.0, None):
        stack = list(blocks) if blocks is not None else find_blocks(model)
        if stack:
            act = ActivationOffloader(model, stack, activation_offload, min_bytes=min_offload_bytes,
                                      prefetch_depth=prefetch_depth, profile=hw,
                                      max_slowdown=max_activation_slowdown, reserved_link_bytes=reserved,
                                      max_inflight_bytes=max_inflight_bytes)
        elif verbose:
            print("[trainstream] no repeated block stack found; activation offload disabled")

    config = dict(optimizer_mode=optimizer_mode, activation_offload=activation_offload,
                  accumulation_steps=opt.accumulation_steps, eager_updates=opt.eager_active,
                  max_grad_norm=max_grad_norm, prefetch_depth=prefetch_depth, min_offload_bytes=min_offload_bytes)
    session = TrainStreamSession(model, opt, act, hw, config)
    object.__setattr__(model, _ATTR, session)
    if verbose:
        if hw is not None:
            print(hw.summary())
        print(f"[trainstream] {opt!r}; activation blocks: {act.n if act else 0}")
    return model, opt


def unwrap_session(model: nn.Module) -> Optional[TrainStreamSession]:
    return model.__dict__.get(_ATTR)


def unwrap(model: nn.Module) -> nn.Module:
    """Remove every hook wrap() installed. Parameters keep their trained values."""
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
