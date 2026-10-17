"""Hugging Face ``transformers`` integration.

Drop-in Trainer (swap one class name):

    from trainstream.integrations.huggingface import TrainStreamTrainer

    trainer = TrainStreamTrainer(model=model, args=TrainingArguments(..., bf16=True), train_dataset=ds)
    trainer.train()

Custom loops / ``accelerate`` users:

    from trainstream.integrations.huggingface import wrap_hf
    model, optimizer = wrap_hf(model, torch.optim.AdamW(model.parameters(), lr=2e-5))

Works with any ``PreTrainedModel`` (Llama, Qwen, Mistral, GPT-2, ...) and with
PEFT/LoRA models: only trainable parameters get host-side optimizer state.

Requires ``transformers`` (and ``accelerate``, which the Trainer needs); the
core ``trainstream`` package does not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

import trainstream
from trainstream import _C
from trainstream.hooks import find_blocks
from trainstream.optimizer import OffloadAdamW
from trainstream.wrapper import unwrap_session

try:
    from transformers import Trainer, TrainerCallback
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "trainstream.integrations.huggingface needs the `transformers` package "
        "(pip install transformers accelerate)"
    ) from exc

GiB = float(1 << 30)


def decoder_layers(model: nn.Module):
    """The transformer block stack of a Hugging Face model (also through PEFT
    wrappers). Falls back to TrainStream's generic detection."""
    seen = set()
    stack = [model]
    while stack:  # breadth-first through .base_model / .model wrappers
        m = stack.pop(0)
        if id(m) in seen:
            continue
        seen.add(id(m))
        for attr in ("layers", "h", "blocks", "block"):
            layers = getattr(m, attr, None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 1:
                return list(layers)
        for attr in ("base_model", "model", "transformer", "decoder"):
            child = getattr(m, attr, None)
            if isinstance(child, nn.Module):
                stack.append(child)
    return find_blocks(model)


def wrap_hf(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, **kwargs):
    """``trainstream.wrap`` with the HF decoder stack pre-selected for activation paging."""
    kwargs.setdefault("blocks", decoder_layers(model))
    return trainstream.wrap(model, optimizer, **kwargs)


class TrainStreamMetrics:
    """TrainStream memory / transfer metrics for log lines.

    ``ts_vram_alloc_gb``  tensors currently allocated on the GPU
    ``ts_vram_peak_gb``   peak GPU allocation since the previous log
    ``ts_pinned_gb``      pinned (page-locked) host RAM held by TrainStream
    ``ts_host_state_gb``  optimizer state living in host RAM
    ``ts_d2h_gb`` / ``ts_h2d_gb``  bytes moved over PCIe since the previous log
    """

    def __init__(self, optimizer: OffloadAdamW):
        self.opt = optimizer
        self._last = None

    def collect(self) -> Dict[str, float]:
        dev = self.opt.device.index or 0
        out: Dict[str, float] = {}
        ts = dict(_C.transfer_stats(dev))
        if self._last is not None:
            out["ts_d2h_gb"] = round((ts["bytes_d2h"] - self._last["bytes_d2h"]) / GiB, 3)
            out["ts_h2d_gb"] = round((ts["bytes_h2d"] - self._last["bytes_h2d"]) / GiB, 3)
        self._last = ts
        out["ts_vram_alloc_gb"] = round(torch.cuda.memory_allocated(dev) / GiB, 3)
        out["ts_vram_peak_gb"] = round(torch.cuda.max_memory_allocated(dev) / GiB, 3)
        torch.cuda.reset_peak_memory_stats(dev)
        out["ts_pinned_gb"] = round(_C.pinned_stats()["live_bytes"] / GiB, 3)
        out["ts_host_state_gb"] = round(self.opt.stats()["host_state_bytes"] / GiB, 3)
        return out


class TrainStreamCallback(TrainerCallback):
    """For a stock ``Trainer`` driving an OffloadAdamW yourself: prints the
    TrainStream metrics at every log step (they are not added to log_history;
    TrainStreamTrainer does that natively)."""

    def __init__(self, optimizer: OffloadAdamW):
        self.metrics = TrainStreamMetrics(optimizer)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            print({k: v for k, v in self.metrics.collect().items()})


class TrainStreamTrainer(Trainer):
    """``transformers.Trainer`` whose optimizer state, gradients and (a profiled
    share of) activations live in host RAM.

    Extra keyword argument ``trainstream`` (dict) is forwarded to
    ``trainstream.wrap`` — e.g. ``{"optimizer_mode": "cpu", "activation_offload": 1.0}``.

    Mapping from TrainingArguments:
      * ``optim`` must be an Adam/AdamW variant (adamw_torch, adamw_torch_fused, ...)
      * ``learning_rate``, ``adam_beta1/2``, ``adam_epsilon``, ``weight_decay`` and
        the decay/no-decay parameter split are taken over unchanged
      * ``max_grad_norm`` is applied inside the optimizer (the Trainer's own clip
        would see no gradients, because they have already streamed to the host)
      * ``gradient_accumulation_steps`` sets TrainStream's ``accumulation_steps``
      * ``bf16=True`` or fp32 are supported; ``fp16=True`` is rejected because
        GradScaler must inspect gradients on the GPU
    """

    def __init__(self, *args, trainstream: Optional[Dict[str, Any]] = None, **kwargs):
        self.trainstream_config = dict(trainstream or {})
        self.ts_optimizer: Optional[OffloadAdamW] = None
        super().__init__(*args, **kwargs)
        if self.args.fp16:
            raise ValueError("TrainStreamTrainer supports bf16=True or fp32; fp16 needs a GradScaler that "
                             "reads gradients on the GPU, but TrainStream streams them to host RAM.")
        if self.args.world_size > 1 or self.args.n_gpu > 1:
            raise ValueError("TrainStreamTrainer currently supports a single GPU per process.")
        if not torch.cuda.is_available():
            raise ValueError("TrainStreamTrainer needs a CUDA GPU.")
        self._ts_metrics: Optional[TrainStreamMetrics] = None

    # -- optimizer ---------------------------------------------------------
    def create_optimizer(self, model=None):
        opt_model = self.model if model is None else model
        if self.optimizer is None:
            # Let the Trainer build its usual AdamW (decay groups, hyper-parameters);
            # torch optimizers allocate no state until the first step, so this is free.
            torch_opt = super().create_optimizer(opt_model)
            self.optimizer = None
            if not isinstance(torch_opt, (torch.optim.AdamW, torch.optim.Adam)):
                raise ValueError(f"TrainStreamTrainer needs optim='adamw_torch' (or another torch Adam/AdamW), "
                                 f"got {type(torch_opt).__name__} from optim={self.args.optim!r}")
            cfg = dict(self.trainstream_config)
            cfg.setdefault("blocks", decoder_layers(opt_model))
            cfg.setdefault("accumulation_steps", self.args.gradient_accumulation_steps)
            if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                cfg.setdefault("max_grad_norm", float(self.args.max_grad_norm))
            if unwrap_session(opt_model) is not None:
                trainstream.unwrap(opt_model)
            _, ts_opt = trainstream.wrap(opt_model, torch_opt, **cfg)
            self.ts_optimizer = ts_opt
            self.optimizer = ts_opt
            self._ts_metrics = TrainStreamMetrics(ts_opt)
        return self.optimizer

    # The Trainer clips and measures the gradient norm itself by reading
    # param.grad *before* optimizer.step(). With TrainStream those gradients have
    # already left the GPU, so clipping happens inside OffloadAdamW.step() and the
    # norm of the *current* step is substituted at log time, after the step.
    def _clip_grad_norm(self, model):
        return None

    def _get_grad_norm(self, model, grad_norm=None):
        return None

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, *args, **kwargs):
        opt = self.ts_optimizer
        if opt is not None and opt.max_grad_norm:
            grad_norm = opt.last_grad_norm
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        if self._ts_metrics is not None and "loss" in logs:
            logs.update(self._ts_metrics.collect())
        return super().log(logs, *args, **kwargs)

    # -- consistency points ---------------------------------------------------
    # step() returns before the new weights have landed on the GPU; anything that
    # reads weights outside a forward pass must wait for them first.
    def _sync(self):
        if self.ts_optimizer is not None:
            self.ts_optimizer.synchronize()

    def save_model(self, output_dir=None, _internal_call=False):
        self._sync()
        return super().save_model(output_dir, _internal_call)

    def _save_checkpoint(self, model, trial):
        self._sync()
        return super()._save_checkpoint(model, trial)

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._sync()
