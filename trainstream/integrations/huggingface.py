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

import contextlib
from typing import Any, Dict, Optional

import torch
from torch import nn

import trainstream
from trainstream import _C
from trainstream.hooks import find_blocks
from trainstream.kvcache import PagedKVCache
from trainstream.losses import causal_lm_loss, sequence_logprobs
from trainstream.optimizer import OffloadAdamW
from trainstream.wrapper import unwrap_session

try:
    from transformers import Trainer, TrainerCallback
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "trainstream.integrations.huggingface needs the `transformers` package "
        "(pip install \"trainstream[hf]\", or uv pip install \"trainstream[hf]\" --torch-backend=auto)"
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


def _owner(model: nn.Module, name: str) -> nn.Module:
    mod = model
    for part in name.split(".")[:-1]:
        mod = getattr(mod, part)
    return mod


def load_model(name_or_path: str, *, quantize="auto", dtype: torch.dtype = torch.bfloat16,
               quant_block_size: int = 64, random_weights: bool = False, ram_fraction: float = 0.85,
               trust_remote_code: bool = False, **config_overrides) -> nn.Module:
    """Load a causal LM for TrainStream training, streaming its checkpoint.

    The model is built with no weights allocated, then its ``.safetensors``
    files are read one tensor at a time. With ``quantize="int8"``, ``"nf4"``,
    ``"fp8"`` or ``"nvfp4"``
    (or ``"auto"`` when the bf16 weights would not fit in ``ram_fraction`` of
    the available RAM) every decoder-layer matrix is quantized the moment it
    is read, so a base larger than RAM never exists in full precision.
    Quantized weights are frozen: train adapters (PEFT/LoRA) on top.

    Returns the model on the CPU; give it to TrainStreamTrainer or to
    ``trainstream.wrap(model, optimizer, device="cuda")``.

    ``random_weights=True`` builds the architecture from its config with random
    weights instead of reading a checkpoint (memory / speed tests at full size
    without a download). ``config_overrides`` set config fields, e.g.
    ``num_hidden_layers=4``.
    """
    import os

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from trainstream import quant as Q
    from trainstream.optimizer import physical_ram_bytes
    from trainstream.planner import available_ram_bytes, local_processes

    if quantize is None or quantize is False:
        quantize = "none"
    if quantize not in ("auto", "none") + Q.KINDS:
        raise ValueError(f"quantize must be 'auto', 'none', 'int8', 'nf4', 'fp8' or 'nvfp4', got {quantize!r}")
    Q._check("nf4", quant_block_size)
    cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    for k, v in config_overrides.items():
        setattr(cfg, k, v)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, trust_remote_code=trust_remote_code)
    inside = {id(m) for layer in decoder_layers(model) for m in layer.modules()}
    params = dict(model.named_parameters())  # tied weights appear once
    elig = {n for n, p in params.items() if p.is_floating_point() and p.dim() >= 2 and p.numel() >= Q.MIN_NUMEL
            and p.numel() % 2 == 0 and id(_owner(model, n)) in inside
            and (quantize not in Q.BLOCKSCALED_KINDS or (p.dim() == 2 and p.shape[1] % 32 == 0 and p.shape[0] % 16 == 0))}
    esize = torch.finfo(dtype).bits // 8
    total = sum(p.numel() for p in params.values()) * esize
    if quantize == "auto":
        room = ram_fraction * available_ram_bytes() / local_processes()
        quantize = "none"
        if total > room:
            rest = total - sum(params[n].numel() for n in elig) * esize
            quantize = "nf4"
            for kind in ("int8", "nf4"):  # the most precise format that fits
                if rest + sum(Q.packed_bytes(params[n].numel(), kind, quant_block_size) for n in elig) <= room:
                    quantize = kind
                    break
    ram = physical_ram_bytes()
    pin_budget = int(0.40 * ram) if ram else 1 << 62
    qcount = 0

    def put(name: str, value: torch.Tensor) -> None:
        nonlocal qcount
        module, attr = _owner(model, name), name.rsplit(".", 1)[-1]
        old = module._parameters[attr]
        if quantize in Q.KINDS and name in elig:
            nb = Q.packed_bytes(value.numel(), quantize, quant_block_size, value.shape)
            qw = Q.make_quantized(value.to(dtype), quantize, quant_block_size,
                                  pin=_C.pinned_stats()["live_bytes"] + nb <= pin_budget)
            holder = nn.Parameter(torch.empty(tuple(value.shape), dtype=dtype), requires_grad=False)
            Q.attach(holder, qw)
            module._parameters[attr] = holder
            qcount += 1
        else:
            # an owned copy: a view would keep the whole memory-mapped shard alive
            v = value.to(dtype, copy=True) if value.is_floating_point() else value.clone()
            module._parameters[attr] = nn.Parameter(v.contiguous(), requires_grad=old.requires_grad and v.is_floating_point())

    if random_weights:
        std = getattr(cfg, "initializer_range", 0.02)
        g = torch.Generator().manual_seed(0)
        for name, p in params.items():
            if p.dim() >= 2:
                t = torch.empty(tuple(p.shape)).normal_(0.0, std, generator=g)
            else:
                t = (torch.zeros if name.endswith("bias") else torch.ones)(tuple(p.shape))
            put(name, t)
            del t
    else:
        from safetensors import safe_open
        if os.path.isdir(name_or_path):
            folder = name_or_path
        else:
            from huggingface_hub import snapshot_download
            folder = snapshot_download(name_or_path, allow_patterns=["*.safetensors", "*.json"])
        files = sorted(f for f in os.listdir(folder) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {folder}")
        prefix = getattr(model, "base_model_prefix", "") + "."
        for f in files:
            with safe_open(os.path.join(folder, f), framework="pt", device="cpu") as fh:
                for key in fh.keys():
                    name = key if key in params else (prefix + key if prefix + key in params else None)
                    if name is not None and params[name].device.type == "meta":
                        put(name, fh.get_tensor(key))
    model.tie_weights()
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise RuntimeError(f"the checkpoint has no values for {len(missing)} parameters, e.g. {missing[:3]}")
    model._ts_load_info = {"quantize": None if quantize == "none" else quantize,
                           "quant_block_size": quant_block_size, "quantized_tensors": qcount}
    return model


def wrap_hf(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, **kwargs):
    """``trainstream.wrap`` with the HF decoder stack pre-selected for activation paging."""
    kwargs.setdefault("blocks", decoder_layers(model))
    return trainstream.wrap(model, optimizer, **kwargs)


def lm_head_of(model: nn.Module) -> Optional[nn.Linear]:
    """The plain nn.Linear output layer of a causal LM (through PEFT), or None
    when the model post-processes its logits (soft-capping, logit scaling) or
    has a non-standard head, in which case the stock loss path is kept."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    name = type(base).__name__
    if not (name.endswith("ForCausalLM") or name.endswith("LMHeadModel")):
        return None
    cfg = getattr(base, "config", None)
    for attr in ("final_logit_softcapping", "logit_softcapping", "logit_scale", "logits_scaling"):
        v = getattr(cfg, attr, None)
        if v not in (None, 0, 1, 1.0):
            return None
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    return head if type(head) is nn.Linear else None


class _CaptureHidden:
    """Temporarily turns the output layer into an identity that records its
    input, so a normal model(**inputs) call (with every TrainStream hook)
    yields the final hidden states instead of [batch, seq, vocab] logits."""

    def __init__(self, head: nn.Linear):
        self.head = head
        self.hidden = None

    def __enter__(self):
        def capture(x):
            self.hidden = x
            return x
        self.head.forward = capture
        return self

    def __exit__(self, *exc):
        del self.head.forward


def causal_lm_logprobs(model: nn.Module, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                       attention_mask: Optional[torch.Tensor] = None, *, chunk_size: int = 1024,
                       average: bool = False, **model_kwargs) -> torch.Tensor:
    """Log-probability of each sequence's labels under a causal LM, shape [B].

    Labels follow the Hugging Face convention (the input ids, shifted inside;
    -100 marks tokens that do not count, e.g. the prompt). The log-probs come
    from the final hidden states a chunk of tokens at a time, so the
    [batch, seq, vocab] logits never exist; models with soft-capped or scaled
    logits fall back to the model's own logits. Differentiable (the policy side
    of DPO); see reference_logprobs for the reference side.
    """
    labels = input_ids if labels is None else labels
    head = lm_head_of(model)
    if head is None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs).logits
        t = labels[:, 1:].to(logits.device)
        valid = t != -100
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, t.clamp_min(0)[..., None]).squeeze(-1)
        out = (lp * valid).sum(-1)
        return out / valid.sum(-1).clamp_min(1) if average else out
    with _CaptureHidden(head) as cap:
        model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
    return sequence_logprobs(cap.hidden, head.weight, labels.to(cap.hidden.device), head.bias,
                             chunk_size=chunk_size, average=average)


@torch.no_grad()
def reference_logprobs(model: nn.Module, batches, *, chunk_size: int = 1024, average: bool = False) -> torch.Tensor:
    """DPO's reference log-probabilities, without a second model.

    For a PEFT model the reference is the same (paged, quantized) base with
    the adapters switched off, so it costs no extra weights. `batches` is an
    iterable of dicts with ``input_ids`` and optionally ``labels`` and
    ``attention_mask``; runs in eval mode (no dropout) and returns one float32
    CPU tensor with the log-prob of every sequence, in order. Compute it once
    for the dataset before training (one forward pass over it) instead of a
    second forward every step.
    """
    disable = getattr(model, "disable_adapter", None)
    ctx = disable() if callable(disable) else contextlib.nullcontext()
    was_training = model.training
    model.eval()
    out = []
    try:
        with ctx:
            for batch in batches:
                lp = causal_lm_logprobs(model, batch["input_ids"], batch.get("labels"), batch.get("attention_mask"),
                                        chunk_size=chunk_size, average=average)
                out.append(lp.float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(out) if out else torch.empty(0)


try:  # the per-layer Cache API (transformers >= 4.56)
    from transformers.cache_utils import Cache as _HFCache, CacheLayerMixin as _HFCacheLayer
except ImportError:  # pragma: no cover - older transformers
    _HFCache = _HFCacheLayer = None


if _HFCacheLayer is not None:
    class _PagedKVLayer(_HFCacheLayer):
        """One attention layer of a TrainStreamCache (delegates to PagedKVCache)."""

        is_sliding = False
        is_croppable = False
        supports_early_init = False

        def __init__(self, kv: PagedKVCache, idx: int):
            super().__init__()
            self.kv = kv
            self.idx = idx

        def lazy_initialization(self, key_states, value_states) -> None:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.is_initialized = True

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            self.keys, self.values = self.kv.update(self.idx, key_states, value_states)
            return self.keys, self.values

        def get_mask_sizes(self, query_length: int):
            return self.get_seq_length() + query_length, 0

        def get_seq_length(self) -> int:
            return self.kv.seq_length(self.idx)

        def get_max_length(self) -> int:
            return -1

        def reset(self) -> None:
            self.kv.reset()
            self.keys = self.values = None
            self.is_initialized = False

        def _unsupported(self, *args, **kwargs):
            raise NotImplementedError("TrainStreamCache does not support beam search or batch reshuffling; "
                                      "use sampling or greedy decoding")

        reorder_cache = batch_repeat_interleave = batch_select_indices = crop = _unsupported


class TrainStreamCache(_HFCache if _HFCache is not None else object):
    """A Hugging Face ``Cache`` whose keys and values live partly in host RAM
    (``trainstream.PagedKVCache``): as many layers as fit in a share of the free
    VRAM stay resident, the others are paged in while earlier layers run.

        cache = TrainStreamCache(model, max_tokens=prompt_len + max_new_tokens)
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=...)

    For decoder models with full attention in every layer (Llama, Qwen2, Mistral
    without sliding windows, ...); greedy decoding or sampling, no beam search.
    Call ``cache.reset()`` (or make a new one) before the next batch of prompts.
    """

    def __init__(self, model: nn.Module, max_tokens: int, *, resident_layers="auto", page_tokens: int = 256,
                 vram_fraction: float = 0.5):
        if _HFCache is None:
            raise ImportError("TrainStreamCache needs transformers >= 4.56 (per-layer Cache API)")
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = base.config.get_text_config() if hasattr(base.config, "get_text_config") else base.config
        types = getattr(cfg, "layer_types", None) or []
        if any(t != "full_attention" for t in types) or getattr(cfg, "use_sliding_window", False):
            raise ValueError("TrainStreamCache supports models with full attention in every layer; this one has "
                             f"sliding-window or other layer types ({sorted(set(types))})")
        self.kv = PagedKVCache(int(cfg.num_hidden_layers), max_tokens, resident_layers=resident_layers,
                               page_tokens=page_tokens, vram_fraction=vram_fraction)
        super().__init__(layers=[_PagedKVLayer(self.kv, i) for i in range(self.kv.num_layers)])


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
        out["ts_host_state_gb"] = round(self.opt.stats().get("host_state_bytes", 0) / GiB, 3)  # 0 once resident
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

    The next-token loss of causal LMs is computed in chunks of tokens from the
    final hidden states (``trainstream.losses``), so the full [batch, seq,
    vocab] logits never exist; disable with ``trainstream={"chunked_loss": False}``.
    Models with soft-capped or scaled logits, label smoothing, or a custom
    ``compute_loss_func`` use the stock loss.

    A model passed on the CPU is placed by ``trainstream.wrap`` when training
    starts: with ``param_offload`` ("auto" pages when the weights would take
    more than half the GPU) its decoder-layer weights stay in host RAM and are
    streamed in layer by layer, so base models larger than VRAM can be
    fine-tuned (e.g. with LoRA).

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
        self.chunked_loss = bool(self.trainstream_config.pop("chunked_loss", True))
        self.loss_chunk_size = int(self.trainstream_config.pop("loss_chunk_size", 1024))
        self.chunked_loss_used = 0
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

    # -- placement ---------------------------------------------------------
    def _move_model_to_device(self, model, device):
        # A model handed over on the CPU (e.g. one too large for the GPU) is
        # placed by trainstream.wrap() in create_optimizer instead: block
        # weights it pages never go to VRAM all at once.
        if next(model.parameters()).device.type == "cpu":
            return
        return super()._move_model_to_device(model, device)

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
            if next(opt_model.parameters()).device.type == "cpu":
                cfg.setdefault("device", self.args.device)
            if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                cfg.setdefault("max_grad_norm", float(self.args.max_grad_norm))
            if unwrap_session(opt_model) is not None:
                trainstream.unwrap(opt_model)
            _, ts_opt = trainstream.wrap(opt_model, torch_opt, **cfg)
            self.ts_optimizer = ts_opt
            self.optimizer = ts_opt
            self._ts_metrics = TrainStreamMetrics(ts_opt)
        return self.optimizer

    # -- loss -------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        head = lm_head_of(model) if self.chunked_loss and not return_outputs else None
        if (head is None or "labels" not in inputs or self.label_smoother is not None
                or self.compute_loss_func is not None):
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        with _CaptureHidden(head) as cap:
            model(**inputs)
        # Same normalisation as the model's own loss: a sum over the tokens of the
        # whole accumulated batch when the Trainer passes num_items_in_batch to
        # models that accept it, else a per-batch mean (the Trainer then divides
        # by the accumulation steps itself).
        num_items = num_items_in_batch if (self.model_accepts_loss_kwargs and num_items_in_batch is not None) else None
        loss = causal_lm_loss(cap.hidden, head.weight, labels, head.bias, chunk_size=self.loss_chunk_size,
                              num_items=num_items)
        self.chunked_loss_used += 1
        return loss

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
