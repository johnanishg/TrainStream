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
import os
from typing import Any, Dict, Optional, Sequence

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


PACKED_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")  # ModelOpt NVFP4 export


def _load_bytes(params, elig, quantize, prequant, esize, block, double_quant, keep_frac, error_rank,
                lora_targets) -> int:
    """Host bytes load_model's result will hold: quantized matrices packed (plus the
    keep_precise allowance and LoftQ / PiSSA factors), everything else at esize."""
    from trainstream import quant as Q

    kind = "nvfp4" if prequant else quantize if quantize in Q.KINDS else None
    total = 0
    for name, p in params.items():
        if kind is None or name not in elig:
            total += p.numel() * esize
            continue
        shape = tuple(p.shape)
        total += int(Q.packed_bytes(p.numel(), kind, block, shape, double_quant=double_quant) * (1 + keep_frac))
        if error_rank and not prequant and Q.targets_match(name, lora_targets):
            total += int(error_rank) * (shape[0] + p.numel() // shape[0]) * esize
    return total


def _too_large(need: int, have: int, quantize, prequant) -> str:
    """load_model's refusal: the model against RAM, and against RAM and VRAM combined."""
    vram = 0
    try:
        if torch.cuda.is_available():
            vram = torch.cuda.mem_get_info()[0]
    except RuntimeError:
        pass
    fmt = "as stored (NVFP4)" if prequant else ("in bf16 / the checkpoint's dtype" if quantize == "none"
                                                else f"quantized to {quantize}")
    gib = lambda b: f"{b / 2**30:.1f} GiB"  # noqa: E731
    if need > have + vram:
        head = (f"TrainStream: the model exceeds RAM and VRAM combined. It needs about {gib(need)} {fmt}; "
                f"RAM available {gib(have)} + VRAM free {gib(vram)} = {gib(have + vram)}.")
    else:
        head = (f"TrainStream: the model exceeds the available RAM. It needs about {gib(need)} {fmt}, but "
                f"only {gib(have)} of RAM is available (VRAM free: {gib(vram)}). load_model holds the whole "
                f"model in RAM before wrap() moves part of it to the GPU, so the rest would go to the OS "
                f"page file on the SSD.")
    return (head + " Nothing was loaded. Options: a smaller model; a smaller format (quantize='nf4', "
            "double_quant=True); close other programs to free RAM; or disk_offload=<directory on an SSD> "
            "(opt-in: the weights are written there once, then only read). check_memory=False loads anyway.")


def load_model(name_or_path: str, *, quantize="auto", dtype: torch.dtype = torch.bfloat16,
               quant_block_size: int = 64, keep_precise=0, error_rank: int = 0, error_iters: int = 1,
               lora_init: str = "loftq", lora_targets: Optional[Sequence[str]] = None,
               double_quant: bool = False, disk_offload: Optional[str] = None,
               random_weights: bool = False, ram_fraction: float = 0.85, check_memory: bool = True,
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

    An NVFP4 checkpoint exported by NVIDIA ModelOpt (``quant_method: modelopt``,
    ``quant_algo: NVFP4`` in its config) is read as it is stored: its packed
    matrices become TrainStream NVFP4 weights without being expanded or
    quantized again (``quantize`` is ignored for it).

    ``keep_precise`` (with ``quantize="nf4"`` / ``"nvfp4"``): the matrices that
    lose the most in that format, measured as they are read, are read again
    afterwards and kept one step more precise (int8 / fp8), within that
    fraction of extra bytes ("auto": 0.05; default 0: none). ``error_rank`` /
    ``error_iters`` (LoftQ): each quantized matrix also keeps a rank-``error_rank``
    approximation of what quantization lost; ``wrap()`` starts the PEFT LoRA
    adapters added on top at it (see trainstream.wrap). ``lora_init="pissa"`` (with
    ``error_rank`` = the r of the LoRA adapters you will add): each matrix is split,
    as it is read, into its rank-``error_rank`` principal part, which wrap() starts
    the adapter at, and the rest, which is what gets quantized (PiSSA; see
    trainstream.wrap). Only matrices that will carry an adapter may be split:
    ``lora_targets`` names them (module-name suffixes, the same list as
    ``LoraConfig.target_modules``; None: every quantized matrix); wrap() raises if a
    split matrix has no adapter. ``double_quant``: nf4 / int8
    block scales stored in one byte each (see trainstream.wrap). ``disk_offload``: a
    directory; decoder-layer matrices that aren't quantized are written to
    memory-mapped files there as they're read (trainstream.disk), so a base
    larger than RAM loads, and wrap() streams them from there.

    ``check_memory`` (default True): before anything is read, the weights'
    size in the chosen format is compared with the RAM available; a model
    that doesn't fit raises MemoryError, saying whether it exceeds RAM and
    VRAM combined, instead of loading into the OS page file (swap) on the
    SSD. The whole model is held in RAM until wrap() moves part of it to the
    GPU, so RAM alone is the limit here. Skipped with ``disk_offload``.
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
    if Q.check_lora_init(lora_init) == "pissa" and not error_rank:
        raise ValueError("lora_init='pissa' needs error_rank = the rank r of the LoRA adapters to be added")
    keep_frac = Q.keep_fraction(keep_precise)
    cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    qc = getattr(cfg, "quantization_config", None) or {}
    qc = qc if isinstance(qc, dict) else (qc.to_dict() if hasattr(qc, "to_dict") else vars(qc))
    # an NVFP4 checkpoint exported by NVIDIA ModelOpt: its matrices are read as they are stored
    prequant = qc.get("quant_method") == "modelopt" and str(qc.get("quant_algo", "")).upper() == "NVFP4"
    if prequant:
        cfg.quantization_config = None  # built as a plain model; the packed weights are attached below
        quantize = "none"  # the matrices the export kept in 16 bits stay so
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
    if check_memory and not disk_offload:
        need = _load_bytes(params, elig, quantize, prequant, esize, quant_block_size, double_quant, keep_frac,
                           error_rank, lora_targets)
        have = available_ram_bytes() // local_processes()
        if have and need > have:  # refused before a byte is read: nothing spills into the OS page file
            raise MemoryError(_too_large(need, have, quantize, prequant))
    ram = physical_ram_bytes()
    pin_budget = int(0.40 * ram) if ram else 1 << 62
    qcount = 0
    errors, where = [], {}  # keep_precise: each matrix's loss in `quantize`, and where to read it again
    packed = {}  # ModelOpt NVFP4: per matrix, its packed codes and scales (possibly in different shards)
    store = None
    if disk_offload:
        from trainstream.disk import DiskStore
        store = DiskStore(str(disk_offload))

    def put(name: str, value: torch.Tensor, kind=None) -> None:
        nonlocal qcount
        module, attr = _owner(model, name), name.rsplit(".", 1)[-1]
        old = module._parameters[attr]
        if quantize in Q.KINDS and name in elig:
            kind = kind or quantize
            nb = Q.packed_bytes(value.numel(), kind, quant_block_size, value.shape)
            if keep_frac and kind == quantize and quantize in Q.KEEP_KIND:
                errors.append((name, Q.quant_error(value.to(dtype), quantize, quant_block_size), value.numel(),
                               tuple(value.shape)))
            rank = max(0, int(error_rank)) if Q.targets_match(name, lora_targets) else 0
            qw = Q.make_quantized(value.to(dtype), kind, quant_block_size,
                                  pin=_C.pinned_stats()["live_bytes"] + nb <= pin_budget,
                                  error_rank=rank, error_iters=max(1, int(error_iters)),
                                  double_quant=bool(double_quant), lora_init=lora_init)
            holder = nn.Parameter(torch.empty(tuple(value.shape), dtype=dtype), requires_grad=False)
            Q.attach(holder, qw)
            module._parameters[attr] = holder
            qcount += kind == quantize
        elif store is not None and value.is_floating_point() and value.dim() >= 2 and id(_owner(model, name)) in inside:
            v = store.store(value.to(dtype)).view(tuple(value.shape))  # a block matrix: in a file, not in RAM
            module._parameters[attr] = nn.Parameter(v, requires_grad=old.requires_grad)
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
                    if prequant and key.endswith(PACKED_SUFFIXES):
                        base, _, suf = key.rpartition(".")
                        t = fh.get_tensor(key)
                        if suf != "weight" or t.dtype == torch.uint8:  # (a 16-bit matrix is read as usual)
                            packed.setdefault(base, {})[suf] = t
                            continue
                    name = key if key in params else (prefix + key if prefix + key in params else None)
                    if name is not None and params[name].device.type == "meta":
                        where[name] = (os.path.join(folder, f), key)
                        put(name, fh.get_tensor(key))
    if prequant:
        for base, parts in packed.items():
            name = base + ".weight"
            name = name if name in params else prefix + name
            if name not in params or not {"weight", "weight_scale", "weight_scale_2"} <= parts.keys():
                continue
            qw = Q.from_modelopt_nvfp4(parts["weight"], parts["weight_scale"], parts["weight_scale_2"], dtype,
                                       pin=_C.pinned_stats()["live_bytes"] + parts["weight"].numel() <= pin_budget)
            module, attr = _owner(model, name), name.rsplit(".", 1)[-1]
            holder = nn.Parameter(torch.empty(qw.shape, dtype=dtype), requires_grad=False)
            Q.attach(holder, qw)
            module._parameters[attr] = holder
            qcount += 1
        quantize = "nvfp4" if qcount else "none"
        packed.clear()
    # random weights have no checkpoint to read again: they stay as they are
    keep = Q.choose_precise([e for e in errors if e[0] in where], quantize, keep_frac, quant_block_size)
    for name in sorted(keep):  # read again and stored one step more precise
        from safetensors import safe_open
        path, key = where[name]
        with safe_open(path, framework="pt", device="cpu") as fh:
            put(name, fh.get_tensor(key), kind=Q.KEEP_KIND[quantize])
    model.tie_weights()
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise RuntimeError(f"the checkpoint has no values for {len(missing)} parameters, e.g. {missing[:3]}")
    model._ts_load_info = {"quantize": None if quantize == "none" else quantize,
                           "quant_block_size": quant_block_size, "quantized_tensors": qcount,
                           "kept_precise": sorted(keep)}
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


CHECKPOINT_NAME = "trainstream.ckpt"  # async_checkpoint: the optimizer state inside a Trainer checkpoint folder


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

    ``trainstream={"packing": N}`` packs each batch's examples into rows of N
    tokens with no padding (``trainstream.packing``; True: rows as long as the
    batch's longest example). position_ids restart at every example, which
    keeps attention inside it, and no example is trained to follow another.
    Models with soft-capped or scaled logits, label smoothing, or a custom
    ``compute_loss_func`` use the stock loss.

    ``trainstream={"async_checkpoint": True}``: the Trainer's checkpoints keep
    their layout, except that the optimizer state (fp32 master weights and both
    moments: most of a checkpoint's bytes) is written by a background thread
    to ``trainstream.ckpt`` in the checkpoint folder instead of being pickled
    into ``optimizer.pt`` on the training thread (trainstream.checkpoint), so
    training goes on while it is written. ``train(resume_from_checkpoint=...)``
    reads either. ``train()`` returns once the last one is complete.

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
        packing = self.trainstream_config.pop("packing", False)
        self.async_checkpoint = bool(self.trainstream_config.pop("async_checkpoint", False))
        self.last_checkpoint = None  # CheckpointHandle of the latest background checkpoint
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
        if packing:
            from trainstream.packing import PackedCollator
            pad = getattr(getattr(self, "processing_class", None), "pad_token_id", None)
            self.data_collator = PackedCollator(None if packing is True else int(packing), pad if pad is not None else 0)
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

    # async_checkpoint: the optimizer state goes to trainstream.ckpt on a background thread
    def _save_optimizer_and_scheduler(self, output_dir):
        opt = self.ts_optimizer
        if not self.async_checkpoint or opt is None or not self.args.should_save:
            return super()._save_optimizer_and_scheduler(output_dir)
        from trainstream.checkpoint import save_checkpoint
        self.last_checkpoint = save_checkpoint(self.model, opt, os.path.join(output_dir, CHECKPOINT_NAME),
                                               extra={"global_step": int(self.state.global_step)})
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _load_optimizer_and_scheduler(self, checkpoint):
        path = os.path.join(checkpoint, CHECKPOINT_NAME) if checkpoint else None
        if path is None or not os.path.isfile(path) or self.ts_optimizer is None:
            return super()._load_optimizer_and_scheduler(checkpoint)
        from trainstream.checkpoint import load_checkpoint
        load_checkpoint(path, self.model, self.ts_optimizer)
        sched = os.path.join(checkpoint, "scheduler.pt")
        if os.path.isfile(sched):
            self.lr_scheduler.load_state_dict(torch.load(sched, weights_only=True))

    def train(self, *args, **kwargs):
        try:
            out = super().train(*args, **kwargs)
        except BaseException:
            self._sync()
            if self.last_checkpoint is not None:
                self.last_checkpoint._job.wait()  # never leave a half-written file behind an error
            raise
        self._sync()
        if self.last_checkpoint is not None:
            self.last_checkpoint.wait()  # complete on disk (raises OSError if the write failed)
        return out
