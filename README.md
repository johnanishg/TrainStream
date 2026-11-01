# TrainStream

Train and fine-tune models larger than your GPU's memory. TrainStream moves optimizer state, gradients, activations and, when needed, the weights themselves into system RAM, and streams them back over PCIe exactly when the GPU needs them. Frozen weights (a LoRA base) can also be stored as int8, NF4, FP8 or NVFP4, the last two multiplied directly by Blackwell tensor cores. It takes one line of code and has no third-party dependencies.

```python
import trainstream
model, optimizer = trainstream.wrap(model, optimizer)
```

TrainStream is a C++/CUDA extension for PyTorch. It uses only the C++ standard library, the CUDA runtime and PyTorch's ATen: no DeepSpeed, no bitsandbytes. Updates match `torch.optim.AdamW` (max parameter difference ≈3·10⁻⁷ after 20 steps), and activation and weight paging are bit-exact.

## What's new in 0.6

- **Outlier channels in bf16 for NVFP4** (`outliers=`). A few hidden dimensions of an LLM's activations are far larger than the rest. In NVFP4, one of them sets the scale of its 16-value block and rounds the other 15 to zero. With `outliers=0.01`, each matmul picks its largest 1% of input channels on the GPU (no host sync). It quantizes them as zero and multiplies them in bf16 against the same weight columns: `y = scaled_mm(nvfp4(x without them), W) + x[:, idx] @ W[:, idx].T`. The input gradient is unchanged. This removes the activation part of NVFP4's error, not the weight part: the weights stay 4-bit. `benchmarks/quality.py --formats nvfp4-w nvfp4 nvfp4-o` measures the split (`nvfp4-w` = 4-bit weights only). Off by default until that measurement is in.
- **DPO without a second model or four logit tensors.**
  - `sequence_logprobs()` gives per-sequence log-probabilities from the hidden states in chunks, and `dpo_loss()` combines them.
  - `reference_logprobs()` (Hugging Face) runs the reference as the same paged or quantized base with the LoRA adapters switched off, once before training.
  - See `examples/dpo_lora.py`.
- **Paged KV cache for generation during training** (online DPO, GRPO, PPO).
  - `PagedKVCache` keeps as many layers' keys and values in VRAM as a budget allows. The others go to pinned host pages, which are copied back into a device slot on TrainStream's H2D stream while the earlier layers run.
  - `TrainStreamCache` plugs it into `model.generate()`.
  - VRAM holds the resident layers plus two or three one-layer slots.
- **Pinned host arena** (`host_arena="auto"`).
  - Page-locked memory is reserved in slabs at `wrap()` and carved into blocks, so a training step never page-locks memory, which stalls on Windows (WDDM).
  - Frozen weights that don't fit the page-lock budget no longer go through the driver's synchronous pageable copy. All CPU cores stage them into a ring of pinned slots, which are then copied asynchronously.
- **Peer GPUs** (`peer="auto"`, data parallel on one machine).
  - When the GPUs can read each other's memory (NVLink, or PCIe P2P where the driver allows it), the frozen blocks every rank streams are split among the GPUs' VRAM. Each rank maps the others' pools through CUDA IPC and copies GPU to GPU instead of from host RAM.
  - Where peer access or CUDA IPC isn't available it stays off, and the events say why. Two GeForce cards in one machine usually report no P2P.
  - Tested with two processes sharing one GPU on Windows 11 and WSL2: the pools mapped through CUDA IPC, and training matched host streaming bit for bit. Copies between two real GPUs are untested.

## What's new in 0.5

- **FP8 and NVFP4 on tensor cores.** `quantize="fp8"` or `"nvfp4"` store frozen weights block-scaled. On Blackwell GPUs (RTX 50-series, B200) the forward matmuls run on 8- and 4-bit tensor cores through PyTorch's own `scaled_mm`: 0.188 ms for a 512×5120×13824 FP4 matmul against 1.215 ms in bf16. Real Qwen2.5-14B LoRA (Windows, same session): 0.586 s/step with NVFP4 against 0.642 s with NF4 at the same memory. NVFP4 also quantizes activations, so it costs more quality (perplexity +5.2 % on 14B, +11 % on 1.5B, against +3.1 % / +5.9 % for NF4). It stays opt-in: `prefer="speed"` still picks NF4.
- **Recompute instead of stream** (`recompute="auto"`). Under memory pressure, activations are recomputed in backward where measured trials show that's faster than sending them over PCIe. Paged bf16 14B LoRA: 1.39 → 0.98 s/step. 14B 4-bit LoRA at 2,048 tokens: 5.10 → 1.96 s/step, together with the next item.
- **Attention backend selection** (`attention="auto"`). It times PyTorch's attention kernels once on your model's shape and uses the fastest. On an RTX 5090 that is usually cuDNN, 1.5–2.8× faster than PyTorch's default pick on the shapes measured. Under `torch.use_deterministic_algorithms(True)` the non-deterministic kernels (cuDNN, flash) are left out; with `warn_only=True` the memory-efficient kernel is left out too, because PyTorch then keeps its non-deterministic backward. The choice is re-read on every training forward, so switching deterministic mode after `wrap()` reaches recomputed blocks as well.
- **It says what it decided.** `memory_report(model)["events"]` now also logs the activation tuner (calibration, how many blocks it pages, each slowdown it measured and where it settled), and `model._trainstream_session.adapting` is `True` while TrainStream is still tuning itself, so you can tell when step times are steady.
- **Several GPUs in one machine.** Data-parallel ranks share one host copy of the frozen base and reduce gradients through shared memory instead of loopback sockets. Tested with two processes on one GPU, not yet on a multi-GPU machine.
- **Benchmarks against other libraries**: `benchmarks/compare.py` (stock PyTorch, bitsandbytes QLoRA, Unsloth, DeepSpeed) and `benchmarks/quality.py` (perplexity per format). Results are below. The harness keeps the comparison fair:
  - every engine in a scenario uses the same learning rate (2e-4 for LoRA, 2e-5 for full fine-tuning)
  - engines that only train adapters on a 4-bit base are skipped in the full fine-tune
  - every engine's step time is the median of the same number of steps, taken after its warm-up; for TrainStream the warm-up runs until it has finished tuning itself
  - peak VRAM is the peak over the whole run, even when an engine resets PyTorch's peak counter
  - `--live` streams every step and memory sample as JSON lines
- **Fixes:**
  - `load_model` with quantization kept every checkpoint shard mapped in memory (40.5 GB instead of 14.2 GB for 14B).
  - Activation calibration could overcommit VRAM at long sequences. A probe step now measures one block's activations first, recomputing the others, so it is safe for host RAM too.
  - LoRA adapters on a quantized base stay in VRAM instead of being paged with the block (14B NVFP4 LoRA: 0.60 → 0.52 s/step).

## What's new in 0.4

- **0.4.1:** install instructions for uv; clearer messages when the installed torch is the CPU-only build.
- **0.4.0:** built-in int8 / NF4 weight quantization (`quantize=`), `load_model()` (quantize while reading a checkpoint), `quantize_model()`. Real Qwen2.5-14B LoRA 1.37 → 0.87 s/step; a 32B-shaped base trains on 64 GB of RAM.

## Against other libraries (same laptop, WSL2 with 31 GB RAM)

> These tables were measured before the harness fixes in this release. TrainStream's step time could include steps where it was still tuning itself, and its peak VRAM could miss a peak from before its memory planner reset PyTorch's counter. The engines also used two learning rates, which changes losses but not step times. Re-run `benchmarks/compare.py` for numbers measured the new way.

**LoRA r=16, Qwen2.5-14B, batch 1 × 512**

| Engine | Step time | Peak VRAM | Outcome |
|---|---|---|---|
| TrainStream, NVFP4 base (FP4 tensor cores) | 0.52 s | 19.1 GB | fastest |
| bitsandbytes QLoRA (NF4) + PEFT | 0.59 s | 17.4 GB | 1.14× the fastest |
| TrainStream, NF4 base | 0.64 s | 19.1 GB | 1.22× the fastest |
| Unsloth (4-bit) | 0.74 s | 11.5 GB | 1.42× the fastest |
| TrainStream (auto) | 0.87 s | 20.6 GB | 1.67× the fastest |
| bitsandbytes QLoRA + gradient checkpointing | 0.89 s | 12.7 GB | 1.71× the fastest |
| DeepSpeed ZeRO-2, optimizer offload | 75.44 s | 35.9 GB | 145.28× the fastest |
| Stock PyTorch (+ PEFT) | 79.60 s | 37.6 GB | 153.30× the fastest |

**LoRA r=16, Qwen2.5-14B, batch 1 × 2048**

| Engine | Step time | Peak VRAM | Outcome |
|---|---|---|---|
| TrainStream, NVFP4 base (FP4 tensor cores) | 2.01 s | 17.8 GB | fastest |
| Unsloth (4-bit) | 2.55 s | 11.7 GB | 1.27× the fastest |
| bitsandbytes QLoRA + gradient checkpointing | 2.89 s | 16.3 GB | 1.44× the fastest |
| TrainStream, NF4 base | 3.15 s | 17.6 GB | 1.57× the fastest |
| TrainStream (auto) | 3.18 s | 22.0 GB | 1.58× the fastest |
| bitsandbytes QLoRA (NF4) + PEFT | 151.37 s | 34.7 GB | 75.30× the fastest |
| Stock PyTorch (+ PEFT) | 280.24 s | 57.9 GB | 139.40× the fastest |
| DeepSpeed ZeRO-2, optimizer offload | — | — | did not finish 12 steps in 50 min (weights spill past VRAM) |

**Full fine-tune, Qwen2.5-1.5B, batch 4 × 512**

| Engine | Step time | Peak VRAM | Outcome |
|---|---|---|---|
| Stock PyTorch (+ PEFT) | 1.33 s | 32.4 GB | fastest |
| TrainStream (auto) | 1.61 s | 20.0 GB | 1.21× the fastest |
| DeepSpeed ZeRO-2, optimizer offload | — | — | killed: out of host RAM (31 GB in WSL) |
- With an NVFP4 base, TrainStream had the fastest step in both 14B LoRA runs: 0.52 s at 512 tokens and 2.01 s at 2,048. It holds more in VRAM than Unsloth (18–19 GB against 11.5), and NVFP4 costs quality (see the perplexity table).
- With an NF4 base, the format bitsandbytes and Unsloth use, TrainStream was 7 % slower than bitsandbytes at 512 tokens, and behind Unsloth and bitsandbytes with checkpointing at 2,048.
- Stock PyTorch, bitsandbytes without checkpointing at 2,048 tokens, and DeepSpeed did not fit in 24 GB of VRAM. The driver spilled into system RAM (75–280 s per step), or the run did not finish or was killed.
- The 1.5B full fine-tune ran with 29 of WSL's 31 GB of RAM in use, and TrainStream's step time varied between runs (1.80, 1.09 and 1.61 s; the table shows the latest). On Windows with 64 GB it holds 1.18 s. Stock PyTorch (1.33 s) allocated 32 GB on the 24 GB GPU, the excess in shared system memory.

Each engine runs in a fresh process with the same model, random-token data, LoRA settings (r=16, all projections), learning rate and loop. Step time is the median of the steps after the warm-up: 2 steps, plus, for TrainStream, the steps until it has finished tuning itself. Unsloth uses its own defaults. Re-run with `benchmarks/compare.py`.

## Results (RTX 5090 Laptop GPU 24 GB, 64 GB RAM)

| Workload | Stock PyTorch | TrainStream |
|---|---|---|
| 1.8B-parameter GPT, fp32, batch 8×512 | 34.1 GiB peak: exceeds VRAM, driver spills to RAM, 13.7 s/step | 20.6 GiB, 1.1 s/step (13.5 GiB with all activations paged) |
| Qwen2.5-1.5B full fine-tune (Hugging Face Trainer) | 35.9 GiB peak (spill), 111.7 s / 10 steps | 18.7 GiB, 28.5 s / 10 steps, same losses |
| 1.8B GPT, weights paged to RAM | — | 2.4 GiB peak, 2.1 s/step; 1.84 GiB inside a 4 GB VRAM cap with bf16 moments |
| 3.6B GPT full fine-tune (all parameters trained) | needs ~58 GB of VRAM | 3.1 GiB peak, 3.15 s/step, 30 GiB of host RAM |
| LoRA r=16, Qwen2.5-7B architecture | 20.3 GB peak, 0.27 s/step | 4.6 GB peak, 1.25 s/step (every layer paged) |
| LoRA r=16, real Qwen2.5-14B, bf16 base | does not fit (29.5 GB of weights) | 1.37 s/step: the memory planner keeps 28 of 48 layers in VRAM and streams 20 from RAM |
| LoRA r=16, real Qwen2.5-14B, NF4 base (`quantize="auto", prefer="speed"`) | does not fit | **0.87 s/step**: all 48 layers in VRAM as 4-bit, 18.3 GB peak, 11.3 GB process RAM |
| LoRA r=16, Qwen2.5-32B architecture, NF4 base (`load_model(quantize="nf4")`) | does not fit (65.5 GB in bf16, more than the RAM) | 2.53 s/step, 21.3 GB peak, 31.4 GB process RAM |
| 85M GPT (fits either way) | 2.55 GiB, 43.7 ms/step | `offload="auto"` switches to stock fused AdamW: 2.56 GiB, 43.8 ms/step |

Offloading costs throughput when a model already fits, which is why `offload="auto"` measures two steps and switches to plain fused GPU AdamW if offloading isn't needed. TrainStream's value is training what doesn't fit, or freeing VRAM for larger batches. All numbers come from one machine. The 7B and 32B rows use the exact Qwen2.5 shapes with random weights; the 14B rows use the real checkpoint.

## Requirements

- An NVIDIA GPU and a CUDA build of PyTorch
- The CUDA toolkit matching PyTorch's CUDA major version, and a C++ compiler (Visual Studio 2022+ Build Tools with "Desktop development with C++" on Windows, GCC 11+ on Linux)
- Optional, for the Hugging Face integration: `transformers`, `accelerate`, and `peft` for LoRA (all three come with `trainstream[hf]`)

Tested on an RTX 5090 Laptop GPU with Windows 11 (PyTorch 2.14.1, CUDA 13.4, MSVC, Python 3.14), where all 222 tests pass with no warnings, and Ubuntu 24.04 under WSL2 (PyTorch 2.14.1, CUDA 13.3, GCC 13.3, Python 3.12), where all 222 pass as well (with bitsandbytes 0.50.2 installed for the NF4 comparison), both from a wheel installed with pip and with uv. Installs were checked with pip and with uv 0.13 (`uv pip install`, `uv sync` in a project, and building from the sdist) on both. Native Linux and other GPUs are untested.

## Install

TrainStream installs with **pip** or **uv**. It is a small pure-Python wheel and doesn't pull in PyTorch, because you choose the CUDA build of PyTorch that matches your driver. Install CUDA PyTorch first (or in the same command), then TrainStream.

**pip**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # if you don't have CUDA PyTorch yet
pip install trainstream                                                # core library
pip install "trainstream[hf]"                                          # + transformers, accelerate, peft
```

**uv** (`uv pip`)

```bash
uv pip install torch trainstream --torch-backend=auto            # picks the CUDA build for your driver
uv pip install torch "trainstream[hf]" --torch-backend=auto      # + transformers, accelerate, peft
```

**uv project** (`uv add` / `uv sync`). Point `torch` at the PyTorch index in your `pyproject.toml`:

```toml
[project]
dependencies = ["torch", "trainstream"]        # or "trainstream[hf]"

[tool.uv.sources]
torch = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Then `uv sync` and `uv run python train.py`.

> **CPU-only torch trap.** On Windows, `torch` from PyPI is the CPU build. If something else that depends on torch (for example `transformers` or `accelerate`) is installed without the CUDA index, the installer may fetch it, and `import trainstream` then stops with a message saying so. Fix: `pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall`, or `uv pip install torch --torch-backend=auto --reinstall-package torch`.

The first `import trainstream` compiles the CUDA extension against your PyTorch (16–19 s on Windows and 18 s on Linux, measured) and caches it; later imports take about a second. On Windows it finds Visual Studio and the CUDA toolkit itself, so any terminal works. The cache is keyed by PyTorch, CUDA and Python version and GPU, so upgrading PyTorch triggers one rebuild.

Upgrade: `pip install -U --no-cache-dir trainstream` or `uv pip install -U trainstream` (`uv lock --upgrade-package trainstream` in a project).

To compile at install time instead of on first import: `pip install trainstream --no-binary trainstream --no-build-isolation` (or the same flags with `uv pip install`). From a checkout: `build.bat` (Windows) or `python setup.py build_ext --inplace`.

## Quick start

```python
import torch, trainstream

model = MyModel().cuda()                                  # 1. build the model
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)   # 2. a normal optimizer
model, optimizer = trainstream.wrap(model, optimizer)    # 3. wrap once, before the first forward

for x, y in loader:                                       # 4. an ordinary training loop
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x.cuda(), y.cuda())
    loss.backward()                                       #    gradients stream to host RAM during backward
    optimizer.step()                                      #    host AdamW; returns once the update is queued
    optimizer.zero_grad()
```

### A model larger than the GPU

Leave it on the CPU and pass `device`. The memory planner measures the machine's VRAM and RAM, keeps as many layers resident as fit, and streams the rest:

```python
model = MyGPT(cfg)                                        # stays on the CPU
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model, optimizer = trainstream.wrap(model, optimizer, device="cuda")
```

### Hugging Face Trainer

Swap one class name (`pip install transformers accelerate`):

```python
from transformers import AutoModelForCausalLM, TrainingArguments
from trainstream.integrations.huggingface import TrainStreamTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float32)
args = TrainingArguments(output_dir="out", bf16=True, per_device_train_batch_size=4,
                         gradient_accumulation_steps=2, learning_rate=2e-5, max_grad_norm=1.0)
trainer = TrainStreamTrainer(model=model, args=args, train_dataset=ds,
                             trainstream={"moment_dtype": "auto"})   # optional: any wrap() option
trainer.train()
```

`TrainStreamTrainer` computes the causal-LM loss in chunks, so the full logits tensor never exists. Hyper-parameters, the LR schedule, accumulation, clipping, checkpoints and `resume_from_checkpoint` carry over. Use `bf16=True` or fp32; fp16 with loss scaling isn't supported. For your own loop: `model, optimizer = wrap_hf(model, optimizer)`.

### LoRA on a quantized base (QLoRA-style)

```python
import peft
from trainstream.integrations.huggingface import TrainStreamTrainer, load_model

model = load_model("Qwen/Qwen2.5-14B", quantize="nf4")    # streams the checkpoint, quantizes while reading
model = peft.get_peft_model(model, peft.LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
trainer = TrainStreamTrainer(model=model, args=TrainingArguments(output_dir="out", bf16=True), train_dataset=ds)
trainer.train()
```

Or let `wrap()` decide: `trainstream.wrap(model, optimizer, device="cuda", quantize="auto", prefer="speed")`. Or quantize beforehand: `trainstream.quantize_model(model, "int8")`. Only frozen weights are quantized; freeze the base (or add adapters) first.

### Data parallel

```python
# torchrun --nproc_per_node=2 train.py
import os, torch.distributed as dist
dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
model, optimizer = trainstream.wrap(MyModel().cuda(), torch.optim.AdamW(...))   # detected automatically
```

### DPO with LoRA

```python
import trainstream
from trainstream.integrations.huggingface import causal_lm_logprobs, load_model, reference_logprobs, wrap_hf

model = peft.get_peft_model(load_model("Qwen/Qwen2.5-7B", quantize="nf4"), lora_config)
model, optimizer = wrap_hf(model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6))
pairs = [(chosen, rejected), ...]   # dicts with input_ids and labels (-100 on the prompt), batch_size rows each
# adapters off: the same base is the reference; computed once, up front
ref = reference_logprobs(model, [b for pair in pairs for b in pair]).split(batch_size)
for k, (chosen, rejected) in enumerate(pairs):
    pc = causal_lm_logprobs(model, chosen["input_ids"], chosen["labels"])
    pr = causal_lm_logprobs(model, rejected["input_ids"], rejected["labels"])
    loss, chosen_reward, rejected_reward = trainstream.dpo_loss(pc, pr, ref[2 * k], ref[2 * k + 1], beta=0.1)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

### Generating with a paged KV cache

```python
from trainstream.integrations.huggingface import TrainStreamCache

cache = TrainStreamCache(model, max_tokens=prompt_len + 512)   # resident layers chosen from free VRAM
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, past_key_values=cache)
cache.reset()                                                  # before the next batch of prompts
```

### Inspecting what it decided

```python
r = trainstream.memory_report(model)
r["events"]                    # plain-text log: quantization, memory plan, attention backend,
                               # activation paging / recompute decisions, resident mode, .to()
r["params"]["paged_blocks"]    # layers whose weights stream from host RAM
r["params"]["quantize"]        # None, "int8", "nf4", "fp8" or "nvfp4"
r["activations"]["policy"]     # per block: "keep", "offload" or "recompute"
r["plan"]["history"]           # every planner move, with the measured peaks
model._trainstream_session.adapting   # True while it is still tuning itself
```

### Rules of thumb

- Order: build the model → freeze / add adapters → (quantize) → create the optimizer → `wrap()` → train.
- Use `max_grad_norm=` instead of `torch.nn.utils.clip_grad_norm_`; gradients have already left the GPU.
- Declare gradient accumulation with `accumulation_steps=`.
- `model.state_dict()` and `optimizer.state_dict()` work while wrapped. `trainstream.unwrap(model)` loads every weight back into VRAM and removes the hooks.

## Options

### `trainstream.wrap(model, optimizer=None, **options) → (model, optimizer)`

| Option | Accepted values | Default | What it does |
|---|---|---|---|
| `optimizer` | `torch.optim.AdamW`, `torch.optim.Adam`, `trainstream.OffloadAdamW`, `None` | `None` | converted, with hyper-parameters and state migrated; `None` creates one with `lr` |
| `optimizer_mode` | `"cpu"`, `"gpu"`, `"auto"` | `"cpu"` | host AdamW, a zero-copy CUDA kernel, or chosen from the hardware profile |
| `offload` | `"auto"`, `True` | `"auto"` | `"auto"` switches to fused torch AdamW after 2 steps if the model fits |
| `activation_offload` | `"auto"`, `True`, `False`, fraction 0.0–1.0 | `"auto"` | how many blocks' activations are paged to RAM |
| `param_offload` | `"auto"`, `True`, `False`, fraction 0.0–1.0, list of block indices | `"auto"` | `"auto"`: the memory planner decides which layers stay in VRAM |
| `quantize` | `"auto"`, `"none"`, `None`, `False`, `"int8"`, `"nf4"`, `"fp8"`, `"nvfp4"` | `"auto"` | frozen weights only; `"auto"` quantizes only when the run can't fit otherwise (int8 before NF4); fp8 / nvfp4 run on Blackwell tensor cores |
| `quant_block_size` | even integer ≥ 2 | `64` | values per fp32 scale (int8 / nf4; fp8 and nvfp4 use fixed 32 / 16) |
| `prefer` | `"exact"`, `"speed"` | `"exact"` | `"speed"` lets `quantize="auto"` choose NF4 when it keeps every layer in VRAM |
| `recompute` | `"auto"`, `True`, `False`, fraction 0.0–1.0, list of block indices | `"auto"` | recompute blocks in backward instead of keeping or paging their activations; `"auto"`: under memory pressure, as many as measured trials show is fastest |
| `attention` | `"auto"`, `"cudnn"`, `"efficient"`, `"flash"`, `"math"`, `None` | `"auto"` | attention backend: the fastest measured on your model's shape, a named one first, or PyTorch's own order; deterministic mode leaves out the non-deterministic kernels |
| `outliers` | `0`, `False`, `None`, `"auto"`, `True`, float 0.0–0.1 | `0` | with `quantize="nvfp4"` on tensor cores: the fraction of input channels (the largest of each call) multiplied in bf16 instead of 4-bit; `"auto"` / `True` mean `0.01`; ignored by other formats |
| `host_arena` | `"auto"`, `False`, `None`, bytes | `"auto"` | page-locked host memory reserved up front, from which paged activations, staging and other pinned buffers are carved; `"auto"`: when the run streams anything, reserve one slab now and grow on demand within the page-lock budget; bytes: reserve that much now and cap growth there |
| `peer` | `"auto"`, `False`, `None`, bytes | `"auto"` | data parallel on one machine: pool the frozen blocks every rank streams in the GPUs' VRAM and copy them GPU to GPU (needs peer access between the GPUs and CUDA IPC; set up collectively at step 10); `"auto"`: when peer copies reach 0.9× the host link, each GPU gives half the headroom under the memory plan's target; bytes: that much per GPU. While active, `unwrap()` is collective |
| `moment_dtype` | `"auto"`, `"float32"`, `"bfloat16"`, `"bf16"`, `torch.bfloat16` | `"auto"` | fp32 unless that wouldn't fit in RAM; bf16 uses stochastic rounding |
| `device` | `None`, `"cuda"`, `"cuda:N"`, `torch.device` | `None` | target GPU when the model is on the CPU |
| `blocks` | `None`, list of layer modules | `None` | the repeated layer stack; auto-detected |
| `distributed` | `None`, `True`, `False` | `None` | data parallel when a process group with > 1 rank exists |
| `guard_params` | `True`, `False` | `True` | reads of a weight wait for its in-flight update |
| `max_grad_norm` | `None`, float | `None` | global-norm clipping inside the optimizer |
| `accumulation_steps` | integer ≥ 1 | `1` | backward passes per `step()` |
| `vram_headroom` | float 0.0–1.0 | `0.10` | share of usable VRAM the planner leaves free |
| `check_memory` | `True`, `False` | `True` | raise `MemoryError` at `wrap()` when the run can't fit |
| `max_inflight_bytes` | `None`, bytes | `None` (1 GiB or 10% of VRAM) | VRAM held by transfers still crossing PCIe |
| `max_activation_slowdown` | float | `0.15` | how much slower `"auto"` activation paging may make a step |
| `prefetch_depth` | integer ≥ 1 | `1` | activation blocks fetched ahead in backward |
| `min_offload_bytes` | bytes | `1048576` | smaller activations stay on the GPU |
| `threads` | `None`, integer | `None` (all cores) | host AdamW threads |
| `profile` | `True`, `False` | `True` | run (or load cached) hardware benchmarks |
| `lr` | float | `1e-3` | used only when `optimizer` is `None` |
| `verbose` | `True`, `False` | `False` | print decisions as they happen |

Anything else raises `ValueError` (for example `quantize="fp4"`).

### Quantization formats

| `quantize=` | Bytes per weight | Mean relative error | Use it when |
|---|---|---|---|
| `"none"` | 2 (bf16) | 0 (bit-exact) | it fits |
| `"int8"` | 1.06 | 0.6 % | bf16 doesn't fit in RAM |
| `"nf4"` | 0.56 | 9 % | int8 doesn't fit either, or with `prefer="speed"` |
| `"fp8"` | 1.03 | perplexity +1.8 % (1.5B) | FP8 tensor-core matmuls on Blackwell, near-bf16 quality |
| `"nvfp4"` | 0.56 | perplexity +5–11 % | the fastest 4-bit path on Blackwell, when its quality is acceptable |
| `"auto"` | — | — | decided from the measured RAM and VRAM |

NF4 is bit-identical to bitsandbytes' NF4 (same codes, packing and scales); TrainStream doesn't depend on bitsandbytes. A quantized run trains on the dequantized weights, as QLoRA does, so it isn't bit-exact with the bf16 base.

### `load_model(name_or_path, ...)` (in `trainstream.integrations.huggingface`)

| Argument | Accepted values | Default |
|---|---|---|
| `name_or_path` | Hub id or local folder with `.safetensors` files | — |
| `quantize` | `"auto"`, `"none"`, `None`, `False`, `"int8"`, `"nf4"`, `"fp8"`, `"nvfp4"` | `"auto"` (quantize only if bf16 wouldn't fit in `ram_fraction` of free RAM) |
| `dtype` | `torch.bfloat16`, `torch.float16`, `torch.float32` | `torch.bfloat16` |
| `quant_block_size` | even integer ≥ 2 | `64` |
| `random_weights` | `True`, `False` | `False` (build the shapes without reading a checkpoint) |
| `ram_fraction` | float 0.0–1.0 | `0.85` |
| `trust_remote_code` | `True`, `False` | `False` |
| `**config_overrides` | any config field, e.g. `num_hidden_layers=4` | — |

### `trainstream.quantize_model(model, quantize="nf4", *, blocks=None, block_size=64, pin_budget=None)`

Quantizes the frozen block weights in place, one tensor at a time, before `wrap()`. Returns `{"tensors": count, "bytes_before": ..., "bytes_after": ...}`.

| Argument | Accepted values | Default | What it does |
|---|---|---|---|
| `model` | `nn.Module` | — | the model; only frozen (`requires_grad=False`) weights of the block stack are quantized |
| `quantize` | `"int8"`, `"nf4"`, `"fp8"`, `"nvfp4"` | `"nf4"` | the storage format (see the table above) |
| `blocks` | `None`, list of layer modules | `None` | the layer stack to quantize; auto-detected with `find_blocks` |
| `block_size` | even integer ≥ 2 | `64` | values per scale for int8 / NF4 (FP8 and NVFP4 use fixed 32 / 16) |
| `pin_budget` | `None`, bytes | `None` (40 % of physical RAM) | how much of the quantized data may be page-locked; the rest stays pageable |

### `trainstream.OffloadAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, *, ...)`

AdamW with its moments and fp32 master weights in host RAM. `wrap()` creates or converts one for you; you can also build it yourself.

| Argument | Accepted values | Default | What it does |
|---|---|---|---|
| `params` | parameters or param-group dicts, on a CUDA device | — | what it trains (frozen parameters are skipped) |
| `lr`, `betas`, `eps`, `weight_decay` | as `torch.optim.AdamW` | `1e-3`, `(0.9, 0.999)`, `1e-8`, `1e-2` | same math as `torch.optim.AdamW` (decoupled weight decay) |
| `mode` | `"cpu"`, `"gpu"` | `"cpu"` | `"cpu"`: gradients stream to RAM and host threads run AdamW; `"gpu"`: a CUDA kernel updates in place, reading the moments from pinned RAM (zero-copy) |
| `max_grad_norm` | `None`, float | `None` | global-norm gradient clipping inside the optimizer |
| `compute_grad_norm` | `True`, `False` | `False` | compute the global gradient norm without clipping (read it from `last_grad_norm`) |
| `threads` | `None`, integer | `None` (all cores) | host AdamW worker threads |
| `chunk_elems` | integer | `1048576` | elements per work chunk when a tensor's update is split across threads |
| `accumulation_steps` | integer ≥ 1 | `1` | backward passes per `step()`; an extra backward before `step()` raises |
| `eager` | `None`, `True`, `False` | `None` (on when there is no clipping or norm) | start each tensor's update as soon as its last gradient lands, during backward |
| `pin_moments` | `None`, `True`, `False` | `None` (pin unless it would lock more than 40 % of RAM) | page-lock the moments in host RAM (`"gpu"` mode always pins) |
| `moment_dtype` | `"float32"`, `"fp32"`, `"bfloat16"`, `"bf16"`, `torch.float32`, `torch.bfloat16` | `"float32"` | bf16 halves moment memory and traffic, with stochastic rounding |
| `distributed` | `True`, `False` | `False` | data parallel: broadcast from rank 0, gradients averaged on the host (`"cpu"` mode only) |

Methods and attributes: `step(closure=None)`, `zero_grad()`, `attach(model, guard=True)` (install the streaming hooks; `wrap()` does this), `detach()`, `synchronize()` (block until every queued update has landed), `stats()`, `state_dict()` / `load_state_dict()` (compatible with `torch.optim.AdamW`, plus the fp32 `master` weights), `add_param_group()`, `last_grad_norm`.

### `trainstream.integrations.huggingface`

| Function / class | Arguments | What it does |
|---|---|---|
| `TrainStreamTrainer(*args, trainstream=None, **kwargs)` | everything `transformers.Trainer` takes, plus `trainstream`: a dict of any `wrap()` option and `"chunked_loss"` (`True`/`False`, default `True`: the causal-LM loss is computed in chunks so the logits never exist) and `"loss_chunk_size"` (tokens per chunk, default `1024`) | a `Trainer` whose optimizer state, gradients and activations live in host RAM |
| `wrap_hf(model, optimizer=None, **options)` | `options`: any `wrap()` option | `wrap()` with `blocks` preset to the decoder layers |
| `load_model(name_or_path, ...)` | see the table above | streams a checkpoint, quantizing while it reads |
| `TrainStreamCallback(optimizer)` | `optimizer`: an `OffloadAdamW` | for a stock `Trainer` that you gave an `OffloadAdamW`: prints the TrainStream metrics at each log step |
| `causal_lm_logprobs(model, input_ids, labels=None, attention_mask=None, *, chunk_size=1024, average=False, **model_kwargs)` | `labels`: as for the model's loss (the input ids; `-100` marks tokens that don't count, e.g. the prompt); default `input_ids`. `chunk_size`: tokens per chunk. `average`: mean per counted token instead of the sum. `model_kwargs`: passed to the model | the log-probability of each sequence, shape `[batch]`, differentiable; computed from the final hidden states in chunks (models with soft-capped or scaled logits use their own logits) |
| `reference_logprobs(model, batches, *, chunk_size=1024, average=False)` | `batches`: iterable of dicts with `"input_ids"` and optionally `"labels"` and `"attention_mask"` | DPO's reference log-probs without a second model: a PEFT model runs with its adapters off, in eval mode and without gradients; one float32 CPU tensor with every sequence's value, in order |
| `TrainStreamCache(model, max_tokens, *, resident_layers="auto", page_tokens=256, vram_fraction=0.5)` | `model`: a decoder with full attention in every layer (sliding-window models are refused). Other arguments as `PagedKVCache` below | a `transformers` `Cache` for `model.generate(past_key_values=...)` whose keys and values are partly in host RAM; greedy or sampling (no beam search); `.reset()` before the next prompts; `.kv` is the `PagedKVCache`. Needs `transformers` ≥ 4.56 |

How `TrainStreamTrainer` uses `TrainingArguments`:
- `optim` must be an Adam/AdamW variant.
- `learning_rate`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay` and the decay / no-decay split carry over.
- `max_grad_norm` is applied inside the optimizer.
- `gradient_accumulation_steps` becomes `accumulation_steps`.
- Use `bf16=True` or fp32; `fp16=True` is rejected.
- It is single-GPU per process.

Its log lines add these metrics:
- `ts_vram_alloc_gb`, `ts_vram_peak_gb` (since the previous log)
- `ts_pinned_gb`, `ts_host_state_gb`
- `ts_d2h_gb` / `ts_h2d_gb` (bytes over PCIe since the previous log)

### Losses

| Function | Arguments |
|---|---|
| `trainstream.chunked_cross_entropy(hidden, weight, targets, bias=None, *, chunk_size=1024, ignore_index=-100, reduction="mean", num_items=None)` | cross-entropy of `hidden @ weight.T (+ bias)` without building the logits. `hidden` `[..., D]` (after the last norm), `weight` `[V, D]`, `bias` `[V]` or `None`, `targets` class indices. `chunk_size`: tokens per chunk. `ignore_index`: positions that don't count. `reduction`: `"mean"` or `"sum"`. `num_items`: divide by this instead of the counted positions (as Hugging Face does with gradient accumulation) |
| `trainstream.causal_lm_loss(hidden, weight, labels, bias=None, **kwargs)` | the next-token loss as Hugging Face computes it (labels shifted by one); `kwargs` as above |
| `trainstream.sequence_logprobs(hidden, weight, labels, bias=None, *, chunk_size=1024, ignore_index=-100, shift=True, average=False)` | the log-probability of each sequence's labels, `[B]` float32, without building the logits; differentiable. `hidden` `[B, T, D]`, `labels` `[B, T]`. `shift=True`: labels are the input ids (position t predicts label t + 1), as Hugging Face passes them; `False`: position t predicts label t. `average`: mean per counted token instead of the sum |
| `trainstream.dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta=0.1, label_smoothing=0.0)` | Direct Preference Optimization from per-sequence log-probs (`[B]` each). Returns `(loss, chosen_rewards, rejected_rewards)`: the mean loss and the implicit rewards `beta * (policy - reference)` (detached). `beta`: strength of the pull toward the reference. `label_smoothing` > 0: the conservative variant for noisy preference labels |

### `trainstream.PagedKVCache(num_layers, max_tokens, *, resident_layers="auto", page_tokens=256, vram_fraction=0.5)`

A KV cache for autoregressive generation whose offloaded layers keep their keys and values in pinned host pages. Inference only. The Hugging Face adapter is `TrainStreamCache` above.

| Argument | Accepted values | Default | What it does |
|---|---|---|---|
| `num_layers` | integer ≥ 1 | — | attention layers of the model |
| `max_tokens` | integer ≥ 1 | — | the longest sequence expected (prompt + new tokens); sizes the slots and the `"auto"` budget. Longer sequences still work, with a reallocation |
| `resident_layers` | `"auto"`, integer ≥ 0, list of layer indices | `"auto"` | layers whose keys and values stay in VRAM; `"auto"`: as many as fit in `vram_fraction` of the free VRAM at the first update; the offloaded ones are spread through the stack |
| `page_tokens` | integer ≥ 1 | `256` | tokens per host page (one D2H copy when a page fills) |
| `vram_fraction` | float in (0, 1] | `0.5` | share of free (plus cached but unused) VRAM the `"auto"` plan may fill |

Methods and attributes:
- `update(layer, k, v) → (keys, values)`: append `[batch, kv_heads, new_tokens, head_dim]` and get the layer's full keys and values, as strided views.
- `seq_length(layer=0)`: tokens cached in that layer.
- `reset()`: forget every token.
- `memory()`: `{"device_bytes", "host_bytes"}`.
- `.resident` and `.offloaded`: layer lists.
- `.stats`: `h2d_bytes`, `d2h_bytes`, `pages`, `prefetched` (slots filled ahead of time), `gathered` (filled on demand).

### Inspection and utilities

| Function | Arguments | Returns |
|---|---|---|
| `trainstream.memory_report(model=None)` | a wrapped model, or `None` | for a wrapped model, a dict with these keys: `"events"` (the decision log), `"gpu"` (`allocated`, `reserved`, `peak_allocated`), `"pinned"`, `"pinned_by_tag"`, `"transfers"`, `"optimizer"`, `"resident"`, and when used, `"attention"` (`mode`, `backend`, `timings_ms`), `"activations"` (`policy`, `offloaded_blocks`, `recompute_blocks`, `decision`, ...), `"params"` (`paged_blocks`, `quantize`, `quantized_blocks`, `outliers`, `load_bytes` from host RAM, `bounce_bytes` staged from pageable RAM, `peer_bytes` copied from other GPUs, `peer_blocks`, `peer_owned_bytes`, ...) and `"plan"` (`target_vram`, `settled`, `history`, `budget`). Without a model: `"gpu"`, `"pinned"`, `"pinned_by_tag"` |
| `trainstream.unwrap(model)` | a wrapped model | loads every weight back into VRAM, removes the hooks, returns the model |
| `trainstream.profile_hardware(device=None, quick=True, force=False, threads=None)` | `device`: GPU to measure (default the current one). `quick`: shorter benchmarks. `force`: re-measure instead of using the cache. `threads`: host threads for the AdamW benchmark | a `HardwareProfile` (PCIe bandwidth, matmul and host AdamW throughput; `.summary()` prints it) |
| `trainstream.pinned_stats()` | — | page-locked allocator counters, plus live bytes per tag in `"by_tag"`. Arena keys: `arena_bytes` (reserved), `arena_used`, `arena_slabs`, `arena_cap`, `arena_fallbacks` (blocks that had to be page-locked on their own), `arena_largest_free`; `host_alloc_calls` counts every page-locking call |
| `trainstream.reserve_pinned(bytes_now, cap=0, slab=0)` | `bytes_now`: page-lock this much for the arena now. `cap`: let it grow on demand up to this (at least `bytes_now`). `slab`: slab size in bytes (`0` keeps the current one) | the bytes reserved so far; `wrap(host_arena=...)` calls it for you |
| `trainstream.transfer_stats(device=None)` | `device`: GPU (default the current one) | cumulative bytes and copies on TrainStream's D2H and H2D streams |
| `trainstream.find_blocks(model)` | a model | the repeated layer stack `wrap()` would use (the largest `ModuleList` / `Sequential` of ≥ 2 same-type children), or `[]` |

The session behind a wrapped model is `model._trainstream_session` (`trainstream.TrainStreamSession`):
- `.adapting`: `True` while it is still tuning itself
- `.events`: the decision log
- `.config`: the options in effect
- `.steps`: optimizer steps so far
- `.peer`: the peer-GPU pools once set up (`.pooled_blocks`, `.bytes_owned`), else `None`

### `trainstream.ActivationOffloader(model, blocks, offload="auto", *, ...)`: activation paging on its own

`wrap()` creates this for you. Use it directly to page the activations of any model with a layer stack, without the rest of TrainStream. Call `.remove()` to take it off.

| Argument | Accepted values | Default | What it does |
|---|---|---|---|
| `model`, `blocks` | the model and its layer stack | — | whose saved activations are paged |
| `offload` | `"auto"`, `True`, `False`, fraction 0.0–1.0, set / list of block indices | `"auto"` | which blocks' activations go to host RAM; `"auto"` calibrates, then halves until the slowdown is within `max_slowdown` |
| `min_bytes` | bytes | `1048576` | smaller tensors stay on the GPU |
| `prefetch_depth` | integer ≥ 1 | `1` | blocks fetched back ahead of backward |
| `calibrate_steps` | integer ≥ 2 | `3` | steps measured with nothing paged before tuning |
| `profile` | `None`, `HardwareProfile` | `None` | link model used to size the first guess |
| `max_slowdown` | float | `0.15` | how much slower `"auto"` may make a step |
| `tune_window` | integer ≥ 1 | `3` | steps measured per setting while tuning (the first after a change is discarded) |
| `max_inflight_bytes` | bytes | `1073741824` | VRAM held by copies still crossing PCIe |
| `recompute` | `"auto"`, `True`, `False`, fraction, list of block indices | `"auto"` | blocks to recompute in backward instead of keeping or paging |
| `reserved_link_bytes`, `memory_pressure`, `vram_target` | set by `wrap()` | `(0, 0)`, `False`, `None` | PCIe already used by the optimizer per step; whether weights page; the VRAM target for the probe step |

Attributes: `.policy` (per block: `"keep"`, `"offload"` or `"recompute"`), `.decision`, `.stats`, `.events`.

`ParamPager`, `GradientOffloader`, `ParamReadyHooks` and `HardwareProfile` are also exported, and `trainstream.peer` holds the peer-GPU setup. `wrap()` and `OffloadAdamW` build and use them internally; they aren't meant to be constructed by hand.

### Environment variables

| Variable | Effect |
|---|---|
| `TRAINSTREAM_CACHE_DIR` | where builds are cached (default `%LOCALAPPDATA%\trainstream\build`; Linux `$XDG_CACHE_HOME/trainstream`, else `~/.cache/trainstream`) |
| `TRAINSTREAM_VERBOSE_BUILD=1` | show compiler output |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for (default: the GPUs present) |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit to use, if it isn't found automatically |
| `LOCAL_WORLD_SIZE` / `SLURM_NTASKS_PER_NODE` | processes sharing the machine; the planner splits RAM between them, and data parallel shares host memory when all ranks are local |
| `TRAINSTREAM_SHM=0` | data parallel without shared host memory (per-rank copies, gloo reduction); also turns off peer-GPU pooling, which uses the same local group |
| `TRAINSTREAM_HOST_ARENA_SLAB` | size of one pinned-arena slab, in bytes or with a `K` / `M` / `G` suffix (default `1G`) |

## How it works

1. **Measure.** At `wrap()`: free VRAM, available RAM, the page-lock budget, PCIe bandwidth, and the size of every layer.
2. **Plan.** Which layers stay in VRAM and which stream from RAM, the optimizer moment precision, whether to quantize the frozen weights, and how many activations to page. Runs that can't fit are refused with an explanation.
3. **Stream.** Gradients go to pinned host memory as autograd produces them; AdamW runs on host threads while backward continues; weights and activations come back on dedicated CUDA streams ordered by events, overlapping compute.
4. **Adapt.** The first steps' measured peaks move layers in or out of VRAM, then the plan settles; a watchdog reacts if a later step grows. If the model fits anyway, it switches to stock fused AdamW.

## Examples

| Script | What it does | Main flags |
|---|---|---|
| `examples/lora_large_model.py` | LoRA on bases larger than the GPU, bf16 or quantized | `--size 1.5b\|7b\|14b\|32b`, `--model HUB_ID`, `--quantize auto\|none\|int8\|nf4\|fp8\|nvfp4`, `--prefer exact\|speed`, `--recompute auto\|on\|off`, `--attention auto\|cudnn\|efficient\|flash\|math\|default`, `--outliers FRACTION`, `--stock`, `--steps`, `--batch`, `--seq`, `--rank` |
| `examples/dpo_lora.py` | DPO on a LoRA model: the base with adapters off is the reference, log-probs computed in chunks | `--model HUB_ID` (default: a random Qwen2.5-0.5B shape), `--quantize`, `--steps`, `--batch`, `--prompt`, `--answer`, `--beta`, `--rank` |
| `examples/hf_finetune.py` | fine-tune a Hugging Face model, optionally beside the stock Trainer | `--model`, `--lora`, `--compare`, `--activation-offload` |
| `examples/train_bigger_than_vram.py` | 1.8B model (7.3 GB of fp32 weights) inside a 4 GB VRAM cap | `--vram-gb`, `--moments fp32\|bf16`, `--layers`, `--d-model` |
| `examples/verify_ram_offload.py` | proves where training memory lives, with a PASS/FAIL verdict | `--gpt`, `--hf`, `--vram-cap-gb`, `--engine both\|baseline\|trainstream` |
| `benchmarks/compare.py` | head-to-head with stock PyTorch, bitsandbytes QLoRA, Unsloth, DeepSpeed | `--scenario full1.5b\|lora14b\|lora14b_2k`, `--engines ...`, `--steps` (2 warm-up + measured; default 10), `--timeout`, `--out` |
| `benchmarks/quality.py` | perplexity of each quantization format, before and after a short LoRA fine-tune | `--model`, `--formats none int8 nf4 fp8 nvfp4 nvfp4-w nvfp4-o` (`nvfp4-w`: 4-bit weights only, expanded before each matmul; `nvfp4-o`: tensor cores with `outliers=0.01`), `--windows`, `--train-steps` |

## Limitations

- Host RAM is the ceiling: 8–12 bytes per trained parameter and 2–4 per frozen one (1.06 with int8, 0.56 with NF4), plus activations. Measured on 64 GB: a 3.6B full fine-tune, and LoRA on a 32.8B NF4 base.
- Quantization covers frozen block weights only (not embeddings, the output layer or trained weights) and isn't bit-exact. FP8 / NVFP4 tensor-core matmuls need a Blackwell GPU and quantize activations too; `outliers=` recovers part of the activation error, never the weights' own 4-bit error. `load_model` reads `.safetensors` causal-LM checkpoints only.
- `PagedKVCache` / `TrainStreamCache`: inference only, full-attention decoders, no beam search. With paged weights, every decoded token also streams the paged blocks over PCIe; a larger generation batch (which the paged cache makes room for) amortizes that.
- Weight paging covers the repeated block stack and needs `optimizer_mode="cpu"`; each paged weight crosses PCIe twice per step.
- Data parallel only (no tensor or pipeline parallelism). Ranks on one machine share host memory; tested with 2 processes on 1 GPU, not on a multi-GPU machine or across machines. Peer-GPU pooling covers frozen weights only and needs CUDA IPC and GPUs with peer access (GeForce pairs usually have none); tested with two processes on one GPU (Windows 11, WSL2), copies between two real GPUs are untested.
- `TrainStreamTrainer` is single-process. fp16 with a GradScaler and AMSGrad aren't supported; `torch.compile` hasn't been tested.
- Host AdamW speed is bounded by system RAM bandwidth.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 John Anish G.
