# TrainStream

Train and fine-tune models larger than your GPU's memory. TrainStream moves optimizer state, gradients, activations and, when needed, the weights themselves into system RAM, and streams them back over PCIe exactly when the GPU needs them. Frozen weights (a LoRA base) can also be stored as int8 or 4-bit NF4. It takes one line of code and has no third-party dependencies.

```python
import trainstream
model, optimizer = trainstream.wrap(model, optimizer)
```

TrainStream is a C++/CUDA extension for PyTorch. It uses only the C++ standard library, the CUDA runtime and PyTorch's ATen: no DeepSpeed, no bitsandbytes. Updates match `torch.optim.AdamW` (max parameter difference ≈3·10⁻⁷ after 20 steps), and activation and weight paging are bit-exact.

## What's new in 0.4.0

- **Built-in weight quantization.** `wrap(..., quantize="auto" | "none" | "int8" | "nf4")` stores frozen weights quantized and expands them with TrainStream's own CUDA kernel just before each layer runs. `"auto"` (the default) quantizes only when a run can't fit otherwise; `prefer="speed"` also lets it choose NF4 when that keeps every layer in VRAM.
- **`load_model()`** reads a Hugging Face checkpoint one tensor at a time and quantizes while reading, so a base larger than your RAM never exists in full precision.
- **`trainstream.quantize_model()`** quantizes a model you've already loaded, in place.
- Real Qwen2.5-14B LoRA on a 24 GB laptop GPU: **1.37 → 0.87 s/step**. A 32B-shaped base (65.5 GB in bf16, more than the machine's 64 GB of RAM) trains at 2.53 s/step.

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
- Optional, for the Hugging Face integration: `transformers`, `accelerate`, and `peft` for LoRA

Tested on an RTX 5090 Laptop GPU with Windows 11 (PyTorch 2.14.1, CUDA 13.4, MSVC, Python 3.14), where all 123 tests pass, and Ubuntu 24.04 under WSL2 (PyTorch 2.14.1+cu130, CUDA 13.3, GCC 13.3, Python 3.12), where 122 pass and 1 is skipped (the bitsandbytes comparison, without bitsandbytes installed). Native Linux and other GPUs are untested.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # if you don't have CUDA PyTorch yet
pip install trainstream
```

The first `import trainstream` compiles the CUDA extension against your PyTorch (16 s on Windows and 18 s on Linux, measured) and caches it; later imports take about a second. On Windows it finds Visual Studio and the CUDA toolkit itself, so any terminal works. The cache is keyed by PyTorch, CUDA and Python version and GPU, so upgrading PyTorch triggers one rebuild. To upgrade TrainStream: `pip install -U --no-cache-dir trainstream`.

To compile ahead of time instead: `pip install trainstream --no-binary trainstream --no-build-isolation`. From a checkout: `build.bat` (Windows) or `python setup.py build_ext --inplace`.

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

### Inspecting what it decided

```python
r = trainstream.memory_report(model)
r["events"]                    # plain-text log: quantization, memory plan, resident mode, .to()
r["params"]["paged_blocks"]    # layers whose weights stream from host RAM
r["params"]["quantize"]        # None, "int8" or "nf4"
r["plan"]["history"]           # every planner move, with the measured peaks
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
| `quantize` | `"auto"`, `"none"`, `None`, `False`, `"int8"`, `"nf4"` | `"auto"` | frozen weights only; `"auto"` quantizes only when the run can't fit otherwise (int8 before NF4) |
| `quant_block_size` | even integer ≥ 2 | `64` | values per fp32 scale |
| `prefer` | `"exact"`, `"speed"` | `"exact"` | `"speed"` lets `quantize="auto"` choose NF4 when it keeps every layer in VRAM |
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
| `"auto"` | — | — | decided from the measured RAM and VRAM |

NF4 is bit-identical to bitsandbytes' NF4 (same codes, packing and scales); TrainStream doesn't depend on bitsandbytes. A quantized run trains on the dequantized weights, as QLoRA does, so it isn't bit-exact with the bf16 base.

### `load_model(name_or_path, ...)` (in `trainstream.integrations.huggingface`)

| Argument | Accepted values | Default |
|---|---|---|
| `name_or_path` | Hub id or local folder with `.safetensors` files | — |
| `quantize` | `"auto"`, `"none"`, `None`, `False`, `"int8"`, `"nf4"` | `"auto"` (quantize only if bf16 wouldn't fit in `ram_fraction` of free RAM) |
| `dtype` | `torch.bfloat16`, `torch.float16`, `torch.float32` | `torch.bfloat16` |
| `quant_block_size` | even integer ≥ 2 | `64` |
| `random_weights` | `True`, `False` | `False` (build the shapes without reading a checkpoint) |
| `ram_fraction` | float 0.0–1.0 | `0.85` |
| `trust_remote_code` | `True`, `False` | `False` |
| `**config_overrides` | any config field, e.g. `num_hidden_layers=4` | — |

### Other functions

| Function | Arguments |
|---|---|
| `trainstream.quantize_model(model, quantize="nf4", *, blocks=None, block_size=64, pin_budget=None)` | `quantize`: `"int8"`, `"nf4"`; returns `{"tensors", "bytes_before", "bytes_after"}` |
| `TrainStreamTrainer(..., trainstream={...})` | any `wrap()` option, plus `"chunked_loss"` (`True`/`False`, default `True`) and `"loss_chunk_size"` (default `1024`) |
| `trainstream.OffloadAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, *, mode="cpu", max_grad_norm=None, accumulation_steps=1, moment_dtype="float32", ...)` | `mode`: `"cpu"`, `"gpu"`; `moment_dtype`: `"float32"`, `"bfloat16"` |
| `trainstream.chunked_cross_entropy(hidden, weight, targets, bias=None, *, chunk_size=1024, ignore_index=-100, reduction="mean", num_items=None)` | `reduction`: `"mean"`, `"sum"`; `causal_lm_loss` takes the same and shifts labels |
| `trainstream.profile_hardware(device=None, quick=True, force=False)` | returns a `HardwareProfile` |
| `trainstream.memory_report(model)`, `trainstream.unwrap(model)` | a wrapped model |

The full reference, with every argument of every function, is in `TrainStreamDocs.html`.

### Environment variables

| Variable | Effect |
|---|---|
| `TRAINSTREAM_CACHE_DIR` | where builds are cached (default `%LOCALAPPDATA%\trainstream\build`, Linux `~/.cache/trainstream`) |
| `TRAINSTREAM_VERBOSE_BUILD=1` | show compiler output |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for (default: the GPUs present) |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit to use, if it isn't found automatically |
| `LOCAL_WORLD_SIZE` / `SLURM_NTASKS_PER_NODE` | processes sharing the machine; the planner splits RAM between them |

## How it works

1. **Measure.** At `wrap()`: free VRAM, available RAM, the page-lock budget, PCIe bandwidth, and the size of every layer.
2. **Plan.** Which layers stay in VRAM and which stream from RAM, the optimizer moment precision, whether to quantize the frozen weights, and how many activations to page. Runs that can't fit are refused with an explanation.
3. **Stream.** Gradients go to pinned host memory as autograd produces them; AdamW runs on host threads while backward continues; weights and activations come back on dedicated CUDA streams ordered by events, overlapping compute.
4. **Adapt.** The first steps' measured peaks move layers in or out of VRAM, then the plan settles; a watchdog reacts if a later step grows. If the model fits anyway, it switches to stock fused AdamW.

## Examples

| Script | What it does | Main flags |
|---|---|---|
| `examples/lora_large_model.py` | LoRA on bases larger than the GPU, bf16 or quantized | `--size 1.5b\|7b\|14b\|32b`, `--model HUB_ID`, `--quantize auto\|none\|int8\|nf4`, `--prefer exact\|speed`, `--stock`, `--steps`, `--batch`, `--seq`, `--rank` |
| `examples/hf_finetune.py` | fine-tune a Hugging Face model, optionally beside the stock Trainer | `--model`, `--lora`, `--compare`, `--activation-offload` |
| `examples/train_bigger_than_vram.py` | 1.8B model (7.3 GB of fp32 weights) inside a 4 GB VRAM cap | `--vram-gb`, `--moments fp32\|bf16`, `--layers`, `--d-model` |
| `examples/verify_ram_offload.py` | proves where training memory lives, with a PASS/FAIL verdict | `--gpt`, `--hf`, `--vram-cap-gb`, `--engine both\|baseline\|trainstream` |

Also in the repository: `TrainStreamDocs.html` (full documentation, including every option and the values it accepts) and `TrainStreamReport.html` (what TrainStream does, how it compares with other libraries, and every measured result). Open them in a browser.

## Limitations

- Host RAM is the ceiling: 8–12 bytes per trained parameter and 2–4 per frozen one (1.06 with int8, 0.56 with NF4), plus activations. Measured on 64 GB: a 3.6B full fine-tune, and LoRA on a 32.8B NF4 base.
- Quantization covers frozen block weights only (not embeddings, the output layer or trained weights) and isn't bit-exact. `load_model` reads `.safetensors` causal-LM checkpoints only.
- Weight paging covers the repeated block stack and needs `optimizer_mode="cpu"`; each paged weight crosses PCIe twice per step.
- Data parallel only (no tensor or pipeline parallelism); tested with 2 processes on 1 GPU, not across several GPUs or machines.
- `TrainStreamTrainer` is single-process. fp16 with a GradScaler and AMSGrad aren't supported; `torch.compile` hasn't been tested.
- Host AdamW speed is bounded by system RAM bandwidth.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 John Anish G.
