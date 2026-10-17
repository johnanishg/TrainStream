# TrainStream

Train and fine-tune models larger than your GPU's memory by moving optimizer state, gradients, activations and, when needed, the weights themselves to system RAM over PCIe. Frozen weights (a LoRA base) can also be stored as int8 or 4-bit NF4. It takes one line of code and has no third-party dependencies.

```python
import trainstream
model, optimizer = trainstream.wrap(model, optimizer)
```

TrainStream is a C++/CUDA extension for PyTorch. It keeps only the weights in VRAM: gradients stream to pinned host memory as autograd produces them, AdamW runs on host threads while the backward pass continues, and a profiled share of activations is paged out and prefetched back block by block. Updates match `torch.optim.AdamW` (max parameter difference ≈3·10⁻⁷ after 20 steps); activation paging is bit-exact.

## Results (RTX 5090 Laptop GPU, 24 GB)

| Workload | Stock PyTorch AdamW | TrainStream |
|---|---|---|
| 1.8B-parameter GPT, fp32, batch 8×512 | 34.1 GiB peak — exceeds VRAM, driver spills to RAM, 13.7 s/step | 20.6 GiB, 1.1 s/step (13.5 GiB with all activations paged) |
| Qwen2.5-1.5B full fine-tune (Hugging Face Trainer) | 35.9 GiB peak (spill), 111.7 s / 10 steps | 18.7 GiB, 28.5 s / 10 steps, same losses |
| 1.8B GPT, weights paged to RAM (0.2) | — | 2.4 GiB peak, 2.1 s/step; 1.84 GiB inside a 4 GB VRAM cap with bf16 moments |
| 3.6B GPT full fine-tune (all parameters trained) | needs ~58 GB of VRAM | 3.1 GiB peak, 3.15 s/step, 30 GiB of host RAM |
| LoRA r=16 on the Qwen2.5-7B architecture (random weights) | 20.3 GB peak, 0.27 s/step | 4.6 GB peak, 1.25 s/step |
| LoRA r=16 on Qwen2.5-14B (29.5 GB of weights) | does not fit | 1.37 s/step: the memory planner keeps 28 of 48 layers in VRAM (19.9 GiB peak) and streams 20 from RAM |
| same, NF4 base (`quantize="auto", prefer="speed"`, 0.4) | does not fit | 0.87 s/step: all 48 layers in VRAM as 4-bit (7.43 GB), 18.3 GB peak, 11.3 GB of process RAM |
| LoRA r=16 on the Qwen2.5-32B architecture (65.5 GB in bf16, more than the 64 GB of RAM) | does not fit | 2.53 s/step with `load_model(quantize="nf4")`: 21.3 GB peak, 31.4 GB of process RAM |
| 85M GPT (fits either way) | 2.55 GiB, 43.7 ms/step | `offload="auto"` switches to stock fused AdamW: 2.56 GiB, 43.8 ms/step |

Offloading costs throughput when a model already fits, which is why `offload="auto"` measures two steps and then switches to plain fused GPU AdamW if offloading isn't needed. TrainStream's value is training what does not fit, or freeing VRAM for larger batches. All numbers come from a single machine (RTX 5090 Laptop, Windows 11).

## Requirements

- An NVIDIA GPU and a CUDA build of PyTorch
- The CUDA toolkit matching PyTorch's CUDA major version, and a C++ compiler (Visual Studio 2022+ Build Tools with "Desktop development with C++" on Windows, GCC 11+ on Linux)
- Tested on Windows 11 (PyTorch 2.14.1, CUDA 13.4, MSVC, Python 3.14) and Ubuntu 24.04 under WSL2 (PyTorch 2.14.1+cu130, CUDA 13.3, GCC 13.3, Python 3.12): all 123 tests pass on Windows; on Linux 122 pass and 1 (the bitsandbytes comparison) is skipped without bitsandbytes installed, RTX 5090 Laptop GPU. Native Linux and other GPUs are untested.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # if you don't have CUDA PyTorch yet
pip install trainstream
```

The first `import trainstream` compiles the CUDA extension against your PyTorch (a few minutes, once) and caches it; every later import is instant. On Windows it finds Visual Studio and the CUDA toolkit itself, so any terminal works. The cache is keyed by PyTorch/CUDA/Python version and GPU, so upgrading PyTorch triggers one rebuild.

| Variable | Effect |
|---|---|
| `TRAINSTREAM_CACHE_DIR` | where builds are cached (default `%LOCALAPPDATA%\trainstream\build`, Linux `~/.cache/trainstream`) |
| `TRAINSTREAM_VERBOSE_BUILD=1` | show compiler output |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for (default: the GPUs present) |
| `CUDA_HOME` | CUDA toolkit to use, if it isn't found automatically |

To compile ahead of time instead: `pip install trainstream --no-binary trainstream --no-build-isolation`. From a checkout: `build.bat` (Windows) or `python setup.py build_ext --inplace`.

## Usage

```python
import torch, trainstream

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model, optimizer = trainstream.wrap(model, optimizer)   # max_grad_norm=..., accumulation_steps=... if you use them

for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x.cuda(), y.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Models bigger than the GPU: build on the CPU and let TrainStream page the transformer-block weights:

```python
model = MyGPT(cfg)                                  # stays on the CPU
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model, optimizer = trainstream.wrap(model, optimizer, device="cuda",
                                    param_offload=True, moment_dtype="bfloat16")
```

LoRA on a quantized base (QLoRA-style), with no extra package:

```python
import peft
from trainstream.integrations.huggingface import TrainStreamTrainer, load_model

model = load_model("Qwen/Qwen2.5-14B", quantize="nf4")   # "auto" | "none" | "int8" | "nf4"; quantized while read
model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
trainer = TrainStreamTrainer(model=model, args=TrainingArguments(..., bf16=True), train_dataset=ds)
trainer.train()
```

Or quantize inside `wrap()`: `trainstream.wrap(model, opt, device="cuda", quantize="auto", prefer="speed")`, or beforehand with `trainstream.quantize_model(model, "int8")`. Only frozen weights are quantized. A quantized run trains on the dequantized weights (as QLoRA does), so it isn't bit-exact with the bf16 base.

`examples/train_bigger_than_vram.py` trains a 1.8B model (7.3 GB of fp32 weights) inside a 4 GB VRAM cap. `examples/lora_large_model.py` LoRA-fine-tunes 7B–32B bases through `TrainStreamTrainer` (`--quantize auto|none|int8|nf4`, `--prefer exact|speed`).

| `wrap()` option | What it does |
|---|---|
| `offload="auto"` | switches to stock fused AdamW after 2 steps if the model fits (`True` always offloads) |
| `param_offload="auto"` | memory planner: measures the VRAM and RAM of the machine it runs on, keeps as many transformer blocks resident as fit, pages the rest, and refines the split from the first steps' measured peaks; `True`, a fraction, or block indices fix it |
| `activation_offload="auto"` | pages a profiled share of activations (`True`, a fraction, `False`) |
| `quantize="auto"` | frozen weights only: `"auto"` quantizes when the run can't fit otherwise (int8 before NF4); `"none"`, `"int8"`, `"nf4"` fix it; `quant_block_size=64` |
| `prefer="exact"` | `"speed"` lets `quantize="auto"` pick NF4 when it keeps every layer in VRAM |
| `moment_dtype="auto"` | fp32 optimizer moments, or bf16 (stochastic rounding, half the memory) when fp32 wouldn't fit in RAM |
| `check_memory=True` | refuse at `wrap()` with a clear explanation when the run can't fit in RAM/VRAM |
| `distributed` | data parallel when a `torch.distributed` group with more than 1 rank exists; gradients averaged on the host over gloo |
| `max_grad_norm`, `accumulation_steps` | clipping and gradient accumulation, done inside the optimizer |

Also: `trainstream.chunked_cross_entropy(hidden, weight, targets)` computes the loss without materializing the logits. `.to()`, `.cuda()` and `.half()` work after `wrap()`, and reading a weight anywhere waits for its in-flight update.

Hugging Face (`pip install transformers accelerate`):

```python
from trainstream.integrations.huggingface import TrainStreamTrainer
trainer = TrainStreamTrainer(model=model, args=TrainingArguments(..., bf16=True), train_dataset=ds)
trainer.train()
```

`TrainStreamTrainer` computes the causal-LM loss in chunks, so the full logits tensor never exists. Pass extra `wrap()` options as `trainstream={...}`. Use `bf16=True` or fp32; fp16 with loss scaling is not supported.

## Tools in this repository

- `examples/verify_ram_offload.py` — proves where training memory lives (VRAM vs system RAM) with a PASS/FAIL verdict
- `examples/hf_finetune.py` — fine-tune a Hugging Face model, optionally side by side with the stock Trainer
- `examples/train_bigger_than_vram.py` — weight paging on a model built on the CPU, under a VRAM cap
- `examples/lora_large_model.py` — LoRA on 7B/14B/32B-size bases larger than the GPU, bf16 or quantized, optionally against the stock Trainer
- `TrainStreamDocs.html` — full documentation, including every option and the values it accepts (open it in a browser)
- `TrainStreamReport.html` — what TrainStream does, how it compares with other libraries, and every measured result

## Limitations

- Host RAM is the ceiling: 8–12 bytes per trained parameter and 2–4 per frozen one (1.06 int8, 0.56 NF4), plus activations; measured on 64 GB: a 3.6B full fine-tune and LoRA on a 32.8B NF4 base
- Quantization covers frozen block weights only (not embeddings, the output layer or trained weights) and is not bit-exact; `load_model` reads `.safetensors` causal-LM checkpoints
- Weight paging covers the repeated block stack and needs `optimizer_mode="cpu"`; each paged weight crosses PCIe twice per step
- Data parallel only (no tensor/pipeline parallelism); tested with 2 processes on 1 GPU, not across several GPUs or machines
- `TrainStreamTrainer` is single-process; fp16 + GradScaler and AMSGrad aren't supported; `torch.compile` hasn't been tested
- Host AdamW speed is bounded by system RAM bandwidth

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 John Anish G.
