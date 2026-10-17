# TrainStream

Train and fine-tune models larger than your GPU's memory by moving optimizer state, gradients and activations to system RAM over PCIe — with one line of code and no third-party dependencies.

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
| 85M GPT (fits either way) | 2.5 GiB, ~94k tokens/s | 1.8 GiB, ~64k tokens/s |

When a model already fits, TrainStream trades throughput for memory. Its value is training what does not fit, or freeing VRAM for larger batches. All numbers come from a single machine (RTX 5090 Laptop, Windows 11).

## Requirements

- An NVIDIA GPU and a CUDA build of PyTorch
- The CUDA toolkit matching PyTorch's CUDA major version, and a C++ compiler (Visual Studio 2022+ Build Tools with "Desktop development with C++" on Windows, GCC 11+ on Linux)
- Tested on Windows 11, PyTorch 2.14.1+cu130, CUDA 13.4, Python 3.14. Linux uses the same build path but is untested.

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

Hugging Face (`pip install transformers accelerate`):

```python
from trainstream.integrations.huggingface import TrainStreamTrainer
trainer = TrainStreamTrainer(model=model, args=TrainingArguments(..., bf16=True), train_dataset=ds)
trainer.train()
```

Use `bf16=True` or fp32; fp16 with loss scaling is not supported.

## Tools in this repository

- `examples/verify_ram_offload.py` — proves where training memory lives (VRAM vs system RAM) with a PASS/FAIL verdict
- `examples/hf_finetune.py` — fine-tune a Hugging Face model, optionally side by side with the stock Trainer
- `docs/index.html` — full documentation

## Limitations

- Weights stay in VRAM (no parameter offloading); large-vocabulary output logits are not offloaded
- Single GPU per process
- Host AdamW speed is bounded by system RAM bandwidth

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 John Anish G.
