# TrainStream

Train and fine-tune models larger than your GPU's memory. TrainStream moves optimizer state, gradients, activations and, when needed, the weights themselves into system RAM, and streams them back over PCIe exactly when the GPU needs them. Frozen weights (a LoRA base) can also be stored as int8, NF4, FP8 or NVFP4, the last two multiplied directly by Blackwell tensor cores. It takes one line of code and has no third-party dependencies.

```python
import trainstream
model, optimizer = trainstream.wrap(model, optimizer)
```

TrainStream is a C++/CUDA extension for PyTorch. It uses only the C++ standard library, the CUDA runtime and PyTorch's ATen: no DeepSpeed, no bitsandbytes. Updates match `torch.optim.AdamW` (max parameter difference ≈3·10⁻⁷ after 20 steps), and activation and weight paging are bit-exact.

## What's new in 0.7

Measured in the full check of 11 November 2026 on an RTX 5090 Laptop GPU (24 GB), in WSL2 (49 GB VM) and on Windows 11 (64 GB). Every number is in the tables below.

- **Faster on 4-bit bases.** Qwen2.5-14B LoRA, TrainStream against the fastest rival that trains the same adapters:
  - NVFP4 base: 1.30× bitsandbytes at 512 tokens (WSL; 1.25× on Windows), 1.74× Unsloth at 2,048 tokens (WSL; 2.10× bitsandbytes with checkpointing on Windows), 1.61× bitsandbytes + Liger on packed documents (1.73× bitsandbytes with checkpointing on Windows).
  - NF4 base, the format bitsandbytes and Unsloth use: 1.13× bitsandbytes at 512 tokens (1.18× on Windows), 1.26× Unsloth at 2,048 tokens, 1.15× bitsandbytes + Liger on packed documents (1.20× bitsandbytes with checkpointing on Windows). In 0.6 TrainStream's NF4 was 7 % slower than bitsandbytes at 512 tokens.
  - Held to 12 GB of VRAM: level with Unsloth (NF4 0.834 s against 0.819 s, inside the 4 % its three runs varied).
- **Fused block kernels** (`kernels=`). One CUDA kernel each for RMSNorm (forward and backward; the weight gradient is summed in a fixed order, so it is the same every run), rotary embedding, and the gated MLP activation (silu / gelu-tanh, keeping only its two inputs for backward). Hugging Face modules are patched only where a check against their own code agrees; anything else is left alone. The fused RMSNorm matches PyTorch's bit for bit; the fused SwiGLU and RoPE round once instead of at every step, so they are closer to an fp32 reference than PyTorch's own code (relative error 1.7·10⁻³ against 2.3·10⁻³, and 1.6·10⁻³ against 2.2·10⁻³). On by default for quantized bases, off for bf16, which stays bit-exact.
- **Fused LoRA** (`fused_lora=`): two matmuls forward and three backward per adapted Linear, with no fp32 round trip of the input and no copy of it kept for the adapter's gradient.
- **The same activation quantized once.** q / k / v (and gate / up) read the same input; on the FP8 / NVFP4 tensor-core path it is quantized once for all of them.
- **Padding-free packing** (`trainstream.packing`, `TrainStreamTrainer(trainstream={"packing": N})`). Variable-length documents are packed into full rows, best fit first; position ids restart per document, so attention stays inside each one and no padding is computed.
- **Quality options**, all opt-in. On Qwen2.5-1.5B, after 40 LoRA steps, three of them beat their plain format by more than runs vary; on Qwen2.5-14B every difference is within run-to-run variation.
  - `keep_precise=`: the matrices quantization hurts most stay int8 (NF4 base) or FP8 (NVFP4 base) within a share of the bytes. 1.5B: +4.29 % against NF4's +5.04 % after fine-tuning.
  - `error_rank=`: LoftQ; the LoRA adapters start at a low-rank fit of the quantization error. 1.5B: NF4 +4.10 % against +5.04 %; NVFP4 +8.71 % against +9.84 %.
  - `double_quant=`: block scales stored in one byte, with an fp32 scale per 256 blocks (4× smaller scales, about 0.37 bits less per weight); perplexity within run-to-run variation of NF4's.
- **8-bit optimizer moments** (`moment_dtype="int8"`, CPU optimizer): blockwise 8-bit moments with stochastic rounding, about 6 host bytes per trained parameter instead of 12. `"auto"` picks it when bf16 moments wouldn't fit. In the test, parameters drift from fp32 AdamW's by 5 % of their total movement (bitsandbytes' AdamW8bit: 4.1 % on the same test).
- **Disk tier** (`disk_offload=`, also in `load_model`): frozen paged blocks that host RAM can't hold live in memory-mapped files and stream from there.
- **Opt-in experiments:**
  - `nvfp4_backward=`: the input gradient on FP4 tensor cores too. 14B: training step 6–8 % faster, perplexity after fine-tuning within run-to-run variation. 1.5B: not faster (1 % faster to 9 % slower across two full checks).
  - `moe_paging=`: frozen MoE experts stay in host RAM and only the experts the router picked are copied to the GPU. Covered by tests, not yet benchmarked on a real MoE model.
  - NVIDIA ModelOpt NVFP4 checkpoints load as stored (`load_model` detects them). Covered by tests.
- **Fixes:**
  - **Host threads no longer spin while they wait for the GPU** (`TRAINSTREAM_EVENT_WAIT`). The CPU optimizer's threads wait for gradient copies hundreds of times a step; with CUDA's default wait each kept a core at 100 % (6–10 cores in all). Waits that sleep until the GPU signals left the cores idle, but on this laptop under Windows the GPU then drew about 70 W instead of 148 W under the same 175 W limit, and a 14B LoRA step (bf16 base, streamed) took 1.87–2.04 s instead of 1.11–1.13 s; under WSL those waits spun anyway. TrainStream now checks the event for 50 µs, then sleeps 100 µs between checks: 1.13–1.20 s with 0.8–0.9 cores busy on Windows, 1.11 s with 1.4 cores under WSL.
  - **Activation paging leaves host RAM a margin.** It took up to 60 % of the RAM available and ignored what the pinned staging arena still needed. Under WSL (49 GB VM) a 14B LoRA run at 2,048 tokens paged 38 blocks' activations (about 17.6 GB, page-locked), reached 36 GB of process RAM and was stopped without an error message, most likely the VM running out of memory. Now the arena's growth counts, 10 % of physical RAM (at least 4 GiB) stays free, and if available RAM falls under that during training, half of the paged blocks are recomputed instead. The same run: 29 blocks paged, 32.1 GB peak, 3.23 s per step (3.41 s before).
  - **Broadcast attention masks with activation paging.** A packed batch's `[B, 1, T, T]` mask is a broadcast view; paging it changed its strides and the memory-efficient attention backward failed ("last dimension must be contiguous"). Broadcast tensors now stay in VRAM, where they cost almost nothing. Inputs with a mask or position ids also keep PyTorch's own attention backend order.
  - **A partial `load_state_dict()` with quantized weights.** With `strict=False` and only some keys (how PEFT loads adapters, including when the Hugging Face Trainer resumes a LoRA run), every quantized weight of a block was re-quantized from its expansion; on FP8 / NVFP4 tensor-core weights that expansion was uninitialized memory, so the base was corrupted. Now only the weights the dict contains are re-read.
  - **Faster NF4 and int8 expansion.** 8 values per thread and one 16-byte store: a 13824 × 5120 weight expands to bf16 in 0.285 ms (NF4) and 0.320 ms (int8) instead of 0.466 / 0.449 ms, bit-identical.
- **Benchmarks that hold up** (`benchmarks/compare.py`, `benchmarks/quality.py`): `--repeat N` (a full check runs close comparisons 3 times and counts a win only beyond the spread); bitsandbytes + Liger kernels, and Unsloth with its own checkpointing, as rivals; a 12 GB-VRAM scenario, a packed-documents scenario and a same-format (NF4 against NF4) comparison; where each engine's time goes (`--profile N`, `--profile-ops`); and the GPU's clock, power draw and enforced power limit for every run. Profiler traces are checked against the wall clock before they are used.
- **Tried and dropped: Hadamard rotation of quantized weights.** It is exact before quantization and helped on synthetic weights with outlier columns (NF4 error 14.6 → 8.9 %), but it made every format worse on Qwen2.5 in three full checks (NF4 +3.09 → +3.45 % on 14B, +5.91 → +7.02 % on 1.5B; NVFP4 +5.20 → +7.09 % and +10.96 → +14.66 %). It isn't offered.

## What's new in 0.6

- **Outlier channels in bf16 for NVFP4** (`outliers=`). A few hidden dimensions of an LLM's activations are far larger than the rest. In NVFP4, one of them sets the scale of its 16-value block and rounds the other 15 to zero. With `outliers=0.01`, each matmul picks its largest 1% of input channels on the GPU (no host sync). It quantizes them as zero and multiplies them in bf16 against the same weight columns: `y = scaled_mm(nvfp4(x without them), W) + x[:, idx] @ W[:, idx].T`. The input gradient is unchanged. This removes the activation part of NVFP4's error, not the weight part: the weights stay 4-bit. `benchmarks/quality.py --formats nvfp4-w nvfp4 nvfp4-o` measures the split (`nvfp4-w` = 4-bit weights only). Measured on held-out text (table below): on Qwen2.5-14B NVFP4 costs +5.20 % perplexity, +2.65 % from the 4-bit weights and the rest from the activations; `outliers=0.01` brings it to +4.44 %; after 20 LoRA steps the difference is within run-to-run variation (+4.64 % against +4.48 % in one run, +4.14 % against +4.59 % in the next). On Qwen2.5-1.5B it helps before and after fine-tuning (+10.96 → +9.21 %; +10.02 → +8.54 % and +9.82 → +8.42 % in two runs). It makes the training step 3 % slower on 14B and 10 % on 1.5B. With that mixed result it stays off by default.
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
- **Activation tuning, corrected and tighter.**
  - Activations that fit in VRAM now stay there. Before, `"auto"` still paged some to free VRAM that nothing used: the 1.5B full fine-tune spent 12 slower tuning steps, then settled on 4 paged blocks at 1,078 ms/step against 1,020 ms with none.
  - When host RAM can't hold every block's activations, the trials now apply the setting they measured fastest. Before, the forced recompute shifted it: 14B NF4 at 2,048 tokens ran 44 recomputed + 3 paged blocks (never measured) instead of the measured 47 recomputed; 14B int8 at 2,048 ran 47 recomputed (3.25 s) instead of the measured 36 (3.17 s).
  - Once the trials settle, the VRAM left under the target keeps the latest blocks' activations instead of recomputing or paging them, measured on a clean step.
  - Measured with the same harness in WSL2, before → after: 1.5B full fine-tune 1.03 → 0.99 s/step; 14B LoRA at 512 tokens with `"auto"` 0.90 → 0.84 s; at 2,048 tokens NVFP4 2.14 → 2.06 s, NF4 3.18 → 3.02 s, `"auto"` 3.28 → 3.20 s. Single runs: differences under about 5 % are within run-to-run noise. The faster 2,048-token runs use more of the VRAM under the target (NF4: 17.8 → 20.4 GB peak).
- **Benchmarks on both systems, every format.** The comparison below now covers WSL2 and Windows 11, with every engine run to the end, and the perplexity table covers every format on Qwen2.5-14B and 1.5B, before and after fine-tuning. `benchmarks/quality.py --format F` runs one format in its own process, and `--live` streams its progress as JSON lines.

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

## Against other libraries (same laptop: WSL2 with a 49 GB VM, then Windows 11 with 64 GB)

> TrainStream 0.7.0, full check of 11 November 2026. Each engine runs in a fresh process, 10 steps after its warm-up (TrainStream: after it has finished tuning itself), and its step time is the median of those. Where two compared engines were within 10 %, both ran 3 times and the median run counts; a win counts only when the gap is wider than the runs' spread. Every engine listed finished all its steps. Between full checks the laptop's speed moves by 5–10 % for every engine alike, so read smaller differences as noise.

**LoRA r=16, Qwen2.5-14B, batch 1 × 512 (WSL2)**

| Engine | Step time | Peak VRAM | Against the fastest |
|---|---|---|---|
| TrainStream, NVFP4 base (FP4 tensor cores) | 0.530 s | 16.8 GB | fastest |
| TrainStream, NF4 base | 0.609 s | 16.8 GB | 1.15× |
| bitsandbytes QLoRA (NF4) + PEFT | 0.687 s | 17.4 GB | 1.30× |
| Unsloth (4-bit) | 0.778 s | 11.5 GB | 1.47× |
| bitsandbytes QLoRA + Liger kernels | 0.909 s | 12.0 GB | 1.72× |
| bitsandbytes QLoRA + gradient checkpointing | 0.988 s | 12.7 GB | 1.86× |
| TrainStream (auto: exact bf16 base, 20 of 48 layers streamed) | 1.079 s | 21.6 GB | 2.04× |

**LoRA r=16, Qwen2.5-14B, batch 1 × 2048 (WSL2)**

| Engine | Step time | Peak VRAM | Against the fastest |
|---|---|---|---|
| TrainStream, NVFP4 base | 1.57 s | 18.7 GB | fastest |
| TrainStream, NF4 base | 2.17 s | 19.2 GB | 1.38× |
| Unsloth (4-bit) | 2.73 s | 11.7 GB | 1.74× |
| bitsandbytes QLoRA + Liger kernels | 2.88 s | 13.0 GB | 1.84× |
| bitsandbytes QLoRA + gradient checkpointing | 3.10 s | 16.3 GB | 1.97× |
| TrainStream (auto) | 3.17 s | 23.6 GB | 2.02× |

**LoRA r=16, Qwen2.5-14B, batch 1 × 512, every engine held to 12 GB of VRAM (WSL2)**

| Engine | Step time | Peak VRAM | Against the fastest |
|---|---|---|---|
| Unsloth (4-bit) | 0.819 s (3 runs: 0.798–0.820) | 11.4 GB | fastest |
| TrainStream, NF4 base | 0.834 s (3 runs: 0.822–0.859) | 11.8 GB | 1.02×, within the spread |
| TrainStream, NVFP4 base | 0.869 s (3 runs: 0.752–0.878) | 11.8 GB | 1.06×, within the spread |
| bitsandbytes QLoRA + Liger kernels | 0.921 s | 12.0 GB | 1.12× |
| TrainStream (auto) | 2.15 s | 11.7 GB | 2.62× |

bitsandbytes QLoRA, with or without gradient checkpointing, ran out of memory under the 12 GB cap.

**LoRA r=16, Qwen2.5-14B, documents of 64–2,048 tokens packed into 2 rows of 2,048 (WSL2)**

| Engine | Step time | Peak VRAM | Against the fastest |
|---|---|---|---|
| TrainStream, NVFP4 base | 4.15 s | 19.5 GB | fastest |
| TrainStream, NF4 base | 5.80 s | 19.7 GB | 1.40× |
| bitsandbytes QLoRA + Liger kernels | 6.68 s | 14.4 GB | 1.61× |
| bitsandbytes QLoRA + gradient checkpointing | 7.24 s | 21.0 GB | 1.74× |
| TrainStream (auto) | 7.35 s | 21.2 GB | 1.77× |

Unsloth isn't compared here: it patches attention itself, so the harness can't check that attention stays inside each packed document.

**Full fine-tune, Qwen2.5-1.5B, batch 4 × 512 (WSL2)**

| Engine | Step time | Peak VRAM | Against the fastest |
|---|---|---|---|
| TrainStream (auto) | 1.23 s | 18.8 GB | fastest |
| Stock PyTorch | 1.85 s | 32.4 GB | 1.51× (spills past VRAM) |
| DeepSpeed ZeRO-2, optimizer offload | 2.12 s | 13.3 GB | 1.73× |

**The same on Windows 11 (64 GB RAM; Unsloth, Liger and DeepSpeed aren't installed there)**

| Engine | LoRA 14B, 1 × 512 | LoRA 14B, 1 × 2048 | 14B, 12 GB VRAM | 14B, packed | Full fine-tune 1.5B |
|---|---|---|---|---|---|
| TrainStream, NVFP4 base | **0.541 s** | **1.50 s** | **0.731 s** | **4.00 s** | — |
| TrainStream, NF4 base | 0.571 s | 2.16 s | 0.773 s | 5.76 s | — |
| TrainStream (auto) | 1.08 s | 3.07 s | 1.70 s | 7.11 s | **1.10 s** |
| bitsandbytes QLoRA (NF4) + PEFT | 0.674 s | spills past VRAM | out of memory | left out (hit the time limit in an earlier run) | — |
| bitsandbytes QLoRA + gradient checkpointing | 0.942 s | 3.16 s | out of memory | 6.91 s | — |
| Stock PyTorch | spills past VRAM | spills past VRAM | — | — | 1.63 s (spills) |

- With an NVFP4 base TrainStream had the fastest step in every 14B scenario, in WSL and on Windows, except under the 12 GB cap in WSL, where Unsloth and TrainStream's NF4 and NVFP4 are within each other's run-to-run spread. NVFP4 costs quality (see the perplexity table below).
- With an NF4 base, the format bitsandbytes and Unsloth use, TrainStream was ahead of every rival in every scenario except under the 12 GB cap, where it is level with Unsloth.
- When it has room, TrainStream holds more in VRAM than Unsloth (16.8 GB against 11.5 GB at 512 tokens) because it plans up to a VRAM target; held to 12 GB it fits like Unsloth does.
- `"auto"` with 49–64 GB of RAM keeps the 14B base exact in bf16 and streams 20–26 of its 48 layers from RAM. That is slower than the 4-bit engines and changes no weight.
- The full fine-tune: TrainStream keeps its activations in VRAM and stays at 18.8 GB, 1.51× faster than stock PyTorch in WSL and 1.48× on Windows. Stock PyTorch peaks at 32.4 GB, more than the GPU's 24 GB, so its step time depends on how the driver spills into system RAM.
- Stock PyTorch and DeepSpeed in the 14B scenarios, and bitsandbytes without checkpointing at 2,048 tokens, don't fit in 24 GB of VRAM: in earlier runs the driver spilled into system RAM at 46–371 s per step, so the full check leaves them out.

**Perplexity against bf16** (`benchmarks/quality.py`, held-out Python standard-library source; lower is better)

| Format | Qwen2.5-14B | after 20 LoRA steps | Qwen2.5-1.5B | after 40 LoRA steps |
|---|---|---|---|---|
| int8 | −0.00 % | +0.08 % | +0.02 % | +0.05 % |
| FP8 (tensor cores) | +0.89 % | +0.72 % | +1.70 % | +1.44 % |
| NF4 | +3.10 % | +2.88 % | +5.94 % | +5.04 % |
| NF4, `keep_precise="auto"` | +2.89 % | +2.70 % | +5.27 % | +4.29 % |
| NF4, `error_rank=16` (LoftQ) | +3.06 % | +2.81 % | +5.76 % | +4.10 % |
| NF4, `double_quant=True` | +2.79 % | +2.72 % | +5.96 % | +4.97 % |
| NVFP4, weights only (bf16 matmuls) | +2.63 % | +2.36 % | +5.42 % | +5.06 % |
| NVFP4 (tensor cores) | +5.19 % | +5.01 % | +10.86 % | +9.84 % |
| NVFP4 with `outliers=0.01` | +4.78 % | +4.21 % | +8.99 % | +8.27 % |
| NVFP4, `error_rank=16` (LoftQ) | +5.14 % | +4.91 % | +10.85 % | +8.71 % |
| NVFP4, `nvfp4_backward=True` | +5.19 % | +4.70 % | +10.86 % | +10.52 % |

bf16 itself: 2.300 (14B) and 3.251 (1.5B) on 16 and 64 windows of 1,024 tokens. NVFP4's 4-bit weights alone cost less than NF4; its extra loss comes from quantizing the activations. On 1.5B the short fine-tune raised bf16's own held-out perplexity (3.251 → 3.305), so compare the formats with each other there. The columns before fine-tuning repeat exactly from run to run. The ones after it don't, since training on the GPU isn't bit-exact: across the saved runs the NF4 formats moved by up to 0.2 points and the NVFP4 formats by up to about 1 point, so differences that small there are noise.

Each engine runs in a fresh process with the same model, random-token data (the packed scenario: random documents of 64–2,048 tokens), LoRA settings (r=16, all projections), learning rate and loop. Each computes its loss its own recommended way: TrainStream in chunks from the hidden states, Liger with its fused linear cross-entropy, Unsloth with its own, the others through the model. Re-run with `benchmarks/compare.py` and `benchmarks/quality.py`.

## Results (RTX 5090 Laptop GPU 24 GB, 64 GB RAM)

Measured with TrainStream 0.1–0.4, when each feature arrived: these rows show what fits and how, not today's speed (the comparison above is 0.7.0).

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

Tested on an RTX 5090 Laptop GPU with Windows 11 (PyTorch 2.14.1, CUDA 13.4, MSVC, Python 3.14), where all 307 tests pass with no warnings, and Ubuntu 24.04 under WSL2 (PyTorch 2.14.1, CUDA 13.3, GCC 13.3, Python 3.12), where all 307 pass as well (with bitsandbytes 0.50.2 installed for the NF4 comparison), both from a wheel installed with pip and with uv. Installs were checked with pip and with uv 0.13 (`uv pip install`, `uv sync` in a project, and building from the sdist) on both. Native Linux and other GPUs are untested.

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
- Under WSL2 the Linux VM gets only part of the machine's RAM (about half by default), and TrainStream plans from what the VM has. If host RAM is the limit (the events say "host RAM holds N of M blocks' activations"), raise `memory=` in `%UserProfile%\.wslconfig`: in 0.6's benchmarks, NF4 at 2,048 tokens ran 3.02 s/step in WSL with 31 GB against 2.57 s on Windows with 64 GB; with a 49 GB VM both ran 2.16–2.17 s in 0.7.

## Options

### `trainstream.wrap(model, optimizer=None, **options) → (model, optimizer)`

| Option | Accepted values | Default | What it does |
|---|---|---|---|
| `optimizer` | `torch.optim.AdamW`, `torch.optim.Adam`, `trainstream.OffloadAdamW`, `None` | `None` | converted, with hyper-parameters and state migrated; `None` creates one with `lr` |
| `optimizer_mode` | `"cpu"`, `"gpu"`, `"auto"` | `"cpu"` | host AdamW, a zero-copy CUDA kernel, or chosen from the hardware profile |
| `offload` | `"auto"`, `True` | `"auto"` | `"auto"` switches to fused torch AdamW after 2 steps if the model fits |
| `activation_offload` | `"auto"`, `True`, `False`, fraction 0.0–1.0 | `"auto"` | how many blocks' activations are paged to RAM; `"auto"`: measured in the first step, kept in VRAM if they fit under the VRAM target, otherwise paged as far as host RAM holds them (see `recompute` for the rest): at most 60 % of the RAM available, after what the pinned staging arena may still grow by, always leaving 10 % of physical RAM (at least 4 GiB) free; if available RAM later falls under that floor, half of the paged blocks are recomputed instead; `True`: all but the last block |
| `param_offload` | `"auto"`, `True`, `False`, fraction 0.0–1.0, list of block indices | `"auto"` | `"auto"`: the memory planner decides which layers stay in VRAM |
| `quantize` | `"auto"`, `"none"`, `None`, `False`, `"int8"`, `"nf4"`, `"fp8"`, `"nvfp4"` | `"auto"` | frozen weights only; `"auto"` quantizes only when the run can't fit otherwise (int8 before NF4); fp8 / nvfp4 run on Blackwell tensor cores |
| `quant_block_size` | even integer ≥ 2 | `64` | values per fp32 scale (int8 / nf4; fp8 and nvfp4 use fixed 32 / 16) |
| `prefer` | `"exact"`, `"speed"` | `"exact"` | `"speed"` lets `quantize="auto"` choose NF4 when it keeps every layer in VRAM |
| `recompute` | `"auto"`, `True`, `False`, fraction 0.0–1.0, list of block indices | `"auto"` | recompute blocks in backward instead of keeping or paging their activations; `"auto"`: when the activations can't all stay in VRAM (weights paged, or too large), as many as measured trials show is fastest, then blocks go back to VRAM as far as the room left under the target allows |
| `attention` | `"auto"`, `"cudnn"`, `"efficient"`, `"flash"`, `"math"`, `None` | `"auto"` | attention backend: the fastest measured on your model's shape, a named one first, or PyTorch's own order; deterministic mode leaves out the non-deterministic kernels |
| `outliers` | `0`, `False`, `None`, `"auto"`, `True`, float 0.0–0.1 | `0` | with `quantize="nvfp4"` on tensor cores: the fraction of input channels (the largest of each call) multiplied in bf16 instead of 4-bit; `"auto"` / `True` mean `0.01`; ignored by other formats |
| `nvfp4_backward` | `True`, `False` | `False` | with `quantize="nvfp4"` on tensor cores: the input gradient of each frozen matmul in NVFP4 too (the gradient and the transposed weight quantized along the reduced dimension, round to nearest) instead of bf16. Faster backward matmuls, about 12 % relative error in those gradients (0.2 % in bf16); measure it with `quality.py --formats nvfp4 nvfp4-b` |
| `kernels` | `"auto"`, `True`, `False`, `None` | `"auto"` | fused CUDA kernels for each block's RMSNorm, rotary embedding and gated MLP activation (`trainstream.fused`); each is checked once against the model's own code and used only where they agree. They compute in fp32 and round once, so results differ from PyTorch's in the last bits: `"auto"` turns them on when the frozen weights are quantized (already not bit-exact), off otherwise |
| `fused_lora` | `"auto"`, `True`, `False`, `None` | `"auto"` | PEFT LoRA adapters (no dropout, DoRA or adapter bias) computed by one autograd function: two matmuls forward, the second accumulating into the base output, three backward, keeping the input the base read instead of a copy (`trainstream.lora`). Matches PEFT up to the rounding of its intermediate products; `"auto"`: on when the frozen weights are quantized |
| `keep_precise` | `0`, `False`, `None`, `"auto"`, `True`, fraction up to 0.5 | `0` | with `quantize="nf4"` / `"nvfp4"`: the weights that lose the most in that format (relative error, measured one by one) are stored one step more precise, int8 / fp8, as long as the extra bytes stay within this fraction of the quantized weights' size; `"auto"` / `True` mean `0.05` |
| `error_rank` | integer ≥ 0 | `0` | LoftQ: quantizing a weight also fits a rank-`error_rank` approximation of what quantization lost, and the PEFT LoRA adapters on it start there instead of at zero, so base + adapter starts close to the original weights (use the LoRA rank). Weights quantized beforehand with `error_rank` start their adapters the same way |
| `error_iters` | integer ≥ 1 | `1` | LoftQ rounds: alternate quantizing `W - BA` and fitting `BA` to what that lost |
| `double_quant` | `True`, `False` | `False` | int8 / NF4 block scales stored as one-byte log codes with an fp32 scale per 256 blocks (decoded to the same values on the GPU and in `dequantize`): about 0.37 bit per weight less, 0.65 GB on a 14B model, at no measurable error |
| `disk_offload` | `None`, a directory | `None` | frozen paged weights the page-lock budget can't hold go to memory-mapped files there (`trainstream.disk`) instead of ordinary RAM: the OS keeps them in its page cache while RAM has room and reads them from the disk when it doesn't, and the RAM check stops counting them. Use a fast SSD; the folder is removed at exit |
| `moe_paging` | `True`, `False` | `False` | the frozen experts of mixture-of-experts layers (Hugging Face's stacked experts: Qwen3-MoE, Qwen2-MoE, Mixtral…) stay in pinned host RAM and each layer copies in only the experts its tokens were routed to, again in backward (`trainstream.experts`). Pays when a step routes to few experts; `memory_report()["experts"]` counts what was copied |
| `host_arena` | `"auto"`, `False`, `None`, bytes | `"auto"` | page-locked host memory reserved up front, from which paged activations, staging and other pinned buffers are carved; `"auto"`: when the run streams anything, reserve one slab now and grow on demand within the page-lock budget; bytes: reserve that much now and cap growth there |
| `peer` | `"auto"`, `False`, `None`, bytes | `"auto"` | data parallel on one machine: pool the frozen blocks every rank streams in the GPUs' VRAM and copy them GPU to GPU (needs peer access between the GPUs and CUDA IPC; set up collectively at step 10); `"auto"`: when peer copies reach 0.9× the host link, each GPU gives half the headroom under the memory plan's target; bytes: that much per GPU. While active, `unwrap()` is collective |
| `moment_dtype` | `"auto"`, `"float32"`, `"bfloat16"`, `"bf16"`, `torch.bfloat16`, `"int8"` | `"auto"` | fp32 unless that wouldn't fit in RAM, then bf16 (stochastic rounding), then 8-bit with the host optimizer; `"int8"`: one byte per moment value (see `OffloadAdamW`), a quarter of fp32's RAM |
| `device` | `None`, `"cuda"`, `"cuda:N"`, `torch.device` | `None` | target GPU when the model is on the CPU |
| `blocks` | `None`, list of layer modules | `None` | the repeated layer stack; auto-detected |
| `distributed` | `None`, `True`, `False` | `None` | data parallel when a process group with > 1 rank exists |
| `guard_params` | `True`, `False` | `True` | reads of a weight wait for its in-flight update |
| `max_grad_norm` | `None`, float | `None` | global-norm clipping inside the optimizer |
| `accumulation_steps` | integer ≥ 1 | `1` | backward passes per `step()` |
| `vram_headroom` | float 0.0–1.0 | `0.10` | share of usable VRAM the planner leaves free |
| `check_memory` | `True`, `False` | `True` | raise `MemoryError` at `wrap()` when the run can't fit |
| `max_inflight_bytes` | `None`, bytes | `None` (1 GiB or 10% of VRAM) | VRAM held by transfers still crossing PCIe |
| `max_activation_slowdown` | float | `0.15` | how much slower `"auto"` activation paging may make a step when there is no VRAM target to measure against (`ActivationOffloader` used on its own); `wrap()` always has one, so activations that fit stay in VRAM instead |
| `prefetch_depth` | integer ≥ 1 | `1` | activation blocks fetched ahead in backward |
| `min_offload_bytes` | bytes | `1048576` | smaller activations stay on the GPU |
| `threads` | `None`, integer | `None` (all cores) | host AdamW threads |
| `profile` | `True`, `False` | `True` | run (or load cached) hardware benchmarks |
| `lr` | float | `1e-3` | used only when `optimizer` is `None` |
| `verbose` | `True`, `False` | `False` | print decisions as they happen |

Anything else raises `ValueError` (for example `quantize="fp4"`).

### Quantization formats

| `quantize=` | Bytes per weight | Quality cost (perplexity against bf16: Qwen2.5-14B / 1.5B) | Use it when |
|---|---|---|---|
| `"none"` | 2 (bf16) | none (bit-exact) | it fits |
| `"int8"` | 1.06 | −0.00 % / +0.02 % (mean weight error 0.6 %) | bf16 doesn't fit in RAM |
| `"nf4"` | 0.56 | +3.1 % / +5.9 % (mean weight error 9 %) | int8 doesn't fit either, or with `prefer="speed"` |
| `"fp8"` | 1.03 | +0.9 % / +1.7 % | FP8 tensor-core matmuls on Blackwell, near-bf16 quality |
| `"nvfp4"` | 0.56 | +5.2 % / +10.9 % (+4.8 % / +9.0 % with `outliers=0.01`) | the fastest 4-bit path on Blackwell, when its quality is acceptable |
| `"auto"` | — | — | decided from the measured RAM and VRAM |

NF4 is bit-identical to bitsandbytes' NF4 (same codes, packing and scales); TrainStream doesn't depend on bitsandbytes. A quantized run trains on the dequantized weights, as QLoRA does, so it isn't bit-exact with the bf16 base.

### `load_model(name_or_path, ...)` (in `trainstream.integrations.huggingface`)

| Argument | Accepted values | Default |
|---|---|---|
| `name_or_path` | Hub id or local folder with `.safetensors` files | — |
| `quantize` | `"auto"`, `"none"`, `None`, `False`, `"int8"`, `"nf4"`, `"fp8"`, `"nvfp4"` | `"auto"` (quantize only if bf16 wouldn't fit in `ram_fraction` of free RAM) |
| `dtype` | `torch.bfloat16`, `torch.float16`, `torch.float32` | `torch.bfloat16` |
| `quant_block_size` | even integer ≥ 2 | `64` |
| `keep_precise` | as `wrap()` | `0` (the chosen matrices are read again from the checkpoint and stored int8 / fp8) |
| `error_rank`, `error_iters` | as `wrap()` | `0`, `1` (LoftQ factors kept with each quantized matrix; `wrap()` starts the adapters at them) |
| `double_quant` | `True`, `False` | `False` (as `wrap()`) |
| `disk_offload` | `None`, a directory | `None` (decoder-layer matrices that aren't quantized are written to memory-mapped files there as they're read, so a base larger than RAM loads) |
| `random_weights` | `True`, `False` | `False` (build the shapes without reading a checkpoint) |
| `ram_fraction` | float 0.0–1.0 | `0.85` |
| `trust_remote_code` | `True`, `False` | `False` |
| `**config_overrides` | any config field, e.g. `num_hidden_layers=4` | — |

An NVFP4 checkpoint exported by NVIDIA ModelOpt (`"quant_method": "modelopt", "quant_algo": "NVFP4"` in its config) is read as it is stored: the packed codes and e4m3 scales become TrainStream NVFP4 weights without being expanded or quantized again (`quantize` is ignored for it; matrices the export kept in 16 bits stay so). Tested on a checkpoint written in that layout, not yet on one exported by ModelOpt itself.

### `trainstream.quantize_model(model, quantize="nf4", *, blocks=None, block_size=64, pin_budget=None, keep_precise=0, error_rank=0, error_iters=1, double_quant=False)`

Quantizes the frozen block weights in place, one tensor at a time, before `wrap()`. Returns `{"tensors": count, "kept_precise": count, "bytes_before": ..., "bytes_after": ...}`.

| Argument | Accepted values | Default | What it does |
|---|---|---|---|
| `model` | `nn.Module` | — | the model; only frozen (`requires_grad=False`) weights of the block stack are quantized |
| `quantize` | `"int8"`, `"nf4"`, `"fp8"`, `"nvfp4"` | `"nf4"` | the storage format (see the table above) |
| `blocks` | `None`, list of layer modules | `None` | the layer stack to quantize; auto-detected with `find_blocks` |
| `block_size` | even integer ≥ 2 | `64` | values per scale for int8 / NF4 (FP8 and NVFP4 use fixed 32 / 16) |
| `pin_budget` | `None`, bytes | `None` (40 % of physical RAM) | how much of the quantized data may be page-locked; the rest stays pageable |
| `keep_precise` | as `wrap()` | `0` | store the weights that lose the most one step more precise, within that fraction of extra bytes |
| `error_rank`, `error_iters` | as `wrap()` | `0`, `1` | keep LoftQ factors with each weight for `init_lora_from_error` / `wrap()` |
| `double_quant` | `True`, `False` | `False` | one-byte int8 / NF4 block scales |

`trainstream.init_lora_from_error(model) → int` starts every PEFT LoRA adapter whose base weight carries LoftQ factors at that weight's quantization error (`lora_B = B / scaling`, `lora_A = A`, their first r components) and returns how many it set; `wrap()` calls it for you.

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
| `moment_dtype` | `"float32"`, `"fp32"`, `"bfloat16"`, `"bf16"`, `torch.float32`, `torch.bfloat16`, `"int8"`, `"8bit"` | `"float32"` | bf16 halves moment memory and traffic, with stochastic rounding. `"int8"` (`"cpu"` mode only) keeps one byte per value, a small float relative to an fp32 absmax per 256 values (`exp_avg_sq` as its square root, stochastic rounding; `exp_avg` rounded to nearest): 2 bytes per parameter instead of 8. After 30 steps of a test MLP the weights are within 5 % of how far fp32 AdamW moved them (bitsandbytes' `AdamW8bit`: 4 %). `state_dict()` hands the moments out as fp32 |
| `distributed` | `True`, `False` | `False` | data parallel: broadcast from rank 0, gradients averaged on the host (`"cpu"` mode only) |

Methods and attributes: `step(closure=None)`, `zero_grad()`, `attach(model, guard=True)` (install the streaming hooks; `wrap()` does this), `detach()`, `synchronize()` (block until every queued update has landed), `stats()`, `state_dict()` / `load_state_dict()` (compatible with `torch.optim.AdamW`, plus the fp32 `master` weights), `add_param_group()`, `last_grad_norm`.

### `trainstream.integrations.huggingface`

| Function / class | Arguments | What it does |
|---|---|---|
| `TrainStreamTrainer(*args, trainstream=None, **kwargs)` | everything `transformers.Trainer` takes, plus `trainstream`: a dict of any `wrap()` option and `"chunked_loss"` (`True`/`False`, default `True`: the causal-LM loss is computed in chunks so the logits never exist), `"loss_chunk_size"` (tokens per chunk, default `1024`) and `"packing"` (`False`, `True` or a row length in tokens, default `False`: each batch's examples packed into rows with no padding, see `trainstream.packing`; `True`: rows as long as the batch's longest example) | a `Trainer` whose optimizer state, gradients and activations live in host RAM |
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

### Padding-free packing: `trainstream.packing`

| Function / class | Arguments | What it does |
|---|---|---|
| `pack_sequences(sequences, max_len, *, labels=None, pad_id=0)` | `sequences`: token lists or 1-D tensors; `max_len`: row length; `labels`: per-sequence labels of the same lengths (e.g. `-100` over a prompt), default the tokens; `pad_id`: the padding token | best-fit-decreasing packing into rows of `max_len` (longer sequences are split): `input_ids`, `position_ids` (restarting at every sequence) and `labels` (`-100` at each sequence's first token and at padding), each `[rows, max_len]`, and `num_tokens`. Call the model with `use_cache=False`: transformers keeps attention inside each sequence only without a KV cache |
| `PackedCollator(max_len=None, pad_id=0)` | `max_len`: row length, `None`: the batch's longest example | a `Trainer` data collator that packs each batch (examples with `input_ids`, optionally `labels`) and passes `use_cache=False` |
| `best_fit(lengths, max_len)` | sequence lengths and a row length | the rows, as lists of indices into `lengths` |

A packed batch gives the same loss and gradients as running its sequences one by one (tested on Llama).

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
- `.adapting`: `True` while it is still tuning itself. Meanwhile it may reset PyTorch's peak-memory counter (`torch.cuda.reset_peak_memory_stats`) to measure single steps; once it is `False`, the counter is left alone
- `.events`: the decision log
- `.config`: the options in effect
- `.steps`: optimizer steps so far
- `.peer`: the peer-GPU pools once set up (`.pooled_blocks`, `.bytes_owned`), else `None`

### `trainstream.ActivationOffloader(model, blocks, offload="auto", *, ...)`: activation paging on its own

`wrap()` creates this for you. Use it directly to page the activations of any model with a layer stack, without the rest of TrainStream. Call `.remove()` to take it off.

| Argument | Accepted values | Default | What it does |
|---|---|---|---|
| `model`, `blocks` | the model and its layer stack | — | whose saved activations are paged |
| `offload` | `"auto"`, `True`, `False`, fraction 0.0–1.0, set / list of block indices | `"auto"` | which blocks' activations go to host RAM; `"auto"` with `vram_target`: a probe step measures them, they stay in VRAM if they fit, otherwise host RAM takes as many as it holds and the rest are recomputed; `"auto"` without: calibrates, then halves until the slowdown is within `max_slowdown` (never more blocks than host RAM holds) |
| `min_bytes` | bytes | `1048576` | smaller tensors stay on the GPU |
| `prefetch_depth` | integer ≥ 1 | `1` | blocks fetched back ahead of backward |
| `calibrate_steps` | integer ≥ 2 | `3` | steps measured with nothing paged before tuning (`"auto"` without `vram_target`) |
| `profile` | `None`, `HardwareProfile` | `None` | link model used to size the first guess |
| `max_slowdown` | float | `0.15` | how much slower `"auto"` may make a step (without `vram_target`) |
| `tune_window` | integer ≥ 1 | `3` | steps measured per setting while tuning (the first after a change is discarded) |
| `max_inflight_bytes` | bytes | `1073741824` | VRAM held by copies still crossing PCIe |
| `recompute` | `"auto"`, `True`, `False`, fraction, list of block indices | `"auto"` | blocks to recompute in backward instead of keeping or paging |
| `reserved_link_bytes`, `memory_pressure`, `vram_target` | set by `wrap()` | `(0, 0)`, `False`, `None` | PCIe already used by the optimizer per step; whether weights page; the VRAM target for the probe step |

Attributes: `.policy` (per block: `"keep"`, `"offload"` or `"recompute"`), `.decision` (what the tuner measured and chose; `kept_blocks` lists blocks moved back to VRAM later), `.stats`, `.events`, `.block_bytes` (measured saved activations of one block), `.adjustable` (`True` when `"auto"` chose the policy, so `keep_resident` may change it), `.weights_paged` (set by `wrap()`: a callable saying whether weights stream at the moment; it words the probe's event).

`.keep_resident(k)`: keep the activations of up to `k` blocks that `"auto"` pages or recomputes in VRAM instead, the latest first (backward reaches them first). Does nothing for policies you set explicitly. Returns the blocks moved. `wrap()` calls it once the policy is final, with as many blocks as the room left under its VRAM target holds.

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
| `TRAINSTREAM_EVENT_WAIT` | how TrainStream's host threads wait for the GPU: `poll` (default) checks for `TRAINSTREAM_EVENT_SPIN_US` microseconds (default 50), then sleeps `TRAINSTREAM_EVENT_SLEEP_US` (default 100) between checks; `spin` keeps a core busy until the GPU is done (CUDA's default; the CPU optimizer's threads then held 6–10 cores at 100 %); `block` sleeps until the GPU signals (on an RTX 5090 Laptop under Windows the GPU then drew about half its power and a 14B LoRA step took 1.8× longer; WSL spins anyway) |
| `TRAINSTREAM_EVENT_SPIN_US` / `TRAINSTREAM_EVENT_SLEEP_US` | with `poll`: how long to check before sleeping, and how long each sleep is, in microseconds |

## How it works

1. **Measure.** At `wrap()`: free VRAM, available RAM, the page-lock budget, PCIe bandwidth, and the size of every layer.
2. **Plan.** Which layers stay in VRAM and which stream from RAM, the optimizer moment precision, whether to quantize the frozen weights, and how many activations to page. Runs that can't fit are refused with an explanation.
3. **Stream.** Gradients go to pinned host memory as autograd produces them; AdamW runs on host threads while backward continues; weights and activations come back on dedicated CUDA streams ordered by events, overlapping compute.
4. **Adapt.** The first steps' measured peaks move layers in or out of VRAM, then the plan settles; a watchdog reacts if a later step grows. Activations stay in VRAM if they fit; otherwise measured trials choose which blocks are paged or recomputed, and the room left afterwards keeps the latest ones in VRAM. If the model fits anyway, it switches to stock fused AdamW.

## Examples

| Script | What it does | Main flags |
|---|---|---|
| `examples/lora_large_model.py` | LoRA on bases larger than the GPU, bf16 or quantized | `--size 1.5b\|7b\|14b\|32b`, `--model HUB_ID`, `--quantize auto\|none\|int8\|nf4\|fp8\|nvfp4`, `--prefer exact\|speed`, `--recompute auto\|on\|off`, `--attention auto\|cudnn\|efficient\|flash\|math\|default`, `--outliers FRACTION`, `--stock`, `--steps`, `--batch`, `--seq`, `--rank` |
| `examples/dpo_lora.py` | DPO on a LoRA model: the base with adapters off is the reference, log-probs computed in chunks | `--model HUB_ID` (default: a random Qwen2.5-0.5B shape), `--quantize`, `--steps`, `--batch`, `--prompt`, `--answer`, `--beta`, `--rank` |
| `examples/hf_finetune.py` | fine-tune a Hugging Face model, optionally beside the stock Trainer | `--model`, `--lora`, `--compare`, `--activation-offload` |
| `examples/train_bigger_than_vram.py` | 1.8B model (7.3 GB of fp32 weights) inside a 4 GB VRAM cap | `--vram-gb`, `--moments fp32\|bf16`, `--layers`, `--d-model` |
| `examples/verify_ram_offload.py` | proves where training memory lives, with a PASS/FAIL verdict | `--gpt`, `--hf`, `--vram-cap-gb`, `--engine both\|baseline\|trainstream` |
| `benchmarks/compare.py` | head-to-head with stock PyTorch, bitsandbytes QLoRA, Unsloth, DeepSpeed; each result has the GPU's median clock while measured | `--scenario full1.5b\|lora14b\|lora14b_2k`, `--engines ...`, `--steps` (2 warm-up + measured; default 10), `--timeout`, `--out`, `--profile N` (N more steps per engine traced with `torch.profiler`: GPU time per kernel category, host↔GPU copies, GPU idle; engines slower than 60 s per step are skipped; default 0), `--profile-ops` (with `--profile N`: N more steps traced with the CPU-side operations and their input shapes, listing the operations whose kernels took the most GPU time), `--repeat N` (each engine N times in fresh processes, reporting the median run and the spread; engines slower than 10 s per step run once; default 1). Each result also has `cpu_cores`, the CPU cores the engine kept busy over its measured steps. Scenarios also `lora14b_12gb` (every engine held to 12 GB of VRAM), `lora8b_llama` (Llama-3.1-8B, when downloaded) and `lora14b_var` (documents of 64–2,048 tokens packed into 2 rows of 2,048, the same for every engine; not Unsloth, which patches attention itself); engine `bnb_liger` (bitsandbytes NF4 with Liger's Triton kernels and checkpointing; Unsloth runs with `use_gradient_checkpointing="unsloth"`) |
| `benchmarks/quality.py` | perplexity of each quantization format, before and after a short LoRA fine-tune, with the steady training step time and the GPU's median clock while it trained | `--model`, `--formats none int8 nf4 fp8 nvfp4 nvfp4-w nvfp4-o nf4-k nf4-l nvfp4-l nf4-dq nvfp4-b` (`nvfp4-w`: 4-bit weights only, expanded before each matmul; `nvfp4-o`: tensor cores with `outliers=0.01`; `nf4-k`: `keep_precise="auto"`; `nf4-l` / `nvfp4-l`: LoRA started at the quantization error, `error_rank=16`; `nf4-dq`: `double_quant=True`; `nvfp4-b`: `nvfp4_backward=True`), `--windows`, `--seq`, `--train-steps`, `--batch`, `--lr`, `--out`; `--format F` runs one format in this process, `--live` with it streams progress as `LIVE {json}` lines and the result as `RESULT {json}` |

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
