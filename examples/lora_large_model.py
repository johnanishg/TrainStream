"""LoRA fine-tuning of a base model larger than the GPU, with TrainStreamTrainer.

    python examples/lora_large_model.py --size 7b               # Qwen2.5-7B architecture
    python examples/lora_large_model.py --size 14b              # Qwen2.5-14B: 29.5 GB of bf16 weights
    python examples/lora_large_model.py --size 7b --stock       # same run with the stock Trainer (all in VRAM)
    python examples/lora_large_model.py --model Qwen/Qwen2.5-7B # a real checkpoint (downloads it)

By default the model is built from the architecture's config with random
weights (no download): memory use and step time depend only on the shapes. It
is created on the CPU in bf16 and handed to TrainStreamTrainer, which keeps
the frozen decoder-layer weights in host RAM and streams each layer onto the
GPU while it runs; only the LoRA adapters are trained. The chunked loss keeps
the 152k-token vocabulary's logits from ever existing in full.

Host RAM needed: about 2 bytes per base parameter (bf16), plus activations.
Writes a JSON summary to --out.
"""

import argparse
import json
import os
import sys
import time

import torch

import trainstream
from trainstream.integrations.huggingface import TrainStreamTrainer

from transformers import AutoModelForCausalLM, Qwen2Config, Trainer, TrainerCallback, TrainingArguments

GB = 1e9
SIZES = {  # Qwen2.5 dense architectures
    "1.5b": dict(hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12,
                 num_key_value_heads=2, tie_word_embeddings=True),
    "7b": dict(hidden_size=3584, intermediate_size=18944, num_hidden_layers=28, num_attention_heads=28,
               num_key_value_heads=4, tie_word_embeddings=False),
    "14b": dict(hidden_size=5120, intermediate_size=13824, num_hidden_layers=48, num_attention_heads=40,
                num_key_value_heads=8, tie_word_embeddings=False),
}


def host_rss() -> int:
    if sys.platform == "win32":
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "a", "b", "c", "d", "e", "f")]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        pmc = PMC(cb=ctypes.sizeof(PMC))
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class Tokens(torch.utils.data.Dataset):
    def __init__(self, vocab, n=16, seq=512):
        g = torch.Generator().manual_seed(0)
        self.data = torch.randint(0, vocab, (n, seq), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"input_ids": self.data[i], "labels": self.data[i]}


class Meter(TrainerCallback):
    def __init__(self):
        self.rows = []
        self.t0 = None

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        row = dict(step=state.global_step, s=time.perf_counter() - self.t0,
                   vram_peak_gb=torch.cuda.max_memory_allocated() / GB, rss_gb=host_rss() / GB)
        self.rows.append(row)
        print(f"step {row['step']:3d}  {row['s']:6.2f} s  peak VRAM {row['vram_peak_gb']:5.2f} GB  "
              f"process RAM {row['rss_gb']:5.1f} GB", flush=True)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs and self.rows:
            self.rows[-1]["loss"] = logs["loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="7b", choices=SIZES)
    ap.add_argument("--model", default=None, help="a Hugging Face checkpoint instead of random weights")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--stock", action="store_true", help="stock Trainer with the model in VRAM, for comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import peft

    t0 = time.perf_counter()
    if args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        name = args.model
    else:
        cfg = Qwen2Config(vocab_size=152064, max_position_embeddings=32768, rope_theta=1e6, **SIZES[args.size])
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        name = f"Qwen2.5-{args.size.upper()} architecture, random weights"
    n_base = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_base / 1e9:.2f}B parameters, {n_base * 2 / GB:.1f} GB in bf16 "
          f"(built in {time.perf_counter() - t0:.0f} s)", flush=True)
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={args.rank}: {trainable / 1e6:.1f}M trainable parameters", flush=True)
    if args.stock:
        model = model.cuda()

    targs = TrainingArguments(output_dir=os.path.join(os.path.dirname(__file__), "..", "runs", "lora_large"),
                              per_device_train_batch_size=args.batch, max_steps=args.steps, learning_rate=2e-4,
                              bf16=True, logging_steps=1, report_to="none", save_strategy="no",
                              optim="adamw_torch", disable_tqdm=True, dataloader_num_workers=0)
    meter = Meter()
    ds = Tokens(model.config.vocab_size if hasattr(model, "config") else 152064, seq=args.seq)
    if args.stock:
        trainer = Trainer(model=model, args=targs, train_dataset=ds, callbacks=[meter])
    else:
        trainer = TrainStreamTrainer(model=model, args=targs, train_dataset=ds, callbacks=[meter])
    status = "finished"
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:
        status = "out of memory: " + str(exc).splitlines()[0][:120]
        print(status)
    tail = meter.rows[2:] or meter.rows
    summary = dict(model=name, base_params=n_base, trainable=trainable, engine="stock" if args.stock else "trainstream",
                   batch=args.batch, seq=args.seq, status=status, steps=len(meter.rows),
                   step_s=sorted(r["s"] for r in tail)[len(tail) // 2] if tail else None,
                   vram_peak_gb=max((r["vram_peak_gb"] for r in meter.rows), default=None),
                   rss_peak_gb=max((r["rss_gb"] for r in meter.rows), default=None),
                   losses=[r.get("loss") for r in meter.rows])
    if not args.stock and status == "finished":
        session = trainstream.wrapper.unwrap_session(model)
        if session is not None and session.pager is not None:
            st = session.stats()["params"]
            summary.update(paged_blocks=len(st["paged_blocks"]), frozen_pinned_gb=st["frozen_pinned_bytes"] / GB,
                           frozen_pageable_gb=st["frozen_pageable_bytes"] / GB)
    print(json.dumps(summary, indent=1))
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
