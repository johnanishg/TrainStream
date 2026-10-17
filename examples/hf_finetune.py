"""Fine-tune a pretrained Hugging Face causal LM with TrainStream.

    python examples/hf_finetune.py                                   # Qwen2.5-0.5B, full fine-tune
    python examples/hf_finetune.py --model Qwen/Qwen2.5-1.5B --steps 30
    python examples/hf_finetune.py --model Qwen/Qwen2.5-1.5B --lora   # PEFT LoRA
    python examples/hf_finetune.py --compare                          # also run the stock Trainer

The training text is Python standard-library source (always available
offline), so the task is "continue writing Python": loss drops quickly and the
before/after samples show it. Swap `build_dataset` for your own data.

Requires: pip install transformers accelerate   (and peft for --lora)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from trainstream.integrations.huggingface import TrainStreamTrainer  # noqa: E402

import _live  # noqa: E402  (optional live dashboard)

GiB = float(1 << 30)


def stdlib_text(nbytes: int) -> str:
    root = os.path.join(sys.base_prefix, "Lib")
    chunks, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("site-packages", "test", "tests", "__pycache__"))
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
                    chunks.append(fh.read())
                total += len(chunks[-1])
                if total >= nbytes:
                    return "\n".join(chunks)
    return "\n".join(chunks)


class BlockDataset(torch.utils.data.Dataset):
    """Fixed-length token blocks for causal-LM training."""

    def __init__(self, ids: torch.Tensor, block: int):
        n = (len(ids) - 1) // block
        self.blocks = ids[: n * block].view(n, block)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        x = self.blocks[i]
        return {"input_ids": x, "labels": x.clone()}


def build_dataset(tok, block: int, nbytes: int):
    ids = torch.tensor(tok(stdlib_text(nbytes))["input_ids"], dtype=torch.long)
    return BlockDataset(ids, block)


@torch.no_grad()
def sample(model, tok, prompt="def merge_sorted(a, b):\n", n=60):
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    model.train()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.lora:
        import peft
        model = peft.get_peft_model(model, peft.LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return model


def run(args, use_trainstream: bool, ds, tok):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_model(args)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\n=== {'TrainStreamTrainer' if use_trainstream else 'stock Trainer'} · {args.model} · "
          f"{n_all / 1e9:.2f}B params ({n_train / 1e6:.1f}M trainable) ===")
    if not args.compare or use_trainstream:
        print("before:", repr(sample(model, tok)[:160]))
    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "trainstream" if use_trainstream else "stock"),
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        max_steps=args.steps, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, args.steps // 10),
        weight_decay=0.0, max_grad_norm=1.0, bf16=True, logging_steps=1,
        report_to="none", save_strategy="no", optim="adamw_torch", seed=0, data_seed=0,
        dataloader_num_workers=0, disable_tqdm=True)
    cls = TrainStreamTrainer if use_trainstream else Trainer
    kw = {"trainstream": {"activation_offload": args.activation_offload}} if use_trainstream else {}
    live = _live.open_run(args, f"hf-{args.model.split('/')[-1]}{'-lora' if args.lora else ''}-"
                                f"{'trainstream' if use_trainstream else 'stock'}")
    cb = live.hf_callback(tokens_per_step=args.batch * args.accum * args.block,
                          optimizer="trainstream" if use_trainstream else "adamw")
    trainer = cls(model=model, args=targs, train_dataset=ds, callbacks=[cb] if cb else None, **kw)
    t0 = time.time()
    try:
        trainer.train()
        status = "ok"
    except torch.OutOfMemoryError as exc:
        status = "OOM: " + str(exc).split(". GPU")[0]
        live.finish("out of memory")
    wall = time.time() - t0
    logs = [h for h in trainer.state.log_history if "loss" in h]
    # TrainStreamTrainer resets the peak counter at every log line (ts_vram_peak_gb).
    peak = max([torch.cuda.max_memory_allocated() / GiB] + [h.get("ts_vram_peak_gb", 0.0) for h in logs])
    print(f"result: {status} · {wall:.1f} s · peak VRAM {peak:.2f} GiB · "
          f"loss {logs[0]['loss']:.3f} -> {logs[-1]['loss']:.3f}" if logs else f"result: {status}")
    if status == "ok":
        print("after: ", repr(sample(model, tok)[:160]))
    del trainer, model
    return status, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--activation-offload", default="auto")
    ap.add_argument("--compare", action="store_true", help="run the stock Trainer first for comparison")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "hf_finetune"))
    _live.add_args(ap)
    args = ap.parse_args()
    if args.activation_offload != "auto":
        args.activation_offload = float(args.activation_offload)
    if args.lora and args.lr == 2e-5:
        args.lr = 2e-4

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = build_dataset(tok, args.block, nbytes=3 << 20)
    print(f"dataset: {len(ds)} blocks of {args.block} tokens from the Python stdlib")
    if args.compare:
        run(args, False, ds, tok)
    run(args, True, ds, tok)
    _live.keep_serving(args)


if __name__ == "__main__":
    main()
