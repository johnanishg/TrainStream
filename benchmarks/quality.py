"""Quality of each frozen-weight format: perplexity and a short LoRA fine-tune.

    python benchmarks/quality.py --model Qwen/Qwen2.5-1.5B
    python benchmarks/quality.py --model Qwen/Qwen2.5-14B --formats none nf4 nvfp4 --windows 16
    python benchmarks/quality.py --formats none nf4 nvfp4-w nvfp4 nvfp4-o   # where NVFP4's loss comes from

For every format the base is loaded with trainstream's load_model (quantized
while read), given zero-initialised LoRA adapters (so the model is exactly the
quantized base) and wrapped. Then:

  1. perplexity of the base on held-out text, and
  2. eval loss after a short LoRA fine-tune (same data, seed and steps for
     every format).

Text: the Python standard library's own source files (offline, identical on
every machine), split into a training half and a held-out half by file.
Writes JSON to --out and prints a table. Each format runs in a fresh process.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import sysconfig
import time

FORMATS = ["none", "int8", "nf4", "fp8", "nvfp4", "nvfp4-w", "nvfp4-o"]
# variants of a stored format: nvfp4-w multiplies expanded weights (no activation
# quantization: what 4-bit weights alone cost), nvfp4-o keeps 1% of the input
# channels in bf16 on the tensor-core path (wrap(outliers=0.01))
VARIANTS = {"nvfp4-w": ("nvfp4", {"weights_only": True}), "nvfp4-o": ("nvfp4", {"outliers": 0.01})}


def corpus_files():
    root = sysconfig.get_paths()["stdlib"]
    files = []
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("test", "tests", "idlelib", "site-packages", "__pycache__"))
        files.extend(os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".py"))
    return files


def token_windows(tok, files, n, seq):
    out, buf = [], []
    for f in files:
        try:
            text = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= seq and len(out) < n:
            out.append(buf[:seq])
            buf = buf[seq:]
        if len(out) >= n:
            break
    return out


def worker(args):
    import peft
    import torch
    from transformers import AutoTokenizer

    import trainstream
    from trainstream.integrations.huggingface import load_model

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    files = corpus_files()
    train_files, eval_files = files[0::2], files[1::2]
    eval_w = torch.tensor(token_windows(tok, eval_files, args.windows, args.seq))
    train_w = torch.tensor(token_windows(tok, train_files, args.train_steps * args.batch, args.seq))

    kind, extra = VARIANTS.get(args.format, (args.format, {}))
    if extra.get("weights_only"):  # expanded before every matmul: only the weights are 4-bit
        from trainstream import lowp
        lowp.native_supported = lambda kind, device: False
    t0 = time.time()
    model = load_model(args.model, quantize=kind)
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=args.lr), device="cuda",
                                  quantize=kind if kind != "none" else "none", outliers=extra.get("outliers", 0.0))

    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(eval_w), 2):
                ids = eval_w[i:i + 2].cuda()
                tot += model(input_ids=ids, labels=ids).loss.float().item() * len(ids)
                n += len(ids)
        model.train()
        return tot / n

    base = eval_loss()
    losses = []
    for s in range(args.train_steps):
        ids = train_w[s * args.batch:(s + 1) * args.batch].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    opt.synchronize()
    tuned = eval_loss()
    rep = trainstream.memory_report(model)
    print(json.dumps(dict(format=args.format, base_loss=base, base_ppl=math.exp(base), tuned_loss=tuned,
                          tuned_ppl=math.exp(tuned), train_losses=losses, seconds=time.time() - t0,
                          events=[e for e in rep["events"] if "quantized" in e or "tensor cores" in e])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS)
    ap.add_argument("--windows", type=int, default=64, help="held-out windows of --seq tokens")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", default=None, help=argparse.SUPPRESS)  # worker mode
    args = ap.parse_args()
    if args.format is not None:
        return worker(args)

    rows = []
    for fmt in args.formats:
        cmd = [sys.executable, __file__, "--model", args.model, "--format", fmt, "--windows", str(args.windows),
               "--seq", str(args.seq), "--train-steps", str(args.train_steps), "--batch", str(args.batch),
               "--lr", str(args.lr)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in res.stdout.splitlines() if l.startswith("{")]
        if res.returncode != 0 or not line:
            rows.append(dict(format=fmt, error=(res.stderr.strip().splitlines() or ["?"])[-1][:300]))
        else:
            rows.append(json.loads(line[-1]))
        print(json.dumps({k: v for k, v in rows[-1].items() if k not in ("train_losses", "events")}), flush=True)

    ok = [r for r in rows if "error" not in r]
    ref = next((r for r in ok if r["format"] == "none"), None)
    print(f"\n{args.model}: perplexity on held-out stdlib source ({args.windows} x {args.seq} tokens)")
    print(f"{'format':8s} {'base ppl':>9s} {'vs bf16':>8s} {'after LoRA':>11s} {'vs bf16':>8s}")
    for r in ok:
        d1 = f"{(r['base_ppl'] / ref['base_ppl'] - 1) * 100:+.2f}%" if ref else ""
        d2 = f"{(r['tuned_ppl'] / ref['tuned_ppl'] - 1) * 100:+.2f}%" if ref else ""
        print(f"{r['format']:8s} {r['base_ppl']:9.3f} {d1:>8s} {r['tuned_ppl']:11.3f} {d2:>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['format']:8s} failed: {r['error']}")
    if args.out:
        json.dump(dict(model=args.model, windows=args.windows, seq=args.seq, train_steps=args.train_steps,
                       rows=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
