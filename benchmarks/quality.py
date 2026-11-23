"""Quality of each frozen-weight format: perplexity and a short LoRA fine-tune.

    python benchmarks/quality.py --model Qwen/Qwen2.5-1.5B
    python benchmarks/quality.py --model Qwen/Qwen2.5-14B --formats none nf4 nvfp4 --windows 16
    python benchmarks/quality.py --formats none nf4 nvfp4-w nvfp4 nvfp4-o   # where NVFP4's loss comes from

For every format the base is loaded with trainstream's load_model (quantized
while read), given zero-initialised LoRA adapters (so the model is exactly the
quantized base; the LoftQ formats start them at the quantization error instead)
and wrapped. Then:

  1. perplexity of the base on held-out text, and
  2. eval loss after a short LoRA fine-tune (same data, seed and steps for
     every format), with the steady training step time (the median of the
     steps after TrainStream has finished tuning itself; synchronized) and
     the GPU's clock meanwhile (NVML or nvidia-smi, as in compare.py).

Text: the Python standard library's own source files (offline, identical on
every machine), split into a training half and a held-out half by file.
Writes JSON to --out and prints a table. Each format runs in a fresh process.

    python benchmarks/quality.py --format nf4 --live    one format, as the test UI runs it

--live (worker only) prints progress as `LIVE {json}` lines (phase, eval batch,
training step) and the result as one `RESULT {json}` line, also on failure.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import sysconfig
import time

FORMATS = ["none", "int8", "nf4", "fp8", "nvfp4", "nvfp4-w", "nvfp4-o", "nf4-k", "nf4-l", "nvfp4-l", "nf4-dq", "nvfp4-b",
           "nf4-p", "nvfp4-p"]
# variants of a stored format: nvfp4-w multiplies expanded weights (no activation
# quantization: what 4-bit weights alone cost), nvfp4-o keeps 1% of the input
# channels in bf16 on the tensor-core path (wrap(outliers=0.01)); nf4-k keeps the
# matrices NF4 hurts most as int8 (keep_precise="auto"); nf4-l / nvfp4-l start the
# LoRA adapters at the weights' quantization error (LoftQ: error_rank=16, the LoRA
# rank); nf4-dq stores NF4's block scales in one byte each (double_quant); nvfp4-b computes
# the input gradients in nvfp4 too (nvfp4_backward); nf4-p / nvfp4-p start the LoRA adapters
# at the weights' principal part and quantize the rest (PiSSA: lora_init="pissa", rank 16)
VARIANTS = {"nvfp4-w": ("nvfp4", {"weights_only": True}), "nvfp4-o": ("nvfp4", {"outliers": 0.01}),
            "nf4-k": ("nf4", {"keep_precise": "auto"}), "nf4-l": ("nf4", {"error_rank": 16}),
            "nvfp4-l": ("nvfp4", {"error_rank": 16}), "nf4-dq": ("nf4", {"double_quant": True}),
            "nvfp4-b": ("nvfp4", {"nvfp4_backward": True}),
            "nf4-p": ("nf4", {"error_rank": 16, "lora_init": "pissa"}),
            "nvfp4-p": ("nvfp4", {"error_rank": 16, "lora_init": "pissa"})}
SETTLE = 1  # steps after TrainStream's last change left out of the step time, as in compare.py


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


def live(args, kind, **fields):
    if args.live:
        print("LIVE " + json.dumps(dict(kind=kind, t=time.time(), **fields)), flush=True)


def worker(args):
    try:
        return _worker(args)
    except BaseException as exc:
        if args.live:  # the reason, before the traceback that follows on stderr
            print("RESULT " + json.dumps(dict(format=args.format, error=f"{type(exc).__name__}: {exc}"[:600])),
                  flush=True)
        raise


def _worker(args):
    live(args, "phase", phase="starting")
    import peft
    import torch
    from transformers import AutoTokenizer

    import trainstream
    from trainstream.integrations.huggingface import load_model
    from trainstream.wrapper import unwrap_session

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
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(), trainstream=trainstream.__version__)
    live(args, "phase", phase="loading", env=env)
    t0 = time.time()
    model = load_model(args.model, quantize=kind, keep_precise=extra.get("keep_precise", 0),
                       error_rank=extra.get("error_rank", 0), double_quant=extra.get("double_quant", False),
                       lora_init=extra.get("lora_init", "loftq"))
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=args.lr), device="cuda",
                                  quantize=kind if kind != "none" else "none", outliers=extra.get("outliers", 0.0),
                                  nvfp4_backward=extra.get("nvfp4_backward", False))
    load_s = time.time() - t0
    session = unwrap_session(model)

    def eval_loss(which):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(eval_w), 2):
                ids = eval_w[i:i + 2].cuda()
                tot += model(input_ids=ids, labels=ids).loss.float().item() * len(ids)
                n += len(ids)
                live(args, "eval", which=which, done=n, of=len(eval_w))
        model.train()
        return tot / n

    live(args, "phase", phase="eval_base", load_s=load_s)
    base = eval_loss("base")
    live(args, "phase", phase="training", base_ppl=math.exp(base))
    from compare import GpuWatch, gpu_target  # GPU samples: how fast the GPU ran meanwhile
    watch = GpuWatch(gpu_target())
    watch.start()
    losses, times, starts, settled = [], [], [], 0
    for s in range(args.train_steps):
        ids = train_w[s * args.batch:(s + 1) * args.batch].cuda()
        torch.cuda.synchronize()
        starts.append(time.time())
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
        if session is not None and session.adapting:  # the next step may still try a setting
            settled = s + 2 + SETTLE
        live(args, "step", i=s + 1, of=args.train_steps, loss=losses[-1], step_s=times[-1])
    opt.synchronize()
    watch.done.set()
    first = settled if len(times) - settled >= 5 else len(times) // 2  # else: the last half
    steady = times[first:]
    step_s = sorted(steady)[len(steady) // 2]
    gpu = watch.summary(starts[first], time.time())
    live(args, "phase", phase="eval_tuned")
    tuned = eval_loss("tuned")
    rep = trainstream.memory_report(model)
    out = dict(format=args.format, base_loss=base, base_ppl=math.exp(base), tuned_loss=tuned,
               tuned_ppl=math.exp(tuned), train_losses=losses, seconds=time.time() - t0, load_s=load_s,
               step_s=step_s, step_times=times, tuning_steps=min(settled, len(times)), gpu=gpu,
               tokens_per_s=args.batch * args.seq / step_s,
               events=[e for e in rep["events"] if "quantized" in e or "tensor cores" in e
                       or "expanded before use" in e], env=env)
    print(("RESULT " if args.live else "") + json.dumps(out), flush=True)


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
    ap.add_argument("--format", default=None, choices=FORMATS, help="run only this format, in this process")
    ap.add_argument("--live", action="store_true", help="with --format: progress as LIVE {json} lines")
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
    print(f"{'format':8s} {'base ppl':>9s} {'vs bf16':>8s} {'after LoRA':>11s} {'vs bf16':>8s} {'step':>8s}")
    for r in ok:
        d1 = f"{(r['base_ppl'] / ref['base_ppl'] - 1) * 100:+.2f}%" if ref else ""
        d2 = f"{(r['tuned_ppl'] / ref['tuned_ppl'] - 1) * 100:+.2f}%" if ref else ""
        step = f"{r['step_s']:.2f} s" if r.get("step_s") else ""
        print(f"{r['format']:8s} {r['base_ppl']:9.3f} {d1:>8s} {r['tuned_ppl']:11.3f} {d2:>8s} {step:>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['format']:8s} failed: {r['error']}")
    if args.out:
        json.dump(dict(model=args.model, windows=args.windows, seq=args.seq, train_steps=args.train_steps,
                       rows=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
