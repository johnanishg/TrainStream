"""Head-to-head: TrainStream against stock PyTorch, bitsandbytes QLoRA, Unsloth
and DeepSpeed ZeRO-Offload, on the same GPU, model, data and settings.

    python benchmarks/compare.py --scenario lora14b --engines stock bnb_qlora unsloth trainstream_nf4 trainstream_nvfp4
    python benchmarks/compare.py --scenario full1.5b --engines stock deepspeed trainstream

Every (engine, scenario) runs in a fresh process; a run that does not fit is
recorded as "out of memory", one whose library is not installed as "not
installed". Same data for all: fixed random tokens. Results: JSON lines to
--out and a Markdown table on stdout. The rival libraries are only needed in
the benchmark environment (benchmarks/requirements-bench.txt), never by
TrainStream itself.

Scenarios
  full1.5b      Qwen2.5-1.5B, every parameter trained, batch 4 x 512
  lora14b       Qwen2.5-14B, LoRA r=16 on all projections, batch 1 x 512
  lora14b_2k    the same at batch 1 x 2048
Engines
  stock             plain PyTorch (+ PEFT for LoRA), weights in VRAM, fused AdamW
  bnb_qlora         bitsandbytes NF4 base (BitsAndBytesConfig) + PEFT
  bnb_qlora_ckpt    the same with Hugging Face gradient checkpointing (the usual QLoRA recipe)
  unsloth           Unsloth FastLanguageModel, 4-bit base, its defaults
  deepspeed         DeepSpeed ZeRO-2 with the optimizer offloaded to the CPU
  trainstream       TrainStream defaults (bf16 base)
  trainstream_nf4   TrainStream, NF4 base         (load_model(quantize="nf4"))
  trainstream_nvfp4 TrainStream, NVFP4 base, tensor-core matmuls
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

SCENARIOS = {
    "full1.5b": dict(model="Qwen/Qwen2.5-1.5B", lora=False, batch=4, seq=512),
    "lora14b": dict(model="Qwen/Qwen2.5-14B", lora=True, batch=1, seq=512),
    "lora14b_2k": dict(model="Qwen/Qwen2.5-14B", lora=True, batch=1, seq=2048),
}
ENGINES = ["stock", "bnb_qlora", "bnb_qlora_ckpt", "unsloth", "deepspeed", "trainstream", "trainstream_nf4", "trainstream_nvfp4"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = dict(r=16, alpha=32, dropout=0.0, targets=TARGETS)
ENGINE_LABELS = {
    "stock": "Stock PyTorch (+ PEFT)", "bnb_qlora": "bitsandbytes QLoRA (NF4)",
    "bnb_qlora_ckpt": "bitsandbytes QLoRA + checkpointing", "unsloth": "Unsloth (4-bit)",
    "deepspeed": "DeepSpeed ZeRO-2, CPU offload", "trainstream": "TrainStream (auto)",
    "trainstream_nf4": "TrainStream, NF4 base", "trainstream_nvfp4": "TrainStream, NVFP4 base",
}
LR = dict(lora=2e-4, full=2e-5)  # one learning rate per kind of scenario, for every engine (comparable losses)
# Engines that can't do what a scenario asks: they're skipped there, with the reason.
NOT_FOR_FULL = {
    "bnb_qlora": "trains LoRA adapters on a 4-bit base; this scenario trains every parameter",
    "bnb_qlora_ckpt": "trains LoRA adapters on a 4-bit base; this scenario trains every parameter",
    "unsloth": "set up here for LoRA on a 4-bit base; this scenario trains every parameter",
    "trainstream_nf4": "quantizes frozen base weights; this scenario trains every weight, so it would run exactly "
                       "as TrainStream (auto)",
    "trainstream_nvfp4": "quantizes frozen base weights; this scenario trains every weight, so it would run exactly "
                         "as TrainStream (auto)",
}
WARMUP = 2  # steps left out of the step-time median (allocator warm-up, autotuning, first-use setup)
# An engine that tunes itself while training (TrainStream) gets its tuning steps, plus
# SETTLE more after its last change, added to the warm-up: the median is taken over the
# same number of steps for every engine, all run with the engine's final setup.
SETTLE = 1
MAX_ADAPT = 40  # at most this many extra warm-up steps; past it the run is measured anyway (and says so)
SEED = 0  # the random-token data
SAMPLE_S = 0.5  # --live memory sampling interval


LIVE = False  # --live: also print "LIVE {json}" lines while running (for the test UI)
LIBRARIES = ("torch", "transformers", "peft", "bitsandbytes", "unsloth", "deepspeed", "trainstream")


def proc_mem_gb():
    """(current, peak) resident memory of this process in GB, or (None, None)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.WorkingSetSize / 1e9, c.PeakWorkingSetSize / 1e9
        return None, None
    cur = peak = None
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 1e6
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) / 1e6
    except OSError:
        pass
    return cur, peak


def rss_gb():
    return proc_mem_gb()[1]


def live(kind, **fields):
    if LIVE:
        print("LIVE " + json.dumps(dict(kind=kind, t=time.time(), **fields), default=str), flush=True)


def source_version(name):
    """__version__ as written in the package `import name` would load, without importing it.
    A source tree on the path can sit next to a stale *.egg-info, which metadata.version
    would answer with instead."""
    import importlib.util
    import re
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not (spec.origin or "").endswith("__init__.py"):
        return None
    try:
        with open(spec.origin, encoding="utf-8") as fh:
            m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", fh.read(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def versions():
    from importlib import metadata
    out = {}
    for name in LIBRARIES:
        mod = sys.modules.get(name)
        v = getattr(mod, "__version__", None) if mod is not None else None
        if v is None:
            v = source_version(name)
        if v is None:
            try:
                v = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        out[name] = str(v)
    return out


class Sampler(threading.Thread):
    """Memory every `interval` seconds while the worker loads and trains."""

    def __init__(self, interval=SAMPLE_S):
        super().__init__(daemon=True)
        self.interval = interval
        self.phase = "loading"
        self.done = threading.Event()

    def run(self):
        import torch
        while not self.done.wait(self.interval):
            cur, _ = proc_mem_gb()
            s = dict(phase=self.phase, rss_gb=cur)
            if torch.cuda.is_initialized():  # never initialize CUDA from here
                free, total = torch.cuda.mem_get_info()
                s.update(vram_alloc_gb=torch.cuda.memory_allocated() / 1e9,
                         vram_reserved_gb=torch.cuda.memory_reserved() / 1e9,
                         gpu_used_gb=(total - free) / 1e9, gpu_total_gb=total / 1e9)
            live("sample", **s)


def batches(vocab, batch, seq):
    """The same random tokens for every engine, step after step."""
    import torch
    g = torch.Generator().manual_seed(SEED)
    while True:
        yield torch.randint(0, vocab, (batch, seq), generator=g)


class PeakVRAM:
    """torch.cuda.max_memory_allocated() over the whole training run, even when the
    engine resets the counter itself (TrainStream does while it plans its memory)."""

    def __init__(self):
        import torch
        self.torch, self.real_reset, self.before = torch, torch.cuda.reset_peak_memory_stats, 0
        torch.cuda.reset_peak_memory_stats = torch.cuda.memory.reset_peak_memory_stats = self._reset

    def _reset(self, device=None):
        self.before = max(self.before, self.torch.cuda.max_memory_allocated(device))
        self.real_reset(device)

    def start(self):
        self.before = 0
        self.real_reset()

    def gb(self):
        return max(self.before, self.torch.cuda.max_memory_allocated()) / 1e9


def lora_config():
    import peft
    return peft.LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], task_type="CAUSAL_LM",
                           target_modules=LORA["targets"])


def optimizer_info(opt):
    g = opt.param_groups[0] if getattr(opt, "param_groups", None) else {}
    return dict(optimizer=type(opt).__name__, lr=g.get("lr"), fused=g.get("fused"))


def not_applicable(engine, scenario):
    """Why `engine` doesn't run in `scenario`, or None."""
    return None if SCENARIOS[scenario]["lora"] else NOT_FOR_FULL.get(engine)


def build(engine, sc):
    """Returns (model, step_fn, setup): step_fn(ids) runs forward, backward and the
    update; setup records how this engine was built (base weights, optimizer)."""
    if engine == "unsloth":
        import unsloth  # noqa: F401  before transformers, as Unsloth requires: it patches transformers on import
    import torch
    from transformers import AutoModelForCausalLM
    name = sc["model"]
    bf16 = torch.bfloat16
    lr = LR["lora" if sc["lora"] else "full"]

    def plain_step(model, opt):
        def step(ids):
            with torch.autocast("cuda", dtype=bf16):
                loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss.item()
        return step

    if engine == "stock":
        if sc["lora"]:
            import peft
            model = AutoModelForCausalLM.from_pretrained(name, dtype=bf16).cuda()
            model = peft.get_peft_model(model, lora_config())
        else:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, fused=True)
        return model, plain_step(model, opt), dict(base="bf16 in VRAM" if sc["lora"] else "fp32 in VRAM",
                                                   **optimizer_info(opt))

    if engine in ("bnb_qlora", "bnb_qlora_ckpt"):
        import peft
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bf16)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, dtype=bf16, device_map={"": 0})
        if engine == "bnb_qlora_ckpt":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model = peft.get_peft_model(model, lora_config())
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        return model, plain_step(model, opt), dict(base=f"{q.bnb_4bit_quant_type} (bitsandbytes), compute "
                                                   f"{str(q.bnb_4bit_compute_dtype).replace('torch.', '')}",
                                                   **optimizer_info(opt))

    if engine == "unsloth":
        from unsloth import FastLanguageModel
        # the same checkpoint as every other engine (not Unsloth's pre-quantized copy)
        model, _ = FastLanguageModel.from_pretrained(name, max_seq_length=sc["seq"], load_in_4bit=True, dtype=bf16,
                                                     use_exact_model_name=True)
        model = FastLanguageModel.get_peft_model(model, r=LORA["r"], lora_alpha=LORA["alpha"],
                                                 lora_dropout=LORA["dropout"], target_modules=LORA["targets"])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        return model, plain_step(model, opt), dict(base="4-bit (Unsloth)", **optimizer_info(opt))

    if engine == "deepspeed":
        # one process without DeepSpeed's launcher: give it the rendezvous it would set
        for k, v in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1"),
                     ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29511")):
            os.environ.setdefault(k, v)
        import deepspeed
        model = AutoModelForCausalLM.from_pretrained(name, dtype=bf16)
        if sc["lora"]:
            import peft
            model = peft.get_peft_model(model, lora_config())
        cfg = {"train_micro_batch_size_per_gpu": sc["batch"], "gradient_accumulation_steps": 1,
               "bf16": {"enabled": True},
               "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True}},
               "optimizer": {"type": "AdamW", "params": {"lr": lr}}}
        params = [p for p in model.parameters() if p.requires_grad]
        engine_, _, _, _ = deepspeed.initialize(model=model, model_parameters=params, config=cfg)

        def step(ids):
            loss = engine_(input_ids=ids, labels=ids).loss
            engine_.backward(loss)
            engine_.step()
            return loss.item()
        z = cfg["zero_optimization"]
        return engine_, step, dict(base="bf16", zero_stage=z["stage"], optimizer_on=z["offload_optimizer"]["device"],
                                   optimizer=cfg["optimizer"]["type"], lr=cfg["optimizer"]["params"]["lr"])

    if engine.startswith("trainstream"):
        import trainstream
        from trainstream.integrations.huggingface import load_model
        from trainstream.wrapper import unwrap_session
        kind = {"trainstream": "auto", "trainstream_nf4": "nf4", "trainstream_nvfp4": "nvfp4"}[engine]
        if not sc["lora"]:
            kind = "none"  # every weight is trained, and trained weights are never quantized
        if sc["lora"]:
            import peft
            model = load_model(name, quantize=kind)
            model = peft.get_peft_model(model, lora_config())
        else:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        params = [p for p in model.parameters() if p.requires_grad]
        model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=lr), device="cuda", quantize=kind)
        session = unwrap_session(model)
        used = (session.config.get("quantize") if session is not None else None) or "none"
        step = plain_step(model, opt)
        step.session = session  # the worker reports the decisions TrainStream logs while training
        if session is not None:
            step.adapting = lambda: session.adapting  # still tuning itself: those steps are warm-up
        return model, step, dict(base=f"quantize={kind}" + (f" → {used}" if used != kind else "") + " (TrainStream)",
                                 quantize_used=used, **optimizer_info(opt))

    raise ValueError(engine)


def describe(model, cfg, setup):
    """What was actually built: the engine's own setup plus what the model reports."""
    d = dict(setup)
    try:
        trainable, total = model.get_nb_trainable_parameters()  # PEFT: counts 4-bit weights unpacked
    except Exception:
        ps = list(model.parameters())
        trainable, total = sum(p.numel() for p in ps if p.requires_grad), sum(p.numel() for p in ps)
    d.update(params=total, trainable=trainable, layers=getattr(cfg, "num_hidden_layers", None),
             hidden=getattr(cfg, "hidden_size", None), vocab=getattr(cfg, "vocab_size", None),
             gradient_checkpointing=bool(getattr(model, "is_gradient_checkpointing", False)))
    pc = (getattr(model, "peft_config", None) or {}).get("default")
    if pc is not None:
        d["lora"] = dict(r=pc.r, alpha=pc.lora_alpha, dropout=pc.lora_dropout, targets=sorted(pc.target_modules or []))
    return d


def worker(engine, scenario, steps):
    import torch
    sc = SCENARIOS[scenario]
    out = dict(engine=engine, scenario=scenario, **sc)
    why = not_applicable(engine, scenario)
    if why:
        out.update(status="not applicable", detail=why)
        return out
    peak = PeakVRAM()
    sampler = Sampler()
    if LIVE:
        sampler.start()
        live("phase", phase="loading")
    t0 = time.time()
    try:
        model, step, setup = build(engine, sc)
    except ImportError as exc:
        out.update(status="not installed", detail=str(exc)[:200])
        return out
    except torch.OutOfMemoryError:
        out.update(status="out of memory (loading)")
        return out
    out["load_s"] = time.time() - t0
    model.train()  # from_pretrained returns eval mode; gradient checkpointing only runs in training mode
    inner = getattr(model, "module", model)  # a DeepSpeed engine wraps the model (its own .config is a dict)
    cfg = inner.config
    out["setup"] = setup = describe(inner, cfg, setup)
    # --steps is the run length of an engine that doesn't tune itself: WARMUP steps, then
    # the measured ones. A self-tuning engine's warm-up grows (see SETTLE); the measured count doesn't.
    measured_n = max(steps - WARMUP, 1)
    method = out["method"] = dict(warmup=WARMUP, statistic="median", seed=SEED, steps=WARMUP + measured_n,
                                  measured=measured_n, tuning_steps=0, still_tuning=False,
                                  vram_metric="torch.cuda.max_memory_allocated(), kept across the engine's own resets",
                                  ram_metric="peak working set" if sys.platform == "win32"
                                  else "peak resident set (VmHWM)")
    data = batches(cfg.vocab_size, sc["batch"], sc["seq"])
    times, losses = [], []
    session, events = getattr(step, "session", None), []  # TrainStream: what it decided, after which step
    adapting = getattr(step, "adapting", None)
    out["events"] = events

    def take_events(after_step):
        for text in (session.events if session is not None else [])[len(events):]:
            events.append(dict(step=after_step, text=text))
            live("event", step=after_step, text=text)
    peak.start()
    if LIVE:
        p = torch.cuda.get_device_properties(0)
        live("env", versions=versions(), gpu=p.name, gpu_total_gb=p.total_memory / 1e9, python=sys.version.split()[0],
             platform=sys.platform, load_s=out["load_s"], setup=setup, method=method)
        sampler.phase = "training"
        live("phase", phase="training")
    take_events(0)  # decided while wrapping
    try:
        while len(times) < method["steps"]:
            ids = next(data)
            torch.cuda.synchronize()
            t = time.perf_counter()
            losses.append(step(ids.cuda()))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            i = len(times)
            live("step", i=i, of=method["steps"], step_s=times[-1], loss=losses[-1],
                 tokens_per_s=sc["batch"] * sc["seq"] / times[-1], vram_peak_gb=peak.gb())
            take_events(i)
            if adapting is not None and adapting():  # the next step may run another setup: not steady yet
                if i + 1 + SETTLE <= WARMUP + MAX_ADAPT:
                    method.update(tuning_steps=i + 1, warmup=max(method["warmup"], i + 1 + SETTLE))
                    method["steps"] = method["warmup"] + measured_n
                else:
                    method["still_tuning"] = True
                live("method", **method)
    except torch.OutOfMemoryError:
        out.update(status="out of memory", steps_done=len(times))
        return out
    finally:
        sampler.done.set()
    tail = sorted(times[method["warmup"]:] or times)
    out.update(status="ok", step_s=tail[len(tail) // 2], tokens_per_s=sc["batch"] * sc["seq"] / tail[len(tail) // 2],
               vram_peak_gb=peak.gb(), rss_peak_gb=rss_gb(), losses=losses[:5], step_times=times)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lora14b", choices=SCENARIOS)
    ap.add_argument("--engines", nargs="+", default=ENGINES, choices=ENGINES)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--live", action="store_true", help=argparse.SUPPRESS)  # with --worker: stream LIVE lines
    args = ap.parse_args()
    if args.worker:
        global LIVE
        LIVE = args.live
        print("RESULT " + json.dumps(worker(args.worker[0], args.worker[1], args.steps)), flush=True)
        return
    rows = []
    for engine in args.engines:
        if not_applicable(engine, args.scenario):
            print(f"{engine}: skipped, {not_applicable(engine, args.scenario)}", flush=True)
            continue
        cmd = [sys.executable, __file__, "--worker", engine, args.scenario, "--steps", str(args.steps)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            line = [l for l in res.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                row = json.loads(line[-1][7:])
            else:
                err = (res.stderr.strip().splitlines() or ["?"])[-1]
                row = dict(engine=engine, scenario=args.scenario,
                           status="out of memory" if "OutOfMemory" in res.stderr or "out of memory" in res.stderr
                           else "failed", detail=err[:300])
        except subprocess.TimeoutExpired:
            # e.g. a model larger than VRAM that the driver spills into system RAM
            row = dict(engine=engine, scenario=args.scenario,
                       status=f"did not finish {args.steps} steps in {args.timeout} s")
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "losses"}), flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    print(f"\n| engine | status | step time | tokens/s | peak VRAM | process RAM |\n|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") == "ok":
            ram = f"{r['rss_peak_gb']:.1f} GB" if r.get("rss_peak_gb") else "—"
            print(f"| {r['engine']} | ok | {r['step_s']:.2f} s | {r['tokens_per_s']:.0f} | "
                  f"{r['vram_peak_gb']:.1f} GB | {ram} |")
        else:
            print(f"| {r['engine']} | {r['status']} | — | — | — | — |")


if __name__ == "__main__":
    main()
