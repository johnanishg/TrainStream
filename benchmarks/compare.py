"""Head-to-head: TrainStream against stock PyTorch, bitsandbytes QLoRA, Unsloth
and DeepSpeed ZeRO-Offload, on the same GPU, model, data and settings.

    python benchmarks/compare.py --scenario lora14b --engines stock bnb_qlora unsloth trainstream_nf4 trainstream_nvfp4
    python benchmarks/compare.py --scenario full1.5b --engines stock deepspeed trainstream
    python benchmarks/compare.py --scenario lora14b_2k --engines bnb_qlora trainstream_nf4 --profile 2

Every (engine, scenario) runs in a fresh process; a run that does not fit is
recorded as "out of memory", one whose library is not installed as "not
installed". Same data for all: fixed random tokens. Results: JSON lines to
--out and a Markdown table on stdout. The rival libraries are only needed in
the benchmark environment (benchmarks/requirements-bench.txt), never by
TrainStream itself.

While an engine trains, the GPU's SM clock, temperature, power draw and clock
limits are sampled four times a second through NVML (the driver's library;
nvidia-smi every 2 s where it can't be loaded); the result carries them for
the measured steps ("gpu"), so a report can tell a slower GPU from a slower
engine (a laptop GPU's clocks follow its temperature and power limits, and
its power limit moves when the CPU draws more).

--profile N runs N more steps after the measured ones under torch.profiler
(GPU activity only) and reports where a step's time goes: GPU time per kernel
category (GEMM, attention, quantize / dequantize, elementwise & norm, loss,
optimizer, other), host <-> GPU copies, and how much of the step the GPU spends
computing, only copying (transfers not hidden behind compute) or idle (waiting
on the host). Tracing slows a step a little, so the profiled step time is a bit
above the measured one; the measured numbers never include profiled steps.
Engines slower than PROFILE_MAX_S per step are not profiled. --profile-ops adds
another N steps traced on the CPU side too, with input shapes, and lists the
operations whose kernels took the most GPU time (e.g. aten::mm [[512, 13824],
[13824, 5120]]): slower to trace and heavier on host RAM, so it is opt-in.

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
import shutil
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
PROFILE = 0  # --profile N: steps traced after the measured ones (0: off)
PROFILE_OPS = False  # --profile-ops: N more steps traced with the operations (and shapes) that launch the kernels
OPS_TOP = 25
PROFILE_MAX_S = 60.0  # engines slower than this per step are not profiled (a long trace for little news)
# GPU kernel categories, matched against the kernel's full (demangled) name; the first match wins
KERNEL_CATEGORIES = [
    ("attention", r"flash|fmha|attention|sdpa|attn"),
    ("GEMM", r"gemm|xmma|nvjet|cutlass|cublas|wgmma|tensorop|matmul|splitkreduce"),
    ("quantize / dequantize", r"quant|absmax|nf4|fp4|e2m1|blockwise"),
    ("loss", r"cross_?entropy|nll_?loss|log_?soft_?max"),
    ("optimizer", r"adam|multi_tensor_apply|foreach|optim|sq_norm|clip"),
    ("elementwise & norm", r"elementwise|reduce|norm|moments|silu|gelu|swiglu|fg_kernel|rotary|rope|embed|cat_|"
                           r"catarray|copy|index|gather|scatter|fill|where|softmax|triton|unrolled|vectorized"),
]


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


GPU_SAMPLE_S = 0.25  # NVML samples of the GPU's clock while an engine trains
SMI_SAMPLE_S = 2.0  # nvidia-smi samples (a process each) where NVML can't be loaded
# why the driver held the SM clock down (NVML's clocks event reasons, nvidia-smi's clocks_event_reasons)
CLOCK_LIMITS = {0x4: "power cap", 0x8: "hardware slowdown", 0x20: "thermal limit", 0x40: "hardware thermal limit",
                0x80: "power brake"}


def gpu_target():
    """CUDA device 0 for NVML / nvidia-smi -i: its PCI bus id (they number GPUs differently)."""
    import torch
    p = torch.cuda.get_device_properties(0)
    if getattr(p, "pci_bus_id", None) is None:
        return "0"
    return f"{getattr(p, 'pci_domain_id', 0):08X}:{p.pci_bus_id:02X}:{getattr(p, 'pci_device_id', 0):02X}.0"


def nvidia_smi():
    """nvidia-smi's path: on PATH, else where WSL keeps it (a benchmark env's PATH may leave it out)."""
    return shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi")
                                          else None)


class Nvml:
    """The few NVML calls GpuWatch needs, through ctypes: the library comes with the driver,
    and a call costs microseconds where nvidia-smi starts a process. OSError without it."""

    LIBS = (("nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll") if os.name == "nt"
            else ("libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"))

    def __init__(self, target="0"):
        import ctypes
        self.c, lib = ctypes, None
        for name in self.LIBS:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None or lib.nvmlInit_v2() != 0:
            raise OSError("NVML can't be loaded")
        self.lib, self.h = lib, ctypes.c_void_p()
        found = target != "0" and lib.nvmlDeviceGetHandleByPciBusId_v2(target.encode(), ctypes.byref(self.h)) == 0
        if not found and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) != 0:
            lib.nvmlShutdown()
            raise OSError("NVML sees no GPU")
        self.reasons = (getattr(lib, "nvmlDeviceGetCurrentClocksEventReasons", None)
                        or getattr(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", None))  # older drivers

    def sample(self):
        """(SM MHz, °C, W, limits bitmask), None for what the driver doesn't report; None without a clock."""
        c, lib, h = self.c, self.lib, self.h
        sm, temp, mw, why = c.c_uint(), c.c_uint(), c.c_uint(), c.c_ulonglong()
        if lib.nvmlDeviceGetClockInfo(h, 1, c.byref(sm)) != 0:  # NVML_CLOCK_SM
            return None
        return (float(sm.value),
                float(temp.value) if lib.nvmlDeviceGetTemperature(h, 0, c.byref(temp)) == 0 else None,
                mw.value / 1000.0 if lib.nvmlDeviceGetPowerUsage(h, c.byref(mw)) == 0 else None,
                why.value if self.reasons is not None and self.reasons(h, c.byref(why)) == 0 else None)

    def close(self):
        self.lib.nvmlShutdown()


class GpuWatch(threading.Thread):
    """The GPU's SM clock, temperature, power draw and clock limits every GPU_SAMPLE_S seconds
    through NVML, else every SMI_SAMPLE_S from nvidia-smi (no Python dependency either way): a
    laptop GPU's clocks follow its temperature and power, so a report can tell a slower GPU from
    a slower engine. Silent without either."""

    FIELDS = ("clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active",
              "clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",  # older drivers
              "clocks.sm,temperature.gpu,power.draw")

    def __init__(self, target="0"):
        super().__init__(daemon=True)
        self.target = target
        self.targets = [target] if target == "0" else [target, "0"]
        self.fields = None
        self.samples = []  # (time, SM MHz, °C, W or None, limits bitmask or None)
        self.done = threading.Event()

    def _smi(self, exe):
        """One nvidia-smi sample as Nvml.sample() gives it, or None when it can't answer."""
        def num(x):
            try:
                return float(x)
            except ValueError:
                return None  # "[N/A]"
        for target in self.targets:
            for fields in [self.fields] if self.fields else self.FIELDS:
                try:
                    res = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", target],
                                         capture_output=True, text=True, timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    return None
                if res.returncode == 0 and res.stdout.strip():
                    self.targets, self.fields = [target], fields
                    v = [x.strip() for x in res.stdout.strip().splitlines()[0].split(",")]
                    return (num(v[0]), num(v[1]), num(v[2]),
                            int(v[3], 16) if len(v) > 3 and v[3].startswith("0x") else None)
        return None

    def run(self):
        try:
            nvml = Nvml(self.target)
        except OSError:
            nvml = None
        exe = None if nvml else nvidia_smi()
        if nvml is None and exe is None:
            return
        try:
            while not self.done.is_set():
                t = time.time()
                vals = nvml.sample() if nvml else self._smi(exe)
                if vals is None and nvml is None:
                    return
                if vals is not None and vals[0] is not None:
                    self.samples.append((t,) + tuple(vals))
                self.done.wait(GPU_SAMPLE_S if nvml else SMI_SAMPLE_S)
        finally:
            if nvml is not None:
                nvml.close()

    def summary(self, t0, t1):
        """The samples taken between t0 and t1 (else every sample): median SM clock, its lowest,
        the hottest temperature, median power and the clock limits seen."""
        xs = [s for s in self.samples if t0 <= s[0] <= t1] or list(self.samples)
        if not xs:
            return None

        def median(v):
            v = sorted(x for x in v if x is not None)
            return v[len(v) // 2] if v else None
        seen = 0
        for s in xs:
            seen |= s[4] or 0
        return dict(samples=len(xs), sm_mhz=median(s[1] for s in xs), sm_mhz_min=min(s[1] for s in xs),
                    temp_c_max=max((s[2] for s in xs if s[2] is not None), default=None),
                    power_w=median(s[3] for s in xs), limits=[n for bit, n in CLOCK_LIMITS.items() if seen & bit])


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
        step.drain = getattr(opt, "synchronize", None)  # its host optimizer runs behind the step: wait for it
        if session is not None:
            step.adapting = lambda: session.adapting  # still tuning itself: those steps are warm-up
        return model, step, dict(base=f"quantize={kind}" + (f" → {used}" if used != kind else "") + " (TrainStream)",
                                 quantize_used=used, **optimizer_info(opt))

    raise ValueError(engine)


def kernel_category(name):
    import re
    for cat, pattern in KERNEL_CATEGORIES:
        if re.search(pattern, name, re.I):
            return cat
    return "other"


def mangled_identifiers(name):
    """The <length><identifier> runs of an Itanium-mangled name, in order (a scan,
    not a demangler: enough to name a kernel)."""
    out, i, n, after_token = [], 0, len(name), False
    while i < n:
        # a length starts right after a token (which may end in a digit) or after a non-digit
        if name[i].isdigit() and (after_token or not name[i - 1].isdigit()):
            j = i
            while j < n and name[j].isdigit():
                j += 1
            tok = name[j:j + int(name[i:j])]
            if len(tok) >= 3 and len(tok) == int(name[i:j]) and (tok[0].isalpha() or tok[0] == "_") \
                    and all(c.isalnum() or c == "_" for c in tok):
                out.append(tok)
                i, after_token = j + len(tok), True
                continue
            i = j
        else:
            i += 1
        after_token = False
    return out


def short_kernel(name):
    """A kernel's name without return type, namespaces, template and call arguments.
    Traces on Windows carry mangled names: the first identifier that is not a
    namespace or a generic wrapper (cutlass's Kernel2: then its template argument)."""
    import re
    if name.startswith("_Z"):
        for tok in mangled_identifiers(name):
            if not (tok.startswith("_GLOBAL__N") or (tok.isalpha() and tok.islower())
                    or re.fullmatch(r"(device_)?kernel\d*", tok, re.I)):
                return tok[:90]
        return name[:90]
    s = name[5:] if name.startswith("void ") else name
    s = s.replace("(anonymous namespace)::", "")
    base = s
    for stop in ("<", "("):
        base = base.split(stop, 1)[0]
    base = base.rsplit("::", 1)[-1] or name
    if re.fullmatch(r"(device_)?kernel\d*", base, re.I) and "<" in s:  # cutlass::Kernel2<the GEMM's own name>
        arg = s.split("<", 1)[1]
        for stop in ("<", "(", ",", ">"):
            arg = arg.split(stop, 1)[0]
        base = arg.strip().rsplit("::", 1)[-1] or base
    return base[:90]


def busy_us(spans):
    """Length of the union of (start, end) intervals (overlap across streams counted once)."""
    total, cur = 0.0, None
    for a, b in sorted(spans):
        if cur is None or a > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return total + (cur[1] - cur[0] if cur is not None else 0.0)


def summarize_trace(events, wall_s, n):
    """Per step, from a chrome trace's GPU events: see the module docstring (--profile)."""
    kernels, k_spans, all_spans, named = {}, [], [], {}
    copies = dict(h2d_ms=0.0, d2h_ms=0.0, d2d_ms=0.0, h2d_gb=0.0, d2h_gb=0.0, d2d_gb=0.0)
    count = 0
    for e in events:
        cat = e.get("cat")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset") or "dur" not in e or "ts" not in e:
            continue
        a = float(e["ts"])
        b = a + float(e["dur"])
        all_spans.append((a, b))
        name = e.get("name") or "?"
        if cat == "kernel":
            count += 1
            k_spans.append((a, b))
            if name not in named:  # (short name, category), per full name: instances of one template differ
                named[name] = (short_kernel(name), kernel_category(name))
            k = kernels.setdefault(named[name], dict(us=0.0, calls=0))
            k["us"] += b - a
            k["calls"] += 1
        elif cat == "gpu_memcpy":
            way = "h2d" if "HtoD" in name else "d2h" if "DtoH" in name else "d2d"
            copies[f"{way}_ms"] += (b - a) / 1000.0
            copies[f"{way}_gb"] += float((e.get("args") or {}).get("bytes") or 0) / 1e9
    if not count:
        return dict(error="the profiler saw no GPU kernels here (GPU tracing through CUPTI unavailable?)")

    def per(us):  # microseconds in total -> ms per step
        return round(us / 1000.0 / n, 2)
    compute, anything = busy_us(k_spans), busy_us(all_spans)
    categories = {}
    for (_, c), k in kernels.items():
        categories[c] = categories.get(c, 0.0) + k["us"]
    top = sorted(kernels.items(), key=lambda kv: -kv[1]["us"])[:10]
    step_ms = wall_s * 1000.0 / n
    return dict(steps=n, step_ms=round(step_ms, 1), compute_ms=per(compute),
                copy_only_ms=per(anything - compute), idle_ms=round(max(step_ms - anything / 1000.0 / n, 0.0), 2),
                kernel_ms=per(sum(k["us"] for k in kernels.values())), kernels_per_step=round(count / n),
                categories={c: per(us) for c, us in sorted(categories.items(), key=lambda kv: -kv[1])},
                copies={k: round(v / n, 3) for k, v in copies.items()},
                top=[dict(name=name, category=c, ms=per(k["us"]), calls=round(k["calls"] / n, 1))
                     for (name, c), k in top])


def profile_steps(step, data, n):
    """Run n more steps under torch.profiler (GPU activity) and summarize the trace."""
    import tempfile
    import torch
    from torch.profiler import ProfilerActivity, profile
    fd, path = tempfile.mkstemp(prefix="ts-profile-", suffix=".json")
    os.close(fd)
    # An engine whose optimizer runs on host threads behind the step (TrainStream) keeps issuing
    # copies after step() returns; the tracer starts and stops only once that work has landed
    # (stopping it under them is the likely cause of a crash seen once on Windows).
    drain = getattr(step, "drain", None) or (lambda: None)
    try:
        batches_ = [next(data) for _ in range(n)]
        drain()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t = time.perf_counter()
            for ids in batches_:
                step(ids.cuda())
            drain()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8") as fh:
            events = json.load(fh).get("traceEvents") or []
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return summarize_trace(events, wall, n)


def profile_ops(step, data, n, top=OPS_TOP):
    """Run n more steps traced with the CPU-side operations and their input shapes; the top
    operations by the GPU time of the kernels they launched themselves, per step."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    drain = getattr(step, "drain", None) or (lambda: None)
    batches_ = [next(data) for _ in range(n)]
    drain()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for ids in batches_:
            step(ids.cuda())
        drain()
        torch.cuda.synchronize()
    rows = []
    cpu = torch.autograd.DeviceType.CPU
    for e in prof.key_averages(group_by_input_shape=True):
        if getattr(e, "device_type", cpu) != cpu:
            continue  # the kernels themselves: counted under the operation that launched them
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0)
        if us > 0:
            shapes = [list(x) for x in (e.input_shapes or []) if x]
            rows.append(dict(op=e.key, shapes=str(shapes)[:200], calls=round(e.count / n, 1), ms=round(us / 1000.0 / n, 2)))
    rows.sort(key=lambda r: -r["ms"])
    return rows[:top]


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
    watch = GpuWatch(gpu_target())
    watch.start()
    starts = []  # wall time each step began: the GPU samples of the measured steps
    try:
        while len(times) < method["steps"]:
            ids = next(data)
            torch.cuda.synchronize()
            starts.append(time.time())
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
        watch.done.set()
    tail = sorted(times[method["warmup"]:] or times)
    out.update(status="ok", step_s=tail[len(tail) // 2], tokens_per_s=sc["batch"] * sc["seq"] / tail[len(tail) // 2],
               vram_peak_gb=peak.gb(), rss_peak_gb=rss_gb(), losses=losses[:5], step_times=times,
               gpu=watch.summary(starts[min(method["warmup"], len(starts) - 1)], time.time()))
    if PROFILE:  # after the measurements: the peaks and the step time above never include traced steps
        if out["step_s"] > PROFILE_MAX_S:
            out["profile"] = dict(skipped=f"{out['step_s']:.0f} s per step, over the {PROFILE_MAX_S:.0f} s "
                                          f"profiling limit")
        else:
            # the measured result first: readers keep the last RESULT line, so it stands even if
            # tracing takes the process down (GPU tracing support varies, e.g. under WSL2)
            print("RESULT " + json.dumps(out), flush=True)
            live("phase", phase="profiling", steps=PROFILE)
            try:
                out["profile"] = profile_steps(step, data, PROFILE)
            except torch.OutOfMemoryError:
                out["profile"] = dict(error="out of memory while profiling (the measured results stand)")
            except Exception as exc:  # noqa: BLE001 - a profile is extra; the run itself succeeded
                out["profile"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])
            if PROFILE_OPS and out["profile"].get("categories"):
                print("RESULT " + json.dumps(out), flush=True)  # the kernel profile stands whatever happens next
                live("phase", phase="profiling", steps=PROFILE, what="operations")
                try:
                    out["profile"]["ops"] = profile_ops(step, data, PROFILE)
                except Exception as exc:  # noqa: BLE001 - extra again
                    out["profile"]["ops_error"] = f"{type(exc).__name__}: {exc}"[:300]
            live("profile", **out["profile"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lora14b", choices=SCENARIOS)
    ap.add_argument("--engines", nargs="+", default=ENGINES, choices=ENGINES)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--profile-ops", action="store_true",
                    help="with --profile: N more steps traced with the operations and shapes behind the kernels")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="trace N more steps after the measured ones: where the time goes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--live", action="store_true", help=argparse.SUPPRESS)  # with --worker: stream LIVE lines
    args = ap.parse_args()
    if args.worker:
        global LIVE, PROFILE, PROFILE_OPS
        LIVE, PROFILE, PROFILE_OPS = args.live, max(args.profile, 0), args.profile_ops
        print("RESULT " + json.dumps(worker(args.worker[0], args.worker[1], args.steps)), flush=True)
        return
    rows = []
    for engine in args.engines:
        if not_applicable(engine, args.scenario):
            print(f"{engine}: skipped, {not_applicable(engine, args.scenario)}", flush=True)
            continue
        cmd = [sys.executable, __file__, "--worker", engine, args.scenario, "--steps", str(args.steps),
               "--profile", str(args.profile)] + (["--profile-ops"] if args.profile_ops else [])
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
    profiled = [r for r in rows if (r.get("profile") or {}).get("categories")]
    if profiled:
        cats = []
        for r in profiled:
            cats += [c for c in r["profile"]["categories"] if c not in cats]
        print("\nWhere a step's time goes (ms per step, over the profiled steps)\n")
        print("| engine | step | " + " | ".join(cats) + " | copies only | GPU idle |")
        print("|---|---|" + "---|" * (len(cats) + 2))
        for r in profiled:
            p = r["profile"]
            print(f"| {r['engine']} | {p['step_ms']:.0f} | "
                  + " | ".join(f"{p['categories'].get(c, 0):.0f}" for c in cats)
                  + f" | {p['copy_only_ms']:.0f} | {p['idle_ms']:.0f} |")


if __name__ == "__main__":
    main()
