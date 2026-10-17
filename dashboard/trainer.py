"""Real jobs behind the dashboard: training runs, limit searches, profiling, parity checks.

Everything here executes on the actual hardware; the dashboard only displays
what these functions measure. PyTorch runs execute in-process; TensorFlow runs
execute in a separate interpreter (tf_worker.py) and stream JSON events back.
"""

from __future__ import annotations

import collections
import copy
import ctypes
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, Optional

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import trainstream  # noqa: E402
from trainstream import _C  # noqa: E402

from model import GPT, PRESETS, ByteCorpus, GPTConfig, count_params  # noqa: E402

Emit = Callable[[dict], None]
GB = 1 << 30
# Windows keeps a few % of VRAM for itself and pages anything beyond the rest
# into shared system RAM instead of failing, so "full" starts below 100%.
VRAM_FULL = 0.95
# A probe whose work rate (tokens/s x params) falls below this fraction of the
# best passing probe is paging memory, not fitting it.
COLLAPSE = 0.4
TF_WORKER = os.path.join(HERE, "tf_worker.py")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------
def host_memory():
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys), int(st.ullAvailPhys)
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":")
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info["MemFree"])
    except OSError:
        return 0, 0


class _NVML:
    """Device-wide memory from NVML (the library behind nvidia-smi).

    cudaMemGetInfo does not see Direct3D 12 / DirectML allocations (the
    TensorFlow GPU backend on Windows); NVML counts every API and process.
    """

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.ok = False
        try:
            name = "nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1"
            self.lib = ctypes.CDLL(name)
            if self.lib.nvmlInit_v2() != 0:
                return
            self.handle = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                return
            self.ok = True
        except OSError:
            pass

    def used_total(self):
        m = self._Mem()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)) != 0:
            raise OSError("nvmlDeviceGetMemoryInfo failed")
        return int(m.used), int(m.total)


_nvml = _NVML()


def device_vram():
    """(used, total) bytes on GPU 0 across all processes and APIs."""
    if _nvml.ok:
        try:
            return _nvml.used_total()
        except OSError:
            pass
    free, total = torch.cuda.mem_get_info(0)
    return int(total - free), int(total)


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    import platform
    return platform.processor() or "unknown CPU"


def tf_python() -> Optional[str]:
    env = os.environ.get("TRAINSTREAM_TF_PYTHON")
    if env:
        return env
    for rel in (os.path.join(".venv-tf", "Scripts", "python.exe"), os.path.join(".venv-tf", "bin", "python")):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            return p
    return None


def tf_env() -> dict:
    # DirectML enumerates every DX12 adapter (e.g. an iGPU too); keep TF on the
    # first one, which Windows orders as the high-performance GPU.
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1")
    env.setdefault("DML_VISIBLE_DEVICES", "0")
    return env


def tensorflow_info() -> dict:
    py = tf_python()
    if not py:
        return {"available": False, "error": "no TensorFlow interpreter: create .venv-tf or set TRAINSTREAM_TF_PYTHON"}
    try:
        out = subprocess.run([py, TF_WORKER, "--probe"], capture_output=True, text=True, timeout=180,
                             creationflags=NO_WINDOW, env=dict(tf_env(), TF_CPP_MIN_LOG_LEVEL="0"))
        adapters = re.findall(r"creating device on adapter \d+ \((.+?)\)", out.stderr)
        for line in reversed(out.stdout.strip().splitlines()):
            if line.startswith("{"):
                info = dict(json.loads(line), interpreter=py)
                if adapters and info.get("gpus"):
                    info["gpus"] = adapters
                return info
        return {"available": False, "interpreter": py, "error": (out.stderr or out.stdout).strip()[-400:]}
    except Exception as exc:
        return {"available": False, "interpreter": py, "error": f"{type(exc).__name__}: {exc}"}


def system_info():
    props = torch.cuda.get_device_properties(0)
    total, avail = host_memory()
    return {
        "gpu": props.name,
        "vram_total": int(props.total_memory),
        "sm_count": props.multi_processor_count,
        "capability": f"{props.major}.{props.minor}",
        "host_ram_total": total,
        "host_ram_avail": avail,
        "cpu": cpu_name(),
        "cpu_threads": os.cpu_count(),
        "os": f"{sys.platform} {__import__('platform').version()}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "trainstream": trainstream.__version__,
        "build": dict(_C.build_info()),
        "presets": {k: dict(v.to_dict(), params=count_params(v)) for k, v in PRESETS.items()},
        "tensorflow": {"available": False, "checking": True},
    }


def nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, creationflags=NO_WINDOW,
        )
        u, p, t, g, w = [x.strip() for x in out.stdout.strip().split(",")]
        num = lambda s: float(s) if s.replace(".", "", 1).isdigit() else None  # noqa: E731
        return {"util": num(u), "power_w": num(p), "temp_c": num(t), "pcie_gen": num(g), "pcie_width": num(w)}
    except Exception:  # nvidia-smi missing or busy: telemetry degrades gracefully
        return {}


def driver_version():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW)
        return out.stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Telemetry sampler (runs for the life of the server)
# ---------------------------------------------------------------------------
class Telemetry(threading.Thread):
    def __init__(self, emit: Emit, period: float = 0.5):
        super().__init__(daemon=True)
        self.emit = emit
        self.period = period
        self.t0 = time.perf_counter()
        self._smi = {}

    def _poll_smi(self):
        # nvidia-smi takes ~0.5-1 s per call; keep it off the 2 Hz sampler.
        while True:
            self._smi = nvidia_smi()
            time.sleep(1.0)

    def run(self):
        threading.Thread(target=self._poll_smi, daemon=True, name="nvidia-smi").start()
        _C.init(0)
        last = dict(_C.transfer_stats(0))
        last_t = time.perf_counter()
        while True:
            time.sleep(self.period)
            now = time.perf_counter()
            ts = dict(_C.transfer_stats(0))
            dt = max(now - last_t, 1e-6)
            used, total = device_vram()
            ram_total, ram_avail = host_memory()
            self.emit({
                "type": "telemetry",
                "t": round(now - self.t0, 3),
                "d2h_gbps": (ts["bytes_d2h"] - last["bytes_d2h"]) / dt / 1e9,
                "h2d_gbps": (ts["bytes_h2d"] - last["bytes_h2d"]) / dt / 1e9,
                "bytes_d2h": ts["bytes_d2h"],
                "bytes_h2d": ts["bytes_h2d"],
                "vram_allocated": torch.cuda.memory_allocated(0),
                "vram_reserved": torch.cuda.memory_reserved(0),
                "vram_device_used": used,
                "vram_total": total,
                "pinned": dict(_C.pinned_stats()),
                "pinned_by_tag": dict(_C.pinned_tag_bytes()),
                "host_ram_used": ram_total - ram_avail,
                "host_ram_total": ram_total,
                "smi": self._smi,
            })
            last, last_t = ts, now


class PeakSampler(threading.Thread):
    """Samples system RAM and device VRAM at 10 Hz for one run.

    Peaks are reported relative to the moment the run started (so they cover
    every allocation the run made, in any process) and in absolute terms.
    `on_low_ram` fires once when free RAM drops below `ram_floor` bytes.
    """

    def __init__(self, ram_floor: int, on_low_ram=None, on_vram_full=None, period: float = 0.1):
        super().__init__(daemon=True)
        self.period = period
        self.ram_floor = ram_floor
        self.on_low_ram = on_low_ram
        self.on_vram_full = on_vram_full
        self.ram_total, avail = host_memory()
        self.ram_base = self.ram_total - avail
        self.vram_base, self.vram_total = device_vram()
        self.ram_peak = self.ram_base
        self.vram_peak = self.vram_base
        self.ram_now = self.ram_base
        self.vram_now = self.vram_base
        self.min_avail = avail
        self.low_ram = False
        self.vram_full = False
        self._halt = threading.Event()

    def run(self):
        while not self._halt.wait(self.period):
            total, avail = host_memory()
            self.ram_now = total - avail
            self.vram_now, _ = device_vram()
            self.ram_peak = max(self.ram_peak, self.ram_now)
            self.vram_peak = max(self.vram_peak, self.vram_now)
            self.min_avail = min(self.min_avail, avail)
            if not self.low_ram and avail < self.ram_floor:
                self.low_ram = True
                if self.on_low_ram:
                    self.on_low_ram()
            if not self.vram_full and self.vram_now >= VRAM_FULL * self.vram_total:
                self.vram_full = True
                if self.on_vram_full:
                    self.on_vram_full()

    def stop(self):
        self._halt.set()

    def current(self):
        return {"sys_ram_delta": self.ram_now - self.ram_base, "dev_vram_delta": self.vram_now - self.vram_base}

    def summary(self):
        return {
            "peak_sys_ram_delta": self.ram_peak - self.ram_base,
            "peak_sys_ram_used": self.ram_peak,
            "sys_ram_total": self.ram_total,
            "sys_ram_base": self.ram_base,
            "min_sys_ram_avail": self.min_avail,
            "peak_dev_vram_delta": self.vram_peak - self.vram_base,
            "peak_dev_vram_used": self.vram_peak,
            "dev_vram_total": self.vram_total,
            "dev_vram_base": self.vram_base,
            "vram_full": self.vram_full,
            "low_ram": self.low_ram,
        }


# ---------------------------------------------------------------------------
# Model configuration and memory estimates
# ---------------------------------------------------------------------------
def model_config(c: dict) -> GPTConfig:
    return GPTConfig(n_layer=int(c["n_layer"]), n_head=int(c["n_head"]), d_model=int(c["d_model"]), seq_len=int(c["seq_len"]))


def estimate(cfg: GPTConfig, param_dtype: str, engine: str, optimizer_mode: str,
             framework: str = "pytorch", device: str = "gpu"):
    """Static memory (weights + grads + optimizer moments) from exact parameter counts.

    TrainStream cpu mode keeps an fp32 master copy + the gradient DMA staging
    buffer pinned (plus a weight staging buffer for bf16 weights); the two fp32
    moments are pinned too while that fits in 40% of RAM, else ordinary RAM.
    Activations are not included; they depend on batch x seq and are measured.
    """
    n = count_params(cfg)
    pb = 2 if (param_dtype == "bf16" and framework == "pytorch") else 4
    static = n * pb * 4
    if framework == "tensorflow" or engine == "baseline":
        if device == "cpu":
            return {"params": n, "vram_static": 0, "host_total": static, "host_pinned": 0}
        return {"params": n, "vram_static": static, "host_total": 0, "host_pinned": 0}
    if optimizer_mode == "gpu":
        pinned = n * 8 + (n * 4 if pb == 2 else 0)  # mapped moments (+ fp32 master)
        return {"params": n, "vram_static": n * pb * 2, "host_total": pinned, "host_pinned": pinned}
    always = n * (4 + pb + (pb if pb == 2 else 0))
    moments = n * 8
    total_ram, _ = host_memory()
    pinned = always + moments if not total_ram or always + moments <= 0.4 * total_ram else always
    return {"params": n, "vram_static": n * pb, "host_total": always + moments, "host_pinned": pinned}


def run_label(c: dict, run_id) -> str:
    dims = f"{c['n_layer']}L·d{c['d_model']}"
    where = "GPU" if c["device"] == "gpu" else "CPU"
    if c["framework"] == "tensorflow":
        return f"#{run_id} TensorFlow {where} · {dims}"
    if c["engine"] == "baseline":
        return f"#{run_id} PyTorch {where} AdamW · {dims}"
    return f"#{run_id} TrainStream {c['optimizer_mode']} · act {c['activation_offload']} · {dims}"


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return True
    s = str(exc)
    return "out of memory" in s.lower() or "not enough memory" in s.lower() or "DefaultCPUAllocator" in s


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------
class Run:
    """One PyTorch training run (GPU with or without TrainStream, or CPU)."""

    _ids = 0

    def __init__(self, cfg: dict, emit: Emit, corpus_getter, quiet: bool = False, ram_floor: int = 3 * GB,
                 stop_on_vram_full: bool = False):
        Run._ids += 1
        self.id = Run._ids
        self.cfg = cfg
        self.emit_raw = emit
        self.quiet = quiet
        self.corpus_getter = corpus_getter
        self.stop_flag = threading.Event()
        self.ram_floor = ram_floor
        self.stop_on_vram_full = stop_on_vram_full
        self.abort_reason = None
        self.summary = None
        self.step_ms = []
        self.tps = []

    def emit(self, ev):
        if self.quiet and ev.get("type") in ("run_start", "step", "eval", "run_end"):
            return
        if self.quiet and ev.get("type") == "log" and ev.get("level") != "error":
            return
        self.emit_raw(ev)

    def label(self):
        return run_label(self.cfg, self.id)

    def log(self, msg, level="info"):
        self.emit({"type": "log", "run": self.id, "level": level, "msg": msg, "t": time.time()})

    def _abort(self, reason):
        if self.abort_reason is None:
            self.abort_reason = reason
            self.log(f"aborting: {reason}", "error")
        self.stop_flag.set()

    def _on_low_ram(self):
        self._abort(f"system RAM free fell below the {self.ram_floor / GB:.1f} GiB safety floor")

    def _on_vram_full(self):
        pct = f"{VRAM_FULL:.0%}"
        if self.stop_on_vram_full:
            self._abort(f"GPU memory is full (≥{pct} of VRAM in use); beyond this Windows pages it into shared system RAM")
        else:
            self.emit_raw({"type": "log", "run": self.id, "level": "warn", "t": time.time(),
                           "msg": f"GPU memory ≥{pct} full: Windows may start paging it into shared system RAM (large slowdown)"})

    def _start_sampler(self):
        s = PeakSampler(self.ram_floor, self._on_low_ram, self._on_vram_full if self.cfg["device"] == "gpu" else None)
        s.start()
        return s

    def _finish_summary(self, summary, sampler):
        sampler.stop()
        summary.update(sampler.summary())
        if self.abort_reason:
            summary["abort_reason"] = self.abort_reason
            if summary.get("status") in ("stopped", "completed", "running"):
                summary["status"] = "limit"
        tail = self.step_ms[2:] or self.step_ms
        summary["median_step_ms"] = sorted(tail)[len(tail) // 2] if tail else None
        tps = self.tps[2:] or self.tps
        summary["median_tokens_per_s"] = sorted(tps)[len(tps) // 2] if tps else None
        self.summary = summary
        return summary

    def execute(self):
        c = self.cfg
        mcfg = model_config(c)
        on_gpu = c["device"] == "gpu"
        dev = "cuda" if on_gpu else "cpu"
        dtype = torch.bfloat16 if c["param_dtype"] == "bf16" else torch.float32
        est = estimate(mcfg, c["param_dtype"], c["engine"], c["optimizer_mode"], "pytorch", c["device"])
        self.emit({"type": "run_start", "run": self.id, "label": self.label(), "config": c,
                   "model": mcfg.to_dict(), "estimate": est})
        model = opt = None
        summary = {"status": "running", "params": est["params"]}
        step = 0
        gc.collect()
        torch.cuda.empty_cache()
        sampler = self._start_sampler()
        try:
            corpus = self.corpus_getter(dev)
            torch.manual_seed(int(c.get("seed", 0)))
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            self.log(f"building {mcfg.n_layer}L·d{mcfg.d_model}·h{mcfg.n_head} ({est['params'] / 1e6:.1f}M params, "
                     f"{c['param_dtype']}) on {dev}")
            with torch.device(dev):
                model = GPT(mcfg)
            model = model.to(dtype)
            torch_opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), betas=(0.9, 0.95),
                                          weight_decay=0.1, fused=on_gpu and c["engine"] == "baseline")
            clip = float(c["max_grad_norm"]) if c.get("max_grad_norm") else None
            ts_session = None
            if c["engine"] == "baseline":
                opt = torch_opt
            else:
                act = c["activation_offload"]
                act = act if act == "auto" else float(act)
                model, opt = trainstream.wrap(model, torch_opt, optimizer_mode=c["optimizer_mode"],
                                              activation_offload=act, max_grad_norm=clip,
                                              prefetch_depth=int(c.get("prefetch_depth", 1)))
                ts_session = trainstream.wrapper.unwrap_session(model)
                st = opt.stats()
                self.log(f"TrainStream engine ready: {int(st['params'])} tensors, "
                         f"{st['host_state_bytes'] / GB:.2f} GiB pinned host state, {opt.engine.threads} host threads")
            weights_alloc = torch.cuda.memory_allocated() - base_alloc
            if on_gpu:
                self.log(f"weights resident in VRAM: {weights_alloc / GB:.2f} GiB")

            batch, seq = int(c["batch"]), int(c["seq_len"])
            steps = int(c["steps"])
            warmup = min(200, max(20, steps // 5))
            base_lr = float(c["lr"])
            gen = torch.Generator(device=dev).manual_seed(1234)
            last_t = time.perf_counter()
            prev_stats = None
            logged_calib = False
            tokens_seen = 0
            peak_overall = 0
            for step in range(1, steps + 1):
                if self.stop_flag.is_set():
                    summary["status"] = "stopped"
                    step -= 1
                    break
                lr = base_lr * min(1.0, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))))
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.cuda.reset_peak_memory_stats()
                x, y = corpus.batch(batch, seq, generator=gen)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=on_gpu):
                    loss = model(x, y)
                loss.backward()
                if c["engine"] == "baseline" and clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                loss_v = loss.item()  # syncs the compute stream only up to this loss
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                tokens_seen += batch * seq
                cur = sampler.current()
                peak = torch.cuda.max_memory_allocated() - base_alloc if on_gpu else cur["sys_ram_delta"]
                peak_overall = max(peak_overall, peak)
                self.step_ms.append(dt * 1e3)
                self.tps.append(batch * seq / dt)
                m = {"type": "step", "run": self.id, "step": step, "loss": loss_v, "lr": lr,
                     "step_ms": dt * 1e3, "tokens_per_s": batch * seq / dt, "tokens": tokens_seen,
                     "peak_alloc": peak, "alloc": torch.cuda.memory_allocated() - base_alloc, **cur}
                if ts_session is not None:
                    s = opt.stats()
                    if prev_stats is not None:
                        m["ready_stall_ms"] = s["ready_stall_ms_total"] - prev_stats["ready_stall_ms_total"]
                        m["host_update_ms"] = s["update_ms_total"] - prev_stats["update_ms_total"]
                        m["grad_stall_ms"] = s["grad_stall_ms_total"] - prev_stats["grad_stall_ms_total"]
                    prev_stats = s
                    a = ts_session.activations
                    if a is not None:
                        m["act"] = {"offloaded_blocks": len(a.offload), "total_blocks": a.n,
                                    "calibrating": a.calibrating, "packed_bytes": a.stats["packed_bytes"],
                                    "hits": a.stats["prefetch_hits"], "misses": a.stats["prefetch_misses"],
                                    "dedup": a.stats["dedup_hits"]}
                        if a.decision and not logged_calib:
                            logged_calib = True
                            d = a.decision
                            self.log(f"activation calibration: block fwd {d['block_fwd_ms']:.2f} ms, "
                                     f"{d['block_bytes'] / 2**20:.1f} MiB/block, D2H {d['d2h_ms']:.2f} ms → "
                                     f"paging {int(d['offloaded_blocks'])}/{int(d['total_blocks'])} blocks")
                    if opt.max_grad_norm:
                        n_ = opt.last_grad_norm
                        if n_ is not None:
                            m["grad_norm"] = n_
                self.emit(m)
                summary["final_loss"] = loss_v
                if not c.get("probe") and (step % int(c.get("eval_every", 50)) == 0 or step == steps):
                    summary["val_loss"] = self.evaluate(model, corpus, batch, seq, step, dev)
            if summary["status"] == "running":
                summary["status"] = "completed"
            summary.update(steps=step, peak_alloc=peak_overall, weights_alloc=weights_alloc)
            if ts_session is not None:
                summary["host_pinned"] = int(opt.stats()["host_state_bytes"])
        except Exception as exc:  # surface any failure in the UI
            if _is_oom(exc):
                summary.update(status="oom", steps=step, error=str(exc).split("\n")[0][:300])
                self.log(f"out of memory at step {step}: {summary['error']}", "error")
            else:
                summary.update(status="error", steps=step, error=f"{type(exc).__name__}: {exc}"[:500])
                self.log(traceback.format_exc(), "error")
        finally:
            try:
                if model is not None and c["engine"] != "baseline":
                    if opt is not None and hasattr(opt, "synchronize"):
                        opt.synchronize()
                    trainstream.unwrap(model)
            except Exception:
                pass
            model = opt = torch_opt = None  # noqa: F841
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _C.pinned_empty_cache()
            self._finish_summary(summary, sampler)
            self.emit({"type": "run_end", "run": self.id, "summary": summary})
        return summary

    def evaluate(self, model, corpus, batch, seq, step, dev):
        model.eval()
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            g = torch.Generator(device=dev).manual_seed(99)
            losses = [model(*corpus.batch(batch, seq, "val", generator=g)).item() for _ in range(4)]
            prompt = torch.tensor([list(b"def ")], device=dev)
            sample = ByteCorpus.decode(model.generate(prompt, 160 if dev == "cuda" else 60)[0])
        model.train()
        val = sum(losses) / len(losses)
        self.emit({"type": "eval", "run": self.id, "step": step, "val_loss": val, "sample": sample})
        return val


class TFRun(Run):
    """One TensorFlow training run in a separate interpreter (see tf_worker.py)."""

    def execute(self):
        c = self.cfg
        mcfg = model_config(c)
        est = estimate(mcfg, "fp32", "baseline", "cpu", "tensorflow", c["device"])
        self.emit({"type": "run_start", "run": self.id, "label": self.label(), "config": c,
                   "model": mcfg.to_dict(), "estimate": est})
        summary = {"status": "running", "params": est["params"]}
        py = tf_python()
        if not py:
            summary.update(status="error", error="TensorFlow interpreter not found")
            self.emit({"type": "run_end", "run": self.id, "summary": summary})
            return summary
        gc.collect()
        torch.cuda.empty_cache()
        _C.pinned_empty_cache()
        sampler = self._start_sampler()
        stderr_tail = collections.deque(maxlen=60)
        killed = threading.Event()
        stop_file = os.path.join(tempfile.gettempdir(), f"trainstream-tf-stop-{os.getpid()}-{self.id}")
        if os.path.exists(stop_file):
            os.remove(stop_file)
        proc = subprocess.Popen([py, "-u", TF_WORKER, "--run", json.dumps(dict(c, stop_file=stop_file))], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                                env=tf_env())

        def drain_stderr():
            for line in proc.stderr:
                if line.strip():
                    stderr_tail.append(line.rstrip())

        def watch_stop():
            self.stop_flag.wait()
            if proc.poll() is not None:
                return
            if self.abort_reason:
                killed.set()
                proc.kill()  # memory limit: do not wait for the step to finish
                return
            with open(stop_file, "w"):
                pass

        threading.Thread(target=drain_stderr, daemon=True).start()
        threading.Thread(target=watch_stop, daemon=True).start()
        self.log(f"launched TensorFlow worker ({os.path.basename(os.path.dirname(os.path.dirname(py)))})")
        got_end = False
        peak = 0
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    if line:
                        stderr_tail.append(line)
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    stderr_tail.append(line)
                    continue
                t = ev.get("type")
                if t == "log":
                    self.emit({"type": "log", "run": self.id, "level": ev.get("level", "info"), "msg": ev["msg"], "t": time.time()})
                elif t == "model":
                    summary["params"] = ev["params"]
                elif t == "step":
                    cur = sampler.current()
                    ev.update(cur)
                    p = cur["dev_vram_delta"] if c["device"] == "gpu" else cur["sys_ram_delta"]
                    if c["device"] == "gpu" and ev.get("tf_gpu_peak"):
                        summary["tf_gpu_peak"] = max(summary.get("tf_gpu_peak", 0), ev["tf_gpu_peak"])
                        if ev["tf_gpu_peak"] > sampler.vram_total and not self.abort_reason:
                            msg = (f"TensorFlow's allocator holds {ev['tf_gpu_peak'] / GB:.1f} GiB, more than the "
                                   f"{sampler.vram_total / GB:.1f} GiB of VRAM: DirectML is paging into shared system RAM")
                            if self.stop_on_vram_full:
                                self._abort(msg)
                            elif not summary.get("spill_logged"):
                                summary["spill_logged"] = True
                                self.emit_raw({"type": "log", "run": self.id, "level": "warn", "t": time.time(), "msg": msg})
                    peak = max(peak, p)
                    ev.update(run=self.id, peak_alloc=p)
                    self.step_ms.append(ev["step_ms"])
                    self.tps.append(ev["tokens_per_s"])
                    summary["final_loss"] = ev["loss"]
                    summary["steps"] = ev["step"]
                    self.emit(ev)
                elif t == "eval":
                    summary["val_loss"] = ev["val_loss"]
                    self.emit(dict(ev, run=self.id))
                elif t == "run_end":
                    got_end = True
                    worker = ev["summary"]
                    summary.update({k: v for k, v in worker.items() if k != "steps" or v})
            proc.wait(timeout=30)
        except Exception as exc:
            summary.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if proc.poll() is None:
                proc.kill()
            self.stop_flag.set()  # release watch_stop
            if os.path.exists(stop_file):
                os.remove(stop_file)
        if not got_end:
            if killed.is_set():
                summary["status"] = "limit"
            elif summary.get("status") == "running":
                tail = "\n".join(stderr_tail)
                oom = any(s in tail for s in ("ResourceExhausted", "OOM", "out of memory", "MemoryError", "E_OUTOFMEMORY"))
                summary.update(status="oom" if oom else "error",
                               error=f"TensorFlow worker exited with code {proc.returncode}: {stderr_tail[-1] if stderr_tail else ''}"[:400])
        if summary.get("status") in ("error", "oom") and stderr_tail:
            self.emit_raw({"type": "log", "run": self.id, "level": "error", "t": time.time(),
                           "msg": "TensorFlow stderr (last lines):\n" + "\n".join(list(stderr_tail)[-15:])})
        summary["peak_alloc"] = peak
        self._finish_summary(summary, sampler)
        self.emit({"type": "run_end", "run": self.id, "summary": summary})
        return summary


def make_run(cfg, emit, corpus_getter, **kw):
    cls = TFRun if cfg["framework"] == "tensorflow" else Run
    return cls(cfg, emit, corpus_getter, **kw)


# ---------------------------------------------------------------------------
# Limit search: largest model / batch that still trains
# ---------------------------------------------------------------------------
AXES = {
    "n_layer": {"label": "layers", "min": 1, "max": 512},
    "d_model": {"label": "width (d_model)", "min": 64, "max": 32768},
    "batch": {"label": "batch size", "min": 1, "max": 4096},
}


def _axis_step(cfg, axis):
    if axis == "d_model":
        h = int(cfg["n_head"])
        return h * 64 // math.gcd(h, 64)  # multiple of 64 and of n_head
    return 1


def _snap(v, step, lo):
    return max(lo, (int(v) // step) * step)


def limit_search(base: dict, axis: str, emit: Emit, corpus_getter, stop: threading.Event,
                 probe_steps: int = 6, ram_floor: int = 4 * GB, max_probes: int = 14, on_run=None):
    """Grow `axis` (doubling, then bisection) until training no longer fits.

    A probe passes when it trains `probe_steps` steps without an out-of-memory
    error, without free system RAM falling below `ram_floor`, and (on GPU)
    without filling VRAM — past that point Windows silently pages GPU memory
    into shared system RAM, which "works" but is not a real fit.
    """
    spec = AXES[axis]
    step_sz = _axis_step(base, axis)
    lo_bound = max(spec["min"], step_sz)
    start = _snap(base[axis], step_sz, lo_bound)
    vram_total = device_vram()[1]
    emit({"type": "limit_start", "axis": axis, "axis_label": spec["label"], "config": base, "probe_steps": probe_steps,
          "ram_floor": ram_floor, "start": start})
    rows = []
    ok_v, fail_v = None, None
    v = start
    probes = 0
    while probes < max_probes and not stop.is_set():
        cfg = copy.deepcopy(base)
        cfg[axis] = int(v)
        cfg.update(steps=probe_steps, probe=True)
        mcfg = model_config(cfg)
        est = estimate(mcfg, cfg["param_dtype"], cfg["engine"], cfg["optimizer_mode"], cfg["framework"], cfg["device"])
        _, avail = host_memory()
        row = {"value": int(v), "params": est["params"], "est": est}
        reason = None
        if est["host_total"] > avail - ram_floor:
            reason = f"predicted: needs {est['host_total'] / GB:.1f} GiB host RAM, {max(avail - ram_floor, 0) / GB:.1f} GiB usable"
        elif cfg["device"] == "gpu" and est["vram_static"] > 0.97 * vram_total:
            reason = f"predicted: needs {est['vram_static'] / GB:.1f} GiB VRAM before activations, GPU has {vram_total / GB:.1f} GiB"
        if reason:
            row.update(status="skipped", ok=False, reason=reason)
        else:
            probes += 1
            emit({"type": "log", "level": "info", "t": time.time(),
                  "msg": f"limit probe {probes}: {spec['label']} = {v} ({est['params'] / 1e6:.0f}M params)"})
            run = make_run(cfg, emit, corpus_getter, quiet=True, ram_floor=ram_floor, stop_on_vram_full=cfg["device"] == "gpu")
            if on_run:
                on_run(run)
            s = run.execute()
            ok = s.get("status") == "completed" and not s.get("low_ram") and not (cfg["device"] == "gpu" and s.get("vram_full"))
            why = s.get("abort_reason") or s.get("error") or ("" if ok else s.get("status"))
            work = (s.get("median_tokens_per_s") or 0) * est["params"]
            best_work = max((r["work"] for r in rows if r.get("ok") and r.get("work")), default=0)
            if ok and best_work and work < COLLAPSE * best_work:
                ok = False
                why = (f"throughput collapsed to {work / best_work:.0%} of the best probe: memory is being paged "
                       f"({'GPU memory into shared system RAM' if cfg['device'] == 'gpu' else 'system RAM to disk'})")
                s["status"] = "paging"
            row["work"] = work
            if stop.is_set() and not ok and not s.get("abort_reason"):
                break
            row.update(status=s.get("status"), ok=ok, reason=why, steps=s.get("steps"),
                       median_step_ms=s.get("median_step_ms"), median_tokens_per_s=s.get("median_tokens_per_s"),
                       peak_alloc=s.get("peak_alloc"), peak_dev_vram_used=s.get("peak_dev_vram_used"),
                       peak_dev_vram_delta=s.get("peak_dev_vram_delta"), dev_vram_total=s.get("dev_vram_total"),
                       peak_sys_ram_used=s.get("peak_sys_ram_used"), peak_sys_ram_delta=s.get("peak_sys_ram_delta"),
                       sys_ram_total=s.get("sys_ram_total"), min_sys_ram_avail=s.get("min_sys_ram_avail"),
                       host_pinned=s.get("host_pinned"), tf_gpu_peak=s.get("tf_gpu_peak"))
        rows.append(row)
        emit({"type": "limit_probe", "row": row})
        if row["ok"]:
            ok_v = v
        else:
            fail_v = v
        if fail_v is None:
            nxt = min(v * 2, spec["max"])
            if nxt == v:
                break
        else:
            lo = ok_v if ok_v is not None else lo_bound - step_sz
            if fail_v - lo <= step_sz:
                break
            nxt = _snap(lo + (fail_v - lo) // 2, step_sz, lo_bound)
            if nxt <= lo:
                nxt = lo + step_sz
            if nxt >= fail_v:
                break
        v = nxt
    best = max((r for r in rows if r["ok"]), key=lambda r: r["value"], default=None)
    first_fail = min((r for r in rows if not r["ok"] and (best is None or r["value"] > best["value"])),
                     key=lambda r: r["value"], default=None)
    result = {"axis": axis, "axis_label": spec["label"], "config": base, "best": best, "first_fail": first_fail,
              "rows": rows, "probes_run": probes, "stopped": stop.is_set(),
              "resolution": step_sz, "converged": bool(best and first_fail and first_fail["value"] - best["value"] <= step_sz)}
    emit({"type": "limit_done", "result": result})
    return result


# ---------------------------------------------------------------------------
# Correctness: OffloadAdamW vs torch.optim.AdamW on identical data
# ---------------------------------------------------------------------------
def parity_check(emit: Emit):
    results = []
    hp = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    def mk(seed=0):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                             nn.Linear(1024, 64)).cuda()

    g = torch.Generator().manual_seed(1)
    data = [(torch.randn(64, 256, generator=g).cuda(), torch.randint(0, 64, (64,), generator=g).cuda())
            for _ in range(20)]
    for mode in ("cpu", "gpu"):
        for clip in (None, 1.0):
            ref, mine = mk(), mk()
            o_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **hp)
            o = trainstream.OffloadAdamW(mine.parameters(), mode=mode, max_grad_norm=clip, **hp).attach(mine)
            t_ref = t_mine = 0.0
            for x, y in data:
                for model, opt, is_ref in ((ref, o_ref, True), (mine, o, False)):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    nn.functional.cross_entropy(model(x), y).backward()
                    if is_ref and clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
                    opt.zero_grad()
                    if not is_ref:
                        opt.synchronize()
                    torch.cuda.synchronize()
                    if is_ref:
                        t_ref += time.perf_counter() - t
                    else:
                        t_mine += time.perf_counter() - t
            diff = max((a - b).abs().max().item() for a, b in zip(ref.parameters(), mine.parameters()))
            scale = max(a.abs().max().item() for a in ref.parameters())
            r = {"mode": mode, "clip": clip, "steps": len(data), "max_abs_diff": diff, "max_param": scale,
                 "pass": diff < 5e-5, "ms_ref": t_ref * 1e3 / len(data), "ms_trainstream": t_mine * 1e3 / len(data)}
            results.append(r)
            emit({"type": "parity_row", "row": r})
            o.detach()
    return results


def optimizer_bench(emit: Emit, numel: int = 100_000_000):
    """Optimizer step on `numel` parameters: VRAM state cost and step latency."""
    rows = []
    shapes = [(numel // 4 // 4096, 4096)] * 4
    for name in ("torch AdamW (fused)", "OffloadAdamW cpu", "OffloadAdamW gpu"):
        gc.collect()
        torch.cuda.empty_cache()
        params = [torch.nn.Parameter(torch.randn(s, device="cuda") * 0.02) for s in shapes]
        base = torch.cuda.memory_allocated()
        if name.startswith("torch"):
            opt = torch.optim.AdamW(params, lr=1e-4, fused=True)
        else:
            opt = trainstream.OffloadAdamW(params, lr=1e-4, mode=name.split()[-1])
        times = []
        for i in range(6):
            for p in params:
                p.grad = torch.randn_like(p) * 1e-3
            torch.cuda.synchronize()
            t = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
        state_vram = torch.cuda.memory_allocated() - base - sum(p.numel() * 4 for p in params if p.grad is not None)
        row = {"name": name, "numel": sum(p.numel() for p in params), "step_ms": sorted(times[1:])[len(times[1:]) // 2],
               "vram_state_bytes": max(state_vram, 0),
               "host_state_bytes": opt.stats()["host_state_bytes"] if hasattr(opt, "engine") else 0}
        rows.append(row)
        emit({"type": "bench_row", "row": row})
        del opt, params
    gc.collect()
    torch.cuda.empty_cache()
    _C.pinned_empty_cache()
    return rows
