"""Prove where TrainStream keeps training memory: VRAM vs system RAM.

    python examples/verify_ram_offload.py                         # GPT 1.2B, VRAM capped at 16 GB
    python examples/verify_ram_offload.py --gpt xxl-1.8B --vram-cap-gb 22
    python examples/verify_ram_offload.py --hf Qwen/Qwen2.5-1.5B --vram-cap-gb 16
    python examples/verify_ram_offload.py --vram-cap-gb 0          # no cap: use the whole GPU

Each engine (stock torch AdamW, then TrainStream) runs in its own fresh process
so its memory can be measured from a clean baseline. The parent samples
Windows' per-process GPU counters ("Dedicated Usage" = VRAM, "Shared Usage" =
system RAM the *driver* maps into the GPU when VRAM overflows), and the child
reports PyTorch's VRAM accounting, its own committed RAM, TrainStream's pinned
allocations and the bytes it pushed over PCIe. A PASS/FAIL verdict follows.

--vram-cap-gb uses torch.cuda.set_per_process_memory_fraction: PyTorch then
raises a real out-of-memory error at the cap instead of letting the Windows
driver silently spill into shared memory, i.e. it simulates a smaller GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GiB = float(1 << 30)


# ---------------------------------------------------------------------------
# OS memory probes (standard library only)
# ---------------------------------------------------------------------------
def system_ram():
    """(total, available) physical RAM in bytes."""
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
        st = MS()
        st.len = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.total), int(st.avail)
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) * 1024
    return info["MemTotal"], info.get("MemAvailable", info["MemFree"])


def process_ram():
    """(committed/private bytes, resident working set) of this process."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # pseudo-handle -1 must stay 64-bit
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(pmc.PrivateUsage), int(pmc.WorkingSetSize)
    vals = {}
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(("VmRSS", "VmData")):
                k, v = line.split(":")
                vals[k] = int(v.split()[0]) * 1024
    return vals.get("VmData", 0), vals.get("VmRSS", 0)


class GpuCounterSampler(threading.Thread):
    """Samples Windows' per-process GPU memory counters for one PID.

    'Dedicated Usage' is VRAM. 'Shared Usage' is system RAM that the WDDM
    driver maps into the GPU's address space; it grows when a process
    overflows VRAM (driver "sysmem fallback"), which is slow and is *not*
    how TrainStream uses RAM. CUDA pinned host buffers can be counted here
    too (they are GPU-mapped system pages), so the report sets them aside.
    """

    def __init__(self, pid: int = 0, period: float = 1.0):
        super().__init__(daemon=True)
        self.pid, self.period = pid, period  # set once the worker reports its real PID
        self.samples = []
        self.stop = threading.Event()
        self.available = sys.platform == "win32"

    def query(self):
        cmd = ("$s=(Get-Counter -ErrorAction SilentlyContinue -Counter "
               f"'\\GPU Process Memory(pid_{self.pid}_*)\\Dedicated Usage',"
               f"'\\GPU Process Memory(pid_{self.pid}_*)\\Shared Usage').CounterSamples;"
               "foreach($x in $s){ $x.Path + '|' + $x.CookedValue }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        ded = shared = 0.0
        found = False
        for line in out.splitlines():
            if "|" not in line:
                continue
            path, val = line.rsplit("|", 1)
            try:
                v = float(val)
            except ValueError:
                continue
            found = True
            if path.lower().endswith("dedicated usage"):
                ded += v
            elif path.lower().endswith("shared usage"):
                shared += v
        return (ded, shared) if found else None

    def run(self):
        if not self.available:
            return
        while not self.stop.is_set():
            if self.pid:
                try:
                    got = self.query()
                    if got is not None:
                        self.samples.append((time.time(), got[0], got[1]))
                except Exception:
                    pass
            self.stop.wait(self.period)

    def peak(self):
        if not self.samples:
            return None, None
        return max(s[1] for s in self.samples), max(s[2] for s in self.samples)


# ---------------------------------------------------------------------------
# Child: one engine, one fresh CUDA context
# ---------------------------------------------------------------------------
def emit(obj):
    sys.stdout.write("@@" + json.dumps(obj) + "\n")
    sys.stdout.flush()


def build_model_and_data(args, torch):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.hf:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf)
        model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=dtype).cuda()
        text = load_corpus_text(2 << 20)
        ids = torch.tensor(tok(text, return_tensors=None)["input_ids"], dtype=torch.long, device="cuda")

        def batch(g):
            ix = torch.randint(0, len(ids) - args.seq - 1, (args.batch,), device="cuda", generator=g)
            win = ids[ix[:, None] + torch.arange(args.seq + 1, device="cuda")]
            return dict(input_ids=win[:, :-1], labels=win[:, :-1])

        def loss_fn(b):
            return model(**b).loss
        blocks = None
        return model, batch, loss_fn, blocks
    sys.path.insert(0, os.path.join(ROOT, "dashboard"))
    from model import GPT, PRESETS, ByteCorpus, GPTConfig  # noqa: E402
    cfg = GPTConfig(**{**PRESETS[args.gpt].to_dict(), "seq_len": args.seq})
    model = GPT(cfg).to("cuda", dtype)
    corpus = ByteCorpus("cuda")

    def batch(g):
        x, y = corpus.batch(args.batch, args.seq, generator=g)
        return (x, y)

    def loss_fn(b):
        return model(*b)
    return model, batch, loss_fn, list(model.blocks)


def load_corpus_text(nbytes):
    root = os.path.join(sys.base_prefix, "Lib")
    chunks, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("site-packages", "test", "tests", "__pycache__"))
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
                    chunks.append(fh.read())
                total += len(chunks[-1])
                if total > nbytes:
                    return "\n".join(chunks)
    return "\n".join(chunks)


def child(args):
    import torch
    sys.path.insert(0, ROOT)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    cap = args.vram_cap_gb * GiB if args.vram_cap_gb > 0 else 0
    if cap:
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total_vram), 0)
    torch.cuda.init()
    ram0_priv, _ = process_ram()
    out = {"engine": args.engine, "pid": os.getpid(), "vram_total": total_vram, "vram_cap": cap,
           "ram_private_start": ram0_priv, "status": "running"}
    emit({"phase": "start", "pid": os.getpid()})
    try:
        torch.manual_seed(0)
        model, batch, loss_fn, blocks = build_model_and_data(args, torch)
        torch.cuda.synchronize()
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pbytes = next(model.parameters()).element_size()
        out.update(params=n, param_bytes=pbytes, vram_after_model=torch.cuda.memory_allocated())
        out["ram_private_after_model"] = process_ram()[0]
        emit({"phase": "model", **out})

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                betas=(0.9, 0.95), weight_decay=0.1, fused=args.engine == "baseline")
        ts = None
        if args.engine == "trainstream":
            import trainstream
            act = args.activation_offload
            act = act if act == "auto" else float(act)
            model, opt = trainstream.wrap(model, opt, activation_offload=act, blocks=blocks)
            ts = trainstream
            out["ram_private_after_wrap"] = process_ram()[0]
            st = opt.stats()
            out.update(host_state_bytes=st["host_state_bytes"], pinned_state_bytes=st["pinned_state_bytes"],
                       pageable_state_bytes=st["pageable_state_bytes"])
        g = torch.Generator(device="cuda").manual_seed(1234)
        steps = []
        from trainstream import _C as tsC  # transfer counters (zero for the baseline)
        last_t = dict(tsC.transfer_stats(0))
        for step in range(1, args.steps + 1):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            b = batch(g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(b)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_v = loss.item()
            if ts is not None:
                opt.synchronize()  # measure a fully landed step
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tr = dict(tsC.transfer_stats(0))
            free, total = torch.cuda.mem_get_info()
            priv, ws = process_ram()
            rec = {"step": step, "loss": loss_v, "ms": dt * 1e3,
                   "alloc": torch.cuda.memory_allocated(), "peak": torch.cuda.max_memory_allocated(),
                   "reserved": torch.cuda.memory_reserved(), "device_used": total - free,
                   "ram_private": priv, "ram_ws": ws, "sys_ram_avail": system_ram()[1],
                   "d2h": tr["bytes_d2h"] - last_t["bytes_d2h"], "h2d": tr["bytes_h2d"] - last_t["bytes_h2d"]}
            last_t = tr
            if ts is not None:
                pst = tsC.pinned_stats()
                rec["pinned_live"] = pst["live_bytes"]
                rec["pinned_reserved"] = pst["reserved_bytes"]  # live + cached (still page-locked)
                rec["pinned_by_tag"] = dict(tsC.pinned_tag_bytes())
            steps.append(rec)
            emit({"phase": "step", **rec})
        out["steps"] = steps
        # Persistent VRAM: what stays allocated between steps beyond the weights.
        out["vram_persistent_extra"] = torch.cuda.memory_allocated() - out["vram_after_model"]
        if ts is not None:
            # Direct evidence: inspect the largest parameter's Adam moments.
            slot = max(range(len(opt._slot_params)), key=lambda i: opt._slot_params[i].numel())
            master, m, v = opt.engine.state(slot)
            out["moment_probe"] = {
                "param_shape": list(opt._slot_params[slot].shape),
                "exp_avg_device": str(m.device), "exp_avg_is_pinned": bool(m.is_pinned()),
                "exp_avg_bytes": m.numel() * m.element_size(),
                "exp_avg_abs_mean": float(m.abs().mean()), "exp_avg_sq_mean": float(v.mean()),
                "master_device": str(master.device) if master is not None else None,
                "gpu_param_vs_master_max_diff": float((opt._slot_params[slot].detach().float().cpu().view(-1)
                                                       - master.view(-1)).abs().max()) if master is not None else None,
            }
            out["engine_stats"] = opt.stats()
        out["status"] = "ok"
    except torch.OutOfMemoryError as exc:
        out["status"] = "oom"
        msg = str(exc).split("\n")[0]
        out["error"] = msg.split(". GPU")[0][:200]  # 'CUDA out of memory. Tried to allocate …'
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"[:600]
    out["ram_private_end"] = process_ram()[0]
    emit({"phase": "done", **out})


# ---------------------------------------------------------------------------
# Parent: orchestrate, sample counters, judge
# ---------------------------------------------------------------------------
def run_engine(args, engine):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--engine", engine]
    for k in ("gpt", "hf", "vram_cap_gb", "steps", "batch", "seq", "dtype", "lr", "activation_offload"):
        v = getattr(args, k)
        if v is not None:
            cmd += ["--" + k.replace("_", "-"), str(v)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            encoding="utf-8", errors="replace")
    # The venv's python.exe is a launcher that spawns the real interpreter, so
    # the GPU counters are sampled for the PID the worker reports itself.
    sampler = GpuCounterSampler()
    sampler.start()
    result = {"engine": engine, "status": "crashed"}
    tail = []
    for line in proc.stdout:
        if line.startswith("@@"):
            ev = json.loads(line[2:])
            if ev.get("pid") and not sampler.pid:
                sampler.pid = ev["pid"]
            if ev["phase"] == "model":
                print(f"  [{engine}] model: {ev['params'] / 1e9:.2f}B params, weights {ev['vram_after_model'] / GiB:.2f} GiB in VRAM")
            elif ev["phase"] == "step":
                pinned = (f" pinned {ev['pinned_live'] / GiB:5.2f} (+{(ev['pinned_reserved'] - ev['pinned_live']) / GiB:4.2f} cached) GiB"
                          if "pinned_live" in ev else "")
                print(f"  [{engine}] step {ev['step']:3d} loss {ev['loss']:.3f} {ev['ms']:8.0f} ms | VRAM peak {ev['peak'] / GiB:5.2f}"
                      f" GiB | proc RAM {ev['ram_private'] / GiB:5.2f} GiB{pinned} | PCIe D2H {ev['d2h'] / GiB:5.2f} H2D {ev['h2d'] / GiB:5.2f} GiB",
                      flush=True)
            elif ev["phase"] == "done":
                result = ev
        else:
            tail.append(line.rstrip())
            tail = tail[-15:]
    proc.wait()
    sampler.stop.set()
    sampler.join(timeout=20)
    result["gpu_counter_peak_dedicated"], result["gpu_counter_peak_shared"] = sampler.peak()
    result["gpu_counters_available"] = sampler.available and bool(sampler.samples)
    if result.get("status") in ("crashed", "error"):
        result["log_tail"] = tail
    status = result.get("status")
    print(f"  [{engine}] finished: {status}" + (f" — {result.get('error')}" if result.get("error") else ""))
    return result


def verdict(args, results):
    checks = []

    def check(name, ok, detail):
        checks.append((name, ok, detail))

    base = results.get("baseline")
    ts = results.get("trainstream")
    cap = (ts or base or {}).get("vram_cap") or (ts or base or {}).get("vram_total", 0)
    if ts and ts.get("params"):
        n, pb = ts["params"], ts["param_bytes"]
        static_stock = n * pb * 4
        check("Stock AdamW's static footprint exceeds the VRAM budget",
              static_stock > cap,
              f"weights+grads+2 moments = {static_stock / GiB:.1f} GiB vs budget {cap / GiB:.1f} GiB")
    if base:
        b_ok = base["status"] == "oom" or (base.get("gpu_counter_peak_shared") or 0) > 2 * GiB
        how = ("PyTorch raised CUDA out-of-memory" if base["status"] == "oom" else
               f"driver spilled {((base.get('gpu_counter_peak_shared') or 0) / GiB):.1f} GiB into shared memory"
               if (base.get("gpu_counter_peak_shared") or 0) > 2 * GiB else f"status {base['status']}")
        check("Stock AdamW cannot train in that budget", b_ok, how)
    if ts:
        steps = ts.get("steps") or []
        check("TrainStream trains inside the same budget", ts["status"] == "ok" and len(steps) == args.steps,
              f"{len(steps)}/{args.steps} steps, status {ts['status']}" + (f" ({ts.get('error')})" if ts.get("error") else ""))
        if ts["status"] == "ok" and steps:
            n, pb = ts["params"], ts["param_bytes"]
            moments = n * 8
            extra = ts["vram_persistent_extra"]
            check("Optimizer moments are NOT in VRAM",
                  extra < 0.05 * moments + 256 * (1 << 20),
                  f"persistent VRAM beyond weights after training: {extra / GiB:.2f} GiB "
                  f"(stock AdamW would hold {moments / GiB:.2f} GiB of moments)")
            if base and base["status"] == "ok":
                check("…while stock AdamW keeps them there", base["vram_persistent_extra"] > 0.8 * moments,
                      f"stock persistent extra {base['vram_persistent_extra'] / GiB:.2f} GiB")
            host = ts.get("host_state_bytes", 0)
            grew = ts["ram_private_after_wrap"] - ts["ram_private_after_model"]
            check("Optimizer state IS in system RAM",
                  host >= 0.95 * (moments + n * 4) and grew >= 0.9 * host,
                  f"engine reports {host / GiB:.2f} GiB host state "
                  f"({ts['pinned_state_bytes'] / GiB:.2f} pinned + {ts['pageable_state_bytes'] / GiB:.2f} pageable); "
                  f"process committed RAM grew {grew / GiB:.2f} GiB when the optimizer was created")
            mp = ts.get("moment_probe") or {}
            check("A live Adam moment tensor sits in host memory and is being updated",
                  mp.get("exp_avg_device") == "cpu" and mp.get("exp_avg_sq_mean", 0) > 0,
                  f"largest param {mp.get('param_shape')}: exp_avg on {mp.get('exp_avg_device')} "
                  f"(pinned={mp.get('exp_avg_is_pinned')}), mean|m|={mp.get('exp_avg_abs_mean', 0):.3e}, "
                  f"mean v={mp.get('exp_avg_sq_mean', 0):.3e}")
            if mp.get("gpu_param_vs_master_max_diff") is not None:
                check("GPU weights equal the host fp32 master after the last update",
                      mp["gpu_param_vs_master_max_diff"] <= (0.0 if pb == 4 else 1e-2),
                      f"max |GPU − host master| = {mp['gpu_param_vs_master_max_diff']:.2e}")
            d2h = [s["d2h"] for s in steps[1:]] or [steps[0]["d2h"]]
            h2d = [s["h2d"] for s in steps[1:]] or [steps[0]["h2d"]]
            grad_bytes = n * pb
            check("Gradients cross PCIe to RAM every step",
                  min(d2h) >= 0.95 * grad_bytes,
                  f"D2H per step {min(d2h) / GiB:.2f}–{max(d2h) / GiB:.2f} GiB (gradients alone: {grad_bytes / GiB:.2f} GiB)")
            check("Updated weights cross PCIe back every step",
                  min(h2d) >= 0.95 * grad_bytes,
                  f"H2D per step {min(h2d) / GiB:.2f}–{max(h2d) / GiB:.2f} GiB (weights: {grad_bytes / GiB:.2f} GiB)")
            peak = max(s["peak"] for s in steps)
            check("Peak VRAM stayed within the budget", peak <= cap * 1.001,
                  f"peak {peak / GiB:.2f} GiB of {cap / GiB:.2f} GiB")
            if ts.get("gpu_counters_available"):
                # Windows reports CUDA pinned host memory as "Shared GPU memory",
                # so TrainStream's own page-locked buffers (optimizer state +
                # cached activation stashes) are subtracted before calling it a spill.
                shared = ts.get("gpu_counter_peak_shared") or 0
                pinned = max(s.get("pinned_reserved", s.get("pinned_live", 0)) for s in steps)
                spill = max(0.0, shared - pinned)
                check("No driver spill of VRAM into shared memory",
                      spill <= 1 * GiB,
                      f"Windows 'Shared GPU memory' peak {shared / GiB:.2f} GiB − TrainStream pinned buffers "
                      f"{pinned / GiB:.2f} GiB = {spill / GiB:.2f} GiB unexplained (driver spill would show here)")
            losses = [s["loss"] for s in steps]
            if base and base["status"] == "ok" and base.get("steps"):
                bl = [s["loss"] for s in base["steps"]]
                k = min(len(bl), len(losses))
                diff = max(abs(a - b) for a, b in zip(bl[:k], losses[:k]))
                check("Training math matches stock AdamW", diff < 0.05, f"max |Δloss| over {k} steps = {diff:.4f}")
            else:
                check("Loss is finite and moving", all(math.isfinite(x) for x in losses) and abs(losses[-1] - losses[0]) > 1e-3,
                      f"loss {losses[0]:.3f} → {losses[-1]:.3f}")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpt", default=None, help="dashboard GPT preset (default xl-1.2B when --hf is not given)")
    ap.add_argument("--hf", default=None, help="Hugging Face model id, e.g. Qwen/Qwen2.5-1.5B")
    ap.add_argument("--vram-cap-gb", type=float, default=16.0, help="simulate a GPU with this much VRAM (0 = no cap)")
    ap.add_argument("--engine", default="both", choices=["both", "baseline", "trainstream"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--activation-offload", default="auto")
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.hf and not args.gpt:
        args.gpt = "xl-1.2B"
    if args.worker:
        return child(args)

    total, avail = system_ram()
    print(f"System RAM: {avail / GiB:.1f} GiB free of {total / GiB:.1f} GiB · VRAM budget: "
          f"{'whole GPU' if not args.vram_cap_gb else f'{args.vram_cap_gb:.1f} GiB (simulated)'} · model: {args.hf or args.gpt}")
    results = {}
    for engine in (["baseline", "trainstream"] if args.engine == "both" else [args.engine]):
        print(f"\n=== {engine} ===")
        results[engine] = run_engine(args, engine)

    print("\n=== Verdict ===")
    checks = verdict(args, results)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    passed = sum(ok for _, ok, _ in checks)
    print(f"\n{passed}/{len(checks)} checks passed.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "results": results,
                       "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]}, fh, indent=2)
        print(f"Report written to {args.json}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
