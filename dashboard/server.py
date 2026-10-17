"""TrainStream live dashboard — standard-library HTTP server + Server-Sent Events.

    python dashboard/server.py [--port 8765] [--host 127.0.0.1]

Then open http://127.0.0.1:8765 . Every number on the page comes from real
GPU work started from the page (training runs, profiler, parity checks).
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import uuid
import os
import queue
import re
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import report  # noqa: E402
import trainer  # noqa: E402
from model import PRESETS, ByteCorpus  # noqa: E402

HISTORY_LIMIT = 20000


class Hub:
    """Fan-out of JSON events to every connected SSE client, with replay."""

    def __init__(self):
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.history: list[str] = []  # run/step/eval/log events for late joiners

    def publish(self, event: dict) -> None:
        data = json.dumps(event, default=float, allow_nan=False) if _finite(event) else json.dumps(_sanitize(event))
        with self._lock:
            if event.get("type") != "telemetry":
                self.history.append(data)
                if len(self.history) > HISTORY_LIMIT:
                    del self.history[: len(self.history) - HISTORY_LIMIT]
            for q in self._clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass  # slow client: drop rather than stall training

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=5000)
        with self._lock:
            for item in self.history:
                q.put_nowait(item)
            self._clients.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)


def _finite(obj) -> bool:
    if isinstance(obj, float):
        return obj == obj and obj not in (float("inf"), float("-inf"))
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    return True


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if _finite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


BOOT_ID = uuid.uuid4().hex
BOOT_TIME = time.time()
RESTART_LOG = os.path.join(__import__('tempfile').gettempdir(), "trainstream-dashboard.log")


def child_pids(pid: int) -> list:
    """Direct child processes of `pid` (TensorFlow workers, nvidia-smi, ...)."""
    if sys.platform == "win32":
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}').ProcessId"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    kids = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as fh:
                    if int(fh.read().rsplit(")", 1)[1].split()[1]) == pid:
                        kids.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return kids


def launch_outside_job(cmd: list, cwd: str, log_path: str, env: dict) -> bool:
    """Windows: start `cmd` through WMI (Win32_Process.Create). The process is
    created by the WMI service, so it is not part of any job object the caller
    is in. Output is appended to `log_path`."""
    sets = "".join(f"set {k}={v}&& " for k, v in env.items())
    line = sets + subprocess.list2cmdline(cmd) + f' >> "{log_path}" 2>&1'
    cmdline = f'cmd.exe /s /c "{line}"'
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0};"
          "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          "@{CommandLine=$env:TS_CMDLINE; CurrentDirectory=$env:TS_CWD; ProcessStartupInformation=$si};"
          "exit [int]$r.ReturnValue")
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=30,
                             env=dict(os.environ, TS_CMDLINE=cmdline, TS_CWD=cwd),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return res.returncode == 0
    except Exception:
        return False


def kill_tree(pid: int) -> None:
    for kid in child_pids(pid):
        kill_tree(kid)
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(kid)], capture_output=True, timeout=20,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                os.kill(kid, 9)
        except Exception:
            pass


class App:
    def __init__(self):
        self.server = None
        self.argv = list(sys.argv)
        self.hub = Hub()
        self.gpu_lock = threading.Lock()  # one GPU job at a time
        self.current_run: trainer.Run | None = None
        self.queue_stop = threading.Event()
        self.job = None
        self._corpus = {}
        self._corpus_lock = threading.Lock()
        self.info = trainer.system_info()
        self.info["driver"] = trainer.driver_version()
        self.telemetry = trainer.Telemetry(self.hub.publish)
        self.telemetry.start()
        threading.Thread(target=self._check_tensorflow, daemon=True, name="tf-probe").start()

    def _check_tensorflow(self):
        info = trainer.tensorflow_info()
        self.info["tensorflow"] = info
        if info.get("available"):
            msg = f"TensorFlow {info['version']} ready (Python {info['python']}, GPU backend: {info.get('gpu_backend')}, devices: {info.get('gpus') or 'none'})"
        else:
            msg = f"TensorFlow unavailable: {info.get('error')}"
        self.hub.publish({"type": "log", "level": "info" if info.get("available") else "warn", "t": time.time(), "msg": msg})
        self.hub.publish({"type": "tf_info", "info": info})

    def corpus(self, device="cuda"):
        with self._corpus_lock:
            if device not in self._corpus:
                c = self._corpus[device] = ByteCorpus(device)
                self.hub.publish({"type": "log", "level": "info", "t": time.time(),
                                  "msg": f"corpus on {device}: {c.files} stdlib files, {c.nbytes / 2**20:.1f} MiB of real source"})
            return self._corpus[device]

    def busy(self):
        return self.gpu_lock.locked()

    def start_job(self, name, fn):
        if not self.gpu_lock.acquire(blocking=False):
            return False
        self.job = name

        def runner():
            self.hub.publish({"type": "job", "job": name, "state": "start"})
            try:
                fn()
            except Exception as exc:
                self.hub.publish({"type": "log", "level": "error", "t": time.time(),
                                  "msg": f"{name} failed: {exc}\n{traceback.format_exc()}"})
            finally:
                self.job = None
                self.current_run = None
                self.gpu_lock.release()
                self.hub.publish({"type": "job", "job": name, "state": "end"})

        threading.Thread(target=runner, daemon=True, name=f"job-{name}").start()
        return True

    # -- job bodies --------------------------------------------------------
    def run_training(self, configs):
        self.queue_stop.clear()
        for cfg in configs:
            if self.queue_stop.is_set():
                break
            run = trainer.make_run(cfg, self.hub.publish, self.corpus)
            self.current_run = run
            run.execute()

    def run_limit(self, configs, axis, probe_steps, ram_floor_gb, max_probes):
        self.queue_stop.clear()

        def on_run(run):
            self.current_run = run
        for cfg in configs:
            if self.queue_stop.is_set():
                break
            trainer.limit_search(cfg, axis, self.hub.publish, self.corpus, self.queue_stop, probe_steps=probe_steps,
                                 ram_floor=int(ram_floor_gb * trainer.GB), max_probes=max_probes, on_run=on_run)
        self.hub.publish({"type": "limit_all_done"})

    def run_report(self, base_cfg, engines, allow_spill, parity):
        self.queue_stop.clear()

        def on_run(run):
            self.current_run = run
        system = {k: self.info.get(k) for k in ("gpu", "vram_total", "host_ram_total", "cpu", "cpu_threads", "driver",
                                                 "torch", "cuda", "python", "trainstream")}
        report.run_suite(base_cfg, engines, self.hub.publish, self.corpus, self.queue_stop,
                         self.info.get("tensorflow") or {}, system, allow_spill=allow_spill, parity=parity,
                         on_run=on_run)

    # -- maintenance -------------------------------------------------------
    def clear(self, free_memory: bool, delete_reports: bool) -> dict:
        """Forget run history (all clients reset), optionally release caches / saved reports."""
        with self.hub._lock:
            self.hub.history.clear()
        trainer.Run._ids = 0
        freed = {}
        if free_memory:
            before_vram = torch.cuda.memory_reserved()
            before_pinned = trainer._C.pinned_stats()["reserved_bytes"]
            self._corpus.clear()
            gc.collect()
            torch.cuda.empty_cache()
            trainer._C.pinned_empty_cache()
            freed = {"vram_bytes": before_vram - torch.cuda.memory_reserved(),
                     "pinned_bytes": before_pinned - trainer._C.pinned_stats()["reserved_bytes"]}
        deleted = 0
        if delete_reports and os.path.isdir(report.REPORT_DIR):
            for f in os.listdir(report.REPORT_DIR):
                if re.fullmatch(r"report-[A-Za-z0-9_.-]+\.(html|json)", f):
                    os.remove(os.path.join(report.REPORT_DIR, f))
                    deleted += 1
        self.hub.publish({"type": "cleared", "freed": freed, "deleted_reports": deleted})
        msg = "dashboard cleared"
        if freed:
            msg += f" · released {freed['vram_bytes'] / 2**30:.2f} GiB cached VRAM and {freed['pinned_bytes'] / 2**30:.2f} GiB pinned RAM"
        if delete_reports:
            msg += f" · deleted {deleted} saved report file(s)"
        self.hub.publish({"type": "log", "level": "info", "t": time.time(), "msg": msg})
        return {"cleared": True, "freed": freed, "deleted_reports": deleted}

    def restart(self) -> None:
        """Replace this process with a fresh server (new CUDA context, all memory released)."""
        try:
            self._restart()
        except Exception as exc:  # never leave the page waiting for a server that is not coming
            self.hub.publish({"type": "log", "level": "error", "t": time.time(),
                              "msg": f"fresh restart failed: {type(exc).__name__}: {exc}"})
            self.hub.publish({"type": "restart_failed", "msg": f"{type(exc).__name__}: {exc}"})

    def _restart(self) -> None:
        self.hub.publish({"type": "restarting", "boot_id": BOOT_ID, "t": time.time()})
        self.queue_stop.set()
        if self.current_run is not None:
            self.current_run.stop_flag.set()
        time.sleep(0.5)  # let the response and the SSE event flush
        kill_tree(os.getpid())  # TensorFlow workers and other helpers
        env = dict(os.environ, TRAINSTREAM_WAIT_FOR_PORT="60")
        # One log file per launch: a running server keeps its own log open (its
        # stdout), so reusing one file name can fail with a sharing violation.
        log_path = RESTART_LOG.replace(".log", time.strftime("-%Y%m%d-%H%M%S.log"))
        log = open(log_path, "a", encoding="utf-8")  # the detached server has no console
        log.write(f"\n--- fresh restart {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log.flush()
        cmd = [sys.executable] + self.argv
        kw = dict(cwd=os.getcwd(), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                  close_fds=True)
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            # If this server runs inside a Windows job object (IDE terminals, task
            # runners, sandboxes), the job's owner kills every member when it cleans
            # up, which would take the new server down too. Escape the job: break
            # away when the job allows it, otherwise have WMI start the process
            # (it is then created by a system service, outside any job of ours).
            try:
                subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
                how = "detached, broke away from job"
            except OSError:
                log.close()
                if launch_outside_job(cmd, os.getcwd(), log_path, {"TRAINSTREAM_WAIT_FOR_PORT": "60"}):
                    how = "started via WMI (outside the job)"
                else:
                    log = open(log_path, "a", encoding="utf-8")
                    kw["stdout"] = log
                    subprocess.Popen(cmd, creationflags=flags, **kw)
                    how = "detached (could not leave the job)"
            with open(log_path.replace(".log", ".launch.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"new server {how}\n")
        else:
            subprocess.Popen(cmd, start_new_session=True, **kw)
        try:
            if self.server is not None:
                self.server.server_close()
        finally:
            os._exit(0)

    def run_profile(self, quick):
        prof = __import__("trainstream").profile_hardware(0, quick=quick, force=True)
        self.hub.publish({"type": "profile", "profile": prof.to_dict()})

    def run_parity(self):
        rows = trainer.parity_check(self.hub.publish)
        self.hub.publish({"type": "parity_done", "rows": rows})

    def run_bench(self):
        rows = trainer.optimizer_bench(self.hub.publish)
        self.hub.publish({"type": "bench_done", "rows": rows})


def normalize_config(body: dict) -> dict:
    preset = body.get("preset", "small-85M")
    if preset != "custom" and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}")
    base = PRESETS.get(preset, PRESETS["small-85M"])
    framework = body.get("framework", "pytorch")
    device = body.get("device", "gpu")
    if framework not in ("pytorch", "tensorflow"):
        raise ValueError("framework must be pytorch or tensorflow")
    if device not in ("gpu", "cpu"):
        raise ValueError("device must be gpu or cpu")
    engine = body.get("engine", "trainstream")
    if framework == "tensorflow" or device == "cpu":
        engine = "baseline"  # TrainStream is a PyTorch + CUDA extension
    cfg = {
        "preset": preset,
        "framework": framework,
        "device": device,
        "n_layer": int(body.get("n_layer") or base.n_layer),
        "n_head": int(body.get("n_head") or base.n_head),
        "d_model": int(body.get("d_model") or base.d_model),
        "engine": engine,
        "optimizer_mode": body.get("optimizer_mode", "cpu"),
        "activation_offload": body.get("activation_offload", "auto"),
        "param_dtype": body.get("param_dtype", "fp32"),
        "batch": int(body.get("batch", 8)),
        "seq_len": int(body.get("seq_len", 512)),
        "steps": int(body.get("steps", 200)),
        "lr": float(body.get("lr", 6e-4)),
        "max_grad_norm": float(body["max_grad_norm"]) if body.get("max_grad_norm") not in (None, "", 0, "0") else None,
        "prefetch_depth": int(body.get("prefetch_depth", 1)),
        "eval_every": int(body.get("eval_every", 50)),
        "seed": int(body.get("seed", 0)),
    }
    if cfg["engine"] not in ("trainstream", "baseline"):
        raise ValueError("engine must be trainstream or baseline")
    if cfg["optimizer_mode"] not in ("cpu", "gpu"):
        raise ValueError("optimizer_mode must be cpu or gpu")
    if cfg["param_dtype"] not in ("fp32", "bf16"):
        raise ValueError("param_dtype must be fp32 or bf16")
    if framework == "tensorflow":
        cfg["param_dtype"] = "fp32"
    if not (1 <= cfg["batch"] <= 4096 and 16 <= cfg["seq_len"] <= 8192 and 1 <= cfg["steps"] <= 100000):
        raise ValueError("batch 1–4096, seq_len 16–8192, steps 1–100000")
    if not (1 <= cfg["n_layer"] <= 512 and 1 <= cfg["n_head"] <= 256 and 64 <= cfg["d_model"] <= 32768):
        raise ValueError("layers 1–512, heads 1–256, d_model 64–32768")
    if cfg["d_model"] % cfg["n_head"]:
        raise ValueError(f"d_model ({cfg['d_model']}) must be divisible by heads ({cfg['n_head']})")
    return cfg


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _json(self, obj, status=200):
            data = json.dumps(_sanitize(obj)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _require_tf(self, configs):
            if any(c["framework"] == "tensorflow" for c in configs):
                tf = app.info.get("tensorflow") or {}
                if not tf.get("available"):
                    raise ValueError("TensorFlow is not available: " + str(tf.get("error") or "still being checked"))
                if any(c["framework"] == "tensorflow" and c["device"] == "gpu" for c in configs) and not tf.get("gpus"):
                    raise ValueError("TensorFlow sees no GPU in its interpreter; choose CPU")

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as fh:
                    data = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/system":
                info = dict(app.info)
                total, avail = trainer.host_memory()
                info.update(host_ram_avail=avail, busy=app.busy(), job=app.job, boot_id=BOOT_ID,
                            uptime_s=time.time() - BOOT_TIME)
                self._json(info)
            elif self.path == "/api/events":
                self._sse()
            elif self.path == "/api/reports":
                os.makedirs(report.REPORT_DIR, exist_ok=True)
                names = sorted((f[:-5] for f in os.listdir(report.REPORT_DIR) if f.endswith(".json")), reverse=True)
                self._json({"reports": names[:50]})
            elif self.path.startswith("/reports/"):
                name = self.path[len("/reports/"):].split("?")[0]
                ok = re.fullmatch(r"report-[A-Za-z0-9_.-]+\.(html|json)", name)  # no path separators
                path = os.path.join(report.REPORT_DIR, name)
                if not ok or not os.path.isfile(path):
                    self._json({"error": "not found"}, 404)
                    return
                with open(path, "rb") as fh:
                    data = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if name.endswith(".html") else "application/json")
                if self.path.endswith("?download=1"):
                    self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/api/estimate":
                    cfg = normalize_config(body)
                    mcfg = trainer.model_config(cfg)
                    out = {e: trainer.estimate(mcfg, cfg["param_dtype"], e, cfg["optimizer_mode"])
                           for e in ("baseline", "trainstream")}
                    out["selected"] = trainer.estimate(mcfg, cfg["param_dtype"], cfg["engine"], cfg["optimizer_mode"],
                                                       cfg["framework"], cfg["device"])
                    out["config"] = cfg
                    self._json(out)
                elif self.path == "/api/run":
                    runs = body.get("runs") or [body]
                    configs = [normalize_config(r) for r in runs]
                    self._require_tf(configs)
                    ok = app.start_job("train", lambda: app.run_training(configs))
                    self._json({"started": ok, "busy": not ok}, 200 if ok else 409)
                elif self.path == "/api/report":
                    base = normalize_config(dict(body.get("config") or {}, framework="pytorch", device="gpu",
                                                 engine="trainstream"))
                    engines = [e for e in (body.get("engines") or ["tensorflow", "adamw", "trainstream"])
                               if e in report.ENGINES]
                    if not engines:
                        raise ValueError("choose at least one engine")
                    allow_spill = bool(body.get("allow_spill", False))
                    parity = bool(body.get("parity", True))
                    ok = app.start_job("report", lambda: app.run_report(base, engines, allow_spill, parity))
                    self._json({"started": ok, "busy": not ok}, 200 if ok else 409)
                elif self.path == "/api/limit":
                    configs = [normalize_config(c) for c in (body.get("configs") or [body.get("config") or {}])]
                    self._require_tf(configs)
                    axis = body.get("axis", "n_layer")
                    if axis not in trainer.AXES:
                        raise ValueError(f"axis must be one of {list(trainer.AXES)}")
                    probe_steps = max(2, min(int(body.get("probe_steps", 6)), 100))
                    floor = max(1.0, min(float(body.get("ram_floor_gb", 4)), 64.0))
                    max_probes = max(2, min(int(body.get("max_probes", 14)), 40))
                    ok = app.start_job("limit", lambda: app.run_limit(configs, axis, probe_steps, floor, max_probes))
                    self._json({"started": ok}, 200 if ok else 409)
                elif self.path == "/api/stop":
                    app.queue_stop.set()
                    if app.current_run is not None:
                        app.current_run.stop_flag.set()
                    self._json({"stopping": True})
                elif self.path == "/api/profile":
                    quick = bool(body.get("quick", True))
                    ok = app.start_job("profile", lambda: app.run_profile(quick))
                    self._json({"started": ok}, 200 if ok else 409)
                elif self.path == "/api/parity":
                    ok = app.start_job("parity", app.run_parity)
                    self._json({"started": ok}, 200 if ok else 409)
                elif self.path == "/api/bench":
                    ok = app.start_job("bench", app.run_bench)
                    self._json({"started": ok}, 200 if ok else 409)
                elif self.path == "/api/clear":
                    if app.busy():
                        self._json({"error": f"a {app.job} job is running; stop it first (or use Fresh restart)"}, 409)
                    else:
                        self._json(app.clear(bool(body.get("free_memory", True)), bool(body.get("delete_reports", False))))
                elif self.path == "/api/restart":
                    self._json({"restarting": True, "boot_id": BOOT_ID})
                    threading.Thread(target=app.restart, daemon=True, name="restart").start()
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = app.hub.subscribe()
            try:
                self.wfile.write(b"retry: 2000\n\n")
                while True:
                    try:
                        item = q.get(timeout=15)
                        chunk = [item]
                        while len(chunk) < 500:  # batch replay / bursts into one write
                            try:
                                chunk.append(q.get_nowait())
                            except queue.Empty:
                                break
                        self.wfile.write("".join(f"data: {c}\n\n" for c in chunk).encode())
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                app.hub.unsubscribe(q)

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Browsers drop idle keep-alive / SSE connections on reload or tab close.
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    app = App()
    # After a "Fresh restart" the previous process may still hold the port for a
    # moment: keep retrying the bind instead of failing.
    deadline = time.time() + float(os.environ.pop("TRAINSTREAM_WAIT_FOR_PORT", "0") or 0)
    while True:
        try:
            server = Server((args.host, args.port), make_handler(app))
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.5)
    app.server = server
    print(f"TrainStream dashboard on http://{args.host}:{args.port}  ({app.info['gpu']})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
