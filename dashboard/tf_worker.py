"""TensorFlow training worker for the dashboard (runs in a separate interpreter).

    <tf-python> dashboard/tf_worker.py --probe
    <tf-python> dashboard/tf_worker.py --run '<json config>'

TensorFlow cannot share a process with the dashboard's PyTorch/CUDA context and
needs its own Python version, so the server launches this script and reads one
JSON event per stdout line. The model is the same byte-level GPT as model.py.
A run ends after the current step once the config's `stop_file` exists.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def process_memory():
    """(private bytes, peak private bytes) of this process."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.PrivateUsage), int(c.PeakPagefileUsage)
    try:
        import resource
        with open("/proc/self/status") as fh:
            rss = next(int(l.split()[1]) * 1024 for l in fh if l.startswith("VmRSS"))
        return rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return 0, 0


def import_tf(device):
    import tensorflow as tf
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass  # DirectML manages its own heap
    return tf


def probe():
    try:
        tf = import_tf("gpu")
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.list_logical_devices("GPU")  # creates the devices, which logs the adapter names
        names = []
        for g in gpus:
            try:
                names.append(tf.config.experimental.get_device_details(g).get("device_name", g.name))
            except Exception:
                names.append(g.name)
        backend = "DirectML" if any("DML" in g.name or "DML" in g.device_type for g in gpus) else ("CUDA" if gpus else "CPU only")
        try:
            import importlib.metadata as md
            if md.version("tensorflow-directml-plugin"):
                backend = "DirectML" if gpus else backend
        except Exception:
            pass
        emit({"available": True, "version": tf.__version__, "python": sys.version.split()[0],
              "gpus": names, "gpu_backend": backend})
    except Exception as exc:
        emit({"available": False, "error": f"{type(exc).__name__}: {exc}", "python": sys.version.split()[0]})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(tf, cfg):
    d, h, L, T, V = cfg["d_model"], cfg["n_head"], cfg["n_layer"], cfg["seq_len"], 256
    seed = iter(range(int(cfg.get("seed", 0)) * 100003, 1 << 30))
    init = lambda: tf.keras.initializers.RandomNormal(stddev=0.02, seed=next(seed))  # noqa: E731
    out_init = lambda: tf.keras.initializers.RandomNormal(stddev=0.02 / math.sqrt(2 * L), seed=next(seed))  # noqa: E731

    class Block(tf.keras.layers.Layer):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.qkv = tf.keras.layers.Dense(3 * d, kernel_initializer=init())
            self.proj = tf.keras.layers.Dense(d, kernel_initializer=out_init())
            self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.fc = tf.keras.layers.Dense(4 * d, kernel_initializer=init())
            self.fc_out = tf.keras.layers.Dense(d, kernel_initializer=out_init())

        def call(self, x):
            b, t = tf.shape(x)[0], tf.shape(x)[1]
            q, k, v = tf.split(self.qkv(self.ln1(x)), 3, axis=-1)
            heads = lambda z: tf.transpose(tf.reshape(z, (b, t, h, d // h)), (0, 2, 1, 3))  # noqa: E731
            q, k, v = heads(q), heads(k), heads(v)
            att = tf.matmul(q, k, transpose_b=True) * (1.0 / math.sqrt(d // h))
            mask = tf.linalg.band_part(tf.ones((t, t)), -1, 0)
            att = att + (1.0 - mask) * -1e9
            y = tf.matmul(tf.nn.softmax(att, axis=-1), v)
            y = tf.reshape(tf.transpose(y, (0, 2, 1, 3)), (b, t, d))
            x = x + self.proj(y)
            return x + self.fc_out(tf.nn.gelu(self.fc(self.ln2(x)), approximate=True))

    class GPT(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.tok = tf.keras.layers.Embedding(V, d, embeddings_initializer=init())
            self.pos = tf.keras.layers.Embedding(T, d, embeddings_initializer=init())
            self.blocks = [Block() for _ in range(L)]
            self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.head = tf.keras.layers.Dense(V, use_bias=False, kernel_initializer=init())

        def call(self, idx):
            t = tf.shape(idx)[1]
            x = self.tok(idx) + self.pos(tf.range(t))
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x))

    model = GPT()
    model(tf.zeros((1, T), dtype=tf.int32))
    return model


class Corpus:
    def __init__(self, max_bytes=24 << 20, val_fraction=0.02):
        import numpy as np
        root = os.path.join(sys.base_prefix, "Lib")
        chunks, total = [], 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(x for x in dirnames if x not in ("site-packages", "test", "tests", "__pycache__"))
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    try:
                        with open(os.path.join(dirpath, fn), "rb") as fh:
                            chunks.append(fh.read())
                    except OSError:
                        continue
                    total += len(chunks[-1])
                    if total >= max_bytes:
                        break
            if total >= max_bytes:
                break
        data = np.frombuffer(b"\n".join(chunks), dtype=np.uint8)
        split = int(len(data) * (1 - val_fraction))
        self.train, self.val = data[:split], data[split:]
        self.files, self.nbytes = len(chunks), len(data)
        self.np = np

    def batch(self, rng, batch, seq, split="train"):
        src = self.train if split == "train" else self.val
        ix = rng.integers(0, len(src) - seq - 1, size=batch)
        win = src[ix[:, None] + self.np.arange(seq + 1)].astype(self.np.int32)
        return win[:, :-1], win[:, 1:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run(cfg):
    stop_file = cfg.get("stop_file")
    stopped = lambda: bool(stop_file) and os.path.exists(stop_file)  # noqa: E731

    log = lambda msg, level="info": emit({"type": "log", "level": level, "msg": msg})  # noqa: E731
    summary = {"status": "running"}
    step = 0
    try:
        tf = import_tf(cfg["device"])
        import numpy as np
        dev_name = "/GPU:0" if cfg["device"] == "gpu" else "/CPU:0"
        if cfg["device"] == "gpu" and not tf.config.list_logical_devices("GPU"):
            raise RuntimeError("TensorFlow sees no GPU in this interpreter")
        if cfg.get("param_dtype") == "bf16":
            log("TensorFlow runs keep fp32 weights (bf16 is not supported by the DirectML backend)", "warn")
        corpus = Corpus()
        log(f"TF {tf.__version__} on {dev_name}: corpus {corpus.files} files, {corpus.nbytes / 2**20:.1f} MiB")
        tf.random.set_seed(int(cfg.get("seed", 0)))
        rng = np.random.default_rng(1234)
        val_rng_seed = 99
        mem0, _ = process_memory()
        with tf.device(dev_name):
            t0 = time.perf_counter()
            model = build_model(tf, cfg)
            params = model.trainable_variables
            n_params = int(sum(int(np.prod(v.shape)) for v in params))
            clip = float(cfg["max_grad_norm"]) if cfg.get("max_grad_norm") else None
            # Hand-written AdamW (betas 0.9/0.95, eps 1e-8, decoupled decay 0.1, as
            # the PyTorch runs). Keras' AdamW and assign_sub/assign_add leak device
            # memory every step on the DirectML plugin; plain assign() does not.
            b1, b2, eps, wd = 0.9, 0.95, 1e-8, 0.1
            m_state = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
            v_state = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
            t_var = tf.Variable(0.0, trainable=False)
            lr_var = tf.Variable(float(cfg["lr"]), trainable=False)
            log(f"built GPT: {n_params / 1e6:.1f}M params in {time.perf_counter() - t0:.1f}s, optimizer AdamW")
            emit({"type": "model", "params": n_params})

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    loss = loss_fn(y, model(x, training=True))
                grads = [tf.convert_to_tensor(g) for g in tape.gradient(loss, params)]
                gnorm = tf.linalg.global_norm(grads)
                if clip:
                    grads, _ = tf.clip_by_global_norm(grads, clip, use_norm=gnorm)
                t = t_var + 1.0
                t_var.assign(t)
                c1 = 1.0 - tf.pow(b1, t)
                c2 = 1.0 - tf.pow(b2, t)
                for p, g, m, v in zip(params, grads, m_state, v_state):
                    m_new = b1 * m + (1.0 - b1) * g
                    v_new = b2 * v + (1.0 - b2) * tf.square(g)
                    m.assign(m_new)
                    v.assign(v_new)
                    p.assign(p * (1.0 - lr_var * wd) - lr_var * (m_new / c1) / (tf.sqrt(v_new / c2) + eps))
                return loss, gnorm

            @tf.function
            def eval_step(x, y):
                return loss_fn(y, model(x, training=False))

            @tf.function
            def next_logits(window, pos):
                # Fixed (1, seq) input: a new shape per token would make DirectML
                # compile and cache kernels for every length.
                return model(window, training=False)[0, pos]

            def sample(n_new, seq):
                toks = list(b"def ")
                for _ in range(n_new):
                    ctx = toks[-seq:]
                    window = np.zeros((1, seq), dtype=np.int32)
                    window[0, :len(ctx)] = ctx
                    logits = next_logits(tf.constant(window), tf.constant(len(ctx) - 1)).numpy() / 0.8
                    top = np.argpartition(logits, -40)[-40:]
                    p = np.exp(logits[top] - logits[top].max())
                    toks.append(int(top[rng.choice(len(top), p=p / p.sum())]))
                return bytes(toks).decode("utf-8", errors="replace")

            batch, seq, steps = int(cfg["batch"]), int(cfg["seq_len"]), int(cfg["steps"])
            warmup = min(200, max(20, steps // 5))
            eval_every = int(cfg.get("eval_every", 50))
            probe = bool(cfg.get("probe"))
            last_t = time.perf_counter()
            tokens = 0
            for step in range(1, steps + 1):
                if stopped():
                    summary["status"] = "stopped"
                    step -= 1
                    break
                lr = float(cfg["lr"]) * min(1.0, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))))
                lr_var.assign(lr)
                x, y = corpus.batch(rng, batch, seq)
                loss, gnorm = train_step(tf.constant(x), tf.constant(y))
                loss_v = float(loss.numpy())
                now = time.perf_counter()
                dt, last_t = now - last_t, now
                tokens += batch * seq
                priv, peak_priv = process_memory()
                ev = {"type": "step", "step": step, "loss": loss_v, "lr": lr, "step_ms": dt * 1e3,
                      "tokens_per_s": batch * seq / dt, "tokens": tokens, "grad_norm": float(gnorm.numpy()),
                      "proc_private": priv - mem0, "proc_peak_private": peak_priv}
                if cfg["device"] == "gpu":
                    try:
                        mi = tf.config.experimental.get_memory_info("GPU:0")
                        ev["tf_gpu_current"], ev["tf_gpu_peak"] = int(mi["current"]), int(mi["peak"])
                    except Exception:
                        pass
                emit(ev)
                if not probe and (step % eval_every == 0 or step == steps):
                    vr = np.random.default_rng(val_rng_seed)
                    vl = [float(eval_step(*map(tf.constant, corpus.batch(vr, batch, seq, "val"))).numpy()) for _ in range(4)]
                    emit({"type": "eval", "step": step, "val_loss": sum(vl) / len(vl),
                          "sample": sample(160 if cfg["device"] == "gpu" else 60, seq)})
        if summary["status"] == "running":
            summary["status"] = "completed"
        summary.update(steps=step, params=n_params)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        oom = any(s in msg for s in ("ResourceExhausted", "OOM", "out of memory", "Out of memory", "MemoryError",
                                     "E_OUTOFMEMORY", "0x8007000E", "DXGI_ERROR_DEVICE_REMOVED"))
        lines = [l.strip() for l in msg.split("\n") if l.strip() and not l.strip().startswith("File ")]
        summary.update(status="oom" if oom else "error", steps=step, error=" | ".join(lines)[:900])
        log(traceback.format_exc()[-3000:], "error")
    _, peak_priv = process_memory()
    summary["proc_peak_private"] = peak_priv
    emit({"type": "run_end", "summary": summary})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run")
    a = ap.parse_args()
    if a.probe:
        probe()
    elif a.run:
        run(json.loads(a.run))
    else:
        ap.error("--probe or --run required")


if __name__ == "__main__":
    main()
