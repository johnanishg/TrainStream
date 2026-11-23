"""Checkpoints written in the background, while training goes on.

    handle = trainstream.save_checkpoint(model, optimizer, "run/step-500.ckpt")   # returns at once
    ...                                    # keep training; handle.done() / handle.wait()
    extra = trainstream.load_checkpoint("run/step-500.ckpt", model, optimizer)    # resume

A checkpoint holds what training changes: the trained parameters and the
optimizer's state (fp32 master weights, both moments in the precision they are
kept in, step counts, the param groups' hyper-parameters), plus an `extra`
dict of your own (anything JSON can hold: the step, the scheduler's state).
Frozen weights are not in it; load the base model the way you did before.

`torch.save(optimizer.state_dict())` stops everything: the optimizer is
synchronized, every tensor is copied and pickled on the training thread, and
the GPU idles until the file is closed. Here the call only describes the file;
a detached C++ thread (csrc/ckpt.cpp) writes it with asynchronous unbuffered
I/O straight from the optimizer's host buffers, without copying them. Each
tensor's *next* update waits only until that tensor is on disk, and tensors
are written in the order the next step updates them, so training waits for a
save only where the SSD is slower than a whole step. State that lives in VRAM
(a resident optimizer, fp32 weights in gpu mode) is copied out by asynchronous
D2H transfers into pinned memory first. The file appears under its final name
only once it is complete (written as `<path>.tmp`, flushed, renamed), so a
crash during a save leaves the previous checkpoint as it was.

A run resumed from a checkpoint continues exactly as the saved one would have:
the masters, moments (raw bytes, 8-bit ones included) and step counts are
restored bit for bit.

File layout: 8 bytes "TSCKPT01", an 8-byte little-endian length, a JSON
manifest, then every tensor at an offset that is a multiple of 4096.
`read_checkpoint(path)` returns the manifest and the tensors.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from . import _C

MAGIC = b"TSCKPT01"
ALIGN = 4096
VERSION = 1

_DTYPES = {str(t): t for t in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.uint8, torch.int8,
                                torch.int16, torch.int32, torch.int64, torch.bool)}


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class CheckpointHandle:
    """A checkpoint being written (or written) in the background."""

    def __init__(self, job, path: str):
        self._job = job
        self.path = path

    def done(self) -> bool:
        """True once the file is complete under its final name, or the write failed."""
        return bool(self._job.done())

    def wait(self) -> "CheckpointHandle":
        """Block until the write has finished; raises OSError if it failed."""
        self._job.wait()
        if self._job.error:
            raise OSError(f"TrainStream: writing checkpoint {self.path} failed: {self._job.error}")
        return self

    @property
    def error(self) -> Optional[str]:
        """What went wrong, once done (None: nothing)."""
        return self._job.error or None

    @property
    def seconds(self) -> float:
        """Wall time of the background write, once done."""
        return float(self._job.seconds)

    @property
    def bytes(self) -> int:
        """Bytes written, once done."""
        return int(self._job.bytes)

    def __repr__(self) -> str:
        state = ("failed: " + self.error) if self.error else ("done" if self.done() else "writing")
        return f"CheckpointHandle({self.path!r}, {state})"


def _snapshot(t: torch.Tensor):
    """(pinned host tensor, event): an asynchronous D2H copy of CUDA tensor `t` as it is now."""
    src = t.detach()
    if not src.is_contiguous():
        src = src.contiguous()
    host = _C.pinned_empty([src.numel()], src.dtype, "checkpoint", False, True)
    ev = _C.d2h_async(host, src.reshape(-1))
    # the D2H stream reads `src` until the event: nothing else may change or free it before then
    return host, ev, src


def save_checkpoint(model: Optional[nn.Module], optimizer, path: str, *, extra: Optional[Dict[str, Any]] = None,
                    blocking: bool = False) -> CheckpointHandle:
    """Write the trained parameters and the optimizer state to `path` on a background thread.

    Args:
        model: the model being trained (wrapped or not). Only used to record each trained
            parameter's name; None records positions only.
        optimizer: the OffloadAdamW (what trainstream.wrap returned).
        path: the checkpoint file. It appears only once complete; `<path>.tmp` exists meanwhile.
        extra: a JSON-serializable dict stored in the file and returned by load_checkpoint
            (the global step, an LR scheduler's state_dict(), ...).
        blocking: wait for the write before returning (default False: return at once).

    Returns a CheckpointHandle: done(), wait(), error, seconds, bytes, path. Call it between
    optimizer.step() and the next backward pass. A save started while another is still being
    written waits for that one first.
    """
    from .optimizer import OffloadAdamW
    if not isinstance(optimizer, OffloadAdamW):
        raise TypeError("save_checkpoint needs the OffloadAdamW that trainstream.wrap() returned")
    opt = optimizer
    if opt._suspended is not None:
        raise RuntimeError("TrainStream: the model is off the GPU; move it back before saving a checkpoint")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    names = {id(p): n for n, p in model.named_parameters()} if model is not None else {}
    device = opt.device.index or 0

    tensors: List[dict] = []   # manifest entries (offsets relative to the start of the data)
    items: List[tuple] = []    # (host tensor, relative offset, event, slot, release)
    slots: List[dict] = []
    keep: List[torch.Tensor] = []
    cursor = 0

    def add(key: str, t: torch.Tensor, event=None, slot: int = -1, release: bool = False) -> None:
        nonlocal cursor
        flat = t.reshape(-1)
        tensors.append({"key": key, "dtype": str(flat.dtype), "numel": flat.numel(), "offset": cursor,
                        "nbytes": flat.numel() * flat.element_size()})
        items.append((flat, cursor, event, slot, release))
        cursor += _up(flat.numel() * flat.element_size())

    engine = opt.engine if opt._resident is None else None
    targets: List[int] = []
    steps = None
    try:
        if engine is not None:
            if opt._dp is not None:
                opt._dp["queue"].join()  # every gradient of the last step has been reduced
            order = opt._update_order()  # the order the next step updates (and so needs) them
            targets = list(engine.checkpoint_begin())
            steps = torch.zeros(len(opt._slot_params), dtype=torch.int64)
            for slot in order:
                p = opt._slot_params[slot]
                raw = dict(engine.raw_state(slot))
                parts = [(k, raw[k], None) for k in ("master", "exp_avg", "exp_avg_sq", "m_scale", "v_scale",
                                                     "param_stage") if k in raw]
                if "master" not in raw:  # gpu mode, fp32 weights: they exist only in VRAM
                    engine.wait_ready([slot])  # the D2H copy is ordered after the last update kernel
                    host, ev, src = _snapshot(p)
                    keep.append(src)
                    parts.append(("param", host, ev))
                for n, (k, t, ev) in enumerate(parts):
                    add(f"slot/{slot}/{k}", t, ev, slot, n == len(parts) - 1)
                slots.append({"slot": slot, "name": names.get(id(p)), "shape": list(p.shape), "dtype": str(p.dtype),
                              "parts": [k for k, _, _ in parts]})
        else:  # resident: torch.optim.AdamW(fused=True), everything in VRAM
            inner = opt._resident
            events = []
            for slot, p in enumerate(opt._slot_params):
                st = inner.state.get(p)
                parts = [("param", p)]
                if st:
                    parts += [("exp_avg", st["exp_avg"]), ("exp_avg_sq", st["exp_avg_sq"]),
                              ("step", torch.as_tensor(st["step"], dtype=torch.float32, device=p.device).reshape(1))]
                for k, t in parts:
                    host, ev, src = _snapshot(t)
                    keep.append(src)
                    events.append(ev)
                    add(f"slot/{slot}/{k}", host, ev)
                slots.append({"slot": slot, "name": names.get(id(p)), "shape": list(p.shape), "dtype": str(p.dtype),
                              "parts": [k for k, _ in parts]})
            opt._ckpt_events = events  # the next optimizer step waits (on the GPU) for these copies
        groups = []
        index = {id(p): i for i, p in enumerate(opt._slot_params)}
        for g in opt.param_groups:
            packed = {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()
                      if k != "params" and isinstance(v, (int, float, bool, str, tuple, list, type(None)))}
            packed["slots"] = [index[id(p)] for p in g["params"] if id(p) in index]
            groups.append(packed)
        steps_rel = None
        if steps is not None:
            steps_rel = cursor
            tensors.append({"key": "steps", "dtype": "torch.int64", "numel": steps.numel(), "offset": cursor,
                            "nbytes": steps.numel() * 8})
            cursor += _up(steps.numel() * 8)
        manifest = {"format": "trainstream-checkpoint", "version": VERSION,
                    "mode": "resident" if engine is None else opt.mode, "moment_dtype": opt.moment_dtype,
                    "param_groups": groups, "slots": slots, "tensors": tensors,
                    "extra": extra if extra is not None else {}}
        body = json.dumps(manifest).encode("utf-8")
        header = MAGIC + struct.pack("<Q", len(body)) + body
        start = _up(len(header))
        header += b"\0" * (start - len(header))
        job = _C.checkpoint_write_async(os.path.abspath(path), header,
                                        [(t, start + off, ev, slot, rel) for t, off, ev, slot, rel in items],
                                        engine, targets, steps, start + (steps_rel or 0), device)
        targets = []  # the writer owns the holds now
    except BaseException:
        if engine is not None and targets:
            engine.checkpoint_end()  # nothing will be written: let the updates go on
        raise
    handle = CheckpointHandle(job, os.path.abspath(path))
    handle._keep = keep  # VRAM tensors the D2H copies read from
    opt._ckpt_last = handle
    if blocking:
        handle.wait()
    return handle


class _Reader:
    """A checkpoint file opened once: the manifest, and tensors on demand."""

    def __init__(self, path: str):
        self.path = path
        self.fh = open(path, "rb")
        try:
            head = self.fh.read(16)
            if len(head) < 16 or head[:8] != MAGIC:
                raise ValueError(f"{path} is not a TrainStream checkpoint")
            (n,) = struct.unpack("<Q", head[8:])
            self.manifest = json.loads(self.fh.read(n).decode("utf-8"))
            if self.manifest.get("version", 0) > VERSION:
                raise ValueError(f"{path} was written by a newer TrainStream (checkpoint version "
                                 f"{self.manifest['version']})")
            self.start = _up(16 + n)
            self.index = {t["key"]: t for t in self.manifest["tensors"]}
        except BaseException:
            self.fh.close()
            raise

    def tensor(self, key: str) -> torch.Tensor:
        t = self.index[key]
        dtype = _DTYPES[t["dtype"]]
        if not t["nbytes"]:
            return torch.empty(0, dtype=dtype)
        buf = bytearray(t["nbytes"])
        self.fh.seek(self.start + t["offset"])
        if self.fh.readinto(buf) != t["nbytes"]:
            raise ValueError(f"{self.path} is truncated (tensor {key})")
        return torch.frombuffer(buf, dtype=dtype)

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_checkpoint(path: str, keys=None) -> Dict[str, Any]:
    """{"manifest": the file's manifest (dict), "tensors": {key: flat CPU tensor}}; `keys` limits
    which tensors are read (None: all; an empty list: the manifest only). Keys are
    "slot/<n>/<part>" (parts: master, exp_avg, exp_avg_sq, m_scale, v_scale, param_stage, param,
    step) and "steps"."""
    with _Reader(path) as r:
        want = list(r.index) if keys is None else [k for k in keys if k in r.index]
        return {"manifest": r.manifest, "tensors": {k: r.tensor(k) for k in want}}


def _moment32(saved_dtype: str, parts: dict, name: str) -> torch.Tensor:
    """A saved moment as fp32, whatever it was stored in."""
    t = parts[name]
    if saved_dtype == "int8" and t.dtype == torch.uint8:
        return _C.decode_moment8(t, parts["m_scale" if name == "exp_avg" else "v_scale"], name == "exp_avg")
    return t.to(torch.float32)


_LOAD_CHUNK = 4 << 30  # trained values handed to model.load_state_dict at a time


def load_checkpoint(path: str, model: Optional[nn.Module], optimizer) -> Dict[str, Any]:
    """Restore the trained parameters and the optimizer state saved by save_checkpoint.

    Args:
        path: the checkpoint file.
        model: the model the optimizer trains, built and wrapped as when the checkpoint was
            saved (required when weights are paged; None sets parameters directly).
        optimizer: its OffloadAdamW.

    The parameters must be the same ones, in the same order, with the same shapes. The
    optimizer may differ from the saving one (mode, moment_dtype, resident or not): moments are
    converted, and restored bit for bit where the storage is the same. Returns the `extra`
    dict that was saved.
    """
    from .optimizer import OffloadAdamW
    if not isinstance(optimizer, OffloadAdamW):
        raise TypeError("load_checkpoint needs the OffloadAdamW that trainstream.wrap() returned")
    opt = optimizer
    with _Reader(path) as r:
        manifest = r.manifest
        saved = {s["slot"]: s for s in manifest["slots"]}
        params = opt._slot_params
        if len(saved) != len(params):
            raise ValueError(f"{path} holds {len(saved)} trained parameters, the optimizer has {len(params)}")
        for slot, p in enumerate(params):
            if list(p.shape) != saved[slot]["shape"]:
                raise ValueError(f"trained parameter {slot} ({saved[slot]['name']}) has shape {tuple(p.shape)}, the "
                                 f"checkpoint {tuple(saved[slot]['shape'])}")
        opt.synchronize()
        last = getattr(opt, "_ckpt_last", None)
        if last is not None:
            last._job.wait()  # a save still reading this state
        steps = r.tensor("steps") if "steps" in r.index else None
        names = {id(p): n for n, p in model.named_parameters()} if model is not None else {}
        mdt = manifest.get("moment_dtype", "float32")
        inner = opt._resident

        # ---- the parameters' values: through the model where there is one, so paged weights and
        # their host copies are set the pager's way; in few calls (each walks every block)
        pending, pending_bytes = {}, 0

        def flush():
            nonlocal pending, pending_bytes
            if pending:
                model.load_state_dict(pending, strict=False)
            pending, pending_bytes = {}, 0

        for slot, p in enumerate(params):
            parts = saved[slot]["parts"]
            src = "param_stage" if "param_stage" in parts else "param" if "param" in parts else "master"
            value = r.tensor(f"slot/{slot}/{src}").to(p.dtype).view(p.shape)
            if model is not None and id(p) in names:
                pending[names[id(p)]] = value
                pending_bytes += value.numel() * value.element_size()
                if pending_bytes >= _LOAD_CHUNK:
                    flush()
            else:
                if p.untyped_storage().nbytes() == 0 and p.numel():
                    raise ValueError("load_checkpoint: this parameter's weights are paged; pass the model")
                with torch.no_grad():
                    p.copy_(value.to(p.device))
        flush()
        opt.synchronize()

        # ---- the optimizer's state, one parameter at a time
        for slot, p in enumerate(params):
            info = saved[slot]
            if "exp_avg" not in info["parts"]:
                continue  # saved before its first update: nothing to restore
            parts = {k: r.tensor(f"slot/{slot}/{k}") for k in info["parts"] if k not in ("param", "param_stage")}
            step = (int(steps[slot]) if steps is not None
                    else int(round(float(parts["step"][0]))) if "step" in parts else 0)
            if inner is not None:
                m = _moment32(mdt, parts, "exp_avg").view(p.shape).to(p.device, p.dtype)
                v = _moment32(mdt, parts, "exp_avg_sq").view(p.shape).to(p.device, p.dtype)
                inner.state[p] = {"step": torch.tensor(float(step), device=p.device, dtype=torch.float32),
                                  "exp_avg": m, "exp_avg_sq": v}
                continue
            engine = opt.engine
            mine = dict(engine.raw_state(slot))
            state_keys = ("master", "exp_avg", "exp_avg_sq", "m_scale", "v_scale")
            same = all((k in mine) == (k in parts) and (k not in mine or (mine[k].dtype == parts[k].dtype
                                                                         and mine[k].numel() == parts[k].numel()))
                       for k in state_keys)
            if same:  # the same storage as the saving optimizer's: bit for bit
                engine.load_state_raw(slot, {k: parts[k] for k in state_keys if k in parts}, step)
            else:
                engine.load_state(slot, _moment32(mdt, parts, "exp_avg"), _moment32(mdt, parts, "exp_avg_sq"),
                                  parts.get("master"), step)
        for group, packed in zip(opt.param_groups, manifest.get("param_groups", [])):
            for k, v in packed.items():
                if k in opt.defaults:
                    group[k] = tuple(v) if k == "betas" else v
        if inner is not None:
            opt.state = inner.state
        return manifest.get("extra", {})
