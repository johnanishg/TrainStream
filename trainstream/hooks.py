"""Module / autograd hooks that drive TrainStream's three data paths.

* ParamReadyHooks      forward pre-hooks: a module waits (GPU-side) for its own
                       parameters' in-flight optimizer update, nothing else.
* GradientOffloader    post-accumulate-grad hooks: each gradient is streamed
                       D2H the moment autograd finishes it, then freed in VRAM.
* ActivationOffloader  saved-tensor hooks: activations of selected blocks are
                       paged out to pinned RAM during forward and prefetched
                       back block-by-block (ping-pong window) during backward.
"""

from __future__ import annotations

import collections
import statistics
import time
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from . import _C


def find_blocks(model: nn.Module) -> List[nn.Module]:
    """Locate the repeated layer stack (e.g. transformer blocks).

    Picks the ModuleList / Sequential of >= 2 same-typed children that holds the
    most parameters. Returns [] when the model has no such stack.
    """
    best: List[nn.Module] = []
    best_numel = 0
    for module in model.modules():
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        children = list(module.children())
        if len(children) < 2 or len({type(c) for c in children}) != 1:
            continue
        numel = sum(p.numel() for p in module.parameters())
        if numel > best_numel or (numel == best_numel and len(children) > len(best)):
            best, best_numel = children, numel
    return best


# ---------------------------------------------------------------------------
# Parameter readiness
# ---------------------------------------------------------------------------
class ParamReadyHooks:
    """Per-module forward pre-hooks that gate compute on pending weight uploads.

    The optimizer step only *queues* work. Before a module runs, its parameters'
    updates must have been issued (host wait, usually already done for early
    layers) and landed (a cudaStreamWaitEvent on the compute stream — the host
    does not block). The first forward pass also records the order in which
    parameters are used so the optimizer can update them in that order.
    """

    def __init__(self, model: nn.Module, engine, slot_of: Dict[int, int], order_sink: List[int]):
        self._engine = engine
        self._order = order_sink
        self._seen = set(order_sink)
        self._recording = True
        self._handles = []
        for module in model.modules():
            ids = [slot_of[id(p)] for p in module.parameters(recurse=False) if id(p) in slot_of]
            if ids:
                self._handles.append(module.register_forward_pre_hook(self._make_hook(ids)))

    def _make_hook(self, ids: List[int]):
        engine = self._engine

        def hook(module, args):
            if self._recording:
                for i in ids:
                    if i not in self._seen:
                        self._seen.add(i)
                        self._order.append(i)
            engine.wait_ready(ids)

        return hook

    def stop_recording(self) -> None:
        self._recording = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Gradient streaming
# ---------------------------------------------------------------------------
class GradientOffloader:
    """Streams each parameter's gradient to pinned host RAM as soon as it is final.

    Uses Tensor.register_post_accumulate_grad_hook, which fires after autograd
    has accumulated into ``param.grad``. ``on_grad(slot, grad)`` queues the D2H
    copy on TrainStream's D2H stream (ordered after the producing kernels via a
    CUDA event) and the VRAM gradient is dropped immediately; the caching
    allocator will not reuse the block until the copy stream has consumed it.
    """

    def __init__(self, params: Iterable[torch.Tensor], on_grad, slot_of: Dict[int, int]):
        self._on_grad = on_grad
        self._handles = []
        self.calls = 0
        self.bytes = 0
        for p in params:
            if p.requires_grad and id(p) in slot_of:
                self._handles.append(p.register_post_accumulate_grad_hook(self._make_hook(slot_of[id(p)])))

    def _make_hook(self, slot: int):
        on_grad = self._on_grad

        def hook(param: torch.Tensor):
            grad = param.grad
            if grad is None:
                return
            on_grad(slot, grad)
            self.calls += 1
            self.bytes += grad.numel() * grad.element_size()
            param.grad = None

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Activation paging
# ---------------------------------------------------------------------------
def _dense_like(t: torch.Tensor) -> torch.Tensor:
    """`t` itself if its memory is dense, else a dense copy with the same
    dimension order (strides sorted like t's, gaps removed).

    Layout matters: some fused backward kernels (e.g. SDPA's memory-efficient
    attention) silently return wrong gradients when a saved tensor comes back
    with a different stride order, so paging never forces contiguous.
    """
    if _C.is_dense(t):
        return t
    order = sorted(range(t.dim()), key=lambda i: (t.stride(i), t.size(i)), reverse=True)
    strides = [0] * t.dim()
    acc = 1
    for i in reversed(order):
        strides[i] = acc
        acc *= t.size(i)
    return torch.empty_strided(t.size(), strides, dtype=t.dtype, device=t.device).copy_(t)


class _Packed:
    """A saved activation that now lives in pinned host memory."""

    __slots__ = ("session", "block", "host", "d2h", "src", "gpu", "gpu_ref", "ready", "device", "size", "stride",
                 "__weakref__")

    def __init__(self, session, block, host, d2h, device, size, stride, src):
        self.session = session
        self.block = block
        self.host = host
        self.d2h = d2h
        self.src = src  # VRAM source, kept alive until the D2H event completes
        self.gpu = None  # strong ref while prefetched but not yet consumed
        self.gpu_ref = None  # weak ref after hand-off to autograd
        self.ready = None
        self.device = device
        self.size = size  # exact layout to restore
        self.stride = stride

    def fetch(self):
        return _C.h2d_alloc_async(self.host, self.d2h, self.device, self.size, self.stride)

    def release_src(self) -> bool:
        if self.src is not None and self.d2h.query():
            self.src = None
        return self.src is None

    def __del__(self):
        try:
            if self.src is not None and not self.d2h.query():
                self.d2h.synchronize()  # rare: graph dropped while the copy is in flight
            if self.gpu is not None and self.ready is not None:
                self.ready.wait()  # unconsumed prefetch: order its reuse after the copy
        except Exception:
            pass


class _Resident:
    """A saved tensor kept in VRAM, tagged with its block so unpacking it can
    trigger prefetch of the blocks below."""

    __slots__ = ("block", "tensor", "session")

    def __init__(self, session, block, tensor):
        self.session = session
        self.block = block
        self.tensor = tensor


class _Session:
    """Bookkeeping for one forward pass (one autograd graph).

    Handles point at their session strongly; the session points back only
    weakly. The autograd graph therefore owns every pinned buffer and frees it
    deterministically — a strong cycle here would leave whole steps of pinned
    activations waiting for Python's cyclic GC.
    """

    __slots__ = ("blocks", "staged", "dedup", "__weakref__")

    def __init__(self, n: int):
        self.blocks: List[List["weakref.ref[_Packed]"]] = [[] for _ in range(n)]
        self.staged = [False] * n
        self.dedup: Dict[int, tuple] = {}


class ActivationOffloader:
    """Pages saved activations of the selected blocks to pinned host RAM.

    Forward:  every tensor autograd saves inside an offloaded block is copied
              D2H on the copy stream (deduplicated when several ops save the same
              tensor) and the VRAM copy is released.
    Backward: when autograd first touches block *b*, blocks b-1 .. b-depth are
              staged H2D on the high-priority prefetch stream, so with depth=1
              exactly two blocks are resident at a time — the one computing and
              the one landing (a ping-pong window over the caching allocator).

    ``offload`` may be a float fraction, an explicit set of block indices, or
    ``"auto"``, which runs a closed loop:
      1. calibrate — ``calibrate_steps`` training steps with nothing paged,
         measuring per-block saved bytes, per-block forward time (CUDA events)
         and the wall-clock step time;
      2. predict  — size the paged prefix from the hardware profile's link
         model (duplex factor, optimizer traffic reserved);
      3. tune     — measure real step time with that prefix and halve it until
         the slowdown is within ``max_slowdown`` of the calibrated baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks: Sequence[nn.Module],
        offload="auto",
        *,
        min_bytes: int = 1 << 20,
        prefetch_depth: int = 1,
        calibrate_steps: int = 3,
        profile=None,
        max_slowdown: float = 0.15,
        reserved_link_bytes: Tuple[float, float] = (0.0, 0.0),
        tune_window: int = 3,
        max_inflight_bytes: int = 1 << 30,
    ):
        self.model = model
        self.blocks = list(blocks)
        self.n = len(self.blocks)
        self.min_bytes = int(min_bytes)
        self.depth = max(1, int(prefetch_depth))
        self.profile = profile
        self.max_slowdown = float(max_slowdown)
        self.reserved_link_bytes = reserved_link_bytes  # (d2h, h2d) bytes per step used by the optimizer
        self.tune_window = max(1, int(tune_window))
        # Back-pressure: VRAM held by sources still crossing PCIe is capped here.
        # Without it a fast GPU outruns the link and paging saves little memory.
        self.max_inflight_bytes = int(max_inflight_bytes)
        self._inflight_bytes = 0
        self.enabled = True
        self.decision: Dict[str, object] = {}
        self._param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self._session: Optional[_Session] = None
        self._ctx = None
        self._depth_guard = 0
        self._block: Optional[int] = None
        self._last_fwd: Optional[float] = None
        self.stats = dict(packed_tensors=0, packed_bytes=0, dedup_hits=0, resident_tensors=0,
                          prefetch_hits=0, prefetch_misses=0, staged_blocks=0, forwards=0,
                          backpressure_waits=0, backpressure_ms=0.0)
        self._calib_target = 0
        self._calib_steps_ms: List[float] = []
        self._calib_bytes: List[Dict[int, int]] = []
        self._calib_events: List[List[tuple]] = []
        self._tune_samples: List[float] = []
        # Weak refs, in issue order, to handles whose VRAM source may still be
        # in flight to the host. Weak so they never extend a handle's life.
        self._inflight: "collections.deque[tuple]" = collections.deque()  # (weakref, nbytes)
        self.phase = "fixed"

        if offload == "auto":
            self.offload = set()
            self.phase = "calibrate"
            self._calib_target = max(2, int(calibrate_steps))
        elif isinstance(offload, (set, frozenset, list, tuple)):
            self.offload = {int(i) for i in offload if 0 <= int(i) < self.n}
        else:
            frac = 1.0 if offload is True else float(offload or 0.0)
            self.offload = set(range(self._blocks_for_fraction(frac)))

        self._handles = [
            model.register_forward_pre_hook(self._root_pre),
            model.register_forward_hook(self._root_post, always_call=True),
        ]
        for i, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(self._make_block_pre(i)))
            self._handles.append(block.register_forward_hook(self._make_block_post(i), always_call=True))

    # -- configuration -------------------------------------------------------
    def _blocks_for_fraction(self, frac: float) -> int:
        # The last block is consumed first by backward; paging it out buys nothing.
        frac = min(max(frac, 0.0), 1.0)
        return int(frac * max(self.n - 1, 0) + 1e-9)

    def set_fraction(self, frac: float) -> None:
        self.phase = "fixed"
        self.offload = set(range(self._blocks_for_fraction(frac)))

    @property
    def calibrating(self) -> bool:
        return self.phase == "calibrate"

    # -- step clock (wall time between successive training forwards) ---------
    def _on_step_time(self, ms: float) -> None:
        if self.phase == "calibrate":
            self._calib_steps_ms.append(ms)
            if len(self._calib_steps_ms) >= self._calib_target:
                self._finish_calibration()
        elif self.phase == "tune":
            self._tune_samples.append(ms)
            if len(self._tune_samples) > self.tune_window:  # first sample absorbs pinned-cache warm-up
                t = statistics.median(self._tune_samples[1:])
                base = float(self.decision["baseline_step_ms"])
                k = len(self.offload)
                self.decision["history"].append({"blocks": k, "step_ms": t})
                if t > base * (1.0 + self.max_slowdown) and k > 0:
                    self.offload = set(range(k // 2))
                    self._tune_samples = []
                    if not self.offload:
                        self.phase = "fixed"
                else:
                    self.phase = "fixed"
                self.decision["offloaded_blocks"] = float(len(self.offload))

    # -- module hooks --------------------------------------------------------
    def _root_pre(self, module, args):
        self._depth_guard += 1
        if self._depth_guard > 1 or not torch.is_grad_enabled() or not self.enabled:
            return
        now = time.perf_counter()
        if self._last_fwd is not None and self.phase != "fixed":
            self._on_step_time((now - self._last_fwd) * 1e3)
        self._last_fwd = now
        self.stats["forwards"] += 1
        self._session = _Session(self.n)
        if self.calibrating:
            self._calib_bytes.append({})
            self._calib_events.append([])
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()

    def _root_post(self, module, args, output):
        self._depth_guard -= 1
        if self._depth_guard > 0 or self._ctx is None:
            return
        self._ctx.__exit__(None, None, None)
        self._ctx = None
        self._reap()
        if self._session is not None:
            self._session.dedup.clear()  # weakrefs only matter during forward
        self._session = None
        self._block = None

    def _make_block_pre(self, i: int):
        def hook(module, args):
            self._block = i
            self._reap()
            if self.calibrating and self._session is not None:
                self._calib_events[-1].append((i, _C.record_event(-1, True)))

        return hook

    def _make_block_post(self, i: int):
        def hook(module, args, output):
            if self.calibrating and self._session is not None:
                self._calib_events[-1].append((i, _C.record_event(-1, True)))
            self._block = None

        return hook

    def _reap(self, block_until: Optional[int] = None) -> None:
        """Release VRAM sources whose D2H copies have completed (cheap event
        polls). With `block_until`, wait (host-side) on the oldest copies until
        at most that many bytes are still in flight."""
        q = self._inflight
        while q:
            ref, nbytes = q[0]
            packed = ref()
            if packed is not None and not packed.release_src():
                if block_until is None or self._inflight_bytes <= block_until:
                    break
                t0 = time.perf_counter()
                packed.d2h.synchronize()
                self.stats["backpressure_waits"] += 1
                self.stats["backpressure_ms"] += (time.perf_counter() - t0) * 1e3
                packed.release_src()
            q.popleft()
            self._inflight_bytes -= nbytes

    # -- saved-tensor hooks --------------------------------------------------
    def _pack(self, t: torch.Tensor):
        b = self._block
        session = self._session
        if b is None or session is None:
            return t
        if not t.is_cuda or t.layout != torch.strided:
            return t
        nbytes = t.numel() * t.element_size()
        is_param = t.untyped_storage().data_ptr() in self._param_storages

        if self.calibrating:
            if not is_param and nbytes >= self.min_bytes:
                seen = session.dedup.get(id(t))
                if seen is None or seen[0]() is not t:
                    session.dedup[id(t)] = (weakref.ref(t), None)
                    sizes = self._calib_bytes[-1]
                    sizes[b] = sizes.get(b, 0) + nbytes
            return t

        if b not in self.offload or is_param or nbytes < self.min_bytes:
            self.stats["resident_tensors"] += 1
            return _Resident(session, b, t)

        seen = session.dedup.get(id(t))
        if seen is not None and seen[0]() is t:
            self.stats["dedup_hits"] += 1
            return seen[1]

        src = _dense_like(t)
        host = _C.pinned_empty([src.numel()], src.dtype, "activations", False, True)
        d2h = _C.d2h_async(host, src)
        packed = _Packed(session, b, host, d2h, t.device.index, list(src.shape), list(src.stride()), src)
        session.blocks[b].append(weakref.ref(packed))
        self._inflight.append((weakref.ref(packed), nbytes))
        self._inflight_bytes += nbytes
        self._reap(block_until=self.max_inflight_bytes)
        session.dedup[id(t)] = (weakref.ref(t), packed)
        self.stats["packed_tensors"] += 1
        self.stats["packed_bytes"] += nbytes
        return packed

    def _unpack(self, obj):
        if self._inflight:
            self._reap()
        if isinstance(obj, _Resident):
            self._stage_below(obj.session, obj.block)
            return obj.tensor
        if not isinstance(obj, _Packed):
            return obj
        session = obj.session
        self._stage(session, obj.block)
        self._stage_below(session, obj.block)

        obj.release_src()
        if obj.gpu is not None:
            t = obj.gpu
            obj.gpu = None
            obj.gpu_ref = weakref.ref(t)
            self.stats["prefetch_hits"] += 1
        else:
            t = obj.gpu_ref() if obj.gpu_ref is not None else None
            if t is None:  # evicted after an earlier hand-off (e.g. retain_graph)
                t, obj.ready = obj.fetch()
                obj.gpu_ref = weakref.ref(t)
                self.stats["prefetch_misses"] += 1
        obj.ready.wait()  # compute stream waits on the H2D event (host does not)
        return t

    def _stage(self, session: _Session, b: int) -> None:
        if b < 0 or session.staged[b]:
            return
        session.staged[b] = True
        if not session.blocks[b]:
            return
        self.stats["staged_blocks"] += 1
        for ref in session.blocks[b]:
            packed = ref()
            if packed is not None and packed.gpu is None and packed.gpu_ref is None:
                packed.gpu, packed.ready = packed.fetch()

    def _stage_below(self, session: _Session, b: int) -> None:
        for d in range(1, self.depth + 1):
            self._stage(session, b - d)

    # -- calibration ---------------------------------------------------------
    def _finish_calibration(self) -> None:
        bytes_per_block = self._calib_bytes[-1] if self._calib_bytes else {}
        events = self._calib_events[-1] if self._calib_events else []
        times: Dict[int, float] = {}
        opened: Dict[int, object] = {}
        for i, ev in events:
            if i in opened:
                times[i] = times.get(i, 0.0) + opened.pop(i).elapsed_ms(ev)
            else:
                opened[i] = ev
        self._calib_bytes.clear()
        self._calib_events.clear()

        fwd_ms = statistics.median(times.values()) if times else 0.0
        blk_bytes = statistics.median(bytes_per_block.values()) if bytes_per_block else 0
        # Step 1 includes allocator / cuBLAS warm-up; the rest is the baseline.
        base_ms = statistics.median(self._calib_steps_ms[1:]) if len(self._calib_steps_ms) > 1 else 0.0
        reserved = self.profile.link_ms(*self.reserved_link_bytes) if self.profile is not None else 0.0
        if self.profile is not None and blk_bytes > 0 and base_ms > 0:
            frac = self.profile.activation_offload_fraction(blk_bytes, base_ms, self.n, reserved)
        else:
            frac = 1.0 if blk_bytes > 0 else 0.0
        self.offload = set(range(self._blocks_for_fraction(frac)))
        p = self.profile
        self.decision = dict(
            block_fwd_ms=fwd_ms,
            block_bytes=float(blk_bytes),
            baseline_step_ms=base_ms,
            reserved_link_ms=reserved,
            block_link_ms=p.link_ms(blk_bytes, blk_bytes) if p else 0.0,
            d2h_ms=p.transfer_ms(blk_bytes, "d2h") if p else 0.0,
            h2d_ms=p.transfer_ms(blk_bytes, "h2d") if p else 0.0,
            fraction=frac,
            predicted_blocks=float(len(self.offload)),
            offloaded_blocks=float(len(self.offload)),
            total_blocks=float(self.n),
            max_slowdown=self.max_slowdown,
            history=[],
        )
        self.phase = "tune" if self.offload and base_ms > 0 else "fixed"
        self._tune_samples = []

    # -- lifecycle -----------------------------------------------------------
    def remove(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        for h in self._handles:
            h.remove()
        self._handles.clear()
