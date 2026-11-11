"""Parameter paging: transformer-block weights live in host RAM, not VRAM.

The optimizer already keeps an authoritative host copy of every trainable
weight (the fp32 master, or its rounded bf16/fp16 staging copy). For a paged
block the VRAM storage of its weights is released (``storage.resize_(0)``, the
same mechanism FSDP uses), and the pager streams the host copy back in on the
H2D stream whenever the block is about to run:

  forward   block i: load i if it is paged (usually prefetched), prefetch the
            next paged block(s); release i after it ran (the last paged block
            stays, backward starts there)
  backward  a hook on block i's output gradient loads i (if paged) and
            prefetches the previous paged block(s); paged blocks above i are
            released — their input gradients, the last computation that reads
            their weights, are already queued

Which blocks are paged can change between steps (``page`` / ``unpage``): the
memory planner keeps as many blocks resident as VRAM allows. Hooks sit on every
block, so a resident block prefetches the next paged one and transfers overlap
with compute even when paged and resident blocks are interleaved.

Quantized frozen weights (trainstream.quant: int8 / NF4) are always expanded
on demand: before the block runs, a kernel on the H2D stream writes the
weight's dtype into its storage from the packed bytes, which live either in
VRAM ("near": no PCIe traffic, just the expansion) or in host RAM ("far":
copied over first, a quarter or half of the bf16 bytes). The planner moves
blocks between near and far exactly as it moves bf16 blocks in and out.

Frozen weights (LoRA base layers) of a paged block get a host copy of their
own: pinned while that fits the page-lock budget, else ordinary RAM. It is
dropped again when the block becomes resident. Under torch.autocast the
cast-weight cache is cleared after each block, or it would keep a bf16 copy of
every weight alive until the autocast region exits. Updates of paged
parameters are never uploaded by the optimizer; the next load reads the
updated host copy after the optimizer has finished with it, and the next
update waits for that DMA to finish reading.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch
from torch import nn

from . import _C
from .optimizer import physical_ram_bytes
from .disk import is_on_disk
from .quant import make_quantized, quantized_of


def _in_backward() -> bool:
    return torch._C._current_graph_task_id() != -1


def _tensors(out) -> Iterable[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        yield out
    elif isinstance(out, (list, tuple)):
        for o in out:
            if isinstance(o, torch.Tensor):
                yield o
    elif isinstance(out, dict):
        for o in out.values():
            if isinstance(o, torch.Tensor):
                yield o


def pageable_params(model: nn.Module, blocks: Sequence[nn.Module], indices: Iterable[int]) -> Dict[int, List[nn.Parameter]]:
    """Parameters of the given blocks that no module outside the block stack
    owns too (a weight shared with, say, the output layer stays resident)."""
    inside: Set[int] = set()
    for b in blocks:
        inside.update(id(m) for m in b.modules())
    outside = {id(p) for m in model.modules() if id(m) not in inside for p in m.parameters(recurse=False)}
    out: Dict[int, List[nn.Parameter]] = {}
    for i in indices:
        seen: Set[int] = set()
        ps = []
        for p in blocks[i].parameters():
            if id(p) in seen or id(p) in outside or getattr(p, "_ts_expert", False):
                continue  # (stacked experts with moe_paging are copied in by trainstream.experts)
            seen.add(id(p))
            ps.append(p)
        out[i] = ps
    return out


class ParamPager:
    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module], optimizer, paged: Iterable[int] = (), *,
                 candidates: Optional[Dict[int, List[nn.Parameter]]] = None,
                 frozen_init: Optional[Dict[int, torch.Tensor]] = None, prefetch_depth: int = 1, disk=None):
        self.model = model
        self.disk = disk  # trainstream.disk.DiskStore: where frozen weights over the pin budget go, if set
        self.frozen_disk_bytes = 0
        self.blocks = list(blocks)
        self.n = len(self.blocks)
        self.opt = optimizer
        self.depth = max(1, int(prefetch_depth))
        self.device = optimizer.device
        if candidates is None:
            candidates = pageable_params(model, self.blocks, range(self.n))
        self._entries: Dict[int, list] = {}
        self._quant: Dict[int, object] = {}  # id(param) -> QuantizedWeight
        self.quant_blocks: Set[int] = set()
        for i in range(self.n):
            self._entries[i] = [(p, optimizer._slot_of.get(id(p)), p.numel() * p.element_size())
                                for p in candidates.get(i, [])]
            for p, _, _ in self._entries[i]:
                qw = quantized_of(p)
                if qw is not None:
                    self._quant[id(p)] = qw
                    self.quant_blocks.add(i)
        self.block_bytes = [sum(nb for _, _, nb in self._entries[i]) for i in range(self.n)]
        # what a block costs in VRAM to keep "near": its packed bytes when quantized
        self.near_bytes = [sum(self._quant[id(p)].nbytes if id(p) in self._quant else nb
                               for p, _, nb in self._entries[i]) for i in range(self.n)]
        self.qnear: Set[int] = set()
        self._keep: Dict[int, list] = {}
        self._native_expanded: Dict[int, bool] = {}
        self._loading: Dict[int, Set[int]] = {}  # per block, the weights a load_state_dict in progress provides
        self._host_frozen: Dict[int, torch.Tensor] = {}
        ram = physical_ram_bytes()
        self.pin_budget = 0.40 * ram if ram else float("inf")
        self.frozen_pinned_bytes = 0
        self.frozen_pageable_bytes = 0
        self.paged: Set[int] = set()
        self._sorted: List[int] = []
        self.allocated = {i: True for i in range(self.n)}
        self.fresh = {i: False for i in range(self.n)}
        self.events: Dict[int, object] = {}
        self.stats = dict(loads=0, load_bytes=0, bounce_bytes=0, peer_bytes=0, releases=0, pages=0, unpages=0)
        self.peer = None  # trainstream.peer.PeerPool: frozen weights other GPUs hold for this one
        frozen_init = frozen_init if frozen_init is not None else {}
        self._shared_frozen: Dict[int, torch.Tensor] = {}
        self.frozen_shared_bytes = 0
        local = (optimizer._dp or {}).get("local") if getattr(optimizer, "_dp", None) else None
        if local is not None:
            self._share_frozen(local, frozen_init)
        for i in sorted(set(paged) | self.quant_blocks):
            self.page(i, frozen_init)
        for i in sorted(self.quant_blocks - set(paged)):
            self.unpage(i)  # quantized and near: packed bytes in VRAM
        self._handles = [model.register_forward_pre_hook(self._root_pre)]
        for i, b in enumerate(self.blocks):
            self._handles.append(b.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(b.register_forward_hook(self._make_post(i), always_call=True))
            self._handles.append(b._register_state_dict_hook(self._make_state_dict_hook(i)))
            self._handles.append(b._register_load_state_dict_pre_hook(self._make_load_pre(i), with_module=True))
            self._handles.append(b.register_load_state_dict_post_hook(self._make_load_post(i)))

    # -- state ------------------------------------------------------------------
    @property
    def remote(self) -> Set[int]:
        """Blocks whose weights come over PCIe when they run."""
        return (self.paged - self.quant_blocks) | (self.quant_blocks - self.qnear)

    @property
    def bytes_paged(self) -> int:
        """Bytes streamed from host RAM per load of every remote block."""
        return sum(self.near_bytes[i] for i in self.remote)

    @property
    def quantized_bytes(self) -> int:
        return sum(qw.nbytes for qw in self._quant.values())

    @property
    def last(self) -> int:
        return self._sorted[-1] if self._sorted else -1

    def _resort(self) -> None:
        self._sorted = sorted(self.paged)

    def _next_paged(self, i: int) -> List[int]:
        return [j for j in self._sorted if j > i][: self.depth]

    def _prev_paged(self, i: int) -> List[int]:
        return [j for j in reversed(self._sorted) if j < i][: self.depth]

    def _share_frozen(self, local, frozen_init) -> None:
        """Data parallel on one machine: every rank maps the same host copy of
        the frozen block weights (and of the packed quantized ones), written
        once by local rank 0. Collective: the same order on every rank."""
        from .shared import view_as
        specs, where = [], {}
        for i in range(self.n):
            for k, (p, slot, nbytes) in enumerate(self._entries[i]):
                if slot is not None:
                    continue
                qw = self._quant.get(id(p))
                if qw is not None:
                    for part, t in (("q", qw.host_q), ("s", qw.host_absmax)):
                        key = f"q{i}_{k}_{part}"
                        specs.append((key, t.numel() * t.element_size()))
                        where[key] = (qw, part, t.dtype, tuple(t.shape))
                else:
                    key = f"f{i}_{k}"
                    specs.append((key, nbytes))
                    where[key] = (p, None, p.dtype, (p.numel(),))

        def fill(key, seg):
            obj, part, dt, shape = where[key]
            dst = view_as(seg, dt, shape)
            if part is not None:
                dst.copy_(obj.host_q if part == "q" else obj.host_absmax)
            else:
                src = frozen_init.get(id(obj))
                dst.copy_((src if src is not None else obj.detach()).reshape(-1))

        segs = local.segments(specs, fill, pin_budget=self.pin_budget - _C.pinned_stats()["live_bytes"])
        for key, seg in segs.items():
            obj, part, dt, shape = where[key]
            view = view_as(seg, dt, shape)
            view._ts_pinned = True
            if part == "q":
                obj.host_q = view
            elif part == "s":
                obj.host_absmax = view
            else:
                self._shared_frozen[id(obj)] = view
            self.frozen_shared_bytes += view.numel() * view.element_size()

    def _resident_trainable(self, i: int, slot) -> bool:
        """Trained parameters of a quantized block (LoRA adapters on a
        quantized base) are small: they stay in VRAM, updated in place."""
        return slot is not None and i in self.quant_blocks

    def _native(self, p) -> bool:
        qw = self._quant.get(id(p))
        return qw is not None and getattr(qw, "native", False)

    @property
    def _native_blocks(self) -> Set[int]:
        return {i for i in self.quant_blocks if any(self._native(p) for p, _, _ in self._entries[i])}

    def _source(self, p, slot) -> torch.Tensor:
        return self.opt.engine.host_param(slot) if slot is not None else self._host_frozen[id(p)]

    # -- moving a block between VRAM and host RAM (between steps) ---------------
    def page(self, i: int, frozen_init: Optional[Dict[int, torch.Tensor]] = None) -> None:
        """Make block i paged: its weights leave VRAM. For a quantized block that
        is already paged: move its packed bytes out of VRAM ("far")."""
        if i in self.quant_blocks and i in self.paged:
            if i in self.qnear:
                self.wait(i)
                torch.cuda.current_stream(self.device).synchronize()  # nothing may still read them
                for p, _, _ in self._entries[i]:
                    qw = self._quant.get(id(p))
                    if qw is not None:
                        qw.drop_device()
                self.qnear.discard(i)
                self.stats["pages"] += 1
            return
        if i in self.paged or not self._entries[i]:
            return
        entries = self._entries[i]
        slots = [s for _, s, _ in entries if s is not None and not self._resident_trainable(i, s)]
        if slots:
            self.opt.engine.wait_ready(slots)  # an in-flight upload into VRAM must land first
        for p, slot, nbytes in entries:
            if slot is not None:
                if self._resident_trainable(i, slot):
                    continue  # an adapter on a quantized base: small, stays in VRAM
                self.opt.engine.set_paged(slot, True)
                self.opt._paged_ids.add(id(p))
                continue
            if id(p) in self._quant:
                continue  # its packed host copy is the source
            # Frozen weight: pinned host copy while that fits the page-lock
            # budget, else ordinary RAM (the CPU tensor a model was built in is
            # reused as is). Taken one by one, so a model never exists twice.
            src = frozen_init.pop(id(p), None) if frozen_init is not None else None
            if id(p) in self._shared_frozen:  # one copy for every rank on this machine
                del src
                self._host_frozen[id(p)] = self._shared_frozen[id(p)]
                continue
            if _C.pinned_stats()["live_bytes"] + nbytes <= self.pin_budget:
                host = _C.pinned_empty([p.numel()], p.dtype, "param.frozen", False, False)
                host.copy_(src.reshape(-1) if src is not None else p.detach().reshape(-1))
                self.frozen_pinned_bytes += nbytes
                host._ts_pinned = True
            else:
                base = src if src is not None else p.detach()
                if self.disk is not None and not is_on_disk(base):
                    host = self.disk.store(base)  # a file, mapped: RAM only while the page cache keeps it
                else:
                    host = base.reshape(-1) if base.device.type == "cpu" else base.reshape(-1).cpu()
                if is_on_disk(host):
                    self.frozen_disk_bytes += nbytes
                else:
                    self.frozen_pageable_bytes += nbytes
                host._ts_pinned = False
                del base
            del src
            self._host_frozen[id(p)] = host
        self.paged.add(i)
        self._resort()
        self.allocated[i] = any(p.untyped_storage().nbytes() > 0 for p, _, _ in entries)
        self.fresh[i] = False
        self.release(i)
        self.stats["pages"] += 1

    def unpage(self, i: int) -> None:
        """Make block i resident: load its weights and keep them in VRAM. For a
        quantized block: keep its packed bytes in VRAM ("near")."""
        if i in self.quant_blocks:
            if i not in self.qnear:
                for p, _, _ in self._entries[i]:
                    qw = self._quant.get(id(p))
                    if qw is not None:
                        qw.to_device(self.device)
                self.qnear.add(i)
                self.stats["unpages"] += 1
            return
        if i not in self.paged:
            return
        self.fresh[i] = False
        self.load(i)
        self.wait(i)
        torch.cuda.current_stream(self.device).synchronize()  # host copies may go only after the copy
        for p, slot, nbytes in self._entries[i]:
            if slot is not None:
                self.opt.engine.set_paged(slot, False)  # updates go straight to VRAM again
                self.opt._paged_ids.discard(id(p))
                continue
            host = self._host_frozen.pop(id(p))
            if id(p) in self._shared_frozen:
                continue  # the shared copy stays for the other ranks
            if getattr(host, "_ts_pinned", False):
                self.frozen_pinned_bytes -= nbytes
            elif is_on_disk(host):
                self.frozen_disk_bytes -= nbytes
            else:
                self.frozen_pageable_bytes -= nbytes
            del host
        self.paged.discard(i)
        self._resort()
        self.allocated[i] = True
        self.stats["unpages"] += 1

    # -- streaming ------------------------------------------------------------------
    def load(self, i: int) -> None:
        """Queue the H2D copy of block i's weights (does not block the GPU).

        Tensor-core (block-scaled) weights stay packed in forward. In backward
        they are expanded here, on the H2D stream and prefetched like nf4, so
        the input-gradient matmul finds a ready bf16 copy instead of expanding
        on the compute stream."""
        if i not in self.paged:
            return
        backward = _in_backward()
        expand_native = backward and i in self._native_blocks and not self._native_expanded.get(i)
        if self.fresh[i] and not expand_native:
            return
        only_native = self.fresh[i]  # already loaded in forward: add the expansions only
        entries = self._entries[i]
        slots = [] if only_native else [s for _, s, _ in entries
                                         if s is not None and not self._resident_trainable(i, s)]
        if slots:
            self.opt.engine.wait_issued(slots)  # host copies hold this step's update
        if not self.allocated[i] or expand_native:
            for p, slot, nbytes in entries:
                if self._resident_trainable(i, slot):
                    continue
                native = self._native(p)
                if (not native or expand_native) and p.untyped_storage().nbytes() == 0:
                    p.untyped_storage().resize_(nbytes)
            self.allocated[i] = True
        after = _C.record_event(self.device.index or 0)  # memory reuse ordered after queued compute
        last = self.events.get(i) if only_native else None
        keep = list(self._keep.get(i, [])) if only_native else []
        dev = self.device.index or 0
        bounce_dst, bounce_src = [], []
        for p, slot, nbytes in entries:
            qw = self._quant.get(id(p))
            native = qw is not None and getattr(qw, "native", False)
            if (only_native and not native) or self._resident_trainable(i, slot):
                continue
            if qw is not None:
                near = qw.dev_q is not None
                via_peer = not near and self.peer is not None and self.peer.source(id(p), "q") is not None
                if native and not expand_native:
                    if via_peer:  # the packed bytes from the GPU that holds them
                        tq, ts, last = self._peer_packed(p, qw, after)
                        qw._staged = (tq, ts)
                    elif not near:  # stage the packed bytes; the matmul reads them as they are
                        tq, _ = _C.h2d_alloc_async(qw.host_q, after, dev)
                        ts, last = _C.h2d_alloc_async(qw.host_absmax, after, dev)
                        qw._staged = (tq, ts)
                        self.stats["load_bytes"] += qw.nbytes
                    continue
                if via_peer:  # on this GPU (H2D stream, in order) before the expansion reads it
                    q, s, last = self._peer_packed(p, qw, after)
                    keep.extend((q, s))
                else:
                    q = qw.dev_q if near else qw.host_q
                    s = qw.dev_absmax if near else qw.host_absmax
                if qw.blockscaled:
                    last, held = _C.dequantize_blockscaled_async(p.detach().view(qw.shape[0], -1), q, s, qw.kind,
                                                                 qw.global_decode, after)
                else:
                    last, held = _C.dequantize_async(p.detach().view(-1), q, s, qw.kind, qw.block_size, after)
                keep.extend(held)
                if not near and not via_peer:
                    self.stats["load_bytes"] += qw.nbytes
                continue
            peer_src = self.peer.source(id(p), "w") if self.peer is not None and slot is None else None
            if peer_src is not None:
                base, off, src_dev = peer_src
                last = _C.peer_copy_async(p.detach().view(-1), base, off, src_dev, after)
                self.stats["peer_bytes"] += nbytes
                continue
            src = self._source(p, slot)
            if getattr(src, "_ts_pinned", True):
                last = _C.h2d_async(p.detach().view(-1), src, after)
            else:  # over the page-lock budget: stage through pinned slots instead of the driver's
                bounce_dst.append(p.detach().view(-1))
                bounce_src.append(src)
            self.stats["load_bytes"] += nbytes
        if bounce_dst:
            last = _C.h2d_bounce_async(bounce_dst, bounce_src, after, self.depth + 2)
            self.stats["bounce_bytes"] += sum(t.numel() * t.element_size() for t in bounce_src)
        self._keep[i] = keep  # staging copies of far packed bytes, alive until the event
        for s in slots:
            self.opt.engine.set_ready(s, last)  # the next update waits until this read is done
        self.events[i] = last
        self.fresh[i] = True
        if expand_native:
            self._native_expanded[i] = True
        self.stats["loads"] += 1

    def _peer_packed(self, p, qw, after):
        """Copy a pooled weight's packed bytes and scales from the GPU that owns them."""
        out, last = [], None
        for part, host in (("q", qw.host_q), ("s", qw.host_absmax)):
            base, off, src_dev = self.peer.source(id(p), part)
            t = torch.empty(host.shape, dtype=host.dtype, device=self.device)
            last = _C.peer_copy_async(t, base, off, src_dev, after)
            out.append(t)
        self.stats["peer_bytes"] += qw.nbytes
        return out[0], out[1], last

    def wait(self, i: int) -> None:
        ev = self.events.get(i)
        if ev is not None:
            ev.wait()  # compute stream waits for the copy (the host does not)

    def release(self, i: int) -> None:
        if i not in self.paged or not self.allocated[i]:
            return
        self.wait(i)  # an unconsumed prefetch must land before its memory is reused
        for p, slot, _ in self._entries[i]:
            if self._resident_trainable(i, slot):
                continue
            p.untyped_storage().resize_(0)
            qw = self._quant.get(id(p))
            if qw is not None:
                qw._staged = None
        self._native_expanded[i] = False
        self._keep.pop(i, None)
        self.allocated[i] = False
        self.fresh[i] = False
        self.stats["releases"] += 1

    def materialize(self) -> None:
        """Make every block resident (e.g. before unwrap). Quantized weights are
        expanded for good: the model then holds their dequantized values."""
        for i in sorted(self.quant_blocks):
            for p, _, _ in self._entries[i]:
                qw = self._quant.get(id(p))
                if qw is not None:
                    qw.native = False  # expand this time (unwrap restores the plain Linear forward)
                    qw._staged = None
            self.allocated[i] = False
            self.fresh[i] = False
            self.load(i)
            self.wait(i)
            torch.cuda.current_stream(self.device).synchronize()
            for p, _, _ in self._entries[i]:
                if id(p) in self._quant:
                    del self._quant[id(p)]
                    del p._ts_qweight
            self._keep.pop(i, None)
            self.quant_blocks.discard(i)
            self.qnear.discard(i)
            self.paged.discard(i)
            self.allocated[i] = True
        self._resort()
        for i in list(self._sorted):
            self.unpage(i)

    # -- hooks ---------------------------------------------------------------------
    def _root_pre(self, module, args):
        if not _in_backward():
            for i in self._sorted:  # weights may have been updated since the last load
                self.fresh[i] = False

    def _make_pre(self, i: int):
        def hook(module, args):
            if not self._sorted:
                return
            if i in self.paged:
                self.load(i)
                self.wait(i)
            for j in self._next_paged(i):
                self.load(j)
        return hook

    def _make_post(self, i: int):
        def hook(module, args, output):
            if not self._sorted:
                return
            # autocast keeps a low-precision copy of every weight it casts until the
            # autocast region exits, which would quietly hold paged weights in VRAM
            if torch.is_autocast_enabled("cuda"):
                torch.clear_autocast_cache()
            if _in_backward():
                return  # recomputation (activation checkpointing): backward still needs the weights
            if torch.is_grad_enabled():
                for t in _tensors(output):
                    if t.requires_grad:
                        t.register_hook(lambda g, i=i: self._on_grad(i))
                if i == self.last:
                    return  # backward starts with this one
            self.release(i)
        return hook

    def _on_grad(self, i: int):
        if i in self.paged:
            self.load(i)
            self.wait(i)
        for j in self._prev_paged(i):
            self.load(j)
        for j in self._sorted:
            if j > i:
                self.release(j)

    # -- (de)serialisation -----------------------------------------------------------
    def _named(self, i: int):
        ids = {id(p) for p, _, _ in self._entries[i]}
        return [(n, p) for n, p in self.blocks[i].named_parameters() if id(p) in ids]

    def host_value(self, p, slot) -> torch.Tensor:
        qw = self._quant.get(id(p))
        if qw is not None:  # the values training actually uses
            dev = self.device if torch.cuda.is_available() else None
            return qw.dequantize(dev).cpu()
        if slot is not None:
            self.opt.engine.wait_issued([slot])
        return self._source(p, slot).view(p.shape).clone()

    def _make_state_dict_hook(self, i: int):
        def hook(module, state_dict, prefix, local_metadata):
            if i not in self.paged:
                return
            slots = {id(p): s for p, s, _ in self._entries[i]}
            for name, p in self._named(i):
                key = prefix + name
                if key in state_dict:
                    state_dict[key] = self.host_value(p, slots[id(p)])
        return hook

    def _make_load_pre(self, i: int):
        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if i in self.paged:
                # a partial load (strict=False, e.g. adapters only) must leave the rest as it is:
                # a quantized weight's expansion is not exactly its value (and is empty for one
                # that runs on tensor cores), so it must not be re-quantized from it
                self._loading[i] = {id(p) for name, p in self._named(i) if prefix + name in state_dict}
                self.load(i)  # real storage for the copy_ that follows
                self.wait(i)
                for p, _, nbytes in self._entries[i]:
                    if self._native(p) and p.untyped_storage().nbytes() == 0:
                        p.untyped_storage().resize_(nbytes)  # load_state_dict copies into it
        return hook

    def _make_load_post(self, i: int):
        def hook(module, incompatible_keys):
            if i not in self.paged:
                return
            loaded = self._loading.pop(i, None)
            torch.cuda.current_stream(self.device).synchronize()
            for p, slot, _ in self._entries[i]:
                if loaded is not None and id(p) not in loaded:
                    continue  # not in this state dict: unchanged (release() drops the expansion)
                qw = self._quant.get(id(p))
                if qw is not None:  # re-quantize the loaded values
                    new = make_quantized(p.detach(), qw.kind, qw.block_size, pin=qw.pinned,
                                         double_quant=qw.host_absmax.dtype == torch.uint8)
                    if qw.dev_q is not None:
                        new.to_device(self.device)
                    new.native = getattr(qw, "native", False)
                    self._quant[id(p)] = new
                    p._ts_qweight = new
                    if new.native:
                        p.untyped_storage().resize_(0)
                elif slot is not None:
                    self.opt.engine.sync_host_from_param(slot)
                else:
                    self._host_frozen[id(p)].copy_(p.detach().reshape(-1))
            self.release(i)
        return hook

    def remove(self, materialize: bool = True) -> None:
        if materialize:
            self.materialize()
        for h in self._handles:
            h.remove()
        self._handles.clear()
