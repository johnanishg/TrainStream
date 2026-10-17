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
            if id(p) in seen or id(p) in outside:
                continue
            seen.add(id(p))
            ps.append(p)
        out[i] = ps
    return out


class ParamPager:
    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module], optimizer, paged: Iterable[int] = (), *,
                 candidates: Optional[Dict[int, List[nn.Parameter]]] = None,
                 frozen_init: Optional[Dict[int, torch.Tensor]] = None, prefetch_depth: int = 1):
        self.model = model
        self.blocks = list(blocks)
        self.n = len(self.blocks)
        self.opt = optimizer
        self.depth = max(1, int(prefetch_depth))
        self.device = optimizer.device
        if candidates is None:
            candidates = pageable_params(model, self.blocks, range(self.n))
        self._entries: Dict[int, list] = {}
        for i in range(self.n):
            self._entries[i] = [(p, optimizer._slot_of.get(id(p)), p.numel() * p.element_size())
                                for p in candidates.get(i, [])]
        self.block_bytes = [sum(nb for _, _, nb in self._entries[i]) for i in range(self.n)]
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
        self.stats = dict(loads=0, load_bytes=0, releases=0, pages=0, unpages=0)
        frozen_init = frozen_init if frozen_init is not None else {}
        for i in sorted(set(paged)):
            self.page(i, frozen_init)
        self._handles = [model.register_forward_pre_hook(self._root_pre)]
        for i, b in enumerate(self.blocks):
            self._handles.append(b.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(b.register_forward_hook(self._make_post(i), always_call=True))
            self._handles.append(b._register_state_dict_hook(self._make_state_dict_hook(i)))
            self._handles.append(b._register_load_state_dict_pre_hook(self._make_load_pre(i), with_module=True))
            self._handles.append(b.register_load_state_dict_post_hook(self._make_load_post(i)))

    # -- state ------------------------------------------------------------------
    @property
    def bytes_paged(self) -> int:
        return sum(self.block_bytes[i] for i in self.paged)

    @property
    def last(self) -> int:
        return self._sorted[-1] if self._sorted else -1

    def _resort(self) -> None:
        self._sorted = sorted(self.paged)

    def _next_paged(self, i: int) -> List[int]:
        return [j for j in self._sorted if j > i][: self.depth]

    def _prev_paged(self, i: int) -> List[int]:
        return [j for j in reversed(self._sorted) if j < i][: self.depth]

    def _source(self, p, slot) -> torch.Tensor:
        return self.opt.engine.host_param(slot) if slot is not None else self._host_frozen[id(p)]

    # -- moving a block between VRAM and host RAM (between steps) ---------------
    def page(self, i: int, frozen_init: Optional[Dict[int, torch.Tensor]] = None) -> None:
        """Make block i paged: its weights leave VRAM."""
        if i in self.paged or not self._entries[i]:
            return
        entries = self._entries[i]
        slots = [s for _, s, _ in entries if s is not None]
        if slots:
            self.opt.engine.wait_ready(slots)  # an in-flight upload into VRAM must land first
        for p, slot, nbytes in entries:
            if slot is not None:
                self.opt.engine.set_paged(slot, True)
                self.opt._paged_ids.add(id(p))
                continue
            # Frozen weight: pinned host copy while that fits the page-lock
            # budget, else ordinary RAM (the CPU tensor a model was built in is
            # reused as is). Taken one by one, so a model never exists twice.
            src = frozen_init.pop(id(p), None) if frozen_init is not None else None
            if _C.pinned_stats()["live_bytes"] + nbytes <= self.pin_budget:
                host = _C.pinned_empty([p.numel()], p.dtype, "param.frozen", False, False)
                host.copy_(src.reshape(-1) if src is not None else p.detach().reshape(-1))
                self.frozen_pinned_bytes += nbytes
                host._ts_pinned = True
            else:
                host = src.reshape(-1) if src is not None else p.detach().reshape(-1).cpu()
                self.frozen_pageable_bytes += nbytes
                host._ts_pinned = False
            del src
            self._host_frozen[id(p)] = host
        self.paged.add(i)
        self._resort()
        self.allocated[i] = all(p.untyped_storage().nbytes() > 0 for p, _, _ in entries)
        self.fresh[i] = False
        self.release(i)
        self.stats["pages"] += 1

    def unpage(self, i: int) -> None:
        """Make block i resident: load its weights and keep them in VRAM."""
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
            if getattr(host, "_ts_pinned", False):
                self.frozen_pinned_bytes -= nbytes
            else:
                self.frozen_pageable_bytes -= nbytes
            del host
        self.paged.discard(i)
        self._resort()
        self.allocated[i] = True
        self.stats["unpages"] += 1

    # -- streaming ------------------------------------------------------------------
    def load(self, i: int) -> None:
        """Queue the H2D copy of block i's weights (does not block the GPU)."""
        if i not in self.paged or self.fresh[i]:
            return
        entries = self._entries[i]
        slots = [s for _, s, _ in entries if s is not None]
        if slots:
            self.opt.engine.wait_issued(slots)  # host copies hold this step's update
        if not self.allocated[i]:
            for p, _, nbytes in entries:
                p.untyped_storage().resize_(nbytes)
            self.allocated[i] = True
        after = _C.record_event(self.device.index or 0)  # memory reuse ordered after queued compute
        last = None
        for p, slot, nbytes in entries:
            last = _C.h2d_async(p.detach().view(-1), self._source(p, slot), after)
            self.stats["load_bytes"] += nbytes
        for s in slots:
            self.opt.engine.set_ready(s, last)  # the next update waits until this read is done
        self.events[i] = last
        self.fresh[i] = True
        self.stats["loads"] += 1

    def wait(self, i: int) -> None:
        ev = self.events.get(i)
        if ev is not None:
            ev.wait()  # compute stream waits for the copy (the host does not)

    def release(self, i: int) -> None:
        if i not in self.paged or not self.allocated[i]:
            return
        self.wait(i)  # an unconsumed prefetch must land before its memory is reused
        for p, _, _ in self._entries[i]:
            p.untyped_storage().resize_(0)
        self.allocated[i] = False
        self.fresh[i] = False
        self.stats["releases"] += 1

    def materialize(self) -> None:
        """Make every block resident (e.g. before unwrap)."""
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
                self.load(i)  # real storage for the copy_ that follows
                self.wait(i)
        return hook

    def _make_load_post(self, i: int):
        def hook(module, incompatible_keys):
            if i not in self.paged:
                return
            torch.cuda.current_stream(self.device).synchronize()
            for p, slot, _ in self._entries[i]:
                if slot is not None:
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
