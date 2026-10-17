"""Parameter paging: transformer-block weights live in host RAM, not VRAM.

The optimizer already keeps an authoritative host copy of every trainable
weight (the fp32 master, or its rounded bf16/fp16 staging copy). For a paged
block the VRAM storage of its weights is released (``storage.resize_(0)``, the
same mechanism FSDP uses), and the pager streams the host copy back in on the
H2D stream whenever the block is about to run:

  forward   block i: load i (usually prefetched), prefetch i+1; release i
            after it ran (the last block stays, backward starts there)
  backward  a hook on block i's output gradient loads i and prefetches i-1;
            blocks above i are released — their input gradients, the last
            computation that reads their weights, are already queued

Frozen weights (LoRA base layers) get their own pinned host copy. Under
torch.autocast the cast-weight cache is cleared after each paged block, or it
would keep a bf16 copy of every paged weight alive until the region exits. Updates of
paged parameters are never uploaded by the optimizer; the next load reads the
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
    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module], optimizer, paged: Dict[int, List[nn.Parameter]],
                 *, frozen_init: Optional[Dict[int, torch.Tensor]] = None, prefetch_depth: int = 1):
        self.model = model
        self.blocks = list(blocks)
        self.n = len(self.blocks)
        self.opt = optimizer
        self.depth = max(1, int(prefetch_depth))
        self.paged = {i: ps for i, ps in paged.items() if ps}
        self.last = max(self.paged) if self.paged else -1
        frozen_init = frozen_init if frozen_init is not None else {}
        self.device = optimizer.device
        self._entries: Dict[int, list] = {}
        self._host_frozen: Dict[int, torch.Tensor] = {}
        ram = physical_ram_bytes()
        pin_budget = 0.40 * ram if ram else float("inf")
        self.frozen_pinned_bytes = 0
        self.frozen_pageable_bytes = 0
        for i, ps in self.paged.items():
            entries = []
            for p in ps:
                slot = optimizer._slot_of.get(id(p))
                if slot is None:
                    # Frozen weight: pinned host copy while that fits the page-lock
                    # budget, else ordinary RAM (the CPU tensor it was built in is
                    # reused as is). Popped one by one, so the model never exists
                    # twice in host memory.
                    src = frozen_init.pop(id(p), None)
                    nbytes = p.numel() * p.element_size()
                    if _C.pinned_stats()["live_bytes"] + nbytes <= pin_budget:
                        host = _C.pinned_empty([p.numel()], p.dtype, "param.frozen", False, False)
                        host.copy_(src.reshape(-1) if src is not None else p.detach().reshape(-1))
                        self.frozen_pinned_bytes += nbytes
                    else:
                        host = (src.reshape(-1) if src is not None else p.detach().reshape(-1).cpu())
                        self.frozen_pageable_bytes += nbytes
                    del src
                    self._host_frozen[id(p)] = host
                entries.append((p, slot, p.numel() * p.element_size()))
            self._entries[i] = entries
        self.allocated = {i: True for i in self.paged}
        self.fresh = {i: False for i in self.paged}
        self.events: Dict[int, object] = {}
        self.stats = dict(loads=0, load_bytes=0, releases=0)
        self.bytes_paged = sum(nb for es in self._entries.values() for _, _, nb in es)
        for i in self.paged:
            self.release(i)
        self._handles = [model.register_forward_pre_hook(self._root_pre)]
        for i in self.paged:
            b = self.blocks[i]
            self._handles.append(b.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(b.register_forward_hook(self._make_post(i), always_call=True))
            self._handles.append(b._register_state_dict_hook(self._make_state_dict_hook(i)))
            self._handles.append(b._register_load_state_dict_pre_hook(self._make_load_pre(i), with_module=True))
            self._handles.append(b.register_load_state_dict_post_hook(self._make_load_post(i)))

    # -- core operations ---------------------------------------------------------
    def _source(self, p, slot) -> torch.Tensor:
        return self.opt.engine.host_param(slot) if slot is not None else self._host_frozen[id(p)]

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
        """Load every paged block (e.g. before unwrap)."""
        for i in self.paged:
            self.fresh[i] = False
            self.load(i)
            self.wait(i)
        torch.cuda.current_stream(self.device).synchronize()

    # -- hooks ---------------------------------------------------------------------
    def _root_pre(self, module, args):
        if not _in_backward():
            for i in self.paged:  # weights may have been updated since the last load
                self.fresh[i] = False

    def _make_pre(self, i: int):
        def hook(module, args):
            self.load(i)
            self.wait(i)
            for d in range(1, self.depth + 1):
                self.load(i + d)
        return hook

    def _make_post(self, i: int):
        def hook(module, args, output):
            # autocast keeps a low-precision copy of every weight it casts until the
            # autocast region exits, which would quietly hold all paged weights in VRAM
            if torch.is_autocast_enabled("cuda"):
                torch.clear_autocast_cache()
            if _in_backward():
                return  # recomputation (activation checkpointing): backward still needs the weights
            if torch.is_grad_enabled():
                for t in _tensors(output):
                    if t.requires_grad:
                        t.register_hook(lambda g, i=i: self._on_grad(i))
                if i == self.last:
                    return  # first block backward needs
            self.release(i)
        return hook

    def _on_grad(self, i: int):
        self.load(i)
        self.wait(i)
        for d in range(1, self.depth + 1):
            self.load(i - d)
        for j in self.paged:
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
            slots = {id(p): s for p, s, _ in self._entries[i]}
            for name, p in self._named(i):
                key = prefix + name
                if key in state_dict:
                    state_dict[key] = self.host_value(p, slots[id(p)])
        return hook

    def _make_load_pre(self, i: int):
        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.load(i)  # real storage for the copy_ that follows
            self.wait(i)
        return hook

    def _make_load_post(self, i: int):
        def hook(module, incompatible_keys):
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
            for es in self._entries.values():
                for _, slot, _ in es:
                    if slot is not None and self.opt.engine is not None:
                        self.opt.engine.set_paged(slot, False)
        for h in self._handles:
            h.remove()
        self._handles.clear()
