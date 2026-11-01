"""Host memory shared by the data-parallel processes of one machine.

With ``distributed`` training on one machine (every rank local), TrainStream
keeps
  * one host copy of the frozen block weights (bf16 or quantized) for all
    ranks instead of one per rank: local rank 0 writes it, the others map it;
  * the gradient all-reduce in shared memory instead of gloo over loopback.

Segments are named after a run id that rank 0 broadcasts, so concurrent runs
never collide; each is page-locked (cudaHostRegister) while the pin budget
allows. ``TRAINSTREAM_SHM=0`` turns both off (every rank keeps its own copies
and gloo reduces), and they are off automatically when ranks span machines.
"""

from __future__ import annotations

import os
import uuid
from typing import Callable, Dict, List, Optional, Sequence

import torch

from . import _C


def enabled() -> bool:
    return os.environ.get("TRAINSTREAM_SHM", "1") not in ("0", "false", "False", "no")


class LocalGroup:
    """The ranks of a process group that share this machine's RAM."""

    def __init__(self, group, world: int, rank: int):
        import torch.distributed as dist
        self.group = group
        self.world = world
        self.rank = rank
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
        self.single_node = local_world == world
        box = [uuid.uuid4().hex[:16] if rank == 0 else None]
        dist.broadcast_object_list(box, src=0, group=group)
        self.run_id = f"ts{box[0]}"
        self._segments: Dict[str, torch.Tensor] = {}

    def barrier(self) -> None:
        import torch.distributed as dist
        dist.barrier(group=self.group)

    def segments(self, specs: Sequence[tuple], fill: Callable[[str, torch.Tensor], None],
                 pin_budget: float = float("inf")) -> Dict[str, torch.Tensor]:
        """Collective: one shared segment per (key, nbytes). Rank 0 creates them
        and calls fill(key, segment); after a barrier every other rank opens
        them. Page-locked while the running total stays within pin_budget."""
        out: Dict[str, torch.Tensor] = {}
        pinned = 0
        create = self.rank == 0
        if create:
            for key, nbytes in specs:
                pin = pinned + nbytes <= pin_budget
                t = _C.shared_host(f"{self.run_id}_{key}", int(nbytes), True, pin)
                pinned += nbytes if pin else 0
                fill(key, t)
                out[key] = t
        self.barrier()
        if not create:
            for key, nbytes in specs:
                pin = pinned + nbytes <= pin_budget
                out[key] = _C.shared_host(f"{self.run_id}_{key}", int(nbytes), False, pin)
                pinned += nbytes if pin else 0
        self.barrier()  # every rank holds its mapping: rank 0 may drop its own reference later
        self._segments.update(out)
        return out


class ShmAllReduce:
    """Sum float32 host buffers across the local ranks through shared memory
    (fixed rank order: bit-identical on every rank)."""

    def __init__(self, local: LocalGroup, capacity: int):
        self.local = local
        self.capacity = int(max(1, capacity))
        nbytes = _C.shm_reducer_bytes(local.world, self.capacity)
        self.segment = local.segments([("reduce", nbytes)], lambda k, t: None, pin_budget=0)["reduce"]
        self.seq = 0

    def __call__(self, buf: torch.Tensor) -> None:
        self.seq += 1
        _C.shm_allreduce(buf, self.segment, self.local.rank, self.local.world, self.capacity, self.seq)


def view_as(segment: torch.Tensor, dtype: torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    """A typed tensor over (the start of) a byte segment."""
    n = 1
    for s in shape:
        n *= int(s)
    nbytes = n * torch.empty(0, dtype=dtype).element_size()
    return segment[:nbytes].view(dtype).view(*shape)
