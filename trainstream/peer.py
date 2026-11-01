"""Peer GPUs: frozen block weights pooled in the VRAM of the local ranks.

With data parallelism on one machine every rank streams the same frozen
weights (a LoRA base) from host RAM each pass. When the GPUs reach each
other's memory directly (NVLink, or PCIe peer-to-peer where the driver allows
it), the blocks that every rank streams are split among the ranks instead:
each rank keeps its share in a VRAM pool, and the others copy from that pool
GPU to GPU (cudaMemcpyPeerAsync on their H2D stream) rather than from host
RAM. The group then holds N shares in VRAM instead of N copies of none, and
the host link carries only what no GPU holds.

Set up once, collectively, at a fixed step (every rank reaches it together):

  1. topology: every pair of the ranks' GPUs must allow peer access (the
     same GPU always does); GeForce cards report no P2P and stop here;
  2. CUDA IPC: each rank exports a test pool, the others map it and
     time copies from it; peering needs at least 0.9x the host link speed;
  3. the blocks every rank streams are assigned round-robin to owners, each
     within the VRAM it can spare; owners fill their pool, the others map it.

Only frozen weights take part (they never change, so no cross-process
synchronization is needed after setup); trained weights still come from the
host. Tested with two processes sharing one GPU (Windows 11 and WSL2: IPC
mapping, routing, results bit-identical); copies between two real GPUs need a
machine whose GPUs allow peer access and are untested.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from . import _C

ALIGN = 256
PROBE_BYTES = 64 << 20
MIN_SPEEDUP = 0.9  # peer copies must reach this share of the host link


def _gather(local, obj):
    import torch.distributed as dist
    out = [None] * local.world
    dist.all_gather_object(out, obj, group=local.group)
    return out


def probe(local, device: torch.device, host_gbps: float) -> Tuple[bool, str, dict]:
    """Collective: can the local ranks copy each other's VRAM fast enough?
    Returns (on, reason, info) — identical on every rank."""
    dev = device.index if device.index is not None else torch.cuda.current_device()
    devs = _gather(local, dev)
    info: dict = {"devices": devs}
    if local.world < 2 or not local.single_node:
        return False, "needs two or more ranks on one machine", info
    rows = {(int(r["src"]), int(r["dst"])): r for r in _C.peer_topology(0)}
    mine_ok = all(d == dev or rows.get((d, dev), {}).get("can_access", 0) > 0 for d in devs)
    oks = _gather(local, mine_ok)
    if not all(oks):
        return False, "the driver reports no peer access between these GPUs", info
    # IPC + measured speed, with a test pool per rank
    err = None
    pool = None
    try:
        pool, handle = _C.peer_pool(PROBE_BYTES, dev)
    except RuntimeError as exc:
        err, handle = str(exc), None
    handles = _gather(local, handle)
    errs = _gather(local, err)
    if any(e is not None for e in errs):
        del pool
        return False, f"CUDA IPC unavailable ({next(e for e in errs if e is not None)})", info
    speeds, err = [], None
    try:
        for r, h in enumerate(handles):
            if r == local.rank:
                continue
            ptr = _C.peer_open(h, dev)
            try:
                speeds.append(_C.peer_copy_gbps(ptr, devs[r], PROBE_BYTES, dev))
            finally:
                _C.peer_close(ptr, dev)
    except RuntimeError as exc:
        err = str(exc)
    torch.cuda.synchronize(device)
    results = _gather(local, (min(speeds) if speeds else 0.0, err))
    local.barrier()  # nobody frees its test pool while another still copies from it
    del pool
    errs = [e for _, e in results if e is not None]
    if errs:
        return False, f"cannot map a peer's memory ({errs[0]})", info
    gbps = min(s for s, _ in results)
    info.update(peer_gbps=gbps, host_gbps=host_gbps)
    if host_gbps > 0 and gbps < MIN_SPEEDUP * host_gbps:
        return False, f"peer copies ({gbps:.1f} GB/s) are slower than the host link ({host_gbps:.1f} GB/s)", info
    return True, f"peer copies at {gbps:.1f} GB/s (host link {host_gbps:.1f} GB/s)", info


def _frozen_parts(pager, i: int) -> List[Tuple[int, str, torch.Tensor]]:
    """(param id, part, host tensor) of block i's frozen weights, in a fixed order."""
    out = []
    for p, slot, _ in pager._entries[i]:
        if slot is not None:
            continue
        qw = pager._quant.get(id(p))
        if qw is not None:
            out.append((id(p), "q", qw.host_q))
            out.append((id(p), "s", qw.host_absmax))
        elif id(p) in pager._host_frozen:
            out.append((id(p), "w", pager._host_frozen[id(p)]))
    return out


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _aligned(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def assign(candidates: List[int], block_bytes: Dict[int, int], world: int, budget: int) -> Dict[int, int]:
    """Block -> owner rank: round-robin over the candidates, each owner within
    `budget` bytes. Deterministic, so every rank computes the same map."""
    used = [0] * world
    owners: Dict[int, int] = {}
    r = 0
    for i in candidates:
        nb = block_bytes[i]
        for k in range(world):  # the next owner in turn that still has room
            o = (r + k) % world
            if used[o] + nb <= budget:
                owners[i] = o
                used[o] += nb
                r = o + 1
                break
    return owners


class PeerPool:
    """The pools of one rank and the addresses of every pooled weight."""

    def __init__(self):
        self.pool: Optional[torch.Tensor] = None
        self.opened: List[Tuple[int, int]] = []  # (address, device) mapped from peers
        self.where: Dict[Tuple[int, str], Tuple[int, int, int]] = {}  # (param id, part) -> (address, offset, device)
        self.owners: Dict[int, int] = {}
        self.bytes_owned = 0
        self.local = None
        self.device = None

    @classmethod
    def build(cls, local, pager, device: torch.device, budget: int) -> "PeerPool":
        """Collective. `pager` may be None (this rank streams nothing)."""
        self = cls()
        self.local, self.device = local, device
        dev = device.index if device.index is not None else torch.cuda.current_device()
        remote = sorted(pager.remote) if pager is not None else []
        remotes = _gather(local, remote)
        budgets = _gather(local, int(budget))
        common = sorted(set.intersection(*(set(r) for r in remotes))) if remotes else []
        parts = {i: _frozen_parts(pager, i) for i in common} if pager is not None else {}
        block_bytes = {i: sum(_aligned(_nbytes(t)) for _, _, t in parts.get(i, [])) for i in common}
        candidates = [i for i in common if block_bytes[i] > 0]
        self.owners = assign(candidates, block_bytes, local.world, max(0, min(budgets)))
        devs = _gather(local, dev)
        # owners fill their pools
        mine = [i for i in candidates if self.owners.get(i) == local.rank]
        handle, offsets, off = None, {}, 0
        for i in mine:
            for pid, part, t in parts[i]:
                offsets[(i, pid, part)] = off
                off += _aligned(_nbytes(t))
        if off:
            self.pool, handle = _C.peer_pool(off, dev)
            for (i, pid, part), o in offsets.items():
                t = next(t for q, pp, t in parts[i] if q == pid and pp == part)
                self.pool[o:o + _nbytes(t)].copy_(t.reshape(-1).view(torch.uint8))
            torch.cuda.synchronize(device)
            self.bytes_owned = off
        handles = _gather(local, handle)
        local.barrier()  # every pool is filled before anyone copies from it
        bases = {}
        for r, h in enumerate(handles):
            if h is None:
                continue
            if r == local.rank:
                bases[r] = self.pool.data_ptr()
            else:
                bases[r] = _C.peer_open(h, dev)
                self.opened.append((bases[r], dev))
        # every rank lays the owners' pools out the same way
        for r in range(local.world):
            off = 0
            for i in candidates:
                if self.owners.get(i) != r:
                    continue
                for pid, part, t in parts[i]:
                    self.where[(pid, part)] = (bases[r], off, devs[r])
                    off += _aligned(_nbytes(t))
        return self

    def source(self, pid: int, part: str) -> Optional[Tuple[int, int, int]]:
        return self.where.get((pid, part))

    @property
    def pooled_blocks(self) -> List[int]:
        return sorted(self.owners)

    def close(self) -> None:
        """Collective (like the setup): unmap the peers' pools, then free ours."""
        torch.cuda.synchronize(self.device)
        for ptr, dev in self.opened:
            _C.peer_close(ptr, dev)
        self.opened.clear()
        self.where.clear()
        if self.local is not None:
            self.local.barrier()  # nobody still maps our pool
        self.pool = None
