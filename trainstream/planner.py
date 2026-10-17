"""Memory planner: decides what lives where from the machine actually present.

Inputs are measured, never assumed:
  * VRAM this process may use: device free memory (other processes excluded),
    capped by torch.cuda.set_per_process_memory_fraction
  * RAM: physical RAM available now, the page-lock budget (40% of physical),
    both shared out between the processes of this machine (LOCAL_WORLD_SIZE)
  * the model: bytes of every transformer block, of everything outside the
    blocks, of trained and of frozen weights

Static plan (in wrap()):
  1. Keep as many blocks resident as fit next to the fixed footprint, a
     transfer window and a reserve for activations; page the rest, spread
     evenly through the stack so each paged block's copy overlaps the compute
     of the resident blocks before it.
  2. Optimizer moments in bf16 when fp32 moments would not fit in RAM.
  3. Refuse up front (MemoryError, with numbers and options) a configuration
     whose host state cannot fit, instead of swapping or dying mid-run.

Measured refinement (first steps of training): the real peak VRAM of each
step decides whether blocks can move into VRAM (headroom of a block or more)
or must move out (peak within 5% of the budget). When no block is paged any
more and the measured activations fit, activation paging is switched off.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set

import torch

GiB = float(1 << 30)


def available_ram_bytes() -> int:
    """Physical RAM available right now (OS view), 0 if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
            st = _MS()
            st.len = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.avail)
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError):
        pass
    return 0


def local_processes() -> int:
    """Training processes sharing this machine's RAM (and, typically, GPU memory)."""
    for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return 1


@dataclass
class Budget:
    vram_total: int          # what this process may use in total (cap-aware)
    vram_usable: int         # vram_total minus what other processes hold
    ram_total: int
    ram_available: int       # for this process, now
    pin_budget: int
    procs: int = 1

    def describe(self) -> str:
        return (f"VRAM usable {self.vram_usable / GiB:.1f} of {self.vram_total / GiB:.1f} GiB; "
                f"RAM available {self.ram_available / GiB:.1f} of {self.ram_total / GiB:.1f} GiB"
                + (f" (shared by {self.procs} processes)" if self.procs > 1 else ""))


def measure_budget(device) -> Budget:
    from .optimizer import physical_ram_bytes
    from .wrapper import vram_budget
    free, total = vram_budget(device)
    others = max(0, total - free - torch.cuda.memory_reserved(device))
    procs = local_processes()
    ram_total = physical_ram_bytes()
    return Budget(vram_total=total, vram_usable=max(0, total - others), ram_total=ram_total,
                  ram_available=available_ram_bytes() // procs, pin_budget=int(0.40 * ram_total) // procs,
                  procs=procs)


def spread(m: int, items: List[int]) -> List[int]:
    """m of `items`, evenly spaced (the first and last chosen last)."""
    n = len(items)
    if m <= 0:
        return []
    if m >= n:
        return list(items)
    return sorted({items[min(n - 1, int((j + 0.5) * n / m))] for j in range(m)})


@dataclass
class Plan:
    paged: Set[int]
    moment_dtype: str
    reason: List[str] = field(default_factory=list)
    host_bytes: int = 0
    vram_estimate: int = 0


class MemoryPlanner:
    """Static plan + closed-loop refinement of the paged block set."""

    def __init__(self, budget: Budget, block_bytes: List[int], *, fixed_vram: int, inflight_bytes: int,
                 margin: float = 0.10, max_refine_steps: int = 8, window_block_bytes: Optional[int] = None):
        self.budget = budget
        self.block_bytes = list(block_bytes)
        self.n = len(block_bytes)
        self.fixed_vram = int(fixed_vram)
        self.inflight = int(inflight_bytes)
        self.margin = float(margin)
        self.target = int(budget.vram_usable * (1.0 - margin))
        self.max_block = max(block_bytes) if block_bytes else 0
        # bytes of one block while it runs (its expanded form when quantized)
        self.window_block = int(window_block_bytes) if window_block_bytes else self.max_block
        self.max_refine_steps = max_refine_steps
        self.steps = 0
        self.stable = 0
        self.done = self.n == 0
        self.history: List[dict] = []

    # -- static ------------------------------------------------------------------
    def initial_paged(self) -> Set[int]:
        if self.n == 0:
            return set()
        weights = sum(self.block_bytes)
        # Activations are unknown until a step has run. While anything is paged,
        # activations are paged too, so what stays in VRAM of them is about the
        # in-flight window; reserve generously and let refinement give it back.
        act_reserve = max(2 * self.inflight, int(0.15 * self.target))
        if self.fixed_vram + weights + act_reserve <= self.target:
            return set()
        window = 3 * self.window_block  # computing + prefetched + lagging release
        room = self.target - self.fixed_vram - window - act_reserve
        keep, acc = 0, 0
        for b in sorted(self.block_bytes):  # count how many blocks fit (sizes are usually uniform)
            if acc + b > room:
                break
            acc += b
            keep += 1
        keep = max(0, min(self.n - 1, keep))
        return set(spread(self.n - keep, list(range(self.n))))

    def fits_at_all(self) -> bool:
        """Could the run fit at all with every block paged? Only a clear "no"
        refuses a run: the fixed parts plus two blocks in flight."""
        return self.fixed_vram + 2 * self.window_block <= self.budget.vram_usable

    # -- refinement ------------------------------------------------------------------
    def refine(self, peak: int, paged: Set[int]) -> Optional[Set[int]]:
        """New paged set after a step that peaked at `peak` bytes, or None to keep it."""
        if self.done:
            return None
        self.steps += 1
        self.history.append({"step": self.steps, "peak": int(peak), "paged": len(paged)})
        new = None
        resident = [i for i in range(self.n) if i not in paged]
        if peak > self.target + 0.05 * self.budget.vram_usable and resident:
            more = min(len(resident), max(1, math.ceil((peak - self.target) / max(self.max_block, 1))))
            new = set(paged) | set(spread(more, resident))
        elif paged and peak + 1.5 * self.max_block < self.target:
            fewer = min(len(paged), int((self.target - peak) / max(self.max_block, 1)) - 1)
            if fewer >= 1:
                ordered = sorted(paged)
                new = set(spread(len(ordered) - fewer, ordered))
        if new is None:
            self.stable += 1
        else:
            self.stable = 0
        if self.stable >= 2 or self.steps >= self.max_refine_steps:
            self.done = True
        return new


def host_bytes_needed(trainable, paged_frozen_bytes: int, *, moment_bytes: int, grad_acc: bool) -> int:
    """Host RAM the optimizer engine and the pager will hold."""
    total = 0
    grad_bytes = 0
    for p in trainable:
        n = p.numel()
        pb = p.element_size()
        total += n * (4 + (pb if pb < 4 else 0) + 2 * moment_bytes + (4 if grad_acc else 0))
        grad_bytes += n * pb
    staging = max(1 << 30, grad_bytes // 4) if trainable else 0
    return total + min(staging, grad_bytes) + paged_frozen_bytes


def explain_ram_shortfall(need: int, have: int, budget: Budget, *, trainable: int, frozen: int) -> str:
    return (f"TrainStream needs about {need / GiB:.1f} GiB of host RAM for this run but only "
            f"{have / GiB:.1f} GiB is available ({budget.describe()}). "
            f"Trained parameters cost 8-16 bytes each in host RAM ({trainable / 1e9:.2f}B here) and paged frozen "
            f"weights 2-4 bytes ({frozen / 1e9:.2f}B here). Options: close other programs, train fewer parameters "
            f"(LoRA), use moment_dtype='bfloat16', a smaller model or a machine with more RAM. "
            f"Pass check_memory=False to try anyway.")
