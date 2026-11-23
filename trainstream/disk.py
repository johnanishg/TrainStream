"""A disk tier for frozen weights: files under a directory you choose.

Frozen weights the page-lock budget can't hold otherwise live in ordinary host
RAM. With wrap(disk_offload=path) (or load_model(disk_offload=path), which
writes a checkpoint's block weights there as it reads them, so a base larger
than RAM can load) they live in files instead, one per tensor.

Two views of each file exist side by side:

* A memory mapping (torch.from_file), which is what the rest of the library
  sees: an ordinary CPU tensor, for writing the values once, for state_dict()
  and for anything else that reads a weight now and then. It costs RAM only
  while the OS keeps its pages cached.
* The training path, which does not touch the mapping. The pager reads a
  block's files with asynchronous unbuffered I/O (csrc/aio.cpp: an I/O
  completion port on Windows, io_uring on Linux, a thread pool where neither
  is available) straight into pinned staging memory, a block or more ahead of
  the layer that needs it, and copies from there to the GPU on a stream that
  waits for the read on the GPU side. No page cache, no page fault, and the
  thread running the training loop never waits for the SSD.

Whatever is written through a mapping is flushed to the device at once
(sync), so the unbuffered reads always see it.

Each DiskStore makes its own folder (trainstream-XXXX) under the directory and
deletes it when the process exits, as far as the OS lets it: on Windows a file
that is still mapped can't be deleted, so a folder can be left behind after a
crash; delete it by hand.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from typing import Optional, Tuple

import torch

ALIGN = 4096  # unbuffered I/O moves whole sectors: every file is padded to a multiple of this

_PATHS = set()  # normalized paths of every DiskStore file


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class DiskStore:
    def __init__(self, root: str):
        os.makedirs(root, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="trainstream-", dir=root)
        self.count = 0
        self.bytes = 0
        self._paths = []
        atexit.register(self.close)

    def store(self, t: torch.Tensor) -> torch.Tensor:
        """A flat CPU tensor backed by a new file, holding t's values (already on the device)."""
        n = t.numel()
        path = os.path.join(self.dir, f"{self.count}.bin")
        self.count += 1
        nbytes = n * t.element_size()
        with open(path, "wb") as fh:
            fh.truncate(max(ALIGN, -(-nbytes // ALIGN) * ALIGN))
        m = torch.from_file(path, shared=True, size=n, dtype=t.dtype)
        m.copy_(t.detach().reshape(-1))
        m._ts_pinned = False
        _PATHS.add(_norm(path))
        self._paths.append(path)
        sync(m)
        self.bytes += nbytes
        return m

    def close(self) -> None:
        try:
            from . import _C
            _C.disk_close(self._paths)  # our read handles, so the files can go
        except Exception:
            pass
        for p in self._paths:
            _PATHS.discard(_norm(p))
        self._paths = []
        shutil.rmtree(self.dir, ignore_errors=True)


def _filename(t) -> Optional[str]:
    """The file `t`'s storage is mapped from, if it is one of DiskStore's."""
    if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
        return None
    try:
        name = t.untyped_storage().filename
    except (AttributeError, RuntimeError):
        return None
    return name if name is not None and _norm(name) in _PATHS else None


def is_on_disk(t) -> bool:
    """Whether t's storage is one of DiskStore's files."""
    return _filename(t) is not None


def file_of(t) -> Optional[Tuple[str, int, int]]:
    """(path, offset, nbytes) when `t` is the whole of one DiskStore file, laid out as the file
    is (what the unbuffered read path can serve); None for anything else."""
    name = _filename(t)
    if name is None or t.storage_offset() != 0 or not t.is_contiguous():
        return None
    nbytes = t.numel() * t.element_size()
    if nbytes != t.untyped_storage().nbytes():
        return None
    return name, 0, nbytes


def sync(t: torch.Tensor) -> None:
    """Write what was stored into disk-backed `t` through its mapping to the device, and drop
    staged copies of the old bytes. Call after writing into a DiskStore tensor."""
    from . import _C
    _C.disk_sync(t)
    _C.disk_invalidate(-1)
