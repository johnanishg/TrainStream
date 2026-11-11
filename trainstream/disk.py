"""A disk tier for frozen weights: memory-mapped files under a directory you choose.

Frozen weights the page-lock budget can't hold otherwise live in ordinary host
RAM. With wrap(disk_offload=path) (or load_model(disk_offload=path), which
writes a checkpoint's block weights there as it reads them, so a base larger
than RAM can load) they live in files mapped into memory instead
(torch.from_file). The pager copies them to the GPU the way it copies any
pageable weight, through pinned bounce buffers on host threads; the reads are
page faults the OS serves from its page cache while RAM has room, and from the
disk (NVMe) when it doesn't. Under memory pressure the OS drops those clean
pages instead of swapping anything out.

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

import torch

_STORAGES = set()  # data pointers of the file-backed storages (views and Parameters lose attributes)


class DiskStore:
    def __init__(self, root: str):
        os.makedirs(root, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="trainstream-", dir=root)
        self.count = 0
        self.bytes = 0
        atexit.register(self.close)

    def store(self, t: torch.Tensor) -> torch.Tensor:
        """A flat CPU tensor backed by a new file, holding t's values."""
        n = t.numel()
        path = os.path.join(self.dir, f"{self.count}.bin")
        self.count += 1
        nbytes = n * t.element_size()
        with open(path, "wb") as fh:
            fh.truncate(nbytes)
        m = torch.from_file(path, shared=True, size=n, dtype=t.dtype)
        m.copy_(t.detach().reshape(-1))
        m._ts_pinned = False
        _STORAGES.add(m.untyped_storage().data_ptr())
        self.bytes += nbytes
        return m

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def is_on_disk(t) -> bool:
    """Whether t's storage is one of DiskStore's files."""
    return isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.untyped_storage().data_ptr() in _STORAGES
