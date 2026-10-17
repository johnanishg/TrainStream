"""TrainStream: PCIe offloading engine for PyTorch training.

    import trainstream
    model, optimizer = trainstream.wrap(model, optimizer)

Optimizer state, gradients and (a profiled fraction of) activations are moved
to pinned host RAM over dedicated CUDA copy streams, so models that would not
fit in VRAM with a stock optimizer can still be trained on a single GPU.
"""

import torch  # noqa: F401  (loads libtorch / c10 before the extension)

try:
    from . import _C  # compiled extension (python setup.py build_ext --inplace)
except ImportError as _exc:  # pragma: no cover - depends on build state
    raise ImportError(
        "trainstream._C is not built. Run `python setup.py build_ext --inplace` "
        "(Windows: build.bat) from the repository root. Original error: " + str(_exc)
    ) from _exc

from .hooks import ActivationOffloader, GradientOffloader, ParamReadyHooks, find_blocks
from .optimizer import OffloadAdamW
from .profiler import HardwareProfile, profile_hardware
from .wrapper import TrainStreamSession, memory_report, unwrap, wrap

__version__ = "0.1.0"

__all__ = [
    "wrap",
    "unwrap",
    "memory_report",
    "OffloadAdamW",
    "ActivationOffloader",
    "GradientOffloader",
    "ParamReadyHooks",
    "TrainStreamSession",
    "HardwareProfile",
    "profile_hardware",
    "find_blocks",
    "pinned_stats",
    "transfer_stats",
]


def pinned_stats():
    """Pin-logging allocator counters plus live bytes per allocation tag."""
    stats = dict(_C.pinned_stats())
    stats["by_tag"] = dict(_C.pinned_tag_bytes())
    return stats


def transfer_stats(device=None):
    """Cumulative bytes / copies issued on TrainStream's D2H and H2D streams."""
    idx = torch.cuda.current_device() if device is None else (torch.device(device).index or 0)
    return dict(_C.transfer_stats(idx))
