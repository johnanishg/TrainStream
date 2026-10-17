"""TrainStream: PCIe offloading engine for PyTorch training.

    import trainstream
    model, optimizer = trainstream.wrap(model, optimizer)

Optimizer state, gradients, a profiled fraction of activations and (when
needed) the transformer-block weights themselves are moved to host RAM over
dedicated CUDA copy streams, so models that would not fit in VRAM with a stock
optimizer can still be trained on a single GPU — or on several, data parallel.
"""

import sys

try:
    import torch  # noqa: F401  (loads libtorch / c10 before the extension)
except ImportError as _exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "TrainStream needs PyTorch with CUDA. Install it first, e.g.\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu130") from _exc

try:
    from . import _C  # prebuilt (build.bat / setup.py build_ext / --no-build-isolation install)
except ImportError:
    from . import _build

    _C = _build.load()  # compiled on first import, then loaded from the cache
    sys.modules[__name__ + "._C"] = _C

from .hooks import ActivationOffloader, GradientOffloader, ParamReadyHooks, find_blocks
from .losses import causal_lm_loss, chunked_cross_entropy
from .optimizer import OffloadAdamW
from .paging import ParamPager
from .profiler import HardwareProfile, profile_hardware
from .wrapper import TrainStreamSession, memory_report, unwrap, wrap

__version__ = "0.2.0"

__all__ = [
    "wrap",
    "unwrap",
    "memory_report",
    "OffloadAdamW",
    "ParamPager",
    "chunked_cross_entropy",
    "causal_lm_loss",
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
