"""TrainStream: PCIe offloading engine for PyTorch training.

    import trainstream
    model, optimizer = trainstream.wrap(model, optimizer)

Optimizer state, gradients, a profiled fraction of activations and (when
needed) the transformer-block weights themselves (frozen ones optionally
quantized to int8 / NF4) are moved to host RAM over
dedicated CUDA copy streams, so models that would not fit in VRAM with a stock
optimizer can still be trained on a single GPU — or on several, data parallel.
"""

import sys

try:
    import torch  # noqa: F401  (loads libtorch / c10 before the extension)
except ImportError as _exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "TrainStream needs PyTorch with CUDA. Install it first, e.g.\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
        "    uv pip install torch --torch-backend=auto") from _exc

try:
    from . import _C  # prebuilt (build.bat / setup.py build_ext / --no-build-isolation install)
except ImportError:
    from . import _build

    _C = _build.load()  # compiled on first import, then loaded from the cache
    sys.modules[__name__ + "._C"] = _C

from .hooks import ActivationOffloader, GradientOffloader, ParamReadyHooks, find_blocks
from .kvcache import PagedKVCache
from .losses import causal_lm_loss, chunked_cross_entropy, dpo_loss, sequence_logprobs
from .optimizer import OffloadAdamW
from .paging import ParamPager
from .profiler import HardwareProfile, profile_hardware
from .packing import PackedCollator, pack_sequences
from .quant import init_lora_from_error, quantize_model
from .wrapper import TrainStreamSession, memory_report, unwrap, wrap

__version__ = "0.7.0"

__all__ = [
    "wrap",
    "unwrap",
    "memory_report",
    "OffloadAdamW",
    "ParamPager",
    "PagedKVCache",
    "chunked_cross_entropy",
    "causal_lm_loss",
    "sequence_logprobs",
    "dpo_loss",
    "ActivationOffloader",
    "GradientOffloader",
    "ParamReadyHooks",
    "TrainStreamSession",
    "HardwareProfile",
    "profile_hardware",
    "quantize_model",
    "find_blocks",
    "pinned_stats",
    "transfer_stats",
    "reserve_pinned",
    "pack_sequences",
    "PackedCollator",
    "init_lora_from_error",
]


def pinned_stats():
    """Pin-logging allocator counters plus live bytes per allocation tag."""
    stats = dict(_C.pinned_stats())
    stats["by_tag"] = dict(_C.pinned_tag_bytes())
    return stats


def reserve_pinned(bytes_now: int, cap: int = 0, slab: int = 0) -> int:
    """Page-lock `bytes_now` of host RAM for TrainStream's pinned arena now and
    let it grow on demand up to `cap` (at least `bytes_now`), in slabs of
    `slab` bytes (0: keep the current size, 1 GiB unless
    TRAINSTREAM_HOST_ARENA_SLAB says otherwise). wrap(host_arena=...) does
    this for you; returns the bytes reserved so far."""
    return int(_C.pinned_reserve(int(bytes_now), int(slab), int(cap)))


def transfer_stats(device=None):
    """Cumulative bytes / copies issued on TrainStream's D2H and H2D streams."""
    idx = torch.cuda.current_device() if device is None else (torch.device(device).index or 0)
    return dict(_C.transfer_stats(idx))
