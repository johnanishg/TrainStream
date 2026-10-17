"""Start-up hardware profiler.

Measures, with CUDA event timers on TrainStream's own copy streams:
  * pinned H2D / D2H / full-duplex PCIe bandwidth, pageable bandwidth, a
    transfer-size sweep and 4 KiB DMA latency;
  * dense matmul throughput (bf16 and fp32);
  * host AdamW throughput (C++ worker threads) and zero-copy AdamW throughput.

From these it derives the quantities the engine needs: how long a transfer of
N bytes takes, how much of the activation stack can be paged without stalling
compute, and which optimizer mode finishes a step sooner.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import torch

from . import _C

_CACHE: Dict[tuple, "HardwareProfile"] = {}


@dataclass
class HardwareProfile:
    device_index: int
    device_name: str
    vram_bytes: int
    sm_count: int
    h2d_gbps: float
    d2h_gbps: float
    bidir_gbps: float
    pageable_h2d_gbps: float
    latency_us_4k: float
    matmul_tflops_bf16: float
    matmul_tflops_fp32: float
    cpu_adamw_gelems: float
    cpu_threads: int
    zero_copy_adamw_gelems: float
    zero_copy_pcie_gbps: float
    sweep_h2d: Dict[int, float] = field(default_factory=dict)
    sweep_d2h: Dict[int, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    # -- derived quantities --------------------------------------------------
    def transfer_ms(self, nbytes: float, direction: str = "h2d") -> float:
        gbps = self.h2d_gbps if direction == "h2d" else self.d2h_gbps
        return 0.0 if gbps <= 0 else nbytes / (gbps * 1e9) * 1e3

    def compute_ms(self, flops: float, dtype: str = "bf16") -> float:
        tflops = self.matmul_tflops_bf16 if dtype == "bf16" else self.matmul_tflops_fp32
        return 0.0 if tflops <= 0 else flops / (tflops * 1e12) * 1e3

    @property
    def break_even_flops_per_byte(self) -> float:
        """Arithmetic intensity (bf16 FLOPs per PCIe byte) above which a
        transfer can be fully hidden behind matmul compute."""
        return (self.matmul_tflops_bf16 * 1e12) / max(self.h2d_gbps * 1e9, 1.0)

    @property
    def duplex_factor(self) -> float:
        """1.0 when H2D and D2H serialise on one copy engine, ~2.0 for full duplex."""
        one_way = max(self.h2d_gbps, self.d2h_gbps, 1e-9)
        return min(2.0, max(1.0, self.bidir_gbps / one_way))

    def link_ms(self, d2h_bytes: float, h2d_bytes: float) -> float:
        """Wall time the PCIe link needs for this mix, honouring the measured duplex factor."""
        d2h, h2d = self.transfer_ms(d2h_bytes, "d2h"), self.transfer_ms(h2d_bytes, "h2d")
        return max(d2h, h2d, (d2h + h2d) / self.duplex_factor)

    def activation_offload_fraction(self, block_bytes: float, step_ms: float, n_blocks: int,
                                    reserved_link_ms: float = 0.0, safety: float = 0.85) -> float:
        """Fraction of the (N-1) pageable blocks whose activations can be paged for free.

        Each paged block costs one D2H (forward) and one H2D (backward) of
        `block_bytes`. They fit for free while the total link time stays under
        the measured step time, after reserving the link time the optimizer's
        own traffic needs:
            K = (step_ms * safety - reserved_link_ms) / link_ms(block, block)
        """
        per_block = self.link_ms(block_bytes, block_bytes)
        if per_block <= 0 or n_blocks <= 1:
            return 1.0 if per_block <= 0 else 0.0
        k = (step_ms * safety - reserved_link_ms) / per_block
        return max(0.0, min(1.0, k / (n_blocks - 1)))

    def optimizer_step_ms(self, numel: int, mode: str, param_bytes: int = 4) -> float:
        """Estimated wall time of one optimizer step over `numel` parameters."""
        if mode == "gpu":
            return numel / max(self.zero_copy_adamw_gelems * 1e9, 1.0) * 1e3
        host = numel / max(self.cpu_adamw_gelems * 1e9, 1.0) * 1e3
        upload = self.transfer_ms(numel * param_bytes, "h2d")
        return max(host, upload)  # pipelined per tensor: the slower stage dominates

    def recommend_optimizer_mode(self, numel: int) -> str:
        # cpu mode also streams gradients out of VRAM and overlaps with the next
        # forward, so prefer it unless the host is dramatically slower.
        cpu = self.optimizer_step_ms(numel, "cpu")
        gpu = self.optimizer_step_ms(numel, "gpu")
        return "cpu" if cpu <= 4.0 * gpu else "gpu"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["break_even_flops_per_byte"] = self.break_even_flops_per_byte
        d["duplex_factor"] = self.duplex_factor
        d["sweep_h2d"] = {str(k): v for k, v in self.sweep_h2d.items()}
        d["sweep_d2h"] = {str(k): v for k, v in self.sweep_d2h.items()}
        return d

    def summary(self) -> str:
        return "\n".join([
            f"TrainStream hardware profile — {self.device_name} (cuda:{self.device_index})",
            f"  PCIe pinned   H2D {self.h2d_gbps:6.1f} GB/s | D2H {self.d2h_gbps:6.1f} GB/s | duplex {self.bidir_gbps:6.1f} GB/s",
            f"  duplex factor {self.duplex_factor:.2f} (1 = copy engines serialise, 2 = full duplex)",
            f"  PCIe pageable H2D {self.pageable_h2d_gbps:6.1f} GB/s | 4 KiB DMA latency {self.latency_us_4k:.1f} us",
            f"  matmul        bf16 {self.matmul_tflops_bf16:6.1f} TFLOPS | fp32 {self.matmul_tflops_fp32:6.1f} TFLOPS",
            f"  AdamW         host {self.cpu_adamw_gelems:5.2f} Gelem/s ({self.cpu_threads} threads) | "
            f"zero-copy {self.zero_copy_adamw_gelems:5.2f} Gelem/s",
            f"  break-even intensity {self.break_even_flops_per_byte:,.0f} FLOP/byte",
        ])


def profile_hardware(device=None, quick: bool = True, force: bool = False,
                     threads: Optional[int] = None) -> HardwareProfile:
    """Run (or return the cached) micro-benchmarks for `device`."""
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (idx, quick, threads)
    if not force and key in _CACHE:
        return _CACHE[key]

    t0 = time.perf_counter()
    with torch.cuda.device(idx):
        _C.init(idx)
        pcie = _C.benchmark_pcie(idx, (64 if quick else 256) << 20, 6 if quick else 12)
        mm_bf16 = _C.benchmark_matmul(idx, 4096 if quick else 8192, torch.bfloat16, 10 if quick else 30)
        mm_fp32 = _C.benchmark_matmul(idx, 2048 if quick else 4096, torch.float32, 5 if quick else 15)
        cpu = _C.benchmark_cpu_adamw((1 << 22) if quick else (1 << 25), int(threads or 0))
        zc = _C.benchmark_zero_copy_adamw(idx, (1 << 22) if quick else (1 << 25))
        _C.pinned_empty_cache()
    props = torch.cuda.get_device_properties(idx)
    prof = HardwareProfile(
        device_index=idx,
        device_name=props.name,
        vram_bytes=int(props.total_memory),
        sm_count=int(props.multi_processor_count),
        h2d_gbps=pcie["h2d_gbps"],
        d2h_gbps=pcie["d2h_gbps"],
        bidir_gbps=pcie["bidir_gbps"],
        pageable_h2d_gbps=pcie["pageable_h2d_gbps"],
        latency_us_4k=pcie["latency_us_4k"],
        matmul_tflops_bf16=mm_bf16["tflops"],
        matmul_tflops_fp32=mm_fp32["tflops"],
        cpu_adamw_gelems=cpu["gelems_per_s"],
        cpu_threads=int(cpu["threads"]),
        zero_copy_adamw_gelems=zc["gelems_per_s"],
        zero_copy_pcie_gbps=zc["pcie_gbps"],
        sweep_h2d={int(k.rsplit("_", 1)[1]): v for k, v in pcie.items() if k.startswith("sweep_h2d_")},
        sweep_d2h={int(k.rsplit("_", 1)[1]): v for k, v in pcie.items() if k.startswith("sweep_d2h_")},
        elapsed_s=time.perf_counter() - t0,
    )
    _CACHE[key] = prof
    return prof
