// TrainStream — peer GPUs: topology probe, IPC-exported VRAM pools and
// device-to-device copies between the processes of one machine.
//
// Data-parallel ranks run one process per GPU, so a peer's VRAM is reached
// through CUDA IPC: the owner exports a cudaMalloc'd pool (outside PyTorch's
// caching allocator, whose blocks cannot be exported on their own), the
// others open it and copy from it with cudaMemcpyPeerAsync on their H2D
// stream (over NVLink or PCIe P2P, never through host RAM). Where CUDA IPC is
// unavailable the calls fail with the driver's reason and TrainStream keeps
// streaming from host RAM.

#include "trainstream.h"

#include <cstring>

namespace trainstream {

std::vector<std::map<std::string, double>> peer_topology(int64_t probe_bytes) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return {};
  }
  std::vector<std::map<std::string, double>> out;
  for (int src = 0; src < count; ++src) {
    for (int dst = 0; dst < count; ++dst) {
      if (src == dst) continue;
      int access = 0, rank = 0, atomics = 0;
      if (cudaDeviceCanAccessPeer(&access, dst, src) != cudaSuccess) {
        (void)cudaGetLastError();
        access = 0;
      }
      if (cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, src, dst) != cudaSuccess) rank = -1;
      if (cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, src, dst) != cudaSuccess)
        atomics = 0;
      (void)cudaGetLastError();
      double gbps = 0.0;
      if (access && probe_bytes > 0) {
        // dst reads from src: enable dst -> src access, time copies on dst's H2D stream
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dst));
        const cudaError_t en = cudaDeviceEnablePeerAccess(src, 0);
        if (en == cudaSuccess || en == cudaErrorPeerAccessAlreadyEnabled) {
          (void)cudaGetLastError();
          void* a = nullptr;
          void* b = nullptr;
          const size_t bytes = static_cast<size_t>(probe_bytes);
          bool ok = cudaMalloc(&b, bytes) == cudaSuccess;
          {
            c10::cuda::CUDAGuard g2(static_cast<c10::DeviceIndex>(src));
            ok = ok && cudaMalloc(&a, bytes) == cudaSuccess;
          }
          if (ok) {
            Runtime& rt = runtime(dst);
            Event start(dst, true), stop(dst, true);
            const int iters = 4;
            TS_CUDA_CHECK(cudaMemcpyPeerAsync(b, dst, a, src, bytes, rt.h2d));  // warm-up
            start.record(rt.h2d);
            for (int i = 0; i < iters; ++i) TS_CUDA_CHECK(cudaMemcpyPeerAsync(b, dst, a, src, bytes, rt.h2d));
            stop.record(rt.h2d);
            const double ms = start.elapsed_ms(stop);
            gbps = ms > 0 ? static_cast<double>(bytes) * iters / (ms * 1e-3) / 1e9 : 0.0;
          }
          (void)cudaGetLastError();
          if (b) cudaFree(b);
          if (a) {
            c10::cuda::CUDAGuard g2(static_cast<c10::DeviceIndex>(src));
            cudaFree(a);
          }
          (void)cudaGetLastError();
        } else {
          (void)cudaGetLastError();
          access = 0;
        }
      }
      out.push_back({{"src", static_cast<double>(src)},
                     {"dst", static_cast<double>(dst)},
                     {"can_access", static_cast<double>(access)},
                     {"performance_rank", static_cast<double>(rank)},
                     {"native_atomics", static_cast<double>(atomics)},
                     {"gbps", gbps}});
    }
  }
  return out;
}

std::string pci_bus_id(int device) {
  char buf[64] = {0};
  if (cudaDeviceGetPCIBusId(buf, sizeof(buf), device) != cudaSuccess) {
    (void)cudaGetLastError();
    return "";
  }
  return std::string(buf);
}

std::tuple<at::Tensor, std::vector<uint8_t>> peer_pool(int64_t bytes, int device) {
  TORCH_CHECK(bytes > 0, "peer_pool: bytes must be positive");
  const int dev = resolve_device(device);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  void* ptr = nullptr;
  TS_CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(bytes)));
  cudaIpcMemHandle_t handle;
  const cudaError_t err = cudaIpcGetMemHandle(&handle, ptr);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    cudaFree(ptr);
    TORCH_CHECK(false, "peer_pool: CUDA IPC is not available here (", cudaGetErrorString(err), ")");
  }
  std::vector<uint8_t> raw(sizeof(handle));
  std::memcpy(raw.data(), &handle, sizeof(handle));
  auto t = at::from_blob(
      ptr, {bytes},
      [dev](void* p) {
        c10::cuda::CUDAGuard g(static_cast<c10::DeviceIndex>(dev));
        cudaFree(p);
        (void)cudaGetLastError();
      },
      at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(dev))));
  return {t, raw};
}

int64_t peer_open(const std::vector<uint8_t>& raw, int device) {
  cudaIpcMemHandle_t handle;
  TORCH_CHECK(raw.size() == sizeof(handle), "peer_open: an IPC handle has ", sizeof(handle), " bytes");
  std::memcpy(&handle, raw.data(), sizeof(handle));
  const int dev = resolve_device(device);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  void* ptr = nullptr;
  const cudaError_t err = cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    TORCH_CHECK(false, "peer_open: cannot map the peer's memory (", cudaGetErrorString(err), ")");
  }
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
}

void peer_close(int64_t ptr, int device) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(resolve_device(device)));
  cudaIpcCloseMemHandle(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
  (void)cudaGetLastError();
}

EventPtr peer_copy_async(const at::Tensor& dst, int64_t src_ptr, int64_t offset, int src_device,
                         const EventPtr& after) {
  TORCH_CHECK(dst.is_cuda() && dst.is_non_overlapping_and_dense(), "peer_copy_async: dst must be a dense CUDA tensor");
  const int dev = dst.get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  if (after) after->block(rt.h2d);
  const size_t bytes = dst.nbytes();
  const void* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(src_ptr)) + offset;
  if (bytes > 0) TS_CUDA_CHECK(cudaMemcpyPeerAsync(dst.data_ptr(), dev, src, src_device, bytes, rt.h2d));
  auto done = make_event(dev);
  done->record(rt.h2d);
  return done;
}

double peer_copy_gbps(int64_t src_ptr, int src_device, int64_t bytes, int device) {
  const int dev = resolve_device(device);
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  void* dst = nullptr;
  TS_CUDA_CHECK(cudaMalloc(&dst, static_cast<size_t>(bytes)));
  const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(src_ptr));
  Event start(dev, true), stop(dev, true);
  const int iters = 4;
  TS_CUDA_CHECK(cudaMemcpyPeerAsync(dst, dev, src, src_device, bytes, rt.h2d));  // warm-up
  start.record(rt.h2d);
  for (int i = 0; i < iters; ++i) TS_CUDA_CHECK(cudaMemcpyPeerAsync(dst, dev, src, src_device, bytes, rt.h2d));
  stop.record(rt.h2d);
  const double ms = start.elapsed_ms(stop);
  cudaFree(dst);
  return ms > 0 ? static_cast<double>(bytes) * iters / (ms * 1e-3) / 1e9 : 0.0;
}

}  // namespace trainstream
