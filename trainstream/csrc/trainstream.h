// TrainStream — shared runtime declarations (events, streams, pinned memory,
// asynchronous transfers). Implemented in memory_pinner.cpp.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#define TS_CUDA_CHECK(expr)                                                                   \
  do {                                                                                       \
    const cudaError_t _ts_err = (expr);                                                      \
    TORCH_CHECK(_ts_err == cudaSuccess, "TrainStream CUDA error: ", cudaGetErrorString(_ts_err), \
                " (", #expr, ") at ", __FILE__, ":", __LINE__);                              \
  } while (0)

namespace trainstream {

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
// RAII wrapper around cudaEvent_t. Non-timing events are recycled through a
// per-device pool because transfer-heavy workloads create thousands per step.
// Re-recording a recycled event is safe: cudaStreamWaitEvent snapshots the
// event's state at call time, so earlier waiters are unaffected.
class Event {
 public:
  Event(int device, bool timing);
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  void record(cudaStream_t stream);
  bool query() const;                    // true if completed (or never recorded)
  void synchronize() const;              // host blocks until completion
  void block(cudaStream_t stream) const; // GPU-side: stream waits for this event
  float elapsed_ms(const Event& end) const;
  bool recorded() const { return recorded_; }
  int device() const { return device_; }

 private:
  cudaEvent_t ev_ = nullptr;
  int device_;
  bool timing_;
  bool recorded_ = false;
};
using EventPtr = std::shared_ptr<Event>;

EventPtr make_event(int device, bool timing = false);
EventPtr record_on_current(int device, bool timing = false);

// ---------------------------------------------------------------------------
// Runtime: dedicated streams + traffic counters, one instance per device.
// ---------------------------------------------------------------------------
struct Runtime {
  explicit Runtime(int device);

  int device;
  int sm_count = 0;
  cudaStream_t d2h = nullptr;  // device -> host (gradients, activations out)
  cudaStream_t h2d = nullptr;  // host -> device (parameters, activation prefetch)
  cudaStream_t upd = nullptr;  // zero-copy optimizer kernels

  std::atomic<uint64_t> bytes_d2h{0};
  std::atomic<uint64_t> bytes_h2d{0};
  std::atomic<uint64_t> copies_d2h{0};
  std::atomic<uint64_t> copies_h2d{0};
};

Runtime& runtime(int device);
int resolve_device(int device);  // -1 -> current device
cudaStream_t compute_stream(int device);
std::vector<int> initialized_devices();

// Tell the CUDA caching allocator that `t`'s memory is in use on `stream`, so
// the block is not handed out again until that stream's queued work finishes.
void record_stream(const at::Tensor& t, cudaStream_t stream, int device);

// ---------------------------------------------------------------------------
// Pinned host memory (pin-logging caching allocator)
// ---------------------------------------------------------------------------
// cached=true  -> returned to a size-bucketed free list on release (cheap reuse;
//                 used for activation stashes that churn every step).
// cached=false -> cudaFreeHost on release (persistent optimizer state).
// mapped=true  -> cudaHostAllocMapped: device-addressable for zero-copy kernels.
at::Tensor pinned_empty(const std::vector<int64_t>& sizes, at::ScalarType dtype,
                        const std::string& tag, bool mapped, bool cached);
std::map<std::string, int64_t> pinned_stats();
std::map<std::string, int64_t> pinned_tag_bytes();
// (timestamp_ms, op, bytes, tag) — op is one of "alloc", "free", "hit", "release".
std::vector<std::tuple<double, std::string, int64_t, std::string>> pinned_log(int64_t limit);
void pinned_empty_cache();
void pinned_set_cache_limit(int64_t bytes);
void* pinned_device_pointer(const at::Tensor& t);  // for mapped allocations

// ---------------------------------------------------------------------------
// Asynchronous transfers on the dedicated streams.
// ---------------------------------------------------------------------------
// D2H: the copy stream first waits for the *current* compute stream so it
// reads fully produced data. The caller must keep `src` alive until the
// returned event completes (no recordStream; see memory_pinner.cpp).
EventPtr d2h_async(const at::Tensor& dst, const at::Tensor& src);
// H2D into an existing device tensor; waits for `after` (if any) first.
EventPtr h2d_async(const at::Tensor& dst, const at::Tensor& src, const EventPtr& after);
// H2D into a freshly allocated device tensor (from the compute stream's pool;
// the copy is ordered after work already queued there). The consumer must
// wait on the returned event before using or releasing the tensor.
// With `sizes`/`strides` the device tensor is created with that (dense) layout,
// restoring the exact strides a tensor had before it was paged out.
std::tuple<at::Tensor, EventPtr> h2d_alloc_async(const at::Tensor& src, const EventPtr& after,
                                                 int device, const std::vector<int64_t>& sizes = {},
                                                 const std::vector<int64_t>& strides = {});
std::map<std::string, int64_t> transfer_stats(int device);

// Expand blockwise-quantized weights (kind "nf4" or "int8") into `dst` on the
// H2D stream, after `after`. `q`/`absmax` may be in host RAM (copied first) or
// already on the GPU. The returned tensors must stay alive until the event
// has completed; the consumer waits on the event before using `dst`.
std::tuple<EventPtr, std::vector<at::Tensor>> dequantize_async(const at::Tensor& dst, const at::Tensor& q,
                                                               const at::Tensor& absmax, const std::string& kind,
                                                               int64_t block, const EventPtr& after);

// ---------------------------------------------------------------------------
// Micro-benchmarks (CUDA-event timed)
// ---------------------------------------------------------------------------
std::map<std::string, double> benchmark_pcie(int device, int64_t bytes, int iters);
std::map<std::string, double> benchmark_matmul(int device, int64_t n, at::ScalarType dtype,
                                               int iters);

double now_ms();

}  // namespace trainstream
