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
// (timestamp_ms, op, bytes, tag) — op is one of "alloc", "free", "hit",
// "release", "arena" (served from the arena) or "slab" (the arena grew).
std::vector<std::tuple<double, std::string, int64_t, std::string>> pinned_log(int64_t limit);
void pinned_empty_cache();
void pinned_set_cache_limit(int64_t bytes);
// Pinned arena: page-locked slabs (`slab` bytes each; <= 0 keeps the current
// size), `bytes` of them reserved now, from which every non-mapped block is
// then carved. Grows on demand up to `cap` (at least `bytes`); returns the
// bytes reserved so far.
int64_t pinned_reserve(int64_t bytes, int64_t slab, int64_t cap);
void pinned_arena_trim();  // give empty slabs back to the OS
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
// H2D from pageable (not page-locked) host tensors: staged with all CPU cores
// into one of `slots` pinned slots, then copied on the H2D stream after
// `after`. The host blocks only for the slot's previous copy.
EventPtr h2d_bounce_async(const std::vector<at::Tensor>& dsts, const std::vector<at::Tensor>& srcs,
                          const EventPtr& after, int64_t slots);
// Pinned host pages copied back to back into the start of a contiguous device
// tensor on the H2D stream, after `after` (the paged KV cache's gather).
EventPtr h2d_pages_async(const at::Tensor& dst, const std::vector<at::Tensor>& pages, const EventPtr& after);
std::map<std::string, int64_t> transfer_stats(int device);

// Expand blockwise-quantized weights (kind "nf4" or "int8") into `dst` on the
// H2D stream, after `after`. `q`/`absmax` may be in host RAM (copied first) or
// already on the GPU. The returned tensors must stay alive until the event
// has completed; the consumer waits on the event before using `dst`.
std::tuple<EventPtr, std::vector<at::Tensor>> dequantize_async(const at::Tensor& dst, const at::Tensor& q,
                                                               const at::Tensor& absmax, const std::string& kind,
                                                               int64_t block, const EventPtr& after);

// Host memory shared by the processes of one machine (see shm.cpp).
at::Tensor shared_host(const std::string& name, int64_t bytes, bool create, bool pin);
int64_t shm_reducer_bytes(int64_t world, int64_t capacity);
void shm_allreduce(const at::Tensor& buf, const at::Tensor& segment, int64_t rank, int64_t world, int64_t capacity,
                   int64_t seq);

// Peer GPUs (see peer.cpp). Topology: one row per ordered pair of visible
// devices (src, dst, can_access, performance_rank, native_atomics, gbps — the
// measured dst <- src copy rate when probe_bytes > 0).
std::vector<std::map<std::string, double>> peer_topology(int64_t probe_bytes);
std::string pci_bus_id(int device);
// A cudaMalloc'd byte pool on `device` and its CUDA IPC handle.
std::tuple<at::Tensor, std::vector<uint8_t>> peer_pool(int64_t bytes, int device);
int64_t peer_open(const std::vector<uint8_t>& handle, int device);  // -> device address in this process
void peer_close(int64_t ptr, int device);
// dst (this GPU) <- [src_ptr + offset, + dst.nbytes()) on src_device, on the H2D stream after `after`.
EventPtr peer_copy_async(const at::Tensor& dst, int64_t src_ptr, int64_t offset, int src_device,
                         const EventPtr& after);
// Measured GB/s of copies into `device` from an opened peer address (no context on the peer GPU).
double peer_copy_gbps(int64_t src_ptr, int src_device, int64_t bytes, int device);

// Block-scaled formats ("nvfp4", "fp8" = MXFP8) that Blackwell tensor cores
// multiply directly (torch's scaled_mm). quantize_blockscaled runs on the
// current stream and returns (packed values, swizzled scales); nvfp4 also
// takes a 1-element device tensor with the global decode scale and, optionally,
// a uint8 column mask (non-zero: quantize that column as 0).
std::tuple<at::Tensor, at::Tensor> quantize_blockscaled(const at::Tensor& x, const std::string& kind,
                                                        const c10::optional<at::Tensor>& global_decode,
                                                        const c10::optional<at::Tensor>& mask);
// Per-column max |x| of a 2-D CUDA tensor, as float32 [cols] (current stream).
at::Tensor col_absmax(const at::Tensor& x);
// Columns `idx` of an nvfp4-packed [rows, cols] matrix, expanded to `dtype`
// as a [rows, idx.numel()] tensor (current stream).
at::Tensor dequantize_blockscaled_cols(const at::Tensor& q, const at::Tensor& scales, const at::Tensor& idx,
                                       int64_t rows, int64_t cols, double global_decode, at::ScalarType dtype);
// Expand into a contiguous 2-D `dst` on the current stream.
void dequantize_blockscaled(const at::Tensor& dst, const at::Tensor& q, const at::Tensor& scales,
                            const std::string& kind, double global_decode);
// The same on the H2D stream after `after`; `q`/`scales` may be in host RAM.
std::tuple<EventPtr, std::vector<at::Tensor>> dequantize_blockscaled_async(const at::Tensor& dst, const at::Tensor& q,
                                                                           const at::Tensor& scales,
                                                                           const std::string& kind,
                                                                           double global_decode,
                                                                           const EventPtr& after);

// ---------------------------------------------------------------------------
// Micro-benchmarks (CUDA-event timed)
// ---------------------------------------------------------------------------
std::map<std::string, double> benchmark_pcie(int device, int64_t bytes, int iters);
std::map<std::string, double> benchmark_matmul(int device, int64_t n, at::ScalarType dtype,
                                               int iters);

double now_ms();

}  // namespace trainstream
