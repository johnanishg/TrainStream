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

// 0 spin, 1 block, 2 poll (default); TRAINSTREAM_EVENT_WAIT, read once. See memory_pinner.cpp.
int event_wait_mode();

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
  cudaStream_t disk = nullptr; // host -> device for weights read from files (waits for the SSD, see disk.cpp)

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
// The same copy issued as several back-to-back pieces, with one event after each:
// bounds[k] is where piece k ends, in bytes from the start (increasing; the last
// is the tensor's size). A consumer may start on the first bytes of `dst` while
// the rest are still crossing PCIe; the last event is the whole copy's. `dst`
// is contiguous here (its layout is src's shape), so byte ranges are row ranges.
std::tuple<at::Tensor, std::vector<EventPtr>> h2d_alloc_chunked_async(const at::Tensor& src, const EventPtr& after,
                                                                      int device, const std::vector<int64_t>& sizes,
                                                                      const std::vector<int64_t>& bounds);
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
// Fused block ops (fused.cpp): on PyTorch's current stream, for trainstream/fused.py
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> rms_norm_forward(const at::Tensor& x, const at::Tensor& w, double eps, bool gemma);
std::tuple<at::Tensor, c10::optional<at::Tensor>> rms_norm_backward(const at::Tensor& dy, const at::Tensor& x,
                                                                    const at::Tensor& w, const at::Tensor& rstd,
                                                                    bool gemma, bool need_dw);
at::Tensor rope(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin, bool backward);
at::Tensor gated_act_forward(const at::Tensor& g, const at::Tensor& u, int64_t act);
std::tuple<at::Tensor, at::Tensor> gated_act_backward(const at::Tensor& dh, const at::Tensor& g, const at::Tensor& u,
                                                      int64_t act);

// The front of a gated MLP on an nvfp4 base in one kernel (gemm_fp4_kernel.cu):
//     out[p] = x W[p]^T + xa[p] sb[p]^T      h = act(out[0]) * out[1]      (two projections)
// x and each W[p] as TrainStream packs nvfp4 (codes, swizzled scales, tensor scale: xg a
// 1-element device tensor, wg[p] numbers); xa[p] [M, R] and sb[p] [N, R] bf16 with R a
// multiple of 16, or empty tensors for a projection without a LoRA term. Returns bf16
// [M, N] tensors on the current stream: h first when want_h (two projections only), then
// each out[p] when want_out.
bool gemm_fp4_supported(int device);  // this build holds the kernel and the GPU is an sm_120 one
std::vector<at::Tensor> gemm_fp4_front(const at::Tensor& xq, const at::Tensor& xs, const at::Tensor& xg,
                                       const std::vector<at::Tensor>& wq, const std::vector<at::Tensor>& ws,
                                       const std::vector<double>& wg, int64_t M, int64_t N, int64_t K,
                                       const std::vector<at::Tensor>& xa, const std::vector<at::Tensor>& sb,
                                       int64_t act, bool want_h, bool want_out);

// ---------------------------------------------------------------------------
// Ring all-reduce over TCP (ring.cpp): data-parallel ranks on different machines
// ---------------------------------------------------------------------------
class Ring {
 public:
  // to_next / from_prev: connected, blocking stream sockets (OS handles) to rank + 1 and from
  // rank - 1. The Ring owns them and closes them. A send or receive that makes no progress for
  // timeout_s seconds fails.
  Ring(int64_t to_next, int64_t from_prev, int rank, int world, int timeout_s = 600);
  ~Ring();
  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;
  // buf (float32, host, contiguous) <- its sum over every rank, in place. Blocks until done;
  // every rank must call it for the same tensors in the same order. Bit-identical on all ranks.
  void allreduce(const at::Tensor& buf);
  std::map<std::string, double> stats() const;  // calls, bytes_sent, bytes_received, wall_ms, rank, world
  struct Impl;

 private:
  Impl* impl_;
};

// ---------------------------------------------------------------------------
// Disk tier (disk.cpp): weights in files, read with asynchronous unbuffered I/O
// ---------------------------------------------------------------------------
namespace aio {
class Batch;
}
// A block's files being read (or already read) into a pinned staging slot.
struct DiskStage {
  int device = 0;
  int64_t slot = 0;
  uint64_t generation = 0;
  int64_t key = -1;
  int64_t bytes = 0;
  bool hit = false;  // the slot already held (or was already reading) these bytes: no new reads
  std::shared_ptr<aio::Batch> batch;
};
// Queue unbuffered reads of files[i][offsets[i], +nbytes[i]) into one of `slots`
// pinned staging slots on `device` and return at once. `key` (>= 0) names the
// contents: asking for a key some slot already holds returns that slot without
// reading again. Offsets must be multiples of 4096.
std::shared_ptr<DiskStage> disk_stage(const std::vector<std::string>& paths, const std::vector<int64_t>& offsets,
                                      const std::vector<int64_t>& nbytes, int64_t key, int device, int64_t slots);
// dsts[i] <- the stage's i-th file, on the disk stream after `after`: the stream
// itself waits for the reads (a host callback), the host does not.
EventPtr h2d_disk_async(const std::vector<at::Tensor>& dsts, const std::shared_ptr<DiskStage>& stage,
                        const EventPtr& after);
bool disk_stage_ready(const DiskStage& stage);
void disk_stage_wait(const DiskStage& stage);
void disk_invalidate(int device);  // staged bytes are stale (a file was rewritten); -1: every device
void disk_release(int device);     // give the staging slots back; -1: every device
void disk_close(const std::vector<std::string>& paths);  // close cached handles (all when empty)
// Write a mapped file's dirty pages to the device, so unbuffered reads see them.
void disk_sync(const at::Tensor& mapped);
// dst <- file[offset, +dst.nbytes()), blocking, through the same unbuffered path.
void disk_read(const at::Tensor& dst, const std::string& path, int64_t offset);
std::string disk_backend();  // "iocp", "io_uring" or "threads"
std::map<std::string, double> disk_stats();

// ---------------------------------------------------------------------------
// Non-finite guard (guard.cpp): where a NaN / Inf first appeared
// ---------------------------------------------------------------------------
// Device-visible address of this device's flag words (kernels::GuardWord), in
// mapped pinned host memory; nullptr while the kernel checks are switched off.
int32_t* guard_flags(int device);
int32_t* guard_flags_always(int device);  // the same, whatever guard_enable says
void guard_enable(bool on);
bool guard_enabled();
// One value per kernels::GuardWord (non-zero: that kernel saw a NaN / Inf since
// the flags were last cleared). Read straight from host memory: no stream sync.
std::vector<int64_t> guard_read(int device, bool clear);
const char* guard_word_name(int word);

// ---------------------------------------------------------------------------
// Micro-benchmarks (CUDA-event timed)
// ---------------------------------------------------------------------------
std::map<std::string, double> benchmark_pcie(int device, int64_t bytes, int iters);
std::map<std::string, double> benchmark_matmul(int device, int64_t n, at::ScalarType dtype,
                                               int iters);

double now_ms();

}  // namespace trainstream
