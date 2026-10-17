// TrainStream — runtime core: CUDA event pool, dedicated copy streams, the
// pin-logging caching host allocator, asynchronous DMA helpers and the
// CUDA-event-timed micro-benchmarks used by the hardware profiler.

#include "trainstream.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace trainstream {

namespace {
constexpr int kMaxDevices = 64;

// Intentionally leaked singletons: destroying CUDA objects from static
// destructors races with driver teardown at interpreter exit.
std::mutex& runtime_mutex() {
  static auto* m = new std::mutex();
  return *m;
}
Runtime** runtime_table() {
  static auto* table = new Runtime*[kMaxDevices]();
  return table;
}

struct EventPool {
  std::mutex mu;
  std::vector<cudaEvent_t> free[kMaxDevices];
};
EventPool& event_pool() {
  static auto* pool = new EventPool();
  return *pool;
}
}  // namespace

double now_ms() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ===========================================================================
// Event
// ===========================================================================
Event::Event(int device, bool timing) : device_(device), timing_(timing) {
  if (!timing_) {
    auto& pool = event_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    auto& bucket = pool.free[device_];
    if (!bucket.empty()) {
      ev_ = bucket.back();
      bucket.pop_back();
      return;
    }
  }
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
  TS_CUDA_CHECK(cudaEventCreateWithFlags(&ev_, timing_ ? cudaEventDefault : cudaEventDisableTiming));
}

Event::~Event() {
  if (ev_ == nullptr) return;
  if (!timing_) {
    auto& pool = event_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    pool.free[device_].push_back(ev_);
  } else {
    cudaEventDestroy(ev_);
  }
}

void Event::record(cudaStream_t stream) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
  TS_CUDA_CHECK(cudaEventRecord(ev_, stream));
  recorded_ = true;
}

bool Event::query() const {
  if (!recorded_) return true;
  const cudaError_t err = cudaEventQuery(ev_);
  if (err == cudaErrorNotReady) {
    (void)cudaGetLastError();  // clear the sticky "not ready" status
    return false;
  }
  TS_CUDA_CHECK(err);
  return true;
}

void Event::synchronize() const {
  if (!recorded_) return;
  TS_CUDA_CHECK(cudaEventSynchronize(ev_));
}

void Event::block(cudaStream_t stream) const {
  if (!recorded_) return;  // waiting on a never-recorded event is a no-op in CUDA too
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
  TS_CUDA_CHECK(cudaStreamWaitEvent(stream, ev_, 0));
}

float Event::elapsed_ms(const Event& end) const {
  TORCH_CHECK(timing_ && end.timing_, "elapsed_ms requires two timing events");
  TORCH_CHECK(recorded_ && end.recorded_, "elapsed_ms requires both events to be recorded");
  end.synchronize();
  float ms = 0.f;
  TS_CUDA_CHECK(cudaEventElapsedTime(&ms, ev_, end.ev_));
  return ms;
}

EventPtr make_event(int device, bool timing) { return std::make_shared<Event>(device, timing); }

EventPtr record_on_current(int device, bool timing) {
  const int dev = resolve_device(device);
  auto ev = make_event(dev, timing);
  ev->record(compute_stream(dev));
  return ev;
}

// ===========================================================================
// Runtime
// ===========================================================================
Runtime::Runtime(int dev) : device(dev) {
  at::globalContext().lazyInitDevice(at::kCUDA);  // torch must own the CUDA context first
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  cudaDeviceProp prop{};
  TS_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  sm_count = prop.multiProcessorCount;
  int least = 0, greatest = 0;
  TS_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  // Prefetch (H2D) sits on the backward critical path -> highest priority.
  TS_CUDA_CHECK(cudaStreamCreateWithPriority(&h2d, cudaStreamNonBlocking, greatest));
  TS_CUDA_CHECK(cudaStreamCreateWithPriority(&d2h, cudaStreamNonBlocking, least));
  TS_CUDA_CHECK(cudaStreamCreateWithPriority(&upd, cudaStreamNonBlocking, least));
}

int resolve_device(int device) {
  if (device >= 0) return device;
  int cur = 0;
  TS_CUDA_CHECK(cudaGetDevice(&cur));
  return cur;
}

Runtime& runtime(int device) {
  const int dev = resolve_device(device);
  TORCH_CHECK(dev >= 0 && dev < kMaxDevices, "invalid CUDA device ", dev);
  Runtime** table = runtime_table();
  if (Runtime* rt = table[dev]) return *rt;
  std::lock_guard<std::mutex> lock(runtime_mutex());
  if (table[dev] == nullptr) table[dev] = new Runtime(dev);
  return *table[dev];
}

std::vector<int> initialized_devices() {
  std::vector<int> out;
  Runtime** table = runtime_table();
  for (int d = 0; d < kMaxDevices; ++d)
    if (table[d] != nullptr) out.push_back(d);
  return out;
}

cudaStream_t compute_stream(int device) {
  return at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device)).stream();
}

void record_stream(const at::Tensor& t, cudaStream_t stream, int device) {
  if (!t.defined() || !t.is_cuda()) return;
  // No-op for memory that did not come from the caching allocator.
  c10::cuda::CUDACachingAllocator::recordStream(
      t.storage().data_ptr(),
      c10::cuda::getStreamFromExternal(stream, static_cast<c10::DeviceIndex>(device)));
}

// ===========================================================================
// Pin-logging caching host allocator
// ===========================================================================
namespace {

class PinnedAllocator {
 public:
  static PinnedAllocator& get() {
    static auto* inst = new PinnedAllocator();
    return *inst;
  }

  void* allocate(size_t bytes, bool mapped, bool cached, const std::string& tag) {
    const size_t size = round_size(std::max<size_t>(bytes, 64));
    std::lock_guard<std::mutex> lock(mu_);
    reclaim_pending_locked();

    if (cached) {
      auto& fl = free_[mapped ? 1 : 0];
      auto it = fl.lower_bound(size);
      // Accept a cached block if it wastes at most 25% (or one 2 MiB page).
      if (it != fl.end() && it->first <= std::max(size + size / 4, size + (2u << 20))) {
        void* ptr = it->second;
        fl.erase(it);
        auto& info = blocks_.at(ptr);
        cached_bytes_ -= info.size;
        info.tag = tag;
        mark_live_locked(ptr, info);
        ++cache_hits_;
        log_locked("hit", info.size, tag);
        return ptr;
      }
      ++cache_misses_;
    }

    void* ptr = nullptr;
    const unsigned flags = mapped ? (cudaHostAllocMapped | cudaHostAllocPortable) : cudaHostAllocPortable;
    cudaError_t err = cudaHostAlloc(&ptr, size, flags);
    if (err != cudaSuccess && cached_bytes_ > 0) {
      // Out of pinnable memory: drop the cache and retry once.
      (void)cudaGetLastError();
      trim_locked(0);
      err = cudaHostAlloc(&ptr, size, flags);
    }
    TORCH_CHECK(err == cudaSuccess, "TrainStream: cudaHostAlloc(", size,
                " bytes) failed: ", cudaGetErrorString(err),
                ". Host RAM may be exhausted or page-locking limits reached.");
    ++host_alloc_calls_;
    reserved_bytes_ += size;
    if (mapped) mapped_bytes_ += size;
    BlockInfo info{size, mapped, cached, tag};
    blocks_.emplace(ptr, info);
    mark_live_locked(ptr, info);
    log_locked("alloc", size, tag);
    return ptr;
  }

  void release(void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = blocks_.find(ptr);
    if (it == blocks_.end()) return;
    BlockInfo& info = it->second;
    live_bytes_ -= info.size;
    --live_blocks_;
    tag_bytes_[info.tag] -= info.size;
    if (!info.cached) {
      log_locked("free", info.size, info.tag);
      free_block_locked(ptr);
      return;
    }
    log_locked("release", info.size, info.tag);
    // The block may still be the source/target of queued DMA on our copy
    // streams. Fence every copy stream and only recycle once all fences pass.
    Pending p{ptr, {}};
    try {
      for (int dev : initialized_devices()) {
        Runtime& rt = runtime(dev);
        for (cudaStream_t s : {rt.d2h, rt.h2d, rt.upd}) {
          auto ev = make_event(dev);
          ev->record(s);
          p.fences.push_back(std::move(ev));
        }
      }
    } catch (...) {
      // Could not create a fence (e.g. the device is out of memory, or CUDA is
      // shutting down). Never recycle unfenced: drain the copy streams instead.
      p.fences.clear();
      (void)cudaGetLastError();
      for (int dev : initialized_devices()) {
        Runtime& rt = runtime(dev);
        cudaStreamSynchronize(rt.d2h);
        cudaStreamSynchronize(rt.h2d);
        cudaStreamSynchronize(rt.upd);
      }
      (void)cudaGetLastError();
    }
    pending_.push_back(std::move(p));
    cached_bytes_ += info.size;
    if (cached_bytes_ > cache_limit_) trim_locked(cache_limit_);
  }

  std::map<std::string, int64_t> stats() {
    std::lock_guard<std::mutex> lock(mu_);
    reclaim_pending_locked();
    return {{"live_bytes", static_cast<int64_t>(live_bytes_)},
            {"peak_live_bytes", static_cast<int64_t>(peak_live_bytes_)},
            {"live_blocks", static_cast<int64_t>(live_blocks_)},
            {"cached_bytes", static_cast<int64_t>(cached_bytes_)},
            {"reserved_bytes", static_cast<int64_t>(reserved_bytes_)},
            {"mapped_bytes", static_cast<int64_t>(mapped_bytes_)},
            {"pending_blocks", static_cast<int64_t>(pending_.size())},
            {"host_alloc_calls", static_cast<int64_t>(host_alloc_calls_)},
            {"host_free_calls", static_cast<int64_t>(host_free_calls_)},
            {"cache_hits", static_cast<int64_t>(cache_hits_)},
            {"cache_misses", static_cast<int64_t>(cache_misses_)},
            {"cache_limit", static_cast<int64_t>(cache_limit_)}};
  }

  std::map<std::string, int64_t> tag_bytes() {
    std::lock_guard<std::mutex> lock(mu_);
    std::map<std::string, int64_t> out;
    for (const auto& kv : tag_bytes_)
      if (kv.second != 0) out[kv.first] = static_cast<int64_t>(kv.second);
    return out;
  }

  std::vector<std::tuple<double, std::string, int64_t, std::string>> log(int64_t limit) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::tuple<double, std::string, int64_t, std::string>> out;
    const int64_t n = static_cast<int64_t>(log_.size());
    const int64_t start = limit > 0 && limit < n ? n - limit : 0;
    for (int64_t i = start; i < n; ++i) {
      const auto& e = log_[static_cast<size_t>(i)];
      out.emplace_back(e.t_ms, e.op, static_cast<int64_t>(e.bytes), e.tag);
    }
    return out;
  }

  void empty_cache() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& p : pending_)
      for (auto& f : p.fences) f->synchronize();
    reclaim_pending_locked();
    trim_locked(0);
  }

  void set_limit(int64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_limit_ = static_cast<size_t>(std::max<int64_t>(bytes, 0));
    trim_locked(cache_limit_);
  }

 private:
  struct BlockInfo {
    size_t size;
    bool mapped;
    bool cached;
    std::string tag;
  };
  struct Pending {
    void* ptr;
    std::vector<EventPtr> fences;
  };
  struct LogEntry {
    double t_ms;
    const char* op;
    size_t bytes;
    std::string tag;
  };

  static size_t round_size(size_t n) {
    constexpr size_t kSmall = 4096, kLarge = 2u << 20;
    if (n < (1u << 20)) return (n + kSmall - 1) / kSmall * kSmall;
    return (n + kLarge - 1) / kLarge * kLarge;
  }

  void mark_live_locked(void* ptr, const BlockInfo& info) {
    (void)ptr;
    live_bytes_ += info.size;
    ++live_blocks_;
    peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
    tag_bytes_[info.tag] += info.size;
  }

  void reclaim_pending_locked() {
    for (auto it = pending_.begin(); it != pending_.end();) {
      bool done = true;
      for (auto& f : it->fences) {
        if (!f->query()) {
          done = false;
          break;
        }
      }
      if (!done) {
        ++it;
        continue;
      }
      const BlockInfo& info = blocks_.at(it->ptr);
      free_[info.mapped ? 1 : 0].emplace(info.size, it->ptr);
      it = pending_.erase(it);
    }
  }

  // Release free (not pending) cached blocks until cached_bytes_ <= target.
  void trim_locked(size_t target) {
    for (int m = 0; m < 2 && cached_bytes_ > target; ++m) {
      auto& fl = free_[m];
      // Largest first: fewest cudaFreeHost calls to get under the limit.
      while (!fl.empty() && cached_bytes_ > target) {
        auto it = std::prev(fl.end());
        void* ptr = it->second;
        fl.erase(it);
        cached_bytes_ -= blocks_.at(ptr).size;
        free_block_locked(ptr);
      }
    }
  }

  void free_block_locked(void* ptr) {
    auto it = blocks_.find(ptr);
    const size_t size = it->second.size;
    reserved_bytes_ -= size;
    if (it->second.mapped) mapped_bytes_ -= size;
    blocks_.erase(it);
    cudaFreeHost(ptr);
    (void)cudaGetLastError();
    ++host_free_calls_;
  }

  void log_locked(const char* op, size_t bytes, const std::string& tag) {
    log_.push_back({now_ms(), op, bytes, tag});
    if (log_.size() > 4096) log_.pop_front();
  }

  std::mutex mu_;
  std::unordered_map<void*, BlockInfo> blocks_;
  std::multimap<size_t, void*> free_[2];
  std::deque<Pending> pending_;
  std::deque<LogEntry> log_;
  std::map<std::string, int64_t> tag_bytes_;
  size_t live_bytes_ = 0, peak_live_bytes_ = 0, live_blocks_ = 0;
  size_t cached_bytes_ = 0, reserved_bytes_ = 0, mapped_bytes_ = 0;
  size_t cache_limit_ = size_t(8) << 30;
  uint64_t host_alloc_calls_ = 0, host_free_calls_ = 0, cache_hits_ = 0, cache_misses_ = 0;
};

}  // namespace

at::Tensor pinned_empty(const std::vector<int64_t>& sizes, at::ScalarType dtype,
                        const std::string& tag, bool mapped, bool cached) {
  int64_t numel = 1;
  for (int64_t s : sizes) {
    TORCH_CHECK(s >= 0, "pinned_empty: negative dimension");
    numel *= s;
  }
  const size_t bytes = static_cast<size_t>(numel) * at::elementSize(dtype);
  // Without torch's CUDA lazy-init, Tensor.is_pinned() reports false for our blocks.
  at::globalContext().lazyInitDevice(at::kCUDA);
  void* ptr = PinnedAllocator::get().allocate(bytes, mapped, cached, tag);
  return at::from_blob(
      ptr, sizes, [](void* p) { PinnedAllocator::get().release(p); },
      at::TensorOptions().dtype(dtype).device(at::kCPU));
}

std::map<std::string, int64_t> pinned_stats() { return PinnedAllocator::get().stats(); }
std::map<std::string, int64_t> pinned_tag_bytes() { return PinnedAllocator::get().tag_bytes(); }
std::vector<std::tuple<double, std::string, int64_t, std::string>> pinned_log(int64_t limit) {
  return PinnedAllocator::get().log(limit);
}
void pinned_empty_cache() { PinnedAllocator::get().empty_cache(); }
void pinned_set_cache_limit(int64_t bytes) { PinnedAllocator::get().set_limit(bytes); }

void* pinned_device_pointer(const at::Tensor& t) {
  void* dptr = nullptr;
  TS_CUDA_CHECK(cudaHostGetDevicePointer(&dptr, t.data_ptr(), 0));
  return dptr;
}

// ===========================================================================
// Asynchronous transfers
// ===========================================================================
namespace {
// Copies move raw bytes, so both sides only need to be non-overlapping and
// dense: their elements then occupy exactly numel * itemsize bytes starting at
// data_ptr() (PyTorch strides are never negative). This lets a permuted layout
// (e.g. attention outputs in (B,T,H,D) memory order) round-trip with its exact
// strides instead of being forced to contiguous.
void check_pair(const at::Tensor& dst, const at::Tensor& src, const char* what) {
  TORCH_CHECK(dst.is_non_overlapping_and_dense() && src.is_non_overlapping_and_dense(), what,
              ": tensors must be non-overlapping and dense");
  TORCH_CHECK(dst.scalar_type() == src.scalar_type(), what, ": dtype mismatch (",
              dst.scalar_type(), " vs ", src.scalar_type(), ")");
  TORCH_CHECK(dst.numel() == src.numel(), what, ": numel mismatch (", dst.numel(), " vs ",
              src.numel(), ")");
}
}  // namespace

EventPtr d2h_async(const at::Tensor& dst, const at::Tensor& src) {
  TORCH_CHECK(src.is_cuda(), "d2h_async: src must be a CUDA tensor");
  TORCH_CHECK(dst.device().is_cpu(), "d2h_async: dst must be a (pinned) CPU tensor");
  check_pair(dst, src, "d2h_async");
  const int dev = src.get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));

  // Producer -> copy-stream dependency, expressed on the GPU (no host stall).
  auto produced = make_event(dev);
  produced->record(compute_stream(dev));
  produced->block(rt.d2h);

  const size_t bytes = src.nbytes();
  if (bytes > 0)
    TS_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDeviceToHost, rt.d2h));
  // No recordStream: tagging the block makes the allocator create CUDA events
  // inside its free path, which can fail (and abort) on a full device. The
  // caller keeps `src` alive until the returned event has completed instead.
  auto done = make_event(dev);
  done->record(rt.d2h);
  rt.bytes_d2h += bytes;
  rt.copies_d2h += 1;
  return done;
}

EventPtr h2d_async(const at::Tensor& dst, const at::Tensor& src, const EventPtr& after) {
  TORCH_CHECK(dst.is_cuda(), "h2d_async: dst must be a CUDA tensor");
  TORCH_CHECK(src.device().is_cpu(), "h2d_async: src must be a (pinned) CPU tensor");
  check_pair(dst, src, "h2d_async");
  const int dev = dst.get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  if (after) after->block(rt.h2d);
  const size_t bytes = src.nbytes();
  if (bytes > 0)
    TS_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes, cudaMemcpyHostToDevice, rt.h2d));
  auto done = make_event(dev);
  done->record(rt.h2d);
  rt.bytes_h2d += bytes;
  rt.copies_h2d += 1;
  return done;
}

std::tuple<at::Tensor, EventPtr> h2d_alloc_async(const at::Tensor& src, const EventPtr& after,
                                                 int device, const std::vector<int64_t>& sizes,
                                                 const std::vector<int64_t>& strides) {
  TORCH_CHECK(src.device().is_cpu() && src.is_contiguous(),
              "h2d_alloc_async: src must be a contiguous (pinned) CPU tensor");
  TORCH_CHECK(sizes.size() == strides.size(), "h2d_alloc_async: sizes and strides must be given together");
  const int dev = resolve_device(device);
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  const cudaStream_t consumer = compute_stream(dev);

  at::Tensor dst;
  {
    // Allocate from the consumer's (compute stream's) pool so paging never
    // splits VRAM into a second per-stream pool. The block's previous users
    // are all queued before `freed`, so the copy waits only for those.
    const auto opts = at::TensorOptions()
                          .dtype(src.scalar_type())
                          .device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(dev)));
    dst = sizes.empty() && src.dim() > 0 ? at::empty(src.sizes(), opts) : at::empty_strided(sizes, strides, opts);
  }
  TORCH_CHECK(dst.numel() == src.numel() && dst.is_non_overlapping_and_dense(),
              "h2d_alloc_async: target layout must be dense with the same numel as the host buffer");
  auto freed = make_event(dev);
  freed->record(consumer);
  freed->block(rt.h2d);
  if (after) after->block(rt.h2d);
  const size_t bytes = src.nbytes();
  if (bytes > 0)
    TS_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes, cudaMemcpyHostToDevice, rt.h2d));
  // The consumer must wait on the returned event before using *or dropping*
  // `dst` (TrainStream's activation pager does both).
  auto done = make_event(dev);
  done->record(rt.h2d);
  rt.bytes_h2d += bytes;
  rt.copies_h2d += 1;
  return {dst, done};
}

std::map<std::string, int64_t> transfer_stats(int device) {
  Runtime& rt = runtime(device);
  return {{"bytes_d2h", static_cast<int64_t>(rt.bytes_d2h.load())},
          {"bytes_h2d", static_cast<int64_t>(rt.bytes_h2d.load())},
          {"copies_d2h", static_cast<int64_t>(rt.copies_d2h.load())},
          {"copies_h2d", static_cast<int64_t>(rt.copies_h2d.load())}};
}

// ===========================================================================
// Micro-benchmarks
// ===========================================================================
namespace {
double timed_copies(int dev, cudaStream_t s, void* dst, const void* src, size_t bytes,
                    cudaMemcpyKind kind, int iters) {
  Event start(dev, true), stop(dev, true);
  TS_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s));  // warm-up (TLB, DMA setup)
  start.record(s);
  for (int i = 0; i < iters; ++i) TS_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s));
  stop.record(s);
  return static_cast<double>(start.elapsed_ms(stop));
}
double gbps(size_t bytes, int iters, double ms) {
  return ms <= 0 ? 0.0 : static_cast<double>(bytes) * iters / (ms * 1e-3) / 1e9;
}
}  // namespace

std::map<std::string, double> benchmark_pcie(int device, int64_t bytes_i, int iters) {
  const int dev = resolve_device(device);
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  const size_t bytes = static_cast<size_t>(std::max<int64_t>(bytes_i, 1 << 20));
  iters = std::max(iters, 1);

  void* host = nullptr;
  void* host2 = nullptr;
  TS_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocPortable));
  TS_CUDA_CHECK(cudaHostAlloc(&host2, bytes, cudaHostAllocPortable));
  void* dev_a = nullptr;
  void* dev_b = nullptr;
  TS_CUDA_CHECK(cudaMalloc(&dev_a, bytes));
  TS_CUDA_CHECK(cudaMalloc(&dev_b, bytes));
  std::memset(host, 1, bytes);
  std::memset(host2, 2, bytes);

  std::map<std::string, double> out;
  const double h2d_ms = timed_copies(dev, rt.h2d, dev_a, host, bytes, cudaMemcpyHostToDevice, iters);
  const double d2h_ms = timed_copies(dev, rt.d2h, host, dev_a, bytes, cudaMemcpyDeviceToHost, iters);
  out["h2d_gbps"] = gbps(bytes, iters, h2d_ms);
  out["d2h_gbps"] = gbps(bytes, iters, d2h_ms);

  // Full duplex: both copy engines busy at once.
  {
    Event start(dev, true), end_d2h(dev, true), stop(dev, true);
    start.record(rt.h2d);
    start.block(rt.d2h);
    for (int i = 0; i < iters; ++i) {
      TS_CUDA_CHECK(cudaMemcpyAsync(dev_a, host, bytes, cudaMemcpyHostToDevice, rt.h2d));
      TS_CUDA_CHECK(cudaMemcpyAsync(host2, dev_b, bytes, cudaMemcpyDeviceToHost, rt.d2h));
    }
    end_d2h.record(rt.d2h);
    end_d2h.block(rt.h2d);
    stop.record(rt.h2d);
    out["bidir_gbps"] = gbps(bytes * 2, iters, start.elapsed_ms(stop));
  }

  // Pageable baseline: the driver must bounce through its own staging buffer.
  {
    void* pageable = std::malloc(bytes);
    TORCH_CHECK(pageable != nullptr, "benchmark_pcie: malloc failed");
    std::memset(pageable, 3, bytes);
    const int it = std::max(1, iters / 2);
    const double ms = timed_copies(dev, rt.h2d, dev_a, pageable, bytes, cudaMemcpyHostToDevice, it);
    out["pageable_h2d_gbps"] = gbps(bytes, it, ms);
    std::free(pageable);
  }

  // Transfer-size sweep: shows where per-DMA setup overhead stops dominating.
  for (size_t sz = 64u << 10; sz <= std::min(bytes, size_t(64) << 20); sz *= 4) {
    const int it = static_cast<int>(std::max<size_t>(4, std::min<size_t>(256, (size_t(256) << 20) / sz)));
    out["sweep_h2d_" + std::to_string(sz)] =
        gbps(sz, it, timed_copies(dev, rt.h2d, dev_a, host, sz, cudaMemcpyHostToDevice, it));
    out["sweep_d2h_" + std::to_string(sz)] =
        gbps(sz, it, timed_copies(dev, rt.d2h, host, dev_a, sz, cudaMemcpyDeviceToHost, it));
  }

  // Small-transfer latency (4 KiB) in microseconds.
  {
    const int it = 200;
    const double ms = timed_copies(dev, rt.h2d, dev_a, host, 4096, cudaMemcpyHostToDevice, it);
    out["latency_us_4k"] = ms * 1e3 / it;
  }

  TS_CUDA_CHECK(cudaStreamSynchronize(rt.h2d));
  TS_CUDA_CHECK(cudaStreamSynchronize(rt.d2h));
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFreeHost(host);
  cudaFreeHost(host2);
  out["bytes"] = static_cast<double>(bytes);
  return out;
}

std::map<std::string, double> benchmark_matmul(int device, int64_t n, at::ScalarType dtype,
                                               int iters) {
  const int dev = resolve_device(device);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  auto opts = at::TensorOptions().dtype(dtype).device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(dev)));
  at::Tensor a = at::randn({n, n}, opts);
  at::Tensor b = at::randn({n, n}, opts);
  at::Tensor c = at::mm(a, b);  // warm-up: cuBLAS handle + heuristics
  c = at::mm(a, b);
  const cudaStream_t s = compute_stream(dev);
  Event start(dev, true), stop(dev, true);
  iters = std::max(iters, 1);
  start.record(s);
  for (int i = 0; i < iters; ++i) at::mm_out(c, a, b);
  stop.record(s);
  const double ms = start.elapsed_ms(stop) / iters;
  const double flops = 2.0 * static_cast<double>(n) * n * n;
  return {{"ms_per_matmul", ms}, {"tflops", flops / (ms * 1e-3) / 1e12}, {"n", static_cast<double>(n)}};
}

}  // namespace trainstream
