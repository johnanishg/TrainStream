// TrainStream — runtime core: CUDA event pool, dedicated copy streams, the
// pin-logging caching host allocator, asynchronous DMA helpers and the
// CUDA-event-timed micro-benchmarks used by the hardware profiler.

#include "trainstream.h"

#include <ATen/Parallel.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
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
  // How a host thread waits on it: see event_wait_mode(). Only "block" asks for an event the GPU
  // signals by interrupt; thousands of those per step held this laptop's GPU near half its power.
  TS_CUDA_CHECK(cudaEventCreateWithFlags(&ev_, (timing_ ? cudaEventDefault : cudaEventDisableTiming) |
                                                    (event_wait_mode() == 1 ? cudaEventBlockingSync : 0u)));
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

// How host threads wait for an event (TRAINSTREAM_EVENT_WAIT):
//   "poll" (default): check the event for up to TRAINSTREAM_EVENT_SPIN_US (50), then check it again
//       after sleeping TRAINSTREAM_EVENT_SLEEP_US (100) at a time. The CPU optimizer's worker threads
//       wait for gradient copies thousands of times a step; spinning, they kept ~6 cores at 100 %.
//   "spin": CUDA's default, a core at 100 % until the event completes; the lowest latency.
//   "block": an interrupt-signalled event the thread sleeps on. No CPU, but on an RTX 5090 Laptop
//       (Windows) the GPU then ran at ~70 W instead of ~148 W and the 14B LoRA step took 1.8x longer.
int event_wait_mode() {
  static const int mode = [] {
    const char* v = std::getenv("TRAINSTREAM_EVENT_WAIT");
    const std::string s(v != nullptr ? v : "");
    return s == "spin" ? 0 : s == "block" ? 1 : 2;
  }();
  return mode;
}

static int64_t env_us(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  return v != nullptr ? std::max<int64_t>(0, std::atoll(v)) : fallback;
}

static void wait_event(cudaEvent_t ev) {
  if (event_wait_mode() != 2) {
    TS_CUDA_CHECK(cudaEventSynchronize(ev));
    return;
  }
  static const int64_t spin_us = env_us("TRAINSTREAM_EVENT_SPIN_US", 50);
  static const int64_t sleep_us = env_us("TRAINSTREAM_EVENT_SLEEP_US", 100);
  const auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
  for (;;) {
    const cudaError_t err = cudaEventQuery(ev);
    if (err == cudaSuccess) return;
    if (err != cudaErrorNotReady) TS_CUDA_CHECK(err);
    (void)cudaGetLastError();  // clear the sticky "not ready" status
    if (std::chrono::steady_clock::now() >= spin_until)
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
  }
}

void Event::synchronize() const {
  if (!recorded_) return;
  wait_event(ev_);
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

// Page-locked slabs carved into blocks. Every cudaHostAlloc page-locks fresh
// pages through the OS (a stall under Windows WDDM, and a cost that grows with
// the size); with the arena reserved up front, a training step that pages
// activations of any size hands out ranges of memory that is already locked.
// Best fit over all free ranges; neighbours coalesce on return.
class HostArena {
 public:
  void* take(size_t size) {
    auto it = by_size_.lower_bound(size);
    if (it == by_size_.end()) return nullptr;
    char* p = it->second;
    const size_t len = it->first;
    by_size_.erase(it);
    Slab& s = slab_of(p);
    s.free.erase(p);
    if (len > size) insert_free(s, p + size, len - size);
    used_ += size;
    return p;
  }

  void give(void* ptr, size_t size) {
    char* p = static_cast<char*>(ptr);
    Slab& s = slab_of(p);
    used_ -= size;
    auto next = s.free.find(p + size);
    if (next != s.free.end()) {
      size += next->second;
      erase_size(next->first, next->second);
      s.free.erase(next);
    }
    auto prev = s.free.lower_bound(p);
    if (prev != s.free.begin()) {
      --prev;
      if (prev->first + prev->second == p) {
        p = prev->first;
        size += prev->second;
        erase_size(prev->first, prev->second);
        s.free.erase(prev);
      }
    }
    insert_free(s, p, size);
  }

  void add_slab(void* base, size_t size) {
    char* b = static_cast<char*>(base);
    Slab& s = slabs_[b];
    s.size = size;
    insert_free(s, b, size);
    bytes_ += size;
  }

  // Detach every slab nothing is allocated from; the caller frees them.
  std::vector<std::pair<void*, size_t>> drop_empty() {
    std::vector<std::pair<void*, size_t>> out;
    for (auto it = slabs_.begin(); it != slabs_.end();) {
      Slab& s = it->second;
      if (s.free.size() == 1 && s.free.begin()->first == it->first && s.free.begin()->second == s.size) {
        erase_size(it->first, s.size);
        bytes_ -= s.size;
        out.emplace_back(it->first, s.size);
        it = slabs_.erase(it);
      } else {
        ++it;
      }
    }
    return out;
  }

  size_t bytes() const { return bytes_; }
  size_t used() const { return used_; }
  size_t slabs() const { return slabs_.size(); }
  size_t largest_free() const { return by_size_.empty() ? 0 : by_size_.rbegin()->first; }

 private:
  struct Slab {
    size_t size = 0;
    std::map<char*, size_t> free;  // offset-ordered free ranges, for coalescing
  };

  Slab& slab_of(char* p) {
    auto it = slabs_.upper_bound(p);
    --it;
    return it->second;
  }
  void insert_free(Slab& s, char* p, size_t len) {
    s.free.emplace(p, len);
    by_size_.emplace(len, p);
  }
  void erase_size(char* p, size_t len) {
    auto range = by_size_.equal_range(len);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == p) {
        by_size_.erase(it);
        return;
      }
    }
  }

  std::map<char*, Slab> slabs_;
  std::multimap<size_t, char*> by_size_;
  size_t bytes_ = 0, used_ = 0;
};

size_t env_bytes(const char* name, size_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const double x = std::strtod(v, &end);
  if (end == v || x <= 0) return fallback;
  double mult = 1;
  switch (*end) {
    case 'k': case 'K': mult = 1024.0; break;
    case 'm': case 'M': mult = 1024.0 * 1024; break;
    case 'g': case 'G': mult = 1024.0 * 1024 * 1024; break;
    default: break;
  }
  return static_cast<size_t>(x * mult);
}

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
    if (!mapped && arena_cap_ > 0) {
      if (void* ptr = arena_take_locked(size)) return place_arena_locked(ptr, size, cached, tag);
      ++arena_fallbacks_;
    }

    void* ptr = nullptr;
    const unsigned flags = mapped ? (cudaHostAllocMapped | cudaHostAllocPortable) : cudaHostAllocPortable;
    cudaError_t err = cudaHostAlloc(&ptr, size, flags);
    if (err != cudaSuccess && (cached_bytes_ > 0 || arena_.bytes() > 0)) {
      // Out of pinnable memory: drop the cache and the empty slabs, retry once.
      (void)cudaGetLastError();
      trim_locked(0);
      drop_empty_slabs_locked();
      err = cudaHostAlloc(&ptr, size, flags);
    }
    TORCH_CHECK(err == cudaSuccess, "TrainStream: cudaHostAlloc(", size,
                " bytes) failed: ", cudaGetErrorString(err),
                ". Host RAM may be exhausted or page-locking limits reached.");
    ++host_alloc_calls_;
    reserved_bytes_ += size;
    if (mapped) mapped_bytes_ += size;
    BlockInfo info{size, mapped, cached, tag, false};
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
    if (!info.cached && !info.arena) {
      log_locked("free", info.size, info.tag);
      free_block_locked(ptr);
      return;
    }
    log_locked("release", info.size, info.tag);
    // The block may still be the source/target of queued DMA on our copy
    // streams. Fence every copy stream and only recycle once all fences pass.
    // An arena block also fences the compute stream: unlike cudaFreeHost,
    // returning it does not wait for a non_blocking copy torch queued there.
    Pending p{ptr, {}};
    try {
      for (int dev : initialized_devices()) {
        Runtime& rt = runtime(dev);
        std::vector<cudaStream_t> streams{rt.d2h, rt.h2d, rt.upd};
        if (info.arena) streams.push_back(compute_stream(dev));
        for (cudaStream_t s : streams) {
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
        if (info.arena) cudaStreamSynchronize(compute_stream(dev));
      }
      (void)cudaGetLastError();
    }
    pending_.push_back(std::move(p));
    if (!info.cached) return;  // an arena block: back to the arena once fenced
    cached_bytes_ += info.size;
    if (cached_bytes_ > cache_limit_) trim_locked(cache_limit_);
  }

  // Grow the arena to at least `bytes` now (slab by slab, stopping at the
  // first slab the OS refuses) and let it grow on demand up to `cap`.
  int64_t reserve(int64_t bytes, int64_t slab, int64_t cap) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slab > 0) slab_bytes_ = round_size(static_cast<size_t>(slab));
    const size_t want = static_cast<size_t>(std::max<int64_t>(bytes, 0));
    arena_cap_ = std::max({arena_cap_, want, static_cast<size_t>(std::max<int64_t>(cap, 0))});
    grow_refused_ = false;
    while (arena_.bytes() < want) {
      const size_t left = round_size(want - arena_.bytes());
      if (!grow_locked(std::min(slab_bytes_, left))) break;
    }
    return static_cast<int64_t>(arena_.bytes());
  }

  void arena_trim() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& p : pending_)
      for (auto& f : p.fences) f->synchronize();
    reclaim_pending_locked();
    drop_empty_slabs_locked();
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
            {"cache_limit", static_cast<int64_t>(cache_limit_)},
            {"arena_bytes", static_cast<int64_t>(arena_.bytes())},
            {"arena_used", static_cast<int64_t>(arena_.used())},
            {"arena_slabs", static_cast<int64_t>(arena_.slabs())},
            {"arena_cap", static_cast<int64_t>(arena_cap_)},
            {"arena_fallbacks", static_cast<int64_t>(arena_fallbacks_)},
            {"arena_largest_free", static_cast<int64_t>(arena_.largest_free())}};
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
    bool arena;  // a range of an arena slab, not its own cudaHostAlloc
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
      if (info.cached)  // the size-bucket cache sits on top of the arena too
        free_[info.mapped ? 1 : 0].emplace(info.size, it->ptr);
      else
        free_block_locked(it->ptr);
      it = pending_.erase(it);
    }
  }

  void* arena_take_locked(size_t size) {
    if (void* ptr = arena_.take(size)) return ptr;
    // Grow by a slab (or by the request, if larger) while the cap allows. After
    // the OS refuses a slab, stop trying (until the next reserve()), instead of
    // paying a failed page-locking call before every fallback allocation.
    const size_t slab = std::max(slab_bytes_, size);
    if (grow_refused_ || arena_.bytes() + slab > arena_cap_) return nullptr;
    if (!grow_locked(slab)) {
      grow_refused_ = true;
      return nullptr;
    }
    return arena_.take(size);
  }

  void* place_arena_locked(void* ptr, size_t size, bool cached, const std::string& tag) {
    BlockInfo info{size, false, cached, tag, true};
    blocks_.emplace(ptr, info);
    mark_live_locked(ptr, info);
    log_locked("arena", size, tag);
    return ptr;
  }

  bool grow_locked(size_t size) {
    void* base = nullptr;
    if (cudaHostAlloc(&base, size, cudaHostAllocPortable) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    ++host_alloc_calls_;
    reserved_bytes_ += size;
    arena_.add_slab(base, size);
    log_locked("slab", size, "arena");
    return true;
  }

  void drop_empty_slabs_locked() {
    for (auto& slab : arena_.drop_empty()) {
      cudaFreeHost(slab.first);
      (void)cudaGetLastError();
      reserved_bytes_ -= slab.second;
      ++host_free_calls_;
      log_locked("free", slab.second, "arena");
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
    if (it->second.arena) {
      blocks_.erase(it);
      arena_.give(ptr, size);
      return;
    }
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
  HostArena arena_;
  size_t arena_cap_ = 0;  // 0: no arena (every block is its own cudaHostAlloc)
  size_t slab_bytes_ = round_size(env_bytes("TRAINSTREAM_HOST_ARENA_SLAB", size_t(1) << 30));
  uint64_t arena_fallbacks_ = 0;
  bool grow_refused_ = false;
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
int64_t pinned_reserve(int64_t bytes, int64_t slab, int64_t cap) {
  return PinnedAllocator::get().reserve(bytes, slab, cap);
}
void pinned_arena_trim() { PinnedAllocator::get().arena_trim(); }

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

// ---------------------------------------------------------------------------
// Pageable host memory -> device through pinned staging slots
// ---------------------------------------------------------------------------
// cudaMemcpyAsync from pageable memory is not asynchronous: the driver copies
// through a small staging buffer of its own while the host waits (and, under
// WDDM, may fault the pages in first). Here one call stages a whole group of
// tensors (a block's weights) into a pinned slot with all CPU cores, then
// queues ordinary DMA from it. The slots form a ring; the host waits for a
// slot's previous DMA only when the ring wraps, i.e. when it runs `slots`
// groups ahead of the GPU's copies.
namespace {
struct BounceRing {
  std::mutex mu;
  std::vector<at::Tensor> slots;
  std::vector<EventPtr> done;
  size_t next = 0;
};
BounceRing& bounce_ring(int dev) {
  static auto* rings = new BounceRing[kMaxDevices];
  return rings[dev];
}
constexpr size_t kBounceAlign = 256;

void parallel_memcpy(char* dst, const char* src, size_t bytes) {
  constexpr size_t kChunk = size_t(8) << 20;
  const int64_t chunks = static_cast<int64_t>((bytes + kChunk - 1) / kChunk);
  at::parallel_for(0, chunks, 1, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      const size_t off = static_cast<size_t>(c) * kChunk;
      std::memcpy(dst + off, src + off, std::min(kChunk, bytes - off));
    }
  });
}
}  // namespace

EventPtr h2d_bounce_async(const std::vector<at::Tensor>& dsts, const std::vector<at::Tensor>& srcs,
                          const EventPtr& after, int64_t slots) {
  TORCH_CHECK(!dsts.empty() && dsts.size() == srcs.size(), "h2d_bounce_async: need matching dst/src lists");
  size_t total = 0;
  for (size_t i = 0; i < dsts.size(); ++i) {
    TORCH_CHECK(dsts[i].is_cuda(), "h2d_bounce_async: dst must be a CUDA tensor");
    TORCH_CHECK(srcs[i].device().is_cpu(), "h2d_bounce_async: src must be a CPU tensor");
    TORCH_CHECK(dsts[i].get_device() == dsts[0].get_device(), "h2d_bounce_async: one device per call");
    check_pair(dsts[i], srcs[i], "h2d_bounce_async");
    total += (srcs[i].nbytes() + kBounceAlign - 1) / kBounceAlign * kBounceAlign;
  }
  const int dev = dsts[0].get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  BounceRing& ring = bounce_ring(dev);
  std::lock_guard<std::mutex> lock(ring.mu);
  const size_t n = static_cast<size_t>(std::max<int64_t>(slots, 2));
  if (ring.slots.size() < n) {
    ring.slots.resize(n);
    ring.done.resize(n);
  }
  const size_t k = ring.next++ % ring.slots.size();
  if (ring.done[k]) ring.done[k]->synchronize();  // the GPU has read the slot's previous contents
  if (!ring.slots[k].defined() || ring.slots[k].nbytes() < total) {
    ring.slots[k] = at::Tensor();  // released (fenced) before the bigger one is taken
    ring.slots[k] = pinned_empty({static_cast<int64_t>(total)}, at::kByte, "bounce", false, false);
  }
  char* base = static_cast<char*>(ring.slots[k].data_ptr());
  size_t off = 0;
  for (const auto& s : srcs) {
    parallel_memcpy(base + off, static_cast<const char*>(s.data_ptr()), s.nbytes());
    off += (s.nbytes() + kBounceAlign - 1) / kBounceAlign * kBounceAlign;
  }
  if (after) after->block(rt.h2d);
  off = 0;
  for (size_t i = 0; i < dsts.size(); ++i) {
    const size_t bytes = srcs[i].nbytes();
    if (bytes > 0)
      TS_CUDA_CHECK(cudaMemcpyAsync(dsts[i].data_ptr(), base + off, bytes, cudaMemcpyHostToDevice, rt.h2d));
    off += (bytes + kBounceAlign - 1) / kBounceAlign * kBounceAlign;
    rt.bytes_h2d += bytes;
    rt.copies_h2d += 1;
  }
  auto done = make_event(dev);
  done->record(rt.h2d);
  ring.done[k] = done;
  return done;
}

EventPtr h2d_pages_async(const at::Tensor& dst, const std::vector<at::Tensor>& pages, const EventPtr& after) {
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "h2d_pages_async: dst must be a contiguous CUDA tensor");
  size_t total = 0;
  for (const auto& p : pages) {
    TORCH_CHECK(p.device().is_cpu() && p.is_contiguous() && p.scalar_type() == dst.scalar_type(),
                "h2d_pages_async: pages must be contiguous CPU tensors of dst's dtype");
    total += p.nbytes();
  }
  TORCH_CHECK(total <= dst.nbytes(), "h2d_pages_async: ", total, " bytes of pages do not fit in ", dst.nbytes());
  const int dev = dst.get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  if (after) after->block(rt.h2d);
  char* d = static_cast<char*>(dst.data_ptr());
  for (const auto& p : pages) {
    const size_t bytes = p.nbytes();
    if (bytes > 0) TS_CUDA_CHECK(cudaMemcpyAsync(d, p.data_ptr(), bytes, cudaMemcpyHostToDevice, rt.h2d));
    d += bytes;
    rt.bytes_h2d += bytes;
    rt.copies_h2d += 1;
  }
  auto done = make_event(dev);
  done->record(rt.h2d);
  return done;
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
