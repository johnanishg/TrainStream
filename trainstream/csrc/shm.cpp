// TrainStream — host memory shared by the training processes of one machine.
//
// Data-parallel ranks on one machine hold identical frozen weights and reduce
// their gradients every step. Two primitives let them do that through RAM
// instead of per-process copies and loopback sockets:
//
//   shared_host(name, bytes, create)  a named shared-memory segment (Windows
//       file mapping / POSIX shm), page-locked with cudaHostRegister so DMA
//       to and from it runs at pinned speed. One process creates it, the
//       others open it by name; it lives until the last process unmaps it.
//   ShmReducer                        an all-reduce (sum) of float32 buffers
//       through such a segment. Every rank copies its buffer into its own
//       area, then sums all areas in rank order 0..N-1 into its buffer, so
//       every rank gets bit-identical results. Sequence numbers in the
//       segment header synchronise the ranks (atomics on shared memory);
//       two alternating areas let a fast rank start the next reduction while
//       slower ranks still read the previous one.

#include "trainstream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trainstream {

namespace {

struct Mapping {
  void* base = nullptr;
  size_t bytes = 0;
  bool registered = false;
#ifdef _WIN32
  HANDLE handle = nullptr;
#else
  int fd = -1;
  std::string name;
  bool owner = false;
#endif
};

void release_mapping(Mapping* m) {
  if (m->registered) cudaHostUnregister(m->base);  // errors at teardown are not actionable
#ifdef _WIN32
  if (m->base) UnmapViewOfFile(m->base);
  if (m->handle) CloseHandle(m->handle);
#else
  if (m->base) munmap(m->base, m->bytes);
  if (m->fd >= 0) close(m->fd);
  if (m->owner) shm_unlink(m->name.c_str());
#endif
  delete m;
}

Mapping* map_segment(const std::string& name, size_t bytes, bool create) {
  auto* m = new Mapping();
  m->bytes = bytes;
#ifdef _WIN32
  const std::string full = "Local\\" + name;
  if (create) {
    m->handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                   static_cast<DWORD>(static_cast<uint64_t>(bytes) >> 32),
                                   static_cast<DWORD>(bytes & 0xffffffffu), full.c_str());
  } else {
    m->handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, full.c_str());
  }
  if (!m->handle) {
    const DWORD err = GetLastError();
    delete m;
    TORCH_CHECK(false, "shared_host: cannot ", create ? "create" : "open", " segment '", name, "' (error ", err, ")");
  }
  m->base = MapViewOfFile(m->handle, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (!m->base) {
    const DWORD err = GetLastError();
    CloseHandle(m->handle);
    delete m;
    TORCH_CHECK(false, "shared_host: cannot map segment '", name, "' (error ", err, ")");
  }
#else
  m->name = "/" + name;
  m->owner = create;
  m->fd = shm_open(m->name.c_str(), create ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR, 0600);
  if (m->fd < 0) {
    delete m;
    TORCH_CHECK(false, "shared_host: cannot ", create ? "create" : "open", " segment '", name, "'");
  }
  if (create && ftruncate(m->fd, static_cast<off_t>(bytes)) != 0) {
    close(m->fd);
    shm_unlink(m->name.c_str());
    delete m;
    TORCH_CHECK(false, "shared_host: cannot size segment '", name, "'");
  }
  m->base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m->fd, 0);
  if (m->base == MAP_FAILED) {
    m->base = nullptr;
    release_mapping(m);
    TORCH_CHECK(false, "shared_host: cannot map segment '", name, "'");
  }
#endif
  return m;
}

}  // namespace

at::Tensor shared_host(const std::string& name, int64_t bytes, bool create, bool pin) {
  TORCH_CHECK(bytes > 0, "shared_host: size must be positive");
  TORCH_CHECK(!name.empty() && name.find('/') == std::string::npos && name.find('\\') == std::string::npos,
              "shared_host: the name must be a plain identifier");
  Mapping* m = map_segment(name, static_cast<size_t>(bytes), create);
  if (create) std::memset(m->base, 0, static_cast<size_t>(bytes));
  if (pin) {
    // Portable: usable for DMA from every CUDA context in this process.
    if (cudaHostRegister(m->base, static_cast<size_t>(bytes), cudaHostRegisterPortable) == cudaSuccess) {
      m->registered = true;
    } else {
      cudaGetLastError();  // over the page-lock limit: still correct, copies go through staging
    }
  }
  return at::from_blob(
      m->base, {bytes}, [m](void*) { release_mapping(m); },
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
}

// ---------------------------------------------------------------------------
// All-reduce through a shared segment.
//
// Layout: [header 4 KiB][area 0: world x capacity floats][area 1: same]
// header: for each rank r, two 64-byte-aligned counters:
//   written[r]  sequence number of the reduction whose data rank r copied in
//   read[r]     sequence number of the reduction rank r finished summing

namespace {
constexpr int64_t kHeader = 4096;
constexpr int64_t kStride = 64;  // one counter per cache line
static_assert(sizeof(std::atomic<int64_t>) == 8, "lock-free 64-bit atomics");

std::atomic<int64_t>* counter(uint8_t* base, int which, int rank) {
  return reinterpret_cast<std::atomic<int64_t>*>(base + (2 * rank + which) * kStride);
}

void spin_until(const std::function<bool()>& ok, const char* what) {
  const auto t0 = std::chrono::steady_clock::now();
  int spins = 0;
  while (!ok()) {
    if (++spins > 64) std::this_thread::yield();
    if (spins > 4096) std::this_thread::sleep_for(std::chrono::microseconds(50));
    if (std::chrono::steady_clock::now() - t0 > std::chrono::minutes(10))
      TORCH_CHECK(false, "shm all-reduce: timed out waiting for the other ranks (", what, ")");
  }
}
}  // namespace

int64_t shm_reducer_bytes(int64_t world, int64_t capacity) {
  TORCH_CHECK(world >= 1 && world * 2 * kStride <= kHeader, "shm all-reduce: unsupported world size ", world);
  return kHeader + 2 * world * capacity * static_cast<int64_t>(sizeof(float));
}

void shm_allreduce(const at::Tensor& buf, const at::Tensor& segment, int64_t rank, int64_t world, int64_t capacity,
                   int64_t seq) {
  TORCH_CHECK(buf.device().is_cpu() && buf.is_contiguous() && buf.scalar_type() == at::kFloat,
              "shm all-reduce: the buffer must be a contiguous float32 CPU tensor");
  TORCH_CHECK(buf.numel() <= capacity, "shm all-reduce: buffer larger than the segment capacity");
  TORCH_CHECK(segment.nbytes() >= static_cast<size_t>(shm_reducer_bytes(world, capacity)),
              "shm all-reduce: segment too small");
  TORCH_CHECK(seq >= 1, "shm all-reduce: sequence numbers start at 1");
  auto* base = static_cast<uint8_t*>(segment.data_ptr());
  const int64_t n = buf.numel();
  float* areas = reinterpret_cast<float*>(base + kHeader) + (seq & 1) * world * capacity;
  float* mine = areas + rank * capacity;
  // The area this reduction uses was last read by reduction seq - 2: every
  // rank must be done with it before it is overwritten.
  spin_until([&] {
    for (int64_t r = 0; r < world; ++r)
      if (counter(base, 1, static_cast<int>(r))->load(std::memory_order_acquire) < seq - 2) return false;
    return true;
  }, "area reuse");
  std::memcpy(mine, buf.data_ptr<float>(), static_cast<size_t>(n) * sizeof(float));
  counter(base, 0, static_cast<int>(rank))->store(seq, std::memory_order_release);
  spin_until([&] {
    for (int64_t r = 0; r < world; ++r)
      if (counter(base, 0, static_cast<int>(r))->load(std::memory_order_acquire) < seq) return false;
    return true;
  }, "peer data");
  // Sum in rank order on every rank: identical results everywhere.
  float* out = buf.data_ptr<float>();
  const int threads = static_cast<int>(std::min<int64_t>(8, std::max<int64_t>(1, n / (1 << 20))));
  auto work = [&](int t) {
    const int64_t lo = n * t / threads, hi = n * (t + 1) / threads;
    for (int64_t i = lo; i < hi; ++i) {
      float s = areas[i];
      for (int64_t r = 1; r < world; ++r) s += areas[r * capacity + i];
      out[i] = s;
    }
  };
  if (threads == 1) {
    work(0);
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
    for (auto& th : pool) th.join();
  }
  counter(base, 1, static_cast<int>(rank))->store(seq, std::memory_order_release);
}

}  // namespace trainstream
