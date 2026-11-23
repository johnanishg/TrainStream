// TrainStream — the disk tier's read path.
//
// Frozen weights the page-lock budget cannot hold live in files (trainstream/
// disk.py). They used to reach the GPU through their memory mapping: a host
// memcpy out of the mapping into a pinned bounce buffer, on the thread that
// runs the training loop, where every page not in the OS cache is a page
// fault that thread sits through.
//
// Here the bytes never pass through the page cache or the training thread:
//
//   disk_stage      queues unbuffered asynchronous reads (aio.cpp) of a block's
//                   files straight into a pinned staging slot and returns at
//                   once. The pager calls it a block or two ahead of the copy.
//   h2d_disk_async  queues, on a CUDA stream of its own, a host callback that
//                   waits for those reads, then the DMA copies out of the slot,
//                   and returns the event that follows them. The compute stream
//                   waits on that event on the GPU, as for any paged weight;
//                   the host waits for nothing.
//
// The stream is separate from the H2D stream so that a slow read holds up only
// disk-backed weights, not the activations and pinned weights queued there.
// A slot keeps its bytes until it is reused, so the blocks around the turn
// from forward to backward are staged once and copied twice.

#include "aio.h"
#include "trainstream.h"

#include <cerrno>
#include <cstring>
#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace trainstream {

namespace {
constexpr int kMaxDiskDevices = 64;

struct Slot {
  at::Tensor mem;          // pinned bytes (kept alive here)
  char* base = nullptr;    // 4096-aligned start inside `mem`
  size_t cap = 0;
  int64_t key = -1;        // whose bytes these are (-1: nobody's / stale)
  uint64_t generation = 0;
  std::shared_ptr<aio::Batch> batch;  // the reads that fill it
  EventPtr done;                      // the last DMA that read from it
  std::vector<size_t> offsets, sizes;
};

struct SlotRing {
  std::mutex mu;
  std::vector<Slot> slots;
  size_t next = 0;
};

// Leaked singletons (see memory_pinner.cpp): no CUDA teardown from static destructors.
SlotRing& ring_for(int dev) {
  static auto* rings = new SlotRing[kMaxDiskDevices];
  TORCH_CHECK(dev >= 0 && dev < kMaxDiskDevices, "invalid CUDA device ", dev);
  return rings[dev];
}

struct Files {
  std::mutex mu;
  std::unordered_map<std::string, int> ids;  // path -> aio file id (read-only handles)
};
Files& files() {
  static auto* f = new Files();
  return *f;
}

int file_id(const std::string& path) {
  Files& f = files();
  std::lock_guard<std::mutex> lock(f.mu);
  auto it = f.ids.find(path);
  if (it != f.ids.end()) return it->second;
  const int id = aio::Engine::get().open(path, false, false);
  f.ids.emplace(path, id);
  return id;
}

struct Counters {
  std::atomic<int64_t> stages{0}, hits{0}, staged_bytes{0}, copies{0}, copied_bytes{0}, wait_us{0};
  std::mutex mu;
  std::string error;
};
Counters& counters() {
  static auto* c = new Counters();
  return *c;
}

void check_disk_error() {
  Counters& c = counters();
  std::string msg;
  {
    std::lock_guard<std::mutex> lock(c.mu);
    msg.swap(c.error);
  }
  TORCH_CHECK(msg.empty(), "TrainStream disk tier: ", msg);
}

// Runs on a CUDA-internal thread, in stream order, and may not call CUDA: it
// only blocks the disk stream until the slot's reads have landed.
struct StageWait {
  std::shared_ptr<aio::Batch> batch;
};
void CUDART_CB wait_stage(void* user) {
  auto* w = static_cast<StageWait*>(user);
  const double t0 = now_ms();
  w->batch->wait();
  Counters& c = counters();
  c.wait_us += static_cast<int64_t>((now_ms() - t0) * 1e3);
  const std::string err = w->batch->error();
  if (!err.empty()) {
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.error.empty()) c.error = err;  // raised on the next disk call from Python
  }
  delete w;
}
}  // namespace

std::shared_ptr<DiskStage> disk_stage(const std::vector<std::string>& paths, const std::vector<int64_t>& offsets,
                                      const std::vector<int64_t>& nbytes, int64_t key, int device, int64_t slots) {
  check_disk_error();
  TORCH_CHECK(!paths.empty() && paths.size() == offsets.size() && paths.size() == nbytes.size(),
              "disk_stage: need matching path / offset / size lists");
  const int dev = resolve_device(device);
  SlotRing& ring = ring_for(dev);
  std::lock_guard<std::mutex> lock(ring.mu);
  const size_t n = static_cast<size_t>(std::max<int64_t>(slots, 2));
  if (ring.slots.size() < n) ring.slots.resize(n);

  auto describe = [&](size_t k, bool hit) {
    Slot& s = ring.slots[k];
    auto st = std::make_shared<DiskStage>();
    st->device = dev;
    st->slot = static_cast<int64_t>(k);
    st->generation = s.generation;
    st->key = s.key;
    st->batch = s.batch;
    st->hit = hit;
    for (size_t b : s.sizes) st->bytes += static_cast<int64_t>(b);
    return st;
  };
  if (key >= 0) {
    for (size_t k = 0; k < ring.slots.size(); ++k) {
      if (ring.slots[k].key == key && ring.slots[k].sizes.size() == paths.size()) {
        counters().hits += 1;
        return describe(k, true);  // staged ahead of time, or still there from the last pass
      }
    }
  }

  size_t total = 0;
  std::vector<size_t> offs, sizes;
  for (size_t i = 0; i < paths.size(); ++i) {
    TORCH_CHECK(nbytes[i] >= 0 && offsets[i] >= 0 && offsets[i] % static_cast<int64_t>(aio::kAlign) == 0,
                "disk_stage: offsets must be multiples of 4096");
    offs.push_back(total);
    sizes.push_back(static_cast<size_t>(nbytes[i]));
    total += aio::align_up(static_cast<size_t>(nbytes[i]));
  }
  const size_t k = ring.next++ % ring.slots.size();
  Slot& s = ring.slots[k];
  if (s.batch) s.batch->wait();        // reads still landing in it
  if (s.done) s.done->synchronize();   // the GPU still copying out of it
  s.key = -1;
  if (!s.mem.defined() || s.cap < total) {
    s.mem = at::Tensor();  // released before the bigger one is taken
    s.base = nullptr;
    s.cap = 0;
    s.mem = pinned_empty({static_cast<int64_t>(total + aio::kAlign)}, at::kByte, "disk.staging", false, false);
    const auto addr = reinterpret_cast<uintptr_t>(s.mem.data_ptr());
    s.base = reinterpret_cast<char*>((addr + aio::kAlign - 1) / aio::kAlign * aio::kAlign);
    s.cap = total;
  }
  std::vector<aio::Request> reqs;
  for (size_t i = 0; i < paths.size(); ++i) {
    aio::Request r;
    r.file = file_id(paths[i]);
    r.offset = offsets[i];
    r.len = aio::align_up(sizes[i]);
    r.buf = s.base + offs[i];
    reqs.push_back(r);
  }
  s.generation += 1;
  s.done.reset();
  s.offsets = offs;
  s.sizes = sizes;
  s.batch = aio::Engine::get().submit(reqs);
  s.key = key;
  counters().stages += 1;
  counters().staged_bytes += static_cast<int64_t>(total);
  return describe(k, false);
}

EventPtr h2d_disk_async(const std::vector<at::Tensor>& dsts, const std::shared_ptr<DiskStage>& stage,
                        const EventPtr& after) {
  check_disk_error();
  TORCH_CHECK(stage != nullptr, "h2d_disk_async: no stage");
  const int dev = stage->device;
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  SlotRing& ring = ring_for(dev);
  std::lock_guard<std::mutex> lock(ring.mu);
  TORCH_CHECK(stage->slot >= 0 && static_cast<size_t>(stage->slot) < ring.slots.size(), "h2d_disk_async: bad stage");
  Slot& s = ring.slots[static_cast<size_t>(stage->slot)];
  TORCH_CHECK(s.generation == stage->generation,
              "h2d_disk_async: the staging slot was reused before its copy was queued (stage it again)");
  TORCH_CHECK(dsts.size() == s.sizes.size(), "h2d_disk_async: ", dsts.size(), " tensors for ", s.sizes.size(),
              " staged files");
  for (size_t i = 0; i < dsts.size(); ++i) {
    TORCH_CHECK(dsts[i].is_cuda() && dsts[i].get_device() == dev && dsts[i].is_non_overlapping_and_dense(),
                "h2d_disk_async: dst must be a dense CUDA tensor on the stage's device");
    TORCH_CHECK(dsts[i].nbytes() == s.sizes[i], "h2d_disk_async: size mismatch (", dsts[i].nbytes(), " vs ",
                s.sizes[i], " bytes)");
  }
  if (after) after->block(rt.disk);
  auto* w = new StageWait{s.batch};
  const cudaError_t err = cudaLaunchHostFunc(rt.disk, &wait_stage, w);
  if (err != cudaSuccess) {
    delete w;
    TS_CUDA_CHECK(err);
  }
  for (size_t i = 0; i < dsts.size(); ++i) {
    if (s.sizes[i] > 0)
      TS_CUDA_CHECK(cudaMemcpyAsync(dsts[i].data_ptr(), s.base + s.offsets[i], s.sizes[i], cudaMemcpyHostToDevice,
                                    rt.disk));
    rt.bytes_h2d += s.sizes[i];
    rt.copies_h2d += 1;
    counters().copied_bytes += static_cast<int64_t>(s.sizes[i]);
  }
  counters().copies += 1;
  auto done = make_event(dev);
  done->record(rt.disk);
  s.done = done;
  return done;
}

void disk_invalidate(int device) {
  for (int d = 0; d < kMaxDiskDevices; ++d) {
    if (device >= 0 && d != device) continue;
    SlotRing& ring = ring_for(d);
    std::lock_guard<std::mutex> lock(ring.mu);
    for (Slot& s : ring.slots) s.key = -1;  // never served again; a stage already handed out stays usable
  }
}

void disk_release(int device) {
  for (int d = 0; d < kMaxDiskDevices; ++d) {
    if (device >= 0 && d != device) continue;
    SlotRing& ring = ring_for(d);
    std::lock_guard<std::mutex> lock(ring.mu);
    for (Slot& s : ring.slots) {
      if (s.batch) s.batch->wait();
      if (s.done) s.done->synchronize();
    }
    ring.slots.clear();  // the pinned slots go back to the arena
    ring.next = 0;
  }
}

void disk_close(const std::vector<std::string>& paths) {
  Files& f = files();
  std::lock_guard<std::mutex> lock(f.mu);
  if (paths.empty()) {
    for (auto& kv : f.ids) aio::Engine::get().close(kv.second);
    f.ids.clear();
    return;
  }
  for (const std::string& p : paths) {
    auto it = f.ids.find(p);
    if (it == f.ids.end()) continue;
    aio::Engine::get().close(it->second);
    f.ids.erase(it);
  }
}

void disk_sync(const at::Tensor& t) {
  TORCH_CHECK(t.device().is_cpu(), "disk_sync: a CPU tensor backed by a mapped file");
  if (t.nbytes() == 0) return;
#ifdef _WIN32
  // Dirty pages of the mapping go to the device; unbuffered reads then see them.
  TORCH_CHECK(FlushViewOfFile(t.data_ptr(), t.nbytes()) != 0, "TrainStream: FlushViewOfFile failed (Windows error ",
              static_cast<long>(GetLastError()), ")");
#else
  const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const auto addr = reinterpret_cast<uintptr_t>(t.data_ptr());
  const uintptr_t start = addr / page * page;
  TORCH_CHECK(msync(reinterpret_cast<void*>(start), addr - start + t.nbytes(), MS_SYNC) == 0,
              "TrainStream: msync failed (", std::strerror(errno), ")");
#endif
}

void disk_read(const at::Tensor& dst, const std::string& path, int64_t offset) {
  TORCH_CHECK(dst.device().is_cpu() && dst.is_non_overlapping_and_dense(), "disk_read: dst must be a dense CPU tensor");
  TORCH_CHECK(offset >= 0 && offset % static_cast<int64_t>(aio::kAlign) == 0, "disk_read: offset must be a multiple of 4096");
  const size_t bytes = dst.nbytes();
  if (bytes == 0) return;
  const size_t len = aio::align_up(bytes);
  char* buf = static_cast<char*>(aio::aligned_alloc_io(len));
  aio::Request r;
  r.file = file_id(path);
  r.offset = offset;
  r.len = len;
  r.buf = buf;
  auto batch = aio::Engine::get().submit({r});
  batch->wait();
  const std::string err = batch->error();
  const int64_t got = batch->bytes();
  if (err.empty() && got >= static_cast<int64_t>(bytes)) std::memcpy(dst.data_ptr(), buf, bytes);
  aio::aligned_free_io(buf);
  TORCH_CHECK(err.empty(), "TrainStream disk tier: ", err);
  TORCH_CHECK(got >= static_cast<int64_t>(bytes), "TrainStream disk tier: '", path, "' holds only ", got,
              " bytes at offset ", offset, ", ", bytes, " wanted");
}

std::string disk_backend() { return aio::Engine::get().backend(); }

std::map<std::string, double> disk_stats() {
  std::map<std::string, double> out = aio::Engine::get().stats();
  Counters& c = counters();
  out["stages"] = static_cast<double>(c.stages.load());
  out["stage_hits"] = static_cast<double>(c.hits.load());
  out["staged_bytes"] = static_cast<double>(c.staged_bytes.load());
  out["copies"] = static_cast<double>(c.copies.load());
  out["copied_bytes"] = static_cast<double>(c.copied_bytes.load());
  out["stream_wait_ms"] = c.wait_us.load() / 1e3;
  int64_t slots = 0, slot_bytes = 0;
  for (int d = 0; d < kMaxDiskDevices; ++d) {
    SlotRing& ring = ring_for(d);
    std::lock_guard<std::mutex> lock(ring.mu);
    for (const Slot& s : ring.slots) {
      slots += s.mem.defined();
      slot_bytes += static_cast<int64_t>(s.cap);
    }
  }
  out["slots"] = static_cast<double>(slots);
  out["slot_bytes"] = static_cast<double>(slot_bytes);
  return out;
}

bool disk_stage_ready(const DiskStage& stage) { return !stage.batch || stage.batch->done(); }

void disk_stage_wait(const DiskStage& stage) {
  if (stage.batch) stage.batch->wait();
  check_disk_error();
}

}  // namespace trainstream
