// TrainStream — background checkpoints.
//
// torch.save pickles a state dict on the training thread: the optimizer has to
// be synchronized first, every tensor is copied into the pickle stream, and
// the GPU sits idle until the file is closed. Here the training thread only
// describes the checkpoint; a detached thread writes it:
//
//   * Optimizer state (fp32 master weights, both moments) is written straight
//     from the host buffers the optimizer works on, with asynchronous
//     unbuffered I/O (aio.cpp): no copy in RAM, no page cache.
//   * Consistency without that copy: the checkpoint covers the updates queued
//     when it was requested. The writer waits, per tensor, for those to finish;
//     that tensor's *next* update waits until the tensor is on disk
//     (OffloadAdamW::checkpoint_*). Tensors are written in the order the next
//     step will update them, so training only ever waits when the SSD is slower
//     than a whole forward and backward pass.
//   * State that lives in VRAM (a resident optimizer, fp32 weights in gpu mode)
//     arrives as private pinned snapshots filled by D2H copies; the writer
//     waits for each copy's event.
//   * The file is written under a temporary name, flushed, then renamed over
//     the target, so a crash mid-write leaves the previous checkpoint intact.

#include "aio.h"
#include "offload_adamw.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace trainstream {

namespace {
constexpr size_t kScratch = size_t(4) << 20;

void run_batch(aio::Engine& io, const aio::Request& r) {
  auto batch = io.submit({r});
  batch->wait();
  const std::string err = batch->error();
  if (!err.empty()) throw std::runtime_error(err);
}

// file[offset, +n) <- p[0, n), the last sector padded with zeros. Whole sectors
// go straight from `p` when it is sector-aligned (pinned memory is); anything
// else passes through `scratch`.
void write_bytes(aio::Engine& io, int file, int64_t offset, const char* p, size_t n, char* scratch) {
  size_t at = 0;
  if (reinterpret_cast<uintptr_t>(p) % aio::kAlign == 0) {
    at = n / aio::kAlign * aio::kAlign;
    if (at > 0) {
      aio::Request r;
      r.file = file;
      r.offset = offset;
      r.len = at;
      r.buf = const_cast<char*>(p);
      r.write = true;
      run_batch(io, r);
    }
  }
  while (at < n) {
    const size_t take = std::min(kScratch, n - at);
    const size_t padded = aio::align_up(take);
    std::memcpy(scratch, p + at, take);
    if (padded > take) std::memset(scratch + take, 0, padded - take);
    aio::Request r;
    r.file = file;
    r.offset = offset + static_cast<int64_t>(at);
    r.len = padded;
    r.buf = scratch;
    r.write = true;
    run_batch(io, r);
    at += take;
  }
}

void write_checkpoint(std::shared_ptr<CheckpointJob> job, std::string path, std::string header,
                      std::vector<CheckpointItem> items, OffloadAdamW* engine, std::vector<int64_t> targets,
                      at::Tensor steps, int64_t steps_offset, int device) {
  const auto t0 = std::chrono::steady_clock::now();
  cudaSetDevice(device);  // CUDA's current device is per host thread (event waits, pinned frees)
  aio::Engine& io = aio::Engine::get();
  const std::string tmp = path + ".tmp";
  int file = -1;
  char* scratch = nullptr;
  std::string error;
  int64_t bytes = 0;
  try {
    scratch = static_cast<char*>(aio::aligned_alloc_io(kScratch));
    file = io.open(tmp, true, true);
    io.truncate(file, 0);  // whatever an interrupted run left under this name
    write_bytes(io, file, 0, header.data(), header.size(), scratch);
    bytes += static_cast<int64_t>(header.size());
    std::vector<char> waited(targets.size(), 0);
    for (CheckpointItem& it : items) {
      if (it.after) it.after->synchronize();  // a snapshot: its D2H copy has landed
      if (engine != nullptr && it.slot >= 0) {
        const auto k = static_cast<size_t>(it.slot);
        TORCH_CHECK(k < targets.size(), "checkpoint: bad slot ", it.slot);
        if (!waited[k]) {  // live state: the updates this checkpoint covers have finished
          const int64_t step = engine->checkpoint_wait(it.slot, targets[k]);
          if (steps.defined()) steps.data_ptr<int64_t>()[k] = step;
          waited[k] = 1;
        }
      }
      write_bytes(io, file, it.offset, static_cast<const char*>(it.data.data_ptr()), it.data.nbytes(), scratch);
      bytes += static_cast<int64_t>(it.data.nbytes());
      if (engine != nullptr && it.slot >= 0 && it.release) engine->checkpoint_release(it.slot);
    }
    if (steps.defined()) {
      write_bytes(io, file, steps_offset, static_cast<const char*>(steps.data_ptr()), steps.nbytes(), scratch);
      bytes += static_cast<int64_t>(steps.nbytes());
    }
    io.flush(file);
    io.close(file);
    file = -1;
    aio::replace_file(tmp, path);
  } catch (const std::exception& e) {
    error = e.what();
  } catch (...) {
    error = "unknown error";
  }
  if (file >= 0) io.close(file);
  if (!error.empty()) aio::remove_file(tmp);
  if (scratch != nullptr) aio::aligned_free_io(scratch);
  try {
    items.clear();  // snapshots go back to the pinned pool
    steps = at::Tensor();
  } catch (...) {
  }
  if (engine != nullptr) engine->checkpoint_end();  // also drops the holds a failure left behind
  checkpoint_finish(*job, error, bytes,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
}  // namespace

bool CheckpointJob::done() {
  std::lock_guard<std::mutex> lock(mu_);
  return done_;
}

void CheckpointJob::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return done_; });
}

std::string CheckpointJob::error() {
  std::lock_guard<std::mutex> lock(mu_);
  return error_;
}

double CheckpointJob::seconds() {
  std::lock_guard<std::mutex> lock(mu_);
  return seconds_;
}

int64_t CheckpointJob::bytes() {
  std::lock_guard<std::mutex> lock(mu_);
  return bytes_;
}

void checkpoint_finish(CheckpointJob& job, const std::string& error, int64_t bytes, double seconds) {
  {
    std::lock_guard<std::mutex> lock(job.mu_);
    job.error_ = error;
    job.bytes_ = bytes;
    job.seconds_ = seconds;
    job.done_ = true;
  }
  job.cv_.notify_all();
}

std::shared_ptr<CheckpointJob> checkpoint_write_async(const std::string& path, const std::string& header,
                                                      std::vector<CheckpointItem> items, OffloadAdamW* engine,
                                                      const std::vector<int64_t>& targets, const at::Tensor& steps,
                                                      int64_t steps_offset, int device) {
  auto job = std::make_shared<CheckpointJob>();
  try {
    for (const CheckpointItem& it : items) {
      TORCH_CHECK(it.data.defined() && it.data.device().is_cpu() && it.data.is_non_overlapping_and_dense(),
                  "checkpoint: every item must be a dense host tensor");
      TORCH_CHECK(it.offset >= 0 && it.offset % static_cast<int64_t>(aio::kAlign) == 0,
                  "checkpoint: offsets must be multiples of 4096");
    }
    TORCH_CHECK(!steps.defined() || (steps.scalar_type() == at::kLong && steps.device().is_cpu() &&
                                     steps.numel() == static_cast<int64_t>(targets.size()) &&
                                     steps_offset % static_cast<int64_t>(aio::kAlign) == 0),
                "checkpoint: steps must be an int64 host tensor with one entry per slot");
    std::thread(write_checkpoint, job, path, header, std::move(items), engine, targets, steps, steps_offset,
                resolve_device(device))
        .detach();
  } catch (...) {
    if (engine != nullptr) engine->checkpoint_end();  // nothing will read the state: let updates go on
    throw;
  }
  return job;
}

}  // namespace trainstream
