// TrainStream — asynchronous unbuffered file I/O (the disk tier and checkpoints).
//
// Reads and writes go straight between the SSD and caller-owned memory: no OS
// page cache, no page faults, and the calling thread never waits for the disk.
// Only operating-system interfaces are used, so nothing here can go away with
// a third-party project:
//
//   Windows  CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) with an
//            I/O completion port
//   Linux    io_uring through its raw system calls (io_uring_setup /
//            io_uring_enter and the kernel's own <linux/io_uring.h>; no
//            liburing) on files opened O_DIRECT
//   anywhere a pool of threads doing positional reads / writes, when io_uring
//            is missing or forbidden (old kernels, seccomp), or on other POSIX
//
// A file system that refuses unbuffered access (tmpfs, some network mounts) is
// opened buffered instead; everything else stays the same.
//
// This header and aio.cpp include neither PyTorch nor CUDA, so the engine can
// be built and tested on its own.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trainstream {
namespace aio {

// Offsets, lengths and buffer addresses of unbuffered I/O are multiples of this
// (the logical sector size of every current SSD divides it).
constexpr size_t kAlign = 4096;

inline size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

// The completion of a group of operations queued together.
class Batch {
 public:
  void wait();                // blocks until every operation has finished (or failed)
  bool done();
  std::string error();        // empty when everything succeeded so far
  // Bytes actually moved. A read that runs past the end of its file moves only
  // what the file holds; the buffer beyond that is unspecified (some systems
  // leave it alone, Linux may zero it).
  int64_t bytes() const { return bytes_.load(); }
  double seconds();           // submit to last completion (0 until done)

 private:
  friend class Engine;
  std::mutex mu_;
  std::condition_variable cv_;
  int pending_ = 0;
  std::string error_;
  std::atomic<int64_t> bytes_{0};
  double t0_ = 0.0, t1_ = 0.0;
};

struct Request {
  int file = -1;
  int64_t offset = 0;  // in the file
  size_t len = 0;      // bytes; a read may run past the end of the file (see Batch::bytes)
  char* buf = nullptr;
  bool write = false;
};

class Engine {
 public:
  static Engine& get();  // one per process, created on first use, never destroyed

  // Opens `path` for asynchronous I/O and returns its id. Unbuffered when the
  // file system allows it (direct()), buffered otherwise.
  int open(const std::string& path, bool writable = false, bool create = false);
  void close(int file);
  bool direct(int file);
  int64_t size(int file);
  void flush(int file);                     // written data is on the device (blocks)
  void truncate(int file, int64_t bytes);   // set the file's length (blocks)

  // Queues the requests and returns at once. For unbuffered files, offsets,
  // lengths and buffers must be multiples of kAlign. The buffers must stay
  // valid until the batch is done.
  std::shared_ptr<Batch> submit(const std::vector<Request>& requests);

  const char* backend() const;  // "iocp", "io_uring" or "threads"
  // files_open, ops, bytes_read, bytes_written, busy_ms (time with I/O in flight)
  std::map<std::string, double> stats();

  struct Impl;

 private:
  Engine();
  Impl* impl_;
};

// `from` takes `to`'s place in one step (an existing `to` is replaced): a reader sees the old
// file or the new one, never a partial one. Throws std::runtime_error on failure.
void replace_file(const std::string& from, const std::string& to);
void remove_file(const std::string& path);  // best effort

// Memory aligned for unbuffered I/O (free with aligned_free_io).
void* aligned_alloc_io(size_t bytes);
void aligned_free_io(void* p);

}  // namespace aio
}  // namespace trainstream
