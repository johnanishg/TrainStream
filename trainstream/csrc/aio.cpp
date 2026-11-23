// TrainStream — asynchronous unbuffered file I/O. See aio.h.
//
// One engine per process. Callers queue requests and get a Batch back at once;
// a submitter thread feeds the operating system up to kDepth operations of
// kChunk bytes at a time, and a completion thread retires them. Nothing here
// touches PyTorch or CUDA.

#include "aio.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#if defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
// IORING_FEAT_RW_CUR_POS arrived with Linux 5.6, the release that added the
// plain IORING_OP_READ / IORING_OP_WRITE used below: older headers fall back
// to the thread pool at compile time, older kernels at run time.
#if defined(IORING_FEAT_RW_CUR_POS) && defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
#define TS_IO_URING 1
#endif
#endif
#endif
#endif
#endif

namespace trainstream {
namespace aio {

namespace {

constexpr size_t kChunk = size_t(4) << 20;  // bytes per operation
constexpr int kDepth = 32;                  // operations in flight (128 MiB)
constexpr int kPoolThreads = 8;             // thread-pool backend

double now_s() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Op;
#ifdef _WIN32
struct Overlapped {  // what the completion port hands back: `ov` first, then the way home
  OVERLAPPED ov;
  Op* op;
};
#endif

struct Op {
#ifdef _WIN32
  Overlapped w;
#endif
  std::shared_ptr<Batch> batch;
  int file = -1;
  int64_t offset = 0;
  size_t len = 0;
  char* buf = nullptr;
  bool write = false;
  size_t done = 0;        // bytes transferred so far; a partial transfer continues from here
  bool retried = false;   // already re-issued once after the file fell back to buffered I/O
  bool dead = false;      // given up while still in the ring: only freed when it comes back
};

struct File {
  bool used = false;
  bool direct = false;
  bool writable = false;
  std::string path;
#ifdef _WIN32
  HANDLE h = INVALID_HANDLE_VALUE;
#else
  int fd = -1;
#endif
};

std::string os_error(const char* what, const std::string& path, long code) {
#ifdef _WIN32
  return std::string(what) + " '" + path + "': Windows error " + std::to_string(code);
#else
  return std::string(what) + " '" + path + "': " + std::strerror(static_cast<int>(code));
#endif
}

#ifdef TS_IO_URING
struct Uring {
  int fd = -1;
  unsigned* sq_tail = nullptr;
  unsigned* sq_mask = nullptr;
  unsigned* sq_array = nullptr;
  io_uring_sqe* sqes = nullptr;
  unsigned* cq_head = nullptr;
  unsigned* cq_tail = nullptr;
  unsigned* cq_mask = nullptr;
  io_uring_cqe* cqes = nullptr;

  bool init(unsigned entries) {
    io_uring_params p;
    std::memset(&p, 0, sizeof(p));
    fd = static_cast<int>(syscall(__NR_io_uring_setup, entries, &p));
    if (fd < 0) return false;  // ENOSYS (old kernel), EPERM (seccomp / sysctl), ENOMEM
    if (!(p.features & IORING_FEAT_RW_CUR_POS)) {  // a kernel before 5.6: no IORING_OP_READ
      ::close(fd);
      fd = -1;
      return false;
    }
    size_t sq_bytes = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_bytes = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (single) sq_bytes = cq_bytes = std::max(sq_bytes, cq_bytes);
    void* sq = mmap(nullptr, sq_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    void* cq = single ? sq
                      : mmap(nullptr, cq_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                             IORING_OFF_CQ_RING);
    void* se = mmap(nullptr, p.sq_entries * sizeof(io_uring_sqe), PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || se == MAP_FAILED) {
      ::close(fd);
      fd = -1;
      return false;
    }
    char* s = static_cast<char*>(sq);
    char* c = static_cast<char*>(cq);
    sq_tail = reinterpret_cast<unsigned*>(s + p.sq_off.tail);
    sq_mask = reinterpret_cast<unsigned*>(s + p.sq_off.ring_mask);
    sq_array = reinterpret_cast<unsigned*>(s + p.sq_off.array);
    sqes = static_cast<io_uring_sqe*>(se);
    cq_head = reinterpret_cast<unsigned*>(c + p.cq_off.head);
    cq_tail = reinterpret_cast<unsigned*>(c + p.cq_off.tail);
    cq_mask = reinterpret_cast<unsigned*>(c + p.cq_off.ring_mask);
    cqes = reinterpret_cast<io_uring_cqe*>(c + p.cq_off.cqes);
    return true;
  }
};
#endif

}  // namespace

// ===========================================================================
// Batch
// ===========================================================================
void Batch::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return pending_ == 0; });
}

bool Batch::done() {
  std::lock_guard<std::mutex> lock(mu_);
  return pending_ == 0;
}

std::string Batch::error() {
  std::lock_guard<std::mutex> lock(mu_);
  return error_;
}

double Batch::seconds() {
  std::lock_guard<std::mutex> lock(mu_);
  return pending_ == 0 ? t1_ - t0_ : 0.0;
}

// ===========================================================================
// Engine
// ===========================================================================
struct Engine::Impl {
  enum Kind { kIocp, kUring, kThreads };
  Kind kind = kThreads;

  std::mutex mu;               // queue, inflight, files, busy clock
  std::condition_variable cv;  // work arrived, or an operation left the system
  std::deque<Op*> queue;
  int inflight = 0;
  int outstanding = 0;         // queued + in flight
  double busy_since = 0.0, busy_total = 0.0;
  std::vector<File> files;
  std::atomic<int64_t> ops{0}, bytes_read{0}, bytes_written{0};

#ifdef _WIN32
  HANDLE iocp = nullptr;
#endif
#ifdef TS_IO_URING
  Uring ring;
  std::mutex sq_mu;
#endif

  void start() {
#ifdef _WIN32
    iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (iocp == nullptr) throw std::runtime_error("TrainStream: CreateIoCompletionPort failed");
    kind = kIocp;
    std::thread([this] { submit_loop(); }).detach();
    std::thread([this] { iocp_reaper(); }).detach();
#else
    rlimit lim;
    if (getrlimit(RLIMIT_NOFILE, &lim) == 0 && lim.rlim_cur < lim.rlim_max) {
      // one descriptor per weight file: a large model has more than the usual soft limit of 1024
      lim.rlim_cur = std::min<rlim_t>(lim.rlim_max, 65536);
      setrlimit(RLIMIT_NOFILE, &lim);
    }
#ifdef TS_IO_URING
    if (std::getenv("TRAINSTREAM_NO_IO_URING") == nullptr && ring.init(2 * kDepth)) {
      kind = kUring;
      std::thread([this] { submit_loop(); }).detach();
      std::thread([this] { uring_reaper(); }).detach();
      return;
    }
#endif
    kind = kThreads;
    for (int i = 0; i < kPoolThreads; ++i) std::thread([this] { pool_worker(); }).detach();
#endif
  }

  File file_of(int id) {
    std::lock_guard<std::mutex> lock(mu);
    if (id < 0 || id >= static_cast<int>(files.size()) || !files[static_cast<size_t>(id)].used)
      throw std::runtime_error("TrainStream: bad file id " + std::to_string(id));
    return files[static_cast<size_t>(id)];
  }

  // ---- completion ---------------------------------------------------------
  void requeue(Op* op) {  // goes again (the rest of a partial transfer, or a retry)
    {
      std::lock_guard<std::mutex> lock(mu);
      queue.push_front(op);
      --inflight;
    }
    cv.notify_all();
  }

  void retire(Op* op, const std::string& error) {
    Batch& b = *op->batch;
    if (error.empty()) {
      (op->write ? bytes_written : bytes_read) += static_cast<int64_t>(op->done);
      b.bytes_ += static_cast<int64_t>(op->done);
    }
    ops += 1;
    {
      std::lock_guard<std::mutex> lock(mu);
      --inflight;
      if (--outstanding == 0) busy_total += now_s() - busy_since;
    }
    cv.notify_all();
    std::shared_ptr<Batch> keep = op->batch;  // the batch outlives the notify below
    if (op->dead) {
      op->batch.reset();  // still in the ring: freed when its completion arrives
    } else {
      delete op;
    }
    {
      std::lock_guard<std::mutex> lock(keep->mu_);
      if (!error.empty() && keep->error_.empty()) keep->error_ = error;
      if (--keep->pending_ == 0) keep->t1_ = now_s();
    }
    keep->cv_.notify_all();
  }

  // One operation came back with `n` bytes moved, or with OS error `err`.
  void finish(Op* op, int64_t n, long err) {
    if (op->dead) {  // given up earlier (and already retired)
      delete op;
      return;
    }
    if (err != 0) {
#ifndef _WIN32
      if (err == EINVAL && !op->retried && to_buffered(op->file)) {
        op->retried = true;  // the file system took O_DIRECT at open() but not for this transfer
        requeue(op);
        return;
      }
#endif
      retire(op, os_error(op->write ? "writing" : "reading", file_path(op->file), err));
      return;
    }
    op->done += static_cast<size_t>(n);
    // Less than asked for: more to come (go again), or the end of the file. A
    // count that is not a whole number of sectors can only be the end.
    if (n > 0 && op->done < op->len && n % static_cast<int64_t>(kAlign) == 0) {
      requeue(op);
      return;
    }
    if (op->write && op->done < op->len) {
      retire(op, "writing '" + file_path(op->file) + "': the device took only part of the data (disk full?)");
      return;
    }
    retire(op, "");  // a read that reached the end of the file: done with what the file holds
  }

  std::string file_path(int id) {
    std::lock_guard<std::mutex> lock(mu);
    return id >= 0 && id < static_cast<int>(files.size()) ? files[static_cast<size_t>(id)].path : std::string("?");
  }

#ifndef _WIN32
  bool to_buffered(int id) {
    std::lock_guard<std::mutex> lock(mu);
    if (id < 0 || id >= static_cast<int>(files.size())) return false;
    File& f = files[static_cast<size_t>(id)];
    if (!f.used || !f.direct) return false;
#ifdef O_DIRECT
    const int flags = fcntl(f.fd, F_GETFL);
    if (flags < 0 || fcntl(f.fd, F_SETFL, flags & ~O_DIRECT) != 0) return false;
#endif
    f.direct = false;
    return true;
  }
#endif

  // ---- submission ---------------------------------------------------------
  void submit_loop() {
    for (;;) {
      Op* op;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this] { return !queue.empty() && inflight < kDepth; });
        op = queue.front();
        queue.pop_front();
        ++inflight;
      }
      issue(op);
    }
  }

  void issue(Op* op) {
    File f;
    try {
      f = file_of(op->file);
    } catch (const std::exception& e) {
      retire(op, e.what());
      return;
    }
#ifdef _WIN32
    std::memset(&op->w.ov, 0, sizeof(op->w.ov));
    op->w.op = op;
    const uint64_t pos = static_cast<uint64_t>(op->offset) + op->done;
    op->w.ov.Offset = static_cast<DWORD>(pos & 0xffffffffull);
    op->w.ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
    const DWORD n = static_cast<DWORD>(op->len - op->done);
    const BOOL ok = op->write ? WriteFile(f.h, op->buf + op->done, n, nullptr, &op->w.ov)
                              : ReadFile(f.h, op->buf + op->done, n, nullptr, &op->w.ov);
    if (ok) return;  // finished already: its completion packet is queued to the port all the same
    const DWORD e = GetLastError();
    if (e == ERROR_IO_PENDING) return;
    finish(op, 0, e == ERROR_HANDLE_EOF ? 0 : static_cast<long>(e));  // no packet follows a refusal
#elif defined(TS_IO_URING)
    std::lock_guard<std::mutex> lock(sq_mu);
    const unsigned tail = *ring.sq_tail;
    const unsigned idx = tail & *ring.sq_mask;
    io_uring_sqe* sqe = &ring.sqes[idx];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = op->write ? IORING_OP_WRITE : IORING_OP_READ;
    sqe->fd = f.fd;
    sqe->off = static_cast<uint64_t>(op->offset) + op->done;
    sqe->addr = reinterpret_cast<uint64_t>(op->buf + op->done);
    sqe->len = static_cast<unsigned>(op->len - op->done);
    sqe->user_data = reinterpret_cast<uint64_t>(op);
    ring.sq_array[idx] = idx;
    __atomic_store_n(ring.sq_tail, tail + 1, __ATOMIC_RELEASE);
    for (int tries = 0;; ++tries) {
      if (syscall(__NR_io_uring_enter, ring.fd, 1, 0, 0, nullptr, 0) >= 0) return;
      if (errno == EINTR) continue;
      if ((errno == EAGAIN || errno == EBUSY) && tries < 10000) {  // completions must be reaped first
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      // The entry stays in the ring and may still be submitted later: the operation is given
      // up now, and its memory is released only if and when its completion shows up.
      const long e = errno;
      op->dead = true;
      retire(op, os_error("queueing I/O on", f.path, e));
      return;
    }
#else
    (void)f;
    retire(op, "TrainStream: no asynchronous I/O backend");
#endif
  }

#ifdef _WIN32
  void iocp_reaper() {
    for (;;) {
      DWORD n = 0;
      ULONG_PTR key = 0;
      LPOVERLAPPED ov = nullptr;
      const BOOL ok = GetQueuedCompletionStatus(iocp, &n, &key, &ov, INFINITE);
      if (ov == nullptr) continue;  // the wait itself failed: nothing was dequeued
      Op* op = reinterpret_cast<Overlapped*>(ov)->op;
      long err = 0;
      if (!ok) {
        const DWORD e = GetLastError();
        if (e != ERROR_HANDLE_EOF) err = static_cast<long>(e);
      }
      finish(op, static_cast<int64_t>(n), err);
    }
  }
#endif

#ifdef TS_IO_URING
  void uring_reaper() {
    for (;;) {
      if (syscall(__NR_io_uring_enter, ring.fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0) < 0 && errno != EINTR)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      unsigned head = *ring.cq_head;
      for (;;) {
        const unsigned tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        if (head == tail) break;
        const io_uring_cqe* cqe = &ring.cqes[head & *ring.cq_mask];
        Op* op = reinterpret_cast<Op*>(cqe->user_data);
        const int res = cqe->res;
        ++head;
        __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
        if (op == nullptr) continue;
        if (res == -EAGAIN || res == -EINTR) {
          if (op->dead) {
            delete op;
          } else {
            requeue(op);
          }
        } else if (res < 0) {
          finish(op, 0, -res);
        } else {
          finish(op, res, 0);
        }
      }
    }
  }
#endif

#ifndef _WIN32
  void pool_worker() {
    for (;;) {
      Op* op;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this] { return !queue.empty(); });
        op = queue.front();
        queue.pop_front();
        ++inflight;
      }
      File f;
      try {
        f = file_of(op->file);
      } catch (const std::exception& e) {
        retire(op, e.what());
        continue;
      }
      const size_t want = op->len - op->done;
      const off_t pos = static_cast<off_t>(op->offset) + static_cast<off_t>(op->done);
      ssize_t n;
      do {
        n = op->write ? pwrite(f.fd, op->buf + op->done, want, pos) : pread(f.fd, op->buf + op->done, want, pos);
      } while (n < 0 && errno == EINTR);
      if (n < 0) {
        finish(op, 0, errno);
      } else {
        finish(op, n, 0);
      }
    }
  }
#endif
};

Engine::Engine() : impl_(new Impl()) { impl_->start(); }

Engine& Engine::get() {
  // Leaked on purpose: its threads may be inside the kernel when the process exits.
  static Engine* e = new Engine();
  return *e;
}

const char* Engine::backend() const {
  switch (impl_->kind) {
    case Impl::kIocp:
      return "iocp";
    case Impl::kUring:
      return "io_uring";
    default:
      return "threads";
  }
}

int Engine::open(const std::string& path, bool writable, bool create) {
  File f;
  f.used = true;
  f.path = path;
  f.writable = writable;
#ifdef _WIN32
  const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring wpath(static_cast<size_t>(std::max(wlen, 1)), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);
  const DWORD access = GENERIC_READ | (writable ? GENERIC_WRITE : 0);
  const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  const DWORD how = create ? OPEN_ALWAYS : OPEN_EXISTING;
  f.h = CreateFileW(wpath.c_str(), access, share, nullptr, how, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
                    nullptr);
  f.direct = f.h != INVALID_HANDLE_VALUE;
  if (!f.direct)  // a file system without unbuffered access: the same calls through the cache
    f.h = CreateFileW(wpath.c_str(), access, share, nullptr, how, FILE_FLAG_OVERLAPPED, nullptr);
  if (f.h == INVALID_HANDLE_VALUE) throw std::runtime_error(os_error("opening", path, GetLastError()));
  if (CreateIoCompletionPort(f.h, impl_->iocp, 0, 0) == nullptr) {
    const DWORD e = GetLastError();
    CloseHandle(f.h);
    throw std::runtime_error(os_error("attaching to the completion port", path, e));
  }
#else
  int flags = (writable ? O_RDWR : O_RDONLY) | (create ? O_CREAT : 0);
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_DIRECT
  f.fd = ::open(path.c_str(), flags | O_DIRECT, 0644);
  f.direct = f.fd >= 0;
#endif
  if (f.fd < 0) f.fd = ::open(path.c_str(), flags, 0644);
  if (f.fd < 0) throw std::runtime_error(os_error("opening", path, errno));
#endif
  std::lock_guard<std::mutex> lock(impl_->mu);
  for (size_t i = 0; i < impl_->files.size(); ++i) {
    if (!impl_->files[i].used) {
      impl_->files[i] = f;
      return static_cast<int>(i);
    }
  }
  impl_->files.push_back(f);
  return static_cast<int>(impl_->files.size() - 1);
}

void Engine::close(int file) {
  File f;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (file < 0 || file >= static_cast<int>(impl_->files.size()) || !impl_->files[static_cast<size_t>(file)].used)
      return;
    f = impl_->files[static_cast<size_t>(file)];
    impl_->files[static_cast<size_t>(file)] = File();
  }
#ifdef _WIN32
  CloseHandle(f.h);
#else
  ::close(f.fd);
#endif
}

bool Engine::direct(int file) { return impl_->file_of(file).direct; }

int64_t Engine::size(int file) {
  const File f = impl_->file_of(file);
#ifdef _WIN32
  LARGE_INTEGER n;
  if (!GetFileSizeEx(f.h, &n)) throw std::runtime_error(os_error("sizing", f.path, GetLastError()));
  return static_cast<int64_t>(n.QuadPart);
#else
  struct stat st;
  if (fstat(f.fd, &st) != 0) throw std::runtime_error(os_error("sizing", f.path, errno));
  return static_cast<int64_t>(st.st_size);
#endif
}

void Engine::flush(int file) {
  const File f = impl_->file_of(file);
#ifdef _WIN32
  if (!FlushFileBuffers(f.h)) throw std::runtime_error(os_error("flushing", f.path, GetLastError()));
#else
  int r;
  do {
#if defined(__linux__)
    r = fdatasync(f.fd);
#else
    r = fsync(f.fd);
#endif
  } while (r != 0 && errno == EINTR);
  if (r != 0) throw std::runtime_error(os_error("flushing", f.path, errno));
#endif
}

void Engine::truncate(int file, int64_t bytes) {
  const File f = impl_->file_of(file);
#ifdef _WIN32
  // Through an ordinary handle of its own: an unbuffered one only takes whole sectors.
  const int wlen = MultiByteToWideChar(CP_UTF8, 0, f.path.c_str(), -1, nullptr, 0);
  std::wstring wpath(static_cast<size_t>(std::max(wlen, 1)), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, f.path.c_str(), -1, &wpath[0], wlen);
  HANDLE h = CreateFileW(wpath.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) throw std::runtime_error(os_error("opening", f.path, GetLastError()));
  LARGE_INTEGER pos;
  pos.QuadPart = bytes;
  const BOOL ok = SetFilePointerEx(h, pos, nullptr, FILE_BEGIN) && SetEndOfFile(h);
  const DWORD e = GetLastError();
  CloseHandle(h);
  if (!ok) throw std::runtime_error(os_error("resizing", f.path, e));
#else
  if (ftruncate(f.fd, static_cast<off_t>(bytes)) != 0) throw std::runtime_error(os_error("resizing", f.path, errno));
#endif
}

std::shared_ptr<Batch> Engine::submit(const std::vector<Request>& requests) {
  auto batch = std::make_shared<Batch>();
  batch->t0_ = batch->t1_ = now_s();
  std::vector<Op*> ops;
  for (const Request& r : requests) {
    const File f = impl_->file_of(r.file);
    if (r.write && !f.writable) throw std::runtime_error("TrainStream: '" + f.path + "' was opened read-only");
    if (f.direct && (r.offset % static_cast<int64_t>(kAlign) != 0 || r.len % kAlign != 0 ||
                     reinterpret_cast<uintptr_t>(r.buf) % kAlign != 0))
      throw std::runtime_error("TrainStream: unbuffered I/O on '" + f.path +
                               "' needs offsets, lengths and buffers in multiples of 4096 bytes");
    for (size_t at = 0; at < r.len; at += kChunk) {
      Op* op = new Op();
      op->batch = batch;
      op->file = r.file;
      op->offset = r.offset + static_cast<int64_t>(at);
      op->len = std::min(kChunk, r.len - at);
      op->buf = r.buf + at;
      op->write = r.write;
      ops.push_back(op);
    }
  }
  if (ops.empty()) return batch;
  batch->pending_ = static_cast<int>(ops.size());  // nothing can complete before the queue sees them
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->outstanding == 0) impl_->busy_since = now_s();
    impl_->outstanding += static_cast<int>(ops.size());
    for (Op* op : ops) impl_->queue.push_back(op);
  }
  impl_->cv.notify_all();
  return batch;
}

std::map<std::string, double> Engine::stats() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  int open_files = 0, direct_files = 0;
  for (const File& f : impl_->files) {
    open_files += f.used;
    direct_files += f.used && f.direct;
  }
  double busy = impl_->busy_total;
  if (impl_->outstanding > 0) busy += now_s() - impl_->busy_since;
  return {{"files_open", static_cast<double>(open_files)},
          {"files_direct", static_cast<double>(direct_files)},
          {"ops", static_cast<double>(impl_->ops.load())},
          {"bytes_read", static_cast<double>(impl_->bytes_read.load())},
          {"bytes_written", static_cast<double>(impl_->bytes_written.load())},
          {"busy_ms", busy * 1e3},
          {"queued", static_cast<double>(impl_->outstanding)}};
}

#ifdef _WIN32
namespace {
std::wstring widen(const std::string& s) {
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(static_cast<size_t>(std::max(n, 1)), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
  return w;
}
}  // namespace
#endif

void replace_file(const std::string& from, const std::string& to) {
#ifdef _WIN32
  if (!MoveFileExW(widen(from).c_str(), widen(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error(os_error("renaming", from, GetLastError()));
#else
  if (std::rename(from.c_str(), to.c_str()) != 0) throw std::runtime_error(os_error("renaming", from, errno));
  // the new name itself must survive a crash: flush the directory entry
  const size_t slash = to.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "." : (slash == 0 ? "/" : to.substr(0, slash));
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if (fd >= 0) {
    fsync(fd);
    ::close(fd);
  }
#endif
}

void remove_file(const std::string& path) {
#ifdef _WIN32
  DeleteFileW(widen(path).c_str());
#else
  ::unlink(path.c_str());
#endif
}

void* aligned_alloc_io(size_t bytes) {
  bytes = align_up(std::max<size_t>(bytes, 1));
#ifdef _WIN32
  void* p = _aligned_malloc(bytes, kAlign);
#else
  void* p = nullptr;
  if (posix_memalign(&p, kAlign, bytes) != 0) p = nullptr;
#endif
  if (p == nullptr) throw std::bad_alloc();
  return p;
}

void aligned_free_io(void* p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  std::free(p);
#endif
}

}  // namespace aio
}  // namespace trainstream
