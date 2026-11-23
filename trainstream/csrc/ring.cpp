// TrainStream — ring all-reduce of host gradients over TCP.
//
// Data-parallel ranks on different machines sum each gradient with a ring:
// rank r has one connection to r + 1 and one from r - 1. The vector is cut
// into `world` segments and summed in two passes of world - 1 steps each:
//
//   reduce-scatter  step s: r sends its (partly summed) segment r - s to r + 1,
//                   which adds it into its own copy. Afterwards rank r holds
//                   the complete sum of segment r + 1.
//   all-gather      step s: r passes the complete segment r + 1 - s on to r + 1.
//
// Every rank sends 2 (world - 1) / world of the vector in total, whatever the
// number of ranks (the bandwidth-optimal schedule), and sends and receives at
// the same time in every step. Each segment's additions happen once, on the
// ranks it passes, and the finished sums are only copied afterwards: every
// rank ends with the same bits, so replicas stay identical however many there
// are.
//
// Only the operating system's sockets are used (Winsock / BSD); the
// connections are made in Python (trainstream/ring.py) and handed over here.
// A header in front of every segment carries a sequence number, so two ranks
// that disagree about which gradient is being reduced fail at once instead of
// summing the wrong things.

#ifdef _WIN32  // before anything that may include <windows.h> (which would bring in the old winsock.h)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <cerrno>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "trainstream.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace trainstream {

namespace {

#ifdef _WIN32
using sock_t = SOCKET;
int send_some(sock_t s, const char* p, size_t n) {
  return ::send(s, p, static_cast<int>(std::min<size_t>(n, 1 << 30)), 0);
}
int recv_some(sock_t s, char* p, size_t n) { return ::recv(s, p, static_cast<int>(std::min<size_t>(n, 1 << 30)), 0); }
std::string sock_error() { return "Winsock error " + std::to_string(WSAGetLastError()); }
bool interrupted() { return WSAGetLastError() == WSAEINTR; }
void close_sock(sock_t s) { ::closesocket(s); }
void set_timeouts(sock_t s, int seconds) {
  const DWORD ms = static_cast<DWORD>(seconds) * 1000;
  ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
  ::setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
}
#else
using sock_t = int;
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
ssize_t send_some(sock_t s, const char* p, size_t n) { return ::send(s, p, n, MSG_NOSIGNAL); }
ssize_t recv_some(sock_t s, char* p, size_t n) { return ::recv(s, p, n, 0); }
std::string sock_error() { return std::strerror(errno); }
bool interrupted() { return errno == EINTR; }
void close_sock(sock_t s) { ::close(s); }
void set_timeouts(sock_t s, int seconds) {
  timeval tv;
  tv.tv_sec = seconds;
  tv.tv_usec = 0;
  ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
#endif

void send_all(sock_t s, const char* p, size_t n) {
  while (n > 0) {
    const auto k = send_some(s, p, n);
    if (k <= 0) {
      if (k < 0 && interrupted()) continue;
      throw std::runtime_error("ring all-reduce: sending to the next rank failed (" + sock_error() + ")");
    }
    p += k;
    n -= static_cast<size_t>(k);
  }
}

void recv_all(sock_t s, char* p, size_t n) {
  while (n > 0) {
    const auto k = recv_some(s, p, n);
    if (k == 0) throw std::runtime_error("ring all-reduce: the previous rank closed its connection");
    if (k < 0) {
      if (interrupted()) continue;
      throw std::runtime_error("ring all-reduce: receiving from the previous rank failed or timed out (" +
                               sock_error() + ")");
    }
    p += k;
    n -= static_cast<size_t>(k);
  }
}

struct Header {
  uint32_t magic;   // also tells a byte-order mismatch apart
  uint32_t step;    // 0 .. 2 (world - 1) - 1
  uint64_t seq;     // which all-reduce
  uint64_t nbytes;  // of the segment that follows
};
constexpr uint32_t kMagic = 0x54535247u;  // "TSRG"
constexpr size_t kChunk = size_t(1) << 20;
}  // namespace

struct Ring::Impl {
  sock_t to_next, from_prev;
  int rank, world;
  uint64_t seq = 0;
  std::vector<char> scratch;
  std::atomic<int64_t> calls{0}, bytes_sent{0}, bytes_received{0}, wall_us{0};

  // The sender: one job at a time, so a step's send runs beside its receive.
  std::thread sender;
  std::mutex mu;
  std::condition_variable cv;
  bool stop = false, busy = false;
  Header job_header{};
  const char* job_data = nullptr;
  std::string send_error;

  void sender_loop() {
    for (;;) {
      Header h;
      const char* data;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this] { return stop || busy; });
        if (stop) return;
        h = job_header;
        data = job_data;
      }
      std::string err;
      try {
        send_all(to_next, reinterpret_cast<const char*>(&h), sizeof(h));
        send_all(to_next, data, static_cast<size_t>(h.nbytes));
        bytes_sent += static_cast<int64_t>(h.nbytes);
      } catch (const std::exception& e) {
        err = e.what();
      }
      {
        std::lock_guard<std::mutex> lock(mu);
        busy = false;
        if (!err.empty() && send_error.empty()) send_error = err;
      }
      cv.notify_all();
    }
  }

  void start_send(uint32_t step, const char* data, size_t nbytes) {
    {
      std::lock_guard<std::mutex> lock(mu);
      job_header = Header{kMagic, step, seq, static_cast<uint64_t>(nbytes)};
      job_data = data;
      busy = true;
    }
    cv.notify_all();
  }

  void finish_send() {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [this] { return !busy; });
    if (!send_error.empty()) {
      const std::string err = send_error;
      send_error.clear();
      throw std::runtime_error(err);
    }
  }

  void expect(uint32_t step, size_t nbytes) {
    Header h;
    recv_all(from_prev, reinterpret_cast<char*>(&h), sizeof(h));
    if (h.magic != kMagic || h.step != step || h.seq != seq || h.nbytes != nbytes)
      throw std::runtime_error(
          "ring all-reduce: rank " + std::to_string(rank) + " and the rank before it disagree about what is being "
          "reduced (reduction " + std::to_string(seq) + " step " + std::to_string(step) + " of " +
          std::to_string(nbytes) + " bytes here; " + std::to_string(h.seq) + " / " + std::to_string(h.step) + " / " +
          std::to_string(h.nbytes) + " there). Every rank must reduce the same tensors in the same order.");
  }
};

Ring::Ring(int64_t to_next, int64_t from_prev, int rank, int world, int timeout_s) : impl_(new Impl()) {
  TORCH_CHECK(world >= 2 && rank >= 0 && rank < world, "Ring: bad rank ", rank, " of ", world);
  impl_->to_next = static_cast<sock_t>(to_next);
  impl_->from_prev = static_cast<sock_t>(from_prev);
  impl_->rank = rank;
  impl_->world = world;
  impl_->scratch.resize(kChunk);
  set_timeouts(impl_->to_next, timeout_s);
  set_timeouts(impl_->from_prev, timeout_s);
  impl_->sender = std::thread([this] { impl_->sender_loop(); });
}

Ring::~Ring() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stop = true;
  }
  impl_->cv.notify_all();
  // Unblock a send still in flight, then join.
  close_sock(impl_->to_next);
  close_sock(impl_->from_prev);
  if (impl_->sender.joinable()) impl_->sender.join();
  delete impl_;
}

void Ring::allreduce(const at::Tensor& buf) {
  TORCH_CHECK(buf.device().is_cpu() && buf.scalar_type() == at::kFloat && buf.is_contiguous(),
              "ring all-reduce: a contiguous float32 host tensor");
  Impl& m = *impl_;
  const double t0 = now_ms();
  m.seq += 1;
  m.calls += 1;
  const int N = m.world, r = m.rank;
  const int64_t n = buf.numel();
  float* x = buf.data_ptr<float>();
  auto seg_begin = [&](int j) { return n * j / N; };
  auto mod = [&](int j) { return ((j % N) + N) % N; };

  uint32_t step = 0;
  // reduce-scatter
  for (int s = 0; s < N - 1; ++s, ++step) {
    const int out = mod(r - s), in = mod(r - 1 - s);
    const int64_t ob = seg_begin(out), oe = seg_begin(out + 1), ib = seg_begin(in), ie = seg_begin(in + 1);
    m.start_send(step, reinterpret_cast<const char*>(x + ob), static_cast<size_t>(oe - ob) * sizeof(float));
    try {
      m.expect(step, static_cast<size_t>(ie - ib) * sizeof(float));
      for (int64_t at_elem = ib; at_elem < ie;) {
        const int64_t take = std::min<int64_t>(ie - at_elem, static_cast<int64_t>(kChunk / sizeof(float)));
        recv_all(m.from_prev, m.scratch.data(), static_cast<size_t>(take) * sizeof(float));
        const float* in_part = reinterpret_cast<const float*>(m.scratch.data());
        float* mine = x + at_elem;
        for (int64_t i = 0; i < take; ++i) mine[i] = in_part[i] + mine[i];
        at_elem += take;
      }
      m.bytes_received += (ie - ib) * static_cast<int64_t>(sizeof(float));
    } catch (...) {
      try {
        m.finish_send();
      } catch (...) {
      }
      throw;
    }
    m.finish_send();
  }
  // all-gather
  for (int s = 0; s < N - 1; ++s, ++step) {
    const int out = mod(r + 1 - s), in = mod(r - s);
    const int64_t ob = seg_begin(out), oe = seg_begin(out + 1), ib = seg_begin(in), ie = seg_begin(in + 1);
    m.start_send(step, reinterpret_cast<const char*>(x + ob), static_cast<size_t>(oe - ob) * sizeof(float));
    try {
      m.expect(step, static_cast<size_t>(ie - ib) * sizeof(float));
      recv_all(m.from_prev, reinterpret_cast<char*>(x + ib), static_cast<size_t>(ie - ib) * sizeof(float));
      m.bytes_received += (ie - ib) * static_cast<int64_t>(sizeof(float));
    } catch (...) {
      try {
        m.finish_send();
      } catch (...) {
      }
      throw;
    }
    m.finish_send();
  }
  m.wall_us += static_cast<int64_t>((now_ms() - t0) * 1e3);
}

std::map<std::string, double> Ring::stats() const {
  return {{"calls", static_cast<double>(impl_->calls.load())},
          {"bytes_sent", static_cast<double>(impl_->bytes_sent.load())},
          {"bytes_received", static_cast<double>(impl_->bytes_received.load())},
          {"wall_ms", impl_->wall_us.load() / 1e3},
          {"rank", static_cast<double>(impl_->rank)},
          {"world", static_cast<double>(impl_->world)}};
}

}  // namespace trainstream
