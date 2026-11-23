// TrainStream — offloaded AdamW engine (host worker pool, gradient pipeline,
// CPU AdamW, zero-copy GPU dispatch) and optimizer micro-benchmarks.

#include "offload_adamw.h"

#include "kernels.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace trainstream {

namespace {

kernels::DType to_kernel_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
      return kernels::DType::kFloat32;
    case at::kHalf:
      return kernels::DType::kFloat16;
    case at::kBFloat16:
      return kernels::DType::kBFloat16;
    default:
      TORCH_CHECK(false, "TrainStream supports float32/float16/bfloat16 parameters, got ", t);
  }
}

struct HostHp {
  float lr, beta1, beta2, eps, weight_decay, step_size, bc2_sqrt, clip;
};

HostHp make_hp(double lr, double b1, double b2, double eps, double wd, int64_t step, double clip) {
  const double bc1 = 1.0 - std::pow(b1, static_cast<double>(step));
  const double bc2 = 1.0 - std::pow(b2, static_cast<double>(step));
  return {static_cast<float>(lr),           static_cast<float>(b1),
          static_cast<float>(b2),           static_cast<float>(eps),
          static_cast<float>(wd),           static_cast<float>(lr / bc1),
          static_cast<float>(std::sqrt(bc2)), static_cast<float>(clip)};
}

// Counter-based random bits for stochastic rounding (same mixer as the kernel).
inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

inline uint32_t slot_seed(int64_t id, int64_t step) {
  return mix32(static_cast<uint32_t>(id) * 0x85ebca6bu ^ mix32(static_cast<uint32_t>(step)));
}

// Moment storage: fp32 as-is, or bf16 written with stochastic rounding so the
// small per-step changes of exp_avg_sq survive (round-to-nearest would freeze
// it once (1 - beta2) * g^2 is below half a bf16 ulp of v).
template <typename M>
struct MomentIO;

template <>
struct MomentIO<float> {
  static float load(float x) { return x; }
  static float store(float x, uint32_t) { return x; }
};

template <>
struct MomentIO<c10::BFloat16> {
  static float load(c10::BFloat16 x) { return static_cast<float>(x); }
  static c10::BFloat16 store(float x, uint32_t r) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) bits += r & 0xffffu;  // inf/nan pass through
    return c10::BFloat16(static_cast<uint16_t>(bits >> 16), c10::BFloat16::from_bits());
  }
};

// Host AdamW over [b, e). Same arithmetic order as torch.optim.AdamW
// (single-tensor path): decoupled decay, lerp for m, addcmul for v,
// denom = sqrt(v)/sqrt(bc2) + eps, p -= (lr/bc1) * m/denom.
// G is the gradient element type (fp32 accumulator or the raw DMA staging
// buffer); with kStage the rounded weights are also written for the upload.
// The loop is branch-free so the compiler can vectorise it.
template <typename G, typename P, bool kStage, typename M>
void adamw_host(float* __restrict p, M* __restrict m, M* __restrict v,
                const G* __restrict g, P* __restrict stage, int64_t b, int64_t e,
                const HostHp& hp, uint32_t seed) {
  const float decay = 1.0f - hp.lr * hp.weight_decay;
  const float omb1 = 1.0f - hp.beta1;
  const float omb2 = 1.0f - hp.beta2;
  for (int64_t i = b; i < e; ++i) {
    const float gi = static_cast<float>(g[i]) * hp.clip;
    const float m0 = MomentIO<M>::load(m[i]);
    const float mi = m0 + omb1 * (gi - m0);
    const float vi = hp.beta2 * MomentIO<M>::load(v[i]) + omb2 * gi * gi;
    const float denom = std::sqrt(vi) / hp.bc2_sqrt + hp.eps;
    const float pi = p[i] * decay - hp.step_size * (mi / denom);
    if constexpr (std::is_same_v<M, float>) {
      m[i] = mi;
      v[i] = vi;
    } else {
      const uint32_t r = mix32(static_cast<uint32_t>(i) * 0x9e3779b1u ^ seed);
      m[i] = MomentIO<M>::store(mi, r);
      v[i] = MomentIO<M>::store(vi, mix32(r ^ 0x85ebca6bu));
    }
    p[i] = pi;
    if constexpr (kStage) stage[i] = static_cast<P>(pi);
  }
}

template <typename G, typename M>
void adamw_dispatch_param(float* p, M* m, M* v, const G* g, at::ScalarType pdtype,
                          const at::Tensor& param_stage, int64_t b, int64_t e, const HostHp& hp,
                          uint32_t seed) {
  switch (pdtype) {
    case at::kFloat:
      adamw_host<G, float, false, M>(p, m, v, g, nullptr, b, e, hp, seed);
      break;
    case at::kHalf:
      adamw_host<G, at::Half, true, M>(p, m, v, g, param_stage.data_ptr<at::Half>(), b, e, hp, seed);
      break;
    case at::kBFloat16:
      adamw_host<G, at::BFloat16, true, M>(p, m, v, g, param_stage.data_ptr<at::BFloat16>(), b, e, hp, seed);
      break;
    default:
      TORCH_CHECK(false, "unsupported parameter dtype");
  }
}

template <typename M>
void adamw_dispatch_m(float* p, M* m, M* v, const at::Tensor& grad, at::ScalarType pdtype,
                      const at::Tensor& param_stage, int64_t b, int64_t e, const HostHp& hp, uint32_t seed) {
  switch (grad.scalar_type()) {
    case at::kFloat:
      adamw_dispatch_param(p, m, v, grad.data_ptr<float>(), pdtype, param_stage, b, e, hp, seed);
      break;
    case at::kHalf:
      adamw_dispatch_param(p, m, v, grad.data_ptr<at::Half>(), pdtype, param_stage, b, e, hp, seed);
      break;
    case at::kBFloat16:
      adamw_dispatch_param(p, m, v, grad.data_ptr<at::BFloat16>(), pdtype, param_stage, b, e, hp, seed);
      break;
    default:
      TORCH_CHECK(false, "unsupported gradient dtype");
  }
}

void adamw_dispatch(float* p, const at::Tensor& m, const at::Tensor& v, const at::Tensor& grad,
                    at::ScalarType pdtype, const at::Tensor& param_stage, int64_t b, int64_t e,
                    const HostHp& hp, uint32_t seed) {
  if (m.scalar_type() == at::kBFloat16)
    adamw_dispatch_m(p, m.data_ptr<c10::BFloat16>(), v.data_ptr<c10::BFloat16>(), grad, pdtype, param_stage, b,
                     e, hp, seed);
  else
    adamw_dispatch_m(p, m.data_ptr<float>(), v.data_ptr<float>(), grad, pdtype, param_stage, b, e, hp, seed);
}

// 8-bit moments ("int8"): per block of kQBlock values an fp32 absmax, and one byte per value,
// a small float relative to it written with stochastic rounding (unbiased, so the tiny per-step
// changes of exp_avg_sq accumulate instead of rounding away). exp_avg_sq is stored as its square
// root u (its range is that of the gradients, not of their squares), and a positive u never
// rounds to zero (that would divide m by eps alone):
//   u (>= 0)   4-bit exponent f, 4-bit mantissa q: u / absmax = (1 + q/16) 2^(f - 15), f = 0 zero
//   m (signed) sign bit, 3-bit exponent f, 4-bit mantissa q: |m| / absmax = (1 + q/16) 2^(f - 7),
//              rounded to nearest
// 15 octaves below the block's largest u, 7 below its largest |m| (smaller m round
// stochastically to 0 or the smallest step). The 2 bytes per parameter replace 8 (fp32) or 4 (bf16).
constexpr int64_t kQBlock = 256;

// exp_bits-bit exponent field f (0: zero; bias 2^exp_bits - 1, so 2^exp_bits - 1 octaves below
// absmax) and mant_bits-bit mantissa q: value = absmax (1 + q / 2^mant_bits) 2^(f - bias)
inline float q8_decode(uint8_t c, int mant_bits, float absmax, int exp_bits = 4) {
  const uint32_t bias = (1u << exp_bits) - 1u;
  const uint32_t f = (c >> mant_bits) & bias;
  if (f == 0) return 0.0f;
  const uint32_t q = c & ((1u << mant_bits) - 1u);
  const uint32_t bits = ((f - bias + 127u) << 23) | (q << (23 - mant_bits));
  float y;
  std::memcpy(&y, &bits, sizeof(y));
  return y * absmax;
}

// y = |x| / absmax in [0, 1]; r: random bits for the stochastic rounding; keep: a positive y
// below the smallest step rounds up to it instead of (stochastically) to zero
inline uint8_t q8_encode(float y, int mant_bits, uint32_t r, bool keep = false, int exp_bits = 4) {
  const int bias = (1 << exp_bits) - 1;
  const float kMin = std::ldexp(1.0f, 1 - bias);  // f = 1, q = 0
  if (!(y > 0.0f)) return 0;
  if (y < kMin)
    return keep || (static_cast<float>(r >> 8) * (1.0f / 16777216.0f)) < y / kMin ? (1u << mant_bits) : 0;
  uint32_t bits;
  std::memcpy(&bits, &y, sizeof(bits));
  const int cut = 23 - mant_bits;
  bits += r & ((1u << cut) - 1u);  // carries into the exponent when the mantissa overflows
  int f = static_cast<int>((bits >> 23) & 0xffu) - 127 + bias;
  if (f > bias) return static_cast<uint8_t>((static_cast<uint32_t>(bias) << mant_bits) | ((1u << mant_bits) - 1u));
  const uint32_t q = (bits >> cut) & ((1u << mant_bits) - 1u);
  return static_cast<uint8_t>((static_cast<uint32_t>(f) << mant_bits) | q);
}

inline float m8_decode(uint8_t c, float absmax) {
  const float y = q8_decode(c & 0x7fu, 4, absmax, 3);
  return (c & 0x80u) ? -y : y;
}

// m rounds to nearest: unlike v it moves by large steps, and stochastic rounding's extra noise would
// go straight into the update
inline uint8_t m8_encode(float x, float absmax, uint32_t) {
  constexpr uint32_t kHalf = (1u << (23 - 4 - 1)) | 0x80000000u;  // half a step; >> 8 gives 1/2 for tiny values
  const uint8_t c = q8_encode(absmax > 0.0f ? std::fabs(x) / absmax : 0.0f, 4, kHalf, false, 3);
  return x < 0.0f ? static_cast<uint8_t>(c | 0x80u) : c;
}

// Host AdamW over [b, e) with 8-bit moments; b is a multiple of kQBlock. The same arithmetic as
// adamw_host on the decoded moments; each block is re-encoded against its new absmax.
template <typename G, typename P, bool kStage>
void adamw_host8(float* __restrict p, uint8_t* __restrict m, uint8_t* __restrict v, float* __restrict ms,
                 float* __restrict vs, const G* __restrict g, P* __restrict stage, int64_t b, int64_t e,
                 const HostHp& hp, uint32_t seed) {
  const float decay = 1.0f - hp.lr * hp.weight_decay;
  const float omb1 = 1.0f - hp.beta1;
  const float omb2 = 1.0f - hp.beta2;
  float mb[kQBlock], vb[kQBlock];
  for (int64_t a = b; a < e; a += kQBlock) {
    const int64_t z = std::min(e, a + kQBlock), k = a / kQBlock;
    const float m_old = ms[k], v_old = vs[k];
    float m_max = 0.0f, v_max = 0.0f;
    for (int64_t i = a; i < z; ++i) {
      const float gi = static_cast<float>(g[i]) * hp.clip;
      const float m0 = m8_decode(m[i], m_old);
      const float mi = m0 + omb1 * (gi - m0);
      const float u0 = q8_decode(v[i], 4, v_old);
      const float vi = hp.beta2 * (u0 * u0) + omb2 * gi * gi;
      const float ui = std::sqrt(vi);
      const float denom = ui / hp.bc2_sqrt + hp.eps;
      const float pi = p[i] * decay - hp.step_size * (mi / denom);
      mb[i - a] = mi;
      vb[i - a] = ui;
      m_max = std::max(m_max, std::fabs(mi));
      v_max = std::max(v_max, ui);
      p[i] = pi;
      if constexpr (kStage) stage[i] = static_cast<P>(pi);
    }
    ms[k] = m_max;
    vs[k] = v_max;
    for (int64_t i = a; i < z; ++i) {
      const uint32_t r = mix32(static_cast<uint32_t>(i) * 0x9e3779b1u ^ seed);
      m[i] = m8_encode(mb[i - a], m_max, r);
      v[i] = q8_encode(v_max > 0.0f ? vb[i - a] / v_max : 0.0f, 4, mix32(r ^ 0x85ebca6bu), true);
    }
  }
}

template <typename G>
void adamw8_dispatch_param(float* p, uint8_t* m, uint8_t* v, float* ms, float* vs, const G* g, at::ScalarType pdtype,
                           const at::Tensor& param_stage, int64_t b, int64_t e, const HostHp& hp, uint32_t seed) {
  switch (pdtype) {
    case at::kFloat:
      adamw_host8<G, float, false>(p, m, v, ms, vs, g, nullptr, b, e, hp, seed);
      break;
    case at::kHalf:
      adamw_host8<G, at::Half, true>(p, m, v, ms, vs, g, param_stage.data_ptr<at::Half>(), b, e, hp, seed);
      break;
    case at::kBFloat16:
      adamw_host8<G, at::BFloat16, true>(p, m, v, ms, vs, g, param_stage.data_ptr<at::BFloat16>(), b, e, hp, seed);
      break;
    default:
      TORCH_CHECK(false, "unsupported parameter dtype");
  }
}

void adamw8_dispatch(float* p, const at::Tensor& m, const at::Tensor& v, const at::Tensor& ms, const at::Tensor& vs,
                     const at::Tensor& grad, at::ScalarType pdtype, const at::Tensor& param_stage, int64_t b,
                     int64_t e, const HostHp& hp, uint32_t seed) {
  uint8_t* mp = m.data_ptr<uint8_t>();
  uint8_t* vp = v.data_ptr<uint8_t>();
  float* msp = ms.data_ptr<float>();
  float* vsp = vs.data_ptr<float>();
  switch (grad.scalar_type()) {
    case at::kFloat:
      adamw8_dispatch_param(p, mp, vp, msp, vsp, grad.data_ptr<float>(), pdtype, param_stage, b, e, hp, seed);
      break;
    case at::kHalf:
      adamw8_dispatch_param(p, mp, vp, msp, vsp, grad.data_ptr<at::Half>(), pdtype, param_stage, b, e, hp, seed);
      break;
    case at::kBFloat16:
      adamw8_dispatch_param(p, mp, vp, msp, vsp, grad.data_ptr<at::BFloat16>(), pdtype, param_stage, b, e, hp, seed);
      break;
    default:
      TORCH_CHECK(false, "unsupported gradient dtype");
  }
}

// fp32 moments of a slot with 8-bit storage, and back (state_dict / load_state_dict)
at::Tensor decode8(const at::Tensor& codes, const at::Tensor& scale, bool is_m) {
  at::Tensor out = at::empty({codes.numel()}, at::TensorOptions().dtype(at::kFloat));
  const uint8_t* c = codes.data_ptr<uint8_t>();
  const float* s = scale.data_ptr<float>();
  float* o = out.data_ptr<float>();
  for (int64_t i = 0; i < codes.numel(); ++i) {
    const float u = is_m ? 0.0f : q8_decode(c[i], 4, s[i / kQBlock]);
    o[i] = is_m ? m8_decode(c[i], s[i / kQBlock]) : u * u;  // exp_avg_sq is stored as its square root
  }
  return out;
}

void encode8(const at::Tensor& values, at::Tensor& codes, at::Tensor& scale, bool is_m) {
  const at::Tensor x = is_m ? values.reshape({-1}).to(at::kFloat).contiguous()
                            : values.reshape({-1}).to(at::kFloat).clamp_min(0).sqrt().contiguous();
  const float* xs = x.data_ptr<float>();
  uint8_t* c = codes.data_ptr<uint8_t>();
  float* s = scale.data_ptr<float>();
  const int64_t n = x.numel();
  for (int64_t a = 0, k = 0; a < n; a += kQBlock, ++k) {
    const int64_t z = std::min(n, a + kQBlock);
    float mx = 0.0f;
    for (int64_t i = a; i < z; ++i) mx = std::max(mx, std::fabs(xs[i]));
    s[k] = mx;
    for (int64_t i = a; i < z; ++i) {
      const uint32_t r = mix32(static_cast<uint32_t>(i) * 0x27d4eb2fu);
      c[i] = is_m ? m8_encode(xs[i], mx, r) : q8_encode(mx > 0.0f ? xs[i] / mx : 0.0f, 4, r, true);
    }
  }
}

// True when any of x[0, n) is NaN or +/-Inf. Integer arithmetic only: it
// vectorises, and no floating-point compiler mode can fold the test away.
inline bool any_non_finite(const float* x, int64_t n) {
  uint32_t m = 0;
  for (int64_t i = 0; i < n; ++i) {
    uint32_t bits;
    std::memcpy(&bits, x + i, sizeof(bits));
    m |= (bits & 0x7f800000u) + 0x00800000u;  // carries into bit 31 only for an all-ones exponent
  }
  return (m >> 31) != 0;
}

// acc[b, e) = or += g[b, e). With `check`, returns whether the accumulated
// values hold a NaN / Inf; they are tested a few KiB at a time, right after
// they were written, while still in the core's cache.
template <typename T>
bool accumulate_host(float* __restrict acc, const T* __restrict g, int64_t b, int64_t e, bool fresh,
                     bool check = false) {
  if (!check) {
    if (fresh) {
      for (int64_t i = b; i < e; ++i) acc[i] = static_cast<float>(g[i]);
    } else {
      for (int64_t i = b; i < e; ++i) acc[i] += static_cast<float>(g[i]);
    }
    return false;
  }
  constexpr int64_t kSpan = 4096;
  bool bad = false;
  for (int64_t a = b; a < e; a += kSpan) {
    const int64_t z = std::min(e, a + kSpan);
    if (fresh) {
      for (int64_t i = a; i < z; ++i) acc[i] = static_cast<float>(g[i]);
    } else {
      for (int64_t i = a; i < z; ++i) acc[i] += static_cast<float>(g[i]);
    }
    bad |= any_non_finite(acc + a, z - a);
  }
  return bad;
}

// The value of every e4m3 code (sign, 4-bit exponent biased by 7, 3-bit mantissa).
const float* e4m3_table() {
  static const std::vector<float> table = [] {
    std::vector<float> t(256);
    for (int c = 0; c < 128; ++c) {
      const int e = c >> 3, m = c & 7;
      const float v = e == 0 ? std::ldexp(static_cast<float>(m) / 8.0f, -6)
                             : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
      t[static_cast<size_t>(c)] = v;
      t[static_cast<size_t>(c | 0x80)] = -v;
    }
    return t;
  }();
  return table.data();
}

// acc[b, e) = or += the fp8-coded gradient (b is a multiple of kQBlock). A block with a NaN
// scale (the GPU saw a NaN / Inf in it) expands to NaN, which `check` reports like any other.
bool accumulate_host_fp8(float* __restrict acc, const uint8_t* __restrict codes, const float* __restrict scales,
                         int64_t b, int64_t e, bool fresh, bool check) {
  const float* tab = e4m3_table();
  bool bad = false;
  for (int64_t a = b; a < e; a += kQBlock) {
    const int64_t z = std::min(e, a + kQBlock);
    const float s = scales[a / kQBlock];
    if (fresh) {
      for (int64_t i = a; i < z; ++i) acc[i] = tab[codes[i]] * s;
    } else {
      for (int64_t i = a; i < z; ++i) acc[i] += tab[codes[i]] * s;
    }
    if (check) bad |= any_non_finite(acc + a, z - a);
  }
  return bad;
}

int64_t elapsed_us(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

// ===========================================================================
// WorkerPool
// ===========================================================================
WorkerPool::WorkerPool(int threads, int device) {
  threads = std::max(1, threads);
  for (int t = 0; t < threads; ++t) {
    auto w = std::make_unique<Worker>();
    Worker* wp = w.get();
    w->thread = std::thread([wp, device] {
      cudaSetDevice(device);  // CUDA's current device is per host thread
      for (;;) {
        std::function<void()> job;
        {
          std::unique_lock<std::mutex> lock(wp->mu);
          wp->cv.wait(lock, [wp] { return wp->stop || !wp->queue.empty(); });
          if (wp->queue.empty()) return;  // stop requested and drained
          job = std::move(wp->queue.front());
          wp->queue.pop_front();
        }
        job();
      }
    });
    workers_.push_back(std::move(w));
  }
}

WorkerPool::~WorkerPool() {
  for (auto& w : workers_) {
    {
      std::lock_guard<std::mutex> lock(w->mu);
      w->stop = true;
    }
    w->cv.notify_all();
  }
  for (auto& w : workers_)
    if (w->thread.joinable()) w->thread.join();
}

void WorkerPool::submit(size_t worker, std::function<void()> fn) {
  Worker& w = *workers_[worker % workers_.size()];
  {
    std::lock_guard<std::mutex> lock(w.mu);
    w.queue.push_back(std::move(fn));
  }
  w.cv.notify_one();
}

// ===========================================================================
// StepGate
// ===========================================================================
void OffloadAdamW::StepGate::enter() {
  std::lock_guard<std::mutex> lock(mu);
  ++open;
}

void OffloadAdamW::StepGate::leave() {
  {
    std::lock_guard<std::mutex> lock(mu);
    --open;
  }
  cv.notify_all();
}

void OffloadAdamW::StepGate::seal() {
  {
    std::lock_guard<std::mutex> lock(mu);
    sealed = true;
  }
  cv.notify_all();
}

bool OffloadAdamW::StepGate::skip() {
  int v = verdict.load(std::memory_order_acquire);
  if (v == 0) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [this] { return sealed && open == 0; });
    v = bad.load() ? 2 : 1;
    verdict.store(v, std::memory_order_release);
  }
  return v == 2;
}

// ===========================================================================
// OffloadAdamW
// ===========================================================================
OffloadAdamW::OffloadAdamW(int device, const std::string& mode, int threads, int64_t chunk_elems,
                           const std::string& moment_dtype)
    : device_(resolve_device(device)), chunk_((std::max<int64_t>(chunk_elems, 4096) + kQBlock - 1) / kQBlock * kQBlock) {
  TORCH_CHECK(mode == "cpu" || mode == "gpu", "OffloadAdamW mode must be 'cpu' or 'gpu', got '", mode, "'");
  TORCH_CHECK(moment_dtype == "float32" || moment_dtype == "bfloat16" || moment_dtype == "int8",
              "OffloadAdamW moment_dtype must be 'float32', 'bfloat16' or 'int8', got '", moment_dtype, "'");
  TORCH_CHECK(moment_dtype != "int8" || mode == "cpu", "8-bit moments (moment_dtype='int8') need cpu mode");
  gpu_mode_ = mode == "gpu";
  moment_dtype_ = moment_dtype == "bfloat16" ? at::kBFloat16 : moment_dtype == "int8" ? at::kByte : at::kFloat;
  runtime(device_);  // create streams eagerly
  if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
  if (!gpu_mode_) pool_ = std::make_unique<WorkerPool>(threads, device_);
  gate_ = std::make_shared<StepGate>();
  if (gpu_mode_) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    TS_CUDA_CHECK(cudaMalloc(&norm_dev_, sizeof(float)));
    norm_host_ = pinned_empty({1}, at::kFloat, "adamw.norm", false, false);
  }
}

OffloadAdamW::~OffloadAdamW() {
  {  // a checkpoint thread still reads this engine's buffers: let it finish
    std::unique_lock<std::mutex> lock(ckpt_mu_);
    ckpt_cv_.wait(lock, [this] { return ckpt_jobs_ == 0; });
  }
  try {
    synchronize();
  } catch (...) {
  }
  pool_.reset();  // joins workers before slots (and their buffers) die
  if (norm_dev_ != nullptr) cudaFree(norm_dev_);
}

OffloadAdamW::Slot& OffloadAdamW::slot(int64_t id) {
  TORCH_CHECK(id >= 0 && id < static_cast<int64_t>(slots_.size()), "OffloadAdamW: bad param id ", id);
  return *slots_[static_cast<size_t>(id)];
}

size_t OffloadAdamW::worker_for(int64_t id, int64_t chunk) const {
  // Deterministic (param, chunk) -> worker mapping gives per-chunk FIFO
  // ordering (accumulate -> norm -> update) without any extra locking.
  return static_cast<size_t>(static_cast<uint64_t>(id) * 2654435761ull + static_cast<uint64_t>(chunk));
}

int64_t OffloadAdamW::add_param(const at::Tensor& param, bool pin_moments, const at::Tensor& host_init) {
  TORCH_CHECK(param.is_cuda(), "OffloadAdamW: parameters must live on a CUDA device");
  TORCH_CHECK(param.get_device() == device_, "OffloadAdamW: parameter on cuda:", param.get_device(),
              " but engine bound to cuda:", device_);
  TORCH_CHECK(param.is_contiguous(), "OffloadAdamW: parameters must be contiguous");
  to_kernel_dtype(param.scalar_type());
  if (host_init.defined()) {
    TORCH_CHECK(!gpu_mode_, "OffloadAdamW: host_init (paged parameters) requires cpu mode");
    TORCH_CHECK(host_init.device().is_cpu() && host_init.numel() == param.numel(),
                "OffloadAdamW: host_init must be a CPU tensor with the parameter's numel");
  }

  auto s = std::make_unique<Slot>();
  s->id = static_cast<int64_t>(slots_.size());
  s->param = param.detach().view({-1});
  s->numel = s->param.numel();
  s->pdtype = param.scalar_type();
  s->nchunks = std::max<int64_t>(1, (s->numel + chunk_ - 1) / chunk_);
  const std::vector<int64_t> shape{s->numel};
  const bool mapped = gpu_mode_;

  if (!gpu_mode_ || s->pdtype != at::kFloat) {
    s->master = pinned_empty(shape, at::kFloat, "adamw.master_fp32", mapped, false);
    // device -> host with dtype conversion (or straight from the CPU copy)
    s->master.copy_(host_init.defined() ? host_init.reshape({-1}) : s->param);
  }
  // Moments: the zero-copy kernel needs them mapped+pinned. In cpu mode only
  // host threads touch them, so page-locking is optional (pin_moments) — it
  // keeps the total page-locked footprint within what the OS will tolerate.
  if (gpu_mode_ || pin_moments) {
    s->exp_avg = pinned_empty(shape, moment_dtype_, "adamw.exp_avg", mapped, false);
    s->exp_avg_sq = pinned_empty(shape, moment_dtype_, "adamw.exp_avg_sq", mapped, false);
    s->exp_avg.zero_();
    s->exp_avg_sq.zero_();
    pinned_state_bytes_ += s->exp_avg.nbytes() * 2;
  } else {
    s->exp_avg = at::zeros(shape, at::TensorOptions().dtype(moment_dtype_));
    s->exp_avg_sq = at::zeros(shape, at::TensorOptions().dtype(moment_dtype_));
    pageable_state_bytes_ += s->exp_avg.nbytes() * 2;
  }
  if (moment_dtype_ == at::kByte) {  // cpu mode only: plain host memory
    const int64_t nb = (s->numel + kQBlock - 1) / kQBlock;
    s->m_scale = at::zeros({nb}, at::TensorOptions().dtype(at::kFloat));
    s->v_scale = at::zeros({nb}, at::TensorOptions().dtype(at::kFloat));
    pageable_state_bytes_ += nb * 8;
  }
  int64_t bytes = s->exp_avg.nbytes() * 2 + (s->master.defined() ? s->master.nbytes() : 0);
  if (s->master.defined()) pinned_state_bytes_ += s->master.nbytes();

  if (gpu_mode_) {
    s->d_m = pinned_device_pointer(s->exp_avg);
    s->d_v = pinned_device_pointer(s->exp_avg_sq);
    if (s->master.defined()) s->d_master = static_cast<float*>(pinned_device_pointer(s->master));
  } else {
    // grad_acc (fp32, host-only) is created lazily: the eager single-micro-batch
    // path reads the DMA staging buffer directly and never needs it. The
    // staging buffer itself comes from a pool per gradient (acquire_stage).
    if (s->pdtype != at::kFloat) {
      s->param_stage = pinned_empty(shape, s->pdtype, "param.staging", false, false);
      // Valid from the start: paged slots stream their weights from here.
      s->param_stage.copy_(host_init.defined() ? host_init.reshape({-1}) : s->param);
      bytes += s->param_stage.nbytes();
      pinned_state_bytes_ += s->param_stage.nbytes();
    }
  }
  host_state_bytes_ += bytes;
  slots_.push_back(std::move(s));
  return slots_.back()->id;
}

void OffloadAdamW::set_max_staging_bytes(int64_t bytes) {
  {
    std::lock_guard<std::mutex> lock(stage_mu_);
    max_stage_bytes_ = std::max<int64_t>(bytes, 1);
  }
  stage_cv_.notify_all();
}

void OffloadAdamW::acquire_stage(Slot& s, at::ScalarType dtype) {
  const int64_t nbytes = s.numel * static_cast<int64_t>(c10::elementSize(dtype));
  {
    // A single tensor larger than the cap still proceeds once nothing else is staged.
    std::unique_lock<std::mutex> lock(stage_mu_);
    if (stage_bytes_ > 0 && stage_bytes_ + nbytes > max_stage_bytes_) {
      const auto t0 = std::chrono::steady_clock::now();
      stage_cv_.wait(lock, [&] { return stage_bytes_ == 0 || stage_bytes_ + nbytes <= max_stage_bytes_; });
      grad_stall_us_ += elapsed_us(t0);
    }
    stage_bytes_ += nbytes;
    stage_peak_ = std::max(stage_peak_, stage_bytes_);
  }
  s.grad_stage = pinned_empty({s.numel}, dtype, "grad.staging", false, true);
}

void OffloadAdamW::release_stage(Slot& s) {
  if (!s.grad_stage.defined()) return;
  const int64_t nbytes = static_cast<int64_t>(s.grad_stage.nbytes());
  s.grad_stage.reset();  // back to the pinned cache
  s.grad_scale.reset();
  {
    std::lock_guard<std::mutex> lock(stage_mu_);
    stage_bytes_ -= nbytes;
  }
  stage_cv_.notify_all();
}

void OffloadAdamW::retire_grad(EventPtr d2h_done, at::Tensor grad) {
  std::lock_guard<std::mutex> lock(retire_mu_);
  retiring_bytes_ += static_cast<int64_t>(grad.nbytes());
  retiring_.emplace_back(std::move(d2h_done), std::move(grad));
}

void OffloadAdamW::reap_grads(bool block) {
  // Dropping the last reference returns the block to the compute stream's
  // pool; the copy that read it has finished, so reuse is safe. Beyond the
  // in-flight cap the caller (the backward pass) waits for the oldest copies.
  const auto t0 = std::chrono::steady_clock::now();
  bool waited = false;
  std::lock_guard<std::mutex> lock(retire_mu_);
  while (!retiring_.empty()) {
    auto& front = retiring_.front();
    if (!front.first->query()) {
      if (!block && retiring_bytes_ <= max_inflight_grad_bytes_) break;
      front.first->synchronize();
      waited = true;
    }
    retiring_bytes_ -= static_cast<int64_t>(front.second.nbytes());
    retiring_.pop_front();
  }
  if (waited && !block) grad_stall_us_ += elapsed_us(t0);
}

void OffloadAdamW::record_error(const std::string& msg) {
  std::lock_guard<std::mutex> lock(err_mu_);
  if (error_.empty()) error_ = msg;
}

void OffloadAdamW::check_error() {
  std::lock_guard<std::mutex> lock(err_mu_);
  if (!error_.empty()) {
    const std::string msg = error_;
    error_.clear();
    TORCH_CHECK(false, "TrainStream optimizer worker failed: ", msg);
  }
}

void OffloadAdamW::run_batch(Slot& s, std::shared_ptr<ChunkFn> body,
                             std::shared_ptr<std::function<void()>> finalize) {
  {
    std::lock_guard<std::mutex> lock(s.mu);
    ++s.inflight;
  }
  auto remaining = std::make_shared<std::atomic<int64_t>>(s.nchunks);
  Slot* sp = &s;
  for (int64_t c = 0; c < s.nchunks; ++c) {
    const int64_t b = c * chunk_;
    const int64_t e = std::min(s.numel, b + chunk_);
    pool_->submit(worker_for(s.id, c), [this, sp, body, finalize, remaining, c, b, e] {
      try {
        (*body)(c, b, e);
      } catch (const std::exception& ex) {
        record_error(ex.what());
      }
      if (remaining->fetch_sub(1) == 1) {
        if (finalize) {
          try {
            (*finalize)();
          } catch (const std::exception& ex) {
            record_error(ex.what());
          }
        }
        {
          std::lock_guard<std::mutex> lock(sp->mu);
          --sp->inflight;
        }
        sp->cv.notify_all();
      }
    });
  }
}

void OffloadAdamW::wait_idle(Slot& s) {
  std::unique_lock<std::mutex> lock(s.mu);
  s.cv.wait(lock, [&s] { return s.inflight == 0; });
}

void OffloadAdamW::offload_grad(int64_t id, const at::Tensor& grad, bool apply_update, double lr,
                                double beta1, double beta2, double eps, double weight_decay) {
  TORCH_CHECK(!gpu_mode_, "offload_grad is only used in 'cpu' mode");
  check_error();
  Slot& s = slot(id);
  TORCH_CHECK(grad.is_cuda() && grad.numel() == s.numel, "offload_grad: gradient shape mismatch for param ", id);

  // The staging buffer may still be read by the previous accumulate batch.
  const auto t0 = std::chrono::steady_clock::now();
  wait_idle(s);
  grad_stall_us_ += elapsed_us(t0);

  at::Tensor g = grad.contiguous().view({-1});
  if (fp8_grads_) {
    // One byte per value crosses PCIe. The encode kernel runs on the compute stream, behind
    // whatever produced the gradient; the gradient itself is not kept (its memory can only be
    // reused by work queued later on that same stream), only the codes until they have landed.
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    const int64_t nb = (s.numel + kQBlock - 1) / kQBlock;
    at::Tensor codes = at::empty({s.numel}, g.options().dtype(at::kByte));
    at::Tensor scales = at::empty({nb}, g.options().dtype(at::kFloat));
    const uint32_t seed = mix32(fp8_seq_.fetch_add(1) * 0x9e3779b1u ^ static_cast<uint32_t>(id));
    TS_CUDA_CHECK(kernels::grad_fp8_encode(to_kernel_dtype(g.scalar_type()), g.data_ptr(), s.numel,
                                           codes.data_ptr<uint8_t>(), scales.data_ptr<float>(), seed,
                                           runtime(device_).sm_count, compute_stream(device_)));
    acquire_stage(s, at::kByte);
    s.grad_scale = pinned_empty({nb}, at::kFloat, "grad.staging", false, true);
    EventPtr codes_done = d2h_async(s.grad_stage, codes);
    s.d2h_done = d2h_async(s.grad_scale, scales);  // the same stream: this event covers both copies
    retire_grad(codes_done, codes);
    retire_grad(s.d2h_done, scales);
    grad_bytes_ += static_cast<int64_t>(codes.nbytes() + scales.nbytes());
  } else {
    acquire_stage(s, g.scalar_type());
    s.d2h_done = d2h_async(s.grad_stage, g);
    retire_grad(s.d2h_done, g);  // VRAM copy freed as soon as the DMA has landed
    grad_bytes_ += static_cast<int64_t>(g.nbytes());
  }
  reap_grads(false);

  const bool fresh = s.acc_fresh;
  EventPtr landed = s.d2h_done;
  // Guarded steps are decided as a whole at step(): no update may run before
  // every gradient of the step has been checked.
  if (nan_guard_) apply_update = false;
  if (apply_update && fresh && !fp8_grads_) {  // (fp8 codes are expanded into the accumulator first)
    // Single micro-batch: AdamW reads the DMA staging buffer directly, skipping
    // the fp32 accumulator round trip (~8 fewer bytes of host traffic/element).
    steps_eager_ += 1;
    s.has_grad = false;  // cleared synchronously: step() must not update twice
    s.acc_fresh = false;
    // Everything that reads this weight in backward is already queued on the
    // compute stream (the post-accumulate hook fires after its last use).
    enqueue_update(s, lr, beta1, beta2, eps, weight_decay, 1.0, record_on_current(device_), landed);
    return;
  }
  s.acc_fresh = false;
  s.has_grad = true;
  if (!s.grad_acc.defined()) {
    s.grad_acc = at::empty({s.numel}, at::TensorOptions().dtype(at::kFloat));  // pageable, host-only
    pageable_state_bytes_ += s.grad_acc.nbytes();
    host_state_bytes_ += s.grad_acc.nbytes();
  }
  Slot* sp = &s;
  std::shared_ptr<StepGate> gate = nan_guard_ ? gate_ : nullptr;
  if (gate) gate->enter();
  auto body = std::make_shared<ChunkFn>([this, sp, landed, fresh, gate](int64_t, int64_t b, int64_t e) {
    landed->synchronize();  // DMA complete -> staging bytes are valid
    const auto t = std::chrono::steady_clock::now();
    float* acc = sp->grad_acc.data_ptr<float>();
    const bool check = gate != nullptr;
    bool bad = false;
    switch (sp->grad_stage.scalar_type()) {
      case at::kFloat:
        bad = accumulate_host(acc, sp->grad_stage.data_ptr<float>(), b, e, fresh, check);
        break;
      case at::kHalf:
        bad = accumulate_host(acc, sp->grad_stage.data_ptr<at::Half>(), b, e, fresh, check);
        break;
      case at::kBFloat16:
        bad = accumulate_host(acc, sp->grad_stage.data_ptr<at::BFloat16>(), b, e, fresh, check);
        break;
      case at::kByte:  // fp8 gradients
        bad = accumulate_host_fp8(acc, sp->grad_stage.data_ptr<uint8_t>(), sp->grad_scale.data_ptr<float>(), b, e,
                                  fresh, check);
        break;
      default:
        TORCH_CHECK(false, "unsupported gradient dtype");
    }
    if (bad) gate->bad.store(true);
    accumulate_us_ += elapsed_us(t);
  });
  // Accumulated: the staging buffer can go back to the pool.
  auto release = std::make_shared<std::function<void()>>([this, sp, gate] {
    release_stage(*sp);
    if (gate) gate->leave();
  });
  run_batch(s, body, release);
  if (apply_update) {
    steps_eager_ += 1;
    s.has_grad = false;  // cleared synchronously: step() must not update twice
    enqueue_update(s, lr, beta1, beta2, eps, weight_decay, 1.0, record_on_current(device_), nullptr);
  }
}

void OffloadAdamW::enqueue_update(Slot& s, double lr, double beta1, double beta2, double eps, double wd,
                                  double clip, EventPtr after, EventPtr from_stage, std::shared_ptr<StepGate> gate) {
  Slot* sp = &s;
  s.step += 1;  // taken back in finalize when the step turns out to be skipped
  const int64_t my_step = s.step;
  const HostHp hp = make_hp(lr, beta1, beta2, eps, wd, s.step, clip);
  const uint32_t seed = slot_seed(s.id, s.step);
  const EventPtr prev_ready = s.ready;
  int64_t seq;
  {
    std::lock_guard<std::mutex> lock(s.mu);
    seq = ++s.enq_seq;
  }

  auto body = std::make_shared<ChunkFn>([this, sp, hp, seed, prev_ready, from_stage, gate, seq](int64_t, int64_t b,
                                                                                                int64_t e) {
    if (gate && gate->skip()) return;  // a gradient of this step was not finite: nothing changes
    {  // a checkpoint is writing this slot's state as of an earlier update: not yet
      std::unique_lock<std::mutex> lock(sp->mu);
      sp->cv.wait(lock, [sp, seq] { return !sp->ckpt_hold || seq <= sp->ckpt_target; });
    }
    // The previous H2D may still be reading master/param_stage.
    if (prev_ready) prev_ready->synchronize();
    if (from_stage) from_stage->synchronize();  // gradient DMA landed in staging
    const auto t = std::chrono::steady_clock::now();
    if (sp->m_scale.defined())
      adamw8_dispatch(sp->master.data_ptr<float>(), sp->exp_avg, sp->exp_avg_sq, sp->m_scale, sp->v_scale,
                      from_stage ? sp->grad_stage : sp->grad_acc, sp->pdtype, sp->param_stage, b, e, hp, seed);
    else
      adamw_dispatch(sp->master.data_ptr<float>(), sp->exp_avg, sp->exp_avg_sq,
                     from_stage ? sp->grad_stage : sp->grad_acc, sp->pdtype, sp->param_stage, b, e, hp, seed);
    update_us_ += elapsed_us(t);
  });

  auto finalize = std::make_shared<std::function<void()>>([this, sp, after, from_stage, gate, my_step] {
    const bool skipped = gate && gate->skip();
    if (from_stage) release_stage(*sp);  // fused path: the update read the staging buffer directly
    EventPtr done;
    if (!skipped && !sp->paged) {  // paged weights are streamed in by the pager instead
      const at::Tensor& src = sp->param_stage.defined() ? sp->param_stage : sp->master;
      done = h2d_async(sp->param, src, after);
      param_bytes_ += static_cast<int64_t>(src.nbytes());
    }
    {
      std::lock_guard<std::mutex> lock(sp->mu);
      if (done) sp->ready = std::move(done);
      sp->has_grad = false;
      sp->acc_fresh = true;  // the next gradient overwrites the accumulator: a skipped step's values are gone
      if (skipped) {
        sp->step -= 1;
      } else {
        sp->step_done = my_step;
      }
      sp->done_seq += 1;  // run_batch notifies the slot's waiters right after this
    }
    if (gate && !gate->counted.exchange(true)) note_step(skipped);
    last_finalize_ms_ = now_ms();
  });
  run_batch(s, body, finalize);
}

void OffloadAdamW::note_step(bool skipped) {
  if (skipped) {
    skipped_steps_ += 1;
    consecutive_skips_ += 1;
  } else {
    consecutive_skips_ = 0;
  }
}

void OffloadAdamW::set_nan_guard(bool on) {
  synchronize();
  nan_guard_ = on;
}

void OffloadAdamW::set_fp8_gradients(bool on) {
  TORCH_CHECK(!on || !gpu_mode_, "fp8 gradients are for cpu mode (in gpu mode gradients never leave VRAM)");
  synchronize();
  fp8_grads_ = on;
}

int64_t OffloadAdamW::skipped_steps(bool wait) {
  if (gpu_mode_ && wait) settle_gpu_step();
  return skipped_steps_.load();
}

int64_t OffloadAdamW::consecutive_skips(bool wait) {
  if (gpu_mode_ && wait) settle_gpu_step();
  return consecutive_skips_.load();
}

void OffloadAdamW::settle_gpu_step() {
  std::lock_guard<std::mutex> settle_lock(settle_mu_);  // the training thread and a checkpoint thread
  if (!norm_ev_) return;
  norm_ev_->synchronize();
  const float sq = norm_host_.data_ptr<float>()[0];
  last_norm_ = std::sqrt(static_cast<double>(sq));
  norm_ev_.reset();
  if (nan_guard_) {
    const bool skipped = !std::isfinite(sq);
    if (skipped)  // the kernels returned without touching anything: the step never happened
      for (int64_t id : gpu_step_ids_) slot(id).step -= 1;
    note_step(skipped);
  }
  gpu_step_ids_.clear();
}

double OffloadAdamW::step(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                          const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                          const std::vector<double>& epss, const std::vector<double>& wds,
                          const std::vector<at::Tensor>& grads, double max_grad_norm,
                          bool compute_norm, double grad_scale) {
  const size_t n = ids.size();
  TORCH_CHECK(lrs.size() == n && beta1s.size() == n && beta2s.size() == n && epss.size() == n &&
                  wds.size() == n,
              "OffloadAdamW.step: hyper-parameter vectors must match ids");
  check_error();
  steps_ += 1;
  if (gpu_mode_) {
    TORCH_CHECK(grads.size() == n, "OffloadAdamW.step(gpu): one gradient (or None) per id required");
    step_gpu(ids, lrs, beta1s, beta2s, epss, wds, grads, max_grad_norm, compute_norm);
    return -1.0;
  }
  return step_cpu(ids, lrs, beta1s, beta2s, epss, wds, max_grad_norm, compute_norm, grad_scale);
}

double OffloadAdamW::step_cpu(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                              const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                              const std::vector<double>& epss, const std::vector<double>& wds,
                              double max_grad_norm, bool compute_norm, double grad_scale) {
  // Everything enqueued on the compute stream so far (the whole backward) must
  // finish before new weights may overwrite the parameters on the GPU.
  EventPtr step_ev = record_on_current(device_);
  reap_grads(false);
  step_start_ms_ = now_ms();

  std::vector<size_t> active;
  for (size_t k = 0; k < ids.size(); ++k)
    if (slot(ids[k]).has_grad) active.push_back(k);

  // Guard: this step's gate stops collecting gradients; the next step gets a new one.
  std::shared_ptr<StepGate> gate;
  if (nan_guard_) {
    gate = gate_;
    gate_ = std::make_shared<StepGate>();
  }

  // grad_scale multiplies the accumulated gradients (1/world_size after a
  // data-parallel sum); the norm and the clip coefficient refer to the scaled values.
  double clip = grad_scale;
  double norm = -1.0;
  if (max_grad_norm > 0.0 || compute_norm) {
    // Global norm needs every gradient: run a reduction batch per slot (FIFO
    // after its accumulations), then barrier.
    std::vector<int64_t> offsets;
    int64_t total = 0;
    for (size_t k : active) {
      offsets.push_back(total);
      total += slot(ids[k]).nchunks;
    }
    auto partials = std::make_shared<std::vector<double>>(static_cast<size_t>(total), 0.0);
    for (size_t a = 0; a < active.size(); ++a) {
      Slot& s = slot(ids[active[a]]);
      Slot* sp = &s;
      const int64_t off = offsets[a];
      auto body = std::make_shared<ChunkFn>([sp, partials, off](int64_t c, int64_t b, int64_t e) {
        const float* g = sp->grad_acc.data_ptr<float>();
        double acc = 0.0;
        for (int64_t i = b; i < e; ++i) acc += static_cast<double>(g[i]) * g[i];
        (*partials)[static_cast<size_t>(off + c)] = acc;
      });
      run_batch(s, body, nullptr);
    }
    for (size_t k : active) wait_idle(slot(ids[k]));
    check_error();
    double sq = 0.0;
    for (double p : *partials) sq += p;
    norm = std::sqrt(sq) * grad_scale;
    last_norm_ = norm;
    if (max_grad_norm > 0.0) clip = std::min(1.0, max_grad_norm / (norm + 1e-6)) * grad_scale;
    // The norm is taken over what the update will read (after a data-parallel
    // reduction: the same values on every rank), so every rank decides alike.
    if (gate && !std::isfinite(norm)) gate->bad.store(true);
  }

  if (gate) gate->seal();
  for (size_t k : active)
    enqueue_update(slot(ids[k]), lrs[k], beta1s[k], beta2s[k], epss[k], wds[k], clip, step_ev, nullptr, gate);
  return norm;
}

void OffloadAdamW::step_gpu(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                            const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                            const std::vector<double>& epss, const std::vector<double>& wds,
                            const std::vector<at::Tensor>& grads, double max_grad_norm,
                            bool compute_norm) {
  Runtime& rt = runtime(device_);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));

  // Normalise gradients first (any copy kernels land on the compute stream
  // before step_ev is recorded, so the update stream sees finished data).
  std::vector<at::Tensor> g(ids.size());
  for (size_t k = 0; k < ids.size(); ++k) {
    if (!grads[k].defined()) continue;
    Slot& s = slot(ids[k]);
    TORCH_CHECK(grads[k].numel() == s.numel, "step(gpu): gradient shape mismatch for param ", ids[k]);
    g[k] = grads[k].to(s.pdtype).contiguous().view({-1});
  }
  {  // a checkpoint still reads the moments these kernels are about to write
    std::unique_lock<std::mutex> lock(ckpt_mu_);
    ckpt_cv_.wait(lock, [this] { return ckpt_holds_ == 0; });
  }
  settle_gpu_step();  // the previous step's verdict, before its step counts are used again
  const auto t0 = std::chrono::steady_clock::now();
  EventPtr step_ev = record_on_current(device_);
  step_ev->block(rt.upd);

  // The guard needs sum(g^2) on the device: one extra pass over gradients that
  // are already in VRAM, small next to the PCIe-bound update that follows.
  const bool use_norm = max_grad_norm > 0.0 || compute_norm || nan_guard_;
  if (use_norm) {
    TS_CUDA_CHECK(cudaMemsetAsync(norm_dev_, 0, sizeof(float), rt.upd));
    for (size_t k = 0; k < ids.size(); ++k) {
      if (!g[k].defined()) continue;
      TS_CUDA_CHECK(kernels::grad_sq_norm_accumulate(to_kernel_dtype(g[k].scalar_type()), g[k].data_ptr(),
                                                     g[k].numel(), norm_dev_, rt.sm_count, rt.upd));
    }
    // Copied out before the update kernels run, so the host can read it early.
    TS_CUDA_CHECK(cudaMemcpyAsync(norm_host_.data_ptr(), norm_dev_, sizeof(float), cudaMemcpyDeviceToHost, rt.upd));
    norm_ev_ = make_event(device_);
    norm_ev_->record(rt.upd);
  }
  const float* norm_arg = (max_grad_norm > 0.0 || nan_guard_) ? norm_dev_ : nullptr;
  int32_t* guard_arg = nan_guard_ ? guard_flags_always(device_) : nullptr;
  for (size_t k = 0; k < ids.size(); ++k) {
    if (!g[k].defined()) continue;
    Slot& s = slot(ids[k]);
    s.step += 1;
    const HostHp h = make_hp(lrs[k], beta1s[k], beta2s[k], epss[k], wds[k], s.step, 1.0);
    kernels::AdamWParams hp{h.lr, h.beta1, h.beta2, h.eps, h.weight_decay, h.step_size, h.bc2_sqrt,
                            static_cast<float>(max_grad_norm)};
    TS_CUDA_CHECK(kernels::adamw_zero_copy(to_kernel_dtype(s.pdtype), s.param.data_ptr(), g[k].data_ptr(),
                                           s.d_m, s.d_v, s.d_master, s.numel, hp,
                                           norm_arg, rt.sm_count, rt.upd, moment_dtype_ == at::kBFloat16,
                                           slot_seed(s.id, s.step), guard_arg));
    if (use_norm) gpu_step_ids_.push_back(s.id);
    // Keep the gradient alive instead of recordStream (whose free-path event
    // creation can abort on a full device). It is replaced next step, after
    // the next forward has waited on this kernel's `ready` event.
    {
      std::lock_guard<std::mutex> lock(s.mu);
      s.grad_keep = g[k];
    }
    auto ev = make_event(device_);
    ev->record(rt.upd);
    std::lock_guard<std::mutex> lock(s.mu);
    s.ready = std::move(ev);
  }
  update_us_ += elapsed_us(t0);  // host launch overhead only; device time is async
}

void OffloadAdamW::wait_ready(const std::vector<int64_t>& ids) {
  check_error();
  if (!gpu_mode_) reap_grads(false);
  const cudaStream_t s = compute_stream(device_);
  const auto t0 = std::chrono::steady_clock::now();
  for (int64_t id : ids) {
    Slot& sl = slot(id);
    EventPtr ready;
    {
      std::unique_lock<std::mutex> lock(sl.mu);
      sl.cv.wait(lock, [&sl] { return sl.inflight == 0; });
      ready = sl.ready;
    }
    if (ready) ready->block(s);
    if (gpu_mode_) {
      // The compute stream now waits on the update kernel, so any reuse of the
      // gradient's block (freed into the compute pool) is ordered after it.
      std::lock_guard<std::mutex> lock(sl.mu);
      sl.grad_keep.reset();
    }
  }
  ready_stall_us_ += elapsed_us(t0);
}

void OffloadAdamW::synchronize() {
  reap_grads(true);
  for (auto& sp : slots_) {
    wait_idle(*sp);
    EventPtr ready;
    {
      std::lock_guard<std::mutex> lock(sp->mu);
      ready = sp->ready;
    }
    if (ready) ready->synchronize();
  }
  if (gpu_mode_) {
    TS_CUDA_CHECK(cudaStreamSynchronize(runtime(device_).upd));
    for (auto& sp : slots_) {
      std::lock_guard<std::mutex> lock(sp->mu);
      sp->grad_keep.reset();  // kernels are done: gradients can go
    }
    settle_gpu_step();
  }
  check_error();
}

double OffloadAdamW::last_grad_norm() {
  if (gpu_mode_) settle_gpu_step();
  return last_norm_;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> OffloadAdamW::state(int64_t id) {
  Slot& s = slot(id);
  wait_idle(s);
  if (s.m_scale.defined())  // 8-bit storage: handed out (and taken back) as fp32
    return {s.master, decode8(s.exp_avg, s.m_scale, true), decode8(s.exp_avg_sq, s.v_scale, false)};
  return {s.master, s.exp_avg, s.exp_avg_sq};
}

void OffloadAdamW::load_state(int64_t id, const at::Tensor& exp_avg, const at::Tensor& exp_avg_sq,
                              const at::Tensor& master, int64_t step) {
  Slot& s = slot(id);
  wait_idle(s);
  if (s.ready) s.ready->synchronize();
  TORCH_CHECK(exp_avg.numel() == s.numel && exp_avg_sq.numel() == s.numel,
              "load_state: moment shape mismatch for param ", id);
  if (s.m_scale.defined()) {
    encode8(exp_avg, s.exp_avg, s.m_scale, true);
    encode8(exp_avg_sq, s.exp_avg_sq, s.v_scale, false);
  } else {
    s.exp_avg.copy_(exp_avg.reshape({-1}));
    s.exp_avg_sq.copy_(exp_avg_sq.reshape({-1}));
  }
  if (master.defined() && s.master.defined()) {
    TORCH_CHECK(master.numel() == s.numel, "load_state: master shape mismatch for param ", id);
    s.master.copy_(master.reshape({-1}));
  } else if (s.master.defined() && !s.paged) {
    s.master.copy_(s.param);  // re-sync master with the live weights (paged: the host copy already is)
  }
  s.step = step;
  s.step_done = step;
}

// ===========================================================================
// Background checkpoints
// ===========================================================================
std::map<std::string, at::Tensor> OffloadAdamW::raw_state(int64_t id) {
  Slot& s = slot(id);
  std::map<std::string, at::Tensor> out;
  if (s.master.defined()) out["master"] = s.master;
  out["exp_avg"] = s.exp_avg;
  out["exp_avg_sq"] = s.exp_avg_sq;
  if (s.m_scale.defined()) {
    out["m_scale"] = s.m_scale;
    out["v_scale"] = s.v_scale;
  }
  if (s.param_stage.defined()) out["param_stage"] = s.param_stage;
  return out;
}

void OffloadAdamW::load_state_raw(int64_t id, const std::map<std::string, at::Tensor>& state, int64_t step) {
  Slot& s = slot(id);
  wait_idle(s);
  if (s.ready) s.ready->synchronize();
  auto put = [&](const char* name, at::Tensor& dst) {
    auto it = state.find(name);
    if (it == state.end()) return;
    TORCH_CHECK(dst.defined() && it->second.numel() == dst.numel() && it->second.scalar_type() == dst.scalar_type(),
                "load_state_raw: '", name, "' of param ", id, " does not match this optimizer's storage");
    dst.copy_(it->second.reshape({-1}));
  };
  put("master", s.master);
  put("exp_avg", s.exp_avg);
  put("exp_avg_sq", s.exp_avg_sq);
  put("m_scale", s.m_scale);
  put("v_scale", s.v_scale);
  put("param_stage", s.param_stage);
  s.step = step;
  s.step_done = step;
}

std::vector<int64_t> OffloadAdamW::checkpoint_begin() {
  check_error();
  {
    std::unique_lock<std::mutex> lock(ckpt_mu_);
    ckpt_cv_.wait(lock, [this] { return ckpt_jobs_ == 0; });  // one at a time
    ckpt_jobs_ = 1;
    ckpt_holds_ = static_cast<int>(slots_.size());
  }
  std::vector<int64_t> targets;
  for (auto& sp : slots_) {
    std::lock_guard<std::mutex> lock(sp->mu);
    sp->ckpt_hold = true;
    sp->ckpt_target = sp->enq_seq;
    targets.push_back(sp->enq_seq);
  }
  return targets;
}

int64_t OffloadAdamW::checkpoint_wait(int64_t id, int64_t target) {
  Slot& s = slot(id);
  if (gpu_mode_) {
    EventPtr ready;
    {
      std::lock_guard<std::mutex> lock(s.mu);
      ready = s.ready;
    }
    if (ready) ready->synchronize();  // the last step's kernel has written this slot's state
    settle_gpu_step();                // ... and its step count stands (or was taken back)
    return s.step;
  }
  std::unique_lock<std::mutex> lock(s.mu);
  s.cv.wait(lock, [&s, target] { return s.done_seq >= target; });
  return s.step_done;
}

void OffloadAdamW::checkpoint_release(int64_t id) {
  Slot& s = slot(id);
  bool held;
  {
    std::lock_guard<std::mutex> lock(s.mu);
    held = s.ckpt_hold;
    s.ckpt_hold = false;
  }
  if (!held) return;
  s.cv.notify_all();
  {
    std::lock_guard<std::mutex> lock(ckpt_mu_);
    --ckpt_holds_;
  }
  ckpt_cv_.notify_all();
}

void OffloadAdamW::checkpoint_end() {
  for (auto& sp : slots_) checkpoint_release(sp->id);
  {
    std::lock_guard<std::mutex> lock(ckpt_mu_);
    ckpt_jobs_ = 0;
  }
  ckpt_cv_.notify_all();
}

std::tuple<at::Tensor, at::Tensor> grad_fp8_encode(const at::Tensor& grad, int64_t seed) {
  TORCH_CHECK(grad.is_cuda(), "grad_fp8_encode: a CUDA tensor");
  const int dev = grad.get_device();
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  const at::Tensor g = grad.contiguous().view({-1});
  const int64_t n = g.numel();
  at::Tensor codes = at::empty({n}, g.options().dtype(at::kByte));
  at::Tensor scales = at::empty({(n + kQBlock - 1) / kQBlock}, g.options().dtype(at::kFloat));
  TS_CUDA_CHECK(kernels::grad_fp8_encode(to_kernel_dtype(g.scalar_type()), g.data_ptr(), n, codes.data_ptr<uint8_t>(),
                                         scales.data_ptr<float>(), static_cast<uint32_t>(seed), runtime(dev).sm_count,
                                         compute_stream(dev)));
  return {codes, scales};
}

at::Tensor grad_fp8_decode(const at::Tensor& codes, const at::Tensor& scales) {
  TORCH_CHECK(codes.device().is_cpu() && scales.device().is_cpu() && codes.scalar_type() == at::kByte &&
                  scales.scalar_type() == at::kFloat && scales.numel() == (codes.numel() + kQBlock - 1) / kQBlock,
              "grad_fp8_decode: uint8 codes and one float32 scale per 256 of them, on the host");
  const at::Tensor c = codes.contiguous().view({-1}), s = scales.contiguous().view({-1});
  at::Tensor out = at::empty({c.numel()}, at::TensorOptions().dtype(at::kFloat));
  accumulate_host_fp8(out.data_ptr<float>(), c.data_ptr<uint8_t>(), s.data_ptr<float>(), 0, c.numel(), true, false);
  return out;
}

at::Tensor decode_moment8(const at::Tensor& codes, const at::Tensor& scale, bool is_m) {
  TORCH_CHECK(codes.scalar_type() == at::kByte && scale.scalar_type() == at::kFloat && codes.device().is_cpu() &&
                  scale.device().is_cpu() && scale.numel() == (codes.numel() + kQBlock - 1) / kQBlock,
              "decode_moment8: uint8 codes and one float32 scale per 256 of them");
  return decode8(codes.contiguous().view({-1}), scale.contiguous().view({-1}), is_m);
}

int64_t OffloadAdamW::get_step(int64_t id) {
  if (gpu_mode_) settle_gpu_step();
  return slot(id).step;
}

void OffloadAdamW::set_paged(int64_t id, bool paged) {
  TORCH_CHECK(!gpu_mode_ || !paged, "parameter paging requires cpu mode");
  Slot& s = slot(id);
  wait_idle(s);
  std::lock_guard<std::mutex> lock(s.mu);
  s.paged = paged;
}

at::Tensor OffloadAdamW::host_param(int64_t id) {
  Slot& s = slot(id);
  return s.param_stage.defined() ? s.param_stage : s.master;
}

void OffloadAdamW::wait_issued(const std::vector<int64_t>& ids) {
  check_error();
  const auto t0 = std::chrono::steady_clock::now();
  for (int64_t id : ids) wait_idle(slot(id));
  ready_stall_us_ += elapsed_us(t0);
  check_error();
}

void OffloadAdamW::set_ready(int64_t id, const EventPtr& ev) {
  Slot& s = slot(id);
  std::lock_guard<std::mutex> lock(s.mu);
  s.ready = ev;
}

void OffloadAdamW::sync_host_from_param(int64_t id) {
  Slot& s = slot(id);
  wait_idle(s);
  if (s.ready) s.ready->synchronize();
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
  if (s.master.defined()) s.master.copy_(s.param);
  if (s.param_stage.defined()) s.param_stage.copy_(s.param);
}

at::Tensor OffloadAdamW::grad_buffer(int64_t id) {
  Slot& s = slot(id);
  wait_idle(s);
  TORCH_CHECK(s.grad_acc.defined(), "grad_buffer: parameter ", id, " has no accumulated gradient");
  return s.grad_acc;
}

void OffloadAdamW::update_slot(int64_t id, double lr, double beta1, double beta2, double eps,
                               double weight_decay, double scale) {
  TORCH_CHECK(!gpu_mode_, "update_slot is only used in 'cpu' mode");
  check_error();
  Slot& s = slot(id);
  TORCH_CHECK(s.grad_acc.defined(), "update_slot: parameter ", id, " has no accumulated gradient");
  steps_eager_ += 1;
  s.has_grad = false;  // cleared synchronously: step() must not update twice
  enqueue_update(s, lr, beta1, beta2, eps, weight_decay, scale, record_on_current(device_), nullptr);
}

std::map<std::string, double> OffloadAdamW::stats() {
  int64_t numel = 0;
  for (auto& s : slots_) numel += s->numel;
  const double last_wall =
      last_finalize_ms_.load() >= step_start_ms_.load() ? last_finalize_ms_.load() - step_start_ms_.load() : 0.0;
  return {{"params", static_cast<double>(slots_.size())},
          {"numel", static_cast<double>(numel)},
          {"threads", static_cast<double>(threads())},
          {"gpu_mode", gpu_mode_ ? 1.0 : 0.0},
          {"steps", static_cast<double>(steps_.load())},
          {"eager_updates", static_cast<double>(steps_eager_.load())},
          {"host_state_bytes", static_cast<double>(host_state_bytes_.load())},
          {"pinned_state_bytes", static_cast<double>(pinned_state_bytes_.load())},
          {"pageable_state_bytes", static_cast<double>(pageable_state_bytes_.load())},
          {"staging_bytes", static_cast<double>(stage_bytes_)},
          {"staging_peak_bytes", static_cast<double>(stage_peak_)},
          {"grad_bytes_offloaded", static_cast<double>(grad_bytes_.load())},
          {"param_bytes_uploaded", static_cast<double>(param_bytes_.load())},
          {"update_ms_total", update_us_.load() / 1e3},
          {"accumulate_ms_total", accumulate_us_.load() / 1e3},
          {"ready_stall_ms_total", ready_stall_us_.load() / 1e3},
          {"grad_stall_ms_total", grad_stall_us_.load() / 1e3},
          {"last_step_wall_ms", gpu_mode_ ? 0.0 : last_wall},
          {"nan_guard", nan_guard_ ? 1.0 : 0.0},
          {"fp8_gradients", fp8_grads_ ? 1.0 : 0.0},
          {"skipped_steps", static_cast<double>(skipped_steps_.load())},
          {"consecutive_skips", static_cast<double>(consecutive_skips_.load())}};
}

// ===========================================================================
// Benchmarks
// ===========================================================================
std::map<std::string, double> benchmark_cpu_adamw(int64_t numel, int threads) {
  if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
  numel = std::max<int64_t>(numel, 1 << 16);
  std::vector<float> p(numel, 0.5f), m(numel, 0.f), v(numel, 0.f), g(numel, 1e-3f);
  const HostHp hp = make_hp(1e-3, 0.9, 0.999, 1e-8, 0.01, 1, 1.0);
  double best_ms = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    const int64_t per = (numel + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
      const int64_t b = t * per, e = std::min(numel, b + per);
      if (b >= e) break;
      ts.emplace_back([&, b, e] { adamw_host<float, float, false, float>(p.data(), m.data(), v.data(), g.data(), nullptr, b, e, hp, 0u); });
    }
    for (auto& t : ts) t.join();
    best_ms = std::min(best_ms, elapsed_us(t0) / 1e3);
  }
  return {{"ms", best_ms},
          {"gelems_per_s", static_cast<double>(numel) / (best_ms * 1e-3) / 1e9},
          {"threads", static_cast<double>(threads)},
          {"numel", static_cast<double>(numel)}};
}

std::map<std::string, double> benchmark_zero_copy_adamw(int device, int64_t numel) {
  const int dev = resolve_device(device);
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  numel = std::max<int64_t>(numel, 1 << 16);
  auto opts = at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(dev)));
  at::Tensor p = at::full({numel}, 0.5, opts);
  at::Tensor g = at::full({numel}, 1e-3, opts);
  at::Tensor m = pinned_empty({numel}, at::kFloat, "bench.exp_avg", true, false);
  at::Tensor v = pinned_empty({numel}, at::kFloat, "bench.exp_avg_sq", true, false);
  m.zero_();
  v.zero_();
  float* dm = static_cast<float*>(pinned_device_pointer(m));
  float* dv = static_cast<float*>(pinned_device_pointer(v));
  const HostHp h = make_hp(1e-3, 0.9, 0.999, 1e-8, 0.01, 1, 1.0);
  kernels::AdamWParams hp{h.lr, h.beta1, h.beta2, h.eps, h.weight_decay, h.step_size, h.bc2_sqrt, 0.f};
  const cudaStream_t s = compute_stream(dev);
  TS_CUDA_CHECK(kernels::adamw_zero_copy(kernels::DType::kFloat32, p.data_ptr(), g.data_ptr(), dm, dv, nullptr,
                                         numel, hp, nullptr, rt.sm_count, s));
  Event start(dev, true), stop(dev, true);
  const int iters = 3;
  start.record(s);
  for (int i = 0; i < iters; ++i)
    TS_CUDA_CHECK(kernels::adamw_zero_copy(kernels::DType::kFloat32, p.data_ptr(), g.data_ptr(), dm, dv,
                                           nullptr, numel, hp, nullptr, rt.sm_count, s));
  stop.record(s);
  const double ms = start.elapsed_ms(stop) / iters;
  // 8 bytes read + 8 bytes written over PCIe per element (m and v).
  return {{"ms", ms},
          {"gelems_per_s", static_cast<double>(numel) / (ms * 1e-3) / 1e9},
          {"pcie_gbps", static_cast<double>(numel) * 16.0 / (ms * 1e-3) / 1e9},
          {"numel", static_cast<double>(numel)}};
}

}  // namespace trainstream
