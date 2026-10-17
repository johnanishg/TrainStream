// TrainStream — offloaded AdamW engine (host worker pool, gradient pipeline,
// CPU AdamW, zero-copy GPU dispatch) and optimizer micro-benchmarks.

#include "offload_adamw.h"

#include "kernels.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <chrono>
#include <cmath>

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

// Host AdamW over [b, e). Same arithmetic order as torch.optim.AdamW
// (single-tensor path): decoupled decay, lerp for m, addcmul for v,
// denom = sqrt(v)/sqrt(bc2) + eps, p -= (lr/bc1) * m/denom.
// G is the gradient element type (fp32 accumulator or the raw DMA staging
// buffer); with kStage the rounded weights are also written for the upload.
// The loop is branch-free so the compiler can vectorise it.
template <typename G, typename P, bool kStage>
void adamw_host(float* __restrict p, float* __restrict m, float* __restrict v,
                const G* __restrict g, P* __restrict stage, int64_t b, int64_t e,
                const HostHp& hp) {
  const float decay = 1.0f - hp.lr * hp.weight_decay;
  const float omb1 = 1.0f - hp.beta1;
  const float omb2 = 1.0f - hp.beta2;
  for (int64_t i = b; i < e; ++i) {
    const float gi = static_cast<float>(g[i]) * hp.clip;
    const float mi = m[i] + omb1 * (gi - m[i]);
    const float vi = hp.beta2 * v[i] + omb2 * gi * gi;
    const float denom = std::sqrt(vi) / hp.bc2_sqrt + hp.eps;
    const float pi = p[i] * decay - hp.step_size * (mi / denom);
    m[i] = mi;
    v[i] = vi;
    p[i] = pi;
    if constexpr (kStage) stage[i] = static_cast<P>(pi);
  }
}

template <typename G>
void adamw_dispatch_param(float* p, float* m, float* v, const G* g, at::ScalarType pdtype,
                          const at::Tensor& param_stage, int64_t b, int64_t e, const HostHp& hp) {
  switch (pdtype) {
    case at::kFloat:
      adamw_host<G, float, false>(p, m, v, g, nullptr, b, e, hp);
      break;
    case at::kHalf:
      adamw_host<G, at::Half, true>(p, m, v, g, param_stage.data_ptr<at::Half>(), b, e, hp);
      break;
    case at::kBFloat16:
      adamw_host<G, at::BFloat16, true>(p, m, v, g, param_stage.data_ptr<at::BFloat16>(), b, e, hp);
      break;
    default:
      TORCH_CHECK(false, "unsupported parameter dtype");
  }
}

void adamw_dispatch(float* p, float* m, float* v, const at::Tensor& grad, at::ScalarType pdtype,
                    const at::Tensor& param_stage, int64_t b, int64_t e, const HostHp& hp) {
  switch (grad.scalar_type()) {
    case at::kFloat:
      adamw_dispatch_param(p, m, v, grad.data_ptr<float>(), pdtype, param_stage, b, e, hp);
      break;
    case at::kHalf:
      adamw_dispatch_param(p, m, v, grad.data_ptr<at::Half>(), pdtype, param_stage, b, e, hp);
      break;
    case at::kBFloat16:
      adamw_dispatch_param(p, m, v, grad.data_ptr<at::BFloat16>(), pdtype, param_stage, b, e, hp);
      break;
    default:
      TORCH_CHECK(false, "unsupported gradient dtype");
  }
}

template <typename T>
void accumulate_host(float* __restrict acc, const T* __restrict g, int64_t b, int64_t e, bool fresh) {
  if (fresh) {
    for (int64_t i = b; i < e; ++i) acc[i] = static_cast<float>(g[i]);
  } else {
    for (int64_t i = b; i < e; ++i) acc[i] += static_cast<float>(g[i]);
  }
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
// OffloadAdamW
// ===========================================================================
OffloadAdamW::OffloadAdamW(int device, const std::string& mode, int threads, int64_t chunk_elems)
    : device_(resolve_device(device)), chunk_(std::max<int64_t>(chunk_elems, 4096)) {
  TORCH_CHECK(mode == "cpu" || mode == "gpu", "OffloadAdamW mode must be 'cpu' or 'gpu', got '", mode, "'");
  gpu_mode_ = mode == "gpu";
  runtime(device_);  // create streams eagerly
  if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
  if (!gpu_mode_) pool_ = std::make_unique<WorkerPool>(threads, device_);
  if (gpu_mode_) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    TS_CUDA_CHECK(cudaMalloc(&norm_dev_, sizeof(float)));
    norm_host_ = pinned_empty({1}, at::kFloat, "adamw.norm", false, false);
  }
}

OffloadAdamW::~OffloadAdamW() {
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

int64_t OffloadAdamW::add_param(const at::Tensor& param, bool pin_moments) {
  TORCH_CHECK(param.is_cuda(), "OffloadAdamW: parameters must live on a CUDA device");
  TORCH_CHECK(param.get_device() == device_, "OffloadAdamW: parameter on cuda:", param.get_device(),
              " but engine bound to cuda:", device_);
  TORCH_CHECK(param.is_contiguous(), "OffloadAdamW: parameters must be contiguous");
  to_kernel_dtype(param.scalar_type());

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
    s->master.copy_(s->param);  // device -> host with dtype conversion
  }
  // Moments: the zero-copy kernel needs them mapped+pinned. In cpu mode only
  // host threads touch them, so page-locking is optional (pin_moments) — it
  // keeps the total page-locked footprint within what the OS will tolerate.
  if (gpu_mode_ || pin_moments) {
    s->exp_avg = pinned_empty(shape, at::kFloat, "adamw.exp_avg", mapped, false);
    s->exp_avg_sq = pinned_empty(shape, at::kFloat, "adamw.exp_avg_sq", mapped, false);
    s->exp_avg.zero_();
    s->exp_avg_sq.zero_();
    pinned_state_bytes_ += s->exp_avg.nbytes() * 2;
  } else {
    s->exp_avg = at::zeros(shape, at::TensorOptions().dtype(at::kFloat));
    s->exp_avg_sq = at::zeros(shape, at::TensorOptions().dtype(at::kFloat));
    pageable_state_bytes_ += s->exp_avg.nbytes() * 2;
  }
  int64_t bytes = s->exp_avg.nbytes() * 2 + (s->master.defined() ? s->master.nbytes() : 0);
  if (s->master.defined()) pinned_state_bytes_ += s->master.nbytes();

  if (gpu_mode_) {
    s->d_m = static_cast<float*>(pinned_device_pointer(s->exp_avg));
    s->d_v = static_cast<float*>(pinned_device_pointer(s->exp_avg_sq));
    if (s->master.defined()) s->d_master = static_cast<float*>(pinned_device_pointer(s->master));
  } else {
    // grad_acc (fp32, host-only) is created lazily: the eager single-micro-batch
    // path reads the DMA staging buffer directly and never needs it.
    s->grad_stage = pinned_empty(shape, s->pdtype, "grad.staging", false, false);
    bytes += s->grad_stage.nbytes();
    pinned_state_bytes_ += s->grad_stage.nbytes();
    if (s->pdtype != at::kFloat) {
      s->param_stage = pinned_empty(shape, s->pdtype, "param.staging", false, false);
      bytes += s->param_stage.nbytes();
      pinned_state_bytes_ += s->param_stage.nbytes();
    }
  }
  host_state_bytes_ += bytes;
  slots_.push_back(std::move(s));
  return slots_.back()->id;
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
  if (g.scalar_type() != s.grad_stage.scalar_type())
    s.grad_stage = pinned_empty({s.numel}, g.scalar_type(), "grad.staging", false, false);
  s.d2h_done = d2h_async(s.grad_stage, g);
  s.grad_keep = g;  // released by the worker once the DMA has landed
  grad_bytes_ += static_cast<int64_t>(g.nbytes());

  const bool fresh = s.acc_fresh;
  EventPtr landed = s.d2h_done;
  if (apply_update && fresh) {
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
  auto body = std::make_shared<ChunkFn>([this, sp, landed, fresh](int64_t, int64_t b, int64_t e) {
    landed->synchronize();  // DMA complete -> staging bytes are valid
    const auto t = std::chrono::steady_clock::now();
    float* acc = sp->grad_acc.data_ptr<float>();
    switch (sp->grad_stage.scalar_type()) {
      case at::kFloat:
        accumulate_host(acc, sp->grad_stage.data_ptr<float>(), b, e, fresh);
        break;
      case at::kHalf:
        accumulate_host(acc, sp->grad_stage.data_ptr<at::Half>(), b, e, fresh);
        break;
      case at::kBFloat16:
        accumulate_host(acc, sp->grad_stage.data_ptr<at::BFloat16>(), b, e, fresh);
        break;
      default:
        TORCH_CHECK(false, "unsupported gradient dtype");
    }
    accumulate_us_ += elapsed_us(t);
  });
  // Every chunk synchronised on `landed`, so the VRAM gradient is free to go.
  auto release = std::make_shared<std::function<void()>>([sp] { sp->grad_keep.reset(); });
  run_batch(s, body, release);
  if (apply_update) {
    steps_eager_ += 1;
    s.has_grad = false;  // cleared synchronously: step() must not update twice
    enqueue_update(s, lr, beta1, beta2, eps, weight_decay, 1.0, record_on_current(device_), nullptr);
  }
}

void OffloadAdamW::enqueue_update(Slot& s, double lr, double beta1, double beta2, double eps, double wd,
                                  double clip, EventPtr after, EventPtr from_stage) {
  Slot* sp = &s;
  s.step += 1;
  const HostHp hp = make_hp(lr, beta1, beta2, eps, wd, s.step, clip);
  const EventPtr prev_ready = s.ready;

  auto body = std::make_shared<ChunkFn>([this, sp, hp, prev_ready, from_stage](int64_t, int64_t b, int64_t e) {
    // The previous H2D may still be reading master/param_stage.
    if (prev_ready) prev_ready->synchronize();
    if (from_stage) from_stage->synchronize();  // gradient DMA landed in staging
    const auto t = std::chrono::steady_clock::now();
    adamw_dispatch(sp->master.data_ptr<float>(), sp->exp_avg.data_ptr<float>(), sp->exp_avg_sq.data_ptr<float>(),
                   from_stage ? sp->grad_stage : sp->grad_acc, sp->pdtype, sp->param_stage, b, e, hp);
    update_us_ += elapsed_us(t);
  });

  auto finalize = std::make_shared<std::function<void()>>([this, sp, after] {
    sp->grad_keep.reset();  // fused path: staging consumed, DMA long complete
    const at::Tensor& src = sp->param_stage.defined() ? sp->param_stage : sp->master;
    EventPtr done = h2d_async(sp->param, src, after);
    param_bytes_ += static_cast<int64_t>(src.nbytes());
    {
      std::lock_guard<std::mutex> lock(sp->mu);
      sp->ready = std::move(done);
      sp->has_grad = false;
      sp->acc_fresh = true;
    }
    last_finalize_ms_ = now_ms();
  });
  run_batch(s, body, finalize);
}

double OffloadAdamW::step(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                          const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                          const std::vector<double>& epss, const std::vector<double>& wds,
                          const std::vector<at::Tensor>& grads, double max_grad_norm,
                          bool compute_norm) {
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
  return step_cpu(ids, lrs, beta1s, beta2s, epss, wds, max_grad_norm, compute_norm);
}

double OffloadAdamW::step_cpu(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                              const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                              const std::vector<double>& epss, const std::vector<double>& wds,
                              double max_grad_norm, bool compute_norm) {
  // Everything enqueued on the compute stream so far (the whole backward) must
  // finish before new weights may overwrite the parameters on the GPU.
  EventPtr step_ev = record_on_current(device_);
  step_start_ms_ = now_ms();

  std::vector<size_t> active;
  for (size_t k = 0; k < ids.size(); ++k)
    if (slot(ids[k]).has_grad) active.push_back(k);

  double clip = 1.0;
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
    norm = std::sqrt(sq);
    last_norm_ = norm;
    if (max_grad_norm > 0.0) clip = std::min(1.0, max_grad_norm / (norm + 1e-6));
  }

  for (size_t k : active)
    enqueue_update(slot(ids[k]), lrs[k], beta1s[k], beta2s[k], epss[k], wds[k], clip, step_ev, nullptr);
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
  const auto t0 = std::chrono::steady_clock::now();
  EventPtr step_ev = record_on_current(device_);
  step_ev->block(rt.upd);

  const bool use_norm = max_grad_norm > 0.0 || compute_norm;
  if (use_norm) {
    TS_CUDA_CHECK(cudaMemsetAsync(norm_dev_, 0, sizeof(float), rt.upd));
    for (size_t k = 0; k < ids.size(); ++k) {
      if (!g[k].defined()) continue;
      TS_CUDA_CHECK(kernels::grad_sq_norm_accumulate(to_kernel_dtype(g[k].scalar_type()), g[k].data_ptr(),
                                                     g[k].numel(), norm_dev_, rt.sm_count, rt.upd));
    }
  }
  for (size_t k = 0; k < ids.size(); ++k) {
    if (!g[k].defined()) continue;
    Slot& s = slot(ids[k]);
    s.step += 1;
    const HostHp h = make_hp(lrs[k], beta1s[k], beta2s[k], epss[k], wds[k], s.step, 1.0);
    kernels::AdamWParams hp{h.lr, h.beta1, h.beta2, h.eps, h.weight_decay, h.step_size, h.bc2_sqrt,
                            static_cast<float>(max_grad_norm)};
    TS_CUDA_CHECK(kernels::adamw_zero_copy(to_kernel_dtype(s.pdtype), s.param.data_ptr(), g[k].data_ptr(),
                                           s.d_m, s.d_v, s.d_master, s.numel, hp,
                                           max_grad_norm > 0.0 ? norm_dev_ : nullptr, rt.sm_count, rt.upd));
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
  if (use_norm) {
    TS_CUDA_CHECK(cudaMemcpyAsync(norm_host_.data_ptr(), norm_dev_, sizeof(float), cudaMemcpyDeviceToHost, rt.upd));
    norm_ev_ = make_event(device_);
    norm_ev_->record(rt.upd);
  }
  update_us_ += elapsed_us(t0);  // host launch overhead only; device time is async
}

void OffloadAdamW::wait_ready(const std::vector<int64_t>& ids) {
  check_error();
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
  }
  check_error();
}

double OffloadAdamW::last_grad_norm() {
  if (gpu_mode_ && norm_ev_) {
    norm_ev_->synchronize();
    last_norm_ = std::sqrt(static_cast<double>(norm_host_.data_ptr<float>()[0]));
    norm_ev_.reset();
  }
  return last_norm_;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> OffloadAdamW::state(int64_t id) {
  Slot& s = slot(id);
  wait_idle(s);
  return {s.master, s.exp_avg, s.exp_avg_sq};
}

void OffloadAdamW::load_state(int64_t id, const at::Tensor& exp_avg, const at::Tensor& exp_avg_sq,
                              const at::Tensor& master, int64_t step) {
  Slot& s = slot(id);
  wait_idle(s);
  if (s.ready) s.ready->synchronize();
  TORCH_CHECK(exp_avg.numel() == s.numel && exp_avg_sq.numel() == s.numel,
              "load_state: moment shape mismatch for param ", id);
  s.exp_avg.copy_(exp_avg.reshape({-1}));
  s.exp_avg_sq.copy_(exp_avg_sq.reshape({-1}));
  if (master.defined() && s.master.defined()) {
    TORCH_CHECK(master.numel() == s.numel, "load_state: master shape mismatch for param ", id);
    s.master.copy_(master.reshape({-1}));
  } else if (s.master.defined()) {
    s.master.copy_(s.param);  // re-sync master with the live weights
  }
  s.step = step;
}

int64_t OffloadAdamW::get_step(int64_t id) { return slot(id).step; }

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
          {"grad_bytes_offloaded", static_cast<double>(grad_bytes_.load())},
          {"param_bytes_uploaded", static_cast<double>(param_bytes_.load())},
          {"update_ms_total", update_us_.load() / 1e3},
          {"accumulate_ms_total", accumulate_us_.load() / 1e3},
          {"ready_stall_ms_total", ready_stall_us_.load() / 1e3},
          {"grad_stall_ms_total", grad_stall_us_.load() / 1e3},
          {"last_step_wall_ms", gpu_mode_ ? 0.0 : last_wall}};
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
      ts.emplace_back([&, b, e] { adamw_host<float, float, false>(p.data(), m.data(), v.data(), g.data(), nullptr, b, e, hp); });
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
