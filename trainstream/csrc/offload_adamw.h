// TrainStream — offloaded AdamW engine.
//
// Two execution modes share one bookkeeping model ("slots", one per parameter):
//
//  * "cpu": gradients are streamed D2H as soon as autograd produces them; host
//    worker threads accumulate them into fp32 buffers, run AdamW on fp32 master
//    weights held in pinned RAM, and stream the updated weights back H2D on a
//    dedicated copy stream. step() returns immediately; the next forward pass
//    waits per-module (wait_ready) so layer 0 can start computing while the
//    CPU is still updating layer N.
//
//  * "gpu": gradients stay in VRAM; a CUDA kernel updates the parameter in
//    place while reading/writing the moments through mapped pinned memory
//    (zero-copy over PCIe). Moments never occupy VRAM.
//
// Moments are fp32, or bf16 with stochastic rounding (half the host memory and
// traffic). Paged slots (cpu mode) keep no up-to-date weights in VRAM: the
// host copy is authoritative and the Python ParamPager streams it in per block.
#pragma once

#include "trainstream.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <deque>

namespace trainstream {

class WorkerPool {
 public:
  WorkerPool(int threads, int device);
  ~WorkerPool();
  int size() const { return static_cast<int>(workers_.size()); }
  // Jobs submitted to the same worker run in FIFO order.
  void submit(size_t worker, std::function<void()> fn);

 private:
  struct Worker {
    std::thread thread;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::function<void()>> queue;
    bool stop = false;
  };
  std::vector<std::unique_ptr<Worker>> workers_;
};

class OffloadAdamW {
 public:
  OffloadAdamW(int device, const std::string& mode, int threads, int64_t chunk_elems,
               const std::string& moment_dtype = "float32");
  ~OffloadAdamW();

  // pin_moments=false keeps exp_avg/exp_avg_sq in ordinary RAM (cpu mode only).
  // host_init (a CPU tensor, optional) seeds the host weights instead of reading
  // `param`, for parameters whose VRAM storage is empty (paged models built on CPU).
  int64_t add_param(const at::Tensor& param, bool pin_moments = true, const at::Tensor& host_init = {});

  // ---- parameter paging (cpu mode) ----------------------------------------
  // A paged slot's update is not uploaded: the host weights (host_param) are
  // copied in by the pager whenever the owning block runs.
  void set_paged(int64_t id, bool paged);
  at::Tensor host_param(int64_t id);           // fp32 master, or the rounded staging copy
  void wait_issued(const std::vector<int64_t>& ids);  // host: queued updates finished
  void set_ready(int64_t id, const EventPtr& ev);     // next update waits for this DMA read
  void sync_host_from_param(int64_t id);       // VRAM weights -> host copies (blocking)

  // ---- data parallel (cpu mode) -------------------------------------------
  at::Tensor grad_buffer(int64_t id);  // fp32 host accumulator (reduced in place by the caller)
  // Queue the AdamW update for one slot from its accumulator, gradients scaled by `scale`.
  void update_slot(int64_t id, double lr, double beta1, double beta2, double eps, double weight_decay,
                   double scale);

  // cpu mode: stream one gradient to host and queue its accumulation. With
  // apply_update the AdamW update for this tensor is queued right behind it
  // ("eager" / optimizer-in-backward), overlapping host math with the rest of
  // backward; the H2D upload is ordered after every kernel queued so far.
  void offload_grad(int64_t id, const at::Tensor& grad, bool apply_update = false, double lr = 0.0,
                    double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0.0);

  // Queue one optimizer step for `ids` (in the order the next forward needs
  // them). `grads` is used by gpu mode only. Returns the global grad norm when
  // it was computed on the host (cpu mode with clipping / compute_norm), else -1.
  double step(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
              const std::vector<double>& beta1s, const std::vector<double>& beta2s,
              const std::vector<double>& epss, const std::vector<double>& wds,
              const std::vector<at::Tensor>& grads, double max_grad_norm, bool compute_norm,
              double grad_scale = 1.0);

  // Host-waits until each slot's update has been *issued*, then makes the
  // current compute stream wait (on the GPU) for the H2D / kernel to land.
  void wait_ready(const std::vector<int64_t>& ids);
  void synchronize();

  double last_grad_norm();
  std::tuple<at::Tensor, at::Tensor, at::Tensor> state(int64_t id);  // master|undef, m, v
  void load_state(int64_t id, const at::Tensor& exp_avg, const at::Tensor& exp_avg_sq,
                  const at::Tensor& master, int64_t step);
  int64_t get_step(int64_t id);
  std::map<std::string, double> stats();
  std::string mode() const { return gpu_mode_ ? "gpu" : "cpu"; }
  int64_t num_params() const { return static_cast<int64_t>(slots_.size()); }
  int threads() const { return pool_ ? pool_->size() : 0; }
  // Back-pressure: backward waits once more gradient bytes than this are
  // still in VRAM waiting for their D2H copy (cpu mode).
  void set_max_inflight_grad_bytes(int64_t bytes) { max_inflight_grad_bytes_ = bytes; }
  // Gradient staging buffers are pooled: a pinned buffer is taken when a
  // gradient arrives and returned once the host has consumed it. Beyond this
  // many bytes in use, the next gradient waits for one to come back.
  void set_max_staging_bytes(int64_t bytes);
  std::string moment_dtype() const {
    return moment_dtype_ == at::kBFloat16 ? "bfloat16" : moment_dtype_ == at::kByte ? "int8" : "float32";
  }

 private:
  struct Slot {
    int64_t id = 0;
    at::Tensor param;  // flat alias of the user's parameter (shares storage)
    int64_t numel = 0;
    at::ScalarType pdtype = at::kFloat;
    at::Tensor master, exp_avg, exp_avg_sq;  // master fp32; moments fp32, bf16 or 8-bit codes; host
    at::Tensor m_scale, v_scale;             // 8-bit moments: one fp32 absmax per kQBlock values
    float* d_master = nullptr;               // device aliases (gpu mode)
    void* d_m = nullptr;
    void* d_v = nullptr;
    bool paged = false;
    at::Tensor grad_stage, grad_acc, param_stage;  // cpu mode staging
    at::Tensor grad_keep;  // VRAM gradient kept alive until its DMA / kernel is done
    int64_t step = 0;
    bool has_grad = false;
    bool acc_fresh = true;
    EventPtr d2h_done;
    EventPtr ready;
    std::mutex mu;
    std::condition_variable cv;
    int inflight = 0;
    int64_t nchunks = 1;
  };
  using ChunkFn = std::function<void(int64_t chunk, int64_t begin, int64_t end)>;

  Slot& slot(int64_t id);
  size_t worker_for(int64_t id, int64_t chunk) const;
  void run_batch(Slot& s, std::shared_ptr<ChunkFn> body, std::shared_ptr<std::function<void()>> finalize);
  void wait_idle(Slot& s);
  // Queue AdamW + H2D for one slot. Reads the fp32 accumulator, or the raw
  // staging buffer once `from_stage` (its D2H event) has completed.
  void enqueue_update(Slot& s, double lr, double beta1, double beta2, double eps, double wd, double clip,
                      EventPtr after, EventPtr from_stage);
  // cpu mode: VRAM gradients are kept alive only until their D2H copy has
  // landed (the host works from the staging copy), polled on every call.
  void retire_grad(EventPtr d2h_done, at::Tensor grad);
  void acquire_stage(Slot& s, at::ScalarType dtype);
  void release_stage(Slot& s);
  void reap_grads(bool block);
  void record_error(const std::string& msg);
  void check_error();
  double step_cpu(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                  const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                  const std::vector<double>& epss, const std::vector<double>& wds,
                  double max_grad_norm, bool compute_norm, double grad_scale);
  void step_gpu(const std::vector<int64_t>& ids, const std::vector<double>& lrs,
                const std::vector<double>& beta1s, const std::vector<double>& beta2s,
                const std::vector<double>& epss, const std::vector<double>& wds,
                const std::vector<at::Tensor>& grads, double max_grad_norm, bool compute_norm);

  int device_;
  bool gpu_mode_;
  at::ScalarType moment_dtype_ = at::kFloat;
  int64_t chunk_;
  std::unique_ptr<WorkerPool> pool_;
  std::vector<std::unique_ptr<Slot>> slots_;

  std::mutex retire_mu_;
  std::deque<std::pair<EventPtr, at::Tensor>> retiring_;
  int64_t retiring_bytes_ = 0;
  std::mutex stage_mu_;
  std::condition_variable stage_cv_;
  int64_t stage_bytes_ = 0;
  int64_t stage_peak_ = 0;
  int64_t max_stage_bytes_ = int64_t(4) << 30;
  int64_t max_inflight_grad_bytes_ = int64_t(1) << 30;
  std::mutex err_mu_;
  std::string error_;

  // gpu-mode global-norm scratch
  float* norm_dev_ = nullptr;
  at::Tensor norm_host_;
  EventPtr norm_ev_;
  double last_norm_ = -1.0;

  // telemetry (microseconds / bytes)
  std::atomic<int64_t> update_us_{0}, accumulate_us_{0}, ready_stall_us_{0}, grad_stall_us_{0};
  std::atomic<int64_t> grad_bytes_{0}, param_bytes_{0}, steps_{0}, steps_eager_{0};
  std::atomic<int64_t> host_state_bytes_{0}, pinned_state_bytes_{0}, pageable_state_bytes_{0};
  std::atomic<double> step_start_ms_{0.0}, last_finalize_ms_{0.0};
};

// Throughput of the host AdamW loop (elements / second) across `threads`.
std::map<std::string, double> benchmark_cpu_adamw(int64_t numel, int threads);
// Throughput of the zero-copy kernel with moments in mapped pinned memory.
std::map<std::string, double> benchmark_zero_copy_adamw(int device, int64_t numel);

}  // namespace trainstream
