// TrainStream — Python bindings (module: trainstream._C).
//
// Every call that can block on the GPU or on worker threads releases the GIL,
// so autograd's backward thread and the training thread never deadlock on it.

#include <torch/extension.h>

#include <cstdio>
#include <exception>

#include "offload_adamw.h"
#include "trainstream.h"

namespace py = pybind11;
using namespace trainstream;

namespace {
using nogil = py::call_guard<py::gil_scoped_release>;

// An exception escaping a destructor (e.g. a CUDA error surfacing inside the
// caching allocator's free path) calls std::terminate, and the default abort
// handler prints only a stack. Print the message first, then chain.
std::terminate_handler g_prev_terminate = nullptr;
void print_then_terminate() {
  if (auto ex = std::current_exception()) {
    try {
      std::rethrow_exception(ex);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[trainstream] terminate: %s\n", e.what());
    } catch (...) {
      std::fprintf(stderr, "[trainstream] terminate: unknown exception\n");
    }
    std::fflush(stderr);
  }
  if (g_prev_terminate) g_prev_terminate();
  std::abort();
}

std::map<std::string, std::string> build_info() {
  int runtime_version = 0, driver_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  return {{"cuda_runtime", std::to_string(runtime_version)},
          {"cuda_driver", std::to_string(driver_version)},
          {"compiled_with_cuda", std::to_string(CUDART_VERSION)},
#ifdef _MSVC_LANG
          {"cxx_standard", std::to_string(_MSVC_LANG)}};  // MSVC pins __cplusplus to 199711L
#else
          {"cxx_standard", std::to_string(__cplusplus)}};
#endif
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "TrainStream native runtime: pinned memory, DMA streams, offloaded AdamW";
  g_prev_terminate = std::set_terminate(print_then_terminate);

  m.def("build_info", &build_info);
  m.def("now_ms", &now_ms);
  m.def("init", [](int device) { runtime(device); }, py::arg("device") = -1);
  m.def("is_dense", [](const at::Tensor& t) { return t.is_non_overlapping_and_dense(); },
        "True if the tensor occupies exactly numel*itemsize contiguous bytes (any dimension order).");

  // ---- events -------------------------------------------------------------
  py::class_<Event, std::shared_ptr<Event>>(m, "Event")
      .def(py::init([](int device, bool timing) { return make_event(resolve_device(device), timing); }),
           py::arg("device") = -1, py::arg("timing") = false)
      .def("record", [](Event& e) { e.record(compute_stream(e.device())); },
           "Record on the current PyTorch compute stream.")
      .def("query", &Event::query)
      .def("synchronize", &Event::synchronize, nogil())
      .def("wait", [](Event& e) { e.block(compute_stream(e.device())); },
           "Make the current compute stream wait for this event (GPU-side, non-blocking for the host).")
      .def("elapsed_ms", &Event::elapsed_ms, nogil())
      .def_property_readonly("recorded", &Event::recorded)
      .def_property_readonly("device", &Event::device);
  m.def("record_event", &record_on_current, py::arg("device") = -1, py::arg("timing") = false);

  // ---- pinned memory --------------------------------------------------------
  m.def("pinned_empty", &pinned_empty, py::arg("sizes"), py::arg("dtype"), py::arg("tag") = "user",
        py::arg("mapped") = false, py::arg("cached") = true);
  m.def("pinned_stats", &pinned_stats);
  m.def("pinned_tag_bytes", &pinned_tag_bytes);
  m.def("pinned_log", &pinned_log, py::arg("limit") = 256);
  m.def("pinned_empty_cache", &pinned_empty_cache, nogil());
  m.def("pinned_set_cache_limit", &pinned_set_cache_limit);
  m.def("pinned_reserve", &pinned_reserve, py::arg("bytes"), py::arg("slab") = 0, py::arg("cap") = 0, nogil());
  m.def("pinned_arena_trim", &pinned_arena_trim, nogil());

  // ---- transfers ------------------------------------------------------------
  m.def("d2h_async", &d2h_async, py::arg("dst"), py::arg("src"));
  m.def("h2d_async", &h2d_async, py::arg("dst"), py::arg("src"), py::arg("after") = EventPtr());
  m.def("h2d_alloc_async", &h2d_alloc_async, py::arg("src"), py::arg("after") = EventPtr(),
        py::arg("device") = -1, py::arg("sizes") = std::vector<int64_t>{},
        py::arg("strides") = std::vector<int64_t>{});
  m.def("h2d_alloc_chunked_async", &h2d_alloc_chunked_async, py::arg("src"), py::arg("after") = EventPtr(),
        py::arg("device") = -1, py::arg("sizes") = std::vector<int64_t>{}, py::arg("bounds") = std::vector<int64_t>{});
  m.def("h2d_bounce_async", &h2d_bounce_async, py::arg("dsts"), py::arg("srcs"), py::arg("after") = EventPtr(),
        py::arg("slots") = 3, nogil());
  m.def("h2d_pages_async", &h2d_pages_async, py::arg("dst"), py::arg("pages"), py::arg("after") = EventPtr());
  m.def("transfer_stats", &transfer_stats, py::arg("device") = -1);
  m.def("dequantize_async", &dequantize_async, py::arg("dst"), py::arg("q"), py::arg("absmax"), py::arg("kind"),
        py::arg("block"), py::arg("after") = EventPtr());
  m.def("peer_topology", &peer_topology, py::arg("probe_bytes") = 0, nogil());
  m.def("peer_copy_gbps", &peer_copy_gbps, py::arg("src_ptr"), py::arg("src_device"), py::arg("bytes"),
        py::arg("device") = -1, nogil());
  m.def("pci_bus_id", &pci_bus_id, py::arg("device"));
  m.def("peer_pool",
        [](int64_t bytes, int device) {
          auto [t, raw] = peer_pool(bytes, device);
          return py::make_tuple(t, py::bytes(reinterpret_cast<const char*>(raw.data()), raw.size()));
        },
        py::arg("bytes"), py::arg("device") = -1);
  m.def("peer_open",
        [](py::bytes handle, int device) {
          const std::string s = handle;
          return peer_open(std::vector<uint8_t>(s.begin(), s.end()), device);
        },
        py::arg("handle"), py::arg("device") = -1);
  m.def("peer_close", &peer_close, py::arg("ptr"), py::arg("device") = -1);
  m.def("peer_copy_async", &peer_copy_async, py::arg("dst"), py::arg("src_ptr"), py::arg("offset"),
        py::arg("src_device"), py::arg("after") = EventPtr());
  m.def("shared_host", &shared_host, py::arg("name"), py::arg("bytes"), py::arg("create"), py::arg("pin") = true);
  m.def("shm_reducer_bytes", &shm_reducer_bytes, py::arg("world"), py::arg("capacity"));
  m.def("shm_allreduce", &shm_allreduce, py::arg("buf"), py::arg("segment"), py::arg("rank"), py::arg("world"),
        py::arg("capacity"), py::arg("seq"), nogil());
  m.def("quantize_blockscaled", &quantize_blockscaled, py::arg("x"), py::arg("kind"),
        py::arg("global_decode") = c10::optional<at::Tensor>(), py::arg("mask") = c10::optional<at::Tensor>());
  m.def("col_absmax", &col_absmax, py::arg("x"));
  m.def("rms_norm_forward", &rms_norm_forward, py::arg("x"), py::arg("w"), py::arg("eps"), py::arg("gemma"));
  m.def("rms_norm_backward", &rms_norm_backward, py::arg("dy"), py::arg("x"), py::arg("w"), py::arg("rstd"),
        py::arg("gemma"), py::arg("need_dw"));
  m.def("rope", &rope, py::arg("x"), py::arg("cos"), py::arg("sin"), py::arg("backward"));
  m.def("gated_act_forward", &gated_act_forward, py::arg("g"), py::arg("u"), py::arg("act"));
  m.def("gated_act_backward", &gated_act_backward, py::arg("dh"), py::arg("g"), py::arg("u"), py::arg("act"));
  m.def("gemm_fp4_supported", &gemm_fp4_supported, py::arg("device") = -1,
        "Whether this build and GPU have TrainStream's own FP4 tensor-core kernel (consumer Blackwell, sm_120).");
  m.def("gemm_fp4_front", &gemm_fp4_front, py::arg("xq"), py::arg("xs"), py::arg("xg"), py::arg("wq"), py::arg("ws"),
        py::arg("wg"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("xa"), py::arg("sb"), py::arg("act") = 0,
        py::arg("want_h") = true, py::arg("want_out") = false);
  m.def("dequantize_blockscaled_cols", &dequantize_blockscaled_cols, py::arg("q"), py::arg("scales"), py::arg("idx"),
        py::arg("rows"), py::arg("cols"), py::arg("global_decode"), py::arg("dtype") = at::kBFloat16);
  m.def("dequantize_blockscaled", &dequantize_blockscaled, py::arg("dst"), py::arg("q"), py::arg("scales"),
        py::arg("kind"), py::arg("global_decode") = 1.0);
  m.def("dequantize_blockscaled_async", &dequantize_blockscaled_async, py::arg("dst"), py::arg("q"),
        py::arg("scales"), py::arg("kind"), py::arg("global_decode") = 1.0, py::arg("after") = EventPtr());
  // ---- ring all-reduce --------------------------------------------------------------
  py::class_<Ring>(m, "Ring")
      .def(py::init<int64_t, int64_t, int, int, int>(), py::arg("to_next"), py::arg("from_prev"), py::arg("rank"),
           py::arg("world"), py::arg("timeout_s") = 600)
      .def("allreduce", &Ring::allreduce, nogil())
      .def("stats", &Ring::stats);

  // ---- disk tier ----------------------------------------------------------------
  py::class_<DiskStage, std::shared_ptr<DiskStage>>(m, "DiskStage")
      .def_property_readonly("ready", [](const DiskStage& s) { return disk_stage_ready(s); },
                             "True once every read of this stage has landed in its pinned slot.")
      .def_property_readonly("hit", [](const DiskStage& s) { return s.hit; })
      .def_property_readonly("bytes", [](const DiskStage& s) { return s.bytes; })
      .def_property_readonly("key", [](const DiskStage& s) { return s.key; })
      .def("wait", [](const DiskStage& s) { disk_stage_wait(s); }, nogil());
  m.def("disk_stage", &disk_stage, py::arg("paths"), py::arg("offsets"), py::arg("nbytes"), py::arg("key") = -1,
        py::arg("device") = -1, py::arg("slots") = 4, nogil());
  m.def("h2d_disk_async", &h2d_disk_async, py::arg("dsts"), py::arg("stage"), py::arg("after") = EventPtr());
  m.def("disk_invalidate", &disk_invalidate, py::arg("device") = -1);
  m.def("disk_release", &disk_release, py::arg("device") = -1, nogil());
  m.def("disk_close", &disk_close, py::arg("paths") = std::vector<std::string>{});
  m.def("disk_sync", &disk_sync, py::arg("mapped"), nogil());
  m.def("disk_read", &disk_read, py::arg("dst"), py::arg("path"), py::arg("offset") = 0, nogil());
  m.def("disk_backend", &disk_backend);
  m.def("disk_stats", &disk_stats);

  // ---- non-finite guard -------------------------------------------------------
  m.def("guard_enable", &guard_enable, py::arg("on"),
        "Switch the NaN / Inf checks inside TrainStream's fused kernels on or off (process-wide).");
  m.def("guard_enabled", &guard_enabled);
  m.def("guard_read",
        [](int device, bool clear) {
          const std::vector<int64_t> words = guard_read(device, clear);
          std::map<std::string, int64_t> out;
          for (size_t i = 0; i < words.size(); ++i) {
            const std::string name = guard_word_name(static_cast<int>(i));
            if (!name.empty()) out[name] = words[i];
          }
          return out;
        },
        py::arg("device") = -1, py::arg("clear") = false,
        "Which kernels saw a NaN / Inf since the flags were last cleared (read from mapped host memory).");
  m.def("synchronize_streams", [](int device) {
        Runtime& rt = runtime(device);
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.d2h));
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.h2d));
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.upd));
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.disk));
      }, py::arg("device") = -1, nogil());

  // ---- background checkpoints ----------------------------------------------
  py::class_<CheckpointJob, std::shared_ptr<CheckpointJob>>(m, "CheckpointJob")
      .def("done", &CheckpointJob::done)
      .def("wait", &CheckpointJob::wait, nogil())
      .def_property_readonly("error", &CheckpointJob::error)
      .def_property_readonly("seconds", &CheckpointJob::seconds)
      .def_property_readonly("bytes", &CheckpointJob::bytes);
  m.def("checkpoint_write_async",
        [](const std::string& path, const py::bytes& header,
           const std::vector<std::tuple<at::Tensor, int64_t, EventPtr, int64_t, bool>>& items, OffloadAdamW* engine,
           const std::vector<int64_t>& targets, const std::optional<at::Tensor>& steps, int64_t steps_offset,
           int device) {
          std::vector<CheckpointItem> list;
          for (const auto& t : items) {
            CheckpointItem it;
            it.data = std::get<0>(t);
            it.offset = std::get<1>(t);
            it.after = std::get<2>(t);
            it.slot = std::get<3>(t);
            it.release = std::get<4>(t);
            list.push_back(std::move(it));
          }
          return checkpoint_write_async(path, std::string(header), std::move(list), engine, targets,
                                        steps.value_or(at::Tensor()), steps_offset, device);
        },
        py::arg("path"), py::arg("header"), py::arg("items"), py::arg("engine") = nullptr,
        py::arg("targets") = std::vector<int64_t>{}, py::arg("steps") = py::none(), py::arg("steps_offset") = 0,
        py::arg("device") = -1,
        "Write a checkpoint file on a background thread; items are (host tensor, file offset, event or None, "
        "engine slot or -1, release the slot afterwards).");
  m.def("decode_moment8", &decode_moment8, py::arg("codes"), py::arg("scale"), py::arg("is_m"));
  m.def("grad_fp8_encode", &grad_fp8_encode, py::arg("grad"), py::arg("seed") = 0);
  m.def("grad_fp8_decode", &grad_fp8_decode, py::arg("codes"), py::arg("scales"));

  // ---- benchmarks -----------------------------------------------------------
  m.def("benchmark_pcie", &benchmark_pcie, py::arg("device") = -1, py::arg("bytes") = 256LL << 20,
        py::arg("iters") = 10, nogil());
  m.def("benchmark_matmul", &benchmark_matmul, py::arg("device") = -1, py::arg("n") = 4096,
        py::arg("dtype") = at::kBFloat16, py::arg("iters") = 20, nogil());
  m.def("benchmark_cpu_adamw", &benchmark_cpu_adamw, py::arg("numel") = 1 << 24, py::arg("threads") = 0,
        nogil());
  m.def("benchmark_zero_copy_adamw", &benchmark_zero_copy_adamw, py::arg("device") = -1,
        py::arg("numel") = 1 << 24, nogil());

  // ---- optimizer engine -----------------------------------------------------
  py::class_<OffloadAdamW>(m, "OffloadAdamWEngine")
      .def(py::init<int, const std::string&, int, int64_t, const std::string&>(), py::arg("device"),
           py::arg("mode") = "cpu", py::arg("threads") = 0, py::arg("chunk_elems") = 1 << 20,
           py::arg("moment_dtype") = "float32")
      .def("add_param",
           [](OffloadAdamW& e, const at::Tensor& p, bool pin, const std::optional<at::Tensor>& host_init) {
             return e.add_param(p, pin, host_init.value_or(at::Tensor()));
           },
           py::arg("param"), py::arg("pin_moments") = true, py::arg("host_init") = py::none())
      .def("set_paged", &OffloadAdamW::set_paged, nogil())
      .def("set_max_inflight_grad_bytes", &OffloadAdamW::set_max_inflight_grad_bytes)
      .def("set_max_staging_bytes", &OffloadAdamW::set_max_staging_bytes)
      .def("host_param", &OffloadAdamW::host_param)
      .def("wait_issued", &OffloadAdamW::wait_issued, nogil())
      .def("set_ready", &OffloadAdamW::set_ready)
      .def("sync_host_from_param", &OffloadAdamW::sync_host_from_param, nogil())
      .def("grad_buffer", &OffloadAdamW::grad_buffer, nogil())
      .def("update_slot", &OffloadAdamW::update_slot, py::arg("id"), py::arg("lr"), py::arg("beta1"),
           py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("scale") = 1.0, nogil())
      .def("offload_grad", &OffloadAdamW::offload_grad, py::arg("id"), py::arg("grad"),
           py::arg("apply_update") = false, py::arg("lr") = 0.0, py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0, nogil())
      .def("step", &OffloadAdamW::step, py::arg("ids"), py::arg("lrs"), py::arg("beta1s"), py::arg("beta2s"),
           py::arg("epss"), py::arg("wds"), py::arg("grads"), py::arg("max_grad_norm") = 0.0,
           py::arg("compute_norm") = false, py::arg("grad_scale") = 1.0, nogil())
      .def("wait_ready", &OffloadAdamW::wait_ready, nogil())
      .def("synchronize", &OffloadAdamW::synchronize, nogil())
      .def("last_grad_norm", &OffloadAdamW::last_grad_norm, nogil())
      .def("raw_state", &OffloadAdamW::raw_state)
      .def("load_state_raw", &OffloadAdamW::load_state_raw, py::arg("id"), py::arg("state"), py::arg("step") = 0,
           nogil())
      .def("checkpoint_begin", &OffloadAdamW::checkpoint_begin, nogil())
      .def("checkpoint_end", &OffloadAdamW::checkpoint_end, nogil())
      .def("set_nan_guard", &OffloadAdamW::set_nan_guard, nogil())
      .def("set_fp8_gradients", &OffloadAdamW::set_fp8_gradients, nogil())
      .def_property_readonly("fp8_gradients", &OffloadAdamW::fp8_gradients)
      .def("skipped_steps", &OffloadAdamW::skipped_steps, py::arg("wait") = true, nogil())
      .def("consecutive_skips", &OffloadAdamW::consecutive_skips, py::arg("wait") = true, nogil())
      .def_property_readonly("nan_guard", &OffloadAdamW::nan_guard)
      .def("state", &OffloadAdamW::state, nogil())
      .def("load_state",
           [](OffloadAdamW& e, int64_t id, const at::Tensor& m, const at::Tensor& v,
              const std::optional<at::Tensor>& master, int64_t step) {
             e.load_state(id, m, v, master.value_or(at::Tensor()), step);
           },
           py::arg("id"), py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("master") = py::none(),
           py::arg("step") = 0, nogil())
      .def("get_step", &OffloadAdamW::get_step, nogil())
      .def("stats", &OffloadAdamW::stats)
      .def_property_readonly("mode", &OffloadAdamW::mode)
      .def_property_readonly("num_params", &OffloadAdamW::num_params)
      .def_property_readonly("threads", &OffloadAdamW::threads)
      .def_property_readonly("moment_dtype", &OffloadAdamW::moment_dtype);
}
