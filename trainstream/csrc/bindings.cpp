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

  // ---- transfers ------------------------------------------------------------
  m.def("d2h_async", &d2h_async, py::arg("dst"), py::arg("src"));
  m.def("h2d_async", &h2d_async, py::arg("dst"), py::arg("src"), py::arg("after") = EventPtr());
  m.def("h2d_alloc_async", &h2d_alloc_async, py::arg("src"), py::arg("after") = EventPtr(),
        py::arg("device") = -1, py::arg("sizes") = std::vector<int64_t>{},
        py::arg("strides") = std::vector<int64_t>{});
  m.def("transfer_stats", &transfer_stats, py::arg("device") = -1);
  m.def("dequantize_async", &dequantize_async, py::arg("dst"), py::arg("q"), py::arg("absmax"), py::arg("kind"),
        py::arg("block"), py::arg("after") = EventPtr());
  m.def("synchronize_streams", [](int device) {
        Runtime& rt = runtime(device);
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.d2h));
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.h2d));
        TS_CUDA_CHECK(cudaStreamSynchronize(rt.upd));
      }, py::arg("device") = -1, nogil());

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
      .def("state", &OffloadAdamW::state, nogil())
      .def("load_state",
           [](OffloadAdamW& e, int64_t id, const at::Tensor& m, const at::Tensor& v,
              const std::optional<at::Tensor>& master, int64_t step) {
             e.load_state(id, m, v, master.value_or(at::Tensor()), step);
           },
           py::arg("id"), py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("master") = py::none(),
           py::arg("step") = 0, nogil())
      .def("get_step", &OffloadAdamW::get_step)
      .def("stats", &OffloadAdamW::stats)
      .def_property_readonly("mode", &OffloadAdamW::mode)
      .def_property_readonly("num_params", &OffloadAdamW::num_params)
      .def_property_readonly("threads", &OffloadAdamW::threads)
      .def_property_readonly("moment_dtype", &OffloadAdamW::moment_dtype);
}
