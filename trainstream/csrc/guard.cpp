// TrainStream — non-finite guard flags.
//
// One small int32 array per device in mapped pinned host memory. Kernels get
// its device-visible address and store 1 into their word (kernels::GuardWord)
// when they see a NaN / Inf; the host reads the same bytes directly, with no
// stream synchronization and no D2H copy. The flags say *where* a bad value
// first appeared. Whether a step is skipped is decided from the gradients
// themselves (offload_adamw.cpp), which is exact and, under data parallel,
// identical on every rank.

#include "kernels.h"
#include "trainstream.h"

#include <mutex>

namespace trainstream {

namespace {
constexpr int kMaxGuardDevices = 64;

struct Flags {
  at::Tensor host;         // int32 [kGuardWords]
  int32_t* dev = nullptr;  // the same bytes, as the GPU addresses them
};

// Leaked on purpose, like the runtime table: freeing pinned memory from a
// static destructor would touch CUDA after the driver has shut down.
Flags** table() {
  static Flags** t = new Flags*[kMaxGuardDevices]();
  return t;
}

std::mutex& table_mutex() {
  static std::mutex* m = new std::mutex();
  return *m;
}

std::atomic<bool>& enabled() {
  static std::atomic<bool>* e = new std::atomic<bool>(true);
  return *e;
}

Flags& flags_for(int device) {
  const int dev = resolve_device(device);
  TORCH_CHECK(dev >= 0 && dev < kMaxGuardDevices, "invalid CUDA device ", dev);
  Flags** t = table();
  if (Flags* f = t[dev]) return *f;
  std::lock_guard<std::mutex> lock(table_mutex());
  if (t[dev] == nullptr) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
    auto* f = new Flags();
    f->host = pinned_empty({kernels::kGuardWords}, at::kInt, "guard", true, false);
    f->host.zero_();
    f->dev = static_cast<int32_t*>(pinned_device_pointer(f->host));
    t[dev] = f;
  }
  return *t[dev];
}
}  // namespace

int32_t* guard_flags(int device) {
  return enabled().load(std::memory_order_relaxed) ? flags_for(device).dev : nullptr;
}

int32_t* guard_flags_always(int device) { return flags_for(device).dev; }

void guard_enable(bool on) { enabled().store(on, std::memory_order_relaxed); }

bool guard_enabled() { return enabled().load(std::memory_order_relaxed); }

std::vector<int64_t> guard_read(int device, bool clear) {
  Flags& f = flags_for(device);
  int32_t* w = f.host.data_ptr<int32_t>();
  std::vector<int64_t> out(kernels::kGuardWords);
  for (int i = 0; i < kernels::kGuardWords; ++i) {
    out[static_cast<size_t>(i)] = w[i];
    if (clear) w[i] = 0;
  }
  return out;
}

const char* guard_word_name(int word) {
  switch (word) {
    case kernels::kGuardRmsNormFwd:
      return "rms_norm_forward";
    case kernels::kGuardRmsNormBwd:
      return "rms_norm_backward";
    case kernels::kGuardGatedActFwd:
      return "gated_act_forward";
    case kernels::kGuardGatedActBwd:
      return "gated_act_backward";
    case kernels::kGuardQuantize:
      return "activation_quantize";
    case kernels::kGuardGemm:
      return "fused_gemm";
    case kernels::kGuardOptimizer:
      return "optimizer";
    default:
      return "";
  }
}

}  // namespace trainstream
