// TrainStream — device kernel launch interface.
//
// This header is deliberately free of PyTorch includes so that nvcc only has to
// parse CUDA runtime headers when compiling offload_adamw_kernel.cu. The C++
// translation units translate at::ScalarType into kernels::DType before calling.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace trainstream {
namespace kernels {

enum class DType : int { kFloat32 = 0, kFloat16 = 1, kBFloat16 = 2 };

// Hyper-parameters for one AdamW step on one tensor. Bias corrections are folded
// on the host so the kernel does no pow() per element:
//   step_size              = lr / (1 - beta1^t)
//   bias_correction2_sqrt  = sqrt(1 - beta2^t)
struct AdamWParams {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float step_size;
  float bias_correction2_sqrt;
  float max_grad_norm;  // only read when a grad_sq_norm pointer is supplied
};

// Fused AdamW where the optimizer state lives in *mapped pinned host memory*.
// `exp_avg`, `exp_avg_sq` and (optionally) `master` are device-visible aliases of
// cudaHostAlloc(cudaHostAllocMapped) buffers, so every state read/write is a PCIe
// transaction and the moments never occupy VRAM. `param` and `grad` live in VRAM.
// If `master` is non-null the fp32 master copy is updated and `param` receives a
// rounded copy (mixed-precision training with fp16/bf16 weights).
// If `grad_sq_norm` is non-null it points at a device float holding sum(g^2) over
// all gradients; the kernel derives the global-norm clip coefficient from it.
cudaError_t adamw_zero_copy(DType dtype, void* param, const void* grad, float* exp_avg,
                            float* exp_avg_sq, float* master, int64_t n,
                            const AdamWParams& hp, const float* grad_sq_norm, int sm_count,
                            cudaStream_t stream);

// out[0] += sum(grad[i]^2). Used for global gradient-norm clipping on the GPU path.
cudaError_t grad_sq_norm_accumulate(DType dtype, const void* grad, int64_t n, float* out,
                                    int sm_count, cudaStream_t stream);

}  // namespace kernels
}  // namespace trainstream
