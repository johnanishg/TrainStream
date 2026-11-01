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
// With bf16_moments the moments are bf16 and written with stochastic rounding
// (counter-based random bits keyed by `seed` and the element index), so tiny
// per-step increments to exp_avg_sq are not lost to round-to-nearest.
cudaError_t adamw_zero_copy(DType dtype, void* param, const void* grad, void* exp_avg,
                            void* exp_avg_sq, float* master, int64_t n,
                            const AdamWParams& hp, const float* grad_sq_norm, int sm_count,
                            cudaStream_t stream, bool bf16_moments = false, uint32_t seed = 0);

// Blockwise-quantized frozen weights (see quant_kernel.cu).
enum class QKind : int { kNF4 = 0, kInt8 = 1, kNVFP4 = 2, kMXFP8 = 3 };

// out[0..n) = dequantized values; one fp32 absmax per `block` values (block even).
cudaError_t dequantize(QKind kind, DType out_dtype, const void* q, const float* absmax, void* out, int64_t n,
                       int block, int sm_count, cudaStream_t stream);

// Block-scaled formats that Blackwell tensor cores multiply directly, for a
// row-major [rows, cols] matrix with cols % 32 == 0:
//   kNVFP4  E2M1 codes, two per byte (first value in the low nibble), one e4m3
//           scale per 16 values; value = e2m1 * scale * global_decode
//   kMXFP8  e4m3 values, one e8m0 (power-of-two) scale per 32 values
// Scales are written in cuBLAS's swizzled layout (128 x 4 tiles); the scale
// buffer must be blockscaled_scale_bytes() long and zero-filled beforehand.
inline int64_t blockscaled_group(QKind kind) { return kind == QKind::kNVFP4 ? 16 : 32; }
inline int64_t blockscaled_scale_bytes(QKind kind, int64_t rows, int64_t cols) {
  const int64_t sc = cols / blockscaled_group(kind);
  return ((rows + 127) / 128) * 128 * ((sc + 3) / 4) * 4;
}
// `global_decode` (nvfp4 only) is read from device memory, so it can be
// computed on the GPU without a host round trip. `mask` (nvfp4 only, one byte
// per column, may be null): columns whose byte is non-zero are quantized as 0.
cudaError_t quantize_blockscaled(QKind kind, DType in_dtype, const void* x, int64_t rows, int64_t cols,
                                 const float* global_decode, uint8_t* q, uint8_t* scales, int sm_count,
                                 cudaStream_t stream, const uint8_t* mask = nullptr);
// amax[c] = max(amax[c], max over rows of |x[r, c]|); amax must start at 0.
cudaError_t col_absmax(DType in_dtype, const void* x, int64_t rows, int64_t cols, float* amax, cudaStream_t stream);
// out[r, j] = W[r, idx[j]] (row-major [rows, k]) for an nvfp4-packed W [rows, cols].
cudaError_t dequantize_nvfp4_cols(DType out_dtype, const uint8_t* q, const uint8_t* scales, const int64_t* idx,
                                  int64_t k, float global_decode, void* out, int64_t rows, int64_t cols, int sm_count,
                                  cudaStream_t stream);
cudaError_t dequantize_blockscaled(QKind kind, DType out_dtype, const uint8_t* q, const uint8_t* scales,
                                   float global_decode, void* out, int64_t rows, int64_t cols, int sm_count,
                                   cudaStream_t stream);

// out[0] += sum(grad[i]^2). Used for global gradient-norm clipping on the GPU path.
cudaError_t grad_sq_norm_accumulate(DType dtype, const void* grad, int64_t n, float* out,
                                    int sm_count, cudaStream_t stream);

}  // namespace kernels
}  // namespace trainstream
