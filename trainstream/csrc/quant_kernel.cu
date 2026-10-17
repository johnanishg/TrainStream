// TrainStream — dequantization kernels for quantized frozen weights.
//
// Frozen weights (a LoRA base) can be stored blockwise-quantized:
//   nf4   4-bit NormalFloat codes (the QLoRA data type), two per byte, high
//         nibble first, one fp32 absmax scale per block of `block` values
//   int8  signed 8-bit linear codes q = round(x / absmax * 127), one fp32
//         absmax per block
// Training never multiplies with packed weights: the pager expands a layer's
// weights to bf16/fp16/fp32 right before the layer runs, with the kernels
// below, and frees the expanded copy afterwards. They are memory-bound
// (read 0.5-1 byte, write 2-4 bytes per weight) and run on TrainStream's H2D
// stream, so the expansion overlaps compute like any other prefetch.

#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trainstream {
namespace kernels {
namespace {

// QLoRA's NF4 code book (Dettmers et al., 2023): quantiles of N(0, 1),
// normalised to [-1, 1], with an exact zero.
__constant__ float kNF4[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

template <typename T>
struct Out;
template <>
struct Out<float> {
  __device__ __forceinline__ static float cvt(float x) { return x; }
};
template <>
struct Out<__half> {
  __device__ __forceinline__ static __half cvt(float x) { return __float2half_rn(x); }
};
template <>
struct Out<__nv_bfloat16> {
  __device__ __forceinline__ static __nv_bfloat16 cvt(float x) { return __float2bfloat16_rn(x); }
};

constexpr int kThreads = 256;

template <typename T>
__global__ void __launch_bounds__(kThreads)
    dequant_nf4_kernel(const uint8_t* __restrict__ q, const float* __restrict__ absmax, T* __restrict__ out,
                       int64_t n, int block) {
  const int64_t pairs = (n + 1) / 2;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < pairs; k += stride) {
    const uint8_t b = q[k];
    const int64_t i = 2 * k;
    const float s = absmax[i / block];  // block is even: both values share a scale
    out[i] = Out<T>::cvt(kNF4[b >> 4] * s);
    if (i + 1 < n) out[i + 1] = Out<T>::cvt(kNF4[b & 15] * s);
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    dequant_int8_kernel(const int8_t* __restrict__ q, const float* __restrict__ absmax, T* __restrict__ out,
                        int64_t n, int block) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < n; i += stride) {
    out[i] = Out<T>::cvt(static_cast<float>(q[i]) * (absmax[i / block] * (1.0f / 127.0f)));
  }
}

inline int grid_for(int64_t work, int sm_count) {
  int64_t blocks = (work + kThreads - 1) / kThreads;
  const int64_t cap = static_cast<int64_t>(sm_count > 0 ? sm_count : 80) * 32;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return static_cast<int>(blocks);
}

template <typename T>
cudaError_t launch(QKind kind, const void* q, const float* absmax, void* out, int64_t n, int block, int sm_count,
                   cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  if (kind == QKind::kNF4) {
    dequant_nf4_kernel<T><<<grid_for((n + 1) / 2, sm_count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(q), absmax, static_cast<T*>(out), n, block);
  } else {
    dequant_int8_kernel<T><<<grid_for(n, sm_count), kThreads, 0, stream>>>(
        static_cast<const int8_t*>(q), absmax, static_cast<T*>(out), n, block);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t dequantize(QKind kind, DType out_dtype, const void* q, const float* absmax, void* out, int64_t n,
                       int block, int sm_count, cudaStream_t stream) {
  switch (out_dtype) {
    case DType::kFloat32:
      return launch<float>(kind, q, absmax, out, n, block, sm_count, stream);
    case DType::kFloat16:
      return launch<__half>(kind, q, absmax, out, n, block, sm_count, stream);
    case DType::kBFloat16:
      return launch<__nv_bfloat16>(kind, q, absmax, out, n, block, sm_count, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace kernels
}  // namespace trainstream
