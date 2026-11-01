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
#include <cuda_fp8.h>

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

// ---------------------------------------------------------------------------
// Block-scaled formats (NVFP4, MXFP8) for Blackwell tensor-core matmuls.
// One thread per scale group. All arithmetic uses round-to-nearest intrinsics,
// so results match the plain-PyTorch reference in trainstream/quant.py bit for
// bit even under --use_fast_math.

template <typename T>
struct In;
template <>
struct In<float> {
  __device__ __forceinline__ static float cvt(float x) { return x; }
};
template <>
struct In<__half> {
  __device__ __forceinline__ static float cvt(__half x) { return __half2float(x); }
};
template <>
struct In<__nv_bfloat16> {
  __device__ __forceinline__ static float cvt(__nv_bfloat16 x) { return __bfloat162float(x); }
};

// Offset of scale (r, c) in cuBLAS's swizzled block-scale layout: 128-row x
// 4-column tiles, each stored as 32 rows of 16 bytes.
__device__ __forceinline__ int64_t swizzled(int64_t r, int64_t c, int64_t col_tiles) {
  const int64_t rr = r & 127;
  return (((r >> 7) * col_tiles + (c >> 2)) * 32 + (rr & 31)) * 16 + (rr >> 5) * 4 + (c & 3);
}

__constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// N values written with 16-byte stores (the destination is 16-byte aligned:
// rows start at multiples of 32 elements of a 256-byte-aligned allocation).
template <typename T, int N>
__device__ __forceinline__ void store16(T* dst, const T (&vals)[N]) {
  static_assert((N * sizeof(T)) % 16 == 0, "whole 16-byte stores");
  uint4* d = reinterpret_cast<uint4*>(dst);
  const uint4* s = reinterpret_cast<const uint4*>(vals);
#pragma unroll
  for (int k = 0; k < static_cast<int>(N * sizeof(T) / 16); ++k) d[k] = s[k];
}

__device__ __forceinline__ uint8_t e2m1_code(float v) {
  const float m = fminf(fabsf(v), 6.0f);
  // nearest code; a tie goes to the smaller magnitude
  const uint8_t c = (m > 0.25f) + (m > 0.75f) + (m > 1.25f) + (m > 1.75f) + (m > 2.5f) + (m > 3.5f) + (m > 5.0f);
  return c | (v < 0.0f ? 8 : 0);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

__device__ __forceinline__ uint8_t float_to_e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    quant_nvfp4_kernel(const T* __restrict__ x, int64_t rows, int64_t cols, const float* __restrict__ gdec,
                       uint8_t* __restrict__ q, uint8_t* __restrict__ scales) {
  const int64_t groups_per_row = cols / 16;
  const int64_t col_tiles = (groups_per_row + 3) / 4;
  const int64_t total = rows * groups_per_row;
  const float g = *gdec;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total; k += stride) {
    const int64_t r = k / groups_per_row, c = k % groups_per_row;
    const T* src = x + r * cols + c * 16;
    float v[16];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      v[j] = In<T>::cvt(src[j]);
      amax = fmaxf(amax, fabsf(v[j]));
    }
    const float bs = fminf(__fdiv_rn(__fdiv_rn(amax, 6.0f), g), 448.0f);
    const uint8_t bs8 = float_to_e4m3(bs);
    float dec = __fmul_rn(e4m3_to_float(bs8), g);
    if (dec == 0.0f) dec = 1e-30f;
    scales[swizzled(r, c, col_tiles)] = bs8;
    uint8_t* dst = q + r * (cols / 2) + c * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint8_t lo = e2m1_code(__fdiv_rn(v[2 * j], dec));
      const uint8_t hi = e2m1_code(__fdiv_rn(v[2 * j + 1], dec));
      dst[j] = lo | (hi << 4);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    quant_mxfp8_kernel(const T* __restrict__ x, int64_t rows, int64_t cols, uint8_t* __restrict__ q,
                       uint8_t* __restrict__ scales) {
  const int64_t groups_per_row = cols / 32;
  const int64_t col_tiles = (groups_per_row + 3) / 4;
  const int64_t total = rows * groups_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total; k += stride) {
    const int64_t r = k / groups_per_row, c = k % groups_per_row;
    const T* src = x + r * cols + c * 32;
    float amax = 0.0f;
#pragma unroll 8
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(In<T>::cvt(src[j])));
    // OCP MX: shared exponent = floor(log2(amax)) - emax(e4m3) = floor(log2(amax)) - 8
    int e = -127;
    if (amax > 0.0f) {
      int ex;
      frexpf(amax, &ex);  // amax = m * 2^ex, m in [0.5, 1)
      e = ex - 1 - 8;
      e = e < -127 ? -127 : (e > 127 ? 127 : e);
    }
    scales[swizzled(r, c, col_tiles)] = static_cast<uint8_t>(e + 127);
    uint8_t* dst = q + r * cols + c * 32;
#pragma unroll 8
    for (int j = 0; j < 32; ++j) {
      const float v = fminf(fmaxf(ldexpf(In<T>::cvt(src[j]), -e), -448.0f), 448.0f);
      dst[j] = float_to_e4m3(v);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    dequant_nvfp4_kernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ scales, float g,
                         T* __restrict__ out, int64_t rows, int64_t cols) {
  const int64_t groups_per_row = cols / 16;
  const int64_t col_tiles = (groups_per_row + 3) / 4;
  const int64_t total = rows * groups_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total; k += stride) {
    const int64_t r = k / groups_per_row, c = k % groups_per_row;
    const float s = __fmul_rn(e4m3_to_float(scales[swizzled(r, c, col_tiles)]), g);
    const uint2 packed = *reinterpret_cast<const uint2*>(q + r * (cols / 2) + c * 8);  // 8 bytes = 16 codes
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&packed);
    alignas(16) T vals[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint8_t b = src[j];
      const float lo = kE2M1[b & 7] * ((b & 8) ? -1.0f : 1.0f);
      const float hi = kE2M1[(b >> 4) & 7] * ((b & 128) ? -1.0f : 1.0f);
      vals[2 * j] = Out<T>::cvt(__fmul_rn(lo, s));
      vals[2 * j + 1] = Out<T>::cvt(__fmul_rn(hi, s));
    }
    store16<T, 16>(out + r * cols + c * 16, vals);
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    dequant_mxfp8_kernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ scales, T* __restrict__ out,
                         int64_t rows, int64_t cols) {
  const int64_t groups_per_row = cols / 32;
  const int64_t col_tiles = (groups_per_row + 3) / 4;
  const int64_t total = rows * groups_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total; k += stride) {
    const int64_t r = k / groups_per_row, c = k % groups_per_row;
    const int e = static_cast<int>(scales[swizzled(r, c, col_tiles)]) - 127;
    alignas(16) uint8_t src[32];
    *reinterpret_cast<uint4*>(src) = *reinterpret_cast<const uint4*>(q + r * cols + c * 32);
    *reinterpret_cast<uint4*>(src + 16) = *reinterpret_cast<const uint4*>(q + r * cols + c * 32 + 16);
    alignas(16) T vals[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) vals[j] = Out<T>::cvt(ldexpf(e4m3_to_float(src[j]), e));
    store16<T, 32>(out + r * cols + c * 32, vals);
  }
}

template <typename T>
cudaError_t launch_quant_bs(QKind kind, const void* x, int64_t rows, int64_t cols, const float* gdec, uint8_t* q,
                            uint8_t* scales, int sm_count, cudaStream_t stream) {
  const int64_t groups = rows * (cols / blockscaled_group(kind));
  if (groups == 0) return cudaSuccess;
  if (kind == QKind::kNVFP4) {
    quant_nvfp4_kernel<T><<<grid_for(groups, sm_count), kThreads, 0, stream>>>(static_cast<const T*>(x), rows, cols,
                                                                                 gdec, q, scales);
  } else {
    quant_mxfp8_kernel<T><<<grid_for(groups, sm_count), kThreads, 0, stream>>>(static_cast<const T*>(x), rows, cols,
                                                                                 q, scales);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_dequant_bs(QKind kind, const uint8_t* q, const uint8_t* scales, float g, void* out, int64_t rows,
                              int64_t cols, int sm_count, cudaStream_t stream) {
  const int64_t groups = rows * (cols / blockscaled_group(kind));
  if (groups == 0) return cudaSuccess;
  if (kind == QKind::kNVFP4) {
    dequant_nvfp4_kernel<T><<<grid_for(groups, sm_count), kThreads, 0, stream>>>(q, scales, g, static_cast<T*>(out),
                                                                                   rows, cols);
  } else {
    dequant_mxfp8_kernel<T><<<grid_for(groups, sm_count), kThreads, 0, stream>>>(q, scales, static_cast<T*>(out),
                                                                                   rows, cols);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t quantize_blockscaled(QKind kind, DType in_dtype, const void* x, int64_t rows, int64_t cols,
                                 const float* global_decode, uint8_t* q, uint8_t* scales, int sm_count,
                                 cudaStream_t stream) {
  if ((kind != QKind::kNVFP4 && kind != QKind::kMXFP8) || cols % 32 != 0) return cudaErrorInvalidValue;
  switch (in_dtype) {
    case DType::kFloat32:
      return launch_quant_bs<float>(kind, x, rows, cols, global_decode, q, scales, sm_count, stream);
    case DType::kFloat16:
      return launch_quant_bs<__half>(kind, x, rows, cols, global_decode, q, scales, sm_count, stream);
    case DType::kBFloat16:
      return launch_quant_bs<__nv_bfloat16>(kind, x, rows, cols, global_decode, q, scales, sm_count, stream);
  }
  return cudaErrorInvalidValue;
}

cudaError_t dequantize_blockscaled(QKind kind, DType out_dtype, const uint8_t* q, const uint8_t* scales,
                                   float global_decode, void* out, int64_t rows, int64_t cols, int sm_count,
                                   cudaStream_t stream) {
  if ((kind != QKind::kNVFP4 && kind != QKind::kMXFP8) || cols % 32 != 0) return cudaErrorInvalidValue;
  switch (out_dtype) {
    case DType::kFloat32:
      return launch_dequant_bs<float>(kind, q, scales, global_decode, out, rows, cols, sm_count, stream);
    case DType::kFloat16:
      return launch_dequant_bs<__half>(kind, q, scales, global_decode, out, rows, cols, sm_count, stream);
    case DType::kBFloat16:
      return launch_dequant_bs<__nv_bfloat16>(kind, q, scales, global_decode, out, rows, cols, sm_count, stream);
  }
  return cudaErrorInvalidValue;
}

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
