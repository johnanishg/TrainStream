// TrainStream — CUDA kernels for the zero-copy offloaded AdamW path.
//
// The optimizer moments are never resident in VRAM: the kernel dereferences
// mapped pinned host pointers directly, so each warp-wide load becomes a PCIe
// read request and each store a posted PCIe write. The kernel is therefore
// PCIe-bound by design (~16-24 bytes of host traffic per parameter), and the
// main tuning lever is keeping enough independent requests in flight to cover
// the ~1-2 us PCIe round-trip latency. We do that with a 4-way unrolled
// "load everything first, then compute" body and a grid sized to fill every SM.

#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trainstream {
namespace kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kUnroll = 4;
constexpr int kBlocksPerSm = 16;

template <typename T>
struct Cvt;

template <>
struct Cvt<float> {
  __device__ __forceinline__ static float load(float x) { return x; }
  __device__ __forceinline__ static float store(float x) { return x; }
};

template <>
struct Cvt<__half> {
  __device__ __forceinline__ static float load(__half x) { return __half2float(x); }
  __device__ __forceinline__ static __half store(float x) { return __float2half_rn(x); }
};

template <>
struct Cvt<__nv_bfloat16> {
  __device__ __forceinline__ static float load(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ __forceinline__ static __nv_bfloat16 store(float x) { return __float2bfloat16_rn(x); }
};

// Moment storage: fp32 as-is, or bf16 with stochastic rounding.
__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

template <typename M>
struct Moment;

template <>
struct Moment<float> {
  __device__ __forceinline__ static float load(float x) { return x; }
  __device__ __forceinline__ static float store(float x, uint32_t) { return x; }
};

template <>
struct Moment<__nv_bfloat16> {
  __device__ __forceinline__ static float load(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ __forceinline__ static __nv_bfloat16 store(float x, uint32_t r) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u) bits += r & 0xffffu;  // inf/nan pass through
    return __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16));
  }
};

inline int grid_for(int64_t n, int per_block, int sm_count) {
  int64_t blocks = (n + per_block - 1) / per_block;
  const int64_t cap = static_cast<int64_t>(sm_count > 0 ? sm_count : 80) * kBlocksPerSm;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return static_cast<int>(blocks);
}

template <typename T, typename M>
__global__ void __launch_bounds__(kThreads)
    adamw_zero_copy_kernel(T* __restrict__ param, const T* __restrict__ grad,
                           M* __restrict__ exp_avg, M* __restrict__ exp_avg_sq,
                           float* __restrict__ master, int64_t n, AdamWParams hp,
                           const float* __restrict__ grad_sq_norm, uint32_t seed) {
  float clip = 1.0f;
  if (grad_sq_norm != nullptr) {
    const float norm = sqrtf(*grad_sq_norm);
    clip = fminf(1.0f, hp.max_grad_norm / (norm + 1e-6f));
  }
  const float decay = 1.0f - hp.lr * hp.weight_decay;
  const float one_minus_b1 = 1.0f - hp.beta1;
  const float one_minus_b2 = 1.0f - hp.beta2;

  const int64_t tile = static_cast<int64_t>(kThreads) * kUnroll;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * tile;

  for (int64_t base = static_cast<int64_t>(blockIdx.x) * tile + threadIdx.x; base < n;
       base += stride) {
    float g[kUnroll], m[kUnroll], v[kUnroll], p[kUnroll];

    // Phase 1: issue every load for the unrolled lanes before consuming any of
    // them, so up to 3*kUnroll host reads per thread are in flight together.
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int64_t i = base + static_cast<int64_t>(u) * kThreads;
      if (i < n) {
        g[u] = Cvt<T>::load(grad[i]) * clip;
        m[u] = Moment<M>::load(exp_avg[i]);
        v[u] = Moment<M>::load(exp_avg_sq[i]);
        p[u] = master != nullptr ? master[i] : Cvt<T>::load(param[i]);
      }
    }

    // Phase 2: AdamW math (matches torch.optim.AdamW, foreach=False semantics).
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int64_t i = base + static_cast<int64_t>(u) * kThreads;
      if (i < n) {
        float pu = p[u] * decay;
        const float mu = m[u] + one_minus_b1 * (g[u] - m[u]);  // lerp
        const float vu = hp.beta2 * v[u] + one_minus_b2 * g[u] * g[u];
        const float denom = sqrtf(vu) / hp.bias_correction2_sqrt + hp.eps;
        pu -= hp.step_size * (mu / denom);
        const uint32_t r = mix32(static_cast<uint32_t>(i) * 0x9e3779b1u ^ seed);
        exp_avg[i] = Moment<M>::store(mu, r);
        exp_avg_sq[i] = Moment<M>::store(vu, mix32(r ^ 0x85ebca6bu));
        if (master != nullptr) master[i] = pu;
        param[i] = Cvt<T>::store(pu);
      }
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    sq_norm_kernel(const T* __restrict__ grad, int64_t n, float* __restrict__ out) {
  __shared__ float warp_sums[kThreads / 32];
  float acc = 0.0f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < n; i += stride) {
    const float x = Cvt<T>::load(grad[i]);
    acc += x * x;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    acc = threadIdx.x < (kThreads / 32) ? warp_sums[threadIdx.x] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0) atomicAdd(out, acc);
  }
}

template <typename T>
cudaError_t launch_adamw(void* param, const void* grad, void* m, void* v, float* master,
                         int64_t n, const AdamWParams& hp, const float* norm, int sm_count,
                         cudaStream_t stream, bool bf16_moments, uint32_t seed) {
  if (n == 0) return cudaSuccess;
  const int grid = grid_for(n, kThreads * kUnroll, sm_count);
  if (bf16_moments) {
    adamw_zero_copy_kernel<T, __nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        static_cast<T*>(param), static_cast<const T*>(grad), static_cast<__nv_bfloat16*>(m),
        static_cast<__nv_bfloat16*>(v), master, n, hp, norm, seed);
  } else {
    adamw_zero_copy_kernel<T, float><<<grid, kThreads, 0, stream>>>(
        static_cast<T*>(param), static_cast<const T*>(grad), static_cast<float*>(m),
        static_cast<float*>(v), master, n, hp, norm, seed);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_sq_norm(const void* grad, int64_t n, float* out, int sm_count,
                           cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  const int grid = grid_for(n, kThreads, sm_count);
  sq_norm_kernel<T><<<grid, kThreads, 0, stream>>>(static_cast<const T*>(grad), n, out);
  return cudaGetLastError();
}

}  // namespace

cudaError_t adamw_zero_copy(DType dtype, void* param, const void* grad, void* exp_avg,
                            void* exp_avg_sq, float* master, int64_t n,
                            const AdamWParams& hp, const float* grad_sq_norm, int sm_count,
                            cudaStream_t stream, bool bf16_moments, uint32_t seed) {
  switch (dtype) {
    case DType::kFloat32:
      return launch_adamw<float>(param, grad, exp_avg, exp_avg_sq, master, n, hp, grad_sq_norm,
                                 sm_count, stream, bf16_moments, seed);
    case DType::kFloat16:
      return launch_adamw<__half>(param, grad, exp_avg, exp_avg_sq, master, n, hp, grad_sq_norm,
                                  sm_count, stream, bf16_moments, seed);
    case DType::kBFloat16:
      return launch_adamw<__nv_bfloat16>(param, grad, exp_avg, exp_avg_sq, master, n, hp,
                                         grad_sq_norm, sm_count, stream, bf16_moments, seed);
  }
  return cudaErrorInvalidValue;
}

cudaError_t grad_sq_norm_accumulate(DType dtype, const void* grad, int64_t n, float* out,
                                    int sm_count, cudaStream_t stream) {
  switch (dtype) {
    case DType::kFloat32:
      return launch_sq_norm<float>(grad, n, out, sm_count, stream);
    case DType::kFloat16:
      return launch_sq_norm<__half>(grad, n, out, sm_count, stream);
    case DType::kBFloat16:
      return launch_sq_norm<__nv_bfloat16>(grad, n, out, sm_count, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace kernels
}  // namespace trainstream
