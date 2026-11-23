// TrainStream — fused kernels for the elementwise parts of a transformer block.
//
// A Hugging Face decoder layer runs its RMSNorm, rotary embedding and gated MLP
// activation as chains of small PyTorch ops: a cast, a square, a mean, a rsqrt,
// two multiplies and another cast for one norm, each a kernel that reads and
// writes the whole activation. Profiles of 14B LoRA steps put these at 0.5 s of
// a 2-3 s step at 2,048 tokens. The kernels below do each in one pass, compute
// in fp32 and round once, and keep less for backward:
//   rms_norm   y = w * (x * rstd)   saves rstd (one float per row) besides x
//   rope       q, k rotated in one pass, forward and backward
//   gated_act  h = act(g) * u        saves g and u; act(g) is recomputed
// They run on PyTorch's current stream, like the ops they replace.

#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trainstream {
namespace kernels {
namespace {

constexpr int kFusedThreads = 256;

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// True for NaN and +/-Inf (a bit test: --use_fast_math cannot fold it away).
__device__ __forceinline__ bool non_finite(float v) { return (__float_as_uint(v) & 0x7f800000u) == 0x7f800000u; }

inline int fused_grid(int64_t work, int sm_count) {
  int64_t blocks = (work + kFusedThreads - 1) / kFusedThreads;
  const int64_t cap = static_cast<int64_t>(sm_count > 0 ? sm_count : 80) * 32;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return static_cast<int>(blocks);
}

// Sum over a block of kFusedThreads threads; every thread gets the result.
__device__ __forceinline__ float block_sum(float v, float* shared) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kFusedThreads / 32 ? shared[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) shared[0] = v;
  }
  __syncthreads();
  v = shared[0];
  __syncthreads();  // shared is reused by the next call
  return v;
}

// ------------------------------------------------------------------ RMSNorm
// One block per row. Llama order (offset 0): y = w * T(x * rstd), the product
// rounded once more, as Hugging Face computes it; Gemma order (offset 1):
// y = T((x * rstd) * (1 + w)).
template <typename T, typename W>
__global__ void __launch_bounds__(kFusedThreads)
    rms_norm_fwd_kernel(const T* __restrict__ x, const W* __restrict__ w, T* __restrict__ y, float* __restrict__ rstd,
                        int64_t rows, int64_t cols, float eps, bool gemma, int32_t* guard) {
  __shared__ float shared[32];
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const T* xr = x + r * cols;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += kFusedThreads) {
      const float v = to_f(xr[c]);
      ss += v * v;
    }
    const float total = block_sum(ss, shared);  // not finite when any value of the row is not
    const float inv = rsqrtf(total / static_cast<float>(cols) + eps);
    if (threadIdx.x == 0) {
      rstd[r] = inv;
      if (guard != nullptr && non_finite(total)) guard[kGuardRmsNormFwd] = 1;
    }
    T* yr = y + r * cols;
    for (int64_t c = threadIdx.x; c < cols; c += kFusedThreads) {
      const float n = to_f(xr[c]) * inv;
      yr[c] = gemma ? from_f<T>(n * (1.0f + to_f(w[c]))) : from_f<T>(to_f(w[c]) * to_f(from_f<T>(n)));
    }
  }
}

// dx = rstd * (dn - n * mean(dn * n)), dn = dy * w' (w' = w or 1 + w). dw: each
// block writes its rows' sum of dy * n' to its own row of `dw_part` [gridDim.x,
// cols], summed afterwards in a fixed order (no atomics: the result is the same
// every run).
template <typename T, typename W>
__global__ void __launch_bounds__(kFusedThreads)
    rms_norm_bwd_kernel(const T* __restrict__ dy, const T* __restrict__ x, const W* __restrict__ w,
                        const float* __restrict__ rstd, T* __restrict__ dx, float* __restrict__ dw_part,
                        int64_t rows, int64_t cols, bool gemma, int32_t* guard) {
  extern __shared__ float acc[];  // cols floats when dw is wanted
  __shared__ float shared[32];
  float* dw = dw_part;
  if (dw != nullptr)
    for (int64_t c = threadIdx.x; c < cols; c += kFusedThreads) acc[c] = 0.0f;
  __syncthreads();
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const T* xr = x + r * cols;
    const T* gr = dy + r * cols;
    const float inv = rstd[r];
    float dot = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += kFusedThreads) {
      const float n = to_f(xr[c]) * inv;
      const float wc = gemma ? 1.0f + to_f(w[c]) : to_f(w[c]);
      dot += to_f(gr[c]) * wc * n;
    }
    const float mean = block_sum(dot, shared) / static_cast<float>(cols);
    if (guard != nullptr && threadIdx.x == 0 && non_finite(mean)) guard[kGuardRmsNormBwd] = 1;
    T* dr = dx + r * cols;
    for (int64_t c = threadIdx.x; c < cols; c += kFusedThreads) {
      const float g = to_f(gr[c]);
      const float n = to_f(xr[c]) * inv;
      const float wc = gemma ? 1.0f + to_f(w[c]) : to_f(w[c]);
      dr[c] = from_f<T>(inv * (g * wc - n * mean));
      if (dw != nullptr) acc[c] += g * (gemma ? n : to_f(from_f<T>(n)));
    }
  }
  if (dw != nullptr) {
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += kFusedThreads) dw[blockIdx.x * cols + c] = acc[c];
  }
}

// --------------------------------------------------------------------- RoPE
// x [B, H, T, D] with any strides but a unit last one, cos / sin [B or 1, T, D]
// (stride 0 over B when broadcast). With the halves x1 = x[:D/2], x2 = x[D/2:]:
//   forward   y1 = x1 c1 - x2 s1        y2 = x2 c2 + x1 s2
//   backward  dx1 = dy1 c1 + dy2 s2     dx2 = dy2 c2 - dy1 s1
// One thread per (row, pair); y may alias nothing it reads.
template <typename T>
__global__ void __launch_bounds__(kFusedThreads)
    rope_kernel(const T* __restrict__ x, T* __restrict__ y, const T* __restrict__ cos_, const T* __restrict__ sin_,
                int64_t B, int64_t H, int64_t Tn, int64_t D, int64_t xb, int64_t xh, int64_t xt, int64_t yb, int64_t yh,
                int64_t yt, int64_t cb, int64_t ct, bool backward) {
  const int64_t half = D / 2;
  const int64_t total = B * H * Tn * half;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(kFusedThreads) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * kFusedThreads) {
    const int64_t d = i % half;
    int64_t rest = i / half;
    const int64_t t = rest % Tn;
    rest /= Tn;
    const int64_t h = rest % H;
    const int64_t b = rest / H;
    const T* xr = x + b * xb + h * xh + t * xt;
    T* yr = y + b * yb + h * yh + t * yt;
    const T* cr = cos_ + b * cb + t * ct;
    const T* sr = sin_ + b * cb + t * ct;
    const float a = to_f(xr[d]), z = to_f(xr[d + half]);
    const float c1 = to_f(cr[d]), c2 = to_f(cr[d + half]), s1 = to_f(sr[d]), s2 = to_f(sr[d + half]);
    if (!backward) {
      yr[d] = from_f<T>(a * c1 - z * s1);
      yr[d + half] = from_f<T>(z * c2 + a * s2);
    } else {
      yr[d] = from_f<T>(a * c1 + z * s2);
      yr[d + half] = from_f<T>(z * c2 - a * s1);
    }
  }
}

// ---------------------------------------------------------------- gated MLP
// act 0: silu(g) = g * sigmoid(g); act 1: gelu, tanh approximation.
__device__ __forceinline__ void act_and_grad(int act, float g, float& a, float& da) {
  if (act == 0) {
    const float s = 1.0f / (1.0f + __expf(-g));
    a = g * s;
    da = s * (1.0f + g * (1.0f - s));
  } else {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;  // sqrt(2 / pi)
    const float inner = k0 * (g + k1 * g * g * g);
    const float th = tanhf(inner);
    a = 0.5f * g * (1.0f + th);
    da = 0.5f * (1.0f + th) + 0.5f * g * (1.0f - th * th) * k0 * (1.0f + 3.0f * k1 * g * g);
  }
}

template <typename T>
__global__ void __launch_bounds__(kFusedThreads)
    gated_act_fwd_kernel(const T* __restrict__ g, const T* __restrict__ u, T* __restrict__ h, int64_t n, int act,
                         int32_t* guard) {
  bool bad = false;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(kFusedThreads) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * kFusedThreads) {
    float a, da;
    act_and_grad(act, to_f(g[i]), a, da);
    const float v = a * to_f(u[i]);
    bad |= non_finite(v);
    h[i] = from_f<T>(v);
  }
  if (bad && guard != nullptr) guard[kGuardGatedActFwd] = 1;
}

template <typename T>
__global__ void __launch_bounds__(kFusedThreads)
    gated_act_bwd_kernel(const T* __restrict__ dh, const T* __restrict__ g, const T* __restrict__ u,
                         T* __restrict__ dg, T* __restrict__ du, int64_t n, int act, int32_t* guard) {
  bool bad = false;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(kFusedThreads) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * kFusedThreads) {
    float a, da;
    act_and_grad(act, to_f(g[i]), a, da);
    const float d = to_f(dh[i]);
    const float vu = d * a, vg = d * to_f(u[i]) * da;
    bad |= non_finite(vu) | non_finite(vg);
    du[i] = from_f<T>(vu);
    dg[i] = from_f<T>(vg);
  }
  if (bad && guard != nullptr) guard[kGuardGatedActBwd] = 1;
}

template <typename F>
cudaError_t by_dtype(DType t, F&& f) {
  switch (t) {
    case DType::kFloat32:
      return f(float{});
    case DType::kFloat16:
      return f(__half{});
    case DType::kBFloat16:
      return f(__nv_bfloat16{});
  }
  return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t rms_norm_forward(DType x_dtype, DType w_dtype, const void* x, const void* w, void* y, float* rstd,
                             int64_t rows, int64_t cols, float eps, bool gemma, int sm_count, cudaStream_t stream,
                             int32_t* guard) {
  if (rows == 0) return cudaSuccess;
  const int grid = static_cast<int>(rows < static_cast<int64_t>(sm_count > 0 ? sm_count : 80) * 16
                                        ? rows : static_cast<int64_t>(sm_count > 0 ? sm_count : 80) * 16);
  return by_dtype(x_dtype, [&](auto xt) {
    using T = decltype(xt);
    return by_dtype(w_dtype, [&](auto wt) {
      using Wt = decltype(wt);
      rms_norm_fwd_kernel<T, Wt><<<grid, kFusedThreads, 0, stream>>>(
          static_cast<const T*>(x), static_cast<const Wt*>(w), static_cast<T*>(y), rstd, rows, cols, eps, gemma,
          guard);
      return cudaGetLastError();
    });
  });
}

int rms_norm_backward_grid(int64_t rows, bool dw, int sm_count) {
  const int64_t cap = static_cast<int64_t>(sm_count > 0 ? sm_count : 80) * (dw ? 2 : 16);
  return static_cast<int>(rows < 1 ? 1 : rows < cap ? rows : cap);
}

cudaError_t rms_norm_backward(DType x_dtype, DType w_dtype, const void* dy, const void* x, const void* w,
                              const float* rstd, void* dx, float* dw, int64_t rows, int64_t cols, bool gemma,
                              int sm_count, cudaStream_t stream, int32_t* guard) {
  if (rows == 0) return cudaSuccess;
  const int grid = rms_norm_backward_grid(rows, dw != nullptr, sm_count);
  const size_t smem = dw != nullptr ? static_cast<size_t>(cols) * sizeof(float) : 0;
  return by_dtype(x_dtype, [&](auto xt) {
    using T = decltype(xt);
    return by_dtype(w_dtype, [&](auto wt) {
      using Wt = decltype(wt);
      if (smem > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(rms_norm_bwd_kernel<T, Wt>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (e != cudaSuccess) return e;
      }
      rms_norm_bwd_kernel<T, Wt><<<grid, kFusedThreads, smem, stream>>>(
          static_cast<const T*>(dy), static_cast<const T*>(x), static_cast<const Wt*>(w), rstd, static_cast<T*>(dx),
          dw, rows, cols, gemma, guard);
      return cudaGetLastError();
    });
  });
}

cudaError_t rope(DType dtype, const void* x, void* y, const void* cos_, const void* sin_, int64_t B, int64_t H,
                 int64_t T_, int64_t D, const int64_t* xs, const int64_t* ys, int64_t cb, int64_t ct, bool backward,
                 int sm_count, cudaStream_t stream) {
  if (D % 2 != 0) return cudaErrorInvalidValue;
  const int64_t total = B * H * T_ * (D / 2);
  if (total == 0) return cudaSuccess;
  return by_dtype(dtype, [&](auto tt) {
    using T = decltype(tt);
    rope_kernel<T><<<fused_grid(total, sm_count), kFusedThreads, 0, stream>>>(
        static_cast<const T*>(x), static_cast<T*>(y), static_cast<const T*>(cos_), static_cast<const T*>(sin_), B, H,
        T_, D, xs[0], xs[1], xs[2], ys[0], ys[1], ys[2], cb, ct, backward);
    return cudaGetLastError();
  });
}

cudaError_t gated_act_forward(DType dtype, const void* g, const void* u, void* h, int64_t n, int act, int sm_count,
                              cudaStream_t stream, int32_t* guard) {
  if (n == 0) return cudaSuccess;
  return by_dtype(dtype, [&](auto tt) {
    using T = decltype(tt);
    gated_act_fwd_kernel<T><<<fused_grid(n, sm_count), kFusedThreads, 0, stream>>>(
        static_cast<const T*>(g), static_cast<const T*>(u), static_cast<T*>(h), n, act, guard);
    return cudaGetLastError();
  });
}

cudaError_t gated_act_backward(DType dtype, const void* dh, const void* g, const void* u, void* dg, void* du, int64_t n,
                               int act, int sm_count, cudaStream_t stream, int32_t* guard) {
  if (n == 0) return cudaSuccess;
  return by_dtype(dtype, [&](auto tt) {
    using T = decltype(tt);
    gated_act_bwd_kernel<T><<<fused_grid(n, sm_count), kFusedThreads, 0, stream>>>(
        static_cast<const T*>(dh), static_cast<const T*>(g), static_cast<const T*>(u), static_cast<T*>(dg),
        static_cast<T*>(du), n, act, guard);
    return cudaGetLastError();
  });
}

}  // namespace kernels
}  // namespace trainstream
