// TrainStream — the low-precision front of a gated MLP (or one LoRA Linear) in a single kernel.
//
// A SwiGLU MLP with LoRA adapters on an NVFP4 base computes, per layer,
//
//     g = x Wg^T + s (x Ag^T) Bg^T        u = x Wu^T + s (x Au^T) Bu^T        h = silu(g) * u
//
// as two tensor-core matmuls (cuBLAS), two rank-r updates and one elementwise
// pass: five kernels, three of which only read and write [tokens, intermediate]
// tensors that the next one reads again. Here one kernel does all of it. Each
// thread block owns a tile of h; for that tile it
//
//   1. multiplies the packed 4-bit operands with Blackwell's block-scaled FP4
//      tensor-core instruction, both projections into fp32 accumulators held in
//      registers;
//   2. scales them by the two tensor scales and adds the LoRA terms as a few
//      more tensor-core steps (bf16, 16 ranks each) into the same registers;
//   3. applies the activation there and writes h. g and u never exist in
//      memory unless they are asked for.
//
// Measured on an RTX 5090 Laptop GPU at Qwen2.5-14B's MLP shape (hidden 5120,
// intermediate 13824, rank 16, 4,096 tokens): 4.3 ms for h, against 4.1 ms for
// the five separate kernels, because cuBLAS's FP4 matmul alone runs at 228 T
// multiply-adds/s and this one at 137. It is therefore not the default path
// (wrap(fused_gemm=True) selects it); what it saves is the memory, not the time.
//
// Operands reach the registers through shared memory: a three-stage pipeline of
// asynchronous copies (cp.async), each stage holding kSub * 64 values of K for
// the whole tile, or one LoRA step, so the copy of a later stage overlaps the
// tensor-core work on an earlier one. Results leave through shared memory too,
// as 16-byte stores of whole rows of the tile.
//
// The FP4 instruction (PTX `mma.sync ... kind::mxf4nvf4.block_scale`) exists on
// consumer Blackwell (sm_120) when the file is compiled with the architecture's
// full feature set (-gencode arch=compute_120a,code=sm_120a); elsewhere the
// kernel body is empty and gemm_fp4_available() is false. Operand layout, as
// measured on the hardware (thread 4 g + q of a warp, g = 0..7, q = 0..3):
//   A  a0 / a1: rows g / g + 8, values k = 8 q .. 8 q + 7 (low nibble first); a2 / a3: k + 32
//   B  b0: column g, k = 8 q .. 8 q + 7; b1: k + 32
//   D  (g, 2 q), (g, 2 q + 1), (g + 8, 2 q), (g + 8, 2 q + 1)
//   scales (one byte per 16 values of K, four per register): row g's from the
//   thread with q = 0, row g + 8's from q = 1; column g's from q = 0
// which is TrainStream's packing (two E2M1 codes per byte, first in the low
// nibble) and cuBLAS's swizzled scale layout read as they are stored: a
// register is one aligned 32-bit load from the packed row, a scale register
// one from the swizzled scales.

#include "kernels.h"

#include <cuda_bf16.h>

namespace trainstream {
namespace kernels {
namespace {

constexpr int kBM = 128;      // rows of x per thread block (one swizzled scale tile)
constexpr int kBN = 64;       // rows of W (columns of the product) per projection per thread block
constexpr int kSub = 1;       // K steps of 64 per pipeline stage (2 measured slower: 4.7 ms against 4.25 at 4,096 tokens)
constexpr int kStages = 3;    // stages in flight in shared memory
constexpr int kGemmThreads = 256;

constexpr int kADataBytes = kBM * 32;   // 64 values of K per row, 4 bits each
constexpr int kAScaleBytes = 512;       // the tile's scales for one K step, as stored
constexpr int kBDataBytes = kBN * 32;
constexpr int kBScaleBytes = 512;
constexpr int kAStage = kSub * (kADataBytes + kAScaleBytes);
constexpr int kBStage = kSub * (kBDataBytes + kBScaleBytes);  // per projection
constexpr int kTileBytes = kBM * kBN * 2;                     // one output tile, bf16

inline int stage_bytes(int nproj) { return kAStage + nproj * kBStage; }

// TS_FP4_MMA: the build asked for the kernel (a toolkit whose assembler knows the instruction);
// the feature macro: this compilation targets an architecture that has it.
#if defined(TS_FP4_MMA) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define TS_FP4_MMA_DEVICE 1
#endif

#ifdef TS_FP4_MMA_DEVICE
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* global_src) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(global_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_but_one() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

// d += A (16 x 64, e2m1, per-16 scales) B (64 x 8)
__device__ __forceinline__ void mma_fp4(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1, uint32_t sfa, uint32_t sfb) {
  float d0, d1, d2, d3;
  const float c0 = d[0], c1 = d[1], c2 = d[2], c3 = d[3];
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "n"(0),
        "n"(0), "r"(sfb), "n"(0), "n"(0));
  d[0] = d0;
  d[1] = d1;
  d[2] = d2;
  d[3] = d3;
}

// d += A (16 x 16, bf16) B (16 x 8, bf16)
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  float d0, d1, d2, d3;
  const float c0 = d[0], c1 = d[1], c2 = d[2], c3 = d[3];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  d[0] = d0;
  d[1] = d1;
  d[2] = d2;
  d[3] = d3;
}

__device__ __forceinline__ float activation(int act, float g) {
  if (act == 0) return g / (1.0f + __expf(-g));  // silu
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;  // gelu, tanh approximation
  return 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
}

__device__ __forceinline__ bool non_finite2(float a, float b) {
  return ((__float_as_uint(a) & 0x7f800000u) == 0x7f800000u) | ((__float_as_uint(b) & 0x7f800000u) == 0x7f800000u);
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}

#endif  // TS_FP4_MMA_DEVICE

// One thread block: rows [m0, m0 + 128) of x against rows [n0, n0 + 64) of each W.
// Warp w covers rows (w / 2) * 32 .. + 32 and columns (w % 2) * 32 .. + 32 of the tile, as
// 2 x 4 instruction tiles (16 x 8 each) per projection.
template <int NP>
__global__ void __launch_bounds__(kGemmThreads) gemm_fp4_kernel(const FusedGemmArgs a) {
#ifdef TS_FP4_MMA_DEVICE
  extern __shared__ __align__(16) unsigned char smem[];
  const int stage = kAStage + NP * kBStage;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3;
  const int wr = warp >> 1, wc = warp & 1;

  const int64_t MB = (a.M + kBM - 1) / kBM;
  const int64_t m0 = static_cast<int64_t>(blockIdx.x % MB) * kBM;  // consecutive blocks share the W tile
  const int64_t n0 = static_cast<int64_t>(blockIdx.x / MB) * kBN;
  const int64_t row_bytes = a.K / 2;

  // this thread's rows and columns inside the tile
  int rows[2], cols[4];
#pragma unroll
  for (int rb = 0; rb < 2; ++rb) rows[rb] = wr * 32 + rb * 16 + g;
#pragma unroll
  for (int cb = 0; cb < 4; ++cb) cols[cb] = wc * 32 + cb * 8 + g;
  // where this thread's values of an output tile sit in shared memory (two bf16 per 32-bit word)
  auto tile_word = [&](int rb, int cb, int half) { return (rows[rb] + 8 * half) * 32 + wc * 16 + cb * 4 + q; };

  float acc[NP][2][4][4];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int cb = 0; cb < 4; ++cb)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[p][rb][cb][i] = 0.0f;

  // ---- the pipeline: FP4 stages (kSub K steps each), then one stage per LoRA step
  const int64_t KS = a.K / 64;
  const int64_t BS = (KS + kSub - 1) / kSub;
  const int64_t RC = a.R / 16;
  int lora_p[2] = {0, 0};
  int nl = 0;
#pragma unroll
  for (int p = 0; p < NP; ++p)
    if (RC > 0 && a.xa[p] != nullptr) lora_p[nl++] = p;
  const int64_t T = BS + nl * RC;

  auto load = [&](int64_t t) {
    unsigned char* st = smem + static_cast<int>(t % kStages) * stage;
    if (t < BS) {
#pragma unroll
      for (int sub = 0; sub < kSub; ++sub) {
        const int64_t ks = t * kSub + sub;
        if (ks >= KS) break;
        const int64_t koff = ks * 32;
        {  // x: 128 rows x 32 bytes, one 16-byte piece per thread
          int64_t row = m0 + (tid >> 1);
          if (row >= a.M) row = a.M - 1;  // rows past the end are computed from the last one and never written
          cp_async16(st + sub * kADataBytes + (tid >> 1) * 32 + (tid & 1) * 16,
                     a.xq + row * row_bytes + koff + (tid & 1) * 16);
        }
        if (tid < NP * 128) {  // W: 64 rows x 32 bytes per projection
          const int p = tid >> 7, c = tid & 127;
          int64_t n = n0 + (c >> 1);
          if (n >= a.N) n = a.N - 1;
          cp_async16(st + kAStage + p * kBStage + sub * kBDataBytes + (c >> 1) * 32 + (c & 1) * 16,
                     a.wq[p] + n * row_bytes + koff + (c & 1) * 16);
        }
        if (tid < 32) {  // x's scales for this tile and K step: 512 bytes, as stored
          cp_async16(st + kSub * kADataBytes + sub * kAScaleBytes + tid * 16,
                     a.xs + ((m0 >> 7) * (a.K / 64) + ks) * 512 + tid * 16);
        } else if (tid < 32 + NP * 32) {  // W's scales: the 128-row tile these 64 rows are in
          const int p = (tid - 32) >> 5, c = (tid - 32) & 31;
          cp_async16(st + kAStage + p * kBStage + kSub * kBDataBytes + sub * kBScaleBytes + c * 16,
                     a.ws[p] + ((n0 >> 7) * (a.K / 64) + ks) * 512 + c * 16);
        }
      }
    } else {  // a LoRA step: 16 ranks of xa [M, R] and of sb [N, R], bf16
      const int64_t i = t - BS;
      const int p = lora_p[i / RC];
      const int64_t rc = i % RC;
      int64_t row = m0 + (tid >> 1);
      if (row >= a.M) row = a.M - 1;
      cp_async16(st + (tid >> 1) * 32 + (tid & 1) * 16,
                 static_cast<const unsigned char*>(a.xa[p]) + (row * a.R + rc * 16) * 2 + (tid & 1) * 16);
      if (tid < 128) {
        int64_t n = n0 + (tid >> 1);
        if (n >= a.N) n = a.N - 1;
        cp_async16(st + kAStage + (tid >> 1) * 32 + (tid & 1) * 16,
                   static_cast<const unsigned char*>(a.sb[p]) + (n * a.R + rc * 16) * 2 + (tid & 1) * 16);
      }
    }
  };

  // scale registers: index of a local row in a swizzled tile block
  const int rs0 = rows[0] + 8 * (q & 1), rs1 = rows[1] + 8 * (q & 1);
  const int sa_idx[2] = {(rs0 & 31) * 4 + (rs0 >> 5), (rs1 & 31) * 4 + (rs1 >> 5)};
  int sb_idx[4];
#pragma unroll
  for (int cb = 0; cb < 4; ++cb) {
    const int rr = static_cast<int>((n0 + cols[cb]) & 127);
    sb_idx[cb] = (rr & 31) * 4 + (rr >> 5);
  }
  bool scaled = false;
  auto scale = [&]() {
    const float xg = *a.xg;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      const float alpha = xg * a.wg[p];
#pragma unroll
      for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int cb = 0; cb < 4; ++cb)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[p][rb][cb][i] *= alpha;
    }
    scaled = true;
  };

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < T) load(s);
    cp_async_commit();
  }
  for (int64_t t = 0; t < T; ++t) {
    cp_async_wait_but_one();
    __syncthreads();  // stage t is in shared memory, and nobody still reads the slot loaded next
    if (t + kStages - 1 < T) load(t + kStages - 1);
    cp_async_commit();
    const unsigned char* st = smem + static_cast<int>(t % kStages) * stage;
    if (t < BS) {
#pragma unroll
      for (int sub = 0; sub < kSub; ++sub) {
        if (t * kSub + sub >= KS) break;
        const uint32_t* sa = reinterpret_cast<const uint32_t*>(st + sub * kADataBytes);
        const uint32_t* ssa = reinterpret_cast<const uint32_t*>(st + kSub * kADataBytes + sub * kAScaleBytes);
        uint32_t ar[2][4], sfa[2];
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          ar[rb][0] = sa[rows[rb] * 8 + q];
          ar[rb][1] = sa[(rows[rb] + 8) * 8 + q];
          ar[rb][2] = sa[rows[rb] * 8 + 4 + q];
          ar[rb][3] = sa[(rows[rb] + 8) * 8 + 4 + q];
          sfa[rb] = ssa[sa_idx[rb]];
        }
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          const unsigned char* bp = st + kAStage + p * kBStage;
          const uint32_t* sb = reinterpret_cast<const uint32_t*>(bp + sub * kBDataBytes);
          const uint32_t* ssb = reinterpret_cast<const uint32_t*>(bp + kSub * kBDataBytes + sub * kBScaleBytes);
#pragma unroll
          for (int cb = 0; cb < 4; ++cb) {
            const uint32_t b0 = sb[cols[cb] * 8 + q], b1 = sb[cols[cb] * 8 + 4 + q];
            const uint32_t sfb = ssb[sb_idx[cb]];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
              mma_fp4(acc[p][rb][cb], ar[rb][0], ar[rb][1], ar[rb][2], ar[rb][3], b0, b1, sfa[rb], sfb);
          }
        }
      }
    } else {
      if (!scaled) scale();  // the FP4 sums become values before the LoRA terms join them
      const int p = lora_p[(t - BS) / RC];
      const uint32_t* sa = reinterpret_cast<const uint32_t*>(st);
      const uint32_t* sb = reinterpret_cast<const uint32_t*>(st + kAStage);
      uint32_t ar[2][4];
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        ar[rb][0] = sa[rows[rb] * 8 + q];
        ar[rb][1] = sa[(rows[rb] + 8) * 8 + q];
        ar[rb][2] = sa[rows[rb] * 8 + 4 + q];
        ar[rb][3] = sa[(rows[rb] + 8) * 8 + 4 + q];
      }
#pragma unroll
      for (int cb = 0; cb < 4; ++cb) {
        const uint32_t b0 = sb[cols[cb] * 8 + q], b1 = sb[cols[cb] * 8 + 4 + q];
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
          for (int pp = 0; pp < NP; ++pp)
            if (pp == p) mma_bf16(acc[pp][rb][cb], ar[rb][0], ar[rb][1], ar[rb][2], ar[rb][3], b0, b1);
        }
      }
    }
  }
  if (!scaled) scale();
  __syncthreads();  // every warp is done with the stages: shared memory now carries the output tiles

  // ---- outputs: each thread puts its values into the tile, then the tile leaves as whole rows
  bool bad = false;
  uint32_t* t32 = reinterpret_cast<uint32_t*>(smem);
  auto flush = [&](void* dst) {
    __syncthreads();
    const uint4* tile = reinterpret_cast<const uint4*>(smem);
#pragma unroll
    for (int j = 0; j < kTileBytes / 16 / kGemmThreads; ++j) {
      const int i = tid + kGemmThreads * j;
      const int64_t m = m0 + (i >> 3), n = n0 + (i & 7) * 8;
      if (m < a.M && n < a.N)
        *reinterpret_cast<uint4*>(static_cast<unsigned char*>(dst) + (m * a.N + n) * 2) = tile[i];
    }
    __syncthreads();
  };
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    const bool write = a.out[p] != nullptr;
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int cb = 0; cb < 4; ++cb)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float v0 = acc[p][rb][cb][2 * half], v1 = acc[p][rb][cb][2 * half + 1];
          bad |= non_finite2(v0, v1);
          if (write) t32[tile_word(rb, cb, half)] = pack_bf16(v0, v1);
        }
    if (write) flush(a.out[p]);
  }
  if (NP == 2 && a.h != nullptr) {
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int cb = 0; cb < 4; ++cb)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float h0 = activation(a.act, acc[0][rb][cb][2 * half]) * acc[NP - 1][rb][cb][2 * half];
          const float h1 = activation(a.act, acc[0][rb][cb][2 * half + 1]) * acc[NP - 1][rb][cb][2 * half + 1];
          bad |= non_finite2(h0, h1);
          t32[tile_word(rb, cb, half)] = pack_bf16(h0, h1);
        }
    flush(a.h);
  }
  if (bad && a.guard != nullptr) a.guard[kGuardGemm] = 1;
#endif  // TS_FP4_MMA_DEVICE
}

template <int NP>
cudaError_t launch_gemm_fp4(const FusedGemmArgs& a, int blocks, cudaStream_t stream) {
  const int smem = kStages * stage_bytes(NP);  // (always at least one output tile)
  if (smem > 48 * 1024) {
    const cudaError_t e =
        cudaFuncSetAttribute(gemm_fp4_kernel<NP>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
  }
  gemm_fp4_kernel<NP><<<blocks, kGemmThreads, smem, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace

bool gemm_fp4_available() {
#ifdef TS_FP4_MMA
  return true;
#else
  return false;
#endif
}

cudaError_t gemm_fp4_fused(const FusedGemmArgs& a, cudaStream_t stream) {
  if (a.M <= 0 || a.N <= 0) return cudaSuccess;
  if ((a.nproj != 1 && a.nproj != 2) || a.K % 64 != 0 || a.N % 16 != 0 || a.R % 16 != 0) return cudaErrorInvalidValue;
  if (!gemm_fp4_available()) return cudaErrorNotSupported;
  const int64_t blocks = ((a.M + kBM - 1) / kBM) * ((a.N + kBN - 1) / kBN);
  if (blocks > 0x7fffffff) return cudaErrorInvalidValue;
  return a.nproj == 2 ? launch_gemm_fp4<2>(a, static_cast<int>(blocks), stream)
                      : launch_gemm_fp4<1>(a, static_cast<int>(blocks), stream);
}

}  // namespace kernels
}  // namespace trainstream
