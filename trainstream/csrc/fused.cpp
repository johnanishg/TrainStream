// TrainStream — fused transformer-block ops (RMSNorm, rotary embedding, gated
// MLP activation) for trainstream/fused.py, on PyTorch's current stream.

#include "kernels.h"
#include "trainstream.h"

namespace trainstream {

namespace {

kernels::DType dtype_of(const at::Tensor& t, const char* what) {
  switch (t.scalar_type()) {
    case at::kFloat:
      return kernels::DType::kFloat32;
    case at::kHalf:
      return kernels::DType::kFloat16;
    case at::kBFloat16:
      return kernels::DType::kBFloat16;
    default:
      TORCH_CHECK(false, what, " must be float32, float16 or bfloat16, got ", t.scalar_type());
  }
}

cudaStream_t current(const at::Tensor& t) { return at::cuda::getCurrentCUDAStream(t.get_device()).stream(); }

int sms(const at::Tensor& t) { return runtime(t.get_device()).sm_count; }

int32_t* flags(const at::Tensor& t) { return guard_flags(static_cast<int>(t.get_device())); }

}  // namespace

std::tuple<at::Tensor, at::Tensor> rms_norm_forward(const at::Tensor& x, const at::Tensor& w, double eps, bool gemma) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && w.dim() == 1 && x.size(-1) == w.size(0),
              "rms_norm: x [..., C] and w [C] must be CUDA tensors");
  c10::cuda::CUDAGuard guard(x.device());
  const at::Tensor x2 = x.contiguous(), w2 = w.contiguous();
  const int64_t cols = x2.size(-1), rows = x2.numel() / cols;
  at::Tensor y = at::empty_like(x2);
  at::Tensor rstd = at::empty({rows}, x2.options().dtype(at::kFloat));
  TS_CUDA_CHECK(kernels::rms_norm_forward(dtype_of(x2, "rms_norm: x"), dtype_of(w2, "rms_norm: w"), x2.data_ptr(),
                                          w2.data_ptr(), y.data_ptr(), rstd.data_ptr<float>(), rows, cols,
                                          static_cast<float>(eps), gemma, sms(x2), current(x2), flags(x2)));
  return {y, rstd};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> rms_norm_backward(const at::Tensor& dy, const at::Tensor& x,
                                                                    const at::Tensor& w, const at::Tensor& rstd,
                                                                    bool gemma, bool need_dw) {
  TORCH_CHECK(dy.is_cuda() && dy.sizes() == x.sizes() && dy.scalar_type() == x.scalar_type(),
              "rms_norm backward: dy must match x");
  c10::cuda::CUDAGuard guard(x.device());
  const at::Tensor g2 = dy.contiguous(), x2 = x.contiguous(), w2 = w.contiguous();
  const int64_t cols = x2.size(-1), rows = x2.numel() / cols;
  at::Tensor dx = at::empty_like(x2);
  at::Tensor part;
  float* dwp = nullptr;
  if (need_dw) {
    part = at::empty({kernels::rms_norm_backward_grid(rows, true, sms(x2)), cols}, x2.options().dtype(at::kFloat));
    dwp = part.data_ptr<float>();
  }
  TS_CUDA_CHECK(kernels::rms_norm_backward(dtype_of(x2, "rms_norm: x"), dtype_of(w2, "rms_norm: w"), g2.data_ptr(),
                                           x2.data_ptr(), w2.data_ptr(), rstd.data_ptr<float>(), dx.data_ptr(), dwp,
                                           rows, cols, gemma, sms(x2), current(x2), flags(x2)));
  c10::optional<at::Tensor> dw;
  if (need_dw) dw = part.sum(0);  // a fixed reduction order: the same result every run
  return {dx, dw};
}

at::Tensor rope(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin, bool backward) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.stride(3) == 1, "rope: x must be a CUDA [B, H, T, D] tensor with a "
              "unit last stride");
  TORCH_CHECK(cos.sizes() == sin.sizes() && cos.dim() == 3 && cos.size(-1) == x.size(3) && cos.size(1) == x.size(2) &&
                  (cos.size(0) == x.size(0) || cos.size(0) == 1) && cos.scalar_type() == x.scalar_type() &&
                  sin.scalar_type() == x.scalar_type(),
              "rope: cos / sin must be [B or 1, T, D] in x's dtype");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor c = cos, s = sin;
  if (c.stride(2) != 1 || s.strides() != c.strides()) {  // the kernel indexes both with one set of strides
    c = cos.contiguous();
    s = sin.contiguous();
  }
  at::Tensor y = at::empty_like(x);  // keeps x's (transposed) layout
  TORCH_CHECK(y.stride(3) == 1, "rope: unexpected output layout");
  const int64_t xs[3] = {x.stride(0), x.stride(1), x.stride(2)};
  const int64_t ys[3] = {y.stride(0), y.stride(1), y.stride(2)};
  const int64_t cb = c.size(0) == 1 ? 0 : c.stride(0);
  TS_CUDA_CHECK(kernels::rope(dtype_of(x, "rope: x"), x.data_ptr(), y.data_ptr(), c.data_ptr(), s.data_ptr(), x.size(0),
                              x.size(1), x.size(2), x.size(3), xs, ys, cb, c.stride(1), backward, sms(x), current(x)));
  return y;
}

at::Tensor gated_act_forward(const at::Tensor& g, const at::Tensor& u, int64_t act) {
  TORCH_CHECK(g.is_cuda() && g.sizes() == u.sizes() && g.scalar_type() == u.scalar_type(),
              "gated_act: g and u must be CUDA tensors of one shape and dtype");
  c10::cuda::CUDAGuard guard(g.device());
  const at::Tensor g2 = g.contiguous(), u2 = u.contiguous();
  at::Tensor h = at::empty_like(g2);
  TS_CUDA_CHECK(kernels::gated_act_forward(dtype_of(g2, "gated_act"), g2.data_ptr(), u2.data_ptr(), h.data_ptr(),
                                           g2.numel(), static_cast<int>(act), sms(g2), current(g2), flags(g2)));
  return h;
}

std::tuple<at::Tensor, at::Tensor> gated_act_backward(const at::Tensor& dh, const at::Tensor& g, const at::Tensor& u,
                                                      int64_t act) {
  TORCH_CHECK(dh.sizes() == g.sizes() && dh.scalar_type() == g.scalar_type(), "gated_act backward: dh must match g");
  c10::cuda::CUDAGuard guard(g.device());
  const at::Tensor d2 = dh.contiguous(), g2 = g.contiguous(), u2 = u.contiguous();
  at::Tensor dg = at::empty_like(g2), du = at::empty_like(u2);
  TS_CUDA_CHECK(kernels::gated_act_backward(dtype_of(g2, "gated_act"), d2.data_ptr(), g2.data_ptr(), u2.data_ptr(),
                                            dg.data_ptr(), du.data_ptr(), g2.numel(), static_cast<int>(act), sms(g2),
                                            current(g2), flags(g2)));
  return {dg, du};
}

bool gemm_fp4_supported(int device) {
  if (!kernels::gemm_fp4_available()) return false;
  const int dev = resolve_device(device);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  return prop.major == 12;  // the architecture the kernel's tensor-core instruction is built for
}

std::vector<at::Tensor> gemm_fp4_front(const at::Tensor& xq, const at::Tensor& xs, const at::Tensor& xg,
                                       const std::vector<at::Tensor>& wq, const std::vector<at::Tensor>& ws,
                                       const std::vector<double>& wg, int64_t M, int64_t N, int64_t K,
                                       const std::vector<at::Tensor>& xa, const std::vector<at::Tensor>& sb,
                                       int64_t act, bool want_h, bool want_out) {
  const size_t np = wq.size();
  TORCH_CHECK((np == 1 || np == 2) && ws.size() == np && wg.size() == np && xa.size() == np && sb.size() == np,
              "gemm_fp4_front: one or two projections, with matching lists");
  TORCH_CHECK(xq.is_cuda(), "gemm_fp4_front: operands must be CUDA tensors");
  const int dev = static_cast<int>(xq.get_device());
  TORCH_CHECK(gemm_fp4_supported(dev), "gemm_fp4_front: this build or GPU has no FP4 tensor-core kernel");
  TORCH_CHECK(M > 0 && N > 0 && K > 0 && K % 64 == 0 && N % 16 == 0,
              "gemm_fp4_front: needs K % 64 == 0 and N % 16 == 0, got K=", K, " N=", N);
  TORCH_CHECK(!want_h || np == 2, "gemm_fp4_front: h needs two projections");
  c10::cuda::CUDAGuard device_guard(xq.device());
  const auto scale_bytes = [K](int64_t rows) {
    return kernels::blockscaled_scale_bytes(kernels::QKind::kNVFP4, rows, K);
  };
  const auto packed = [dev](const at::Tensor& t, int64_t bytes, const char* what) {
    TORCH_CHECK(t.is_cuda() && t.get_device() == dev && t.scalar_type() == at::kByte && t.is_contiguous() &&
                    static_cast<int64_t>(t.nbytes()) == bytes &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "gemm_fp4_front: ", what, " must be ", bytes, " contiguous uint8 bytes on the GPU, 16-byte aligned");
  };
  packed(xq, M * K / 2, "x's codes");
  packed(xs, scale_bytes(M), "x's scales");
  TORCH_CHECK(xg.is_cuda() && xg.get_device() == dev && xg.scalar_type() == at::kFloat && xg.numel() == 1,
              "gemm_fp4_front: x's tensor scale must be a 1-element float32 CUDA tensor");
  kernels::FusedGemmArgs a;
  a.xq = xq.data_ptr<uint8_t>();
  a.xs = xs.data_ptr<uint8_t>();
  a.xg = xg.data_ptr<float>();
  a.M = M;
  a.N = N;
  a.K = K;
  a.nproj = static_cast<int>(np);
  a.act = static_cast<int>(act);
  a.guard = guard_flags(dev);
  std::vector<at::Tensor> keep;  // contiguous bf16 copies, alive until the launch
  for (size_t p = 0; p < np; ++p) {
    packed(wq[p], N * K / 2, "a weight's codes");
    packed(ws[p], scale_bytes(N), "a weight's scales");
    a.wq[p] = wq[p].data_ptr<uint8_t>();
    a.ws[p] = ws[p].data_ptr<uint8_t>();
    a.wg[p] = static_cast<float>(wg[p]);
    if (xa[p].numel() == 0) continue;
    const at::Tensor& l = xa[p];
    const at::Tensor& r = sb[p];
    TORCH_CHECK(l.dim() == 2 && r.dim() == 2 && l.size(0) == M && r.size(0) == N && l.size(1) == r.size(1) &&
                    l.size(1) % 16 == 0 && l.scalar_type() == at::kBFloat16 && r.scalar_type() == at::kBFloat16 &&
                    l.is_cuda() && r.is_cuda() && l.is_contiguous() && r.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(l.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(r.data_ptr()) % 16 == 0,
                "gemm_fp4_front: a LoRA term needs contiguous bf16 [M, R] and [N, R] with R a multiple of 16");
    TORCH_CHECK(a.R == 0 || a.R == l.size(1), "gemm_fp4_front: both LoRA terms must have the same (padded) rank");
    a.R = l.size(1);
    a.xa[p] = l.data_ptr();
    a.sb[p] = r.data_ptr();
  }
  const auto opts = xq.options().dtype(at::kBFloat16);
  std::vector<at::Tensor> out;
  if (want_h) {
    out.push_back(at::empty({M, N}, opts));
    a.h = out.back().data_ptr();
  }
  if (want_out) {
    for (size_t p = 0; p < np; ++p) {
      out.push_back(at::empty({M, N}, opts));
      a.out[p] = out.back().data_ptr();
    }
  }
  TORCH_CHECK(!out.empty(), "gemm_fp4_front: nothing asked for");
  TS_CUDA_CHECK(kernels::gemm_fp4_fused(a, at::cuda::getCurrentCUDAStream(dev).stream()));
  return out;
}

}  // namespace trainstream
