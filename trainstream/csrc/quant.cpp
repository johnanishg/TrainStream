// TrainStream — expanding quantized frozen weights on TrainStream's H2D stream.

#include "kernels.h"
#include "trainstream.h"

namespace trainstream {

namespace {

kernels::DType out_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
      return kernels::DType::kFloat32;
    case at::kHalf:
      return kernels::DType::kFloat16;
    case at::kBFloat16:
      return kernels::DType::kBFloat16;
    default:
      TORCH_CHECK(false, "dequantize: output must be float32, float16 or bfloat16, got ", t);
  }
}

kernels::QKind blockscaled_kind(const std::string& kind) {
  TORCH_CHECK(kind == "nvfp4" || kind == "fp8", "block-scaled kind must be 'nvfp4' or 'fp8', got '", kind, "'");
  return kind == "nvfp4" ? kernels::QKind::kNVFP4 : kernels::QKind::kMXFP8;
}

// rows / cols of the matrix a packed buffer encodes
std::pair<int64_t, int64_t> blockscaled_shape(const at::Tensor& dst) {
  TORCH_CHECK(dst.dim() == 2 && dst.is_contiguous(), "block-scaled tensors must be contiguous 2-D [rows, cols]");
  TORCH_CHECK(dst.size(1) % 32 == 0, "block-scaled formats need the column count to be a multiple of 32, got ",
              dst.size(1));
  return {dst.size(0), dst.size(1)};
}

void check_aligned(const at::Tensor& t, const char* what) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "block-scaled: ", what,
              " must be 16-byte aligned (pass a tensor that starts its own allocation)");
}

void check_packed(kernels::QKind k, int64_t rows, int64_t cols, const at::Tensor& q, const at::Tensor& scales) {
  check_aligned(q, "the packed buffer");
  const int64_t qbytes = k == kernels::QKind::kNVFP4 ? rows * cols / 2 : rows * cols;
  TORCH_CHECK(q.is_contiguous() && scales.is_contiguous() && static_cast<int64_t>(q.nbytes()) >= qbytes,
              "block-scaled: packed buffer too small");
  TORCH_CHECK(static_cast<int64_t>(scales.nbytes()) >= kernels::blockscaled_scale_bytes(k, rows, cols),
              "block-scaled: scale buffer too small");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> quantize_blockscaled(const at::Tensor& x, const std::string& kind,
                                                        const c10::optional<at::Tensor>& global_decode,
                                                        const c10::optional<at::Tensor>& mask) {
  const auto k = blockscaled_kind(kind);
  TORCH_CHECK(x.is_cuda(), "quantize_blockscaled: x must be a CUDA tensor");
  const at::Tensor x2 = x.contiguous();
  const auto [rows, cols] = blockscaled_shape(x2);
  const float* g = nullptr;
  if (k == kernels::QKind::kNVFP4) {
    TORCH_CHECK(global_decode.has_value() && global_decode->is_cuda() && global_decode->scalar_type() == at::kFloat &&
                    global_decode->numel() == 1 && global_decode->get_device() == x.get_device(),
                "quantize_blockscaled: nvfp4 needs a 1-element float32 CUDA global_decode tensor");
    g = global_decode->data_ptr<float>();
  }
  const uint8_t* mk = nullptr;
  if (mask.has_value()) {
    TORCH_CHECK(k == kernels::QKind::kNVFP4, "quantize_blockscaled: a column mask is only supported for nvfp4");
    TORCH_CHECK(mask->is_cuda() && mask->get_device() == x.get_device() && mask->scalar_type() == at::kByte &&
                    mask->is_contiguous() && mask->numel() == cols,
                "quantize_blockscaled: mask must be a contiguous uint8 CUDA tensor with one byte per column");
    mk = mask->data_ptr<uint8_t>();
  }
  c10::cuda::CUDAGuard guard(x.device());
  const auto opts = x.options().dtype(at::kByte);
  at::Tensor q = at::empty({rows, k == kernels::QKind::kNVFP4 ? cols / 2 : cols}, opts);
  at::Tensor scales = at::zeros({kernels::blockscaled_scale_bytes(k, rows, cols)}, opts);
  Runtime& rt = runtime(x.get_device());
  TS_CUDA_CHECK(kernels::quantize_blockscaled(k, out_dtype(x2.scalar_type()), x2.data_ptr(), rows, cols, g,
                                              q.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), rt.sm_count,
                                              compute_stream(x.get_device()), mk));
  return {q, scales};
}

at::Tensor col_absmax(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "col_absmax: x must be a 2-D CUDA tensor");
  const at::Tensor x2 = x.contiguous();
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor amax = at::zeros({x2.size(1)}, x2.options().dtype(at::kFloat));
  TS_CUDA_CHECK(kernels::col_absmax(out_dtype(x2.scalar_type()), x2.data_ptr(), x2.size(0), x2.size(1),
                                    amax.data_ptr<float>(), compute_stream(x.get_device())));
  return amax;
}

at::Tensor dequantize_blockscaled_cols(const at::Tensor& q, const at::Tensor& scales, const at::Tensor& idx,
                                       int64_t rows, int64_t cols, double global_decode, at::ScalarType dtype) {
  TORCH_CHECK(q.is_cuda() && scales.device() == q.device() && idx.device() == q.device(),
              "dequantize_blockscaled_cols: q, scales and idx must be on the same GPU");
  TORCH_CHECK(cols % 32 == 0, "dequantize_blockscaled_cols: cols must be a multiple of 32");
  TORCH_CHECK(idx.scalar_type() == at::kLong && idx.dim() == 1, "dequantize_blockscaled_cols: idx must be 1-D int64");
  check_packed(kernels::QKind::kNVFP4, rows, cols, q, scales);
  const at::Tensor ix = idx.contiguous();
  c10::cuda::CUDAGuard guard(q.device());
  at::Tensor out = at::empty({rows, ix.numel()}, q.options().dtype(dtype));
  Runtime& rt = runtime(q.get_device());
  TS_CUDA_CHECK(kernels::dequantize_nvfp4_cols(out_dtype(dtype), q.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
                                               ix.data_ptr<int64_t>(), ix.numel(), static_cast<float>(global_decode),
                                               out.data_ptr(), rows, cols, rt.sm_count,
                                               compute_stream(q.get_device())));
  return out;
}

void dequantize_blockscaled(const at::Tensor& dst, const at::Tensor& q, const at::Tensor& scales,
                            const std::string& kind, double global_decode) {
  const auto k = blockscaled_kind(kind);
  TORCH_CHECK(dst.is_cuda() && q.device() == dst.device() && scales.device() == dst.device(),
              "dequantize_blockscaled: dst, q and scales must be on the same GPU");
  const auto [rows, cols] = blockscaled_shape(dst);
  check_packed(k, rows, cols, q, scales);
  check_aligned(dst, "dst");
  c10::cuda::CUDAGuard guard(dst.device());
  Runtime& rt = runtime(dst.get_device());
  TS_CUDA_CHECK(kernels::dequantize_blockscaled(k, out_dtype(dst.scalar_type()), q.data_ptr<uint8_t>(),
                                                scales.data_ptr<uint8_t>(), static_cast<float>(global_decode),
                                                dst.data_ptr(), rows, cols, rt.sm_count,
                                                compute_stream(dst.get_device())));
}

std::tuple<EventPtr, std::vector<at::Tensor>> dequantize_blockscaled_async(const at::Tensor& dst, const at::Tensor& q,
                                                                           const at::Tensor& scales,
                                                                           const std::string& kind,
                                                                           double global_decode,
                                                                           const EventPtr& after) {
  const auto k = blockscaled_kind(kind);
  TORCH_CHECK(dst.is_cuda(), "dequantize_blockscaled_async: dst must be a CUDA tensor");
  const auto [rows, cols] = blockscaled_shape(dst);
  check_packed(k, rows, cols, q, scales);
  check_aligned(dst, "dst");
  TORCH_CHECK(q.device() == scales.device(), "dequantize_blockscaled_async: q and scales on different devices");
  const int dev = dst.get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  std::vector<at::Tensor> keep;
  at::Tensor dq = q, ds = scales;
  if (q.device().is_cpu()) {
    const auto opts = at::TensorOptions().device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(dev)));
    dq = at::empty(q.sizes(), opts.dtype(q.scalar_type()));
    ds = at::empty(scales.sizes(), opts.dtype(scales.scalar_type()));
    auto freed = make_event(dev);
    freed->record(compute_stream(dev));
    freed->block(rt.h2d);
    TS_CUDA_CHECK(cudaMemcpyAsync(dq.data_ptr(), q.data_ptr(), q.nbytes(), cudaMemcpyHostToDevice, rt.h2d));
    TS_CUDA_CHECK(cudaMemcpyAsync(ds.data_ptr(), scales.data_ptr(), scales.nbytes(), cudaMemcpyHostToDevice, rt.h2d));
    rt.bytes_h2d += q.nbytes() + scales.nbytes();
    rt.copies_h2d += 2;
    keep.push_back(dq);
    keep.push_back(ds);
  } else {
    TORCH_CHECK(q.get_device() == dev, "dequantize_blockscaled_async: packed weights on another GPU");
  }
  if (after) after->block(rt.h2d);
  TS_CUDA_CHECK(kernels::dequantize_blockscaled(k, out_dtype(dst.scalar_type()),
                                                static_cast<const uint8_t*>(dq.data_ptr()),
                                                static_cast<const uint8_t*>(ds.data_ptr()),
                                                static_cast<float>(global_decode), dst.data_ptr(), rows, cols,
                                                rt.sm_count, rt.h2d));
  auto done = make_event(dev);
  done->record(rt.h2d);
  return {done, keep};
}

std::tuple<EventPtr, std::vector<at::Tensor>> dequantize_async(const at::Tensor& dst, const at::Tensor& q,
                                                               const at::Tensor& absmax, const std::string& kind,
                                                               int64_t block, const EventPtr& after) {
  TORCH_CHECK(kind == "nf4" || kind == "int8", "dequantize: kind must be 'nf4' or 'int8', got '", kind, "'");
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dequantize: dst must be a contiguous CUDA tensor");
  TORCH_CHECK(block > 0 && block % 2 == 0, "dequantize: block size must be a positive even number");
  // absmax: float32 (one per block), or uint8 packing double-quantized scales: one code per
  // block, padded to 4 bytes, then one float32 per 256 blocks (quant.py: pack_double_quant)
  const bool dqs = absmax.scalar_type() == at::kByte;
  TORCH_CHECK(q.is_contiguous() && absmax.is_contiguous() && (absmax.scalar_type() == at::kFloat || dqs),
              "dequantize: q and absmax must be contiguous; absmax float32 (or uint8, double-quantized)");
  TORCH_CHECK(q.device() == absmax.device(), "dequantize: q and absmax must be on the same device");
  const bool nf4 = kind == "nf4";
  const int64_t n = dst.numel();
  TORCH_CHECK(q.numel() >= (nf4 ? (n + 1) / 2 : n), "dequantize: packed buffer too small for ", n, " values");
  const int64_t nblocks = (n + block - 1) / block, coded = (nblocks + 3) / 4 * 4;
  TORCH_CHECK(dqs ? absmax.numel() >= coded + (nblocks + 255) / 256 * 4 : absmax.numel() >= nblocks,
              "dequantize: too few block scales");

  const int dev = dst.get_device();
  Runtime& rt = runtime(dev);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
  std::vector<at::Tensor> keep;
  at::Tensor dq = q, ds = absmax;
  if (q.device().is_cpu()) {
    // Packed bytes live in host RAM: copy them over first, into buffers from
    // the compute stream's pool, ordered after everything queued there.
    const auto opts = at::TensorOptions().device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(dev)));
    dq = at::empty(q.sizes(), opts.dtype(q.scalar_type()));
    ds = at::empty(absmax.sizes(), opts.dtype(absmax.scalar_type()));
    auto freed = make_event(dev);
    freed->record(compute_stream(dev));
    freed->block(rt.h2d);
    TS_CUDA_CHECK(cudaMemcpyAsync(dq.data_ptr(), q.data_ptr(), q.nbytes(), cudaMemcpyHostToDevice, rt.h2d));
    TS_CUDA_CHECK(cudaMemcpyAsync(ds.data_ptr(), absmax.data_ptr(), absmax.nbytes(), cudaMemcpyHostToDevice, rt.h2d));
    rt.bytes_h2d += q.nbytes() + absmax.nbytes();
    rt.copies_h2d += 2;
    keep.push_back(dq);
    keep.push_back(ds);
  } else {
    TORCH_CHECK(q.get_device() == dev, "dequantize: packed weights on another GPU");
  }
  if (after) after->block(rt.h2d);
  const auto qk = nf4 ? kernels::QKind::kNF4 : kernels::QKind::kInt8;
  if (dqs) {
    const uint8_t* base = ds.data_ptr<uint8_t>();
    TS_CUDA_CHECK(kernels::dequantize_dq(qk, out_dtype(dst.scalar_type()), dq.data_ptr(), base,
                                        reinterpret_cast<const float*>(base + coded), dst.data_ptr(), n,
                                        static_cast<int>(block), rt.sm_count, rt.h2d));
  } else {
    TS_CUDA_CHECK(kernels::dequantize(qk, out_dtype(dst.scalar_type()), dq.data_ptr(), ds.data_ptr<float>(),
                                      dst.data_ptr(), n, static_cast<int>(block), rt.sm_count, rt.h2d));
  }
  auto done = make_event(dev);
  done->record(rt.h2d);
  return {done, keep};
}

}  // namespace trainstream
