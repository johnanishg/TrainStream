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

}  // namespace

std::tuple<EventPtr, std::vector<at::Tensor>> dequantize_async(const at::Tensor& dst, const at::Tensor& q,
                                                               const at::Tensor& absmax, const std::string& kind,
                                                               int64_t block, const EventPtr& after) {
  TORCH_CHECK(kind == "nf4" || kind == "int8", "dequantize: kind must be 'nf4' or 'int8', got '", kind, "'");
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dequantize: dst must be a contiguous CUDA tensor");
  TORCH_CHECK(block > 0 && block % 2 == 0, "dequantize: block size must be a positive even number");
  TORCH_CHECK(q.is_contiguous() && absmax.is_contiguous() && absmax.scalar_type() == at::kFloat,
              "dequantize: q and absmax must be contiguous; absmax float32");
  TORCH_CHECK(q.device() == absmax.device(), "dequantize: q and absmax must be on the same device");
  const bool nf4 = kind == "nf4";
  const int64_t n = dst.numel();
  TORCH_CHECK(q.numel() >= (nf4 ? (n + 1) / 2 : n), "dequantize: packed buffer too small for ", n, " values");
  TORCH_CHECK(absmax.numel() >= (n + block - 1) / block, "dequantize: too few block scales");

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
    ds = at::empty(absmax.sizes(), opts.dtype(at::kFloat));
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
  TS_CUDA_CHECK(kernels::dequantize(nf4 ? kernels::QKind::kNF4 : kernels::QKind::kInt8, out_dtype(dst.scalar_type()),
                                    dq.data_ptr(), ds.data_ptr<float>(), dst.data_ptr(), n, static_cast<int>(block),
                                    rt.sm_count, rt.h2d));
  auto done = make_event(dev);
  done->record(rt.h2d);
  return {done, keep};
}

}  // namespace trainstream
