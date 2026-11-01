"""Outlier-aware NVFP4: the largest input channels multiplied in bf16."""

import pytest
import torch
from torch import nn

import trainstream
from trainstream import _C, lowp
from trainstream import quant as Q

from test_quant import HP, make, trainable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"


def needs_nvfp4():
    if not lowp.native_supported("nvfp4", DEV):
        pytest.skip("this GPU has no nvfp4 tensor-core matmul")


def activations(m, k, outliers=(), scale=100.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, k, device=DEV, generator=g)
    for c in outliers:
        x[:, c] *= scale
    return x.to(torch.bfloat16)


def frozen_linear(k, n, seed=1):
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=True, device=DEV, dtype=torch.bfloat16)
    lin.weight.requires_grad_(False)
    w = lin.weight.detach().clone()
    Q.attach(lin.weight, Q.make_quantized(lin.weight, "nvfp4", 16, pin=False))
    return lin, w


# --------------------------------------------------------------------------- kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 64), (37, 512), (300, 1024)])
def test_column_absmax_matches_torch(dtype, shape):
    x = activations(*shape, outliers=(3,)).to(dtype)
    assert torch.equal(_C.col_absmax(x), x.float().abs().amax(0))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_masked_quantize_equals_quantizing_with_those_columns_zeroed(dtype):
    x = activations(96, 512, outliers=(5, 200, 511)).to(dtype)
    idx = torch.tensor([5, 200, 511], device=DEV)
    mask = torch.zeros(512, dtype=torch.uint8, device=DEV).index_fill_(0, idx, 1)
    zeroed = x.clone()
    zeroed[:, idx] = 0
    g = Q.global_decode_scale(zeroed)
    q, s = _C.quantize_blockscaled(x, "nvfp4", g, mask)
    rq, rs, _ = Q.quantize_blockscaled_reference(zeroed, "nvfp4", g)
    assert torch.equal(q, rq) and torch.equal(s, rs)


def test_column_dequant_gathers_the_full_expansion():
    w = torch.randn(256, 1024, device=DEV, dtype=torch.bfloat16)
    q, s, g = Q.quantize_blockscaled(w, "nvfp4")
    full = torch.empty(256, 1024, device=DEV, dtype=torch.float32)
    _C.dequantize_blockscaled(full, q, s, "nvfp4", float(g))
    idx = torch.tensor([0, 1, 17, 500, 1023], device=DEV)
    cols = _C.dequantize_blockscaled_cols(q, s, idx, 256, 1024, float(g), torch.float32)
    assert torch.equal(cols, full[:, idx])


def test_outlier_count():
    assert lowp.outlier_count(0.0, 4096) == 0
    assert lowp.outlier_count(0.01, 4096) == 41
    assert lowp.outlier_count(0.01, 64) == 1  # at least one
    assert lowp.outlier_count(0.1, 8) == 1


# --------------------------------------------------------------------------- the split matmul
def emulated(x, lin, w_deq, k):
    """Plain PyTorch: nvfp4 of x without its k largest channels @ W, plus those channels in full."""
    x2 = x.float()
    amax = x2.abs().amax(0)
    vals, idx = torch.topk(amax, k + 1)
    idx = idx[:k]
    g = (vals[k:].clamp_min(1e-12) / (6.0 * 448.0))
    rest = x2.clone()
    rest[:, idx] = 0
    _, _, xq = Q.quantize_blockscaled_reference(rest, "nvfp4", g)
    return xq @ w_deq.t() + x2[:, idx] @ w_deq[:, idx].t() + lin.bias.float()


def test_split_forward_matches_the_emulation():
    needs_nvfp4()
    lin, _ = frozen_linear(512, 384)
    lowp.install([lin], DEV, outliers=0.02)
    x = activations(64, 512, outliers=(7, 300))
    w_deq = Q.quantized_of(lin.weight).dequantize(DEV).float()
    k = lowp.outlier_count(0.02, 512)
    with torch.no_grad():
        y = lin(x).float()
    ref = emulated(x, lin, w_deq, k)
    assert ((y - ref).norm() / ref.norm()).item() < 1e-2
    lowp.uninstall([lin])


def test_outliers_cut_the_error_on_activations_with_outlier_channels():
    needs_nvfp4()
    lin, _ = frozen_linear(1024, 512)
    w_deq = Q.quantized_of(lin.weight).dequantize(DEV).float()
    x = activations(128, 1024, outliers=(11, 400, 777, 900))
    exact = x.float() @ w_deq.t() + lin.bias.float()  # the stored weight, unquantized activations
    errs = {}
    for frac in (0.0, 0.01):
        lowp.install([lin], DEV, outliers=frac)
        with torch.no_grad():
            y = lin(x).float()
        errs[frac] = ((y - exact).norm() / exact.norm()).item()
        lowp.uninstall([lin])
    assert errs[0.01] < 0.5 * errs[0.0], errs


def test_backward_is_the_same_with_and_without_outliers():
    needs_nvfp4()
    lin, _ = frozen_linear(512, 256)
    r = torch.randn(32, 256, device=DEV)  # a linear loss: grad_y = r whatever the forward gave
    grads = []
    for frac in (0.0, 0.02):
        lowp.install([lin], DEV, outliers=frac)
        x = activations(32, 512, outliers=(9,)).requires_grad_(True)
        (lin(x).float() * r).sum().backward()
        grads.append(x.grad.clone())
        lowp.uninstall([lin])
    assert torch.equal(grads[0], grads[1])  # grad_x = grad_y @ dequant(W) either way


# --------------------------------------------------------------------------- wrap()
def test_wrap_option_values_and_reporting():
    for bad in ("many", 0.5, -0.01):
        with pytest.raises(ValueError):
            trainstream.wrap(make(), outliers=bad)
    m = make()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize="nf4",
                              outliers="auto", blocks=list(m.blocks), profile=False)
    s = trainstream.wrapper.unwrap_session(m)
    assert s.config["outliers"] == 0.0  # only the nvfp4 tensor-core path uses it
    trainstream.unwrap(m)
    m = make()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize="nvfp4",
                              outliers="auto", blocks=list(m.blocks), profile=False)
    s = trainstream.wrapper.unwrap_session(m)
    assert s.config["outliers"] == 0.01
    if s.lowp_modules:
        assert all(Q.quantized_of(mod.weight).outliers == 0.01 for mod in s.lowp_modules)
        assert any("input channels, multiplied in bf16" in e for e in s.events)
        assert s.stats()["params"]["outliers"] == 0.01
    trainstream.unwrap(m)
