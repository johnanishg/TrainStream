"""Block-scaled frozen weights (NVFP4, MXFP8) and tensor-core matmuls."""

import copy

import pytest
import torch
from torch import nn

import trainstream
from trainstream import _C, lowp
from trainstream import quant as Q

from test_quant import HP, make, tokens, train, trainable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
KINDS = Q.BLOCKSCALED_KINDS


def needs_native(kind):
    if not lowp.native_supported(kind, DEV):
        pytest.skip(f"this GPU has no {kind} tensor-core matmul")


def ref_quant(x, kind):
    g = Q.global_decode_scale(x) if kind == "nvfp4" else None
    return Q.quantize_blockscaled_reference(x, kind, g), g


# --------------------------------------------------------------------------- kernels
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(7, 64), (200, 512), (513, 1024)])
def test_blockscaled_kernels_match_reference(kind, dtype, shape):
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=DEV) * torch.logspace(-3, 2, shape[1], device=DEV)).to(dtype)
    (rq, rs, rdeq), g = ref_quant(x, kind)
    q, s = _C.quantize_blockscaled(x, kind, g)
    assert torch.equal(q, rq) and torch.equal(s, rs)
    out = torch.empty(*shape, device=DEV)
    _C.dequantize_blockscaled(out, q, s, kind, float(g) if g is not None else 1.0)
    assert torch.equal(out, rdeq)


@pytest.mark.parametrize("kind", KINDS)
def test_async_expansion_from_host_ram(kind):
    torch.manual_seed(0)
    w = torch.randn(384, 512, device=DEV, dtype=torch.bfloat16) * 0.02
    q, s, g = Q.quantize_blockscaled(w, kind)
    want = torch.empty_like(w)
    _C.dequantize_blockscaled(want, q, s, kind, float(g))
    out = torch.empty_like(w)
    ev, keep = _C.dequantize_blockscaled_async(out, q.cpu().pin_memory(), s.cpu().pin_memory(), kind, float(g),
                                               _C.record_event(0))
    ev.wait()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    err = ((want.float() - w.float()).abs().mean() / w.float().abs().mean()).item()
    assert err < (0.15 if kind == "nvfp4" else 0.04)


# --------------------------------------------------------------------------- the tensor-core layer
def emulated_linear(x, qw, bias):
    """Plain PyTorch: quantize-dequantize the activations, multiply with the
    dequantized weight; straight-through gradient for the activations."""
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    (_, _, xdeq), _ = ref_quant(x2.detach(), qw.kind)
    xs = x2 + (xdeq.to(x2.dtype) - x2).detach()
    w = qw.dequantize(DEV).float()
    y = xs.float() @ w.t()
    if bias is not None:
        y = y + bias.float()
    return y.view(*x.shape[:-1], -1)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quant_linear_matches_emulation(kind, dtype):
    needs_native(kind)
    torch.manual_seed(0)
    lin = nn.Linear(512, 384).to(DEV, dtype)
    lin.weight.requires_grad_(False)
    qw = Q.make_quantized(lin.weight, kind, 64, pin=False)
    qw.to_device(DEV)
    x = torch.randn(3, 7, 512, device=DEV, dtype=dtype, requires_grad=True)
    y = lowp.QuantLinearFn.apply(x, qw, lin.bias)
    x2 = x.detach().clone().requires_grad_(True)
    y_ref = emulated_linear(x2, qw, lin.bias)
    assert y.dtype == dtype and y.shape == (3, 7, 384)
    rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
    assert rel < (1e-5 if dtype == torch.float32 else 1e-2)
    gy = torch.randn_like(y)
    y.backward(gy)
    y_ref.backward(gy.float())
    relg = ((x.grad.float() - x2.grad.float()).norm() / x2.grad.float().norm()).item()
    assert relg < (1e-5 if dtype == torch.float32 else 1e-2)
    assert lin.bias.grad is not None


# --------------------------------------------------------------------------- training
class Emulated(nn.Module):
    def __init__(self, lin, kind):
        super().__init__()
        self.qw = Q.make_quantized(lin.weight, kind, 64, pin=False)
        self.bias = lin.bias

    def forward(self, x):
        return emulated_linear(x, self.qw, self.bias)


def emulated_reference(model, kind):
    ref = copy.deepcopy(model)
    for b in ref.blocks:
        b.fc1, b.fc2 = Emulated(b.fc1, kind), Emulated(b.fc2, kind)
    return ref


@pytest.mark.parametrize("kind", KINDS)
def test_tensor_core_training_near_far_and_emulation(kind):
    needs_native(kind)
    base = make()
    runs = {}
    for placement in ("near", "far"):
        m = copy.deepcopy(base)
        m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                                  param_offload=(placement == "far"), activation_offload=False)
        try:
            s = trainstream.wrapper.unwrap_session(m)
            assert len(s.lowp_modules) == 12  # fc1 and fc2 of six blocks
            assert (len(s.pager.qnear) == 6) == (placement == "near")
            for b in m.blocks:
                assert b.fc1.weight.untyped_storage().nbytes() == 0  # never expanded
            runs[placement] = train(m, opt, tokens(6))
            ev = trainstream.memory_report(m)["events"]
            assert any("tensor cores" in e for e in ev)
            if placement == "far":
                assert s.pager.stats["load_bytes"] > 0
        finally:
            trainstream.unwrap(m)
    assert runs["near"] == runs["far"]  # staged or resident packed bytes: the same arithmetic
    ref = emulated_reference(base, kind)
    opt_ref = torch.optim.AdamW(trainable(ref), foreach=False, **HP)
    # FP4 rounding of activations amplifies accumulation-order differences a little each step
    assert runs["near"] == pytest.approx(train(ref, opt_ref, tokens(6)), rel=2e-3)


@pytest.mark.parametrize("kind", KINDS)
def test_without_tensor_core_support_weights_are_expanded(kind, monkeypatch):
    monkeypatch.setattr(lowp, "native_supported", lambda k, d: False)
    base = make()
    ref = copy.deepcopy(base)
    with torch.no_grad():
        for b in ref.blocks:
            for lin in (b.fc1, b.fc2):
                lin.weight.copy_(Q.make_quantized(lin.weight, kind, 64, pin=False).dequantize(DEV))
    opt_ref = torch.optim.AdamW(trainable(ref), foreach=False, **HP)
    m = copy.deepcopy(base)
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                              param_offload=True, activation_offload=False)
    try:
        s = trainstream.wrapper.unwrap_session(m)
        assert s.lowp_modules == []
        assert any("expanded before use" in e for e in s.events)
        data = tokens(5)
        assert train(m, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
    finally:
        trainstream.unwrap(m)
    assert torch.equal(m.blocks[0].fc1.weight, ref.blocks[0].fc1.weight)


@pytest.mark.parametrize("kind", KINDS)
def test_state_dict_unwrap_and_shapes(kind):
    needs_native(kind)
    base = make()
    m = copy.deepcopy(base)
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                              activation_offload=False)
    try:
        qw = Q.quantized_of(m.blocks[2].fc2.weight)
        want = qw.dequantize(DEV)
        sd = m.state_dict()
        assert torch.equal(sd["blocks.2.fc2.weight"].to(DEV), want)  # the values training uses
        m.load_state_dict(sd)  # round trip re-quantizes to the same codes
        assert torch.equal(Q.quantized_of(m.blocks[2].fc2.weight).dequantize(DEV), want)
        train(m, opt, tokens(2))
    finally:
        trainstream.unwrap(m)
    assert torch.equal(m.blocks[2].fc2.weight, want)  # unwrap leaves the dequantized values
    assert "forward" not in m.blocks[2].fc2.__dict__  # the plain Linear forward is back
    m(tokens(1)[0][:, :-1])
    # block-scaled formats only take tensor-core shapes
    odd = nn.Linear(100, 64).to(DEV)
    odd.weight.requires_grad_(False)
    assert not Q.eligible(odd.weight, kind) and Q.eligible(odd.weight, "nf4")


@pytest.mark.parametrize("kind", KINDS)
def test_a_partial_load_keeps_tensor_core_weights(kind):
    # load_state_dict(strict=False) with adapters only (how PEFT loads them): tensor-core
    # weights have no expansion, so the ones the dict leaves out must not be re-quantized
    needs_native(kind)
    m = make()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                              activation_offload=False)
    try:
        want = {n: Q.quantized_of(p).dequantize(DEV) for n, p in m.named_parameters() if Q.quantized_of(p) is not None}
        adapters = {k: v.clone() for k, v in m.state_dict().items() if ".down." in k or ".up." in k}
        m.load_state_dict(adapters, strict=False)
        for n, p in m.named_parameters():
            if n in want:
                assert torch.equal(Q.quantized_of(p).dequantize(DEV), want[n]), n
                assert p.untyped_storage().nbytes() == 0  # still never expanded
        train(m, opt, tokens(1))
    finally:
        trainstream.unwrap(m)


def test_invalid_kind_names_the_choices():
    m = make()
    with pytest.raises(ValueError, match="nvfp4"):
        trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), quantize="fp4")
