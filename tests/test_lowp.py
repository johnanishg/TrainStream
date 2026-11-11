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


@pytest.mark.parametrize("kind, outliers", [("fp8", 0.0), ("nvfp4", 0.0), ("nvfp4", 0.02)],
                         ids=["fp8", "nvfp4", "nvfp4-outliers"])  # outlier channels are an nvfp4 option
def test_a_shared_input_is_quantized_once(kind, outliers, monkeypatch):
    # q / k / v read the same activation: one quantization serves all three, with the
    # same results as quantizing it for each; an in-place write invalidates it
    needs_native(kind)
    torch.manual_seed(0)
    lins = [nn.Linear(512, n, bias=False).to(DEV, torch.bfloat16) for n in (512, 128, 128)]
    qws = []
    for lin in lins:
        qw = Q.make_quantized(lin.weight, kind, 64, pin=False)
        qw.to_device(DEV)
        qw.outliers = outliers
        qws.append(qw)
    calls = []
    for name in ("_quantize", "_quantize_split"):
        real = getattr(lowp, name)
        monkeypatch.setattr(lowp, name, lambda *a, _real=real: calls.append(1) or _real(*a))
    x = torch.randn(2, 9, 512, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    lowp._CACHE.clear()
    ys = [lowp.QuantLinearFn.apply(x, qw, None) for qw in qws]
    assert len(calls) == 1
    want = []
    for qw in qws:
        lowp._CACHE.clear()
        want.append(lowp.QuantLinearFn.apply(x, qw, None))
    assert all(torch.equal(a, b) for a, b in zip(ys, want))
    with torch.no_grad():
        x.mul_(2)  # bumps the version counter: quantized again
    calls.clear()
    lowp.QuantLinearFn.apply(x, qws[0], None)
    assert len(calls) == 1
    sum(y.float().sum() for y in ys).backward()  # the first backward empties the cache
    assert lowp._CACHE.x is None


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


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp4_backward_trades_precision_for_tensor_cores(dtype):
    needs_native("nvfp4")
    torch.manual_seed(0)
    lin = nn.Linear(512, 384, bias=False).to(DEV, dtype)
    lin.weight.requires_grad_(False)
    qw = Q.make_quantized(lin.weight, "nvfp4", 64, pin=False)
    qw.to_device(DEV)
    x = torch.randn(64, 512, device=DEV, dtype=dtype, requires_grad=True)
    g = torch.randn(64, 384, device=DEV, dtype=dtype)
    exact = g.float() @ qw.dequantize(DEV).float()
    err = {}
    for fp4 in (False, True):
        qw.fp4_backward = fp4
        x.grad = None
        lowp.QuantLinearFn.apply(x, qw, None).backward(g)
        err[fp4] = ((x.grad.float() - exact).norm() / exact.norm()).item()
    assert err[False] < 0.01 < err[True] < 0.2  # measured: 0.002 and 0.12


def test_modelopt_nvfp4_checkpoint_is_read_as_stored(tmp_path):
    needs_native("nvfp4")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    import json
    from safetensors.torch import save_file
    from trainstream.integrations.huggingface import load_model
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    model = transformers.LlamaForCausalLM(cfg).to(torch.bfloat16)
    tensors, want = {}, {}
    for name, p in model.state_dict().items():
        if ".layers." in name and name.endswith("_proj.weight"):  # what ModelOpt exports in NVFP4
            w = p.float().to(DEV)
            g = Q.global_decode_scale(w)
            q, _, deq = Q.quantize_blockscaled_reference(w, "nvfp4", g)
            n, k = w.shape
            scale = (w.view(n, k // 16, 16).abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
            base = name[: -len(".weight")]
            tensors.update({name: q.cpu(), base + ".weight_scale": scale.cpu(), base + ".weight_scale_2": g.cpu(),
                            base + ".input_scale": torch.ones(1)})
            want[name] = deq.to(torch.bfloat16).cpu()
        else:
            tensors[name] = p.contiguous()
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg.save_pretrained(str(tmp_path))
    meta = json.loads((tmp_path / "config.json").read_text())
    meta["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16}
    (tmp_path / "config.json").write_text(json.dumps(meta))
    loaded = load_model(str(tmp_path), dtype=torch.bfloat16)
    assert loaded._ts_load_info["quantize"] == "nvfp4"
    assert loaded._ts_load_info["quantized_tensors"] == len(want)
    for name, p in loaded.named_parameters():
        if name in want:
            qw = Q.quantized_of(p)
            assert qw is not None and qw.kind == "nvfp4"
            assert torch.equal(qw.dequantize(DEV).cpu(), want[name]), name
