"""Quality options for quantized bases: keep_precise, LoftQ (error_rank) and double_quant."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import trainstream  # noqa: E402
from trainstream import _C, lowp  # noqa: E402
from trainstream import quant as Q  # noqa: E402

from test_quant import HP, make, tokens, train, trainable  # noqa: E402

DEV = "cuda"


# --------------------------------------------------------------------------- double_quant
@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
@pytest.mark.parametrize("n,offset", [(64 * 4097, 0), (64 * 513 + 6, 1)], ids=["8-per-thread", "odd-unaligned"])
def test_double_quant_kernel_matches_reference(kind, n, offset):
    torch.manual_seed(0)
    x = torch.randn(n, device=DEV) * torch.logspace(-2, 1, n, device=DEV)  # scales across several octaves
    q, a = Q.quantize_tensor(x, kind, 64, double_quant=True)
    assert a.dtype == torch.uint8 and a.numel() == Q.dq_bytes(-(-n // 64))
    out = torch.empty(n + offset, device=DEV, dtype=torch.bfloat16)[offset:]
    ev, keep = _C.dequantize_async(out, q, a, kind, 64, _C.record_event(0))
    ev.wait()
    torch.cuda.synchronize()
    assert torch.equal(out, Q.dequantize_reference(q, a, kind, 64, n, torch.bfloat16))


@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
def test_double_quant_costs_little_precision(kind):
    torch.manual_seed(0)
    w = torch.randn(4096, 512, device=DEV) * 0.02
    q0, a0 = Q.quantize_tensor(w, kind, 64)
    q1, a1 = Q.quantize_tensor(w, kind, 64, double_quant=True)
    e0 = (Q.dequantize_reference(q0, a0, kind, 64, w.numel(), torch.float32).view_as(w) - w).norm() / w.norm()
    e1 = (Q.dequantize_reference(q1, a1, kind, 64, w.numel(), torch.float32).view_as(w) - w).norm() / w.norm()
    assert a1.numel() < a0.numel() * a0.element_size() / 3.5
    assert e1 <= e0 * 1.08


@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
def test_double_quant_trains_like_its_dequantized_weights(kind):
    m = make()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                              double_quant=True, activation_offload=False, profile=False)
    try:
        qws = [Q.quantized_of(p) for p in m.parameters() if Q.quantized_of(p) is not None]
        assert qws and all(qw.host_absmax.dtype == torch.uint8 for qw in qws)
        losses = train(m, opt, tokens(3))
        assert all(l == l for l in losses)  # finite
    finally:
        trainstream.unwrap(m)


# --------------------------------------------------------------------------- keep_precise
def test_choose_precise_takes_the_worst_within_the_budget():
    errors = [("a", 0.10, 1 << 16, (256, 256)), ("b", 0.30, 1 << 16, (256, 256)), ("c", 0.20, 1 << 16, (256, 256))]
    extra = Q.packed_bytes(1 << 16, "int8") - Q.packed_bytes(1 << 16, "nf4")
    total = 3 * Q.packed_bytes(1 << 16, "nf4")
    assert Q.choose_precise(errors, "nf4", extra / total) == {"b"}
    assert Q.choose_precise(errors, "nf4", 2 * extra / total) == {"b", "c"}
    assert Q.choose_precise(errors, "int8", 0.5) == set()  # already 8-bit
    with pytest.raises(ValueError):
        Q.keep_fraction(0.9)


@pytest.mark.parametrize("kind", ["nf4", "nvfp4"])
def test_keep_precise_keeps_the_weights_that_lose_the_most(kind):
    # the weights kept one step more precise are exactly those with the largest measured error in
    # `kind`, within the budget. What hurts differs by format: NF4 (one scale per 64) loses most on
    # scattered spikes; NVFP4 (one per 16) absorbs those and loses most on plain Gaussian weights.
    if kind == "nvfp4" and not lowp.native_supported(kind, DEV):
        pytest.skip("this GPU has no nvfp4 tensor-core matmul")
    m = make()
    target = m.blocks[1].fc2.weight
    if kind == "nf4":
        with torch.no_grad():
            spikes = torch.rand_like(target) < 0.01
            target[spikes] *= 30
    frozen = [p for p in m.parameters() if Q.eligible(p, kind)]
    want = Q.choose_precise([(id(p), Q.quant_error(p, kind), p.numel(), tuple(p.shape)) for p in frozen],
                            kind, 0.2)
    assert want
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                              keep_precise=0.2, activation_offload=False, profile=False)
    try:
        kept = {id(p) for p in m.parameters()
                if Q.quantized_of(p) is not None and Q.quantized_of(p).kind == Q.KEEP_KIND[kind]}
        assert kept == want
        if kind == "nf4":
            assert id(target) in kept
        session = trainstream.wrapper.unwrap_session(m)
        assert any("lose the most" in e for e in session.events)
        train(m, opt, tokens(2))
    finally:
        trainstream.unwrap(m)


# --------------------------------------------------------------------------- LoftQ
def test_lowrank_error_brings_the_weight_closer():
    torch.manual_seed(0)
    w = torch.randn(512, 1024, device=DEV) * 0.02
    w[:, :8] *= 30
    plain = Q.quant_error(w, "nf4")
    stored, a, b = Q.lowrank_error(w, "nf4", 64, 16, 2)
    approx = Q.roundtrip(stored.to(DEV), "nf4") + b.to(DEV).float() @ a.to(DEV).float()
    assert ((approx - w).norm() / w.norm()).item() < 0.85 * plain


def test_wrap_starts_lora_adapters_at_the_quantization_error():
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    ref = LlamaForCausalLM(cfg).cuda()
    ids = torch.randint(0, 256, (2, 17), device=DEV)
    with torch.no_grad():
        want = ref(ids).logits
    outs = {}
    for rank in (0, 8):
        torch.manual_seed(0)
        base = LlamaForCausalLM(cfg).cuda()
        model = peft.get_peft_model(base, peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                                          target_modules=["q_proj", "v_proj", "down_proj"]))
        params = [p for p in model.parameters() if p.requires_grad]
        model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize="nf4", error_rank=rank,
                                      kernels=False, fused_lora=False, profile=False)
        try:
            if rank:
                session = trainstream.wrapper.unwrap_session(model)
                assert any("LoftQ: 6 LoRA adapters" in e for e in session.events)
            with torch.no_grad():
                outs[rank] = model(ids).logits
        finally:
            trainstream.unwrap(model)
    assert (outs[8] - want).norm() < (outs[0] - want).norm()
