"""Fused RMSNorm / rotary embedding / gated MLP kernels against PyTorch, and their install on HF models."""

import sys

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import trainstream  # noqa: E402
from trainstream import _C, fused  # noqa: E402

DTYPES = [torch.float32, torch.bfloat16, torch.float16]
TOL = {torch.float32: 1e-5, torch.bfloat16: 2e-2, torch.float16: 2e-3}


def close(a, b, dtype, scale=1.0):
    assert a.shape == b.shape
    err = (a.float() - b.float()).abs().max().item()
    assert err <= TOL[dtype] * scale * max(1.0, b.float().abs().max().item()), err


def ref_norm(x, w, eps, gemma):
    n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (n * (1 + w.float())).to(x.dtype) if gemma else w * n.to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gemma", [False, True], ids=["llama", "gemma"])
@pytest.mark.parametrize("cols", [1000, 4096])
def test_rms_norm_forward_and_backward(dtype, gemma, cols):
    torch.manual_seed(0)
    x = torch.randn(3, 37, cols, device="cuda", dtype=dtype, requires_grad=True)
    w = (1 + 0.1 * torch.randn(cols, device="cuda")).to(dtype).requires_grad_()
    y = fused.RMSNormFn.apply(x, w, 1e-6, gemma)
    x2, w2 = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    y2 = ref_norm(x2.float(), w2.float(), 1e-6, gemma)
    close(y, y2, dtype)
    g = torch.randn_like(y)
    y.backward(g)
    y2.backward(g.float())
    close(x.grad, x2.grad, dtype, 4)
    close(w.grad, w2.grad, dtype, 4)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_cos", [True, False], ids=["cos-per-batch", "cos-broadcast"])
def test_rope_forward_and_backward(dtype, batch_cos):
    torch.manual_seed(0)
    B, T, H, D = 2, 9, 4, 64
    q = torch.randn(B, T, H, D, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()  # HF layout
    ang = torch.randn(B if batch_cos else 1, T, D // 2, device="cuda")
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    y = fused.RoPEFn.apply(q, cos, sin)
    q2 = q.detach().float().requires_grad_()
    y2 = q2 * cos.float().unsqueeze(1) + rotate_half(q2) * sin.float().unsqueeze(1)
    close(y, y2, dtype)
    g = torch.randn_like(y)
    y.backward(g)
    y2.backward(g.float())
    close(q.grad, q2.grad, dtype, 2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act", ["silu", "gelu_tanh"])
def test_gated_activation_forward_and_backward(dtype, act):
    torch.manual_seed(0)
    g = torch.randn(5, 333, device="cuda", dtype=dtype, requires_grad=True)
    u = torch.randn(5, 333, device="cuda", dtype=dtype, requires_grad=True)
    code = fused.ACT_SILU if act == "silu" else fused.ACT_GELU_TANH
    h = fused.GatedActFn.apply(g, u, code)
    g2, u2 = g.detach().float().requires_grad_(), u.detach().float().requires_grad_()
    a = F.silu(g2) if act == "silu" else F.gelu(g2, approximate="tanh")
    h2 = a * u2
    close(h, h2, dtype)
    d = torch.randn_like(h)
    h.backward(d)
    h2.backward(d.float())
    close(g.grad, g2.grad, dtype, 2)
    close(u.grad, u2.grad, dtype, 2)


transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402


def tiny_llama(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).cuda().to(dtype)


def test_install_matches_the_model_and_remove_restores_it():
    ref, mine = tiny_llama(), tiny_llama()
    mod = sys.modules[type(mine.model.layers[0].self_attn).__module__]
    orig_rope = mod.apply_rotary_pos_emb
    done = fused.install(mine, "cuda")
    try:
        assert len(done.norms) == 2 * 2 + 1 and len(done.mlps) == 2 and done.rope
        assert mod.apply_rotary_pos_emb is not orig_rope
        ids = torch.randint(0, 256, (2, 33), device="cuda")
        a = ref(ids, labels=ids)
        b = mine(ids, labels=ids)
        assert torch.allclose(a.logits, b.logits, atol=1e-4, rtol=1e-4)
        a.loss.backward()
        b.loss.backward()
        for (n, p), q in zip(ref.named_parameters(), mine.parameters()):
            assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-3), n
    finally:
        done.remove()
    assert mod.apply_rotary_pos_emb is orig_rope
    assert all("forward" not in m.__dict__ for m in mine.modules())


def test_unknown_norms_are_left_alone():
    class OddRMSNorm(torch.nn.Module):  # a norm with an extra offset: not what the kernel computes
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))
            self.eps = 1e-6

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight + 0.5

    m = torch.nn.Sequential(OddRMSNorm()).cuda()
    done = fused.install(m, "cuda")
    assert not done.norms
    done.remove()


@pytest.mark.parametrize("quantize, kernels, on", [("nf4", "auto", True), ("none", "auto", False),
                                                   ("none", True, True), ("nf4", False, False)])
def test_wrap_turns_kernels_on_for_quantized_bases(quantize, kernels, on):
    model = tiny_llama(dtype=torch.bfloat16)
    for n, p in model.named_parameters():
        p.requires_grad_("embed" in n)  # a frozen block stack, as in LoRA
    model, opt = trainstream.wrap(model, None, quantize=quantize, kernels=kernels, profile=False, lr=1e-3)
    try:
        session = trainstream.wrapper.unwrap_session(model)
        assert (session.fused is not None) == on
        assert any(e.startswith("fused kernels:") for e in session.events) == on
        ids = torch.randint(0, 256, (2, 17), device="cuda")
        model(ids, labels=ids).loss.backward()
        opt.step()
        opt.zero_grad()
    finally:
        trainstream.unwrap(model)
    assert all("forward" not in m.__dict__ for m in model.modules() if not isinstance(m, torch.nn.Linear))
