"""DPO building blocks: chunked per-sequence log-probs, the loss, and the
reference model as the same (paged) base with LoRA adapters switched off."""

import math

import pytest
import torch
from torch.nn import functional as F

import trainstream
from trainstream.losses import dpo_loss, sequence_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"


def dense_logprobs(hidden, weight, bias, labels, average=False):
    logits = (hidden[:, :-1].float() @ weight.float().t()) + (bias.float() if bias is not None else 0)
    t = labels[:, 1:]
    valid = t != -100
    lp = torch.log_softmax(logits, -1).gather(-1, t.clamp_min(0)[..., None]).squeeze(-1) * valid
    out = lp.sum(-1)
    return out / valid.sum(-1).clamp_min(1) if average else out


@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_sequence_logprobs_match_dense_values_and_gradients(average, with_bias):
    torch.manual_seed(0)
    B, T, D, V = 3, 17, 32, 101
    hidden = torch.randn(B, T, D, device=DEV, requires_grad=True)
    weight = torch.randn(V, D, device=DEV, requires_grad=True)
    bias = torch.randn(V, device=DEV, requires_grad=True) if with_bias else None
    labels = torch.randint(0, V, (B, T), device=DEV)
    labels[:, :5] = -100  # the prompt does not count
    labels[2, 10:] = -100  # padding
    got = sequence_logprobs(hidden, weight, labels, bias, chunk_size=7, average=average)
    want = dense_logprobs(hidden, weight, bias, labels, average=average)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-4)
    coef = torch.randn(B, device=DEV)
    g1 = torch.autograd.grad((got * coef).sum(), [hidden, weight] + ([bias] if with_bias else []))
    g2 = torch.autograd.grad((want * coef).sum(), [hidden, weight] + ([bias] if with_bias else []))
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_dpo_loss_is_the_published_formula():
    pc, pr = torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -18.0])
    rc, rr = torch.tensor([-11.0, -19.0]), torch.tensor([-11.0, -19.0])
    beta = 0.2
    loss, cr, rj = dpo_loss(pc, pr, rc, rr, beta=beta)
    logits = beta * ((pc - pr) - (rc - rr))
    assert torch.allclose(loss, -F.logsigmoid(logits).mean())
    assert torch.allclose(cr, beta * (pc - rc)) and torch.allclose(rj, beta * (pr - rr))
    smooth, _, _ = dpo_loss(pc, pr, rc, rr, beta=beta, label_smoothing=0.1)
    assert torch.allclose(smooth, (-0.9 * F.logsigmoid(logits) - 0.1 * F.logsigmoid(-logits)).mean())
    same, _, _ = dpo_loss(rc, rr, rc, rr)
    assert math.isclose(same.item(), math.log(2), rel_tol=1e-6)  # policy == reference


# --------------------------------------------------------------------------- Hugging Face + PEFT
def _hf():
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    return pytest.importorskip("peft")


def tiny_lora(seed=0):
    peft = _hf()
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = LlamaForCausalLM(cfg).cuda()
    lcfg = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    return peft.get_peft_model(base, lcfg)


def pairs(n=2, seed=5):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, 256, (4, 24), generator=g).cuda()
        labels = ids.clone()
        labels[:, :8] = -100
        out.append({"input_ids": ids, "labels": labels})
    return out


def test_causal_lm_logprobs_match_the_models_own_logits():
    from trainstream.integrations.huggingface import causal_lm_logprobs
    model = tiny_lora()
    b = pairs(1)[0]
    with torch.no_grad():
        logits = model(input_ids=b["input_ids"]).logits.float()
        t = b["labels"][:, 1:]
        valid = t != -100
        want = (torch.log_softmax(logits[:, :-1], -1).gather(-1, t.clamp_min(0)[..., None]).squeeze(-1) * valid).sum(-1)
        got = causal_lm_logprobs(model, b["input_ids"], b["labels"])
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-3)


def test_reference_is_the_paged_base_with_adapters_off_and_pages_no_activations():
    from trainstream.integrations.huggingface import causal_lm_logprobs, reference_logprobs, wrap_hf
    model = tiny_lora()
    for n, p in model.named_parameters():  # adapters that are not zero: policy != reference
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    data = pairs(2)
    with torch.no_grad(), model.disable_adapter():
        want = torch.cat([causal_lm_logprobs(model, b["input_ids"], b["labels"]).cpu() for b in data])
    with torch.no_grad():
        policy = torch.cat([causal_lm_logprobs(model, b["input_ids"], b["labels"]).cpu() for b in data])
    assert not torch.allclose(policy, want)
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = wrap_hf(model, torch.optim.AdamW(params, lr=1e-3), offload=True, param_offload=True,
                         activation_offload=1.0, min_offload_bytes=0, profile=False)
    s = trainstream.wrapper.unwrap_session(model)
    packed = s.activations.stats["packed_tensors"] if s.activations is not None else 0
    got = reference_logprobs(model, data)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-4)  # paging is exact
    assert model.training  # restored
    if s.activations is not None:
        assert s.activations.stats["packed_tensors"] == packed  # no-grad forwards page nothing
    trainstream.unwrap(model)


def test_dpo_steps_through_a_wrapped_lora_model_raise_the_margin():
    from trainstream.integrations.huggingface import causal_lm_logprobs, reference_logprobs, wrap_hf
    model = tiny_lora()
    chosen, rejected = pairs(1, seed=11)[0], pairs(1, seed=12)[0]
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = wrap_hf(model, torch.optim.AdamW(params, lr=5e-3), offload=True, param_offload=True,
                         activation_offload=False, profile=False)
    ref = reference_logprobs(model, [chosen, rejected])
    ref_c, ref_r = ref[:4], ref[4:]
    margins, losses = [], []
    for _ in range(6):
        pc = causal_lm_logprobs(model, chosen["input_ids"], chosen["labels"])
        pr = causal_lm_logprobs(model, rejected["input_ids"], rejected["labels"])
        loss, cr, rr = dpo_loss(pc, pr, ref_c, ref_r, beta=0.1)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        margins.append((cr - rr).mean().item())
    assert math.isclose(losses[0], math.log(2), rel_tol=1e-3)  # LoRA starts at zero: policy == reference
    assert losses[-1] < losses[0] and margins[-1] > margins[0]
    trainstream.unwrap(model)
