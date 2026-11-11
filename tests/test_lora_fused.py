"""The fused LoRA path (trainstream.lora) against PEFT's own forward and backward."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import trainstream  # noqa: E402
from trainstream import lora, lowp  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_lora(seed=0, dropout=0.0, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = LlamaForCausalLM(cfg).cuda().to(dtype)
    model = peft.get_peft_model(base, peft.LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS,
                                                      lora_dropout=dropout))
    for n, p in model.named_parameters():  # non-zero B, so every gradient path is exercised
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    return model


def lora_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_matches_peft_forward_and_gradients(autocast):
    ref = tiny_lora()
    mine = copy.deepcopy(ref)
    layers = lora.install(mine)
    assert len(layers) == 2 * len(TARGETS)
    ids = torch.randint(0, 256, (2, 21), device="cuda")
    outs = []
    for m in (ref, mine):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        outs.append(out)
    tol = dict(rtol=2e-2, atol=2e-2) if autocast else dict(rtol=1e-4, atol=1e-5)
    assert torch.allclose(outs[0].logits.float(), outs[1].logits.float(), **tol)
    g_ref, g_mine = lora_grads(ref), lora_grads(mine)
    assert g_ref.keys() == g_mine.keys()
    for n in g_ref:
        scale = g_ref[n].abs().max().item() + 1e-6
        assert (g_ref[n] - g_mine[n]).abs().max().item() <= (5e-2 if autocast else 1e-4) * scale, n
    lora.uninstall(layers)
    assert all("forward" not in m.__dict__ for m in mine.modules())


def test_layers_it_cannot_compute_keep_peft():
    ref = tiny_lora(dropout=0.1)
    mine = copy.deepcopy(ref)
    lora.install(mine)
    ref.eval()
    mine.eval()  # dropout off in eval: the fused path takes over
    ids = torch.randint(0, 256, (1, 9), device="cuda")
    with torch.no_grad():
        assert torch.allclose(ref(input_ids=ids).logits, mine(input_ids=ids).logits, atol=1e-5)
    ref.train()
    mine.train()  # dropout active: PEFT's forward, unchanged
    torch.manual_seed(1)
    a = ref(input_ids=ids).logits
    torch.manual_seed(1)
    b = mine(input_ids=ids).logits
    assert torch.equal(a, b)
    with mine.disable_adapter():  # adapters off: the base alone, as PEFT does it
        with torch.no_grad():
            base = mine(input_ids=ids).logits
    with ref.disable_adapter():
        with torch.no_grad():
            assert torch.equal(base, ref(input_ids=ids).logits)


@pytest.mark.parametrize("kind", ["int8", "nf4", "fp8", "nvfp4"])
def test_trains_like_peft_on_every_quantized_base(kind):
    if kind in ("fp8", "nvfp4") and not lowp.native_supported(kind, "cuda"):
        pytest.skip(f"this GPU has no {kind} tensor-core matmul")
    losses = {}
    for fused_lora in (False, True):
        model = tiny_lora(dtype=torch.bfloat16)
        params = [p for p in model.parameters() if p.requires_grad]
        model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize=kind, kernels=False,
                                      fused_lora=fused_lora, profile=False)
        try:
            session = trainstream.wrapper.unwrap_session(model)
            assert bool(session.lora_layers) == fused_lora
            out = []
            g = torch.Generator().manual_seed(3)
            for _ in range(4):
                ids = torch.randint(0, 256, (2, 33), generator=g).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                out.append(loss.item())
            losses[fused_lora] = out
        finally:
            trainstream.unwrap(model)
    assert losses[True] == pytest.approx(losses[False], rel=1e-2, abs=1e-2)
