"""MoE expert paging (trainstream.experts): frozen experts in host RAM, routed ones copied in."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
transformers = pytest.importorskip("transformers")

import trainstream  # noqa: E402
from trainstream import experts  # noqa: E402


def tiny_moe(seed=0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=8,
                         num_experts_per_tok=2, head_dim=16)
    model = Qwen3MoeForCausalLM(cfg).cuda()
    for n, p in model.named_parameters():
        if ".experts." in n:
            p.requires_grad_(False)  # frozen experts; the rest trains
    return model


def test_same_loss_and_gradients_and_remove_restores():
    ref = tiny_moe()
    mine = copy.deepcopy(ref)
    pagers = experts.install(mine, "cuda")
    assert len(pagers) == 2
    assert all(p.untyped_storage().nbytes() == 0 for n, p in mine.named_parameters() if ".experts." in n)
    ids = torch.randint(0, 128, (2, 9), device="cuda")
    a, b = ref(ids, labels=ids), mine(ids, labels=ids)
    assert torch.equal(a.loss, b.loss)
    a.loss.backward()
    b.loss.backward()
    for (n, p), q in zip(ref.named_parameters(), mine.parameters()):
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), n
    for p in pagers:
        p.remove()
    assert all(torch.equal(p, q) for p, q in zip(ref.parameters(), mine.parameters()))


def test_only_routed_experts_are_copied():
    model = tiny_moe()
    pagers = experts.install(model, "cuda")
    with torch.no_grad():
        model(torch.randint(0, 128, (1, 1), device="cuda"))  # one token: 2 of 8 experts per layer
    assert sum(p.experts_used for p in pagers) == 2 * 2
    one = pagers[0].nbytes // 8  # bytes of one expert
    assert sum(p.bytes_in for p in pagers) == 4 * one
    for p in pagers:
        p.remove()


def test_wrap_with_moe_paging_trains_and_reports():
    model = tiny_moe()
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), moe_paging=True, profile=False)
    try:
        session = trainstream.wrapper.unwrap_session(model)
        assert any(e.startswith("MoE: 2 stacks") for e in session.events)
        for _ in range(2):
            ids = torch.randint(0, 128, (2, 9), device="cuda")
            model(ids, labels=ids).loss.backward()
            opt.step()
            opt.zero_grad()
        rep = trainstream.memory_report(model)
        assert rep["experts"]["stacks"] == 2 and rep["experts"]["experts_copied"] > 0
    finally:
        trainstream.unwrap(model)
    assert all(p.untyped_storage().nbytes() > 0 for n, p in model.named_parameters() if ".experts." in n)
