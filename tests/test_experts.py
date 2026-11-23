"""MoE expert paging (trainstream.experts): frozen experts in host RAM, routed ones copied in."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
transformers = pytest.importorskip("transformers")

import trainstream  # noqa: E402
from trainstream import experts  # noqa: E402


def tiny_moe(seed=0, layers=2, **kw):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    base = dict(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, num_experts=8,
                num_experts_per_tok=2, head_dim=16)
    base.update(kw)
    cfg = Qwen3MoeConfig(**base)
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


# --------------------------------------------------------------- cache, lookahead, capacity
def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def one_expert(pagers):
    return pagers[0].nbytes // 8


@pytest.mark.parametrize("cache_experts", [0, 3, 64], ids=["no-cache", "small-cache", "everything-fits"])
def test_cache_and_lookahead_change_nothing_but_the_copies(cache_experts):
    ref = tiny_moe(layers=3)
    mine = copy.deepcopy(ref)
    probe = experts.install(copy.deepcopy(ref), "cuda")
    budget = cache_experts * one_expert(probe)
    for p in probe:
        p.remove()
    pagers = experts.install(mine, "cuda", cache_bytes=budget)
    cache = pagers[0].cache
    assert all(p.cache is cache for p in pagers) and all(p.router is not None for p in pagers)
    g = torch.Generator().manual_seed(1)
    for step in range(3):
        ids = torch.randint(0, 128, (2, 11), generator=g).cuda()
        a, b = ref(ids, labels=ids), mine(ids, labels=ids)
        assert torch.equal(a.loss, b.loss)
        ref.zero_grad()
        mine.zero_grad()
        a.loss.backward()
        b.loss.backward()
        ga, gb = grads(ref), grads(mine)
        assert ga.keys() == gb.keys() and all(torch.equal(ga[n], gb[n]) for n in ga)  # bit for bit, every pass
        assert cache.bytes <= cache.budget
    torch.cuda.synchronize()
    st = experts.stats(pagers)
    used = st["experts_used"]  # per forward call; backward needs each of them once more
    if cache_experts == 0:
        assert st["cache_hits"] == 0 and st["prefetch_issued"] == 0 and st["cache_bytes"] == 0
        assert st["experts_copied"] == 2 * used  # forward and backward each copy what they use
    elif cache_experts == 64:
        assert st["cache_evictions"] == 0
        assert st["experts_copied"] <= 3 * 8  # each expert of each stack crossed PCIe once at most
        assert st["cache_hits"] >= used  # backward found every expert, later passes most of them
        assert st["prefetch_issued"] > 0 and 0 < st["prefetch_hits"] <= st["prefetch_issued"]
    else:
        assert st["cache_evictions"] > 0 and st["experts_copied"] < 2 * used + st["prefetch_issued"]
    for p in pagers:
        p.remove()
    assert cache.bytes == 0
    assert all(torch.equal(p, q) for p, q in zip(ref.parameters(), mine.parameters()))


def test_backward_queues_the_layer_below_before_it_gets_there():
    model = tiny_moe(layers=3)
    pagers = experts.install(model, "cuda")  # no cache: every backward copy is a real one
    ids = torch.randint(0, 128, (1, 5), device="cuda")
    loss = model(ids, labels=ids).loss
    calls = [p.last_call() for p in pagers]
    assert all(c is not None for c in calls)
    assert calls[1].below() is calls[0] and calls[2].below() is calls[1] and calls[0].below is None
    before = [p.bytes_in for p in pagers]
    calls[2].start_backward()  # what the top layer's first expert does in backward
    assert set(calls[2].entries) == set(calls[2].experts) and set(calls[1].entries) == set(calls[1].experts)
    assert not calls[0].entries  # one layer ahead, no further
    assert pagers[2].bytes_in > before[2] and pagers[1].bytes_in > before[1] and pagers[0].bytes_in == before[0]
    loss.backward()
    assert all(not c.entries for c in calls)  # each expert let go as soon as its backward ran
    for p in pagers:
        p.remove()


def test_guessing_stops_for_a_stack_whose_guesses_are_wrong(monkeypatch):
    monkeypatch.setattr(experts, "SPEC_PROBATION", 4)
    model = tiny_moe()
    pagers = experts.install(model, "cuda", cache_bytes=1 << 30)
    top = pagers[1]
    assert pagers[0].next_to_guess() is top and top.next_to_guess() is None  # nothing above the last stack
    for e in range(6):
        guess = [0] * 8
        guess[e] = 5
        top.speculate(guess)  # copied on a guess, never used
    assert top.spec_issued == 4 and top.spec_hits == 0 and top.spec_off
    assert pagers[0].next_to_guess() is None
    st = experts.stats(pagers)
    assert st["prefetch_stopped_stacks"] == 1 and st["prefetch_issued"] == 4 and st["prefetch_hits"] == 0
    with torch.no_grad():  # the model still runs, and what was guessed and is needed after all counts as a hit
        model(torch.randint(0, 128, (2, 9), device="cuda"))
    for p in pagers:
        p.remove()


def test_capacity_bounds_the_copies_and_reroutes_the_rest():
    ref = tiny_moe()
    ids = torch.randint(0, 128, (2, 17), device="cuda")
    with torch.no_grad():
        want = ref(ids).logits

    roomy = copy.deepcopy(ref)
    pagers = experts.install(roomy, "cuda", capacity=8)  # never exceeded: the model's own routing
    with torch.no_grad():
        assert torch.equal(roomy(ids).logits, want)
    assert experts.stats(pagers)["rerouted_layers"] == 0
    for p in pagers:
        p.remove()

    tight = copy.deepcopy(ref)
    pagers = experts.install(tight, "cuda", capacity=2)
    out = tight(ids, labels=ids)
    st = experts.stats(pagers)
    assert st["capacity"] == 2 and st["rerouted_layers"] == 2 and st["rerouted_tokens"] > 0
    assert st["experts_used"] == 2 * 2 and st["experts_copied"] == 2 * 2  # two experts per layer, no more
    assert torch.isfinite(out.logits).all() and not torch.equal(out.logits, want)  # a different model
    out.loss.backward()
    routers = [p.router.weight.grad for p in pagers]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in routers)  # the router trains
    for p in pagers:
        p.remove()

    with pytest.raises(ValueError, match="moe_capacity"):
        experts.install(copy.deepcopy(ref), "cuda", capacity=0)


def test_wrap_options_reach_the_pagers_and_the_report():
    model = tiny_moe()
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), moe_paging=True, moe_cache=1 << 20,
                                  moe_capacity=3, profile=False)
    try:
        session = trainstream.wrapper.unwrap_session(model)
        assert session.config["moe_cache"] == 1 << 20 and session.config["moe_capacity"] == 3
        assert any("cache in VRAM" in e and "at most 3 experts" in e for e in session.events)
        for _ in range(2):
            ids = torch.randint(0, 128, (2, 9), device="cuda")
            model(ids, labels=ids).loss.backward()
            opt.step()
            opt.zero_grad()
        rep = trainstream.memory_report(model)["experts"]
        assert rep["cache_budget"] == 1 << 20 and 0 < rep["cache_bytes"] <= 1 << 20 and rep["capacity"] == 3
        for key in ("cache_hits", "cache_misses", "prefetch_issued", "prefetch_hits", "rerouted_tokens", "bytes_copied"):
            assert key in rep
    finally:
        trainstream.unwrap(model)

    model = tiny_moe()
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), moe_paging=True, moe_cache=False,
                                  profile=False)
    try:
        assert trainstream.memory_report(model)["experts"]["cache_budget"] == 0
        assert trainstream.wrapper.unwrap_session(model).config["moe_capacity"] is None
    finally:
        trainstream.unwrap(model)
    for bad in (dict(moe_cache=-1), dict(moe_cache="big"), dict(moe_capacity=0), dict(moe_capacity=1.5)):
        m = tiny_moe()
        with pytest.raises(ValueError, match="moe_"):
            trainstream.wrap(m, torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]), moe_paging=True,
                             profile=False, **bad)


def test_experts_in_host_ram_are_not_planned_as_resident(monkeypatch):
    """The memory plan counts what VRAM holds: with moe_paging that is the expert cache, not the
    experts. A budget the rest of the model fits in must not page blocks because of them."""
    import trainstream.wrapper as wrapper
    model = tiny_moe(hidden_size=256, moe_intermediate_size=256, num_experts=64, head_dim=64)
    expert_bytes = sum(p.numel() * p.element_size() for n, p in model.named_parameters() if ".experts." in n)
    assert expert_bytes > 90 << 20
    monkeypatch.setattr(wrapper, "vram_budget", lambda device: (200 << 20, 200 << 20))  # 200 MiB "GPU"
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), moe_paging=True, profile=False,
                                  activation_offload=False)
    try:
        session = trainstream.wrapper.unwrap_session(model)
        assert session.config["plan"]["initial_paged"] == 0
        assert session.config["moe_cache"] == 20 << 20  # "auto": 10 % of the budget
    finally:
        trainstream.unwrap(model)
