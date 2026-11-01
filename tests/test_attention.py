"""Attention-backend selection."""

import copy

import pytest
import torch

import trainstream
from trainstream.attention import time_backends

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

DEV = "cuda"


def tiny_llama(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    return LlamaForCausalLM(cfg).to(DEV)


def step(model, opt, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


def ids(n=4, seq=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 512, (2, seq), generator=g).to(DEV) for _ in range(n)]


def test_timing_lists_only_backends_that_run():
    t = time_backends(1, 4, 256, 64, torch.bfloat16, DEV)
    assert t and all(v > 0 for v in t.values())
    assert set(t) <= {"cudnn", "flash", "efficient"}
    t32 = time_backends(1, 4, 256, 64, torch.float32, DEV)
    assert "cudnn" not in t32  # cuDNN attention takes half precision only


def test_auto_picks_the_measured_fastest_and_trains_like_the_default():
    base = tiny_llama()
    runs = {}
    for mode in ("auto", None):
        m = copy.deepcopy(base)
        m, opt = trainstream.wrap(m, torch.optim.AdamW(m.parameters(), lr=1e-3), offload=True, attention=mode,
                                  activation_offload=False, profile=False)
        try:
            runs[mode] = [step(m, opt, x) for x in ids()]
            rep = trainstream.memory_report(m)
            if mode == "auto":
                a = rep["attention"]
                assert a["backend"] == min(a["timings_ms"], key=a["timings_ms"].get)
                assert any(e.startswith("attention:") for e in rep["events"])
            else:
                assert "attention" not in rep
        finally:
            trainstream.unwrap(m)
    assert runs["auto"] == pytest.approx(runs[None], rel=2e-2)  # a different kernel, the same math (bf16)


def test_recomputed_blocks_use_the_same_backend_order():
    base = tiny_llama()
    runs = []
    for rc in (False, True):
        m = copy.deepcopy(base)
        m, opt = trainstream.wrap(m, torch.optim.AdamW(m.parameters(), lr=1e-3), offload=True, attention="efficient",
                                  activation_offload=False, recompute=rc, profile=False)
        try:
            runs.append([step(m, opt, x) for x in ids(3)])
            s = trainstream.wrapper.unwrap_session(m)
            assert all(getattr(b, "_ts_sdpa_order", None) for b in s.blocks)
        finally:
            trainstream.unwrap(m)
    assert runs[0] == pytest.approx(runs[1], rel=1e-3)


def test_explicit_backends_and_invalid_values():
    m = tiny_llama()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(m.parameters(), lr=1e-3), offload=True, attention="math",
                              activation_offload=False, profile=False)
    try:
        step(m, opt, ids(1)[0])
        assert trainstream.memory_report(m)["attention"]["backend"] == "math"
    finally:
        trainstream.unwrap(m)
    assert all("_ts_sdpa_order" not in b.__dict__ for b in m.model.layers)  # unwrap removes it
    with pytest.raises(ValueError, match="attention"):
        trainstream.wrap(tiny_llama(), torch.optim.AdamW(tiny_llama().parameters()), attention="flash3")


def test_deterministic_mode_leaves_out_nondeterministic_kernels():
    from trainstream.attention import _order
    from torch.nn.attention import SDPBackend
    measured = {"cudnn": 1.0, "efficient": 2.0}
    try:
        # warn_only=False: PyTorch runs memory-efficient attention's deterministic backward
        torch.use_deterministic_algorithms(True)
        order = _order("cudnn", measured)
        assert SDPBackend.CUDNN_ATTENTION not in order and SDPBackend.EFFICIENT_ATTENTION in order
        # warn_only=True: it keeps the nondeterministic one (and only warns), so it is left out too
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert _order("cudnn", measured) == [SDPBackend.MATH]
    finally:
        torch.use_deterministic_algorithms(False)
    assert _order("cudnn", measured)[0] == SDPBackend.CUDNN_ATTENTION


def test_deterministic_mode_switched_after_wrap_reaches_recomputed_blocks():
    from torch.nn.attention import SDPBackend
    m = tiny_llama()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(m.parameters(), lr=1e-3), offload=True, attention="efficient",
                              activation_offload=False, recompute=True, profile=False)
    s = trainstream.wrapper.unwrap_session(m)
    batches = ids(2)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        step(m, opt, batches[0])
        assert all(b._ts_sdpa_order == [SDPBackend.MATH] for b in s.blocks)
    finally:
        torch.use_deterministic_algorithms(False)
    try:
        step(m, opt, batches[1])
        assert all(b._ts_sdpa_order[0] == SDPBackend.EFFICIENT_ATTENTION for b in s.blocks)
    finally:
        trainstream.unwrap(m)
