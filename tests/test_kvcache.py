"""Paged KV cache: keys and values of offloaded layers in pinned host pages."""

import pytest
import torch

from trainstream import PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
B, H, D = 2, 3, 16


def run(cache, layers, steps, prefill=7, seed=0):
    """Prefill, then one token at a time through every layer in order; every
    returned k / v must equal the plain concatenation of what went in."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ref = {i: (None, None) for i in range(layers)}
    for step in range(steps):
        s = prefill if step == 0 else 1
        for i in range(layers):
            k = torch.randn(B, H, s, D, device=DEV, generator=g).to(torch.bfloat16)
            v = torch.randn(B, H, s, D, device=DEV, generator=g).to(torch.bfloat16)
            rk, rv = ref[i]
            rk = k if rk is None else torch.cat([rk, k], 2)
            rv = v if rv is None else torch.cat([rv, v], 2)
            ref[i] = (rk, rv)
            ok, ov = cache.update(i, k, v)
            assert ok.shape == rk.shape and torch.equal(ok, rk), (step, i)
            assert torch.equal(ov, rv), (step, i)
    return ref


@pytest.mark.parametrize("resident", [0, 1, 2, 4, 5])  # 5, 4, 3 (odd: three slots), 1, 0 offloaded layers
def test_every_layout_returns_exactly_the_concatenation(resident):
    cache = PagedKVCache(5, max_tokens=24, resident_layers=resident, page_tokens=4)
    run(cache, 5, steps=14)
    assert len(cache.resident) == resident
    off = 5 - resident
    if off:
        assert cache.stats["pages"] > 0 and cache.memory()["host_bytes"] > 0
    if off >= 2:
        assert cache.stats["prefetched"] > 0  # pages were copied in while earlier layers ran


def test_growing_past_max_tokens_and_reset():
    cache = PagedKVCache(4, max_tokens=8, resident_layers=[0, 2], page_tokens=4)
    run(cache, 4, steps=12)  # 7 + 11 = 18 tokens > max_tokens: slots and buffers grow
    assert cache.seq_length(0) == 18
    cache.reset()
    assert cache.seq_length(3) == 0
    run(cache, 4, steps=5, seed=3)  # starts again from nothing


def test_offloaded_layers_hold_far_less_vram():
    full = PagedKVCache(8, max_tokens=256, resident_layers=8, page_tokens=64)
    paged = PagedKVCache(8, max_tokens=256, resident_layers=0, page_tokens=64)
    run(full, 8, steps=3, prefill=200)
    run(paged, 8, steps=3, prefill=200)
    assert paged.memory()["device_bytes"] < 0.5 * full.memory()["device_bytes"]


def test_invalid_arguments():
    for kw in (dict(resident_layers="most"), dict(resident_layers=[9]), dict(page_tokens=0),
               dict(vram_fraction=0.0)):
        with pytest.raises(ValueError):
            PagedKVCache(4, 16, **kw)
    cache = PagedKVCache(2, 16)
    cache.update(0, torch.zeros(1, 2, 3, 8, device=DEV), torch.zeros(1, 2, 3, 8, device=DEV))
    with pytest.raises(ValueError):
        cache.update(1, torch.zeros(2, 2, 1, 8, device=DEV), torch.zeros(2, 2, 1, 8, device=DEV))


# --------------------------------------------------------------------------- Hugging Face generate()
def tiny_llama(seed=0):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=5,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    # fp32: the cache returns strided views, and in bf16 a different kernel's rounding
    # could flip a near-tie in greedy decoding of this random model
    return LlamaForCausalLM(cfg).cuda().float().eval()


@pytest.mark.parametrize("resident", [0, 2, "auto"])
def test_generate_gives_the_same_tokens_as_the_default_cache(resident):
    model = tiny_llama()
    from trainstream.integrations.huggingface import TrainStreamCache
    prompt = torch.randint(0, 256, (3, 9), generator=torch.Generator().manual_seed(1)).cuda()
    new = 40
    with torch.no_grad():
        want = model.generate(prompt, max_new_tokens=new, do_sample=False, pad_token_id=0)
        cache = TrainStreamCache(model, max_tokens=9 + new, resident_layers=resident, page_tokens=8)
        got = model.generate(prompt, max_new_tokens=new, do_sample=False, pad_token_id=0, past_key_values=cache)
    assert torch.equal(got, want)
    if resident != "auto":
        assert len(cache.kv.offloaded) == 5 - resident and cache.kv.stats["pages"] > 0
