"""Pinned allocator, DMA helpers, activation paging, profiler and wrap()."""

import copy

import pytest
import torch
from torch import nn

import trainstream
from trainstream import _C
from trainstream.hooks import ActivationOffloader, find_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab=128, d=128, n_layer=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def lm(seed=0, **kw):
    torch.manual_seed(seed)
    return TinyLM(**kw).to(DEV)


def tokens(n, seed=3, batch=8, seq=64, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (batch, seq + 1), generator=g).to(DEV) for _ in range(n)]


def lm_loss(model, t):
    logits = model(t[:, :-1])
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))


# --------------------------------------------------------------------------- runtime
def test_pinned_allocator_roundtrip_and_cache():
    before = _C.pinned_stats()
    a = _C.pinned_empty([1024, 1024], torch.float32, "test.cache", False, True)
    assert a.is_pinned()
    assert _C.pinned_tag_bytes()["test.cache"] >= 4 << 20
    del a
    torch.cuda.synchronize()
    b = _C.pinned_empty([1024, 1024], torch.float32, "test.cache", False, True)
    after = _C.pinned_stats()
    assert after["cache_hits"] > before["cache_hits"], "released block should be reused"
    del b
    log = _C.pinned_log(16)
    assert any(op == "hit" and tag == "test.cache" for _, op, _, tag in log)


def test_mapped_allocation_is_device_visible():
    h = _C.pinned_empty([4096], torch.float32, "test.mapped", True, False)
    h.fill_(2.0)
    assert _C.pinned_stats()["mapped_bytes"] >= 4096 * 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_d2h_h2d_roundtrip(dtype):
    x = torch.randn(3, 517, 33, device=DEV).to(dtype)
    host = _C.pinned_empty(list(x.shape), dtype, "test.rt", False, True)
    ev = _C.d2h_async(host, x)
    y, ev2 = _C.h2d_alloc_async(host, ev, 0)
    ev2.wait()
    assert torch.equal(x, y)
    z = torch.empty_like(x)
    _C.h2d_async(z, host, ev).wait()
    assert torch.equal(x, z)
    ts = _C.transfer_stats(0)
    assert ts["bytes_d2h"] > 0 and ts["bytes_h2d"] > 0


def test_d2h_orders_after_producer_kernel():
    """The copy stream must wait for work queued on the compute stream."""
    x = torch.zeros(1 << 22, device=DEV)
    torch.cuda._sleep(5_000_000)  # long kernel ahead of the producer
    x.add_(7.0)
    host = _C.pinned_empty([1 << 22], torch.float32, "test.order", False, True)
    _C.d2h_async(host, x).synchronize()
    assert torch.all(host == 7.0)


def test_timing_events():
    a = _C.record_event(-1, True)
    torch.cuda._sleep(1_000_000)
    b = _C.record_event(-1, True)
    assert a.elapsed_ms(b) > 0


# --------------------------------------------------------------------------- activations
def test_find_blocks():
    model = lm()
    assert find_blocks(model) == list(model.blocks)


@pytest.mark.parametrize("depth", [1, 2])
def test_activation_offload_gradients_bit_exact(depth):
    ref, mine = lm(), lm()
    t = tokens(1)[0]
    lm_loss(ref, t).backward()

    off = ActivationOffloader(mine, list(mine.blocks), offload=1.0, min_bytes=0, prefetch_depth=depth)
    try:
        lm_loss(mine, t).backward()
    finally:
        off.remove()
    torch.cuda.synchronize()
    for pr, pm in zip(ref.parameters(), mine.parameters()):
        assert torch.equal(pr.grad, pm.grad)
    s = off.stats
    assert s["packed_tensors"] > 0 and s["packed_bytes"] > 0
    assert s["prefetch_hits"] > 0
    assert s["staged_blocks"] == 3  # all but the last block were paged


class SdpaBlock(nn.Module):
    """Fused-qkv attention: q/k/v reach SDPA as strided (B,H,T,D) views."""

    def __init__(self, d, h=8):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.h, d // self.h).transpose(1, 2) for z in (q, k, v))
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x + self.out(y.transpose(1, 2).reshape(b, t, d))


@pytest.mark.parametrize("backend", ["EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"])
def test_activation_offload_preserves_layout_for_fused_attention(backend):
    """Regression: memory-efficient attention backward is wrong if its saved
    tensors come back with a different stride order.

    The memory-efficient backward accumulates with atomics, so its last bits
    depend on scheduling: even without any paging, two runs differ (up to
    ~3e-3 relative) when other GPU work changes the timing. That backend is
    therefore compared bit-exactly in PyTorch's deterministic mode. cuDNN
    attention is not available in that mode (PyTorch: "cuDNN SDPA is not
    deterministic") and is compared within 1% instead — the layout bug this
    guards against produced errors of 54.75."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def run(offload):
        torch.manual_seed(0)
        model = nn.Sequential(*[SdpaBlock(512) for _ in range(3)]).to(DEV)
        off = ActivationOffloader(model, list(model), offload=offload, min_bytes=0)
        x = torch.randn(4, 256, 512, device=DEV)
        was = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(backend == "EFFICIENT_ATTENTION")
        try:
            with sdpa_kernel(getattr(SDPBackend, backend)), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x).float().square().mean().backward()
        finally:
            torch.use_deterministic_algorithms(was)
            off.remove()
        return [p.grad.clone() for p in model.parameters()], off.stats

    try:
        ref, _ = run(0.0)
    except RuntimeError as exc:
        pytest.skip(f"{backend} unavailable: {exc}")
    mine, stats = run(1.0)
    assert stats["packed_tensors"] > 0
    for a, b in zip(ref, mine):
        rel = ((a - b).abs().max() / a.abs().max()).item()
        if backend == "CUDNN_ATTENTION":
            assert rel < 1e-2, f"max relative difference {rel:.3e}"
        else:
            assert torch.equal(a, b), f"max relative difference {rel:.3e}"


def test_dense_like_keeps_dimension_order():
    from trainstream.hooks import _dense_like

    base = torch.randn(2, 16, 3 * 64, device=DEV)
    q = base[..., :64].view(2, 16, 4, 16).transpose(1, 2)  # gapped, permuted view
    d = _dense_like(q)
    assert _C.is_dense(d) and torch.equal(d, q)
    # Same memory order as q: (B, T, H, D) outer-to-inner.
    assert sorted(range(4), key=lambda i: -d.stride(i)) == sorted(range(4), key=lambda i: -q.stride(i))


def test_paged_activations_are_freed_every_step():
    """Regression: pinned stashes must be released when the graph dies, not
    whenever Python's cyclic GC happens to run."""
    import gc

    model = lm(d=256, n_layer=4)
    off = ActivationOffloader(model, list(model.blocks), offload=1.0, min_bytes=0)
    gc.disable()
    try:
        live = []
        for t in tokens(5, batch=8, seq=128):
            lm_loss(model, t).backward()
            torch.cuda.synchronize()
            live.append(_C.pinned_tag_bytes().get("activations", 0))
        assert off.stats["packed_tensors"] > 0
        assert live[-1] == 0, live
    finally:
        gc.enable()
        off.remove()


def test_activation_offload_reduces_vram():
    def peak(offload):
        model = lm(d=256, n_layer=6)
        t = tokens(1, batch=16, seq=256)[0]
        off = ActivationOffloader(model, list(model.blocks), offload=offload, min_bytes=0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = lm_loss(model, t)
        torch.cuda.synchronize()
        # Every D2H copy has landed; release their sources as the next block
        # boundary would. Otherwise the number depends on how far the forward
        # ran ahead of the copy engine when it last checked.
        off._reap()
        after_fwd = torch.cuda.memory_allocated() - base
        loss.backward()
        off.remove()
        return after_fwd

    resident = peak(0.0)
    paged = peak(1.0)
    assert paged < 0.6 * resident, (resident, paged)


def test_activation_auto_calibrates_then_tunes():
    model = lm(d=256, n_layer=6)
    prof = trainstream.profile_hardware()
    off = ActivationOffloader(model, list(model.blocks), offload="auto", min_bytes=0, calibrate_steps=3,
                              profile=prof, max_slowdown=0.15, tune_window=2)
    try:
        for t in tokens(12, batch=8, seq=128):
            lm_loss(model, t).backward()
            torch.cuda.synchronize()
        assert not off.calibrating
        d = off.decision
        assert d["block_bytes"] > 0 and d["block_fwd_ms"] > 0 and d["baseline_step_ms"] > 0
        assert 0.0 <= d["fraction"] <= 1.0
        assert d["predicted_blocks"] == off._blocks_for_fraction(d["fraction"])
        # Tuning only ever shrinks the prediction, and stops once settled.
        assert len(off.offload) <= d["predicted_blocks"]
        assert off.phase in ("tune", "fixed")
        for h in d["history"]:
            assert h["step_ms"] > 0
    finally:
        off.remove()


def test_no_offload_under_no_grad():
    model = lm()
    off = ActivationOffloader(model, list(model.blocks), offload=1.0, min_bytes=0)
    try:
        with torch.no_grad():
            model(tokens(1)[0][:, :-1])
        assert off.stats["packed_tensors"] == 0
    finally:
        off.remove()


# --------------------------------------------------------------------------- profiler
def test_profiler_measures_sane_numbers():
    p = trainstream.profile_hardware()
    assert p.h2d_gbps > 1 and p.d2h_gbps > 1
    assert p.matmul_tflops_bf16 > 1
    assert p.cpu_adamw_gelems > 0 and p.zero_copy_adamw_gelems > 0
    assert p.sweep_h2d and max(p.sweep_h2d) >= 1 << 20
    assert p.transfer_ms(p.h2d_gbps * 1e9 / 1e3) == pytest.approx(1.0)
    assert 0 <= p.activation_offload_fraction(64 << 20, 50.0, 12) <= 1
    assert 1.0 <= p.duplex_factor <= 2.0
    assert p.recommend_optimizer_mode(10_000_000) in ("cpu", "gpu")
    assert "PCIe" in p.summary()


# --------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_wrap_end_to_end_matches_baseline(mode):
    ref = lm()
    mine = copy.deepcopy(ref)
    hp = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **hp)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **hp), offload=True,
                                 optimizer_mode=mode, activation_offload=1.0, min_offload_bytes=0)
    try:
        losses_ref, losses = [], []
        for t in tokens(6):
            for model, o, out in ((ref, opt_ref, losses_ref), (mine, opt, losses)):
                loss = lm_loss(model, t)
                loss.backward()
                o.step()
                o.zero_grad()
                out.append(loss.item())
        opt.synchronize()
        assert losses == pytest.approx(losses_ref, abs=1e-4)
        assert losses[-1] < losses[0]
        stats = trainstream.memory_report(mine)
        assert stats["activations"]["packed_bytes"] > 0
        assert stats["transfers"]["bytes_h2d"] > 0
    finally:
        trainstream.unwrap(mine)
    assert trainstream.wrapper.unwrap_session(mine) is None
