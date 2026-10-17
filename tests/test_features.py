"""bf16 moments, chunked cross-entropy, the parameter guard, .to() after wrap,
and the automatic switch to a resident optimizer."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import trainstream
from trainstream import OffloadAdamW, chunked_cross_entropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
HP = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)


def mlp(seed=0, width=256):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, 10)).to(DEV)


def batches(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(32, 64, generator=g).to(DEV), torch.randint(0, 10, (32,), generator=g).to(DEV))
            for _ in range(n)]


def run(model, opt, data):
    out = []
    for x, y in data:
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        out.append(loss.item())
    return out


def max_param_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


# --------------------------------------------------------------------------- bf16 moments
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_bf16_moments_track_fp32_moments(mode):
    ref, mine = mlp(), mlp()
    opt_ref = OffloadAdamW(ref.parameters(), mode=mode, **HP)
    opt = OffloadAdamW(mine.parameters(), mode=mode, moment_dtype="bfloat16", **HP)
    data = batches(4) * 50  # a fixed set, so there is something to learn
    la, lb = run(ref, opt_ref, data), run(mine, opt, data)
    assert lb[-1] < lb[0] * 0.5
    assert abs(la[-1] - lb[-1]) < 0.05 * la[-1] + 0.02
    # second moments: stochastic rounding keeps them unbiased relative to fp32
    for slot in range(len(list(mine.parameters()))):
        _, _, v_ref = opt_ref.engine.state(slot)
        _, _, v = opt.engine.state(slot)
        assert v.dtype == torch.bfloat16
        rel = (v.float().sum() - v_ref.sum()).abs() / v_ref.sum()
        assert rel < 0.03, (slot, rel.item())
    s, s_ref = opt.stats(), opt_ref.stats()
    numel = sum(p.numel() for p in mine.parameters())
    assert s_ref["host_state_bytes"] - s["host_state_bytes"] == numel * 4  # two moments, 2 bytes saved each
    sd = opt.state_dict()  # exported as fp32, torch-compatible
    assert sd["state"][0]["exp_avg"].dtype == torch.float32


# --------------------------------------------------------------------------- chunked loss
@pytest.mark.parametrize("bias", [False, True])
def test_chunked_cross_entropy_matches_dense(bias):
    torch.manual_seed(0)
    h = torch.randn(3, 37, 64, device=DEV, requires_grad=True)
    w = torch.randn(1000, 64, device=DEV, requires_grad=True)
    b = torch.randn(1000, device=DEV, requires_grad=True) if bias else None
    t = torch.randint(0, 1000, (3, 37), device=DEV)
    t[0, :5] = -100
    dense = F.cross_entropy(F.linear(h, w, b).reshape(-1, 1000), t.reshape(-1), ignore_index=-100)
    gd = torch.autograd.grad(dense, [h, w] + ([b] if bias else []))
    ch = chunked_cross_entropy(h, w, t, b, chunk_size=16)
    gc = torch.autograd.grad(ch, [h, w] + ([b] if bias else []))
    assert ch.item() == pytest.approx(dense.item(), rel=1e-5)
    for x, y in zip(gc, gd):
        assert (x - y).abs().max().item() < 1e-5
    # sum reduction and Hugging Face's num_items normalisation
    s = chunked_cross_entropy(h, w, t, b, reduction="sum")
    assert s.item() == pytest.approx(dense.item() * int((t != -100).sum()), rel=1e-5)
    assert chunked_cross_entropy(h, w, t, b, num_items=50).item() == pytest.approx(s.item() / 50, rel=1e-5)


def test_chunked_cross_entropy_under_bf16_autocast():
    torch.manual_seed(0)
    h = torch.randn(512, 128, device=DEV, requires_grad=True)
    w = torch.randn(4000, 128, device=DEV, requires_grad=True)
    t = torch.randint(0, 4000, (512,), device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense = F.cross_entropy(F.linear(h, w).float(), t)
        ch = chunked_cross_entropy(h, w, t, chunk_size=100)
    gd = torch.autograd.grad(dense, [h, w])
    gc = torch.autograd.grad(ch, [h, w])
    assert ch.item() == pytest.approx(dense.item(), rel=1e-3)
    for x, y in zip(gc, gd):
        assert ((x.float() - y.float()).norm() / y.float().norm()).item() < 2e-2


def test_chunked_cross_entropy_never_materializes_the_logits():
    rows, vocab, d = 4096, 50000, 256
    h = torch.randn(rows, d, device=DEV, requires_grad=True)
    w = torch.randn(vocab, d, device=DEV, requires_grad=True)
    t = torch.randint(0, vocab, (rows,), device=DEV)

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = fn()
        loss.backward()
        torch.cuda.synchronize()
        h.grad = w.grad = None
        return torch.cuda.max_memory_allocated() - base

    dense = peak(lambda: F.cross_entropy(F.linear(h, w), t))
    chunked = peak(lambda: chunked_cross_entropy(h, w, t, chunk_size=512))
    logits_bytes = rows * vocab * 4
    assert dense > logits_bytes
    print(f"dense {dense / 2**20:.0f} MiB, chunked {chunked / 2**20:.0f} MiB")
    assert chunked < 0.25 * dense  # one 512-row slice at a time (plus the weight gradient)


# --------------------------------------------------------------------------- parameter guard
class EarlyUse(nn.Module):
    """Reads its output layer's (large) weight at the very start of forward,
    long before that layer's own pre-hook would wait for its update."""

    def __init__(self, d=512, vocab=32768):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.mid = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        gate = torch.sigmoid(self.head.weight.mean()) * 2
        x = self.emb(idx) * gate
        return self.head(torch.tanh(self.mid(x)))


def test_guard_waits_for_weights_used_outside_their_module():
    torch.manual_seed(0)
    ref = EarlyUse().to(DEV)
    mine = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 activation_offload=False, param_offload=False)
    g = torch.Generator().manual_seed(5)
    data = [torch.randint(0, 32768, (8, 32), generator=g).to(DEV) for _ in range(6)]
    try:
        for idx in data:
            for m, o in ((ref, opt_ref), (mine, opt)):
                F.cross_entropy(m(idx).reshape(-1, 32768).float(), idx.reshape(-1)).backward()
                o.step()
                o.zero_grad()
        opt.synchronize()
        assert opt.stats()["guard_waits"] > 0
        assert max_param_diff(ref, mine) < 1e-4
    finally:
        trainstream.unwrap(mine)


def test_guard_ignores_metadata_and_grad_access():
    model = mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False)
    try:
        x, y = batches(1)[0]
        F.cross_entropy(model(x), y).backward()
        opt.step()
        before = opt.stats()["guard_waits"]
        for p in model.parameters():  # what zero_grad, schedulers and loggers touch
            _ = p.grad, p.shape, p.dtype, p.requires_grad, p.numel(), p.size()
        opt.zero_grad()
        assert opt.stats()["guard_waits"] == before
        _ = model[0].weight.sum()  # a real read waits
        assert opt.stats()["guard_waits"] == before + 1
    finally:
        trainstream.unwrap(model)


# --------------------------------------------------------------------------- .to() after wrap
def test_round_trip_to_cpu_and_back_keeps_training_exact():
    ref, mine = mlp(), mlp()
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 activation_offload=False)
    try:
        data = batches(8)
        a = run(ref, opt_ref, data[:4])
        b = run(mine, opt, data[:4])
        mine.to("cpu")
        assert next(mine.parameters()).device.type == "cpu"
        with pytest.raises(RuntimeError, match="moved off the GPU"):
            opt.step()
        mine.cuda()
        a += run(ref, opt_ref, data[4:])
        b += run(mine, opt, data[4:])
        opt.synchronize()
        assert b == pytest.approx(a, abs=1e-4)
        assert max_param_diff(ref, mine) < 1e-4
        assert any("engine rebuilt" in e for e in trainstream.memory_report(mine)["events"])
    finally:
        trainstream.unwrap(mine)


def test_cast_after_wrap_keeps_fp32_master():
    model = mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False)
    try:
        data = batches(6)
        run(model, opt, data[:3])
        master_before = [e["master"].clone() for e in opt.state_dict()["state"].values()]
        model.to(torch.bfloat16)
        assert next(model.parameters()).dtype == torch.bfloat16
        master_after = [e["master"] for e in opt.state_dict()["state"].values()]
        assert all(torch.equal(x, y) for x, y in zip(master_before, master_after))  # not rounded to bf16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = run(model, opt, [(x.bfloat16(), y) for x, y in data[3:]])
        assert all(l == l for l in losses)
    finally:
        trainstream.unwrap(model)


# --------------------------------------------------------------------------- resident switch
def test_small_model_switches_to_resident_and_stays_exact():
    ref, mine = mlp(), mlp()
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), activation_offload=False)
    try:
        data = batches(10)
        a, b = run(ref, opt_ref, data), run(mine, opt, data)
        assert opt.resident
        assert b == pytest.approx(a, abs=1e-4)
        assert max_param_diff(ref, mine) < 1e-4
        assert opt.stats()["mode"] == "resident"
    finally:
        trainstream.unwrap(mine)


def test_offload_true_never_goes_resident():
    model = mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False)
    try:
        run(model, opt, batches(4))
        assert not opt.resident
    finally:
        trainstream.unwrap(model)


def test_resident_optimizer_is_fused_and_follows_lr_schedulers():
    model = mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), activation_offload=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 ** step)
    try:
        for x, y in batches(4):
            F.cross_entropy(model(x), y).backward()
            opt.step()
            opt.zero_grad()
            sched.step()
        inner = opt._resident
        assert inner is not None and inner.param_groups[0] is opt.param_groups[0]
        assert inner.param_groups[0]["fused"] is True
        assert inner.param_groups[0]["lr"] == pytest.approx(HP["lr"] * 0.5 ** 4)
        assert all(st["step"].is_cuda for st in inner.state.values())
        opt.load_state_dict(opt.state_dict())  # torch's loader replaces group dicts; we relink them
        assert inner.param_groups[0] is opt.param_groups[0] and inner.param_groups[0]["fused"] is True
    finally:
        trainstream.unwrap(model)
