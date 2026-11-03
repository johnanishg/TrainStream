"""Quantized frozen weights (int8 / NF4): kernels, format, training, planner, decisions."""

import copy

import pytest
import torch
from torch import nn

import trainstream
from trainstream import _C
from trainstream import quant as Q
from trainstream.planner import Budget

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
HP = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)


# --------------------------------------------------------------------------- kernels and format
@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("where", ["cuda", "host"])
def test_kernel_matches_reference(kind, dtype, where):
    torch.manual_seed(0)
    n = 64 * 513 + 6  # a partial last block
    w = torch.randn(n, device=DEV) * 0.02
    q, a = Q.quantize_tensor(w, kind, 64)
    ref = Q.dequantize_reference(q, a, kind, 64, n, dtype)
    if where == "host":
        q, a = q.cpu().pin_memory(), a.cpu().pin_memory()
    out = torch.empty(n, device=DEV, dtype=dtype)
    ev, keep = _C.dequantize_async(out, q, a, kind, 64, _C.record_event(0))
    ev.wait()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    err = ((ref.float() - w).abs().mean() / w.abs().mean()).item()
    assert err < (0.12 if kind == "nf4" else 0.01)


@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,block,offset", [(64 * 513, 64, 0), (6 * 1000, 6, 0), (64 * 513, 64, 1)],
                         ids=["8-per-thread", "odd-block", "unaligned"])
def test_kernel_paths_match_reference(kind, dtype, n, block, offset):
    # eight values per thread when sizes and alignment allow, one at a time otherwise: the same values
    torch.manual_seed(0)
    w = torch.randn(n, device=DEV) * 0.02
    q, a = Q.quantize_tensor(w, kind, block)
    out = torch.empty(n + offset, device=DEV, dtype=dtype)[offset:]
    ev, keep = _C.dequantize_async(out, q, a, kind, block, _C.record_event(0))
    ev.wait()
    torch.cuda.synchronize()
    assert torch.equal(out, Q.dequantize_reference(q, a, kind, block, n, dtype))


def test_nf4_is_bit_identical_to_bitsandbytes():
    bf = pytest.importorskip("bitsandbytes.functional")
    torch.manual_seed(0)
    w = (torch.randn(1024, 512, device=DEV) * 0.02).to(torch.bfloat16)
    qb, st = bf.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=False)
    q, a = Q.quantize_tensor(w, "nf4", 64)
    assert torch.equal(qb.view(-1), q)  # same packed bytes
    assert torch.equal(st.absmax.float(), a)
    assert torch.equal(bf.dequantize_4bit(qb, st, quant_type="nf4"),
                       Q.dequantize_reference(q, a, "nf4", 64, w.numel(), torch.bfloat16).view_as(w))


# --------------------------------------------------------------------------- training
class Block(nn.Module):
    """A frozen 'base' MLP plus a small trainable adapter (LoRA-like)."""

    def __init__(self, d=256, r=8):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.down, self.up = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        h = self.ln(x)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(h))) + self.up(self.down(h))


class Stack(nn.Module):
    def __init__(self, d=256, n=6, vocab=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n))
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def make(seed=0):
    torch.manual_seed(seed)
    m = Stack().to(DEV)
    for b in m.blocks:  # the base is frozen, the adapters and the rest train
        for lin in (b.fc1, b.fc2):
            lin.weight.requires_grad_(False)
    return m


def tokens(n, seed=3, batch=8, seq=32, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (batch, seq + 1), generator=g).to(DEV) for _ in range(n)]


def train(model, opt, data, between=None):
    out = []
    for k, t in enumerate(data):
        logits = model(t[:, :-1])
        loss = nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        out.append(loss.item())
        if between:
            between(k)
    return out


def dequantized_reference(model, kind):
    """A copy whose frozen weights hold exactly the values quantized training uses."""
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for b in ref.blocks:
            for lin in (b.fc1, b.fc2):
                q, a = Q.quantize_tensor(lin.weight, kind, 64)
                lin.weight.copy_(Q.dequantize_reference(q, a, kind, 64, lin.weight.numel(), lin.weight.dtype)
                                 .view_as(lin.weight))
    return ref


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
@pytest.mark.parametrize("placement", ["near", "far"])
def test_quantized_training_equals_training_on_the_dequantized_weights(kind, placement):
    base = make()
    ref = dequantized_reference(base, kind)
    opt_ref = torch.optim.AdamW(trainable(ref), foreach=False, **HP)
    mine = copy.deepcopy(base)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(trainable(mine), **HP), offload=True, quantize=kind,
                                 param_offload=(placement == "far"), activation_offload=False)
    try:
        s = trainstream.wrapper.unwrap_session(mine)
        assert s.config["quantize"] == kind
        assert len(s.pager.quant_blocks) == 6
        assert (len(s.pager.qnear) == 6) == (placement == "near")
        for b in mine.blocks:  # no full-precision copy of the frozen base stays in VRAM between uses
            assert Q.quantized_of(b.fc1.weight) is not None
            assert all(Q.quantized_of(p) is None for p in trainable(b))  # trained weights never quantized
        data = tokens(6)
        assert train(mine, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
        rep = trainstream.memory_report(mine)
        assert rep["params"]["quantize"] == kind and rep["params"]["quantized_bytes"] > 0
        assert any(f"to {kind}" in e for e in rep["events"])
    finally:
        trainstream.unwrap(mine)
    # unwrap leaves the dequantized values in place
    assert torch.equal(mine.blocks[0].fc1.weight, ref.blocks[0].fc1.weight)


def test_moving_quantized_blocks_near_and_far_does_not_change_results():
    base = make()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    a, opt_a = trainstream.wrap(a, torch.optim.AdamW(trainable(a), **HP), offload=True, quantize="nf4",
                                param_offload=False, activation_offload=False)
    b, opt_b = trainstream.wrap(b, torch.optim.AdamW(trainable(b), **HP), offload=True, quantize="nf4",
                                param_offload=False, activation_offload=False)
    pager = trainstream.wrapper.unwrap_session(b).pager
    moves = {0: ([], [0, 2, 4]), 2: ([0], [1]), 4: ([1, 2, 4], [])}  # after step k: (near, far)

    def between(k):
        if k in moves:
            near, far = moves[k]
            for i in far:
                pager.page(i)
            for i in near:
                pager.unpage(i)

    try:
        data = tokens(7)
        assert train(a, opt_a, data) == train(b, opt_b, data, between)
        # far {0,2,4} after step 0; {1,2,4} after step 2; all near again after step 4
        assert pager.remote == set() and pager.qnear == set(range(6))
        assert pager.stats["pages"] >= 4 and pager.stats["unpages"] >= 4
    finally:
        trainstream.unwrap(a)
        trainstream.unwrap(b)


def test_state_dict_serves_the_dequantized_values():
    base = make()
    ref = dequantized_reference(base, "nf4")
    mine, opt = trainstream.wrap(copy.deepcopy(base), torch.optim.AdamW(trainable(base), **HP), offload=True,
                                 quantize="nf4", activation_offload=False)
    try:
        sd = mine.state_dict()
        assert torch.equal(sd["blocks.2.fc1.weight"].to(DEV), ref.blocks[2].fc1.weight)
    finally:
        trainstream.unwrap(mine)


@pytest.mark.parametrize("kind", Q.EXPAND_KINDS)
def test_a_partial_load_keeps_the_quantized_weights(kind):
    # load_state_dict(strict=False) with adapters only (how PEFT loads them): the block hooks
    # expand every weight first, and in bf16 that expansion isn't exactly the stored values,
    # so the weights the dict leaves out must not be re-quantized from it
    m = make().to(torch.bfloat16)
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, quantize=kind,
                              activation_offload=False, profile=False)
    try:
        want = {n: Q.quantized_of(p).dequantize(DEV) for n, p in m.named_parameters() if Q.quantized_of(p) is not None}
        adapters = {k: v.clone() for k, v in m.state_dict().items() if ".down." in k or ".up." in k}
        m.load_state_dict(adapters, strict=False)
        for n, p in m.named_parameters():
            if n in want:
                assert torch.equal(Q.quantized_of(p).dequantize(DEV), want[n]), n
        train(m, opt, tokens(1))
    finally:
        trainstream.unwrap(m)


# --------------------------------------------------------------------------- decisions and errors
def fake_budget(vram_gib, ram_gib):
    G = 1 << 30
    return Budget(vram_total=int(vram_gib * G), vram_usable=int(vram_gib * G), ram_total=int(64 * G),
                  ram_available=int(ram_gib * G), pin_budget=int(25 * G))


def frozen_bytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters() if not p.requires_grad)


# frozen weights here are fp32: int8 keeps ~27% of their bytes, nf4 ~14%
@pytest.mark.parametrize("ram_factor,expect", [(1.5, None), (0.5, "int8"), (0.2, "nf4")])
def test_auto_quantizes_only_when_bf16_cannot_fit(monkeypatch, ram_factor, expect):
    m = make()
    t = sum(p.numel() for p in trainable(m))
    room = 12 * t + frozen_bytes(m) * ram_factor  # master + bf16 moments + staging for the trained part
    monkeypatch.setattr(trainstream.wrapper, "measure_budget", lambda dev: fake_budget(20, room / 2**30 / 0.9))
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, activation_offload=False)
    try:
        assert trainstream.wrapper.unwrap_session(m).config["quantize"] == expect
    finally:
        trainstream.unwrap(m)


def test_auto_refuses_when_even_nf4_cannot_fit(monkeypatch):
    m = make()
    monkeypatch.setattr(trainstream.wrapper, "measure_budget", lambda dev: fake_budget(20, 0.001))
    with pytest.raises(MemoryError, match="host RAM"):
        trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, activation_offload=False)


def test_prefer_speed_picks_4bit_when_it_fits_in_vram_and_bf16_would_stream(monkeypatch):
    kind = "nf4"  # NVFP4 compute stays opt-in: it quantizes activations too
    m = make()
    blocks = list(m.blocks)
    bf16 = sum(p.numel() * p.element_size() for b in blocks for p in b.parameters())
    nf4 = sum(Q.packed_bytes(p.numel(), kind, 64, p.shape) if Q.eligible(p, kind) else p.numel() * p.element_size()
              for b in blocks for p in b.parameters())
    outside = sum(p.numel() * p.element_size() for p in m.parameters()) - bf16
    # With a small budget V the planner keeps 0.6 V - outside - 96 MiB for block weights
    # (10% headroom, 10% in-flight cap, 20% activation reserve, workspace): put that
    # between the 4-bit and the bf16 size of the blocks.
    v = (outside + (96 << 20) + (nf4 + bf16) / 2) / 0.6
    monkeypatch.setattr(trainstream.wrapper, "measure_budget", lambda dev: fake_budget(v / 2**30, 32))
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, activation_offload=False,
                              prefer="speed", check_memory=False)
    try:
        assert trainstream.wrapper.unwrap_session(m).config["quantize"] == kind
    finally:
        trainstream.unwrap(m)
    m2 = make()
    m2, opt2 = trainstream.wrap(m2, torch.optim.AdamW(trainable(m2), **HP), offload=True, activation_offload=False,
                                check_memory=False)  # prefer="exact": stream bf16 instead
    try:
        s2 = trainstream.wrapper.unwrap_session(m2)
        assert s2.config["quantize"] is None and s2.pager is not None and s2.pager.remote
    finally:
        trainstream.unwrap(m2)


def test_invalid_quantization_requests():
    full = Stack().to(DEV)  # everything trainable: nothing frozen to quantize
    with pytest.raises(ValueError, match="frozen"):
        trainstream.wrap(full, quantize="nf4")
    with pytest.raises(ValueError, match="quantize must be"):
        trainstream.wrap(make(), quantize="fp3")
    with pytest.raises(ValueError, match="prefer must be"):
        trainstream.wrap(make(), prefer="fast")
    pre = make()
    Q.quantize_model(pre, "int8")
    with pytest.raises(ValueError, match="already holds quantized"):
        trainstream.wrap(pre, torch.optim.AdamW(trainable(pre)), quantize="none")


def test_quantize_model_then_wrap_uses_the_given_kind():
    m = make()
    info = Q.quantize_model(m, "int8")
    assert info["tensors"] == 12 and info["bytes_after"] < info["bytes_before"] / 3
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, activation_offload=False)
    try:
        assert trainstream.wrapper.unwrap_session(m).config["quantize"] == "int8"
        train(m, opt, tokens(2))
    finally:
        trainstream.unwrap(m)
