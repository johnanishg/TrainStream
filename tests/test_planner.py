"""The memory planner: decisions from the measured VRAM / RAM, and live re-planning."""

import copy
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

import trainstream
from trainstream.planner import Budget, MemoryPlanner, spread

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

GiB = 1 << 30
DEV = "cuda"
HP = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def budget(vram_gib, ram_gib=64):
    return Budget(vram_total=int(vram_gib * GiB), vram_usable=int(vram_gib * GiB), ram_total=int(ram_gib * GiB),
                  ram_available=int(ram_gib * GiB), pin_budget=int(0.4 * ram_gib * GiB))


# --------------------------------------------------------------------------- decisions
def test_spread_is_even_and_exact():
    assert spread(0, list(range(10))) == []
    assert spread(10, list(range(10))) == list(range(10))
    s = spread(4, list(range(16)))
    assert len(s) == 4 and all(b - a == 4 for a, b in zip(s, s[1:]))
    assert len(spread(7, list(range(48)))) == 7


@pytest.mark.parametrize("vram,expect_paged", [(80, 0), (24, None), (8, None), (4, None)])
def test_static_plan_follows_the_vram_budget(vram, expect_paged):
    """48 blocks of 0.55 GiB (Qwen2.5-14B in bf16) and 3 GiB outside them."""
    blocks = [int(0.55 * GiB)] * 48
    p = MemoryPlanner(budget(vram), blocks, fixed_vram=3 * GiB, inflight_bytes=GiB // 2)
    paged = p.initial_paged()
    if expect_paged == 0:
        assert paged == set()  # an 80 GB card holds the whole model
        return
    resident = 48 - len(paged)
    usable = 0.9 * vram * GiB
    assert 3 * GiB + resident * 0.55 * GiB < usable  # what stays resident fits
    if vram >= 24:
        assert resident >= 20  # most of a 24 GB card is used for weights
    assert len(paged) == len(set(paged)) and all(0 <= i < 48 for i in paged)


def test_refinement_moves_blocks_in_with_headroom_and_out_when_full():
    blocks = [GiB // 2] * 20
    p = MemoryPlanner(budget(16), blocks, fixed_vram=2 * GiB, inflight_bytes=GiB // 4)
    paged = set(range(0, 20, 2))  # 10 paged
    grown = p.refine(peak=6 * GiB, paged=paged)  # lots of headroom
    assert grown is not None and grown < paged
    shrunk = p.refine(peak=int(15.5 * GiB), paged=grown)  # too close to the budget
    assert shrunk is not None and shrunk > grown
    p.refine(peak=int(13.5 * GiB), paged=shrunk)
    p.refine(peak=int(13.5 * GiB), paged=shrunk)
    assert p.done  # two steady steps settle the plan


def test_wrap_refuses_what_cannot_fit_in_ram(monkeypatch):
    import trainstream.planner as planner
    monkeypatch.setattr(planner, "available_ram_bytes", lambda: 64 << 20)  # 64 MiB "free"
    model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(8)]).to(DEV)
    with pytest.raises(MemoryError, match="host RAM"):
        trainstream.wrap(model, torch.optim.AdamW(model.parameters()), offload=True, activation_offload=False)
    # the explicit opt-out still wraps
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters()), offload=True,
                                  activation_offload=False, check_memory=False)
    trainstream.unwrap(model)


def test_moment_precision_follows_available_ram(monkeypatch):
    import trainstream.planner as planner
    model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(8)]).to(DEV)
    numel = sum(p.numel() for p in model.parameters())
    # host need: fp32 master 4 + staging 4 + moments 2x(4 or 2) B/param -> 16 (fp32) vs 12 (bf16); give 14
    room = int(numel * 14 / 0.9)
    monkeypatch.setattr(planner, "available_ram_bytes", lambda: room)
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters()), offload=True, activation_offload=False)
    try:
        assert opt.moment_dtype == "bfloat16"
    finally:
        trainstream.unwrap(model)
    monkeypatch.setattr(planner, "available_ram_bytes", lambda: 64 << 30)
    model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(8)]).to(DEV)
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters()), offload=True, activation_offload=False)
    try:
        assert opt.moment_dtype == "float32"
    finally:
        trainstream.unwrap(model)


# --------------------------------------------------------------------------- live re-planning
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln(x))))


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


def lm_steps(model, opt, data, between=None):
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


def tokens(n, seed=3, batch=8, seq=32, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (batch, seq + 1), generator=g).to(DEV) for _ in range(n)]


def test_blocks_move_between_vram_and_ram_mid_training_without_changing_results():
    torch.manual_seed(0)
    ref = Stack().to(DEV)
    mine = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 param_offload=[0, 2, 4], activation_offload=False)
    pager = trainstream.wrapper.unwrap_session(mine).pager
    moves = {1: ([0], [1, 5]), 3: ([1, 2, 4, 5], []), 5: ([], [0, 3])}  # after step k: (unpage, page)

    def between(k):
        if k in moves:
            unpage, page = moves[k]
            for i in page:
                pager.page(i)
            for i in unpage:
                pager.unpage(i)

    try:
        data = tokens(8)
        a = lm_steps(ref, opt_ref, data)
        b = lm_steps(mine, opt, data, between)
        opt.synchronize()
        assert b == pytest.approx(a, abs=1e-4)
        assert pager.paged == {0, 3}
        sizes = [sum(p.untyped_storage().nbytes() for p in blk.parameters()) for blk in mine.blocks]
        assert all(sizes[i] > 0 for i in (1, 2, 4, 5)) and sizes[3] == 0  # resident blocks hold weights
        sd, sd_ref = mine.state_dict(), ref.state_dict()
        assert max((sd[k].float().cpu() - sd_ref[k].float().cpu()).abs().max().item() for k in sd_ref) < 2e-4
    finally:
        trainstream.unwrap(mine)


CAP_SCRIPT = textwrap.dedent(r"""
    import json, sys, torch
    sys.path[:0] = [ROOT, HERE]
    import trainstream
    from test_planner import Stack, tokens, lm_steps, HP
    cap = CAP
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.manual_seed(0)
    model = Stack(d=1024, n=8)  # 8 blocks of 50 MB
    opt = torch.optim.AdamW(model.parameters(), **HP)
    model, opt = trainstream.wrap(model, opt, device="cuda", offload=True)
    s = trainstream.wrapper.unwrap_session(model)
    first = sorted(s.pager.paged)
    peaks = []
    def between(k):
        peaks.append(torch.cuda.max_memory_allocated())
    losses = lm_steps(model, opt, tokens(14, batch=4, seq=64), between)
    opt.synchronize()
    print("@@" + json.dumps(dict(first=first, final=sorted(s.pager.paged), settled=s.planner.done, losses=losses,
                                 events=s.events, moments=opt.moment_dtype, cap=cap)))
""")


def test_auto_plan_under_a_vram_cap_finds_the_split_and_trains_correctly():
    cap = 450 << 20  # 8 blocks of 50 MB do not all fit next to the rest
    code = CAP_SCRIPT.replace("ROOT", repr(ROOT)).replace("HERE", repr(HERE)).replace("CAP", str(cap))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    res = json.loads([l for l in out.stdout.splitlines() if l.startswith("@@")][-1][2:])
    assert res["first"], "a 450 MB cap must start with some blocks paged"
    assert res["settled"]
    assert res["final"] != res["first"]  # measured peaks changed the plan
    assert 0 < len(res["final"]) < len(res["first"])  # measured headroom moved some blocks into VRAM
    assert any("memory plan settled" in e for e in res["events"])
    # Moving blocks between VRAM and RAM must not change the arithmetic at all: the
    # capped, re-planned run equals an uncapped, unpaged TrainStream run exactly.
    # (Against torch.optim.AdamW this lr is unstable enough to amplify last-bit
    # differences after ~8 steps, so that comparison lives in the smaller tests.)
    torch.manual_seed(0)
    ref = Stack(d=1024, n=8)
    ref, opt = trainstream.wrap(ref, torch.optim.AdamW(ref.parameters(), **HP), device=DEV, offload=True,
                                param_offload=False)
    try:
        a = lm_steps(ref, opt, tokens(14, batch=4, seq=64))
    finally:
        trainstream.unwrap(ref)
    assert res["losses"] == a


def test_watchdog_pages_a_block_when_a_later_step_grows():
    torch.manual_seed(0)
    model = Stack().to(DEV)
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  param_offload=[0], activation_offload=False)
    s = trainstream.wrapper.unwrap_session(model)
    from trainstream.planner import Budget, MemoryPlanner
    used = torch.cuda.memory_allocated()
    s.planner = MemoryPlanner(Budget(vram_total=used + (64 << 20), vram_usable=used + (64 << 20), ram_total=1 << 36,
                                     ram_available=1 << 35, pin_budget=1 << 34),
                              s.pager.block_bytes, fixed_vram=0, inflight_bytes=1 << 20)
    s.planner.done = True  # settled; now a step with a much larger batch arrives
    try:
        before = len(s.pager.paged)
        lm_steps(model, opt, tokens(1, batch=64, seq=128))
        assert len(s.pager.paged) == before + 1
        assert any("memory watchdog" in e for e in s.events)
    finally:
        trainstream.unwrap(model)
