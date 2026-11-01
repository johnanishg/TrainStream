"""Pinned host arena (page-locked slabs carved into blocks) and the pageable
bounce copy. The arena is process-wide, so tests compare deltas."""

import copy

import pytest
import torch
from torch import nn

import trainstream
from trainstream import _C

from test_quant import HP, make, tokens, train, trainable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MiB = 1 << 20


def ensure_arena(nbytes=64 * MiB):
    """At least `nbytes` of the arena free (in one new slab if need be), and
    every earlier release reclaimed, so the counters move only with the test."""
    torch.cuda.synchronize()
    _C.pinned_stats()  # reclaims every block whose fences have passed
    st = _C.pinned_stats()
    want = st["arena_used"] + nbytes
    if st["arena_largest_free"] < nbytes:
        want = st["arena_bytes"] + nbytes
    _C.pinned_reserve(want, 0, max(st["arena_cap"], want))
    assert _C.pinned_stats()["arena_largest_free"] >= nbytes


def test_blocks_come_from_the_arena_and_return_to_it():
    ensure_arena()
    before = _C.pinned_stats()
    a = _C.pinned_empty([MiB], torch.float32, "test.arena", False, False)  # 4 MiB, persistent
    b = _C.pinned_empty([3 * 1024], torch.float32, "test.arena", False, False)  # 12 KiB
    mid = _C.pinned_stats()
    assert a.is_pinned() and b.is_pinned()
    assert mid["arena_used"] >= before["arena_used"] + 4 * MiB + 12 * 1024
    assert mid["host_alloc_calls"] == before["host_alloc_calls"]  # no cudaHostAlloc per block
    assert sum(1 for _, op, _, tag in _C.pinned_log(8) if op == "arena" and tag == "test.arena") == 2
    a.fill_(1.0)
    b.fill_(2.0)
    del a, b
    torch.cuda.synchronize()
    after = _C.pinned_stats()  # reclaims fenced blocks
    assert after["arena_used"] == before["arena_used"]
    assert after["arena_largest_free"] >= before["arena_largest_free"]  # neighbours coalesced again


def test_cached_blocks_still_hit_the_size_cache_on_top_of_the_arena():
    ensure_arena()
    a = _C.pinned_empty([256 * 1024], torch.float32, "test.arena.cache", False, True)
    before = _C.pinned_stats()["cache_hits"]
    del a
    torch.cuda.synchronize()
    b = _C.pinned_empty([256 * 1024], torch.float32, "test.arena.cache", False, True)
    assert _C.pinned_stats()["cache_hits"] == before + 1
    del b


def test_a_released_range_is_not_reused_while_a_copy_still_reads_it():
    ensure_arena()
    n = 4 * MiB // 4
    src = _C.pinned_empty([n], torch.float32, "test.arena.fence", False, False)
    assert _C.pinned_log(1)[0][1] == "arena"  # an arena range, not its own cudaHostAlloc
    src.fill_(7.0)
    dst = torch.empty(n, device="cuda")
    torch.cuda._sleep(200_000_000)  # the copy below waits for this on the GPU
    _C.h2d_async(dst, src, _C.record_event(0))
    ptr = src.data_ptr()
    del src  # released while the H2D copy has not started
    other = _C.pinned_empty([n], torch.float32, "test.arena.fence", False, False)
    assert other.data_ptr() != ptr  # the fence is still pending: a different range
    other.fill_(-1.0)
    torch.cuda.synchronize()
    assert torch.equal(dst, torch.full_like(dst, 7.0))
    del other


def test_reserve_grows_by_slabs_and_trim_gives_empty_ones_back():
    st = _C.pinned_stats()
    got = trainstream.reserve_pinned(st["arena_bytes"] + 32 * MiB, cap=st["arena_cap"] + 32 * MiB)
    mid = _C.pinned_stats()
    assert got == mid["arena_bytes"] >= st["arena_bytes"] + 32 * MiB
    assert mid["arena_slabs"] > st["arena_slabs"]
    assert any(op == "slab" for _, op, _, _ in _C.pinned_log(64))
    _C.pinned_arena_trim()
    assert _C.pinned_stats()["arena_slabs"] <= mid["arena_slabs"]


@pytest.mark.parametrize("sizes", [[1000], [3 * MiB + 17, 5, 9 * MiB + 3]])
def test_bounce_copy_equals_a_direct_copy(sizes):
    g = torch.Generator().manual_seed(0)
    for rnd in range(5):  # more rounds than ring slots: slots are reused
        srcs = [torch.randn(n, generator=g).to(torch.bfloat16) for n in sizes]  # pageable
        dsts = [torch.empty(n, dtype=torch.bfloat16, device="cuda") for n in sizes]
        ev = _C.h2d_bounce_async(dsts, srcs, _C.record_event(0), 2)
        ev.wait()
        for d, s in zip(dsts, srcs):
            assert torch.equal(d.cpu(), s)


def test_pages_copy_back_to_back():
    pages = [_C.pinned_empty([4, 3, 8], torch.bfloat16, "test.pages", False, True) for _ in range(3)]
    for k, p in enumerate(pages):
        p.copy_(torch.full((4, 3, 8), float(k)))
    dst = torch.zeros(16, 3, 8, dtype=torch.bfloat16, device="cuda")
    _C.h2d_pages_async(dst[:12], pages, _C.record_event(0)).wait()
    want = torch.cat([p.cpu() for p in pages] + [torch.zeros(4, 3, 8, dtype=torch.bfloat16)])
    assert torch.equal(dst.cpu(), want)


def test_wrap_reserves_the_arena_up_front():
    m = make()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, param_offload=True,
                              activation_offload=False, blocks=list(m.blocks), host_arena=32 * MiB, profile=False)
    s = trainstream.wrapper.unwrap_session(m)
    assert s.config["host_arena"]["cap"] == 32 * MiB and s.config["host_arena"]["reserved"] >= 32 * MiB
    assert any(e.startswith("pinned host arena") for e in s.events)
    trainstream.unwrap(m)
    m2 = make()
    m2, opt2 = trainstream.wrap(m2, torch.optim.AdamW(trainable(m2), **HP), offload=True, param_offload=True,
                                activation_offload=False, blocks=list(m2.blocks), host_arena=False, profile=False)
    assert trainstream.wrapper.unwrap_session(m2).config["host_arena"] == {"reserved": 0, "cap": 0}
    trainstream.unwrap(m2)
    with pytest.raises(ValueError):
        trainstream.wrap(make(), host_arena="lots")


def test_frozen_weights_over_the_pin_budget_stream_through_the_bounce_slots(monkeypatch):
    data = tokens(4)
    base = make()
    runs = {}
    for name in ("pinned", "pageable"):
        if name == "pageable":
            monkeypatch.setattr(trainstream.paging, "physical_ram_bytes", lambda: 1)  # page-lock budget ~0
        m = copy.deepcopy(base)
        m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, param_offload=True,
                                  activation_offload=False, blocks=list(m.blocks), profile=False)
        s = trainstream.wrapper.unwrap_session(m)
        losses = train(m, opt, data)
        runs[name] = (losses, s.pager.frozen_pageable_bytes, s.pager.stats["bounce_bytes"])
        trainstream.unwrap(m)
    assert runs["pinned"][1] == 0 and runs["pinned"][2] == 0
    assert runs["pageable"][1] > 0 and runs["pageable"][2] > 0
    assert runs["pinned"][0] == runs["pageable"][0]  # bit-identical losses
