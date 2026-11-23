"""The disk tier's read path: asynchronous unbuffered file reads into pinned memory, copied to
the GPU on a stream that waits for the read itself (csrc/aio.cpp, csrc/disk.cpp)."""

import copy
import gc
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import trainstream  # noqa: E402
from trainstream import _C  # noqa: E402
from trainstream import disk as D  # noqa: E402

from test_paging import HP, TinyLM, tokens, train  # noqa: E402

DEV = "cuda"


def stored(tmp_path, *shapes, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    store = D.DiskStore(str(tmp_path))
    values = [torch.randn(*s, generator=g).to(dtype) for s in shapes]
    return store, values, [store.store(v) for v in values]


# ------------------------------------------------------------------ the files
def test_backend_is_one_of_the_operating_systems_own():
    assert _C.disk_backend() in ("iocp", "io_uring", "threads")
    assert _C.disk_backend() == ("iocp" if os.name == "nt" else _C.disk_backend())


def test_store_pads_files_to_whole_sectors_and_knows_them(tmp_path):
    store, values, mapped = stored(tmp_path, (3, 5), (1000, 7))
    try:
        for v, m in zip(values, mapped):
            path, offset, nbytes = D.file_of(m)
            assert offset == 0 and nbytes == v.numel() * 2 and os.path.getsize(path) % D.ALIGN == 0
            assert torch.equal(m.view(v.shape), v)
            assert D.is_on_disk(m.view(v.shape)) and D.is_on_disk(torch.nn.Parameter(m.view(v.shape), False).data)
        assert D.file_of(mapped[1][10:]) is None  # a part of a file is not served by the read path
        assert not D.is_on_disk(torch.zeros(4)) and D.file_of(torch.zeros(4)) is None
    finally:
        del mapped
        gc.collect()
        store.close()
    assert not os.path.exists(store.dir)


@pytest.mark.parametrize("numel", [1, 4095, 4096, 4097, (9 << 20) + 13])
def test_unbuffered_read_returns_the_stored_bytes(tmp_path, numel):
    store, (v,), (m,) = stored(tmp_path, (numel,), dtype=torch.uint8)
    try:
        path, _, nbytes = D.file_of(m)
        out = torch.empty(numel, dtype=torch.uint8)
        _C.disk_read(out, path, 0)
        assert torch.equal(out, v.view(-1))
        if numel > 2 * 4096:  # from a sector boundary inside the file
            part = torch.empty(numel - 4096, dtype=torch.uint8)
            _C.disk_read(part, path, 4096)
            assert torch.equal(part, v.view(-1)[4096:])
        with pytest.raises(RuntimeError, match="holds only"):
            _C.disk_read(torch.empty(numel + 8192, dtype=torch.uint8), path, 0)
        with pytest.raises(RuntimeError, match="multiple of 4096"):
            _C.disk_read(out, path, 100)
    finally:
        del m
        gc.collect()
        store.close()


def test_a_write_through_the_mapping_reaches_the_unbuffered_reads(tmp_path):
    store, (v,), (m,) = stored(tmp_path, (300_000,), dtype=torch.float32)
    try:
        path = D.file_of(m)[0]
        m.mul_(-3.0)
        D.sync(m)
        out = torch.empty(300_000)
        _C.disk_read(out, path, 0)
        assert torch.equal(out, v * -3.0)
    finally:
        del m
        gc.collect()
        store.close()


# -------------------------------------------------------- stage, then copy
def test_stage_then_copy_to_the_gpu(tmp_path):
    shapes = [(257, 129), (64, 64), (5,), (1 << 20,)]
    store, values, mapped = stored(tmp_path, *shapes)
    try:
        files = [D.file_of(m) for m in mapped]
        before = _C.disk_stats()
        stage = _C.disk_stage([f[0] for f in files], [f[1] for f in files], [f[2] for f in files], 12345, 0, 3)
        assert not stage.hit and stage.bytes == sum(f[2] for f in files)
        dsts = [torch.empty(s, dtype=torch.bfloat16, device=DEV) for s in shapes]
        ev = _C.h2d_disk_async([d.view(-1) for d in dsts], stage, _C.record_event(0))
        ev.wait()  # the compute stream waits; this thread did not
        for d, v in zip(dsts, values):
            assert torch.equal(d.cpu(), v)
        assert stage.ready
        # the same key again: served from the slot, no new reads
        again = _C.disk_stage([f[0] for f in files], [f[1] for f in files], [f[2] for f in files], 12345, 0, 3)
        after = _C.disk_stats()
        assert again.hit and after["stage_hits"] == before["stage_hits"] + 1
        assert after["stages"] == before["stages"] + 1
        assert after["bytes_read"] - before["bytes_read"] >= sum(f[2] for f in files)
        assert after["copied_bytes"] - before["copied_bytes"] == sum(f[2] for f in files)
        # after an invalidation the bytes are read again
        _C.disk_invalidate(0)
        third = _C.disk_stage([f[0] for f in files], [f[1] for f in files], [f[2] for f in files], 12345, 0, 3)
        assert not third.hit
        third.wait()
    finally:
        torch.cuda.synchronize()
        _C.disk_release(0)
        del mapped
        gc.collect()
        store.close()


def test_slots_rotate_and_a_reused_slot_is_refused(tmp_path):
    store, values, mapped = stored(tmp_path, (40_000,), (40_000,), (40_000,), seed=3)
    try:
        files = [D.file_of(m) for m in mapped]
        stages = [_C.disk_stage([f[0]], [f[1]], [f[2]], 100 + k, 0, 2) for k, f in enumerate(files)]  # 3 keys, 2 slots
        dst = torch.empty(40_000, dtype=torch.bfloat16, device=DEV)
        with pytest.raises(RuntimeError, match="reused"):
            _C.h2d_disk_async([dst], stages[0], None)  # its slot now holds the third file
        for k in (1, 2):
            _C.h2d_disk_async([dst], stages[k], None).wait()
            assert torch.equal(dst.cpu(), values[k])
        with pytest.raises(RuntimeError, match="size mismatch"):
            _C.h2d_disk_async([dst[:10]], stages[2], None)
    finally:
        torch.cuda.synchronize()
        _C.disk_release(0)
        del mapped
        gc.collect()
        store.close()


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="opening"):
        _C.disk_stage([str(tmp_path / "nope.bin")], [0], [4096], -1, 0, 2)
    _C.disk_release(0)


# ------------------------------------------------------------- the pager
def frozen_blocks_model():
    torch.manual_seed(0)
    cpu_model = TinyLM()
    for b in cpu_model.blocks:
        for p in b.parameters():
            p.requires_grad_(False)
    return cpu_model


@pytest.mark.parametrize("lookahead", ["auto", 1, 3])
def test_training_from_disk_matches_training_from_vram(monkeypatch, tmp_path, lookahead):
    import trainstream.paging as paging
    monkeypatch.setattr(paging, "physical_ram_bytes", lambda: 1)  # pin budget ~0: frozen weights go to files
    cpu_model = frozen_blocks_model()
    ref = copy.deepcopy(cpu_model).to(DEV)
    trainable = lambda m: [p for p in m.parameters() if p.requires_grad]  # noqa: E731
    opt_ref = torch.optim.AdamW(trainable(ref), foreach=False, **HP)
    before = _C.disk_stats()
    mine, opt = trainstream.wrap(cpu_model, torch.optim.AdamW(trainable(cpu_model), **HP), device=DEV, offload=True,
                                 param_offload=True, activation_offload=False, disk_offload=str(tmp_path),
                                 disk_lookahead=lookahead)
    try:
        session = trainstream.wrapper.unwrap_session(mine)
        pager = session.pager
        assert pager.frozen_disk_bytes > 0 and pager.disk_ahead == (2 if lookahead == "auto" else max(1, lookahead))
        assert any("asynchronous unbuffered I/O" in e for e in session.events)
        data = tokens(5)
        assert train(mine, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
        torch.cuda.synchronize()
        rep = trainstream.memory_report(mine)
        disk = rep["disk"]
        assert disk["backend"] == _C.disk_backend() and disk["lookahead"] == pager.disk_ahead
        assert rep["params"]["disk_bytes"] > 0 and rep["params"]["bounce_bytes"] == 0  # nothing went through the mapping
        assert disk["copied_bytes"] - before["copied_bytes"] == rep["params"]["disk_bytes"]
        assert disk["bytes_read"] > before["bytes_read"]
        # blocks around the turn from forward to backward are read once and copied twice
        assert disk["stage_hits"] > before["stage_hits"]
        assert disk["bytes_read"] - before["bytes_read"] < disk["copied_bytes"] - before["copied_bytes"] + 16 * 4096 * 64
        # weights still read back whole (state_dict goes through the mapping)
        sd = mine.state_dict()
        for k, v in ref.state_dict().items():
            if "blocks" in k:
                assert torch.equal(sd[k].cpu(), v.cpu()), k
    finally:
        trainstream.unwrap(mine)
    assert _C.disk_stats()["slots"] == 0  # the staging memory was given back


def test_loading_new_frozen_values_reaches_the_next_step(monkeypatch, tmp_path):
    import trainstream.paging as paging
    monkeypatch.setattr(paging, "physical_ram_bytes", lambda: 1)
    cpu_model = frozen_blocks_model()
    trainable = lambda m: [p for p in m.parameters() if p.requires_grad]  # noqa: E731
    mine, opt = trainstream.wrap(cpu_model, torch.optim.AdamW(trainable(cpu_model), **HP), device=DEV, offload=True,
                                 param_offload=True, activation_offload=False, disk_offload=str(tmp_path))
    try:
        data = tokens(2)
        train(mine, opt, data[:1])
        sd = {k: (v * 0.5 if "blocks" in k else v) for k, v in mine.state_dict().items()}
        mine.load_state_dict(sd)  # written through the mappings, then flushed
        opt.synchronize()
        ref = TinyLM().to(DEV)
        ref.load_state_dict(sd)
        x = data[1]
        with torch.no_grad():
            assert torch.allclose(mine(x[:, :-1]), ref(x[:, :-1]), atol=1e-5)
    finally:
        trainstream.unwrap(mine)
