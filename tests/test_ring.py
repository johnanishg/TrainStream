"""Ring all-reduce over TCP (trainstream.ring): sums, bit-identical replicas, and data-parallel
training across three processes."""

import os
import pathlib
import tempfile
import threading

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from trainstream import ring as R  # noqa: E402


def reduce_all(rings, buffers):
    """Every member reduces its buffer on a thread of its own; returns the exceptions (None: fine)."""
    errors = [None] * len(rings)

    def work(i):
        try:
            rings[i](buffers[i])
        except Exception as exc:  # noqa: BLE001
            errors[i] = exc

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(rings))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


@pytest.mark.parametrize("world", [2, 3, 5])
def test_sum_is_the_same_bits_on_every_rank(world):
    rings = R.connect_local(world)
    g = torch.Generator().manual_seed(world)
    for n in (0, 1, world - 1, 1000, 3_000_017):
        data = [torch.randn(n, generator=g) * 10.0 ** (r % 3) for r in range(world)]
        want = torch.stack([d.double() for d in data]).sum(0)
        bufs = [d.clone() for d in data]
        assert reduce_all(rings, bufs) == [None] * world
        for b in bufs[1:]:
            assert torch.equal(b, bufs[0])  # replicas stay identical
        if n:
            assert (bufs[0].double() - want).abs().max().item() <= 1e-5 * want.abs().max().clamp_min(1).item()
    st = rings[0].stats()
    assert st["calls"] == 5 and st["world"] == world and st["rank"] == 0
    total = sum(4 * n for n in (0, 1, world - 1, 1000, 3_000_017))
    # the bandwidth-optimal schedule: each rank sends 2 (N - 1) / N of the data
    assert st["bytes_sent"] == pytest.approx(2 * (world - 1) / world * total, rel=1e-3, abs=64 * world)
    # segments differ by a value when the length is not a multiple of the ranks, so one rank's two
    # directions can differ by a few bytes; around the whole ring everything sent is received
    everyone = [r.stats() for r in rings]
    assert sum(s["bytes_sent"] for s in everyone) == sum(s["bytes_received"] for s in everyone)
    assert sum(s["bytes_sent"] for s in everyone) == 2 * (world - 1) * total


def test_many_reductions_in_a_row_stay_in_step():
    rings = R.connect_local(3)
    for i in range(50):
        bufs = [torch.full((257,), float(r + i)) for r in range(3)]
        assert reduce_all(rings, bufs) == [None] * 3
        assert all(torch.equal(b, torch.full((257,), float(3 * i + 3))) for b in bufs)


def test_ranks_that_disagree_fail_instead_of_summing_the_wrong_things():
    rings = R.connect_local(3, timeout=3)
    bufs = [torch.ones(1000), torch.ones(1000), torch.ones(999)]  # rank 2 reduces a different tensor
    errors = reduce_all(rings, bufs)
    assert all(e is not None for e in errors)
    assert any("disagree" in str(e) for e in errors)


def test_only_float32_host_tensors():
    rings = R.connect_local(2)
    with pytest.raises(RuntimeError, match="float32"):
        rings[0](torch.ones(4, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="float32"):
        rings[0](torch.ones(4, device="cuda"))


def test_local_address_choices(monkeypatch):
    monkeypatch.delenv("TRAINSTREAM_RING_HOST", raising=False)
    assert R.local_address(True) == "127.0.0.1"
    monkeypatch.setenv("TRAINSTREAM_RING_HOST", "127.0.0.1")
    assert R.local_address(False) == "127.0.0.1"


# ------------------------------------------------------------ data parallel, three processes
HP = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)
WORLD = 3
STEPS = 5


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8))


def full_batches():
    g = torch.Generator().manual_seed(7)
    return [(torch.randn(18, 32, generator=g), torch.randint(0, 8, (18,), generator=g)) for _ in range(STEPS)]


def _worker(rank, store, out, backend, clip):
    import trainstream
    dist.init_process_group("gloo", init_method=pathlib.Path(store).as_uri(), rank=rank, world_size=WORLD)
    model = make_model(seed=100 + rank).cuda()  # different init per rank: wrap() broadcasts rank 0's
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False, max_grad_norm=clip, distributed_backend=backend)
    for x, y in full_batches():
        xs, ys = x.chunk(WORLD)[rank].cuda(), y.chunk(WORLD)[rank].cuda()
        F.cross_entropy(model(xs), ys).backward()
        opt.step()
        opt.zero_grad()
    opt.synchronize()
    stats = opt.stats()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"sd": sd, "backend": stats["distributed_backend"], "ring": stats.get("ring"),
                "config": model._trainstream_session.config["distributed_backend"]},
               os.path.join(out, f"rank{rank}.pt"))
    trainstream.unwrap(model)
    dist.destroy_process_group()


def reference(clip):
    model = make_model(seed=100).cuda()
    opt = torch.optim.AdamW(model.parameters(), foreach=False, **HP)
    for x, y in full_batches():
        F.cross_entropy(model(x.cuda()), y.cuda()).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("clip", [None, 0.5], ids=["plain", "clipped"])
def test_three_ranks_train_over_the_ring(clip):
    tmp = tempfile.mkdtemp()
    mp.spawn(_worker, args=(os.path.join(tmp, "store"), tmp, "ring", clip), nprocs=WORLD, join=True)
    ranks = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]
    assert all(r["backend"] == "ring" and r["config"] == "ring" for r in ranks)
    assert all(r["ring"]["calls"] == STEPS * 6 for r in ranks)  # six tensors a step, each around the ring
    want = reference(clip)
    for k in want:
        for r in ranks[1:]:
            assert torch.equal(r["sd"][k], ranks[0]["sd"][k]), k  # three replicas, the same bits
        assert (ranks[0]["sd"][k] - want[k]).abs().max().item() < 1e-4, k


def test_auto_keeps_shared_memory_on_one_machine_and_gloo_stays_available():
    for backend, expect in (("auto", "shm"), ("gloo", "shm")):
        tmp = tempfile.mkdtemp()
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), tmp, backend, None), nprocs=WORLD, join=True)
        ranks = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]
        assert all(r["backend"] == expect and r["ring"] is None for r in ranks)
