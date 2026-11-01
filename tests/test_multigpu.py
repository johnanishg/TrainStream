"""Data parallel on one machine through shared host memory.

Two ranks share one GPU here (as in test_distributed): what is tested is the
host side — one shared copy of the frozen weights, and the gradient
all-reduce through shared memory — which is the same for one GPU per rank.
"""

import os
import pathlib
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

WORLD = 2


# --------------------------------------------------------------------------- primitives
def _prims_worker(rank, store, out):
    from trainstream import _C, shared
    dist.init_process_group("gloo", init_method=pathlib.Path(store).as_uri(), rank=rank, world_size=WORLD)
    local = shared.LocalGroup(dist.new_group(backend="gloo"), WORLD, rank)
    segs = local.segments([("a", 4096)], lambda k, t: t[:4].copy_(torch.tensor([1, 2, 3, 4], dtype=torch.uint8)))
    seen = segs["a"][:4].tolist()
    red = shared.ShmAllReduce(local, capacity=1000)
    results = []
    for step in range(5):  # several rounds: both alternating areas get reused
        g = torch.Generator().manual_seed(100 * step + rank)
        buf = torch.randn(777, generator=g)
        mine = buf.clone()
        red(buf)
        results.append((mine, buf))
    torch.save({"seen": seen, "results": results, "pinned": segs["a"].is_pinned()}, os.path.join(out, f"p{rank}.pt"))
    dist.destroy_process_group()


def test_shared_segments_and_allreduce_across_processes():
    tmp = tempfile.mkdtemp()
    mp.spawn(_prims_worker, args=(os.path.join(tmp, "store"), tmp), nprocs=WORLD, join=True)
    r = [torch.load(os.path.join(tmp, f"p{k}.pt")) for k in range(WORLD)]
    assert r[0]["seen"] == r[1]["seen"] == [1, 2, 3, 4]  # rank 1 sees what rank 0 wrote
    for step in range(5):
        a, b = r[0]["results"][step], r[1]["results"][step]
        want = a[0] + b[0]  # rank order 0..N-1
        assert torch.equal(a[1], want) and torch.equal(b[1], want)  # bit-identical on every rank


# --------------------------------------------------------------------------- training
class Block(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.base = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)
        self.lora = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.lora.weight)

    def forward(self, x):
        return x + self.out(torch.relu(self.base(x))) + self.lora(x)


def make_model(seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(16, 128), Block(), Block(), Block(), nn.Linear(128, 8))
    for b in m[1:4]:
        b.base.weight.requires_grad_(False)
        b.out.weight.requires_grad_(False)
    return m


def _train_worker(rank, store, out, shm, quantize):
    os.environ["TRAINSTREAM_SHM"] = "1" if shm else "0"
    import trainstream
    dist.init_process_group("gloo", init_method=pathlib.Path(store).as_uri(), rank=rank, world_size=WORLD)
    model = make_model(seed=100 + rank).cuda()  # wrap() broadcasts rank 0's weights
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=2e-3), offload=True, activation_offload=False,
                                  param_offload=True, quantize=quantize, blocks=list(model[1:4]), profile=False)
    s = trainstream.wrapper.unwrap_session(model)
    g = torch.Generator().manual_seed(7)
    for _ in range(5):
        x, y = torch.randn(16, 16, generator=g), torch.randint(0, 8, (16,), generator=g)
        xs, ys = x.chunk(WORLD)[rank].cuda(), y.chunk(WORLD)[rank].cuda()
        nn.functional.cross_entropy(model(xs), ys).backward()
        opt.step()
        opt.zero_grad()
    opt.synchronize()
    info = dict(shared=s.pager.frozen_shared_bytes, shm_reduce=opt._dp["shm_reduce"] is not None,
                sd={k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save(info, os.path.join(out, f"rank{rank}.pt"))
    trainstream.unwrap(model)
    dist.destroy_process_group()


def run(shm, quantize):
    tmp = tempfile.mkdtemp()
    mp.spawn(_train_worker, args=(os.path.join(tmp, "store"), tmp, shm, quantize), nprocs=WORLD, join=True)
    return [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]


@pytest.mark.parametrize("quantize", ["none", "nf4"])
def test_shared_memory_data_parallel_equals_per_rank_copies(quantize):
    shm = run(True, quantize)
    sep = run(False, quantize)
    for r in range(WORLD):
        assert shm[r]["shm_reduce"] and not sep[r]["shm_reduce"]
        assert shm[r]["shared"] > 0 and sep[r]["shared"] == 0  # the frozen base is held once per machine
    for k, v in shm[0]["sd"].items():
        assert torch.equal(v, shm[1]["sd"][k]), k  # ranks stay identical
        assert torch.equal(v, sep[0]["sd"][k]), k  # and equal to per-rank copies + gloo
