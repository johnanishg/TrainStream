"""Data-parallel training: N processes, gradients averaged on the host.

Two ranks share one GPU here (gloo needs no second card). Each rank trains on
its own half of every batch; the result must equal one process training on the
whole batch with torch.optim.AdamW.
"""

import copy
import os
import pathlib
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HP = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)
WORLD = 2
STEPS = 6


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8))


def full_batches(accum):
    g = torch.Generator().manual_seed(7)
    return [[(torch.randn(16, 32, generator=g), torch.randint(0, 8, (16,), generator=g)) for _ in range(accum)]
            for _ in range(STEPS)]


def _worker(rank, store, out, clip, accum, param_offload):
    import trainstream
    dist.init_process_group("gloo", init_method=pathlib.Path(store).as_uri(), rank=rank, world_size=WORLD)
    model = make_model(seed=100 + rank).cuda()  # different init per rank: wrap() broadcasts rank 0's
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False, param_offload=param_offload, max_grad_norm=clip,
                                  accumulation_steps=accum, blocks=[model[0], model[2]] if param_offload else None)
    assert opt._dp is not None  # detected from the initialized process group
    for micro in full_batches(accum):
        for x, y in micro:
            xs, ys = x.chunk(WORLD)[rank].cuda(), y.chunk(WORLD)[rank].cuda()
            (F.cross_entropy(model(xs), ys) / accum).backward()
        opt.step()
        opt.zero_grad()
    opt.synchronize()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, os.path.join(out, f"rank{rank}.pt"))
    trainstream.unwrap(model)
    dist.destroy_process_group()


def reference(clip, accum):
    model = make_model(seed=100).cuda()
    opt = torch.optim.AdamW(model.parameters(), foreach=False, **HP)
    for micro in full_batches(accum):
        for x, y in micro:
            (F.cross_entropy(model(x.cuda()), y.cuda()) / accum).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("clip,accum,param_offload", [(None, 1, False), (0.5, 1, False), (None, 2, False),
                                                      (None, 1, True)])
def test_data_parallel_matches_single_process(clip, accum, param_offload):
    tmp = tempfile.mkdtemp()
    store = os.path.join(tmp, "store")
    mp.spawn(_worker, args=(store, tmp, clip, accum, param_offload), nprocs=WORLD, join=True)
    ranks = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]
    ref = reference(clip, accum)
    for k in ref:
        assert torch.equal(ranks[0][k], ranks[1][k]), k  # replicas stay identical
        assert (ranks[0][k] - ref[k]).abs().max().item() < 1e-4, k
