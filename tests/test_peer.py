"""Peer GPUs: ownership, topology probe, routing of pooled frozen weights.

One GPU is enough for most of it: the routing test pools the weights in this
process's own VRAM (a same-device peer copy), and the two-process test maps
the other rank's pool through CUDA IPC (it works on Windows 11 and WSL2 too),
skipping where it is unavailable with the reason the probe gives. Copies between two real
GPUs need a machine with two of them.
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

import trainstream
from trainstream import _C, peer

from test_quant import HP, make, tokens, train, trainable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# --------------------------------------------------------------------------- pure logic
def test_assignment_is_round_robin_within_the_budget():
    assert peer.assign([0, 1, 2, 3, 4, 5], {i: 10 for i in range(6)}, 2, 25) == {0: 0, 1: 1, 2: 0, 3: 1}
    assert peer.assign([0, 1, 2], {0: 30, 1: 10, 2: 10}, 2, 20) == {1: 0, 2: 1}  # block 0 fits nowhere
    assert peer.assign([4, 7], {4: 5, 7: 5}, 3, 0) == {}
    owners = peer.assign(list(range(9)), {i: 1 for i in range(9)}, 3, 100)
    assert sorted(owners.values()).count(0) == 3  # spread evenly


def test_topology_probe_shape():
    rows = _C.peer_topology(0)
    n = torch.cuda.device_count()
    assert len(rows) == n * (n - 1)
    for r in rows:
        assert {"src", "dst", "can_access", "performance_rank", "native_atomics", "gbps"} <= set(r)
    assert _C.pci_bus_id(0)


# --------------------------------------------------------------------------- routing (one process)
def pool_in_this_process(pager):
    """A PeerPool whose 'peer' is this GPU: every remote frozen weight copied
    into one VRAM buffer, addressed exactly as an IPC-mapped pool would be."""
    parts = [(pid, part, t) for i in sorted(pager.remote) for pid, part, t in peer._frozen_parts(pager, i)]
    total = sum(peer._aligned(peer._nbytes(t)) for _, _, t in parts)
    buf = torch.empty(total, dtype=torch.uint8, device="cuda")
    pool = peer.PeerPool()
    off = 0
    for pid, part, t in parts:
        buf[off:off + peer._nbytes(t)].copy_(t.reshape(-1).view(torch.uint8))
        pool.where[(pid, part)] = (buf.data_ptr(), off, 0)
        off += peer._aligned(peer._nbytes(t))
    pool.pool = buf
    pool.owners = {i: 0 for i in pager.remote}
    torch.cuda.synchronize()
    return pool


@pytest.mark.parametrize("quantize", ["none", "nf4", "nvfp4"])
def test_pooled_weights_train_exactly_like_host_streamed_ones(quantize):
    if quantize == "nvfp4":
        from trainstream import lowp
        if not lowp.native_supported("nvfp4", "cuda"):
            pytest.skip("no nvfp4 tensor cores")
    data = tokens(4)
    base = make()
    runs = {}
    for mode in ("host", "peer"):
        m = copy.deepcopy(base)
        m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, param_offload=True,
                                  activation_offload=False, quantize=quantize, blocks=list(m.blocks), profile=False)
        s = trainstream.wrapper.unwrap_session(m)
        if mode == "peer":
            s.pager.peer = pool_in_this_process(s.pager)
        losses = train(m, opt, data)
        runs[mode] = (losses, s.pager.stats["peer_bytes"], s.pager.stats["load_bytes"])
        s.pager.peer = None
        trainstream.unwrap(m)
    assert runs["host"][1] == 0 and runs["peer"][1] > 0
    assert runs["peer"][2] < runs["host"][2]  # frozen weights no longer cross the host link
    assert runs["host"][0] == runs["peer"][0]  # bit-identical


# --------------------------------------------------------------------------- two processes, CUDA IPC
WORLD = 2


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


def _worker(rank, store, out, use_peer):
    os.environ["LOCAL_WORLD_SIZE"] = str(WORLD)
    dist.init_process_group("gloo", init_method=pathlib.Path(store).as_uri(), rank=rank, world_size=WORLD)
    model = make_model(seed=100 + rank).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=2e-3), offload=True, activation_offload=False,
                                  param_offload=True, blocks=list(model[1:4]), profile=False,
                                  peer=(1 << 30) if use_peer else False)
    s = trainstream.wrapper.unwrap_session(model)
    g = torch.Generator().manual_seed(7)
    for _ in range(trainstream.wrapper.PEER_SETUP_STEP + 4):
        x, y = torch.randn(16, 16, generator=g), torch.randint(0, 8, (16,), generator=g)
        xs, ys = x.chunk(WORLD)[rank].cuda(), y.chunk(WORLD)[rank].cuda()
        nn.functional.cross_entropy(model(xs), ys).backward()
        opt.step()
        opt.zero_grad()
    opt.synchronize()
    info = dict(events=[e for e in s.events if e.startswith("peer GPUs")],
                peer_bytes=s.pager.stats["peer_bytes"], pooled=s.peer.pooled_blocks if s.peer else [],
                sd={k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save(info, os.path.join(out, f"rank{rank}.pt"))
    trainstream.unwrap(model)  # collective while pooling is on
    dist.destroy_process_group()


def run(use_peer):
    tmp = tempfile.mkdtemp()
    mp.spawn(_worker, args=(os.path.join(tmp, "store"), tmp, use_peer), nprocs=WORLD, join=True)
    return [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]


def test_two_ranks_pool_frozen_blocks_through_ipc():
    pooled = run(True)
    ev = pooled[0]["events"]
    assert ev, "the peer setup did not run"
    if "off" in ev[0]:
        assert ev == pooled[1]["events"]  # both ranks reached the same decision
        pytest.skip(ev[0])
    plain = run(False)
    for r in range(WORLD):
        assert pooled[r]["pooled"] == [0, 1, 2] and pooled[r]["peer_bytes"] > 0
        for k, v in pooled[r]["sd"].items():
            assert torch.equal(v, plain[r]["sd"][k]), k  # the same bytes, from another rank's pool
