"""Non-finite guard: a step whose gradients hold a NaN / Inf is skipped as a whole.

After a skipped step the weights, the moments and the step counts are exactly
what they were, and the run continues as if that batch had never been drawn:
the reference in every test is torch.optim.AdamW trained on the same batches
minus the poisoned one.
"""

import os
import pathlib
import tempfile
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import trainstream  # noqa: E402
from trainstream import OffloadAdamW, _C, fused  # noqa: E402
from trainstream import optimizer as ts_optimizer  # noqa: E402

DEV = "cuda"
HP = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def make_mlp(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 10)).to(DEV, dtype)


def batches(n, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [(torch.randn(32, 64, generator=g).to(DEV), torch.randint(0, 10, (32,), generator=g).to(DEV))
            for _ in range(n)]


def poison(param, value):
    """One element of `param`'s gradient becomes `value` while the handle lives."""
    def hook(g):
        g = g.clone()
        g.view(-1)[3] = value
        return g
    return param.register_hook(hook)


def train(model, opt, data, bad=(), value=float("nan"), accum=1, clip=None):
    """`bad`: indices of batches whose gradient gets one non-finite element."""
    for i, (x, y) in enumerate(data):
        handle = poison(model[2].weight, value) if i in bad else None
        (F.cross_entropy(model(x).float(), y) / accum).backward()
        if handle is not None:
            handle.remove()
        if (i + 1) % accum == 0:
            if clip is not None and not isinstance(opt, OffloadAdamW):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
    if isinstance(opt, OffloadAdamW):
        opt.synchronize()
    torch.cuda.synchronize()


def max_diff(a, b):
    return max((pa.float() - pb.float()).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))


def snapshot(model, opt):
    opt.synchronize()
    sd = opt.state_dict()
    return ([p.detach().clone() for p in model.parameters()],
            {k: {n: (t.clone() if torch.is_tensor(t) else t) for n, t in e.items()} for k, e in sd["state"].items()})


def assert_unchanged(model, opt, snap):
    params, state = snap
    opt.synchronize()
    for p, q in zip(model.parameters(), params):
        assert torch.equal(p.detach(), q)
    now = opt.state_dict()["state"]
    assert now.keys() == state.keys()
    for k, entry in state.items():
        for name, t in entry.items():
            assert torch.equal(now[k][name], t), (k, name)


# ------------------------------------------------------------------ the optimizer
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
@pytest.mark.parametrize("attached", [False, True], ids=["standalone", "attached"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_bad_step_is_skipped_and_training_continues(mode, attached, value):
    data = batches(9)
    ref, mine = make_mlp(), make_mlp()
    train(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), [b for i, b in enumerate(data) if i != 4])

    opt = OffloadAdamW(mine.parameters(), mode=mode, **HP)
    if attached:
        opt.attach(mine)
    assert opt.nan_guard and not opt.eager_active
    train(mine, opt, data[:4])
    snap = snapshot(mine, opt)
    with pytest.warns(RuntimeWarning, match="skipped 1 optimizer step"):
        train(mine, opt, data[4:5], bad={0}, value=value)
    assert opt.skipped_steps == 1
    assert_unchanged(mine, opt, snap)  # weights, moments, step counts: bit for bit

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no further skips
        train(mine, opt, data[5:])
    assert opt.skipped_steps == 1
    assert max_diff(ref, mine) < 2e-5
    assert all(torch.isfinite(p).all() for p in mine.parameters())


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_bad_micro_batch_drops_the_whole_accumulated_step(mode):
    data = batches(12)
    ref, mine = make_mlp(), make_mlp()
    # step 1 (batches 3..5) is poisoned in its first micro-batch: the reference never sees those three
    train(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), data[:3] + data[6:], accum=3, clip=0.5)
    opt = OffloadAdamW(mine.parameters(), mode=mode, max_grad_norm=0.5, accumulation_steps=3, **HP).attach(mine)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        train(mine, opt, data, bad={3}, accum=3)
    assert opt.skipped_steps == 1
    assert max_diff(ref, mine) < 5e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_params_and_bf16_moments(dtype):
    data = [(x.to(dtype), y) for x, y in batches(6)]
    mine = make_mlp(dtype=dtype)
    opt = OffloadAdamW(mine.parameters(), mode="cpu", moment_dtype="bfloat16", **HP).attach(mine)
    train(mine, opt, data[:3])
    snap = snapshot(mine, opt)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        train(mine, opt, data[3:4], bad={0})
    assert_unchanged(mine, opt, snap)
    train(mine, opt, data[4:])
    assert all(torch.isfinite(p.float()).all() for p in mine.parameters())


def test_int8_moments_are_left_alone_by_a_skipped_step():
    data = batches(6)
    mine = make_mlp()
    opt = OffloadAdamW(mine.parameters(), mode="cpu", moment_dtype="int8", **HP).attach(mine)
    train(mine, opt, data[:3])
    snap = snapshot(mine, opt)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        train(mine, opt, data[3:4], bad={0})
    assert_unchanged(mine, opt, snap)


def test_guard_off_lets_the_nan_through():
    """What the guard prevents: without it one NaN reaches the weights."""
    mine = make_mlp()
    opt = OffloadAdamW(mine.parameters(), mode="cpu", nan_guard=False, **HP).attach(mine)
    assert opt.eager_active  # no clipping, no guard: updates run during backward
    train(mine, opt, batches(2), bad={1})
    assert opt.skipped_steps == 0
    assert not torch.isfinite(mine[2].weight).all()


def test_eager_request_is_ignored_while_guarded():
    model = make_mlp()
    with pytest.warns(RuntimeWarning, match="eager=True has no effect"):
        opt = OffloadAdamW(model.parameters(), mode="cpu", eager=True, **HP).attach(model)
    assert not opt.eager_active
    opt.detach()


def test_resident_optimizer_skips_too():
    data = batches(7)
    ref, mine = make_mlp(), make_mlp()
    train(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), [b for i, b in enumerate(data) if i != 3])
    opt = OffloadAdamW(mine.parameters(), mode="cpu", **HP).attach(mine)
    train(mine, opt, data[:2])
    opt._go_resident()
    assert opt.resident
    with pytest.warns(RuntimeWarning, match="non-finite"):
        train(mine, opt, data[2:], bad={1})
    assert opt.skipped_steps == 1
    assert max_diff(ref, mine) < 2e-5


def test_a_run_that_only_skips_stops(monkeypatch):
    monkeypatch.setattr(ts_optimizer, "NAN_GUARD_LIMIT", 3)
    model = make_mlp()
    opt = OffloadAdamW(model.parameters(), mode="cpu", **HP)
    data = batches(3)
    with pytest.warns(RuntimeWarning):
        train(model, opt, data[:2], bad={0, 1})
        with pytest.raises(FloatingPointError, match="3 optimizer steps in a row"):
            train(model, opt, data[2:], bad={0})
    assert opt.skipped_steps == 3
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_set_nan_guard_switches_it():
    model = make_mlp()
    opt = OffloadAdamW(model.parameters(), mode="cpu", **HP)
    opt.set_nan_guard(False)
    train(model, opt, batches(1), bad={0})
    assert opt.skipped_steps == 0 and not torch.isfinite(model[2].weight).all()


# ----------------------------------------------------------------- kernel flags
def flags():
    torch.cuda.synchronize()
    return _C.guard_read(0, True)


def test_kernels_flag_where_a_bad_value_appeared():
    _C.guard_enable(True)
    flags()  # clear
    x = torch.randn(4, 8, 512, device=DEV, dtype=torch.bfloat16)
    w = torch.ones(512, device=DEV, dtype=torch.bfloat16)
    fused.RMSNormFn.apply(x, w, 1e-6, False)
    g, u = torch.randn(2, 4096, device=DEV, dtype=torch.bfloat16), torch.randn(2, 4096, device=DEV, dtype=torch.bfloat16)
    fused.GatedActFn.apply(g, u, 0)
    assert not any(flags().values())  # finite inputs: nothing flagged

    bad = x.clone()
    bad[1, 2, 7] = float("inf")
    y = fused.RMSNormFn.apply(bad, w, 1e-6, False)
    got = flags()
    assert got["rms_norm_forward"] and not got["gated_act_forward"]
    assert not torch.isfinite(y[1, 2].float()).all() and torch.isfinite(y[0].float()).all()  # only that row

    u2 = u.clone()
    u2[0, 5] = float("nan")
    fused.GatedActFn.apply(g, u2, 0)
    got = flags()
    assert got["gated_act_forward"] and not got["rms_norm_forward"]

    xg = x.clone().requires_grad_()
    out = fused.RMSNormFn.apply(xg, w, 1e-6, False)
    dy = torch.randn_like(out)
    dy[0, 0, 0] = float("nan")
    out.backward(dy)
    assert flags()["rms_norm_backward"]

    gg, ug = g.clone().requires_grad_(), u.clone().requires_grad_()
    h = fused.GatedActFn.apply(gg, ug, 0)
    dh = torch.randn_like(h)
    dh[1, 9] = float("inf")
    h.backward(dh)
    assert flags()["gated_act_backward"]


def test_kernel_checks_can_be_switched_off():
    flags()
    _C.guard_enable(False)
    try:
        x = torch.full((2, 256), float("nan"), device=DEV)
        fused.RMSNormFn.apply(x, torch.ones(256, device=DEV), 1e-6, False)
        assert not any(flags().values())
    finally:
        _C.guard_enable(True)


# ------------------------------------------------------------------------ wrap()
def test_wrap_reports_the_guard():
    model = make_mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False)
    try:
        assert opt.nan_guard and model._trainstream_session.config["nan_guard"] is True
        assert model._trainstream_session.config["eager_updates"] is False
        assert any(e.startswith("non-finite guard:") and "nan_guard=False" in e
                   for e in model._trainstream_session.events)
        with pytest.warns(RuntimeWarning, match="non-finite"):
            train(model, opt, batches(3), bad={1})
        rep = trainstream.memory_report(model)["guard"]
        assert rep["enabled"] and rep["skipped_steps"] == 1 and "rms_norm_forward" in rep["kernel_flags"]
    finally:
        trainstream.unwrap(model)

    model = make_mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False, nan_guard=False)
    try:
        assert not opt.nan_guard and opt.eager_active
        assert not any(e.startswith("non-finite guard:") for e in model._trainstream_session.events)
        assert trainstream.memory_report(model)["guard"]["enabled"] is False
    finally:
        trainstream.unwrap(model)
        _C.guard_enable(True)


# ----------------------------------------------------------------- data parallel
WORLD = 2
DP_STEPS = 6
DP_BAD = 2  # the step poisoned on rank 1 only


def dp_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8))


def dp_batches():
    g = torch.Generator().manual_seed(7)
    return [(torch.randn(16, 32, generator=g), torch.randint(0, 8, (16,), generator=g)) for _ in range(DP_STEPS)]


def _dp_worker(rank, store, out, clip):
    import trainstream
    dist.init_process_group("gloo", init_method=pathlib.Path(store).as_uri(), rank=rank, world_size=WORLD)
    model = dp_model(seed=100 + rank).cuda()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95),
                                                           weight_decay=0.05),
                                  offload=True, activation_offload=False, max_grad_norm=clip)
    assert opt._dp is not None and opt.nan_guard
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, (x, y) in enumerate(dp_batches()):
            xs, ys = x.chunk(WORLD)[rank].cuda(), y.chunk(WORLD)[rank].cuda()
            handle = poison(model[2].weight, float("nan")) if (i == DP_BAD and rank == 1) else None
            F.cross_entropy(model(xs), ys).backward()
            if handle is not None:
                handle.remove()
            opt.step()
            opt.zero_grad()
        opt.synchronize()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"sd": sd, "skipped": opt.skipped_steps}, os.path.join(out, f"rank{rank}.pt"))
    trainstream.unwrap(model)
    dist.destroy_process_group()


@pytest.mark.parametrize("clip", [None, 0.5])
def test_data_parallel_ranks_skip_the_same_step(clip):
    """Only rank 1's gradient is bad; the verdict comes from the averaged
    gradients, so both ranks drop the step and stay identical."""
    tmp = tempfile.mkdtemp()
    mp.spawn(_dp_worker, args=(os.path.join(tmp, "store"), tmp, clip), nprocs=WORLD, join=True)
    ranks = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]
    assert [r["skipped"] for r in ranks] == [1, 1]

    ref = dp_model(seed=100).cuda()
    opt = torch.optim.AdamW(ref.parameters(), foreach=False, lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)
    for i, (x, y) in enumerate(dp_batches()):
        if i == DP_BAD:
            continue
        F.cross_entropy(ref(x.cuda()), y.cuda()).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(ref.parameters(), clip)
        opt.step()
        opt.zero_grad()
    want = {k: v.detach().cpu() for k, v in ref.state_dict().items()}
    for k in want:
        assert torch.equal(ranks[0]["sd"][k], ranks[1]["sd"][k]), k
        assert (ranks[0]["sd"][k] - want[k]).abs().max().item() < 1e-4, k
