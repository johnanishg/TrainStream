"""OffloadAdamW numerical parity with torch.optim.AdamW and state handling."""

import copy

import pytest
import torch
from torch import nn

import trainstream
from trainstream import OffloadAdamW

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"


def make_mlp(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 10)
    ).to(DEV, dtype)


def batches(n, seed=1, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [(torch.randn(32, 64, generator=g).to(DEV, dtype), torch.randint(0, 10, (32,), generator=g).to(DEV))
            for _ in range(n)]


def run(model, opt, data, accum=1, clip=None):
    for i, (x, y) in enumerate(data):
        loss = nn.functional.cross_entropy(model(x).float(), y) / accum
        loss.backward()
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


HP = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_parity_standalone(mode):
    ref, mine = make_mlp(), make_mlp()
    data = batches(8)
    run(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), data)
    run(mine, OffloadAdamW(mine.parameters(), mode=mode, **HP), data)
    assert max_diff(ref, mine) < 2e-5


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_parity_attached_async(mode):
    """With hooks attached step() is asynchronous; results must not change."""
    ref, mine = make_mlp(), make_mlp()
    data = batches(10)
    run(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), data)
    opt = OffloadAdamW(mine.parameters(), mode=mode, **HP).attach(mine)
    run(mine, opt, data)
    assert opt.attached
    assert max_diff(ref, mine) < 2e-5


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_parity_grad_accumulation_and_clipping(mode):
    ref, mine = make_mlp(), make_mlp()
    data = batches(12)
    run(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), data, accum=3, clip=0.5)
    opt = OffloadAdamW(mine.parameters(), mode=mode, max_grad_norm=0.5, **HP).attach(mine)
    run(mine, opt, data, accum=3)
    assert max_diff(ref, mine) < 5e-5


def test_param_groups_and_lr_scheduler():
    ref, mine = make_mlp(), make_mlp()

    def groups(m):
        decay = [p for p in m.parameters() if p.dim() > 1]
        nodecay = [p for p in m.parameters() if p.dim() <= 1]
        return [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}]

    opt_ref = torch.optim.AdamW(groups(ref), lr=1e-2, foreach=False)
    opt_mine = OffloadAdamW(groups(mine), lr=1e-2).attach(mine)
    sched_ref = torch.optim.lr_scheduler.StepLR(opt_ref, step_size=2, gamma=0.5)
    sched_mine = torch.optim.lr_scheduler.StepLR(opt_mine, step_size=2, gamma=0.5)
    for x, y in batches(6):
        for model, opt, sched in ((ref, opt_ref, sched_ref), (mine, opt_mine, sched_mine)):
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            opt.zero_grad()
            sched.step()
    opt_mine.synchronize()
    assert opt_mine.param_groups[0]["lr"] == pytest.approx(opt_ref.param_groups[0]["lr"])
    assert max_diff(ref, mine) < 2e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_low_precision_params_use_fp32_master(dtype, mode):
    """bf16/fp16 weights are updated from an fp32 master copy in host RAM."""
    ref32 = make_mlp()
    low = copy.deepcopy(ref32).to(dtype)
    with torch.no_grad():  # start the reference from the same rounded weights
        for p_ref, p_low in zip(ref32.parameters(), low.parameters()):
            p_ref.copy_(p_low.float())
    data = batches(6)
    opt_ref = torch.optim.AdamW(ref32.parameters(), foreach=False, **HP)
    opt = OffloadAdamW(low.parameters(), mode=mode, **HP).attach(low)
    for x, y in data:
        nn.functional.cross_entropy(ref32(x), y).backward()
        # Both optimizers consume the *same* rounded gradients, so only the
        # optimizer path differs (fp16 flushes tiny gradients to zero, and Adam
        # turns any such difference into a full lr-sized step).
        for p_low, p_ref in zip(low.parameters(), ref32.parameters()):
            p_low.grad = p_ref.grad.to(dtype)
            p_ref.grad = p_low.grad.float()
        opt_ref.step()
        opt.step()
        opt_ref.zero_grad()
    opt.synchronize()
    for p_low, p_ref in zip(low.parameters(), ref32.parameters()):
        master = opt.engine.state(opt._slot_of[id(p_low)])[0].view(p_low.shape).to(DEV)
        # The fp32 master follows the fp32 reference ...
        assert (master - p_ref).abs().max().item() < 2e-5
        # ... and the device weight is exactly the rounded master.
        assert torch.equal(p_low, master.to(dtype))


def test_cpu_mode_streams_gradients_out_of_vram():
    model = make_mlp()
    opt = OffloadAdamW(model.parameters(), mode="cpu", **HP).attach(model)
    x, y = batches(1)[0]
    nn.functional.cross_entropy(model(x), y).backward()
    assert all(p.grad is None for p in model.parameters()), "hooks should free VRAM gradients"
    opt.step()
    opt.synchronize()
    stats = opt.stats()
    nbytes = sum(p.numel() * 4 for p in model.parameters())
    assert stats["grad_bytes_offloaded"] == nbytes
    assert stats["param_bytes_uploaded"] == nbytes
    assert stats["grad_hook_calls"] == len(list(model.parameters()))


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_state_dict_roundtrip_and_torch_interop(mode):
    model = make_mlp()
    opt = OffloadAdamW(model.parameters(), mode=mode, **HP).attach(model)
    run(model, opt, batches(4))
    sd = opt.state_dict()
    assert len(sd["state"]) == len(list(model.parameters()))
    assert all(v["step"].item() == 4 for v in sd["state"].values())

    # Restore into a fresh OffloadAdamW and continue: identical to never stopping.
    clone = copy.deepcopy(model)
    opt2 = OffloadAdamW(clone.parameters(), mode=mode, **HP).attach(clone)
    opt2.load_state_dict(sd)
    more = batches(3, seed=9)
    run(model, opt, more)
    run(clone, opt2, more)
    assert max_diff(model, clone) == 0.0

    # The same dict drives a stock torch AdamW to the same result.
    ref = copy.deepcopy(clone)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    sd_t = opt2.state_dict()
    for v in sd_t["state"].values():
        v.pop("master", None)
        for k in ("exp_avg", "exp_avg_sq"):
            v[k] = v[k].to(DEV)
    opt_ref.load_state_dict(sd_t)
    final = batches(2, seed=11)
    run(clone, opt2, final)
    run(ref, opt_ref, final)
    assert max_diff(clone, ref) < 2e-5


def test_wrap_migrates_existing_torch_state():
    ref, mine = make_mlp(), make_mlp()
    warm = batches(3)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    opt_mine = torch.optim.AdamW(mine.parameters(), foreach=False, **HP)
    run(ref, opt_ref, warm)
    run(mine, opt_mine, warm)
    mine, ts_opt = trainstream.wrap(mine, opt_mine, offload=True, activation_offload=False, profile=False)
    try:
        assert len(opt_mine.state) == 0  # VRAM moments released
        more = batches(4, seed=5)
        run(ref, opt_ref, more)
        run(mine, ts_opt, more)
        assert max_diff(ref, mine) < 2e-5
    finally:
        trainstream.unwrap(mine)


def test_eager_updates_with_declared_accumulation():
    """cpu mode updates during the last backward of each accumulation window."""
    ref, mine = make_mlp(), make_mlp()
    data = batches(12)
    run(ref, torch.optim.AdamW(ref.parameters(), foreach=False, **HP), data, accum=3)
    opt = OffloadAdamW(mine.parameters(), mode="cpu", accumulation_steps=3, **HP).attach(mine)
    assert opt.eager_active
    run(mine, opt, data, accum=3)
    assert opt.stats()["eager_updates"] == 4 * len(list(mine.parameters()))
    assert max_diff(ref, mine) < 5e-5


def test_undeclared_accumulation_raises():
    model = make_mlp()
    opt = OffloadAdamW(model.parameters(), mode="cpu", **HP).attach(model)
    (x, y), (x2, y2) = batches(2)
    nn.functional.cross_entropy(model(x), y).backward()
    with pytest.raises(RuntimeError, match="accumulation_steps"):
        nn.functional.cross_entropy(model(x2), y2).backward()
    opt.synchronize()
