"""fp8_gradients: gradients cross PCIe as one e4m3 byte per value, rounded stochastically."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import trainstream  # noqa: E402
from trainstream import OffloadAdamW, _C  # noqa: E402

DEV = "cuda"
HP = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def roundtrip(x, seed=0):
    codes, scales = _C.grad_fp8_encode(x, seed)
    codes, scales = codes.cpu(), scales.cpu()
    return _C.grad_fp8_decode(codes, scales).view(x.shape), codes, scales


def per_value(scales, n):
    return scales.repeat_interleave(256)[:n]


# ------------------------------------------------------------------- the code
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_codes_are_e4m3_against_one_scale_per_256_values(dtype):
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1000, generator=g) * torch.logspace(-3, 2, 1000)).to(dtype).to(DEV)
    y, codes, scales = roundtrip(x)
    assert codes.dtype == torch.uint8 and codes.numel() == 1000 and scales.dtype == torch.float32 and scales.numel() == 4
    # the bytes are torch.float8_e4m3fn's, and the host expands them as torch does
    assert torch.equal(y, codes.view(torch.float8_e4m3fn).float() * per_value(scales, 1000))
    ref = x.float().cpu()
    for k in range(4):
        blk = ref[k * 256:(k + 1) * 256]
        top = blk.abs().max()
        assert scales[k].item() == pytest.approx(top.item() / 448, rel=1e-6)
        assert y[k * 256:(k + 1) * 256].abs().max().item() == pytest.approx(top.item(), rel=1e-6)  # the largest is exact
    # everything within 2^14 of its block's largest lands on one of its two e4m3 neighbours: at most 1/8 off
    step_floor = per_value(scales, 1000) * 2.0 ** -6
    normal = ref.abs() >= step_floor
    assert normal.sum() > 900
    assert ((y - ref).abs()[normal] <= ref.abs()[normal] / 8 * (1 + 1e-5)).all()
    assert torch.equal(torch.sign(y)[normal], torch.sign(ref)[normal])
    # smaller ones move by at most one subnormal step
    assert ((y - ref).abs()[~normal] <= per_value(scales, 1000)[~normal] * 2.0 ** -9 * (1 + 1e-5)).all()


def test_rounding_is_stochastic_and_right_on_average():
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(4096, generator=g) * torch.logspace(-6, 0, 4096)).to(DEV)
    x[::256] = 3.0  # each block has one large value: the rest are small next to it
    ref = x.cpu()
    seeds = 400
    total = torch.zeros(4096)
    first = None
    differ = False
    for seed in range(seeds):
        y, codes, _ = roundtrip(x, seed)
        total += y
        if first is None:
            first = codes.clone()
        differ = differ or not torch.equal(codes, first)
    assert differ  # not round-to-nearest
    mean = total / seeds
    assert (mean - ref).abs().sum().item() < 0.01 * ref.abs().sum().item()
    tiny = ref.abs() < 3.0 / 448 * 2.0 ** -9  # below the smallest step: 0 or one step, never biased to 0
    assert tiny.sum() > 50
    assert mean[tiny].abs().sum().item() == pytest.approx(ref[tiny].abs().sum().item(), rel=0.2)
    one, _, _ = roundtrip(x, 7)
    assert (one[tiny] == 0).any() and (one[tiny] != 0).any()


def test_zeros_and_non_finite_blocks():
    x = torch.zeros(600, device=DEV)
    x[300:] = torch.randn(300, device=DEV)
    y, codes, scales = roundtrip(x)
    assert scales[0] == 0 and (codes[:256] == 0).all() and (y[:256] == 0).all()
    assert torch.isfinite(y).all()
    for bad in (float("nan"), float("inf"), float("-inf")):
        z = x.clone()
        z[700 - 256] = bad  # in the second block
        y, _, scales = roundtrip(z)
        assert torch.isnan(scales[1]) and torch.isnan(y[256:512]).all()  # the whole block is marked
        assert torch.isfinite(y[:256]).all() and torch.isfinite(y[512:]).all()


# ---------------------------------------------------------------- the optimizer
def make_mlp(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 10)).to(DEV, dtype)


def task(n, seed=1, dtype=torch.float32):
    """A learnable task: labels from a fixed random teacher."""
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(64, 10, generator=g)
    out = []
    for _ in range(n):
        x = torch.randn(64, 64, generator=g)
        out.append((x.to(DEV, dtype), (x @ teacher).argmax(-1).to(DEV)))
    return out


def train(model, opt, data):
    losses = []
    for x, y in data:
        loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    opt.synchronize()
    return losses


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("attached", [False, True], ids=["standalone", "attached"])
def test_trains_like_full_precision_gradients_on_less_traffic(dtype, attached):
    data = task(60, dtype=dtype)
    runs = {}
    for fp8 in (False, True):
        model = make_mlp(dtype=dtype)
        opt = OffloadAdamW(model.parameters(), mode="cpu", fp8_gradients=fp8, **HP)
        if attached:
            opt.attach(model)
        losses = train(model, opt, data)
        stats = opt.stats()
        assert stats["fp8_gradients"] is fp8
        runs[fp8] = (losses, stats["grad_bytes_offloaded"], model)
    plain, coded = runs[False], runs[True]
    assert plain[0][0] == coded[0][0]  # the same start
    end_plain, end_coded = sum(plain[0][-10:]) / 10, sum(coded[0][-10:]) / 10
    assert end_plain < 0.6 * plain[0][0]  # the task is learned ...
    assert end_coded == pytest.approx(end_plain, rel=0.1)  # ... equally well from one byte per value
    numel = sum(p.numel() for p in coded[2].parameters())
    blocks = sum(-(-p.numel() // 256) for p in coded[2].parameters())
    assert coded[1] == 60 * (numel + 4 * blocks)  # codes and block scales
    assert plain[1] == 60 * numel * (4 if dtype == torch.float32 else 2)
    assert coded[1] < (0.27 if dtype == torch.float32 else 0.52) * plain[1]


def test_accumulated_micro_batches_and_clipping():
    data = task(40)
    ends = {}
    for fp8 in (False, True):
        model = make_mlp()
        opt = OffloadAdamW(model.parameters(), mode="cpu", fp8_gradients=fp8, max_grad_norm=1.0, accumulation_steps=2,
                           **HP).attach(model)
        losses = []
        for i, (x, y) in enumerate(data):
            loss = F.cross_entropy(model(x), y) / 2
            loss.backward()
            losses.append(2 * loss.item())
            if i % 2:
                opt.step()
                opt.zero_grad(set_to_none=True)
        opt.synchronize()
        ends[fp8] = sum(losses[-8:]) / 8
        assert opt.last_grad_norm is not None and opt.last_grad_norm > 0
    assert ends[True] == pytest.approx(ends[False], rel=0.1)


def test_a_non_finite_gradient_is_still_caught():
    model = make_mlp()
    opt = OffloadAdamW(model.parameters(), mode="cpu", fp8_gradients=True, **HP).attach(model)
    data = task(3)
    train(model, opt, data[:2])
    before = [p.detach().clone() for p in model.parameters()]
    handle = model[2].weight.register_hook(lambda g: g * float("nan"))
    with pytest.warns(RuntimeWarning, match="non-finite"):
        train(model, opt, data[2:])
    handle.remove()
    assert opt.skipped_steps == 1
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_only_the_host_optimizer_takes_them():
    model = make_mlp()
    with pytest.raises(ValueError, match="fp8_gradients"):
        OffloadAdamW(model.parameters(), mode="gpu", fp8_gradients=True, **HP)
    with pytest.raises(ValueError, match="fp8_gradients"):
        trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), optimizer_mode="gpu", fp8_gradients=True,
                         offload=True, activation_offload=False, profile=False)


def test_wrap_passes_it_on():
    model = make_mlp()
    model, opt = trainstream.wrap(model, torch.optim.AdamW(model.parameters(), **HP), offload=True,
                                  activation_offload=False, fp8_gradients=True, profile=False)
    try:
        assert opt.fp8_gradients and trainstream.wrapper.unwrap_session(model).config["fp8_gradients"] is True
        losses = train(model, opt, task(30))
        assert sum(losses[-5:]) < sum(losses[:5])
        assert trainstream.memory_report(model)["optimizer"]["fp8_gradients"] is True
    finally:
        trainstream.unwrap(model)
