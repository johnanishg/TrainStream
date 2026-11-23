"""Background checkpoints: saved while training goes on, resumed bit for bit.

Three runs are compared in most tests:
  plain   trains all the batches, never saves
  saved   the same, but writes a checkpoint in the background part-way and keeps training
  resumed a fresh model and optimizer, restored from that file, trained on the rest
Saving must not change the run (saved == plain) and the file must hold exactly the state at
the moment of the call (resumed == plain), although the updates of the step before it were
still in flight when it was made.
"""

import copy
import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import trainstream  # noqa: E402
from trainstream import OffloadAdamW  # noqa: E402

from test_paging import TinyLM, tokens  # noqa: E402
from test_paging import train as train_lm  # noqa: E402

DEV = "cuda"
HP = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def make_mlp(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 10)).to(DEV, dtype)


def batches(n, seed=1, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [(torch.randn(32, 64, generator=g).to(DEV, dtype), torch.randint(0, 10, (32,), generator=g).to(DEV))
            for _ in range(n)]


def train(model, opt, data):
    for x, y in data:
        F.cross_entropy(model(x).float(), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


def finish(opt):
    opt.synchronize()
    torch.cuda.synchronize()


def same(a, b):
    return all(torch.equal(p.detach(), q.detach()) for p, q in zip(a.parameters(), b.parameters()))


def build(kind, dtype=torch.float32):
    """kind: mode[-moments][+attached] or "resident"."""
    model = make_mlp(dtype=dtype)
    if kind == "resident":
        opt = OffloadAdamW(model.parameters(), mode="cpu", **HP).attach(model)
        opt._go_resident()
        return model, opt
    mode, _, moments = kind.partition("-")
    opt = OffloadAdamW(model.parameters(), mode=mode, moment_dtype=moments or "float32", **HP).attach(model)
    return model, opt


KINDS = ["cpu", "cpu-bfloat16", "cpu-int8", "gpu", "gpu-bfloat16", "resident"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_save_while_training_and_resume_bit_for_bit(tmp_path, kind, dtype):
    data = batches(9, dtype=dtype)
    path = str(tmp_path / "step5.ckpt")

    plain, opt = build(kind, dtype)
    train(plain, opt, data)
    finish(opt)

    saved, opt = build(kind, dtype)
    train(saved, opt, data[:5])
    handle = trainstream.save_checkpoint(saved, opt, path, extra={"step": 5, "note": "half way"})
    train(saved, opt, data[5:])  # no wait: the step-5 updates may still be in flight, and training goes on
    finish(opt)
    handle.wait()
    assert handle.done() and handle.error is None and handle.bytes > 0 and handle.seconds >= 0
    assert os.path.isfile(path) and not os.path.exists(path + ".tmp")
    assert same(saved, plain)  # saving changed nothing

    resumed, opt = build(kind, dtype)
    extra = trainstream.load_checkpoint(path, resumed, opt)
    assert extra == {"step": 5, "note": "half way"}
    train(resumed, opt, data[5:])
    finish(opt)
    assert same(resumed, plain)  # the file held exactly the state after step 5


def test_manifest_describes_the_file(tmp_path):
    model, opt = build("cpu-int8", torch.bfloat16)
    data = batches(2, dtype=torch.bfloat16)
    train(model, opt, data)
    path = str(tmp_path / "a.ckpt")
    opt.save_async(path, extra={"k": [1, 2]}, model=model).wait()
    got = trainstream.read_checkpoint(path)
    man, tensors = got["manifest"], got["tensors"]
    assert man["format"] == "trainstream-checkpoint" and man["mode"] == "cpu" and man["moment_dtype"] == "int8"
    assert man["extra"] == {"k": [1, 2]}
    assert [s["name"] for s in sorted(man["slots"], key=lambda s: s["slot"])] == [n for n, _ in model.named_parameters()]
    assert man["param_groups"][0]["lr"] == HP["lr"] and man["param_groups"][0]["betas"] == list(HP["betas"])
    assert all(t["offset"] % 4096 == 0 for t in man["tensors"])
    assert tensors["steps"].tolist() == [2] * 6
    w = model[0].weight
    assert tensors["slot/0/master"].dtype == torch.float32 and tensors["slot/0/master"].numel() == w.numel()
    assert tensors["slot/0/exp_avg"].dtype == torch.uint8 and tensors["slot/0/m_scale"].dtype == torch.float32
    assert torch.equal(tensors["slot/0/param_stage"].view(w.shape), w.detach().cpu())
    assert torch.equal(tensors["slot/0/master"].view(w.shape).to(torch.bfloat16), w.detach().cpu())  # weights = rounded masters
    assert list(trainstream.read_checkpoint(path, keys=[])["tensors"]) == []


@pytest.mark.parametrize("src,dst", [("cpu-int8", "cpu"), ("cpu-bfloat16", "gpu"), ("gpu", "cpu-bfloat16"),
                                     ("resident", "cpu"), ("cpu", "resident")])
def test_a_checkpoint_loads_into_a_different_optimizer(tmp_path, src, dst):
    data = batches(4)
    a, opt_a = build(src)
    train(a, opt_a, data)
    finish(opt_a)
    path = str(tmp_path / "x.ckpt")
    trainstream.save_checkpoint(a, opt_a, path, blocking=True)
    want = opt_a.state_dict()["state"]

    b, opt_b = build(dst)
    trainstream.load_checkpoint(path, b, opt_b)
    finish(opt_b)
    assert same(a, b)
    got = opt_b.state_dict()["state"]
    assert got.keys() == want.keys()
    tol = 0.2 if "int8" in src else 2e-2 if "bfloat16" in (src + dst) else 1e-6  # the coarser storage's rounding
    for k in want:
        assert float(got[k]["step"]) == float(want[k]["step"]) == 4
        for name in ("exp_avg", "exp_avg_sq"):
            x, y = got[k][name].float().cpu(), want[k][name].float().cpu()
            assert (x - y).abs().max() <= tol * y.abs().max() + 1e-12, (k, name)
    train(a, opt_a, data[:2])
    train(b, opt_b, data[:2])  # and both go on training
    finish(opt_a)
    finish(opt_b)
    assert all(torch.isfinite(p).all() for p in b.parameters())


def test_a_failed_save_leaves_the_last_checkpoint_and_training_alone(tmp_path):
    model, opt = build("cpu")
    data = batches(6)
    train(model, opt, data[:2])
    path = str(tmp_path / "keep.ckpt")
    trainstream.save_checkpoint(model, opt, path, blocking=True)
    before = open(path, "rb").read()

    os.mkdir(path + ".tmp")  # the temporary name is taken by a folder: the write cannot start
    train(model, opt, data[2:3])
    handle = trainstream.save_checkpoint(model, opt, path)
    train(model, opt, data[3:])  # updates are not left waiting for a writer that gave up
    finish(opt)
    with pytest.raises(OSError, match="writing checkpoint"):
        handle.wait()
    assert handle.done() and handle.error
    assert open(path, "rb").read() == before
    os.rmdir(path + ".tmp")
    trainstream.save_checkpoint(model, opt, path, blocking=True)  # and the next save works
    assert open(path, "rb").read() != before


def test_saves_queue_up_behind_each_other(tmp_path):
    model, opt = build("cpu")
    data = batches(4)
    handles = []
    for i, b in enumerate(data):
        train(model, opt, [b])
        handles.append(trainstream.save_checkpoint(model, opt, str(tmp_path / f"s{i}.ckpt")))
    finish(opt)
    for i, h in enumerate(handles):
        h.wait()
        assert trainstream.read_checkpoint(h.path, keys=["steps"])["tensors"]["steps"].tolist() == [i + 1] * 6


def test_checkpoint_of_a_step_the_guard_skipped_holds_the_state_before_it(tmp_path):
    model, opt = build("cpu")
    data = batches(3)
    train(model, opt, data[:2])
    handle = model[2].weight.register_hook(lambda g: g * float("nan"))
    with pytest.warns(RuntimeWarning, match="non-finite"):
        F.cross_entropy(model(data[2][0]), data[2][1]).backward()
        handle.remove()
        opt.step()
        opt.zero_grad()
        ckpt = trainstream.save_checkpoint(model, opt, str(tmp_path / "g.ckpt"))  # the skipped step is in flight
        finish(opt)
    ckpt.wait()
    assert trainstream.read_checkpoint(ckpt.path, keys=["steps"])["tensors"]["steps"].tolist() == [2] * 6
    other, opt2 = build("cpu")
    trainstream.load_checkpoint(ckpt.path, other, opt2)
    finish(opt2)
    assert same(other, model)


def test_wrong_model_is_refused(tmp_path):
    model, opt = build("cpu")
    train(model, opt, batches(1))
    path = str(tmp_path / "m.ckpt")
    trainstream.save_checkpoint(model, opt, path, blocking=True)
    torch.manual_seed(0)
    small = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 10)).to(DEV)
    with pytest.raises(ValueError, match="trained parameters"):
        trainstream.load_checkpoint(path, small, OffloadAdamW(small.parameters(), **HP))
    wide = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 10)).to(DEV)
    with pytest.raises(ValueError, match="has shape"):
        trainstream.load_checkpoint(path, wide, OffloadAdamW(wide.parameters(), **HP))
    with open(str(tmp_path / "junk.ckpt"), "wb") as fh:
        fh.write(b"not a checkpoint at all")
    with pytest.raises(ValueError, match="not a TrainStream checkpoint"):
        trainstream.load_checkpoint(str(tmp_path / "junk.ckpt"), model, opt)
    finish(opt)


# ---------------------------------------------------------------- a wrapped, paged model
def wrapped_lm(**kw):
    torch.manual_seed(0)
    model = TinyLM().to(DEV)
    return trainstream.wrap(model, torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05),
                            offload=True, param_offload=True, activation_offload=False, **kw)


@pytest.mark.parametrize("max_grad_norm", [None, 0.5], ids=["plain", "clipped"])
def test_paged_model_saves_in_the_background_and_resumes(tmp_path, max_grad_norm):
    data = tokens(7)
    path = str(tmp_path / "lm.ckpt")
    kw = dict(max_grad_norm=max_grad_norm)

    plain, opt = wrapped_lm(**kw)
    try:
        train_lm(plain, opt, data)
        opt.synchronize()
        want = {k: v.clone() for k, v in plain.state_dict().items()}
    finally:
        trainstream.unwrap(plain)

    saved, opt = wrapped_lm(**kw)
    try:
        train_lm(saved, opt, data[:4])
        handle = trainstream.save_checkpoint(saved, opt, path, extra={"seen": 4})
        train_lm(saved, opt, data[4:])
        opt.synchronize()
        handle.wait()
        got = saved.state_dict()
        assert all(torch.equal(got[k], want[k]) for k in want)
    finally:
        trainstream.unwrap(saved)

    resumed, opt = wrapped_lm(**kw)
    try:
        assert trainstream.load_checkpoint(path, resumed, opt) == {"seen": 4}
        train_lm(resumed, opt, data[4:])
        opt.synchronize()
        got = resumed.state_dict()
        assert all(torch.equal(got[k], want[k]) for k in want)
    finally:
        trainstream.unwrap(resumed)
