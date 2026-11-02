"""Recomputing activations in backward instead of keeping or paging them."""

import copy

import pytest
import torch
from torch import nn

import trainstream
from trainstream.hooks import ActivationOffloader, recompute_count

from test_quant import HP, make, tokens, train, trainable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def wrap(m, **kw):
    kw.setdefault("activation_offload", False)
    return trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, profile=False, **kw)


def run(m, n=5, **kw):
    m, opt = wrap(m, **kw)
    try:
        s = trainstream.wrapper.unwrap_session(m)
        losses = train(m, opt, tokens(n))
        info = dict(policy=s.activations.policy if s.activations else None,
                    loads=s.pager.stats["loads"] if s.pager else None,
                    report=trainstream.memory_report(m))
    finally:
        trainstream.unwrap(m)
    return losses, [p.detach().clone() for p in m.parameters()], info


@pytest.mark.parametrize("paged", [False, True])
def test_recomputed_blocks_train_bit_identically(paged):
    base = make()
    ref_losses, ref_params, _ = run(copy.deepcopy(base), param_offload=paged)
    losses, params, info = run(copy.deepcopy(base), param_offload=paged, recompute=[1, 3, 4])
    assert info["policy"] == ["keep", "recompute", "keep", "recompute", "recompute", "keep"]
    assert losses == ref_losses
    assert all(torch.equal(a, b) for a, b in zip(params, ref_params))
    assert info["report"]["activations"]["recompute_blocks"] == [1, 3, 4]


def test_paged_recompute_loads_each_block_once_per_pass():
    base = make()
    _, _, plain = run(copy.deepcopy(base), param_offload=True)
    _, _, rc = run(copy.deepcopy(base), param_offload=True, recompute=True)
    assert rc["policy"][:5] == ["recompute"] * 5 and rc["policy"][5] == "keep"
    assert rc["loads"] == plain["loads"]  # recomputation reuses the backward load, no extra traffic


def test_recompute_with_activation_offload_and_low_precision():
    base = make()
    kw = dict(activation_offload=True, quantize="nvfp4", param_offload=True)
    ref_losses, _, _ = run(copy.deepcopy(base), **kw)
    losses, _, info = run(copy.deepcopy(base), recompute=[0, 2], **kw)
    assert info["policy"][0] == "recompute" and info["policy"][1] == "offload"  # never both
    assert losses == ref_losses


def test_recompute_count_balances_link_and_compute():
    # link-bound: 50 ms of paged weights plus 10 streamed blocks of 8 ms vs 66 ms of compute
    r = recompute_count(offloaded=10, blocks=11, fwd_ms=2.0, link_ms=8.0, reserved_link_ms=50.0)
    link = 50.0 + (10 - r) * 8.0
    gpu = 3 * 11 * 2.0 + r * 2.0
    assert 0 < r < 10 and abs(link - gpu) <= 8.0 + 2.0  # balanced to within one block
    # the paged weights alone saturate the link: recompute every offloaded block
    assert recompute_count(10, 11, fwd_ms=2.0, link_ms=8.0, reserved_link_ms=500.0) == 10
    # compute-bound: streaming is free, recompute nothing
    assert recompute_count(10, 11, fwd_ms=20.0, link_ms=1.0, reserved_link_ms=5.0) == 0
    # nothing offloaded or nothing measured
    assert recompute_count(0, 11, 2.0, 8.0, 100.0) == 0
    assert recompute_count(10, 11, 0.0, 8.0, 100.0) == 0


def test_auto_recompute_engages_only_under_memory_pressure():
    base = make()
    # weights fit: auto never recomputes
    _, _, info = run(copy.deepcopy(base), activation_offload="auto", n=6)
    assert "recompute" not in (info["policy"] or [])
    # weights paged and all activations streamed: auto measures, then decides (any r is valid on this GPU)
    m = copy.deepcopy(base)
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), offload=True, param_offload=True,
                              activation_offload="auto")
    try:
        s = trainstream.wrapper.unwrap_session(m)
        assert s.activations.phase == "probe"  # one block paged to measure it, the rest recomputed
        assert len(s.activations.offload) == 1 and len(s.activations.recompute) == 4
        train(m, opt, tokens(20))  # probe, measure, then try the model's answer and recompute-all
        assert s.activations._recompute_auto
        assert s.activations.phase == "fixed"
        d = s.activations.decision
        assert "recompute_block_fwd_ms" in d and d["recompute_block_bytes"] > 0
        kept = set(d.get("kept_blocks", ()))  # VRAM left over afterwards keeps these blocks' activations
        assert set(d["recompute_blocks"]) - kept == s.activations.recompute
        assert not kept & (s.activations.recompute | s.activations.offload)
        trials = {int(k): v for k, v in d["recompute_trials_ms"].items()}
        assert 0 in trials and len(trials) >= 2  # the baseline and at least one tried setting
        assert len(d["recompute_blocks"]) == min(trials, key=trials.get)  # the fastest measured one is applied
    finally:
        trainstream.unwrap(m)


def test_trials_apply_the_measured_setting_when_host_ram_forces_recompute():
    # Trial keys count every recomputed block, those host RAM cannot hold
    # included: the setting applied must be the one measured, not one shifted
    # by the forced blocks.
    m = make().cuda()
    off = ActivationOffloader(m, list(m.blocks), offload=False, recompute=False)
    try:
        off._forced_recompute = {4}  # host RAM could not hold block 4's activations
        off._offload_all = {0, 1, 2, 3}
        off._trial_results = {1: 30.0, 2: 10.0, 5: 20.0}  # forced only; one more recomputed; all of them
        off._next_trial(first=True)
        assert off.recompute == {3, 4} and off.offload == {0, 1, 2}
        assert off.decision["recompute_blocks"] == [3, 4]
        # VRAM found later: the latest (recomputed) blocks are kept first
        off.adjustable = True
        assert off.keep_resident(1) == [4]
        assert off.recompute == {3} and off.offload == {0, 1, 2} and off.policy[4] == "keep"
        assert off.keep_resident(10) == [0, 1, 2, 3] and not off.recompute and not off.offload
    finally:
        off.remove()


def test_recompute_keeps_the_model_from_going_resident():
    # the resident switch drops the activation machinery: the step-2 peak only
    # holds while blocks are recomputed, so it must not switch
    m = make()
    m, opt = trainstream.wrap(m, torch.optim.AdamW(trainable(m), **HP), activation_offload=False,
                              recompute=[1, 3], profile=False)
    try:
        s = trainstream.wrapper.unwrap_session(m)
        train(m, opt, tokens(5))
        assert not opt.resident
        assert s.activations.policy[1] == "recompute" and s.activations.policy[3] == "recompute"
    finally:
        trainstream.unwrap(m)


def test_models_with_their_own_checkpointing_are_left_alone():
    m = make()
    m.is_gradient_checkpointing = True
    m, opt = wrap(m, recompute=True)
    try:
        s = trainstream.wrapper.unwrap_session(m)
        assert s.config["recompute"] is False and s.activations is None
    finally:
        trainstream.unwrap(m)
    with pytest.raises(ValueError, match="recompute"):
        wrap(make(), recompute="sometimes")
