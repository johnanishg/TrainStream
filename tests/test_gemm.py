"""The gated MLP front: the fused FP4 kernel (csrc/gemm_fp4_kernel.cu) against cuBLAS, and
GatedFrontFn, which keeps the MLP's input for backward instead of gate / up outputs."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

import trainstream  # noqa: E402
from trainstream import _C, fused, lowp  # noqa: E402
from trainstream import _build  # noqa: E402

from test_lora_fused import TARGETS, tiny_lora  # noqa: E402

DEV = "cuda"
bf = torch.bfloat16


def need_kernel():
    if not _C.gemm_fp4_supported(0):
        pytest.skip("no FP4 tensor-core kernel in this build or on this GPU (needs sm_120)")


def need_fp4():
    if not lowp.native_supported("nvfp4", DEV):
        pytest.skip("this GPU has no nvfp4 tensor-core matmul")


# -------------------------------------------------------------------- the build
def test_blackwell_is_compiled_with_its_full_feature_set(monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("TRAINSTREAM_NO_FP4", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: [(8, 9), (12, 0)][i])
    assert _build._arch_list() == "8.9;12.0a"
    assert _build._arch_list(fp4=False) == "8.9;12.0"
    assert "-DTS_FP4_MMA=1" in _build._nvcc_flags("8.9;12.0a")
    assert "-DTS_FP4_MMA=1" not in _build._nvcc_flags("8.9;12.0")
    assert "-DTS_FP4_MMA=1" not in _build._nvcc_flags("10.0a")  # the instruction is sm_120's
    assert _build._build_key("8.9;12.0a") != _build._build_key("8.9;12.0")
    monkeypatch.setenv("TRAINSTREAM_NO_FP4", "1")
    assert _build._arch_list() == "8.9;12.0" and "-DTS_FP4_MMA=1" not in _build._nvcc_flags("12.0a")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    assert _build._arch_list() == "9.0"  # an explicit list is taken as it is


# ------------------------------------------------------------------- the kernel
def operands(M, N, K, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(M, K, device=DEV, generator=g) * torch.logspace(-1, 0.5, K, device=DEV)).to(bf)
    ws = [(torch.randn(N, K, device=DEV, generator=g) * 0.05).to(bf) for _ in range(2)]
    return x, ws


def reference(x, ws, loras):
    """What the model's own path computes: cuBLAS FP4 products in bf16, then the LoRA terms."""
    xq, xs, xg = lowp._quantize(x, "nvfp4")
    outs = []
    for w, lo in zip(ws, loras):
        wq, wsc, wg = lowp._quantize(w, "nvfp4")
        y = lowp._mm("nvfp4", xq, xs, xg, wq, wsc, wg, torch.float32)
        if lo is not None:
            y = y + lo[0].float() @ lo[1].float().t()
        outs.append(y)
    return outs


def run_kernel(x, ws, loras, want_h=True, want_out=True, act=0):
    xq, xs, xg = lowp._quantize(x, "nvfp4")
    wq, wsc, wg = zip(*[lowp._quantize(w, "nvfp4") for w in ws])
    empty = torch.empty(0, dtype=bf, device=DEV)
    xa = [empty if lo is None else lo[0] for lo in loras]
    sb = [empty if lo is None else lo[1] for lo in loras]
    return _C.gemm_fp4_front(xq, xs, xg, list(wq), list(wsc), [float(g) for g in wg], x.shape[0], ws[0].shape[0],
                             x.shape[1], xa, sb, act, want_h, want_out)


def lora_pair(M, N, R, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, R, device=DEV, generator=g).to(bf).contiguous(),
            (torch.randn(N, R, device=DEV, generator=g) * 0.1).to(bf).contiguous())


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("M,N,K", [(200, 176, 192), (64, 64, 64), (129, 208, 320), (1000, 512, 1024)])
@pytest.mark.parametrize("which", ["both", "none", "gate-only"])
def test_kernel_matches_cublas_products_lora_and_activation(M, N, K, which):
    need_kernel()
    x, ws = operands(M, N, K)
    loras = {"both": [lora_pair(M, N, 16, 1), lora_pair(M, N, 16, 2)], "none": [None, None],
             "gate-only": [lora_pair(M, N, 32, 3), None]}[which]
    g_ref, u_ref = reference(x, ws, loras)
    h, g, u = run_kernel(x, ws, loras)
    assert h.dtype == g.dtype == bf and h.shape == g.shape == u.shape == (M, N)
    # bf16 keeps 8 bits: both sides round, so they agree to a few parts in a thousand
    assert rel(g, g_ref) < 4e-3 and rel(u, u_ref) < 4e-3
    h_ref = torch.nn.functional.silu(g_ref) * u_ref
    assert rel(h, h_ref) < 6e-3
    (h_only,) = run_kernel(x, ws, loras, want_h=True, want_out=False)
    assert torch.equal(h_only, h)  # asking for g and u as well changes nothing
    (h_gelu,) = run_kernel(x, ws, loras, want_out=False, act=1)
    assert rel(h_gelu, torch.nn.functional.gelu(g_ref, approximate="tanh") * u_ref) < 6e-3
    assert not _C.guard_read(0, True)["fused_gemm"]


def test_kernel_with_one_projection_is_a_lora_linear():
    need_kernel()
    x, ws = operands(300, 96, 256, seed=4)
    lo = lora_pair(300, 96, 16, 5)
    (y_ref,) = reference(x, ws[:1], [lo])
    (y,) = run_kernel(x, ws[:1], [lo], want_h=False, want_out=True)
    assert rel(y, y_ref) < 4e-3


def test_kernel_flags_a_non_finite_result_and_refuses_shapes_it_cannot_take():
    need_kernel()
    x, ws = operands(64, 64, 64, seed=6)
    lo = lora_pair(64, 64, 16, 7)
    _C.guard_read(0, True)
    bad = (lo[0].clone(), lo[1])
    bad[0][5, 3] = float("inf")
    h, _, _ = run_kernel(x, ws, [bad, None])
    torch.cuda.synchronize()
    assert _C.guard_read(0, True)["fused_gemm"] and not torch.isfinite(h[5].float()).all()
    assert torch.isfinite(h[6].float()).all()  # only the row the bad value is in
    with pytest.raises(RuntimeError, match="multiple of 16"):
        run_kernel(x, ws, [lora_pair(64, 64, 8, 8), None])
    xq, xs, xg = lowp._quantize(x, "nvfp4")
    wq, wsc, wg = lowp._quantize(ws[0], "nvfp4")
    e = torch.empty(0, dtype=bf, device=DEV)
    with pytest.raises(RuntimeError, match="K % 64"):
        _C.gemm_fp4_front(xq, xs, xg, [wq], [wsc], [float(wg)], 64, 64, 96, [e], [e], 0, False, True)
    with pytest.raises(RuntimeError, match="two projections"):
        _C.gemm_fp4_front(xq, xs, xg, [wq], [wsc], [float(wg)], 64, 64, 64, [e], [e], 0, True, False)


# ------------------------------------------------------- the front inside a model
def ids(n=4, seq=33, seed=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (2, seq), generator=g).cuda() for _ in range(n)]


def train(kind, steps=4, model=None, autocast=False, **kw):
    """Train the tiny LoRA model on a `kind` base; returns (losses, trained weights, report, session events)."""
    model = model if model is not None else tiny_lora(dtype=bf)
    params = [p for p in model.parameters() if p.requires_grad]
    before = dict(fused.FRONT_STATS)
    # attention=None: PyTorch's own backend order. "auto" times the backends in each run and may
    # pick another one from run to run, which moves the last bits: the comparisons here are exact.
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize=kind, kernels=True,
                                  fused_lora=True, profile=False, attention=None, **kw)
    try:
        losses = []
        for batch in ids(steps):
            with torch.autocast("cuda", dtype=bf, enabled=autocast):
                loss = model(input_ids=batch, labels=batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        opt.synchronize()
        report = trainstream.memory_report(model)
        weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        events = list(trainstream.wrapper.unwrap_session(model).events)
        delta = {k: fused.FRONT_STATS[k] - before[k] for k in before}
        return losses, weights, report, events, delta
    finally:
        trainstream.unwrap(model)


@pytest.mark.parametrize("kind", ["nvfp4", "nf4"])
@pytest.mark.parametrize("autocast", [False, True], ids=["bf16", "autocast"])
def test_recomputing_gate_and_up_trains_exactly_as_keeping_them(kind, autocast):
    if kind == "nvfp4":
        need_fp4()
    kept = train(kind, autocast=autocast, mlp_recompute=False)
    again = train(kind, autocast=autocast, mlp_recompute=True)
    assert kept[4]["recomputed"] == 0 and "mlp_front" not in kept[2]
    assert again[4]["recomputed"] == 4 * 2 and again[4]["kernel"] == 0  # every MLP, every step
    assert again[4]["bytes_not_kept"] == 4 * 2 * 2 * (2 * 33 * 256 * 2)  # g and u, bf16
    assert again[2]["mlp_front"]["mode"] is True and again[2]["mlp_front"]["engaged"]
    assert any("keep their input for backward" in e for e in again[3])
    assert again[0] == kept[0]  # the same losses, to the last bit
    for n in kept[1]:
        assert torch.equal(again[1][n], kept[1][n]), n  # and the same trained adapters


def test_auto_takes_tensor_core_mlps_in_blocks_whose_activations_are_paged():
    need_fp4()
    calm = train("nvfp4", mlp_recompute="auto")  # everything fits: g and u are kept, nothing recomputed
    assert calm[4]["recomputed"] == 0 and calm[2]["mlp_front"]["engaged"] is False

    kw = dict(activation_offload=True, min_offload_bytes=0)  # block 0 paged; the last block never is
    paged = train("nvfp4", mlp_recompute="auto", **kw)
    assert paged[2]["activations"]["offloaded_blocks"] == [0]
    assert paged[4]["recomputed"] == 4 * 1 and paged[2]["mlp_front"]["engaged"] is True  # block 0's MLP only
    assert any("recomputed in backward instead of paged" in e for e in paged[3])
    kept = train("nvfp4", mlp_recompute=False, **kw)
    assert paged[0] == kept[0]  # the same training as with gate / up paged, bit for bit
    # (against the unpaged run only closely: that one moves to the stock optimizer after two steps)
    assert paged[0] == pytest.approx(calm[0], rel=1e-5)
    assert paged[2]["activations"]["packed_bytes"] < 0.8 * kept[2]["activations"]["packed_bytes"]

    whole = train("nvfp4", mlp_recompute="auto", recompute=True)  # blocks recomputed, nothing paged
    assert whole[4]["recomputed"] == 0

    expanded = train("nf4", mlp_recompute="auto", **kw)
    assert expanded[4]["recomputed"] == 0  # two full bf16 matmuls per MLP: not without being asked


def test_a_block_moved_back_to_vram_is_sized_with_its_gate_and_up_outputs():
    """While a block's activations are paged, "auto" leaves gate / up out of what is measured
    per block; kept in VRAM the block holds them again, and the room is computed for that."""
    need_fp4()
    model = tiny_lora(dtype=bf)
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize="nvfp4", kernels=True,
                                  fused_lora=True, profile=False, activation_offload=True, min_offload_bytes=0)
    try:
        act = trainstream.wrapper.unwrap_session(model).activations
        batch = ids(1)[0]
        model(input_ids=batch, labels=batch).loss.backward()
        opt.step()
        opt.zero_grad()
        assert act.unkept_block_bytes == 2 * (2 * 33 * 256 * 2)  # g and u of the paged block's MLP
        assert act.paging_now() is False  # no block is running
    finally:
        trainstream.unwrap(model)


def test_fewer_activation_bytes_go_to_host_ram():
    need_fp4()
    kw = dict(activation_offload=True, min_offload_bytes=0)
    kept = train("nvfp4", mlp_recompute=False, **kw)
    again = train("nvfp4", mlp_recompute=True, **kw)
    a, b = kept[2]["activations"]["packed_bytes"], again[2]["activations"]["packed_bytes"]
    # per step, in the one paged block (the last is never paged): g and u (bf16), and the two small
    # x A^T [tokens, 8] (fp32) the adapters of gate and up kept beside them
    gu = 4 * 2 * (2 * 33 * 256 * 2)
    xa = 4 * 2 * (2 * 33 * 8 * 4)
    assert a - b == gu + xa and b < 0.8 * a
    assert again[0] == kept[0]


def test_fused_kernel_front_trains_like_the_default_path():
    need_fp4()
    need_kernel()
    plain = train("nvfp4", steps=6, mlp_recompute=False)
    fusedk = train("nvfp4", steps=6, fused_gemm=True)
    assert fusedk[4]["kernel"] == 6 * 2 and fusedk[4]["recomputed"] == 6 * 2
    assert fusedk[2]["mlp_front"]["fused_gemm"] and fusedk[2]["mlp_front"]["kernel_available"]
    assert any("2 of 2 MLP fronts run in one FP4 kernel" in e for e in fusedk[3])
    assert fusedk[0] == pytest.approx(plain[0], rel=5e-3)  # rounding differs (fp32 accumulators), nothing else
    assert not trainstream._C.guard_read(0, False)["fused_gemm"]


def test_fused_kernel_is_left_out_where_it_does_not_apply():
    need_fp4()
    # outlier channels are multiplied in bf16 beside the FP4 product: not something the kernel does
    out = train("nvfp4", fused_gemm=True, outliers=0.05)
    assert out[4]["kernel"] == 0 and out[4]["recomputed"] == 0
    assert any("0 of 2 MLP fronts" in e or "no FP4 tensor-core kernel" in e for e in out[3])
    nf4 = train("nf4", fused_gemm=True)
    assert nf4[4]["kernel"] == 0 and nf4[4]["recomputed"] == 0


def test_blocks_recomputed_in_backward_take_the_usual_path_both_times():
    """torch.utils.checkpoint compares what a block's forward and its recomputation keep: the
    choice must be the same in both. Blocks TrainStream recomputes keep nothing, so their MLPs
    stay on the usual path; the last block (never recomputed) uses the front."""
    need_fp4()
    plain = train("nvfp4", mlp_recompute=False, recompute=True)
    both = train("nvfp4", mlp_recompute=True, recompute=True)
    assert both[2]["activations"]["recompute_blocks"] == [0]
    assert both[4]["recomputed"] == 4 * 1  # the last block's MLP only
    assert both[0] == plain[0]
    for n in plain[1]:
        assert torch.equal(both[1][n], plain[1][n]), n


def test_under_the_models_own_gradient_checkpointing():
    """Hugging Face's gradient checkpointing runs every layer twice: the front is used in both
    runs alike, and training is the same as without it."""
    need_fp4()
    outs = {}
    for mode in (False, True):
        model = tiny_lora(dtype=bf)
        model.gradient_checkpointing_enable()
        outs[mode] = train("nvfp4", model=model, mlp_recompute=mode)
    assert outs[True][4]["recomputed"] > 0
    assert outs[True][0] == outs[False][0]
    for n in outs[False][1]:
        assert torch.equal(outs[True][1][n], outs[False][1][n]), n


def test_dropout_in_an_adapter_keeps_the_usual_path():
    need_fp4()
    torch.manual_seed(0)
    model = tiny_lora(dropout=0.1, dtype=bf)
    out = train("nvfp4", model=model, mlp_recompute=True)
    assert out[4]["recomputed"] == 0  # a second run would draw another dropout mask


def test_gradients_reach_the_input_and_the_adapters():
    """One MLP, called directly: input and parameter gradients against the same module with the
    outputs kept."""
    need_fp4()
    model = tiny_lora(dtype=bf)
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize="nvfp4", kernels=True,
                                  fused_lora=True, profile=False, mlp_recompute=False)
    try:
        session = trainstream.wrapper.unwrap_session(model)
        mlp = session.fused.mlps[0]
        if session.pager is not None:
            session.pager.load(0)
            session.pager.wait(0)
        grads = {}
        for label, policy in (("kept", None), ("again", fused.FrontPolicy(True, False))):
            if policy is not None:
                mlp._ts_front = policy
            x = torch.randn(2, 9, 128, device=DEV, generator=torch.Generator(device=DEV).manual_seed(1)).to(bf)
            x.requires_grad_()
            y = mlp(x)
            # (autograd.grad: the wrapped optimizer streams .grad to the host as backward() fills it)
            trained = [p for p in mlp.parameters() if p.requires_grad]
            grads[label] = torch.autograd.grad(y.float().square().sum(), [x] + trained)
            mlp.__dict__.pop("_ts_front", None)
        assert len(grads["kept"]) == 1 + 6  # x, and A / B of gate, up and down
        for a, b in zip(grads["kept"], grads["again"]):
            assert torch.equal(a, b)
    finally:
        trainstream.unwrap(model)


def test_options_are_checked():
    for bad in (dict(mlp_recompute="yes"), dict(fused_gemm="auto")):
        m = tiny_lora(dtype=bf)
        with pytest.raises(ValueError, match="mlp_recompute|fused_gemm"):
            trainstream.wrap(m, torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]), profile=False, **bad)
