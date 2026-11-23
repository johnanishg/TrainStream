"""The fused LoRA path (trainstream.lora) against PEFT's own forward and backward."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import trainstream  # noqa: E402
from trainstream import _C, hooks, lora, lowp  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_lora(seed=0, dropout=0.0, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = LlamaForCausalLM(cfg).cuda().to(dtype)
    model = peft.get_peft_model(base, peft.LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS,
                                                      lora_dropout=dropout))
    for n, p in model.named_parameters():  # non-zero B, so every gradient path is exercised
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    return model


def lora_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_matches_peft_forward_and_gradients(autocast):
    ref = tiny_lora()
    mine = copy.deepcopy(ref)
    layers = lora.install(mine)
    assert len(layers) == 2 * len(TARGETS)
    ids = torch.randint(0, 256, (2, 21), device="cuda")
    outs = []
    for m in (ref, mine):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        outs.append(out)
    tol = dict(rtol=2e-2, atol=2e-2) if autocast else dict(rtol=1e-4, atol=1e-5)
    assert torch.allclose(outs[0].logits.float(), outs[1].logits.float(), **tol)
    g_ref, g_mine = lora_grads(ref), lora_grads(mine)
    assert g_ref.keys() == g_mine.keys()
    for n in g_ref:
        scale = g_ref[n].abs().max().item() + 1e-6
        assert (g_ref[n] - g_mine[n]).abs().max().item() <= (5e-2 if autocast else 1e-4) * scale, n
    lora.uninstall(layers)
    assert all("forward" not in m.__dict__ for m in mine.modules())


def test_layers_it_cannot_compute_keep_peft():
    ref = tiny_lora(dropout=0.1)
    mine = copy.deepcopy(ref)
    lora.install(mine)
    ref.eval()
    mine.eval()  # dropout off in eval: the fused path takes over
    ids = torch.randint(0, 256, (1, 9), device="cuda")
    with torch.no_grad():
        assert torch.allclose(ref(input_ids=ids).logits, mine(input_ids=ids).logits, atol=1e-5)
    ref.train()
    mine.train()  # dropout active: PEFT's forward, unchanged
    torch.manual_seed(1)
    a = ref(input_ids=ids).logits
    torch.manual_seed(1)
    b = mine(input_ids=ids).logits
    assert torch.equal(a, b)
    with mine.disable_adapter():  # adapters off: the base alone, as PEFT does it
        with torch.no_grad():
            base = mine(input_ids=ids).logits
    with ref.disable_adapter():
        with torch.no_grad():
            assert torch.equal(base, ref(input_ids=ids).logits)


@pytest.mark.parametrize("kind", ["int8", "nf4", "fp8", "nvfp4"])
def test_trains_like_peft_on_every_quantized_base(kind):
    if kind in ("fp8", "nvfp4") and not lowp.native_supported(kind, "cuda"):
        pytest.skip(f"this GPU has no {kind} tensor-core matmul")
    losses = {}
    for fused_lora in (False, True):
        model = tiny_lora(dtype=torch.bfloat16)
        params = [p for p in model.parameters() if p.requires_grad]
        model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize=kind, kernels=False,
                                      fused_lora=fused_lora, profile=False)
        try:
            session = trainstream.wrapper.unwrap_session(model)
            assert bool(session.lora_layers) == fused_lora
            out = []
            g = torch.Generator().manual_seed(3)
            for _ in range(4):
                ids = torch.randint(0, 256, (2, 33), generator=g).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                out.append(loss.item())
            losses[fused_lora] = out
        finally:
            trainstream.unwrap(model)
    assert losses[True] == pytest.approx(losses[False], rel=1e-2, abs=1e-2)


# ------------------------------------------------- an input that is still arriving from host RAM
def test_chunk_plan_splits_large_activations_into_row_ranges():
    plan = hooks.chunk_plan((16, 1024, 4096), 2, "auto")  # 128 MiB: 8 pieces of 16 MiB
    assert len(plan) == 8 and plan[-1] == 16 * 1024 and plan == sorted(set(plan))
    assert hooks.chunk_plan((4, 256, 1024), 2, "auto") is None  # 2 MiB: one copy
    assert len(hooks.chunk_plan((8, 512, 1024), 2, "auto")) == 2  # 8 MiB: two pieces of 4 MiB
    assert hooks.chunk_plan((2048, 1024), 2, 3) == [682, 1365, 2048]
    assert hooks.chunk_plan((3, 1 << 20), 2, 8) == [1, 2, 3]  # never more pieces than rows
    for off in (False, None, 0, 1):
        assert hooks.chunk_plan((16, 1024, 4096), 2, off) is None
    assert hooks.chunk_plan((1 << 22,), 2, 4) is None  # no rows to split
    assert hooks.chunk_plan((512, 1024), 2, 4) is None  # 1 MiB: too small for an explicit count


def test_chunked_copy_gives_an_event_per_piece():
    src = torch.randn(64, 4096).to(torch.bfloat16)
    host = _C.pinned_empty([64 * 4096], torch.bfloat16, "test", False, True)
    host.copy_(src.view(-1))
    row = 4096 * 2
    before = dict(_C.transfer_stats(0))
    t, events = _C.h2d_alloc_chunked_async(host, None, 0, [4, 16, 4096], [16 * row, 40 * row, 64 * row])
    assert len(events) == 3 and tuple(t.shape) == (4, 16, 4096)
    events[0].wait()  # the compute stream waits for the first 16 rows only
    first = t.view(64, 4096)[:16].clone()
    events[-1].wait()
    assert torch.equal(first.cpu(), src[:16]) and torch.equal(t.view(64, 4096).cpu(), src)
    after = dict(_C.transfer_stats(0))
    assert after["bytes_h2d"] - before["bytes_h2d"] == 64 * row and after["copies_h2d"] - before["copies_h2d"] == 1
    with pytest.raises(RuntimeError, match="last bound"):
        _C.h2d_alloc_chunked_async(host, None, 0, [64, 4096], [100])
    with pytest.raises(RuntimeError, match="must increase"):
        _C.h2d_alloc_chunked_async(host, None, 0, [64, 4096], [32 * row, 16 * row, 64 * row])
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_gradient_summed_over_pieces_matches_one_matmul(dtype):
    torch.manual_seed(0)
    gb = torch.randn(1000, 8, device="cuda").to(dtype)
    x = torch.randn(1000, 512, device="cuda").to(dtype)
    done = _C.record_event(0)
    chunks = [(1, done), (333, done), (999, done), (1000, done)]
    got = lora.grad_a_chunked(gb, x, chunks, dtype)
    want = gb.float().t() @ x.float()
    assert got.dtype == torch.float32 and got.shape == (8, 512)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert (got - want).abs().max().item() <= tol * want.abs().max().item()


def run_paged_lora(lora_chunks, steps=3):
    model = tiny_lora()
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), offload=True, activation_offload=True,
                                  kernels=False, fused_lora=True, lora_chunks=lora_chunks, profile=False)
    try:
        g = torch.Generator().manual_seed(5)
        losses = []
        for _ in range(steps):
            ids = torch.randint(0, 256, (32, 128), generator=g).cuda()  # x of a block: 2 MiB
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        opt.synchronize()
        report = trainstream.memory_report(model)["activations"]
        weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        return losses, weights, report
    finally:
        trainstream.unwrap(model)


def test_backward_starts_on_the_rows_that_have_landed(monkeypatch):
    plain_losses, plain_w, plain_rep = run_paged_lora(False)
    assert plain_rep["packed_tensors"] > 0  # activations really were paged to host RAM
    before = dict(hooks.CHUNK_STATS)
    assert plain_rep["chunked"] == before

    monkeypatch.setattr(lora, "CHUNK_ALWAYS", True)  # as if every input were still crossing PCIe
    losses, w, rep = run_paged_lora(4)
    assert rep["chunked"]["tensors"] > before["tensors"]
    assert rep["chunked"]["pieces"] - before["pieces"] == 4 * (rep["chunked"]["tensors"] - before["tensors"])
    assert rep["chunked"]["overlapped"] > before["overlapped"]
    assert losses == pytest.approx(plain_losses, rel=1e-5, abs=1e-6)
    for n in plain_w:  # summed piece by piece or in one matmul: the same training
        assert (w[n] - plain_w[n]).abs().max().item() <= 1e-5 * (plain_w[n].abs().max().item() + 1e-3), n

    monkeypatch.setattr(lora, "CHUNK_ALWAYS", False)  # normal use: pieces, but one matmul once all have landed
    mid = dict(hooks.CHUNK_STATS)
    losses2, _, rep2 = run_paged_lora("auto")
    assert losses2 == pytest.approx(plain_losses, rel=1e-5, abs=1e-6)
    assert rep2["chunked"]["tensors"] == mid["tensors"]  # 2 MiB is below "auto"'s 4 MiB pieces: one copy
