"""PiSSA (lora_init="pissa"): LoRA adapters start at their base weights' principal part.

The base stores only what is left (quantized), so three things must hold: the
split really is the principal part, base + adapter starts at the original
weight up to the residual's quantization, and a trained adapter can be turned
back into an ordinary LoRA for the original weights.
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    # these tests train fast on purpose (lr 1e-3); the warning itself is tested where it is recorded
    pytest.mark.filterwarnings("ignore:TrainStream. lora_init='pissa' adapters train at learning rate:UserWarning"),
]

import trainstream  # noqa: E402
from trainstream import quant as Q  # noqa: E402

DEV = "cuda"


def weight(n=512, k=1024, seed=0):
    """A matrix shaped like an LLM weight: a few strong directions over broadband noise."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, device=DEV, generator=g) * 0.02
    u, v = torch.randn(n, 6, device=DEV, generator=g), torch.randn(6, k, device=DEV, generator=g)
    return w + (u * torch.tensor([3.0, 2.0, 1.5, 1.0, 0.7, 0.5], device=DEV)) @ v * 0.02


# ------------------------------------------------------------------- the split
def test_split_is_the_principal_part_and_sums_back():
    w = weight()
    stored, a, b = Q.pissa_split(w, "nf4", 64, 16)
    assert a.shape == (16, 1024) and b.shape == (512, 16) and a.dtype == b.dtype == torch.float32
    ba = b.to(DEV) @ a.to(DEV)
    assert torch.allclose(stored.to(DEV) + ba, w, atol=1e-6)  # the residual is exactly what the factors left
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    best = (u[:, :16] * s[:16]) @ vh[:16]  # the optimal rank-16 approximation
    # as good as the exact SVD's (among the near-equal singular values of the noise floor the
    # subspace itself is not unique, so the two are compared by what they leave behind) ...
    assert (w - ba).norm().item() <= 1.005 * (w - best).norm().item()
    # ... and the six strong directions are in it
    strong = (u[:, :6] * s[:6]) @ vh[:6]
    proj = b.to(DEV) @ torch.linalg.lstsq(b.to(DEV), strong).solution
    assert ((proj - strong).norm() / strong.norm()).item() < 1e-3
    # balanced factors: both carry sqrt(S)
    assert torch.allclose(a.to(DEV).norm(dim=1), b.to(DEV).norm(dim=0), rtol=1e-3)


def test_split_is_the_same_every_time():
    w = weight(seed=3)
    _, a1, b1 = Q.pissa_split(w, "nf4", 64, 8)
    torch.manual_seed(1234)  # the global generator plays no part
    torch.randn(7, device=DEV)
    _, a2, b2 = Q.pissa_split(w, "nf4", 64, 8)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)


@pytest.mark.parametrize("kind", ["nf4", "int8", "nvfp4", "fp8"])
def test_residual_quantizes_better_than_the_weight(kind):
    w = weight()
    plain = Q.quant_error(w, kind)
    errs = []
    for iters in (1, 2):
        stored, a, b = Q.pissa_split(w, kind, 64, 16, iters)
        approx = Q.roundtrip(stored.to(DEV), kind) + b.to(DEV) @ a.to(DEV)
        errs.append(((approx - w).norm() / w.norm()).item())
    assert errs[0] < 0.9 * plain  # the strong directions no longer set the block scales
    assert errs[1] <= errs[0] * 1.001  # a second round never hurts


def test_make_quantized_carries_the_factors():
    w = weight(256, 512).to(torch.bfloat16)
    qw = Q.make_quantized(w, "nf4", 64, pin=False, error_rank=8, lora_init="pissa")
    assert qw.lora_init_kind == "pissa" and qw.lora_init[0].shape == (8, 512)
    a, b = qw.lora_init
    back = qw.dequantize(DEV).float() + b.to(DEV) @ a.to(DEV)
    plain = Q.make_quantized(w, "nf4", 64, pin=False).dequantize(DEV).float()
    assert (back - w.float()).norm() < (plain - w.float()).norm()
    with pytest.raises(ValueError, match="lora_init"):
        Q.make_quantized(w, "nf4", 64, pin=False, error_rank=8, lora_init="svd")


def test_targets_match_reads_names_like_peft():
    assert Q.targets_match("model.layers.0.self_attn.q_proj.weight", ["q_proj", "v_proj"])
    assert not Q.targets_match("model.layers.0.self_attn.k_proj.weight", ["q_proj", "v_proj"])
    assert not Q.targets_match("model.layers.0.mlp.xq_proj.weight", ["q_proj"])  # a suffix of the name, not of the path
    assert Q.targets_match("model.layers.0.mlp.up_proj.weight", None)


# --------------------------------------------------------------------- models
def tiny_llama(seed=0):
    """A 2-layer Llama whose matrices, like a trained model's, have a few dominant directions
    (a freshly initialized one is white noise: nothing for a principal part to take out)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for layer in model.model.layers:
            for p in layer.parameters():
                if p.dim() == 2:
                    u, v = torch.randn(p.shape[0], 4, generator=g), torch.randn(4, p.shape[1], generator=g)
                    p.add_((u * torch.tensor([2.0, 1.5, 1.0, 0.7])) @ v * 0.02)
    return model.cuda()


TARGETS = ["q_proj", "v_proj", "down_proj"]


def lora(base, r=8, targets=TARGETS, **kw):
    import peft
    return peft.get_peft_model(base, peft.LoraConfig(r=r, lora_alpha=16, lora_dropout=0.0, target_modules=targets, **kw))


def wrapped(model, **kw):
    params = [p for p in model.parameters() if p.requires_grad]
    return trainstream.wrap(model, torch.optim.AdamW(params, lr=1e-3), quantize="nf4", kernels=False,
                            profile=False, **kw)


def test_wrap_starts_adapters_at_the_principal_part():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    ref = tiny_llama()
    ids = torch.randint(0, 256, (2, 17), device=DEV)
    with torch.no_grad():
        want = ref(ids).logits
    outs = {}
    for init in ("none", "pissa"):
        model = lora(tiny_llama())
        model, opt = wrapped(model, **({"lora_init": "pissa"} if init == "pissa" else {}))
        try:
            session = trainstream.wrapper.unwrap_session(model)
            if init == "pissa":
                assert session.config["lora_init"] == "pissa"
                assert any("PiSSA: 6 LoRA adapters" in e for e in session.events)
                for name, m in model.named_modules():
                    if hasattr(m, "lora_A") and hasattr(m, "base_layer"):
                        start = Q.lora_init_record(m)["default"]
                        assert start["kind"] == "pissa" and start["a"].shape[0] == 8
                        assert Q.quantized_of(m.base_layer.weight).lora_init is None  # handed to the layer
                        assert m.lora_B["default"].weight.abs().max() > 0  # no longer zero
                # matrices without an adapter are quantized whole
                k = model.base_model.model.model.layers[0].self_attn.k_proj.weight
                assert Q.quantized_of(k) is not None and Q.quantized_of(k).lora_init is None
            with torch.no_grad():
                outs[init] = model(ids).logits
        finally:
            trainstream.unwrap(model)
    assert (outs["pissa"] - want).norm() < (outs["none"] - want).norm()


@pytest.mark.parametrize("lr,warns", [(2e-4, True), (2e-5, False)])
def test_wrap_warns_when_pissa_trains_at_a_lora_learning_rate(lr, warns):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import warnings
    model = lora(tiny_llama())
    params = [p for p in model.parameters() if p.requires_grad]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        model, opt = trainstream.wrap(model, torch.optim.AdamW(params, lr=lr), quantize="nf4", kernels=False,
                                      profile=False, lora_init="pissa")
    try:
        hits = [w for w in seen if "lora_init='pissa' adapters train at learning rate" in str(w.message)]
        assert bool(hits) == warns
        if warns:
            assert "0.0002" in str(hits[0].message)
            # points at the caller (normcase: Windows may spell the drive letter c: or C:)
            assert os.path.normcase(hits[0].filename) == os.path.normcase(__file__)
    finally:
        trainstream.unwrap(model)


def test_no_pissa_warning_without_pissa():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import warnings
    model = lora(tiny_llama())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        model, opt = wrapped(model)  # lr 1e-3, LoftQ off: an ordinary LoRA
    try:
        assert not [w for w in seen if "lora_init='pissa'" in str(w.message)]
    finally:
        trainstream.unwrap(model)


def test_wrapping_again_keeps_trained_adapters():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    model = lora(tiny_llama())
    model, opt = wrapped(model, lora_init="pissa")
    a = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"].weight
    with torch.no_grad():
        a.add_(1.0)  # "training"
    opt.synchronize()
    before = a.detach().clone()
    trainstream.unwrap(model)
    for again in ({"lora_init": "pissa"}, {"error_rank": 8}):  # neither a second split nor LoftQ touches it
        model, opt = wrapped(model, **again)
        try:
            assert torch.equal(model.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"].weight, before)
            assert not any("LoRA adapters start" in e for e in trainstream.wrapper.unwrap_session(model).events)
        finally:
            trainstream.unwrap(model)


def test_quantize_model_splits_only_the_named_targets():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = tiny_llama()
    for p in base.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="error_rank"):
        Q.quantize_model(base, "nf4", lora_init="pissa")
    Q.quantize_model(base, "nf4", error_rank=8, lora_init="pissa", lora_targets=TARGETS)
    layer = base.model.layers[0]
    assert Q.quantized_of(layer.self_attn.q_proj.weight).lora_init is not None
    assert Q.quantized_of(layer.self_attn.k_proj.weight).lora_init is None
    model = lora(base)
    model, opt = wrapped(model)  # the factors ride on the weights: wrap needs no lora_init
    try:
        assert any("PiSSA: 6 LoRA adapters" in e for e in trainstream.wrapper.unwrap_session(model).events)
    finally:
        trainstream.unwrap(model)


def test_a_split_weight_without_an_adapter_is_refused():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = tiny_llama()
    for p in base.parameters():
        p.requires_grad_(False)
    Q.quantize_model(base, "nf4", error_rank=8, lora_init="pissa")  # every matrix split ...
    model = lora(base, targets=["q_proj"])  # ... but only q_proj gets an adapter
    with pytest.raises(ValueError, match="no LoRA adapter sits on them"):
        wrapped(model)


def test_a_smaller_adapter_is_refused():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = tiny_llama()
    for p in base.parameters():
        p.requires_grad_(False)
    Q.quantize_model(base, "nf4", error_rank=8, lora_init="pissa", lora_targets=["q_proj"])
    model = lora(base, r=4, targets=["q_proj"])
    with pytest.raises(ValueError, match="split at rank 8.*r=4"):
        wrapped(model)


# ------------------------------------------------------------- pissa_to_lora
def test_trained_adapter_converts_to_a_lora_on_the_original_weights(tmp_path):
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    original = {n: p.detach().clone() for n, p in tiny_llama().named_parameters()}
    model = lora(tiny_llama())
    model, opt = wrapped(model, lora_init="pissa")
    try:
        ids = torch.randint(0, 256, (4, 33), device=DEV)
        for _ in range(4):
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step()
            opt.zero_grad()
        opt.synchronize()
        with torch.no_grad():
            trained = model(ids).logits
        sd = trainstream.pissa_to_lora(model, save_directory=str(tmp_path))
        # matrix identity, layer by layer:  W + s B'A'  ==  (W - s B0 A0) + s B A
        checked = 0
        for name, m in model.named_modules():
            if not (hasattr(m, "lora_A") and hasattr(m, "base_layer")):
                continue
            start = Q.lora_init_record(m)["default"]
            a0, b0 = start["a"], start["b"]
            s = float(m.scaling["default"])
            A, B = m.lora_A["default"].weight.detach().float().cpu(), m.lora_B["default"].weight.detach().float().cpu()
            A2, B2 = sd[f"{name}.lora_A.weight"], sd[f"{name}.lora_B.weight"]
            assert A2.shape[0] == 16 and B2.shape[1] == 16
            w = original[name.replace("base_model.model.", "", 1) + ".weight"].float().cpu()
            assert torch.allclose(w + s * B2 @ A2, (w - b0 @ a0) + s * B @ A, atol=1e-5)
            assert not torch.allclose(B @ A, (b0 / s) @ a0, atol=1e-6)  # training did move it
            checked += 1
        assert checked == 6
    finally:
        trainstream.unwrap(model)
    again = trainstream.pissa_to_lora(model)  # after unwrap too: the layers kept B0 / A0
    assert all(torch.equal(again[k], sd[k]) for k in sd)

    # the saved folder is an ordinary PEFT adapter for the untouched base
    assert os.path.isfile(tmp_path / "adapter_model.safetensors")
    loaded = peft.PeftModel.from_pretrained(tiny_llama(), str(tmp_path))
    cfg = loaded.peft_config["default"]
    assert cfg.r == 16 and cfg.lora_alpha == 32
    with torch.no_grad():
        got = loaded(ids).logits
        plain = tiny_llama()(ids).logits
    # full-precision base + converted adapter vs the trained model (whose base is the nf4 residual):
    # they differ by that quantization only, far less than the adapter's own effect
    assert (got - trained).norm() < 0.5 * (plain - trained).norm()
