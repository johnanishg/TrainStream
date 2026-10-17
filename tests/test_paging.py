"""Parameter paging: block weights live in host RAM and are streamed in per block."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import trainstream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
HP = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab=128, d=128, n_layer=4, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt and self.training else b(x)
        return self.head(self.ln(x))


def tokens(n, seed=3, batch=8, seq=64, vocab=128):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (batch, seq + 1), generator=g).to(DEV) for _ in range(n)]


def lm_loss(model, t):
    logits = model(t[:, :-1])
    return nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))


def train(model, opt, data):
    out = []
    for t in data:
        loss = lm_loss(model, t)
        loss.backward()
        opt.step()
        opt.zero_grad()
        out.append(loss.item())
    return out


def max_diff(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return max((sa[k].float().cpu() - sb[k].float().cpu()).abs().max().item() for k in sa)


def paged_storage_bytes(model):
    return [sum(p.untyped_storage().nbytes() for p in b.parameters()) for b in model.blocks]


@pytest.mark.parametrize("act", [False, 1.0])
def test_paged_training_matches_torch_adamw(act):
    torch.manual_seed(0)
    ref = TinyLM().to(DEV)
    mine = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 param_offload=True, activation_offload=act, min_offload_bytes=0)
    try:
        data = tokens(8)
        losses_ref = train(ref, opt_ref, data)
        losses = train(mine, opt, data)
        opt.synchronize()
        assert losses == pytest.approx(losses_ref, abs=1e-4)
        stats = trainstream.memory_report(mine)
        assert stats["params"]["paged_blocks"] == [0, 1, 2, 3]
        assert stats["params"]["loads"] >= 8 * 7  # every block loaded for forward and again for backward
        # between steps only block 0 (first in the next forward) still holds weights
        sizes = paged_storage_bytes(mine)
        assert sizes[1:] == [0, 0, 0]
        assert max_diff(ref, mine) < 2e-4  # state_dict() serves paged weights from host RAM
        if act:
            assert stats["activations"]["packed_bytes"] > 0
    finally:
        trainstream.unwrap(mine)
    assert all(s > 0 for s in paged_storage_bytes(mine))  # unwrap loads every block back
    assert max_diff(ref, mine) < 2e-4


def test_model_built_on_cpu_never_holds_all_block_weights_in_vram():
    torch.manual_seed(0)
    cpu_model = TinyLM(d=256, n_layer=6)
    ref = copy.deepcopy(cpu_model).to(DEV)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    block_bytes = sum(p.numel() * 4 for p in cpu_model.blocks.parameters())
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    mine, opt = trainstream.wrap(cpu_model, torch.optim.AdamW(cpu_model.parameters(), **HP), device=DEV,
                                 offload=True, param_offload=True, activation_offload=False)
    try:
        resident = torch.cuda.memory_allocated() - before
        assert resident < 0.2 * block_bytes  # only embeddings, final norm and head went to the GPU
        data = tokens(6)
        losses_ref = train(ref, opt_ref, data)
        losses = train(mine, opt, data)
        opt.synchronize()
        assert losses == pytest.approx(losses_ref, abs=1e-4)
        assert max_diff(ref, mine) < 2e-4
    finally:
        trainstream.unwrap(mine)


def test_frozen_weights_in_paged_blocks_stay_frozen():
    torch.manual_seed(0)
    ref = TinyLM().to(DEV)
    for m in (ref,):
        for b in m.blocks:
            b.mlp[0].weight.requires_grad_(False)
    mine = copy.deepcopy(ref)
    frozen = [b.mlp[0].weight.detach().clone() for b in ref.blocks]
    opt_ref = torch.optim.AdamW([p for p in ref.parameters() if p.requires_grad], foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW([p for p in mine.parameters() if p.requires_grad], **HP),
                                 offload=True, param_offload=True, activation_offload=False)
    try:
        data = tokens(5)
        assert train(mine, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
        opt.synchronize()
        sd = mine.state_dict()
        for i, w in enumerate(frozen):
            assert torch.equal(sd[f"blocks.{i}.mlp.0.weight"].to(DEV), w)
    finally:
        trainstream.unwrap(mine)


def test_paging_with_activation_checkpointing():
    torch.manual_seed(0)
    ref = TinyLM(ckpt=True).to(DEV)
    mine = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 param_offload=True, activation_offload=False)
    try:
        data = tokens(5)
        assert train(mine, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
        opt.synchronize()
        assert max_diff(ref, mine) < 2e-4
    finally:
        trainstream.unwrap(mine)


def test_load_state_dict_into_paged_model_then_resume():
    torch.manual_seed(0)
    ref = TinyLM().to(DEV)
    mine = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 param_offload=True, activation_offload=False)
    try:
        data = tokens(8)
        train(ref, opt_ref, data[:3])
        train(mine, opt, data[:3])
        # overwrite the paged model's weights and optimizer state with the reference's
        mine.load_state_dict(ref.state_dict())
        opt.load_state_dict(opt_ref.state_dict())
        assert max_diff(ref, mine) == 0.0
        assert train(mine, opt, data[3:]) == pytest.approx(train(ref, opt_ref, data[3:]), abs=1e-4)
        opt.synchronize()
        assert max_diff(ref, mine) < 2e-4
    finally:
        trainstream.unwrap(mine)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paged_is_bit_identical_to_unpaged(dtype):
    torch.manual_seed(0)
    base = TinyLM().to(DEV).to(dtype)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    a, opt_a = trainstream.wrap(a, torch.optim.AdamW(a.parameters(), **HP), offload=True, param_offload=False,
                                activation_offload=False)
    b, opt_b = trainstream.wrap(b, torch.optim.AdamW(b.parameters(), **HP), offload=True, param_offload=True,
                                activation_offload=False)
    try:
        data = tokens(5)
        la, lb = train(a, opt_a, data), train(b, opt_b, data)
        opt_a.synchronize()
        opt_b.synchronize()
        assert la == lb  # paging moves bytes, it does not change the math
        assert max_diff(a, b) == 0.0
    finally:
        trainstream.unwrap(a)
        trainstream.unwrap(b)


def test_paging_requires_cpu_mode():
    model = TinyLM().to(DEV)
    with pytest.raises(ValueError, match="cpu"):
        trainstream.wrap(model, optimizer_mode="gpu", param_offload=True)


def test_pooled_gradient_staging_under_a_tiny_cap():
    torch.manual_seed(0)
    ref = TinyLM().to(DEV)
    mine = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **HP)
    mine, opt = trainstream.wrap(mine, torch.optim.AdamW(mine.parameters(), **HP), offload=True,
                                 param_offload=True, activation_offload=False)
    opt.engine.set_max_staging_bytes(1)  # one gradient in flight at a time
    try:
        data = tokens(5)
        assert train(mine, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
        opt.synchronize()
        s = opt.stats()
        biggest = max(p.numel() * p.element_size() for p in mine.parameters())
        assert s["staging_bytes"] == 0  # every buffer went back to the pool
        assert 0 < s["staging_peak_bytes"] <= biggest
        # no per-parameter staging in the persistent host state any more
        numel = sum(p.numel() for p in mine.parameters())
        assert s["host_state_bytes"] == numel * 12  # fp32 master + two fp32 moments
    finally:
        trainstream.unwrap(mine)


def test_frozen_weights_beyond_the_pin_budget_stay_pageable(monkeypatch):
    import trainstream.paging as paging
    monkeypatch.setattr(paging, "physical_ram_bytes", lambda: 1)  # pin budget ~0
    torch.manual_seed(0)
    cpu_model = TinyLM()
    for b in cpu_model.blocks:
        for p in b.parameters():
            p.requires_grad_(False)  # LoRA-like: frozen block weights
    ref = copy.deepcopy(cpu_model).to(DEV)
    trainable = lambda m: [p for p in m.parameters() if p.requires_grad]
    opt_ref = torch.optim.AdamW(trainable(ref), foreach=False, **HP)
    mine, opt = trainstream.wrap(cpu_model, torch.optim.AdamW(trainable(cpu_model), **HP), device=DEV,
                                 offload=True, param_offload=True, activation_offload=False)
    try:
        params = trainstream.memory_report(mine)["params"]
        assert params["frozen_pinned_bytes"] == 0 and params["frozen_pageable_bytes"] > 0
        data = tokens(5)
        assert train(mine, opt, data) == pytest.approx(train(ref, opt_ref, data), abs=1e-4)
    finally:
        trainstream.unwrap(mine)
