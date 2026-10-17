"""Hugging Face integration: Trainer parity, PEFT/LoRA, checkpoint resume."""

import copy
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments  # noqa: E402

import trainstream  # noqa: E402
from trainstream.integrations.huggingface import TrainStreamTrainer, decoder_layers, wrap_hf  # noqa: E402

VOCAB = 512


def tiny_llama(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).cuda()


def tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=VOCAB, n_embd=128, n_layer=4, n_head=4, n_positions=128)
    return GPT2LMHeadModel(cfg).cuda()


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, n=64, seq=64, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, VOCAB, (n, seq), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"input_ids": self.data[i], "labels": self.data[i]}


def args(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path), per_device_train_batch_size=4, max_steps=8, learning_rate=2e-3,
                lr_scheduler_type="linear", warmup_steps=2, weight_decay=0.1, max_grad_norm=1.0,
                gradient_accumulation_steps=2, logging_steps=1, report_to="none", save_strategy="no",
                seed=0, data_seed=0, dataloader_num_workers=0, optim="adamw_torch", disable_tqdm=True)
    base.update(kw)
    return TrainingArguments(**base)


def losses(trainer):
    return [h["loss"] for h in trainer.state.log_history if "loss" in h]


def max_diff(a, b):
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(a.parameters(), b.parameters()))


@pytest.mark.parametrize("factory, attr", [(tiny_llama, "model.layers"), (tiny_gpt2, "transformer.h")])
def test_decoder_layers_found(factory, attr):
    model = factory()
    expected = model
    for part in attr.split("."):
        expected = getattr(expected, part)
    assert decoder_layers(model) == list(expected)


def test_wrap_hf_manual_loop_matches_torch():
    ref, mine = tiny_llama(), tiny_llama()
    hp = dict(lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    opt_ref = torch.optim.AdamW(ref.parameters(), foreach=False, **hp)
    mine, opt = wrap_hf(mine, torch.optim.AdamW(mine.parameters(), **hp), offload=True, activation_offload=1.0,
                        min_offload_bytes=0, profile=False)
    try:
        data = TokenDataset(n=6)
        for i in range(6):
            batch = data.data[i:i + 1].cuda()
            for model, o in ((ref, opt_ref), (mine, opt)):
                model(input_ids=batch, labels=batch).loss.backward()
                o.step()
                o.zero_grad()
        opt.synchronize()
        assert max_diff(ref, mine) < 2e-5
        stats = trainstream.memory_report(mine)
        assert stats["activations"]["packed_bytes"] > 0  # decoder activations really were paged
        assert stats["optimizer"]["grad_bytes_offloaded"] > 0
    finally:
        trainstream.unwrap(mine)


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_trainer_parity_with_stock_trainer(tmp_path, mode):
    """Same model, data, schedule, clipping and accumulation: TrainStreamTrainer
    must reproduce the stock Trainer step for step."""
    ref_model = tiny_llama()
    ts_model = copy.deepcopy(ref_model)
    ds = TokenDataset()
    ref = Trainer(model=ref_model, args=args(tmp_path / "ref"), train_dataset=ds)
    ref.train()
    ts = TrainStreamTrainer(model=ts_model, args=args(tmp_path / "ts"), train_dataset=ds,
                            trainstream={"offload": True, "optimizer_mode": mode, "activation_offload": False})
    ts.train()
    assert isinstance(ts.ts_optimizer, trainstream.OffloadAdamW)
    assert ts.ts_optimizer.accumulation_steps == 2 and ts.ts_optimizer.max_grad_norm == 1.0
    assert ts.chunked_loss_used == 16  # 8 steps x 2 micro-batches through the chunked loss
    assert losses(ts) == pytest.approx(losses(ref), abs=2e-4)
    assert max_diff(ref_model, ts_model) < 1e-4
    logs = [h for h in ts.state.log_history if "ts_vram_alloc_gb" in h]
    assert logs and "ts_host_state_gb" in logs[-1]
    if mode == "cpu":
        assert any(h.get("ts_d2h_gb", 0) > 0 for h in logs)


def test_trainer_with_activation_paging_matches_stock(tmp_path):
    ref_model = tiny_gpt2()
    model = copy.deepcopy(ref_model)
    ds = TokenDataset()
    kw = dict(max_steps=6, gradient_accumulation_steps=1)
    ref = Trainer(model=ref_model, args=args(tmp_path / "ref", **kw), train_dataset=ds)
    ref.train()
    ts = TrainStreamTrainer(model=model, args=args(tmp_path / "ts", **kw), train_dataset=ds,
                            trainstream={"offload": True, "activation_offload": 1.0, "min_offload_bytes": 0})
    ts.train()
    session = trainstream.wrapper.unwrap_session(model)
    assert session.activations is not None and session.activations.stats["packed_bytes"] > 0
    assert session.activations.stats["prefetch_misses"] == 0
    assert losses(ts) == pytest.approx(losses(ref), abs=2e-4)
    assert max_diff(ref_model, model) < 1e-4


def test_trainer_lora_only_trainable_params_offloaded(tmp_path):
    peft = pytest.importorskip("peft")
    base = tiny_llama()
    model = peft.get_peft_model(base, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                                      lora_dropout=0.0, task_type="CAUSAL_LM"))
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    ts = TrainStreamTrainer(model=model, args=args(tmp_path, max_steps=6, learning_rate=5e-3),
                            train_dataset=TokenDataset(), trainstream={"offload": True})
    ts.train()
    opt = ts.ts_optimizer
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert opt.stats()["numel"] == trainable  # no host state for frozen base weights
    for n, p in model.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), n
    assert any(not torch.equal(p, lora_before[n]) for n, p in model.named_parameters() if n in lora_before)


def test_trainer_checkpoint_resume_matches_uninterrupted(tmp_path):
    ds = TokenDataset()
    full_model = tiny_llama()
    resumed_model = copy.deepcopy(full_model)
    full = TrainStreamTrainer(model=full_model, args=args(tmp_path / "full", max_steps=6), train_dataset=ds,
                              trainstream={"offload": True, "activation_offload": False})
    full.train()

    first = TrainStreamTrainer(model=tiny_llama(), args=args(tmp_path / "part", max_steps=6, save_strategy="steps",
                                                             save_steps=3), train_dataset=ds,
                               trainstream={"offload": True, "activation_offload": False})
    first.train()
    ckpt = os.path.join(tmp_path / "part", "checkpoint-3")
    assert os.path.isfile(os.path.join(ckpt, "optimizer.pt"))

    resumed = TrainStreamTrainer(model=resumed_model, args=args(tmp_path / "part", max_steps=6), train_dataset=ds,
                                 trainstream={"offload": True, "activation_offload": False})
    resumed.train(resume_from_checkpoint=ckpt)
    assert max_diff(full_model, resumed_model) < 1e-5


def test_trainer_rejects_fp16(tmp_path):
    with pytest.raises(ValueError, match="fp16"):
        TrainStreamTrainer(model=tiny_gpt2(), args=args(tmp_path, fp16=True), train_dataset=TokenDataset())


def test_trainer_tied_embeddings_with_param_paging_matches_stock(tmp_path):
    """GPT-2 ties its output layer to the token embedding; with paged decoder
    weights and the chunked loss the run must still equal the stock Trainer."""
    ref_model = tiny_gpt2()
    model = copy.deepcopy(ref_model)
    ds = TokenDataset()
    ref = Trainer(model=ref_model, args=args(tmp_path / "ref"), train_dataset=ds)
    ref.train()
    ts = TrainStreamTrainer(model=model, args=args(tmp_path / "ts"), train_dataset=ds,
                            trainstream={"offload": True, "param_offload": True, "activation_offload": False})
    ts.train()
    session = trainstream.wrapper.unwrap_session(model)
    assert session.pager is not None and len(session.pager.paged) == 4
    assert ts.chunked_loss_used > 0
    assert losses(ts) == pytest.approx(losses(ref), abs=2e-4)
    sd_ref, sd = ref_model.state_dict(), model.state_dict()
    assert max((sd[k].float().cpu() - sd_ref[k].float().cpu()).abs().max().item() for k in sd_ref) < 2e-4


def test_trainer_stock_loss_still_available(tmp_path):
    model = tiny_llama()
    ts = TrainStreamTrainer(model=model, args=args(tmp_path, max_steps=2), train_dataset=TokenDataset(),
                            trainstream={"offload": True, "chunked_loss": False, "activation_offload": False})
    ts.train()
    assert ts.chunked_loss_used == 0 and len(losses(ts)) == 2


def test_trainer_takes_a_cpu_model_and_pages_its_layers(tmp_path):
    """A model handed over on the CPU is placed by wrap(): decoder-layer
    weights stay in host RAM, the rest goes to the GPU, and training matches
    the stock Trainer on a GPU copy."""
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    lora = dict(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM")
    base = LlamaForCausalLM(cfg)  # CPU
    ref_model = peft.get_peft_model(copy.deepcopy(base).cuda(), peft.LoraConfig(**lora))
    model = peft.get_peft_model(base, peft.LoraConfig(**lora))
    model.load_state_dict({k: v.cpu() for k, v in ref_model.state_dict().items()})  # same adapter init
    ds = TokenDataset()
    ref = Trainer(model=ref_model, args=args(tmp_path / "ref", max_steps=6), train_dataset=ds)
    ref.train()
    ts = TrainStreamTrainer(model=model, args=args(tmp_path / "ts", max_steps=6), train_dataset=ds,
                            trainstream={"offload": True, "param_offload": True, "activation_offload": False})
    ts.train()
    session = trainstream.wrapper.unwrap_session(model)
    assert session.pager is not None and len(session.pager.paged) == 4
    assert session.pager.frozen_pinned_bytes + session.pager.frozen_pageable_bytes > 0
    assert losses(ts) == pytest.approx(losses(ref), abs=2e-4)
