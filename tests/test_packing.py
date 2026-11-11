"""Padding-free packing (trainstream.packing): layout, and the same loss and gradients as unpacked."""

import pytest
import torch

from trainstream.packing import IGNORE, PackedCollator, best_fit, pack_sequences


def test_best_fit_fills_rows_longest_first():
    lengths = [5, 3, 4, 2, 6]
    rows = best_fit(lengths, 8)  # 6 | 5 | 4, then 3 into the 5's row, 2 into the 6's
    assert sorted(sum(lengths[i] for i in r) for r in rows) == [4, 8, 8]
    assert sorted(i for r in rows for i in r) == [0, 1, 2, 3, 4]


def test_layout_positions_restart_and_document_starts_are_not_trained():
    seqs = [[11, 12, 13], [21, 22], [31, 32, 33, 34, 35, 36, 37, 38, 39]]  # the last is split at max_len
    out = pack_sequences(seqs, 6, pad_id=0)
    ids, pos, lab = out["input_ids"], out["position_ids"], out["labels"]
    assert ids.shape == pos.shape == lab.shape and ids.shape[1] == 6
    assert int(out["num_tokens"]) == 14
    for r in range(ids.shape[0]):
        starts = [0] + [c for c in range(1, 6) if pos[r, c] != pos[r, c - 1] + 1]
        for c in starts:
            assert lab[r, c] == IGNORE
        for c in range(6):
            if ids[r, c] != 0 and c not in starts:
                assert lab[r, c] == ids[r, c]
    assert (lab == IGNORE).sum() == 4 + (ids == 0).sum()  # 4 documents (one split) and the padding
    real = sorted(t for t in ids.reshape(-1).tolist() if t)
    assert real == sorted(sum(seqs, []))


def test_collator_keeps_prompt_masks():
    feats = [{"input_ids": [1, 2, 3, 4], "labels": [IGNORE, IGNORE, 3, 4]}, {"input_ids": [5, 6]}]
    out = PackedCollator()(feats)
    assert out.pop("use_cache") is False
    assert out["input_ids"].shape[1] == 4  # rows as long as the longest example
    pairs = {(int(i), int(l)) for i, l in zip(out["input_ids"].reshape(-1), out["labels"].reshape(-1))}
    assert (3, 3) in pairs and (4, 4) in pairs and (2, IGNORE) in pairs and (6, 6) in pairs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_batch_matches_the_documents_run_one_by_one():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).cuda()
    g = torch.Generator().manual_seed(1)
    docs = [torch.randint(1, 128, (n,), generator=g) for n in (17, 5, 30, 9, 12)]
    batch = pack_sequences(docs, 32)
    assert batch["input_ids"].shape[0] < len(docs)  # several documents share a row
    out = model(input_ids=batch["input_ids"].cuda(), position_ids=batch["position_ids"].cuda(),
                labels=batch["labels"].cuda(), use_cache=False)  # the packed mask needs no cache
    out.loss.backward()
    packed_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    total, n = 0.0, sum(len(d) - 1 for d in docs)
    for d in docs:
        ids = d[None].cuda()
        total = total + model(input_ids=ids, labels=ids, use_cache=False).loss * (len(d) - 1) / n
    total.backward()
    assert torch.allclose(out.loss, total, rtol=1e-4, atol=1e-5)
    for a, p in zip(packed_grads, model.parameters()):
        assert torch.allclose(a, p.grad, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trainer_packs_when_asked(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
    from trainstream.integrations.huggingface import TrainStreamTrainer

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).cuda()
    g = torch.Generator().manual_seed(2)
    data = [{"input_ids": torch.randint(1, 128, (int(n),), generator=g).tolist()}
            for n in torch.randint(4, 40, (16,), generator=g)]
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=4, max_steps=3,
                             report_to="none", save_strategy="no", remove_unused_columns=False, disable_tqdm=True)
    trainer = TrainStreamTrainer(model=model, args=args, train_dataset=data, trainstream={"packing": 64})
    assert isinstance(trainer.data_collator, PackedCollator)
    trainer.train()
    assert trainer.state.global_step == 3
