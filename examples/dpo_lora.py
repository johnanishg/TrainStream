"""DPO with LoRA under TrainStream: one base model serves as policy and reference.

    python examples/dpo_lora.py                                   # Qwen2.5-0.5B shape, random weights
    python examples/dpo_lora.py --model Qwen/Qwen2.5-7B --quantize nf4 --steps 20

The reference log-probabilities come from the same (paged, quantized) base
with the LoRA adapters switched off, computed once before training
(``reference_logprobs``); the policy's come from the final hidden states in
chunks (``causal_lm_logprobs``), so the four [batch, seq, vocab] logit tensors
of a dense DPO step never exist. The preference pairs here are random token
sequences: this shows the mechanics and the memory, not a useful model.
"""

import argparse
import tempfile
import time

import torch

import trainstream
from trainstream.integrations.huggingface import causal_lm_logprobs, load_model, reference_logprobs, wrap_hf

GB = float(1 << 30)


def pairs(vocab, n, batch, prompt, answer, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        p = torch.randint(0, vocab, (batch, prompt), generator=g)
        rows = []
        for _ in range(2):  # chosen, rejected: the same prompt, different answers
            ids = torch.cat([p, torch.randint(0, vocab, (batch, answer), generator=g)], 1)
            labels = ids.clone()
            labels[:, :prompt] = -100  # only the answer counts
            rows.append({"input_ids": ids.cuda(), "labels": labels.cuda()})
        out.append(tuple(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="a Hugging Face checkpoint (default: random Qwen2.5-0.5B shape)")
    ap.add_argument("--quantize", default="auto", choices=["auto", "none", "int8", "nf4", "fp8", "nvfp4"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--answer", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    import peft

    src = args.model
    if src is None:
        from transformers import Qwen2Config
        src = tempfile.mkdtemp(prefix="ts_dpo_")
        Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                    num_attention_heads=14, num_key_value_heads=2, tie_word_embeddings=True).save_pretrained(src)
    model = load_model(src, quantize=args.quantize, random_weights=args.model is None)
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    model, opt = wrap_hf(model, torch.optim.AdamW(params, lr=5e-5), device="cuda", quantize=args.quantize)
    data = pairs(model.config.vocab_size, args.steps, args.batch, args.prompt, args.answer)

    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = reference_logprobs(model, [b for pair in data for b in pair])
    print(f"reference log-probs of {len(ref)} sequences in {time.perf_counter() - t0:.1f} s "
          f"(adapters off, same base)", flush=True)

    for step, (chosen, rejected) in enumerate(data):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pc = causal_lm_logprobs(model, chosen["input_ids"], chosen["labels"])
            pr = causal_lm_logprobs(model, rejected["input_ids"], rejected["labels"])
        b = args.batch
        rc, rr = ref[2 * step * b:(2 * step + 1) * b], ref[(2 * step + 1) * b:(2 * step + 2) * b]
        loss, cr, rj = trainstream.dpo_loss(pc, pr, rc, rr, beta=args.beta)
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        print(f"step {step + 1}: loss {loss.item():.4f}  margin {(cr - rj).mean().item():+.4f}  "
              f"{time.perf_counter() - t0:.2f} s  peak VRAM {torch.cuda.max_memory_allocated() / GB:.2f} GB",
              flush=True)
    for e in trainstream.memory_report(model)["events"]:
        print("  -", e)


if __name__ == "__main__":
    main()
