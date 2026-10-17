"""Train a GPT whose weights do not fit in (a capped amount of) VRAM.

    python examples/train_bigger_than_vram.py                       # 1.8B params (7.3 GB fp32) in a 4 GB cap
    python examples/train_bigger_than_vram.py --layers 12 --vram-gb 2

The model is built on the CPU and handed to trainstream.wrap(device="cuda",
param_offload=True): every transformer block's weights go straight to host RAM
and are streamed onto the GPU only while that block runs (forward and
backward). Optimizer state and gradients live in host RAM as usual, and
activations are paged automatically. ``--vram-gb`` caps PyTorch's allocator
(torch.cuda.set_per_process_memory_fraction) to show the run staying inside a
GPU smaller than the model.

Host RAM needed: about 16 bytes per parameter with fp32 moments, 12 with
``--moments bf16`` (fp32 master + gradient staging + moments).
"""

import argparse
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

import trainstream


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.qkv(self.ln1(x)).split(d, dim=2))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, layers, d, heads, vocab=256, seq=512):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * layers))

    def forward(self, idx, targets):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return F.cross_entropy(self.head(self.ln(x)).float().flatten(0, 1), targets.flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--vram-gb", type=float, default=4.0)
    ap.add_argument("--moments", default="bf16", choices=["fp32", "bf16"])
    args = ap.parse_args()

    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 1e9 / total), dev)

    torch.manual_seed(0)
    model = GPT(args.layers, args.d_model, args.heads, seq=args.seq)  # on the CPU
    n = sum(p.numel() for p in model.parameters())
    print(f"{n / 1e9:.2f}B parameters = {n * 4 / 1e9:.1f} GB of fp32 weights; VRAM cap {args.vram_gb:.1f} GB")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    model, opt = trainstream.wrap(model, opt, device="cuda", param_offload=True,
                                  moment_dtype="bfloat16" if args.moments == "bf16" else "float32")
    pager = trainstream.wrapper.unwrap_session(model).pager
    print(f"paged {len(pager.paged)} of {pager.n} blocks ({pager.bytes_paged / 1e9:.1f} GB of weights in host RAM)")

    data = torch.randint(0, 256, (64, args.seq + 1), device="cuda")
    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        batch = data[torch.randint(0, 64, (args.batch,))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch[:, :-1], batch[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        print(f"step {step:3d}  loss {loss.item():.3f}  {time.perf_counter() - t0:6.2f} s  "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:5.2f} GB")


if __name__ == "__main__":
    main()
