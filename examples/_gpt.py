"""Byte-level GPT and a real text corpus shared by the examples.

The corpus is the Python standard library's own source code, read as bytes:
real, structured text that ships with every Python install, so runs show
genuine learning curves without downloading anything.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    seq_len: int = 512
    vocab: int = 256  # raw bytes

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "tiny-20M": GPTConfig(n_layer=6, n_head=8, d_model=512),
    "small-85M": GPTConfig(n_layer=12, n_head=12, d_model=768),
    "medium-300M": GPTConfig(n_layer=24, n_head=16, d_model=1024),
    "large-680M": GPTConfig(n_layer=24, n_head=16, d_model=1536),
    "xl-1.2B": GPTConfig(n_layer=24, n_head=16, d_model=2048),
    "xxl-1.8B": GPTConfig(n_layer=36, n_head=16, d_model=2048),
    "3.6B": GPTConfig(n_layer=32, n_head=24, d_model=3072),
}


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc_out = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layer) if name.endswith(("proj.weight", "fc_out.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, targets):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))


class ByteCorpus:
    """Python stdlib source as one uint8 tensor on the GPU."""

    def __init__(self, device, max_bytes: int = 24 << 20):
        root = os.path.join(sys.base_prefix, "Lib" if sys.platform == "win32" else "lib")
        if sys.platform != "win32":
            root = os.path.join(root, f"python{sys.version_info[0]}.{sys.version_info[1]}")
        chunks, total = [], 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in ("site-packages", "test", "tests", "__pycache__"))
            for fn in sorted(f for f in filenames if f.endswith(".py")):
                try:
                    with open(os.path.join(dirpath, fn), "rb") as fh:
                        chunks.append(fh.read())
                except OSError:
                    continue
                total += len(chunks[-1])
                if total >= max_bytes:
                    break
            if total >= max_bytes:
                break
        if not chunks:
            raise RuntimeError(f"no Python sources found under {root}")
        self.data = torch.frombuffer(bytearray(b"\n".join(chunks)), dtype=torch.uint8).to(device)

    def batch(self, batch: int, seq: int, generator=None):
        ix = torch.randint(len(self.data) - seq - 1, (batch,), device=self.data.device, generator=generator)
        chunk = self.data[ix[:, None] + torch.arange(seq + 1, device=self.data.device)].long()
        return chunk[:, :-1], chunk[:, 1:]
