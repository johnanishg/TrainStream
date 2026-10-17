"""Byte-level GPT and a real text corpus for the live dashboard.

The corpus is the Python standard library's own source code (a few MB of real,
structured text that ships with every Python install), so runs show genuine
learning curves without downloading anything.
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
    "tiny-20M": GPTConfig(n_layer=6, n_head=8, d_model=512, seq_len=512),
    "small-85M": GPTConfig(n_layer=12, n_head=12, d_model=768, seq_len=512),
    "medium-300M": GPTConfig(n_layer=24, n_head=16, d_model=1024, seq_len=512),
    "large-680M": GPTConfig(n_layer=24, n_head=16, d_model=1536, seq_len=512),
    "xl-1.2B": GPTConfig(n_layer=24, n_head=16, d_model=2048, seq_len=512),
    "xxl-1.8B": GPTConfig(n_layer=36, n_head=16, d_model=2048, seq_len=512),
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
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        t = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 0.8, top_k: int = 40):
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.seq_len:])[:, -1, :].float() / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


def count_params(cfg: GPTConfig) -> int:
    with torch.device("meta"):
        return sum(p.numel() for p in GPT(cfg).parameters())


class ByteCorpus:
    """Python stdlib source as a byte stream, resident on the GPU (uint8)."""

    def __init__(self, device, max_bytes: int = 24 << 20, val_fraction: float = 0.02):
        root = os.path.join(sys.base_prefix, "Lib")
        chunks, total = [], 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in ("site-packages", "test", "tests", "__pycache__"))
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    try:
                        with open(os.path.join(dirpath, fn), "rb") as fh:
                            data = fh.read()
                    except OSError:
                        continue
                    chunks.append(data)
                    total += len(data)
                    if total >= max_bytes:
                        break
            if total >= max_bytes:
                break
        if total == 0:
            raise RuntimeError(f"no Python sources found under {root}")
        blob = b"\n".join(chunks)
        data = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        split = int(len(data) * (1 - val_fraction))
        self.train = data[:split].to(device)
        self.val = data[split:].to(device)
        self.nbytes = len(data)
        self.files = len(chunks)
        self.device = device

    def batch(self, batch: int, seq: int, split: str = "train", generator=None):
        src = self.train if split == "train" else self.val
        ix = torch.randint(0, len(src) - seq - 1, (batch,), device=self.device, generator=generator)
        offs = torch.arange(seq + 1, device=self.device)
        win = src[ix[:, None] + offs].long()
        return win[:, :-1], win[:, 1:]

    @staticmethod
    def decode(t: torch.Tensor) -> str:
        return bytes(t.tolist()).decode("utf-8", errors="replace")
