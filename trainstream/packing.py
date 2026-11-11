"""Padding-free batches: variable-length sequences packed into rows of one length.

A batch of documents padded to its longest one spends compute and memory on
padding: with lengths spread from 64 to 2,048 tokens, most of it. Packing puts
several documents in one row instead, best-fit decreasing (the longest first,
each into the row with the least room that still takes it), so rows come out
nearly full. Each row carries position_ids that restart at every document;
Hugging Face models read those as document boundaries and keep attention
inside each document (transformers' packed-sequence mask, which it applies
only without a KV cache: call the model with use_cache=False, as
PackedCollator's batches do). The label of each
document's first token is -100, so no document is trained to continue the one
before it.

    batch = pack_sequences(token_lists, max_len=2048)   # input_ids, position_ids, labels
    batch.pop("num_tokens"); model(**batch, use_cache=False)
    TrainStreamTrainer(..., trainstream={"packing": 2048})  # or True: rows as long as the longest

Sequences longer than max_len are split into max_len pieces, each its own document.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

IGNORE = -100


def _ids(x) -> List[int]:
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).tolist()
    return list(x)


def best_fit(lengths: Sequence[int], max_len: int) -> List[List[int]]:
    """Rows of indices into `lengths`: longest first, each into the fullest row it still fits."""
    rows: List[List[int]] = []
    room: List[int] = []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        n = lengths[i]
        best = min((r for r in range(len(rows)) if room[r] >= n), key=lambda r: room[r], default=None)
        if best is None:
            rows.append([i])
            room.append(max_len - n)
        else:
            rows[best].append(i)
            room[best] -= n
    return rows


def pack_sequences(sequences, max_len: int, *, labels=None, pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pack token sequences (lists or 1-D tensors) into rows of `max_len`.

    labels: per-sequence labels of the same lengths (e.g. -100 over a prompt); default the
    tokens themselves. Returns input_ids, position_ids and labels, each [rows, max_len]
    (int64), and num_tokens, the real (unpadded) token count. Padding at a row's end has
    label -100 and positions that continue the last document's (it attends only to that
    document, and nothing attends to it)."""
    if max_len < 1:
        raise ValueError(f"max_len must be positive, got {max_len}")
    docs, labs = [], []
    for k, s in enumerate(sequences):
        ids = _ids(s)
        lab = _ids(labels[k]) if labels is not None else list(ids)
        if len(lab) != len(ids):
            raise ValueError(f"sequence {k}: {len(ids)} tokens but {len(lab)} labels")
        for a in range(0, len(ids), max_len):  # longer than a row: pieces, each a document
            if ids[a:a + max_len]:
                docs.append(ids[a:a + max_len])
                labs.append(lab[a:a + max_len])
    plan = best_fit([len(d) for d in docs], max_len) if docs else []
    n_rows = max(len(plan), 1)
    input_ids = torch.full((n_rows, max_len), pad_id, dtype=torch.long)
    position_ids = torch.zeros((n_rows, max_len), dtype=torch.long)
    out_labels = torch.full((n_rows, max_len), IGNORE, dtype=torch.long)
    for r, members in enumerate(plan):
        at = 0
        for i in members:
            n = len(docs[i])
            input_ids[r, at:at + n] = torch.tensor(docs[i])
            position_ids[r, at:at + n] = torch.arange(n)
            lab = torch.tensor(labs[i])
            lab[0] = IGNORE  # not trained to follow the previous document
            out_labels[r, at:at + n] = lab
            at += n
        if at < max_len:  # padding: positions go on from the last document's
            last = position_ids[r, at - 1].item() if at else -1
            position_ids[r, at:] = torch.arange(last + 1, last + 1 + max_len - at)
    return dict(input_ids=input_ids, position_ids=position_ids, labels=out_labels,
                num_tokens=torch.tensor(sum(len(d) for d in docs)))


class PackedCollator:
    """A data collator for Hugging Face Trainer: the examples of a batch (each with input_ids,
    optionally labels) packed into rows (see pack_sequences). max_len None: rows as long as
    the batch's longest example, so none is split. The number of rows varies with the batch."""

    def __init__(self, max_len: Optional[int] = None, pad_id: int = 0):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, features):
        seqs = [f["input_ids"] for f in features]
        labels = [f.get("labels", f["input_ids"]) for f in features]
        longest = max(len(_ids(s)) for s in seqs)
        out = pack_sequences(seqs, self.max_len or longest, labels=labels, pad_id=self.pad_id)
        out.pop("num_tokens")
        out["use_cache"] = False  # transformers keeps attention per document only without a cache
        return out
