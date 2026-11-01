"""Paged KV cache: generation with part of the keys and values in host RAM.

Methods that generate during training (online DPO, GRPO, PPO) keep a KV
cache of 2 x layers x tokens x kv_heads x head_dim values per sequence; for
long answers sampled in big batches it outgrows the VRAM the weights leave.
``PagedKVCache`` keeps as many layers resident as a VRAM budget allows and
pages the others' keys and values to pinned host RAM in pages of
``page_tokens`` tokens (from TrainStream's pinned arena):

  update(layer, k, v)   (offloaded layer)
      the layer's host pages were copied into a device slot ahead of time (on
      TrainStream's H2D stream, while the layers before it ran); the tail that
      has not filled a page yet stays in VRAM; the new tokens are appended and
      the full keys / values returned as a view of the slot. Pages that fill
      up go to host RAM on the D2H stream. Then the next offloaded layer's
      pages start copying into the other slot.

VRAM therefore holds the resident layers plus two (three for an odd number of
offloaded layers) slots of one layer each, instead of every layer.

Ordering (all GPU-side, the host never waits for a copy):
  * a layer's computation waits on the event of the copy that filled its slot;
  * a slot is refilled only after an event recorded on the compute stream
    behind every attention that read it, and after its pages went out (the
    compute stream waits for that D2H first, so one event covers both).

Storage is token-major ([tokens, batch, heads, head_dim]) so pages and slot
ranges are contiguous; the returned keys / values are [batch, heads, tokens,
head_dim] views with those strides, which scaled_dot_product_attention takes
as they are. Inference only (no autograd through the cache). The Hugging Face
adapter is ``trainstream.integrations.huggingface.TrainStreamCache``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from . import _C
from .planner import spread


class PagedKVCache:
    """Per-layer KV storage for autoregressive generation.

    Args:
        num_layers: attention layers of the model.
        max_tokens: the longest sequence (prompt + generated tokens) expected;
            sizes the slots and the "auto" budget. Longer sequences still
            work (buffers grow), at the cost of a reallocation.
        resident_layers: "auto" (default) keeps as many layers in VRAM as fit
            in ``vram_fraction`` of the free VRAM at the first update; an int
            keeps that many; a list keeps those layer indices.
        page_tokens: tokens per host page (the unit of a D2H copy).
        vram_fraction: share of the free (and cached but unused) VRAM the
            "auto" plan may fill with keys and values.
    """

    def __init__(self, num_layers: int, max_tokens: int, *,
                 resident_layers: Union[str, int, Sequence[int]] = "auto", page_tokens: int = 256,
                 vram_fraction: float = 0.5):
        if num_layers < 1 or max_tokens < 1:
            raise ValueError("num_layers and max_tokens must be positive")
        if page_tokens < 1:
            raise ValueError(f"page_tokens must be positive, got {page_tokens}")
        if not (0.0 < vram_fraction <= 1.0):
            raise ValueError(f"vram_fraction must be in (0, 1], got {vram_fraction}")
        if not (resident_layers == "auto" or (isinstance(resident_layers, int) and not isinstance(resident_layers, bool)
                                              and resident_layers >= 0)
                or (isinstance(resident_layers, (list, tuple, set))
                    and all(isinstance(i, int) and 0 <= i < num_layers for i in resident_layers))):
            raise ValueError(f"resident_layers must be 'auto', a count or layer indices, got {resident_layers!r}")
        self.num_layers = int(num_layers)
        self.max_tokens = int(max_tokens)
        self.page = int(page_tokens)
        self.vram_fraction = float(vram_fraction)
        self._resident_arg = resident_layers
        self.ready = False
        self.n = [0] * self.num_layers
        self.resident: List[int] = []
        self.offloaded: List[int] = []
        self.stats = dict(h2d_bytes=0, d2h_bytes=0, pages=0, prefetched=0, gathered=0)

    # -- plan ------------------------------------------------------------------------
    def _choose(self, layer_bytes: int) -> List[int]:
        L = self.num_layers
        arg = self._resident_arg
        if arg == "auto":
            from .wrapper import vram_budget
            free, _ = vram_budget(self.device)
            spare = max(0, torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device))
            budget = self.vram_fraction * (free + spare)
            if L * layer_bytes <= budget:
                return list(range(L))
            r = min(L - 1, int(max(0.0, budget - 3 * layer_bytes) // max(layer_bytes, 1)))
        elif isinstance(arg, int):
            r = min(L, arg)
        else:
            return sorted(set(arg))
        off = set(spread(L - r, list(range(L))))  # spread: resident layers between them give copies time
        return [i for i in range(L) if i not in off]

    def _setup(self, k: torch.Tensor) -> None:
        B, H, _, D = k.shape
        self.shape = (B, H, D)
        self.dtype = k.dtype
        self.device = k.device
        self.dev = k.device.index if k.device.index is not None else torch.cuda.current_device()
        self.layer_bytes = 2 * self.max_tokens * B * H * D * k.element_size()
        self.resident = self._choose(self.layer_bytes)
        res = set(self.resident)
        self.offloaded = [i for i in range(self.num_layers) if i not in res]
        m = len(self.offloaded)
        # consecutive offloaded layers use different slots (also across the wrap-around)
        self.slot_of: Dict[int, int] = {}
        for j, i in enumerate(self.offloaded):
            self.slot_of[i] = 2 if (m > 1 and m % 2 == 1 and j == m - 1) else (j % 2 if m > 1 else 0)
        self.next_of = {i: self.offloaded[(j + 1) % m] for j, i in enumerate(self.offloaded)} if m > 1 else {}
        nslots = 0 if m == 0 else (1 if m == 1 else (2 if m % 2 == 0 else 3))
        self.slot_cap = self.max_tokens
        self.slots = [self._pair(self.slot_cap) for _ in range(nslots)]
        self.slot_owner: List[Optional[int]] = [None] * nslots
        self.flushes: List[Optional[object]] = [None] * nslots
        self.page_flush: Dict[int, object] = {}  # layer -> D2H event of its newest page
        self.prefetch: Dict[int, tuple] = {}
        self.buf: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.pages: Dict[int, Tuple[list, list]] = {i: ([], []) for i in self.offloaded}
        self.tail: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {i: None for i in self.offloaded}
        self.ready = True

    def _pair(self, tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, D = self.shape
        return (torch.empty(tokens, B, H, D, dtype=self.dtype, device=self.device),
                torch.empty(tokens, B, H, D, dtype=self.dtype, device=self.device))

    # -- public ----------------------------------------------------------------------
    def seq_length(self, layer: int = 0) -> int:
        return self.n[layer]

    @torch.no_grad()
    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append k / v ([batch, kv_heads, new_tokens, head_dim]) to `layer`
        and return all of its keys and values in the same layout."""
        if not self.ready:
            self._setup(k)
        if (k.shape[0], k.shape[1], k.shape[3]) != self.shape or k.dtype != self.dtype:
            raise ValueError(f"PagedKVCache holds [{self.shape[0]}, {self.shape[1]}, T, {self.shape[2]}] "
                             f"{self.dtype}; got {list(k.shape)} {k.dtype}")
        s = k.shape[2]
        n = self.n[layer]
        kt, vt = k.permute(2, 0, 1, 3), v.permute(2, 0, 1, 3)
        if layer in self.slot_of:
            ok, ov = self._update_offloaded(layer, n, s, kt, vt)
        else:
            bk, bv = self._resident(layer, n + s)
            bk[n:n + s].copy_(kt)
            bv[n:n + s].copy_(vt)
            ok, ov = bk[:n + s], bv[:n + s]
        self.n[layer] = n + s
        return ok.permute(1, 2, 0, 3), ov.permute(1, 2, 0, 3)

    @torch.no_grad()
    def reset(self) -> None:
        """Forget every token (e.g. before the next batch of prompts)."""
        if self.ready:
            self._drain()
            for i in self.offloaded:
                self.pages[i] = ([], [])
                self.tail[i] = None
            self.slot_owner = [None] * len(self.slots)
        self.n = [0] * self.num_layers

    def memory(self) -> dict:
        """Bytes of keys / values in VRAM (resident layers, slots, tails) and in host RAM."""
        if not self.ready:
            return dict(device_bytes=0, host_bytes=0)
        nb = lambda t: t.numel() * t.element_size()  # noqa: E731
        dev = sum(nb(a) + nb(b) for a, b in self.buf.values()) + sum(nb(a) + nb(b) for a, b in self.slots)
        dev += sum(nb(t[0]) + nb(t[1]) for t in self.tail.values() if t is not None)
        host = sum(nb(t) for pk, pv in self.pages.values() for t in pk + pv)
        return dict(device_bytes=dev, host_bytes=host)

    # -- internals ---------------------------------------------------------------------
    def _resident(self, layer: int, need: int):
        cur = self.buf.get(layer)
        if cur is not None and cur[0].shape[0] >= need:
            return cur
        new = self._pair(max(need, self.max_tokens, 2 * cur[0].shape[0] if cur is not None else 0))
        if cur is not None:
            n = self.n[layer]
            new[0][:n].copy_(cur[0][:n])
            new[1][:n].copy_(cur[1][:n])
        self.buf[layer] = new
        return new

    def _drain(self) -> None:
        """The compute stream waits for every copy still touching a slot."""
        for ev in self.flushes:
            if ev is not None:
                ev.wait()
        for ev, _, _ in self.prefetch.values():
            ev.wait()
        for ev in self.page_flush.values():
            ev.wait()
        self.flushes = [None] * len(self.slots)
        self.page_flush.clear()
        self.prefetch.clear()

    def _ensure_slots(self, need: int) -> None:
        if need <= self.slot_cap:
            return
        self._drain()  # nothing may still copy into or out of the old slots
        self.slot_cap = max(need, 2 * self.slot_cap)
        self.slots = [self._pair(self.slot_cap) for _ in self.slots]
        self.slot_owner = [None] * len(self.slots)

    def _release_slot(self, sid: int, layer: Optional[int] = None):
        """An event after every queued reader of slot `sid`, its last D2H and
        (when refilling it for `layer`) the D2H that wrote that layer's pages."""
        if self.flushes[sid] is not None:
            self.flushes[sid].wait()
            self.flushes[sid] = None
        if layer is not None and self.page_flush.get(layer) is not None:
            self.page_flush.pop(layer).wait()
        return _C.record_event(self.dev)

    def _fill(self, layer: int, sid: int) -> object:
        after = self._release_slot(sid, layer)
        pk, pv = self.pages[layer]
        held = len(pk) * self.page
        sk, sv = self.slots[sid]
        _C.h2d_pages_async(sk[:held], pk, after)
        ev = _C.h2d_pages_async(sv[:held], pv, after)
        self.stats["h2d_bytes"] += 2 * held * sk[0].numel() * sk.element_size()
        self.slot_owner[sid] = layer
        return ev

    def _update_offloaded(self, layer: int, n: int, s: int, kt: torch.Tensor, vt: torch.Tensor):
        self._ensure_slots(n + s)
        sid = self.slot_of[layer]
        sk, sv = self.slots[sid]
        pk, pv = self.pages[layer]
        held = len(pk) * self.page
        pre = self.prefetch.pop(layer, None)
        if held and pre is not None and pre[1] == held and pre[2] is sk and self.slot_owner[sid] == layer:
            pre[0].wait()
            self.stats["prefetched"] += 1
        elif held:
            self._fill(layer, sid).wait()
            self.stats["gathered"] += 1
        elif self.flushes[sid] is not None:  # nothing to load, but the slot's last pages may still be leaving
            self._release_slot(sid)
        self.slot_owner[sid] = layer
        t = 0
        tail = self.tail[layer]
        if tail is not None:
            t = tail[0].shape[0]
            sk[held:held + t].copy_(tail[0])
            sv[held:held + t].copy_(tail[1])
        total = held + t + s
        sk[held + t:total].copy_(kt)
        sv[held + t:total].copy_(vt)
        full = (t + s) // self.page
        for j in range(full):  # pages that filled up leave for host RAM
            a, b = held + j * self.page, held + (j + 1) * self.page
            hk = _C.pinned_empty(list(sk[a:b].shape), self.dtype, "kvcache", False, True)
            hv = _C.pinned_empty(list(sv[a:b].shape), self.dtype, "kvcache", False, True)
            _C.d2h_async(hk, sk[a:b])
            self.flushes[sid] = _C.d2h_async(hv, sv[a:b])  # one stream, in order: covers both
            self.page_flush[layer] = self.flushes[sid]
            pk.append(hk)
            pv.append(hv)
            self.stats["d2h_bytes"] += 2 * hk.numel() * hk.element_size()
            self.stats["pages"] += 1
        rest = held + full * self.page
        self.tail[layer] = (sk[rest:total].clone(), sv[rest:total].clone()) if total > rest else None
        nxt = self.next_of.get(layer)
        if nxt is not None and self.pages[nxt][0]:
            nsid = self.slot_of[nxt]
            self.prefetch[nxt] = (self._fill(nxt, nsid), len(self.pages[nxt][0]) * self.page, self.slots[nsid][0])
        return sk[:total], sv[:total]
