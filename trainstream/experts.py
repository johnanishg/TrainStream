"""Mixture-of-experts layers whose frozen experts stay in host RAM: only routed experts are copied in.

A MoE decoder layer holds every expert's weights, but each token uses a few
of them (8 of 128 in Qwen3-30B-A3B). Paging a whole layer, as the block pager
does, copies all of them over PCIe every pass. With wrap(moe_paging=True) the
frozen expert tensors of Hugging Face's stacked-experts modules (gate_up_proj
[E, 2I, H] and down_proj [E, H, I]; Qwen3-MoE, Qwen2-MoE, Mixtral, ...) live
in pinned host memory, and a layer copies in only the experts some token was
routed to.

How the copies are kept off the compute path:

* One look at the routing per layer. The tokens are grouped by expert on the
  GPU (one sort); a single count vector comes to the host, the only time the
  layer waits for the GPU. From it the layer knows every expert it needs.
* All of them are queued at once on the H2D stream, before the first expert
  is computed, each with its own CUDA event. Expert i's matmuls wait (on the
  GPU) for expert i's event only, so they run while experts i+1.. are still
  crossing PCIe. Experts are computed in the model's own order, so the result
  is bit-identical to the unpaged model.
* A cache in VRAM (moe_cache) keeps recently used experts, least recently
  used out first. A hit costs no transfer; with enough room backward finds
  the experts forward used still there.
* Backward: when it reaches a layer, every expert that layer's forward used
  and the cache no longer holds is queued at once, and so is the MoE layer
  below (the next one backward will reach). That list is exact: it was
  recorded in forward.
* Forward lookahead: a layer's routing depends on the layer before it, so the
  next layer's experts cannot be known exactly in advance. They can be
  guessed: the next layer's router applied to this layer's input picks mostly
  the same experts it will pick from its own input a moment later. Those are
  queued behind this layer's own copies and land in the cache. A wrong guess
  costs PCIe time, never correctness; guessing stops for a layer whose
  guesses turn out wrong more often than right (see memory_report).
* moe_capacity (off by default) bounds the copies per layer: beyond that many
  experts, tokens routed to an expert that is not in VRAM are re-routed to
  the best-scoring one that is, by running the router's top-k again over the
  allowed experts. This changes what the model computes; it is for runs where
  PCIe, not the GPU, sets the step time.

It pays when a step routes to few experts: short sequences, small batches,
generation during RL. With thousands of tokens per step nearly every expert
is hit and the copy volume is the same as paging the layer. The expert
weights must be frozen (LoRA on the attention, the router, the dense layers).
"""

from __future__ import annotations

import types
import weakref
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from . import _C

NAMES = ("gate_up_proj", "down_proj")
SPEC_PROBATION = 64    # speculative copies of a layer before its guesses are judged
SPEC_MIN_HIT_RATE = 0.5


def is_stacked_experts(m: nn.Module) -> bool:
    """Hugging Face's stacked-experts module: 3-D gate_up_proj / down_proj, num_experts, act_fn."""
    return (all(isinstance(getattr(m, n, None), nn.Parameter) and getattr(m, n).dim() == 3 for n in NAMES)
            and isinstance(getattr(m, "num_experts", None), int) and callable(getattr(m, "act_fn", None))
            and not getattr(m, "has_bias", False))


class _Entry:
    """One expert's two matrices in VRAM, and the event of the copy that fills them."""

    __slots__ = ("w1", "w2", "event", "nbytes", "spec", "landed")

    def __init__(self, w1, w2, event, spec=False):
        self.w1, self.w2, self.event = w1, w2, event
        self.nbytes = w1.numel() * w1.element_size() + w2.numel() * w2.element_size()
        self.spec = spec      # copied on a guess; not yet used
        self.landed = False   # a consumer has made the compute stream wait for the copy

    def wait(self) -> None:
        if not self.landed:
            self.event.wait()  # the compute stream waits; the host does not
            self.landed = True


class ExpertCache:
    """Experts kept in VRAM across layers and passes, least recently used out first.

    `budget` bytes at most (0: nothing is kept; each layer holds its experts only while it
    computes). One cache serves every stack of a model."""

    def __init__(self, budget: int = 0):
        self.budget = max(0, int(budget))
        self.bytes = 0
        self._entries: Dict[tuple, _Entry] = {}  # insertion order = least recently used first
        self.hits = self.misses = self.evictions = 0

    def get(self, key) -> Optional[_Entry]:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._entries[key] = entry  # most recently used
        return entry

    def has(self, key) -> bool:
        return key in self._entries

    def room_for(self, nbytes: int, protect=()) -> bool:
        """Whether `nbytes` more would fit after evicting entries outside `protect`."""
        if nbytes > self.budget:
            return False
        evictable = sum(e.nbytes for k, e in self._entries.items() if k not in protect)
        return self.bytes - evictable + nbytes <= self.budget

    def put(self, key, entry: _Entry, protect=()) -> bool:
        if entry.nbytes > self.budget:
            return False
        for k in [k for k in self._entries if k not in protect]:
            if self.bytes + entry.nbytes <= self.budget:
                break
            self._drop(k)
        if self.bytes + entry.nbytes > self.budget:
            return False
        self._entries[key] = entry
        self.bytes += entry.nbytes
        return True

    def _drop(self, key) -> None:
        entry = self._entries.pop(key)
        self.bytes -= entry.nbytes
        self.evictions += 1
        # Its memory goes back to the allocator now. A copy that is still writing into it must
        # finish before anything queued later on the compute stream can be given that memory.
        entry.wait()

    def drop_owner(self, owner) -> None:
        for k in [k for k in self._entries if k[0] == owner]:
            self._drop(k)

    def clear(self) -> None:
        for k in list(self._entries):
            self._drop(k)


class _Call:
    """One forward of one stack: the experts it used and, while they are needed, their weights."""

    __slots__ = ("pager", "experts", "entries", "below", "bwd_started", "__weakref__")

    def __init__(self, pager, experts, below):
        self.pager = pager
        self.experts = experts          # ascending: the order forward computes them in
        self.entries: Dict[int, _Entry] = {}
        self.below = below              # weakref to the previous MoE layer's call in this pass
        self.bwd_started = False

    def weights(self, e: int):
        entry = self.entries.get(e)
        if entry is None:  # backward of an expert nothing has queued yet (e.g. a second backward)
            entry = self.entries[e] = self.pager.acquire(e)
        entry.wait()
        return entry.w1, entry.w2

    def start_backward(self) -> None:
        """Backward has reached this layer: queue every expert it will need, in the order it
        needs them (the reverse of forward), then those of the MoE layer below, which backward
        reaches next. One layer ahead, no further: two layers' experts in VRAM at most."""
        if self.bwd_started:
            return
        self.bwd_started = True
        self._queue()
        below = self.below() if self.below is not None else None
        if below is not None:
            below._queue()

    def _queue(self) -> None:
        for e in reversed(self.experts):
            if e not in self.entries:
                self.entries[e] = self.pager.acquire(e)

    def release(self, e: int) -> None:
        self.entries.pop(e, None)  # used (its copy has landed); the cache keeps it if it has room


class ExpertFn(torch.autograd.Function):
    """One expert, x [T, H] -> [T, H], with weights that come from the host (or the cache) for
    forward and again for backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, call, e):
        w1, w2 = call.weights(e)
        gate, up = F.linear(x, w1).chunk(2, dim=-1)
        y = F.linear(call.pager.module.act_fn(gate) * up, w2)
        ctx.save_for_backward(x, gate, up)
        ctx.call, ctx.e = call, e
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        call = ctx.call
        call.start_backward()
        x, gate, up = ctx.saved_tensors
        w1, w2 = call.weights(ctx.e)
        da = gy.to(w2.dtype) @ w2  # y = a w2^T
        with torch.enable_grad():
            g = gate.detach().requires_grad_()
            u = up.detach().requires_grad_()
            a = call.pager.module.act_fn(g) * u
            dg, du = torch.autograd.grad(a, (g, u), da.to(a.dtype))
        dx = torch.cat((dg, du), dim=-1).to(w1.dtype) @ w1  # h = x w1^T
        call.release(ctx.e)
        return dx.to(x.dtype), None, None


def _forward(self, hidden_states, top_k_index, top_k_weights):
    """Hugging Face's loop over the experts some token was routed to, with each expert's
    weights copied in from the host (or found in the cache) for it."""
    pager = self._ts_expert_pager
    E = self.num_experts
    tokens = hidden_states.shape[0]
    nxt = pager.next_to_guess()
    with torch.no_grad():
        counts = torch.bincount(top_k_index.reshape(-1), minlength=E + 1)[:E]
        if nxt is not None:
            counts = torch.cat((counts, nxt.guess_counts(hidden_states)))
        host = counts.tolist()  # the layer's one wait for the GPU
    need = [e for e in range(E) if host[e]]
    guess = host[E:]

    if pager.capacity is not None:
        absent = [e for e in need if not pager.cache.has((pager, e))]
        if len(absent) > pager.capacity:
            keep = set(sorted(absent, key=lambda e: -host[e])[: pager.capacity])  # the most used ones
            dropped = [e for e in absent if e not in keep]
            allowed = [e for e in need if e not in dropped]
            pager.rerouted_tokens += sum(host[e] for e in dropped)  # (token, slot) pairs that move
            pager.rerouted_layers += 1
            top_k_index, top_k_weights = pager.reroute(hidden_states, allowed)
            with torch.no_grad():
                host = torch.bincount(top_k_index.reshape(-1), minlength=E + 1)[:E].tolist()  # a second wait
            need = [e for e in range(E) if host[e]]

    # tokens grouped by expert, in the order torch.where(mask[e]) gives them: by top-k slot, then token
    with torch.no_grad():
        flat = top_k_index.t().reshape(-1)
        order = torch.sort(flat, stable=True).indices
        slot_of, token_of = order // tokens, order % tokens

    call = _Call(pager, need, pager.link_call())
    pager.last_call = weakref.ref(call)
    for e in need:  # every copy is queued before the first expert computes
        call.entries[e] = pager.acquire(e)
    if nxt is not None:
        nxt.speculate(guess, protect={(pager, e) for e in need})

    final = torch.zeros_like(hidden_states)
    start = 0  # experts no token uses take no rows
    for e in need:
        n = host[e]
        token_idx, top_k_pos = token_of[start:start + n], slot_of[start:start + n]
        start += n
        out = ExpertFn.apply(hidden_states[token_idx], call, e)
        out = out * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, out.to(final.dtype))
    call.entries.clear()  # all used; what the cache has room for stays there
    pager.calls += 1
    pager.experts_used += len(need)
    return final


class ExpertPager:
    """The host copies of one stacked-experts module and the copies in."""

    def __init__(self, module: nn.Module, device, cache: Optional[ExpertCache] = None, router: Optional[nn.Module] = None,
                 capacity: Optional[int] = None):
        self.module = module
        self.device = torch.device(device)
        self.cache = cache if cache is not None else ExpertCache(0)
        self.router = router      # the top-k router that feeds this stack (speculation, capacity)
        self.capacity = capacity
        self.index = 0            # depth order among the model's stacks
        self.below: Optional["ExpertPager"] = None
        self.above: Optional["ExpertPager"] = None
        self.last_call = None
        self.host = {}
        self.calls = self.experts_used = self.bytes_in = 0
        self.spec_issued = self.spec_hits = 0
        self.spec_off = False
        self.rerouted_layers = self.rerouted_tokens = 0
        for n in NAMES:
            p = getattr(module, n)
            src = p.detach()
            h = _C.pinned_empty(list(src.shape), src.dtype, "param.experts", False, False)
            h.copy_(src)
            self.host[n] = h
            with torch.no_grad():  # the module keeps its shapes; the values live on the host
                p.data = torch.empty(src.shape, dtype=src.dtype, device=self.device)
            p.untyped_storage().resize_(0)
            p._ts_expert = True
        self.expert_bytes = sum(h[0].numel() * h.element_size() for h in self.host.values())
        module._ts_expert_pager = self
        module.forward = types.MethodType(_forward, module)

    # -- copies ---------------------------------------------------------------------
    def _copy(self, e: int, spec: bool) -> _Entry:
        dev = self.device.index if self.device.index is not None else -1
        w1, _ = _C.h2d_alloc_async(self.host["gate_up_proj"][e], None, dev)
        w2, ev = _C.h2d_alloc_async(self.host["down_proj"][e], None, dev)  # same stream: its event covers both
        self.bytes_in += self.expert_bytes
        return _Entry(w1, w2, ev, spec)

    def acquire(self, e: int) -> _Entry:
        """Expert e's weights in VRAM or on their way: from the cache, else queued on the H2D
        stream now (and offered to the cache)."""
        cache = self.cache
        entry = cache.get((self, e))
        if entry is not None:
            cache.hits += 1
            if entry.spec:
                entry.spec = False
                self.spec_hits += 1
            return entry
        cache.misses += 1
        entry = self._copy(e, False)
        cache.put((self, e), entry)
        return entry

    def fetch(self, e: int, device=None):
        """(gate_up_proj[e], down_proj[e]) in VRAM, ready to read on the compute stream."""
        entry = self.acquire(e)
        entry.wait()
        return entry.w1, entry.w2

    # -- one forward pass: which call came before this one -----------------------------
    def link_call(self):
        """The previous MoE layer's call of the pass this forward belongs to (a weakref), so
        backward can queue that layer's experts as soon as it reaches this one."""
        below = self.below
        return below.last_call if below is not None else None

    # -- the next layer, guessed ---------------------------------------------------------
    def next_to_guess(self) -> Optional["ExpertPager"]:
        nxt = self.above
        if nxt is None or nxt.router is None or nxt.spec_off or not self.cache.budget:
            return None
        return nxt

    def guess_counts(self, hidden_states) -> torch.Tensor:
        """How many tokens this stack's router would send to each expert if it saw
        `hidden_states` (the layer below's input): a guess at its routing a layer early."""
        w = self.router.weight
        logits = F.linear(hidden_states.detach().reshape(-1, w.shape[1]).to(w.dtype), w.detach())
        k = min(int(getattr(self.router, "top_k", 1)), self.module.num_experts)
        return torch.bincount(logits.topk(k, dim=-1).indices.reshape(-1), minlength=self.module.num_experts)

    def speculate(self, guess: List[int], protect=()) -> None:
        """Queue the experts the guess says this stack will need, most used first, as far as
        the cache has room for them without giving up `protect` (the running layer's own)."""
        cache = self.cache
        if self.spec_issued >= SPEC_PROBATION and self.spec_hits < SPEC_MIN_HIT_RATE * self.spec_issued:
            self.spec_off = True  # this layer's routing does not follow the layer below's input
            return
        limit = cache.budget // 2  # guesses never take more than half of the cache
        used = 0
        for e in sorted((e for e in range(len(guess)) if guess[e]), key=lambda e: -guess[e]):
            if cache.has((self, e)):
                continue
            if used + self.expert_bytes > limit or not cache.room_for(self.expert_bytes, protect):
                break
            entry = self._copy(e, True)
            if not cache.put((self, e), entry, protect):
                entry.wait()  # cannot happen after room_for; never drop a copy in flight unwaited
                break
            used += self.expert_bytes
            self.spec_issued += 1

    # -- capacity ------------------------------------------------------------------------
    def reroute(self, hidden_states, allowed: List[int]):
        """The router's top-k again, over the `allowed` experts only: (top_k_index,
        top_k_weights) as Hugging Face's top-k router returns them."""
        router = self.router
        E = self.module.num_experts
        w = router.weight
        logits = F.linear(hidden_states.reshape(-1, w.shape[1]).to(w.dtype), w)
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
        mask = torch.zeros(E, dtype=torch.bool, device=probs.device)
        mask[torch.tensor(allowed, device=probs.device)] = True
        probs = probs.masked_fill(~mask, 0.0)
        top_k = int(getattr(router, "top_k", 1))
        k = min(top_k, len(allowed))
        values, index = torch.topk(probs, k, dim=-1)
        if getattr(router, "norm_topk_prob", False):
            values = values / values.sum(dim=-1, keepdim=True)
        values = values.to(logits.dtype)
        if k < top_k:  # fewer allowed experts than slots: the spare slots point at "no expert"
            pad = top_k - k
            index = torch.cat((index, index.new_full((index.shape[0], pad), E)), dim=-1)
            values = torch.cat((values, values.new_zeros((values.shape[0], pad))), dim=-1)
        return index, values

    @property
    def nbytes(self) -> int:
        return sum(h.numel() * h.element_size() for h in self.host.values())

    def remove(self) -> None:
        """Put the experts back in VRAM, as they were."""
        m = self.module
        self.cache.drop_owner(self)
        self.last_call = None
        m.__dict__.pop("forward", None)
        m.__dict__.pop("_ts_expert_pager", None)
        for n in NAMES:
            p = getattr(m, n)
            with torch.no_grad():
                p.data = self.host[n].to(self.device)
            p.__dict__.pop("_ts_expert", None)
        self.host = {}


def _router_of(model: nn.Module, experts_module: nn.Module) -> Optional[nn.Module]:
    """The top-k router feeding `experts_module`: its sibling `gate` in Hugging Face's sparse MoE
    block (a module with a [num_experts, hidden] weight)."""
    for parent in model.modules():
        if getattr(parent, "experts", None) is experts_module:
            gate = getattr(parent, "gate", None)
            w = getattr(gate, "weight", None)
            if isinstance(w, torch.Tensor) and w.dim() == 2 and w.shape[0] == experts_module.num_experts:
                return gate
    return None


def install(model: nn.Module, device, cache_bytes: int = 0, capacity: Optional[int] = None) -> List[ExpertPager]:
    """Page the frozen stacked experts of `model` from host RAM (see the module docstring).

    cache_bytes: VRAM for the expert cache shared by every stack (0: none, and no guessing of
    the next layer). capacity: at most this many experts copied in per layer and pass; tokens
    routed to others are re-routed (None: no limit)."""
    if capacity is not None and (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1):
        raise ValueError(f"moe_capacity must be None or a number of experts (>= 1), got {capacity!r}")
    cache = ExpertCache(cache_bytes)
    out: List[ExpertPager] = []
    for m in model.modules():
        if is_stacked_experts(m) and "forward" not in m.__dict__ and not any(getattr(m, n).requires_grad for n in NAMES):
            router = _router_of(model, m)
            if capacity is not None and router is None:
                raise ValueError("moe_capacity needs the top-k router next to each stack of experts (Hugging Face's "
                                 "sparse MoE block: .gate with a [num_experts, hidden] weight); none found for "
                                 f"{type(m).__name__}")
            out.append(ExpertPager(m, device, cache, router, capacity))
    for i, p in enumerate(out):
        p.index = i
        p.below = out[i - 1] if i > 0 else None
        p.above = out[i + 1] if i + 1 < len(out) else None
    return out


def stats(pagers: List[ExpertPager]) -> dict:
    """Totals over a model's stacks, for memory_report()["experts"]."""
    cache = pagers[0].cache if pagers else ExpertCache(0)
    issued = sum(p.spec_issued for p in pagers)
    return dict(stacks=len(pagers), host_bytes=sum(p.nbytes for p in pagers), calls=sum(p.calls for p in pagers),
                experts_used=sum(p.experts_used for p in pagers),
                experts_copied=sum(p.bytes_in // max(p.expert_bytes, 1) for p in pagers),
                bytes_copied=sum(p.bytes_in for p in pagers),
                cache_budget=cache.budget, cache_bytes=cache.bytes, cache_hits=cache.hits, cache_misses=cache.misses,
                cache_evictions=cache.evictions,
                prefetch_issued=issued, prefetch_hits=sum(p.spec_hits for p in pagers),
                prefetch_stopped_stacks=sum(1 for p in pagers if p.spec_off),
                capacity=pagers[0].capacity if pagers else None,
                rerouted_layers=sum(p.rerouted_layers for p in pagers),
                rerouted_tokens=sum(p.rerouted_tokens for p in pagers))
