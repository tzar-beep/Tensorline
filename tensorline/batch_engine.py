"""Phase 2 engine: continuous batching over a paged KV cache.

A single scheduler thread owns the GPU. Every iteration it:
  1. gives each running sequence its next tokens (1 for decode, a chunk for an unfinished prefill),
     preempting the newest sequence (freeing its blocks; it is recomputed later) when KV
     blocks run out;
  2. admits waiting requests FCFS while the token budget, seq limit and free blocks allow;
  3. runs ONE forward pass over all scheduled tokens and samples one token per sequence
     that reached the end of its scheduled tokens;
  4. retires finished sequences and returns their blocks immediately.
Requests therefore join and leave the batch at every step (iteration-level scheduling, as in
Orca/vLLM) instead of waiting for a whole batch to drain.

Output path: the scheduler never touches per-request threads. Sampled token ids are collected
during a step and delivered at the end of it: to asyncio callers (`astream`, used by the server)
with ONE `call_soon_threadsafe` wakeup per event loop per step, carrying every sequence's
tokens; to sync callers (`stream`) through a queue. Detokenization happens on the consumer
side, incrementally (see detok.py).
"""
from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

import torch

from .detok import IncrementalDetokenizer
from .engine import Engine, GenStats, SamplingParams, sample
from .metrics import EngineMetrics
from .paged import BlockAllocator, PagedModel, SeqChunk, block_hash


class _Sink:
    """Where a sequence's outputs go: token ids, then None (done) or an Exception."""
    __slots__ = ("q", "loop")

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.loop = loop
        self.q = queue.Queue() if loop is None else asyncio.Queue()


def _deliver(batch):          # runs on the event loop thread
    for q, item in batch:
        q.put_nowait(item)


@dataclass(eq=False)
class Sequence:
    sid: int
    prompt_len: int
    tokens: list[int]
    params: SamplingParams
    stops: set[int]
    out: _Sink
    stats: GenStats
    t_submit: float
    blocks: list[int] = field(default_factory=list)
    num_computed: int = 0      # tokens whose K/V is in the cache
    hashes: list[int] = field(default_factory=list)   # content hashes of its full, computed blocks
    done: bool = False
    ngrams: dict = field(default_factory=dict)        # n-gram -> position of the token after it
    ngram_upto: int = 0
    spec_prop: int = 0
    spec_acc: int = 0
    spec_miss: int = 0                                # consecutive lookups that found no draft

    @property
    def num_generated(self) -> int:
        return len(self.tokens) - self.prompt_len


class BatchEngine(Engine):
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", *,
                 block_size: int = 16, num_blocks: int | None = None,
                 kv_mem_fraction: float = 0.6, max_num_seqs: int = 64,
                 max_batched_tokens: int = 2048, max_model_len: int = 4096,
                 cuda_graphs: bool = True, prefix_caching: bool = True,
                 quant: str | None = None, kv_cache_gb: float | None = None,
                 graph_warmup_tokens: int = 512, overlap: bool = True, spec_tokens: int = 0):
        super().__init__(model_id, device, dtype, quant)
        if self.quant and self.device == "cuda":
            self._warm_quant_kernels(max_num_seqs)
        if num_blocks is None:
            per_block = PagedModel.bytes_per_block(self.model, block_size)
            if kv_cache_gb is not None:
                num_blocks = int(kv_cache_gb * (1 << 30)) // per_block
            elif self.device == "cuda":
                free, _ = torch.cuda.mem_get_info()
                num_blocks = int(free * kv_mem_fraction) // per_block
            else:
                num_blocks = (512 << 20) // per_block   # 512 MiB on CPU
        if num_blocks * block_size < max_num_seqs * 256:
            import warnings
            warnings.warn(f"KV cache holds only {num_blocks * block_size} tokens ({num_blocks} blocks): "
                          f"{max_num_seqs} concurrent sequences will queue or preempt. Free GPU memory "
                          "(another process may be holding it) or pass --kv-cache-gb.", stacklevel=2)
        self.paged = PagedModel(self.model, num_blocks, block_size)
        self.alloc = BlockAllocator(num_blocks, prefix_caching)
        self.graphs = None
        if cuda_graphs and self.device == "cuda":
            from .cuda_graphs import CudaGraphDecoder
            self.graphs = CudaGraphDecoder(self.paged, max_num_seqs, -(-max_model_len // block_size))
            if graph_warmup_tokens:
                self.graphs.warmup(min(graph_warmup_tokens, max_model_len))
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.max_model_len = max_model_len
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._cv = threading.Condition()
        self._ids = itertools.count()
        # counters for metrics / tests
        self.steps = 0
        self.preemptions = 0
        self.max_running_seen = 0
        self.prompt_tokens_total = 0
        self.prefix_hit_tokens = 0
        self.metrics = EngineMetrics()
        self._pending: list[tuple[_Sink, object]] = []   # outputs produced during this step
        self._aborted: list[Sequence] = []
        # Overlap scheduler (CUDA-graph decode only): step t+1 is launched before step t's tokens
        # are read back; `_inflight` = (seqs, sampled-token tensor on GPU, pinned host copy, event).
        # Speculative decoding (prompt-lookup drafts, verified inside the CUDA-graph decode batch).
        # Speculative steps are synchronous, so it replaces the overlap pipeline when enabled.
        self.spec_tokens = spec_tokens if self.graphs is not None else 0
        self.spec_proposed = self.spec_accepted = self.spec_steps = 0
        self.overlap = overlap and self.graphs is not None
        self._inflight = None
        self._tok_host = ([torch.empty(max_num_seqs, dtype=torch.long).pin_memory() for _ in range(2)]
                          if self.overlap else None)
        self._flip = 0
        self.overlapped_steps = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tl-scheduler")
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the scheduler, fail anything still queued, and drop every GPU tensor this engine
        owns (weights, KV cache, CUDA graphs) so the memory can be reclaimed."""
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join()
        err = RuntimeError("model unloaded")
        for seq in self.running + list(self.waiting):
            self._emit(seq, err)
        self.running.clear()
        self.waiting.clear()
        self._flush()
        self.graphs = self.paged = self.model = None

    def _warm_quant_kernels(self, max_batch: int) -> None:
        """Run every quantised layer once per row-tile size so Triton autotuning happens now,
        not on the first live requests (or inside a CUDA-graph capture)."""
        from .quant import QuantLinear
        layers = [m for m in self.model.modules() if isinstance(m, QuantLinear)]
        with torch.inference_mode():
            for M in sorted({1, min(32, max_batch), min(64, max_batch)}):
                for m in layers:
                    m(torch.zeros(M, m.in_features, device=self.device, dtype=m.dtype))
        torch.cuda.synchronize()

    # ---- public API ----
    def _submit(self, prompt: str, params: SamplingParams, stats: GenStats, sink: _Sink) -> Sequence:
        ids = self.tokenizer(prompt).input_ids
        if len(ids) >= self.max_model_len:
            raise ValueError(f"prompt has {len(ids)} tokens; max_model_len is {self.max_model_len}")
        seq = Sequence(next(self._ids), len(ids), list(ids), params,
                       params.stop_token_ids or self.default_stops, sink, stats,
                       time.perf_counter())
        stats.prompt_tokens = len(ids)
        with self._cv:
            self.waiting.append(seq)
            self._cv.notify()
        return seq

    def abort(self, seq: Sequence) -> None:
        """Stop generating for `seq` (e.g. client disconnected); no-op if already finished."""
        with self._cv:
            self._aborted.append(seq)
            self._cv.notify()

    def stream(self, prompt: str, params: SamplingParams, stats: GenStats) -> Iterator[str]:
        """Blocking generator of text deltas (same shape as the naive Engine)."""
        seq = self._submit(prompt, params, stats, _Sink())
        detok = IncrementalDetokenizer(self.tokenizer)
        try:
            while (item := seq.out.q.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                if delta := detok.push([item]):
                    yield delta
        finally:
            self.abort(seq)

    async def astream(self, prompt: str, params: SamplingParams, stats: GenStats) -> AsyncIterator[str]:
        """Async generator of text deltas, with no per-request thread. Tokens that arrive while
        the consumer is busy are coalesced into one delta, so a slow client costs one event per
        wakeup rather than one per token."""
        seq = self._submit(prompt, params, stats, _Sink(asyncio.get_running_loop()))
        q, detok = seq.out.q, IncrementalDetokenizer(self.tokenizer)
        try:
            while True:
                items = [await q.get()]
                while not q.empty():
                    items.append(q.get_nowait())
                ids, done = [], False
                for it in items:
                    if it is None:
                        done = True
                        break
                    if isinstance(it, Exception):
                        raise it
                    ids.append(it)
                if ids and (delta := detok.push(ids)):
                    yield delta
                if done:
                    return
        finally:
            self.abort(seq)

    # ---- output delivery (scheduler thread) ----
    def _emit(self, seq: Sequence, item) -> None:
        self._pending.append((seq.out, item))

    def _flush(self) -> None:
        pending, self._pending = self._pending, []
        by_loop: dict = {}
        for sink, item in pending:
            if sink.loop is None:
                sink.q.put(item)
            else:
                by_loop.setdefault(sink.loop, []).append((sink.q, item))
        for loop, batch in by_loop.items():
            try:
                loop.call_soon_threadsafe(_deliver, batch)
            except RuntimeError:      # event loop closed: nobody is listening any more
                pass

    # ---- scheduler ----
    def _blocks_needed(self, seq: Sequence, upto: int) -> int:
        return max(0, -(-upto // self.block_size) - len(seq.blocks))

    def _grow(self, seq: Sequence, upto: int) -> bool:
        need = self._blocks_needed(seq, upto)
        if need == 0:
            return True
        got = self.alloc.allocate(need)
        if got is None:
            return False
        seq.blocks += got
        return True

    def _release(self, seq: Sequence) -> None:
        self.alloc.free(seq.blocks)
        seq.blocks = []
        seq.hashes = []

    def _match_prefix(self, seq: Sequence) -> None:
        """Point a fresh sequence at cached blocks for the longest already-computed prefix."""
        bs, parent = self.block_size, None
        for i in range(len(seq.tokens) // bs):
            h = block_hash(parent, seq.tokens[i * bs:(i + 1) * bs])
            blk = self.alloc.lookup(h)
            if blk is None:
                break
            seq.blocks.append(blk)
            seq.hashes.append(h)
            parent = h
        if len(seq.blocks) * bs == len(seq.tokens):   # fully cached: still need logits for the
            self.alloc.free([seq.blocks.pop()])        # last token, so recompute the last block
            seq.hashes.pop()
        seq.num_computed = len(seq.blocks) * bs

    def _publish_full_blocks(self, seq: Sequence) -> None:
        bs = self.block_size
        # With overlap, KV can be computed for a position whose token id is still on the GPU,
        # so only blocks whose tokens are all known on the host can be hashed.
        while len(seq.hashes) < min(seq.num_computed, len(seq.tokens)) // bs:
            i = len(seq.hashes)
            h = block_hash(seq.hashes[-1] if i else None, seq.tokens[i * bs:(i + 1) * bs])
            seq.hashes.append(h)
            self.alloc.register(seq.blocks[i], h)

    def _preempt(self, seq: Sequence) -> None:
        """Recompute-style preemption: drop the KV cache, requeue at the front."""
        self.running.remove(seq)
        self._release(seq)
        seq.num_computed = 0
        self.waiting.appendleft(seq)
        self.preemptions += 1

    def _drop_aborted(self) -> None:
        for seq in self._aborted:
            if seq in self.running:
                self.running.remove(seq)
            elif seq in self.waiting:
                self.waiting.remove(seq)
            else:
                continue              # already finished
            self._release(seq)
            seq.done = True
            seq.stats.finish_reason = "abort"
            self.metrics.requests_finished["abort"] = self.metrics.requests_finished.get("abort", 0) + 1
        self._aborted.clear()

    def _schedule(self) -> list[tuple[Sequence, int]]:
        if self._aborted:
            self._drop_aborted()
        budget = self.max_batched_tokens
        plan: list[tuple[Sequence, int]] = []
        for seq in list(self.running):          # oldest first
            if seq not in self.running:          # preempted earlier in this pass
                continue
            n = min(len(seq.tokens) - seq.num_computed, budget)
            if n <= 0:
                continue
            while not self._grow(seq, seq.num_computed + n):
                victim = self.running[-1]        # newest running sequence
                self._preempt(victim)
                budget += sum(k for s, k in plan if s is victim)
                plan = [(s, k) for s, k in plan if s is not victim]
                if victim is seq:
                    break
            else:
                plan.append((seq, n))
                budget -= n
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if self.alloc.prefix_caching and not seq.blocks:
                self._match_prefix(seq)
            n = min(len(seq.tokens) - seq.num_computed, budget)
            if not self._grow(seq, seq.num_computed + n):
                self._release(seq)                # drop any prefix refs while it waits
                seq.num_computed = 0
                if not self.running:              # can never fit, even alone
                    self.waiting.popleft()
                    self._emit(seq, RuntimeError("request does not fit in the KV cache"))
                    continue
                break                             # wait for blocks to free up
            self.waiting.popleft()
            self.running.append(seq)
            if seq.num_generated == 0:            # first admission (not a preemption re-admit)
                seq.stats.cached_tokens = seq.num_computed
                self.prompt_tokens_total += seq.prompt_len
                self.prefix_hit_tokens += seq.num_computed
            plan.append((seq, n))
            budget -= n
        return plan

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.done = True
        seq.stats.finish_reason = reason
        seq.stats.completion_tokens = seq.num_generated
        seq.stats.total_s = time.perf_counter() - seq.t_submit
        self.metrics.e2e.observe(seq.stats.total_s)
        self.metrics.requests_finished[reason] = self.metrics.requests_finished.get(reason, 0) + 1
        self.running.remove(seq)
        self._release(seq)
        self._emit(seq, None)

    def _step(self, plan: list[tuple[Sequence, int]]) -> None:
        if not plan:
            return
        chunks = []
        for seq, n in plan:
            end = seq.num_computed + n
            chunks.append(SeqChunk(seq.tokens[seq.num_computed:end], seq.num_computed,
                                   seq.blocks, want_logits=end == len(seq.tokens)))
        if self.graphs is not None and self.graphs.can_run(chunks):
            logits = self.graphs.run(chunks)        # pure decode step: replay a CUDA graph
        else:
            logits = self.paged.forward(chunks)     # prefill / mixed step: eager
        self.steps += 1
        self.max_running_seen = max(self.max_running_seen, len(self.running))
        self.metrics.batch.observe(len(plan))

        sampled = [s for (s, _), c in zip(plan, chunks) if c.want_logits]
        greedy = logits.argmax(dim=-1).tolist() if sampled else []
        for (seq, n) in plan:
            seq.num_computed += n
            self._publish_full_blocks(seq)
        now = time.perf_counter()
        for i, seq in enumerate(sampled):
            tok = greedy[i] if seq.params.temperature <= 0 else sample(logits[i], seq.params)
            self._accept(seq, tok, now)

    def _accept(self, seq: Sequence, tok: int, now: float) -> None:
        if seq.num_generated == 0:
            seq.stats.ttft_s = now - seq.t_submit
            self.metrics.ttft.observe(seq.stats.ttft_s)
        if tok in seq.stops:
            self._finish(seq, "stop")
            return
        seq.tokens.append(tok)
        self.metrics.tokens_generated += 1
        self._emit(seq, tok)
        if seq.num_generated >= seq.params.max_tokens or len(seq.tokens) >= self.max_model_len:
            self._finish(seq, "length")

    # ---- speculative decoding ----
    NGRAM = 3

    def _propose(self, seq: Sequence, k: int) -> list[int]:
        """Prompt-lookup draft: if the last NGRAM tokens occurred earlier in the context, propose
        the (up to k) tokens that followed that occurrence. The n-gram index is updated
        incrementally, so each lookup is O(1) amortised."""
        t, n = seq.tokens, self.NGRAM
        for i in range(seq.ngram_upto, len(t) - n):
            seq.ngrams[tuple(t[i:i + n])] = i + n          # latest occurrence wins
        seq.ngram_upto = max(seq.ngram_upto, len(t) - n)
        pos = seq.ngrams.get(tuple(t[-n:]))
        return t[pos:pos + k] if pos is not None else []

    def _spec_cold(self, s: Sequence) -> bool:
        """Adaptive: drafting stops for a sequence whose drafts keep being rejected or whose
        lookups keep finding nothing (open-ended text), so it returns to the overlapped pipeline
        instead of paying for verification. It is re-probed every 32 generated tokens."""
        if s.num_generated % 32 == 0:
            return False
        return (s.spec_prop >= 16 and s.spec_acc < 0.3 * s.spec_prop) or s.spec_miss >= 4

    def _spec_step(self, plan) -> bool:
        """One verification step for a pure greedy decode batch. Each sequence contributes its
        last token plus its draft as separate rows of the graph-decoded batch (row j sits at
        position start+j and attends to the same block table up to its own length). The longest
        draft prefix matching the model's own argmax is accepted, plus one bonus token, so the
        output is token-identical to plain greedy decoding. Returns False if nothing was drafted."""
        seqs = [s for s, _ in plan]
        room = self.graphs.batch_buckets[-1] - len(seqs)
        drafts = []
        for s in seqs:
            limit = min(self.spec_tokens, room, s.params.max_tokens - s.num_generated - 1,
                        self.max_model_len - len(s.tokens) - 1)
            d = self._propose(s, limit) if limit > 0 and not self._spec_cold(s) else []
            s.spec_miss = 0 if d else s.spec_miss + 1
            if d and not self._grow(s, s.num_computed + 1 + len(d)):
                d = []
            room -= len(d)
            drafts.append(d)
        if not any(drafts):
            return False
        chunks, offs = [], []
        for s, d in zip(seqs, drafts):
            offs.append(len(chunks))
            for j, tok in enumerate([s.tokens[-1]] + d):
                chunks.append(SeqChunk([tok], s.num_computed + j, s.blocks, True))
        if not self.graphs.can_run(chunks):
            return False
        preds = self.graphs.run(chunks).argmax(dim=-1).tolist()
        self.steps += 1
        self.spec_steps += 1
        self.metrics.batch.observe(len(chunks))
        now = time.perf_counter()
        for s, d, off in zip(seqs, drafts, offs):
            a = 0
            while a < len(d) and d[a] == preds[off + a]:
                a += 1
            self.spec_proposed += len(d)
            self.spec_accepted += a
            s.spec_prop += len(d)
            s.spec_acc += a
            s.num_computed += 1 + a                         # KV past this is stale; overwritten later
            self._publish_full_blocks(s)
            for tok in d[:a] + [preds[off + a]]:
                self._accept(s, tok, now)
                if s.done:
                    break
        return True

    # ---- overlap scheduler ----
    def _overlappable(self, plan, chunks) -> bool:
        return (self.overlap and all(s.params.temperature <= 0 for s, _ in plan)
                and self.graphs.can_run(chunks))

    def _launch(self, seqs: list[Sequence], chunks: list[SeqChunk], dev_ids=None):
        """Replay a decode graph and start the token readback without waiting for it."""
        logits = self.graphs.run(chunks, dev_ids)
        tok_dev = logits.argmax(dim=-1)
        host = self._tok_host[self._flip][:len(seqs)]
        self._flip ^= 1                              # double-buffered: t's readback vs t+1's
        host.copy_(tok_dev, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        for seq in seqs:
            seq.num_computed += 1
        self.steps += 1
        self.max_running_seen = max(self.max_running_seen, len(self.running))
        self.metrics.batch.observe(len(seqs))
        return seqs, tok_dev, host, ev

    def _resolve(self, inflight) -> None:
        seqs, _, host, ev = inflight
        ev.synchronize()
        toks = host.tolist()
        now = time.perf_counter()
        for seq, tok in zip(seqs, toks):
            if seq.done:                             # finished one step earlier: wasted token
                continue
            self._publish_full_blocks(seq)   # before accept, like the sync path: a finishing
            self._accept(seq, tok, now)      # sequence releases its blocks inside _accept

    def _can_continue(self) -> bool:
        """Steady-state decode: can step t+1 be launched before step t is read back? (Holds _cv.)"""
        seqs = [s for s in self._inflight[0] if not s.done]
        if not seqs or self.waiting or self._aborted or len(seqs) != len(self.running):
            return False
        if self.spec_tokens and not all(self._spec_cold(s) for s in seqs):
            return False                             # someone may draft: go through _spec_step
        for s in seqs:   # the pending token gets appended at resolve; leave room for one more
            if (s.params.temperature > 0 or s.num_generated + 2 > s.params.max_tokens
                    or len(s.tokens) + 2 > self.max_model_len):
                return False
        return all(self._grow(s, s.num_computed + 1) for s in seqs)

    def _continue(self) -> None:
        """Launch step t+1 from step t's on-GPU tokens, then read back and process step t."""
        prev = self._inflight
        seqs, tok_dev = prev[0], prev[1]
        live = [i for i, s in enumerate(seqs) if not s.done]
        nxt = [seqs[i] for i in live]
        ids = tok_dev if len(live) == len(seqs) else tok_dev[torch.tensor(live, device=tok_dev.device)]
        chunks = [SeqChunk([0], s.num_computed, s.blocks, True) for s in nxt]
        self._inflight = self._launch(nxt, chunks, ids)
        self.overlapped_steps += 1
        self._resolve(prev)

    def _loop(self) -> None:
        while not self._stop:
            try:
                with self._cv:            # only the queues are shared with request threads
                    while (not self.waiting and not self.running and not self._aborted
                           and not self._stop and self._inflight is None):
                        self._cv.wait()
                    if self._stop:
                        return
                    cont = self._inflight is not None and self._can_continue()
                if cont:
                    self._continue()
                    self._flush()
                    continue
                if self._inflight is not None:  # leaving steady state: catch up synchronously
                    prev, self._inflight = self._inflight, None
                    self._resolve(prev)
                    self._flush()
                with self._cv:
                    plan = self._schedule()
                if not plan:
                    self._flush()                # e.g. a "does not fit" error emitted by _schedule
                    continue
                chunks = [SeqChunk(s.tokens[s.num_computed:s.num_computed + n], s.num_computed, s.blocks,
                                   want_logits=s.num_computed + n == len(s.tokens)) for s, n in plan]
                if (self.spec_tokens and all(s.params.temperature <= 0 for s, _ in plan)
                        and self.graphs.can_run(chunks) and self._spec_step(plan)):
                    pass
                elif self._overlappable(plan, chunks):
                    self._inflight = self._launch([s for s, _ in plan], chunks)
                else:
                    self._step(plan)
                self._flush()
            except Exception as e:  # fail every in-flight request rather than hang them
                self._inflight = None
                with self._cv:
                    for seq in self.running + list(self.waiting):
                        self._release(seq)
                        self._emit(seq, e)
                    self.running.clear()
                    self.waiting.clear()
                    self._flush()
