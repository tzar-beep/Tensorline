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

    @property
    def num_generated(self) -> int:
        return len(self.tokens) - self.prompt_len


class BatchEngine(Engine):
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", *,
                 block_size: int = 16, num_blocks: int | None = None,
                 kv_mem_fraction: float = 0.6, max_num_seqs: int = 64,
                 max_batched_tokens: int = 2048, max_model_len: int = 4096,
                 cuda_graphs: bool = True, prefix_caching: bool = True):
        super().__init__(model_id, device, dtype)
        if num_blocks is None:
            per_block = PagedModel.bytes_per_block(self.model, block_size)
            if self.device == "cuda":
                free, _ = torch.cuda.mem_get_info()
                num_blocks = int(free * kv_mem_fraction) // per_block
            else:
                num_blocks = (512 << 20) // per_block   # 512 MiB on CPU
        self.paged = PagedModel(self.model, num_blocks, block_size)
        self.alloc = BlockAllocator(num_blocks, prefix_caching)
        self.graphs = None
        if cuda_graphs and self.device == "cuda":
            from .cuda_graphs import CudaGraphDecoder
            self.graphs = CudaGraphDecoder(self.paged, max_num_seqs, -(-max_model_len // block_size))
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
        self._pending: list[tuple[_Sink, object]] = []   # outputs produced during this step
        self._aborted: list[Sequence] = []
        threading.Thread(target=self._loop, daemon=True, name="tl-scheduler").start()

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
        while len(seq.hashes) < seq.num_computed // bs:
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
            seq.stats.finish_reason = "abort"
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
        seq.stats.finish_reason = reason
        seq.stats.completion_tokens = seq.num_generated
        seq.stats.total_s = time.perf_counter() - seq.t_submit
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

        sampled = [s for (s, _), c in zip(plan, chunks) if c.want_logits]
        greedy = logits.argmax(dim=-1).tolist() if sampled else []
        for (seq, n) in plan:
            seq.num_computed += n
            self._publish_full_blocks(seq)
        now = time.perf_counter()
        for i, seq in enumerate(sampled):
            tok = greedy[i] if seq.params.temperature <= 0 else sample(logits[i], seq.params)
            if seq.num_generated == 0:
                seq.stats.ttft_s = now - seq.t_submit
            if tok in seq.stops:
                self._finish(seq, "stop")
                continue
            seq.tokens.append(tok)
            self._emit(seq, tok)
            if seq.num_generated >= seq.params.max_tokens or len(seq.tokens) >= self.max_model_len:
                self._finish(seq, "length")

    def _loop(self) -> None:
        while True:
            try:
                with self._cv:            # only the queues are shared with request threads
                    while not self.waiting and not self.running and not self._aborted:
                        self._cv.wait()
                    plan = self._schedule()
                self._step(plan)
                self._flush()
            except Exception as e:  # fail every in-flight request rather than hang them
                with self._cv:
                    for seq in self.running + list(self.waiting):
                        self._release(seq)
                        self._emit(seq, e)
                    self.running.clear()
                    self.waiting.clear()
                    self._flush()
