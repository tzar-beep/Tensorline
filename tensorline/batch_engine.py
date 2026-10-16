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
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import torch

from .engine import Engine, GenStats, SamplingParams, sample
from .paged import BlockAllocator, PagedModel, SeqChunk


@dataclass(eq=False)
class Sequence:
    sid: int
    prompt_len: int
    tokens: list[int]
    params: SamplingParams
    stops: set[int]
    out: queue.Queue
    stats: GenStats
    t_submit: float
    blocks: list[int] = field(default_factory=list)
    num_computed: int = 0      # tokens whose K/V is in the cache

    @property
    def num_generated(self) -> int:
        return len(self.tokens) - self.prompt_len


class BatchEngine(Engine):
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", *,
                 block_size: int = 16, num_blocks: int | None = None,
                 kv_mem_fraction: float = 0.6, max_num_seqs: int = 64,
                 max_batched_tokens: int = 2048, max_model_len: int = 4096):
        super().__init__(model_id, device, dtype)
        if num_blocks is None:
            per_block = PagedModel.bytes_per_block(self.model, block_size)
            if self.device == "cuda":
                free, _ = torch.cuda.mem_get_info()
                num_blocks = int(free * kv_mem_fraction) // per_block
            else:
                num_blocks = (512 << 20) // per_block   # 512 MiB on CPU
        self.paged = PagedModel(self.model, num_blocks, block_size)
        self.alloc = BlockAllocator(num_blocks)
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
        threading.Thread(target=self._loop, daemon=True, name="tl-scheduler").start()

    # ---- public API (same shape as the naive Engine) ----
    def stream(self, prompt: str, params: SamplingParams, stats: GenStats) -> Iterator[str]:
        ids = self.tokenizer(prompt).input_ids
        if len(ids) >= self.max_model_len:
            raise ValueError(f"prompt has {len(ids)} tokens; max_model_len is {self.max_model_len}")
        seq = Sequence(next(self._ids), len(ids), list(ids), params,
                       params.stop_token_ids or self.default_stops, queue.Queue(), stats,
                       time.perf_counter())
        stats.prompt_tokens = len(ids)
        with self._cv:
            self.waiting.append(seq)
            self._cv.notify()
        out_ids, emitted = [], ""
        while (tok := seq.out.get()) is not None:
            if isinstance(tok, Exception):
                raise tok
            out_ids.append(tok)
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            if not text.endswith("�"):
                delta, emitted = text[len(emitted):], text
                if delta:
                    yield delta

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

    def _preempt(self, seq: Sequence) -> None:
        """Recompute-style preemption: drop the KV cache, requeue at the front."""
        self.running.remove(seq)
        self._release(seq)
        seq.num_computed = 0
        self.waiting.appendleft(seq)
        self.preemptions += 1

    def _schedule(self) -> list[tuple[Sequence, int]]:
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
            n = min(len(seq.tokens) - seq.num_computed, budget)
            if not self._grow(seq, seq.num_computed + n):
                if not self.running:              # can never fit, even alone
                    self.waiting.popleft()
                    seq.out.put(RuntimeError("request does not fit in the KV cache"))
                    continue
                break                             # wait for blocks to free up
            self.waiting.popleft()
            self.running.append(seq)
            plan.append((seq, n))
            budget -= n
        return plan

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.stats.finish_reason = reason
        seq.stats.completion_tokens = seq.num_generated
        seq.stats.total_s = time.perf_counter() - seq.t_submit
        self.running.remove(seq)
        self._release(seq)
        seq.out.put(None)

    def _step(self, plan: list[tuple[Sequence, int]]) -> None:
        if not plan:
            return
        chunks = []
        for seq, n in plan:
            end = seq.num_computed + n
            chunks.append(SeqChunk(seq.tokens[seq.num_computed:end], seq.num_computed,
                                   seq.blocks, want_logits=end == len(seq.tokens)))
        logits = self.paged.forward(chunks)
        self.steps += 1
        self.max_running_seen = max(self.max_running_seen, len(self.running))

        sampled = [s for (s, _), c in zip(plan, chunks) if c.want_logits]
        greedy = logits.argmax(dim=-1).tolist() if sampled else []
        for (seq, n) in plan:
            seq.num_computed += n
        now = time.perf_counter()
        for i, seq in enumerate(sampled):
            tok = greedy[i] if seq.params.temperature <= 0 else sample(logits[i], seq.params)
            if seq.num_generated == 0:
                seq.stats.ttft_s = now - seq.t_submit
            if tok in seq.stops:
                self._finish(seq, "stop")
                continue
            seq.tokens.append(tok)
            seq.out.put(tok)
            if seq.num_generated >= seq.params.max_tokens or len(seq.tokens) >= self.max_model_len:
                self._finish(seq, "length")

    def _loop(self) -> None:
        while True:
            try:
                with self._cv:            # only the queues are shared with request threads
                    while not self.waiting and not self.running:
                        self._cv.wait()
                    plan = self._schedule()
                self._step(plan)
            except Exception as e:  # fail every in-flight request rather than hang them
                with self._cv:
                    for seq in self.running + list(self.waiting):
                        self._release(seq)
                        seq.out.put(e)
                    self.running.clear()
                    self.waiting.clear()
