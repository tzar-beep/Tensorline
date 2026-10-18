"""CUDA-graph capture and replay of the decode step.

At 0.5B parameters a decode step is a few hundred small kernels, and launching them from
Python costs far more than the GPU work. A CUDA graph records the whole step once and
replays it with a single launch.

Graphs need fixed shapes and fixed memory addresses, so:
  - the batch is padded up to a bucket (1, 2, 4, ... max_num_seqs) and the block-table width
    up to a power of two. One graph is captured lazily per (batch, width) bucket;
  - all inputs live in one int64 device buffer per graph, filled by a single host-to-device copy;
  - padding rows read from and write to PagedModel's scratch KV block, never a real sequence;
  - every graph writes its logits into one shared output buffer, and all graphs share one
    memory pool, so capturing more buckets costs almost no extra memory. This is safe because
    graphs only ever replay one at a time on the scheduler thread.
"""
from __future__ import annotations

import torch

from .paged import PagedModel, SeqChunk


def _bucket(n: int, buckets: list[int]) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} exceeds largest bucket {buckets[-1]}")


def _pow2_upto(limit: int) -> list[int]:
    out, b = [], 1
    while b < limit:
        out.append(b)
        b *= 2
    return out + [limit]


class CudaGraphDecoder:
    def __init__(self, pm: PagedModel, max_batch: int, max_blocks_per_seq: int):
        self.pm = pm
        self.batch_buckets = _pow2_upto(max_batch)
        self.block_buckets = _pow2_upto(max_blocks_per_seq)
        self.pool = torch.cuda.graph_pool_handle()
        vocab = pm.hf.lm_head.out_features
        self.out = torch.empty(max_batch, vocab, device=pm.device, dtype=pm.dtype)
        self.graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self.replays = 0
        # One warm-up stream for every capture: each new stream gets its own cuBLAS workspace
        # (~8 MiB each), which leaked ~260 MiB across 42 captures before this was shared.
        self.side = torch.cuda.Stream()

    def can_run(self, chunks: list[SeqChunk]) -> bool:
        return (len(chunks) <= self.batch_buckets[-1]
                and all(len(c.token_ids) == 1 and c.start > 0 and c.want_logits for c in chunks)
                and max(len(c.block_table) for c in chunks) <= self.block_buckets[-1])

    def _padding(self, B: int, NB: int) -> list[int]:
        sb = self.pm.scratch_block
        # ids, pos, slots, lens, then the [B, NB] block table
        return [0] * B + [0] * B + [sb * self.pm.block_size] * B + [1] * B + [sb] * (B * NB)

    @torch.inference_mode()
    def _capture(self, B: int, NB: int):
        dev = self.pm.device
        buf = torch.tensor(self._padding(B, NB), dtype=torch.long, device=dev)
        host = torch.empty(buf.numel(), dtype=torch.long).pin_memory()
        ids, pos, slots, lens = (buf[i * B:(i + 1) * B] for i in range(4))
        tables = buf[4 * B:].view(B, NB)

        def step():
            self.out[:B].copy_(self.pm.decode_core(ids, pos, slots, tables, lens))

        side = self.side
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):          # warm-up: lazy inits, cuBLAS handles, allocator
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
            step()
        self.graphs[(B, NB)] = (g, buf, host)

    def warmup(self, max_context_tokens: int) -> int:
        """Capture every batch bucket for contexts up to `max_context_tokens` before serving, so
        early requests don't pay ~50-100 ms of lazy capture per new bucket (it showed up as
        multi-second p99 TTFT). Capture runs only touch the scratch KV block. Returns #graphs."""
        nb_max = -(-max_context_tokens // self.pm.block_size)
        nbs = [nb for nb in self.block_buckets if nb <= nb_max] or self.block_buckets[:1]
        # Largest first: later (smaller) captures reuse the shared pool's blocks.
        for NB in reversed(nbs):
            for B in reversed(self.batch_buckets):
                if (B, NB) not in self.graphs:
                    self._capture(B, NB)
        torch.cuda.synchronize()
        return len(self.graphs)

    @torch.inference_mode()
    def run(self, chunks: list[SeqChunk], dev_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Replay the bucket's graph. `dev_ids` (optional, [n] on the GPU) overrides the chunks'
        token ids with a device-to-device copy, used by the overlap scheduler to feed step t's
        sampled tokens into step t+1 without a round trip through the host."""
        n = len(chunks)
        B = _bucket(n, self.batch_buckets)
        NB = _bucket(max(len(c.block_table) for c in chunks), self.block_buckets)
        if (B, NB) not in self.graphs:
            self._capture(B, NB)
        g, buf, host = self.graphs[(B, NB)]

        bs, sb = self.pm.block_size, self.pm.scratch_block
        vals = self._padding(B, NB)
        for i, c in enumerate(chunks):
            p = c.start
            vals[i] = c.token_ids[0]
            vals[B + i] = p
            vals[2 * B + i] = c.block_table[p // bs] * bs + p % bs
            vals[3 * B + i] = p + 1
            row = 4 * B + i * NB
            vals[row:row + len(c.block_table)] = c.block_table
        host.copy_(torch.tensor(vals, dtype=torch.long))
        buf.copy_(host, non_blocking=True)
        if dev_ids is not None:
            buf[:n].copy_(dev_ids)
        g.replay()
        self.replays += 1
        return self.out[:n]
