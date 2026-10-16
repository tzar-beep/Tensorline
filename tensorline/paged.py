"""Paged KV cache and a paged-attention forward pass over Hugging Face weights.

The KV cache is a pool of fixed-size blocks per layer: `k[num_blocks, block_size, kv_heads, head_dim]`.
Each sequence owns a list of block ids (its block table), so memory is allocated in small
chunks as sequences grow and returned to a free list as soon as they finish. There is no
per-sequence max-length reservation and no fragmentation.

`PagedModel.forward` runs one engine step over a *flattened* batch: every scheduled token from
every sequence (a prefill chunk or a single decode token) is concatenated along one axis. The
linear layers (QKV, O, MLP) see one big matrix, and only attention is per-sequence:
  - decode tokens: one batched SDPA call over K/V gathered through the block tables;
  - prefill chunks: one causal SDPA call per chunk against that sequence's cached context.
Only the module weights are reused from HF (projections, norms, MLP, rotary embedding);
the attention, KV cache and batching are this file's.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class BlockAllocator:
    """Ref-counted KV block allocator with automatic prefix caching.

    A *full* block is identified by a content hash chained over everything before it,
    hash(parent_hash, its 16 token ids), so equal hashes mean equal K/V contents. Freed blocks
    keep their contents and hash and sit in an LRU free list: a later request with the same
    prefix takes them back (`lookup`) instead of recomputing them, and they are evicted only
    when `allocate` needs the memory. Blocks shared by several sequences are only ever *read*:
    a sequence writes K/V only at positions past its fully computed (and therefore hashed)
    prefix.
    """

    def __init__(self, num_blocks: int, prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.prefix_caching = prefix_caching
        self._free: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))  # LRU first
        self._ref = [0] * num_blocks
        self._hash_of: list[int | None] = [None] * num_blocks
        self._cached: dict[int, int] = {}            # content hash -> block id

    @property
    def num_free(self) -> int:
        """Blocks with no owner, including evictable cached ones."""
        return len(self._free)

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    def allocate(self, n: int) -> list[int] | None:
        if n > len(self._free):
            return None
        out = []
        for _ in range(n):
            blk, _ = self._free.popitem(last=False)  # least recently freed
            h = self._hash_of[blk]
            if h is not None:                        # evict its cached contents
                del self._cached[h]
                self._hash_of[blk] = None
            self._ref[blk] = 1
            out.append(blk)
        return out

    def free(self, blocks: list[int]) -> None:
        # Tail blocks are the least shareable, so they go to the front of the LRU (evicted first).
        for blk in reversed(blocks):
            self._ref[blk] -= 1
            assert self._ref[blk] >= 0, f"double free of block {blk}"
            if self._ref[blk] == 0:
                self._free[blk] = None

    def lookup(self, h: int) -> int | None:
        """Take a reference on the cached block with content hash `h`, if any."""
        blk = self._cached.get(h) if self.prefix_caching else None
        if blk is None:
            return None
        if self._ref[blk] == 0:
            del self._free[blk]
        self._ref[blk] += 1
        return blk

    def register(self, blk: int, h: int) -> None:
        """Publish a fully computed block under its content hash."""
        if self.prefix_caching and h not in self._cached:
            self._cached[h] = blk
            self._hash_of[blk] = h


def block_hash(parent: int | None, tokens) -> int:
    return hash((parent, tuple(tokens)))


@dataclass
class SeqChunk:
    """One sequence's share of a step: tokens [start, start+n) at those positions."""
    token_ids: list[int]
    start: int              # position of first token == tokens already in the KV cache
    block_table: list[int]
    want_logits: bool       # True if the last token of this chunk should be sampled from


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class PagedModel:
    def __init__(self, hf_model, num_blocks: int, block_size: int = 16):
        self.hf = hf_model
        self.inner = hf_model.model
        cfg = hf_model.config
        attn0 = self.inner.layers[0].self_attn
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = attn0.head_dim
        self.group = self.n_heads // self.n_kv
        self.scale = attn0.scaling
        self.block_size = block_size
        self.num_blocks = num_blocks
        p = next(hf_model.parameters())
        self.device, self.dtype = p.device, p.dtype
        # One extra block past the allocator's range: a scratch sink for the K/V writes of
        # padding rows in fixed-shape (CUDA-graph) batches. It is never part of a real sequence.
        self.scratch_block = num_blocks
        shape = (num_blocks + 1, block_size, self.n_kv, self.head_dim)
        self.k_cache = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in self.inner.layers]
        self.v_cache = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in self.inner.layers]

    @staticmethod
    def bytes_per_block(hf_model, block_size: int = 16) -> int:
        cfg = hf_model.config
        a = hf_model.model.layers[0].self_attn
        elem = next(hf_model.parameters()).element_size()
        return 2 * cfg.num_hidden_layers * block_size * cfg.num_key_value_heads * a.head_dim * elem

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """[ctx, n_kv, d] -> [1, n_heads, ctx, d], repeating each KV head for its query group."""
        ctx = x.shape[0]
        x = x.transpose(0, 1)[:, None].expand(self.n_kv, self.group, ctx, self.head_dim)
        return x.reshape(1, self.n_heads, ctx, self.head_dim)

    def decode_core(self, ids, pos, slots, tables, lens) -> torch.Tensor:
        """Pure-tensor decode step for B sequences, one token each, with fixed shapes and no host
        syncs, so it can be captured in a CUDA graph. ids/pos/slots/lens: [B]; tables: [B, NB].
        Returns logits [B, vocab]."""
        B, NB = tables.shape
        L = NB * self.block_size
        mask = (torch.arange(L, device=ids.device)[None, :] < lens[:, None])[:, None, None, :]
        x = self.inner.embed_tokens(ids)
        cos, sin = self.inner.rotary_emb(x[None], pos[None])
        cos, sin = cos[0][:, None, :], sin[0][:, None, :]
        for layer, kc, vc in zip(self.inner.layers, self.k_cache, self.v_cache):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_proj(h).view(B, self.n_heads, self.head_dim)
            k = attn.k_proj(h).view(B, self.n_kv, self.head_dim)
            v = attn.v_proj(h).view(B, self.n_kv, self.head_dim)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            kc.view(-1, self.n_kv, self.head_dim).index_copy_(0, slots, k)
            vc.view(-1, self.n_kv, self.head_dim).index_copy_(0, slots, v)
            kd = kc[tables].view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
            vd = vc[tables].view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
            qd = q.view(B, self.n_kv, self.group, self.head_dim)
            od = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask, scale=self.scale)
            x = x + attn.o_proj(od.reshape(B, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.hf.lm_head(self.inner.norm(x))

    @torch.inference_mode()
    def forward(self, chunks: list[SeqChunk]) -> torch.Tensor:
        """Run one step. Returns logits [num_chunks_wanting_logits, vocab] in chunk order."""
        dev, bs = self.device, self.block_size
        ids, pos, slots = [], [], []
        decode_rows, prefill = [], []           # decode: (row, chunk); prefill: (row0, chunk)
        for c in chunks:
            row0 = len(ids)
            for i, t in enumerate(c.token_ids):
                p_ = c.start + i
                ids.append(t)
                pos.append(p_)
                slots.append(c.block_table[p_ // bs] * bs + p_ % bs)
            (decode_rows if len(c.token_ids) == 1 and c.start > 0 else prefill).append((row0, c))
        T = len(ids)
        ids_t = torch.tensor(ids, device=dev)
        pos_t = torch.tensor(pos, device=dev)
        slots_t = torch.tensor(slots, device=dev)

        if decode_rows:
            d_idx = torch.tensor([r for r, _ in decode_rows], device=dev)
            max_blocks = max(len(c.block_table) for _, c in decode_rows)
            d_tables = torch.tensor([c.block_table + [0] * (max_blocks - len(c.block_table))
                                     for _, c in decode_rows], device=dev)
            d_lens = torch.tensor([c.start + 1 for _, c in decode_rows], device=dev)
            L = max_blocks * bs
            d_mask = (torch.arange(L, device=dev)[None, :] < d_lens[:, None])[:, None, None, :]

        x = self.inner.embed_tokens(ids_t)                       # [T, H]
        cos, sin = self.inner.rotary_emb(x[None], pos_t[None])    # [1, T, d]
        cos, sin = cos[0][:, None, :], sin[0][:, None, :]

        for layer, kc, vc in zip(self.inner.layers, self.k_cache, self.v_cache):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_proj(h).view(T, self.n_heads, self.head_dim)
            k = attn.k_proj(h).view(T, self.n_kv, self.head_dim)
            v = attn.v_proj(h).view(T, self.n_kv, self.head_dim)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            kc.view(-1, self.n_kv, self.head_dim)[slots_t] = k
            vc.view(-1, self.n_kv, self.head_dim)[slots_t] = v

            o = torch.empty_like(q)
            if decode_rows:
                # GQA without copying K/V: a decode step has one query per head, so the
                # `group` query heads sharing a KV head are folded into SDPA's query-length
                # axis: q [B, n_kv, group, d] against k [B, n_kv, L, d]. (Passing enable_gqa
                # with a mask instead drops SDPA to its slow math kernel.)
                B = len(decode_rows)
                kd = kc[d_tables].view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
                vd = vc[d_tables].view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
                qd = q[d_idx].view(B, self.n_kv, self.group, self.head_dim)
                od = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=d_mask, scale=self.scale)
                o[d_idx] = od.reshape(B, self.n_heads, self.head_dim)
            for row0, c in prefill:
                n, ctx = len(c.token_ids), c.start + len(c.token_ids)
                nb = (ctx + bs - 1) // bs
                bt = torch.tensor(c.block_table[:nb], device=dev)
                kp = self._expand_kv(kc[bt].view(-1, self.n_kv, self.head_dim)[:ctx])
                vp = self._expand_kv(vc[bt].view(-1, self.n_kv, self.head_dim)[:ctx])
                qp = q[row0:row0 + n].transpose(0, 1)[None]      # [1, nh, n, d]
                if c.start == 0:
                    op = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True, scale=self.scale)
                else:  # chunked prefill / recompute: query i sees keys <= start + i
                    m = (torch.arange(ctx, device=dev)[None, :]
                         <= (c.start + torch.arange(n, device=dev))[:, None])
                    op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=m, scale=self.scale)
                o[row0:row0 + n] = op[0].transpose(0, 1)
            x = x + attn.o_proj(o.reshape(T, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        rows, r = [], 0
        for c in chunks:
            r += len(c.token_ids)
            if c.want_logits:
                rows.append(r - 1)
        if not rows:
            return torch.empty(0, device=dev)
        h = self.inner.norm(x[torch.tensor(rows, device=dev)])
        return self.hf.lm_head(h)
