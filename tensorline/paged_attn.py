"""Paged-attention decode kernel (Triton): one query token per sequence, K/V read in place.

The PyTorch decode path gathers each sequence's whole KV history into a padded
[B, n_kv, NB*16, d] buffer (kc[block_tables]) and then runs SDPA with a length mask. Its cost
is proportional to the *padded bucket* (up to 2x the real context), and it materialises a copy
of the KV cache every step: at batch 64 with ~1.4k-token contexts, that is gigabytes of extra
memory traffic per step.

This kernel instead walks each sequence's block table and reads K/V blocks where they live,
stopping exactly at the sequence length. One program per (sequence, KV head) handles all
`group` query heads sharing that KV head (GQA) with a single tl.dot, and does an online
softmax over 64-token tiles, as in FlashAttention / FlashDecoding. It reads only device
tensors (no host values), so it can be captured in a CUDA graph.
"""
from __future__ import annotations

import torch

from . import kernels

if kernels.available():
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, o_ptr, scale,
                             sq_b, sq_h, sk_blk, sk_tok, sk_h, sbt, so_b, so_h,
                             GROUP: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr,
                             BS: tl.constexpr, TILE: tl.constexpr):
        b, kvh = tl.program_id(0), tl.program_id(1)
        g = tl.arange(0, G_PAD)
        d = tl.arange(0, D)
        gmask = g < GROUP
        qh = kvh * GROUP + g                                          # query heads of this KV head
        q = tl.load(q_ptr + b * sq_b + qh[:, None] * sq_h + d[None, :], mask=gmask[:, None], other=0.0)
        n = tl.load(len_ptr + b)
        m_i = tl.full((G_PAD,), float("-inf"), tl.float32)
        l_i = tl.zeros((G_PAD,), tl.float32)
        acc = tl.zeros((G_PAD, D), tl.float32)
        t = tl.arange(0, TILE)
        for start in range(0, n, TILE):
            pos = start + t
            valid = pos < n
            blk = tl.load(bt_ptr + b * sbt + pos // BS, mask=valid, other=0)
            off = blk[:, None] * sk_blk + (pos % BS)[:, None] * sk_tok + kvh * sk_h + d[None, :]
            k = tl.load(k_ptr + off, mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale                         # [G_PAD, TILE]
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + off, mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        out = acc / l_i[:, None]
        tl.store(o_ptr + b * so_b + qh[:, None] * so_h + d[None, :], out.to(o_ptr.dtype.element_ty),
                 mask=gmask[:, None])


def available() -> bool:
    return kernels.available()


def paged_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                 block_tables: torch.Tensor, lens: torch.Tensor, scale: float) -> torch.Tensor:
    """q [B, n_heads, D]; k/v_cache [num_blocks, BS, n_kv, D]; block_tables [B, NB]; lens [B]
    (context length including the current token). Returns attention output [B, n_heads, D]."""
    B, H, D = q.shape
    _, BS, n_kv, _ = k_cache.shape
    group = H // n_kv
    q = q.contiguous()
    out = torch.empty_like(q)
    _paged_decode_kernel[(B, n_kv)](
        q, k_cache, v_cache, block_tables, lens, out, scale,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0), out.stride(0), out.stride(1),
        GROUP=group, G_PAD=max(16, triton.next_power_of_2(group)), D=D, BS=BS, TILE=64,
        num_warps=4)
    return out
