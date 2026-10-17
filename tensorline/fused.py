"""Fused elementwise kernels for the decode step (Triton, with PyTorch fallbacks for CPU).

Profiled inside a CUDA graph at batch 1 (Qwen2.5-0.5B), a decode step spent ~1.4 ms of 8.9 ms
on small elementwise work: Hugging Face's RMSNorm is 7 kernels (cast, pow, mean, add, rsqrt,
mul, cast, mul) x 48 norms, and rotate-half RoPE is ~12 kernels per layer. Each kernel moves only
a few KB, so the cost is per-launch latency, not bandwidth. Fusing turns each norm (plus the
residual add before it) and each layer's RoPE into a single kernel.
"""
from __future__ import annotations

import torch

from . import kernels

if kernels.available():
    import triton
    import triton.language as tl

    @triton.jit
    def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, H, eps,
                            HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < H
        x = tl.load(x_ptr + row * H + cols, mask=m, other=0.0)
        if HAS_RES:                                   # x <- x + r, written back as the new residual
            x = x + tl.load(r_ptr + row * H + cols, mask=m, other=0.0)
            tl.store(x_ptr + row * H + cols, x, mask=m)
        xf = x.to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / H + eps)
        w = tl.load(w_ptr + cols, mask=m, other=0.0)
        tl.store(out_ptr + row * H + cols, (xf * rstd).to(x.dtype) * w, mask=m)   # HF's cast order

    @triton.jit
    def _rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, NQ, NK, D: tl.constexpr):
        t, h = tl.program_id(0), tl.program_id(1)
        HALF: tl.constexpr = D // 2
        i = tl.arange(0, HALF)
        c1 = tl.load(cos_ptr + t * D + i).to(tl.float32)
        c2 = tl.load(cos_ptr + t * D + HALF + i).to(tl.float32)
        s1 = tl.load(sin_ptr + t * D + i).to(tl.float32)
        s2 = tl.load(sin_ptr + t * D + HALF + i).to(tl.float32)
        if h < NQ:
            base = q_ptr + (t * NQ + h) * D
        else:
            base = k_ptr + (t * NK + (h - NQ)) * D
        a = tl.load(base + i)
        b = tl.load(base + HALF + i)
        af, bf = a.to(tl.float32), b.to(tl.float32)
        tl.store(base + i, (af * c1 - bf * s1).to(a.dtype))
        tl.store(base + HALF + i, (bf * c2 + af * s2).to(a.dtype))


def _fast(x: torch.Tensor) -> bool:
    return kernels.available() and x.is_cuda and x.is_contiguous()


def add_rmsnorm(x: torch.Tensor, residual: torch.Tensor | None, norm) -> torch.Tensor:
    """If `residual` is given: x += residual (in place). Returns RMSNorm(x) with `norm`'s weight."""
    if not _fast(x) or (residual is not None and not residual.is_contiguous()):
        if residual is not None:
            x.add_(residual)
        return norm(x)
    H = x.shape[-1]
    out = torch.empty_like(x)
    _add_rmsnorm_kernel[(x.numel() // H,)](x, residual if residual is not None else x, norm.weight, out,
                                           H, norm.variance_epsilon, HAS_RES=residual is not None,
                                           BLOCK=triton.next_power_of_2(H))
    return out


def rope_(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate-half RoPE in place. q [T, nq, D], k [T, nk, D], cos/sin [T, D]."""
    if not (_fast(q) and _fast(k) and cos.is_contiguous() and sin.is_contiguous()):
        c, s = cos[:, None, :], sin[:, None, :]
        q.copy_(q * c + _rotate_half(q) * s)
        k.copy_(k * c + _rotate_half(k) * s)
        return
    T, NQ, D = q.shape
    _rope_kernel[(T, NQ + k.shape[1])](q, k, cos, sin, NQ, k.shape[1], D=D)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
