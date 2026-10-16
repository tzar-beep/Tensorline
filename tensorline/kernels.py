"""Fused dequantise-matmul Triton kernels for W8A16 / W4A16 (small M: decode and short chunks).

y[M, N] = x[M, K] @ W^T where W is stored quantised (see quant.py). At decode, streaming the weights
*is* the cost, so int8 halves it and int4 quarters it relative to fp16, provided each weight byte
is read exactly once and enough programs are in flight to saturate memory bandwidth:
  - BLOCK_M covers all M rows (16/32/64), so the weights are never re-streamed per row tile;
  - split-K: when N alone gives too few tiles to fill the GPU (e.g. N=896), the K loop is split
    across SPLIT programs that write fp32 partials [SPLIT, M, N], summed by a second tiny
    reduction. That is deterministic, with no atomics, so results are bit-reproducible.

The dequantisation is folded algebraically rather than applied per element:
  int8:  sum_k x_k * q_k * s          = s * (x . q)                    scale applied once per tile
  int4:  sum_k x_k * (q_k * s + m)    = s * (x . q) + m * sum_k x_k    per 128-wide group
so the inner loop is a plain fp16 dot over small integers (exact in fp16).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _OK = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception:  # pragma: no cover - CPU-only installs
    _OK = False


def available() -> bool:
    return _OK


if _OK:
    _configs = [triton.Config({}, num_warps=w, num_stages=s) for w in (2, 4, 8) for s in (2, 3, 4)]

    @triton.autotune(configs=_configs, key=["N", "K", "BLOCK_M", "BLOCK_N", "SPLIT"])
    @triton.jit
    def _w8a16_kernel(x_ptr, q_ptr, s_ptr, p_ptr, M, N, K, K_PER_SPLIT, sxm, sqn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      SPLIT: tl.constexpr):
        pid_n, pid_s = tl.program_id(0), tl.program_id(1)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        k_lo = pid_s * K_PER_SPLIT
        k_hi = tl.minimum(k_lo + K_PER_SPLIT, K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            kk = k0 + rk
            x = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :],
                        mask=(rm[:, None] < M) & (kk[None, :] < k_hi), other=0.0)
            q = tl.load(q_ptr + rn[:, None] * sqn + kk[None, :],
                        mask=(rn[:, None] < N) & (kk[None, :] < k_hi), other=0)
            acc += tl.dot(x, tl.trans(q.to(tl.float16)))
        s = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc = acc * s[None, :]
        tl.store(p_ptr + pid_s * M * N + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.autotune(configs=_configs, key=["N", "K", "BLOCK_M", "BLOCK_N", "SPLIT"])
    @triton.jit
    def _w4a16_kernel(x_ptr, q_ptr, s_ptr, m_ptr, p_ptr, M, N, K, G_PER_SPLIT, sxm, sqn, ssn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP: tl.constexpr,
                      SPLIT: tl.constexpr):
        HALF: tl.constexpr = GROUP // 2
        pid_n, pid_s, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rh = tl.arange(0, HALF)
        mmask, nmask = rm < M, rn < N
        g_lo = pid_s * G_PER_SPLIT
        g_hi = tl.minimum(g_lo + G_PER_SPLIT, K // GROUP)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(g_lo, g_hi):
            k0 = g * GROUP
            xl = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rh)[None, :], mask=mmask[:, None], other=0.0)
            xh = tl.load(x_ptr + rm[:, None] * sxm + (k0 + HALF + rh)[None, :], mask=mmask[:, None], other=0.0)
            p = tl.load(q_ptr + rn[:, None] * sqn + (g * HALF + rh)[None, :], mask=nmask[:, None], other=0)
            ql = (p & 0xF).to(tl.float16)
            qh = (p >> 4).to(tl.float16)
            dot = tl.dot(xl, tl.trans(ql)) + tl.dot(xh, tl.trans(qh))          # [BM, BN] fp32
            xsum = tl.sum(xl.to(tl.float32), axis=1) + tl.sum(xh.to(tl.float32), axis=1)
            s = tl.load(s_ptr + rn * ssn + g, mask=nmask, other=0.0).to(tl.float32)
            mn = tl.load(m_ptr + rn * ssn + g, mask=nmask, other=0.0).to(tl.float32)
            acc += dot * s[None, :] + xsum[:, None] * mn[None, :]
        tl.store(p_ptr + pid_s * M * N + rm[:, None] * N + rn[None, :], acc,
                 mask=mmask[:, None] & nmask[None, :])

    _SMS = torch.cuda.get_device_properties(0).multi_processor_count


def _plan(M: int, N: int, k_steps: int, max_block_m: int = 64) -> tuple[int, int, int]:
    """Tile shape and split-K factor: BLOCK_M covers M; aim for >= ~4 programs per SM."""
    block_m = min(16 if M <= 16 else 32 if M <= 32 else 64, max_block_m)
    block_n = 64 if N >= 4096 else 32
    tiles = triton.cdiv(N, block_n)
    split = 1
    while tiles * split < 4 * _SMS and split < 8 and k_steps // (split * 2) >= 2:
        split *= 2
    return block_m, block_n, split


MAX_BM_INT4 = 64   # tried 32 (M=64 as two row tiles): 0.6-0.8x vs 0.74-0.82x at 64, so kept at 64


def qmatmul(x: torch.Tensor, layer) -> torch.Tensor:
    """x [M<=64, K] (fp16, CUDA) times a QuantLinear's weight^T -> [M, N]. Bias not included."""
    x = x.contiguous()
    M, K = x.shape
    N = layer.out_features
    if layer.mode == "int8":
        BK = 64
        block_m, block_n, split = _plan(M, N, triton.cdiv(K, BK))
        k_per = triton.cdiv(triton.cdiv(K, BK), split) * BK
        part = torch.empty(split, M, N, device=x.device, dtype=torch.float32)
        _w8a16_kernel[(triton.cdiv(N, block_n), split)](
            x, layer.qweight, layer.scale, part, M, N, K, k_per, x.stride(0), layer.qweight.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BK, SPLIT=split)
    else:
        groups = K // layer.group
        block_m, block_n, split = _plan(M, N, groups, max_block_m=MAX_BM_INT4)
        g_per = triton.cdiv(groups, split)
        part = torch.empty(split, M, N, device=x.device, dtype=torch.float32)
        _w4a16_kernel[(triton.cdiv(N, block_n), split, triton.cdiv(M, block_m))](
            x, layer.qweight, layer.scale, layer.minv, part, M, N, K, g_per,
            x.stride(0), layer.qweight.stride(0), layer.scale.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, GROUP=layer.group, SPLIT=split)
    return (part[0] if split == 1 else part.sum(0)).to(x.dtype)
