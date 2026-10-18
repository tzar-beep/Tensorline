"""Triton paged-attention decode kernel vs the gather + SDPA reference."""
import pytest
import torch
import torch.nn.functional as F

from tensorline import paged_attn

pytestmark = pytest.mark.skipif(not paged_attn.available(), reason="needs CUDA + Triton")


def reference(q, kc, vc, tables, lens, scale):
    B, H, D = q.shape
    NB, BS, n_kv = tables.shape[1], kc.shape[1], kc.shape[2]
    L = NB * BS
    k = kc[tables].view(B, L, n_kv, D).transpose(1, 2)
    v = vc[tables].view(B, L, n_kv, D).transpose(1, 2)
    mask = (torch.arange(L, device=q.device)[None] < lens[:, None])[:, None, None, :]
    qg = q.view(B, n_kv, H // n_kv, D)
    return F.scaled_dot_product_attention(qg, k, v, attn_mask=mask, scale=scale).reshape(B, H, D)


@pytest.mark.parametrize("heads", [(14, 2), (12, 2), (16, 2), (4, 4)])   # Qwen 0.5B/1.5B/3B, MHA
@pytest.mark.parametrize("lens", [[1], [15, 16, 17], [63, 64, 65, 300], [1400] * 3 + [2, 129]])
def test_paged_decode_matches_reference(heads, lens):
    torch.manual_seed(0)
    H, n_kv = heads
    D, BS = 64, 16
    nb = max(-(-n // BS) for n in lens)
    num_blocks = len(lens) * nb + 8
    kc = torch.randn(num_blocks, BS, n_kv, D, device="cuda", dtype=torch.float16)
    vc = torch.randn_like(kc)
    perm = torch.randperm(num_blocks, device="cuda")[: len(lens) * nb]    # scattered blocks
    tables = perm.view(len(lens), nb)
    lens_t = torch.tensor(lens, device="cuda")
    q = torch.randn(len(lens), H, D, device="cuda", dtype=torch.float16)
    got = paged_attn.paged_decode(q, kc, vc, tables, lens_t, D ** -0.5)
    want = reference(q, kc, vc, tables, lens_t, D ** -0.5)
    torch.testing.assert_close(got.float(), want.float(), atol=2e-3, rtol=2e-3)
