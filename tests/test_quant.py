"""Quantisation formats and fused kernels."""
import pytest
import torch
import torch.nn as nn

from tensorline import kernels
from tensorline.quant import QuantLinear, dequantize_int4, dequantize_int8, quantize_int4, quantize_int8

gpu = pytest.mark.skipif(not kernels.available(), reason="needs CUDA + Triton")


def test_int8_roundtrip_error_bounded_by_half_step():
    w = torch.randn(64, 256)
    q, s = quantize_int8(w)
    err = (dequantize_int8(q, s, torch.float32) - w).abs()
    assert torch.all(err <= s[:, None] / 2 + 1e-6)


def test_int4_roundtrip_error_bounded_and_packing_layout():
    w = torch.randn(32, 512)
    packed, s, mn = quantize_int4(w)
    assert packed.shape == (32, 256) and packed.dtype == torch.uint8
    err = (dequantize_int4(packed, s, mn, torch.float32) - w).abs().view(32, 4, 128)
    assert torch.all(err <= s[..., None] / 2 + 1e-5)
    # byte j of a group holds k=j (low nibble) and k=j+64 (high nibble)
    w2 = torch.zeros(1, 128)
    w2[0, 0], w2[0, 64] = 1.0, 0.5
    p, s2, m2 = quantize_int4(w2)
    assert int(p[0, 0]) & 0xF == 15 and int(p[0, 0]) >> 4 == round(0.5 / s2.item())


@gpu
@pytest.mark.parametrize("mode", ["int8", "int4"])
@pytest.mark.parametrize("M", [1, 3, 16, 17, 64])
@pytest.mark.parametrize("NK", [(128, 896), (896, 4864), (4864, 896)])
def test_fused_kernel_matches_dequant_reference(mode, M, NK):
    torch.manual_seed(0)
    N, K = NK
    lin = nn.Linear(K, N, bias=True).cuda().half()
    ql = QuantLinear(lin, mode, torch.float16)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    ref = torch.nn.functional.linear(x.float(), ql.dequantize().float(), ql.bias.float())
    out = ql(x)                                            # M <= 64 -> fused Triton path
    torch.testing.assert_close(out.float(), ref, atol=2e-2 * ref.abs().max().item(), rtol=1e-2)


@gpu
def test_fused_kernel_captures_in_cuda_graph():
    lin = nn.Linear(896, 896).cuda().half()
    ql = QuantLinear(lin, "int4", torch.float16)
    x = torch.randn(4, 896, device="cuda", dtype=torch.float16)
    with torch.inference_mode():                           # as in CudaGraphDecoder
        eager = ql(x).clone()                              # also triggers autotuning outside capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = ql(x)
        g.replay()
    torch.testing.assert_close(y, eager)


QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


@gpu
@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_quantized_paged_forward_matches_hf_forward(mode):
    """Same quantised weights through our paged path (fused kernels at decode) and HF's forward
    (dequantise + cuBLAS): logits must agree."""
    from tensorline.engine import Engine
    from tensorline.paged import PagedModel, SeqChunk
    eng = Engine(QWEN, device="cuda", quant=mode)
    ids = eng.tokenizer("The capital of France is Paris. The capital of Japan is").input_ids
    pm = PagedModel(eng.model, num_blocks=8, block_size=16)
    with torch.inference_mode():
        logits = pm.forward([SeqChunk(ids, 0, [3, 1], True)])
        for _ in range(4):
            ref = eng.model(torch.tensor([ids], device="cuda")).logits[0, -1]
            torch.testing.assert_close(logits[0].float(), ref.float(), atol=0.15, rtol=0.05)
            nxt = int(ref.argmax())
            assert int(logits[0].argmax()) == nxt
            ids.append(nxt)
            logits = pm.forward([SeqChunk([nxt], len(ids) - 1, [3, 1], True)])


@gpu
@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_quantized_engine_graphs_match_eager(mode):
    from tensorline.batch_engine import BatchEngine
    from tests.test_batching import PROMPTS, run_concurrent
    jobs = [(p, 20) for p in PROMPTS]
    out = {}
    for graphs in (False, True):
        eng = BatchEngine(QWEN, device="cuda", num_blocks=256, cuda_graphs=graphs, quant=mode)
        out[graphs], _ = run_concurrent(eng, jobs)
        del eng
        torch.cuda.empty_cache()
    assert out[True] == out[False]
