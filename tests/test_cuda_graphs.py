"""CUDA-graph decode must match the eager paged path (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tensorline.batch_engine import BatchEngine  # noqa: E402
from tensorline.engine import Engine  # noqa: E402
from tensorline.paged import PagedModel, SeqChunk  # noqa: E402
from tests.test_batching import PROMPTS, run_concurrent  # noqa: E402

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen():
    return Engine(QWEN, device="cuda")


def test_graph_decode_matches_eager_and_isolates_padding(qwen):
    from tensorline.cuda_graphs import CudaGraphDecoder
    bs = 16
    pm = PagedModel(qwen.model, num_blocks=32, block_size=bs)
    gd = CudaGraphDecoder(pm, max_batch=8, max_blocks_per_seq=8)
    prompts = ["hi", "Explain paged attention in detail, step by step, for a systems engineer.",
               "One two three four five six seven eight nine ten eleven twelve thirteen"]
    seqs = [qwen.tokenizer(p).input_ids for p in prompts]
    tables = [[0, 1, 2], [5, 9, 3], [20, 11, 4]]           # non-contiguous on purpose
    bystander = 30                                          # a block no sequence in the batch owns
    pm.k_cache[0][bystander].fill_(7.0)
    with torch.inference_mode():
        logits = pm.forward([SeqChunk(s, 0, t, True) for s, t in zip(seqs, tables)])
    for step in range(20):                                  # crosses a block boundary
        nxt = logits.argmax(-1).tolist()
        for s, t in zip(seqs, nxt):
            s.append(t)
        chunks = [SeqChunk([s[-1]], len(s) - 1, t, True) for s, t in zip(seqs, tables)]
        with torch.inference_mode():
            eager = pm.forward(chunks).clone()
        graph = gd.run(chunks).clone()                      # batch 3 -> bucket 4: one padding row
        assert graph.argmax(-1).tolist() == eager.argmax(-1).tolist(), step
        torch.testing.assert_close(graph.float(), eager.float(), atol=5e-2, rtol=1e-2)
        logits = graph
    assert torch.all(pm.k_cache[0][bystander] == 7.0)      # padding rows only touched scratch
    assert gd.replays == 20 and len(gd.graphs) >= 1


def test_engine_with_graphs_matches_eager_engine():
    jobs = [(p, 24) for p in PROMPTS]
    out = {}
    for graphs in (False, True):
        eng = BatchEngine(QWEN, device="cuda", num_blocks=256, cuda_graphs=graphs)
        out[graphs], stats = run_concurrent(eng, jobs)
        assert all(s.completion_tokens > 0 for s in stats)
        if graphs:
            assert eng.graphs.replays > 0
        del eng
        torch.cuda.empty_cache()
    assert out[True] == out[False]
