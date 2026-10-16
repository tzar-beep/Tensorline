"""Differential tests for Phase 2: the paged model and batching scheduler must reproduce the
reference (HF forward / naive one-at-a-time engine) exactly, under concurrency, chunked
prefill and KV-cache preemption."""
import threading

import pytest
import torch

from tensorline.batch_engine import BatchEngine
from tensorline.engine import Engine, GenStats, SamplingParams
from tensorline.paged import PagedModel, SeqChunk

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPTS = ["hello", "the quick brown fox jumps over the lazy dog", "a", "1 2 3 4 5 6 7 8 9 10 11 12",
           "paged attention", "continuous batching is iteration level scheduling", "x y", "zz"]
NO_STOP = {-1}   # random tiny model: never stop on EOS so every sequence runs full length


@pytest.fixture(scope="module")
def naive():
    return Engine(TINY, device="cpu")


@pytest.fixture(scope="module")
def reference(naive):
    out = {}
    for p in PROMPTS:
        for mt in (5, 17):
            out[(p, mt)] = "".join(naive.stream(p, SamplingParams(mt, stop_token_ids=NO_STOP), GenStats()))
    return out


def run_concurrent(engine, jobs):
    """jobs: list of (prompt, max_tokens). Returns texts in job order, submitted concurrently."""
    res, stats = [None] * len(jobs), [GenStats() for _ in jobs]

    def go(i, p, mt):
        res[i] = "".join(engine.stream(p, SamplingParams(mt, stop_token_ids=NO_STOP), stats[i]))

    ts = [threading.Thread(target=go, args=(i, p, mt)) for i, (p, mt) in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    return res, stats


@pytest.mark.parametrize("model_id", [TINY, "Qwen/Qwen2.5-0.5B-Instruct"])  # MHA, and GQA 14:2 + qkv bias
def test_paged_forward_matches_hf(model_id):
    """Prefill + several decode steps through the paged cache == HF full-sequence logits."""
    ref_engine = Engine(model_id, device="cpu")
    model, tok = ref_engine.model, ref_engine.tokenizer
    ids = tok("the quick brown fox jumps over", return_tensors="pt").input_ids[0].tolist()
    pm = PagedModel(model, num_blocks=8, block_size=4)
    table = [5, 2, 7, 0]                               # deliberately non-contiguous blocks
    seq = list(ids)
    logits = pm.forward([SeqChunk(seq, 0, table, True)])
    for _ in range(4):
        with torch.inference_mode():
            ref = model(torch.tensor([seq])).logits[0, -1]
        torch.testing.assert_close(logits[0], ref, atol=2e-3, rtol=1e-3)
        nxt = int(ref.argmax())
        seq.append(nxt)
        logits = pm.forward([SeqChunk([nxt], len(seq) - 1, table, True)])


@pytest.mark.parametrize("kw", [
    dict(),                                            # plenty of KV, big budget
    dict(max_batched_tokens=4),                        # forces chunked prefill
    dict(num_blocks=12, block_size=4),                 # forces preemption + recompute
    dict(max_num_seqs=2),                              # forces queueing
], ids=["plain", "chunked-prefill", "preemption", "seq-limit"])
def test_batched_matches_naive(reference, kw):
    eng = BatchEngine(TINY, device="cpu", **{"num_blocks": 256, **kw})
    jobs = [(p, mt) for p in PROMPTS for mt in (5, 17)]
    texts, stats = run_concurrent(eng, jobs)
    for (p, mt), t, s in zip(jobs, texts, stats):
        assert t == reference[(p, mt)], (p, mt)
        assert s.completion_tokens == mt and s.finish_reason == "length"
    assert eng.alloc.num_free == eng.alloc.num_blocks          # no leaked KV blocks
    assert not eng.running and not eng.waiting
    if "num_blocks" in kw:
        assert eng.preemptions > 0
    if kw == {}:
        assert eng.max_running_seen > 1                        # actually batched


def test_oversized_request_errors_not_hangs():
    eng = BatchEngine(TINY, device="cpu", num_blocks=2, block_size=4)
    with pytest.raises(RuntimeError):
        list(eng.stream("one two three four five six seven eight nine ten",
                        SamplingParams(4, stop_token_ids=NO_STOP), GenStats()))
