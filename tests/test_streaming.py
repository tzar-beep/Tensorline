"""Streaming path of the batch engine: incremental detokenization, the async (thread-free) server
path, and cancellation when a client goes away."""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from tensorline.batch_engine import BatchEngine
from tensorline.detok import IncrementalDetokenizer
from tensorline.engine import GenStats, SamplingParams
from tensorline.server import create_app
from tests.test_batching import NO_STOP, TINY

TEXT = "Paged KV caches 🚀 — 分页注意力 naïve café ﬁne\n  indented\tTAB 12345 ‘quotes’ ∑∫"


@pytest.mark.parametrize("model_id", [TINY, "Qwen/Qwen2.5-0.5B-Instruct"])  # SentencePiece, byte BPE
def test_incremental_detok_matches_full_decode(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(TEXT, add_special_tokens=False).input_ids
    d = IncrementalDetokenizer(tok)
    out = "".join(d.push([i]) for i in ids)
    assert out == tok.decode(ids, skip_special_tokens=True)
    for k in (2, 3, 5):                                    # coalesced pushes give the same text
        d = IncrementalDetokenizer(tok)
        assert "".join(d.push(ids[i:i + k]) for i in range(0, len(ids), k)) == out


@pytest.fixture(scope="module")
def engine():
    return BatchEngine(TINY, device="cpu", num_blocks=64, block_size=4)


def test_api_on_batch_engine_stream_equals_nonstream(engine):
    client = TestClient(create_app(engine))
    body = {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 10}
    full = client.post("/v1/chat/completions", json=body).json()
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: {")]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == full["choices"][0]["message"]["content"]
    u_stream, u_full = events[-1]["usage"], full["usage"]
    assert u_stream["completion_tokens"] == u_full["completion_tokens"]
    assert u_full["prompt_tokens_details"]["cached_tokens"] == 0
    assert u_stream["prompt_tokens_details"]["cached_tokens"] > 0   # same prompt: prefix-cache hit


def test_abandoned_stream_is_aborted_and_frees_blocks(engine):
    async def consume_two_then_leave():
        gen = engine.astream("count to a thousand", SamplingParams(500, stop_token_ids=NO_STOP), st)
        got = 0
        async for _ in gen:
            got += 1
            if got == 2:
                break
        await gen.aclose()

    st = GenStats()
    asyncio.run(consume_two_then_leave())
    deadline = time.time() + 10
    while (engine.running or engine.waiting) and time.time() < deadline:
        time.sleep(0.01)
    assert not engine.running and not engine.waiting
    assert st.finish_reason == "abort"
    assert engine.alloc.num_free == engine.alloc.num_blocks
