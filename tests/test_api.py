"""End-to-end API tests against a tiny random model (fast, no GPU needed)."""
import json

import pytest
from fastapi.testclient import TestClient

from tensorline.engine import Engine, GenStats, SamplingParams
from tensorline.server import create_app

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def engine():
    return Engine(TINY, device="cpu")


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine))


def body(**kw):
    return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, **kw}


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["data"][0]["id"] == TINY


def test_non_streaming_shape(client):
    r = client.post("/v1/chat/completions", json=body()).json()
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"]["role"] == "assistant"
    u = r["usage"]
    assert 0 < u["completion_tokens"] <= 8
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_streaming_matches_non_streaming(client):
    """Greedy decoding is deterministic: concatenated SSE deltas == non-stream content."""
    full = client.post("/v1/chat/completions", json=body()).json()["choices"][0]["message"]["content"]
    with client.stream("POST", "/v1/chat/completions", json=body(stream=True)) as r:
        lines = [l[6:] for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(l) for l in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == full


def test_kv_cache_decode_matches_hf_generate(engine):
    """Our hand-written decode loop must reproduce HF greedy generate token-for-token."""
    import torch
    prompt = engine.render_chat([{"role": "user", "content": "hi"}])
    stats = GenStats()
    ours = "".join(engine.stream(prompt, SamplingParams(max_tokens=12, stop_token_ids={-1}), stats))
    ids = engine.tokenizer(prompt, return_tensors="pt").input_ids
    with torch.inference_mode():
        ref = engine.model.generate(ids, max_new_tokens=12, do_sample=False,
                                    eos_token_id=None, pad_token_id=0)
    ref_text = engine.tokenizer.decode(ref[0, ids.shape[1]:], skip_special_tokens=True)
    assert ours == ref_text
    assert stats.completion_tokens == 12 and stats.ttft_s > 0
