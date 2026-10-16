"""Multi-model manager: LRU eviction under a budget, busy models are never evicted, single
in-flight load per model, HTTP routing, and (GPU) unloading really frees VRAM."""
import asyncio
import gc

import pytest
import torch
from fastapi.testclient import TestClient

from tensorline.batch_engine import BatchEngine
from tensorline.manager import ModelManager, ModelSpec, estimate_bytes
from tensorline.server import create_app
from tests.test_batching import TINY

KV_GB = 0.001


def cpu_factory(spec):
    return BatchEngine(spec.model_id, device="cpu", num_blocks=64, block_size=4, quant=spec.quant)


def manager(names, fit=2, **kw):
    specs = [ModelSpec(n, TINY, None, KV_GB) for n in names]
    one = estimate_bytes(specs[0])
    return ModelManager(specs, budget_bytes=int(one * (fit + 0.5)), engine_factory=cpu_factory, **kw)


def loaded(m):
    return {n for n, s in m.slots.items() if s.engine is not None}


def test_lru_eviction_and_busy_models_are_never_evicted():
    async def scenario():
        m = manager(["a", "b", "c"])
        await m.load("a")
        await m.load("b")
        assert loaded(m) == {"a", "b"}
        await m.load("c")                      # over budget: evict least recently used ("a")
        assert loaded(m) == {"b", "c"} and m.evictions == 1
        await m.acquire("b")                   # b busy
        await m.load("a")                      # must evict c (idle), not b (busy)
        assert loaded(m) == {"a", "b"}
        await m.release("b")
        return m

    m = asyncio.run(scenario())
    assert m.slots["a"].loads == 2 and m.slots["c"].loads == 1


def test_load_times_out_when_every_model_is_busy():
    async def scenario():
        m = manager(["a", "b"], fit=1, wait_timeout_s=0.3)
        await m.acquire("a")
        with pytest.raises(TimeoutError):
            await m.acquire("b")

    asyncio.run(scenario())


def test_concurrent_requests_trigger_a_single_load():
    async def scenario():
        m = manager(["a"])
        engines = await asyncio.gather(*(m.acquire("a") for _ in range(8)))
        assert len({id(e) for e in engines}) == 1
        assert m.slots["a"].loads == 1 and m.slots["a"].inflight == 8

    asyncio.run(scenario())


def test_model_larger_than_budget_is_rejected():
    async def scenario():
        specs = [ModelSpec("big", TINY, None, kv_cache_gb=8.0)]
        m = ModelManager(specs, budget_bytes=1 << 30, engine_factory=cpu_factory)
        with pytest.raises(MemoryError):
            await m.acquire("big")

    asyncio.run(scenario())


def test_http_routing_and_admin_endpoints():
    m = ModelManager([ModelSpec("small", TINY, None, KV_GB), ModelSpec("small-int8", TINY, "int8", KV_GB)],
                     budget_bytes=1 << 34, engine_factory=cpu_factory)
    c = TestClient(create_app(m))
    assert {d["id"]: d["loaded"] for d in c.get("/v1/models").json()["data"]} == {"small": False, "small-int8": False}
    body = {"model": "small-int8", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}
    r = c.post("/v1/chat/completions", json=body).json()
    assert r["model"] == "small-int8" and r["usage"]["completion_tokens"] > 0
    st = {s["name"]: s for s in c.get("/admin/models").json()}
    assert st["small-int8"]["loaded"] and not st["small"]["loaded"] and st["small-int8"]["inflight"] == 0
    assert c.post("/v1/chat/completions", json={**body, "model": "nope"}).status_code == 404
    assert c.post("/admin/models/small-int8/unload").json() == {"unloaded": True}
    assert not {s["name"]: s for s in c.get("/admin/models").json()}["small-int8"]["loaded"]


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


@gpu
def test_unload_returns_vram_and_estimate_matches_measurement():
    async def scenario():
        specs = [ModelSpec(q, QWEN, q, kv_cache_gb=0.25) for q in ("none", "int8", "int4")]
        m = ModelManager(specs, budget_bytes=5 << 30)
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        for cycle in range(2):
            for s in specs:
                await m.load(s.name)
                est, got = estimate_bytes(s), m.slots[s.name].footprint
                assert abs(got - est) / est < 0.15, (s.name, est, got)
                await m.unload(s.name)
                assert torch.cuda.memory_allocated() - base < (64 << 20), (cycle, s.name)

    asyncio.run(scenario())
