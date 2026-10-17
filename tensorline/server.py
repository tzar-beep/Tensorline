"""OpenAI-compatible HTTP API (/v1/chat/completions, /v1/models, /health) plus model admin
endpoints. Serves one engine, or several models through a ModelManager (the request's `model`
field picks one; models load on demand and are evicted LRU under a memory budget)."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .engine import Engine, GenStats, SamplingParams
from .manager import ModelManager, ModelSpec


class Message(BaseModel):
    role: str
    content: str


class BurstRequest(BaseModel):
    n: int = 16
    max_tokens: int = 128
    model: Optional[str] = None


BURST_TOPICS = ["hash maps", "TCP vs UDP", "Raft consensus", "paged attention", "B-trees",
                "garbage collection", "TLS handshakes", "consistent hashing"]


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


def create_app(target: Engine | ModelManager) -> FastAPI:
    single = not isinstance(target, ModelManager)
    manager = ModelManager.single(target) if single else target
    app = FastAPI(title="Tensorline")

    def resolve(name: str | None) -> str:
        if single or not name:                # one model: any name (e.g. an OpenAI default) maps to it
            return manager.default
        if name not in manager.slots:
            raise HTTPException(404, f"model {name!r} not found; see /v1/models")
        return name

    @app.get("/health")
    def health():
        out = {"status": "ok", "used_gb": round(manager.used_bytes() / 2**30, 3),
               "budget_gb": round(manager.budget / 2**30, 3) if not single else None,
               "evictions": manager.evictions, "models": manager.status()}
        if single:
            e = target
            out.update(model=e.model_id, device=e.device, quant=e.quant or "fp16",
                       weights_gb=round(e.weight_bytes / 2**30, 3))
        return out

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": s["name"], "object": "model", "owned_by": "tensorline", "root": s["model_id"],
             "quant": s["quant"], "loaded": s["loaded"]} for s in manager.status()]}

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))

    @app.get("/api/stats")
    def stats():
        """Live engine state as JSON, for the dashboard (Prometheus scrapes /metrics)."""
        engines = {}
        for name, slot in manager.slots.items():
            e = slot.engine
            if e is None or not hasattr(e, "metrics"):
                continue
            m = e.metrics
            engines[name] = {
                "device": e.device, "quant": e.quant or "fp16", "weights_gb": e.weight_bytes / 2**30,
                "running": len(e.running), "waiting": len(e.waiting),
                "kv_total": e.alloc.num_blocks, "kv_free": e.alloc.num_free, "kv_cached": e.alloc.num_cached,
                "steps": e.steps, "tokens_generated": m.tokens_generated, "preemptions": e.preemptions,
                "prompt_tokens": e.prompt_tokens_total, "prefix_hit_tokens": e.prefix_hit_tokens,
                "graph_replays": e.graphs.replays if e.graphs is not None else 0,
                "finished": m.requests_finished,
                "ttft": {"buckets": list(m.ttft.buckets), "counts": m.ttft.counts},
                "latency": {"buckets": list(m.e2e.buckets), "counts": m.e2e.counts},
            }
        return {"time": time.time(), "multi_model": not single, "used_gb": manager.used_bytes() / 2**30,
                "budget_gb": None if single else manager.budget / 2**30, "evictions": manager.evictions,
                "models": manager.status(), "engines": engines}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        from .metrics import render
        return PlainTextResponse(render(manager), media_type="text/plain; version=0.0.4")

    @app.get("/admin/models")
    def admin_models():
        return manager.status()

    @app.post("/admin/models/{name}/load")
    async def admin_load(name: str):
        await acquire(resolve(name))
        await manager.release(name)
        return manager.status()

    @app.post("/admin/models/{name}/unload")
    async def admin_unload(name: str):
        try:
            return {"unloaded": await manager.unload(resolve(name))}
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/admin/burst")
    async def burst(req: BurstRequest):
        """Load generator inside the server: N concurrent generations, bypassing the browser's
        6-connections-per-site limit so the dashboard can demo real batching."""
        if not 1 <= req.n <= 256:
            raise HTTPException(422, "n must be in [1, 256]")
        name = resolve(req.model)
        engine = await acquire(name)
        if not hasattr(engine, "astream"):
            await manager.release(name)
            raise HTTPException(400, "burst needs the batch engine")

        async def one(i: int):
            prompt = engine.render_chat([{"role": "user", "content":
                                          f"Request {i}: explain {BURST_TOPICS[i % len(BURST_TOPICS)]} in detail."}])
            st, t0, ttft = GenStats(), time.perf_counter(), None
            async for _ in engine.astream(prompt, SamplingParams(req.max_tokens), st):
                ttft = ttft if ttft is not None else time.perf_counter() - t0
            return ttft or 0.0, st.completion_tokens

        try:
            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(i) for i in range(req.n)))
            wall = time.perf_counter() - t0
        finally:
            await manager.release(name)
        ttfts = sorted(r[0] for r in res)
        toks = sum(r[1] for r in res)
        return {"n": req.n, "tokens": toks, "seconds": wall, "tok_s": toks / wall,
                "ttft_p50_ms": 1000 * ttfts[len(ttfts) // 2], "ttft_max_ms": 1000 * ttfts[-1]}

    async def acquire(name: str):
        try:
            return await manager.acquire(name)
        except MemoryError as e:
            raise HTTPException(507, str(e))
        except TimeoutError as e:
            raise HTTPException(503, str(e))

    async def run(engine, req: ChatRequest):
        """Yield (text delta, stats), then (None, stats) when generation ends."""
        prompt = engine.render_chat([m.model_dump() for m in req.messages])
        params = SamplingParams(req.max_tokens, req.temperature, req.top_p)
        stats = GenStats()
        if hasattr(engine, "astream"):        # batch engine: native asyncio, no thread per request
            async for delta in engine.astream(prompt, params, stats):
                yield delta, stats
            yield None, stats
            return
        # naive engine: run its blocking generator in a worker thread, bridge through a queue
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for delta in engine.stream(prompt, params, stats):
                    loop.call_soon_threadsafe(q.put_nowait, delta)
            except Exception as e:  # surface engine errors to the client
                loop.call_soon_threadsafe(q.put_nowait, e)
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while (item := await q.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item, stats
        yield None, stats

    def usage(s: GenStats):
        return {
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.prompt_tokens + s.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": s.cached_tokens},
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        name = resolve(req.model)
        engine = await acquire(name)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        base = {"id": cid, "created": created, "model": name}

        if not req.stream:
            parts, stats = [], None
            try:
                async for delta, stats in run(engine, req):
                    if delta is not None:
                        parts.append(delta)
            finally:
                await manager.release(name)
            return {
                **base, "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": stats.finish_reason,
                             "message": {"role": "assistant", "content": "".join(parts)}}],
                "usage": usage(stats),
            }

        async def sse():
            try:
                async for frame in frames():
                    yield frame
            finally:                           # also runs when the client disconnects
                await manager.release(name)

        async def frames():
            chunk = {**base, "object": "chat.completion.chunk"}
            first = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            yield f"data: {json.dumps({**chunk, 'choices': [first]})}\n\n"
            # Content frames are pre-rendered around the delta: one json.dumps of a string per
            # event instead of serialising the whole chunk dict.
            head = json.dumps(chunk)[:-1] + ', "choices": [{"index": 0, "delta": {"content": '
            tail = '}, "finish_reason": null}]}\n\n'
            async for delta, stats in run(engine, req):
                if delta is None:
                    end = {"index": 0, "delta": {}, "finish_reason": stats.finish_reason}
                    yield f"data: {json.dumps({**chunk, 'choices': [end], 'usage': usage(stats)})}\n\n"
                else:
                    yield "data: " + head + json.dumps(delta) + tail
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser("tensorline")
    ap.add_argument("--model", default=os.getenv("TL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--engine", choices=["batch", "naive"], default="batch",
                    help="batch: continuous batching + paged KV (Phase 2); naive: one request at a time")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--num-blocks", type=int, default=None, help="KV blocks (default: fit 60%% of free VRAM)")
    ap.add_argument("--no-cuda-graphs", action="store_true", help="run decode steps eagerly")
    ap.add_argument("--graph-warmup-tokens", type=int, default=512,
                    help="pre-capture decode CUDA graphs for contexts up to this many tokens (0 = lazy)")
    ap.add_argument("--no-prefix-caching", action="store_true", help="never reuse KV blocks across requests")
    ap.add_argument("--quant", choices=["none", "int8", "int4", "int4-all"], default="none",
                    help="weight-only quantisation (W8A16 / W4A16 with fused Triton kernels); "
                         "int4 keeps v_proj/down_proj in int8, int4-all does not")
    ap.add_argument("--kv-cache-gb", type=float, default=None, help="KV cache size (default: 60%% of free VRAM)")
    ap.add_argument("--serve", action="append", default=[], metavar="NAME=HF_ID[:QUANT]",
                    help="multi-model mode: repeatable, e.g. --serve qwen-3b=Qwen/Qwen2.5-3B-Instruct:int4; "
                         "models load on first request and are evicted LRU under --gpu-budget-gb")
    ap.add_argument("--gpu-budget-gb", type=float, default=None,
                    help="multi-model memory budget (default: 90%% of total VRAM)")
    a = ap.parse_args()
    if a.serve:
        import torch
        budget = a.gpu_budget_gb or 0.9 * torch.cuda.get_device_properties(0).total_memory / 2**30
        specs = [ModelSpec.parse(s, a.kv_cache_gb or 1.0) for s in a.serve]
        uvicorn.run(create_app(ModelManager(specs, int(budget * 2**30))), host=a.host, port=a.port)
        return
    if a.engine == "naive":
        engine = Engine(a.model, a.device, a.dtype, quant=a.quant)
    else:
        from .batch_engine import BatchEngine
        engine = BatchEngine(a.model, a.device, a.dtype, num_blocks=a.num_blocks,
                             max_num_seqs=a.max_num_seqs, max_batched_tokens=a.max_batched_tokens,
                             cuda_graphs=not a.no_cuda_graphs,
                             prefix_caching=not a.no_prefix_caching,
                             quant=a.quant, kv_cache_gb=a.kv_cache_gb,
                             graph_warmup_tokens=a.graph_warmup_tokens)
    uvicorn.run(create_app(engine), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
