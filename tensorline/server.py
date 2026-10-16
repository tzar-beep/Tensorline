"""OpenAI-compatible HTTP API (/v1/chat/completions, /v1/models, /health)."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import Engine, GenStats, SamplingParams


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Tensorline")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_id, "device": engine.device}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": engine.model_id, "object": "model"}]}

    async def run(req: ChatRequest):
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
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        base = {"id": cid, "created": created, "model": engine.model_id}

        if not req.stream:
            parts, stats = [], None
            async for delta, stats in run(req):
                if delta is not None:
                    parts.append(delta)
            return {
                **base, "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": stats.finish_reason,
                             "message": {"role": "assistant", "content": "".join(parts)}}],
                "usage": usage(stats),
            }

        async def sse():
            chunk = {**base, "object": "chat.completion.chunk"}
            first = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            yield f"data: {json.dumps({**chunk, 'choices': [first]})}\n\n"
            # Content frames are pre-rendered around the delta: one json.dumps of a string per
            # event instead of serialising the whole chunk dict.
            head = json.dumps(chunk)[:-1] + ', "choices": [{"index": 0, "delta": {"content": '
            tail = '}, "finish_reason": null}]}\n\n'
            async for delta, stats in run(req):
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
    ap.add_argument("--no-prefix-caching", action="store_true", help="never reuse KV blocks across requests")
    a = ap.parse_args()
    if a.engine == "naive":
        engine = Engine(a.model, a.device, a.dtype)
    else:
        from .batch_engine import BatchEngine
        engine = BatchEngine(a.model, a.device, a.dtype, num_blocks=a.num_blocks,
                             max_num_seqs=a.max_num_seqs, max_batched_tokens=a.max_batched_tokens,
                             cuda_graphs=not a.no_cuda_graphs,
                             prefix_caching=not a.no_prefix_caching)
    uvicorn.run(create_app(engine), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
