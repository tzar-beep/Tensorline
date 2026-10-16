"""Load generator: N concurrent streaming clients against any OpenAI-compatible endpoint.

Reports TTFT, end-to-end latency (p50/p99) and aggregate output tokens/sec.
Works unchanged against Tensorline, Ollama (/v1) and vLLM, so comparisons are apples-to-apples.

    python bench/bench.py --url http://127.0.0.1:8000 --concurrency 1 4 8 --requests 16
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPTS = [
    "Explain how a hash map works in three sentences.",
    "Write a haiku about distributed systems.",
    "What is the difference between TCP and UDP?",
    "Summarise the idea of attention in transformers.",
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def one(client, url, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft, ntok = None, 0
    payload = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
               "messages": [{"role": "user", "content": prompt}]}
    async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                ntok = c["usage"]["completion_tokens"]
            if c["choices"] and c["choices"][0]["delta"].get("content"):
                ttft = ttft or time.perf_counter() - t0
    return ttft or 0.0, time.perf_counter() - t0, ntok


async def run_level(url, model, conc, n, max_tokens):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=None) as client:
        async def task(i):
            async with sem:
                return await one(client, url, model, PROMPTS[i % len(PROMPTS)], max_tokens)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(task(i) for i in range(n)))
        wall = time.perf_counter() - t0
    ttft, lat, toks = zip(*res)
    return {
        "concurrency": conc, "requests": n,
        "ttft_p50_ms": 1000 * pct(ttft, 50), "ttft_p99_ms": 1000 * pct(ttft, 99),
        "lat_p50_s": pct(lat, 50), "lat_p99_s": pct(lat, 99),
        "throughput_tok_s": sum(toks) / wall,
        "per_req_tok_s": statistics.mean(t / l for t, l in zip(toks, lat) if l > 0),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None, help="write results as JSON")
    a = ap.parse_args()
    async with httpx.AsyncClient() as c:
        model = a.model or (await c.get(f"{a.url}/v1/models")).json()["data"][0]["id"]
    await run_level(a.url, model, 1, 2, 16)  # warm-up (CUDA kernels, allocator)
    # at least 2 requests per client so every level reaches steady state
    rows = [await run_level(a.url, model, k, max(a.requests, 2 * k), a.max_tokens) for k in a.concurrency]
    print(f"{'conc':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'lat p50':>8} {'lat p99':>8} {'tok/s':>7} {'tok/s/req':>9}")
    for r in rows:
        print(f"{r['concurrency']:>4} {r['ttft_p50_ms']:>7.0f}ms {r['ttft_p99_ms']:>7.0f}ms "
              f"{r['lat_p50_s']:>7.2f}s {r['lat_p99_s']:>7.2f}s {r['throughput_tok_s']:>7.1f} {r['per_req_tok_s']:>9.1f}")
    if a.out:
        json.dump({"model": model, "url": a.url, "max_tokens": a.max_tokens, "results": rows},
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    asyncio.run(main())
