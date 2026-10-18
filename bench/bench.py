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


TOPICS = ["distributed consensus", "memory allocators", "query planners", "TCP congestion control",
          "garbage collectors", "B-tree storage engines", "GPU schedulers", "compilers", "CRDTs",
          "service meshes", "vector databases", "kernel bypass networking"]


def long_prompt(i: int, words: int) -> str:
    """A unique ~`words`-word document per request. The request id comes FIRST, so nothing
    beyond the chat template can be served from a prefix cache: every prompt token is real
    prefill work."""
    import random
    rnd = random.Random(i)
    vocab = ("system latency throughput replica shard index cache commit log node leader follower "
             "quorum lease epoch checkpoint snapshot partition buffer queue batch kernel thread lock "
             "page block tensor gradient stream packet socket route schema query plan cost model").split()
    body = " ".join(rnd.choice(vocab) for _ in range(words))
    return (f"Ticket #{i}-{rnd.randrange(10**6)}. Notes on {TOPICS[i % len(TOPICS)]}: {body}\n\n"
            f"Using the notes above, write a detailed technical explanation of {TOPICS[i % len(TOPICS)]}.")


def shared_prefix(words: int) -> str:
    """A deterministic ~`words`-word system prompt (e.g. a long policy/RAG preamble)."""
    clause = "Rule {k}: answer precisely, cite section {k}, and never invent facts."
    return " ".join(clause.format(k=k) for k in range(max(1, words // 11)))


async def one(client, url, model, prompt, max_tokens, system=None):
    t0 = time.perf_counter()
    ttft, ntok, cached, ptok = None, 0, 0, 0
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
               "stream_options": {"include_usage": True},   # OpenAI: usage in the final chunk
               "messages": msgs}
    async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                ntok = c["usage"]["completion_tokens"]
                ptok = c["usage"]["prompt_tokens"]
                cached = (c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            if c["choices"] and c["choices"][0]["delta"].get("content"):
                ttft = ttft or time.perf_counter() - t0
    return ttft or 0.0, time.perf_counter() - t0, ntok, cached, ptok


async def run_level(url, model, conc, n, max_tokens, system=None, prompt_words=0):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=None) as client:
        async def task(i):
            async with sem:
                if prompt_words:
                    p = long_prompt(i, prompt_words)
                else:
                    p = PROMPTS[i % len(PROMPTS)]
                    p = f"[{i}] {p}" if system else p
                return await one(client, url, model, p, max_tokens, system)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(task(i) for i in range(n)))
        wall = time.perf_counter() - t0
    ttft, lat, toks, cached, ptoks = zip(*res)
    return {
        "concurrency": conc, "requests": n,
        "ttft_p50_ms": 1000 * pct(ttft, 50), "ttft_p99_ms": 1000 * pct(ttft, 99),
        "lat_p50_s": pct(lat, 50), "lat_p99_s": pct(lat, 99),
        "throughput_tok_s": sum(toks) / wall,
        "per_req_tok_s": statistics.mean(t / l for t, l in zip(toks, lat) if l > 0),
        "prompt_tokens_mean": statistics.mean(ptoks),
        "cached_prompt_frac": sum(cached) / max(1, sum(ptoks)),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--prompt-words", type=int, default=0,
                    help="heavy workload: a unique ~N-word document per request (no prefix-cache help)")
    ap.add_argument("--shared-prefix-words", type=int, default=0,
                    help="prepend the same ~N-word system prompt to every request (prefix-cache workload)")
    a = ap.parse_args()
    async with httpx.AsyncClient() as c:
        model = a.model or (await c.get(f"{a.url}/v1/models")).json()["data"][0]["id"]
    await run_level(a.url, model, 1, 2, 16)  # warm-up (CUDA kernels, allocator)
    # at least 2 requests per client so every level reaches steady state
    system = shared_prefix(a.shared_prefix_words) if a.shared_prefix_words else None
    rows = [await run_level(a.url, model, k, max(a.requests, 2 * k), a.max_tokens, system, a.prompt_words)
            for k in a.concurrency]
    print(f"{'conc':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'lat p50':>8} {'lat p99':>8} {'tok/s':>7} {'tok/s/req':>9} {'cached':>7}")
    for r in rows:
        print(f"{r['concurrency']:>4} {r['ttft_p50_ms']:>7.0f}ms {r['ttft_p99_ms']:>7.0f}ms "
              f"{r['lat_p50_s']:>7.2f}s {r['lat_p99_s']:>7.2f}s {r['throughput_tok_s']:>7.1f} {r['per_req_tok_s']:>9.1f} "
              f"{100 * r['cached_prompt_frac']:>6.0f}%")
    if a.out:
        json.dump({"model": model, "url": a.url, "max_tokens": a.max_tokens,
                   "shared_prefix_words": a.shared_prefix_words, "results": rows},
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    asyncio.run(main())
