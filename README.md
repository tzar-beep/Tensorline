# Tensorline

A from-scratch LLM inference and serving engine — a mini vLLM. Tensorline serves open-weight
models behind an OpenAI-compatible API, and is built up phase by phase toward continuous batching,
a paged KV cache, quantisation and multi-model serving, with every claim backed by a reproducible
benchmark.

> Status: **Phase 1 complete** — single-model streaming server, OpenAI-compatible API, tests, first benchmark.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .[dev]
tensorline --model Qwen/Qwen2.5-0.5B-Instruct          # serves on http://127.0.0.1:8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

Any OpenAI client works: `OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")`.

## Benchmarks

Reference hardware: **NVIDIA RTX 3050 Laptop (6 GB VRAM)**, 16 GB RAM, Windows 11.
Model: Qwen2.5-0.5B-Instruct, fp16, greedy, 128 max output tokens. Harness: [`bench/bench.py`](bench/bench.py)
(streaming clients, works against any OpenAI-compatible server). Raw results live in `bench/results/`.

### Phase 1 — naive serving (one request at a time)

16 requests per level, 128 output tokens each ([raw JSON](bench/results/phase1_naive.json)):

| Concurrency | TTFT p50 | TTFT p99 | Latency p50 | Latency p99 | Throughput (tok/s) | tok/s per request |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 66 ms | 76 ms | 6.72 s | 6.94 s | 18.9 | 18.7 |
| 4 | 13.9 s | 19.5 s | 20.33 s | 20.43 s | 19.6 | 6.0 |
| 8 | 34.0 s | 40.2 s | 40.42 s | 41.78 s | 19.3 | 4.1 |

Single-stream decode speed in-process: **Tensorline loop 19.9 tok/s, Hugging Face `generate`
18.2 tok/s**. The hand-written loop is no slower than the reference.

What this shows:
- Throughput is flat at about 19 tok/s regardless of load, because requests queue behind one lock.
  Waiting time ends up in TTFT, which grows from 66 ms to 34 s at concurrency 8.
- Each decode step (about 50 ms) is bound by launch and Python overhead, not compute: 24 layers of
  small kernels for a 0.5B model on a laptop GPU barely use it. Batching B sequences into one
  forward pass costs nearly the same per step, so Phase 2's continuous batching should raise
  throughput close to linearly with batch size. Phase 2 has to beat this baseline.

## Architecture (Phase 1)

```
client ──HTTP/SSE──▶ FastAPI (asyncio) ──queue──▶ worker thread ──▶ Engine
                                                                 prefill: full prompt → KV cache
                                                                 decode:  1 token/step, reuse KV cache
```

- `tensorline/engine.py`: model loading (CUDA fp16, CPU fp32 fallback) and a hand-written
  prefill/decode loop with explicit KV-cache reuse and greedy/temperature/top-p sampling. It does
  **not** call `model.generate`, so Phase 2 can swap in batched scheduling.
- `tensorline/server.py`: `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/health`.
- `tests/`: API-shape tests, a stream/non-stream equivalence test, and a correctness test showing
  the custom KV-cache decode loop matches Hugging Face `generate` token for token.

## Roadmap

Each phase ships finished and demoable, with benchmark numbers, before the next starts.

1. **Single-model server** ✅: load an open model with a CPU fallback, stream tokens, OpenAI-compatible
   `/v1/chat/completions`.
2. **Continuous batching + KV-cache management**: an iteration-level scheduler (requests join and
   leave the running batch at every decode step), a paged KV cache (fixed-size blocks, a block
   table per sequence, free-list allocator), and preemption when blocks run out. Target: much
   higher throughput than Phase 1 at concurrency 8+, with bounded p99 TTFT.
3. **Quantisation + multi-model**: int8 and 4-bit weights (bitsandbytes first, then a hand-written
   int8 matmul for comparison), loading and unloading several models under a VRAM budget with LRU
   eviction, and quality checks (perplexity delta) for each quantisation mode.
4. **Benchmark harness**: tokens/sec, TTFT and p50/p99 latency against concurrency; comparisons
   with Phase 1 naive serving, Hugging Face `generate`, Ollama, and vLLM (via WSL2/Docker, where
   it runs on this GPU); charts generated from JSON results.
5. **Production polish**: Prometheus `/metrics` (queue depth, batch size, KV-block utilisation,
   TTFT histograms) with a Grafana dashboard, Docker (CUDA base image), CI running the test suite,
   and the final README charts.

## Stack: what and why

| Part | Choice | Why |
|---|---|---|
| Model execution, KV cache, scheduler | **Python + PyTorch** | Model weights, kernels, and the HF ecosystem are all in Python. The scheduler drives the GPU at every decode step, and putting it in another language would add a cross-language hop on the hottest path. vLLM and SGLang make the same choice. GPU work dominates, so Python overhead is small and gets measured. |
| HTTP API | **FastAPI + uvicorn (asyncio)** | Async SSE streaming with many idle connections; it shares a process with the engine, so there is no IPC. |
| Load generator | **Python (httpx, asyncio)** for now | Good enough at the concurrency one 6 GB GPU can serve. If the client becomes the bottleneck in Phase 4, it gets rewritten in **Go** (goroutines, no GIL), and that check is part of the benchmark methodology. |
| Custom kernels (stretch) | **Triton** | Paged-attention and int8 matmul kernels are written in Python syntax and compiled for the GPU, without a C++/CUDA toolchain on Windows. |
| Not Rust/Go for the engine | — | Rust (candle) and Go have no mature GPU tensor stack for transformer serving. Rewriting the model in either language would mean months on kernels instead of the systems problems: scheduling, memory management, and batching. |

## Hardware notes

6 GB of VRAM shapes the design. A 0.5B model in fp16 is about 1 GB of weights, which leaves about
4 GB for KV cache; that headroom is what makes the batching experiments in Phase 2 meaningful.
1–3B models need the Phase 3 quantisation. Without CUDA, everything falls back to CPU fp32.
