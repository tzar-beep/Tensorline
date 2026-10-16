# Tensorline

A from-scratch LLM inference and serving engine — a mini vLLM. Tensorline serves open-weight
models behind an OpenAI-compatible API, and is built up phase by phase toward continuous batching,
a paged KV cache, quantisation and multi-model serving, with every claim backed by a reproducible
benchmark.

> Status: **Phase 2 complete**: continuous batching over a paged KV cache, chunked prefill, preemption, and CUDA-graph decode.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .[dev]
tensorline --model Qwen/Qwen2.5-0.5B-Instruct          # serves on http://127.0.0.1:8000
tensorline --engine naive                               # Phase 1 one-at-a-time baseline
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

### Naive vs continuous batching vs batching + CUDA graphs

All three engines were run back to back by [`bench/run_compare.sh`](bench/run_compare.sh) in the
same machine state. Each level sends max(16, 2 × concurrency) streaming requests with 128 output
tokens each ([naive](bench/results/naive.json), [eager](bench/results/eager.json),
[graphs](bench/results/graphs.json)).

**Throughput (output tok/s)**

| Concurrency | Naive (Phase 1) | Batched, eager | **Batched + CUDA graphs** | vs naive |
|---:|---:|---:|---:|---:|
| 1  | 13.7 | 18.4  | **80.5**   | 5.9× |
| 4  | 14.8 | 52.9  | **211.8**  | 14× |
| 8  | 15.0 | 90.5  | **335.7**  | 22× |
| 16 | 15.8 | 174.1 | **558.1**  | 35× |
| 32 | —    | 317.6 | **866.4**  | — |
| 64 | —    | 557.5 | **1258.8** | 80× vs naive peak |

**Latency (graphs engine; naive in brackets)**

| Concurrency | TTFT p50 | TTFT p99 | Request latency p50 | Request latency p99 |
|---:|---:|---:|---:|---:|
| 1  | 77 ms (89 ms)   | 94 ms   | 1.51 s (9.16 s)  | 2.08 s |
| 8  | 132 ms (43.9 s) | 221 ms  | 2.20 s (54.1 s)  | 2.92 s |
| 16 | 299 ms (92.3 s) | 378 ms  | 2.72 s (100.3 s) | 3.61 s |
| 64 | 770 ms          | 1.36 s  | 4.77 s           | 6.56 s |

What the numbers show:
- **Continuous batching** makes throughput scale with load (18 → 558 tok/s eager), where naive
  stays flat at about 15 tok/s. It also removes queueing from TTFT: naive request N waits for
  N−1 whole generations (92 s at concurrency 16), while batched requests join on the next step.
- **CUDA graphs** fix what batching cannot: per-step overhead. Measured in-process, a decode
  step drops from **69 → 11 ms at batch 1 (6.3×)** and from **64 → 20 ms at batch 64 (3.2×)**.
  At 0.5B parameters, the eager step is a few hundred small kernels, each dispatched from
  Python. Replaying the captured graph is a single launch.
- **Where the time goes now:** the in-process decode ceiling at batch 64 is about 3,200 tok/s
  (64 tokens / 20 ms). End to end, the server reaches 1,259 tok/s, about 40% of that. The rest
  is host-side work: the Python scheduler, incremental detokenisation and SSE streaming. That is
  the next bottleneck to attack.

> Methodology note: laptop GPU clocks and power state vary between sessions (an earlier standalone
> run measured naive single-stream at 18.9 tok/s). Comparisons are therefore only made between
> runs taken back to back, never across sessions.

## Architecture

```
client ──HTTP/SSE──▶ FastAPI (asyncio) ──▶ request thread ──submit──▶ waiting queue
                                                ▲                         │
                                       token ids│                         ▼
                              ┌──────────── scheduler thread (owns the GPU) ─────────────┐
                              │ 1. running seqs: next token (decode) or next prompt chunk │
                              │    out of KV blocks? preempt newest seq, recompute later  │
                              │ 2. admit waiting seqs FCFS under token budget + free blocks│
                              │ 3. ONE forward pass over all scheduled tokens (flattened) │
                              │ 4. sample, stream tokens, retire finished seqs, free blocks│
                              └───────────────────────────────────────────────────────────┘
```

- [`tensorline/paged.py`](tensorline/paged.py): the paged KV cache and the forward pass. Each layer's
  K/V lives in a block pool `[num_blocks, 16, kv_heads, head_dim]`, and each sequence has a block
  table, so memory grows 16 tokens at a time and is freed when the sequence finishes. A step
  flattens every scheduled token from every sequence into one `[T, hidden]` matrix, so the
  QKV/O/MLP matmuls are shared. Only attention is per-sequence: decode tokens go through one
  batched SDPA call over K/V gathered via block tables, and prefill chunks run causal attention
  against their cached context. GQA is handled by folding a KV head's query group into SDPA's
  query axis, so K/V are never copied. The HF model supplies only the weights.
- [`tensorline/batch_engine.py`](tensorline/batch_engine.py): an iteration-level scheduler (Orca/vLLM
  style). Requests join and leave the batch on every step. Prompts longer than the token budget
  are **chunked** across steps, so a long prefill never stalls everyone's decode. When blocks run
  out, the newest sequence is **preempted**: its blocks are freed and it is recomputed later, and
  its stream continues seamlessly.
- [`tensorline/cuda_graphs.py`](tensorline/cuda_graphs.py): CUDA-graph capture and replay of
  pure-decode steps. Batches are padded to power-of-two buckets for batch size and block-table
  width, and each bucket's graph is captured lazily. All graphs share one memory pool and one
  logits buffer. Inputs go in with a single host-to-device copy, and padding rows write only to
  a reserved scratch KV block. Steps containing prefill run eagerly.
- [`tensorline/engine.py`](tensorline/engine.py): model loading and the Phase 1 one-at-a-time loop,
  kept as `--engine naive` as the baseline.
- [`tensorline/server.py`](tensorline/server.py): `/v1/chat/completions` (streaming and
  non-streaming), `/v1/models`, `/health`.

### Testing

Correctness is checked by **differential testing** against reference implementations
([`tests/test_batching.py`](tests/test_batching.py)):
- paged forward vs Hugging Face logits over prefill + decode with deliberately non-contiguous
  blocks, for both MHA (Llama) and GQA 14:2 with QKV bias (Qwen2);
- 16 concurrent requests through the batching engine are token-identical to the naive engine
  in four regimes: plain, forced chunked prefill (4-token budget), forced preemption (12 KV
  blocks), and a 2-sequence cap. Each run also asserts that no KV blocks leak;
- a request too large for the whole KV cache fails cleanly instead of hanging the scheduler;
- CUDA-graph decode vs eager decode, run for 20 steps across a block boundary with a padded
  batch (logits and argmax match). A bystander KV block is checked to stay untouched by padding
  rows, and the engine generates identical text with graphs on and off
  ([`tests/test_cuda_graphs.py`](tests/test_cuda_graphs.py));
- API tests: OpenAI response shape, and streamed output identical to non-streamed output.

## Roadmap

Each phase ships finished and demoable, with benchmark numbers, before the next starts.

1. **Single-model server** ✅: load an open model with a CPU fallback, stream tokens, OpenAI-compatible
   `/v1/chat/completions`.
2. **Continuous batching + KV-cache management** ✅: an iteration-level scheduler, a paged KV cache
   (16-token blocks, a block table per sequence, free-list allocator), chunked prefill, and
   recompute preemption, and CUDA-graph decode (5.9× single-stream, 80× peak throughput vs naive).
   - *Next:* automatic prefix caching (hash-addressed KV blocks shared across requests), then
     cutting host-side overhead (detokenisation and streaming off the scheduler's critical path).
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
| Model execution, KV cache, scheduler | **Python + PyTorch** | Model weights, kernels, and the HF ecosystem are all in Python. The scheduler drives the GPU at every decode step, and putting it in another language would add a cross-language hop on the hottest path. vLLM and SGLang make the same choice. Python/dispatch overhead is real at this model size (measured below); it is amortised by batching and next attacked with CUDA graphs rather than by changing language. |
| HTTP API | **FastAPI + uvicorn (asyncio)** | Async SSE streaming with many idle connections; it shares a process with the engine, so there is no IPC. |
| Load generator | **Python (httpx, asyncio)** for now | Good enough at the concurrency one 6 GB GPU can serve. If the client becomes the bottleneck in Phase 4, it gets rewritten in **Go** (goroutines, no GIL), and that check is part of the benchmark methodology. |
| Custom kernels (stretch) | **Triton** | Paged-attention and int8 matmul kernels are written in Python syntax and compiled for the GPU, without a C++/CUDA toolchain on Windows. |
| Not Rust/Go for the engine | — | Rust (candle) and Go have no mature GPU tensor stack for transformer serving. Rewriting the model in either language would mean months on kernels instead of the systems problems: scheduling, memory management, and batching. |

## Hardware notes

6 GB of VRAM shapes the design. A 0.5B model in fp16 is about 1 GB of weights, which leaves about
4 GB for KV cache; that headroom is what makes the batching experiments in Phase 2 meaningful.
1–3B models need the Phase 3 quantisation. Without CUDA, everything falls back to CPU fp32.
