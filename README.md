# Tensorline

A from-scratch LLM inference and serving engine — a mini vLLM. Tensorline serves open-weight
models behind an OpenAI-compatible API, and is built up phase by phase toward continuous batching,
a paged KV cache, quantisation and multi-model serving, with every claim backed by a reproducible
benchmark.

> Status: **Phase 2 complete**: continuous batching over a paged KV cache, with chunked prefill and preemption.

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

### Phase 2 vs Phase 1: continuous batching against naive serving

Both engines were run back to back by [`bench/run_compare.sh`](bench/run_compare.sh) in the same
machine state. Each level sends max(16, 2 × concurrency) requests, each with 128 output tokens
([naive JSON](bench/results/phase1_naive.json), [batch JSON](bench/results/phase2_batch.json)).

| Concurrency | Naive tok/s | **Batched tok/s** | Speed-up | Naive TTFT p50 | **Batched TTFT p50** | Batched TTFT p99 | Batched latency p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 13.2 | **16.0**  | 1.2× | 92 ms   | **72 ms**  | 95 ms   | 8.79 s  |
| 4  | 14.3 | **54.9**  | 3.8× | 20.6 s  | **154 ms** | 219 ms  | 9.05 s  |
| 8  | 13.5 | **88.8**  | 6.6× | 50.3 s  | **188 ms** | 270 ms  | 9.14 s  |
| 16 | 13.3 | **169.6** | 12.8× | 110.1 s | **310 ms** | 482 ms  | 9.44 s  |
| 32 | —    | **321.1** | —    | —       | **481 ms** | 863 ms  | 10.41 s |
| 64 | —    | **558.3** | 39× vs naive peak | — | **874 ms** | 1.47 s | 12.55 s |

What the numbers show:
- **Throughput scales almost linearly with load**, from 16 to 558 tok/s, while per-request speed
  only drops from 16.2 to 10.1 tok/s. One forward pass over 64 sequences costs barely more than
  one over a single sequence.
- **Queueing disappears from TTFT.** Naive serving makes request N wait for N−1 full generations
  (110 s at concurrency 16). With iteration-level scheduling, new requests join on the next step:
  p99 TTFT stays under 1.5 s even at concurrency 64.
- **Single-stream speed also improved** (16.0 vs 13.2 tok/s). The paged decode step uses fused
  SDPA with a grouped-query layout and avoids HF's cache bookkeeping. Measured in-process, a
  decode step takes 64.9 ms for Tensorline vs 74.4 ms for HF.
- **Where the time goes:** a decode step costs roughly the same at batch 1 and batch 64. At
  0.5B parameters the step is bound by CPU dispatch of a few hundred small kernels, not GPU
  FLOPs. That is why batching pays off so well, and why CUDA graphs come next.

> Methodology note: laptop GPU clocks and power state vary between sessions. An earlier
> standalone run measured naive single-stream at 18.9 tok/s. Comparisons are therefore only made
> between runs taken back to back, never across sessions.

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
- API tests: OpenAI response shape, and streamed output identical to non-streamed output.

## Roadmap

Each phase ships finished and demoable, with benchmark numbers, before the next starts.

1. **Single-model server** ✅: load an open model with a CPU fallback, stream tokens, OpenAI-compatible
   `/v1/chat/completions`.
2. **Continuous batching + KV-cache management** ✅: an iteration-level scheduler, a paged KV cache
   (16-token blocks, a block table per sequence, free-list allocator), chunked prefill, and
   recompute preemption.
   - *Next:* CUDA-graph capture of the decode step. Step time is dominated by per-kernel launch
     overhead (see the benchmarks), so replaying a captured graph per batch-size bucket is the
     biggest remaining single-stream win. Prefix caching (sharing blocks across a common system
     prompt) is also planned.
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
