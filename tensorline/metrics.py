"""Minimal Prometheus metrics (text exposition format 0.0.4), no client library needed.

Counters and histograms are updated only on the scheduler thread; scrapes read them from the
event loop. A scrape may see a value one step stale, which is fine for monitoring.
"""
from __future__ import annotations

import bisect

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


class Histogram:
    def __init__(self, buckets):
        self.buckets = tuple(buckets)
        self.counts = [0] * (len(self.buckets) + 1)   # last = +Inf
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, v)] += 1
        self.sum += v
        self.n += 1

    def render(self, name: str, labels: str) -> list[str]:
        out, cum = [], 0
        sep = "," if labels else ""
        for b, c in zip(self.buckets + (float("inf"),), self.counts):
            cum += c
            le = "+Inf" if b == float("inf") else repr(float(b))
            out.append(f'{name}_bucket{{{labels}{sep}le="{le}"}} {cum}')
        out += [f"{name}_sum{{{labels}}} {self.sum}", f"{name}_count{{{labels}}} {self.n}"]
        return out


class EngineMetrics:
    def __init__(self):
        self.ttft = Histogram(LATENCY_BUCKETS)
        self.e2e = Histogram(LATENCY_BUCKETS)
        self.batch = Histogram(BATCH_BUCKETS)
        self.tokens_generated = 0
        self.requests_finished = {}          # finish_reason -> count


def render(manager) -> str:
    """Exposition for every loaded model of a ModelManager."""
    lines = []

    def meta(name, kind, help_):
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} {kind}")

    engines = [(s.spec.name, s.engine) for s in manager.slots.values()
               if s.engine is not None and hasattr(s.engine, "metrics")]
    gauges = [
        ("tensorline_requests_running", "Sequences in the running batch", lambda e: len(e.running)),
        ("tensorline_requests_waiting", "Sequences queued for admission", lambda e: len(e.waiting)),
        ("tensorline_kv_blocks_total", "KV cache blocks", lambda e: e.alloc.num_blocks),
        ("tensorline_kv_blocks_free", "Unowned KV blocks (incl. evictable cached)", lambda e: e.alloc.num_free),
        ("tensorline_kv_blocks_cached", "KV blocks holding a reusable prefix", lambda e: e.alloc.num_cached),
    ]
    counters = [
        ("tensorline_steps_total", "Engine steps (forward passes)", lambda e: e.steps),
        ("tensorline_preemptions_total", "Sequences preempted for KV memory", lambda e: e.preemptions),
        ("tensorline_generation_tokens_total", "Generated tokens", lambda e: e.metrics.tokens_generated),
        ("tensorline_prompt_tokens_total", "Prompt tokens admitted", lambda e: e.prompt_tokens_total),
        ("tensorline_prefix_cache_hit_tokens_total", "Prompt tokens served from the prefix cache",
         lambda e: e.prefix_hit_tokens),
        ("tensorline_cuda_graph_replays_total", "Decode steps run as CUDA-graph replays",
         lambda e: e.graphs.replays if e.graphs is not None else 0),
    ]
    for kind, rows in (("gauge", gauges), ("counter", counters)):
        for name, help_, fn in rows:
            meta(name, kind, help_)
            for model, e in engines:
                lines.append(f'{name}{{model="{model}"}} {fn(e)}')
    meta("tensorline_requests_finished_total", "counter", "Finished requests by reason")
    for model, e in engines:
        for reason, n in sorted(e.metrics.requests_finished.items()):
            lines.append(f'tensorline_requests_finished_total{{model="{model}",reason="{reason}"}} {n}')
    for name, help_, attr in (("tensorline_time_to_first_token_seconds", "Time to first token", "ttft"),
                              ("tensorline_request_latency_seconds", "End-to-end request latency", "e2e"),
                              ("tensorline_batch_size", "Sequences per engine step", "batch")):
        meta(name, "histogram", help_)
        for model, e in engines:
            lines += getattr(e.metrics, attr).render(name, f'model="{model}"')
    meta("tensorline_model_loaded", "gauge", "1 if the model is resident")
    meta("tensorline_model_memory_bytes", "gauge", "Accounted memory footprint")
    for s in manager.slots.values():
        lines.append(f'tensorline_model_loaded{{model="{s.spec.name}"}} {int(s.engine is not None)}')
        lines.append(f'tensorline_model_memory_bytes{{model="{s.spec.name}"}} {s.footprint}')
    meta("tensorline_model_evictions_total", "counter", "Models unloaded to fit another")
    lines.append(f"tensorline_model_evictions_total {manager.evictions}")
    return "\n".join(lines) + "\n"
