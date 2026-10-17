"""Multi-model serving on one GPU under a memory budget.

Models are declared up front (`ModelSpec`: an API name, a Hugging Face id, a quantisation mode,
a KV-cache size) and loaded on first use. Before a load, the manager estimates the new model's
footprint from its config (exact per-layer bytes for the quantisation mode, plus embeddings and
KV cache). If loaded models plus the newcomer would exceed the budget, it unloads the least
recently used *idle* models (no requests in flight). If everything is busy, the load waits for
a model to go idle, up to a timeout. After a load, the estimate is replaced by the measured
footprint.

Unloading is a real shutdown: the scheduler thread stops, CUDA graphs and the KV cache are
released, and the CUDA caching allocator is emptied, so the memory is actually free for the next model.
"""
from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig

from .quant import GROUP, SENSITIVE

OVERHEAD_BYTES = 300 << 20   # activations, CUDA-graph pool, cuBLAS/Triton workspaces (measured ~0.2 GB)


@dataclass
class ModelSpec:
    name: str
    model_id: str
    quant: str | None = None
    kv_cache_gb: float = 1.0

    @classmethod
    def parse(cls, text: str, kv_cache_gb: float = 1.0) -> "ModelSpec":
        """'name=hf/model-id[:quant]', e.g. 'qwen-3b=Qwen/Qwen2.5-3B-Instruct:int4'."""
        name, _, rest = text.partition("=")
        model_id, _, quant = rest.partition(":")
        if not name or not model_id:
            raise ValueError(f"bad model spec {text!r}; expected name=hf_id[:quant]")
        return cls(name, model_id, quant or None, kv_cache_gb)


def estimate_bytes(spec: ModelSpec) -> int:
    """Footprint from the config alone (no download of weights needed)."""
    c = AutoConfig.from_pretrained(spec.model_id)
    h, inter, layers = c.hidden_size, c.intermediate_size, c.num_hidden_layers
    hd = getattr(c, "head_dim", None) or h // c.num_attention_heads
    kv = c.num_key_value_heads * hd
    shapes = {"q_proj": (h, c.num_attention_heads * hd), "k_proj": (h, kv), "v_proj": (h, kv),
              "o_proj": (c.num_attention_heads * hd, h), "gate_proj": (h, inter),
              "up_proj": (h, inter), "down_proj": (inter, h)}

    def bytes_per_weight(name: str) -> float:
        q = spec.quant
        if q in (None, "none", "fp16"):
            return 2.0
        if q == "int8" or (q == "int4" and name in SENSITIVE):
            return 1.0
        return 0.5 + 4.0 / GROUP                       # packed nibbles + fp16 scale & min per group

    per_layer = sum(k * n * bytes_per_weight(nm) for nm, (k, n) in shapes.items())
    embed = c.vocab_size * h * 2 * (1 if c.tie_word_embeddings else 2)
    return int(layers * per_layer + embed + spec.kv_cache_gb * (1 << 30) + OVERHEAD_BYTES)


@dataclass
class _Slot:
    spec: ModelSpec
    engine: object | None = None
    footprint: int = 0
    inflight: int = 0
    last_used: float = 0.0
    loads: int = 0
    loading: asyncio.Future | None = field(default=None, repr=False)


class ModelManager:
    def __init__(self, specs: list[ModelSpec], budget_bytes: int, engine_factory=None,
                 wait_timeout_s: float = 60.0):
        if not specs:
            raise ValueError("no models configured")
        self.slots = {s.name: _Slot(s) for s in specs}
        self.default = specs[0].name
        self.budget = budget_bytes
        self.wait_timeout_s = wait_timeout_s
        self._factory = engine_factory or self._default_factory
        self._lock = asyncio.Lock()             # serialises loads/unloads
        self._idle = asyncio.Condition()        # signalled when a request releases a model
        self.evictions = 0

    @classmethod
    def single(cls, engine, name: str | None = None) -> "ModelManager":
        """Wrap one already-loaded engine (the single-model server and the tests)."""
        spec = ModelSpec(name or engine.model_id, engine.model_id, getattr(engine, "quant", None))
        m = cls([spec], budget_bytes=1 << 62, engine_factory=lambda s: engine)
        slot = m.slots[spec.name]
        slot.engine, slot.last_used, slot.loads = engine, time.monotonic(), 1
        kv = getattr(engine, "paged", None)
        kv_bytes = sum(t.numel() * t.element_size() for t in kv.k_cache + kv.v_cache) if kv else 0
        slot.footprint = getattr(engine, "weight_bytes", 0) + kv_bytes
        return m

    @staticmethod
    def _default_factory(spec: ModelSpec):
        from .batch_engine import BatchEngine
        return BatchEngine(spec.model_id, quant=spec.quant, kv_cache_gb=spec.kv_cache_gb)

    # ---- accounting ----
    def used_bytes(self) -> int:
        return sum(s.footprint for s in self.slots.values() if s.engine is not None)

    def status(self) -> list[dict]:
        return [{"name": n, "model_id": s.spec.model_id, "quant": s.spec.quant or "fp16",
                 "loaded": s.engine is not None, "inflight": s.inflight, "loads": s.loads,
                 "footprint_gb": round(s.footprint / 2**30, 3)} for n, s in self.slots.items()]

    # ---- request lifecycle ----
    async def acquire(self, name: str | None):
        name = name or self.default
        if name not in self.slots:
            raise KeyError(name)
        slot = self.slots[name]
        while slot.engine is None:
            if slot.loading is None:
                slot.loading = asyncio.get_running_loop().create_future()
                try:
                    await self._load(slot)
                    slot.loading.set_result(True)
                except BaseException as e:
                    slot.loading.set_exception(e)
                    slot.loading.exception()     # mark retrieved; waiters re-raise below
                    raise
                finally:
                    slot.loading = None
            else:
                await asyncio.shield(slot.loading)
        slot.inflight += 1
        slot.last_used = time.monotonic()
        return slot.engine

    async def release(self, name: str | None) -> None:
        slot = self.slots[name or self.default]
        slot.inflight -= 1
        slot.last_used = time.monotonic()
        async with self._idle:
            self._idle.notify_all()

    # ---- loading / eviction ----
    async def _load(self, slot: _Slot) -> None:
        async with self._lock:
            need = await asyncio.to_thread(estimate_bytes, slot.spec)
            if need > self.budget:
                raise MemoryError(f"{slot.spec.name} needs ~{need / 2**30:.2f} GB; budget is "
                                  f"{self.budget / 2**30:.2f} GB")
            deadline = time.monotonic() + self.wait_timeout_s
            while self.used_bytes() + need > self.budget:
                idle = [s for s in self.slots.values() if s.engine is not None and s.inflight == 0]
                if idle:
                    victim = min(idle, key=lambda s: s.last_used)
                    await asyncio.to_thread(self._unload_slot, victim)
                    self.evictions += 1
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no idle model to evict for {slot.spec.name}")
                async with self._idle:
                    try:
                        await asyncio.wait_for(self._idle.wait(), remaining)
                    except asyncio.TimeoutError:
                        pass
            before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            engine = await asyncio.to_thread(self._factory, slot.spec)
            after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            measured = after - before
            slot.engine, slot.loads = engine, slot.loads + 1
            # Measured weights + KV (+ overhead headroom) when on GPU; the estimate otherwise.
            slot.footprint = measured + OVERHEAD_BYTES if measured > 0 else need
            slot.last_used = time.monotonic()

    def _unload_slot(self, slot: _Slot) -> None:
        engine, slot.engine, slot.footprint = slot.engine, None, 0
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def unload(self, name: str) -> bool:
        async with self._lock:
            slot = self.slots[name]
            if slot.engine is None:
                return False
            if slot.inflight:
                raise RuntimeError(f"{name} has {slot.inflight} requests in flight")
            await asyncio.to_thread(self._unload_slot, slot)
            return True

    async def load(self, name: str) -> None:
        await self.acquire(name)
        await self.release(name)
