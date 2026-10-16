"""Phase 1 engine: one model, one request at a time, hand-written decode loop.

The decode loop is written by hand (instead of calling `model.generate`) so the
prefill/decode split and the KV cache are explicit. Phase 2 replaces the
single-request loop with a batched scheduler over a paged KV cache.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0
    stop_token_ids: set[int] = field(default_factory=set)


@dataclass
class GenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0      # prompt tokens served from the prefix cache
    ttft_s: float = 0.0
    total_s: float = 0.0
    finish_reason: str = "length"


def sample(logits: torch.Tensor, p: SamplingParams) -> int:
    if p.temperature <= 0:
        return int(torch.argmax(logits, dim=-1))
    probs = torch.softmax(logits.float() / p.temperature, dim=-1)
    if p.top_p < 1.0:
        sorted_p, idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_p, dim=-1) - sorted_p < p.top_p
        sorted_p = sorted_p * keep
        return int(idx[torch.multinomial(sorted_p / sorted_p.sum(), 1)])
    return int(torch.multinomial(probs, 1))


class Engine:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", quant: str | None = None):
        self.model_id = model_id
        self.device = pick_device(device)
        self.quant = None if quant in (None, "none", "fp16") else quant
        if dtype == "auto":
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load on CPU first: a quantised model is converted layer by layer on the target device,
        # so its fp16 weights never have to fit in VRAM all at once.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        if self.quant:
            from .quant import quantize_model
            quantize_model(self.model, self.quant, self.device, torch_dtype)
        self.model.to(self.device).eval()
        from .quant import weight_bytes
        self.weight_bytes = weight_bytes(self.model)
        # Phase 1 serves strictly one request at a time; this lock is the
        # "naive serving" baseline Phase 2's continuous batching is measured against.
        self._lock = threading.Lock()
        eos = self.tokenizer.eos_token_id
        self.default_stops = {eos} if eos is not None else set()

    def render_chat(self, messages: list[dict]) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"

    @torch.inference_mode()
    def stream(self, prompt: str, params: SamplingParams, stats: GenStats) -> Iterator[str]:
        """Yield decoded text deltas. Fills `stats` as it goes."""
        stops = params.stop_token_ids or self.default_stops
        with self._lock:
            t0 = time.perf_counter()
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            stats.prompt_tokens = ids.shape[1]
            out_ids: list[int] = []
            emitted = ""
            past = None
            inp = ids
            for step in range(params.max_tokens):
                out = self.model(input_ids=inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tok = sample(out.logits[0, -1], params)
                if step == 0:
                    stats.ttft_s = time.perf_counter() - t0
                if tok in stops:
                    stats.finish_reason = "stop"
                    break
                out_ids.append(tok)
                # Decode the full sequence and diff, so multi-byte tokens render correctly.
                text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                if not text.endswith("�"):
                    delta, emitted = text[len(emitted):], text
                    if delta:
                        yield delta
                inp = torch.tensor([[tok]], device=self.device)
            stats.completion_tokens = len(out_ids)
            stats.total_s = time.perf_counter() - t0
