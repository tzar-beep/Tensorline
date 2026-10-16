"""WikiText-2 (raw, test split) perplexity for a model under each quantisation mode.

Non-overlapping 1024-token windows over the whole test set, the same windows for every mode,
so deltas between fp16 / int8 / int4 are apples-to-apples.

    python bench/perplexity.py --model Qwen/Qwen2.5-0.5B-Instruct --quant none int8 int4
"""
import argparse
import gc
import json
import math
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from tensorline.engine import Engine


def wikitext2_test() -> str:
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    return "".join(pq.read_table(path).column("text").to_pylist())


@torch.inference_mode()
def perplexity(eng: Engine, ids: torch.Tensor, window: int) -> float:
    nll, count = 0.0, 0
    for start in range(0, ids.numel() - 1, window):
        chunk = ids[start:start + window + 1].to(eng.device)
        if chunk.numel() < 2:
            break
        logits = eng.model(chunk[None, :-1]).logits[0].float()
        nll += torch.nn.functional.cross_entropy(logits, chunk[1:], reduction="sum").item()
        count += chunk.numel() - 1
    return math.exp(nll / count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", nargs="+", default=["none", "int8", "int4", "int4-all"])
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = wikitext2_test()
    rows = []
    for q in a.quant:
        t0 = time.perf_counter()
        eng = Engine(a.model, quant=q)
        ids = eng.tokenizer(text, return_tensors="pt").input_ids[0]
        ppl = perplexity(eng, ids, a.window)
        rows.append({"quant": q, "ppl": ppl, "weights_gb": eng.weight_bytes / 2**30,
                     "tokens": int(ids.numel()), "seconds": time.perf_counter() - t0})
        print(f"{a.model} {q:5s} ppl {ppl:7.3f}  weights {rows[-1]['weights_gb']:.2f} GB", flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    if a.out:
        json.dump({"model": a.model, "window": a.window, "dataset": "wikitext-2-raw-v1/test",
                   "results": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
