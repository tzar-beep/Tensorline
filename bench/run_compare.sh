#!/usr/bin/env bash
# Back-to-back naive vs continuous-batching comparison on the same machine state.
# Usage: bash bench/run_compare.sh [model]
set -e
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
PY=${PY:-.venv/Scripts/python.exe}
mkdir -p bench/results
run() {  # $1=engine $2=out $3..=concurrency levels
  local eng=$1 out=$2; shift 2
  $PY -m tensorline.server --model "$MODEL" --engine "$eng" > "bench/results/server_$eng.log" 2>&1 &
  local pid=$!
  until curl -sf localhost:8000/health > /dev/null; do sleep 2; kill -0 $pid || exit 1; done
  $PY bench/bench.py --concurrency "$@" --requests 16 --out "$out"
  kill $pid; wait $pid 2>/dev/null || true
}
echo "== naive (Phase 1) =="; run naive bench/results/phase1_naive.json 1 4 8 16
echo "== batch (Phase 2) =="; run batch bench/results/phase2_batch.json 1 4 8 16 32 64
