#!/usr/bin/env bash
# Tensorline vs Ollama on the same GPU, same model (Qwen2.5-0.5B-Instruct fp16), same client, back to back.
# Ollama is tuned for concurrency: 16 parallel slots (beat 64 in testing), 512-token context per slot, flash attention.
set -e
PY=${PY:-.venv/Scripts/python.exe}
OLLAMA=${OLLAMA:-"$LOCALAPPDATA/Programs/Ollama/ollama.exe"}
LEVELS=${LEVELS:-"1 4 16 64"}
BENCH_ARGS=${BENCH_ARGS:-""}        # e.g. "--prompt-words 800 --max-tokens 512" for the heavy workload
SUFFIX=${SUFFIX:-""}                # result-file suffix, e.g. _heavy
OLLAMA_CTX=${OLLAMA_CTX:-512}       # per-slot context; must fit prompt + output
stop_ollama() {   # llama-server children survive their parent being killed, so kill them explicitly
  taskkill //F //IM "ollama app.exe" > /dev/null 2>&1 || true
  taskkill //F //IM ollama.exe > /dev/null 2>&1 || true
  taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
}
stop_ollama

echo "== tensorline =="
$PY -m tensorline.server --model Qwen/Qwen2.5-0.5B-Instruct ${TL_ARGS:-} > bench/results/server_vs_tensorline.log 2>&1 &
pid=$!
until curl -sf localhost:8000/health > /dev/null; do sleep 2; kill -0 $pid || exit 1; done
$PY bench/bench.py --concurrency $LEVELS --requests 16 $BENCH_ARGS --out bench/results/vs_tensorline$SUFFIX.json
kill $pid; wait $pid 2>/dev/null || true

echo "== ollama =="
OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=$OLLAMA_CTX OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=30m \
  "$OLLAMA" serve > bench/results/server_vs_ollama.log 2>&1 &
until curl -sf localhost:11434/api/version > /dev/null; do sleep 1; done
$PY bench/bench.py --url http://127.0.0.1:11434 --model qwen2.5:0.5b-instruct-fp16 \
  --concurrency $LEVELS --requests 16 $BENCH_ARGS --out bench/results/vs_ollama_p16$SUFFIX.json
stop_ollama
