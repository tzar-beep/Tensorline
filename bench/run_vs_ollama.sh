#!/usr/bin/env bash
# Tensorline vs Ollama on the same GPU, same model (Qwen2.5-0.5B-Instruct fp16), same client, back to back.
# Ollama is tuned for concurrency: 16 parallel slots (beat 64 in testing), 512-token context per slot, flash attention.
set -e
PY=${PY:-.venv/Scripts/python.exe}
OLLAMA=${OLLAMA:-"$LOCALAPPDATA/Programs/Ollama/ollama.exe"}
LEVELS="1 4 16 64"
taskkill //F //IM "ollama app.exe" > /dev/null 2>&1 || true
taskkill //F //IM ollama.exe > /dev/null 2>&1 || true

echo "== tensorline =="
$PY -m tensorline.server --model Qwen/Qwen2.5-0.5B-Instruct > bench/results/server_vs_tensorline.log 2>&1 &
pid=$!
until curl -sf localhost:8000/health > /dev/null; do sleep 2; kill -0 $pid || exit 1; done
$PY bench/bench.py --concurrency $LEVELS --requests 16 --out bench/results/vs_tensorline.json
kill $pid; wait $pid 2>/dev/null || true

echo "== ollama =="
OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=512 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=30m \
  "$OLLAMA" serve > bench/results/server_vs_ollama.log 2>&1 &
until curl -sf localhost:11434/api/version > /dev/null; do sleep 1; done
$PY bench/bench.py --url http://127.0.0.1:11434 --model qwen2.5:0.5b-instruct-fp16 \
  --concurrency $LEVELS --requests 16 --out bench/results/vs_ollama_p16.json
"$PY" -c "import subprocess; subprocess.run(['taskkill','/F','/IM','ollama.exe'], capture_output=True)"
