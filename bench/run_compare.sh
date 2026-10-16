#!/usr/bin/env bash
# Back-to-back engine comparison on the same machine state.
# Usage: bash bench/run_compare.sh [model]      RUNS="eager graphs" to pick a subset.
# Prefix-cache workload: RUNS="prefix-off prefix-on"
# Quantisation: TAG=qwen1.5b RUNS="q-none q-int8 q-int4" bash bench/run_compare.sh Qwen/Qwen2.5-1.5B-Instruct
set -e
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
PY=${PY:-.venv/Scripts/python.exe}
RUNS=${RUNS:-"naive eager graphs"}
TAG=${TAG:-qwen0.5b}
mkdir -p bench/results
run() {  # $1=name $2=server args $3=bench args $4..=concurrency levels
  local name=$1 args=$2 bargs=$3; shift 3
  echo "== $name =="
  $PY -m tensorline.server --model "$MODEL" $args > "bench/results/server_$name.log" 2>&1 &
  local pid=$!
  until curl -sf localhost:8000/health > /dev/null; do sleep 2; kill -0 $pid || exit 1; done
  $PY bench/bench.py --concurrency "$@" --requests 16 $bargs --out "bench/results/$name.json"
  kill $pid; wait $pid 2>/dev/null || true
  while curl -sf localhost:8000/health > /dev/null; do sleep 1; done
}
for r in $RUNS; do
  case $r in
    naive)  run naive  "--engine naive" "" 1 4 8 16 ;;
    eager)  run eager  "--engine batch --no-cuda-graphs" "" 1 4 8 16 32 64 ;;
    graphs) run graphs "--engine batch" "" 1 4 8 16 32 64 ;;
    prefix-off) run prefix_off "--engine batch --no-prefix-caching" "--shared-prefix-words 600" 1 4 16 32 ;;
    prefix-on)  run prefix_on  "--engine batch" "--shared-prefix-words 600" 1 4 16 32 ;;
    q-*) q=${r#q-}; run "quant_${TAG}_$q" "--engine batch --quant $q --kv-cache-gb 1" "" 1 16 64 ;;
  esac
done
