#!/usr/bin/env bash
# Usage, from the repository root: bash scripts/bench/run_all.sh <model path>
set -euo pipefail
MODEL=${1:?usage: bash scripts/bench/run_all.sh <model path>}
mkdir -p results/bench

run() {
  local tag=$1; shift
  echo "================ $tag ================"
  python scripts/bench/run_bench.py --model "$MODEL" --tag "$tag" "$@" 2>&1 | tee "results/bench/$tag.log"
  grep -iE "KV cache|concurrency|GPU blocks" "results/bench/$tag.log" | tail -n 5 || true
}

# batch size x guided decoding
run base
# prefix caching off
run no_prefix --prefix-caching off --batch-sizes 32 --guided on
# KV cache size at a shorter max_model_len
run len4096 --max-model-len 4096 --kv-only

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/bench/gpu.txt
python scripts/bench/run_bench.py --summarize
