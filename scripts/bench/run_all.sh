#!/usr/bin/env bash
# Run all throughput experiments on the GPU machine, from the repository root:
#   bash scripts/bench/run_all.sh /model/ModelScope/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
# Takes roughly 10-20 minutes on one RTX 4090. Results: results/bench/summary.md
set -euo pipefail
MODEL=${1:?usage: bash scripts/bench/run_all.sh <model path>}
mkdir -p results/bench

run() {
  local tag=$1; shift
  echo "================ $tag ================"
  python scripts/bench/run_bench.py --model "$MODEL" --tag "$tag" "$@" 2>&1 | tee "results/bench/$tag.log"
  echo "---- KV cache lines from the vLLM log ($tag) ----"
  grep -iE "KV cache|concurrency|GPU blocks" "results/bench/$tag.log" | tail -n 5 || true
}

# 1 + 2: batch size sweep x guided decoding on/off (prefix caching on, the vLLM default)
run base
# 3: prefix caching off, otherwise the same as the main setting
run no_prefix --prefix-caching off --batch-sizes 32 --guided on
# 4: KV cache capacity at a shorter max_model_len (engine load only, no throughput run)
run len4096 --max-model-len 4096 --kv-only

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/bench/gpu.txt
python scripts/bench/run_bench.py --summarize
