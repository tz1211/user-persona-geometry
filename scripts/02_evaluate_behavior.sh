#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
RESULTS_ROOT="${RESULTS_ROOT:-results/behavioral}"
SEEDS="${SEEDS:-42}"
JUDGE_BACKEND="${JUDGE_BACKEND:-vllm}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"

EXTRA_ARGS=()
if [[ -n "$LIMIT" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [[ "$FORCE" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

for SEED in $SEEDS; do
  python dimensions/evaluate_results.py \
    --results-dir "$RESULTS_ROOT/$MODEL_ID/seed_$SEED" \
    --benchmarks refusal \
    --judge-backend "$JUDGE_BACKEND" \
    --judge-model "$JUDGE_MODEL" \
    --judge-batch-size "$JUDGE_BATCH_SIZE" \
    --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
    "${EXTRA_ARGS[@]}"
done
