#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-dimensions/configs/models/qwen3_4b.yaml}"
VECTORS_DIR="${VECTORS_DIR:-results/user_attr_vectors}"
OUTPUT_DIR="${STEERING_DIR:-results/steering}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-data/benchmarks/refusal}"
DIMENSIONS="${DIMENSIONS:-knowledge intent emotion belief}"
COEFFICIENTS="${COEFFICIENTS:--2 -1 -0.5 0 0.5 1 2}"
SEEDS="${SEEDS:-0 42 67 69 420}"
LAYER="${LAYER:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SPLIT_SEED="${SPLIT_SEED:-42}"
JUDGE_BACKEND="${JUDGE_BACKEND:-vllm}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

EXTRA_ARGS=()
if [[ -n "$LIMIT" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "$FORCE_EVAL" == "1" ]]; then
  EXTRA_ARGS+=(--force-eval)
fi

for SEED in $SEEDS; do
  for DIMENSION in $DIMENSIONS; do
    python -m representation.steering \
      --model-config "$MODEL_CONFIG" \
      --vectors-dir "$VECTORS_DIR" \
      --benchmark-data-dir "$BENCHMARK_DATA_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --benchmark refusal \
      --dimension "$DIMENSION" \
      --coefficients $COEFFICIENTS \
      --layer "$LAYER" \
      --steering-position P4 \
      --steering-mode response_only \
      --batch-size "$BATCH_SIZE" \
      --record-split heldout \
      --split-seed "$SPLIT_SEED" \
      --seed "$SEED" \
      --judge-backend "$JUDGE_BACKEND" \
      --judge-model "$JUDGE_MODEL" \
      --judge-batch-size "$JUDGE_BATCH_SIZE" \
      --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
      "${EXTRA_ARGS[@]}"
  done
done

python -m viz.plot_steering \
  --results-dir "$OUTPUT_DIR" \
  --benchmark refusal \
  --dimensions knowledge intent emotion belief \
  --formats png pdf
