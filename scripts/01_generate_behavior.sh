#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-dimensions/configs/models/qwen3_4b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/behavioral}"
SEEDS="${SEEDS:-42}"
LIMIT="${LIMIT:-}"
BATCH_ARGS=()

if [[ -n "$LIMIT" ]]; then
  BATCH_ARGS+=(--limit "$LIMIT")
fi

for SEED in $SEEDS; do
  python dimensions/run_experiment.py \
    --model-config "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --benchmarks refusal \
    "${BATCH_ARGS[@]}"
done
