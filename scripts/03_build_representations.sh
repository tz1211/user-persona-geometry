#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-dimensions/configs/models/qwen3_4b.yaml}"
OUTPUT_DIR="${VECTORS_DIR:-results/user_attr_vectors}"
BEHAVIORAL_DIR="${BEHAVIORAL_DIR:-results/behavioral}"
LAYER="${LAYER:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.7}"
BUILD_POS="${BUILD_POS:-P4}"
EVAL_POS="${EVAL_POS:-P4}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ -n "$LIMIT" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python -m representation.run_pipeline \
  --model-config "$MODEL_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --behavioral-dir "$BEHAVIORAL_DIR" \
  --layer "$LAYER" \
  --batch-size "$BATCH_SIZE" \
  --split-seed "$SPLIT_SEED" \
  --train-fraction "$TRAIN_FRACTION" \
  --build-pos "$BUILD_POS" \
  --eval-pos "$EVAL_POS" \
  "${EXTRA_ARGS[@]}"

python -m representation.geometry \
  --model-config "$MODEL_CONFIG" \
  --vectors-dir "$OUTPUT_DIR" \
  --layer "$LAYER" \
  --positions P4 \
  --device auto
