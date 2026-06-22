#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-dimensions/configs/models/qwen3_4b.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
VECTORS_DIR="${VECTORS_DIR:-results/user_attr_vectors}"
BEHAVIORAL_DIR="${BEHAVIORAL_DIR:-results/behavioral}"
FIGS_DIR="${FIGS_DIR:-figs}"
LAYER="${LAYER:-20}"
SPLIT_SEED="${SPLIT_SEED:-42}"
N_BINS="${N_BINS:-10}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

for BUILD_POS in P1 P2 P3 P4; do
  for EVAL_POS in P3 P4; do
    python -m representation.projection \
      --model-config "$MODEL_CONFIG" \
      --vectors-dir "$VECTORS_DIR" \
      --behavioral-dir "$BEHAVIORAL_DIR" \
      --layer "$LAYER" \
      --build-pos "$BUILD_POS" \
      --eval-pos "$EVAL_POS" \
      --split-seed "$SPLIT_SEED" \
      --n-bins "$N_BINS" \
      "${EXTRA_ARGS[@]}"

    python -m viz.plot_projection \
      --vectors-dir "$VECTORS_DIR" \
      --model-id "$MODEL_ID" \
      --output-dir "$FIGS_DIR/projection" \
      --build-pos "$BUILD_POS" \
      --eval-pos "$EVAL_POS" \
      --formats png pdf \
      --no-plot-title
  done
done
