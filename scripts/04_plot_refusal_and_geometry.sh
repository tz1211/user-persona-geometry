#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-dimensions/configs/models/qwen3_4b.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
RESULTS_ROOT="${RESULTS_ROOT:-results/behavioral}"
VECTORS_DIR="${VECTORS_DIR:-results/user_attr_vectors}"
FIGS_DIR="${FIGS_DIR:-figs}"
BUILD_POS="${BUILD_POS:-P4}"
EVAL_POS="${EVAL_POS:-P4}"
FORMATS="${FORMATS:-png pdf}"

python -m viz.plot_refusal \
  --results-dir "$RESULTS_ROOT" \
  --model-id "$MODEL_ID" \
  --benchmark refusal \
  --output-dir "$FIGS_DIR/behavioral/$MODEL_ID/refusal" \
  --formats $FORMATS

python -m viz.plot_geometry \
  --model-config "$MODEL_CONFIG" \
  --model-id "$MODEL_ID" \
  --vectors-dir "$VECTORS_DIR" \
  --output-dir "$FIGS_DIR/geometry/P4" \
  --formats $FORMATS

python -m viz.plot_geometry_3d \
  --model-config "$MODEL_CONFIG" \
  --model-id "$MODEL_ID" \
  --vectors-dir "$VECTORS_DIR" \
  --output-dir "$FIGS_DIR/geometry/P4" \
  --position P4 \
  --formats $FORMATS

python -m viz.plot_projection \
  --vectors-dir "$VECTORS_DIR" \
  --model-id "$MODEL_ID" \
  --output-dir "$FIGS_DIR/projection" \
  --build-pos "$BUILD_POS" \
  --eval-pos "$EVAL_POS" \
  --formats $FORMATS \
  --no-plot-title
