#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-dimensions/configs/models/qwen3_4b.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
VECTORS_DIR="${VECTORS_DIR:-results/user_attr_vectors}"
OUTPUT_DIR="${CHOICE_PROBE_DIR:-results/trait_choice_probe}"
COEFFICIENTS="${COEFFICIENTS:--2 -1 -0.5 0 0.5 1 2}"
SEEDS="${SEEDS:-0 42 67 69 420}"
LAYER="${LAYER:-20}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python -m representation.trait_choice_probe \
  --model-config "$MODEL_CONFIG" \
  --vectors-dir "$VECTORS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --coefficients $COEFFICIENTS \
  --seeds $SEEDS \
  --layer "$LAYER" \
  --steering-position P4 \
  --prompt-mode completion_sentence \
  --steering-target decode \
  "${EXTRA_ARGS[@]}"

python -m viz.plot_trait_choice_probe \
  --results-dir "$OUTPUT_DIR" \
  --model-id "$MODEL_ID" \
  --coefficients $COEFFICIENTS \
  --formats png pdf
