#!/usr/bin/env bash
# =============================================================================
# scripts/run_predict.sh
#
# Inference pipeline: ingest → tokenize → predict → postprocess.
# Does not run step03_add_labels (no gold labels needed at inference time).
# Does not run step04_finalize_config (config is already finalised).
#
# Usage:
#   ./scripts/run_predict.sh [--model-name MODEL] [--config configs/biored_base.json]
#                            [--input-dir /path/to/annotations]
#                            [--output-dir /path/to/predictions]
#                            [--batch-size 8] [--num-workers 4]
#                            [--force]
#
# =============================================================================

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-biored_base}"
CONFIG="${CONFIG:-configs/biored_base.json}"
SPEC="${SPEC:-specs/label_spec.json}"
INPUT_DIR="${INPUT_DIR:-data/raw/annotations}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/predictions}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE="${FORCE:-}"

# TODO: parse CLI args from $@

echo "=== Step 1: Ingest ==="
# TODO: python -m pipeline.step01_ingest \
#         --input-dir  "$INPUT_DIR" \
#         --output-dir "$DATA_DIR/step01_output" \
#         $FORCE

echo "=== Step 2: Tokenize ==="
# TODO: python -m pipeline.step02_tokenize \
#         --input-dir  "$DATA_DIR/step01_output" \
#         --output-dir "$DATA_DIR/step02_output" \
#         --config     "$CONFIG" \
#         $FORCE

echo "=== Step 5: Predict ==="
# TODO: python -m pipeline.step05_predict \
#         --input-dir   "$DATA_DIR/step02_output" \
#         --output-dir  "$DATA_DIR/step05_output" \
#         --config      "$CONFIG" \
#         --batch-size  "$BATCH_SIZE" \
#         --num-workers "$NUM_WORKERS" \
#         $FORCE

echo "=== Step 6: Postprocess ==="
# TODO: python -m pipeline.step06_postprocess \
#         --tokens-dir   "$DATA_DIR/step02_output" \
#         --pred-dir     "$DATA_DIR/step05_output" \
#         --fulltext-dir "$DATA_DIR/step01_output" \
#         --output-dir   "$OUTPUT_DIR" \
#         $FORCE

echo "=== Prediction complete. Results in $OUTPUT_DIR ==="
