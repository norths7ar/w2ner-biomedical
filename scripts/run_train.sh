#!/usr/bin/env bash
# =============================================================================
# scripts/run_train.sh
#
# Full training pipeline: ingest → tokenize → add labels →
#   finalize config → train model.
#
# Usage:
#   ./scripts/run_train.sh [--model-name MODEL] [--config configs/biored_base.json]
#                          [--input-dir /path/to/annotations]
#                          [--val-dir /path/to/val_annotations]
#                          [--output-dir /path/to/model/output]
#                          [--force]
#
# Each step writes a manifest alongside its JSONL output.
# If a step fails, subsequent steps are not run (set -e).
# =============================================================================

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-biored_base}"
CONFIG="${CONFIG:-configs/biored_base.json}"
SPEC="${SPEC:-specs/label_spec.json}"
INPUT_DIR="${INPUT_DIR:-data/raw/annotations}"
VAL_DIR="${VAL_DIR:-}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-models/${MODEL_NAME}}"
FORCE="${FORCE:-}"

# TODO: parse --model-name, --config, --input-dir, --val-dir, --output-dir, --force from $@

echo "=== Step 1: Ingest ==="
# TODO: python -m pipeline.step01_ingest \
#         --input-dir  "$INPUT_DIR" \
#         --output-dir "$DATA_DIR/step01_output" \
#         $FORCE

echo "=== Step 2: Tokenize (merged step03+step04) ==="
# TODO: python -m pipeline.step02_tokenize \
#         --input-dir  "$DATA_DIR/step01_output" \
#         --output-dir "$DATA_DIR/step02_output" \
#         --config     "$CONFIG" \
#         $FORCE

echo "=== Step 3: Add Labels ==="
# TODO: python -m pipeline.step03_add_labels \
#         --tokens-dir "$DATA_DIR/step02_output" \
#         --input-dir  "$INPUT_DIR" \
#         --output-dir "$DATA_DIR/step03_output" \
#         --spec       "$SPEC" \
#         --model-name "$MODEL_NAME" \
#         $FORCE

echo "=== Step 4: Finalize Config ==="
# TODO: python -m pipeline.step04_finalize_config \
#         --config       "$CONFIG" \
#         --spec         "$SPEC" \
#         --step03-dir   "$DATA_DIR/step03_output" \
#         --cache-dir    "cache/"

echo "=== Step 5: Train ==="
# TODO: VAL_ARG=""
# TODO: [ -n "$VAL_DIR" ] && VAL_ARG="--val-dir $VAL_DIR"
# TODO: python -m model.train \
#         --config     "$CONFIG" \
#         --spec       "$SPEC" \
#         --input-dir  "$DATA_DIR/step03_output" \
#         $VAL_ARG \
#         --output-dir "$OUTPUT_DIR"

echo "=== Training complete. Model saved to $OUTPUT_DIR ==="
