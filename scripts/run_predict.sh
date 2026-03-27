#!/usr/bin/env bash
# =============================================================================
# scripts/run_predict.sh
#
# Inference pipeline: ingest → tokenize → predict → postprocess.
# Does not run step03_add_labels (no gold labels needed at inference time).
# Does not run step04_finalize_config (config is already finalised).
#
# Usage:
#   ./scripts/run_predict.sh [--model-name MODEL] [--bert-name HF_ID]
#                            [--config configs/biored_base.json]
#                            [--model-dir models/biored_base]
#                            [--input-dir /path/to/documents]
#                            [--data-dir data] [--output-dir data/predictions]
#                            [--cache-dir cache]
#                            [--batch-size 8] [--num-workers 0]
#                            [--force]
#
# Environment variables (all overridable via CLI flags above):
#   MODEL_NAME   friendly name; used to derive default MODEL_DIR
#   BERT_NAME    HuggingFace model id for subword tokeniser
#   CONFIG       path to finalised model config JSON
#   MODEL_DIR    directory containing model.pt and label2id.json
#   INPUT_DIR    directory of raw annotation / document *.json files
#   DATA_DIR     root for intermediate step outputs
#   OUTPUT_DIR   where step06 PostprocessRecord JSONL files are written
#   CACHE_DIR    HuggingFace model cache directory
#   BATCH_SIZE   inference batch size
#   NUM_WORKERS  DataLoader worker processes (0 = main process only)
#   FORCE        set to --force to reprocess existing outputs
# =============================================================================

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-biored_base}"
BERT_NAME="${BERT_NAME:-dmis-lab/biobert-base-cased-v1.1}"
CONFIG="${CONFIG:-configs/biored_base.json}"
MODEL_DIR="${MODEL_DIR:-models/${MODEL_NAME}}"
INPUT_DIR="${INPUT_DIR:-data/raw/annotations}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/predictions}"
CACHE_DIR="${CACHE_DIR:-cache}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FORCE="${FORCE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)  MODEL_NAME="$2";  shift 2 ;;
        --bert-name)   BERT_NAME="$2";   shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        --model-dir)   MODEL_DIR="$2";   shift 2 ;;
        --input-dir)   INPUT_DIR="$2";   shift 2 ;;
        --data-dir)    DATA_DIR="$2";    shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --cache-dir)   CACHE_DIR="$2";   shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --force)       FORCE="--force";  shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Derived paths
STEP01_DIR="${DATA_DIR}/step01_output"
STEP02_DIR="${DATA_DIR}/step02_output"
STEP05_DIR="${DATA_DIR}/step05_output"

echo "=== Step 1: Ingest ==="
python -m w2ner_biomedical.pipeline.step01_ingest \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$STEP01_DIR" \
    $FORCE

echo "=== Step 2: Tokenize ==="
python -m w2ner_biomedical.pipeline.step02_tokenize \
    --input-dir  "$STEP01_DIR" \
    --output-dir "$STEP02_DIR" \
    --bert-name  "$BERT_NAME" \
    --cache-dir  "$CACHE_DIR" \
    $FORCE

echo "=== Step 5: Predict ==="
python -m w2ner_biomedical.pipeline.step05_predict \
    --input-dir   "$STEP02_DIR" \
    --output-dir  "$STEP05_DIR" \
    --config      "$CONFIG" \
    --model-dir   "$MODEL_DIR" \
    --cache-dir   "$CACHE_DIR" \
    --batch-size  "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    $FORCE

echo "=== Step 6: Postprocess ==="
python -m w2ner_biomedical.pipeline.step06_postprocess \
    --tokens-dir   "$STEP02_DIR" \
    --pred-dir     "$STEP05_DIR" \
    --fulltext-dir "$STEP01_DIR" \
    --output-dir   "$OUTPUT_DIR" \
    $FORCE

echo "=== Prediction complete. Results in $OUTPUT_DIR ==="
