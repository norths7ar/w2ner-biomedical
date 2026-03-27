#!/usr/bin/env bash
# =============================================================================
# scripts/run_train.sh
#
# Full training pipeline: ingest → tokenize → add labels →
#   finalize config → train model.
#
# Usage:
#   ./scripts/run_train.sh [--model-name MODEL] [--bert-name HF_ID]
#                          [--config configs/biored_base.json]
#                          [--spec specs/label_spec.json]
#                          [--model-suffix _biored]
#                          [--input-dir /path/to/annotations]
#                          [--val-dir /path/to/val_annotations]
#                          [--data-dir data] [--output-dir models/biored_base]
#                          [--cache-dir cache] [--workers 1]
#                          [--force]
#
# Environment variables (all overridable via CLI flags above):
#   MODEL_NAME   friendly name used to derive default output paths
#   BERT_NAME    HuggingFace model id for subword tokeniser
#   CONFIG       path to model config JSON template
#   SPEC         path to label_spec.json
#   MODEL_SUFFIX model_filters key in label_spec (e.g. _biored, _bc5cdr)
#   INPUT_DIR    directory of raw annotation *.json files
#   VAL_DIR      optional validation annotation directory
#   DATA_DIR     root for intermediate step outputs
#   OUTPUT_DIR   where model.pt, label2id.json and logs are written
#   CACHE_DIR    HuggingFace model cache directory
#   WORKERS      worker processes for step02 tokenization
#   FORCE        set to --force to reprocess existing outputs
#
# Each step writes a .meta.json manifest alongside its JSONL output.
# If a step fails, subsequent steps are not run (set -e).
# =============================================================================

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-biored_base}"
BERT_NAME="${BERT_NAME:-dmis-lab/biobert-base-cased-v1.1}"
CONFIG="${CONFIG:-configs/biored_base.json}"
SPEC="${SPEC:-specs/label_spec.json}"
MODEL_SUFFIX="${MODEL_SUFFIX:-}"
INPUT_DIR="${INPUT_DIR:-data/raw/annotations}"
VAL_DIR="${VAL_DIR:-}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-models/${MODEL_NAME}}"
CACHE_DIR="${CACHE_DIR:-cache}"
WORKERS="${WORKERS:-1}"
FORCE="${FORCE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)   MODEL_NAME="$2";   shift 2 ;;
        --bert-name)    BERT_NAME="$2";    shift 2 ;;
        --config)       CONFIG="$2";       shift 2 ;;
        --spec)         SPEC="$2";         shift 2 ;;
        --model-suffix) MODEL_SUFFIX="$2"; shift 2 ;;
        --input-dir)    INPUT_DIR="$2";    shift 2 ;;
        --val-dir)      VAL_DIR="$2";      shift 2 ;;
        --data-dir)     DATA_DIR="$2";     shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --cache-dir)    CACHE_DIR="$2";    shift 2 ;;
        --workers)      WORKERS="$2";      shift 2 ;;
        --force)        FORCE="--force";   shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Derived paths
STEP01_DIR="${DATA_DIR}/step01_output"
STEP02_DIR="${DATA_DIR}/step02_output"
STEP03_DIR="${DATA_DIR}/step03_output"

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
    --workers    "$WORKERS" \
    $FORCE

echo "=== Step 3: Add Labels ==="
SUFFIX_ARG=""
[ -n "$MODEL_SUFFIX" ] && SUFFIX_ARG="--model-suffix $MODEL_SUFFIX"
python -m w2ner_biomedical.pipeline.step03_add_labels \
    --input-dir  "$INPUT_DIR" \
    --tokens-dir "$STEP02_DIR" \
    --output-dir "$STEP03_DIR" \
    --spec       "$SPEC" \
    $SUFFIX_ARG \
    $FORCE

echo "=== Step 4: Finalize Config ==="
python -m w2ner_biomedical.pipeline.step04_finalize_config \
    --config     "$CONFIG" \
    --spec       "$SPEC" \
    --step03-dir "$STEP03_DIR" \
    --cache-dir  "$CACHE_DIR" \
    $SUFFIX_ARG

echo "=== Step 5: Train ==="
VAL_ARG=""
[ -n "$VAL_DIR" ] && VAL_ARG="--val-dir $VAL_DIR"
python -m w2ner_biomedical.model.train \
    --config     "$CONFIG" \
    --spec       "$SPEC" \
    --input-dir  "$STEP03_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    $VAL_ARG

echo "=== Training complete. Model saved to $OUTPUT_DIR ==="
