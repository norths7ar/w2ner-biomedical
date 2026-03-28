#!/usr/bin/env bash
# =============================================================================
# scripts/run_train.sh
#
# Full training pipeline: ingest → tokenize → add labels →
#   finalize config → train model.
#
# Both training and validation data are run through steps 1-3 independently.
# Intermediate outputs are kept under separate train/ and val/ subdirectories
# inside --data-dir so the split is always identifiable from the directory tree.
#
# Folder layout produced under --data-dir:
#   {data-dir}/
#     train/
#       step01_output/    ← IngestRecord JSONL for training split
#       step02_output/    ← TokenRecord JSONL for training split
#       step03_output/    ← TokenRecord+NER JSONL for training split
#     val/                ← only created when --val-dir is supplied
#       step01_output/
#       step02_output/
#       step03_output/
#
# Usage:
#   ./scripts/run_train.sh \
#       --input-dir    data/raw/biored/train.json \
#       --val-dir      data/raw/biored/dev.json \
#       --data-dir     data/biored \
#       --output-dir   models/biored_base \
#       --model-suffix _biored
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

SUFFIX_ARG=""
[ -n "$MODEL_SUFFIX" ] && SUFFIX_ARG="--model-suffix $MODEL_SUFFIX"

# Train split dirs
TRAIN_STEP01="${DATA_DIR}/train/step01_output"
TRAIN_STEP02="${DATA_DIR}/train/step02_output"
TRAIN_STEP03="${DATA_DIR}/train/step03_output"

# Val split dirs (only used when --val-dir is provided)
VAL_STEP01="${DATA_DIR}/val/step01_output"
VAL_STEP02="${DATA_DIR}/val/step02_output"
VAL_STEP03="${DATA_DIR}/val/step03_output"

# ── Training split: steps 1, 2, 3 ─────────────────────────────────────────

echo "=== [Train] Step 1: Ingest ==="
python -m w2ner_biomedical.pipeline.step01_ingest \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$TRAIN_STEP01" \
    $FORCE

echo "=== [Train] Step 2: Tokenize ==="
python -m w2ner_biomedical.pipeline.step02_tokenize \
    --input-dir  "$TRAIN_STEP01" \
    --output-dir "$TRAIN_STEP02" \
    --bert-name  "$BERT_NAME" \
    --cache-dir  "$CACHE_DIR" \
    --workers    "$WORKERS" \
    $FORCE

echo "=== [Train] Step 3: Add Labels ==="
python -m w2ner_biomedical.pipeline.step03_add_labels \
    --input-dir  "$INPUT_DIR" \
    --tokens-dir "$TRAIN_STEP02" \
    --output-dir "$TRAIN_STEP03" \
    --spec       "$SPEC" \
    $SUFFIX_ARG \
    $FORCE

# ── Validation split: steps 1, 2, 3 (only if --val-dir provided) ──────────

if [ -n "$VAL_DIR" ]; then
    echo "=== [Val] Step 1: Ingest ==="
    python -m w2ner_biomedical.pipeline.step01_ingest \
        --input-dir  "$VAL_DIR" \
        --output-dir "$VAL_STEP01" \
        $FORCE

    echo "=== [Val] Step 2: Tokenize ==="
    python -m w2ner_biomedical.pipeline.step02_tokenize \
        --input-dir  "$VAL_STEP01" \
        --output-dir "$VAL_STEP02" \
        --bert-name  "$BERT_NAME" \
        --cache-dir  "$CACHE_DIR" \
        --workers    "$WORKERS" \
        $FORCE

    echo "=== [Val] Step 3: Add Labels ==="
    python -m w2ner_biomedical.pipeline.step03_add_labels \
        --input-dir  "$VAL_DIR" \
        --tokens-dir "$VAL_STEP02" \
        --output-dir "$VAL_STEP03" \
        --spec       "$SPEC" \
        $SUFFIX_ARG \
        $FORCE
fi

# ── Step 4: Finalize Config ────────────────────────────────────────────────

echo "=== Step 4: Finalize Config ==="
python -m w2ner_biomedical.pipeline.step04_finalize_config \
    --config     "$CONFIG" \
    --spec       "$SPEC" \
    --step03-dir "$TRAIN_STEP03" \
    --cache-dir  "$CACHE_DIR" \
    $SUFFIX_ARG

# ── Step 5: Train ──────────────────────────────────────────────────────────

echo "=== Step 5: Train ==="
VAL_ARG=""
[ -n "$VAL_DIR" ] && VAL_ARG="--val-dir $VAL_STEP03"
python -m w2ner_biomedical.model.train \
    --config     "$CONFIG" \
    --spec       "$SPEC" \
    --input-dir  "$TRAIN_STEP03" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    $VAL_ARG

echo "=== Training complete. Model saved to $OUTPUT_DIR ==="
