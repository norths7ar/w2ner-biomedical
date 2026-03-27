#!/usr/bin/env bash
# =============================================================================
# scripts/run_cv.sh
#
# K-fold cross-validation: for each fold, run the full training pipeline
# on the training split and the full inference pipeline on the held-out
# split, then aggregate performance metrics across folds.
#
# Usage:
#   ./scripts/run_cv.sh [--folds 5] [--bert-name HF_ID]
#                       [--config configs/biored_base.json]
#                       [--spec specs/label_spec.json]
#                       [--model-suffix _biored]
#                       [--cv-dir data/cv_splits]
#                       [--output-dir data/cv_results]
#                       [--cache-dir cache]
#                       [--batch-size 8] [--num-workers 0]
#                       [--force]
#
# Prerequisites:
#   CV split directories must already exist under --cv-dir, each containing
#   raw annotation *.json files for that fold's train/val/test split:
#     {cv-dir}/fold_{k}_train/   ← training annotations
#     {cv-dir}/fold_{k}_val/     ← validation annotations (optional)
#     {cv-dir}/fold_{k}_test/    ← held-out test annotations
#
# NOTE: Folds run sequentially.  For parallel execution, launch multiple
#   instances with different --fold-start / --fold-end arguments, or
#   invoke run_train.sh and run_predict.sh directly per fold.
# =============================================================================

set -euo pipefail

FOLDS="${FOLDS:-5}"
BERT_NAME="${BERT_NAME:-dmis-lab/biobert-base-cased-v1.1}"
CONFIG="${CONFIG:-configs/biored_base.json}"
SPEC="${SPEC:-specs/label_spec.json}"
MODEL_SUFFIX="${MODEL_SUFFIX:-}"
CV_DIR="${CV_DIR:-data/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cv_results}"
CACHE_DIR="${CACHE_DIR:-cache}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FORCE="${FORCE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds)        FOLDS="$2";        shift 2 ;;
        --bert-name)    BERT_NAME="$2";    shift 2 ;;
        --config)       CONFIG="$2";       shift 2 ;;
        --spec)         SPEC="$2";         shift 2 ;;
        --model-suffix) MODEL_SUFFIX="$2"; shift 2 ;;
        --cv-dir)       CV_DIR="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --cache-dir)    CACHE_DIR="$2";    shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";   shift 2 ;;
        --num-workers)  NUM_WORKERS="$2";  shift 2 ;;
        --force)        FORCE="--force";   shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for FOLD in $(seq 0 $((FOLDS - 1))); do
    echo "=== Fold ${FOLD} / $((FOLDS - 1)) ==="

    FOLD_TRAIN_DIR="${CV_DIR}/fold_${FOLD}_train"
    FOLD_VAL_DIR="${CV_DIR}/fold_${FOLD}_val"
    FOLD_TEST_DIR="${CV_DIR}/fold_${FOLD}_test"
    FOLD_MODEL_DIR="${OUTPUT_DIR}/fold_${FOLD}/model"
    FOLD_DATA_DIR="${OUTPUT_DIR}/fold_${FOLD}/data"
    FOLD_PRED_DIR="${OUTPUT_DIR}/fold_${FOLD}/predictions"

    echo "--- Fold ${FOLD}: Train ---"
    BERT_NAME="$BERT_NAME" CONFIG="$CONFIG" SPEC="$SPEC" \
    MODEL_SUFFIX="$MODEL_SUFFIX" CACHE_DIR="$CACHE_DIR" \
    INPUT_DIR="$FOLD_TRAIN_DIR" VAL_DIR="$FOLD_VAL_DIR" \
    DATA_DIR="$FOLD_DATA_DIR/train" OUTPUT_DIR="$FOLD_MODEL_DIR" \
    FORCE="$FORCE" \
        bash "$SCRIPT_DIR/run_train.sh"

    echo "--- Fold ${FOLD}: Predict on test split ---"
    BERT_NAME="$BERT_NAME" CONFIG="$CONFIG" \
    MODEL_DIR="$FOLD_MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
    INPUT_DIR="$FOLD_TEST_DIR" \
    DATA_DIR="$FOLD_DATA_DIR/test" OUTPUT_DIR="$FOLD_PRED_DIR" \
    BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" FORCE="$FORCE" \
        bash "$SCRIPT_DIR/run_predict.sh"

    echo "--- Fold ${FOLD}: Evaluate ---"
    # TODO: python -m w2ner_biomedical.tools.calc_performance \
    #         --gold "$FOLD_TEST_DIR" \
    #         --pred "$FOLD_PRED_DIR" \
    #         --output "${OUTPUT_DIR}/fold_${FOLD}_metrics.json"

done

echo "=== Aggregating CV metrics ==="
# TODO: python -m w2ner_biomedical.tools.eval_cv_performance \
#         --cv-dir "$OUTPUT_DIR" \
#         --folds  "$FOLDS"

echo "=== Cross-validation complete. Results in $OUTPUT_DIR ==="
