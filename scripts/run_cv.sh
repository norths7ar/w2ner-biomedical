#!/usr/bin/env bash
# =============================================================================
# scripts/run_cv.sh
#
# K-fold cross-validation: for each fold, run the full training pipeline
# on the training split and the full inference pipeline on the held-out
# split, then aggregate performance metrics across folds.
#
# Usage:
#   ./scripts/run_cv.sh [--folds 5] [--config configs/biored_base.json]
#                       [--input-dir /path/to/annotations]
#                       [--cv-dir /path/to/cv_splits]
#                       [--output-dir /path/to/cv_results]
#
# Prerequisites:
#   CV split files must already exist in --cv-dir, produced by a separate
#   split script (e.g. tool_split_train_test equivalent).
#   Each fold's train/val/test splits are identified by filename convention:
#     {cv-dir}/fold_{k}_train.jsonl, fold_{k}_val.jsonl, fold_{k}_test.jsonl
#
# NOTE: This script runs folds sequentially.  For parallel fold execution,
#   launch multiple instances with different --fold-id arguments.
# =============================================================================

set -euo pipefail

FOLDS="${FOLDS:-5}"
CONFIG="${CONFIG:-configs/biored_base.json}"
SPEC="${SPEC:-specs/label_spec.json}"
CV_DIR="${CV_DIR:-data/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cv_results}"
FORCE="${FORCE:-}"

# TODO: parse CLI args from $@

for FOLD in $(seq 0 $((FOLDS - 1))); do
    echo "=== Fold ${FOLD} / $((FOLDS - 1)) ==="

    FOLD_TRAIN_DIR="${CV_DIR}/fold_${FOLD}_train"
    FOLD_VAL_DIR="${CV_DIR}/fold_${FOLD}_val"
    FOLD_TEST_DIR="${CV_DIR}/fold_${FOLD}_test"
    FOLD_MODEL_DIR="${OUTPUT_DIR}/fold_${FOLD}/model"
    FOLD_PRED_DIR="${OUTPUT_DIR}/fold_${FOLD}/predictions"

    echo "--- Fold ${FOLD}: Train ---"
    # TODO: MODEL_NAME="cv_fold${FOLD}" CONFIG="$CONFIG" \
    #       INPUT_DIR="$FOLD_TRAIN_DIR" \
    #       VAL_DIR="$FOLD_VAL_DIR" \
    #       OUTPUT_DIR="$FOLD_MODEL_DIR" \
    #       bash scripts/run_train.sh

    echo "--- Fold ${FOLD}: Predict on test split ---"
    # TODO: MODEL_NAME="cv_fold${FOLD}" CONFIG="$CONFIG" \
    #       INPUT_DIR="$FOLD_TEST_DIR" \
    #       OUTPUT_DIR="$FOLD_PRED_DIR" \
    #       bash scripts/run_predict.sh

    echo "--- Fold ${FOLD}: Evaluate ---"
    # TODO: python -m tool_calc_performance \
    #         --gold "$FOLD_TEST_DIR" \
    #         --pred "$FOLD_PRED_DIR" \
    #         --output "${OUTPUT_DIR}/fold_${FOLD}_metrics.json"

done

echo "=== Aggregating CV metrics ==="
# TODO: python -m tool_eval_cv_performance \
#         --cv-dir "$OUTPUT_DIR" \
#         --folds  "$FOLDS"

echo "=== Cross-validation complete. Results in $OUTPUT_DIR ==="
