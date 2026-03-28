# =============================================================================
# scripts/run_cv.ps1
#
# K-fold cross-validation: for each fold, run the full training pipeline
# on the training split and the full inference pipeline on the held-out
# split, then aggregate performance metrics across folds.
#
# Usage:
#   .\scripts\run_cv.ps1 [-Folds 5] [-BertName dmis-lab/biobert-base-cased-v1.1]
#                        [-Config configs/biored_base.json]
#                        [-Spec specs/label_spec.json]
#                        [-ModelSuffix _biored]
#                        [-CvDir data/cv_splits]
#                        [-OutputDir data/cv_results]
#                        [-CacheDir cache]
#                        [-BatchSize 8] [-NumWorkers 0]
#                        [-Force]
#
# Prerequisites:
#   CV split directories must already exist under -CvDir, each containing
#   raw annotation *.json files for that fold's train/val/test split:
#     {CvDir}/fold_{k}_train/   <- training annotations
#     {CvDir}/fold_{k}_val/     <- validation annotations (optional)
#     {CvDir}/fold_{k}_test/    <- held-out test annotations
# =============================================================================

param(
    [int]$Folds          = 5,
    [string]$BertName    = "dmis-lab/biobert-base-cased-v1.1",
    [string]$Config      = "configs/biored_base.json",
    [string]$Spec        = "specs/label_spec.json",
    [string]$ModelSuffix = "",
    [string]$CvDir       = "data/cv_splits",
    [string]$OutputDir   = "data/cv_results",
    [string]$CacheDir    = "cache",
    [int]$BatchSize      = 8,
    [int]$NumWorkers     = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

for ($Fold = 0; $Fold -lt $Folds; $Fold++) {
    Write-Host "=== Fold $Fold / $($Folds - 1) ===" -ForegroundColor Yellow

    $FoldTrainDir = "$CvDir/fold_${Fold}_train"
    $FoldValDir   = "$CvDir/fold_${Fold}_val"
    $FoldTestDir  = "$CvDir/fold_${Fold}_test"
    $FoldModelDir = "$OutputDir/fold_${Fold}/model"
    $FoldDataDir  = "$OutputDir/fold_${Fold}/data"
    $FoldPredDir  = "$OutputDir/fold_${Fold}/predictions"

    Write-Host "--- Fold ${Fold}: Train ---" -ForegroundColor Cyan
    $trainArgs = @(
        "-BertName",    $BertName,
        "-Config",      $Config,
        "-Spec",        $Spec,
        "-InputDir",    $FoldTrainDir,
        "-ValDir",      $FoldValDir,
        "-DataDir",     $FoldDataDir,
        "-OutputDir",   $FoldModelDir,
        "-CacheDir",    $CacheDir
    )
    if ($ModelSuffix) { $trainArgs += "-ModelSuffix", $ModelSuffix }
    if ($Force)       { $trainArgs += "-Force" }
    & "$ScriptDir\run_train.ps1" @trainArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "--- Fold ${Fold}: Predict on test split ---" -ForegroundColor Cyan
    $predictArgs = @(
        "-BertName",    $BertName,
        "-Config",      $Config,
        "-ModelDir",    $FoldModelDir,
        "-InputDir",    $FoldTestDir,
        "-DataDir",     "$FoldDataDir/test",
        "-OutputDir",   $FoldPredDir,
        "-CacheDir",    $CacheDir,
        "-BatchSize",   $BatchSize,
        "-NumWorkers",  $NumWorkers
    )
    if ($Force) { $predictArgs += "-Force" }
    & "$ScriptDir\run_predict.ps1" @predictArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "--- Fold ${Fold}: Evaluate ---" -ForegroundColor Cyan
    # TODO: python -m w2ner_biomedical.tools.calc_performance `
    #         --gold $FoldTestDir `
    #         --pred $FoldPredDir `
    #         --output "$OutputDir/fold_${Fold}_metrics.json"
}

Write-Host "=== Aggregating CV metrics ===" -ForegroundColor Yellow
# TODO: python -m w2ner_biomedical.tools.eval_cv_performance `
#         --cv-dir $OutputDir `
#         --folds  $Folds

Write-Host "=== Cross-validation complete. Results in $OutputDir ===" -ForegroundColor Green
