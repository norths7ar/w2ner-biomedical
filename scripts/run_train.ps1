# =============================================================================
# scripts/run_train.ps1
#
# Full training pipeline: ingest -> tokenize -> add labels ->
#   finalize config -> train model.
#
# Both training and validation data are run through steps 1-3 independently.
# Intermediate outputs are kept under separate train/ and val/ subdirectories
# inside -DataDir so the split is always identifiable from the directory tree.
#
# Folder layout produced under -DataDir:
#   {DataDir}/
#     train/
#       step01_output/    <- IngestRecord JSONL for training split
#       step02_output/    <- TokenRecord JSONL for training split
#       step03_output/    <- TokenRecord+NER JSONL for training split
#     val/                <- only created when -ValDir is supplied
#       step01_output/
#       step02_output/
#       step03_output/
#
# Usage:
#   .\scripts\run_train.ps1 `
#       -InputDir    data/raw/biored/train.json `
#       -ValDir      data/raw/biored/dev.json `
#       -DataDir     data/biored `
#       -OutputDir   models/biored_base `
#       -ModelSuffix _biored
# =============================================================================

param(
    [string]$ModelName   = "biored_base",
    [string]$BertName    = "dmis-lab/biobert-base-cased-v1.1",
    [string]$Config      = "configs/biored_base.json",
    [string]$Spec        = "specs/label_spec.json",
    [string]$ModelSuffix = "",
    [string]$InputDir    = "data/raw/annotations",
    [string]$ValDir      = "",
    [string]$DataDir     = "data",
    [string]$OutputDir   = "",
    [string]$CacheDir    = "cache",
    [int]$Workers        = 1,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not $OutputDir) { $OutputDir = "models/$ModelName" }

# Train split dirs
$TrainStep01 = "$DataDir/train/step01_output"
$TrainStep02 = "$DataDir/train/step02_output"
$TrainStep03 = "$DataDir/train/step03_output"

# Val split dirs (only used when -ValDir is provided)
$ValStep01 = "$DataDir/val/step01_output"
$ValStep02 = "$DataDir/val/step02_output"
$ValStep03 = "$DataDir/val/step03_output"

# ── Training split: steps 1, 2, 3 ─────────────────────────────────────────

Write-Host "=== [Train] Step 1: Ingest ===" -ForegroundColor Cyan
python -m w2ner_biomedical.pipeline.step01_ingest `
    --input-dir  $InputDir `
    --output-dir $TrainStep01 `
    $(if ($Force) { "--force" })
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [Train] Step 2: Tokenize ===" -ForegroundColor Cyan
python -m w2ner_biomedical.pipeline.step02_tokenize `
    --input-dir  $TrainStep01 `
    --output-dir $TrainStep02 `
    --bert-name  $BertName `
    --cache-dir  $CacheDir `
    --workers    $Workers `
    $(if ($Force) { "--force" })
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [Train] Step 3: Add Labels ===" -ForegroundColor Cyan
$step3TrainArgs = @(
    "-m", "w2ner_biomedical.pipeline.step03_add_labels",
    "--input-dir",  $InputDir,
    "--tokens-dir", $TrainStep02,
    "--output-dir", $TrainStep03,
    "--spec",       $Spec
)
if ($ModelSuffix) { $step3TrainArgs += "--model-suffix", $ModelSuffix }
if ($Force)       { $step3TrainArgs += "--force" }
python @step3TrainArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ── Validation split: steps 1, 2, 3 (only if -ValDir provided) ────────────

if ($ValDir) {
    Write-Host "=== [Val] Step 1: Ingest ===" -ForegroundColor Cyan
    python -m w2ner_biomedical.pipeline.step01_ingest `
        --input-dir  $ValDir `
        --output-dir $ValStep01 `
        $(if ($Force) { "--force" })
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "=== [Val] Step 2: Tokenize ===" -ForegroundColor Cyan
    python -m w2ner_biomedical.pipeline.step02_tokenize `
        --input-dir  $ValStep01 `
        --output-dir $ValStep02 `
        --bert-name  $BertName `
        --cache-dir  $CacheDir `
        --workers    $Workers `
        $(if ($Force) { "--force" })
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "=== [Val] Step 3: Add Labels ===" -ForegroundColor Cyan
    $step3ValArgs = @(
        "-m", "w2ner_biomedical.pipeline.step03_add_labels",
        "--input-dir",  $ValDir,
        "--tokens-dir", $ValStep02,
        "--output-dir", $ValStep03,
        "--spec",       $Spec
    )
    if ($ModelSuffix) { $step3ValArgs += "--model-suffix", $ModelSuffix }
    if ($Force)       { $step3ValArgs += "--force" }
    python @step3ValArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ── Step 4: Finalize Config ────────────────────────────────────────────────

Write-Host "=== Step 4: Finalize Config ===" -ForegroundColor Cyan
$step4Args = @(
    "-m", "w2ner_biomedical.pipeline.step04_finalize_config",
    "--config",     $Config,
    "--spec",       $Spec,
    "--step03-dir", $TrainStep03,
    "--cache-dir",  $CacheDir
)
if ($ModelSuffix) { $step4Args += "--model-suffix", $ModelSuffix }
python @step4Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ── Step 5: Train ──────────────────────────────────────────────────────────

Write-Host "=== Step 5: Train ===" -ForegroundColor Cyan
$step5Args = @(
    "-m", "w2ner_biomedical.model.train",
    "--config",     $Config,
    "--spec",       $Spec,
    "--input-dir",  $TrainStep03,
    "--output-dir", $OutputDir,
    "--cache-dir",  $CacheDir
)
if ($ValDir) { $step5Args += "--val-dir", $ValStep03 }
python @step5Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Training complete. Model saved to $OutputDir ===" -ForegroundColor Green
