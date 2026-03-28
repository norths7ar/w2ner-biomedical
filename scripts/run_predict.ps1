# =============================================================================
# scripts/run_predict.ps1
#
# Inference pipeline: ingest -> tokenize -> predict -> postprocess.
# Does not run step03_add_labels (no gold labels needed at inference time).
# Does not run step04_finalize_config (config is already finalised).
#
# Usage:
#   .\scripts\run_predict.ps1 [-ModelName biored_base] [-BertName dmis-lab/biobert-base-cased-v1.1]
#                             [-Config configs/biored_base.json]
#                             [-ModelDir models/biored_base]
#                             [-InputDir data/raw/annotations]
#                             [-DataDir data] [-OutputDir data/predictions]
#                             [-CacheDir cache]
#                             [-BatchSize 8] [-NumWorkers 0]
#                             [-Force]
# =============================================================================

param(
    [string]$ModelName  = "biored_base",
    [string]$BertName   = "dmis-lab/biobert-base-cased-v1.1",
    [string]$Config     = "configs/biored_base.json",
    [string]$ModelDir   = "",
    [string]$InputDir   = "data/raw/annotations",
    [string]$DataDir    = "data",
    [string]$OutputDir  = "data/predictions",
    [string]$CacheDir   = "cache",
    [int]$BatchSize     = 8,
    [int]$NumWorkers    = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not $ModelDir) { $ModelDir = "models/$ModelName" }

$Step01Dir = "$DataDir/step01_output"
$Step02Dir = "$DataDir/step02_output"
$Step05Dir = "$DataDir/step05_output"

Write-Host "=== Step 1: Ingest ===" -ForegroundColor Cyan
python -m w2ner_biomedical.pipeline.step01_ingest `
    --input-dir  $InputDir `
    --output-dir $Step01Dir `
    $(if ($Force) { "--force" })
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 2: Tokenize ===" -ForegroundColor Cyan
python -m w2ner_biomedical.pipeline.step02_tokenize `
    --input-dir  $Step01Dir `
    --output-dir $Step02Dir `
    --bert-name  $BertName `
    --cache-dir  $CacheDir `
    $(if ($Force) { "--force" })
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 5: Predict ===" -ForegroundColor Cyan
$step5Args = @(
    "-m", "w2ner_biomedical.pipeline.step05_predict",
    "--input-dir",   $Step02Dir,
    "--output-dir",  $Step05Dir,
    "--config",      $Config,
    "--model-dir",   $ModelDir,
    "--cache-dir",   $CacheDir,
    "--batch-size",  $BatchSize,
    "--num-workers", $NumWorkers
)
if ($Force) { $step5Args += "--force" }
python @step5Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 6: Postprocess ===" -ForegroundColor Cyan
$step6Args = @(
    "-m", "w2ner_biomedical.pipeline.step06_postprocess",
    "--tokens-dir",   $Step02Dir,
    "--pred-dir",     $Step05Dir,
    "--fulltext-dir", $Step01Dir,
    "--output-dir",   $OutputDir
)
if ($Force) { $step6Args += "--force" }
python @step6Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Prediction complete. Results in $OutputDir ===" -ForegroundColor Green
