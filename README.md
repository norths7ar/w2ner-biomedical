# w2ner-biomedical

A rebuilt biomedical Named Entity Recognition pipeline based on the
[W2NER](https://arxiv.org/abs/2112.10070) (Word-to-Word NER) architecture,
using `dmis-lab/biobert-base-cased-v1.1` as the backbone encoder.

The starting point for this repository is an existing internal pipeline that
already represents a substantial refactoring of the original academic W2NER
codebase — with improved modularity, a cleaner model interface, a step-based
preprocessing chain, and a working inference pipeline.  This repository is
not a fix of the raw original W2NER; it is an architectural redesign of that
already-refactored internal pipeline, focused on making stage contracts
explicit, eliminating silent failure modes, and separating concerns that were
conflated in the previous design (most notably: vocabulary derivation from
data, config mutation as a side effect of data processing, and position-based
joining between pipeline stages).

The core W2NER model architecture is preserved unchanged (BioBERT → BiLSTM →
dilated CNN → biaffine co-predictor → word-pair label grid).  The changes are
in the preprocessing and orchestration layer.

---

## Directory structure

```
w2ner-biomedical/               ← git repo root
├── configs/
│   └── biored_base.json        # Architecture + optimiser hyperparameters only.
│                               # Vocabulary (entity_types, label_num) is injected by step04.
├── scripts/
│   ├── run_train.sh            # Steps 1→2→3→4→train.
│   ├── run_predict.sh          # Steps 1→2→5→6 (no labels required).
│   ├── run_cv.sh               # K-fold loop over run_train.sh + run_predict.sh.
│   ├── run_train.ps1           # PowerShell equivalent of run_train.sh.
│   ├── run_predict.ps1         # PowerShell equivalent of run_predict.sh.
│   └── run_cv.ps1              # PowerShell equivalent of run_cv.sh.
├── pyproject.toml              # pip install -e . for intra-package imports.
├── README.md
└── w2ner_biomedical/           ← Python package (import w2ner_biomedical.*)
    ├── specs/
    │   ├── label_spec.json     # Authoritative entity type vocabulary. Edit deliberately.
    │   └── schemas.py          # Pydantic stage-boundary schemas (IngestRecord → PostprocessRecord).
    ├── pipeline/
    │   ├── _utils.py           # Shared helpers: file_sha256, write_stage_manifest, build_base_parser.
    │   ├── step01_ingest.py    # Unicode normalise; emit IngestRecord JSONL.
    │   ├── step02_tokenize.py  # Sentence-split + word-tokenise + subword-chunk (merged).
    │   ├── step03_add_labels.py   # Align annotations to sentence chunks; populate ner field.
    │   ├── step04_finalize_config.py  # Validate vocabulary; write entity_types into config.
    │   ├── step05_predict.py   # Inference: encode + forward pass + decode_grid.
    │   └── step06_postprocess.py  # Char span recovery + majority-vote type normalisation.
    ├── model/
    │   ├── constants.py        # Shared numeric constants: NNW_LABEL, CLS_OFFSET, DIST_DIAGONAL.
    │   ├── ner_model.py        # W2NER neural architecture.
    │   ├── trainer.py          # Training loop, loss, validation, early stopping.
    │   ├── decoding.py         # Grid → entity span decoder.
    │   └── train.py            # Training entry point.
    ├── data/
    │   ├── feature_builder.py  # TokenRecord → model tensors.
    │   └── collate.py          # DataLoader collate function.
    ├── guards/
    │   └── validators.py       # Five explicit pipeline guards.
    ├── converters/
    │   ├── bc5cdr_to_schema.py # BC5CDR BioC XML → annotation JSON.
    │   ├── biored_to_schema.py # BioRED BioC XML → annotation JSON (incl. discontinuous).
    │   └── README.md           # Corpus format notes and usage examples.
    ├── tools/
    │   └── evaluate.py         # Entity-level P/R/F1 evaluation (step06 output vs. gold JSON).
    └── tests/
        └── test_roundtrip.py   # Encode→decode round-trip tests.
```

---

## Key design decisions vs. the previous pipeline

The previous pipeline (`E:/GitHub Repos/W2NER`) was already a substantial
improvement over the raw academic W2NER codebase.  The changes below are
relative to that already-refactored version, not to the original paper
implementation.

### 1. Vocabulary defined explicitly, not discovered from data

**Previously:** `train05_add_labels.py` derived `entity_types` at runtime
from whatever entity type strings happened to appear in the annotation data,
sorted by frequency, then wrote them into the model config. This meant the
training vocabulary changed silently between runs if annotation data changed.

**This version:** `specs/label_spec.json` is the single source of truth for
entity types. It is version-controlled and edited deliberately. `step03_add_labels.py`
validates that all annotation types are present in the spec (unknown types
raise an error). `step04_finalize_config.py` writes `entity_types` into the
config from the spec in a deterministic order, then validates `bert_hid_size`
against the encoder. `label2id` is reproducible across runs.

### 2. Single-pass tokenization (step02 merges previous step03 + step04)

**Previously:** Two tokenization scripts. `step03_tokenize.py` used a heuristic
`-6 buffer` for chunking; `step04_apply_token_limit.py` re-tokenized with a
hard `max_length=500` limit. Two-pass chunking meant word boundaries could
differ between passes, misaligning character spans.

**This version:** One tokenization step (`step02_tokenize.py`) with a single
greedy forward pass over `word_ids()` output. One set of chunk boundaries,
one set of character spans, no re-basing.

### 3. Config mutation separated from data processing

**Previously:** `train05_add_labels.py` both assigned labels and rewrote the
model config (entity_types, label_num) as a side effect. A killed mid-run
left the config in a partially updated state.

**This version:** `step03_add_labels.py` is pure data transformation (no
config writes). `step04_finalize_config.py` is the only step that touches
the config, runs only after step03 has fully completed, and is idempotent.

### 4. ID-keyed joins replace position-based zip

**Previously:** `step07_post_process.py` joined token records and prediction
records via `zip()`, which silently truncated to the shorter list if step05
failed partway through a file.

**This version:** `step06_postprocess.py` joins on `TokenRecord.id ==
PredictRecord.id`. Missing IDs raise `ValueError` before any output is
written.

### 5. Explicit pipeline guards

`guards/validators.py` implements five guards that surface silent failures:

| Guard | What it detects |
|---|---|
| `check_record_count_parity` | Steps that silently drop >5% of records |
| `check_entity_alignment_rate` | Entity annotation drops (unicode drift, offset mismatch) |
| `check_type_vocabulary_consistency` | Unknown entity types reaching feature_builder |
| `check_id_join_completeness` | Missing prediction records in postprocessing join |
| `check_label_vocab_consistency` | Model head / label2id dimension mismatch at load time |

### 6. Shared constants in model/constants.py

`NNW_LABEL`, `CLS_OFFSET`, and `DIST_DIAGONAL` were previously duplicated
across `feature_builder.py`, `step02_tokenize.py`, `decoding.py`, and
`train.py`. All four now import from `model/constants.py`.

### 7. Shared pipeline utilities in pipeline/_utils.py

`file_sha256`, `write_stage_manifest`, and `build_base_parser` were previously
re-implemented per step. All six pipeline steps now import from `pipeline/_utils.py`.
`build_base_parser` provides the common `--output-dir` (required) and `--force`
arguments so each step script only declares its own specific arguments.

### 8. Dependency direction: model/ never imports from pipeline/

The `model/` layer (ner_model, trainer, decoding, train) has no imports from
`pipeline/` or `guards/`. The dependency graph flows strictly:
`pipeline/ → data/ → model/` and `pipeline/ → guards/`.

---

## Benchmark results

| Dataset | Backbone | Precision | Recall | F1 |
|---------|----------|-----------|--------|----|
| BioRED (test) | BioBERT-base-cased-v1.1 | 0.466 | 0.890 | 0.612 |

Trained with `min_bg_weight=0.05`, `epochs=20`. 
No hyperparameter search — these are baseline results.
Published SOTA on BioRED with PubMedBERT: ~0.89 (Luo et al. 2022).

---

## Known issues addressed

The following bugs from the previous internal pipeline are addressed in this
codebase. Each is tagged in the relevant file's header comment with its priority.

### CRITICAL

| ID | Description | Fixed in |
|---|---|---|
| Silent type fallback | `label2id.get(type, 0)` maps unknown entity types to background label with no error | `data/feature_builder.py` + `pipeline/step04_finalize_config.py` |
| Discontinuous entity encoding | `range(si, ei+1)` expands discontinuous spans into contiguous ones | `pipeline/step03_add_labels.py` (confirmed fix via multi-span input from upstream) |

### HIGH

| ID | Description | Fixed in |
|---|---|---|
| Class imbalance ~68:1 | No loss weighting for background-dominant grid | `model/trainer.py` |
| No validation loop | Fixed 20 epochs, last checkpoint only | `model/trainer.py` |
| min_value sentinel | Batch-dependent mask fill in max-pooling; should be -inf | `model/ner_model.py` |
| Bug B — Unicode offset drift | Normalisation shifts char offsets silently | `pipeline/step01_ingest.py` + `guards/validators.py` |
| Bug D — zip truncation | step06 assumes step05 never produces partial output | `pipeline/step06_postprocess.py` + `guards/validators.py` |
| CLS offset hardcoded | `+1` magic number in pieces2word construction | `data/feature_builder.py` (named constant `CLS_OFFSET`) |

### MEDIUM

| ID | Description | Fixed in |
|---|---|---|
| Upper-triangle NNW mask missing from loss | Model wastes capacity on unreachable lower-triangle cells | `model/trainer.py` |
| Explosion guard silent | Decoder silently continues with partial results | `model/decoding.py` |
| Bug E — majority vote key | Entity text string collapses context-dependent predictions | `pipeline/step06_postprocess.py` (quarantined with TODO) |
| Bug H — config clobbered | Partial train05 run corrupts entity_types | Eliminated by separating label assignment from config rewrite |
| bert_hid_size not validated | Shape mismatch silent until forward pass | `pipeline/step04_finalize_config.py` |

### LOW

| ID | Description | Fixed in |
|---|---|---|
| output_hidden_states=True always | Memory waste in last-layer-only mode | `model/ner_model.py` |
| Bug I — hardcoded batch_size/workers | Cannot tune inference throughput without editing source | `pipeline/step05_predict.py` |
| Training metric ≠ eval metric | Label-level macro F1 not comparable to entity-level F1 | `model/trainer.py` (both logged separately) |

---

## Running the pipeline

Install the package and its dependencies:

```bash
pip install -e .
```

**`en_core_sci_sm` (biomedical sentence model for step02) must be installed separately** — it is a model file distributed by the scispacy project, not a PyPI package, so it cannot be listed in `pyproject.toml`. Find the release matching your installed scispacy version at:

> https://github.com/allenai/scispacy/releases

Then install it directly, for example:

```powershell
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

Replace `v0.5.4` with the version that matches your `scispacy` install (`pip show scispacy` to check).

### Input data layout

The scripts accept a single JSON file (or a directory of JSON files) per split via `-InputDir` and `-ValDir`. Each split is processed through steps 1–3 into its own subdirectory under `-DataDir`. See the [Data storage and split identity](#data-storage-and-split-identity) section for the full folder layout.

The converter outputs (`data/converted/biored/`) can be passed directly — no need to copy files into extra directories first.

---

### Training (steps 1 → 2 → 3 → 4 → train)

**PowerShell (Windows):**

```powershell
.\scripts\run_train.ps1 `
    -BertName    dmis-lab/biobert-base-cased-v1.1 `
    -Config      configs/biored_base.json `
    -Spec        specs/label_spec.json `
    -ModelSuffix _biored `
    -InputDir    data/raw/biored/train.json `
    -ValDir      data/raw/biored/dev.json `
    -DataDir     data/biored `
    -OutputDir   models/biored_base
```

**Bash (Linux / macOS / WSL):**

```bash
bash scripts/run_train.sh \
  --bert-name    dmis-lab/biobert-base-cased-v1.1 \
  --config       configs/biored_base.json \
  --spec         specs/label_spec.json \
  --model-suffix _biored \
  --input-dir    data/raw/biored/train.json \
  --val-dir      data/raw/biored/dev.json \
  --data-dir     data/biored \
  --output-dir   models/biored_base
```

Both splits are processed through steps 1–3 independently. Intermediate outputs land in `data/biored/train/` and `data/biored/val/`. The trained model is saved to `--output-dir`.

---

### Inference (steps 1 → 2 → 5 → 6, no gold labels required)

**PowerShell:**

```powershell
.\scripts\run_predict.ps1 `
    -BertName  dmis-lab/biobert-base-cased-v1.1 `
    -Config    configs/biored_base.json `
    -ModelDir  models/biored_base `
    -InputDir  data/raw/biored_test `
    -OutputDir data/predictions
```

**Bash:**

```bash
bash scripts/run_predict.sh \
  --bert-name  dmis-lab/biobert-base-cased-v1.1 \
  --config     configs/biored_base.json \
  --model-dir  models/biored_base \
  --input-dir  data/raw/biored_test \
  --output-dir data/predictions
```

---

### Cross-validation (K-fold loop over train + predict)

CV split directories must be pre-built, one subdirectory per fold:

```
data/cv_splits/
├── fold_0_train/   ← training annotations for fold 0
├── fold_0_val/     ← validation annotations for fold 0
├── fold_0_test/    ← held-out test annotations for fold 0
├── fold_1_train/
...
```

**PowerShell:**

```powershell
.\scripts\run_cv.ps1 `
    -Folds       5 `
    -BertName    dmis-lab/biobert-base-cased-v1.1 `
    -Config      configs/biored_base.json `
    -ModelSuffix _biored `
    -CvDir       data/cv_splits `
    -OutputDir   data/cv_results
```

**Bash:**

```bash
bash scripts/run_cv.sh \
  --folds        5 \
  --bert-name    dmis-lab/biobert-base-cased-v1.1 \
  --config       configs/biored_base.json \
  --model-suffix _biored \
  --cv-dir       data/cv_splits \
  --output-dir   data/cv_results
```

Results for each fold are written to `data/cv_results/fold_N/model/` (weights) and `data/cv_results/fold_N/predictions/` (postprocessed spans).

---

### Reprocessing (--force / -Force)

By default, each step skips files whose output already exists. Pass `--force` (bash) or `-Force` (PowerShell) to reprocess everything from scratch:

```powershell
.\scripts\run_train.ps1 -InputDir data/raw/biored_train -Force
```

---

### PyTorch + CUDA setup (Windows)

The default conda environment installs a CPU-only PyTorch. For GPU training, reinstall with a CUDA-enabled build. The RTX 5070 (Blackwell, sm_120) requires **PyTorch 2.7+** with **CUDA 12.8**:

```powershell
conda activate w2ner-biomedical
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

### PowerShell execution policy

If PowerShell blocks the scripts, allow local scripts for your user session:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Annotation input format

Each annotation JSON file is an array of document objects:

```json
[
  {
    "PMID": "12345678",
    "articleTitle": "...",
    "abstract": "...",
    "label": [
      [[[10, 15]], "Gene_Or_GeneProduct"],
      [[[10, 12], [40, 45]], "Gene_Or_GeneProduct"]
    ]
  }
]
```

The `label` field is a list of `[spans_list, type_string]` pairs.
`spans_list` contains one `[start_char, end_char]` entry per fragment.
Discontinuous entities must provide **separate entries per fragment** — a
single wide span covering the whole extent is not supported and will be
treated as a contiguous entity (all intermediate words included).

Character offsets are absolute within the concatenated fulltext string
(`title + " " + abstract` after NFKC unicode normalisation).

---

## Data storage and split identity

### Folder layout

The orchestration scripts (`run_train.ps1` / `run_train.sh`) enforce a fixed sub-structure inside whatever root you pass as `-DataDir` / `--data-dir`. Each split gets its own subdirectory, so you can always identify what is in a folder from its path alone.

> **Note:** The individual step scripts (`step01_ingest.py`, etc.) write to wherever you point `--output-dir`. The directory layout below is only enforced when running through the orchestration scripts. If you invoke steps manually, you are responsible for keeping the paths consistent.

```
data/
├── raw/
│   ├── biored/
│   │   ├── train.json          ← converter output, one file per split
│   │   ├── dev.json
│   │   └── test.json
│   └── bc5cdr/
│       ├── train.json
│       ├── dev.json
│       └── test.json
│
├── biored/                     ← --data-dir data/biored
│   ├── train/                  ← training split intermediates
│   │   ├── step01_output/      ← IngestRecord JSONL
│   │   ├── step02_output/      ← TokenRecord JSONL (tokenized)
│   │   └── step03_output/      ← TokenRecord JSONL (with NER labels)
│   └── val/                    ← validation split intermediates (same layout)
│       ├── step01_output/
│       ├── step02_output/
│       └── step03_output/
│
└── bc5cdr/                     ← --data-dir data/bc5cdr
    ├── train/
    └── val/

models/
├── biored_base/                ← --output-dir models/biored_base
│   ├── model.pt
│   ├── label2id.json
│   ├── label_spec.json         ← copy saved alongside model for step05
│   └── logs/
│       └── train.log
└── bc5cdr_base/
```

Both train and val data are processed through steps 1–3 independently inside their respective subdirectories. `train.py` then receives:

- `--input-dir data/biored/train/step03_output/`
- `--val-dir   data/biored/val/step03_output/`

This means the split identity is always recoverable from the filesystem path, not just from memory of how the scripts were invoked.

### What the step manifest does NOT record

Every JSONL output has a `.meta.json` sidecar (`StageManifest`) recording the stage name, input filenames, a SHA-256 input hash, a record count, and a timestamp. It is used for cache invalidation and cross-stage record-count auditing.

The manifest does **not** record the dataset name (BioRED, BC5CDR, …) or the split role. That information is carried by the directory path.

### Running multiple datasets without collision

Pass a distinct `-DataDir` per dataset so intermediate outputs never overwrite each other:

```powershell
# BioRED
.\scripts\run_train.ps1 `
    -InputDir data/raw/biored/train.json `
    -ValDir   data/raw/biored/dev.json `
    -DataDir  data/biored `
    -OutputDir models/biored_base `
    -ModelSuffix _biored

# BC5CDR
.\scripts\run_train.ps1 `
    -InputDir data/raw/bc5cdr/train.json `
    -ValDir   data/raw/bc5cdr/dev.json `
    -DataDir  data/bc5cdr `
    -OutputDir models/bc5cdr_base `
    -ModelSuffix _bc5cdr
```

### What is embedded in each record

Individual `TokenRecord` objects carry a `pmid`, `document_id`, and a chunk `id` of the form `{pmid}_{sent_idx}_{chunk_idx}`. These trace any prediction back to its source document and sentence, but they do not encode split or dataset membership — that is the directory's job.
