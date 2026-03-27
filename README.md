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
w2ner-biomedical/
├── configs/
│   └── biored_base.json        # Architecture + optimiser hyperparameters only.
│                               # Vocabulary (entity_types, label_num) is NOT here.
├── specs/
│   ├── label_spec.json         # Authoritative entity type vocabulary. Edit deliberately.
│   └── schemas.py              # Pydantic stage-boundary schemas.
├── pipeline/
│   ├── step01_ingest.py        # Unicode normalise; emit IngestRecord JSONL.
│   ├── step02_tokenize.py      # Sentence-split + word-tokenise + subword-chunk (merged).
│   ├── step03_add_labels.py    # Align annotations to sentence chunks; populate ner field.
│   ├── step04_finalize_config.py  # Validate vocabulary; write entity_types into config.
│   ├── step05_predict.py       # Inference: encode + forward pass + decode_grid.
│   └── step06_postprocess.py   # Char span recovery + majority-vote type normalisation.
├── model/
│   ├── ner_model.py            # W2NER neural architecture.
│   ├── trainer.py              # Training loop, loss, validation, early stopping.
│   ├── decoding.py             # Grid → entity span decoder.
│   └── train.py                # Training entry point.
├── data/
│   ├── feature_builder.py      # TokenRecord → model tensors.
│   └── collate.py              # DataLoader collate function.
├── guards/
│   └── validators.py           # Five explicit pipeline guards.
├── converters/
│   ├── bc5cdr_to_schema.py     # BC5CDR BioC XML → IngestRecord JSON.
│   ├── biored_to_schema.py     # BioRED BioC XML → IngestRecord JSON (incl. discontinuous).
│   └── README.md               # Corpus format notes and usage examples.
├── scripts/
│   ├── run_train.sh
│   ├── run_predict.sh
│   └── run_cv.sh
├── tests/
│   └── test_roundtrip.py       # Encode→decode round-trip tests.
├── pyproject.toml              # pip install -e . for cross-script imports.
└── README.md
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

### 6. Dependency direction: model/ never imports from pipeline/

The `model/` layer (ner_model, trainer, decoding, train) has no imports from
`pipeline/` or `guards/`. The dependency graph flows strictly:
`pipeline/ → data/ → model/` and `pipeline/ → guards/`.

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

```bash
# Training
bash scripts/run_train.sh \
  --model-name biored_base \
  --input-dir  /path/to/annotations \
  --val-dir    /path/to/val_annotations

# Inference
bash scripts/run_predict.sh \
  --model-name biored_base \
  --input-dir  /path/to/new_documents

# Cross-validation (5-fold)
bash scripts/run_cv.sh \
  --folds    5 \
  --cv-dir   /path/to/cv_splits \
  --output-dir /path/to/cv_results
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
