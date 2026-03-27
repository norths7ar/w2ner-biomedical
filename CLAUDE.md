# CLAUDE.md — w2ner-biomedical

Developer guide for Claude Code when working in this repository.

---

## Environment

- **Conda env**: `w2ner-biomedical` (miniconda3 at `C:/Users/jnkyl/miniconda3`)
- **Python**: Run all commands via `/c/Users/jnkyl/miniconda3/envs/w2ner-biomedical/python.exe`
- **Tests**: `python -m pytest w2ner_biomedical/tests/ -m "not integration" -q`
- **Integration tests** (need real corpus files under `data/`): add `-m integration`
- `conda` is not on PATH in the shell; use the full interpreter path above.

---

## Repo layout

```
w2ner_biomedical/
  converters/        # BioC-XML → annotation JSON (bc5cdr_to_schema, biored_to_schema)
  data/              # feature_builder.py, NERDataset, collate
  guards/            # validators.py — 5 explicit pipeline guards
  model/             # W2NER model, decoding, train, constants
  pipeline/          # step01–step06 scripts + _utils.py
  specs/             # schemas.py (Pydantic), label_spec.json
  tests/             # pytest suite

configs/             # model hyperparameter JSON (no label vocab — that lives in specs/)
data/                # runtime data (gitignored); raw corpora downloaded here
scripts/             # run_train.sh, run_predict.sh, run_cv.sh
```

---

## Pipeline overview

Six sequential steps. Each writes JSONL + a `.meta.json` sidecar (StageManifest).

| Step | Script | Input → Output |
|------|--------|----------------|
| 01 | `step01_ingest.py` | raw annotation JSON → `IngestRecord` JSONL |
| 02 | `step02_tokenize.py` | IngestRecord → `TokenRecord` JSONL (spaCy senter, word spans) |
| 03 | `step03_add_labels.py` | TokenRecord + annotation JSON → TokenRecord with NER |
| 04 | `step04_finalize_config.py` | config JSON → config JSON (injects `entity_types`, `label_num`) |
| 05 | `step05_predict.py` | TokenRecord → `PredictRecord` JSONL (model inference) |
| 06 | `step06_postprocess.py` | PredictRecord + IngestRecord → `PostprocessRecord` JSONL |

All steps share `--output-dir` and `--force` from `pipeline/_utils.py:build_base_parser`.

---

## Key design invariants

**Label vocabulary** — `specs/label_spec.json` is the single source of truth. Never derive entity types from runtime data. `step04` is the only script that writes `entity_types` / `label_num` into a config. Sentinel IDs: `<pad>=0`, `<suc>=1`; real types start at 2.

**Character offsets** — All offsets are fulltext-absolute (`title + " " + abstract`, NFKC-normalised once in step01). No re-basing downstream.

**ID-keyed join** — `step06` joins on `TokenRecord.id == PredictRecord.id`. Never use `zip()` across record lists (Bug D).

**Atomic writes** — All JSONL outputs use `tempfile + os.replace`.

**model/constants.py** — Single source for `NNW_LABEL=1`, `CLS_OFFSET=1`, `DIST_DIAGONAL=19`. Import from there; do not redefine locally.

**pipeline/_utils.py** — Shared `file_sha256`, `write_stage_manifest`, `build_base_parser`. All step scripts use these; do not duplicate.

---

## Guards (guards/validators.py)

| Guard | Trigger point | Detects |
|-------|--------------|---------|
| 1 `check_record_count_parity` | After steps that preserve record count | >5% record drop |
| 2 `check_entity_alignment_rate` | After step03 per file | Entity annotation drop (offset drift) |
| 3 `check_type_vocabulary_consistency` | step04 before rewriting config | Entity type vocabulary shrinkage |
| 4 `check_id_join_completeness` | step06 before join | Missing prediction records |
| 5 `check_label_vocab_consistency` | step05 at model load | Model head dim ≠ len(label2id) |

---

## Schemas (specs/schemas.py)

`IngestRecord → TokenRecord → (PredictRecord) → PostprocessRecord`

Each stage reads one schema and writes another. `StageManifest` is written as a sidecar `.meta.json` alongside every JSONL output.

---

## Converters

`converters/bc5cdr_to_schema.py` and `converters/biored_to_schema.py` convert BioC-XML corpora to the annotation JSON format that `step01_ingest.py` consumes. Both use shared offset utilities in `converters/_bioc_utils.py`.

**BC5CDR**: 500 train / 500 dev / 500 test documents; 73 multi-location (discontinuous) Disease annotations in the training set.

**BioRED**: 400 train / 100 dev / 100 test documents; no multi-location annotations.

Corpus files live under `data/` (gitignored). Expected paths:
- `data/BioRED/Train.BioC.XML`, `Dev.BioC.XML`, `Test.BioC.XML`
- `data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml`, etc.

---

## Commit conventions

- Stage only the files relevant to the change — never `git add -A`.
- Commit message format: `<type>(<scope>): <description>` (feat, fix, refactor, tests, docs, chore).

---

## What NOT to do

- Do not redefine `NNW_LABEL`, `CLS_OFFSET`, or `DIST_DIAGONAL` locally — import from `model/constants.py`.
- Do not add `_file_sha256` or `_write_manifest` helpers to step scripts — use `pipeline/_utils.py`.
- Do not use `zip()` to pair token and prediction records — use `join_by_id` + Guard 4.
- Do not write `entity_types` or `label_num` into a config anywhere except `step04_finalize_config.py`.
- Do not leave `__init__.py` files in `data/`, `model/`, or `pipeline/` with re-exports — keep them empty.
- Do not commit files under `data/`, `models/`, or `cache/` — they are gitignored runtime outputs.
