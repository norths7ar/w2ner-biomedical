# converters/

Standalone scripts that convert public biomedical NER corpora into the
`IngestRecord`-compatible JSON format consumed by `pipeline/step01_ingest.py`.

Each converter is self-contained and produces a single JSON array file that
can be dropped directly into the pipeline's `INPUT_DIR`.

---

## Supported corpora

### BC5CDR — BioCreative V Chemical-Disease Relation

| Property | Value |
|---|---|
| Task | Chemical and Disease NER + relation extraction |
| Entity types | `Chemical`, `Disease` |
| Discontinuous entities | None (single span per annotation) |
| Source format | BioC XML (`CDR_TrainingSet.BioC.xml` etc.) |
| Official splits | Training / Development / Test |
| Download | https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/ |

**Usage**
```bash
for split in Training Development Test; do
  python -m converters.bc5cdr_to_schema \
    --input  data/raw/bc5cdr/CDR_${split}Set.BioC.xml \
    --output data/raw/bc5cdr_converted/${split,,}.json
done
```

---

### BioRED — Biomedical Relation Extraction Dataset

| Property | Value |
|---|---|
| Task | Multi-type NER + relation extraction |
| Entity types | `Chemical`, `Disease`, `Gene` (→`Gene_Or_GeneProduct`), `Variant` (→`VariantOrPolymorphism`), `Species`, `CellLine`, `DNAMutation` (→`DNA_Mutation`) |
| Discontinuous entities | **Yes** — multi-location annotations in BioC |
| Source format | BioC XML (`BioRED.xml`) |
| Official splits | Train / Dev / Test (provided as separate files or as a single file with split annotations) |
| Download | https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/ |

**Usage**
```bash
python -m converters.biored_to_schema \
  --input  data/raw/biored/BioRED_train.xml \
  --output data/raw/biored_converted/train.json
```

---

## Output format

Both converters produce a JSON array of objects compatible with the
`IngestRecord` schema defined in `specs/schemas.py`:

```json
[
  {
    "PMID": "12345678",
    "articleTitle": "Title of the article.",
    "abstract": "Abstract text goes here ...",
    "label": [
      [[[5, 13]], "Chemical"],
      [[[25, 35], [80, 86]], "Gene_Or_GeneProduct"]
    ]
  }
]
```

### `label` field structure

Each element of `label` is `[spans_list, entity_type]`:

- `spans_list` — a list of `[start_char, end_char]` pairs, one per contiguous
  fragment of the entity. For most entities this has exactly one entry.
  For discontinuous entities (BioRED only) it has two or more entries.
- `entity_type` — the normalised type string, matching an entry in
  `specs/label_spec.json`.

### Character offset coordinate system

All character offsets are **fulltext-absolute**, where:

```
fulltext = title + " " + abstract
```

This matches the coordinate system established by `pipeline/step01_ingest.py`
and used throughout the rest of the pipeline.  Both converters handle the
re-basing from BioC's document-absolute offsets to this coordinate system.

---

## BioC offset convention

Both BC5CDR and BioRED follow BioC's document-absolute offset convention:
annotation offsets are measured from the start of the full document (offset 0
= first character of the title), not from the start of each passage.

This means a BioC annotation with `offset=150` is at character 150 of the
full document, regardless of whether it is in the title or abstract passage.

Our `fulltext` string uses a single space (`" "`) as the separator between
title and abstract.  If the BioC document uses a different separator (e.g.
`\n` in some BC5CDR versions), the converters detect and correct for this so
that our fulltext offsets remain accurate.

---

## Entity type mapping

| BioRED / BC5CDR type | `label_spec.json` type |
|---|---|
| `Chemical` | `Chemical` |
| `Disease` | `Disease` |
| `Gene` | `Gene_Or_GeneProduct` |
| `Variant`, `SequenceVariant` | `VariantOrPolymorphism` |
| `Species` | `Species` |
| `CellLine` | `CellLine` |
| `DNAMutation` | `DNA_Mutation` |

Unknown types are **skipped with a WARNING** at conversion time.
They never reach `step03_add_labels.py`'s vocabulary guard.

---

## Known limitations

- **Relation annotations** (R-lines in BioRED) are ignored. These converters
  are NER-only.
- **Normalisation identifiers** (MeSH, NCBIGene, etc.) are not carried
  through; the pipeline operates on span + type only.
- **Cross-sentence entities** (entities whose span crosses a sentence
  boundary assigned by `step02_tokenize.py`) will be dropped silently by
  `step03_add_labels.py` and counted in the entity alignment drop rate guard.
  These are rare in both corpora but present.
