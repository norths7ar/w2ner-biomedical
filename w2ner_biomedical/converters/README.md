# converters/

Standalone scripts that convert public biomedical NER corpora into the
annotation JSON format consumed by `pipeline/step01_ingest.py`.

Each converter is self-contained and produces a single JSON array file that
can be dropped directly into the pipeline's `INPUT_DIR`.

---

## Supported corpora

### BC5CDR — BioCreative V Chemical-Disease Relation

| Property | Value |
|---|---|
| Task | Chemical and Disease NER + relation extraction |
| Entity types | `Chemical`, `Disease` |
| Discontinuous entities | **Yes** — 73 multi-location Disease annotations in training |
| Source format | BioC XML (`CDR_TrainingSet.BioC.xml` etc.) |
| Official splits | Training / Development / Test (500 docs each) |
| Standard path | `data/CDR_Data/CDR.Corpus.v010516/` |
| Download | https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/ |

**Usage**
```bash
for split in Training Development Test; do
  python -m w2ner_biomedical.converters.bc5cdr_to_schema \
    --input  data/CDR_Data/CDR.Corpus.v010516/CDR_${split}Set.BioC.xml \
    --output data/converted/bc5cdr/${split,,}.json
done
```

---

### BioRED — Biomedical Relation Extraction Dataset

| Property | Value |
|---|---|
| Task | Multi-type NER + relation extraction |
| Entity types | `ChemicalEntity` (→`Chemical`), `DiseaseOrPhenotypicFeature` (→`Disease`), `GeneOrGeneProduct` (→`Gene_Or_GeneProduct`), `OrganismTaxon` (→`Species`), `CellLine`, `SequenceVariant` (→`VariantOrPolymorphism`) |
| Discontinuous entities | None (all annotations have exactly one location) |
| Source format | BioC XML (`Train.BioC.XML`, `Dev.BioC.XML`, `Test.BioC.XML`) |
| Official splits | Train (400 docs) / Dev (100 docs) / Test (100 docs) |
| Standard path | `data/BioRED/` |
| Download | https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/ |

**Usage**
```bash
for split in Train Dev Test; do
  python -m w2ner_biomedical.converters.biored_to_schema \
    --input  data/BioRED/${split}.BioC.XML \
    --output data/converted/biored/${split,,}.json
done
```

---

## Output format

Both converters produce a JSON array of objects compatible with the
annotation format consumed by `pipeline/step01_ingest.py`:

```json
[
  {
    "PMID": "12345678",
    "articleTitle": "Title of the article.",
    "abstract": "Abstract text goes here ...",
    "label": [
      [[[5, 13]], "Chemical"],
      [[[0, 6], [20, 28]], "Disease"]
    ]
  }
]
```

### `label` field structure

Each element of `label` is `[spans_list, entity_type]`:

- `spans_list` — a list of `[start_char, end_char]` pairs, one per contiguous
  fragment of the entity. For most entities this has exactly one entry.
  For discontinuous entities (BC5CDR Disease annotations) it has two or more entries.
- `entity_type` — the normalised type string matching `specs/label_spec.json`.

### Character offset coordinate system

All character offsets are **fulltext-absolute**, where:

```
fulltext = title + " " + abstract
```

Both corpora use BioC document-absolute offsets starting at 0 for the title,
with a single-space separator between title and abstract (verified empirically).
The converters re-base offsets to this coordinate system; in practice no
correction is needed since the BioC separator is also a single space.

---

## Entity type mapping

| BioRED corpus type | BC5CDR corpus type | `label_spec.json` type |
|---|---|---|
| `ChemicalEntity` | `Chemical` | `Chemical` |
| `DiseaseOrPhenotypicFeature` | `Disease` | `Disease` |
| `GeneOrGeneProduct` | — | `Gene_Or_GeneProduct` |
| `OrganismTaxon` | — | `Species` |
| `CellLine` | — | `CellLine` |
| `SequenceVariant` | — | `VariantOrPolymorphism` |

These mappings are also registered as `aliases` in `specs/label_spec.json`
so that `step03_add_labels.py` can resolve them as a fallback.

Unknown types are **skipped with a WARNING** at conversion time.

---

## Known limitations

- **Relation annotations** are ignored. These converters are NER-only.
- **Normalisation identifiers** (MeSH, NCBIGene, etc.) are not carried
  through; the pipeline operates on span + type only.
- **Cross-sentence entities** (entities whose span crosses a sentence
  boundary assigned by `step02_tokenize.py`) will be dropped by
  `step03_add_labels.py` and counted in the entity alignment drop rate guard.
  These are rare in both corpora but present.
