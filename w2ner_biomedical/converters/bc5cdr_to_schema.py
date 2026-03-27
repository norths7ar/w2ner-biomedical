# =============================================================================
# converters/bc5cdr_to_schema.py
#
# PURPOSE
#   Convert a BC5CDR BioC corpus file (XML) into the annotation JSON format
#   consumed by pipeline/step01_ingest.py.
#
# INPUT FORMAT — BC5CDR (BioC XML)
#   Files: CDR_TrainingSet.BioC.xml, CDR_DevelopmentSet.BioC.xml,
#          CDR_TestSet.BioC.xml
#   Located under CDR_Data/CDR.Corpus.v010516/ in the standard distribution.
#   Standard BioC structure: document-absolute annotation offsets, title at
#   offset=0, single-space separator between title and abstract.
#
#   Entity types in this corpus:
#     Chemical  → Chemical
#     Disease   → Disease
#
#   DISCONTINUOUS ENTITIES: BC5CDR contains 73 multi-location annotations
#   in the training set (Disease entities with non-contiguous fragments,
#   e.g. "Ocular ... toxicity").  Each location becomes a separate [start, end]
#   entry in spans_list.  These are handled identically to BioRED multi-location
#   annotations by step03_add_labels.py via its fragment-union logic.
#
# OUTPUT FORMAT
#   [
#     {
#       "PMID": "227508",
#       "articleTitle": "Naloxone reverses ...",
#       "abstract": "In unanesthetized ...",
#       "label": [
#         [[[0, 8]], "Chemical"],
#         [[[49, 58]], "Chemical"],
#         [[[0, 6], [20, 28]], "Disease"]   ← discontinuous
#       ]
#     },
#     ...
#   ]
#
# USAGE
#   for split in Training Development Test; do
#     python -m w2ner_biomedical.converters.bc5cdr_to_schema \
#       --input  data/CDR_Data/CDR.Corpus.v010516/CDR_${split}Set.BioC.xml \
#       --output data/raw/bc5cdr_converted/${split,,}.json
#   done
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import bioc

from myutils import get_logger

from ._bioc_utils import validate_bioc_offsets, extract_passages, bioc_offset_to_fulltext_offset

LOGGER: logging.Logger = logging.getLogger(__name__)

# BC5CDR corpus type → label_spec.json canonical type (direct 1:1 mapping)
BC5CDR_TYPE_MAP: dict[str, str] = {
    "Chemical": "Chemical",
    "Disease":  "Disease",
}


def convert_annotation(
    ann,
    title_len: int,
    abstract_bioc_offset: int,
) -> tuple[list[list[int]], str] | None:
    """Convert one BioC annotation to a (spans_list, type_str) pair.

    Returns None if the annotation type is not in BC5CDR_TYPE_MAP (logged
    as WARNING) or if any location produces a degenerate span (start >= end).
    Multi-location annotations (discontinuous entities) produce multiple
    [start, end] entries in spans_list.
    """
    raw_type = ann.infons.get("type", "")
    mapped_type = BC5CDR_TYPE_MAP.get(raw_type)
    if mapped_type is None:
        LOGGER.warning("Unknown BC5CDR type %r in annotation %s — skipping.", raw_type, ann.id)
        return None

    spans_list: list[list[int]] = []
    for loc in ann.locations:
        start = bioc_offset_to_fulltext_offset(loc.offset, title_len, abstract_bioc_offset)
        end = bioc_offset_to_fulltext_offset(loc.offset + loc.length, title_len, abstract_bioc_offset)
        if start >= end:
            LOGGER.warning(
                "Annotation %s: degenerate span [%d, %d] after offset re-base — skipping annotation.",
                ann.id, start, end,
            )
            return None
        spans_list.append([start, end])

    if not spans_list:
        return None
    return spans_list, mapped_type


def convert_document(document) -> dict | None:
    """Convert one BioC document to an annotation-compatible dict.

    Returns None if the document cannot be converted (missing passages,
    bad offsets).  Logs a WARNING and continues so a single bad document
    does not abort the whole file.
    """
    try:
        validate_bioc_offsets(document)
        title, abstract, abstract_bioc_offset = extract_passages(document)
    except ValueError as exc:
        LOGGER.warning("Skipping document %s: %s", document.id, exc)
        return None

    title_len = len(title)
    labels: list[list] = []

    for passage in document.passages:
        for ann in passage.annotations:
            result = convert_annotation(ann, title_len, abstract_bioc_offset)
            if result is not None:
                spans_list, mapped_type = result
                labels.append([spans_list, mapped_type])

    return {
        "PMID": document.id,
        "articleTitle": title,
        "abstract": abstract,
        "label": labels,
    }


def convert_bc5cdr_file(input_path: Path, output_path: Path) -> int:
    """Convert a BC5CDR BioC-XML file and write the output JSON array.

    Returns the number of documents successfully converted.
    """
    LOGGER.info("Loading %s ...", input_path)
    with open(input_path, encoding="utf-8") as f:
        collection = bioc.load(f)

    records: list[dict] = []
    for document in collection.documents:
        rec = convert_document(document)
        if rec is not None:
            records.append(rec)

    LOGGER.info(
        "Converted %d / %d documents from %s.",
        len(records), len(collection.documents), input_path.name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    LOGGER.info("Written to %s.", output_path)
    return len(records)


def main() -> None:
    global LOGGER

    parser = argparse.ArgumentParser(
        description="Convert a BC5CDR BioC-XML file to annotation JSON."
    )
    parser.add_argument("--input", required=True, help="Path to BC5CDR BioC-XML file (e.g. CDR_TrainingSet.BioC.xml).")
    parser.add_argument("--output", required=True, help="Path to write the output JSON array.")
    args = parser.parse_args()

    output_path = Path(args.output)
    LOGGER = get_logger("bc5cdr_to_schema", log_dir=output_path.parent / "logs")

    convert_bc5cdr_file(Path(args.input), output_path)


if __name__ == "__main__":
    main()
