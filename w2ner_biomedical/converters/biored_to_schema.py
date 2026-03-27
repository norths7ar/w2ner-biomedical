# =============================================================================
# converters/biored_to_schema.py
#
# PURPOSE
#   Convert a BioRED BioC-XML corpus file into the annotation JSON format
#   consumed by pipeline/step01_ingest.py.  Each BioRED document becomes
#   one record with its NER annotations expressed as a `label` list of
#   [spans_list, entity_type] pairs.
#
# INPUT FORMAT — BioRED (BioC XML)
#   Files: Train.BioC.XML, Dev.BioC.XML, Test.BioC.XML
#   Standard BioC structure: <collection> → <document> → <passage> → <annotation>
#   Annotation offsets are DOCUMENT-ABSOLUTE (from start of title at offset=0).
#
#   Entity types in this corpus:
#     ChemicalEntity           → Chemical
#     DiseaseOrPhenotypicFeature → Disease
#     GeneOrGeneProduct        → Gene_Or_GeneProduct
#     OrganismTaxon            → Species
#     CellLine                 → CellLine
#     SequenceVariant          → VariantOrPolymorphism
#
#   Relation annotations (R-lines) are ignored; this converter is NER-only.
#   Normalisation identifiers (MeSH, NCBIGene, etc.) are not carried through.
#
#   NOTE: BioRED has NO multi-location (discontinuous) annotations in any
#   split (verified across Train/Dev/Test).  Each annotation has exactly one
#   <location> element.  The converter still handles multiple locations
#   correctly for forward compatibility.
#
# OUTPUT FORMAT
#   [
#     {
#       "PMID": "12345678",
#       "articleTitle": "Title text.",
#       "abstract": "Abstract text...",
#       "label": [
#         [[[5, 13]], "Chemical"],
#         [[[25, 35]], "Gene_Or_GeneProduct"]
#       ]
#     },
#     ...
#   ]
#
# USAGE
#   for split in Train Dev Test; do
#     python -m w2ner_biomedical.converters.biored_to_schema \
#       --input  data/BioRED/${split}.BioC.XML \
#       --output data/raw/biored_converted/${split,,}.json
#   done
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import bioc

from ._bioc_utils import validate_bioc_offsets, extract_passages, bioc_offset_to_fulltext_offset

LOGGER: logging.Logger = logging.getLogger(__name__)

# BioRED corpus type → label_spec.json canonical type
BIORED_TYPE_MAP: dict[str, str] = {
    "ChemicalEntity":           "Chemical",
    "DiseaseOrPhenotypicFeature": "Disease",
    "GeneOrGeneProduct":        "Gene_Or_GeneProduct",
    "OrganismTaxon":            "Species",
    "CellLine":                 "CellLine",
    "SequenceVariant":          "VariantOrPolymorphism",
}


def convert_annotation(
    ann,
    title_len: int,
    abstract_bioc_offset: int,
) -> tuple[list[list[int]], str] | None:
    """Convert one BioC annotation to a (spans_list, type_str) pair.

    Returns None if the annotation type is not in BIORED_TYPE_MAP (logged
    as WARNING) or if any location produces an invalid span (start >= end).
    Each location becomes a [start, end] entry in spans_list.
    """
    raw_type = ann.infons.get("type", "")
    mapped_type = BIORED_TYPE_MAP.get(raw_type)
    if mapped_type is None:
        LOGGER.warning("Unknown BioRED type %r in annotation %s — skipping.", raw_type, ann.id)
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


def convert_biored_file(input_path: Path, output_path: Path) -> int:
    """Convert a BioRED BioC-XML file and write the output JSON array.

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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert a BioRED BioC-XML file to annotation JSON."
    )
    parser.add_argument("--input", required=True, help="Path to BioRED BioC-XML file (e.g. Train.BioC.XML).")
    parser.add_argument("--output", required=True, help="Path to write the output JSON array.")
    args = parser.parse_args()

    convert_biored_file(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
