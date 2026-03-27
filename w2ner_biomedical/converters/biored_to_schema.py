# =============================================================================
# converters/biored_to_schema.py
#
# PURPOSE
#   Convert a BioRED BioC-XML corpus file into the IngestRecord JSON format
#   expected by pipeline/step01_ingest.py.  Each BioRED document becomes
#   one IngestRecord with its NER annotations expressed as a `label` list of
#   [spans_list, entity_type] pairs.
#
# INPUT FORMAT — BioRED (BioC XML)
#   BioRED is distributed as a BioC-XML file (e.g. BioRED.xml) with the
#   following document structure:
#
#     <collection>
#       <document>
#         <id>12345678</id>            ← PMID
#         <passage>
#           <infons><infon key="type">title</infon></infons>
#           <offset>0</offset>
#           <text>Title text here.</text>
#           <annotation id="T1">
#             <infons>
#               <infon key="type">Chemical</infon>
#               <infon key="identifier">MESH:D000068878</infon>
#             </infons>
#             <location offset="5" length="8"/>
#             <text>imatinib</text>
#           </annotation>
#         </passage>
#         <passage>
#           <infons><infon key="type">abstract</infon></infons>
#           <offset>17</offset>        ← document-absolute offset of abstract start
#           <text>Abstract text...</text>
#           <annotation id="T2">
#             <location offset="25" length="10"/>  ← document-absolute
#             <location offset="80" length="6"/>   ← second location → discontinuous entity
#           </annotation>
#         </passage>
#       </document>
#     </collection>
#
#   CRITICAL: BioC annotation offsets are DOCUMENT-ABSOLUTE, not passage-relative.
#   An annotation with offset=25 means character 25 of the full document, regardless
#   of which passage it appears in.
#
#   DISCONTINUOUS ENTITIES: BioRED annotations may carry multiple <location>
#   elements.  Each location becomes one [start, end] entry in spans_list,
#   producing a multi-span annotation that step03_add_labels.py handles correctly
#   via its fragment-union logic.
#
# OUTPUT FORMAT — IngestRecord-compatible JSON array
#   [
#     {
#       "PMID": "12345678",
#       "articleTitle": "Title text here.",
#       "abstract": "Abstract text...",
#       "label": [
#         [[[5, 13]], "Chemical"],
#         [[[25, 35], [80, 86]], "Gene_Or_GeneProduct"]   ← discontinuous
#       ]
#     },
#     ...
#   ]
#
#   Character offsets in spans_list are FULLTEXT-ABSOLUTE, where fulltext =
#   title + " " + abstract (mirroring IngestRecord.fulltext).  Because BioC
#   offsets are document-absolute from offset=0, and the BioC document's
#   fulltext is identical to our fulltext construction, no re-basing is needed
#   IF the BioC document uses offset=0 for the title passage start (which
#   BioRED does).  This assumption is validated at parse time.
#
# ENTITY TYPE MAPPING
#   BioRED type       →  label_spec.json type
#   ─────────────────────────────────────────
#   Chemical          →  Chemical
#   Disease           →  Disease
#   Gene              →  Gene_Or_GeneProduct
#   Variant           →  VariantOrPolymorphism
#   Species           →  Species
#   CellLine          →  CellLine
#   DNAMutation       →  DNA_Mutation      (alias in label_spec.json)
#
#   Any type not in this mapping is logged as a WARNING and the annotation
#   is SKIPPED (not converted).  This surfaces unknown types at conversion
#   time rather than at step03_add_labels time.
#
# KEY DESIGN DECISIONS
#   - Relation annotations in BioRED (the R lines) are ignored — this
#     converter is for NER only.  A separate relation extraction pipeline
#     would need a different converter.
#   - Normalisation identifiers (MeSH, NCBIGene, etc.) are not carried
#     through; the NER pipeline operates on span + type only.
#   - The `bioc` Python library is used for parsing rather than raw XML
#     manipulation; it handles the BioC schema nuances (collection offsets,
#     passage offsets) correctly.
#
# USAGE
#   python -m converters.biored_to_schema \
#       --input  /path/to/BioRED.xml \
#       --output /path/to/output/BioRED_converted.json
#
#   The output file is a JSON array ready to be placed in the step01_ingest
#   INPUT_DIR.
# =============================================================================

from __future__ import annotations

# TODO: import json, logging
# TODO: from pathlib import Path
# TODO: import bioc                   # pip install bioc

LOGGER = ...  # TODO: logging.getLogger(__name__)

# BioRED entity type → label_spec.json entity type
BIORED_TYPE_MAP: dict[str, str] = {
    "Chemical":         "Chemical",
    "Disease":          "Disease",
    "Gene":             "Gene_Or_GeneProduct",
    "Variant":          "VariantOrPolymorphism",
    "Species":          "Species",
    "CellLine":         "CellLine",
    "DNAMutation":      "DNA_Mutation",
    "SequenceVariant":  "VariantOrPolymorphism",
}


def validate_bioc_offsets(document) -> None:
    """Assert that the title passage starts at offset 0.

    BioRED documents use document-absolute offsets starting from 0 for the
    title passage.  If this assumption fails, our fulltext offset re-basing
    would be incorrect, so we fail loudly rather than silently misaligning
    all annotations in this document.
    """
    # TODO: title_passage = first passage with infons["type"] == "title"
    # TODO: if title_passage.offset != 0:
    #         raise ValueError(f"Document {document.id}: expected title offset 0, "
    #                          f"got {title_passage.offset}. Cannot safely re-base offsets.")
    ...


def extract_passages(document) -> tuple[str, str, int]:
    """Extract title text, abstract text, and the BioC abstract offset.

    Returns (title, abstract, abstract_bioc_offset).
    abstract_bioc_offset is the document-absolute character position where
    the abstract passage starts in the BioC document.
    """
    # TODO: find passage with type "title"  → title_text
    # TODO: find passage with type "abstract" → abstract_text, abstract_offset
    # TODO: return (title_text, abstract_text, abstract_offset)
    ...


def bioc_offset_to_fulltext_offset(
    bioc_offset: int,
    title_len: int,
    abstract_bioc_offset: int,
) -> int:
    """Convert a BioC document-absolute offset to our fulltext-absolute offset.

    Our fulltext = title + " " + abstract.
    BioC document = title + (possible whitespace) + abstract.

    The separator between title and abstract in BioC is captured by
    abstract_bioc_offset - title_len.  Our fulltext always uses exactly
    one space as separator.

    This function remaps BioC offsets into our coordinate system.
    If the BioC separator is something other than a single space, there is
    a systematic offset difference that this function corrects.
    """
    # TODO: if bioc_offset < abstract_bioc_offset:
    #         # offset is in the title portion — same in both systems
    #         return bioc_offset
    # TODO: else:
    #         # offset is in the abstract portion
    #         # BioC abstract starts at abstract_bioc_offset
    #         # Our abstract starts at title_len + 1 (title + space)
    #         pos_within_abstract = bioc_offset - abstract_bioc_offset
    #         return (title_len + 1) + pos_within_abstract
    ...


def convert_annotation(
    ann,
    title_len: int,
    abstract_bioc_offset: int,
) -> tuple[list[list[int]], str] | None:
    """Convert one BioC annotation to a (spans_list, type_str) pair.

    Returns None if the annotation type is not in BIORED_TYPE_MAP (with
    a WARNING logged) or if any location produces an invalid span.

    For annotations with multiple locations (discontinuous entities), each
    location becomes a separate [start, end] entry in spans_list.
    """
    # TODO: raw_type = ann.infons.get("type", "")
    # TODO: mapped_type = BIORED_TYPE_MAP.get(raw_type)
    # TODO: if mapped_type is None:
    #         LOGGER.warning("Unknown BioRED type %r in annotation %s — skipping", raw_type, ann.id)
    #         return None
    # TODO: spans_list = []
    # TODO: for location in ann.locations:
    #         start = bioc_offset_to_fulltext_offset(location.offset, title_len, abstract_bioc_offset)
    #         end   = bioc_offset_to_fulltext_offset(location.offset + location.length,
    #                                                title_len, abstract_bioc_offset)
    #         spans_list.append([start, end])
    # TODO: if not spans_list: return None
    # TODO: return (spans_list, mapped_type)
    ...


def convert_document(document) -> dict | None:
    """Convert one BioC document to an IngestRecord-compatible dict.

    Returns None if the document cannot be converted (e.g. missing passages).
    """
    # TODO: validate_bioc_offsets(document)
    # TODO: title, abstract, abstract_bioc_offset = extract_passages(document)
    # TODO: title_len = len(title)
    # TODO: labels = []
    # TODO: for passage in document.passages:
    #         for ann in passage.annotations:
    #             result = convert_annotation(ann, title_len, abstract_bioc_offset)
    #             if result: labels.append(list(result))
    # TODO: return {"PMID": document.id, "articleTitle": title, "abstract": abstract, "label": labels}
    ...


def convert_biored_file(input_path: ..., output_path: ...) -> int:  # type: ignore[name-defined]
    """Convert a BioRED BioC-XML file and write the output JSON array.

    Returns the number of documents converted.
    """
    # TODO: collection = bioc.load(str(input_path))
    # TODO: records = [convert_document(doc) for doc in collection.documents]
    # TODO: records = [r for r in records if r is not None]
    # TODO: write JSON array to output_path
    # TODO: LOGGER.info("Converted %d / %d BioRED documents", len(records), len(collection.documents))
    # TODO: return len(records)
    ...


def main() -> None:
    # TODO: argparse: --input, --output
    # TODO: call convert_biored_file(args.input, args.output)
    ...


if __name__ == "__main__":
    main()
