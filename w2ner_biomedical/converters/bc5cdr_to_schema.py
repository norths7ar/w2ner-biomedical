# =============================================================================
# converters/bc5cdr_to_schema.py
#
# PURPOSE
#   Convert a BC5CDR BioC corpus file (XML or JSON) into the IngestRecord
#   JSON format expected by pipeline/step01_ingest.py.
#
# INPUT FORMAT — BC5CDR (BioC XML or BioC JSON)
#   BC5CDR is distributed in BioC format via the BioCreative V shared task.
#   The standard split files are:
#     CDR_TrainingSet.BioC.xml
#     CDR_DevelopmentSet.BioC.xml
#     CDR_TestSet.BioC.xml
#
#   Document structure (same BioC schema as BioRED):
#
#     <collection>
#       <document>
#         <id>2339463</id>
#         <passage>
#           <infons><infon key="type">title</infon></infons>
#           <offset>0</offset>
#           <text>Sulindac-induced ...</text>
#           <annotation id="T1">
#             <infons>
#               <infon key="type">Chemical</infon>
#             </infons>
#             <location offset="0" length="8"/>
#             <text>Sulindac</text>
#           </annotation>
#         </passage>
#         <passage>
#           <infons><infon key="type">abstract</infon></infons>
#           <offset>33</offset>
#           <text>...</text>
#           <annotation .../>
#         </passage>
#       </document>
#     </collection>
#
#   ENTITY TYPES: BC5CDR contains only two entity types:
#     "Chemical" → Chemical
#     "Disease"  → Disease
#   Both map directly to label_spec.json with no aliasing needed.
#
#   OFFSETS: Same convention as BioRED — annotation offsets are
#   DOCUMENT-ABSOLUTE (relative to the start of the whole document,
#   not the passage).  The same bioc_offset_to_fulltext_offset()
#   logic applies.
#
#   NO DISCONTINUOUS ENTITIES: BC5CDR annotations always have exactly one
#   <location> element per annotation.  The converter validates this
#   assumption and raises if a multi-location annotation is encountered
#   (to catch format changes across BC5CDR versions).
#
# OUTPUT FORMAT — same IngestRecord-compatible JSON array as biored_to_schema.py
#   [
#     {
#       "PMID": "2339463",
#       "articleTitle": "Sulindac-induced ...",
#       "abstract": "...",
#       "label": [
#         [[[0, 8]], "Chemical"],
#         [[[45, 60]], "Disease"]
#       ]
#     },
#     ...
#   ]
#
# DIFFERENCES FROM biored_to_schema.py
#   - Only two entity types (no mapping table needed, direct pass-through).
#   - No discontinuous entities expected; converter validates this.
#   - BC5CDR has a known issue in some versions where the abstract passage
#     offset includes the title-terminating newline (`\n`) rather than a
#     space.  The converter detects and handles this: if the separator
#     character between title end and abstract start is `\n` rather than
#     ` `, it is treated as a single character (same width) so no offset
#     delta is introduced.  A WARNING is logged when this case is detected.
#   - Some BC5CDR annotations span across the passage boundary (an
#     annotation starts in the title and ends in the abstract).  These are
#     extremely rare but present in the training set.  They are converted
#     with a single span covering both passages and logged as INFO.
#
# USAGE
#   python -m converters.bc5cdr_to_schema \
#       --input  /path/to/CDR_TrainingSet.BioC.xml \
#       --output /path/to/output/bc5cdr_train_converted.json
#
#   To convert all three splits:
#   for split in Training Development Test; do
#     python -m converters.bc5cdr_to_schema \
#       --input  /path/CDR_${split}Set.BioC.xml \
#       --output /path/bc5cdr_${split,,}_converted.json
#   done
# =============================================================================

from __future__ import annotations

# TODO: import json, logging
# TODO: from pathlib import Path
# TODO: import bioc

LOGGER = ...  # TODO: logging.getLogger(__name__)

# BC5CDR entity types — both map directly, no aliasing required.
BC5CDR_TYPE_MAP: dict[str, str] = {
    "Chemical": "Chemical",
    "Disease":  "Disease",
}


def validate_bioc_offsets(document) -> None:
    """Assert that the title passage starts at offset 0.

    Same validation as biored_to_schema.py — BioC documents in BC5CDR also
    use document-absolute offsets starting from 0 for the title.
    """
    # TODO: identical logic to biored_to_schema.validate_bioc_offsets
    ...


def detect_title_abstract_separator(document, title_text: str, abstract_bioc_offset: int) -> str:
    """Detect the character(s) between the title and abstract in the BioC document.

    BC5CDR sometimes uses '\n' as the separator; our fulltext always uses ' '.
    Returns the detected separator string (1 or more characters).
    Logs a WARNING if the separator is not a single space.
    """
    # TODO: separator_len = abstract_bioc_offset - len(title_text)
    # TODO: if separator_len == 1 and separator == '\n':
    #         LOGGER.warning("Document %s: BioC uses '\\n' as title-abstract separator "
    #                        "(offset %d). Fulltext will use ' ' — offsets remain aligned "
    #                        "because both separators are 1 character wide.", ...)
    # TODO: elif separator_len != 1:
    #         raise ValueError(f"Unexpected separator length {separator_len} in document ...")
    # TODO: return separator
    ...


def convert_annotation(
    ann,
    title_len: int,
    abstract_bioc_offset: int,
) -> tuple[list[list[int]], str] | None:
    """Convert one BioC annotation to a (spans_list, type_str) pair.

    BC5CDR-specific: validates that each annotation has exactly one location.
    Multi-location annotations indicate a format change and are rejected.
    """
    # TODO: if len(ann.locations) != 1:
    #         raise ValueError(f"BC5CDR annotation {ann.id} has {len(ann.locations)} locations "
    #                          f"(expected 1). BC5CDR does not use discontinuous entities. "
    #                          f"Check for format version differences.")
    # TODO: raw_type = ann.infons.get("type", "")
    # TODO: mapped_type = BC5CDR_TYPE_MAP.get(raw_type)
    # TODO: if mapped_type is None:
    #         LOGGER.warning("Unknown BC5CDR type %r — skipping", raw_type)
    #         return None
    # TODO: location = ann.locations[0]
    # TODO: start = bioc_offset_to_fulltext_offset(location.offset, title_len, abstract_bioc_offset)
    # TODO: end   = bioc_offset_to_fulltext_offset(location.offset + location.length,
    #                                              title_len, abstract_bioc_offset)
    # TODO: return ([[start, end]], mapped_type)
    ...


def bioc_offset_to_fulltext_offset(
    bioc_offset: int,
    title_len: int,
    abstract_bioc_offset: int,
) -> int:
    """Convert a BioC document-absolute offset to our fulltext-absolute offset.

    Identical logic to biored_to_schema.bioc_offset_to_fulltext_offset.
    Duplicated here rather than imported to keep converters self-contained
    and independently deployable.

    TODO: once both converters are stable, extract shared BioC utilities to
    converters/_bioc_utils.py.
    """
    # TODO: if bioc_offset < abstract_bioc_offset:
    #         return bioc_offset
    # TODO: else:
    #         pos_within_abstract = bioc_offset - abstract_bioc_offset
    #         return (title_len + 1) + pos_within_abstract
    ...


def convert_document(document) -> dict | None:
    """Convert one BioC document to an IngestRecord-compatible dict."""
    # TODO: validate_bioc_offsets(document)
    # TODO: extract title, abstract, abstract_bioc_offset
    # TODO: detect_title_abstract_separator (log if non-space)
    # TODO: convert all annotations across both passages
    # TODO: return {"PMID": ..., "articleTitle": ..., "abstract": ..., "label": [...]}
    ...


def convert_bc5cdr_file(input_path: ..., output_path: ...) -> int:  # type: ignore[name-defined]
    """Convert a BC5CDR BioC file and write the output JSON array.

    Returns the number of documents converted.
    Supports both XML (.xml) and JSON (.json) BioC formats.
    """
    # TODO: use bioc.load() for XML or bioc.loads() for JSON
    # TODO: convert each document
    # TODO: write JSON array to output_path
    # TODO: return count
    ...


def main() -> None:
    # TODO: argparse: --input, --output
    # TODO: call convert_bc5cdr_file(args.input, args.output)
    ...


if __name__ == "__main__":
    main()
