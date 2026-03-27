# =============================================================================
# converters/_bioc_utils.py
#
# Shared BioC parsing utilities used by both bc5cdr_to_schema.py and
# biored_to_schema.py.  Both corpora use the same BioC format conventions:
# document-absolute offsets, title passage at offset=0, single-space
# separator between title and abstract.
# =============================================================================

from __future__ import annotations

import logging

LOGGER: logging.Logger = logging.getLogger(__name__)


def validate_bioc_offsets(document) -> None:
    """Assert that the title passage starts at offset 0.

    Both BC5CDR and BioRED use document-absolute offsets starting from 0
    for the title passage.  If this assumption fails, all annotation offsets
    in this document would be misaligned, so we raise rather than silently
    corrupting the output.
    """
    for passage in document.passages:
        if passage.infons.get("type") == "title":
            if passage.offset != 0:
                raise ValueError(
                    f"Document {document.id}: expected title passage at offset 0, "
                    f"got {passage.offset}. Cannot safely re-base offsets."
                )
            return
    raise ValueError(f"Document {document.id}: no title passage found.")


def extract_passages(document) -> tuple[str, str, int]:
    """Extract title text, abstract text, and the BioC abstract offset.

    Returns (title_text, abstract_text, abstract_bioc_offset).
    abstract_bioc_offset is the document-absolute character position where
    the abstract passage starts in the BioC document.

    Raises ValueError if either passage is missing.
    """
    title_text: str = ""
    abstract_text: str = ""
    abstract_bioc_offset: int = 0

    for passage in document.passages:
        ptype = passage.infons.get("type", "")
        if ptype == "title":
            title_text = passage.text or ""
        elif ptype == "abstract":
            abstract_text = passage.text or ""
            abstract_bioc_offset = passage.offset

    if not title_text:
        raise ValueError(f"Document {document.id}: no title passage found.")
    if not abstract_text:
        raise ValueError(f"Document {document.id}: no abstract passage found.")

    sep_width = abstract_bioc_offset - len(title_text)
    if sep_width != 1:
        LOGGER.warning(
            "Document %s: BioC title-abstract separator is %d character(s) wide "
            "(expected 1). Offsets will be re-based to a single-space separator.",
            document.id, sep_width,
        )

    return title_text, abstract_text, abstract_bioc_offset


def bioc_offset_to_fulltext_offset(
    bioc_offset: int,
    title_len: int,
    abstract_bioc_offset: int,
) -> int:
    """Convert a BioC document-absolute offset to our fulltext-absolute offset.

    Our fulltext = title + " " + abstract (exactly one space separator).
    BioC fulltext = title + <sep> + abstract, where <sep> may differ.

    If bioc_offset is in the title (< abstract_bioc_offset): same in both.
    If bioc_offset is in the abstract: re-base to title_len + 1 + position
    within abstract, correcting for any separator width difference.
    """
    if bioc_offset < abstract_bioc_offset:
        return bioc_offset
    pos_within_abstract = bioc_offset - abstract_bioc_offset
    return (title_len + 1) + pos_within_abstract
