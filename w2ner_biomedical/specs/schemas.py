# =============================================================================
# specs/schemas.py
#
# PURPOSE
#   Pydantic models that define the explicit data contract at every stage
#   boundary in the pipeline.  Every step reads one schema and writes
#   another; validation is enforced at the reader, not the writer.
#
# CORRESPONDS TO
#   Nothing in the original W2NER repo.  Stage contracts were entirely
#   implicit there — enforced only by shared naming conventions and field
#   access patterns scattered across six separate scripts.
#
# KEY DESIGN CHANGES
#   - Each pipeline stage now has a named input/output schema.
#   - Schemas are versioned (schema_version field) so stale JSONL files
#     produced by an older code version are detectable at load time.
#   - Character offset coordinate system is made explicit: all offsets are
#     fulltext-absolute (title + " " + abstract, NFKC-normalised) so that
#     every downstream stage uses the same coordinate origin without
#     implicit re-basing.
#   - The NER entry schema carries both word-level indices (for W2NER) and
#     sub-span fragments (for discontinuous entity round-trip testing),
#     keeping the two representations in sync rather than deriving one
#     from the other at the last moment.
#   - A StageManifest model is written alongside every JSONL output to
#     enable cache-invalidation and lineage tracing without re-reading the
#     full file.
#
# BUGS ADDRESSED
#   None directly — this file is pure schema definition.  However, making
#   the contracts explicit eliminates the class of "wrong-field-name" bugs
#   (e.g. "tokens" vs "sentence" rename, Bug C) that were invisible in the
#   original pipeline because no validation layer existed.
# =============================================================================

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Stage 1 — Ingest (step01_ingest.py output)
# ---------------------------------------------------------------------------

class IngestRecord(BaseModel):
    """One document as it leaves step01_ingest.py."""

    schema_version: str = "1"
    pmid: str
    title: str
    abstract: str
    # Full concatenated text: title + " " + abstract (or just abstract if no
    # title).  Pre-computed here so all downstream stages share a single
    # canonical fulltext string and the same absolute character offsets.
    fulltext: str

    @model_validator(mode="after")
    def _fulltext_consistent(self) -> "IngestRecord":
        # TODO: assert that self.fulltext == " ".join(filter(None,[title,abstract]))
        ...
        return self


# ---------------------------------------------------------------------------
# Stage 2 — Tokenize (step02_tokenize.py output)
# ---------------------------------------------------------------------------

class TokenRecord(BaseModel):
    """One sentence-chunk as it leaves step02_tokenize.py.

    This is the record type that step03_add_labels.py, step05_predict.py,
    and step06_postprocess.py all read.  Keeping it as one canonical type
    eliminates the step03/step04 split and the associated boundary
    misalignment risk present in the original pipeline.
    """

    schema_version: str = "1"
    # Unique sentence identifier: "{pmid}_{sentence_index}_{chunk_index}"
    id: str
    document_id: str
    pmid: str
    # word-level tokens (spaCy + split_punct, same as original "sentence" key)
    sentence: list[str]
    # parallel list of [start_char, end_char] pairs; fulltext-absolute
    spans: list[tuple[int, int]]
    # populated by step03_add_labels, empty here
    ner: list[NEREntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# NER entry  (shared between training and evaluation)
# ---------------------------------------------------------------------------

class SubSpan(BaseModel):
    """One contiguous fragment of a (possibly discontinuous) entity.

    Mirrors the annotation's spans_list entries after offset resolution.
    Storing fragments explicitly keeps the discontinuous encoding lossless
    through the pipeline; the word-index list in NEREntry is derived from
    these fragments, not the other way round.
    """
    start_char: int   # fulltext-absolute
    end_char: int     # fulltext-absolute


class NEREntry(BaseModel):
    """A single NER annotation attached to a TokenRecord.

    indices  — word-level, 0-based within the sentence chunk.
               For a discontinuous entity these are the UNION of all
               fragment word ranges (no fill-in of gap words).
    fragments — the original sub-span char boundaries, preserved for
                round-trip testing and postprocessing.
    type     — normalised entity type string; must be present in
               LabelSpec.entity_types.
    """
    indices: list[int]
    fragments: list[SubSpan] = Field(default_factory=list)
    type: str


# ---------------------------------------------------------------------------
# Stage 5 — Predict (step05_predict.py output)
# ---------------------------------------------------------------------------

class PredictedEntity(BaseModel):
    text: list[str]        # word tokens
    indices: list[int]     # word-level indices within sentence chunk
    type: str              # decoded entity type label


class PredictRecord(BaseModel):
    """One sentence chunk as it leaves step05_predict.py."""

    schema_version: str = "1"
    id: str                          # matches TokenRecord.id for keyed join
    pmid: str
    sentence: list[str]
    entity: list[PredictedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 6 — Postprocess (step06_postprocess.py output)
# ---------------------------------------------------------------------------

class RefinedEntity(BaseModel):
    text_str: str             # surface string recovered from fulltext
    # sentence_spans: character offsets relative to the sentence chunk's start
    # character (i.e. TokenRecord.spans[0][0]), NOT fulltext-absolute.
    # To convert back to fulltext-absolute: add PostprocessRecord.fulltext_offset
    # to each boundary.  Named explicitly to distinguish from TokenRecord.spans,
    # which uses the fulltext-absolute coordinate system.
    sentence_spans: list[tuple[int, int]]
    type: str                 # majority-voted entity type


class PostprocessRecord(BaseModel):
    """One sentence chunk as it leaves step06_postprocess.py."""

    schema_version: str = "1"
    id: str
    pmid: str
    sentence_str: str
    fulltext_offset: int
    entity: list[RefinedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Label spec  (loaded from specs/label_spec.json)
# ---------------------------------------------------------------------------

class LabelSpec(BaseModel):
    """Vocabulary definition for a trained model.

    This is the single source of truth for entity types.  It is committed
    to source control and edited deliberately — never derived at runtime
    from whatever types happen to appear in a training data file.

    The W2NER protocol reserves two sentinel IDs:
        0  → <pad>  (background / no-entity cell)
        1  → <suc>  (NNW edge — "next-neighbour-word")
    Real entity types start at ID 2.  This is captured explicitly here
    rather than being an invisible property of list ordering.
    """

    schema_version: str = "1"
    # Sentinels must be exactly ["<pad>", "<suc>"] in this order.
    sentinels: list[str] = ["<pad>", "<suc>"]
    # Curated list of real entity types, in the order that defines label IDs.
    entity_types: list[str]
    # Optional human-readable aliases used only for display; do not affect IDs.
    aliases: dict[str, str] = Field(default_factory=dict)
    # Per-model-name filter sets.  Key is a model name suffix (e.g. "_6types").
    model_filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sentinels_fixed(self) -> "LabelSpec":
        # TODO: assert self.sentinels == ["<pad>", "<suc>"]
        ...
        return self

    def label2id(self) -> dict[str, int]:
        """Build the complete label → id mapping including sentinels.

        Sentinel IDs are always first:
            0 → <pad>   (background / no-entity cell)
            1 → <suc>   (NNW "next-neighbour-word" edge)
        Real entity types follow, starting at ID 2.  Order is determined
        by the position in entity_types in label_spec.json — never by
        runtime data.
        """
        return {s: i for i, s in enumerate(self.sentinels + self.entity_types)}

    def id2label(self) -> dict[int, str]:
        """Inverse mapping: label id → label string."""
        return {v: k for k, v in self.label2id().items()}


# ---------------------------------------------------------------------------
# Stage manifest  (written alongside every JSONL output)
# ---------------------------------------------------------------------------

class StageManifest(BaseModel):
    """Sidecar metadata written as {output}.meta.json next to every JSONL.

    Enables cache-invalidation (compare input_hash to detect stale outputs)
    and lineage tracing (compare record_count across stages to find where
    entities were silently dropped).
    """

    schema_version: str = "1"
    stage: str                   # e.g. "step01_ingest"
    input_files: list[str]       # basenames of files consumed
    input_hash: str              # sha256 of concatenated input content
    record_count: int            # number of records written
    timestamp: str               # ISO-8601
    code_version: str = "unknown"
