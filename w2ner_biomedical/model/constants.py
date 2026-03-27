# =============================================================================
# model/constants.py
#
# PURPOSE
#   Single source of truth for numeric constants shared across model/,
#   data/, and pipeline/.  Centralising them here prevents the silent
#   drift that occurs when the same magic number is repeated in multiple
#   files (e.g. CLS_OFFSET was independently defined in both
#   data/feature_builder.py and pipeline/step02_tokenize.py).
# =============================================================================

# ---------------------------------------------------------------------------
# Label IDs
# Must match the sentinel order defined in LabelSpec (specs/label_spec.json):
#   index 0 → "<pad>"   background / no-entity cell
#   index 1 → "<suc>"   NNW "next-neighbour-word" edge
# Real entity type IDs start at 2.
# ---------------------------------------------------------------------------

NNW_LABEL: int = 1
"""Label ID for the NNW (next-neighbour-word) edge in the W2NER grid."""

# ---------------------------------------------------------------------------
# Subword tokenizer offset
# [CLS] occupies position 0 in every BERT input sequence, so all subword
# piece positions are shifted right by 1 when building the pieces2word
# alignment matrix.
# ---------------------------------------------------------------------------

CLS_OFFSET: int = 1
"""Offset applied to subword piece indices to account for the leading [CLS] token."""

# ---------------------------------------------------------------------------
# Distance embedding
# dis2idx maps positive distances to buckets 0–9 and negative distances to
# buckets 9–18 (mirrored).  The diagonal (i == j, distance = 0) is assigned
# this sentinel so it is distinct from all off-diagonal buckets.
# ---------------------------------------------------------------------------

DIST_DIAGONAL: int = 19
"""Distance-embedding index used for the grid diagonal (token paired with itself)."""
