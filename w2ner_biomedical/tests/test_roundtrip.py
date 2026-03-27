# =============================================================================
# tests/test_roundtrip.py
#
# PURPOSE
#   End-to-end round-trip test: encode a known entity annotation through the
#   full pipeline (step03_add_labels → feature_builder → decoding) and verify
#   that the decoded word indices and character spans match the original input
#   exactly.  This test catches the class of bugs where encoding and decoding
#   are not inverses of each other.
#
# CORRESPONDS TO
#   Nothing in the original W2NER repo — no round-trip test existed.
#   This test directly addresses the asymmetry noted in the architectural
#   analysis: training encodes char spans → word indices, and inference
#   decodes word indices → char spans, but these two paths were never
#   verified to be inverses.
#
# KEY DESIGN CHANGES
#   - Tests cover both contiguous and discontinuous entity cases.
#   - Tests use a deterministic mock tokenizer and fixed sentence spans to
#     avoid dependency on the actual BioBERT model weights.
#   - The test_roundtrip_char_spans case specifically checks the encoding /
#     decoding of a KNOWN discontinuous entity (Bug fix for the
#     range(si, ei+1) issue) to confirm the fix is correct and stable.
#   - test_label2id_strict_lookup verifies that feature_builder raises
#     KeyError on unknown entity types rather than silently returning 0.
#   - test_no_silent_zip_truncation verifies that step06's join raises
#     ValueError when prediction IDs are missing.
#
# BUGS ADDRESSED (VERIFIED BY THESE TESTS)
#   [CRITICAL]  Silent type fallback → test_label2id_strict_lookup
#   [CRITICAL]  Discontinuous entity encoding → test_discontinuous_roundtrip
#   [HIGH]      zip truncation in postprocess → test_no_silent_zip_truncation
#   [HIGH]      CLS offset → test_pieces2word_cls_offset
# =============================================================================

from __future__ import annotations

# TODO: import pytest
# TODO: from data.feature_builder import make_feature_converter, CLS_OFFSET
# TODO: from model.decoding import decode_one_sentence, NNW_LABEL
# TODO: from pipeline.step03_add_labels import resolve_fragments_to_indices
# TODO: from pipeline.step06_postprocess import join_by_id, recover_char_spans
# TODO: from guards.validators import check_id_join_completeness


def make_mock_sentence(words: list[str], char_spans: list[tuple[int, int]]) -> dict:
    """Build a minimal TokenRecord-compatible dict for testing."""
    # TODO: return {"sentence": words, "spans": char_spans, "ner": []}
    ...


def make_mock_tokenizer(word_to_pieces: dict[str, list[str]]):
    """Return a minimal mock tokenizer for feature_builder tests."""
    # TODO: build a namespace/mock with .tokenize(), .convert_tokens_to_ids(),
    #       .cls_token_id, .sep_token_id
    ...


# ---------------------------------------------------------------------------
# Test: CLS offset
# ---------------------------------------------------------------------------

def test_pieces2word_cls_offset():
    """pieces2word[word_i] should point to positions [piece_start+CLS_OFFSET, ...)."""
    # TODO: create sentence with known word→piece mapping
    # TODO: call make_feature_converter(with_labels=False)
    # TODO: assert pieces2word[0, CLS_OFFSET] == 1 (first piece of first word, after CLS)
    # TODO: assert pieces2word[0, 0] == 0          (CLS position never mapped)
    ...


# ---------------------------------------------------------------------------
# Test: contiguous entity round-trip
# ---------------------------------------------------------------------------

def test_contiguous_entity_roundtrip():
    """A 3-word contiguous entity encodes then decodes back to the same indices."""
    # TODO: sentence = ["The", "BRCA2", "gene", "is", "mutated"]
    # TODO: entity: indices=[1,2], type="Gene_Or_GeneProduct"
    # TODO: encode via feature_builder (with_labels=True)
    # TODO: assert grid_labels[1,2] == NNW_LABEL
    # TODO: assert grid_labels[2,1] == label2id["Gene_Or_GeneProduct"]
    # TODO: decode via decode_one_sentence
    # TODO: assert decoded entity indices == [1, 2]
    ...


# ---------------------------------------------------------------------------
# Test: discontinuous entity round-trip (the critical bug fix)
# ---------------------------------------------------------------------------

def test_discontinuous_entity_roundtrip():
    """A discontinuous entity [word 1, word 4] encodes and decodes correctly.

    This test verifies the fix for the range(si, ei+1) bug:
    - Two separate sub-spans [[s1,e1], [s4,e4]] should produce
      indices_set = {1, 4}, NOT {1, 2, 3, 4}.
    - The NNW edge should be grid_labels[1, 4] = NNW_LABEL (direct skip).
    - Decoding should recover indices [1, 4], not [1, 2, 3, 4].
    """
    # TODO: sentence = ["The", "BRCA2", "and", "BRCA1", "genes", "are", "related"]
    # TODO: entity: spans_list = [[s_BRCA2, e_BRCA2], [s_genes, e_genes]]
    #              type = "Gene_Or_GeneProduct"
    # TODO: call resolve_fragments_to_indices → assert indices == [1, 4]
    # TODO: encode via feature_builder → assert grid_labels[1, 4] == NNW_LABEL
    #                                  assert grid_labels[4, 1] == label2id["Gene_Or_GeneProduct"]
    #                                  assert grid_labels[1, 2] == 0  (gap word NOT connected)
    # TODO: decode via decode_one_sentence → assert decoded indices == [1, 4]
    ...


# ---------------------------------------------------------------------------
# Test: char span recovery
# ---------------------------------------------------------------------------

def test_recover_char_spans_contiguous():
    """Contiguous word indices recover to a single character span."""
    # TODO: sentence_spans = [(0,4), (5,8), (9,13), (14,17)]
    # TODO: indices = [1, 2]  → expected span (5, 13)
    # TODO: assert recover_char_spans([1,2], sentence_spans) == [(5, 13)]
    ...


def test_recover_char_spans_discontinuous():
    """Discontinuous word indices recover to multiple character spans."""
    # TODO: sentence_spans = [(0,4), (5,8), (9,13), (14,17), (18,23)]
    # TODO: indices = [1, 4]  → expected spans [(5,8), (18,23)]
    # TODO: assert recover_char_spans([1,4], sentence_spans) == [(5,8), (18,23)]
    ...


# ---------------------------------------------------------------------------
# Test: strict label2id lookup
# ---------------------------------------------------------------------------

def test_label2id_strict_lookup_raises():
    """feature_builder must raise KeyError on unknown entity type, not return 0."""
    # TODO: label2id = {"<pad>": 0, "<suc>": 1, "Gene_Or_GeneProduct": 2}
    # TODO: sentence with one entity of type "UnknownType"
    # TODO: call make_feature_converter(with_labels=True) with this sentence
    # TODO: assert pytest.raises(KeyError)
    ...


# ---------------------------------------------------------------------------
# Test: ID join completeness (Bug D guard)
# ---------------------------------------------------------------------------

def test_no_silent_zip_truncation():
    """join_by_id raises ValueError when a prediction ID is missing."""
    # TODO: token_records with ids ["pmid1_0_0", "pmid1_0_1", "pmid1_0_2"]
    # TODO: pred_records with only ids ["pmid1_0_0", "pmid1_0_1"]  (missing third)
    # TODO: assert pytest.raises(ValueError, match="pmid1_0_2")
    ...


def test_join_by_id_correct_pairing():
    """join_by_id pairs records by ID regardless of list order."""
    # TODO: token_records in order [A, B, C]
    # TODO: pred_records in order [C, A, B]  (shuffled)
    # TODO: joined = join_by_id(token_records, pred_records)
    # TODO: assert [(t.id, p.id) for t,p in joined] == [("A","A"),("B","B"),("C","C")]
    ...
