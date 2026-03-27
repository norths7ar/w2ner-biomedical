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
#   - The test_discontinuous_entity_roundtrip case specifically checks the
#     encoding/decoding of a known discontinuous entity (Bug fix for the
#     range(si, ei+1) issue) to confirm the fix is correct and stable.
#   - test_label2id_strict_lookup_raises verifies that feature_builder raises
#     KeyError on unknown entity types rather than silently returning 0.
#   - test_no_silent_zip_truncation verifies that step06's join raises
#     ValueError when prediction IDs are missing.
#
# BUGS ADDRESSED (VERIFIED BY THESE TESTS)
#   [CRITICAL]  Silent type fallback → test_label2id_strict_lookup_raises
#   [CRITICAL]  Discontinuous entity encoding → test_discontinuous_entity_roundtrip
#   [HIGH]      zip truncation in postprocess → test_no_silent_zip_truncation
#   [HIGH]      CLS offset → test_pieces2word_cls_offset
# =============================================================================

from __future__ import annotations

import pytest

from w2ner_biomedical.data.feature_builder import build_dis2idx, make_feature_converter
from w2ner_biomedical.model.constants import CLS_OFFSET, NNW_LABEL
from w2ner_biomedical.model.decoding import decode_one_sentence
from w2ner_biomedical.pipeline.step03_add_labels import resolve_fragments_to_indices
from w2ner_biomedical.pipeline.step06_postprocess import join_by_id, recover_char_spans
from w2ner_biomedical.guards.validators import check_id_join_completeness
from w2ner_biomedical.specs.schemas import (
    TokenRecord, PredictRecord, SubSpan,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Deterministic tokenizer: each word maps to exactly one piece (word.lower()).

    CLS id=0, SEP id=1; piece ids start at 2 in insertion order.
    No external model dependency.
    """

    cls_token_id = 0
    sep_token_id = 1

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 2

    def _get_id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = self._next_id
            self._next_id += 1
        return self._vocab[piece]

    def tokenize(self, word: str) -> list[str]:
        return [word.lower()]

    def convert_tokens_to_ids(self, pieces: list[str]) -> list[int]:
        return [self._get_id(p) for p in pieces]


def make_converter(label2id: dict[str, int]):
    return make_feature_converter(MockTokenizer(), build_dis2idx(), label2id)


def _token_rec(id_: str, pmid: str = "1") -> TokenRecord:
    return TokenRecord(id=id_, document_id="doc1", pmid=pmid,
                       sentence=["word"], spans=[(0, 4)])


def _pred_rec(id_: str, pmid: str = "1") -> PredictRecord:
    return PredictRecord(id=id_, pmid=pmid, sentence=["word"])


# ---------------------------------------------------------------------------
# Test: CLS offset
# ---------------------------------------------------------------------------

def test_pieces2word_cls_offset():
    """pieces2word[word_i] should point to piece_start+CLS_OFFSET, not piece_start.

    With CLS_OFFSET=1, word 0's single piece sits at bert_input position 1
    (position 0 is reserved for [CLS]).  Verifies the constant is applied
    correctly in feature_builder.
    """
    label2id = {"<pad>": 0, "<suc>": 1, "Chemical": 2}
    convert = make_converter(label2id)
    out = convert({"sentence": ["Hello", "world"], "ner": []}, with_labels=False)
    p2w = out["pieces2word"]   # [2, seq_len]

    # Position 0 in the BERT sequence is [CLS] — must never be mapped to any word.
    assert not p2w[0, 0].item(), "Word 0 must not map to CLS position (index 0)."
    assert not p2w[1, 0].item(), "Word 1 must not map to CLS position (index 0)."

    # Each word has exactly one piece; with CLS_OFFSET=1:
    #   word 0 → bert_input position 0 + CLS_OFFSET = 1
    #   word 1 → bert_input position 1 + CLS_OFFSET = 2
    assert p2w[0, CLS_OFFSET].item(), "Word 0 must map to position CLS_OFFSET (1)."
    assert p2w[1, CLS_OFFSET + 1].item(), "Word 1 must map to position CLS_OFFSET+1 (2)."


# ---------------------------------------------------------------------------
# Test: contiguous entity round-trip
# ---------------------------------------------------------------------------

def test_contiguous_entity_roundtrip():
    """A 2-word contiguous entity encodes then decodes back to the same indices."""
    label2id = {"<pad>": 0, "<suc>": 1, "Gene_Or_GeneProduct": 2}
    id2label = {v: k for k, v in label2id.items()}
    convert = make_converter(label2id)

    # sentence: ["The", "BRCA2", "gene", "is", "mutated"]
    # entity:   indices=[1, 2], type="Gene_Or_GeneProduct"
    instance = {
        "sentence": ["The", "BRCA2", "gene", "is", "mutated"],
        "ner": [{"indices": [1, 2], "type": "Gene_Or_GeneProduct"}],
    }
    out = convert(instance, with_labels=True)
    grid = out["grid_labels"]   # torch.Tensor [L, L]

    # NNW edge: upper-triangle cell [1, 2]
    assert grid[1, 2].item() == NNW_LABEL, "NNW edge [1,2] must equal NNW_LABEL."
    # THW anchor: lower-triangle cell [last_idx=2, first_idx=1]
    assert grid[2, 1].item() == label2id["Gene_Or_GeneProduct"], "THW anchor [2,1] wrong."
    # No stray NNW edges to unrelated words
    assert grid[1, 3].item() == 0, "No NNW edge [1,3] for this entity."

    # Decode and verify round-trip
    L = len(instance["sentence"])
    entities, warnings = decode_one_sentence(grid.numpy(), L, id2label)
    assert not warnings, f"Unexpected decode warnings: {warnings}"
    assert len(entities) == 1
    assert sorted(entities[0].indices) == [1, 2]
    assert id2label[entities[0].label_id] == "Gene_Or_GeneProduct"


# ---------------------------------------------------------------------------
# Test: discontinuous entity round-trip (critical bug fix)
# ---------------------------------------------------------------------------

def test_discontinuous_entity_roundtrip():
    """A discontinuous entity [word 1, word 3] encodes and decodes correctly.

    Verifies that gap word 2 is NOT connected by an NNW edge.  The old
    range(si, ei+1) fill-in bug would have produced indices [1, 2, 3] with
    NNW edges [1,2] and [2,3]; the correct fix gives indices [1, 3] with a
    direct NNW edge [1,3] and no edge through the gap.
    """
    label2id = {"<pad>": 0, "<suc>": 1, "Gene_Or_GeneProduct": 2}
    id2label = {v: k for k, v in label2id.items()}
    convert = make_converter(label2id)

    # sentence: ["The", "BRCA2", "and", "BRCA1", "genes", "are", "related"]
    # entity:   indices=[1, 3] (BRCA2 and BRCA1, skipping "and" at index 2)
    instance = {
        "sentence": ["The", "BRCA2", "and", "BRCA1", "genes", "are", "related"],
        "ner": [{"indices": [1, 3], "type": "Gene_Or_GeneProduct"}],
    }
    out = convert(instance, with_labels=True)
    grid = out["grid_labels"]

    # Direct NNW edge [1, 3] — no fill-in of gap word 2
    assert grid[1, 3].item() == NNW_LABEL, "NNW edge [1,3] must be NNW_LABEL."
    assert grid[3, 1].item() == label2id["Gene_Or_GeneProduct"], "THW anchor [3,1] wrong."
    # Gap word must NOT have NNW edges
    assert grid[1, 2].item() == 0, "Gap word: NNW edge [1,2] must be 0."
    assert grid[2, 3].item() == 0, "Gap word: NNW edge [2,3] must be 0."

    # Decode — must recover [1, 3], not [1, 2, 3]
    L = len(instance["sentence"])
    entities, warnings = decode_one_sentence(grid.numpy(), L, id2label)
    assert not warnings
    assert len(entities) == 1
    assert sorted(entities[0].indices) == [1, 3], (
        f"Expected [1, 3], got {sorted(entities[0].indices)}. "
        "If this is [1,2,3] the range fill-in bug has regressed."
    )
    assert id2label[entities[0].label_id] == "Gene_Or_GeneProduct"


# ---------------------------------------------------------------------------
# Test: resolve_fragments_to_indices (step03 pure function)
# ---------------------------------------------------------------------------

def test_resolve_fragments_discontinuous_no_fill():
    """Two non-adjacent fragments produce exactly {word1, word3}, not a range fill."""
    # 5 words, char spans: 0-4, 5-8, 9-12, 13-17, 18-22
    start_to_idx = {0: 0, 5: 1, 9: 2, 13: 3, 18: 4}
    end_to_idx   = {4: 0, 8: 1, 12: 2, 17: 3, 22: 4}

    # Fragments: word 1 [5,8] and word 3 [13,17]
    spans_list = [[5, 8], [13, 17]]
    indices, fragments, ok = resolve_fragments_to_indices(spans_list, start_to_idx, end_to_idx)

    assert ok
    assert indices == [1, 3], f"Expected [1, 3], got {indices}. Gap fill-in bug?"
    assert len(fragments) == 2
    assert fragments[0] == SubSpan(start_char=5, end_char=8)
    assert fragments[1] == SubSpan(start_char=13, end_char=17)


def test_resolve_fragments_single_span():
    """A single contiguous fragment resolves correctly."""
    start_to_idx = {0: 0, 5: 1, 9: 2}
    end_to_idx   = {4: 0, 8: 1, 12: 2}
    indices, fragments, ok = resolve_fragments_to_indices([[5, 8]], start_to_idx, end_to_idx)
    assert ok
    assert indices == [1]
    assert fragments == [SubSpan(start_char=5, end_char=8)]


def test_resolve_fragments_missing_key_returns_false():
    """Returns success=False when a char offset has no matching word."""
    start_to_idx = {0: 0, 5: 1}
    end_to_idx   = {4: 0, 8: 1}
    # Fragment starting at 99 is not in start_to_idx
    indices, fragments, ok = resolve_fragments_to_indices([[99, 110]], start_to_idx, end_to_idx)
    assert not ok
    assert indices == [] and fragments == []


def test_resolve_fragments_multi_word_contiguous():
    """A single fragment spanning multiple words produces range(si, ei+1)."""
    start_to_idx = {0: 0, 5: 1, 9: 2, 13: 3}
    end_to_idx   = {4: 0, 8: 1, 12: 2, 17: 3}
    # Fragment [5, 17] covers words 1, 2, 3
    indices, _, ok = resolve_fragments_to_indices([[5, 17]], start_to_idx, end_to_idx)
    assert ok
    assert indices == [1, 2, 3]


# ---------------------------------------------------------------------------
# Test: char span recovery (step06 pure function)
# ---------------------------------------------------------------------------

def test_recover_char_spans_empty():
    assert recover_char_spans([], [(0, 4), (5, 8)]) == []


def test_recover_char_spans_single_word():
    spans = [(0, 4), (5, 8), (9, 13)]
    assert recover_char_spans([1], spans) == [(5, 8)]


def test_recover_char_spans_contiguous():
    """Contiguous indices [1, 2] collapse into one span from spans[1][0] to spans[2][1]."""
    spans = [(0, 4), (5, 8), (9, 13), (14, 17)]
    assert recover_char_spans([1, 2], spans) == [(5, 13)]


def test_recover_char_spans_discontinuous():
    """Discontinuous indices [1, 3] produce two separate spans."""
    spans = [(0, 4), (5, 8), (9, 13), (14, 17), (18, 23)]
    assert recover_char_spans([1, 3], spans) == [(5, 8), (14, 17)]


def test_recover_char_spans_three_runs():
    """Three non-consecutive indices produce three separate spans."""
    spans = [(0, 4), (5, 8), (9, 13), (14, 17), (18, 23)]
    assert recover_char_spans([0, 2, 4], spans) == [(0, 4), (9, 13), (18, 23)]


def test_recover_char_spans_unsorted_input():
    """Unsorted indices are sorted before grouping — same result as sorted input."""
    spans = [(0, 4), (5, 8), (9, 13)]
    assert recover_char_spans([2, 0, 1], spans) == recover_char_spans([0, 1, 2], spans)


# ---------------------------------------------------------------------------
# Test: strict label2id lookup (feature_builder)
# ---------------------------------------------------------------------------

def test_label2id_strict_lookup_raises():
    """feature_builder raises KeyError on an unknown entity type, never returns 0."""
    label2id = {"<pad>": 0, "<suc>": 1, "Chemical": 2}
    convert = make_converter(label2id)
    instance = {
        "sentence": ["aspirin", "toxicity"],
        "ner": [{"indices": [0], "type": "UnknownType"}],
    }
    with pytest.raises(KeyError):
        convert(instance, with_labels=True)


def test_label2id_known_type_does_not_raise():
    """Known entity type encodes without error."""
    label2id = {"<pad>": 0, "<suc>": 1, "Chemical": 2}
    convert = make_converter(label2id)
    instance = {
        "sentence": ["aspirin"],
        "ner": [{"indices": [0], "type": "Chemical"}],
    }
    out = convert(instance, with_labels=True)
    assert out["grid_labels"][0, 0].item() == label2id["Chemical"]


# ---------------------------------------------------------------------------
# Test: ID join completeness (Bug D — guard + step06 join)
# ---------------------------------------------------------------------------

def test_no_silent_zip_truncation():
    """Guard raises ValueError naming the missing ID when a prediction is absent."""
    token_records = [_token_rec(f"pmid1_0_{i}") for i in range(3)]
    pred_records  = [_pred_rec(f"pmid1_0_{i}") for i in range(2)]   # _2 missing

    with pytest.raises(ValueError, match="pmid1_0_2"):
        check_id_join_completeness(
            [r.id for r in token_records],
            [r.id for r in pred_records],
            stage="test",
        )


def test_join_by_id_correct_pairing():
    """join_by_id pairs by id regardless of list order."""
    ids = ["A", "B", "C"]
    tokens = [_token_rec(i) for i in ids]
    preds  = [_pred_rec(i) for i in reversed(ids)]

    joined = join_by_id(tokens, preds)
    assert [(t.id, p.id) for t, p in joined] == [("A", "A"), ("B", "B"), ("C", "C")]


def test_join_by_id_preserves_token_order():
    """join_by_id returns pairs in token_records order."""
    tokens = [_token_rec("X"), _token_rec("Y")]
    preds  = [_pred_rec("Y"), _pred_rec("X")]   # reversed

    joined = join_by_id(tokens, preds)
    assert joined[0][0].id == "X" and joined[0][1].id == "X"
    assert joined[1][0].id == "Y" and joined[1][1].id == "Y"
