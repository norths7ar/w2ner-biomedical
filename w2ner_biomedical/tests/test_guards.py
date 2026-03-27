# =============================================================================
# tests/test_guards.py
#
# Unit tests for guards/validators.py.
# All five guards are pure functions; no external data or model dependencies.
# Each guard is tested for: the passing case, the warning-only case, and the
# hard-raise case.
# =============================================================================

from __future__ import annotations

import logging
import pytest

from w2ner_biomedical.guards.validators import (
    check_record_count_parity,
    check_entity_alignment_rate,
    check_type_vocabulary_consistency,
    check_id_join_completeness,
    check_label_vocab_consistency,
)


# ---------------------------------------------------------------------------
# Guard 1 — check_record_count_parity
# ---------------------------------------------------------------------------

class TestRecordCountParity:
    def test_no_drop_passes(self):
        check_record_count_parity(100, 100, stage="test")

    def test_count_increase_passes(self):
        check_record_count_parity(100, 110, stage="test")

    def test_previous_none_skips(self):
        check_record_count_parity(None, 50, stage="test")

    def test_previous_zero_skips(self):
        check_record_count_parity(0, 0, stage="test")

    def test_tiny_drop_passes(self):
        # 0.5% — below warn threshold
        check_record_count_parity(1000, 995, stage="test")

    def test_drop_between_warn_and_max_warns(self, caplog):
        # 3% drop — above warn (1%) but below max (5%)
        with caplog.at_level(logging.WARNING, logger="w2ner_biomedical.guards.validators"):
            check_record_count_parity(1000, 970, stage="test")
        assert any("drop rate" in r.message.lower() for r in caplog.records)

    def test_drop_above_max_raises(self):
        with pytest.raises(ValueError, match="drop rate"):
            check_record_count_parity(100, 90, stage="test")   # 10%

    def test_custom_threshold_tighter(self):
        # 3% drop exceeds custom max of 2%
        with pytest.raises(ValueError):
            check_record_count_parity(100, 97, stage="test", max_drop_rate=0.02)

    def test_exactly_at_default_max_passes(self):
        # 5.0% is exactly the threshold; implementation uses drop_rate > max,
        # so exactly 5% does NOT raise.
        check_record_count_parity(100, 95, stage="test")

    def test_just_over_default_max_raises(self):
        with pytest.raises(ValueError):
            check_record_count_parity(1000, 944, stage="test")   # 5.6%


# ---------------------------------------------------------------------------
# Guard 2 — check_entity_alignment_rate
# ---------------------------------------------------------------------------

class TestEntityAlignmentRate:
    def test_no_drops_passes(self):
        check_entity_alignment_rate(0, 100, stage="test")

    def test_zero_total_skips(self):
        check_entity_alignment_rate(0, 0, stage="test")

    def test_tiny_drop_passes(self):
        check_entity_alignment_rate(4, 1000, stage="test")   # 0.4%

    def test_drop_between_warn_and_max_warns(self, caplog):
        # 2% drop — above warn (1%) but below max (5%)
        with caplog.at_level(logging.WARNING, logger="w2ner_biomedical.guards.validators"):
            check_entity_alignment_rate(20, 1000, stage="test")
        assert any("drop rate" in r.message.lower() for r in caplog.records)

    def test_drop_above_max_raises(self):
        with pytest.raises(ValueError, match="alignment drop rate"):
            check_entity_alignment_rate(10, 100, stage="test")   # 10%

    def test_custom_threshold(self):
        with pytest.raises(ValueError):
            check_entity_alignment_rate(3, 100, stage="test", max_drop_rate=0.02)


# ---------------------------------------------------------------------------
# Guard 3 — check_type_vocabulary_consistency
# ---------------------------------------------------------------------------

class TestTypeVocabularyConsistency:
    def test_empty_existing_config_skips(self):
        check_type_vocabulary_consistency({"Chemical", "Disease"}, [], stage="test")

    def test_all_types_present_passes(self):
        check_type_vocabulary_consistency(
            {"Chemical", "Disease"},
            ["Chemical", "Disease"],
            stage="test",
        )

    def test_new_type_not_in_old_config_passes(self):
        # Observed MORE types than previously configured — fine, vocab grew.
        check_type_vocabulary_consistency(
            {"Chemical", "Disease", "Species"},
            ["Chemical", "Disease"],
            stage="test",
        )

    def test_disappeared_type_raises(self):
        with pytest.raises(ValueError, match="disappeared"):
            check_type_vocabulary_consistency(
                {"Chemical"},
                ["Chemical", "Disease"],
                stage="test",
            )

    def test_disappeared_type_with_allow_removal_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="w2ner_biomedical.guards.validators"):
            check_type_vocabulary_consistency(
                {"Chemical"},
                ["Chemical", "Disease"],
                stage="test",
                allow_removal=True,
            )
        assert any("no longer observed" in r.message.lower() for r in caplog.records)

    def test_multiple_disappeared_types_all_listed(self):
        with pytest.raises(ValueError) as exc_info:
            check_type_vocabulary_consistency(
                set(),
                ["Chemical", "Disease", "Species"],
                stage="test",
            )
        msg = str(exc_info.value)
        assert "Chemical" in msg
        assert "Disease" in msg
        assert "Species" in msg


# ---------------------------------------------------------------------------
# Guard 4 — check_id_join_completeness
# ---------------------------------------------------------------------------

class TestIdJoinCompleteness:
    def test_complete_match_passes(self):
        check_id_join_completeness(["a", "b", "c"], ["a", "b", "c"], stage="test")

    def test_order_independent(self):
        # Guard uses sets internally — order should not matter.
        check_id_join_completeness(["b", "a", "c"], ["c", "a", "b"], stage="test")

    def test_empty_both_passes(self):
        check_id_join_completeness([], [], stage="test")

    def test_extra_pred_id_warns(self, caplog):
        # pred has an extra ID not in token_ids — warns but does not raise.
        with caplog.at_level(logging.WARNING, logger="w2ner_biomedical.guards.validators"):
            check_id_join_completeness(["a", "b"], ["a", "b", "c"], stage="test")
        assert any("no matching token" in r.message.lower() for r in caplog.records)

    def test_missing_pred_id_raises(self):
        with pytest.raises(ValueError, match="no corresponding prediction"):
            check_id_join_completeness(["a", "b", "c"], ["a", "b"], stage="test")

    def test_missing_pred_message_contains_id(self):
        with pytest.raises(ValueError) as exc_info:
            check_id_join_completeness(["a", "b", "missing_id"], ["a", "b"], stage="test")
        assert "missing_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Guard 5 — check_label_vocab_consistency
# ---------------------------------------------------------------------------

class TestLabelVocabConsistency:
    def test_matching_dimensions_passes(self):
        label2id = {"<pad>": 0, "<suc>": 1, "Chemical": 2, "Disease": 3}
        check_label_vocab_consistency(4, label2id, stage="test")

    def test_mismatch_raises(self):
        label2id = {"<pad>": 0, "<suc>": 1, "Chemical": 2}
        with pytest.raises(ValueError, match="label2id"):
            check_label_vocab_consistency(5, label2id, stage="test")

    def test_mismatch_message_contains_both_counts(self):
        label2id = {"<pad>": 0, "<suc>": 1}
        with pytest.raises(ValueError) as exc_info:
            check_label_vocab_consistency(10, label2id, stage="test")
        msg = str(exc_info.value)
        assert "10" in msg
        assert "2" in msg
