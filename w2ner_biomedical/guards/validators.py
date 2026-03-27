# =============================================================================
# guards/validators.py
#
# PURPOSE
#   Explicit pipeline guards that surface silent failures as visible errors
#   or prominent warnings.  These five functions implement the "minimum guard
#   set" identified in the architectural analysis: together they cover the
#   most dangerous classes of silent degradation in the pipeline.
#
# CORRESPONDS TO
#   Nothing in the original W2NER repo — all failures were silent there.
#
# KEY DESIGN CHANGES
#   Design principle: aggregate before judging.
#   Each guard collects a statistic (count, rate, set diff) and compares it
#   against a threshold.  If the threshold is exceeded, the guard raises a
#   descriptive ValueError rather than logging a per-record DEBUG message.
#   Thresholds are configurable so callers can tighten or loosen them for
#   different datasets without editing guard internals.
#
#   The five guards and their failure modes:
#
#   Guard 1 — check_record_count_parity
#     Detects: steps that silently drop >N% of records.
#     Trigger: after every step that passes TokenRecords to the next stage
#              (step02 output → step03 input, step03 output → step05 input,
#              step05 output → step06 input).  Not called across unit changes
#              (step01 documents → step02 sentences, where count goes up).
#
#   Guard 2 — check_entity_alignment_rate
#     Detects: entity annotation drops due to offset drift (Bug B),
#              sentence boundary mismatches, tokenizer changes.
#     Trigger: after step03_add_labels completes processing a file.
#
#   Guard 3 — check_type_vocabulary_consistency
#     Detects: entity types that were previously written into the model config
#              but are no longer present in the newly observed training data.
#              Prevents silent vocabulary shrinkage between pipeline runs —
#              either a type was intentionally removed (pass allow_removal=True)
#              or data was silently lost (hard error by default).
#     Trigger: in step04_finalize_config before rewriting entity_types in the
#              config.  No-op on the first run when existing_config_types=[].
#
#   Guard 4 — check_id_join_completeness
#     Detects: missing prediction records in step06_postprocess ID join.
#              Replaces the original zip() truncation (Bug D).
#     Trigger: in step06_postprocess before any output is written.
#
#   Guard 5 — check_label_vocab_consistency
#     Detects: mismatch between model output head dimension and label2id
#              size.  Prevents silent wrong-argmax outputs when the model
#              was saved with a different vocabulary.
#     Trigger: in step05_predict at model load time.
#
# BUGS ADDRESSED
#   [HIGH]   Unicode normalisation shifts char offsets silently (Bug B):
#            check_entity_alignment_rate surfaces this as a >threshold drop.
#
#   [HIGH]   zip truncation in postprocess assumes step05 never fails (Bug D):
#            check_id_join_completeness replaces zip with an explicit keyed
#            join that raises on missing IDs.
#
#   [MEDIUM] Config entity_types silently shrinks between runs (Bug H variant):
#            check_type_vocabulary_consistency raises when previously configured
#            types disappear from newly observed data.
# =============================================================================

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

# Default thresholds — callers can override via keyword argument.
DEFAULT_MAX_DROP_RATE: float = 0.05    # 5% record/entity drop triggers error
DEFAULT_WARN_DROP_RATE: float = 0.01   # 1% triggers WARNING (not error)


# ---------------------------------------------------------------------------
# Guard 1 — Record count parity
# ---------------------------------------------------------------------------

def check_record_count_parity(
    previous_count: int | None,
    current_count: int,
    stage: str,
    max_drop_rate: float = DEFAULT_MAX_DROP_RATE,
    warn_drop_rate: float = DEFAULT_WARN_DROP_RATE,
) -> None:
    """Assert that current_count has not dropped more than max_drop_rate below previous_count.

    Call this after any pipeline step that transforms a JSONL file where the
    unit of records stays the same (e.g. TokenRecords in → TokenRecords out).
    Do NOT call it across unit-boundary steps where record count legitimately
    increases (e.g. step01 documents → step02 sentence chunks).

    Parameters
    ----------
    previous_count:
        Record count from the upstream stage.  Pass None for the first stage
        (no prior count to compare against) — the guard returns immediately.
    current_count:
        Record count from the current stage output.
    stage:
        Human-readable label for error/warning messages, e.g. "step03_add_labels".
    """
    if previous_count is None:
        return
    if previous_count <= 0:
        return

    drop_rate = (previous_count - current_count) / previous_count
    if drop_rate <= 0:
        # Count stayed the same or went up — not a drop, nothing to report.
        return

    if drop_rate > max_drop_rate:
        raise ValueError(
            f"[{stage}] Record drop rate {drop_rate:.1%} exceeds "
            f"{max_drop_rate:.1%} threshold "
            f"({previous_count} -> {current_count} records). "
            f"Check unicode normalisation, sentence splitting, or step output."
        )
    elif drop_rate > warn_drop_rate:
        LOGGER.warning(
            "[%s] Record drop rate %.1f%% (%d -> %d) exceeds warning threshold %.1f%%.",
            stage, drop_rate * 100, previous_count, current_count, warn_drop_rate * 100,
        )


# ---------------------------------------------------------------------------
# Guard 2 — Entity alignment rate
# ---------------------------------------------------------------------------

def check_entity_alignment_rate(
    n_dropped: int,
    n_total: int,
    stage: str,
    max_drop_rate: float = DEFAULT_MAX_DROP_RATE,
    warn_drop_rate: float = DEFAULT_WARN_DROP_RATE,
) -> None:
    """Assert that entity annotation drop rate is within acceptable bounds.

    A high drop rate is the primary symptom of Bug B (unicode offset drift),
    sentence boundary misalignment, or annotation coordinate system mismatch.

    Parameters
    ----------
    n_dropped:
        Number of entities that could not be aligned to any sentence chunk.
    n_total:
        Total number of entities in the input annotation file.
    stage:
        Human-readable label, e.g. "step03_add_labels".
    """
    if n_total == 0:
        return

    drop_rate = n_dropped / n_total

    if drop_rate > max_drop_rate:
        raise ValueError(
            f"[{stage}] Entity alignment drop rate {drop_rate:.1%} exceeds "
            f"{max_drop_rate:.1%} threshold "
            f"({n_dropped} of {n_total} entities could not be aligned). "
            f"Check unicode normalisation in step01, sentence chunking in step02, "
            f"or character offset coordinate system in the annotation source."
        )
    elif drop_rate > warn_drop_rate:
        LOGGER.warning(
            "[%s] Entity alignment drop rate %.1f%% (%d of %d) exceeds warning threshold %.1f%%.",
            stage, drop_rate * 100, n_dropped, n_total, warn_drop_rate * 100,
        )


# ---------------------------------------------------------------------------
# Guard 3 — Type vocabulary consistency
# ---------------------------------------------------------------------------

def check_type_vocabulary_consistency(
    observed_types: set[str],
    existing_config_types: list[str],
    stage: str,
    allow_removal: bool = False,
) -> None:
    """Assert that types previously in the config are still present in observed data.

    Before step04_finalize_config rewrites entity_types, this guard compares
    the types seen in the new training data against any types already written
    into the config from a previous run.  If a type has disappeared it means
    either:
      - annotations for that type were silently lost (hard error by default), or
      - the type was intentionally removed (pass allow_removal=True).

    This is a no-op on the first run because existing_config_types will be [].

    Parameters
    ----------
    observed_types:
        Set of entity type strings seen in the current step03 output.
    existing_config_types:
        The entity_types list currently in the model config (from the previous
        run).  Empty list on first run.
    stage:
        Human-readable label, e.g. "step04_finalize_config".
    allow_removal:
        If True, log a warning instead of raising when types disappear.
        Pass this when intentionally reducing the entity type vocabulary.
    """
    if not existing_config_types:
        return

    disappeared = set(existing_config_types) - observed_types
    if not disappeared:
        return

    if not allow_removal:
        raise ValueError(
            f"[{stage}] Entity types previously in config have disappeared from "
            f"observed training data: {sorted(disappeared)}. "
            f"This may indicate silent data loss or an incomplete annotation run. "
            f"If this removal is intentional, pass allow_removal=True (or "
            f"--allow-type-removal on the CLI)."
        )
    else:
        LOGGER.warning(
            "[%s] Entity types no longer observed in data (allow_removal=True): %s. "
            "These types will be removed from the config.",
            stage, sorted(disappeared),
        )


# ---------------------------------------------------------------------------
# Guard 4 — ID join completeness
# ---------------------------------------------------------------------------

def check_id_join_completeness(
    token_ids: list[str],
    pred_ids: list[str],
    stage: str,
) -> None:
    """Assert that every token record ID has a corresponding prediction record ID.

    Replaces the original zip() truncation in step07_post_process.py (Bug D)
    with an explicit, traceable error.  Called in step06_postprocess before
    any output is written so that a partial step05 run fails loudly rather
    than producing silently truncated output.

    Parameters
    ----------
    token_ids:
        Ordered list of IDs from the TokenRecord JSONL (step02/step03 output).
    pred_ids:
        Ordered list of IDs from the PredictRecord JSONL (step05 output).
    stage:
        Human-readable label, e.g. "step06_postprocess".
    """
    token_set = set(token_ids)
    pred_set = set(pred_ids)

    missing = token_set - pred_set
    extra = pred_set - token_set

    if missing:
        sample = sorted(missing)[:10]
        raise ValueError(
            f"[{stage}] {len(missing)} token record ID(s) have no corresponding "
            f"prediction record. step05_predict likely failed or was interrupted "
            f"partway through. Re-run step05 before continuing. "
            f"Missing IDs (first {len(sample)}): {sample}"
        )

    if extra:
        LOGGER.warning(
            "[%s] %d prediction ID(s) have no matching token record and will be "
            "ignored during join. First 10: %s",
            stage, len(extra), sorted(extra)[:10],
        )


# ---------------------------------------------------------------------------
# Guard 5 — Label vocabulary vs. model head dimension
# ---------------------------------------------------------------------------

def check_label_vocab_consistency(
    model_head_dim: int,
    label2id: dict[str, int],
    stage: str,
) -> None:
    """Assert that the model's output head dimension matches len(label2id).

    A mismatch means the model weights and the label2id file are from
    different training runs.  argmax on mismatched dimensions produces
    silently wrong type predictions with no runtime error.

    Parameters
    ----------
    model_head_dim:
        Output feature count of the model's final linear layer
        (``model.predictor.linear.out_features``).
    label2id:
        Label vocabulary mapping loaded from label2id.json.
    stage:
        Human-readable label, e.g. "step05_predict".
    """
    if model_head_dim != len(label2id):
        raise ValueError(
            f"[{stage}] Model output head has {model_head_dim} classes but "
            f"label2id has {len(label2id)} entries. "
            f"The model weights and label2id.json are from different training runs. "
            f"Load the label2id.json that was saved alongside these model weights."
        )
