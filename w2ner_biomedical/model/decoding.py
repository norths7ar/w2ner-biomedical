# =============================================================================
# model/decoding.py
#
# PURPOSE
#   Decode W2NER label grids into concrete entity spans.  Takes a batch of
#   argmax-label matrices and returns lists of (word_indices, entity_type)
#   tuples per sentence.
#
# CORRESPONDS TO
#   core/model/decoding.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - Explosion guard is explicit, not silent:
#     The original decode_grid contained a guard against combinatorial
#     explosion in the BFS entity reconstruction loop.  When triggered it
#     set a break_outer_loop flag and silently dropped the in-progress entity,
#     logging at ERROR level but continuing without raising.  The result was
#     an incomplete entity list with no signal to the caller that truncation
#     had occurred.
#     This version replaces the silent continue with a structured DecodeWarning
#     object that is returned alongside the entity list.  The caller decides
#     whether to treat it as a logged anomaly or a hard error.  Importantly,
#     the partial chain is INCLUDED in the result (partial is better than
#     silent loss) and the caller can inspect the warning to triage the cause.
#
#   - Decoding is separated from logging:
#     decode_one_sentence() takes no logger; it returns (entities, warnings).
#     The caller (decode_grid) aggregates all warnings and logs them once
#     after the batch, with sentence indices for traceability.  This makes
#     decode_one_sentence pure and unit-testable without mock loggers.
#
#   - Entity-level F1 helper (cal_f1) is co-located here:
#     In the original, cal_f1 was in decoding.py but the evaluate() method
#     in trainer.py also reimplemented parts of it inline.  Both now use
#     this single canonical implementation.
#
# BUGS ADDRESSED
#   [MEDIUM] Explosion guard in decoding silently continues with partial
#            results.
#            Fixed: guard emits a structured DecodeWarning and includes the
#            partial chain in the result.  Caller receives the warning and
#            can surface it.
# =============================================================================

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


from .constants import NNW_LABEL  # noqa: F401  re-exported for callers that import from here


@dataclass
class DecodeWarning:
    """Structured warning emitted when decoding encounters an anomaly.

    Returned by decode_one_sentence; aggregated and logged by decode_grid.
    The caller can treat this as a logged anomaly or raise based on severity.
    """
    sentence_idx: int
    head_idx: int
    tail_idx: int
    message: str
    stats: dict = field(default_factory=dict)


class EntitySpan(NamedTuple):
    """Decoded entity: word indices within the sentence + label id."""
    indices: list[int]
    label_id: int


@dataclass
class _Node:
    """Internal BFS state: per-word bookkeeping for entity reconstruction."""
    THW: list = field(default_factory=list)             # [(tail_idx, type_id), ...]
    NNW: dict = field(default_factory=lambda: defaultdict(set))  # (head,tail) → {next_idx}


def decode_one_sentence(
    instance: np.ndarray,        # [L, L] — argmax label matrix, instance[tail, head]
    length: int,
    id2label: dict[int, str],
    max_total_paths: int = 50_000,
    max_path_len: int = 30,
    max_branching: int = 128,
) -> tuple[list[EntitySpan], list[DecodeWarning]]:
    """Decode one label matrix into a list of EntitySpan objects.

    Returns (entities, warnings).
    - entities: all decoded entity spans including any partial chains.
    - warnings: non-empty when the explosion guard fires.  The guard
      fires when the BFS path count exceeds max_total_paths OR when
      both max_path_len and max_branching thresholds are exceeded.

    The caller decides how to handle warnings — this function never
    silently discards partial results.
    """
    nodes: list[_Node] = [_Node() for _ in range(length)]
    predicts: list[EntitySpan] = []
    warnings: list[DecodeWarning] = []

    # --- Build graph by scanning the label matrix ---
    for cur in reversed(range(length)):
        heads: list[int] = []
        for pre in range(cur + 1):
            label = instance[cur, pre]

            # THW edge: lower-triangle, label > 1 means a real entity type
            if label > 1:
                nodes[pre].THW.append((cur, label))
                heads.append(pre)

            # NNW edge: upper-triangle, label == NNW_LABEL
            if pre < cur and instance[pre, cur] == NNW_LABEL:
                # Propagate to all active heads for the current tail position
                for head in heads:
                    nodes[pre].NNW[(head, cur)].add(cur)
                # Also extend any open chains whose tail reaches cur
                for (head, tail) in nodes[cur].NNW.keys():
                    if tail >= cur and head <= pre:
                        nodes[pre].NNW[(head, tail)].add(cur)

    # --- BFS entity reconstruction ---
    for idx, node in enumerate(nodes):
        for tail, type_id in node.THW:
            # Single-word entity: head == tail
            if idx == tail:
                predicts.append(EntitySpan([idx], type_id))
                continue

            q: deque[list[int]] = deque([[idx]])
            visited: set[tuple[int, ...]] = set()
            total_paths = 0
            max_path_len_observed = 0
            max_branching_observed = 0
            explosion_triggered = False

            while q:
                chain = q.pop()
                chain_tuple = tuple(chain)
                if chain_tuple in visited:
                    continue
                visited.add(chain_tuple)

                total_paths += 1
                max_path_len_observed = max(max_path_len_observed, len(chain))
                branching = len(nodes[chain[-1]].NNW[(idx, tail)])
                max_branching_observed = max(max_branching_observed, branching)

                # [MEDIUM bug fix] Explosion guard: emit a structured warning
                # instead of silently breaking.  Include the partial chain in
                # the output — partial evidence is better than silent loss.
                if total_paths > max_total_paths or (
                    max_path_len_observed > max_path_len and
                    max_branching_observed > max_branching
                ):
                    warnings.append(DecodeWarning(
                        sentence_idx=-1,   # filled in by decode_grid with batch index
                        head_idx=idx,
                        tail_idx=tail,
                        message=(
                            f"BFS explosion guard triggered: "
                            f"total_paths={total_paths}, "
                            f"max_path_len={max_path_len_observed}, "
                            f"max_branching={max_branching_observed}"
                        ),
                        stats={
                            "total_paths": total_paths,
                            "max_path_len": max_path_len_observed,
                            "max_branching": max_branching_observed,
                        },
                    ))
                    # Include the partial chain rather than discarding it
                    predicts.append(EntitySpan(list(chain), type_id))
                    explosion_triggered = True
                    break

                for next_idx in nodes[chain[-1]].NNW[(idx, tail)]:
                    if next_idx in chain:
                        # Cycle guard — skip to avoid infinite loops
                        continue
                    if next_idx == tail:
                        predicts.append(EntitySpan(chain + [next_idx], type_id))
                    else:
                        q.append(chain + [next_idx])

            if explosion_triggered:
                break

    return predicts, warnings


def decode_grid(
    logits: np.ndarray,
    length_list: list[int],
    tokens_list: list[list[str]],
    id2label: dict[int, str],
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Decode a batch of argmax label matrices into per-sentence entity dicts.

    Returns a list of dicts compatible with PredictRecord schema:
        {
            "sentence": ["word", ...],
            "entity": [
                {"text": ["word"], "indices": [i, ...], "type": "Chemical"},
                ...
            ]
        }

    All DecodeWarnings from decode_one_sentence are stamped with the
    batch-relative sentence index and logged at WARNING level after the full
    batch completes.  Using a single post-batch log avoids interleaving
    warning messages with per-sentence debug output.
    """
    if len(logits) != len(tokens_list):
        raise ValueError(
            f"decode_grid: logits batch size ({len(logits)}) != "
            f"tokens_list length ({len(tokens_list)})"
        )

    decoded_results: list[dict] = []
    all_warnings: list[DecodeWarning] = []

    for sent_idx, (instance, length, sentence) in enumerate(
        zip(logits, length_list, tokens_list)
    ):
        entities, warnings = decode_one_sentence(
            instance=instance[:length, :length],
            length=length,
            id2label=id2label,
        )

        # Stamp sentence index into warnings for traceability
        for w in warnings:
            w.sentence_idx = sent_idx
        all_warnings.extend(warnings)

        # Deduplicate (same indices + same type can appear from multiple BFS paths)
        seen: set[tuple] = set()
        entity_list: list[dict] = []
        for span in entities:
            key = (tuple(span.indices), span.label_id)
            if key in seen:
                continue
            seen.add(key)
            entity_list.append({
                "text":    [sentence[i] for i in span.indices],
                "indices": span.indices,
                "type":    id2label[span.label_id],
            })

        if logger:
            logger.debug("Sentence %d: decoded %d entities", sent_idx, len(entity_list))

        decoded_results.append({"sentence": sentence, "entity": entity_list})

    # Emit all warnings together after the batch
    if all_warnings and logger:
        for w in all_warnings:
            logger.warning(
                "Decode explosion [sent=%d head=%d tail=%d]: %s | stats=%s",
                w.sentence_idx, w.head_idx, w.tail_idx, w.message, w.stats,
            )

    return decoded_results


def decode_and_compare(
    logits: np.ndarray,
    length_list: list[int],
    tokens_list: list[list[str]],
    gold_entities: list[list[dict]],
    id2label: dict[int, str],
    logger: logging.Logger | None = None,
) -> tuple[int, int, int]:
    """Decode predictions and compare to gold entities for entity-level F1.

    gold_entities[i] is a list of dicts with "indices" (list[int]) and
    "type" (str) keys, matching the NEREntry schema.

    Returns (n_correct, n_predicted, n_gold).
    """
    pred_results = decode_grid(
        logits=logits,
        length_list=length_list,
        tokens_list=tokens_list,
        id2label=id2label,
        logger=logger,
    )

    pred_set: set[tuple] = set()
    gold_set: set[tuple] = set()

    for sent_idx, pred in enumerate(pred_results):
        for ent in pred["entity"]:
            pred_set.add((sent_idx, tuple(ent["indices"]), ent["type"]))

    for sent_idx, golds in enumerate(gold_entities):
        for ent in golds:
            gold_set.add((sent_idx, tuple(ent["indices"]), ent["type"]))

    if logger:
        logger.debug("decode_and_compare: pred=%d  gold=%d", len(pred_set), len(gold_set))

    n_correct = len(pred_set & gold_set)
    n_predicted = len(pred_set)
    n_gold = len(gold_set)
    return n_correct, n_predicted, n_gold


def cal_f1(n_correct: int, n_predicted: int, n_gold: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from entity counts.

    Returns (f1, precision, recall).  All three are 0.0 for degenerate
    inputs (zero predicted or zero gold entities).
    """
    if n_predicted == 0 or n_gold == 0:
        return 0.0, 0.0, 0.0
    p = n_correct / n_predicted
    r = n_correct / n_gold
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r
