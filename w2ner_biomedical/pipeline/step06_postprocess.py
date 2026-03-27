# =============================================================================
# pipeline/step06_postprocess.py
#
# PURPOSE
#   Join step05 predictions with step03 token spans to recover character-
#   level entity boundaries, then apply per-PMID majority-vote type
#   normalisation.  Emits one PostprocessRecord per sentence chunk as JSONL.
#
# CORRESPONDS TO
#   step07_post_process.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - ID-keyed join replaces position-based zip:
#     The original used zip(token_data, pred_data) which silently truncated
#     to the shorter list (Bug D).  Guard 4 (check_id_join_completeness)
#     verifies that every TokenRecord.id has a matching PredictRecord.id
#     before any processing starts.  join_by_id then does an O(n) dict
#     lookup that is guaranteed to succeed after the guard passes.
#
#   - Char span recovery and majority voting are separate functions:
#     recover_char_spans, compute_majority_types, and the final
#     PostprocessRecord build are distinct passes, each testable in
#     isolation.
#
#   - Majority vote key uses entity surface text within PMID (Bug E):
#     Preserved from the original for compatibility, but quarantined in
#     get_majority_vote_key() with an explicit TODO so a context-aware
#     key can be substituted without touching the rest of the code.
#
#   - fulltext is loaded from step01 IngestRecord JSONL (the pre-normalised
#     canonical fulltext), not reconstructed from title + abstract via
#     pandas (original Bug-B-adjacent issue).
#
#   - Atomic JSONL writes + StageManifest per output file.
#
# BUGS ADDRESSED
#   [HIGH]   zip truncation assumes step05 never fails (Bug D):
#            Guard 4 + ID-keyed join raise explicitly before any output
#            is written.
#
#   [MEDIUM] Majority vote uses entity text as key — collapses context-
#            dependent predictions (Bug E): quarantined with a TODO.
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from myutils import load_json, load_jsonl, save_jsonl, get_logger

from ..specs.schemas import (
    TokenRecord, PredictRecord, PostprocessRecord, RefinedEntity, StageManifest,
)
from ..guards.validators import check_record_count_parity, check_id_join_completeness

LOGGER: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span / text helpers
# ---------------------------------------------------------------------------

def recover_char_spans(
    indices: list[int],
    sentence_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map word-level indices to fulltext-absolute character spans.

    Groups consecutive indices into contiguous runs; each run becomes one
    (start_char, end_char) tuple.  Non-consecutive runs (discontinuous
    entities) become separate tuples.

    Returns an empty list if indices is empty.
    """
    if not indices:
        return []

    sorted_idx = sorted(indices)
    result: list[tuple[int, int]] = []
    run_start_idx = sorted_idx[0]

    for i in range(1, len(sorted_idx)):
        if sorted_idx[i] != sorted_idx[i - 1] + 1:
            result.append((
                sentence_spans[run_start_idx][0],
                sentence_spans[sorted_idx[i - 1]][1],
            ))
            run_start_idx = sorted_idx[i]

    result.append((
        sentence_spans[run_start_idx][0],
        sentence_spans[sorted_idx[-1]][1],
    ))
    return result


def get_entity_surface_text(sentence_str: str, sentence_spans: list[tuple[int, int]]) -> str:
    """Extract entity text from sentence string using sentence-relative spans.

    Fragments for discontinuous entities are joined with a single space.
    sentence_spans must be relative to the start of sentence_str (i.e. each
    span is already offset-subtracted by fulltext_offset).
    """
    return " ".join(sentence_str[s:e] for s, e in sentence_spans)


def get_majority_vote_key(pmid: str, entity_text: str) -> str:
    """Return the key used for majority-vote type aggregation.

    CURRENT IMPLEMENTATION: uses entity surface text within PMID.
    This collapses context-dependent entities with the same surface text
    (Bug E — open design question).

    TODO: replace with (pmid, sentence_id, span_tuple) for a context-aware
    key that does not merge predictions across sentences.
    """
    return f"{pmid}::{entity_text}"


# ---------------------------------------------------------------------------
# ID-keyed join (Guard 4)
# ---------------------------------------------------------------------------

def join_by_id(
    token_records: list[TokenRecord],
    pred_records: list[PredictRecord],
) -> list[tuple[TokenRecord, PredictRecord]]:
    """Join token and prediction records on id field.

    Guard 4 (check_id_join_completeness) must be called before this function
    to guarantee that every token record id has a matching prediction record id.
    This function assumes the guard has already passed and does a simple O(n)
    dict lookup.

    Returns pairs in token_record order.
    """
    pred_by_id: dict[str, PredictRecord] = {r.id: r for r in pred_records}
    return [(tr, pred_by_id[tr.id]) for tr in token_records]


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def compute_majority_types(
    joined: list[tuple[TokenRecord, PredictRecord]],
    pmid_text_map: dict[str, str],
) -> dict[str, dict[str, str]]:
    """First pass: count entity type votes per (pmid, entity_key).

    Returns majority_types[pmid][entity_key] = most_common_type.
    entity_key is produced by get_majority_vote_key (currently surface text).
    """
    # entity_counter[pmid][entity_key][type] = count
    entity_counter: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for token_rec, pred_rec in joined:
        pmid = token_rec.pmid
        fulltext = pmid_text_map.get(pmid, "")
        if not token_rec.spans:
            continue

        fulltext_offset = token_rec.spans[0][0]
        sentence_end = token_rec.spans[-1][1]
        sentence_str = fulltext[fulltext_offset:sentence_end]

        for entity in pred_rec.entity:
            fulltext_spans = recover_char_spans(entity.indices, token_rec.spans)
            if not fulltext_spans:
                continue
            sentence_spans = [(s - fulltext_offset, e - fulltext_offset) for s, e in fulltext_spans]
            entity_text = get_entity_surface_text(sentence_str, sentence_spans)

            if not entity_text.strip():
                LOGGER.warning(
                    "Empty surface text for entity indices=%s in record %s — skipping.",
                    entity.indices, token_rec.id,
                )
                continue

            key = get_majority_vote_key(pmid, entity_text)
            entity_counter[pmid][key][entity.type] += 1

    majority_types: dict[str, dict[str, str]] = {}
    for pmid, entities in entity_counter.items():
        majority_types[pmid] = {
            key: max(type_counts, key=lambda t: type_counts[t])
            for key, type_counts in entities.items()
        }
    return majority_types


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def postprocess_file(
    token_path: Path,
    pred_path: Path,
    output_path: Path,
    pmid_text_map: dict[str, str],
    force: bool,
) -> int:
    """Full postprocessing pipeline for one file pair.

    1. Load TokenRecords and PredictRecords.
    2. Guard 4: check ID join completeness (raises on any missing pred ID).
    3. Guard 1: check record count parity vs step05 manifest.
    4. join_by_id (O(n) dict lookup, safe after Guard 4).
    5. compute_majority_types (first pass).
    6. Build PostprocessRecords with majority-voted types (second pass).
    7. Write atomically.

    Returns record count, or -1 if skipped.
    """
    if output_path.exists() and not force:
        LOGGER.info("Skipping %s (output exists; use --force to overwrite)", token_path.stem)
        return -1

    token_records = [TokenRecord.model_validate(r) for r in load_jsonl(token_path)]
    pred_records = [PredictRecord.model_validate(r) for r in load_jsonl(pred_path)]

    # Guard 4: ID join completeness — raises with clear message if pred IDs are missing
    check_id_join_completeness(
        token_ids=[r.id for r in token_records],
        pred_ids=[r.id for r in pred_records],
        stage="step06_postprocess",
    )

    # Guard 1: record count parity vs step05 manifest
    prev_count: int | None = None
    step05_manifest_path = pred_path.with_suffix(pred_path.suffix + ".meta.json")
    if step05_manifest_path.exists():
        prev_data = json.loads(step05_manifest_path.read_text(encoding="utf-8"))
        prev_count = prev_data.get("record_count")
    check_record_count_parity(prev_count, len(pred_records), "step06_postprocess")

    joined = join_by_id(token_records, pred_records)

    # First pass: majority vote types
    majority_types = compute_majority_types(joined, pmid_text_map)

    # Second pass: build PostprocessRecords
    postprocess_records: list[PostprocessRecord] = []

    for token_rec, pred_rec in joined:
        pmid = token_rec.pmid
        fulltext = pmid_text_map.get(pmid, "")

        if not token_rec.spans:
            LOGGER.warning("TokenRecord %s has empty spans — skipping.", token_rec.id)
            continue

        fulltext_offset = token_rec.spans[0][0]
        sentence_end = token_rec.spans[-1][1]
        sentence_str = fulltext[fulltext_offset:sentence_end]

        refined_entities: list[RefinedEntity] = []
        for entity in pred_rec.entity:
            fulltext_spans = recover_char_spans(entity.indices, token_rec.spans)
            if not fulltext_spans:
                continue

            sentence_spans = [
                (s - fulltext_offset, e - fulltext_offset)
                for s, e in fulltext_spans
            ]
            entity_text = get_entity_surface_text(sentence_str, sentence_spans)

            if not entity_text.strip():
                continue

            key = get_majority_vote_key(pmid, entity_text)
            refined_type = majority_types.get(pmid, {}).get(key, entity.type)

            refined_entities.append(RefinedEntity(
                text_str=entity_text,
                sentence_spans=sentence_spans,
                type=refined_type,
            ))

        postprocess_records.append(PostprocessRecord(
            id=token_rec.id,
            pmid=pmid,
            sentence_str=sentence_str,
            fulltext_offset=fulltext_offset,
            entity=refined_entities,
        ))

    # Write atomically
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_path.parent, suffix=".jsonl.tmp")
    try:
        os.close(tmp_fd)
        save_jsonl([r.model_dump() for r in postprocess_records], Path(tmp_path_str))
        os.replace(tmp_path_str, output_path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise

    LOGGER.info(
        "%s: %d records written (%d token_records, %d pred_records).",
        output_path.name, len(postprocess_records), len(token_records), len(pred_records),
    )
    return len(postprocess_records)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    output_path: Path,
    input_files: list[str],
    input_hash: str,
    record_count: int,
) -> None:
    manifest = StageManifest(
        stage="step06_postprocess",
        input_files=input_files,
        input_hash=input_hash,
        record_count=record_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def main() -> None:
    global LOGGER

    parser = argparse.ArgumentParser(
        description="Step 06: join predictions with token spans and apply majority-vote type refinement."
    )
    parser.add_argument(
        "--tokens-dir", required=True,
        help="Directory containing step03 TokenRecord *.jsonl files (for spans and pmid).",
    )
    parser.add_argument(
        "--pred-dir", required=True,
        help="Directory containing step05 PredictRecord *.jsonl files.",
    )
    parser.add_argument(
        "--fulltext-dir", required=True,
        help="Directory containing step01 IngestRecord *.jsonl files (for fulltext strings).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write PostprocessRecord *.jsonl files.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = get_logger("step06_postprocess", log_dir=output_dir / "logs")

    tokens_dir = Path(args.tokens_dir)
    pred_dir = Path(args.pred_dir)
    fulltext_dir = Path(args.fulltext_dir)

    token_files = sorted(tokens_dir.glob("*.jsonl"))
    if not token_files:
        LOGGER.warning("No *.jsonl files found in %s", tokens_dir)
        return

    LOGGER.info("Found %d token files.", len(token_files))

    total_records = 0
    total_skipped = 0

    for token_file in token_files:
        stem = token_file.stem
        pred_file = pred_dir / f"{stem}.jsonl"
        fulltext_file = fulltext_dir / f"{stem}.jsonl"
        output_file = output_dir / f"{stem}.jsonl"

        if not pred_file.exists():
            LOGGER.warning("No prediction file for %s (expected %s) — skipping.", stem, pred_file.name)
            continue

        if not fulltext_file.exists():
            LOGGER.warning("No fulltext file for %s (expected %s) — skipping.", stem, fulltext_file.name)
            continue

        # Build pmid → fulltext map from step01 IngestRecord JSONL
        pmid_text_map: dict[str, str] = {}
        for raw in load_jsonl(fulltext_file):
            pmid = raw.get("pmid", "")
            fulltext = raw.get("fulltext", "")
            if pmid and fulltext:
                pmid_text_map[pmid] = fulltext

        try:
            n = postprocess_file(
                token_path=token_file,
                pred_path=pred_file,
                output_path=output_file,
                pmid_text_map=pmid_text_map,
                force=args.force,
            )

            if n == -1:
                total_skipped += 1
                continue

            input_hash = _file_sha256(token_file)
            _write_manifest(
                output_path=output_file,
                input_files=[token_file.name, pred_file.name],
                input_hash=input_hash,
                record_count=n,
            )
            total_records += n

        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", stem, exc, exc_info=True)

    LOGGER.info(
        "Done. %d files processed (%d skipped), %d total records written.",
        len(token_files) - total_skipped, total_skipped, total_records,
    )


if __name__ == "__main__":
    main()
