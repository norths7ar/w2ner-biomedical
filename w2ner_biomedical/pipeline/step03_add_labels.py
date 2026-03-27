# =============================================================================
# pipeline/step03_add_labels.py
#
# PURPOSE
#   Assign abstract-level NER annotations to sentence-chunk TokenRecords by
#   resolving character-level span fragments to word-level indices.
#   Emits updated TokenRecords (with ner field populated) as JSONL.
#   Does NOT touch the model config — that is step04_finalize_config.py.
#
# CORRESPONDS TO
#   train05_add_labels.py  (original W2NER — label assignment portion only)
#
# KEY DESIGN CHANGES
#   - Config rewrite (entity_types, label_num) has been moved entirely to
#     step04_finalize_config.py.  This step is now pure data transformation:
#     it reads TokenRecords and annotation JSON, writes TokenRecords with ner
#     populated, and nothing else.  The original train05 mixed data processing
#     with model-config mutation, which caused the partial-run config
#     corruption bug (Bug H) and made the step non-idempotent.
#
#   - Label-centric assignment: iterates labels first, finds the unique
#     sentence whose span range contains all sub-spans, then resolves.
#     This makes it straightforward to count dropped entities (those that
#     no sentence claims) rather than scattering the count across the
#     sentence-first loop in the original.
#
#   - Discontinuous entity encoding: each fragment's [si, ei] range is
#     unioned into a single index set per entity, preserving the original
#     range(si, ei+1) logic.  Sub-span boundaries are also stored verbatim
#     in NEREntry.fragments for round-trip auditability.
#
#   - Entity alignment drop rate measured explicitly via Guard 2:
#     If more than MAX_DROP_RATE of entities are dropped the step raises
#     rather than completing silently with degraded labels.
#
#   - Label normalisation and model-filter logic driven by LabelSpec
#     (specs/label_spec.json), not by hardcoded conditionals.  Unknown
#     types after normalisation raise ValueError instead of silently
#     producing wrong label IDs.
#
# BUGS ADDRESSED
#   [CRITICAL]  Discontinuous entity encoding via range(si, ei+1):
#               The range-union logic is preserved per fragment, not
#               over the whole entity extent.  Sub-spans are stored in
#               NEREntry.fragments for auditability.
#
#   [MEDIUM]    Config clobbered by partial train05 run (Bug H):
#               Eliminated — config rewrite is not done here.
#
#   [HIGH]      Unicode normalisation shifts char offsets (Bug B):
#               Reads IngestRecord.fulltext (normalised in step01) and
#               TokenRecord.spans (offsets computed against the same
#               normalised string in step02).  No second normalisation.
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from myutils import load_json, load_jsonl, save_jsonl, get_logger

from ..specs.schemas import TokenRecord, NEREntry, SubSpan, LabelSpec, StageManifest
from ..guards.validators import check_entity_alignment_rate, check_record_count_parity

LOGGER: logging.Logger = logging.getLogger(__name__)


def normalize_label_type(raw: str, spec: LabelSpec) -> str:
    """Apply alias normalisation from label_spec.json.

    Checks spec.aliases first; falls back to raw if absent.  Raises
    ValueError if the result is not in spec.entity_types — unknown types
    are never silently mapped to label 0 as they were in the original.
    """
    canonical = spec.aliases.get(raw, raw)
    if canonical not in spec.entity_types:
        raise ValueError(
            f"Unknown entity type {raw!r} (normalised: {canonical!r}). "
            f"Add an alias or add {canonical!r} to entity_types in label_spec.json."
        )
    return canonical


def filter_label_by_model(label_type: str, model_suffix: str | None, spec: LabelSpec) -> bool:
    """Return True if this label type should be included for this model variant.

    If model_suffix is None, all types are included.
    If model_suffix is not in spec.model_filters, all types are included
    (with a warning so the mismatch is visible).
    Otherwise checks the filter's 'include' list.
    """
    if model_suffix is None:
        return True
    filter_spec = spec.model_filters.get(model_suffix)
    if filter_spec is None:
        LOGGER.warning(
            "model_suffix %r not found in spec.model_filters; including all types.",
            model_suffix,
        )
        return True
    include = filter_spec.get("include")
    if include is None:
        return True
    return label_type in include


def resolve_fragments_to_indices(
    spans_list: list[list[int]],
    start_to_idx: dict[int, int],
    end_to_idx: dict[int, int],
) -> tuple[list[int], list[SubSpan], bool]:
    """Resolve char-level sub-span fragments to word indices within a sentence.

    For each fragment [s, e]:
      - si = start_to_idx[s]  (index of word whose char start == s)
      - ei = end_to_idx[e]    (index of word whose char end == e)
      - add range(si, ei+1) to the index set (all words in this fragment)

    Returns (sorted_indices, subspan_list, success).
    success=False if any fragment cannot be aligned (missing key, or si > ei).
    The caller should count a False return as one dropped entity.

    NOTE: range(si, ei+1) is the same logic as the original train05, which is
    correct for contiguous sub-spans.  Discontinuous entities are handled
    because each fragment is resolved independently and their index ranges
    are unioned — NOT filled between fragments.
    """
    indices_set: set[int] = set()
    fragments: list[SubSpan] = []

    for s, e in spans_list:
        si = start_to_idx.get(s)
        ei = end_to_idx.get(e)
        if si is None or ei is None or si > ei:
            return [], [], False
        indices_set.update(range(si, ei + 1))
        fragments.append(SubSpan(start_char=s, end_char=e))

    return sorted(indices_set), fragments, True


def assign_labels_to_document(
    sentences: list[TokenRecord],
    labels: list,
    model_suffix: str | None,
    spec: LabelSpec,
) -> tuple[list[TokenRecord], int, int]:
    """Assign annotation labels to all TokenRecords for one document.

    Iterates labels first.  For each label, finds the unique sentence whose
    span range fully contains all sub-spans, then resolves to word indices.

    Returns (sentences, n_assigned, n_dropped).
    n_dropped: entities that no sentence could claim (span-crossing) or where
               fragment alignment failed.  Type-filtered entities are NOT counted.
    """
    # Pre-build per-sentence range and lookup structures
    sent_starts: list[int] = []
    sent_ends: list[int] = []
    start_to_idx_list: list[dict[int, int]] = []
    end_to_idx_list: list[dict[int, int]] = []

    for sent in sentences:
        if sent.spans:
            sent_starts.append(sent.spans[0][0])
            sent_ends.append(sent.spans[-1][1])
        else:
            sent_starts.append(-1)
            sent_ends.append(-1)
        start_to_idx_list.append({s: i for i, (s, _) in enumerate(sent.spans)})
        end_to_idx_list.append({e: i for i, (_, e) in enumerate(sent.spans)})

    n_assigned = 0
    n_dropped = 0

    for label in labels:
        spans_list, raw_type = label

        # Normalise type — skip with debug if unknown (not counted as dropped)
        try:
            label_type = normalize_label_type(raw_type, spec)
        except ValueError as exc:
            LOGGER.debug("Skipping entity with unknown type: %s", exc)
            continue

        # Model filter — excluded types are not counted as dropped
        if not filter_label_by_model(label_type, model_suffix, spec):
            continue

        # Find the sentence that fully contains all sub-spans
        matched_idx: int | None = None
        for si_idx, (ss, se) in enumerate(zip(sent_starts, sent_ends)):
            if ss < 0:
                continue
            if all(ss <= s and e <= se for s, e in spans_list):
                matched_idx = si_idx
                break

        if matched_idx is None:
            n_dropped += 1
            LOGGER.debug(
                "Entity type=%r spans=%s: no sentence contains all spans — dropped.",
                label_type, spans_list,
            )
            continue

        # Resolve sub-span chars to word indices
        indices, fragments, ok = resolve_fragments_to_indices(
            spans_list,
            start_to_idx_list[matched_idx],
            end_to_idx_list[matched_idx],
        )
        if not ok or not indices:
            n_dropped += 1
            LOGGER.debug(
                "Entity type=%r spans=%s: fragment alignment failed in sentence %d — dropped.",
                label_type, spans_list, matched_idx,
            )
            continue

        sentences[matched_idx].ner.append(NEREntry(
            indices=indices,
            fragments=fragments,
            type=label_type,
        ))
        n_assigned += 1

    return sentences, n_assigned, n_dropped


# ---------------------------------------------------------------------------
# File-level processing and manifest writing
# ---------------------------------------------------------------------------

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
        stage="step03_add_labels",
        input_files=input_files,
        input_hash=input_hash,
        record_count=record_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def process_file(
    annotation_path: Path,
    tokens_path: Path,
    output_path: Path,
    spec: LabelSpec,
    model_suffix: str | None,
    force: bool,
) -> int:
    """Assign labels for one annotation file. Returns record count, or -1 if skipped."""
    if output_path.exists() and not force:
        LOGGER.info("Skipping %s (output exists; use --force to overwrite)", annotation_path.stem)
        return -1

    annotation_data: list[dict] = load_json(annotation_path)
    pmid_annotation_mapping: dict[str, dict] = {}
    for a in annotation_data:
        pmid = str(a.get("PMID") or a.get("pmid") or "").strip()
        if pmid:
            pmid_annotation_mapping[pmid] = a

    token_records_raw = list(load_jsonl(tokens_path))
    pmid_sentences: dict[str, list[TokenRecord]] = {}
    for raw in token_records_raw:
        rec = TokenRecord.model_validate(raw)
        pmid_sentences.setdefault(rec.pmid, []).append(rec)

    # Load step02 manifest for record count parity check
    prev_count: int | None = None
    step02_manifest = tokens_path.with_suffix(tokens_path.suffix + ".meta.json")
    if step02_manifest.exists():
        prev_data = json.loads(step02_manifest.read_text(encoding="utf-8"))
        prev_count = prev_data.get("record_count")

    all_updated: list[TokenRecord] = []
    total_assigned = 0
    total_dropped = 0
    total_entities = 0
    type_counter: Counter = Counter()

    for pmid, ann in pmid_annotation_mapping.items():
        sentences = pmid_sentences.get(pmid, [])
        labels = ann.get("label", [])
        total_entities += len(labels)

        updated, n_assigned, n_dropped = assign_labels_to_document(
            sentences, labels, model_suffix, spec,
        )
        all_updated.extend(updated)
        total_assigned += n_assigned
        total_dropped += n_dropped

        for sent in updated:
            for ner_entry in sent.ner:
                type_counter[ner_entry.type] += 1

    # Guard 1: TokenRecord count should not drop between step02 and step03
    check_record_count_parity(prev_count, len(all_updated), "step03_add_labels")

    # Guard 2: entity alignment drop rate
    check_entity_alignment_rate(total_dropped, total_entities, "step03_add_labels")

    # Write atomically
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_path.parent, suffix=".jsonl.tmp")
    try:
        os.close(tmp_fd)
        save_jsonl([r.model_dump() for r in all_updated], Path(tmp_path_str))
        os.replace(tmp_path_str, output_path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise

    input_hash = _file_sha256(annotation_path)
    _write_manifest(
        output_path=output_path,
        input_files=[annotation_path.name, tokens_path.name],
        input_hash=input_hash,
        record_count=len(all_updated),
    )

    LOGGER.info(
        "%s: %d sentences, %d entities assigned, %d dropped. Types: %s",
        annotation_path.name, len(all_updated), total_assigned, total_dropped,
        dict(type_counter.most_common()),
    )
    return len(all_updated)


def main() -> None:
    global LOGGER

    parser = argparse.ArgumentParser(
        description="Step 03: assign annotation labels to TokenRecord JSONL."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing raw annotation *.json files.",
    )
    parser.add_argument(
        "--tokens-dir", required=True,
        help="Directory containing step02 TokenRecord *.jsonl files.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write labelled TokenRecord *.jsonl files.",
    )
    parser.add_argument(
        "--spec", required=True,
        help="Path to label_spec.json (LabelSpec).",
    )
    parser.add_argument(
        "--model-suffix", default=None,
        help="Model filter key in spec.model_filters (e.g. '_bc5cdr', '_biored'). "
             "If omitted, all entity types are included.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = get_logger("step03_add_labels", log_dir=output_dir / "logs")

    spec = LabelSpec.model_validate(load_json(Path(args.spec)))
    LOGGER.info(
        "Loaded spec: %d entity types, model_suffix=%r",
        len(spec.entity_types), args.model_suffix,
    )

    input_dir = Path(args.input_dir)
    tokens_dir = Path(args.tokens_dir)

    annotation_files = sorted(input_dir.glob("*.json"))
    if not annotation_files:
        LOGGER.warning("No *.json files found in %s", input_dir)
        return

    LOGGER.info("Found %d annotation files.", len(annotation_files))

    total_records = 0
    total_skipped = 0

    for ann_file in annotation_files:
        tokens_file = tokens_dir / f"{ann_file.stem}.jsonl"
        if not tokens_file.exists():
            LOGGER.warning(
                "No tokenized file for %s (expected %s) — skipping.",
                ann_file.name, tokens_file.name,
            )
            continue

        output_file = output_dir / f"{ann_file.stem}.jsonl"
        try:
            n = process_file(
                annotation_path=ann_file,
                tokens_path=tokens_file,
                output_path=output_file,
                spec=spec,
                model_suffix=args.model_suffix,
                force=args.force,
            )
            if n == -1:
                total_skipped += 1
            else:
                total_records += n
        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", ann_file.name, exc, exc_info=True)

    LOGGER.info(
        "Done. %d files processed (%d skipped), %d total records written.",
        len(annotation_files) - total_skipped, total_skipped, total_records,
    )


if __name__ == "__main__":
    main()
