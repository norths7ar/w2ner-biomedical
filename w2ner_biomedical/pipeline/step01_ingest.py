# =============================================================================
# pipeline/step01_ingest.py
#
# PURPOSE
#   Ingest raw annotation JSON files (one per study/batch), normalise
#   unicode, concatenate title + abstract into a single canonical fulltext
#   string, and emit one IngestRecord per document as JSONL.
#   Also writes a StageManifest sidecar for cache-invalidation.
#
# CORRESPONDS TO
#   step02_generate_input.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - Unicode normalisation (NFKC + dash remapping) is still applied here,
#     but a length check catches any normalisation that deletes characters
#     (e.g. soft hyphens) and would therefore shift annotation offsets.
#     The offending document is skipped with a WARNING rather than silently
#     producing a misaligned output.
#   - The canonical fulltext string (title + " " + abstract) is computed
#     once here and stored in IngestRecord.fulltext.  Every downstream step
#     reads this field directly instead of re-joining independently, which
#     was the original source of the off-by-one title-separator bug.
#   - Duplicate PMID detection now raises a hard error rather than logging
#     a warning and silently keeping the first copy.
#   - Output is validated against the IngestRecord Pydantic schema before
#     writing, so schema violations are caught at the producing step rather
#     than propagating to the consuming step.
#   - --workers N enables document-level parallelism via ProcessPoolExecutor.
#     Each worker is stateless (no model to load), so this is purely I/O +
#     CPU normalisation work.
#
# BUGS ADDRESSED
#   [HIGH]  Unicode normalisation silently shifts char offsets (Bug B):
#           normalize_unicode raises ValueError if the output length differs
#           from the input length.  The caller skips the document and logs a
#           WARNING with the PMID so the problem is visible rather than
#           silently producing misaligned annotations downstream.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from myutils import load_json, save_jsonl, get_logger

from ..specs.schemas import IngestRecord
from ..guards.validators import check_record_count_parity
from ._utils import file_sha256, write_stage_manifest, build_base_parser

LOGGER: logging.Logger = logging.getLogger(__name__)

# Unicode characters remapped or deleted during normalisation.
# Dashes are replaced 1-for-1 with ASCII hyphen-minus so offsets are preserved.
# None-chars are deleted — any document containing them triggers a length
# mismatch and is skipped (Bug B guard).
_DASH_CHARS: frozenset[str] = frozenset({
    '\u2010',  # HYPHEN
    '\u2012',  # FIGURE DASH
    '\u2013',  # EN DASH
    '\u2014',  # EM DASH
    '\u2015',  # HORIZONTAL BAR
    '\u2212',  # MINUS SIGN
    '\u2500',  # BOX DRAWINGS LIGHT HORIZONTAL
})
_NONE_CHARS: frozenset[str] = frozenset({
    '\u00ad',  # SOFT HYPHEN
    '\u2029',  # PARAGRAPH SEPARATOR
})
_TRANSLATION_TABLE = str.maketrans(
    {c: '-' for c in _DASH_CHARS} | {c: '' for c in _NONE_CHARS}
)


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalisation and remap Unicode dashes and soft-hyphens.

    Returns the normalised string.

    Raises ValueError if the output length differs from the input length.
    A length change means at least one character was deleted (e.g. a soft
    hyphen via _NONE_CHARS, or a multi-codepoint ligature that NFKC
    decomposes differently).  Either case shifts all subsequent annotation
    character offsets and must be treated as an error rather than silently
    accepted.

    The caller (ingest_file) catches this ValueError, logs a WARNING with
    the PMID, and skips the document.
    """
    result = unicodedata.normalize('NFKC', text).translate(_TRANSLATION_TABLE)
    if len(result) != len(text):
        raise ValueError(
            f"Unicode normalisation changed string length from {len(text)} to "
            f"{len(result)}.  The text contains characters that expand or "
            f"contract under NFKC normalisation or are deleted by the "
            f"none-char filter (e.g. soft hyphens \\u00ad).  This would "
            f"shift annotation character offsets for all positions after the "
            f"affected character."
        )
    return result


def build_fulltext(title: str, abstract: str) -> str:
    """Produce the canonical fulltext string used as the shared coordinate origin.

    Rule: non-empty parts are joined with a single space.
    If only one part is present, fulltext equals that part.
    This is explicit and tested rather than re-derived independently in
    each downstream step.
    """
    return " ".join(filter(None, [title, abstract]))


def ingest_file(input_path: Path, deleted_pmids: set[str]) -> list[IngestRecord]:
    """Load one annotation JSON file and return a list of IngestRecord objects.

    Documents are skipped (with WARNING) on:
      - PMID in deleted_pmids
      - Empty title AND empty abstract
      - Unicode normalisation length mismatch (Bug B guard)

    Raises ValueError on:
      - Duplicate PMID within the same file (hard error: indicates a data
        preparation mistake that must be fixed before training)
    """
    data: list[dict] = load_json(input_path)
    records: list[IngestRecord] = []
    seen_pmids: set[str] = set()

    for item in data:
        # Support both key variants from different annotation sources
        pmid: str = (
            item.get("PMID") or item.get("pmid") or ""
        ).strip()
        if not pmid:
            LOGGER.warning("Skipping item with missing PMID in %s", input_path.name)
            continue

        if pmid in deleted_pmids:
            LOGGER.info("Skipping deleted PMID %s", pmid)
            continue

        if pmid in seen_pmids:
            raise ValueError(
                f"Duplicate PMID {pmid!r} within file {input_path.name}. "
                f"Deduplicate the annotation source before ingesting."
            )
        seen_pmids.add(pmid)

        raw_title: str = (
            item.get("articleTitle") or item.get("title") or ""
        ).strip()
        raw_abstract: str = (
            item.get("abstract") or item.get("text") or ""
        ).strip()

        if not raw_title and not raw_abstract:
            LOGGER.warning("Skipping PMID %s: both title and abstract are empty", pmid)
            continue

        # Normalise — skip document on length mismatch to avoid offset drift
        try:
            title = normalize_unicode(raw_title) if raw_title else ""
            abstract = normalize_unicode(raw_abstract) if raw_abstract else ""
        except ValueError as exc:
            LOGGER.warning(
                "Skipping PMID %s: unicode normalisation error in %s: %s",
                pmid, input_path.name, exc,
            )
            continue

        fulltext = build_fulltext(title, abstract)
        records.append(IngestRecord(
            pmid=pmid,
            title=title,
            abstract=abstract,
            fulltext=fulltext,
        ))

    return records


def _process_one_file(
    input_path: Path,
    output_dir: Path,
    deleted_pmids: set[str],
    force: bool,
) -> tuple[str, int]:
    """Ingest one JSON file and write the JSONL + manifest.

    Returns (stem, record_count).  Designed to be called from a worker
    process (ProcessPoolExecutor) as well as directly from the main process.
    """
    output_path = output_dir / f"{input_path.stem}.jsonl"

    if output_path.exists() and not force:
        LOGGER.info("Skipping %s (output exists; use --force to overwrite)", input_path.stem)
        return input_path.stem, -1  # -1 signals "skipped"

    records = ingest_file(input_path, deleted_pmids)

    # Guard 1: no-op (previous_count=None), but documents intent
    check_record_count_parity(None, len(records), "step01_ingest")

    # Write JSONL atomically
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".jsonl.tmp")
    try:
        os.close(tmp_fd)
        save_jsonl([r.model_dump() for r in records], Path(tmp_path))
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    input_hash = file_sha256(input_path)
    write_stage_manifest(
        output_path=output_path,
        stage="step01_ingest",
        input_files=[input_path.name],
        input_hash=input_hash,
        record_count=len(records),
    )

    LOGGER.info("Wrote %d records -> %s", len(records), output_path.name)
    return input_path.stem, len(records)


def main() -> None:
    global LOGGER

    parser = build_base_parser("Step 01: ingest annotation JSON files into IngestRecord JSONL.")
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing raw annotation *.json files.",
    )
    parser.add_argument(
        "--deleted-pmids", default=None,
        help="Path to a file with one PMID per line to exclude (optional).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger("step01_ingest", log_dir=output_dir / "logs")

    # Load deleted PMIDs allowlist
    deleted_pmids: set[str] = set()
    if args.deleted_pmids:
        deleted_path = Path(args.deleted_pmids)
        if deleted_path.exists():
            deleted_pmids = {
                line.strip() for line in deleted_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            LOGGER.info("Loaded %d deleted PMIDs from %s", len(deleted_pmids), deleted_path.name)
        else:
            LOGGER.warning("--deleted-pmids path not found: %s", deleted_path)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        LOGGER.warning("No *.json files found in %s", input_dir)
        return

    LOGGER.info("Found %d input files, workers=%d", len(input_files), args.workers)

    total_records = 0
    total_skipped = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_one_file, f, output_dir, deleted_pmids, args.force): f
                for f in input_files
            }
            for future in as_completed(futures):
                input_file = futures[future]
                try:
                    stem, count = future.result()
                    if count == -1:
                        total_skipped += 1
                    else:
                        total_records += count
                except Exception as exc:
                    LOGGER.error("Failed to process %s: %s", input_file.name, exc)
    else:
        for input_file in input_files:
            try:
                stem, count = _process_one_file(
                    input_file, output_dir, deleted_pmids, args.force
                )
                if count == -1:
                    total_skipped += 1
                else:
                    total_records += count
            except Exception as exc:
                LOGGER.error("Failed to process %s: %s", input_file.name, exc)

    LOGGER.info(
        "Done. %d files processed (%d skipped), %d total records written.",
        len(input_files) - total_skipped, total_skipped, total_records,
    )


if __name__ == "__main__":
    main()
