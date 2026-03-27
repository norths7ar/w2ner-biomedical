# =============================================================================
# pipeline/_utils.py
#
# PURPOSE
#   Shared utilities for pipeline steps: SHA-256 hashing, StageManifest
#   writing, and the base argparse parser.  Extracted here to eliminate
#   the identical _file_sha256 / _write_manifest helpers that were
#   duplicated across steps 01–03, 05, and 06.
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..specs.schemas import StageManifest


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_stage_manifest(
    output_path: Path,
    stage: str,
    input_files: list[str],
    input_hash: str,
    record_count: int,
) -> None:
    """Write a StageManifest sidecar as {output_path}.meta.json.

    Parameters
    ----------
    output_path:
        Path of the JSONL file this manifest describes.
    stage:
        Human-readable step name, e.g. "step01_ingest".
    input_files:
        Basenames of input files consumed by this step.
    input_hash:
        SHA-256 hex digest of the primary input file.
    record_count:
        Number of records written to output_path.
    """
    manifest = StageManifest(
        stage=stage,
        input_files=input_files,
        input_hash=input_hash,
        record_count=record_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def build_base_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-loaded with arguments common to all steps.

    Common arguments added here:
      --output-dir   (required) directory to write output files
      --force        overwrite existing output files

    Each step adds its own step-specific arguments after calling this
    function::

        parser = build_base_parser("Step 01: ingest annotation JSON files.")
        parser.add_argument("--input-dir", required=True, ...)
        args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    return parser
