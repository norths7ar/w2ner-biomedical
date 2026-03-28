# =============================================================================
# pipeline/step04_finalize_config.py
#
# PURPOSE
#   Read the completed step03 JSONL output, derive the final entity type
#   set and label count, and write them into the model config JSON.
#   This is the ONLY step that mutates the config — all other steps are
#   read-only with respect to model configuration.
#
# CORRESPONDS TO
#   The config-rewrite tail of train05_add_labels.py (original W2NER).
#   In the original pipeline this happened as a side effect of label
#   assignment; here it is an explicit, separate step.
#
# KEY DESIGN CHANGES
#   - Separation of concerns: data processing (step03) and config mutation
#     (this step) are now independent.  Killing step03 mid-run no longer
#     corrupts the config (Bug H in original).
#
#   - entity_types in the config is derived from the authoritative
#     label_spec.json (in spec order), NOT sorted from observed data.
#     Observed types from step03 are only used for validation.
#
#   - The config receives entity_types filtered by --model-suffix
#     (e.g. '_bc5cdr' keeps only Chemical and Disease).  label_num is
#     recomputed as len(sentinels) + len(filtered_entity_types).
#
#   - Guard 3 (check_type_vocabulary_consistency) detects vocabulary
#     shrinkage: if types previously in the config no longer appear in
#     observed step03 data, the run raises unless --allow-type-removal
#     is passed.
#
#   - bert_hid_size in the config is validated against the actual hidden
#     size reported by BertConfig (config only — no weights are loaded).
#
# BUGS ADDRESSED
#   [MEDIUM]  Config clobbered by partial train05 run (Bug H):
#             Eliminated — this step is separate and idempotent.
#
#   [MEDIUM]  bert_hid_size not validated against loaded model:
#             validate_bert_hid_size loads BertConfig (not weights) and
#             asserts hidden_size == config["bert_hid_size"].
#
#   [CRITICAL] Unknown types reaching label2id.get(type, 0):
#              validate_types_against_spec raises if any observed type is
#              absent from spec.entity_types, so unknown types never
#              silently become background cells.
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from myutils import load_json, load_jsonl, get_logger

from ..specs.schemas import LabelSpec
from ..guards.validators import check_type_vocabulary_consistency

LOGGER: logging.Logger = logging.getLogger(__name__)


def load_observed_types(step03_output_dir: Path) -> set[str]:
    """Scan step03 JSONL output and collect every entity type that appears.

    Reads all *.jsonl files in step03_output_dir.  Each record's ner list
    contributes its type strings.  Returns a set of unique type strings.
    """
    observed: set[str] = set()
    for jsonl_path in sorted(step03_output_dir.glob("*.jsonl")):
        for record in load_jsonl(jsonl_path):
            for ner_entry in record.get("ner", []):
                t = ner_entry.get("type", "")
                if t:
                    observed.add(t)
    LOGGER.info(
        "Observed %d unique entity types in step03 output: %s",
        len(observed), sorted(observed),
    )
    return observed


def validate_types_against_spec(observed: set[str], spec: LabelSpec) -> None:
    """Assert that every observed type is in spec.entity_types.

    Raises ValueError listing all unknown types if any are found.
    This prevents unknown types from silently falling through to
    feature_builder's label2id.get(type, 0) and becoming background cells.
    """
    unknown = observed - set(spec.entity_types)
    if unknown:
        raise ValueError(
            f"Entity types in step03 output are not in label_spec.json: "
            f"{sorted(unknown)}. "
            f"Add an alias or add the type to entity_types in label_spec.json."
        )


def validate_bert_hid_size(config_dict: dict, cache_dir: str) -> None:
    """Load encoder config (not weights) and assert hidden_size matches bert_hid_size.

    Raises ValueError with a clear message if they disagree, before any
    training or prediction attempts are made.
    """
    from transformers import BertConfig

    model_name = config_dict["bert_name"]
    encoder_cfg = BertConfig.from_pretrained(model_name, cache_dir=cache_dir)
    actual = encoder_cfg.hidden_size
    expected = config_dict["bert_hid_size"]
    if actual != expected:
        raise ValueError(
            f"bert_hid_size mismatch: config has {expected} but "
            f"{model_name} reports hidden_size={actual}. "
            f"Update bert_hid_size in the config template to {actual}."
        )
    LOGGER.info("bert_hid_size validated: %d matches %s hidden_size.", actual, model_name)


def finalize_config(
    config_path: Path,
    spec: LabelSpec,
    step03_output_dir: Path,
    cache_dir: str,
    allow_type_removal: bool = False,
    model_suffix: str | None = None,
    output_path: Path | None = None,
    skip_bert_check: bool = False,
) -> None:
    """Validate and inject entity_types + label_num into the model config.

    Steps:
      1. Load raw config dict from config_path.
      2. Collect observed types from step03 output.
      3. validate_types_against_spec — hard error on unknown types.
      4. Guard 3 — detect vocabulary shrinkage vs. existing config.
      5. Optionally validate bert_hid_size (skippable for offline tests).
      6. Build filtered entity_types from spec (spec order, not data order).
      7. Write entity_types and label_num into the config dict.
      8. Validate full config with ModelConfig.
      9. Write atomically to output_path (defaults to config_path).
    """
    from ..model.model_config import ModelConfig

    config_dict: dict = load_json(config_path)

    # Step 2: what types did step03 actually emit?
    observed_types = load_observed_types(step03_output_dir)

    # Step 3: all observed types must be in spec
    validate_types_against_spec(observed_types, spec)

    # Step 4: Guard 3 — warn/error if previously configured types have disappeared
    existing_entity_types: list[str] = config_dict.get("entity_types", [])
    check_type_vocabulary_consistency(
        observed_types=observed_types,
        existing_config_types=existing_entity_types,
        stage="step04_finalize_config",
        allow_removal=allow_type_removal,
    )

    # Step 5: validate bert_hid_size against the HuggingFace encoder config
    if not skip_bert_check:
        validate_bert_hid_size(config_dict, cache_dir)

    # Step 6: build filtered entity_types from spec (spec-order, not data-order)
    if model_suffix is not None:
        filter_spec = spec.model_filters.get(model_suffix)
        if filter_spec is None:
            LOGGER.warning(
                "model_suffix %r not found in spec.model_filters; using all entity types.",
                model_suffix,
            )
            entity_types = list(spec.entity_types)
        else:
            include = filter_spec.get("include", list(spec.entity_types))
            # Preserve spec order, keep only included types
            entity_types = [t for t in spec.entity_types if t in include]
    else:
        entity_types = list(spec.entity_types)

    label_num = len(spec.sentinels) + len(entity_types)

    LOGGER.info(
        "Writing entity_types (%d types, label_num=%d): %s",
        len(entity_types), label_num, entity_types,
    )

    # Step 7: inject into config dict
    config_dict["entity_types"] = entity_types
    config_dict["label_num"] = label_num

    # Step 8: validate with ModelConfig (catches shape mismatches before training)
    try:
        ModelConfig.model_validate(config_dict)
    except Exception as exc:
        raise ValueError(
            f"Config validation failed after injecting entity_types: {exc}"
        ) from exc

    # Step 9: write atomically
    if output_path is None:
        output_path = config_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_path.parent, suffix=".json.tmp")
    try:
        os.close(tmp_fd)
        Path(tmp_path_str).write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path_str, output_path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise

    LOGGER.info("Finalized config written to %s", output_path)


def main() -> None:
    global LOGGER

    parser = argparse.ArgumentParser(
        description="Step 04: inject entity_types and label_num into model config."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to model config JSON template (e.g. configs/biored_base.json).",
    )
    parser.add_argument(
        "--spec", required=True,
        help="Path to label_spec.json (LabelSpec).",
    )
    parser.add_argument(
        "--step03-dir", required=True,
        help="Directory containing step03 labelled TokenRecord *.jsonl files.",
    )
    parser.add_argument(
        "--output-config", default=None,
        help="Path to write the finalized config JSON. "
             "Defaults to overwriting --config in place.",
    )
    parser.add_argument(
        "--cache-dir", default="cache",
        help="HuggingFace model cache directory (for bert_hid_size validation).",
    )
    parser.add_argument(
        "--model-suffix", default=None,
        help="Model filter key in spec.model_filters (e.g. '_bc5cdr'). "
             "Filters entity_types to only the types in that filter's include list.",
    )
    parser.add_argument(
        "--allow-type-removal", action="store_true",
        help="Allow entity types to disappear from config vs. previous run "
             "(passes allow_removal=True to Guard 3).",
    )
    parser.add_argument(
        "--skip-bert-check", action="store_true",
        help="Skip bert_hid_size validation (useful in offline / CI environments "
             "where the HuggingFace hub is not reachable).",
    )
    args = parser.parse_args()

    # Output dir is determined by output_config path; fall back to config's dir for logs
    config_path = Path(args.config)
    output_path = Path(args.output_config) if args.output_config else config_path
    log_dir = output_path.parent / "logs"

    LOGGER = get_logger("step04_finalize_config", log_dir=log_dir)

    spec = LabelSpec.model_validate(load_json(Path(args.spec)))
    LOGGER.info(
        "Loaded spec: %d entity types, model_suffix=%r",
        len(spec.entity_types), args.model_suffix,
    )

    finalize_config(
        config_path=config_path,
        spec=spec,
        step03_output_dir=Path(args.step03_dir),
        cache_dir=args.cache_dir,
        allow_type_removal=args.allow_type_removal,
        model_suffix=args.model_suffix,
        output_path=output_path,
        skip_bert_check=args.skip_bert_check,
    )

    LOGGER.info("step04_finalize_config complete.")


if __name__ == "__main__":
    main()
