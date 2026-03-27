# =============================================================================
# pipeline/step05_predict.py
#
# PURPOSE
#   Run the trained W2NER model in inference mode over step03 TokenRecords.
#   Emits one PredictRecord per sentence chunk as JSONL.  Each PredictRecord
#   carries the TokenRecord.id so that step06_postprocess can join on ID
#   rather than on list position (fixes Bug D root cause).
#
# CORRESPONDS TO
#   step05_predict.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - batch_size and num_workers are CLI-configurable (override config);
#     originals were hardcoded at 8 (Bug I).
#
#   - label2id is loaded from model-dir/label2id.json and validated against
#     model.predictor.linear.out_features via Guard 5 before any inference.
#     A mismatch raises immediately rather than producing silently wrong
#     argmax outputs.
#
#   - output_hidden_states is set from config.use_bert_last_4_layers, same
#     as train.py.  The original always passed output_hidden_states=True,
#     wasting memory when use_bert_last_4_layers=False.
#
#   - Each output file is written atomically (temp + rename) so a killed
#     run never leaves a partial file that looks complete.
#
#   - PredictRecord carries id (= TokenRecord.id) for downstream ID-keyed
#     joining in step06 (Guard 4 prerequisite).
#
#   - A StageManifest is written alongside each output file.
#
# BUGS ADDRESSED
#   [HIGH]   zip truncation in postprocess assumes step05 never fails (Bug D):
#            PredictRecord.id matches TokenRecord.id; step06 joins on id,
#            not position.
#
#   [LOW]    Hardcoded batch_size=8 / num_workers=8 (Bug I):
#            Both are CLI-configurable with config defaults.
#
#   [MEDIUM] label vocabulary mismatch produces silent wrong argmax outputs:
#            Guard 5 validates head dim == len(label2id) at load time.
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from myutils import load_json, load_jsonl, save_jsonl, get_logger

from ..specs.schemas import TokenRecord, PredictRecord, PredictedEntity, StageManifest
from ..model.model_config import ModelConfig
from ..model.ner_model import NERModel
from ..model.decoding import decode_grid
from ..data.feature_builder import build_dis2idx, make_feature_converter
from ..data.collate import NERDataset, make_ner_collate_fn
from ..guards.validators import check_label_vocab_consistency, check_record_count_parity

LOGGER: logging.Logger = logging.getLogger(__name__)


def load_model(
    config: ModelConfig,
    model_dir: Path,
    cache_dir: str,
    device: torch.device,
) -> tuple[object, NERModel]:
    """Load tokenizer and NERModel; validate label vocabulary size.

    Loads encoder with output_hidden_states=config.use_bert_last_4_layers
    (matching train.py) to avoid loading the hidden state stack when it is
    not needed.

    Raises ValueError (via Guard 5) if model output head dimension does not
    match the label2id vocabulary loaded from model_dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir=cache_dir)

    encoder = AutoModel.from_pretrained(
        config.bert_name,
        cache_dir=cache_dir,
        output_hidden_states=config.use_bert_last_4_layers,
    ).to(device)

    model = NERModel(config=config, encoder=encoder).to(device)

    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found in {model_dir}")

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    LOGGER.info("Loaded model weights from %s", model_path)

    return tokenizer, model


def validate_label_vocab(model: NERModel, label2id: dict[str, int]) -> None:
    """Assert output head dimension == len(label2id) via Guard 5."""
    head_dim = model.predictor.linear.out_features
    check_label_vocab_consistency(head_dim, label2id, stage="step05_predict")


def predict_file(
    input_path: Path,
    output_path: Path,
    model: NERModel,
    tokenizer,
    id2label: dict[int, str],
    label2id: dict[str, int],
    dis2idx: dict[int, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> int:
    """Run inference over one TokenRecord JSONL file.

    Returns the number of PredictRecords written, or -1 if skipped.

    Each output PredictRecord carries the source TokenRecord.id so that
    step06_postprocess can join on ID rather than position (Bug D fix).
    Output is written atomically via temp-file rename.
    """
    if output_path.exists() and not force:
        LOGGER.info("Skipping %s (output exists; use --force to overwrite)", input_path.stem)
        return -1

    token_records_raw: list[dict] = list(load_jsonl(input_path))
    if not token_records_raw:
        LOGGER.warning("%s: no records found, writing empty output.", input_path.name)
        _write_output([], output_path)
        return 0

    feature_converter = make_feature_converter(tokenizer, dis2idx, label2id)
    dataset = NERDataset(token_records_raw, feature_converter, with_labels=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_ner_collate_fn(with_labels=False),
        num_workers=num_workers,
    )

    model.eval()
    all_decoded: list[dict] = []

    with torch.inference_mode():
        for batch in dataloader:
            bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, tokens_list = batch

            bert_inputs = {k: v.to(device) for k, v in bert_inputs.items()}
            grid_mask2d = grid_mask2d.to(device)
            dist_inputs = dist_inputs.to(device)
            pieces2word = pieces2word.to(device)
            sent_length = sent_length.to(device)

            logits = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            logits = torch.argmax(logits, dim=-1).detach().cpu().numpy()

            decoded = decode_grid(
                logits=logits,
                length_list=sent_length.cpu().tolist(),
                tokens_list=tokens_list,
                id2label=id2label,
                logger=LOGGER,
            )
            all_decoded.extend(decoded)

    # Build PredictRecords — zip with originals (shuffle=False preserves order)
    if len(all_decoded) != len(token_records_raw):
        raise ValueError(
            f"Decoded {len(all_decoded)} results from {len(token_records_raw)} input records. "
            f"DataLoader order invariant violated."
        )

    predict_records: list[PredictRecord] = []
    for raw, decoded in zip(token_records_raw, all_decoded):
        predict_records.append(PredictRecord(
            id=raw["id"],
            pmid=raw["pmid"],
            sentence=decoded["sentence"],
            entity=[PredictedEntity(**e) for e in decoded["entity"]],
        ))

    _write_output([r.model_dump() for r in predict_records], output_path)
    LOGGER.info("%s: %d records written.", output_path.name, len(predict_records))
    return len(predict_records)


def _write_output(records: list[dict], output_path: Path) -> None:
    """Write records as JSONL atomically via temp-file rename."""
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_path.parent, suffix=".jsonl.tmp")
    try:
        os.close(tmp_fd)
        save_jsonl(records, Path(tmp_path_str))
        os.replace(tmp_path_str, output_path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(output_path: Path, input_files: list[str], input_hash: str, record_count: int) -> None:
    manifest = StageManifest(
        stage="step05_predict",
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
        description="Step 05: run W2NER inference over step03 TokenRecord JSONL."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing step03 labelled TokenRecord *.jsonl files.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write PredictRecord *.jsonl files.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to finalized model config JSON (entity_types and label_num must be present).",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing model.pt and label2id.json (written by train.py).",
    )
    parser.add_argument(
        "--cache-dir", default="cache",
        help="HuggingFace model cache directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Inference batch size. Defaults to config.batch_size.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes (default: 0 — main process only).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device string, e.g. 'cuda:0' or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = get_logger("step05_predict", log_dir=output_dir / "logs")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # --- Load config and vocabulary ---
    config = ModelConfig.model_validate(load_json(Path(args.config)))
    model_dir = Path(args.model_dir)

    label2id_path = model_dir / "label2id.json"
    if not label2id_path.exists():
        raise FileNotFoundError(f"label2id.json not found in {model_dir}")
    label2id: dict[str, int] = load_json(label2id_path)
    id2label: dict[int, str] = {v: k for k, v in label2id.items()}

    LOGGER.info("Loaded label2id (%d labels) from %s", len(label2id), label2id_path)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # --- Load model and validate label vocab (Guard 5) ---
    tokenizer, model = load_model(config, model_dir, args.cache_dir, device)
    validate_label_vocab(model, label2id)
    LOGGER.info("Label vocabulary validated: %d classes.", len(label2id))

    dis2idx = build_dis2idx()
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    LOGGER.info("batch_size=%d, num_workers=%d", batch_size, args.num_workers)

    # --- Run inference ---
    input_dir = Path(args.input_dir)
    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        LOGGER.warning("No *.jsonl files found in %s", input_dir)
        return

    LOGGER.info("Found %d input files.", len(input_files))

    total_records = 0
    total_skipped = 0

    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}.jsonl"
        try:
            n = predict_file(
                input_path=input_file,
                output_path=output_file,
                model=model,
                tokenizer=tokenizer,
                id2label=id2label,
                label2id=label2id,
                dis2idx=dis2idx,
                device=device,
                batch_size=batch_size,
                num_workers=args.num_workers,
                force=args.force,
            )

            if n == -1:
                total_skipped += 1
                continue

            # Guard 1: token record count should be preserved through prediction
            prev_count: int | None = None
            step03_manifest = input_file.with_suffix(input_file.suffix + ".meta.json")
            if step03_manifest.exists():
                prev_data = json.loads(step03_manifest.read_text(encoding="utf-8"))
                prev_count = prev_data.get("record_count")
            check_record_count_parity(prev_count, n, "step05_predict")

            input_hash = _file_sha256(input_file)
            _write_manifest(
                output_path=output_file,
                input_files=[input_file.name],
                input_hash=input_hash,
                record_count=n,
            )
            total_records += n

        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", input_file.name, exc, exc_info=True)

    LOGGER.info(
        "Done. %d files processed (%d skipped), %d total records written.",
        len(input_files) - total_skipped, total_skipped, total_records,
    )


if __name__ == "__main__":
    main()
