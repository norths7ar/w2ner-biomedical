# =============================================================================
# model/train.py
#
# PURPOSE
#   Entry point for model training.  Orchestrates data loading, model
#   construction, class-weight computation, and the Trainer.fit() call.
#   Contains no training logic of its own — all behaviour lives in Trainer.
#
# CORRESPONDS TO
#   train06_train_model.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - Vocabulary loaded from label_spec.json, not from config entity_types:
#     The original derived label2id from config["entity_types"], which was
#     written by train05 at runtime and therefore changed whenever the
#     annotation data changed.  This version loads LabelSpec from the
#     committed specs/label_spec.json and calls spec.label2id() to get a
#     reproducible, order-stable mapping.  The config still carries
#     entity_types (written by step04_finalize_config.py) but this file
#     treats the spec as authoritative and validates that the two agree at
#     startup via check_label_vocab_consistency().
#
#   - Class weights computed before training begins:
#     A single pass over the training data counts entity labels in grid
#     cells and passes the resulting weight tensor to Trainer.__init__.
#     Unknown entity types have already been caught by step04; if one
#     somehow reaches here, label2id.get(type, 0) maps it to background
#     and a WARNING is emitted (not a silent corruption).
#
#   - output_hidden_states gated on use_bert_last_4_layers:
#     The encoder is loaded here with output_hidden_states set correctly
#     before being handed to NERModel, so the model class itself does not
#     need to know where the encoder came from.
#
#   - Validation split is optional but supported:
#     If --val-dir is provided, a validation DataLoader is built and passed
#     to Trainer.fit().  Without it, the original fixed-epoch / last-
#     checkpoint behaviour is preserved.
#
#   - Saved artifacts include label_spec.json:
#     model.pt, label2id.json, and a copy of label_spec.json are all saved
#     together in the output directory.  step05_predict.py loads all three
#     and validates consistency at load time.
#
# BUGS ADDRESSED
#   No new bugs are introduced here.  This file wires together fixes
#   implemented in ner_model.py, trainer.py, and data/feature_builder.py.
# =============================================================================

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModel
from myutils import load_json, save_json, load_jsonl, get_logger

from .model_config import ModelConfig
from .constants import NNW_LABEL
from .ner_model import NERModel
from .trainer import Trainer, compute_class_weights
from ..specs.schemas import LabelSpec


# Module-level fallback — no handlers attached.  main() replaces this with a
# fully configured logger once output_dir is known, so all messages land in
# output_dir/logs/train.log as well as stdout.
LOGGER: logging.Logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across Python, NumPy, and PyTorch.

    cudnn.deterministic is False for production speed (fastest algorithm
    preferred); benchmark is False because input shapes vary across batches.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def check_label_vocab_consistency(
    config_entity_types: list[str],
    spec_entity_types: list[str],
) -> None:
    """Assert that the config and label_spec agree on entity types.

    Raises ValueError if they differ.  Discrepancies indicate that
    step04_finalize_config.py was not run after label_spec.json was edited,
    or that the config was edited manually without updating the spec.
    """
    if config_entity_types != spec_entity_types:
        config_set = set(config_entity_types)
        spec_set = set(spec_entity_types)
        in_config_not_spec = config_set - spec_set
        in_spec_not_config = spec_set - config_set
        raise ValueError(
            "Entity type vocabulary mismatch between config and label_spec.json.\n"
            f"  In config but not spec: {sorted(in_config_not_spec)}\n"
            f"  In spec but not config: {sorted(in_spec_not_config)}\n"
            "Run step04_finalize_config.py to reconcile."
        )


def load_encoder(config: ModelConfig, cache_dir: Path) -> torch.nn.Module:
    """Load the HuggingFace encoder with the correct output_hidden_states flag.

    output_hidden_states is only requested when use_bert_last_4_layers=True.
    In last-layer-only mode the hidden state stack is never read, so always
    requesting it wastes memory ([LOW] bug fix from ner_model.py).
    """
    return AutoModel.from_pretrained(
        config.bert_name,
        cache_dir=str(cache_dir),
        output_hidden_states=config.use_bert_last_4_layers,
    )


def load_training_data(input_dir: Path) -> list[dict]:
    """Load all JSONL files from input_dir into a flat list of records."""
    records: list[dict] = []
    for file in sorted(input_dir.glob("*.jsonl")):
        LOGGER.info("Loading training data: %s", file.name)
        records.extend(load_jsonl(file))
    LOGGER.info("Total training instances: %d", len(records))
    return records


def count_label_distribution(
    data: list[dict],
    label2id: dict[str, int],
    num_labels: int,
) -> dict[int, int]:
    """Count label ID occurrences across all NER entries in the training data.

    Only counts cells that will be included in the loss mask (upper-triangle
    NNW edges + THW cells) to ensure the class weights match the gradient
    distribution.  Background cells are counted as the total upper-triangle
    cells minus entity cells.

    Returns {label_id: count} for use by compute_class_weights().
    """

    counts: dict[int, int] = {i: 0 for i in range(num_labels)}

    for instance in data:
        sent_len = len(instance.get("sentence", []))
        if sent_len == 0:
            continue

        # Total upper-triangle cells (including diagonal) in this sentence
        upper_tri_cells = sent_len * (sent_len + 1) // 2
        entity_cells = 0

        for ner_entry in instance.get("ner", []):
            indices = ner_entry.get("indices", [])
            ent_type = ner_entry.get("type", "")
            type_id = label2id.get(ent_type, 0)

            if type_id == 0:
                LOGGER.warning(
                    "Unknown entity type %r in training data — mapped to background (label 0). "
                    "Run step04_finalize_config.py to catch this earlier.",
                    ent_type,
                )

            # NNW edges: len(indices)-1 upper-triangle cells
            n_nnw = max(0, len(indices) - 1)
            counts[NNW_LABEL] = counts.get(NNW_LABEL, 0) + n_nnw
            entity_cells += n_nnw

            # THW cell: 1 cell per entity
            counts[type_id] = counts.get(type_id, 0) + 1
            entity_cells += 1

        # Background: remaining upper-triangle cells
        bg_cells = max(0, upper_tri_cells - entity_cells)
        counts[0] = counts.get(0, 0) + bg_cells

    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train W2NER biomedical NER model")
    parser.add_argument("--config",     required=True,  help="Path to biored_base.json (ModelConfig)")
    parser.add_argument("--spec",       required=True,  help="Path to label_spec.json (LabelSpec)")
    parser.add_argument("--input-dir",  required=True,  help="Directory of training JSONL files (step03 output)")
    parser.add_argument("--output-dir", required=True,  help="Directory to save model.pt, label2id.json, etc.")
    parser.add_argument("--val-dir",    default=None,   help="Optional: directory of validation JSONL files")
    parser.add_argument("--cache-dir",  default="cache", help="HuggingFace model cache directory")
    parser.add_argument("--device",     default=None,   help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--patience",   type=int, default=0, help="Early stopping patience (0=disabled)")
    args = parser.parse_args()

    # Resolve output_dir first so the logger can write train.log there.
    # All subsequent LOGGER calls — including those inside helper functions
    # that reference the module-level LOGGER — will use this configured instance.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global LOGGER
    LOGGER = get_logger("train", log_dir=output_dir / "logs")

    # Silence tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Load config and spec ---
    config = ModelConfig.model_validate(load_json(Path(args.config)))
    spec = LabelSpec.model_validate(load_json(Path(args.spec)))

    # Validate that config and spec agree on entity types
    # (spec is authoritative; config is derived by step04_finalize_config.py)
    check_label_vocab_consistency(config.entity_types, spec.entity_types)

    set_seed(config.seed)

    label2id = spec.label2id()
    id2label = spec.id2label()

    LOGGER.info("label2id (%d labels): %s", len(label2id), label2id)

    # --- Device selection ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # --- Load encoder and model ---
    cache_dir = Path(args.cache_dir)
    tokenizer = BertTokenizer.from_pretrained(config.bert_name, cache_dir=str(cache_dir))
    encoder = load_encoder(config, cache_dir).to(device)
    model = NERModel(config=config, encoder=encoder).to(device)

    # --- Load training data ---
    train_data = load_training_data(Path(args.input_dir))
    if not train_data:
        raise ValueError(f"No training data found in {args.input_dir}")

    # --- Compute class weights to address ~68:1 imbalance ---
    label_counts = count_label_distribution(train_data, label2id, config.label_num)
    class_weights = compute_class_weights(label_counts, config.label_num)
    LOGGER.info("Class weights: %s", {id2label[i]: f"{w:.4f}" for i, w in enumerate(class_weights.tolist())})

    # --- Build DataLoaders ---
    from ..data.feature_builder import build_dis2idx, make_feature_converter
    from ..data.collate import NERDataset, make_ner_collate_fn

    dis2idx = build_dis2idx()
    feature_converter = make_feature_converter(tokenizer, dis2idx, label2id)

    train_dataset = NERDataset(train_data, feature_converter, with_labels=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_ner_collate_fn(with_labels=True),
        num_workers=4,
    )

    val_loader: DataLoader | None = None
    if args.val_dir is not None:
        val_data = load_training_data(Path(args.val_dir))
        val_dataset = NERDataset(val_data, feature_converter, with_labels=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=make_ner_collate_fn(with_labels=True),
            num_workers=4,
        )

    # --- Train ---
    save_path = output_dir / "model.pt"

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        len_dataset=len(train_dataset),
        class_weights=class_weights,
        patience=args.patience,
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        id2label=id2label,
        save_path=save_path,
        logger=LOGGER,
    )

    # --- Save artifacts ---
    save_json(label2id, output_dir / "label2id.json")

    # Save a copy of label_spec.json alongside the model so that
    # step05_predict.py can verify the vocabulary at load time.
    shutil.copy(args.spec, output_dir / "label_spec.json")

    LOGGER.info("Training complete.  Artifacts written to %s", output_dir)
    LOGGER.info("  model.pt       — model weights")
    LOGGER.info("  label2id.json  — label vocabulary")
    LOGGER.info("  label_spec.json — vocabulary spec (copy)")


if __name__ == "__main__":
    main()
