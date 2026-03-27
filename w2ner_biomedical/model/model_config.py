# =============================================================================
# model/model_config.py
#
# PURPOSE
#   Pydantic model for W2NER architecture and training hyperparameters.
#   Loaded from configs/biored_base.json (after it has been resolved by
#   step04_finalize_config.py, which injects label_num and entity_types).
#
# CORRESPONDS TO
#   core/model/model_config.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - label_num and entity_types are still required fields here:
#     The raw configs/biored_base.json template intentionally omits them
#     (they are injected by step04_finalize_config.py from label_spec.json).
#     The resolved config written by step04 IS a valid ModelConfig.
#     Nothing in this class changes; the requirement is enforced naturally.
#
#   - save_path / load_path validator softened:
#     The original raised ValueError when both were None.  In the new
#     pipeline, paths are always passed as CLI arguments to train.py and
#     step05_predict.py; the config JSON has both as null.  Raising here
#     would break on load.  The new validator allows both-None and emits
#     no error — the calling script is responsible for providing a path.
#
#   - Pydantic v2 style:
#     Inner class `class Config: extra = "forbid"` replaced with
#     `model_config = ConfigDict(extra="ignore")`.  Changed from "forbid"
#     to "ignore" so that JSON files with comment keys (e.g. "_comment",
#     "_TODO") can be loaded without preprocessing.  Real typos in field
#     names are caught at the required-field level rather than by the extra-
#     field check, which is sufficient for this use case.
# =============================================================================

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class ModelConfig(BaseModel):
    """Architecture and training hyperparameters for the W2NER model.

    All fields are required except save_path and load_path.
    label_num and entity_types must be present (injected by
    step04_finalize_config.py before this class is instantiated at
    training or inference time).
    """

    model_config = ConfigDict(extra="ignore")

    # -------------------------------------------------------------------------
    # BERT encoder
    # -------------------------------------------------------------------------
    bert_name: str                   # HuggingFace model identifier
    bert_hid_size: int               # hidden size of the encoder (e.g. 768)
    use_bert_last_4_layers: bool     # True = average last 4 layers; False = last only

    # -------------------------------------------------------------------------
    # Architecture dims
    # -------------------------------------------------------------------------
    lstm_hid_size: int               # BiLSTM output size (total, both directions)
    conv_hid_size: int               # dilated CNN channel width per dilation
    biaffine_size: int               # MLP projection size fed into biaffine scorer
    ffnn_hid_size: int               # MLP hidden size in the conv branch
    dist_emb_size: int               # distance embedding dim
    type_emb_size: int               # region-type embedding dim
    dilation: list[int]              # list of dilation factors for CNN layers

    # -------------------------------------------------------------------------
    # Dropout
    # -------------------------------------------------------------------------
    emb_dropout: float               # applied after max-pool word embeddings
    conv_dropout: float              # Dropout2d inside ConvolutionLayer
    out_dropout: float               # applied inside CoPredictor MLPs

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    epochs: int
    learning_rate: float             # non-BERT parameter learning rate
    bert_learning_rate: float        # BERT encoder learning rate (usually much smaller)
    weight_decay: float
    clip_grad_norm: float
    warm_factor: float               # fraction of total steps used for warmup
    seed: int
    batch_size: int

    # -------------------------------------------------------------------------
    # Vocabulary — injected by step04_finalize_config.py from label_spec.json
    # -------------------------------------------------------------------------
    label_num: int                   # total number of labels: len(sentinels) + len(entity_types)
    entity_types: list[str]          # ordered list of real entity types (no sentinels)

    # -------------------------------------------------------------------------
    # I/O paths — optional; callers pass these via CLI args, not config JSON
    # -------------------------------------------------------------------------
    load_path: Path | None = None
    save_path: Path | None = None

    @model_validator(mode="after")
    def _reconcile_paths(self) -> "ModelConfig":
        """Mirror save_path from load_path if only one is set.

        Both-None is allowed — the calling script supplies paths via CLI.
        The original validator raised when both were None, which broke on
        loading the raw template JSON (biored_base.json has both as null).
        """
        if self.save_path is None and self.load_path is not None:
            self.save_path = self.load_path
        elif self.load_path is None and self.save_path is not None:
            self.load_path = self.save_path
        # both-None: no-op, caller must provide paths explicitly
        return self

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------

    @property
    def n_dilation_layers(self) -> int:
        """Number of dilated CNN channels = conv_hid_size * len(dilation)."""
        return self.conv_hid_size * len(self.dilation)
