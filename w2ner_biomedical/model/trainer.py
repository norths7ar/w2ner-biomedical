# =============================================================================
# model/trainer.py
#
# PURPOSE
#   Training loop, optimiser construction, loss computation, and
#   (new) validation loop with checkpoint selection.  Separated from
#   the entry-point script (model/train.py) so it can be unit-tested.
#
# CORRESPONDS TO
#   core/model/trainer.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - Class imbalance handling:
#     The original used unweighted CrossEntropyLoss over all cells in the
#     grid mask.  The background label (0) dominates at roughly 68:1 over
#     all entity-related labels combined.  The trainer now accepts a
#     class_weights tensor (pre-computed from training label distribution)
#     and passes it to CrossEntropyLoss(weight=...).  The weight for label 0
#     is set to approximately 1/68 relative to the mean entity label weight,
#     capped at a configurable max_bg_weight to avoid instability.
#     Alternatively, the constructor accepts loss_fn=... so a focal loss or
#     other custom criterion can be injected without changing this file.
#
#   - Validation loop and early stopping:
#     The original trained for a fixed number of epochs and saved only the
#     last checkpoint.  This class now accepts an optional val_loader; if
#     provided, evaluate() is called after each epoch and the best
#     entity-level F1 checkpoint is saved.  Early stopping patience is
#     configurable (default: 0 = no early stopping, preserving original
#     behaviour).
#
#   - Upper-triangle NNW mask applied to loss:
#     The grid is logically upper-triangular for NNW edges (i < j) and
#     diagonal/sub-diagonal for THW edges.  The original loss was computed
#     over the full grid_mask2d without masking the lower-triangle, meaning
#     the model was penalised for predicting NNW=1 in cells that the decoder
#     never reads.  The loss now uses a combined mask: grid_mask2d AND the
#     upper-triangle, so lower-triangle NNW cells are excluded from
#     gradient updates.  THW cells (lower-triangle, label > 1) are still
#     included because the combined mask retains all cells where the gold
#     label is non-zero.
#
#   - Training metric vs. evaluation metric clarity:
#     The original logged macro-F1 over label indices (label-level), which
#     is not directly comparable to the entity-level F1 reported by the
#     offline evaluation tool.  This trainer logs both metrics with explicit
#     labels: "label-level macro F1" and "entity-level F1".
#
# BUGS ADDRESSED
#   [HIGH]   Class imbalance ~68:1 with no loss weighting:
#            Addressed via configurable class_weights tensor passed to
#            CrossEntropyLoss(weight=...) or an injected loss_fn.
#
#   [HIGH]   No validation loop or early stopping — fixed 20 epochs, saves
#            last checkpoint only:
#            Validation loop added; best entity-F1 checkpoint saved when
#            val_loader is provided.
#
#   [MEDIUM] Upper-triangle NNW mask missing from loss — model wastes
#            capacity on lower-triangle label=1 predictions:
#            Loss mask now combines grid_mask2d with upper-triangle
#            constraint; lower-triangle cells with label=0 are excluded
#            while THW cells (label > 1) are preserved.
#
#   [LOW]    Training-time metric is label-level macro F1, not entity-level
#            F1 — not directly comparable to eval metric:
#            Both metrics logged with explicit distinguishing labels.
# =============================================================================

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup

from .model_config import ModelConfig
from .decoding import decode_and_compare, cal_f1


def compute_class_weights(
    label_counts: dict[int, int],
    num_labels: int,
    bg_label: int = 0,
    max_bg_weight: float = 0.1,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Weights are normalised so the mean entity-label weight is 1.0.  The
    background label weight is then capped at max_bg_weight to prevent
    numerical instability when the background count is orders of magnitude
    larger than entity label counts.

    At a ~68:1 ratio the uncapped background weight would be ~0.015.
    max_bg_weight=0.1 provides a soft cap that keeps the background
    gradient contribution small but non-zero.

    Args:
        label_counts: {label_id: count} from count_label_distribution().
        num_labels:   Total number of labels (len(label2id)).
        bg_label:     ID of the background / no-entity label (default 0).
        max_bg_weight: Upper bound for the background label weight.

    Returns:
        Float tensor of shape [num_labels] for use as CrossEntropyLoss weight.
    """
    # Inverse-frequency weights: rare labels get higher weights
    raw = [1.0 / max(label_counts.get(i, 1), 1) for i in range(num_labels)]

    # Collect entity label weights (all except background) and normalise
    entity_weights = [w for i, w in enumerate(raw) if i != bg_label]
    if entity_weights:
        mean_entity_weight = sum(entity_weights) / len(entity_weights)
        raw = [w / mean_entity_weight for w in raw]

    # Cap the background label weight
    raw[bg_label] = min(raw[bg_label], max_bg_weight)

    return torch.FloatTensor(raw)


def build_loss_mask(grid_mask2d: torch.Tensor, grid_labels: torch.Tensor) -> torch.Tensor:
    """Build the boolean mask of cells to include in the training loss.

    W2NER grid layout (for an entity spanning words head..tail, head < tail):
      - grid_labels[head, tail] = NNW_LABEL (1)   upper triangle (head < tail)
      - grid_labels[tail, head] = type_id   (>1)  lower triangle (tail > head)
      - all other cells          = 0               background

    Three cell classes exist within grid_mask2d:
      1. Upper-triangle NNW cells  (label = 1, head < tail) — always included
      2. Lower-triangle THW cells  (label > 1, tail > head) — always included
      3. Background cells          (label = 0, anywhere)   — included in upper
                                                              triangle only

    Excluding lower-triangle background cells from the loss removes positions
    that the decoder never reads for NNW decisions, preventing the model from
    wasting capacity predicting label=0 in those structurally irrelevant cells.
    """
    B, L, _ = grid_mask2d.shape
    device = grid_mask2d.device

    # Upper-triangle mask including diagonal — covers all NNW cells and
    # upper-triangle background cells.
    upper = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device)
    ).unsqueeze(0).expand(B, -1, -1)                       # [B, L, L]

    # Lower-triangle cells: keep only those with a non-zero gold label
    # (i.e. THW cells).  Pure background lower cells (label=0) are excluded.
    lower = ~upper                                          # [B, L, L]
    lower_nonzero = lower & (grid_labels > 0)

    return grid_mask2d & (upper | lower_nonzero)


class Trainer:
    """Training and evaluation wrapper for NERModel."""

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: torch.device,
        len_dataset: int,
        class_weights: torch.Tensor | None = None,
        loss_fn: nn.Module | None = None,
        patience: int = 0,
    ) -> None:
        """Initialise the trainer.

        Args:
            model:         NERModel (already on device).
            config:        ModelConfig with architecture and training hyperparams.
            device:        torch.device to run on.
            len_dataset:   Number of training samples, for scheduler setup.
            class_weights: Optional [num_labels] weight tensor for
                           CrossEntropyLoss.  Computed by compute_class_weights().
                           Ignored if loss_fn is provided.
            loss_fn:       Optional custom loss module (e.g. focal loss).
                           If provided, class_weights is ignored.
            patience:      Early stopping patience in epochs.  0 = disabled.
        """
        self.model = model
        self.config = config
        self.device = device
        self.patience = patience
        self.best_f1 = 0.0
        self.patience_counter = 0

        # [HIGH bug fix] Build loss with class weights to address 68:1 imbalance.
        # If a custom loss_fn is injected, use it directly.
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            # Fallback: unweighted (original behaviour, preserved for
            # backward-compatibility when running without pre-computed weights)
            self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(len_dataset)

    def _build_optimizer(self) -> torch.optim.AdamW:
        """AdamW with separate learning rates for BERT and task-specific layers."""
        no_decay = ["bias", "LayerNorm.weight"]
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)

        param_groups = [
            {
                "params": [
                    p for n, p in self.model.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": self.config.bert_learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": self.config.bert_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": other_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
        ]
        return torch.optim.AdamW(param_groups)

    def _build_scheduler(self, len_dataset: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup + linear decay over all training steps."""
        steps_per_epoch = math.ceil(len_dataset / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(self.config.warm_factor * total_steps)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_one_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        logger: logging.Logger,
    ) -> float:
        """One training epoch.  Returns label-level macro F1 (fast proxy metric).

        The label-level F1 is logged with an explicit "label-level" prefix so
        it cannot be confused with the entity-level F1 from evaluate().
        """
        self.model.train()
        logger.info("Epoch %d: training on %d batches", epoch, len(loader))

        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in loader:
            # Unpack — collate fn returns (bert_inputs, grid_mask2d, dist_inputs,
            # pieces2word, sent_length, grid_labels, tokens_list)
            *tensors, tokens_list = batch
            bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, grid_labels = [
                t.to(self.device) for t in tensors
            ]

            logits = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            # [MEDIUM bug fix] Apply upper-triangle loss mask.
            # Pass grid_labels so build_loss_mask can distinguish THW cells
            # (label > 0, lower triangle, keep) from background lower-triangle
            # cells (label = 0, lower triangle, exclude).
            loss_mask = build_loss_mask(grid_mask2d, grid_labels)

            # logits: [B, L, L, num_labels] → select masked cells → [N, num_labels]
            # grid_labels: [B, L, L] → [N]
            loss = self.loss_fn(logits[loss_mask], grid_labels[loss_mask])
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            # Collect for label-level F1 (use grid_mask2d, not loss_mask, to
            # match original reporting scope and keep the metric stable)
            all_preds.append(preds[grid_mask2d].contiguous().view(-1).cpu())
            all_labels.append(grid_labels[grid_mask2d].contiguous().view(-1).cpu())

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        _, _, label_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        avg_loss = total_loss / len(loader)

        # [LOW bug fix] Log metric with an explicit "label-level" prefix so it
        # is not confused with entity-level F1 from evaluate().
        logger.info(
            "Epoch %d | loss=%.4f | label-level macro F1=%.4f  "
            "(NOTE: this is NOT entity-level F1; see evaluate() for that)",
            epoch, avg_loss, label_f1,
        )
        return label_f1

    def evaluate(
        self,
        loader: DataLoader,
        id2label: dict[int, str],
        logger: logging.Logger,
    ) -> float:
        """Validation pass.  Returns entity-level F1 (comparable to offline eval).

        Entity-level F1 is the primary checkpoint selection metric and the
        metric that directly corresponds to what tool_calc_performance.py
        computes on the held-out test set.
        """
        self.model.eval()
        logger.info("Evaluating on %d batches", len(loader))

        total_ent_c = total_ent_p = total_ent_r = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                *tensors, tokens_list = batch
                bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, grid_labels = [
                    t.to(self.device) for t in tensors
                ]

                logits = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                preds = torch.argmax(logits, dim=-1)   # [B, L, L]

                # Build gold entity list from grid_labels for this batch
                # Format: list of lists of {"indices": [...], "type": str}
                gold_batch = self._extract_gold_entities(
                    grid_labels, sent_length, id2label
                )

                ent_c, ent_p, ent_r = decode_and_compare(
                    logits=preds.cpu().numpy(),
                    length_list=sent_length.cpu().tolist(),
                    tokens_list=tokens_list,
                    gold_entities=gold_batch,
                    id2label=id2label,
                    logger=logger,
                )
                total_ent_c += ent_c
                total_ent_p += ent_p
                total_ent_r += ent_r

                all_preds.append(preds[grid_mask2d].contiguous().view(-1).cpu())
                all_labels.append(grid_labels[grid_mask2d].contiguous().view(-1).cpu())

        # Label-level metrics (for diagnostic reference)
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        _, _, label_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

        entity_f1, entity_p, entity_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)

        logger.info(
            "VAL label-level F1=%.4f  |  "
            "VAL entity-level F1=%.4f  P=%.4f  R=%.4f",
            label_f1, entity_f1, entity_p, entity_r,
        )
        return entity_f1

    @staticmethod
    def _extract_gold_entities(
        grid_labels: torch.Tensor,      # [B, L, L]
        sent_length: torch.Tensor,      # [B]
        id2label: dict[int, str],
    ) -> list[list[dict]]:
        """Decode gold grid_labels back into entity lists for decode_and_compare.

        For each sentence in the batch, scans the label matrix for THW cells
        (grid_labels[tail, head] > 1) and reconstructs entity indices by
        following NNW edges — mirroring what decode_one_sentence does on
        predictions.

        This is needed because evaluate() compares batch predictions to gold
        entities extracted from the same batch's grid_labels tensor, rather
        than loading gold entities from a separate file.
        """
        from .decoding import decode_one_sentence

        batch_gold: list[list[dict]] = []
        labels_np = grid_labels.cpu().numpy()
        lengths = sent_length.cpu().tolist()

        for i, length in enumerate(lengths):
            length = int(length)
            gold_spans, _ = decode_one_sentence(
                instance=labels_np[i, :length, :length],
                length=length,
                id2label=id2label,
            )
            gold_entities = [
                {"indices": span.indices, "type": id2label[span.label_id]}
                for span in gold_spans
            ]
            batch_gold.append(gold_entities)

        return batch_gold

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        id2label: dict[int, str] | None = None,
        save_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Full training loop with optional validation and early stopping.

        If val_loader is None:
            Trains for config.epochs, saves the last checkpoint.
            This preserves original behaviour for callers that manage their
            own evaluation externally.

        If val_loader is provided:
            Evaluates after each epoch, saves the best entity-F1 checkpoint,
            and stops early if patience > 0 and val F1 has not improved for
            `patience` consecutive epochs.

        Args:
            train_loader: DataLoader yielding training batches.
            val_loader:   Optional validation DataLoader.
            id2label:     Required when val_loader is provided (for decode).
            save_path:    Where to write model.pt.  If None, no checkpoint saved.
            logger:       Logger for progress messages.  Uses root logger if None.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        if val_loader is not None and id2label is None:
            raise ValueError("id2label must be provided when val_loader is used.")

        for epoch in range(self.config.epochs):
            self.train_one_epoch(epoch, train_loader, logger)

            if val_loader is not None:
                val_f1 = self.evaluate(val_loader, id2label, logger)

                # [HIGH bug fix] Save best checkpoint rather than always saving last
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.patience_counter = 0
                    if save_path is not None:
                        torch.save(self.model.state_dict(), save_path)
                        logger.info(
                            "Epoch %d: new best entity-F1=%.4f → checkpoint saved to %s",
                            epoch, val_f1, save_path,
                        )
                else:
                    self.patience_counter += 1
                    logger.info(
                        "Epoch %d: entity-F1=%.4f (best=%.4f, patience=%d/%s)", # %s because %d cannot represent infinity when patience=0
                        epoch, val_f1, self.best_f1,
                        self.patience_counter, self.patience if self.patience > 0 else float("inf"),
                    )
                    if self.patience > 0 and self.patience_counter >= self.patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d exhausted)",
                            epoch, self.patience,
                        )
                        break
            else:
                # Original behaviour: always save after each epoch (last wins)
                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info("Epoch %d: checkpoint saved to %s", epoch, save_path)
