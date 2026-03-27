# =============================================================================
# data/collate.py
#
# PURPOSE
#   NERDataset and the collate function: turn a list of per-sentence feature
#   dicts into padded batched tensors suitable for NERModel.forward().
#
# CORRESPONDS TO
#   core/dataset/collate.py + core/dataset/ner_dataset.py  (original W2NER)
#
# KEY DESIGN CHANGES
#   - NERDataset merged here from a separate ner_dataset.py:
#     train.py imports both NERDataset and make_ner_collate_fn from this
#     module, so keeping them together avoids a third file for a ten-line
#     class.
#
#   - NERDataset constructor binds feature_converter at construction time:
#     The original dataset took tokenizer, dis2idx, and label2id directly
#     and called make_feature_converter internally.  The new version accepts
#     the already-built converter closure, which is cleaner when train.py
#     builds one converter and shares it across train/val datasets.
#
#   - Collation logic is unchanged from the original:
#     bert_inputs are padded via pad_sequence; 2-D grids are zero-padded
#     via a local pad_matrix helper.  The return tuple element order is
#     documented explicitly so training and inference unpacking is not
#     implicit knowledge scattered across call sites.
#
# BUGS ADDRESSED
#   None — collate logic was not a source of bugs in the original.
# =============================================================================

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class NERDataset(Dataset):
    """PyTorch Dataset wrapping a list of TokenRecord-compatible dicts.

    Parameters
    ----------
    data:
        List of plain dicts (loaded from step03 JSONL output); each must
        have at least ``"sentence": list[str]``.  For training, also needs
        ``"ner": list[dict]``.
    feature_converter:
        Callable returned by ``feature_builder.make_feature_converter``.
        Signature: ``convert(instance: dict, with_labels: bool) -> dict``.
    with_labels:
        Passed through to ``feature_converter`` on every ``__getitem__``
        call.  Set ``True`` for training / validation, ``False`` for
        inference.
    """

    def __init__(
        self,
        data: list[dict],
        feature_converter,
        with_labels: bool = False,
    ) -> None:
        self.data = data
        self.feature_converter = feature_converter
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.feature_converter(self.data[idx], with_labels=self.with_labels)


def make_ner_collate_fn(with_labels: bool = False):
    """Return a collate_fn closure for use with torch.utils.data.DataLoader.

    Return tuple element order
    --------------------------
    with_labels=True  (training / validation):
        (bert_inputs, grid_mask2d, dist_inputs, pieces2word,
         sent_length, grid_labels, tokens_list)

    with_labels=False (inference):
        (bert_inputs, grid_mask2d, dist_inputs, pieces2word,
         sent_length, tokens_list)

    Tensor shapes (B = batch size, L = max sentence length, P = max pieces):
        bert_inputs  : [B, P]         long
        grid_mask2d  : [B, L, L]      bool
        dist_inputs  : [B, L, L]      long
        pieces2word  : [B, L, P]      bool
        sent_length  : [B]            long
        grid_labels  : [B, L, L]      long  (only when with_labels=True)
        tokens_list  : list[list[str]]       (plain Python, not a tensor)

    All tensor elements are batch-first.  ``tokens_list`` is always the
    last element regardless of ``with_labels``.
    """

    def collate_fn(batch: list[dict]) -> tuple:
        # ----------------------------------------------------------------
        # bert_inputs: variable-length 1-D sequences → [B, max_pieces]
        # Padded with 0 (tokenizer pad_token_id is also 0 for BERT-family).
        # ----------------------------------------------------------------
        bert_inputs = pad_sequence(
            [x["bert_input"] for x in batch],
            batch_first=True,
            padding_value=0,
        )

        max_sent_len = max(x["pieces2word"].shape[0] for x in batch)
        max_piece_len = bert_inputs.shape[1]

        def pad_matrix(
            tensors: list[torch.Tensor],
            shape: tuple[int, ...],
            dtype: torch.dtype,
        ) -> torch.Tensor:
            """Zero-pad a list of 2-D tensors to a common [B, *shape]."""
            out = torch.zeros((len(batch), *shape), dtype=dtype)
            for i, t in enumerate(tensors):
                slices = tuple(slice(0, s) for s in t.shape)
                out[i][slices] = t
            return out

        # pieces2word: [L, P] → [B, max_sent_len, max_piece_len]
        pieces2word = pad_matrix(
            [x["pieces2word"] for x in batch],
            (max_sent_len, max_piece_len),
            torch.bool,
        )
        # dist_inputs: [L, L] → [B, max_sent_len, max_sent_len]
        dist_inputs = pad_matrix(
            [x["dist_inputs"] for x in batch],
            (max_sent_len, max_sent_len),
            torch.long,
        )
        # grid_mask2d: [L, L] → [B, max_sent_len, max_sent_len]
        grid_mask2d = pad_matrix(
            [x["grid_mask2d"] for x in batch],
            (max_sent_len, max_sent_len),
            torch.bool,
        )

        sent_length = torch.LongTensor([x["pieces2word"].shape[0] for x in batch])
        tokens_list = [x["sentence"] for x in batch]

        if with_labels:
            # grid_labels: [L, L] → [B, max_sent_len, max_sent_len]
            grid_labels = pad_matrix(
                [x["grid_labels"] for x in batch],
                (max_sent_len, max_sent_len),
                torch.long,
            )
            return (
                bert_inputs, grid_mask2d, dist_inputs, pieces2word,
                sent_length, grid_labels, tokens_list,
            )
        else:
            return (
                bert_inputs, grid_mask2d, dist_inputs, pieces2word,
                sent_length, tokens_list,
            )

    return collate_fn
