# =============================================================================
# data/feature_builder.py
#
# PURPOSE
#   Convert a single TokenRecord (with optional NER annotations) into the
#   tensor representation consumed by NERModel: bert_input, pieces2word,
#   dist_inputs, grid_mask2d, and (in training mode) grid_labels.
#
# CORRESPONDS TO
#   core/dataset/feature_builder.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - Silent type fallback eliminated:
#     The original used `label2id.get(ent["type"], 0)` for THW cell
#     encoding.  Index 0 is the background label (<pad>), so an unknown
#     entity type silently became background, producing a training signal
#     that actively teaches the model to ignore that entity.  This version
#     raises KeyError on unknown types.  Because step03 and step04 together
#     guarantee that all entity types in the training data are in the spec,
#     this error should never fire in a correctly-run pipeline — but if it
#     does, it fires loudly rather than silently corrupting labels.
#
#   - CLS offset is a named constant:
#     The original had a hardcoded `+1` in the pieces2word construction
#     loop (to skip the [CLS] token).  This is replaced with CLS_OFFSET = 1,
#     defined at module level with an explanatory comment.  step02_tokenize.py
#     imports and references this constant so both the tokenization and
#     feature-building steps use the same documented value.
#
#   - NNW_LABEL is imported from model.decoding:
#     The original used the magic number 1 for NNW edge labels in multiple
#     places.  This is now model.decoding.NNW_LABEL, imported here and used
#     in the grid_labels construction loop so the two modules stay in sync
#     if the sentinel value ever changes.
#
#   - make_feature_converter API change:
#     The original factory took only `with_labels` and returned a function
#     taking `(instance, tokenizer, dis2idx, label2id)`.  The new signature
#     binds tokenizer, dis2idx, and label2id at factory time, returning a
#     simpler `convert(instance, with_labels)`.  This matches the call site
#     in train.py and makes NERDataset cleaner (no per-item arg threading).
#
#   - Grid label construction validates index bounds:
#     If any NER entry has an index that exceeds the sentence length, an
#     IndexError is raised.  The original silently wrote out-of-bounds into
#     a tensor that would have been caught (likely) by a CUDA error, but
#     without a meaningful message.
#
# BUGS ADDRESSED
#   [CRITICAL]  Silent type fallback — label2id.get(type, 0) maps unknown
#               entity types to background label with no error:
#               Fixed: label2id[ent["type"]] (strict lookup, raises KeyError
#               on miss).
#
#   [HIGH]      CLS offset hardcoded as +1 in pieces2word:
#               Fixed: replaced with named constant CLS_OFFSET = 1 with a
#               comment explaining why the offset exists and where it comes
#               from.
# =============================================================================

from __future__ import annotations

import numpy as np
import torch

from ..model.constants import NNW_LABEL, CLS_OFFSET, DIST_DIAGONAL


def build_dis2idx(max_distance: int = 1000) -> dict[int, int]:
    """Build the distance-to-bucket lookup table.

    Maps an absolute word distance to a bucket index in [0, 9].
    Bucket boundaries (log2-spaced, same as original W2NER):

        0        → 0  (diagonal placeholder, remapped to DIST_DIAGONAL later)
        1        → 1
        2-3      → 2
        4-7      → 3
        8-15     → 4
        16-31    → 5
        32-63    → 6
        64-127   → 7
        128-255  → 8
        256+     → 9

    Combined with the +9 offset applied to upper-triangle cells in
    convert_instance, the full range of dist_inputs values is 1-19
    (diagonal becomes 19 via DIST_DIAGONAL substitution).  The model's
    dist_emb_size must therefore be ≥ 20.
    """
    dis2idx = np.zeros(max_distance, dtype="int64")
    dis2idx[1] = 1
    dis2idx[2:] = 2
    dis2idx[4:] = 3
    dis2idx[8:] = 4
    dis2idx[16:] = 5
    dis2idx[32:] = 6
    dis2idx[64:] = 7
    dis2idx[128:] = 8
    dis2idx[256:] = 9
    return {i: int(x) for i, x in enumerate(dis2idx)}


def make_feature_converter(tokenizer, dis2idx, label2id: dict[str, int]):
    """Return a convert_instance closure bound to tokenizer, dis2idx, label2id.

    Parameters
    ----------
    tokenizer:
        A HuggingFace fast tokenizer with ``tokenize``,
        ``convert_tokens_to_ids``, ``cls_token_id``, ``sep_token_id``.
    dis2idx:
        Distance-bucket lookup; supports integer index access (dict or
        numpy array).  Typically produced by ``build_dis2idx()``.
    label2id:
        Complete label→id mapping including sentinels; produced by
        ``LabelSpec.label2id()``.

    Returns
    -------
    Callable: ``convert_instance(instance, with_labels) -> dict``
    """

    def convert_instance(instance: dict, with_labels: bool = False) -> dict:
        """Convert one TokenRecord dict to model-ready tensors.

        Parameters
        ----------
        instance:
            Dict with at least ``"sentence": list[str]``.
            When ``with_labels=True``, also needs ``"ner": list[dict]``
            where each NER dict has ``"indices": list[int]`` and
            ``"type": str``.

        Raises
        ------
        KeyError:
            If ``with_labels=True`` and an entity type is absent from
            ``label2id``.  (Replaces the original silent fallback to 0.)
        IndexError:
            If any NER entry index is out of range for the sentence length.
        """
        words = instance["sentence"]
        L = len(words)

        # ------------------------------------------------------------------
        # Tokenise each word → list of subword pieces
        # ------------------------------------------------------------------
        tokens = [tokenizer.tokenize(word) for word in words]
        pieces = [p for ps in tokens for p in ps]

        # bert_input: [CLS] + pieces + [SEP],  shape [P + 2]
        piece_ids = tokenizer.convert_tokens_to_ids(pieces)
        bert_input = torch.LongTensor(
            [tokenizer.cls_token_id] + piece_ids + [tokenizer.sep_token_id]
        )

        # ------------------------------------------------------------------
        # pieces2word: [L, P+2] bool
        #   pieces2word[word_i, piece_start+CLS_OFFSET : piece_end+CLS_OFFSET+1] = True
        # CLS_OFFSET = 1 because [CLS] sits at bert_input position 0.
        # ------------------------------------------------------------------
        pieces2word = torch.zeros((L, len(bert_input)), dtype=torch.bool)
        piece_cursor = 0
        for i, ps in enumerate(tokens):
            if not ps:
                continue
            start = piece_cursor
            end = piece_cursor + len(ps) - 1
            pieces2word[i, start + CLS_OFFSET : end + CLS_OFFSET + 1] = True
            piece_cursor += len(ps)

        # ------------------------------------------------------------------
        # dist_inputs: [L, L] long — log2-bucketed relative distances.
        #
        #   i < j  (upper triangle): dis2idx[j - i] + 9
        #   i > j  (lower triangle): dis2idx[i - j]
        #   i == j (diagonal):       dis2idx[0] = 0 → remapped to DIST_DIAGONAL
        #
        # The +9 offset ensures upper- and lower-triangle distances occupy
        # non-overlapping ranges in the embedding table.
        # ------------------------------------------------------------------
        dist_inputs = torch.zeros((L, L), dtype=torch.long)
        for i in range(L):
            for j in range(L):
                relative = i - j
                if relative < 0:
                    dist_inputs[i, j] = dis2idx[-relative] + 9
                else:
                    dist_inputs[i, j] = dis2idx[relative]
        # Replace 0 (diagonal) with the dedicated diagonal sentinel.
        dist_inputs[dist_inputs == 0] = DIST_DIAGONAL

        # ------------------------------------------------------------------
        # grid_mask2d: [L, L] bool — all cells True for this sentence.
        # Padding cells (shorter sentences in a batch) are handled in collate.
        # ------------------------------------------------------------------
        grid_mask2d = torch.ones((L, L), dtype=torch.bool)

        output: dict = {
            "bert_input": bert_input,
            "pieces2word": pieces2word,
            "dist_inputs": dist_inputs,
            "grid_mask2d": grid_mask2d,
            "sentence": words,
        }

        if with_labels:
            # ------------------------------------------------------------------
            # grid_labels: [L, L] long
            #
            # For each NER entry with sorted indices [i_0, i_1, …, i_n]
            # and entity type t:
            #   NNW edges:  grid_labels[i_k, i_{k+1}] = NNW_LABEL (= 1)
            #               for each consecutive pair  (upper triangle)
            #   THW anchor: grid_labels[i_n, i_0] = label2id[t]   (≥ 2)
            #               strict lookup — KeyError if type is unknown
            # ------------------------------------------------------------------
            grid_labels = torch.zeros((L, L), dtype=torch.long)
            for ent in instance.get("ner", []):
                indices = ent["indices"]
                if not indices:
                    continue

                # Bounds check: gives a clear error instead of a silent
                # out-of-bounds write that may crash later in CUDA.
                bad = [idx for idx in indices if not (0 <= idx < L)]
                if bad:
                    raise IndexError(
                        f"NER entry indices {bad} out of range for sentence "
                        f"length {L}: {words!r}"
                    )

                # NNW edges (consecutive word pairs along the entity span)
                for k in range(len(indices) - 1):
                    grid_labels[indices[k], indices[k + 1]] = NNW_LABEL

                # THW anchor — strict lookup; unknown type raises KeyError
                grid_labels[indices[-1], indices[0]] = label2id[ent["type"]]

            output["grid_labels"] = grid_labels

        return output

    return convert_instance
