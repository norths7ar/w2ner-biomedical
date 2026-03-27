# =============================================================================
# model/ner_model.py
#
# PURPOSE
#   Define the W2NER neural architecture: BioBERT encoder → max-pool word
#   representations → BiLSTM → conditional layer norm → dilated CNN →
#   biaffine + MLP co-predictor → per-cell label logits over the word-pair
#   grid.
#
# CORRESPONDS TO
#   core/model/ner_model.py  (original W2NER repo)
#
# KEY DESIGN CHANGES
#   - min_value sentinel replaced with -inf:
#     The original used `torch.min(bert_embs).item()` as the mask fill value
#     for the max-pooling step.  This value is batch-dependent: in different
#     batches the minimum embedding value is different, so the effective
#     "masked-out" signal varies across batches and across training vs.
#     inference.  If a legitimate subword embedding happens to equal min_value,
#     it is incorrectly treated as masked.  Replaced with float("-inf"),
#     which is the mathematically correct identity for max-pooling and is
#     batch-independent.
#
#   - output_hidden_states controlled by use_bert_last_4_layers:
#     The original always passed output_hidden_states=True to AutoModel,
#     even when use_bert_last_4_layers=False (last-layer-only mode).  In
#     last-layer-only mode the hidden state stack is never read but is still
#     computed and stored, wasting memory proportional to 4 × seq_len ×
#     hidden_size per batch.  Now output_hidden_states=True is only set when
#     use_bert_last_4_layers=True.
#
#   - bert_hid_size validated at construction time:
#     If an encoder is passed in (as in the train.py entry point), the
#     encoder's actual hidden_size is checked against config.bert_hid_size
#     before the model is assembled.  This makes shape mismatches loud at
#     construction rather than silent during the first forward pass.
#
#   - encoder always provided by the caller:
#     The original had a fallback that loaded the encoder from disk inside
#     __init__ if encoder=None, using a hardcoded relative cache path
#     ("../../cache/").  This is removed: the caller (train.py /
#     step05_predict.py) is always responsible for loading the encoder and
#     passing it in.  This makes the model class independent of filesystem
#     layout and easier to test.
#
# BUGS ADDRESSED
#   [HIGH]   min_value sentinel in max-pooling is batch-dependent; should
#            be -inf.
#            Fixed: MAX_POOL_MASK_VALUE = float("-inf").
#
#   [LOW]    output_hidden_states=True always wastes memory in last-layer
#            mode.
#            Fixed: flag is now gated on use_bert_last_4_layers.
#
#   [MEDIUM] bert_hid_size not validated against loaded model.
#            Fixed: construction-time assertion checks encoder hidden_size.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model_config import ModelConfig


# The correct mask fill value for max-pooling over subword pieces.
# float("-inf") is the identity element for max: max(x, -inf) == x for any x,
# so masked positions can never "win" the reduction regardless of embedding
# values in the unmasked positions.  Replacing the original batch-dependent
# torch.min(bert_embs).item() sentinel with this constant is the fix for the
# [HIGH] min_value bug.
MAX_POOL_MASK_VALUE: float = float("-inf")


class LayerNorm(nn.Module):
    """Conditional layer normalisation.

    When conditional=True, shift (beta) and scale (gamma) are modulated by
    a conditioning tensor, implementing FiLM-style feature-wise linear
    modulation.  Used in the W2NER architecture to condition word-pair
    representations on their paired word representation.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        center: bool = True,
        scale: bool = True,
        epsilon: float | None = None,
        conditional: bool = False,
        hidden_units: int | None = None,
        hidden_activation: str = "linear",
        hidden_initializer: str = "xaiver",
    ) -> None:
        super().__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(self.cond_dim, self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(self.cond_dim, input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(self.cond_dim, input_dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == "normal":
                    nn.init.normal_(self.hidden_dense.weight)
                elif self.hidden_initializer in ("xavier", "xaiver"):
                    nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if cond is None:
                raise ValueError("Conditional LayerNorm requires a `cond` tensor, got None.")
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # Broadcast cond to match inputs rank
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)
            beta = self.beta_dense(cond) + self.beta if self.center else self.beta
            gamma = self.gamma_dense(cond) + self.gamma if self.scale else self.gamma
        else:
            beta = self.beta if self.center else None
            gamma = self.gamma if self.scale else None

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1, keepdim=True)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    """Stacked dilated depthwise-separable 2D convolution block.

    Applies a 1×1 projection followed by N dilated 3×3 depthwise convolutions
    (one per dilation factor), concatenates all N outputs channel-wise, and
    returns a [B, L, L, channels * len(dilation)] tensor.
    """

    def __init__(self, input_size: int, channels: int, dilation: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d)
            for d in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, L, C] → [B, C, L, L] for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)
        outputs = []
        for conv in self.convs:
            x = F.gelu(conv(x))
            outputs.append(x)
        # Concatenate channels, then back to [B, L, L, C*N]
        return torch.cat(outputs, dim=1).permute(0, 2, 3, 1).contiguous()


class Biaffine(nn.Module):
    """Biaffine attention scorer for word-pair label prediction.

    Computes s[b,i,j,o] = x[b,i] · W[o] · y[b,j]ᵀ for each output
    dimension o via einsum.  Optional bias terms are appended to x and/or y.
    """

    def __init__(self, n_in: int, n_out: int = 1, bias_x: bool = True, bias_y: bool = True) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight)

    def extra_repr(self) -> str:
        return f"n_in={self.n_in}, n_out={self.n_out}, bias_x={self.bias_x}, bias_y={self.bias_y}"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)
        # [B, L, L, n_out]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        return s.permute(0, 2, 3, 1)


class MLP(nn.Module):
    """Single linear layer with GELU activation and optional input dropout."""

    def __init__(self, n_in: int, n_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(self.dropout(x)))


class CoPredictor(nn.Module):
    """Combine biaffine (from BiLSTM states) and MLP (from conv features) scores.

    Both paths produce [B, L, L, cls_num] tensors that are summed element-wise,
    so they share the same label space.
    """

    def __init__(
        self,
        cls_num: int,
        hid_size: int,
        biaffine_size: int,
        channels: int,
        ffnn_hid_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp1 = MLP(hid_size, biaffine_size, dropout=dropout)
        self.mlp2 = MLP(hid_size, biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(biaffine_size, cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,   # [B, L, hid_size]    — BiLSTM output (head role)
        y: torch.Tensor,   # [B, L, hid_size]    — BiLSTM output (tail role)
        z: torch.Tensor,   # [B, L, L, channels] — conv output
    ) -> torch.Tensor:     # [B, L, L, cls_num]
        o1 = self.biaffine(self.dropout(self.mlp1(x)), self.dropout(self.mlp2(y)))
        o2 = self.linear(self.dropout(self.mlp_rel(z)))
        return o1 + o2


class NERModel(nn.Module):
    """Full W2NER model.

    The encoder (BioBERT or any HuggingFace AutoModel) must be provided by
    the caller — this class does not load from disk.  Keeping loading outside
    the model makes the class filesystem-independent and enables construction-
    time bert_hid_size validation.

    Architecture:
        encoder → max-pool pieces2word → dropout → BiLSTM
               → conditional LayerNorm over word-pairs
               → concat(dist_emb, reg_emb, cln)
               → dilated CNN → CoPredictor
               → logits [B, L, L, label_num]
    """

    def __init__(self, config: ModelConfig, encoder: nn.Module) -> None:
        super().__init__()

        # [MEDIUM bug fix] Validate that the encoder's hidden size matches
        # config before any downstream layers are sized.  Catches backbone
        # swap mistakes (e.g. using a 1024-dim model with a 768-dim config)
        # at construction time rather than during the first forward pass.
        actual_hid = encoder.config.hidden_size
        if actual_hid != config.bert_hid_size:
            raise ValueError(
                f"Encoder hidden_size={actual_hid} does not match "
                f"config.bert_hid_size={config.bert_hid_size}. "
                "Update biored_base.json or load the correct encoder."
            )

        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.bert = encoder

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(
            config.bert_hid_size,
            config.lstm_hid_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size
        self.convLayer = ConvolutionLayer(
            conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout
        )
        self.dropout = nn.Dropout(config.emb_dropout)
        self.predictor = CoPredictor(
            config.label_num,
            config.lstm_hid_size,
            config.biaffine_size,
            config.conv_hid_size * len(config.dilation),
            config.ffnn_hid_size,
            config.out_dropout,
        )
        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

    def forward(
        self,
        bert_inputs: torch.Tensor,   # [B, L'] — subword token IDs (padded)
        grid_mask2d: torch.Tensor,   # [B, L, L] — valid word-pair positions
        dist_inputs: torch.Tensor,   # [B, L, L] — bucketed distance IDs
        pieces2word: torch.Tensor,   # [B, L, L'] — 1 where subword j belongs to word i
        sent_length: torch.Tensor,   # [B]        — true word lengths
    ) -> torch.Tensor:               # [B, L, L, label_num]

        # --- 1. Subword encoding ---
        attention_mask = bert_inputs.ne(0).float()
        bert_out = self.bert(input_ids=bert_inputs, attention_mask=attention_mask)

        if self.use_bert_last_4_layers:
            # [LOW bug fix] output_hidden_states is only requested when needed.
            # bert_out[2] is the tuple of all layer hidden states.
            bert_embs = torch.stack(bert_out[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_out[0]   # final hidden state only: [B, L', H]

        # --- 2. Max-pool subword → word representations ---
        # Expand bert_embs to [B, L, L', H] then mask subwords not belonging
        # to each word position, take max over L' dimension.
        length = pieces2word.size(1)
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)

        # [HIGH bug fix] Use -inf as the fill value.  The original used
        # torch.min(bert_embs).item(), which is batch-dependent.  Any
        # embedding value equal to the batch minimum would be silently masked,
        # and the sentinel changes between training and inference batches.
        _bert_embs = _bert_embs.masked_fill(pieces2word.eq(0).unsqueeze(-1), MAX_POOL_MASK_VALUE)
        word_reps, _ = torch.max(_bert_embs, dim=2)   # [B, L, H]

        # --- 3. BiLSTM contextualisation ---
        word_reps = self.dropout(word_reps)
        packed = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        word_reps, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=sent_length.max())
        # word_reps: [B, L, lstm_hid_size]

        # --- 4. Conditional layer norm for word-pair features ---
        # unsqueeze(2) broadcasts word_reps as "head" across the pair dimension;
        # the conditioning signal is the "tail" word's representation.
        cln = self.cln(word_reps.unsqueeze(2), word_reps)   # [B, L, L, lstm_hid_size]

        # --- 5. Distance and region type embeddings ---
        dis_emb = self.dis_embs(dist_inputs)                              # [B, L, L, dist_emb_size]
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()               # values in {0, 1, 2}
        reg_emb = self.reg_embs(reg_inputs)                               # [B, L, L, type_emb_size]

        # --- 6. Dilated CNN over the word-pair grid ---
        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)         # [B, L, L, conv_input_size]
        conv_inputs = conv_inputs.masked_fill(grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)                        # [B, L, L, conv_hid*N_dilations]
        conv_outputs = conv_outputs.masked_fill(grid_mask2d.eq(0).unsqueeze(-1), 0.0)

        # --- 7. Co-predictor: biaffine + MLP-on-conv → label logits ---
        return self.predictor(word_reps, word_reps, conv_outputs)         # [B, L, L, label_num]
