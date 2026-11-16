"""FAFIRec implementation corresponding to the revised diagnostic-control method.

The default controller is a zero-initialized, unrestricted linear map of cached
teacher diagnostics. Ablations share the same encoder, text projection, item
gate, and scoring function. Historical published numbers do not validate this
revised implementation; new runs must record their protocol and feature source.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _CausalBlock(nn.Module):
    """Small pre-normalized Transformer with explicitly safe padding queries."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, blocked: Tensor, valid: Tensor) -> Tensor:
        normed = self.attn_norm(x)
        attended = self.attn(
            normed, normed, normed, attn_mask=blocked, need_weights=False
        )[0]
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class FAFIRecModel(nn.Module):
    """Causal next-item model with diagnostic-guided alignment and fusion.

    ``hist_ids``, ``pos_ids`` and ``neg_ids`` are aligned [B, L] integer tensors.
    At each nonpadding history position, the positive is its next item. Either
    left or right padding is accepted; padding never contributes attention keys
    for real queries, positions, recommendation loss, or alignment examples.

    Diagnostics are detached [B, 2] training-reference-standardized features in
    the order (log gradient magnitude, log collaborative loss). This class does
    not estimate or restandardize them. Diagnostic control requires diagnostics
    in both training and retrieval. User-level diagnostics are broadcast over
    causal prefixes. ``frequency`` is an explicitly labeled prefix-length
    control, not PAD's target-item-frequency router.

    Text features are frozen buffers and excluded from state_dict to avoid
    duplicating the external feature artifact. Reconstruct with the same feature
    tensor before loading a checkpoint. Call eval() for deterministic retrieval;
    score_all deliberately does not change mode or suppress caller gradients.
    """

    DIAGNOSTIC_CONTROLS = frozenset({
        "diagnostic", "diagnostic_uniform_alignment", "diagnostic_no_conflict",
        "diagnostic_loss_only", "diagnostic_gradient_only", "diagnostic_no_alignment",
    })
    CONTROLS = DIAGNOSTIC_CONTROLS | {"cf", "uniform", "frequency"}

    def __init__(
        self,
        n_items: int,
        text_features: Tensor,
        d_model: int = 32,
        max_len: int = 30,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        control: str = "diagnostic",
        align_weight: float = 0.03,
        bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 3.5, 5.0),
    ) -> None:
        super().__init__()
        if control not in self.CONTROLS:
            raise ValueError(f"Unknown control {control!r}; use {sorted(self.CONTROLS)}")
        if n_items < 1 or max_len < 1 or n_layers < 1 or n_heads < 1:
            raise ValueError("Item count, max_len, n_layers and n_heads must be positive")
        if d_model < 1 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(align_weight) or align_weight < 0:
            raise ValueError("align_weight must be finite and nonnegative")
        widths = tuple(float(value) for value in bandwidths)
        if not widths or any(not math.isfinite(value) or value <= 0 for value in widths):
            raise ValueError("bandwidths must contain positive finite numbers")
        if not isinstance(text_features, Tensor) or text_features.ndim != 2:
            raise ValueError("text_features must be a [n_items + 1, content_dim] Tensor")
        if text_features.shape[0] != n_items + 1 or text_features.shape[1] < 1:
            raise ValueError("text_features must include padding row 0 and all items")
        if not bool(torch.isfinite(text_features).all()):
            raise ValueError("text_features must be finite")

        self.n_items = int(n_items)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dropout_rate = float(dropout)
        self.control = control
        self.align_weight = float(align_weight)
        self.bandwidths = widths
        self.tau = 1.0
        self.score_scale = math.sqrt(self.d_model)

        features = text_features.detach().to(dtype=torch.float32).clone()
        features[0].zero_()
        self.register_buffer("text_features", features, persistent=False)
        self.register_buffer(
            "_future_mask",
            torch.ones(max_len, max_len, dtype=torch.bool).triu(1),
            persistent=False,
        )
        self.item_id_table = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.proj_content = nn.Linear(features.shape[1], d_model, bias=False)
        self.pos_emb = nn.Embedding(max_len + 1, d_model, padding_idx=0)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)

        # This item gate is learned, dimension-wise, and independent of users.
        # Its architecture and initial value are identical for all multimodal controls.
        self.item_gate_logit = nn.Parameter(torch.zeros(d_model))
        self.gate_bias = nn.Parameter(torch.zeros(d_model))
        self.control_bias = nn.Parameter(torch.zeros(()))
        self.control_weights = nn.Parameter(torch.zeros(2))
        feature_mask = torch.ones(2)
        if control == "diagnostic_loss_only":
            feature_mask[0] = 0
        elif control == "diagnostic_gradient_only":
            feature_mask[1] = 0
        self.register_buffer("_diagnostic_mask", feature_mask, persistent=False)
        self.conflict_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.5))))

        nn.init.normal_(self.item_id_table.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_content.weight)
        with torch.no_grad():
            self.item_id_table.weight[0].zero_()
            self.pos_emb.weight[0].zero_()
        # The CF teacher has no text/gating/reliability computation or updates.
        if control == "cf":
            self.proj_content.requires_grad_(False)
            self.item_gate_logit.requires_grad_(False)
            self.gate_bias.requires_grad_(False)
            self.conflict_scale_raw.requires_grad_(False)
        if control not in self.DIAGNOSTIC_CONTROLS:
            self.control_bias.requires_grad_(False)
            self.control_weights.requires_grad_(False)
        if control == "diagnostic_no_conflict":
            self.conflict_scale_raw.requires_grad_(False)

    def _check_ids(self, ids: Tensor, name: str) -> Tensor:
        if not isinstance(ids, Tensor) or ids.ndim != 2:
            raise ValueError(f"{name} must be a [batch, length] Tensor")
        if ids.shape[0] < 1 or not (1 <= ids.shape[1] <= self.max_len):
            raise ValueError(f"{name} needs a nonempty batch and length in [1, max_len]")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must contain integer item IDs")
        if ids.device != self.item_id_table.weight.device:
            raise ValueError(f"{name} must be on the model's device")
        if bool(((ids < 0) | (ids > self.n_items)).any()):
            raise ValueError(f"{name} contains item IDs outside [0, n_items]")
        return ids.long()

    @staticmethod
    def _last(states: Tensor, valid: Tensor) -> Tensor:
        """Gather the actual last valid index, independent of padding orientation."""
        indices = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        last = indices.masked_fill(~valid, -1).amax(dim=1)
        gathered = states[torch.arange(valid.shape[0], device=valid.device), last.clamp_min(0)]
        return gathered.masked_fill(~valid.any(dim=1).unsqueeze(-1), 0.0)

    def _encode_tokens(self, tokens: Tensor, valid: Tensor) -> Tensor:
        length = valid.shape[1]
        # Rank real events rather than array columns, so left/right padding agrees.
        positions = valid.long().cumsum(dim=1) * valid.long()
        x = self.input_dropout(self.input_norm(tokens + self.pos_emb(positions)))
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        blocked = self._future_mask[:length, :length].unsqueeze(0) | ~valid.unsqueeze(1)
        # A padded query with no preceding key would otherwise have all -inf
        # logits. Allow only its own padding key additionally, then zero its output.
        diagonal = torch.eye(length, dtype=torch.bool, device=valid.device).unsqueeze(0)
        blocked = blocked & ~(~valid.unsqueeze(-1) & diagonal)
        blocked = blocked.repeat_interleave(self.n_heads, dim=0)
        for block in self.blocks:
            x = block(x, blocked, valid)
        x = self.output_norm(x).masked_fill(~valid.unsqueeze(-1), 0.0)
        return F.normalize(x, p=2, dim=-1)

    def encode_cf(self, hist_ids: Tensor) -> Tensor:
        """Last causal ID-only state [B, D], retaining gradients for diagnostics."""
        hist_ids = self._check_ids(hist_ids, "hist_ids")
        valid = hist_ids.ne(0)
        states = self._encode_tokens(self.item_id_table(hist_ids), valid)
        return self._last(states, valid)

    def _control_scores(
        self, valid: Tensor, diagnostics: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch, length = valid.shape
        dtype = self.item_id_table.weight.dtype
        if self.control == "uniform":
            scores = torch.zeros(batch, length, dtype=dtype, device=valid.device)
        elif self.control == "frequency":
            prefix_lengths = valid.cumsum(dim=1).to(dtype=dtype)
            scores = torch.log1p(prefix_lengths) - math.log1p(self.max_len / 2.0)
        else:
            if not isinstance(diagnostics, Tensor) or diagnostics.shape != (batch, 2):
                raise ValueError("diagnostic control requires diagnostics with shape [B, 2]")
            diag = diagnostics.detach().to(device=valid.device, dtype=dtype)
            if not bool(torch.isfinite(diag).all()):
                raise ValueError("diagnostics must be finite, standardized training features")
            user_scores = self.control_bias + (diag * self._diagnostic_mask) @ self.control_weights
            scores = user_scores.unsqueeze(1).expand(batch, length)
        return scores, torch.sigmoid(scores)

    def _fusion_gate(self, scores: Tensor, rho: Tensor, discrepancy: Tensor) -> Tensor:
        """Gate before normalization; conflict direction is determined by rho."""
        logits = scores.unsqueeze(-1) + self.gate_bias
        if self.control != "diagnostic_no_conflict":
            logits = logits + F.softplus(self.conflict_scale_raw) * (
                2.0 * rho.unsqueeze(-1) - 1.0
            ) * discrepancy
        return torch.sigmoid(logits)

    def encode_branches(
        self, hist_ids: Tensor, diagnostics: Optional[Tensor] = None
    ) -> Dict[str, Optional[Tensor]]:
        """Expose normalized states and the actual gate for same-checkpoint audits.

        Outputs retain gradients. Call ``eval()`` and ``no_grad()`` for retrieval;
        no checkpoint, mode, or feature cache is changed by this method.
        """
        hist_ids = self._check_ids(hist_ids, "hist_ids")
        valid = hist_ids.ne(0)
        ids = self.item_id_table(hist_ids)
        if self.control == "cf":
            collaborative = self._encode_tokens(ids, valid)
            scores = torch.zeros_like(hist_ids, dtype=collaborative.dtype)
            return {
                "fused": collaborative, "collaborative": collaborative, "textual": None,
                "scores": scores, "rho": torch.ones_like(scores),
                "gate": torch.ones_like(collaborative), "valid": valid,
            }
        text = self.proj_content(F.embedding(hist_ids, self.text_features, padding_idx=0))
        encoded = self._encode_tokens(torch.cat((ids, text), dim=0), valid.repeat(2, 1))
        collaborative, textual = encoded.chunk(2, dim=0)
        scores, rho = self._control_scores(valid, diagnostics)
        gate = self._fusion_gate(scores, rho, (collaborative - textual).abs())
        fused = F.normalize(gate * collaborative + (1.0 - gate) * textual, dim=-1)
        fused = fused.masked_fill(~valid.unsqueeze(-1), 0.0)
        return {
            "fused": fused, "collaborative": collaborative, "textual": textual,
            "scores": scores, "rho": rho, "gate": gate, "valid": valid,
        }

    def _user_states(
        self, hist_ids: Tensor, diagnostics: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        states = self.encode_branches(hist_ids, diagnostics)
        return (states["fused"], states["collaborative"], states["textual"],
                states["scores"], states["rho"])

    def user_vectors(self, hist_ids: Tensor, diagnostics: Optional[Tensor] = None) -> Tensor:
        states = self.encode_branches(hist_ids, diagnostics)
        return self._last(states["fused"], states["valid"])

    def _item_vectors(self, ids: Optional[Tensor] = None) -> Tensor:
        collaborative = self.item_id_table.weight if ids is None else self.item_id_table(ids)
        if self.control == "cf":
            combined = collaborative
        else:
            raw = self.text_features if ids is None else F.embedding(
                ids, self.text_features, padding_idx=0
            )
            textual = self.proj_content(raw)
            gate = torch.sigmoid(self.item_gate_logit)
            combined = gate * collaborative + (1.0 - gate) * textual
        normalized = F.normalize(combined, p=2, dim=-1)
        if ids is None:
            return torch.cat((torch.zeros_like(normalized[:1]), normalized[1:]), dim=0)
        return normalized.masked_fill(ids.eq(0).unsqueeze(-1), 0.0)

    def alignment_weights(self, scores: Tensor) -> Tensor:
        """Return detached simplex weights; the ablation forces uniform weights."""
        if scores.ndim != 1 or scores.numel() == 0:
            raise ValueError("alignment scores must be a nonempty 1D tensor")
        if self.control == "diagnostic_uniform_alignment":
            scores = torch.zeros_like(scores)
        return torch.softmax(scores / self.tau, dim=0).detach()

    def _weighted_mmd(self, collaborative: Tensor, textual: Tensor, scores: Tensor) -> Tensor:
        # B x B only: kernels never expand across sequence tokens. Weights are
        # detached so alignment cannot improve just by discarding difficult users.
        weights = self.alignment_weights(scores)
        outer = weights.unsqueeze(1) * weights.unsqueeze(0)

        def distances(left: Tensor, right: Tensor) -> Tensor:
            return (
                left.square().sum(-1, keepdim=True)
                + right.square().sum(-1).unsqueeze(0)
                - 2.0 * (left @ right.transpose(0, 1))
            ).clamp_min(0.0)

        d_cc = distances(collaborative, collaborative)
        d_tt = distances(textual, textual)
        d_ct = distances(collaborative, textual)
        loss = collaborative.new_zeros(())
        for bandwidth in self.bandwidths:
            scale = 2.0 * bandwidth * bandwidth
            kernels = torch.exp(-d_cc / scale) + torch.exp(-d_tt / scale)
            kernels = kernels - 2.0 * torch.exp(-d_ct / scale)
            loss = loss + (outer * kernels).sum()
        # Fixed uniform kernel mixture; clamp removes roundoff below zero.
        return (loss / len(self.bandwidths)).clamp_min(0.0)

    def forward(
        self,
        hist_ids: Tensor,
        pos_ids: Tensor,
        neg_ids: Tensor,
        diagnostics: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        hist_ids = self._check_ids(hist_ids, "hist_ids")
        pos_ids = self._check_ids(pos_ids, "pos_ids")
        neg_ids = self._check_ids(neg_ids, "neg_ids")
        if pos_ids.shape != hist_ids.shape or neg_ids.shape != hist_ids.shape:
            raise ValueError("hist_ids, pos_ids and neg_ids must have identical shapes")
        supervised = hist_ids.ne(0) & pos_ids.ne(0)
        if bool((supervised & neg_ids.eq(0)).any()):
            raise ValueError("A supervised next-item position needs a nonpadding negative")
        zero = self.item_id_table.weight[0].sum() * 0.0
        if not bool(supervised.any()):
            return {"loss": zero, "rec_loss": zero, "align_loss": zero,
                    "mean_rho": zero.detach()}
        users, collaborative, textual, scores, rho = self._user_states(hist_ids, diagnostics)
        positive = self._item_vectors(pos_ids)
        negative = self._item_vectors(neg_ids)
        margin = self.score_scale * (users * (positive - negative)).sum(dim=-1)
        rec_loss = F.softplus(-margin[supervised]).mean()
        rows = supervised.any(dim=1)
        # Exactly one alignment example per supervised user, at its last usable
        # causal prefix. This also handles all-padding rows in a mixed batch.
        last_rho = self._last(rho.unsqueeze(-1), supervised).squeeze(-1)[rows]
        align_loss = zero
        if self.control not in {"cf", "diagnostic_no_alignment"} and self.align_weight > 0:
            last_c = self._last(collaborative, supervised)[rows]
            last_t = self._last(textual, supervised)[rows]
            last_scores = self._last(scores.unsqueeze(-1), supervised).squeeze(-1)[rows]
            align_loss = self._weighted_mmd(last_c, last_t, last_scores)
        return {
            "loss": rec_loss + self.align_weight * align_loss,
            "rec_loss": rec_loss,
            "align_loss": align_loss,
            "mean_rho": last_rho.mean(),
        }

    def score_all(self, hist_ids: Tensor, diagnostics: Optional[Tensor] = None) -> Tensor:
        """Full-catalog scores [B, N+1]; caller handles seen-item exclusion."""
        hist_ids = self._check_ids(hist_ids, "hist_ids")
        states, _, _, _, _ = self._user_states(hist_ids, diagnostics)
        user = self._last(states, hist_ids.ne(0))
        scores = self.score_scale * (user @ self._item_vectors().transpose(0, 1))
        # Padding ID is never a recommendation candidate, including empty users.
        scores = scores.clone()
        scores[:, 0] = float("-inf")
        return scores

    def item_vectors(self, ids: Optional[Tensor] = None) -> Tensor:
        """Current normalized item vectors, optionally for a shared 1D catalog chunk."""
        if ids is not None:
            self._check_candidate_ids(ids)
        return self._item_vectors(ids)

    def _check_candidate_ids(self, ids: Tensor) -> None:
        if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.numel() < 1:
            raise ValueError("candidate_ids must be a nonempty 1D integer tensor")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_ids must contain integer IDs")
        if ids.device != self.item_id_table.weight.device:
            raise ValueError("candidate_ids must be on the model's device")
        if bool(((ids < 0) | (ids > self.n_items)).any()):
            raise ValueError("candidate_ids outside [0, n_items]")

    def score_candidates(
        self, hist_ids: Tensor, candidate_ids: Tensor,
        diagnostics: Optional[Tensor] = None,
    ) -> Tensor:
        """Score a shared 1D item chunk using the same representations as training.

        For many chunks, callers can compute ``user_vectors`` once and multiply
        them by ``item_vectors(chunk).T`` with ``score_scale``. This avoids
        allocating the complete catalog score matrix and repeated user encoding.
        """
        self._check_candidate_ids(candidate_ids)
        user = self.user_vectors(hist_ids, diagnostics)
        scores = self.score_scale * (user @ self._item_vectors(candidate_ids).T)
        return scores.masked_fill(candidate_ids.eq(0).unsqueeze(0), float("-inf"))


def initialize_from_teacher(student: FAFIRecModel, teacher: FAFIRecModel) -> None:
    """Copy only collaborative ID/position/encoder parameters from a CF teacher.

    The student's text projection and all control/gate initializations are
    retained. The copy does not alias tensors, freeze the student, or modify the
    teacher. Caller-controlled seeds give matched fresh text/gate initial values.
    """
    if teacher.control != "cf":
        raise ValueError("teacher must be collaborative-only (control='cf')")
    if (student.n_items, student.d_model, student.max_len, student.n_heads, student.n_layers) != (
        teacher.n_items, teacher.d_model, teacher.max_len, teacher.n_heads, teacher.n_layers
    ):
        raise ValueError("teacher and student must have matching collaborative architectures")
    prefixes = ("item_id_table.", "pos_emb.", "input_norm.", "blocks.", "output_norm.")
    source = {k: v for k, v in teacher.state_dict().items() if k.startswith(prefixes)}
    target = student.state_dict()
    expected = {k for k in target if k.startswith(prefixes)}
    if set(source) != expected or any(source[k].shape != target[k].shape for k in expected):
        raise ValueError("teacher collaborative state keys or shapes do not match student")
    with torch.no_grad():
        for key in expected:
            target[key].copy_(source[key])
