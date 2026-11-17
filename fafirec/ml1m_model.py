"""Memory-efficient execution of the unchanged FAFIRec integration equations.

Projecting the small movie catalog once avoids retaining B*L*feature_dim text
copies for history, positive and negative lookups at the paper's batch2048.
The projection is linear and has no dropout, so its placement commutes with
lookup; the same autograd graph accumulates gradients for repeated items.
"""
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import FAFIRecModel


class PaperFAFIRecModel(FAFIRecModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._projected_catalog = None

    def _text_table(self):
        if self._projected_catalog is not None:
            return self._projected_catalog
        return self.proj_content(self.text_features)

    def encode_branches(self, hist_ids: Tensor, diagnostics: Optional[Tensor] = None):
        hist_ids = self._check_ids(hist_ids, 'hist_ids')
        valid = hist_ids.ne(0)
        ids = self.item_id_table(hist_ids)
        if self.control == 'cf':
            return super().encode_branches(hist_ids, diagnostics)
        text = F.embedding(hist_ids, self._text_table(), padding_idx=0)
        encoded = self._encode_tokens(torch.cat((ids, text), dim=0), valid.repeat(2, 1))
        collaborative, textual = encoded.chunk(2, dim=0)
        scores, rho = self._control_scores(valid, diagnostics)
        gate = self._fusion_gate(scores, rho, (collaborative-textual).abs())
        fused = F.normalize(gate*collaborative+(1-gate)*textual, dim=-1)
        fused = fused.masked_fill(~valid.unsqueeze(-1), 0)
        return {'fused':fused, 'collaborative':collaborative, 'textual':textual,
                'scores':scores, 'rho':rho, 'gate':gate, 'valid':valid}

    def _item_vectors(self, ids=None):
        if self.control == 'cf':
            return super()._item_vectors(ids)
        collaborative = self.item_id_table.weight if ids is None else self.item_id_table(ids)
        textual = self._text_table() if ids is None else F.embedding(ids, self._text_table(), padding_idx=0)
        gate = torch.sigmoid(self.item_gate_logit)
        normalized = F.normalize(gate*collaborative+(1-gate)*textual, dim=-1)
        if ids is None:
            return torch.cat((torch.zeros_like(normalized[:1]),normalized[1:]),dim=0)
        return normalized.masked_fill(ids.eq(0).unsqueeze(-1),0)

    def forward(self, hist_ids, pos_ids, neg_ids, diagnostics=None):
        if self._projected_catalog is not None:
            raise RuntimeError('Nested forward calls are unsupported')
        if self.control != 'cf':
            self._projected_catalog = self.proj_content(self.text_features)
        try:
            return super().forward(hist_ids,pos_ids,neg_ids,diagnostics)
        finally:
            self._projected_catalog = None
