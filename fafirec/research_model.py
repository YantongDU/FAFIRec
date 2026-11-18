"""CF-preserving semantic residuals for controlled, exploratory experiments.

The independent expert follows the separation principle of PAD (SIGIR 2025,
https://arxiv.org/abs/2412.04107), the optional identity contrast follows RLMRec
(WWW 2024, https://arxiv.org/abs/2310.15950), and cached frozen text follows the
decoupled adaptation principle of IISAN (SIGIR 2024,
https://arxiv.org/abs/2404.02059). This is an original implementation of a new
experimental combination, not a reproduction of those methods.
"""
from __future__ import annotations

import copy
import hashlib
import math
from contextlib import nullcontext
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ml1m_model import PaperFAFIRecModel
from .research_losses import identity_infonce, masked_catalog_ce


class _IndependentEncoder(nn.Module):
    def __init__(self, anchor):
        super().__init__()
        self.n_heads = anchor.n_heads
        for name in ('pos_emb', 'input_norm', 'input_dropout', 'blocks', 'output_norm'):
            setattr(self, name, copy.deepcopy(getattr(anchor, name)))
        self.register_buffer('_future_mask', anchor._future_mask.detach().clone(), persistent=False)
        self.requires_grad_(True)

    def forward(self, tokens, valid):
        length = valid.shape[1]
        positions = valid.long().cumsum(dim=1) * valid.long()
        x = self.input_dropout(self.input_norm(tokens + self.pos_emb(positions)))
        x = x.masked_fill(~valid.unsqueeze(-1), 0.)
        blocked = self._future_mask[:length, :length].unsqueeze(0) | ~valid.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=valid.device).unsqueeze(0)
        blocked = blocked & ~(~valid.unsqueeze(-1) & diagonal)
        blocked = blocked.repeat_interleave(self.n_heads, dim=0)
        for block in self.blocks:
            x = block(x, blocked, valid)
        return F.normalize(self.output_norm(x).masked_fill(~valid.unsqueeze(-1), 0.), dim=-1)


class ResearchResidualModel(nn.Module):
    """Frozen CF anchor plus an independently trained, factorized score residual.

    ``relative_features`` returns unstandardized, detached causal features. Its
    first two features summarize *previous* predictions whose outcomes are now
    observed in ``hist_ids``; they never read the pending next-item target.
    A full-data trained expert makes these training signals in-sample: this
    method does not claim cross-fitting or sample-outcome calibration.
    """
    def __init__(self, anchor: PaperFAFIRecModel, text_features, records,
                 kind='text', objective='bpr', alignment_weight=0., auxiliary_weight=.1,
                 score_scale=8., ridge=1., probe_negatives=4, probe_window=4,
                 ce_chunk_size=1024, seed=2025, route='global'):
        super().__init__()
        if not isinstance(anchor, PaperFAFIRecModel) or anchor.control != 'cf':
            raise ValueError('anchor must be an independently selected CF PaperFAFIRecModel')
        if kind not in {'text', 'id', 'shuffled'} or objective not in {'bpr', 'ce'}:
            raise ValueError('Unknown expert kind or objective')
        if route not in {'global', 'frequency', 'relative'}:
            raise ValueError('Unknown routing policy')
        for name, value in [('alignment_weight', alignment_weight), ('auxiliary_weight', auxiliary_weight),
                            ('score_scale', score_scale), ('ridge', ridge)]:
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if score_scale <= 0 or ridge <= 0:
            raise ValueError('score_scale and ridge must be positive')
        for value in (probe_negatives, probe_window, ce_chunk_size):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError('Probe counts and chunk size must be positive integers')
        if anchor.n_items < 2:
            raise ValueError('Relative probes require at least two real items')
        self.anchor = copy.deepcopy(anchor).requires_grad_(False).eval()
        self.n_items, self.d_model, self.max_len = anchor.n_items, anchor.d_model, anchor.max_len
        self.n_heads, self.n_layers = anchor.n_heads, anchor.n_layers
        self.kind, self.objective = kind, objective
        self.control = f'research_{kind}_{objective}'
        self.score_scale = float(anchor.score_scale)
        self.expert_score_scale = float(score_scale)
        self.alignment_weight, self.auxiliary_weight = float(alignment_weight), float(auxiliary_weight)
        self.ridge, self.seed = float(ridge), int(seed)
        self.probe_negatives, self.probe_window = int(probe_negatives), int(probe_window)
        self.ce_chunk_size = int(ce_chunk_size)
        self.expert_encoder = _IndependentEncoder(anchor)
        self.gain_raw = nn.Parameter(torch.zeros((), device=anchor.item_id_table.weight.device))
        self.router = nn.Linear(4, 1, device=self.gain_raw.device)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.register_buffer('router_mean', torch.zeros(4, device=self.gain_raw.device))
        self.register_buffer('router_std', torch.ones(4, device=self.gain_raw.device))

        counts = torch.zeros(self.n_items + 1, dtype=torch.long)
        if not records:
            raise ValueError('Training records must not be empty')
        for record in records:
            sequence = record['train']
            if len(sequence) < 2 or any(isinstance(i, bool) or not isinstance(i, Integral)
                                      or not 1 <= i <= self.n_items for i in sequence):
                raise ValueError('Invalid training sequence for residual initialization')
            counts += torch.bincount(torch.tensor(sequence), minlength=self.n_items + 1)
        self.register_buffer('training_counts', counts.to(self.gain_raw.device))
        seen = counts.nonzero().flatten()
        self.register_buffer('text_features', torch.empty(0, device=self.gain_raw.device), persistent=False)
        self.register_buffer('feature_mean', torch.empty(0, device=self.gain_raw.device))
        permutation = torch.arange(self.n_items + 1)
        self.id_table, self.projection = None, None
        report = {'kind': kind, 'training_seen_items': len(seen), 'training_events': int(counts.sum()),
                  'heldout_interactions_used': False, 'anchor_frozen': True,
                  'independent_encoder': True, 'ridge': self.ridge}
        if kind == 'id':
            self.id_table = copy.deepcopy(anchor.item_id_table).requires_grad_(True)
        else:
            features = torch.as_tensor(text_features).detach().to(device='cpu', dtype=torch.float64).clone()
            if features.ndim != 2 or features.shape[0] != self.n_items + 1 or not torch.isfinite(features).all():
                raise ValueError('Text features must be finite, with PAD and every catalog item')
            features[0].zero_()
            if kind == 'shuffled':
                generator = torch.Generator().manual_seed(self.seed)
                permutation[1:] = torch.randperm(self.n_items, generator=generator) + 1
                features = features[permutation]
            x = features[seen]
            target = anchor.item_id_table.weight.detach().cpu().to(torch.float64)[seen]
            x_mean, target_mean = x.mean(0), target.mean(0)
            centered, y = x - x_mean, target - target_mean
            # Choose the smaller primal/dual solve; regularization is SSE + ridge*||W||^2.
            if centered.shape[0] < centered.shape[1]:
                gram = centered @ centered.T
                gram.diagonal().add_(self.ridge)
                weight = centered.T @ torch.linalg.solve(gram, y)
            else:
                gram = centered.T @ centered
                gram.diagonal().add_(self.ridge)
                weight = torch.linalg.solve(gram, centered.T @ y)
            self.feature_mean = x_mean.to(device=self.gain_raw.device, dtype=torch.float32)
            self.text_features = features.to(device=self.gain_raw.device, dtype=torch.float32)
            self.projection = nn.Linear(features.shape[1], self.d_model, bias=True,
                                        device=self.gain_raw.device)
            with torch.no_grad():
                self.projection.weight.copy_(weight.T.to(self.projection.weight))
                self.projection.bias.copy_(target_mean.to(self.projection.bias))
            predicted = centered @ weight + target_mean
            report.update({'projection': 'train-seen centered affine ridge',
                           'ridge_objective': 'SSE + ridge*||W||^2; unpenalized intercept',
                           'solve_dtype_device': 'float64 cpu',
                           'feature_dim': features.shape[1],
                           'relative_squared_error': float((predicted - target).square().sum()
                                                           / target.square().sum().clamp_min(1e-12))})
        self.register_buffer('feature_permutation', permutation.to(self.gain_raw.device))
        report['permutation_sha256'] = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
        self.initialization_report = report
        self._catalog_cache = None
        self.alpha_multiplier, self.route, self.stage = 1., 'global', 'expert'
        self.set_stage('expert')
        self.set_calibration(1., route)

    def model_config(self):
        return {'kind': self.kind, 'objective': self.objective, 'alignment_weight': self.alignment_weight,
                'auxiliary_weight': self.auxiliary_weight, 'score_scale': self.expert_score_scale,
                'ridge': self.ridge, 'probe_negatives': self.probe_negatives,
                'probe_window': self.probe_window, 'ce_chunk_size': self.ce_chunk_size,
                'seed': self.seed, 'route': self.route}

    def set_calibration(self, multiplier=1., route=None):
        if isinstance(multiplier, bool) or not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError('Calibration multiplier must be finite and nonnegative')
        if route is not None and route not in {'global', 'frequency', 'relative'}:
            raise ValueError('Unknown routing policy')
        self.alpha_multiplier = float(multiplier)
        if route is not None:
            self.route = route
        return self

    @torch.no_grad()
    def set_router_statistics(self, mean, std):
        mean = torch.as_tensor(mean, device=self.router_mean.device, dtype=self.router_mean.dtype)
        std = torch.as_tensor(std, device=self.router_std.device, dtype=self.router_std.dtype)
        if mean.shape != (4,) or std.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError('Router statistics must be finite four-vectors with nonnegative std')
        self.router_mean.copy_(mean)
        self.router_std.copy_(std.clamp_min(1e-6))
        return self

    def set_stage(self, stage):
        if stage not in {'expert', 'router'}:
            raise ValueError('Stage must be expert or router')
        self.stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('router.') if stage == 'router'
                                     else not name.startswith(('anchor.', 'router.')))
        self.route = 'global' if stage == 'expert' else 'relative'
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        if getattr(self, 'stage', None) == 'router':
            self.expert_encoder.eval()
        return self

    def _expert_catalog(self):
        if self._catalog_cache is not None:
            return self._catalog_cache
        table = self.id_table.weight if self.kind == 'id' else self.projection(self.text_features - self.feature_mean)
        return table.masked_fill(torch.arange(self.n_items + 1, device=table.device).eq(0).unsqueeze(-1), 0.)

    def _expert_items(self):
        return F.normalize(self._expert_catalog(), dim=-1)

    def _states(self, hist_ids):
        hist_ids = self.anchor._check_ids(hist_ids, 'hist_ids')
        valid = hist_ids.ne(0)
        with torch.no_grad():
            self.anchor.eval()
            cf = self.anchor._encode_tokens(self.anchor.item_id_table(hist_ids), valid)
        context = torch.no_grad() if self.stage == 'router' else nullcontext()
        with context:
            expert = self.expert_encoder(F.embedding(hist_ids, self._expert_catalog()), valid)
        return cf, expert, valid

    def branch_states(self, hist_ids):
        """Independent normalized states; the CF result is always detached."""
        old = self._catalog_cache
        self._catalog_cache = self._expert_catalog()
        try:
            cf, expert, _ = self._states(hist_ids)
            return cf, expert
        finally:
            self._catalog_cache = old

    def branch_user_vectors(self, hist_ids):
        cf, expert = self.branch_states(hist_ids)
        valid = hist_ids.ne(0)
        return self.anchor._last(cf, valid), self.anchor._last(expert, valid)

    def branch_item_vectors(self):
        with torch.no_grad():
            cf = self.anchor.item_vectors()
        return cf, self._expert_items()

    def _item_gate(self):
        gate = torch.ones_like(self.training_counts, dtype=self.gain_raw.dtype)
        if self.route == 'frequency':
            gate = 1. / (1. + torch.log1p(self.training_counts.to(gate.dtype)))
        return gate.masked_fill(torch.arange(len(gate), device=gate.device).eq(0), 0.)

    def _user_gate(self, hist_ids, cf, expert, valid):
        if self.route == 'relative':
            features = self.relative_features(hist_ids, cf, expert)
            standardized = (features - self.router_mean) / self.router_std
            return (2. * torch.sigmoid(self.router(standardized).squeeze(-1))).masked_fill(~valid, 0.)
        return valid.to(cf.dtype)

    def _query_vectors(self, hist_ids, cf, expert, valid):
        gate = self._user_gate(hist_ids, cf, expert, valid)
        gain = self.alpha_multiplier * torch.tanh(self.gain_raw) * self.expert_score_scale / self.score_scale
        return torch.cat((cf, gain * gate.unsqueeze(-1) * expert), dim=-1)

    def user_vectors(self, hist_ids, diagnostics=None):
        if self.alpha_multiplier == 0.:
            with torch.no_grad():
                self.anchor.eval()
                return self.anchor.user_vectors(hist_ids)
        old = self._catalog_cache
        self._catalog_cache = self._expert_catalog()
        try:
            cf, expert, valid = self._states(hist_ids)
            return self.anchor._last(self._query_vectors(hist_ids, cf, expert, valid), valid)
        finally:
            self._catalog_cache = old

    def item_vectors(self, ids=None):
        if ids is not None:
            self.anchor._check_candidate_ids(ids)
        with torch.no_grad():
            cf = self.anchor.item_vectors(ids)
        if self.alpha_multiplier == 0.:
            return cf
        expert = self._expert_items() * self._item_gate().unsqueeze(-1)
        if ids is not None:
            expert = expert[ids]
        return torch.cat((cf, expert), dim=-1)

    def score_all(self, hist_ids, diagnostics=None):
        scores = self.score_scale * (self.user_vectors(hist_ids) @ self.item_vectors().T)
        return scores.masked_fill(torch.arange(self.n_items + 1, device=scores.device).eq(0).unsqueeze(0), -torch.inf)

    @torch.no_grad()
    def relative_features(self, hist_ids, cf_states=None, expert_states=None):
        hist_ids = self.anchor._check_ids(hist_ids, 'hist_ids')
        if cf_states is None or expert_states is None:
            cf_states, expert_states, valid = self._states(hist_ids)
        else:
            valid = hist_ids.ne(0)
        cf_states, expert_states = cf_states.detach(), expert_states.detach()
        cf_items, expert_items = self.anchor.item_vectors().detach(), self._expert_items().detach()
        batch, length = hist_ids.shape
        ranks = valid.long().cumsum(1)
        offsets = torch.arange(self.probe_negatives, device=hist_ids.device)
        hashed = (hist_ids.unsqueeze(-1) * 1103515245 + ranks.unsqueeze(-1) * 12345
                  + self.seed + offsets * 2654435761)
        negatives = hashed.remainder(self.n_items - 1) + 1
        negatives = negatives + (negatives >= hist_ids.unsqueeze(-1)).long()
        previous_valid = valid & F.pad(valid[:, :-1], (1, 0), value=False)
        previous_cf = F.pad(cf_states[:, :-1], (0, 0, 1, 0))
        previous_expert = F.pad(expert_states[:, :-1], (0, 0, 1, 0))

        def probe_loss(states, items):
            positive = F.embedding(hist_ids, items)
            negative = F.embedding(negatives, items)
            margin = 8. * (states.unsqueeze(-2) * (positive.unsqueeze(-2) - negative)).sum(-1)
            return F.softplus(-margin).mean(-1)

        gap = (probe_loss(previous_cf, cf_items) - probe_loss(previous_expert, expert_items)).masked_fill(~previous_valid, 0.)

        def rolling(values):
            cumulative = values.cumsum(1)
            if self.probe_window < length:
                cumulative = cumulative - F.pad(cumulative[:, :-self.probe_window], (self.probe_window, 0))
            return cumulative

        count = rolling(previous_valid.to(cf_states.dtype))
        mean = rolling(gap) / count.clamp_min(1.)
        variance = (rolling(gap.square()) / count.clamp_min(1.) - mean.square()).clamp_min(0.)
        std = variance.sqrt()

        def entropy(states, items):
            flat = states.reshape(-1, self.d_model)
            result = torch.empty(flat.shape[0], device=flat.device, dtype=flat.dtype)
            for start in range(0, len(flat), self.ce_chunk_size):
                logits = 8. * (flat[start:start + self.ce_chunk_size] @ items[1:].T)
                log_z = torch.logsumexp(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                result[start:start + self.ce_chunk_size] = log_z - (probs * logits).sum(-1)
            return result.reshape(batch, length)

        entropy_gap = entropy(cf_states, cf_items) - entropy(expert_states, expert_items)
        features = torch.stack((mean, std, entropy_gap, torch.log1p(ranks.to(cf_states.dtype))), dim=-1)
        return features.masked_fill(~valid.unsqueeze(-1), 0.).detach()

    def forward(self, hist_ids, pos_ids, neg_ids, seen_mask=None):
        return self.loss_batch(hist_ids, pos_ids, neg_ids, seen_mask)

    def loss_batch(self, hist_ids, pos_ids, neg_ids, seen_mask=None):
        hist_ids = self.anchor._check_ids(hist_ids, 'hist_ids')
        pos_ids = self.anchor._check_ids(pos_ids, 'pos_ids')
        neg_ids = self.anchor._check_ids(neg_ids, 'neg_ids')
        if pos_ids.shape != hist_ids.shape or neg_ids.shape != hist_ids.shape:
            raise ValueError('Aligned history, positive and negative tensors are required')
        supervised = hist_ids.ne(0) & pos_ids.ne(0)
        if ((neg_ids == 0) & supervised).any():
            raise ValueError('Supervised positions require non-PAD negatives')
        if self._catalog_cache is not None:
            raise RuntimeError('Nested loss calls are unsupported')
        self._catalog_cache = self._expert_catalog()
        try:
            cf, expert, valid = self._states(hist_ids)
            query = self._query_vectors(hist_ids, cf, expert, valid)
            with torch.no_grad():
                cf_items = self.anchor.item_vectors()
            expert_items = self._expert_items()
            items = torch.cat((cf_items, self._item_gate().unsqueeze(-1) * expert_items), dim=-1)
            zero = self.router.weight.sum() * 0. if self.stage == 'router' else self.gain_raw * 0.
            rec, auxiliary, alignment = zero, zero, zero
            if supervised.any():
                if self.objective == 'bpr':
                    margin = self.score_scale * (query * (F.embedding(pos_ids, items) - F.embedding(neg_ids, items))).sum(-1)
                    rec = F.softplus(-margin[supervised]).mean()
                    if self.stage == 'expert':
                        expert_margin = self.expert_score_scale * (expert * (F.embedding(pos_ids, expert_items) - F.embedding(neg_ids, expert_items))).sum(-1)
                        auxiliary = F.softplus(-expert_margin[supervised]).mean()
                else:
                    owners = torch.arange(len(hist_ids), device=hist_ids.device).unsqueeze(1).expand_as(hist_ids)[supervised]
                    targets = pos_ids[supervised]
                    rec = masked_catalog_ce(query[supervised], items, targets, owners, seen_mask,
                                            self.score_scale, self.ce_chunk_size)
                    if self.stage == 'expert':
                        auxiliary = masked_catalog_ce(expert[supervised], expert_items, targets, owners,
                                                      seen_mask, self.expert_score_scale, self.ce_chunk_size)
                if self.stage == 'expert' and self.alignment_weight > 0:
                    unique = torch.unique(torch.cat((hist_ids.flatten(), pos_ids.flatten())))
                    unique = unique[(unique > 0) & (self.training_counts[unique] > 0)]
                    if len(unique):
                        alignment = identity_infonce(expert_items[unique], cf_items[unique].detach(), temperature=.1)
            loss = rec + self.auxiliary_weight * auxiliary + self.alignment_weight * alignment
            return {'loss': loss, 'rec_loss': rec, 'aux_loss': auxiliary, 'align_loss': alignment}
        finally:
            self._catalog_cache = None
