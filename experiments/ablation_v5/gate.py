"""Matched, fixed-budget ablations of the frozen v4 candidate controller.

Every variant retains the original 17 registered parameters. Removed inputs
are zero after the same training-only standardization, and removed candidate
terms are omitted from the forward computation. Inactive parameters remain in
the optimizer and in the 17-value mean L2 penalty, avoiding a change in the
regularization denominator when a mechanism is removed.

This module accepts an authenticated *internal calibration* cache from its
caller. It has no file/data loader or validation/test evaluation entry point.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from experiments.diagnostic_v4.fusion_model import DiagnosticGate, require
from fafirec.runner import seed_everything


SPEC = {
    'full': {'feature_mask': [1, 1, 1, 1, 1, 1, 1], 'pop_term': True, 'conflict_term': True},
    'gain_only': {'feature_mask': [0, 0, 0, 0, 0, 0, 0], 'pop_term': False, 'conflict_term': False},
    'length_candidate': {'feature_mask': [1, 0, 0, 0, 0, 0, 0], 'pop_term': True, 'conflict_term': True},
    'no_length': {'feature_mask': [0, 1, 1, 1, 1, 1, 1], 'pop_term': True, 'conflict_term': True},
    'user_only': {'feature_mask': [1, 1, 1, 1, 1, 1, 1], 'pop_term': False, 'conflict_term': False},
    'no_past': {'feature_mask': [1, 0, 0, 1, 1, 0, 0], 'pop_term': True, 'conflict_term': True},
    'no_current': {'feature_mask': [1, 1, 1, 0, 0, 1, 1], 'pop_term': True, 'conflict_term': True},
    'no_pop': {'feature_mask': [1, 1, 1, 1, 1, 1, 1], 'pop_term': False, 'conflict_term': True},
    'no_conflict': {'feature_mask': [1, 1, 1, 1, 1, 1, 1], 'pop_term': True, 'conflict_term': False},
}
# Both names are exposed for callers that index one specification or enumerate
# the fixed experiment matrix. Contents must be frozen before an actual run.
SPECS = SPEC
TRAINING_RECIPE = {
    'epochs': 20, 'lr': .02, 'batch_size': 256, 'alpha': .1, 'beta': .2,
    'l2': .001, 'initial_seed_offset': 9000, 'order_seed_offset': 10000,
    'registered_parameter_count': 17,
}


class AblatedGate(DiagnosticGate):
    """Candidate gate with explicit input/term removal and matched L2 scale."""
    def __init__(self, variant='full'):
        require(variant in SPEC, 'Unknown fixed ablation variant')
        # Do not add or reorder any trainable parameter after this constructor.
        super().__init__(n_features=7, mode='candidate')
        self.variant = variant
        specification = SPEC[variant]
        self.register_buffer('ablation_feature_mask', torch.tensor(specification['feature_mask'], dtype=torch.float32))
        self.register_buffer('ablation_terms', torch.tensor(
            [specification['pop_term'], specification['conflict_term']], dtype=torch.bool))
        require(sum(p.numel() for p in self.parameters() if p.requires_grad) == 17,
                'Ablations must retain the original 17-parameter optimizer/L2 space')

    def _standardized(self, features):
        standardized = super()._standardized(features)
        # Preserve the original floating operation graph in the full variant.
        if self.variant == 'full':
            return standardized
        return standardized * self.ablation_feature_mask

    def forward(self, sem_scores, cf_scores, features, item_counts=None, alpha=.1):
        # For full and feature-only removals the parent preserves the original
        # addition/multiplication order; our _standardized supplies the mask.
        if bool(self.ablation_terms.all()):
            return super().forward(sem_scores, cf_scores, features, item_counts, alpha)
        require(isinstance(alpha, Real) and not isinstance(alpha, bool) and math.isfinite(alpha) and alpha >= 0,
                'Finite nonnegative residual multiplier required')
        return sem_scores + float(alpha) * self.multipliers(sem_scores, cf_scores, features, item_counts) * cf_scores

    def multipliers(self, sem_scores, cf_scores, features, item_counts=None):
        """Return the detached-input gate values with [users,catalog] shape.

        This exposes the exact controller computation for predefined
        sensitivity analyses. It does not read outcome labels or refit any
        statistics. Parameters still carry gradients when used for training.
        """
        require(sem_scores.ndim == 2 and cf_scores.shape == sem_scores.shape
            and sem_scores.shape[0] == len(features) and sem_scores.device == cf_scores.device == features.device
            and sem_scores.dtype == cf_scores.dtype == features.dtype
            and torch.isfinite(sem_scores).all().item() and torch.isfinite(cf_scores).all().item(),
            'Gate score/feature shape, dtype or device differs')
        require(self.item_statistics_fitted.item(), 'Fit item statistics using inner train counts first')
        counts = self._counts(item_counts)
        require(len(counts) == sem_scores.shape[1] == self.item_catalog_size.item() + 1,
                'Inference training-count catalog mapping differs')
        logits = self.coefficients(features)[:, None]
        use_pop, use_conflict = bool(self.ablation_terms[0]), bool(self.ablation_terms[1])
        if use_pop:
            pop = (counts.log1p() - self.item_logcount_mean) / self.item_logcount_std
            pop = pop.clone()
            pop[0] = 0.
            slope = self.pop_slope(self._standardized(features))
            if use_conflict:
                # Same grouping as the frozen original candidate forward.
                logits = logits + slope * pop[None] + self.conflict_coef * (cf_scores - sem_scores).clamp(-5., 5.)
            else:
                logits = logits + slope * pop[None]
        elif use_conflict:
            logits = logits + self.conflict_coef * (cf_scores - sem_scores).clamp(-5., 5.)
        return (2. * torch.sigmoid(logits)).expand_as(sem_scores)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # An apparently valid checkpoint must not silently change which
        # mechanism the constructor promised to remove.
        for name in ('ablation_feature_mask', 'ablation_terms'):
            if name in state_dict:
                expected = getattr(self, name).detach().cpu()
                actual = state_dict[name].detach().cpu()
                require(torch.equal(actual, expected), 'Checkpoint ablation specification differs: ' + name)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def make_gate(variant):
    return AblatedGate(variant)


def fit_ablation(data, variant, seed, device):
    """Fit one fixed ablation using only supplied internal-calibration labels.

    Operation order, epoch minibatches, objective reduction, initialization and
    optimizer match v4 ``fit_gate(..., mode='candidate', l2=.001)``. No early
    stopping, hyperparameter search or outer-label access occurs here.
    """
    require(type(seed) is int and seed >= 0, 'Nonnegative integer training seed required')
    require(isinstance(data, dict) and {'sem', 'cf', 'features', 'available', 'targets', 'counts', 'pop'} <= set(data),
            'Complete authenticated internal calibration cache required')
    seed_everything(seed + TRAINING_RECIPE['initial_seed_offset'])
    model = make_gate(variant)
    model.fit_feature_statistics(data['features'])
    model.fit_item_statistics(data['counts'])
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    require(sum(p.numel() for p in parameters) == TRAINING_RECIPE['registered_parameter_count'],
            'Optimizer/L2 denominator changed across ablations')
    optimizer = torch.optim.Adam(parameters, lr=TRAINING_RECIPE['lr'])
    require(data['targets'] and all(ids for ids in data['targets']), 'Nonempty internal target block required per user')
    maximum = max(map(len, data['targets']))
    target = torch.empty(len(data['targets']), maximum, dtype=torch.long)
    target_mask = torch.zeros_like(target, dtype=torch.float32)
    for u, ids in enumerate(data['targets']):
        target[u].fill_(ids[0]); target[u, :len(ids)] = torch.tensor(ids)
        target_mask[u, :len(ids)] = 1. / len(ids)
        require(bool(data['available'][u, ids].all()), 'Calibration target already observed')
    logs = []
    model.train()
    for epoch in range(1, TRAINING_RECIPE['epochs'] + 1):
        order = torch.randperm(len(target), generator=torch.Generator().manual_seed(
            seed + TRAINING_RECIPE['order_seed_offset'] + epoch))
        total = 0.
        for index in order.split(TRAINING_RECIPE['batch_size']):
            sem = (data['sem'][index] + TRAINING_RECIPE['beta'] * data['pop'][index]).to(device)
            scores = model(sem, data['cf'][index].to(device), data['features'][index].to(device),
                data['counts'].to(device), alpha=TRAINING_RECIPE['alpha'])
            logp = scores.masked_fill(~data['available'][index].to(device), -torch.inf).log_softmax(1)
            loss = -(logp.gather(1, target[index].to(device)) * target_mask[index].to(device)).sum(1).mean()
            penalty = torch.cat([p.flatten() for p in parameters]).square().mean()
            objective = loss + TRAINING_RECIPE['l2'] * penalty
            require(bool(torch.isfinite(objective)), 'Nonfinite diagnostic ablation loss')
            optimizer.zero_grad(set_to_none=True)
            objective.backward(); optimizer.step()
            total += float(loss.detach()) * len(index)
        logs.append({'epoch': epoch, 'user_mean_calibration_ce': total / len(target)})
    return model.cpu().eval(), logs
