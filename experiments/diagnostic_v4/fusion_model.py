"""Observed-prefix diagnostics and a small collaborative/semantic score gate.

There is no pending-target argument. Probe outcomes come exclusively from the
observed suffix after a train-internal cutoff. Candidate statistics exclude PAD
and all observed identities; unavailable cached logits are zero placeholders and
MUST be masked using the returned ``available`` matrix by the caller.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn

FEATURE_NAMES = ('log1p_history_length', 'past_nll_cf_minus_sem_mean',
    'past_nll_cf_minus_sem_std', 'current_entropy_cf_minus_sem_over_log_catalog',
    'current_top20_jaccard', 'observed_probe_fraction_of_two',
    'past_rank_cf_minus_sem_over_catalog_mean')
HISTORY_LIMIT = 50
STD_FLOOR = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probe_tasks(histories, probe_start, probe_window=2):
    """Return causal (owner, position, prefix, observed_positive) probe tuples.

    Only the most recent <=2 observed events at/after ``probe_start`` qualify.
    A repeated positive already in its prefix is skipped rather than restored
    into the candidate set. Empty-prefix probes are also skipped.
    """
    require(type(probe_window) is int and 0 <= probe_window <= 2, 'Probe window must be 0, 1 or 2')
    starts = probe_start.tolist() if isinstance(probe_start, torch.Tensor) else list(probe_start)
    require(len(starts) == len(histories), 'One internal cutoff required per history')
    tasks, repeated = [], 0
    for owner, (history, cutoff) in enumerate(zip(histories, starts)):
        require(isinstance(cutoff, Integral) and not isinstance(cutoff, bool)
            and 0 <= cutoff <= len(history), 'Invalid internal probe cutoff')
        for position in range(max(int(cutoff), len(history) - probe_window, 1), len(history)):
            prefix, positive = list(history[:position]), int(history[position])
            if positive in prefix:
                repeated += 1
            else:
                tasks.append((owner, position, prefix, positive))
    return tasks, repeated


def candidate_standardize(scores, available):
    """Population mean/std over available candidates, with zero masked slots."""
    require(scores.ndim == 2 and available.shape == scores.shape and available.dtype == torch.bool,
            'Score/candidate tensors must have matching matrix shapes')
    require(scores.is_floating_point() and torch.isfinite(scores).all().item(), 'Finite floating scores required')
    count = available.sum(1, keepdim=True)
    require((count > 0).all().item(), 'At least one available candidate per prefix required')
    mean = scores.masked_fill(~available, 0.).sum(1, keepdim=True) / count
    variance = (scores - mean).square().masked_fill(~available, 0.).sum(1, keepdim=True) / count
    return ((scores - mean) / variance.sqrt().clamp_min(STD_FLOOR)).masked_fill(~available, 0.)


def past_features(nll_gaps, rank_gaps, valid):
    """Pure reduction of already-observed probe diagnostics (no item labels)."""
    require(nll_gaps.ndim == 2 and rank_gaps.shape == valid.shape == nll_gaps.shape
        and valid.dtype == torch.bool and nll_gaps.shape[1] <= 2, 'Probe statistic matrices differ')
    require(torch.isfinite(nll_gaps).all().item() and torch.isfinite(rank_gaps).all().item(),
            'Finite probe statistics required')
    n = valid.sum(1).to(nll_gaps.dtype)
    mean = nll_gaps.masked_fill(~valid, 0.).sum(1) / n.clamp_min(1.)
    var = (nll_gaps - mean[:, None]).square().masked_fill(~valid, 0.).sum(1) / n.clamp_min(1.)
    rank_mean = rank_gaps.masked_fill(~valid, 0.).sum(1) / n.clamp_min(1.)
    return mean, var.clamp_min(0.).sqrt(), n / 2., rank_mean


def features_from_scores(sem_scores, cf_scores, available, history_lengths,
                         probe_nll_gaps=None, probe_rank_gaps=None, probe_valid=None):
    """Pure, detached seven-feature construction from scores and past summaries.

    Inputs are candidate-standardized scores. Current entropy uses the same
    available mask for both experts and divides by log(non-PAD catalog size).
    Missing probes have zero gap statistics and an explicit zero fraction.
    """
    require(sem_scores.ndim == 2 and sem_scores.shape == cf_scores.shape == available.shape,
            'Expert scores/candidate shapes differ')
    require(available.dtype == torch.bool and sem_scores.shape[1] > 2
        and torch.isfinite(sem_scores).all().item() and torch.isfinite(cf_scores).all().item(),
        'Finite expert scores and a nontrivial catalog required')
    require((available.sum(1) > 0).all().item() and not available[:, 0].any().item(), 'Invalid available candidates')
    device, dtype = sem_scores.device, sem_scores.dtype
    lengths = torch.as_tensor(history_lengths, dtype=dtype, device=device)
    require(lengths.shape == (len(sem_scores),) and (lengths > 0).all().item(), 'Positive history lengths required')
    def entropy(scores):
        masked = scores.masked_fill(~available, -torch.inf)
        return torch.logsumexp(masked, 1) - (masked.softmax(1) * scores.masked_fill(~available, 0.)).sum(1)
    entropy_gap = (entropy(cf_scores) - entropy(sem_scores)) / math.log(sem_scores.shape[1] - 1)
    k = min(20, sem_scores.shape[1] - 1)
    def top_mask(scores):
        ordered = torch.argsort(scores.masked_fill(~available, -torch.inf), dim=1, descending=True, stable=True)[:, :k]
        return torch.zeros_like(available).scatter_(1, ordered, True) & available
    a, b = top_mask(sem_scores), top_mask(cf_scores)
    jaccard = (a & b).sum(1).to(dtype) / (a | b).sum(1).clamp_min(1)
    if probe_nll_gaps is None:
        require(probe_rank_gaps is None and probe_valid is None, 'Incomplete past probe summaries')
        probe_nll_gaps = torch.zeros(len(sem_scores), 0, dtype=dtype, device=device)
        probe_rank_gaps = torch.zeros_like(probe_nll_gaps)
        probe_valid = torch.zeros_like(probe_nll_gaps, dtype=torch.bool)
    mean, std, fraction, rank_mean = past_features(probe_nll_gaps, probe_rank_gaps, probe_valid)
    return torch.stack((lengths.log1p(), mean, std, entropy_gap, jaccard, fraction, rank_mean), 1).detach()


@torch.no_grad()
def score_histories(semantic_model, B, histories, probe_start, probe_window=2, batch_size=128):
    """Cache normalized scores, candidate mask and observed-prefix diagnostics.

    Both branches receive the last 50 observed items. CF inputs are binary, while
    all observed identities (including older ones) are excluded from candidates.
    ``B`` is a CPU float64 [items,items] operator without a PAD row/column.
    The returned tensors are detached CPU float32 (mask: bool).
    """
    require(isinstance(B, torch.Tensor) and B.device.type == 'cpu' and B.dtype == torch.float64
        and B.ndim == 2 and B.shape[0] == B.shape[1] and B.shape[0] > 1
        and torch.isfinite(B).all().item(), 'B must be a finite square CPU float64 non-PAD operator')
    require(type(batch_size) is int and batch_size > 0, 'Positive scoring batch size required')
    require(isinstance(histories, list) and histories, 'Nonempty observed histories required')
    n_items = len(B)
    copied = []
    for history in histories:
        require(isinstance(history, (list, tuple)) and len(history) > 0
            and all(isinstance(i, Integral) and not isinstance(i, bool) and 1 <= i <= n_items for i in history),
            'Histories contain invalid observed item IDs')
        copied.append([int(i) for i in history])
    tasks, repeated = probe_tasks(copied, probe_start, probe_window)
    require(getattr(semantic_model, 'max_len', HISTORY_LIMIT) == HISTORY_LIMIT,
            'Semantic and collaborative history limits must both equal 50')
    was_training = semantic_model.training
    semantic_model.eval()
    try:
        table = semantic_model.item_vectors().detach()
        require(table.ndim == 2 and len(table) == n_items + 1 and torch.isfinite(table).all().item(),
                'Semantic catalog and CF operator mapping differ')
        device, dtype = table.device, table.dtype
        require(dtype == torch.float32, 'Semantic scoring must use float32')
        operator = B.to(device=device, dtype=torch.float32)
        ids = torch.arange(n_items + 1, device=device)
        scale = float(getattr(semantic_model, 'score_scale', 1.))
        require(math.isfinite(scale) and scale > 0, 'Finite positive semantic score scale required')
        def score_batch(batch):
            padded = [[0] * (HISTORY_LIMIT - len(h[-HISTORY_LIMIT:])) + h[-HISTORY_LIMIT:] for h in batch]
            tokens = torch.tensor(padded, dtype=torch.long, device=device)
            available = torch.ones(len(batch), n_items + 1, dtype=torch.bool)
            available[:, 0] = False
            for row, history in enumerate(batch):
                available[row, history] = False
            available = available.to(device)
            query = torch.zeros(len(batch), n_items + 1, dtype=torch.float32, device=device)
            query.scatter_(1, tokens, 1.)
            query[:, 0] = 0.
            cf_raw = torch.zeros_like(query)
            cf_raw[:, 1:] = query[:, 1:] @ operator
            sem_raw = scale * (semantic_model.user_vectors(tokens) @ table.T)
            return candidate_standardize(sem_raw, available), candidate_standardize(cf_raw, available), available
        n = len(copied)
        sem_cache = torch.empty(n, n_items + 1, dtype=torch.float32)
        cf_cache = torch.empty_like(sem_cache)
        candidate_cache = torch.empty(n, n_items + 1, dtype=torch.bool)
        features = torch.empty(n, 7, dtype=torch.float32)
        for start in range(0, n, batch_size):
            batch = copied[start:start + batch_size]
            sem, cf, available = score_batch(batch)
            end = start + len(batch)
            sem_cache[start:end], cf_cache[start:end], candidate_cache[start:end] = sem.cpu(), cf.cpu(), available.cpu()
            features[start:end] = features_from_scores(sem, cf, available, [len(h) for h in batch]).cpu()
        gaps, ranks = torch.zeros(n, 2), torch.zeros(n, 2)
        probe_mask = torch.zeros(n, 2, dtype=torch.bool)
        counts = [0] * n
        for start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[start:start + batch_size]
            sem, cf, available = score_batch([t[2] for t in batch_tasks])
            target = torch.tensor([t[3] for t in batch_tasks], device=device, dtype=torch.long)
            row = torch.arange(len(batch_tasks), device=device)
            require(available[row, target].all().item(), 'Observed probe positive was unavailable; never restore targets')
            def nll(scores):
                return torch.logsumexp(scores.masked_fill(~available, -torch.inf), 1) - scores[row, target]
            def rank(scores):
                positive = scores[row, target][:, None]
                return 1 + (available & ((scores > positive) | ((scores == positive) & (ids[None] < target[:, None])))).sum(1)
            difference = (nll(cf) - nll(sem)).cpu()
            rank_difference = ((rank(cf) - rank(sem)).to(torch.float32) / n_items).cpu()
            for j, (owner, _position, _prefix, _positive) in enumerate(batch_tasks):
                slot = counts[owner]
                require(slot < 2, 'More than two observed probes generated')
                gaps[owner, slot], ranks[owner, slot], probe_mask[owner, slot] = difference[j], rank_difference[j], True
                counts[owner] += 1
        mean, std, fraction, rank_mean = past_features(gaps, ranks, probe_mask)
        features[:, 1], features[:, 2], features[:, 5], features[:, 6] = mean, std, fraction, rank_mean
        report = {'feature_names': list(FEATURE_NAMES), 'users': n, 'items': n_items,
            'history_limit': HISTORY_LIMIT, 'cf_query': 'binary last50 observed items',
            'candidate_policy': 'exclude PAD and all observed identities, no target restoration',
            'score_normalization': 'available-candidate population mean/std; floor1e-6; masked slots zero',
            'probe_window': probe_window, 'probe_count': len(tasks), 'probe_counts': counts,
            'zero_probe_users': sum(x == 0 for x in counts), 'skipped_repeated_positive_probes': repeated,
            'probe_fraction_denominator': 2, 'rank_gap_denominator': n_items,
            'pending_targets_accepted': False, 'all_features_detached': True,
            'device': str(device), 'batch_size': batch_size}
        return sem_cache, cf_cache, features, candidate_cache, report
    finally:
        semantic_model.train(was_training)


class DiagnosticGate(nn.Module):
    """Zero-initialized score residual with train-only fixed feature statistics."""
    MODES = ('global', 'length', 'diagnostic', 'candidate')

    def __init__(self, n_features=7, mode='diagnostic'):
        super().__init__()
        require(n_features == 7 and mode in self.MODES, 'Seven diagnostic features and a known gate mode required')
        self.n_features, self.mode = n_features, mode
        self.user_linear = nn.Linear(1 if mode == 'length' else n_features, 1)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.user_linear.bias)
        self.user_linear.requires_grad_(mode != 'global')
        self.pop_slope = nn.Linear(n_features, 1) if mode == 'candidate' else None
        if self.pop_slope is not None:
            nn.init.zeros_(self.pop_slope.weight)
            nn.init.zeros_(self.pop_slope.bias)
        self.conflict_coef = nn.Parameter(torch.zeros(()), requires_grad=mode == 'candidate')
        self.register_buffer('feature_mean', torch.zeros(n_features))
        self.register_buffer('feature_std', torch.ones(n_features))
        self.register_buffer('feature_statistics_fitted', torch.tensor(False))
        self.register_buffer('item_logcount_mean', torch.zeros(()))
        self.register_buffer('item_logcount_std', torch.ones(()))
        self.register_buffer('item_statistics_fitted', torch.tensor(False))
        self.register_buffer('item_catalog_size', torch.zeros((), dtype=torch.long))

    def _features(self, features):
        require(features.ndim == 2 and features.shape[1] == self.n_features
            and features.is_floating_point() and torch.isfinite(features).all().item(), 'Finite [users,7] features required')
        require(features.device == self.feature_mean.device, 'Gate and features must use the same device')
        return features.detach()

    @torch.no_grad()
    def fit_feature_statistics(self, features):
        require(not self.feature_statistics_fitted.item(), 'Feature statistics are already fitted and remain fixed')
        values = self._features(features)
        require(len(values) > 0, 'Training features required to fit statistics')
        self.feature_mean.copy_(values.mean(0))
        self.feature_std.copy_(values.std(0, unbiased=False).clamp_min(STD_FLOOR))
        self.feature_statistics_fitted.fill_(True)
        return self

    def _counts(self, item_counts):
        require(item_counts is not None, 'Training item counts required')
        counts = torch.as_tensor(item_counts, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        require(counts.ndim == 1 and len(counts) > 2 and torch.isfinite(counts).all().item()
            and (counts >= 0).all().item() and counts[0].item() == 0.,
            'Nonnegative finite catalog counts with zero PAD count required')
        return counts.detach()

    @torch.no_grad()
    def fit_item_statistics(self, item_counts):
        require(not self.item_statistics_fitted.item(), 'Item statistics are already fitted and remain fixed')
        values = self._counts(item_counts)[1:].log1p()
        self.item_logcount_mean.copy_(values.mean())
        self.item_logcount_std.copy_(values.std(unbiased=False).clamp_min(STD_FLOOR))
        self.item_catalog_size.fill_(len(values))
        self.item_statistics_fitted.fill_(True)
        return self

    def _standardized(self, features):
        values = self._features(features)
        require(self.feature_statistics_fitted.item(), 'Fit feature statistics using inner training records first')
        return (values - self.feature_mean) / self.feature_std

    def coefficients(self, features):
        """Return s_user [users]; alignment weights can be 2*sigmoid(-s_user)."""
        values = self._features(features)
        if self.mode == 'global':
            return torch.zeros(len(values), device=values.device, dtype=values.dtype)
        values = self._standardized(values)
        return self.user_linear(values[:, :1] if self.mode == 'length' else values).squeeze(-1)

    def alignment_weights(self, features):
        return 2. * torch.sigmoid(-self.coefficients(features))

    def forward(self, sem_scores, cf_scores, features, item_counts=None, alpha=.1):
        require(sem_scores.ndim == 2 and cf_scores.shape == sem_scores.shape
            and sem_scores.shape[0] == len(features) and sem_scores.device == cf_scores.device == features.device
            and sem_scores.dtype == cf_scores.dtype == features.dtype
            and torch.isfinite(sem_scores).all().item() and torch.isfinite(cf_scores).all().item(),
            'Gate score/feature shape, dtype or device differs')
        require(isinstance(alpha, Real) and not isinstance(alpha, bool) and math.isfinite(alpha) and alpha >= 0,
                'Finite nonnegative residual multiplier required')
        user = self.coefficients(features)
        logits = user[:, None]
        if self.mode == 'candidate':
            require(self.item_statistics_fitted.item(), 'Fit item statistics using inner train counts first')
            counts = self._counts(item_counts)
            require(len(counts) == sem_scores.shape[1] == self.item_catalog_size.item() + 1,
                    'Inference training-count catalog mapping differs')
            # Counts may be recomputed on the full outer training fold. Their
            # transformation always uses the inner-fit fixed mean/std.
            pop = (counts.log1p() - self.item_logcount_mean) / self.item_logcount_std
            pop = pop.clone()
            pop[0] = 0.
            slope = self.pop_slope(self._standardized(features))
            logits = logits + slope * pop[None] + self.conflict_coef * (cf_scores - sem_scores).clamp(-5., 5.)
        return sem_scores + float(alpha) * (2. * torch.sigmoid(logits)) * cf_scores
