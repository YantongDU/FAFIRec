"""Train-only linear relations with an optional semantic prediction target.

All relation matrices omit PAD. Their row/column identities are the caller's
catalog order; this module never reads user IDs, splits or labels. The zero
alignment case is EASE (Steck, WWW 2019, arXiv:1905.03375). The additional
weighted prediction-target term is a separately testable experimental change.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import torch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, positive=False):
    _require(not isinstance(value, bool) and isinstance(value, Real)
             and math.isfinite(value) and (value > 0 if positive else value >= 0),
             name + (' must be finite and positive' if positive else ' must be finite and nonnegative'))
    return float(value)


def _cpu_matrix(value, name, double=True):
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
             and value.ndim == 2 and all(size > 0 for size in value.shape)
             and value.is_floating_point() and torch.isfinite(value).all().item(),
             name + ' must be a finite floating CPU matrix')
    if double:
        _require(value.dtype == torch.float64, name + ' must have dtype float64')
    return value.detach()


def _hash(value):
    value = value.detach().cpu().contiguous()
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def fit_relation(X, ridge, semantic_prior=None, alignment_weight=0., user_weights=None):
    """Solve the unique diagonal-zero quadratic optimum in CPU float64.

    Minimize ||X-XB||_F^2 + ridge*||B||_F^2
      + mu*||D^(1/2)(XB-XS)||_F^2, subject to diag(B)=0.

    X is a binary [users, items] matrix without PAD. S is [items, items] in
    exactly the same input order. Any finite S is mathematically admissible;
    ``semantic_relation`` supplies a nonnegative, self-loop-free S. Weights are
    nonnegative and are not implicitly normalized. At mu=0, supplied S and
    weights are validated but have no effect on the fitted operator.
    """
    X = _cpu_matrix(X, 'X')
    _require(X.shape[1] >= 2 and ((X == 0) | (X == 1)).all().item(),
             'X must be binary and contain at least two catalog columns, with PAD omitted')
    ridge = _number(ridge, 'ridge', True)
    mu = _number(alignment_weight, 'alignment_weight')
    users, items = X.shape
    prior = None
    if semantic_prior is not None:
        prior = _cpu_matrix(semantic_prior, 'semantic_prior')
        _require(prior.shape == (items, items), 'semantic_prior must match the no-PAD catalog shape')
    _require(mu == 0. or prior is not None, 'Positive alignment_weight requires semantic_prior')
    weights = None
    if user_weights is not None:
        _require(isinstance(user_weights, torch.Tensor) and user_weights.device.type == 'cpu'
                 and user_weights.dtype == torch.float64 and user_weights.shape == (users,)
                 and torch.isfinite(user_weights).all().item() and (user_weights >= 0).all().item(),
                 'user_weights must be a finite nonnegative CPU float64 vector with one entry per row')
        weights = user_weights.detach()

    gram = X.T @ X
    A = gram.clone()
    A.diagonal().add_(ridge)
    C = gram
    target_constant = 0.
    if mu > 0.:
        weighted_gram = gram if weights is None else X.T @ (weights[:, None] * X)
        target_cross = weighted_gram @ prior
        A.add_(weighted_gram, alpha=mu)
        C = gram + mu * target_cross
        target_constant = mu * float((prior * target_cross).sum())
    precision = torch.linalg.solve(A, torch.eye(items, dtype=torch.float64))
    denominator = precision.diagonal()
    _require(torch.isfinite(precision).all().item() and (denominator > 0).all().item(),
             'Regularized normal matrix did not yield a finite positive precision diagonal')
    if mu == 0.:
        # Algebraically identical to P@G followed by diagonal correction. This
        # route avoids cancellation in P@G = I-ridge*P and exactly retains EASE.
        B = -precision / denominator[None, :]
        route = 'EASE special case: -P / diag(P) by column; diagonal zero'
    else:
        unconstrained = precision @ C
        B = unconstrained - precision * (unconstrained.diagonal() / denominator)[None, :]
        route = 'Z=P@C; B=Z-P*(diag(Z)/diag(P)) by column; diagonal zero'
    B.fill_diagonal_(0.)
    _require(torch.isfinite(B).all().item(), 'Fitted relation contains nonfinite values')

    # Report the actual constrained normal-equation residual. Computing the
    # quadratic objective this way avoids a second users-by-items prediction
    # matrix and does not assume the solver residual is zero.
    normal_residual = A @ B - C
    linear = float((B * C).sum())
    quadratic = float((B * (normal_residual + C)).sum())
    objective = float(gram.diagonal().sum()) + target_constant - 2. * linear + quadratic
    diagonal_multiplier = normal_residual.diagonal().clone()
    normal_residual.fill_diagonal_(0.)
    offdiag_norm = float(torch.linalg.vector_norm(normal_residual))
    report = {
        'schema': 'diagnostic_v4_relation_closed_form_v1', 'users': users, 'items': items,
        'matrix_convention': 'No PAD; input row order for users and catalog column order for items',
        'objective': '||X-XB||^2 + ridge||B||^2 + mu||D^(1/2)(XB-XS)||^2; diag(B)=0',
        'ridge': ridge, 'alignment_weight': mu, 'closed_form_route': route,
        'semantic_prior_provided': prior is not None, 'semantic_prior_used': mu > 0.,
        'user_weights_provided': weights is not None, 'user_weights_used': mu > 0. and weights is not None,
        'user_weights_are_normalized_internally': False,
        'objective_value': objective, 'ridge_penalty_value': ridge * float(B.square().sum()),
        'residual': {'kkt_offdiagonal_max_abs': float(normal_residual.abs().max()),
            'kkt_offdiagonal_frobenius': offdiag_norm,
            'kkt_relative_frobenius': offdiag_norm / max(float(torch.linalg.vector_norm(C)), 1.),
            'diagonal_constraint_max_abs': float(B.diagonal().abs().max()),
            'lagrange_multiplier_max_abs': 2. * float(diagonal_multiplier.abs().max())},
        'sha256': {'X': _hash(X), 'semantic_prior': _hash(prior) if prior is not None else None,
            'user_weights': _hash(weights) if weights is not None else None,
            'normal_matrix_A': _hash(A), 'cross_matrix_C': _hash(C),
            'precision_P': _hash(precision), 'relation_B': _hash(B)},
    }
    return B.contiguous(), report


@torch.no_grad()
def semantic_relation(features, train_seen_mask, topk=30):
    """Build a static positive-cosine graph, preserving input catalog order.

    ``features`` has [items, width] rows without PAD. Only train-seen items
    define the center. Non-PAD raw embedding rows must be nonzero; centered
    zero rows are legitimate and produce zero outgoing mass. Stable sorting
    resolves cosine ties by smaller input column index. Nonzero rows sum to 1.
    No interaction targets, user IDs or split data are accepted by this API.
    """
    raw = _cpu_matrix(features, 'features', double=False)
    _require(raw.shape[0] >= 2, 'At least two no-PAD item feature rows required')
    _require(raw.abs().sum(1).gt(0).all().item(),
             'Raw zero item feature rows are invalid; strip PAD before calling semantic_relation')
    _require(isinstance(train_seen_mask, torch.Tensor) and train_seen_mask.device.type == 'cpu'
             and train_seen_mask.dtype == torch.bool and train_seen_mask.shape == (len(raw),)
             and train_seen_mask.any().item(),
             'train_seen_mask must be a CPU bool vector with at least one seen item')
    _require(type(topk) is int and topk > 0, 'topk must be a positive integer')
    values = raw.to(torch.float64)
    center = values[train_seen_mask].mean(0)
    centered = values - center
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    _require(torch.isfinite(center).all().item() and torch.isfinite(norms).all().item(),
             'Centering or row norm overflowed; item features must have a finite float64 scale')
    normalized = centered / norms.clamp_min(torch.finfo(torch.float64).tiny)
    similarities = (normalized @ normalized.T).clamp(min=0., max=1.)
    similarities.fill_diagonal_(0.)
    effective = min(topk, len(values) - 1)
    indices = torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :effective]
    relation = torch.zeros_like(similarities)
    relation.scatter_(1, indices, similarities.gather(1, indices))
    totals = relation.sum(1, keepdim=True)
    relation /= totals.clamp_min(torch.finfo(torch.float64).tiny)
    nonzero = totals.squeeze(1).gt(0)
    row_error = float((relation.sum(1)[nonzero] - 1.).abs().max()) if nonzero.any() else 0.
    report = {'schema': 'diagnostic_v4_static_semantic_relation_v1',
        'items': len(values), 'feature_width': values.shape[1],
        'row_and_column_identity': 'Input catalog row order; PAD omitted',
        'center': 'Unweighted mean of train-seen static features only',
        'train_seen_items': int(train_seen_mask.sum()), 'topk_requested': topk, 'topk_effective': effective,
        'similarity': 'Train-seen-centered row-L2 positive cosine; no self loops',
        'tie_break': 'Stable descending cosine; smaller input column index first',
        'row_normalization': 'Nonzero rows sum to one; zero-mass rows stay zero',
        'zero_centered_rows': int(norms.squeeze(1).eq(0).sum()),
        'zero_mass_rows': int((~nonzero).sum()), 'nonzero_edges': int(relation.gt(0).sum()),
        'row_sum_max_error': row_error, 'self_loop_max_abs': float(relation.diagonal().abs().max()),
        'static_features_only': True,
        'sha256': {'features_as_float64': _hash(values), 'train_seen_mask': _hash(train_seen_mask),
            'center': _hash(center), 'semantic_relation_S': _hash(relation)},
    }
    return relation.contiguous(), report
