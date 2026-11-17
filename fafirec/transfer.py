"""Training-only initialization of the unchanged FAFIRec fusion equations.

Ridge projection initialization uses static text and the selected CF teacher's
training-seen item embeddings. It does not consume held-out interactions or
turn the projection into a separate encoder. CF-favoring gate priors are an
explicit initialization choice; the gates remain trainable.
"""
import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F

from .model import initialize_from_teacher


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


@torch.no_grad()
def initialize_transfer(student, teacher, records, projection_init='random', cf_prior=.5, ridge=1.):
    """Copy the CF state, optionally fit its text map, and set matched gate priors.

    The ridge objective is ||X W.T - E||_F^2 + ridge * ||W||_F^2 over
    unique TRAINING-SEEN catalog items, with no intercept. The solve is CPU
    float64; the resulting matrix is copied into the trainable projection.
    Unseen teacher rows are excluded because they have no learned CF signal.
    ``records`` may omit validation/test fields; only ``train`` is read.
    """
    if projection_init not in {'random', 'ridge'}:
        raise ValueError('projection_init must be random or ridge')
    cf_prior = _finite_real(cf_prior, 'cf_prior')
    ridge = _finite_real(ridge, 'ridge')
    if not 0 < cf_prior < 1:
        raise ValueError('cf_prior must lie strictly between zero and one')
    if ridge <= 0:
        raise ValueError('ridge must be positive')
    if student.control == 'cf' and projection_init != 'random':
        raise ValueError('CF control has no text projection initialization')
    if not records:
        raise ValueError('Training records must not be empty')
    seen = set()
    events = 0
    for record in records:
        sequence = record['train']
        if len(sequence) < 2 or any(isinstance(item, bool) or not isinstance(item, Integral)
                                    or not 1 <= item <= student.n_items for item in sequence):
            raise ValueError('Invalid training items for transfer initialization')
        events += len(sequence)
        seen.update(int(item) for item in sequence)
    initialize_from_teacher(student, teacher)
    report = {'projection_init': projection_init, 'cf_prior': cf_prior, 'ridge': ridge,
              'training_seen_items': len(seen), 'training_events': events,
              'scope': 'unique training-seen items; static text and selected CF teacher only',
              'heldout_interactions_used': False, 'controller_initialization': 'zero coefficients and intercept'}
    if student.control == 'cf':
        return report
    logit = math.log(cf_prior / (1. - cf_prior))
    student.gate_bias.fill_(logit)
    student.item_gate_logit.fill_(logit)
    student.control_bias.zero_()
    student.control_weights.zero_()
    if projection_init == 'ridge':
        ids = torch.tensor(sorted(seen), dtype=torch.long)
        x = student.text_features.detach().cpu()[ids].to(torch.float64)
        target = teacher.item_id_table.weight.detach().cpu()[ids].to(torch.float64)
        if not bool(torch.isfinite(x).all() and torch.isfinite(target).all()):
            raise ValueError('Nonfinite training inputs in ridge initialization')
        gram = x.T @ x
        gram.diagonal().add_(ridge)
        coefficient = torch.linalg.solve(gram, x.T @ target).T
        if not bool(torch.isfinite(coefficient).all()):
            raise RuntimeError('Nonfinite ridge projection')
        student.proj_content.weight.copy_(coefficient.to(student.proj_content.weight))
        predicted = x @ coefficient.T
        report.update({'ridge_objective': 'sum squared error + ridge * squared weight norm; no intercept',
            'solve_device_dtype': 'cpu float64',
            'training_relative_squared_error': float((predicted - target).square().sum() / target.square().sum().clamp_min(1e-12)),
            'training_mean_cosine': float((F.normalize(predicted, dim=-1) * F.normalize(target, dim=-1)).sum(-1).mean()),
            'projected_mean_norm': float(predicted.norm(dim=-1).mean()),
            'teacher_mean_norm': float(target.norm(dim=-1).mean())})
    return report


def optimizer_groups(model, lr, gate_lr_multiplier=1.):
    """Allow low-dimensional controllers/gates to adapt without speeding embeddings.

    A multiplier of one returns the historical single parameter group exactly.
    This is an explicit optimizer setting, selected using validation only.
    """
    multiplier = _finite_real(gate_lr_multiplier, 'gate_lr_multiplier')
    lr = _finite_real(lr, 'lr')
    if multiplier <= 0 or lr <= 0:
        raise ValueError('Learning rate and gate_lr_multiplier must be positive')
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if multiplier == 1.:
        return [{'params': [p for _, p in trainable], 'lr': lr}]
    gate_names = {'gate_bias', 'item_gate_logit', 'control_bias', 'control_weights', 'conflict_scale_raw'}
    groups = []
    for is_gate in (False, True):
        parameters = [p for name, p in trainable if (name in gate_names) == is_gate]
        if parameters:
            groups.append({'params': parameters, 'lr': lr * (multiplier if is_gate else 1.)})
    return groups
