"""Fixed training-only query diagnostics from a frozen collaborative teacher.

Loss and query-gradient norm are in-sample descriptors, not calibrated
correctness probabilities. The gradient is measured with respect to an
independent detached normalized query; it does not include the teacher encoder
or the derivative of query normalization. The diagnostic score has unit scale,
even though recommendation logits use sqrt(d).
"""

from numbers import Integral
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import FAFIRecModel


def query_diagnostics(query: Tensor, positive: Tensor, negative: Tensor) -> Tuple[Tensor, Tensor]:
    """Return detached per-query loss and exact query-gradient norm.

    Item vectors and queries are assumed to have already been normalized by the
    teacher. This function never changes or backpropagates into those inputs.
    """
    if query.ndim != 2 or positive.shape != query.shape or negative.shape != query.shape:
        raise ValueError("query, positive and negative need matching [batch, width] shapes")
    if not all(bool(torch.isfinite(x).all()) for x in (query, positive, negative)):
        raise ValueError("diagnostic inputs must be finite")
    # Explicitly enable this local derivative if the caller disabled gradients.
    # Cloning outside inference mode produces an independent differentiable leaf.
    with torch.inference_mode(False), torch.enable_grad():
        leaf = query.detach().clone().requires_grad_(True)
        contrast = (positive.detach() - negative.detach()).clone()
        losses = F.softplus(-(leaf * contrast).sum(dim=-1))
        gradient = torch.autograd.grad(losses.sum(), leaf, create_graph=False)[0]
    return losses.detach(), gradient.detach().norm(dim=-1)


def _unseen_negative(seen: set, n_items: int, generator: torch.Generator) -> int:
    """Uniform sample from the complement, with bounded work for dense users."""
    if len(seen) >= n_items:
        raise ValueError("a diagnostic user must have at least one unobserved negative")
    if len(seen) * 2 > n_items:
        candidates = [i for i in range(1, n_items + 1) if i not in seen]
        return candidates[int(torch.randint(len(candidates), (), generator=generator))]
    while True:
        item = int(torch.randint(1, n_items + 1, (), generator=generator))
        if item not in seen:
            return item


def compute_diagnostics(
    teacher: FAFIRecModel,
    records: Sequence[dict],
    max_len: int,
    batch_size: int,
    seed: int,
    device: Optional[str] = None,
) -> Tuple[Tensor, dict]:
    """Cache one diagnostic for every eligible training user in record order.

    Only ``record['user']`` and ``record['train']`` are read. Validation/test
    labels never determine the transition, negative sample, or normalization.
    All records must contain a training prefix, its subsequent training item,
    and at least one training-unobserved catalog item. Invalid users fail
    explicitly instead of silently changing the evaluation population.

    Returns a CPU [users, 2] tensor and a JSON-serializable provenance manifest.
    The feature order is (log query-gradient norm, log teacher loss). Teacher
    evaluation mode is retained after this call; its parameters and gradients
    are otherwise unmodified.
    """
    if teacher.control != "cf":
        raise ValueError("diagnostics require a collaborative-only teacher")
    if not records:
        raise ValueError("at least one eligible training user is required")
    if not isinstance(max_len, Integral) or not 1 <= max_len <= teacher.max_len:
        raise ValueError("max_len must be a positive integer at most teacher.max_len")
    if not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    actual_device = teacher.item_id_table.weight.device
    if device is not None:
        requested = torch.device(device)
        if requested.type != actual_device.type or (
            requested.index is not None and requested.index != actual_device.index
        ):
            raise ValueError("teacher must already be on the requested diagnostic device")

    user_order = [str(record['user']) for record in records]
    if len(set(user_order)) != len(user_order):
        raise ValueError("diagnostic user IDs must be unique")
    training = []
    for record in records:
        history = list(record['train'])
        if len(history) < 2:
            raise ValueError("diagnostic users require at least two training interactions")
        if any(not isinstance(i, Integral) or isinstance(i, bool) or not 1 <= i <= teacher.n_items
               for i in history):
            raise ValueError("training histories require nonpadding catalog item IDs")
        if len(set(history)) >= teacher.n_items:
            raise ValueError("a diagnostic user must have at least one unobserved negative")
        training.append(history)

    teacher.eval()
    sampling_seed = int(seed) + 101
    generator = torch.Generator(device='cpu').manual_seed(sampling_seed)
    negatives = [_unseen_negative(set(h), teacher.n_items, generator) for h in training]
    all_loss, all_gradient = [], []
    for start in range(0, len(training), int(batch_size)):
        batch = training[start:start + int(batch_size)]
        prefixes = [h[:-1][-int(max_len):] for h in batch]
        histories = torch.tensor(
            [[0] * (int(max_len) - len(h)) + h for h in prefixes],
            dtype=torch.long, device=actual_device,
        )
        positives = torch.tensor([h[-1] for h in batch], dtype=torch.long, device=actual_device)
        sampled = torch.tensor(negatives[start:start + len(batch)], dtype=torch.long, device=actual_device)
        with torch.no_grad():
            query = teacher.encode_cf(histories).detach()
            positive = F.normalize(teacher.item_id_table(positives), dim=-1).detach()
            negative = F.normalize(teacher.item_id_table(sampled), dim=-1).detach()
        loss, gradient = query_diagnostics(query, positive, negative)
        all_loss.append(loss.cpu())
        all_gradient.append(gradient.cpu())

    losses, gradients = torch.cat(all_loss), torch.cat(all_gradient)
    raw = torch.stack([torch.log(gradients + 1e-8), torch.log(losses + 1e-8)], dim=-1)
    mean = raw.mean(dim=0)
    population_std = raw.std(dim=0, unbiased=False)
    std = population_std.clamp_min(1e-6)
    standardized = (raw - mean) / std
    if not bool(torch.isfinite(standardized).all()):
        raise RuntimeError("non-finite teacher diagnostic statistics")
    manifest = {
        'feature_names': ['log_query_gradient_norm', 'log_teacher_loss'],
        'user_order': user_order,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'population_std': population_std.tolist(),
        'log_epsilon': 1e-8,
        'std_floor': 1e-6,
        'loss': losses.tolist(),
        'query_gradient_norm': gradients.tolist(),
        'raw_log_features': raw.tolist(),
        'sampled_negatives': negatives,
        'negative_sampling_seed': sampling_seed,
        'negative_exclusion': 'all training items only',
        'scope': 'all eligible training users; each last training transition',
        'teacher_mode': 'evaluation; dropout disabled',
        'teacher_score_scale': 1.0,
        'query_derivative': 'independent detached normalized teacher query',
        'interpretation': 'in-sample fit and contrast-pair geometry; not held-out reliability',
        'refresh': 'cached once per teacher/run; reused without batch restandardization',
    }
    return standardized.detach().cpu(), manifest
