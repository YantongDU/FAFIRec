"""Identity-preserving alignment and memory-bounded catalog supervision.

These are experimental objectives, not reproductions of another project's
implementation. Catalog CE keeps the existing train-positive-excluding BPR
negative pool: only the current positive is restored, and PAD is never a class.
"""
import math
from numbers import Real

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


def _masked_ce_sum(queries, items, targets, owners, seen_mask, scale):
    scores = scale * (queries @ items.T)
    blocked = seen_mask[owners].clone()
    blocked[:, 0] = True
    blocked.scatter_(1, targets.unsqueeze(1), False)
    scores = scores.masked_fill(blocked, float('-inf'))
    return F.cross_entropy(scores, targets, reduction='sum')


def masked_catalog_ce(queries, items, targets, owners, seen_mask, score_scale=1., chunk_size=1024):
    """Mean CE over valid query rows with exact full-catalog negatives.

    ``owners`` maps each flattened supervised query back to its user's row in
    ``seen_mask``. Chunk activations are recomputed during backward, so logits
    need not be retained for every sequence position simultaneously. No RNG is
    consumed in this checkpointed function. All model/user weighting is retained
    outside this objective; this function averages over positions as before.
    """
    score_scale = _positive_number(score_scale, 'score_scale')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if queries.ndim != 2 or items.ndim != 2 or queries.shape[1] != items.shape[1]:
        raise ValueError('queries and items need matching rank-two feature matrices')
    if not queries.is_floating_point() or queries.dtype != items.dtype or queries.device != items.device:
        raise ValueError('query/item dtype and device must match')
    count, catalog = len(queries), len(items)
    if catalog < 2 or targets.shape != (count,) or owners.shape != (count,):
        raise ValueError('invalid catalog, target or owner shape')
    if targets.dtype not in (torch.int32, torch.int64) or owners.dtype not in (torch.int32, torch.int64):
        raise ValueError('targets and owners must contain integer indices')
    if seen_mask.ndim != 2 or seen_mask.shape[1] != catalog or seen_mask.dtype != torch.bool:
        raise ValueError('seen_mask must be a boolean [users,catalog] matrix')
    if any(value.device != queries.device for value in (targets, owners, seen_mask)):
        raise ValueError('all CE inputs must use the same device')
    if bool(((targets < 1) | (targets >= catalog)).any()):
        raise ValueError('targets must be nonpadding catalog IDs')
    if bool(((owners < 0) | (owners >= len(seen_mask))).any()):
        raise ValueError('owner indices exceed the seen-mask population')
    if count == 0:
        return queries.sum() * 0. + items.sum() * 0.
    targets, owners = targets.long(), owners.long()
    total = queries.new_zeros(())
    needs_grad = torch.is_grad_enabled() and (queries.requires_grad or items.requires_grad)
    for start in range(0, count, chunk_size):
        args = (queries[start:start + chunk_size], items, targets[start:start + chunk_size],
                owners[start:start + chunk_size], seen_mask, score_scale)
        loss = checkpoint(_masked_ce_sum, *args, use_reentrant=False, preserve_rng_state=False) if needs_grad else _masked_ce_sum(*args)
        total = total + loss
    return total / count


def identity_infonce(text_items, cf_items, temperature=.1):
    """Match each distinct text item to its same-ID frozen collaborative target.

    The caller deduplicates actual item IDs before constructing these matrices.
    This objective does not attempt to match anonymous marginal distributions.
    """
    temperature = _positive_number(temperature, 'temperature')
    if text_items.ndim != 2 or text_items.shape != cf_items.shape:
        raise ValueError('paired text/CF matrices must share [items,width] shape')
    if text_items.device != cf_items.device or text_items.dtype != cf_items.dtype:
        raise ValueError('paired matrices must share device and dtype')
    if not len(text_items):
        return text_items.sum() * 0.
    query = F.normalize(text_items, dim=-1)
    target = F.normalize(cf_items.detach(), dim=-1)
    logits = query @ target.T / temperature
    return F.cross_entropy(logits, torch.arange(len(query), device=query.device))
