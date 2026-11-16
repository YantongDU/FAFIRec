"""Post-normalization ranking checks against the same checkpoint's CF branch.

These are sufficient, observable preservation certificates, not constraints
enforced by training or guarantees of better relevance. The caller must supply
the same fixed candidate set for both rankings. A subset certifies that subset.
"""
import math

import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _vectors(c_user, z_user, c_items, z_items):
    c, z, ci, zi = map(_array, (c_user, z_user, c_items, z_items))
    if c.ndim != 1 or z.shape != c.shape or ci.ndim != 2 or zi.shape != ci.shape:
        raise ValueError("Need user vectors [d] and matching item tables [N+1,d]")
    if ci.shape[1] != len(c) or len(ci) < 3:
        raise ValueError("Representation dimensions or candidate count are invalid")
    if not all(np.isfinite(v).all() for v in (c, z, ci, zi)):
        raise ValueError("Representations must be finite")
    return c, z, ci, zi


def _topk(scores, ids, k):
    """O(N) partition plus O(K log K) ordering, including deterministic ties."""
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - len(higher)
    if need < len(tied):
        chosen_ties = tied[np.argpartition(ids[tied], need - 1)[:need]]
    else:
        chosen_ties = tied
    chosen = np.concatenate((higher, chosen_ties))
    return chosen[np.lexsort((ids[chosen], -scores[chosen]))]


def pair_certificate(c_user, z_user, c_i, c_j, z_i, z_j, tolerance=1e-10):
    """Bound change in one reference margin, with a conservative numeric guard."""
    values = list(map(_array, (c_user, z_user, c_i, c_j, z_i, z_j)))
    if values[0].ndim != 1 or not all(v.shape == values[0].shape for v in values):
        raise ValueError("All representations must have the same vector shape")
    if not all(np.isfinite(v).all() for v in values) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Representations and tolerance must be finite/nonnegative")
    c, z, ci, cj, zi, zj = values
    du, di, dj = z - c, zi - ci, zj - cj
    margin = float(c @ (ci - cj))
    fused_margin = float(z @ (zi - zj))
    bound = float(np.linalg.norm(du) * np.linalg.norm(ci - cj)
                  + np.linalg.norm(z) * np.linalg.norm(di - dj))
    guard = tolerance * max(1., abs(margin), bound)
    return {'reference_margin': margin, 'fused_margin': fused_margin,
            'margin_change_bound': bound, 'certified': margin > bound + guard,
            'reference_positive': margin > 0., 'actual_preserved': fused_margin > 0.}


def ranking_certificate(c_user, z_user, c_items, z_items, k=10,
                        candidate_ids=None, target=None, score_scale=1., tolerance=1e-10):
    """Audit top-k set/order preservation with an O(Nd) interval scan.

    Input item arrays include padding row zero. The reference top-k is found
    by a partition; internal ordering costs O(K log K). No K*(N-K) crossing
    pair matrix is constructed. Bounds account for user AND item drift after
    normalization. An optional held-out target adds observed one-positive
    metric changes and certificates for already-correct target-vs-item pairs.
    """
    c, z, ci_all, zi_all = _vectors(c_user, z_user, c_items, z_items)
    if candidate_ids is None:
        ids = np.arange(1, len(ci_all), dtype=np.int64)
    else:
        ids = np.asarray(candidate_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("candidate_ids must be a vector of integer IDs")
        if len(set(ids.tolist())) != len(ids) or np.any(ids <= 0) or np.any(ids >= len(ci_all)):
            raise ValueError("Candidates must be distinct nonpadding catalog IDs")
        ids = ids.astype(np.int64)
    if not isinstance(k, (int, np.integer)) or not 1 <= k < len(ids):
        raise ValueError("Certificate requires 1 <= k < number of candidates")
    if not math.isfinite(score_scale) or score_scale <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("score_scale must be positive; tolerance finite and nonnegative")
    ci, zi = ci_all[ids], zi_all[ids]
    du, di = z - c, zi - ci
    du_norm, z_norm = np.linalg.norm(du), np.linalg.norm(z)
    drift_norms = np.linalg.norm(di, axis=1)
    reference = score_scale * (ci @ c)
    fused = score_scale * (zi @ z)
    epsilon = score_scale * (du_norm * np.linalg.norm(ci, axis=1) + z_norm * drift_norms)
    top = _topk(reference, ids, k)
    fused_top = _topk(fused, ids, k)
    outside = np.ones(len(ids), dtype=bool)
    outside[top] = False
    lower = float(np.min(reference[top] - epsilon[top]))
    upper = float(np.max(reference[outside] + epsilon[outside]))
    guard = tolerance * max(1., abs(lower), abs(upper))
    set_certified = bool(lower > upper + guard)
    adjacent_certificates = []
    for i, j in zip(top[:-1], top[1:]):
        bound = score_scale * (du_norm * np.linalg.norm(ci[i] - ci[j])
                               + z_norm * np.linalg.norm(di[i] - di[j]))
        margin = float(reference[i] - reference[j])
        adjacent_certificates.append(bool(margin > bound + tolerance * max(1., abs(margin), bound)))
    result = {
        'reference': 'collaborative branch at the same jointly trained checkpoint',
        'candidate_count': len(ids), 'candidate_scope': 'full_catalog' if candidate_ids is None else 'supplied_fixed_subset',
        'k': int(k), 'score_scale': float(score_scale),
        'reference_topk': ids[top].tolist(), 'fused_topk': ids[fused_top].tolist(),
        'interval_lower': lower, 'interval_upper': upper, 'interval_gap': lower - upper,
        'certified_topk_set': set_certified,
        'certified_ordered_topk': set_certified and all(adjacent_certificates),
        'adjacent_pair_certificates': adjacent_certificates,
        'actual_topk_set_preserved': set(ids[top]) == set(ids[fused_top]),
        'actual_ordered_topk_preserved': bool(np.array_equal(ids[top], ids[fused_top])),
        'user_drift_norm': float(du_norm), 'mean_item_drift_norm': float(drift_norms.mean()),
        'max_item_drift_norm': float(drift_norms.max()),
        'maximum_observed_score_change': float(np.max(np.abs(fused - reference))),
        'maximum_score_change_bound': float(epsilon.max()),
    }
    if target is not None:
        if not isinstance(target, (int, np.integer)) or int(target) not in set(ids.tolist()):
            raise ValueError("The held-out target must belong to the candidate set")
        t = int(np.flatnonzero(ids == target)[0])
        ref_rank = 1 + int(np.count_nonzero((reference > reference[t]) |
                                           ((reference == reference[t]) & (ids < target))))
        fused_rank = 1 + int(np.count_nonzero((fused > fused[t]) |
                                             ((fused == fused[t]) & (ids < target))))
        margins = reference[t] - reference
        bounds = score_scale * (du_norm * np.linalg.norm(ci[t] - ci, axis=1)
                                 + z_norm * np.linalg.norm(di[t] - di, axis=1))
        correct = (margins > 0) & (ids != target)
        certified = correct & (margins > bounds + tolerance * np.maximum(1., np.maximum(np.abs(margins), bounds)))
        reversal = correct & (fused[t] <= fused)
        ref_ndcg = 1 / np.log2(ref_rank + 1) if ref_rank <= k else 0.
        fused_ndcg = 1 / np.log2(fused_rank + 1) if fused_rank <= k else 0.
        result.update({'target': int(target), 'reference_target_rank': ref_rank,
                       'fused_target_rank': fused_rank, 'reference_ndcg': float(ref_ndcg),
                       'fused_ndcg': float(fused_ndcg), 'ndcg_change': float(fused_ndcg - ref_ndcg),
                       'correct_reference_target_pairs': int(correct.sum()),
                       'certified_correct_target_pairs': int(certified.sum()),
                       'correct_target_pair_strict_preferences_lost': int(reversal.sum())})
    return result
