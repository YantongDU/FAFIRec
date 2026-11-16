"""Exact catalog ranking with explicit candidate policies and deterministic ties.

``exclude_seen_restore_target`` reproduces a common offline convention, but is
target-conditioned: it restores the held-out identity after history filtering.
``full`` is the default and does not consult the target to construct candidates.
All reported recall/hit metrics assume one relevant held-out item per user.
"""
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


CANDIDATE_POLICIES = ("full", "exclude_seen", "exclude_seen_restore_target")


def _integer_vector(value, name, *, allow_empty=False):
    array = np.asarray(value)
    if array.ndim != 1 or (not allow_empty and len(array) == 0):
        raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
    if array.size and (not np.issubdtype(array.dtype, np.integer) or array.dtype == bool):
        raise ValueError(f"{name} must contain integer IDs/ranks")
    return array.astype(np.int64, copy=False)


def _history_exclusions(targets, histories, policy, n_items=None):
    if policy not in CANDIDATE_POLICIES:
        raise ValueError(f"Unknown candidate_policy {policy!r}")
    if histories is not None and len(histories) != len(targets):
        raise ValueError("One observed history is required per target")
    if policy == "full":
        return [set() for _ in targets]
    if histories is None:
        raise ValueError("History filtering requires observed histories")
    excluded = []
    for history, target in zip(histories, targets):
        ids = _integer_vector(history, "history", allow_empty=True)
        if np.any(ids < 0) or (n_items is not None and np.any(ids > n_items)):
            raise ValueError("Observed history contains an out-of-catalog item")
        seen = set(ids[ids > 0].tolist())
        if policy == "exclude_seen" and int(target) in seen:
            raise ValueError("A repeated target is excluded by exclude_seen; choose a declared repeat policy")
        if policy == "exclude_seen_restore_target":
            seen.discard(int(target))
        excluded.append(seen)
    return excluded


def _valid_targets(targets, n_items=None):
    targets = _integer_vector(targets, "targets")
    if np.any(targets <= 0) or (n_items is not None and np.any(targets > n_items)):
        raise ValueError("Targets must be nonpadding catalog item IDs")
    return targets


def target_ranks(scores, targets, histories=None, candidate_policy="full"):
    """Return one-based exact ranks; ties prefer smaller integer item IDs.

    The score matrix includes padding column zero. Negative infinity is allowed
    for excluded candidates, but every target must have a finite score. This
    function copies scores and never mutates the caller's array.
    """
    values = np.array(scores, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must have shape [users, n_items + 1]")
    targets = _valid_targets(targets, values.shape[1] - 1)
    if len(targets) != len(values):
        raise ValueError("One scalar target ID is required per score row")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("NaN or positive infinite scores invalidate ranking")
    excluded = _history_exclusions(targets, histories, candidate_policy, values.shape[1] - 1)
    values[:, 0] = -np.inf
    for row, blocked in enumerate(excluded):
        if blocked:
            values[row, list(blocked)] = -np.inf
    true_scores = values[np.arange(len(values)), targets]
    if not np.isfinite(true_scores).all():
        raise ValueError("Ground-truth target is not rankable")
    candidates = np.arange(values.shape[1])[None, :]
    not_target = candidates != targets[:, None]
    better = (values > true_scores[:, None]) & not_target
    tied_before = (values == true_scores[:, None]) & (candidates < targets[:, None])
    return 1 + better.sum(axis=1) + tied_before.sum(axis=1)


def exact_target_ranks_from_chunks(
    chunks: Iterable[Tuple[np.ndarray, np.ndarray]],
    target_scores,
    targets,
    histories=None,
    candidate_policy="full",
    n_items: Optional[int] = None,
):
    """Accumulate exact ranks without storing a [users, catalog] score matrix.

    Each chunk is ``(item_ids[C], scores[B,C])``; chunks must cover each item
    1..n_items exactly once (padding zero is optional). If n_items is omitted,
    the largest received ID defines the catalog, which must still be complete.
    The caller should obtain target_scores from an identical chunked scoring
    pass. The target's own chunk entry is excluded from its rank comparison to
    avoid a spurious extra rank caused by dot-product rounding across kernels.
    """
    if n_items is not None and (not isinstance(n_items, (int, np.integer)) or n_items < 1):
        raise ValueError("n_items must be positive")
    targets = _valid_targets(targets, n_items)
    true_scores = np.asarray(target_scores, dtype=np.float64)
    if true_scores.shape != targets.shape or not np.isfinite(true_scores).all():
        raise ValueError("One finite target score is required per user")
    exclusions = _history_exclusions(targets, histories, candidate_policy, n_items)
    counts = np.zeros(len(targets), dtype=np.int64)
    seen_ids = set()
    target_found = np.zeros(len(targets), dtype=bool)
    for ids, raw_scores in chunks:
        ids = _integer_vector(ids, "chunk item IDs")
        if np.any(ids < 0) or (n_items is not None and np.any(ids > n_items)):
            raise ValueError("Chunk contains an out-of-catalog item")
        if len(set(ids.tolist())) != len(ids) or seen_ids.intersection(ids.tolist()):
            raise ValueError("An item occurs in more than one chunk position")
        seen_ids.update(ids.tolist())
        values = np.asarray(raw_scores, dtype=np.float64)
        if values.shape != (len(targets), len(ids)):
            raise ValueError("Chunk scores must have shape [users, chunk_items]")
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("NaN or positive infinite scores invalidate ranking")
        mask = np.broadcast_to(ids[None, :] > 0, values.shape).copy()
        for row, blocked in enumerate(exclusions):
            if blocked:
                mask[row] &= ~np.isin(ids, list(blocked))
        is_target = ids[None, :] == targets[:, None]
        found_rows, found_cols = np.where(is_target)
        if len(found_rows):
            if not np.isfinite(values[found_rows, found_cols]).all():
                raise ValueError("Ground-truth target is not rankable")
            if not np.array_equal(values[found_rows, found_cols], true_scores[found_rows]):
                raise ValueError("Target scores differ between scoring passes; use identical chunked scoring")
            target_found[found_rows] = True
        mask &= ~is_target
        better = values > true_scores[:, None]
        tied = (values == true_scores[:, None]) & (ids[None, :] < targets[:, None])
        counts += ((better | tied) & mask).sum(axis=1)
    catalog_size = n_items if n_items is not None else max(seen_ids, default=0)
    if catalog_size < 1 or len(seen_ids - {0}) != catalog_size:
        raise ValueError("Chunks do not cover the full contiguous catalog")
    if not target_found.all():
        raise ValueError("A target is absent from the scored catalog")
    if n_items is None and any(any(i > catalog_size for i in h) for h in exclusions):
        raise ValueError("Observed history contains an out-of-catalog item")
    return counts + 1


def metrics_from_ranks(ranks, k=10):
    """Return (Hit/Recall@k, NDCG@k) arrays for one held-out item per user."""
    ranks = _integer_vector(ranks, "ranks", allow_empty=True)
    if np.any(ranks < 1) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Ranks and k must be positive integers")
    hits = (ranks <= k).astype(np.float64)
    return hits, hits / np.log2(ranks + 1.0)


def summarize_rows(rows, ks: Sequence[int] = (5, 10, 20)):
    """Aggregate per-user ranks; no averaging of subgroup averages."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation population")
    result = {}
    for group in ("all", "cold", "middle", "warm"):
        selected = rows if group == "all" else [r for r in rows if r.get("group") == group]
        if not selected:
            continue
        ranks = _integer_vector([r["rank"] for r in selected], "ranks")
        result[group] = {"users": len(selected)}
        for k in ks:
            hits, ndcgs = metrics_from_ranks(ranks, k)
            result[group].update({f"hit{k}": float(hits.mean()), f"ndcg{k}": float(ndcgs.mean()),
                                  f"mrr{k}": float(np.mean(hits / ranks))})
    return result
