"""Binary-relevance top-K metrics for held-out interaction blocks.

Recall divides by the number of distinct relevant items for each user. NDCG
normalizes by that user's ideal binary-relevance DCG, not by a single hit.
Candidate membership is declared independently of target identities: history
filtering never restores a target that it would otherwise remove.
"""
from numbers import Integral, Real

import numpy as np


def _cutoffs(ks):
    try:
        values = tuple(ks)
    except TypeError as error:
        raise ValueError("ks must be a nonempty sequence of positive integer cutoffs") from error
    if not values or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1 for k in values):
        raise ValueError("ks must be a nonempty sequence of positive integer cutoffs")
    return tuple(sorted({int(k) for k in values}))


def _ids(value, name, n_items, *, padding=False, empty=False, unique=False):
    if isinstance(value, (set, frozenset)):
        value = list(value)
    array = np.asarray(value)
    if array.ndim != 1 or (not empty and array.size == 0):
        raise ValueError(f"{name} must be a one-dimensional collection of integer item IDs")
    if array.size and (not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(f"{name} must contain integer item IDs")
    lower = 0 if padding else 1
    # Check before int64 conversion so that very large unsigned IDs cannot wrap.
    if np.any(array < lower) or np.any(array > n_items):
        raise ValueError(f"{name} contains an out-of-catalog or padding item")
    array = array.astype(np.int64, copy=False)
    if unique and np.unique(array).size != array.size:
        raise ValueError(f"{name} must contain distinct relevant item IDs")
    return array


def _prefix(scores, ids, count):
    """Partition the catalog and sort only the selected deterministic prefix."""
    amount = min(count, len(ids))
    if amount < len(ids):
        values = scores[ids]
        threshold = np.partition(values, len(values) - amount)[len(values) - amount]
        above = ids[values > threshold]
        # ids is ascending; selecting the first tied IDs implements the tie rule.
        tied = ids[values == threshold][:amount - len(above)]
        ids = np.concatenate((above, tied))
    order = np.lexsort((ids, -scores[ids]))
    return ids[order][:amount]


def evaluate_topk(scores, target_sets, histories=None, ks=(10, 50), candidate_policy="exclude_seen"):
    """Return one metric dictionary per user for a multi-positive held-out block.

    scores has shape [users, N+1], including padding column zero. Each entry in
    target_sets is a nonempty collection of distinct IDs in 1..N. A detached
    PyTorch tensor or a numeric NumPy-compatible score matrix is accepted; the
    caller's data are not changed.

    ``full`` ranks all finite nonpadding items. ``exclude_seen`` additionally
    removes the supplied observed history. A relevant item excluded by that
    history is an error, not silently restored. NaN/+infinity invalidate scores;
    negative infinity represents an already masked candidate and cannot be a
    relevant score. With fewer than K candidates, the list contains all eligible
    items. Recall/NDCG still use the requested K and the user's relevance count.

    Ties prefer the smaller integer item ID. top_items contains the ordered
    prefix through max(ks), with no padding or excluded identities.
    """
    cutoffs = _cutoffs(ks)
    if candidate_policy not in {"full", "exclude_seen"}:
        raise ValueError("candidate_policy must be full or exclude_seen")
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    raw = np.asarray(scores)
    if (raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 2
            or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)):
        raise ValueError("scores must be a real numeric [users, N+1] matrix")
    values = np.array(raw, dtype=np.float64, copy=True)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("NaN or positive infinite scores invalidate ranking")
    try:
        if len(target_sets) != len(values):
            raise ValueError("One relevant target set is required per user")
    except TypeError as error:
        raise ValueError("One relevant target set is required per user") from error
    if histories is not None:
        try:
            if len(histories) != len(values):
                raise ValueError("One observed history is required per user")
        except TypeError as error:
            raise ValueError("One observed history is required per user") from error
    elif candidate_policy == "exclude_seen":
        raise ValueError("exclude_seen requires one observed history per user")

    n_items = values.shape[1] - 1
    max_k = min(max(cutoffs), n_items)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
    ideal_prefix = np.cumsum(discounts)
    rows = []
    for index in range(len(values)):
        relevant = _ids(target_sets[index], "target set", n_items, unique=True)
        observed = (_ids(histories[index], "observed history", n_items, padding=True, empty=True)
                    if histories is not None else np.empty(0, dtype=np.int64))
        if not np.isfinite(values[index, relevant]).all():
            raise ValueError("Every relevant item must have a finite score")
        values[index, 0] = -np.inf
        if candidate_policy == "exclude_seen":
            if np.intersect1d(relevant, observed).size:
                raise ValueError("A relevant target is excluded by the observed history; no target is restored")
            values[index, observed] = -np.inf
        candidates = np.flatnonzero(np.isfinite(values[index]))
        ordered = _prefix(values[index], candidates, max_k)
        is_relevant = np.isin(ordered, relevant)
        hits = np.cumsum(is_relevant, dtype=np.int64)
        gains = np.cumsum(is_relevant * discounts[:len(ordered)])
        row = {"relevant_count": int(len(relevant)), "top_items": ordered.tolist()}
        for k in cutoffs:
            length = min(k, len(ordered))
            n_hits = int(hits[length - 1])
            dcg = float(gains[length - 1])
            idcg = float(ideal_prefix[min(k, len(relevant)) - 1])
            row.update({f"recall{k}": n_hits / len(relevant), f"ndcg{k}": dcg / idcg,
                        f"hit{k}": float(n_hits > 0)})
        rows.append(row)
    return rows


def summarize_multitarget_rows(rows, ks=(10, 50)):
    """Return macro-averaged metrics over users, optionally by training group.

    Row metrics, rather than aggregated hit counts, determine Recall. A user
    with many relevant items has the same evaluation weight as any other user.
    Integer relevant_count remains descriptive and never weights that mean.
    """
    cutoffs = _cutoffs(ks)
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation population")
    metric_keys = [f"{metric}{k}" for k in cutoffs for metric in ("recall", "ndcg", "hit")]
    for row in rows:
        count = row.get("relevant_count")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
            raise ValueError("Each row needs a positive integer relevant_count")
        if row.get("group") not in {None, "cold", "middle", "warm"}:
            raise ValueError("Unknown training-count group")
        for key in metric_keys:
            value = row.get(key)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be a finite numeric metric in [0, 1]")
    result = {}
    for group in ("all", "cold", "middle", "warm"):
        selected = rows if group == "all" else [row for row in rows if row.get("group") == group]
        if not selected:
            continue
        result[group] = {"users": len(selected), "relevant_items": sum(int(row["relevant_count"]) for row in selected)}
        result[group].update({key: float(np.mean([row[key] for row in selected])) for key in metric_keys})
    return result
