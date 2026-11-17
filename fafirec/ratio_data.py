"""Version-2 chronological 80/10/10 data with complete held-out item sets.

The version-1 leave-two-out interface remains in ``fafirec.data``. Here each
validation/test value is an event list; evaluation relevance deduplicates item
identities while histories preserve every observed event.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import torch

from .data import (_features, _jsonl, _positive_item, assign_training_groups,
                   file_sha256, pad_history)


def ratio_partition_lengths(count):
    """Exact integer floors of 0.8n and 0.1n; test receives the remainder."""
    if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
        raise ValueError("Event count must be a nonnegative integer")
    training, validation = (8 * int(count)) // 10, int(count) // 10
    return training, validation, int(count) - training - validation


def load_ratio_bundle(data_dir, feature_mode='embeddings', embedding_path=None,
                      lexical_dim=128):
    """Load the strict version-2 ratio bundle, without truncating target sets.

    Returns records, CPU float32 feature tensor, and a provenance manifest. The
    loader recomputes reporting groups from complete training counts only. It
    never uses validation/test labels to form features or training negatives.
    """
    root = Path(data_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('format_version') != 2:
        raise ValueError('Expected ratio format_version 2; use prepare_ml1m_paper.py')
    if manifest.get('split', {}).get('ratios') != [0.8, 0.1, 0.1]:
        raise ValueError('Ratio manifest must declare 80/10/10 event partitions')
    for name in ('items.jsonl', 'interactions.jsonl'):
        expected = manifest.get('files', {}).get(name)
        if not expected or file_sha256(root / name) != expected:
            raise ValueError(f'Data file SHA-256 mismatch or missing hash: {name}')

    items = list(_jsonl(root / 'items.jsonl'))
    if not items:
        raise ValueError('The metadata catalog must be nonempty')
    raw_ids = set()
    for index, item in enumerate(items, 1):
        if not _positive_item(item.get('id'), len(items)) or item['id'] != index:
            raise ValueError('Item IDs must be contiguous 1..N in metadata row order')
        raw_id = item.get('raw_id')
        if not isinstance(raw_id, str) or not raw_id or raw_id == '<PAD>' or raw_id in raw_ids:
            raise ValueError('Raw item identifiers must be unique nonempty strings excluding <PAD>')
        if not isinstance(item.get('text'), str):
            raise ValueError('Every metadata item needs a text string')
        raw_ids.add(raw_id)

    records = list(_jsonl(root / 'interactions.jsonl'))
    if not records:
        raise ValueError('No eligible users in the ratio bundle')
    users = set()
    for record in records:
        user = record.get('user')
        if not isinstance(user, str) or not user or user in users:
            raise ValueError('User identifiers must be unique nonempty strings')
        users.add(user)
        for split in ('train', 'validation', 'test'):
            values = record.get(split)
            if not isinstance(values, list):
                raise ValueError(f'{split} must be an event list, not a scalar target')
            required = 2 if split == 'train' else 1
            if len(values) < required:
                raise ValueError('Need at least two training events and nonempty validation/test partitions')
            if any(not _positive_item(value, len(items)) for value in values):
                raise ValueError(f'Invalid catalog item in {split} event list')
        actual_lengths = tuple(len(record[split]) for split in ('train', 'validation', 'test'))
        if actual_lengths != ratio_partition_lengths(sum(actual_lengths)):
            raise ValueError('Event counts do not match exact floor-rounded 80/10/10 partitions')
        if len(set(record['train'])) == len(items):
            raise ValueError('A training user has no unseen negative candidate')

    groups = assign_training_groups(records)
    features, feature_info = _features(items, feature_mode, embedding_path, lexical_dim)
    result = dict(manifest)
    result.update({'users': len(records), 'items': len(items),
                   'interactions': sum(len(r[s]) for r in records for s in ('train', 'validation', 'test')),
                   'training_events': sum(len(r['train']) for r in records),
                   'validation_events': sum(len(r['validation']) for r in records),
                   'test_events': sum(len(r['test']) for r in records),
                   'grouping': groups, **feature_info})
    result['data_manifest_sha256'] = file_sha256(root / 'manifest.json')
    result['evaluation_relevance'] = 'unique held-out item IDs; event histories retain duplicates'
    return records, features, result


def ratio_evaluation_batch(records, indices, split, max_len):
    """Return histories [B,L] and complete, deduplicated relevant item-ID lists.

    Validation observes training events only. Test observes training plus the
    complete validation segment; no test event is appended to a history.
    Deduplication applies only to relevance, never to observed event histories.
    Candidate filtering is a separate, explicitly selected evaluation policy.
    """
    if split not in ('validation', 'test'):
        raise ValueError("Evaluation split must be 'validation' or 'test'")
    histories, target_sets = [], []
    for index in indices:
        if not isinstance(index, Integral) or isinstance(index, bool):
            raise ValueError('Evaluation indices must be integer record indices')
        record = records[int(index)]
        history = list(record['train'])
        if split == 'test':
            history.extend(record['validation'])
        targets = record[split]
        if not isinstance(targets, list) or not targets:
            raise ValueError('Evaluation needs a nonempty held-out event list')
        histories.append(pad_history(history, max_len))
        target_sets.append(list(dict.fromkeys(targets)))
    if not histories:
        pad_history([], max_len)
        return torch.empty((0, max_len), dtype=torch.long), []
    return torch.tensor(histories, dtype=torch.long), target_sets
