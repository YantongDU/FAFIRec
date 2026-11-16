"""Validated prepared data, explicit feature provenance, and next-item batches."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{Path(path).name}:{line_number}: invalid JSON') from error
            if not isinstance(value, dict):
                raise ValueError(f'{Path(path).name}:{line_number}: expected an object')
            yield value


def _positive_item(value, n_items):
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= n_items


def assign_training_groups(records):
    """Stable bottom/top ten-percent *rank* groups, without overlapping users.

    This is a descriptive relative split, not an absolute cold-start budget.
    User order in interactions.jsonl breaks count ties. With a single user no
    nonoverlapping tails exist, so its group remains middle.
    """
    count = len(records)
    order = sorted(range(count), key=lambda i: (len(records[i]['train']), i))
    tail = min(max(1, int(count * 0.1)), count // 2)
    for record in records:
        record['group'] = 'middle'
        record['training_interactions'] = len(record['train'])
    for index in order[:tail]:
        records[index]['group'] = 'cold'
    if tail:
        for index in order[-tail:]:
            records[index]['group'] = 'warm'
    result = {'rule': 'stable training-count rank; bottom/top 10%; input order breaks ties',
              'tail_users': tail, 'overlap': 0}
    for group in ('cold', 'middle', 'warm'):
        lengths = [len(r['train']) for r in records if r['group'] == group]
        result[group] = {'users': len(lengths),
                         'training_count_range': [min(lengths), max(lengths)] if lengths else None}
    return result


def _features(items, feature_mode, embedding_path, lexical_dim):
    if feature_mode == 'embeddings':
        if embedding_path is None:
            raise ValueError('Embeddings mode requires embedding_path in the experiment configuration; lexical mode must be explicit')
        path = Path(embedding_path)
        manifest_path = path.with_suffix('.manifest.json')
        if not manifest_path.is_file():
            raise ValueError(f'Missing feature mapping manifest: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        expected = ['<PAD>'] + [item['raw_id'] for item in items]
        if manifest.get('row_ids') != expected:
            raise ValueError('Embedding row-ID mapping does not match items.jsonl order')
        actual_hash = file_sha256(path)
        if manifest.get('embedding_sha256') != actual_hash:
            raise ValueError('Embedding SHA-256 does not match its feature manifest')
        features = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError('Text embeddings must be a two-dimensional tensor')
        if features.shape[0] != len(items) + 1 or features.shape[1] < 1:
            raise ValueError('Text embedding shape must include exactly padding row zero and all item rows')
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError('Text embeddings must be finite floating-point values')
        if bool(features[0].ne(0).any()):
            raise ValueError('Embedding padding row zero must contain only zeros')
        features = features.detach().to(dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Text embeddings overflow float32')
        info = {'feature_type': 'externally_provided_frozen_embeddings',
                'feature_provenance': manifest,
                'embedding_mapping_verified': True,
                'embedding_sha256': actual_hash,
                'embedding_manifest_sha256': file_sha256(manifest_path)}
    elif feature_mode == 'lexical':
        if embedding_path is not None:
            raise ValueError('Lexical mode cannot also specify an embedding artifact')
        if isinstance(lexical_dim, bool) or not isinstance(lexical_dim, int) or lexical_dim < 1:
            raise ValueError('lexical_dim must be a positive integer')
        features = np.zeros((len(items) + 1, lexical_dim), dtype=np.float32)
        for index, item in enumerate(items, 1):
            for token in re.findall(r'[a-z0-9]+', item['text'].lower()):
                digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], 'little') % lexical_dim
                features[index, bucket] += 1 if digest[4] & 1 else -1
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        features = torch.from_numpy(features)
        info = {'feature_type': 'signed_hash_metadata_bag_of_words; NOT LLM features',
                'embedding_mapping_verified': False,
                'feature_provenance': {'algorithm': 'blake2b-64 signed hashing',
                                       'tokenizer': '[a-z0-9]+ after lowercasing',
                                       'feature_dim': lexical_dim,
                                       'empty_text_items': sum(not item['text'].strip() for item in items)}}
    else:
        raise ValueError("feature_mode must be 'embeddings' or 'lexical'")
    info['feature_dim'] = int(features.shape[1])
    info['feature_tensor_sha256'] = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    return features, info


def load_bundle(data_dir, feature_mode='embeddings', embedding_path=None, lexical_dim=128):
    """Load a version-1 bundle without changing splits or guessing feature rows.

    Returns (records, CPU float32 features including padding, manifest). Target
    values are only validated here; they do not construct diagnostics, group
    assignments, lexical features, or training negative pools.
    """
    root = Path(data_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported data format_version; run prepare_data.py')
    for name in ('items.jsonl', 'interactions.jsonl'):
        expected = manifest.get('files', {}).get(name)
        if not expected or expected != file_sha256(root / name):
            raise ValueError(f'Data file SHA-256 mismatch or missing hash: {name}')
    items = list(_jsonl(root / 'items.jsonl'))
    if not items:
        raise ValueError('The item catalog must be nonempty')
    raw_ids = set()
    for index, item in enumerate(items, 1):
        if not _positive_item(item.get('id'), len(items)) or item['id'] != index:
            raise ValueError('Item IDs must be contiguous 1..N in items.jsonl row order')
        raw = item.get('raw_id')
        if not isinstance(raw, str) or not raw or raw in raw_ids or raw == '<PAD>':
            raise ValueError('Raw item identifiers must be unique nonempty strings excluding <PAD>')
        if not isinstance(item.get('text'), str):
            raise ValueError('Each catalog item needs a text string, which may be empty')
        raw_ids.add(raw)
    records = list(_jsonl(root / 'interactions.jsonl'))
    if not records:
        raise ValueError('No eligible training users')
    users = set()
    for record in records:
        user = record.get('user')
        if not isinstance(user, str) or not user or user in users:
            raise ValueError('User identifiers must be unique nonempty strings')
        users.add(user)
        train = record.get('train')
        if not isinstance(train, list) or len(train) < 2:
            raise ValueError('Each user needs at least two training observations for diagnostics')
        if any(not _positive_item(item, len(items)) for item in train):
            raise ValueError('Invalid item ID in training history')
        if any(not _positive_item(record.get(split), len(items)) for split in ('validation', 'test')):
            raise ValueError('Validation and test targets must be scalar nonpadding catalog item IDs')
        if len(set(train)) == len(items):
            raise ValueError('A training user has no unseen negative candidate')
    groups = assign_training_groups(records)
    features, feature_info = _features(items, feature_mode, embedding_path, lexical_dim)
    result = dict(manifest)
    result.update({'dataset': manifest.get('dataset', root.name), 'users': len(records),
                   'items': len(items), 'interactions': sum(len(r['train']) + 2 for r in records),
                   'grouping': groups, **feature_info})
    result['data_manifest_sha256'] = file_sha256(root / 'manifest.json')
    return records, features, result


def pad_history(history, max_len):
    if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1:
        raise ValueError('max_len must be a positive integer')
    recent = list(history[-max_len:])
    return [0] * (max_len - len(recent)) + recent


class TrainingDataset(Dataset):
    """One user per row, shifted next-item supervision, train-set-excluding negatives.

    The local RNG is deterministic with num_workers=0. Multiprocess worker RNG
    replay is deliberately not supported by the experiment runner.
    """
    def __init__(self, records, n_items, max_len, seed):
        if n_items < 1:
            raise ValueError('n_items must be positive')
        pad_history([], max_len)
        self.records, self.n_items, self.max_len = records, n_items, max_len
        self.rng = np.random.default_rng(seed)
        self._dense_pools = {}
        for index, record in enumerate(records):
            seq = record['train']
            if len(seq) < 2 or any(not _positive_item(item, n_items) for item in seq):
                raise ValueError('Invalid training sequence')
            seen = set(seq)
            if len(seen) >= n_items:
                raise ValueError('No unobserved training negative candidates')
            if len(seen) > n_items // 2:
                self._dense_pools[index] = np.array([i for i in range(1, n_items + 1) if i not in seen], dtype=np.int64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        seq = self.records[index]['train']
        history, positive = seq[:-1][-self.max_len:], seq[1:][-self.max_len:]
        if index in self._dense_pools:
            negative = self.rng.choice(self._dense_pools[index], size=len(positive)).tolist()
        else:
            seen, negative = set(seq), []
            for _ in positive:
                candidate = int(self.rng.integers(1, self.n_items + 1))
                while candidate in seen:
                    candidate = int(self.rng.integers(1, self.n_items + 1))
                negative.append(candidate)
        tensors = [torch.tensor(pad_history(value, self.max_len), dtype=torch.long)
                   for value in (history, positive, negative)]
        return (*tensors, index)


def evaluation_batch(records, indices, split, max_len):
    if split not in ('validation', 'test'):
        raise ValueError("Evaluation split must be 'validation' or 'test'")
    histories, targets = [], []
    for index in indices:
        record = records[int(index)]
        history = record['train'] + ([record['validation']] if split == 'test' else [])
        histories.append(pad_history(history, max_len))
        targets.append(record[split])
    if not histories:
        return torch.empty((0, max_len), dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(histories, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
