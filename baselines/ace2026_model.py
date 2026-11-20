"""Independent ACE-SASRec implementation for the shared ML-1M protocol.

The initialization follows Eqs. (5), (6), and (8) of Lee et al., SIGIR 2026,
https://arxiv.org/abs/2605.29322. The published implementation was inspected at
DCheol/ACE commit 3e099052d9759b841a5192a2ae20a3b3ac02c6c6; its source is
not vendored here. Attention uses PyTorch's standard multi-head attention and
post-residual layer normalization. No collaborative checkpoint is used.

This adapter supervises every valid shifted position in each retained training
history. The authors instead use RecBole prefix examples. Evaluation and input
features are shared with the local comparison; this is not a reproduction of
the paper's original leave-one-out results.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _number(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and ' + ('nonnegative' if zero else 'positive'))
    return float(value)


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _digest_arrays(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        result.update(json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':')).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def _features(text_features):
    value = torch.as_tensor(text_features).detach().cpu()
    if value.ndim != 2 or min(value.shape) < 2 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError('Features must be a finite floating matrix with PAD plus real items')
    if bool(value[0].ne(0).any()):
        raise ValueError('Feature row zero must be PAD zero')
    return np.ascontiguousarray(value.double().numpy())


def build_ace_svd(text_features):
    """Uncentered CPU float64 thin SVD, reusable across initialization grids.

    Returned tensors are CPU float64 and safe for weights-only torch storage.
    The exact feature/factor hashes prevent stale or modified cache reuse.
    """
    features = _features(text_features)
    u, s, _ = np.linalg.svd(features[1:], full_matrices=False)
    return {'u': torch.from_numpy(np.ascontiguousarray(u)),
            's': torch.from_numpy(np.ascontiguousarray(s)),
            'feature_sha256': _digest_arrays(features), 'feature_shape': list(features.shape),
            'decomposition': 'uncentered_numpy_thin_svd_float64', 'factor_sha256': _digest_arrays(u, s)}


def ace_item_initialization(text_features, d_model, reg_weight, scale, svd_factors=None, dtype=torch.float32):
    """Return PAD-inclusive ACE table and JSON-safe initialization provenance.

    Scale is the population standard deviation across *all* non-PAD matrix
    entries, not row normalization and not a logit temperature. Numerically
    null singular directions are discarded at lambda=0, avoiding 0/0.
    """
    _integer(d_model, 'd_model')
    reg_weight, scale = _number(reg_weight, 'reg_weight', True), _number(scale, 'scale')
    features = _features(text_features)
    factors = build_ace_svd(text_features) if svd_factors is None else svd_factors
    if not isinstance(factors, dict) or factors.get('feature_sha256') != _digest_arrays(features) or factors.get('feature_shape') != list(features.shape):
        raise ValueError('ACE SVD cache belongs to different features')
    if factors.get('decomposition') != 'uncentered_numpy_thin_svd_float64':
        raise ValueError('ACE requires an uncentered thin SVD cache')
    u, s = factors.get('u'), factors.get('s')
    rank = min(features.shape[0] - 1, features.shape[1])
    if (not isinstance(u, torch.Tensor) or not isinstance(s, torch.Tensor)
            or u.device.type != 'cpu' or s.device.type != 'cpu'
            or u.dtype != torch.float64 or s.dtype != torch.float64
            or u.shape != (features.shape[0] - 1, rank) or s.shape != (rank,)
            or not torch.isfinite(u).all() or not torch.isfinite(s).all()
            or bool((s < 0).any()) or bool((s[1:] > s[:-1]).any()) or d_model > rank):
        raise ValueError('Invalid ACE SVD factor dimensions, dtype or spectrum')
    u_np, s_np = u.numpy(), s.numpy()
    if factors.get('factor_sha256') != _digest_arrays(u_np, s_np):
        raise ValueError('ACE SVD cache factor hash differs')
    threshold = np.finfo(np.float64).eps * max(features[1:].shape) * float(s_np[0])
    selected = s_np[:d_model]
    if reg_weight == 0:
        shrink = (selected > threshold).astype(np.float64)
    else:
        shrink = np.sqrt(selected ** 2 / (selected ** 2 + reg_weight))
    reshaped = u_np[:, :d_model] * shrink
    dispersion = float(reshaped.std(ddof=0))
    if not math.isfinite(dispersion) or dispersion <= 0:
        raise ValueError('ACE initialization has no nonzero spectral information')
    scaled = reshaped / (dispersion + 1e-12) * scale
    full = np.zeros((len(features), d_model), dtype=np.float64)
    full[1:] = scaled
    result = torch.from_numpy(full).to(dtype=dtype)
    report = {'method': 'ACE equations 5/6/8; uncentered thin SVD',
              'reference_commit': '3e099052d9759b841a5192a2ae20a3b3ac02c6c6',
              'feature_sha256': factors['feature_sha256'], 'feature_shape': list(features.shape),
              'factor_sha256': factors['factor_sha256'], 'd_model': d_model,
              'reg_weight': reg_weight, 'scale': scale, 'centered': False,
              'svd_dtype_device': 'float64 CPU', 'scale_rule': 'whole non-PAD matrix std(ddof=0)',
              'initial_std': float(result[1:].double().std(unbiased=False)),
              'numerical_rank': int((s_np > threshold).sum()), 'pad_zero': True,
              'item_table_trainable': True, 'user_item_l2_normalized': False,
              'initial_table_sha256': _digest_arrays(result.numpy())}
    return result, report


def catalog_ce(queries, items, targets, chunk_size=1024):
    """Position-mean raw-dot CE including PAD class and all known positives."""
    _integer(chunk_size, 'chunk_size')
    if (queries.ndim != 2 or items.ndim != 2 or queries.shape[1] != items.shape[1]
            or not queries.is_floating_point() or queries.dtype != items.dtype or queries.device != items.device
            or targets.shape != (len(queries),) or targets.dtype != torch.long or targets.device != queries.device
            or bool(((targets <= 0) | (targets >= len(items))).any())):
        raise ValueError('Invalid catalog CE tensors or non-PAD targets')
    if not len(queries):
        return queries.sum() * 0. + items.sum() * 0.
    def summed(query, table, target):
        return F.cross_entropy(query @ table.T, target, reduction='sum')
    total = queries.new_zeros(())
    gradients = torch.is_grad_enabled() and (queries.requires_grad or items.requires_grad)
    for start in range(0, len(queries), chunk_size):
        args = (queries[start:start + chunk_size], items, targets[start:start + chunk_size])
        total = total + (checkpoint(summed, *args, use_reentrant=False, preserve_rng_state=False)
                         if gradients else summed(*args))
    return total / len(queries)


class _PostNormBlock(nn.Module):
    def __init__(self, width, heads, inner, dropout, attn_dropout, eps):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=attn_dropout, batch_first=True)
        self.attention_out = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(width, eps=eps)
        self.expand = nn.Linear(width, inner)
        self.contract = nn.Linear(inner, width)
        self.ff_out = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(width, eps=eps)

    def forward(self, values, mask):
        attended, _ = self.attention(values, values, values, attn_mask=mask, need_weights=False)
        values = self.attention_norm(values + self.attention_out(attended))
        transformed = self.contract(F.gelu(self.expand(values), approximate='none'))
        return self.ff_norm(values + self.ff_out(transformed))


class ACE2026Model(nn.Module):
    """Trainable ACE item initialization followed by a raw-dot SASRec encoder."""
    def __init__(self, text_features, n_items=None, d_model=128, max_len=50, n_layers=2, n_heads=2,
                 inner_size=256, dropout=.5, attn_dropout=.3, layer_norm_eps=1e-12,
                 initializer_range=.02, reg_weight=50., scale=.1, ce_chunk_size=1024, svd_factors=None):
        super().__init__()
        for name, value in [('d_model', d_model), ('max_len', max_len), ('n_layers', n_layers),
                            ('n_heads', n_heads), ('inner_size', inner_size), ('ce_chunk_size', ce_chunk_size)]:
            _integer(value, name)
        if d_model % n_heads:
            raise ValueError('d_model must divide into n_heads')
        for name, value in [('dropout', dropout), ('attn_dropout', attn_dropout)]:
            if not 0 <= _number(value, name, True) < 1:
                raise ValueError(name + ' must be in [0,1)')
        layer_norm_eps, initializer_range = _number(layer_norm_eps, 'layer_norm_eps'), _number(initializer_range, 'initializer_range')
        features = _features(text_features)
        inferred = len(features) - 1
        if n_items is not None and (type(n_items) is not int or n_items != inferred):
            raise ValueError('n_items differs from the PAD-inclusive feature matrix')
        initial, report = ace_item_initialization(text_features, d_model, reg_weight, scale, svd_factors)
        self.n_items, self.d_model, self.max_len = inferred, d_model, max_len
        self.n_layers, self.n_heads, self.inner_size = n_layers, n_heads, inner_size
        self.dropout_prob, self.attn_dropout_prob = float(dropout), float(attn_dropout)
        self.layer_norm_eps, self.initializer_range = layer_norm_eps, initializer_range
        self.reg_weight, self.scale, self.ce_chunk_size = float(reg_weight), float(scale), ce_chunk_size
        self.control, self.score_scale = 'ace2026_sasrec', 1.
        self.item_embedding = nn.Embedding(self.n_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_PostNormBlock(d_model, n_heads, inner_size, dropout, attn_dropout, layer_norm_eps)
                                     for _ in range(n_layers)])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attention.in_proj_weight, std=initializer_range)
            nn.init.zeros_(block.attention.in_proj_bias)
        nn.init.normal_(self.position_embedding.weight, std=initializer_range)
        with torch.no_grad():
            self.item_embedding.weight.copy_(initial)
        self.initialization_report = report

    def model_config(self):
        return {'n_items': self.n_items, 'd_model': self.d_model, 'max_len': self.max_len,
                'n_layers': self.n_layers, 'n_heads': self.n_heads, 'inner_size': self.inner_size,
                'dropout': self.dropout_prob, 'attn_dropout': self.attn_dropout_prob,
                'layer_norm_eps': self.layer_norm_eps, 'initializer_range': self.initializer_range,
                'reg_weight': self.reg_weight, 'scale': self.scale, 'ce_chunk_size': self.ce_chunk_size}

    def _ids(self, ids, name):
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2
                or not 1 <= ids.shape[1] <= self.max_len or bool(((ids < 0) | (ids > self.n_items)).any())):
            raise ValueError(name + ' must be a valid [batch,length] integer catalog tensor')
        return ids

    def _encode(self, histories):
        histories = self._ids(histories, 'histories')
        valid = histories.ne(0)
        lengths = valid.sum(1)
        ranks = valid.long().cumsum(1) - 1
        packed = torch.zeros_like(histories)
        owners = torch.arange(len(histories), device=histories.device)[:, None].expand_as(histories)
        packed[owners[valid], ranks[valid]] = histories[valid]
        positions = torch.arange(histories.shape[1], device=histories.device)
        values = self.input_dropout(self.input_norm(self.item_embedding(packed) + self.position_embedding(positions)[None]))
        causal = torch.ones(histories.shape[1], histories.shape[1], dtype=torch.bool, device=histories.device).tril()
        permitted = causal[None] & packed.ne(0)[:, None, :]
        # Finite bias matches native SASRec and keeps unused all-PAD queries finite.
        mask = torch.zeros_like(permitted, dtype=values.dtype).masked_fill(~permitted, -10000.)
        mask = mask.repeat_interleave(self.n_heads, dim=0)
        for block in self.blocks:
            values = block(values, mask)
        return values, ranks, valid, lengths

    def encode_states(self, histories):
        """Causal states aligned to the caller's original PAD layout."""
        packed, ranks, valid, _ = self._encode(histories)
        aligned = packed.gather(1, ranks.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.d_model))
        return aligned.masked_fill(~valid.unsqueeze(-1), 0.)

    def user_vectors(self, histories, diagnostics=None):
        packed, _, _, lengths = self._encode(histories)
        result = packed[torch.arange(len(histories), device=histories.device), (lengths - 1).clamp_min(0)]
        return result.masked_fill(lengths.eq(0).unsqueeze(-1), 0.)

    def item_vectors(self):
        # Lookup, rather than exposing the parameter directly, keeps the PAD
        # class fixed at zero even when full CE differentiates all candidates.
        return self.item_embedding(torch.arange(self.n_items + 1, device=self.item_embedding.weight.device))

    def loss_batch(self, histories, positives, neg_ids=None):
        histories, positives = self._ids(histories, 'histories'), self._ids(positives, 'positives')
        if histories.shape != positives.shape:
            raise ValueError('Aligned history and next-item target tensors required')
        valid = histories.ne(0) & positives.ne(0)
        states = self.encode_states(histories)
        loss = catalog_ce(states[valid], self.item_vectors(), positives[valid], self.ce_chunk_size)
        return {'loss': loss, 'rec_loss': loss, 'reg_loss': loss * 0.}

    def forward(self, histories, positives=None, neg_ids=None):
        return self.user_vectors(histories) if positives is None else self.loss_batch(histories, positives, neg_ids)


ACEModel = ACE2026Model
