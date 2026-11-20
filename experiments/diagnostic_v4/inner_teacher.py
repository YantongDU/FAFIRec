"""Fixed-budget, train-internal ACE-style teacher for diagnostic calibration.

ACE mathematics is reused with attribution to Lee et al. (SIGIR 2026), not
presented as a FAFIRec contribution. The internal calibration tail is excluded
from every training record. This module has no evaluation/early-stop path and
never uses outer validation/test labels to construct a training example.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.utils.data import DataLoader

from baselines.ace2026_model import ACE2026Model
from fafirec.data import TrainingDataset
from fafirec.runner import (cpu_state, digest, file_digest, json_dump, restore_rng,
    rng_state, runtime_identity, seed_everything, torch_dump)
from tools import ace2026_experiment as ace

PROTOCOL = 'fafirec_v4_train_internal_fixed_budget_teacher_v1'
DEFAULT_REFERENCE = 'E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas/seed_2025/run'
DEFAULT_CACHE = 'E:/论文/FAFIRec_ACE2026/data/ace2026_qwen_svd.pt'
MODEL = dict(d_model=128, max_len=50, n_layers=2, n_heads=2, inner_size=256,
    dropout=.5, attn_dropout=.3, layer_norm_eps=1e-12, initializer_range=.02,
    reg_weight=500., scale=.1, ce_chunk_size=1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_inner_records(records):
    """Return (prefix-only training rows, calibration rows, split metadata).

    Calibration rows have ``user``, optional ``group``, ``prefix`` and
    ``calibration``. Prefix/tail positions partition that user's original train
    sequence. Repeated item identities are allowed across different positions.
    A length-two train has an empty calibration tail to retain two train events.
    All unrecognized fields, including outer validation/test labels, are dropped.
    """
    require(isinstance(records, list) and records, 'Nonempty training records required')
    inner, calibration, original, seen = [], [], [], set()
    for record in records:
        user, seq = record.get('user'), record.get('train')
        require(isinstance(user, str) and user and user not in seen, 'Distinct nonempty user IDs required')
        seen.add(user)
        require(isinstance(seq, list) and len(seq) >= 2
            and all(type(i) is int and i > 0 for i in seq), 'At least two positive train item IDs required')
        tail = min(10, max(1, int(.2 * len(seq))))
        tail = min(tail, len(seq) - 2)
        boundary = len(seq) - tail
        prefix, held = list(seq[:boundary]), list(seq[boundary:])
        require(len(prefix) >= 2 and prefix + held == seq
            and boundary >= 2 and boundary + len(held) == len(seq), 'Internal split overlaps or drops positions')
        common = {'user': user}
        if 'group' in record:
            common['group'] = copy.deepcopy(record['group'])
        inner.append({**common, 'train': prefix})
        calibration.append({**common, 'prefix': list(prefix), 'calibration': held})
        original.append({**common, 'train': list(seq)})
    metadata = {'rule': 'tail=min(10,max(1,int(.2*n))); tail=min(tail,n-2)',
        'users': len(inner), 'calibration_users': sum(bool(r['calibration']) for r in calibration),
        'inner_train_events': sum(len(r['train']) for r in inner),
        'calibration_events': sum(len(r['calibration']) for r in calibration),
        'source_train_records_sha256': digest(original), 'inner_train_records_sha256': digest(inner),
        'calibration_records_sha256': digest(calibration),
        'strict_position_disjoint': True, 'outer_labels_discarded': True}
    return inner, calibration, metadata


def source_identity():
    # Explicit additions: other new v4 modules may be edited while this teacher
    # trains without altering its source identity or its resume contract.
    paths = [ROOT / 'baselines/ace2026_model.py', ROOT / 'baselines/__init__.py',
             ROOT / 'tools/ace2026_experiment.py', Path(__file__)]
    return {**ace.reference.source_identity(), **{
        p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths if p.is_file()}}


def normalize_config(raw):
    defaults = dict(seed=2025, epochs=135, output_dir=None, device='cuda', threads=4,
        reference_run=DEFAULT_REFERENCE, reference_seed=2025, reference_variant='ace_selected',
        svd_cache=DEFAULT_CACHE)
    require(isinstance(raw, dict) and not set(raw) - set(defaults), 'Unknown teacher configuration fields')
    cfg = {**defaults, **raw}
    for key in ('seed', 'reference_seed'):
        require(type(cfg[key]) is int and cfg[key] >= 0, key + ' must be a nonnegative integer')
    for key in ('epochs', 'threads'):
        require(type(cfg[key]) is int and cfg[key] > 0, key + ' must be a positive integer')
    require(cfg['device'] in ('cpu', 'cuda'), 'Teacher device must be cpu or cuda')
    require(isinstance(cfg['reference_variant'], str) and cfg['reference_variant'], 'Reference variant required')
    for key in ('output_dir', 'reference_run', 'svd_cache'):
        require(isinstance(cfg[key], str) and cfg[key], key + ' required')
        p = Path(cfg[key])
        cfg[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    out, ref = Path(cfg['output_dir']), Path(cfg['reference_run'])
    require(out != ref and out not in ref.parents and ref not in out.parents,
            'Teacher output overlaps frozen reference')
    return cfg


def prepare_inputs(cfg):
    """Read actual frozen ACE inputs, immediately strip outer labels and split."""
    reference = Path(cfg['reference_run'])
    manifest = ace.read(reference / 'manifest.json')
    require(manifest.get('protocol') == ace.PROTOCOL
        and manifest.get('fingerprint') == ace.fingerprint(manifest)
        and manifest.get('source_hashes') == ace.source_identity(), 'Frozen ACE reference source/manifest differs')
    base = ace.normalize_config(manifest['config'])
    require(base['output_dir'] == str(reference) and cfg['reference_seed'] in base['seeds']
        and cfg['reference_variant'] in {v['name'] for v in base['variants']}, 'Reference seed or variant differs')
    base['svd_cache'] = cfg['svd_cache']
    records, features, identity, _unused_eval, provenance, factors, factor_identity = ace.inputs(base)
    inner, calibration, split = split_inner_records(records)
    del records, _unused_eval
    require(identity == manifest['identity'], 'Frozen reference data/feature identity differs')
    require(all(set(r) <= {'user', 'group', 'train'} for r in inner), 'Teacher training exposes outer labels')
    provenance = {'reference_manifest_sha256': file_digest(reference / 'manifest.json'),
        'reference_seed': cfg['reference_seed'], 'reference_variant': cfg['reference_variant'],
        'identity': identity, 'svd': factor_identity,
        'svd_cache_sha256': file_digest(cfg['svd_cache']),
        'reference_input_provenance_sha256': digest(provenance),
        'features_are_static_metadata': True, 'cf_weights_used_to_initialize_teacher': False}
    return inner, calibration, features, factors, split, provenance


def fingerprint(manifest):
    return digest({k: v for k, v in manifest.items() if k not in ('fingerprint', 'created_utc')})


def manifest_for(cfg, split, provenance, device):
    value = {'protocol': PROTOCOL, 'config': cfg, 'model_config': MODEL, 'batch_size': 256,
        'optimizer': {'name': 'Adam', 'lr': .0005, 'weight_decay': 0.}, 'grad_clip': None,
        'split': split, 'inputs': provenance, 'source_hashes': source_identity(),
        'runtime': runtime_identity(device), 'checkpoint_selection': 'last fixed epoch, no scoring',
        'calibration_used_for_training_or_early_stop': False, 'outer_labels_used_for_training_or_scoring': False,
        'initialization': 'ACE spectral initialization from static text; random encoder; no fitted reference weights',
        'created_utc': datetime.now(timezone.utc).isoformat()}
    value['fingerprint'] = fingerprint(value)
    return value


def _header(manifest, model):
    return {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'],
        'seed': manifest['config']['seed'], 'fixed_epoch_budget': manifest['config']['epochs'],
        'model_config': model.model_config(),
        'inner_train_records_sha256': manifest['split']['inner_train_records_sha256']}


def fit_teacher(model, records, manifest, folder, resume=False):
    """Train only to a fixed epoch count; callable with a small CPU model in tests."""
    require(all(set(r) <= {'user', 'group', 'train'} for r in records), 'Teacher may only receive prefix-only train records')
    require(digest(records) == manifest['split']['inner_train_records_sha256'], 'Teacher training rows changed')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg, expected = manifest['config'], _header(manifest, model)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=0.)
    logs, first = [], 1
    if resume and (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        require(all(saved.get(k) == v for k, v in expected.items()), 'Teacher resume identity differs')
        require(type(saved['epoch']) is int and 1 <= saved['epoch'] <= cfg['epochs']
            and len(saved['logs']) == saved['epoch']
            and saved['finished'] == (saved['epoch'] == cfg['epochs']), 'Teacher resume budget differs')
        logs, first = saved['logs'], saved['epoch'] + 1
        expected_steps = math.ceil(len(records) / manifest['batch_size'])
        for i, row in enumerate(logs, 1):
            require(row['epoch'] == i and row['optimizer_steps'] == expected_steps
                and set(row) == {'epoch', 'optimizer_steps', 'positions', 'loss', 'rec_loss', 'reg_loss'}
                and all(math.isfinite(row[k]) for k in ('loss', 'rec_loss', 'reg_loss')),
                'Teacher resume training log differs')
        if (folder / 'training.json').exists():
            prior = ace.read(folder / 'training.json')
            require(prior == logs[:len(prior)], 'Teacher training log is not a checkpoint prefix')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
    else:
        require(not (folder / 'last.pt').exists() and not (folder / 'teacher.pt').exists(), 'Existing teacher requires resume')
    for epoch in range(first, cfg['epochs'] + 1):
        model.train()
        epoch_seed = cfg['seed'] + 1000 + epoch
        data = TrainingDataset(records, model.n_items, model.max_len, epoch_seed)
        loader = DataLoader(data, batch_size=manifest['batch_size'], shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(epoch_seed))
        totals, positions, steps = {k: 0. for k in ('loss', 'rec_loss', 'reg_loss')}, 0, 0
        for hist, positive, _unused_negative, _indices in loader:
            optimizer.zero_grad(set_to_none=True)
            hist, positive = hist.to(device), positive.to(device)
            losses = model.loss_batch(hist, positive)
            require(all(torch.isfinite(losses[k]).item() for k in totals), 'Nonfinite inner-teacher loss')
            losses['loss'].backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
                'Nonfinite inner-teacher gradient')
            optimizer.step()
            amount = int((hist.ne(0) & positive.ne(0)).sum())
            for key in totals:
                totals[key] += float(losses[key].detach()) * amount
            positions, steps = positions + amount, steps + 1
        log = {'epoch': epoch, 'optimizer_steps': steps, 'positions': positions,
            **{k: v / positions for k, v in totals.items()}}
        logs.append(log)
        saved = {**expected, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
            'rng': rng_state(), 'epoch': epoch, 'finished': epoch == cfg['epochs'], 'logs': logs}
        torch_dump(folder / 'last.pt', saved)
        json_dump(folder / 'training.json', logs)
        print(json.dumps({'seed': cfg['seed'], **log}), flush=True)
    # Reconcile a JSON log that lagged the last atomic checkpoint at interruption.
    json_dump(folder / 'training.json', logs)
    final = {**expected, 'epoch': cfg['epochs'], 'model': cpu_state(model),
        'optimizer_steps': sum(r['optimizer_steps'] for r in logs),
        'checkpoint_selection': 'last fixed epoch; calibration and outer labels never scored'}
    path = folder / 'teacher.pt'
    if path.exists():
        prior = torch.load(path, map_location='cpu', weights_only=True)
        require({k: v for k, v in prior.items() if k != 'model'}
            == {k: v for k, v in final.items() if k != 'model'}
            and set(prior['model']) == set(final['model'])
            and all(torch.equal(v, prior['model'][k]) for k, v in final['model'].items()),
            'Completed teacher differs from fixed final checkpoint')
    else:
        torch_dump(path, final)
    return final


def run(raw, resume=False):
    cfg = normalize_config(raw)
    torch.set_num_threads(cfg['threads'])
    require(cfg['device'] != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    device = torch.device(cfg['device'])
    inner, _unused_calibration, features, factors, split, provenance = prepare_inputs(cfg)
    del _unused_calibration
    fresh = manifest_for(cfg, split, provenance, device)
    folder = Path(cfg['output_dir'])
    if folder.exists() and any(folder.iterdir()):
        require(resume, 'Output exists; resume the identical teacher or choose a fresh directory')
        manifest = ace.read(folder / 'manifest.json')
        require(manifest.get('fingerprint') == fingerprint(manifest) == fresh['fingerprint'],
            'Teacher source, runtime, data, split or configuration changed')
    else:
        folder.mkdir(parents=True, exist_ok=True)
        manifest = fresh
        json_dump(folder / 'manifest.json', manifest)
        json_dump(folder / 'split_manifest.json', split)
    require(ace.read(folder / 'split_manifest.json') == split, 'Stored inner split differs')
    seed_everything(cfg['seed'] + 1000)
    model = ACE2026Model(features, svd_factors=factors, **MODEL).float().to(device)
    fit_teacher(model, inner, manifest, folder, resume)
    result = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': cfg['seed'],
        'epochs': cfg['epochs'], 'teacher_sha256': file_digest(folder / 'teacher.pt'),
        'last_sha256': file_digest(folder / 'last.pt'), 'training_sha256': file_digest(folder / 'training.json')}
    if (folder / 'summary.json').exists():
        require(ace.read(folder / 'summary.json') == result, 'Completed teacher artifacts changed')
    else:
        json_dump(folder / 'summary.json', result)
    return result


def load_teacher(output_dir, device=None):
    """Return (frozen model, inner_train_records, calibration_records, manifest)."""
    folder = Path(output_dir).resolve()
    manifest = ace.read(folder / 'manifest.json')
    require(manifest.get('protocol') == PROTOCOL and manifest.get('fingerprint') == fingerprint(manifest)
        and manifest.get('source_hashes') == source_identity(), 'Teacher manifest or source changed')
    cfg = normalize_config(manifest['config'])
    require(cfg['output_dir'] == str(folder), 'Teacher output identity differs')
    inner, calibration, features, factors, split, provenance = prepare_inputs(cfg)
    require(split == manifest['split'] and provenance == manifest['inputs'], 'Teacher inputs or split changed')
    summary = ace.read(folder / 'summary.json')
    for key, filename in (('teacher_sha256', 'teacher.pt'), ('last_sha256', 'last.pt'), ('training_sha256', 'training.json')):
        require(file_digest(folder / filename) == summary[key], 'Teacher artifact changed: ' + filename)
    require(summary['fingerprint'] == manifest['fingerprint'] and summary['epochs'] == cfg['epochs'],
        'Teacher completed budget differs')
    target_device = torch.device(device or cfg['device'])
    model = ACE2026Model(features, svd_factors=factors, **manifest['model_config']).float().to(target_device)
    saved = torch.load(folder / 'teacher.pt', map_location='cpu', weights_only=True)
    require(all(saved.get(k) == v for k, v in _header(manifest, model).items())
        and saved['epoch'] == cfg['epochs'], 'Teacher checkpoint identity differs')
    model.load_state_dict(saved['model'])
    return model.requires_grad_(False).eval(), inner, calibration, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=135)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reference-run', default=DEFAULT_REFERENCE)
    parser.add_argument('--reference-seed', type=int, default=2025)
    parser.add_argument('--reference-variant', default='ace_selected')
    parser.add_argument('--svd-cache', default=DEFAULT_CACHE)
    parser.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args())
    resume = args.pop('resume')
    print(json.dumps(run(args, resume), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
