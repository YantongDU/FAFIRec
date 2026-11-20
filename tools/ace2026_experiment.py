"""Validation-first ACE experiments under the frozen FAFIRec ML-1M protocol.

ACE is initialized from item text alone, never from the collaborative reference.
The public ACE geometry and vanilla catalog CE are preserved. The training
layout is explicitly adapted to all causal shifted positions, rather than the
paper's one-prefix-per-example layout. Test is an explicit, five-seed-locked step.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader

from analyze_research import audit_rows
from analyze_transfer import _same
from baselines.ace2026_model import ACE2026Model, build_ace_svd
from fafirec import research_runner as reference
from fafirec.data import TrainingDataset
from fafirec.ratio_runner import evaluate, write_rows
from fafirec.runner import (cpu_state, digest, file_digest, json_dump, resolve_device,
    restore_rng, rng_state, runtime_identity, seed_everything, torch_dump)

PROTOCOL = 'ace2026_ml1m_shifted_catalog_v1'
DEFAULTS = dict(master_config=None, output_dir=None, seeds=[2025, 2026, 2027, 2028, 2029],
    variants=[], epochs=400, min_epochs=30, eval_interval=5, patience=3,
    batch_size=256, eval_batch=128, max_len=50, weight_decay=0., grad_clip=None,
    device='cuda', threads=4, d_model=128, n_layers=2, n_heads=2, inner_size=256,
    dropout=.5, attn_dropout=.3, layer_norm_eps=1e-12, initializer_range=.02,
    ce_chunk_size=1024, svd_cache=None)
VARIANT_DEFAULTS = dict(reg_weight=50., scale=.1, lr=.0005)
MODEL_KEYS = ('d_model', 'max_len', 'n_layers', 'n_heads', 'inner_size', 'dropout',
    'attn_dropout', 'layer_norm_eps', 'initializer_range', 'ce_chunk_size')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, positive=False):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0), 'Invalid ' + name)
    return float(value)


def normalize_config(raw):
    require(isinstance(raw, dict) and not set(raw) - set(DEFAULTS), 'Unknown ACE configuration keys')
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(raw)
    for name in ('master_config', 'output_dir', 'svd_cache'):
        if name == 'svd_cache' and cfg[name] is None:
            continue
        require(isinstance(cfg[name], str) and cfg[name], name + ' is required')
        path = Path(cfg[name])
        cfg[name] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    require(isinstance(cfg['seeds'], list) and cfg['seeds']
        and all(type(s) is int and s >= 0 for s in cfg['seeds'])
        and len(set(cfg['seeds'])) == len(cfg['seeds']), 'Distinct seeds required')
    for name in ('epochs', 'eval_interval', 'patience', 'batch_size', 'eval_batch', 'max_len',
                 'threads', 'd_model', 'n_layers', 'n_heads', 'inner_size', 'ce_chunk_size'):
        require(type(cfg[name]) is int and cfg[name] > 0, name + ' must be a positive integer')
    require(type(cfg['min_epochs']) is int and 0 <= cfg['min_epochs'] <= cfg['epochs'], 'Invalid min_epochs')
    require(cfg['d_model'] % cfg['n_heads'] == 0, 'Heads must divide model dimension')
    require(cfg['device'] in {'cpu', 'cuda', 'auto'}, 'Unknown device')
    for name in ('dropout', 'attn_dropout'):
        cfg[name] = number(cfg[name], name)
        require(cfg[name] < 1, name + ' must be below one')
    for name in ('layer_norm_eps', 'initializer_range'):
        cfg[name] = number(cfg[name], name, True)
    cfg['weight_decay'] = number(cfg['weight_decay'], 'weight_decay')
    if cfg['grad_clip'] is not None:
        cfg['grad_clip'] = number(cfg['grad_clip'], 'grad_clip', True)
    require(isinstance(cfg['variants'], list) and cfg['variants'], 'Variants required')
    variants, names = [], set()
    for raw_variant in cfg['variants']:
        require(isinstance(raw_variant, dict) and not set(raw_variant) - {'name', *VARIANT_DEFAULTS}, 'Unknown variant keys')
        variant = {**VARIANT_DEFAULTS, **raw_variant}
        name = variant.get('name')
        require(isinstance(name, str) and re.fullmatch('[A-Za-z][A-Za-z0-9_-]*', name)
                and name not in names, 'Distinct safe variant names required')
        names.add(name)
        for key in VARIANT_DEFAULTS:
            variant[key] = number(variant[key], key, key != 'reg_weight')
        variants.append(variant)
    cfg['variants'] = variants
    return cfg


def load_config(path):
    return normalize_config(read(path))


def master_config(path):
    """Resolve the reference config relative to its own project, not this worktree."""
    path = Path(path).resolve()
    raw = read(path)
    base = path.parent.parent if path.parent.name == 'configs' else path.parent
    for key in ('reference_main_run', 'anchor_run', 'output_dir'):
        if raw.get(key) and not Path(raw[key]).is_absolute():
            raw[key] = str((base / raw[key]).resolve())
    return reference.normalize_config(raw)


def source_identity():
    paths = sorted((ROOT / 'baselines').glob('*.py')) + [Path(__file__)]
    return {**reference.source_identity(), **{p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths}}


def inputs(cfg):
    master = master_config(cfg['master_config'])
    out = Path(cfg['output_dir'])
    for name in ('output_dir', 'reference_main_run', 'anchor_run'):
        other = Path(master[name])
        require(out != other and out not in other.parents and other not in out.parents, 'Output overlaps frozen reference')
    records, features, identity, base, _unused_anchors, provenance = reference.load_anchors({**master, 'seeds': cfg['seeds']})
    require(cfg['max_len'] == base['max_len'], 'History length differs from fixed comparison protocol')
    factors = (torch.load(cfg['svd_cache'], map_location='cpu', weights_only=True)
               if cfg['svd_cache'] else build_ace_svd(features))
    # ACE construction independently verifies canonical feature and factor hashes.
    factor_identity = {k: factors[k] for k in ('feature_sha256', 'feature_shape', 'decomposition', 'factor_sha256')}
    factor_identity['cache_sha256'] = file_digest(cfg['svd_cache']) if cfg['svd_cache'] else None
    return records, features, identity, base, provenance, factors, factor_identity


def fingerprint(manifest):
    return digest({k: v for k, v in manifest.items() if k not in {'fingerprint', 'created_utc'}})


def manifest_for(cfg, identity, base, provenance, factor_identity, device):
    eval_cfg = {k: base[k] for k in ('max_len', 'ks', 'selection_k', 'candidate_policy')}
    eval_cfg.update({k: cfg[k] for k in ('eval_batch', 'device', 'threads')})
    value = {'protocol': PROTOCOL, 'config': cfg, 'master_config_sha256': file_digest(cfg['master_config']),
        'evaluation_config': eval_cfg, 'identity': identity, 'reference': provenance,
        'svd': factor_identity, 'source_hashes': source_identity(), 'runtime': runtime_identity(device),
        'initialization': 'from static text SVD and random transformer only; CF reference weights never loaded',
        'training_layout': 'all nonpadding causal shifted positions from last 50 train transitions; adapted from official one-prefix examples',
        'objective': 'native vanilla full-catalog CE including PAD class and known train positives; raw dot products; no output normalization',
        'selection': 'validation all-user NDCG@50; epoch0 eligible; strict improvement; no training test evaluation',
        'historical_test_known': True, 'test_evaluated_by_training_entrypoint': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    value['fingerprint'] = fingerprint(value)
    return value


def new_model(features, factors, cfg, variant, device):
    return ACE2026Model(features, n_items=len(features) - 1, svd_factors=factors,
        **{k: cfg[k] for k in MODEL_KEYS}, **{k: variant[k] for k in ('reg_weight', 'scale')}).float().to(device)


def header(manifest, seed, variant, model):
    return {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed,
        'variant': variant['name'], 'variant_config': variant, 'model_config': model.model_config()}


def check_checkpoint(saved, expected):
    require(all(saved.get(k) == v for k, v in expected.items()), 'ACE checkpoint identity/configuration differs')


def fit(model, records, cfg, eval_cfg, seed, variant, folder, manifest, resume=False):
    require(all('test' not in r for r in records), 'Training records expose test labels')
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=variant['lr'], weight_decay=cfg['weight_decay'])
    expected = header(manifest, seed, variant, model)
    device = next(model.parameters()).device
    diagnostics = torch.zeros(len(records), 2)
    best, snapshot, chosen, stale, logs, first, finished = -math.inf, None, 0, 0, [], 1, False
    started = time.monotonic()
    if resume and (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        check_checkpoint(saved, expected)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, snapshot, chosen, stale = saved['best_metric'], saved['best_snapshot'], saved['selected_epoch'], saved['stale']
        logs, first, finished = saved['logs'], saved['epoch'] + 1, saved['finished']
        restore_rng(saved['rng'])
    else:
        _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
        best, snapshot = metrics['all']['ndcg50'], {'model': cpu_state(model), 'validation': metrics}
        json_dump(folder / 'initial_validation.json', metrics)
    if not finished:
        for epoch in range(first, cfg['epochs'] + 1):
            model.train()
            data = TrainingDataset(records, model.n_items, cfg['max_len'], seed + 1000 + epoch)
            loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(seed + 1000 + epoch))
            totals, positions, steps = {k: 0. for k in ('loss', 'rec_loss', 'reg_loss')}, 0, 0
            for hist, positive, _negative_unused, _indices in loader:
                optimizer.zero_grad(set_to_none=True)
                hist, positive = hist.to(device), positive.to(device)
                losses = model.loss_batch(hist, positive)
                require(all(torch.isfinite(losses[k]).item() for k in totals), 'Nonfinite ACE loss')
                losses['loss'].backward()
                require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Nonfinite ACE gradient')
                if cfg['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                amount = int((hist.ne(0) & positive.ne(0)).sum())
                for key in totals:
                    totals[key] += float(losses[key].detach()) * amount
                positions, steps = positions + amount, steps + 1
            log = {'epoch': epoch, 'optimizer_steps': steps, **{k: v / positions for k, v in totals.items()}}
            if epoch % cfg['eval_interval'] == 0 or epoch == cfg['epochs']:
                _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
                metric, log['validation'] = metrics['all']['ndcg50'], metrics
                if metric > best:
                    best, chosen, stale = metric, epoch, 0
                    snapshot = {'model': cpu_state(model), 'validation': metrics}
                else:
                    stale += 1
                print(json.dumps({'seed': seed, 'variant': variant['name'], 'epoch': epoch,
                                  'validation_ndcg50': metric}), flush=True)
            finished = epoch == cfg['epochs'] or (epoch >= cfg['min_epochs'] and stale >= cfg['patience'])
            logs.append(log)
            torch_dump(folder / 'last.pt', {**expected, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                'rng': rng_state(), 'best_metric': best, 'best_snapshot': snapshot, 'selected_epoch': chosen,
                'epoch': epoch, 'stale': stale, 'logs': logs, 'finished': finished})
            json_dump(folder / 'training.json', logs)
            if finished:
                break
    require(snapshot is not None, 'No selected ACE model')
    model.load_state_dict(snapshot['model'])
    torch_dump(folder / 'best.pt', {**expected, **snapshot, 'selected_epoch': chosen, 'selected_metric': best})
    return {'selected_epoch': chosen, 'selected_validation_metric': best, 'epochs_completed': len(logs),
        'optimizer_steps': sum(r['optimizer_steps'] for r in logs),
        'elapsed_seconds_this_invocation': time.monotonic() - started}


def verify_job(folder, manifest, seed, variant, model, n_users):
    expected = header(manifest, seed, variant, model)
    summary = read(folder / 'summary.json')
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
    check_checkpoint(best, expected)
    check_checkpoint(last, expected)
    require(all(summary.get(k) == expected[k] for k in expected), 'ACE summary identity differs')
    require(summary['checkpoint_sha256'] == file_digest(folder / 'best.pt')
        and summary['per_user_sha256'] == file_digest(folder / 'validation.csv'), 'ACE selected artifact changed')
    _same(best['validation'], summary['validation'], 'Selected validation metrics')
    require(best['selected_epoch'] == summary['selected_epoch'] == last['selected_epoch']
        and best['selected_metric'] == summary['selected_validation_metric'] == last['best_metric']
        == best['validation']['all']['ndcg50'], 'ACE epoch/metric selection differs')
    require(set(best['model']) == set(last['best_snapshot']['model'])
        and all(torch.equal(t, last['best_snapshot']['model'][k]) for k, t in best['model'].items()), 'Best/last ACE model differs')
    initial, cfg = read(folder / 'initial_validation.json'), manifest['config']
    metric, chosen, stale = initial['all']['ndcg50'], 0, 0
    logs = last['logs']
    require(logs and summary['epochs_completed'] == last['epoch'] == len(logs), 'ACE training budget differs')
    require(len(logs) <= cfg['epochs'], 'Training exceeded budget')
    for i, row in enumerate(logs, 1):
        require(row['epoch'] == i and row['optimizer_steps'] == math.ceil(n_users / cfg['batch_size']), 'Epoch/step layout differs')
        require(all(math.isfinite(row[k]) for k in ('loss', 'rec_loss', 'reg_loss')), 'Nonfinite logged loss')
        checked = i % cfg['eval_interval'] == 0 or i == cfg['epochs']
        require(('validation' in row) == checked, 'Validation look budget differs')
        if checked:
            value = row['validation']['all']['ndcg50']
            if value > metric:
                metric, chosen, stale = value, i, 0
            else:
                stale += 1
        stopped = i == cfg['epochs'] or (i >= cfg['min_epochs'] and stale >= cfg['patience'])
        require(not stopped or i == len(logs), 'Training continued after early stopping')
    require(last['finished'] is True and stopped and chosen == best['selected_epoch']
            and metric == best['selected_metric'] and stale == last['stale'], 'Selected ACE history differs')
    require(summary['optimizer_steps'] == sum(r['optimizer_steps'] for r in logs), 'Total optimizer steps differ')
    return summary, best


def run(config, resume=False):
    cfg = normalize_config(config)
    torch.set_num_threads(cfg['threads'])
    device = resolve_device(cfg)
    records, features, identity, base, provenance, factors, factor_identity = inputs(cfg)
    training_records = [{k: v for k, v in r.items() if k != 'test'} for r in records]
    fresh = manifest_for(cfg, identity, base, provenance, factor_identity, device)
    root = Path(cfg['output_dir'])
    if root.exists() and any(root.iterdir()):
        require(resume, 'Output exists; resume identical experiment or choose a new directory')
        manifest = read(root / 'manifest.json')
        require(manifest.get('fingerprint') == fingerprint(manifest) == fresh['fingerprint'], 'Cannot resume: ACE experiment identity changed')
    else:
        root.mkdir(parents=True, exist_ok=True)
        manifest = fresh
        json_dump(root / 'manifest.json', manifest)
    summaries = []
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            seed_everything(seed + 1000)
            model = new_model(features, factors, cfg, variant, device)
            folder = root / f"seed_{seed}/{variant['name']}"
            if resume and (folder / 'summary.json').exists():
                result, _ = verify_job(folder, manifest, seed, variant, model, len(records))
            else:
                result = fit(model, training_records, cfg, manifest['evaluation_config'], seed, variant, folder, manifest, resume)
                rows, metrics = evaluate(model, training_records, 'validation', torch.zeros(len(records), 2), manifest['evaluation_config'], seed)
                for row in rows:
                    row.update(variant=variant['name'], stage='single')
                write_rows(folder / 'validation.csv', rows)
                result = {**header(manifest, seed, variant, model), **result, 'split': 'validation',
                    'control': model.control, 'initialization': model.initialization_report,
                    'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'validation': metrics, 'checkpoint_sha256': file_digest(folder / 'best.pt'),
                    'per_user_sha256': file_digest(folder / 'validation.csv')}
                json_dump(folder / 'summary.json', result)
            summaries.append(result)
            json_dump(root / 'summary.json', summaries)
            del model
    candidates = [{'variant': v['name'], 'mean_validation_ndcg50': statistics.mean(
        row['validation']['all']['ndcg50'] for row in summaries if row['variant'] == v['name'])} for v in cfg['variants']]
    candidates.sort(key=lambda r: (-r['mean_validation_ndcg50'], r['variant']))
    selection = {'fingerprint': manifest['fingerprint'], 'split': 'validation',
        'selected_variant': candidates[0]['variant'], 'candidates': candidates}
    json_dump(root / 'selection.json', selection)
    return selection


def load_selected(run_dir, seed, variant_name):
    root = Path(run_dir).resolve()
    manifest = read(root / 'manifest.json')
    require(manifest.get('protocol') == PROTOCOL and manifest.get('fingerprint') == fingerprint(manifest), 'ACE manifest identity differs')
    cfg = normalize_config(manifest['config'])
    require(str(root) == cfg['output_dir'] and seed in cfg['seeds'], 'ACE seed/output differs')
    variants = {v['name']: v for v in cfg['variants']}
    require(variant_name in variants, 'Unknown ACE variant')
    torch.set_num_threads(cfg['threads'])
    records, features, identity, base, provenance, factors, fi = inputs(cfg)
    fresh = manifest_for(cfg, identity, base, provenance, fi, resolve_device(cfg))
    require(fresh['fingerprint'] == manifest['fingerprint'], 'ACE source/runtime/data/feature/reference differs')
    seed_everything(seed + 1000)
    model = new_model(features, factors, cfg, variants[variant_name], resolve_device(cfg))
    summary, checkpoint = verify_job(root / f'seed_{seed}/{variant_name}', manifest, seed, variants[variant_name], model, len(records))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, records, manifest['evaluation_config'], manifest, summary


def check_grid(root, manifest):
    cfg = manifest['config']
    expected = {(seed, variant['name']) for seed in cfg['seeds'] for variant in cfg['variants']}
    rows = read(root / 'summary.json')
    indexed = {(r['seed'], r['variant']): r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected, 'Complete ACE summary grid required')
    require({p.relative_to(root).as_posix() for p in root.glob('seed_*/*/summary.json')}
        == {f'seed_{s}/{v}/summary.json' for s, v in expected}, 'Extra or missing ACE job directories')
    for seed, variant in expected:
        _same(read(root / f'seed_{seed}/{variant}/summary.json'), indexed[seed, variant], 'ACE summary grid')
    selection = read(root / 'selection.json')
    candidates = [{'variant': v['name'], 'mean_validation_ndcg50': statistics.mean(
        indexed[s, v['name']]['validation']['all']['ndcg50'] for s in cfg['seeds'])} for v in cfg['variants']]
    candidates.sort(key=lambda r: (-r['mean_validation_ndcg50'], r['variant']))
    require(selection['fingerprint'] == manifest['fingerprint'] and selection['split'] == 'validation'
        and selection['selected_variant'] == candidates[0]['variant'], 'ACE common validation choice differs')
    _same(selection['candidates'], candidates, 'ACE candidate grid')


def _runs(paths):
    require(isinstance(paths, list) and paths and len(set(paths)) == len(paths), 'Explicit distinct replica run directories required')
    jobs, common, seeds = [], None, set()
    for raw in paths:
        root = Path(raw).resolve()
        manifest = read(root / 'manifest.json')
        cfg = normalize_config(manifest['config'])
        check_grid(root, manifest)
        require(len(cfg['variants']) == 1, 'Final lock requires one preselected common variant')
        normalized = {k: v for k, v in cfg.items() if k not in {'seeds', 'output_dir'}}
        comparison = {'config': normalized, **{k: manifest[k] for k in ('source_hashes', 'runtime', 'identity', 'evaluation_config', 'svd')}}
        comparison['reference'] = {k: v for k, v in manifest['reference'].items() if k != 'anchors'}
        require(common is None or common == comparison, 'Final replica common configuration/provenance differs')
        common = comparison
        for seed in cfg['seeds']:
            require(seed not in seeds, 'Duplicate final seed')
            seeds.add(seed)
            variant = cfg['variants'][0]['name']
            model, records, eval_cfg, checked, summary = load_selected(root, seed, variant)
            folder = root / f'seed_{seed}/{variant}'
            metrics = audit_rows(folder / 'validation.csv', records, eval_cfg, seed, variant, 'single', model.control, 'validation', checked['identity']['items'])
            _same(metrics, summary['validation'], 'Independent validation rankings')
            jobs.append({'run_dir': str(root), 'seed': seed, 'variant': variant, 'fingerprint': checked['fingerprint'],
                'selected_epoch': summary['selected_epoch'], 'validation': summary['validation']})
            del model
    require(len(seeds) == 5, 'Exactly five distinct final seeds required')
    return sorted(jobs, key=lambda j: j['seed']), common


def lock_selection(replica_runs, output):
    output = Path(output).resolve()
    require(not output.exists(), 'Selection lock is immutable')
    jobs, common = _runs(replica_runs)
    hashes = {}
    for job in jobs:
        root = Path(job['run_dir'])
        for path in [root / 'manifest.json', root / 'selection.json', root / 'summary.json'] + [root / f"seed_{job['seed']}/{job['variant']}/{name}" for name in
            ('best.pt', 'last.pt', 'summary.json', 'validation.csv', 'initial_validation.json')]:
            hashes[str(path)] = file_digest(path)
    value = {'protocol': PROTOCOL, 'selection_split': 'validation', 'replica_runs': replica_runs,
        'common': common, 'jobs': jobs, 'seed_ids': [j['seed'] for j in jobs], 'artifact_sha256': hashes,
        'source_hashes': source_identity(), 'historical_test_known': True, 'new_test_used_for_selection': False,
        'mean_validation_ndcg50': statistics.mean(j['validation']['all']['ndcg50'] for j in jobs)}
    value['lock_fingerprint'] = digest(value)
    json_dump(output, value)
    return value


def verify_lock(path, run_dir=None, seed=None, variant=None):
    lock = read(path)
    require(lock.get('protocol') == PROTOCOL and lock.get('selection_split') == 'validation'
        and lock.get('new_test_used_for_selection') is False and lock.get('source_hashes') == source_identity()
        and lock.get('lock_fingerprint') == digest({k: v for k, v in lock.items() if k != 'lock_fingerprint'}), 'ACE selection lock differs')
    require(len(lock['jobs']) == len(set(lock['seed_ids'])) == 5
            and {j['seed'] for j in lock['jobs']} == set(lock['seed_ids']), 'Complete five-seed lock required')
    expected = set()
    for job in lock['jobs']:
        root = Path(job['run_dir'])
        expected.update(str(root / name) for name in ('manifest.json', 'selection.json', 'summary.json'))
        expected.update(str(root / f"seed_{job['seed']}/{job['variant']}/{name}") for name in
            ('best.pt', 'last.pt', 'summary.json', 'validation.csv', 'initial_validation.json'))
        manifest = read(root / 'manifest.json')
        summary = read(root / f"seed_{job['seed']}/{job['variant']}/summary.json")
        require(manifest['fingerprint'] == fingerprint(manifest) == job['fingerprint']
            and manifest['source_hashes'] == lock['source_hashes']
            and summary['selected_epoch'] == job['selected_epoch']
            and summary['validation'] == job['validation'], 'ACE locked job selection differs')
    require(set(lock['artifact_sha256']) == expected, 'ACE lock omits selected validation artifacts')
    if run_dir is not None:
        require(any(j['seed'] == seed and j['variant'] == variant and j['run_dir'] == str(Path(run_dir).resolve()) for j in lock['jobs']), 'Test job absent from lock')
    for filename, checksum in lock['artifact_sha256'].items():
        require(file_digest(filename) == checksum, 'Locked ACE artifact changed: ' + filename)
    return lock


def replay(run_dir, seed, variant, split, output_dir, selection_lock=None):
    require(split in {'validation', 'test'}, 'Unknown split')
    if split == 'test':
        require(selection_lock is not None, 'Test requires explicit five-seed validation lock')
        verify_lock(selection_lock, run_dir, seed, variant)
    out = Path(output_dir).resolve()
    require(not out.exists(), 'Choose a new replay output directory')
    model, records, cfg, manifest, summary = load_selected(run_dir, seed, variant)
    if split == 'validation':
        records = [{k: v for k, v in r.items() if k != 'test'} for r in records]
    rows, metrics = evaluate(model, records, split, torch.zeros(len(records), 2), cfg, seed)
    for row in rows:
        row.update(variant=variant, stage='single')
    out.mkdir(parents=True)
    write_rows(out / 'per_user.csv', rows)
    value = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed, 'variant': variant,
        'split': split, 'metrics': metrics, 'checkpoint_sha256': summary['checkpoint_sha256'],
        'per_user_sha256': file_digest(out / 'per_user.csv'), 'candidate_policy': cfg['candidate_policy']}
    if split == 'validation':
        require(value['per_user_sha256'] == summary['per_user_sha256'], 'Independent ACE validation ranking differs')
        _same(metrics, summary['validation'], 'Independent ACE validation metrics')
        value['identical_to_original'] = True
    else:
        value['selection_lock_sha256'] = file_digest(selection_lock)
    recomputed = audit_rows(out / 'per_user.csv', records, cfg, seed, variant, 'single', model.control, split, manifest['identity']['items'])
    _same(recomputed, metrics, 'Replayed ranking metric recomputation')
    json_dump(out / 'metrics.json', value)
    return value


def aggregate(replica_runs, split, selection_lock, output, replay_root=None):
    output = Path(output).resolve()
    require(not output.exists(), 'Choose a new aggregate output')
    lock = verify_lock(selection_lock)
    require(replica_runs == lock['replica_runs'], 'Replica scope differs from lock')
    jobs, common = _runs(replica_runs)
    _same(jobs, lock['jobs'], 'Locked jobs')
    require(common == lock['common'], 'Locked common configuration changed')
    values, hashes = [], dict(lock['artifact_sha256'])
    for job in jobs:
        model, records, cfg, manifest, summary = load_selected(job['run_dir'], job['seed'], job['variant'])
        if split == 'validation':
            metrics = summary['validation']
        else:
            require(split == 'test' and replay_root is not None, 'Explicit test replay root required')
            folder = Path(replay_root).resolve() / f"seed_{job['seed']}"
            saved = read(folder / 'metrics.json')
            require(saved['protocol'] == PROTOCOL and saved['fingerprint'] == manifest['fingerprint']
                and saved['seed'] == job['seed'] and saved['variant'] == job['variant'] and saved['split'] == 'test'
                and saved['selection_lock_sha256'] == file_digest(selection_lock)
                and saved['checkpoint_sha256'] == summary['checkpoint_sha256']
                and saved['per_user_sha256'] == file_digest(folder / 'per_user.csv'), 'ACE test replay identity differs')
            metrics = audit_rows(folder / 'per_user.csv', records, cfg, job['seed'], job['variant'], 'single', model.control, 'test', manifest['identity']['items'])
            _same(metrics, saved['metrics'], 'ACE aggregate independent ranking metrics')
            hashes.update({str(p): file_digest(p) for p in (folder / 'per_user.csv', folder / 'metrics.json')})
        values.append({'seed': job['seed'], 'metrics': metrics, **{k: summary[k] for k in
            ('selected_epoch', 'epochs_completed', 'optimizer_steps', 'trainable_parameter_count')}})
        del model
    rows = []
    for group in values[0]['metrics']:
        for metric in ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50'):
            scores = [j['metrics'][group][metric] for j in values]
            rows.append({'variant': jobs[0]['variant'], 'stage': 'single', 'group': group, 'metric': metric,
                'seed_ids': lock['seed_ids'], 'seed_count': 5, 'per_seed_scores': scores,
                'mean': statistics.mean(scores), 'sample_std': statistics.stdev(scores)})
    result = {'protocol': PROTOCOL, 'split': split, 'seed_ids': lock['seed_ids'], 'aggregates': rows,
        'epoch_selections': [{k: v for k, v in j.items() if k != 'metrics'} for j in values],
        'selection_lock_sha256': file_digest(selection_lock), 'selection_lock_file': str(Path(selection_lock).resolve()),
        'selection_lock_fingerprint': lock['lock_fingerprint'], 'common': common, 'artifact_sha256': hashes,
        'all_artifact_hashes_verified': True, 'historical_test_known': True}
    json_dump(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--resume', action='store_true')
    for name in ('lock', 'aggregate'):
        command = sub.add_parser(name)
        command.add_argument('--replica-runs', type=Path, required=True, help='JSON list of explicit run directories')
        command.add_argument('--output', type=Path, required=True)
        if name == 'aggregate':
            command.add_argument('--split', choices=('validation', 'test'), required=True)
            command.add_argument('--selection-lock', type=Path, required=True)
            command.add_argument('--replay-root', type=Path)
    rep = sub.add_parser('replay')
    rep.add_argument('--run-dir', type=Path, required=True)
    rep.add_argument('--seed', type=int, required=True)
    rep.add_argument('--variant', required=True)
    rep.add_argument('--split', choices=('validation', 'test'), required=True)
    rep.add_argument('--output-dir', type=Path, required=True)
    rep.add_argument('--selection-lock', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'train':
        result = run(load_config(args.pop('config')), **args)
    else:
        if 'replica_runs' in args:
            args['replica_runs'] = read(args['replica_runs'])
        result = {'lock': lock_selection, 'replay': replay, 'aggregate': aggregate}[command](**args)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
