"""Validation-first, resumable experiments with immutable strong CF anchors.

The training entry point removes test labels before constructing any expert or
router. Test evaluation is available only from the separate replay entry point.
This is iterative benchmark development: historical test results were known.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import validation_runner as old
from .data import TrainingDataset
from .ml1m_model import PaperFAFIRecModel
from .ratio_runner import evaluate, write_rows
from .research_model import ResearchResidualModel
from .runner import (cpu_state, digest, file_digest, json_dump, resolve_device,
                     restore_rng, rng_state, runtime_identity, seed_everything,
                     tensor_digest, torch_dump)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ml1m_research_residual_v3'
DEFAULTS = dict(reference_main_run=None, anchor_run=None, output_dir=None,
    seeds=[2025, 2026, 2027, 2028, 2029], variants=[], epochs=150,
    min_epochs=30, patience=3, eval_interval=5, batch_size=2048,
    eval_batch=128, alpha_grid=[0., .5, 1., 2., 4.], routes=['global'],
    ce_chunk_size=1024, ridge=1., device='cpu', threads=4,
    weight_decay=1e-5, grad_clip=5., train_router=False, router_epochs=30,
    router_min_epochs=10, router_lr=.01, probe_negatives=4, probe_window=4,
    experiment_note='Historical test results were known; validation-only iterative benchmark development.')
VARIANT_DEFAULTS = dict(kind='text', objective='bpr', alignment_weight=0.,
                        auxiliary_weight=.1, lr=.001, score_scale=8.)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, positive=False):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and (value > 0 if positive else value >= 0),
             name + ' must be a finite ' + ('positive' if positive else 'nonnegative') + ' number')
    return float(value)


def normalize_config(supplied):
    _require(isinstance(supplied, dict) and not set(supplied) - set(DEFAULTS), 'Unknown research configuration keys')
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(supplied)
    for key in ('reference_main_run', 'anchor_run', 'output_dir'):
        _require(isinstance(cfg[key], str) and bool(cfg[key]), key + ' is required')
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    out = Path(cfg['output_dir'])
    for key in ('reference_main_run', 'anchor_run'):
        ref = Path(cfg[key])
        _require(out != ref and out not in ref.parents and ref not in out.parents,
                 'Output must not overlap immutable reference runs')
    _require(isinstance(cfg['seeds'], list) and cfg['seeds']
             and all(type(s) is int and s >= 0 for s in cfg['seeds'])
             and len(set(cfg['seeds'])) == len(cfg['seeds']), 'Seeds must be distinct nonnegative integers')
    for key in ('epochs', 'patience', 'eval_interval', 'batch_size', 'eval_batch',
                'ce_chunk_size', 'threads', 'router_epochs', 'probe_negatives', 'probe_window'):
        _require(type(cfg[key]) is int and cfg[key] >= 1, key + ' must be a positive integer')
    for key, limit in (('min_epochs', 'epochs'), ('router_min_epochs', 'router_epochs')):
        _require(type(cfg[key]) is int and 0 <= cfg[key] <= cfg[limit], key + ' exceeds its training budget')
    _require(cfg['device'] in {'cpu', 'cuda', 'auto'}, 'Unknown device')
    _require(type(cfg['train_router']) is bool, 'train_router must be boolean')
    _require(isinstance(cfg['routes'], list) and cfg['routes'] and len(set(cfg['routes'])) == len(cfg['routes'])
             and not set(cfg['routes']) - {'global', 'frequency'}, 'Expert routes must be distinct global/frequency policies')
    _require(not cfg['train_router'] or cfg['routes'] == ['global'],
             'Relative router must start from a global expert; frequency fusion is a separate control')
    _require(isinstance(cfg['alpha_grid'], list) and cfg['alpha_grid'], 'alpha_grid must be nonempty')
    cfg['alpha_grid'] = [_number(v, 'alpha') for v in cfg['alpha_grid']]
    _require(0. in cfg['alpha_grid'] and len(set(cfg['alpha_grid'])) == len(cfg['alpha_grid']),
             'alpha_grid must contain zero and distinct values')
    cfg['alpha_grid'] = sorted(cfg['alpha_grid'])
    for key in ('ridge', 'grad_clip', 'router_lr'):
        cfg[key] = _number(cfg[key], key, True)
    cfg['weight_decay'] = _number(cfg['weight_decay'], 'weight_decay')
    _require(isinstance(cfg['variants'], list) and cfg['variants'], 'variants must be nonempty')
    variants, names = [], set()
    for raw in cfg['variants']:
        _require(isinstance(raw, dict) and not set(raw) - ({'name'} | set(VARIANT_DEFAULTS)), 'Unknown variant keys')
        variant = {**VARIANT_DEFAULTS, **raw}
        name = variant.get('name')
        _require(isinstance(name, str) and bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name))
                 and name not in names, 'Variant names must be distinct safe directory names')
        names.add(name)
        _require(variant['kind'] in {'id', 'text', 'shuffled'} and variant['objective'] in {'bpr', 'ce'},
                 'Unknown expert kind or objective')
        for key in ('alignment_weight', 'auxiliary_weight', 'lr', 'score_scale'):
            variant[key] = _number(variant[key], key, key in {'lr', 'score_scale'})
        variants.append(variant)
    cfg['variants'] = variants
    return cfg


def load_config(path, overrides=None):
    value = _read(path)
    value.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return normalize_config(value)


def source_identity():
    paths = sorted((ROOT / 'fafirec').glob('*.py'))
    paths += [ROOT / name for name in ('research_ml1m.py', 'evaluate_research.py', 'analyze_research.py')]
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths if p.is_file()}


def load_anchors(cfg):
    """Verify the old validation-selected CF, without consuming its test scores."""
    main, anchor = Path(cfg['reference_main_run']), Path(cfg['anchor_run'])
    sm, mm = _read(anchor / 'manifest.json'), _read(main / 'manifest.json')
    _require(sm.get('fingerprint') == old._fingerprint(sm), 'Anchor search fingerprint differs')
    _require(Path(sm['search_config']['reference_run_dir']).resolve() == main,
             'Anchor search references another main run')
    _require(Path(sm['search_config']['output_dir']).resolve() == anchor, 'Anchor search output identity differs')
    # Existing shared modules must retain their exact old semantics. New modules
    # do not invalidate the immutable old checkpoint class.
    for name, checksum in mm['source']['file_hashes'].items():
        _require((ROOT / name).is_file() and file_digest(ROOT / name) == checksum,
                 'Shared reference source changed: ' + name)
    for name, checksum in sm['source_hashes'].items():
        _require((ROOT / name).is_file() and file_digest(ROOT / name) == checksum,
                 'Shared anchor source changed: ' + name)
    search = old.normalize_config(sm['search_config'])
    _require(not set(cfg['seeds']) - set(search['seeds']), 'Research seed absent from strong CF search')
    records, features, identity, base, reference, _ = old.load_reference(search)
    _require(reference == sm['reference'] and identity == mm['identity'] == sm['identity'],
             'Reference data or teacher/cache provenance changed')
    _require(base['ks'] == [10, 50] and base['selection_k'] == 50
             and base['candidate_policy'] == 'exclude_seen', 'Research requires the unchanged multi-target evaluation contract')
    comparison_path = main / 'analysis/strong_cf_comparison.json'
    comparison = _read(comparison_path)
    _require(comparison.get('complete') is True and comparison['selection']['selected'] == 'cf_lr1e3',
             'Strong CF comparison is not finalized or selected another anchor')
    lock_path = Path(comparison['source']['selection_lock_file'])
    _require(file_digest(lock_path) == comparison['selection']['lock_sha256'], 'Strong CF selection lock changed')
    lock = _read(lock_path)
    _require(lock.get('selected') == 'cf_lr1e3' and lock.get('selection_split') == 'validation'
             and lock.get('new_test_metrics_used_for_baseline_selection') is False
             and lock['main_fingerprint'] == mm['fingerprint']
             and lock['search_fingerprint'] == sm['fingerprint'], 'Strong CF validation choice differs')
    for filename, checksum in lock['artifact_sha256'].items():
        _require(file_digest(filename) == checksum, 'Frozen selection artifact changed: ' + filename)
    selection = _read(anchor / 'selection.json')
    _require(selection.get('selected_variant') == 'cf_lr1e3' and selection.get('split') == 'validation'
             and selection.get('fingerprint') == sm['fingerprint'], 'Anchor selection differs')
    variants = {v['name']: v for v in search['variants']}
    _require('cf_lr1e3' in variants and variants['cf_lr1e3']['control'] == 'cf', 'Missing strong CF candidate')
    variant = variants['cf_lr1e3']
    local = old._variant_config(sm['training_config'], variant)
    jobs = {row['seed']: row for row in comparison['per_seed']}
    checkpoints, provenance = {}, {}
    for seed in cfg['seeds']:
        path = anchor / f'seed_{seed}/cf_lr1e3/best.pt'
        checksum = file_digest(path)
        _require(seed in jobs and jobs[seed]['checkpoint_sha256'] == checksum, 'Selected anchor checksum differs')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        old._check_checkpoint(checkpoint, identity, sm['fingerprint'], old._artifact(sm, seed, variant),
                              old.model_config(local, 'cf', identity['items']))
        summary = _read(anchor / f'seed_{seed}/cf_lr1e3_summary.json')
        _require(summary['checkpoint_sha256'] == checksum and summary['fingerprint'] == sm['fingerprint'],
                 'Anchor summary checksum differs')
        checkpoints[seed] = checkpoint
        provenance[str(seed)] = {'checkpoint_sha256': checksum,
            'tensor_sha256': digest({k: tensor_digest(v) for k, v in checkpoint['model'].items()}),
            'selected_epoch': checkpoint['selected_epoch'], 'selected_validation_metric': checkpoint['selected_metric']}
    return records, features, identity, base, checkpoints, {
        'main_fingerprint': mm['fingerprint'], 'anchor_fingerprint': sm['fingerprint'],
        'main_manifest_sha256': file_digest(main / 'manifest.json'),
        'anchor_manifest_sha256': file_digest(anchor / 'manifest.json'),
        'selection_sha256': file_digest(anchor / 'selection.json'),
        'comparison_sha256': file_digest(comparison_path), 'lock_sha256': file_digest(lock_path),
        'old_reference_provenance': reference, 'anchors': provenance,
        'old_diagnostics_used_as_features': False}


def _fingerprint(manifest):
    return digest({k: manifest[k] for k in ('protocol', 'config', 'evaluation_config', 'identity',
                                          'source_hashes', 'runtime', 'reference', 'selection')})


def _evaluation_config(base, cfg):
    value = {k: copy.deepcopy(base[k]) for k in ('max_len', 'ks', 'selection_k', 'candidate_policy')}
    value.update({k: cfg[k] for k in ('eval_batch', 'device', 'threads')})
    return value


def _new_model(checkpoint, features, records, variant, cfg, seed, device):
    anchor = PaperFAFIRecModel(text_features=features, **checkpoint['model_config'])
    anchor.load_state_dict(checkpoint['model'])
    anchor.to(device).eval().requires_grad_(False)
    return ResearchResidualModel(anchor, features, records,
        **{k: variant[k] for k in ('kind', 'objective', 'alignment_weight', 'auxiliary_weight', 'score_scale')},
        **{k: cfg[k] for k in ('ridge', 'probe_negatives', 'probe_window', 'ce_chunk_size')}, seed=seed).to(device)


def _artifact(manifest, seed, variant, stage):
    return {'seed': seed, 'variant': variant['name'], 'stage': stage,
            'anchor': manifest['reference']['anchors'][str(seed)], 'variant_config': variant}


def _model_config(model):
    # Route is a validation calibration, not an architectural parameter.
    return {k: v for k, v in model.model_config().items() if k != 'route'}


def _check_checkpoint(checkpoint, manifest, artifact, model):
    _require(checkpoint.get('protocol') == PROTOCOL and checkpoint.get('fingerprint') == manifest['fingerprint']
             and checkpoint.get('artifact') == artifact and checkpoint.get('model_config') == _model_config(model),
             'Research checkpoint identity or configuration differs')
    stored_anchor = {k.removeprefix('anchor.'): tensor_digest(v)
                     for k, v in checkpoint['model'].items() if k.startswith('anchor.')}
    _require(digest(stored_anchor) == artifact['anchor']['tensor_sha256'],
             'Research checkpoint changed the frozen CF anchor')


@torch.no_grad()
def calibrate(model, records, diagnostics, eval_cfg, cfg, seed, stage):
    candidates = []
    for route in (['relative'] if stage == 'router' else cfg['routes']):
        for alpha in cfg['alpha_grid']:
            model.set_calibration(alpha, route)
            _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
            candidates.append({'alpha': alpha, 'route': route, 'validation': metrics})
    # Zero residual wins exact ties, followed by deterministic route order.
    best = max(candidates, key=lambda r: r['validation']['all'][f"ndcg{eval_cfg['selection_k']}"])
    model.set_calibration(best['alpha'], best['route'])
    return best, candidates


@torch.no_grad()
def router_statistics(model, records, cfg):
    model.eval()
    data = TrainingDataset(records, model.n_items, model.max_len, 0)
    total = torch.zeros(4, dtype=torch.float64)
    squares, count = total.clone(), 0
    device = next(model.parameters()).device
    for hist, _, _, _ in DataLoader(data, batch_size=min(128, cfg['batch_size']), shuffle=False, num_workers=0):
        raw = model.relative_features(hist.to(device)).detach().cpu().double()[hist.ne(0)]
        total += raw.sum(0)
        squares += raw.square().sum(0)
        count += len(raw)
    _require(count > 0, 'No training-only router feature observations')
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    model.set_router_statistics(mean, std)
    return {'mean': mean.tolist(), 'std': std.tolist(), 'positions': count,
            'scope': 'training histories only; no positive, validation or test target statistics'}


def fit(model, records, cfg, eval_cfg, seed, stage, out, manifest, artifact, resume=False):
    """Save sufficient state for deterministic resume at every epoch boundary."""
    _require(all('test' not in r for r in records), 'Training records must not expose test labels')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.set_stage('router' if stage == 'router' else 'expert')
    budget = cfg['router_epochs'] if stage == 'router' else cfg['epochs']
    min_epochs = cfg['router_min_epochs'] if stage == 'router' else cfg['min_epochs']
    lr = cfg['router_lr'] if stage == 'router' else artifact['variant_config']['lr']
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=cfg['weight_decay'])
    device = next(model.parameters()).device
    diagnostics = torch.zeros(len(records), 2)
    seen = torch.zeros(len(records), model.n_items + 1, dtype=torch.bool)
    for i, row in enumerate(records):
        seen[i, row['train']] = True
    best, state, selected, best_epoch, stale, logs, first_epoch, stopped = -math.inf, None, None, 0, 0, [], 1, False
    last_path = out / 'last.pt'
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        _check_checkpoint(saved, manifest, artifact, model)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, state, selected = saved['best_metric'], saved['best_model'], saved['calibration']
        best_epoch, stale, logs, first_epoch, stopped = saved['selected_epoch'], saved['stale'], saved['logs'], saved['epoch'] + 1, saved['stopped']
        restore_rng(saved['rng'])
    else:
        selected, grid = calibrate(model, records, diagnostics, eval_cfg, cfg, seed, stage)
        best = selected['validation']['all'][f"ndcg{eval_cfg['selection_k']}"]
        state = cpu_state(model)
        json_dump(out / 'initial_validation.json', {'selected': selected, 'grid': grid})
    started = time.monotonic()
    if not stopped:
        for epoch in range(first_epoch, budget + 1):
            model.set_calibration(1., 'relative' if stage == 'router' else 'global')
            model.train()
            data = TrainingDataset(records, model.n_items, model.max_len, seed + epoch)
            loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed + epoch))
            totals, count, steps = {k: 0. for k in ('loss', 'rec_loss', 'aux_loss', 'align_loss')}, 0, 0
            for hist, pos, neg, indices in loader:
                hist, pos, neg = hist.to(device), pos.to(device), neg.to(device)
                optimizer.zero_grad(set_to_none=True)
                parts = model.loss_batch(hist, pos, neg, seen_mask=seen[indices].to(device))
                if not torch.isfinite(parts['loss']):
                    raise RuntimeError('Nonfinite research loss')
                parts['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite research gradient')
                optimizer.step()
                amount = int(hist.ne(0).sum())
                for key in totals:
                    totals[key] += float(parts[key].detach()) * amount
                count, steps = count + amount, steps + 1
            log = {'epoch': epoch, 'optimizer_steps': steps, **{k: v / count for k, v in totals.items()}}
            if epoch % cfg['eval_interval'] == 0 or epoch == budget:
                chosen, grid = calibrate(model, records, diagnostics, eval_cfg, cfg, seed, stage)
                metric = chosen['validation']['all'][f"ndcg{eval_cfg['selection_k']}"]
                if metric > best:
                    best, state, selected, best_epoch, stale = metric, cpu_state(model), chosen, epoch, 0
                else:
                    stale += 1
                log.update(calibration=chosen, grid=grid)
                print(json.dumps({'seed': seed, 'variant': artifact['variant'], 'stage': stage,
                    'epoch': epoch, 'validation_ndcg50': metric, 'alpha': chosen['alpha'], 'route': chosen['route']}), flush=True)
                stopped = epoch >= min_epochs and stale >= cfg['patience']
            logs.append(log)
            payload = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'artifact': artifact,
                'model_config': _model_config(model), 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                'best_model': state, 'best_metric': best, 'calibration': selected,
                'selected_epoch': best_epoch, 'epoch': epoch, 'stale': stale, 'logs': logs,
                'stopped': stopped, 'rng': rng_state()}
            torch_dump(last_path, payload)
            json_dump(out / 'training.json', logs)
            if stopped:
                break
    _require(state is not None and selected is not None, 'No validation-selected research checkpoint')
    model.load_state_dict(state)
    model.set_calibration(selected['alpha'], selected['route'])
    torch_dump(out / 'best.pt', {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'],
        'artifact': artifact, 'model_config': _model_config(model), 'model': state,
        'selected_epoch': best_epoch, 'selected_metric': best, 'calibration': selected})
    return {'selected_epoch': best_epoch, 'selected_validation_metric': best,
            'epochs_completed': len(logs), 'optimizer_steps': sum(log['optimizer_steps'] for log in logs),
            'calibration': selected, 'elapsed_seconds_this_invocation': time.monotonic() - started}


def _verify_result(root, manifest, seed, variant, stage, model):
    folder = root / f"seed_{seed}/{variant['name']}/{stage}"
    result = _read(folder / 'summary.json')
    checkpoint = torch.load(folder / 'best.pt', weights_only=True, map_location='cpu')
    _check_checkpoint(checkpoint, manifest, _artifact(manifest, seed, variant, stage), model)
    _require(result['fingerprint'] == manifest['fingerprint'] and result['seed'] == seed
             and result['variant'] == variant['name'] and result['stage'] == stage
             and result['split'] == 'validation'
             and result['checkpoint_sha256'] == file_digest(folder / 'best.pt')
             and result['per_user_sha256'] == file_digest(folder / 'validation.csv')
             and result['calibration'] == checkpoint['calibration']
             and result['selected_epoch'] == checkpoint['selected_epoch']
             and result['selected_validation_metric'] == checkpoint['selected_metric']
             and result['validation'] == checkpoint['calibration']['validation']
             and checkpoint['selected_metric'] == checkpoint['calibration']['validation']['all']['ndcg50'],
             'Research selected artifacts changed')
    return result, checkpoint


def _aggregate(results, cfg):
    groups = {}
    for row in results:
        groups.setdefault((row['variant'], row['stage']), []).append(row)
    values = []
    for (variant, stage), jobs in sorted(groups.items()):
        _require({r['seed'] for r in jobs} == set(cfg['seeds']) and len(jobs) == len(cfg['seeds']),
                 'Complete configured seed grid required')
        metrics = {}
        for group in jobs[0]['validation']:
            metrics[group] = {}
            for metric in jobs[0]['validation'][group]:
                series = [r['validation'][group][metric] for r in jobs]
                if metric in {'users', 'relevant_items'}:
                    _require(len(set(series)) == 1, 'Evaluation population differs')
                    metrics[group][metric] = series[0]
                else:
                    metrics[group][metric] = {'mean': statistics.mean(series),
                        'sample_std': statistics.stdev(series) if len(series) > 1 else None}
        values.append({'variant': variant, 'stage': stage, 'seed_count': len(jobs), 'metrics': metrics})
    return sorted(values, key=lambda row: (-row['metrics']['all']['ndcg50']['mean'], row['variant'], row['stage']))


def run(cfg, resume=False):
    cfg = normalize_config(cfg)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(cfg['threads'])
    device = resolve_device(cfg)
    records, features, identity, base, anchors, provenance = load_anchors(cfg)
    training_records = [{k: v for k, v in r.items() if k != 'test'} for r in records]
    eval_cfg = _evaluation_config(base, cfg)
    fresh = {'protocol': PROTOCOL, 'config': cfg, 'evaluation_config': eval_cfg, 'identity': identity,
        'source_hashes': source_identity(), 'runtime': runtime_identity(device), 'reference': provenance,
        'selection': 'validation NDCG@50; per-seed alpha calibration on one common declared grid; epoch0 eligible; no test access',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'test_evaluated_by_training_entrypoint': False}
    fresh['fingerprint'] = _fingerprint(fresh)
    root = Path(cfg['output_dir'])
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError('Choose a new research output directory or use --resume')
        manifest = _read(root / 'manifest.json')
        _require(manifest.get('fingerprint') == _fingerprint(manifest) == fresh['fingerprint'],
                 'Cannot resume: source, configuration, runtime or reference changed')
    else:
        root.mkdir(parents=True, exist_ok=True)
        manifest = fresh
        json_dump(root / 'manifest.json', manifest)
    results = []
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            seed_everything(seed + 1000)
            model = _new_model(anchors[seed], features, training_records, variant, cfg, seed, device)
            for stage in (['expert', 'router'] if cfg['train_router'] else ['expert']):
                folder = root / f"seed_{seed}/{variant['name']}/{stage}"
                if resume and (folder / 'summary.json').exists():
                    result, checkpoint = _verify_result(root, manifest, seed, variant, stage, model)
                    model.load_state_dict(checkpoint['model'])
                    model.set_calibration(checkpoint['calibration']['alpha'], checkpoint['calibration']['route'])
                    results.append(result)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                stats = None
                if stage == 'router':
                    # A separate deterministic stage seed also handles a restart
                    # between completed expert fitting and initial router fitting.
                    seed_everything(seed + 2000)
                    model.set_stage('router')
                    stats = router_statistics(model, training_records, cfg)
                fitted = fit(model, training_records, cfg, eval_cfg, seed + 1000, stage, folder,
                             manifest, _artifact(manifest, seed, variant, stage), resume)
                rows, metrics = evaluate(model, training_records, 'validation', torch.zeros(len(records), 2), eval_cfg, seed)
                for row in rows:
                    row.update(variant=variant['name'], stage=stage)
                write_rows(folder / 'validation.csv', rows)
                result = {'seed': seed, 'variant': variant['name'], 'stage': stage, 'split': 'validation',
                    'control': model.control, 'fingerprint': manifest['fingerprint'], 'initialization': model.initialization_report,
                    'router_statistics': stats, **fitted, 'validation': metrics,
                    'checkpoint_sha256': file_digest(folder / 'best.pt'), 'per_user_sha256': file_digest(folder / 'validation.csv')}
                json_dump(folder / 'summary.json', result)
                results.append(result)
                json_dump(root / 'summary.json', results)
            del model
    candidates = _aggregate(results, cfg)
    selection = {'fingerprint': manifest['fingerprint'], 'split': 'validation',
        'selection': manifest['selection'], 'selected_variant': candidates[0]['variant'],
        'selected_stage': candidates[0]['stage'], 'candidates': candidates}
    json_dump(root / 'summary.json', results)
    json_dump(root / 'selection.json', selection)
    return selection


def load_selected(run_dir, seed, variant, stage='expert', device=None):
    root = Path(run_dir)
    manifest = _read(root / 'manifest.json')
    _require(manifest.get('protocol') == PROTOCOL and manifest.get('fingerprint') == _fingerprint(manifest),
             'Research manifest fingerprint or protocol differs')
    _require(source_identity() == manifest['source_hashes'], 'Current research source changed')
    cfg = normalize_config(manifest['config'])
    _require(seed in cfg['seeds'], 'Seed absent from research configuration')
    variants = {v['name']: v for v in cfg['variants']}
    _require(variant in variants and stage in (['expert', 'router'] if cfg['train_router'] else ['expert']),
             'Variant or stage absent from research configuration')
    records, features, identity, base, anchors, provenance = load_anchors(cfg)
    _require(identity == manifest['identity'] and provenance == manifest['reference'], 'Research anchor or input identity changed')
    eval_cfg = _evaluation_config(base, cfg)
    _require(eval_cfg == manifest['evaluation_config'], 'Evaluation convention changed')
    if device is not None:
        _require(device in {'cpu', 'cuda', 'auto'}, 'Unknown replay device')
        cfg['device'], eval_cfg['device'] = device, device
    torch.set_num_threads(cfg['threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_everything(seed + 1000)
    training_records = [{k: v for k, v in row.items() if k != 'test'} for row in records]
    model = _new_model(anchors[seed], features, training_records, variants[variant], cfg, seed, resolve_device(cfg))
    _, checkpoint = _verify_result(root, manifest, seed, variants[variant], stage, model)
    model.load_state_dict(checkpoint['model'])
    model.set_stage('router' if stage == 'router' else 'expert')
    model.set_calibration(checkpoint['calibration']['alpha'], checkpoint['calibration']['route'])
    model.eval()
    return model, records, torch.zeros(len(records), 2), eval_cfg, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    selected = run(load_config(args.config, {'output_dir': args.output_dir, 'device': args.device}), args.resume)
    print(json.dumps({'selected_variant': selected['selected_variant'], 'selected_stage': selected['selected_stage'],
                      'split': selected['split']}), flush=True)
