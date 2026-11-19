"""Single-branch CF continuation with the matched catalog-CE training recipe.

This is a baseline, not a FAFIRec variant. Training never evaluates test. A test
replay requires an explicit lock binding all five independently selected seeds.
Existing research source is imported and hashed without modification.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fafirec import research_runner as research
from fafirec.data import TrainingDataset
from fafirec.ml1m_model import PaperFAFIRecModel
from fafirec.ratio_runner import evaluate, write_rows
from fafirec.research_losses import masked_catalog_ce
from fafirec.runner import (cpu_state, digest, file_digest, json_dump, resolve_device,
                            restore_rng, rng_state, runtime_identity, seed_everything,
                            tensor_digest, torch_dump)

PROTOCOL = 'single_cf_catalog_ce_v1'
TRAINABLE_PREFIXES = ('item_id_table.', 'pos_emb.', 'input_norm.', 'blocks.', 'output_norm.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_identity():
    return {**research.source_identity(), 'tools/single_ce_baseline.py': file_digest(__file__)}


class SingleCEBaseline(nn.Module):
    """Independent copy of the selected CF; only its effective CF parameters train."""
    def __init__(self, anchor, score_scale=16., ce_chunk_size=1024):
        super().__init__()
        require(isinstance(anchor, PaperFAFIRecModel) and anchor.control == 'cf',
                'Single CE requires a CF PaperFAFIRecModel anchor')
        require(not isinstance(score_scale, bool) and isinstance(score_scale, (int, float))
                and math.isfinite(score_scale) and score_scale > 0, 'Score scale must be finite and positive')
        require(type(ce_chunk_size) is int and ce_chunk_size > 0, 'CE chunk size must be a positive integer')
        self.cf = copy.deepcopy(anchor)
        for name, parameter in self.cf.named_parameters():
            parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        self.n_items, self.d_model, self.max_len = anchor.n_items, anchor.d_model, anchor.max_len
        self.score_scale, self.ce_chunk_size = float(score_scale), int(ce_chunk_size)
        self.control = 'single_cf_ce'

    def user_vectors(self, hist_ids, diagnostics=None):
        return self.cf.user_vectors(hist_ids)

    def item_vectors(self, ids=None):
        return self.cf.item_vectors(ids)

    def loss_batch(self, hist_ids, pos_ids, seen_mask):
        hist_ids = self.cf._check_ids(hist_ids, 'hist_ids')
        pos_ids = self.cf._check_ids(pos_ids, 'pos_ids')
        require(hist_ids.shape == pos_ids.shape, 'Aligned training history and targets required')
        supervised = hist_ids.ne(0) & pos_ids.ne(0)
        states = self.cf._encode_tokens(self.cf.item_id_table(hist_ids), hist_ids.ne(0))
        owners = torch.arange(len(hist_ids), device=hist_ids.device).unsqueeze(1).expand_as(hist_ids)[supervised]
        return masked_catalog_ce(states[supervised], self.item_vectors(), pos_ids[supervised], owners,
                                 seen_mask, self.score_scale, self.ce_chunk_size)


def baseline_config(master):
    return {**{key: master[key] for key in ('epochs', 'min_epochs', 'patience', 'eval_interval',
            'batch_size', 'eval_batch', 'ce_chunk_size', 'threads', 'device', 'grad_clip')},
            'score_scale': 16., 'lr': .001, 'weight_decay': 1e-5,
            'optimizer': 'Adam', 'dtype': 'float32', 'initial_checkpoint_eligible': True,
            'training_queries': 'all non-PAD shifted positions in the last max_len train transitions',
            'negative_classes': 'full catalog minus full user train positives; current target restored; PAD excluded'}


def fingerprint(manifest):
    return digest({k: v for k, v in manifest.items() if k not in {'fingerprint', 'created_utc'}})


def _master(path):
    path = Path(path).resolve()
    return path, research.load_config(path)


def _check_output(out, seed, master):
    require(out.name == f'seed_{seed}', 'Output directory must end in seed_<seed>')
    for key in ('output_dir', 'reference_main_run', 'anchor_run'):
        reference = Path(master[key]).resolve()
        require(out != reference and out not in reference.parents and reference not in out.parents,
                'Single baseline output overlaps an immutable or active research run')


def _inputs(master, seed):
    require(type(seed) is int and seed in master['seeds'], 'Seed absent from master configuration')
    return research.load_anchors({**master, 'seeds': [seed]})


def _new_model(checkpoint, features, cfg, device):
    anchor = PaperFAFIRecModel(text_features=features, **checkpoint['model_config'])
    anchor.load_state_dict(checkpoint['model'])
    return SingleCEBaseline(anchor, cfg['score_scale'], cfg['ce_chunk_size']).float().to(device)


def _manifest(master_path, master, out, seed, identity, base, reference, checkpoint, device):
    cfg = baseline_config(master)
    eval_cfg = research._evaluation_config(base, master)
    value = {'protocol': PROTOCOL, 'seed': seed, 'output_dir': str(out),
        'master_config_file': str(master_path), 'master_config_sha256': file_digest(master_path),
        'master_config': master, 'training_config': cfg, 'evaluation_config': eval_cfg,
        'model_config': {'class': 'SingleCEBaseline', 'cf': checkpoint['model_config'],
                         'score_scale': cfg['score_scale'], 'ce_chunk_size': cfg['ce_chunk_size'],
                         'trainable_prefixes': list(TRAINABLE_PREFIXES)},
        'identity': identity, 'reference': reference, 'source_hashes': source_identity(),
        'runtime': runtime_identity(device),
        'anchor_tensor_sha256': digest({k: tensor_digest(v) for k, v in checkpoint['model'].items()}),
        'selection': 'per-seed validation NDCG@50 only; epoch0 eligible; no cross-seed model search',
        'test_evaluated_by_training_entrypoint': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    value['fingerprint'] = fingerprint(value)
    return value


def _guard_checkpoint(saved, manifest):
    require(saved.get('protocol') == PROTOCOL and saved.get('fingerprint') == manifest['fingerprint']
            and saved.get('seed') == manifest['seed'] and saved.get('model_config') == manifest['model_config']
            and saved.get('anchor_tensor_sha256') == manifest['anchor_tensor_sha256']
            and saved.get('optimizer_name') == 'Adam', 'Single CE checkpoint identity differs')


def _header(manifest):
    return {key: manifest[key] for key in ('protocol', 'fingerprint', 'seed', 'model_config', 'anchor_tensor_sha256')}


def _fit(model, records, manifest, out, resume=False):
    require(all('test' not in row for row in records), 'Training records expose test labels')
    cfg, eval_cfg, seed = manifest['training_config'], manifest['evaluation_config'], manifest['seed']
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    seen = torch.zeros(len(records), model.n_items + 1, dtype=torch.bool)
    for index, record in enumerate(records):
        seen[index, record['train']] = True
    diagnostics = torch.zeros(len(records), 2)
    best, best_snapshot, best_epoch, stale, logs, first, finished = -math.inf, None, 0, 0, [], 1, False
    if resume and (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        _guard_checkpoint(saved, manifest)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, best_snapshot, best_epoch = saved['best_metric'], saved['best_snapshot'], saved['selected_epoch']
        stale, logs, first, finished = saved['stale'], saved['logs'], saved['epoch'] + 1, saved['finished']
        restore_rng(saved['rng'])
    else:
        _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
        best = metrics['all']['ndcg50']
        best_snapshot = {'model': cpu_state(model), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                         'rng': rng_state(), 'validation': metrics}
        json_dump(out / 'initial_validation.json', metrics)
    if not finished:
        for epoch in range(first, cfg['epochs'] + 1):
            model.train()
            data = TrainingDataset(records, model.n_items, model.max_len, seed + 1000 + epoch)
            loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed + 1000 + epoch))
            total, positions, steps = 0., 0, 0
            for hist, positive, _, indices in loader:
                hist, positive = hist.to(device), positive.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss_batch(hist, positive, seen[indices].to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite single CE loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, cfg['grad_clip'])
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite single CE gradient')
                optimizer.step()
                count = int((hist.ne(0) & positive.ne(0)).sum())
                total, positions, steps = total + float(loss.detach()) * count, positions + count, steps + 1
            log = {'epoch': epoch, 'loss': total / positions, 'optimizer_steps': steps}
            if epoch % cfg['eval_interval'] == 0 or epoch == cfg['epochs']:
                _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
                log['validation'] = metrics
                metric = metrics['all']['ndcg50']
                if metric > best:
                    best, best_epoch, stale = metric, epoch, 0
                    best_snapshot = {'model': cpu_state(model), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                                     'rng': rng_state(), 'validation': metrics}
                else:
                    stale += 1
                print(json.dumps({'seed': seed, 'baseline': model.control, 'epoch': epoch,
                                  'validation_ndcg50': metric}), flush=True)
            finished = epoch == cfg['epochs'] or (epoch >= cfg['min_epochs'] and stale >= cfg['patience'])
            logs.append(log)
            torch_dump(out / 'last.pt', {**_header(manifest), 'optimizer_name': 'Adam',
                'model': cpu_state(model), 'optimizer': optimizer.state_dict(), 'rng': rng_state(),
                'best_snapshot': best_snapshot, 'best_metric': best, 'selected_epoch': best_epoch,
                'epoch': epoch, 'stale': stale, 'logs': logs, 'finished': finished})
            json_dump(out / 'training.json', logs)
            if finished:
                break
    require(best_snapshot is not None, 'No validation-selected single CE checkpoint')
    model.load_state_dict(best_snapshot['model'])
    torch_dump(out / 'best.pt', {**_header(manifest), 'optimizer_name': 'Adam', **best_snapshot,
                               'selected_epoch': best_epoch, 'selected_metric': best})
    return {'selected_epoch': best_epoch, 'selected_validation_metric': best,
            'epochs_completed': len(logs), 'optimizer_steps': sum(log['optimizer_steps'] for log in logs)}


def _verify_selected(out, manifest):
    summary = read(out / 'summary.json')
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    _guard_checkpoint(checkpoint, manifest)
    require(summary.get('fingerprint') == manifest['fingerprint'] and summary.get('seed') == manifest['seed']
            and summary.get('split') == 'validation' and summary.get('protocol') == PROTOCOL
            and summary.get('checkpoint_sha256') == file_digest(out / 'best.pt')
            and summary.get('per_user_sha256') == file_digest(out / 'validation.csv')
            and summary.get('validation') == checkpoint['validation']
            and summary.get('selected_epoch') == checkpoint['selected_epoch']
            and summary.get('selected_validation_metric') == checkpoint['selected_metric']
            and checkpoint['selected_metric'] == checkpoint['validation']['all']['ndcg50'],
            'Single CE selected artifacts changed')
    prefixes = tuple('cf.' + prefix for prefix in TRAINABLE_PREFIXES)
    require(summary['active_trainable_parameter_count'] == sum(v.numel() for k, v in checkpoint['model'].items()
                                                              if k.startswith(prefixes)),
            'Single CE trainable parameter count differs')
    _audit_training(out, manifest, summary, checkpoint)
    return summary, checkpoint


def _audit_training(out, manifest, summary, checkpoint):
    """Check the epoch/selection budget independently of summary claims."""
    cfg = manifest['training_config']
    last = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
    _guard_checkpoint(last, manifest)
    logs = last['logs']
    completed = len(logs)
    require(0 < completed <= cfg['epochs'] and last['epoch'] == completed and last['finished'] is True
            and completed == summary['epochs_completed'], 'Single CE completed epoch budget differs')
    initial = read(out / 'initial_validation.json')
    best, chosen, stale = initial['all']['ndcg50'], 0, 0
    expected_steps = math.ceil(len(manifest['identity']['user_order']) / cfg['batch_size'])
    checks = 1
    for epoch, log in enumerate(logs, 1):
        require(log['epoch'] == epoch and log['optimizer_steps'] == expected_steps
                and isinstance(log['loss'], (int, float)) and math.isfinite(log['loss']) and log['loss'] >= 0,
                'Single CE optimizer/log history differs')
        validating = epoch % cfg['eval_interval'] == 0 or epoch == cfg['epochs']
        require(('validation' in log) == validating, 'Single CE validation schedule differs')
        if validating:
            checks += 1
            metric = log['validation']['all']['ndcg50']
            require(isinstance(metric, (int, float)) and math.isfinite(metric) and 0 <= metric <= 1,
                    'Invalid single CE validation score')
            if metric > best:
                best, chosen, stale = metric, epoch, 0
            else:
                stale += 1
        stopped = epoch == cfg['epochs'] or (epoch >= cfg['min_epochs'] and stale >= cfg['patience'])
        require(stopped == (epoch == completed), 'Single CE early-stop history differs')
    require(chosen == summary['selected_epoch'] == last['selected_epoch']
            and best == summary['selected_validation_metric'] == last['best_metric']
            and stale == last['stale']
            and summary['optimizer_steps'] == completed * expected_steps,
            'Single CE validation-best selection differs')
    require(last['best_snapshot']['validation'] == checkpoint['validation']
            and set(last['best_snapshot']['model']) == set(checkpoint['model'])
            and all(torch.equal(last['best_snapshot']['model'][k], v) for k, v in checkpoint['model'].items()),
            'Single CE best snapshot differs from atomic training state')
    # A crash may happen after the authoritative atomic checkpoint and before
    # the readable JSON log is replaced. Its existing prefix must still agree.
    readable = read(out / 'training.json') if (out / 'training.json').exists() else []
    require(len(readable) <= len(logs) and readable == logs[:len(readable)], 'Single CE readable history changed')
    return {'validation_checks_including_epoch0': checks,
            'training_json_lags_atomic_checkpoint': len(readable) < len(logs)}


def run(master_config, seed, output_dir, resume=False):
    master_path, master = _master(master_config)
    out = Path(output_dir).resolve()
    _check_output(out, seed, master)
    cfg = baseline_config(master)
    torch.set_num_threads(cfg['threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = resolve_device(cfg)
    records, features, identity, base, anchors, reference = _inputs(master, seed)
    training_records = [{key: value for key, value in row.items() if key != 'test'} for row in records]
    del records
    fresh = _manifest(master_path, master, out, seed, identity, base, reference, anchors[seed], device)
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError('Choose a new baseline output directory or resume identical settings')
        manifest = read(out / 'manifest.json')
        require(manifest.get('fingerprint') == fingerprint(manifest) == fresh['fingerprint'],
                'Cannot resume: single CE data/source/runtime/configuration/reference changed')
        if (out / 'summary.json').exists():
            return _verify_selected(out, manifest)[0]
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest = fresh
        json_dump(out / 'manifest.json', manifest)
    seed_everything(seed + 1000)
    model = _new_model(anchors[seed], features, cfg, device)
    fitted = _fit(model, training_records, manifest, out, resume)
    rows, metrics = evaluate(model, training_records, 'validation', torch.zeros(len(training_records), 2),
                             manifest['evaluation_config'], seed)
    for row in rows:
        row.update(variant='single_cf_ce', stage='single')
    write_rows(out / 'validation.csv', rows)
    summary = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed,
        'control': model.control, 'split': 'validation', **fitted, 'validation': metrics,
        'active_trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'checkpoint_sha256': file_digest(out / 'best.pt'), 'per_user_sha256': file_digest(out / 'validation.csv')}
    json_dump(out / 'summary.json', summary)
    return summary


def _load_run(run_dir, seed):
    out = Path(run_dir).resolve()
    manifest = read(out / 'manifest.json')
    require(manifest.get('protocol') == PROTOCOL and manifest.get('seed') == seed
            and manifest.get('output_dir') == str(out) and manifest.get('fingerprint') == fingerprint(manifest),
            'Single CE manifest identity differs')
    master_path, master = _master(manifest['master_config_file'])
    require(file_digest(master_path) == manifest['master_config_sha256'] and master == manifest['master_config'],
            'Single CE master configuration changed')
    _check_output(out, seed, master)
    cfg = baseline_config(master)
    torch.set_num_threads(cfg['threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = resolve_device(cfg)
    records, features, identity, base, anchors, reference = _inputs(master, seed)
    fresh = _manifest(master_path, master, out, seed, identity, base, reference, anchors[seed], device)
    require(fresh['fingerprint'] == manifest['fingerprint'],
            'Single CE source, runtime, input or anchor identity changed')
    summary, checkpoint = _verify_selected(out, manifest)
    for name, value in anchors[seed]['model'].items():
        if not name.startswith(TRAINABLE_PREFIXES):
            require(torch.equal(value, checkpoint['model']['cf.' + name]),
                    'Single CE changed a frozen, unused CF parameter')
    return records, features, cfg, manifest, summary, checkpoint, anchors[seed]


def load_selected(run_dir, seed):
    records, features, cfg, manifest, summary, checkpoint, anchor = _load_run(run_dir, seed)
    model = _new_model(anchor, features, cfg, resolve_device(cfg))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, records, manifest['evaluation_config'], manifest, summary


def lock_selection(master_config, replica_root, output):
    master_path, master = _master(master_config)
    seeds = sorted(master['seeds'])
    require(len(seeds) == 5, 'Test selection lock requires all five configured seeds')
    root, output = Path(replica_root).resolve(), Path(output).resolve()
    require({p.name for p in root.glob('seed_*') if p.is_dir()} == {f'seed_{s}' for s in seeds},
            'Missing or extra single CE seed directories')
    if output.exists():
        raise FileExistsError('Selection lock is immutable; choose a new path')
    artifacts, jobs, shared_runtime, shared_identity = {}, [], None, None
    for seed in seeds:
        folder = root / f'seed_{seed}'
        records, _, _, manifest, summary, _, _ = _load_run(folder, seed)
        require(manifest['master_config_file'] == str(master_path), 'Baseline seed uses another master configuration')
        if shared_runtime is None:
            shared_runtime = manifest['runtime']
            shared_identity = manifest['identity']
        require(manifest['runtime'] == shared_runtime and manifest['identity'] == shared_identity,
                'Baseline seed runtimes or data identities differ')
        from analyze_research import audit_rows
        metrics = audit_rows(folder / 'validation.csv', records, manifest['evaluation_config'], seed,
                             'single_cf_ce', 'single', 'single_cf_ce', 'validation', manifest['identity']['items'])
        require(metrics == summary['validation'], 'Single CE lock validation rankings differ')
        for name in ('manifest.json', 'best.pt', 'last.pt', 'summary.json', 'validation.csv', 'initial_validation.json'):
            path = folder / name
            artifacts[str(path)] = file_digest(path)
        jobs.append({'seed': seed, 'run_fingerprint': manifest['fingerprint'],
                     'selected_epoch': summary['selected_epoch'], 'validation': summary['validation']})
    value = {'protocol': PROTOCOL, 'selection_split': 'validation', 'seed_ids': seeds,
        'master_config_file': str(master_path), 'master_config_sha256': file_digest(master_path),
        'master_config': master, 'replica_root': str(root), 'source_hashes': source_identity(),
        'runtime': shared_runtime, 'validation_artifact_sha256': artifacts, 'jobs': jobs,
        'mean_validation_ndcg50': statistics.mean(row['validation']['all']['ndcg50'] for row in jobs),
        'new_test_used_for_selection': False, 'historical_test_known': True,
        'selection': 'independent single-branch CE baseline; each seed selects its own validation-best epoch'}
    value['lock_fingerprint'] = digest(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(output, value)
    return value


def verify_lock(path, run_dir, seed):
    value = read(path)
    require(value.get('protocol') == PROTOCOL and value.get('selection_split') == 'validation'
            and value.get('new_test_used_for_selection') is False
            and value.get('historical_test_known') is True
            and value.get('lock_fingerprint') == digest({k: v for k, v in value.items() if k != 'lock_fingerprint'}),
            'Single CE validation lock identity differs')
    master_path, master = _master(value['master_config_file'])
    require(sorted(master['seeds']) == value['seed_ids'] and len(value['seed_ids']) == 5
            and seed in value['seed_ids'] and master == value['master_config']
            and file_digest(master_path) == value['master_config_sha256']
            and source_identity() == value['source_hashes'], 'Single CE five-seed lock source/configuration changed')
    root = Path(value['replica_root'])
    require(Path(run_dir).resolve() == (root / f'seed_{seed}').resolve(), 'Lock refers to another seed directory')
    require({p.name for p in root.glob('seed_*') if p.is_dir()} == {f'seed_{s}' for s in value['seed_ids']},
            'Missing or extra locked single CE seed directories')
    expected = {str(root / f'seed_{s}' / name) for s in value['seed_ids']
                for name in ('manifest.json', 'best.pt', 'last.pt', 'summary.json', 'validation.csv', 'initial_validation.json')}
    require(set(value['validation_artifact_sha256']) == expected, 'Single CE lock omits validation artifacts')
    for filename, checksum in value['validation_artifact_sha256'].items():
        require(file_digest(filename) == checksum, 'Locked single CE validation artifact changed')
    require(len(value['jobs']) == 5 and {j['seed'] for j in value['jobs']} == set(value['seed_ids']),
            'Locked single CE job coverage differs')
    jobs = {j['seed']: j for j in value['jobs']}
    for s in value['seed_ids']:
        manifest = read(root / f'seed_{s}/manifest.json')
        summary = read(root / f'seed_{s}/summary.json')
        require(manifest['seed'] == s and manifest['fingerprint'] == fingerprint(manifest)
                and manifest['runtime'] == value['runtime'] and manifest['source_hashes'] == value['source_hashes'],
                'Locked single CE runtime or source differs')
        require(jobs[s]['run_fingerprint'] == manifest['fingerprint']
                and jobs[s]['selected_epoch'] == summary['selected_epoch']
                and jobs[s]['validation'] == summary['validation'], 'Locked single CE selection metadata differs')
    require(value['mean_validation_ndcg50'] == statistics.mean(j['validation']['all']['ndcg50'] for j in value['jobs']),
            'Locked single CE aggregate selection differs')
    return value


def replay(run_dir, seed, split, output_dir, selection_lock=None):
    require(split in {'validation', 'test'}, 'Replay split must be validation or test')
    if split == 'test':
        require(selection_lock is not None, 'Test replay requires an explicit all-five-seed --selection-lock')
        verify_lock(selection_lock, run_dir, seed)
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new replay output directory')
    model, records, cfg, manifest, summary = load_selected(run_dir, seed)
    if split == 'validation':
        records = [{key: value for key, value in row.items() if key != 'test'} for row in records]
    rows, metrics = evaluate(model, records, split, torch.zeros(len(records), 2), cfg, seed)
    for row in rows:
        row.update(variant='single_cf_ce', stage='single')
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / 'per_user.csv', rows)
    result = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed,
        'split': split, 'control': model.control, 'metrics': metrics,
        'per_user_sha256': file_digest(out / 'per_user.csv'), 'checkpoint_sha256': summary['checkpoint_sha256'],
        'candidate_policy': cfg['candidate_policy'], 'target_mode': 'multi_positive_block'}
    if split == 'validation':
        require(result['per_user_sha256'] == summary['per_user_sha256'] and metrics == summary['validation'],
                'Independent single CE validation replay differs')
        result['identical_to_original'] = True
    else:
        result.update(selection_lock_file=str(Path(selection_lock).resolve()),
                      selection_lock_sha256=file_digest(selection_lock))
    json_dump(out / 'metrics.json', result)
    return result


def aggregate(replica_root, split, output, selection_lock, replay_root=None):
    """CPU-only audit of all saved rankings; never evaluates a model."""
    from analyze_research import audit_rows
    from analyze_transfer import _same
    root, output = Path(replica_root).resolve(), Path(output).resolve()
    require(split in {'validation', 'test'}, 'Unknown aggregate split')
    require(split != 'test' or replay_root is not None, 'Test aggregate requires --replay-root')
    if output.exists():
        raise FileExistsError('Choose a new baseline aggregate output file')
    lock = read(selection_lock)
    seeds = lock['seed_ids']
    require(len(seeds) == 5, 'Baseline aggregate requires all five seeds')
    verify_lock(selection_lock, root / f'seed_{seeds[0]}', seeds[0])
    jobs, hashes, common, anchors = [], {}, None, {}
    for seed in seeds:
        folder = root / f'seed_{seed}'
        records, _, cfg, manifest, summary, _, _ = _load_run(folder, seed)
        current = {key: manifest[key] for key in ('source_hashes', 'runtime', 'identity', 'evaluation_config')}
        current['reference'] = {key: value for key, value in manifest['reference'].items() if key != 'anchors'}
        if common is None:
            common = current
        require(current == common, 'Single CE replica provenance differs')
        anchors.update(manifest['reference']['anchors'])
        if split == 'validation':
            path, expected = folder / 'validation.csv', summary['validation']
        else:
            destination = Path(replay_root).resolve() / f'seed_{seed}'
            path, report_path = destination / 'per_user.csv', destination / 'metrics.json'
            saved = read(report_path)
            require(saved.get('protocol') == PROTOCOL and saved.get('fingerprint') == manifest['fingerprint']
                    and saved.get('split') == 'test' and saved.get('seed') == seed
                    and saved.get('control') == 'single_cf_ce'
                    and saved.get('candidate_policy') == manifest['evaluation_config']['candidate_policy']
                    and saved.get('target_mode') == 'multi_positive_block'
                    and saved.get('per_user_sha256') == file_digest(path)
                    and saved.get('checkpoint_sha256') == summary['checkpoint_sha256']
                    and saved.get('selection_lock_sha256') == file_digest(selection_lock),
                    'Single CE test replay assets changed')
            hashes[str(report_path)] = file_digest(report_path)
            expected = saved['metrics']
        metrics = audit_rows(path, records, manifest['evaluation_config'], seed, 'single_cf_ce',
                             'single', 'single_cf_ce', split, manifest['identity']['items'])
        _same(metrics, expected, 'Independently audited single CE metrics')
        hashes[str(path)] = file_digest(path)
        jobs.append({'seed': seed, 'metrics': metrics, **{key: summary[key] for key in
                     ('selected_epoch', 'epochs_completed', 'optimizer_steps', 'checkpoint_sha256',
                      'active_trainable_parameter_count')},
                     'validation_ndcg50': summary['validation']['all']['ndcg50']})
    rows = []
    for group in jobs[0]['metrics']:
        for metric in ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50'):
            values = [row['metrics'][group][metric] for row in jobs]
            rows.append({'variant': 'single_cf_ce', 'stage': 'single', 'group': group, 'metric': metric,
                         'seed_ids': seeds, 'seed_count': 5, 'per_seed_scores': values,
                         'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)})
    hashes[str(Path(selection_lock).resolve())] = file_digest(selection_lock)
    for relative, checksum in source_identity().items():
        hashes[str(ROOT / relative)] = checksum
    artifacts = {**lock['validation_artifact_sha256'], **hashes}
    for path, checksum in artifacts.items():
        require(file_digest(path) == checksum, 'Single CE artifact changed during aggregation')
    value = {'protocol': PROTOCOL, 'split': split, 'seed_ids': seeds, 'aggregates': rows,
        'paired_comparisons': [], 'selection_lock_file': str(Path(selection_lock).resolve()),
        'selection_lock_sha256': file_digest(selection_lock), 'selection_lock_fingerprint': lock['lock_fingerprint'],
        **common, 'reference': {**common['reference'], 'anchors': anchors},
        'recipe': baseline_config(lock['master_config']),
        'epoch_selections': [{key: value for key, value in row.items() if key != 'metrics'} for row in jobs],
        'artifact_sha256': artifacts,
        'role': 'independently epoch-selected single-branch CE baseline; excluded from the original seven-row method grid',
        'all_artifact_hashes_verified': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--master-config', type=Path, required=True)
    train.add_argument('--seed', type=int, required=True)
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--resume', action='store_true')
    lock = commands.add_parser('lock')
    lock.add_argument('--master-config', type=Path, required=True)
    lock.add_argument('--replica-root', type=Path, required=True)
    lock.add_argument('--output', type=Path, required=True)
    rep = commands.add_parser('replay')
    rep.add_argument('--run-dir', type=Path, required=True)
    rep.add_argument('--seed', type=int, required=True)
    rep.add_argument('--split', choices=('validation', 'test'), required=True)
    rep.add_argument('--output-dir', type=Path, required=True)
    rep.add_argument('--selection-lock', type=Path)
    agg = commands.add_parser('aggregate')
    agg.add_argument('--replica-root', type=Path, required=True)
    agg.add_argument('--split', choices=('validation', 'test'), required=True)
    agg.add_argument('--output', type=Path, required=True)
    agg.add_argument('--selection-lock', type=Path, required=True)
    agg.add_argument('--replay-root', type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = {'train': run, 'lock': lock_selection, 'replay': replay, 'aggregate': aggregate}[command](**args)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
