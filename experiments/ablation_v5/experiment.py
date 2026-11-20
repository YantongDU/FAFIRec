"""Fixed five-seed retraining ablations and separate inference sensitivities.

No model is selected on validation or test. Frozen v4 experts and authenticated
internal calibration caches are shared by all variants. Test caches may only be
created after the complete 45-model ablation lock exists.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.runner import cpu_state, digest, file_digest, json_dump, torch_dump
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import final_runs as fr
from experiments.ablation_v5 import gate as ag

PROTOCOL = 'fafirec_v5_ablation_experiment_v1'
PLAN_PROTOCOL = 'fafirec_v5_ablation_plan_v1'
LOCK_PROTOCOL = 'fafirec_v5_ablation_lock_v1'
SEEDS = [2025, 2026, 2027, 2028, 2029]
SENSITIVITIES = ['mean_diagnostics', 'permuted_quality', 'scalar_mean_gate', 'user_mean_gate']
GROUPS = ('all', 'cold', 'middle', 'warm')
METRICS = ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50')
INFERENCE = {'alpha': .025, 'beta': 0., 'alignment_weight': 0.}


def source_identity():
    return {**fr.source_identity(), **{'experiments/ablation_v5/' + name:
        file_digest(Path(__file__).parent / name) for name in ('gate.py', 'experiment.py')}}


def contained(root, path):
    path, root = Path(path).resolve(), Path(root).resolve()
    ace.require(path != root and path.is_relative_to(root), 'Output escapes ablation root: ' + str(path))
    return path


def immutable(path, value):
    path = Path(path)
    if path.exists():
        ace.require(ace.read(path) == value, 'Immutable ablation artifact differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_dump(path, value)


def fingerprint(value, key='fingerprint'):
    return digest({k: v for k, v in value.items() if k != key})


def verify_hashes(mapping):
    ace.require(isinstance(mapping, dict) and mapping, 'Explicit artifact closure required')
    for path, sha in mapping.items():
        ace.require(Path(path).is_absolute() and file_digest(path) == sha, 'Artifact changed: ' + path)


def variants(plan):
    return plan['retrained_variants'] + plan['sensitivity_variants']


def csv_variant(name):
    return 'diagnostic_fusion' if name == 'full' else name


def fixed_permutation(lengths, fractions, seed=20251120):
    """Joint quality-column permutation within training-derived strata."""
    lengths, fractions = np.asarray(lengths), np.asarray(fractions)
    ace.require(lengths.ndim == fractions.ndim == 1 and len(lengths) == len(fractions), 'Permutation strata shape differs')
    cuts = np.quantile(lengths, np.arange(.1, 1., .1))
    bins = np.searchsorted(cuts, lengths, side='right')
    counts = np.rint(2 * fractions).astype(np.int64)
    ace.require(np.all(np.isin(counts, [0, 1, 2])), 'Probe availability must represent zero, one or two probes')
    rng, permutation = np.random.default_rng(seed), np.arange(len(lengths))
    strata = {}
    for bucket, number in sorted(set(zip(bins.tolist(), counts.tolist()))):
        members = np.flatnonzero((bins == bucket) & (counts == number))
        permutation[members] = rng.permutation(members)
        strata[f'{bucket}:{number}'] = len(members)
    ace.require(sorted(permutation.tolist()) == list(range(len(lengths))), 'Permutation is not bijective')
    return permutation.tolist(), {'seed': seed, 'quantile_cutpoints': cuts.tolist(), 'stratum_sizes': strata,
        'quality_columns': [1, 2, 3, 4, 6], 'preserved_columns': [0, 5],
        'stratification': 'outer train count decile x train-end proxy availability',
        'same_mapping_for_all_seeds_and_splits': True}


def make_plan(root, baseline, parallel):
    oldlock = baseline / 'five_seed_selection_lock.json'
    lock = fr.verify_lock(oldlock)
    ace.require(lock['selected_recipe'] == 'diagnostic_fusion', 'Expected frozen v4 reference recipe')
    records_file = baseline / 'seed_2025/run/cache/records_validation.json'
    val_file = baseline / 'seed_2025/run/cache/validation.pt'
    records = ace.read(records_file)
    values = torch.load(val_file, map_location='cpu', weights_only=True)
    permutation, strata = fixed_permutation([len(r['train']) for r in records], values['features'][:, 5].numpy())
    value = {'protocol': PLAN_PROTOCOL, 'output_root': str(root), 'baseline_root': str(baseline),
        'baseline_lock_file': str(oldlock), 'baseline_lock_sha256': file_digest(oldlock),
        'seed_ids': SEEDS, 'retrained_variants': list(ag.SPECS), 'ablation_specifications': ag.SPECS,
        'sensitivity_variants': SENSITIVITIES, 'training_recipe': ag.TRAINING_RECIPE,
        'inference_parameters': INFERENCE, 'parallel': parallel, 'source_hashes': source_identity(),
        'permutation': permutation, 'permutation_strata': strata,
        'permutation_sha256': digest(permutation), 'user_order': [r['user'] for r in records],
        'baseline_artifact_sha256': {**lock['artifact_sha256'], str(oldlock): file_digest(oldlock)},
        'no_model_selection': True, 'hyperparameters_retuned': False, 'historical_test_known': True,
        'scalar_mean_source': 'frozen full gate on outer train observed-prefix cache; user-macro available-candidate mean',
        'distribution_quantiles': 'deterministic evenly spaced sample of <=100000 available entries; exact pooled mean/std'}
    value['fingerprint'] = fingerprint(value)
    return value


def read_plan(root):
    plan = ace.read(Path(root) / 'plan.json')
    ace.require(plan['protocol'] == PLAN_PROTOCOL and plan['seed_ids'] == SEEDS
        and plan['source_hashes'] == source_identity() and plan['fingerprint'] == fingerprint(plan)
        and plan['ablation_specifications'] == ag.SPECS and plan['training_recipe'] == ag.TRAINING_RECIPE
        and plan['inference_parameters'] == INFERENCE, 'Ablation plan/source differs')
    ace.require(plan['permutation_sha256'] == digest(plan['permutation']), 'Permutation changed')
    ace.require(file_digest(plan['baseline_lock_file']) == plan['baseline_lock_sha256'], 'Baseline lock changed')
    return plan


def model_path(root, seed, name):
    return Path(root) / f'seed_{seed}/models/{name}.pt'


def load_gate(path, variant, plan, seed):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    ace.require(saved['protocol'] == PROTOCOL and saved['plan_fingerprint'] == plan['fingerprint']
        and saved['seed'] == seed and saved['variant'] == variant, 'Ablation checkpoint identity differs')
    model = ag.make_gate(variant)
    model.load_state_dict(saved['model'], strict=True)
    return model.requires_grad_(False).eval(), saved


@torch.no_grad()
def distribution(multipliers, available):
    values = (INFERENCE['alpha'] * multipliers).masked_select(available).double()
    ace.require(len(values) > 0 and torch.isfinite(values).all(), 'Invalid gate distribution')
    stride = max(1, int(np.ceil(len(values) / 100000)))
    sample = values[::stride].numpy()
    return {'pooled_mean': float(values.mean()), 'pooled_std': float(values.std(unbiased=False)),
        'min': float(values.min()), 'max': float(values.max()), 'p05_sampled': float(np.quantile(sample, .05)),
        'p50_sampled': float(np.quantile(sample, .5)), 'p95_sampled': float(np.quantile(sample, .95)),
        'sample_count': len(sample), 'sample_stride': stride, 'available_entry_count': len(values),
        'units': 'effective alpha times gate multiplier'}


@torch.no_grad()
def predict(model, data, plan, sensitivity=None, scalar=None):
    features = data['features']
    if sensitivity == 'mean_diagnostics':
        features = model.feature_mean[None].expand_as(features)
    elif sensitivity == 'permuted_quality':
        features = features.clone()
        columns = plan['permutation_strata']['quality_columns']
        source = data['features'][torch.tensor(plan['permutation'])]
        features[:, columns] = source[:, columns]
    multipliers = model.multipliers(data['sem'], data['cf'], features, data['counts'])
    if sensitivity == 'scalar_mean_gate':
        ace.require(scalar is not None and 0 <= scalar <= 2, 'Frozen train-only mean gate required')
        multipliers = torch.full_like(multipliers, scalar)
    elif sensitivity == 'user_mean_gate':
        mean = multipliers.masked_fill(~data['available'], 0.).sum(1) / data['available'].sum(1)
        multipliers = mean[:, None].expand_as(multipliers)
    if sensitivity in ('scalar_mean_gate', 'user_mean_gate'):
        scores = data['sem'] + INFERENCE['alpha'] * multipliers * data['cf']
    else:
        scores = model(data['sem'], data['cf'], features, data['counts'], alpha=INFERENCE['alpha'])
    return scores.numpy(), distribution(multipliers, data['available'])


def ranking_similarity(path, reference):
    import csv
    with Path(path).open(encoding='utf-8', newline='') as a, Path(reference).open(encoding='utf-8', newline='') as b:
        rows, base = list(csv.DictReader(a)), list(csv.DictReader(b))
    ace.require([r['user'] for r in rows] == [r['user'] for r in base], 'Ranking comparison user order differs')
    same10 = same50 = 0
    for row, original in zip(rows, base):
        top, previous = json.loads(row['top_items']), json.loads(original['top_items'])
        same10 += top[:10] == previous[:10]; same50 += top[:50] == previous[:50]
    return {'users': len(rows), 'top10_identical_fraction': same10 / len(rows),
        'top50_identical_fraction': same50 / len(rows)}


def train_seed(root, seed):
    root = Path(root).resolve(); plan = read_plan(root)
    baseline_run = Path(plan['baseline_root']) / f'seed_{seed}/run'
    bm, bs, cache, training, validation = fr.load_run(baseline_run)
    records = ace.read(baseline_run / 'cache/records_validation.json')
    ace.require([r['user'] for r in records] == plan['user_order'], 'Ablation user mapping differs')
    reference = next(r for r in bs['recipes'] if r['name'] == 'diagnostic_fusion')
    ace.require({k: reference['recipe'][k] for k in ('alpha', 'beta', 'alignment_weight')} == INFERENCE,
        'Reference inference recipe differs')
    original = torch.load(reference['model_file'], map_location='cpu', weights_only=True)
    folder = contained(root, root / f'seed_{seed}')
    folder.mkdir(parents=True, exist_ok=True)
    for name in plan['retrained_variants']:
        path, summary_path = model_path(root, seed, name), folder / f'training/{name}/summary.json'
        if summary_path.exists():
            value = ace.read(summary_path); verify_hashes(value['artifact_sha256'])
            load_gate(path, name, plan, seed)
            continue
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            model, saved = load_gate(path, name, plan, seed)
        else:
            model, logs = ag.fit_ablation(training, name, seed, torch.device('cuda'))
            saved = {'protocol': PROTOCOL, 'plan_fingerprint': plan['fingerprint'], 'seed': seed,
                'variant': name, 'model': cpu_state(model), 'training': logs, 'source_hashes': source_identity()}
            torch_dump(path, saved)
        if name == 'full':
            ace.require(all(torch.equal(saved['model'][k], v) for k, v in original['gate_model'].items()),
                'Full retraining does not exactly reproduce frozen v4 gate tensors')
            ace.require(saved['training'] == original['gate_training'], 'Full training log differs from v4')
            with torch.no_grad():
                g = model.multipliers(validation['sem'], validation['cf'], validation['features'], validation['counts'])
                scalar = float((g.masked_fill(~validation['available'], 0.).sum(1) / validation['available'].sum(1)).mean())
            immutable(folder / 'scalar_mean_gate.json', {'protocol': PROTOCOL, 'seed': seed,
                'plan_fingerprint': plan['fingerprint'], 'value': scalar,
                'source': 'outer train observed prefix; no validation targets read',
                'reference_model_sha256': file_digest(path), 'source_validation_cache_sha256': file_digest(baseline_run / 'cache/validation.pt')})
        scores, gate_stats = predict(model, validation, plan)
        target_folder = folder / f'training/{name}'
        csv = target_folder / 'validation.csv'
        if not csv.exists():
            metrics = fr.write_scored(target_folder, scores, records, bm['evaluation_config'], seed,
                csv_variant(name), 'validation', 'validation.csv')
        else:
            metrics = audit_rows(csv, records, bm['evaluation_config'], seed, csv_variant(name), 'fusion',
                'fafirec_v4_diagnostic', 'validation', bm['identity']['items'])
        if name == 'full':
            ace.require(file_digest(csv) == reference['per_user_sha256'], 'Full validation CSV differs from frozen baseline')
        value = {'protocol': PROTOCOL, 'plan_fingerprint': plan['fingerprint'], 'seed': seed, 'variant': name,
            'csv_variant': csv_variant(name), 'validation': metrics, 'effective_multiplier_distribution': gate_stats,
            'baseline_full_identical': name == 'full', 'source_hashes': source_identity(),
            'artifact_sha256': {str(path): file_digest(path), str(csv): file_digest(csv)}}
        immutable(summary_path, value)
        print(json.dumps({'seed': seed, 'variant': name, 'validation_ndcg50': metrics['all']['ndcg50']}), flush=True)
    hashes = {**bs['artifact_sha256'], str(baseline_run / 'summary.json'): file_digest(baseline_run / 'summary.json')}
    for name in plan['retrained_variants']:
        leaf = folder / f'training/{name}/summary.json'
        hashes[str(leaf)] = file_digest(leaf); hashes.update(ace.read(leaf)['artifact_sha256'])
    scalar_path = folder / 'scalar_mean_gate.json'; hashes[str(scalar_path)] = file_digest(scalar_path)
    immutable(folder / 'training_summary.json', {'protocol': PROTOCOL, 'seed': seed,
        'plan_fingerprint': plan['fingerprint'], 'variants': plan['retrained_variants'],
        'source_hashes': source_identity(), 'artifact_sha256': hashes, 'full_parity_verified': True})


def lock_training(root, plan):
    hashes = {str(root / 'plan.json'): file_digest(root / 'plan.json')}; jobs = []
    for seed in SEEDS:
        folder = root / f'seed_{seed}'; path = folder / 'training_summary.json'; value = ace.read(path)
        ace.require(value['seed'] == seed and value['variants'] == plan['retrained_variants']
            and value['plan_fingerprint'] == plan['fingerprint'] and value['source_hashes'] == source_identity()
            and value['full_parity_verified'], 'Incomplete or changed training scope')
        verify_hashes(value['artifact_sha256']); hashes.update(value['artifact_sha256'])
        hashes[str(path)] = file_digest(path); jobs.append({'seed': seed, 'run_dir': str(folder)})
    lock = {'protocol': LOCK_PROTOCOL, 'seed_ids': SEEDS, 'jobs': jobs, 'source_hashes': source_identity(),
        'plan_sha256': file_digest(root / 'plan.json'), 'variants': variants(plan),
        'artifact_sha256': hashes, 'no_model_selection': True}
    lock['lock_fingerprint'] = fingerprint(lock, 'lock_fingerprint')
    path = root / 'ablation_lock.json'; immutable(path, lock); verify_lock(path)
    return path, lock


def verify_lock(path):
    lock = ace.read(path)
    ace.require(lock['protocol'] == LOCK_PROTOCOL and lock['seed_ids'] == SEEDS
        and lock['source_hashes'] == source_identity() and lock['lock_fingerprint'] == fingerprint(lock, 'lock_fingerprint')
        and sorted(j['seed'] for j in lock['jobs']) == SEEDS
        and len({j['run_dir'] for j in lock['jobs']}) == 5, 'Complete immutable five-seed ablation lock required')
    verify_hashes(lock['artifact_sha256'])
    return lock


def test_cache(root, seed):
    lock_path = root / 'ablation_lock.json'; verify_lock(lock_path); plan = read_plan(root)
    folder = contained(root, root / f'seed_{seed}/test_cache'); meta_file = folder / 'manifest.json'
    if meta_file.exists():
        saved = ace.read(meta_file)
        ace.require(saved['source_hashes'] == source_identity() and saved['seed'] == seed
            and saved['ablation_lock_sha256'] == file_digest(lock_path), 'Test cache identity changed')
        verify_hashes(saved['artifact_sha256']); return
    baseline_run = Path(plan['baseline_root']) / f'seed_{seed}/run'
    bm, _bs, cp, training, validation = fr.load_run(baseline_run)
    del training, validation; gc.collect()
    records, histories, data = fr.reconstruct(baseline_run, 'test', bm, cp)
    B = torch.load(baseline_run / 'cache/relations.pt', map_location='cpu', weights_only=True)['B']
    data['cf'] = fr.fusion.score_operator(B, histories, data['available'])
    folder.mkdir(parents=True, exist_ok=True)
    torch_dump(folder / 'scores.pt', data); json_dump(folder / 'records.json', records)
    saved = {'protocol': PROTOCOL, 'seed': seed, 'source_hashes': source_identity(),
        'plan_fingerprint': plan['fingerprint'], 'ablation_lock_sha256': file_digest(lock_path),
        'baseline_run_fingerprint': bm['fingerprint'], 'identity': bm['identity'], 'evaluation_config': bm['evaluation_config'],
        'test_labels_used_only_for_evaluation': True,
        'artifact_sha256': {str(p): file_digest(p) for p in (folder / 'scores.pt', folder / 'records.json')}}
    immutable(meta_file, saved)


def replay_seed(root, seed, split, out):
    root, out = Path(root).resolve(), Path(out).resolve(); plan = read_plan(root)
    ace.require(not out.exists(), 'Fresh replay output required')
    lock_path = root / 'ablation_lock.json'; verify_lock(lock_path)
    baseline_run = Path(plan['baseline_root']) / f'seed_{seed}/run'
    if split == 'test':
        cache_folder = root / f'seed_{seed}/test_cache'; meta = ace.read(cache_folder / 'manifest.json')
        ace.require(meta['source_hashes'] == source_identity() and meta['ablation_lock_sha256'] == file_digest(lock_path),
            'Test scoring cache/source differs')
        verify_hashes(meta['artifact_sha256'])
        records = ace.read(cache_folder / 'records.json')
        data = torch.load(cache_folder / 'scores.pt', map_location='cpu', weights_only=True)
        cfg = meta['evaluation_config']; n_items = meta['identity']['items']
    else:
        bm, _bs, _cp, training, data = fr.load_run(baseline_run)
        del training
        records = ace.read(baseline_run / 'cache/records_validation.json')
        cfg, n_items = bm['evaluation_config'], bm['identity']['items']
    scalar_file = root / f'seed_{seed}/scalar_mean_gate.json'
    scalar = ace.read(scalar_file)['value']; full, _ = load_gate(model_path(root, seed, 'full'), 'full', plan, seed)
    out.mkdir(parents=True)
    for name in variants(plan):
        sensitivity = name if name in SENSITIVITIES else None
        model = full if sensitivity else load_gate(model_path(root, seed, name), name, plan, seed)[0]
        scores, stats = predict(model, data, plan, sensitivity, scalar)
        folder = out / name
        metrics = fr.write_scored(folder, scores, records, cfg, seed, csv_variant(name), split, 'per_user.csv')
        csv = folder / 'per_user.csv'; checksum = file_digest(csv)
        baseline_equal = False
        if name == 'full':
            original = Path(plan['baseline_root']) / f'replay_{split}/seed_{seed}/diagnostic_fusion/per_user.csv'
            ace.require(checksum == file_digest(original), 'Full ranking differs from frozen v4 baseline')
            baseline_equal = True
        identical = False
        if split == 'validation' and not sensitivity:
            ace.require(checksum == file_digest(root / f'seed_{seed}/training/{name}/validation.csv'),
                'New model validation reload differs from training export')
            identical = True
        value = {'protocol': PROTOCOL, 'seed': seed, 'variant': name, 'csv_variant': csv_variant(name),
            'stage': 'fusion', 'split': split, 'category': 'sensitivity' if sensitivity else 'retrained',
            'plan_fingerprint': plan['fingerprint'], 'ablation_lock_sha256': file_digest(lock_path),
            'metrics': metrics, 'per_user_sha256': checksum,
            'model_sha256': file_digest(model_path(root, seed, 'full' if sensitivity else name)),
            'identical_to_training': identical, 'baseline_full_identical': baseline_equal,
            'effective_multiplier_distribution': stats,
            'ranking_vs_full': ranking_similarity(csv, out / 'full/per_user.csv')}
        json_dump(folder / 'metrics.json', value)
    print(f'{split} seed {seed}: {len(variants(plan))} ablation/sensitivity scores complete', flush=True)


def command(root, arguments, label):
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    logfile = logs / f'{label}_{uuid.uuid4().hex}.log'
    with logfile.open('xb') as stream:
        result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--output-root', str(root), *arguments],
            cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    ace.require(result.returncode == 0, 'Ablation worker failed; inspect ' + str(logfile))


def dispatch_training(root, plan):
    pending, active = list(SEEDS), {}
    try:
        while pending or active:
            while pending and len(active) < plan['parallel']:
                seed = pending.pop(0)
                summary = root / f'seed_{seed}/training_summary.json'
                if summary.exists():
                    value = ace.read(summary); verify_hashes(value['artifact_sha256'])
                    ace.require(value['source_hashes'] == source_identity() and value['plan_fingerprint'] == plan['fingerprint'],
                        'Completed training source differs')
                    continue
                logs = root / 'logs'; logs.mkdir(exist_ok=True)
                log = logs / f'training_{seed}_{uuid.uuid4().hex}.log'; stream = log.open('xb')
                args = [sys.executable, '-u', str(Path(__file__).resolve()), '--output-root', str(root), '--worker', 'training', '--seed', str(seed)]
                child = subprocess.Popen(args, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                active[seed] = (child, stream)
                print(f'Started ablation training seed {seed} PID {child.pid}', flush=True)
            for seed, (child, stream) in list(active.items()):
                status = child.poll()
                if status is not None:
                    stream.close(); del active[seed]
                    ace.require(status == 0, f'Training seed {seed} failed; inspect logs before resuming')
                    print(f'Completed ablation training seed {seed}', flush=True)
            if active: time.sleep(2)
    finally:
        for child, stream in active.values():
            if child.poll() is None: child.terminate(); child.wait(timeout=30)
            stream.close()


def verify_replay(folder, root, plan, seed, split):
    if split == 'test':
        records = ace.read(root / f'seed_{seed}/test_cache/records.json')
        metadata = ace.read(root / f'seed_{seed}/test_cache/manifest.json')
        cfg, n_items = metadata['evaluation_config'], metadata['identity']['items']
    else:
        run = Path(plan['baseline_root']) / f'seed_{seed}/run'
        metadata = ace.read(run / 'manifest.json')
        records = ace.read(run / 'cache/records_validation.json')
        cfg, n_items = metadata['evaluation_config'], metadata['identity']['items']
    values = {}
    for name in variants(plan):
        base = folder / name; saved = ace.read(base / 'metrics.json')
        ace.require(saved['protocol'] == PROTOCOL and saved['seed'] == seed and saved['variant'] == name
            and saved['split'] == split and saved['plan_fingerprint'] == plan['fingerprint']
            and saved['ablation_lock_sha256'] == file_digest(root / 'ablation_lock.json')
            and saved['per_user_sha256'] == file_digest(base / 'per_user.csv')
            and saved['model_sha256'] == file_digest(model_path(root, seed, 'full' if name in SENSITIVITIES else name)),
            'Replay artifact/identity differs: ' + str(base))
        actual = audit_rows(base / 'per_user.csv', records, cfg, seed, csv_variant(name), 'fusion',
            'fafirec_v4_diagnostic', split, n_items)
        _same(actual, saved['metrics'], 'Independent ablation CSV metric recomputation')
        ace.require(name != 'full' or saved['baseline_full_identical'], 'Full baseline parity missing')
        values[name] = saved
    return values


def ensure_replay(root, plan, seed, split, repeated=False):
    parent = 'replay_test_reloaded' if repeated else 'replay_' + split
    folder = contained(root, root / f'{parent}/seed_{seed}')
    if not folder.exists():
        pending = contained(root, root / '.replay_pending'); pending.mkdir(exist_ok=True)
        stage = contained(root, pending / f'{split}_{seed}_{uuid.uuid4().hex}')
        command(root, ['--worker', 'replay', '--seed', str(seed), '--split', split, '--replay-dir', str(stage)], f'{split}_{seed}')
        verify_replay(stage, root, plan, seed, split)
        folder.parent.mkdir(exist_ok=True)
        ace.require(not folder.exists(), 'Replay destination appeared')
        contained(root, stage).rename(folder)
    return folder, verify_replay(folder, root, plan, seed, split)


def evaluate_all(root, plan, split):
    lock_path = root / 'ablation_lock.json'; lock = verify_lock(lock_path)
    results, metric_map = [], {}
    for seed in SEEDS:
        if split == 'test': test_cache(root, seed)
        first_folder, first = ensure_replay(root, plan, seed, split)
        if split == 'test': second_folder, second = ensure_replay(root, plan, seed, split, True)
        for name in variants(plan):
            value = first[name]; metric_map[seed, name] = value['metrics']
            row = {k: value[k] for k in ('seed', 'variant', 'csv_variant', 'category', 'identical_to_training',
                'baseline_full_identical', 'model_sha256', 'effective_multiplier_distribution', 'ranking_vs_full')}
            row.update(per_user_file=str(first_folder / name / 'per_user.csv'), per_user_sha256=value['per_user_sha256'],
                metrics_file=str(first_folder / name / 'metrics.json'), metrics_sha256=file_digest(first_folder / name / 'metrics.json'))
            if split == 'test':
                other = second[name]
                ace.require(value == other and row['metrics_sha256'] == file_digest(second_folder / name / 'metrics.json'),
                    'Two independently loaded ablation test scores differ')
                row.update(independent_replay_identical=True, reloaded_file=str(second_folder / name / 'per_user.csv'),
                    reloaded_sha256=other['per_user_sha256'], reloaded_metrics_file=str(second_folder / name / 'metrics.json'),
                    reloaded_metrics_sha256=file_digest(second_folder / name / 'metrics.json'))
            results.append(row)
        print(f'{split}: verified seed {seed}, {len(variants(plan))} variants', flush=True)
    proof = {'protocol': PROTOCOL, 'split': split, 'complete': True, 'seed_ids': SEEDS,
        'source_hashes': source_identity(), 'ablation_lock_sha256': file_digest(lock_path), 'results': results,
        'no_model_selection': True}
    proof_file = root / f'{split}_verification.json'; immutable(proof_file, proof)
    rows = []
    for name in variants(plan):
        for group in GROUPS:
            for metric in METRICS:
                scores = [metric_map[s, name][group][metric] for s in SEEDS]
                rows.append({'variant': name, 'stage': 'fusion', 'group': group, 'metric': metric, 'seed_ids': SEEDS,
                    'seed_count': 5, 'per_seed_scores': scores, 'mean': statistics.mean(scores), 'sample_std': statistics.stdev(scores)})
    hashes = {**lock['artifact_sha256'], str(lock_path): file_digest(lock_path), str(proof_file): file_digest(proof_file)}
    for r in results:
        for p, s in (('per_user_file', 'per_user_sha256'), ('metrics_file', 'metrics_sha256'),
            ('reloaded_file', 'reloaded_sha256'), ('reloaded_metrics_file', 'reloaded_metrics_sha256')):
            if p in r: hashes[r[p]] = r[s]
    if split == 'test':
        for seed in SEEDS:
            meta = root / f'seed_{seed}/test_cache/manifest.json'
            hashes[str(meta)] = file_digest(meta); hashes.update(ace.read(meta)['artifact_sha256'])
    verify_hashes(hashes)
    combined = {'protocol': PROTOCOL, 'split': split, 'seed_ids': SEEDS, 'variants': variants(plan),
        'retrained_variants': plan['retrained_variants'], 'sensitivity_variants': plan['sensitivity_variants'],
        'aggregates': rows, 'source_hashes': source_identity(), 'artifact_sha256': hashes,
        'ablation_lock_file': str(lock_path), 'ablation_lock_sha256': file_digest(lock_path),
        'all_artifact_hashes_verified': True, 'no_model_selection': True}
    immutable(root / f'combined_{split}.json', combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('runs/ablation_v5'))
    parser.add_argument('--baseline-root', type=Path, default=Path('E:/论文/FAFIRec_Diagnostic_v4/runs/diagnostic_v4_final'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--through', choices=('training', 'validation', 'test'), default='test')
    parser.add_argument('--parallel', type=int, choices=(1, 2), default=2)
    parser.add_argument('--worker', choices=('training', 'replay', 'cache'))
    parser.add_argument('--seed', type=int); parser.add_argument('--split', choices=('validation', 'test'))
    parser.add_argument('--replay-dir', type=Path)
    args = parser.parse_args(); torch.set_num_threads(4)
    root, baseline = args.output_root.resolve(), args.baseline_root.resolve()
    if args.worker:
        ace.require(args.seed in SEEDS, 'Worker needs a known seed')
        if args.worker == 'training': train_seed(root, args.seed)
        elif args.worker == 'cache': test_cache(root, args.seed)
        else: replay_seed(root, args.seed, args.split, args.replay_dir)
        return
    fresh = make_plan(root, baseline, args.parallel)
    if root.exists() and any(root.iterdir()):
        ace.require(args.resume and ace.read(root / 'plan.json') == fresh, 'Existing ablation plan requires identical resume')
    root.mkdir(parents=True, exist_ok=True); immutable(root / 'plan.json', fresh)
    dispatch_training(root, fresh); lock_training(root, fresh)
    if args.through in ('validation', 'test'): evaluate_all(root, fresh, 'validation')
    if args.through == 'test': evaluate_all(root, fresh, 'test')
    verify_lock(root / 'ablation_lock.json')
    print('Complete fixed ablations through ' + args.through, flush=True)


if __name__ == '__main__': main()
