"""Complete five fixed FAFIRec v4 recipes, validation lock and independent replay.

Training and each replay use separate Python processes. Completed immutable
outputs are re-audited before reuse. Tests cannot run before all five validation
runs are locked; their scores never select a recipe.
"""
from __future__ import annotations

import argparse
import gc
import importlib
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

from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.runner import digest, file_digest, json_dump
from tools import ace2026_experiment as ace

SEEDS = [2025, 2026, 2027, 2028, 2029]
GROUPS = ('all', 'cold', 'middle', 'warm')
METRICS = ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50')
SCHEMA = 'fafirec_v4_complete_five_seed_v1'
LOCK_PROTOCOL = 'fafirec_v4_five_seed_validation_lock_v1'
fr = None


def contained(root, path):
    path = Path(path).resolve()
    ace.require(path != root and path.is_relative_to(root), 'Output must stay within the experiment root: ' + str(path))
    return path


def immutable(path, value):
    if path.exists():
        ace.require(ace.read(path) == value, 'Immutable output differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_dump(path, value)


def identity():
    return {**fr.source_identity(), Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def make_plan(root, components, recipe_path, parallel):
    recipe = ace.read(recipe_path)
    ace.require(recipe['protocol'] == 'fafirec_v4_recipe_plan_v1' and recipe['seed_ids'] == SEEDS
        and recipe['source_hashes'] == fr.fusion.source_identity()
        and recipe['fingerprint'] == digest({k: v for k, v in recipe.items() if k != 'fingerprint'}),
        'Fixed validation recipe plan differs')
    recipes = [r['name'] for r in recipe['recipes']]
    ace.require(len(recipes) == len(set(recipes)) and recipes, 'Distinct fixed recipe names required')
    for path, checksum in recipe['artifact_sha256'].items():
        ace.require(file_digest(path) == checksum, 'Recipe selection input changed: ' + path)
    ace.require((components / 'completion.json').is_file(),
        'Component training is incomplete: completion.json is not available yet')
    complete = ace.read(components / 'completion.json')
    ace.require(complete['complete'] and complete['seed_ids'] == SEEDS
        and complete['plan_sha256'] == file_digest(components / 'master_plan.json')
        and len(complete['receipts']) == 10, 'All ten component jobs must complete before final fusion')
    ace.require({(r['seed'], r['component']) for r in complete['receipts'].values()}
        == {(seed, component) for seed in SEEDS for component in ('coverage', 'teacher')},
        'Completed component receipts do not cover both models for every seed')
    for receipt in complete['receipts'].values():
        for path, checksum in receipt['artifact_sha256'].items():
            ace.require(file_digest(path) == checksum, 'Completed component changed: ' + path)
    value = {'schema': SCHEMA, 'root': str(root), 'components_root': str(components),
        'recipe_plan_file': str(recipe_path), 'recipe_plan_sha256': file_digest(recipe_path),
        'components_completion_sha256': file_digest(components / 'completion.json'),
        'seed_ids': SEEDS, 'recipe_names': recipes, 'parallel': parallel,
        'source_hashes': identity(), 'selection_split': 'validation',
        'new_test_used_for_selection': False, 'historical_test_known': True,
        'jobs': [{'seed': seed, 'run_dir': str(root / f'seed_{seed}/run')} for seed in SEEDS]}
    value['fingerprint'] = digest(value)
    return value


def alive(pid):
    if not pid:
        return False
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def load_context(job):
    manifest, summary, cache_plan, calibration, val = fr.load_run(Path(job['run_dir']))
    del calibration, val
    gc.collect()
    ace.require(manifest['seed'] == summary['seed'] == job['seed'], 'Final seed identity differs')
    indexed = {r['name']: r for r in summary['recipes']}
    ace.require(len(indexed) == len(summary['recipes']), 'Duplicate recipe summary')
    return manifest, summary, {**cache_plan, '_run_fingerprint': manifest['fingerprint']}, indexed


def train_all(root, plan, resume):
    status_path = root / 'dispatch_status.json'
    if status_path.exists():
        prior = ace.read(status_path)
        ace.require(not (prior.get('master_pid') != os.getpid() and alive(prior.get('master_pid')))
            and not any(alive(pid) for pid in prior.get('active', {}).values()), 'Previous fusion dispatcher is still running')
    pending, active, completed, failed = [], {}, [], []
    for job in plan['jobs']:
        folder = Path(job['run_dir'])
        if (folder / 'summary.json').exists():
            load_context(job)
            completed.append(job['seed'])
        else:
            pending.append(job)
    try:
        while pending or active:
            while pending and len(active) < plan['parallel'] and not failed:
                job = pending.pop(0)
                folder = contained(root, Path(job['run_dir']).parent)
                folder.mkdir(parents=True, exist_ok=True)
                stream = (folder / ('training_resume.log' if resume else 'training.log')).open('ab')
                command = [sys.executable, '-u', str(Path(fr.__file__).resolve()), 'train',
                    '--seed', str(job['seed']), '--components-root', plan['components_root'],
                    '--recipe-plan', plan['recipe_plan_file'], '--output-dir', job['run_dir']]
                if resume and Path(job['run_dir']).exists():
                    command.append('--resume')
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                active[job['seed']] = child, stream, job
                print(f"Started final seed {job['seed']} PID {child.pid}", flush=True)
            for seed, (child, stream, job) in list(active.items()):
                code = child.poll()
                if code is not None:
                    stream.close(); del active[seed]
                    if code == 0:
                        load_context(job); completed.append(seed)
                    else:
                        failed.append(seed)
                    print(f'Finished final seed {seed} exit {code}', flush=True)
            json_dump(status_path, {'master_pid': os.getpid(), 'plan_fingerprint': plan['fingerprint'],
                'active': {str(seed): value[0].pid for seed, value in active.items()},
                'completed': sorted(completed), 'pending': [j['seed'] for j in pending], 'failed': failed})
            if failed and not active:
                raise RuntimeError('Final seed failed; inspect its training log before resuming')
            if active:
                time.sleep(2)
    finally:
        for child, stream, _ in active.values():
            if child.poll() is None:
                child.terminate(); child.wait(timeout=30)
            stream.close()
    ace.require(sorted(completed) == SEEDS, 'Incomplete final five-seed grid')


def lock_selection(root, plan):
    hashes = {str(root / 'master_plan.json'): file_digest(root / 'master_plan.json'),
        plan['recipe_plan_file']: plan['recipe_plan_sha256']}
    scores, common = {name: [] for name in plan['recipe_names']}, None
    for job in plan['jobs']:
        manifest, summary, cache, recipes = load_context(job)
        ace.require(set(recipes) == set(plan['recipe_names']), 'Final recipe coverage differs')
        ace.require(manifest['recipe_plan_sha256'] == plan['recipe_plan_sha256']
            and manifest['recipe_plan_file'] == plan['recipe_plan_file'], 'Final seed uses another recipe plan')
        comparison = {key: cache[key] for key in ('identity', 'evaluation_config')}
        ace.require(common is None or common == comparison, 'Common inputs/evaluation differ across seeds')
        common = comparison
        for name in plan['recipe_names']:
            scores[name].append(recipes[name]['validation']['all']['ndcg50'])
        hashes.update(summary['artifact_sha256'])
        for name in ('manifest.json', 'summary.json'):
            path = Path(job['run_dir']) / name
            hashes[str(path)] = file_digest(path)
    candidates = sorted([{'recipe': name, 'seed_ids': SEEDS, 'per_seed_validation_ndcg50': values,
        'mean_validation_ndcg50': statistics.mean(values)} for name, values in scores.items()],
        key=lambda row: (-row['mean_validation_ndcg50'], row['recipe']))
    lock = {'protocol': LOCK_PROTOCOL, 'seed_ids': SEEDS, 'jobs': plan['jobs'],
        'source_hashes': fr.source_identity(), 'recipe_plan_sha256': plan['recipe_plan_sha256'],
        'recipe_plan_file': plan['recipe_plan_file'], 'artifact_sha256': hashes,
        'selected_recipe': candidates[0]['recipe'], 'candidates': candidates,
        'common': common, 'selection_split': 'validation', 'new_test_used_for_selection': False,
        'historical_test_known': True}
    lock['lock_fingerprint'] = digest(lock)
    path = contained(root, root / 'five_seed_selection_lock.json')
    immutable(path, lock)
    fr.verify_lock(path)
    return path, lock


def records_for_split(job, cache, split):
    if split == 'validation':
        return ace.read(Path(job['run_dir']) / 'cache/records_validation.json')
    cfg = ace.normalize_config(ace.read(Path(cache['ace_run']) / 'manifest.json')['config'])
    records, features, identity_value, *_ = ace.inputs(cfg)
    ace.require(identity_value == cache['identity'], 'Test record identity differs from locked training inputs')
    del features
    return records


def verify_replay(folder, job, split, lock_path, cache, recipes, records):
    values = {}
    ace.require(folder.is_dir(), 'Replay directory missing')
    for name, selected in recipes.items():
        sub = folder / name
        csv_path, metric_path = sub / 'per_user.csv', sub / 'metrics.json'
        saved = ace.read(metric_path)
        ace.require(saved['protocol'] == fr.PROTOCOL and saved['fingerprint'] == cache['_run_fingerprint']
            and saved['seed'] == job['seed'] and saved['variant'] == name and saved['stage'] == 'fusion'
            and saved['split'] == split and saved['candidate_policy'] == cache['evaluation_config']['candidate_policy']
            and saved['model_sha256'] == selected['model_sha256']
            and saved['per_user_sha256'] == file_digest(csv_path), 'Replay identity differs: ' + str(sub))
        if split == 'validation':
            ace.require(saved['identical_to_original'] and saved['per_user_sha256'] == selected['per_user_sha256'],
                'Validation ranking differs from training export')
            _same(saved['metrics'], selected['validation'], 'Independent validation replay metrics')
        else:
            ace.require(saved['selection_lock_sha256'] == file_digest(lock_path), 'Test replay uses a different validation lock')
        actual = audit_rows(csv_path, records, cache['evaluation_config'], job['seed'], name, 'fusion',
            'fafirec_v4_diagnostic', split, cache['identity']['items'])
        _same(actual, saved['metrics'], 'Replay CSV independent metric recomputation')
        values[name] = {'seed': job['seed'], 'variant': name, 'split': split,
            'model_sha256': selected['model_sha256'], 'per_user_file': str(csv_path),
            'per_user_sha256': file_digest(csv_path), 'metrics_file': str(metric_path),
            'metrics_sha256': file_digest(metric_path), 'metrics': saved['metrics']}
    ace.require({p.name for p in folder.iterdir() if p.is_dir()} == set(recipes), 'Extra/missing replay recipe directory')
    return values


def ensure_replay(root, folder, job, split, lock_path, cache, recipes, records):
    if not folder.exists():
        pending = contained(root, root / '.replay_pending')
        pending.mkdir(exist_ok=True)
        stage = contained(root, pending / f'{split}_{job["seed"]}_{uuid.uuid4().hex}')
        logs = root / 'replay_logs'; logs.mkdir(exist_ok=True)
        log_path = logs / f'{split}_{job["seed"]}_{uuid.uuid4().hex}.log'
        command = [sys.executable, '-u', str(Path(fr.__file__).resolve()), 'replay', '--run-dir', job['run_dir'],
            '--split', split, '--output-dir', str(stage)]
        if split == 'test':
            command += ['--selection-lock', str(lock_path)]
        with log_path.open('xb') as stream:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ace.require(result.returncode == 0, 'Independent replay failed; inspect ' + str(log_path))
        verify_replay(stage, job, split, lock_path, cache, recipes, records)
        destination = contained(root, folder)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ace.require(not destination.exists(), 'Replay output appeared during verification')
        contained(root, stage).rename(destination)
    return verify_replay(folder, job, split, lock_path, cache, recipes, records)


def replay_all(root, plan, lock_path, split):
    results, metrics = [], {}
    for job in plan['jobs']:
        fr.verify_lock(lock_path)
        _, _, cache, recipes = load_context(job)
        records = records_for_split(job, cache, split)
        folder = contained(root, root / f'replay_{split}/seed_{job["seed"]}')
        first = ensure_replay(root, folder, job, split, lock_path, cache, recipes, records)
        if split == 'test':
            second_folder = contained(root, root / f'replay_test_reloaded/seed_{job["seed"]}')
            second = ensure_replay(root, second_folder, job, split, lock_path, cache, recipes, records)
        for name, original in first.items():
            values = original['metrics']
            row = {key: value for key, value in original.items() if key != 'metrics'}
            metrics[job['seed'], name] = values
            if split == 'validation':
                row['identical_to_original'] = True
            else:
                other = second[name]
                ace.require(row['per_user_sha256'] == other['per_user_sha256']
                    and row['metrics_sha256'] == other['metrics_sha256'], 'Independent test ranking/metric bytes differ')
                _same(values, other['metrics'], 'Independent test repeated metrics')
                row.update(independent_replay_identical=True, reloaded_file=other['per_user_file'],
                    reloaded_sha256=other['per_user_sha256'], reloaded_metrics_file=other['metrics_file'],
                    reloaded_metrics_sha256=other['metrics_sha256'])
            results.append(row)
        print(f'{split}: audited seed {job["seed"]}, {len(recipes)} recipes', flush=True)
    expected = {(seed, name) for seed in SEEDS for name in plan['recipe_names']}
    ace.require(set(metrics) == expected and len(results) == len(expected), 'Replay grid is incomplete')
    report = {'schema': SCHEMA, 'split': split, 'complete': True, 'seed_ids': SEEDS,
        'recipe_names': plan['recipe_names'], 'source_hashes': identity(),
        'selection_lock_sha256': file_digest(lock_path), 'master_plan_sha256': file_digest(root / 'master_plan.json'),
        'results': results, 'independent_process_per_replay': True}
    immutable(root / f'{split}_verification.json', report)
    return metrics, report


def aggregate(root, plan, lock_path, lock, split, scores, verification):
    rows = []
    for name in plan['recipe_names']:
        for group in GROUPS:
            for metric in METRICS:
                values = [scores[seed, name][group][metric] for seed in SEEDS]
                rows.append({'variant': name, 'stage': 'fusion', 'group': group, 'metric': metric,
                    'seed_ids': SEEDS, 'seed_count': 5, 'per_seed_scores': values,
                    'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)})
    hashes = dict(lock['artifact_sha256'])
    hashes[str(lock_path)] = file_digest(lock_path)
    hashes[str(root / f'{split}_verification.json')] = file_digest(root / f'{split}_verification.json')
    for row in verification['results']:
        for path_key, sha_key in (('per_user_file', 'per_user_sha256'), ('metrics_file', 'metrics_sha256'),
            ('reloaded_file', 'reloaded_sha256'), ('reloaded_metrics_file', 'reloaded_metrics_sha256')):
            if path_key in row:
                hashes[row[path_key]] = row[sha_key]
    for path, checksum in hashes.items():
        ace.require(file_digest(path) == checksum, 'Aggregate input artifact changed: ' + path)
    report = {'schema': SCHEMA, 'split': split, 'seed_ids': SEEDS, 'aggregates': rows,
        'selected_recipe': lock['selected_recipe'], 'decision': lock['candidates'],
        'common': {**lock['common'], 'recipe_plan_sha256': plan['recipe_plan_sha256'],
            'recipes': ace.read(plan['recipe_plan_file'])['recipes'], 'source_hashes': fr.source_identity()},
        'selection_lock_file': str(lock_path), 'selection_lock_sha256': file_digest(lock_path),
        'artifact_sha256': hashes, 'all_artifact_hashes_verified': True,
        'historical_test_known': True, 'new_test_used_for_selection': False}
    immutable(root / f'combined_{split}.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components-root', type=Path, required=True)
    parser.add_argument('--recipe-plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--parallel', type=int, choices=(1, 2), default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--through', choices=('training', 'validation', 'test'), default='test')
    args = parser.parse_args()
    global fr
    fr = importlib.import_module('experiments.diagnostic_v4.final_runs')
    fr.torch.set_num_threads(4)
    root, components, recipe_path = args.output_root.resolve(), args.components_root.resolve(), args.recipe_plan.resolve()
    plan = make_plan(root, components, recipe_path, args.parallel)
    if root.exists() and any(root.iterdir()):
        ace.require(args.resume and ace.read(root / 'master_plan.json') == plan, 'Five-seed master plan changed')
    else:
        root.mkdir(parents=True, exist_ok=True)
    immutable(root / 'master_plan.json', plan)
    train_all(root, plan, args.resume)
    lock_path, lock = lock_selection(root, plan)
    if args.through != 'training':
        metrics, verification = replay_all(root, plan, lock_path, 'validation')
        aggregate(root, plan, lock_path, lock, 'validation', metrics, verification)
    if args.through == 'test':
        metrics, verification = replay_all(root, plan, lock_path, 'test')
        aggregate(root, plan, lock_path, lock, 'test', metrics, verification)
    ace.require(make_plan(root, components, recipe_path, args.parallel) == plan, 'Master inputs changed during execution')
    fr.verify_lock(lock_path)
    print(f'Complete through {args.through}: five-seed recipes and selection lock audited.', flush=True)


if __name__ == '__main__':
    main()
