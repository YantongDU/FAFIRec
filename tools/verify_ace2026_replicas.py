"""Lock and independently replay five ACE replicas, with audited resume.

Every ranking replay runs in a fresh Python process. Existing completed outputs
are checked against the lock, frozen inputs and independently recomputed CSV
metrics before being reused. Immutable outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.runner import digest, file_digest, json_dump
from tools import ace2026_experiment as ace

SCHEMA = 'ace2026_five_seed_independent_replay_v1'


def immutable_json(path, value):
    """A completed result may only be reused when its whole value matches."""
    if path.exists():
        ace.require(ace.read(path) == value, 'Existing immutable result differs: ' + str(path))
    else:
        json_dump(path, value)


def make_plan(root):
    runs_file = root / 'replica_runs.json'
    runs = ace.read(runs_file)
    ace.require(isinstance(runs, list) and len(runs) == len(set(runs)) == 5
                and all(isinstance(p, str) and Path(p).is_absolute() for p in runs),
                'Five distinct absolute replica run directories required')
    initial = {str(runs_file): file_digest(runs_file)}
    if (root / 'master_plan.json').exists():
        initial[str(root / 'master_plan.json')] = file_digest(root / 'master_plan.json')
    seeds = []
    for run in runs:
        folder = Path(run).resolve()
        manifest = ace.read(folder / 'manifest.json')
        cfg = ace.normalize_config(manifest['config'])
        ace.require(len(cfg['seeds']) == len(cfg['variants']) == 1
                    and cfg['output_dir'] == str(folder)
                    and manifest['fingerprint'] == ace.fingerprint(manifest)
                    and manifest['source_hashes'] == ace.source_identity(),
                    'Complete isolated frozen replicas required')
        ace.check_grid(folder, manifest)
        seeds += cfg['seeds']
        for name in ('manifest.json', 'selection.json', 'summary.json'):
            initial[str(folder / name)] = file_digest(folder / name)
    ace.require(len(set(seeds)) == 5, 'Five distinct seed identifiers required')
    value = {'schema': SCHEMA, 'root': str(root), 'replica_runs': runs,
             'seed_ids': sorted(seeds), 'driver_sha256': file_digest(__file__),
             'python_executable': str(Path(sys.executable).resolve()),
             'source_hashes': ace.source_identity(), 'input_artifact_sha256': initial,
             'replay_policy': 'validation exact original; test two fresh processes; CSV metrics independently recomputed',
             'historical_test_known': True, 'new_test_used_for_selection': False}
    value['fingerprint'] = digest(value)
    return value


def run_command(root, arguments, label):
    logs = root / 'replay_logs'
    logs.mkdir(exist_ok=True)
    log_path = logs / (label + '_' + uuid.uuid4().hex + '.log')
    command = [sys.executable, '-u', str(ROOT / 'tools/ace2026_experiment.py'), *arguments]
    with log_path.open('xb') as stream:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                                stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    ace.require(result.returncode == 0, 'ACE verification step failed; inspect ' + str(log_path))


def context(job):
    model, records, cfg, manifest, summary = ace.load_selected(job['run_dir'], job['seed'], job['variant'])
    control = model.control
    del model
    return records, cfg, manifest, summary, control


def verify_replay(folder, job, split, lock, lock_path, ctx):
    records, cfg, manifest, summary, control = ctx
    value = ace.read(folder / 'metrics.json')
    csv_path = folder / 'per_user.csv'
    checksum = file_digest(csv_path)
    ace.require(value['protocol'] == ace.PROTOCOL and value['fingerprint'] == manifest['fingerprint']
                and value['seed'] == job['seed'] and value['variant'] == job['variant']
                and value['split'] == split and value['candidate_policy'] == cfg['candidate_policy']
                and value['checkpoint_sha256'] == summary['checkpoint_sha256']
                and value['per_user_sha256'] == checksum,
                'Replay identity or artifact hash differs: ' + str(folder))
    if split == 'validation':
        original = str(Path(job['run_dir']) / f"seed_{job['seed']}/{job['variant']}/validation.csv")
        ace.require(value.get('identical_to_original') is True
                    and checksum == summary['per_user_sha256'] == lock['artifact_sha256'][original],
                    'Validation replay differs from the locked training ranking')
        _same(value['metrics'], summary['validation'], 'Validation replay metrics')
        records = [{k: v for k, v in record.items() if k != 'test'} for record in records]
    else:
        ace.require(value.get('selection_lock_sha256') == file_digest(lock_path), 'Test selection lock differs')
    recomputed = audit_rows(csv_path, records, cfg, job['seed'], job['variant'], 'single',
                            control, split, manifest['identity']['items'])
    _same(recomputed, value['metrics'], 'Independent replay CSV metric recomputation')
    return value


def ensure_replay(root, folder, job, split, lock, lock_path, ctx):
    if not folder.exists():
        pending = root / '.replay_pending'
        pending.mkdir(exist_ok=True)
        stage = pending / (split + '_' + str(job['seed']) + '_' + uuid.uuid4().hex)
        arguments = ['replay', '--run-dir', job['run_dir'], '--seed', str(job['seed']),
                     '--variant', job['variant'], '--split', split, '--output-dir', str(stage)]
        if split == 'test':
            arguments += ['--selection-lock', str(lock_path)]
        run_command(root, arguments, split + '_' + str(job['seed']))
        verify_replay(stage, job, split, lock, lock_path, ctx)
        folder.parent.mkdir(parents=True, exist_ok=True)
        ace.require(not folder.exists(), 'Replay destination appeared during verification')
        stage.rename(folder)
    return verify_replay(folder, job, split, lock, lock_path, ctx)


def verification(root, split, lock, lock_path, plan_path):
    checksum = file_digest(lock_path)
    results = []
    for job in lock['jobs']:
        ace.require(file_digest(lock_path) == checksum, 'Selection lock changed during replay')
        ace.verify_lock(lock_path, job['run_dir'], job['seed'], job['variant'])
        ctx = context(job)
        folder = root / f"replay_{split}/seed_{job['seed']}"
        first = ensure_replay(root, folder, job, split, lock, lock_path, ctx)
        row = {'seed': job['seed'], 'variant': job['variant'], 'split': split,
               'per_user_file': str(folder / 'per_user.csv'),
               'per_user_sha256': first['per_user_sha256'],
               'metrics_file': str(folder / 'metrics.json'),
               'metrics_sha256': file_digest(folder / 'metrics.json')}
        if split == 'validation':
            row['identical_to_original'] = True
        else:
            second_folder = root / f"replay_test_reloaded/seed_{job['seed']}"
            second = ensure_replay(root, second_folder, job, split, lock, lock_path, ctx)
            ace.require(first == second and first['per_user_sha256'] == second['per_user_sha256']
                        and file_digest(folder / 'metrics.json') == file_digest(second_folder / 'metrics.json'),
                        'Independent test processes produced different rankings or metrics')
            row.update(independent_replay_identical=True,
                       reloaded_file=str(second_folder / 'per_user.csv'),
                       reloaded_sha256=second['per_user_sha256'],
                       reloaded_metrics_file=str(second_folder / 'metrics.json'),
                       reloaded_metrics_sha256=file_digest(second_folder / 'metrics.json'))
        results.append(row)
        print(f"{split}: {len(results)}/{len(lock['jobs'])} complete replay audits", flush=True)
    report = {'schema': SCHEMA, 'protocol': ace.PROTOCOL, 'complete': True, 'split': split,
              'seed_ids': lock['seed_ids'], 'selection_lock_sha256': checksum,
              'verification_plan_sha256': file_digest(plan_path), 'driver_sha256': file_digest(__file__),
              'independent_process_per_replay': True, 'results': results}
    immutable_json(root / f'{split}_verification.json', report)
    return report


def expected_aggregate(root, split, lock, lock_path):
    values, hashes = [], dict(lock['artifact_sha256'])
    for job in lock['jobs']:
        summary = ace.read(Path(job['run_dir']) / f"seed_{job['seed']}/{job['variant']}/summary.json")
        metrics = summary['validation']
        if split == 'test':
            folder = root / f"replay_test/seed_{job['seed']}"
            metrics = ace.read(folder / 'metrics.json')['metrics']
            hashes.update({str(p): file_digest(p) for p in (folder / 'per_user.csv', folder / 'metrics.json')})
        values.append({'seed': job['seed'], 'metrics': metrics, **{k: summary[k] for k in
                      ('selected_epoch', 'epochs_completed', 'optimizer_steps', 'trainable_parameter_count')}})
    rows = []
    for group in values[0]['metrics']:
        for metric in ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50'):
            scores = [j['metrics'][group][metric] for j in values]
            rows.append({'variant': lock['jobs'][0]['variant'], 'stage': 'single', 'group': group,
                         'metric': metric, 'seed_ids': lock['seed_ids'], 'seed_count': 5,
                         'per_seed_scores': scores, 'mean': statistics.mean(scores),
                         'sample_std': statistics.stdev(scores)})
    return {'protocol': ace.PROTOCOL, 'split': split, 'seed_ids': lock['seed_ids'], 'aggregates': rows,
            'epoch_selections': [{k: v for k, v in j.items() if k != 'metrics'} for j in values],
            'selection_lock_sha256': file_digest(lock_path), 'selection_lock_file': str(lock_path),
            'selection_lock_fingerprint': lock['lock_fingerprint'], 'common': lock['common'],
            'artifact_sha256': hashes, 'all_artifact_hashes_verified': True, 'historical_test_known': True}


def ensure_aggregate(root, split, lock, lock_path):
    output = root / f'combined_{split}.json'
    expected = expected_aggregate(root, split, lock, lock_path)
    if not output.exists():
        pending = root / '.replay_pending'
        pending.mkdir(exist_ok=True)
        stage = pending / ('aggregate_' + split + '_' + uuid.uuid4().hex + '.json')
        arguments = ['aggregate', '--replica-runs', str(root / 'replica_runs.json'), '--split', split,
                     '--selection-lock', str(lock_path), '--output', str(stage)]
        if split == 'test':
            arguments += ['--replay-root', str(root / 'replay_test')]
        run_command(root, arguments, 'aggregate_' + split)
        _same(ace.read(stage), expected, 'Independent aggregate construction')
        ace.require(not output.exists(), 'Aggregate destination appeared during verification')
        stage.rename(output)
    _same(ace.read(output), expected, 'Existing aggregate recomputation')
    for filename, checksum in expected['artifact_sha256'].items():
        ace.require(file_digest(filename) == checksum, 'Aggregate input changed: ' + filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed five-replica root containing replica_runs.json')
    parser.add_argument('--resume', action='store_true', help='Audit and reuse complete immutable outputs')
    parser.add_argument('--through', choices=('validation', 'test'), default='test',
                        help='Stop after validation or finish both validation and two test replays')
    args = parser.parse_args()
    root = args.root.resolve()
    ace.require(root.is_dir(), 'Completed training root is required')
    plan_path = root / 'verification_plan.json'
    plan = make_plan(root)
    if plan_path.exists():
        ace.require(args.resume, 'Existing verification plan requires --resume')
        immutable_json(plan_path, plan)
    else:
        prior = ['five_seed_selection_lock.json', 'validation_verification.json', 'test_verification.json',
                 'combined_validation.json', 'combined_test.json', 'replay_validation',
                 'replay_test', 'replay_test_reloaded']
        ace.require(not any((root / name).exists() for name in prior),
                    'Prior verification outputs exist without this immutable plan; choose a fresh root')
        immutable_json(plan_path, plan)
    lock_path = root / 'five_seed_selection_lock.json'
    if not lock_path.exists():
        run_command(root, ['lock', '--replica-runs', str(root / 'replica_runs.json'),
                           '--output', str(lock_path)], 'selection_lock')
    lock = ace.verify_lock(lock_path)
    ace.require(lock['replica_runs'] == plan['replica_runs'] and lock['seed_ids'] == plan['seed_ids']
                and [job['seed'] for job in lock['jobs']] == plan['seed_ids'],
                'Selection lock scope differs from verification plan')
    verification(root, 'validation', lock, lock_path, plan_path)
    ensure_aggregate(root, 'validation', lock, lock_path)
    if args.through == 'test':
        verification(root, 'test', lock, lock_path, plan_path)
        ensure_aggregate(root, 'test', lock, lock_path)
    ace.require(make_plan(root) == plan, 'Frozen verification inputs changed during execution')
    ace.verify_lock(lock_path)
    print(f'Complete: {args.through} rankings, replay identities and five-seed aggregates audited.', flush=True)


if __name__ == '__main__':
    main()
