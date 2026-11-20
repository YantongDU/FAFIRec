"""Verify ACE replicas and finalize aggregates by explicit metric row keys.

This entry point preserves the frozen v1 verifier. It changes only aggregate
comparison order: every complete row and every other field remain audited.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from analyze_transfer import _same
from fafirec.runner import digest, file_digest
from tools import ace2026_experiment as ace
from tools import verify_ace2026_replicas as v1

SCHEMA = 'ace2026_complete_aggregate_key_audit_v1'
KEYS = ('variant', 'stage', 'group', 'metric')
GROUPS = ('all', 'cold', 'middle', 'warm')
METRICS = ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50')


def contained(root, path):
    resolved = Path(path).resolve()
    ace.require(resolved != root and resolved.is_relative_to(root),
                'Verification path must stay inside the experiment root: ' + str(resolved))
    return resolved


def preflight(root, lock):
    # The frozen verifier can create or rename beneath these directories.
    for name in ('.replay_pending', 'replay_logs', 'replay_validation', 'replay_test',
                 'replay_test_reloaded', 'combined_validation.json', 'combined_test.json',
                 'validation_verification.json', 'test_verification.json'):
        contained(root, root / name)
    for job in lock['jobs']:
        for name in ('replay_validation', 'replay_test', 'replay_test_reloaded'):
            contained(root, root / name / f"seed_{job['seed']}")


def canonical(report, expected_keys):
    ace.require(isinstance(report, dict) and isinstance(report.get('aggregates'), list),
                'Complete aggregate object required')
    indexed = {}
    for row in report['aggregates']:
        ace.require(isinstance(row, dict) and all(k in row for k in KEYS), 'Aggregate row identity missing')
        key = tuple(row[k] for k in KEYS)
        ace.require(key not in indexed, 'Duplicate aggregate metric row: ' + str(key))
        indexed[key] = row
    ace.require(set(indexed) == set(expected_keys), 'Missing or extra aggregate metric rows')
    result = copy.deepcopy(report)
    result['aggregates'] = [copy.deepcopy(indexed[key]) for key in sorted(indexed)]
    return result


def compare_complete(actual, expected, lock):
    variant = lock['common']['config']['variants'][0]['name']
    expected_keys = {(variant, 'single', group, metric) for group in GROUPS for metric in METRICS}
    ace.require(actual.get('seed_ids') == expected.get('seed_ids') == lock['seed_ids'],
                'Aggregate seed order differs')
    _same(canonical(actual, expected_keys), canonical(expected, expected_keys),
          'Complete aggregate object after metric-key ordering')
    for filename, checksum in expected['artifact_sha256'].items():
        ace.require(file_digest(filename) == checksum, 'Aggregate input artifact changed: ' + filename)
    return [list(key) for key in sorted(expected_keys)]


def finalize(root, split, lock, lock_path):
    output = contained(root, root / f'combined_{split}.json')
    expected = v1.expected_aggregate(root, split, lock, lock_path)
    if not output.exists():
        pending = contained(root, root / '.replay_pending')
        pending.mkdir(exist_ok=True)
        prior = [contained(root, p) for p in pending.glob(f'aggregate_{split}_*.json')]
        stage = None
        if len(prior) == 1:
            # A complete native output left by v1's order-only failure is reusable.
            try:
                compare_complete(ace.read(prior[0]), expected, lock)
                stage = prior[0]
            except (ValueError, KeyError, TypeError, OSError) as error:
                print(f'Incomplete or invalid pending aggregate retained: {prior[0]} ({error})', flush=True)
        if stage is None:
            stage = contained(root, pending / f'finalize_aggregate_{split}_{uuid.uuid4().hex}.json')
            arguments = ['aggregate', '--replica-runs', str(root / 'replica_runs.json'), '--split', split,
                         '--selection-lock', str(lock_path), '--output', str(stage)]
            if split == 'test':
                arguments += ['--replay-root', str(root / 'replay_test')]
            v1.run_command(root, arguments, 'finalize_aggregate_' + split)
        compare_complete(ace.read(stage), expected, lock)
        source, destination = contained(root, stage), contained(root, output)
        ace.require(source.is_file() and not destination.exists(), 'Aggregate rename requires a new contained destination')
        source.rename(destination)
    coverage = compare_complete(ace.read(output), expected, lock)
    return {'split': split, 'file': str(output), 'sha256': file_digest(output),
            'row_count': len(coverage), 'metric_row_keys': coverage,
            'all_fields_checked': True, 'only_row_order_normalized': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed five-replica training root')
    parser.add_argument('--resume', action='store_true', help='Audit and reuse existing immutable verification artifacts')
    args = parser.parse_args()
    root = args.root.resolve()
    ace.require(root.is_dir(), 'Completed experiment root required')
    initial_scope = v1.make_plan(root)
    preflight(root, {'jobs': [{'seed': seed} for seed in initial_scope['seed_ids']]})
    original_plan = contained(root, root / 'verification_plan.json')
    if original_plan.exists():
        ace.require(args.resume, 'Existing verification outputs require --resume')
    else:
        # Avoid the v1 test aggregate path entirely for a fresh experiment.
        command = [sys.executable, '-u', str(Path(v1.__file__).resolve()), '--root', str(root), '--through', 'validation']
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ace.require(result.returncode == 0, 'Validation initialization failed; inspect the v1 verification output')
    original = ace.read(original_plan)
    ace.require(v1.make_plan(root) == original and original['driver_sha256'] == file_digest(v1.__file__),
                'Frozen v1 verification plan or inputs changed')
    lock_path = contained(root, root / 'five_seed_selection_lock.json')
    lock = ace.verify_lock(lock_path)
    ace.require(lock['replica_runs'] == original['replica_runs'] and lock['seed_ids'] == original['seed_ids']
                and [job['seed'] for job in lock['jobs']] == original['seed_ids'], 'Lock scope differs from v1 plan')
    preflight(root, lock)
    plan_path = contained(root, root / 'aggregate_finalization_plan.json')
    plan = {'schema': SCHEMA, 'root': str(root), 'seed_ids': lock['seed_ids'],
            'driver_sha256': file_digest(__file__), 'v1_driver_sha256': file_digest(v1.__file__),
            'verification_plan_sha256': file_digest(original_plan),
            'selection_lock_sha256': file_digest(lock_path),
            'replica_runs_sha256': file_digest(root / 'replica_runs.json'),
            'source_hashes': ace.source_identity(),
            'comparison_policy': 'Unique complete aggregate rows keyed by variant/stage/group/metric; no fields dropped; all other lists keep original order'}
    plan['fingerprint'] = digest(plan)
    if plan_path.exists():
        ace.require(args.resume, 'Existing finalization plan requires --resume')
    v1.immutable_json(plan_path, plan)
    reports = []
    for split in ('validation', 'test'):
        # Completed rankings are checked and reused; missing replays use fresh
        # processes through v1, without changing its source or immutable plan.
        v1.verification(root, split, lock, lock_path, original_plan)
        reports.append(finalize(root, split, lock, lock_path))
    ace.require(v1.make_plan(root) == original, 'Frozen verification inputs changed during finalization')
    ace.verify_lock(lock_path)
    paths = [original_plan, lock_path, plan_path, Path(__file__).resolve(), Path(v1.__file__).resolve(),
             root / 'validation_verification.json', root / 'test_verification.json',
             root / 'combined_validation.json', root / 'combined_test.json']
    audit = {'schema': SCHEMA, 'complete': True, 'seed_ids': lock['seed_ids'],
             'finalization_plan_sha256': file_digest(plan_path),
             'driver_sha256': file_digest(__file__), 'v1_driver_sha256': file_digest(v1.__file__),
             'verification_plan_sha256': file_digest(original_plan),
             'selection_lock_sha256': file_digest(lock_path), 'aggregates': reports,
             'artifact_sha256': {str(path): file_digest(path) for path in paths}}
    v1.immutable_json(contained(root, root / 'aggregate_finalization_audit.json'), audit)
    print('Complete: five-seed validation/test replays and all aggregate fields verified by explicit metric keys.', flush=True)


if __name__ == '__main__':
    main()
