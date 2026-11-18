"""Replay one complete, frozen experiment grid without selecting on test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_research import replay
from fafirec.research_runner import _fingerprint, _read, _require, normalize_config, source_identity
from fafirec.runner import file_digest, json_dump


def identity(root):
    manifest = _read(root / 'manifest.json')
    _require(manifest['fingerprint'] == _fingerprint(manifest), 'Manifest changed')
    _require(source_identity() == manifest['source_hashes'], 'Frozen source changed')
    cfg = normalize_config(manifest['config'])
    stages = ['expert', 'router'] if cfg['train_router'] else ['expert']
    expected = {(s, v['name'], stage) for s in cfg['seeds'] for v in cfg['variants'] for stage in stages}
    summaries = _read(root / 'summary.json')
    _require(len(summaries) == len(expected) and {(r['seed'], r['variant'], r['stage']) for r in summaries} == expected,
             'Complete configured grid required')
    selected = _read(root / 'selection.json')
    _require(selected['fingerprint'] == manifest['fingerprint'] and selected['split'] == 'validation', 'Selection differs')
    hashes = {str(root / name): file_digest(root / name) for name in ('manifest.json', 'selection.json', 'summary.json')}
    for seed, variant, stage in expected:
        folder = root / f'seed_{seed}/{variant}/{stage}'
        for name in ('best.pt', 'summary.json', 'validation.csv'):
            hashes[str(folder / name)] = file_digest(folder / name)
    return manifest, cfg, sorted(expected), selected, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--selection-lock', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cuda')
    parser.add_argument('--repeat', action='store_true', help='Independently reload and replay every job twice')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest, _, jobs, selected, hashes = identity(root)
    if args.output_dir.exists():
        raise FileExistsError('Choose a new replay output directory')
    if args.split == 'validation':
        if args.selection_lock.exists():
            raise FileExistsError('Choose a new selection-lock path')
        # This lock is written before validation replays and before any test
        # evaluation; selection.json was computed exclusively on validation.
        lock = {'fingerprint': manifest['fingerprint'], 'selection_split': 'validation',
            'selected_variant': selected['selected_variant'], 'selected_stage': selected['selected_stage'],
            'historical_test_known': True, 'new_test_used_for_selection': False,
            'jobs': jobs, 'artifact_sha256': hashes,
            'validation_replay_dir': str(args.output_dir.resolve())}
        args.selection_lock.parent.mkdir(parents=True, exist_ok=True)
        json_dump(args.selection_lock, lock)
    else:
        lock = _read(args.selection_lock)
        _require(lock['fingerprint'] == manifest['fingerprint'] and lock['selection_split'] == 'validation'
                 and lock['new_test_used_for_selection'] is False and lock['artifact_sha256'] == hashes
                 and lock['jobs'] == [list(j) for j in jobs]
                 and lock['selected_variant'] == selected['selected_variant']
                 and lock['selected_stage'] == selected['selected_stage'], 'Validation selection lock changed')
        validated = _read(Path(lock['validation_replay_dir']) / 'verification.json')
        _require(validated['complete'] is True and validated['expected_jobs'] == len(jobs)
                 and validated['fingerprint'] == manifest['fingerprint'] and validated['split'] == 'validation'
                 and validated['selection_lock_sha256'] == file_digest(args.selection_lock),
                 'Complete exact validation replay required before test')
        for seed, variant, stage in jobs:
            folder = Path(lock['validation_replay_dir']) / f'seed_{seed}/{variant}/{stage}'
            metric = _read(folder / 'metrics.json')
            _require(metric['fingerprint'] == manifest['fingerprint'] and metric['identical_to_original'] is True
                     and metric['per_user_sha256'] == file_digest(folder / 'per_user.csv')
                     == hashes[str(root / f'seed_{seed}/{variant}/{stage}/validation.csv')],
                     'Validation replay artifact changed')
    lock_hash = file_digest(args.selection_lock)
    results = []
    for seed, variant, stage in jobs:
        _require(file_digest(args.selection_lock) == lock_hash, 'Selection lock changed during replay')
        for path, checksum in hashes.items():
            _require(file_digest(path) == checksum, 'Frozen selected artifact changed: ' + path)
        folder = args.output_dir / f'seed_{seed}/{variant}/{stage}'
        first = replay(root, seed, variant, stage, args.split, folder, args.device)
        item = {'seed': seed, 'variant': variant, 'stage': stage, 'per_user_sha256': first['per_user_sha256']}
        if args.repeat:
            second = replay(root, seed, variant, stage, args.split,
                            args.output_dir / f'independent_reload/seed_{seed}/{variant}/{stage}', args.device)
            _require(first['per_user_sha256'] == second['per_user_sha256'] and first['metrics'] == second['metrics'],
                     'Independent checkpoint replay differs')
            item.update(independent_replay_identical=True, second_per_user_sha256=second['per_user_sha256'])
        results.append(item)
        json_dump(args.output_dir / 'verification.json', {'fingerprint': manifest['fingerprint'],
            'split': args.split, 'selection_lock_sha256': lock_hash, 'complete': len(results) == len(jobs),
            'expected_jobs': len(jobs), 'results': results})
        print(json.dumps({'split': args.split, 'completed': len(results), 'total': len(jobs), **item}), flush=True)


if __name__ == '__main__':
    main()
