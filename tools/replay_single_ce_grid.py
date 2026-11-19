"""Replay all five single-CF CE models with the training numerical settings."""
import argparse
import os
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from tools import single_ce_baseline as baseline
from fafirec.runner import file_digest, json_dump, seed_everything
from analyze_research import audit_rows
from analyze_transfer import _same


def audited_lock(master_config, root, output):
    """Audit CSV recomputation using the core audit's summation tolerance.

    The frozen trainer's lock entry point compares floating aggregates with ==.
    Independent NumPy/Python logarithms can differ in their final bits. Keep all
    source/checkpoint/CSV identities exact, and record the measured roundoff.
    """
    master_path, master = baseline._master(master_config)
    seeds = sorted(master['seeds'])
    baseline.require(len(seeds) == 5 and not output.exists(), 'New complete five-seed lock required')
    baseline.require({p.name for p in root.glob('seed_*') if p.is_dir()} == {f'seed_{s}' for s in seeds},
                     'Missing or extra seed directory')
    artifacts, jobs, errors, shared = {}, [], {}, None
    for seed in seeds:
        folder = root / f'seed_{seed}'
        records, _, _, manifest, summary, _, _ = baseline._load_run(folder, seed)
        current = (manifest['runtime'], manifest['identity'])
        baseline.require(shared is None or shared == current, 'Seed runtime/data differs')
        baseline.require(manifest['master_config_file'] == str(master_path), 'Master configuration differs')
        shared = current
        computed = audit_rows(folder / 'validation.csv', records, manifest['evaluation_config'], seed,
                              'single_cf_ce', 'single', 'single_cf_ce', 'validation', manifest['identity']['items'])
        _same(computed, summary['validation'], 'Independent single CE validation recomputation')
        errors[str(seed)] = max(abs(computed[g][m] - summary['validation'][g][m])
                                for g in computed for m in computed[g])
        for name in ('manifest.json', 'best.pt', 'last.pt', 'summary.json', 'validation.csv', 'initial_validation.json'):
            path = folder / name
            artifacts[str(path)] = file_digest(path)
        jobs.append({'seed': seed, 'run_fingerprint': manifest['fingerprint'],
                     'selected_epoch': summary['selected_epoch'], 'validation': summary['validation']})
    value = {'protocol': baseline.PROTOCOL, 'selection_split': 'validation', 'seed_ids': seeds,
        'master_config_file': str(master_path), 'master_config_sha256': file_digest(master_path),
        'master_config': master, 'replica_root': str(root), 'source_hashes': baseline.source_identity(),
        'runtime': shared[0], 'validation_artifact_sha256': artifacts, 'jobs': jobs,
        'mean_validation_ndcg50': statistics.mean(j['validation']['all']['ndcg50'] for j in jobs),
        'new_test_used_for_selection': False, 'historical_test_known': True,
        'selection': 'independent single-branch CE baseline; each seed selects its own validation-best epoch',
        'lock_driver_sha256': file_digest(__file__), 'csv_recomputation_max_absolute_error': errors,
        'numeric_audit_tolerance': 'same as analyze_transfer._same: rtol1e-10, atol1e-12; all file hashes exact'}
    value['lock_fingerprint'] = baseline.digest(value)
    json_dump(output, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-config', type=Path, required=True)
    parser.add_argument('--replica-root', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--selection-lock', type=Path, required=True)
    args = parser.parse_args()
    root, lock = args.replica_root.resolve(), args.selection_lock.resolve()
    _, master = baseline._master(args.master_config)
    seeds = sorted(master['seeds'])
    baseline.require(len(seeds) == 5, 'Complete five-seed grid required')
    if args.split == 'validation':
        audited_lock(args.master_config, root, lock)
    else:
        validated = baseline.read(root / 'validation_verification.json')
        baseline.require(validated['complete'] is True and validated['seed_ids'] == seeds
                         and validated['selection_lock_sha256'] == file_digest(lock),
                         'Complete validation replay is required before test')
        for row in validated['results']:
            baseline.require(file_digest(row['per_user_file']) == row['per_user_sha256'],
                             'Independent validation replay changed')
    checksum, results = file_digest(lock), []
    for seed in seeds:
        baseline.require(file_digest(lock) == checksum, 'Selection lock changed during replay')
        # Training enables deterministic algorithms. Restore that setting in
        # every independent evaluation process/session as well.
        seed_everything(seed + 1000)
        folder = root / f'replay_{args.split}/seed_{seed}'
        first = baseline.replay(root / f'seed_{seed}', seed, args.split, folder,
                                lock if args.split == 'test' else None)
        row = {'seed': seed, 'per_user_file': str(folder / 'per_user.csv'),
               'per_user_sha256': first['per_user_sha256']}
        if args.split == 'test':
            seed_everything(seed + 1000)
            reloaded = root / f'replay_test_reloaded/seed_{seed}'
            second = baseline.replay(root / f'seed_{seed}', seed, 'test', reloaded, lock)
            baseline.require(first['per_user_sha256'] == second['per_user_sha256']
                             and first['metrics'] == second['metrics'], 'Independent test reload differs')
            row.update(independent_replay_identical=True, reloaded_file=str(reloaded / 'per_user.csv'),
                       reloaded_sha256=second['per_user_sha256'])
        results.append(row)
        json_dump(root / f'{args.split}_verification.json', {'complete': len(results) == len(seeds),
            'split': args.split, 'seed_ids': seeds, 'selection_lock_sha256': checksum,
            'driver_sha256': file_digest(__file__),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'results': results})
        print(f'{args.split}: {len(results)}/{len(seeds)} models verified', flush=True)
    baseline.aggregate(root, args.split, root / f'combined_{args.split}.json', lock,
                       root / 'replay_test' if args.split == 'test' else None)
    print(f'{args.split}: all rankings and aggregate metrics independently audited', flush=True)


if __name__ == '__main__':
    main()
