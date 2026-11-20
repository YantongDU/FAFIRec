"""Predeclared fixed global residual scale, evaluated without model selection.

The alpha=.05/beta=0/mu=0 recipe was frozen from the existing validation pilot
before new five-seed testing. Every test replay uses a fresh Python process.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from analyze_transfer import _same
from fafirec.runner import digest, file_digest, json_dump
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import final_runs as fr

PROTOCOL = 'fafirec_v4_effective_scale_control_v1'
VARIANT = 'global_no_pop_control'
PLAN_FILE = ROOT / 'configs/diagnostic_v4_scale_control.json'
BINDING_FILE = ROOT / 'configs/diagnostic_v4_scale_control_code_binding.json'
PARAMETERS = {'mode': 'global', 'l2': None, 'alpha': .05, 'beta': 0., 'alignment_weight': 0.}
GROUPS = ('all', 'cold', 'middle', 'warm')
METRICS = ('recall10', 'ndcg10', 'hit10', 'recall50', 'ndcg50', 'hit50')


def sources():
    return {**fr.source_identity(), Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def immutable(path, value):
    path = Path(path)
    if path.exists():
        ace.require(ace.read(path) == value, 'Immutable scale-control artifact differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_dump(path, value)


def verify_plan():
    plan = ace.read(PLAN_FILE)
    ace.require(plan['protocol'] == 'fafirec_v4_effective_scale_control_plan_v1'
        and plan['seed_ids'] == fr.SEEDS and plan['variant'] == VARIANT
        and plan['recipe'] == PARAMETERS and plan['source_hashes'] == fr.source_identity()
        and plan['selection_split'] == 'validation' and plan['no_new_test_used'] is True
        and plan['declared_before_new_test'] is True
        and plan['fingerprint'] == fr.fingerprint(plan), 'Predeclared fixed scale-control plan changed')
    ace.require(file_digest(plan['pilot_selection_file']) == plan['pilot_selection_sha256'], 'Original pilot changed')
    pilot = ace.read(plan['pilot_selection_file'])
    matches = [r for r in pilot['candidates'] if r['variant'] == plan['pilot_variant']]
    ace.require(len(matches) == 1 and all(matches[0][k] == v for k, v in PARAMETERS.items())
        and matches[0]['score'] == plan['pilot_validation_ndcg50'], 'Fixed control is absent from the prior pilot')
    return plan


def bind_code():
    verify_plan()
    binding = {'protocol': PROTOCOL, 'predeclared_plan_sha256': file_digest(PLAN_FILE),
        'source_hashes': sources(), 'parameters': PARAMETERS, 'new_optimization_performed': False}
    binding['fingerprint'] = digest(binding)
    immutable(BINDING_FILE, binding)
    return binding


def verify_binding():
    verify_plan()
    binding = ace.read(BINDING_FILE)
    ace.require(binding['protocol'] == PROTOCOL and binding['predeclared_plan_sha256'] == file_digest(PLAN_FILE)
        and binding['source_hashes'] == sources() and binding['parameters'] == PARAMETERS
        and binding['new_optimization_performed'] is False
        and binding['fingerprint'] == fr.fingerprint(binding), 'Scale-control code binding changed')
    return binding


def worker(root, seed, split, output_dir):
    verify_binding()
    root, out = Path(root).resolve(), Path(output_dir).resolve()
    ace.require(split in {'validation', 'test'} and seed in fr.SEEDS and not out.exists(), 'Fresh fixed-control output required')
    lock_path = root / 'five_seed_selection_lock.json'
    lock = fr.verify_lock(lock_path)
    job = next(j for j in lock['jobs'] if j['seed'] == seed)
    manifest, _summary, cache_plan, training, validation = fr.load_run(job['run_dir'])
    del training
    if split == 'validation':
        records = ace.read(Path(job['run_dir']) / 'cache/records_validation.json')
        values = validation['sem'] + PARAMETERS['alpha'] * validation['cf']
    else:
        del validation
        records, histories, data = fr.reconstruct(job['run_dir'], split, manifest, cache_plan)
        relations = torch.load(Path(job['run_dir']) / 'cache/relations.pt', map_location='cpu', weights_only=True)
        cf = fr.fusion.score_operator(relations['B'], histories, data['available'])
        values = data['sem'] + PARAMETERS['alpha'] * cf
    metrics = fr.write_scored(out, values.numpy(), records, manifest['evaluation_config'], seed,
        VARIANT, split, 'per_user.csv')
    result = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed, 'variant': VARIANT,
        'stage': 'fusion', 'split': split, 'parameters': PARAMETERS, 'metrics': metrics,
        'per_user_sha256': file_digest(out / 'per_user.csv'),
        'candidate_policy': manifest['evaluation_config']['candidate_policy'],
        'selection_lock_sha256': file_digest(lock_path), 'predeclared_plan_sha256': file_digest(PLAN_FILE),
        'code_binding_sha256': file_digest(BINDING_FILE)}
    json_dump(out / 'metrics.json', result)


def contained(root, path):
    path = Path(path).resolve()
    ace.require(path != root and path.is_relative_to(root), 'Scale-control output escapes experiment root')
    return path


def verify_output(folder, seed, split, lock_path, manifest):
    value = ace.read(folder / 'metrics.json')
    ace.require(value['protocol'] == PROTOCOL and value['seed'] == seed and value['variant'] == VARIANT
        and value['split'] == split and value['parameters'] == PARAMETERS
        and value['fingerprint'] == manifest['fingerprint']
        and value['selection_lock_sha256'] == file_digest(lock_path)
        and value['predeclared_plan_sha256'] == file_digest(PLAN_FILE)
        and value['code_binding_sha256'] == file_digest(BINDING_FILE)
        and value['per_user_sha256'] == file_digest(folder / 'per_user.csv'), 'Fixed-control replay identity changed')
    return value


def ensure_worker(root, aux, job, split, label, lock_path, manifest):
    folder = contained(aux, aux / label / f"seed_{job['seed']}")
    if not folder.exists():
        pending = contained(aux, aux / '.pending')
        pending.mkdir(exist_ok=True)
        stage = contained(aux, pending / f"{label}_{job['seed']}_{uuid.uuid4().hex}")
        logs = aux / 'logs'
        logs.mkdir(exist_ok=True)
        log = logs / f"{label}_{job['seed']}_{uuid.uuid4().hex}.log"
        args = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(root),
            '--seed', str(job['seed']), '--split', split, '--output-dir', str(stage)]
        with log.open('xb') as stream:
            result = subprocess.run(args, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ace.require(result.returncode == 0, 'Fixed-control replay failed; inspect ' + str(log))
        verify_output(stage, job['seed'], split, lock_path, manifest)
        folder.parent.mkdir(parents=True, exist_ok=True)
        ace.require(not folder.exists(), 'Fixed-control destination appeared')
        contained(aux, stage).rename(contained(aux, folder))
    return folder, verify_output(folder, job['seed'], split, lock_path, manifest)


def run(root):
    """Audit and evaluate the fixed auxiliary control after the five-seed lock."""
    verify_binding()
    root = Path(root).resolve()
    lock_path = root / 'five_seed_selection_lock.json'
    lock = fr.verify_lock(lock_path)
    aux = contained(root, root / 'effective_scale_control')
    aux.mkdir(exist_ok=True)
    jobs = sorted(lock['jobs'], key=lambda j: j['seed'])
    manifest = {'protocol': PROTOCOL, 'root': str(root), 'seed_ids': fr.SEEDS, 'jobs': jobs,
        'parameters': PARAMETERS, 'source_hashes': sources(), 'selection_lock_sha256': file_digest(lock_path),
        'predeclared_plan_sha256': file_digest(PLAN_FILE), 'code_binding_sha256': file_digest(BINDING_FILE),
        'selected_on_test': False, 'new_optimization_performed': False}
    manifest['fingerprint'] = digest(manifest)
    immutable(aux / 'master_plan.json', manifest)
    reports = {'validation': [], 'test': []}
    artifacts = {str(p): file_digest(p) for p in (PLAN_FILE, BINDING_FILE, lock_path, aux / 'master_plan.json', Path(__file__))}
    identity = evaluation = None
    for job in jobs:
        original = ace.read(Path(job['run_dir']) / 'manifest.json')
        ace.require(original['seed'] == job['seed'], 'Fixed-control seed mapping differs')
        identity = original['identity'] if identity is None else identity
        evaluation = original['evaluation_config'] if evaluation is None else evaluation
        ace.require(original['identity'] == identity and original['evaluation_config'] == evaluation, 'Common inputs differ')
        for split in ('validation', 'test'):
            folder, first = ensure_worker(root, aux, job, split, split, lock_path, original)
            row = {'seed': job['seed'], 'metrics': first['metrics'], 'per_user_file': str(folder / 'per_user.csv'),
                'per_user_sha256': first['per_user_sha256'], 'metrics_file': str(folder / 'metrics.json'),
                'metrics_sha256': file_digest(folder / 'metrics.json')}
            paths = [folder / 'per_user.csv', folder / 'metrics.json']
            if split == 'test':
                second_folder, second = ensure_worker(root, aux, job, split, 'test_reloaded', lock_path, original)
                ace.require(first == second and file_digest(folder / 'metrics.json') == file_digest(second_folder / 'metrics.json'),
                    'Fixed-control independent test rankings or metrics differ')
                row.update(independent_replay_identical=True, reloaded_file=str(second_folder / 'per_user.csv'),
                    reloaded_sha256=second['per_user_sha256'], reloaded_metrics_file=str(second_folder / 'metrics.json'),
                    reloaded_metrics_sha256=file_digest(second_folder / 'metrics.json'))
                paths += [second_folder / 'per_user.csv', second_folder / 'metrics.json']
            artifacts.update({str(p): file_digest(p) for p in paths})
            reports[split].append(row)
        print(f"Fixed global scale: completed seed {job['seed']}", flush=True)
    for split, values in reports.items():
        rows = []
        for group in GROUPS:
            for metric in METRICS:
                scores = [r['metrics'][group][metric] for r in values]
                rows.append({'variant': VARIANT, 'stage': 'fusion', 'group': group, 'metric': metric,
                    'seed_ids': fr.SEEDS, 'seed_count': 5, 'per_seed_scores': scores,
                    'mean': statistics.mean(scores), 'sample_std': statistics.stdev(scores)})
        value = {'protocol': PROTOCOL, 'split': split, 'seed_ids': fr.SEEDS, 'aggregates': rows,
            'parameters': PARAMETERS, 'results': values, 'selection_lock_sha256': file_digest(lock_path),
            'predeclared_plan_sha256': file_digest(PLAN_FILE), 'code_binding_sha256': file_digest(BINDING_FILE),
            'common': {'identity': identity, 'evaluation_config': evaluation}, 'source_hashes': sources(),
            'artifact_sha256': artifacts, 'all_artifact_hashes_verified': True, 'selected_on_test': False}
        immutable(aux / f'combined_{split}.json', value)
    fr.verify_lock(lock_path)
    return ace.read(aux / 'combined_test.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('plan', help='Verify the fixed declaration and bind this frozen implementation')
    q = sub.add_parser('run'); q.add_argument('--root', type=Path, required=True)
    q = sub.add_parser('worker'); q.add_argument('--root', type=Path, required=True)
    q.add_argument('--seed', type=int, required=True); q.add_argument('--split', choices=('validation', 'test'), required=True)
    q.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.command == 'plan':
        bind_code(); print('Verified fixed validation declaration and bound implementation source.', flush=True)
    elif args.command == 'worker':
        worker(args.root, args.seed, args.split, args.output_dir)
    else:
        run(args.root)


if __name__ == '__main__':
    main()
