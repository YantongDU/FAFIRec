"""Bounded five-seed coverage/inner-teacher component queue; never score test."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from analyze_transfer import _same
from experiments.diagnostic_v4 import coverage_probe as coverage, inner_teacher as teacher
from fafirec.runner import digest, file_digest, json_dump
from tools import ace2026_experiment as ace

SEEDS = [2025, 2026, 2027, 2028, 2029]
SCHEMA = 'diagnostic_v4_five_seed_component_dispatch_v1'


def sources():
    return {**coverage.source_identity(), **teacher.source_identity(),
            Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def immutable(path, value):
    if path.exists():
        ace.require(ace.read(path) == value, 'Immutable component artifact differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_dump(path, value)


def make_plan(root, original_config, reference_root, parallel):
    cfg = ace.load_config(original_config)
    ace.require((cfg['epochs'], cfg['min_epochs'], cfg['eval_interval'], cfg['patience'], cfg['batch_size'])
                == (400, 30, 5, 3, 256), 'Fixed original ACE training budget required')
    ace.require(len(cfg['variants']) == 1 and {k: cfg['variants'][0][k] for k in ('reg_weight', 'scale', 'lr')}
                == {'reg_weight': 500., 'scale': .1, 'lr': .0005}, 'Fixed spectral/lr recipe required')
    ace.require((cfg['d_model'], cfg['n_layers'], cfg['n_heads'], cfg['inner_size']) == (128, 2, 2, 256)
                and cfg['weight_decay'] == 0 and cfg['grad_clip'] is None, 'Fixed native ACE architecture/optimizer required')
    jobs, hashes = [], {str(original_config): file_digest(original_config), cfg['svd_cache']: file_digest(cfg['svd_cache'])}
    for seed in SEEDS:
        reference = reference_root / f'seed_{seed}/run'
        native = ace.read(reference / 'manifest.json')
        ace.require(native['fingerprint'] == ace.fingerprint(native) and native['source_hashes'] == ace.source_identity()
                    and native['config']['seeds'] == [seed], 'Frozen per-seed reference identity differs')
        hashes[str(reference / 'manifest.json')] = file_digest(reference / 'manifest.json')
        configs = {'coverage': ace.normalize_config({**cfg, 'seeds': [seed],
            'output_dir': str(root / f'seed_{seed}/coverage')}),
            'teacher': teacher.normalize_config({'seed': seed, 'epochs': 135,
                'reference_seed': seed, 'reference_run': str(reference), 'reference_variant': 'ace_selected',
                'svd_cache': cfg['svd_cache'], 'device': cfg['device'], 'threads': cfg['threads'],
                'output_dir': str(root / f'seed_{seed}/teacher')})}
        for component, config in configs.items():
            config_file = root / f'seed_{seed}/{component}_config.json'
            jobs.append({'id': f'seed_{seed}_{component}', 'seed': seed, 'component': component,
                         'config': config, 'config_file': str(config_file)})
    value = {'schema': SCHEMA, 'seed_ids': SEEDS, 'root': str(root), 'parallel': parallel,
        'jobs': jobs, 'source_hashes': sources(), 'input_artifact_sha256': hashes,
        'coverage_mode': 'all_windows', 'teacher_epochs': 135, 'test_evaluated': False,
        'scope': 'Coverage selects epoch on outer validation only; teacher is fixed-budget train-internal only.'}
    value['fingerprint'] = digest(value)
    return value


def verify_plan(plan):
    ace.require(plan['schema'] == SCHEMA and plan['fingerprint'] == digest({k: v for k, v in plan.items() if k != 'fingerprint'})
                and plan['source_hashes'] == sources(), 'Component source/plan changed')
    for filename, checksum in plan['input_artifact_sha256'].items():
        ace.require(file_digest(filename) == checksum, 'Frozen component input changed: ' + filename)
    for job in plan['jobs']:
        ace.require(ace.read(job['config_file']) == job['config'], 'Per-component config changed')


def complete_receipt(job, plan):
    """Fully check a finished job before marking it reusable without training."""
    out = Path(job['config']['output_dir'])
    manifest = ace.read(out / 'manifest.json')
    summary = ace.read(out / 'summary.json')
    paths = [out / 'manifest.json', out / 'summary.json']
    if job['component'] == 'coverage':
        ace.require(manifest['config'] == job['config'] and manifest['modes'] == ['all_windows']
                    and manifest['fingerprint'] == ace.fingerprint(manifest)
                    and manifest['source_hashes'] == coverage.source_identity(), 'Completed coverage manifest differs')
        ace.require(len(summary) == 1 and summary[0]['mode'] == 'all_windows', 'Coverage summary differs')
        value = summary[0]
        folder = out / 'all_windows'
        ace.require(ace.read(folder / 'summary.json') == value, 'Coverage root/leaf summaries differ')
        best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        ace.require(last['finished'] and last['epoch'] == value['epochs_completed']
                    and last['selected_epoch'] == best['selected_epoch'] == value['selected_epoch']
                    and best['fingerprint'] == last['fingerprint'] == manifest['fingerprint']
                    and best['seed'] == last['seed'] == job['seed']
                    and best['selected_metric'] == value['selected_validation_metric']
                    and file_digest(folder / 'best.pt') == value['checkpoint_sha256']
                    and file_digest(folder / 'validation.csv') == value['per_user_sha256'], 'Completed coverage checkpoint differs')
        _same(best['validation'], value['validation'], 'Completed coverage selected metrics')
        ace.require(ace.read(folder / 'training.json') == last['logs'], 'Coverage log differs from checkpoint')
        ace.require(set(best['model']) == set(last['best_snapshot']['model'])
                    and all(torch.equal(t, last['best_snapshot']['model'][k]) for k, t in best['model'].items()),
                    'Coverage selected snapshot differs')
        paths += [out / 'selection.json'] + [folder / name for name in
            ('summary.json', 'best.pt', 'last.pt', 'validation.csv', 'training.json', 'initial_validation.json')]
    else:
        ace.require(manifest['config'] == job['config'] and manifest['fingerprint'] == teacher.fingerprint(manifest)
                    and manifest['source_hashes'] == teacher.source_identity()
                    and summary['fingerprint'] == manifest['fingerprint'] and summary['seed'] == job['seed']
                    and summary['epochs'] == 135, 'Completed teacher identity differs')
        for key, name in (('teacher_sha256', 'teacher.pt'), ('last_sha256', 'last.pt'), ('training_sha256', 'training.json')):
            ace.require(file_digest(out / name) == summary[key], 'Completed teacher artifact changed: ' + name)
        final = torch.load(out / 'teacher.pt', map_location='cpu', weights_only=True)
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        ace.require(final['epoch'] == last['epoch'] == 135 and last['finished']
                    and final['fingerprint'] == last['fingerprint'] == manifest['fingerprint']
                    and ace.read(out / 'training.json') == last['logs']
                    and set(final['model']) == set(last['model'])
                    and all(torch.equal(t, last['model'][k]) for k, t in final['model'].items()),
                    'Teacher fixed final checkpoint differs')
        paths += [out / name for name in ('teacher.pt', 'last.pt', 'training.json', 'split_manifest.json')]
    value = {'schema': SCHEMA, 'job_id': job['id'], 'seed': job['seed'], 'component': job['component'],
        'master_fingerprint': plan['fingerprint'], 'component_fingerprint': manifest['fingerprint'],
        'job_config_sha256': file_digest(job['config_file']),
        'artifact_sha256': {str(path): file_digest(path) for path in paths}, 'test_evaluated': False}
    immutable(out / 'completion_receipt.json', value)
    return value


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def dispatch(root, plan, resume):
    status_path = root / 'dispatch_status.json'
    if status_path.exists():
        prior = ace.read(status_path)
        ace.require(not any(process_alive(pid) for pid in prior.get('active', {}).values())
                    and not (prior.get('master_pid') != os.getpid() and process_alive(prior.get('master_pid', 0))),
                    'Previous component dispatcher or worker is still running')
    pending, completed, active, failed = [], [], {}, []
    receipts = {}
    for job in plan['jobs']:
        out = Path(job['config']['output_dir'])
        if (out / 'summary.json').exists():
            receipts[job['id']] = complete_receipt(job, plan)
            completed.append(job['id'])
        else:
            pending.append(job)
    try:
        while pending or active:
            while pending and len(active) < plan['parallel'] and not failed:
                job = pending.pop(0)
                log = Path(job['config_file']).parent / f"{job['component']}{'_resume' if resume else ''}.log"
                stream = log.open('ab')
                command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                    '--plan', str(root / 'master_plan.json'), '--job-id', job['id']]
                if resume: command.append('--resume')
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                active[job['id']] = (child, stream, job)
                print(f"Started {job['id']} PID {child.pid}", flush=True)
            for key, (child, stream, job) in list(active.items()):
                code = child.poll()
                if code is not None:
                    stream.close()
                    del active[key]
                    if code == 0:
                        receipts[key] = complete_receipt(job, plan)
                        completed.append(key)
                    else:
                        failed.append(key)
                    print(f'Finished {key} exit {code}', flush=True)
            json_dump(status_path, {'master_pid': os.getpid(), 'plan_fingerprint': plan['fingerprint'],
                'active': {key: value[0].pid for key, value in active.items()}, 'completed': completed,
                'failed': failed, 'pending': [j['id'] for j in pending]})
            if failed and not active:
                raise RuntimeError('Component failed; inspect logs before resuming')
            if active: time.sleep(2)
    finally:
        for child, stream, _ in active.values():
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            stream.close()
    verify_plan(plan)
    ace.require(len(completed) == len(plan['jobs']) == 10, 'Incomplete ten-component grid')
    result = {'schema': SCHEMA, 'complete': True, 'seed_ids': SEEDS,
        'plan_sha256': file_digest(root / 'master_plan.json'), 'plan_fingerprint': plan['fingerprint'],
        'receipts': receipts, 'test_evaluated': False}
    immutable(root / 'completion.json', result)
    print('All ten five-seed coverage/teacher components completed and audited.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output-dir', type=Path, default=Path('runs/diagnostic_v4_final_components'))
    run.add_argument('--ace-config', type=Path, default=Path('E:/论文/FAFIRec_ACE2026/configs/ace2026_selected.json'))
    run.add_argument('--reference-root', type=Path, default=Path('E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas'))
    run.add_argument('--parallel', choices=(1, 2), type=int, default=2)
    run.add_argument('--resume', action='store_true')
    worker = sub.add_parser('worker')
    worker.add_argument('--plan', type=Path, required=True)
    worker.add_argument('--job-id', required=True)
    worker.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'worker':
        plan = ace.read(args.plan)
        verify_plan(plan)
        jobs = [j for j in plan['jobs'] if j['id'] == args.job_id]
        ace.require(len(jobs) == 1, 'Worker absent from fixed plan')
        job = jobs[0]
        if job['component'] == 'coverage':
            coverage.run(job['config'], modes=['all_windows'], resume=args.resume)
        else:
            teacher.run(job['config'], resume=args.resume)
        complete_receipt(job, plan)
    else:
        root = args.output_dir.resolve()
        fresh = make_plan(root, args.ace_config.resolve(), args.reference_root.resolve(), args.parallel)
        if root.exists() and any(root.iterdir()):
            ace.require(args.resume and ace.read(root / 'master_plan.json') == fresh, 'Component plan changed')
        else:
            root.mkdir(parents=True, exist_ok=True)
        for job in fresh['jobs']:
            immutable(Path(job['config_file']), job['config'])
        immutable(root / 'master_plan.json', fresh)
        verify_plan(fresh)
        dispatch(root, fresh, args.resume)


if __name__ == '__main__':
    main()
