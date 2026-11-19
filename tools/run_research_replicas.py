"""Dispatch a predeclared seed grid into isolated, resumable run directories."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fafirec.research_runner import load_config, source_identity, _read, _require
from fafirec.runner import digest, file_digest, json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--parallel', type=int, choices=(1, 2), default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['output_dir'])
    plan = {'config': cfg, 'master_config_sha256': file_digest(args.config),
            'source_hashes': source_identity(), 'parallel_workers': args.parallel,
            'protocol': 'isolated_replicas_identical_settings_except_seed_and_output',
            'seeds': cfg['seeds']}
    plan['fingerprint'] = digest(plan)
    if root.exists() and any(root.iterdir()):
        _require(args.resume and _read(root / 'master_plan.json') == plan,
                 'Existing replica plan differs; choose a new directory or resume identical settings')
    else:
        root.mkdir(parents=True, exist_ok=True)
        json_dump(root / 'master_plan.json', plan)
    pending, active, completed, failed = list(cfg['seeds']), {}, [], []
    try:
        while pending or active:
            while pending and len(active) < args.parallel and not failed:
                seed = pending.pop(0)
                folder = root / f'seed_{seed}'
                folder.mkdir(exist_ok=True)
                local_cfg = {**cfg, 'seeds': [seed], 'output_dir': str(folder / 'run')}
                path = folder / 'config.json'
                if path.exists():
                    _require(_read(path) == local_cfg, 'Replica configuration differs')
                else:
                    json_dump(path, local_cfg)
                log = (folder / ('training_resume.log' if args.resume else 'training.log')).open('ab')
                command = [sys.executable, '-u', str(ROOT / 'research_ml1m.py'), '--config', str(path)]
                if args.resume:
                    command.append('--resume')
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=os.environ.copy(), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                active[seed] = (child, log)
                print(f'Started seed {seed}, process {child.pid}', flush=True)
            for seed, (child, log) in list(active.items()):
                status = child.poll()
                if status is not None:
                    log.close()
                    del active[seed]
                    (completed if status == 0 else failed).append(seed)
                    print(f'Finished seed {seed}, exit {status}', flush=True)
            json_dump(root / 'dispatch_status.json', {'plan_fingerprint': plan['fingerprint'],
                'active': {str(seed): child.pid for seed, (child, _) in active.items()},
                'completed': completed, 'failed': failed, 'pending': pending})
            if failed and not active:
                raise RuntimeError('A replica failed; inspect its log and resume after resolving the cause')
            if active:
                time.sleep(2)
    finally:
        # Interrupting this dispatcher stops only the exact children it owns;
        # each worker retains atomic epoch checkpoints for an identical resume.
        for child, log in active.values():
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            log.close()
    _require(set(completed) == set(cfg['seeds']), 'Complete configured grid required')


if __name__ == '__main__':
    main()
