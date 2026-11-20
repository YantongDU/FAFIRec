"""Run one validation-selected ACE recipe in five isolated seed directories."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ace2026_experiment as ace
from fafirec.runner import digest, file_digest, json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--parallel', type=int, choices=(1, 2), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = ace.load_config(args.config)
    ace.require(len(cfg['seeds']) == 5 and len(cfg['variants']) == 1, 'Five seeds and one common selected recipe required')
    root = Path(cfg['output_dir'])
    plan = {'config': cfg, 'config_sha256': file_digest(args.config), 'source_hashes': ace.source_identity(),
            'driver_sha256': file_digest(__file__), 'parallel': args.parallel,
            'svd_cache_sha256': file_digest(cfg['svd_cache']) if cfg['svd_cache'] else None}
    plan['fingerprint'] = digest(plan)
    if root.exists() and any(root.iterdir()):
        ace.require(args.resume and ace.read(root / 'master_plan.json') == plan, 'Replica plan changed')
    else:
        root.mkdir(parents=True, exist_ok=True)
        json_dump(root / 'master_plan.json', plan)
    paths = [str(root / f'seed_{seed}/run') for seed in cfg['seeds']]
    if (root / 'replica_runs.json').exists():
        ace.require(ace.read(root / 'replica_runs.json') == paths, 'Replica scope changed')
    else:
        json_dump(root / 'replica_runs.json', paths)
    pending, active, completed, failed = list(cfg['seeds']), {}, [], []
    try:
        while pending or active:
            while pending and len(active) < args.parallel and not failed:
                seed = pending.pop(0)
                folder = root / f'seed_{seed}'
                folder.mkdir(exist_ok=True)
                local = {**cfg, 'seeds': [seed], 'output_dir': str(folder / 'run')}
                config_path = folder / 'config.json'
                if config_path.exists():
                    ace.require(ace.read(config_path) == local, 'Replica config changed')
                else:
                    json_dump(config_path, local)
                log = (folder / ('training_resume.log' if args.resume else 'training.log')).open('ab')
                command = [sys.executable, '-u', str(ROOT / 'tools/ace2026_experiment.py'), 'train', '--config', str(config_path)]
                if args.resume: command.append('--resume')
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
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
                'active': {str(s): p.pid for s, (p, _) in active.items()},
                'completed': completed, 'failed': failed, 'pending': pending})
            if failed and not active: raise RuntimeError('Inspect failed replica logs before resuming')
            if active: time.sleep(2)
    finally:
        for child, log in active.values():
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            log.close()
    ace.require(set(completed) == set(cfg['seeds']), 'Incomplete five-seed grid')


if __name__ == '__main__':
    main()
