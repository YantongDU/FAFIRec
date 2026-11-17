"""Independently replay every selected ratio-split checkpoint without overwrites."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from fafirec.ratio_runner import evaluate, load_selected, write_rows
from fafirec.runner import file_digest, json_dump, resolve_device, runtime_identity


def verify(run_dir):
    root = Path(run_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'ml1m_80_10_10_multi_positive':
        raise ValueError('Expected the multi-positive ratio-split experiment')
    cfg = manifest['config']
    if runtime_identity(resolve_device(cfg)) != manifest['runtime']:
        raise ValueError('Replay requires the recorded device and software environment')
    summaries = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    expected = {(s, c) for s in cfg['seeds'] for c in cfg['controls']}
    if {(row['seed'], row['control']) for row in summaries} != expected or len(summaries) != len(expected):
        raise ValueError('The configured seed/control grid is incomplete')
    out = root / 'reloaded_all'
    if out.exists():
        raise FileExistsError('Reload output exists; inspect it rather than overwriting')
    out.mkdir()
    jobs = []
    for seed in cfg['seeds']:
        for control in cfg['controls']:
            model, records, diagnostics, loaded_cfg, _ = load_selected(root, seed, control)
            rows, metrics = evaluate(model, records, 'test', diagnostics, loaded_cfg, seed)
            job_out = out / f'seed_{seed}' / control
            job_out.mkdir(parents=True)
            csv_path = job_out / 'per_user.csv'
            write_rows(csv_path, rows)
            original = root / f'seed_{seed}' / f'{control}_per_user.csv'
            saved = json.loads((root / f'seed_{seed}' / f'{control}_summary.json').read_text(encoding='utf-8'))
            result = {'seed': seed, 'control': control, 'original_sha256': file_digest(original),
                      'reloaded_sha256': file_digest(csv_path), 'metrics_identical': metrics == saved['test']}
            result['identical'] = result['original_sha256'] == result['reloaded_sha256'] and result['metrics_identical']
            jobs.append(result)
            json_dump(job_out / 'verification.json', result)
            print(json.dumps({'seed': seed, 'control': control, 'identical': result['identical']}), flush=True)
            del model
    report = {'fingerprint': manifest['fingerprint'], 'device': cfg['device'],
              'all_identical': all(row['identical'] for row in jobs), 'jobs': jobs}
    (root / 'analysis').mkdir(exist_ok=True)
    json_dump(root / 'analysis/reload_verification.json', report)
    if not report['all_identical']:
        raise RuntimeError('One or more checkpoint replays differs from the original output')
    print(f'VERIFIED {len(jobs)} selected checkpoints', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    verify(args.run_dir)


if __name__ == '__main__':
    main()
