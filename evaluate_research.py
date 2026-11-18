"""Explicit independent replay of a frozen research checkpoint."""
import argparse
from pathlib import Path

from fafirec.ratio_runner import evaluate, write_rows
from fafirec.research_runner import load_selected
from fafirec.runner import file_digest, json_dump


def replay(run_dir, seed, variant, stage, split, output_dir, device=None):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new replay output directory')
    model, records, diagnostics, cfg, manifest = load_selected(run_dir, seed, variant, stage, device)
    if split == 'validation':
        records = [{k: v for k, v in r.items() if k != 'test'} for r in records]
    rows, metrics = evaluate(model, records, split, diagnostics, cfg, seed)
    for row in rows:
        row.update(variant=variant, stage=stage)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / 'per_user.csv', rows)
    result = {'fingerprint': manifest['fingerprint'], 'seed': seed, 'variant': variant, 'stage': stage,
        'split': split, 'metrics': metrics, 'candidate_policy': cfg['candidate_policy'],
        'target_mode': 'multi_positive_block', 'per_user_sha256': file_digest(out / 'per_user.csv')}
    if split == 'validation':
        original = Path(run_dir) / f'seed_{seed}/{variant}/{stage}/validation.csv'
        result['identical_to_original'] = file_digest(original) == result['per_user_sha256']
        if not result['identical_to_original']:
            raise ValueError('Independent validation replay ranking differs')
    json_dump(out / 'metrics.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--stage', choices=('expert', 'router'), default='expert')
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print(replay(args.run_dir, args.seed, args.variant, args.stage, args.split, args.output_dir, args.device))


if __name__ == '__main__':
    main()
