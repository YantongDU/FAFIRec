"""Aggregate completed runs and compare matched controls using seeds as units.

The seed bootstrap is conditional on the fixed evaluation users. It does not
claim user-population uncertainty, and repeated users across seeds are never
pooled as independent samples. Only real per-user rank files are consumed.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from fafirec.metrics import summarize_rows


def load_rows(run_dir, split="test"):
    paths = sorted(Path(run_dir).glob('seed_*/*_per_user.csv'))
    if not paths:
        raise ValueError("No seed_*/<control>_per_user.csv files found")
    rows, seen = [], set()
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        if not records:
            raise ValueError(f"Empty per-user file: {path}")
        for row in records:
            if row.get('split') != split:
                continue
            control = row.get('control') or row.get('model')
            if not control or not row.get('user') or row.get('group') not in {'cold', 'middle', 'warm'}:
                raise ValueError(f"Missing control/user or unknown group in {path}")
            if path.name != f'{control}_per_user.csv':
                raise ValueError(f"Control column disagrees with per-user filename in {path}")
            try:
                seed, rank = int(row['seed']), int(row['rank'])
                if str(rank) != str(row['rank']).strip() or rank < 1:
                    raise ValueError()
            except (ValueError, KeyError):
                raise ValueError(f"Invalid seed or one-based integer rank in {path}") from None
            if path.parent.name != f'seed_{seed}':
                raise ValueError(f"Seed column disagrees with directory in {path}")
            key = (control, seed, row['user'], split)
            if key in seen:
                raise ValueError(f"Duplicate user/control/seed/split result: {key}")
            seen.add(key)
            rows.append({**row, 'control': control, 'seed': seed, 'rank': rank})
    if not rows:
        raise ValueError(f"No rows for split {split!r}")
    return rows


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_pipeline_integrity(run_dir, rows, split='test'):
    """Bind a complete result grid to saved inputs, source, and candidate policy.

    A saved source fingerprint records the source used for the experiment; it
    deliberately does not require today's working tree to be unchanged. Legacy
    CSV-only inputs remain descriptive, with provenance explicitly unverified.
    Hash checks detect accidental changes, not malicious rewriting of all logs.
    """
    root = Path(run_dir)
    manifest_path = root/'manifest.json'
    if not manifest_path.exists():
        return {'status': 'unverified_csv_only', 'configured_jobs_verified': False,
                'note': 'No run manifest: input/source/candidate-policy provenance is unverified.'}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    try:
        cfg, identity, source = manifest['config'], manifest['identity'], manifest['source']
        seeds, controls = cfg['seeds'], cfg['controls']
        users = identity['user_order']
        policy, fingerprint = cfg['candidate_policy'], manifest['fingerprint']
        source_hashes = source['file_hashes']
    except (KeyError, TypeError):
        raise ValueError('Run manifest is missing configuration, identity, or source provenance') from None
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds) or not isinstance(controls, list) or not controls
            or any(not isinstance(c, str) or not c or Path(c).name != c or '/' in c or '\\' in c for c in controls)
            or len(set(controls)) != len(controls)):
        raise ValueError('Manifest seeds/controls must be nonempty unique configured jobs')
    if (not isinstance(users, list) or not users or any(not isinstance(u, str) or not u for u in users)
            or len(set(users)) != len(users)):
        raise ValueError('Manifest identity.user_order must contain unique user IDs')
    if policy not in {'full', 'exclude_seen', 'exclude_seen_restore_target'}:
        raise ValueError('Run manifest has an unknown candidate policy')
    if manifest.get('candidate_policy', policy) != policy:
        raise ValueError('Candidate-policy mix between manifest and configuration')
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError('Run manifest lacks source hashes')
    fingerprint_inputs = {'config': cfg, 'identity': identity, 'source_hashes': source_hashes}
    if 'runtime' in manifest:
        fingerprint_inputs['runtime'] = manifest['runtime']
    calculated = _json_digest(fingerprint_inputs)
    if calculated != fingerprint:
        raise ValueError('Run manifest fingerprint does not match its source/configuration/input identity')
    expected = {(c, s) for c in controls for s in seeds}
    seed_dirs = {p.name for p in root.glob('seed_*') if p.is_dir()}
    if seed_dirs != {f'seed_{s}' for s in seeds}:
        raise ValueError('Missing/extra seed directories relative to configured jobs')
    expected_csv = {f'seed_{s}/{c}_per_user.csv' for c, s in expected}
    expected_summary = {f'seed_{s}/{c}_summary.json' for c, s in expected}
    actual_csv = {p.relative_to(root).as_posix() for p in root.glob('seed_*/*_per_user.csv')}
    actual_summary = {p.relative_to(root).as_posix() for p in root.glob('seed_*/*_summary.json')}
    if actual_csv != expected_csv or actual_summary != expected_summary:
        raise ValueError('Incomplete pipeline or extra configured jobs: per-user CSV/summary grid differs from manifest')
    grouped = {}
    for row in rows:
        grouped.setdefault((row['control'], row['seed']), []).append(row)
    if set(grouped) != expected:
        raise ValueError('Evaluation split does not cover every configured seed/control job')
    root_summary_path = root/'summary.json'
    if not root_summary_path.exists():
        raise ValueError('Incomplete pipeline: root summary.json is missing')
    summaries = json.loads(root_summary_path.read_text(encoding='utf-8-sig'))
    if not isinstance(summaries, list):
        raise ValueError('Root summary.json must list completed jobs')
    indexed = {}
    for summary in summaries:
        try:
            key = (summary['control'], summary['seed'])
        except (KeyError, TypeError):
            raise ValueError('Invalid completed job in root summary.json') from None
        if key in indexed:
            raise ValueError('Duplicate completed job in root summary.json')
        indexed[key] = summary
    if set(indexed) != expected:
        raise ValueError('Incomplete pipeline or extra jobs in root summary.json')
    checked_csv_hashes, checked_checkpoint_hashes, checked_diagnostic_hashes = 0, 0, 0
    profile = None
    for control, seed in sorted(expected):
        job_rows = grouped[(control, seed)]
        if [row['user'] for row in job_rows] != users:
            raise ValueError(f'Evaluation users/order differ from manifest identity for {control}/{seed}')
        current_profile = [(r['user'], r['group'], r.get('target'), r.get('training_interactions')) for r in job_rows]
        if profile is None:
            profile = current_profile
        elif current_profile != profile:
            raise ValueError('Targets/groups/training counts differ across configured jobs')
        csv_path = root/f'seed_{seed}/{control}_per_user.csv'
        summary_path = root/f'seed_{seed}/{control}_summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        if summary != indexed[(control, seed)]:
            raise ValueError(f'Per-job and root summaries disagree for {control}/{seed}')
        if summary.get('seed') != seed or summary.get('control') != control or summary.get('fingerprint') != fingerprint:
            raise ValueError(f'Source/configuration fingerprint mix in completed job {control}/{seed}')
        if summary.get('candidate_policy', policy) != policy:
            raise ValueError(f'Candidate-policy mix in completed job {control}/{seed}')
        for row in job_rows:
            if row.get('fingerprint', fingerprint) != fingerprint or row.get('candidate_policy', policy) != policy:
                raise ValueError(f'Source/candidate-policy mix in per-user rows for {control}/{seed}')
        if 'per_user_sha256' in summary:
            if summary['per_user_sha256'] != _file_digest(csv_path):
                raise ValueError(f'Per-user CSV checksum mismatch for {control}/{seed}')
            checked_csv_hashes += 1
        if 'checkpoint_sha256' in summary:
            checkpoint = root/f'seed_{seed}/{control}/best.pt'
            if not checkpoint.exists() or summary['checkpoint_sha256'] != _file_digest(checkpoint):
                raise ValueError(f'Checkpoint checksum mismatch for {control}/{seed}')
            checked_checkpoint_hashes += 1
        if 'diagnostics_sha256' in summary:
            diagnostic = root/f'seed_{seed}/diagnostics.pt'
            if not diagnostic.exists() or summary['diagnostics_sha256'] != _file_digest(diagnostic):
                raise ValueError(f'Diagnostic-cache checksum mismatch for {control}/{seed}')
            checked_diagnostic_hashes += 1
        # The summary must agree with actual rank rows, even for older runs
        # that predate per-user checksums. Use the experiment's own cutoffs.
        configured_ks = tuple(cfg.get('ks', (5, 10, 20)))
        observed_metrics = summarize_rows(job_rows, configured_ks)
        recorded_metrics = summary.get(split)
        if not isinstance(recorded_metrics, dict) or set(recorded_metrics) != set(observed_metrics):
            raise ValueError(f'Summary population/groups disagree with CSV for {control}/{seed}')
        for group, values in observed_metrics.items():
            for metric, value in values.items():
                recorded = recorded_metrics[group].get(metric)
                if (not isinstance(recorded, (float, int)) or not math.isfinite(recorded)
                        or not math.isclose(recorded, value, rel_tol=1e-10, abs_tol=1e-12)):
                    raise ValueError(f'Summary {group}/{metric} disagrees with CSV for {control}/{seed}')
    return {'status': 'manifest_and_job_fingerprints_verified', 'configured_jobs_verified': True,
            'job_count': len(expected), 'manifest_sha256': _file_digest(manifest_path),
            'experiment_fingerprint': fingerprint, 'candidate_policy': policy,
            'per_user_checksums_verified': checked_csv_hashes,
            'checkpoint_checksums_verified': checked_checkpoint_hashes,
            'diagnostic_checksums_verified': checked_diagnostic_hashes,
            'experiment_status': manifest.get('status'),
            'feature_type': manifest.get('data', {}).get('feature_type'),
            'base_commit': source.get('base_commit'),
            'note': 'Complete configured jobs and saved provenance verified; historical working-tree source is not reexecuted.'}


def paired_seed_comparison(differences, bootstrap_samples=10000, seed=2025):
    """Mean difference, seed-resampling CI, and two-sided paired sign flips.

    The sign-flip test assumes the paired seed differences are symmetric under
    the null. At five seeds its exact two-sided resolution is only 2/32.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need at least one finite paired seed difference")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 seed-bootstrap replicates")
    n = len(values)
    result = {'paired_seeds': n, 'mean_difference': float(values.mean()),
              'sample_std_difference': float(values.std(ddof=1)) if n > 1 else None,
              'per_seed_differences': values.tolist(), 'bootstrap_95_ci': None,
              'two_sided_sign_flip_p': None, 'sign_flip_method': None}
    if n < 2:
        result['inference_note'] = 'One training seed: descriptive difference only; no interval or p-value.'
        return result
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 1000):
        amount = min(1000, bootstrap_samples - start)
        sample = rng.integers(0, n, size=(amount, n))
        means[start:start + amount] = values[sample].mean(axis=1)
    result['bootstrap_95_ci'] = np.quantile(means, [.025, .975]).tolist()
    observed = abs(values.mean())
    tolerance = 1e-12 * max(1., observed)
    if n <= 16:
        count = sum(abs(np.dot(signs, values) / n) >= observed - tolerance
                    for signs in itertools.product((-1., 1.), repeat=n))
        result['two_sided_sign_flip_p'] = float(count / (2 ** n))
        result['sign_flip_method'] = f'exact enumeration of {2 ** n} paired seed sign patterns'
    else:
        count = 0
        for start in range(0, bootstrap_samples, 1000):
            amount = min(1000, bootstrap_samples - start)
            signs = rng.choice((-1., 1.), size=(amount, n))
            count += int((np.abs(signs @ values / n) >= observed - tolerance).sum())
        result['two_sided_sign_flip_p'] = float((count + 1) / (bootstrap_samples + 1))
        result['sign_flip_method'] = f'Monte Carlo with {bootstrap_samples} paired seed sign patterns'
    result['inference_note'] = ('Uncertainty over training seeds, conditional on these fixed users. '
                                'Few seeds give coarse p-values and unstable bootstrap intervals.')
    return result


def _holm_adjust(comparisons):
    available = [(i, c['two_sided_sign_flip_p']) for i, c in enumerate(comparisons)
                 if c['two_sided_sign_flip_p'] is not None]
    ordered = sorted(available, key=lambda item: item[1])
    running = 0.
    for rank, (index, p) in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - rank) * p))
        comparisons[index]['holm_adjusted_p'] = running


def analyze_runs(run_dir, comparison='diagnostic', baseline='uniform', ks=(5, 10, 20),
                 bootstrap_samples=10000, seed=2025, split='test'):
    if comparison == baseline:
        raise ValueError("Comparison and baseline must differ")
    ks = tuple(sorted(set(ks)))
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integer cutoffs")
    rows = load_rows(run_dir, split)
    provenance = verify_pipeline_integrity(run_dir, rows, split)
    by_control_seed = {}
    for row in rows:
        by_control_seed.setdefault((row['control'], row['seed']), []).append(row)
    metrics = {key: summarize_rows(group, ks) for key, group in by_control_seed.items()}
    controls = sorted({key[0] for key in metrics})
    aggregates = []
    for control in controls:
        seeds = sorted(s for c, s in metrics if c == control)
        for group in ('all', 'cold', 'middle', 'warm'):
            present = [s for s in seeds if group in metrics[(control, s)]]
            if not present:
                continue
            populations = [{r['user'] for r in by_control_seed[(control, s)]
                            if group == 'all' or r['group'] == group} for s in present]
            if any(population != populations[0] for population in populations[1:]):
                raise ValueError(f"Evaluation users differ between seeds for {control}/{group}")
            for metric in (f'{name}{k}' for k in ks for name in ('hit', 'ndcg', 'mrr')):
                values = np.array([metrics[(control, s)][group][metric] for s in present])
                aggregates.append({'control': control, 'group': group, 'metric': metric,
                                   'mean': float(values.mean()),
                                   'sample_std': float(values.std(ddof=1)) if len(values) > 1 else None,
                                   'seeds': present, 'seed_count': len(present),
                                   'users_per_seed': len(populations[0])})
    comparisons = []
    if comparison in controls and baseline in controls:
        comparison_seeds = {s for c, s in metrics if c == comparison}
        baseline_seeds = {s for c, s in metrics if c == baseline}
        if comparison_seeds != baseline_seeds:
            raise ValueError("Paired controls have different seed sets; complete matched runs first")
        seeds = sorted(comparison_seeds)
        for group in ('all', 'cold', 'middle', 'warm'):
            if not all(group in metrics[(comparison, s)] and group in metrics[(baseline, s)] for s in seeds):
                continue
            for s in seeds:
                populations = []
                for control in (comparison, baseline):
                    populations.append({r['user']: (r['group'], r.get('target'))
                                        for r in by_control_seed[(control, s)]
                                        if group == 'all' or r['group'] == group})
                if populations[0] != populations[1]:
                    raise ValueError(f"Paired users, targets, or groups differ for seed {s}/{group}")
            for metric in (f'{name}{k}' for k in ks for name in ('hit', 'ndcg', 'mrr')):
                deltas = [metrics[(comparison, s)][group][metric] - metrics[(baseline, s)][group][metric]
                          for s in seeds]
                result = paired_seed_comparison(deltas, bootstrap_samples, seed)
                comparisons.append({'comparison': comparison, 'baseline': baseline,
                                    'group': group, 'metric': metric, 'seed_ids': seeds,
                                    'primary_candidate': group == 'all' and metric == 'ndcg10', **result})
        _holm_adjust(comparisons)
    else:
        missing = [c for c in (comparison, baseline) if c not in controls]
        comparison_note = f'No paired comparison generated; missing controls: {missing}'
    return {'split': split, 'cutoffs': list(ks), 'controls': controls, 'provenance': provenance,
            'aggregation_unit': 'training seed; metrics computed over fixed users within each seed',
            'standard_deviation': 'sample standard deviation across seeds (ddof=1); null for one seed',
            'uncertainty_scope': 'training randomness conditional on the fixed evaluation population',
            'multiple_comparisons': 'Holm adjustment across all reported paired group/metric tests',
            'comparison_note': locals().get('comparison_note', 'Matched seed sets and identical evaluation users required.'),
            'aggregates': aggregates, 'paired_comparisons': comparisons}


def write_analysis(report, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    aggregates = report['aggregates']
    with (out/'aggregate_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        for row in aggregates:
            writer.writerow({**row, 'seeds': ','.join(map(str, row['seeds']))})
    paper = {}
    for row in aggregates:
        key = (row['control'], row['group'])
        entry = paper.setdefault(key, {'control': key[0], 'group': key[1],
                                      'seeds': row['seed_count'], 'users_per_seed': row['users_per_seed']})
        sd = 'n/a' if row['sample_std'] is None else f"{row['sample_std']:.6f}"
        entry[row['metric']] = f"{row['mean']:.6f} +/- {sd}"
    columns = ['control', 'group', 'seeds', 'users_per_seed']
    columns += [f'{name}{k}' for k in report['cutoffs'] for name in ('hit', 'ndcg', 'mrr')]
    with (out/'paper_table.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paper.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--comparison', default='diagnostic')
    parser.add_argument('--baseline', default='uniform')
    parser.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    args = parser.parse_args()
    report = analyze_runs(args.run_dir, args.comparison, args.baseline, args.ks,
                          args.bootstrap_samples, args.seed, args.split)
    write_analysis(report, args.output_dir or args.run_dir/'analysis')
    print(json.dumps({'controls': report['controls'], 'paired_tests': len(report['paired_comparisons']),
                      'output_dir': str(args.output_dir or args.run_dir/'analysis')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
