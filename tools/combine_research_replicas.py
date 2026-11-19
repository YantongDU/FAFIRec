"""Combine independently audited seed replicas without running any model.

Validation writes one immutable five-seed selection lock. Test mode requires
that explicit lock and retains its common variant/stage choice for every seed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_results import _holm_adjust, paired_seed_comparison
from analyze_transfer import _same
from fafirec import research_runner as research
from fafirec.runner import digest, file_digest

SCHEMA = 'research_v3_five_seed_replica_selection_v1'
RULE = 'Choose one common variant and stage by five-seed mean validation all NDCG@50; exact ties use variant then stage name. Never choose a stage separately for each seed.'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_config(cfg):
    return {k: v for k, v in cfg.items() if k not in {'seeds', 'output_dir'}}


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def lock_fingerprint(lock):
    return digest({k: v for k, v in lock.items() if k != 'lock_fingerprint'})


def _aggregate_key(row):
    return (row['variant'], row['stage'], row['group'], row['metric'])


def _comparison_key(row):
    return (row['comparison'], tuple(row['candidate']), tuple(row['baseline']), row['group'], row['metric'])


def _capture(hashes, filename, checksum=None):
    filename = str(Path(filename).resolve())
    actual = file_digest(filename)
    require(checksum is None or actual == checksum, 'Artifact hash changed: ' + filename)
    require(filename not in hashes or hashes[filename] == actual, 'Conflicting shared artifact hash: ' + filename)
    hashes[filename] = actual


def load_replica(root, seed, master, split, hashes):
    folder, run = root / f'seed_{seed}', root / f'seed_{seed}/run'
    manifest_path, config_path = run / 'manifest.json', folder / 'config.json'
    manifest = read(manifest_path)
    require(manifest.get('protocol') == research.PROTOCOL
            and manifest.get('fingerprint') == research._fingerprint(manifest), 'Replica manifest fingerprint differs')
    cfg = research.normalize_config(manifest['config'])
    require(cfg['seeds'] == [seed] and Path(cfg['output_dir']).resolve() == run.resolve(), 'Replica seed/output identity differs')
    require(common_config(cfg) == common_config(master), 'Replica common configuration differs from master')
    require(research.normalize_config(read(config_path)) == cfg, 'Replica config file differs from manifest')
    require(manifest['source_hashes'] == research.source_identity(), 'Frozen training source changed')
    report_path = folder / f'compare_{split}.json'
    report = read(report_path)
    require(report.get('schema') == 'research_v3_controlled_comparison_v1'
            and report.get('split') == split and report.get('seed_ids') == [seed]
            and report.get('fingerprint') == manifest['fingerprint'], 'Replica audit report identity differs')
    require(report.get('cpu_checkpoint_identity_verified') is True
            and report.get('model_evaluation_called') is False
            and report.get('complete_configured_grid_verified') is True
            and report.get('all_ranking_metrics_independently_recomputed') is True, 'Replica lacks a complete independent audit')
    helper = Path(report['helper_file']).resolve()
    expected_helper = (ROOT / 'tools/compare_research.py').resolve()
    require(helper == expected_helper and report['helper_sha256'] == file_digest(helper), 'Replica comparison helper source changed')
    require(report['reference'] == manifest['reference'] and report['variants'] == master['variants'], 'Replica report reference or variants differ')
    require(report['selection_budget'] == {k: master[k] for k in report['selection_budget']}, 'Replica reported selection budget differs')
    stages = ['expert', 'router'] if master['train_router'] else ['expert']
    jobs = {(v['name'], stage) for v in master['variants'] for stage in stages}
    selections = {(r['variant'], r['stage']): r for r in report['selections']}
    require(report['job_count'] == len(jobs) and len(selections) == len(report['selections'])
            and set(selections) == jobs and all(r['seed'] == seed for r in selections.values()), 'Replica audit job grid differs')
    for variant, stage in jobs:
        path = run / f'seed_{seed}/{variant}/{stage}/summary.json'
        saved = read(path)
        require(saved['seed'] == seed and saved['variant'] == variant and saved['stage'] == stage
                and saved['fingerprint'] == manifest['fingerprint'], 'Replica stage summary identity differs')
        selection = selections[variant, stage]
        for key in ('selected_epoch', 'epochs_completed', 'optimizer_steps', 'calibration', 'checkpoint_sha256'):
            _same(selection[key], saved[key], 'Audit selection metadata differs')
        checkpoint_path = path.parent / 'best.pt'
        _capture(hashes, checkpoint_path, saved['checkpoint_sha256'])
        _capture(hashes, path.parent / 'validation.csv', saved['per_user_sha256'])
        _capture(hashes, path)
    aggregate = {_aggregate_key(r): r for r in report['aggregates']}
    require(len(aggregate) == len(report['aggregates']) and aggregate, 'Replica aggregate grid is empty or duplicated')
    expected_models = jobs | {('strong_cf_anchor', 'frozen')}
    groups = {'all', 'cold', 'middle', 'warm'}
    metric_names = {f'{m}{k}' for k in (10, 50) for m in ('recall', 'ndcg', 'hit')}
    require(set(aggregate) == {(v, s, g, m) for v, s in expected_models for g in groups for m in metric_names},
            'Replica aggregate metric/population grid differs')
    for key, row in aggregate.items():
        require(row['seed_ids'] == [seed] and len(row['per_seed_scores']) == 1
                and row['sample_std'] is None, 'Replica must have exactly one score per aggregate')
        value = row['per_seed_scores'][0]
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                and 0 <= value <= 1 and row['mean'] == value, 'Invalid replica metric score')
    comparisons = {_comparison_key(r): r for r in report['paired_comparisons']}
    require(len(comparisons) == len(report['paired_comparisons']), 'Duplicated replica comparison specifications')
    for key, row in comparisons.items():
        require(row['seed_ids'] == [seed] and row['paired_seeds'] == 1
                and row['bootstrap_95_ci'] is None and row['two_sided_sign_flip_p'] is None,
                'Single-seed replica must not report seed inference')
        left = aggregate[*key[1], key[3], key[4]]['per_seed_scores'][0]
        right = aggregate[*key[2], key[3], key[4]]['per_seed_scores'][0]
        _same(row['candidate_scores'], [left], 'Replica comparison candidate score')
        _same(row['baseline_scores'], [right], 'Replica comparison baseline score')
        _same(row['per_seed_differences'], [left - right], 'Replica comparison difference')
        _same(row['mean_difference'], left - right, 'Replica comparison mean difference')
    _capture(hashes, manifest_path)
    _capture(hashes, config_path)
    _capture(hashes, report_path)
    _capture(hashes, helper, report['helper_sha256'])
    for filename, checksum in report['artifact_hashes'].items():
        _capture(hashes, filename, checksum)
    for key, expected in [('source_hashes', manifest['source_hashes'])]:
        for filename, checksum in expected.items():
            _capture(hashes, ROOT / filename, checksum)
    return {'manifest': manifest, 'report': report, 'aggregates': aggregate, 'comparisons': comparisons,
            'selections': list(selections.values()), 'report_file': str(report_path.resolve()),
            'report_sha256': file_digest(report_path)}


def validate_common(replicas, seeds):
    first = replicas[seeds[0]]
    common_reference = {k: v for k, v in first['manifest']['reference'].items() if k != 'anchors'}
    anchors = {}
    for seed in seeds:
        value = replicas[seed]
        for field in ('source_hashes', 'runtime', 'identity', 'evaluation_config', 'selection'):
            require(value['manifest'][field] == first['manifest'][field], 'Replicas differ in ' + field)
        reference = value['manifest']['reference']
        require({k: v for k, v in reference.items() if k != 'anchors'} == common_reference,
                'Replica common reference provenance differs')
        require(set(reference['anchors']) == {str(seed)}, 'Replica anchor seed scope differs')
        anchors.update(reference['anchors'])
        require(set(value['aggregates']) == set(first['aggregates']), 'Replica aggregate specifications differ')
        require(set(value['comparisons']) == set(first['comparisons']), 'Replica paired-comparison specifications differ')
        for key in value['comparisons']:
            require(value['comparisons'][key]['attribution'] == first['comparisons'][key]['attribution'],
                    'Replica comparison attribution differs')
        require(value['report']['missing_matched_controls'] == first['report']['missing_matched_controls'],
                'Replica matched-control availability differs')
        require(value['report']['helper_sha256'] == first['report']['helper_sha256'], 'Replica helper versions differ')
    return {'source_hashes': first['manifest']['source_hashes'], 'runtime': first['manifest']['runtime'],
            'identity': first['manifest']['identity'], 'evaluation_config': first['manifest']['evaluation_config'],
            'reference': {**common_reference, 'anchors': anchors}, 'comparison_helper_sha256': first['report']['helper_sha256']}


def aggregate(replicas, seeds, bootstrap_samples, bootstrap_seed):
    first = replicas[seeds[0]]
    aggregates, comparisons = [], []
    for key in sorted(first['aggregates']):
        row = first['aggregates'][key]
        scores = [replicas[s]['aggregates'][key]['per_seed_scores'][0] for s in seeds]
        aggregates.append({k: row[k] for k in ('variant', 'stage', 'group', 'metric')} | {
            'seed_ids': seeds, 'seed_count': len(seeds), 'per_seed_scores': scores,
            'mean': statistics.mean(scores), 'sample_std': statistics.stdev(scores)})
    for key in sorted(first['comparisons']):
        row = first['comparisons'][key]
        left = [replicas[s]['comparisons'][key]['candidate_scores'][0] for s in seeds]
        right = [replicas[s]['comparisons'][key]['baseline_scores'][0] for s in seeds]
        differences = [a - b for a, b in zip(left, right)]
        statistics_value = paired_seed_comparison(differences, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed)
        baseline_mean = statistics.mean(right)
        comparisons.append({k: row[k] for k in ('comparison', 'candidate', 'baseline', 'group', 'metric', 'attribution')} | {
            'seed_ids': seeds, 'candidate_scores': left, 'baseline_scores': right,
            'relative_mean_gain_percent': 100. * statistics_value['mean_difference'] / baseline_mean if baseline_mean else None,
            **statistics_value})
    _holm_adjust(comparisons)
    return aggregates, comparisons


def validation_choice(aggregates):
    candidates = [r for r in aggregates if r['group'] == 'all' and r['metric'] == 'ndcg50'
                  and r['variant'] != 'strong_cf_anchor']
    require(bool(candidates), 'No validation selection candidates')
    ranked = sorted(candidates, key=lambda r: (-r['mean'], r['variant'], r['stage']))
    return {'selected_variant': ranked[0]['variant'], 'selected_stage': ranked[0]['stage'],
            'rule': RULE, 'candidates': ranked}


def execute(args):
    root, master_path = args.replica_root.resolve(), args.master_config.resolve()
    master = research.load_config(master_path)
    seeds = list(master['seeds'])
    require(len(seeds) == 5 and len(set(seeds)) == 5, 'Exactly five declared seeds are required')
    require(Path(master['output_dir']).resolve() == root, 'Master output_dir must equal replica root')
    require({p.name for p in root.glob('seed_*') if p.is_dir()} == {f'seed_{s}' for s in seeds},
            'Missing or extra replica seed directories')
    require(args.bootstrap_samples >= 100, 'Use at least 100 bootstrap replicates')
    output = args.output.resolve() if args.output is not None else root / f'combined_{args.split}.json'
    if output.exists():
        raise FileExistsError('Choose a new combined output file')
    if args.split == 'test':
        require(args.selection_lock is not None, 'Test aggregation requires an explicit five-seed --selection-lock')
    lock_path = args.selection_lock.resolve() if args.selection_lock else root / 'five_seed_selection_lock.json'
    if args.split == 'validation' and lock_path.exists():
        raise FileExistsError('Choose a new five-seed selection lock path')
    require(output != lock_path, 'Output and selection lock paths must differ')
    validation_hashes = {}
    _capture(validation_hashes, master_path)
    _capture(validation_hashes, Path(__file__))
    plan_path = root / 'master_plan.json'
    plan_verified = False
    if plan_path.is_file():
        plan = read(plan_path)
        require(plan['fingerprint'] == digest({k: v for k, v in plan.items() if k != 'fingerprint'}), 'Master dispatch plan fingerprint differs')
        require(plan['config'] == master and plan['seeds'] == seeds
                and plan['master_config_sha256'] == file_digest(master_path)
                and plan['source_hashes'] == research.source_identity()
                and plan['parallel_workers'] in {1, 2}, 'Master dispatch plan differs')
        _capture(validation_hashes, plan_path)
        plan_verified = True
    validation_replicas = {s: load_replica(root, s, master, 'validation', validation_hashes) for s in seeds}
    common = validate_common(validation_replicas, seeds)
    validation_aggregates, _ = aggregate(validation_replicas, seeds, args.bootstrap_samples, args.bootstrap_seed)
    decision = validation_choice(validation_aggregates)
    validation_reports = {str(s): {'file': validation_replicas[s]['report_file'],
        'sha256': validation_replicas[s]['report_sha256'],
        'run_fingerprint': validation_replicas[s]['manifest']['fingerprint']} for s in seeds}
    if args.split == 'validation':
        lock = {'schema': SCHEMA, 'selection_split': 'validation', 'seed_ids': seeds,
            'master_config': master, 'master_config_file': str(master_path),
            'master_config_sha256': file_digest(master_path), 'replica_root': str(root),
            'common_provenance': common, 'validation_reports': validation_reports,
            'validation_artifact_sha256': validation_hashes, 'decision': decision,
            'historical_test_known': True, 'new_test_used_for_selection': False,
            'master_plan_verified': plan_verified, 'combiner_sha256': file_digest(Path(__file__)),
            'created_utc': datetime.now(timezone.utc).isoformat()}
        lock['lock_fingerprint'] = lock_fingerprint(lock)
        reports = validation_replicas
        aggregates, comparisons = aggregate(reports, seeds, args.bootstrap_samples, args.bootstrap_seed)
        hashes = validation_hashes
    else:
        lock = read(lock_path)
        require(lock.get('schema') == SCHEMA and lock.get('lock_fingerprint') == lock_fingerprint(lock)
                and lock.get('selection_split') == 'validation' and lock.get('seed_ids') == seeds
                and lock.get('new_test_used_for_selection') is False and lock.get('historical_test_known') is True,
                'Five-seed validation lock identity differs')
        require(lock['master_config'] == master and lock['master_config_file'] == str(master_path)
                and lock['master_config_sha256'] == file_digest(master_path)
                and lock['replica_root'] == str(root) and lock['common_provenance'] == common
                and lock['validation_reports'] == validation_reports
                and lock['validation_artifact_sha256'] == validation_hashes
                and lock['combiner_sha256'] == file_digest(Path(__file__)), 'Locked five-seed validation assets changed')
        _same(lock['decision'], decision, 'Locked common validation decision')
        hashes = copy.deepcopy(validation_hashes)
        reports = {s: load_replica(root, s, master, 'test', hashes) for s in seeds}
        require(validate_common(reports, seeds) == common, 'Test replicas differ from locked validation provenance')
        for seed in seeds:
            require(reports[seed]['manifest'] == validation_replicas[seed]['manifest'], 'Test audit uses another trained replica')
            require(reports[seed]['selections'] == validation_replicas[seed]['selections'], 'Test audit changed selected snapshots or calibration')
            require(set(reports[seed]['comparisons']) == set(validation_replicas[seed]['comparisons'])
                    and reports[seed]['report']['missing_matched_controls']
                    == validation_replicas[seed]['report']['missing_matched_controls'],
                    'Test report changed the declared comparison/control family')
            for key in reports[seed]['comparisons']:
                require(reports[seed]['comparisons'][key]['attribution']
                        == validation_replicas[seed]['comparisons'][key]['attribution'],
                        'Test comparison attribution differs from validation')
        aggregates, comparisons = aggregate(reports, seeds, args.bootstrap_samples, args.bootstrap_seed)
    for filename, checksum in hashes.items():
        require(file_digest(filename) == checksum, 'Asset changed during combination: ' + filename)
    selected_rows = [r for r in aggregates if r['variant'] == decision['selected_variant'] and r['stage'] == decision['selected_stage']]
    result = {'schema': 'research_v3_combined_five_seed_report_v1', 'split': args.split,
        'seed_ids': seeds, 'replica_count': len(seeds), 'jobs_per_replica': reports[seeds[0]]['report']['job_count'],
        'complete_configured_grid_verified': True, 'master_plan_verified': plan_verified,
        'model_evaluation_called': False, 'combiner_sha256': file_digest(Path(__file__)),
        'common_provenance': common, 'master_config': master,
        'decision': decision, 'locked_selected_model_scores': selected_rows,
        'aggregates': aggregates, 'paired_comparisons': comparisons,
        'selections': [r for seed in seeds for r in reports[seed]['selections']],
        'replica_report_files': {str(s): {'file': reports[s]['report_file'], 'sha256': reports[s]['report_sha256']} for s in seeds},
        'artifact_sha256': hashes, 'missing_matched_controls': reports[seeds[0]]['report']['missing_matched_controls'],
        'selection_lock_file': str(lock_path), 'selection_lock_fingerprint': lock['lock_fingerprint'],
        'bootstrap_samples': args.bootstrap_samples, 'bootstrap_seed': args.bootstrap_seed,
        'signflip_minimum_two_sided_p': .0625,
        'multiple_comparisons': 'Holm adjustment across every reported paired comparison. Exact two-sided sign flips use five independent training seeds.',
        'iteration_boundary': 'Historical test scores were exposed. Common configuration, snapshot calibration and the shared variant/stage decision use validation only. This is an exploratory follow-up benchmark, not an untouched confirmatory holdout.'}
    if args.split == 'validation':
        write_once(lock_path, lock)
    result['selection_lock_sha256'] = file_digest(lock_path)
    write_once(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-config', type=Path, required=True)
    parser.add_argument('--replica-root', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2025)
    result = execute(parser.parse_args(argv))
    print(json.dumps({'split': result['split'], 'seeds': result['replica_count'],
        'selected_variant': result['decision']['selected_variant'], 'selected_stage': result['decision']['selected_stage'],
        'selection_lock': result['selection_lock_file']}))


if __name__ == '__main__':
    main()
