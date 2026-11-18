"""CPU-only audit of a complete frozen Research v3 grid and its CF anchor.

No model is evaluated. Checkpoints are independently loaded on CPU; all scores
are recomputed from saved user rankings. Test files are accessed only when the
caller explicitly chooses --split test and supplies --test-dir.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--test-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2025)
    return parser.parse_args(argv)


def execute(args):
    project, root = args.project_dir.resolve(), args.run_dir.resolve()
    if args.output.exists():
        raise FileExistsError('Choose a new comparison output file')
    if args.split == 'test' and args.test_dir is None:
        raise ValueError('Test comparison requires an explicit --test-dir')
    if args.bootstrap_samples < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    sys.path.insert(0, str(project))
    import torch
    from analyze_results import _holm_adjust, paired_seed_comparison
    from analyze_transfer import _ids, _rows, _same
    from analyze_cf_baseline import test_metrics
    from analyze_research import audit_rows
    from fafirec import research_runner as research
    from fafirec.multitarget_metrics import summarize_multitarget_rows
    from fafirec.runner import file_digest

    # This helper never resolves the training config's CUDA device. Explicit CPU
    # replay construction is used solely for source/artifact identity checks.
    torch.set_num_threads(4)
    read, require = research._read, research._require
    manifest = read(root / 'manifest.json')
    require(manifest.get('protocol') == research.PROTOCOL
            and manifest.get('fingerprint') == research._fingerprint(manifest), 'Research manifest differs')
    require(research.source_identity() == manifest['source_hashes'], 'Frozen research source changed')
    cfg = research.normalize_config(manifest['config'])
    records, _, identity, _, _, provenance = research.load_anchors(cfg)
    require(identity == manifest['identity'] and provenance == manifest['reference'], 'Anchor inputs changed')
    eval_cfg = manifest['evaluation_config']
    stages = ['expert', 'router'] if cfg['train_router'] else ['expert']
    expected = {(seed, v['name'], stage) for seed in cfg['seeds'] for v in cfg['variants'] for stage in stages}
    summaries = read(root / 'summary.json')
    indexed = {(r['seed'], r['variant'], r['stage']): r for r in summaries}
    require(len(indexed) == len(summaries) and set(indexed) == expected, 'Complete configured grid required')
    for filename in ('summary.json', 'best.pt', 'validation.csv'):
        require({p.relative_to(root).as_posix() for p in root.glob('seed_*/*/*/' + filename)}
                == {f'seed_{seed}/{variant}/{stage}/{filename}' for seed, variant, stage in expected},
                'Missing or extra research artifacts: ' + filename)
    values, selections, hashes = {}, [], {}
    def check_grid(event, stage):
        routes = ['relative'] if stage == 'router' else cfg['routes']
        declared = [(route, alpha) for route in routes for alpha in cfg['alpha_grid']]
        require([(r['route'], r['alpha']) for r in event['grid']] == declared,
                'Validation alpha/route selection budget changed')
        chosen = event.get('selected', event.get('calibration'))
        best = max(event['grid'], key=lambda r: r['validation']['all']['ndcg50'])
        _same(chosen, best, 'Validation alpha tie-breaking or selection changed')

    for seed, variant, stage in sorted(expected):
        model, replay_records, _, replay_cfg, replay_manifest = research.load_selected(
            root, seed, variant, stage=stage, device='cpu')
        require(replay_manifest['fingerprint'] == manifest['fingerprint']
                and replay_records == records, 'Independent checkpoint replay identity differs')
        require({k: v for k, v in replay_cfg.items() if k != 'device'}
                == {k: v for k, v in eval_cfg.items() if k != 'device'}, 'Replay evaluation convention changed')
        summary = indexed[seed, variant, stage]
        folder = root / f'seed_{seed}/{variant}/{stage}'
        _same(summary, read(folder / 'summary.json'), 'Summary grid identity')
        # Audit every scheduled validation decision, not only the chosen one.
        # last.pt is authoritative if interruption happened after its atomic
        # write and before the human-readable training.json write.
        initial_path, last_path = folder / 'initial_validation.json', folder / 'last.pt'
        initial, last = read(initial_path), torch.load(last_path, weights_only=True, map_location='cpu')
        research._check_checkpoint(last, manifest, research._artifact(manifest, seed, specs_variant := next(
            v for v in cfg['variants'] if v['name'] == variant), stage), model)
        check_grid(initial, stage)
        logs = last['logs']
        require([r['epoch'] for r in logs] == list(range(1, len(logs) + 1))
                and len(logs) == last['epoch'] == summary['epochs_completed'], 'Training epoch accounting differs')
        budget = cfg['router_epochs'] if stage == 'router' else cfg['epochs']
        minimum = cfg['router_min_epochs'] if stage == 'router' else cfg['min_epochs']
        steps = (len(records) + cfg['batch_size'] - 1) // cfg['batch_size']
        require(len(logs) <= budget and all(r['optimizer_steps'] == steps for r in logs)
                and sum(r['optimizer_steps'] for r in logs) == summary['optimizer_steps'], 'Optimizer step budget differs')
        checks = []
        best_metric, best_epoch, best_calibration, stale = initial['selected']['validation']['all']['ndcg50'], 0, initial['selected'], 0
        for log in logs:
            scheduled = log['epoch'] % cfg['eval_interval'] == 0 or log['epoch'] == budget
            require(('calibration' in log) == scheduled, 'Validation epoch schedule changed')
            if not scheduled:
                continue
            check_grid(log, stage)
            checks.append(log['epoch'])
            score = log['calibration']['validation']['all']['ndcg50']
            if score > best_metric:
                best_metric, best_epoch, best_calibration, stale = score, log['epoch'], log['calibration'], 0
            else:
                stale += 1
            should_stop = log['epoch'] >= minimum and stale >= cfg['patience']
            require(not should_stop or log['epoch'] == len(logs), 'Training continued beyond declared early stopping')
        require(last['selected_epoch'] == summary['selected_epoch'] == best_epoch
                and last['best_metric'] == summary['selected_validation_metric'] == best_metric
                and last['stale'] == stale, 'Best-epoch or early-stopping selection changed')
        _same(summary['calibration'], best_calibration, 'Whole-history calibration selection')
        require(last['stopped'] is (len(logs) >= minimum and stale >= cfg['patience']), 'Recorded stop flag differs')
        require(len(logs) == budget or last['stopped'], 'Incomplete training budget without early stopping')
        training_path = folder / 'training.json'
        readable = read(training_path) if training_path.is_file() else []
        _same(readable, logs[:len(readable)], 'Human-readable training log differs from epoch checkpoints')
        hashes.update({str(p): file_digest(p) for p in (initial_path, last_path)})
        if training_path.is_file():
            hashes[str(training_path)] = file_digest(training_path)
        validation = audit_rows(folder / 'validation.csv', records, eval_cfg, seed, variant, stage,
                                model.control, 'validation', identity['items'])
        _same(validation, summary['validation'], 'Recomputed validation scores')
        hashes[str(folder / 'best.pt')] = file_digest(folder / 'best.pt')
        hashes[str(folder / 'validation.csv')] = file_digest(folder / 'validation.csv')
        if args.split == 'validation':
            metrics = validation
        else:
            test_folder = args.test_dir.resolve() / f'seed_{seed}/{variant}/{stage}'
            test_path = test_folder / 'per_user.csv'
            saved = read(test_folder / 'metrics.json')
            require(saved.get('fingerprint') == manifest['fingerprint'] and saved.get('seed') == seed
                    and saved.get('variant') == variant and saved.get('stage') == stage
                    and saved.get('split') == 'test' and saved.get('candidate_policy') == eval_cfg['candidate_policy']
                    and saved.get('target_mode') == 'multi_positive_block'
                    and saved.get('per_user_sha256') == file_digest(test_path), 'Research test replay identity changed')
            metrics = audit_rows(test_path, records, eval_cfg, seed, variant, stage,
                                 model.control, 'test', identity['items'])
            _same(metrics, saved['metrics'], 'Recomputed test scores')
            hashes[str(test_path)] = file_digest(test_path)
            hashes[str(test_folder / 'metrics.json')] = file_digest(test_folder / 'metrics.json')
        values[seed, variant, stage] = metrics
        adapter = model.id_table if model.kind == 'id' else model.projection
        adapter_parameters = sum(p.numel() for p in adapter.parameters())
        encoder_parameters = sum(p.numel() for p in model.expert_encoder.parameters())
        gain_raw = float(model.gain_raw.detach().cpu())
        tanh_gain = float(torch.tanh(model.gain_raw.detach()).cpu())
        calibrated_gain = summary['calibration']['alpha'] * tanh_gain
        selections.append({'seed': seed, 'variant': variant, 'stage': stage,
            'selected_epoch': summary['selected_epoch'], 'epochs_completed': summary['epochs_completed'],
            'optimizer_steps': summary['optimizer_steps'], 'calibration': summary['calibration'],
            'validation_checks_including_epoch0': 1 + len(checks),
            'validation_epochs': [0] + checks, 'candidates_per_validation': len(initial['grid']),
            'training_json_lags_atomic_checkpoint': len(readable) < len(logs),
            'gain_raw': gain_raw, 'tanh_gain': tanh_gain,
            'effective_alpha_times_tanh_gain': calibrated_gain,
            'expert_score_scale': model.expert_score_scale, 'anchor_score_scale': model.score_scale,
            'residual_query_multiplier_before_user_gate': calibrated_gain * model.expert_score_scale / model.score_scale,
            'active_trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'expert_parameter_count': adapter_parameters + encoder_parameters,
            'item_adapter_parameter_count': adapter_parameters,
            'expert_encoder_parameter_count': encoder_parameters,
            'router_weight': model.router.weight.detach().cpu().tolist(),
            'router_bias': model.router.bias.detach().cpu().tolist(),
            'router_feature_mean': model.router_mean.detach().cpu().tolist(),
            'router_feature_std': model.router_std.detach().cpu().tolist(),
            'initialization': summary['initialization'],
            'checkpoint_sha256': summary['checkpoint_sha256']})
        del model

    # Audit the already-frozen anchor from its original CSVs. Validation mode
    # does not inspect any stored historical test score or test ranking.
    anchor = Path(cfg['anchor_run'])
    report_path = Path(cfg['reference_main_run']) / 'analysis/strong_cf_comparison.json'
    report = read(report_path)
    old_jobs = {row['seed']: row for row in report['per_seed']}
    anchor_values = {}
    for seed in cfg['seeds']:
        original_summary_path = anchor / f'seed_{seed}/cf_lr1e3_summary.json'
        original_summary = read(original_summary_path)
        require(original_summary['seed'] == seed and original_summary['variant'] == 'cf_lr1e3'
                and original_summary['control'] == 'cf' and original_summary['split'] == 'validation'
                and original_summary['fingerprint'] == provenance['anchor_fingerprint'], 'Old anchor summary identity differs')
        require(original_summary['checkpoint_sha256'] == provenance['anchors'][str(seed)]['checkpoint_sha256'],
                'Old anchor checkpoint hash differs')
        hashes[str(original_summary_path)] = file_digest(original_summary_path)
        if args.split == 'validation':
            path = anchor / f'seed_{seed}/cf_lr1e3_validation.csv'
            require(original_summary['per_user_sha256'] == file_digest(path), 'Old anchor validation ranking changed')
            rows = _rows(path, eval_cfg, records, seed, {'name': 'cf_lr1e3', 'control': 'cf'}, identity['items'])
            metrics = summarize_multitarget_rows(rows, eval_cfg['ks'])
            _same(metrics, original_summary['validation'], 'Old anchor validation metrics')
        else:
            job = old_jobs[seed]
            path = Path(job['per_user_file'])
            reloaded = Path(job['reloaded_file'])
            require(job['source_fingerprint'] == provenance['anchor_fingerprint']
                    and job['source_summary_sha256'] == file_digest(original_summary_path)
                    and job['checkpoint_sha256'] == provenance['anchors'][str(seed)]['checkpoint_sha256']
                    and job['per_user_sha256'] == file_digest(path) == job['reloaded_sha256'] == file_digest(reloaded)
                    and job['reloaded_identical'] is True, 'Old strong CF test/reload hashes changed')
            saved_path, verification_path = path.parent / 'metrics.json', path.parent / 'verification.json'
            saved, verification = read(saved_path), read(verification_path)
            require(saved['fingerprint'] == provenance['anchor_fingerprint'] and saved['seed'] == seed
                    and saved['control'] == 'cf' and saved['variant'] == 'cf_lr1e3' and saved['split'] == 'test'
                    and saved['selection_lock_sha256'] == provenance['lock_sha256']
                    and saved['candidate_policy'] == eval_cfg['candidate_policy'], 'Old strong CF test metrics identity differs')
            require(verification['seed'] == seed and verification['original_sha256'] == file_digest(path)
                    and verification['reloaded_sha256'] == file_digest(reloaded)
                    and verification['identical'] is True and verification['metrics_identical'] is True,
                    'Old strong CF replay verification differs')
            metrics, row_count = test_metrics(path, records, {**eval_cfg, 'n_items': identity['items']},
                                              seed, 'cf', _same, _ids)
            require(row_count == len(records), 'Old strong CF test population changed')
            _same(metrics, saved['metrics'], 'Old strong CF saved test metrics')
            _same(metrics, job['test'], 'Old strong CF finalized comparison metrics')
            hashes.update({str(p): file_digest(p) for p in (reloaded, saved_path, verification_path)})
        hashes[str(path)] = file_digest(path)
        anchor_values[seed] = metrics
        values[seed, 'strong_cf_anchor', 'frozen'] = metrics

    aggregated = []
    keys = sorted({(v, stage) for _, v, stage in values})
    for variant, stage in keys:
        for group in anchor_values[cfg['seeds'][0]]:
            for metric in (f'{m}{k}' for k in eval_cfg['ks'] for m in ('recall', 'ndcg', 'hit')):
                scores = [values[seed, variant, stage][group][metric] for seed in cfg['seeds']]
                aggregated.append({'variant': variant, 'stage': stage, 'group': group, 'metric': metric,
                    'seed_ids': cfg['seeds'], 'per_seed_scores': scores, 'mean': statistics.mean(scores),
                    'sample_std': statistics.stdev(scores) if len(scores) > 1 else None})

    specs = {v['name']: v for v in cfg['variants']}
    pairs, seen_pairs, missing = [], set(), []
    def add(left, right, label, attribution):
        if left == right or (left, right, label) in seen_pairs:
            return
        seen_pairs.add((left, right, label))
        pairs.append((left, right, label, attribution))
    def match(left, right, ignored=()):
        return all(left[k] == right[k] for k in research.VARIANT_DEFAULTS if k not in {'kind', *ignored})
    for variant in cfg['variants']:
        for stage in stages:
            add((variant['name'], stage), ('strong_cf_anchor', 'frozen'), 'candidate-minus-frozen-CF',
                'Total benefit of the whole additional expert training, objective, calibration and fusion recipe; not isolated text benefit.')
            if variant['kind'] in {'text', 'shuffled'}:
                controls = [v for v in cfg['variants'] if v['kind'] == 'id' and match(v, variant)]
                if not controls:
                    missing.append({'variant': variant['name'], 'stage': stage, 'missing': 'ID with identical objective, score scale, alignment, auxiliary weight and learning rate'})
                for control in controls:
                    add((variant['name'], stage), (control['name'], stage), 'semantic-minus-matched-ID',
                        'Strong ID continuation/ensemble control with the same declared optimization and selection recipe. It is not capacity-matched: on ML-1M the ID adapter has about 248k parameters versus about 65.6k for the text projection, and their initial scores differ. Excess improvement is evidence relative to this strong ID control, not a pure semantic effect isolated from parameterization and initialization.')
            if variant['kind'] == 'text':
                controls = [v for v in cfg['variants'] if v['kind'] == 'shuffled' and match(v, variant)]
                if not controls:
                    missing.append({'variant': variant['name'], 'stage': stage, 'missing': 'shuffled text with identical recipe'})
                for control in controls:
                    add((variant['name'], stage), (control['name'], stage), 'text-minus-shuffled',
                        'Preserves feature distribution and capacity; tests whether correct item-text correspondence is useful.')
            if variant['alignment_weight'] > 0:
                controls = [v for v in cfg['variants'] if v['kind'] == variant['kind'] and v['alignment_weight'] == 0
                            and match(v, variant, {'alignment_weight'})]
                for control in controls:
                    add((variant['name'], stage), (control['name'], stage), 'alignment-minus-no-alignment',
                        'Same feature source and recipe apart from alignment weight; measures alignment contribution within this model.')
        if cfg['train_router']:
            add((variant['name'], 'router'), (variant['name'], 'expert'), 'relative-router-minus-global',
                'Same selected expert snapshot; measures router-stage training and adaptive routing together, with the same alpha grid.')
    for left in cfg['variants']:
        for right in cfg['variants']:
            if (left['kind'] == right['kind'] and left['objective'] == 'ce' and right['objective'] == 'bpr'
                    and left['alignment_weight'] == right['alignment_weight']):
                changed = [k for k in research.VARIANT_DEFAULTS if left[k] != right[k]]
                for stage in stages:
                    add((left['name'], stage), (right['name'], stage), 'CE-recipe-minus-BPR-recipe',
                        'Changed fields: ' + ', '.join(changed) + '. When score scale also differs this is a recipe comparison, not an isolated CE effect.')
    comparisons = []
    for left, right, label, attribution in pairs:
        for group in anchor_values[cfg['seeds'][0]]:
            for metric in (f'{m}{k}' for k in eval_cfg['ks'] for m in ('recall', 'ndcg')):
                left_scores = [values[seed, *left][group][metric] for seed in cfg['seeds']]
                right_scores = [values[seed, *right][group][metric] for seed in cfg['seeds']]
                statistics_value = paired_seed_comparison([a - b for a, b in zip(left_scores, right_scores)],
                    bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
                baseline_mean = statistics.mean(right_scores)
                comparisons.append({'comparison': label, 'candidate': list(left), 'baseline': list(right),
                    'attribution': attribution, 'group': group, 'metric': metric, 'seed_ids': cfg['seeds'],
                    'candidate_scores': left_scores, 'baseline_scores': right_scores,
                    'relative_mean_gain_percent': 100. * statistics_value['mean_difference'] / baseline_mean if baseline_mean else None,
                    **statistics_value})
    _holm_adjust(comparisons)
    selection = read(root / 'selection.json')
    require(selection['fingerprint'] == manifest['fingerprint'] and selection['split'] == 'validation', 'Selection identity changed')
    validation_results = [{'seed': s, 'variant': v, 'stage': stage, 'validation': indexed[s, v, stage]['validation']}
                          for s, v, stage in sorted(expected)]
    _same(research._aggregate(validation_results, cfg), selection['candidates'], 'Whole-grid validation selection')
    require((selection['selected_variant'], selection['selected_stage'])
            == (selection['candidates'][0]['variant'], selection['candidates'][0]['stage']), 'Validation selection winner changed')
    # Detect changes during the audit, including an accidentally running writer.
    for path, checksum in hashes.items():
        require(file_digest(path) == checksum, 'Artifact changed during comparison: ' + path)
    result = {'schema': 'research_v3_controlled_comparison_v1', 'split': args.split,
        'fingerprint': manifest['fingerprint'], 'seed_ids': cfg['seeds'], 'job_count': len(expected),
        'cpu_checkpoint_identity_verified': True, 'model_evaluation_called': False,
        'helper_sha256': file_digest(Path(__file__).resolve()),
        'helper_file': str(Path(__file__).resolve()),
        'complete_configured_grid_verified': True, 'all_ranking_metrics_independently_recomputed': True,
        'reference': provenance, 'selection_budget': {k: cfg[k] for k in ('epochs', 'min_epochs', 'patience',
            'eval_interval', 'batch_size', 'alpha_grid', 'routes', 'train_router', 'router_epochs', 'router_min_epochs', 'router_lr')},
        'variants': cfg['variants'], 'selected_validation_variant': selection['selected_variant'],
        'selected_validation_stage': selection['selected_stage'], 'selections': selections,
        'aggregates': aggregated, 'paired_comparisons': comparisons, 'missing_matched_controls': missing,
        'artifact_hashes': hashes,
        'uncertainty_scope': 'Training seeds conditional on the same fixed users. Seed bootstrap intervals are descriptive with few seeds; repeated users across seeds are not independent samples.',
        'signflip_minimum_two_sided_p': 2. / (2 ** len(cfg['seeds'])) if len(cfg['seeds']) > 1 else None,
        'single_seed_inference': 'Unavailable: no confidence interval or p-value.' if len(cfg['seeds']) == 1 else None,
        'multiple_comparisons': 'Holm adjustment across every available p-value in this output; comparisons are exploratory.',
        'iteration_boundary': 'Historical test scores were previously exposed. All new model and calibration selection is validation-only; this is not an untouched confirmatory holdout.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    result = execute(_args(argv))
    print(json.dumps({'jobs': result['job_count'], 'split': result['split'],
                      'verified': True, 'matched_control_gaps': len(result['missing_matched_controls'])}))


if __name__ == '__main__':
    main()
