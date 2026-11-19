"""Recompute every research metric independently from saved user rankings."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from analyze_results import paired_seed_comparison
from analyze_transfer import _ids, _same
from fafirec.multitarget_metrics import summarize_multitarget_rows
from fafirec.research_runner import (_aggregate, _fingerprint, _new_model, _read,
    _require, _verify_result, load_anchors, normalize_config, source_identity)
from fafirec.runner import file_digest, json_dump


def audit_rows(path, records, cfg, seed, variant, stage, control, split, n_items):
    prepared = {str(r['user']): r for r in records}
    values, seen = [], set()
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            user = row.get('user')
            _require(user in prepared and user not in seen, 'Unknown or duplicate ranking user')
            seen.add(user)
            record = prepared[user]
            _require(row.get('seed') == str(seed) and row.get('variant') == variant
                     and row.get('stage') == stage and row.get('control') == control
                     and row.get('split') == split, 'Ranking job identity differs')
            _require(row.get('candidate_policy') == cfg['candidate_policy']
                     and row.get('group') == record['group'], 'Ranking candidate policy or user group differs')
            target = set(_ids(row.get('targets'), n_items, 'targets'))
            expected = set(record['validation' if split == 'validation' else 'test'])
            _require(target == expected, 'Ranking targets differ from prepared targets')
            top = _ids(row.get('top_items'), n_items, 'top_items')
            observed = set(record['train']) | (set(record['validation']) if split == 'test' else set())
            if cfg['candidate_policy'] == 'exclude_seen':
                _require(not target & observed and not set(top) & observed, 'Seen-history exclusion differs')
                available = n_items - len(observed)
            else:
                _require(cfg['candidate_policy'] == 'full', 'Unknown candidate convention')
                available = n_items
            _require(len(top) == min(max(cfg['ks']), available), 'Incomplete full-catalog ranking')
            _require(int(row['target_count']) == int(row['relevant_count']) == len(target)
                     and int(row['training_interactions']) == len(record['train']), 'Ranking counts differ')
            value = {'user': user, 'group': record['group'], 'relevant_count': len(target)}
            for k in cfg['ks']:
                relevant = [rank for rank, item in enumerate(top[:k]) if item in target]
                dcg = sum(1. / math.log2(rank + 2) for rank in relevant)
                idcg = sum(1. / math.log2(rank + 2) for rank in range(min(k, len(target))))
                for metric, computed in ((f'recall{k}', len(relevant) / len(target)),
                                         (f'ndcg{k}', dcg / idcg), (f'hit{k}', float(bool(relevant)))):
                    _same(computed, float(row[metric]), 'Ranking ' + metric)
                    value[metric] = computed
            values.append(value)
    _require(seen == set(prepared), 'Rankings do not cover the complete prepared population')
    return summarize_multitarget_rows(values, cfg['ks'])


def analyze(run_dir, split='validation', test_dir=None):
    root = Path(run_dir)
    manifest = _read(root / 'manifest.json')
    _require(manifest.get('fingerprint') == _fingerprint(manifest), 'Research manifest fingerprint differs')
    _require(source_identity() == manifest['source_hashes'], 'Research source changed')
    cfg = normalize_config(manifest['config'])
    records, features, identity, _, anchors, provenance = load_anchors(cfg)
    _require(identity == manifest['identity'] and provenance == manifest['reference'], 'Research inputs changed')
    stages = ['expert', 'router'] if cfg['train_router'] else ['expert']
    expected = {(s, v['name'], stage) for s in cfg['seeds'] for v in cfg['variants'] for stage in stages}
    original = _read(root / 'summary.json')
    indexed = {(r['seed'], r['variant'], r['stage']): r for r in original}
    _require(len(indexed) == len(original) and set(indexed) == expected, 'Complete research grid required')
    _require({p.relative_to(root).as_posix() for p in root.glob('seed_*/*/*/summary.json')}
             == {f'seed_{s}/{v}/{stage}/summary.json' for s, v, stage in expected}, 'Extra or missing research stage summaries')
    _require({p.relative_to(root).as_posix() for p in root.glob('seed_*/*/*/best.pt')}
             == {f'seed_{s}/{v}/{stage}/best.pt' for s, v, stage in expected}, 'Extra or missing research selected checkpoints')
    eval_cfg, results, hashes = manifest['evaluation_config'], [], {}
    train_records = [{k: v for k, v in r.items() if k != 'test'} for r in records]
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            model = _new_model(anchors[seed], features, train_records, variant, cfg, seed, torch.device('cpu'))
            for stage in stages:
                result, _ = _verify_result(root, manifest, seed, variant, stage, model)
                _same(result, indexed[seed, variant['name'], stage], 'Summary grid consistency')
                folder = root / f"seed_{seed}/{variant['name']}/{stage}"
                if split == 'validation':
                    path = folder / 'validation.csv'
                    metrics = audit_rows(path, records, eval_cfg, seed, variant['name'], stage,
                                         result['control'], split, identity['items'])
                    _same(metrics, result['validation'], 'Recomputed validation metrics')
                else:
                    _require(test_dir is not None, 'Explicit test replay directory required')
                    destination = Path(test_dir) / f"seed_{seed}/{variant['name']}/{stage}"
                    path = destination / 'per_user.csv'
                    saved = _read(destination / 'metrics.json')
                    _require(saved['fingerprint'] == manifest['fingerprint'] and saved['split'] == 'test'
                             and saved['seed'] == seed and saved['variant'] == variant['name'] and saved['stage'] == stage
                             and saved['per_user_sha256'] == file_digest(path), 'Test replay provenance changed')
                    metrics = audit_rows(path, records, eval_cfg, seed, variant['name'], stage,
                                         result['control'], split, identity['items'])
                    _same(metrics, saved['metrics'], 'Recomputed test metrics')
                hashes[str(path)] = file_digest(path)
                results.append({'seed': seed, 'variant': variant['name'], 'stage': stage, 'validation': metrics})
    comparisons = []
    if cfg['train_router']:
        for variant in cfg['variants']:
            for group in results[0]['validation']:
                for metric in (f'{m}{k}' for k in eval_cfg['ks'] for m in ('recall', 'ndcg')):
                    def value(seed, stage):
                        return next(r for r in results if r['seed'] == seed and r['variant'] == variant['name']
                                    and r['stage'] == stage)['validation'][group][metric]
                    differences = [value(seed, 'router') - value(seed, 'expert') for seed in cfg['seeds']]
                    comparisons.append({'variant': variant['name'], 'comparison': 'router-minus-expert',
                        'group': group, 'metric': metric, **paired_seed_comparison(differences)})
    output = {'protocol': manifest['protocol'], 'fingerprint': manifest['fingerprint'], 'split': split,
        'job_count': len(results), 'all_artifact_hashes_verified': True, 'ranking_hashes': hashes,
        'aggregates': _aggregate(results, cfg), 'paired_comparisons': comparisons,
        'statistical_note': f"{len(cfg['seeds'])} seed units; exact two-sided seed signflip minimum attainable p={2. ** (1 - len(cfg['seeds'])):g}. Exploratory iteration after historical test exposure."}
    if split == 'validation':
        selection = _read(root / 'selection.json')
        _require(selection['fingerprint'] == manifest['fingerprint'] and selection['split'] == 'validation', 'Selection identity differs')
        _same(selection['candidates'], output['aggregates'], 'Recomputed validation selection')
        _require((selection['selected_variant'], selection['selected_stage'])
                 == (output['aggregates'][0]['variant'], output['aggregates'][0]['stage']), 'Selected variant differs')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--test-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a new analysis output path')
    result = analyze(args.run_dir, args.split, args.test_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, result)
    print(json.dumps({'split': result['split'], 'jobs': result['job_count'], 'verified': True}))


if __name__ == '__main__':
    main()
