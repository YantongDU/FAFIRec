"""Fixed-recipe five-seed training and independently reconstructed rank replay.

Outer validation selects one common recipe only after all seeds are complete.
Test scoring requires an immutable complete five-seed validation lock. Cached
validation features speed training; replay reconstructs every score and feature
from the frozen coverage backbone, internal teacher and observed histories.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.multitarget_metrics import evaluate_topk, summarize_multitarget_rows
from fafirec.ratio_runner import write_rows
from fafirec.runner import cpu_state, digest, file_digest, json_dump, torch_dump
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import coverage_probe, fusion_experiment as fusion, inner_teacher
from experiments.diagnostic_v4.fusion_model import DiagnosticGate, score_histories
from experiments.diagnostic_v4.relation_model import fit_relation

PROTOCOL = 'fafirec_v4_final_run_v1'
PLAN_PROTOCOL = 'fafirec_v4_recipe_plan_v1'
LOCK_PROTOCOL = 'fafirec_v4_five_seed_validation_lock_v1'
SEEDS = [2025, 2026, 2027, 2028, 2029]
MODEL_FIELDS = ('mode', 'l2', 'alpha', 'beta', 'alignment_weight')


def source_identity():
    return {**fusion.source_identity(), Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def fingerprint(value):
    return digest({k: v for k, v in value.items() if k != 'fingerprint'})


def verify_hashes(hashes, context):
    ace.require(isinstance(hashes, dict) and hashes, context + ' requires explicit artifact hashes')
    for filename, checksum in hashes.items():
        ace.require(Path(filename).is_absolute() and file_digest(filename) == checksum,
                    context + ' artifact changed: ' + filename)


def verify_recipe_plan(path):
    path = Path(path).resolve()
    plan = ace.read(path)
    ace.require(plan.get('protocol') == PLAN_PROTOCOL and plan.get('seed_ids') == SEEDS
                and plan.get('source_hashes') == fusion.source_identity()
                and plan.get('selection_split') == 'validation' and plan.get('new_test_used') is False,
                'Fixed validation-only recipe plan differs')
    ace.require(plan.get('fingerprint') == fingerprint(plan), 'Recipe plan fingerprint differs')
    recipes, names = plan.get('recipes'), set()
    ace.require(isinstance(recipes, list) and recipes, 'Explicit nonempty recipes required')
    for recipe in recipes:
        name = recipe.get('name')
        ace.require(isinstance(name, str) and re.fullmatch('[A-Za-z][A-Za-z0-9_-]*', name)
                    and name not in names, 'Distinct safe recipe names required')
        names.add(name)
        ace.require(all(k in recipe for k in MODEL_FIELDS) and recipe['mode'] in DiagnosticGate.MODES,
                    'Recipe model fields missing')
        for k in ('alpha', 'beta', 'alignment_weight'):
            ace.require(isinstance(recipe[k], (int, float)) and not isinstance(recipe[k], bool)
                        and math.isfinite(recipe[k]) and recipe[k] >= 0, 'Invalid recipe ' + k)
        if recipe['mode'] == 'global':
            ace.require(recipe['l2'] is None, 'Global control has no gate optimizer')
        else:
            ace.require(isinstance(recipe['l2'], (int, float)) and not isinstance(recipe['l2'], bool)
                        and math.isfinite(recipe['l2']) and recipe['l2'] >= 0, 'Invalid gate regularization')
    verify_hashes(plan['artifact_sha256'], 'Recipe selection')
    return plan


def make_rows(scores, records, cfg, seed, name, split):
    ace.require(split in {'validation', 'test'}, 'Unknown evaluation split')
    targets = [list(dict.fromkeys(r[split])) for r in records]
    observed = [r['train'] + (r['validation'] if split == 'test' else []) for r in records]
    values = evaluate_topk(scores, targets, observed, cfg['ks'], cfg['candidate_policy'])
    rows = []
    for record, metric in zip(records, values):
        rows.append({'user': str(record['user']), 'seed': seed, 'control': 'fafirec_v4_diagnostic',
            'variant': name, 'stage': 'fusion', 'split': split, 'group': record['group'],
            'training_interactions': len(record['train']), 'candidate_policy': cfg['candidate_policy'],
            'target_count': metric['relevant_count'], 'targets': json.dumps(sorted(set(record[split]))),
            **{k: v for k, v in metric.items() if k != 'top_items'}, 'top_items': json.dumps(metric['top_items'])})
    metrics = summarize_multitarget_rows([{'group': record['group'],
        **{k: v for k, v in metric.items() if k != 'top_items'}}
        for record, metric in zip(records, values)], cfg['ks'])
    return rows, metrics


def write_scored(folder, scores, records, cfg, seed, name, split, filename):
    rows, metrics = make_rows(scores, records, cfg, seed, name, split)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    ace.require(not path.exists(), 'Ranking output already exists')
    write_rows(path, rows)
    checked = audit_rows(path, records, cfg, seed, name, 'fusion', 'fafirec_v4_diagnostic',
        split, scores.shape[1] - 1)
    _same(checked, metrics, 'Independently recomputed final rankings')
    return metrics


def recipe_key(recipe):
    return digest({k: recipe[k] for k in ('mode', 'l2')})[:16]


def load_gate(saved, recipe):
    ace.require(saved['recipe'] == recipe and saved['protocol'] == PROTOCOL, 'Saved recipe differs')
    gate = DiagnosticGate(mode=recipe['mode'])
    gate.load_state_dict(saved['gate_model'], strict=True)
    ace.require(gate.feature_statistics_fitted.item(), 'Saved gate lacks fixed feature normalization')
    ace.require(recipe['mode'] != 'candidate' or gate.item_statistics_fitted.item(),
                'Saved candidate gate lacks fixed item normalization')
    return gate.requires_grad_(False).eval()


def train(seed, components_root, recipe_plan, output_dir, resume=False):
    out = Path(output_dir).resolve()
    recipe_file = Path(recipe_plan).resolve()
    plan = verify_recipe_plan(recipe_file)
    ace.require(seed in SEEDS, 'Known final seed required')
    if out.exists():
        ace.require(resume, 'Existing final run requires --resume')
        if (out / 'summary.json').exists():
            manifest, summary, *_ = load_run(out)
            ace.require(manifest['seed'] == seed and manifest['recipe_plan_sha256'] == file_digest(recipe_file)
                and manifest['components_root'] == str(Path(components_root).resolve()), 'Resumed final job scope differs')
            return summary
        ace.require(set(p.name for p in out.iterdir()) <= {'cache', 'manifest.json'},
            'Partial model outputs are immutable; choose a fresh run directory')
    components = Path(components_root).resolve() / f'seed_{seed}'
    teacher_dir, coverage_dir = components / 'teacher', components / 'coverage'
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'cache'
    if not cache.exists():
        fusion.prepare(seed, teacher_dir, cache, coverage_dir, 'all_windows')
    cache_plan, training, validation = fusion.load_cache(cache)
    ace.require(cache_plan['seed'] == seed and cache_plan['teacher_dir'] == str(teacher_dir.resolve())
        and cache_plan['backbone']['run'] == str(coverage_dir.resolve())
        and cache_plan['backbone']['mode'] == 'all_windows', 'Prepared calibration component scope differs')
    records = ace.read(cache / 'records_validation.json')
    manifest = {'protocol': PROTOCOL, 'seed': seed, 'run_dir': str(out),
        'recipe_plan_file': str(recipe_file), 'recipe_plan_sha256': file_digest(recipe_file),
        'recipes': plan['recipes'], 'source_hashes': source_identity(), 'components_root': str(Path(components_root).resolve()),
        'cache_fingerprint': cache_plan['fingerprint'], 'cache_plan_sha256': file_digest(cache / 'plan.json'),
        'cache_manifest_sha256': file_digest(cache / 'cache_manifest.json'), 'identity': cache_plan['identity'],
        'evaluation_config': cache_plan['evaluation_config'], 'test_evaluated_by_training': False}
    manifest['fingerprint'] = fingerprint(manifest)
    if (out / 'manifest.json').exists():
        ace.require(ace.read(out / 'manifest.json') == manifest, 'Partial run manifest changed')
    else:
        json_dump(out / 'manifest.json', manifest)
    relations = torch.load(cache / 'relations.pt', map_location='cpu', weights_only=True)
    gates, operators, rows = {}, {}, []
    for recipe in plan['recipes']:
        key = recipe_key(recipe)
        if key not in gates:
            gates[key] = fusion.fit_gate(training, recipe['mode'], recipe['l2'], seed, torch.device('cuda'))
        gate, logs = gates[key]
        mu = recipe['alignment_weight']
        relation_key = digest({'gate_key': key if mu else None, 'alignment_weight': mu})[:16]
        if relation_key not in operators:
            if mu == 0:
                B, relation_file = relations['B'], cache / 'relations.pt'
            else:
                weights = gate.alignment_weights(validation['features']).double().detach()
                B, report = fit_relation(relations['X'], fusion.RECIPE['ridge'], relations['prior'], mu, weights)
                relation_file = out / 'relations' / (relation_key + '.pt')
                relation_file.parent.mkdir(exist_ok=True)
                torch_dump(relation_file, {'B': B, 'weights': weights, 'report': report,
                    'fingerprint': manifest['fingerprint'], 'gate_key': key, 'alignment_weight': mu})
            operators[relation_key] = (B, relation_file)
        B, relation_file = operators[relation_key]
        folder = out / recipe['name']
        folder.mkdir()
        saved = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed, 'recipe': recipe,
            'gate_model': cpu_state(gate), 'gate_training': logs, 'relation_file': str(relation_file),
            'relation_sha256': file_digest(relation_file), 'relation_key': 'B'}
        torch_dump(folder / 'model.pt', saved)
        data = {**validation, 'cf': fusion.score_operator(B, [r['train'] for r in records], validation['available'])}
        metrics = write_scored(folder, fusion.predict(gate, data, recipe['alpha'], recipe['beta']), records,
            manifest['evaluation_config'], seed, recipe['name'], 'validation', 'validation.csv')
        row = {'name': recipe['name'], 'recipe': recipe, 'validation': metrics,
            'model_file': str(folder / 'model.pt'), 'model_sha256': file_digest(folder / 'model.pt'),
            'per_user_file': str(folder / 'validation.csv'), 'per_user_sha256': file_digest(folder / 'validation.csv'),
            'relation_file': str(relation_file), 'relation_sha256': saved['relation_sha256']}
        json_dump(folder / 'summary.json', row)
        rows.append(row)
        print(json.dumps({'seed': seed, 'recipe': recipe['name'], 'validation_ndcg50': metrics['all']['ndcg50']}), flush=True)
    artifacts = {str(out / 'manifest.json'): file_digest(out / 'manifest.json'),
        str(recipe_file): file_digest(recipe_file), str(cache / 'plan.json'): file_digest(cache / 'plan.json'),
        str(cache / 'cache_manifest.json'): file_digest(cache / 'cache_manifest.json'),
        **ace.read(cache / 'cache_manifest.json')['artifact_sha256'], **cache_plan['external_artifact_sha256'],
        **plan['artifact_sha256']}
    for row in rows:
        for filename in ('model_file', 'per_user_file', 'relation_file'):
            path = row[filename]
            artifacts[path] = file_digest(path)
        leaf = out / row['name'] / 'summary.json'
        artifacts[str(leaf)] = file_digest(leaf)
    summary = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed,
        'recipes': rows, 'artifact_sha256': artifacts, 'test_evaluated': False}
    json_dump(out / 'summary.json', summary)
    load_run(out)
    return summary


def load_run(run_dir):
    root = Path(run_dir).resolve()
    manifest = ace.read(root / 'manifest.json')
    ace.require(manifest.get('protocol') == PROTOCOL and manifest.get('run_dir') == str(root)
        and manifest.get('source_hashes') == source_identity() and manifest.get('fingerprint') == fingerprint(manifest),
        'Final run source or manifest differs')
    recipe_file = Path(manifest['recipe_plan_file'])
    plan = verify_recipe_plan(recipe_file)
    ace.require(file_digest(recipe_file) == manifest['recipe_plan_sha256'] and plan['recipes'] == manifest['recipes'],
                'Final recipe plan changed')
    cache_plan, training, validation = fusion.load_cache(root / 'cache')
    ace.require(cache_plan['seed'] == manifest['seed'] and cache_plan['identity'] == manifest['identity']
        and cache_plan['fingerprint'] == manifest['cache_fingerprint']
        and file_digest(root / 'cache/plan.json') == manifest['cache_plan_sha256']
        and file_digest(root / 'cache/cache_manifest.json') == manifest['cache_manifest_sha256'], 'Final cache identity differs')
    summary = ace.read(root / 'summary.json')
    ace.require(summary['protocol'] == PROTOCOL and summary['fingerprint'] == manifest['fingerprint']
        and summary['seed'] == manifest['seed'] and summary['test_evaluated'] is False
        and [r['recipe'] for r in summary['recipes']] == manifest['recipes'], 'Final summary recipe coverage differs')
    verify_hashes(summary['artifact_sha256'], 'Final run')
    required = {str(root / 'manifest.json'), str(recipe_file), str(root / 'cache/plan.json'),
                str(root / 'cache/cache_manifest.json')}
    required.update(ace.read(root / 'cache/cache_manifest.json')['artifact_sha256'])
    required.update(cache_plan['external_artifact_sha256'])
    required.update(plan['artifact_sha256'])
    records = ace.read(root / 'cache/records_validation.json')
    for row in summary['recipes']:
        folder = root / row['name']
        ace.require(row == ace.read(folder / 'summary.json') and row['model_file'] == str(folder / 'model.pt')
            and row['per_user_file'] == str(folder / 'validation.csv'), 'Final recipe folder identity differs')
        for key in ('model', 'per_user', 'relation'):
            ace.require(file_digest(row[key + '_file']) == row[key + '_sha256'], 'Final ' + key + ' artifact changed')
            required.add(row[key + '_file'])
        required.add(str(folder / 'summary.json'))
        saved = torch.load(row['model_file'], map_location='cpu', weights_only=True)
        ace.require(saved['fingerprint'] == manifest['fingerprint'] and saved['seed'] == manifest['seed']
            and saved['relation_file'] == row['relation_file'] and saved['relation_sha256'] == row['relation_sha256'],
            'Final model identity differs')
        load_gate(saved, row['recipe'])
        checked = audit_rows(row['per_user_file'], records, manifest['evaluation_config'], manifest['seed'],
            row['name'], 'fusion', 'fafirec_v4_diagnostic', 'validation', manifest['identity']['items'])
        _same(checked, row['validation'], 'Final saved validation rankings')
    ace.require(set(summary['artifact_sha256']) == required, 'Final artifact closure omits or adds files')
    return manifest, summary, cache_plan, training, validation


def verify_lock(path, run_dir=None, seed=None, recipe_plan_sha256=None):
    lock = ace.read(path)
    ace.require(lock.get('protocol') == LOCK_PROTOCOL and lock.get('seed_ids') == SEEDS
        and lock.get('source_hashes') == source_identity()
        and (recipe_plan_sha256 is None or lock.get('recipe_plan_sha256') == recipe_plan_sha256)
        and lock.get('lock_fingerprint') == digest({k: v for k, v in lock.items() if k != 'lock_fingerprint'}),
        'Complete validation lock identity differs')
    jobs = lock.get('jobs')
    ace.require(isinstance(jobs, list) and len(jobs) == 5 and sorted(j['seed'] for j in jobs) == SEEDS
        and len({str(Path(j['run_dir']).resolve()) for j in jobs}) == 5, 'Five distinct final jobs required')
    recipe_plan_sha256 = lock['recipe_plan_sha256']
    if run_dir is not None:
        ace.require(seed is not None and any(j['seed'] == seed
            and str(Path(j['run_dir']).resolve()) == str(Path(run_dir).resolve()) for j in jobs),
            'Test run absent from five-seed lock')
    required = {}
    recipes = None
    for job in jobs:
        root = Path(job['run_dir']).resolve()
        manifest = ace.read(root / 'manifest.json')
        summary = ace.read(root / 'summary.json')
        ace.require(manifest['source_hashes'] == lock['source_hashes'] and manifest['seed'] == job['seed']
            and manifest['recipe_plan_sha256'] == recipe_plan_sha256 and manifest['fingerprint'] == fingerprint(manifest)
            and summary['fingerprint'] == manifest['fingerprint'], 'Locked final job differs')
        names = [r['name'] for r in manifest['recipes']]
        ace.require(recipes is None or names == recipes, 'Locked common recipes differ')
        recipes = names
        required.update(summary['artifact_sha256'])
        required[str(root / 'manifest.json')] = file_digest(root / 'manifest.json')
        required[str(root / 'summary.json')] = file_digest(root / 'summary.json')
    ace.require(lock.get('selected_recipe') in recipes, 'Locked selected recipe absent')
    ace.require(all(lock['artifact_sha256'].get(p) == sha for p, sha in required.items()),
                'Lock omits or changes final artifact closure')
    verify_hashes(lock['artifact_sha256'], 'Five-seed selection lock')
    return lock


def load_observed_records(manifest, split):
    """Authenticated input records; this helper never performs prediction."""
    ace.require(split in {'validation', 'test'}, 'Known split required')
    cache_plan = ace.read(Path(manifest['run_dir']) / 'cache/plan.json')
    model, raw, cfg, original, _ = ace.load_selected(cache_plan['ace_run'], manifest['seed'], 'ace_selected')
    del model
    ace.require(original['identity'] == manifest['identity'] and cfg == manifest['evaluation_config'], 'Audit records differ')
    return [{k: v for k, v in row.items() if k != 'test'} for row in raw] if split == 'validation' else raw


def reconstruct(run_dir, split, manifest, cache_plan):
    """Read components again; no cached outer score/feature is used."""
    root = Path(run_dir).resolve()
    seed = manifest['seed']
    main, raw, cfg, original, _ = ace.load_selected(cache_plan['ace_run'], seed, 'ace_selected')
    if split == 'validation':
        records = [{k: v for k, v in r.items() if k != 'test'} for r in raw]
    else:
        records = raw
    del raw
    ace.require(original['identity'] == manifest['identity'] and cfg == manifest['evaluation_config'], 'Replay inputs differ')
    backbone = cache_plan['backbone']
    ace.require(backbone['kind'] == 'coverage_ace' and backbone['mode'] == 'all_windows', 'Final backbone must cover all windows')
    coverage_dir = Path(backbone['run'])
    cm = ace.read(coverage_dir / 'manifest.json')
    cs = ace.read(coverage_dir / 'all_windows/summary.json')
    cp = coverage_dir / 'all_windows/best.pt'
    ace.require(cm['protocol'] == coverage_probe.PROTOCOL and cm['fingerprint'] == ace.fingerprint(cm)
        and cm['source_hashes'] == coverage_probe.source_identity() and cm['identity'] == manifest['identity']
        and cm['config']['seeds'] == [seed] and cm['evaluation_config'] == cfg
        and file_digest(cp) == backbone['checkpoint_sha256'] == cs['checkpoint_sha256'], 'Replay coverage identity differs')
    saved = torch.load(cp, map_location='cpu', weights_only=True)
    ace.check_checkpoint(saved, coverage_probe.header(cm, 'all_windows', seed, main))
    ace.require(saved['selected_epoch'] == cs['selected_epoch'], 'Replay selected backbone epoch differs')
    main.load_state_dict(saved['model']); main.requires_grad_(False).eval()
    teacher, inner, calibration, tm = inner_teacher.load_teacher(cache_plan['teacher_dir'])
    ace.require(tm['config']['seed'] == seed and tm['config']['epochs'] == 135
        and tm['inputs']['identity'] == manifest['identity']
        and [r['user'] for r in inner] == [r['user'] for r in records]
        and all(c['prefix'] + c['calibration'] == r['train'] for c, r in zip(calibration, records)),
        'Replay diagnostic teacher differs')
    relations = torch.load(root / 'cache/relations.pt', map_location='cpu', weights_only=True)
    histories = [r['train'] + (r['validation'] if split == 'test' else []) for r in records]
    cutoff = [len(r['prefix']) for r in calibration]
    _, _, features, _, _ = score_histories(teacher, relations['B_inner'], histories, cutoff)
    sem, _, _, available, _ = score_histories(main, relations['B'], histories, cutoff, probe_window=0)
    counts = torch.cat((torch.zeros(1), fusion.history_matrix([r['train'] for r in records],
        manifest['identity']['items'], 51).sum(0).float()))
    data = {'sem': sem, 'features': features, 'available': available, 'counts': counts,
        'pop': fusion.popularity(counts, available)}
    return records, histories, data


def replay(run_dir, split, output_dir, selection_lock=None):
    root, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    ace.require(split in {'validation', 'test'} and not out.exists(), 'Known split and new replay directory required')
    # Authenticate the complete scope before loading any test labels or scorer.
    if split == 'test':
        ace.require(selection_lock is not None, 'Test requires complete five-seed validation lock')
        header = ace.read(root / 'manifest.json')
        verify_lock(selection_lock, root, header['seed'], header['recipe_plan_sha256'])
    manifest, summary, cache_plan, _training, _validation = load_run(root)
    del _training, _validation
    records, histories, data = reconstruct(root, split, manifest, cache_plan)
    out.mkdir(parents=True)
    result = []
    score_cache = {}
    for row in summary['recipes']:
        recipe = row['recipe']
        saved = torch.load(row['model_file'], map_location='cpu', weights_only=True)
        gate = load_gate(saved, recipe)
        relation_file = saved['relation_file']
        if relation_file not in score_cache:
            B = torch.load(relation_file, map_location='cpu', weights_only=True)[saved['relation_key']]
            score_cache[relation_file] = fusion.score_operator(B, histories, data['available'])
        values = {**data, 'cf': score_cache[relation_file]}
        folder = out / row['name']
        metrics = write_scored(folder, fusion.predict(gate, values, recipe['alpha'], recipe['beta']), records,
            manifest['evaluation_config'], manifest['seed'], row['name'], split, 'per_user.csv')
        checksum = file_digest(folder / 'per_user.csv')
        value = {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': manifest['seed'],
            'variant': row['name'], 'stage': 'fusion', 'split': split, 'metrics': metrics,
            'model_sha256': row['model_sha256'], 'per_user_sha256': checksum,
            'candidate_policy': manifest['evaluation_config']['candidate_policy']}
        if split == 'validation':
            ace.require(checksum == row['per_user_sha256'], 'Independent validation CSV differs')
            _same(metrics, row['validation'], 'Independent reconstructed validation metrics')
            value['identical_to_original'] = True
        else:
            value['selection_lock_sha256'] = file_digest(selection_lock)
        json_dump(folder / 'metrics.json', value)
        result.append(value)
        print(json.dumps({'seed': manifest['seed'], 'recipe': row['name'], 'split': split,
            'ndcg50': metrics['all']['ndcg50']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--seed', type=int, required=True); t.add_argument('--components-root', type=Path, required=True)
    t.add_argument('--recipe-plan', type=Path, required=True); t.add_argument('--output-dir', type=Path, required=True)
    t.add_argument('--resume', action='store_true')
    r = sub.add_parser('replay')
    r.add_argument('--run-dir', type=Path, required=True); r.add_argument('--split', choices=('validation', 'test'), required=True)
    r.add_argument('--output-dir', type=Path, required=True); r.add_argument('--selection-lock', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    torch.set_num_threads(4)
    {'train': train, 'replay': replay}[command](**args)


if __name__ == '__main__':
    main()
