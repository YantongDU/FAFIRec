"""Train-internal diagnostic fusion; outer validation selects explicit recipes.

ACE, EASE and popularity correction are credited external/simple controls.
The experimental contribution under study is out-of-fit diagnostic control of
fusion and an optional identity-conditioned relation alignment. No test scorer
is present in this pilot entry point.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.multitarget_metrics import evaluate_topk, summarize_multitarget_rows
from fafirec.ratio_runner import write_rows
from fafirec.runner import cpu_state, digest, file_digest, json_dump, seed_everything, torch_dump
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import inner_teacher, coverage_probe
from experiments.diagnostic_v4.fusion_model import DiagnosticGate, score_histories
from experiments.diagnostic_v4.relation_model import fit_relation, semantic_relation
from experiments.diagnostic_v4.longterm_probe import candidate_standardize

PROTOCOL = 'fafirec_v4_internal_diagnostic_fusion_validation_v1'
RECIPE = dict(ridge=5000., gate_epochs=20, gate_lr=.02, gate_batch=256,
    train_alpha=.1, train_beta=.2, modes=['global', 'length', 'diagnostic', 'candidate'],
    l2_grid=[.001, .1], alpha_grid=[0., .025, .05, .1, .15, .2], beta_grid=[0., .2],
    alignment_grid=[0., .01, .1], semantic_topk=30, query_window=50,
    popularity_history=51, probe_window=2)


def source_identity():
    names = ['fusion_experiment.py', 'fusion_model.py', 'relation_model.py', 'inner_teacher.py', 'longterm_probe.py', 'coverage_probe.py']
    return {**ace.source_identity(), **{
        'experiments/diagnostic_v4/' + name: file_digest(Path(__file__).parent / name) for name in names}}


def history_matrix(histories, n_items, window=None):
    matrix = torch.zeros(len(histories), n_items, dtype=torch.float64)
    for u, history in enumerate(histories):
        ids = history[-window:] if window else history
        ace.require(ids and all(type(i) is int and 1 <= i <= n_items for i in ids), 'Invalid observed history')
        matrix[u, torch.tensor(ids) - 1] = 1.
    return matrix


def popularity(counts, available):
    prior = np.log1p(counts.detach().cpu().double().numpy())
    return torch.from_numpy(candidate_standardize(np.broadcast_to(prior[None], available.shape), available.numpy()))


def score_operator(B, histories, available):
    query = history_matrix(histories, len(B), 50)
    raw = torch.cat((torch.zeros(len(query), 1, dtype=torch.float64), query @ B), 1).numpy()
    return torch.from_numpy(candidate_standardize(raw, available.numpy()))


def write_metric_rows(scores, records, cfg, seed, name, folder):
    metrics = evaluate_topk(scores, [list(dict.fromkeys(r['validation'])) for r in records],
        [r['train'] for r in records], cfg['ks'], cfg['candidate_policy'])
    rows = []
    for r, m in zip(records, metrics):
        rows.append({'user': str(r['user']), 'seed': seed, 'control': 'fafirec_v4_diagnostic',
            'variant': name, 'stage': 'fusion', 'split': 'validation', 'group': r['group'],
            'training_interactions': len(r['train']), 'candidate_policy': cfg['candidate_policy'],
            'target_count': m['relevant_count'], 'targets': json.dumps(sorted(set(r['validation']))),
            **{k: v for k, v in m.items() if k != 'top_items'}, 'top_items': json.dumps(m['top_items'])})
    value = summarize_multitarget_rows([{'group': r['group'], **{k: v for k, v in m.items() if k != 'top_items'}}
        for r, m in zip(records, metrics)], cfg['ks'])
    folder.mkdir(parents=True, exist_ok=True)
    write_rows(folder / 'validation.csv', rows)
    recomputed = audit_rows(folder / 'validation.csv', records, cfg, seed, name, 'fusion',
        'fafirec_v4_diagnostic', 'validation', scores.shape[1] - 1)
    _same(recomputed, value, 'Diagnostic fusion ranking metrics')
    return value


def prepare(seed, teacher_dir, out, coverage_dir=None, coverage_mode='all_windows'):
    ace.require(not out.exists(), 'Use a new immutable cache directory')
    run = Path(f'E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas/seed_{seed}/run')
    main, raw, cfg, manifest, summary = ace.load_selected(run, seed, 'ace_selected')
    records = [{k: v for k, v in r.items() if k != 'test'} for r in raw]
    del raw
    backbone = {'kind': 'original_retained50_ace', 'run': str(run),
        'checkpoint_sha256': summary['checkpoint_sha256'], 'validation': summary['validation']}
    if coverage_dir is not None:
        cm = ace.read(coverage_dir / 'manifest.json')
        cs = ace.read(coverage_dir / coverage_mode / 'summary.json')
        ace.require(cm['protocol'] == coverage_probe.PROTOCOL and cm['fingerprint'] == ace.fingerprint(cm)
            and cm['source_hashes'] == coverage_probe.source_identity() and cm['identity'] == manifest['identity']
            and cm['config']['seeds'] == [seed] and cm['evaluation_config'] == cfg,
            'Coverage backbone identity differs')
        cp = coverage_dir / coverage_mode / 'best.pt'
        ace.require(cs['checkpoint_sha256'] == file_digest(cp), 'Coverage checkpoint changed')
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        ace.check_checkpoint(saved, coverage_probe.header(cm, coverage_mode, seed, main))
        ace.require(saved['selected_epoch'] == cs['selected_epoch'], 'Coverage epoch selection differs')
        _same(saved['validation'], cs['validation'], 'Coverage selected metric')
        main.load_state_dict(saved['model']); main.eval()
        backbone = {'kind': 'coverage_ace', 'run': str(coverage_dir), 'mode': coverage_mode,
            'checkpoint_sha256': file_digest(cp), 'manifest_sha256': file_digest(coverage_dir / 'manifest.json'),
            'summary_sha256': file_digest(coverage_dir / coverage_mode / 'summary.json'), 'validation': cs['validation']}
    teacher, inner, calibration, teacher_manifest = inner_teacher.load_teacher(teacher_dir)
    ace.require(teacher_manifest['config']['seed'] == seed and teacher_manifest['config']['epochs'] == 135
        and teacher_manifest['inputs']['identity'] == manifest['identity'], 'Diagnostic teacher seed/budget/input differs')
    ace.require([r['user'] for r in records] == [r['user'] for r in inner], 'Calibration user order differs')
    ace.require(all(c['prefix'] + c['calibration'] == r['train'] for c, r in zip(calibration, records)),
        'Calibration is not confined to original train')
    plan = {'protocol': PROTOCOL, 'seed': seed, 'recipe_grid': RECIPE, 'source_hashes': source_identity(),
        'identity': manifest['identity'], 'evaluation_config': cfg, 'ace_run': str(run),
        'ace_checkpoint_sha256': summary['checkpoint_sha256'], 'teacher_dir': str(teacher_dir.resolve()),
        'backbone': backbone,
        'teacher_sha256': file_digest(teacher_dir / 'teacher.pt'),
        'teacher_manifest_sha256': file_digest(teacher_dir / 'manifest.json'),
        'inner_split': teacher_manifest['split'], 'selection_split': 'validation', 'test_evaluated': False,
        'scope': 'Full train collaborative relations; recent50 sequence queries. Extra past training coverage is separately controlled.'}
    inputs = [run / 'manifest.json', run / f'seed_{seed}/ace_selected/best.pt',
        run / f'seed_{seed}/ace_selected/summary.json', teacher_dir / 'manifest.json', teacher_dir / 'teacher.pt']
    if coverage_dir is not None:
        inputs.extend([coverage_dir / 'manifest.json', coverage_dir / coverage_mode / 'best.pt',
            coverage_dir / coverage_mode / 'summary.json', coverage_dir / coverage_mode / 'validation.csv'])
    plan['external_artifact_sha256'] = {str(p): file_digest(p) for p in inputs}
    plan['fingerprint'] = digest(plan)
    out.mkdir(parents=True)
    json_dump(out / 'plan.json', plan)
    n_items = main.n_items
    X_inner = history_matrix([r['train'] for r in inner], n_items)
    X = history_matrix([r['train'] for r in records], n_items)
    B_inner, inner_relation = fit_relation(X_inner, RECIPE['ridge'])
    B, relation = fit_relation(X, RECIPE['ridge'])
    counts_inner = torch.cat((torch.zeros(1), history_matrix([r['train'] for r in inner], n_items, 51).sum(0).float()))
    counts = torch.cat((torch.zeros(1), history_matrix([r['train'] for r in records], n_items, 51).sum(0).float()))
    cutoff = [len(r['prefix']) for r in calibration]
    cal_hist, targets = [], []
    for row in calibration:
        tail = row['calibration']
        ace.require(tail, 'Every user requires an internal calibration target')
        middle = min(len(tail) - 1, len(tail) // 2)
        cal_hist.append(row['prefix'] + tail[:middle])
        targets.append(tail[middle:])
    sem, cf, features, available, train_diag = score_histories(teacher, B_inner, cal_hist, cutoff)
    train = {'sem': sem, 'cf': cf, 'features': features, 'available': available,
        'targets': targets, 'counts': counts_inner, 'pop': popularity(counts_inner, available)}
    observed = [r['train'] for r in records]
    _, _, diag, _, eval_diag = score_histories(teacher, B_inner, observed, cutoff)
    sem, _, _, available, _ = score_histories(main, B, observed, cutoff, probe_window=0)
    baseline = write_metric_rows(sem.numpy(), records, cfg, seed, 'backbone_only', out / 'baseline')
    _same(baseline, backbone['validation'], 'Normalized backbone exact ranking fallback')
    # Use float64 collaborative multiplication consistently for every candidate.
    cf = score_operator(B, observed, available)
    val = {'sem': sem, 'cf': cf, 'features': diag, 'available': available,
        'counts': counts, 'pop': popularity(counts, available)}
    feature_cfg = ace.normalize_config(ace.read(run / 'manifest.json')['config'])
    _, feature_tensor, *_ = ace.inputs(feature_cfg)
    prior, semantic_report = semantic_relation(feature_tensor[1:].double(), X.any(0), RECIPE['semantic_topk'])
    torch_dump(out / 'calibration.pt', train)
    torch_dump(out / 'validation.pt', val)
    torch_dump(out / 'relations.pt', {'B': B, 'B_inner': B_inner, 'X': X, 'prior': prior})
    json_dump(out / 'records_validation.json', records)
    meta = {'fingerprint': plan['fingerprint'], 'relations': relation, 'inner_relation': inner_relation,
        'semantic_relation': semantic_report, 'training_diagnostics': train_diag, 'validation_diagnostics': eval_diag,
        'artifact_sha256': {str(p): file_digest(p) for p in [out / 'calibration.pt', out / 'validation.pt',
            out / 'relations.pt', out / 'records_validation.json']}}
    json_dump(out / 'cache_manifest.json', meta)
    print('Prepared train-internal calibration and outer validation caches.', flush=True)


def load_cache(path):
    plan = ace.read(path / 'plan.json')
    ace.require(plan['source_hashes'] == source_identity() and plan['recipe_grid'] == RECIPE
        and plan['fingerprint'] == digest({k: v for k, v in plan.items() if k != 'fingerprint'}), 'Cache/source plan differs')
    meta = ace.read(path / 'cache_manifest.json')
    ace.require(meta['fingerprint'] == plan['fingerprint'], 'Cache fingerprint differs')
    for file, sha in meta['artifact_sha256'].items():
        ace.require(file_digest(file) == sha, 'Diagnostic cache changed: ' + file)
    for file, sha in plan['external_artifact_sha256'].items():
        ace.require(file_digest(file) == sha, 'Diagnostic source artifact changed: ' + file)
    return plan, torch.load(path / 'calibration.pt', map_location='cpu', weights_only=True), \
        torch.load(path / 'validation.pt', map_location='cpu', weights_only=True)


def fit_gate(data, mode, l2, seed, device):
    seed_everything(seed + 9000)
    model = DiagnosticGate(n_features=data['features'].shape[1], mode=mode)
    model.fit_feature_statistics(data['features'])
    if mode == 'candidate':
        model.fit_item_statistics(data['counts'])
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return model.cpu(), []
    optimizer = torch.optim.Adam(parameters, lr=RECIPE['gate_lr'])
    maximum = max(map(len, data['targets']))
    target = torch.empty(len(data['targets']), maximum, dtype=torch.long)
    target_mask = torch.zeros_like(target, dtype=torch.float32)
    for u, ids in enumerate(data['targets']):
        target[u].fill_(ids[0]); target[u, :len(ids)] = torch.tensor(ids)
        target_mask[u, :len(ids)] = 1. / len(ids)
        ace.require(bool(data['available'][u, ids].all()), 'Calibration target already observed')
    logs = []
    model.train()
    for epoch in range(1, RECIPE['gate_epochs'] + 1):
        order = torch.randperm(len(target), generator=torch.Generator().manual_seed(seed + 10000 + epoch))
        total = 0.
        for index in order.split(RECIPE['gate_batch']):
            sem = (data['sem'][index] + RECIPE['train_beta'] * data['pop'][index]).to(device)
            scores = model(sem, data['cf'][index].to(device), data['features'][index].to(device),
                data['counts'].to(device), alpha=RECIPE['train_alpha'])
            logp = scores.masked_fill(~data['available'][index].to(device), -torch.inf).log_softmax(1)
            loss = -(logp.gather(1, target[index].to(device)) * target_mask[index].to(device)).sum(1).mean()
            penalty = torch.cat([p.flatten() for p in parameters]).square().mean()
            objective = loss + float(l2) * penalty
            ace.require(bool(torch.isfinite(objective)), 'Nonfinite diagnostic gate loss')
            optimizer.zero_grad(set_to_none=True)
            objective.backward(); optimizer.step()
            total += float(loss.detach()) * len(index)
        logs.append({'epoch': epoch, 'user_mean_calibration_ce': total / len(target)})
    return model.cpu().eval(), logs


@torch.no_grad()
def predict(model, data, alpha, beta):
    return model(data['sem'] + beta * data['pop'], data['cf'], data['features'],
        data['counts'], alpha=alpha).numpy()


def pilot(cache, out):
    ace.require(not out.exists(), 'Choose a new immutable pilot output')
    plan, training, val = load_cache(cache)
    records = ace.read(cache / 'records_validation.json')
    out.mkdir(parents=True)
    json_dump(out / 'plan.json', {'protocol': PROTOCOL, 'cache_plan_sha256': file_digest(cache / 'plan.json'),
        'cache_manifest_sha256': file_digest(cache / 'cache_manifest.json'), 'source_hashes': source_identity(),
        'recipe_grid': RECIPE, 'selection_split': 'validation', 'test_evaluated': False})
    candidates, gates = [], {}
    for mode in RECIPE['modes']:
        for l2 in ([None] if mode == 'global' else RECIPE['l2_grid']):
            name = mode if l2 is None else mode + '_l2_' + str(l2).replace('.', 'p')
            model, logs = fit_gate(training, mode, l2, plan['seed'], torch.device('cuda'))
            gates[name] = model
            folder = out / name
            folder.mkdir()
            torch_dump(folder / 'gate.pt', {'mode': mode, 'l2': l2, 'model': cpu_state(model),
                'fingerprint': plan['fingerprint'], 'training': logs})
            for beta in RECIPE['beta_grid']:
                for alpha in RECIPE['alpha_grid']:
                    variant = name + f'_a{alpha:g}_b{beta:g}'
                    scores = predict(model, val, alpha, beta)
                    result = write_metric_rows(scores, records, plan['evaluation_config'], plan['seed'], variant, out / variant)
                    row = {'variant': variant, 'mode': mode, 'gate': name, 'l2': l2, 'alpha': alpha, 'beta': beta,
                        'alignment_weight': 0., 'validation': result, 'score': result['all']['ndcg50']}
                    candidates.append(row)
            best = max(r['score'] for r in candidates if r['gate'] == name)
            print(json.dumps({'gate': name, 'best_validation_ndcg50': best}), flush=True)
            json_dump(out / 'candidates.json', candidates)
    # Optional alignment is evaluated for the best complete diagnostic gate,
    # plus a uniform alignment control, with all candidate alpha/beta values retained.
    best_diag = sorted([r for r in candidates if r['mode'] in {'diagnostic', 'candidate'}],
        key=lambda r: (-r['score'], r['variant']))[0]
    relations = torch.load(cache / 'relations.pt', map_location='cpu', weights_only=True)
    for gate_name in ('global', best_diag['gate']):
        gate = gates[gate_name]
        weights = (2 * torch.sigmoid(-gate.coefficients(val['features']))).double().detach()
        for mu in RECIPE['alignment_grid'][1:]:
            B, report = fit_relation(relations['X'], RECIPE['ridge'], relations['prior'], mu, weights)
            tag = gate_name + f'_mu{mu:g}'
            folder = out / tag
            folder.mkdir()
            torch_dump(folder / 'relation.pt', {'B': B, 'weights': weights, 'report': report})
            aligned = {**val, 'cf': score_operator(B, [r['train'] for r in records], val['available'])}
            for beta in RECIPE['beta_grid']:
                for alpha in RECIPE['alpha_grid']:
                    variant = tag + f'_a{alpha:g}_b{beta:g}'
                    result = write_metric_rows(predict(gate, aligned, alpha, beta), records,
                        plan['evaluation_config'], plan['seed'], variant, out / variant)
                    candidates.append({'variant': variant, 'mode': gate.mode, 'gate': gate_name,
                        'l2': best_diag['l2'] if gate_name != 'global' else None,
                        'alpha': alpha, 'beta': beta, 'alignment_weight': mu,
                        'validation': result, 'score': result['all']['ndcg50']})
            json_dump(out / 'candidates.json', candidates)
    candidates.sort(key=lambda r: (-r['score'], r['variant']))
    artifact_paths = list(out.glob('*/gate.pt')) + list(out.glob('*/relation.pt')) + list(out.glob('*/validation.csv'))
    selection = {'protocol': PROTOCOL, 'selection_split': 'validation', 'seed': plan['seed'],
        'cache_fingerprint': plan['fingerprint'], 'selected': candidates[0], 'candidates': candidates,
        'candidate_count': len(candidates), 'source_hashes': source_identity(), 'test_evaluated': False,
        'artifact_sha256': {str(p): file_digest(p) for p in artifact_paths}}
    json_dump(out / 'selection.json', selection)
    print(json.dumps({'selected': candidates[0]['variant'], 'validation_ndcg50': candidates[0]['score']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--teacher-dir', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path); p.add_argument('--coverage-mode', choices=coverage_probe.MODES, default='all_windows')
    p = sub.add_parser('pilot'); p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'prepare':
        prepare(args.seed, args.teacher_dir.resolve(), args.output_dir.resolve(),
            args.coverage_dir.resolve() if args.coverage_dir else None, args.coverage_mode)
    else:
        pilot(args.cache_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
