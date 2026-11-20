"""Validation-only EASE/semantic-sequence complementarity probe.

This is a diagnostic experiment, not a new FAFIRec method. EASE fits binary
full training histories; its long-history and last-50 query controls share
the same fitted operator. No held-out target enters score normalization.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.multitarget_metrics import evaluate_topk, summarize_multitarget_rows
from fafirec.ratio_data import ratio_evaluation_batch
from fafirec.ratio_runner import write_rows
from fafirec.runner import digest, file_digest, json_dump, torch_dump
from tools import ace2026_experiment as ace

REGULARIZERS = [50., 200., 500., 1000., 5000.]
ALPHAS = [0., .1, .25, .5, 1., 2., 4.]
CONTROL = 'longterm_cf_semantic_probe'
SEED = 2025


def rows_and_metrics(scores, records, cfg, name):
    """Reuse the existing evaluator and preserve the independent CSV schema."""
    values = evaluate_topk(scores, [list(dict.fromkeys(r['validation'])) for r in records],
                          [r['train'] for r in records], cfg['ks'], cfg['candidate_policy'])
    rows = []
    for record, metric in zip(records, values):
        rows.append({'user': str(record['user']), 'seed': SEED, 'control': CONTROL,
            'variant': name, 'stage': 'probe', 'split': 'validation', 'group': record['group'],
            'training_interactions': len(record['train']), 'candidate_policy': cfg['candidate_policy'],
            'target_count': metric['relevant_count'],
            'targets': json.dumps(sorted(set(record['validation']))),
            **{k: v for k, v in metric.items() if k != 'top_items'},
            'top_items': json.dumps(metric['top_items'])})
    summary = summarize_multitarget_rows([{'group': record['group'],
        **{k: v for k, v in metric.items() if k != 'top_items'}}
        for record, metric in zip(records, values)], cfg['ks'])
    return rows, summary


def candidate_standardize(scores, available):
    """One scale per observed user prefix, excluding PAD and train history."""
    values = np.asarray(scores, dtype=np.float64)
    count = available.sum(1, keepdims=True)
    mean = np.where(available, values, 0.).sum(1, keepdims=True) / count
    variance = np.where(available, (values - mean) ** 2, 0.).sum(1, keepdims=True) / count
    return ((values - mean) / np.maximum(np.sqrt(variance), 1e-12)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ace-run', type=Path, default=Path('E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas/seed_2025/run'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/diagnostic_v4_longterm_probe'))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    ace.require(not out.exists(), 'Choose a new immutable probe output')
    torch.set_num_threads(4)
    model, raw_records, cfg, source_manifest, ace_summary = ace.load_selected(args.ace_run, SEED, 'ace_selected')
    records = [{k: v for k, v in row.items() if k != 'test'} for row in raw_records]
    del raw_records
    ace.require(all('test' not in r for r in records), 'Test data must be stripped immediately')
    out.mkdir(parents=True)
    manifest = {'protocol': 'diagnostic_v4_validation_only_longterm_probe_v1', 'split': 'validation',
        'seed': SEED, 'source_hashes': {**ace.source_identity(),
            Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)},
        'input_identity': source_manifest['identity'], 'evaluation_config': cfg,
        'train_validation_records_sha256': digest(records),
        'ace_run': str(args.ace_run.resolve()), 'ace_fingerprint': source_manifest['fingerprint'],
        'ace_checkpoint_sha256': ace_summary['checkpoint_sha256'],
        'ace_summary_sha256': file_digest(args.ace_run / 'seed_2025/ace_selected/summary.json'),
        'regularizers': REGULARIZERS, 'alpha_grid': ALPHAS, 'query_histories': ['full', 'last50'],
        'score_normalization': 'per-user mean/std over non-PAD non-observed candidates; targets not indexed',
        'fusion': 'z(ACE) + alpha * z(EASE); separate EASE-only candidates',
        'ease_fit': 'binary user-item train matrix; B=-inverse(XtX+lambdaI)/diag(inverse) by column; diagonal zero',
        'selection': 'all-user validation NDCG@50 only, descending score then variant name',
        'test_evaluated': False, 'historical_test_known': True,
        'interpretation': 'Probe only; not original FAFIRec novelty or matched-capacity claim. EASE operator sees full training histories, whereas the previous neural trainer supervises only its retained last50 transitions.'}
    manifest['fingerprint'] = digest(manifest)
    json_dump(out / 'manifest.json', manifest)
    n_users, n_items = len(records), model.n_items
    available = np.ones((n_users, n_items + 1), dtype=bool)
    available[:, 0] = False
    full = torch.zeros(n_users, n_items, dtype=torch.float64)
    recent = torch.zeros_like(full)
    for index, row in enumerate(records):
        full[index, torch.tensor(row['train']) - 1] = 1.
        recent[index, torch.tensor(row['train'][-50:]) - 1] = 1.
        available[index, row['train']] = False
    logits = np.empty((n_users, n_items + 1), dtype=np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        table = model.item_vectors()
        for start in range(0, n_users, cfg['eval_batch']):
            stop = min(start + cfg['eval_batch'], n_users)
            hist, _ = ratio_evaluation_batch(records, list(range(start, stop)), 'validation', cfg['max_len'])
            logits[start:stop] = (model.score_scale * model.user_vectors(hist.to(device)) @ table.T).cpu().numpy()
    ace_rows, ace_metrics = rows_and_metrics(logits, records, cfg, 'ace_only')
    _same(ace_metrics, ace_summary['validation'], 'Independent ACE validation logits')
    np.save(out / 'ace_validation_logits.npy', logits, allow_pickle=False)
    torch_dump(out / 'binary_history.pt', {'full': full.float(), 'last50': recent.float(),
        'user_order': [str(r['user']) for r in records], 'column_ids': list(range(1, n_items + 1)),
        'train_validation_records_sha256': digest(records), 'observed': 'train only',
        'rows_are_binary': True})
    normalized_ace = candidate_standardize(logits, available)
    del logits, table, model
    gc.collect()
    torch.cuda.empty_cache()
    candidates = [{'variant': 'ace_only', 'kind': 'ace', 'validation': ace_metrics,
                   'score': ace_metrics['all']['ndcg50']}]
    best = candidates[0]
    best_rows = ace_rows
    started = time.monotonic()
    gram = full.T @ full
    for ridge in REGULARIZERS:
        regularized = gram.clone()
        regularized.diagonal().add_(ridge)
        precision = torch.linalg.inv(regularized)
        operator = -precision / precision.diagonal()[None, :]
        operator.fill_diagonal_(0.)
        ace.require(torch.isfinite(operator).all(), 'Nonfinite EASE operator')
        for history, design in (('full', full), ('last50', recent)):
            scores = torch.cat((torch.zeros(n_users, 1), (design @ operator).float()), dim=1).numpy()
            normalized = candidate_standardize(scores, available)
            for alpha in [None, *ALPHAS[1:]]:
                name = f'ease_r{int(ridge)}_{history}' + ('_only' if alpha is None else f'_a{str(alpha).replace(".", "p")}')
                fused = normalized if alpha is None else normalized_ace + alpha * normalized
                rows, metrics = rows_and_metrics(fused, records, cfg, name)
                candidate = {'variant': name, 'kind': 'ease' if alpha is None else 'fusion',
                    'regularizer': ridge, 'history': history, 'alpha': alpha,
                    'validation': metrics, 'score': metrics['all']['ndcg50']}
                candidates.append(candidate)
                if candidate['score'] > best['score'] or (candidate['score'] == best['score'] and name < best['variant']):
                    best, best_rows = candidate, rows
                    torch_dump(out / 'selected_ease_operator.pt', {'operator': operator,
                        'regularizer': ridge, 'variant': name, 'rows_and_columns': list(range(1, n_items + 1)),
                        'training_matrix_file_sha256': file_digest(out / 'binary_history.pt'),
                        'operator_training_history': 'full train binary matrix',
                        'selected_query_history': history, 'selected_alpha': alpha,
                        'selection_split': 'validation', 'fingerprint': manifest['fingerprint']})
                print(json.dumps({k: candidate[k] for k in ('variant', 'score')}) , flush=True)
            del scores, normalized
        del regularized, precision, operator
        json_dump(out / 'candidates.json', candidates)
    candidates.sort(key=lambda row: (-row['score'], row['variant']))
    best = candidates[0]
    ace.require(best_rows[0]['variant'] == best['variant'], 'Tie-selection mismatch')
    write_rows(out / 'selected_validation.csv', best_rows)
    audited = audit_rows(out / 'selected_validation.csv', records, cfg, SEED,
                         best['variant'], 'probe', CONTROL, 'validation', n_items)
    _same(audited, best['validation'], 'Independent selected validation rankings')
    result = {'fingerprint': manifest['fingerprint'], 'split': 'validation',
        'selected': best, 'candidate_count': len(candidates), 'candidates': candidates,
        'per_user_sha256': file_digest(out / 'selected_validation.csv'),
        'artifact_sha256': {str(path): file_digest(path) for path in
            (out / 'ace_validation_logits.npy', out / 'binary_history.pt', out / 'selected_ease_operator.pt')
            if path.exists()},
        'elapsed_seconds': time.monotonic() - started, 'test_evaluated': False}
    json_dump(out / 'selection.json', result)
    print(json.dumps({'selected': best['variant'], 'validation_ndcg50': best['score'],
                      'ace_validation_ndcg50': ace_metrics['all']['ndcg50'],
                      'candidate_count': len(candidates)}), flush=True)


if __name__ == '__main__':
    main()
