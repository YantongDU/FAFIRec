"""Validation-only EASE control with ACE's retained training-event horizon.

The operator sees only each user's last 51 training events (the information
covering 50 shifted ACE transitions); queries use their last 50 events.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.diagnostic_v4.longterm_probe import (REGULARIZERS, ALPHAS, SEED, CONTROL,
    rows_and_metrics, candidate_standardize)
from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.runner import digest, file_digest, json_dump, torch_dump
from tools import ace2026_experiment as ace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ace-run', type=Path, default=Path('E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas/seed_2025/run'))
    parser.add_argument('--cache-run', type=Path, default=Path('runs/diagnostic_v4_longterm_probe'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/diagnostic_v4_retained_probe'))
    args = parser.parse_args()
    out, cache = args.output_dir.resolve(), args.cache_run.resolve()
    ace.require(not out.exists(), 'Choose a fresh immutable output')
    torch.set_num_threads(4)
    model, raw_records, cfg, source_manifest, ace_summary = ace.load_selected(args.ace_run, SEED, 'ace_selected')
    records = [{k: v for k, v in r.items() if k != 'test'} for r in raw_records]
    del raw_records, model
    gc.collect()
    torch.cuda.empty_cache()
    original = ace.read(cache / 'manifest.json')
    ace.require(original['train_validation_records_sha256'] == digest(records)
                and original['ace_checkpoint_sha256'] == ace_summary['checkpoint_sha256']
                and original['source_hashes'] == {**ace.source_identity(),
                    'experiments/diagnostic_v4/longterm_probe.py': file_digest(ROOT / 'experiments/diagnostic_v4/longterm_probe.py')},
                'Cache input/source identity differs')
    out.mkdir(parents=True)
    manifest = {'protocol': 'diagnostic_v4_retained51_train_last50_query_probe_v1',
        'split': 'validation', 'seed': SEED, 'regularizers': REGULARIZERS, 'alpha_grid': ALPHAS,
        'train_validation_records_sha256': digest(records), 'input_identity': source_manifest['identity'],
        'evaluation_config': cfg, 'ace_checkpoint_sha256': ace_summary['checkpoint_sha256'],
        'cache_manifest_sha256': file_digest(cache / 'manifest.json'),
        'ace_logits_sha256': file_digest(cache / 'ace_validation_logits.npy'),
        'source_hashes': {**ace.source_identity(),
            'experiments/diagnostic_v4/longterm_probe.py': file_digest(ROOT / 'experiments/diagnostic_v4/longterm_probe.py'),
            Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)},
        'operator_history': 'last51 training events only', 'query_history': 'last50 training events only',
        'normalization': original['score_normalization'], 'fusion': original['fusion'],
        'test_evaluated': False, 'historical_test_known': True,
        'selection': 'all-user validation NDCG50 descending, variant name ascending'}
    manifest['fingerprint'] = digest(manifest)
    json_dump(out / 'manifest.json', manifest)
    n_users, n_items = len(records), source_manifest['identity']['items']
    retained = torch.zeros(n_users, n_items, dtype=torch.float64)
    queries = torch.zeros_like(retained)
    available = np.ones((n_users, n_items + 1), dtype=bool)
    available[:, 0] = False
    for i, r in enumerate(records):
        retained[i, torch.tensor(r['train'][-51:]) - 1] = 1.
        queries[i, torch.tensor(r['train'][-50:]) - 1] = 1.
        available[i, r['train']] = False
    raw = np.load(cache / 'ace_validation_logits.npy', allow_pickle=False)
    ace_rows, ace_metrics = rows_and_metrics(raw, records, cfg, 'ace_only')
    _same(ace_metrics, ace_summary['validation'], 'Cached ACE validation')
    normalized_ace = candidate_standardize(raw, available)
    del raw
    gram = retained.T @ retained
    candidates = [{'variant': 'ace_only', 'kind': 'ace', 'validation': ace_metrics,
        'score': ace_metrics['all']['ndcg50']}]
    best, best_rows = candidates[0], ace_rows
    started = time.monotonic()
    for ridge in REGULARIZERS:
        regularized = gram.clone()
        regularized.diagonal().add_(ridge)
        precision = torch.linalg.inv(regularized)
        operator = -precision / precision.diagonal()[None, :]
        operator.fill_diagonal_(0.)
        ace.require(torch.isfinite(operator).all(), 'Nonfinite EASE operator')
        scores = torch.cat((torch.zeros(n_users, 1), (queries @ operator).float()), dim=1).numpy()
        normalized = candidate_standardize(scores, available)
        for alpha in [None, *ALPHAS[1:]]:
            name = f'ease_retained51_r{int(ridge)}' + ('_only' if alpha is None else f'_a{str(alpha).replace(".", "p")}')
            fused = normalized if alpha is None else normalized_ace + alpha * normalized
            rows, metrics = rows_and_metrics(fused, records, cfg, name)
            candidate = {'variant': name, 'kind': 'ease' if alpha is None else 'fusion',
                'regularizer': ridge, 'alpha': alpha, 'validation': metrics, 'score': metrics['all']['ndcg50']}
            candidates.append(candidate)
            if candidate['score'] > best['score'] or (candidate['score'] == best['score'] and name < best['variant']):
                best, best_rows = candidate, rows
                torch_dump(out / 'selected_ease_operator.pt', {'operator': operator, 'regularizer': ridge,
                    'variant': name, 'rows_and_columns': list(range(1, n_items + 1)),
                    'train_validation_records_sha256': digest(records), 'selected_alpha': alpha,
                    'operator_training_history': 'last51 training events', 'selected_query_history': 'last50',
                    'selection_split': 'validation', 'fingerprint': manifest['fingerprint']})
            print(f"{name} NDCG50={candidate['score']:.12f}", flush=True)
        json_dump(out / 'candidates.json', candidates)
        del regularized, precision, operator, scores, normalized
    candidates.sort(key=lambda r: (-r['score'], r['variant']))
    ace.require(candidates[0]['variant'] == best_rows[0]['variant'], 'Selected ranking mismatch')
    write_path = out / 'selected_validation.csv'
    from fafirec.ratio_runner import write_rows
    write_rows(write_path, best_rows)
    _same(audit_rows(write_path, records, cfg, SEED, best['variant'], 'probe', CONTROL,
                     'validation', n_items), best['validation'], 'Selected retained-history rankings')
    paths = [write_path, out / 'selected_ease_operator.pt']
    result = {'fingerprint': manifest['fingerprint'], 'split': 'validation', 'selected': candidates[0],
        'candidate_count': len(candidates), 'candidates': candidates, 'test_evaluated': False,
        'retained_fit_events': int(retained.sum()), 'query_events': int(queries.sum()),
        'artifact_sha256': {str(p): file_digest(p) for p in paths if p.exists()},
        'elapsed_seconds': time.monotonic() - started}
    json_dump(out / 'selection.json', result)
    print(f"SELECTED {best['variant']} NDCG50={best['score']:.12f}; ACE={ace_metrics['all']['ndcg50']:.12f}", flush=True)


if __name__ == '__main__':
    main()
