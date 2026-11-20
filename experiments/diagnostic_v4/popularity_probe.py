"""Validation-only popularity controls for the independent-view probe."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from experiments.diagnostic_v4.longterm_probe import candidate_standardize, rows_and_metrics
from tools import ace2026_experiment as ace
from fafirec.runner import digest, file_digest, json_dump
from fafirec.ratio_runner import write_rows
from analyze_research import audit_rows
from analyze_transfer import _same


def main():
    out = ROOT / 'runs/diagnostic_v4_popularity_probe'
    ace.require(not out.exists(), 'Use a fresh popularity control output')
    torch.set_num_threads(4)
    run = Path('E:/论文/FAFIRec_ACE2026/runs/ace2026_final_replicas/seed_2025/run')
    model, raw, cfg, manifest, selected = ace.load_selected(run, 2025, 'ace_selected')
    records = [{k: v for k, v in r.items() if k != 'test'} for r in raw]
    del raw, model
    probe = ROOT / 'runs/diagnostic_v4_longterm_probe'
    previous = ace.read(probe / 'manifest.json')
    ace.require(previous['input_identity'] == manifest['identity'], 'Probe inputs differ')
    logits_path = probe / 'ace_validation_logits.npy'
    logits = np.load(logits_path, allow_pickle=False)
    available = np.ones_like(logits, dtype=bool)
    available[:, 0] = False
    for u, r in enumerate(records):
        available[u, r['train']] = False
    base = candidate_standardize(logits, available)
    alphas = [0., .01, .025, .05, .1, .2, .4]
    protocol = {'split': 'validation', 'alpha_grid': alphas, 'fit_history': ['full', 'retained51'],
        'transform': ['count', 'log1p'], 'test_evaluated': False,
        'source_hashes': {**ace.source_identity(), str(Path(__file__).relative_to(ROOT)): file_digest(__file__),
            'experiments/diagnostic_v4/longterm_probe.py': file_digest(ROOT / 'experiments/diagnostic_v4/longterm_probe.py')},
        'identity': manifest['identity'], 'ace_checkpoint_sha256': selected['checkpoint_sha256'],
        'ace_logits_sha256': file_digest(logits_path), 'evaluation_config': cfg}
    protocol['fingerprint'] = digest(protocol)
    out.mkdir(parents=True)
    json_dump(out / 'manifest.json', protocol)
    rows, metrics = rows_and_metrics(logits, records, cfg, 'ace_only')
    _same(metrics, selected['validation'], 'ACE popularity-control reference')
    candidates = [{'variant': 'ace_only', 'validation': metrics, 'score': metrics['all']['ndcg50']}]
    best, best_rows = candidates[0], rows
    for history in protocol['fit_history']:
        counts = np.zeros(logits.shape[1], dtype=np.float64)
        for r in records:
            np.add.at(counts, r['train'] if history == 'full' else r['train'][-51:], 1.)
        for transform in protocol['transform']:
            prior = np.log1p(counts) if transform == 'log1p' else counts
            normalized = candidate_standardize(np.broadcast_to(prior[None], logits.shape), available)
            for alpha in alphas[1:]:
                name = f'{history}_{transform}_a{alpha:g}'
                rows, metrics = rows_and_metrics(base + alpha * normalized, records, cfg, name)
                row = {'variant': name, 'fit_history': history, 'transform': transform,
                    'alpha': alpha, 'validation': metrics, 'score': metrics['all']['ndcg50']}
                candidates.append(row)
                if row['score'] > best['score']:
                    best, best_rows = row, rows
    write_rows(out / 'selected_validation.csv', best_rows)
    checked = audit_rows(out / 'selected_validation.csv', records, cfg, 2025,
        best['variant'], 'probe', 'longterm_cf_semantic_probe', 'validation', manifest['identity']['items'])
    _same(checked, best['validation'], 'Popularity selected ranking audit')
    json_dump(out / 'selection.json', {'fingerprint': protocol['fingerprint'], 'selected': best,
        'candidates': candidates, 'candidate_count': len(candidates),
        'selected_csv_sha256': file_digest(out / 'selected_validation.csv')})
    print(f"Popularity control: {best['variant']} validation NDCG@50={best['score']:.12f}", flush=True)


if __name__ == '__main__':
    main()
