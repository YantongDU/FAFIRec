"""Validation-only ACE training-coverage controls, preserving frozen old code."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.data import pad_history
from fafirec.ratio_runner import evaluate, write_rows
from fafirec.runner import (cpu_state, digest, file_digest, json_dump, restore_rng,
    rng_state, seed_everything, torch_dump, resolve_device)
from tools import ace2026_experiment as ace

PROTOCOL = 'diagnostic_v4_ace_coverage_validation_v1'
MODES = ('random_window', 'all_windows')


class CoverageDataset(Dataset):
    """Each span identifies target indices [start,end) within a train sequence."""
    def __init__(self, records, max_len, mode, seed, epoch):
        ace.require(mode in MODES and type(max_len) is int and max_len > 0, 'Invalid coverage mode/length')
        ace.require(all('test' not in r and len(r['train']) >= 2 for r in records), 'Training-only records required')
        self.records, self.max_len, self.spans = records, max_len, []
        lengths = [len(r['train']) - 1 for r in records]
        self.offsets = np.cumsum([0, *lengths])
        self.coverage_counts = torch.zeros(int(self.offsets[-1]), dtype=torch.int32)
        generator = np.random.default_rng(seed + 1000 + epoch)
        for user, record in enumerate(records):
            size = len(record['train'])
            if mode == 'random_window':
                end = int(generator.integers(2, size + 1))
                spans = [(max(1, end - max_len), end)]
            else:
                spans = [(start, min(start + max_len, size)) for start in range(1, size, max_len)]
            for start, end in spans:
                self.spans.append((user, start, end))
                offset = int(self.offsets[user])
                self.coverage_counts[offset + start - 1:offset + end - 1] += 1
        if mode == 'all_windows':
            ace.require(bool(self.coverage_counts.eq(1).all()), 'All-window targets must be covered exactly once')

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        user, start, end = self.spans[index]
        sequence = self.records[user]['train']
        hist, pos = sequence[start - 1:end - 1], sequence[start:end]
        return (torch.tensor(pad_history(hist, self.max_len), dtype=torch.long),
                torch.tensor(pad_history(pos, self.max_len), dtype=torch.long))


def source_identity():
    return {**ace.source_identity(), Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def header(manifest, mode, seed, model):
    return {'protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'],
            'mode': mode, 'seed': seed, 'model_config': model.model_config()}


def fit(model, records, cfg, eval_cfg, seed, mode, folder, manifest, resume=False):
    ace.require(all('test' not in r for r in records), 'Test labels exposed to coverage training')
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['variants'][0]['lr'], weight_decay=cfg['weight_decay'])
    expected, device = header(manifest, mode, seed, model), next(model.parameters()).device
    diagnostics = torch.zeros(len(records), 2)
    total = sum(len(r['train']) - 1 for r in records)
    covered = torch.zeros(total, dtype=torch.bool)
    best, snapshot, chosen, stale, logs, first, finished = -math.inf, None, 0, 0, [], 1, False
    started = time.monotonic()
    if resume and (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        ace.check_checkpoint(saved, expected)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, snapshot, chosen, stale = saved['best_metric'], saved['best_snapshot'], saved['selected_epoch'], saved['stale']
        logs, first, finished, covered = saved['logs'], saved['epoch'] + 1, saved['finished'], saved['covered']
        ace.require(covered.shape == (total,) and covered.dtype == torch.bool, 'Resume coverage identity differs')
        restore_rng(saved['rng'])
    else:
        _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
        best, snapshot = metrics['all']['ndcg50'], {'model': cpu_state(model), 'validation': metrics}
        json_dump(folder / 'initial_validation.json', metrics)
    if not finished:
        for epoch in range(first, cfg['epochs'] + 1):
            data = CoverageDataset(records, cfg['max_len'], mode, seed, epoch)
            loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(seed + 1000 + epoch))
            covered |= data.coverage_counts.gt(0)
            model.train()
            totals, positions, steps = {k: 0. for k in ('loss', 'rec_loss', 'reg_loss')}, 0, 0
            for hist, positive in loader:
                optimizer.zero_grad(set_to_none=True)
                hist, positive = hist.to(device), positive.to(device)
                parts = model.loss_batch(hist, positive)
                ace.require(all(torch.isfinite(parts[k]).item() for k in totals), 'Nonfinite coverage loss')
                parts['loss'].backward()
                ace.require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Nonfinite coverage gradient')
                if cfg['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                amount = int((hist.ne(0) & positive.ne(0)).sum())
                for key in totals:
                    totals[key] += float(parts[key].detach()) * amount
                positions += amount
                steps += 1
            ace.require(positions == int(data.coverage_counts.sum()), 'Supervision count differs from target coverage')
            log = {'epoch': epoch, 'optimizer_steps': steps, 'samples': len(data),
                'supervised_positions': positions, 'original_transitions': total,
                'epoch_unique_covered': int(data.coverage_counts.gt(0).sum()),
                'cumulative_unique_covered': int(covered.sum()),
                'cumulative_coverage_fraction': float(covered.float().mean()),
                **{k: v / positions for k, v in totals.items()}}
            if epoch % cfg['eval_interval'] == 0 or epoch == cfg['epochs']:
                _, metrics = evaluate(model, records, 'validation', diagnostics, eval_cfg, seed, False)
                metric, log['validation'] = metrics['all']['ndcg50'], metrics
                if metric > best:
                    best, chosen, stale = metric, epoch, 0
                    snapshot = {'model': cpu_state(model), 'validation': metrics}
                else:
                    stale += 1
                print(json.dumps({'mode': mode, 'epoch': epoch, 'validation_ndcg50': metric,
                    'covered_fraction': log['cumulative_coverage_fraction']}), flush=True)
            finished = epoch == cfg['epochs'] or (epoch >= cfg['min_epochs'] and stale >= cfg['patience'])
            logs.append(log)
            torch_dump(folder / 'last.pt', {**expected, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                'rng': rng_state(), 'best_metric': best, 'best_snapshot': snapshot, 'selected_epoch': chosen,
                'epoch': epoch, 'stale': stale, 'logs': logs, 'finished': finished, 'covered': covered})
            json_dump(folder / 'training.json', logs)
            if finished:
                break
    ace.require(snapshot is not None, 'Missing validation-selected coverage model')
    model.load_state_dict(snapshot['model'])
    torch_dump(folder / 'best.pt', {**expected, **snapshot, 'selected_epoch': chosen, 'selected_metric': best})
    return {'selected_epoch': chosen, 'selected_validation_metric': best, 'epochs_completed': len(logs),
        'optimizer_steps': sum(r['optimizer_steps'] for r in logs),
        'original_transitions': total, 'cumulative_covered': int(covered.sum()),
        'cumulative_coverage_fraction': float(covered.double().mean()),
        'elapsed_seconds_this_invocation': time.monotonic() - started}


def run(cfg, modes=MODES, resume=False):
    cfg = ace.normalize_config(cfg)
    ace.require(len(cfg['seeds']) == len(cfg['variants']) == 1, 'One seed and fixed architecture required')
    torch.set_num_threads(cfg['threads'])
    device = resolve_device(cfg)
    raw, features, identity, base, provenance, factors, fi = ace.inputs(cfg)
    records = [{k: v for k, v in row.items() if k != 'test'} for row in raw]
    del raw
    manifest = ace.manifest_for(cfg, identity, base, provenance, fi, device)
    manifest.update(protocol=PROTOCOL, modes=list(modes), source_hashes=source_identity(),
        training_layout='random per-user end in [2,n], or all blocks covering target indices1..n-1 exactly once; max50; block positions restart',
        records_without_test_sha256=digest(records), test_evaluated=False)
    manifest['fingerprint'] = ace.fingerprint(manifest)
    out = Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()):
        saved = ace.read(out / 'manifest.json')
        ace.require(resume and saved['fingerprint'] == ace.fingerprint(saved) == manifest['fingerprint'],
                    'Coverage run identity changed')
        manifest = saved
    else:
        out.mkdir(parents=True, exist_ok=True)
        json_dump(out / 'manifest.json', manifest)
    results = []
    seed = cfg['seeds'][0]
    for mode in modes:
        seed_everything(seed + 1000)
        model = ace.new_model(features, factors, cfg, cfg['variants'][0], device)
        folder = out / mode
        summary = fit(model, records, cfg, manifest['evaluation_config'], seed, mode, folder, manifest, resume)
        rows, metrics = evaluate(model, records, 'validation', torch.zeros(len(records), 2), manifest['evaluation_config'], seed)
        for row in rows:
            row.update(variant=mode, stage='coverage')
        write_rows(folder / 'validation.csv', rows)
        audited = audit_rows(folder / 'validation.csv', records, manifest['evaluation_config'], seed,
            mode, 'coverage', model.control, 'validation', identity['items'])
        _same(audited, metrics, 'Coverage validation metric recomputation')
        summary.update(validation=metrics, mode=mode, split='validation',
            checkpoint_sha256=file_digest(folder / 'best.pt'), per_user_sha256=file_digest(folder / 'validation.csv'))
        json_dump(folder / 'summary.json', summary)
        results.append(summary)
        json_dump(out / 'summary.json', results)
        del model
    selected = sorted(results, key=lambda r: (-r['validation']['all']['ndcg50'], r['mode']))[0]
    json_dump(out / 'selection.json', {'split': 'validation', 'selected_mode': selected['mode'],
        'validation_ndcg50': selected['validation']['all']['ndcg50'], 'fingerprint': manifest['fingerprint']})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ace-config', type=Path, default=Path('E:/论文/FAFIRec_ACE2026/configs/ace2026_selected.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/diagnostic_v4_coverage_probe'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = ace.load_config(args.ace_config)
    cfg.update(output_dir=str(args.output_dir.resolve()), seeds=[2025])
    results = run(cfg, resume=args.resume)
    print(json.dumps([{k: r[k] for k in ('mode', 'selected_epoch', 'epochs_completed', 'selected_validation_metric',
        'cumulative_coverage_fraction')} for r in results]), flush=True)


if __name__ == '__main__':
    main()
