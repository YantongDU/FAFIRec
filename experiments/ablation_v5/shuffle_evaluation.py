"""Partial text-identity intervention with every other V4 component frozen."""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from analyze_transfer import _same
from fafirec.runner import digest, file_digest, json_dump
from fafirec.ratio_runner import evaluate, write_rows
from tools import ace2026_experiment as ace
from experiments.diagnostic_v4 import final_runs as fr
from experiments.ablation_v5 import experiment as ex, shuffled_backbone as sb
from experiments.diagnostic_v4.fusion_model import score_histories

PROTOCOL = 'fafirec_v5_shuffled_identity_evaluation_v2'
OUTPUT_NAME = 'shuffled_identity_native_v2'
VARIANTS = ('shuffled_backbone_only', 'shuffled_backbone_frozen_fusion')


def sources():
    return {**ex.source_identity(), **sb.source_identity(),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def locks(root, shuffle_root):
    ex.verify_lock(root / 'ablation_lock.json')
    path = shuffle_root / 'five_seed_selection_lock.json'
    value = ace.read(path)
    ace.require(value['protocol'] == sb.PROTOCOL + '_five_seed_selection_lock'
        and value['seed_ids'] == ex.SEEDS and value['source_hashes'] == sb.source_identity()
        and value['lock_fingerprint'] == ex.fingerprint(value, 'lock_fingerprint')
        and sorted(j['seed'] for j in value['jobs']) == ex.SEEDS, 'Complete shuffled validation lock required')
    ex.verify_hashes(value['artifact_sha256'])
    return value


def worker(root, shuffle_root, seed, split, out):
    root, shuffle_root, out = Path(root).resolve(), Path(shuffle_root).resolve(), Path(out).resolve()
    locks(root, shuffle_root)
    ace.require(not out.exists() and seed in ex.SEEDS, 'Fresh known shuffle intervention job required')
    plan = ex.read_plan(root)
    original_run = Path(plan['baseline_root']) / f'seed_{seed}/run'
    if split == 'test':
        cache = root / f'seed_{seed}/test_cache'
        metadata = ace.read(cache / 'manifest.json'); ex.verify_hashes(metadata['artifact_sha256'])
        ace.require(metadata['ablation_lock_sha256'] == file_digest(root / 'ablation_lock.json'), 'Test cache lock differs')
        data = torch.load(cache / 'scores.pt', map_location='cpu', weights_only=True)
        records = ace.read(cache / 'records.json'); cfg = metadata['evaluation_config']
    else:
        manifest, _, _, training, data = fr.load_run(original_run)
        del training
        records = ace.read(original_run / 'cache/records_validation.json'); cfg = manifest['evaluation_config']
    model, validation_records, eval_cfg, sm, summary = sb.load_model(shuffle_root / f'seed_{seed}')
    ace.require(eval_cfg == cfg and [r['user'] for r in records] == [r['user'] for r in validation_records],
        'Shuffled intervention user/evaluation mapping differs')
    histories = [r['train'] + (r['validation'] if split == 'test' else []) for r in records]
    B = torch.load(original_run / 'cache/relations.pt', map_location='cpu', weights_only=True)['B']
    sem, _, _, available, _ = score_histories(model, B, histories, [0] * len(records), probe_window=0)
    ace.require(torch.equal(available, data['available']), 'Shuffled candidate policy changed')
    data = {**data, 'sem': sem}
    gate, _ = ex.load_gate(ex.model_path(root, seed, 'full'), 'full', plan, seed)
    out.mkdir(parents=True)
    for name in VARIANTS:
        if name == VARIANTS[0]:
            # Use exactly the native baseline scorer. Float32 normalization
            # can collapse near-ties relative to the original raw logits.
            rows, metrics = evaluate(model, records, split, torch.zeros(len(records), 2), cfg, seed)
            if split == 'validation':
                _same(metrics, summary['validation'], 'Native shuffled backbone validation')
                with (shuffle_root / f'seed_{seed}/validation.csv').open(encoding='utf-8', newline='') as stream:
                    original = list(csv.DictReader(stream))
                ace.require(len(rows) == len(original) and all(str(a['user']) == b['user']
                    and a['top_items'] == b['top_items'] for a, b in zip(rows, original)),
                    'Native shuffled validation rankings differ')
            for row in rows:
                row.update(control='fafirec_v4_diagnostic', variant=name, stage='fusion')
            (out / name).mkdir()
            write_rows(out / name / 'per_user.csv', rows)
            checked = ex.audit_rows(out / name / 'per_user.csv', records, cfg, seed, name,
                'fusion', 'fafirec_v4_diagnostic', split, sm['identity']['items'])
            _same(checked, metrics, 'Native shuffled backbone ranking audit')
        else:
            scores = fr.fusion.predict(gate, data, .025, 0.)
            metrics = fr.write_scored(out / name, scores, records, cfg, seed, name, split, 'per_user.csv')
        json_dump(out / name / 'metrics.json', {'protocol': PROTOCOL, 'seed': seed, 'variant': name,
            'split': split, 'metrics': metrics, 'candidate_policy': cfg['candidate_policy'],
            'per_user_sha256': file_digest(out / name / 'per_user.csv'), 'source_hashes': sources(),
            'ablation_lock_sha256': file_digest(root / 'ablation_lock.json'),
            'shuffle_lock_sha256': file_digest(shuffle_root / 'five_seed_selection_lock.json'),
            'shuffled_checkpoint_sha256': summary['checkpoint_sha256'],
            'original_gate_sha256': file_digest(ex.model_path(root, seed, 'full')),
            'score_policy': 'native_raw_logits' if name == VARIANTS[0] else 'original_normalized_fusion',
            'native_validation_rankings_identical': split == 'validation' and name == VARIANTS[0],
            'partial_ablation': True, 'diagnostic_proxy_and_other_branches_unchanged': True})


def verify(folder, root, shuffle_root, seed, split):
    plan = ex.read_plan(root)
    old_run = Path(plan['baseline_root']) / f'seed_{seed}/run'
    if split == 'test':
        records = ace.read(root / f'seed_{seed}/test_cache/records.json')
        cfg = ace.read(root / f'seed_{seed}/test_cache/manifest.json')['evaluation_config']
    else:
        records = ace.read(old_run / 'cache/records_validation.json')
        cfg = ace.read(old_run / 'manifest.json')['evaluation_config']
    n_items = ace.read(old_run / 'manifest.json')['identity']['items']
    values = {}
    for name in VARIANTS:
        base = folder / name; value = ace.read(base / 'metrics.json')
        ace.require(value['protocol'] == PROTOCOL and value['seed'] == seed and value['variant'] == name
            and value['split'] == split and value['source_hashes'] == sources()
            and value['per_user_sha256'] == file_digest(base / 'per_user.csv')
            and value['ablation_lock_sha256'] == file_digest(root / 'ablation_lock.json')
            and value['shuffle_lock_sha256'] == file_digest(shuffle_root / 'five_seed_selection_lock.json'),
            'Shuffled evaluation artifact differs')
        actual = ex.audit_rows(base / 'per_user.csv', records, cfg, seed, name, 'fusion', 'fafirec_v4_diagnostic', split, n_items)
        _same(actual, value['metrics'], 'Independent shuffled intervention metrics')
        values[name] = value
    return values


def ensure(out, root, shuffle_root, seed, split, label):
    folder = ex.contained(out, out / label / f'seed_{seed}')
    if not folder.exists():
        pending = out / '.pending'; pending.mkdir(exist_ok=True)
        stage = ex.contained(out, pending / (label + '_' + str(seed) + '_' + uuid.uuid4().hex))
        logs = out / 'logs'; logs.mkdir(exist_ok=True)
        with (logs / (label + '_' + str(seed) + '_' + uuid.uuid4().hex + '.log')).open('xb') as stream:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root),
                '--shuffle-root', str(shuffle_root), '--worker', '--seed', str(seed), '--split', split,
                '--output-dir', str(stage)], cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ace.require(result.returncode == 0, 'Shuffle intervention worker failed; inspect ' + str(logs))
        verify(stage, root, shuffle_root, seed, split)
        folder.parent.mkdir(parents=True, exist_ok=True)
        ace.require(not folder.exists(), 'Shuffle evaluation destination appeared')
        ex.contained(out, stage).rename(folder)
    return folder, verify(folder, root, shuffle_root, seed, split)


def run(root, shuffle_root):
    root, shuffle_root = Path(root).resolve(), Path(shuffle_root).resolve()
    sl = locks(root, shuffle_root)
    out = root / OUTPUT_NAME; out.mkdir(exist_ok=True)
    plan = {'protocol': PROTOCOL, 'seed_ids': ex.SEEDS, 'variants': list(VARIANTS), 'source_hashes': sources(),
        'ablation_lock_sha256': file_digest(root / 'ablation_lock.json'),
        'shuffle_lock_sha256': file_digest(shuffle_root / 'five_seed_selection_lock.json'),
        'score_policies': {VARIANTS[0]: 'native_raw_logits', VARIANTS[1]: 'original_normalized_fusion'},
        'partial_ablation': True, 'other_components_frozen': True, 'no_hyperparameter_selection': True}
    plan['fingerprint'] = digest(plan); ex.immutable(out / 'plan.json', plan)
    values = {'validation': {}, 'test': {}}
    hashes = {str(out / 'plan.json'): file_digest(out / 'plan.json'), **sl['artifact_sha256']}
    for seed in ex.SEEDS:
        for split in ('validation', 'test'):
            folder, first = ensure(out, root, shuffle_root, seed, split, split)
            if split == 'test':
                second_folder, second = ensure(out, root, shuffle_root, seed, split, 'test_reloaded')
                ace.require(first == second, 'Independent shuffled intervention metrics differ')
            for name in VARIANTS:
                values[split][seed, name] = first[name]['metrics']
                for filename in ('per_user.csv', 'metrics.json'):
                    file = folder / name / filename; hashes[str(file)] = file_digest(file)
                    if split == 'test':
                        other = second_folder / name / filename
                        ace.require(file_digest(other) == hashes[str(file)], 'Independent shuffled CSV/metrics bytes differ')
                        hashes[str(other)] = file_digest(other)
        print(f'Shuffled intervention: verified seed {seed}', flush=True)
    for split in values:
        rows = []
        for name in VARIANTS:
            for group in ex.GROUPS:
                for metric in ex.METRICS:
                    scores = [values[split][seed, name][group][metric] for seed in ex.SEEDS]
                    rows.append({'variant': name, 'stage': 'fusion', 'group': group, 'metric': metric,
                        'seed_ids': ex.SEEDS, 'seed_count': 5, 'per_seed_scores': scores,
                        'mean': statistics.mean(scores), 'sample_std': statistics.stdev(scores)})
        ex.verify_hashes(hashes)
        ex.immutable(out / f'combined_{split}.json', {'protocol': PROTOCOL, 'split': split, 'seed_ids': ex.SEEDS,
            'aggregates': rows, 'source_hashes': sources(), 'partial_ablation': True,
            'artifact_sha256': hashes, 'all_artifact_hashes_verified': True, 'no_model_selection': True})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/ablation_v5'))
    p.add_argument('--shuffle-root', type=Path, default=Path('runs/ablation_v5_shuffled_backbone'))
    p.add_argument('--worker', action='store_true'); p.add_argument('--seed', type=int)
    p.add_argument('--split', choices=('validation', 'test')); p.add_argument('--output-dir', type=Path)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.worker: worker(args.root, args.shuffle_root, args.seed, args.split, args.output_dir)
    else: run(args.root, args.shuffle_root)


if __name__ == '__main__': main()
