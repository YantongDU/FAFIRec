"""Matched all-window backbone training with shuffled item/text identity.

Only the initial non-PAD item rows are permuted. Encoder initialization,
positions, global training RNG and the complete training recipe are preserved.
This is a partial text-identity ablation: the separate V4 diagnostic proxy is
unchanged. This entry point never scores the actual test split.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from analyze_research import audit_rows
from analyze_transfer import _same
from experiments.diagnostic_v4 import coverage_probe as coverage
from fafirec.ratio_runner import evaluate, write_rows
from fafirec.runner import digest, file_digest, json_dump, resolve_device, seed_everything, tensor_digest, torch_dump
from tools import ace2026_experiment as ace

PROTOCOL = 'fafirec_v5_shuffled_backbone_identity_v1'
VARIANT = 'shuffled_text_identity'
SEEDS = [2025, 2026, 2027, 2028, 2029]
BASELINE_ROOT = Path('E:/论文/FAFIRec_Diagnostic_v4/runs/diagnostic_v4_final_components')
ORIGINAL_CONFIG = Path('E:/论文/FAFIRec_ACE2026/configs/ace2026_selected.json')


def source_identity():
    return {**coverage.source_identity(), Path(__file__).relative_to(ROOT).as_posix(): file_digest(__file__)}


def immutable(path, value):
    path = Path(path)
    if path.exists():
        ace.require(ace.read(path) == value, 'Immutable shuffled artifact changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_dump(path, value)


def state_digest(model, exclude_item=False):
    return digest({k: tensor_digest(v) for k, v in model.state_dict().items()
                   if not (exclude_item and k == 'item_embedding.weight')})


@torch.no_grad()
def shuffle_initial_items(model, seed):
    """Return the PAD-inclusive permutation and independently checked metadata."""
    ace.require(type(seed) is int and seed >= 0 and hasattr(model, 'item_embedding'), 'Valid model and seed required')
    original = model.item_embedding.weight.detach().cpu().clone()
    ace.require(original.ndim == 2 and len(original) >= 3 and original[0].eq(0).all(), 'Zero PAD and real item rows required')
    before = state_digest(model, exclude_item=True)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    permutation = torch.arange(len(original), dtype=torch.long)
    permutation[1:] = torch.randperm(len(original) - 1, generator=torch.Generator(device='cpu').manual_seed(seed + 50000)) + 1
    shuffled = original[permutation]
    model.item_embedding.weight.copy_(shuffled.to(model.item_embedding.weight.device))
    ace.require(state_digest(model, exclude_item=True) == before, 'Row permutation changed encoder/position parameters')
    ace.require(torch.equal(cpu_rng, torch.get_rng_state()) and (not cuda_rng or
        all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))),
        'Local permutation advanced global training RNG')
    inverse = torch.argsort(permutation)
    ace.require(torch.equal(shuffled[inverse], original) and shuffled[0].eq(0).all(), 'Permutation lost item values or PAD')
    first, second = original[1:].double(), shuffled[1:].double()
    spectrum = torch.linalg.svdvals(first)
    shuffled_spectrum = torch.linalg.svdvals(second)
    spectrum_error = float((spectrum - shuffled_spectrum).abs().max())
    ace.require(torch.allclose(spectrum, shuffled_spectrum, rtol=1e-11, atol=1e-11), 'Row permutation changed the singular spectrum')
    report = {'kind': 'initial_nonpadding_item_identity_permutation', 'permutation_seed': seed + 50000,
        'permutation_sha256': tensor_digest(permutation), 'native_initial_table_sha256': tensor_digest(original),
        'shuffled_initial_table_sha256': tensor_digest(shuffled), 'initial_nonitem_state_sha256': before,
        'nonitem_parameters_unchanged': True, 'global_training_rng_unchanged': True,
        'inverse_recovers_native_table_exactly': True, 'pad_zero': True,
        'native_nonpad_mean': float(first.mean()), 'shuffled_nonpad_mean': float(second.mean()),
        'native_nonpad_population_std': float(first.std(unbiased=False)),
        'shuffled_nonpad_population_std': float(second.std(unbiased=False)),
        'singular_spectrum_max_abs_difference': spectrum_error,
        'scope': 'Only backbone initial text-to-item row identity; original diagnostic proxies remain unchanged.'}
    return permutation, report


def make_model(features, factors, cfg, seed):
    seed_everything(seed + 1000)
    model = ace.new_model(features, factors, cfg, cfg['variants'][0], resolve_device(cfg))
    permutation, report = shuffle_initial_items(model, seed)
    return model, permutation, report


def baseline_proof(seed, cfg):
    folder = BASELINE_ROOT / f'seed_{seed}/coverage'
    manifest = ace.read(folder / 'manifest.json')
    ace.require(manifest['protocol'] == coverage.PROTOCOL and manifest['modes'] == ['all_windows']
        and manifest['config']['seeds'] == [seed] and manifest['fingerprint'] == ace.fingerprint(manifest)
        and manifest['source_hashes'] == coverage.source_identity(),
        'Original matched coverage baseline changed')
    ignored = {'output_dir', 'seeds'}
    ace.require({k: v for k, v in cfg.items() if k not in ignored}
        == {k: v for k, v in manifest['config'].items() if k not in ignored}, 'Training recipe differs from matched baseline')
    paths = [folder / 'manifest.json', folder / 'all_windows/summary.json', folder / 'all_windows/best.pt',
        folder / 'all_windows/validation.csv']
    return manifest, {str(p): file_digest(p) for p in paths}


def prepare(seed_root, cfg, seed):
    torch.set_num_threads(cfg['threads'])
    records, features, identity, base, provenance, factors, fi = ace.inputs(cfg)
    # No test label is retained by training or validation replay.
    records = [{k: v for k, v in r.items() if k != 'test'} for r in records]
    baseline, proof = baseline_proof(seed, cfg)
    ace.require(identity == baseline['identity'], 'Shuffled input mapping differs from native baseline')
    model, permutation, initialization = make_model(features, factors, cfg, seed)
    manifest = ace.manifest_for(cfg, identity, base, provenance, fi, resolve_device(cfg))
    manifest.update(protocol=coverage.PROTOCOL, experiment_protocol=PROTOCOL, modes=['all_windows'],
        seed=seed, source_hashes=source_identity(), baseline_artifact_sha256=proof,
        ablation_initialization=initialization, train_validation_records_sha256=digest(records),
        training_layout='all_windows; every original training target once per epoch, max50 positions; same as native control',
        partial_ablation=True, diagnostic_proxies_unchanged=True, test_evaluated=False)
    manifest['fingerprint'] = ace.fingerprint(manifest)
    return model, records, permutation, manifest


def run_one(seed_root, cfg, seed, resume=False):
    root = Path(seed_root).resolve()
    model, records, permutation, fresh = prepare(root, cfg, seed)
    if root.exists() and any(root.iterdir()):
        ace.require(resume, 'Existing shuffled run requires --resume')
        manifest = ace.read(root / 'manifest.json')
        ace.require(manifest['fingerprint'] == ace.fingerprint(manifest) == fresh['fingerprint'], 'Cannot resume changed shuffled experiment')
        saved = torch.load(root / 'permutation.pt', map_location='cpu', weights_only=True)
        ace.require(torch.equal(saved['permutation'], permutation) and saved['report'] == fresh['ablation_initialization'], 'Saved permutation changed')
        if (root / 'summary.json').exists():
            _, _, _, _, summary = load_model(root)
            return summary
    else:
        root.mkdir(parents=True, exist_ok=True)
        manifest = fresh
        json_dump(root / 'manifest.json', manifest)
        torch_dump(root / 'permutation.pt', {'permutation': permutation, 'report': manifest['ablation_initialization']})
    selected = coverage.fit(model, records, cfg, manifest['evaluation_config'], seed, 'all_windows', root, manifest, resume)
    rows, metrics = evaluate(model, records, 'validation', torch.zeros(len(records), 2), manifest['evaluation_config'], seed)
    for row in rows:
        row.update(variant=VARIANT, stage='ablation')
    write_rows(root / 'validation.csv', rows)
    checked = audit_rows(root / 'validation.csv', records, manifest['evaluation_config'], seed, VARIANT,
        'ablation', model.control, 'validation', manifest['identity']['items'])
    _same(checked, metrics, 'Shuffled selected validation ranking audit')
    summary = {'experiment_protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'], 'seed': seed,
        'validation': metrics, **selected, 'checkpoint_sha256': file_digest(root / 'best.pt'),
        'per_user_sha256': file_digest(root / 'validation.csv'), 'split': 'validation', 'test_evaluated': False}
    json_dump(root / 'summary.json', summary)
    print(f"Completed shuffled seed {seed}: validation NDCG@50={metrics['all']['ndcg50']:.12f}", flush=True)
    return summary


def load_model(seed_root, device=None):
    """Return (frozen model, train/validation records, eval_cfg, manifest, summary).

    Actual test labels are deliberately omitted. A caller with a complete
    five-seed test lock may authenticate the original records through ace.inputs.
    """
    root = Path(seed_root).resolve()
    manifest = ace.read(root / 'manifest.json')
    ace.require(manifest['experiment_protocol'] == PROTOCOL and manifest['protocol'] == coverage.PROTOCOL
        and manifest['source_hashes'] == source_identity() and manifest['fingerprint'] == ace.fingerprint(manifest),
        'Shuffled source/manifest identity differs')
    cfg, seed = ace.normalize_config(manifest['config']), manifest['seed']
    ace.require(cfg['output_dir'] == str(root) and cfg['seeds'] == [seed], 'Shuffled output/seed differs')
    model, records, permutation, fresh = prepare(root, cfg, seed)
    ace.require(fresh['fingerprint'] == manifest['fingerprint'], 'Shuffled data/runtime/initialization changed')
    perm = torch.load(root / 'permutation.pt', map_location='cpu', weights_only=True)
    ace.require(torch.equal(perm['permutation'], permutation) and perm['report'] == manifest['ablation_initialization'], 'Shuffled permutation artifact changed')
    summary = ace.read(root / 'summary.json')
    best = torch.load(root / 'best.pt', map_location='cpu', weights_only=True)
    last = torch.load(root / 'last.pt', map_location='cpu', weights_only=True)
    ace.check_checkpoint(best, coverage.header(manifest, 'all_windows', seed, model))
    ace.check_checkpoint(last, coverage.header(manifest, 'all_windows', seed, model))
    ace.require(last['finished'] and best['selected_epoch'] == last['selected_epoch'] == summary['selected_epoch']
        and summary['epochs_completed'] == last['epoch'] and summary['fingerprint'] == manifest['fingerprint']
        and summary['checkpoint_sha256'] == file_digest(root / 'best.pt')
        and summary['per_user_sha256'] == file_digest(root / 'validation.csv'), 'Shuffled selected artifact differs')
    _same(best['validation'], summary['validation'], 'Shuffled checkpoint validation')
    ace.require(set(best['model']) == set(last['best_snapshot']['model'])
        and all(torch.equal(t, last['best_snapshot']['model'][k]) for k, t in best['model'].items()), 'Shuffled selected weights differ')
    ace.require(ace.read(root / 'training.json') == last['logs'], 'Shuffled training log changed')
    for path, checksum in manifest['baseline_artifact_sha256'].items():
        ace.require(file_digest(path) == checksum, 'Matched baseline source changed: ' + path)
    model.load_state_dict(best['model'])
    if device is not None:
        model.to(device)
    return model.requires_grad_(False).eval(), records, manifest['evaluation_config'], manifest, summary


def replay_validation(seed_root, output_dir):
    out = Path(output_dir).resolve()
    ace.require(not out.exists(), 'Validation replay output must be fresh')
    model, records, cfg, manifest, summary = load_model(seed_root)
    rows, metrics = evaluate(model, records, 'validation', torch.zeros(len(records), 2), cfg, manifest['seed'])
    for row in rows:
        row.update(variant=VARIANT, stage='ablation')
    out.mkdir(parents=True)
    write_rows(out / 'per_user.csv', rows)
    ace.require(file_digest(out / 'per_user.csv') == summary['per_user_sha256'], 'Fresh shuffled validation ranking differs')
    _same(metrics, summary['validation'], 'Fresh shuffled validation metrics')
    checked = audit_rows(out / 'per_user.csv', records, cfg, manifest['seed'], VARIANT, 'ablation',
        model.control, 'validation', manifest['identity']['items'])
    _same(checked, metrics, 'Shuffled replay independent metric recomputation')
    json_dump(out / 'metrics.json', {'experiment_protocol': PROTOCOL, 'fingerprint': manifest['fingerprint'],
        'seed': manifest['seed'], 'split': 'validation', 'identical_to_original': True,
        'metrics': metrics, 'per_user_sha256': file_digest(out / 'per_user.csv'),
        'checkpoint_sha256': summary['checkpoint_sha256']})


def run(root, resume=False):
    root = Path(root).resolve()
    cfg = ace.load_config(ORIGINAL_CONFIG)
    ace.require((cfg['d_model'], cfg['n_layers'], cfg['n_heads'], cfg['inner_size'], cfg['batch_size'],
        cfg['epochs'], cfg['min_epochs'], cfg['eval_interval'], cfg['patience']) == (128, 2, 2, 256, 256, 400, 30, 5, 3),
        'Frozen native architecture/budget required')
    jobs = []
    for seed in SEEDS:
        folder = root / f'seed_{seed}'
        jobs.append({'seed': seed, 'run_dir': str(folder),
            'config': ace.normalize_config({**cfg, 'seeds': [seed], 'output_dir': str(folder)})})
    plan = {'experiment_protocol': PROTOCOL, 'seed_ids': SEEDS, 'jobs': jobs,
        'source_hashes': source_identity(), 'original_config_sha256': file_digest(ORIGINAL_CONFIG),
        'selection_split': 'validation', 'test_evaluated': False, 'permutation_seed_rule': 'seed+50000',
        'partial_ablation': True, 'unchanged_diagnostic_proxy': True}
    plan['fingerprint'] = digest(plan)
    if root.exists() and any(root.iterdir()):
        ace.require(resume and ace.read(root / 'master_plan.json') == plan, 'Shuffled master plan changed')
    else:
        root.mkdir(parents=True, exist_ok=True)
    immutable(root / 'master_plan.json', plan)
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    artifacts, summaries = {str(root / 'master_plan.json'): file_digest(root / 'master_plan.json')}, []
    for job in jobs:
        folder = Path(job['run_dir'])
        config_path = root / f"config_{job['seed']}.json"
        immutable(config_path, job['config'])
        args = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--config', str(config_path)]
        if resume:
            args.append('--resume')
        with (logs / f"seed_{job['seed']}{'_resume' if resume else ''}.log").open('ab') as stream:
            completed = subprocess.run(args, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ace.require(completed.returncode == 0, 'Shuffled training failed; inspect ' + str(logs))
        replay = folder / 'independent_validation_replay'
        if not replay.exists():
            stage = folder / ('.validation_replay_' + uuid.uuid4().hex)
            args = [sys.executable, '-u', str(Path(__file__).resolve()), 'replay-validation',
                '--run-dir', str(folder), '--output-dir', str(stage)]
            with (logs / f"validation_{job['seed']}.log").open('ab') as stream:
                completed = subprocess.run(args, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            ace.require(completed.returncode == 0, 'Independent shuffled validation failed; inspect logs')
            ace.require(stage.resolve().is_relative_to(root) and replay.resolve().is_relative_to(root), 'Replay move escapes output root')
            stage.rename(replay)
        model, records, eval_cfg, manifest, summary = load_model(folder)
        saved = ace.read(replay / 'metrics.json')
        ace.require(saved['identical_to_original'] and saved['fingerprint'] == manifest['fingerprint']
            and saved['seed'] == job['seed'] and file_digest(replay / 'per_user.csv') == summary['per_user_sha256'] == saved['per_user_sha256'],
            'Stored shuffled validation replay changed')
        checked = audit_rows(replay / 'per_user.csv', records, eval_cfg, job['seed'], VARIANT,
            'ablation', model.control, 'validation', manifest['identity']['items'])
        _same(checked, summary['validation'], 'Stored shuffled validation metrics')
        summaries.append(summary)
        for name in ('manifest.json', 'permutation.pt', 'best.pt', 'last.pt', 'summary.json', 'training.json',
            'initial_validation.json', 'validation.csv', 'independent_validation_replay/per_user.csv', 'independent_validation_replay/metrics.json'):
            artifacts[str(folder / name)] = file_digest(folder / name)
        artifacts[str(config_path)] = file_digest(config_path)
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"Audited shuffled seed {job['seed']}; validation NDCG@50={summary['validation']['all']['ndcg50']:.12f}", flush=True)
        immutable(root / f"receipt_{job['seed']}.json", {'seed': job['seed'], 'manifest_fingerprint': manifest['fingerprint'],
            'selected_epoch': summary['selected_epoch'], 'checkpoint_sha256': summary['checkpoint_sha256'],
            'per_user_sha256': summary['per_user_sha256'], 'independent_validation_identical': True})
    lock = {'protocol': PROTOCOL + '_five_seed_selection_lock', 'seed_ids': SEEDS,
        'source_hashes': source_identity(), 'jobs': jobs, 'selection_split': 'validation',
        'test_evaluated': False, 'summaries': summaries, 'artifact_sha256': artifacts,
        'partial_ablation': True, 'diagnostic_proxies_unchanged': True}
    lock['lock_fingerprint'] = digest(lock)
    immutable(root / 'five_seed_selection_lock.json', lock)
    print('All five shuffled backbones trained, independently reloaded and locked; no test scoring performed.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('train'); q.add_argument('--output-dir', type=Path, default=Path('runs/ablation_v5_shuffled_backbone'))
    q.add_argument('--resume', action='store_true')
    q = sub.add_parser('worker'); q.add_argument('--config', type=Path, required=True); q.add_argument('--resume', action='store_true')
    q = sub.add_parser('replay-validation'); q.add_argument('--run-dir', type=Path, required=True); q.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'train': run(args.output_dir, args.resume)
    elif args.command == 'worker':
        cfg = ace.load_config(args.config)
        run_one(Path(cfg['output_dir']), cfg, cfg['seeds'][0], args.resume)
    else: replay_validation(args.run_dir, args.output_dir)


if __name__ == '__main__':
    main()
