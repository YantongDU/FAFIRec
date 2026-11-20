"""Audit validation-only ACE tuning runs and freeze the next experiment config."""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ace2026_experiment as ace
from analyze_research import audit_rows
from analyze_transfer import _same
from fafirec.runner import file_digest, json_dump


def absolute(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def audit_run(path):
    root = absolute(path)
    manifest = ace.read(root / 'manifest.json')
    cfg = ace.normalize_config(manifest['config'])
    ace.require(cfg['seeds'] == [2025], 'Tuning uses only seed2025')
    ace.require(cfg['output_dir'] == str(root), 'Tuning output identity differs')
    ace.require(manifest.get('test_evaluated_by_training_entrypoint') is False,
                'Tuning entrypoint must not evaluate test')
    ace.check_grid(root, manifest)
    hashes = {str(root / name): file_digest(root / name)
              for name in ('manifest.json', 'selection.json', 'summary.json')}
    candidates = []
    for variant in cfg['variants']:
        name = variant['name']
        model, records, eval_cfg, checked, summary = ace.load_selected(root, 2025, name)
        ace.require(checked == manifest, 'Tuning manifest changed during audit')
        folder = root / f'seed_2025/{name}'
        metrics = audit_rows(folder / 'validation.csv', records, eval_cfg, 2025,
                             name, 'single', model.control, 'validation',
                             manifest['identity']['items'])
        _same(metrics, summary['validation'], 'Recomputed tuning validation metrics')
        for filename in ('best.pt', 'last.pt', 'summary.json', 'validation.csv',
                         'training.json', 'initial_validation.json'):
            hashes[str(folder / filename)] = file_digest(folder / filename)
        candidates.append({'run_dir': str(root), 'seed': 2025, 'variant': name,
                           'parameters': variant, 'selected_epoch': summary['selected_epoch'],
                           'epochs_completed': summary['epochs_completed'],
                           'candidate_validation_ndcg50': metrics['all']['ndcg50'],
                           'validation': metrics})
        del model
    return cfg, manifest, candidates, hashes


def winner(candidates):
    return sorted(candidates, key=lambda row: (-row['candidate_validation_ndcg50'],
                                             row['variant']))[0]


def check_pilot(cfg):
    actual = {(v['reg_weight'], v['scale'], v['lr']) for v in cfg['variants']}
    expected = {(reg, scale, .0005) for reg in (0., 50., 500.)
                for scale in (.1, .5, 1.)}
    ace.require(len(cfg['variants']) == 9 and actual == expected,
                'Complete predeclared nine-case pilot required')


def verify_hashes(record):
    ace.require(record.get('selection_split') == 'validation'
                and record.get('no_new_test_used') is True
                and record.get('source_hashes') == ace.source_identity()
                and record.get('driver_sha256') == file_digest(__file__),
                'Prior tuning record/source differs')
    for filename, checksum in record['artifact_sha256'].items():
        ace.require(file_digest(filename) == checksum, 'Prior tuning artifact changed: ' + filename)
    ace.require(file_digest(record['config_file']) == record['config_sha256'],
                'Prior selected refinement config changed')


def write_outputs(args, cfg, record):
    config_path, record_path = absolute(args.output_config), absolute(args.output_record)
    ace.require(config_path != record_path and not config_path.exists() and not record_path.exists(),
                'Tuning configs and records are immutable; choose fresh output paths')
    output = Path(cfg['output_dir'])
    ace.require(not output.exists() or not any(output.iterdir()),
                'Selected output directory already contains artifacts')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(config_path, cfg)
    record.update(config_file=str(config_path), config_sha256=file_digest(config_path))
    if args.command == 'final':
        record.update(final_config_file=str(config_path),
                      final_config_sha256=file_digest(config_path),
                      selected_variant=cfg['variants'][0])
    json_dump(record_path, record)
    print(f"Frozen {args.command} configuration: {config_path}", flush=True)
    print(f"Validation choice: {record['chosen_candidate']['variant']} "
          f"NDCG@50={record['chosen_candidate']['candidate_validation_ndcg50']:.12f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('refine', 'final'):
        command = sub.add_parser(name)
        command.add_argument('--pilot', required=True, type=Path)
        command.add_argument('--output-config', required=True, type=Path)
        command.add_argument('--output-record', required=True, type=Path)
        command.add_argument('--output-run-dir', type=Path,
                             default=Path('runs/ace2026_lr_validation' if name == 'refine'
                                          else 'runs/ace2026_final_replicas'))
        if name == 'final':
            command.add_argument('--refinement', required=True, type=Path)
            command.add_argument('--pilot-record', type=Path,
                                 default=Path('runs/ace2026_tuning/pilot_selection.json'))
    args = parser.parse_args()
    # Reject occupied outputs before any expensive checkpoint/data audit.
    ace.require(not absolute(args.output_config).exists() and not absolute(args.output_record).exists(),
                'Choose fresh immutable config and record paths')
    prior = None
    if args.command == 'final':
        prior_path = absolute(args.pilot_record)
        prior = ace.read(prior_path)
        verify_hashes(prior)
        ace.require(prior.get('schema') == 'ace2026_validation_tuning_refine_v1'
                    and prior['pilot_run'] == str(absolute(args.pilot)),
                    'Refinement record belongs to another pilot')
    cfg, manifest, pilot, hashes = audit_run(args.pilot)
    check_pilot(cfg)
    chosen = winner(pilot)
    output_cfg = copy.deepcopy(cfg)
    output_cfg['output_dir'] = str(absolute(args.output_run_dir))
    record = {'schema': f'ace2026_validation_tuning_{args.command}_v1',
              'selection_split': 'validation', 'selection_metric': 'all.ndcg50',
              'selection_seed': 2025, 'no_new_test_used': True,
              'historical_test_known': True, 'pilot_run': str(absolute(args.pilot)),
              'source_hashes': ace.source_identity(), 'driver_sha256': file_digest(__file__),
              'tie_break': 'descending validation NDCG@50, then ascending variant name',
              'pretest_statement': 'Only completed validation rankings select hyperparameters; '
                  'no new ACE test evaluation is performed by this tool.',
              'pilot_candidates': pilot}
    if args.command == 'refine':
        output_cfg['variants'] = [dict(chosen['parameters'], name=name, lr=lr)
                                  for name, lr in (('ace_lr1e3', .001), ('ace_lr1e4', .0001))]
        record['eligible_candidates'] = pilot
    else:
        _same(chosen, prior['chosen_candidate'], 'Prior pilot choice')
        refinement_cfg, refinement_manifest, refinement, refinement_hashes = audit_run(args.refinement)
        ace.require(refinement_cfg == ace.load_config(prior['config_file']),
                    'Refinement config differs from the frozen pilot choice')
        common_keys = ('identity', 'evaluation_config', 'source_hashes', 'runtime', 'svd')
        ace.require(all(manifest[k] == refinement_manifest[k] for k in common_keys),
                    'Tuning inputs, runtime or evaluation differ')
        expected_variants = [dict(chosen['parameters'], name=name, lr=lr)
                             for name, lr in (('ace_lr1e3', .001), ('ace_lr1e4', .0001))]
        ace.require(refinement_cfg['variants'] == expected_variants,
                    'Refinement must fix pilot geometry and compare only the two planned rates')
        ace.require({k: v for k, v in cfg.items() if k not in ('output_dir', 'variants')}
                    == {k: v for k, v in refinement_cfg.items() if k not in ('output_dir', 'variants')},
                    'Refinement changed the common training budget')
        eligible = [chosen, *refinement]
        chosen = winner(eligible)
        output_cfg['seeds'] = [2025, 2026, 2027, 2028, 2029]
        output_cfg['variants'] = [dict(chosen['parameters'], name='ace_selected')]
        hashes.update(refinement_hashes)
        hashes.update(prior['artifact_sha256'])
        hashes[str(prior_path)] = file_digest(prior_path)
        hashes[prior['config_file']] = file_digest(prior['config_file'])
        record.update(refinement_run=str(absolute(args.refinement)),
                      refinement_candidates=refinement, eligible_candidates=eligible,
                      pilot_record_file=str(prior_path), pilot_record_sha256=file_digest(prior_path))
    record.update(chosen_candidate=chosen, artifact_sha256=hashes)
    write_outputs(args, ace.normalize_config(output_cfg), record)


if __name__ == '__main__':
    main()
