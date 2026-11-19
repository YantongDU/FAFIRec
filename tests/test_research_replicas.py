"""Portable metadata-only CPU regression checks for audited seed replicas.

The synthetic reports stand in for already completed independent model audits.
These checks exercise combination and locking; they never construct a model.
"""
import importlib.util
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('combine_replicas', PROJECT / 'tools/combine_research_replicas.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def build(root):
    seeds = [101, 102, 103, 104, 105]
    cfg = c.research.normalize_config(dict(reference_main_run=str(root / 'oldmain'), anchor_run=str(root / 'oldanchor'),
        output_dir=str(root / 'replicas'), seeds=seeds, variants=[{'name': 'id', 'kind': 'id'},
            {'name': 'text', 'kind': 'text'}, {'name': 'shuffle', 'kind': 'shuffled'}],
        epochs=2, min_epochs=1, router_epochs=2, router_min_epochs=1, train_router=True, device='cpu'))
    master = root / 'master.json'
    dump(master, cfg)
    replica = Path(cfg['output_dir'])
    plan = {'config': cfg, 'master_config_sha256': c.file_digest(master), 'source_hashes': c.research.source_identity(),
        'parallel_workers': 2, 'protocol': 'isolated_replicas_identical_settings_except_seed_and_output', 'seeds': seeds}
    plan['fingerprint'] = c.digest(plan)
    dump(replica / 'master_plan.json', plan)
    helper = PROJECT / 'tools/compare_research.py'
    for index, seed in enumerate(seeds):
        folder = replica / f'seed_{seed}'
        local = {**cfg, 'seeds': [seed], 'output_dir': str(folder / 'run')}
        dump(folder / 'config.json', local)
        manifest = {'protocol': c.research.PROTOCOL, 'config': local,
            'evaluation_config': {'max_len': 6, 'ks': [10, 50], 'selection_k': 50,
                'candidate_policy': 'exclude_seen', 'device': 'cpu', 'threads': 4, 'eval_batch': 128},
            'identity': {'items': 12, 'users': 6}, 'source_hashes': c.research.source_identity(),
            'runtime': {'device': 'cpu', 'fixture': True},
            'reference': {'common': 'immutable reference fixture', 'anchors': {str(seed): {'hash': f'anchor{seed}'}}},
            'selection': 'same declared validation rule', 'test_evaluated_by_training_entrypoint': False}
        manifest['fingerprint'] = c.research._fingerprint(manifest)
        dump(folder / 'run/manifest.json', manifest)
        selections, shared_assets = [], {}
        for variant in ('id', 'text', 'shuffle'):
            for stage in ('expert', 'router'):
                job = folder / f'run/seed_{seed}/{variant}/{stage}'
                job.mkdir(parents=True)
                checkpoint, csv = job / 'best.pt', job / 'validation.csv'
                checkpoint.write_text(f'fixture checkpoint {seed} {variant} {stage}', encoding='utf-8')
                csv.write_text(f'fixture audited rankings {seed} {variant} {stage}', encoding='utf-8')
                summary = {'seed': seed, 'variant': variant, 'stage': stage, 'fingerprint': manifest['fingerprint'],
                    'selected_epoch': 2, 'epochs_completed': 2, 'optimizer_steps': 2,
                    'calibration': {'alpha': 1., 'route': 'global' if stage == 'expert' else 'relative'},
                    'checkpoint_sha256': c.file_digest(checkpoint), 'per_user_sha256': c.file_digest(csv)}
                dump(job / 'summary.json', summary)
                selections.append({k: summary[k] for k in ('seed', 'variant', 'stage', 'selected_epoch',
                    'epochs_completed', 'optimizer_steps', 'calibration', 'checkpoint_sha256')})
                shared_assets[str(checkpoint)] = c.file_digest(checkpoint)
                shared_assets[str(csv)] = c.file_digest(csv)
        for split in ('validation', 'test'):
            aggregates = []
            for variant, stage in [(v, s) for v in ('id', 'text', 'shuffle') for s in ('expert', 'router')] + [('strong_cf_anchor', 'frozen')]:
                if split == 'validation':
                    if variant == 'text':
                        score = (.5 if index == 0 else .25) if stage == 'expert' else (.1 if index == 0 else .34)
                    else:
                        score = {'id': .20, 'shuffle': .23, 'strong_cf_anchor': .19}[variant] + (.01 if stage == 'router' else 0)
                    score += .001 * index
                else:
                    score = .9 if variant == 'id' else .2 + .001 * index
                for group in ('all', 'cold', 'middle', 'warm'):
                    for metric in (f'{m}{k}' for k in (10, 50) for m in ('recall', 'ndcg', 'hit')):
                        aggregates.append({'variant': variant, 'stage': stage, 'group': group, 'metric': metric,
                            'seed_ids': [seed], 'per_seed_scores': [score], 'mean': score, 'sample_std': None})
            def score(v, s, g, m):
                return next(r['mean'] for r in aggregates if (r['variant'], r['stage'], r['group'], r['metric']) == (v, s, g, m))
            comparisons = []
            for group in ('all', 'cold', 'middle', 'warm'):
                for metric in ('recall10', 'ndcg10', 'recall50', 'ndcg50'):
                    left, right = score('text', 'expert', group, metric), score('id', 'expert', group, metric)
                    comparisons.append({'comparison': 'semantic-minus-strong-ID', 'candidate': ['text', 'expert'],
                        'baseline': ['id', 'expert'], 'group': group, 'metric': metric, 'attribution': 'fixture',
                        'seed_ids': [seed], 'paired_seeds': 1, 'candidate_scores': [left], 'baseline_scores': [right],
                        'per_seed_differences': [left - right], 'mean_difference': left - right,
                        'bootstrap_95_ci': None, 'two_sided_sign_flip_p': None})
            extra = folder / f'{split}_audit_fixture.txt'
            extra.write_text('previous independent audit fixture', encoding='utf-8')
            report = {'schema': 'research_v3_controlled_comparison_v1', 'split': split,
                'fingerprint': manifest['fingerprint'], 'seed_ids': [seed], 'job_count': 6,
                'cpu_checkpoint_identity_verified': True, 'model_evaluation_called': False,
                'complete_configured_grid_verified': True, 'all_ranking_metrics_independently_recomputed': True,
                'helper_file': str(helper), 'helper_sha256': c.file_digest(helper),
                'reference': manifest['reference'], 'variants': cfg['variants'],
                'selection_budget': {k: cfg[k] for k in ('epochs', 'min_epochs', 'patience', 'eval_interval', 'batch_size',
                    'alpha_grid', 'routes', 'train_router', 'router_epochs', 'router_min_epochs', 'router_lr')},
                'selections': selections, 'aggregates': aggregates, 'paired_comparisons': comparisons,
                'missing_matched_controls': [], 'artifact_hashes': {**shared_assets, str(extra): c.file_digest(extra)}}
            dump(folder / f'compare_{split}.json', report)
    return master, replica


class ResearchReplicaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.master, self.replicas = build(self.root)
        self.args = SimpleNamespace(master_config=self.master, replica_root=self.replicas,
            split='validation', output=None, selection_lock=None,
            bootstrap_samples=1000, bootstrap_seed=2025)

    def test_five_seed_global_selection_does_not_pick_local_stages(self):
        # Four seeds prefer the text router, but one large expert gain makes
        # text/expert the winner of the predeclared five-seed mean criterion.
        locally_preferred = []
        for seed in (101, 102, 103, 104, 105):
            report = c.read(self.replicas / f'seed_{seed}/compare_validation.json')
            candidates = [r for r in report['aggregates'] if r['variant'] == 'text'
                          and r['group'] == 'all' and r['metric'] == 'ndcg50']
            locally_preferred.append(max(candidates, key=lambda r: r['mean'])['stage'])
        self.assertEqual(locally_preferred.count('router'), 4)
        result = c.execute(self.args)
        self.assertEqual(result['decision']['selected_variant'], 'text')
        self.assertEqual(result['decision']['selected_stage'], 'expert')
        self.assertTrue(result['master_plan_verified'])
        self.assertEqual(result['signflip_minimum_two_sided_p'], .0625)
        self.assertTrue(all(r['stage'] == 'expert' for r in result['locked_selected_model_scores']))
        metric = next(r for r in result['aggregates'] if (r['variant'], r['stage'], r['group'], r['metric'])
                      == ('text', 'expert', 'all', 'ndcg50'))
        self.assertAlmostEqual(metric['mean'], .302)
        self.assertEqual(len(metric['per_seed_scores']), 5)
        self.assertGreater(metric['sample_std'], 0)
        lock = c.read(Path(result['selection_lock_file']))
        self.assertEqual(lock['decision'], result['decision'])
        self.assertFalse(lock['new_test_used_for_selection'])

    def test_test_requires_an_explicit_global_selection_lock(self):
        self.args.split = 'test'
        with self.assertRaisesRegex(ValueError, 'explicit five-seed'):
            c.execute(self.args)

    def test_higher_test_scores_do_not_change_common_validation_choice(self):
        validation = c.execute(self.args)
        self.args.split = 'test'
        self.args.selection_lock = Path(validation['selection_lock_file'])
        result = c.execute(self.args)
        self.assertEqual(result['decision'], validation['decision'])
        self.assertTrue(any(r['variant'] == 'id' and r['mean'] == .9 for r in result['aggregates']))
        self.assertTrue(all(r['variant'] == 'text' and r['stage'] == 'expert'
                            for r in result['locked_selected_model_scores']))
        for comparison in result['paired_comparisons']:
            self.assertIsNotNone(comparison['bootstrap_95_ci'])
            self.assertGreaterEqual(comparison['two_sided_sign_flip_p'], .0625)
            self.assertEqual(comparison['paired_seeds'], 5)

    def test_changed_audited_asset_hash_is_rejected(self):
        damaged = self.replicas / 'seed_103/validation_audit_fixture.txt'
        damaged.write_text('changed audited asset', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Artifact hash changed'):
            c.execute(self.args)

    def test_replica_configuration_drift_is_rejected(self):
        path = self.replicas / 'seed_102/config.json'
        config = c.read(path)
        config['epochs'] = 3
        dump(path, config)
        with self.assertRaisesRegex(ValueError, 'config file differs'):
            c.execute(self.args)

    def test_extra_seed_directory_is_rejected(self):
        (self.replicas / 'seed_999').mkdir()
        with self.assertRaisesRegex(ValueError, 'extra replica seed directories'):
            c.execute(self.args)


if __name__ == '__main__':
    unittest.main()


