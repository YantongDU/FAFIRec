"""Final-recipe isolation, complete lock coverage and exact gate-state reload."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from analyze_research import audit_rows
from analyze_transfer import _same
from experiments.diagnostic_v4 import final_runs as final
from experiments.diagnostic_v4.fusion_model import DiagnosticGate
from fafirec.ratio_runner import write_rows
from fafirec.runner import cpu_state, digest, file_digest, json_dump


class FinalRunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = {'unit_source.py': 'frozen-source'}
        self.recipe = dict(name='global_control', mode='global', l2=None,
            alpha=.025, beta=.2, alignment_weight=0., pilot_variant='known_validation_choice')

    def write_plan(self, value):
        value = copy.deepcopy(value)
        value['fingerprint'] = final.fingerprint(value)
        path = self.root / 'recipe_plan.json'
        json_dump(path, value)
        return path

    def test_recipe_plan_uses_frozen_validation_selection_and_checks_semantic_fields(self):
        witness = self.root / 'validation_selection.json'
        witness.write_text('only validation', encoding='utf-8')
        base = dict(protocol=final.PLAN_PROTOCOL, seed_ids=final.SEEDS, recipes=[self.recipe],
            selection_split='validation', new_test_used=False, source_hashes=self.source,
            artifact_sha256={str(witness): file_digest(witness)})
        with patch.object(final.fusion, 'source_identity', return_value=self.source):
            self.assertEqual(final.verify_recipe_plan(self.write_plan(base))['recipes'], [self.recipe])
            invalids = []
            bad = copy.deepcopy(base); bad['recipes'] *= 2; invalids.append(bad)
            bad = copy.deepcopy(base); bad['recipes'][0]['name'] = '../escape'; invalids.append(bad)
            bad = copy.deepcopy(base); bad['recipes'][0]['alpha'] = -1; invalids.append(bad)
            bad = copy.deepcopy(base); bad['recipes'][0]['mode'] = 'unknown'; invalids.append(bad)
            bad = copy.deepcopy(base); bad['new_test_used'] = True; invalids.append(bad)
            bad = copy.deepcopy(base); bad['source_hashes'] = {}; invalids.append(bad)
            for bad in invalids:
                with self.subTest(value=bad), self.assertRaises(ValueError):
                    final.verify_recipe_plan(self.write_plan(bad))
            path = self.write_plan(base)
            witness.write_text('changed after selection', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                final.verify_recipe_plan(path)

    def make_lock(self):
        jobs, hashes = [], {}
        witness = self.root / 'trained_model.pt'
        witness.write_bytes(b'immutable weights')
        hashes[str(witness)] = file_digest(witness)
        for seed in final.SEEDS:
            folder = self.root / f'seed_{seed}/run'
            folder.mkdir(parents=True)
            manifest = dict(seed=seed, source_hashes=self.source, recipe_plan_sha256='recipe-plan-sha',
                recipes=[self.recipe])
            manifest['fingerprint'] = final.fingerprint(manifest)
            json_dump(folder / 'manifest.json', manifest)
            json_dump(folder / 'summary.json', dict(fingerprint=manifest['fingerprint'],
                artifact_sha256={str(witness): file_digest(witness)}))
            jobs.append(dict(seed=seed, run_dir=str(folder)))
            for name in ('manifest.json', 'summary.json'):
                hashes[str(folder / name)] = file_digest(folder / name)
        return dict(protocol=final.LOCK_PROTOCOL, seed_ids=final.SEEDS, jobs=jobs,
            source_hashes=self.source, recipe_plan_sha256='recipe-plan-sha', selected_recipe='global_control',
            artifact_sha256=hashes), witness

    def write_lock(self, value):
        value = copy.deepcopy(value)
        value['lock_fingerprint'] = digest({k: v for k, v in value.items() if k != 'lock_fingerprint'})
        path = self.root / 'selection_lock.json'
        json_dump(path, value)
        return path

    def test_test_lock_requires_all_five_jobs_membership_and_unchanged_artifacts(self):
        base, witness = self.make_lock()
        with patch.object(final, 'source_identity', return_value=self.source):
            path = self.write_lock(base)
            self.assertEqual(final.verify_lock(path)['seed_ids'], final.SEEDS)
            final.verify_lock(path, base['jobs'][0]['run_dir'], 2025, 'recipe-plan-sha')
            invalids = []
            bad = copy.deepcopy(base); bad['jobs'] = bad['jobs'][:-1]; invalids.append(bad)
            bad = copy.deepcopy(base); bad['jobs'][1]['seed'] = 2025; invalids.append(bad)
            bad = copy.deepcopy(base); bad['artifact_sha256'].pop(str(witness)); invalids.append(bad)
            bad = copy.deepcopy(base); bad['selected_recipe'] = 'untrained'; invalids.append(bad)
            for bad in invalids:
                with self.subTest(value=bad), self.assertRaises(ValueError):
                    final.verify_lock(self.write_lock(bad))
            path = self.write_lock(base)
            with self.assertRaisesRegex(ValueError, 'absent'):
                final.verify_lock(path, self.root / 'another_run', 2025, 'recipe-plan-sha')
            with self.assertRaisesRegex(ValueError, 'identity'):
                final.verify_lock(path, recipe_plan_sha256='changed-plan')
            witness.write_bytes(b'changed weights')
            with self.assertRaises(ValueError):
                final.verify_lock(path)

    def test_multi_positive_writer_preserves_split_history_and_independent_metric_audit(self):
        records = [dict(user='a', group='cold', train=[1, 2], validation=[3, 4], test=[5, 6]),
            dict(user='b', group='warm', train=[2, 3], validation=[4, 5], test=[6, 7])]
        scores = np.array([[100., 99., 98., 8., 7., 6., 5., 4., 3.],
                           [100., 3., 99., 98., 8., 7., 6., 5., 4.]])
        cfg = dict(ks=[2, 5], candidate_policy='exclude_seen')
        for split in ('validation', 'test'):
            rows, metrics = final.make_rows(scores, records, cfg, 2025, 'global_control', split)
            path = self.root / f'{split}.csv'
            write_rows(path, rows)
            checked = audit_rows(path, records, cfg, 2025, 'global_control', 'fusion',
                'fafirec_v4_diagnostic', split, 8)
            _same(checked, metrics, 'Fixture split ranking audit')
            for row, record in zip(rows, records):
                observed = set(record['train']) | (set(record['validation']) if split == 'test' else set())
                self.assertEqual(set(json.loads(row['targets'])), set(record[split]))
                self.assertFalse(set(json.loads(row['top_items'])) & observed)
                self.assertEqual(row['training_interactions'], 2)

    def test_gate_reload_preserves_learned_parameters_and_both_fixed_normalizers(self):
        features = torch.randn(5, 7, generator=torch.Generator().manual_seed(9))
        counts = torch.arange(12, dtype=torch.float32)
        recipe = dict(self.recipe, mode='candidate', l2=.001)
        model = DiagnosticGate(mode='candidate').fit_feature_statistics(features).fit_item_statistics(counts)
        with torch.no_grad():
            model.user_linear.weight.fill_(.15)
            model.pop_slope.bias.fill_(-.07)
            model.conflict_coef.fill_(.03)
        saved = dict(protocol=final.PROTOCOL, recipe=recipe, gate_model=cpu_state(model))
        loaded = final.load_gate(saved, recipe)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        sem, cf = torch.randn(5, 12), torch.randn(5, 12)
        outer = counts * 1.4
        self.assertTrue(torch.equal(model(sem, cf, features, outer), loaded(sem, cf, features, outer)))
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        with self.assertRaises(ValueError):
            final.load_gate(saved, dict(recipe, alpha=.1))

    def test_test_path_rejects_before_loading_scorer_without_complete_lock(self):
        with patch.object(final, 'load_run', side_effect=AssertionError('should not load')):
            with self.assertRaisesRegex(ValueError, 'five-seed validation lock'):
                final.replay(self.root / 'not_loaded', 'test', self.root / 'new_output')


if __name__ == '__main__':
    unittest.main()
