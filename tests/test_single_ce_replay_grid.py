"""CPU regressions for roundoff-tolerant locking with exact file identities."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import replay_single_ce_grid as grid


class SingleCEReplayGridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seeds = [2025, 2026, 2027, 2028, 2029]
        self.master = {'seeds': self.seeds}
        self.master_path = self.root / 'master.json'
        grid.json_dump(self.master_path, self.master)
        self.replica_root = self.root / 'replicas'
        self.fields = ('manifest.json', 'best.pt', 'last.pt', 'summary.json',
                       'validation.csv', 'initial_validation.json')
        self.metrics = {group: {'users': 2, 'relevant_items': 3,
                               'recall10': .125, 'ndcg50': .25}
                        for group in ('all', 'cold', 'middle', 'warm')}
        self.artifacts = {}
        for seed in self.seeds:
            folder = self.replica_root / f'seed_{seed}'
            folder.mkdir(parents=True)
            for name in self.fields:
                path = folder / name
                path.write_bytes(f'CPU fixture {seed} {name}\n'.encode())
                self.artifacts[str(path)] = grid.file_digest(path)
        self.output = self.root / 'selection_lock.json'

    def load_run(self, folder, seed):
        self.assertEqual(folder, self.replica_root / f'seed_{seed}')
        manifest = {'runtime': {'device': 'cpu'}, 'identity': {'items': 12},
                    'master_config_file': str(self.master_path),
                    'evaluation_config': {'candidate_policy': 'exclude_seen'},
                    'fingerprint': f'fixture-seed-{seed}'}
        summary = {'selected_epoch': 10, 'validation': copy.deepcopy(self.metrics)}
        return [{'user': 'fixture'}], None, None, manifest, summary, None, None

    def test_full_five_seed_roundoff_passes_and_keeps_exact_artifact_hashes(self):
        def audit(*args):
            computed = copy.deepcopy(self.metrics)
            computed['all']['ndcg50'] += 1e-15
            return computed
        with patch.object(grid.baseline, '_master', return_value=(self.master_path, self.master)), \
             patch.object(grid.baseline, '_load_run', side_effect=self.load_run) as loading, \
             patch.object(grid, 'audit_rows', side_effect=audit) as auditing:
            grid.audited_lock(self.master_path, self.replica_root, self.output)
        self.assertEqual(loading.call_count, 5)
        self.assertEqual(auditing.call_count, 5)
        value = grid.baseline.read(self.output)
        self.assertEqual(value['seed_ids'], self.seeds)
        self.assertEqual(value['validation_artifact_sha256'], self.artifacts)
        self.assertEqual(len(value['validation_artifact_sha256']), 30)
        for path, checksum in value['validation_artifact_sha256'].items():
            self.assertEqual(grid.file_digest(path), checksum)
        self.assertEqual(value['master_config_sha256'], grid.file_digest(self.master_path))
        self.assertEqual(value['lock_driver_sha256'], grid.file_digest(grid.__file__))
        self.assertEqual(value['source_hashes'], grid.baseline.source_identity())
        self.assertEqual(value['lock_fingerprint'], grid.baseline.digest(
            {k: v for k, v in value.items() if k != 'lock_fingerprint'}))
        self.assertFalse(value['new_test_used_for_selection'])
        self.assertEqual(value['mean_validation_ndcg50'], .25)
        self.assertEqual([j['seed'] for j in value['jobs']], self.seeds)
        self.assertTrue(all(j['validation'] == self.metrics for j in value['jobs']))
        self.assertEqual(set(value['csv_recomputation_max_absolute_error']), set(map(str, self.seeds)))
        for error in value['csv_recomputation_max_absolute_error'].values():
            self.assertGreater(error, 0.)
            self.assertLess(error, 1e-12)

    def test_substantive_numeric_discrepancy_rejects_before_writing_lock(self):
        def audit(*args):
            computed = copy.deepcopy(self.metrics)
            if args[3] == 2027:
                computed['all']['ndcg50'] += 1e-3
            return computed
        with patch.object(grid.baseline, '_master', return_value=(self.master_path, self.master)), \
             patch.object(grid.baseline, '_load_run', side_effect=self.load_run), \
             patch.object(grid, 'audit_rows', side_effect=audit), \
             self.assertRaisesRegex(ValueError, 'numeric value differs'):
            grid.audited_lock(self.master_path, self.replica_root, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual({path: grid.file_digest(path) for path in self.artifacts}, self.artifacts)


if __name__ == '__main__':
    unittest.main()
