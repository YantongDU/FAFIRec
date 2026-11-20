"""CPU regressions for order-independent, otherwise complete ACE aggregates."""
import copy
import statistics
import tempfile
import unittest
from pathlib import Path

from fafirec.runner import file_digest
from tools import finalize_ace2026_aggregate as finalizer


class ACEAggregateKeyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.artifact = Path(temporary.name) / 'rankings.csv'
        self.artifact.write_bytes(b'user,metric\n1,0.25\n')
        self.seeds = [2025, 2026, 2027, 2028, 2029]
        self.lock = {'seed_ids': self.seeds,
                     'common': {'config': {'variants': [{'name': 'ace_selected'}]}}}
        rows = []
        for group_index, group in enumerate(finalizer.GROUPS):
            for metric_index, metric in enumerate(finalizer.METRICS):
                scores = [.1 + .01 * group_index + .001 * metric_index + .0001 * seed
                          for seed in range(len(self.seeds))]
                rows.append({'variant': 'ace_selected', 'stage': 'single',
                             'group': group, 'metric': metric, 'seed_ids': self.seeds,
                             'seed_count': len(self.seeds), 'per_seed_scores': scores,
                             'mean': statistics.mean(scores),
                             'sample_std': statistics.stdev(scores)})
        self.expected = {
            'protocol': 'ace2026_ml1m_shifted_catalog_v1', 'split': 'test',
            'seed_ids': self.seeds, 'aggregates': rows,
            'epoch_selections': [{'seed': seed, 'selected_epoch': 45,
                                  'epochs_completed': 60, 'optimizer_steps': 1440}
                                 for seed in self.seeds],
            'common': {'config': {'lr': .0005}, 'identity': {'items': 3883}},
            'artifact_sha256': {str(self.artifact): file_digest(self.artifact)},
            'all_artifact_hashes_verified': True,
        }

    def test_complete_metric_rows_allow_reordering_without_mutating_inputs(self):
        expected_snapshot = copy.deepcopy(self.expected)
        expected_keys = sorted(tuple(row[key] for key in finalizer.KEYS)
                               for row in self.expected['aggregates'])
        for ordered_rows in (list(reversed(self.expected['aggregates'])),
                             self.expected['aggregates'][7:] + self.expected['aggregates'][:7]):
            with self.subTest(first_key=ordered_rows[0]['metric']):
                actual = copy.deepcopy(self.expected)
                actual['aggregates'] = copy.deepcopy(ordered_rows)
                actual_snapshot = copy.deepcopy(actual)
                coverage = finalizer.compare_complete(actual, self.expected, self.lock)
                self.assertEqual(len(coverage), 24)
                self.assertEqual(coverage, [list(key) for key in expected_keys])
                self.assertEqual(actual, actual_snapshot)
                self.assertEqual(self.expected, expected_snapshot)

    def test_missing_duplicate_extra_and_changed_metric_rows_are_rejected(self):
        cases = {}
        cases['missing'] = copy.deepcopy(self.expected)
        cases['missing']['aggregates'].pop()
        cases['duplicate'] = copy.deepcopy(self.expected)
        cases['duplicate']['aggregates'].append(copy.deepcopy(cases['duplicate']['aggregates'][0]))
        cases['extra'] = copy.deepcopy(self.expected)
        extra = copy.deepcopy(cases['extra']['aggregates'][0])
        extra['metric'] = 'recall100'
        cases['extra']['aggregates'].append(extra)
        cases['changed_score'] = copy.deepcopy(self.expected)
        changed = cases['changed_score']['aggregates'][0]
        changed['per_seed_scores'][0] += .01
        # A self-consistent altered mean/SD must still differ from the full audit.
        changed['mean'] = statistics.mean(changed['per_seed_scores'])
        changed['sample_std'] = statistics.stdev(changed['per_seed_scores'])
        for name, actual in cases.items():
            with self.subTest(change=name), self.assertRaises(ValueError):
                finalizer.compare_complete(actual, self.expected, self.lock)

    def test_nonaggregate_fields_and_seed_order_remain_strict(self):
        cases = {}
        cases['epoch'] = copy.deepcopy(self.expected)
        cases['epoch']['epoch_selections'][0]['selected_epoch'] += 5
        cases['seed_order'] = copy.deepcopy(self.expected)
        cases['seed_order']['seed_ids'] = list(reversed(self.seeds))
        cases['selection_order'] = copy.deepcopy(self.expected)
        cases['selection_order']['epoch_selections'].reverse()
        cases['configuration'] = copy.deepcopy(self.expected)
        cases['configuration']['common']['config']['lr'] = .001
        cases['unrecognized_added_field'] = copy.deepcopy(self.expected)
        cases['unrecognized_added_field']['unaudited_override'] = True
        for name, actual in cases.items():
            with self.subTest(change=name), self.assertRaises(ValueError):
                finalizer.compare_complete(actual, self.expected, self.lock)

    def test_matching_report_checksums_still_require_correct_file_sha(self):
        bad_expected = copy.deepcopy(self.expected)
        bad_expected['artifact_sha256'][str(self.artifact)] = '0' * 64
        with self.subTest(change='incorrect_declared_sha'), self.assertRaises(ValueError):
            finalizer.compare_complete(copy.deepcopy(bad_expected), bad_expected, self.lock)
        self.artifact.write_bytes(b'user,metric\n1,0.75\n')
        with self.subTest(change='artifact_modified_after_recording'), self.assertRaises(ValueError):
            finalizer.compare_complete(copy.deepcopy(self.expected), self.expected, self.lock)


if __name__ == '__main__':
    unittest.main()
