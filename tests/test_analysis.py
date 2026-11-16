import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_results import analyze_runs, paired_seed_comparison, write_analysis
from fafirec.metrics import summarize_rows


class AnalysisTests(unittest.TestCase):
    def test_paired_seed_unit_and_exact_sign_flip_resolution(self):
        result = paired_seed_comparison([.1] * 5, bootstrap_samples=200, seed=1)
        self.assertAlmostEqual(result['mean_difference'], .1)
        self.assertAlmostEqual(result['two_sided_sign_flip_p'], 2 / 32)
        np.testing.assert_allclose(result['bootstrap_95_ci'], [.1, .1])
        single = paired_seed_comparison([.1], bootstrap_samples=200)
        self.assertIsNone(single['bootstrap_95_ci'])
        self.assertIsNone(single['two_sided_sign_flip_p'])

    def write_fixture(self, root, seed, control, rank, user='u'):
        path = root/f'seed_{seed}'
        path.mkdir(exist_ok=True)
        with (path/f'{control}_per_user.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['user', 'seed', 'control', 'group', 'target', 'rank', 'split'])
            writer.writeheader()
            writer.writerow({'user': user, 'seed': seed, 'control': control, 'group': 'cold',
                             'target': 5, 'rank': rank, 'split': 'test'})

    def test_identical_users_across_seeds_are_not_pooled_as_independent(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            for seed in (1, 2, 3):
                self.write_fixture(root, seed, 'diagnostic', 1)
                self.write_fixture(root, seed, 'uniform', 2)
            report = analyze_runs(root, ks=(10,), bootstrap_samples=200)
            self.assertEqual(report['provenance']['status'], 'unverified_csv_only')
            comparison = report['paired_comparisons'][0]
            self.assertEqual(comparison['paired_seeds'], 3)
            self.assertEqual(comparison['seed_ids'], [1, 2, 3])
            aggregate = report['aggregates'][0]
            self.assertEqual(aggregate['users_per_seed'], 1)
            self.assertEqual(aggregate['seed_count'], 3)
            write_analysis(report, root/'analysis')
            self.assertTrue((root/'analysis/paper_table.csv').exists())

    def test_mismatched_populations_and_seed_sets_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_fixture(root, 1, 'diagnostic', 1, 'u1')
            self.write_fixture(root, 1, 'uniform', 2, 'u2')
            with self.assertRaisesRegex(ValueError, 'Paired users'):
                analyze_runs(root, bootstrap_samples=200)
            self.write_fixture(root, 1, 'uniform', 2, 'u1')
            self.write_fixture(root, 2, 'diagnostic', 1, 'u1')
            with self.assertRaisesRegex(ValueError, 'different seed sets'):
                analyze_runs(root, bootstrap_samples=200)

    def write_complete_pipeline(self, root):
        cfg = {'seeds': [1, 2], 'controls': ['diagnostic', 'uniform'],
               'candidate_policy': 'full', 'ks': [10]}
        identity = {'user_order': ['u'], 'items': 5, 'records_sha256': 'fixture-records'}
        source = {'file_hashes': {'model.py': 'a' * 64}, 'base_commit': 'fixture-commit'}
        runtime = {'torch': 'fixture-runtime'}
        fingerprint = hashlib.sha256(json.dumps({'config': cfg, 'identity': identity,
                                                 'source_hashes': source['file_hashes'], 'runtime': runtime},
                                                sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        manifest = {'config': cfg, 'identity': identity, 'source': source, 'runtime': runtime,
                    'candidate_policy': 'full', 'fingerprint': fingerprint, 'status': 'fixture_only'}
        (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        summaries = []
        for seed in cfg['seeds']:
            for control in cfg['controls']:
                rank = 1 if control == 'diagnostic' else 2
                self.write_fixture(root, seed, control, rank)
                job = root/f'seed_{seed}'
                (job/control).mkdir()
                checkpoint = job/control/'best.pt'
                checkpoint.write_bytes(b'checkpoint fixture, not a torch model')
                diagnostic = job/'diagnostics.pt'
                diagnostic.write_bytes(b'diagnostic fixture')
                summary = {'seed': seed, 'control': control, 'fingerprint': fingerprint,
                           'test': summarize_rows([{'rank': rank, 'group': 'cold'}], (10,)),
                           'per_user_sha256': hashlib.sha256((job/f'{control}_per_user.csv').read_bytes()).hexdigest(),
                           'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           'diagnostics_sha256': hashlib.sha256(diagnostic.read_bytes()).hexdigest()}
                (job/f'{control}_summary.json').write_text(json.dumps(summary), encoding='utf-8')
                summaries.append(summary)
        (root/'summary.json').write_text(json.dumps(summaries), encoding='utf-8')

    def test_complete_pipeline_checks_fingerprints_populations_and_hashes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_complete_pipeline(root)
            report = analyze_runs(root, ks=(10,), bootstrap_samples=200)
            provenance = report['provenance']
            self.assertTrue(provenance['configured_jobs_verified'])
            self.assertEqual(provenance['job_count'], 4)
            self.assertEqual(provenance['checkpoint_checksums_verified'], 4)
            self.assertEqual(provenance['diagnostic_checksums_verified'], 4)

    def test_corrupted_csv_rejected_even_if_parsed_ranks_unchanged(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_complete_pipeline(root)
            with (root/'seed_1/diagnostic_per_user.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'CSV checksum mismatch'):
                analyze_runs(root, ks=(10,), bootstrap_samples=200)

    def test_incomplete_pipeline_is_not_reported_as_complete_paired_result(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_complete_pipeline(root)
            (root/'seed_2/uniform_summary.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete pipeline'):
                analyze_runs(root, ks=(10,), bootstrap_samples=200)

    def test_policy_mix_and_control_filename_mismatch_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_complete_pipeline(root)
            manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            manifest['candidate_policy'] = 'exclude_seen_restore_target'
            (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Candidate-policy mix'):
                analyze_runs(root, ks=(10,), bootstrap_samples=200)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_fixture(root, 1, 'uniform', 2)
            (root/'seed_1/uniform_per_user.csv').rename(root/'seed_1/diagnostic_per_user.csv')
            with self.assertRaisesRegex(ValueError, 'filename'):
                analyze_runs(root, ks=(10,), bootstrap_samples=200)


if __name__ == '__main__':
    unittest.main()
