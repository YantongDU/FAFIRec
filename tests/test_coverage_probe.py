"""Target coverage and exact epoch-boundary resume for the coverage controls."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from baselines.ace2026_model import ACE2026Model
from fafirec.runner import seed_everything
from experiments.diagnostic_v4 import coverage_probe as coverage


class CoverageProbeTests(unittest.TestCase):
    def test_all_windows_cover_every_target_once_with_shifted_pairs(self):
        records = [{'train': list(range(1, 115))}, {'train': [8, 3]}]
        data = coverage.CoverageDataset(records, 50, 'all_windows', 2025, 1)
        self.assertEqual(len(data), 4)
        self.assertEqual(int(data.coverage_counts.sum()), 114)
        self.assertTrue(data.coverage_counts.eq(1).all())
        expected_targets = list(range(2, 115)) + [3]
        targets = []
        for i in range(len(data)):
            hist, pos = data[i]
            targets += pos[pos.ne(0)].tolist()
            user, start, end = data.spans[i]
            self.assertEqual(hist[hist.ne(0)].tolist(), records[user]['train'][start - 1:end - 1])
            self.assertEqual(pos[pos.ne(0)].tolist(), records[user]['train'][start:end])
        self.assertEqual(targets, expected_targets)

    def test_random_window_is_epoch_deterministic_and_uses_only_train(self):
        records = [{'train': list(range(1, 125))}, {'train': [8, 3]}]
        first = coverage.CoverageDataset(records, 50, 'random_window', 2025, 17)
        same = coverage.CoverageDataset(records, 50, 'random_window', 2025, 17)
        self.assertEqual(first.spans, same.spans)
        self.assertEqual(len(first), len(records))
        self.assertTrue(first.coverage_counts.le(1).all())
        spans = set()
        for epoch in range(1, 60):
            data = coverage.CoverageDataset(records, 50, 'random_window', 2025, epoch)
            spans.add(data.spans[0])
            for user, start, end in data.spans:
                self.assertGreaterEqual(end, 2)
                self.assertLessEqual(end, len(records[user]['train']))
                self.assertLessEqual(end - start, 50)
                self.assertEqual(start, max(1, end - 50))
        self.assertGreater(len(spans), 20)
        with self.assertRaisesRegex(ValueError, 'Training-only'):
            coverage.CoverageDataset([{'train': [1, 2], 'test': [3]}], 50, 'random_window', 2025, 1)

    def test_epoch_boundary_resume_is_bit_exact_for_both_modes(self):
        torch.set_num_threads(2)
        records = [{'user': str(i), 'group': 'middle', 'train': [1, 2, 3, 1, 4, 2, 1],
                    'validation': [5, 6]} for i in range(3)]
        features = torch.randn(13, 8, generator=torch.Generator().manual_seed(50))
        features[0].zero_()
        cfg = {'variants': [{'lr': .001}], 'weight_decay': 0., 'max_len': 4,
            'epochs': 3, 'min_epochs': 1, 'eval_interval': 1, 'patience': 9,
            'batch_size': 2, 'grad_clip': None}
        eval_cfg = {'max_len': 4, 'eval_batch': 2, 'ks': [10, 50], 'candidate_policy': 'exclude_seen'}
        manifest = {'fingerprint': 'fixed-test-fixture'}
        def new_model():
            seed_everything(3025)
            return ACE2026Model(features, d_model=8, max_len=4, n_layers=1, n_heads=2,
                inner_size=16, dropout=.1, attn_dropout=.1, reg_weight=500., scale=.1, ce_chunk_size=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in coverage.MODES:
                full, interrupted = root / (mode + '_full'), root / (mode + '_interrupted')
                coverage.fit(new_model(), records, cfg, eval_cfg, 2025, mode, full, manifest)
                dump = coverage.torch_dump
                def stop(path, value):
                    dump(path, value)
                    if Path(path).name == 'last.pt' and value['epoch'] == 1:
                        raise RuntimeError('saved epoch interruption')
                with patch.object(coverage, 'torch_dump', side_effect=stop), self.assertRaisesRegex(RuntimeError, 'interruption'):
                    coverage.fit(new_model(), records, cfg, eval_cfg, 2025, mode, interrupted, manifest)
                coverage.fit(new_model(), records, cfg, eval_cfg, 2025, mode, interrupted, manifest, resume=True)
                left, right = [torch.load(path / 'last.pt', weights_only=True) for path in (full, interrupted)]
                self.assertEqual(left['logs'], right['logs'])
                self.assertEqual(left['selected_epoch'], right['selected_epoch'])
                self.assertTrue(torch.equal(left['covered'], right['covered']))
                for key in left['model']:
                    torch.testing.assert_close(left['model'][key], right['model'][key], atol=0, rtol=0)
                for key in left['best_snapshot']['model']:
                    torch.testing.assert_close(left['best_snapshot']['model'][key], right['best_snapshot']['model'][key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
