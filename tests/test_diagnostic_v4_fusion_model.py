"""Observed-prefix causality, neutral residuals and fixed normalization checks."""
import inspect
import unittest

import torch
from torch import nn

from experiments.diagnostic_v4.fusion_model import (DiagnosticGate,
    candidate_standardize, probe_tasks, score_histories)


class SmallSemantic(nn.Module):
    """Order-sensitive deterministic scorer; records its actual model inputs."""
    def __init__(self):
        super().__init__()
        table = torch.randn(33, 6, generator=torch.Generator().manual_seed(22))
        table[0].zero_()
        self.register_buffer('table', table)
        self.max_len, self.score_scale, self.calls = 50, 1., []

    def item_vectors(self):
        return self.table

    def user_vectors(self, tokens):
        self.calls += [[int(x) for x in row if x] for row in tokens.tolist()]
        weights = tokens.ne(0).cumsum(1).to(torch.float32)
        return (self.table[tokens] * weights[:, :, None]).sum(1) / weights.sum(1).clamp_min(1.)[:, None]


class DiagnosticFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.semantic = SmallSemantic().eval()
        self.B = torch.randn(32, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        self.B.fill_diagonal_(0.)

    def test_current_scores_use_only_observed_candidates_and_binary_last50(self):
        histories = [[1, 2, 3, 4], [2, 3], [5, 5, 6, 7]]
        sem, cf, features, available, report = score_histories(self.semantic, self.B,
            histories, [2, 2, 2], batch_size=1)
        self.assertEqual(sem.dtype, torch.float32)
        self.assertEqual(sem.device.type, 'cpu')
        self.assertEqual(features.shape, (3, 7))
        for owner, history in enumerate(histories):
            self.assertFalse(available[owner, 0])
            self.assertFalse(available[owner, list(set(history))].any())
            self.assertEqual(sem[owner, ~available[owner]].abs().sum().item(), 0.)
            values = sem[owner, available[owner]]
            torch.testing.assert_close(values.mean(), torch.tensor(0.), atol=1e-6, rtol=0)
            torch.testing.assert_close(values.std(unbiased=False), torch.tensor(1.), atol=1e-6, rtol=0)
            raw = torch.zeros(1, 33)
            raw[0, 1:] = self.B[[i - 1 for i in set(history[-50:])]].sum(0).float()
            torch.testing.assert_close(cf[owner:owner + 1], candidate_standardize(raw, available[owner:owner + 1]), atol=1e-6, rtol=1e-6)
        self.assertEqual(report['probe_counts'], [2, 0, 2])
        self.assertEqual(features[1, [1, 2, 5, 6]].abs().sum().item(), 0.)
        self.assertFalse(features.requires_grad)
        self.assertNotIn('targets', inspect.signature(score_histories).parameters)
        with self.assertRaises(TypeError):
            score_histories(self.semantic, self.B, histories, [2, 2, 2], targets=[8, 9, 10])

    def test_probe_cutoff_and_model_calls_exclude_the_probe_outcome_and_later_events(self):
        history = [1, 2, 3, 4, 5]
        tasks, _ = probe_tasks([history], [4])
        self.assertEqual(tasks, [(0, 4, [1, 2, 3, 4], 5)])
        score_histories(self.semantic, self.B, [history], [3], batch_size=1)
        self.assertEqual(self.semantic.calls, [history, [1, 2, 3], [1, 2, 3, 4]])
        no_probes = score_histories(self.semantic, self.B, [history], [3], probe_window=0)
        self.assertEqual(no_probes[-1]['probe_count'], 0)
        self.assertEqual(no_probes[2][0, [1, 2, 5, 6]].abs().sum().item(), 0.)
        self.assertEqual(probe_tasks([history], [5])[0], [])
        repeated, skipped = probe_tasks([[1, 2, 1]], [2])
        self.assertEqual(repeated, [])
        self.assertEqual(skipped, 1)
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            probe_tasks([history], [6])

    def test_all_zero_initialized_modes_exactly_recover_the_global_residual(self):
        gen = torch.Generator().manual_seed(9)
        features = torch.randn(6, 7, generator=gen, requires_grad=True)
        sem, cf = torch.randn(6, 33, generator=gen), torch.randn(6, 33, generator=gen)
        counts = torch.arange(33, dtype=torch.float32)
        for mode in DiagnosticGate.MODES:
            gate = DiagnosticGate(mode=mode).fit_feature_statistics(features)
            if mode == 'candidate':
                gate.fit_item_statistics(counts)
            output = gate(sem, cf, features, counts, alpha=.1)
            self.assertTrue(torch.equal(output, sem + .1 * cf), mode)
            self.assertTrue(torch.equal(gate.coefficients(features), torch.zeros(6)), mode)
            self.assertTrue(torch.equal(gate.alignment_weights(features), torch.ones(6)), mode)
            if mode == 'global':
                self.assertFalse(any(p.requires_grad for p in gate.parameters()))

    def test_training_statistics_stay_fixed_when_full_train_counts_and_features_change(self):
        features = torch.randn(6, 7, generator=torch.Generator().manual_seed(17))
        gate = DiagnosticGate(mode='candidate').fit_feature_statistics(features)
        inner_counts = torch.arange(33, dtype=torch.float32)
        gate.fit_item_statistics(inner_counts)
        with torch.no_grad():
            gate.user_linear.weight.fill_(.1)
            gate.pop_slope.bias.fill_(.2)
        buffers = {k: v.clone() for k, v in gate.named_buffers()}
        sem, cf = torch.ones(6, 33), torch.ones(6, 33)
        outer_counts = inner_counts * 3
        gate.eval()
        first = gate(sem, cf, features, inner_counts)
        second = gate(sem, cf, features + 2, outer_counts)
        self.assertFalse(torch.equal(first, second))
        for key, value in gate.named_buffers():
            self.assertTrue(torch.equal(value, buffers[key]), key)
        with self.assertRaisesRegex(ValueError, 'already fitted'):
            gate.fit_feature_statistics(features + 2)
        with self.assertRaisesRegex(ValueError, 'already fitted'):
            gate.fit_item_statistics(outer_counts)
        for bad in (torch.ones(33), -inner_counts, torch.full((33,), torch.nan), torch.zeros(34)):
            with self.assertRaises(ValueError):
                gate(sem, cf, features, bad)


if __name__ == '__main__':
    unittest.main()
