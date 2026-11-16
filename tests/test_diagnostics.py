import copy
import unittest

import torch
from torch.nn import functional as F

from fafirec.diagnostics import compute_diagnostics, query_diagnostics
from fafirec.model import FAFIRecModel


class TeacherDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.teacher = FAFIRecModel(8, torch.randn(9, 5), d_model=8,
                                   max_len=5, n_heads=2, dropout=0.5, control='cf')
        self.records = [
            {'user': 'a', 'train': [1, 2, 3], 'validation': 4, 'test': 5},
            {'user': 'b', 'train': [2, 4, 6, 7], 'validation': 1, 'test': 8},
            {'user': 'c', 'train': [1, 7], 'validation': 8, 'test': 3},
        ]

    def test_query_gradient_is_closed_form_unit_scale_and_input_detached(self):
        query = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
        positive = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
        negative = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
        loss, gradient = query_diagnostics(query, positive, negative)
        margin = (query.detach() * (positive.detach() - negative.detach())).sum(-1)
        expected_loss = F.softplus(-margin)
        expected_gradient = torch.sigmoid(-margin) * (positive.detach() - negative.detach()).norm(dim=-1)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(gradient, expected_gradient)
        torch.testing.assert_close(gradient, (1 - (-loss).exp()) *
                                   (positive.detach() - negative.detach()).norm(dim=-1))
        self.assertIsNone(query.grad)
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)
        self.assertFalse(loss.requires_grad)
        self.assertFalse(gradient.requires_grad)
        with torch.inference_mode():
            second = query_diagnostics(query, positive, negative)
        torch.testing.assert_close(loss, second[0])
        torch.testing.assert_close(gradient, second[1])

    def test_heldout_targets_do_not_enter_diagnostics_or_negatives(self):
        first, provenance = compute_diagnostics(self.teacher, self.records, 5, 2, 2025)
        altered = copy.deepcopy(self.records)
        for record in altered:
            del record['validation']
            del record['test']
        second, same = compute_diagnostics(self.teacher, altered, 5, 2, 2025)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(provenance, same)
        self.assertEqual(provenance['user_order'], ['a', 'b', 'c'])
        for record, negative in zip(self.records, provenance['sampled_negatives']):
            self.assertNotIn(negative, record['train'])
            self.assertTrue(1 <= negative <= self.teacher.n_items)

    def test_cache_normalization_is_training_population_not_prediction_batch(self):
        first, manifest = compute_diagnostics(self.teacher, self.records, 5, 1, 99)
        second, other = compute_diagnostics(self.teacher, self.records, 5, 3, 99)
        self.assertEqual(first.device.type, 'cpu')
        self.assertFalse(first.requires_grad)
        torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
        raw = torch.tensor(manifest['raw_log_features'])
        expected = (raw - raw.mean(0)) / raw.std(0, unbiased=False).clamp_min(1e-6)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(first.mean(0), torch.zeros(2), atol=2e-5, rtol=0)
        self.assertEqual(manifest['sampled_negatives'], other['sampled_negatives'])

    def test_teacher_is_dropout_free_and_parameters_are_unchanged(self):
        self.teacher.train()
        original = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        rng_state = torch.get_rng_state().clone()
        first, _ = compute_diagnostics(self.teacher, self.records, 5, 2, 22)
        self.assertFalse(self.teacher.training)
        second, _ = compute_diagnostics(self.teacher, self.records, 5, 2, 22)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)
        for key, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_diagnostic_uses_last_training_transition_and_ignores_sqrt_d(self):
        _, manifest = compute_diagnostics(self.teacher, self.records, 5, 3, 11)
        hist = torch.tensor([[0, 0, 0, 1, 2]])
        with torch.no_grad():
            q = self.teacher.encode_cf(hist)[0]
            positive = F.normalize(self.teacher.item_id_table.weight[3], dim=-1)
            negative = F.normalize(self.teacher.item_id_table.weight[manifest['sampled_negatives'][0]], dim=-1)
            margin = (q * (positive - negative)).sum()
        self.assertAlmostEqual(manifest['loss'][0], float(F.softplus(-margin)), places=6)
        self.assertAlmostEqual(manifest['query_gradient_norm'][0],
                               float(torch.sigmoid(-margin) * (positive - negative).norm()), places=6)

    def test_dense_user_and_zero_variance_remain_finite(self):
        records = [{'user': 'only', 'train': list(range(1, 8))}]
        tensor, manifest = compute_diagnostics(self.teacher, records, 5, 2, 4)
        self.assertEqual(manifest['sampled_negatives'], [8])
        torch.testing.assert_close(tensor, torch.zeros(1, 2))
        self.assertEqual(manifest['population_std'], [0, 0])

    def test_ineligible_users_and_wrong_teacher_fail_before_sampling(self):
        for records in [[], [{'user': 'x', 'train': [1]}],
                        [{'user': 'x', 'train': list(range(1, 9))}],
                        [{'user': 'x', 'train': [0, 1]}],
                        [{'user': 'x', 'train': [True, 2]}],
                        [{'user': 'x', 'train': [1, 2]}, {'user': 'x', 'train': [2, 3]}]]:
            with self.assertRaises(ValueError):
                compute_diagnostics(self.teacher, records, 5, 2, 4)
        wrong = FAFIRecModel(8, torch.randn(9, 5), d_model=8, max_len=5, n_heads=2)
        with self.assertRaisesRegex(ValueError, 'collaborative-only'):
            compute_diagnostics(wrong, self.records, 5, 2, 4)


if __name__ == '__main__':
    unittest.main()
