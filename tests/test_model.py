import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fafirec.model import FAFIRecModel, initialize_from_teacher


class MethodModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(51)
        self.features = torch.randn(13, 7)
        self.features[0].zero_()
        self.history = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
        self.positive = torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]])
        self.negative = torch.tensor([[0, 0, 9, 9, 9], [0, 10, 10, 10, 10]])
        self.diagnostic = torch.tensor([[0.8, -0.2], [-0.7, 1.1]])

    def model(self, control='diagnostic', **kwargs):
        return FAFIRecModel(12, self.features, d_model=8, max_len=5,
                           n_heads=2, dropout=0, control=control, **kwargs).eval()

    def test_default_is_zero_initialized_linear_control_and_matches_uniform(self):
        diagnostic = self.model()
        uniform = self.model('uniform')
        uniform.load_state_dict(diagnostic.state_dict())
        self.assertEqual(diagnostic.control, 'diagnostic')
        self.assertEqual(float(diagnostic.control_bias), 0)
        self.assertEqual(diagnostic.control_weights.tolist(), [0, 0])
        branches = diagnostic.encode_branches(self.history, self.diagnostic)
        self.assertTrue((branches['rho'] == 0.5).all())
        self.assertTrue((branches['gate'] == 0.5).all())
        torch.testing.assert_close(diagnostic.score_all(self.history, self.diagnostic),
                                   uniform.score_all(self.history), rtol=0, atol=0)
        for key in ('loss', 'rec_loss', 'align_loss'):
            torch.testing.assert_close(
                diagnostic(self.history, self.positive, self.negative, self.diagnostic)[key],
                uniform(self.history, self.positive, self.negative)[key], rtol=0, atol=0)

    def test_control_coefficients_are_signed_and_diagnostics_detached(self):
        model = self.model()
        with torch.no_grad():
            model.control_bias.fill_(0.3)
            model.control_weights.copy_(torch.tensor([-2.0, 3.0]))
        features = self.diagnostic.clone().requires_grad_(True)
        score, _ = model._control_scores(self.history.ne(0), features)
        expected = 0.3 - 2 * features[:, 0] + 3 * features[:, 1]
        torch.testing.assert_close(score[:, -1], expected)
        score.sum().backward()
        self.assertIsNone(features.grad)
        self.assertIsNotNone(model.control_weights.grad)

    def test_gate_derivatives_match_signed_conflict_response(self):
        model = self.model().double()
        eta = F.softplus(model.conflict_scale_raw).detach()
        for value in (-0.7, 0.0, 0.7):
            score = torch.tensor([[value]], dtype=torch.double, requires_grad=True)
            delta = torch.full((1, 1, 8), 0.4, dtype=torch.double, requires_grad=True)
            rho = torch.sigmoid(score)
            gate = model._fusion_gate(score, rho, delta)
            d_score, d_delta = torch.autograd.grad(gate.sum(), (score, delta))
            expected_s = (gate * (1 - gate) * (1 + 2 * eta * rho.unsqueeze(-1) *
                                             (1 - rho.unsqueeze(-1)) * delta)).sum(-1)
            expected_delta = eta * (2 * rho.unsqueeze(-1) - 1) * gate * (1 - gate)
            torch.testing.assert_close(d_score, expected_s)
            torch.testing.assert_close(d_delta, expected_delta)
            self.assertTrue((d_score > 0).all())
            if value == 0:
                torch.testing.assert_close(d_score, ((1 + eta * delta / 2) / 4).sum(-1))

    def test_alignment_updates_states_but_has_no_controller_gradient(self):
        model = self.model()
        with torch.no_grad():
            model.control_weights.copy_(torch.tensor([0.4, -0.8]))
        branches = model.encode_branches(self.history, self.diagnostic)
        last_c = model._last(branches['collaborative'], self.history.ne(0))
        last_t = model._last(branches['textual'], self.history.ne(0))
        scores = model._last(branches['scores'].unsqueeze(-1), self.history.ne(0)).squeeze(-1)
        weights = model.alignment_weights(scores)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.))
        model._weighted_mmd(last_c, last_t, scores).backward()
        self.assertIsNone(model.control_weights.grad)
        self.assertIsNone(model.control_bias.grad)
        self.assertIsNotNone(model.proj_content.weight.grad)
        self.assertTrue(torch.isfinite(model.proj_content.weight.grad).all())

    def test_neutral_start_keeps_recommendation_gradient_path(self):
        model = self.model()
        loss = model(self.history, self.positive, self.negative, self.diagnostic)['rec_loss']
        loss.backward()
        self.assertIsNotNone(model.control_weights.grad)
        self.assertTrue(torch.isfinite(model.control_weights.grad).all())
        self.assertGreater(float(model.control_weights.grad.abs().sum()), 0)

    def test_left_right_padding_and_causal_prefixes(self):
        left = torch.tensor([[0, 0, 1, 2, 3]])
        right = torch.tensor([[1, 2, 3, 0, 0]])
        for control in sorted(FAFIRecModel.CONTROLS):
            model = self.model(control)
            torch.testing.assert_close(model.score_all(left, self.diagnostic[:1]),
                                       model.score_all(right, self.diagnostic[:1]),
                                       rtol=1e-5, atol=1e-6)
        model = self.model()
        histories = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 7, 8]])
        branch = model.encode_branches(histories, torch.zeros(2, 2))['fused']
        torch.testing.assert_close(branch[0, 1:3], branch[1, 1:3])

    def test_chunk_scores_and_training_margin_match_retrieval(self):
        model = self.model()
        with torch.no_grad():
            model.control_weights.copy_(torch.tensor([0.2, -0.6]))
        full = model.score_all(self.history, self.diagnostic)
        chunks = [torch.arange(0, 4), torch.arange(4, 9), torch.arange(9, 13)]
        joined = torch.cat([model.score_candidates(self.history, ids, self.diagnostic)
                            for ids in chunks], dim=1)
        torch.testing.assert_close(joined, full, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isneginf(full[:, 0]).all())
        # Independently retrieve after every valid prefix to reconstruct BPR.
        margins = []
        for u in range(len(self.history)):
            for t in range(self.history.shape[1]):
                if self.history[u, t] and self.positive[u, t]:
                    prefix = self.history[u:u+1].clone()
                    prefix[:, t+1:] = 0
                    scores = model.score_all(prefix, self.diagnostic[u:u+1])[0]
                    margins.append(scores[self.positive[u, t]] - scores[self.negative[u, t]])
        expected = F.softplus(-torch.stack(margins)).mean()
        torch.testing.assert_close(model(self.history, self.positive, self.negative,
                                        self.diagnostic)['rec_loss'], expected)

    def test_alignment_gathers_last_supervised_position_and_omits_empty_user(self):
        model = self.model()
        history = torch.cat([self.history, torch.zeros(1, 5, dtype=torch.long)])
        positive = torch.cat([self.positive, torch.zeros(1, 5, dtype=torch.long)])
        positive[0, -1] = 0
        negative = torch.cat([self.negative, torch.zeros(1, 5, dtype=torch.long)])
        diag = torch.cat([self.diagnostic, torch.zeros(1, 2)])
        branches = model.encode_branches(history, diag)
        supervised = history.ne(0) & positive.ne(0)
        expected = model._last(branches['collaborative'], supervised)[:2]
        with patch.object(model, '_weighted_mmd', wraps=model._weighted_mmd) as measured:
            parts = model(history, positive, negative, diag)
            self.assertTrue(torch.isfinite(parts['loss']))
            torch.testing.assert_close(measured.call_args.args[0], expected)
        empty = torch.zeros(2, 5, dtype=torch.long)
        empty_loss = model(empty, empty, empty, torch.zeros(2, 2))['loss']
        self.assertEqual(float(empty_loss), 0)
        empty_loss.backward()

    def test_item_fusion_mixes_raw_vectors_and_frozen_features_are_external(self):
        model = self.model()
        features = model.text_features.clone()
        ids = torch.tensor([1, 2, 3])
        raw_id = model.item_id_table(ids)
        raw_text = model.proj_content(features[ids])
        expected = F.normalize((raw_id + raw_text) / 2, dim=-1)
        torch.testing.assert_close(model.item_vectors(ids), expected)
        separately_normalized = F.normalize((F.normalize(raw_id, dim=-1) +
                                            F.normalize(raw_text, dim=-1)) / 2, dim=-1)
        self.assertFalse(torch.allclose(expected, separately_normalized))
        model(self.history, self.positive, self.negative, self.diagnostic)['loss'].backward()
        self.assertFalse(model.text_features.requires_grad)
        self.assertNotIn('text_features', model.state_dict())
        torch.testing.assert_close(model.text_features, features)

    def test_structural_ablations_change_only_their_declared_path(self):
        model = self.model('diagnostic_uniform_alignment')
        weights = model.alignment_weights(torch.tensor([-8., 0., 6.]))
        torch.testing.assert_close(weights, torch.full((3,), 1/3))
        model = self.model('diagnostic_no_conflict')
        score = torch.tensor([[-1.], [1.]])
        rho = score.sigmoid()
        a = model._fusion_gate(score, rho, torch.zeros(2, 1, 8))
        b = model._fusion_gate(score, rho, torch.ones(2, 1, 8))
        torch.testing.assert_close(a, b)
        for control, masked_column in [('diagnostic_loss_only', 0), ('diagnostic_gradient_only', 1)]:
            model = self.model(control)
            with torch.no_grad():
                model.control_weights.fill_(1.)
            changed = self.diagnostic.clone()
            changed[:, masked_column] += 100
            first = model._control_scores(self.history.ne(0), self.diagnostic)[0]
            second = model._control_scores(self.history.ne(0), changed)[0]
            torch.testing.assert_close(first, second)

    def test_no_alignment_ablation_keeps_predictions_and_recommendation_only_gradients(self):
        reference = self.model('diagnostic')
        with torch.no_grad():
            reference.control_weights.copy_(torch.tensor([0.3, -0.7]))
        ablation = self.model('diagnostic_no_alignment')
        ablation.load_state_dict(reference.state_dict())
        torch.testing.assert_close(ablation.score_all(self.history, self.diagnostic),
                                   reference.score_all(self.history, self.diagnostic),
                                   rtol=0, atol=0)
        expected = reference(self.history, self.positive, self.negative, self.diagnostic)
        self.assertGreater(float(expected['align_loss'].detach()), 0)
        with patch.object(ablation, '_weighted_mmd', side_effect=AssertionError('MMD must not run')):
            actual = ablation(self.history, self.positive, self.negative, self.diagnostic)
        self.assertEqual(float(actual['align_loss'].detach()), 0)
        torch.testing.assert_close(actual['loss'], actual['rec_loss'], rtol=0, atol=0)
        torch.testing.assert_close(actual['rec_loss'], expected['rec_loss'], rtol=0, atol=0)
        actual['loss'].backward()
        expected['rec_loss'].backward()
        for name, parameter in reference.named_parameters():
            gradient = dict(ablation.named_parameters())[name].grad
            if parameter.grad is None:
                self.assertIsNone(gradient, name)
            else:
                self.assertIsNotNone(gradient, name)
                torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)

    def test_teacher_initialization_copies_only_shared_state_without_aliasing(self):
        teacher, student = self.model('cf'), self.model()
        projection = student.proj_content.weight.detach().clone()
        initialize_from_teacher(student, teacher)
        torch.testing.assert_close(student.item_id_table.weight, teacher.item_id_table.weight)
        torch.testing.assert_close(student.proj_content.weight, projection)
        self.assertTrue(student.control_weights.requires_grad)
        self.assertIsNot(student.item_id_table.weight, teacher.item_id_table.weight)
        before = teacher.item_id_table.weight.detach().clone()
        with torch.no_grad():
            student.item_id_table.weight[1].add_(1.)
        torch.testing.assert_close(teacher.item_id_table.weight, before)
        with self.assertRaisesRegex(ValueError, 'collaborative-only'):
            initialize_from_teacher(student, self.model())

    def test_invalid_diagnostics_candidates_and_negatives_fail_explicitly(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'diagnostics'):
            model.score_all(self.history)
        with self.assertRaisesRegex(ValueError, 'finite'):
            model.score_all(self.history, torch.full((2, 2), float('nan')))
        with self.assertRaisesRegex(ValueError, 'integer'):
            model.score_candidates(self.history, torch.tensor([1.5]), self.diagnostic)
        negatives = self.negative.clone()
        negatives[0, -1] = 0
        with self.assertRaisesRegex(ValueError, 'nonpadding negative'):
            model(self.history, self.positive, negatives, self.diagnostic)


if __name__ == '__main__':
    unittest.main()
