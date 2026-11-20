"""CPU mathematical and input-boundary checks for diagonal-zero relations."""
import unittest

import torch

from experiments.diagnostic_v4.relation_model import fit_relation, semantic_relation


class DiagnosticRelationTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.],
                               [1., 0., 1., 1.], [0., 1., 0., 1.]], dtype=torch.float64)
        self.S = torch.tensor([[0., .8, .2, 0.], [.5, 0., .5, 0.],
                               [0., 0., 0., 1.], [.1, .2, .7, 0.]], dtype=torch.float64)

    def test_zero_alignment_exactly_recovers_ease_and_constrained_kkt(self):
        ridge = 2.5
        B, report = fit_relation(self.X, ridge)
        gram = self.X.T @ self.X
        precision = torch.linalg.solve(gram + ridge * torch.eye(4, dtype=torch.float64),
                                       torch.eye(4, dtype=torch.float64))
        ease = -precision / precision.diagonal()[None, :]
        ease.fill_diagonal_(0.)
        self.assertTrue(torch.equal(B, ease))
        offdiag = (gram + ridge * torch.eye(4, dtype=torch.float64)) @ B - gram
        offdiag.fill_diagonal_(0.)
        self.assertLess(float(offdiag.abs().max()), 1e-13)
        self.assertEqual(report['residual']['diagonal_constraint_max_abs'], 0.)
        self.assertLess(report['residual']['kkt_relative_frobenius'], 1e-13)
        expected_objective = float((self.X - self.X @ B).square().sum() + ridge * B.square().sum())
        self.assertAlmostEqual(report['objective_value'], expected_objective, places=12)

    def test_weighted_alignment_matches_independent_constrained_least_squares(self):
        ridge, mu = 1.3, .7
        weights = torch.tensor([.2, 1., 2., .8], dtype=torch.float64)
        X, S = self.X, self.S
        before = (X.clone(), S.clone(), weights.clone())
        B, report = fit_relation(X, ridge, S, mu, weights)
        # Solve each column from the original stacked least-squares design,
        # eliminating its forbidden self coefficient rather than using P/Z.
        direct = torch.zeros_like(B)
        identity = torch.eye(X.shape[1], dtype=torch.float64)
        root_weight = weights.sqrt()[:, None]
        design = torch.cat([X, ridge ** .5 * identity, mu ** .5 * root_weight * X])
        target = torch.cat([X, torch.zeros_like(identity), mu ** .5 * root_weight * (X @ S)])
        for column in range(X.shape[1]):
            keep = [i for i in range(X.shape[1]) if i != column]
            direct[keep, column] = torch.linalg.lstsq(design[:, keep], target[:, column]).solution
        torch.testing.assert_close(B, direct, rtol=1e-12, atol=1e-12)
        def objective(value):
            return ((X - X @ value).square().sum() + ridge * value.square().sum()
                    + mu * (root_weight * (X @ value - X @ S)).square().sum())
        self.assertAlmostEqual(float(objective(B)), report['objective_value'], places=12)
        perturbation = torch.tensor([[0., .1, -.2, .3], [.1, 0., .2, -.1],
                                      [-.2, .2, 0., .1], [.3, -.1, .1, 0.]], dtype=torch.float64)
        self.assertLess(float(objective(B)), float(objective(B + perturbation)))
        self.assertTrue(torch.equal(B.diagonal(), torch.zeros(4, dtype=torch.float64)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((X, S, weights), before)))
        self.assertEqual(len(report['sha256']['relation_B']), 64)

    def test_weights_affect_only_positive_alignment_and_identity_is_valid(self):
        weights = torch.tensor([0., 3., .5, 2.], dtype=torch.float64)
        unweighted, _ = fit_relation(self.X, 2., self.S, 0.)
        ignored, report = fit_relation(self.X, 2., self.S, 0., weights)
        self.assertTrue(torch.equal(unweighted, ignored))
        self.assertFalse(report['user_weights_used'])
        a, _ = fit_relation(self.X, 2., self.S, 1.)
        b, _ = fit_relation(self.X, 2., self.S, 1., weights)
        self.assertGreater(float((a - b).abs().max()), 1e-3)
        zero, _ = fit_relation(torch.eye(4, dtype=torch.float64), 2.)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        # All-zero item columns may be genuine unseen items; they are not
        # silently stripped or inferred to be PAD from interaction data.
        padded_catalog = torch.cat([self.X, torch.zeros(4, 1, dtype=torch.float64)], dim=1)
        expanded, _ = fit_relation(padded_catalog, 2.)
        self.assertEqual(expanded.shape, (5, 5))
        self.assertTrue(torch.equal(expanded[:, -1], torch.zeros(5, dtype=torch.float64)))

    def test_static_semantic_graph_and_invalid_input_boundaries(self):
        features = torch.tensor([[1., 1., 1.], [2., 1., 1.], [1., 2., 1.],
                                  [3., 2., 1.], [1., 3., 1.]], dtype=torch.float64)
        seen = torch.tensor([True, True, True, False, False])
        snapshot = features.clone()
        S, report = semantic_relation(features, seen, topk=2)
        self.assertTrue(torch.equal(features, snapshot))
        self.assertEqual(S.dtype, torch.float64)
        self.assertEqual(S.shape, (5, 5))
        self.assertTrue(torch.equal(S.diagonal(), torch.zeros(5, dtype=torch.float64)))
        self.assertTrue(bool(((S > 0).sum(1) <= 2).all()))
        torch.testing.assert_close(S.sum(1)[S.sum(1) > 0], torch.ones_like(S.sum(1)[S.sum(1) > 0]))
        S2, report2 = semantic_relation(features, seen, topk=2)
        self.assertTrue(torch.equal(S, S2))
        self.assertEqual(report, report2)
        # Changing an unseen static feature must not change the fitted center.
        changed = features.clone()
        changed[-1] += 10.
        _, changed_report = semantic_relation(changed, seen, topk=2)
        self.assertEqual(report['sha256']['center'], changed_report['sha256']['center'])
        self.assertNotEqual(report['sha256']['semantic_relation_S'], changed_report['sha256']['semantic_relation_S'])
        constant, constant_report = semantic_relation(torch.ones(3, 2, dtype=torch.float64),
                                                       torch.ones(3, dtype=torch.bool))
        self.assertTrue(torch.equal(constant, torch.zeros(3, 3, dtype=torch.float64)))
        self.assertEqual(constant_report['zero_centered_rows'], 3)
        bad_inputs = [
            lambda: fit_relation(self.X.float(), 1.),
            lambda: fit_relation(self.X * .5, 1.),
            lambda: fit_relation(self.X, 0.),
            lambda: fit_relation(self.X, 1., alignment_weight=1.),
            lambda: fit_relation(self.X, 1., torch.zeros(5, 5, dtype=torch.float64), 1.),
            lambda: fit_relation(self.X, 1., self.S, 1., torch.tensor([1., -1., 1., 1.], dtype=torch.float64)),
            lambda: semantic_relation(torch.cat([torch.zeros(1, 3), features]), torch.ones(6, dtype=torch.bool)),
            lambda: semantic_relation(features, torch.zeros(5, dtype=torch.bool)),
            lambda: semantic_relation(features, seen.long()),
            lambda: semantic_relation(features, seen, topk=0),
        ]
        for index, operation in enumerate(bad_inputs):
            with self.subTest(invalid_case=index), self.assertRaises(ValueError):
                operation()


if __name__ == '__main__':
    unittest.main()
