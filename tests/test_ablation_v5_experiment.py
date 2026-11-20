import unittest
import numpy as np
import torch
from experiments.ablation_v5 import experiment as ex
from experiments.ablation_v5.gate import make_gate


class AblationExperimentTests(unittest.TestCase):
    def test_fixed_permutation_preserves_joint_strata_and_is_reproducible(self):
        lengths = [4] * 20 + [8] * 20 + [30] * 20
        fractions = [0., .5, 1., 1.] * 15
        permutation, report = ex.fixed_permutation(lengths, fractions)
        again, same = ex.fixed_permutation(lengths, fractions)
        self.assertEqual(permutation, again); self.assertEqual(report, same)
        self.assertEqual(sorted(permutation), list(range(60)))
        bins = np.searchsorted(report['quantile_cutpoints'], lengths, side='right')
        for target, source in enumerate(permutation):
            self.assertEqual(bins[target], bins[source])
            self.assertEqual(fractions[target], fractions[source])
        self.assertGreater(sum(a != b for a, b in enumerate(permutation)), 20)

    def test_mean_diagnostics_is_training_mean_and_retains_candidate_terms(self):
        torch.manual_seed(2)
        gate = make_gate('full')
        train = torch.randn(8, 7) + 3
        counts = torch.tensor([0., 2., 4., 7., 1.])
        gate.fit_feature_statistics(train); gate.fit_item_statistics(counts)
        with torch.no_grad():
            gate.user_linear.weight.fill_(.4); gate.user_linear.bias.fill_(.2)
            gate.pop_slope.bias.fill_(.3); gate.conflict_coef.fill_(.1)
        data = {'sem': torch.randn(3, 5), 'cf': torch.randn(3, 5), 'features': torch.randn(3, 7),
            'counts': counts, 'available': torch.tensor([[False,True,True,True,True]] * 3)}
        scores, _ = ex.predict(gate, data, {}, 'mean_diagnostics')
        expected = gate(data['sem'], data['cf'], gate.feature_mean[None].expand(3, -1), counts, alpha=.025)
        self.assertTrue(np.array_equal(scores, expected.detach().numpy()))
        self.assertFalse(torch.equal(expected, data['sem'] + .025 * data['cf']))

    def test_scalar_and_user_mean_gate_preserve_only_declared_variation(self):
        torch.manual_seed(9)
        gate = make_gate('full'); gate.fit_feature_statistics(torch.randn(7, 7))
        counts = torch.tensor([0., 1., 2., 8., 3.]); gate.fit_item_statistics(counts)
        with torch.no_grad(): gate.conflict_coef.fill_(.7)
        data = {'sem': torch.randn(3, 5), 'cf': torch.randn(3, 5), 'features': torch.randn(3, 7),
            'counts': counts, 'available': torch.tensor([[False,True,True,True,True]] * 3)}
        scalar, _ = ex.predict(gate, data, {}, 'scalar_mean_gate', 1.4)
        self.assertTrue(np.allclose(scalar, (data['sem'] + .025 * 1.4 * data['cf']).numpy()))
        per_user, _ = ex.predict(gate, data, {}, 'user_mean_gate')
        with torch.no_grad():
            g = gate.multipliers(data['sem'], data['cf'], data['features'], counts)
            mean = g[:, 1:].mean(1, keepdim=True)
        self.assertTrue(np.array_equal(per_user, (data['sem'] + .025 * mean * data['cf']).numpy()))


if __name__ == '__main__': unittest.main()
