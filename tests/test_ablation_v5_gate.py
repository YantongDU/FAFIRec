"""Mechanism removal and bit-exact full-controller retraining regression."""
import copy
import unittest

import torch

from experiments.ablation_v5.gate import AblatedGate, SPEC, TRAINING_RECIPE, fit_ablation
from experiments.diagnostic_v4.fusion_model import DiagnosticGate
from experiments.diagnostic_v4.fusion_experiment import RECIPE, fit_gate


def calibration_fixture():
    generator = torch.Generator().manual_seed(73)
    users, items = 11, 19
    available = torch.ones(users, items, dtype=torch.bool)
    available[:, 0] = False
    available[torch.arange(users), 1 + torch.arange(users)] = False
    targets = [[1 + (u + 4) % (items - 1), 1 + (u + 7) % (items - 1)] for u in range(users)]
    return {'sem': torch.randn(users, items, generator=generator),
        'cf': torch.randn(users, items, generator=generator),
        'pop': torch.randn(users, items, generator=generator),
        'features': torch.randn(users, 7, generator=generator), 'available': available,
        'counts': torch.arange(items, dtype=torch.float32), 'targets': targets}


class AblationGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.data = calibration_fixture()

    def fitted(self, variant):
        model = AblatedGate(variant)
        model.fit_feature_statistics(self.data['features'])
        model.fit_item_statistics(self.data['counts'])
        with torch.no_grad():
            model.user_linear.weight.copy_(torch.tensor([[.11, -.3, .2, .17, -.1, .28, -.21]]))
            model.user_linear.bias.fill_(.4)
            model.pop_slope.weight.fill_(.15)
            model.pop_slope.bias.fill_(-.1)
            model.conflict_coef.fill_(.37)
        return model.eval()

    def output(self, model, features=None, counts=None):
        return model(self.data['sem'], self.data['cf'],
            self.data['features'] if features is None else features,
            self.data['counts'] if counts is None else counts, alpha=.025)

    def test_full_twenty_epoch_training_is_bit_exact_to_frozen_v4(self):
        self.assertEqual((RECIPE['gate_epochs'], RECIPE['gate_lr'], RECIPE['gate_batch'],
            RECIPE['train_alpha'], RECIPE['train_beta']), (20, .02, 256, .1, .2))
        snapshot = {k: v.clone() for k, v in self.data.items() if isinstance(v, torch.Tensor)}
        original, old_logs = fit_gate(self.data, 'candidate', .001, 2025, torch.device('cpu'))
        ablated, new_logs = fit_ablation(self.data, 'full', 2025, torch.device('cpu'))
        self.assertEqual(old_logs, new_logs)
        self.assertEqual(len(new_logs), TRAINING_RECIPE['epochs'])
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, ablated.state_dict()[name]), name)
        self.assertTrue(torch.equal(self.output(original), self.output(ablated)))
        self.assertEqual(list(dict(original.named_parameters())), list(dict(ablated.named_parameters())))
        for key, value in snapshot.items():
            self.assertTrue(torch.equal(value, self.data[key]), key)

    def test_feature_family_removal_is_invariant_to_only_the_removed_values(self):
        removals = {'gain_only': list(range(7)), 'length_candidate': [1, 2, 3, 4, 5, 6],
            'no_length': [0], 'no_past': [1, 2, 5, 6], 'no_current': [3, 4]}
        for variant, removed in removals.items():
            model = self.fitted(variant)
            changed = self.data['features'].clone()
            changed[:, removed] += 23.
            self.assertTrue(torch.equal(self.output(model), self.output(model, changed)), variant)
            active = [i for i in range(7) if i not in removed]
            if active:
                changed = self.data['features'].clone(); changed[:, active[0]] += 1.
                self.assertFalse(torch.equal(self.output(model), self.output(model, changed)), variant)

    def test_candidate_term_removal_and_parameter_denominator_stay_matched(self):
        base_order = list(dict(DiagnosticGate(mode='candidate').named_parameters()))
        for variant in SPEC:
            model = self.fitted(variant)
            self.assertEqual(list(dict(model.named_parameters())), base_order)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 17)
            multipliers = model.multipliers(self.data['sem'], self.data['cf'], self.data['features'], self.data['counts'])
            self.assertEqual(multipliers.shape, self.data['sem'].shape)
            self.assertTrue(torch.equal(self.output(model), self.data['sem'] + .025 * multipliers * self.data['cf']), variant)
        no_pop = self.fitted('no_pop')
        counts = self.data['counts'].clone(); counts[1:] = counts[1:].square() + 4.
        self.assertTrue(torch.equal(self.output(no_pop), self.output(no_pop, counts=counts)))
        full = self.fitted('full')
        self.assertFalse(torch.equal(self.output(full), self.output(full, counts=counts)))
        no_conflict = self.fitted('no_conflict')
        before = self.output(no_conflict)
        with torch.no_grad(): no_conflict.conflict_coef.fill_(11.)
        self.assertTrue(torch.equal(before, self.output(no_conflict)))
        before = self.output(full)
        with torch.no_grad(): full.conflict_coef.fill_(11.)
        self.assertFalse(torch.equal(before, self.output(full)))

    def test_gain_only_training_changes_only_the_scalar_bias_and_stays_constant(self):
        model, logs = fit_ablation(self.data, 'gain_only', 2025, torch.device('cpu'))
        self.assertEqual(len(logs), 20)
        self.assertNotEqual(model.user_linear.bias.item(), 0.)
        for name, parameter in model.named_parameters():
            if name != 'user_linear.bias':
                self.assertEqual(parameter.abs().sum().item(), 0., name)
        value = 2. * model.user_linear.bias.sigmoid()
        expected = self.data['sem'] + .025 * value * self.data['cf']
        self.assertTrue(torch.equal(self.output(model), expected))
        changed = self.data['features'] * 13. - 7.
        self.assertTrue(torch.equal(self.output(model), self.output(model, changed)))

    def test_checkpoint_cannot_silently_change_the_removed_mechanisms(self):
        model = self.fitted('no_past')
        copied = AblatedGate('no_past')
        copied.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(torch.equal(self.output(model), self.output(copied)))
        with self.assertRaisesRegex(ValueError, 'specification differs'):
            AblatedGate('full').load_state_dict(model.state_dict())
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            AblatedGate('not_an_ablation')
        bad = copy.deepcopy(self.data); bad['targets'][0] = [0]
        with self.assertRaisesRegex(ValueError, 'already observed'):
            fit_ablation(bad, 'no_current', 2025, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
