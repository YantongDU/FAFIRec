import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import contextlib
import io

import torch

from fafirec.ml1m_model import PaperFAFIRecModel
from fafirec.model import initialize_from_teacher
from fafirec.transfer import initialize_transfer, optimizer_groups


class TransferInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.randn(10, 4)
        self.features[0].zero_()
        self.kwargs = {'n_items': 9, 'text_features': self.features, 'd_model': 4,
                       'max_len': 5, 'n_heads': 2, 'n_layers': 1, 'dropout': 0.}
        self.teacher = PaperFAFIRecModel(control='cf', **self.kwargs)
        self.records = [{'train': [1, 2, 3, 4, 5]}, {'train': [2, 4, 6, 7, 8]}]

    def student(self):
        return PaperFAFIRecModel(control='diagnostic', **self.kwargs)

    def test_random_half_prior_preserves_historical_initialization_exactly(self):
        torch.manual_seed(9)
        reference = self.student()
        initialize_from_teacher(reference, self.teacher)
        torch.manual_seed(9)
        changed = self.student()
        initialize_transfer(changed, self.teacher, self.records)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, changed.state_dict()[name], rtol=0, atol=0)

    def test_ridge_satisfies_normal_equation_and_excludes_unseen_teacher_rows(self):
        student = self.student()
        teacher_before = {name: value.clone() for name, value in self.teacher.state_dict().items()}
        info = initialize_transfer(student, self.teacher, self.records, 'ridge', .9, .1)
        ids = torch.arange(1, 9)
        x = self.features[ids].double()
        target = self.teacher.item_id_table.weight.detach()[ids].double()
        expected = torch.linalg.solve(x.T @ x + .1 * torch.eye(4).double(), x.T @ target).T
        torch.testing.assert_close(student.proj_content.weight, expected.float())
        self.assertTrue(student.proj_content.weight.requires_grad)
        self.assertEqual(info['training_seen_items'], 8)
        self.assertFalse(info['heldout_interactions_used'])
        with torch.no_grad():
            self.teacher.item_id_table.weight[9].fill_(9999)
        other = self.student()
        initialize_transfer(other, self.teacher, self.records, 'ridge', .9, .1)
        torch.testing.assert_close(student.proj_content.weight, other.proj_content.weight, rtol=0, atol=0)
        for name, value in teacher_before.items():
            if name != 'item_id_table.weight':
                torch.testing.assert_close(value, self.teacher.state_dict()[name], rtol=0, atol=0)

    def test_initialization_does_not_read_heldout_fields_and_matches_controls(self):
        class TrainOnly(dict):
            def __getitem__(self, key):
                if key != 'train':
                    raise AssertionError('held-out fields must not be read')
                return super().__getitem__(key)
        student = self.student()
        records = [TrainOnly(row) for row in self.records]
        initialize_transfer(student, self.teacher, records, 'ridge', .9, 1.)
        self.assertEqual(student.control_weights.tolist(), [0., 0.])
        torch.testing.assert_close(student.gate_bias.sigmoid(), torch.full((4,), .9))
        torch.testing.assert_close(student.item_gate_logit.sigmoid(), torch.full((4,), .9))
        h = torch.tensor([[0, 0, 1, 2, 3]])
        states = student.encode_branches(h, torch.ones(1, 2))
        torch.testing.assert_close(states['gate'], torch.full_like(states['gate'], .9))
        self.assertEqual(float(student.item_vectors()[0].abs().sum()), 0.)

    def test_optimizer_partition_is_exhaustive_disjoint_and_legacy_default(self):
        student = self.student()
        default = optimizer_groups(student, .001)
        self.assertEqual(len(default), 1)
        groups = optimizer_groups(student, .001, 10.)
        self.assertEqual([g['lr'] for g in groups], [.001, .01])
        all_params = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(all_params), len(set(all_params)))
        self.assertEqual(set(all_params), {id(p) for p in student.parameters() if p.requires_grad})
        gate_ids = {id(p) for n, p in student.named_parameters() if n in {'gate_bias', 'item_gate_logit', 'control_bias', 'control_weights', 'conflict_scale_raw'}}
        self.assertEqual({id(p) for p in groups[1]['params']}, gate_ids)
        for invalid in (0, -1, True, float('nan')):
            with self.assertRaises(ValueError):
                optimizer_groups(student, .001, invalid)

    def test_invalid_initialization_settings_fail(self):
        for settings in ({'cf_prior': 1.}, {'cf_prior': 0.}, {'cf_prior': True}, {'ridge': 0.},
                         {'ridge': float('inf')}, {'projection_init': 'unreported'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                initialize_transfer(self.student(), self.teacher, self.records, **settings)

    def test_initial_checkpoint_can_win_without_silently_discarding_it(self):
        from fafirec.ratio_runner import DEFAULTS, fit
        model = self.student()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        cfg = {**DEFAULTS, 'd_model': 4, 'max_len': 5, 'n_heads': 2, 'n_layers': 1,
               'epochs': 1, 'batch_size': 2, 'threads': 2, 'include_initial_checkpoint': True}
        validation = iter([{'all': {'ndcg50': .9}}, {'all': {'ndcg50': .1}}])
        with tempfile.TemporaryDirectory() as directory:
            with patch('fafirec.ratio_runner.evaluate', side_effect=lambda *a, **k: ([], next(validation))):
                with contextlib.redirect_stdout(io.StringIO()):
                    selected = fit(model, self.records, torch.zeros(2, 2), cfg, 17, directory,
                                   'fingerprint', {}, 1, artifact={'stage': 'validation_candidate'})
            self.assertEqual(selected['selected_epoch'], 0)
            self.assertEqual(selected['epochs_completed'], 1)
            self.assertTrue((Path(directory) / 'initial_validation.json').exists())
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_adaptation_window_keeps_epoch_zero_but_allows_recovery_and_resume(self):
        from fafirec.ratio_runner import DEFAULTS, fit
        cfg = {**DEFAULTS, 'd_model': 4, 'max_len': 5, 'n_heads': 2, 'n_layers': 1,
               'batch_size': 2, 'threads': 2, 'patience': 1, 'include_initial_checkpoint': True,
               'min_epochs': 3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run_model(model, out, epochs, values, resume=False):
                iterator = iter(values)
                with patch('fafirec.ratio_runner.evaluate', side_effect=lambda *a, **k: ([], {'all': {'ndcg50': next(iterator)}})):
                    with contextlib.redirect_stdout(io.StringIO()):
                        return fit(model, self.records, torch.zeros(2, 2), cfg, 17, out,
                                   'same', {}, epochs, resume, {'stage': 'validation_candidate'})
            torch.manual_seed(71)
            full = self.student()
            selected = run_model(full, root / 'full', 3, [.9, .1, .2, .95])
            self.assertEqual(selected['selected_epoch'], 3)
            self.assertEqual(selected['epochs_completed'], 3)
            torch.manual_seed(71)
            partial = self.student()
            run_model(partial, root / 'resumed', 1, [.9, .1])
            torch.manual_seed(999)
            resumed = self.student()
            restored = run_model(resumed, root / 'resumed', 3, [.2, .95], True)
            self.assertEqual(restored['selected_epoch'], 3)
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
