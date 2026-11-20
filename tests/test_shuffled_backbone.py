"""Initial identity isolation and exact continuation for the shuffled backbone."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from baselines.ace2026_model import build_ace_svd
from experiments.ablation_v5 import shuffled_backbone as shuffled
from experiments.diagnostic_v4 import coverage_probe as coverage
from fafirec.runner import seed_everything
from tools import ace2026_experiment as ace


class ShuffledBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.features = torch.randn(13, 8, generator=torch.Generator().manual_seed(50))
        self.features[0].zero_()
        self.factors = build_ace_svd(self.features)
        self.cfg = {'d_model': 8, 'max_len': 4, 'n_layers': 1, 'n_heads': 2,
            'inner_size': 16, 'dropout': .1, 'attn_dropout': .1,
            'layer_norm_eps': 1e-12, 'initializer_range': .02, 'ce_chunk_size': 8,
            'device': 'cpu', 'variants': [{'name': 'native', 'reg_weight': 500., 'scale': .1, 'lr': .001}]}

    def test_item_identity_changes_but_encoder_and_training_rng_are_exact(self):
        seed_everything(3025)
        native = ace.new_model(self.features, self.factors, self.cfg, self.cfg['variants'][0], torch.device('cpu'))
        state = torch.get_rng_state().clone()
        model, permutation, report = shuffled.make_model(self.features, self.factors, self.cfg, 2025)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(permutation[0].item(), 0)
        self.assertTrue(torch.equal(model.item_embedding.weight.cpu()[torch.argsort(permutation)], native.item_embedding.weight.cpu()))
        self.assertFalse(torch.equal(model.item_embedding.weight, native.item_embedding.weight))
        for key in native.state_dict():
            if key != 'item_embedding.weight':
                self.assertTrue(torch.equal(native.state_dict()[key], model.state_dict()[key]), key)
        self.assertTrue(report['nonitem_parameters_unchanged'])
        self.assertTrue(report['global_training_rng_unchanged'])
        self.assertTrue(report['inverse_recovers_native_table_exactly'])

    def test_permutation_preserves_mean_std_spectrum_and_is_seed_reproducible(self):
        first, p, report = shuffled.make_model(self.features, self.factors, self.cfg, 2025)
        second, q, same = shuffled.make_model(self.features, self.factors, self.cfg, 2025)
        self.assertTrue(torch.equal(p, q))
        self.assertEqual(report, same)
        self.assertAlmostEqual(report['native_nonpad_mean'], report['shuffled_nonpad_mean'], places=14)
        self.assertAlmostEqual(report['native_nonpad_population_std'], report['shuffled_nonpad_population_std'], places=14)
        self.assertLess(report['singular_spectrum_max_abs_difference'], 1e-11)
        self.assertTrue(first.item_embedding.weight[0].eq(0).all())
        _, other, _ = shuffled.make_model(self.features, self.factors, self.cfg, 2026)
        self.assertFalse(torch.equal(p, other))

    def test_nonzero_pad_is_rejected(self):
        model, _, _ = shuffled.make_model(self.features, self.factors, self.cfg, 2025)
        with torch.no_grad(): model.item_embedding.weight[0].fill_(1.)
        with self.assertRaisesRegex(ValueError, 'Zero PAD'):
            shuffled.shuffle_initial_items(model, 2025)

    def test_shuffled_training_epoch_resume_is_bit_identical(self):
        records = [{'user': str(i), 'group': 'middle', 'train': [1, 2, 3, 1, 4, 2, 1],
                    'validation': [5, 6]} for i in range(3)]
        cfg = {**self.cfg, 'weight_decay': 0., 'epochs': 3, 'min_epochs': 1,
            'eval_interval': 1, 'patience': 9, 'batch_size': 2, 'grad_clip': None}
        eval_cfg = {'max_len': 4, 'eval_batch': 2, 'ks': [10, 50], 'candidate_policy': 'exclude_seen'}
        manifest = {'fingerprint': 'shuffled-cpu-fixture'}
        def model():
            return shuffled.make_model(self.features, self.factors, cfg, 2025)[0]
        with tempfile.TemporaryDirectory() as temporary:
            full, interrupted = Path(temporary) / 'full', Path(temporary) / 'interrupted'
            coverage.fit(model(), records, cfg, eval_cfg, 2025, 'all_windows', full, manifest)
            dump = coverage.torch_dump
            def stop(path, value):
                dump(path, value)
                if Path(path).name == 'last.pt' and value['epoch'] == 1:
                    raise RuntimeError('saved epoch interruption')
            with patch.object(coverage, 'torch_dump', side_effect=stop), self.assertRaisesRegex(RuntimeError, 'interruption'):
                coverage.fit(model(), records, cfg, eval_cfg, 2025, 'all_windows', interrupted, manifest)
            coverage.fit(model(), records, cfg, eval_cfg, 2025, 'all_windows', interrupted, manifest, True)
            left, right = [torch.load(p / 'last.pt', weights_only=True) for p in (full, interrupted)]
            self.assertEqual(left['logs'], right['logs'])
            self.assertEqual(left['selected_epoch'], right['selected_epoch'])
            for key in left['model']:
                self.assertTrue(torch.equal(left['model'][key], right['model'][key]), key)
            for key in left['best_snapshot']['model']:
                self.assertTrue(torch.equal(left['best_snapshot']['model'][key], right['best_snapshot']['model'][key]), key)


if __name__ == '__main__':
    unittest.main()
