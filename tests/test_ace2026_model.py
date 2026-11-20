"""CPU checks for ACE geometry, raw-dot supervision, and sequence semantics."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from baselines.ace2026_model import ACE2026Model, ace_item_initialization, build_ace_svd, catalog_ce


ROOT = Path(__file__).resolve().parents[1]


class ACE2026ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(101)
        self.features = torch.randn(13, 8, dtype=torch.float64)
        self.features[0].zero_()
        # Give the matrix a substantial nonzero mean, so accidental centering fails.
        self.features[1:] += 2.
        self.factors = build_ace_svd(self.features)

    def model(self, **kwargs):
        values = dict(d_model=4, max_len=6, n_layers=2, n_heads=2, inner_size=7,
                      dropout=0., attn_dropout=0., reg_weight=1.5, scale=.1,
                      ce_chunk_size=3, svd_factors=self.factors)
        values.update(kwargs)
        features = values.pop('text_features', self.features)
        return ACE2026Model(features, **values)

    def test_spectral_geometry_scale_and_untouched_initial_table(self):
        table, report = ace_item_initialization(self.features, 4, 1.5, .1, self.factors, torch.float64)
        matrix = self.features[1:].numpy()
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        expected = u[:, :4] * np.sqrt(singular[:4] ** 2 / (singular[:4] ** 2 + 1.5))
        expected *= .1 / (expected.std() + 1e-12)
        np.testing.assert_allclose(table[1:].numpy() @ table[1:].numpy().T, expected @ expected.T,
                                   rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(float(table[1:].std(unbiased=False)), .1, places=10)
        model = self.model()
        torch.testing.assert_close(model.item_embedding.weight, table.float(), rtol=0, atol=0)
        self.assertFalse(report['centered'])
        self.assertEqual(model.score_scale, 1.)
        self.assertFalse(torch.allclose(model.item_vectors()[1:].norm(dim=1), torch.ones(12)))
        larger = self.model(scale=.5)
        torch.testing.assert_close(larger.item_vectors()[1:], model.item_vectors()[1:] * 5., rtol=1e-6, atol=1e-7)

    def test_lambda_zero_projection_gram_and_degenerate_input(self):
        table, _ = ace_item_initialization(self.features, 8, 0., 1., self.factors, torch.float64)
        # Undo the documented global scale to recover the orthogonal projector.
        u = self.factors['u'].numpy()
        expected = u @ u.T
        base = u / (u.std() + 1e-12)
        np.testing.assert_allclose(table[1:].numpy() @ table[1:].numpy().T, base @ base.T,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(expected @ expected, expected, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'no nonzero'):
            ace_item_initialization(torch.zeros_like(self.features), 4, 0., .1)

    def test_cache_guards_and_no_random_fallback(self):
        changed = self.features.clone()
        changed[2, 3] += .01
        with self.assertRaisesRegex(ValueError, 'different features'):
            self.model(text_features=changed)
        bad = copy.deepcopy(self.factors)
        bad['u'][0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'factor hash'):
            self.model(svd_factors=bad)
        with self.assertRaises(ValueError):
            self.model(reg_weight=-1)
        with self.assertRaises(ValueError):
            self.model(d_model=16)

    def test_padding_layout_empty_histories_and_causal_states(self):
        model = self.model().eval()
        left = torch.tensor([[0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 5, 6]])
        right = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]])
        torch.testing.assert_close(model.user_vectors(left), model.user_vectors(right), rtol=0, atol=0)
        left_states, right_states = model.encode_states(left), model.encode_states(right)
        for row in range(2):
            torch.testing.assert_close(left_states[row][left[row].ne(0)], right_states[row][right[row].ne(0)], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(left_states[left.eq(0)])), 0)
        empty = torch.zeros(2, 6, dtype=torch.long)
        self.assertEqual(int(torch.count_nonzero(model.user_vectors(empty))), 0)
        self.assertTrue(torch.isfinite(model.encode_states(empty)).all())
        changed = left.clone()
        changed[:, -1] = torch.tensor([7, 8])
        torch.testing.assert_close(model.encode_states(left)[:, :-1], model.encode_states(changed)[:, :-1], rtol=0, atol=0)

    def test_raw_catalog_ce_dense_value_and_gradients(self):
        queries = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        items = torch.randn(13, 4, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1, 2, 3, 2, 7, 8, 9])
        dense = F.cross_entropy(queries @ items.T, targets)
        actual = catalog_ce(queries, items, targets, chunk_size=3)
        dense_grad = torch.autograd.grad(dense, (queries, items), retain_graph=True)
        actual_grad = torch.autograd.grad(actual, (queries, items))
        torch.testing.assert_close(actual, dense, rtol=1e-12, atol=1e-12)
        for value, expected in zip(actual_grad, dense_grad):
            torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
        # PAD and other observed items remain classes in this objective.
        self.assertGreater(float(dense_grad[1][0].abs().sum()), 0)
        without_pad = F.cross_entropy((queries @ items.T)[:, 1:], targets - 1)
        self.assertGreater(abs(float(actual - without_pad)), 1e-5)

    def test_training_pad_gradient_zero_and_encoder_updates(self):
        model = self.model().train()
        hist = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]])
        positive = torch.tensor([[0, 2, 3, 4], [0, 0, 5, 6]])
        initial = model.item_embedding.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
        parts = model.loss_batch(hist, positive, torch.zeros_like(hist))
        parts['loss'].backward()
        self.assertEqual(float(parts['reg_loss']), 0.)
        self.assertEqual(float(model.item_embedding.weight.grad[0].abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertEqual(float(model.item_embedding.weight[0].abs().sum()), 0.)
        self.assertFalse(torch.equal(model.item_embedding.weight[1:], initial[1:]))

    def test_reconstruction_config_and_state(self):
        model = self.model().eval()
        rebuilt = ACE2026Model(self.features, **model.model_config(), svd_factors=self.factors)
        rebuilt.load_state_dict(model.state_dict())
        rebuilt.eval()
        hist = torch.tensor([[0, 1, 2, 3]])
        torch.testing.assert_close(model.user_vectors(hist), rebuilt.user_vectors(hist), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.loss_batch(hist, torch.tensor([[0, 2, 3, 15]]))

    def test_native_recbole_encoder_golden_forward_if_reference_available(self):
        # Read the locally pinned author reference only. No upstream source is
        # included in this test/repository, and its math is not mocked.
        reference = ROOT / 'data/ACE_reference/recbole'
        if not (reference / 'model/layers.py').is_file():
            self.skipTest('Pinned local author reference is not bundled')
        spec = importlib.util.spec_from_file_location('_ace_reference_enums', reference / 'utils/enum_type.py')
        enums = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enums)
        package, utils = types.ModuleType('recbole'), types.ModuleType('recbole.utils')
        utils.FeatureType, utils.FeatureSource = enums.FeatureType, enums.FeatureSource
        layer_spec = importlib.util.spec_from_file_location('_ace_reference_layers', reference / 'model/layers.py')
        layers = importlib.util.module_from_spec(layer_spec)
        with patch.dict(sys.modules, {'recbole': package, 'recbole.utils': utils}):
            layer_spec.loader.exec_module(layers)
        native = layers.TransformerEncoder(n_layers=2, n_heads=2, hidden_size=4, inner_size=7,
            hidden_dropout_prob=0., attn_dropout_prob=0., hidden_act='gelu', layer_norm_eps=1e-12).double().eval()
        model = self.model().double().eval()
        with torch.no_grad():
            for block, original in zip(model.blocks, native.layer):
                attn = original.multi_head_attention
                block.attention.in_proj_weight.copy_(torch.cat([attn.query.weight, attn.key.weight, attn.value.weight]))
                block.attention.in_proj_bias.copy_(torch.cat([attn.query.bias, attn.key.bias, attn.value.bias]))
                block.attention.out_proj.load_state_dict(attn.dense.state_dict())
                block.attention_norm.load_state_dict(attn.LayerNorm.state_dict())
                block.expand.load_state_dict(original.feed_forward.dense_1.state_dict())
                block.contract.load_state_dict(original.feed_forward.dense_2.state_dict())
                block.ff_norm.load_state_dict(original.feed_forward.LayerNorm.state_dict())
        hist = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]])
        length = hist.ne(0).sum(1)
        values = model.input_norm(model.item_embedding(hist) + model.position_embedding(torch.arange(6))[None])
        allowed = torch.ones(6, 6, dtype=torch.bool).tril()[None] & hist.ne(0)[:, None]
        bias = torch.zeros(2, 1, 6, 6, dtype=torch.float64).masked_fill(~allowed[:, None], -10000.)
        expected = native(values, bias, output_all_encoded_layers=False)[-1]
        expected = expected[torch.arange(2), length - 1]
        torch.testing.assert_close(model.user_vectors(hist), expected, rtol=1e-11, atol=1e-11)


if __name__ == '__main__':
    unittest.main()
