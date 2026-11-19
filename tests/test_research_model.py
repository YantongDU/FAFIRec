"""Invariants for residual retrieval, frozen anchors, and causal diagnostics."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fafirec.ml1m_model import PaperFAFIRecModel
from fafirec.research_model import ResearchResidualModel


class ResearchModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.features = F.normalize(torch.randn(10, 6), dim=-1)
        self.features[0].zero_()
        self.records = [{'train': [1, 2, 3, 4, 5]}, {'train': [2, 4, 6, 7]}]
        self.anchor = PaperFAFIRecModel(n_items=9, text_features=self.features,
                                       d_model=4, max_len=5, n_heads=2, dropout=.35,
                                       control='cf').eval()
        self.hist = torch.tensor([[0, 1, 2, 3, 4], [2, 4, 6, 0, 0]])
        self.pos = torch.tensor([[0, 2, 3, 4, 5], [4, 6, 7, 0, 0]])
        self.neg = torch.tensor([[0, 8, 8, 8, 8], [9, 9, 9, 0, 0]])

    def make(self, **kwargs):
        return ResearchResidualModel(self.anchor, self.features, self.records,
                                     ce_chunk_size=3, seed=19, **kwargs)

    def test_zero_multiplier_exact_cf_and_pad(self):
        model = self.make().eval().set_calibration(0.)
        with torch.no_grad():
            model.gain_raw.fill_(.7)
            self.assertTrue(torch.equal(model.score_all(self.hist), self.anchor.score_all(self.hist)))
            self.assertEqual(model.user_vectors(self.hist).shape[-1], 4)
            self.assertTrue(torch.equal(model.item_vectors(), self.anchor.item_vectors()))
        model.set_calibration(1.)
        self.assertEqual(model.user_vectors(self.hist).shape[-1], 8)
        self.assertTrue(torch.equal(model.item_vectors()[0], torch.zeros(8)))

    def test_anchor_immutable_and_independent_expert(self):
        model = self.make()
        snapshot = {name: p.detach().clone() for name, p in model.anchor.named_parameters()}
        anchor_ptrs = {p.data_ptr() for p in model.anchor.parameters()}
        self.assertFalse(any(p.data_ptr() in anchor_ptrs for p in model.expert_encoder.parameters()))
        model.train()
        self.assertFalse(model.anchor.training)
        self.assertTrue(model.expert_encoder.training)
        self.assertFalse(any(p.requires_grad for p in model.anchor.parameters()))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            model(self.hist, self.pos, self.neg)['loss'].backward()
            optimizer.step()
        for name, p in model.anchor.named_parameters():
            self.assertTrue(torch.equal(snapshot[name], p), name)
            self.assertIsNone(p.grad)
        self.assertGreater(float(model.gain_raw.detach().abs()), 0.)

    def test_factorized_scores_and_frequency(self):
        model = self.make().eval()
        with torch.no_grad():
            model.gain_raw.fill_(.4)
        cf, semantic = model.branch_user_vectors(self.hist)
        ci, si = model.branch_item_vectors()
        for route in ('global', 'frequency'):
            model.set_calibration(.3, route)
            item_gate = model._item_gate()
            expected = model.score_scale * (cf @ ci.T) + model.expert_score_scale * .3 * torch.tanh(model.gain_raw) * (semantic @ (si * item_gate.unsqueeze(-1)).T)
            actual = model.score_all(self.hist)
            torch.testing.assert_close(actual[:, 1:], expected[:, 1:], atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.isneginf(actual[:, 0]).all())
        model.set_calibration(.3, 'relative')
        relative = model.score_all(self.hist)
        model.set_calibration(.3, 'global')
        torch.testing.assert_close(relative, model.score_all(self.hist), atol=1e-6, rtol=1e-6)

    def test_relative_features_are_strictly_causal(self):
        model = self.make().eval()
        hist = torch.tensor([[1, 2, 3, 4, 5]])
        changed_future = torch.tensor([[1, 2, 3, 9, 8]])
        a = model.relative_features(hist)
        b = model.relative_features(changed_future)
        self.assertFalse(a.requires_grad)
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(a[:, 0, :2], torch.zeros(1, 2)))
        torch.testing.assert_close(a[..., 3], torch.log1p(torch.arange(1, 6).float()).unsqueeze(0))
        model.set_calibration(1., 'relative')
        with torch.no_grad():
            model.router.weight.copy_(torch.tensor([[.2, .3, .4, .5]]))
        cf1, ex1, v1 = model._states(hist)
        cf2, ex2, v2 = model._states(changed_future)
        ga = model._user_gate(hist, cf1, ex1, v1)
        gb = model._user_gate(changed_future, cf2, ex2, v2)
        torch.testing.assert_close(ga[:, :3], gb[:, :3], atol=1e-6, rtol=1e-6)
        padded = model.relative_features(self.hist)
        self.assertTrue(torch.equal(padded[self.hist == 0], torch.zeros(3, 4)))

    def test_probe_uses_previous_query_and_observed_outcome(self):
        model = self.make().eval()
        cf, ex = model.branch_states(self.hist)
        cf_items, ex_items = model.branch_item_vectors()
        features = model.relative_features(self.hist, cf, ex)
        # At index 2 in the first row, item2 is known and query(item1) is past.
        item = self.hist[0, 2]
        rank = torch.tensor(2)
        offsets = torch.arange(model.probe_negatives)
        negative = (item * 1103515245 + rank * 12345 + model.seed + offsets * 2654435761).remainder(model.n_items - 1) + 1
        negative += (negative >= item).long()
        def loss(q, table):
            return F.softplus(-8 * (q * (table[item] - table[negative])).sum(-1)).mean()
        expected_gap = loss(cf[0, 1], cf_items) - loss(ex[0, 1], ex_items)
        torch.testing.assert_close(features[0, 2, 0], expected_gap)

    def test_unique_alignment_and_bpr_backward(self):
        model = self.make(alignment_weight=.01)
        with patch('fafirec.research_model.identity_infonce', return_value=model.gain_raw * 0. + .2) as alignment:
            parts = model(self.hist, self.pos, self.neg)
            parts['loss'].backward()
            self.assertEqual(set(parts), {'loss', 'rec_loss', 'aux_loss', 'align_loss'})
            self.assertEqual(alignment.call_count, 1)
            text, cf = alignment.call_args.args
            self.assertEqual(text.shape, (7, 4))
            self.assertFalse(cf.requires_grad)
        self.assertIsNotNone(model.gain_raw.grad)
        self.assertGreater(float(model.projection.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))

    def test_router_stage_has_only_router_gradients(self):
        model = self.make().eval()
        with torch.no_grad():
            model.gain_raw.fill_(.5)
        model.set_router_statistics([.1, .2, .3, .4], [1, 2, 3, 0])
        model.set_stage('router').train()
        self.assertFalse(model.anchor.training)
        self.assertFalse(model.expert_encoder.training)
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad], ['router.weight', 'router.bias'])
        parts = model(self.hist, self.pos, self.neg)
        parts['loss'].backward()
        self.assertGreater(float(model.router.weight.grad.norm()), 0.)
        self.assertEqual(float(parts['aux_loss']), 0.)
        self.assertEqual(float(parts['align_loss']), 0.)
        model.set_stage('expert')
        self.assertEqual(model.route, 'global')
        self.assertTrue(model.expert_encoder.training)

    def test_state_reload_and_shuffling(self):
        model = self.make(kind='shuffled').eval()
        with torch.no_grad():
            model.gain_raw.fill_(.2)
        rebuilt = ResearchResidualModel(copy.deepcopy(self.anchor), self.features, self.records, **model.model_config()).eval()
        rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(torch.equal(model.score_all(self.hist), rebuilt.score_all(self.hist)))
        self.assertTrue(torch.equal(model.feature_permutation, rebuilt.feature_permutation))
        self.assertEqual(int(model.feature_permutation[0]), 0)
        self.assertFalse(torch.equal(model.feature_permutation, torch.arange(10)))
        json.dumps(model.model_config())
        json.dumps(model.initialization_report)

    def test_ce_backward_and_id_control(self):
        for kind in ('text', 'id'):
            model = self.make(kind=kind, objective='ce')
            seen = torch.zeros(2, 10, dtype=torch.bool)
            for row, record in enumerate(self.records):
                seen[row, record['train']] = True
            parts = model(self.hist, self.pos, self.neg, seen)
            self.assertTrue(torch.isfinite(parts['loss']))
            parts['loss'].backward()
            self.assertIsNotNone(model.gain_raw.grad)
            table = model.projection.weight if kind == 'text' else model.id_table.weight
            self.assertGreater(float(table.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))


if __name__ == '__main__':
    unittest.main()
