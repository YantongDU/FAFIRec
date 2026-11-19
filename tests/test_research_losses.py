import unittest

import torch
from torch.nn import functional as F

from fafirec.research_losses import identity_infonce, masked_catalog_ce


class ResearchLossTests(unittest.TestCase):
    def test_chunked_ce_values_and_gradients_match_exact_dense_objective(self):
        torch.manual_seed(71)
        q = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
        v = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 1, 2])
        owners = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        seen = torch.zeros(3, 8, dtype=torch.bool)
        seen[0, [1, 2, 3, 4]] = True
        seen[1, [4, 5]] = True
        seen[2, [7, 1]] = True
        initial = seen.clone()
        blocked = seen[owners].clone()
        blocked[:, 0] = True
        blocked.scatter_(1, y[:, None], False)
        dense = F.cross_entropy((3. * q @ v.T).masked_fill(blocked, float('-inf')), y)
        expected = torch.autograd.grad(dense, (q, v))
        chunked = masked_catalog_ce(q, v, y, owners, seen, 3., 2)
        actual = torch.autograd.grad(chunked, (q, v))
        torch.testing.assert_close(chunked, dense, rtol=1e-12, atol=1e-12)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(seen, initial, rtol=0, atol=0)
        self.assertEqual(float(actual[1][0].abs().sum()), 0.)

    def test_known_positive_exclusion_and_current_target_restore_are_exact(self):
        q = torch.tensor([[1., 0.]])
        items = torch.tensor([[999., 0.], [2., 0.], [999., 0.], [0., 0.]])
        seen = torch.tensor([[False, True, True, False]])
        loss = masked_catalog_ce(q, items, torch.tensor([1]), torch.tensor([0]), seen)
        torch.testing.assert_close(loss, F.softplus(torch.tensor(-2.)))

    def test_identity_alignment_stops_cf_gradient_and_preserves_correspondence(self):
        x = torch.eye(4, requires_grad=True)
        target = torch.eye(4, requires_grad=True)
        paired = identity_infonce(x, target, .1)
        wrong = identity_infonce(x, target.roll(1, 0), .1)
        self.assertLess(float(paired), float(wrong))
        paired.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(target.grad)

    def test_invalid_indices_and_empty_rows_are_safe(self):
        q, items = torch.randn(1, 3), torch.randn(4, 3)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        for targets, owners in ((torch.tensor([0]), torch.tensor([0])), (torch.tensor([1]), torch.tensor([1]))):
            with self.assertRaises(ValueError):
                masked_catalog_ce(q, items, targets, owners, mask)
        empty = torch.empty(0, 3, requires_grad=True)
        loss = masked_catalog_ce(empty, items, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), mask)
        loss.backward()
        self.assertEqual(float(loss), 0.)


if __name__ == '__main__':
    unittest.main()
