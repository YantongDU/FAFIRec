import unittest
import numpy as np

from fafirec.certificates import pair_certificate, ranking_certificate


class CertificateTests(unittest.TestCase):
    def test_random_pair_bounds_and_certificates(self):
        rng = np.random.default_rng(29)
        count = 0
        for scale in (.0001, .02, .5, 2.):
            for _ in range(150):
                c, ci, cj = rng.normal(size=(3, 7))
                z, zi, zj = [v + scale * rng.normal(size=7) for v in (c, ci, cj)]
                result = pair_certificate(c, z, ci, cj, zi, zj)
                self.assertLessEqual(abs(result['fused_margin'] - result['reference_margin']),
                                     result['margin_change_bound'] + 1e-10)
                if result['certified']:
                    count += 1
                    self.assertTrue(result['actual_preserved'])
        self.assertGreater(count, 20)

    def test_random_catalog_certificate_has_no_false_positives(self):
        rng = np.random.default_rng(31)
        certified = 0
        for _ in range(80):
            c = rng.normal(size=5)
            ci = rng.normal(size=(24, 5))
            z = c + .001 * rng.normal(size=5)
            zi = ci + .001 * rng.normal(size=ci.shape)
            result = ranking_certificate(c, z, ci, zi, k=4, target=9)
            if result['certified_topk_set']:
                certified += 1
                self.assertTrue(result['actual_topk_set_preserved'])
            if result['certified_ordered_topk']:
                self.assertTrue(result['actual_ordered_topk_preserved'])
            self.assertLessEqual(result['maximum_observed_score_change'],
                                 result['maximum_score_change_bound'] + 1e-10)
        self.assertGreater(certified, 10)

    def test_set_preservation_does_not_establish_order_or_ndcg(self):
        c = z = np.array([1.])
        ci = np.array([[0.], [3.], [2.9], [-10.]])
        zi = np.array([[0.], [2.9], [3.], [-10.]])
        result = ranking_certificate(c, z, ci, zi, k=2, target=1)
        self.assertTrue(result['certified_topk_set'])
        self.assertFalse(result['certified_ordered_topk'])
        self.assertTrue(result['actual_topk_set_preserved'])
        self.assertFalse(result['actual_ordered_topk_preserved'])
        self.assertLess(result['ndcg_change'], 0.)

    def test_candidate_subset_and_ties_are_declared(self):
        result = ranking_certificate([1.], [1.], [[0.], [2.], [2.], [1.]],
                                     [[0.], [2.], [2.], [1.]], k=1, candidate_ids=[3, 2, 1])
        self.assertEqual(result['reference_topk'], [1])
        self.assertFalse(result['certified_topk_set'])
        self.assertTrue(result['actual_topk_set_preserved'])
        self.assertEqual(result['candidate_scope'], 'supplied_fixed_subset')
        with self.assertRaises(ValueError):
            ranking_certificate([1.], [1.], [[0.], [1.], [2.]], [[0.], [1.], [2.]], 2)


if __name__ == '__main__':
    unittest.main()
