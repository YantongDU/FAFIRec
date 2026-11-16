import unittest

import numpy as np

from fafirec.metrics import (exact_target_ranks_from_chunks, metrics_from_ranks,
                             summarize_rows, target_ranks)


class MetricTests(unittest.TestCase):
    def test_rank_ten_and_eleven_boundary(self):
        hit, ndcg = metrics_from_ranks([1, 10, 11], 10)
        np.testing.assert_array_equal(hit, [1, 1, 0])
        np.testing.assert_allclose(ndcg, [1, 1 / np.log2(11), 0])

    def test_padding_ties_and_policies(self):
        scores = [[100, 5, 5, 5, 1]]
        self.assertEqual(target_ranks(scores, [3]).item(), 3)
        self.assertEqual(target_ranks(scores, [3], [[1, 2, 3]], "exclude_seen_restore_target").item(), 1)
        with self.assertRaisesRegex(ValueError, "repeated target"):
            target_ranks(scores, [3], [[3]], "exclude_seen")
        self.assertEqual(target_ranks(scores, [3], [[3]], "full").item(), 3)

    def test_chunks_equal_dense_under_all_policies(self):
        rng = np.random.default_rng(19)
        scores = rng.integers(-2, 3, size=(7, 24)).astype(float)
        targets = np.array([2, 8, 4, 18, 23, 10, 5])
        histories = [[1, 3, 7]] * len(targets)
        for policy in ("full", "exclude_seen", "exclude_seen_restore_target"):
            for chunk_size in (1, 3, 8, 25):
                chunks = [(np.arange(s, min(s + chunk_size, 24)), scores[:, s:s + chunk_size])
                          for s in range(0, 24, chunk_size)]
                ranks = exact_target_ranks_from_chunks(chunks, scores[np.arange(7), targets], targets,
                                                      histories, policy, n_items=23)
                np.testing.assert_array_equal(ranks, target_ranks(scores, targets, histories, policy))

    def test_chunk_catalog_integrity(self):
        with self.assertRaisesRegex(ValueError, "full contiguous catalog"):
            exact_target_ranks_from_chunks([(np.array([1, 3]), np.array([[1., 2.]]))], [1.], [1], n_items=3)
        with self.assertRaisesRegex(ValueError, "more than one"):
            exact_target_ranks_from_chunks([(np.array([1, 1]), np.ones((1, 2)))], [1.], [1])
        with self.assertRaisesRegex(ValueError, "differ between"):
            exact_target_ranks_from_chunks([(np.array([1]), np.array([[2.]]))], [1.], [1])
        # A tolerance-based equality check can change exact tie ordering.
        with self.assertRaisesRegex(ValueError, "differ between"):
            exact_target_ranks_from_chunks([(np.array([1]), np.array([[1. + 1e-9]]))], [1.], [1])

    def test_invalid_shapes_and_infinite_target_fail(self):
        for bad in (np.array([[0, 1, 0]]), [1.5], [0], [4]):
            with self.assertRaises(ValueError):
                target_ranks([[0, 1, 2, 3]], bad)
        with self.assertRaises(ValueError):
            target_ranks([[0, -np.inf]], [1])
        with self.assertRaises(ValueError):
            target_ranks([[0, np.nan]], [1])
        with self.assertRaises(ValueError):
            target_ranks([[0, 1]], [1], [[], []])

    def test_summary_weights_users_and_reports_single_positive_mrr(self):
        rows = [{'rank': 1, 'group': 'cold'}, {'rank': 2, 'group': 'middle'},
                {'rank': 11, 'group': 'middle'}]
        result = summarize_rows(rows, (10,))
        self.assertAlmostEqual(result['all']['hit10'], 2 / 3)
        self.assertAlmostEqual(result['all']['mrr10'], (1 + .5) / 3)
        self.assertEqual(result['cold']['users'], 1)


if __name__ == '__main__':
    unittest.main()
