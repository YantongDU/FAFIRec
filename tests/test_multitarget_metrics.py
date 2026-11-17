"""Manual binary-relevance examples and protocol guards for block evaluation."""
import unittest

import numpy as np

from fafirec.multitarget_metrics import evaluate_topk, summarize_multitarget_rows


class MultiTargetMetricTests(unittest.TestCase):
    def test_multiple_hits_use_recall_denominator_and_ideal_dcg(self):
        # Item ranking 1,2,3,4,5; relevant items occur at ranks1,3,5.
        scores = [[100., 5., 4., 3., 2., 1.]]
        row = evaluate_topk(scores, [[1, 3, 5]], ks=(3, 5), candidate_policy="full")[0]
        self.assertEqual(row["relevant_count"], 3)
        self.assertEqual(row["top_items"], [1, 2, 3, 4, 5])
        self.assertEqual(row["recall3"], 2 / 3)
        self.assertEqual(row["hit3"], 1.)
        self.assertAlmostEqual(row["ndcg3"], (1. + 1. / np.log2(4)) / (1. + 1. / np.log2(3) + 1. / np.log2(4)))
        self.assertEqual(row["recall5"], 1.)
        self.assertLess(row["ndcg5"], 1.)

    def test_ndcg_ideal_length_is_minimum_of_k_and_user_relevance(self):
        rows = evaluate_topk([[0, 5, 4, 3, 2, 1]] * 2, [[1, 2], [1, 2, 3, 4]],
                             ks=(2,), candidate_policy="full")
        self.assertEqual(rows[0]["recall2"], 1.)
        self.assertEqual(rows[1]["recall2"], .5)
        self.assertEqual(rows[0]["ndcg2"], 1.)
        self.assertEqual(rows[1]["ndcg2"], 1.)

    def test_stable_ties_exclude_padding_and_history_without_mutation(self):
        scores = np.array([[999., 5., 5., 5., 4., -np.inf]])
        original = scores.copy()
        row = evaluate_topk(scores, [{3, 4}], [[0, 1, 1]], ks=(3,))[0]
        self.assertEqual(row["top_items"], [2, 3, 4])
        self.assertEqual(row["recall3"], 1.)
        np.testing.assert_array_equal(scores, original)

    def test_topk_partition_selects_smaller_ids_at_tie_boundary(self):
        scores = [[0., 9., 5., 5., 5., 1.]]
        row = evaluate_topk(scores, [[4]], ks=(2,), candidate_policy="full")[0]
        self.assertEqual(row["top_items"], [1, 2])
        self.assertEqual(row["recall2"], 0.)

    def test_requested_k_beyond_candidate_count_does_not_return_masked_items(self):
        row = evaluate_topk([[0., 5., 4., 3., -np.inf]], [[3]], [[1]], ks=(10, 50))[0]
        self.assertEqual(row["top_items"], [2, 3])
        self.assertEqual(row["recall10"], 1.)
        self.assertAlmostEqual(row["ndcg50"], 1. / np.log2(3))

    def test_relevant_seen_item_is_an_error_not_silently_restored(self):
        with self.assertRaisesRegex(ValueError, "no target is restored"):
            evaluate_topk([[0., 3., 2., 1.]], [[2, 3]], [[1, 2]])
        row = evaluate_topk([[0., 3., 2., 1.]], [[2, 3]], [[1, 2]], ks=(2,), candidate_policy="full")[0]
        self.assertEqual(row["top_items"], [1, 2])
        self.assertEqual(row["recall2"], .5)

    def test_target_changes_do_not_change_candidate_membership_or_ranking(self):
        scores = [[0., 5., 4., 3., 2., 1.]]
        a = evaluate_topk(scores, [[3]], [[1]], ks=(3,))[0]
        b = evaluate_topk(scores, [[4, 5]], [[1]], ks=(3,))[0]
        self.assertEqual(a["top_items"], b["top_items"])
        self.assertEqual(a["top_items"], [2, 3, 4])

    def test_rank_ten_and_fifty_cutoffs_with_two_relevant_items(self):
        # Sorted item IDs are the exact rank order.
        scores = np.concatenate(([999.], np.arange(60., 0., -1)))[None, :]
        row = evaluate_topk(scores, [[10, 50]], ks=(10, 50), candidate_policy="full")[0]
        self.assertEqual(row["recall10"], .5)
        self.assertEqual(row["recall50"], 1.)
        ideal = 1. + 1. / np.log2(3)
        self.assertAlmostEqual(row["ndcg10"], (1. / np.log2(11)) / ideal)
        self.assertAlmostEqual(row["ndcg50"], (1. / np.log2(11) + 1. / np.log2(51)) / ideal)

    def test_invalid_targets_do_not_coerce_scalars_floats_bools_or_duplicates(self):
        scores = [[0., 3., 2., 1.]]
        for targets in ([2], [[1.5]], [[True]], [[0]], [[4]], [[]], [[1, 1]], [[[1, 2]]]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                evaluate_topk(scores, targets, candidate_policy="full")

    def test_bad_scores_histories_and_cutoffs_fail(self):
        for scores in ([[0., np.nan, 1.]], [[0., np.inf, 1.]], [[0., 1., -np.inf]]):
            with self.assertRaises(ValueError):
                evaluate_topk(scores, [[2]], candidate_policy="full")
        for history in (None, [], [[-1]], [[4]], [[True]], [1], [[1.5]]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                evaluate_topk([[0., 3., 2., 1.]], [[2]], history)
        for ks in ([], [0], [10.0], [True], 10):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                evaluate_topk([[0., 1.]], [[1]], ks=ks, candidate_policy="full")
        with self.assertRaises(ValueError):
            evaluate_topk([[0., 1.]], [[1]], candidate_policy="exclude_seen_restore_target")
        with self.assertRaises(ValueError):
            evaluate_topk([[0., 1.]], [[1], [1]], candidate_policy="full")

    def test_summary_is_user_macro_average_not_micro_recall_or_hit(self):
        # One relevant item for cold user, four for warm user, same top2.
        rows = evaluate_topk([[0, 5, 4, 3, 2, 1]] * 2, [[1], [1, 2, 3, 4]],
                             ks=(2,), candidate_policy="full")
        rows[0]["group"], rows[1]["group"] = "cold", "warm"
        summary = summarize_multitarget_rows(rows, ks=(2,))
        self.assertEqual(summary["all"]["users"], 2)
        self.assertEqual(summary["all"]["relevant_items"], 5)
        self.assertEqual(summary["all"]["recall2"], .75)
        self.assertNotEqual(summary["all"]["recall2"], 3 / 5)
        self.assertEqual(summary["all"]["hit2"], 1.)
        self.assertEqual(summary["cold"]["recall2"], 1.)
        self.assertEqual(summary["warm"]["recall2"], .5)

    def test_summary_without_groups_and_validation(self):
        rows = evaluate_topk([[0., 1.]], [[1]], ks=(10,), candidate_policy="full")
        self.assertEqual(set(summarize_multitarget_rows(rows, (10,))), {"all"})
        for change in ({"relevant_count": 0}, {"relevant_count": 1.5}, {"recall10": np.nan},
                       {"ndcg10": 1.1}, {"hit10": -1.}, {"group": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_multitarget_rows([{**rows[0], **change}], (10,))
        with self.assertRaises(ValueError):
            summarize_multitarget_rows([])


if __name__ == "__main__":
    unittest.main()
