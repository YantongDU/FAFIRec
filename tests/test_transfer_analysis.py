"""Output auditing catches metric-only and prepared-target inconsistencies."""
import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import tests.test_validation_runner as fixtures
from analyze_transfer import analyze_runs, write_analysis
from fafirec.runner import file_digest


class TransferAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ValidationRunnerTests()
        self.fixture.setUp()
        self.fixture.run_search()
        self.root = Path(self.fixture.search["output_dir"])

    def tearDown(self):
        self.fixture.tearDown()

    def read(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def write(self, relative, value):
        (self.root / relative).write_text(json.dumps(value), encoding="utf-8")

    def change_job(self, edit):
        saved = self.read("seed_17/ridge_prior_summary.json")
        edit(saved)
        self.write("seed_17/ridge_prior_summary.json", saved)
        self.write("summary.json", [saved])

    def test_complete_analysis_recomputes_metrics_without_model_evaluation(self):
        with patch("fafirec.ratio_runner.evaluate", side_effect=AssertionError("No scoring in analysis")), \
                patch("fafirec.validation_runner.evaluate", side_effect=AssertionError("No scoring in analysis")):
            report = analyze_runs(self.root, "ridge_prior", "reference:cf", bootstrap_samples=100)
        self.assertEqual(report["selected_variant"], "ridge_prior")
        self.assertEqual(report["provenance"]["job_count"], 1)
        self.assertTrue(report["provenance"]["candidate_metrics_recomputed_from_rankings"])
        self.assertTrue(all(row["sample_std"] is None for row in report["aggregates"]))
        self.assertTrue(all(row["bootstrap_95_ci"] is None for row in report["paired_comparisons"]))
        saved = self.read("seed_17/ridge_prior_summary.json")
        row = next(row for row in report["aggregates"] if row["role"] == "candidate"
                   and row["group"] == "all" and row["metric"] == "ndcg10")
        self.assertAlmostEqual(row["mean"], saved["validation"]["all"]["ndcg10"])
        write_analysis(report, self.root / "analysis")
        self.assertTrue((self.root / "analysis/paper_table.csv").is_file())
        self.assertTrue((self.root / "analysis/aggregate_metrics.csv").is_file())
        self.assertEqual(self.read("analysis/analysis.json")["split"], "validation")

    def test_metric_only_summary_and_selection_tampering_are_rejected(self):
        original_job = self.read("seed_17/ridge_prior_summary.json")
        self.change_job(lambda saved: saved["validation"]["all"].__setitem__(
            "ndcg10", saved["validation"]["all"]["ndcg10"] + .001))
        with self.assertRaisesRegex(ValueError, "Validation summary"):
            analyze_runs(self.root)
        self.write("seed_17/ridge_prior_summary.json", original_job)
        self.write("summary.json", [original_job])
        self.change_job(lambda saved: saved["optimization_configuration"].__setitem__("lr", .123))
        with self.assertRaisesRegex(ValueError, "Effective optimization configuration"):
            analyze_runs(self.root)
        self.write("seed_17/ridge_prior_summary.json", original_job)
        self.write("summary.json", [original_job])
        selection = self.read("selection.json")
        selection["candidates"][0]["metrics"]["all"]["ndcg50"]["mean"] += .001
        self.write("selection.json", selection)
        with self.assertRaisesRegex(ValueError, "Selection summary"):
            analyze_runs(self.root)

    def test_csv_checked_against_prepared_records_even_with_updated_hash(self):
        path = self.root / "seed_17/ridge_prior_validation.csv"
        original = path.read_bytes()
        for field, change, expected in (
                ("training_interactions", lambda _: "999", "training count"),
                ("targets", lambda _: "[9, 10]", "prepared validation targets"),
                ("group", lambda _: "warm", "training-count group"),
                ("top_items", lambda _: "[1, 2, 3, 4, 5, 6, 7, 8]", "training-observed")):
            with self.subTest(field=field):
                path.write_bytes(original)
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    rows, fields = list(reader), reader.fieldnames
                rows[0][field] = change(rows[0][field])
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                self.change_job(lambda saved: saved.__setitem__("per_user_sha256", file_digest(path)))
                with self.assertRaisesRegex(ValueError, expected):
                    analyze_runs(self.root)

    def test_incomplete_duplicate_or_extra_job_grid_is_rejected(self):
        summary = self.read("summary.json")
        for bad in ([], summary + summary):
            self.write("summary.json", bad)
            with self.assertRaisesRegex(ValueError, "Complete configured grid"):
                analyze_runs(self.root)
        self.write("summary.json", summary)
        extra = self.root / "seed_999"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "extra validation job"):
            analyze_runs(self.root)
        (extra / "other_validation.csv").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra validation job"):
            analyze_runs(self.root)


if __name__ == "__main__":
    unittest.main()
