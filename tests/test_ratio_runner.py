"""End-to-end ratio protocol, optimizer, checkpoint and multi-target checks."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from analyze_ml1m import analyze_runs
from fafirec.data import file_sha256
from fafirec.ml1m_model import PaperFAFIRecModel
from fafirec.ratio_runner import (DEFAULTS, evaluate, fit, load_config,
                                  load_selected, model_config, run,
                                  seed_everything)


class RatioRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        items = [{"id": item, "raw_id": str(item), "text": f"static movie {item}"} for item in range(1, 13)]
        self.records = [{"user": str(index), "train": [1 + index % 2, 3, 4, 2] * 4,
                         "validation": [5, 6], "test": [7, 8], "group": "middle"} for index in range(6)]
        for name, values in (("items.jsonl", items), ("interactions.jsonl", self.records)):
            (self.data / name).write_text("\n".join(map(json.dumps, values)) + "\n", encoding="utf-8")
        manifest = {"format_version": 2, "split": {"ratios": [.8, .1, .1]},
                    "files": {name: file_sha256(self.data / name) for name in ("items.jsonl", "interactions.jsonl")}}
        (self.data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.cfg = {**DEFAULTS, "data_dir": str(self.data), "output_dir": str(self.root / "run"),
                    "feature_mode": "lexical", "embedding_path": None, "lexical_dim": 8,
                    "seeds": [17], "controls": ["uniform", "diagnostic"],
                    "epochs": 2, "teacher_epochs": 1, "patience": 4,
                    "batch_size": 3, "eval_batch": 3, "d_model": 8, "max_len": 6,
                    "n_layers": 1, "n_heads": 2, "dropout": .1, "device": "cpu", "threads": 2}
        seed_everything(17)
        self.features = torch.randn(13, 5)
        self.features[0].zero_()

    def tearDown(self):
        self.temporary.cleanup()

    def model(self):
        return PaperFAFIRecModel(text_features=self.features, **model_config(self.cfg, "diagnostic", 12))

    def test_paper_defaults_and_adam_contract_are_explicit(self):
        self.assertEqual((DEFAULTS["d_model"], DEFAULTS["batch_size"], DEFAULTS["selection_k"]), (64, 2048, 50))
        self.assertEqual((DEFAULTS["optimizer"], DEFAULTS["weight_decay"], DEFAULTS["epochs"]), ("adam", 1e-5, 150))
        self.assertEqual(DEFAULTS["align_weight"], .3)
        path = self.root / "config.json"
        path.write_text(json.dumps({"optimizer": "adamw"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Adam"):
            load_config(path)
        for invalid in (-1., float("nan"), float("inf"), True, "0.001"):
            path.write_text(json.dumps({"weight_decay": invalid}), encoding="utf-8")
            with self.subTest(weight_decay=invalid), self.assertRaisesRegex(ValueError, "weight_decay"):
                load_config(path)
        for valid in (0., 1e-4):
            path.write_text(json.dumps({"weight_decay": valid}), encoding="utf-8")
            self.assertEqual(load_config(path)["weight_decay"], valid)

    def test_epoch_resume_is_exact_and_uses_adam_not_adamw(self):
        diag = torch.randn(6, 2)
        with patch("torch.optim.AdamW", side_effect=AssertionError("AdamW must not be used")):
            with contextlib.redirect_stdout(io.StringIO()):
                seed_everything(40)
                full = self.model()
                fit(full, self.records, diag, self.cfg, 20, self.root / "full", "same", {}, 2)
                seed_everything(40)
                partial = self.model()
                fit(partial, self.records, diag, self.cfg, 20, self.root / "partial", "same", {}, 1)
                seed_everything(999)
                restored = self.model()
                fit(restored, self.records, diag, self.cfg, 20, self.root / "partial", "same", {}, 2, True)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        state = torch.load(self.root / "full/last.pt", weights_only=True)
        self.assertEqual(state["optimizer_name"], "Adam")
        self.assertEqual(state["optimizer"]["param_groups"][0]["weight_decay"], self.cfg["weight_decay"])

    def test_ratio_run_reload_metrics_analysis_and_completed_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run(self.cfg)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["selection_k"] == 50 for result in results))
        model, records, diag, cfg, _ = load_selected(self.root / "run", 17, "diagnostic", "cpu")
        rows, metrics = evaluate(model, records, "test", diag, cfg, 17)
        self.assertEqual(metrics, results[-1]["test"])
        self.assertTrue(all(row["target_count"] == 2 and json.loads(row["targets"]) == [7, 8] for row in rows))
        self.assertTrue(all("recall10" in row and "recall50" in row and "rank" not in row for row in rows))
        best = torch.load(self.root / "run/seed_17/diagnostic/best.pt", weights_only=True)
        self.assertEqual(best["selection_k"], 50)
        self.assertEqual(best["selected_metric"], results[-1]["selected_validation_ndcg50"])
        report = analyze_runs(self.root / "run")
        self.assertEqual(report["provenance"]["job_count"], 2)
        self.assertTrue(all(value["sample_std"] is None for value in report["aggregates"]))
        self.assertTrue(all(value["bootstrap_95_ci"] is None for value in report["paired_comparisons"]))
        before = (self.root / "run/seed_17/diagnostic_per_user.csv").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(self.cfg, True), results)
        self.assertEqual((self.root / "run/seed_17/diagnostic_per_user.csv").read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            run({**self.cfg, "lr": .002}, True)
        with self.assertRaises(FileExistsError):
            run(self.cfg)
        summary = self.root / "run/summary.json"
        original = summary.read_bytes()
        summary.write_text(json.dumps(results[:1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Complete configured"):
            analyze_runs(self.root / "run")
        summary.write_bytes(original)
        path = self.root / "run/seed_17/diagnostic_per_user.csv"
        path.write_bytes(before + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            analyze_runs(self.root / "run")


if __name__ == "__main__":
    unittest.main()
