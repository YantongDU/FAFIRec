"""Small CPU checks for a test-target-isolated, resumable transfer search."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import fafirec.ratio_runner as ratio
import fafirec.validation_runner as search_runner
from fafirec.data import file_sha256
from fafirec.runner import file_digest


class ValidationRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        data.mkdir()
        items = [{"id": i, "raw_id": str(i), "text": f"static movie {i}"} for i in range(1, 13)]
        records = [{"user": str(i), "train": [1 + i % 2, 3, 4, 2] * 4,
                    "validation": [5, 6], "test": [7, 8]} for i in range(6)]
        for name, values in (("items.jsonl", items), ("interactions.jsonl", records)):
            (data / name).write_text("\n".join(map(json.dumps, values)) + "\n", encoding="utf-8")
        manifest = {"format_version": 2, "split": {"ratios": [.8, .1, .1]},
                    "files": {name: file_sha256(data / name) for name in ("items.jsonl", "interactions.jsonl")}}
        (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cfg = {**ratio.DEFAULTS, "data_dir": str(data), "output_dir": str(self.root / "reference"),
            "feature_mode": "lexical", "embedding_path": None, "lexical_dim": 8,
            "seeds": [17], "controls": ["cf", "uniform", "diagnostic"],
            "epochs": 2, "teacher_epochs": 1, "patience": 4, "batch_size": 3, "eval_batch": 3,
            "d_model": 8, "max_len": 6, "n_heads": 2, "n_layers": 1, "dropout": .1,
            "device": "cpu", "threads": 2}
        with contextlib.redirect_stdout(io.StringIO()):
            ratio.run(cfg)
        self.search = {"reference_run_dir": cfg["output_dir"], "output_dir": str(self.root / "search"),
            "seeds": [17], "epochs": 2, "patience": 4, "device": "cpu", "threads": 2,
            "variants": [{"name": "ridge_prior", "control": "diagnostic", "projection_init": "ridge",
                          "cf_prior": .9, "ridge": 1., "align_weight": .03, "gate_lr_multiplier": 10.}]}

    def tearDown(self):
        self.temporary.cleanup()

    def run_search(self, cfg=None, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return search_runner.run(cfg or self.search, resume)

    def test_search_never_exposes_test_targets_and_replays_validation(self):
        original = ratio.evaluate
        calls = []

        def validation_only(model, records, split, *args, **kwargs):
            self.assertEqual(split, "validation")
            self.assertTrue(all("test" not in record for record in records))
            calls.append(split)
            return original(model, records, split, *args, **kwargs)

        with patch.object(ratio, "evaluate", side_effect=validation_only), \
                patch.object(search_runner, "evaluate", side_effect=validation_only):
            selected = self.run_search()
        self.assertGreaterEqual(len(calls), 5)
        self.assertEqual(selected["selected_variant"], "ridge_prior")
        self.assertEqual(selected["split"], "validation")
        self.assertEqual({r["variant"] for r in selected["references"]}, {"cf", "uniform", "diagnostic"})
        root = Path(self.search["output_dir"])
        model, records, diag, cfg, manifest = search_runner.load_validation_checkpoint(root, 17, "ridge_prior")
        self.assertTrue(cfg["include_initial_checkpoint"])
        self.assertTrue(all("test" in record for record in records))
        rows, metrics = original(model, records, "validation", diag, cfg, 17)
        saved = json.loads((root / "seed_17/ridge_prior_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics, saved["validation"])
        replay = root / "replayed.csv"
        ratio.write_rows(replay, rows)
        self.assertEqual(file_digest(replay), file_digest(root / "seed_17/ridge_prior_validation.csv"))
        self.assertIn(saved["selected_epoch"], (0, 1, 2))
        self.assertFalse(manifest["test_evaluated_by_training_entrypoint"])
        before = (root / "seed_17/ridge_prior_validation.csv").read_bytes()
        self.assertEqual(self.run_search(resume=True), selected)
        self.assertEqual((root / "seed_17/ridge_prior_validation.csv").read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.run_search()

    def test_interrupted_epoch_resume_matches_complete_training(self):
        self.run_search()
        full_path = Path(self.search["output_dir"]) / "seed_17/ridge_prior/best.pt"
        partial_cfg = {**self.search, "output_dir": str(self.root / "interrupted")}
        original = search_runner.fit

        def interrupt(*args, **kwargs):
            arguments = list(args)
            arguments[8] = 1  # end at an epoch boundary with the same target config/fingerprint
            original(*arguments, **kwargs)
            raise RuntimeError("simulated interruption")

        with patch.object(search_runner, "fit", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "simulated"):
            self.run_search(partial_cfg)
        self.run_search(partial_cfg, resume=True)
        resumed = torch.load(Path(partial_cfg["output_dir"]) / "seed_17/ridge_prior/best.pt", weights_only=True)
        full = torch.load(full_path, weights_only=True)
        self.assertEqual(full["selected_epoch"], resumed["selected_epoch"])
        for key, value in full["model"].items():
            torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)

    def test_configuration_reference_and_artifact_guards(self):
        for update in ({"candidate_policy": "full"}, {"d_model": 16}, {"epochs": True},
                       {"seeds": [17, 17]}, {"output_dir": self.search["reference_run_dir"]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                search_runner.normalize_config({**self.search, **update})
        for update in ({"name": "../escape"}, {"gate_lr_multiplier": 0}, {"cf_prior": 1.},
                       {"ridge": float("nan")}, {"unknown": 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                search_runner.normalize_config({**self.search, "variants": [{**self.search["variants"][0], **update}]})
        with self.assertRaisesRegex(ValueError, "absent"):
            self.run_search({**self.search, "seeds": [999]})
        self.run_search()
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.run_search({**self.search, "epochs": 3}, True)
        root = Path(self.search["output_dir"])
        with self.assertRaisesRegex(ValueError, "Variant"):
            search_runner.load_validation_checkpoint(root, 17, "absent")
        path = root / "seed_17/ridge_prior_validation.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            search_runner.load_validation_checkpoint(root, 17, "ridge_prior")
        reference = Path(self.search["reference_run_dir"])
        cache_path = reference / "seed_17/diagnostics.pt"
        cache = torch.load(cache_path, weights_only=True)
        cache["features"][0, 0] += 1
        torch.save(cache, cache_path)
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            search_runner.load_reference(search_runner.normalize_config(self.search))


if __name__ == "__main__":
    unittest.main()
