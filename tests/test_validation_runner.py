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
        self.reference_cfg = cfg
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
                       {"seeds": [17, 17]}, {"output_dir": self.search["reference_run_dir"]},
                       {"allow_partial_reference": 1}, {"lr": 0}, {"lr": True},
                       {"min_epochs": -1}, {"min_epochs": True}, {"min_epochs": 3}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                search_runner.normalize_config({**self.search, **update})
        for update in ({"name": "../escape"}, {"gate_lr_multiplier": 0}, {"cf_prior": 1.},
                       {"ridge": float("nan")}, {"unknown": 1}, {"lr": float("inf")},
                       {"lr": False}, {"min_epochs": 3}, {"min_epochs": .5}):
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

    def test_partial_reference_requires_opt_in_and_complete_requested_seed(self):
        cfg = {**self.reference_cfg, "output_dir": str(self.root / "partial_reference"), "seeds": [17, 18]}
        original_fit = ratio.fit

        def stop_second_seed(*args, **kwargs):
            artifact = kwargs.get("artifact", args[10] if len(args) > 10 else None)
            if artifact["seed"] == 18 and artifact["stage"] == "teacher":
                raise RuntimeError("requested stopping point")
            return original_fit(*args, **kwargs)

        with patch.object(ratio, "fit", side_effect=stop_second_seed), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "stopping point"):
            ratio.run(cfg)
        search = {**self.search, "reference_run_dir": cfg["output_dir"], "output_dir": str(self.root / "partial_search")}
        with self.assertRaisesRegex(ValueError, "complete configured grid"):
            search_runner.load_reference(search_runner.normalize_config(search))
        search["allow_partial_reference"] = True
        _, _, _, _, provenance, loaded = search_runner.load_reference(search_runner.normalize_config(search))
        self.assertTrue(provenance["partial_reference"])
        self.assertTrue(provenance["allow_partial_reference"])
        self.assertEqual(provenance["requested_seed_scope"]["seeds"], [17])
        self.assertEqual(set(loaded[17]["references"]), {"cf", "uniform", "diagnostic"})
        self.run_search(search)
        _, _, _, _, recorded = search_runner.load_validation_checkpoint(search["output_dir"], 17, "ridge_prior")
        self.assertTrue(recorded["reference"]["partial_reference"])
        with self.assertRaisesRegex(ValueError, "requested seed"):
            search_runner.load_reference(search_runner.normalize_config({**search, "seeds": [18]}))
        summary_path = Path(cfg["output_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for invalid in (summary + [summary[0]], summary + [{"seed": 999, "control": "cf"}]):
            summary_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate or undeclared"):
                search_runner.load_reference(search_runner.normalize_config(search))
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        cache_path = Path(cfg["output_dir"]) / "seed_17/diagnostics.pt"
        cache = torch.load(cache_path, weights_only=True)
        cache["features"][0, 0] += 1
        torch.save(cache, cache_path)
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            search_runner.load_reference(search_runner.normalize_config(search))

    def test_lr_and_min_epochs_are_inherited_overridden_and_bound_on_reload(self):
        variant = {**self.search["variants"][0], "lr": .0003, "min_epochs": 2}
        search = {**self.search, "lr": .0007, "min_epochs": 1, "variants": [variant]}
        normalized = search_runner.normalize_config(search)
        inherited = search_runner._variant_config(
            search_runner._training_config(self.reference_cfg, normalized),
            {**normalized["variants"][0], "lr": None, "min_epochs": None})
        self.assertEqual((inherited["lr"], inherited["min_epochs"]), (.0007, 1))
        self.run_search(search)
        _, _, _, cfg, manifest = search_runner.load_validation_checkpoint(search["output_dir"], 17, "ridge_prior")
        self.assertEqual((cfg["lr"], cfg["min_epochs"]), (.0003, 2))
        self.assertEqual((manifest["training_config"]["lr"], manifest["training_config"]["min_epochs"]), (.0007, 1))
        checkpoint = torch.load(Path(search["output_dir"]) / "seed_17/ridge_prior/last.pt", weights_only=True)
        self.assertEqual(checkpoint["artifact"]["optimization_configuration"]["min_epochs"], 2)
        self.assertEqual(checkpoint["artifact"]["optimization_configuration"]["lr"], .0003)
        self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], .0003)
        self.assertAlmostEqual(checkpoint["optimizer"]["param_groups"][1]["lr"], .003)
        with self.assertRaisesRegex(ValueError, "min_epochs"):
            search_runner._variant_config(self.reference_cfg, {"min_epochs": self.reference_cfg["epochs"] + 1})


if __name__ == "__main__":
    unittest.main()
