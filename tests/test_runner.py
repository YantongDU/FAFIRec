"""Integration checks for replayable optimization, checkpoint reload, and ranks."""
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from evaluate import load_selected
from fafirec.data import evaluation_batch, file_sha256
from fafirec.metrics import target_ranks
from fafirec.model import FAFIRecModel
from fafirec.runner import (DEFAULTS, evaluate, fit, model_config, run,
                            seed_everything)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        items = [{"id": i, "raw_id": str(i), "text": f"item text {i}"} for i in range(1, 13)]
        self.records = [{"user": str(i), "train": [1 + i % 2, 3, 4, 5],
                         "validation": 6, "test": 7, "group": "middle"} for i in range(6)]
        (self.data / "items.jsonl").write_text("\n".join(map(json.dumps, items)) + "\n", encoding="utf-8")
        (self.data / "interactions.jsonl").write_text("\n".join(map(json.dumps, self.records)) + "\n", encoding="utf-8")
        (self.data / "manifest.json").write_text(json.dumps({"format_version": 1,
            "files": {n: file_sha256(self.data / n) for n in ["items.jsonl", "interactions.jsonl"]}}), encoding="utf-8")
        self.cfg = {**DEFAULTS, "data_dir": str(self.data), "output_dir": str(self.root / "run"),
                    "feature_mode": "lexical", "seeds": [17], "controls": ["uniform", "diagnostic"],
                    "epochs": 2, "teacher_epochs": 1, "batch_size": 3, "eval_batch": 3,
                    "item_chunk_size": 4, "d_model": 8, "n_layers": 1, "n_heads": 2,
                    "max_len": 6, "dropout": .1, "patience": 4, "threads": 2, "device": "cpu"}
        seed_everything(17)
        self.features = torch.randn(13, 5)
        self.features[0].zero_()

    def tearDown(self):
        self.temporary.cleanup()

    def model(self):
        return FAFIRecModel(text_features=self.features, **model_config(self.cfg, "diagnostic", 12))

    def test_epoch_boundary_resume_matches_uninterrupted_weights(self):
        diag = torch.randn(6, 2)
        seed_everything(50)
        full = self.model()
        with contextlib.redirect_stdout(io.StringIO()):
            fit(full, self.records, diag, self.cfg, 30, self.root / "full", "same", {}, 2, False)
            seed_everything(50)
            partial = self.model()
            fit(partial, self.records, diag, self.cfg, 30, self.root / "partial", "same", {}, 1, False)
            seed_everything(999)
            restored = self.model()
            fit(restored, self.records, diag, self.cfg, 30, self.root / "partial", "same", {}, 2, True)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        a = torch.load(self.root / "full/last.pt", weights_only=True)
        b = torch.load(self.root / "partial/last.pt", weights_only=True)
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)

    def test_chunked_evaluation_matches_dense_catalog_ranks(self):
        model = self.model().eval()
        diag = torch.randn(6, 2)
        rows, _ = evaluate(model, self.records, "test", diag, self.cfg, 17)
        hist, targets = evaluation_batch(self.records, range(6), "test", self.cfg["max_len"])
        with torch.no_grad():
            dense = model.score_all(hist, diag).numpy()
        self.assertEqual([r["rank"] for r in rows], target_ranks(dense, targets.numpy()).tolist())

    def test_run_reload_and_resume_reproduce_selected_predictions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run(self.cfg)
        path = self.root / "run/seed_17/diagnostic_per_user.csv"
        before = path.read_bytes()
        model, records, diag, cfg, _ = load_selected(self.root / "run", 17, "diagnostic", "cpu")
        rows, summary = evaluate(model, records, "test", diag, cfg, 17)
        self.assertEqual(summary, results[-1]["test"])
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = run(self.cfg, resume=True)
        self.assertEqual(results, resumed)
        self.assertEqual(before, path.read_bytes())
        with self.assertRaises(FileExistsError):
            run(self.cfg)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            run({**self.cfg, "lr": .002}, resume=True)
        checkpoint = self.root / "run/seed_17/diagnostic/best.pt"
        original = checkpoint.read_bytes()
        checkpoint.write_bytes((self.root / "run/seed_17/uniform/best.pt").read_bytes())
        with self.assertRaisesRegex(ValueError, "seed, control"):
            load_selected(self.root / "run", 17, "diagnostic", "cpu")
        checkpoint.write_bytes(original)
        changed = torch.load(checkpoint, weights_only=True)
        changed["model"]["control_bias"] += .1
        torch.save(changed, checkpoint)
        with self.assertRaisesRegex(ValueError, "Checkpoint file differs"):
            load_selected(self.root / "run", 17, "diagnostic", "cpu")
        checkpoint.write_bytes(original)
        cache_path = self.root / "run/seed_17/diagnostics.pt"
        cache_original = cache_path.read_bytes()
        corrupted = torch.load(cache_path, weights_only=True)
        corrupted["features"][0, 0] += 1
        torch.save(corrupted, cache_path)
        with self.assertRaisesRegex(ValueError, "Diagnostic tensor"):
            load_selected(self.root / "run", 17, "diagnostic", "cpu")
        cache_path.write_bytes(cache_original)
        with patch("evaluate.source_identity", return_value={"file_hashes": {}}):
            with self.assertRaisesRegex(ValueError, "Current source"):
                load_selected(self.root / "run", 17, "diagnostic", "cpu")
        mapping = self.data / "items.jsonl"
        mapping.write_text(mapping.read_text(encoding="utf-8").replace('"raw_id": "1"', '"raw_id": "different"'), encoding="utf-8")
        metadata = json.loads((self.data / "manifest.json").read_text(encoding="utf-8"))
        metadata["files"]["items.jsonl"] = file_sha256(mapping)
        (self.data / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Evaluation inputs differ"):
            load_selected(self.root / "run", 17, "diagnostic", "cpu")


if __name__ == "__main__":
    unittest.main()
