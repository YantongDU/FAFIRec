"""ML-1M ratio-split experiments with multi-positive ranking evaluation.

This is the revised method with the manuscript's stated experiment settings.
Unspecified chronological ordering, rounding, candidate policy, teacher budget
and encoder details are explicitly completed and recorded, not attributed to
the historical paper's undisclosed implementation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrainingDataset
from .diagnostics import compute_diagnostics
from .ml1m_model import PaperFAFIRecModel
from .multitarget_metrics import evaluate_topk, summarize_multitarget_rows
from .ratio_data import load_ratio_bundle, ratio_evaluation_batch
from .transfer import initialize_transfer, optimizer_groups
from .runner import (cpu_state, digest, file_digest, json_dump, model_config,
                     resolve_device, restore_rng, rng_state, runtime_identity,
                     seed_everything, tensor_digest, tensor_state_digest,
                     torch_dump, validate_diagnostics_cache)

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "data_dir": "data/ml1m_paper", "embedding_path": "data/ml1m_llm2vec.pt",
    "feature_mode": "embeddings", "lexical_dim": 128,
    "output_dir": "runs/ml1m_paper", "seeds": [2025, 2026, 2027, 2028, 2029],
    "controls": ["cf", "uniform", "diagnostic"],
    "epochs": 150, "teacher_epochs": 30, "patience": 10,
    "batch_size": 2048, "eval_batch": 128, "item_chunk_size": 4096,
    "max_len": 50, "d_model": 64, "n_heads": 2, "n_layers": 1,
    "dropout": 0.1, "lr": 0.001, "weight_decay": 1e-5, "optimizer": "adam",
    "grad_clip": 5.0, "align_weight": 0.3,
    "bandwidths": [0.5, 1.0, 2.0, 3.5, 5.0],
    "ks": [10, 50], "selection_k": 50,
    "candidate_policy": "exclude_seen", "device": "cpu", "threads": 4,
    "max_users": None, "experiment_note": "",
    "projection_init": "random", "cf_prior": .5, "ridge": 1.,
    "gate_lr_multiplier": 1., "include_initial_checkpoint": False,
}
ASSUMPTIONS = {
    "split": "per-user timestamp order, stable source-order ties; floor(.8*n), floor(.1*n), remainder",
    "evaluation": "multi-positive user-macro Recall/NDCG; full metadata catalog minus observed history",
    "validation_history": "training only",
    "test_history": "training plus the complete validation block",
    "teacher_budget": "30 epochs maximum by default; unspecified in historical manuscript",
    "encoder": "one causal layer, two heads, max length50, dropout0.1; unspecified manuscript details",
    "diagnostics": "revised method's fixed linear control; not historical EMA/nonlinear control",
    "optimizer": "Adam with coupled L2 weight decay, not AdamW; default1e-5 lies in the published {1e-6,1e-5,1e-4} range; validation tuning is not claimed",
    "user_profiles": "static demographics may be audited by data preparation; revised model consumes item text only",
    "data_statistics": "actual retained raw-release population; no sampling to match undocumented historical counts",
}


def load_config(path, overrides=None):
    supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if set(supplied) - set(DEFAULTS):
        raise ValueError(f"Unknown configuration keys: {sorted(set(supplied) - set(DEFAULTS))}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(supplied)
    cfg.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for key in ("epochs", "teacher_epochs", "patience", "batch_size", "eval_batch", "item_chunk_size",
                "max_len", "d_model", "n_heads", "n_layers", "threads", "lexical_dim", "selection_k"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["seeds"] or any(type(seed) is not int or seed < 0 for seed in cfg["seeds"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not cfg["controls"] or len(set(cfg["controls"])) != len(cfg["controls"]) or set(cfg["controls"]) - PaperFAFIRecModel.CONTROLS:
        raise ValueError("controls must be distinct known control names")
    if not cfg["ks"] or any(type(k) is not int or k < 1 for k in cfg["ks"]) or cfg["selection_k"] not in cfg["ks"]:
        raise ValueError("ks must include positive integer selection_k")
    for key in ("lr", "grad_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if cfg["optimizer"] != "adam":
        raise ValueError("This paper-protocol entry point requires Adam, not AdamW")
    if cfg['projection_init'] not in {'random', 'ridge'}:
        raise ValueError('projection_init must be random or ridge')
    for key in ('cf_prior', 'ridge', 'gate_lr_multiplier'):
        if (isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float))
                or not math.isfinite(cfg[key]) or cfg[key] <= 0):
            raise ValueError(f'{key} must be a positive finite number')
    if cfg['cf_prior'] >= 1:
        raise ValueError('cf_prior must lie strictly between zero and one')
    if type(cfg['include_initial_checkpoint']) is not bool:
        raise ValueError('include_initial_checkpoint must be boolean')
    if (isinstance(cfg["weight_decay"], bool) or not isinstance(cfg["weight_decay"], (int, float))
            or not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if cfg["feature_mode"] not in {"embeddings", "lexical"}:
        raise ValueError("feature_mode must be embeddings or lexical")
    if cfg["feature_mode"] == "lexical" and cfg["embedding_path"] is not None:
        raise ValueError("lexical mode must explicitly set embedding_path=null")
    if cfg["candidate_policy"] not in {"full", "exclude_seen"}:
        raise ValueError("candidate_policy must be full or exclude_seen; target restoration is unsupported")
    if cfg["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if cfg["max_users"] is not None and (type(cfg["max_users"]) is not int or cfg["max_users"] < 2):
        raise ValueError("max_users is an explicit debug limit of at least two users")
    for key in ("data_dir", "embedding_path", "output_dir"):
        if cfg[key] is not None:
            path = Path(cfg[key]).expanduser()
            cfg[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    return cfg


def source_identity():
    paths = list((ROOT / "fafirec").glob("*.py"))
    paths += [ROOT / name for name in ("train_ml1m.py", "evaluate_ml1m.py", "analyze_ml1m.py", "prepare_ml1m_paper.py")]
    hashes = {path.relative_to(ROOT).as_posix(): file_digest(path) for path in sorted(paths) if path.is_file()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"base_commit": commit, "file_hashes": hashes}


def load_inputs(cfg):
    records, features, manifest = load_ratio_bundle(cfg["data_dir"], feature_mode=cfg["feature_mode"],
                                                  embedding_path=cfg["embedding_path"], lexical_dim=cfg["lexical_dim"])
    if cfg["max_users"] is not None:
        records = records[:cfg["max_users"]]
    if len(records) < 2:
        raise ValueError("Need at least two eligible users")
    identity = {"records_sha256": digest(records), "features_sha256": tensor_digest(features),
                "data_manifest_sha256": manifest["data_manifest_sha256"],
                "item_mapping_sha256": manifest["files"]["items.jsonl"],
                "embedding_manifest_sha256": manifest.get("embedding_manifest_sha256"),
                "user_order": [str(record["user"]) for record in records], "items": len(features) - 1}
    return records, features, manifest, identity


@torch.no_grad()
def evaluate(model, records, split, diagnostics, cfg, seed, return_rows=True):
    """Score each held-out block against the same fixed candidate convention."""
    model.eval()
    device = next(model.parameters()).device
    item_vectors = model.item_vectors()
    rows, totals = [], {}
    for start in range(0, len(records), cfg["eval_batch"]):
        indices = np.arange(start, min(start + cfg["eval_batch"], len(records)))
        histories, targets = ratio_evaluation_batch(records, indices, split, cfg["max_len"])
        users = model.user_vectors(histories.to(device), diagnostics[indices].to(device))
        scores = (model.score_scale * (users @ item_vectors.T)).cpu().numpy()
        observed = [records[index]["train"] + (records[index]["validation"] if split == "test" else []) for index in indices]
        metrics = evaluate_topk(scores, targets, observed, cfg["ks"], cfg["candidate_policy"])
        for index, target_set, metric in zip(indices, targets, metrics):
            group = records[index]["group"]
            row = {"user": str(records[index]["user"]), "seed": seed, "control": model.control,
                   "split": split, "group": group, "training_interactions": len(records[index]["train"]),
                   "candidate_policy": cfg["candidate_policy"], "target_count": metric["relevant_count"],
                   "targets": json.dumps(sorted(int(item) for item in target_set)),
                   **{key: value for key, value in metric.items() if key != "top_items"},
                   "top_items": json.dumps(metric["top_items"])}
            for name in ("all", group):
                total = totals.setdefault(name, {"users": 0, "relevant_items": 0,
                    **{f"{m}{k}": 0. for k in cfg["ks"] for m in ("recall", "ndcg", "hit")}})
                total["users"] += 1
                total["relevant_items"] += metric["relevant_count"]
                for key in total:
                    if key not in {"users", "relevant_items"}:
                        total[key] += metric[key]
            if return_rows:
                rows.append(row)
    summary = {group: {key: value if key in {"users", "relevant_items"} else value / total["users"]
                       for key, value in total.items()} for group, total in totals.items()}
    return rows, summary


def fit(model, records, diagnostics, cfg, seed, out, fingerprint, identity, epochs, resume=False, artifact=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(optimizer_groups(model, cfg['lr'], cfg.get('gate_lr_multiplier', 1.)),
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, best_epoch, stale, logs, first_epoch = -math.inf, None, 0, 0, [], 1
    last_path = out / "last.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if (checkpoint["fingerprint"] != fingerprint or checkpoint.get("artifact") != artifact
                or checkpoint.get("optimizer_name") != "Adam"):
            raise ValueError("Resume checkpoint identity or optimizer differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best, best_state, best_epoch = checkpoint["best_metric"], checkpoint["best_model"], checkpoint["best_epoch"]
        stale, logs, first_epoch = checkpoint["stale"], checkpoint["logs"], checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"])
    device = next(model.parameters()).device
    started = time.monotonic()
    if first_epoch == 1 and cfg.get('include_initial_checkpoint', False):
        _, initial_validation = evaluate(model, records, 'validation', diagnostics, cfg, seed, False)
        best = initial_validation['all'][f"ndcg{cfg['selection_k']}"]
        best_state, best_epoch = cpu_state(model), 0
        json_dump(out / 'initial_validation.json', initial_validation)
    if stale < cfg["patience"]:
        for epoch in range(first_epoch, epochs + 1):
            data = TrainingDataset(records, model.n_items, cfg["max_len"], seed + epoch)
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=generator)
            model.train()
            rec_total, positions, align_total, users_total = 0., 0, 0., 0
            for histories, positive, negative, indices in loader:
                histories, positive, negative = histories.to(device), positive.to(device), negative.to(device)
                optimizer.zero_grad(set_to_none=True)
                parts = model(histories, positive, negative, diagnostics[indices].to(device))
                if not torch.isfinite(parts["loss"]):
                    raise RuntimeError("Nonfinite training loss")
                parts["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite training gradient")
                optimizer.step()
                count = int((histories.ne(0) & positive.ne(0)).sum())
                rec_total += float(parts["rec_loss"].detach()) * count
                positions += count
                align_total += float(parts["align_loss"].detach()) * len(histories)
                users_total += len(histories)
            _, validation = evaluate(model, records, "validation", diagnostics, cfg, seed, False)
            metric = validation["all"][f"ndcg{cfg['selection_k']}"]
            if metric > best:
                best, best_state, best_epoch, stale = metric, cpu_state(model), epoch, 0
            else:
                stale += 1
            logs.append({"epoch": epoch, "rec_loss": rec_total / positions,
                         "align_loss": align_total / users_total, "validation": validation,
                         "elapsed_seconds_this_invocation": time.monotonic() - started})
            json_dump(out / "training.json", logs)
            torch_dump(last_path, {"format_version": 2, "fingerprint": fingerprint, "artifact": artifact,
                "optimizer_name": "Adam", "model": cpu_state(model), "optimizer": optimizer.state_dict(),
                "best_model": best_state, "best_metric": best, "best_epoch": best_epoch,
                "epoch": epoch, "stale": stale, "logs": logs, "rng": rng_state()})
            print(json.dumps({"control": model.control, "seed": seed, "epoch": epoch,
                              f"validation_ndcg{cfg['selection_k']}": metric}), flush=True)
            if stale >= cfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No selected checkpoint")
    model.load_state_dict(best_state)
    torch_dump(out / "best.pt", {"format_version": 2, "model_class": "PaperFAFIRecModel",
        "fingerprint": fingerprint, "artifact": artifact, "optimizer_name": "Adam", "model": best_state,
        "model_config": model_config(cfg, model.control, model.n_items), "identity": identity,
        "selected_epoch": best_epoch, "selected_metric": best, "selection_k": cfg["selection_k"]})
    return {"selected_epoch": best_epoch, "selection_k": cfg["selection_k"],
            "selected_validation_metric": best, f"selected_validation_ndcg{cfg['selection_k']}": best,
            "epochs_completed": len(logs)}


def write_rows(path, rows):
    if not rows:
        raise ValueError("Cannot export empty evaluation")
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(cfg, resume=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg["threads"])
    device = resolve_device(cfg)
    records, features, data_manifest, identity = load_inputs(cfg)
    source, runtime = source_identity(), runtime_identity(device)
    fingerprint = digest({"config": cfg, "identity": identity, "source_hashes": source["file_hashes"], "runtime": runtime})
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError("Choose a new output directory or use --resume")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume: source, configuration, runtime, or inputs changed")
    else:
        out.mkdir(parents=True, exist_ok=True)
        json_dump(out / "manifest.json", {"format_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(), "method": "revised diagnostic FAFIRec",
            "protocol": "ml1m_80_10_10_multi_positive", "fingerprint": fingerprint,
            "config": cfg, "identity": identity, "source": source, "runtime": runtime,
            "data": data_manifest, "assumptions": ASSUMPTIONS,
            "selection": f"validation NDCG@{cfg['selection_k']} only", "optimizer": "Adam",
            "candidate_policy": cfg["candidate_policy"], "ties": "smaller item ID first",
            "status": "lexical_or_debug_attempt_NOT_formal_LLM_results" if cfg["feature_mode"] == "lexical" or cfg["max_users"] else "explicit_LLM_feature_protocol_attempt_not_historical_table_reproduction"})
    summaries = []
    for seed in cfg["seeds"]:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_everything(seed)
        teacher = PaperFAFIRecModel(text_features=features, **model_config(cfg, "cf", identity["items"])).to(device)
        fit(teacher, records, torch.zeros(len(records), 2), cfg, seed, seed_dir / "teacher", fingerprint,
            identity, cfg["teacher_epochs"], resume, {"seed": seed, "control": "cf", "stage": "teacher"})
        teacher_hash = tensor_state_digest(teacher)
        diag_path = seed_dir / "diagnostics.pt"
        if resume and diag_path.exists():
            diagnostics = validate_diagnostics_cache(torch.load(diag_path, weights_only=True, map_location="cpu"), identity, teacher_hash)
        else:
            diagnostics, provenance = compute_diagnostics(teacher, records, cfg["max_len"], cfg["eval_batch"], seed, device)
            provenance.update({"identity": identity, "teacher_sha256": teacher_hash})
            json_dump(seed_dir / "diagnostics.json", provenance)
            torch_dump(diag_path, {"features": diagnostics, "identity": identity,
                "features_sha256": tensor_digest(diagnostics), "teacher_sha256": teacher_hash})
        teacher.requires_grad_(False)
        for control in cfg["controls"]:
            initialization_config = {
                'projection_init': cfg.get('projection_init', 'random') if control != 'cf' else 'random',
                'cf_prior': cfg.get('cf_prior', .5) if control != 'cf' else .5,
                'ridge': cfg.get('ridge', 1.),
            }
            artifact = {"seed": seed, "control": control, "stage": "student", "teacher_sha256": teacher_hash,
                        "diagnostics_tensor_sha256": tensor_digest(diagnostics), "diagnostics_file_sha256": file_digest(diag_path),
                        'initialization_configuration': initialization_config}
            result_path = seed_dir / f"{control}_summary.json"
            checkpoint_path = seed_dir / control / "best.pt"
            csv_path = seed_dir / f"{control}_per_user.csv"
            if resume and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (result["fingerprint"] != fingerprint or result["per_user_sha256"] != file_digest(csv_path)
                        or result["checkpoint_sha256"] != file_digest(checkpoint_path)
                        or result["diagnostics_sha256"] != file_digest(diag_path)):
                    raise ValueError("Completed artifact integrity check failed")
                if torch.load(checkpoint_path, weights_only=True, map_location="cpu")["artifact"] != artifact:
                    raise ValueError("Completed checkpoint seed/control/cache differs")
                summaries.append(result)
                continue
            seed_everything(seed + 1000)
            model = PaperFAFIRecModel(text_features=features, **model_config(cfg, control, identity["items"])).to(device)
            initialization = initialize_transfer(model, teacher, records, **initialization_config)
            (seed_dir / control).mkdir(exist_ok=True)
            json_dump(seed_dir / control / 'initialization.json', initialization)
            selected = fit(model, records, diagnostics, cfg, seed + 1000, seed_dir / control,
                           fingerprint, identity, cfg["epochs"], resume, artifact)
            rows, test = evaluate(model, records, "test", diagnostics, cfg, seed)
            write_rows(csv_path, rows)
            result = {"seed": seed, "control": control, "fingerprint": fingerprint, **selected,
                      'initialization': initialization,
                      "test": test, "per_user_sha256": file_digest(csv_path),
                      "checkpoint_sha256": file_digest(checkpoint_path), "diagnostics_sha256": file_digest(diag_path)}
            json_dump(result_path, result)
            summaries.append(result)
            json_dump(out / "summary.json", summaries)
            del model
        del teacher
    json_dump(out / "summary.json", summaries)
    print(f"COMPLETED {out}", flush=True)
    return summaries


def load_selected(run_dir, seed, control, device=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "ml1m_80_10_10_multi_positive":
        raise ValueError("This is not a ratio-split multi-positive experiment")
    if source_identity()["file_hashes"] != manifest["source"]["file_hashes"]:
        raise ValueError("Current source differs from recorded experiment")
    cfg = manifest["config"].copy()
    if device:
        cfg["device"] = device
    torch.set_num_threads(cfg["threads"])
    records, features, _, identity = load_inputs(cfg)
    if identity != manifest["identity"]:
        raise ValueError("Evaluation input mapping or features changed")
    seed_dir = root / f"seed_{seed}"
    path = seed_dir / control / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    artifact = checkpoint.get("artifact", {})
    if (checkpoint["identity"] != identity or checkpoint["fingerprint"] != manifest["fingerprint"]
            or artifact.get("seed") != seed or artifact.get("control") != control or artifact.get("stage") != "student"
            or checkpoint["model_config"]["control"] != control or checkpoint.get("optimizer_name") != "Adam"):
        raise ValueError("Checkpoint seed, control, optimizer, or input identity differs")
    result_path = seed_dir / f"{control}_summary.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["fingerprint"] != manifest["fingerprint"] or result["checkpoint_sha256"] != file_digest(path):
            raise ValueError("Checkpoint file differs from completed run")
    teacher = torch.load(seed_dir / "teacher/best.pt", map_location="cpu", weights_only=True)
    if (teacher["identity"] != identity or teacher["fingerprint"] != manifest["fingerprint"]
            or teacher["artifact"] != {"seed": seed, "control": "cf", "stage": "teacher"}):
        raise ValueError("Teacher artifact identity differs")
    teacher_hash = digest({key: tensor_digest(value) for key, value in teacher["model"].items()})
    cache_path = seed_dir / "diagnostics.pt"
    diagnostics = validate_diagnostics_cache(torch.load(cache_path, map_location="cpu", weights_only=True), identity, teacher_hash)
    if (artifact["teacher_sha256"] != teacher_hash or artifact["diagnostics_tensor_sha256"] != tensor_digest(diagnostics)
            or artifact["diagnostics_file_sha256"] != file_digest(cache_path)):
        raise ValueError("Student teacher/diagnostic binding differs")
    model = PaperFAFIRecModel(text_features=features, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(resolve_device(cfg)).eval()
    return model, records, diagnostics, cfg, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, {"output_dir": args.output_dir, "device": args.device})
    run(cfg, args.resume)
