"""Reproducible teacher/student training with validation-only model selection."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrainingDataset, evaluation_batch, load_bundle
from .diagnostics import compute_diagnostics
from .metrics import exact_target_ranks_from_chunks, metrics_from_ranks, summarize_rows
from .model import FAFIRecModel, initialize_from_teacher

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "data_dir": "data/prime_pantry", "embedding_path": None,
    "feature_mode": "embeddings", "lexical_dim": 128,
    "output_dir": "runs/experiment", "seeds": [2025, 2026, 2027, 2028, 2029],
    "controls": ["cf", "uniform", "frequency", "diagnostic"],
    "epochs": 100, "teacher_epochs": 50, "patience": 10,
    "batch_size": 128, "eval_batch": 128, "item_chunk_size": 4096,
    "max_len": 50, "d_model": 64, "n_heads": 2, "n_layers": 2,
    "dropout": 0.1, "lr": 0.001, "weight_decay": 0.00001,
    "grad_clip": 5.0, "align_weight": 0.03,
    "bandwidths": [0.5, 1.0, 2.0, 3.5, 5.0],
    "ks": [5, 10, 20], "selection_k": 10,
    "candidate_policy": "full", "device": "auto", "threads": 4,
    "max_users": None,
}


def json_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def torch_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def source_identity():
    paths = [ROOT / "training.py", ROOT / "evaluate.py", ROOT / "prepare_data.py"]
    paths += sorted((ROOT / "fafirec").glob("*.py"))
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"base_commit": commit, "file_hashes": files}


def load_config(path, overrides=None):
    supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(supplied)
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    integer_keys = ["epochs", "teacher_epochs", "patience", "batch_size", "eval_batch", "item_chunk_size",
                    "max_len", "d_model", "n_heads", "n_layers", "threads", "lexical_dim", "selection_k"]
    for key in integer_keys:
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]) or any(type(s) is not int or s < 0 for s in cfg["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not cfg["controls"] or len(set(cfg["controls"])) != len(cfg["controls"]):
        raise ValueError("controls must be nonempty and distinct")
    if set(cfg["controls"]) - FAFIRecModel.CONTROLS:
        raise ValueError("Unknown model control")
    if not cfg["ks"] or any(type(k) is not int or k < 1 for k in cfg["ks"]):
        raise ValueError("ks must contain positive integers")
    if cfg["selection_k"] not in cfg["ks"]:
        raise ValueError("selection_k must be included in ks")
    for key in ["lr", "grad_clip"]:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if cfg["feature_mode"] not in {"embeddings", "lexical"}:
        raise ValueError("feature_mode must be embeddings or lexical")
    if cfg["feature_mode"] == "lexical" and cfg["embedding_path"]:
        raise ValueError("lexical mode must not silently ignore an embedding_path")
    if cfg["max_users"] is not None and (type(cfg["max_users"]) is not int or cfg["max_users"] < 2):
        raise ValueError("max_users is a debug limit of at least two users")
    if cfg["candidate_policy"] not in {"full", "exclude_seen_restore_target", "exclude_seen"}:
        raise ValueError("Unknown candidate_policy")
    if cfg["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    for key in ["data_dir", "embedding_path", "output_dir"]:
        if cfg[key] is not None:
            p = Path(cfg[key]).expanduser()
            cfg[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    return cfg


def resolve_device(cfg):
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if cfg["device"] == "cuda" or (cfg["device"] == "auto" and torch.cuda.is_available()) else "cpu")


def runtime_identity(device):
    return {"torch": str(torch.__version__), "numpy": str(np.__version__),
            "device": str(device), "machine": platform.machine(), "processor": platform.processor(),
            "cuda_runtime": torch.version.cuda,
            "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def validate_diagnostics_cache(cached, identity, teacher_hash):
    if cached["identity"] != identity or cached["teacher_sha256"] != teacher_hash:
        raise ValueError("Diagnostic cache does not match the teacher and users")
    features = cached["features"]
    if (not isinstance(features, torch.Tensor) or features.shape != (len(identity["user_order"]), 2)
            or not features.is_floating_point() or not bool(torch.isfinite(features).all())
            or cached.get("features_sha256") != tensor_digest(features)):
        raise ValueError("Diagnostic tensor shape, values, or integrity hash is invalid")
    return features


def model_config(cfg, control, n_items):
    return {"n_items": n_items, "d_model": cfg["d_model"], "max_len": cfg["max_len"],
            "n_heads": cfg["n_heads"], "n_layers": cfg["n_layers"], "dropout": cfg["dropout"],
            "control": control, "align_weight": cfg["align_weight"], "bandwidths": cfg["bandwidths"]}


@torch.no_grad()
def evaluate(model, records, split, diagnostics, cfg, seed, return_rows=True):
    """Two-pass chunked exact ranks, with the same GEMM for target and candidate scores."""
    model.eval()
    device = next(model.parameters()).device
    # Reproject the catalog once per checkpoint, not once per user batch.
    item_vectors = torch.empty(model.n_items + 1, model.d_model, device=device)
    item_vectors[0].zero_()
    for first in range(1, model.n_items + 1, cfg["item_chunk_size"]):
        ids = torch.arange(first, min(first + cfg["item_chunk_size"], model.n_items + 1), device=device)
        item_vectors[ids] = model._item_vectors(ids)
    rows = []
    totals = {}
    for start in range(0, len(records), cfg["eval_batch"]):
        indices = np.arange(start, min(start + cfg["eval_batch"], len(records)))
        hist, targets = evaluation_batch(records, indices, split, cfg["max_len"])
        hist = hist.to(device)
        diag = diagnostics[indices].to(device)
        states = model._user_states(hist, diag)[0]
        users = model._last(states, hist.ne(0))
        target_ids = targets.numpy()
        histories = [records[i]["train"] + ([records[i]["validation"]] if split == "test" else []) for i in indices]

        def chunks():
            for first in range(1, model.n_items + 1, cfg["item_chunk_size"]):
                ids = torch.arange(first, min(first + cfg["item_chunk_size"], model.n_items + 1), device=device)
                values = model.score_scale * (users @ item_vectors[ids].T)
                yield ids.cpu().numpy(), values.cpu().numpy()

        target_scores = np.full(len(indices), np.nan)
        for ids, values in chunks():
            present = (target_ids >= ids[0]) & (target_ids <= ids[-1])
            r = np.flatnonzero(present)
            target_scores[r] = values[r, target_ids[r] - ids[0]]
        ranks = exact_target_ranks_from_chunks(chunks(), target_scores, target_ids, histories,
                                               cfg["candidate_policy"], n_items=model.n_items)
        values = {k: metrics_from_ranks(ranks, k) for k in cfg["ks"]}
        batch_groups = np.array([records[i]["group"] for i in indices])
        for group in ["all", "cold", "middle", "warm"]:
            selected = np.ones(len(indices), dtype=bool) if group == "all" else batch_groups == group
            if not selected.any():
                continue
            total = totals.setdefault(group, {"users": 0, **{f"{m}{k}": 0.0 for k in cfg["ks"] for m in ["hit", "ndcg", "mrr"]}})
            total["users"] += int(selected.sum())
            for k, (hits, ndcgs) in values.items():
                total[f"hit{k}"] += float(hits[selected].sum())
                total[f"ndcg{k}"] += float(ndcgs[selected].sum())
                total[f"mrr{k}"] += float((hits[selected] / ranks[selected]).sum())
        if not return_rows:
            continue
        for j, index in enumerate(indices):
            row = {"user": records[index]["user"], "seed": seed, "control": model.control,
                   "model": model.control, "split": split, "group": records[index]["group"],
                   "training_interactions": len(records[index]["train"]),
                   "target": int(target_ids[j]), "rank": int(ranks[j])}
            for k, (hits, ndcgs) in values.items():
                row.update({f"hit{k}": float(hits[j]), f"ndcg{k}": float(ndcgs[j]),
                            f"mrr{k}": float(1.0 / ranks[j] if ranks[j] <= k else 0.0)})
            rows.append(row)
    summary = {group: {key: value if key == "users" else value / total["users"]
                       for key, value in total.items()} for group, total in totals.items()}
    return rows, summary


def fit(model, records, diagnostics, cfg, seed, out, fingerprint, identity, epochs, resume, artifact=None):
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, best_epoch, stale, logs, first_epoch = -math.inf, None, 0, 0, [], 1
    last_path = out / "last.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["fingerprint"] != fingerprint or checkpoint.get("artifact") != artifact:
            raise ValueError("Resume checkpoint does not match data, configuration, and source")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best, best_state, best_epoch = checkpoint["best_metric"], checkpoint["best_model"], checkpoint["best_epoch"]
        stale, logs, first_epoch = checkpoint["stale"], checkpoint["logs"], checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"])
    device = next(model.parameters()).device
    started = time.monotonic()
    if stale < cfg["patience"]:
        for epoch in range(first_epoch, epochs + 1):
            # Independent per-epoch streams make epoch-boundary resume exact.
            data = TrainingDataset(records, model.n_items, cfg["max_len"], seed + epoch)
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=generator)
            model.train()
            rec_total, rec_positions, align_total, align_users, gate_total = 0.0, 0, 0.0, 0, 0.0
            for hist, positive, negative, indices in loader:
                hist, positive, negative = hist.to(device), positive.to(device), negative.to(device)
                diag = diagnostics[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                parts = model(hist, positive, negative, diag)
                if not torch.isfinite(parts["loss"]):
                    raise RuntimeError("Nonfinite training loss")
                parts["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite training gradient")
                optimizer.step()
                n_pos = int((hist.ne(0) & positive.ne(0)).sum())
                n_users = len(hist)
                rec_total += float(parts["rec_loss"].detach()) * n_pos
                rec_positions += n_pos
                align_total += float(parts["align_loss"].detach()) * n_users
                gate_total += float(parts["mean_rho"].detach()) * n_users
                align_users += n_users
            _, validation = evaluate(model, records, "validation", diagnostics, cfg, seed, False)
            metric = validation["all"][f"ndcg{cfg['selection_k']}"]
            if metric > best:
                best, best_state, best_epoch, stale = metric, cpu_state(model), epoch, 0
            else:
                stale += 1
            row = {"epoch": epoch, "rec_loss": rec_total / rec_positions,
                   "align_loss": align_total / align_users, "mean_rho": gate_total / align_users,
                   "validation": validation, "elapsed_seconds_this_invocation": time.monotonic() - started}
            logs.append(row)
            json_dump(out / "training.json", logs)
            torch_dump(last_path, {"format_version": 1, "fingerprint": fingerprint, "artifact": artifact, "model": cpu_state(model),
                                  "optimizer": optimizer.state_dict(), "best_model": best_state,
                                  "best_metric": best, "best_epoch": best_epoch, "epoch": epoch,
                                  "stale": stale, "logs": logs, "rng": rng_state()})
            print(json.dumps({"control": model.control, "seed": seed, "epoch": epoch,
                              f"validation_ndcg{cfg['selection_k']}": metric}), flush=True)
            if stale >= cfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No selected checkpoint")
    model.load_state_dict(best_state)
    torch_dump(out / "best.pt", {"format_version": 1, "fingerprint": fingerprint,
                                "artifact": artifact,
                                "model": best_state, "model_config": model_config(cfg, model.control, model.n_items),
                                "selected_epoch": best_epoch, "selected_metric": best, "identity": identity})
    return {"selected_epoch": best_epoch,
            **({"selected_validation_ndcg10": best} if cfg["selection_k"] == 10 else {}),
            "selection_k": cfg["selection_k"], "selected_validation_metric": best,
            "epochs_completed": len(logs)}


def write_rows(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty evaluation")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(cfg):
    records, features, manifest = load_bundle(cfg["data_dir"], feature_mode=cfg["feature_mode"],
                                            embedding_path=cfg["embedding_path"], lexical_dim=cfg["lexical_dim"])
    if cfg["max_users"] is not None:
        records = records[:cfg["max_users"]]
    if len(records) < 2:
        raise ValueError("Need at least two diagnostic-eligible users")
    identity = {"records_sha256": digest(records), "features_sha256": tensor_digest(features),
                "data_manifest_sha256": manifest["data_manifest_sha256"],
                "item_mapping_sha256": manifest["files"]["items.jsonl"],
                "embedding_manifest_sha256": manifest.get("embedding_manifest_sha256"),
                "user_order": [str(r["user"]) for r in records], "items": int(features.shape[0] - 1)}
    return records, features, manifest, identity


def run(cfg, resume=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg["threads"])
    device = resolve_device(cfg)
    records, features, data_manifest, identity = load_inputs(cfg)
    source = source_identity()
    # The exact source hashes are stronger provenance than a base commit alone.
    runtime = runtime_identity(device)
    fingerprint = digest({"config": cfg, "identity": identity, "source_hashes": source["file_hashes"], "runtime": runtime})
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError("Output directory is nonempty; choose a new directory or use --resume")
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume: source, inputs, or configuration changed")
    else:
        out.mkdir(parents=True, exist_ok=True)
        json_dump(out / "manifest.json", {
            "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "zero-initialized linear diagnostic FAFIRec", "fingerprint": fingerprint,
            "config": cfg, "data": data_manifest, "identity": identity, "source": source,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "device": str(device), "selection": f"validation NDCG@{cfg['selection_k']} only",
            "runtime": runtime,
            "status": "pipeline_smoke_NOT_formal_LLM_results" if cfg["feature_mode"] == "lexical" or cfg["max_users"] else "method_aligned_experiment",
            "candidate_policy": cfg["candidate_policy"], "ties": "smaller item ID first",
            "teacher": "same-seed CF teacher; same selected initialization across controls",
            "resume": "exact epoch-boundary RNG restoration; completed controls are verified and skipped",
        })
    summaries = []
    for seed in cfg["seeds"]:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_everything(seed)
        teacher = FAFIRecModel(text_features=features, **model_config(cfg, "cf", identity["items"])).to(device)
        zeros = torch.zeros(len(records), 2)
        teacher_artifact = {"seed": seed, "control": "cf", "stage": "teacher"}
        fit(teacher, records, zeros, cfg, seed, seed_dir / "teacher", fingerprint, identity, cfg["teacher_epochs"], resume, teacher_artifact)
        diag_path = seed_dir / "diagnostics.pt"
        if resume and diag_path.exists():
            cached = torch.load(diag_path, map_location="cpu", weights_only=True)
            diagnostics = validate_diagnostics_cache(cached, identity, tensor_state_digest(teacher))
        else:
            diagnostics, diag_manifest = compute_diagnostics(teacher, records, cfg["max_len"], cfg["eval_batch"], seed + 101, device)
            diag_manifest.update({"identity": identity, "teacher_sha256": tensor_state_digest(teacher)})
            json_dump(seed_dir / "diagnostics.json", diag_manifest)
            torch_dump(diag_path, {"features": diagnostics, "identity": identity,
                                  "features_sha256": tensor_digest(diagnostics),
                                  "teacher_sha256": tensor_state_digest(teacher)})
        teacher.requires_grad_(False)
        for control in cfg["controls"]:
            artifact = {"seed": seed, "control": control, "stage": "student",
                        "teacher_sha256": tensor_state_digest(teacher),
                        "diagnostics_tensor_sha256": tensor_digest(diagnostics),
                        "diagnostics_file_sha256": file_digest(diag_path)}
            result_path = seed_dir / f"{control}_summary.json"
            if resume and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                per_user = seed_dir / f"{control}_per_user.csv"
                checkpoint = seed_dir / control / "best.pt"
                if result["fingerprint"] != fingerprint or not per_user.exists() or not checkpoint.exists():
                    raise ValueError("Completed-control artifact is incomplete or mismatched")
                if result["per_user_sha256"] != file_digest(per_user) or result["checkpoint_sha256"] != file_digest(checkpoint):
                    raise ValueError("Completed-control artifacts changed; refusing to resume")
                selected_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if selected_checkpoint.get("artifact") != artifact or result["diagnostics_sha256"] != file_digest(diag_path):
                    raise ValueError("Completed control has a mismatched seed or diagnostic cache")
                summaries.append(result)
                continue
            seed_everything(seed + 1000)
            model = FAFIRecModel(text_features=features, **model_config(cfg, control, identity["items"])).to(device)
            initialize_from_teacher(model, teacher)
            selected = fit(model, records, diagnostics, cfg, seed + 1000, seed_dir / control,
                           fingerprint, identity, cfg["epochs"], resume, artifact)
            rows, test = evaluate(model, records, "test", diagnostics, cfg, seed)
            write_rows(seed_dir / f"{control}_per_user.csv", rows)
            learned = {name: value.detach().cpu().tolist() for name, value in model.named_parameters()
                       if name in {"control_bias", "control_weights", "gate_bias", "item_gate_logit", "conflict_scale_raw"}}
            result = {"seed": seed, "control": control, "fingerprint": fingerprint, **selected, "test": test,
                      "per_user_sha256": file_digest(seed_dir / f"{control}_per_user.csv"),
                      "checkpoint_sha256": file_digest(seed_dir / control / "best.pt"),
                      "diagnostics_sha256": file_digest(diag_path), "learned_control_parameters": learned,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            json_dump(result_path, result)
            summaries.append(result)
            json_dump(out / "summary.json", summaries)
        del teacher
    json_dump(out / "summary.json", summaries)
    print(f"COMPLETED {out}", flush=True)
    return summaries


def tensor_state_digest(model):
    return digest({key: tensor_digest(value) for key, value in model.state_dict().items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, {"output_dir": args.output_dir, "device": args.device})
    run(cfg, args.resume)


if __name__ == "__main__":
    main()
