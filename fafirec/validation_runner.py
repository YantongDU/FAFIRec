"""Validation-only transfer experiments using an immutable selected CF teacher.

Training records deliberately omit test targets. A separate, explicit replay
entry point can evaluate the locked checkpoint after validation selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .ml1m_model import PaperFAFIRecModel
from .ratio_runner import evaluate, fit, load_inputs, model_config, write_rows
from .runner import (digest, file_digest, json_dump, resolve_device, runtime_identity,
                     seed_everything, tensor_digest, tensor_state_digest,
                     validate_diagnostics_cache)
from .transfer import initialize_transfer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "ml1m_validation_only_transfer_search_v1"
OVERRIDES = {"epochs", "patience", "device", "threads"}
CONFIG_KEYS = {"reference_run_dir", "output_dir", "seeds", "variants"} | OVERRIDES
VARIANT_DEFAULTS = {"control": "diagnostic", "projection_init": "random", "cf_prior": .5,
                    "ridge": 1., "align_weight": .3, "gate_lr_multiplier": 1.}
PAYLOAD_KEYS = ("protocol", "search_config", "training_config", "identity", "source_hashes",
                "runtime", "reference", "selection")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def normalize_config(supplied):
    if not isinstance(supplied, dict) or set(supplied) - CONFIG_KEYS:
        raise ValueError("Unknown validation-search configuration keys")
    required = {"reference_run_dir", "output_dir", "seeds", "variants"}
    if required - set(supplied):
        raise ValueError(f"Missing configuration keys: {sorted(required - set(supplied))}")
    cfg = copy.deepcopy(supplied)
    for key in ("reference_run_dir", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty path")
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    reference, output = Path(cfg["reference_run_dir"]), Path(cfg["output_dir"])
    if reference == output or reference in output.parents or output in reference.parents:
        raise ValueError("Output must not overlap the immutable reference run")
    seeds = cfg["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not isinstance(cfg["variants"], list) or not cfg["variants"]:
        raise ValueError("variants must be a nonempty list")
    variants, names = [], set()
    for supplied_variant in cfg["variants"]:
        if (not isinstance(supplied_variant, dict)
                or set(supplied_variant) - ({"name"} | set(VARIANT_DEFAULTS))):
            raise ValueError("Unknown variant keys")
        variant = {**VARIANT_DEFAULTS, **supplied_variant}
        name = variant.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Variant names must be distinct safe directory names")
        names.add(name)
        if variant["control"] not in PaperFAFIRecModel.CONTROLS:
            raise ValueError("Unknown variant control")
        if variant["projection_init"] not in {"random", "ridge"}:
            raise ValueError("projection_init must be random or ridge")
        if variant["control"] == "cf" and variant["projection_init"] != "random":
            raise ValueError("CF has no text projection")
        for key in ("cf_prior", "ridge", "align_weight", "gate_lr_multiplier"):
            variant[key] = _number(variant[key], key, key in {"ridge", "gate_lr_multiplier"})
        if not 0 < variant["cf_prior"] < 1 or variant["align_weight"] < 0:
            raise ValueError("cf_prior must be in (0,1) and align_weight nonnegative")
        variants.append(variant)
    cfg["variants"] = variants
    for key in ("epochs", "patience", "threads"):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if "device" in cfg and cfg["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda or auto")
    return cfg


def load_config(path, overrides=None):
    supplied = _read(path)
    supplied.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return normalize_config(supplied)


def source_identity():
    paths = list((ROOT / "fafirec").glob("*.py"))
    paths += [ROOT / name for name in ("validate_ml1m.py", "evaluate_transfer.py")]
    hashes = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(paths)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"base_commit": commit, "file_hashes": hashes}


def _check_reference_fingerprint(manifest):
    if manifest.get("protocol") != "ml1m_80_10_10_multi_positive":
        raise ValueError("Reference must use the version-2 multi-positive ratio protocol")
    expected = digest({"config": manifest["config"], "identity": manifest["identity"],
                       "source_hashes": manifest["source"]["file_hashes"], "runtime": manifest["runtime"]})
    if expected != manifest["fingerprint"]:
        raise ValueError("Reference manifest fingerprint is inconsistent")


def _check_checkpoint(checkpoint, identity, fingerprint, artifact, config):
    if (checkpoint.get("identity") != identity or checkpoint.get("fingerprint") != fingerprint
            or checkpoint.get("artifact") != artifact or checkpoint.get("optimizer_name") != "Adam"
            or checkpoint.get("model_config") != config
            or checkpoint.get("selection_k") is None
            or type(checkpoint.get("selected_epoch")) is not int or checkpoint["selected_epoch"] < 0
            or not math.isfinite(checkpoint.get("selected_metric", float("nan")))):
        raise ValueError("Checkpoint identity, architecture, selection or artifact differs")


def load_reference(search):
    """Validate immutable training inputs, teacher/cache and selected references.

    Completed old summaries are used only for identity/checksum binding. Their
    stored test metric fields are never inspected or used for search decisions.
    """
    root = Path(search["reference_run_dir"])
    manifest = _read(root / "manifest.json")
    _check_reference_fingerprint(manifest)
    if set(search["seeds"]) - set(manifest["config"]["seeds"]):
        raise ValueError("Search seed is absent from the reference run")
    records, features, _, identity = load_inputs(manifest["config"])
    if identity != manifest["identity"]:
        raise ValueError("Reference input mapping, users or text features changed")
    summary = _read(root / "summary.json")
    expected = {(s, c) for s in manifest["config"]["seeds"] for c in manifest["config"]["controls"]}
    indexed = {(row["seed"], row["control"]): row for row in summary}
    if len(indexed) != len(summary) or set(indexed) != expected:
        raise ValueError("Reference must contain its complete configured grid")
    provenance = {"manifest_sha256": file_digest(root / "manifest.json"),
                  "summary_sha256": file_digest(root / "summary.json"),
                  "fingerprint": manifest["fingerprint"], "seeds": {}}
    loaded = {}
    for seed in search["seeds"]:
        directory = root / f"seed_{seed}"
        teacher = torch.load(directory / "teacher/best.pt", weights_only=True, map_location="cpu")
        teacher_artifact = {"seed": seed, "control": "cf", "stage": "teacher"}
        _check_checkpoint(teacher, identity, manifest["fingerprint"], teacher_artifact,
                          model_config(manifest["config"], "cf", identity["items"]))
        if teacher["selection_k"] != manifest["config"]["selection_k"]:
            raise ValueError("Reference teacher selection cutoff differs")
        teacher_hash = digest({key: tensor_digest(value) for key, value in teacher["model"].items()})
        diagnostic_path = directory / "diagnostics.pt"
        diagnostics = validate_diagnostics_cache(torch.load(diagnostic_path, weights_only=True, map_location="cpu"),
                                                  identity, teacher_hash)
        diagnostic_hash = file_digest(diagnostic_path)
        references, checksums = {}, {}
        for control in ("cf", "uniform", "diagnostic"):
            if control not in manifest["config"]["controls"]:
                continue
            result_path = directory / f"{control}_summary.json"
            result = _read(result_path)
            checkpoint_path = directory / control / "best.pt"
            if (result != indexed[(seed, control)] or result["fingerprint"] != manifest["fingerprint"]
                    or result["checkpoint_sha256"] != file_digest(checkpoint_path)
                    or result["diagnostics_sha256"] != diagnostic_hash
                    or result["per_user_sha256"] != file_digest(directory / f"{control}_per_user.csv")):
                raise ValueError("Reference completed artifact integrity check failed")
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            artifact = {"seed": seed, "control": control, "stage": "student", "teacher_sha256": teacher_hash,
                        "diagnostics_tensor_sha256": tensor_digest(diagnostics),
                        "diagnostics_file_sha256": diagnostic_hash}
            if "projection_init" in manifest["config"]:
                ref_cfg = manifest["config"]
                artifact["initialization_configuration"] = {
                    "projection_init": ref_cfg["projection_init"] if control != "cf" else "random",
                    "cf_prior": ref_cfg.get("cf_prior", .5) if control != "cf" else .5,
                    "ridge": ref_cfg.get("ridge", 1.)}
            _check_checkpoint(checkpoint, identity, manifest["fingerprint"], artifact,
                              model_config(manifest["config"], control, identity["items"]))
            if checkpoint["selection_k"] != manifest["config"]["selection_k"]:
                raise ValueError("Reference student selection cutoff differs")
            references[control] = checkpoint
            checksums[control] = {"checkpoint_sha256": file_digest(checkpoint_path),
                                  "summary_sha256": file_digest(result_path)}
        provenance["seeds"][str(seed)] = {"teacher_file_sha256": file_digest(directory / "teacher/best.pt"),
            "teacher_tensor_sha256": teacher_hash, "diagnostics_file_sha256": diagnostic_hash,
            "diagnostics_tensor_sha256": tensor_digest(diagnostics), "references": checksums}
        loaded[seed] = {"teacher": teacher, "diagnostics": diagnostics, "references": references}
    return records, features, identity, manifest["config"], provenance, loaded


def _fingerprint(manifest):
    return digest({key: manifest[key] for key in PAYLOAD_KEYS})


def _manifest(search, training, identity, reference, device):
    source = source_identity()
    result = {"protocol": PROTOCOL, "search_config": search, "training_config": training,
        "identity": identity, "source_hashes": source["file_hashes"], "runtime": runtime_identity(device),
        "reference": reference, "selection": f"mean validation NDCG@{training['selection_k']} across declared seeds; name breaks exact ties",
        "source_commit": source["base_commit"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated_by_training_entrypoint": False}
    result["fingerprint"] = _fingerprint(result)
    return result


def _artifact(manifest, seed, variant):
    cached = manifest["reference"]["seeds"][str(seed)]
    return {"seed": seed, "control": variant["control"], "variant": variant["name"],
        "stage": "validation_candidate", "teacher_sha256": cached["teacher_tensor_sha256"],
        "diagnostics_tensor_sha256": cached["diagnostics_tensor_sha256"],
        "diagnostics_file_sha256": cached["diagnostics_file_sha256"], "initialization": variant}


def _training_config(base, search):
    cfg = copy.deepcopy(base)
    cfg.update({key: search[key] for key in OVERRIDES if key in search})
    cfg["include_initial_checkpoint"] = True
    return cfg


def _aggregate(results, names, metric):
    aggregated = []
    for name in names:
        values = [row for row in results if row["variant"] == name]
        metrics = {}
        for group in values[0]["validation"]:
            metrics[group] = {}
            for key in values[0]["validation"][group]:
                series = [row["validation"][group][key] for row in values]
                if key in {"users", "relevant_items"}:
                    if len(set(series)) != 1:
                        raise ValueError("Validation population differs across seeds")
                    metrics[group][key] = series[0]
                else:
                    metrics[group][key] = {"mean": statistics.mean(series),
                        "sample_std": statistics.stdev(series) if len(series) > 1 else None}
        aggregated.append({"variant": name, "seed_count": len(values), "metrics": metrics,
                           "selection_score": metrics["all"][metric]["mean"]})
    return sorted(aggregated, key=lambda row: (-row["selection_score"], row["variant"]))


def run(search, resume=False):
    search = normalize_config(search)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    records, features, identity, base, reference, loaded = load_reference(search)
    cfg = _training_config(base, search)
    torch.set_num_threads(cfg["threads"])
    device = resolve_device(cfg)
    fresh = _manifest(search, cfg, identity, reference, device)
    out = Path(search["output_dir"])
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError("Choose a new validation-search output directory or use --resume")
        manifest = _read(out / "manifest.json")
        if manifest.get("fingerprint") != _fingerprint(manifest) or manifest["fingerprint"] != fresh["fingerprint"]:
            raise ValueError("Cannot resume: source, configuration, runtime or reference changed")
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest = fresh
        json_dump(out / "manifest.json", manifest)
    # Test labels are not available to initialization, fitting, or validation.
    validation_records = [{key: value for key, value in record.items() if key != "test"} for record in records]
    results, reference_results = [], []
    for seed in search["seeds"]:
        directory = out / f"seed_{seed}"
        directory.mkdir(exist_ok=True)
        cached = loaded[seed]
        diagnostics = cached["diagnostics"]
        teacher = PaperFAFIRecModel(text_features=features, **model_config(base, "cf", identity["items"])).to(device)
        teacher.load_state_dict(cached["teacher"]["model"])
        teacher.eval().requires_grad_(False)
        for control, checkpoint in cached["references"].items():
            model = PaperFAFIRecModel(text_features=features, **checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model"])
            _, value = evaluate(model, validation_records, "validation", diagnostics, cfg, seed, False)
            reference_results.append({"seed": seed, "variant": control, "selected_epoch": checkpoint["selected_epoch"],
                                      "validation": value})
            del model
        for variant in search["variants"]:
            name = variant["name"]
            result_path = directory / f"{name}_summary.json"
            model_path = directory / name / "best.pt"
            csv_path = directory / f"{name}_validation.csv"
            local_cfg = {**cfg, "align_weight": variant["align_weight"],
                         "gate_lr_multiplier": variant["gate_lr_multiplier"]}
            artifact = _artifact(manifest, seed, variant)
            if resume and result_path.exists():
                result = _read(result_path)
                checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
                _check_checkpoint(checkpoint, identity, manifest["fingerprint"], artifact,
                                  model_config(local_cfg, variant["control"], identity["items"]))
                if (result["fingerprint"] != manifest["fingerprint"] or result["seed"] != seed
                        or result["variant"] != name or result["split"] != "validation"
                        or result["per_user_sha256"] != file_digest(csv_path)
                        or result["checkpoint_sha256"] != file_digest(model_path)):
                    raise ValueError("Completed validation artifact integrity check failed")
                results.append(result)
                continue
            seed_everything(seed + 1000)
            model = PaperFAFIRecModel(text_features=features,
                                     **model_config(local_cfg, variant["control"], identity["items"])).to(device)
            initialization = initialize_transfer(model, teacher, validation_records,
                projection_init=variant["projection_init"], cf_prior=variant["cf_prior"], ridge=variant["ridge"])
            _, initial_validation = evaluate(model, validation_records, "validation", diagnostics, local_cfg, seed, False)
            selected = fit(model, validation_records, diagnostics, local_cfg, seed + 1000,
                           directory / name, manifest["fingerprint"], identity, cfg["epochs"], resume, artifact)
            rows, value = evaluate(model, validation_records, "validation", diagnostics, local_cfg, seed)
            write_rows(csv_path, rows)
            result = {"seed": seed, "variant": name, "control": variant["control"], "split": "validation",
                "fingerprint": manifest["fingerprint"], "initialization": initialization,
                "initial_validation": initial_validation, **selected, "validation": value,
                "per_user_sha256": file_digest(csv_path), "checkpoint_sha256": file_digest(model_path)}
            json_dump(result_path, result)
            results.append(result)
            json_dump(out / "summary.json", results)
            del model
        del teacher
    json_dump(out / "summary.json", results)
    json_dump(out / "reference_validation.json", reference_results)
    candidates = _aggregate(results, [v["name"] for v in search["variants"]], f"ndcg{cfg['selection_k']}")
    reference_names = sorted({r["variant"] for r in reference_results})
    selection = {"fingerprint": manifest["fingerprint"], "split": "validation",
        "selection": manifest["selection"], "selected_variant": candidates[0]["variant"],
        "candidates": candidates, "references": _aggregate(reference_results, reference_names, f"ndcg{cfg['selection_k']}")}
    json_dump(out / "selection.json", selection)
    print(json.dumps({"completed_jobs": len(results), "selected_variant": selection["selected_variant"],
                      "selection_split": "validation", "output": str(out)}), flush=True)
    return selection


def load_validation_checkpoint(run_dir, seed, variant, device=None):
    root = Path(run_dir)
    manifest = _read(root / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("fingerprint") != _fingerprint(manifest):
        raise ValueError("Validation-run manifest protocol or fingerprint differs")
    if source_identity()["file_hashes"] != manifest["source_hashes"]:
        raise ValueError("Current source differs from recorded validation search")
    search = normalize_config(manifest["search_config"])
    if seed not in search["seeds"]:
        raise ValueError("Seed is absent from the validation search")
    variants = {v["name"]: v for v in search["variants"]}
    if variant not in variants:
        raise ValueError("Variant is absent from the validation search")
    records, features, identity, base, reference, loaded = load_reference(search)
    if identity != manifest["identity"] or reference != manifest["reference"]:
        raise ValueError("Reference inputs, teacher or diagnostics changed")
    cfg = _training_config(base, search)
    choice = variants[variant]
    cfg.update({"align_weight": choice["align_weight"], "gate_lr_multiplier": choice["gate_lr_multiplier"]})
    if cfg != {**manifest["training_config"], "align_weight": choice["align_weight"],
               "gate_lr_multiplier": choice["gate_lr_multiplier"]}:
        raise ValueError("Recorded training configuration differs")
    if device is not None:
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Unknown replay device")
        cfg["device"] = device
    torch.set_num_threads(cfg["threads"])
    checkpoint_path = root / f"seed_{seed}" / variant / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    _check_checkpoint(checkpoint, identity, manifest["fingerprint"], _artifact(manifest, seed, choice),
                      model_config(cfg, choice["control"], identity["items"]))
    summary_path = root / f"seed_{seed}" / f"{variant}_summary.json"
    result = _read(summary_path)
    if (result["fingerprint"] != manifest["fingerprint"]
            or result["checkpoint_sha256"] != file_digest(checkpoint_path)
            or result["per_user_sha256"] != file_digest(root / f"seed_{seed}" / f"{variant}_validation.csv")):
        raise ValueError("Selected validation artifacts changed")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model = PaperFAFIRecModel(text_features=features, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(resolve_device(cfg)).eval()
    return model, records, loaded[seed]["diagnostics"], cfg, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(load_config(args.config, {"output_dir": args.output_dir, "device": args.device}), args.resume)
