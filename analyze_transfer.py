"""Audit a complete transfer search by recomputing its validation rankings.

This entry point never evaluates a model. Candidate metrics are reconstructed
from per-user target/ranking CSVs and checked against immutable prepared input.
Old reference controls have no exported validation rankings, so their reported
validation means are checked against the original selected-epoch logs.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from analyze_results import _holm_adjust, paired_seed_comparison
from fafirec.multitarget_metrics import summarize_multitarget_rows
from fafirec.runner import file_digest, json_dump
from fafirec.validation_runner import (
    PROTOCOL, _artifact, _check_checkpoint, _fingerprint, _optimization_config,
    _training_config, _variant_config,
    load_reference, model_config, normalize_config, source_identity,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _same(actual, recorded, context):
    """Compare complete structures, allowing only numeric summation roundoff."""
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or set(actual) != set(recorded):
            raise ValueError(f"{context}: keys differ")
        for key in actual:
            _same(actual[key], recorded[key], f"{context}.{key}")
    elif isinstance(actual, list):
        if not isinstance(recorded, list) or len(actual) != len(recorded):
            raise ValueError(f"{context}: list length differs")
        for index, (left, right) in enumerate(zip(actual, recorded)):
            _same(left, right, f"{context}[{index}]")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                or not math.isfinite(recorded)
                or not math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"{context}: numeric value differs")
    elif type(actual) is not type(recorded) or actual != recorded:
        raise ValueError(f"{context}: value differs")


def _ids(value, n_items, name):
    try:
        items = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {name} JSON") from error
    if (not isinstance(items, list) or not items
            or any(type(item) is not int or not 1 <= item <= n_items for item in items)
            or len(set(items)) != len(items)):
        raise ValueError(f"{name} must contain distinct nonpadding catalog IDs")
    return items


def _rows(path, cfg, records, seed, variant, n_items):
    prepared = {record["user"]: record for record in records}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError("Empty validation per-user output")
    rows, seen_users = [], set()
    for value in raw:
        user = value.get("user")
        if user not in prepared or user in seen_users:
            raise ValueError("Unknown or duplicate validation user")
        seen_users.add(user)
        record = prepared[user]
        if (value.get("seed") != str(seed) or value.get("split") != "validation"
                or value.get("control") != variant["control"]
                or ("variant" in value and value["variant"] != variant["name"])):
            raise ValueError("CSV seed, split, control or variant differs")
        if value.get("candidate_policy") != cfg["candidate_policy"]:
            raise ValueError("CSV candidate policy differs")
        if value.get("group") != record["group"]:
            raise ValueError("CSV group differs from training-count group")
        targets = _ids(value.get("targets"), n_items, "targets")
        expected_targets = set(record["validation"])
        if set(targets) != expected_targets:
            raise ValueError("CSV targets differ from prepared validation targets")
        top = _ids(value.get("top_items"), n_items, "top_items")
        observed = set(record["train"])
        if cfg["candidate_policy"] == "exclude_seen":
            if expected_targets & observed:
                raise ValueError("Validation target is excluded by observed history")
            if set(top) & observed:
                raise ValueError("Ranking contains a training-observed item")
            candidate_count = n_items - len(observed)
        elif cfg["candidate_policy"] == "full":
            candidate_count = n_items
        else:
            raise ValueError("Unexpected target-conditioned candidate policy")
        if len(top) != min(max(cfg["ks"]), candidate_count):
            raise ValueError("Stored ranking length differs from full eligible catalog")
        try:
            count = int(value["target_count"])
            relevant = int(value["relevant_count"])
            training = int(value["training_interactions"])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Invalid integer validation target/training counts") from error
        if count != len(expected_targets) or relevant != count:
            raise ValueError("CSV relevant counts differ from prepared targets")
        if training != len(record["train"]):
            raise ValueError("CSV training count differs from prepared history")
        row = {"user": user, "group": record["group"], "relevant_count": count}
        for k in cfg["ks"]:
            ranked = top[:k]
            hits = sum(item in expected_targets for item in ranked)
            dcg = sum(1. / math.log2(rank + 2) for rank, item in enumerate(ranked) if item in expected_targets)
            idcg = sum(1. / math.log2(rank + 2) for rank in range(min(k, count)))
            for metric, computed in ((f"recall{k}", hits / count), (f"ndcg{k}", dcg / idcg),
                                     (f"hit{k}", float(hits > 0))):
                try:
                    saved = float(value[metric])
                except (KeyError, ValueError, TypeError) as error:
                    raise ValueError(f"Missing numeric {metric}") from error
                _same(computed, saved, f"CSV {metric}")
                row[metric] = computed
        rows.append(row)
    if seen_users != set(prepared):
        raise ValueError("Validation output does not cover every prepared user")
    return rows


def _index(values, jobs, key, context):
    if not isinstance(values, list):
        raise ValueError(f"{context} must be a list")
    try:
        indexed = {(value["seed"], value[key]): value for value in values}
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid {context} job identity") from error
    if (any(type(value.get("seed")) is not int for value in values)
            or len(indexed) != len(values) or set(indexed) != jobs):
        raise ValueError(f"Complete configured grid required: {context} has missing, duplicate or extra jobs")
    return indexed


def _aggregate(results, names, metric):
    rows = []
    for name in names:
        jobs = [result for result in results if result["variant"] == name]
        metrics = {}
        for group in jobs[0]["validation"]:
            metrics[group] = {}
            for key in jobs[0]["validation"][group]:
                values = [job["validation"][group][key] for job in jobs]
                if key in {"users", "relevant_items"}:
                    if len(set(values)) != 1:
                        raise ValueError("Validation population differs across seeds")
                    metrics[group][key] = values[0]
                else:
                    metrics[group][key] = {"mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        rows.append({"variant": name, "seed_count": len(jobs), "metrics": metrics,
                     "selection_score": metrics["all"][metric]["mean"]})
    return sorted(rows, key=lambda row: (-row["selection_score"], row["variant"]))


def _selected_log(path, checkpoint, cfg, provenance):
    epoch = checkpoint["selected_epoch"]
    if epoch == 0:
        selected_path = path / "initial_validation.json"
        value = _read(selected_path)
    else:
        selected_path = path / "training.json"
        logs = _read(selected_path)
        if (not isinstance(logs, list) or [log.get("epoch") for log in logs] != list(range(1, len(logs) + 1))
                or epoch > len(logs)):
            raise ValueError("Selected epoch is absent from the contiguous training log")
        value = logs[epoch - 1]["validation"]
    _same(checkpoint["selected_metric"], value["all"][f"ndcg{cfg['selection_k']}"], "Selected log metric")
    provenance[str(selected_path)] = file_digest(selected_path)
    return value


def analyze_runs(run_dir, comparison=None, baseline=None, bootstrap_samples=10000, seed=2025):
    root = Path(run_dir)
    manifest = _read(root / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("fingerprint") != _fingerprint(manifest):
        raise ValueError("Validation-search manifest protocol or fingerprint differs")
    if source_identity()["file_hashes"] != manifest["source_hashes"]:
        raise ValueError("Current source differs from recorded validation search")
    search = normalize_config(manifest["search_config"])
    torch.set_num_threads(manifest["training_config"]["threads"])
    records, features, identity, base_cfg, reference, loaded = load_reference(search)
    del features
    cfg = _training_config(base_cfg, search)
    if (identity != manifest["identity"] or reference != manifest["reference"]
            or cfg != manifest["training_config"]):
        raise ValueError("Reference inputs or training configuration changed")
    variants = {variant["name"]: variant for variant in search["variants"]}
    jobs = {(s, name) for s in search["seeds"] for name in variants}
    indexed = _index(_read(root / "summary.json"), jobs, "variant", "search summary")
    seed_directories = {root / f"seed_{s}" for s in search["seeds"]}
    if {p for p in root.glob("seed_*") if p.is_dir()} != seed_directories:
        raise ValueError("Missing or extra validation job directories")
    for directory in seed_directories:
        if {p.name for p in directory.iterdir() if p.is_dir()} != set(variants):
            raise ValueError("Missing or extra validation job directories")
    csvs = {root / f"seed_{s}" / f"{name}_validation.csv" for s, name in jobs}
    summaries = {root / f"seed_{s}" / f"{name}_summary.json" for s, name in jobs}
    checkpoints = {root / f"seed_{s}" / name / "best.pt" for s, name in jobs}
    if (set(root.glob("seed_*/*_validation.csv")) != csvs
            or set(root.glob("seed_*/*_summary.json")) != summaries
            or set(root.glob("seed_*/*/best.pt")) != checkpoints):
        raise ValueError("Missing or extra validation job artifacts")
    metrics, results, log_hashes = {}, [], {}
    for s, name in sorted(jobs):
        variant = variants[name]
        directory = root / f"seed_{s}"
        saved = _read(directory / f"{name}_summary.json")
        if saved != indexed[s, name] or saved.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError("Per-job summary differs from root summary or fingerprint")
        if (saved.get("seed") != s or saved.get("variant") != name or saved.get("split") != "validation"
                or saved.get("control") != variant["control"]):
            raise ValueError("Summary seed, variant, control or split differs")
        csv_path, checkpoint_path = directory / f"{name}_validation.csv", directory / name / "best.pt"
        if (saved["per_user_sha256"] != file_digest(csv_path)
                or saved["checkpoint_sha256"] != file_digest(checkpoint_path)):
            raise ValueError("Validation CSV or checkpoint checksum differs")
        local_cfg = _variant_config(cfg, variant)
        _same(_optimization_config(local_cfg), saved.get("optimization_configuration"), "Effective optimization configuration")
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        _check_checkpoint(checkpoint, identity, manifest["fingerprint"], _artifact(manifest, s, variant),
                          model_config(local_cfg, variant["control"], identity["items"]))
        if (checkpoint["selection_k"] != cfg["selection_k"] or checkpoint["selected_epoch"] > cfg["epochs"]
                or checkpoint["selected_epoch"] != saved["selected_epoch"]):
            raise ValueError("Checkpoint selection epoch or cutoff differs")
        rows = _rows(csv_path, cfg, records, s, variant, identity["items"])
        actual = summarize_multitarget_rows(rows, cfg["ks"])
        _same(actual, saved["validation"], "Validation summary")
        for key in ("selected_validation_metric", f"selected_validation_ndcg{cfg['selection_k']}"):
            _same(actual["all"][f"ndcg{cfg['selection_k']}"], saved[key], f"Summary {key}")
        _same(actual, _selected_log(directory / name, checkpoint, local_cfg, log_hashes), "Selected validation log")
        initial_path = directory / name / "initial_validation.json"
        _same(_read(initial_path), saved["initial_validation"], "Initial validation summary")
        log_hashes[str(initial_path)] = file_digest(initial_path)
        results.append({"seed": s, "variant": name, "validation": actual})
        metrics["candidate", name, s] = actual
    # Reference rankings were not exported in the old experiment. Check the
    # original selected-epoch validation trace rather than trusting the new JSON.
    ref_jobs = {(s, name) for s in search["seeds"] for name in loaded[s]["references"]}
    ref_saved = _index(_read(root / "reference_validation.json"), ref_jobs, "variant", "reference validation")
    ref_results = []
    counts = {g: {"users": sum(g == "all" or r["group"] == g for r in records),
                  "relevant_items": sum(len(set(r["validation"])) for r in records if g == "all" or r["group"] == g)}
              for g in ("all", "cold", "middle", "warm")}
    for s, name in sorted(ref_jobs):
        checkpoint = loaded[s]["references"][name]
        value = _selected_log(Path(search["reference_run_dir"]) / f"seed_{s}" / name,
                              checkpoint, cfg, log_hashes)
        saved = ref_saved[s, name]
        if saved["selected_epoch"] != checkpoint["selected_epoch"]:
            raise ValueError("Reference selected epoch differs")
        _same(value, saved["validation"], "Reference validation summary")
        for group in value:
            for key in ("users", "relevant_items"):
                _same(counts[group][key], value[group][key], "Reference validation population")
        ref_results.append({"seed": s, "variant": name, "validation": value})
        metrics["reference", name, s] = value
    selection_key = f"ndcg{cfg['selection_k']}"
    candidates = _aggregate(results, list(variants), selection_key)
    ref_names = sorted({name for _, name in ref_jobs})
    reference_aggregates = _aggregate(ref_results, ref_names, selection_key)
    expected_selection = {"fingerprint": manifest["fingerprint"], "split": "validation",
        "selection": manifest["selection"], "selected_variant": candidates[0]["variant"],
        "candidates": candidates, "references": reference_aggregates}
    _same(expected_selection, _read(root / "selection.json"), "Selection summary")
    keys = [f"{m}{k}" for k in cfg["ks"] for m in ("recall", "ndcg", "hit")]
    aggregates = []
    for role, names in (("candidate", list(variants)), ("reference", ref_names)):
        for name in names:
            for group in ("all", "cold", "middle", "warm"):
                if group not in metrics[role, name, search["seeds"][0]]:
                    continue
                for key in keys:
                    values = [metrics[role, name, s][group][key] for s in search["seeds"]]
                    aggregates.append({"role": role, "variant": name, "group": group, "metric": key,
                        "mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                        "seed_count": len(values), "seeds": search["seeds"],
                        "users_per_seed": metrics[role, name, search["seeds"][0]][group]["users"]})
    comparisons = []
    if (comparison is None) != (baseline is None):
        raise ValueError("Specify both comparison and baseline")
    if comparison is not None:
        def resolve(label):
            options = [(role, name) for role, names in (("candidate", variants), ("reference", ref_names))
                       for name in names if label in {name, f"{role}:{name}"}]
            if len(options) != 1:
                raise ValueError("Unknown or ambiguous comparison; qualify as candidate:name or reference:name")
            return options[0]
        left, right = resolve(comparison), resolve(baseline)
        if left == right:
            raise ValueError("Comparison and baseline must differ")
        for group in ("all", "cold", "middle", "warm"):
            if group not in metrics[*left, search["seeds"][0]]:
                continue
            for key in keys:
                differences = [metrics[*left, s][group][key] - metrics[*right, s][group][key] for s in search["seeds"]]
                comparisons.append({"comparison": ":".join(left), "baseline": ":".join(right), "group": group,
                    "metric": key, "seed_ids": search["seeds"], **paired_seed_comparison(differences, bootstrap_samples, seed)})
        _holm_adjust(comparisons)
    return {"protocol": PROTOCOL, "split": "validation", "cutoffs": cfg["ks"],
        "selected_variant": candidates[0]["variant"], "selection": manifest["selection"],
        "aggregation": "multi-positive metrics macro-averaged over users; training seeds are inference units",
        "uncertainty_scope": "training randomness conditional on the fixed validation population",
        "multiple_comparisons": "Holm adjustment across this analysis's reported group/metric family",
        "provenance": {"fingerprint": manifest["fingerprint"], "job_count": len(jobs),
            "all_artifact_hashes_verified": True, "candidate_metrics_recomputed_from_rankings": True,
            "prepared_targets_groups_counts_and_seen_exclusion_verified": True,
            "reference_metrics_verification": "original selected-epoch validation logs; old reference validation rankings were not exported",
            "validation_log_hashes": log_hashes,
            "analysis_source_hashes": {p.name: file_digest(p) for p in (Path(__file__), ROOT_ANALYSIS_HELPER)}},
        "aggregates": aggregates, "paired_comparisons": comparisons}


ROOT_ANALYSIS_HELPER = Path(__file__).resolve().with_name("analyze_results.py")


def write_analysis(report, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out / "analysis.json", report)
    with (out / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["aggregates"][0]))
        writer.writeheader()
        writer.writerows(report["aggregates"])
    table = {}
    for row in report["aggregates"]:
        key = row["role"], row["variant"], row["group"]
        entry = table.setdefault(key, {"role": key[0], "variant": key[1], "group": key[2],
                                      "seeds": row["seed_count"], "users_per_seed": row["users_per_seed"]})
        sd = "n/a" if row["sample_std"] is None else f"{row['sample_std']:.6f}"
        entry[row["metric"]] = f"{row['mean']:.6f} +/- {sd}"
    fields = ["role", "variant", "group", "seeds", "users_per_seed"] + [f"{m}{k}" for k in report["cutoffs"] for m in ("recall", "ndcg", "hit")]
    with (out / "paper_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--comparison")
    parser.add_argument("--baseline")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    report = analyze_runs(args.run_dir, args.comparison, args.baseline, args.bootstrap_samples)
    write_analysis(report, args.output_dir or args.run_dir / "analysis")
    print(json.dumps({"jobs": report["provenance"]["job_count"], "selected_variant": report["selected_variant"],
                      "output": str(args.output_dir or args.run_dir / "analysis")}))


if __name__ == "__main__":
    main()
