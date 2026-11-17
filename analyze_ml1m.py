"""Analyze complete multi-positive ML-1M runs without single-target rank shortcuts."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_results import _holm_adjust, paired_seed_comparison
from fafirec.multitarget_metrics import summarize_multitarget_rows
from fafirec.runner import digest, file_digest, json_dump


def _valid_ids(value, n_items, name):
    try:
        ids = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {name} JSON") from error
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or any(type(item) is not int or not 1 <= item <= n_items for item in ids)):
        raise ValueError(f"{name} must contain distinct nonpadding catalog IDs")
    return ids


def _read_rows(path, cfg, identity, seed, control):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError(f"Empty per-user output: {path}")
    rows, profiles = [], {}
    ks = cfg["ks"]
    for value in raw:
        if value.get("split") != "test" or value.get("control") != control or value.get("seed") != str(seed):
            raise ValueError("CSV split, control, or seed differs from its recorded job")
        user = value.get("user")
        if not user or user in profiles:
            raise ValueError("Duplicate or missing per-user identity")
        if value.get("candidate_policy") != cfg["candidate_policy"]:
            raise ValueError("Mixed candidate policy in per-user output")
        targets = _valid_ids(value.get("targets"), identity["items"], "targets")
        top = _valid_ids(value.get("top_items"), identity["items"], "top_items")
        if len(top) > max(ks):
            raise ValueError("Stored ranking exceeds the declared maximum cutoff")
        try:
            count, declared, training = int(value["target_count"]), int(value["relevant_count"]), int(value["training_interactions"])
        except (ValueError, KeyError) as error:
            raise ValueError("Missing integer target/training counts") from error
        if count != len(targets) or declared != count or training < 2:
            raise ValueError("Relevant target count or training eligibility differs")
        row = {**value, "seed": seed, "target_count": count, "relevant_count": count}
        target_set = set(targets)
        for k in ks:
            relevant = np.array([item in target_set for item in top[:k]], dtype=float)
            hits = float(relevant.sum())
            dcg = float(np.sum(relevant / np.log2(np.arange(2, len(relevant) + 2))))
            idcg = float(np.sum(1. / np.log2(np.arange(2, min(k, count) + 2))))
            expected = {f"recall{k}": hits / count, f"ndcg{k}": dcg / idcg, f"hit{k}": float(hits > 0)}
            for key, computed in expected.items():
                try:
                    metric = float(value[key])
                except (ValueError, KeyError) as error:
                    raise ValueError(f"Missing numeric metric {key}") from error
                if not math.isfinite(metric) or not math.isclose(metric, computed, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Stored {key} differs from the multi-positive ranking")
                row[key] = metric
        profiles[user] = (value.get("group"), tuple(sorted(targets)), training)
        rows.append(row)
    if set(profiles) != set(identity["user_order"]):
        raise ValueError("Evaluation population differs from the run manifest")
    return rows, profiles


def _check_summary(actual, declared):
    if set(actual) != set(declared):
        raise ValueError("CSV groups differ from the saved metric summary")
    for group, values in actual.items():
        if set(values) != set(declared[group]):
            raise ValueError("CSV metric keys differ from the saved summary")
        for key, value in values.items():
            if not math.isclose(value, declared[group][key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Saved summary differs from recomputed per-user metrics")


def analyze_runs(run_dir, comparison="diagnostic", baseline="uniform", bootstrap_samples=10000, seed=2025):
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("protocol") != "ml1m_80_10_10_multi_positive":
        raise ValueError("Expected the ratio-split multi-positive experiment, not single-target rank CSVs")
    cfg, identity = manifest["config"], manifest["identity"]
    recomputed = digest({"config": cfg, "identity": identity,
                         "source_hashes": manifest["source"]["file_hashes"], "runtime": manifest["runtime"]})
    if manifest["fingerprint"] != recomputed:
        raise ValueError("Run fingerprint is inconsistent with saved provenance")
    jobs = {(training_seed, control) for training_seed in cfg["seeds"] for control in cfg["controls"]}
    summaries = json.loads((root / "summary.json").read_text(encoding="utf-8-sig"))
    indexed = {(result["seed"], result["control"]): result for result in summaries}
    if len(indexed) != len(summaries) or set(indexed) != jobs:
        raise ValueError("Complete configured seed/control grid is required")
    expected_csvs = {root / f"seed_{s}" / f"{c}_per_user.csv" for s, c in jobs}
    if set(root.glob("seed_*/*_per_user.csv")) != expected_csvs:
        raise ValueError("Missing or extra per-user experiment jobs")
    metrics, reference_profiles = {}, None
    for training_seed, control in sorted(jobs):
        directory = root / f"seed_{training_seed}"
        saved = json.loads((directory / f"{control}_summary.json").read_text(encoding="utf-8-sig"))
        if saved != indexed[(training_seed, control)] or saved["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Per-job summary differs from the root experiment record")
        for name, checksum in ((f"{control}_per_user.csv", "per_user_sha256"),
                               (f"{control}/best.pt", "checkpoint_sha256"), ("diagnostics.pt", "diagnostics_sha256")):
            if saved[checksum] != file_digest(directory / name):
                raise ValueError(f"Recorded artifact checksum differs: {name}")
        rows, profiles = _read_rows(directory / f"{control}_per_user.csv", cfg, identity, training_seed, control)
        if reference_profiles is None:
            reference_profiles = profiles
        elif profiles != reference_profiles:
            raise ValueError("Paired users, relevant targets, groups, or training counts differ")
        value = summarize_multitarget_rows(rows, cfg["ks"])
        _check_summary(value, saved["test"])
        metrics[(training_seed, control)] = value
    keys = [f"{name}{k}" for k in cfg["ks"] for name in ("recall", "ndcg", "hit")]
    aggregates = []
    for control in cfg["controls"]:
        for group in ("all", "cold", "middle", "warm"):
            if group not in metrics[(cfg["seeds"][0], control)]:
                continue
            for key in keys:
                values = np.asarray([metrics[(s, control)][group][key] for s in cfg["seeds"]])
                aggregates.append({"control": control, "group": group, "metric": key,
                    "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "seed_count": len(values), "seeds": cfg["seeds"],
                    "users_per_seed": metrics[(cfg["seeds"][0], control)][group]["users"]})
    comparisons = []
    if comparison == baseline:
        raise ValueError("Comparison and baseline must differ")
    if comparison in cfg["controls"] and baseline in cfg["controls"]:
        for group in ("all", "cold", "middle", "warm"):
            if group not in metrics[(cfg["seeds"][0], comparison)]:
                continue
            for key in keys:
                differences = [metrics[(s, comparison)][group][key] - metrics[(s, baseline)][group][key] for s in cfg["seeds"]]
                comparisons.append({"comparison": comparison, "baseline": baseline, "group": group,
                    "metric": key, "seed_ids": cfg["seeds"],
                    **paired_seed_comparison(differences, bootstrap_samples, seed)})
        _holm_adjust(comparisons)
    return {"protocol": manifest["protocol"], "status": manifest["status"], "cutoffs": cfg["ks"],
            "aggregation": "multi-positive metrics macro-averaged over users; training seeds are inference units",
            "uncertainty_scope": "training randomness conditional on this fixed evaluation population",
            "multiple_comparisons": "Holm adjustment across the reported group/metric family",
            "provenance": {"job_count": len(jobs), "all_artifact_hashes_verified": True,
                           "candidate_policy": cfg["candidate_policy"], "fingerprint": manifest["fingerprint"]},
            "aggregates": aggregates, "paired_comparisons": comparisons}


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
        key = (row["control"], row["group"])
        entry = table.setdefault(key, {"control": key[0], "group": key[1],
                                      "seeds": row["seed_count"], "users_per_seed": row["users_per_seed"]})
        sd = "n/a" if row["sample_std"] is None else f"{row['sample_std']:.6f}"
        entry[row["metric"]] = f"{row['mean']:.6f} +/- {sd}"
    fields = ["control", "group", "seeds", "users_per_seed"] + [f"{m}{k}" for k in report["cutoffs"] for m in ("recall", "ndcg", "hit")]
    with (out / "paper_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--comparison", default="diagnostic")
    parser.add_argument("--baseline", default="uniform")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    report = analyze_runs(args.run_dir, args.comparison, args.baseline, args.bootstrap_samples)
    write_analysis(report, args.output_dir or args.run_dir / "analysis")
    print(json.dumps({"jobs": report["provenance"]["job_count"], "output": str(args.output_dir or args.run_dir / 'analysis')}))


if __name__ == "__main__":
    main()
