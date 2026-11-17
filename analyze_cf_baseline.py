"""Lock a stronger CF baseline by validation, then finalize its test comparison.

Importing/compiling this file does not inspect experiment artifacts. Execution
requires completed, analyzed five-seed main/search runs and all 15 main replays.
Before the choice lock, only validation scores and integrity metadata are used:
the frozen reference loader parses old job records for binding/checksums, but no
test metric field is accessed by the selection code. Test evaluation and metric
inspection occur only after the immutable validation-choice lock is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


CONTROLS = ("cf", "uniform", "diagnostic")
GROUPS = ("all", "cold", "middle", "warm")
METRICS = ("recall10", "ndcg10", "hit10", "recall50", "ndcg50", "hit50")
CANDIDATE = "cf_lr1e3"
RULE = "Select the higher five-seed mean validation NDCG@50; exact ties retain the matched main CF. No test metric is used."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def index(rows, jobs, name, label):
    require(isinstance(rows, list), label + " is not a list")
    try:
        result = {(row["seed"], row[name]): row for row in rows}
    except (KeyError, TypeError) as error:
        raise ValueError(label + " has invalid job identities") from error
    require(all(type(row.get("seed")) is int for row in rows)
            and len(result) == len(rows) and set(result) == jobs,
            label + " has missing, duplicate or extra jobs")
    return result


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an existing lock or result from being replaced.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def test_metrics(path, records, cfg, seed, control, same, ids):
    """Independently check test rankings against the actual prepared population."""
    prepared = {record["user"]: record for record in records}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    seen, totals = set(), {}
    n_items = cfg["n_items"]
    for row in raw:
        user = row.get("user")
        require(user in prepared and user not in seen, "Unknown or duplicate test user")
        seen.add(user)
        record = prepared[user]
        require(row.get("split") == "test" and row.get("seed") == str(seed)
                and row.get("control") == control, "Test CSV job identity differs")
        require(row.get("candidate_policy") == cfg["candidate_policy"]
                and row.get("group") == record["group"], "Test group/candidate convention differs")
        targets = set(ids(row.get("targets"), n_items, "test targets"))
        require(targets == set(record["test"]), "Test targets differ from prepared inputs")
        require(int(row["training_interactions"]) == len(record["train"])
                and int(row["target_count"]) == int(row["relevant_count"]) == len(targets),
                "Test target/training counts differ")
        top = ids(row.get("top_items"), n_items, "test ranking")
        observed = set(record["train"]) | set(record["validation"])
        if cfg["candidate_policy"] == "exclude_seen":
            require(not (targets & observed) and not (set(top) & observed), "Seen-history exclusion differs")
            available = n_items - len(observed)
        else:
            require(cfg["candidate_policy"] == "full", "Unexpected candidate policy")
            available = n_items
        require(len(top) == min(max(cfg["ks"]), available), "Incomplete eligible-catalog test ranking")
        values = {}
        for k in cfg["ks"]:
            ranked = top[:k]
            hits = sum(item in targets for item in ranked)
            dcg = sum(1. / math.log2(rank + 2) for rank, item in enumerate(ranked) if item in targets)
            ideal = sum(1. / math.log2(rank + 2) for rank in range(min(k, len(targets))))
            for metric, value in ((f"recall{k}", hits / len(targets)), (f"ndcg{k}", dcg / ideal),
                                   (f"hit{k}", float(hits > 0))):
                same(value, float(row[metric]), "Test ranking " + metric)
                values[metric] = value
        for group in ("all", record["group"]):
            total = totals.setdefault(group, {"users": 0, "relevant_items": 0, **{m: 0. for m in METRICS}})
            total["users"] += 1
            total["relevant_items"] += len(targets)
            for metric, value in values.items():
                total[metric] += value
    require(seen == set(prepared), "Test CSV does not cover the full population")
    require(set(totals) == set(GROUPS), "Expected all four nonempty user groups")
    return {group: {key: value if key in {"users", "relevant_items"} else value / totals[group]["users"]
                    for key, value in values.items()} for group, values in totals.items()}, len(raw)


def execute(args):
    project = args.project_dir.resolve()
    sys.path.insert(0, str(project))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import analyze_ml1m
    import analyze_transfer
    from analyze_results import _holm_adjust, paired_seed_comparison
    from fafirec import ratio_runner, validation_runner
    from fafirec.runner import digest, file_digest, resolve_device, runtime_identity

    same = analyze_transfer._same
    sha = file_digest
    main = resolve(project, args.main_run, "runs/ml1m_stable_transfer_v2")
    search = resolve(project, args.search_run, "runs/cf_lr_validation_v2")
    main_config = resolve(project, args.main_config, "data/ml1m_stable_transfer_local.json")
    config_lock = resolve(project, args.config_lock, "data/final_transfer_selection_lock.json")
    search_config = resolve(project, args.search_config, "configs/ml1m_cf_validation.json")
    choice_path = resolve(project, args.selection_lock, "data/cf_baseline_selection_lock.json")
    destination = main / "analysis/strong_cf_comparison.json"
    require(main != search, "Main and search runs must differ")
    required = [main / "manifest.json", main / "summary.json", main / "analysis/analysis.json",
                main / "analysis_vs_cf/analysis.json", main / "analysis/reload_verification.json",
                search / "manifest.json", search / "summary.json", search / "selection.json",
                search / "analysis/analysis.json", main_config, config_lock, search_config]
    require(all(path.is_file() for path in required), "Both completed analyses, main15 replays and config locks are required")
    require(not destination.exists(), "Strong-CF comparison already exists; inspect it instead of overwriting")
    captured = {str(path): sha(path) for path in required}
    main_manifest, search_manifest = read(main / "manifest.json"), read(search / "manifest.json")
    cfg, declared_lock = main_manifest["config"], read(config_lock)
    require(main_manifest["protocol"] == "ml1m_80_10_10_multi_positive", "Unexpected main protocol")
    fingerprint = digest({"config": cfg, "identity": main_manifest["identity"],
                          "source_hashes": main_manifest["source"]["file_hashes"], "runtime": main_manifest["runtime"]})
    require(fingerprint == main_manifest["fingerprint"], "Main fingerprint differs")
    require(ratio_runner.source_identity()["file_hashes"] == main_manifest["source"]["file_hashes"], "Main source changed")
    same(ratio_runner.load_config(main_config), cfg, "Locked main configuration")
    require(declared_lock["selection_split"] == "validation"
            and declared_lock["new_test_scores_used_for_this_selection"] is False
            and declared_lock["final_local_config_sha256"] == sha(main_config), "Main validation configuration lock differs")
    require(Path(cfg["output_dir"]).resolve() == main, "Main output directory differs")
    seeds = cfg["seeds"]
    require(len(seeds) == len(set(seeds)) == 5 and all(type(s) is int for s in seeds), "Exactly five integer seeds are required")
    require(len(cfg["controls"]) == 3 and set(cfg["controls"]) == set(CONTROLS)
            and cfg["ks"] == [10, 50] and cfg["selection_k"] == 50, "Unexpected main controls or cutoffs")
    jobs = {(s, c) for s in seeds for c in CONTROLS}
    # These records are inspected for integrity metadata only until the lock.
    main_records = index(read(main / "summary.json"), jobs, "control", "Main summaries")
    require(set(main.glob("seed_*/*_per_user.csv")) == {main / f"seed_{s}/{c}_per_user.csv" for s, c in jobs},
            "Main per-user grid differs")
    for path in (main / "analysis/analysis.json", main / "analysis_vs_cf/analysis.json"):
        metadata = read(path)
        require(metadata["protocol"] == main_manifest["protocol"] and metadata["cutoffs"] == cfg["ks"]
                and metadata["provenance"]["fingerprint"] == fingerprint
                and metadata["provenance"]["job_count"] == 15
                and metadata["provenance"]["all_artifact_hashes_verified"] is True,
                "Main analysis metadata is incomplete")
    replays = read(main / "analysis/reload_verification.json")
    require(replays["fingerprint"] == fingerprint and replays["device"] == cfg["device"]
            and replays["all_identical"] is True, "Main replay provenance differs")
    replay_jobs = index(replays["jobs"], jobs, "control", "Main15 replays")
    for (s, control), record in main_records.items():
        csv_path = main / f"seed_{s}/{control}_per_user.csv"
        checkpoint_path = main / f"seed_{s}/{control}/best.pt"
        replay_path = main / f"reloaded_all/seed_{s}/{control}/per_user.csv"
        replay = replay_jobs[s, control]
        require(record["fingerprint"] == fingerprint and record["checkpoint_sha256"] == sha(checkpoint_path)
                and record["diagnostics_sha256"] == sha(main / f"seed_{s}/diagnostics.pt")
                and record["per_user_sha256"] == sha(csv_path) == sha(replay_path)
                == replay["original_sha256"] == replay["reloaded_sha256"]
                and replay["identical"] is True and replay["metrics_identical"] is True,
                "Main checkpoint/CSV/cache or replay checksum differs")
    search_spec = validation_runner.load_config(search_config)
    require(Path(search_spec["reference_run_dir"]).resolve() == main
            and Path(search_spec["output_dir"]).resolve() == search and search_spec["seeds"] == seeds,
            "CF validation search does not use this full main run")
    same(search_spec, search_manifest["search_config"], "Predeclared CF search configuration")
    require(len(search_spec["variants"]) == 1 and search_spec["variants"][0]["name"] == CANDIDATE
            and search_spec["variants"][0]["control"] == "cf", "Expected the single predeclared CF candidate")
    candidate = search_spec["variants"][0]
    effective = validation_runner._variant_config(search_manifest["training_config"], candidate)
    require(effective["lr"] == .001 and effective["min_epochs"] == 30, "CF candidate optimization differs")
    require(validation_runner.source_identity()["file_hashes"] == search_manifest["source_hashes"], "Search source changed")
    require(search_manifest["fingerprint"] == validation_runner._fingerprint(search_manifest), "Search fingerprint differs")
    torch.set_num_threads(cfg["threads"])
    actual_runtime = runtime_identity(resolve_device(cfg))
    same(actual_runtime, main_manifest["runtime"], "Main evaluation runtime")
    same(actual_runtime, search_manifest["runtime"], "Search evaluation runtime")
    records, features, identity, _, reference, _ = validation_runner.load_reference(search_spec)
    del features
    same(identity, main_manifest["identity"], "Main/search data identity")
    same(reference, search_manifest["reference"], "Main/search teacher and diagnostic provenance")
    require(len(records) == 5950, "This finalized ML-1M contract expects 5,950 users")
    audit = analyze_transfer.analyze_runs(search, comparison=CANDIDATE, baseline="reference:cf")
    same(audit, read(search / "analysis/analysis.json"), "Verified CF validation analysis")
    require(audit["provenance"]["job_count"] == 5 and audit["provenance"]["all_artifact_hashes_verified"] is True,
            "CF validation search is not a complete audited five-seed grid")
    def validation_mean(role, name):
        matching = [r for r in audit["aggregates"] if r["role"] == role and r["variant"] == name
                    and r["group"] == "all" and r["metric"] == "ndcg50"]
        require(len(matching) == 1 and matching[0]["seed_count"] == 5, "Missing five-seed validation mean")
        return matching[0]["mean"]
    candidate_mean = validation_mean("candidate", CANDIDATE)
    matched_mean = validation_mean("reference", "cf")
    selected = CANDIDATE if candidate_mean > matched_mean else "reference:cf"
    if selected == CANDIDATE:
        require(not (search / "test_selected").exists(), "CF test_selected directory already exists; no output is overwritten")
    payload = {"schema": "strong_cf_validation_selection_v1", "selection_split": "validation",
        "selection_metric": "five-seed mean all NDCG@50", "selected": selected,
        "candidate_validation_mean": candidate_mean, "matched_cf_validation_mean": matched_mean,
        "rule": RULE, "seeds": seeds, "main_fingerprint": fingerprint,
        "search_fingerprint": search_manifest["fingerprint"], "main_run_dir": str(main), "search_run_dir": str(search),
        "artifact_sha256": captured, "script_sha256": sha(Path(__file__)),
        "main_source_hashes": main_manifest["source"]["file_hashes"],
        "search_source_hashes": search_manifest["source_hashes"],
        "new_test_metrics_used_for_baseline_selection": False}
    if choice_path.exists():
        existing = read(choice_path)
        same(payload, {key: existing[key] for key in payload}, "Existing immutable CF selection lock")
        require(set(existing) == set(payload) | {"created_utc"}, "Unexpected CF lock fields")
    else:
        write_once(choice_path, {**payload, "created_utc": datetime.now(timezone.utc).isoformat()})
    lock_sha = sha(choice_path)
    print(json.dumps({"selection_locked": selected, "lock_sha256": lock_sha,
                      "selection_split": "validation"}), flush=True)

    # Test-score access starts here, strictly after the validated choice lock.
    for filename, checksum in captured.items():
        require(sha(filename) == checksum, "A captured experiment artifact changed after selection lock")
    for baseline_name, directory in (("uniform", "analysis"), ("cf", "analysis_vs_cf")):
        audited_main = analyze_ml1m.analyze_runs(main, comparison="diagnostic", baseline=baseline_name)
        same(audited_main, read(main / directory / "analysis.json"), "Recomputed main test analysis")
    scoring_cfg = {**cfg, "n_items": identity["items"]}
    full_test, selected_test, checked = {}, [], 0
    for s in seeds:
        full_metrics, full_count = test_metrics(main / f"seed_{s}/diagnostic_per_user.csv", records, scoring_cfg,
                                               s, "diagnostic", same, analyze_transfer._ids)
        same(full_metrics, main_records[s, "diagnostic"]["test"], "Main full-method test summary")
        require(full_count == len(records), "Full-method test population differs")
        full_test[s] = full_metrics
        if selected == CANDIDATE:
            model, actual_records, diagnostics, actual_cfg, loaded_manifest = validation_runner.load_validation_checkpoint(search, s, CANDIDATE)
            require(loaded_manifest["fingerprint"] == search_manifest["fingerprint"], "Loaded candidate fingerprint differs")
            rows, result_metrics = ratio_runner.evaluate(model, actual_records, "test", diagnostics, actual_cfg, s)
            output = search / f"test_selected/seed_{s}"
            output.mkdir(parents=True, exist_ok=False)
            csv_path = output / "per_user.csv"
            ratio_runner.write_rows(csv_path, rows)
            del model
            reloaded, actual_records, diagnostics, actual_cfg, _ = validation_runner.load_validation_checkpoint(search, s, CANDIDATE)
            replay_rows, replay_metrics = ratio_runner.evaluate(reloaded, actual_records, "test", diagnostics, actual_cfg, s)
            replay_path = output / "replayed_per_user.csv"
            ratio_runner.write_rows(replay_path, replay_rows)
            identical = csv_path.read_bytes() == replay_path.read_bytes() and result_metrics == replay_metrics
            require(identical, "Independent CF replay differs")
            del reloaded
            metrics, count = test_metrics(csv_path, records, scoring_cfg, s, "cf", same, analyze_transfer._ids)
            same(metrics, result_metrics, "Fresh CF test ranking summary")
            checkpoint_path = search / f"seed_{s}/{CANDIDATE}/best.pt"
            summary_path = search / f"seed_{s}/{CANDIDATE}_summary.json"
            validation_job = read(summary_path)
            require(sha(checkpoint_path) == validation_job["checkpoint_sha256"], "Chosen CF checkpoint changed")
            job = {"seed": s, "test": result_metrics, "per_user_sha256": sha(csv_path),
                "checkpoint_sha256": sha(checkpoint_path), "reloaded_identical": True,
                "per_user_file": str(csv_path), "checkpoint_file": str(checkpoint_path),
                "reloaded_file": str(replay_path), "reloaded_sha256": sha(replay_path),
                "source_summary_sha256": sha(summary_path), "source_fingerprint": search_manifest["fingerprint"]}
            write_once(output / "metrics.json", {"fingerprint": search_manifest["fingerprint"], "seed": s,
                "control": "cf", "variant": CANDIDATE, "split": "test", "selection_lock_sha256": lock_sha,
                "candidate_policy": cfg["candidate_policy"], "metrics": result_metrics})
            write_once(output / "verification.json", {"seed": s, "original_sha256": sha(csv_path),
                "reloaded_sha256": sha(replay_path), "metrics_identical": True, "identical": True})
        else:
            csv_path = main / f"seed_{s}/cf_per_user.csv"
            checkpoint_path = main / f"seed_{s}/cf/best.pt"
            metrics, count = test_metrics(csv_path, records, scoring_cfg, s, "cf", same, analyze_transfer._ids)
            same(metrics, main_records[s, "cf"]["test"], "Selected matched CF test summary")
            job = {"seed": s, "test": main_records[s, "cf"]["test"], "per_user_sha256": sha(csv_path),
                "checkpoint_sha256": sha(checkpoint_path), "reloaded_identical": True,
                "per_user_file": str(csv_path), "checkpoint_file": str(checkpoint_path),
                "reloaded_file": str(main / f"reloaded_all/seed_{s}/cf/per_user.csv"),
                "reloaded_sha256": replay_jobs[s, "cf"]["reloaded_sha256"],
                "source_summary_sha256": sha(main / f"seed_{s}/cf_summary.json"), "source_fingerprint": fingerprint}
            require(job["per_user_sha256"] == main_records[s, "cf"]["per_user_sha256"]
                    and job["checkpoint_sha256"] == main_records[s, "cf"]["checkpoint_sha256"], "Matched CF selection differs")
        checked += count
        selected_test.append(job)
        print(json.dumps({"verified_strong_cf_seed": s, "selected": selected, "reloaded_identical": True}), flush=True)
    require(checked == 29750 and len(selected_test) == 5, "Strong-CF test population is incomplete")
    aggregates, paired = [], []
    for group in GROUPS:
        for metric in METRICS:
            values = [job["test"][group][metric] for job in selected_test]
            users = [job["test"][group]["users"] for job in selected_test]
            require(len(set(users)) == 1, "Strong-CF grouped user counts differ")
            aggregates.append({"control": "strong_cf", "group": group, "metric": metric,
                "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                "seed_count": 5, "seeds": seeds, "users_per_seed": users[0]})
            differences = [full_test[s][group][metric] - job["test"][group][metric]
                           for s, job in zip(seeds, selected_test)]
            paired.append({"comparison": "diagnostic", "baseline": "strong_cf", "group": group,
                "metric": metric, "seed_ids": seeds, **paired_seed_comparison(differences, 10000, 2025)})
    _holm_adjust(paired)
    report = {"complete": True, "selection": {"selected": selected,
        "candidate_validation_mean": candidate_mean, "matched_cf_validation_mean": matched_mean,
        "rule": RULE, "lock_sha256": lock_sha}, "source": {"main_fingerprint": fingerprint,
        "search_fingerprint": search_manifest["fingerprint"], "main_run_dir": str(main), "search_run_dir": str(search),
        "main_config_sha256": sha(main_config), "main_config_lock_sha256": sha(config_lock),
        "search_config_sha256": sha(search_config), "main_analysis_sha256": sha(main / "analysis/analysis.json"),
        "search_analysis_sha256": sha(search / "analysis/analysis.json"), "selection_lock_file": str(choice_path),
        "main15_reload_sha256": sha(main / "analysis/reload_verification.json"), "script_sha256": sha(Path(__file__))},
        "per_seed": selected_test, "aggregates": aggregates, "paired_comparisons": paired,
        "verification": {"all_5_reloaded_identical": True, "selected_baseline_is_main_cf": selected == "reference:cf",
                         "checked_test_rows": checked, "main15_replays_verified": True},
        "uncertainty_scope": "training seeds are the inference units, conditional on this fixed test population",
        "multiple_comparisons": "Holm adjustment over these 24 reported group/metric comparisons"}
    require(sha(choice_path) == lock_sha, "CF choice lock changed during finalization")
    write_once(destination, report)
    print(json.dumps({"complete": True, "selected": selected, "output": str(destination)}), flush=True)
    return report


def resolve(project, argument, default):
    path = Path(default if argument is None else argument).expanduser()
    return (project / path).resolve() if not path.is_absolute() else path.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parent = Path(__file__).resolve().parent
    default_project = parent if (parent / "fafirec").is_dir() else Path("E:/论文/FAFIRec_Improve_v2")
    parser.add_argument("--project-dir", type=Path, default=default_project)
    parser.add_argument("--main-run", type=Path)
    parser.add_argument("--search-run", type=Path)
    parser.add_argument("--main-config", type=Path)
    parser.add_argument("--config-lock", type=Path)
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--selection-lock", type=Path)
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    main()
