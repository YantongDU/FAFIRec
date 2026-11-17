"""Reload a selected checkpoint, verify its inputs, and reproduce exact ranks."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fafirec.certificates import ranking_certificate
from fafirec.data import evaluation_batch
from fafirec.model import FAFIRecModel
from fafirec.runner import (digest, evaluate, file_digest, json_dump, load_inputs, resolve_device,
                            source_identity, tensor_digest, validate_diagnostics_cache, write_rows)


def load_selected(run_dir, seed, control, device=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if source_identity()["file_hashes"] != manifest["source"]["file_hashes"]:
        raise ValueError("Current source differs from the recorded experiment source")
    cfg = manifest["config"].copy()
    if device:
        cfg["device"] = device
    torch.set_num_threads(cfg["threads"])
    records, features, _, identity = load_inputs(cfg)
    if identity != manifest["identity"]:
        raise ValueError("Evaluation inputs differ from the recorded item features or user split")
    checkpoint_path = run_dir / f"seed_{seed}" / control / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity or checkpoint["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Selected checkpoint does not match this experiment")
    artifact = checkpoint.get("artifact", {})
    if (artifact.get("seed") != seed or artifact.get("control") != control or artifact.get("stage") != "student"
            or checkpoint["model_config"]["control"] != control):
        raise ValueError("Checkpoint seed, control, or training stage differs from the requested artifact")
    summary_path = run_dir / f"seed_{seed}" / f"{control}_summary.json"
    if summary_path.exists():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result["fingerprint"] != manifest["fingerprint"] or result["checkpoint_sha256"] != file_digest(checkpoint_path):
            raise ValueError("Checkpoint file differs from the completed experiment record")
    diagnostics = torch.load(run_dir / f"seed_{seed}" / "diagnostics.pt", map_location="cpu", weights_only=True)
    teacher_checkpoint = torch.load(run_dir / f"seed_{seed}" / "teacher" / "best.pt", map_location="cpu", weights_only=True)
    if teacher_checkpoint["identity"] != identity or teacher_checkpoint["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Teacher checkpoint input identity differs")
    teacher_hash = digest({key: tensor_digest(value) for key, value in teacher_checkpoint["model"].items()})
    if teacher_checkpoint.get("artifact") != {"seed": seed, "control": "cf", "stage": "teacher"}:
        raise ValueError("Teacher checkpoint seed or stage differs")
    diag = validate_diagnostics_cache(diagnostics, identity, teacher_hash)
    if (artifact["teacher_sha256"] != teacher_hash or artifact["diagnostics_tensor_sha256"] != tensor_digest(diag)
            or artifact["diagnostics_file_sha256"] != file_digest(run_dir / f"seed_{seed}" / "diagnostics.pt")):
        raise ValueError("Student checkpoint was fitted with different diagnostics or teacher")
    model = FAFIRecModel(text_features=features, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(resolve_device(cfg)).eval()
    return model, records, diag, cfg, manifest


@torch.no_grad()
def audit_rankings(model, records, diagnostics, cfg, split, users, seed, k):
    """Fixed-checkpoint reference audit; the audit subset is explicitly recorded."""
    if not 1 <= k < model.n_items:
        raise ValueError("Ranking audit needs 1 <= k < catalog size")
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=min(users, len(records)), replace=False))
    collaborative_items = F.normalize(model.item_id_table.weight, dim=-1).cpu()
    fused_items = model._item_vectors().cpu()
    rows = []
    for index in indices:
        hist, target = evaluation_batch(records, [index], split, cfg["max_len"])
        branches = model.encode_branches(hist.to(device), diagnostics[[index]].to(device))
        valid = hist.to(device).ne(0)
        c_user = model._last(branches["collaborative"], valid)[0].cpu()
        z_user = model._last(branches["fused"], valid)[0].cpu()
        candidates = np.arange(1, model.n_items + 1)
        if cfg["candidate_policy"] != "full":
            observed = set(records[index]["train"] + ([records[index]["validation"]] if split == "test" else []))
            if cfg["candidate_policy"] == "exclude_seen_restore_target":
                observed.discard(int(target[0]))
            elif int(target[0]) in observed:
                raise ValueError("Audit target is excluded under declared candidate policy")
            candidates = candidates[~np.isin(candidates, list(observed))]
        result = ranking_certificate(c_user, z_user, collaborative_items, fused_items,
                                     k=k, candidate_ids=candidates, target=int(target[0]),
                                     score_scale=model.score_scale)
        rows.append({"user": records[index]["user"], "group": records[index]["group"], **result})
    return {"reference": "current checkpoint collaborative branch; not an independently trained CF baseline",
            "selection": "uniform user sample without replacement", "sample_seed": seed,
            "audit_users": len(rows), "population_users": len(records), "k": k,
            "candidate_policy": cfg["candidate_policy"],
            "scope": "full declared catalog for each sampled user; sufficient certificates only", "users": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control", default="diagnostic")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-users", type=int, default=0)
    parser.add_argument("--audit-k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.audit_users < 0:
        parser.error("--audit-users must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    model, records, diag, cfg, manifest = load_selected(args.run_dir, args.seed, args.control, args.device)
    rows, metrics = evaluate(model, records, args.split, diag, cfg, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_user.csv", rows)
    json_dump(args.output_dir / "metrics.json", {"fingerprint": manifest["fingerprint"],
                                               "seed": args.seed, "control": args.control,
                                               "split": args.split, "device": str(next(model.parameters()).device), "metrics": metrics})
    if args.audit_users:
        audit = audit_rankings(model, records, diag, cfg, args.split, args.audit_users, args.seed, args.audit_k)
        json_dump(args.output_dir / "ranking_audit.json", audit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
