"""Explicitly replay a validation-selected transfer checkpoint."""
import argparse
from pathlib import Path

from fafirec.ratio_runner import evaluate, write_rows
from fafirec.runner import json_dump
from fafirec.validation_runner import load_validation_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new replay output directory")
    model, records, diagnostics, cfg, manifest = load_validation_checkpoint(
        args.run_dir, args.seed, args.variant, args.device)
    rows, summary = evaluate(model, records, args.split, diagnostics, cfg, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_user.csv", rows)
    json_dump(args.output_dir / "metrics.json", {"fingerprint": manifest["fingerprint"],
        "seed": args.seed, "variant": args.variant, "split": args.split, "metrics": summary,
        "candidate_policy": cfg["candidate_policy"], "target_mode": "multi_positive_block"})
    print(summary)


if __name__ == "__main__":
    main()
