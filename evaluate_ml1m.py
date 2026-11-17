"""Reload selected ML-1M checkpoints with multi-positive evaluation."""
import argparse
from pathlib import Path

from fafirec.ratio_runner import evaluate, json_dump, load_selected, write_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control", default="diagnostic")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    model, records, diagnostics, cfg, manifest = load_selected(args.run_dir, args.seed, args.control, args.device)
    rows, summary = evaluate(model, records, args.split, diagnostics, cfg, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_user.csv", rows)
    json_dump(args.output_dir / "metrics.json", {"fingerprint": manifest["fingerprint"],
        "seed": args.seed, "control": args.control, "split": args.split, "metrics": summary,
        "candidate_policy": cfg["candidate_policy"], "target_mode": "multi_positive_block"})
    print(summary)


if __name__ == "__main__":
    main()
