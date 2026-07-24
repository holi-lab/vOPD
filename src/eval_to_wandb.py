from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCALAR_KEYS = (
    "pass_at_n_pct",
    "average_at_n_pct",
    "majority_vote_at_n_pct",
    "format_rate",
    "num_problems",
    "total_solutions",
    "val_n",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--temperature", default="0.6")
    parser.add_argument("--output_file", default=None, help="Where the evaluator writes its JSON summary.")
    parser.add_argument("--wandb_run_id", default=None, help="Existing run to resume; omit to skip wandb.")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "OPD_TRL"))
    parser.add_argument("--prefix", default="eval", help="Metric name prefix.")
    parser.add_argument("--skip_eval", action="store_true", help="Only log an existing --output_file.")
    return parser


def run_eval(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "evaluate.py"),
        "--config", args.config,
        "--datasets", args.dataset,
        "--temperature", str(args.temperature),
        "--checkpoint_dir", args.checkpoint_dir,
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def find_summary(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    candidates = sorted(
        Path(PROJECT_ROOT).rglob(f"*{args.dataset}*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No eval JSON found for dataset {args.dataset}; pass --output_file explicitly."
        )
    return candidates[-1]


def main() -> None:
    args = build_parser().parse_args()

    if not args.skip_eval:
        run_eval(args)

    summary_path = find_summary(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"loaded summary: {summary_path}", flush=True)

    metrics = {
        f"{args.prefix}/{args.dataset}/{key}": summary[key]
        for key in SCALAR_KEYS
        if key in summary
    }
    for key, value in metrics.items():
        print(f"  {key} = {value}")

    if not args.wandb_run_id:
        print("no --wandb_run_id given; not logging to wandb")
        return

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume="must",
        settings=wandb.Settings(silent=True),
    )
    run.summary.update(metrics)
    run.log(metrics)
    run.finish()
    print(f"logged {len(metrics)} metrics to wandb run {args.wandb_run_id}", flush=True)


if __name__ == "__main__":
    main()
