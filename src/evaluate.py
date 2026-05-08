from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from util import PROJECT_ROOT, get_dataset_spec, print_section, resolve_eval_base_model


MATH_DATASETS = {"math500", "aime24", "aime25", "minerva", "amc23"}
MCQ_DATASETS = {"chemistry", "gpqa-d"}


@dataclass(frozen=True)
class EvalTask:
    dataset: str
    route: str
    checkpoint_dir: str | None


def resolve_eval_config_path(config_path: str | Path | None) -> str | None:
    if config_path is None:
        return None

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    if not path.exists():
        raise FileNotFoundError(f"Eval config does not exist: {path}")
    return str(path)


def load_eval_config(config_path: str | Path | None) -> dict[str, Any]:
    resolved_path = resolve_eval_config_path(config_path)
    if resolved_path is None:
        raise ValueError("--config is required")

    import yaml

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    if raw_config is None:
        raise ValueError(f"Eval config is empty: {resolved_path}")
    if not isinstance(raw_config, dict):
        raise ValueError(f"Eval config must be a YAML mapping: {resolved_path}")
    return raw_config


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    value = str(value).strip()
    if value == "":
        return None
    return value


def split_dataset_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if clean_str(part) is not None]
    value = clean_str(value)
    if value is None:
        raise ValueError("Eval config 'datasets' is required")
    return [part for part in re.split(r"[\s,]+", value) if part]


def normalize_dataset_name(dataset: str) -> str:
    return dataset.strip().lower().replace("_", "-")


def route_dataset(dataset: str) -> str:
    normalized = normalize_dataset_name(dataset)
    if normalized in MCQ_DATASETS:
        return "mcq"
    if normalized in MATH_DATASETS or get_dataset_spec(dataset) is not None:
        return "math"
    raise ValueError(
        f"Unknown eval dataset '{dataset}'. Known math datasets: {sorted(MATH_DATASETS)}; "
        f"known MCQ datasets: {sorted(MCQ_DATASETS)}."
    )


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def resolve_path(path: str | Path, *, base_dir: str | Path = PROJECT_ROOT) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return Path(base_dir) / path_obj


def find_checkpoint_dirs(search_root: str | Path) -> list[Path]:
    root = resolve_path(search_root)
    if not root.exists():
        return []
    checkpoints = [path for path in root.rglob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=natural_key)


def find_checkpoint_step_dirs(checkpoint_dir: str | Path) -> list[Path]:
    root = resolve_path(checkpoint_dir)
    if not root.exists():
        return []
    if root.name.startswith("checkpoint-") and root.is_dir():
        return [root]
    checkpoints = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    return sorted(checkpoints, key=natural_key)


def find_latest_checkpoint_dir(search_root: str | Path) -> Path | None:
    checkpoints = find_checkpoint_dirs(search_root)
    return checkpoints[-1] if checkpoints else None


def resolve_checkpoint_dir(
    checkpoint_dir: str | None,
    checkpoint_step: str | None,
    checkpoint_root: str | Path,
) -> str:
    if checkpoint_dir is None:
        latest = find_latest_checkpoint_dir(checkpoint_root)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint-* directory found under: {checkpoint_root}")
        return str(latest)

    checkpoint_path = resolve_path(checkpoint_dir)
    if checkpoint_path.name.startswith("checkpoint-"):
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
        return str(checkpoint_path)

    if checkpoint_step:
        step_name = checkpoint_step if checkpoint_step.startswith("checkpoint-") else f"checkpoint-{checkpoint_step}"
        candidates = [checkpoint_path / checkpoint_step, checkpoint_path / step_name]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)

    latest = find_latest_checkpoint_dir(checkpoint_path)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint-* directory found under: {checkpoint_path}")
    return str(latest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified evaluator for math and MCQ datasets")
    parser.add_argument("--config", required=True, help="Eval YAML config path.")
    parser.add_argument("--all_checkpoints", action="store_true", help="Evaluate every checkpoint-* global step under --checkpoint_dir.")
    parser.add_argument("--datasets", required=True, help="Dataset list, separated by commas or spaces.")
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--checkpoint_step")
    parser.add_argument("--base_model")
    parser.add_argument("--dry_run", action="store_true", help="Print routed evaluator commands without running them.")
    return parser


def apply_runtime_env(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    os.environ["NCCL_P2P_DISABLE"] = str(runtime["nccl_p2p_disable"])


def build_tasks(config: dict[str, Any], args: argparse.Namespace) -> list[EvalTask]:
    datasets = split_dataset_list(args.datasets)
    checkpoint_root = resolve_path(args.checkpoint_root)
    checkpoint_dir = clean_str(args.checkpoint_dir)
    checkpoint_step = clean_str(args.checkpoint_step)
    if args.all_checkpoints:
        if checkpoint_dir is None:
            raise ValueError("--checkpoint_dir is required with --all_checkpoints")
        if checkpoint_step is not None:
            raise ValueError("--checkpoint_step cannot be used with --all_checkpoints")
        checkpoint_dirs = [str(path) for path in find_checkpoint_step_dirs(checkpoint_dir)]
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoint-* global-step directory found under: {checkpoint_dir}")
    elif checkpoint_dir is not None or checkpoint_step is not None or clean_str(args.base_model) is None:
        checkpoint_dirs = [resolve_checkpoint_dir(checkpoint_dir, checkpoint_step, checkpoint_root)]
    else:
        checkpoint_dirs = [None]

    tasks = []
    for checkpoint in checkpoint_dirs:
        for dataset in datasets:
            tasks.append(EvalTask(dataset=dataset, route=route_dataset(dataset), checkpoint_dir=checkpoint))
    return tasks


def build_eval_args(
    config: dict[str, Any],
    args: argparse.Namespace,
    task: EvalTask,
    task_count: int,
) -> list[str]:
    route_config = config[task.route]
    runtime_config = config["runtime"]
    base_model = clean_str(args.base_model)

    if task.checkpoint_dir is None and base_model is None:
        raise ValueError("--base_model is required for base-model evaluation")
    if task.checkpoint_dir is not None:
        base_model = resolve_eval_base_model(task.checkpoint_dir, base_model)

    output_file = clean_str(config["output_file"])
    if output_file is not None and task_count > 1:
        raise ValueError("--output_file can only be used when evaluating one dataset/checkpoint task")

    eval_args = [
        "--dataset",
        task.dataset,
        "--base_model",
        base_model,
        "--max_new_tokens",
        str(route_config["max_new_tokens"]),
        "--max_len",
        str(route_config["max_len"]),
        "--val_n",
        str(route_config["val_n"]),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(route_config["top_p"]),
        "--top_k",
        str(route_config["top_k"]),
        "--min_p",
        str(route_config["min_p"]),
        "--presence_penalty",
        str(route_config["presence_penalty"]),
        "--gpu_memory_utilization",
        str(runtime_config["gpu_memory_utilization"]),
        "--tensor_parallel_size",
        str(runtime_config["tensor_parallel_size"]),
    ]

    num_samples = route_config["num_samples"]
    if num_samples is not None:
        eval_args += ["--num_samples", str(num_samples)]
    if task.checkpoint_dir is not None:
        eval_args += ["--checkpoint_dir", task.checkpoint_dir]
    if output_file is not None:
        eval_args += ["--output_file", output_file]

    eval_args.append("--enable_thinking" if route_config["enable_thinking"] else "--no_thinking")

    if task.route == "math":
        eval_args += ["--root_dir", str(resolve_path(config["root_dir"]))]
        eval_args.append("--use_chat_template" if route_config["use_chat_template"] else "--no_chat_template")
    else:
        eval_args += ["--root_dir", str(resolve_path(config["root_dir"]))]
        eval_args += ["--answer_index_base", str(route_config["answer_index_base"])]
        eval_args.append("--use_chat_template" if route_config["use_chat_template"] else "--no-use_chat_template")

    return eval_args


def run_sub_evaluator(route: str, eval_args: list[str]) -> None:
    if route == "math":
        import evaluate_math

        module_main = evaluate_math.main
        program = "evaluate_math.py"
    else:
        import evaluate_mcq

        module_main = evaluate_mcq.main
        program = "evaluate_mcq.py"

    previous_argv = sys.argv
    try:
        sys.argv = [program, *eval_args]
        module_main()
    finally:
        sys.argv = previous_argv


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_eval_config(args.config)
    apply_runtime_env(config)
    tasks = build_tasks(config, args)
    is_base_eval = clean_str(args.base_model) is not None and not args.checkpoint_dir and not args.checkpoint_step
    target_label = "all" if args.all_checkpoints else ("base" if is_base_eval else "lora")
    print_section(
        "UNIFIED EVALUATION",
        {
            "Config": args.config,
            "Target": target_label,
            "Datasets": ", ".join(task.dataset for task in tasks),
            "Task count": len(tasks),
        },
    )

    for index, task in enumerate(tasks, start=1):
        print_section(
            f"EVAL TASK {index}/{len(tasks)}",
            {
                "Dataset": task.dataset,
                "Route": task.route,
                "Checkpoint": task.checkpoint_dir or "None (base model only)",
            },
        )
        eval_args = build_eval_args(config, args, task, len(tasks))
        if args.dry_run:
            print(f"{task.route}: {' '.join(eval_args)}")
        else:
            run_sub_evaluator(task.route, eval_args)


if __name__ == "__main__":
    main()
