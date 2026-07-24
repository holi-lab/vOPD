from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAINING_CONFIG_KEY = "training_config"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)
TRAINING_CONFIG_ALLOWED_KEYS = {
    "attn_implementation",
    "beta",
    "dataset_name",
    "distill_mode",
    "eval_split",
    "eval_steps",
    "gamma",
    "grad_variance_logging_steps",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "kl_top_k",
    "log_grad_variance",
    "learning_rate",
    "logging_steps",
    "lora_alpha",
    "lora_dropout",
    "lora_r",
    "lora_target_modules",
    "lr_scheduler_type",
    "max_completion_length",
    "max_eval_samples",
    "max_grad_norm",
    "max_prompt_length",
    "max_steps",
    "max_train_samples",
    "min_p",
    "model_name_or_path",
    "num_generations",
    "num_train_epochs",
    "opd_top_k",
    "output_dir",
    "per_device_eval_batch_size",
    "per_device_train_batch_size",
    "presence_penalty",
    "report_to",
    "resume_from_checkpoint",
    "run_name",
    "save_steps",
    "save_total_limit",
    "seed",
    "teacher_model_name_or_path",
    "temperature",
    "top_k",
    "top_p",
    "torch_dtype",
    "train_split",
    "use_baseline",
    "use_chat_template",
    "use_peft",
    "use_vllm",
    "vllm_gpu_memory_utilization",
    "vllm_mode",
    "vllm_tensor_parallel_size",
    "wandb_project",
    "warmup_ratio",
    "weight_decay",
}


@dataclass(frozen=True)
class DatasetSpec:
    canonical_name: str
    path: str
    aliases: tuple[str, ...]
    train_split: str = "train"
    eval_split: str = "test"
    config_name: str | None = None
    problem_key: str = "problem"
    answer_key: str | None = "answer"
    solution_key: str | None = None
    id_key: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class MathEvalExample:
    problem: str
    ground_truth: str
    question_id: Any = None


DATASET_SPECS = {
    "dapo14k": DatasetSpec(
        canonical_name="dapo14k",
        path="guanning-ai/dapo14k",
        aliases=("dapo", "dapo14k"),
        train_split="train",
        eval_split="train",
        problem_key="problem",
    ),
    "math500": DatasetSpec(
        canonical_name="math500",
        path="HuggingFaceH4/MATH-500",
        aliases=("math500", "math-500"),
        train_split="test",
        eval_split="test",
        problem_key="problem",
        answer_key=None,
        solution_key="solution",
    ),
    "minerva": DatasetSpec(
        canonical_name="minerva",
        path="/path/to/minerva",
        aliases=("minerva", "minervamath"),
        train_split="test",
        eval_split="test",
        problem_key="question",
        answer_key="answer",
        id_key="id",
    ),
    "amc23": DatasetSpec(
        canonical_name="amc23",
        path="/path/to/amc23",
        aliases=("amc23", "amc2023"),
        train_split="test",
        eval_split="test",
        problem_key="question",
        answer_key="answer",
        id_key="id",
    ),
    "aime24": DatasetSpec(
        canonical_name="aime24",
        path="/path/to/aime24",
        aliases=("aime24", "aime2024", "aime-2024"),
        train_split="train",
        eval_split="train",
        problem_key="problem",
        answer_key="answer",
        id_key="id",
    ),
    "aime25": DatasetSpec(
        canonical_name="aime25",
        path="/path/to/aime25",
        aliases=("aime25", "aime2025", "aime-2025"),
        train_split="train",
        eval_split="train",
        problem_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        trust_remote_code=True,
    ),
}
DATASET_NAME_ALIASES = {
    normalize_alias: canonical
    for canonical, spec in DATASET_SPECS.items()
    for normalize_alias in (alias.lower().replace("_", "-") for alias in (*spec.aliases, spec.path, canonical))
}


def safe_init_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def resolve_dtype(dtype: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported --torch_dtype: {dtype}")
    return mapping[key]


def normalize_dtype_str(dtype: str) -> str:
    key = dtype.lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    return aliases.get(key, key)


def normalize_dataset_alias_key(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("_", "-")


def get_dataset_spec(dataset_name: str) -> DatasetSpec | None:
    canonical_name = DATASET_NAME_ALIASES.get(normalize_dataset_alias_key(dataset_name))
    if canonical_name is None:
        return None
    return DATASET_SPECS[canonical_name]


def known_dataset_aliases() -> list[str]:
    return sorted({alias for spec in DATASET_SPECS.values() for alias in spec.aliases})


def resolve_dataset_name(dataset_name: str) -> str:
    spec = get_dataset_spec(dataset_name)
    return spec.path if spec is not None else dataset_name


def _load_dataset_from_spec(spec: DatasetSpec, split: str | None = None, **kwargs: Any) -> Any:
    from datasets import load_dataset

    load_args = [spec.path]
    if spec.config_name is not None:
        load_args.append(spec.config_name)
    load_kwargs = dict(kwargs)
    if split is not None:
        load_kwargs["split"] = split
    if spec.trust_remote_code:
        load_kwargs.setdefault("trust_remote_code", True)
    return load_dataset(*load_args, **load_kwargs)


def load_dataset_by_alias(dataset_name: str, split: str | None = None, **kwargs: Any) -> Any:
    from datasets import load_dataset

    spec = get_dataset_spec(dataset_name)
    if spec is not None:
        return _load_dataset_from_spec(spec, split=split, **kwargs)
    if split is not None:
        kwargs["split"] = split
    return load_dataset(dataset_name, **kwargs)


def resolve_dataset_split(dataset_name: str, split: str | None, purpose: str) -> str | None:
    spec = get_dataset_spec(dataset_name)
    if spec is None:
        return split
    default_split = spec.train_split if purpose == "train" else spec.eval_split
    if split is None:
        return default_split
    if purpose == "train" and split == "train" and default_split != "train":
        return default_split
    if purpose == "eval" and split == "test" and default_split != "test":
        return default_split
    return split


def load_training_dataset_by_alias(dataset_name: str, split: str | None = None, **kwargs: Any) -> Any:
    return load_dataset_by_alias(
        dataset_name,
        split=resolve_dataset_split(dataset_name, split, "train"),
        **kwargs,
    )


def load_eval_dataset_by_alias(dataset_name: str, split: str | None = None, **kwargs: Any) -> Any:
    return load_dataset_by_alias(
        dataset_name,
        split=resolve_dataset_split(dataset_name, split, "eval"),
        **kwargs,
    )


def extract_problem(example: dict[str, Any], dataset_name: str | None = None) -> str:
    spec = get_dataset_spec(dataset_name) if dataset_name is not None else None
    candidate_keys = [spec.problem_key] if spec is not None else []
    candidate_keys.extend(("problem", "Question", "question", "prompt"))
    for key in dict.fromkeys(candidate_keys):
        value = example.get(key, None)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"Could not find problem text in sample keys: {list(example.keys())}")


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def extract_math_eval_example(example: dict[str, Any], dataset_name: str) -> MathEvalExample:
    spec = get_dataset_spec(dataset_name)
    problem = extract_problem(example, dataset_name)
    question_id = example.get(spec.id_key, None) if spec is not None and spec.id_key is not None else None

    if spec is not None and spec.answer_key is not None and spec.answer_key in example:
        return MathEvalExample(problem=problem, ground_truth=str(example[spec.answer_key]), question_id=question_id)
    if spec is not None and spec.solution_key is not None and spec.solution_key in example:
        solution = str(example[spec.solution_key])
        return MathEvalExample(
            problem=problem,
            ground_truth=extract_boxed_answer(solution) or solution,
            question_id=question_id,
        )

    if "answer" in example:
        return MathEvalExample(problem=problem, ground_truth=str(example["answer"]), question_id=question_id)
    if "solution" in example:
        solution = str(example["solution"])
        return MathEvalExample(problem=problem, ground_truth=extract_boxed_answer(solution) or solution, question_id=question_id)
    raise KeyError(f"Could not find answer text in sample keys: {list(example.keys())}")


def load_math_eval_dataset(dataset_name: str, num_samples: int | None = None, split: str | None = None) -> tuple[Any, DatasetSpec | None]:
    dataset = load_eval_dataset_by_alias(dataset_name, split=split)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset, get_dataset_spec(dataset_name)


def dataset_display_name(dataset_name: str) -> str:
    spec = get_dataset_spec(dataset_name)
    return spec.path if spec is not None else dataset_name


def print_section(title: str, values: dict[str, Any] | None = None) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if values is not None:
        for key, value in values.items():
            print(f"{key}: {value}")
    print("=" * 70 + "\n")


def find_lora_adapter_weights(lora_adapter_path: str | None) -> Path | None:
    if lora_adapter_path is None:
        return None

    adapter_dir = Path(lora_adapter_path)
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str | None = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    enable_thinking: bool = True,
):
    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"Loading model with vLLM from: {base_model_path}")

    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        print(f"LoRA adapter path provided: {lora_adapter_path}")
        if find_lora_adapter_weights(lora_adapter_path) is not None:
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}")
            print("Continuing with base model only...")

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print_section(
        "MODEL DTYPE INFORMATION",
        {
            "vLLM Model Config dtype": llm.llm_engine.model_config.dtype,
            "vLLM Model quantization": llm.llm_engine.model_config.quantization,
            "KV cache dtype": llm.llm_engine.cache_config.cache_dtype,
        },
    )
    print("vLLM model loaded successfully!")
    return llm, tokenizer


def create_lora_request(checkpoint_dir: str | None):
    if checkpoint_dir is None:
        return None

    try:
        from vllm.lora.request import LoRARequest
    except ImportError:
        print("Warning: Could not import LoRARequest. Running without LoRA.")
        return None

    try:
        if find_lora_adapter_weights(checkpoint_dir) is not None:
            request = LoRARequest("checkpoint_lora", 1, checkpoint_dir)
            print(f"OK: Successfully created LoRA request for: {checkpoint_dir}")
            return request

        print(f"Warning: No LoRA adapter weights found at {checkpoint_dir}")
        print("Expected 'adapter_model.safetensors' or 'adapter_model.bin'")
        print("Continuing with base model only...")
    except Exception as e:
        print(f"Warning: Could not create LoRA request: {e}")
        print("Continuing without LoRA.")
    return None


def resolve_eval_base_model(
    checkpoint_dir: str | None,
    base_model: str | None,
) -> str:
    if checkpoint_dir is None:
        if base_model is not None:
            return base_model
        raise ValueError("--base_model is required when --checkpoint_dir is not provided")

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(
            "Checkpoint directory does not exist\n"
            f"Provided checkpoint directory: {checkpoint_dir}\n"
            "Please provide a valid checkpoint directory or omit --checkpoint_dir to use the base model only."
        )

    if base_model is not None:
        return base_model

    adapter_config = checkpoint_path / "adapter_config.json"
    if not adapter_config.exists():
        raise ValueError(
            "LoRA adapter config not found\n"
            f"Expected adapter config: {adapter_config}\n"
            "Cannot infer base model. Provide --base_model explicitly or use a valid LoRA checkpoint."
        )

    try:
        with adapter_config.open("r", encoding="utf-8") as f:
            adapter_config_data = json.load(f)
        return adapter_config_data["base_model_name_or_path"]
    except KeyError as e:
        raise ValueError(
            "base_model_name_or_path missing from adapter_config.json\n"
            f"Adapter config: {adapter_config}\n"
            "Provide --base_model explicitly or use a PEFT adapter config with base_model_name_or_path."
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse adapter_config.json: {adapter_config}\nJSON error: {e}") from e


def build_math_eval_output_file(
    root_dir: str,
    dataset_name: str,
    base_model: str,
    checkpoint_dir: str | None,
) -> str:
    root_path = Path(root_dir).resolve()
    dataset_name_for_path = sanitize_for_path(dataset_name)
    if checkpoint_dir:
        label = eval_checkpoint_label(checkpoint_dir, root_path / "checkpoints")
        filename = f"{label}_results.json"
    else:
        base_name = sanitize_for_path(Path(base_model).name)
        filename = f"{base_name}_base_results.json"
    result_dir = root_path / "evaluations" / dataset_name_for_path
    return str(result_dir / filename)


def eval_checkpoint_label(checkpoint_dir: str | Path, checkpoint_root: str | Path | None = None) -> str:
    checkpoint_path = Path(checkpoint_dir).expanduser()
    try:
        checkpoint_path = checkpoint_path.resolve()
    except OSError:
        checkpoint_path = checkpoint_path.absolute()

    roots = []
    if checkpoint_root is not None:
        roots.append(Path(checkpoint_root).expanduser())
    roots.append(Path.cwd() / "checkpoints")

    for root in roots:
        try:
            rel_path = checkpoint_path.relative_to(root.resolve())
            return sanitize_for_path("__".join(rel_path.parts))
        except (OSError, ValueError):
            continue

    label_parts = checkpoint_path.parts[-5:]
    return sanitize_for_path("__".join(label_parts))


def build_tagged_eval_output_file(
    dataset_name: str,
    base_model: str,
    checkpoint_dir: str | None,
    enable_thinking: bool,
    use_chat_template: bool | None,
    temperature: float,
    val_n: int,
    root_dir: str | None = None,
) -> str:
    output_root = Path(root_dir) if root_dir is not None else Path(PROJECT_ROOT) / "evaluations"
    parts = [sanitize_for_path(dataset_name), sanitize_for_path(Path(base_model).name)]
    if checkpoint_dir:
        parts.append(eval_checkpoint_label(checkpoint_dir))
    else:
        parts.append("base")
    parts += [
        "thinking" if enable_thinking else "nonthinking",
        "chat" if use_chat_template else "raw",
        f"temp{temperature}",
        f"valn{val_n}",
    ]
    return str(output_root / sanitize_for_path(dataset_name) / ("_".join(parts) + ".json"))


def dataset_canonical_name(dataset_name: str) -> str:
    spec = get_dataset_spec(dataset_name)
    return spec.canonical_name if spec is not None else dataset_name


def sanitize_for_path(value: str) -> str:
    # Keep directory names filesystem-safe and stable.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return safe.strip("-_.") or "cfg"


def format_prompt(problem: str) -> str:
    return f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return prompt


def ensure_chat_template(tokenizer: Any) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = CHAT_TEMPLATE
    return True


def format_prompt_with_chat_template(tokenizer: Any, problem: str, enable_thinking: bool = False) -> str:
    ensure_chat_template(tokenizer)
    user_message = format_prompt(problem)
    messages = [{"role": "user", "content": user_message}]
    try:
        # Qwen-style tokenizers may support `enable_thinking`.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def format_prompt_for_model(
    tokenizer: Any,
    problem: str,
    use_chat_template: bool | None = None,
    enable_thinking: bool = False,
) -> str:
    if use_chat_template:
        return format_prompt_with_chat_template(tokenizer, problem, enable_thinking=enable_thinking)
    return format_prompt(problem)


@dataclass
class DistillModeConfig:
    rkl_advantage: bool
    single_step_decomposition: bool
    gamma: Any


@dataclass
class RunNaming:
    run_name: str
    checkpoint_name: str
    output_dir_root: str
    output_dir: str
    method_name: str
    legacy_baseline_suffix: str
    model_name: str
    dataset_name: str
    effective_batch_size: int


def distill_mode_to_config(mode: str) -> DistillModeConfig:
    mode = mode.lower()
    if mode == "tml":
        # Think Machine Lab style on-policy KD per TRL MiniLLM docs.
        return DistillModeConfig(rkl_advantage=True, single_step_decomposition=False, gamma=False)
    if mode == "gkd_rkl":
        # Reverse-KL single-step distillation signal.
        return DistillModeConfig(rkl_advantage=False, single_step_decomposition=True, gamma=None)
    raise ValueError(f"Unknown distill mode: {mode}")


def build_baseline_suffix(args: Any) -> str:
    if not args.use_baseline:
        return "_sample" if args.distill_mode == "tml" else "_full-vocab"
    if args.kl_top_k > 0:
        return f"_baseline-top-{args.kl_top_k}"
    return "_baseline"


def format_learning_rate_for_name(learning_rate: float) -> str:
    return f"{learning_rate:.3g}".replace("e-0", "e-").replace("e+0", "e")


def build_method_name(args: Any, effective_distill_mode: str) -> str:
    if args.use_baseline:
        if args.kl_top_k > 0:
            return f"{effective_distill_mode}-baseline-kl{args.kl_top_k}-norm"
        return f"{effective_distill_mode}-baseline-full"

    if effective_distill_mode == "tml":
        return "tml-sample"
    if getattr(args, "opd_top_k", -1) > 0:
        return f"{effective_distill_mode}-opd-top{args.opd_top_k}"
    return f"{effective_distill_mode}-full"


def build_run_naming(args: Any, *, world_size: int, effective_distill_mode: str) -> RunNaming:
    model_name = os.path.basename(args.model_name_or_path.rstrip("/"))
    dataset_name = sanitize_for_path(dataset_canonical_name(args.dataset_name))
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    lr_str = format_learning_rate_for_name(args.learning_rate)
    method_name = sanitize_for_path(build_method_name(args, effective_distill_mode))
    param_leaf = f"lr-{lr_str}_ebs-{effective_batch_size}"
    run_leaf = (
        f"{sanitize_for_path(args.run_name)}__{param_leaf}"
        if args.run_name
        else param_leaf
    )
    checkpoint_name = os.path.join(dataset_name, sanitize_for_path(model_name), method_name, run_leaf)
    output_dir_root = args.output_dir.rstrip("/")
    output_dir = os.path.join(output_dir_root, checkpoint_name)
    run_name = sanitize_for_path("__".join(checkpoint_name.split(os.sep)))
    return RunNaming(
        run_name=run_name,
        checkpoint_name=checkpoint_name,
        output_dir_root=output_dir_root,
        output_dir=output_dir,
        method_name=method_name,
        legacy_baseline_suffix=build_baseline_suffix(args),
        model_name=model_name,
        dataset_name=dataset_name,
        effective_batch_size=effective_batch_size,
    )


def resolve_default_config_path(parent_config_path: str, default_entry: str) -> str:
    default_path = default_entry
    if not default_path.endswith((".yaml", ".yml")):
        default_path = f"{default_path}.yaml"
    if not os.path.isabs(default_path):
        default_path = os.path.join(os.path.dirname(parent_config_path), default_path)
    return default_path


def load_omegaconf_config(config_path: str, seen: set[str]) -> DictConfig:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    config_path = os.path.abspath(os.path.expanduser(config_path))
    if config_path in seen:
        chain = " -> ".join([*seen, config_path])
        raise ValueError(f"Circular training config defaults: {chain}")
    if not os.path.exists(config_path):
        raise ValueError(f"--training_config does not exist: {config_path}")

    seen.add(config_path)
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"--training_config must contain a YAML mapping: {config_path}")

    defaults = config.pop("defaults", [])
    if defaults is None:
        defaults = []
    if isinstance(defaults, ListConfig):
        defaults = list(defaults)
    if isinstance(defaults, str):
        defaults = [defaults]
    if isinstance(defaults, list) and all(isinstance(entry, str) for entry in defaults):
        parents = defaults
    else:
        raise ValueError(f"'defaults' must be a string or list of strings: {config_path}")

    merged = OmegaConf.create({})
    for parent in parents:
        parent_path = resolve_default_config_path(config_path, parent)
        merged = OmegaConf.merge(merged, load_omegaconf_config(parent_path, seen))
    seen.remove(config_path)
    return OmegaConf.merge(merged, config)


def validate_training_config_keys(config: dict[str, Any], config_path: str | None = None) -> None:
    unknown_keys = sorted(set(config) - TRAINING_CONFIG_ALLOWED_KEYS)
    if unknown_keys:
        source = f" {config_path}" if config_path else ""
        raise ValueError(
            f"Unknown keys in training config{source}: {unknown_keys}. "
            f"Allowed keys: {sorted(TRAINING_CONFIG_ALLOWED_KEYS)}"
        )


def load_training_config(config_path: str | None) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if config_path is None:
        return {}

    config_path = resolve_training_config_path(config_path)
    config = OmegaConf.to_container(
        load_omegaconf_config(config_path, seen=set()),
        resolve=True,
    )
    if not isinstance(config, dict):
        raise ValueError(f"--training_config must contain a YAML mapping: {config_path}")
    validate_training_config_keys(config, config_path)
    return config


def get_training_config_path(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg in ("--training_config", TRAINING_CONFIG_KEY):
            if index + 1 >= len(argv):
                raise ValueError("--training_config requires a path")
            return argv[index + 1]
        if arg.startswith("--training_config=") or arg.startswith(f"{TRAINING_CONFIG_KEY}="):
            return arg.split("=", 1)[1]
    return None


def resolve_training_config_path(config_path: str) -> str:
    return os.path.abspath(os.path.expanduser(config_path))


def cli_args_to_dotlist(argv: list[str]) -> list[str]:
    list_keys = {"lora_target_modules"}
    dotlist = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--training_config", TRAINING_CONFIG_KEY):
            index += 2
            continue
        if arg.startswith("--training_config=") or arg.startswith(f"{TRAINING_CONFIG_KEY}="):
            index += 1
            continue
        if arg.startswith("--no-"):
            dotlist.append(f"{arg[5:]}=false")
            index += 1
            continue
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                dotlist.append(key)
                index += 1
                continue
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                if key in list_keys:
                    values = []
                    index += 1
                    while index < len(argv) and not argv[index].startswith("--"):
                        values.append(argv[index])
                        index += 1
                    dotlist.append(f"{key}=[{','.join(values)}]")
                    continue
                dotlist.append(f"{key}={argv[index + 1]}")
                index += 2
                continue
            dotlist.append(f"{key}=true")
            index += 1
            continue
        dotlist.append(arg)
        index += 1
    return dotlist


def load_training_args(argv: list[str]) -> DictConfig:
    from omegaconf import OmegaConf

    config_path = get_training_config_path(argv)
    if config_path is None:
        raise ValueError("--training_config is required")
    resolved_config_path = resolve_training_config_path(config_path)
    config = OmegaConf.create(load_training_config(resolved_config_path))
    cli_config = OmegaConf.from_dotlist(cli_args_to_dotlist(argv))
    merged = OmegaConf.merge(config, cli_config)
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Training arguments must resolve to a mapping")
    validate_training_config_keys(resolved)
    merged.training_config = resolved_config_path
    merged.training_config_values = load_training_config(resolved_config_path)
    return merged


def effective_distill_mode(args: Any) -> str:
    if args.distill_mode is not None:
        return args.distill_mode
    if args.use_baseline:
        return "tml"
    raise AssertionError("--distill_mode must be set when --use_baseline is false")
