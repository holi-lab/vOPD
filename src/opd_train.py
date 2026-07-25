import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from omegaconf import OmegaConf
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

from trl.experimental.minillm import MiniLLMConfig
from src.opd_trainer import CustomMiniLLMTrainer as MiniLLMTrainer
from src.util import (
    build_run_naming,
    distill_mode_to_config,
    effective_distill_mode as get_effective_distill_mode,
    extract_problem,
    format_prompt_for_model,
    load_training_args,
    load_training_dataset_by_alias,
    normalize_dtype_str,
    resolve_dataset_name,
    resolve_dtype,
    safe_init_kwargs,
)


if __name__ == "__main__":
    args = load_training_args(sys.argv[1:])
    if "wandb" in [target.strip() for target in args.report_to.split(",")]:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    teacher_model = args.teacher_model_name_or_path or args.model_name_or_path
    resolved_dataset_name = resolve_dataset_name(args.dataset_name)
    effective_distill_mode = get_effective_distill_mode(args)
    mode_cfg = distill_mode_to_config(effective_distill_mode)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    naming = build_run_naming(args, world_size=world_size, effective_distill_mode=effective_distill_mode)
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir_root = naming.output_dir_root
    output_dir = naming.output_dir
    checkpoint_name = naming.checkpoint_name
    run_name = naming.run_name
    model_name = naming.model_name
    method_name = naming.method_name
    baseline_suffix = naming.legacy_baseline_suffix
    effective_batch_size = naming.effective_batch_size
    resume_from_checkpoint = args.resume_from_checkpoint
    os.makedirs(output_dir, exist_ok=True)

    cfg_meta = {
        "timestamp_utc": timestamp_utc,
        "output_dir_root": output_dir_root,
        "output_dir": output_dir,
        "cfg_subdir": checkpoint_name,
        "checkpoint_name": checkpoint_name,
        "model_name": model_name,
        "method_name": method_name,
        "teacher_model": teacher_model,
        "dataset_name": args.dataset_name,
        "resolved_dataset_name": resolved_dataset_name,
        "run_name": run_name,
        "naming_scheme": "dataset/model/method/lr-<learning_rate>_ebs-<effective_batch_size>",
        "baseline_suffix": baseline_suffix,
        "resume_from_checkpoint": resume_from_checkpoint,
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "args": OmegaConf.to_container(args, resolve=True),
    }
    with open(os.path.join(output_dir, "cfg_meta.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_meta, f, indent=2, sort_keys=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _to_train_prompt(ex: dict[str, Any]) -> dict[str, str]:
        problem = extract_problem(ex, args.dataset_name)
        prompt = format_prompt_for_model(
            tokenizer,
            problem,
            use_chat_template=args.use_chat_template,
            enable_thinking=False,
        )
        return {"prompt": prompt}

    train_dataset = load_training_dataset_by_alias(args.dataset_name, split=args.train_split)
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    train_dataset = train_dataset.map(
        _to_train_prompt,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = None
    if args.eval_split is not None:
        eval_dataset = load_training_dataset_by_alias(args.dataset_name, split=args.eval_split)
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
        eval_dataset = eval_dataset.map(
            _to_train_prompt,
            remove_columns=eval_dataset.column_names,
        )

    dtype_str = normalize_dtype_str(args.torch_dtype)
    dtype_obj = resolve_dtype(args.torch_dtype)

    model_init_kwargs = {
        "attn_implementation": args.attn_implementation,
        # Keep both for compatibility across TRL/Transformers versions.
        "dtype": dtype_str,
        "torch_dtype": dtype_obj,
        "trust_remote_code": True,
        "use_cache": False if args.gradient_checkpointing else True,
    }
    teacher_model_init_kwargs = {
        "attn_implementation": args.attn_implementation,
        # MiniLLMTrainer in your installed TRL expects this exact key.
        "dtype": dtype_str,
        "torch_dtype": dtype_obj,
        "trust_remote_code": True,
        "use_cache": True,
    }

    config_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": ({"use_reentrant": False} if args.gradient_checkpointing else None),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "do_eval": eval_dataset is not None,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "num_generations": args.num_generations,
        "rkl_advantage": mode_cfg.rkl_advantage,
        "single_step_decomposition": mode_cfg.single_step_decomposition,
        "gamma": args.gamma if args.gamma is not None else mode_cfg.gamma,
        "beta": args.beta,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "report_to": args.report_to,
        "run_name": run_name,
        "seed": args.seed,
        "model_init_kwargs": model_init_kwargs,
        "teacher_model_init_kwargs": teacher_model_init_kwargs,
    }
    training_args = MiniLLMConfig(**safe_init_kwargs(MiniLLMConfig, config_kwargs))

    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
        )

    trainer = MiniLLMTrainer(
        model=args.model_name_or_path,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        use_baseline=args.use_baseline,
        kl_top_k=args.kl_top_k,
        opd_top_k=args.opd_top_k,
        reward_clip_lambda=args.reward_clip_lambda,
        log_grad_variance=args.log_grad_variance,
        grad_variance_logging_steps=args.grad_variance_logging_steps,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
