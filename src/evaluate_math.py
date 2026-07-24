from __future__ import annotations

import argparse
import json
from pathlib import Path

from util import (
    build_math_eval_output_file,
    create_lora_request,
    dataset_display_name,
    extract_boxed_answer,
    extract_math_eval_example,
    format_prompt_for_model,
    load_vllm_model,
    load_math_eval_dataset,
    PROJECT_ROOT,
    print_section,
    resolve_eval_base_model,
)


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """
    Grade the predicted answer against ground truth using math_verify.

    Args:
        predicted: The predicted answer (already extracted from \\boxed{})
        ground_truth: The ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    try:
        from math_verify import parse, verify

        # Ensure answers are wrapped in $ for latex parsing
        if not "$" in predicted:
            predicted = f"${predicted}$"
        if not "$" in ground_truth:
            ground_truth = f"${ground_truth}$"

        # Parse both answers
        pred_parsed = parse(predicted, fallback_mode="no_fallback")
        gt_parsed = parse(ground_truth, fallback_mode="no_fallback")

        # Verify equivalence
        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception as e:
        # If math_verify fails, try simple string comparison
        # Normalize by removing spaces, $, and converting to lowercase
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def evaluate_math500(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int = None,
    output_file: str = None,
    lora_request=None,
    dataset_name: str = "math500",
    base_model_name: str = None,
    enable_thinking: bool = True,
    val_n: int = 1,
    use_chat_template: bool | None = None,
):
    from vllm import SamplingParams

    """
    Evaluate model on MATH500 or other datasets using Qwen3 thinking mode with best practices.

    Args:
        llm: The vLLM LLM instance
        tokenizer: The tokenizer for chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.6 for thinking, 0.7 for non-thinking)
        top_p: Top-p sampling parameter (0.95 for thinking, 0.8 for non-thinking)
        top_k: Top-k sampling parameter (20 recommended)
        min_p: Minimum probability threshold (0 recommended)
        presence_penalty: Presence penalty to reduce repetitions (0-2)
        num_samples: Number of samples to evaluate (None = all)
        output_file: Path to save detailed results
        lora_request: Optional LoRA request for inference
        dataset_name: Name of dataset to use
        base_model_name: Base model name for logging
        enable_thinking: Whether to use thinking mode
        use_chat_template: Whether to apply tokenizer chat template to prompts
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Chat Template: {'ENABLED' if use_chat_template else 'DISABLED'}")
    print(f"Temperature: {temperature} (Qwen3 {'thinking' if enable_thinking else 'non-thinking'} mode)")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"{'='*70}\n")

    print(f"Loading {dataset_name.upper()} dataset...")
    dataset, _ = load_math_eval_dataset(dataset_name, num_samples=num_samples)
    print(f"Loaded {dataset_display_name(dataset_name)} dataset with {len(dataset)} problems")

    print(f"Evaluating on {len(dataset)} problems with vLLM batch inference...")

    # Setup sampling parameters following Qwen3 best practices
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,  # Generate val_n solutions per prompt
    )

    total = 0
    formatted_count = 0
    results = []

    # Metrics for val_n > 1
    pass_at_n = 0  # At least one correct
    total_correct_per_problem = 0  # Sum of correct solutions across all problems

    # Prepare all prompts/messages for batch inference
    all_prompts = []
    all_messages = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []

    for example in dataset:
        eval_example = extract_math_eval_example(example, dataset_name)
        problem = eval_example.problem
        gt_answer = eval_example.ground_truth
        question_id = eval_example.question_id

        user_message = format_prompt_for_model(
            tokenizer,
            problem,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )
        messages = [{"role": "user", "content": user_message}]

        all_messages.append(messages)
        all_gt_answers.append(gt_answer)
        all_problems.append(problem)
        all_question_ids.append(question_id)

    # Run batch inference with vLLM using generate interface
    print(f"\nRunning vLLM batch inference on {len(all_messages)} problems...")
    print(
        "Using generate interface with tokenizer chat template..."
        if use_chat_template
        else "Using generate interface with plain prompts..."
    )

    all_prompts = []
    for messages in all_messages:
        all_prompts.append(messages[0]["content"])

    # Print dtype info before generation
    print("\n" + "=" * 70)
    print("GENERATION DTYPE CHECK")
    print("=" * 70)
    print(f"Model dtype: {llm.llm_engine.model_config.dtype}")
    print(f"Quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {llm.llm_engine.cache_config.cache_dtype}")
    print(f"Using LoRA: {lora_request is not None}")
    if lora_request is not None:
        if lora_request.lora_path is None:
            raise ValueError(
                "LoRA request created but lora_local_path is None; lora weights are empty, might be issue with using zero3 + peft; try using zero2"
            )
        print(f"LoRA path: {lora_request.lora_path}")
    print("=" * 70 + "\n")

    # Generate outputs
    if lora_request is not None:
        outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    # Process results
    print("\nProcessing results...")
    for idx, (output, problem, gt_answer, question_id) in enumerate(
        zip(outputs, all_problems, all_gt_answers, all_question_ids)
    ):
        # Process all val_n generations for this problem
        generations = []
        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []

        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text

            # Extract answer from generated text
            predicted_answer = extract_boxed_answer(generated_text)

            # Check if answer was properly formatted
            is_formatted = predicted_answer is not None

            # Grade the answer
            is_correct = grade_answer(predicted_answer, gt_answer)

            generations.append(generated_text)
            predicted_answers.append(predicted_answer if predicted_answer else "[No boxed answer found]")
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)

        # Calculate metrics for this problem
        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        # Majority vote: find the most common answer among formatted predictions
        majority_vote_correct = False
        if num_formatted > 0:
            from collections import Counter

            formatted_predictions = [pred for pred, fmt in zip(predicted_answers, is_formatted_list) if fmt]
            if formatted_predictions:
                most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
                majority_vote_correct = grade_answer(most_common_answer, gt_answer)

        # Update global metrics
        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += val_n

        # Store result with all generations
        result = {
            "problem_id": question_id if question_id is not None else idx,
            "problem": problem,
            "ground_truth": gt_answer,
            "val_n": val_n,
            "generations": [
                {"predicted_answer": pred, "full_generation": gen, "correct": corr, "formatted": fmt}
                for pred, gen, corr, fmt in zip(
                    predicted_answers, generations, is_correct_list, is_formatted_list
                )
            ],
            "num_correct": num_correct,
            "pass_at_n": has_correct,
            "majority_vote_correct": majority_vote_correct,
            # For backward compatibility
            "predicted_answer": predicted_answers[0],
            "full_generation": generations[0],
            "correct": is_correct_list[0],
            "formatted": is_formatted_list[0],
        }
        results.append(result)

        # Print progress for each problem
        format_rate = formatted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        # Print brief update for every problem
        status = "✓" if has_correct else "✗"
        print(
            f"{status} [{idx + 1}/{len(dataset)}] Pass@{val_n}: {current_pass_at_n:.1f}% | Avg@{val_n}: {current_avg_at_n:.1f}% | Formatted: {format_rate:.1f}%"
        )

        # Print detailed info every 10 problems
        if (idx + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Progress: {idx + 1}/{len(dataset)}")
            print(f"Pass@{val_n}: {current_pass_at_n:.2f}%")
            print(f"Average@{val_n}: {current_avg_at_n:.2f}%")
            print(f"Format Rate: {format_rate:.2f}%")
            print(f"Last problem: {problem[:100]}...")
            print(f"Solutions correct: {num_correct}/{val_n}")
            print(f"Majority vote: {'✓' if majority_vote_correct else '✗'}")
            print(f"Ground truth: {gt_answer}")
            print(f"{'='*70}\n")

    # Calculate final metrics
    num_problems = len(dataset)
    format_rate = formatted_count / total * 100

    # Calculate pass@n, average@n, and majority vote metrics
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100

    # Calculate majority vote accuracy
    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print(f"\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})")
    print(
        f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_correct_count}/{num_problems})"
    )
    print(f"\nFormatting:")
    print(f"  Formatted (boxed) answers: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)

    # Save detailed results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "dataset": dataset_name,
            "enable_thinking": enable_thinking,
            "use_chat_template": use_chat_template,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MATH tasks with Qwen3 thinking mode")

    parser.add_argument(
        "--base_model",
        type=str,
        help="Path to base model. If omitted with --checkpoint_dir, reads base_model_name_or_path from adapter_config.json.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to checkpoint directory with LoRA adapters. If not provided, will use base model only.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["math500", "aime24", "aime25", "minerva", "amc23"],
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        required=True,
        help="Maximum tokens to generate",
    )
    thinking_group = parser.add_mutually_exclusive_group(required=True)
    thinking_group.add_argument("--enable_thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=True,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top_k", type=int, required=True, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--min_p", type=float, required=True, help="Minimum probability threshold"
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        required=True,
        help="Presence penalty to reduce repetitions",
    )
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Repository root used for default evaluation output paths.",
    )
    parser.add_argument("--output_file", type=str, help="Path to save detailed results JSON")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        required=True,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        required=True,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=True,
        help="Maximum model context length used for evaluation",
    )
    parser.add_argument(
        "--val_n", type=int, required=True, help="Number of solutions to sample per problem"
    )
    chat_template_group = parser.add_mutually_exclusive_group(required=True)
    chat_template_group.add_argument(
        "--use_chat_template",
        dest="use_chat_template",
        action="store_true",
        help="Apply tokenizer chat template to evaluation prompts.",
    )
    chat_template_group.add_argument(
        "--no_chat_template",
        dest="use_chat_template",
        action="store_false",
        help="Disable tokenizer chat template for evaluation prompts.",
    )

    args = parser.parse_args()

    if args.use_chat_template is None:
        args.use_chat_template = False

    try:
        args.base_model = resolve_eval_base_model(
            checkpoint_dir=args.checkpoint_dir,
            base_model=args.base_model,
        )
    except ValueError as e:
        print_section("ERROR", {"Message": e})
        raise SystemExit(1) from e

    # Warn if using greedy decoding in thinking mode
    if args.enable_thinking and args.temperature == 0.0:
        print("\n" + "!" * 70)
        print("WARNING: Using greedy decoding (temperature=0.0) in thinking mode!")
        print("Qwen3 recommends temperature=0.6 for thinking mode to avoid")
        print("performance degradation and endless repetitions.")
        print("!" * 70 + "\n")

    # Auto-generate output file if not specified
    if args.output_file is None:
        args.output_file = build_math_eval_output_file(
            root_dir=args.root_dir,
            dataset_name=args.dataset,
            base_model=args.base_model,
            checkpoint_dir=args.checkpoint_dir,
        )

    print(f"Results will be saved to: {args.output_file}")

    print_section(
        "QWEN3 MATH EVALUATION WITH THINKING MODE",
        {
            "Dataset": args.dataset.upper(),
            "Base model": args.base_model,
            "Checkpoint": args.checkpoint_dir or "None (base model only)",
            "Thinking Mode": "ENABLED" if args.enable_thinking else "DISABLED",
            "Chat Template": "ENABLED" if args.use_chat_template else "DISABLED",
            "Max tokens": args.max_new_tokens,
            "Temperature": f"{args.temperature} (Qwen3 {'thinking' if args.enable_thinking else 'non-thinking'} mode)",
            "Top-p": args.top_p,
            "Top-k": args.top_k,
            "Min-p": args.min_p,
            "Presence penalty": args.presence_penalty,
            "Num samples": args.num_samples or "All",
            "Val-N (solutions per problem)": args.val_n,
            "Output file": args.output_file,
            "GPU memory utilization": args.gpu_memory_utilization,
            "Tensor parallel size": args.tensor_parallel_size,
            "Max model len": args.max_len,
        },
    )

    # Load model with vLLM
    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = create_lora_request(args.checkpoint_dir)

    # Run evaluation
    average_at_n_pct, results = evaluate_math500(
        llm,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        num_samples=args.num_samples,
        output_file=args.output_file,
        lora_request=lora_request,
        dataset_name=args.dataset,
        base_model_name=args.base_model,
        enable_thinking=args.enable_thinking,
        val_n=args.val_n,
        use_chat_template=args.use_chat_template,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
