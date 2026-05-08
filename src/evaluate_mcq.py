import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from util import (
    build_tagged_eval_output_file,
    create_lora_request,
    load_vllm_model,
    PROJECT_ROOT,
    print_section,
)

ChoiceSet = Optional[set[str]]
ANSWER_KEYS = {"answer", "choice"}


def _as_choice_set(values: Optional[Iterable[str]]) -> ChoiceSet:
    if values is None:
        return None
    choices = {value.strip().upper() for value in values if value and value.strip()}
    return choices or None


def _choice_allowed(choice: str, valid_choices: ChoiceSet) -> bool:
    return valid_choices is None or choice in valid_choices


def normalize_choice(value: Optional[Any], valid_choices: ChoiceSet = None) -> Optional[str]:
    if value is None:
        return None

    token = extract_choice_token(value)
    if token is None:
        return None

    choice = token.upper()
    if not _choice_allowed(choice, valid_choices):
        return None
    return choice


def extract_choice_token(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    match = re.fullmatch(r"[\(\[\{]?\s*([A-Za-z])\s*[\)\]\}\.]?", text)
    if not match:
        return None
    return match.group(1)


def parse_original_choice_aliases(problem: str, valid_choices: ChoiceSet = None) -> dict[str, str]:
    """Map original lowercase option labels to final answer labels.

    GPQA-D prompts may contain original options like `a) ...` and then shuffled
    answer labels like `A. d`, `B. a`. In that shape, a JSON answer of `"a"`
    usually refers to the original `a)` option, not final answer label `A`.
    """
    allowed = valid_choices or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    aliases = {}
    for final_label, original_label in re.findall(r"(?im)^\s*([A-Z])\.\s*([a-z])\s*$", problem):
        final_label = final_label.upper()
        original_label = original_label.upper()
        if final_label in allowed:
            aliases[original_label] = final_label
    return aliases


def normalize_choice_for_problem(
    value: Optional[Any],
    problem: str,
    valid_choices: ChoiceSet = None,
) -> Optional[str]:
    token = extract_choice_token(value)
    if token is None:
        return None

    aliases = parse_original_choice_aliases(problem, valid_choices)
    if token.islower() and aliases:
        mapped = aliases.get(token.upper())
        if mapped is not None and _choice_allowed(mapped, valid_choices):
            return mapped

    return normalize_choice(token, valid_choices)


def extract_choice_from_answer_value(
    value: Any,
    problem: str,
    valid_choices: ChoiceSet = None,
) -> Optional[str]:
    choice = normalize_choice_for_problem(value, problem, valid_choices)
    if choice is not None:
        return choice

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if str(key).lower() in ANSWER_KEYS:
                choice = extract_choice_from_answer_value(nested_value, problem, valid_choices)
                if choice is not None:
                    return choice

        for key, nested_value in value.items():
            key_choice = normalize_choice_for_problem(key, problem, valid_choices)
            if key_choice is not None:
                return key_choice
            value_choice = normalize_choice_for_problem(nested_value, problem, valid_choices)
            if value_choice is not None:
                return value_choice

    return None


def index_to_choice(
    value: Any,
    valid_choices: ChoiceSet = None,
    answer_index_base: int = 0,
) -> Optional[str]:
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        return None

    idx = int(text) - answer_index_base
    if idx < 0:
        return None

    if valid_choices is not None:
        ordered_choices = sorted(valid_choices)
        if idx >= len(ordered_choices):
            return None
        return ordered_choices[idx]

    if idx <= 25:
        return chr(ord("A") + idx)
    return None


def normalize_ground_truth(
    value: Any,
    valid_choices: ChoiceSet = None,
    answer_index_base: int = 0,
) -> Optional[str]:
    if value is None:
        return None

    normalized = normalize_choice(value, valid_choices)
    if normalized is not None:
        return normalized

    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return index_to_choice(value, valid_choices, answer_index_base)

    return None


def first_present(example: Mapping[str, Any], *keys: str) -> Any:
    """Return the first non-None field. Keeps falsy labels like 0."""
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def infer_choice_letters(problem: str) -> set[str]:
    letters = set()
    option_patterns = [
        r"(?im)(?:^|[\n\r])\s*([A-Z])\s*[\.\)]\s+\S",
        r"(?i)(?:^|\s)([A-D])\)\s+\S",
        r"(?i)(?:^|\s)([A-D])\.\s+\S",
    ]
    for pattern in option_patterns:
        letters.update(match.upper() for match in re.findall(pattern, problem))
    return letters or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_option_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", text)
    text = re.sub(r"[^a-z0-9.+\-/%^]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_mcq_options(problem: str, valid_choices: ChoiceSet = None) -> dict[str, str]:
    allowed = valid_choices or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    direct_options = {}
    for label, option_text in re.findall(
        r"(?ims)^\s*([A-Z])\.\s*(.*?)(?=^\s*[A-Z]\.\s*|\Z)", problem
    ):
        label = label.upper()
        option_text = re.sub(r"\s+", " ", option_text).strip()
        if label in allowed and option_text:
            direct_options[label] = option_text

    alias_like = (
        direct_options
        and all(re.fullmatch(r"[a-z]", text.strip(), flags=re.IGNORECASE) for text in direct_options.values())
    )
    if direct_options and not alias_like:
        return direct_options

    original_options = {}
    for label, option_text in re.findall(
        r"(?ims)^\s*([a-d])\)\s*(.*?)(?=^\s*[a-d]\)\s*|^\s*[A-D]\.\s*[a-d]\s*$|\Z)",
        problem,
    ):
        original_options[label.upper()] = re.sub(r"\s+", " ", option_text).strip()

    if alias_like and original_options:
        remapped = {}
        for final_label, original_label in direct_options.items():
            original_label = original_label.strip().upper()
            if original_label in original_options:
                remapped[final_label] = original_options[original_label]
        if remapped:
            return remapped

    return direct_options


def extract_choice_from_option_text(
    generated_text: str,
    options: Mapping[str, str],
) -> Optional[str]:
    if not options:
        return None

    normalized_generation = _normalize_option_text(answer_window(strip_thinking(generated_text), 8192))
    if not normalized_generation:
        return None

    matches = []
    for label, option_text in options.items():
        normalized_option = _normalize_option_text(option_text)
        if len(normalized_option) < 4:
            continue

        if normalized_option in normalized_generation:
            matches.append((label, len(normalized_option)))
            continue

        tokens = [token for token in normalized_option.split() if len(token) >= 3]
        if not tokens:
            continue
        overlap = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", normalized_generation))
        if len(tokens) >= 4 and overlap / len(tokens) >= 0.85:
            matches.append((label, overlap))

    if not matches:
        return None

    matches.sort(key=lambda item: item[1], reverse=True)
    if len(matches) > 1 and matches[0][1] == matches[1][1]:
        return None
    return matches[0][0]


def extract_choice_for_problem(
    text: str,
    problem: str,
    valid_choices: ChoiceSet = None,
) -> Optional[str]:
    json_choice = extract_json_choice_for_problem(text, problem, valid_choices)
    if json_choice is not None:
        return json_choice

    jsonish_choice = extract_jsonish_choice_for_problem(text, problem, valid_choices)
    if jsonish_choice is not None:
        return jsonish_choice

    choice = extract_choice(text, valid_choices)
    if choice is not None:
        return choice

    options = parse_mcq_options(problem, valid_choices)
    choice = extract_choice_from_option_text(text, options)
    if choice is not None and _choice_allowed(choice, valid_choices):
        return choice
    return None


def extract_json_choice_for_problem(
    text: str,
    problem: str,
    valid_choices: ChoiceSet = None,
) -> Optional[str]:
    if not text:
        return None

    answer_text = answer_window(strip_thinking(text))
    json_answers = []
    for obj in iter_json_objects(answer_text):
        for key, value in obj.items():
            if key.lower() in ANSWER_KEYS:
                choice = extract_choice_from_answer_value(value, problem, valid_choices)
                if choice is not None:
                    json_answers.append(choice)

    if json_answers:
        return json_answers[-1]
    return None


def extract_jsonish_choice_for_problem(
    text: str,
    problem: str,
    valid_choices: ChoiceSet = None,
) -> Optional[str]:
    if not text:
        return None

    answer_text = answer_window(strip_thinking(text))
    patterns = [
        r"""(?is)["']?(?:answer|choice)["']?\s*[:=]\s*\{\s*["']?(?:answer|choice)["']?\s*[:=]\s*["']?\s*([A-Za-z])\s*["']?\s*\}""",
        r"""(?is)["']?(?:answer|choice)["']?\s*[:=]\s*\{\s*["']?\s*([A-Za-z])\s*["']?\s*\}""",
        r"""(?im)["']?(?:answer|choice)["']?\s*[:=]\s*["']?\s*([A-Za-z])\s*["']?""",
        r"""(?is)\{\s*["']?(?:answer|choice)["']?\s*[:=]\s*["']?\s*([A-Za-z])\s*["']?\s*\}""",
    ]
    choices = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer_text):
            candidate = match.group(1)
            prefix = answer_text[max(0, match.start() - 16) : match.start()].lower()
            if re.search(r"\bnot\s*$", prefix):
                continue
            choice = normalize_choice_for_problem(candidate, problem, valid_choices)
            if choice is not None:
                choices.append(choice)

    if choices:
        return choices[-1]
    return None


def strip_thinking(text: str) -> str:
    # Qwen-style thinking often contains many candidate letters. Grade the final answer.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def answer_window(text: str, edge_chars: int = 4096) -> str:
    if len(text) <= edge_chars * 2:
        return text
    return text[:edge_chars] + "\n...\n" + text[-edge_chars:]


def iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
    if "{" not in text or "}" not in text:
        return

    candidates = [text]
    candidates.extend(
        match.group(1)
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    )

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                yield parsed

        for match in re.finditer(r"\{[^{}]{0,2000}\}", candidate, flags=re.DOTALL):
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


def extract_choice(text: str, valid_choices: ChoiceSet = None) -> Optional[str]:
    if not text:
        return None

    answer_text = answer_window(strip_thinking(text))

    json_answers = []
    for obj in iter_json_objects(answer_text):
        for key, value in obj.items():
            if key.lower() in ANSWER_KEYS:
                choice = normalize_choice(value, valid_choices)
                if choice is not None:
                    json_answers.append(choice)
    if json_answers:
        return json_answers[-1]

    patterns = [
        r"(?is)[\"'](?:answer|choice)[\"']\s*:\s*[\"']?\s*[\(\[]?\s*([A-Z])\s*[\)\]]?\s*[\"']?",
        r"(?is)<answer>\s*[\(\[]?\s*([A-Z])\s*[\)\]]?\s*</answer>",
        r"(?im)\b(?:final\s+answer|answer|ans|정답)\s*[:=：]\s*[\(\[]?\s*([A-Z])\s*[\)\]]?\b",
        r"(?im)\b(?:the\s+answer\s+is|answer\s+is|correct\s+answer\s+is)\s*[\"']?[\(\[]?\s*([A-Z])\s*[\)\]]?[\"']?\b",
        r"(?im)\b(?:option|choice)\s+([A-Z])\b",
        r"(?im)^\s*[\(\[]?\s*([A-Z])\s*[\)\]\.]?\s*$",
        r"(?im)[\(\[]\s*([A-Z])\s*[\)\]]\s*$",
        r"(?im)\b([A-Z])\s*[\.\)]?\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, answer_text)
        for raw_candidate in reversed(matches):
            choice = normalize_choice(raw_candidate, valid_choices)
            if choice is not None:
                return choice

    return None


def grade_answer(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    if predicted is None or ground_truth is None:
        return False
    return predicted == ground_truth


def build_prompt(problem: str) -> str:
    instruction = (
        "Return only one JSON object with the key \"answer\". "
        "The value must be exactly one choice letter, for example: {\"answer\": \"C\"}."
    )
    return f"{problem}\n\n{instruction}"


def evaluate_mcq(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    num_samples: Optional[int],
    output_file: Optional[str],
    lora_request,
    dataset_name: str,
    base_model_name: Optional[str],
    enable_thinking: bool,
    use_chat_template: bool,
    val_n: int,
    answer_index_base: int,
):
    from datasets import load_dataset
    from vllm import SamplingParams

    print(f"\n{'='*70}")
    print("MCQ EVALUATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Chat Template: {'ENABLED' if use_chat_template else 'DISABLED'}")
    print(f"Temperature: {temperature}")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"Answer index base: {answer_index_base}")
    print(f"{'='*70}\n")

    if dataset_name.lower() == "chemistry":
        dataset = load_dataset("json", data_files="./datasets/chemistry/test.json", split="train")
        print(f"Loaded chemistry dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "gpqa-d":
        dataset = load_dataset("fingertap/GPQA-Diamond", split="test", trust_remote_code=True)
        print(f"Loaded GPQA-Diamond dataset with {len(dataset)} problems")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'chemistry' or 'gpqa-d'.")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    all_prompts = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []
    all_valid_choices = []
    missing_gt_count = 0

    for row_idx, example in enumerate(dataset):
        problem = first_present(example, "prompt", "question")
        if problem is None:
            raise ValueError("Each example must contain a 'prompt' or 'question' field.")

        valid_choices = _as_choice_set(infer_choice_letters(problem))
        answer_value = first_present(example, "answer", "label", "target", "gold", "correct_answer")
        gt_answer = normalize_ground_truth(answer_value, valid_choices, answer_index_base)
        if gt_answer is None:
            missing_gt_count += 1

        question_id = first_present(example, "idx", "id", "question_id")
        if question_id is None:
            question_id = row_idx

        user_message = build_prompt(problem)
        all_problems.append(problem)
        all_gt_answers.append(gt_answer)
        all_question_ids.append(question_id)
        all_valid_choices.append(valid_choices)

        if use_chat_template:
            messages = [{"role": "user", "content": user_message}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            text = user_message
        all_prompts.append(text)

    if missing_gt_count:
        print(
            f"Warning: {missing_gt_count}/{len(dataset)} examples have no normalized ground-truth answer. "
            "They will be counted as incorrect."
        )

    print(f"\nRunning vLLM batch inference on {len(all_prompts)} problems...")

    if lora_request is not None:
        outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    total = 0
    formatted_count = 0
    results = []

    pass_at_n = 0
    total_correct_per_problem = 0

    for idx, (output, problem, gt_answer, question_id, valid_choices) in enumerate(
        zip(outputs, all_problems, all_gt_answers, all_question_ids, all_valid_choices)
    ):
        generations = []
        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []

        for generation in output.outputs:
            generated_text = generation.text
            predicted_answer = extract_choice_for_problem(generated_text, problem, valid_choices)
            is_formatted = predicted_answer is not None
            is_correct = grade_answer(predicted_answer, gt_answer)

            generations.append(generated_text)
            predicted_answers.append(predicted_answer if predicted_answer else "[No answer found]")
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)

        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        majority_vote_correct = False
        if num_formatted > 0:
            formatted_predictions = [
                pred for pred, formatted in zip(predicted_answers, is_formatted_list) if formatted
            ]
            most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
            majority_vote_correct = grade_answer(most_common_answer, gt_answer)

        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += len(output.outputs)

        result = {
            "problem_id": question_id,
            "problem": problem,
            "ground_truth": gt_answer,
            "valid_choices": sorted(valid_choices) if valid_choices else None,
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
            "predicted_answer": predicted_answers[0] if predicted_answers else "[No answer found]",
            "full_generation": generations[0] if generations else "",
            "correct": is_correct_list[0] if is_correct_list else False,
            "formatted": is_formatted_list[0] if is_formatted_list else False,
        }
        results.append(result)

        format_rate = formatted_count / total * 100 if total else 0.0
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100 if total else 0.0

        status = "OK" if has_correct else "X"
        print(
            f"{status} [{idx + 1}/{len(dataset)}] Pass@{val_n}: {current_pass_at_n:.1f}% | "
            f"Avg@{val_n}: {current_avg_at_n:.1f}% | Formatted: {format_rate:.1f}%"
        )

    num_problems = len(dataset)
    format_rate = formatted_count / total * 100 if total else 0.0
    pass_at_n_pct = pass_at_n / num_problems * 100 if num_problems else 0.0
    average_at_n_pct = total_correct_per_problem / total * 100 if total else 0.0

    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100 if num_problems else 0.0

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print("\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})")
    print(
        f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% "
        f"({majority_vote_correct_count}/{num_problems})"
    )
    print("\nFormatting:")
    print(f"  Answers found: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "dataset": dataset_name,
            "use_chat_template": use_chat_template,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "answer_index_base": answer_index_base,
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
            "missing_ground_truth_count": missing_gt_count,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def print_final_results(
    dataset_name: Optional[str],
    num_problems: int,
    val_n: int,
    total: int,
    pass_at_n: int,
    pass_at_n_pct: float,
    average_at_n: int,
    average_at_n_pct: float,
    majority_vote_at_n: int,
    majority_vote_at_n_pct: float,
    formatted_count: int,
    format_rate: float,
) -> None:
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    if dataset_name is not None:
        print(f"Dataset: {dataset_name}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print("\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({average_at_n}/{total})")
    print(
        f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% "
        f"({majority_vote_at_n}/{num_problems})"
    )
    print("\nFormatting:")
    print(f"  Answers found: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)


def valid_choices_for_saved_result(result: Mapping[str, Any]) -> ChoiceSet:
    saved_choices = result.get("valid_choices")
    valid_choices = _as_choice_set(saved_choices) if saved_choices else None
    if valid_choices is not None:
        return valid_choices

    problem = result.get("problem")
    if isinstance(problem, str) and problem.strip():
        return _as_choice_set(infer_choice_letters(problem))

    ground_truth = normalize_choice(result.get("ground_truth"))
    if ground_truth in set("ABCD"):
        return set("ABCD")
    return None


def regrade_existing_results_file(output_file: str, answer_index_base: int = 0) -> tuple[float, int]:
    output_path = Path(output_file)
    with open(output_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Existing result file has no list-valued 'results': {output_file}")

    total = 0
    formatted_count = 0
    pass_at_n = 0
    total_correct_per_problem = 0
    missing_gt_count = 0

    for idx, result in enumerate(results):
        problem = result.get("problem")
        if not isinstance(problem, str):
            problem = ""

        valid_choices = valid_choices_for_saved_result(result)
        gt_answer = normalize_ground_truth(
            result.get("ground_truth"),
            valid_choices,
            answer_index_base,
        )
        if gt_answer is None:
            missing_gt_count += 1
        else:
            result["ground_truth"] = gt_answer

        generations_data = result.get("generations")
        if not isinstance(generations_data, list):
            generations_data = [
                {
                    "full_generation": result.get("full_generation", ""),
                    "predicted_answer": result.get("predicted_answer", "[No answer found]"),
                }
            ]
            result["generations"] = generations_data

        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []
        full_generations = []

        for generation in generations_data:
            if not isinstance(generation, dict):
                generation = {"full_generation": str(generation)}

            generated_text = generation.get("full_generation", "")
            if generated_text is None:
                generated_text = ""
            generated_text = str(generated_text)

            predicted_answer = extract_choice_for_problem(generated_text, problem, valid_choices)
            is_formatted = predicted_answer is not None
            is_correct = grade_answer(predicted_answer, gt_answer)

            generation["predicted_answer"] = predicted_answer if predicted_answer else "[No answer found]"
            generation["correct"] = is_correct
            generation["formatted"] = is_formatted
            generation["full_generation"] = generated_text

            predicted_answers.append(generation["predicted_answer"])
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)
            full_generations.append(generated_text)

        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        majority_vote_correct = False
        if num_formatted > 0:
            formatted_predictions = [
                pred for pred, formatted in zip(predicted_answers, is_formatted_list) if formatted
            ]
            most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
            majority_vote_correct = grade_answer(most_common_answer, gt_answer)

        if valid_choices is not None:
            result["valid_choices"] = sorted(valid_choices)
        result["problem_id"] = result.get("problem_id", idx)
        result["val_n"] = len(generations_data)
        result["num_correct"] = num_correct
        result["pass_at_n"] = has_correct
        result["majority_vote_correct"] = majority_vote_correct
        result["predicted_answer"] = predicted_answers[0] if predicted_answers else "[No answer found]"
        result["full_generation"] = full_generations[0] if full_generations else ""
        result["correct"] = is_correct_list[0] if is_correct_list else False
        result["formatted"] = is_formatted_list[0] if is_formatted_list else False

        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += len(generations_data)

    num_problems = len(results)
    val_n = int(summary.get("val_n") or (len(results[0].get("generations", [])) if results else 1))
    format_rate = formatted_count / total * 100 if total else 0.0
    pass_at_n_pct = pass_at_n / num_problems * 100 if num_problems else 0.0
    average_at_n_pct = total_correct_per_problem / total * 100 if total else 0.0
    majority_vote_correct_count = sum(1 for result in results if result["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100 if num_problems else 0.0

    summary["num_problems"] = num_problems
    summary["total_solutions"] = total
    summary["pass_at_n"] = pass_at_n
    summary["pass_at_n_pct"] = pass_at_n_pct
    summary["average_at_n"] = total_correct_per_problem
    summary["average_at_n_pct"] = average_at_n_pct
    summary["majority_vote_at_n"] = majority_vote_correct_count
    summary["majority_vote_at_n_pct"] = majority_vote_at_n_pct
    summary["formatted_count"] = formatted_count
    summary["format_rate"] = format_rate
    summary["missing_ground_truth_count"] = missing_gt_count
    summary["results"] = results

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    tmp_path.replace(output_path)

    print_final_results(
        dataset_name=summary.get("dataset"),
        num_problems=num_problems,
        val_n=val_n,
        total=total,
        pass_at_n=pass_at_n,
        pass_at_n_pct=pass_at_n_pct,
        average_at_n=total_correct_per_problem,
        average_at_n_pct=average_at_n_pct,
        majority_vote_at_n=majority_vote_correct_count,
        majority_vote_at_n_pct=majority_vote_at_n_pct,
        formatted_count=formatted_count,
        format_rate=format_rate,
    )
    print(f"\nExisting results regraded in place: {output_file}")
    return average_at_n_pct, val_n


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice QA datasets")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to checkpoint directory with LoRA adapters. If not provided, will use base model only.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["chemistry", "gpqa-d"],
        help="Dataset to use for evaluation",
    )
    parser.add_argument("--max_new_tokens", type=int, required=True, help="Maximum tokens to generate")
    thinking_group = parser.add_mutually_exclusive_group(required=True)
    thinking_group.add_argument("--enable_thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, required=True, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, required=True, help="Minimum probability threshold")
    parser.add_argument("--presence_penalty", type=float, required=True, help="Presence penalty")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Repository root used for default evaluation output paths.",
    )
    parser.add_argument("--output_file", type=str, help="Path to save detailed results JSON")
    parser.add_argument("--gpu_memory_utilization", type=float, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--max_len", type=int, required=True)
    chat_template_group = parser.add_mutually_exclusive_group(required=True)
    chat_template_group.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        help="Wrap eval prompts with tokenizer.apply_chat_template before generation.",
    )
    chat_template_group.add_argument(
        "--no_chat_template",
        dest="use_chat_template",
        action="store_false",
        help="Disable tokenizer chat template for evaluation prompts.",
    )
    parser.add_argument("--val_n", type=int, required=True, help="Number of solutions per problem")
    parser.add_argument(
        "--answer_index_base",
        type=int,
        choices=[0, 1],
        required=True,
        help="Index base for numeric ground-truth labels. Use 0 for 0=A, 1 for 1=A.",
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = build_tagged_eval_output_file(
            dataset_name=args.dataset,
            base_model=args.base_model,
            checkpoint_dir=args.checkpoint_dir,
            enable_thinking=args.enable_thinking,
            use_chat_template=args.use_chat_template,
            temperature=args.temperature,
            val_n=args.val_n,
            root_dir=Path(args.root_dir or PROJECT_ROOT) / "evaluations",
        )

    if Path(args.output_file).exists():
        print("\n" + "=" * 70)
        print("EXISTING EVALUATION RESULT FOUND")
        print("=" * 70)
        print(f"Output file: {args.output_file}")
        print("Skipping model loading and generation. Regrading saved responses in place.")
        print("=" * 70)

        average_at_n_pct, regraded_val_n = regrade_existing_results_file(
            args.output_file,
            answer_index_base=args.answer_index_base,
        )

        print("\n" + "=" * 70)
        print("REGRADING COMPLETE!")
        print("=" * 70)
        print(f"Final Average@{regraded_val_n}: {average_at_n_pct:.2f}%")
        print(f"Results updated in place: {args.output_file}")
        print("=" * 70 + "\n")
        return

    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            print("\n" + "=" * 70)
            print("ERROR: Checkpoint directory does not exist")
            print("=" * 70)
            print(f"Provided checkpoint directory: {args.checkpoint_dir}")
            print("This directory does not exist.")
            print(
                "\nPlease provide a valid checkpoint directory or omit --checkpoint_dir to use the base model only."
            )
            print("=" * 70 + "\n")
            raise SystemExit(1)

    if args.enable_thinking and args.temperature == 0.0:
        print("\n" + "!" * 70)
        print("WARNING: Using greedy decoding (temperature=0.0) in thinking mode!")
        print("Consider temperature=0.6 for Qwen3 thinking mode.")
        print("!" * 70 + "\n")

    print(f"Results will be saved to: {args.output_file}")

    print_section(
        "MCQ EVALUATION",
        {
            "Dataset": args.dataset,
            "Base model": args.base_model,
            "Checkpoint": args.checkpoint_dir or "None (base model only)",
            "Thinking Mode": "ENABLED" if args.enable_thinking else "DISABLED",
            "Chat Template": "ENABLED" if args.use_chat_template else "DISABLED",
            "Max tokens": args.max_new_tokens,
            "Temperature": args.temperature,
            "Top-p": args.top_p,
            "Top-k": args.top_k,
            "Min-p": args.min_p,
            "Presence penalty": args.presence_penalty,
            "Num samples": args.num_samples or "All",
            "Val-N (solutions per problem)": args.val_n,
            "Answer index base": args.answer_index_base,
            "Output file": args.output_file,
            "GPU memory utilization": args.gpu_memory_utilization,
            "Tensor parallel size": args.tensor_parallel_size,
            "Max model len": args.max_len,
        },
    )

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = create_lora_request(args.checkpoint_dir)

    average_at_n_pct, _ = evaluate_mcq(
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
        use_chat_template=args.use_chat_template,
        val_n=args.val_n,
        answer_index_base=args.answer_index_base,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
