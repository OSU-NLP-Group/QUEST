#!/usr/bin/env python3
"""
GAIA judging script: reads result files and uses litellm to determine whether the
content inside the final <answer></answer> tag matches the dataset answer.
Supports multithreaded processing for faster evaluation.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from litellm import completion

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

# Global lock for thread-safe printing.
_print_lock = Lock()


def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract the last <answer></answer> block from a text blob."""
    if not text:
        return None

    matches = _ANSWER_TAG_RE.findall(text)
    if matches:
        return matches[-1].strip()

    return None


def extract_answer_from_record(data: Dict) -> Optional[str]:
    """
    Extract the predicted answer from a result record.

    Prefer the final <answer></answer> in messages, then fall back to prediction.
    """
    messages = data.get("messages", [])
    if messages:
        last_message = messages[-1]
        answer = extract_answer_from_text(last_message.get("content", ""))
        if answer is not None:
            return answer

    prediction = data.get("prediction", "")
    return extract_answer_from_text(prediction)


def normalize_question(text: str) -> str:
    """
    Normalize question text for robust matching.
    Removes whitespace differences, normalizes punctuation, and handles special characters.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text


def parse_judgment_to_bool(judgment: str) -> Optional[bool]:
    """Parse the model's judgment string into a boolean."""
    if not judgment:
        return None
    normalized = judgment.strip().lower()
    if normalized in {"true", "一致"}:
        return True
    if normalized in {"false", "不一致"}:
        return False
    if "true" in normalized and "false" not in normalized:
        return True
    if "false" in normalized and "true" not in normalized:
        return False
    return None


def call_litellm_for_judgment(ground_truth: str, predicted_answer: str, question: str) -> str:
    """Call a model through litellm to determine answer consistency."""
    model_name = os.environ.get("JUDGE_MODEL_NAME", "")

    if not model_name:
        raise ValueError("JUDGE_MODEL_NAME environment variable must be set")

    judgment_prompt = f"""Please determine whether the predicted answer matches the correct answer.

Question: {question}

Correct answer: {ground_truth}

Predicted answer:
{predicted_answer}

Compare the predicted answer and the correct answer carefully and decide whether they are consistent.

If the predicted answer explicitly contains the correct answer, or its core meaning matches the correct answer, respond "true".
If the predicted answer does not match or contradicts the correct answer, respond "false".
For multiple-choice questions, if the predicted answer mentions the correct option (e.g., "A", "B"), treat it as consistent.
For numerical answers, if the predicted answer contains the number, treat it as consistent.
Reply with only "true" or "false". Do not add any other content."""

    messages = [{"role": "user", "content": judgment_prompt}]

    call_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": 1,
        "num_retries": 2,
    }

    if model_name.startswith("azure/"):
        api_key = os.environ.get("JUDGE_AZURE_API_KEY")
        api_base = os.environ.get("JUDGE_AZURE_API_BASE")
        api_version = os.environ.get("JUDGE_AZURE_API_VERSION")
        if api_key:
            call_kwargs["api_key"] = api_key
        if api_base:
            call_kwargs["api_base"] = api_base
        if api_version:
            call_kwargs["api_version"] = api_version
    elif model_name.startswith("bedrock/"):
        aws_access_key_id = os.environ.get("JUDGE_AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("JUDGE_AWS_SECRET_ACCESS_KEY")
        aws_region_name = os.environ.get("JUDGE_AWS_REGION_NAME")
        if aws_access_key_id:
            call_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            call_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_region_name:
            call_kwargs["aws_region_name"] = aws_region_name
    else:
        api_key = os.environ.get("JUDGE_OPENAI_API_KEY")
        if api_key:
            call_kwargs["api_key"] = api_key

    try:
        response = completion(**call_kwargs)
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as exc:
        print(f"Error calling litellm: {exc}")
        return ""


def load_answer_map(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """
    Build a normalized question -> {id, answer} mapping from the GAIA dataset file.

    Supports either:
    - a JSON array of objects
    - a JSONL file
    """
    answer_map: Dict[str, Dict[str, str]] = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return answer_map

    try:
        dataset = json.loads(raw)
        if isinstance(dataset, list):
            records = dataset
        else:
            records = [dataset]
    except json.JSONDecodeError:
        records = []
        for line_num, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Error parsing JSON in dataset {dataset_path}, line {line_num}: {exc}")

    for data in records:
        question = normalize_question(data.get("Question") or data.get("question", ""))
        answer = str(data.get("answer", "")).strip()
        question_id = str(data.get("task_id") or data.get("id", "")).strip()
        if question and question not in answer_map:
            answer_map[question] = {"id": question_id, "answer": answer}

    return answer_map


def process_single_question(
    data: Dict,
    line_num: int,
    file_path: str,
    answer_map: Dict[str, Dict[str, str]],
    verbose: bool = True,
) -> Dict:
    """Process a single result row and return the judgment result."""
    question = data.get("question", "")
    question_key = normalize_question(question)

    matched_data = answer_map.get(question_key, {})
    question_id = matched_data.get("id", "")
    ground_truth = matched_data.get("answer", "")

    if verbose:
        with _print_lock:
            print(f"[Line {line_num}] Processing question: {question[:80]}...")
            print(f"  Matched ID: {question_id}")
            print(f"  Ground truth: {ground_truth}")

    # predicted_answer = extract_answer_from_record(data)
    predicted_answer = data.get("prediction", "").strip()

    if predicted_answer is None:
        if verbose:
            with _print_lock:
                print(f"  Warning: No <answer></answer> tag found in line {line_num}")
        return {
            "file": os.path.basename(file_path),
            "line": line_num,
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": None,
            "judgment": "Answer tag not found",
            "is_correct": False,
        }

    if not ground_truth:
        if verbose:
            with _print_lock:
                print(f"  Warning: Empty ground_truth in line {line_num}")
                if not question_id:
                    print("  Warning: Question not found in dataset (possible mismatch)")
        return {
            "file": os.path.basename(file_path),
            "line": line_num,
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "judgment": "Ground-truth answer is empty or missing; cannot judge",
            "is_correct": False,
        }

    judgment = call_litellm_for_judgment(ground_truth, predicted_answer, question)
    parsed = parse_judgment_to_bool(judgment)
    is_correct = bool(parsed) if parsed is not None else False

    if verbose:
        with _print_lock:
            print(f"  Judgment: {judgment}, Is correct: {is_correct}")

    return {
        "file": os.path.basename(file_path),
        "line": line_num,
        "question_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "judgment": judgment,
        "is_correct": is_correct,
    }


def load_all_questions(target_dir: str, verbose: bool = True) -> List[Dict]:
    """Load all JSONL files in the directory and return records with metadata."""
    records: List[Dict] = []
    jsonl_files = sorted(Path(target_dir).glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        if verbose:
            with _print_lock:
                print(f"Loading file: {jsonl_file.name}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    records.append(
                        {"data": data, "line": line_num, "file_path": str(jsonl_file)}
                    )
                except json.JSONDecodeError as exc:
                    if verbose:
                        with _print_lock:
                            print(f"Error parsing JSON in file {jsonl_file}, line {line_num}: {exc}")
                    continue
    return records


def main():
    parser = argparse.ArgumentParser(
        description="GAIA judging script: use an LLM to determine whether answers are correct"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=(
            "./results/deepresearch/"
            "gaia-text-only-103"
        ),
        help="Directory containing result JSONL files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaia-103-org.json",
        help="GAIA dataset file path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose logs (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode; suppress detailed logs",
    )

    args = parser.parse_args()

    target_dir = args.target_dir
    dataset_path = args.dataset
    workers = args.workers
    verbose = args.verbose and not args.quiet

    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        return

    jsonl_files = sorted(Path(target_dir).glob("*.jsonl"))
    if not jsonl_files:
        print(f"No jsonl files found in {target_dir}")
        return

    print(f"\n{'=' * 80}")
    print("Configuration:")
    print(f"  Target directory: {target_dir}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Files to process: {len(jsonl_files)}")
    print(f"  Workers: {workers}")
    print(f"  Verbose: {verbose}")
    print(f"{'=' * 80}\n")

    print("Loading answer map from dataset...")
    answer_map = load_answer_map(dataset_path)
    if not answer_map:
        print(f"Error: No answers loaded from dataset: {dataset_path}")
        return
    print(f"Loaded {len(answer_map)} questions from dataset\n")

    records = load_all_questions(target_dir, verbose)
    if not records:
        print("No records found to process")
        return
    print(f"\nProcessing {len(records)} questions with {workers} workers...\n")

    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_rec = {
            executor.submit(
                process_single_question,
                rec["data"],
                rec["line"],
                rec["file_path"],
                answer_map,
                verbose,
            ): rec
            for rec in records
        }
        for future in as_completed(future_to_rec):
            rec = future_to_rec[future]
            try:
                all_results.append(future.result())
            except Exception as exc:
                if verbose:
                    with _print_lock:
                        print(f"Error processing file {rec['file_path']}, line {rec['line']}: {exc}")
                continue

    all_results.sort(key=lambda x: (x["file"], x["line"]))
    sorted_filenames = [f.name for f in jsonl_files]

    question_results = {}
    for result in all_results:
        question = result.get("question", "")
        file_name = result.get("file", "")
        is_correct = result.get("is_correct", False)
        question_results.setdefault(question, {})
        if file_name not in question_results[question]:
            question_results[question][file_name] = is_correct
        elif is_correct:
            question_results[question][file_name] = True

    total_questions = len(question_results)

    print(f"\n=== Best@k Statistics (Total questions: {total_questions}) ===")

    best_at_k_stats = {}
    question_solved_status = {question: False for question in question_results}

    for k, filename in enumerate(sorted_filenames, 1):
        for question in question_results:
            if not question_solved_status[question] and question_results[question].get(filename, False):
                question_solved_status[question] = True

        correct_count = sum(1 for solved in question_solved_status.values() if solved)
        accuracy = correct_count / total_questions if total_questions > 0 else 0

        print(f"Best@{k}: {accuracy:.2%} ({correct_count}/{total_questions})")
        best_at_k_stats[f"best@{k}"] = {
            "correct": correct_count,
            "total": total_questions,
            "accuracy": accuracy,
        }

    output_file = os.path.join(target_dir, "judgment_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    summary = {"total_questions": total_questions, "best_at_k": best_at_k_stats, "by_file": {}}

    per_file_question = {}
    for result in all_results:
        file_name = result["file"]
        question = result.get("question", "")
        per_file_question.setdefault(file_name, {})
        if question not in per_file_question[file_name]:
            per_file_question[file_name][question] = False
        if result.get("is_correct", False):
            per_file_question[file_name][question] = True

    for file_name, question_map in per_file_question.items():
        total_q = len(question_map)
        correct_q = sum(1 for is_correct in question_map.values() if is_correct)
        summary["by_file"][file_name] = {
            "total": total_q,
            "correct": correct_q,
            "accuracy": correct_q / total_q if total_q > 0 else 0,
        }

    summary_file = os.path.join(target_dir, "judgment_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary statistics saved to: {summary_file}")


if __name__ == "__main__":
    main()
