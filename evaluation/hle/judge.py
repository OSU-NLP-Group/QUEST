#!/usr/bin/env python3
"""
Judging script: reads result files and uses litellm to determine whether the content inside
the final <answer></answer> tag matches the dataset answer.
Supports multithreaded processing for faster evaluation.
"""

import json
import os
import re
from typing import Dict, List, Optional
from pathlib import Path
from litellm import completion
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SCRIPT_DIR = Path(__file__).resolve().parent

# Global lock for thread-safe printing.
_print_lock = Lock()


def extract_answer_from_messages(messages: List[Dict]) -> Optional[str]:
    """
    Extract the content inside <answer></answer> from the last message.

    Args:
        messages: message list

    Returns:
        Extracted answer content, or None if no answer tag is found.
    """
    if not messages:
        return None
    
    last_message = messages[-1]
    content = last_message.get("content", "")
    
    # Extract the content inside <answer></answer> and keep the last occurrence.
    matches = _ANSWER_TAG_RE.findall(content)
    if matches:
        return matches[-1].strip()
    
    return None


def normalize_question(text: str) -> str:
    """
    Normalize question text for robust matching.
    Removes whitespace differences, normalizes punctuation, and handles special characters.
    """
    if not text:
        return ""
    # Normalize whitespace to single spaces.
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace.
    text = text.strip()
    # Normalize quotation marks (convert smart quotes to plain quotes).
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Normalize dashes.
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Remove zero-width characters.
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text


def parse_judgment_to_bool(judgment: str) -> Optional[bool]:
    """
    Parse the model's judgment string into a boolean.

    Returns:
        True/False or None if the output cannot be parsed.
    """
    if not judgment:
        return None
    normalized = judgment.strip().lower()
    if normalized in {"true", "一致"}:
        return True
    if normalized in {"false", "不一致"}:
        return False
    # Handle outputs that contain additional whitespace or newlines.
    if "true" in normalized and "false" not in normalized:
        return True
    if "false" in normalized and "true" not in normalized:
        return False
    return None


def call_litellm_for_judgment(ground_truth: str, predicted_answer: str, question: str) -> str:
    """
    Call a model through litellm to determine whether the predicted answer matches the ground truth.

    Args:
        ground_truth: correct answer
        predicted_answer: predicted answer extracted from <answer></answer>
        question: question text

    Returns:
        The judgment returned by the model.
    """
    # Use environment variables with the JUDGE_* prefix.
    model_name = os.environ.get("JUDGE_MODEL_NAME", "")
    
    if not model_name:
        raise ValueError("JUDGE_MODEL_NAME environment variable must be set")
    
    # Build the judgment prompt.
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
    
    # Prepare messages.
    messages = [
        {
            "role": "user",
            "content": judgment_prompt
        }
    ]
    
    # Prepare litellm call arguments.
    call_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": 1,  # Lower temperature for more stable judgments.
        "num_retries": 2
    }
    
    # Add the appropriate API configuration based on model type.
    if model_name.startswith("azure/"):
        # Azure OpenAI configuration.
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
        # AWS Bedrock configuration.
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
        # OpenAI configuration.
        api_key = os.environ.get("JUDGE_OPENAI_API_KEY")
        if api_key:
            call_kwargs["api_key"] = api_key
    
    # Call litellm.
    try:
        response = completion(**call_kwargs)
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"Error calling litellm: {e}")
        return ""


def load_answer_map(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """
    Build a question -> {id, answer} mapping from the dataset file.
    """
    answer_map: Dict[str, Dict[str, str]] = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                question = normalize_question(data.get("question", ""))
                answer = data.get("answer", "")
                question_id = data.get("id", "")
                if question and question not in answer_map:
                    answer_map[question] = {
                        "id": question_id,
                        "answer": answer
                    }
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in dataset {dataset_path}, line {line_num}: {e}")
                continue
    return answer_map


def process_single_question(
    data: Dict,
    line_num: int,
    file_path: str,
    answer_map: Dict[str, Dict[str, str]],
    verbose: bool = True
) -> Dict:
    """
    Process a single question and return the judgment result.

    Args:
        data: one question record
        line_num: line number
        file_path: file path
        answer_map: question -> {id, answer} mapping
        verbose: whether to print verbose logs

    Returns:
        A judgment result dictionary.
    """
    question = data.get("question", "")
    question_key = normalize_question(question)

    # Match the id and answer through the normalized question text.
    matched_data = answer_map.get(question_key, {})
    question_id = matched_data.get("id", "")
    ground_truth = matched_data.get("answer", "")

    if verbose:
        with _print_lock:
            print(f"[Line {line_num}] Processing question: {question[:80]}...")
            print(f"  Matched ID: {question_id}")
            print(f"  Ground truth: {ground_truth}")

    messages = data.get("messages", [])

    # Extract the predicted answer.
    predicted_answer = extract_answer_from_messages(messages)

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
            "is_correct": False
        }

    # Skip judging if ground_truth is empty.
    if not ground_truth or not ground_truth.strip():
        if verbose:
            with _print_lock:
                print(f"  Warning: Empty ground_truth in line {line_num}")
                if not question_id:
                    print(f"  Warning: Question not found in dataset (possible mismatch)")
        return {
            "file": os.path.basename(file_path),
            "line": line_num,
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "judgment": "Ground-truth answer is empty or missing; cannot judge",
            "is_correct": False
        }

    # Call the model for judgment.
    judgment = call_litellm_for_judgment(ground_truth, predicted_answer, question)

    # Parse correctness, supporting true/false outputs in either English or Chinese.
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
        "is_correct": is_correct
    }


def load_all_questions(target_dir: str, verbose: bool = True) -> List[Dict]:
    """
    Load all JSONL files in the directory at once and return records with file names and line numbers.
    """
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
                    records.append({
                        "data": data,
                        "line": line_num,
                        "file_path": str(jsonl_file)
                    })
                except json.JSONDecodeError as e:
                    if verbose:
                        with _print_lock:
                            print(f"Error parsing JSON in file {jsonl_file}, line {line_num}: {e}")
                    continue
    return records


def main():
    """
    Main entry point: load all JSONL files in the directory and judge them with multithreading.
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Judging script: use an LLM to determine whether answers are correct (multithreaded)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Directory containing the result files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(SCRIPT_DIR / "hle_text_only_130.jsonl"),
        help="Dataset file path"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose logs (default: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode; suppress detailed logs"
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

    # Collect all JSONL files.
    jsonl_files = sorted(Path(target_dir).glob("*.jsonl"))
    if not jsonl_files:
        print(f"No jsonl files found in {target_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Target directory: {target_dir}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Files to process: {len(jsonl_files)}")
    print(f"  Workers: {workers}")
    print(f"  Verbose: {verbose}")
    print(f"{'='*80}\n")

    # Build the question-to-answer mapping.
    print("Loading answer map from dataset...")
    answer_map = load_answer_map(dataset_path)
    if not answer_map:
        print(f"Error: No answers loaded from dataset: {dataset_path}")
        return
    print(f"Loaded {len(answer_map)} questions from dataset\n")

    # Load all files first, then process them in parallel.
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
                verbose
            ): rec
            for rec in records
        }
        for future in as_completed(future_to_rec):
            rec = future_to_rec[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                if verbose:
                    with _print_lock:
                        print(f"Error processing file {rec['file_path']}, line {rec['line']}: {e}")
                continue

    # Sort by file name and line number to keep the output stable.
    all_results.sort(key=lambda x: (x["file"], x["line"]))
    
    # Collect all involved file names in sorted order.
    sorted_filenames = [f.name for f in jsonl_files]

    # Preprocess results by grouping them per question.
    # question -> { filename -> is_correct }
    question_results = {}
    for r in all_results:
        q = r.get("question", "")
        f_name = r.get("file", "")
        is_correct = r.get("is_correct", False)
        
        if q not in question_results:
            question_results[q] = {}
        
        # Record whether this question is correct in the current file.
        if f_name not in question_results[q]:
            question_results[q][f_name] = is_correct
        elif is_correct:
            question_results[q][f_name] = True

    total_questions = len(question_results)
    
    print(f"\n=== Best@k Statistics (Total questions: {total_questions}) ===")
    
    best_at_k_stats = {}
    # Track whether each question has already been solved.
    question_solved_status = {q: False for q in question_results}
    
    for k, filename in enumerate(sorted_filenames, 1):
        # Update solved status using the current file.
        for q in question_results:
            if not question_solved_status[q]:
                if question_results[q].get(filename, False):
                    question_solved_status[q] = True
        
        correct_count = sum(1 for v in question_solved_status.values() if v)
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        print(f"Best@{k}: {accuracy:.2%} ({correct_count}/{total_questions})")
        
        best_at_k_stats[f"best@{k}"] = {
            "correct": correct_count,
            "total": total_questions,
            "accuracy": accuracy
        }
    
    # Save detailed results to a JSON file.
    output_file = os.path.join(target_dir, "judgment_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save a simplified summary.
    summary = {
        "total_questions": total_questions,
        "best_at_k": best_at_k_stats,
        "by_file": {}
    }
    
    per_file_question = {}
    for result in all_results:
        file_name = result["file"]
        q = result.get("question", "")
        if file_name not in per_file_question:
            per_file_question[file_name] = {}
        if q not in per_file_question[file_name]:
            per_file_question[file_name][q] = False
        if result.get("is_correct", False):
            per_file_question[file_name][q] = True
    
    for file_name, q_map in per_file_question.items():
        total_q = len(q_map)
        correct_q = sum(1 for v in q_map.values() if v)
        summary["by_file"][file_name] = {
            "total": total_q,
            "correct": correct_q,
            "accuracy": correct_q / total_q if total_q > 0 else 0
        }
    
    summary_file = os.path.join(target_dir, "judgment_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Summary statistics saved to: {summary_file}")


if __name__ == "__main__":
    main()
