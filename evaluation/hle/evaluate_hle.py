"""
HLE evaluation script:
- File handling matches judge.py: --target-dir points to a directory and processes all .jsonl files in it;
  --dataset provides the reference answers; answers are extracted from <answer></answer> in each messages field
  and matched against answer_map.
- Judging uses the same model and prompt setup as evaluate_hle_official.py
  (OpenAI + JUDGE_PROMPT + ExtractedAnswer).
- Pass@k / Best@k are computed automatically: after sorting by filename, a question counts as passed if
  any of the first k files is correct. Outputs include pass@1, pass@2, ... plus best_at_k and by_file.
- Outputs: judgment_results.json and judgment_summary.json (including pass_at_k, best_at_k, and by_file).
"""
from openai import OpenAI
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, local
import argparse
from tqdm import tqdm

# ---------- Judge configuration aligned with evaluate_hle_official.py ----------
JUDGE_MODEL = "o4-mini"
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")

thread_local = local()


def get_client():
    if not hasattr(thread_local, "client"):
        kwargs = {"api_key": API_KEY}
        if BASE_URL:
            kwargs["base_url"] = BASE_URL
        thread_local.client = OpenAI(**kwargs)
    return thread_local.client


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


def extract_answer(question: str, correct_answer: str, response: str) -> Optional[Dict]:
    """
    Judge call aligned with evaluate_hle_official.py.
    Returns a dict containing fields such as correct/reasoning on success;
    returns a dict with an "error" key on failure for easier debugging.
    """
    client = get_client()
    prompt = JUDGE_PROMPT.format(
        question=question, correct_answer=correct_answer, response=response
    )
    last_error = None
    for attempt in range(6):
        try:
            response_obj = client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                max_completion_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedAnswer,
                timeout=60.0,
            )
            content = response_obj.choices[0].message.parsed
            return {
                "correct_answer": correct_answer,
                "model_answer": content.extracted_final_answer,
                "reasoning": content.reasoning,
                "correct": content.correct,
                "confidence": content.confidence,
            }
        except Exception as e:
            last_error = e
            if "length limit" in str(e).lower():
                return {"error": f"length limit: {e!s}"}
            time.sleep(1)
    return {"error": f"after 6 retries: {last_error!s}"}


# ---------- File handling aligned with judge.py: answer extraction, question normalization, answer_map, and load_all_questions ----------
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_print_lock = Lock()


def extract_answer_from_messages(messages: List[Dict]) -> Optional[str]:
    """Extract the final <answer></answer> content from the last message, consistent with judge.py."""
    if not messages:
        return None
    last_message = messages[-1]
    content = last_message.get("content", "")
    matches = _ANSWER_TAG_RE.findall(content)
    if matches:
        return matches[-1].strip()
    return None


def normalize_question(text: str) -> str:
    """Normalize question text for robust matching, consistent with judge.py."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text


def load_answer_map(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """Build a question -> {id, answer} mapping from the dataset JSONL, consistent with judge.py."""
    answer_map: Dict[str, Dict[str, str]] = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                question = normalize_question(data.get("question", ""))
                answer = data.get("answer", "")
                question_id = data.get("id", "")
                if question and question not in answer_map:
                    answer_map[question] = {"id": question_id, "answer": answer}
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in dataset {dataset_path}, line {line_num}: {e}")
                continue
    return answer_map


def load_all_questions(target_dir: str, verbose: bool = True) -> List[Dict]:
    """Load all JSONL files in a directory and return records with data/line/file_path, consistent with judge.py."""
    records: List[Dict] = []
    for jsonl_file in sorted(Path(target_dir).glob("*.jsonl")):
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
                        "file_path": str(jsonl_file),
                    })
                except json.JSONDecodeError as e:
                    if verbose:
                        with _print_lock:
                            print(f"Error parsing JSON in file {jsonl_file}, line {line_num}: {e}")
                    continue
    return records


def _is_correct_from_judge(judge_result: Optional[Dict]) -> bool:
    """Convert the official judge's correct field to bool; failures (including error) return False."""
    if not judge_result or "error" in judge_result:
        return False
    c = judge_result.get("correct", "")
    return str(c).lower() in ("y", "yes", "true", "positive")


def process_single_question(
    data: Dict,
    line_num: int,
    file_path: str,
    answer_map: Dict[str, Dict[str, str]],
    verbose: bool = True,
) -> Dict:
    """
    Process one record: read question/messages from data, match ground_truth through answer_map,
    judge it with evaluate_hle_official's extract_answer, and return a result structure aligned with judge.py.
    """
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

    messages = data.get("messages", [])
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
            "is_correct": False,
        }

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
            "is_correct": False,
        }

    # Judge using the model and prompt from evaluate_hle_official.
    judge_result = extract_answer(question, ground_truth, predicted_answer)
    is_correct = _is_correct_from_judge(judge_result)
    if judge_result and "error" not in judge_result:
        judgment = (
            f"{judge_result.get('reasoning', '')} [correct={judge_result.get('correct', '')}]"
        )
    else:
        err_msg = (judge_result or {}).get("error", "unknown")
        judgment = f"Judge call failed: {err_msg}"

    if verbose:
        with _print_lock:
            if judge_result and "error" in judge_result:
                print(f"  Judgment: failed - {judge_result['error']}")
            else:
                print(f"  Judgment: correct={judge_result.get('correct') if judge_result else 'N/A'}, Is correct: {is_correct}")

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


def main():
    parser = argparse.ArgumentParser(
        description="HLE evaluation: file handling follows judge.py, while judging uses evaluate_hle_official's model and prompt"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="",
        help="Directory containing the result files (.jsonl)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset JSONL path (must include question, answer, and id)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of worker threads",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Print verbose logs")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()
    target_dir = args.target_dir
    dataset_path = args.dataset
    workers = args.workers
    verbose = args.verbose and not args.quiet

    if not target_dir or not os.path.isdir(target_dir):
        raise SystemExit("--target-dir must point to an existing directory")
    if not dataset_path or not os.path.isfile(dataset_path):
        raise SystemExit("--dataset must point to an existing dataset JSONL file")

    jsonl_files = sorted(Path(target_dir).glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found under {target_dir}")

    print(f"\n{'='*80}")
    print("Configuration:")
    print(f"  Target directory: {target_dir}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Files to process: {len(jsonl_files)}")
    print(f"  Workers: {workers}")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"{'='*80}\n")

    print("Loading answer map from dataset...")
    answer_map = load_answer_map(dataset_path)
    if not answer_map:
        raise SystemExit(f"No entries were loaded from dataset: {dataset_path}")
    print(f"Loaded {len(answer_map)} questions from dataset\n")

    records = load_all_questions(target_dir, verbose)
    if not records:
        raise SystemExit("No processable records found")

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
        for future in tqdm(as_completed(future_to_rec), total=len(records), desc="Judging"):
            rec = future_to_rec[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                if verbose:
                    with _print_lock:
                        print(f"Error processing file {rec['file_path']}, line {rec['line']}: {e}")
                continue

    all_results.sort(key=lambda x: (x["file"], x["line"]))
    sorted_filenames = [f.name for f in jsonl_files]

    question_results = {}
    for r in all_results:
        q = r.get("question", "")
        f_name = r.get("file", "")
        is_correct = r.get("is_correct", False)
        if q not in question_results:
            question_results[q] = {}
        if f_name not in question_results[q]:
            question_results[q][f_name] = is_correct
        elif is_correct:
            question_results[q][f_name] = True

    total_questions = len(question_results)
    num_files = len(sorted_filenames)
    print(f"\n=== Multi-file statistics ({num_files} files, {total_questions} unique questions) ===")
    print(f"=== Pass@k / Best@k (a question passes if at least one of the first k files is correct) ===\n")
    best_at_k_stats = {}
    pass_at_k_stats = {}
    question_solved_status = {q: False for q in question_results}
    for k, filename in enumerate(sorted_filenames, 1):
        for q in question_results:
            if not question_solved_status[q] and question_results[q].get(filename, False):
                question_solved_status[q] = True
        correct_count = sum(1 for v in question_solved_status.values() if v)
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        print(f"  Pass@{k} (Best@{k}): {accuracy:.2%}  ({correct_count}/{total_questions})")
        key = f"best@{k}"
        best_at_k_stats[key] = {
            "correct": correct_count,
            "total": total_questions,
            "accuracy": accuracy,
        }
        pass_at_k_stats[f"pass@{k}"] = {
            "correct": correct_count,
            "total": total_questions,
            "accuracy": accuracy,
        }

    output_file = os.path.join(target_dir, "judgment_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

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

    summary = {
        "total_questions": total_questions,
        "num_files": num_files,
        "best_at_k": best_at_k_stats,
        "pass_at_k": pass_at_k_stats,
        "by_file": {},
        "judge_model": JUDGE_MODEL,
    }
    for file_name, q_map in per_file_question.items():
        total_q = len(q_map)
        correct_q = sum(1 for v in q_map.values() if v)
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
