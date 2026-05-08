#!/usr/bin/env python3
"""
Grade DeepResearch BrowseComp rollout JSONLs against the official BrowseComp test set.

Uses litellm for grading, reads all .jsonl files from --target-dir,
outputs judgment_results.json and judgment_summary.json to the same directory.

Usage:
  export JUDGE_MODEL_NAME=...      # e.g. "gpt-4.1-2025-04-14" or "azure/..." or "bedrock/..."
  export JUDGE_OPENAI_API_KEY=...  # or JUDGE_AZURE_API_KEY / JUDGE_AWS_* for other providers
  python eval.py --target-dir /path/to/results --dataset /path/to/browsecomp.csv
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.request import Request, urlopen

from litellm import completion

DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

GRADER_TEMPLATE = """
Here is a question and sample answer:
<question>
{question}
</question>
<sample_answer>
{answer}
</sample_answer>
Here is the actual, ground-truth, correct answer:
<answer_key>
{ground_truth_answer}
</answer_key>
Now consider the following statements about the sample answer:
(A) The response given in the sample answer matches the ground truth answer.
(B) The response given in the sample answer does not match the ground truth answer.
(C) The response given in the sample answer just says something like "I'm not sure" or
"I don't know"

Which statement do you think is more likely: (A), (B), or (C)?
""".strip()

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_EXACT_ANSWER_RE = re.compile(r"^Exact Answer:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

_print_lock = Lock()


# ---------------------------------------------------------------------------
# BrowseComp CSV decryption helpers
# ---------------------------------------------------------------------------

def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _read_remote_csv(url: str) -> Iterator[Dict[str, str]]:
    req = Request(url, headers={"User-Agent": "DeepResearch browsecomp eval.py"})
    with urlopen(req) as resp:
        data = resp.read()
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        yield {k: (v if v is not None else "") for k, v in row.items()}


def _read_local_csv(path: str) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v if v is not None else "") for k, v in row.items()}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """Extract the answer from <answer> tags, 'Exact Answer:' line, or return raw text."""
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _EXACT_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_answer_from_messages(messages: List[Dict]) -> Optional[str]:
    """Extract the last <answer> tag from the last message in the conversation."""
    if not messages:
        return None
    last_message = messages[-1]
    content = last_message.get("content", "")
    matches = _ANSWER_TAG_RE.findall(content)
    if matches:
        return matches[-1].strip()
    return None


def normalize_question(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text


# ---------------------------------------------------------------------------
# Gold answer loading
# ---------------------------------------------------------------------------

def load_gold_qa(dataset_path: str) -> Dict[str, str]:
    """
    Load gold question-answer pairs.
    If dataset_path is a URL or the default remote CSV, download and decrypt.
    If it's a local CSV, read and decrypt.
    """
    if dataset_path.startswith("http://") or dataset_path.startswith("https://"):
        rows = _read_remote_csv(dataset_path)
    else:
        rows = _read_local_csv(dataset_path)

    qa: Dict[str, str] = {}
    for row in rows:
        canary = row.get("canary", "")
        if canary:
            problem = decrypt(row.get("problem", ""), canary).strip()
            answer = decrypt(row.get("answer", ""), canary).strip()
        else:
            problem = (row.get("problem", "") or row.get("question", "")).strip()
            answer = (row.get("answer", "")).strip()
        if problem:
            qa[problem] = answer
    return qa


# ---------------------------------------------------------------------------
# Grading via litellm
# ---------------------------------------------------------------------------

def call_litellm_for_judgment(question: str, gold_answer: str, predicted_answer: str) -> str:
    """Call litellm to judge whether the predicted answer matches the gold answer."""
    model_name = os.environ.get("JUDGE_MODEL_NAME", "")
    if not model_name:
        raise ValueError("JUDGE_MODEL_NAME environment variable must be set")

    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        answer=predicted_answer,
        ground_truth_answer=gold_answer,
    )

    call_kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": grader_prompt}],
        "temperature": 1,
        "max_tokens": 2048,
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
        for env_key, kwarg_key in [
            ("JUDGE_AWS_ACCESS_KEY_ID", "aws_access_key_id"),
            ("JUDGE_AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
            ("JUDGE_AWS_REGION_NAME", "aws_region_name"),
        ]:
            val = os.environ.get(env_key)
            if val:
                call_kwargs[kwarg_key] = val
    else:
        api_key = os.environ.get("JUDGE_OPENAI_API_KEY")
        if api_key:
            call_kwargs["api_key"] = api_key

    try:
        response = completion(**call_kwargs)
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"Error calling litellm: {e}")
        return ""


def _extract_statement_choice(text: str) -> Optional[str]:
    """Parse judge reply for (A), (B), or (C) after ABC-style grading prompt."""
    if not text or not text.strip():
        return None
    tail = "\n".join(text.strip().splitlines()[-20:])

    for pat in (
        r"(?:more likely|most likely)[^\n.:]{0,80}?(?:\(|\s)([ABC])\)",
        r"(?:choose|prefer|would (?:say|pick|select)|answer is)[^\n.:]{0,40}?\(?([ABC])\)?",
        r"(?:statement|option)[^\n.:]{0,20}?\(?([ABC])\)?",
        r"\(([ABC])\)",
    ):
        matches = list(re.finditer(pat, tail, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    last_line = tail.splitlines()[-1] if tail else ""
    lone = re.search(r"\b([ABC])\b(?![\da-z])\s*\.?$", last_line.strip(), re.IGNORECASE)
    if lone:
        return lone.group(1).upper()
    return None


def parse_verdict(grader_output: str) -> Optional[str]:
    """Extract a yes/no verdict from ABC-style or legacy judge output."""
    choice = _extract_statement_choice(grader_output)
    if choice in ("A", "B", "C"):
        return "yes" if choice == "A" else "no"

    m = re.search(r"correct:\s*(yes|no)\b", grader_output, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

def process_single_question(
    data: Dict,
    line_num: int,
    file_path: str,
    gold_qa: Dict[str, str],
    verbose: bool = True,
) -> Dict:
    """Process a single question and return the judgment result."""
    question = (data.get("question") or "").strip()

    # Try to get prediction: either from "prediction" field or from messages
    prediction = data.get("prediction")
    if prediction is None:
        messages = data.get("messages", [])
        prediction = extract_answer_from_messages(messages)

    if prediction is None:
        prediction = ""

    # Match question to gold
    gold_answer = gold_qa.get(question, "")
    if not gold_answer:
        # Try normalized matching
        norm_q = normalize_question(question)
        for gq, ga in gold_qa.items():
            if normalize_question(gq) == norm_q:
                gold_answer = ga
                break

    if verbose:
        with _print_lock:
            print(f"[Line {line_num}] Processing: {question[:80]}...")
            print(f"  Gold answer: {gold_answer[:80]}..." if gold_answer else "  Gold answer: (not found)")

    if not gold_answer:
        if verbose:
            with _print_lock:
                print(f"  Warning: No gold answer found for line {line_num}")
        return {
            "file": os.path.basename(file_path),
            "line": line_num,
            "question": question,
            "ground_truth": "",
            "predicted_answer": prediction,
            "extracted_answer": extract_final_answer(prediction) if prediction else "",
            "grader_output": "gold answer not found",
            "verdict": None,
            "is_correct": False,
        }

    extracted = extract_final_answer(prediction) if prediction else ""

    if not extracted:
        if verbose:
            with _print_lock:
                print(f"  Warning: Empty prediction for line {line_num}")
        return {
            "file": os.path.basename(file_path),
            "line": line_num,
            "question": question,
            "ground_truth": gold_answer,
            "predicted_answer": prediction,
            "extracted_answer": "",
            "grader_output": "empty prediction",
            "verdict": "no",
            "is_correct": False,
        }

    # Call grader
    grader_output = call_litellm_for_judgment(question, gold_answer, extracted)
    verdict = parse_verdict(grader_output)
    is_correct = verdict == "yes"

    if verbose:
        with _print_lock:
            print(f"  Verdict: {verdict}, Is correct: {is_correct}")

    return {
        "file": os.path.basename(file_path),
        "line": line_num,
        "question": question,
        "ground_truth": gold_answer,
        "predicted_answer": prediction,
        "extracted_answer": extracted,
        "grader_output": grader_output,
        "verdict": verdict,
        "is_correct": is_correct,
    }


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_all_questions(target_dir: str, verbose: bool = True) -> List[Dict]:
    """Load all questions from .jsonl files in the target directory."""
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
                        "file_path": str(jsonl_file),
                    })
                except json.JSONDecodeError as e:
                    if verbose:
                        with _print_lock:
                            print(f"Error parsing JSON in {jsonl_file}, line {line_num}: {e}")
                    continue
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BrowseComp evaluation script: use an LLM to judge whether answers are correct (multithreaded)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory containing the result files (.jsonl)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_URL,
        help="BrowseComp dataset path (local CSV or remote URL); defaults to the official remote CSV",
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

    # Load gold answers
    print("Loading gold answers from dataset...")
    gold_qa = load_gold_qa(dataset_path)
    if not gold_qa:
        print(f"Error: No gold answers loaded from dataset: {dataset_path}")
        return
    print(f"Loaded {len(gold_qa)} gold questions from dataset\n")

    # Load all questions from jsonl files
    records = load_all_questions(target_dir, verbose)
    if not records:
        print("No records found to process")
        return
    print(f"\nProcessing {len(records)} questions with {workers} workers...\n")

    # Grade all questions in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_rec = {
            executor.submit(
                process_single_question,
                rec["data"],
                rec["line"],
                rec["file_path"],
                gold_qa,
                verbose,
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
                        print(f"Error processing {rec['file_path']}, line {rec['line']}: {e}")
                continue

    # Sort results by file name and line number
    all_results.sort(key=lambda x: (x["file"], x["line"]))

    sorted_filenames = [f.name for f in jsonl_files]

    # ---- Best@k statistics ----
    question_results: Dict[str, Dict[str, bool]] = {}
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

    print(f"\n=== Best@k Statistics (Total questions: {total_questions}) ===")

    best_at_k_stats = {}
    question_solved_status = {q: False for q in question_results}

    for k, filename in enumerate(sorted_filenames, 1):
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
            "accuracy": accuracy,
        }

    # ---- Save detailed results ----
    output_file = os.path.join(target_dir, "judgment_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    # ---- Save summary ----
    per_file_question: Dict[str, Dict[str, bool]] = {}
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
        "best_at_k": best_at_k_stats,
        "by_file": {},
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
