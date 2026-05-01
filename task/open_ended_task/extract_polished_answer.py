#!/usr/bin/env python3
"""
Extract answers from iter*_replace.jsonl files (output by polish_answer.py),
output format same as extracted_answers_all_iters.jsonl: each line {"question", "answer", "iter"}.
The output answer must contain <answer>...</answer> tags (auto-wrap if missing).
Prefer replace_answer (when replace_status == "Success" and non-empty),
otherwise extract <answer> content or full content from the last assistant message in messages.
At the end, prints count for each iteration and total: replace_answer vs from_messages (only counting successfully written rows).
"""
import json
import re
import argparse
from pathlib import Path

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_HAS_ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)


def ensure_answer_tags(text: str) -> str:
    """Ensure text is wrapped in <answer>...</answer>; return as-is if already present."""
    if not (text or "").strip():
        return text
    s = text.strip()
    if _HAS_ANSWER_TAG_RE.search(s):
        return s
    return "<answer>\n" + s + "\n</answer>"


def extract_answer_from_messages(messages: list) -> str | None:
    """Extract <answer> content or full content from the last assistant message in messages (excluding tool_call)."""
    if not messages:
        return None
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").strip()
        if not content or content.lstrip().startswith("<tool_call"):
            continue
        # Prefer <answer>...</answer> content
        mat = _ANSWER_TAG_RE.findall(content)
        if mat:
            return mat[-1].strip()
        return content
    return None


def get_answer_and_source(data: dict) -> tuple[str | None, str | None]:
    """Get (answer, source) from a row of data; source is 'replace' or 'messages', returns (None, None) if no answer."""
    if data.get("replace_status") == "Success":
        ra = (data.get("replace_answer") or "").strip()
        if ra and not ra.lstrip().startswith("<tool_call"):
            return ra, "replace"
    ans = extract_answer_from_messages(data.get("messages") or [])
    if ans is not None:
        return ans, "messages"
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Extract polished answers from replace JSONL files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/deepresearch/extracted_questions_longform_v4_2000",
        help="Directory containing iter*_replace.jsonl files (output by polish_answer.py)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Single input file (if specified, overrides output_dir and iter detection)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL file path (default: <output_dir>/extracted_answers_all_iters.jsonl)",
    )
    args = parser.parse_args()

    if args.input_file:
        input_files = [Path(args.input_file)]
        output_path = Path(args.output_file) if args.output_file else Path(__file__).resolve().parent / "extracted_answers_all_iters.jsonl"
        _process_files(input_files, output_path, iter_num=None)
    else:
        output_dir = Path(args.output_dir)
        output_path = Path(args.output_file) if args.output_file else output_dir / "extracted_answers_all_iters.jsonl"
        iter_files = sorted(output_dir.glob("iter*_replace.jsonl"), key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1)) if re.search(r"iter(\d+)", p.name) else 0)
        _process_files(iter_files, output_path, iter_num=None)


def _process_files(input_files: list, output_path: Path, iter_num: int | None):
    total_written = 0
    total_replace = 0
    total_messages = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for p in input_files:
            if not p.exists():
                print(f"[WARN] Not found: {p}, skip.")
                continue
            file_iter_num = int(re.search(r"iter(\d+)", p.name).group(1)) if re.search(r"iter(\d+)", p.name) else None
            written = 0
            skipped = 0
            n_replace = 0
            n_messages = 0
            with open(p, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    question = (data.get("question") or "").strip()
                    answer, src = get_answer_and_source(data)
                    if not question or not (answer or "").strip():
                        skipped += 1
                        continue
                    if src == "replace":
                        n_replace += 1
                    elif src == "messages":
                        n_messages += 1
                    answer = ensure_answer_tags(answer.strip())
                    fout.write(
                        json.dumps(
                            {"question": question, "answer": answer, "iter": file_iter_num},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
            total_replace += n_replace
            total_messages += n_messages
            print(
                f"{p.name}: written={written}, skipped={skipped}, "
                f"replace_answer={n_replace}, from_messages={n_messages}"
            )
            total_written += written

    print(f"Total: {total_written} lines -> {output_path}")
    print(
        f"Source statistics (only counting successfully written rows): replace_answer={total_replace}, "
        f"from_messages={total_messages}"
    )


if __name__ == "__main__":
    main()