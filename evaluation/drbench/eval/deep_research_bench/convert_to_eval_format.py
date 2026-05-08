#!/usr/bin/env python3
"""
Convert iter1-iter5.jsonl files into the claude-3-7-sonnet-latest.jsonl format.
The article field is taken from the prediction field.
"""

import argparse
import json
import os
from pathlib import Path


def load_questions(questions_file):
    """Load the question file and build a question-to-id mapping."""
    question_to_id = {}
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question_to_id[data['question']] = data['id']
    return question_to_id


def convert_iter_to_claude_format(iter_file, questions_file, output_file):
    """
    Convert a single iter file to the claude format.

    Args:
        iter_file: input iter file path
        questions_file: question file path
        output_file: output file path
    """
    # Load the question mapping.
    question_to_id = load_questions(questions_file)

    # Read the iter file and convert each record.
    converted_data = []
    with open(iter_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            question = data.get('question', '')
            prediction = data.get('prediction', '')

            # Look up the id from the question mapping.
            question_id = question_to_id.get(question)

            if question_id is None:
                print(f"Warning: question ID not found: {question[:50]}...")
                continue

            # Build the converted record.
            converted = {
                "id": question_id,
                "prompt": question,
                "article": prediction
            }

            converted_data.append(converted)

    # Write the converted output.
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Conversion complete: {iter_file} -> {output_file}")
    print(f"Converted {len(converted_data)} records in total")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert iter*.jsonl inference outputs into DeepResearch Bench raw_data format."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing iter{N}.jsonl files. Can be passed multiple times for multiple runs.",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        required=True,
        help="Path to the deepresearch_bench_questions.jsonl file (provides question -> id mapping).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test_data/raw_data"),
        help="Output directory for converted jsonl files (default: data/test_data/raw_data).",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        action="append",
        default=None,
        help="Identifier prefix for output filenames. If passed multiple times, must match the number of --base-dir. "
             "If omitted, defaults to the parent directory name of each base-dir.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which iter indices to convert (default: 1 2 3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dirs = args.base_dir
    questions_file = args.questions_file
    output_dir = args.output_dir

    if args.identifier is not None:
        if len(args.identifier) != len(base_dirs):
            raise ValueError(
                f"--identifier was passed {len(args.identifier)} times but --base-dir was passed "
                f"{len(base_dirs)} times; counts must match."
            )
        identifiers = args.identifier
    else:
        identifiers = [bd.parent.name or "unknown" for bd in base_dirs]

    # Ensure the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each base_dir in order.
    for base_dir, identifier in zip(base_dirs, identifiers):
        print(f"\n{'='*80}")
        print(f"Processing directory: {base_dir}")
        print(f"Identifier: {identifier}")
        print(f"{'='*80}")

        if not base_dir.exists():
            print(f"Warning: directory does not exist, skipping: {base_dir}")
            continue

        for i in args.iters:
            iter_file = base_dir / f"iter{i}.jsonl"
            output_file = output_dir / f"{identifier}-iter{i}.jsonl"

            if iter_file.exists():
                print(f"\nProcessing {iter_file.name}...")
                convert_iter_to_claude_format(iter_file, questions_file, output_file)
            else:
                print(f"Warning: file does not exist - {iter_file}")


if __name__ == "__main__":
    main()
