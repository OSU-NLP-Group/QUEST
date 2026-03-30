#!/usr/bin/env python3
"""
Convert iter1-iter5.jsonl files into the claude-3-7-sonnet-latest.jsonl format.
The article field is taken from the prediction field.
"""

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


def main():
    # File path configuration.
    # Define multiple base_dir entries. Add more directories as needed.
    base_dirs = [
        Path("/fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/datasets/drbench/a3b-results/qwen3-moe-rl-45steps-16k-output-80k-memory-200turns/results/deepresearch/deepresearch_bench_questions"),
    ]
    questions_file = "/fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/datasets/drbench/deepresearch_bench_questions.jsonl"
    output_dir = Path("/fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/datasets/drbench/eval/deep_research_bench/data/test_data/raw_data")

    # Ensure the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each base_dir in order.
    for base_dir in base_dirs:
        # Extract a directory identifier from the path (for example: vanilla or search-only).
        # Assume the path format is .../drbench/{identifier}/results/...
        path_parts = base_dir.parts
        try:
            # Use the directory name after 'a3b-results' as the identifier.
            a3b_idx = path_parts.index('a3b-results')
            if a3b_idx + 1 < len(path_parts):
                identifier = path_parts[a3b_idx + 1]
            else:
                identifier = base_dir.name
        except (ValueError, IndexError):
            # Fall back to the parent directory name if not found.
            identifier = base_dir.parent.name if base_dir.parent.name else "unknown"
        
        print(f"\n{'='*80}")
        print(f"Processing directory: {base_dir}")
        print(f"Identifier: {identifier}")
        print(f"{'='*80}")
        
        if not base_dir.exists():
            print(f"Warning: directory does not exist, skipping: {base_dir}")
            continue

        # Convert iter1 through iter5.
        for i in range(1, 4):
            iter_file = base_dir / f"iter{i}.jsonl"
            # Include the identifier in the output filename to distinguish different sources.
            output_file = output_dir / f"{identifier}-iter{i}.jsonl"

            if iter_file.exists():
                print(f"\nProcessing {iter_file.name}...")
                convert_iter_to_claude_format(iter_file, questions_file, output_file)
            else:
                print(f"Warning: file does not exist - {iter_file}")


if __name__ == "__main__":
    main()
