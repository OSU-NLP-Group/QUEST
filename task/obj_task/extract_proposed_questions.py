#!/usr/bin/env python3
"""Documentation omitted."""

import json
import os
from pathlib import Path

def extract_questions_from_directory(input_dir, output_file):
    """Documentation omitted."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: directory does not exist: {input_dir}")
        return
    
    if not input_path.is_dir():
        print(f"Error: path is not a directory: {input_dir}")
        return
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"Warning: no JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    extracted_count = 0
    skipped_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                answer_str = data.get("answer", "")
                
                if not answer_str:
                    print(f"Warning: {json_file.name} has no answer field, skipping")
                    skipped_count += 1
                    continue
                try:
                    answer_json = json.loads(answer_str)
                except json.JSONDecodeError as e:
                    print(f"Warning: answer field in {json_file.name} is not valid JSON, skipping: {e}")
                    skipped_count += 1
                    continue
                question = answer_json.get("proposed_question", "")
                
                if not question:
                    print(f"Warning: answer in {json_file.name} has no proposed_question field, skipping")
                    skipped_count += 1
                    continue
                filename = json_file.name
                output_data = {
                    "question": question,
                    "answer": "",
                    "filename": filename
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                extracted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error: failed to parse JSON file {json_file.name}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error: failed while processing {json_file.name}: {e}")
                skipped_count += 1
    
    print("\nDone!")
    print(f"Extracted: {extracted_count} files")
    print(f"Skipped/failed: {skipped_count} files")
    print(f"Output file: {output_file}")

def main():
    input_dir = "/fs/scratch/PAS1576/jianxie/DeepResearch/proposer_v1/inference/server/claude-sonnet-4.5/traj_v6_8k_RL/valid/formatted/verifier/accepted_trajectories"
    output_file = "/fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/inference_2/eval_data/extracted_questions_v5_8k_RL_20260321.jsonl"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_questions_from_directory(input_dir, output_file)

if __name__ == "__main__":
    main()
