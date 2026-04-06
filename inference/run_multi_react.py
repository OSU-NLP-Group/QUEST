import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
from react_agent import MultiTurnReactAgent, PYTHON_TOOL_ENABLED, SCHOLAR_TOOL_ENABLED
import time
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    args = parser.parse_args()

    model = args.model
    model_path = args.model_path
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    model_name = os.path.basename(model.rstrip('/'))

    model_dir = os.path.join(output_base, f"{model_name}")
    # Extract the dataset name from the dataset argument(remove the path and extension)
    dataset_name = os.path.basename(args.dataset)
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        dataset_name = os.path.splitext(dataset_name)[0]
    dataset_dir = os.path.join(model_dir, dataset_name)

    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    items = items[start_idx:end_idx]
    
    # max_samples = 2
    # items = items[:max_samples]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    if total_splits > 1:
        # Add split suffix to output files when using splits
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    processed_keys_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_filenames = set()
        processed_questions = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "error" in data:
                                continue
                            # Validate whether rollout_idx matches(if present)
                            file_rollout_idx = data.get("rollout_idx") or data.get("rollout_id")
                            if file_rollout_idx is not None and file_rollout_idx != rollout_idx:
                                print(f"Warning: Found mismatch rollout_idx in {output_file}: expected {rollout_idx}, got {file_rollout_idx}")
                                continue
                            # Use filename to track processed items
                            filename_val = data.get("filename", "")
                            if filename_val:
                                processed_filenames.add(filename_val.strip())
                            # Track processed questions(used for resume when filename is missing)
                            question_val = data.get("question", "")
                            if question_val:
                                processed_questions.add(question_val.strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_keys_per_rollout[rollout_idx] = {
            "filenames": processed_filenames,
            "questions": processed_questions,
        }

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_filenames = processed_keys_per_rollout[rollout_idx]["filenames"]
        processed_questions = processed_keys_per_rollout[rollout_idx]["questions"]
        for item_idx, item in enumerate(items):
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            # Get filename, Remove the .jsonl suffix(if any)
            filename = item.get("filename", "")
            if filename:
                # Remove the .jsonl suffix
                if filename.endswith(".jsonl"):
                    filename = filename[:-6]  # Remove the .jsonl
                elif filename.endswith(".json"):
                    filename = filename[:-5]  # Remove the .json
            
            # Resumelogic:use filename if available; otherwise use question
            if not filename:
                print(f"Warning: Item has no filename field, resume will use question: {question[:50]}...")
            
            if filename:
                if filename in processed_filenames:
                    # Already processed, skip
                    continue
            else:
                if question in processed_questions:
                    # Already processed, skip
                    continue

            # Tasks that still need processing
            # Get task_id if available; otherwise use the index
            task_id = item.get("task_id") or item.get("id") or f"idx_{start_idx + item_idx}"
            
            tasks_to_run_all.append({
                "item": item.copy(),
                "rollout_idx": rollout_idx,
                "task_id": task_id,
                "filename": filename,  # Add the filename field
            })
            per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        processed_filename_count = len(processed_keys_per_rollout[rollout_idx]["filenames"])
        processed_question_count = len(processed_keys_per_rollout[rollout_idx]["questions"])
        print(f"Rollout {rollout_idx}: already successfully processed (by filename: {processed_filename_count}, by question: {processed_question_count}), to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        function_list = ["search", "visit", "condenser"]
        if SCHOLAR_TOOL_ENABLED:
            function_list.append("google_scholar")
        if PYTHON_TOOL_ENABLED:
            function_list.append("PythonInterpreter")
        print(f"Function list: {function_list}")

        llm_cfg = {
            'model': model,
            'model_path': model_path,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            },
            'model_type': 'qwen_dashscope'
        }

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=function_list
        )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model
                ): task for task in tasks_to_run_all
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                # try:
                result = future.result()
                # Add rollout_idx to the result for resume validation
                result["rollout_idx"] = rollout_idx
                result["rollout_id"] = rollout_idx  # Keep compatibility
                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
