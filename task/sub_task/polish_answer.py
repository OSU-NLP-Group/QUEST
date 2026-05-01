from tqdm import tqdm
import concurrent.futures
import threading
from litellm import completion
import os
import re
import json
from tqdm import trange
import copy
import argparse
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "longform_rubric", "prompt"))

from polish_prompt import POLISH_TEMPLATE

# model configuration
model="openai/gpt-5.2"


class AIClient():
    def generate(self,message):
        """Call OpenAI API"""
        response = completion(
            model=model,
            messages=message,
            max_tokens=40*1024
        )
        print("api call success")
        return response.choices[0].message.content
ai_client = AIClient()


parser = argparse.ArgumentParser()
parser.add_argument("--files_to_polish", type=str, default="datasets/longform_v4_2000/memory_logs/", help="Files to polish")
parser.add_argument("--iter", type=str, default=None, help="Iteration number filter")
parser.add_argument("--output_dir", type=str, default="results/deepresearch/extracted_questions_longform_v4_2000", help="Output directory")
parser.add_argument(
    "--max-workers", type=int, default=100, help="Maximum number of worker threads"
)
args = parser.parse_args()
ITER = args.iter
MAX_WORKERS = args.max_workers
FILES_TO_POLISH = args.files_to_polish
OUTPUT_DIR = args.output_dir

cnt = 0
cnt_lock = threading.Lock()
write_lock = threading.Lock()

def get_iter_to_item_ids():
    iter_to_item_ids = {}
    for entry in os.scandir(FILES_TO_POLISH):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        for iter_entry in os.scandir(entry.path):
            if not iter_entry.is_dir() or not re.fullmatch(r"iter\d+", iter_entry.name):
                continue
            if ITER is not None and iter_entry.name != f"iter{ITER}":
                continue
            file_path = os.path.join(iter_entry.path, "trajectories_no_memory.jsonl")
            if os.path.isfile(file_path):
                iter_to_item_ids.setdefault(iter_entry.name, []).append(int(entry.name))
    for iter_name in iter_to_item_ids:
        iter_to_item_ids[iter_name].sort()
    return dict(sorted(iter_to_item_ids.items(), key=lambda item: int(item[0][4:])))


def process_item(iter_name, item_id):
    global cnt
    file_path = os.path.join(
        FILES_TO_POLISH, str(item_id), iter_name, "trajectories_no_memory.jsonl"
    )
    output_file = os.path.join(OUTPUT_DIR, f"{iter_name}_replace.jsonl")
    try:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
            assert len(data) in [1, 2]
            # Prefer the second entry, fall back to the first if not available
            data = data[1] if len(data) > 1 else data[0]
            messages = data["messages"]
            # If messages[-1]["content"] does not contain <answer>, set "replace_status" to Fail due to no <answer> in the last message
            if "<answer>" not in messages[-1]["content"]:
                data["replace_status"] = "Fail due to no <answer> in the last message"
                data["replace_answer"] = ""
                with cnt_lock:
                    cnt += 1
            # If messages[-1]["content"] contains multiple <answer>, set "replace_status" to Fail due to multiple <answer> in the last message
            elif messages[-1]["content"].count("<answer>") > 1:
                data["replace_status"] = "Fail due to multiple <answer> in the last message"
                data["replace_answer"] = ""
                with cnt_lock:
                    cnt += 1
            else:
                messages_new=copy.deepcopy(messages)
                messages_new[-1]["content"] = messages_new[-1]["content"].split("<answer>")[0] + "Let's begin writing the final report with inline urls for every nontrivial claim in retrieved snippets.<answer>" 
                status = True
                for _ in range(3):
                    if not status:
                        break
                    try:
                        answer = ai_client.generate(messages_new)
                        if answer.strip() == "":
                            data["replace_status"] = "Fail due to empty answer generated"
                            data["replace_answer"] = ""
                        else:
                            data["replace_status"] = "Success"
                            data["replace_answer"] = answer.strip()
                            status = False
                    except Exception as e:
                        print(f"Error: {e}, retrying...")
        with write_lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error processing {iter_name} item {item_id}: {e}")

iter_to_item_ids = get_iter_to_item_ids()
total_items = sum(len(item_ids) for item_ids in iter_to_item_ids.values())
max_workers = min(MAX_WORKERS, total_items) if total_items > 0 else 1

print("########## total iters", len(iter_to_item_ids))
print("########## total files", total_items)
print("########## max_workers", max_workers)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(process_item, iter_name, item_id)
        for iter_name, item_ids in iter_to_item_ids.items()
        for item_id in item_ids
    ]
    for _ in tqdm(concurrent.futures.as_completed(futures), total=total_items):
        pass

print(cnt)
