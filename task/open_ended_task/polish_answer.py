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
parser.add_argument("--files_to_polish", type=str, required=True, help="Directory containing ref_gen iter*.jsonl files")
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

def get_ref_gen_items():
    items = []
    for entry in os.scandir(FILES_TO_POLISH):
        match = re.fullmatch(r"(iter\d+)(?:_split\d+of\d+)?\.jsonl", entry.name)
        if not entry.is_file() or not match:
            continue
        iter_name = match.group(1)
        if ITER is not None and iter_name != f"iter{ITER}":
            continue
        with open(entry.path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    items.append((iter_name, f"{entry.name}:{line_number}", json.loads(line)))
    return items


def process_item(iter_name, item_id, data):
    global cnt
    output_file = os.path.join(OUTPUT_DIR, f"{iter_name}_replace.jsonl")
    try:
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

ref_gen_items = get_ref_gen_items()
total_items = len(ref_gen_items)
max_workers = min(MAX_WORKERS, total_items) if total_items > 0 else 1

print("########## total iters", len({item[0] for item in ref_gen_items}))
print("########## total files", total_items)
print("########## max_workers", max_workers)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(process_item, *item)
        for item in ref_gen_items
    ]
    for _ in tqdm(concurrent.futures.as_completed(futures), total=total_items):
        pass

print(cnt)
