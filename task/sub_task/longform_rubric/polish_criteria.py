from polish_prompt import POLISH_TEMPLATE
from tqdm import tqdm
import concurrent.futures
import threading
import argparse
from litellm import completion
import os
import re
import json

# Set API key environment variables
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID"]
os.environ["AWS_REGION_NAME"] = os.environ["AWS_REGION_NAME"]

# Model configuration
model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_NUM_THREADS = 50
write_lock = threading.Lock()

class AIClient():
    def generate(self,user_prompt, system_prompt=""):
        """Call Azure OpenAI API"""
        response = completion(
            model=model,
            messages=[{ "content": user_prompt,"role": "user"}],
            reasoning_effort="high",
            max_tokens=30*1024
        )
        # print(response)
        print(1)
        # if hasattr(response.choices[0].message, 'reasoning_content'):
        #     return response.choices[0].message.content, response.choices[0].message.reasoning_content
        return response.choices[0].message.content, response
ai_client = AIClient()

def parse_json(text):
    """Extract JSON code block content from text"""
    json_pattern = r"```json(.*?)```"
    match = re.search(json_pattern, text, re.DOTALL)
    try: 
        if match:
            json_content = match.group(1).strip()
            return eval(json_content)
        return eval(text)
    except:
        return text
    return text
    


def get_polish_prompt(task_description: str, criteria_list: str) -> str:
    return POLISH_TEMPLATE.format(
        task_description=task_description,
        criteria_list=criteria_list
    )

def process_item(item, output_file):
    task_description = item["prompt"]
    criteria_list = item["criterions"]
    criteria_list_cleaned = {}
    for dimension, criteria in criteria_list.items():
        criteria_list_cleaned[dimension] = []
        for criterion in criteria:
            criterion_cleaned = {
                "criterion": criterion["criterion"],
                "explanation": criterion["explanation"]
            }
            criteria_list_cleaned[dimension].append(criterion_cleaned)
    for retries in range(3):
        prompt = get_polish_prompt(task_description, criteria_list_cleaned)
        result, reasoning = ai_client.generate(prompt)
        item["response"] = result
        polished_criteria = parse_json(result)
        item["polished_criterions"] = polished_criteria
        if isinstance(polished_criteria, dict):
            break
        print("[retry] due to invalid json")
    with write_lock:
        with open(output_file, "a") as f_out:
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write("\n")
    return item


def main():
    parser = argparse.ArgumentParser(description="Polish evaluation criteria")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS, help="Number of threads")
    args = parser.parse_args()

    with open(args.input_file, "r") as f_in:
        lines = f_in.readlines()
        items = [eval(line) for line in lines]

    polished_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_item, item, args.output_file) for item in items]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            polished_items.append(future.result())

    return polished_items


if __name__ == "__main__":
    main()
