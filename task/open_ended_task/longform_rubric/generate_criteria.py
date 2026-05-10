import json
import os
import re
import time
import argparse
from tqdm import tqdm
import concurrent.futures
import threading
from litellm import completion

# OPENAI
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
# model="openai/gpt-5" 

# AZURE
os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"
model="azure/gpt-5" 


# Import dimension weight generation prompts for English
from prompt.criteria_prompt_en import (
    generate_eval_dimension_weight_prompt,
    generate_eval_criteria_prompt_comp,
    generate_eval_criteria_prompt_insight,
    generate_eval_criteria_prompt_Inst,
    generate_eval_criteria_prompt_readability
)


# Processing parameters
RETRY_ATTEMPTS = 5
RETRY_DELAY = 5
PROCESS_LIMIT = 500
MAX_WORKERS = 10
DEFAULT_SAMPLE_COUNT = 3

# Thread-safe locks
print_lock = threading.Lock()
counter_lock = threading.Lock()

# Initialize AI client
class AIClient():
    def generate(self,user_prompt, system_prompt=""):
        """Call Azure OpenAI API"""
        response = completion(
            model=model,
            messages=[{ "content": user_prompt,"role": "user"}],
            reasoning_effort="low",
            max_tokens=5120
        )
        # print(response)
        print(1)
        return response.choices[0].message.content
ai_client = AIClient()

def parse_llm_output_as_json(text: str, expected_type: type = list) -> dict or list or None:
    """Parse JSON output from text"""
    # Try to extract JSON from special markers
    match = re.search(r'<json_output>(.*?)</json_output>', text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # Handle possible code block format
        json_str = text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
    
    # Parse JSON
    parsed_data = json.loads(json_str)
    
    # Type checking
    if not isinstance(parsed_data, expected_type) or (isinstance(parsed_data, (list, dict)) and not parsed_data):
        return None
    
    return parsed_data

def validate_weights(data, expected_sum=1.0, tolerance=1e-6):
    """Validate if the sum of weights is close to the expected value"""
    if isinstance(data, dict):
        # Dictionary form of weights
        if not data:
            return False
        total_weight = sum(float(value) for value in data.values())
        return abs(total_weight - expected_sum) < tolerance
        
    elif isinstance(data, list):
        # List form of weights
        if not data or not all(isinstance(item, dict) and 'weight' in item for item in data):
            return False
        total_weight = sum(float(item['weight']) for item in data)
        return abs(total_weight - expected_sum) < tolerance
        
    return False

def round_weights_and_adjust(weights, decimal_places=2):
    """
    Round weights to specified decimal places and ensure the sum is 1
    """
    # Round to specified decimal places
    rounded_weights = {dim: round(float(weight), decimal_places) 
                     for dim, weight in weights.items()}
    
    # Calculate sum and check if it equals 1
    total = sum(rounded_weights.values())
    diff = 1.0 - total
    
    # If there's a difference, add it to readability
    if abs(diff) > 1e-10:
        rounded_weights["readability"] = round(rounded_weights["readability"] + diff, decimal_places)
    
    return rounded_weights

def get_prompts_by_language(language):
    """Select prompt templates based on language"""
    return {
        "weight_prompt": generate_eval_dimension_weight_prompt, 
        "criteria_prompts": {
            "comprehensiveness": generate_eval_criteria_prompt_comp,
            "insight": generate_eval_criteria_prompt_insight,
            "instruction_following": generate_eval_criteria_prompt_Inst,
            "readability": generate_eval_criteria_prompt_readability,
        }
    }

def generate_weights_multiple_times(item_id, prompt, language, sample_count=DEFAULT_SAMPLE_COUNT):
    """Generate dimension weights multiple times for a single prompt and take the average"""
    weights_samples = []
    
    # Get weight prompt template for the corresponding language
    prompts = get_prompts_by_language(language)
    weight_prompt_template = prompts["weight_prompt"]
    
    user_prompt = weight_prompt_template.format(task_prompt=prompt)
    
    # Multiple sampling
    for _ in range(sample_count):
        for attempt in range(RETRY_ATTEMPTS):
            weights_output = ai_client.generate(user_prompt=user_prompt, system_prompt="")
            
            try:
                parsed_weights = parse_llm_output_as_json(weights_output, expected_type=dict)
                if parsed_weights and validate_weights(parsed_weights):
                    weights_samples.append(parsed_weights)
                    break
            except Exception:
                if attempt == RETRY_ATTEMPTS - 1:
                    raise  # Raise exception on the last attempt
            
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    
    # Return failure if no successful sampling
    if not weights_samples:
        return None
    
    # Calculate average weights
    dimensions = set()
    for sample in weights_samples:
        dimensions.update(sample.keys())
    
    avg_weights = {}
    for dim in dimensions:
        # Only calculate average for dimensions that appear in all samples
        values = [sample.get(dim, 0) for sample in weights_samples if dim in sample]
        if len(values) == len(weights_samples):  # Ensure dimension appears in all samples
            avg_weights[dim] = sum(values) / len(values)
    
    # Normalize weights to ensure sum is 1
    weight_sum = sum(avg_weights.values())
    for dim in avg_weights:
        avg_weights[dim] = avg_weights[dim] / weight_sum
    
    # Round to two decimal places and ensure sum is 1
    rounded_weights = round_weights_and_adjust(avg_weights, decimal_places=2)
    
    return rounded_weights

def process_single_item_sequential(item):
    """Process a single data item, executing two phases: generating weights and generating criteria"""
    item_id = item.get('id')
    prompt = item.get('prompt')
    language = item.get('language', 'en')  
    
    # === Phase 1: Multiple sampling to generate dimension weights and take average ===
    current_weights = generate_weights_multiple_times(
        item_id, prompt, language, DEFAULT_SAMPLE_COUNT
    )
    
    # === Phase 2: Generate detailed criteria ===
    prompts = get_prompts_by_language(language)
    criteria_prompts = prompts["criteria_prompts"]
    current_criterions = {}

    for dim_name, criteria_prompt_template in criteria_prompts.items():
        user_prompt_criteria = criteria_prompt_template.format(task_prompt=prompt)
        
        for attempt in range(RETRY_ATTEMPTS):
            criteria_output = ai_client.generate(user_prompt=user_prompt_criteria, system_prompt="")
            
            try:
                parsed_criteria = parse_llm_output_as_json(criteria_output, expected_type=list)
                if parsed_criteria and validate_weights(parsed_criteria):
                    current_criterions[dim_name] = parsed_criteria
                    break
            except Exception as e:
                print(f"Error processing item {item_id}: {e}")
                if attempt == RETRY_ATTEMPTS - 1:
                    raise  # Raise exception on the last attempt
            
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)

    # === Successfully completed all phases ===
    final_data = {
        "id": item_id,
        "prompt": prompt,
        "dimension_weight": current_weights,
        "criterions": current_criterions
    }
    
    return final_data, item_id

def generate_criteria_pipeline(input_file, output_file, process_limit, max_workers, sample_count=DEFAULT_SAMPLE_COUNT):
    """Main process: read data, process, write to final file"""
    start_time = time.time()
    success_count_this_run = 0
    error_count_this_run = 0
    results_this_run = []

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read input data
    input_data_to_process = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if process_limit is not None and len(input_data_to_process) >= process_limit:
                print("skip")
                break
            data = json.loads(line)
            if data.get('id') and data.get('prompt'):
                # Also include language field in processing items
                language = data.get('language', 'en')  # Default to English
                input_data_to_process.append({
                    'id': data['id'], 
                    'prompt': data['prompt'],
                    'language': language
                })

    if not input_data_to_process:
        print("No items to process.")
        return

    print(f"Starting to process {len(input_data_to_process)} items using up to {max_workers} threads...")
    print(f"Each weight will be sampled {sample_count} times and averaged...")
    
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in input_data_to_process:
            futures.append(executor.submit(process_single_item_sequential, item))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating criteria"):
            try:
                result_data, item_id = future.result()
                success_count_this_run += 1
                results_this_run.append(result_data)
            except Exception as exc:
                print(f'Failed to process item: {exc}')
                error_count_this_run += 1

    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        results_this_run.sort(key=lambda x: x.get('id', ''))
        for result in results_this_run:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    end_time = time.time()
    duration = end_time - start_time
    items_attempted = len(input_data_to_process)
    items_per_sec = items_attempted / duration if duration > 0 else 0

    print("\n--- Run Summary ---")
    print(f"Total items attempted:      {items_attempted}")
    print(f"Items successfully processed:  {success_count_this_run}")
    print(f"Items failed:        {error_count_this_run}")
    print(f"Total runtime:        {duration:.2f} seconds ({items_per_sec:.2f} items/sec)")
    print(f"Maximum threads used:      {max_workers}")
    print(f"Samples per weight:    {sample_count}")
    print(f"Final result file:          {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation criteria for longform tasks")
    parser.add_argument("--input_file", type=str, default="./outputs/extracted_questions.jsonl", help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, default="./outputs/criteria.jsonl", help="Output JSONL file path")
    args = parser.parse_args()
    
    generate_criteria_pipeline(
        args.input_file, 
        args.output_file, 
        PROCESS_LIMIT, 
        MAX_WORKERS,
        DEFAULT_SAMPLE_COUNT
    )
