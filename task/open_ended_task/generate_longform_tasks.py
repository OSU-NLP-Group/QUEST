from generation_agent_longform import MultiTurnReactAgent
import os
import random
import csv
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Read model name from environment variable, use default value if not set
# Use LiteLLM format model name, supports: openai/, azure/, bedrock/, vllm/
model = os.environ.get("DEEPRESEARCH_MODEL_NAME", "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Automatically determine model_type based on model name prefix (for backward compatibility)
if model.startswith("bedrock/"):
    model_type = 'bedrock'
elif model.startswith("azure/"):
    model_type = 'azure'
elif model.startswith("openai/"):
    model_type = 'openai'
elif model.startswith("vllm/"):
    model_type = 'vllm'
else:
    # If model name does not match known format, default to 'openai' (for backward compatibility)
    # But it is recommended to use the correct prefix format, otherwise errors may occur
    model_type = 'openai'
    print(f"Warning: Model name '{model}' does not start with a known prefix (openai/, azure/, bedrock/, vllm/). "
          f"Defaulting to 'openai' type. Please use the correct format.")

# Set different parameter configurations based on model type
# OpenAI models (such as gpt-5) do not support top_p parameter
if model_type == 'openai':
    generate_cfg = {
        'max_tokens': 20000,
        'max_retries': 10,
        'temperature': 1,
        # OpenAI models do not support top_p, so it is not included
        # 'presence_penalty': 1.1  # Some OpenAI models may support this
    }
else:
    # Bedrock, Azure, vLLM and other platforms support top_p
    generate_cfg = {
        'max_tokens': 20000,
        'max_retries': 10,
        'temperature': 1,
        'top_p': 0.95,
        # 'presence_penalty': 1.1  # Some platforms may not support this parameter
    }

llm_cfg = {
    'model': model,
    'generate_cfg': generate_cfg,
    'model_type': model_type
}

# Thread pool executor, will be created in main function based on workers count
executor = None

# Define category structure
category_structure = {
    "Lifestyle & Leisure": {
        "Shopping": [],
        "Food & Cooking": [],
        "Sports & Fitness": [],
        "Health & Medicine": [],
        "Pets & Animal Welfare": [],
        "Fashion & Beauty": [],
        "Hobbies & DIY": []
    },
    "Entertainment": {
        "Films & TV Shows": [],
        "Gaming & Virtual Worlds": [],
        "Live Shows & Performances": [],
        "Music": [],
        "Books & Reading": []
    },
    "Misc.": {
        "General Info.": [],
        "News": [],
        "Legal & Government Services": [],
        "Real Estate": [],
        "Finance & Investment": []
    },
    "Science & Research": {
        "Research & Academia": [],
        "Technology & Science": []
    },
    "Career & Education": {
        "Education & Learning": [],
        "Jobs & Career": []
    },
    "Travel & Transportation": {
        "Travel & Accommodation": [],
        "Outdoor & Recreation": [],
        "Ticketed Activities": []
    }
}

def initialize_subcategory_counts(category_structure):
    counts = {}
    for main_category, subcategories in category_structure.items():
        for subcategory in subcategories.keys():
            counts[(main_category, subcategory)] = 0
    return counts

def sample_subcategory_with_weights(category_structure, subcategory_counts, lock):
    with lock:
        # Get all main categories
        main_categories = list(category_structure.keys())
        
        # Randomly select a main category (each main category has equal probability)
        selected_main_category = random.choice(main_categories)
        
        # Get all subcategories under this main category
        subcategories = list(category_structure[selected_main_category].keys())
        
        # Calculate the sampling count for all subcategories under this main category
        counts = []
        for subcategory in subcategories:
            count = subcategory_counts[(selected_main_category, subcategory)]
            counts.append(count)
        
        # Find the minimum and maximum sampling counts under this main category
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        
        # Priority strategy: if there are unsampled subcategories (count=0), prioritize selecting from these subcategories
        # If all subcategories have been sampled, select from subcategories with the fewest samples
        if min_count == 0:
            # There are still unsampled subcategories, only select from these subcategories
            candidate_subcategories = [sub for sub in subcategories 
                                       if subcategory_counts[(selected_main_category, sub)] == 0]
            selected_subcategory = random.choice(candidate_subcategories)
        else:
            # All subcategories have been sampled, select from subcategories with the fewest samples (linear weights)
            # Calculate weights: max_count - current_count + epsilon (linear decreasing relationship)
            epsilon = 1.0
            weights = []
            candidate_subcategories = []
            for subcategory in subcategories:
                count = subcategory_counts[(selected_main_category, subcategory)]
                # Only consider subcategories with the fewest samples
                if count == min_count:
                    candidate_subcategories.append(subcategory)
                    weight = max_count - count + epsilon  # Linear decreasing relationship
                    weights.append(weight)
            
            # If only one candidate, return directly; otherwise use weighted random selection
            if len(candidate_subcategories) == 1:
                selected_subcategory = candidate_subcategories[0]
            else:
                selected_subcategory = random.choices(candidate_subcategories, weights=weights, k=1)[0]
        
        return selected_main_category, selected_subcategory

# Define complexity classes
import json
import random
def get_difficulty():
    with open('./longform_utils/ResearchRubrics_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    return [{'conceptual_breadth': item['conceptual_breadth'],
 'logical_nesting': item['logical_nesting'],
 'exploration': item['exploration']} for item in data]

complexity_classes= get_difficulty()


def sample_complexity_class_with_weights(complexity_classes, lock):
    with lock:
        selected_class = random.choice(complexity_classes)
        return selected_class

# Mapping from Domain to CSV files
DOMAIN_TO_CSV = {
    # Lifestyle & Leisure
    "Shopping": "shopping.csv",
    "Food & Cooking": "food_drink.csv",
    "Sports & Fitness": "sports.csv",
    "Health & Medicine": "health.csv",
    "Pets & Animal Welfare": "pets_animals.csv",
    "Fashion & Beauty": "beauty_fashion.csv",
    "Hobbies & DIY": "hobbies_leisure.csv",
    # Entertainment
    "Films & TV Shows": "entertainment.csv",
    "Gaming & Virtual Worlds": "games.csv",
    "Live Shows & Performances": "entertainment.csv",
    "Music": "entertainment.csv",
    "Books & Reading": "entertainment.csv",
    # Misc.
    "General Info.": "entertainment.csv",  
    "News": "politics.csv",  
    "Legal & Government Services": "law_government.csv",
    "Real Estate": "business_finance.csv",  
    "Finance & Investment": "business_finance.csv",
    # Science & Research
    "Research & Academia": "science.csv",
    "Technology & Science": "technology.csv",
    # Career & Education
    "Education & Learning": "jobs_education.csv",
    "Jobs & Career": "jobs_education.csv",
    # Travel & Transportation
    "Travel & Accommodation": "travel_transportation.csv",
    "Outdoor & Recreation": "travel_transportation.csv",
    "Ticketed Activities": "entertainment.csv",  
}

TRENDING_KEYWORDS_DIR = "../trending_keywords/merge_keywords"

def sample_keywords_from_csv(domain, num_keywords=10):
    """
    Read corresponding CSV file from trending_keywords directory based on domain,
    randomly sample num_keywords keywords
    
    Args:
        domain: Subcategory name (e.g., "Films & TV Shows")
        num_keywords: Number of keywords to sample, default 10
    
    Returns:
        list: List of sampled keywords
    """
    # Get corresponding CSV file name
    csv_file = DOMAIN_TO_CSV.get(domain)
    if not csv_file:
        print(f"Warning: No CSV mapping found for domain '{domain}', returning empty list")
        return []
    
    csv_path = os.path.join(TRENDING_KEYWORDS_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}, returning empty list")
        return []
    
    keywords = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Unify lowercase column names for compatibility with different file formats
            fieldnames = [name.lower() for name in (reader.fieldnames or [])]
            has_trends = 'trends' in fieldnames
            has_trend_breakdown = 'trend breakdown' in fieldnames or 'trend_breakdown' in fieldnames
            has_keyword = 'keyword' in fieldnames or 'keywords' in fieldnames

            for row in reader:
                # 1) Compatible with old format: using "Trends" / "Trend breakdown"
                trend = ''
                trend_breakdown = ''
                if has_trends:
                    trend = (row.get('Trends') or row.get('trends') or '').strip().strip('"')
                if has_trend_breakdown:
                    trend_breakdown = (
                        row.get('Trend breakdown')
                        or row.get('trend breakdown')
                        or row.get('trend_breakdown')
                        or ''
                    ).strip().strip('"')

                # 2) Compatible with current format: only "keyword" / "keywords" column
                if not trend and has_keyword:
                    trend = (
                        row.get('keyword')
                        or row.get('keywords')
                        or row.get('Keyword')
                        or ''
                    ).strip().strip('"')

                # 3) If still empty, but this row has only one field, treat that field as keyword
                if not trend and not trend_breakdown and len(row) == 1:
                    only_val = list(row.values())[0]
                    if isinstance(only_val, str):
                        trend = only_val.strip().strip('"')

                if trend:
                    keywords.append(trend)

                if trend_breakdown:
                    # Split and clean keywords
                    breakdown_keywords = [k.strip() for k in trend_breakdown.split(',') if k.strip()]
                    keywords.extend(breakdown_keywords)
                
                # If old format doesn't exist, try to get from "keyword" column (new format)
                if not trend and not trend_breakdown:
                    keyword = row.get('keyword', '').strip().strip('"')
                    if keyword:
                        keywords.append(keyword)
        
        # Deduplicate and randomly sample
        keywords = list(set(keywords))  # Deduplicate
        if len(keywords) >= num_keywords:
            sampled_keywords = random.sample(keywords, num_keywords)
        else:
            # If insufficient keywords, return all available keywords
            sampled_keywords = keywords
            print(f"Warning: Only {len(keywords)} keywords available for domain '{domain}', less than requested {num_keywords}")
        
        return sampled_keywords
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return []

# Initialize subcategory sampling count
subcategory_counts = initialize_subcategory_counts(category_structure)

# Create thread lock for protecting shared count dictionary
subcategory_lock = threading.Lock()
complexity_lock = threading.Lock()

async def run_single_iteration(iteration_id, subcategory_counts, 
                                subcategory_lock, complexity_lock, category_structure, 
                                complexity_classes, model, semaphore):
    """
    Async function to execute a single iteration
    """
    try:
        async with semaphore:
            # Sample a subcategory based on weights (thread-safe)
            main_category, random_question = sample_subcategory_with_weights(
                category_structure, subcategory_counts, subcategory_lock
            )
            
            # Sample a complexity class based on weights (thread-safe)
            complexity_class = sample_complexity_class_with_weights(
                complexity_classes, complexity_lock
            )
            
            # Update the sampling count for this subcategory (thread-safe)
            with subcategory_lock:
                subcategory_counts[(main_category, random_question)] += 1
                current_subcategory_count = subcategory_counts[(main_category, random_question)]
            
            # Get task name (if available)
            task = asyncio.current_task()
            task_name = task.get_name() if task else f"Task-{iteration_id}"
            
            print(f"\n=== Iteration {iteration_id+1} ({task_name}) ===")
            print(f"Main category: {main_category}")
            print(f"Subcategory: {random_question}")
            print(f"Complexity class: {complexity_class}")
            print(f"This subcategory has been sampled {current_subcategory_count} times")
            
            # Randomly sample 1 keyword from trending_keywords
            sampled_keywords = sample_keywords_from_csv(random_question, num_keywords=1)
            print(f"Sampled keywords ({len(sampled_keywords)}): {sampled_keywords}")
            
            # Create independent agent instance for each task (avoid multi-threading conflicts)
            agent = MultiTurnReactAgent(
                llm=llm_cfg,
                function_list=["search", "visit"]
            )
            
            # Define function to run agent
            def run_agent():
                return agent._run(
                    {
                        "item": {'question': random_question, 'answer': '1'},
                        "complexity_class": complexity_class,
                        "sampled_keywords": sampled_keywords,
                        "iteration_id": iteration_id + 1,
                        "subcategory": random_question,
                    },
                    model
                )
            
            # Run synchronous _run method in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                run_agent
            )
            
            print(f"Result: {result}")
            print("-" * 50)
            
            return result
    finally:
        pass

async def main():
    """
    Main async function, execute all iterations concurrently
    """
    # Total number of tasks: control the total number to generate
    num_iterations = 10  # Can be modified as needed
    
    # Concurrency control: use Semaphore to limit the number of workers running simultaneously
    workers = 1  # Can be modified as needed
    
    # Create thread pool executor
    global executor
    executor = ThreadPoolExecutor(max_workers=workers)
    
    # Create Semaphore to control concurrency
    semaphore = asyncio.Semaphore(workers)
    
    # Create all tasks
    tasks = []
    for i in range(num_iterations):
        task = asyncio.create_task(
            run_single_iteration(
                i, subcategory_counts,
                subcategory_lock, complexity_lock,
                category_structure, complexity_classes, model, semaphore
            ),
            name=f"Task-{i}"
        )
        tasks.append(task)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Calculate total cost
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    
    # Print execution result summary and detailed error information
    print("\n" + "=" * 50)
    print("All tasks completed!")
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    failure_count = sum(1 for r in results if isinstance(r, Exception))
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")

    # Accumulate cost information for all tasks
    for result in results:
        if not isinstance(result, Exception) and isinstance(result, dict):
            cost_info = result.get("cost_info", {})
            total_cost += cost_info.get("cost", 0.0)
            total_prompt_tokens += cost_info.get("prompt_tokens", 0)
            total_completion_tokens += cost_info.get("completion_tokens", 0)
            total_tokens += cost_info.get("total_tokens", 0)
    
    # Print cost statistics
    print("\n" + "-" * 50)
    print("Cost Statistics:")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Prompt Tokens: {total_prompt_tokens:,}")
    print(f"  Completion Tokens: {total_completion_tokens:,}")
    print(f"  Total Tokens: {total_tokens:,}")
    print("-" * 50)
    
    # Print detailed error information
    if failure_count > 0:
        print("\n" + "-" * 50)
        print("Failed Task Details:")
        print("-" * 50)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"\nTask {i} failed:")
                print(f"Exception type: {type(result).__name__}")
                print(f"Exception message: {str(result)}")
                import traceback
                print("Detailed stack trace:")
                traceback.print_exception(type(result), result, result.__traceback__)
        print("-" * 50)
    
    print("=" * 50)
    
    return results

# Run main async function
if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    finally:
        # Shutdown thread pool executor
        if executor is not None:
            executor.shutdown(wait=True)
