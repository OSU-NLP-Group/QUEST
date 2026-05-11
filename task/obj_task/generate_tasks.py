from generation_agent import MultiTurnReactAgent
import os
import random
import csv
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
VISIT_SERVICE = "jina"
model = os.environ.get("DEEPRESEARCH_MODEL_NAME", "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
if model.startswith("bedrock/"):
    model_type = 'bedrock'
elif model.startswith("azure/"):
    model_type = 'azure'
elif model.startswith("openai/"):
    model_type = 'openai'
elif model.startswith("vllm/"):
    model_type = 'vllm'
else:
    model_type = 'openai'
    print(f"Warning: Model name '{model}' does not start with a known prefix (openai/, azure/, bedrock/, vllm/). "
          f"Defaulting to 'openai' type. Please use the correct format.")
if model_type == 'openai':
    generate_cfg = {
        'max_tokens': 30000,
        'max_retries': 10,
        'temperature': 1,
    }
else:
    generate_cfg = {
        'max_tokens': 30000,
        'max_retries': 10,
        'temperature': 1,
        'top_p': 0.95,
    }

llm_cfg = {
    'model': model,
    'generate_cfg': generate_cfg,
    'model_type': model_type,
    'model_path': os.getenv("MEMORY_TOKENIZER_PATH", "")
}
executor = None
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
    """Documentation omitted."""
    counts = {}
    for main_category, subcategories in category_structure.items():
        for subcategory in subcategories.keys():
            counts[(main_category, subcategory)] = 0
    return counts

def sample_subcategory_with_weights(category_structure, subcategory_counts, lock):
    """Documentation omitted."""
    with lock:
        main_categories = list(category_structure.keys())
        selected_main_category = random.choice(main_categories)
        subcategories = list(category_structure[selected_main_category].keys())
        counts = []
        for subcategory in subcategories:
            count = subcategory_counts[(selected_main_category, subcategory)]
            counts.append(count)
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        if min_count == 0:
            candidate_subcategories = [sub for sub in subcategories 
                                       if subcategory_counts[(selected_main_category, sub)] == 0]
            selected_subcategory = random.choice(candidate_subcategories)
        else:
            epsilon = 1.0
            weights = []
            candidate_subcategories = []
            for subcategory in subcategories:
                count = subcategory_counts[(selected_main_category, subcategory)]
                if count == min_count:
                    candidate_subcategories.append(subcategory)
                    weight = max_count - count + epsilon
                    weights.append(weight)
            if len(candidate_subcategories) == 1:
                selected_subcategory = candidate_subcategories[0]
            else:
                selected_subcategory = random.choices(candidate_subcategories, weights=weights, k=1)[0]
        
        return selected_main_category, selected_subcategory
complexity_classes = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

def initialize_complexity_class_counts(complexity_classes):
    """Documentation omitted."""
    counts = {}
    for complexity_class in complexity_classes:
        counts[complexity_class] = 0
    return counts

def sample_complexity_class_with_weights(complexity_classes, complexity_counts, lock):
    """Documentation omitted."""
    with lock:
        groups = {
            'group1': {'classes': ['C1', 'C4', 'C7'], 'target_prob': 0.50},
            'group2': {'classes': ['C2', 'C5', 'C8'], 'target_prob': 0.40},
            'group3': {'classes': ['C3', 'C6', 'C9'], 'target_prob': 0.10}
        }
        group_avg_counts = {}
        for group_name, group_info in groups.items():
            group_classes = group_info['classes']
            group_counts = [complexity_counts[cls] for cls in group_classes]
            group_avg_counts[group_name] = sum(group_counts) / len(group_counts) if group_counts else 0
        all_avg_counts = list(group_avg_counts.values())
        min_avg_count = min(all_avg_counts) if all_avg_counts else 0
        max_avg_count = max(all_avg_counts) if all_avg_counts else 0
        group_weights = []
        group_names = []
        for group_name, group_info in groups.items():
            target_prob = group_info['target_prob']
            avg_count = group_avg_counts[group_name]
            if min_avg_count == 0:
                if avg_count == 0:
                    balance_factor = 1.0
                else:
                    balance_factor = 0.1
            else:
                epsilon = 0.1
                balance_factor = (max_avg_count - avg_count + epsilon) / (max_avg_count + epsilon)
            weight = target_prob * balance_factor
            group_weights.append(weight)
            group_names.append(group_name)
        selected_group_name = random.choices(group_names, weights=group_weights, k=1)[0]
        selected_group_classes = groups[selected_group_name]['classes']
        group_counts = [complexity_counts[cls] for cls in selected_group_classes]
        min_count_in_group = min(group_counts) if group_counts else 0
        max_count_in_group = max(group_counts) if group_counts else 0
        if min_count_in_group == 0:
            candidate_classes = [cls for cls in selected_group_classes 
                              if complexity_counts[cls] == 0]
            selected_class = random.choice(candidate_classes)
        else:
            epsilon = 1.0
            weights = []
            candidate_classes = []
            for complexity_class in selected_group_classes:
                count = complexity_counts[complexity_class]
                if count == min_count_in_group:
                    candidate_classes.append(complexity_class)
                    weight = max_count_in_group - count + epsilon
                    weights.append(weight)
            if len(candidate_classes) == 1:
                selected_class = candidate_classes[0]
            else:
                selected_class = random.choices(candidate_classes, weights=weights, k=1)[0]
        
        return selected_class
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

TRENDING_KEYWORDS_DIR = os.getenv(
    "TRENDING_KEYWORDS_DIR",
    "../trending_keywords/merge_keywords",
)

def sample_keywords_from_csv(domain, num_keywords=10):
    """Documentation omitted."""
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
            fieldnames = [name.lower() for name in (reader.fieldnames or [])]
            has_trends = 'trends' in fieldnames
            has_trend_breakdown = 'trend breakdown' in fieldnames or 'trend_breakdown' in fieldnames
            has_keyword = 'keyword' in fieldnames or 'keywords' in fieldnames

            for row in reader:
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
                if not trend and has_keyword:
                    trend = (
                        row.get('keyword')
                        or row.get('keywords')
                        or row.get('Keyword')
                        or ''
                    ).strip().strip('"')
                if not trend and not trend_breakdown and len(row) == 1:
                    only_val = list(row.values())[0]
                    if isinstance(only_val, str):
                        trend = only_val.strip().strip('"')

                if trend:
                    keywords.append(trend)

                if trend_breakdown:
                    breakdown_keywords = [k.strip() for k in trend_breakdown.split(',') if k.strip()]
                    keywords.extend(breakdown_keywords)
                if not trend and not trend_breakdown:
                    keyword = row.get('keyword', '').strip().strip('"')
                    if keyword:
                        keywords.append(keyword)
        keywords = list(set(keywords))
        if len(keywords) >= num_keywords:
            sampled_keywords = random.sample(keywords, num_keywords)
        else:
            sampled_keywords = keywords
            print(f"Warning: Only {len(keywords)} keywords available for domain '{domain}', less than requested {num_keywords}")
        
        return sampled_keywords
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return []
subcategory_counts = initialize_subcategory_counts(category_structure)
complexity_counts = initialize_complexity_class_counts(complexity_classes)
subcategory_lock = threading.Lock()
complexity_lock = threading.Lock()

async def run_single_iteration(iteration_id, subcategory_counts, complexity_counts, 
                                subcategory_lock, complexity_lock, category_structure, 
                                complexity_classes, model, semaphore):
    """Documentation omitted."""
    global instance_manager
    
    task_id = f"task_{iteration_id}"
    instance_ip = None
    
    try:
        async with semaphore:
            main_category, random_question = sample_subcategory_with_weights(
                category_structure, subcategory_counts, subcategory_lock
            )
            complexity_class = sample_complexity_class_with_weights(
                complexity_classes, complexity_counts, complexity_lock
            )
            with subcategory_lock:
                subcategory_counts[(main_category, random_question)] += 1
                current_subcategory_count = subcategory_counts[(main_category, random_question)]
            with complexity_lock:
                complexity_counts[complexity_class] += 1
                current_complexity_count = complexity_counts[complexity_class]
            task = asyncio.current_task()
            task_name = task.get_name() if task else f"Task-{iteration_id}"
            
            print(f"\n=== Iteration {iteration_id+1} ({task_name}) ===")
            print(f"Main category: {main_category}")
            print(f"Subcategory: {random_question}")
            print(f"Complexity class: {complexity_class}")
            print(f"Subcategory sampled count: {current_subcategory_count}")
            print(f"Complexity sampled count: {current_complexity_count}")
            print(f"Visit service: {VISIT_SERVICE.upper()}")
            sampled_keywords = sample_keywords_from_csv(random_question, num_keywords=10)
            print(f"Sampled keywords ({len(sampled_keywords)}): {sampled_keywords}")
            agent = MultiTurnReactAgent(
                llm=llm_cfg,
                function_list=["search", "visit"],
            )
            
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
    """Documentation omitted."""
    num_iterations = 20
    workers = 20
    global executor
    executor = ThreadPoolExecutor(max_workers=workers)
    semaphore = asyncio.Semaphore(workers)
    tasks = []
    for i in range(num_iterations):
        task = asyncio.create_task(
            run_single_iteration(
                i, subcategory_counts, complexity_counts,
                subcategory_lock, complexity_lock,
                category_structure, complexity_classes, model, semaphore
            ),
            name=f"Task-{i}"
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    print("\n" + "=" * 50)
    print("All tasks completed!")
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    failure_count = sum(1 for r in results if isinstance(r, Exception))
    print(f"Succeeded: {success_count}")
    print(f"Failed: {failure_count}")
    for result in results:
        if not isinstance(result, Exception) and isinstance(result, dict):
            cost_info = result.get("cost_info", {})
            total_cost += cost_info.get("cost", 0.0)
            total_prompt_tokens += cost_info.get("prompt_tokens", 0)
            total_completion_tokens += cost_info.get("completion_tokens", 0)
            total_tokens += cost_info.get("total_tokens", 0)
    print("\n" + "-" * 50)
    print("Cost summary:")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Prompt Tokens: {total_prompt_tokens:,}")
    print(f"  Completion Tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print("-" * 50)
    if failure_count > 0:
        print("\n" + "-" * 50)
        print("Failed task details:")
        print("-" * 50)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"\nTask {i} failed:")
                print(f"Exception type: {type(result).__name__}")
                print(f"Exception message: {str(result)}")
                import traceback
                print("Traceback:")
                traceback.print_exception(type(result), result, result.__traceback__)
        print("-" * 50)
    
    print("=" * 50)
    
    return results
if __name__ == "__main__":
    print(f"=== Using {VISIT_SERVICE.upper()} for web access ===")
    
    try:
        results = asyncio.run(main())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
