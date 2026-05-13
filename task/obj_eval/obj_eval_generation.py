import os
import time
import re
import json
import argparse
import threading
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from litellm import completion

DEFAULT_MODEL = os.environ.get("OBJ_EVAL_MODEL_NAME", "openai/gpt-5")

# Global token statistics and lock
token_stats = {
    'completion_tokens': 0,
    'prompt_tokens': 0,
    'total_tokens': 0,
    'reasoning_tokens': 0
}
token_lock = threading.Lock()

# Completed task counter
completed_tasks = 0
completed_tasks_lock = threading.Lock()

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=5, backoff_factor=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"❌ Error occurred: {e}")
                    retries += 1
                    if retries < max_retries:
                        wait_time = backoff_factor ** retries
                        print(f"🔄 Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
        return wrapper
    return decorator


def read_file(filepath: Path) -> str:
    """Read file contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return f"[File not found: {filepath}]"
    except Exception as e:
        print(f"❌ Error reading file {filepath}: {e}")
        return f"[Read error: {e}]"


def format_json(json_path: Path) -> str:
    """Read and pretty-format a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        print(f"⚠️  File not found: {json_path}")
        return f"[File not found: {json_path}]"
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error in {json_path}: {e}")
        return f"[JSON parse error: {e}]"
    except Exception as e:
        print(f"❌ Error reading file {json_path}: {e}")
        return f"[Read error: {e}]"


def extract_python_code(content: str) -> str:
    """Extract Python code block content; return original text if none exists."""
    # Regex match for ```python ... ```
    pattern = r'```python\s*(.*?)\s*```'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    return content


def extract_file_number(filename: str) -> int:
    """Extract numeric index from filename."""
    pattern = r'traj_(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return 0


@retry_with_backoff(max_retries=5, backoff_factor=2)
def call_model_api(content, model_name):
    """Call the model API."""
    response = completion(
        model=model_name,
        messages=[{ "content": content,"role": "user"}],
        reasoning_effort="high",
        max_tokens=32384
    )
    
    # Update token usage statistics
    with token_lock:
        if hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, 'completion_tokens'):
                token_stats['completion_tokens'] += usage.completion_tokens
            if hasattr(usage, 'prompt_tokens'):
                token_stats['prompt_tokens'] += usage.prompt_tokens
            if hasattr(usage, 'total_tokens'):
                token_stats['total_tokens'] += usage.total_tokens
            
            # Track reasoning token count when available
            if hasattr(usage, 'completion_tokens_details'):
                completion_details = usage.completion_tokens_details
                if hasattr(completion_details, 'reasoning_tokens'):
                    token_stats['reasoning_tokens'] += completion_details.reasoning_tokens
    
    return response


def process_single_json(json_file: Path, template: str, static_replacements: dict, output_dir: Path, model_name: str):
    """
    Process a single JSON file and generate Python code directly.
    
    Args:
        json_file: Input JSON file path.
        template: Prompt template content.
        static_replacements: Static placeholder replacement dictionary.
        output_dir: Output directory for generated Python code.
    """
    try:
        # Read input JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # js = json.loads(data['answer'])
        # Build task payload from required fields
        task = {
            'task_description': data['proposed_question'],
            'rubric_tree': data["rubric_tree_analysis_refined"]['formatted_tree']
        }
        # Convert task to formatted string
        task_str = json.dumps(task, indent=2, ensure_ascii=False)
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Merge all replacement values
        replacements = static_replacements.copy()
        replacements.update({
            "{current_date}": current_date,
            "{task}": task_str
        })
        
        # Replace placeholders in template
        formatted_content = template
        for placeholder, value in replacements.items():
            if placeholder in formatted_content:
                formatted_content = formatted_content.replace(placeholder, value)
        
        # Skip API call if output already exists
        py_filename = f"tree2py_{json_file.stem}.py"
        output_file = output_dir / py_filename
        
        if output_file.exists():
            tqdm.write(f"⚠️ Output file already exists, skipping API call: {output_file}")
            return False
        
        # Call API
        response = call_model_api(formatted_content, model_name)
        
        # Extract response content
        response_content = response.choices[0].message.content
        
        # Extract Python code block
        extracted_content = extract_python_code(response_content)

        if extracted_content.strip() == "":
            tqdm.write(f"⚠️ Empty code block extracted, skipping save: {output_file}")
            # Save raw model response to a separate file
            raw_filename = f"tree2script_formatted_{json_file.stem}_raw.md"
            raw_file = output_dir / raw_filename
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(response_content)
            return False
        
        # Write generated code to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_content)
        
        tqdm.write(f"✅ Saved generated result to: {output_file}")
        return True
        
    except Exception as e:
        tqdm.write(f"❌ Error processing file {json_file}: {e}")
        return False


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate Python evaluation scripts directly from JSON files')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input JSON file or directory path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory path for generated Python code')
    parser.add_argument('--template', '-t', type=str, default=None,
                        help='Template file path')
    parser.add_argument('--concurrency', '-c', type=int, default=50,
                        help='Number of worker threads for concurrent processing (default: 50)')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        help='LiteLLM model name for script generation')
    args = parser.parse_args()
    
    # Define default paths
    base_dir = Path(__file__).parent
    tree2script_dir = base_dir / "utils"
    
    # Template file
    if args.template:
        template_path = Path(args.template)
    else:
        template_path = base_dir / "generation_prompt.md"
    
    # Input JSON file or directory
    if args.input:
        input_path = Path(args.input)
    else:
        obj_task_output = base_dir.parent / "obj_task" / "outputs" / "objective_trajectories" / "formatted"
        refined_input = obj_task_output / "refined"
        input_path = refined_input if refined_input.exists() else obj_task_output

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_dir / "obj_eval"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📝 Starting generation of Python evaluation scripts from JSON...")
    print(f"   Template file: {template_path}")
    print(f"   Input path: {input_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Concurrency: {args.concurrency}")
    print(f"   Model: {args.model}")
    
    # Read template
    template = read_file(template_path)
    if "[" in template[:50]:  # Basic check for read failure marker
        print("❌ Failed to read template file. Exiting.")
        return
    
    # Read static code files once
    print("\n📂 Reading static code files...")
    evaluator_code = read_file(tree2script_dir / "evaluator.py")
    verification_tree_code = read_file(tree2script_dir / "verification_tree.py")
    eval_toolkit_code = read_file(tree2script_dir / "eval_toolkit.py")
    
    # Read example 1 (yu_lineage)
    rubric_tree_json1 = format_json(tree2script_dir / "yu_lineage.json")
    python_code1 = read_file(tree2script_dir / "yu_lineage.py")
    
    # Read example 2 (ad_patent)
    rubric_tree_json2 = format_json(tree2script_dir / "ad_patent.json")
    python_code2 = read_file(tree2script_dir / "ad_patent.py")
    
    # Build static replacement dictionary
    static_replacements = {
        "{evaluator_code}": evaluator_code,
        "{verification_tree_code}": verification_tree_code,
        "{eval_toolkit_code}": eval_toolkit_code,
        "{rubric_tree_json1}": rubric_tree_json1,
        "{python_code1}": python_code1,
        "{rubric_tree_json2}": rubric_tree_json2,
        "{python_code2}": python_code2
    }
    
    # Collect JSON files to process
    if input_path.is_file() and input_path.suffix == '.json':
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
    else:
        print(f"❌ Input path is neither a JSON file nor a directory: {input_path}")
        return

    # Filter by verification results - only process files with decision "YES"
    verification_results_path = base_dir.parent / "obj_task" / "outputs" / "objective_trajectories" / "formatted" / "refined" / "verifier" / "rubrc-tree-verification-results.json"
    if verification_results_path.exists():
        print(f"\n🔍 Loading verification results from: {verification_results_path}")
        try:
            with open(verification_results_path, 'r', encoding='utf-8') as f:
                verification_results = json.load(f)
            # Build set of approved file paths (decision == "YES")
            approved_files = set()
            for item in verification_results:
                if item.get("decision", "").upper() == "YES":
                    task_file = item.get("task_file", "")
                    if task_file:
                        approved_files.add(Path(task_file).name)
            # Filter json_files to only include approved ones
            original_count = len(json_files)
            json_files = [f for f in json_files if f.name in approved_files]
            print(f"   Filtered: {original_count} -> {len(json_files)} files (only decision='YES')")
        except Exception as e:
            print(f"⚠️ Failed to load verification results: {e}")
            print("   Proceeding without filtering...")
    else:
        print(f"⚠️ Verification results file not found: {verification_results_path}")
        print("   Proceeding without filtering...")

    # Sort by filename index
    json_files_sorted = sorted(json_files, key=lambda x: extract_file_number(x.name))
    
    print(f"\n📊 Found {len(json_files_sorted)} JSON files")
    
    # Process each JSON file
    print("\n🔄 Starting batch processing...")
    success_count = 0
    
    # Process concurrently
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Submit tasks
        futures = {
            executor.submit(
                process_single_json,
                json_file,
                template,
                static_replacements,
                output_dir,
                args.model,
            ): json_file
            for json_file in json_files_sorted
        }
        
        # Handle task results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            json_file = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
            except Exception as e:
                tqdm.write(f"❌ Error processing file {json_file}: {e}")
            finally:
                # Update completed-task count and average token usage
                with completed_tasks_lock:
                    global completed_tasks
                    completed_tasks += 1
                
                with token_lock:
                    # Compute running average token usage
                    avg_prompt_tokens = token_stats['prompt_tokens'] / completed_tasks if completed_tasks > 0 else 0
                    avg_completion_tokens = token_stats['completion_tokens'] / completed_tasks if completed_tasks > 0 else 0
                    avg_reasoning_tokens = token_stats['reasoning_tokens'] / completed_tasks if completed_tasks > 0 else 0
                    avg_total_tokens = token_stats['total_tokens'] / completed_tasks if completed_tasks > 0 else 0
                
                # Print running average token usage
                tqdm.write(f"📊 Progress: {completed_tasks}/{len(futures)}")
                tqdm.write(f"   Avg prompt tokens: {int(avg_prompt_tokens)}")
                tqdm.write(f"   Avg completion tokens: {int(avg_completion_tokens)}")
                tqdm.write(f"   Avg reasoning tokens: {int(avg_reasoning_tokens)}")
                tqdm.write(f"   Avg total tokens: {int(avg_total_tokens)}")
    
    # Print summary statistics
    print(f"\n📊 Batch processing completed!")
    print(f"   Total files: {len(json_files_sorted)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(json_files_sorted) - success_count}")
    
    # Print token usage statistics
    print(f"\n💡 Token usage:")
    print(f"   Prompt tokens: {token_stats['prompt_tokens']}")
    print(f"   Completion tokens: {token_stats['completion_tokens']}")
    print(f"   Reasoning tokens: {token_stats['reasoning_tokens']}")
    print(f"   Total tokens: {token_stats['total_tokens']}")
    
    # Print final average token usage
    print(f"\n📈 Average token usage:")
    print(f"   Avg prompt tokens: {int(token_stats['prompt_tokens'] / completed_tasks) if completed_tasks > 0 else 0}")
    print(f"   Avg completion tokens: {int(token_stats['completion_tokens'] / completed_tasks) if completed_tasks > 0 else 0}")
    print(f"   Avg reasoning tokens: {int(token_stats['reasoning_tokens'] / completed_tasks) if completed_tasks > 0 else 0}")
    print(f"   Avg total tokens: {int(token_stats['total_tokens'] / completed_tasks) if completed_tasks > 0 else 0}")
    
    print("\n🎉 All operations completed!")


if __name__ == "__main__":
    main()
