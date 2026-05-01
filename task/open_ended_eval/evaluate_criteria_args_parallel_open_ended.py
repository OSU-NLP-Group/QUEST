import json
import os
import re
import time
import argparse
import random
from tqdm import tqdm
import concurrent.futures
import threading
from pathlib import Path
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from eva_open_ended import SYSTEM_PROMPT, USER_PROMPT


# Initialize AI client
class AIClient():
    def __init__(self, model):
        self.model = model

    def _strip_wrapping_quotes(self, value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1].strip()
        return value

    def _load_server_endpoints(self):
        """Read env or config file on each call to support hot-swapping."""
        hostname_list = os.getenv('HOSTNAME_LIST', 'localhost')
        port_list = os.getenv('PORTS', '8000')
        endpoint_source = "env"

        config_file = os.getenv('SERVER_ENDPOINTS_FILE', '').strip()
        if config_file:
            config_path = Path(config_file).expanduser()
            if config_path.is_file():
                try:
                    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = self._strip_wrapping_quotes(value)
                        if key == "HOSTNAME_LIST" and value:
                            hostname_list = value
                        elif key == "PORTS" and value:
                            port_list = value
                    endpoint_source = f"file:{config_path}"
                except Exception as e:
                    print(f"Warning: failed to read endpoint config {config_path}: {e}. Falling back to env values.")
            else:
                print(f"Warning: SERVER_ENDPOINTS_FILE not found: {config_path}. Falling back to env values.")

        hosts = [h.strip() for h in hostname_list.split(',') if h.strip()]
        if not hosts:
            hosts = ['localhost']

        ports = []
        for raw_port in port_list.split(','):
            raw_port = raw_port.strip()
            if not raw_port:
                continue
            try:
                ports.append(int(raw_port))
            except ValueError:
                print(f"Warning: invalid port ignored: {raw_port}")
        if not ports:
            ports = [8000]

        return hosts, ports, endpoint_source

    def generate(self, user_prompt, system_prompt=""):
        """Single request: one service call corresponds to exactly one request, returns None on failure.
        Retry logic is handled by the caller (evaluate_single_criterion)."""
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        hosts, ports, endpoint_source = self._load_server_endpoints()
        all_endpoints = [(host, port) for host in hosts for port in ports]
        selected_host, selected_port = random.choice(all_endpoints)
        openai_api_base = f"http://{selected_host}:{selected_port}/v1"

        print(f"--- API call | endpoint source: {endpoint_source} | using: {selected_host}:{selected_port} ---")

        client = OpenAI(
            api_key="EMPTY",
            base_url=openai_api_base,
            timeout=600.0,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=0.6,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            else:
                print(f"Warning: empty response from {selected_host}:{selected_port}.")
                return None
        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error: API/network error on {selected_host}:{selected_port}: {e}")
            return None
        except Exception as e:
            print(f"Error: unexpected error on {selected_host}:{selected_port}: {e}")
            return None


# Global AI client, initialized in main()
ai_client = None


def extract_answer_content(text: str) -> str:
    """Strip <answer>...</answer> wrapper tags and return pure answer text.
    If <answer> is present, extract content inside tags; if no tags but text is non-empty, return full text (for compatibility with untagged ref/answer)."""
    text = (text or '').strip()
    if not text:
        return ''
    if '<answer>' not in text:
        return text  # Treat as valid if no tags but non-empty (some refs have plain text only)
    start = text.find('<answer>') + len('<answer>')
    end = text.find('</answer>')
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def extract_json_from_response(response_text):
    """Extract JSON content from response."""
    # Try to extract JSON code block
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        # If no code block, try parsing the whole response directly
        json_str = response_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {response_text[:500]}...")
        return None


def normalize_result_types(result):
    """Convert numeric fields in result to float type uniformly."""
    if not result:
        return result

    # Convert score_a
    if 'score_a' in result:
        try:
            result['score_a'] = float(result['score_a'])
        except (ValueError, TypeError):
            print(f"    Warning: cannot convert score_a to float: {result['score_a']}")
            result['score_a'] = 0.0

    # Convert score_b
    if 'score_b' in result:
        try:
            result['score_b'] = float(result['score_b'])
        except (ValueError, TypeError):
            print(f"    Warning: cannot convert score_b to float: {result['score_b']}")
            result['score_b'] = 0.0

    # Convert confidence
    if 'confidence' in result:
        try:
            result['confidence'] = float(result['confidence'])
        except (ValueError, TypeError):
            print(f"    Warning: cannot convert confidence to float: {result['confidence']}")
            result['confidence'] = 0.0

    # Convert weight
    if 'weight' in result:
        try:
            result['weight'] = float(result['weight'])
        except (ValueError, TypeError):
            print(f"    Warning: cannot convert weight to float: {result['weight']}")
            result['weight'] = 1.0

    return result


def validate_score(result):
    """Validate that score_a and score_b fields in evaluation result are valid."""
    if not result:
        return False

    if 'score_a' not in result or 'score_b' not in result:
        return False

    try:
        score_a = float(result['score_a'])
        score_b = float(result['score_b'])

        # Check if scores are within reasonable range (0-10)
        if 0 <= score_a <= 10 and 0 <= score_b <= 10:
            return True
        else:
            print(f"    Warning: score out of range [0, 10]: score_a={score_a}, score_b={score_b}")
            return False
    except (ValueError, TypeError):
        print(f"    Warning: cannot convert scores to numbers: score_a={result.get('score_a', 'N/A')}, score_b={result.get('score_b', 'N/A')}")
        return False


def evaluate_single_criterion(document_content, ref_content, query, criterion_name, criterion_data, category, max_retries=3):
    """Evaluate a single criterion with retry mechanism."""
    user_prompt = USER_PROMPT.replace('<<document_content>>', document_content) \
                              .replace('<<ref_content>>', ref_content) \
                              .replace('<<query>>', query) \
                              .replace('<<rubric_title>>', criterion_name) \
                              .replace('<<rubric_category>>', category) \
                              .replace('<<rubric_explanation>>', criterion_data['explanation'])

    for attempt in range(max_retries):
        response = ai_client.generate(user_prompt, SYSTEM_PROMPT)

        if response is None:
            print(f"    Attempt {attempt + 1}/{max_retries}: API call failed")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                break

        result = extract_json_from_response(response)

        if result:
            result['criterion_name'] = criterion_name
            result['category'] = category
            result['weight'] = criterion_data['weight']

            # Normalize numeric types
            result = normalize_result_types(result)

            # Validate scores
            if validate_score(result):
                return result
            else:
                print(f"    Attempt {attempt + 1}/{max_retries}: score validation failed")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
        else:
            print(f"    Attempt {attempt + 1}/{max_retries}: JSON parsing failed")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue

    # All retries failed, return default result with 0 score
    print(f"    All retries failed, assigning 0 score")
    default_result = {
        'score_a': 0.0,  # Document A (to evaluate)
        'score_b': 0.0,  # Document B (reference)
        'reason': 'Evaluation failed, could not get valid score after multiple retries',
        'criterion_name': criterion_name,
        'category': category,
        'weight': float(criterion_data['weight'])
    }
    return normalize_result_types(default_result)


def evaluate_single_criterion_wrapper(args):
    """Wrapper for evaluate_single_criterion function, used for concurrent execution."""
    document_content, ref_content, query, criterion_data, dimension = args
    criterion_name = criterion_data['criterion']
    
    print(f"\n  Evaluating criterion: {criterion_name}")

    result = evaluate_single_criterion(
        document_content=document_content,
        ref_content=ref_content,
        query=query,
        criterion_name=criterion_name,
        criterion_data=criterion_data,
        category=dimension
    )

    print(f"    Document score: {result.get('score_a', 'N/A')}, Reference score: {result.get('score_b', 'N/A')}")
    return result


def evaluate_document(criteria_item, answer_item, ref_item):
    """Evaluate all criteria for a single document (scored individually)."""
    document_id = criteria_item['id']
    query = criteria_item['prompt']
    document_content = extract_answer_content(answer_item.get('prediction', answer_item.get('answer', '')))
    ref_content = extract_answer_content(ref_item.get('prediction', ref_item.get('answer', '')))
    if "answer_related_urls" in answer_item and answer_item["answer_related_urls"]:
        document_content = document_content + "\nRelated URL:" + str(answer_item["answer_related_urls"])
    if "answer_related_urls" in ref_item and ref_item["answer_related_urls"]:
        ref_content = ref_content + "\nRelated URL:" + str(ref_item["answer_related_urls"])
    dimension_weights = criteria_item['dimension_weight']
    criterions = criteria_item['criterions']

    print(f"\n{'='*80}")
    print(f"Starting evaluation for document ID: {document_id}")
    print(f"{'='*80}")

    # Store all evaluation results
    all_evaluations = {}
    dimension_scores_a = {}  # Document to evaluate dimension scores
    dimension_scores_b = {}  # Reference dimension scores

    # Iterate over each dimension
    for dimension, criteria_list in criterions.items():
        print(f"\nEvaluating dimension: {dimension}")

        # Prepare concurrent task arguments
        task_args = [
            (document_content, ref_content, query, criterion_data, dimension)
            for criterion_data in criteria_list
        ]

        # Execute concurrent evaluation
        dimension_evaluations = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Use tqdm to show progress
            results = list(tqdm(executor.map(evaluate_single_criterion_wrapper, task_args),
                              total=len(criteria_list), desc=f"  {dimension}"))
            dimension_evaluations.extend(results)

        all_evaluations[dimension] = dimension_evaluations

        # Calculate weighted score for this dimension (document and reference)
        if dimension_evaluations:
            # Calculate weighted sum for document
            weighted_sum_a = sum(float(eval_result.get('score_a', 0)) * float(eval_result.get('weight', 1))
                              for eval_result in dimension_evaluations)
            # Calculate weighted sum for reference
            weighted_sum_b = sum(float(eval_result.get('score_b', 0)) * float(eval_result.get('weight', 1))
                              for eval_result in dimension_evaluations)
            total_weight = sum(float(eval_result.get('weight', 1)) for eval_result in dimension_evaluations)

            # Calculate weighted average score
            dimension_score_a = weighted_sum_a / total_weight if total_weight > 0 else 0
            dimension_score_b = weighted_sum_b / total_weight if total_weight > 0 else 0

            dimension_scores_a[dimension] = dimension_score_a
            dimension_scores_b[dimension] = dimension_score_b
            print(f"\n  {dimension} dimension score - Document: {dimension_score_a:.4f}, Reference: {dimension_score_b:.4f}")

    # Calculate total score (based on dimension weights)
    total_score_a = sum(dimension_scores_a.get(dim, 0) * weight
                      for dim, weight in dimension_weights.items())
    total_score_b = sum(dimension_scores_b.get(dim, 0) * weight
                      for dim, weight in dimension_weights.items())

    # Calculate score ratio for each dimension
    dimension_score_ratios = {}
    for dim in dimension_scores_a:
        score_a = dimension_scores_a.get(dim, 0)
        score_b = dimension_scores_b.get(dim, 0)
        if score_a + score_b > 0:
            ratio = score_a / (score_a + score_b)
        else:
            ratio = 0.0
        dimension_score_ratios[dim] = ratio

    # Calculate final score ratio
    if total_score_a + total_score_b > 0:
        final_score = total_score_a / (total_score_a + total_score_b)
    else:
        final_score = 0.0

    print(f"\n{'='*80}")
    print(f"Document ID {document_id} evaluation complete")
    print(f"Document total score: {total_score_a:.4f}")
    print(f"Reference total score: {total_score_b:.4f}")
    print(f"Final score: {final_score:.4f}")
    print(f"Dimension scores - Document: {dimension_scores_a}")
    print(f"Dimension scores - Reference: {dimension_scores_b}")
    print(f"Dimension score ratios: {dimension_score_ratios}")
    print(f"{'='*80}\n")

    return {
        'id': document_id,
        'iter': answer_item.get('iter'),
        'query': query,
        'final_score': final_score,
        'total_score_a': total_score_a,  # Document total score
        'total_score_b': total_score_b,  # Reference total score
        'dimension_scores_a': dimension_scores_a,  # Document dimension scores
        'dimension_scores_b': dimension_scores_b,  # Reference dimension scores
        'dimension_scores': dimension_score_ratios,  # Dimension score ratios
        'dimension_weights': dimension_weights,
        'detailed_evaluations': all_evaluations
    }


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Document quality evaluation tool')
    parser.add_argument('--model', type=str, default='Qwen3-4B-Thinking-2507', help='Model name to use')
    parser.add_argument('--prompt_to_eval', type=str, help='Evaluation criteria file path')
    parser.add_argument('--answer_to_eval', type=str, help='Answer file path')
    parser.add_argument('--ref_to_eval', type=str, help='Reference file path, multiple files separated by comma (auto dedup)')
    parser.add_argument('--output_file', type=str, help='Output result file path')
    parser.add_argument('--hostname_list', type=str, default="localhost",
                        help='Comma-separated list of LLM service hosts, e.g. "host1,host2"')
    parser.add_argument('--ports', type=str, default="8000",
                        help='Comma-separated list of ports, e.g. "8000,8001"')
    parser.add_argument('--endpoints_file', type=str, default="",
                        help='Hot-swap endpoint config file path (takes effect at runtime), file format: HOSTNAME_LIST=... / PORTS=...')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Document-level concurrency (how many documents to evaluate simultaneously), default 8')
    args = parser.parse_args()

    # Write parameters to env vars, _load_server_endpoints() reads them on each call for hot-swap support
    os.environ['HOSTNAME_LIST'] = args.hostname_list
    os.environ['PORTS'] = args.ports
    if args.endpoints_file:
        os.environ['SERVER_ENDPOINTS_FILE'] = args.endpoints_file

    # Initialize AI client
    global ai_client
    ai_client = AIClient(args.model)

    print("Loading data...")
    print(f"Output file: {args.output_file}")

    # Load criteria, answer and reference data
    criteria_data = []
    with open(args.prompt_to_eval, 'r', encoding='utf-8') as f:
        for line in f:
            criteria_data.append(json.loads(line.strip()))

    answer_data = []
    with open(args.answer_to_eval, 'r', encoding='utf-8') as f:
        for line in f:
            answer_data.append(json.loads(line.strip()))

    ref_data = []
    ref_seen_questions = set()
    for ref_path in args.ref_to_eval.split(','):
        ref_path = ref_path.strip()
        with open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                q = item.get('question', '')
                if q not in ref_seen_questions:
                    ref_seen_questions.add(q)
                    ref_data.append(item)

    print(f"Loaded {len(criteria_data)} evaluation criteria")
    print(f"Loaded {len(answer_data)} answers")
    print(f"Loaded {len(ref_data)} reference answers")

    # Use prefix matching: criteria's prompt is prefix of answer/ref's question
    # (answer/ref's question has fixed "reliable source" requirement at the end)
    criteria_dict = {item['prompt']: item for item in criteria_data}

    # answer_dict: prompt -> list[item], same prompt may have multiple iters
    answer_dict = {}
    for item in answer_data:
        q = item.get('question', '')
        for prompt in criteria_dict:
            if q.startswith(prompt):
                answer_dict.setdefault(prompt, []).append(item)
                break

    ref_dict = {}
    for item in ref_data:
        q = item.get('question', '')
        raw_ref = item.get('prediction', item.get('answer', ''))
        if not extract_answer_content(raw_ref):  # Skip ref with empty answer
            continue
        for prompt in criteria_dict:
            if q.startswith(prompt):
                ref_dict[prompt] = item
                break

    # Find prompts matched by all three parties
    common_prompts = set(criteria_dict.keys()) & set(answer_dict.keys()) & set(ref_dict.keys())

    # Read existing result file, get evaluated (query, iter) pairs
    evaluated_keys = set()
    if os.path.exists(args.output_file):
        print(f"Reading existing result file: {args.output_file}")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            if 'query' in result:
                                evaluated_keys.add((result['query'], result.get('iter')))
                        except json.JSONDecodeError:
                            continue
            print(f"Number of evaluated (document, iter) pairs: {len(evaluated_keys)}")
        except Exception as e:
            print(f"Error reading result file: {e}")

    # Build list of (prompt, answer_item) to evaluate, filter out already evaluated (query, iter) pairs
    # Also skip entries without valid answer content (e.g., model produced no answer, only tool_call)
    skipped_no_answer = 0
    tasks_to_evaluate = []
    for prompt in common_prompts:
        for answer_item in answer_dict[prompt]:
            raw = answer_item.get('prediction', answer_item.get('answer', ''))
            if not extract_answer_content(raw):
                print(f"Skipping invalid answer (iter={answer_item.get('iter')}): {prompt[:50]}...")
                skipped_no_answer += 1
                continue
            key = (prompt, answer_item.get('iter'))
            if key not in evaluated_keys:
                tasks_to_evaluate.append((prompt, answer_item))
    if skipped_no_answer:
        print(f"Skipped {skipped_no_answer} records with no valid answer content")

    total_tasks = sum(len(v) for v in answer_dict.values() if v)
    print(f"Found {len(common_prompts)} matching documents, total {total_tasks} (document, iter) pairs, "
          f"of which {len(tasks_to_evaluate)} need evaluation\n")

    # Sort by (prompt, iter) to ensure stable output order
    tasks_to_evaluate.sort(key=lambda x: (x[0], x[1].get('iter', 0) if isinstance(x[1].get('iter'), int) else 0))

    # Evaluate each (document, iter) pair (document-level concurrency)
    file_lock = threading.Lock()

    def _eval_one(task):
        prompt, answer_item = task
        cur_iter = answer_item.get('iter')
        try:
            result = evaluate_document(criteria_dict[prompt], answer_item, ref_dict[prompt])
            with file_lock:
                with open(args.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            return result
        except Exception as e:
            print(f"Error evaluating document (prompt: {prompt[:50]}..., iter: {cur_iter}): {e}")
            import traceback
            traceback.print_exc()
            return None

    results = []
    print(f"Document-level concurrency: {args.max_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_eval_one, task): task for task in tasks_to_evaluate}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(tasks_to_evaluate), desc="Overall progress"):
            result = future.result()
            if result is not None:
                results.append(result)

    # Generate summary report
    print("\n" + "="*80)
    print("Evaluation Summary Report")
    print("="*80)

    for result in results:
        print(f"\nDocument ID: {result['id']} | iter: {result.get('iter', 'N/A')}")
        print(f"Final score: {result['final_score']:.4f}")
        print(f"Document total score: {result['total_score_a']:.4f}")
        print(f"Reference total score: {result['total_score_b']:.4f}")
        print(f"Dimension scores - Document:")
        for dim, score in result['dimension_scores_a'].items():
            weight = result['dimension_weights'][dim]
            print(f"  - {dim}: {score:.4f} (weight: {weight})")
        print(f"Dimension scores - Reference:")
        for dim, score in result['dimension_scores_b'].items():
            weight = result['dimension_weights'][dim]
            print(f"  - {dim}: {score:.4f} (weight: {weight})")

    print(f"\nAll evaluation results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
