# Example evaluation script for a specific task
# This demonstrates how to write custom evaluation logic
#
# File naming: <task_id>.py
# Required function: evaluate_answer(solution_str, ground_truth, task_id, **kwargs) -> Dict

import re
import sys
import os
from typing import Dict, Any, List, Optional

# Add the deepresearch package to path
_deepresearch_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_examples_dir = os.path.dirname(_deepresearch_dir)
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

# Import from deepresearch package
from deepresearch import VerificationNode, AggregationStrategy, Evaluator


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer> tags."""
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_urls_from_response(text: str) -> List[str]:
    """Extract all URLs mentioned in the response."""
    # Match markdown links [text](url) and plain URLs
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    plain_url_pattern = r'https?://[^\s\)\]<>]+'

    urls = set()

    # Extract from markdown links
    for match in re.finditer(markdown_pattern, text):
        urls.add(match.group(2))

    # Extract plain URLs
    for match in re.finditer(plain_url_pattern, text):
        urls.add(match.group(0))

    return list(urls)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return ' '.join(text.lower().split())


def evaluate_answer(
    solution_str: str,
    ground_truth: Dict[str, Any],
    task_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Custom evaluation function for this task.

    This example demonstrates:
    1. Using the Evaluator to build a verification tree
    2. Adding custom checks (format, URL extraction, answer matching)
    3. Using sequential/parallel strategies

    Args:
        solution_str: Full model response
        ground_truth: Expected answers and criteria
            Expected format:
            {
                "target": ["acceptable_answer_1", "answer_2"],
                "required_sources": ["https://example.com"],  # Optional
                "min_tool_calls": 2,  # Optional
            }
        task_id: Task identifier
        **kwargs: Additional arguments

    Returns:
        Dict with final_score and verification_tree
    """
    # Initialize evaluator (skip_llm_init=True since we only use custom nodes)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=task_id,
        task_description=ground_truth.get("task_description", f"Evaluate {task_id}"),
        strategy=AggregationStrategy.PARALLEL,
        skip_llm_init=True  # No LLM calls needed for this eval script
    )

    # Extract answer
    answer = extract_answer(solution_str)

    # Get expected answers
    expected_answers = ground_truth.get("target", [])
    if isinstance(expected_answers, str):
        expected_answers = [expected_answers]

    # ============== Format Check (Sequential - must pass before content check) ==============
    format_group = evaluator.add_sequential("format", "Format validation", critical=True)

    # Check think tags
    has_think = "<think>" in solution_str and "</think>" in solution_str
    evaluator.add_custom_node(
        result=has_think,
        id="has_think_tags",
        desc="Response contains thinking process",
        parent=format_group,
        critical=True
    )

    # Check answer tag exists
    has_answer = answer is not None
    evaluator.add_custom_node(
        result=has_answer,
        id="has_answer_tag",
        desc="Response contains answer",
        parent=format_group,
        critical=True
    )

    # ============== Content Check (Parallel - all evaluated, average score) ==============
    content_group = evaluator.add_parallel("content", "Content validation", critical=False)

    # Check answer correctness
    if answer:
        normalized_answer = normalize_text(answer)

        # Exact match check
        exact_match = any(
            normalize_text(exp) == normalized_answer
            for exp in expected_answers
        )

        # Partial match check (answer contains expected or vice versa)
        partial_match = any(
            normalize_text(exp) in normalized_answer or
            normalized_answer in normalize_text(exp)
            for exp in expected_answers
        )

        evaluator.add_custom_node(
            result=exact_match,
            id="exact_match",
            desc="Answer exactly matches expected",
            parent=content_group,
            critical=False  # Non-critical, contributes to average
        )

        if not exact_match:
            evaluator.add_custom_node(
                result=partial_match,
                id="partial_match",
                desc="Answer partially matches expected",
                parent=content_group,
                critical=False
            )

    # ============== Source Check (Optional - only if required_sources specified) ==============
    required_sources = ground_truth.get("required_sources", [])
    if required_sources:
        source_group = evaluator.add_parallel("sources", "Source validation", critical=False)

        extracted_urls = extract_urls_from_response(solution_str)

        for i, required_url in enumerate(required_sources):
            # Check if required source was used
            source_used = any(
                required_url in url or url in required_url
                for url in extracted_urls
            )

            evaluator.add_custom_node(
                result=source_used,
                id=f"source_{i}",
                desc=f"Used required source: {required_url[:50]}",
                parent=source_group,
                critical=False
            )

    # ============== Tool Usage Check (Optional) ==============
    min_tool_calls = ground_truth.get("min_tool_calls", 0)
    if min_tool_calls > 0:
        tool_group = evaluator.add_parallel("tools", "Tool usage validation", critical=False)

        # Count tool calls
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
        num_tool_calls = len(tool_calls)

        evaluator.add_custom_node(
            result=num_tool_calls >= min_tool_calls,
            id="min_tool_calls",
            desc=f"Used at least {min_tool_calls} tool calls",
            parent=tool_group,
            critical=False
        )

    # Compute and return summary
    return evaluator.get_summary()


# For testing
if __name__ == "__main__":
    # Example test
    test_solution = """
<think>I need to find information about the capital of France.</think>
<tool_call>{"name": "search", "arguments": {"query": "capital of France"}}</tool_call>

<tool_response>
Search results for 'capital of France':
1. [Paris - Wikipedia](https://en.wikipedia.org/wiki/Paris)
Paris is the capital of France...
</tool_response>

<think>Based on my search, Paris is the capital of France.</think>
<answer>Paris is the capital of France.</answer>
"""

    test_ground_truth = {
        "task_description": "Find the capital of France",
        "target": ["Paris", "Paris is the capital of France"],
        "required_sources": ["wikipedia.org"],
        "min_tool_calls": 1
    }

    result = evaluate_answer(test_solution, test_ground_truth, "example_task")
    print(f"Final Score: {result['final_score']}")
    print(f"Verification Tree: {result['verification_tree']}")
