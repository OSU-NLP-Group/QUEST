import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

EvalLLMAddresses = Optional[Union[str, List[str]]]
EvalLLMChatFn = Callable[..., Awaitable[Optional[str]]]

OPENENDED_SYSTEM_PROMPT = (
    "You are an expert evaluator tasked with scoring two documents (both presenting "
    "research findings in response to the user's query) on specific rubric criteria. "
    "Your evaluation must be precise, objective, and based solely on the evidence present "
    "in both documents.\n\n"
    "## Evaluation Framework\n"
    "For each criterion, score both documents on a scale of 0-10 (continuous values). "
    "The score should reflect the quality of performance on that criterion:\n"
    "*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.\n"
    "*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.\n"
    "*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.\n"
    "*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.\n"
    "*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.\n\n"
    "## Evaluation Process\n"
    "1. **Understand the Criterion**: Carefully read and interpret what the rubric is asking for.\n"
    "2. **Search for Evidence**: Systematically review both documents for relevant content that addresses the criterion.\n"
    "3. **Score Each Document**: Evaluate how each document performs against the criterion and assign a score from 0-10.\n"
    "4. **Provide Reasoning**: Explain your evaluation with specific references to both documents.\n\n"
    "## Important Guidelines\n"
    "- Base your evaluation ONLY on what is explicitly present in both documents\n"
    "- Do not make assumptions about implied or missing content\n"
    "- Consider the quality, completeness, and relevance of the evidence in both documents\n"
    "- Be consistent in your evaluation standards across all criteria\n"
    "- Provide specific examples from both documents to support your scores"
)

OPENENDED_USER_PROMPT_TEMPLATE = (
    "## Document A (Content to Evaluate)\n"
    "{document_content}\n\n"
    "## Document B (Reference Content)\n"
    "{ref_content}\n\n"
    "## Original Query\n"
    "{query}\n\n"
    "## Rubric Criterion\n"
    "Title: {rubric_title}\n"
    "Category: {rubric_category}\n"
    "Explanation: {rubric_explanation}\n\n"
    "Evaluate both documents on this criterion and return ONLY a JSON object in this format:\n"
    "```json\n"
    "{{\n"
    '"score_a": 0-10,\n'
    '"score_b": 0-10,\n'
    '"confidence": 0.0-1.0\n'
    "}}\n"
    "```\n"
    "Where:\n"
    "- score_a: The score for Document A (content to evaluate)\n"
    "- score_b: The score for Document B (reference content)\n"
    "Ensure your final answer is wrapped in the JSON code block."
)

OPENENDED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def parse_openended_eval_response(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        return None
    m = OPENENDED_JSON_RE.search(response_text)
    json_str = m.group(1) if m else response_text
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    try:
        score_a = float(result.get("score_a", -1))
        score_b = float(result.get("score_b", -1))
        if not (0 <= score_a <= 10 and 0 <= score_b <= 10):
            return None
        result["score_a"] = score_a
        result["score_b"] = score_b
    except (ValueError, TypeError):
        return None
    return result


async def _evaluate_single_criterion_async(
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    document_content: str,
    ref_content: str,
    query: str,
    criterion_data: Dict[str, Any],
    dimension: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str = "openended",
    max_retries: int = 3,
) -> Dict[str, Any]:
    user_prompt = OPENENDED_USER_PROMPT_TEMPLATE.format(
        document_content=document_content,
        ref_content=ref_content,
        query=query,
        rubric_title=criterion_data.get("criterion", ""),
        rubric_category=dimension,
        rubric_explanation=criterion_data.get("explanation", ""),
    )
    messages = [
        {"role": "system", "content": OPENENDED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_retries):
        response = await llm_chat_fn(
            eval_llm_addresses=eval_llm_addresses,
            messages=messages,
            model=eval_llm_model,
            profile=profile,
            temperature=0.6,
            max_tokens=4096,
        )
        result = parse_openended_eval_response(response or "")
        if result is not None:
            result["criterion_name"] = criterion_data.get("criterion", "")
            result["category"] = dimension
            result["weight"] = float(criterion_data.get("weight", 1.0))
            return result

    return {
        "score_a": 0.0,
        "score_b": 0.0,
        "reason": "evaluation failed after retries",
        "criterion_name": criterion_data.get("criterion", ""),
        "category": dimension,
        "weight": float(criterion_data.get("weight", 1.0)),
    }


async def compute_score_openended(
    extracted_answer: str,
    ground_truth: Dict[str, Any],
    question: str,
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str = "openended",
) -> Dict[str, Any]:
    criterions = ground_truth.get("criterions", {})
    dimension_weights = ground_truth.get("dimension_weight", {})
    ref_content = ground_truth.get("ref_answer", "")

    if not criterions or not dimension_weights:
        return {"score": 0.0, "error": "missing_criterions_or_weights"}

    tasks = []
    task_meta = []
    for dimension, criteria_list in criterions.items():
        if not isinstance(criteria_list, list):
            continue
        for crit in criteria_list:
            crit_dict = dict(crit) if not isinstance(crit, dict) else crit
            tasks.append(
                _evaluate_single_criterion_async(
                    eval_llm_addresses=eval_llm_addresses,
                    eval_llm_model=eval_llm_model,
                    document_content=extracted_answer,
                    ref_content=ref_content,
                    query=question,
                    criterion_data=crit_dict,
                    dimension=dimension,
                    llm_chat_fn=llm_chat_fn,
                    profile=profile,
                )
            )
            task_meta.append(dimension)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_evaluations: Dict[str, list] = defaultdict(list)
    for dim, res in zip(task_meta, results):
        if isinstance(res, Exception):
            res = {"score_a": 0.0, "score_b": 0.0, "weight": 1.0, "error": str(res)}
        all_evaluations[dim].append(res)

    dimension_scores_a: Dict[str, float] = {}
    dimension_scores_b: Dict[str, float] = {}

    for dimension, evals in all_evaluations.items():
        weighted_sum_a = sum(float(e.get("score_a", 0)) * float(e.get("weight", 1)) for e in evals)
        weighted_sum_b = sum(float(e.get("score_b", 0)) * float(e.get("weight", 1)) for e in evals)
        total_weight = sum(float(e.get("weight", 1)) for e in evals)
        dimension_scores_a[dimension] = weighted_sum_a / total_weight if total_weight > 0 else 0.0
        dimension_scores_b[dimension] = weighted_sum_b / total_weight if total_weight > 0 else 0.0

    total_score_a = sum(dimension_scores_a.get(dim, 0) * w for dim, w in dimension_weights.items())
    total_score_b = sum(dimension_scores_b.get(dim, 0) * w for dim, w in dimension_weights.items())

    if total_score_a + total_score_b > 0:
        raw_score = total_score_a / (total_score_a + total_score_b)
    else:
        raw_score = 0.0

    if raw_score >= 0.5:
        final_score = 1.0
    elif raw_score >= 0.475:
        final_score = 0.75
    elif raw_score >= 0.45:
        final_score = 0.5
    elif raw_score >= 0.425:
        final_score = 0.25
    else:
        final_score = 0.0

    return {
        "score": final_score,
        "final_score": final_score,
        "raw_score": raw_score,
        "acc": final_score,
        "total_score_a": total_score_a,
        "total_score_b": total_score_b,
        "dimension_scores_a": dimension_scores_a,
        "dimension_scores_b": dimension_scores_b,
        "dimension_weights": dimension_weights,
        "eval_type": "openended_criteria",
    }
