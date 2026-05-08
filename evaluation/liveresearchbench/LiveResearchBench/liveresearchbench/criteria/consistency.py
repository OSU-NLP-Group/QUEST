"""Factual & Logical Consistency criterion - pointwise/additive evaluation."""

# Rubric for consistency evaluation (pointwise: count errors)
CONSISTENCY_RUBRIC = """
| Score | Issue Count | Description |
|-------|-------------|-------------|
| 100 | 0 | Perfect - No contradictions or inconsistencies detected |
| 90 | 1-2 | Excellent - Very minor inconsistencies |
| 80 | 3-4 | Very Good - Few minor inconsistencies |
| 70 | 5-6 | Good - Some inconsistencies |
| 60 | 7-8 | Above Average - Several inconsistencies |
| 50 | 9-10 | Average - Multiple inconsistencies |
| 40 | 11-12 | Below Average - Many inconsistencies |
| 30 | 13-14 | Poor - Extensive inconsistencies |
| 20 | 15-17 | Very Poor - Pervasive inconsistencies |
| 10 | 18+ | Unacceptable - Overwhelming inconsistencies that undermine validity |

Focus on:
- Logical consistency  
- Factual contradictions (facts, claims, numbers, dates, names, etc.)

Remarks:
1. Factual accuracy (e.g., whether the report is free of factual errors) is a separate criterion and should NOT be considered here.
2. The same source can be used to cite multiple claims. This does NOT constitute a contradiction.
"""

# JSON schema for consistency evaluation
CONSISTENCY_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "specific_issues": {
            "type": "array",
            "items": {"type": "string"}
        },
        "total_issues": {"type": "integer", "minimum": 0},
        "score": {"type": "integer", "minimum": 10, "maximum": 100},
        "justification": {"type": "string"}
    },
    "required": ["specific_issues", "total_issues", "score", "justification"],
    "additionalProperties": False
}


def create_consistency_prompt(query: str, report_content: str, current_date: str = None) -> tuple:
    """
    Create system and user prompts for consistency evaluation.
    
    Returns:
        (system_prompt, user_prompt)
    """
    if not current_date:
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

    system_prompt = f"""You are an expert evaluator assessing research reports based on logical and factual consistency. Your task is to identify concrete contradictions and inconsistencies in the report, then determine a score based on the number of issues found.

IMPORTANT: You need to be very careful and detail-oriented. Read the report sentence by sentence and identify concrete issues.

Be critical and thorough in your evaluation - do not overlook problems or give inflated scores.

You should find as many substantive issues RELEVANT to the criteria as possible. However, these should be real issues that CLEARLY dissatisfy the criteria - NOT minor nitpicks or subjective preferences.

For your reference:
- Today's date is: {current_date}
- When evaluating temporal consistency, consider what timeframes would be logical given this current date

Grading Scale for Factual & Logical Consistency:
{CONSISTENCY_RUBRIC}

Provide your assessment in the specified JSON format with:
- specific_issues: List of specific contradictions/inconsistencies found with exact quotes or locations in the text where applicable (e.g., "In section X...", "The claim that '...' contradicts...")
- total_issues: Total count of issues you identified (should match the length of specific_issues array)
- score: Integer from 10-100 based on the rubric and the issues you identified
- justification: Detailed explanation of your assessment based on the specific issues found (2-3 sentences)"""

    user_prompt = f"""TASK: {query}

REPORT TO EVALUATE:
{report_content}

Please evaluate this report on the criterion of "Factual & Logical Consistency" and provide your assessment in JSON format."""
    
    return system_prompt, user_prompt

