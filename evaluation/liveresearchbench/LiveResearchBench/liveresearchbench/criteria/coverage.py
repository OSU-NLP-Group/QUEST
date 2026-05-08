"""Coverage & Comprehensiveness criterion - checklist-based evaluation."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# JSON schema for coverage evaluation
COVERAGE_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "object",
            "patternProperties": {
                "^q[0-9]+$": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 1},
                        "justification": {"type": "string"}
                    },
                    "required": ["score", "justification"],
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False
}


def create_coverage_prompt(query: str, report_content: str, checklists: List[str], current_date: str = None) -> tuple:
    """
    Create system and user prompts for coverage evaluation.
    
    Args:
        query: Original research query
        report_content: The research report to evaluate
        checklists: List of checklist items (strings)
        current_date: Current date for temporal context
        
    Returns:
        (system_prompt, user_prompt)
    """
    if not current_date:
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

    system_prompt = f"""You are an expert evaluator assessing research reports against specific checklist criteria derived from an original research query. Your sole task is to determine if the report fully and completely delivers the specific data and information requested for each item.

EVALUATION CRITERIA:
- Score 1: The report fully and completely provides all specific data and information required by the checklist item.
- Score 0: The report fails to provide the specific data or information, or provides an incomplete response.

INSTRUCTIONS:
- Read the original research query to understand the exact data being requested.
- Read the research report to find the data that directly and completely answers the query.
- For each checklist item, determine if the report delivers all required components of the answer.
- Provide a binary score (0 or 1).
- Provide a clear justification for your score, referencing the completeness or incompleteness of the provided data.

IMPORTANT GUIDELINES:
- Strict Requirement for Completeness: A score of 1 requires 100% fulfillment of the checklist item. If any part of the requested information for that item is missing, incorrect, or incomplete, the score must be 0.
- Data vs. Methodology: Your evaluation must distinguish between a report that provides the answer versus one that describes a process to find the answer. A methodology, plan, or description of how the data could be found is not a substitute for the data itself and must be scored 0.
- No Credit for Placeholders: A report that acknowledges a checklist item but explicitly states the information is missing, unavailable, or not provided (e.g., "Information not provided," "Data not found") must receive a score of 0 for that item.
- Focus on Delivery: Base your evaluation strictly on the final data delivered in the report. Do not score based on what the report promises, implies, or outlines as its goal.

Respond with a JSON object containing your evaluations for each checklist item."""

    # Format checklist items
    checklist_items = []
    for i, checklist_text in enumerate(checklists, 1):
        checklist_items.append(f"- q{i}: {checklist_text}")
    
    checklist_section = "\n".join(checklist_items)
    
    user_prompt = f"""ORIGINAL RESEARCH QUERY:
{query}

RESEARCH REPORT TO EVALUATE:
{report_content}

CHECKLIST ITEMS TO VERIFY:
{checklist_section}

Please evaluate this report against each checklist item and provide your assessment in JSON format."""
    
    return system_prompt, user_prompt

