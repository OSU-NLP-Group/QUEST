"""Presentation & Organization criterion - 10 binary checklist questions."""

# The 10 presentation quality questions (checklist-based, binary 0/1)
PRESENTATION_QUESTIONS = {
    "p1": "Does the report present a clear, coherent, and logically ordered structure so that the organization is easy to follow and directly addresses the research question?",
    "p2": "Does the report contain zero grammar and spelling errors?",
    "p3": "Does every entry in the reference list correspond to at least one in-text citation?",
    "p4": "Does every in-text citation have a corresponding entry in the reference list?",
    "p5": "Is there exactly one \"References\" (or \"Bibliography\" / \"Sources\") section, and are its entries sorted according to a single, consistent scheme?",
    "p6": "Is a single, consistent citation style used throughout the entire document?",
    "p7": "Are all in-text citations placed logically at the end of a clause or sentence, without interrupting grammatical flow?",
    "p8": "If the report includes figures or tables, does each one contain complete data or a valid visual element? (If none are included, the report automatically passes this test.)",
    "p9": "Is the formatting correct and consistent? For example: (a) If delivered in Markdown, are proper heading levels (#, ##, etc.) used instead of plain text for section titles; (b) if Markdown tables are included, is their syntax valid and renderable?",
    "p10": "If the citations are numbered, are there no skipped numbers (e.g., [23],[25],[26] with [24] missing) and no duplicates (i.e. two different sources assigned the same number, or one source assigned multiple numbers)?"
}

# JSON schema for presentation evaluation
PRESENTATION_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "object",
            "patternProperties": {
                "^p[0-9]+$": {
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


def create_presentation_prompt(query: str, report_content: str, current_date: str = None) -> tuple:
    """
    Create system and user prompts for presentation quality evaluation.
    
    Returns:
        (system_prompt, user_prompt)
    """
    if not current_date:
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

    system_prompt = f"""You are an expert evaluator assessing research reports for presentation quality and formatting standards. Your sole task is to determine if the report meets specific presentation criteria with binary scoring (0 or 1).

EVALUATION CRITERIA:
- Score 1: The report fully satisfies the presentation criterion with no issues.
- Score 0: The report fails to meet the presentation criterion or has any issues that prevent it from passing.

INSTRUCTIONS:
- Read the research report carefully to assess presentation quality, formatting, and structural elements.
- For each presentation criterion, determine if the report fully meets the standard.
- Provide a binary score (0 or 1) for each criterion.
- Provide a clear justification for your score, referencing specific elements in the report.

IMPORTANT GUIDELINES:
- Strict Binary Assessment: A score of 1 requires complete satisfaction of the criterion. Any failure to meet the standard results in a score of 0.
- Be Precise: Reference specific sections, examples, or issues in your justifications.
- No Partial Credit: Unlike subjective quality assessments, presentation standards are met or not met—there is no middle ground.

Respond with a JSON object containing your evaluations for each presentation criterion."""

    # Format checklist items for the prompt
    checklist_items = []
    for item_id, item_text in PRESENTATION_QUESTIONS.items():
        checklist_items.append(f"- {item_id}: {item_text}")
    
    checklist_section = "\n".join(checklist_items)
    
    user_prompt = f"""ORIGINAL RESEARCH QUERY:
{query}

RESEARCH REPORT TO EVALUATE:
{report_content}

PRESENTATION CRITERIA TO EVALUATE:
{checklist_section}

Please evaluate this report on each presentation criterion and provide your assessment in JSON format."""
    
    return system_prompt, user_prompt

