"""Citation Association criterion - pointwise/additive evaluation."""

# Rubric for citation association evaluation (pointwise: count errors)
CITATION_RUBRIC = """
| Score | Issue Count | Description |
|-------|-------------|-------------|
| 100 | 0 | Perfect - All major claims/facts have citations; all are fully traceable |
| 90 | 1-2 | Excellent - Very minor citation omissions |
| 80 | 3-4 | Very Good - Few minor citation omissions |
| 70 | 5-6 | Good - Some citation omissions |
| 60 | 7-8 | Above Average - Several citation omissions |
| 50 | 9-10 | Average - Multiple citation omissions |
| 40 | 11-12 | Below Average - Many citation omissions |
| 30 | 13-14 | Poor - Extensive citation omissions |
| 20 | 15-17 | Very Poor - Pervasive citation omissions |
| 10 | 18+ | Unacceptable - Most claims lack citations; claims are untraceable |

Focus on:
- Major claims and factual statements that lack citations
- Traceability of claims to their sources
- Whether citations adequately support the claims being made

What counts as a "major claim/fact":
- Specific statistics, numbers, percentages, or quantitative data
- Direct quotes or paraphrased statements from sources
- Expert opinions or findings from research
- Historical events, dates, or specific factual assertions
- Causal relationships or correlations stated as fact

What does NOT count as requiring citation:
- General knowledge or common facts
- The author's own analysis or synthesis
- Logical inferences drawn from cited information
- Transitional or organizational statements

Remarks:
1. A single citation can support multiple related claims if they come from the same source.
2. Focus on WHETHER claims are cited, not on citation formatting (that's covered in Presentation).
3. Factual accuracy of the cited information is a separate criterion and should NOT be considered here.
"""

# JSON schema for citation evaluation
CITATION_EVALUATION_SCHEMA = {
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


def create_citation_prompt(query: str, report_content: str, current_date: str = None) -> tuple:
    """
    Create system and user prompts for citation association evaluation.
    
    Returns:
        (system_prompt, user_prompt)
    """
    if not current_date:
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

    system_prompt = f"""You are an expert evaluator assessing research reports based on citation association. Your task is to identify major claims and factual statements that lack proper citations, then determine a score based on the number of issues found.

IMPORTANT: You need to be very careful and detail-oriented. Read the report sentence by sentence and identify concrete issues.

Be critical and thorough in your evaluation - do not overlook problems or give inflated scores.

You should find as many substantive issues RELEVANT to the criteria as possible. However, these should be real issues that CLEARLY dissatisfy the criteria - NOT minor nitpicks or subjective preferences.

For your reference:
- Today's date is: {current_date}

Grading Scale for Citation Association:
{CITATION_RUBRIC}

Provide your assessment in the specified JSON format with:
- specific_issues: List of specific uncited claims/facts with exact quotes or locations in the text where applicable (e.g., "In section X, the claim that '...' lacks citation", "The statistic '...' is not cited")
- total_issues: Total count of issues you identified (should match the length of specific_issues array)
- score: Integer from 10-100 based on the rubric and the issues you identified
- justification: Detailed explanation of your assessment based on the specific issues found (2-3 sentences)"""

    user_prompt = f"""TASK: {query}

REPORT TO EVALUATE:
{report_content}

Please evaluate this report on the criterion of "Citation Association" and provide your assessment in JSON format."""
    
    return system_prompt, user_prompt

