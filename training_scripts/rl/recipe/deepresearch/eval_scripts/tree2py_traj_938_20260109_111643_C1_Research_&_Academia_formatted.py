import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "usnews_2025_csphd_ra_stipend"
TASK_DESCRIPTION = (
    "What is the annual 12-month base stipend amount for doctoral research assistant positions at the #1 ranked "
    "computer science PhD program according to the U.S. News 2025 Best Computer Science Schools rankings for the "
    "2025-2026 academic year?"
)


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    """
    Structured extraction of the key facts from the agent's answer.

    Notes:
    - Extract exactly what the answer explicitly states; do not infer.
    - URLs must be explicit in the answer (markdown links allowed).
    """
    # Program identification (US News #1 program)
    program_name: Optional[str] = None
    ranking_source_urls: List[str] = Field(default_factory=list)

    # Stipend specifics (for 2025-2026, RA, 12-month base, annual amount)
    stipend_amount_annual_12mo_base: Optional[str] = None
    academic_year: Optional[str] = None
    assistantship_type: Optional[str] = None
    rate_basis: Optional[str] = None
    stipend_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                            #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_info() -> str:
    return """
Extract the specific information from the answer that is needed to judge the question.

Required fields to extract:
1) program_name: The formal name of the computer science PhD program or institution that the answer claims is ranked #1 in the U.S. News 2025 Best Computer Science Schools rankings. Return null if not stated.
2) ranking_source_urls: A list of all URLs the answer cites to support the ranking claim (e.g., U.S. News ranking page). If none are provided, return an empty list.
3) stipend_amount_annual_12mo_base: The single annual 12-month base stipend amount the answer claims for doctoral research assistant (RA) positions. Keep any currency symbols and formatting as provided (e.g., "$45,000"). Return null if not provided.
4) academic_year: The academic year associated with the stipend, as quoted (ideally "2025-2026", "AY 2025–2026", etc.). Return null if not provided.
5) assistantship_type: The role for which the stipend applies (e.g., "RA", "Research Assistant", "Graduate Student Researcher/GSR"). Return null if not provided.
6) rate_basis: The basis for the rate (e.g., "12-month", "annual 12-month base", "12-month base", "annual"). Return null if not provided.
7) stipend_source_urls: A list of all URLs the answer cites as evidence for the stipend information (e.g., official program/graduate school pages). If none are provided, return an empty list.

General rules:
- Extract only what is explicitly present in the answer. Do not fabricate.
- For URLs, extract the actual URL targets (including markdown link targets).
- If any field is not present, set it to null (or empty list for URL lists).
"""


# --------------------------------------------------------------------------- #
# Verification helper to construct claims                                     #
# --------------------------------------------------------------------------- #
def _safe(value: Optional[str]) -> str:
    return value if value is not None else ""


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for:
    - Correctly identifying the #1 CS PhD program per U.S. News 2025 Best Computer Science Schools rankings
    - Providing the correct annual 12-month base RA stipend for the 2025-2026 academic year
    """
    # Initialize evaluator with a simple root; we'll add our task node under it
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # 1) Extract structured information from the answer
    extracted: AnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_answer_info(),
        template_class=AnswerExtraction,
        extraction_name="answer_structured_extraction",
    )

    # 2) Build verification tree according to rubric JSON

    # Top-level task node (critical, sequential)
    # "Answer_Meets_All_Constraints"
    task_node = evaluator.add_sequential(
        id="Answer_Meets_All_Constraints",
        desc=(
            "Answer identifies the correct #1-ranked CS PhD program (per U.S. News 2025) and provides the correct "
            "annual 12-month base RA stipend amount for the 2025-2026 academic year."
        ),
        parent=root,
        critical=True,
    )

    # 2.1 Correct program identification (leaf, critical)
    node_prog = evaluator.add_leaf(
        id="Correct_Program_Identification",
        desc="Identifies the computer science PhD program that is ranked #1 in the U.S. News 2025 Best Computer Science Schools rankings.",
        parent=task_node,
        critical=True,
    )

    program_name = _safe(extracted.program_name)
    ranking_urls = extracted.ranking_source_urls or []

    # Claim: Program X is ranked #1 (ties permitted) in U.S. News 2025 Best Computer Science Schools.
    claim_program = (
        f"According to the U.S. News 2025 Best Computer Science Schools graduate rankings, "
        f"the #1-ranked program includes '{program_name}' (ties for #1 are acceptable)."
    )

    await evaluator.verify(
        claim=claim_program,
        node=node_prog,
        sources=ranking_urls,  # Must be supported by the provided ranking source(s)
        additional_instruction=(
            "Focus on verifying the 2025 U.S. News Best Computer Science Schools (graduate) rankings. "
            "It must clearly show that the specified program is ranked #1 (ties for #1 acceptable). "
            "If the provided URL(s) are not an official or clearly labeled 2025 U.S. News ranking page "
            "or do not show the 2025 ranking explicitly, judge as not supported."
        ),
    )

    # 2.2 Stipend definition and value checks (parallel, critical)
    stipend_node = evaluator.add_parallel(
        id="Stipend_Matches_Required_Definition",
        desc=(
            "The stipend information provided matches the required stipend definition in the question "
            "(year, role, rate basis), and the amount corresponds to that definition for the identified program."
        ),
        parent=task_node,
        critical=True,
    )

    stipend_urls = extracted.stipend_source_urls or []
    stipend_amount = _safe(extracted.stipend_amount_annual_12mo_base)
    academic_year = _safe(extracted.academic_year)
    assistantship_type = _safe(extracted.assistantship_type)
    rate_basis = _safe(extracted.rate_basis)

    # Create leaves for stipend constraints
    leaf_year = evaluator.add_leaf(
        id="Correct_Academic_Year_2025_2026",
        desc="Stipend is explicitly for the 2025-2026 academic year (not another year).",
        parent=stipend_node,
        critical=True,
    )

    leaf_role = evaluator.add_leaf(
        id="Correct_Assistantship_Type_RA",
        desc="Stipend is for doctoral research assistant (RA) positions (not TA-only, fellowship-only, etc.).",
        parent=stipend_node,
        critical=True,
    )

    leaf_basis = evaluator.add_leaf(
        id="Correct_Rate_Basis_Annual_12_Month_Base",
        desc="Stipend is the annual 12-month base rate (not 9-month, not a monthly figure without annualization, not including discretionary/top-up components unless explicitly part of base).",
        parent=stipend_node,
        critical=True,
    )

    leaf_amount = evaluator.add_leaf(
        id="Correct_Stipend_Amount_Value",
        desc="Provides a single annual monetary amount and it matches the official 2025-2026 12-month base RA stipend rate for the identified #1 program.",
        parent=stipend_node,
        critical=True,
    )

    # Prepare claims and run stipend verifications (in parallel when possible)
    stipend_claims: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Year check
    claim_year = (
        f"The stipend described for '{program_name}' is explicitly for the 2025–2026 academic year (AY 2025–2026). "
        f"The answer cites: '{academic_year}'."
    )
    stipend_claims.append((
        claim_year,
        stipend_urls,
        leaf_year,
        "Look for explicit mention of '2025–2026', 'AY 2025–2026', or equivalent on the provided source page(s). "
        "If the page shows any other academic year (e.g., 2024–2025, 2026–2027) or is ambiguous, judge as not supported."
    ))

    # Role (RA) check
    claim_role = (
        f"The stipend described applies to doctoral research assistant (RA) positions at '{program_name}'. "
        f"The answer cites the role as: '{assistantship_type}'. "
        "Synonyms like 'Research Assistant', 'RA', or 'Graduate Student Researcher (GSR)' count as RA."
    )
    stipend_claims.append((
        claim_role,
        stipend_urls,
        leaf_role,
        "Verify that the cited stipend is specifically for research assistant roles (doctoral RA/GSR). "
        "Do not accept TA-only or fellowship-only stipends unless the page explicitly states it is the base RA stipend."
    ))

    # Rate basis (12-month base) check
    claim_basis = (
        f"The stipend described for '{program_name}' corresponds to the base 12-month annual rate (not 9-month, not monthly-only). "
        f"The answer cites the basis as: '{rate_basis}'."
    )
    stipend_claims.append((
        claim_basis,
        stipend_urls,
        leaf_basis,
        "Confirm that the figure is the base 12-month annual stipend amount for RA positions. "
        "Reject if it is a 9-month figure, a monthly figure without annualization, or includes discretionary/top-up amounts "
        "that are not part of the base rate."
    ))

    # Amount value check
    claim_amount = (
        f"For '{program_name}', the official 2025–2026 base 12-month RA stipend is {stipend_amount} (a single annual amount)."
    )
    stipend_claims.append((
        claim_amount,
        stipend_urls,
        leaf_amount,
        "Verify that the official page(s) show exactly one base 12-month annual RA stipend amount matching the stated figure. "
        "Minor formatting differences (currency symbols, commas) are acceptable. "
        "If the source shows multiple levels, ranges, or department-dependent amounts without a single base figure, judge as not supported."
    ))

    # Ensure the program identification check is executed before stipend checks
    # (so that sequential precondition logic can skip stipend checks if program fails)
    # Already awaited above.

    # Run stipend checks (parallelized)
    await evaluator.batch_verify(stipend_claims)

    # Return the evaluation summary
    return evaluator.get_summary()