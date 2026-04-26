import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "p4_hc_opportunities_2025"
TASK_DESCRIPTION = """You are an experienced college football coordinator evaluating open head coaching opportunities as of November 30, 2025. Identify all Power 4 head coaching position(s) that meet ALL of the following criteria:

- Ranked in the top 3 among open Power 4 coaching jobs according to The Athletic's November 29, 2025 rankings
- Located in either the SEC or Big Ten conference
- Program has an estimated valuation of at least $1 billion according to The Athletic
- Expected head coach salary of at least $10 million annually (based on program tier and comparable positions)
- Job grade of A or A- according to The Athletic's evaluation

For each qualifying position, provide the school name, its ranking, conference affiliation, program valuation, job grade, and supporting reference URL(s).
"""

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    """One qualifying head coaching position, as extracted from the answer."""
    school_name: Optional[str] = None
    ranking: Optional[str] = None  # e.g., "No. 2", "2", "#2"
    conference: Optional[str] = None  # e.g., "SEC", "Big Ten"
    valuation: Optional[str] = None  # e.g., "$1.2B", "$1 billion", "USD 1,100 million"
    job_grade: Optional[str] = None  # e.g., "A", "A-"
    expected_salary: Optional[str] = None  # e.g., "$10M+", "$11 million"
    vacancy_status_statement: Optional[str] = None  # e.g., "open as of Nov 30, 2025"
    urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    """All positions the answer claims qualify under the stated criteria."""
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    From the provided answer, extract all head coaching positions that the answer claims meet ALL of the stated criteria. For each listed position, extract the following fields exactly as stated in the answer:
    - school_name: The school/program name (string)
    - ranking: The ranking number (string as presented; e.g., 'No. 2', '2', '#2'). If missing, null.
    - conference: The conference name (string; e.g., 'SEC' or 'Big Ten'). If missing, null.
    - valuation: The program valuation figure (string, as presented; do not convert to numeric). If missing, null.
    - job_grade: The job grade (string as presented; e.g., 'A', 'A-'). If missing, null.
    - expected_salary: The expected head coach salary estimate (string; e.g., '$10M', '$11 million', '>= $10M'). If missing, null.
    - vacancy_status_statement: The answer's statement indicating the job was an open vacancy as of Nov 30, 2025 (string; can be any phrasing). If missing, null.
    - urls: An array of all supporting reference URLs given specifically for this position. Extract actual URLs in any reasonable format. If none are provided for this position, return an empty list.

    Only include positions the answer claims qualify; do not invent or add positions beyond the answer.
    If any field is not explicitly stated in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def collect_all_urls(extraction: PositionsExtraction) -> List[str]:
    urls: List[str] = []
    for pos in extraction.positions:
        urls.extend(pos.urls)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def has_athletic_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and ("theathletic.com" in u.lower()) for u in urls)


def has_non_athletic_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and ("theathletic.com" not in u.lower()) for u in urls)


def grade_is_a_or_aminus(grade: Optional[str]) -> bool:
    if not grade:
        return False
    g = grade.strip().upper()
    return g in {"A", "A-"}


def salary_mentions_10m_or_more(salary_str: Optional[str]) -> bool:
    if not salary_str:
        return False
    s = salary_str.lower()
    # Heuristic checks that commonly appear in answers
    return ("$10m" in s) or ("$10 m" in s) or ("10 million" in s) or (">= $10" in s) or ("at least $10" in s) or ("$11" in s) or ("$12" in s)


def valuation_mentions_1b_or_more(val_str: Optional[str]) -> bool:
    if not val_str:
        return False
    s = val_str.lower()
    return ("$1b" in s) or ("$1 b" in s) or ("1 billion" in s) or ("billion" in s) or ("1,000 million" in s) or ("1000 million" in s)


def conference_is_sec_or_bigten(conf_str: Optional[str]) -> bool:
    if not conf_str:
        return False
    s = conf_str.strip().lower()
    return s in {"sec", "big ten", "big-ten", "bigten"}


# --------------------------------------------------------------------------- #
# Verification Functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree for a single position.
    Uses a sequential node to gate later checks on earlier failures.
    """
    pos_node = evaluator.add_sequential(
        id=f"position_{idx}",
        desc=f"Position #{idx + 1}: {position.school_name or 'Unknown'} – full criteria verification",
        parent=parent_node,
        critical=True,
    )

    # 1) School name provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(position.school_name and position.school_name.strip()),
        id=f"position_{idx}_school_provided",
        desc="School/program name is provided",
        parent=pos_node,
        critical=True,
    )

    # 2) Top-3 Athletic ranking (parallel: number provided + supported by sources)
    top3_node = evaluator.add_parallel(
        id=f"position_{idx}_top3_main",
        desc="Top-3 Athletic ranking verification",
        parent=pos_node,
        critical=True,
    )

    # 2.1) Ranking number provided (critical existence)
    evaluator.add_custom_node(
        result=bool(position.ranking and position.ranking.strip()),
        id=f"position_{idx}_ranking_provided",
        desc="Ranking number is provided in the answer",
        parent=top3_node,
        critical=True,
    )

    # 2.2) Ranked in Athletic top-3 (verify via provided URLs)
    top3_leaf = evaluator.add_leaf(
        id=f"position_{idx}_rank_top3_athletic",
        desc="Ranked in The Athletic's Nov 29, 2025 top-3 open Power 4 coaching jobs",
        parent=top3_node,
        critical=True,
    )
    ranking_claim = (
        f"The head coaching job at {position.school_name or 'the school'} is ranked in the top 3 among open Power 4 "
        f"coaching jobs according to The Athletic’s November 29, 2025 rankings."
    )
    await evaluator.verify(
        claim=ranking_claim,
        node=top3_leaf,
        sources=position.urls,
        additional_instruction=(
            "Use the provided URLs (prefer The Athletic when available) to confirm the job appears in the top three of "
            "the November 29, 2025 list. Minor variations in formatting or wording are acceptable; focus on whether "
            "the job is explicitly listed in the top three."
        ),
    )

    # 3) Conference SEC or Big Ten (parallel: field provided + value is SEC/Big Ten)
    conf_node = evaluator.add_parallel(
        id=f"position_{idx}_conference_main",
        desc="Conference verification",
        parent=pos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(position.conference and position.conference.strip()),
        id=f"position_{idx}_conference_provided",
        desc="Conference field is provided",
        parent=conf_node,
        critical=True,
    )
    conf_leaf = evaluator.add_leaf(
        id=f"position_{idx}_conference_sec_or_bigten",
        desc="Conference is stated and is either SEC or Big Ten",
        parent=conf_node,
        critical=True,
    )
    conf_claim = (
        f"The program competes in the {position.conference or 'stated'} conference, and that conference is either SEC or Big Ten."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=position.urls,
        additional_instruction=(
            "Accept if the provided conference value is SEC or Big Ten (case-insensitive). Use sources when present; "
            "otherwise rely on the answer text."
        ),
    )

    # 4) Valuation ≥ $1B and attributed to The Athletic
    val_node = evaluator.add_parallel(
        id=f"position_{idx}_valuation_main",
        desc="Program valuation verification (The Athletic, ≥ $1B)",
        parent=pos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(position.valuation and position.valuation.strip()),
        id=f"position_{idx}_valuation_provided",
        desc="Program valuation is provided",
        parent=val_node,
        critical=True,
    )
    val_threshold_leaf = evaluator.add_leaf(
        id=f"position_{idx}_valuation_at_least_1b_athletic",
        desc="According to The Athletic, valuation is ≥ $1B",
        parent=val_node,
        critical=True,
    )
    val_claim = (
        f"According to The Athletic, {position.school_name or 'this program'} has a valuation of at least $1 billion."
    )
    await evaluator.verify(
        claim=val_claim,
        node=val_threshold_leaf,
        sources=position.urls,
        additional_instruction=(
            "Confirm the valuation threshold (≥ $1B) using The Athletic pages among the provided URLs. If multiple "
            "figures are shown, accept as long as the valuation is at least $1B."
        ),
    )
    # Attribution check: ensure at least one Athletic URL is present
    evaluator.add_custom_node(
        result=has_athletic_url(position.urls),
        id=f"position_{idx}_valuation_attributed_athletic",
        desc="Valuation attribution to The Athletic is supported by presence of The Athletic URL(s)",
        parent=val_node,
        critical=True,
    )

    # 5) Expected head coach salary ≥ $10M annually AND basis explained
    sal_node = evaluator.add_parallel(
        id=f"position_{idx}_salary_main",
        desc="Expected salary verification (≥ $10M with basis explained)",
        parent=pos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(position.expected_salary and position.expected_salary.strip()),
        id=f"position_{idx}_salary_provided",
        desc="Expected salary figure is provided",
        parent=sal_node,
        critical=True,
    )
    sal_threshold_leaf = evaluator.add_leaf(
        id=f"position_{idx}_salary_at_least_10m",
        desc="Expected head coach salary is stated as ≥ $10M annually",
        parent=sal_node,
        critical=True,
    )
    sal_threshold_claim = (
        f"The answer estimates the head coach salary for {position.school_name or 'this position'} at least $10 million annually."
    )
    await evaluator.verify(
        claim=sal_threshold_claim,
        node=sal_threshold_leaf,
        additional_instruction=(
            "Judge based on the answer text. Accept reasonable phrasing such as '$10M+', '>= $10M', or '$11 million'. "
            "Focus on whether the estimate is at least $10M annually."
        ),
    )
    sal_basis_leaf = evaluator.add_leaf(
        id=f"position_{idx}_salary_basis_explained",
        desc="Basis for salary estimate uses program tier and comparable positions",
        parent=sal_node,
        critical=True,
    )
    sal_basis_claim = (
        "The answer explains the basis for the salary estimate using program tier and comparable positions."
    )
    await evaluator.verify(
        claim=sal_basis_claim,
        node=sal_basis_leaf,
        additional_instruction=(
            "Review the answer text. The explanation should reference program tier/valuation and comparisons to other "
            "similar positions' salaries. If no such justification is provided, mark as incorrect."
        ),
    )

    # 6) Job grade is A or A- (The Athletic)
    grade_node = evaluator.add_parallel(
        id=f"position_{idx}_grade_main",
        desc="Job grade verification (The Athletic, A or A-)",
        parent=pos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(position.job_grade and position.job_grade.strip()),
        id=f"position_{idx}_grade_provided",
        desc="Job grade is provided",
        parent=grade_node,
        critical=True,
    )
    grade_value_leaf = evaluator.add_leaf(
        id=f"position_{idx}_grade_a_or_aminus_value",
        desc="Job grade is A or A-",
        parent=grade_node,
        critical=True,
    )
    grade_value_claim = f"The job grade value '{(position.job_grade or '').strip()}' is A or A-."
    await evaluator.verify(
        claim=grade_value_claim,
        node=grade_value_leaf,
        additional_instruction=(
            "Accept case-insensitive 'A' or 'A-'. If the value is not one of these, mark incorrect."
        ),
    )
    grade_attr_leaf = evaluator.add_leaf(
        id=f"position_{idx}_grade_athletic_attribution",
        desc="According to The Athletic, the job grade is A or A-",
        parent=grade_node,
        critical=True,
    )
    grade_attr_claim = (
        f"According to The Athletic, the job grade for {position.school_name or 'this program'} is A or A-."
    )
    await evaluator.verify(
        claim=grade_attr_claim,
        node=grade_attr_leaf,
        sources=position.urls,
        additional_instruction=(
            "Use The Athletic sources among the provided URLs to confirm the job grade classification (A or A-). "
            "Minor formatting differences are acceptable."
        ),
    )

    # 7) Open vacancy as of Nov 30, 2025
    vac_node = evaluator.add_parallel(
        id=f"position_{idx}_vacancy_main",
        desc="Vacancy status verification (as of Nov 30, 2025)",
        parent=pos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(position.vacancy_status_statement and position.vacancy_status_statement.strip()),
        id=f"position_{idx}_vacancy_statement_provided",
        desc="Vacancy statement is provided",
        parent=vac_node,
        critical=True,
    )
    vac_leaf = evaluator.add_leaf(
        id=f"position_{idx}_vacancy_open_as_of_date",
        desc="Position is stated as an open vacancy as of Nov 30, 2025",
        parent=vac_node,
        critical=True,
    )
    vac_claim = (
        f"As of November 30, 2025, the {position.school_name or 'school'} head coaching position was an open vacancy."
    )
    await evaluator.verify(
        claim=vac_claim,
        node=vac_leaf,
        sources=position.urls,
        additional_instruction=(
            "Use news or official reporting among the provided URLs to confirm the vacancy status as of Nov 30, 2025."
        ),
    )

    # 8) Supporting reference URLs sufficiency (Athletic claims + vacancy coverage)
    supp_leaf = evaluator.add_leaf(
        id=f"position_{idx}_supporting_urls_sufficient",
        desc="Supporting URLs are sufficient (include The Athletic for Athletic-based claims and at least one other for vacancy)",
        parent=pos_node,
        critical=True,
    )
    supp_claim = (
        "This position's provided URLs include at least one The Athletic page to substantiate Athletic-based claims "
        "(ranking/valuation/grade) and at least one additional source to substantiate the vacancy claim."
    )
    # Use a custom heuristic result if we want direct pass/fail; however per framework, verification by LLM is preferred.
    # We'll still call verify, and also record heuristic via a custom node to gate correctness.
    # First heuristic gate:
    supp_ok = has_athletic_url(position.urls) and has_non_athletic_url(position.urls) and len(position.urls) >= 2
    if not supp_ok:
        # If heuristic fails, we can short-circuit by marking failure directly via custom node before the leaf.
        evaluator.add_custom_node(
            result=False,
            id=f"position_{idx}_supporting_urls_heuristic_gate",
            desc="Heuristic check: URLs contain Athletic AND at least one non-Athletic source",
            parent=pos_node,
            critical=True,
        )
        # Mark leaf as failed explicitly
        supp_leaf.score = 0.0
        supp_leaf.status = "failed"
    else:
        # Proceed with LLM-supported verification using all URLs
        await evaluator.verify(
            claim=supp_claim,
            node=supp_leaf,
            sources=position.urls,
            additional_instruction=(
                "Confirm the presence and relevance of The Athletic and non-Athletic references within the provided URLs. "
                "It's sufficient if collectively they substantiate the Athletic-based claims and the vacancy status."
            ),
        )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the answer for Power 4 head coaching positions meeting The Athletic-based criteria.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level aggregation
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

    # Create a critical top-level node to mirror rubric strictness
    task_main = evaluator.add_parallel(
        id="task_main",
        desc="Identify all qualifying Power 4 head coaching position(s) meeting The Athletic-based criteria as of Nov 30, 2025",
        parent=root,
        critical=True,
    )

    # Extract positions from the answer
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Add custom info (e.g., count of positions parsed)
    evaluator.add_custom_info(
        info={"positions_count": len(extracted_positions.positions)},
        info_type="extraction_stats",
        info_name="positions_stats",
    )

    # Build per-position verification under a critical parent
    positions_parent = evaluator.add_parallel(
        id="positions_verification",
        desc="Per-position verification of all stated criteria",
        parent=task_main,
        critical=True,
    )

    # Verify each extracted position
    for idx, pos in enumerate(extracted_positions.positions):
        await verify_position(evaluator, positions_parent, pos, idx)

    # Completeness and exclusivity check at the task level
    completeness_leaf = evaluator.add_leaf(
        id="completeness_exclusivity",
        desc="The answer includes all and only the positions that satisfy every stated constraint",
        parent=task_main,
        critical=True,
    )
    schools_list = [p.school_name for p in extracted_positions.positions if p.school_name]
    schools_str = ", ".join(schools_list) if schools_list else "none listed"
    completeness_claim = (
        f"The answer's listed positions ({schools_str}) include all and only the positions that satisfy every stated "
        f"constraint as of Nov 30, 2025 (Athletic top-3 ranking, SEC/Big Ten, ≥ $1B valuation per Athletic, ≥ $10M "
        f"expected salary with basis, Athletic job grade A/A-, and vacancy status)."
    )
    await evaluator.verify(
        claim=completeness_claim,
        node=completeness_leaf,
        sources=collect_all_urls(extracted_positions),
        additional_instruction=(
            "Judge using the provided sources collectively. If any listed position fails a required criterion or if "
            "sources indicate a qualifying position not included (e.g., Athletic top-3 includes a different job meeting "
            "the other constraints), mark this claim incorrect."
        ),
    )

    # Return summary
    return evaluator.get_summary()