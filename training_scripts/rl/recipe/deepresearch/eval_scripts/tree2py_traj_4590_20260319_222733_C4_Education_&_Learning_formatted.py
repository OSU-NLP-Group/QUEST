import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ct_fl_district_compare"
TASK_DESCRIPTION = (
    "Compare Waterbury Public Schools in Connecticut and Duval County Public Schools in Florida by providing the "
    "following information for each district: (1) The exact student enrollment (total number of students), "
    "(2) The total number of schools operated by the district, (3) The minority enrollment percentage "
    "(percentage of students who are non-white), (4) The approximate population of the city or county served by "
    "each district based on 2024 U.S. Census data, (5) Connecticut's national ranking for public school quality, "
    "(6) The per-pupil spending amount for each state (Connecticut and Florida) with a comparison statement, and "
    "(7) Context on whether each district is among the largest in its respective state. For each piece of information, "
    "provide supporting reference URL(s) from official or authoritative sources."
)

YEAR_CENSUS = 2024

# Expected values per rubric
WATERBURY_EXPECTED = {
    "district_label": "Waterbury Public Schools (Connecticut)",
    "enrollment": "18,956",
    "schools": "29",
    "minority": "90%",
    "population": "115,908",  # approx, 2024 Census-based
    "largest_phrase": "among the three largest school districts in Connecticut",
}
DUVAL_EXPECTED = {
    "district_label": "Duval County Public Schools (Florida)",
    "enrollment": "127,971",
    "schools": "208",
    "minority": "70%",
    "population": "1,055,159",  # approx, 2024 Census-based
    "largest_phrase": "among the largest school districts in Florida",
}
CONNECTICUT_RANKING_EXPECTED = "#2"
SPENDING_EXPECTED = {
    "ct": "$20,635",
    "fl": "$9,406",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MetricValue(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DistrictLargestContext(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DistrictMetrics(BaseModel):
    enrollment: Optional[MetricValue] = None
    school_count: Optional[MetricValue] = None
    minority_pct: Optional[MetricValue] = None
    population_2024_census: Optional[MetricValue] = None
    largest_context: Optional[DistrictLargestContext] = None


class StateRanking(BaseModel):
    ranking_value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PerPupilSpending(BaseModel):
    ct_amount: Optional[str] = None
    fl_amount: Optional[str] = None
    comparison_statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AllExtraction(BaseModel):
    waterbury: Optional[DistrictMetrics] = None
    duval: Optional[DistrictMetrics] = None
    connecticut_ranking: Optional[StateRanking] = None
    per_pupil_spending: Optional[PerPupilSpending] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return f"""
Extract the following structured data exactly as stated in the provided answer and the URLs expressly cited therein.

GENERAL INSTRUCTIONS:
- Do not infer or fabricate any values or URLs. Only extract what the answer explicitly states and the URLs it explicitly cites.
- For each metric that asks for URL(s), extract all URLs the answer associates with that metric. Accept plain URLs or markdown links; output the resolved URL string.
- If a required field is missing, return null (for a single value) or an empty list (for URLs).
- Preserve number formatting (commas, percent signs, currency symbols) exactly as in the answer text for the 'value' fields.

STRUCTURE TO RETURN (JSON):

waterbury:
  enrollment:
    value: the enrollment number stated for Waterbury Public Schools, e.g., "18,956"
    urls: list of URLs cited to support Waterbury enrollment
  school_count:
    value: total number of schools stated for Waterbury, e.g., "29"
    urls: list of URLs cited to support Waterbury total schools
  minority_pct:
    value: stated minority enrollment percentage for Waterbury, e.g., "90%"
    urls: list of URLs cited to support this percentage
  population_2024_census:
    value: the approximate population stated for the City of Waterbury based on {YEAR_CENSUS} U.S. Census data, e.g., "115,908"
    urls: list of URLs cited to support this population (ideally U.S. Census)
  largest_context:
    statement: the exact statement used to describe Waterbury's standing among CT districts (e.g., "among the three largest school districts in Connecticut")
    urls: list of URLs cited to support this statement

duval:
  enrollment:
    value: the enrollment number stated for Duval County Public Schools, e.g., "127,971"
    urls: list of URLs cited to support Duval enrollment
  school_count:
    value: total number of schools stated for Duval, e.g., "208"
    urls: list of URLs cited to support this
  minority_pct:
    value: stated minority enrollment percentage for Duval, e.g., "70%"
    urls: list of URLs cited to support this percentage
  population_2024_census:
    value: the approximate population stated for Duval County based on {YEAR_CENSUS} U.S. Census data, e.g., "1,055,159"
    urls: list of URLs cited to support this population (ideally U.S. Census)
  largest_context:
    statement: the exact statement used to describe Duval's standing among FL districts (e.g., "among the largest school districts in Florida")
    urls: list of URLs cited to support this statement

connecticut_ranking:
  ranking_value: the stated national ranking for Connecticut public school quality, e.g., "#2"
  urls: list of URLs cited to support this ranking

per_pupil_spending:
  ct_amount: the stated per-pupil spending for Connecticut, e.g., "$20,635"
  fl_amount: the stated per-pupil spending for Florida, e.g., "$9,406"
  comparison_statement: the explicit comparison statement (e.g., "Connecticut spends more per pupil than Florida")
  urls: list of URLs cited to support the spending amounts and/or comparison
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls or [])


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_value_with_support(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    district_label: str,
    metric_label: str,
    expected_value: str,
    metric_obj: Optional[MetricValue],
    allow_approx_support: bool = False,
    authoritative_required: bool = True,
) -> None:
    """
    Generic checker for: existence (value + URLs), value stated in answer, and supported by URLs.
    """
    agg = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    exists = (
        metric_obj is not None
        and (metric_obj.value or "").strip() != ""
        and _has_urls(metric_obj.urls)
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc=f"{district_label} {metric_label}: value present in answer and at least one supporting URL cited",
        parent=agg,
        critical=True
    )

    stated_leaf = evaluator.add_leaf(
        id=f"{node_id}_stated_value_correct",
        desc=f"Answer states {district_label} {metric_label} as {expected_value}",
        parent=agg,
        critical=True
    )
    stated_claim = (
        f"In the answer text, the reported {district_label} {metric_label} equals '{expected_value}'. "
        f"Treat minor numeric formatting variations (commas, dollar sign, percent sign, spaces) as equivalent."
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction="Only pass if the answer explicitly conveys this same value (considering formatting)."
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc=f"The cited URL(s) support {district_label} {metric_label} = {expected_value}",
        parent=agg,
        critical=True
    )
    if allow_approx_support:
        support_claim = (
            f"According to the provided source page(s), the {metric_label} for {district_label} is approximately {expected_value}."
        )
        add_ins = (
            "Mark as supported only if the page(s) explicitly substantiate the figure within reasonable rounding "
            "or small tolerance. Consider it supported if the number equals or is within ~2% rounding difference. "
        )
    else:
        support_claim = (
            f"According to the provided source page(s), the {metric_label} for {district_label} is {expected_value}."
        )
        add_ins = "Treat formatting differences (commas, $ or %) as irrelevant; the numeric value must match."

    if authoritative_required:
        add_ins += " Only pass if at least one page is official/authoritative (e.g., district site, state DOE, NCES, U.S. Census, .gov, .edu, or a recognized reputable source)."

    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=metric_obj.urls if metric_obj else [],
        additional_instruction=add_ins
    )


async def verify_population_with_census(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    district_label: str,
    expected_value: str,
    metric_obj: Optional[MetricValue],
) -> None:
    """
    Specialized checker for the population metric requiring 2024 U.S. Census basis.
    """
    agg = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    exists = (
        metric_obj is not None
        and (metric_obj.value or "").strip() != ""
        and _has_urls(metric_obj.urls)
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc=f"{district_label} population: value present in answer and at least one supporting URL cited",
        parent=agg,
        critical=True
    )

    # Value stated (approximate allowed)
    stated_leaf = evaluator.add_leaf(
        id=f"{node_id}_stated_value_correct",
        desc=f"Answer states {district_label} population as approximately {expected_value}",
        parent=agg,
        critical=True
    )
    stated_claim = (
        f"In the answer text, the population for {district_label} is presented as approximately '{expected_value}'. "
        "Allow minor rounding or formatting differences but ensure the figure aligns with the stated approximate value."
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction="Accept small rounding differences; ensure the answer conveys an approximate value close to the expected figure."
    )

    # Mentions 2024 U.S. Census in answer
    census_mention_leaf = evaluator.add_leaf(
        id=f"{node_id}_mentions_{YEAR_CENSUS}_census",
        desc=f"Answer explicitly indicates the population figure is based on {YEAR_CENSUS} U.S. Census data",
        parent=agg,
        critical=True
    )
    census_mention_claim = (
        f"The answer explicitly states that the population figure for {district_label} is based on {YEAR_CENSUS} U.S. Census data "
        f"(e.g., mentions '{YEAR_CENSUS} U.S. Census', '{YEAR_CENSUS} Census estimate', or similar)."
    )
    await evaluator.verify(
        claim=census_mention_claim,
        node=census_mention_leaf,
        additional_instruction="Look for explicit reference to 2024 U.S. Census or equivalent phrasing."
    )

    # Supported by URLs (must be Census and 2024 aligned)
    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported_by_census",
        desc=f"The cited URL(s) (U.S. Census) support {district_label} population ≈ {expected_value} based on {YEAR_CENSUS} data",
        parent=agg,
        critical=True
    )
    support_claim = (
        f"According to the provided source page(s), which should be official U.S. Census pages (e.g., census.gov QuickFacts), "
        f"the population for the jurisdiction served by {district_label} is approximately {expected_value} and corresponds to {YEAR_CENSUS} data."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=metric_obj.urls if metric_obj else [],
        additional_instruction="Only pass if at least one URL is a census.gov page and the figure is consistent with 2024 data (allow small rounding differences)."
    )


async def verify_largest_context(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    district_label: str,
    expected_phrase: str,
    ctx_obj: Optional[DistrictLargestContext],
) -> None:
    """
    Verify the 'largest in state' context: existence (statement + URLs), claim in answer, and support by URLs.
    """
    agg = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    exists = (
        ctx_obj is not None
        and (ctx_obj.statement or "").strip() != ""
        and _has_urls(ctx_obj.urls)
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc=f"{district_label} largest-in-state context: statement present and at least one supporting URL cited",
        parent=agg,
        critical=True
    )

    stated_leaf = evaluator.add_leaf(
        id=f"{node_id}_stated_context_correct",
        desc=f"Answer gives correct-style context for {district_label} (e.g., '{expected_phrase}')",
        parent=agg,
        critical=True
    )
    stated_claim = (
        f"In the answer text, the district context for {district_label} asserts it is {expected_phrase} "
        f"(or equivalent wording such as 'top three largest' for CT or 'among the largest' for FL)."
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction="Allow synonymous phrasing that clearly conveys the same meaning."
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc=f"The cited URL(s) support the '{expected_phrase}' context for {district_label}",
        parent=agg,
        critical=True
    )
    support_claim = (
        f"The provided page(s) substantiate that {district_label} is {expected_phrase}."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=ctx_obj.urls if ctx_obj else [],
        additional_instruction="Only pass if at least one URL is official/authoritative (e.g., state DOE, NCES, district, reputable reports) and clearly supports the claim."
    )


async def verify_connecticut_ranking(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    expected_ranking: str,
    ranking_obj: Optional[StateRanking],
) -> None:
    agg = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    exists = (
        ranking_obj is not None
        and (ranking_obj.ranking_value or "").strip() != ""
        and _has_urls(ranking_obj.urls)
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc="Connecticut ranking: value present and at least one supporting URL cited",
        parent=agg,
        critical=True
    )

    stated_leaf = evaluator.add_leaf(
        id=f"{node_id}_stated_value_correct",
        desc=f"Answer states Connecticut's national public school quality ranking as {expected_ranking}",
        parent=agg,
        critical=True
    )
    stated_claim = (
        f"In the answer text, Connecticut's national ranking for public school quality is given as '{expected_ranking}'. "
        "Allow minor formatting variants like 'No. 2' or 'ranked 2nd' to count as the same."
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction="Check the answer content for an explicit #2 (or equivalent 2nd) ranking for CT."
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc=f"The cited URL(s) support that Connecticut is ranked {expected_ranking} nationally for public school quality",
        parent=agg,
        critical=True
    )
    support_claim = (
        f"The provided page(s) confirm that Connecticut is ranked {expected_ranking} nationally for public school quality."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=ranking_obj.urls if ranking_obj else [],
        additional_instruction="Only pass if the page is authoritative and clearly about national public school system quality ranking (e.g., WalletHub, recognized reputable ranking)."
    )


async def verify_per_pupil_spending(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    expected_ct: str,
    expected_fl: str,
    spend_obj: Optional[PerPupilSpending],
) -> None:
    agg = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    exists = (
        spend_obj is not None
        and (spend_obj.ct_amount or "").strip() != ""
        and (spend_obj.fl_amount or "").strip() != ""
        and (spend_obj.comparison_statement or "").strip() != ""
        and _has_urls(spend_obj.urls)
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc="Per-pupil spending: CT and FL amounts present, explicit comparison statement present, and at least one supporting URL cited",
        parent=agg,
        critical=True
    )

    # Stated CT amount
    ct_stated = evaluator.add_leaf(
        id=f"{node_id}_ct_amount_stated_correct",
        desc=f"Answer states Connecticut per-pupil spending as {expected_ct}",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"In the answer text, Connecticut per-pupil spending is stated as '{expected_ct}'. "
            "Treat trivial formatting differences (commas, currency symbol placement) as equivalent."
        ),
        node=ct_stated,
        additional_instruction="Pass only if this amount is clearly stated for CT."
    )

    # Stated FL amount
    fl_stated = evaluator.add_leaf(
        id=f"{node_id}_fl_amount_stated_correct",
        desc=f"Answer states Florida per-pupil spending as {expected_fl}",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"In the answer text, Florida per-pupil spending is stated as '{expected_fl}'. "
            "Treat trivial formatting differences (commas, currency symbol placement) as equivalent."
        ),
        node=fl_stated,
        additional_instruction="Pass only if this amount is clearly stated for FL."
    )

    # Comparison statement present (explicit)
    cmp_present = evaluator.add_leaf(
        id=f"{node_id}_comparison_statement_present",
        desc="Answer includes an explicit comparison statement about which state spends more per pupil",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states which state spends more per pupil (not just numbers).",
        node=cmp_present,
        additional_instruction="Look for explicit wording such as 'Connecticut spends more per pupil than Florida'."
    )

    # Comparison consistency with amounts
    cmp_consistent = evaluator.add_leaf(
        id=f"{node_id}_comparison_consistent_with_amounts",
        desc="The comparison statement is logically consistent with the stated amounts (CT > FL)",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Given per-pupil spending amounts of {expected_ct} for Connecticut and {expected_fl} for Florida in the answer, "
            "the comparison statement correctly asserts that Connecticut spends more per pupil than Florida."
        ),
        node=cmp_consistent,
        additional_instruction="Check both the stated numbers and the comparison statement for logical consistency."
    )

    # Support by URLs - CT
    ct_supported = evaluator.add_leaf(
        id=f"{node_id}_ct_amount_supported",
        desc=f"The cited URL(s) support Connecticut per-pupil spending ≈ {expected_ct}",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"According to the provided source page(s), Connecticut per-pupil spending is approximately {expected_ct}."
        ),
        node=ct_supported,
        sources=spend_obj.urls if spend_obj else [],
        additional_instruction="Only pass if at least one authoritative page (NCES, U.S. Census School Finance, state DOE/report) supports that magnitude; allow small rounding."
    )

    # Support by URLs - FL
    fl_supported = evaluator.add_leaf(
        id=f"{node_id}_fl_amount_supported",
        desc=f"The cited URL(s) support Florida per-pupil spending ≈ {expected_fl}",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"According to the provided source page(s), Florida per-pupil spending is approximately {expected_fl}."
        ),
        node=fl_supported,
        sources=spend_obj.urls if spend_obj else [],
        additional_instruction="Only pass if at least one authoritative page (NCES, U.S. Census School Finance, state DOE/report) supports that magnitude; allow small rounding."
    )


async def verify_district_block(
    evaluator: Evaluator,
    parent,
    block_id: str,
    block_desc: str,
    expected: Dict[str, str],
    metrics: Optional[DistrictMetrics],
) -> None:
    node = evaluator.add_parallel(
        id=block_id,
        desc=block_desc,
        parent=parent,
        critical=True
    )
    district_label = expected["district_label"]

    # Enrollment
    await verify_value_with_support(
        evaluator=evaluator,
        parent=node,
        node_id=f"{block_id.split('_')[0]}_enrollment" if "waterbury" in block_id or "duval" in block_id else f"{block_id}_enrollment",
        node_desc=f"States {district_label} enrollment as {expected['enrollment']} AND provides supporting authoritative URL(s)",
        district_label=district_label,
        metric_label="student enrollment (total)",
        expected_value=expected["enrollment"],
        metric_obj=metrics.enrollment if metrics else None,
        allow_approx_support=False,
        authoritative_required=True
    )

    # School count
    await verify_value_with_support(
        evaluator=evaluator,
        parent=node,
        node_id=f"{block_id.split('_')[0]}_school_count" if "waterbury" in block_id or "duval" in block_id else f"{block_id}_school_count",
        node_desc=f"States {district_label} total schools operated as {expected['schools']} AND provides supporting authoritative URL(s)",
        district_label=district_label,
        metric_label="total number of schools operated",
        expected_value=expected["schools"],
        metric_obj=metrics.school_count if metrics else None,
        allow_approx_support=False,
        authoritative_required=True
    )

    # Minority %
    await verify_value_with_support(
        evaluator=evaluator,
        parent=node,
        node_id=f"{block_id.split('_')[0]}_minority_pct" if "waterbury" in block_id or "duval" in block_id else f"{block_id}_minority_pct",
        node_desc=f"States {district_label} minority enrollment percentage as {expected['minority']} AND provides supporting authoritative URL(s)",
        district_label=district_label,
        metric_label="minority enrollment percentage",
        expected_value=expected["minority"],
        metric_obj=metrics.minority_pct if metrics else None,
        allow_approx_support=True,   # allow small rounding on support pages
        authoritative_required=True
    )

    # Population with 2024 Census
    await verify_population_with_census(
        evaluator=evaluator,
        parent=node,
        node_id=f"{block_id.split('_')[0]}_population_{YEAR_CENSUS}_census" if "waterbury" in block_id or "duval" in block_id else f"{block_id}_population_{YEAR_CENSUS}_census",
        node_desc=f"States the population (approx) for the served city/county based on {YEAR_CENSUS} U.S. Census data AND provides supporting authoritative URL(s)",
        district_label=district_label,
        expected_value=expected["population"],
        metric_obj=metrics.population_2024_census if metrics else None
    )

    # Largest-in-state context
    await verify_largest_context(
        evaluator=evaluator,
        parent=node,
        node_id=f"{block_id.split('_')[0]}_largest_in_state_context" if "waterbury" in block_id or "duval" in block_id else f"{block_id}_largest_in_state_context",
        node_desc=f"Provides correct-style 'largest in state' context for {district_label} AND provides supporting authoritative URL(s)",
        district_label=district_label,
        expected_phrase=expected["largest_phrase"],
        ctx_obj=metrics.largest_context if metrics else None
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator with a critical parallel root per rubric
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
        default_model=model
    )
    # Make root critical, as rubric treats the overall as essential
    root.critical = True

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllExtraction,
        extraction_name="extracted_metrics"
    )

    # Ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_waterbury": WATERBURY_EXPECTED,
        "expected_duval": DUVAL_EXPECTED,
        "expected_connecticut_ranking": CONNECTICUT_RANKING_EXPECTED,
        "expected_per_pupil_spending": SPENDING_EXPECTED,
        "notes": {
            "population_year": YEAR_CENSUS,
            "authoritative_sources_examples": ["district/DOE/NCES/U.S. Census/.gov/.edu/recognized reports"]
        }
    })

    # Waterbury metrics block
    await verify_district_block(
        evaluator=evaluator,
        parent=root,
        block_id="waterbury_metrics",
        block_desc="Provide required metrics for Waterbury Public Schools with supporting reference URL(s) from official or authoritative sources for each metric",
        expected=WATERBURY_EXPECTED,
        metrics=extracted.waterbury if extracted and extracted.waterbury else None
    )

    # Duval metrics block
    await verify_district_block(
        evaluator=evaluator,
        parent=root,
        block_id="duval_metrics",
        block_desc="Provide required metrics for Duval County Public Schools with supporting reference URL(s) from official or authoritative sources for each metric",
        expected=DUVAL_EXPECTED,
        metrics=extracted.duval if extracted and extracted.duval else None
    )

    # Connecticut ranking block
    await verify_connecticut_ranking(
        evaluator=evaluator,
        parent=root,
        node_id="connecticut_public_school_quality_ranking",
        node_desc="Identifies Connecticut's national ranking for public school quality as #2 AND provides supporting authoritative URL(s)",
        expected_ranking=CONNECTICUT_RANKING_EXPECTED,
        ranking_obj=extracted.connecticut_ranking if extracted and extracted.connecticut_ranking else None
    )

    # Per-pupil spending comparison block
    await verify_per_pupil_spending(
        evaluator=evaluator,
        parent=root,
        node_id="state_per_pupil_spending_comparison",
        node_desc="Provides CT per-pupil spending as $20,635 and FL per-pupil spending as approximately $9,406; includes an explicit comparison statement consistent with the stated figures; AND provides supporting authoritative URL(s)",
        expected_ct=SPENDING_EXPECTED["ct"],
        expected_fl=SPENDING_EXPECTED["fl"],
        spend_obj=extracted.per_pupil_spending if extracted and extracted.per_pupil_spending else None
    )

    # Return summary
    return evaluator.get_summary()