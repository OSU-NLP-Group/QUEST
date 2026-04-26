import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_school_districts_2019_gradreqs_funding"
TASK_DESCRIPTION = """
Identify exactly four public school districts in the United States that meet all of the following criteria. Each district must be from a different state, and all must satisfy these requirements:

1. District Size: The school district must have had an enrollment of at least 130,000 students in autumn 2019, placing it among the top 20 largest school districts in the nation.

2. State Graduation Requirements: The state where the district is located must have high school graduation requirements that include:
   - At least 22 total credits required for graduation
   - Exactly 4.00 credits required in English/Language Arts
   - At least 3.00 credits required in Mathematics
   - At least 3.00 credits required in Science

3. State Funding Level: The state where the district is located must have annual per-pupil spending of at least $14,000 (based on 2025 data).

4. Geographic Diversity: All four districts must be located in different states.

For each of the four districts, provide:
- The official name of the school district
- The state where it is located
- The specific enrollment number for autumn 2019
- Verification of the state's graduation credit requirements (total credits, English credits, Mathematics credits, and Science credits)
- Verification of the state's per-pupil spending amount
- Reference URLs for all factual claims
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    # Identification
    name: Optional[str] = None
    state: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)

    # Enrollment
    enrollment_2019: Optional[str] = None  # keep as string to be robust to formatting (e.g., "130,014", "~131k")
    enrollment_urls: List[str] = Field(default_factory=list)

    # Graduation requirements (state-level)
    total_credits: Optional[str] = None
    english_credits: Optional[str] = None
    math_credits: Optional[str] = None
    science_credits: Optional[str] = None
    requirements_urls: List[str] = Field(default_factory=list)

    # Funding (state-level)
    spending_amount_2025: Optional[str] = None  # keep as string "$14,123"
    funding_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to the first four (4) public school districts mentioned in the answer that the author claims satisfy the criteria.
    For each district, strictly extract only what is explicitly stated in the answer. Do not infer or invent.
    If more than four districts are provided in the answer, keep only the first four in order. If fewer are given, extract what is available.

    For each district, extract the following fields:
    - name: Official name of the public school district (string, exactly as written)
    - state: The U.S. state where the district is located (string; can be full name or 2-letter code, as written)
    - identification_urls: All URLs that verify the district identification and its location (list of URLs)
    - enrollment_2019: The enrollment number for autumn 2019 as written (string; keep punctuation/commas if any)
    - enrollment_urls: URLs cited as the source for the 2019 autumn enrollment (list of URLs)
    - total_credits: State total high school graduation credits mentioned (string; e.g., "22", "24"; keep as written or null)
    - english_credits: State English/Language Arts credits for graduation (string; e.g., "4", "4.0"; keep as written or null)
    - math_credits: State Mathematics credits for graduation (string; keep as written or null)
    - science_credits: State Science credits for graduation (string; keep as written or null)
    - requirements_urls: URLs cited as the source for the state graduation requirements (list of URLs)
    - spending_amount_2025: The state's per-pupil spending amount for 2025 as written (string, include $ if present; or null)
    - funding_urls: URLs cited as the source for the 2025 per-pupil spending (list of URLs)

    Important:
    - Only return URLs that actually appear in the answer text (including markdown links). Do not fabricate URLs.
    - If a field is not present in the answer, set it to null (for strings) or [] (for lists).
    - Maintain the order of the districts as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _list_non_empty(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _normalize_state(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _first_k(items: List[DistrictItem], k: int = 4) -> List[DistrictItem]:
    return items[:k] if items else []


def _pad_to_k(items: List[DistrictItem], k: int = 4) -> List[DistrictItem]:
    out = list(items)
    while len(out) < k:
        out.append(DistrictItem())
    return out[:k]


# --------------------------------------------------------------------------- #
# Verification for a single district                                          #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    d: DistrictItem,
    index: int,
) -> None:
    """
    Build verification sub-tree for one district with sequential gating across major sections.
    """
    idx = index + 1
    district_node = evaluator.add_sequential(
        id=f"district_{idx}",
        desc=f"District #{idx} verification (must meet all criteria)",
        parent=parent_node,
        critical=False  # allow partial credit across districts at the Task_Completion level
    )

    # 1) Identification
    ident_node = evaluator.add_parallel(
        id=f"district_{idx}_identification",
        desc=f"District #{idx} identification (name, state, and supporting references)",
        parent=district_node,
        critical=True  # critical stage in the sequence
    )

    evaluator.add_custom_node(
        result=_non_empty(d.name),
        id=f"district_{idx}_name",
        desc="Provide the official name of the school district",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(d.state),
        id=f"district_{idx}_state",
        desc="Provide the state where the district is located",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_list_non_empty(d.identification_urls),
        id=f"district_{idx}_identification_url_present",
        desc="Provide reference URL verifying the district identification",
        parent=ident_node,
        critical=True
    )

    # Verify identification with sources
    ident_verify_leaf = evaluator.add_leaf(
        id=f"district_{idx}_identification_supported",
        desc="District identification (name + state) is supported by cited sources",
        parent=ident_node,
        critical=True
    )
    ident_claim = (
        f"The organization named '{d.name or ''}' is a public school district located in "
        f"{d.state or ''}, United States."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_verify_leaf,
        sources=d.identification_urls,
        additional_instruction=(
            "Verify that the page indicates the entity is a public school district (K-12) in the specified U.S. state. "
            "Allow reasonable variants in naming (e.g., abbreviations, 'USD', 'ISD', 'County Public Schools')."
        )
    )

    # 2) Enrollment (Autumn 2019) >= 130,000
    enroll_node = evaluator.add_parallel(
        id=f"district_{idx}_enrollment",
        desc="Verify district enrollment is at least 130,000 students based on autumn 2019 data",
        parent=district_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(d.enrollment_2019),
        id=f"district_{idx}_enrollment_value",
        desc="Provide the specific enrollment number for autumn 2019",
        parent=enroll_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_list_non_empty(d.enrollment_urls),
        id=f"district_{idx}_enrollment_url_present",
        desc="Provide reference URL for the enrollment data",
        parent=enroll_node,
        critical=True
    )

    enrollment_verify_leaf = evaluator.add_leaf(
        id=f"district_{idx}_enrollment_supported",
        desc="Autumn 2019 enrollment is at least 130,000 (supported by sources)",
        parent=enroll_node,
        critical=True
    )
    enrollment_claim = (
        f"In autumn 2019 (e.g., Fall 2019 or school year 2019–20), the district '{d.name or ''}' "
        f"had at least 130,000 students enrolled."
    )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_verify_leaf,
        sources=d.enrollment_urls,
        additional_instruction=(
            "Confirm the enrollment figure for Fall 2019 (or SY 2019–20). Accept small rounding or formatting differences "
            "(e.g., 130,000 vs 130k). If the page shows an exact 2019 or 2019–20 total enrollment >= 130,000, mark supported."
        )
    )

    # 3) State Graduation Requirements thresholds
    req_node = evaluator.add_parallel(
        id=f"district_{idx}_requirements",
        desc="Verify the state's high school graduation credit requirements meet all thresholds",
        parent=district_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_list_non_empty(d.requirements_urls),
        id=f"district_{idx}_requirements_url_present",
        desc="Provide reference URL for state graduation requirements",
        parent=req_node,
        critical=True
    )

    # 3.a total credits >= 22
    total_leaf = evaluator.add_leaf(
        id=f"district_{idx}_total_credits",
        desc="Verify state requires at least 22 total credits",
        parent=req_node,
        critical=True
    )
    total_claim = (
        f"The state of {d.state or ''} requires at least 22 total credits for high school graduation."
    )
    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=d.requirements_urls,
        additional_instruction=(
            "Look for state-level minimum graduation requirements. Accept synonyms such as 'units' or 'Carnegie units'. "
            "Local district add-ons are irrelevant; check the state minimum policy."
        )
    )

    # 3.b English exactly 4.00
    eng_leaf = evaluator.add_leaf(
        id=f"district_{idx}_english_credits",
        desc="Verify state requires exactly 4.00 credits in English",
        parent=req_node,
        critical=True
    )
    eng_claim = (
        f"The state of {d.state or ''} requires exactly 4 English (English Language Arts) credits to graduate."
    )
    await evaluator.verify(
        claim=eng_claim,
        node=eng_leaf,
        sources=d.requirements_urls,
        additional_instruction=(
            "Verify that the state's minimum requirement for English/English Language Arts is exactly 4 credits "
            "(not 3.5 or 4.5). Accept wording like 'English Language Arts'."
        )
    )

    # 3.c Math >= 3.00
    math_leaf = evaluator.add_leaf(
        id=f"district_{idx}_math_credits",
        desc="Verify state requires at least 3.00 credits in Mathematics",
        parent=req_node,
        critical=True
    )
    math_claim = (
        f"The state of {d.state or ''} requires at least 3 Mathematics credits to graduate from high school."
    )
    await evaluator.verify(
        claim=math_claim,
        node=math_leaf,
        sources=d.requirements_urls,
        additional_instruction=(
            "Confirm the state's minimum math credit requirement is 3 or more credits. "
            "Synonyms like 'units' should be treated as credits."
        )
    )

    # 3.d Science >= 3.00
    sci_leaf = evaluator.add_leaf(
        id=f"district_{idx}_science_credits",
        desc="Verify state requires at least 3.00 credits in Science",
        parent=req_node,
        critical=True
    )
    sci_claim = (
        f"The state of {d.state or ''} requires at least 3 Science credits to graduate from high school."
    )
    await evaluator.verify(
        claim=sci_claim,
        node=sci_leaf,
        sources=d.requirements_urls,
        additional_instruction=(
            "Confirm the state's minimum science credit requirement is 3 or more credits. "
            "Accept 'lab science' or similar as fulfilling science credit language."
        )
    )

    # 4) State Funding (2025) per-pupil spending >= $14,000
    fund_node = evaluator.add_parallel(
        id=f"district_{idx}_funding",
        desc="Verify the state's per-pupil spending meets the threshold",
        parent=district_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(d.spending_amount_2025),
        id=f"district_{idx}_spending_value",
        desc="Spending amount for 2025 is provided (as written in the answer)",
        parent=fund_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_list_non_empty(d.funding_urls),
        id=f"district_{idx}_funding_url_present",
        desc="Provide reference URL for per-pupil spending data",
        parent=fund_node,
        critical=True
    )

    funding_leaf = evaluator.add_leaf(
        id=f"district_{idx}_spending_supported",
        desc="Verify state per-pupil spending is at least $14,000 annually (2025)",
        parent=fund_node,
        critical=True
    )
    funding_claim = (
        f"In 2025, the per-pupil spending in {d.state or ''} is at least $14,000."
    )
    await evaluator.verify(
        claim=funding_claim,
        node=funding_leaf,
        sources=d.funding_urls,
        additional_instruction=(
            "Check for 2025 data. Accept terms like 'per-pupil spending', 'per student spending', "
            "'current expenditures per pupil', or 'PPE'. If a 2025 figure is clearly >= $14,000, mark supported. "
            "If only FY 2025 or 2024-25 reporting is available, it's acceptable as 2025 data."
        )
    )


# --------------------------------------------------------------------------- #
# Global checks                                                               #
# --------------------------------------------------------------------------- #
def add_global_checks(evaluator: Evaluator, parent_node, districts: List[DistrictItem]) -> None:
    """
    Add global constraints, such as geographic diversity and presence of 4 districts.
    """
    global_node = evaluator.add_parallel(
        id="global_checks",
        desc="Global constraints (e.g., geographic diversity, item count)",
        parent=parent_node,
        critical=False
    )

    # Exactly four districts provided (based on the filtered/padded first 4)
    exactly_four_present = all(_non_empty(d.name) and _non_empty(d.state) for d in districts)
    evaluator.add_custom_node(
        result=exactly_four_present,
        id="exactly_four_items_present",
        desc="Exactly four districts (with name and state) are provided",
        parent=global_node,
        critical=True
    )

    # Geographic diversity: all four states are different
    non_empty_states = [d.state for d in districts if _non_empty(d.state)]
    norm_states = [_normalize_state(s) for s in non_empty_states]
    states_all_distinct = len(non_empty_states) == 4 and len(set(norm_states)) == 4
    evaluator.add_custom_node(
        result=states_all_distinct,
        id="geographic_diversity_distinct_states",
        desc="All four districts are located in different states",
        parent=global_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the US public school districts task with 2019 enrollment, state graduation requirements,
    and 2025 funding thresholds.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level parallel aggregator
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

    # Add a Task_Completion wrapper node (non-critical to allow partial credit aggregation across districts)
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Identify four qualifying public school districts from different states with required evidence",
        parent=root,
        critical=False
    )

    # 2) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    districts = _first_k(extracted.districts or [], 4)
    districts = _pad_to_k(districts, 4)

    # 3) Build verification tree for each of the four districts (sequential per district)
    for i in range(4):
        await verify_single_district(evaluator, task_node, districts[i], i)

    # 4) Global checks (e.g., geographic diversity across the four)
    add_global_checks(evaluator, task_node, districts)

    # 5) Return summary
    return evaluator.get_summary()