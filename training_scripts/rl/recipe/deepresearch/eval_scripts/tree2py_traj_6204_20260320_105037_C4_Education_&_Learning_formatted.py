import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "two_superintendents_date_range_prev_sup"
TASK_DESCRIPTION = """
Identify two U.S. school district superintendents who both assumed their current superintendent positions between July 1, 2023 and August 31, 2024 (inclusive), and who had previously served as superintendent of a different school district before their current appointment. For each superintendent, provide the following information with reference URLs:
1) The name and U.S. state location of their current school district,
2) The specific date (month and year at minimum) they began serving in their current superintendent position,
3) The name and state of the school district where they previously served as superintendent,
4) Their highest educational degree earned (degree type, institution, and field of study or program area),
5) Their initial career position when they first entered the education field,
6) One verifiable district-specific metric from their current district (either: the exact enrollment number for the 2024-2025 school year, OR the exact dollar amount of the budget increase for the 2025-2026 school year, OR the district's ranking status within their state).
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StartDateInfo(BaseModel):
    date_text: Optional[str] = None  # e.g., "July 2023" or "July 10, 2023"
    sources: List[str] = Field(default_factory=list)


class PreviousPosition(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DegreeInfo(BaseModel):
    degree_type: Optional[str] = None  # e.g., "Ed.D.", "Ph.D.", "M.Ed."
    institution: Optional[str] = None
    field_or_program: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InitialCareerInfo(BaseModel):
    role: Optional[str] = None  # e.g., "Teacher", "Paraprofessional", etc.
    sources: List[str] = Field(default_factory=list)


class DistrictMetric(BaseModel):
    metric_type: Optional[str] = None  # one of: "enrollment_2024_2025" | "budget_increase_2025_2026" | "state_ranking"
    metric_value: Optional[str] = None  # keep string form exactly as in answer
    sources: List[str] = Field(default_factory=list)


class Superintendent(BaseModel):
    full_name: Optional[str] = None

    current_district: Optional[DistrictInfo] = None
    start_date: Optional[StartDateInfo] = None
    previous_superintendent_position: Optional[PreviousPosition] = None

    highest_degree: Optional[DegreeInfo] = None
    initial_career: Optional[InitialCareerInfo] = None

    district_metric: Optional[DistrictMetric] = None

    # Fallback sources if item-specific sources are missing
    general_sources: List[str] = Field(default_factory=list)


class SuperintendentsExtraction(BaseModel):
    superintendents: List[Superintendent] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendents() -> str:
    return """
Extract up to TWO distinct U.S. school district superintendents from the answer. For each superintendent, extract the following fields EXACTLY as stated in the answer text (do not invent or infer):

- full_name: The superintendent’s name.
- current_district: 
  - name: Current school district name.
  - state: U.S. state of the current district (full name or postal abbreviation as stated).
  - sources: An array of URLs cited in the answer that support the current district info.
- start_date:
  - date_text: The start date (month and year at minimum) they began serving in their current superintendent role (as quoted in answer).
  - sources: URLs cited that support this start date.
- previous_superintendent_position:
  - district_name: The previous district where they served as superintendent.
  - state: The U.S. state of that previous district.
  - sources: URLs cited that support the previous superintendent role.
- highest_degree:
  - degree_type: e.g., "Ed.D.", "Ph.D.", "M.Ed.", etc., as stated.
  - institution: Granting institution as stated.
  - field_or_program: The field or program area as stated (e.g., "Educational Leadership").
  - sources: URLs cited that support the degree info.
- initial_career:
  - role: The initial/first position in the education field (e.g., "teacher", "paraprofessional").
  - sources: URLs cited that support the initial role info.
- district_metric:
  - metric_type: One of exactly: "enrollment_2024_2025", "budget_increase_2025_2026", or "state_ranking".
  - metric_value: The exact value text for the metric as stated in the answer (e.g., "6,412", "$3.5 million", "ranked #2 in state").
  - sources: URLs cited that support the metric.
- general_sources: Any additional URLs cited for this superintendent that are not tied to a specific field above.

Rules:
1) Only include URLs explicitly present in the answer (plain or within markdown links).
2) If a field is not present in the answer, set it to null (or empty array for sources).
3) Do not invent values or URLs.
4) Preserve the exact surface form of values where possible (e.g., keep commas and $ for monetary amounts; keep month/year format as shown).
5) Return a JSON with a 'superintendents' array. If more than two are in the answer, include the first two.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _pick_sources(*source_lists: Optional[List[str]]) -> List[str]:
    """
    Return the first non-empty list among the given lists; otherwise return [].
    """
    for lst in source_lists:
        if lst and len(lst) > 0:
            return lst
    return []


def _safe(val: Optional[str]) -> str:
    return val if val is not None else ""


# --------------------------------------------------------------------------- #
# Verification for a single superintendent                                    #
# --------------------------------------------------------------------------- #
async def verify_superintendent(
    evaluator: Evaluator,
    parent_node,
    sup: Superintendent,
    sup_idx: int,
) -> None:
    """
    Build verification subtree for one superintendent. All checks under this superintendent
    are critical (as per rubric) and are evaluated in parallel (no order dependency).
    """
    node = evaluator.add_parallel(
        id=f"superintendent_{sup_idx}",
        desc=f"{'First' if sup_idx == 1 else 'Second'} superintendent's information is complete and accurate",
        parent=parent_node,
        critical=False
    )

    sup_name = _safe(sup.full_name)

    # 1) Current district name and state
    cur_district = sup.current_district or DistrictInfo()
    cur_sources = _pick_sources(cur_district.sources, sup.general_sources)

    n1 = evaluator.add_leaf(
        id=f"sup{sup_idx}_current_district",
        desc="Current district name and U.S. state location are correctly identified",
        parent=node,
        critical=True
    )
    claim1 = (
        f"{sup_name} currently serves as superintendent of the district '{_safe(cur_district.name)}' "
        f"in the U.S. state '{_safe(cur_district.state)}'."
    )
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=cur_sources,
        additional_instruction=(
            "Confirm that the cited page(s) explicitly indicate the person is the current superintendent of the named district, "
            "and that the district's state matches. Allow reasonable naming variants (e.g., 'USD 123', 'Public Schools', etc.)."
        )
    )

    # 2) Start date in role and falls within [2023-07-01, 2024-08-31]
    start_date = sup.start_date or StartDateInfo()
    start_sources = _pick_sources(start_date.sources, sup.general_sources)

    n2 = evaluator.add_leaf(
        id=f"sup{sup_idx}_start_date",
        desc="Start date in current position is correctly identified and falls between July 1, 2023 and August 31, 2024 (inclusive)",
        parent=node,
        critical=True
    )
    claim2 = (
        f"{sup_name} began serving as superintendent of '{_safe(cur_district.name)}' in '{_safe(start_date.date_text)}', "
        "and this date falls between July 1, 2023 and August 31, 2024, inclusive."
    )
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=start_sources,
        additional_instruction=(
            "Verify the page explicitly supports the start date. Then judge whether this date is on or after July 1, 2023 and on or before August 31, 2024. "
            "If only month/year is given, interpret it naturally (e.g., 'July 2023' is within range). If the cited date is outside the range, mark as not supported."
        )
    )

    # 3) Previous superintendent position (different district and state provided)
    prev_pos = sup.previous_superintendent_position or PreviousPosition()
    prev_sources = _pick_sources(prev_pos.sources, sup.general_sources)

    n3 = evaluator.add_leaf(
        id=f"sup{sup_idx}_previous_position",
        desc="Previous superintendent position (district name and state) is correctly identified",
        parent=node,
        critical=True
    )
    claim3 = (
        f"Before the current role, {sup_name} previously served as superintendent of '{_safe(prev_pos.district_name)}' "
        f"in '{_safe(prev_pos.state)}', and that district is different from the current district '{_safe(cur_district.name)}'."
    )
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=prev_sources,
        additional_instruction=(
            "Confirm that the cited page explicitly describes the person as a prior superintendent of the named (different) district. "
            "Ensure it is a superintendent role (not assistant/associate). The previous district must not be the same as the current district."
        )
    )

    # 4) Highest educational degree
    deg = sup.highest_degree or DegreeInfo()
    deg_sources = _pick_sources(deg.sources, sup.general_sources)

    n4 = evaluator.add_leaf(
        id=f"sup{sup_idx}_highest_degree",
        desc="Highest educational degree earned (degree type, institution, and field/program) is correctly identified",
        parent=node,
        critical=True
    )
    claim4 = (
        f"{sup_name}'s highest educational degree is '{_safe(deg.degree_type)}' in '{_safe(deg.field_or_program)}' "
        f"from '{_safe(deg.institution)}'."
    )
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=deg_sources,
        additional_instruction=(
            "Check that the page supports the exact degree type, the institution, and the field/program. "
            "If multiple degrees are listed, verify that the claimed one is indeed the highest-level degree."
        )
    )

    # 5) Initial career position in education
    init_career = sup.initial_career or InitialCareerInfo()
    init_sources = _pick_sources(init_career.sources, sup.general_sources)

    n5 = evaluator.add_leaf(
        id=f"sup{sup_idx}_initial_career",
        desc="Initial career position in education is correctly identified",
        parent=node,
        critical=True
    )
    claim5 = f"{sup_name}'s initial/first position in the education field was '{_safe(init_career.role)}'."
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=init_sources,
        additional_instruction=(
            "Confirm that the cited page clearly indicates the person's first role in education (e.g., teacher, paraprofessional). "
            "Allow reasonable synonyms but ensure it is described as the first or initial role."
        )
    )

    # 6) District-specific metric (one of the allowed types)
    metric = sup.district_metric or DistrictMetric()
    metric_sources = _pick_sources(metric.sources, sup.general_sources)

    n6 = evaluator.add_leaf(
        id=f"sup{sup_idx}_district_metric",
        desc="A verifiable district-specific metric (enrollment number for 2024-2025, budget increase amount for 2025-2026, or ranking status within state) is correctly provided",
        parent=node,
        critical=True
    )

    # Build claim tailored to the metric_type when possible; otherwise general check.
    mtype = (metric.metric_type or "").strip().lower()
    mval = _safe(metric.metric_value)
    cur_dist_label = f"{_safe(cur_district.name)} ({_safe(cur_district.state)})"

    if mtype == "enrollment_2024_2025":
        claim6 = (
            f"The exact enrollment for the 2024-2025 school year in {cur_dist_label} is {mval}."
        )
        add_ins6 = (
            "Verify that the page shows the district's enrollment for the 2024-2025 school year and that the number exactly matches the claim "
            "(allow formatting differences like commas)."
        )
    elif mtype == "budget_increase_2025_2026":
        claim6 = (
            f"The exact budget increase amount for the 2025-2026 school year in {cur_dist_label} is {mval}."
        )
        add_ins6 = (
            "Verify that the page shows a budget increase for the 2025-2026 school year and that the dollar amount exactly matches the claim "
            "(allow $ signs and comma formatting). Ensure it is an increase amount, not total budget."
        )
    elif mtype == "state_ranking":
        claim6 = (
            f"The district {cur_dist_label} has the following ranking within its state: {mval}."
        )
        add_ins6 = (
            "Verify that the page reports a ranking within the state (not national), and that the stated ranking exactly matches the claim. "
            "If the ranking is category-specific (e.g., test scores), ensure the category context matches."
        )
    else:
        # Fallback generic claim if extractor couldn't classify
        claim6 = (
            f"A district-specific metric for {cur_dist_label} is correctly provided: '{mval}', "
            "and it is one of (enrollment for 2024-2025, budget increase for 2025-2026, or state ranking)."
        )
        add_ins6 = (
            "First, assess whether the claimed metric is one of the allowed types: enrollment for 2024-2025, budget increase for 2025-2026, or state ranking. "
            "Then verify the exact value against the page."
        )

    await evaluator.verify(
        claim=claim6,
        node=n6,
        sources=metric_sources,
        additional_instruction=add_ins6
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    """
    Evaluate an answer for the 'two superintendents with date range and prior superintendent role' task.
    """
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendents(),
        template_class=SuperintendentsExtraction,
        extraction_name="superintendents_extraction"
    )

    # Ensure we have exactly two entries for downstream verification (pad with empty if fewer)
    supers: List[Superintendent] = list(extracted.superintendents[:2])
    while len(supers) < 2:
        supers.append(Superintendent())

    # Build verification subtrees for each superintendent
    await verify_superintendent(evaluator, root, supers[0], 1)
    await verify_superintendent(evaluator, root, supers[1], 2)

    # Return final summary
    return evaluator.get_summary()