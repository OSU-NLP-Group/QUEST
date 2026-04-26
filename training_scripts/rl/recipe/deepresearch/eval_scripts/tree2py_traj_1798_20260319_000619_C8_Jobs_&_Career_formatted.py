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
TASK_ID = "fds_public_universities_2024"
TASK_DESCRIPTION = (
    "Identify three public universities in different U.S. states that published First Destination Survey (FDS) "
    "career outcomes data for the 2023-24 academic year or the Class of 2024, with a knowledge rate of at least 65% "
    "and a career outcomes rate of at least 85%. For each university, provide the requested details. "
    "Ensure that each university is a public institution and that all three universities are located in different states."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    report_url: Optional[str] = None
    report_year: Optional[str] = None  # e.g., "2023-24", "Class of 2024"
    knowledge_rate: Optional[str] = None  # keep as string to be robust (e.g., "67%", "about 70%")
    career_outcomes_rate: Optional[str] = None  # e.g., "88%"
    overall_employment_rate: Optional[str] = None
    full_time_rate: Optional[str] = None
    continuing_ed_rate: Optional[str] = None
    salary: Optional[str] = None  # median or average starting salary
    sample_size: Optional[str] = None  # number of graduates included / responses
    top_sectors: List[str] = Field(default_factory=list)  # at least one sector if provided
    institution_type: Optional[str] = None  # e.g., "public", "private"
    extra_urls: List[str] = Field(default_factory=list)  # any additional URLs explicitly cited in the answer


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "From the answer, extract up to all universities mentioned that purportedly meet the task requirements. "
        "Return a JSON object with a field 'universities' which is an array of entries; for each university, extract:\n"
        "- name: University name as stated.\n"
        "- state: The U.S. state provided for the university (do not infer; use what's in the answer if present).\n"
        "- report_url: A URL pointing to the First Destination Survey (FDS) or career outcomes report/data page explicitly cited in the answer for that university.\n"
        "- report_year: The academic year or class year string as given (e.g., '2023-24', 'Class of 2024').\n"
        "- knowledge_rate: The knowledge rate as a string exactly as written in the answer (e.g., '68%', 'about 70%').\n"
        "- career_outcomes_rate: The career outcomes rate as a string.\n"
        "- overall_employment_rate: Overall employment rate/percentage string (if provided).\n"
        "- full_time_rate: Full-time employment rate/percentage string (if provided).\n"
        "- continuing_ed_rate: Continuing education rate/percentage string (if provided).\n"
        "- salary: Median or average starting salary string for bachelor's graduates (if provided).\n"
        "- sample_size: The number of graduates or responses included (if provided), as a string.\n"
        "- top_sectors: A list of at least one top employing sector/industry (strings) if provided in the answer.\n"
        "- institution_type: Institution type string if stated (e.g., 'public', 'private').\n"
        "- extra_urls: Any additional URLs explicitly provided in the answer for this university (besides report_url). "
        "Only include valid URLs actually present in the answer; do not invent.\n"
        "If a field is missing from the answer for a university, set it to null (or empty list for top_sectors/extra_urls). "
        "Do not invent values."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(u: UniversityItem) -> List[str]:
    urls: List[str] = []
    if u.report_url and u.report_url.strip():
        urls.append(u.report_url.strip())
    for x in u.extra_urls or []:
        if x and x.strip() and x.strip() not in urls:
            urls.append(x.strip())
    return urls


def norm_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return "".join(s.strip().lower().split())


def unique_state_for_index(states: List[Optional[str]], idx: int) -> bool:
    """
    Return True if the state at index idx is non-empty and distinct from the other two states (which also must be non-empty).
    """
    if idx < 0 or idx >= len(states):
        return False
    base = norm_state(states[idx])
    if not base:
        return False
    others = [norm_state(states[i]) for i in range(len(states)) if i != idx]
    # All others must be non-empty and different
    return all(o is not None and o != base for o in others)


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    root_parent,
    uni: UniversityItem,
    index: int,
    all_states: List[Optional[str]],
) -> None:
    """
    Build verification subtree for a single university and run checks.
    """
    uidx = index + 1
    uni_node = evaluator.add_parallel(
        id=f"University_{uidx}",
        desc=(
            "First qualifying public university with FDS data meeting all specified criteria"
            if uidx == 1 else
            ("Second qualifying public university with FDS data meeting all specified criteria"
             if uidx == 2 else
             "Third qualifying public university with FDS data meeting all specified criteria")
        ),
        parent=root_parent,
        critical=False
    )

    sources = collect_sources(uni)

    # Critical leaves requiring URL-backed verification
    # 1) Report URL validity/relevance
    report_url_node = evaluator.add_leaf(
        id=f"U{uidx}_Report_URL",
        desc=f"A valid URL to the university #{uidx} career outcomes report or First Destination Survey data is provided",
        parent=uni_node,
        critical=True
    )
    claim_url = (
        f"This URL is a valid page for {uni.name or 'the university'}'s First Destination Survey (FDS) "
        f"or career outcomes report/data for bachelor's graduates (or an official page that directly presents such outcomes)."
    )
    # Prefer verifying only the main report_url for this node
    await evaluator.verify(
        claim=claim_url,
        node=report_url_node,
        sources=uni.report_url,
        additional_instruction=(
            "Confirm the page is specifically about career outcomes/First Destination/graduate outcomes. "
            "Look for terms like 'First Destination', 'career outcomes', 'knowledge rate', 'outcomes rate', etc."
        ),
    )

    # 2) Institution type is public
    inst_type_node = evaluator.add_leaf(
        id=f"U{uidx}_Institution_Type",
        desc=f"The university #{uidx} is a public institution",
        parent=uni_node,
        critical=True
    )
    claim_public = (
        f"{uni.name or 'This university'} is a public institution (a public university)."
    )
    await evaluator.verify(
        claim=claim_public,
        node=inst_type_node,
        sources=sources,
        additional_instruction=(
            "Verify from the provided page(s) that the institution is public. "
            "If the career outcomes page does not state it, other provided URLs may indicate 'public', "
            "e.g., on an about page or similar. If none of the provided URLs support this, mark as not supported."
        ),
    )

    # 3) Report Year is 2023-24 or Class of 2024
    year_node = evaluator.add_leaf(
        id=f"U{uidx}_Report_Year",
        desc=f"The career outcomes data for university #{uidx} is from the 2023-24 academic year or Class of 2024",
        parent=uni_node,
        critical=True
    )
    claim_year = (
        "This page reports First Destination or career outcomes for either the 2023-24 academic year "
        "or the Class of 2024 (bachelor's)."
    )
    await evaluator.verify(
        claim=claim_year,
        node=year_node,
        sources=sources if sources else uni.report_url,
        additional_instruction=(
            "Look for explicit phrases like 'Class of 2024', '2023-24', '2024 graduates', or similar. "
            "If the page reports a combined multi-year dataset but clearly includes 2023-24 or Class of 2024 "
            "as the data being summarized, that is acceptable."
        ),
    )

    # 4) Knowledge rate ≥ 65%
    kr_node = evaluator.add_leaf(
        id=f"U{uidx}_Knowledge_Rate",
        desc=f"The university #{uidx}'s knowledge rate is at least 65%",
        parent=uni_node,
        critical=True
    )
    claim_kr = (
        "The page reports a knowledge rate for the First Destination (Class of 2024 or 2023-24) that is at least 65%."
    )
    await evaluator.verify(
        claim=claim_kr,
        node=kr_node,
        sources=sources if sources else uni.report_url,
        additional_instruction=(
            "Identify the 'knowledge rate' (percent of graduates with known outcomes). "
            "If the page explicitly reports a number ≥ 65%, the claim is supported. "
            "Allow minor rounding differences."
        ),
    )

    # 5) Career outcomes rate ≥ 85%
    cor_node = evaluator.add_leaf(
        id=f"U{uidx}_Career_Outcomes_Rate",
        desc=f"The university #{uidx}'s career outcomes rate is at least 85%",
        parent=uni_node,
        critical=True
    )
    claim_cor = (
        "The page reports a career outcomes rate (percent employed, military/service, or continuing education, "
        "among those with known outcomes and excluding not seeking) for the Class of 2024 or 2023-24 that is at least 85%."
    )
    await evaluator.verify(
        claim=claim_cor,
        node=cor_node,
        sources=sources if sources else uni.report_url,
        additional_instruction=(
            "Look specifically for the 'career outcomes rate' (sometimes labeled 'career outcome success rate' "
            "or similar) and verify it is ≥ 85%."
        ),
    )

    # Existence checks (critical or non-critical as per rubric)
    # Critical: state provided
    evaluator.add_custom_node(
        result=bool(uni.state and uni.state.strip()),
        id=f"U{uidx}_State_Location",
        desc=f"The state location of the university #{uidx} is provided",
        parent=uni_node,
        critical=True
    )

    # Critical: state uniqueness across all three
    evaluator.add_custom_node(
        result=unique_state_for_index(all_states, index),
        id=f"U{uidx}_State_Uniqueness",
        desc=f"The university #{uidx} is located in a different state from the other two universities",
        parent=uni_node,
        critical=True
    )

    # Non-critical: additional fields presence
    evaluator.add_custom_node(
        result=bool(uni.overall_employment_rate and uni.overall_employment_rate.strip()),
        id=f"U{uidx}_Overall_Employment",
        desc=f"The overall employment rate or percentage for university #{uidx} is provided",
        parent=uni_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(uni.full_time_rate and uni.full_time_rate.strip()),
        id=f"U{uidx}_Fulltime_Employment",
        desc=f"The full-time employment rate or percentage for university #{uidx} is provided",
        parent=uni_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(uni.continuing_ed_rate and uni.continuing_ed_rate.strip()),
        id=f"U{uidx}_Continuing_Education",
        desc=f"The continuing education rate or percentage for university #{uidx} is provided",
        parent=uni_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(uni.salary and uni.salary.strip()),
        id=f"U{uidx}_Salary_Data",
        desc=f"Median or average starting salary data for university #{uidx} is provided",
        parent=uni_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(uni.sample_size and uni.sample_size.strip()),
        id=f"U{uidx}_Sample_Size",
        desc=f"The number of graduates or response count included in university #{uidx}'s data is provided",
        parent=uni_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(uni.top_sectors and len(uni.top_sectors) > 0 and any(s.strip() for s in uni.top_sectors)),
        id=f"U{uidx}_Top_Sectors",
        desc=f"At least one top employing sector or industry for university #{uidx}'s graduates is provided",
        parent=uni_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating an answer for the FDS 2023-24 / Class of 2024 public universities task.
    """
    # Initialize evaluator (Root: use non-critical to allow partial credit across universities)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Take first three universities; pad with empty placeholders if fewer
    items: List[UniversityItem] = list(extraction.universities[:3])
    while len(items) < 3:
        items.append(UniversityItem())

    # Prepare states for uniqueness checks
    states_list = [itm.state for itm in items]
    evaluator.add_custom_info(
        info={"extracted_states": states_list, "university_count": len(items)},
        info_type="extraction_meta",
        info_name="extraction_meta"
    )

    # Build and verify each university subtree
    await asyncio.gather(*[
        verify_university(evaluator, root, items[i], i, states_list)
        for i in range(3)
    ])

    # Return summary
    return evaluator.get_summary()