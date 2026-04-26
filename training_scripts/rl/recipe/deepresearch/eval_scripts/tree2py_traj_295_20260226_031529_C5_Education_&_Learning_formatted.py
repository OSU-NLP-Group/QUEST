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
TASK_ID = "edu_comp_study"
TASK_DESCRIPTION = """A researcher is conducting a comparative study of large public school districts and Harvard University's academic leadership programs. Please provide the following information:

Part 1: School Districts
Identify and provide details about three specific public school districts:

1. The second largest school division in Virginia:
   - Name of the district
   - Student enrollment for the 2023-2024 school year
   - Number of schools in the district
   
2. The largest school district in Maryland:
   - Name of the district
   - Student enrollment for the 2024-2025 school year
   - Number of schools in the district

3. The largest school district in Georgia:
   - Name of the district
   - Student enrollment for 2024-2025 or October 2025
   - Number of schools in the district

Part 2: Harvard University
Provide the following information about Harvard University:

1. The value of Harvard University's endowment as of June 30, 2025, and confirm its ranking status (whether it is the largest academic endowment in the world)

2. The founding year of the Program on Negotiation at Harvard Law School

3. The name of the person who has served as Chair of the Program on Negotiation since 1994

For all information, please provide reference URLs to support your answers.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    """Information for a single district, with per-field sources when available."""
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    enrollment: Optional[str] = None
    enrollment_year: Optional[str] = None  # e.g., "2023-2024", "2024-2025", "October 2025"
    enrollment_sources: List[str] = Field(default_factory=list)

    number_of_schools: Optional[str] = None
    schools_sources: List[str] = Field(default_factory=list)

    # Fallback general sources if the answer did not separate per field
    general_sources: List[str] = Field(default_factory=list)


class SchoolDistrictsExtraction(BaseModel):
    """Three specific districts requested in the task."""
    virginia_second_largest: Optional[DistrictInfo] = None
    maryland_largest: Optional[DistrictInfo] = None
    georgia_largest: Optional[DistrictInfo] = None


class HarvardInfo(BaseModel):
    """Harvard University info with sources."""
    endowment_value: Optional[str] = None
    endowment_value_sources: List[str] = Field(default_factory=list)

    endowment_ranking_status: Optional[str] = None  # e.g., "largest", "not largest"
    endowment_ranking_sources: List[str] = Field(default_factory=list)

    pon_founding_year: Optional[str] = None
    pon_founding_sources: List[str] = Field(default_factory=list)

    pon_chair_since_1994: Optional[str] = None
    pon_chair_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_school_districts() -> str:
    return """
    Extract structured information for the three requested public school districts exactly as presented in the answer. For each district, include per-field sources when available; otherwise include general sources applicable to the district.

    District A (Virginia second largest school division):
    - name: The district name identified as the second largest school division in Virginia.
    - name_sources: URLs that support that this district is the second largest school division in Virginia.
    - enrollment: The student enrollment figure for the 2023–2024 school year.
    - enrollment_year: The exact label provided in the answer (e.g., "2023–2024").
    - enrollment_sources: URLs that directly support the enrollment figure for the stated year.
    - number_of_schools: The number of schools in the district.
    - schools_sources: URLs that support the number of schools.
    - general_sources: Additional URLs cited for this district if sources are not separated per field.

    District B (Maryland largest school district):
    - name: The district name identified as the largest school district in Maryland.
    - name_sources: URLs that support that this district is the largest school district in Maryland.
    - enrollment: The student enrollment figure for the 2024–2025 school year.
    - enrollment_year: The exact label provided in the answer (e.g., "2024–2025").
    - enrollment_sources: URLs that directly support the enrollment figure for the stated year.
    - number_of_schools: The number of schools in the district.
    - schools_sources: URLs that support the number of schools.
    - general_sources: Additional URLs cited for this district if sources are not separated per field.

    District C (Georgia largest school district):
    - name: The district name identified as the largest school district in Georgia.
    - name_sources: URLs that support that this district is the largest school district in Georgia.
    - enrollment: The student enrollment figure for either "2024–2025" or "October 2025" (as allowed by the question).
    - enrollment_year: The exact label provided in the answer (e.g., "2024–2025" or "October 2025").
    - enrollment_sources: URLs that directly support the enrollment figure for the stated year.
    - number_of_schools: The number of schools in the district.
    - schools_sources: URLs that support the number of schools.
    - general_sources: Additional URLs cited for this district if sources are not separated per field.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent values.
    - For URL fields, return an array of complete URLs. If the answer uses markdown links, return the URLs.
    - If per-field sources are not provided, leave those arrays empty and include any overall references in general_sources.
    - If a required value is missing, set it to null.
    """


def prompt_extract_harvard() -> str:
    return """
    Extract structured information for Harvard University exactly as presented in the answer:
    - endowment_value: The value of Harvard's endowment as of June 30, 2025 (the textual figure as stated).
    - endowment_value_sources: URLs that explicitly support the endowment value as of June 30, 2025.
    - endowment_ranking_status: The ranking status as stated in the answer (e.g., "largest", "not largest", or a sentence indicating whether it is the largest academic endowment in the world).
    - endowment_ranking_sources: URLs that support the ranking status claim.
    - pon_founding_year: The founding year of the Program on Negotiation at Harvard Law School.
    - pon_founding_sources: URLs that support the founding year.
    - pon_chair_since_1994: The name of the person who has served as Chair of the Program on Negotiation since 1994.
    - pon_chair_sources: URLs that support the chair information.

    IMPORTANT:
    - Extract only what appears in the answer text.
    - Return arrays of complete URLs for sources.
    - If any field or its sources are missing, set the field to null or leave the array empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _use_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Choose primary sources if present; otherwise fallback."""
    if primary and len(primary) > 0:
        return primary
    return fallback or []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    code: str,
    title_desc: str,
    state_label: str,
    rank_phrase: str,  # e.g., "second largest school division" or "largest school district"
    info: Optional[DistrictInfo],
) -> None:
    """
    Verify one district: name with ranking claim, enrollment with year, and number of schools.
    Creates isolated sequential subpaths for each field to avoid cross-field gating.
    """
    district_node = evaluator.add_parallel(
        id=f"{code}",
        desc=title_desc,
        parent=parent_node,
        critical=False
    )

    # Handle None info gracefully
    name = info.name if info else None
    enroll = info.enrollment if info else None
    enroll_year = info.enrollment_year if info else None
    num_sch = info.number_of_schools if info else None

    name_sources = _use_sources(info.name_sources if info else [], info.general_sources if info else [])
    enroll_sources = _use_sources(info.enrollment_sources if info else [], info.general_sources if info else [])
    schools_sources = _use_sources(info.schools_sources if info else [], info.general_sources if info else [])

    # 1) Name + ranking claim
    name_seq = evaluator.add_sequential(
        id=f"{code}_name_main",
        desc=f"{title_desc} - Name and ranking claim verification",
        parent=district_node,
        critical=False
    )
    name_exist = evaluator.add_custom_node(
        result=bool(name) and len(name_sources) > 0,
        id=f"{code}_name_sources_provided",
        desc=f"{title_desc} - Name value and at least one supporting URL provided",
        parent=name_seq,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id=f"{code}_district_name",
        desc=f"Provide the name of the district and confirm it is the {rank_phrase} in {state_label}, supported by a reference URL",
        parent=name_seq,
        critical=True
    )
    name_claim = f"The {rank_phrase} in {state_label} is {name}."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=name_sources,
        additional_instruction=f"Verify that the cited page(s) explicitly support that {name} is the {rank_phrase} in {state_label}. Allow reasonable phrasing variants (e.g., 'second-largest', '2nd largest')."
    )

    # 2) Enrollment with specified year
    enroll_seq = evaluator.add_sequential(
        id=f"{code}_enroll_main",
        desc=f"{title_desc} - Enrollment verification",
        parent=district_node,
        critical=False
    )
    enroll_exist = evaluator.add_custom_node(
        result=bool(enroll) and len(enroll_sources) > 0,
        id=f"{code}_enrollment_sources_provided",
        desc=f"{title_desc} - Enrollment value and at least one supporting URL provided",
        parent=enroll_seq,
        critical=True
    )
    enroll_leaf = evaluator.add_leaf(
        id=f"{code}_enrollment_value",
        desc=f"Provide the student enrollment for the stated year, supported by a reference URL",
        parent=enroll_seq,
        critical=True
    )
    year_text = enroll_year if enroll_year else "the stated year"
    enroll_claim = f"The student enrollment of {name} for {year_text} is {enroll}."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=enroll_sources,
        additional_instruction="Verify the enrollment figure against the cited page(s). Allow minor rounding or formatting differences (e.g., commas). Ensure the year context matches the claim (e.g., 2023–2024, 2024–2025, or October 2025 as applicable)."
    )

    # 3) Number of schools
    schools_seq = evaluator.add_sequential(
        id=f"{code}_schools_main",
        desc=f"{title_desc} - Number of schools verification",
        parent=district_node,
        critical=False
    )
    schools_exist = evaluator.add_custom_node(
        result=bool(num_sch) and len(schools_sources) > 0,
        id=f"{code}_schools_sources_provided",
        desc=f"{title_desc} - Number of schools value and at least one supporting URL provided",
        parent=schools_seq,
        critical=True
    )
    schools_leaf = evaluator.add_leaf(
        id=f"{code}_number_of_schools",
        desc=f"Provide the number of schools in the district, supported by a reference URL",
        parent=schools_seq,
        critical=True
    )
    schools_claim = f"The number of schools in {name} is {num_sch}."
    await evaluator.verify(
        claim=schools_claim,
        node=schools_leaf,
        sources=schools_sources,
        additional_instruction="Verify that the cited page(s) support the total number of schools. Accept reasonable phrasing variants (e.g., 'X schools', 'operates X schools')."
    )


async def verify_harvard(
    evaluator: Evaluator,
    parent_node,
    info: HarvardInfo
) -> None:
    """
    Verify Harvard University information with strong source grounding for each requested fact.
    """
    harvard_node = evaluator.add_parallel(
        id="harvard_university",
        desc="Provide the requested Harvard University endowment and Program on Negotiation information.",
        parent=parent_node,
        critical=True  # Harvard info is essential for the overall study
    )

    # Endowment value as of June 30, 2025
    endow_seq = evaluator.add_sequential(
        id="endowment_value_main",
        desc="Harvard endowment value as of June 30, 2025",
        parent=harvard_node,
        critical=True
    )
    endow_exist = evaluator.add_custom_node(
        result=bool(info.endowment_value) and len(info.endowment_value_sources) > 0,
        id="endowment_value_sources_provided",
        desc="Endowment value and at least one supporting URL provided",
        parent=endow_seq,
        critical=True
    )
    endow_leaf = evaluator.add_leaf(
        id="endowment_value_as_of_june_30_2025",
        desc="Provide the value of Harvard University's endowment as of June 30, 2025, with supporting URL",
        parent=endow_seq,
        critical=True
    )
    endow_claim = f"As of June 30, 2025, Harvard University's endowment value is {info.endowment_value}."
    await evaluator.verify(
        claim=endow_claim,
        node=endow_leaf,
        sources=info.endowment_value_sources,
        additional_instruction="Verify that the cited page(s) explicitly state Harvard's endowment value with the date reference 'as of June 30, 2025'. Accept formatting variations."
    )

    # Endowment ranking status
    rank_seq = evaluator.add_sequential(
        id="endowment_ranking_main",
        desc="Harvard endowment ranking status",
        parent=harvard_node,
        critical=True
    )
    rank_exist = evaluator.add_custom_node(
        result=bool(info.endowment_ranking_status) and len(info.endowment_ranking_sources) > 0,
        id="endowment_ranking_sources_provided",
        desc="Endowment ranking status and at least one supporting URL provided",
        parent=rank_seq,
        critical=True
    )
    rank_leaf = evaluator.add_leaf(
        id="endowment_ranking_status",
        desc="State whether Harvard's endowment is the largest academic endowment in the world, supported by URL",
        parent=rank_seq,
        critical=True
    )
    # Build the ranking claim based on extracted status
    status_text = (info.endowment_ranking_status or "").strip().lower()
    if status_text in ["largest", "is largest", "largest in the world", "largest academic endowment"]:
        rank_claim = "Harvard University's endowment is the largest academic endowment in the world."
    elif status_text:
        rank_claim = "Harvard University's endowment is NOT the largest academic endowment in the world."
    else:
        # Fallback claim text if status missing; verification will likely fail via precondition
        rank_claim = "Harvard University's endowment ranking status is confirmed."
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=info.endowment_ranking_sources,
        additional_instruction="Verify the claim using the cited page(s). If the claim is that Harvard is NOT the largest, confirm evidence indicating a larger endowment at another institution or an authoritative ranking showing Harvard is not #1."
    )

    # PON founding year
    pon_found_seq = evaluator.add_sequential(
        id="pon_founding_year_main",
        desc="Program on Negotiation founding year",
        parent=harvard_node,
        critical=True
    )
    pon_found_exist = evaluator.add_custom_node(
        result=bool(info.pon_founding_year) and len(info.pon_founding_sources) > 0,
        id="pon_founding_year_sources_provided",
        desc="PON founding year value and at least one supporting URL provided",
        parent=pon_found_seq,
        critical=True
    )
    pon_found_leaf = evaluator.add_leaf(
        id="pon_founding_year",
        desc="Provide the founding year of the Program on Negotiation at Harvard Law School, supported by URL",
        parent=pon_found_seq,
        critical=True
    )
    pon_found_claim = f"The Program on Negotiation at Harvard Law School was founded in {info.pon_founding_year}."
    await evaluator.verify(
        claim=pon_found_claim,
        node=pon_found_leaf,
        sources=info.pon_founding_sources,
        additional_instruction="Verify the founding year using authoritative sources (e.g., the official PON site or Harvard resources). Accept reasonable wording variants."
    )

    # PON chair since 1994
    pon_chair_seq = evaluator.add_sequential(
        id="pon_chair_main",
        desc="Program on Negotiation Chair since 1994",
        parent=harvard_node,
        critical=True
    )
    pon_chair_exist = evaluator.add_custom_node(
        result=bool(info.pon_chair_since_1994) and len(info.pon_chair_sources) > 0,
        id="pon_chair_sources_provided",
        desc="PON chair name and at least one supporting URL provided",
        parent=pon_chair_seq,
        critical=True
    )
    pon_chair_leaf = evaluator.add_leaf(
        id="pon_chair_since_1994",
        desc="Provide the name of the person who has served as Chair of PON since 1994, supported by URL",
        parent=pon_chair_seq,
        critical=True
    )
    pon_chair_claim = f"The person who has served as Chair of the Program on Negotiation since 1994 is {info.pon_chair_since_1994}."
    await evaluator.verify(
        claim=pon_chair_claim,
        node=pon_chair_leaf,
        sources=info.pon_chair_sources,
        additional_instruction="Verify that the cited page(s) explicitly state the chair's service since 1994. Accept wording variants (e.g., 'has served as chair since 1994')."
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
    """
    Evaluate an answer for the comparative study task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: districts and Harvard evaluated independently
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

    # Extract information (in parallel)
    districts_task = evaluator.extract(
        prompt=prompt_extract_school_districts(),
        template_class=SchoolDistrictsExtraction,
        extraction_name="school_districts"
    )
    harvard_task = evaluator.extract(
        prompt=prompt_extract_harvard(),
        template_class=HarvardInfo,
        extraction_name="harvard_university"
    )
    districts_info, harvard_info = await asyncio.gather(districts_task, harvard_task)

    # Build School Districts subtree (non-critical to allow partial credit across districts)
    districts_node = evaluator.add_parallel(
        id="school_districts",
        desc="Provide details for the three requested public school districts.",
        parent=root,
        critical=False
    )

    # Virginia second largest school division
    await verify_single_district(
        evaluator=evaluator,
        parent_node=districts_node,
        code="district_1_va_second_largest",
        title_desc="Second largest school division in Virginia (provide required details).",
        state_label="Virginia",
        rank_phrase="second largest school division",
        info=districts_info.virginia_second_largest
    )

    # Maryland largest school district
    await verify_single_district(
        evaluator=evaluator,
        parent_node=districts_node,
        code="district_2_md_largest",
        title_desc="Largest school district in Maryland (provide required details).",
        state_label="Maryland",
        rank_phrase="largest school district",
        info=districts_info.maryland_largest
    )

    # Georgia largest school district
    await verify_single_district(
        evaluator=evaluator,
        parent_node=districts_node,
        code="district_3_ga_largest",
        title_desc="Largest school district in Georgia (provide required details).",
        state_label="Georgia",
        rank_phrase="largest school district",
        info=districts_info.georgia_largest
    )

    # Harvard University subtree (critical)
    await verify_harvard(
        evaluator=evaluator,
        parent_node=root,
        info=harvard_info
    )

    # Return standard summary
    return evaluator.get_summary()