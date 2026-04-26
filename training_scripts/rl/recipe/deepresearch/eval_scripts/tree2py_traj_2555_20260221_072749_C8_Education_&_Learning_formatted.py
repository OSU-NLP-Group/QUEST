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
TASK_ID = "us_school_districts_large_criteria"
TASK_DESCRIPTION = """Identify 3 large public school districts in the United States that meet ALL of the following criteria:

Enrollment & Location:
- Student enrollment between 80,000 and 95,000 for the 2024-2025 or 2025-2026 school year
- Each district must be located in a different U.S. state

Governance:
- The district superintendent must be appointed by the local school board (not elected by voters)

Student Demographics:
- At least 50% of enrolled students must be from minority racial/ethnic backgrounds
- At least 35% of students must be economically disadvantaged (eligible for free or reduced-price lunch)

District Characteristics:
- The district must operate at least 85 schools
- The district must offer both traditional schools AND at least one type of alternative/specialized program (such as magnet schools, alternative schools, or charter schools within the district)

Programs & Services:
- The district must have schools participating in the federal Title I program
- The district must provide IDEA-compliant special education services
- The district must participate in the National School Lunch Program (NSLP)

Accreditation:
- The district must be accredited by an appropriate regional accrediting agency

For each district identified, provide the district name, state, and reference URLs that verify it meets these criteria using current data from the 2024-2025 or 2025-2026 school year.
"""
ACCEPTABLE_SCHOOL_YEARS = {"2024-2025", "2025-2026"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None

    enrollment: Optional[str] = None
    enrollment_year: Optional[str] = None

    superintendent_appointment: Optional[str] = None  # e.g., "appointed", "elected", or description

    minority_percentage: Optional[str] = None  # e.g., "52%", "approximately 55%"
    econ_disadv_percentage: Optional[str] = None  # e.g., "38%", ">=35%"

    school_count: Optional[str] = None

    title_i: Optional[str] = None  # "yes"/"no"/"unknown"
    idea_services: Optional[str] = None  # "yes"/"no"/"unknown"
    nslp: Optional[str] = None  # "yes"/"no"/"unknown"

    program_types: List[str] = Field(default_factory=list)  # e.g., ["traditional", "magnet", "alternative", "charter"]

    accreditation_agency: Optional[str] = None

    reference_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to three qualifying large public school districts from the answer. If more than three are mentioned, return only the first three. For each district, extract the following fields exactly as provided in the answer (use strings for numeric values):

    - district_name: The official district name
    - state: The U.S. state where the district is located
    - enrollment: The student enrollment figure stated
    - enrollment_year: The school year associated with the enrollment figure (prefer "2024-2025" or "2025-2026")
    - superintendent_appointment: How the superintendent is selected (e.g., "appointed by board", "elected by voters")
    - minority_percentage: Percentage of students from minority backgrounds (string; include symbols like % if present)
    - econ_disadv_percentage: Percentage of economically disadvantaged students (string)
    - school_count: Number of schools the district operates (string)
    - title_i: Whether the district has Title I schools ("yes"/"no"/"unknown")
    - idea_services: Whether the district provides IDEA-compliant special education services ("yes"/"no"/"unknown")
    - nslp: Whether the district participates in NSLP ("yes"/"no"/"unknown")
    - program_types: List of program types the district offers, including "traditional" and any specialized programs ("magnet", "alternative", "charter", etc.)
    - accreditation_agency: Name of the regional accreditation agency, if stated (e.g., "Cognia")
    - reference_urls: All URLs cited in the answer that support this district’s attributes; include valid URLs only. If the answer uses markdown links, extract their URLs.

    Return a JSON object with a single field "districts", an array of up to three district objects with the fields above. Use null for any missing field. If no districts are found, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_k(items: List[DistrictItem], k: int) -> List[DistrictItem]:
    result = items[:k]
    if len(result) < k:
        # pad with empty placeholders
        for _ in range(k - len(result)):
            result.append(DistrictItem())
    return result


def _states_list(districts: List[DistrictItem]) -> List[Optional[str]]:
    return [d.state for d in districts]


def _is_state_unique_for_index(states: List[Optional[str]], idx: int) -> bool:
    my_state = states[idx]
    if not my_state:
        return False
    others = [s for i, s in enumerate(states) if i != idx]
    return all((s is None or s.strip().lower() != my_state.strip().lower()) for s in others)


# --------------------------------------------------------------------------- #
# Verification for a single district                                          #
# --------------------------------------------------------------------------- #
async def verify_district(
    evaluator: Evaluator,
    parent_node,
    district: DistrictItem,
    idx: int,
    all_states: List[Optional[str]]
) -> None:
    """
    Build verification subtree and perform checks for one district.
    """
    # Create district node (parallel, non-critical to allow partial credit per district)
    district_node = evaluator.add_parallel(
        id=f"District_{idx + 1}",
        desc=f"{['First', 'Second', 'Third'][idx]} qualifying school district meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Reference URLs existence (Critical) – gate other verifications
    urls_exist = bool(district.reference_urls)
    evaluator.add_custom_node(
        result=urls_exist,
        id=f"District_{idx + 1}_Reference_URLs",
        desc="Provide reference URLs supporting the district identification and attribute verification",
        parent=district_node,
        critical=True
    )

    # Enrollment range (Critical)
    node_enroll = evaluator.add_leaf(
        id=f"District_{idx + 1}_Enrollment_Range",
        desc="District enrollment is between 80,000-95,000 students for 2024-2025 or 2025-2026 school year",
        parent=district_node,
        critical=True
    )
    enroll_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' has a stated enrollment of {district.enrollment or 'UNKNOWN'} "
        f"for the {district.enrollment_year or 'UNKNOWN YEAR'} school year, and this figure lies between 80,000 and 95,000 students."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=node_enroll,
        sources=district.reference_urls,
        additional_instruction=(
            "Verify the enrollment number and the stated school year (prefer 2024-2025 or 2025-2026) from the provided URLs. "
            "Confirm that the enrollment falls within [80,000, 95,000]."
        ),
    )

    # State location checks: split into two critical leaves under a critical parallel node
    state_main = evaluator.add_parallel(
        id=f"District_{idx + 1}_State_Location",
        desc="State location checks for district (correct state and distinct from others)",
        parent=district_node,
        critical=True
    )

    node_state_source = evaluator.add_leaf(
        id=f"District_{idx + 1}_State_Location_Source",
        desc="District is located in the stated U.S. state (source-supported)",
        parent=state_main,
        critical=True
    )
    state_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' is located in the U.S. state of {district.state or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=state_claim,
        node=node_state_source,
        sources=district.reference_urls,
        additional_instruction="Confirm the district's state based on the provided URLs."
    )

    # State uniqueness across all three districts (Critical, custom)
    evaluator.add_custom_node(
        result=_is_state_unique_for_index(all_states, idx),
        id=f"District_{idx + 1}_State_Location_Unique",
        desc="District is located in a U.S. state different from the other two identified districts",
        parent=state_main,
        critical=True
    )

    # Superintendent appointed by board (Critical)
    node_sup = evaluator.add_leaf(
        id=f"District_{idx + 1}_Appointed_Superintendent",
        desc="District superintendent is appointed by the local school board, not elected by voters",
        parent=district_node,
        critical=True
    )
    sup_claim = (
        f"The superintendent of {district.district_name or 'the district'} is appointed by the local school board (not elected by voters). "
        f"Statement from the answer: {district.superintendent_appointment or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=sup_claim,
        node=node_sup,
        sources=district.reference_urls,
        additional_instruction="Look for board policies or governance descriptions confirming appointment rather than election."
    )

    # Minority enrollment >= 50% (Critical)
    node_minority = evaluator.add_leaf(
        id=f"District_{idx + 1}_Minority_Enrollment",
        desc="At least 50% of enrolled students are from minority racial/ethnic backgrounds",
        parent=district_node,
        critical=True
    )
    minority_claim = (
        f"At least 50% of students in {district.district_name or 'the district'} are from minority racial/ethnic backgrounds; "
        f"the answer cites {district.minority_percentage or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=minority_claim,
        node=node_minority,
        sources=district.reference_urls,
        additional_instruction="Verify the stated percentage or a clear statement indicating ≥ 50% minority enrollment."
    )

    # Economically disadvantaged >= 35% (Critical)
    node_econ = evaluator.add_leaf(
        id=f"District_{idx + 1}_Economic_Disadvantage",
        desc="At least 35% of students are economically disadvantaged (eligible for free or reduced-price lunch)",
        parent=district_node,
        critical=True
    )
    econ_claim = (
        f"At least 35% of students in {district.district_name or 'the district'} are economically disadvantaged; "
        f"the answer cites {district.econ_disadv_percentage or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=econ_claim,
        node=node_econ,
        sources=district.reference_urls,
        additional_instruction="Confirm the proportion of economically disadvantaged students (free/reduced-price lunch eligibility)."
    )

    # School count >= 85 (Critical)
    node_school_count = evaluator.add_leaf(
        id=f"District_{idx + 1}_School_Count",
        desc="District operates at least 85 schools",
        parent=district_node,
        critical=True
    )
    count_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' operates at least 85 schools; "
        f"the reported number is {district.school_count or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=count_claim,
        node=node_school_count,
        sources=district.reference_urls,
        additional_instruction="Verify the total number of schools operated by the district; confirm it is ≥ 85."
    )

    # Title I participation (Critical)
    node_title_i = evaluator.add_leaf(
        id=f"District_{idx + 1}_Title_I_Participation",
        desc="District has schools participating in the federal Title I program",
        parent=district_node,
        critical=True
    )
    title_i_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' has schools that participate in the Title I program."
    )
    await evaluator.verify(
        claim=title_i_claim,
        node=node_title_i,
        sources=district.reference_urls,
        additional_instruction="Confirm mention of Title I schools/program participation on district or authoritative pages."
    )

    # IDEA-compliant special education services (Critical)
    node_idea = evaluator.add_leaf(
        id=f"District_{idx + 1}_Special_Education_Services",
        desc="District provides IDEA-compliant special education services",
        parent=district_node,
        critical=True
    )
    idea_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' provides special education services compliant with IDEA."
    )
    await evaluator.verify(
        claim=idea_claim,
        node=node_idea,
        sources=district.reference_urls,
        additional_instruction="Look for descriptions/policies aligning with IDEA, e.g., procedural safeguards, FAPE, IEPs."
    )

    # NSLP participation (Critical)
    node_nslp = evaluator.add_leaf(
        id=f"District_{idx + 1}_NSLP_Participation",
        desc="District participates in the National School Lunch Program (NSLP)",
        parent=district_node,
        critical=True
    )
    nslp_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' participates in the National School Lunch Program (NSLP)."
    )
    await evaluator.verify(
        claim=nslp_claim,
        node=node_nslp,
        sources=district.reference_urls,
        additional_instruction="Confirm NSLP participation through district nutrition services pages or authoritative references."
    )

    # Program variety: traditional + at least one specialized (Critical)
    node_programs = evaluator.add_leaf(
        id=f"District_{idx + 1}_Program_Variety",
        desc="District offers both traditional schools and at least one type of alternative/specialized program (magnet, alternative, or charter schools)",
        parent=district_node,
        critical=True
    )
    described_programs = ", ".join(district.program_types) if district.program_types else "UNKNOWN"
    program_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' offers traditional schools and at least one specialized program "
        f"(e.g., magnet, alternative, or charter). The answer mentions: {described_programs}."
    )
    await evaluator.verify(
        claim=program_claim,
        node=node_programs,
        sources=district.reference_urls,
        additional_instruction="Confirm existence of traditional schools and at least one of magnet/alternative/charter programs within the district."
    )

    # Data currency (Critical)
    node_currency = evaluator.add_leaf(
        id=f"District_{idx + 1}_Data_Currency",
        desc="All verification information is from the 2024-2025 or 2025-2026 school year",
        parent=district_node,
        critical=True
    )
    currency_claim = (
        "The data cited for this district (enrollment and student statistics) is from the 2024-2025 or 2025-2026 school year."
    )
    await evaluator.verify(
        claim=currency_claim,
        node=node_currency,
        sources=district.reference_urls,
        additional_instruction=(
            "Check the page(s) for explicit mention of the school year; accept '2024-2025' or '2025-2026'. "
            "If multiple pages are used, at least one should clearly denote the school year applicable to enrollment/statistics."
        )
    )

    # Regional accreditation (Critical)
    node_accred = evaluator.add_leaf(
        id=f"District_{idx + 1}_Regional_Accreditation",
        desc="District is accredited by an appropriate regional accrediting agency",
        parent=district_node,
        critical=True
    )
    accred_claim = (
        f"The district '{district.district_name or 'UNKNOWN'}' is accredited by {district.accreditation_agency or 'an appropriate regional accrediting agency'}."
    )
    await evaluator.verify(
        claim=accred_claim,
        node=node_accred,
        sources=district.reference_urls,
        additional_instruction=(
            "Confirm accreditation by a recognized K-12 regional agency (e.g., Cognia/AdvancED, MSA-CESS, WASC, etc.). "
            "District-level accreditation or explicit district-wide recognition is acceptable."
        )
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
    Evaluate an answer for identifying three qualifying U.S. public school districts.
    """
    # Initialize evaluator with root node (parallel aggregation)
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

    # Extract districts
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Use only the first three districts, pad if fewer
    districts = _first_k(extracted.districts, 3)
    states = _states_list(districts)

    # Build and verify each district subtree
    # Parent node for identified districts
    top_node = evaluator.add_parallel(
        id="Identify_Three_Qualifying_Districts",
        desc="Identify 3 large public school districts in the United States, each from a different state, with enrollment between 80,000-95,000 students, meeting all specified criteria",
        parent=root,
        critical=False
    )

    for i, dist in enumerate(districts):
        await verify_district(evaluator, top_node, dist, i, states)

    # Optional: record custom info to help debugging
    evaluator.add_custom_info(
        {
            "extracted_states": states,
            "acceptable_school_years": list(ACCEPTABLE_SCHOOL_YEARS),
            "district_count_extracted": len(extracted.districts)
        },
        info_type="debug_meta"
    )

    # Return summary
    return evaluator.get_summary()