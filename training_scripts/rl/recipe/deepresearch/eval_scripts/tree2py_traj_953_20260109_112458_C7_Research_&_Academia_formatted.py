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
TASK_ID = "cs_phd_program_info_top50_one_university"
TASK_DESCRIPTION = """
You are preparing to apply for computer science PhD programs in the United States. To make an informed decision, you need comprehensive information about potential programs.

Task: Select ONE U.S. research university whose computer science or EECS graduate program is ranked in the top 50 nationally (according to U.S. News, QS, or Times Higher Education rankings). For your chosen university, research and provide the following 13 pieces of specific, verifiable information, each supported by an official source URL:

1. Program Ranking: Confirm the CS/EECS program's top-50 ranking with the specific ranking source and year
2. PhD Stipend: Monthly or annual stipend amount for PhD students in teaching or research assistantships for the 2025-26 or 2026-27 academic year
3. Faculty Teaching Load: Standard teaching load for tenure-track assistant professors (courses per semester or year)
4. Office Hours Policy: Minimum weekly office hours required for full-time faculty members
5. Sabbatical Policy: Eligibility period (years of service) and compensation terms for sabbatical leave
6. PhD Timeline: Expected or average time to complete the PhD degree
7. Department Size: Number of faculty members or number of research areas in the department
8. Travel Funding: Availability and typical amount of conference travel grants for PhD students
9. Parking Costs: Annual cost of parking permits for faculty/staff
10. Health Insurance: Confirmation of health insurance coverage for PhD students
11. Computing Resources: Availability of high-performance computing (HPC) cluster or research computing facilities
12. IRB Process: Existence and description of the Institutional Review Board process for human subjects research
13. Application Deadline: PhD program application deadline for Fall 2025 or Fall 2026 admission

Each piece of information must include a reference to an official university webpage, policy document, or departmental announcement.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RankingInfo(BaseModel):
    source: Optional[str] = None  # e.g., "U.S. News", "QS", "Times Higher Education"
    year: Optional[str] = None    # e.g., "2025"
    position: Optional[str] = None  # e.g., "Top 50", "47"
    ranking_url: Optional[str] = None


class StipendInfo(BaseModel):
    amount: Optional[str] = None  # string to allow ranges or descriptive values
    period: Optional[str] = None  # e.g., "per month", "per year"
    academic_year: Optional[str] = None  # "2025-26" or "2026-27"
    source_urls: List[str] = Field(default_factory=list)


class TeachingLoadInfo(BaseModel):
    load: Optional[str] = None  # e.g., "2 courses per semester", "2-2 load"
    source_urls: List[str] = Field(default_factory=list)


class OfficeHoursInfo(BaseModel):
    min_hours_per_week: Optional[str] = None  # e.g., "3 hours/week"
    source_urls: List[str] = Field(default_factory=list)


class SabbaticalInfo(BaseModel):
    eligibility_years: Optional[str] = None  # e.g., "6 years of service"
    compensation: Optional[str] = None       # e.g., "full pay for one semester"
    source_urls: List[str] = Field(default_factory=list)


class PhDTimelineInfo(BaseModel):
    duration: Optional[str] = None  # e.g., "5-6 years"
    source_urls: List[str] = Field(default_factory=list)


class DepartmentSizeInfo(BaseModel):
    faculty_count: Optional[str] = None            # e.g., "75"
    research_areas_count: Optional[str] = None     # e.g., "12"
    source_urls: List[str] = Field(default_factory=list)


class TravelFundingInfo(BaseModel):
    availability: Optional[str] = None     # e.g., "Available for conference travel"
    typical_amount: Optional[str] = None   # e.g., "$1,500 per year"
    source_urls: List[str] = Field(default_factory=list)


class ParkingCostsInfo(BaseModel):
    annual_cost: Optional[str] = None      # e.g., "$720/year"
    permit_name: Optional[str] = None      # e.g., "Faculty/Staff A permit"
    source_urls: List[str] = Field(default_factory=list)


class HealthInsuranceInfo(BaseModel):
    coverage: Optional[str] = None         # e.g., "Graduate students receive health insurance"
    source_urls: List[str] = Field(default_factory=list)


class ComputingResourcesInfo(BaseModel):
    facility_name: Optional[str] = None    # e.g., "HPC cluster Nimbus"
    hpc_available: Optional[str] = None    # e.g., "Yes"
    source_urls: List[str] = Field(default_factory=list)


class IRBInfo(BaseModel):
    description: Optional[str] = None      # brief description of IRB process
    source_urls: List[str] = Field(default_factory=list)


class ApplicationDeadlineInfo(BaseModel):
    term: Optional[str] = None             # "Fall 2025" or "Fall 2026"
    deadline_date: Optional[str] = None    # e.g., "December 15, 2024"
    source_urls: List[str] = Field(default_factory=list)


class ProgramData(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None
    # For confirming U.S. research university status/location via official/authoritative URLs
    us_research_university_urls: List[str] = Field(default_factory=list)

    # Ranking (used both for eligibility and item #1)
    ranking: Optional[RankingInfo] = None

    # 13 items
    phd_stipend: Optional[StipendInfo] = None
    faculty_teaching_load: Optional[TeachingLoadInfo] = None
    office_hours_policy: Optional[OfficeHoursInfo] = None
    sabbatical_policy: Optional[SabbaticalInfo] = None
    phd_timeline: Optional[PhDTimelineInfo] = None
    department_size: Optional[DepartmentSizeInfo] = None
    travel_funding: Optional[TravelFundingInfo] = None
    parking_costs: Optional[ParkingCostsInfo] = None
    health_insurance: Optional[HealthInsuranceInfo] = None
    computing_resources: Optional[ComputingResourcesInfo] = None
    irb_process: Optional[IRBInfo] = None
    application_deadline: Optional[ApplicationDeadlineInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_data() -> str:
    return """
    You must extract structured information for a SINGLE chosen U.S. research university CS/EECS PhD program from the answer.

    Identify ONE chosen university/program and extract:
    - university_name: The full name of the university selected.
    - program_name: The specific CS or EECS graduate program name (e.g., "Ph.D. in Computer Science").
    - program_url: An official program page URL (department/program site) that directly mentions the PhD in CS/EECS.
    - us_research_university_urls: Official or authoritative URLs confirming the university is a U.S. research university and located in the U.S. Accept Carnegie Classification pages, AAU membership pages, or official university pages. Return all such URLs mentioned in the answer.

    Ranking details (used for eligibility and item #1):
    - ranking.source: The ranking organization (must be one of "U.S. News", "QS", or "Times Higher Education").
    - ranking.year: The ranking year stated in the answer.
    - ranking.position: The position or explicit "top 50" phrasing used in the answer.
    - ranking.ranking_url: The URL to the ranking page cited in the answer.

    Extract all 13 required items, each with official source URLs:

    1) phd_stipend:
       - amount: stipend amount (string, allow ranges)
       - period: "per month" or "per year"
       - academic_year: must be "2025-26" or "2026-27" if specified
       - source_urls: official URLs

    2) faculty_teaching_load:
       - load: standard teaching load for tenure-track assistant professors
       - source_urls

    3) office_hours_policy:
       - min_hours_per_week: minimum weekly office hours required for full-time faculty
       - source_urls

    4) sabbatical_policy:
       - eligibility_years
       - compensation
       - source_urls

    5) phd_timeline:
       - duration: expected/average time to complete the PhD
       - source_urls

    6) department_size:
       - faculty_count OR research_areas_count (provide whichever is specified)
       - source_urls

    7) travel_funding:
       - availability: whether conference travel funding exists
       - typical_amount: typical or maximum amounts if stated
       - source_urls

    8) parking_costs:
       - annual_cost
       - permit_name
       - source_urls

    9) health_insurance:
       - coverage: statement confirming coverage for PhD students
       - source_urls

    10) computing_resources:
       - facility_name: HPC or research computing facility name
       - hpc_available: "Yes"/"No" or description
       - source_urls

    11) irb_process:
       - description: brief summary of IRB process existence
       - source_urls

    12) application_deadline:
       - term: "Fall 2025" or "Fall 2026"
       - deadline_date: e.g., "December 15, 2024"
       - source_urls

    RULES:
    - Extract ONLY what is explicitly present in the answer.
    - For each item, include only official URLs (university domains, official policy pages, authoritative listings). If no official URL is cited, return an empty list for source_urls.
    - If a field is not present in the answer, return null for that field (or empty list for source_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]], fallback: Optional[str] = None) -> List[str]:
    """Return URLs list; if empty and a single fallback is provided, return [fallback] if not None."""
    if urls and len(urls) > 0:
        return urls
    return [fallback] if fallback else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_program_eligibility(evaluator: Evaluator, parent_node, data: ProgramData) -> None:
    """
    Build and verify the 'program_eligibility' critical parallel node.
    Children:
      - single_program_selected
      - us_research_university
      - cs_or_eecs_phd_program
      - top50_ranking_with_source_and_year
    """
    elig_node = evaluator.add_parallel(
        id="program_eligibility",
        desc="Chosen program/university meets the selection requirements (one program; U.S. research university; CS/EECS PhD; top-50 ranking)",
        parent=parent_node,
        critical=True,
    )

    # 1) Single program selected
    single_node = evaluator.add_leaf(
        id="single_program_selected",
        desc="Response selects exactly ONE university/program (not multiple options)",
        parent=elig_node,
        critical=True,
    )
    single_claim = (
        "The answer selects exactly one distinct university/program and does not present multiple alternatives."
    )
    await evaluator.verify(
        claim=single_claim,
        node=single_node,
        additional_instruction=(
            "Examine the provided answer text. Accept only if exactly one specific institution/program is chosen. "
            "If the answer lists or compares multiple universities/programs or provides several options, mark incorrect."
        )
    )

    # 2) U.S. research university confirmation (with sources)
    usru_node = evaluator.add_leaf(
        id="us_research_university",
        desc="University is a U.S. research university (supported by an official or authoritative listing/source URL)",
        parent=elig_node,
        critical=True,
    )
    usru_claim = (
        f"The chosen university '{data.university_name or 'the selected university'}' is a U.S. research university located within the United States."
    )
    usru_sources = _safe_sources(data.us_research_university_urls, fallback=data.program_url)
    await evaluator.verify(
        claim=usru_claim,
        node=usru_node,
        sources=usru_sources,
        additional_instruction=(
            "Use authoritative pages (e.g., Carnegie Classification R1/R2, AAU membership pages, or official university pages) "
            "to confirm that this institution is a U.S. research university and located in the United States. "
            "If no official/authoritative URL is provided, mark incorrect."
        )
    )

    # 3) CS/EECS PhD program existence (with sources)
    phdprog_node = evaluator.add_leaf(
        id="cs_or_eecs_phd_program",
        desc="Selected program is a CS/EECS graduate program that offers a PhD (supported by an official program URL)",
        parent=elig_node,
        critical=True,
    )
    phdprog_claim = (
        f"The selected program '{data.program_name or 'the chosen program'}' is a Computer Science or EECS graduate program and explicitly offers a PhD degree."
    )
    await evaluator.verify(
        claim=phdprog_claim,
        node=phdprog_node,
        sources=_safe_sources([data.program_url] if data.program_url else []),
        additional_instruction=(
            "Verify the program page explicitly offers a PhD track in Computer Science or EECS (or closely related), "
            "not just MS/other degrees. If no official program URL is provided, mark incorrect."
        )
    )

    # 4) Top-50 ranking confirmation (with source and year)
    top50_node = evaluator.add_leaf(
        id="top50_ranking_with_source_and_year",
        desc="Program is ranked in the national top 50 by U.S. News, QS, or Times Higher Education, explicitly stating the ranking source and year, with a supporting URL",
        parent=elig_node,
        critical=True,
    )
    r_src = data.ranking.source if data.ranking else None
    r_year = data.ranking.year if data.ranking else None
    top50_claim = (
        f"The program is ranked in the national top 50 by {r_src or 'a recognized source'} in {r_year or 'the cited year'}."
    )
    await evaluator.verify(
        claim=top50_claim,
        node=top50_node,
        sources=_safe_sources([data.ranking.ranking_url] if (data.ranking and data.ranking.ranking_url) else []),
        additional_instruction=(
            "Check the cited ranking page to confirm both the source (U.S. News, QS, or THE) and the specific year are stated in the answer, "
            "and that the program ranks within the top 50. If the answer fails to specify source AND year, or no valid ranking URL is provided, mark incorrect."
        )
    )


async def verify_required_items(evaluator: Evaluator, parent_node, data: ProgramData) -> None:
    """
    Build and verify the 'required_13_information_items' critical parallel node.
    Each child is a critical leaf with a claim verified against official sources.
    """
    items_node = evaluator.add_parallel(
        id="required_13_information_items",
        desc="Provides all 13 specified pieces of information for the chosen program, each supported by an official source URL",
        parent=parent_node,
        critical=True,
    )

    claims_to_verify: List[
        tuple[str, List[str] | str | None, Any, Optional[str]]
    ] = []

    # 1) Program ranking details
    node_1 = evaluator.add_leaf(
        id="1_program_ranking_details",
        desc="Confirms top-50 program ranking including ranking source and year, with official/authoritative source URL",
        parent=items_node,
        critical=True,
    )
    r_src = data.ranking.source if data.ranking else None
    r_year = data.ranking.year if data.ranking else None
    r_pos = data.ranking.position if data.ranking else None
    claim_1 = (
        f"The program's top-50 ranking is confirmed by {r_src or 'a recognized source'} for {r_year or 'the cited year'} "
        f"(position: {r_pos or 'within top 50'})."
    )
    claims_to_verify.append((
        claim_1,
        _safe_sources([data.ranking.ranking_url] if (data.ranking and data.ranking.ranking_url) else []),
        node_1,
        "Confirm the page shows the program in the national top 50 and that the answer includes both source and year. "
        "If no official ranking URL is provided, mark incorrect."
    ))

    # 2) PhD stipend (2025–26 or 2026–27)
    node_2 = evaluator.add_leaf(
        id="2_phd_stipend",
        desc="Provides PhD TA/RA stipend amount (monthly or annual) for 2025–26 or 2026–27, with official source URL",
        parent=items_node,
        critical=True,
    )
    st = data.phd_stipend or StipendInfo()
    claim_2 = (
        f"The PhD TA/RA stipend amount is {st.amount or 'unspecified'} {st.period or ''} for the {st.academic_year or 'required'} academic year."
    )
    claims_to_verify.append((
        claim_2,
        _safe_sources(st.source_urls),
        node_2,
        "Accept only if the official source page supports the stipend amount and the academic year is 2025–26 or 2026–27. "
        "Reject if no official source URL is provided or if the timeframe does not match."
    ))

    # 3) Faculty teaching load
    node_3 = evaluator.add_leaf(
        id="3_faculty_teaching_load",
        desc="States standard teaching load for tenure-track assistant professors (courses per semester or year), with official policy/source URL",
        parent=items_node,
        critical=True,
    )
    tl = data.faculty_teaching_load or TeachingLoadInfo()
    claim_3 = f"The standard teaching load for tenure-track assistant professors is {tl.load or 'unspecified'}."
    claims_to_verify.append((
        claim_3,
        _safe_sources(tl.source_urls),
        node_3,
        "Verify the policy page states the typical teaching load for tenure-track assistant professors. Reject if no official source URL."
    ))

    # 4) Office hours policy
    node_4 = evaluator.add_leaf(
        id="4_office_hours_policy",
        desc="States minimum weekly office hours required for full-time faculty, with official policy/source URL",
        parent=items_node,
        critical=True,
    )
    oh = data.office_hours_policy or OfficeHoursInfo()
    claim_4 = f"The minimum weekly office hours required for full-time faculty is {oh.min_hours_per_week or 'unspecified'}."
    claims_to_verify.append((
        claim_4,
        _safe_sources(oh.source_urls),
        node_4,
        "Confirm the official policy page specifies a minimum weekly office hours requirement. Reject if no official source URL."
    ))

    # 5) Sabbatical policy
    node_5 = evaluator.add_leaf(
        id="5_sabbatical_policy",
        desc="Provides sabbatical eligibility period (years of service) and compensation terms, with official policy/source URL",
        parent=items_node,
        critical=True,
    )
    sb = data.sabbatical_policy or SabbaticalInfo()
    claim_5 = (
        f"Sabbatical eligibility requires {sb.eligibility_years or 'unspecified'} of service and compensation terms are {sb.compensation or 'unspecified'}."
    )
    claims_to_verify.append((
        claim_5,
        _safe_sources(sb.source_urls),
        node_5,
        "Verify eligibility period and compensation terms on the official policy page. Reject if no official source URL."
    ))

    # 6) PhD timeline
    node_6 = evaluator.add_leaf(
        id="6_phd_timeline",
        desc="Provides expected or average time to complete the PhD, with official source URL",
        parent=items_node,
        critical=True,
    )
    pt = data.phd_timeline or PhDTimelineInfo()
    claim_6 = f"The expected/average time to complete the PhD is {pt.duration or 'unspecified'}."
    claims_to_verify.append((
        claim_6,
        _safe_sources(pt.source_urls),
        node_6,
        "Verify the official page states expected/average time-to-degree for the PhD. Reject if no official source URL."
    ))

    # 7) Department size
    node_7 = evaluator.add_leaf(
        id="7_department_size",
        desc="Provides department size as either number of faculty members OR number of research areas, with official source URL",
        parent=items_node,
        critical=True,
    )
    ds = data.department_size or DepartmentSizeInfo()
    if ds.faculty_count:
        claim_7 = f"The department has {ds.faculty_count} faculty members."
    elif ds.research_areas_count:
        claim_7 = f"The department lists {ds.research_areas_count} research areas."
    else:
        claim_7 = "The department size (faculty count or research areas count) is unspecified."
    claims_to_verify.append((
        claim_7,
        _safe_sources(ds.source_urls),
        node_7,
        "Confirm the official page states either faculty count or the number of research areas. Reject if no official source URL."
    ))

    # 8) Travel funding
    node_8 = evaluator.add_leaf(
        id="8_travel_funding",
        desc="Documents availability of PhD conference travel funding and typical amount(s), with official source URL",
        parent=items_node,
        critical=True,
    )
    tf = data.travel_funding or TravelFundingInfo()
    claim_8 = (
        f"PhD students have conference travel funding available; typical amount is {tf.typical_amount or 'unspecified'}."
    )
    claims_to_verify.append((
        claim_8,
        _safe_sources(tf.source_urls),
        node_8,
        "Verify availability and typical/maximum amounts for conference travel on official pages. Reject if no official source URL."
    ))

    # 9) Parking costs
    node_9 = evaluator.add_leaf(
        id="9_parking_costs",
        desc="Provides annual parking permit cost for faculty/staff, with official source URL",
        parent=items_node,
        critical=True,
    )
    pk = data.parking_costs or ParkingCostsInfo()
    claim_9 = (
        f"The annual parking permit cost for faculty/staff is {pk.annual_cost or 'unspecified'}"
        f"{(' for ' + pk.permit_name) if pk.permit_name else ''}."
    )
    claims_to_verify.append((
        claim_9,
        _safe_sources(pk.source_urls),
        node_9,
        "Verify the annual cost for faculty/staff permits on the official transportation/parking page. Reject if no official source URL."
    ))

    # 10) Health insurance
    node_10 = evaluator.add_leaf(
        id="10_health_insurance",
        desc="Confirms health insurance coverage for PhD/graduate students, with official source URL",
        parent=items_node,
        critical=True,
    )
    hi = data.health_insurance or HealthInsuranceInfo()
    claim_10 = "Graduate/PhD students receive health insurance coverage."
    claims_to_verify.append((
        claim_10,
        _safe_sources(hi.source_urls),
        node_10,
        "Confirm coverage for graduate/PhD students on official benefits/insurance pages. Reject if no official source URL."
    ))

    # 11) Computing resources
    node_11 = evaluator.add_leaf(
        id="11_computing_resources",
        desc="Confirms availability of HPC cluster or research computing facilities, with official source URL",
        parent=items_node,
        critical=True,
    )
    cr = data.computing_resources or ComputingResourcesInfo()
    claim_11 = (
        f"The university provides HPC cluster or research computing facilities such as {cr.facility_name or 'unspecified facility'}."
    )
    claims_to_verify.append((
        claim_11,
        _safe_sources(cr.source_urls),
        node_11,
        "Verify the availability of HPC or research computing facilities on official IT/RCC pages. Reject if no official source URL."
    ))

    # 12) IRB process
    node_12 = evaluator.add_leaf(
        id="12_irb_process",
        desc="Confirms existence and describes the IRB process for human subjects research, with official source URL",
        parent=items_node,
        critical=True,
    )
    irb = data.irb_process or IRBInfo()
    claim_12 = (
        f"An Institutional Review Board (IRB) process exists for human subjects research; description: {irb.description or 'unspecified'}."
    )
    claims_to_verify.append((
        claim_12,
        _safe_sources(irb.source_urls),
        node_12,
        "Confirm the IRB process existence and description on the official university IRB/Compliance pages. Reject if no official source URL."
    ))

    # 13) Application deadline
    node_13 = evaluator.add_leaf(
        id="13_application_deadline",
        desc="Provides PhD application deadline for Fall 2025 or Fall 2026 admission, with official source URL",
        parent=items_node,
        critical=True,
    )
    ad = data.application_deadline or ApplicationDeadlineInfo()
    claim_13 = (
        f"The PhD program application deadline for {ad.term or 'required term'} admission is {ad.deadline_date or 'unspecified'}."
    )
    claims_to_verify.append((
        claim_13,
        _safe_sources(ad.source_urls),
        node_13,
        "Verify the PhD application deadline for Fall 2025 or Fall 2026 on the official department/graduate admissions page. "
        "Reject if no official source URL is provided."
    ))

    # Execute all verifications in parallel for efficiency
    await evaluator.batch_verify(claims_to_verify)


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
    Evaluate an answer for the single top-50 CS/EECS PhD program with 13 required items.
    Returns a standardized summary dictionary including the verification tree and final score.
    """
    # Initialize evaluator with sequential root (eligibility first, then details)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured data from the answer
    program_data = await evaluator.extract(
        prompt=prompt_extract_program_data(),
        template_class=ProgramData,
        extraction_name="program_data",
    )

    # Optional: Record basic custom info for convenience
    evaluator.add_custom_info(
        info={
            "university_name": program_data.university_name,
            "program_name": program_data.program_name,
            "program_url": program_data.program_url,
            "ranking_source": program_data.ranking.source if program_data.ranking else None,
            "ranking_year": program_data.ranking.year if program_data.ranking else None,
            "ranking_position": program_data.ranking.position if program_data.ranking else None,
            "ranking_url": program_data.ranking.ranking_url if program_data.ranking else None,
        },
        info_type="selection_overview",
        info_name="selection_overview"
    )

    # 1) Verify eligibility node first (critical; sequential gating)
    await verify_program_eligibility(evaluator, root, program_data)

    # 2) Verify all 13 required items (critical; will be skipped automatically if eligibility fails)
    await verify_required_items(evaluator, root, program_data)

    # Return structured result
    return evaluator.get_summary()