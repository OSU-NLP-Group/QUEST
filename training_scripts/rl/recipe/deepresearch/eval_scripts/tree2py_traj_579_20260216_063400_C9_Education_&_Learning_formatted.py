import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_search_2026"
TASK_DESCRIPTION = """
Several large school districts across the United States are currently conducting superintendent searches for positions beginning in 2026. These districts are seeking highly qualified educational leaders with specific credentials, extensive experience, and proven track records in school administration.

Your task is to identify three large U.S. public school districts that are currently conducting active superintendent searches meeting ALL of the following criteria:

District Requirements:
- The district must be located in the United States
- The district must have a current student enrollment of at least 50,000 students
- At least two of the three districts must be located in different states

Superintendent Position Requirements:
- The position must require a minimum of a Master's degree in educational administration, educational leadership, or a closely related field from an accredited institution
- The position must require a valid superintendent certification or license (state-specific certification requirements apply)
- The position must require a minimum of 3 years of school administrative leadership experience (such as experience as a principal, assistant superintendent, or equivalent administrative role)
- At least one of the three districts must explicitly require or strongly prefer central office administrative experience

Timeline Requirements:
- The application deadline for the superintendent position must fall between December 1, 2025 and March 31, 2026
- The anticipated or target start date for the new superintendent must be between June 1, 2026 and August 1, 2026

For each of the three districts you identify, provide:
1. The full official name of the school district
2. The state where the district is located
3. The current student enrollment (approximate number)
4. The application deadline date
5. The anticipated start date for the new superintendent
6. A brief summary of the key qualification and experience requirements
7. Reference URL(s) from official district sources, official job postings, or recognized educational job boards that verify the superintendent search details

Your answer must be based on verifiable, publicly available information from official sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None  # Keep as string to maximize compatibility
    application_deadline: Optional[str] = None
    start_date: Optional[str] = None
    qualifications_summary: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Optional textual statements to aid verification (extracted verbatim from answer)
    degree_requirement_text: Optional[str] = None
    certificate_requirement_text: Optional[str] = None
    leadership_requirement_text: Optional[str] = None
    central_office_requirement_text: Optional[str] = None


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to three U.S. public school districts that the answer claims are currently conducting superintendent searches for positions starting in 2026. For each district, extract the following fields EXACTLY as stated in the answer:

    - district_name: Full official name of the school district
    - state: The U.S. state where the district is located (use the full state name or common abbreviation as provided)
    - enrollment: Current student enrollment (approximate number or descriptive text as provided)
    - application_deadline: The exact application deadline date as stated
    - start_date: The anticipated/target start date as stated
    - qualifications_summary: A brief summary of key qualification and experience requirements as presented in the answer
    - reference_urls: A list of URLs from official district sources, official job postings, or recognized educational job boards that the answer cites for this district. Include all URLs mentioned for this district. Extract actual URLs; keep them as-is. If none are provided, return an empty list.

    Additionally, if the answer text explicitly mentions any of the following requirement statements for a district, extract them verbatim:
    - degree_requirement_text: The text indicating a minimum of a Master's degree requirement (e.g., "Master's degree in educational leadership required")
    - certificate_requirement_text: The text indicating a valid superintendent certification/license requirement
    - leadership_requirement_text: The text indicating a minimum number of years of school administrative leadership experience (e.g., "at least 3 years")
    - central_office_requirement_text: The text indicating central office administrative experience is required or strongly preferred

    Return a JSON object with a top-level field 'districts' that is an array of district objects with the above fields. If any field is missing for a district, set it to null (or empty list for URLs). Do not infer or invent any data not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _has_urls(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0)


def _date_parse(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    # Try multiple common date formats
    formats = [
        "%B %d, %Y",      # January 15, 2026
        "%b %d, %Y",      # Jan 15, 2026
        "%Y-%m-%d",       # 2026-01-15
        "%m/%d/%Y",       # 01/15/2026
        "%m-%d-%Y",       # 01-15-2026
        "%d %B %Y",       # 15 January 2026
        "%d %b %Y",       # 15 Jan 2026
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Try to handle cases like "March 2026" or "June 1 2026"
    rough_formats = [
        "%B %Y",
        "%b %Y",
        "%B %d %Y",
        "%b %d %Y",
    ]
    for fmt in rough_formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _in_date_range(date_str: Optional[str], start: datetime, end: datetime) -> bool:
    d = _date_parse(date_str)
    if d is None:
        return False
    return start <= d <= end


def _states_diversity(states: List[Optional[str]]) -> bool:
    # At least two of the three districts must be in different states
    normalized = [(_normalize_text(s).upper() if s else "") for s in states]
    unique = set([s for s in normalized if s != ""])
    return len(unique) >= 2


# --------------------------------------------------------------------------- #
# Verification functions (build subtrees and verify leaves)                   #
# --------------------------------------------------------------------------- #
async def verify_district(
    evaluator: Evaluator,
    root_parent,
    district: DistrictItem,
    idx: int
) -> None:
    """
    Build verification subtree and run checks for one district.
    """
    di = district  # alias
    name = _normalize_text(di.district_name) or f"District #{idx + 1}"
    state = _normalize_text(di.state)
    urls = di.reference_urls

    # District node (parallel, non-critical to allow partial credit per district)
    district_node = evaluator.add_parallel(
        id=f"District_{idx + 1}",
        desc=f"{['First','Second','Third'][idx]} qualifying district meeting all specified criteria",
        parent=root_parent,
        critical=False
    )

    # 1) Qualifications (critical, all children are critical)
    qual_node = evaluator.add_parallel(
        id=f"District_{idx + 1}_Qualifications",
        desc=f"Qualification requirements for the superintendent position in {name}",
        parent=district_node,
        critical=True
    )

    # Degree requirement
    qual_degree_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Qual_Degree",
        desc="Position requires minimum of Master's degree in educational administration, educational leadership, or related field from accredited institution",
        parent=qual_node,
        critical=True
    )
    degree_claim = (
        f"The superintendent position in {name} requires at least a Master's degree in educational administration, "
        f"educational leadership, or a closely related field from an accredited institution."
    )
    add_ins_degree = (
        "Verify on the provided official posting or job board page that a minimum of a Master’s degree (e.g., "
        "Master's in educational leadership/administration or related) is required. Accept reasonable synonyms "
        "like M.Ed., MPA with educational leadership focus, or similar graduate degree explicitly stated."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=qual_degree_leaf,
        sources=urls,
        additional_instruction=add_ins_degree
    )

    # Superintendent certificate/license requirement
    qual_cert_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Qual_Certificate",
        desc="Position requires valid state-specific superintendent certification or license",
        parent=qual_node,
        critical=True
    )
    cert_claim = (
        f"The superintendent position in {name} requires a valid state-specific superintendent certification or license."
    )
    add_ins_cert = (
        "Verify the posting explicitly mentions a superintendent certificate/license (e.g., state superintendent "
        "endorsement, school administrator license at the superintendent level). Allow state-specific phrasing."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=qual_cert_leaf,
        sources=urls,
        additional_instruction=add_ins_cert
    )

    # Qualifications references existence (critical)
    qual_refs_node = evaluator.add_custom_node(
        result=_has_urls(urls),
        id=f"District_{idx + 1}_Qual_References",
        desc=f"URL reference supporting qualification requirements for {name}",
        parent=qual_node,
        critical=True
    )

    # 2) Timeline (critical)
    timeline_node = evaluator.add_parallel(
        id=f"District_{idx + 1}_Timeline",
        desc=f"Timeline requirements for {name} superintendent search",
        parent=district_node,
        critical=True
    )

    # Application deadline leaf
    app_deadline_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Timeline_AppDeadline",
        desc="Application deadline falls between December 1, 2025 and March 31, 2026",
        parent=timeline_node,
        critical=True
    )
    app_claim = (
        f"The application deadline for the superintendent position in {name} is '{_normalize_text(di.application_deadline)}', "
        f"which falls between December 1, 2025 and March 31, 2026."
        if di.application_deadline else
        f"The superintendent position in {name} has an application deadline that falls between December 1, 2025 and March 31, 2026."
    )
    add_ins_app = (
        "Confirm the posting's application deadline date on the referenced page. Also judge whether the date lies "
        "within the specified window [Dec 1, 2025, Mar 31, 2026]. If the posted deadline is outside the window or "
        "no deadline is present, mark incorrect."
    )
    await evaluator.verify(
        claim=app_claim,
        node=app_deadline_leaf,
        sources=urls,
        additional_instruction=add_ins_app
    )

    # Start date leaf
    start_date_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Timeline_StartDate",
        desc="Anticipated start date for new superintendent is between June 1, 2026 and August 1, 2026",
        parent=timeline_node,
        critical=True
    )
    start_claim = (
        f"The anticipated start date for the new superintendent in {name} is '{_normalize_text(di.start_date)}', "
        f"which falls between June 1, 2026 and August 1, 2026."
        if di.start_date else
        f"The anticipated start date for the new superintendent in {name} falls between June 1, 2026 and August 1, 2026."
    )
    add_ins_start = (
        "Confirm the posting's anticipated/target start date and judge whether it lies within [Jun 1, 2026, Aug 1, 2026]. "
        "If the page indicates a different start period or none, mark incorrect."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_date_leaf,
        sources=urls,
        additional_instruction=add_ins_start
    )

    # Timeline references existence (critical)
    timeline_refs_node = evaluator.add_custom_node(
        result=_has_urls(urls),
        id=f"District_{idx + 1}_Timeline_References",
        desc=f"URL reference supporting timeline information for {name}",
        parent=timeline_node,
        critical=True
    )

    # 3) Characteristics (critical)
    char_node = evaluator.add_parallel(
        id=f"District_{idx + 1}_Characteristics",
        desc=f"District characteristics and enrollment requirements for {name}",
        parent=district_node,
        critical=True
    )

    # Location leaf
    char_location_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Char_Location",
        desc="District is located in the United States",
        parent=char_node,
        critical=True
    )
    loc_claim = (
        f"{name} is located in {state}, United States."
        if state else
        f"{name} is located in the United States."
    )
    add_ins_loc = (
        "Verify the district is a U.S. public school district. If the page indicates a U.S. location (state, city, "
        "district profile), mark supported."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=char_location_leaf,
        sources=urls,
        additional_instruction=add_ins_loc
    )

    # Enrollment leaf
    char_enrollment_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Char_Enrollment",
        desc="District has student enrollment of at least 50,000 students",
        parent=char_node,
        critical=True
    )
    enr_text = _normalize_text(di.enrollment)
    enr_claim = (
        f"{name} has a student enrollment of approximately {enr_text} students, which is at least 50,000."
        if enr_text else
        f"{name} has a student enrollment of at least 50,000 students."
    )
    add_ins_enr = (
        "Check official district profiles, reports, or the posting itself for enrollment. The exact number can be "
        "approximate; accept ranges/approximate figures as long as they are clearly ≥ 50,000."
    )
    await evaluator.verify(
        claim=enr_claim,
        node=char_enrollment_leaf,
        sources=urls,
        additional_instruction=add_ins_enr
    )

    # Characteristics references existence (critical)
    char_refs_node = evaluator.add_custom_node(
        result=_has_urls(urls),
        id=f"District_{idx + 1}_Char_References",
        desc=f"URL reference supporting district characteristics for {name}",
        parent=char_node,
        critical=True
    )

    # 4) Experience (non-critical parent to allow partial credit and central-office optionality per district)
    exp_node = evaluator.add_parallel(
        id=f"District_{idx + 1}_Experience",
        desc=f"Experience requirements for the superintendent position in {name}",
        parent=district_node,
        critical=False
    )

    # Leadership experience (critical leaf under non-critical parent)
    exp_lead_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Exp_Leadership",
        desc="Position requires minimum of 3 years of school administrative leadership experience",
        parent=exp_node,
        critical=True
    )
    lead_claim = (
        f"The superintendent position in {name} requires a minimum of 3 years of school administrative leadership experience "
        f"(e.g., principal, assistant superintendent, or equivalent)."
    )
    add_ins_lead = (
        "Verify the posting explicitly states a minimum experience threshold. If it states 3+ years, 5+ years, or similar, "
        "this satisfies the '≥ 3 years' requirement."
    )
    await evaluator.verify(
        claim=lead_claim,
        node=exp_lead_leaf,
        sources=urls,
        additional_instruction=add_ins_lead
    )

    # Central office experience (non-critical leaf)
    exp_central_leaf = evaluator.add_leaf(
        id=f"District_{idx + 1}_Exp_CentralOffice",
        desc="Position requires or prefers central office experience",
        parent=exp_node,
        critical=False
    )
    central_claim = (
        f"The superintendent position in {name} explicitly requires or strongly prefers central office administrative experience."
    )
    add_ins_central = (
        "Verify language like 'central office experience required/strongly preferred', 'district-level administrative experience', "
        "or equivalent phrasing indicating central office background."
    )
    await evaluator.verify(
        claim=central_claim,
        node=exp_central_leaf,
        sources=urls,
        additional_instruction=add_ins_central
    )

    # Experience references existence (critical leaf under non-critical parent)
    exp_refs_node = evaluator.add_custom_node(
        result=_has_urls(urls),
        id=f"District_{idx + 1}_Exp_References",
        desc=f"URL reference supporting experience requirements for {name}",
        parent=exp_node,
        critical=True
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the superintendent searches task.
    """
    # Initialize evaluator (root non-critical to allow partial scoring; rubric's root critical adjusted for framework consistency)
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

    # Extract districts from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Keep only the first three districts; pad with empty placeholders if fewer provided
    districts: List[DistrictItem] = list(extracted.districts[:3])
    while len(districts) < 3:
        districts.append(DistrictItem())

    # Build verification subtrees for each district
    for i in range(3):
        await verify_district(evaluator, root, districts[i], i)

    # Cross-district constraints (critical)
    cross_node = evaluator.add_parallel(
        id="Cross_District_Constraints",
        desc="Cross-district requirements that must be satisfied by the set of three identified districts",
        parent=root,
        critical=True
    )

    # State diversity: at least two districts in different states
    states = [d.state for d in districts]
    state_diversity_result = _states_diversity(states)
    evaluator.add_custom_node(
        result=state_diversity_result,
        id="State_Diversity",
        desc="At least two of the three identified districts are located in different states",
        parent=cross_node,
        critical=True
    )

    # Central office requirement across set: at least one district explicitly requires or strongly prefers central office experience
    # We check statuses of the previously created per-district central office leaf nodes.
    def _central_office_any_passed() -> bool:
        passed_any = False
        for i in range(3):
            node = evaluator.find_node(f"District_{i + 1}_Exp_CentralOffice")
            if node and node.status == "passed":
                passed_any = True
                break
        return passed_any

    central_office_any = _central_office_any_passed()
    evaluator.add_custom_node(
        result=central_office_any,
        id="Central_Office_Requirement",
        desc="At least one of the three identified districts explicitly requires or strongly prefers central office administrative experience for the superintendent position",
        parent=cross_node,
        critical=True
    )

    # Add custom info about evaluation windows used
    evaluator.add_custom_info(
        info={
            "application_deadline_window": {"start": "2025-12-01", "end": "2026-03-31"},
            "start_date_window": {"start": "2026-06-01", "end": "2026-08-01"},
        },
        info_type="windows",
        info_name="timeline_windows"
    )

    return evaluator.get_summary()