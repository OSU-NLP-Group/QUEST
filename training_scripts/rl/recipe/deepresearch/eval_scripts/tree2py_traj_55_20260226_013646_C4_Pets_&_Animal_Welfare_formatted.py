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
TASK_ID = "national_dog_show_2025_venue"
TASK_DESCRIPTION = (
    "Identify the venue that hosted the 2025 National Dog Show Presented by Purina. "
    "Provide the following information: (1) the facility name, (2) the city where it is located, "
    "(3) the state where it is located, (4) the total indoor exhibition space in square feet, "
    "(5) the typical spectator attendance capacity or documented attendance figures, and "
    "(6) the month(s) when the 2025 show was held."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Facility / host
    facility_name: Optional[str] = None
    facility_host_urls: List[str] = Field(default_factory=list)

    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # Indoor exhibition space
    indoor_exhibition_space_sqft: Optional[str] = None
    indoor_space_urls: List[str] = Field(default_factory=list)

    # Spectator capacity or attendance
    spectator_capacity_or_attendance: Optional[str] = None
    spectator_urls: List[str] = Field(default_factory=list)

    # Month(s) of 2025 show
    show_months_2025: Optional[str] = None
    show_months_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the specific venue information for the 2025 National Dog Show Presented by Purina as stated in the answer. 
    Return a single JSON object with the following fields (use null for any missing values, and [] for any missing URL lists):
    - facility_name: string | null  (the facility/venue name the answer claims hosted the 2025 National Dog Show Presented by Purina)
    - facility_host_urls: string[]  (all URLs cited that directly support that this facility hosted the 2025 event; e.g., official event website, press releases, reputable news articles)
    - city: string | null           (city where the facility is located, as stated in the answer)
    - state: string | null          (state where the facility is located, as stated in the answer; can be full name or postal abbreviation)
    - location_urls: string[]       (URLs that support the facility's address/location, city and/or state; e.g., venue site, Wikipedia, contact page)
    - indoor_exhibition_space_sqft: string | null  (the total indoor exhibition space figure as stated; keep the exact text, e.g., '240,000 sq ft', 'over 200,000 square feet', etc.)
    - indoor_space_urls: string[]   (URLs that support the indoor exhibition space figure)
    - spectator_capacity_or_attendance: string | null  (the spectator capacity or attendance figure as stated; keep the text including the number, e.g., 'seating for 6,000', 'attendance around 20,000')
    - spectator_urls: string[]      (URLs that support the spectator capacity or attendance figure; can be venue capacity or documented event attendance)
    - show_months_2025: string | null  (the month(s) when the 2025 show was held, as stated in the answer; keep the phrasing from the answer, e.g., 'November 2025', 'November', 'Nov. 16–17, 2025', 'Thanksgiving weekend 2025')
    - show_months_urls: string[]    (URLs that support the 2025 event dates/schedule which substantiate the stated month(s))
    
    STRICT RULES:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer any URL.
    - Return null for any field that is not explicitly stated in the answer. Do not guess.
    - For URL fields, include every URL that is associated with the specific field; include markdown link targets if present.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _contains_digit(s: Optional[str]) -> bool:
    if not _nonempty_text(s):
        return False
    return any(ch.isdigit() for ch in s)  # simple numeric presence check


def _pick_sources(*candidates: List[str]) -> List[str]:
    for lst in candidates:
        if lst and len(lst) > 0:
            return lst
    return []


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, ext: VenueExtraction) -> None:
    """
    Build verification tree according to rubric. 
    We create one critical parallel node that holds 6 critical sequential sub-nodes,
    each containing specific existence checks and a final URL-grounded verification leaf.
    """

    # Top-level critical node mirroring the rubric root
    top = evaluator.add_parallel(
        id="2025_National_Dog_Show_Venue_Identification",
        desc="Identify the venue that hosted the 2025 National Dog Show Presented by Purina and provide the required venue/show attributes requested in the question.",
        parent=root_node,
        critical=True
    )

    # 1) Facility Name (host of 2025 show)
    facility_group = evaluator.add_sequential(
        id="Facility_Name",
        desc="Provide the facility/venue name, and it must be the venue that hosted the 2025 National Dog Show Presented by Purina.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_text(ext.facility_name),
        id="Facility_Name_provided",
        desc="Facility name provided in the answer",
        parent=facility_group,
        critical=True
    )
    facility_host_sources = _pick_sources(ext.facility_host_urls)
    evaluator.add_custom_node(
        result=len(facility_host_sources) > 0,
        id="Facility_Name_has_sources",
        desc="At least one source URL is provided to support the hosting claim",
        parent=facility_group,
        critical=True
    )
    facility_verify = evaluator.add_leaf(
        id="Facility_Name_verify",
        desc="Provide the facility/venue name, and it must be the venue that hosted the 2025 National Dog Show Presented by Purina.",
        parent=facility_group,
        critical=True
    )
    fac_claim = f"The 2025 National Dog Show Presented by Purina was hosted at {ext.facility_name}."
    await evaluator.verify(
        claim=fac_claim,
        node=facility_verify,
        sources=facility_host_sources,
        additional_instruction="Verify that the provided sources explicitly indicate that this named facility hosted the 2025 National Dog Show Presented by Purina. Accept reasonable naming variants of the event (e.g., 'National Dog Show', 'National Dog Show Presented by Purina')."
    )

    # 2) City Location
    city_group = evaluator.add_sequential(
        id="City_Location",
        desc="Provide the city where the identified facility is located; the city must be correct for that facility.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_text(ext.city),
        id="City_Location_provided",
        desc="City provided in the answer",
        parent=city_group,
        critical=True
    )
    city_sources = _pick_sources(ext.location_urls, ext.facility_host_urls)
    evaluator.add_custom_node(
        result=len(city_sources) > 0,
        id="City_Location_has_sources",
        desc="At least one source URL provided to support the city location",
        parent=city_group,
        critical=True
    )
    city_verify = evaluator.add_leaf(
        id="City_Location_verify",
        desc="Provide the city where the identified facility is located; the city must be correct for that facility.",
        parent=city_group,
        critical=True
    )
    city_claim = f"The facility {ext.facility_name} is located in the city of {ext.city}."
    await evaluator.verify(
        claim=city_claim,
        node=city_verify,
        sources=city_sources,
        additional_instruction="Verify that the sources show the facility's address/city. Allow common variations or locality descriptors (e.g., 'Oaks' vs. nearby metro area mentions). Focus on the city-level locality stated in the answer."
    )

    # 3) State Location
    state_group = evaluator.add_sequential(
        id="State_Location",
        desc="Provide the state where the identified facility is located; the state must be correct for that facility.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_text(ext.state),
        id="State_Location_provided",
        desc="State provided in the answer",
        parent=state_group,
        critical=True
    )
    state_sources = _pick_sources(ext.location_urls, ext.facility_host_urls)
    evaluator.add_custom_node(
        result=len(state_sources) > 0,
        id="State_Location_has_sources",
        desc="At least one source URL provided to support the state location",
        parent=state_group,
        critical=True
    )
    state_verify = evaluator.add_leaf(
        id="State_Location_verify",
        desc="Provide the state where the identified facility is located; the state must be correct for that facility.",
        parent=state_group,
        critical=True
    )
    state_claim = f"The facility {ext.facility_name} is located in the U.S. state of {ext.state}."
    await evaluator.verify(
        claim=state_claim,
        node=state_verify,
        sources=state_sources,
        additional_instruction="Verify the state for the facility. Accept common state abbreviations vs. full name equivalence (e.g., 'PA' vs. 'Pennsylvania') as matching."
    )

    # 4) Indoor Exhibition Space (sq ft)
    space_group = evaluator.add_sequential(
        id="Indoor_Exhibition_Space_SqFt",
        desc="Provide the total indoor exhibition space in square feet for the identified facility; the value must be correct and expressed in sq ft (or clearly convertible to sq ft).",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_text(ext.indoor_exhibition_space_sqft),
        id="Indoor_Exhibition_Space_SqFt_provided",
        desc="Indoor exhibition space figure provided in the answer",
        parent=space_group,
        critical=True
    )
    space_sources = _pick_sources(ext.indoor_space_urls, ext.location_urls, ext.facility_host_urls)
    evaluator.add_custom_node(
        result=len(space_sources) > 0,
        id="Indoor_Exhibition_Space_SqFt_has_sources",
        desc="At least one source URL provided to support the indoor exhibition space figure",
        parent=space_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_digit(ext.indoor_exhibition_space_sqft),
        id="Indoor_Exhibition_Space_SqFt_has_number",
        desc="Indoor exhibition space includes a numeric figure",
        parent=space_group,
        critical=True
    )
    space_verify = evaluator.add_leaf(
        id="Indoor_Exhibition_Space_SqFt_verify",
        desc="Provide the total indoor exhibition space in square feet for the identified facility; the value must be correct and expressed in sq ft (or clearly convertible to sq ft).",
        parent=space_group,
        critical=True
    )
    space_claim = f"The total indoor exhibition space of {ext.facility_name} is {ext.indoor_exhibition_space_sqft}."
    await evaluator.verify(
        claim=space_claim,
        node=space_verify,
        sources=space_sources,
        additional_instruction="Confirm the total indoor exhibition space. Accept reasonable formatting variants (e.g., 'sq ft', 'square feet', commas, 'over ~X sq ft'). If the source uses other units (e.g., sq meters, acres), treat it as acceptable so long as the number clearly corresponds or is convertible to the stated figure."
    )

    # 5) Spectator Capacity or Attendance
    spec_group = evaluator.add_sequential(
        id="Spectator_Capacity_or_Attendance",
        desc="Provide either (a) typical spectator attendance capacity or (b) documented attendance figures for the event/venue; must include a quantitative figure and be correct for the cited context.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_text(ext.spectator_capacity_or_attendance),
        id="Spectator_Capacity_or_Attendance_provided",
        desc="Spectator capacity/attendance figure provided in the answer",
        parent=spec_group,
        critical=True
    )
    spectator_sources = _pick_sources(ext.spectator_urls, ext.facility_host_urls)
    evaluator.add_custom_node(
        result=len(spectator_sources) > 0,
        id="Spectator_Capacity_or_Attendance_has_sources",
        desc="At least one source URL provided to support the spectator capacity or attendance figure",
        parent=spec_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_digit(ext.spectator_capacity_or_attendance),
        id="Spectator_Capacity_or_Attendance_has_number",
        desc="Spectator capacity or attendance includes a numeric figure",
        parent=spec_group,
        critical=True
    )
    spectator_verify = evaluator.add_leaf(
        id="Spectator_Capacity_or_Attendance_verify",
        desc="Provide either (a) typical spectator attendance capacity or (b) documented attendance figures for the event/venue; must include a quantitative figure and be correct for the cited context.",
        parent=spec_group,
        critical=True
    )
    spectator_claim = (
        f"The spectator capacity or documented attendance for the venue/event is reported as "
        f"'{ext.spectator_capacity_or_attendance}'."
    )
    await evaluator.verify(
        claim=spectator_claim,
        node=spectator_verify,
        sources=spectator_sources,
        additional_instruction="Verify the numeric spectator figure. It can be the venue's seating capacity or a documented attendance figure for the National Dog Show; ensure the context matches what the figure refers to (venue capacity vs. event attendance). Minor rounding/formatting differences are acceptable."
    )

    # 6) Show Months (2025)
    months_group = evaluator.add_sequential(
        id="Show_Months_2025",
        desc="Provide the month(s) when the 2025 show was held; the month(s) must match the actual 2025 event dates.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_text(ext.show_months_2025),
        id="Show_Months_2025_provided",
        desc="Show month(s) for 2025 provided in the answer",
        parent=months_group,
        critical=True
    )
    months_sources = _pick_sources(ext.show_months_urls, ext.facility_host_urls)
    evaluator.add_custom_node(
        result=len(months_sources) > 0,
        id="Show_Months_2025_has_sources",
        desc="At least one source URL provided to support the 2025 show month(s)",
        parent=months_group,
        critical=True
    )
    months_verify = evaluator.add_leaf(
        id="Show_Months_2025_verify",
        desc="Provide the month(s) when the 2025 show was held; the month(s) must match the actual 2025 event dates.",
        parent=months_group,
        critical=True
    )
    months_claim = f"The 2025 National Dog Show Presented by Purina was held during {ext.show_months_2025}."
    await evaluator.verify(
        claim=months_claim,
        node=months_verify,
        sources=months_sources,
        additional_instruction="Verify that the sources indicate the 2025 event schedule or dates. If exact dates are shown, confirm that they correspond to the claimed month(s). Accept reasonable phrasing variants (e.g., 'Thanksgiving weekend 2025' aligning with late November)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the 2025 National Dog Show venue identification task.
    """
    # 1) Initialize evaluator (root is a non-critical container; we add a critical child)
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # 3) Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # 4) Return structured evaluation summary
    return evaluator.get_summary()