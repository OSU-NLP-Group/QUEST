import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "west_coast_observatory_2026_dual_event"
TASK_DESCRIPTION = """
An undergraduate astronomy student is planning a research project that requires observing two significant celestial events in early 2026: the February 28, 2026 planetary parade (featuring six planets including Uranus and Neptune) and the March 3, 2026 total lunar eclipse. The student needs to conduct both observations from the same location to maintain consistency in their research methodology.

Identify a university observatory located in a U.S. West Coast state (California, Oregon, or Washington) that satisfies all of the following requirements:

1. The observatory must be affiliated with a university that has an astronomy or physics department.

2. The observatory must offer student observation opportunities or public access programs that would allow the student to use the facility.

3. The observatory must have telescope equipment suitable for observing faint celestial objects not visible to the naked eye (specifically Uranus and Neptune during the planetary parade).

4. For the February 28, 2026 planetary parade observation:
   - Confirm that the event will be visible from the observatory's location
   - Specify the observation timing window (shortly after sunset on February 28)
   - Note that an unobstructed western horizon is required for optimal viewing

5. For the March 3, 2026 total lunar eclipse observation:
   - Confirm that the total eclipse, including the totality phase, will be visible from the observatory's location
   - Provide the timing of the totality phase (which occurs from 3:04 a.m. to 4:03 a.m. PST or equivalent in the local time zone)
   - Acknowledge that this observation occurs in the early morning hours

Provide the following information in your answer:
- The name of the university and the observatory
- The state where the observatory is located
- Evidence of telescope equipment availability and suitability
- Confirmation of student or public accessibility
- Verification of visibility and timing for both astronomical events
- Supporting URL references for all major claims
"""

ALLOWED_STATES = {
    "california": "California",
    "ca": "California",
    "oregon": "Oregon",
    "or": "Oregon",
    "washington": "Washington",
    "wa": "Washington",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ObservatoryCore(BaseModel):
    university: Optional[str] = None
    observatory: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    same_location_statement: Optional[str] = None  # exact sentence/phrase from answer, if present


class ObservatoryURLs(BaseModel):
    identity_urls: List[str] = Field(default_factory=list)       # identity/location/state of observatory
    affiliation_urls: List[str] = Field(default_factory=list)    # confirms affiliation with university
    department_urls: List[str] = Field(default_factory=list)     # astronomy/physics department pages
    access_urls: List[str] = Field(default_factory=list)         # public nights/student access
    equipment_urls: List[str] = Field(default_factory=list)      # telescope/equipment pages
    parade_urls: List[str] = Field(default_factory=list)         # Feb 28, 2026 planetary parade visibility/timing
    horizon_urls: List[str] = Field(default_factory=list)        # evidence for unobstructed western horizon
    eclipse_urls: List[str] = Field(default_factory=list)        # Mar 3, 2026 total lunar eclipse visibility/timing


class EquipmentInfo(BaseModel):
    description: Optional[str] = None   # summarized description from the answer (e.g., telescope specs)


class AccessInfo(BaseModel):
    description: Optional[str] = None   # summarized description of access program from the answer


class ParadeInfo(BaseModel):
    visibility_statement: Optional[str] = None       # answer's statement that parade is visible at location
    timing_window_statement: Optional[str] = None    # "shortly after sunset (~30 min)"
    horizon_requirement_statement: Optional[str] = None  # answer notes western horizon requirement


class EclipseInfo(BaseModel):
    totality_visibility_statement: Optional[str] = None
    totality_timing_statement: Optional[str] = None         # "3:04–4:03 a.m. PST" (or local equivalent)
    full_window_statement: Optional[str] = None             # "~12:44 a.m.–6:23 a.m. PST"
    early_morning_statement: Optional[str] = None           # answer acknowledges early morning


class ObservatorySelectionExtraction(BaseModel):
    core: ObservatoryCore = Field(default_factory=ObservatoryCore)
    urls: ObservatoryURLs = Field(default_factory=ObservatoryURLs)
    equipment: EquipmentInfo = Field(default_factory=EquipmentInfo)
    access: AccessInfo = Field(default_factory=AccessInfo)
    parade: ParadeInfo = Field(default_factory=ParadeInfo)
    eclipse: EclipseInfo = Field(default_factory=EclipseInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_observatory_selection() -> str:
    return """
    Extract the requested structured information from the answer about a single university observatory in a West Coast U.S. state (California, Oregon, or Washington) that can be used to observe both the Feb 28, 2026 planetary parade and the Mar 3, 2026 total lunar eclipse from the SAME location. Return JSON matching the schema exactly.

    Fields to extract:

    core:
      - university: The university name explicitly stated in the answer (string or null)
      - observatory: The observatory/facility name (string or null)
      - state: The state where the observatory is located, as mentioned (e.g., "California", "CA", "Oregon", "OR", "Washington", "WA")
      - city: The city, if mentioned (string or null)
      - same_location_statement: Copy the exact sentence/phrase where the answer indicates BOTH observations will be conducted from the same location/observatory (null if not explicitly stated)

    urls:
      - identity_urls: URL(s) supporting the observatory identity and location/state
      - affiliation_urls: URL(s) that show the observatory is affiliated with the named university
      - department_urls: URL(s) that show the university has an astronomy or physics department (or equivalent)
      - access_urls: URL(s) showing student observing opportunities or public access programs at the observatory
      - equipment_urls: URL(s) showing telescope/equipment details suitable for faint objects (e.g., Uranus/Neptune)
      - parade_urls: URL(s) supporting visibility and timing info for the Feb 28, 2026 planetary parade
      - horizon_urls: URL(s) or map/satellite links that support an unobstructed western horizon at the site (if included)
      - eclipse_urls: URL(s) supporting visibility and timing info (including totality) for the Mar 3, 2026 total lunar eclipse

    equipment:
      - description: The answer's description of the telescope/equipment suitability for faint objects (verbatim or concise summary)

    access:
      - description: The answer’s description that public nights or student observing are offered (verbatim or concise summary)

    parade:
      - visibility_statement: The answer’s sentence/phrase confirming visibility of the Feb 28, 2026 planetary parade from that location
      - timing_window_statement: The answer’s sentence/phrase indicating the timing window is shortly after local sunset (around 30 minutes after)
      - horizon_requirement_statement: The answer’s sentence/phrase noting an unobstructed WESTERN horizon is required

    eclipse:
      - totality_visibility_statement: The answer’s sentence/phrase confirming totality is visible from that location
      - totality_timing_statement: The answer’s sentence/phrase giving totality ~3:04 a.m. to 4:03 a.m. PST (or equivalent in local time)
      - full_window_statement: The answer’s sentence/phrase giving full eclipse window ~12:44 a.m. to 6:23 a.m. PST
      - early_morning_statement: The answer’s sentence/phrase acknowledging the observation occurs in the early morning hours

    URL rules:
    - Extract only URLs that are explicitly present in the answer (plain URL or markdown link).
    - Do not fabricate URLs. If a category has no URLs, return an empty list for that field.

    If any field is not present in the answer, return null for strings and [] for lists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    return ALLOWED_STATES.get(s, state.strip())


def is_allowed_state(state: Optional[str]) -> bool:
    st = normalize_state(state)
    return st in {"California", "Oregon", "Washington"}


def ensure_list(urls: Optional[List[str]]) -> List[str]:
    return urls if isinstance(urls, list) else []


async def verify_claim_with_urls_or_fail(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    critical: bool = True,
    additional_instruction: str = "None",
):
    url_list = ensure_list(urls)
    if len(url_list) == 0:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (failed: no supporting URLs provided in the answer)",
            parent=parent,
            critical=critical,
        )
        return False
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=url_list,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ObservatorySelectionExtraction):
    # Create the main critical validation node (as child of root, since framework root is non-critical)
    main = evaluator.add_parallel(
        id="Observatory_Selection_Validation",
        desc="Validates that the identified university observatory satisfies all stated requirements/constraints for observing both events from the same location, and that major claims are supported by URLs.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Observatory Identity and Location
    identity_loc = evaluator.add_parallel(
        id="Observatory_Identity_and_Location",
        desc="Answer identifies a specific university observatory and places it in an allowed West Coast state.",
        parent=main,
        critical=True,
    )

    # 1.a) Names Provided
    names_provided = evaluator.add_custom_node(
        result=bool(extracted.core.university) and bool(extracted.core.observatory),
        id="Names_Provided",
        desc="Provides the name of the university and the name of the observatory/facility.",
        parent=identity_loc,
        critical=True,
    )

    # 1.b) State Allowed
    state_ok = evaluator.add_custom_node(
        result=is_allowed_state(extracted.core.state),
        id="State_Allowed",
        desc="Observatory is located in California, Oregon, or Washington.",
        parent=identity_loc,
        critical=True,
    )

    # 2) Same Location For Both Observations
    same_loc_leaf = evaluator.add_leaf(
        id="Same_Location_For_Both_Observations",
        desc="Explicitly indicates both the Feb 28, 2026 planetary parade observation and the Mar 3, 2026 total lunar eclipse observation are conducted from the same observatory/location.",
        parent=main,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly indicates that both the February 28, 2026 planetary parade and the March 3, 2026 total lunar eclipse will be observed from the same single observatory/location.",
        node=same_loc_leaf,
        additional_instruction="Look for explicit wording such as 'both from the same observatory', 'same site', 'same location for both events', or equivalent phrasing in the provided answer.",
    )

    # 3) University Affiliation and Department
    uni_aff = evaluator.add_parallel(
        id="University_Affiliation_and_Department",
        desc="Observatory is affiliated with a university that has an astronomy or physics department (or equivalent).",
        parent=main,
        critical=True,
    )

    # 3.a) Affiliated With University (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=uni_aff,
        node_id="Affiliated_With_University",
        desc="Provides evidence the observatory is affiliated with the named university.",
        claim=f"The observatory '{extracted.core.observatory or 'the observatory'}' is affiliated with the university '{extracted.core.university or 'the named university'}'.",
        urls=(ensure_list(extracted.urls.identity_urls) + ensure_list(extracted.urls.affiliation_urls)),
        additional_instruction="Accept official university or observatory pages that clearly indicate the observatory is part of or operated by the named university.",
    )

    # 3.b) Astronomy or Physics Department Exists (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=uni_aff,
        node_id="Astronomy_or_Physics_Department_Exists",
        desc="Provides evidence the university has an astronomy department or a physics department (or equivalent).",
        claim=f"The university '{extracted.core.university or 'the named university'}' has either an astronomy department or a physics department (or an equivalent program/department).",
        urls=extracted.urls.department_urls,
        additional_instruction="Department or program pages (e.g., Department of Physics, Department of Astronomy, Physics & Astronomy) count as valid support.",
    )

    # 4) Accessibility Programs (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=main,
        node_id="Accessibility_Programs",
        desc="Provides evidence the observatory offers student observation opportunities or public access programs that would allow undergraduate use of the facility.",
        claim="The observatory offers public viewing nights, open houses, or student observing opportunities that allow students or the public to use the facility.",
        urls=extracted.urls.access_urls,
        additional_instruction="Look for terms like 'public viewing', 'open nights', 'student observing', 'open to students/public', or reservation procedures for visitors or classes.",
    )

    # 5) Telescope Equipment Suitability (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=main,
        node_id="Telescope_Equipment_Suitability",
        desc="Provides evidence the observatory has telescope/optical equipment suitable for observing faint objects like Uranus and Neptune (i.e., not naked-eye) and is described as research-grade / sufficiently capable per constraints.",
        claim="The observatory has telescope equipment suitable for observing faint planets like Uranus and Neptune (i.e., larger-aperture telescopes or observatory-class instruments).",
        urls=extracted.urls.equipment_urls,
        additional_instruction="Equipment pages listing telescopes (e.g., 8-inch or larger reflectors/refractors, observatory-class instruments, imaging capability) count as support.",
    )

    # 6) February 28, 2026 Planetary Parade
    parade = evaluator.add_parallel(
        id="February_28_2026_Planetary_Parade",
        desc="Meets event-specific requirements/constraints for the Feb 28, 2026 planetary parade from the observatory location.",
        parent=main,
        critical=True,
    )

    normalized_state = normalize_state(extracted.core.state) or "the observatory's state"

    # 6.a) Visible from Location (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=parade,
        node_id="Visible_From_Observatory_Location",
        desc="Confirms the Feb 28, 2026 planetary parade is visible from the observatory’s specific location.",
        claim=f"The February 28, 2026 planetary alignment/parade (featuring six planets including Uranus and Neptune) is visible from {normalized_state}, United States (and thus from the observatory's location).",
        urls=extracted.urls.parade_urls,
        additional_instruction="Accept reputable astronomy sources describing the visibility across the U.S. West Coast or specifically the given state. Minor wording differences like 'alignment' vs 'parade' are acceptable.",
    )

    # 6.b) Timing window ~30 min after sunset (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=parade,
        node_id="Timing_Window_Approx_30_Min_After_Sunset",
        desc="Specifies the observation timing window as shortly after local sunset and approximately 30 minutes after local sunset (per constraints).",
        claim="On February 28, 2026, the recommended observation time for the planetary parade is shortly after local sunset, approximately 30 minutes after sunset.",
        urls=extracted.urls.parade_urls,
        additional_instruction="Look for guidance indicating the evening/twilight window and particularly ~30 minutes after sunset as the optimal time.",
    )

    # 6.c) Western horizon requirement noted (simple verify from answer)
    west_note_leaf = evaluator.add_leaf(
        id="Western_Horizon_Requirement_Noted",
        desc="Notes that an unobstructed western horizon is required for optimal viewing.",
        parent=parade,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly notes that an unobstructed western horizon is required for optimal viewing of the February 28, 2026 planetary parade.",
        node=west_note_leaf,
        additional_instruction="Look for exact or equivalent phrasing indicating 'unobstructed western horizon' or clear westward view.",
    )

    # 6.d) Site has unobstructed western horizon (verify via URLs - allow horizon_urls or identity_urls if horizon-specific not provided)
    horizon_sources = ensure_list(extracted.urls.horizon_urls)
    if len(horizon_sources) == 0:
        horizon_sources = ensure_list(extracted.urls.identity_urls)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=parade,
        node_id="Site_Has_Unobstructed_Western_Horizon",
        desc="States (with evidence or a concrete site description) that the observatory site provides an unobstructed western horizon as required by the constraints.",
        claim="The observatory site provides an unobstructed (or notably clear) view toward the western horizon.",
        urls=horizon_sources,
        additional_instruction="Accept explicit site descriptions, photos, maps, or official statements indicating a clear westward view or elevated/hilltop location affording unobstructed western horizon.",
    )

    # 7) March 3, 2026 Total Lunar Eclipse
    eclipse = evaluator.add_parallel(
        id="March_3_2026_Total_Lunar_Eclipse",
        desc="Meets event-specific requirements/constraints for the Mar 3, 2026 total lunar eclipse from the observatory location.",
        parent=main,
        critical=True,
    )

    # 7.a) Totality visible from location (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=eclipse,
        node_id="Totality_Visible_From_Observatory_Location",
        desc="Confirms the total lunar eclipse (including totality) is visible from the observatory’s specific location.",
        claim=f"The March 3, 2026 total lunar eclipse (including the totality phase) is visible from {normalized_state}, United States.",
        urls=extracted.urls.eclipse_urls,
        additional_instruction="Accept authoritative eclipse visibility maps/tables indicating West Coast visibility of totality.",
    )

    # 7.b) Totality timing provided (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=eclipse,
        node_id="Totality_Timing_Provided",
        desc="Provides the timing of totality as 3:04 a.m. to 4:03 a.m. PST or equivalent in the local time zone.",
        claim="For the March 3, 2026 total lunar eclipse, the totality phase occurs from approximately 3:04 a.m. to 4:03 a.m. PST (or an equivalent local time).",
        urls=extracted.urls.eclipse_urls,
        additional_instruction="Allow minor rounding (±1–2 minutes). If the source lists times in UTC or another zone, equivalently converting to PST is acceptable.",
    )

    # 7.c) Full eclipse duration provided (verify via URLs)
    await verify_claim_with_urls_or_fail(
        evaluator,
        parent=eclipse,
        node_id="Full_Eclipse_Duration_Provided",
        desc="Provides the full eclipse timing window (approximately 12:44 a.m. to 6:23 a.m. PST) as stated in the constraints.",
        claim="For the March 3, 2026 total lunar eclipse, the full eclipse window spans approximately 12:44 a.m. to 6:23 a.m. PST.",
        urls=extracted.urls.eclipse_urls,
        additional_instruction="Allow minor rounding differences. If listed in another time zone, equivalent PST conversion is acceptable.",
    )

    # 7.d) Early morning acknowledged (simple verify from answer)
    early_leaf = evaluator.add_leaf(
        id="Early_Morning_Acknowledged",
        desc="Acknowledges the eclipse observation occurs in the early morning hours.",
        parent=eclipse,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly acknowledges that the March 3, 2026 lunar eclipse observation occurs in the early morning hours.",
        node=early_leaf,
        additional_instruction="Look for phrases like 'early morning', 'pre-dawn', 'in the early hours', or explicit times around 3–4 a.m.",
    )

    # 8) Supporting URL References
    urls_grp = evaluator.add_parallel(
        id="Supporting_URL_References",
        desc="Provides supporting URL references covering all major claims required by the question/constraints.",
        parent=main,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(ensure_list(extracted.urls.identity_urls)) > 0,
        id="URLs_For_Identity_and_Location",
        desc="Provides at least one URL supporting the observatory identity and its location/state.",
        parent=urls_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(len(ensure_list(extracted.urls.affiliation_urls)) > 0 and len(ensure_list(extracted.urls.department_urls)) > 0),
        id="URLs_For_Affiliation_and_Department",
        desc="Provides at least one URL supporting the observatory’s university affiliation and the existence of an astronomy/physics department (or equivalent).",
        parent=urls_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(ensure_list(extracted.urls.access_urls)) > 0,
        id="URLs_For_Accessibility",
        desc="Provides at least one URL supporting student/public access or observing opportunity claims.",
        parent=urls_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(ensure_list(extracted.urls.equipment_urls)) > 0,
        id="URLs_For_Equipment",
        desc="Provides at least one URL supporting telescope/equipment availability and suitability (for Uranus/Neptune visibility).",
        parent=urls_grp,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(len(ensure_list(extracted.urls.parade_urls)) > 0 and len(ensure_list(extracted.urls.eclipse_urls)) > 0),
        id="URLs_For_Event_Visibility_and_Timing",
        desc="Provides at least one URL supporting the visibility and timing claims for both the Feb 28, 2026 planetary parade and the Mar 3, 2026 total lunar eclipse (including totality timing).",
        parent=urls_grp,
        critical=True,
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
    # Initialize evaluator
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_observatory_selection(),
        template_class=ObservatorySelectionExtraction,
        extraction_name="observatory_selection_extraction",
    )

    # Add a small custom info block with normalized state (for debugging)
    evaluator.add_custom_info(
        info={
            "university": extracted.core.university,
            "observatory": extracted.core.observatory,
            "state_raw": extracted.core.state,
            "state_normalized": normalize_state(extracted.core.state),
        },
        info_type="parsed_core_info",
        info_name="core_info_summary",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()