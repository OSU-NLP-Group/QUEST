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
TASK_ID = "chappell_roan_chicago_arena_2025"
TASK_DESCRIPTION = (
    "Following Chappell Roan's Best New Artist Grammy Award win at the 2025 Grammy Awards ceremony held on "
    "February 2, 2025, her management team is planning her first major headlining arena concert tour. They want "
    "to launch the tour in the Midwest region, specifically choosing a venue in Chicago, Illinois.\n\n"
    "Identify a suitable major arena venue in Chicago that meets the following industry-standard requirements for hosting "
    "a rising star's first arena headlining concert:\n\n"
    "Venue Requirements:\n"
    "1. Minimum concert seating capacity of 18,000 attendees\n"
    "2. Classification as a major professional sports and entertainment arena\n"
    "3. Technical infrastructure capable of supporting:\n"
    "   - Concert stage setup with minimum 20'×20' dimensions\n"
    "   - Stage load capacity of at least 150 pounds per square foot\n"
    "   - Professional sound systems capable of concert sound levels (90-120 dB range)\n"
    "4. Backstage facilities including a minimum of 3 artist dressing rooms\n"
    "5. Full ADA compliance including:\n"
    "   - Wheelchair accessible seating dispersed both horizontally and vertically\n"
    "   - Minimum of 2 accessible means of egress for high-capacity events\n"
    "6. Standard operational capabilities including:\n"
    "   - Specified liability insurance requirements (typically $1,000,000 minimum coverage)\n"
    "   - Acceptance of concert bookings with 6-12 months advance notice\n"
    "   - Early morning load-in capabilities (starting between 7:30-8:00 AM)\n"
    "   - Infrastructure to support arena tour production crews (50-100+ personnel)\n\n"
    "Required Information:\n"
    "Provide the official name of the venue, its specific concert capacity, the professional sports teams that call this "
    "venue home, and a reference URL that confirms the venue information and specifications."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueCore(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None
    concert_capacity: Optional[str] = None
    arena_classification: Optional[str] = None
    sports_teams: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class VenueTechnical(BaseModel):
    stage_dimensions: Optional[str] = None
    stage_load_capacity_psf: Optional[str] = None
    sound_system_db_range: Optional[str] = None
    dressing_rooms_count: Optional[str] = None
    hospitality_desc: Optional[str] = None
    ada_wheelchair_seating_desc: Optional[str] = None
    accessible_egress_count: Optional[str] = None
    sound_management_desc: Optional[str] = None
    insurance_requirements_desc: Optional[str] = None
    booking_timeline_desc: Optional[str] = None
    load_in_start_time: Optional[str] = None
    crew_support_capacity_desc: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    core: Optional[VenueCore] = None
    technical: Optional[VenueTechnical] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    From the provided answer text, extract the core venue information. Only extract what is explicitly stated in the answer.
    Return a JSON object with the following fields:

    core:
      name: The official venue name (string)
      city: The city where the venue is located (string)
      state: The U.S. state where the venue is located (string, e.g., "Illinois")
      region: The U.S. region mentioned in the answer if any (e.g., "Midwest"), otherwise null
      concert_capacity: The specific concert seating capacity number or phrase as written (e.g., "20,000", "20,000+", "approx. 20,000")
      arena_classification: The classification/description (e.g., "major professional sports and entertainment arena")
      sports_teams: List of professional sports teams that call this venue home, as named in the answer
      reference_urls: List of all URLs in the answer that substantiate venue identity, location, capacity, sports teams, or specifications

    Rules:
    - Do not invent or infer values not present in the answer; use null or empty list if missing.
    - Include URLs exactly as they appear (plain or markdown links).
    """


def prompt_extract_technical() -> str:
    return """
    From the provided answer text, extract the technical, safety/ADA, and operational details for the chosen venue.
    Return a JSON object with the following fields:

    technical:
      stage_dimensions: The stated or implied concert stage dimensions text (e.g., "40' x 60'", "at least 20' x 20'"), else null
      stage_load_capacity_psf: Any stated stage/floor load capacity in pounds per square foot (e.g., "150 psf", "250 lb/sf"), else null
      sound_system_db_range: Any stated info implying concert-level sound capability (e.g., "90–120 dB", "professional concert sound"), else null
      dressing_rooms_count: Number of artist dressing rooms or description implying count (e.g., "8 dressing rooms", "multiple dressing rooms"), else null
      hospitality_desc: Any description of green rooms or hospitality facilities, else null

      ada_wheelchair_seating_desc: Any text indicating ADA wheelchair-accessible seating and dispersion, else null
      accessible_egress_count: Any statement about accessible means of egress count (e.g., "2 accessible exits minimum"), else null
      sound_management_desc: Any mention of sound level monitoring/management policies or capabilities, else null

      insurance_requirements_desc: Any text specifying liability insurance requirements (target around $1,000,000), else null
      booking_timeline_desc: Any text specifying booking lead time (e.g., "6–12 months"), else null
      load_in_start_time: Any text indicating early morning load-in windows (e.g., "7:30–8:00 AM"), else null
      crew_support_capacity_desc: Any text indicating support for large touring crews (e.g., "50–100+ personnel"), else null

      additional_urls: Any additional URLs in the answer specifically supporting technical/ADA/operational details

    Rules:
    - Do not invent; return null or empty list if not explicitly provided in the answer.
    - Preserve units and phrasing as in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(extracted: VenueExtraction) -> List[str]:
    urls: List[str] = []
    if extracted and extracted.core and extracted.core.reference_urls:
        urls.extend([u for u in extracted.core.reference_urls if isinstance(u, str) and u.strip()])
    if extracted and extracted.technical and extracted.technical.additional_urls:
        urls.extend([u for u in extracted.technical.additional_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


async def _add_and_verify(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str] | str],
    critical: bool,
    additional_instruction: str = "None",
    extra_prereq_nodes: Optional[List] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prereq_nodes,
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: VenueExtraction, logger: logging.Logger):
    # Create the main task node as sequential aggregator (set non-critical to allow mixed criticality children)
    main = evaluator.add_sequential(
        id="Venue_Identification_Task",
        desc="Comprehensive evaluation of the identified arena venue for suitability",
        parent=root_node,
        critical=False,  # use child criticality to gate score; sequential to short-circuit on early failure
    )

    # 0) Early source presence gate (custom, critical) to enforce source-grounding
    all_sources = merge_sources(extracted)
    url_gate = evaluator.add_custom_node(
        result=(len(all_sources) > 0),
        id="Reference_URL_Presence",
        desc="At least one reference URL is provided in the answer (source-grounding gate)",
        parent=main,
        critical=True,
    )

    # Prepare some handy strings
    venue_name = extracted.core.name if extracted and extracted.core and extracted.core.name else ""
    venue_city = extracted.core.city if extracted and extracted.core and extracted.core.city else ""
    venue_state = extracted.core.state if extracted and extracted.core and extracted.core.state else ""
    sports_teams = extracted.core.sports_teams if extracted and extracted.core and extracted.core.sports_teams else []
    capacity_val = extracted.core.concert_capacity if extracted and extracted.core and extracted.core.concert_capacity else ""
    # For many checks, pass extra prerequisite to enforce source presence
    prereqs = [url_gate]

    # 1) Geographic_Requirements (parallel, critical)
    geo = evaluator.add_parallel(
        id="Geographic_Requirements",
        desc="Venue is located in the correct geographic area as specified",
        parent=main,
        critical=True,
    )

    # 1.1 Midwest_Region (simple check; general knowledge)
    await _add_and_verify(
        evaluator,
        parent=geo,
        node_id="Midwest_Region",
        desc="Venue is located in the Midwest region of the United States",
        claim="Chicago, Illinois is located in the Midwestern region of the United States.",
        sources=None,
        critical=True,
        additional_instruction="Use well-known U.S. regional classification; Chicago, Illinois is considered Midwest. "
                               "This is a general factual check and does not require a URL.",
    )

    # 1.2 Chicago_City (URL-grounded)
    await _add_and_verify(
        evaluator,
        parent=geo,
        node_id="Chicago_City",
        desc="Venue is located in Chicago, Illinois",
        claim=f"The venue '{venue_name}' is located in Chicago, Illinois.",
        sources=all_sources,
        critical=True,
        additional_instruction="Confirm that the referenced page(s) explicitly place the venue in Chicago, Illinois.",
        extra_prereq_nodes=prereqs,
    )

    # 1.3 Geographic_Reference (URL-grounded)
    await _add_and_verify(
        evaluator,
        parent=geo,
        node_id="Geographic_Reference",
        desc="Reference URL confirms geographic location",
        claim="The provided reference page(s) confirm the venue's geographic location in Chicago, Illinois.",
        sources=all_sources,
        critical=True,
        additional_instruction="The evidence should clearly state the venue's location in Chicago, IL (address, city/state, or equivalent).",
        extra_prereq_nodes=prereqs,
    )

    # 2) Capacity_Requirements (parallel, critical)
    cap = evaluator.add_parallel(
        id="Capacity_Requirements",
        desc="Venue meets minimum capacity thresholds for arena-scale concert",
        parent=main,
        critical=True,
    )

    # 2.1 Minimum_Concert_Capacity
    await _add_and_verify(
        evaluator,
        parent=cap,
        node_id="Minimum_Concert_Capacity",
        desc="Venue has minimum concert capacity of 18,000 or greater",
        claim="The venue can seat at least 18,000 attendees for a concert configuration.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept if the cited capacity for concerts or comparable full bowl events is >= 18,000. "
                               "If only basketball/hockey capacity is given and it is >= 18,000, that is acceptable.",
        extra_prereq_nodes=prereqs,
    )

    # 2.2 Arena_Classification
    await _add_and_verify(
        evaluator,
        parent=cap,
        node_id="Arena_Classification",
        desc="Venue is classified as a major arena (not theater or club)",
        claim="The venue is a major professional sports and entertainment arena (not a theater or club).",
        sources=all_sources,
        critical=True,
        additional_instruction="Look for indicators such as hosting NBA/NHL or other top-tier pro teams, and explicit 'arena' classification.",
        extra_prereq_nodes=prereqs,
    )

    # 2.3 Capacity_Reference
    await _add_and_verify(
        evaluator,
        parent=cap,
        node_id="Capacity_Reference",
        desc="Reference URL confirms capacity specifications",
        claim="The provided reference page(s) include the venue's seating or concert capacity specification.",
        sources=all_sources,
        critical=True,
        additional_instruction="The page should contain capacity information (concert, seating bowl, or event capacity).",
        extra_prereq_nodes=prereqs,
    )

    # 3) Technical_Infrastructure (parallel, critical)
    tech = evaluator.add_parallel(
        id="Technical_Infrastructure",
        desc="Venue has adequate technical capabilities for major concert production",
        parent=main,
        critical=True,
    )

    # 3.A) Stage_Specifications (parallel, critical)
    stage = evaluator.add_parallel(
        id="Stage_Specifications",
        desc="Venue can accommodate proper stage setup",
        parent=tech,
        critical=True,
    )

    await _add_and_verify(
        evaluator,
        parent=stage,
        node_id="Stage_Dimensions",
        desc="Venue can accommodate minimum 20'×20' stage for band setup",
        claim="The venue can accommodate a concert stage with minimum dimensions of 20 feet by 20 feet or larger.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept if the technical guide or specs show typical stage plots or dimensions equal to or exceeding 20'×20'.",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=stage,
        node_id="Stage_Load_Capacity",
        desc="Stage infrastructure supports minimum 150 lbs/sq ft load rating",
        claim="The venue's stage or arena floor supports at least a 150 pounds per square foot load rating.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept equivalent units (lb/ft^2 or psf). If a higher rating is given (e.g., 200+ psf), that also satisfies the minimum.",
        extra_prereq_nodes=prereqs,
    )

    # 3.B) Sound_System_Capability
    await _add_and_verify(
        evaluator,
        parent=tech,
        node_id="Sound_System_Capability",
        desc="Venue has professional sound system capable of concert sound levels (90-120 dB)",
        claim="The venue has a professional concert-grade sound system capable of concert sound levels around 90–120 dB.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept if the page describes professional line array systems or equivalent concert-grade audio, even if exact dB is not listed.",
        extra_prereq_nodes=prereqs,
    )

    # 3.C) Backstage_Facilities (parallel, critical)
    backstage = evaluator.add_parallel(
        id="Backstage_Facilities",
        desc="Venue provides adequate backstage facilities",
        parent=tech,
        critical=True,
    )

    await _add_and_verify(
        evaluator,
        parent=backstage,
        node_id="Dressing_Rooms",
        desc="Venue has minimum 3 artist dressing rooms",
        claim="The venue provides at least three artist dressing rooms.",
        sources=all_sources,
        critical=True,
        additional_instruction="Confirm the number of dressing rooms or multiple rooms sufficient for headliner + support acts.",
        extra_prereq_nodes=prereqs,
    )

    # Hospitality_Areas (set critical True to satisfy immediate parent constraint)
    await _add_and_verify(
        evaluator,
        parent=backstage,
        node_id="Hospitality_Areas",
        desc="Venue provides artist hospitality and green room facilities",
        claim="The venue provides artist hospitality and/or green room facilities.",
        sources=all_sources,
        critical=True,
        additional_instruction="Look for mentions of green rooms, lounges, catering areas, or hospitality spaces.",
        extra_prereq_nodes=prereqs,
    )

    # 3.D) Technical_Reference
    await _add_and_verify(
        evaluator,
        parent=tech,
        node_id="Technical_Reference",
        desc="Reference URL confirms technical capabilities or venue specifications",
        claim="The provided reference page(s) include technical or event production specifications for the venue (e.g., rigging, stage, power, or backstage).",
        sources=all_sources,
        critical=True,
        additional_instruction="The page should be an official or authoritative venue/event guide/specifications page.",
        extra_prereq_nodes=prereqs,
    )

    # 4) Safety_and_Accessibility (parallel, critical)
    safety = evaluator.add_parallel(
        id="Safety_and_Accessibility",
        desc="Venue meets safety regulations and ADA accessibility requirements",
        parent=main,
        critical=True,
    )

    await _add_and_verify(
        evaluator,
        parent=safety,
        node_id="Emergency_Egress",
        desc="Venue has minimum 2 accessible means of egress for high-capacity events",
        claim="The venue provides at least two accessible means of egress suitable for high-capacity events.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept compliance statements or plans indicating multiple accessible exits or egress routes.",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=safety,
        node_id="ADA_Wheelchair_Seating",
        desc="Venue provides ADA-compliant wheelchair accessible seating dispersed horizontally and vertically",
        claim="The venue provides ADA-compliant wheelchair-accessible seating that is dispersed both horizontally and vertically.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept ADA seating maps or statements indicating dispersed accessible seating across sections and levels.",
        extra_prereq_nodes=prereqs,
    )

    # Sound_Management (set critical True to satisfy immediate parent constraint)
    await _add_and_verify(
        evaluator,
        parent=safety,
        node_id="Sound_Management",
        desc="Venue has capability to manage and monitor concert sound levels",
        claim="The venue has policies or capabilities to manage and monitor concert sound levels.",
        sources=all_sources,
        critical=True,
        additional_instruction="Look for any mention of SPL monitoring, sound policies, or house engineering support.",
        extra_prereq_nodes=prereqs,
    )

    # Safety_Reference (set critical True to satisfy immediate parent constraint)
    await _add_and_verify(
        evaluator,
        parent=safety,
        node_id="Safety_Reference",
        desc="Reference URL confirms safety and accessibility features",
        claim="The provided reference page(s) include information about safety, egress, and/or ADA accessibility features for the venue.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept ADA policy pages, accessibility guides, or venue safety/evacuation information.",
        extra_prereq_nodes=prereqs,
    )

    # 5) Operational_Requirements (parallel, non-critical as in rubric)
    ops = evaluator.add_parallel(
        id="Operational_Requirements",
        desc="Venue meets operational and logistical requirements for concert production",
        parent=main,
        critical=False,
    )

    await _add_and_verify(
        evaluator,
        parent=ops,
        node_id="Insurance_Requirements",
        desc="Venue specifies liability insurance requirements (typically $1M minimum)",
        claim="The venue specifies liability insurance requirements around $1,000,000 minimum coverage for events.",
        sources=all_sources,
        critical=False,
        additional_instruction="Accept if the policy states $1,000,000 or higher per occurrence for liability.",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=ops,
        node_id="Booking_Timeline",
        desc="Venue accepts bookings with 6-12 months advance notice for major concerts",
        claim="The venue accepts concert bookings with approximately 6–12 months of advance notice for major events.",
        sources=all_sources,
        critical=False,
        additional_instruction="Accept explicit lead-time, or language indicating typical arena booking windows in that range.",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=ops,
        node_id="Load_In_Capability",
        desc="Venue can accommodate early morning load-in (7:30-8:00 AM start)",
        claim="The venue can accommodate early morning load-ins starting around 7:30–8:00 AM.",
        sources=all_sources,
        critical=False,
        additional_instruction="Accept load-in schedules or day-of-show timelines indicating early morning access.",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=ops,
        node_id="Crew_Accommodation",
        desc="Venue can support arena tour crew requirements (50-100+ personnel)",
        claim="The venue can support arena tour production crews of roughly 50–100 or more personnel.",
        sources=all_sources,
        critical=False,
        additional_instruction="Accept staffing/production notes indicating support for large touring crews.",
        extra_prereq_nodes=prereqs,
    )

    # 6) Venue_Verification (parallel, critical)
    verify = evaluator.add_parallel(
        id="Venue_Verification",
        desc="Complete venue information provided with proper documentation",
        parent=main,
        critical=True,
    )

    await _add_and_verify(
        evaluator,
        parent=verify,
        node_id="Venue_Name",
        desc="Official name of the arena venue provided",
        claim=f"The official name of the arena is '{venue_name}'.",
        sources=all_sources,
        critical=True,
        additional_instruction="The page should clearly show the official, current venue name.",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=verify,
        node_id="Capacity_Specification",
        desc="Specific concert capacity number provided",
        claim=f"The venue's concert capacity is stated as '{capacity_val}'.",
        sources=all_sources,
        critical=True,
        additional_instruction="Match the number/phrase as written in the answer (allow minor formatting differences, like commas or plus signs).",
        extra_prereq_nodes=prereqs,
    )

    await _add_and_verify(
        evaluator,
        parent=verify,
        node_id="Sports_Teams_Hosted",
        desc="Professional sports teams that call the venue home identified",
        claim=f"The venue is home to the following professional team(s): {sports_teams}.",
        sources=all_sources,
        critical=True,
        additional_instruction="Accept official team/venue pages listing the venue as home arena for these teams. "
                               "Minor name-variant tolerance (e.g., 'Chicago Bulls' vs 'Bulls').",
        extra_prereq_nodes=prereqs,
    )

    # Reference_URL (existence check as a critical custom node within Venue_Verification)
    evaluator.add_custom_node(
        result=(len(all_sources) > 0),
        id="Reference_URL",
        desc="Valid reference URL provided confirming venue information",
        parent=verify,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; real logic under child node
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

    # Extract structured info
    core = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=VenueExtraction,
        extraction_name="venue_core_extraction",
    )

    tech = await evaluator.extract(
        prompt=prompt_extract_technical(),
        template_class=VenueExtraction,
        extraction_name="venue_technical_extraction",
    )

    # Merge the two extractions into a single VenueExtraction object
    # Prefer non-null sections from each
    merged = VenueExtraction(
        core=core.core if core and core.core else (tech.core if tech and tech.core else None),
        technical=tech.technical if tech and tech.technical else (core.technical if core and core.technical else None),
    )

    # Add a compact info summary for debugging/traceability
    evaluator.add_custom_info(
        info={
            "chosen_venue_name": merged.core.name if merged and merged.core else None,
            "city_state": f"{(merged.core.city if merged and merged.core else None)}, {(merged.core.state if merged and merged.core else None)}",
            "concert_capacity_answer_text": (merged.core.concert_capacity if merged and merged.core else None),
            "sports_teams_list": (merged.core.sports_teams if merged and merged.core else []),
            "reference_urls_count": len(merge_sources(merged)),
        },
        info_type="parsed_overview",
        info_name="extraction_overview",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, root, merged, logger)

    # Return summary
    return evaluator.get_summary()