import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "socall_amphitheaters_2025"
TASK_DESCRIPTION = """
Identify 3 outdoor amphitheater venues in Southern California (Los Angeles County or San Diego County) that are suitable for hosting major touring artists during summer 2025. For each venue, provide: (1) The venue's official name and specific location (city and county), (2) Total capacity (must be between 5,000 and 20,000), (3) Seating configuration (reserved seating and lawn/general admission breakdown), (4) ADA accessibility features including accessible parking and wheelchair seating, (5) Stage and technical specifications, (6) Backstage facilities (dressing rooms, green rooms), (7) Operational details including parking availability and security policies. Each venue must be a permanent outdoor amphitheater (not an indoor arena, stadium, or temporary structure) that regularly hosts professional concert events. Provide reference URLs for each piece of information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Identification and location
    official_name: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    name_location_urls: List[str] = Field(default_factory=list)

    # Venue type/permanence
    type_permanence_desc: Optional[str] = None
    type_permanence_urls: List[str] = Field(default_factory=list)

    # Professional concert activity
    professional_events_desc: Optional[str] = None
    professional_events_urls: List[str] = Field(default_factory=list)

    # Capacity
    capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    # Seating configuration
    seating_configuration: Optional[str] = None
    seating_urls: List[str] = Field(default_factory=list)

    # ADA accessibility
    ada_accessible_parking_desc: Optional[str] = None
    ada_parking_urls: List[str] = Field(default_factory=list)

    ada_wheelchair_seating_desc: Optional[str] = None
    ada_wheelchair_urls: List[str] = Field(default_factory=list)

    # Stage/technical
    stage_technical_specs: Optional[str] = None
    stage_tech_urls: List[str] = Field(default_factory=list)

    # Backstage
    backstage_facilities: Optional[str] = None
    backstage_urls: List[str] = Field(default_factory=list)

    # Operational details
    operational_parking_desc: Optional[str] = None
    parking_general_urls: List[str] = Field(default_factory=list)

    security_policies_desc: Optional[str] = None
    security_policies_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to all venue entries mentioned in the answer that claim to be permanent outdoor amphitheaters in Los Angeles County or San Diego County. For each venue as presented in the answer, extract the following fields exactly as stated and attach the specific supporting URL(s) used in the answer for each field.

For each venue, extract:
- official_name: The official venue name as stated in the answer.
- city: The city where the venue is located (as stated).
- county: The county where the venue is located (as stated), if provided (e.g., "Los Angeles County" or "San Diego County").
- name_location_urls: All URLs cited in the answer that support the venue name and location (city and/or county).

- type_permanence_desc: The description that indicates the venue is a permanent outdoor amphitheater (not indoor arena/stadium/temporary), as stated in the answer.
- type_permanence_urls: All URLs used in the answer to support the permanence/outdoor amphitheater classification.

- professional_events_desc: The description or statement indicating the venue regularly hosts professional concert events or major touring artists.
- professional_events_urls: All URLs used in the answer to support that the venue regularly hosts professional concerts.

- capacity: The total capacity as stated (keep as string; do not convert to number). If a range is mentioned, keep the range string (e.g., "16,000").
- capacity_urls: All URLs used in the answer to support the capacity.

- seating_configuration: Summary of reserved seating vs. lawn/general admission, as stated in the answer.
- seating_urls: All URLs used in the answer to support the seating configuration.

- ada_accessible_parking_desc: The description indicating ADA accessible parking, as stated in the answer.
- ada_parking_urls: All URLs used in the answer to support ADA accessible parking.

- ada_wheelchair_seating_desc: The description indicating wheelchair accessible seating (ADA seating).
- ada_wheelchair_urls: All URLs used in the answer to support wheelchair accessible seating.

- stage_technical_specs: Summary of stage and technical specifications as stated in the answer.
- stage_tech_urls: All URLs used in the answer to support stage/technical specifications.

- backstage_facilities: Summary of backstage facilities (dressing rooms, green rooms) as stated in the answer.
- backstage_urls: All URLs used in the answer to support backstage facilities.

- operational_parking_desc: Summary of parking facilities for general attendees as stated in the answer.
- parking_general_urls: All URLs used in the answer to support general attendee parking.

- security_policies_desc: Summary of documented security policies/procedures as stated in the answer.
- security_policies_urls: All URLs used in the answer to support security policies.

IMPORTANT:
- Extract values exactly as written in the answer. Do not invent or infer missing details.
- For each field, include only the URLs explicitly present in the answer. If none are provided for a field, return an empty array for that field.
- Include each venue in the order they appear in the answer.
- Do not attempt to clean or normalize; keep strings as-is from the answer text.
- Return a JSON with a top-level 'venues' array of venue objects with the fields defined.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_unique_nonempty_names(venues: List[VenueItem]) -> List[str]:
    seen = set()
    ordered = []
    for v in venues:
        if not v or not v.official_name:
            continue
        name = v.official_name.strip()
        if name == "":
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


def _ensure_three_venues(venues: List[VenueItem]) -> List[VenueItem]:
    picked = list(venues[:3])
    while len(picked) < 3:
        picked.append(VenueItem())
    return picked


def _all_urls_for_venue(v: VenueItem) -> List[str]:
    urls = []
    urls.extend(v.name_location_urls or [])
    urls.extend(v.type_permanence_urls or [])
    urls.extend(v.professional_events_urls or [])
    urls.extend(v.capacity_urls or [])
    urls.extend(v.seating_urls or [])
    urls.extend(v.ada_parking_urls or [])
    urls.extend(v.ada_wheelchair_urls or [])
    urls.extend(v.stage_tech_urls or [])
    urls.extend(v.backstage_urls or [])
    urls.extend(v.parking_general_urls or [])
    urls.extend(v.security_policies_urls or [])
    # Deduplicate, keep order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _url_required_instruction() -> str:
    return (
        "You must rely on the provided webpage(s) as explicit evidence for the claim. "
        "If there are no valid URLs provided for this check, judge the claim as Not Supported (Incorrect) and do not rely on the answer text alone."
    )


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(evaluator: Evaluator, root_node, v: VenueItem, venue_index: int) -> None:
    """
    Build verification subtree for one venue.
    """
    venue_node = evaluator.add_parallel(
        id=f"Venue_{venue_index + 1}",
        desc=f"Venue {venue_index + 1} (amphitheater) requirements",
        parent=root_node,
        critical=False
    )

    name = v.official_name or ""
    city = v.city or ""
    county = v.county or ""
    capacity_text = v.capacity or ""
    seating_text = v.seating_configuration or ""
    ada_parking_text = v.ada_accessible_parking_desc or ""
    ada_wheelchair_text = v.ada_wheelchair_seating_desc or ""
    stage_text = v.stage_technical_specs or ""
    backstage_text = v.backstage_facilities or ""
    op_parking_text = v.operational_parking_desc or ""
    security_text = v.security_policies_desc or ""

    # 1) Name + Location + County in LA or SD, with source
    nlc_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Name_Location_Citation",
        desc="Official venue name plus specific location (city and county) is provided; county is Los Angeles County or San Diego County; includes a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_nlc = (
        f"The venue's official name is '{name}', located in {city}, {county}. "
        "The county is either Los Angeles County or San Diego County."
    )
    await evaluator.verify(
        claim=claim_nlc,
        node=nlc_node,
        sources=v.name_location_urls,
        additional_instruction=(
            _url_required_instruction() + " If county is abbreviated (e.g., 'LA County'/'L.A. County'), "
            "treat it as 'Los Angeles County'. Do not accept if the webpages do not clearly support the stated location."
        )
    )

    # 2) Type/Permanence: permanent outdoor amphitheater
    type_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Type_Permanence_Citation",
        desc="Venue is confirmed to be a permanent outdoor amphitheater (not an indoor arena, stadium, or temporary structure), with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_type = (
        f"'{name}' is a permanent outdoor amphitheater, not an indoor arena, stadium, or temporary structure."
    )
    await evaluator.verify(
        claim=claim_type,
        node=type_node,
        sources=v.type_permanence_urls,
        additional_instruction=_url_required_instruction()
    )

    # 3) Regularly hosts professional concerts/major touring artists
    pro_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Regularly_Hosts_Pro_Concerts_Citation",
        desc="Evidence the venue regularly hosts professional concert events / major touring artists, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_pro = (
        f"'{name}' regularly hosts professional concert events and major touring artists."
    )
    await evaluator.verify(
        claim=claim_pro,
        node=pro_node,
        sources=v.professional_events_urls,
        additional_instruction=(
            _url_required_instruction() + " Event calendars, booking pages, "
            "or credible ticketing pages are acceptable evidence."
        )
    )

    # 4) Capacity between 5,000 and 20,000
    cap_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Capacity_Citation",
        desc="Total capacity is provided, documented to be between 5,000 and 20,000, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_cap = (
        f"The total audience capacity of '{name}' is approximately {capacity_text} and falls between 5,000 and 20,000."
    )
    await evaluator.verify(
        claim=claim_cap,
        node=cap_node,
        sources=v.capacity_urls,
        additional_instruction=_url_required_instruction()
    )

    # 5) Seating configuration: reserved vs lawn/GA
    seat_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Seating_Config_Citation",
        desc="Seating configuration is provided, including reserved seating vs. lawn/general admission breakdown, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_seat = (
        f"The seating configuration for '{name}' includes reserved seating and a lawn/general admission area, summarized as: {seating_text}"
    )
    await evaluator.verify(
        claim=claim_seat,
        node=seat_node,
        sources=v.seating_urls,
        additional_instruction=_url_required_instruction()
    )

    # 6) ADA accessible parking
    ada_parking_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_ADA_Accessible_Parking_Citation",
        desc="ADA accessibility feature: accessible parking is documented, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_ada_parking = (
        f"The venue '{name}' provides ADA accessible parking for guests with disabilities. Details: {ada_parking_text}"
    )
    await evaluator.verify(
        claim=claim_ada_parking,
        node=ada_parking_node,
        sources=v.ada_parking_urls,
        additional_instruction=_url_required_instruction()
    )

    # 7) ADA wheelchair-accessible seating
    ada_wheel_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_ADA_Wheelchair_Seating_Citation",
        desc="ADA accessibility feature: wheelchair accessible seating is documented, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_ada_wheel = (
        f"The venue '{name}' provides wheelchair-accessible seating (ADA seating). Details: {ada_wheelchair_text}"
    )
    await evaluator.verify(
        claim=claim_ada_wheel,
        node=ada_wheel_node,
        sources=v.ada_wheelchair_urls,
        additional_instruction=_url_required_instruction()
    )

    # 8) Stage and technical specifications
    stage_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Stage_Technical_Specs_Citation",
        desc="Stage and technical specifications are provided (as documented by official/credible sources), with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_stage = (
        f"Stage and technical specifications for '{name}' include: {stage_text}"
    )
    await evaluator.verify(
        claim=claim_stage,
        node=stage_node,
        sources=v.stage_tech_urls,
        additional_instruction=_url_required_instruction()
    )

    # 9) Backstage facilities (dressing rooms/green rooms)
    back_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Backstage_Facilities_Citation",
        desc="Backstage facilities (dressing rooms and/or green room areas) are documented, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_back = (
        f"Backstage facilities for '{name}' include: {backstage_text}"
    )
    await evaluator.verify(
        claim=claim_back,
        node=back_node,
        sources=v.backstage_urls,
        additional_instruction=_url_required_instruction()
    )

    # 10) Parking facilities for general attendees
    park_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Parking_Facilities_General_Attendees_Citation",
        desc="Operational detail: parking facilities for general attendees/general admission are documented, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_parking = (
        f"Parking facilities for general attendees are available at '{name}'. Details: {op_parking_text}"
    )
    await evaluator.verify(
        claim=claim_parking,
        node=park_node,
        sources=v.parking_general_urls,
        additional_instruction=_url_required_instruction()
    )

    # 11) Security policies/procedures documented
    sec_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Security_Policies_Citation",
        desc="Operational detail: documented security policies/procedures are provided, with a supporting reference URL.",
        parent=venue_node,
        critical=True
    )
    claim_sec = (
        f"The venue '{name}' has documented security policies/procedures. Summary: {security_text}"
    )
    await evaluator.verify(
        claim=claim_sec,
        node=sec_node,
        sources=v.security_policies_urls,
        additional_instruction=_url_required_instruction()
    )

    # 12) Credibility of all sources used for the venue
    cred_node = evaluator.add_leaf(
        id=f"Venue_{venue_index + 1}_Citation_Source_Credibility",
        desc="All reference URLs used for this venue are from the official venue website or credible venue listing sources.",
        parent=venue_node,
        critical=True
    )
    all_urls = _all_urls_for_venue(v)
    urls_list_str = "\n".join(f"- {u}" for u in all_urls) if all_urls else "(none)"
    cred_claim = (
        f"Each URL listed below is either the official website for '{name}' or a credible venue-listing or official government/ticketing source. "
        "Judge 'Correct' only if ALL of the URLs meet this standard.\n"
        f"URLs:\n{urls_list_str}"
    )
    await evaluator.verify(
        claim=cred_claim,
        node=cred_node,
        sources=None,
        additional_instruction=(
            "Use general domain knowledge to assess credibility from domain names if necessary (e.g., official venue domain, government .gov, city sites, and well-known ticketing platforms such as livenation.com, ticketmaster.com, axs.com). "
            "Personal blogs, user-generated forums, and low-credibility aggregators should not be considered credible. "
            "If the list is empty, judge as Incorrect."
        )
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Southern California outdoor amphitheaters task.
    """
    # Initialize the evaluator with PARALLEL root
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

    # Extract venues and their per-attribute sources
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Record custom info for transparency
    extracted_names = _list_unique_nonempty_names(extraction.venues)
    evaluator.add_custom_info(
        info={
            "extracted_names_in_answer_order": extracted_names,
            "total_extracted_venues": len(extraction.venues),
            "unique_names_count": len(extracted_names)
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # Global set-level requirements (critical)
    global_node = evaluator.add_parallel(
        id="Global_Venue_Set_Requirements",
        desc="Global requirements about the set of venues returned",
        parent=root,
        critical=True
    )
    # Exactly 3 distinct venues provided in the answer
    exactly_three_distinct = (len(extracted_names) == 3)
    evaluator.add_custom_node(
        result=exactly_three_distinct,
        id="Three_Distinct_Venues_Provided",
        desc="Response provides exactly 3 venues and they are distinct (not the same venue repeated under different labels).",
        parent=global_node,
        critical=True
    )

    # Prepare exactly 3 venues for detailed checks (first three, pad if needed)
    picked_venues = _ensure_three_venues(extraction.venues)

    # Venue 1, 2, 3 detailed verification (each as a parallel node under root)
    for i in range(3):
        await verify_single_venue(evaluator, root, picked_venues[i], i)

    # Return evaluation summary
    return evaluator.get_summary()