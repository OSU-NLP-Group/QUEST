import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "la_awards_venue_2026"
TASK_DESCRIPTION = """
A major television production company is planning to host a prestigious televised awards ceremony in late March 2026 and needs to identify the optimal venue in the Los Angeles metropolitan area. The selected venue must meet the following requirements:

Capacity & Space:
- Minimum seating capacity of 18,000 for large-scale events
- Arena-style seating configuration suitable for awards ceremonies
- Adequate stage and performance space

Technical Infrastructure:
- Professional-grade sound system suitable for televised productions
- Advanced lighting infrastructure
- Television broadcast production capabilities
- Center-hung display system for visual content
- Proper load-in and load-out capabilities for broadcast equipment

Accessibility:
- ADA compliant with wheelchair accessible seating
- Multiple accessible entrances
- Accessible parking facilities

Scheduling:
- Available for either March 28 or March 29, 2026
- No scheduling conflicts with major sporting events on the selected date

Location:
- Located in the Los Angeles metropolitan area
- Accessible transportation options

Premium Amenities:
- Luxury suite options available for VIP guests
- Premium seating areas
- Multiple VIP entrance facilities
- Adequate backstage and green room facilities
- Professional catering capabilities

Operations & Security:
- Adequate parking facilities
- Security screening infrastructure

Identify the specific venue that meets all these requirements, the exact date (March 28 or 29, 2026) when it is available without sporting event conflicts, and provide comprehensive documentation with URL references for each major category of requirements (capacity specifications, technical capabilities, accessibility features, scheduling availability, location information, VIP amenities, luxury suites, operational details, and security features).
"""

ALLOWED_DATES = {"March 28, 2026", "March 29, 2026"}


class VenueSelectionExtraction(BaseModel):
    venue_name: Optional[str] = None
    other_venues_mentioned: List[str] = Field(default_factory=list)

    selected_date: Optional[str] = None
    other_allowed_dates_mentioned: List[str] = Field(default_factory=list)

    capacity_space_urls: List[str] = Field(default_factory=list)
    technical_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    scheduling_urls: List[str] = Field(default_factory=list)
    location_transport_urls: List[str] = Field(default_factory=list)
    premium_amenities_urls: List[str] = Field(default_factory=list)
    luxury_suites_urls: List[str] = Field(default_factory=list)
    operations_urls: List[str] = Field(default_factory=list)
    security_urls: List[str] = Field(default_factory=list)


def prompt_extract_venue_selection() -> str:
    return """
Extract the final, committed venue choice and the committed single date from the answer, plus all cited source URLs grouped by requirement category.

Required fields:
- venue_name: The single selected venue name that the answer commits to for hosting the ceremony. If several venues are discussed, return only the one explicitly selected; otherwise null.
- other_venues_mentioned: A list of any other distinct venue names mentioned in the answer (exclude the selected one).

- selected_date: The single chosen date if the answer clearly commits to exactly one of "March 28, 2026" or "March 29, 2026". If the answer provides both as options or is non-committal, set this to null.
- other_allowed_dates_mentioned: List any other mentions of the two allowed dates besides the selected one (e.g., if both dates are mentioned but only one is selected, include the unselected one here). Only include these exact allowed dates (or reasonable variations like "Mar 28, 2026" / "Mar 29, 2026").

For each category below, extract every URL explicitly provided in the answer that supports that category. Only include URLs that appear in the answer (plain links or markdown). Do not invent any URLs.

- capacity_space_urls: URLs supporting capacity/space claims (capacity numbers, arena seating, stage/performance space, spec sheets, venue profile pages).
- technical_urls: URLs supporting technical infrastructure (sound, lighting, broadcast facilities, center-hung display, loading docks, production guides).
- accessibility_urls: URLs supporting ADA compliance, wheelchair seating, accessible entrances, accessible parking.
- scheduling_urls: URLs supporting date availability and no sporting-event conflict (venue/event calendar pages, team schedules at the same venue, booking pages).
- location_transport_urls: URLs supporting LA metro location and transportation access (official address pages, transit pages, maps, visitor guides).
- premium_amenities_urls: URLs supporting premium seating, VIP entrances, backstage/green rooms, catering capabilities.
- luxury_suites_urls: URLs supporting availability of luxury suites.
- operations_urls: URLs supporting operational details like adequate parking facilities.
- security_urls: URLs supporting security screening infrastructure/policies.

Rules:
- Only extract URLs explicitly present in the answer.
- Use full URLs including protocol.
- If a category lacks URLs in the answer, return an empty list for that category.
- Be strict about extracting the committed single venue and the committed single date; do not infer beyond what the answer states.
"""


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def verify_venue_identification(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="venue_identification",
        desc="Names exactly one specific venue (not multiple candidates).",
        parent=parent,
        critical=True,
    )
    # Existence of a single committed venue
    evaluator.add_custom_node(
        result=bool(data.venue_name and data.venue_name.strip()),
        id="venue_named_exists",
        desc="A single committed venue name is provided.",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len([v for v in data.other_venues_mentioned if v and v.strip()]) == 0),
        id="venue_no_multiple_candidates",
        desc="No additional venues are presented as candidates.",
        parent=node,
        critical=True,
    )


async def verify_selected_date(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="selected_date",
        desc="States exactly one selected date and it is either March 28, 2026 or March 29, 2026.",
        parent=parent,
        critical=True,
    )

    # Provided
    provided = evaluator.add_custom_node(
        result=bool(data.selected_date and data.selected_date.strip()),
        id="selected_date_provided",
        desc="A single selected date is explicitly provided.",
        parent=node,
        critical=True,
    )

    # Allowed value
    allowed_leaf = evaluator.add_leaf(
        id="selected_date_allowed",
        desc="Selected date is one of the allowed dates (March 28 or March 29, 2026).",
        parent=node,
        critical=True,
    )
    sel = data.selected_date or ""
    claim = f"The selected date is exactly one of: March 28, 2026 or March 29, 2026. The answer's selected date is '{sel}'."
    await evaluator.verify(
        claim=claim,
        node=allowed_leaf,
        additional_instruction="Judge Correct only if the provided selected date unambiguously equals one of the two allowed dates (allow minor formatting variants like 'Mar 28, 2026').",
    )

    # Exactly one allowed date committed (no other allowed date also claimed)
    evaluator.add_custom_node(
        result=(len([d for d in data.other_allowed_dates_mentioned if d and d.strip()]) == 0),
        id="selected_date_exactly_one",
        desc="Exactly one of the allowed dates is selected (the other is not also selected).",
        parent=node,
        critical=True,
    )


async def verify_capacity_and_space(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="capacity_and_space",
        desc="Meets capacity & space requirements, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.capacity_space_urls)
    # URL presence gate
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="capacity_space_urls_present",
        desc="Provides at least one URL that supports the capacity/space claims.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    # Minimum seating capacity >= 18,000
    leaf_cap = evaluator.add_leaf(
        id="minimum_seating_capacity",
        desc="Evidence the venue has seating capacity of at least 18,000.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} has a seating capacity of at least 18,000 in a standard large-event configuration (e.g., basketball, concert, or arena setup)."
    await evaluator.verify(
        claim=claim,
        node=leaf_cap,
        sources=urls,
        additional_instruction="Accept if any standard configuration listed on the referenced page(s) shows capacity >= 18,000; minor numeric variations due to configuration are fine.",
    )

    # Arena-style seating configuration
    leaf_arena = evaluator.add_leaf(
        id="arena_style_seating",
        desc="Evidence the venue has an arena-style seating configuration suitable for large events.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} is an arena with an arena-style seating configuration suitable for large-scale events and award ceremonies."
    await evaluator.verify(
        claim=claim,
        node=leaf_arena,
        sources=urls,
        additional_instruction="Look for wording such as arena, bowl seating, 360-degree seating, end-stage setups, or similar.",
    )

    # Adequate stage/performance space
    leaf_stage = evaluator.add_leaf(
        id="adequate_stage_space",
        desc="Evidence the venue has adequate stage/performance space for an awards-ceremony style event.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} provides adequate stage and performance space suitable for a televised awards ceremony."
    await evaluator.verify(
        claim=claim,
        node=leaf_stage,
        sources=urls,
        additional_instruction="Support can include production specs, stage configurations, rigging capacity, or prior event setups indicating substantial staging is feasible.",
    )


async def verify_technical_infrastructure(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="technical_infrastructure",
        desc="Meets technical infrastructure requirements, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.technical_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="technical_urls_present",
        desc="Provides at least one URL that supports the technical-infrastructure claims.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_sound = evaluator.add_leaf(
        id="sound_system",
        desc="Evidence the venue has a professional-grade sound system suitable for televised productions.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} features a professional-grade sound system appropriate for concert-scale or televised productions."
    await evaluator.verify(
        claim=claim,
        node=leaf_sound,
        sources=urls,
        additional_instruction="Accept references to installed concert-grade/arena sound systems, distributed audio, or similar professional systems.",
    )

    leaf_light = evaluator.add_leaf(
        id="lighting_infrastructure",
        desc="Evidence the venue has advanced lighting infrastructure.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} has advanced lighting infrastructure appropriate for large events or broadcasts."
    await evaluator.verify(
        claim=claim,
        node=leaf_light,
        sources=urls,
        additional_instruction="Look for mentions of production lighting, rigging points, dimming/control systems, or arena lighting systems.",
    )

    leaf_broadcast = evaluator.add_leaf(
        id="broadcast_capabilities",
        desc="Evidence the venue has television broadcast production capabilities.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} supports television broadcast production, such as TV compounds, fiber connectivity, or broadcast facilities."
    await evaluator.verify(
        claim=claim,
        node=leaf_broadcast,
        sources=urls,
        additional_instruction="Accept evidence of hosting televised events, dedicated broadcast compound, fiber drops, or similar broadcast infrastructure.",
    )

    leaf_center = evaluator.add_leaf(
        id="center_hung_display",
        desc="Evidence the venue has a center-hung display system.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} has a center-hung display system (e.g., center-hung video board/scoreboard)."
    await evaluator.verify(
        claim=claim,
        node=leaf_center,
        sources=urls,
        additional_instruction="Look for 'center-hung scoreboard', 'center-hung videoboard', 'center-hung display'.",
    )

    leaf_load = evaluator.add_leaf(
        id="load_in_load_out",
        desc="Evidence the venue has proper load-in/load-out capabilities for broadcast equipment.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} provides appropriate load-in and load-out capabilities for large productions and broadcast equipment (e.g., truck bays, loading docks, freight access)."
    await evaluator.verify(
        claim=claim,
        node=leaf_load,
        sources=urls,
        additional_instruction="Evidence includes loading docks, truck bays, freight elevators, production compound access, or similar logistics features.",
    )


async def verify_accessibility(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="accessibility",
        desc="Meets accessibility requirements, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.accessibility_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="accessibility_urls_present",
        desc="Provides at least one URL that supports the accessibility claims.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_ada = evaluator.add_leaf(
        id="ada_wheelchair_seating",
        desc="Evidence the venue is ADA compliant and has wheelchair accessible seating.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} is ADA compliant and offers wheelchair-accessible seating."
    await evaluator.verify(
        claim=claim,
        node=leaf_ada,
        sources=urls,
        additional_instruction="Accept references to ADA policies, accessible seating maps, companion seating, or official accessibility statements.",
    )

    leaf_entrances = evaluator.add_leaf(
        id="multiple_accessible_entrances",
        desc="Evidence the venue has multiple accessible entrances.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} provides multiple accessible entrances for guests with disabilities."
    await evaluator.verify(
        claim=claim,
        node=leaf_entrances,
        sources=urls,
        additional_instruction="Look for mention of more than one accessible entry point or that all/most entrances are accessible.",
    )

    leaf_park = evaluator.add_leaf(
        id="accessible_parking",
        desc="Evidence the venue has accessible parking facilities.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} offers accessible parking facilities (e.g., ADA parking)."
    await evaluator.verify(
        claim=claim,
        node=leaf_park,
        sources=urls,
        additional_instruction="Accept explicit mention of ADA/accessible parking on-site or nearby with appropriate access.",
    )


async def verify_scheduling(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="scheduling",
        desc="Shows the selected date is available and has no major sporting-event conflict, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.scheduling_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="scheduling_urls_present",
        desc="Provides at least one URL that supports the scheduling availability and no-conflict claims.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"
    sel = data.selected_date or ""

    # Venue available/bookable on selected date
    leaf_avail = evaluator.add_leaf(
        id="venue_available_on_selected_date",
        desc="Evidence the venue is available/bookable on the selected date.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} is available or can be booked on {sel}."
    await evaluator.verify(
        claim=claim,
        node=leaf_avail,
        sources=urls,
        additional_instruction="Support may include venue calendars showing no conflicting events at the venue on that date, booking confirmations, or statements indicating availability.",
    )

    # No major sporting event conflict at the venue on selected date
    leaf_conflict = evaluator.add_leaf(
        id="no_major_sporting_event_conflict",
        desc="Evidence there is no major sporting event scheduled at the venue on the selected date.",
        parent=node,
        critical=True,
    )
    claim = f"There is no major sporting event scheduled at {venue} on {sel}."
    await evaluator.verify(
        claim=claim,
        node=leaf_conflict,
        sources=urls,
        additional_instruction="Accept if referenced pages (venue or tenant team schedules) show no home game or major sporting event at the same venue on that date. Disregard events at other venues.",
    )


async def verify_location_and_transport(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="location_and_transportation",
        desc="Meets location and transportation-access requirements, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.location_transport_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="location_transport_urls_present",
        desc="Provides at least one URL that supports the location/transportation claims.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_location = evaluator.add_leaf(
        id="los_angeles_metro_location",
        desc="Evidence the venue is located in the Los Angeles metropolitan area.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} is located within the Los Angeles metropolitan area."
    await evaluator.verify(
        claim=claim,
        node=leaf_location,
        sources=urls,
        additional_instruction="Accept if the address or official info places the venue in Los Angeles or LA County municipalities such as Los Angeles, Inglewood, or nearby cities commonly considered part of the LA metro area.",
    )

    leaf_transport = evaluator.add_leaf(
        id="accessible_transportation_options",
        desc="Evidence the venue has accessible transportation options.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} offers accessible transportation options such as public transit access, rideshare drop-offs, or dedicated transit connections."
    await evaluator.verify(
        claim=claim,
        node=leaf_transport,
        sources=urls,
        additional_instruction="Look for transit guides, Metro connections, bus/rail info, shuttle or rideshare details on official or venue-provided pages.",
    )


async def verify_premium_amenities(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="premium_amenities",
        desc="Meets premium amenities requirements, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.premium_amenities_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="premium_amenities_urls_present",
        desc="Provides at least one URL that supports the premium-amenities claims.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_premium_seating = evaluator.add_leaf(
        id="premium_seating_areas",
        desc="Evidence the venue offers premium seating areas.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} offers premium seating areas (e.g., club seats, premium sections)."
    await evaluator.verify(
        claim=claim,
        node=leaf_premium_seating,
        sources=urls,
        additional_instruction="Accept mentions of premium seating, club sections, or comparable premium areas.",
    )

    leaf_vip_entries = evaluator.add_leaf(
        id="multiple_vip_entrances",
        desc="Evidence the venue offers multiple VIP entrance facilities.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} provides multiple VIP entrance facilities for premium or suite guests."
    await evaluator.verify(
        claim=claim,
        node=leaf_vip_entries,
        sources=urls,
        additional_instruction="Look for VIP-only entries, several VIP gates, or multiple premium access points.",
    )

    leaf_backstage = evaluator.add_leaf(
        id="backstage_green_rooms",
        desc="Evidence the venue has adequate backstage and green room facilities.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} has adequate backstage and green room facilities to support a large televised production."
    await evaluator.verify(
        claim=claim,
        node=leaf_backstage,
        sources=urls,
        additional_instruction="Evidence may include production guides, dressing room counts, artist compound references, or similar backstage facilities.",
    )

    leaf_catering = evaluator.add_leaf(
        id="professional_catering",
        desc="Evidence the venue has professional catering capabilities.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} provides professional catering capabilities suitable for VIPs and large events."
    await evaluator.verify(
        claim=claim,
        node=leaf_catering,
        sources=urls,
        additional_instruction="Accept mentions of in-house catering, preferred caterers, premium food services, or hospitality catering.",
    )


async def verify_luxury_suites(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="luxury_suites",
        desc="Meets luxury suite requirement, and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.luxury_suites_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="luxury_suites_urls_present",
        desc="Provides at least one URL that supports the luxury suites claim.",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_suites = evaluator.add_leaf(
        id="luxury_suites_available",
        desc="Evidence the venue has luxury suite options available for VIP guests.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} offers luxury suites for VIP guests."
    await evaluator.verify(
        claim=claim,
        node=leaf_suites,
        sources=urls,
        additional_instruction="Look for suites pages, hospitality suites, or premium suites offerings.",
    )


async def verify_operations(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="operations",
        desc="Meets operations requirement(s), and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.operations_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="operations_urls_present",
        desc="Provides at least one URL that supports the operations/parking claim(s).",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_parking = evaluator.add_leaf(
        id="adequate_parking_facilities",
        desc="Evidence the venue has adequate parking facilities.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} offers adequate parking facilities for large events."
    await evaluator.verify(
        claim=claim,
        node=leaf_parking,
        sources=urls,
        additional_instruction="Accept references to on-site or adjacent parking structures/lots with capacity suitable for arena-scale events.",
    )


async def verify_security(evaluator: Evaluator, parent, data: VenueSelectionExtraction):
    node = evaluator.add_parallel(
        id="security",
        desc="Meets security requirement(s), and provides at least one supporting URL for this category.",
        parent=parent,
        critical=True,
    )
    urls = _dedup_urls(data.security_urls)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="security_urls_present",
        desc="Provides at least one URL that supports the security claim(s).",
        parent=node,
        critical=True,
    )

    venue = data.venue_name or "the venue"

    leaf_security = evaluator.add_leaf(
        id="security_screening_infrastructure",
        desc="Evidence the venue has security screening infrastructure.",
        parent=node,
        critical=True,
    )
    claim = f"{venue} has security screening infrastructure (e.g., magnetometers, bag screening, security checkpoints)."
    await evaluator.verify(
        claim=claim,
        node=leaf_security,
        sources=urls,
        additional_instruction="Accept mentions of metal detectors, magnetometers, security screening checkpoints, or formal screening policies indicating infrastructure.",
    )


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

    data: VenueSelectionExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_selection(),
        template_class=VenueSelectionExtraction,
        extraction_name="venue_selection_extraction",
    )

    evaluator.add_ground_truth(
        {
            "allowed_dates": sorted(list(ALLOWED_DATES)),
            "requirement_categories": [
                "capacity_and_space",
                "technical_infrastructure",
                "accessibility",
                "scheduling",
                "location_and_transportation",
                "premium_amenities",
                "luxury_suites",
                "operations",
                "security",
            ],
        },
        gt_type="constraints",
    )

    # Top-level critical node aggregating all requirements
    top = evaluator.add_parallel(
        id="venue_selection",
        desc="Response identifies exactly one specific venue in the LA metro area and exactly one allowed date (Mar 28 or Mar 29, 2026), and provides URL-backed documentation for each major requirement category.",
        parent=root,
        critical=True,
    )

    # Build and verify all categories (all critical under top)
    await verify_venue_identification(evaluator, top, data)
    await verify_selected_date(evaluator, top, data)
    await verify_capacity_and_space(evaluator, top, data)
    await verify_technical_infrastructure(evaluator, top, data)
    await verify_accessibility(evaluator, top, data)
    await verify_scheduling(evaluator, top, data)
    await verify_location_and_transport(evaluator, top, data)
    await verify_premium_amenities(evaluator, top, data)
    await verify_luxury_suites(evaluator, top, data)
    await verify_operations(evaluator, top, data)
    await verify_security(evaluator, top, data)

    return evaluator.get_summary()