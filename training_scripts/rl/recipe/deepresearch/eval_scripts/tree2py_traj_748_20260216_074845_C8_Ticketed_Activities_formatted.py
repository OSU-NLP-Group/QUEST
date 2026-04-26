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
TASK_ID = "entertainment_tour_venues_2026"
TASK_DESCRIPTION = """A national entertainment company is organizing a 2026 multi-city performance tour across the United States featuring four distinct events at different venue types. Identify one suitable venue for each of the following events, ensuring each venue meets all specified requirements:

Event 1 - Major Concert in New York City
Requirements:
- Must be a concert arena in New York City
- Must have a concert seating capacity of at least 18,000
- Must be a multi-purpose arena capable of hosting major concerts
- Must meet ADA accessibility standards
- Must have professional audio-visual and stage capabilities
- Provide venue name and reference URL

Event 2 - Awards Ceremony in Los Angeles Area
Requirements:
- Must be located in the Los Angeles area of California
- Must accommodate at least 1,000 attendees for a formal awards ceremony
- Must be an upscale hotel ballroom or similar facility appropriate for prestigious awards events
- Must be capable of meeting insurance requirements ($1 million per occurrence, $2 million aggregate minimum)
- Must have banquet seating capability and stage area
- Provide venue name and reference URL

Event 3 - Broadway-Style Theatrical Production
Requirements:
- Must be located in a major US city with an established theater district
- Must meet the minimum 500-seat capacity to qualify as a Broadway-class theater
- Must have adequate stage facilities for theatrical productions
- Must meet ADA accessibility requirements
- Must be capable of accommodating standard venue insurance requirements
- Provide venue name and reference URL

Event 4 - Outdoor Music Festival
Requirements:
- Must be an outdoor amphitheater in the United States
- Must have a minimum capacity of 5,000 for outdoor concerts
- Must be configured as an outdoor venue
- Must have adequate stage dimensions including minimum 16 feet stage depth for full band setup
- Must operate with proper permits and accommodate temporary event requirements
- Provide venue name and reference URL

For each venue, provide the venue name, a brief description of how it meets the requirements, and at least one reference URL from your research supporting the identification.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    ny_concert: Optional[VenueInfo] = None
    la_awards: Optional[VenueInfo] = None
    broadway_theater: Optional[VenueInfo] = None
    outdoor_amphitheater: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract exactly one venue for each of the four events described in the task from the provided answer. For each event, extract:
- name: the venue name as written in the answer
- description: a brief summary (1–3 sentences) from the answer explaining why this venue meets the requirements (if present)
- reference_urls: an array of all explicit URLs the answer cites for that event/venue (include every URL shown in plain text or as markdown links)

Events to extract (pick only the first mentioned venue for each if multiple are presented):
- ny_concert: Major Concert in New York City
- la_awards: Awards Ceremony in the Los Angeles area
- broadway_theater: Broadway-style theatrical production
- outdoor_amphitheater: Outdoor music festival amphitheater

Rules:
- Only include URLs that are explicitly present in the answer.
- Ensure URLs are full (include http:// or https://); if missing, prepend http://.
- If any field is missing, set it to null or use an empty array for reference_urls.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(venue: Optional[VenueInfo]) -> str:
    if venue and venue.name and venue.name.strip():
        return venue.name.strip()
    return "the venue"


def _get_sources(venue: Optional[VenueInfo]) -> List[str]:
    if venue and venue.reference_urls:
        # Normalize: keep only plausible URLs (basic filter)
        urls = []
        for u in venue.reference_urls:
            if isinstance(u, str) and u.strip():
                s = u.strip()
                if not (s.startswith("http://") or s.startswith("https://")):
                    s = "http://" + s
                urls.append(s)
        return urls
    return []


def _has_valid_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")) for u in urls)


# --------------------------------------------------------------------------- #
# Verification functions per event                                            #
# --------------------------------------------------------------------------- #
async def verify_new_york_concert_venue(evaluator: Evaluator, parent_node, venue: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="new_york_concert_venue",
        desc="Verify a suitable large concert arena in New York City is identified",
        parent=parent_node,
        critical=False
    )

    sources = _get_sources(venue)
    name = _safe_name(venue)

    # Critical: Reference URL present
    evaluator.add_custom_node(
        result=_has_valid_url(sources),
        id="ny_arena_reference",
        desc="Verify a valid reference URL is provided supporting the New York arena identification",
        parent=node,
        critical=True
    )

    # Capacity ≥ 18,000
    ny_capacity_leaf = evaluator.add_leaf(
        id="ny_arena_capacity",
        desc="Verify the venue has a concert seating capacity of at least 18,000",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} has a concert seating capacity of at least 18,000.",
        node=ny_capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Check the venue page or reputable sources for 'concert capacity', 'seating capacity', or stated maximums. "
            "If multiple configurations are listed (sports vs concerts), use the concert configuration. "
            "Accept if the capacity is ≥ 18,000."
        ),
    )

    # Location = NYC
    ny_location_leaf = evaluator.add_leaf(
        id="ny_arena_location",
        desc="Verify the venue is located in New York City",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} is located in New York City, New York (NYC).",
        node=ny_location_leaf,
        sources=sources,
        additional_instruction=(
            "NYC includes the five boroughs: Manhattan, Brooklyn, Queens, the Bronx, and Staten Island. "
            "Accept 'New York, NY' or borough-specific addresses as NYC."
        ),
    )

    # Multi-purpose arena hosting concerts
    ny_multi_purpose_leaf = evaluator.add_leaf(
        id="ny_arena_multi_purpose",
        desc="Verify the venue is a multi-purpose arena capable of hosting concerts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} is a multi-purpose arena that hosts major concerts.",
        node=ny_multi_purpose_leaf,
        sources=sources,
        additional_instruction=(
            "Look for wording such as 'multi-purpose', 'multi-use', and evidence of hosting concerts "
            "(e.g., events calendar, past concert listings)."
        ),
    )

    # ADA accessibility
    ny_ada_leaf = evaluator.add_leaf(
        id="ny_arena_ada_compliance",
        desc="Verify the venue meets ADA accessibility requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} meets ADA accessibility requirements (e.g., accessible seating, ADA policies).",
        node=ny_ada_leaf,
        sources=sources,
        additional_instruction=(
            "Look for accessibility/ADA policy pages or mentions of accessible seating, wheelchair access, "
            "assistive listening, or ADA compliance."
        ),
    )

    # Technical specs (pro AV and stage)
    ny_tech_leaf = evaluator.add_leaf(
        id="ny_arena_technical_specs",
        desc="Verify the venue has professional audio-visual and stage capabilities for major concerts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} has professional audio-visual and stage capabilities suitable for major concerts.",
        node=ny_tech_leaf,
        sources=sources,
        additional_instruction=(
            "Check for production guides, technical specs, rigging information, in-house AV, "
            "or references to hosting large-scale touring productions."
        ),
    )


async def verify_california_awards_venue(evaluator: Evaluator, parent_node, venue: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="california_awards_venue",
        desc="Verify a suitable awards ceremony venue in Los Angeles area is identified",
        parent=parent_node,
        critical=False
    )

    sources = _get_sources(venue)
    name = _safe_name(venue)

    # Critical: Reference URL present
    evaluator.add_custom_node(
        result=_has_valid_url(sources),
        id="ca_awards_reference",
        desc="Verify a valid reference URL is provided supporting the California awards venue identification",
        parent=node,
        critical=True
    )

    # Capacity ≥ 1,000 attendees for awards ceremony
    ca_capacity_leaf = evaluator.add_leaf(
        id="ca_awards_capacity",
        desc="Verify the venue can accommodate at least 1,000 attendees for an awards ceremony",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} can accommodate at least 1,000 attendees for an awards ceremony.",
        node=ca_capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Look for ballroom or event space capacities in banquet or theater style. "
            "Accept if any single ballroom/space lists capacity ≥ 1,000 for banquet/theater style."
        ),
    )

    # Location = Los Angeles area
    ca_location_leaf = evaluator.add_leaf(
        id="ca_awards_location",
        desc="Verify the venue is located in the Los Angeles area of California",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} is in the Los Angeles area of California.",
        node=ca_location_leaf,
        sources=sources,
        additional_instruction=(
            "Los Angeles area includes Los Angeles city and nearby municipalities such as Beverly Hills, "
            "West Hollywood, Hollywood, Santa Monica, Pasadena, Burbank, Universal City, etc."
        ),
    )

    # Facility type = upscale hotel ballroom or similar
    ca_facility_leaf = evaluator.add_leaf(
        id="ca_awards_facility_type",
        desc="Verify the venue is an upscale hotel ballroom or similar facility appropriate for formal awards ceremonies",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} is an upscale hotel ballroom or similar high-end facility appropriate for formal awards ceremonies.",
        node=ca_facility_leaf,
        sources=sources,
        additional_instruction=(
            "Look for 'hotel ballroom', 'grand ballroom', 'luxury/upscale hotel', or comparable high-end event spaces. "
            "Accept well-known luxury/upper-upscale brands and venues hosting prestigious events."
        ),
    )

    # Insurance capability
    ca_insurance_leaf = evaluator.add_leaf(
        id="ca_awards_insurance",
        desc="Verify the venue can accommodate standard insurance requirements ($1M per occurrence, $2M aggregate)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} can accommodate standard event insurance requirements of at least $1M per occurrence and $2M aggregate, including providing/accepting COI.",
        node=ca_insurance_leaf,
        sources=sources,
        additional_instruction=(
            "Look for rental policies, event guidelines, or procurement requirements referencing Certificates of Insurance (COI) "
            "and minimums around $1,000,000 per occurrence and $2,000,000 aggregate. "
            "Accept explicit mention of COI requirements meeting or exceeding these amounts."
        ),
    )

    # Event features: banquet seating and stage area
    ca_features_leaf = evaluator.add_leaf(
        id="ca_awards_event_features",
        desc="Verify the venue has appropriate features including banquet seating capability and stage area",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} offers banquet seating capability and has a stage or can provide a stage area for ceremonies.",
        node=ca_features_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the venue supports banquet/rounds or theater seating layouts and mentions a stage or "
            "ability to set up a stage for ceremonies."
        ),
    )


async def verify_broadway_theater_venue(evaluator: Evaluator, parent_node, venue: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="broadway_theater_venue",
        desc="Verify a suitable Broadway theater is identified",
        parent=parent_node,
        critical=False
    )

    sources = _get_sources(venue)
    name = _safe_name(venue)

    # Critical: Reference URL present
    evaluator.add_custom_node(
        result=_has_valid_url(sources),
        id="broadway_reference",
        desc="Verify a valid reference URL is provided supporting the Broadway theater identification",
        parent=node,
        critical=True
    )

    # Capacity ≥ 500
    broadway_capacity_leaf = evaluator.add_leaf(
        id="broadway_capacity",
        desc="Verify the theater meets the minimum 500-seat capacity requirement for Broadway classification",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater {name} has a seating capacity of at least 500.",
        node=broadway_capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Check official theater specs or reputable sources for seating capacity. "
            "Accept if capacity is ≥ 500 seats."
        ),
    )

    # Location in major US city with established theater district
    broadway_location_leaf = evaluator.add_leaf(
        id="broadway_location",
        desc="Verify the theater is located in a major US city with established theater district",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater {name} is located in a major US city with an established theater district.",
        node=broadway_location_leaf,
        sources=sources,
        additional_instruction=(
            "Accept cities known for theater districts (e.g., New York City Theater District/Broadway, Chicago Loop, etc.). "
            "The page should indicate the theater lies in such a city/district."
        ),
    )

    # Adequate stage facilities
    broadway_stage_leaf = evaluator.add_leaf(
        id="broadway_stage_facilities",
        desc="Verify the theater has adequate stage facilities for theatrical productions",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater {name} has adequate stage facilities for theatrical productions (e.g., stage size, fly system, pit, backstage, dressing rooms).",
        node=broadway_stage_leaf,
        sources=sources,
        additional_instruction=(
            "Look for technical specs or descriptions indicating a full theatrical stage, fly system, orchestra pit, "
            "backstage areas, and dressing rooms."
        ),
    )

    # ADA accessibility
    broadway_ada_leaf = evaluator.add_leaf(
        id="broadway_ada",
        desc="Verify the theater meets ADA accessibility requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater {name} meets ADA accessibility requirements.",
        node=broadway_ada_leaf,
        sources=sources,
        additional_instruction=(
            "Look for accessibility policies, accessible seating, wheelchair access, assisted listening devices, etc."
        ),
    )

    # Insurance capability
    broadway_ins_leaf = evaluator.add_leaf(
        id="broadway_insurance_capability",
        desc="Verify the theater can accommodate standard venue insurance requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater {name} can accommodate standard venue insurance requirements for productions and events.",
        node=broadway_ins_leaf,
        sources=sources,
        additional_instruction=(
            "Check rental policies or guidelines referencing Certificates of Insurance (COI) or standard liability coverage "
            "acceptable for productions."
        ),
    )


async def verify_outdoor_amphitheater_venue(evaluator: Evaluator, parent_node, venue: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="outdoor_amphitheater_venue",
        desc="Verify a suitable outdoor amphitheater in a US state is identified",
        parent=parent_node,
        critical=False
    )

    sources = _get_sources(venue)
    name = _safe_name(venue)

    # Critical: Reference URL present
    evaluator.add_custom_node(
        result=_has_valid_url(sources),
        id="amphitheater_reference",
        desc="Verify a valid reference URL is provided supporting the outdoor amphitheater identification",
        parent=node,
        critical=True
    )

    # Capacity ≥ 5,000
    amp_capacity_leaf = evaluator.add_leaf(
        id="amphitheater_capacity",
        desc="Verify the amphitheater has a minimum capacity of 5,000 for outdoor concerts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The amphitheater {name} has a capacity of at least 5,000 for outdoor concerts.",
        node=amp_capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Verify stated capacities on official or reputable pages. Accept if capacity is ≥ 5,000."
        ),
    )

    # US location
    amp_location_leaf = evaluator.add_leaf(
        id="amphitheater_us_location",
        desc="Verify the amphitheater is located in the United States",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The amphitheater {name} is located in the United States.",
        node=amp_location_leaf,
        sources=sources,
        additional_instruction=(
            "Look for city/state information within the USA on the venue page."
        ),
    )

    # Outdoor configuration
    amp_outdoor_leaf = evaluator.add_leaf(
        id="amphitheater_outdoor_config",
        desc="Verify the venue is configured as an outdoor amphitheater",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {name} is configured as an outdoor amphitheater.",
        node=amp_outdoor_leaf,
        sources=sources,
        additional_instruction=(
            "Look for phrases like 'outdoor amphitheater', 'open-air', lawn seating, or photos indicating outdoor configuration."
        ),
    )

    # Stage depth ≥ 16 feet
    amp_stage_depth_leaf = evaluator.add_leaf(
        id="amphitheater_stage_depth",
        desc="Verify the venue has adequate stage dimensions including minimum 16 feet stage depth for full band setup",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The amphitheater {name} has a stage depth of at least 16 feet.",
        node=amp_stage_depth_leaf,
        sources=sources,
        additional_instruction=(
            "Check technical specs or production guides for stage dimensions. "
            "Accept if the stage depth is explicitly listed as ≥ 16 feet."
        ),
    )

    # Permit and temporary event accommodation
    amp_permit_leaf = evaluator.add_leaf(
        id="amphitheater_permit_compliance",
        desc="Verify the venue operates with proper permits and can accommodate temporary event requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The amphitheater {name} operates with proper permits and can accommodate temporary event requirements.",
        node=amp_permit_leaf,
        sources=sources,
        additional_instruction=(
            "Look for venue policies, municipal compliance notes, permitting information, or rental guidelines "
            "indicating adherence to permits and temporary event operations."
        ),
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
    Evaluate an answer for the four-venue 2026 entertainment tour task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root aggregates four independent event verifications
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

    # Extract proposed venues and their sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="tour_venues_extraction"
    )

    # Optional: record task metadata
    evaluator.add_custom_info(
        info={"year": 2026, "events_expected": 4},
        info_type="task_metadata",
        info_name="tour_task_metadata"
    )

    # Build and verify each event subtree
    await verify_new_york_concert_venue(evaluator, root, extracted.ny_concert if extracted else None)
    await verify_california_awards_venue(evaluator, root, extracted.la_awards if extracted else None)
    await verify_broadway_theater_venue(evaluator, root, extracted.broadway_theater if extracted else None)
    await verify_outdoor_amphitheater_venue(evaluator, root, extracted.outdoor_amphitheater if extracted else None)

    return evaluator.get_summary()