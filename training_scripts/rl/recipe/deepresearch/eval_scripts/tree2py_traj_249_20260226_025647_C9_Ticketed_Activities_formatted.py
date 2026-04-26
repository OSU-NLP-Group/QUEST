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
TASK_ID = "fifa2026_venues_5"
TASK_DESCRIPTION = """
For the 2026 FIFA World Cup being held across North America, identify the specific stadiums serving as official venues for five designated host cities, ensuring each venue meets the following requirements:

1. Identify the stadium hosting the FIFA World Cup 2026 Final match, which must be located in the New York/New Jersey area, have a seating capacity of at least 80,000 for World Cup matches, and host exactly 8 matches during the tournament.

2. Identify the FIFA World Cup 2026 stadium located in Georgia (Atlanta area), which must have a seating capacity of at least 70,000 for World Cup matches, host exactly 8 matches during the tournament, and host at least one semifinal match.

3. Identify the FIFA World Cup 2026 stadium located in Santa Clara (San Francisco Bay Area), California, which must have a seating capacity between 68,000 and 72,000 for World Cup matches, and host at least 6 matches during the tournament.

4. Identify the FIFA World Cup 2026 stadium located in Kansas City, Missouri, which must have a seating capacity of at least 73,000 for World Cup matches, and host at least 6 matches during the tournament.

5. Identify the FIFA World Cup 2026 stadium located in Miami, Florida, which must have a seating capacity of at least 65,000 for World Cup matches, and host exactly 7 matches during the tournament.

For each of the five venues, provide: (a) the official stadium name, (b) the specific location (city and state), (c) the exact seating capacity for FIFA World Cup 2026 matches, (d) the exact number of matches hosted at that venue, and (e) verification through official FIFA World Cup 2026 sources or official stadium websites that confirm these details.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumVenue(BaseModel):
    stadium_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_wc: Optional[str] = None  # keep as string to maximize compatibility
    matches_count: Optional[str] = None  # exact number of matches as stated in the answer
    hosts_final: Optional[bool] = None
    hosts_semifinal: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    final_venue: Optional[StadiumVenue] = None
    georgia_venue: Optional[StadiumVenue] = None
    california_venue: Optional[StadiumVenue] = None
    kansas_city_venue: Optional[StadiumVenue] = None
    miami_venue: Optional[StadiumVenue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for five designated FIFA World Cup 2026 venues mentioned in the answer.
    Use the following JSON schema. If an item is missing in the answer, set it to null (or empty array for sources).

    Object keys:
    - final_venue: The stadium designated to host the FIFA World Cup 2026 Final (NY/NJ area).
    - georgia_venue: The stadium in Georgia (Atlanta area).
    - california_venue: The stadium in Santa Clara, California.
    - kansas_city_venue: The stadium in Kansas City, Missouri.
    - miami_venue: The stadium in Miami, Florida (Miami Gardens area).

    For each object, extract:
    - stadium_name: Official stadium name.
    - city: City name.
    - state: State name.
    - capacity_wc: The exact seating capacity specifically for FIFA World Cup 2026 matches, as stated in the answer (string).
    - matches_count: The exact number of FIFA World Cup 2026 matches hosted at the venue (string).
    - hosts_final: true/false if the venue is stated to host the Final; null if unspecified.
    - hosts_semifinal: true/false if the venue is stated to host at least one semifinal; null if unspecified.
    - sources: Array of URL strings explicitly cited in the answer that confirm any of the above details. Include only actual URLs (plain or markdown), not named references.

    Do not invent any data. If the answer mentions multiple sources, include all.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""

def _sources_list(v: Optional[StadiumVenue]) -> List[str]:
    return (v.sources if v and v.sources else [])

# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_final_venue(evaluator: Evaluator, parent_node, info: Optional[StadiumVenue]) -> None:
    """
    Build and verify the subtree for the Final venue (NY/NJ area).
    """
    venue_node = evaluator.add_parallel(
        id="final_venue",
        desc="Identify the stadium hosting the FIFA World Cup 2026 Final, located in the New York/New Jersey area, with capacity ≥80,000 and hosting exactly 8 matches",
        parent=parent_node,
        critical=False
    )

    name = _safe(info.stadium_name)
    city = _safe(info.city)
    state = _safe(info.state)
    capacity_wc = _safe(info.capacity_wc)
    matches_count = _safe(info.matches_count)
    sources = _sources_list(info)

    # Basic identification (critical)
    basic_node = evaluator.add_parallel(
        id="final_venue_basic_identification",
        desc="Provide basic identifying information for the Final venue",
        parent=venue_node,
        critical=True
    )

    # Leaves
    leaf_name = evaluator.add_leaf(
        id="final_venue_name",
        desc="Provide the official stadium name for the venue hosting the FIFA World Cup 2026 Final",
        parent=basic_node,
        critical=True
    )
    leaf_location = evaluator.add_leaf(
        id="final_venue_location",
        desc="Verify the Final venue is located in the New York/New Jersey area (specific city and state)",
        parent=basic_node,
        critical=True
    )
    leaf_basic_ref = evaluator.add_leaf(
        id="final_venue_basic_reference",
        desc="Provide valid URL(s) from official FIFA or stadium sources confirming the venue name and location",
        parent=basic_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The official stadium name for the FIFA World Cup 2026 Final venue is '{name}'.",
            sources,
            leaf_name,
            "Confirm the official stadium name using the provided sources; allow minor naming variants (sponsor/corporate naming)."
        ),
        (
            f"The Final venue is located in {city}, {state}, in the New York/New Jersey area.",
            sources,
            leaf_location,
            "Confirm both city and state; acknowledge that East Rutherford, NJ and similar are in the NY/NJ area."
        ),
        (
            f"The provided sources explicitly confirm the venue's official stadium name '{name}' and its location ({city}, {state}).",
            sources,
            leaf_basic_ref,
            "Use only the content in the provided URLs to confirm both name and location."
        ),
    ])

    # Compliance verification (critical)
    compliance_node = evaluator.add_parallel(
        id="final_venue_compliance_verification",
        desc="Verify the Final venue meets all capacity and match-hosting requirements",
        parent=venue_node,
        critical=True
    )

    leaf_capacity = evaluator.add_leaf(
        id="final_venue_capacity_requirement",
        desc="Verify the Final venue has a seating capacity of at least 80,000 for FIFA World Cup matches",
        parent=compliance_node,
        critical=True
    )
    leaf_matches = evaluator.add_leaf(
        id="final_venue_matches_requirement",
        desc="Verify the Final venue hosts exactly 8 FIFA World Cup 2026 matches",
        parent=compliance_node,
        critical=True
    )
    leaf_final_host = evaluator.add_leaf(
        id="final_venue_hosts_final_match",
        desc="Verify this venue is designated to host the FIFA World Cup 2026 Final match",
        parent=compliance_node,
        critical=True
    )
    leaf_compliance_ref = evaluator.add_leaf(
        id="final_venue_compliance_reference",
        desc="Provide valid URL(s) from official sources confirming capacity, match count, and Final match hosting",
        parent=compliance_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The stadium has a seating capacity of at least 80,000 for FIFA World Cup 2026 matches (the answer states capacity '{capacity_wc}').",
            sources,
            leaf_capacity,
            "Confirm the capacity for World Cup configuration; if a range or exact number >= 80,000 is cited by official sources, pass."
        ),
        (
            f"This venue hosts exactly 8 FIFA World Cup 2026 matches (the answer states '{matches_count}').",
            sources,
            leaf_matches,
            "Validate match count using official FIFA match schedule pages."
        ),
        (
            "This venue is designated to host the FIFA World Cup 2026 Final match.",
            sources,
            leaf_final_host,
            "Confirm with official FIFA schedule/announcement pages."
        ),
        (
            "The provided sources explicitly confirm the venue's seating capacity (for World Cup matches), the exact total of 8 matches, and Final match hosting.",
            sources,
            leaf_compliance_ref,
            "Ensure the sources directly support these exact details."
        ),
    ])


async def verify_georgia_venue(evaluator: Evaluator, parent_node, info: Optional[StadiumVenue]) -> None:
    """
    Georgia (Atlanta area): capacity ≥70,000, exactly 8 matches, hosts ≥1 semifinal.
    """
    venue_node = evaluator.add_parallel(
        id="georgia_venue",
        desc="Identify the FIFA World Cup 2026 stadium in Georgia (Atlanta area), with capacity ≥70,000, hosting exactly 8 matches including at least one semifinal",
        parent=parent_node,
        critical=False
    )

    name = _safe(info.stadium_name)
    city = _safe(info.city)
    state = _safe(info.state)
    capacity_wc = _safe(info.capacity_wc)
    matches_count = _safe(info.matches_count)
    sources = _sources_list(info)

    basic_node = evaluator.add_parallel(
        id="georgia_venue_basic_identification",
        desc="Provide basic identifying information for the Georgia venue",
        parent=venue_node,
        critical=True
    )

    leaf_name = evaluator.add_leaf(
        id="georgia_venue_name",
        desc="Provide the official stadium name for the FIFA World Cup 2026 venue in Georgia",
        parent=basic_node,
        critical=True
    )
    leaf_location = evaluator.add_leaf(
        id="georgia_venue_location",
        desc="Verify the venue is located in Georgia (Atlanta area) with specific city and state",
        parent=basic_node,
        critical=True
    )
    leaf_basic_ref = evaluator.add_leaf(
        id="georgia_venue_basic_reference",
        desc="Provide valid URL(s) from official sources confirming the venue name and location",
        parent=basic_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The official stadium name for the Georgia (Atlanta area) venue is '{name}'.",
            sources,
            leaf_name,
            "Confirm the official stadium name via provided sources."
        ),
        (
            f"The venue is located in {city}, {state}, in the Atlanta, Georgia area.",
            sources,
            leaf_location,
            "Confirm both city and state; ensure it is clearly in Georgia (Atlanta area)."
        ),
        (
            f"The provided sources confirm the stadium name '{name}' and the location ({city}, {state}).",
            sources,
            leaf_basic_ref,
            "Use only the provided URLs."
        ),
    ])

    compliance_node = evaluator.add_parallel(
        id="georgia_venue_compliance_verification",
        desc="Verify the Georgia venue meets all capacity and match-hosting requirements",
        parent=venue_node,
        critical=True
    )

    leaf_capacity = evaluator.add_leaf(
        id="georgia_venue_capacity_requirement",
        desc="Verify the Georgia venue has a seating capacity of at least 70,000 for FIFA World Cup matches",
        parent=compliance_node,
        critical=True
    )
    leaf_matches = evaluator.add_leaf(
        id="georgia_venue_matches_requirement",
        desc="Verify the Georgia venue hosts exactly 8 FIFA World Cup 2026 matches",
        parent=compliance_node,
        critical=True
    )
    leaf_semifinal = evaluator.add_leaf(
        id="georgia_venue_semifinal_requirement",
        desc="Verify the Georgia venue hosts at least one semifinal match during FIFA World Cup 2026",
        parent=compliance_node,
        critical=True
    )
    leaf_compliance_ref = evaluator.add_leaf(
        id="georgia_venue_compliance_reference",
        desc="Provide valid URL(s) from official sources confirming capacity, match count, and semifinal hosting",
        parent=compliance_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The stadium has a seating capacity of at least 70,000 for FIFA World Cup 2026 matches (the answer states capacity '{capacity_wc}').",
            sources,
            leaf_capacity,
            "Confirm capacity for the tournament configuration."
        ),
        (
            f"This venue hosts exactly 8 FIFA World Cup 2026 matches (the answer states '{matches_count}').",
            sources,
            leaf_matches,
            "Confirm using official match schedule."
        ),
        (
            "This venue hosts at least one semifinal match during FIFA World Cup 2026.",
            sources,
            leaf_semifinal,
            "Confirm semifinal hosting using official FIFA schedule/announcements."
        ),
        (
            "The provided sources explicitly confirm the venue's capacity (World Cup configuration), exact total of 8 matches, and semifinal hosting.",
            sources,
            leaf_compliance_ref,
            "Ensure explicit support from the sources."
        ),
    ])


async def verify_california_venue(evaluator: Evaluator, parent_node, info: Optional[StadiumVenue]) -> None:
    """
    California (Santa Clara): capacity between 68,000-72,000, hosts at least 6 matches.
    """
    venue_node = evaluator.add_parallel(
        id="california_venue",
        desc="Identify the FIFA World Cup 2026 stadium in Santa Clara (San Francisco Bay Area), California, with capacity between 68,000-72,000 and hosting at least 6 matches",
        parent=parent_node,
        critical=False
    )

    name = _safe(info.stadium_name)
    city = _safe(info.city)
    state = _safe(info.state)
    capacity_wc = _safe(info.capacity_wc)
    matches_count = _safe(info.matches_count)
    sources = _sources_list(info)

    basic_node = evaluator.add_parallel(
        id="california_venue_basic_identification",
        desc="Provide basic identifying information for the California venue",
        parent=venue_node,
        critical=True
    )

    leaf_name = evaluator.add_leaf(
        id="california_venue_name",
        desc="Provide the official stadium name for the FIFA World Cup 2026 venue in Santa Clara, California",
        parent=basic_node,
        critical=True
    )
    leaf_location = evaluator.add_leaf(
        id="california_venue_location",
        desc="Verify the venue is located in Santa Clara (San Francisco Bay Area), California with specific city and state",
        parent=basic_node,
        critical=True
    )
    leaf_basic_ref = evaluator.add_leaf(
        id="california_venue_basic_reference",
        desc="Provide valid URL(s) from official sources confirming the venue name and location",
        parent=basic_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The official stadium name for the Santa Clara venue is '{name}'.",
            sources,
            leaf_name,
            "Confirm the official stadium name via provided sources."
        ),
        (
            f"The venue is located in {city}, {state}, in the Santa Clara (San Francisco Bay Area) of California.",
            sources,
            leaf_location,
            "Confirm both city and state; ensure Santa Clara, California."
        ),
        (
            f"The provided sources confirm the stadium name '{name}' and the location ({city}, {state}).",
            sources,
            leaf_basic_ref,
            "Use only the provided URLs."
        ),
    ])

    compliance_node = evaluator.add_parallel(
        id="california_venue_compliance_verification",
        desc="Verify the California venue meets all capacity and match-hosting requirements",
        parent=venue_node,
        critical=True
    )

    leaf_capacity = evaluator.add_leaf(
        id="california_venue_capacity_requirement",
        desc="Verify the California venue has a seating capacity between 68,000 and 72,000 for FIFA World Cup matches",
        parent=compliance_node,
        critical=True
    )
    leaf_matches = evaluator.add_leaf(
        id="california_venue_matches_requirement",
        desc="Verify the California venue hosts at least 6 FIFA World Cup 2026 matches",
        parent=compliance_node,
        critical=True
    )
    leaf_compliance_ref = evaluator.add_leaf(
        id="california_venue_compliance_reference",
        desc="Provide valid URL(s) from official sources confirming capacity and match count",
        parent=compliance_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The stadium has a seating capacity between 68,000 and 72,000 for FIFA World Cup 2026 matches (the answer states capacity '{capacity_wc}').",
            sources,
            leaf_capacity,
            "Confirm that the official capacity for World Cup configuration lies within the stated range."
        ),
        (
            f"This venue hosts at least 6 FIFA World Cup 2026 matches (the answer states '{matches_count}').",
            sources,
            leaf_matches,
            "Confirm using official match schedule; at least 6 matches must be hosted."
        ),
        (
            "The provided sources explicitly confirm the venue's seating capacity (World Cup configuration) and the total matches hosted (≥6).",
            sources,
            leaf_compliance_ref,
            "Ensure explicit support from the sources."
        ),
    ])


async def verify_kansas_city_venue(evaluator: Evaluator, parent_node, info: Optional[StadiumVenue]) -> None:
    """
    Kansas City, Missouri: capacity ≥73,000, hosts at least 6 matches.
    """
    venue_node = evaluator.add_parallel(
        id="kansas_city_venue",
        desc="Identify the FIFA World Cup 2026 stadium in Kansas City, Missouri, with capacity ≥73,000 and hosting at least 6 matches",
        parent=parent_node,
        critical=False
    )

    name = _safe(info.stadium_name)
    city = _safe(info.city)
    state = _safe(info.state)
    capacity_wc = _safe(info.capacity_wc)
    matches_count = _safe(info.matches_count)
    sources = _sources_list(info)

    basic_node = evaluator.add_parallel(
        id="kansas_city_venue_basic_identification",
        desc="Provide basic identifying information for the Kansas City venue",
        parent=venue_node,
        critical=True
    )

    leaf_name = evaluator.add_leaf(
        id="kansas_city_venue_name",
        desc="Provide the official stadium name for the FIFA World Cup 2026 venue in Kansas City, Missouri",
        parent=basic_node,
        critical=True
    )
    leaf_location = evaluator.add_leaf(
        id="kansas_city_venue_location",
        desc="Verify the venue is located in Kansas City, Missouri with specific city and state",
        parent=basic_node,
        critical=True
    )
    leaf_basic_ref = evaluator.add_leaf(
        id="kansas_city_venue_basic_reference",
        desc="Provide valid URL(s) from official sources confirming the venue name and location",
        parent=basic_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The official stadium name for the Kansas City venue is '{name}'.",
            sources,
            leaf_name,
            "Confirm the official stadium name via provided sources."
        ),
        (
            f"The venue is located in {city}, {state}, i.e., Kansas City, Missouri.",
            sources,
            leaf_location,
            "Confirm both city and state; ensure Missouri."
        ),
        (
            f"The provided sources confirm the stadium name '{name}' and the location ({city}, {state}).",
            sources,
            leaf_basic_ref,
            "Use only the provided URLs."
        ),
    ])

    compliance_node = evaluator.add_parallel(
        id="kansas_city_venue_compliance_verification",
        desc="Verify the Kansas City venue meets all capacity and match-hosting requirements",
        parent=venue_node,
        critical=True
    )

    leaf_capacity = evaluator.add_leaf(
        id="kansas_city_venue_capacity_requirement",
        desc="Verify the Kansas City venue has a seating capacity of at least 73,000 for FIFA World Cup matches",
        parent=compliance_node,
        critical=True
    )
    leaf_matches = evaluator.add_leaf(
        id="kansas_city_venue_matches_requirement",
        desc="Verify the Kansas City venue hosts at least 6 FIFA World Cup 2026 matches",
        parent=compliance_node,
        critical=True
    )
    leaf_compliance_ref = evaluator.add_leaf(
        id="kansas_city_venue_compliance_reference",
        desc="Provide valid URL(s) from official sources confirming capacity and match count",
        parent=compliance_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The stadium has a seating capacity of at least 73,000 for FIFA World Cup 2026 matches (the answer states capacity '{capacity_wc}').",
            sources,
            leaf_capacity,
            "Confirm the capacity for the World Cup configuration."
        ),
        (
            f"This venue hosts at least 6 FIFA World Cup 2026 matches (the answer states '{matches_count}').",
            sources,
            leaf_matches,
            "Confirm using official match schedule."
        ),
        (
            "The provided sources explicitly confirm the venue's seating capacity (World Cup configuration) and the total matches hosted (≥6).",
            sources,
            leaf_compliance_ref,
            "Ensure explicit support from the sources."
        ),
    ])


async def verify_miami_venue(evaluator: Evaluator, parent_node, info: Optional[StadiumVenue]) -> None:
    """
    Miami, Florida (Miami Gardens area): capacity ≥65,000, hosts exactly 7 matches.
    """
    venue_node = evaluator.add_parallel(
        id="miami_venue",
        desc="Identify the FIFA World Cup 2026 stadium in Miami, Florida, with capacity ≥65,000 and hosting exactly 7 matches",
        parent=parent_node,
        critical=False
    )

    name = _safe(info.stadium_name)
    city = _safe(info.city)
    state = _safe(info.state)
    capacity_wc = _safe(info.capacity_wc)
    matches_count = _safe(info.matches_count)
    sources = _sources_list(info)

    basic_node = evaluator.add_parallel(
        id="miami_venue_basic_identification",
        desc="Provide basic identifying information for the Miami venue",
        parent=venue_node,
        critical=True
    )

    leaf_name = evaluator.add_leaf(
        id="miami_venue_name",
        desc="Provide the official stadium name for the FIFA World Cup 2026 venue in Miami, Florida",
        parent=basic_node,
        critical=True
    )
    leaf_location = evaluator.add_leaf(
        id="miami_venue_location",
        desc="Verify the venue is located in Miami, Florida (Miami Gardens area) with specific city and state",
        parent=basic_node,
        critical=True
    )
    leaf_basic_ref = evaluator.add_leaf(
        id="miami_venue_basic_reference",
        desc="Provide valid URL(s) from official sources confirming the venue name and location",
        parent=basic_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The official stadium name for the Miami venue is '{name}'.",
            sources,
            leaf_name,
            "Confirm the official stadium name via provided sources."
        ),
        (
            f"The venue is located in {city}, {state}, in the Miami, Florida (Miami Gardens area).",
            sources,
            leaf_location,
            "Confirm both city and state; acknowledge Miami Gardens is part of the Miami area."
        ),
        (
            f"The provided sources confirm the stadium name '{name}' and the location ({city}, {state}).",
            sources,
            leaf_basic_ref,
            "Use only the provided URLs."
        ),
    ])

    compliance_node = evaluator.add_parallel(
        id="miami_venue_compliance_verification",
        desc="Verify the Miami venue meets all capacity and match-hosting requirements",
        parent=venue_node,
        critical=True
    )

    leaf_capacity = evaluator.add_leaf(
        id="miami_venue_capacity_requirement",
        desc="Verify the Miami venue has a seating capacity of at least 65,000 for FIFA World Cup matches",
        parent=compliance_node,
        critical=True
    )
    leaf_matches = evaluator.add_leaf(
        id="miami_venue_matches_requirement",
        desc="Verify the Miami venue hosts exactly 7 FIFA World Cup 2026 matches",
        parent=compliance_node,
        critical=True
    )
    leaf_compliance_ref = evaluator.add_leaf(
        id="miami_venue_compliance_reference",
        desc="Provide valid URL(s) from official sources confirming capacity and match count",
        parent=compliance_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The stadium has a seating capacity of at least 65,000 for FIFA World Cup 2026 matches (the answer states capacity '{capacity_wc}').",
            sources,
            leaf_capacity,
            "Confirm capacity for World Cup configuration."
        ),
        (
            f"This venue hosts exactly 7 FIFA World Cup 2026 matches (the answer states '{matches_count}').",
            sources,
            leaf_matches,
            "Confirm using official match schedule."
        ),
        (
            "The provided sources explicitly confirm the venue's seating capacity (World Cup configuration) and exactly 7 hosted matches.",
            sources,
            leaf_compliance_ref,
            "Ensure explicit support from the sources."
        ),
    ])


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
    Evaluate a single answer for the FIFA World Cup 2026 venues task.
    """
    evaluator = Evaluator()
    # Note: Set root critical to False to allow partial credit and avoid critical-child constraint violations
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

    # Extract structured venue information from the answer
    venues_info = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build and verify each venue subtree
    await verify_final_venue(evaluator, root, venues_info.final_venue)
    await verify_georgia_venue(evaluator, root, venues_info.georgia_venue)
    await verify_california_venue(evaluator, root, venues_info.california_venue)
    await verify_kansas_city_venue(evaluator, root, venues_info.kansas_city_venue)
    await verify_miami_venue(evaluator, root, venues_info.miami_venue)

    # Optional: add custom info block summarizing extracted venue names
    evaluator.add_custom_info({
        "final_venue_name": venues_info.final_venue.stadium_name if venues_info.final_venue else None,
        "georgia_venue_name": venues_info.georgia_venue.stadium_name if venues_info.georgia_venue else None,
        "california_venue_name": venues_info.california_venue.stadium_name if venues_info.california_venue else None,
        "kansas_city_venue_name": venues_info.kansas_city_venue.stadium_name if venues_info.kansas_city_venue else None,
        "miami_venue_name": venues_info.miami_venue.stadium_name if venues_info.miami_venue else None,
    }, info_type="extraction_summary")

    # Return evaluation summary
    return evaluator.get_summary()