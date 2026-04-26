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
TASK_ID = "nyc_performance_venues"
TASK_DESCRIPTION = (
    "I am planning a comprehensive guide to live performance venues in New York City for a travel publication. "
    "I need detailed information about four different types of performance venues, each meeting specific criteria:\n\n"
    "1. Broadway Theater: Identify a Broadway theater that currently hosts a long-running musical, has a seating capacity "
    "of at least 1,500 seats, and is located within the official Theater District boundaries (between 41st and 54th Streets, "
    "and between 6th and 8th Avenues). Provide the theater name, exact seating capacity, street address confirming Theater "
    "District location, the name of the currently running musical, and the musical's total runtime including intermission.\n\n"
    "2. Television Studio: Identify a television studio in New York City that hosts a late-night talk show. The show must "
    "tape on weekdays (Monday through Friday), not on weekends. Provide the studio name, the show name, confirmation of weekday "
    "taping schedule, the minimum age requirement for audience members, and the approximate audience seating capacity of the studio.\n\n"
    "3. Large Concert Hall: Identify a historic concert hall located in Manhattan with a seating capacity between 5,000 and 7,000 seats. "
    "The venue must have been built before 1950. Provide the venue name, exact seating capacity, Manhattan location, and the year it was built or opened.\n\n"
    "4. Off-Broadway Theater: Identify an Off-Broadway theater with a seating capacity between 100 and 499 seats that provides ADA-compliant accessible seating "
    "(wheelchair spaces or transfer seats). Provide the theater name, exact seating capacity confirming Off-Broadway classification, confirmation of ADA "
    "accessibility features, and the name of a current or recent production at this theater.\n\n"
    "For each venue, include a reference URL that supports the provided information."
)

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class BroadwayVenue(BaseModel):
    name: Optional[str] = None
    seating_capacity: Optional[str] = None
    address: Optional[str] = None
    current_musical: Optional[str] = None
    runtime_including_intermission: Optional[str] = None
    ada_features: Optional[str] = None
    long_running_evidence: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TVStudioVenue(BaseModel):
    studio_name: Optional[str] = None
    show_name: Optional[str] = None
    weekday_taping_statement: Optional[str] = None
    minimum_age_requirement: Optional[str] = None
    audience_capacity_approx: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ConcertHallVenue(BaseModel):
    name: Optional[str] = None
    seating_capacity: Optional[str] = None
    manhattan_location_statement: Optional[str] = None
    year_built_or_opened: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OffBroadwayVenue(BaseModel):
    name: Optional[str] = None
    seating_capacity: Optional[str] = None
    ada_features: Optional[str] = None
    current_or_recent_production: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AllVenuesExtraction(BaseModel):
    broadway: Optional[BroadwayVenue] = None
    tv_studio: Optional[TVStudioVenue] = None
    concert_hall: Optional[ConcertHallVenue] = None
    off_broadway: Optional[OffBroadwayVenue] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information for four NYC venue types from the answer. If an item is missing, set it to null (or [] for sources). 
    For each venue, also extract all reference URLs explicitly mentioned in the answer that support any of the facts for that venue.
    
    1) Broadway theater (fields: name, seating_capacity, address, current_musical, runtime_including_intermission, ada_features, long_running_evidence, sources[])
       - seating_capacity: extract as written (e.g., "1,761" or "1761 seats")
       - address: full street address as written
       - runtime_including_intermission: include any qualifier like "including intermission"
       - ada_features: any text indicating wheelchair/transfer/companion seating or ADA accessibility
       - long_running_evidence: any phrase/dates indicating it is “long-running” (e.g., opening year and still running, or explicitly called long-running)
       - sources: list of URLs cited for this venue (official site, Broadway League, ticketing, etc.)

    2) Television studio (fields: studio_name, show_name, weekday_taping_statement, minimum_age_requirement, audience_capacity_approx, sources[])
       - weekday_taping_statement: text confirming weekdays taping (Mon–Fri) and not weekends; allow Mon–Thu/M–F formulations
       - minimum_age_requirement: extract as written, e.g., "16+" or "minimum age 18"
       - audience_capacity_approx: extract as given (approximate number)
       - sources: list of URLs cited for this venue/show

    3) Large concert hall (fields: name, seating_capacity, manhattan_location_statement, year_built_or_opened, sources[])
       - seating_capacity: extract as written
       - manhattan_location_statement: text that confirms the venue is located in Manhattan, NYC
       - year_built_or_opened: extract the year (as written in the answer)
       - sources: list of URLs cited for this venue

    4) Off-Broadway theater (fields: name, seating_capacity, ada_features, current_or_recent_production, sources[])
       - seating_capacity: extract as written
       - ada_features: accessibility specifics (wheelchair spaces, transfer/companion seating, etc.)
       - current_or_recent_production: a show name at this venue currently or recently
       - sources: list of URLs cited for this venue

    Return a JSON object with keys: 
    {
      "broadway": {...},
      "tv_studio": {...},
      "concert_hall": {...},
      "off_broadway": {...}
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""

def _sources_or_none(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic filtering: ensure strings and non-empty
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_broadway(
    evaluator: Evaluator,
    root_node,
    data: Optional[BroadwayVenue]
) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_1_Broadway_Theater",
        desc="Broadway theater that meets Theater District geography and capacity threshold and currently hosts a long-running musical; provide required details and supporting URL(s).",
        parent=root_node,
        critical=False
    )

    sources = _sources_or_none(data.sources if data else None)
    # Supporting reference URL existence (critical)
    evaluator.add_custom_node(
        result=bool(sources),
        id="V1_Supporting_Reference_URL",
        desc="Provide at least one reference URL that supports the Broadway venue information (capacity/location/show/runtime/long-running status as applicable).",
        parent=venue_node,
        critical=True
    )

    # V1 Theater Name
    v1_name = evaluator.add_leaf(
        id="V1_Theater_Name",
        desc="Provide the name of the Broadway theater.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Broadway theater is named '{_safe(data.name) if data else ''}'.",
        node=v1_name,
        sources=sources,
        additional_instruction="Verify the exact theater name as shown on the cited page(s). Allow minor punctuation or style variations."
    )

    # V1 Seating Capacity >= 1500
    v1_cap = evaluator.add_leaf(
        id="V1_Seating_Capacity_Min1500",
        desc="Provide the exact seating capacity and verify it is at least 1,500 seats.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater has a seating capacity of {_safe(data.seating_capacity) if data else ''}, which is at least 1,500 seats.",
        node=v1_cap,
        sources=sources,
        additional_instruction="Confirm both the stated capacity value and that it meets or exceeds 1,500 seats as supported by the page(s)."
    )

    # V1 Address within Theater District bounds
    v1_addr = evaluator.add_leaf(
        id="V1_Address_Theater_District_Bounds",
        desc="Provide the street address and verify it lies within 41st–54th Streets and 6th–8th Avenues.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The theater's address is '{_safe(data.address) if data else ''}', and this address lies within the Theater District boundaries "
            f"(between 41st and 54th Streets, and between 6th and 8th Avenues)."
        ),
        node=v1_addr,
        sources=sources,
        additional_instruction="Use the provided address on the page(s) to judge if the street number is between 41st–54th and avenues between 6th–8th. It is acceptable if the address clearly indicates a street in this range and cross-streets within 6th–8th."
    )

    # V1 ADA Accessible Seating
    v1_ada = evaluator.add_leaf(
        id="V1_ADA_Accessible_Seating",
        desc="Confirm the theater provides ADA-compliant accessible seating (e.g., wheelchair spaces and/or transfer/companion seats).",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="The theater provides ADA-compliant accessible seating (for example, wheelchair spaces and/or transfer/companion seating).",
        node=v1_ada,
        sources=sources,
        additional_instruction="Look for wheelchair accessibility or ADA seating details on the page(s). Phrases like 'wheelchair seating', 'accessible seating', or 'ADA-compliant' support this."
    )

    # V1 Current Musical Name
    v1_show = evaluator.add_leaf(
        id="V1_Current_Musical_Name",
        desc="Provide the name of the currently running musical at the theater.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The currently running musical at this theater is '{_safe(data.current_musical) if data else ''}'.",
        node=v1_show,
        sources=sources,
        additional_instruction="Verify the currently running musical as stated on official or credible ticketing/listing pages."
    )

    # V1 Long-Running Status
    v1_long = evaluator.add_leaf(
        id="V1_Long_Running_Status",
        desc="Verify the currently running musical qualifies as long-running (by credible evidence such as long run length, opening date plus still-running status, or explicit characterization as long-running).",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The musical '{_safe(data.current_musical) if data else ''}' is a long-running Broadway production.",
        node=v1_long,
        sources=sources,
        additional_instruction="Accept evidence such as (a) explicit 'long-running' wording; (b) an opening year far in the past with continued performances; or (c) credible references to number of performances or 'longest-running' lists."
    )

    # V1 Musical Runtime Including Intermission
    v1_runtime = evaluator.add_leaf(
        id="V1_Musical_Runtime_Including_Intermission",
        desc="Provide the musical's total runtime including intermission.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The musical's total runtime including intermission is {_safe(data.runtime_including_intermission) if data else ''}.",
        node=v1_runtime,
        sources=sources,
        additional_instruction="Verify the specific runtime including intermission as listed on official show or ticketing pages. Accept small rounding differences."
    )


async def build_and_verify_tv_studio(
    evaluator: Evaluator,
    root_node,
    data: Optional[TVStudioVenue]
) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_2_Television_Studio",
        desc="NYC television studio that hosts a late-night talk show with weekday tapings (no weekend tapings); provide required details and supporting URL(s).",
        parent=root_node,
        critical=False
    )

    sources = _sources_or_none(data.sources if data else None)
    evaluator.add_custom_node(
        result=bool(sources),
        id="V2_Supporting_Reference_URL",
        desc="Provide at least one reference URL that supports the studio/show information (weekday taping schedule/age requirement/capacity).",
        parent=venue_node,
        critical=True
    )

    # Studio Name
    v2_studio = evaluator.add_leaf(
        id="V2_Studio_Name",
        desc="Provide the name of the television studio.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The television studio is '{_safe(data.studio_name) if data else ''}'.",
        node=v2_studio,
        sources=sources,
        additional_instruction="Verify the studio's exact name on the cited page(s)."
    )

    # Show Name
    v2_show = evaluator.add_leaf(
        id="V2_Show_Name",
        desc="Provide the name of the late-night talk show hosted/filmed at this studio.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The studio hosts the late-night talk show '{_safe(data.show_name) if data else ''}'.",
        node=v2_show,
        sources=sources,
        additional_instruction="Confirm that the show is filmed/hosted at the specified studio."
    )

    # Weekday Taping (no weekends)
    v2_weekday = evaluator.add_leaf(
        id="V2_Weekday_Taping_No_Weekends",
        desc="Confirm the show tapes on weekdays (Monday through Friday) and not on weekends.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="The show tapes on weekdays (Monday through Friday) and does not tape on weekends.",
        node=v2_weekday,
        sources=sources,
        additional_instruction="Accept schedules like 'Mon–Thu' or 'Mon–Fri' as evidence of weekday tapings; explicitly verify no weekend tapings (Sat/Sun) are indicated."
    )

    # Minimum Age Requirement
    v2_age = evaluator.add_leaf(
        id="V2_Minimum_Age",
        desc="Provide the minimum age requirement for audience members.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum age requirement for audience members is {_safe(data.minimum_age_requirement) if data else ''}.",
        node=v2_age,
        sources=sources,
        additional_instruction="Verify minimum age (e.g., 16+, 18+) as stated on ticketing or official audience info pages."
    )

    # Studio Audience Capacity (approximate)
    v2_capacity = evaluator.add_leaf(
        id="V2_Studio_Audience_Capacity",
        desc="Provide the approximate audience seating capacity of the studio.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The studio's approximate audience seating capacity is about {_safe(data.audience_capacity_approx) if data else ''}.",
        node=v2_capacity,
        sources=sources,
        additional_instruction="Accept approximate numbers or ranges stated on credible sources."
    )


async def build_and_verify_concert_hall(
    evaluator: Evaluator,
    root_node,
    data: Optional[ConcertHallVenue]
) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_3_Large_Concert_Hall",
        desc="Historic concert hall located in Manhattan with capacity 5,000–7,000 and built/opened before 1950; provide required details and supporting URL(s).",
        parent=root_node,
        critical=False
    )

    sources = _sources_or_none(data.sources if data else None)
    evaluator.add_custom_node(
        result=bool(sources),
        id="V3_Supporting_Reference_URL",
        desc="Provide at least one reference URL that supports the concert hall information (capacity/location/year built/opened).",
        parent=venue_node,
        critical=True
    )

    # Venue Name
    v3_name = evaluator.add_leaf(
        id="V3_Venue_Name",
        desc="Provide the name of the concert hall.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The historic concert hall is named '{_safe(data.name) if data else ''}'.",
        node=v3_name,
        sources=sources,
        additional_instruction="Verify the hall's official or commonly used name on the cited page(s)."
    )

    # Seating Capacity 5,000–7,000
    v3_cap = evaluator.add_leaf(
        id="V3_Seating_Capacity_5000_to_7000",
        desc="Provide the exact seating capacity and verify it is between 5,000 and 7,000 seats.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert hall has a seating capacity of {_safe(data.seating_capacity) if data else ''}, which is between 5,000 and 7,000 seats.",
        node=v3_cap,
        sources=sources,
        additional_instruction="Confirm both the capacity value and that it lies within the 5,000–7,000 range."
    )

    # Manhattan Location
    v3_loc = evaluator.add_leaf(
        id="V3_Manhattan_Location",
        desc="Verify the venue is located in Manhattan, New York City.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="The concert hall is located in Manhattan, New York City.",
        node=v3_loc,
        sources=sources,
        additional_instruction="Verify that the venue's location is in Manhattan (not Brooklyn, Queens, Bronx, or Staten Island)."
    )

    # Year Built/Opened pre-1950
    v3_year = evaluator.add_leaf(
        id="V3_Year_Built_Or_Opened_Pre1950",
        desc="Provide the year the venue was built or opened and verify it is before 1950.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert hall was built/opened in {_safe(data.year_built_or_opened) if data else ''}, which is before 1950.",
        node=v3_year,
        sources=sources,
        additional_instruction="Verify that the listed year precedes 1950 on credible sources."
    )


async def build_and_verify_off_broadway(
    evaluator: Evaluator,
    root_node,
    data: Optional[OffBroadwayVenue]
) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_4_Off_Broadway_Theater",
        desc="Off-Broadway theater with capacity 100–499 and ADA-compliant accessible seating; provide required details and supporting URL(s).",
        parent=root_node,
        critical=False
    )

    sources = _sources_or_none(data.sources if data else None)
    evaluator.add_custom_node(
        result=bool(sources),
        id="V4_Supporting_Reference_URL",
        desc="Provide at least one reference URL that supports the Off-Broadway theater information (capacity/accessibility/production as applicable).",
        parent=venue_node,
        critical=True
    )

    # Theater Name
    v4_name = evaluator.add_leaf(
        id="V4_Theater_Name",
        desc="Provide the name of the Off-Broadway theater.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Off-Broadway theater is named '{_safe(data.name) if data else ''}'.",
        node=v4_name,
        sources=sources,
        additional_instruction="Verify the theater name on official or credible pages."
    )

    # Capacity 100–499
    v4_cap = evaluator.add_leaf(
        id="V4_Exact_Capacity_100_to_499",
        desc="Provide the exact seating capacity and verify it is between 100 and 499 seats.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater has an exact seating capacity of {_safe(data.seating_capacity) if data else ''}, which is between 100 and 499 seats (Off-Broadway classification).",
        node=v4_cap,
        sources=sources,
        additional_instruction="Confirm both the specific capacity and that it falls within 100–499 seats."
    )

    # ADA Compliant Accessible Seating
    v4_ada = evaluator.add_leaf(
        id="V4_ADA_Compliant_Accessible_Seating",
        desc="Verify the theater provides ADA-compliant accessible seating (e.g., wheelchair spaces and/or transfer/companion seats).",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="The theater provides ADA-compliant accessible seating (for example, wheelchair spaces and/or transfer/companion seating).",
        node=v4_ada,
        sources=sources,
        additional_instruction="Look for statements indicating wheelchair-accessible seating, transfer seats, companion seats, or ADA compliance."
    )

    # Current or Recent Production
    v4_prod = evaluator.add_leaf(
        id="V4_Current_Or_Recent_Production",
        desc="Provide the name of a current or recent production at this theater.",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A current or recent production at this theater is '{_safe(data.current_or_recent_production) if data else ''}'.",
        node=v4_prod,
        sources=sources,
        additional_instruction="Accept a production that is listed as current or recently presented at the theater on the cited page(s)."
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllVenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build and verify each venue subtree
    await build_and_verify_broadway(evaluator, root, extracted.broadway if extracted else None)
    await build_and_verify_tv_studio(evaluator, root, extracted.tv_studio if extracted else None)
    await build_and_verify_concert_hall(evaluator, root, extracted.concert_hall if extracted else None)
    await build_and_verify_off_broadway(evaluator, root, extracted.off_broadway if extracted else None)

    # Return evaluation summary
    return evaluator.get_summary()