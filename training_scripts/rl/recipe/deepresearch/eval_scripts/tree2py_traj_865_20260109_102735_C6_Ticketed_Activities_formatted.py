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
TASK_ID = "nyc_indoor_arena_constraints"
TASK_DESCRIPTION = """Identify the major indoor multi-purpose arena located in New York City that meets all of the following criteria:

1. Opening Timeline: The venue must have opened in the 1960s.

2. Capacity Specifications: The venue must have the following capacity ranges for different event types:
   - Concert capacity: between 20,000 and 23,000 attendees
   - Basketball capacity: between 18,000 and 21,000 attendees
   - Ice hockey capacity: between 17,000 and 19,000 attendees

3. Renovation History: The venue must have undergone two major renovations:
   - One renovation between 1989 and 1992 that cost more than $100 million
   - One renovation between 2010 and 2015 that cost more than $500 million and was completed in multiple phases

4. Operational Requirements:
   - Must currently serve as the home venue for both an NBA team and an NHL team
   - Must host at least 250 events per year

5. Accessibility: Must provide wheelchair-accessible seating compliant with ADA requirements.

Provide the venue name and document each of the above requirements with specific details and supporting URLs.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RenovationInfo(BaseModel):
    time_window: Optional[str] = None
    cost: Optional[str] = None
    phases: Optional[str] = None  # e.g., "three phases", "multi-phase", etc.


class Capacities(BaseModel):
    concert: Optional[str] = None
    basketball: Optional[str] = None
    ice_hockey: Optional[str] = None


class VenueExtraction(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # e.g., "Manhattan, New York City"
    venue_type: Optional[str] = None  # e.g., "indoor multi-purpose arena"
    opening_year: Optional[str] = None
    capacities: Optional[Capacities] = None
    renovation_1989_1992: Optional[RenovationInfo] = None
    renovation_2010_2015: Optional[RenovationInfo] = None
    nba_team: Optional[str] = None
    nhl_team: Optional[str] = None
    events_per_year: Optional[str] = None
    ada_wheelchair: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    You will extract structured information from the answer about a specific New York City indoor multi-purpose arena and the URLs cited.

    Extract the following fields as strings exactly as they appear in the answer (do not invent values):
    - name: the venue name (e.g., "Madison Square Garden")
    - location: the location or city reference (e.g., "Manhattan, New York City" or "New York, NY")
    - venue_type: description of facility type (e.g., "indoor multi-purpose arena")
    - opening_year: the opening year as written (e.g., "1968", or a phrase like "opened in 1968")
    - capacities: an object containing:
        - concert: concert capacity details if provided
        - basketball: basketball capacity details if provided
        - ice_hockey: ice hockey capacity details if provided
    - renovation_1989_1992: an object containing any mentioned details about the renovation between 1989 and 1992:
        - time_window: the time window description if provided
        - cost: the cost description if provided (include currency symbol if present)
        - phases: description if phases are mentioned (can be null)
    - renovation_2010_2015: an object containing any mentioned details about the renovation between 2010 and 2015:
        - time_window
        - cost
        - phases (e.g., "multi-phase", "three phases", "completed in phases")
    - nba_team: the NBA team named as the current home team for this venue, if provided
    - nhl_team: the NHL team named as the current home team for this venue, if provided
    - events_per_year: any mention of annual event counts (e.g., "hosts 320 events per year")
    - ada_wheelchair: any mention of ADA-compliant or wheelchair-accessible seating
    - urls: extract all URLs explicitly listed in the answer text that support the information (include full URLs; do not invent)

    Notes:
    - Return null for fields not stated in the answer.
    - For URLs, extract only valid URLs explicitly present in the answer. Include URLs found in markdown links.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if name else "the venue"


def _all_urls(extracted: Optional[VenueExtraction]) -> List[str]:
    if not extracted or not extracted.urls:
        return []
    # Deduplicate casually by preserving order
    seen = set()
    out = []
    for u in extracted.urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: VenueExtraction) -> None:
    root = evaluator.root
    if not root:
        raise RuntimeError("Evaluator root not initialized")

    venue_name = _safe_name(extracted.name)
    urls = _all_urls(extracted)

    # 0) Supporting URLs gate (critical sibling at root; evaluate first)
    supporting_urls_node = evaluator.add_custom_node(
        result=len(urls) >= 1,
        id="supporting_urls",
        desc="Supporting URLs are provided that substantiate the venue identification and each required constraint",
        parent=root,
        critical=True
    )

    # 1) Venue name provided (critical sibling at root; evaluate early)
    venue_name_node = evaluator.add_custom_node(
        result=bool(extracted.name and extracted.name.strip()),
        id="venue_name",
        desc="Venue name is provided",
        parent=root,
        critical=True
    )

    # 2) Location and facility type
    location_type_node = evaluator.add_parallel(
        id="location_and_type",
        desc="Venue meets the location and facility-type constraints",
        parent=root,
        critical=True
    )

    nyc_node = evaluator.add_leaf(
        id="nyc_location",
        desc="Venue is located in New York City",
        parent=location_type_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} is located in New York City (NYC). Mentions of Manhattan, Midtown Manhattan, or New York, NY count as NYC.",
        node=nyc_node,
        sources=urls,
        additional_instruction="Confirm the venue is in New York City. Accept references such as 'Manhattan', 'Midtown Manhattan', or 'New York, NY' as NYC."
    )

    type_node = evaluator.add_leaf(
        id="indoor_multi_purpose_arena",
        desc="Venue is an indoor multi-purpose arena (not an outdoor amphitheater)",
        parent=location_type_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} is an indoor multi-purpose arena (i.e., indoor venue designed for multiple event types).",
        node=type_node,
        sources=urls,
        additional_instruction="Look for descriptors like 'indoor arena', 'multi-purpose arena', or similar language. Reject outdoor-only amphitheater descriptions."
    )

    # 3) Opening timeline
    opening_timeline_node = evaluator.add_parallel(
        id="opening_timeline",
        desc="Venue opening timeline meets the constraint",
        parent=root,
        critical=True
    )
    opened_1960s_node = evaluator.add_leaf(
        id="opened_in_1960s",
        desc="Venue opened in the 1960s",
        parent=opening_timeline_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} opened in the 1960s (between 1960 and 1969 inclusive).",
        node=opened_1960s_node,
        sources=urls,
        additional_instruction="Check the opening year on authoritative sources; 1960–1969 counts as '1960s'."
    )

    # 4) Capacity specifications
    capacity_node = evaluator.add_parallel(
        id="capacity_specifications",
        desc="Venue capacities meet all required ranges",
        parent=root,
        critical=True
    )

    concert_cap_node = evaluator.add_leaf(
        id="concert_capacity",
        desc="Concert capacity is within 20,000–23,000",
        parent=capacity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For {venue_name}, the concert capacity is between 20,000 and 23,000 attendees.",
        node=concert_cap_node,
        sources=urls,
        additional_instruction="Confirm concert capacity from seating/venue data. If multiple configurations exist, use the typical maximum concert capacity; allow minor rounding differences."
    )

    basketball_cap_node = evaluator.add_leaf(
        id="basketball_capacity",
        desc="Basketball capacity is within 18,000–21,000",
        parent=capacity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For {venue_name}, the basketball seating capacity is between 18,000 and 21,000 attendees.",
        node=basketball_cap_node,
        sources=urls,
        additional_instruction="Check basketball capacity for NBA configuration; allow small variance or rounding."
    )

    hockey_cap_node = evaluator.add_leaf(
        id="ice_hockey_capacity",
        desc="Ice hockey capacity is within 17,000–19,000",
        parent=capacity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For {venue_name}, the ice hockey seating capacity is between 17,000 and 19,000 attendees.",
        node=hockey_cap_node,
        sources=urls,
        additional_instruction="Check NHL/ice hockey capacity; allow small variance or rounding."
    )

    # 5) Renovation history
    renovation_node = evaluator.add_parallel(
        id="renovation_history",
        desc="Venue renovation history meets both renovation constraints",
        parent=root,
        critical=True
    )

    # 5.1) 1989–1992 renovation
    ren_8992_node = evaluator.add_parallel(
        id="renovation_1989_1992",
        desc="A major renovation occurred between 1989 and 1992 and cost more than $100 million",
        parent=renovation_node,
        critical=True
    )
    ren_8992_time = evaluator.add_leaf(
        id="renovation_1989_1992_time_window",
        desc="Renovation occurred between 1989 and 1992",
        parent=ren_8992_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} underwent a major renovation between 1989 and 1992.",
        node=ren_8992_time,
        sources=urls,
        additional_instruction="Look for language like 'renovation from 1989–1991' or '1991–1992'; any timeframe with overlap in 1989–1992 is acceptable if clearly a renovation."
    )
    ren_8992_cost = evaluator.add_leaf(
        id="renovation_1989_1992_cost_threshold",
        desc="Renovation cost was more than $100 million",
        parent=ren_8992_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The renovation of {venue_name} that took place between 1989 and 1992 cost more than $100 million.",
        node=ren_8992_cost,
        sources=urls,
        additional_instruction="Verify the cited cost; accept statements like ~$200M or >$100M; reject if only minor upgrades are mentioned."
    )

    # 5.2) 2010–2015 renovation (multi-phase and >$500M)
    ren_1015_node = evaluator.add_parallel(
        id="renovation_2010_2015",
        desc="A major renovation occurred between 2010 and 2015, cost more than $500 million, and was completed in multiple phases",
        parent=renovation_node,
        critical=True
    )
    ren_1015_time = evaluator.add_leaf(
        id="renovation_2010_2015_time_window",
        desc="Renovation occurred between 2010 and 2015",
        parent=ren_1015_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} underwent a major renovation between 2010 and 2015.",
        node=ren_1015_time,
        sources=urls,
        additional_instruction="Look for renovation timelines such as 2011–2013, 2010–2014, etc."
    )

    ren_1015_cost = evaluator.add_leaf(
        id="renovation_2010_2015_cost_threshold",
        desc="Renovation cost was more than $500 million",
        parent=ren_1015_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The renovation of {venue_name} during 2010–2015 cost more than $500 million.",
        node=ren_1015_cost,
        sources=urls,
        additional_instruction="Accept ~$1 billion or >$500M figures when clearly tied to this renovation project."
    )

    ren_1015_phases = evaluator.add_leaf(
        id="renovation_2010_2015_multiple_phases",
        desc="Renovation was completed in multiple phases",
        parent=ren_1015_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The renovation of {venue_name} during 2010–2015 was completed in multiple phases.",
        node=ren_1015_phases,
        sources=urls,
        additional_instruction="Look for explicit 'multi-phase' language or breakdowns into phases (e.g., Phase I, Phase II, Phase III)."
    )

    # 6) Operational requirements
    ops_node = evaluator.add_parallel(
        id="operational_requirements",
        desc="Venue meets current operational constraints",
        parent=root,
        critical=True
    )

    nba_node = evaluator.add_leaf(
        id="nba_home_venue",
        desc="Venue currently serves as the home venue for an NBA team",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} currently serves as the home venue for an NBA team.",
        node=nba_node,
        sources=urls,
        additional_instruction="Look for mentions like 'home of the New York Knicks (NBA)' or equivalent current affiliation."
    )

    nhl_node = evaluator.add_leaf(
        id="nhl_home_venue",
        desc="Venue currently serves as the home venue for an NHL team",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} currently serves as the home venue for an NHL team.",
        node=nhl_node,
        sources=urls,
        additional_instruction="Look for mentions like 'home of the New York Rangers (NHL)' or equivalent current affiliation."
    )

    events_node = evaluator.add_leaf(
        id="events_per_year",
        desc="Venue hosts at least 250 events per year",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} hosts at least 250 events per year.",
        node=events_node,
        sources=urls,
        additional_instruction="Seek phrasing such as 'hosts more than 250 events annually' or '300+ events each year'."
    )

    # 7) Accessibility
    access_node = evaluator.add_parallel(
        id="accessibility",
        desc="Venue meets accessibility constraint",
        parent=root,
        critical=True
    )
    ada_node = evaluator.add_leaf(
        id="ada_wheelchair_seating",
        desc="Venue provides wheelchair-accessible seating compliant with ADA requirements",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue_name} provides wheelchair-accessible seating compliant with ADA requirements.",
        node=ada_node,
        sources=urls,
        additional_instruction="Look for explicit ADA references, wheelchair-accessible seating, companion seating, and accessibility policies on official or authoritative sources."
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_venue_name": extracted.name,
            "total_urls_extracted": len(urls),
            "sample_urls": urls[:5]
        },
        info_type="debug",
        info_name="extraction_summary"
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point to evaluate an agent's answer for the NYC indoor multi-purpose arena constraints task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric
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

    # 1) Extract structured information from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # 2) Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()