import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "broadway_touring_venues_usa"
TASK_DESCRIPTION = (
    "A theater industry researcher is compiling a database of major touring venues in the United States. "
    "Identify 4 theater venues located in 4 different U.S. states that regularly host touring Broadway productions, "
    "where each venue has a seating capacity of at least 2,000 seats. For each venue, provide the following information: "
    "(1) the official venue name, (2) the city and state location, (3) the seating capacity, (4) the year it originally opened, "
    "and (5) a reference URL documenting this information."
)

class Venue(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    opening_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    broadway_evidence_urls: List[str] = Field(default_factory=list)
    renovations_evidence_urls: List[str] = Field(default_factory=list)

class VenuesExtraction(BaseModel):
    venues: List[Venue] = Field(default_factory=list)

def prompt_extract_venues() -> str:
    return (
        "Extract up to 4 theater venues listed in the answer that are said to host touring Broadway productions. "
        "For each venue, return a JSON object with the following fields:\n"
        "1) official_name: the official or commonly accepted venue name as stated in the answer\n"
        "2) city: the city name\n"
        "3) state: the U.S. state name or its 2-letter abbreviation (e.g., CA, NY)\n"
        "4) seating_capacity: the seating capacity exactly as written in the answer (string). Do not normalize; keep formatting such as commas or '+'\n"
        "5) opening_year: the original opening year (string; include only the year if possible)\n"
        "6) reference_urls: a list of URLs explicitly cited in the answer that document the venue details (name/location/capacity/opening year)\n"
        "7) broadway_evidence_urls: a list of URLs explicitly cited in the answer that show the venue hosts touring Broadway productions "
        "(e.g., schedule pages, Broadway Across America pages, major press or venue season announcements)\n"
        "8) renovations_evidence_urls: a list of URLs explicitly cited in the answer that document major renovations or significant reopenings "
        "if applicable (e.g., for venues originally opened before 1950). If not mentioned, return an empty list.\n\n"
        "Return a JSON object with a single key 'venues' that is an array of these venue objects. "
        "If any field is not present in the answer for a given venue, return null for that field or an empty list as appropriate. "
        "Extract only URLs explicitly present in the answer text."
    )

US_STATE_ABBR_TO_FULL = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}
US_STATE_FULL_UPPER = {name.upper(): name for name in US_STATE_ABBR_TO_FULL.values()}

def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper().replace(".", "")
    if s in US_STATE_ABBR_TO_FULL:
        return US_STATE_ABBR_TO_FULL[s]
    if s in US_STATE_FULL_UPPER:
        return US_STATE_FULL_UPPER[s]
    return state.strip()

def parse_year(year_str: Optional[str]) -> Optional[int]:
    if not year_str:
        return None
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", year_str)
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None

def collect_all_urls(v: Venue) -> List[str]:
    urls = []
    urls.extend(v.reference_urls or [])
    urls.extend(v.broadway_evidence_urls or [])
    urls.extend(v.renovations_evidence_urls or [])
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

async def verify_venue(evaluator: Evaluator, parent_node, venue: Venue, index: int) -> None:
    venue_node = evaluator.add_parallel(
        id=f"Venue_{index+1}",
        desc=f"Venue #{index+1} (one of four) meets all per-venue requirements",
        parent=parent_node,
        critical=False
    )

    all_urls = collect_all_urls(venue)
    ref_urls = venue.reference_urls if venue.reference_urls else []

    name_leaf = evaluator.add_leaf(
        id=f"Venue_{index+1}_Official_Name",
        desc="Official, documented venue name is provided",
        parent=venue_node,
        critical=True
    )
    name_claim = f"The official name of the venue is '{venue.official_name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=ref_urls if ref_urls else all_urls,
        additional_instruction=(
            "Verify that the referenced page(s) explicitly show the venue name as stated. "
            "Allow minor variations like punctuation, 'Theatre' vs 'Theater', or corporate prefixes. "
            "If no URL is provided, judge based on the answer text."
        )
    )

    location_leaf = evaluator.add_leaf(
        id=f"Venue_{index+1}_Location",
        desc="City and U.S. state are provided for the venue",
        parent=venue_node,
        critical=True
    )
    loc_city = venue.city or ""
    loc_state = venue.state or ""
    location_claim = f"The venue is located in {loc_city}, {loc_state}."
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=ref_urls if ref_urls else all_urls,
        additional_instruction=(
            "Verify that the page(s) show the venue's location (city and U.S. state). "
            "Accept state abbreviations (e.g., CA) and allow minor formatting differences."
        )
    )

    capacity_leaf = evaluator.add_leaf(
        id=f"Venue_{index+1}_Capacity",
        desc="Seating capacity is provided and is at least 2,000 seats",
        parent=venue_node,
        critical=True
    )
    capacity_str = venue.seating_capacity or ""
    capacity_claim = (
        f"The venue has a seating capacity of {capacity_str} and it is at least 2,000 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=ref_urls if ref_urls else all_urls,
        additional_instruction=(
            "Check the page(s) for the stated seating capacity and confirm that the capacity is ≥ 2,000. "
            "Allow reasonable rounding or approximate statements like 'about 2,100'."
        )
    )

    opening_leaf = evaluator.add_leaf(
        id=f"Venue_{index+1}_Opening_Year",
        desc="Original opening year is provided",
        parent=venue_node,
        critical=True
    )
    year_str = venue.opening_year or ""
    opening_claim = (
        f"The venue originally opened in {year_str}."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_leaf,
        sources=ref_urls if ref_urls else all_urls,
        additional_instruction=(
            "Verify the original opening year (first opening), not later reopenings. "
            "If multiple dates appear, prefer the earliest 'opened' or 'dedicated' year described for the venue."
        )
    )

    pre1950_year = parse_year(venue.opening_year)
    if pre1950_year is not None and pre1950_year < 1950:
        pre1950_leaf = evaluator.add_leaf(
            id=f"Venue_{index+1}_Pre1950_Renovations_If_Applicable",
            desc="If the venue originally opened before 1950, major renovations or reopenings are documented (otherwise not required)",
            parent=venue_node,
            critical=True
        )
        pre1950_claim = (
            "This venue, originally opened before 1950, has documented major renovations or a significant reopening."
        )
        reno_sources = venue.renovations_evidence_urls if venue.renovations_evidence_urls else (ref_urls if ref_urls else all_urls)
        await evaluator.verify(
            claim=pre1950_claim,
            node=pre1950_leaf,
            sources=reno_sources,
            additional_instruction=(
                "Look for explicit indications of major renovation, restoration, expansion, or a significant reopening "
                "after the original pre-1950 opening. Accept credible sources including the venue's official site, "
                "recognized cultural institutions, reputable news outlets, or Wikipedia with citations."
            )
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"Venue_{index+1}_Pre1950_Renovations_If_Applicable",
            desc="If the venue originally opened before 1950, major renovations or reopenings are documented (otherwise not required)",
            parent=venue_node,
            critical=True
        )

    operational_leaf = evaluator.add_leaf(
        id=f"Venue_{index+1}_Operational_And_Broadway",
        desc="Evidence the venue is currently operational and actively hosts touring Broadway productions",
        parent=venue_node,
        critical=True
    )
    broadway_sources = venue.broadway_evidence_urls if venue.broadway_evidence_urls else (ref_urls if ref_urls else all_urls)
    operational_claim = (
        "The venue is currently operational and regularly hosts touring Broadway productions."
    )
    await evaluator.verify(
        claim=operational_claim,
        node=operational_leaf,
        sources=broadway_sources,
        additional_instruction=(
            "Confirm that the venue is active (operational) and hosts touring Broadway shows. "
            "Accept season schedules, 'Broadway Across America' or equivalent networks, booking pages, "
            "and press releases listing national Broadway tours. Words like 'Broadway season', 'national tour', "
            "'touring musical' should count."
        )
    )

    ref_leaf = evaluator.add_custom_node(
        result=bool(venue.reference_urls),
        id=f"Venue_{index+1}_Reference_URL",
        desc="At least one reference URL from an official or otherwise reliable source is provided for the venue",
        parent=venue_node,
        critical=True
    )

async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(Venue())

    venue_nodes_parent = root
    for i in range(4):
        await verify_venue(evaluator, venue_nodes_parent, venues[i], i)

    states = [normalize_state(v.state) for v in venues]
    diversity_ok = (all(s is not None and s.strip() != "" for s in states) and len(set(states)) == 4)

    evaluator.add_custom_node(
        result=diversity_ok,
        id="Geographic_Diversity",
        desc="All 4 venues are located in 4 different U.S. states (no shared state among the four)",
        parent=root,
        critical=True
    )

    evaluator.add_custom_info(
        info={"canonical_states": states, "unique_state_count": len(set(states))},
        info_type="custom",
        info_name="geographic_diversity_check"
    )

    return evaluator.get_summary()