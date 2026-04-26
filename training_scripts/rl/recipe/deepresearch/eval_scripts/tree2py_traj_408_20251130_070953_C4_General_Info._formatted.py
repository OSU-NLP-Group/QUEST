import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "super_bowl_lix_venue_guide"
TASK_DESCRIPTION = (
    "I am writing a comprehensive guide about Super Bowl LIX (2025) and need detailed information about the venue. "
    "Please provide the following information: (1) The official stadium name and city where Super Bowl LIX was held, "
    "(2) The complete street address of the stadium, (3) The exact date the game was played, "
    "(4) The stadium's standard football seating capacity, (5) The stadium's expandable seating capacity for major events, "
    "(6) The diameter of the stadium's dome structure, (7) The year the stadium originally opened, "
    "(8) How many Super Bowls have been held in this city and specifically at this stadium, "
    "and (9) When and where the next Super Bowl (LX) will be held. All information should be factual and verifiable from official sources."
)


# ------------------------- Extraction Models ------------------------- #
class VenueField(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NextSBInfo(BaseModel):
    date: Optional[str] = None
    stadium: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SBVenueExtraction(BaseModel):
    stadium_name: Optional[VenueField] = None
    city: Optional[VenueField] = None
    state: Optional[VenueField] = None
    address: Optional[VenueField] = None
    game_date: Optional[VenueField] = None
    standard_capacity: Optional[VenueField] = None
    expandable_capacity: Optional[VenueField] = None
    dome_diameter: Optional[VenueField] = None
    opening_year: Optional[VenueField] = None
    sb_count_city: Optional[VenueField] = None
    sb_count_stadium: Optional[VenueField] = None
    next_sb: Optional[NextSBInfo] = None


# ------------------------- Extraction Prompt ------------------------- #
def prompt_extract_sb_venue() -> str:
    return """
Extract the Super Bowl LIX (2025) venue details from the answer, returning a JSON that matches the following schema:

{
  "stadium_name": {"value": string|null, "sources": [urls...]},
  "city": {"value": string|null, "sources": [urls...]},
  "state": {"value": string|null, "sources": [urls...]},
  "address": {"value": string|null, "sources": [urls...]},
  "game_date": {"value": string|null, "sources": [urls...]},
  "standard_capacity": {"value": string|null, "sources": [urls...]},
  "expandable_capacity": {"value": string|null, "sources": [urls...]},
  "dome_diameter": {"value": string|null, "sources": [urls...]},
  "opening_year": {"value": string|null, "sources": [urls...]},
  "sb_count_city": {"value": string|null, "sources": [urls...]},
  "sb_count_stadium": {"value": string|null, "sources": [urls...]},
  "next_sb": {
    "date": string|null,
    "stadium": string|null,
    "city": string|null,
    "state": string|null,
    "sources": [urls...]
  }
}

Rules:
- Extract values exactly as stated in the answer (prefer strings for numbers/dates).
- For each field, extract ONLY URLs explicitly present in the answer as sources for that specific datum. Include markdown link URLs by extracting the actual link target.
- If a field is missing in the answer, set its "value" to null and its "sources" to an empty array.
- For next_sb, return null for any missing subfield and an empty array if no sources are provided.
- Do not invent or infer any URLs; only include those that appear in the answer text.
"""


# ------------------------- Helper Functions ------------------------- #
def _norm_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    result = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Basic validity check: has a netloc
        parsed = urlparse(u if "://" in u else f"http://{u}")
        if not parsed.netloc:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _has_value_and_sources(field: Optional[VenueField]) -> bool:
    return bool(field and field.value and field.value.strip() and len(_norm_sources(field.sources)) > 0)


def _domain(url: str) -> str:
    p = urlparse(url if "://" in url else f"http://{url}")
    return p.netloc.lower()


def compute_source_stats(extract: SBVenueExtraction) -> Dict[str, Any]:
    official_domains = {
        "nfl.com",
        "caesarsuperdome.com",
        "superdome.com",
        "mercedesbenzsuperdome.com",
        "neworleanssaints.com",
        "cityofneworleans.gov",
        "neworleans.com",
    }

    def is_official(d: str) -> bool:
        return d.endswith(".gov") or d in official_domains

    field_map = {
        "stadium_name": extract.stadium_name,
        "city": extract.city,
        "state": extract.state,
        "address": extract.address,
        "game_date": extract.game_date,
        "standard_capacity": extract.standard_capacity,
        "expandable_capacity": extract.expandable_capacity,
        "dome_diameter": extract.dome_diameter,
        "opening_year": extract.opening_year,
        "sb_count_city": extract.sb_count_city,
        "sb_count_stadium": extract.sb_count_stadium,
    }

    stats: Dict[str, Any] = {"fields": {}, "next_sb": {}}
    for k, v in field_map.items():
        srcs = _norm_sources(v.sources if v else [])
        doms = [_domain(u) for u in srcs]
        stats["fields"][k] = {
            "source_count": len(srcs),
            "domains": doms,
            "official_count": sum(1 for d in doms if is_official(d)),
        }

    next_srcs = _norm_sources(extract.next_sb.sources if extract.next_sb else [])
    next_doms = [_domain(u) for u in next_srcs]
    stats["next_sb"] = {
        "source_count": len(next_srcs),
        "domains": next_doms,
        "official_count": sum(1 for d in next_doms if is_official(d)),
    }
    return stats


# ------------------------- Verification Builders ------------------------- #
async def verify_value_supported(
    evaluator: Evaluator,
    parent: Any,
    group_id: str,
    group_desc: str,
    field: Optional[VenueField],
    claim: str,
    add_ins: str,
) -> None:
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=True,
    )

    existence_node = evaluator.add_custom_node(
        result=_has_value_and_sources(field),
        id=f"{group_id}_provided",
        desc=f"{group_desc} is provided with at least one source",
        parent=group_node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc=f"{group_desc} is supported by cited sources",
        parent=group_node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_norm_sources(field.sources if field else []),
        additional_instruction=add_ins,
    )


async def verify_city(
    evaluator: Evaluator,
    parent: Any,
    city_field: Optional[VenueField],
    state_field: Optional[VenueField],
) -> None:
    state_text = state_field.value.strip() if (state_field and state_field.value) else ""
    city_text = city_field.value.strip() if (city_field and city_field.value) else ""
    location_display = f"{city_text}, {state_text}" if state_text else city_text

    # Merge sources from city and state if provided
    merged_sources = _norm_sources(
        (city_field.sources if city_field else []) + (state_field.sources if state_field else [])
    )

    group_id = "Venue_City"
    group_desc = "Provides the city (and state, if provided) where Super Bowl LIX was held"
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=True,
    )

    existence_node = evaluator.add_custom_node(
        result=bool(city_text) and len(merged_sources) > 0,
        id=f"{group_id}_provided",
        desc="City (and optionally state) is provided with at least one source",
        parent=group_node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc="City/state location is supported by cited sources",
        parent=group_node,
        critical=True,
    )

    claim = f"Super Bowl LIX was held in {location_display}."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=merged_sources,
        additional_instruction="Verify the host city and state for Super Bowl LIX (2025). Allow minor formatting variations (e.g., abbreviations like 'LA' for Louisiana).",
    )


async def verify_next_super_bowl(
    evaluator: Evaluator,
    parent: Any,
    next_sb: Optional[NextSBInfo],
) -> None:
    group_id = "Next_Super_Bowl_LX_Date_And_Location"
    group_desc = "States when and where Super Bowl LX will be held"
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=True,
    )

    has_all_core = bool(
        next_sb
        and next_sb.date
        and next_sb.stadium
        and next_sb.city
        and len(_norm_sources(next_sb.sources)) > 0
    )

    existence_node = evaluator.add_custom_node(
        result=has_all_core,
        id=f"{group_id}_provided",
        desc="Next Super Bowl (LX) date and location are provided with at least one source",
        parent=group_node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc="Next Super Bowl LX date and location are supported by cited sources",
        parent=group_node,
        critical=True,
    )

    date_text = next_sb.date.strip() if (next_sb and next_sb.date) else ""
    stadium_text = next_sb.stadium.strip() if (next_sb and next_sb.stadium) else ""
    city_text = next_sb.city.strip() if (next_sb and next_sb.city) else ""
    state_text = next_sb.state.strip() if (next_sb and next_sb.state) else ""
    loc = f"{city_text}, {state_text}" if state_text else city_text

    claim = f"Super Bowl LX will be held on {date_text} at {stadium_text} in {loc}."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_norm_sources(next_sb.sources if next_sb else []),
        additional_instruction="Verify the officially announced date and venue/city of Super Bowl LX on authoritative sources such as NFL.com or the host committee/venue's official site.",
    )


# ------------------------- Main Evaluation ------------------------- #
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

    extract = await evaluator.extract(
        prompt=prompt_extract_sb_venue(),
        template_class=SBVenueExtraction,
        extraction_name="sb_lix_venue_extraction",
    )

    # Create critical guide root under actual root
    guide_node = evaluator.add_parallel(
        id="Super_Bowl_LIX_Venue_Guide",
        desc="Evaluates that the answer provides all requested Super Bowl LIX venue details and next Super Bowl info, with verifiable official sourcing.",
        parent=root,
        critical=True,
    )

    # 1. Official Stadium Name
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Venue_Official_Stadium_Name",
        group_desc="Provides the official stadium name where Super Bowl LIX was held",
        field=extract.stadium_name,
        claim=f"The official name of the stadium is '{(extract.stadium_name.value if extract.stadium_name and extract.stadium_name.value else '').strip()}'." if extract.stadium_name and extract.stadium_name.value else "The official name of the stadium is provided.",
        add_ins="Verify the current official stadium name from the venue's official site or NFL materials. Allow minor punctuation or branding variations.",
    )

    # 2. City (and state)
    await verify_city(
        evaluator=evaluator,
        parent=guide_node,
        city_field=extract.city,
        state_field=extract.state,
    )

    # 3. Complete street address
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Venue_Complete_Street_Address",
        group_desc="Provides the complete street address of the stadium",
        field=extract.address,
        claim=f"The stadium's complete street address is '{(extract.address.value if extract.address and extract.address.value else '').strip()}'." if extract.address and extract.address.value else "The stadium's complete street address is provided.",
        add_ins="Verify street, city, state, and ZIP/postal code. Allow minor formatting differences and abbreviations (e.g., 'St.' for Street).",
    )

    # 4. Game date
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Game_Date",
        group_desc="States the exact calendar date on which Super Bowl LIX was played",
        field=extract.game_date,
        claim=f"Super Bowl LIX was played on {(extract.game_date.value if extract.game_date and extract.game_date.value else '').strip()}." if extract.game_date and extract.game_date.value else "The game date is provided.",
        add_ins="Verify the exact calendar date. Allow reasonable date format variations (e.g., 'Feb 9, 2025' vs 'February 9, 2025').",
    )

    # 5. Standard football seating capacity
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Standard_Football_Seating_Capacity",
        group_desc="Gives the stadium's standard football seating capacity (in seats)",
        field=extract.standard_capacity,
        claim=f"The stadium's standard football seating capacity is {(extract.standard_capacity.value if extract.standard_capacity and extract.standard_capacity.value else '').strip()}." if extract.standard_capacity and extract.standard_capacity.value else "The standard football seating capacity is provided.",
        add_ins="Verify the typical capacity for football configuration. Accept minor numeric variations due to renovations or configuration notes.",
    )

    # 6. Expandable seating capacity for major events
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Expandable_Seating_Capacity",
        group_desc="Gives the stadium's expandable seating capacity for major events (in seats), distinguishing it from standard capacity",
        field=extract.expandable_capacity,
        claim=f"The stadium's expandable seating capacity for major events is {(extract.expandable_capacity.value if extract.expandable_capacity and extract.expandable_capacity.value else '').strip()}." if extract.expandable_capacity and extract.expandable_capacity.value else "The expandable seating capacity is provided.",
        add_ins="Verify expandable capacity (e.g., with temporary seating). Distinguish it from standard capacity. Allow slight numeric ranges.",
    )

    # 7. Dome diameter
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Dome_Diameter",
        group_desc="States the diameter of the stadium's dome structure, including units",
        field=extract.dome_diameter,
        claim=f"The diameter of the stadium's dome is {(extract.dome_diameter.value if extract.dome_diameter and extract.dome_diameter.value else '').strip()}." if extract.dome_diameter and extract.dome_diameter.value else "The dome diameter is provided.",
        add_ins="Verify the dome diameter with units (e.g., feet or meters). Allow minor rounding or unit conversions.",
    )

    # 8. Stadium opening year
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Stadium_Opening_Year",
        group_desc="States the year the stadium originally opened",
        field=extract.opening_year,
        claim=f"The stadium originally opened in {(extract.opening_year.value if extract.opening_year and extract.opening_year.value else '').strip()}." if extract.opening_year and extract.opening_year.value else "The stadium opening year is provided.",
        add_ins="Verify the original opening year (first opening to the public). Ignore later re-openings post-renovation.",
    )

    # 9. Super Bowl count in city
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Super_Bowl_Count_In_City",
        group_desc="States how many Super Bowls have been held in the host city (as of Super Bowl LIX)",
        field=extract.sb_count_city,
        claim=f"As of Super Bowl LIX, the host city has hosted {(extract.sb_count_city.value if extract.sb_count_city and extract.sb_count_city.value else '').strip()} Super Bowls." if extract.sb_count_city and extract.sb_count_city.value else "The count of Super Bowls in the host city is provided.",
        add_ins="Verify the total number of Super Bowls hosted by the city, including those at any stadiums within the city limits.",
    )

    # 10. Super Bowl count at stadium
    await verify_value_supported(
        evaluator=evaluator,
        parent=guide_node,
        group_id="Super_Bowl_Count_At_Stadium",
        group_desc="States how many Super Bowls have been held at the specific stadium (as of Super Bowl LIX)",
        field=extract.sb_count_stadium,
        claim=f"As of Super Bowl LIX, the specific stadium has hosted {(extract.sb_count_stadium.value if extract.sb_count_stadium and extract.sb_count_stadium.value else '').strip()} Super Bowls." if extract.sb_count_stadium and extract.sb_count_stadium.value else "The count of Super Bowls at the stadium is provided.",
        add_ins="Verify the total number of Super Bowls hosted at this stadium specifically. Exclude other stadiums in the city.",
    )

    # 11. Next Super Bowl (LX) date and location
    await verify_next_super_bowl(
        evaluator=evaluator,
        parent=guide_node,
        next_sb=extract.next_sb,
    )

    # 12. Official Source Verifiability (existence of sources for all requested data)
    #    We implement this as a custom node verifying that each required datum has at least one source.
    required_fields = [
        extract.stadium_name,
        extract.city,
        extract.address,
        extract.game_date,
        extract.standard_capacity,
        extract.expandable_capacity,
        extract.dome_diameter,
        extract.opening_year,
        extract.sb_count_city,
        extract.sb_count_stadium,
    ]
    per_field_sources_present = all(_has_value_and_sources(f) for f in required_fields)

    # Next SB requires sources too
    next_has_sources = bool(extract.next_sb and extract.next_sb.date and extract.next_sb.stadium and extract.next_sb.city and len(_norm_sources(extract.next_sb.sources)) > 0)

    evaluator.add_custom_node(
        result=(per_field_sources_present and next_has_sources),
        id="Official_Source_Verifiability",
        desc="Includes citations/links to sources sufficient to verify each requested datum",
        parent=guide_node,
        critical=True,
    )

    # Add helpful source stats in the summary
    evaluator.add_custom_info(
        info=compute_source_stats(extract),
        info_type="source_stats",
        info_name="official_source_statistics",
    )

    return evaluator.get_summary()