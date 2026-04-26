import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nfl_stadiums_two_city_tour"
TASK_DESCRIPTION = """
A major entertainment production company is planning a two-city North American stadium tour and needs to identify suitable NFL venues that can accommodate large audiences with weather protection. The production requires finding two venues that meet the following specifications:

Regional Requirements:
- One venue must be located in the Eastern United States (in a state east of the Mississippi River)
- One venue must be located in the Western United States (in a state west of the Mississippi River, including Texas)

Technical Requirements (both venues must meet all of these):
- Seating capacity of at least 70,000
- Weather protection via either a retractable roof or a fixed dome (open-air stadiums are not acceptable)
- At least 700 wheelchair accessible seats to ensure ADA compliance
- Opened before January 1, 2020 (to ensure established operational infrastructure)

For each identified venue, provide:
1. The official name of the stadium
2. The city and state where it is located
3. The seating capacity
4. The roof type (retractable roof or fixed dome)
5. The year it opened
6. The NFL team(s) that call it home
7. Confirmation that it meets the wheelchair accessible seating requirement

Provide supporting reference URLs from official stadium websites or reliable sources for each venue's specifications.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accept full state name or 2-letter code
    capacity: Optional[str] = None  # Keep as text to preserve how the answer stated it
    roof_type: Optional[str] = None  # e.g., "retractable roof", "fixed dome", "open-air"
    opening_year: Optional[str] = None  # e.g., "2009"
    nfl_teams: List[str] = Field(default_factory=list)  # e.g., ["Dallas Cowboys"]
    ada_wheelchair_seats: Optional[str] = None  # e.g., "1200", "1,200", "at least 700"
    reference_urls: List[str] = Field(default_factory=list)  # Supporting sources for this venue


class VenueListExtraction(BaseModel):
    venues: List[VenueInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all NFL stadium venues mentioned in the answer. For each venue, return the following fields:

    - name: Official stadium name as stated in the answer.
    - city: City where the stadium is located.
    - state: State where the stadium is located (full name or 2-letter postal abbreviation).
    - capacity: Seating capacity as stated (keep the exact text, e.g., "71,000", "70,240", "70k+", "approximately 72,000", etc.).
    - roof_type: Extract the roof type exactly as described. If the answer indicates a fully enclosed, non-open-air stadium (e.g., "fixed roof", "fixed dome", "domed stadium", "enclosed stadium"), keep that text. If it indicates "retractable roof", keep that text. If it explicitly says "open-air", set to "open-air".
    - opening_year: The first year the stadium officially opened to the public (not renovation years). Prefer a 4-digit year if present.
    - nfl_teams: A list of NFL team names that call the stadium home, as stated in the answer.
    - ada_wheelchair_seats: If the answer states any number or claim about wheelchair accessible seating (e.g., "at least 700", "1,200 wheelchair seats"), extract that text. If absent, set to null.
    - reference_urls: All URLs cited for this venue's specifications. Include only actual URLs explicitly present in the answer (plain URLs or in markdown links).

    Return a JSON object with a single key "venues" containing an array of venue objects.
    If any field for a given venue is not present in the answer, set it to null or an empty list accordingly.
    Extract up to the first 5 venues mentioned if the answer lists more.
    """


# --------------------------------------------------------------------------- #
# Helper: State normalization and region classification                       #
# --------------------------------------------------------------------------- #
STATE_FULL_TO_ABBR: Dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington, dc": "DC", "washington d c": "DC",
    "d.c.": "DC", "dc": "DC"
}
# Allow direct abbreviations mapping to themselves (for convenience)
for abbr in list(set(STATE_FULL_TO_ABBR.values())):
    STATE_FULL_TO_ABBR[abbr.lower()] = abbr

EAST_OF_MISSISSIPPI: set = {
    "AL", "CT", "DE", "FL", "GA", "IL", "IN", "KY", "ME", "MD",
    "MA", "MI", "MS", "NH", "NJ", "NY", "NC", "OH", "PA", "RI",
    "SC", "TN", "VT", "VA", "WV", "WI", "DC"
}
WEST_OF_MISSISSIPPI: set = {
    "AR", "IA", "LA", "MN", "MO", "ND", "SD", "NE", "KS", "OK",
    "TX", "NM", "CO", "WY", "MT", "ID", "WA", "OR", "CA", "NV",
    "AZ", "UT", "AK", "HI"
}


def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_clean = re.sub(r"[^\w\s]", "", s).strip().lower()
    # Try direct abbr
    if len(s_clean) == 2 and s_clean.isalpha():
        return s_clean.upper()
    # Try full name mapping
    return STATE_FULL_TO_ABBR.get(s_clean, None)


def classify_state_region(state_str: Optional[str]) -> str:
    abbr = _normalize_state(state_str)
    if not abbr:
        return "unknown"
    if abbr in EAST_OF_MISSISSIPPI:
        return "east"
    if abbr in WEST_OF_MISSISSIPPI:
        return "west"
    return "unknown"


def pick_regional_candidates(venues: List[VenueInfo]) -> Tuple[Optional[VenueInfo], Optional[VenueInfo], Dict[str, str]]:
    classification: Dict[str, str] = {}
    east_candidate: Optional[VenueInfo] = None
    west_candidate: Optional[VenueInfo] = None

    for v in venues:
        region = classify_state_region(v.state)
        classification[v.name or f"{v.city}, {v.state}"] = region
        if region == "east" and not east_candidate:
            east_candidate = v
        if region == "west" and not west_candidate:
            west_candidate = v
        if east_candidate and west_candidate:
            break

    # Fallbacks if one side missing: keep None for missing side to let checks fail appropriately
    return east_candidate, west_candidate, classification


def fmt_team_list(teams: List[str]) -> str:
    teams = [t for t in teams if t and t.strip()]
    if not teams:
        return ""
    if len(teams) == 1:
        return teams[0]
    return ", ".join(teams[:-1]) + f", and {teams[-1]}"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: Optional[VenueInfo],
    role_prefix: str  # "Eastern" or "Western"
) -> None:
    """
    Build the subtree for one venue and run verifications based on the rubric.
    """
    # Parent node for this venue
    region_node = evaluator.add_parallel(
        id=f"{role_prefix}_US_Venue",
        desc=f"An NFL stadium located in the {role_prefix} United States that meets all technical and operational requirements",
        parent=parent_node,
        critical=False
    )

    # Convenience values
    name = venue.name.strip() if (venue and venue.name) else ""
    city = venue.city.strip() if (venue and venue.city) else ""
    state = venue.state.strip() if (venue and venue.state) else ""
    capacity_text = venue.capacity.strip() if (venue and venue.capacity) else ""
    roof_text = venue.roof_type.strip().lower() if (venue and venue.roof_type) else ""
    opening_year = venue.opening_year.strip() if (venue and venue.opening_year) else ""
    team_list = venue.nfl_teams if (venue and venue.nfl_teams) else []
    ada_text = venue.ada_wheelchair_seats.strip() if (venue and venue.ada_wheelchair_seats) else ""
    sources = venue.reference_urls if (venue and venue.reference_urls) else []

    # 1) Name provided (critical existence)
    evaluator.add_custom_node(
        result=bool(name),
        id=f"{role_prefix}_Venue_Name",
        desc=f"The name of the identified {role_prefix} US venue is provided",
        parent=region_node,
        critical=True
    )

    # 2) City provided (critical existence)
    evaluator.add_custom_node(
        result=bool(city),
        id=f"{role_prefix}_City_Provided",
        desc=f"The city where the venue is located is provided",
        parent=region_node,
        critical=True
    )

    # 3) Reference URLs provided (critical existence)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id=f"{role_prefix}_Reference_URLs",
        desc=f"Supporting reference URLs from official stadium websites or reliable sources are provided for the {role_prefix} venue's specifications",
        parent=region_node,
        critical=True
    )

    # 4) State location region check (critical)
    # Use code-based classification (non-web factual)
    region = classify_state_region(state)
    is_correct_region = (region == "east") if role_prefix == "Eastern" else (region == "west")
    evaluator.add_custom_node(
        result=is_correct_region,
        id=f"{role_prefix}_State_Location",
        desc=(
            f"The venue is located in a state "
            f"{'east' if role_prefix == 'Eastern' else 'west'} of the Mississippi River"
            f"{' (including Texas)' if role_prefix == 'Western' else ''}"
        ),
        parent=region_node,
        critical=True
    )

    # 5) Capacity requirement >= 70,000 (critical, verify via cited sources)
    cap_node = evaluator.add_leaf(
        id=f"{role_prefix}_Capacity_Requirement",
        desc="The venue has a seating capacity of at least 70,000",
        parent=region_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of {name or 'this stadium'} is at least 70,000."
        + (f" The answer states the capacity as '{capacity_text}'." if capacity_text else "")
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=sources,
        additional_instruction="Confirm from the provided webpage(s) that the standard seating capacity (not 'expandable to') is >= 70,000."
    )

    # 6) Roof type requirement: retractable roof or fixed dome (critical, verify via cited sources)
    roof_node = evaluator.add_leaf(
        id=f"{role_prefix}_Roof_Type",
        desc="The venue has either a retractable roof or a fixed dome (not open-air)",
        parent=region_node,
        critical=True
    )
    # Tailor claim slightly using extracted text if present
    if "retractable" in roof_text:
        roof_claim = f"{name or 'This stadium'} has a retractable roof (i.e., it is not an open-air stadium)."
    elif any(k in roof_text for k in ["fixed", "dome", "domed", "enclosed", "indoor", "roofed"]):
        roof_claim = f"{name or 'This stadium'} is fully enclosed with a fixed roof/dome (i.e., it is not an open-air stadium)."
    else:
        roof_claim = f"{name or 'This stadium'} features either a retractable roof or a fixed/enclosed roof (not open-air)."
    await evaluator.verify(
        claim=roof_claim,
        node=roof_node,
        sources=sources,
        additional_instruction=(
            "Accept synonyms such as 'fixed roof', 'domed stadium', 'enclosed stadium', or 'indoor stadium' as fixed dome/roof. "
            "Reject if the stadium is explicitly open-air or only has a canopy/partial cover."
        )
    )

    # 7) Opening date before 2020 (critical, verify via cited sources)
    open_node = evaluator.add_leaf(
        id=f"{role_prefix}_Opening_Date",
        desc="The venue opened before January 1, 2020",
        parent=region_node,
        critical=True
    )
    if opening_year:
        open_claim = f"{name or 'This stadium'} opened in {opening_year}, which is before January 1, 2020."
    else:
        open_claim = f"{name or 'This stadium'} opened before January 1, 2020."
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=sources,
        additional_instruction="Use the initial grand opening year (not renovation or re-opening dates)."
    )

    # 8) NFL team(s) identified (critical, verify via cited sources)
    team_node = evaluator.add_leaf(
        id=f"{role_prefix}_NFL_Team",
        desc="The NFL team(s) that call this venue home are identified",
        parent=region_node,
        critical=True
    )
    teams_text = fmt_team_list(team_list)
    if teams_text:
        team_claim = f"{name or 'This stadium'} is the home stadium of the NFL team(s): {teams_text}."
    else:
        team_claim = f"{name or 'This stadium'} is the home stadium of at least one NFL team."
    await evaluator.verify(
        claim=team_claim,
        node=team_node,
        sources=sources,
        additional_instruction="Confirm the official NFL home team(s) from the provided source(s). Allow that the stadium may also host other events or teams."
    )

    # 9) ADA compliance: at least 700 wheelchair accessible seats (critical, verify via cited sources)
    ada_node = evaluator.add_leaf(
        id=f"{role_prefix}_ADA_Compliance",
        desc="The venue provides at least 700 wheelchair accessible seats (meeting 1% ADA requirement for 70,000+ capacity)",
        parent=region_node,
        critical=True
    )
    if ada_text:
        ada_claim = f"{name or 'This stadium'} provides at least 700 wheelchair accessible seats (e.g., the answer notes '{ada_text}')."
    else:
        ada_claim = f"{name or 'This stadium'} provides at least 700 wheelchair accessible seats."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit counts or clear statements demonstrating accessible seating >= 700. "
            "Accept if the source explicitly totals to >= 700 across sections or states a number >= 700. "
            "Do not accept vague ADA compliance statements without quantities."
        )
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
    Evaluate an answer for the NFL stadium selection task.
    """
    # Initialize evaluator (root as a neutral container)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at the very top
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

    # Add a top-level node representing the two-venue selection rubric
    selection_node = evaluator.add_parallel(
        id="Two_Venue_Selection",
        desc="Identify two suitable NFL stadiums meeting all specified criteria, one in the Eastern US and one in the Western US",
        parent=root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenueListExtraction,
        extraction_name="venue_list"
    )
    venues: List[VenueInfo] = extracted.venues if extracted and extracted.venues else []

    # Classify and pick candidates
    east_candidate, west_candidate, classification = pick_regional_candidates(venues)

    evaluator.add_custom_info(
        info={
            "extracted_venue_count": len(venues),
            "state_region_classification": classification,
            "east_candidate": (east_candidate.dict() if east_candidate else None),
            "west_candidate": (west_candidate.dict() if west_candidate else None)
        },
        info_type="analysis",
        info_name="candidate_selection_details"
    )

    # Build verification subtrees
    await verify_single_venue(evaluator, selection_node, east_candidate, "Eastern")
    await verify_single_venue(evaluator, selection_node, west_candidate, "Western")

    # Return structured summary
    return evaluator.get_summary()