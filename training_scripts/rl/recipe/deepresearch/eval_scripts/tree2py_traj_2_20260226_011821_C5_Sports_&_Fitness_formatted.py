import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nfl_stadiums_eligibility"
TASK_DESCRIPTION = (
    "Identify 4 NFL stadiums that meet all the following Super Bowl hosting eligibility criteria: "
    "(1) have a seating capacity of at least 70,000 during regular season operations, "
    "(2) are home to an NFL team, "
    "(3) are NOT currently scheduled to host Super Bowl LX (2026 at Levi's Stadium in Santa Clara), "
    "Super Bowl LXI (2027 at SoFi Stadium in Los Angeles), or Super Bowl LXII (2028 at Mercedes-Benz Stadium in Atlanta), and "
    "(4) if located in a city where the average temperature drops below 50°F, the stadium must be domed or have a retractable roof. "
    "For each stadium, provide its name, location, seating capacity, and a reference URL verifying this information."
)

SCHEDULED_SUPER_BOWLS = {
    "LX (2026)": "Levi's Stadium",
    "LXI (2027)": "SoFi Stadium",
    "LXII (2028)": "Mercedes-Benz Stadium",
}
SCHEDULED_STADIUM_NAMES = {"levi's stadium", "sofi stadium", "mercedes-benz stadium"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StadiumItem(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # City, State
    capacity: Optional[str] = None  # Keep as string (e.g., "70,240")
    home_team: Optional[str] = None  # If provided by the answer
    roof_type: Optional[str] = None  # e.g., "domed", "retractable", "open-air", or other string
    climate_cold: Optional[bool] = None  # True/False if explicitly stated; else null
    reference_urls: List[str] = Field(default_factory=list)

    # Optional extra fields (if provided in the answer)
    premium_seats: Optional[str] = None
    luxury_suites: Optional[str] = None
    electrical_kva: Optional[str] = None
    hotel_rooms_within_60min: Optional[str] = None


class StadiumsExtraction(BaseModel):
    stadiums: List[StadiumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return """
    Extract up to FOUR NFL stadiums mentioned in the answer that the user claims meet the eligibility criteria.
    For EACH stadium, return the following fields (use null when not available in the answer):
    - name: the stadium name
    - location: city and state (or equivalent location string)
    - capacity: the seating capacity during REGULAR SEASON operations (if a specific number is given, extract that number string; if only ranges or approximate wording are present, extract the exact phrase)
    - home_team: the NFL franchise that uses it as home stadium (if explicitly provided in the answer; otherwise null)
    - roof_type: one of ["domed", "retractable roof", "open-air", "covered", "unknown"] based on the answer text
    - climate_cold: true/false if the answer explicitly states the city's average temperature drops below 50°F; otherwise null
    - reference_urls: ALL URLs cited for this stadium (include any official site, Wikipedia, team page, stadium page, articles)—extract only actual URLs present in the answer, including Markdown links

    OPTIONAL (extract ONLY if the answer explicitly provides them):
    - premium_seats: string for premium seat count or description (e.g., "6,300 club seats"); else null
    - luxury_suites: string for luxury suite count (e.g., "72 suites"); else null
    - electrical_kva: string for electrical load capability (e.g., "≥6,000 kVA"); else null
    - hotel_rooms_within_60min: string indicating hotel room availability within 60-minute drive (e.g., "55,000 rooms"); else null

    IMPORTANT:
    - Do NOT invent or infer URLs or numbers; only extract what is explicitly present in the answer.
    - If more than 4 stadiums are provided, include ONLY the first 4.
    - If fewer than 4 are provided, include as many as appear (the rest will be null).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_to_int(capacity_str: Optional[str]) -> Optional[int]:
    if not capacity_str:
        return None
    # Extract digits from string, e.g., "70,240" -> 70240; "approx. 75,000 seats" -> 75000
    digits = re.findall(r"\d+", capacity_str.replace(",", ""))
    if not digits:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def make_sources_list(item: StadiumItem) -> List[str]:
    # Use all provided reference URLs for verification
    return item.reference_urls if item.reference_urls else []


# --------------------------------------------------------------------------- #
# Verification for a single stadium                                           #
# --------------------------------------------------------------------------- #
async def verify_stadium(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    item: StadiumItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single stadium and run all checks.
    """
    stadium_idx = index + 1
    stadium_title = item.name or f"Stadium #{stadium_idx}"

    # Create stadium node (non-critical to allow partial credit across the set of 4)
    stadium_node = evaluator.add_parallel(
        id=f"stadium_{stadium_idx}",
        desc=f"{['First','Second','Third','Fourth'][index]} eligible stadium identification and verification",
        parent=parent_node,
        critical=False,
    )

    # ------------------------ Existence checks (critical) ------------------------
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id=f"stadium_{stadium_idx}_name",
        desc="The stadium name is provided",
        parent=stadium_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(item.location and item.location.strip()),
        id=f"stadium_{stadium_idx}_location",
        desc="The stadium location is provided",
        parent=stadium_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(item.capacity and item.capacity.strip()),
        id=f"stadium_{stadium_idx}_capacity_value",
        desc="The specific seating capacity value is provided",
        parent=stadium_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(item.reference_urls and len(item.reference_urls) > 0),
        id=f"stadium_{stadium_idx}_reference",
        desc="Provide a reference URL supporting the stadium's capacity and characteristics",
        parent=stadium_node,
        critical=True,
    )

    # ------------------------ Capacity threshold (critical) ---------------------
    cap_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_capacity",
        desc="The stadium has a seating capacity of at least 70,000 during regular season operations",
        parent=stadium_node,
        critical=True,
    )
    capacity_claim = (
        f"{stadium_title} has a regular-season seating capacity of at least 70,000."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=cap_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "Use the reference page(s) to confirm seating capacity. Prefer 'seating capacity' for regular season, "
            "not 'expandable' event capacities. Minor variations or approximate numbers are acceptable as long as "
            "the regular seating capacity is >= 70,000."
        ),
    )

    # ------------------------ Home NFL team (critical) --------------------------
    nfl_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_nfl_team",
        desc="The stadium is home to an NFL team",
        parent=stadium_node,
        critical=True,
    )
    team_phrase = item.home_team if (item.home_team and item.home_team.strip()) else "an NFL franchise"
    nfl_claim = f"{stadium_title} is the home stadium for {team_phrase}."
    await evaluator.verify(
        claim=nfl_claim,
        node=nfl_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "From the provided reference(s), verify that the stadium serves as the home venue for an NFL team "
            "(e.g., 'home of the [Team]' or 'plays home games at'). Allow minor variations in team naming."
        ),
    )

    # ------------------------ Not scheduled for SB LX/LXI/LXII (critical) -------
    not_sched_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_not_scheduled",
        desc="The stadium is not scheduled to host Super Bowl LX (2026 at Levi's Stadium), LXI (2027 at SoFi Stadium), or LXII (2028 at Mercedes-Benz Stadium)",
        parent=stadium_node,
        critical=True,
    )
    not_sched_claim = (
        f"The stadium named '{stadium_title}' is not Levi's Stadium, SoFi Stadium, nor Mercedes-Benz Stadium."
    )
    await evaluator.verify(
        claim=not_sched_claim,
        node=not_sched_leaf,
        sources=None,  # Pure logical name check; no external sources needed
        additional_instruction=(
            "This is a simple name check. Consider case-insensitive and minor punctuation differences. If the name "
            "corresponds to Levi's Stadium (Santa Clara), SoFi Stadium (Inglewood/Los Angeles), or Mercedes-Benz Stadium (Atlanta), "
            "mark Incorrect."
        ),
    )

    # ------------------------ Cold-climate roof rule (non-critical) -------------
    roof_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_weather_dome",
        desc="If located in a cold climate (average temperature below 50°F), the stadium must be domed or have a retractable roof",
        parent=stadium_node,
        critical=False,
    )
    roof_claim = (
        f"If {stadium_title} is located in a city where the average temperature drops below 50°F, then the stadium "
        f"is domed or has a retractable roof; otherwise, any roof type is acceptable."
    )
    await evaluator.verify(
        claim=roof_claim,
        node=roof_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "Use the provided reference(s) to determine roof type if possible. If the stadium is open-air and "
            "no climate information is available from the reference(s), mark as Not Supported. If the stadium has a "
            "domed or retractable roof (explicitly stated), mark as Supported."
        ),
    )

    # ------------------------ Optional extra checks (non-critical) --------------
    # Premium seats >= 6,000
    prem_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_premium_seats",
        desc="The stadium has at least 6,000 premium seats",
        parent=stadium_node,
        critical=False,
    )
    prem_claim = f"{stadium_title} has at least 6,000 premium or club seats."
    await evaluator.verify(
        claim=prem_claim,
        node=prem_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "Check the reference(s) for premium/club seating counts. If not present or fewer than 6,000, mark Not Supported."
        ),
    )

    # Luxury suites >= 70
    suites_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_luxury_suites",
        desc="The stadium has at least 70 luxury suites",
        parent=stadium_node,
        critical=False,
    )
    suites_claim = f"{stadium_title} has at least 70 luxury suites."
    await evaluator.verify(
        claim=suites_claim,
        node=suites_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "Check the reference(s) for suite counts. If not present or fewer than 70, mark Not Supported."
        ),
    )

    # Electrical loads >= 6,000 kVA
    elec_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_electrical",
        desc="The stadium has at least 6,000 kVA electrical loads",
        parent=stadium_node,
        critical=False,
    )
    elec_claim = f"{stadium_title} supports electrical loads of at least 6,000 kVA."
    await evaluator.verify(
        claim=elec_claim,
        node=elec_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "If the electrical capacity is not provided in the reference(s), mark Not Supported."
        ),
    )

    # Hotels within 60-minute drive >= 35% of stadium capacity
    hotels_leaf = evaluator.add_leaf(
        id=f"stadium_{stadium_idx}_hotels",
        desc="The host city has hotel rooms equal to at least 35% of the stadium's capacity within a 60-minute drive",
        parent=stadium_node,
        critical=False,
    )
    # Try computing threshold if capacity is numeric; otherwise keep generic claim
    cap_num = parse_capacity_to_int(item.capacity)
    if cap_num is not None:
        threshold = int(cap_num * 0.35)
        hotels_claim = (
            f"The hotels within a 60-minute drive of {item.location or 'the stadium'} provide at least {threshold} rooms "
            f"(>= 35% of the stadium's capacity of approximately {cap_num})."
        )
    else:
        hotels_claim = (
            f"The host city has hotel rooms equal to at least 35% of {stadium_title}'s seating capacity within a 60-minute drive."
        )
    await evaluator.verify(
        claim=hotels_claim,
        node=hotels_leaf,
        sources=make_sources_list(item),
        additional_instruction=(
            "If the reference(s) do not provide hotel room counts or suitable evidence, mark Not Supported."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the answer for the NFL stadium eligibility task using the Mind2Web2 evaluator.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across the four stadiums
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

    # Record ground truth context for scheduled Super Bowls
    evaluator.add_ground_truth(
        {
            "scheduled_super_bowls": SCHEDULED_SUPER_BOWLS,
            "scheduled_stadium_names": list(SCHEDULED_STADIUM_NAMES),
        },
        gt_type="scheduled_super_bowls_info",
    )

    # Extract stadiums from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumsExtraction,
        extraction_name="stadiums_extraction",
    )

    # Use only the first 4 stadiums; pad with empty items if fewer
    extracted_items = extraction.stadiums[:4]
    while len(extracted_items) < 4:
        extracted_items.append(StadiumItem())

    # Build verification for each stadium
    for i, item in enumerate(extracted_items):
        await verify_stadium(evaluator, root, item, i)

    # Return evaluation summary
    return evaluator.get_summary()