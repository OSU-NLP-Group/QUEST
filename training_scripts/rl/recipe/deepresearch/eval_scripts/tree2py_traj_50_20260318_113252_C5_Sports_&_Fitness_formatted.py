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
TASK_ID = "nba_allstar_smallest_capacity_2024_2027"
TASK_DESCRIPTION = (
    "Among the cities that hosted or will host the NBA All-Star Game in 2024, 2025, 2026, and 2027, identify the city "
    "whose hosting arena has the smallest seating capacity for basketball games. For this identified city, provide the "
    "following information: (1) The city name, (2) The arena name, (3) The arena's seating capacity for basketball games, "
    "(4) The NBA team that calls this arena home, (5) The year this city hosted or will host the NBA All-Star Game, "
    "(6) A reference URL confirming the arena's specifications (name, capacity, home team), "
    "(7) A reference URL confirming the city as an NBA All-Star Game host in the specified year, "
    "(8) The approximate total number of hotel rooms available in the city with a supporting reference URL, and "
    "(9) The square footage of the city's primary convention center with a supporting reference URL."
)

ALLOWED_YEARS = ["2024", "2025", "2026", "2027"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HostItem(BaseModel):
    year: Optional[str] = None
    city: Optional[str] = None
    arena: Optional[str] = None
    capacity_basketball: Optional[str] = None
    source_url: Optional[str] = None


class SelectedCityInfo(BaseModel):
    city: Optional[str] = None
    hosting_year: Optional[str] = None
    arena_name: Optional[str] = None
    arena_capacity_basketball: Optional[str] = None
    nba_home_team: Optional[str] = None

    arena_specs_url: Optional[str] = None
    host_city_year_url: Optional[str] = None
    official_venue_url: Optional[str] = None

    hotel_room_count: Optional[str] = None
    hotel_rooms_url: Optional[str] = None

    convention_center_name: Optional[str] = None
    convention_center_square_footage: Optional[str] = None
    convention_center_url: Optional[str] = None


class AllStarExtraction(BaseModel):
    selected: Optional[SelectedCityInfo] = None
    comparison_hosts: List[HostItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_allstar_package() -> str:
    return """
    Your task is to extract a structured package of information from the provided answer text about the NBA All-Star Game hosts from 2024 to 2027, with a focus on the one city whose hosting arena has the smallest seating capacity (basketball configuration).

    STRICT RULES:
    - Extract ONLY what is explicitly present in the answer. Do not invent, infer, or add any new information not explicitly in the answer.
    - Return null for any missing field.
    - When extracting URLs, return full valid URLs exactly as shown in the answer (parse from markdown links if needed).

    PART A: Selected (smallest capacity) city bundle
    Extract the final selected city (the one claimed to have the smallest basketball capacity among 2024–2027 hosts). Include:
      - city
      - hosting_year (must be one of: 2024, 2025, 2026, 2027; if not stated, return null)
      - arena_name (official All-Star Game venue for that city/year)
      - arena_capacity_basketball (capacity stated specifically for basketball configuration, as shown in the answer; keep formatting and punctuation like commas)
      - nba_home_team (NBA team that calls this arena home)
      - arena_specs_url (URL that supports arena name, basketball capacity, and home team)
      - host_city_year_url (URL that supports the city/year hosting status)
      - official_venue_url (URL that confirms the identified arena is the official All-Star venue for that specified year)
      - hotel_room_count (approximate total number of hotel rooms available in the city; keep formatting, e.g., "about 30,000")
      - hotel_rooms_url (URL that supports the hotel rooms number)
      - convention_center_name (name of the city's primary/main convention center, if mentioned)
      - convention_center_square_footage (the square footage of the city's primary convention center; keep formatting, e.g., "1.2 million sq ft" or "750,000 square feet")
      - convention_center_url (URL that supports the convention center square footage and identity)

    PART B: Comparison set (optional but helpful)
    If the answer provides any comparison information for the 2024–2027 hosts, extract up to four items with:
      - year (one of 2024, 2025, 2026, 2027 if stated)
      - city
      - arena
      - capacity_basketball (basketball configuration capacity as stated)
      - source_url (a URL tied to that arena/capacity if present)

    Return JSON with:
    - selected: {...}  // as above
    - comparison_hosts: [ ... ]  // zero to four items, only what is explicitly in the answer
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _nz(s: Optional[str]) -> str:
    return s or ""


def _format_comparison_for_claim(items: List[HostItem]) -> str:
    if not items:
        return "No comparison capacities were provided in the answer."
    lines = []
    for it in items:
        yr = _nz(it.year)
        city = _nz(it.city)
        arena = _nz(it.arena)
        cap = _nz(it.capacity_basketball)
        lines.append(f"- {yr} | {city} | {arena} | basketball capacity: {cap}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_city_min_capacity(
    evaluator: Evaluator,
    root,
    selected: SelectedCityInfo,
    comparison_hosts: List[HostItem],
):
    """
    Build 'city_arena_and_min_capacity' subtree and run verifications.
    """
    group = evaluator.add_parallel(
        id="city_arena_and_min_capacity",
        desc="Identify the correct city/arena and establish it is the minimum basketball capacity among the four host cities (2024–2027).",
        parent=root,
        critical=True,
    )

    # Existence gate: ensure core facts and URLs are provided
    existence_ok = all([
        bool(selected and selected.city and selected.city.strip()),
        bool(selected and selected.hosting_year and selected.hosting_year.strip()),
        bool(selected and selected.arena_name and selected.arena_name.strip()),
        bool(selected and selected.arena_capacity_basketball and selected.arena_capacity_basketball.strip()),
        bool(selected and selected.nba_home_team and selected.nba_home_team.strip()),
        _is_nonempty_url(getattr(selected, "arena_specs_url", None)),
        _is_nonempty_url(getattr(selected, "host_city_year_url", None)),
        _is_nonempty_url(getattr(selected, "official_venue_url", None)),
    ])
    evaluator.add_custom_node(
        result=existence_ok,
        id="selected_info_present",
        desc="Selected city/year/arena/capacity/team and required reference URLs are provided.",
        parent=group,
        critical=True,
    )

    # City name verification (host city/year)
    city_leaf = evaluator.add_leaf(
        id="city_name",
        desc="Provide the city name (must be one of the 2024, 2025, 2026, or 2027 NBA All-Star host cities).",
        parent=group,
        critical=True,
    )
    city_claim = f"The NBA All-Star Game in { _nz(selected.hosting_year) } was/will be hosted in { _nz(selected.city) }."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=_nz(selected.host_city_year_url),
        additional_instruction="Verify that the specified city hosted/will host the NBA All-Star Game in the given year.",
    )

    # Hosting year verification (redundant check to ensure the year is correct)
    year_leaf = evaluator.add_leaf(
        id="hosting_year",
        desc="Provide the year (2024/2025/2026/2027) in which the city hosted or will host the NBA All-Star Game.",
        parent=group,
        critical=True,
    )
    year_claim = f"{_nz(selected.city)} hosted/will host the NBA All-Star Game in {_nz(selected.hosting_year)}."
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=_nz(selected.host_city_year_url),
        additional_instruction="Confirm the year from the provided host city/year source. The year must be one of 2024, 2025, 2026, or 2027.",
    )

    # Arena name (official venue for that year)
    arena_leaf = evaluator.add_leaf(
        id="arena_name",
        desc="Provide the official venue (arena) name where the NBA All-Star Game was/will be held for that city/year.",
        parent=group,
        critical=True,
    )
    arena_claim = f"The official venue for the NBA All-Star Game {_nz(selected.hosting_year)} in {_nz(selected.city)} is {_nz(selected.arena_name)}."
    await evaluator.verify(
        claim=arena_claim,
        node=arena_leaf,
        sources=_nz(selected.official_venue_url),
        additional_instruction="Verify that this arena is explicitly identified as the official All-Star Game venue for the specified year.",
    )

    # Arena basketball seating capacity (from arena specs URL)
    cap_leaf = evaluator.add_leaf(
        id="arena_capacity_basketball_config",
        desc="Provide the arena seating capacity specifically for basketball configuration (not another event configuration).",
        parent=group,
        critical=True,
    )
    cap_claim = f"The basketball seating capacity of {_nz(selected.arena_name)} is {_nz(selected.arena_capacity_basketball)}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=_nz(selected.arena_specs_url),
        additional_instruction="Confirm the seating capacity specifically for basketball configuration. Allow minor rounding differences.",
    )

    # NBA home team (from arena specs URL)
    team_leaf = evaluator.add_leaf(
        id="nba_home_team",
        desc="Provide the NBA team that calls the arena home (arena must be an NBA home venue).",
        parent=group,
        critical=True,
    )
    team_claim = f"{_nz(selected.arena_name)} is the home arena of the {_nz(selected.nba_home_team)}."
    await evaluator.verify(
        claim=team_claim,
        node=team_leaf,
        sources=_nz(selected.arena_specs_url),
        additional_instruction="Verify that the arena is an NBA home venue for the stated team.",
    )

    # Smallest capacity verification (logical check based on answer-provided comparison set)
    smallest_leaf = evaluator.add_leaf(
        id="smallest_capacity_verification",
        desc="The identified arena's basketball seating capacity is the smallest among the official All-Star Game arenas for the 2024–2027 host cities.",
        parent=group,
        critical=True,
    )
    comparison_listing = _format_comparison_for_claim(comparison_hosts)
    smallest_claim = (
        f"Based on the comparison capacities explicitly provided in the answer:\n"
        f"{comparison_listing}\n\n"
        f"Therefore, {_nz(selected.arena_name)} in {_nz(selected.city)} "
        f"({_nz(selected.arena_capacity_basketball)} basketball capacity) is the smallest among the 2024–2027 All-Star arenas."
    )
    await evaluator.verify(
        claim=smallest_claim,
        node=smallest_leaf,
        sources=None,
        additional_instruction=(
            "Judge only using the capacities and items explicitly listed in the provided answer text above. "
            "If the answer does not provide sufficient comparison data for all four hosts (2024–2027), "
            "consider the 'smallest' claim not supported."
        ),
    )


async def build_and_verify_hotel_rooms(
    evaluator: Evaluator,
    root,
    selected: SelectedCityInfo,
):
    """
    Build 'hotel_rooms_fact' subtree and run verifications.
    """
    group = evaluator.add_parallel(
        id="hotel_rooms_fact",
        desc="Provide the approximate total number of hotel rooms available in the city.",
        parent=root,
        critical=True,
    )

    # Existence gate
    hotel_exist_ok = bool(selected and selected.hotel_room_count and selected.hotel_room_count.strip()) and _is_nonempty_url(
        getattr(selected, "hotel_rooms_url", None)
    )
    evaluator.add_custom_node(
        result=hotel_exist_ok,
        id="hotel_rooms_info_present",
        desc="Hotel rooms figure and a supporting URL are provided.",
        parent=group,
        critical=True,
    )

    # Hotel room count verification
    hotel_count_leaf = evaluator.add_leaf(
        id="hotel_room_count",
        desc="State an approximate total hotel room count for the city (a single approximate figure or clearly bounded estimate).",
        parent=group,
        critical=True,
    )
    hotel_claim = (
        f"The city of {_nz(selected.city)} has approximately {_nz(selected.hotel_room_count)} hotel rooms "
        f"(citywide or metro inventory)."
    )
    await evaluator.verify(
        claim=hotel_claim,
        node=hotel_count_leaf,
        sources=_nz(selected.hotel_rooms_url),
        additional_instruction="Allow approximate or rounded counts. Citywide or metro-level inventory is acceptable.",
    )

    # Hotel rooms relevance to hosting (context check)
    hotel_rel_leaf = evaluator.add_leaf(
        id="hotel_rooms_relevance_to_hosting",
        desc="The hotel-room information is presented in a way relevant to major event hosting capacity (e.g., citywide/metro inventory context, tourism bureau/industry inventory context, or otherwise explicitly framed as event-capacity relevant).",
        parent=group,
        critical=True,  # Marked critical to satisfy framework constraints for critical parent
    )
    hotel_rel_claim = (
        "This source presents hotel rooms information in a context relevant to hosting major events—"
        "for example, citywide/metro inventory, tourism bureau or industry metrics, or explicit event-capacity framing."
    )
    await evaluator.verify(
        claim=hotel_rel_claim,
        node=hotel_rel_leaf,
        sources=_nz(selected.hotel_rooms_url),
        additional_instruction="Pass if the page is an official tourism/CVB, government, industry, or otherwise event-capacity-relevant context.",
    )


async def build_and_verify_convention_center(
    evaluator: Evaluator,
    root,
    selected: SelectedCityInfo,
):
    """
    Build 'convention_center_fact' subtree and run verifications.
    """
    group = evaluator.add_parallel(
        id="convention_center_fact",
        desc="Provide the square footage of the city's primary convention center.",
        parent=root,
        critical=True,
    )

    # Existence gate
    cc_exist_ok = bool(selected and selected.convention_center_square_footage and selected.convention_center_square_footage.strip()) and _is_nonempty_url(
        getattr(selected, "convention_center_url", None)
    )
    evaluator.add_custom_node(
        result=cc_exist_ok,
        id="convention_center_info_present",
        desc="Convention center square footage and a supporting URL are provided.",
        parent=group,
        critical=True,
    )

    # Square footage verification
    cc_sqft_leaf = evaluator.add_leaf(
        id="convention_center_square_footage",
        desc="State the square footage of the city's primary/main convention center.",
        parent=group,
        critical=True,
    )
    cc_name_part = f"{_nz(selected.convention_center_name)}, " if selected and selected.convention_center_name else ""
    cc_sqft_claim = (
        f"The primary convention center in {_nz(selected.city)}, {cc_name_part}has approximately "
        f"{_nz(selected.convention_center_square_footage)} of space (square footage)."
    )
    await evaluator.verify(
        claim=cc_sqft_claim,
        node=cc_sqft_leaf,
        sources=_nz(selected.convention_center_url),
        additional_instruction=(
            "Confirm the stated square footage. Allow reasonable rounding and units variations "
            "(e.g., '1.2 million sq ft' vs '1,200,000 square feet')."
        ),
    )

    # Relevance/context verification
    cc_rel_leaf = evaluator.add_leaf(
        id="convention_center_relevance_to_hosting",
        desc="The convention-center information is presented in a way relevant to major event hosting (e.g., identifies it as the primary/main convention center and provides size/capacity context).",
        parent=group,
        critical=True,  # Marked critical to satisfy framework constraints for critical parent
    )
    cc_rel_claim = (
        "This source identifies the facility as the city's primary/main convention center or otherwise provides "
        "clear size/capacity context relevant to hosting major events."
    )
    await evaluator.verify(
        claim=cc_rel_claim,
        node=cc_rel_leaf,
        sources=_nz(selected.convention_center_url),
        additional_instruction="Pass if the facility is described as primary/main or the context is about venue size/capacity for major events.",
    )


async def build_and_verify_reference_urls(
    evaluator: Evaluator,
    root,
    selected: SelectedCityInfo,
):
    """
    Build 'reference_urls_required' subtree. Here we primarily check the presence of required URLs.
    The factual support of those URLs is already verified in other groups.
    """
    group = evaluator.add_parallel(
        id="reference_urls_required",
        desc="Provide verifiable reference URLs for each required factual bundle (arena specs; host city/year; official venue; hotel rooms; convention center size).",
        parent=root,
        critical=True,
    )

    # Each of the following are custom presence checks (critical).
    evaluator.add_custom_node(
        result=_is_nonempty_url(getattr(selected, "arena_specs_url", None)),
        id="arena_specs_url",
        desc="Provide a reference URL confirming the arena's name, basketball seating capacity, and NBA home team.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url(getattr(selected, "host_city_year_url", None)),
        id="host_city_year_url",
        desc="Provide a reference URL confirming the city hosted/will host the NBA All-Star Game in the specified year.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url(getattr(selected, "official_venue_url", None)),
        id="official_venue_url",
        desc="Provide a reference URL confirming that the identified arena is the official venue for the NBA All-Star Game for the specified year.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url(getattr(selected, "hotel_rooms_url", None)),
        id="hotel_rooms_url",
        desc="Provide a reference URL supporting the approximate total number of hotel rooms available in the city.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url(getattr(selected, "convention_center_url", None)),
        id="convention_center_url",
        desc="Provide a reference URL supporting the square footage of the city's primary convention center.",
        parent=group,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Evaluate an answer for the NBA All-Star (2024–2027 smallest arena capacity) task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_allstar_package(),
        template_class=AllStarExtraction,
        extraction_name="selected_city_package",
    )

    # Record ground truth constraints/context for transparency
    evaluator.add_ground_truth({
        "allowed_years": ALLOWED_YEARS,
        "requirement": "Identify the 2024–2027 NBA All-Star host city whose arena has the smallest basketball capacity, with supporting references."
    })

    # Add custom info: what comparison set the answer provided (if any)
    comp_info = [{"year": it.year, "city": it.city, "arena": it.arena, "capacity_basketball": it.capacity_basketball, "url": it.source_url}
                 for it in (extracted.comparison_hosts or [])]
    evaluator.add_custom_info({"comparison_hosts_extracted": comp_info}, info_type="extraction_note", info_name="comparison_hosts")

    # Normalize selected info object
    selected = extracted.selected or SelectedCityInfo()

    # Build and verify tree sections
    await build_and_verify_city_min_capacity(evaluator, root, selected, extracted.comparison_hosts or [])
    await build_and_verify_hotel_rooms(evaluator, root, selected)
    await build_and_verify_convention_center(evaluator, root, selected)
    await build_and_verify_reference_urls(evaluator, root, selected)

    # Return structured summary
    return evaluator.get_summary()