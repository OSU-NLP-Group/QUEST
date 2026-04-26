import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_indoor_arenas_grammys_2026"
TASK_DESCRIPTION = """
Identify 4 major indoor concert arenas, each located in a different US city, that meet the following requirements:

1. Chicago Arena: An indoor concert arena in Chicago, Illinois, with a concert seating capacity of at least 20,000, that has hosted at least one artist who won a major Grammy award (Album of the Year, Record of the Year, Song of the Year, or Best New Artist) at the 2026 Grammy Awards (held February 1, 2026).

2. Atlanta Arena: An indoor concert arena in Atlanta, Georgia, with a concert seating capacity between 15,000 and 18,000, that has hosted at least one artist who won a major Grammy award (Album of the Year, Record of the Year, Song of the Year, or Best New Artist) at the 2026 Grammy Awards.

3. New York Arena: An indoor concert arena located in one of the five boroughs of New York City (Manhattan, Brooklyn, Queens, Bronx, or Staten Island), with a concert seating capacity of at least 17,000, that has hosted at least one artist who won a major Grammy award (Album of the Year, Record of the Year, Song of the Year, or Best New Artist) at the 2026 Grammy Awards.

4. Los Angeles Arena: An indoor concert arena in Los Angeles, California, with a concert seating capacity of at least 18,000, that hosted the 68th Annual Grammy Awards ceremony on February 1, 2026.

For each arena, provide:
- The arena name
- Its location (city and state)
- Its concert seating capacity
- At least one Grammy-winning artist (from the 2026 major categories) who has performed there
- A reference URL confirming the arena's details
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArtistInfo(BaseModel):
    name: Optional[str] = None
    category_won: Optional[str] = None  # Expected: Album of the Year / Record ... / Song ... / Best New Artist
    award_year: Optional[str] = None    # Expected "2026" or a date string indicating 2026 Grammys
    performance_urls: List[str] = Field(default_factory=list)  # URLs evidencing performance at the arena
    award_urls: List[str] = Field(default_factory=list)        # URLs evidencing the 2026 major Grammy award


class ArenaInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    borough: Optional[str] = None  # For NYC arena specifically
    capacity: Optional[str] = None  # Keep as string to be robust to ranges or approx
    is_indoor: Optional[str] = None  # e.g., "indoor", "outdoor", "unknown"
    reference_urls: List[str] = Field(default_factory=list)  # URLs confirming arena details (location, capacity)
    # Special-case for LA: URLs confirming Grammy hosting detail
    grammy_host_urls: List[str] = Field(default_factory=list)
    # Artists list (we will use the first one for verification)
    artists: List[ArtistInfo] = Field(default_factory=list)


class ArenasExtraction(BaseModel):
    chicago: Optional[ArenaInfo] = None
    atlanta: Optional[ArenaInfo] = None
    new_york: Optional[ArenaInfo] = None
    los_angeles: Optional[ArenaInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return """
    Extract structured information for four arenas mentioned in the answer, one for each target city/region:
    - Chicago, Illinois
    - Atlanta, Georgia
    - New York City (one of the five boroughs: Manhattan, Brooklyn, Queens, Bronx, Staten Island)
    - Los Angeles, California

    For each of these four, return a JSON object with fields:
    1) name: The arena name as stated.
    2) city: The city for the arena (e.g., "Chicago", "Atlanta", "Brooklyn", "Los Angeles").
    3) state: The state for the arena (e.g., "Illinois", "Georgia", "New York", "California").
    4) borough: ONLY for NYC arena—if the answer indicates a borough explicitly (Manhattan/Brooklyn/Queens/Bronx/Staten Island). Otherwise null.
    5) capacity: The concert seating capacity as stated (string; do not convert to number; include any ranges or qualifiers).
    6) is_indoor: If the answer explicitly states indoor/outdoor, put "indoor" or "outdoor"; else "unknown".
    7) reference_urls: An array of URL(s) that confirm the arena’s details (location and capacity). Extract only URLs explicitly present in the answer.
    8) grammy_host_urls: For the Los Angeles arena ONLY—include any URL(s) that confirm the arena hosted the 68th Annual Grammy Awards on February 1, 2026. If not provided, return an empty array.
    9) artists: An array of artist objects (extract at least one if present; if multiple are given, include them in order).
       Each artist object has:
         - name: The artist's name.
         - category_won: The major Grammy category the artist won at the 2026 Grammys, if stated (Album of the Year, Record of the Year, Song of the Year, Best New Artist). If not clearly stated, put null.
         - award_year: The year of the award if stated (expect "2026"). If not clearly stated, put null.
         - performance_urls: URLs that indicate the artist performed at the arena (e.g., event listing, news, ticketing).
         - award_urls: URLs that indicate the artist won a major award at the 2026 Grammys.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer; do not invent URLs.
    - If a field is missing in the answer, return null (or empty array for list fields).
    - If the answer mentions more than one candidate per city/region, return the first reasonable one mentioned.
    - The "capacity" field should be a string exactly as written in the answer; do not normalize to numbers.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_arena_common_details(
    evaluator: Evaluator,
    parent_node,
    city_label: str,
    info: ArenaInfo,
    *,
    location_claim: str,
    capacity_claim: str,
    indoor_claim: str,
) -> None:
    """
    Verify name provided, reference URLs provided, and core details (location, capacity, indoor).
    Core detail validations depend on reference URLs existence.
    """
    # Existence checks
    evaluator.add_custom_node(
        result=_non_empty_str(info.name),
        id=f"{city_label}_Name",
        desc="Provide the arena name.",
        parent=parent_node,
        critical=True
    )

    ref_node = evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id=f"{city_label}_Reference_URL",
        desc="Provide at least one reference URL confirming the arena's details (e.g., location and concert seating capacity).",
        parent=parent_node,
        critical=True
    )

    # Group core details to avoid global critical sibling gating
    details_group = evaluator.add_parallel(
        id=f"{city_label}_Details",
        desc=f"{city_label.replace('_', ' ')} core details verification (location, capacity, indoor)",
        parent=parent_node,
        critical=False
    )

    # Location verification (critical)
    loc_node = evaluator.add_leaf(
        id=f"{city_label}_Location",
        desc=f"Arena is located as required for {city_label.replace('_', ' ')}.",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim=location_claim,
        node=loc_node,
        sources=info.reference_urls,
        additional_instruction="Verify the arena's city and state (or borough for NYC) using the provided reference URL(s).",
        extra_prerequisites=[ref_node]
    )

    # Indoor verification (critical)
    indoor_node = evaluator.add_leaf(
        id=f"{city_label}_Indoor",
        desc="Arena is an indoor concert arena (not an outdoor amphitheater/stadium).",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=info.reference_urls,
        additional_instruction="Check if the reference page indicates 'indoor arena' or similar. If the page clearly identifies the venue as an indoor arena, pass.",
        extra_prerequisites=[ref_node]
    )

    # Capacity verification (critical)
    cap_node = evaluator.add_leaf(
        id=f"{city_label}_Capacity",
        desc="Concert seating capacity requirement is satisfied.",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=cap_node,
        sources=info.reference_urls,
        additional_instruction="Use the reference URL(s) to confirm the stated seating capacity meets the threshold/range. Accept typical concert/basketball seating capacity figures when clearly applicable. Allow minor rounding or phrasing variants.",
        extra_prerequisites=[ref_node]
    )


async def _verify_artist_requirements(
    evaluator: Evaluator,
    parent_node,
    city_label: str,
    info: ArenaInfo
) -> None:
    """
    Verify that at least one artist satisfies:
    - Performed at the arena
    - Won a major Grammy category at the 2026 Grammys
    """
    # Create a critical artist verification group
    artist_group = evaluator.add_parallel(
        id=f"{city_label}_Artist_Group",
        desc=f"{city_label.replace('_', ' ')} qualifying artist verification",
        parent=parent_node,
        critical=True
    )

    has_artist = bool(info.artists) and _non_empty_str(info.artists[0].name)
    artist_exists_node = evaluator.add_custom_node(
        result=has_artist,
        id=f"{city_label}_Qualifying_Artist_Provided",
        desc="Provide at least one artist who (a) won a major Grammy category at the 2026 Grammy Awards and (b) is stated to have performed at this arena.",
        parent=artist_group,
        critical=True
    )

    # If artist exists, verify performance and award status separately
    artist_name = info.artists[0].name if has_artist else ""
    perf_urls = info.artists[0].performance_urls if has_artist else []
    award_urls = info.artists[0].award_urls if has_artist else []

    # Performance verification
    perf_node = evaluator.add_leaf(
        id=f"{city_label}_Artist_Performed",
        desc="The provided artist performed at this arena.",
        parent=artist_group,
        critical=True
    )
    perf_sources = _combine_sources(perf_urls, info.reference_urls)
    await evaluator.verify(
        claim=f"{artist_name} has performed at {info.name}.",
        node=perf_node,
        sources=perf_sources,
        additional_instruction="Confirm via credible event pages, official venue listings, or news articles that the artist performed at the specified arena.",
        extra_prerequisites=[artist_exists_node]
    )

    # Award verification (2026 major category)
    award_node = evaluator.add_leaf(
        id=f"{city_label}_Artist_2026_Major_Winner",
        desc="The provided artist won a major Grammy category at the 2026 Grammy Awards.",
        parent=artist_group,
        critical=True
    )
    # Clarify major categories
    major_categories = "Album of the Year, Record of the Year, Song of the Year, or Best New Artist"
    await evaluator.verify(
        claim=f"{artist_name} won a major Grammy category ({major_categories}) at the 2026 Grammy Awards (held on February 1, 2026).",
        node=award_node,
        sources=award_urls,
        additional_instruction="Verify the artist's 2026 Grammy win in the specified major categories using authoritative sources (e.g., Grammy.com, reputable news).",
        extra_prerequisites=[artist_exists_node]
    )


# --------------------------------------------------------------------------- #
# City-specific verification builders                                         #
# --------------------------------------------------------------------------- #
async def verify_chicago(evaluator: Evaluator, root_node, info: ArenaInfo) -> None:
    group = evaluator.add_parallel(
        id="Arena_in_Chicago",
        desc="Chicago arena requirements and required provided details",
        parent=root_node,
        critical=False
    )

    # Common details
    await _verify_arena_common_details(
        evaluator,
        group,
        city_label="Chicago",
        info=info,
        location_claim=f"The arena is located in Chicago, Illinois.",
        capacity_claim=f"The arena's concert seating capacity is at least 20,000.",
        indoor_claim="The arena is an indoor concert arena."
    )

    # Artist requirements
    await _verify_artist_requirements(evaluator, group, "Chicago", info)


async def verify_atlanta(evaluator: Evaluator, root_node, info: ArenaInfo) -> None:
    group = evaluator.add_parallel(
        id="Arena_in_Atlanta",
        desc="Atlanta arena requirements and required provided details",
        parent=root_node,
        critical=False
    )

    # Common details
    await _verify_arena_common_details(
        evaluator,
        group,
        city_label="Atlanta",
        info=info,
        location_claim=f"The arena is located in Atlanta, Georgia.",
        capacity_claim=f"The arena's concert seating capacity is between 15,000 and 18,000, inclusive.",
        indoor_claim="The arena is an indoor concert arena."
    )

    # Artist requirements
    await _verify_artist_requirements(evaluator, group, "Atlanta", info)


async def verify_new_york(evaluator: Evaluator, root_node, info: ArenaInfo) -> None:
    group = evaluator.add_parallel(
        id="Arena_in_New_York",
        desc="New York City (borough) arena requirements and required provided details",
        parent=root_node,
        critical=False
    )

    # Build borough claim if possible; otherwise generic borough-in-NYC claim
    if _non_empty_str(info.borough):
        borough = info.borough.strip()
        location_claim = f"The arena is located in the {borough} borough of New York City, New York."
    else:
        # Use city/state as a fallback; LLM can judge borough membership if page indicates one of the five
        location_claim = (
            "The arena is located in one of the five boroughs of New York City "
            "(Manhattan, Brooklyn, Queens, the Bronx, or Staten Island), New York."
        )

    await _verify_arena_common_details(
        evaluator,
        group,
        city_label="NY",
        info=info,
        location_claim=location_claim,
        capacity_claim=f"The arena's concert seating capacity is at least 17,000.",
        indoor_claim="The arena is an indoor concert arena."
    )

    # Artist requirements
    await _verify_artist_requirements(evaluator, group, "NY", info)


async def verify_los_angeles(evaluator: Evaluator, root_node, info: ArenaInfo) -> None:
    group = evaluator.add_parallel(
        id="Arena_in_Los_Angeles",
        desc="Los Angeles arena requirements and required provided details",
        parent=root_node,
        critical=False
    )

    # Common details
    await _verify_arena_common_details(
        evaluator,
        group,
        city_label="LA",
        info=info,
        location_claim=f"The arena is located in Los Angeles, California.",
        capacity_claim=f"The arena's concert seating capacity is at least 18,000.",
        indoor_claim="The arena is an indoor concert arena."
    )

    # Grammy hosting verification (critical)
    host_node = evaluator.add_leaf(
        id="LA_Grammy_Hosting",
        desc="Arena hosted the 68th Annual Grammy Awards ceremony on February 1, 2026.",
        parent=group,
        critical=True
    )
    host_sources = _combine_sources(info.grammy_host_urls, info.reference_urls)
    await evaluator.verify(
        claim="This arena hosted the 68th Annual Grammy Awards ceremony on February 1, 2026.",
        node=host_node,
        sources=host_sources,
        additional_instruction="Confirm via authoritative sources (e.g., Grammy.com, major news outlets) that this venue hosted the 68th Annual Grammy Awards in 2026."
    )

    # Artist requirements (same as others)
    await _verify_artist_requirements(evaluator, group, "LA", info)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate an answer for the 4 major indoor concert arenas task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: arenas evaluated independently for partial credit
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

    # Extract structured info for all four arenas
    arenas = await evaluator.extract(
        prompt=prompt_extract_arenas(),
        template_class=ArenasExtraction,
        extraction_name="arenas_extraction"
    )

    # Ensure placeholders for missing arenas
    chicago_info = arenas.chicago or ArenaInfo()
    atlanta_info = arenas.atlanta or ArenaInfo()
    ny_info = arenas.new_york or ArenaInfo()
    la_info = arenas.los_angeles or ArenaInfo()

    # Build and verify subtrees for each arena (independently)
    await verify_chicago(evaluator, root, chicago_info)
    await verify_atlanta(evaluator, root, atlanta_info)
    await verify_new_york(evaluator, root, ny_info)
    await verify_los_angeles(evaluator, root, la_info)

    # Return standardized summary
    return evaluator.get_summary()