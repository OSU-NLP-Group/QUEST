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
TASK_ID = "arena_tour_2026"
TASK_DESCRIPTION = """
A major concert tour is being planned for summer 2026 in the United States. The tour management company needs to identify four suitable arenas for the tour, each located in a different U.S. state. Each arena must meet the following requirements:

1. Seating capacity of at least 18,000 for concerts or basketball games
2. Currently serves as the home venue for at least one NBA or NHL team during the 2024-2025 or 2025-2026 season
3. Has loading dock facilities capable of accommodating semi-trucks for equipment load-in
4. Has backstage dressing room facilities suitable for touring artists
5. Is located in a different U.S. state from the other three selected arenas

Identify four arenas that meet all these requirements. For each arena, provide:
- The arena name
- The city and state where it is located
- The seating capacity for concerts or basketball
- The NBA and/or NHL team(s) that call it home
- A reference URL from an official source (arena website, NBA/NHL team site, or venue management company)
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_value: Optional[str] = None
    teams: List[str] = Field(default_factory=list)
    official_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four arenas mentioned in the answer that are proposed for the summer 2026 U.S. concert tour.
    For each arena, extract the following fields exactly as stated in the answer:
    - name: The arena name
    - city: The city the arena is located in
    - state: The U.S. state the arena is located in (use the full state name if present; otherwise the abbreviation)
    - capacity_value: The seating capacity for concerts or basketball (extract the numeric value or the stated phrase, e.g., "20,000 for basketball")
    - teams: A list of NBA and/or NHL team names that call the arena home
    - official_url: A single reference URL that is from an official source (arena website, NBA or NHL team site, or venue management company). Extract only actual URLs present in the answer. If multiple URLs are given, pick the one most official (prefer arena website; else team site; else venue management company).
    
    Rules:
    - Only extract information explicitly mentioned in the answer.
    - If any field is missing for a venue, set it to null (or empty list for teams).
    - If the answer mentions more than 4 arenas, extract the first 4 in order of appearance.
    - For the official_url, extract only valid URLs with protocol (http/https). If none is mentioned, set to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_valid_url(url: Optional[str]) -> bool:
    if not _non_empty_str(url):
        return False
    u = str(url).strip().lower()
    return u.startswith("http://") or u.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int,
    earlier_states: List[str],
) -> None:
    """
    Build verification sub-tree and run checks for a single venue.
    """
    # Create Venue_i container node (parallel, non-critical)
    venue_node = evaluator.add_parallel(
        id=f"Venue_{venue_index + 1}",
        desc=[
            "First arena identification and verification",
            "Second arena identification and verification",
            "Third arena identification and verification",
            "Fourth arena identification and verification",
        ][venue_index],
        parent=parent_node,
        critical=False,
    )

    # --------------- Info Provided (Non-Critical existence checks) ---------------
    evaluator.add_custom_node(
        result=_non_empty_str(venue.name),
        id=f"Venue_{venue_index + 1}_InfoProvided_Name",
        desc="Arena name is provided",
        parent=venue_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(venue.city) and _non_empty_str(venue.state),
        id=f"Venue_{venue_index + 1}_InfoProvided_CityState",
        desc="City and state location are provided",
        parent=venue_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(venue.capacity_value),
        id=f"Venue_{venue_index + 1}_InfoProvided_CapacityValue",
        desc="Specific seating capacity value is provided",
        parent=venue_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=bool(venue.teams),
        id=f"Venue_{venue_index + 1}_InfoProvided_Teams",
        desc="NBA and/or NHL team names are provided",
        parent=venue_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_has_valid_url(venue.official_url),
        id=f"Venue_{venue_index + 1}_InfoProvided_URL",
        desc="Reference URL from official source is provided",
        parent=venue_node,
        critical=False,
    )

    # --------------- Critical verifications ---------------

    # Capacity >= 18,000 (requires official URL)
    if _has_valid_url(venue.official_url):
        capacity_leaf = evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_Capacity",
            desc="The arena has a seating capacity of at least 18,000 for concerts or basketball games",
            parent=venue_node,
            critical=True,
        )
        capacity_claim = (
            f"The arena '{venue.name or 'the arena'}' has a seating capacity of at least 18,000 "
            f"for concerts or basketball games."
        )
        await evaluator.verify(
            claim=capacity_claim,
            node=capacity_leaf,
            sources=venue.official_url,
            additional_instruction=(
                "Use the official page to confirm that the listed capacity meets or exceeds 18,000 "
                "for concerts or basketball configurations. Accept equivalent wordings (e.g., "
                "basketball capacity 19,000; concert capacity 20,000). Numeric ranges are fine as long "
                "as max or typical capacity ≥ 18,000."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_Capacity",
            desc="The arena has a seating capacity of at least 18,000 for concerts or basketball games",
            parent=venue_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # State verification:
    # - Venue 1: verify location (city, state) with official source
    # - Venue 2/3/4: verify state is different from earlier venues (simple logical check)
    if venue_index == 0:
        if _has_valid_url(venue.official_url) and _non_empty_str(venue.city) and _non_empty_str(venue.state):
            state_leaf = evaluator.add_leaf(
                id="Venue_1_State",
                desc="The arena is located in a specific U.S. state",
                parent=venue_node,
                critical=True,
            )
            state_claim = f"The arena '{venue.name or 'the arena'}' is located in {venue.city}, {venue.state}."
            await evaluator.verify(
                claim=state_claim,
                node=state_leaf,
                sources=venue.official_url,
                additional_instruction=(
                    "Confirm the stated city and U.S. state on the official page. Allow minor formatting variants "
                    "or abbreviations (e.g., 'CA' vs 'California')."
                ),
            )
        else:
            evaluator.add_leaf(
                id="Venue_1_State",
                desc="The arena is located in a specific U.S. state",
                parent=venue_node,
                critical=True,
                score=0.0,
                status="failed",
            )
    else:
        if _non_empty_str(venue.state) and all(_non_empty_str(s) for s in earlier_states):
            state_leaf = evaluator.add_leaf(
                id=f"Venue_{venue_index + 1}_State",
                desc=[
                    "The arena is located in a U.S. state different from Venue 1",
                    "The arena is located in a U.S. state different from Venues 1 and 2",
                    "The arena is located in a U.S. state different from Venues 1, 2, and 3",
                ][venue_index - 1],
                parent=venue_node,
                critical=True,
            )
            if venue_index == 1:
                state_claim = f"The state '{venue.state}' is different from Venue 1's state '{earlier_states[0]}'."
            elif venue_index == 2:
                state_claim = (
                    f"The state '{venue.state}' is different from Venue 1's state '{earlier_states[0]}' "
                    f"and Venue 2's state '{earlier_states[1]}'."
                )
            else:
                state_claim = (
                    f"The state '{venue.state}' is different from Venue 1's state '{earlier_states[0]}', "
                    f"Venue 2's state '{earlier_states[1]}', and Venue 3's state '{earlier_states[2]}'."
                )
            await evaluator.verify(
                claim=state_claim,
                node=state_leaf,
                sources=None,
                additional_instruction=(
                    "This is a pure logical comparison of state strings extracted from the answer. "
                    "Treat standard abbreviations and full names as equivalent (e.g., 'CA' equals 'California'). "
                    "The claim should be marked correct only if the states are genuinely different."
                ),
            )
        else:
            evaluator.add_leaf(
                id=f"Venue_{venue_index + 1}_State",
                desc=[
                    "The arena is located in a U.S. state different from Venue 1",
                    "The arena is located in a U.S. state different from Venues 1 and 2",
                    "The arena is located in a U.S. state different from Venues 1, 2, and 3",
                ][venue_index - 1],
                parent=venue_node,
                critical=True,
                score=0.0,
                status="failed",
            )

    # Home team (NBA or NHL) during 2024-25 or 2025-26 (requires official URL)
    if _has_valid_url(venue.official_url):
        hometeam_leaf = evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_HomeTeam",
            desc="The arena serves as the home venue for at least one NBA or NHL team during the 2024-2025 or 2025-2026 season",
            parent=venue_node,
            critical=True,
        )
        team_list_text = ", ".join(venue.teams) if venue.teams else "at least one NBA or NHL team"
        hometeam_claim = (
            f"The arena '{venue.name or 'the arena'}' serves as the home venue for {team_list_text} "
            f"during the 2024-2025 or 2025-2026 season."
        )
        await evaluator.verify(
            claim=hometeam_claim,
            node=hometeam_leaf,
            sources=venue.official_url,
            additional_instruction=(
                "Verify on the official page or team site that the arena is the home venue of the listed "
                "NBA/NHL team(s) for seasons 2024-25 or 2025-26. Accept explicit statements like 'home arena of' "
                "or official team/arena profile pages indicating home venue."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_HomeTeam",
            desc="The arena serves as the home venue for at least one NBA or NHL team during the 2024-2025 or 2025-2026 season",
            parent=venue_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Loading dock facilities (requires official URL)
    if _has_valid_url(venue.official_url):
        dock_leaf = evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_LoadingDock",
            desc="The arena has loading dock facilities capable of accommodating semi-trucks for equipment load-in",
            parent=venue_node,
            critical=True,
        )
        dock_claim = (
            f"The arena '{venue.name or 'the arena'}' has loading dock facilities capable of accommodating semi-trucks "
            f"for equipment load-in."
        )
        await evaluator.verify(
            claim=dock_claim,
            node=dock_leaf,
            sources=venue.official_url,
            additional_instruction=(
                "Check the venue's event planning guide, production manual, or facility specifications page for mention of "
                "loading docks that accept semi-trucks, trailers, or similar heavy vehicles."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_LoadingDock",
            desc="The arena has loading dock facilities capable of accommodating semi-trucks for equipment load-in",
            parent=venue_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Backstage dressing rooms (requires official URL)
    if _has_valid_url(venue.official_url):
        dressing_leaf = evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_DressingRooms",
            desc="The arena has backstage dressing room facilities suitable for touring artists",
            parent=venue_node,
            critical=True,
        )
        dressing_claim = (
            f"The arena '{venue.name or 'the arena'}' has backstage dressing room facilities suitable for touring artists."
        )
        await evaluator.verify(
            claim=dressing_claim,
            node=dressing_leaf,
            sources=venue.official_url,
            additional_instruction=(
                "Look for references to dressing rooms, green rooms, star rooms, or similar backstage facilities "
                "in venue specs or production guides. These should be suitable for touring artists."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_DressingRooms",
            desc="The arena has backstage dressing room facilities suitable for touring artists",
            parent=venue_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Operational status as of February 2026 (requires official URL)
    if _has_valid_url(venue.official_url):
        operational_leaf = evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_Operational",
            desc="The arena is currently operational and hosting events as of February 2026",
            parent=venue_node,
            critical=True,
        )
        operational_claim = (
            f"As of February 2026, the arena '{venue.name or 'the arena'}' is operational and hosting events."
        )
        await evaluator.verify(
            claim=operational_claim,
            node=operational_leaf,
            sources=venue.official_url,
            additional_instruction=(
                "Confirm recent or upcoming events, ticketing, or schedules around February 2026 on the official page. "
                "Evidence like an event calendar, ticket links, or news updates indicating current operations is sufficient."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_Operational",
            desc="The arena is currently operational and hosting events as of February 2026",
            parent=venue_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Source is official (requires official URL)
    if _has_valid_url(venue.official_url):
        source_leaf = evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_SourceVerification",
            desc="Venue information is verifiable through official sources (arena website, NBA/NHL team site, or venue management company)",
            parent=venue_node,
            critical=True,
        )
        source_claim = (
            f"The provided URL is an official source for information about '{venue.name or 'the arena'}', "
            f"such as an arena website, NBA/NHL team site, or venue management company page."
        )
        await evaluator.verify(
            claim=source_claim,
            node=source_leaf,
            sources=venue.official_url,
            additional_instruction=(
                "Judge whether the site appears official (e.g., owned by the arena, NBA.com, NHL.com, ASM Global, AEG, OVG). "
                "Look for branding, official domain patterns, and About/Contact pages indicating official ownership."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Venue_{venue_index + 1}_SourceVerification",
            desc="Venue information is verifiable through official sources (arena website, NBA/NHL team site, or venue management company)",
            parent=venue_node,
            critical=True,
            score=0.0,
            status="failed",
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
) -> Dict:
    """
    Evaluate an answer for the 2026 arena tour suitability task.
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

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Select first 4 venues, padding with empty entries if fewer provided
    venues: List[VenueItem] = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Build verification trees for each venue
    earlier_states: List[str] = []
    for idx, venue in enumerate(venues):
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=root,
            venue=venue,
            venue_index=idx,
            earlier_states=earlier_states.copy(),
        )
        # Track earlier states for uniqueness checks
        if _non_empty_str(venue.state):
            earlier_states.append(venue.state.strip())

    # Return structured evaluation summary
    return evaluator.get_summary()