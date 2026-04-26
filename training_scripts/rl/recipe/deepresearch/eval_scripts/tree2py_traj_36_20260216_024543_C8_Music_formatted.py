import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_indoor_arenas_billboard_2025"
TASK_DESCRIPTION = """
Identify four distinct U.S. indoor arenas with a concert seating capacity of 15,001 or more that appeared in Billboard's 2025 Year-End Top Venues rankings (15,001+ capacity category). For each arena, provide: (1) the arena name, (2) the city and state where it is located, (3) its documented concert seating capacity, (4) at least one professional sports team that uses it as their home venue, (5) evidence that it hosted concerts during the 2024-2025 season, and (6) a reference URL supporting this information.
"""

SEASON_START_YEAR = 2024
SEASON_END_YEAR = 2025

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArenaEntry(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # keep as string to allow ranges/approximate text
    home_teams: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)
    billboard_urls: List[str] = Field(default_factory=list)
    concert_evidence_urls: List[str] = Field(default_factory=list)


class ArenasExtraction(BaseModel):
    arenas: List[ArenaEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return f"""
    Extract all U.S. indoor arenas mentioned in the answer that claim to have concert seating capacity of 15,001 or more and appeared in Billboard's 2025 Year-End Top Venues rankings (15,001+ capacity category).
    For each arena mentioned in the answer, extract the following fields, exactly as stated in the answer:
    - name: The arena name.
    - city: City where the arena is located.
    - state: U.S. state where the arena is located (full name or postal abbreviation).
    - capacity: The documented concert seating capacity (text as presented, e.g., "18,000", "approx. 19,000", "up to 20,000 for concerts").
    - home_teams: A list of at least one professional sports team that uses the arena as home venue (NBA, NHL, WNBA, etc.) mentioned in the answer.
    - reference_urls: All general reference URLs explicitly provided in the answer for this arena (official site, Wikipedia, team site, venue page, etc.).
    - billboard_urls: Any explicit URLs pointing to Billboard's 2025 Year-End Top Venues rankings or subpages that mention this specific arena and/or the 15,001+ capacity category.
    - concert_evidence_urls: Any explicit URLs that show the arena hosted concerts during {SEASON_START_YEAR}-{SEASON_END_YEAR} (e.g., event calendars, news articles, tour dates pages).

    Rules:
    - Only extract URLs that are explicitly present in the answer. If none are provided for a category, use an empty list.
    - Do not invent or infer any values. If a field is missing in the answer for an arena, set it to null (strings) or [] (lists).
    - Return all arenas the answer mentions; we will consider only the first four for evaluation.
    - Normalize city/state spelling minimally (keep as presented), and keep capacity as string.

    Output:
    Return a JSON object with a single key "arenas" that is an array of ArenaEntry objects as described.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch.isspace())


def _combine_sources(arena: ArenaEntry) -> List[str]:
    """Combine all available sources for robust verification."""
    urls = []
    urls.extend(arena.reference_urls or [])
    urls.extend(arena.billboard_urls or [])
    urls.extend(arena.concert_evidence_urls or [])
    # Deduplicate while preserving order
    seen = set()
    combined = []
    for u in urls:
        if u and u not in seen:
            combined.append(u)
            seen.add(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_arena(
    evaluator: Evaluator,
    parent_node,
    arena: ArenaEntry,
    index: int
) -> None:
    """
    Build verification sub-tree and run checks for a single arena.
    """
    # Map index to human-readable ordinal for description
    ord_map = {0: "First", 1: "Second", 2: "Third", 3: "Fourth"}
    ord_text = ord_map.get(index, f"Arena #{index + 1}")

    # Parent node for the arena
    arena_node = evaluator.add_parallel(
        id=f"arena_{index + 1}",
        desc=f"{ord_text} qualifying U.S. arena meeting all criteria",
        parent=parent_node,
        critical=False  # allow partial credit per arena
    )

    all_sources = _combine_sources(arena)

    # 1) Arena name provided (existence check)
    evaluator.add_custom_node(
        result=bool(arena.name and arena.name.strip()),
        id=f"arena_{index + 1}_name",
        desc="Arena name is provided",
        parent=arena_node,
        critical=True
    )

    # 2) City location correctly identified (verify with sources)
    city_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_city",
        desc="City location is correctly identified",
        parent=arena_node,
        critical=True
    )
    city_claim = f"The arena named '{arena.name or ''}' is located in the city of '{arena.city or ''}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=all_sources,
        additional_instruction="Verify that the referenced pages explicitly indicate the arena's city. Minor spelling variations are acceptable."
    )

    # 3) State location correctly identified (verify with sources)
    state_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_state",
        desc="U.S. state location is correctly identified",
        parent=arena_node,
        critical=True
    )
    state_claim = f"The arena named '{arena.name or ''}' is located in the U.S. state of '{arena.state or ''}'."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=all_sources,
        additional_instruction="Verify that the referenced pages explicitly indicate the arena's state (U.S.). State abbreviations are acceptable."
    )

    # 4) Indoor type verification
    indoor_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_indoor_type",
        desc="Arena is verified as an indoor venue (not an outdoor stadium or amphitheater)",
        parent=arena_node,
        critical=True
    )
    indoor_claim = f"The venue '{arena.name or ''}' is an indoor arena (i.e., an enclosed multipurpose arena), not an outdoor stadium or amphitheater."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=all_sources,
        additional_instruction="Check the venue description. It should be an indoor arena (enclosed building), not an outdoor stadium/amphitheater."
    )

    # 5) Capacity verification: documented concert seating capacity is 15,001+
    capacity_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_capacity",
        desc="Documented seating capacity for concerts is 15,001 or higher",
        parent=arena_node,
        critical=True
    )
    capacity_text = arena.capacity or ""
    capacity_claim = (
        f"The documented concert seating capacity of '{arena.name or ''}' is '{capacity_text}', "
        f"and it is at least 15,001."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=all_sources,
        additional_instruction="Interpret 'concert capacity' or end-stage seating. Approximate or range values are acceptable if clearly ≥ 15,001."
    )

    # 6) Billboard ranking verification (2025 Year-End Top Venues, 15,001+ capacity)
    billboard_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_billboard_ranking",
        desc="Arena appeared in Billboard's 2025 Year-End Top Venues (15,001+ capacity) rankings",
        parent=arena_node,
        critical=True
    )
    billboard_claim = (
        f"The arena '{arena.name or ''}' appeared in Billboard's 2025 Year-End Top Venues rankings "
        f"in the 15,001+ capacity category."
    )
    billboard_sources = arena.billboard_urls if arena.billboard_urls else all_sources
    await evaluator.verify(
        claim=billboard_claim,
        node=billboard_leaf,
        sources=billboard_sources,
        additional_instruction="Confirm the arena is listed on Billboard's 2025 Year-End Top Venues, 15,001+ capacity category pages."
    )

    # 7) Home team identification (non-critical, but still source-verified)
    home_team_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_home_team",
        desc="At least one professional sports team that uses this arena as home venue is identified (where applicable)",
        parent=arena_node,
        critical=False
    )
    # Use at least one team if provided, else empty indicates likely fail
    first_team = arena.home_teams[0] if arena.home_teams else ""
    home_team_claim = (
        f"At least one professional sports team, such as '{first_team}', uses the arena '{arena.name or ''}' as its home venue."
    )
    await evaluator.verify(
        claim=home_team_claim,
        node=home_team_leaf,
        sources=all_sources,
        additional_instruction="Confirm that at least one pro sports team (NBA, NHL, WNBA, etc.) uses this arena as home venue."
    )

    # 8) Concert hosting evidence during 2024-2025 season
    concert_evidence_leaf = evaluator.add_leaf(
        id=f"arena_{index + 1}_concert_evidence",
        desc="Evidence of hosting concerts during 2024-2025 season is provided",
        parent=arena_node,
        critical=True
    )
    concert_claim = (
        f"The arena '{arena.name or ''}' hosted concerts during the {SEASON_START_YEAR}-{SEASON_END_YEAR} season."
    )
    concert_sources = arena.concert_evidence_urls if arena.concert_evidence_urls else all_sources
    await evaluator.verify(
        claim=concert_claim,
        node=concert_evidence_leaf,
        sources=concert_sources,
        additional_instruction=f"Look for event pages, schedules, or news confirming concerts took place in {SEASON_START_YEAR} or {SEASON_END_YEAR}."
    )

    # 9) Reference URL existence (critical existence check)
    evaluator.add_custom_node(
        result=bool(all_sources),
        id=f"arena_{index + 1}_reference_url",
        desc="Reference URL supporting the arena's attributes is provided",
        parent=arena_node,
        critical=True
    )


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
    Evaluate an answer for the U.S. indoor arenas Billboard 2025 task.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify and verify four distinct U.S. indoor arenas (15,001+ capacity) appearing in Billboard's 2025 Year-End Top Venues rankings",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract arenas data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_arenas(),
        template_class=ArenasExtraction,
        extraction_name="arenas_extraction"
    )

    # Keep only the first four arenas for evaluation; pad with empty entries if fewer provided
    arenas = list(extracted.arenas[:4])
    while len(arenas) < 4:
        arenas.append(ArenaEntry())

    # Add distinctness check (critical)
    names = [_normalize_name(a.name) for a in arenas]
    non_empty_names = [n for n in names if n]
    all_distinct = len(non_empty_names) == 4 and len(set(non_empty_names)) == 4
    evaluator.add_custom_node(
        result=all_distinct,
        id="all_arenas_distinct",
        desc="All four arenas are distinct from each other (no duplicates)",
        parent=root,
        critical=True
    )

    # Build verification branches for each arena
    for idx, arena in enumerate(arenas):
        await verify_arena(evaluator, root, arena, idx)

    # Return the evaluation summary
    return evaluator.get_summary()