import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_indoor_arenas_modern"
TASK_DESCRIPTION = """
A major touring production company is planning a North American arena tour and needs to identify suitable venues that meet their modern facility requirements. Identify exactly four major indoor arenas in the United States that meet all of the following criteria:

1. Each arena must be located in a different U.S. state
2. Each arena must have a basketball seating capacity of at least 18,000
3. Each arena must have either opened or undergone a major renovation in or after 2012
4. Each arena must currently serve as the home venue for at least one major professional sports team (NBA or NHL)

For each arena, provide:
- The arena's official name
- The state where it is located
- The basketball seating capacity
- The year it opened or underwent major renovation (if both apply, specify which is 2012 or later)
- The name of at least one professional sports team (NBA or NHL) that calls it home
- Reference URLs supporting each piece of information
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArenaItem(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    state: Optional[str] = None
    state_sources: List[str] = Field(default_factory=list)

    basketball_capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    year_kind: Optional[str] = None  # "opening" or "renovation"
    year_value: Optional[str] = None
    year_sources: List[str] = Field(default_factory=list)

    team: Optional[str] = None  # At least one current NBA or NHL team
    team_sources: List[str] = Field(default_factory=list)

    classification: Optional[str] = None  # e.g., "indoor arena", "multi-purpose arena"
    classification_sources: List[str] = Field(default_factory=list)


class ArenasExtraction(BaseModel):
    arenas: List[ArenaItem] = Field(default_factory=list)
    total_arenas_mentioned: Optional[int] = None  # Count of arenas mentioned in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return """
    Extract all U.S. indoor arenas explicitly mentioned in the answer. For each arena, return the following fields exactly as stated in the answer:

    - name: The official arena name
    - name_sources: URLs that support the arena's official name
    - state: The U.S. state where the arena is located (use the state name, not abbreviation; if only city/state is provided, use the state)
    - state_sources: URLs that support the arena's location/state
    - basketball_capacity: The basketball seating capacity value (as presented; prefer a single numeric string like "19,000" or "19000" specifically for basketball configuration)
    - capacity_sources: URLs that support the basketball seating capacity value
    - year_kind: Either "opening" or "renovation" — the event used to satisfy the 2012-or-later constraint
    - year_value: The year associated with year_kind (prefer a 4-digit year, e.g., "2018")
    - year_sources: URLs that support this opening/renovation year
    - team: The name of at least one current major professional sports team (NBA or NHL) that calls this arena home
    - team_sources: URLs that support the team-home relationship
    - classification: A phrase indicating the venue type (e.g., "indoor arena", "multi-purpose arena")
    - classification_sources: URLs that support the venue classification as an indoor multi-purpose arena

    Additionally, return:
    - total_arenas_mentioned: the total number of distinct arenas mentioned in the answer (count all, even if more than 4)

    Rules:
    - Only extract arenas explicitly mentioned in the answer.
    - Extract URLs exactly as presented (plain URLs or markdown links). If a field lacks any URL, return an empty list for that field's sources.
    - Do not invent data; if a field is missing, set it to null.
    - Prefer the basketball-specific capacity if multiple capacities are listed; if none are labeled, use the value the answer associates with basketball.
    - For year_kind, choose the one (opening or renovation) that is 2012 or later if both exist and are mentioned; otherwise choose the one the answer uses to claim compliance.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_first_int(text: Optional[str]) -> Optional[int]:
    """Extract the first reasonable integer from a text string (handles commas)."""
    if not text:
        return None
    # Prefer 4-digit year if present when year context, else any integer
    # General fallback: remove commas and get first integer
    cleaned = re.sub(r"[^\d]", " ", text)
    nums = re.findall(r"\d{4}|\d{2,}", cleaned)
    if not nums:
        return None
    try:
        return int(nums[0])
    except Exception:
        return None


def _normalize_kind(kind: Optional[str]) -> Optional[str]:
    if not kind:
        return None
    k = kind.strip().lower()
    if "open" in k:
        return "opening"
    if "renov" in k or "moderniz" in k or "upgrade" in k:
        return "renovation"
    return k


def _has_non_empty_sources(urls: List[str]) -> bool:
    return isinstance(urls, list) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_arena(
    evaluator: Evaluator,
    parent_node,
    arena: ArenaItem,
    arena_index: int,
) -> None:
    """
    Verify all per-arena requirements using leaf checks.
    """
    # Create arena node (parallel aggregation, non-critical to allow partial credit per arena)
    arena_node = evaluator.add_parallel(
        id=f"arena_{arena_index + 1}",
        desc=f"Arena #{arena_index + 1} (must satisfy all per-arena constraints; provide all required fields with URLs).",
        parent=parent_node,
        critical=False,
    )

    # 1) Official name + source
    # Existence check (critical)
    name_exists_node = evaluator.add_custom_node(
        result=(arena.name is not None and arena.name.strip() != "" and _has_non_empty_sources(arena.name_sources)),
        id=f"arena_{arena_index + 1}_name_exists",
        desc="Provide the arena's official name AND at least one URL that supports the name.",
        parent=arena_node,
        critical=True,
    )
    # Source-supported verification leaf (critical)
    name_verify_node = evaluator.add_leaf(
        id=f"arena_{arena_index + 1}_name_and_source",
        desc="The official arena name is supported by the cited source(s).",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the arena is '{arena.name}'.",
        node=name_verify_node,
        sources=arena.name_sources,
        additional_instruction="Verify that the cited source(s) explicitly present the official name of the arena. Allow minor formatting variations.",
    )

    # 2) State + source
    state_exists_node = evaluator.add_custom_node(
        result=(arena.state is not None and arena.state.strip() != "" and _has_non_empty_sources(arena.state_sources)),
        id=f"arena_{arena_index + 1}_state_exists",
        desc="Provide the U.S. state where the arena is located AND at least one URL that supports the state/location.",
        parent=arena_node,
        critical=True,
    )
    state_verify_node = evaluator.add_leaf(
        id=f"arena_{arena_index + 1}_state_and_source",
        desc="The arena's state location is supported by the cited source(s).",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The arena is located in the state of {arena.state}.",
        node=state_verify_node,
        sources=arena.state_sources,
        additional_instruction="Consider it supported if the source shows the arena's location with city and state that matches the given state.",
    )

    # 3) Basketball capacity + source + threshold >= 18,000
    cap_exists_node = evaluator.add_custom_node(
        result=(arena.basketball_capacity is not None and arena.basketball_capacity.strip() != "" and _has_non_empty_sources(arena.capacity_sources)),
        id=f"arena_{arena_index + 1}_capacity_exists",
        desc="Provide the basketball seating capacity value AND at least one URL supporting it.",
        parent=arena_node,
        critical=True,
    )
    cap_verify_node = evaluator.add_leaf(
        id=f"arena_{arena_index + 1}_basketball_capacity_and_source",
        desc="Basketball seating capacity value is supported by the cited source(s).",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The arena has a basketball seating capacity of {arena.basketball_capacity}.",
        node=cap_verify_node,
        sources=arena.capacity_sources,
        additional_instruction="Verify the basketball-specific capacity. Allow minor formatting variations (commas, rounding).",
    )
    # Threshold custom check (critical)
    parsed_capacity = _parse_first_int(arena.basketball_capacity)
    cap_threshold_node = evaluator.add_custom_node(
        result=(parsed_capacity is not None and parsed_capacity >= 18000),
        id=f"arena_{arena_index + 1}_capacity_threshold",
        desc=f"Basketball seating capacity ({arena.basketball_capacity}) is at least 18,000.",
        parent=arena_node,
        critical=True,
    )

    # 4) Opening or major renovation year (>= 2012) + source
    ynorm = _normalize_kind(arena.year_kind)
    year_exists_node = evaluator.add_custom_node(
        result=(arena.year_value is not None and arena.year_value.strip() != "" and ynorm in ("opening", "renovation") and _has_non_empty_sources(arena.year_sources)),
        id=f"arena_{arena_index + 1}_year_exists",
        desc="Provide the year the arena opened or underwent a major renovation AND a supporting URL.",
        parent=arena_node,
        critical=True,
    )
    year_verify_node = evaluator.add_leaf(
        id=f"arena_{arena_index + 1}_opened_or_renovated_year_and_source",
        desc="Opening/major renovation year is supported by the cited source(s).",
        parent=arena_node,
        critical=True,
    )
    # Construct claim based on kind
    kind_phrase = "opened" if ynorm == "opening" else "underwent a major renovation"
    await evaluator.verify(
        claim=f"The arena {kind_phrase} in {arena.year_value}.",
        node=year_verify_node,
        sources=arena.year_sources,
        additional_instruction="Verify that the cited source explicitly states the given year and whether it is the opening year or a major renovation year.",
    )
    # Threshold >= 2012 custom check (critical)
    parsed_year = _parse_first_int(arena.year_value)
    year_threshold_node = evaluator.add_custom_node(
        result=(parsed_year is not None and parsed_year >= 2012),
        id=f"arena_{arena_index + 1}_year_threshold",
        desc=f"The {ynorm or 'year'} year ({arena.year_value}) is in or after 2012.",
        parent=arena_node,
        critical=True,
    )

    # 5) Current pro team (NBA or NHL) + source
    team_exists_node = evaluator.add_custom_node(
        result=(arena.team is not None and arena.team.strip() != "" and _has_non_empty_sources(arena.team_sources)),
        id=f"arena_{arena_index + 1}_team_exists",
        desc="Provide at least one current home team (NBA or NHL) AND at least one URL supporting the team-home relationship.",
        parent=arena_node,
        critical=True,
    )
    team_verify_node = evaluator.add_leaf(
        id=f"arena_{arena_index + 1}_pro_team_home_and_source",
        desc="The cited source(s) support that the team is a current NBA or NHL home team at the arena.",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{arena.team}' is a current home team (NBA or NHL) that plays its home games at the arena.",
        node=team_verify_node,
        sources=arena.team_sources,
        additional_instruction="Verify that the source(s) confirm the team plays home games at the arena and is an NBA or NHL team.",
    )

    # 6) Indoor multi-purpose classification + source
    class_exists_node = evaluator.add_custom_node(
        result=(arena.classification is not None and arena.classification.strip() != "" and _has_non_empty_sources(arena.classification_sources)),
        id=f"arena_{arena_index + 1}_classification_exists",
        desc="Confirm the venue is a multi-purpose indoor arena AND provide at least one URL supporting this classification.",
        parent=arena_node,
        critical=True,
    )
    class_verify_node = evaluator.add_leaf(
        id=f"arena_{arena_index + 1}_indoor_multipurpose_and_source",
        desc="The cited source(s) support that the venue is a multi-purpose indoor arena.",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The arena is an indoor multi-purpose arena.",
        node=class_verify_node,
        sources=arena.classification_sources,
        additional_instruction="Consider synonymous phrases such as 'indoor arena', 'multi-purpose venue', 'multi-use arena' as equivalent.",
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
    Evaluate an answer for the modern U.S. indoor arenas task.
    """
    # Initialize evaluator (root node non-critical parallel aggregation)
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

    # Extract all arenas mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_arenas(),
        template_class=ArenasExtraction,
        extraction_name="arenas_extraction",
    )

    total_mentioned = extracted.total_arenas_mentioned if extracted.total_arenas_mentioned is not None else len(extracted.arenas)

    # Select exactly 4 arenas (first 4 if more were provided); pad with empty items if fewer
    selected_arenas: List[ArenaItem] = list(extracted.arenas[:4])
    while len(selected_arenas) < 4:
        selected_arenas.append(ArenaItem())

    evaluator.add_custom_info(
        info={
            "total_arenas_mentioned_in_answer": total_mentioned,
            "selected_arenas_count": len(selected_arenas),
            "selected_arenas_preview": [a.name for a in selected_arenas],
        },
        info_type="selected_arenas_info",
    )

    # Build per-arena verification nodes
    for idx in range(4):
        await verify_single_arena(evaluator, root, selected_arenas[idx], idx)

    # Cross-arena constraints node under root (critical to gate overall success)
    cross_node = evaluator.add_parallel(
        id="cross_arena_constraints",
        desc="Constraints that apply to the set of arenas as a whole.",
        parent=root,
        critical=True,
    )

    # Exactly four arenas
    exactly_four_node = evaluator.add_custom_node(
        result=(total_mentioned == 4),
        id="exactly_four_arenas",
        desc="The response identifies exactly four arenas (no more, no fewer).",
        parent=cross_node,
        critical=True,
    )

    # All states distinct (based on the selected four)
    states = [a.state.strip() if a.state else None for a in selected_arenas]
    states_valid = all(s is not None and s != "" for s in states)
    distinct_states = len({s for s in states if s}) == 4 if states_valid else False

    distinct_states_node = evaluator.add_custom_node(
        result=distinct_states,
        id="all_states_distinct",
        desc="All four arenas are located in four different U.S. states (state values are pairwise distinct).",
        parent=cross_node,
        critical=True,
    )

    # Return structured evaluation summary
    return evaluator.get_summary()