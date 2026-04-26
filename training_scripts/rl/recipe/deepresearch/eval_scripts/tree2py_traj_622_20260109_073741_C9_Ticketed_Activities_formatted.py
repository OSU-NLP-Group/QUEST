import asyncio
import logging
import re
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_indoor_arenas_multi_purpose"
TASK_DESCRIPTION = (
    "Identify 4 major multi-purpose indoor arenas in the United States that serve as primary venues for professional sports teams. "
    "Each arena must be located in a different US state and meet ALL of the following requirements:\n\n"
    "1. Seating Capacity: Total capacity between 15,000 and 25,000 seats for sporting events\n"
    "2. Professional Sports Tenant: Currently serves as the home venue for at least one NBA team, NHL team, or both\n"
    "3. Facility Age: Opened or underwent major renovation after January 1, 1990\n"
    "4. Premium Seating: Offers luxury suites or club-level seating options\n"
    "5. Corporate Naming Rights: Currently has a corporate sponsor name (not a geographic or historic name)\n"
    "6. ADA Compliance: Provides wheelchair-accessible seating in compliance with ADA standards\n"
    "7. Multi-Purpose Use: Hosts both sporting events and concerts or other entertainment events\n"
    "8. Parking Availability: Has on-site or immediately adjacent parking facilities\n"
    "9. Food and Beverage: Provides concession stands or club dining options\n"
    "10. Modern Ticketing: Supports online ticket sales and mobile/digital ticket entry\n"
    "11. Technical Systems: Equipped with professional-grade sound and lighting systems\n"
    "12. Indoor or Climate-Controlled: Either fully enclosed indoor facility or has retractable roof capability\n"
    "13. Distinct States: All four arenas must be located in four different US states\n\n"
    "For each arena, provide: arena name (with current corporate sponsor name), city and state location, total seating capacity for basketball and/or hockey, "
    "current professional sports team tenant(s) with league designation, year opened or year of most recent major renovation, at least one type of premium seating offered, "
    "and official website URL or authoritative source documenting these details."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArenaTenant(BaseModel):
    team_name: Optional[str] = None
    league: Optional[str] = None  # e.g., "NBA" or "NHL"


class ArenaInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_basketball: Optional[str] = None
    capacity_hockey: Optional[str] = None
    tenants: List[ArenaTenant] = Field(default_factory=list)
    year_open_or_renovation: Optional[str] = None  # Prefer the most recent qualifying year; number only if possible
    premium_seating_examples: List[str] = Field(default_factory=list)  # e.g., ["Luxury suites", "Club level"]
    source_urls: List[str] = Field(default_factory=list)


class ArenasExtraction(BaseModel):
    arenas: List[ArenaInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return """
    Extract up to four multi-purpose indoor arenas listed in the answer that serve as primary venues for professional sports teams in the United States.
    For each arena, return an object containing the following fields extracted EXACTLY from the answer text (do not invent):
    - name: The arena's current official name (should reflect corporate sponsor naming if provided)
    - city: City where the arena is located
    - state: US state where the arena is located (use full state name or standard two-letter abbreviation)
    - capacity_basketball: The seating capacity for basketball (as written, may be a number or text)
    - capacity_hockey: The seating capacity for hockey (as written, may be a number or text). If not mentioned, return null.
    - tenants: The current professional sports team tenant(s). For each, provide:
        * team_name: Full team name (e.g., "Los Angeles Lakers")
        * league: League code (e.g., "NBA" or "NHL")
    - year_open_or_renovation: The year the arena opened OR the year of its most recent major renovation (as written in the answer, prefer the most recent year if multiple are given)
    - premium_seating_examples: At least one example of premium seating mentioned (e.g., "luxury suites", "club level"). If none mentioned, return an empty list.
    - source_urls: A list of official website URLs or other authoritative sources referenced in the answer for this arena. Include only URLs explicitly present in the answer. If none, return an empty list.

    Rules:
    - Extract directly from the answer text; do not add or infer any info.
    - Return 'null' for missing scalar fields; empty lists for missing list fields.
    - If more than four arenas are mentioned, include only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d{2,}", text.replace(",", ""))
    try:
        return int(m.group(0)) if m else None
    except Exception:
        return None


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    return re.sub(r"[^A-Za-z]", "", state).strip().upper()


def build_capacity_claim(arena: ArenaInfo) -> str:
    b = (arena.capacity_basketball or "").strip()
    h = (arena.capacity_hockey or "").strip()
    if b and h:
        return f"The seating capacity for basketball is '{b}' and for hockey is '{h}' at {arena.name}."
    elif b:
        return f"The seating capacity for basketball at {arena.name} is '{b}'."
    elif h:
        return f"The seating capacity for hockey at {arena.name} is '{h}'."
    else:
        return f"The arena {arena.name} has a published seating capacity for sporting events."


def build_tenants_claim(arena: ArenaInfo) -> str:
    if arena.tenants:
        parts = []
        for t in arena.tenants:
            if t.team_name and t.league:
                parts.append(f"{t.team_name} ({t.league})")
            elif t.team_name:
                parts.append(t.team_name)
        if parts:
            return f"{arena.name} currently serves as the home venue for {', '.join(parts)}."
    return f"{arena.name} currently serves as the home venue for at least one NBA or NHL team."


def premium_example(arena: ArenaInfo) -> str:
    return arena.premium_seating_examples[0] if arena.premium_seating_examples else "premium seating (e.g., luxury suites or club level)"


def year_after_1990_claim(arena: ArenaInfo) -> str:
    if arena.year_open_or_renovation and arena.year_open_or_renovation.strip():
        return f"{arena.name} opened or underwent a major renovation in {arena.year_open_or_renovation}, which is after January 1, 1990."
    else:
        return f"{arena.name} opened or underwent a major renovation after January 1, 1990."


# --------------------------------------------------------------------------- #
# Verification per arena                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_arena(
    evaluator: Evaluator,
    parent_node,
    arena: ArenaInfo,
    arena_index: int,
    previous_states: List[str],
) -> None:
    """
    Build verification subtree for a single arena based on the rubric.
    All verification leaves (except distinct-state checks) will depend on the 'source' existence node via extra prerequisites.
    """
    idx = arena_index  # 1-based index

    # Arena group node (parallel aggregation, non-critical to allow partial credit per arena)
    group = evaluator.add_parallel(
        id=f"arena_{idx}",
        desc=(
            f"Arena {idx} (must satisfy all constraints"
            + (", and be in a different US state than previous arenas)" if idx > 1 else ")")
        ),
        parent=parent_node,
        critical=False,
    )

    # Source existence (critical)
    source_exists = bool(arena.source_urls)
    source_node = evaluator.add_custom_node(
        result=source_exists,
        id=f"arena_{idx}_source",
        desc="Provide an official website URL or other authoritative source documenting the required details",
        parent=group,
        critical=True,
    )

    # Name + corporate sponsor (critical)
    name_node = evaluator.add_leaf(
        id=f"arena_{idx}_name_corporate",
        desc="Provide the arena’s current name and ensure it reflects a corporate sponsor naming (i.e., not purely geographic/historic)",
        parent=group,
        critical=True,
    )
    name_claim = f"The current official name of the arena is '{arena.name}', and it reflects corporate sponsorship."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Use official or authoritative sources to confirm the arena's current sponsor-branded name. Reject purely geographic/historic names.",
    )

    # Location (critical)
    loc_node = evaluator.add_leaf(
        id=f"arena_{idx}_location",
        desc="Provide the arena’s city and US state",
        parent=group,
        critical=True,
    )
    location_claim = f"{arena.name} is located in {arena.city}, {arena.state}."
    await evaluator.verify(
        claim=location_claim,
        node=loc_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm the arena's city and US state location from official or authoritative sources.",
    )

    # Distinct state check (critical for arenas 2–4)
    if idx >= 2:
        this_state_norm = normalize_state(arena.state)
        prev_norms = set(s for s in previous_states if s)
        distinct_result = bool(this_state_norm) and (this_state_norm not in prev_norms)
        evaluator.add_custom_node(
            result=distinct_result,
            id=f"arena_{idx}_state_distinct",
            desc=f"Verify arena {idx} is located in a different US state than earlier arenas",
            parent=group,
            critical=True,
        )

    # Capacity numbers (critical)
    cap_numbers_node = evaluator.add_leaf(
        id=f"arena_{idx}_capacity_numbers",
        desc="Provide the arena’s sporting-event seating capacity for basketball and/or hockey (as applicable)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=build_capacity_claim(arena),
        node=cap_numbers_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm the specific seating capacity numbers stated for basketball and/or hockey from the source pages. Minor rounding differences are acceptable.",
    )

    # Capacity range (critical)
    cap_range_node = evaluator.add_leaf(
        id=f"arena_{idx}_capacity_range",
        desc="Verify the arena’s sporting-event seating capacity is between 15,000 and 25,000 seats",
        parent=group,
        critical=True,
    )
    cap_range_claim = f"The sporting-event seating capacity at {arena.name} is between 15,000 and 25,000 seats."
    await evaluator.verify(
        claim=cap_range_claim,
        node=cap_range_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Use the published capacity values on the sources to determine if the capacity lies within [15,000, 25,000].",
    )

    # Tenants with leagues (critical)
    tenants_node = evaluator.add_leaf(
        id=f"arena_{idx}_tenants_with_leagues",
        desc="Identify at least one current home professional sports tenant and specify whether it is NBA, NHL, or both",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=build_tenants_claim(arena),
        node=tenants_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm current home tenant(s) and their league (NBA or NHL) from official or authoritative sources.",
    )

    # Year open or renovation after 1990 (critical)
    year_node = evaluator.add_leaf(
        id=f"arena_{idx}_year_open_or_renovation",
        desc="Provide year opened or most recent major renovation, and verify it is after January 1, 1990",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=year_after_1990_claim(arena),
        node=year_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Validate from sources that the arena opened or had its most recent major renovation after Jan 1, 1990. If multiple years are listed, consider the most recent major renovation.",
    )

    # Premium seating (critical)
    premium_node = evaluator.add_leaf(
        id=f"arena_{idx}_premium_seating",
        desc="Identify at least one premium seating option offered (e.g., luxury suites or club-level seating)",
        parent=group,
        critical=True,
    )
    premium_claim = f"{arena.name} offers premium seating such as {premium_example(arena)}."
    await evaluator.verify(
        claim=premium_claim,
        node=premium_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm the existence of premium seating options (e.g., suites, club level) from official or authoritative sources.",
    )

    # ADA compliance (critical)
    ada_node = evaluator.add_leaf(
        id=f"arena_{idx}_ada",
        desc="Confirm ADA-compliant wheelchair-accessible seating is provided",
        parent=group,
        critical=True,
    )
    ada_claim = f"{arena.name} provides wheelchair-accessible seating in compliance with ADA standards."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm ADA or accessibility statement indicating wheelchair-accessible seating at the arena.",
    )

    # Multipurpose (critical)
    multipurpose_node = evaluator.add_leaf(
        id=f"arena_{idx}_multipurpose",
        desc="Confirm the arena hosts both sporting events and concerts/other entertainment events",
        parent=group,
        critical=True,
    )
    multipurpose_claim = f"{arena.name} hosts both sporting events and concerts or other entertainment events."
    await evaluator.verify(
        claim=multipurpose_claim,
        node=multipurpose_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Use event calendars or venue descriptions to confirm both sports and entertainment events are hosted.",
    )

    # Parking (critical)
    parking_node = evaluator.add_leaf(
        id=f"arena_{idx}_parking",
        desc="Verify on-site or immediately adjacent parking facilities are available",
        parent=group,
        critical=True,
    )
    parking_claim = f"{arena.name} has on-site or immediately adjacent parking facilities."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm parking information (on-site or adjacent) from the venue's official site or authoritative sources.",
    )

    # Food and beverage (critical)
    fb_node = evaluator.add_leaf(
        id=f"arena_{idx}_food_beverage",
        desc="Confirm food and beverage service is provided (concessions and/or club dining)",
        parent=group,
        critical=True,
    )
    fb_claim = f"{arena.name} provides food and beverage services such as concessions and/or club dining."
    await evaluator.verify(
        claim=fb_claim,
        node=fb_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm concessions or dining options offered at the venue from official pages.",
    )

    # Ticketing (critical)
    ticketing_node = evaluator.add_leaf(
        id=f"arena_{idx}_ticketing",
        desc="Verify online ticket sales and mobile/digital ticket entry are supported",
        parent=group,
        critical=True,
    )
    ticketing_claim = f"{arena.name} supports online ticket sales and mobile/digital ticket entry."
    await evaluator.verify(
        claim=ticketing_claim,
        node=ticketing_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm that tickets can be purchased online and that mobile/digital entry is supported.",
    )

    # Technical systems (critical)
    tech_node = evaluator.add_leaf(
        id=f"arena_{idx}_technical_systems",
        desc="Confirm professional-grade sound and lighting systems are available",
        parent=group,
        critical=True,
    )
    tech_claim = f"{arena.name} is equipped with professional-grade sound and lighting systems."
    await evaluator.verify(
        claim=tech_claim,
        node=tech_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Look for statements indicating 'state-of-the-art' or professional sound and lighting capabilities.",
    )

    # Indoor or climate-controlled (critical)
    indoor_node = evaluator.add_leaf(
        id=f"arena_{idx}_indoor_climate",
        desc="Verify the facility is fully enclosed indoor or has retractable-roof climate-control capability",
        parent=group,
        critical=True,
    )
    indoor_claim = f"{arena.name} is a fully enclosed indoor facility or has retractable-roof climate-control capability."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=arena.source_urls,
        extra_prerequisites=[source_node],
        additional_instruction="Confirm that the venue is indoor (enclosed) or provides climate control via a roof mechanism.",
    )

    # Update distinct-state tracking
    norm_state = normalize_state(arena.state)
    if norm_state:
        previous_states.append(norm_state)


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
    Evaluate an agent's answer for the multi-purpose indoor arenas task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation; allow independent partial credit per arena
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

    # Extract arenas from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_arenas(),
        template_class=ArenasExtraction,
        extraction_name="arenas_extracted",
    )

    # Limit to first 4 arenas; pad if fewer
    arenas: List[ArenaInfo] = list(extracted.arenas[:4])
    while len(arenas) < 4:
        arenas.append(ArenaInfo())

    # Track distinct states (normalized) of prior arenas
    prior_states: List[str] = []

    # Build and verify nodes for each arena (1..4)
    for i in range(4):
        await verify_one_arena(
            evaluator=evaluator,
            parent_node=root,
            arena=arenas[i],
            arena_index=i + 1,
            previous_states=prior_states,
        )

    # Return evaluation summary
    return evaluator.get_summary()