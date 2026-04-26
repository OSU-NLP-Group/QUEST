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
TASK_ID = "ca_concert_arenas"
TASK_DESCRIPTION = (
    "Identify four major indoor concert arenas in California that meet all of the following requirements for hosting "
    "a large-scale touring music production:\n\n"
    "Capacity Requirements:\n"
    "- Minimum seating capacity of 15,000 for concert configurations\n"
    "- Arena must be capable of accommodating concert stage setups with floor seating\n\n"
    "Accessibility Requirements:\n"
    "- Wheelchair-accessible seating must be provided at a minimum of 1% of total seating capacity (as required by ADA standards)\n"
    "- Companion seats must be provided adjacent to all wheelchair spaces\n"
    "- ADA-compliant accessible restrooms must be available\n\n"
    "Technical and Safety Requirements:\n"
    "- Arena floor must be capable of supporting concert stage equipment and production loads\n"
    "- Venue must have emergency exits and safety features required for large-capacity events\n\n"
    "For each arena, provide:\n"
    "1. The arena's official name\n"
    "2. Its specific location (city in California)\n"
    "3. Its concert seating capacity\n"
    "4. A reference URL from the arena's official website or a reliable source confirming the capacity and accessibility features"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArenaItem(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # e.g., "CA" or "California"
    concert_capacity_text: Optional[str] = None  # free-form, e.g., "18,000 for concerts"
    concert_capacity_number: Optional[str] = None  # keep string for robustness
    reference_urls: List[str] = Field(default_factory=list)


class ArenasExtraction(BaseModel):
    arenas: List[ArenaItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return (
        "Extract up to six indoor concert arenas mentioned in the answer that are located in California. For each arena, "
        "return the following fields:\n"
        "1. official_name: The arena's official name as stated in the answer.\n"
        "2. city: The city name stated in the answer.\n"
        "3. state: The state as stated (prefer 'CA' or 'California' if present).\n"
        "4. concert_capacity_text: The concert seating capacity description exactly as written in the answer (free-form text).\n"
        "5. concert_capacity_number: If the answer presents a specific numeric concert capacity (e.g., 18000), extract it as a string; otherwise null.\n"
        "6. reference_urls: All URLs explicitly provided in the answer that are associated with this arena. Extract only valid URLs that appear in the answer text (including markdown links). Deduplicate identical URLs.\n\n"
        "Special rules:\n"
        "- Do not invent arenas or URLs; extract only what appears.\n"
        "- If the answer lists more than four arenas, still extract all you find (up to six). The evaluator will handle the first four.\n"
        "- If a field is missing for an arena, set it to null. For URLs, use an empty array if none are given.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def is_ca_string(s: Optional[str]) -> bool:
    if not s:
        return False
    ss = s.strip().lower()
    return ss in {"ca", "california"}


def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


# --------------------------------------------------------------------------- #
# Verification for a single arena                                             #
# --------------------------------------------------------------------------- #
async def verify_arena(
    evaluator: Evaluator,
    parent_node,
    arena: ArenaItem,
    index: int,
) -> None:
    """
    Build verification subtree for a single arena and execute checks.
    """
    ord_txt = ordinal(index + 1)
    arena_node = evaluator.add_parallel(
        id=f"Arena_{index + 1}",
        desc=f"Evaluation of the {ord_txt} identified arena",
        parent=parent_node,
        critical=False
    )

    sources = arena.reference_urls if arena.reference_urls else []

    # Reference evidence gating (critical sequential)
    reference_main = evaluator.add_sequential(
        id=f"Arena_{index + 1}_Reference_Main",
        desc="Reference evidence for capacity and accessibility",
        parent=arena_node,
        critical=True
    )

    ref_exists_leaf = evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"Arena_{index + 1}_Reference_URL_Exists",
        desc="At least one reference URL is provided for this arena.",
        parent=reference_main,
        critical=True
    )

    ref_official_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Reference_URL_Official_or_Reliable",
        desc="At least one provided source is official (arena's site) or generally reliable.",
        parent=reference_main,
        critical=True
    )
    claim_official = (
        "At least one of these sources is from the arena's official website or a generally reliable authority for venue information."
    )
    await evaluator.verify(
        claim=claim_official,
        node=ref_official_leaf,
        sources=sources,
        additional_instruction=(
            "Treat the arena's own domain (official site), government (.gov) pages, and major industry operators "
            "or databases (e.g., Ticketmaster, AXS, ASM Global, Pollstar) as reliable IF the page directly describes the venue. "
            "Random blogs/unknown directories are not reliable."
        ),
    )

    ref_confirms_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Reference_Confirms_Capacity_and_Accessibility",
        desc="Provided sources collectively confirm concert capacity and ADA accessibility features.",
        parent=reference_main,
        critical=True
    )
    claim_confirms = (
        "The provided sources collectively confirm both the concert seating capacity and ADA accessibility features "
        "(wheelchair seating with companion seats and accessible restrooms) for this arena."
    )
    await evaluator.verify(
        claim=claim_confirms,
        node=ref_confirms_leaf,
        sources=sources,
        additional_instruction=(
            "It's acceptable if capacity is confirmed on one official page and accessibility is confirmed on another. "
            "Evaluate the collection of URLs together for explicit support."
        ),
    )

    # Official name provided (critical)
    name_provided = bool(arena.official_name and arena.official_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id=f"Arena_{index + 1}_Official_Name_Provided",
        desc="Arena official name is provided.",
        parent=arena_node,
        critical=True
    )

    # Location provided and verified (critical sequential)
    location_main = evaluator.add_sequential(
        id=f"Arena_{index + 1}_Location_Main",
        desc="Location is provided and verified as a city in California.",
        parent=arena_node,
        critical=True
    )
    loc_provided = bool(arena.city and arena.city.strip()) and is_ca_string(arena.state)
    evaluator.add_custom_node(
        result=loc_provided,
        id=f"Arena_{index + 1}_Location_City_in_CA_Provided",
        desc="Arena location is provided as a city in California.",
        parent=location_main,
        critical=True
    )

    loc_verified_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Location_City_in_CA_Verified",
        desc="Arena location is verified as being in the stated California city.",
        parent=location_main,
        critical=True
    )
    claim_loc = f"The arena '{arena.official_name or 'the arena'}' is located in {arena.city or 'a California city'}, California."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_verified_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the venue's city and state on official or reliable pages. The venue must be in California, and the city must match the claim."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    # Indoor arena (critical)
    indoor_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Is_Indoor_Arena",
        desc="Arena is an indoor venue (indoor arena).",
        parent=arena_node,
        critical=True
    )
    claim_indoor = f"The venue '{arena.official_name or 'the arena'}' is an indoor arena."
    await evaluator.verify(
        claim=claim_indoor,
        node=indoor_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the venue is an indoor arena (roofed/enclosed). Look for explicit mentions or clear context such as hosting NBA/NHL games or indoor concerts."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    # Capacity checks (critical sequential)
    capacity_main = evaluator.add_sequential(
        id=f"Arena_{index + 1}_Capacity_Main",
        desc="Concert capacity is provided and meets the ≥15,000 requirement.",
        parent=arena_node,
        critical=True
    )
    cap_provided = bool(arena.concert_capacity_text and arena.concert_capacity_text.strip())
    evaluator.add_custom_node(
        result=cap_provided,
        id=f"Arena_{index + 1}_Concert_Capacity_Provided",
        desc="Concert seating capacity is provided in the answer.",
        parent=capacity_main,
        critical=True
    )

    cap_gte_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Capacity_GTE_15000",
        desc="Concert seating capacity is at least 15,000.",
        parent=capacity_main,
        critical=True
    )
    claim_cap = (
        f"The concert seating capacity of '{arena.official_name or 'the arena'}' is at least 15,000 (concert configuration)."
    )
    await evaluator.verify(
        claim=claim_cap,
        node=cap_gte_leaf,
        sources=sources,
        additional_instruction=(
            "Use official or reliable venue sources. Accept if the page states concert capacity ≥15,000, or a general capacity ≥15,000 that applies to concert configuration."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    # Floor seating / stage config capability (critical)
    floor_seating_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Floor_Seating_and_Stage_Config_Capable",
        desc="Arena accommodates concert stage setups with floor seating.",
        parent=arena_node,
        critical=True
    )
    claim_floor_seating = (
        f"The arena '{arena.official_name or 'the arena'}' is capable of accommodating concert stage setups with floor seating."
    )
    await evaluator.verify(
        claim=claim_floor_seating,
        node=floor_seating_leaf,
        sources=sources,
        additional_instruction=(
            "Look for seating charts or venue info indicating 'floor seating', 'GA floor', or end-stage configurations for concerts."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    # Accessibility requirements (critical parallel)
    accessibility_main = evaluator.add_parallel(
        id=f"Arena_{index + 1}_Accessibility_Requirements",
        desc="Arena meets the listed ADA accessibility requirements.",
        parent=arena_node,
        critical=True
    )

    wc_1pct_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Wheelchair_Seating_GTE_1pct",
        desc="Wheelchair-accessible seating is at least 1% of total seating capacity.",
        parent=accessibility_main,
        critical=True
    )
    claim_wc = (
        "The venue provides wheelchair-accessible seating at a minimum of 1% of total seating capacity (ADA compliance)."
    )
    await evaluator.verify(
        claim=claim_wc,
        node=wc_1pct_leaf,
        sources=sources,
        additional_instruction=(
            "Pass if the venue explicitly states ADA compliance with required minimums or provides figures showing ≥1% wheelchair seating; fail if no support."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    companion_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Companion_Seats_Adjacent",
        desc="Companion seats are provided adjacent to all wheelchair spaces.",
        parent=accessibility_main,
        critical=True
    )
    claim_companion = (
        "Companion seats are provided adjacent to all wheelchair-accessible seating locations at this venue."
    )
    await evaluator.verify(
        claim=claim_companion,
        node=companion_leaf,
        sources=sources,
        additional_instruction=(
            "Look for ADA/accessibility pages stating companion seating adjacent to wheelchair spaces."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    restrooms_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Accessible_Restrooms_ADA",
        desc="ADA-compliant accessible restrooms are available.",
        parent=accessibility_main,
        critical=True
    )
    claim_restrooms = "ADA-compliant accessible restrooms are available at this venue."
    await evaluator.verify(
        claim=claim_restrooms,
        node=restrooms_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm accessible restrooms via ADA or guest services pages; must explicitly mention accessibility."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    # Technical and Safety requirements (critical parallel)
    tech_safety_main = evaluator.add_parallel(
        id=f"Arena_{index + 1}_Technical_and_Safety_Requirements",
        desc="Arena meets technical and safety requirements for large-scale productions.",
        parent=arena_node,
        critical=True
    )

    floor_load_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Floor_Load_Supports_Production",
        desc="Arena floor supports concert stage equipment and production loads.",
        parent=tech_safety_main,
        critical=True
    )
    claim_floor_load = (
        "The arena floor can support concert stage equipment and production loads typical for large touring shows."
    )
    await evaluator.verify(
        claim=claim_floor_load,
        node=floor_load_leaf,
        sources=sources,
        additional_instruction=(
            "Look for technical specs, rigging guides, or event production information indicating load capacities or production support on the floor."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    exits_leaf = evaluator.add_leaf(
        id=f"Arena_{index + 1}_Emergency_Exits_and_Safety_Features",
        desc="Arena has emergency exits and safety features required for large-capacity events.",
        parent=tech_safety_main,
        critical=True
    )
    claim_exits = (
        "This venue has emergency exits and safety features appropriate for large-capacity events, complying with relevant codes."
    )
    await evaluator.verify(
        claim=claim_exits,
        node=exits_leaf,
        sources=sources,
        additional_instruction=(
            "Evidence may include emergency procedures, safety policies, or venue compliance statements regarding egress and safety systems."
        ),
        extra_prerequisites=[ref_exists_leaf]
    )

    # Final check: The rubric includes a leaf about reference URLs confirming capacity and accessibility.
    # We implemented it above within reference_main (existence + official/reliable + confirmation).


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
    Evaluate an answer for the California concert arenas task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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
        extraction_name="arenas_extraction",
    )

    # Add a top-level node representing the task (non-critical to allow partial credit on arenas)
    top_node = evaluator.add_parallel(
        id="Concert_Arena_Selection_Task",
        desc="Evaluate whether the answer identifies four qualifying California indoor concert arenas and provides the required details and evidence for each.",
        parent=root,
        critical=False
    )

    # Global count and uniqueness check (critical leaf under top)
    names = [normalize_name(a.official_name) for a in extracted.arenas]
    distinct_names = [n for n in names if n]
    unique_count = len(set(distinct_names))
    exact_four = len(extracted.arenas) == 4
    distinct_ok = unique_count == 4 if exact_four else False

    evaluator.add_custom_node(
        result=(exact_four and distinct_ok),
        id="Global_Count_and_Uniqueness_Check",
        desc="The answer lists exactly four arenas and they are distinct (no duplicates).",
        parent=top_node,
        critical=True
    )

    # Record custom info
    evaluator.add_custom_info(
        info={
            "extracted_count": len(extracted.arenas),
            "unique_count_among_extracted": len(set(distinct_names)),
            "used_first_four": True
        },
        info_type="extraction_stats",
        info_name="arena_extraction_stats"
    )

    # Prepare the four arenas to evaluate (pad with empty if fewer than four)
    arenas_to_check: List[ArenaItem] = list(extracted.arenas[:4])
    while len(arenas_to_check) < 4:
        arenas_to_check.append(ArenaItem())

    # Build verification subtrees for each arena
    for idx, arena in enumerate(arenas_to_check):
        await verify_arena(evaluator, top_node, arena, idx)

    # Return structured result using the evaluator's summary
    return evaluator.get_summary()