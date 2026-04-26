import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_2025_superbowl_lx"
TASK_DESCRIPTION = (
    "On November 27, 2025, three NFL stadiums hosted Thanksgiving Day games with major halftime show performances. "
    "For each of these three stadiums, provide the following information: (1) the official stadium name, "
    "(2) the city where it is located, (3) the state where it is located, (4) the stadium's seating capacity, "
    "(5) the name of the main halftime show performer, and (6) the name of any special guest performer who appeared "
    "(if applicable). After identifying all three stadiums, determine which of the three has the largest seating capacity. "
    "Additionally, identify the venue that will host Super Bowl LX in 2026, and provide: (1) the venue name, "
    "(2) the city where it is located, (3) the state where it is located, (4) the venue's seating capacity, "
    "(5) the date of Super Bowl LX, (6) the name of the halftime show headliner, and (7) the corporate sponsor of the halftime show."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ThanksgivingStadiumItem(BaseModel):
    stadium_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    halftime_main: Optional[str] = None
    halftime_guest: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ThanksgivingExtraction(BaseModel):
    stadiums: List[ThanksgivingStadiumItem] = Field(default_factory=list)
    largest_capacity_stadium: Optional[str] = None


class SuperBowlLXExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    date: Optional[str] = None
    halftime_headliner: Optional[str] = None
    halftime_sponsor: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving() -> str:
    return (
        "Extract up to three NFL stadium entries from the answer that correspond to Thanksgiving Day games on "
        "November 27, 2025 and include major halftime show performances. For each stadium entry, return the fields:\n"
        "1) stadium_name: official stadium name (string)\n"
        "2) city: city where the stadium is located (string)\n"
        "3) state: state where the stadium is located (string)\n"
        "4) seating_capacity: the stadium seating capacity (string, as stated; do not convert to number)\n"
        "5) halftime_main: main halftime show performer name (string)\n"
        "6) halftime_guest: special guest performer name(s) if any appeared; otherwise set to 'None' or leave null\n"
        "7) sources: an array of URL strings cited in the answer that support the stadium, game, location, capacity, and/or halftime details\n"
        "Also, if the answer explicitly identifies which of these three stadiums has the largest seating capacity, extract that into 'largest_capacity_stadium'. "
        "If the answer lists more than three stadiums, include only the first three relevant entries. If fewer are provided, return as many as available."
    )


def prompt_extract_superbowl_lx() -> str:
    return (
        "Extract the Super Bowl LX (2026) host venue and event details from the answer. Return the fields:\n"
        "1) venue_name: official host venue name (string)\n"
        "2) city: the city where the venue is located (string)\n"
        "3) state: the state where the venue is located (string)\n"
        "4) seating_capacity: the venue seating capacity (string, as stated; do not convert to number)\n"
        "5) date: the date of Super Bowl LX (string as presented, e.g., 'February 8, 2026')\n"
        "6) halftime_headliner: the halftime show headliner (string)\n"
        "7) halftime_sponsor: the corporate sponsor of the halftime show (string)\n"
        "8) sources: an array of URL strings cited in the answer that support these details\n"
        "If any field is missing in the answer, set it to null."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def take_first_three_stadiums(extraction: ThanksgivingExtraction) -> List[ThanksgivingStadiumItem]:
    # Ensure we only handle at most three stadiums; pad with empty items if fewer.
    items = extraction.stadiums[:3]
    while len(items) < 3:
        items.append(ThanksgivingStadiumItem())
    return items


def is_none_like(text: Optional[str]) -> bool:
    if not text:
        return True
    normalized = text.strip().lower()
    return normalized in {"none", "n/a", "na", "not applicable", "no", "no special guest", "none noted", "none listed"}


def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    if not cap_str:
        return None
    digits = "".join(ch for ch in cap_str if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_thanksgiving_block(
    evaluator: Evaluator,
    parent_task_node,
    tgk: ThanksgivingExtraction,
) -> None:
    # Thanksgiving_2025 node (sequential, critical)
    thanksgiving_node = evaluator.add_sequential(
        id="Thanksgiving_2025",
        desc="Identify the three NFL stadiums that hosted Thanksgiving Day games on Nov 27, 2025 with major halftime performances; provide required attributes for each; identify which has the largest seating capacity.",
        parent=parent_task_node,
        critical=True,
    )

    # Child: Thanksgiving_Stadiums_Set (parallel, critical)
    stadiums_set_node = evaluator.add_parallel(
        id="Thanksgiving_Stadiums_Set",
        desc="Provide exactly three distinct Thanksgiving 2025 stadium venues and required attributes for each.",
        parent=thanksgiving_node,
        critical=True,
    )

    # Prepare the 3 stadiums
    stadium_items = take_first_three_stadiums(tgk)

    # Stadium_Count_And_Distinctness (leaf/custom as critical)
    names = [s.stadium_name.strip() for s in stadium_items if s.stadium_name and s.stadium_name.strip()]
    count_ok = len([s for s in stadium_items if s.stadium_name and s.stadium_name.strip()]) == 3
    distinct_ok = len(set(names)) == 3

    evaluator.add_custom_node(
        result=(count_ok and distinct_ok),
        id="Stadium_Count_And_Distinctness",
        desc="Lists exactly 3 distinct NFL stadiums that hosted Thanksgiving Day games on Nov 27, 2025 (no duplicates).",
        parent=stadiums_set_node,
        critical=True,
    )

    # For each stadium i: parallel node (critical to satisfy framework constraints)
    for i, s in enumerate(stadium_items, start=1):
        st_node = evaluator.add_parallel(
            id=f"Thanksgiving_Stadium_{i}",
            desc=f"Thanksgiving 2025 stadium #{i} entry with all required attributes, and the stadium must be a correct venue for a Nov 27, 2025 Thanksgiving Day game with a major halftime performance.",
            parent=stadiums_set_node,
            critical=True,
        )

        # 1) Official_Stadium_Name_Correct
        official_leaf = evaluator.add_leaf(
            id=f"stadium_{i}_Official_Stadium_Name_Correct",
            desc="Provides the official stadium name and it is correct for one of the Nov 27, 2025 Thanksgiving Day game venues.",
            parent=st_node,
            critical=True,
        )
        official_claim = (
            f"The official stadium name is '{s.stadium_name}', and it hosted an NFL Thanksgiving Day game on November 27, 2025."
        )
        await evaluator.verify(
            claim=official_claim,
            node=official_leaf,
            sources=s.sources,
            additional_instruction="Confirm the page(s) support that this official stadium name is correct and that it hosted a Thanksgiving Day NFL game on Nov 27, 2025.",
        )

        # 2) City_Correct
        city_leaf = evaluator.add_leaf(
            id=f"stadium_{i}_City_Correct",
            desc="Provides the correct city where this stadium is located.",
            parent=st_node,
            critical=True,
        )
        city_claim = f"The stadium '{s.stadium_name}' is located in the city of {s.city}."
        await evaluator.verify(
            claim=city_claim,
            node=city_leaf,
            sources=s.sources,
            additional_instruction="Verify that the stadium's city matches what is shown on authoritative sources (e.g., official venue/team pages, reliable outlets).",
        )

        # 3) State_Correct
        state_leaf = evaluator.add_leaf(
            id=f"stadium_{i}_State_Correct",
            desc="Provides the correct state where this stadium is located.",
            parent=st_node,
            critical=True,
        )
        state_claim = f"The stadium '{s.stadium_name}' is located in the state of {s.state}."
        await evaluator.verify(
            claim=state_claim,
            node=state_leaf,
            sources=s.sources,
            additional_instruction="Verify that the stadium's state matches what is shown on authoritative sources.",
        )

        # 4) Seating_Capacity_Correct
        capacity_leaf = evaluator.add_leaf(
            id=f"stadium_{i}_Seating_Capacity_Correct",
            desc="Provides the stadium seating capacity (numeric) and it is correct.",
            parent=st_node,
            critical=True,
        )
        capacity_claim = f"The seating capacity of '{s.stadium_name}' is {s.seating_capacity}."
        await evaluator.verify(
            claim=capacity_claim,
            node=capacity_leaf,
            sources=s.sources,
            additional_instruction="Confirm the stated seating capacity on authoritative sources. Allow small variations (e.g., event configuration).",
        )

        # 5) Main_Halftime_Performer_Correct
        headliner_leaf = evaluator.add_leaf(
            id=f"stadium_{i}_Main_Halftime_Performer_Correct",
            desc="Provides the correct name of the main halftime show performer for this game.",
            parent=st_node,
            critical=True,
        )
        headliner_claim = (
            f"The main halftime show performer for the Thanksgiving Day game at '{s.stadium_name}' on November 27, 2025 was {s.halftime_main}."
        )
        await evaluator.verify(
            claim=headliner_claim,
            node=headliner_leaf,
            sources=s.sources,
            additional_instruction="Verify the halftime headliner from credible game recaps, official team announcements, or reputable media.",
        )

        # 6) Special_Guest_Correct_Or_None
        guest_leaf = evaluator.add_leaf(
            id=f"stadium_{i}_Special_Guest_Correct_Or_None",
            desc="Provides the correct special guest performer name(s) if any appeared; otherwise explicitly indicates none/not applicable.",
            parent=st_node,
            critical=True,
        )
        if is_none_like(s.halftime_guest):
            guest_claim = (
                f"There were no special guest performers beyond the main headliner for the Thanksgiving Day game at '{s.stadium_name}' on November 27, 2025."
            )
            add_ins = "If sources mention any guest performers, the claim should be considered not supported."
        else:
            guest_claim = (
                f"The special guest performer(s) for the Thanksgiving Day game at '{s.stadium_name}' on November 27, 2025 included {s.halftime_guest}."
            )
            add_ins = "Verify the presence and naming of special guest performer(s) from credible sources."
        await evaluator.verify(
            claim=guest_claim,
            node=guest_leaf,
            sources=s.sources,
            additional_instruction=add_ins,
        )

    # Largest_Thanksgiving_Capacity (leaf under Thanksgiving_2025)
    largest_leaf = evaluator.add_leaf(
        id="Largest_Thanksgiving_Capacity",
        desc="Correctly identifies which of the three identified Thanksgiving stadiums has the largest seating capacity, consistent with the capacities provided.",
        parent=thanksgiving_node,
        critical=True,
    )

    # Construct a consistency claim using the extracted capacities
    cap_info = []
    for s in stadium_items:
        nm = s.stadium_name or "Unknown Stadium"
        cp = s.seating_capacity or "Unknown Capacity"
        cap_info.append((nm, cp))

    # Use the answer’s identified largest if provided; otherwise compute from capacities
    if tgk.largest_capacity_stadium:
        largest_choice = tgk.largest_capacity_stadium
    else:
        # Fallback: compute from numeric parsing
        best_name = None
        best_val = -1
        for nm, cp in cap_info:
            val = parse_capacity_to_int(cp) or -1
            if val > best_val:
                best_val = val
                best_name = nm
        largest_choice = best_name or (cap_info[0][0] if cap_info else "Unknown Stadium")

    # Build claim listing capacities for all three and the chosen largest
    cap_list_str = "; ".join([f"{nm}: {cp}" for nm, cp in cap_info])
    largest_claim = (
        f"Among the three Thanksgiving stadiums with capacities [{cap_list_str}], the one with the largest seating capacity is '{largest_choice}'."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        additional_instruction="Judge based on the capacities listed in the claim; ensure the named stadium is indeed the maximum given those values.",
    )


async def verify_super_bowl_lx_block(
    evaluator: Evaluator,
    parent_task_node,
    sb: SuperBowlLXExtraction,
) -> None:
    # Super_Bowl_LX node (parallel, critical)
    sb_node = evaluator.add_parallel(
        id="Super_Bowl_LX",
        desc="Identify the Super Bowl LX (2026) host venue and provide all requested venue/event details.",
        parent=parent_task_node,
        critical=True,
    )

    # Use the same sources for all sub-claims
    sb_sources = sb.sources

    # Venue_Name_Correct
    vname_leaf = evaluator.add_leaf(
        id="Venue_Name_Correct",
        desc="Provides the correct Super Bowl LX host venue name.",
        parent=sb_node,
        critical=True,
    )
    vname_claim = f"The host venue for Super Bowl LX is '{sb.venue_name}'."
    await evaluator.verify(
        claim=vname_claim,
        node=vname_leaf,
        sources=sb_sources,
        additional_instruction="Confirm the official Super Bowl LX host venue name from authoritative sources (NFL, venue, reputable media).",
    )

    # Venue_City_Correct
    vcity_leaf = evaluator.add_leaf(
        id="Venue_City_Correct",
        desc="Provides the correct city for the Super Bowl LX host venue.",
        parent=sb_node,
        critical=True,
    )
    vcity_claim = f"The Super Bowl LX host venue '{sb.venue_name}' is located in {sb.city}."
    await evaluator.verify(
        claim=vcity_claim,
        node=vcity_leaf,
        sources=sb_sources,
        additional_instruction="Verify the venue's city from authoritative sources.",
    )

    # Venue_State_Correct
    vstate_leaf = evaluator.add_leaf(
        id="Venue_State_Correct",
        desc="Provides the correct state for the Super Bowl LX host venue.",
        parent=sb_node,
        critical=True,
    )
    vstate_claim = f"The Super Bowl LX host venue '{sb.venue_name}' is located in the state of {sb.state}."
    await evaluator.verify(
        claim=vstate_claim,
        node=vstate_leaf,
        sources=sb_sources,
        additional_instruction="Verify the venue's state from authoritative sources.",
    )

    # Venue_Capacity_Correct
    vcap_leaf = evaluator.add_leaf(
        id="Venue_Capacity_Correct",
        desc="Provides the correct seating capacity for the Super Bowl LX host venue (numeric).",
        parent=sb_node,
        critical=True,
    )
    vcap_claim = f"The seating capacity of the Super Bowl LX host venue '{sb.venue_name}' is {sb.seating_capacity}."
    await evaluator.verify(
        claim=vcap_claim,
        node=vcap_leaf,
        sources=sb_sources,
        additional_instruction="Confirm the venue capacity from authoritative sources; allow small configuration differences.",
    )

    # Super_Bowl_LX_Date_Correct
    vdate_leaf = evaluator.add_leaf(
        id="Super_Bowl_LX_Date_Correct",
        desc="Provides the correct date of Super Bowl LX.",
        parent=sb_node,
        critical=True,
    )
    vdate_claim = f"Super Bowl LX will take place on {sb.date}."
    await evaluator.verify(
        claim=vdate_claim,
        node=vdate_leaf,
        sources=sb_sources,
        additional_instruction="Confirm the official scheduled date for Super Bowl LX from authoritative sources (NFL announcements, venue, reputable media).",
    )

    # Halftime_Headliner_Correct
    headliner_leaf = evaluator.add_leaf(
        id="Halftime_Headliner_Correct",
        desc="Provides the correct halftime show headliner for Super Bowl LX.",
        parent=sb_node,
        critical=True,
    )
    headliner_claim = f"The halftime show headliner for Super Bowl LX is {sb.halftime_headliner}."
    await evaluator.verify(
        claim=headliner_claim,
        node=headliner_leaf,
        sources=sb_sources,
        additional_instruction="Verify the named halftime headliner for Super Bowl LX from authoritative sources.",
    )

    # Halftime_Sponsor_Correct
    sponsor_leaf = evaluator.add_leaf(
        id="Halftime_Sponsor_Correct",
        desc="Provides the correct corporate sponsor of the Super Bowl LX halftime show.",
        parent=sb_node,
        critical=True,
    )
    sponsor_claim = f"The corporate sponsor of the Super Bowl LX halftime show is {sb.halftime_sponsor}."
    await evaluator.verify(
        claim=sponsor_claim,
        node=sponsor_leaf,
        sources=sb_sources,
        additional_instruction="Verify the halftime show's corporate sponsor for Super Bowl LX from authoritative sources.",
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
    # Initialize evaluator (framework root is always a non-critical 'root')
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

    # Create Task_Completion node (critical parallel root of our rubric)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Answer all parts: Thanksgiving 2025 stadiums (3 items with required attributes) + largest-capacity determination + Super Bowl LX venue/event details.",
        parent=root,
        critical=True,
    )

    # Run extractions in parallel
    tgk_task = evaluator.extract(
        prompt=prompt_extract_thanksgiving(),
        template_class=ThanksgivingExtraction,
        extraction_name="thanksgiving_2025_stadiums",
    )
    sb_task = evaluator.extract(
        prompt=prompt_extract_superbowl_lx(),
        template_class=SuperBowlLXExtraction,
        extraction_name="super_bowl_lx",
    )
    thanksgiving_extraction, superbowl_extraction = await asyncio.gather(tgk_task, sb_task)

    # Verification: Thanksgiving block (sequential under Task_Completion)
    await verify_thanksgiving_block(evaluator, task_node, thanksgiving_extraction)

    # Verification: Super Bowl LX block (parallel under Task_Completion)
    await verify_super_bowl_lx_block(evaluator, task_node, superbowl_extraction)

    # Return evaluation summary
    return evaluator.get_summary()