import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_spring_2026_music_festivals"
TASK_DESCRIPTION = (
    "I'm planning a music festival trip to California during spring 2026 (April and May). I want to identify 4 different "
    "outdoor music festivals taking place in California during this period. For each festival, please provide the following information: "
    "(1) The official festival name, (2) The exact dates (start and end dates), (3) The venue name and its location (city in California), "
    "(4) Confirmation that it's an outdoor venue, (5) Confirmation that it's a multi-day event (at least 2 consecutive days), "
    "(6) Verification that the venue or festival has a minimum capacity of 5,000 attendees, (7) At least one confirmed headlining artist from the 2026 lineup, "
    "(8) The current ticket availability status, and (9) An official source URL for verification. Please ensure all festivals meet these criteria: "
    "outdoor venue, multi-day format, spring 2026 timeframe (April 1 - May 31, 2026), located in California, and with publicly announced 2026 lineups."
)

SPRING_START = "April 1, 2026"
SPRING_END = "May 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    official_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_address_or_specific_location: Optional[str] = None
    outdoor_venue_confirmed_text: Optional[str] = None
    capacity_info: Optional[str] = None
    confirmed_headliner_2026: Optional[str] = None
    ticket_status: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class FestivalsExtraction(BaseModel):
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
    Extract up to the first 8 distinct California outdoor music festivals mentioned in the answer that are scheduled for spring 2026 (April–May).
    For each festival, return an object with these fields exactly:
    - official_name: The official festival name as stated in the answer.
    - start_date: The exact start date (month/day/year) for the 2026 edition, if provided.
    - end_date: The exact end date (month/day/year) for the 2026 edition, if provided.
    - venue_name: The venue or festival grounds name.
    - venue_city: The city in California (USA) where the venue is located.
    - venue_address_or_specific_location: A specific publicly available address or location description (e.g., street address, park name + city).
    - outdoor_venue_confirmed_text: Any explicit statement in the answer confirming the venue is outdoor (text snippet or phrase).
    - capacity_info: Any capacity number or description (e.g., “capacity ~125,000” or “venue holds at least 5,000”) cited in the answer.
    - confirmed_headliner_2026: At least one named headliner from the 2026 lineup, if provided in the answer.
    - ticket_status: The current ticket availability status (e.g., “on sale,” “sold out,” “waitlist,” “TBA”), as stated in the answer.
    - source_urls: An array of URLs that the answer cites for this festival. Include only valid URLs explicitly present in the answer.
    
    Rules:
    - Do not invent or infer any value. If a value is not present in the answer, return null for that field (or [] for source_urls).
    - Only include URLs explicitly present in the answer text (plain link or markdown link). Ignore non-URL references.
    - Keep all strings as they appear in the answer; do not normalize formats.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _festival_key(f: FestivalItem) -> str:
    # Use a composite key to detect duplicates
    return "|".join([_norm(f.official_name), _norm(f.venue_name), _norm(f.venue_city)])


async def _verify_value_or_fail(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    value: Optional[str],
    claim_template: str,
    sources: List[str],
    add_ins: str,
    critical: bool = True,
) -> None:
    """
    If a specific value is required but missing, fail immediately via custom node.
    Otherwise, verify the value against sources.
    """
    if value is None or _norm(value) == "":
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=critical,
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    claim = claim_template.format(value=value)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Festival verification sub-tree                                              #
# --------------------------------------------------------------------------- #
async def verify_one_festival(
    evaluator: Evaluator,
    root_parent,
    fest: FestivalItem,
    index: int,
) -> None:
    """
    Build verification nodes for one festival and run checks.
    """
    # Create festival-level node (non-critical to allow partial credit across festivals)
    festival_node = evaluator.add_parallel(
        id=f"Festival_{index+1}",
        desc=f"{index+1}st qualifying festival." if index == 0 else (
             f"{index+1}nd qualifying festival." if index == 1 else (
             f"{index+1}rd qualifying festival." if index == 2 else f"{index+1}th qualifying festival."
        )),
        parent=root_parent,
        critical=False,
    )

    # Prepare sources
    sources = fest.source_urls if fest.source_urls else []

    # 1) Verification_Source_URL (critical, existence)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"Festival_{index+1}_Verification_Source_URL",
        desc="Provide at least one official festival website URL or reputable music-industry source URL supporting the provided details.",
        parent=festival_node,
        critical=True,
    )

    # 2) Official Festival Name (critical, value + source check)
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=festival_node,
        node_id=f"Festival_{index+1}_Official_Festival_Name",
        desc="Provide the official festival name.",
        value=fest.official_name,
        claim_template="The official or commonly used festival name for the 2026 edition is '{value}'.",
        sources=sources,
        add_ins="Use official site or highly reputable sources. Allow common abbreviations or well-known short names if they clearly refer to the same festival.",
        critical=True,
    )

    # 3) Dates & Duration (critical group)
    dates_group = evaluator.add_parallel(
        id=f"Festival_{index+1}_Dates_And_Duration",
        desc="Provide exact start/end dates and confirm timeframe and multi-day consecutive requirement.",
        parent=festival_node,
        critical=True,
    )

    # 3.1) Start Date
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=dates_group,
        node_id=f"Festival_{index+1}_Start_Date",
        desc="Provide the start date (month/day/year).",
        value=fest.start_date,
        claim_template="The start date for the festival's 2026 edition is '{value}'.",
        sources=sources,
        add_ins="Confirm the exact start date (month/day/year) for the 2026 edition on official or reputable pages.",
        critical=True,
    )

    # 3.2) End Date
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=dates_group,
        node_id=f"Festival_{index+1}_End_Date",
        desc="Provide the end date (month/day/year).",
        value=fest.end_date,
        claim_template="The end date for the festival's 2026 edition is '{value}'.",
        sources=sources,
        add_ins="Confirm the exact end date (month/day/year) for the 2026 edition on official or reputable pages.",
        critical=True,
    )

    # 3.3) Within April 1 to May 31, 2026
    within_leaf = evaluator.add_leaf(
        id=f"Festival_{index+1}_Within_Apr1_to_May31_2026",
        desc="Festival occurs within April 1, 2026 through May 31, 2026.",
        parent=dates_group,
        critical=True,
    )
    within_claim = (
        f"The festival's 2026 edition takes place entirely between {SPRING_START} and {SPRING_END}."
    )
    await evaluator.verify(
        claim=within_claim,
        node=within_leaf,
        sources=sources,
        additional_instruction="Use the official schedule or announcement to confirm that all event days fall within April 1–May 31, 2026.",
    )

    # 3.4) Multi-Day Consecutive
    multi_day_leaf = evaluator.add_leaf(
        id=f"Festival_{index+1}_Multi_Day_Consecutive",
        desc="Festival spans at least 2 consecutive days.",
        parent=dates_group,
        critical=True,
    )
    multi_day_claim = "The festival spans at least two consecutive days in 2026."
    await evaluator.verify(
        claim=multi_day_claim,
        node=multi_day_leaf,
        sources=sources,
        additional_instruction="Confirm from the official dates that the 2026 edition includes at least two consecutive days.",
    )

    # 4) Venue, Location & Outdoor (critical group)
    venue_group = evaluator.add_parallel(
        id=f"Festival_{index+1}_Venue_Location_And_Outdoor",
        desc="Provide venue details, confirm California location, and confirm outdoor setting.",
        parent=festival_node,
        critical=True,
    )

    # 4.1) Venue Name
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=venue_group,
        node_id=f"Festival_{index+1}_Venue_Name",
        desc="Provide the venue name.",
        value=fest.venue_name,
        claim_template="The 2026 edition is held at '{value}'.",
        sources=sources,
        add_ins="Confirm the venue or festival grounds name (e.g., 'Empire Polo Club').",
        critical=True,
    )

    # 4.2) Venue City in California
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=venue_group,
        node_id=f"Festival_{index+1}_Venue_City_In_California",
        desc="Provide the venue city and confirm it is in California (USA).",
        value=fest.venue_city,
        claim_template="The venue city for the 2026 edition is '{value}', California (USA).",
        sources=sources,
        add_ins="Confirm the city is in California using official site or reputable sources.",
        critical=True,
    )

    # 4.3) Venue Address or Specific Location (public)
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=venue_group,
        node_id=f"Festival_{index+1}_Venue_Address_Or_Specific_Location_Public",
        desc="Provide an exact venue address OR a specific publicly available venue location description sufficient to identify the venue location.",
        value=fest.venue_address_or_specific_location,
        claim_template="A publicly available address or specific location description includes: '{value}'.",
        sources=sources,
        add_ins="The provided address/location must be consistent with official or reputable sources.",
        critical=True,
    )

    # 4.4) Outdoor Venue Confirmed
    outdoor_leaf = evaluator.add_leaf(
        id=f"Festival_{index+1}_Outdoor_Venue_Confirmed",
        desc="Confirm the venue is an outdoor venue.",
        parent=venue_group,
        critical=True,
    )
    outdoor_claim = "The festival venue for the 2026 edition is an outdoor venue."
    await evaluator.verify(
        claim=outdoor_claim,
        node=outdoor_leaf,
        sources=sources,
        additional_instruction="Look for terms like 'outdoor', 'festival grounds', 'park', 'open-air', or clear photos/maps indicating outdoor setting.",
    )

    # 5) Minimum Capacity ≥ 5,000 (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"Festival_{index+1}_Minimum_Capacity_5000",
        desc="Verify the venue or festival has a minimum capacity of at least 5,000 attendees.",
        parent=festival_node,
        critical=True,
    )
    capacity_claim = "The festival or its venue has a minimum capacity of at least 5,000 attendees."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=sources,
        additional_instruction="Use official site or reputable sources to confirm capacity; venue capacity counts as valid if festival capacity is not explicitly stated.",
    )

    # 6) Festival type: Multi-Artist, Multi-Stage (critical)
    type_leaf = evaluator.add_leaf(
        id=f"Festival_{index+1}_Festival_Type_MultiArtist_MultiStage",
        desc="Confirm it is a music festival with multiple artists across multiple stages (not a single-artist concert).",
        parent=festival_node,
        critical=True,
    )
    type_claim = "This is a music festival with multiple artists across multiple stages (not a single-artist concert)."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=sources,
        additional_instruction="Confirm there are multiple artists and multiple stages from lineup/schedule pages.",
    )

    # 7) Lineup 2026 publicly announced (critical)
    announced_leaf = evaluator.add_leaf(
        id=f"Festival_{index+1}_Lineup_2026_Publicly_Announced",
        desc="Confirm the 2026 lineup has been officially announced and is publicly available.",
        parent=festival_node,
        critical=True,
    )
    announced_claim = "The 2026 lineup for this festival has been officially announced and is publicly available."
    await evaluator.verify(
        claim=announced_claim,
        node=announced_leaf,
        sources=sources,
        additional_instruction="Look for official posts/pages or reputable news confirming the 2026 lineup announcement.",
    )

    # 8) Confirmed 2026 Headliner (critical, value + source check)
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=festival_node,
        node_id=f"Festival_{index+1}_Confirmed_2026_Headliner",
        desc="Provide at least one confirmed headlining artist from the 2026 lineup.",
        value=fest.confirmed_headliner_2026,
        claim_template="The 2026 lineup includes the headliner '{value}'.",
        sources=sources,
        add_ins="Verify the specified artist is explicitly presented as a 'headliner' or equivalent for the 2026 edition on official or reputable sources. Do not infer a different artist.",
        critical=True,
    )

    # 9) Ticket Availability Status (critical, value + source check)
    await _verify_value_or_fail(
        evaluator=evaluator,
        parent_node=festival_node,
        node_id=f"Festival_{index+1}_Ticket_Availability_Status",
        desc="Provide the current ticket availability status, with publicly available verification.",
        value=fest.ticket_status,
        claim_template="The current ticket availability status for the 2026 edition is '{value}'.",
        sources=sources,
        add_ins="Check official ticketing page or reputable announcements (e.g., 'On sale', 'Sold out', 'Waitlist', 'Presale', 'TBA'). Accept reasonable synonyms.",
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
    Evaluate an answer for the California Spring 2026 music festivals task and return a structured result dictionary.
    """
    evaluator = Evaluator()

    # Root is non-critical to allow partial credit; critical checks are added under root
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

    # Extract festivals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction",
    )

    # Keep original count for "Exactly 4 Festivals Provided"
    original_count = len(extracted.festivals)

    # Select exactly the first 4 festivals (per Final Reminder guidance)
    selected: List[FestivalItem] = list(extracted.festivals[:4])

    # If fewer than 4 provided, pad with empty placeholders to still build the tree
    while len(selected) < 4:
        selected.append(FestivalItem())

    # Root-level critical check: Exactly 4 Festivals Provided
    evaluator.add_custom_node(
        result=(original_count == 4),
        id="Exactly_4_Festivals_Provided",
        desc="The response provides exactly 4 festivals (not fewer or more).",
        parent=root,
        critical=True,
    )

    # Root-level critical check: All Festivals Are Distinct (based on selected 4)
    keys = [_festival_key(f) for f in selected]
    distinct_result = len(set(k for k in keys if k.strip() != "")) == len([k for k in keys if k.strip() != ""])
    evaluator.add_custom_node(
        result=distinct_result,
        id="All_Festivals_Are_Distinct",
        desc="All listed festivals are different entities (no festival repeated).",
        parent=root,
        critical=True,
    )

    # Build verification subtrees for four festivals
    tasks = []
    for idx, fest in enumerate(selected):
        tasks.append(verify_one_festival(evaluator, root, fest, idx))
    # Run sequentially to maintain deterministic logging/order (could be gathered if desired)
    for t in tasks:
        await t

    # Return evaluation summary
    return evaluator.get_summary()