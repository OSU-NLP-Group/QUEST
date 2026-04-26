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
TASK_ID = "entertainment_venues_2025_2026"
TASK_DESCRIPTION = """
Identify four major entertainment venues across different cities in the United States that each hosted significant events between November 2025 and February 2026. Specifically, find:

1. A large indoor arena in Miami, Florida with a seating capacity of at least 19,000 that hosted a professional boxing match on December 19, 2025. The arena must have opened in 1999. Provide the venue's complete name, exact address, and seating capacity.

2. A historic movie palace in Hollywood, California with approximately 1,000 seats that hosted the world premiere of a Disney animated film on November 13, 2025. The theater must have originally opened in 1926. Provide the theater's complete name, exact address, and seating capacity.

3. A studio facility in Indianapolis, Indiana that serves as the regular filming location for a daily sports talk show that signed an $85 million, five-year contract with ESPN in 2023. The show broadcasts Monday through Friday from noon to 3 PM EST. Provide the studio's name, complete address, and the show's name.

4. A streaming service that added all 214 episodes across 10 seasons of a science fiction television series on February 15, 2026, after the series had been unavailable on that platform for over three years. Provide the streaming service name and the TV series name.

For each answer, provide supporting reference URLs that verify the key facts.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MiamiArena(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity: Optional[str] = None
    opened_year: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    boxing_event_name: Optional[str] = None
    boxing_event_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HollywoodTheater(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity: Optional[str] = None
    opened_year: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    premiere_film: Optional[str] = None
    premiere_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IndianapolisStudio(BaseModel):
    studio_name: Optional[str] = None
    address: Optional[str] = None
    show_name: Optional[str] = None
    contract_value: Optional[str] = None
    contract_year: Optional[str] = None
    broadcast_schedule: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StreamingServiceItem(BaseModel):
    service_name: Optional[str] = None
    series_name: Optional[str] = None
    episodes_count: Optional[str] = None
    seasons_count: Optional[str] = None
    added_date: Optional[str] = None
    absent_years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    miami_arena: Optional[MiamiArena] = None
    hollywood_theater: Optional[HollywoodTheater] = None
    indianapolis_studio: Optional[IndianapolisStudio] = None
    streaming_service: Optional[StreamingServiceItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_items() -> str:
    return """
    Extract structured information for the four required items from the provided answer text. Return a JSON object with the following top-level keys:
    - miami_arena
    - hollywood_theater
    - indianapolis_studio
    - streaming_service

    For each key, extract the fields listed below exactly as mentioned in the answer. If a field is not mentioned, return null for that field. For each item, also extract all explicit URL sources cited for that item.

    1) miami_arena:
       - name: full official name of the arena
       - address: exact street address (including city and state if present)
       - capacity: seating capacity value stated in the answer (string as-is)
       - opened_year: opening year stated in the answer
       - city: city name (e.g., Miami)
       - state: state abbreviation or full state name (e.g., FL or Florida)
       - boxing_event_name: name or description of the professional boxing match (if provided)
       - boxing_event_date: date of the match (e.g., December 19, 2025)
       - sources: array of all URLs explicitly cited for this arena

    2) hollywood_theater:
       - name: full official name of the theater
       - address: exact street address
       - capacity: seating capacity value stated in the answer (string as-is)
       - opened_year: original opening year stated in the answer
       - city: city or district (e.g., Hollywood / Los Angeles)
       - state: state abbreviation or name (e.g., CA or California)
       - premiere_film: title of the Disney animated film
       - premiere_date: date of the world premiere (e.g., November 13, 2025)
       - sources: array of all URLs explicitly cited for this theater

    3) indianapolis_studio:
       - studio_name: name of the studio facility
       - address: complete street address
       - show_name: name of the daily sports talk show
       - contract_value: the contract amount stated (e.g., $85 million)
       - contract_year: the year stated for the contract (e.g., 2023)
       - broadcast_schedule: schedule description (e.g., Monday–Friday, noon to 3 PM EST)
       - sources: array of all URLs explicitly cited for this studio/show

    4) streaming_service:
       - service_name: name of the streaming service/platform
       - series_name: name of the science fiction TV series
       - episodes_count: total number of episodes added (e.g., 214)
       - seasons_count: total number of seasons added (e.g., 10)
       - added_date: date added (e.g., February 15, 2026)
       - absent_years: description of absence (e.g., "over three years")
       - sources: array of all URLs explicitly cited for this streaming item

    IMPORTANT:
    - Extract only information explicitly present in the answer text.
    - For sources, only include actual URLs mentioned (plain URLs or URLs inside markdown links).
    - If a URL is missing protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_string(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _sources_exist(sources: Optional[List[str]]) -> bool:
    return bool(sources) and len(sources) > 0


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_miami_arena(evaluator: Evaluator, parent_node, info: Optional[MiamiArena]) -> None:
    node = evaluator.add_sequential(
        id="miami_arena",
        desc="Large indoor arena in Miami, Florida: capacity ≥ 19,000; opened in 1999; hosted a professional boxing match on Dec 19, 2025; provide complete name, exact address, and seating capacity",
        parent=parent_node,
        critical=False
    )

    # Required info existence check
    exists = bool(info) and _has_nonempty_string(info.name) and _has_nonempty_string(info.address) and _has_nonempty_string(info.capacity) and _sources_exist(info.sources)
    evaluator.add_custom_node(
        result=exists,
        id="miami_arena_required_info",
        desc="Miami arena has required information (name, address, capacity) and sources",
        parent=node,
        critical=True
    )

    # Constraints and facts (parallel)
    constraints = evaluator.add_parallel(
        id="miami_arena_constraints",
        desc="Miami arena constraints verification",
        parent=node,
        critical=False
    )

    sources = info.sources if info else []

    # Location check: Miami, Florida
    loc_node = evaluator.add_leaf(
        id="miami_arena_location_miami",
        desc="Arena is located in Miami, Florida",
        parent=constraints,
        critical=True
    )
    loc_claim = f"The venue '{info.name if info else ''}' at address '{info.address if info else ''}' is located in Miami, Florida."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm the venue is in Miami, FL. Accept 'Miami, FL' or 'Miami, Florida'. If the address shows Miami, FL, consider it correct."
    )

    # Type check: indoor arena
    type_node = evaluator.add_leaf(
        id="miami_arena_type_indoor",
        desc="Venue is an indoor arena",
        parent=constraints,
        critical=True
    )
    type_claim = f"The venue '{info.name if info else ''}' is an indoor arena."
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=sources,
        additional_instruction="Verify from sources that the venue is described as an indoor arena or a multi-purpose indoor arena."
    )

    # Capacity threshold
    cap_thresh_node = evaluator.add_leaf(
        id="miami_arena_capacity_threshold",
        desc="Arena capacity is at least 19,000 seats",
        parent=constraints,
        critical=True
    )
    cap_thresh_claim = f"The seating capacity of '{info.name if info else ''}' is at least 19,000."
    await evaluator.verify(
        claim=cap_thresh_claim,
        node=cap_thresh_node,
        sources=sources,
        additional_instruction="Use the cited sources to confirm the maximum seating capacity is ≥ 19,000. Accept basketball or concert capacity if the source defines a typical capacity above this threshold."
    )

    # Opening year 1999
    open_node = evaluator.add_leaf(
        id="miami_arena_opened_1999",
        desc="Arena opened in 1999",
        parent=constraints,
        critical=True
    )
    open_claim = f"The venue '{info.name if info else ''}' opened in 1999."
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=sources,
        additional_instruction="Confirm the original opening year is 1999 as stated by reliable sources."
    )

    # Boxing event Dec 19, 2025
    boxing_node = evaluator.add_leaf(
        id="miami_arena_boxing_2025_12_19",
        desc="Arena hosted a professional boxing match on December 19, 2025",
        parent=constraints,
        critical=True
    )
    boxing_claim = f"The venue '{info.name if info else ''}' hosted a professional boxing match on December 19, 2025."
    await evaluator.verify(
        claim=boxing_claim,
        node=boxing_node,
        sources=sources,
        additional_instruction="Verify that a professional boxing event took place at this venue on Dec 19, 2025. Accept sources such as official venue pages, credible news coverage, or boxing event listings."
    )

    # Details (parallel)
    details = evaluator.add_parallel(
        id="miami_arena_details",
        desc="Miami arena details accuracy (name, address, capacity value)",
        parent=node,
        critical=False
    )

    # Name accuracy
    name_node = evaluator.add_leaf(
        id="miami_arena_name_supported",
        desc="Arena full official name is supported by sources",
        parent=details,
        critical=True
    )
    name_claim = f"The full official name of the arena is '{info.name if info else ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=sources,
        additional_instruction="Confirm the official naming of the venue; allow minor formatting differences (e.g., with/without 'Arena' or sponsorship where applicable) if sources indicate equivalence."
    )

    # Address accuracy
    addr_node = evaluator.add_leaf(
        id="miami_arena_address_supported",
        desc="Arena exact address is supported by sources",
        parent=details,
        critical=True
    )
    addr_claim = f"The exact address of the arena is '{info.address if info else ''}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=sources,
        additional_instruction="Verify the exact street address. Allow minor formatting variations (e.g., abbreviations like 'Ave.' vs 'Avenue') if clearly the same address."
    )

    # Capacity value accuracy (as provided)
    cap_val_node = evaluator.add_leaf(
        id="miami_arena_capacity_value_supported",
        desc="Arena seating capacity value (as provided) is supported by sources",
        parent=details,
        critical=True
    )
    cap_val_claim = f"The seating capacity of the arena is '{info.capacity if info else ''}'."
    await evaluator.verify(
        claim=cap_val_claim,
        node=cap_val_node,
        sources=sources,
        additional_instruction="Check the capacity value stated in the answer is supported by the cited sources; accept reasonable rounding differences."
    )


async def verify_hollywood_theater(evaluator: Evaluator, parent_node, info: Optional[HollywoodTheater]) -> None:
    node = evaluator.add_sequential(
        id="hollywood_theater",
        desc="Historic movie palace in Hollywood, California: approx 1,000 seats; originally opened in 1926; hosted Disney animated film world premiere on Nov 13, 2025; provide complete name, exact address, and seating capacity",
        parent=parent_node,
        critical=False
    )

    exists = bool(info) and _has_nonempty_string(info.name) and _has_nonempty_string(info.address) and _has_nonempty_string(info.capacity) and _sources_exist(info.sources)
    evaluator.add_custom_node(
        result=exists,
        id="hollywood_theater_required_info",
        desc="Hollywood theater has required information (name, address, capacity) and sources",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="hollywood_theater_constraints",
        desc="Hollywood theater constraints verification",
        parent=node,
        critical=False
    )

    sources = info.sources if info else []

    # Location: Hollywood, California (allow Los Angeles/Hollywood neighborhood)
    loc_node = evaluator.add_leaf(
        id="hollywood_theater_location_hollywood",
        desc="Theater is located in Hollywood, California",
        parent=constraints,
        critical=True
    )
    loc_claim = f"The theater '{info.name if info else ''}' at address '{info.address if info else ''}' is located in Hollywood, California."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Accept 'Hollywood' as a neighborhood of Los Angeles, CA. If sources show a Hollywood/Los Angeles address, consider it correct."
    )

    # Opening year 1926
    open_node = evaluator.add_leaf(
        id="hollywood_theater_opened_1926",
        desc="Theater originally opened in 1926",
        parent=constraints,
        critical=True
    )
    open_claim = f"The theater '{info.name if info else ''}' originally opened in 1926."
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=sources,
        additional_instruction="Confirm the original opening year is 1926 per reliable sources."
    )

    # Approx 1,000 seats
    cap_approx_node = evaluator.add_leaf(
        id="hollywood_theater_capacity_approx_1000",
        desc="Theater has approximately 1,000 seats",
        parent=constraints,
        critical=True
    )
    cap_approx_claim = f"The theater '{info.name if info else ''}' has approximately 1,000 seats."
    await evaluator.verify(
        claim=cap_approx_claim,
        node=cap_approx_node,
        sources=sources,
        additional_instruction="Accept seat counts roughly in the 850–1,200 range as 'approximately 1,000' if supported by sources."
    )

    # World premiere of Disney animated film on Nov 13, 2025
    premiere_node = evaluator.add_leaf(
        id="hollywood_theater_premiere_2025_11_13",
        desc="Theater hosted world premiere of a Disney animated film on November 13, 2025",
        parent=constraints,
        critical=True
    )
    film_title = info.premiere_film if info and _has_nonempty_string(info.premiere_film) else ""
    prem_claim = f"On November 13, 2025, the theater '{info.name if info else ''}' hosted the world premiere of the Disney animated film '{film_title}'."
    await evaluator.verify(
        claim=prem_claim,
        node=premiere_node,
        sources=sources,
        additional_instruction="Confirm it was the world premiere, that the film is a Disney animated title, and the date is Nov 13, 2025. Accept official theater announcements or reputable press coverage."
    )

    details = evaluator.add_parallel(
        id="hollywood_theater_details",
        desc="Hollywood theater details accuracy (name, address, capacity value)",
        parent=node,
        critical=False
    )

    # Name accuracy
    name_node = evaluator.add_leaf(
        id="hollywood_theater_name_supported",
        desc="Theater full official name is supported by sources",
        parent=details,
        critical=True
    )
    name_claim = f"The full official name of the theater is '{info.name if info else ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=sources,
        additional_instruction="Verify official naming; allow minor formatting or sponsorship variations if sources indicate equivalence."
    )

    # Address accuracy
    addr_node = evaluator.add_leaf(
        id="hollywood_theater_address_supported",
        desc="Theater exact address is supported by sources",
        parent=details,
        critical=True
    )
    addr_claim = f"The exact address of the theater is '{info.address if info else ''}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=sources,
        additional_instruction="Verify the exact street address; accept minor abbreviation variants if clearly the same location."
    )

    # Capacity value accuracy (as provided)
    cap_val_node = evaluator.add_leaf(
        id="hollywood_theater_capacity_value_supported",
        desc="Theater seating capacity value (as provided) is supported by sources",
        parent=details,
        critical=True
    )
    cap_val_claim = f"The seating capacity of the theater is '{info.capacity if info else ''}'."
    await evaluator.verify(
        claim=cap_val_claim,
        node=cap_val_node,
        sources=sources,
        additional_instruction="Confirm the capacity value stated in the answer is supported; accept reasonable rounding."
    )


async def verify_indianapolis_studio(evaluator: Evaluator, parent_node, info: Optional[IndianapolisStudio]) -> None:
    node = evaluator.add_sequential(
        id="indianapolis_studio",
        desc="Indianapolis studio facility: regular filming location for daily sports talk show; show signed $85M five-year ESPN contract in 2023; broadcasts Mon–Fri noon–3 PM EST; provide studio name, complete address, and show name",
        parent=parent_node,
        critical=False
    )

    exists = bool(info) and _has_nonempty_string(info.studio_name) and _has_nonempty_string(info.address) and _has_nonempty_string(info.show_name) and _sources_exist(info.sources)
    evaluator.add_custom_node(
        result=exists,
        id="indianapolis_studio_required_info",
        desc="Indianapolis studio has required information (studio name, address, show name) and sources",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="indianapolis_studio_constraints",
        desc="Indianapolis studio constraints verification",
        parent=node,
        critical=False
    )

    sources = info.sources if info else []

    # Regular filming location
    loc_node = evaluator.add_leaf(
        id="indianapolis_studio_regular_location",
        desc="Studio is the regular filming location for the show",
        parent=constraints,
        critical=True
    )
    loc_claim = f"The show '{info.show_name if info else ''}' regularly films at the studio '{info.studio_name if info else ''}' located at '{info.address if info else ''}' in Indianapolis, Indiana."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm the studio is the recurring/regular filming location for the show. Accept credible show pages, official announcements, or reputable media coverage."
    )

    # ESPN contract in 2023: $85M, five-year
    contract_node = evaluator.add_leaf(
        id="indianapolis_studio_contract_espn_2023",
        desc="Show signed an $85M five-year contract with ESPN in 2023",
        parent=constraints,
        critical=True
    )
    contract_claim = f"In 2023, the show '{info.show_name if info else ''}' signed a five-year contract worth $85 million with ESPN."
    await evaluator.verify(
        claim=contract_claim,
        node=contract_node,
        sources=sources,
        additional_instruction="Confirm contract year (2023), length (five years), and amount ($85M). Accept reputable news sources or ESPN announcements."
    )

    # Broadcast schedule: Mon–Fri noon–3 PM EST
    sched_node = evaluator.add_leaf(
        id="indianapolis_studio_schedule_mf_noon_3pm_est",
        desc="Show broadcasts Monday–Friday from noon to 3 PM EST",
        parent=constraints,
        critical=True
    )
    sched_claim = f"The show '{info.show_name if info else ''}' broadcasts Monday through Friday from noon to 3 PM EST."
    await evaluator.verify(
        claim=sched_claim,
        node=sched_node,
        sources=sources,
        additional_instruction="Confirm weekday broadcast window is 12:00–3:00 PM Eastern. Accept 'ET' as equivalent to 'EST' if the show page uses ET."
    )

    details = evaluator.add_parallel(
        id="indianapolis_studio_details",
        desc="Indianapolis studio details accuracy (studio name, address, show name)",
        parent=node,
        critical=False
    )

    # Studio name supported
    studio_name_node = evaluator.add_leaf(
        id="indianapolis_studio_name_supported",
        desc="Studio name is supported by sources",
        parent=details,
        critical=True
    )
    studio_name_claim = f"The studio facility name is '{info.studio_name if info else ''}'."
    await evaluator.verify(
        claim=studio_name_claim,
        node=studio_name_node,
        sources=sources,
        additional_instruction="Verify the studio facility naming; allow minor variations if sources indicate equivalence."
    )

    # Address supported
    studio_addr_node = evaluator.add_leaf(
        id="indianapolis_studio_address_supported",
        desc="Studio exact address is supported by sources",
        parent=details,
        critical=True
    )
    studio_addr_claim = f"The studio address is '{info.address if info else ''}'."
    await evaluator.verify(
        claim=studio_addr_claim,
        node=studio_addr_node,
        sources=sources,
        additional_instruction="Verify the exact street address; allow minor abbreviation variants."
    )

    # Show name supported
    show_name_node = evaluator.add_leaf(
        id="indianapolis_studio_show_name_supported",
        desc="Show name is supported by sources",
        parent=details,
        critical=True
    )
    show_name_claim = f"The show name is '{info.show_name if info else ''}'."
    await evaluator.verify(
        claim=show_name_claim,
        node=show_name_node,
        sources=sources,
        additional_instruction="Verify the show title as used by official or reputable sources; allow minor formatting variations."
    )


async def verify_streaming_service(evaluator: Evaluator, parent_node, info: Optional[StreamingServiceItem]) -> None:
    node = evaluator.add_sequential(
        id="streaming_service",
        desc="Streaming service: added all 214 episodes across 10 seasons of a sci‑fi series on Feb 15, 2026 after >3 years absence; provide service name and series name",
        parent=parent_node,
        critical=False
    )

    exists = bool(info) and _has_nonempty_string(info.service_name) and _has_nonempty_string(info.series_name) and _sources_exist(info.sources)
    evaluator.add_custom_node(
        result=exists,
        id="streaming_service_required_info",
        desc="Streaming service item has required information (service name, series name) and sources",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="streaming_service_constraints",
        desc="Streaming service constraints verification",
        parent=node,
        critical=False
    )

    sources = info.sources if info else []

    # Added date check
    date_node = evaluator.add_leaf(
        id="streaming_service_added_2026_02_15",
        desc="Service added the series on February 15, 2026",
        parent=constraints,
        critical=True
    )
    date_claim = f"On February 15, 2026, the streaming service '{info.service_name if info else ''}' added the science fiction TV series '{info.series_name if info else ''}'."
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources,
        additional_instruction="Confirm add date is Feb 15, 2026 via the platform announcement, press releases, or reputable coverage."
    )

    # Episodes and seasons total
    eps_seasons_node = evaluator.add_leaf(
        id="streaming_service_214_episodes_10_seasons",
        desc="Service added all 214 episodes across 10 seasons",
        parent=constraints,
        critical=True
    )
    eps_seasons_claim = f"The streaming service '{info.service_name if info else ''}' added all 214 episodes across 10 seasons of '{info.series_name if info else ''}'."
    await evaluator.verify(
        claim=eps_seasons_claim,
        node=eps_seasons_node,
        sources=sources,
        additional_instruction="Confirm that the catalog addition includes 214 episodes across 10 seasons; accept official platform libraries or credible reports."
    )

    # Absence duration (>3 years)
    absence_node = evaluator.add_leaf(
        id="streaming_service_absent_over_3_years",
        desc="Series was absent from the platform for over three years prior to the addition",
        parent=constraints,
        critical=True
    )
    absence_claim = f"Before February 15, 2026, the series '{info.series_name if info else ''}' had been unavailable on '{info.service_name if info else ''}' for over three years."
    await evaluator.verify(
        claim=absence_claim,
        node=absence_node,
        sources=sources,
        additional_instruction="Confirm the duration of absence on that specific platform exceeded three years, based on date ranges in credible sources."
    )

    details = evaluator.add_parallel(
        id="streaming_service_details",
        desc="Streaming service details accuracy (service name, series name)",
        parent=node,
        critical=False
    )

    # Service name supported
    service_name_node = evaluator.add_leaf(
        id="streaming_service_name_supported",
        desc="Streaming service name is supported by sources",
        parent=details,
        critical=True
    )
    service_name_claim = f"The streaming service name is '{info.service_name if info else ''}'."
    await evaluator.verify(
        claim=service_name_claim,
        node=service_name_node,
        sources=sources,
        additional_instruction="Verify the platform/service naming; allow branding variations if sources indicate equivalence."
    )

    # Series name supported
    series_name_node = evaluator.add_leaf(
        id="streaming_service_series_name_supported",
        desc="Series name is supported by sources",
        parent=details,
        critical=True
    )
    series_name_claim = f"The TV series name is '{info.series_name if info else ''}'."
    await evaluator.verify(
        claim=series_name_claim,
        node=series_name_node,
        sources=sources,
        additional_instruction="Verify the series title; allow minor subtitle or punctuation variations if sources indicate equivalence."
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
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the entertainment venues/services events task (Nov 2025–Feb 2026).
    """
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

    # Extract all items from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_items(),
        template_class=TaskExtraction,
        extraction_name="venues_and_services_extraction",
    )

    # Build verification subtrees for each item
    await verify_miami_arena(evaluator, root, extraction.miami_arena)
    await verify_hollywood_theater(evaluator, root, extraction.hollywood_theater)
    await verify_indianapolis_studio(evaluator, root, extraction.indianapolis_studio)
    await verify_streaming_service(evaluator, root, extraction.streaming_service)

    # Return structured result
    return evaluator.get_summary()