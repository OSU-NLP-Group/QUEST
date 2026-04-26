import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "netflix_christmas_nfl_event_2025"
TASK_DESCRIPTION = (
    "On Christmas Day 2025, Netflix streamed an NFL game featuring a halftime show with multiple performers. "
    "Provide complete details about this event, including: (1) the exact date and kickoff time, (2) the two competing teams, "
    "(3) the venue name and city, (4) the streaming platform and event series name, (5) the halftime show title, "
    "(6) the headlining performer, and (7) all additional performers who participated in the halftime show. "
    "For each piece of information, include a reference URL."
)

# Ground-truth targets (used for claim phrasing and logging)
GT = {
    "date": "December 25, 2025",
    "kickoff_time_et": "4:30 PM ET",
    "teams": ("Detroit Lions", "Minnesota Vikings"),
    "venue_name": "U.S. Bank Stadium",
    "venue_city_state": "Minneapolis, Minnesota",
    "platform": "Netflix",
    "event_series": "NFL Christmas Gameday",
    "show_title": "Snoop's Holiday Halftime Party",
    "headliner": "Snoop Dogg",
    "supporting_performers": {
        "lainey_wilson": "Lainey Wilson",
        "kpop_demon_hunters": "K-Pop Demon Hunters",
        "bocelli_family": ("Andrea Bocelli", "Matteo Bocelli"),
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventCoreExtraction(BaseModel):
    # Date & Time
    date: Optional[str] = None
    kickoff_time: Optional[str] = None   # Preferably including timezone (e.g., "4:30 PM ET")
    date_time_sources: List[str] = Field(default_factory=list)

    # Teams & Venue
    team1: Optional[str] = None
    team2: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    teams_venue_sources: List[str] = Field(default_factory=list)

    # Platform & Event Series
    streaming_platform: Optional[str] = None     # e.g., Netflix
    event_series_name: Optional[str] = None      # e.g., NFL Christmas Gameday
    platform_sources: List[str] = Field(default_factory=list)


class HalftimeMainExtraction(BaseModel):
    show_title: Optional[str] = None
    headliner: Optional[str] = None
    title_headliner_sources: List[str] = Field(default_factory=list)


class PerformerEntry(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SupportingPerformersExtraction(BaseModel):
    performers: List[PerformerEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_core() -> str:
    return """
Extract the core event details exactly as presented in the answer. Return the following fields:

1) Date and kickoff time (include timezone if present):
- date: textual date string (e.g., "December 25, 2025")
- kickoff_time: textual time string as given (e.g., "4:30 PM ET", "4:30 p.m. Eastern", etc.)
- date_time_sources: an array of all URL(s) cited in the answer that support the date and/or time. If none are provided, return an empty array.

2) Teams and venue:
- team1: name of the first team (string)
- team2: name of the second team (string)
- venue_name: venue/stadium name (string)
- venue_city: city name (string)
- venue_state: state (or state/region) (string)
- teams_venue_sources: an array of URL(s) that support the teams and/or venue/location. If none are provided, return an empty array.

3) Streaming platform and event series name:
- streaming_platform: the platform (e.g., "Netflix")
- event_series_name: the event series name as stated (e.g., "NFL Christmas Gameday", "Christmas Day Game", etc.)
- platform_sources: an array of URL(s) that support the streaming/platform information and/or series branding. If none are provided, return an empty array.

If any field is missing in the answer, set it to null (or empty array for URL lists).
"""


def prompt_extract_halftime_main() -> str:
    return """
Extract the halftime show’s main details exactly as presented in the answer:

- show_title: the halftime show’s title (e.g., "Snoop's Holiday Halftime Party")
- headliner: the headlining performer’s name (e.g., "Snoop Dogg")
- title_headliner_sources: an array of URL(s) cited to support the show title and/or headliner. If none are provided, return an empty array.

If any field is missing, set it to null (or empty array for URLs).
"""


def prompt_extract_supporting_performers() -> str:
    return """
Extract all supporting (additional) performers for the halftime show as a list named 'performers'.
For each performer, include:
- name: performer's name as stated in the answer (e.g., "Lainey Wilson", "K-Pop Demon Hunters", "Andrea Bocelli", "Matteo Bocelli")
- sources: an array of URL(s) provided in the answer that specifically support that performer's participation. If none are provided for that performer, return an empty array.

Return JSON:
{
  "performers": [
    {"name": "...", "sources": ["...", "..."]},
    ...
  ]
}

Include every supporting performer mentioned in the answer. Do not invent performers or URLs.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[:'’\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _find_performer_sources(performers: List[PerformerEntry], target_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Given a list of performer entries and one or more target names, return:
    - combined_sources: union of sources for any entry whose normalized name equals or contains any target normalized name
    - matched_names: list of matched performer names (original, not normalized)
    """
    combined_sources: List[str] = []
    matched_names: List[str] = []
    norm_targets = [_normalize_name(n) for n in target_names if n]

    for entry in performers:
        norm_entry = _normalize_name(entry.name)
        if not norm_entry:
            continue
        for t in norm_targets:
            # Match if equals or contains (to be robust against punctuation variants, e.g., "k pop demon hunters" vs "k-pop demon hunters" or "k-pop: demon hunters")
            if norm_entry == t or t in norm_entry or norm_entry in t:
                matched_names.append(entry.name or "")
                for u in entry.sources:
                    if u and u not in combined_sources:
                        combined_sources.append(u)
                break

    return combined_sources, matched_names


def _city_state_str(city: Optional[str], state: Optional[str]) -> str:
    city = (city or "").strip()
    state = (state or "").strip()
    if city and state:
        return f"{city}, {state}"
    return f"{city}{state}" if (city or state) else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_event_core_nodes(evaluator: Evaluator, parent) -> None:
    """
    Build 'Event_Core_Details' subtree with verification leaves tied to provided sources.
    """
    # Container node
    event_core_node = evaluator.add_parallel(
        id="Event_Core_Details",
        desc="Essential event information including date, teams, venue, and platform",
        parent=parent,
        critical=True,
    )

    # Extract event core details
    core: EventCoreExtraction = await evaluator.extract(
        prompt=prompt_extract_event_core(),
        template_class=EventCoreExtraction,
        extraction_name="event_core",
    )

    # Record optional ground-truth info
    evaluator.add_ground_truth({
        "expected_date": GT["date"],
        "expected_kickoff_time_ET": GT["kickoff_time_et"],
        "expected_teams": list(GT["teams"]),
        "expected_venue": GT["venue_name"],
        "expected_city_state": GT["venue_city_state"],
        "expected_platform": GT["platform"],
        "expected_series": GT["event_series"],
    }, gt_type="ground_truth_event_core")

    # 1) Date & Time
    dt_parent = evaluator.add_parallel(
        id="Event_DateTime",
        desc="Date and time of the event",
        parent=event_core_node,
        critical=True,
    )

    # Reference existence node (critical)
    evaluator.add_custom_node(
        result=bool(core.date_time_sources),
        id="DateTime_Reference",
        desc="Provide URL reference for date and time",
        parent=dt_parent,
        critical=True,
    )

    dt_leaf = evaluator.add_leaf(
        id="Date_and_Time",
        desc="Event occurred on December 25, 2025 at 4:30 PM EST",
        parent=dt_parent,
        critical=True,
    )
    claim_dt = (
        "The Netflix-streamed NFL game occurred on December 25, 2025 with a kickoff around 4:30 PM Eastern Time "
        "(accept ET/EST/“Eastern” and minor formatting variants)."
    )
    await evaluator.verify(
        claim=claim_dt,
        node=dt_leaf,
        sources=core.date_time_sources,
        additional_instruction="Accept 'ET', 'EST', 'Eastern Time', and minor formatting variants like '4:30 p.m.'.",
    )

    # 2) Teams & Venue
    tv_parent = evaluator.add_parallel(
        id="Teams_and_Venue",
        desc="Participating teams and game location",
        parent=event_core_node,
        critical=True,
    )

    # Reference existence node (critical)
    evaluator.add_custom_node(
        result=bool(core.teams_venue_sources),
        id="Location_Reference",
        desc="Provide URL reference for teams and venue",
        parent=tv_parent,
        critical=True,
    )

    teams_leaf = evaluator.add_leaf(
        id="Teams",
        desc="Detroit Lions vs Minnesota Vikings",
        parent=tv_parent,
        critical=True,
    )
    claim_teams = (
        "The two teams that played were the Detroit Lions and the Minnesota Vikings."
    )
    await evaluator.verify(
        claim=claim_teams,
        node=teams_leaf,
        sources=core.teams_venue_sources,
        additional_instruction="Allow minor formatting variants (e.g., order of teams).",
    )

    venue_leaf = evaluator.add_leaf(
        id="Venue",
        desc="U.S. Bank Stadium in Minneapolis, Minnesota",
        parent=tv_parent,
        critical=True,
    )
    claim_venue = (
        "The game was played at U.S. Bank Stadium in Minneapolis, Minnesota."
    )
    await evaluator.verify(
        claim=claim_venue,
        node=venue_leaf,
        sources=core.teams_venue_sources,
        additional_instruction="Verify both venue name and city/state appear or are clearly implied.",
    )

    # 3) Broadcasting: Platform & Event Series
    b_parent = evaluator.add_parallel(
        id="Broadcasting",
        desc="Streaming platform and event series",
        parent=event_core_node,
        critical=True,
    )

    # Reference existence node (critical)
    evaluator.add_custom_node(
        result=bool(core.platform_sources),
        id="Platform_Reference",
        desc="Provide URL reference for platform information",
        parent=b_parent,
        critical=True,
    )

    platform_leaf = evaluator.add_leaf(
        id="Platform",
        desc="Streamed on Netflix as part of NFL Christmas Gameday",
        parent=b_parent,
        critical=True,
    )
    claim_platform = (
        "The NFL Christmas Day game streamed on Netflix, as part of the NFL's Christmas Day programming "
        "(also referred to as 'NFL Christmas Gameday' or similar branding)."
    )
    await evaluator.verify(
        claim=claim_platform,
        node=platform_leaf,
        sources=core.platform_sources,
        additional_instruction="Accept reasonable variants of series branding such as 'Christmas Day game(s)' or 'NFL Christmas Gameday'.",
    )


async def build_halftime_show_nodes(evaluator: Evaluator, parent) -> None:
    """
    Build 'Halftime_Show_Complete' subtree with verification leaves tied to provided sources.
    All child nodes are marked critical to satisfy framework constraints for critical parents.
    """
    # Container node
    halftime_root = evaluator.add_parallel(
        id="Halftime_Show_Complete",
        desc="Complete details of the halftime show including title and all performers",
        parent=parent,
        critical=True,
    )

    # Extract main halftime details
    main: HalftimeMainExtraction = await evaluator.extract(
        prompt=prompt_extract_halftime_main(),
        template_class=HalftimeMainExtraction,
        extraction_name="halftime_main",
    )

    evaluator.add_ground_truth({
        "expected_show_title": GT["show_title"],
        "expected_headliner": GT["headliner"],
    }, gt_type="ground_truth_halftime_main")

    # Show title and headliner
    sh_parent = evaluator.add_parallel(
        id="Show_Title_and_Headliner",
        desc="Show name and main performer",
        parent=halftime_root,
        critical=True,
    )

    # Reference existence
    evaluator.add_custom_node(
        result=bool(main.title_headliner_sources),
        id="Title_Headliner_Reference",
        desc="Provide URL reference for show title and headliner",
        parent=sh_parent,
        critical=True,
    )

    # Show Title leaf
    show_title_leaf = evaluator.add_leaf(
        id="Show_Title",
        desc="Show titled 'Snoop's Holiday Halftime Party'",
        parent=sh_parent,
        critical=True,
    )
    claim_title = "The halftime show was titled “Snoop's Holiday Halftime Party.”"
    await evaluator.verify(
        claim=claim_title,
        node=show_title_leaf,
        sources=main.title_headliner_sources,
        additional_instruction="Allow minor punctuation or apostrophe variants and casing differences.",
    )

    # Headliner leaf
    headliner_leaf = evaluator.add_leaf(
        id="Headliner_Name",
        desc="Headlined by Snoop Dogg",
        parent=sh_parent,
        critical=True,
    )
    claim_headliner = "Snoop Dogg was the headlining performer for the halftime show."
    await evaluator.verify(
        claim=claim_headliner,
        node=headliner_leaf,
        sources=main.title_headliner_sources,
        additional_instruction="Treat 'Snoop Dogg' variants (e.g., with/without middle initials) as equivalent.",
    )

    # Extract supporting performers
    supp: SupportingPerformersExtraction = await evaluator.extract(
        prompt=prompt_extract_supporting_performers(),
        template_class=SupportingPerformersExtraction,
        extraction_name="halftime_supporting_performers",
    )

    evaluator.add_ground_truth({
        "expected_supporting_performers": [
            GT["supporting_performers"]["lainey_wilson"],
            GT["supporting_performers"]["kpop_demon_hunters"],
            *GT["supporting_performers"]["bocelli_family"],
        ]
    }, gt_type="ground_truth_supporting_performers")

    # Supporting performers container (critical)
    supp_parent = evaluator.add_parallel(
        id="Supporting_Performers",
        desc="All additional performers",
        parent=halftime_root,
        critical=True,
    )

    # Country_Performer (Lainey Wilson)
    country_parent = evaluator.add_parallel(
        id="Country_Performer",
        desc="Lainey Wilson performed",
        parent=supp_parent,
        critical=True,
    )
    lainey_sources, lainey_matched = _find_performer_sources(
        supp.performers, [GT["supporting_performers"]["lainey_wilson"]]
    )
    evaluator.add_custom_node(
        result=bool(lainey_sources),
        id="Lainey_Reference",
        desc="Provide URL reference for Lainey Wilson",
        parent=country_parent,
        critical=True,
    )
    lainey_leaf = evaluator.add_leaf(
        id="Lainey_Wilson",
        desc="Lainey Wilson was a featured performer",
        parent=country_parent,
        critical=True,
    )
    claim_lainey = "Lainey Wilson performed during the halftime show."
    await evaluator.verify(
        claim=claim_lainey,
        node=lainey_leaf,
        sources=lainey_sources,
        additional_instruction="Allow that Lainey Wilson could be described as 'performing' or 'appearing' in the halftime show.",
    )

    # KPop_Performers (K-Pop Demon Hunters)
    kpop_parent = evaluator.add_parallel(
        id="KPop_Performers",
        desc="K-Pop Demon Hunters performed",
        parent=supp_parent,
        critical=True,
    )
    # Robust matching for 'K-Pop Demon Hunters' vs 'K-Pop: Demon Hunters'
    kpop_targets = [
        GT["supporting_performers"]["kpop_demon_hunters"],
        "K-Pop: Demon Hunters",
        "K Pop Demon Hunters"
    ]
    kpop_sources, kpop_matched = _find_performer_sources(supp.performers, kpop_targets)
    evaluator.add_custom_node(
        result=bool(kpop_sources),
        id="KPop_Reference",
        desc="Provide URL reference for K-Pop performers",
        parent=kpop_parent,
        critical=True,
    )
    kpop_leaf = evaluator.add_leaf(
        id="KPop_Group",
        desc="K-Pop Demon Hunters participated in the show",
        parent=kpop_parent,
        critical=True,
    )
    claim_kpop = "K-Pop Demon Hunters (also referred to as 'K-Pop: Demon Hunters') participated in the halftime show."
    await evaluator.verify(
        claim=claim_kpop,
        node=kpop_leaf,
        sources=kpop_sources,
        additional_instruction="Treat 'K-Pop Demon Hunters' and 'K-Pop: Demon Hunters' as equivalent naming variants.",
    )

    # Classical_Performers (Bocelli family)
    classical_parent = evaluator.add_parallel(
        id="Classical_Performers",
        desc="Bocelli family performed",
        parent=supp_parent,
        critical=True,
    )
    bocelli_names = list(GT["supporting_performers"]["bocelli_family"])
    bocelli_sources, bocelli_matched = _find_performer_sources(supp.performers, bocelli_names)
    evaluator.add_custom_node(
        result=bool(bocelli_sources),
        id="Bocelli_Reference",
        desc="Provide URL reference for Bocelli performers",
        parent=classical_parent,
        critical=True,
    )
    bocelli_leaf = evaluator.add_leaf(
        id="Bocelli_Family",
        desc="Andrea Bocelli and Matteo Bocelli performed",
        parent=classical_parent,
        critical=True,
    )
    claim_bocelli = "Andrea Bocelli and Matteo Bocelli performed during the halftime show."
    await evaluator.verify(
        claim=claim_bocelli,
        node=bocelli_leaf,
        sources=bocelli_sources,
        additional_instruction="Allow that the page may refer to them collectively as 'the Bocelli family' while naming Andrea and Matteo.",
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
    Entry point for evaluating the Netflix Christmas Day 2025 NFL event details and halftime show.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root strategy matches rubric root
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

    # Top-level event node mirroring rubric root (critical, parallel)
    event_node = evaluator.add_parallel(
        id="Netflix_Christmas_NFL_Event",
        desc="Verification of all details about the Netflix Christmas Day NFL game event",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_event_core_nodes(evaluator, event_node)
    await build_halftime_show_nodes(evaluator, event_node)

    # Return the structured evaluation summary
    return evaluator.get_summary()