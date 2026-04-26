import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "super_bowl_lx_2026"
TASK_DESCRIPTION = "Provide comprehensive information about Super Bowl LX in 2026 by answering the following: What is the date of the game? What stadium will host it, and in which city is that stadium located? What is the stadium's regular seating capacity? Who will headline the halftime show, and how many Grammy Awards has this performer won? Which network will broadcast the game, and what is the kickoff time (ET)? Additionally, provide details about these official Super Bowl LX Week events: (1) NFL Honors presented by Invisalign - what venue and date? (2) Pro Bowl Games powered by Verizon - what venue and date? (3) Super Bowl Opening Night fueled by Gatorade - what venue and date? (4) BAHC Live! Concert Series on February 7, 2026 - which headlining artist and what venue? (5) Taste of the NFL - what venue and date?"


EXPECTED_FACTS = {
    "game_date": "February 8, 2026",
    "kickoff_time_et": "6:30 p.m. ET",
    "broadcast_network": "NBC",
    "stadium_name": "Levi's Stadium",
    "stadium_city": "Santa Clara, California",
    "stadium_capacity": "68,500",
    "halftime_performer": "Bad Bunny",
    "performer_grammy_count": "3",
    "events": {
        "nfl_honors": {
            "venue": "Palace of Fine Arts",
            "date": "Thursday, February 5, 2026",
        },
        "pro_bowl_games": {
            "venue": "Moscone Center South Building",
            "date": "Tuesday, February 3, 2026",
        },
        "opening_night": {
            "venue": "San Jose Convention Center",
            "date": "Monday, February 2, 2026",
        },
        "bahc_live": {
            "venue": "Bill Graham Civic Auditorium",
            "date": "Saturday, February 7, 2026",
            "artist": "Chris Stapleton",
        },
        "taste_of_nfl": {
            "venue": "The Hibernia",
            "date": "Saturday, February 7, 2026",
        },
    }
}


# ------------------------- Data Models ------------------------- #
class GameInfo(BaseModel):
    date: Optional[str] = None
    kickoff_time_et: Optional[str] = None
    broadcast_network: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class StadiumInfo(BaseModel):
    stadium_name: Optional[str] = None
    stadium_city: Optional[str] = None  # include state if present in the answer (e.g., "Santa Clara, California")
    regular_seating_capacity: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class HalftimeInfo(BaseModel):
    performer_name: Optional[str] = None
    grammy_award_count: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class EventInfo(BaseModel):
    venue: Optional[str] = None
    date: Optional[str] = None
    headliner: Optional[str] = None  # only applicable for BAHC Live! Concert Series
    source_urls: List[str] = Field(default_factory=list)


class SuperBowlLXExtraction(BaseModel):
    game: Optional[GameInfo] = None
    stadium: Optional[StadiumInfo] = None
    halftime: Optional[HalftimeInfo] = None
    event_nfl_honors: Optional[EventInfo] = None
    event_pro_bowl_games: Optional[EventInfo] = None
    event_opening_night: Optional[EventInfo] = None
    event_bahc_live_concert: Optional[EventInfo] = None
    event_taste_of_nfl: Optional[EventInfo] = None


# ------------------------- Extraction Prompt ------------------------- #
def prompt_extract_super_bowl_info() -> str:
    return """
    Extract all the requested Super Bowl LX (2026) information exactly as stated in the provided answer text. Return a JSON object with the following nested structure:

    1) game:
       - date: The date of the game as stated in the answer (e.g., "February 8, 2026").
       - kickoff_time_et: The kickoff time in ET as stated (e.g., "6:30 p.m. ET").
       - broadcast_network: The primary broadcast network (e.g., "NBC").
       - source_urls: An array of URLs cited in the answer that specifically support the game date/time/network. Extract explicit URLs only (plain or markdown), avoid duplicates.

    2) stadium:
       - stadium_name: The stadium name (e.g., "Levi's Stadium").
       - stadium_city: The city and state where the stadium is located, as stated (e.g., "Santa Clara, California").
       - regular_seating_capacity: The stadium's regular seating capacity as stated (e.g., "68,500").
       - source_urls: An array of URLs cited in the answer supporting the stadium identity/location/capacity.

    3) halftime:
       - performer_name: The halftime show headliner (e.g., "Bad Bunny").
       - grammy_award_count: The count of Grammy Awards this performer has won as stated (e.g., "3").
       - source_urls: An array of URLs cited in the answer that support the halftime headliner and Grammy count.

    4) event_nfl_honors:
       - venue: Venue for NFL Honors presented by Invisalign.
       - date: Date for NFL Honors.
       - source_urls: URLs cited in the answer supporting the venue/date.

    5) event_pro_bowl_games:
       - venue: Venue for Pro Bowl Games powered by Verizon.
       - date: Date for the Pro Bowl Games.
       - source_urls: URLs cited in the answer supporting the venue/date.

    6) event_opening_night:
       - venue: Venue for Super Bowl Opening Night fueled by Gatorade.
       - date: Date for Opening Night.
       - source_urls: URLs cited in the answer supporting the venue/date.

    7) event_bahc_live_concert:
       - venue: Venue for BAHC Live! Concert Series on February 7, 2026.
       - date: Date (should be February 7, 2026).
       - headliner: Headlining artist for the concert (e.g., "Chris Stapleton").
       - source_urls: URLs cited in the answer supporting the headliner and venue.

    8) event_taste_of_nfl:
       - venue: Venue for Taste of the NFL.
       - date: Date for Taste of the NFL.
       - source_urls: URLs cited in the answer supporting the venue/date.

    Rules:
    - If any requested field is not mentioned in the answer, return null for that field.
    - For URL fields, extract only explicit URLs present in the answer text. If URLs are not provided, return an empty list for source_urls.
    - Do not invent or infer any information beyond the answer.
    """


# ------------------------- Helper ------------------------- #
def _sources_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    # Deduplicate while preserving order
    seen = set()
    cleaned = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned if cleaned else None


# ------------------------- Verification Builders ------------------------- #
async def build_game_info_checks(evaluator: Evaluator,
                                 parent,
                                 extracted: SuperBowlLXExtraction) -> None:
    node = evaluator.add_parallel(
        id="Game_Info",
        desc="Core game details match required constraints",
        parent=parent,
        critical=True
    )
    game_sources = _sources_or_none(extracted.game.source_urls if extracted.game else None)

    # Game Date
    leaf_date = evaluator.add_leaf(
        id="Game_Date",
        desc="Game date is February 8, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Super Bowl LX (2026) is scheduled for February 8, 2026.",
        node=leaf_date,
        sources=game_sources,
        additional_instruction="Accept minor formatting variations (e.g., 'Feb. 8, 2026'). Reject if the page is about a different year or event."
    )

    # Kickoff Time
    leaf_ko = evaluator.add_leaf(
        id="Kickoff_Time",
        desc="Kickoff time is 6:30 p.m. ET",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The kickoff time for Super Bowl LX is 6:30 p.m. ET.",
        node=leaf_ko,
        sources=game_sources,
        additional_instruction="Allow '6:30 PM ET' or '6:30 Eastern Time' as equivalent. If multiple times are shown, the primary national broadcast kickoff should be 6:30 ET."
    )

    # Broadcast Network
    leaf_net = evaluator.add_leaf(
        id="Broadcast_Network",
        desc="Primary broadcast network is NBC",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="NBC is the primary broadcast network for Super Bowl LX.",
        node=leaf_net,
        sources=game_sources,
        additional_instruction="Streaming references (e.g., Peacock) may be present; ensure NBC is the primary broadcast TV network."
    )


async def build_stadium_info_checks(evaluator: Evaluator,
                                    parent,
                                    extracted: SuperBowlLXExtraction) -> None:
    node = evaluator.add_parallel(
        id="Stadium_Info",
        desc="Stadium identity, location, and capacity match required constraints",
        parent=parent,
        critical=True
    )
    stadium_sources = _sources_or_none(extracted.stadium.source_urls if extracted.stadium else None)

    # Stadium Name
    leaf_name = evaluator.add_leaf(
        id="Stadium_Name",
        desc="Stadium name is Levi's Stadium",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Super Bowl LX will be hosted at Levi's Stadium.",
        node=leaf_name,
        sources=stadium_sources,
        additional_instruction="Allow possessive/apostrophe variants ('Levi’s' vs 'Levi's'). Reject if a different stadium is indicated."
    )

    # Stadium City
    leaf_city = evaluator.add_leaf(
        id="Stadium_City",
        desc="Stadium location city is Santa Clara, California",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Levi's Stadium is located in Santa Clara, California.",
        node=leaf_city,
        sources=stadium_sources,
        additional_instruction="Accept formatting variants like 'Santa Clara, CA'. City and state must be correct."
    )

    # Stadium Capacity
    leaf_cap = evaluator.add_leaf(
        id="Stadium_Capacity",
        desc="Regular seating capacity is 68,500",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The regular seating capacity of Levi's Stadium is 68,500.",
        node=leaf_cap,
        sources=stadium_sources,
        additional_instruction="Accept minor phrasing variations (e.g., 'approximately 68,500'). Event-day capacity expansions do not count; verify regular seating capacity."
    )


async def build_halftime_info_checks(evaluator: Evaluator,
                                     parent,
                                     extracted: SuperBowlLXExtraction) -> None:
    node = evaluator.add_parallel(
        id="Halftime_Show_Info",
        desc="Halftime headliner and Grammy count match required constraints",
        parent=parent,
        critical=True
    )
    halftime_sources = _sources_or_none(extracted.halftime.source_urls if extracted.halftime else None)

    # Performer
    leaf_perf = evaluator.add_leaf(
        id="Halftime_Performer",
        desc="Halftime show headliner is Bad Bunny",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Bad Bunny is the headlining performer for the Super Bowl LX halftime show.",
        node=leaf_perf,
        sources=halftime_sources,
        additional_instruction="Allow minor name formatting variations. Ensure this is for Super Bowl LX (2026), not a prior year."
    )

    # Grammy Count
    leaf_grammys = evaluator.add_leaf(
        id="Performer_Grammy_Count",
        desc="Bad Bunny Grammy Award count is 3",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Bad Bunny has won 3 Grammy Awards.",
        node=leaf_grammys,
        sources=halftime_sources,
        additional_instruction="Count only Recording Academy Grammy Awards (not Latin Grammy Awards) unless the source explicitly clarifies equivalence."
    )


async def build_event_checks(evaluator: Evaluator,
                             parent,
                             extracted: SuperBowlLXExtraction) -> None:
    # NFL Honors
    honors_node = evaluator.add_parallel(
        id="Event_1_NFL_Honors",
        desc="NFL Honors presented by Invisalign: correct venue and date",
        parent=parent,
        critical=True
    )
    honors_sources = _sources_or_none(extracted.event_nfl_honors.source_urls if extracted.event_nfl_honors else None)

    leaf_honors_venue = evaluator.add_leaf(
        id="NFL_Honors_Venue",
        desc="NFL Honors venue is Palace of Fine Arts",
        parent=honors_node,
        critical=True
    )
    await evaluator.verify(
        claim="NFL Honors presented by Invisalign will be held at the Palace of Fine Arts.",
        node=leaf_honors_venue,
        sources=honors_sources,
        additional_instruction="Accept 'Palace of Fine Arts Theatre' as equivalent."
    )

    leaf_honors_date = evaluator.add_leaf(
        id="NFL_Honors_Date",
        desc="NFL Honors date is Thursday, February 5, 2026",
        parent=honors_node,
        critical=True
    )
    await evaluator.verify(
        claim="NFL Honors presented by Invisalign will take place on Thursday, February 5, 2026.",
        node=leaf_honors_date,
        sources=honors_sources,
        additional_instruction="Day-of-week must align with the date; allow abbreviated formats (e.g., 'Thu, Feb 5, 2026')."
    )

    # Pro Bowl Games
    pro_node = evaluator.add_parallel(
        id="Event_2_Pro_Bowl_Games",
        desc="Pro Bowl Games powered by Verizon: correct venue and date",
        parent=parent,
        critical=True
    )
    pro_sources = _sources_or_none(extracted.event_pro_bowl_games.source_urls if extracted.event_pro_bowl_games else None)

    leaf_pro_venue = evaluator.add_leaf(
        id="Pro_Bowl_Venue",
        desc="Pro Bowl Games venue is Moscone Center South Building",
        parent=pro_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Pro Bowl Games powered by Verizon will be held at the Moscone Center South Building.",
        node=leaf_pro_venue,
        sources=pro_sources,
        additional_instruction="Accept variants like 'Moscone South' if clearly referring to the South Building."
    )

    leaf_pro_date = evaluator.add_leaf(
        id="Pro_Bowl_Date",
        desc="Pro Bowl Games date is Tuesday, February 3, 2026",
        parent=pro_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Pro Bowl Games powered by Verizon will take place on Tuesday, February 3, 2026.",
        node=leaf_pro_date,
        sources=pro_sources,
        additional_instruction="Allow abbreviated formats (e.g., 'Tue, Feb 3, 2026')."
    )

    # Opening Night
    opening_node = evaluator.add_parallel(
        id="Event_3_Opening_Night",
        desc="Super Bowl Opening Night fueled by Gatorade: correct venue and date",
        parent=parent,
        critical=True
    )
    opening_sources = _sources_or_none(extracted.event_opening_night.source_urls if extracted.event_opening_night else None)

    leaf_opening_venue = evaluator.add_leaf(
        id="Opening_Night_Venue",
        desc="Opening Night venue is San Jose Convention Center",
        parent=opening_node,
        critical=True
    )
    await evaluator.verify(
        claim="Super Bowl Opening Night fueled by Gatorade will be held at the San Jose Convention Center.",
        node=leaf_opening_venue,
        sources=opening_sources,
        additional_instruction="Accept variants like 'SJCC' if clearly referring to the San Jose Convention Center."
    )

    leaf_opening_date = evaluator.add_leaf(
        id="Opening_Night_Date",
        desc="Opening Night date is Monday, February 2, 2026",
        parent=opening_node,
        critical=True
    )
    await evaluator.verify(
        claim="Super Bowl Opening Night fueled by Gatorade will take place on Monday, February 2, 2026.",
        node=leaf_opening_date,
        sources=opening_sources,
        additional_instruction="Allow abbreviated formats (e.g., 'Mon, Feb 2, 2026')."
    )

    # BAHC Live! Concert Series
    bahc_node = evaluator.add_parallel(
        id="Event_4_BAHC_Live_Concert_Series",
        desc="BAHC Live! Concert Series on February 7, 2026: correct headlining artist and venue",
        parent=parent,
        critical=True
    )
    bahc_sources = _sources_or_none(extracted.event_bahc_live_concert.source_urls if extracted.event_bahc_live_concert else None)

    leaf_bahc_artist = evaluator.add_leaf(
        id="Concert_Artist",
        desc="Headlining artist is Chris Stapleton",
        parent=bahc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Chris Stapleton is the headlining artist for the BAHC Live! Concert Series on February 7, 2026.",
        node=leaf_bahc_artist,
        sources=bahc_sources,
        additional_instruction="Ensure the headliner and date match the concert series described."
    )

    leaf_bahc_venue = evaluator.add_leaf(
        id="Concert_Venue",
        desc="Concert venue is Bill Graham Civic Auditorium",
        parent=bahc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The BAHC Live! Concert Series on February 7, 2026 will be held at the Bill Graham Civic Auditorium.",
        node=leaf_bahc_venue,
        sources=bahc_sources,
        additional_instruction="Allow variants like 'Bill Graham Auditorium' if clearly referring to the same venue."
    )

    # Taste of the NFL
    taste_node = evaluator.add_parallel(
        id="Event_5_Taste_of_the_NFL",
        desc="Taste of the NFL: correct venue and date",
        parent=parent,
        critical=True
    )
    taste_sources = _sources_or_none(extracted.event_taste_of_nfl.source_urls if extracted.event_taste_of_nfl else None)

    leaf_taste_venue = evaluator.add_leaf(
        id="Taste_NFL_Venue",
        desc="Taste of the NFL venue is The Hibernia",
        parent=taste_node,
        critical=True
    )
    await evaluator.verify(
        claim="Taste of the NFL will be held at The Hibernia.",
        node=leaf_taste_venue,
        sources=taste_sources,
        additional_instruction="Accept 'Hibernia Bank Building' if it clearly refers to the same venue."
    )

    leaf_taste_date = evaluator.add_leaf(
        id="Taste_NFL_Date",
        desc="Taste of the NFL date is Saturday, February 7, 2026",
        parent=taste_node,
        critical=True
    )
    await evaluator.verify(
        claim="Taste of the NFL will take place on Saturday, February 7, 2026.",
        node=leaf_taste_date,
        sources=taste_sources,
        additional_instruction="Allow abbreviated formats (e.g., 'Sat, Feb 7, 2026')."
    )


# ------------------------- Main Evaluation Entry ------------------------- #
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_super_bowl_info(),
        template_class=SuperBowlLXExtraction,
        extraction_name="super_bowl_lx_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected": EXPECTED_FACTS
    }, gt_type="expected_facts")

    # Build the verification tree according to rubric
    top = evaluator.add_parallel(
        id="Super_Bowl_LX_2026_Information",
        desc="Comprehensive factual information about Super Bowl LX 2026 and associated official events, meeting all stated constraints",
        parent=root,
        critical=True
    )

    await build_game_info_checks(evaluator, top, extracted)
    await build_stadium_info_checks(evaluator, top, extracted)
    await build_halftime_info_checks(evaluator, top, extracted)

    events_parent = evaluator.add_parallel(
        id="Official_Super_Bowl_LX_Week_Events",
        desc="All listed official Super Bowl LX Week events have correct venue and date (or artist/venue where applicable) per constraints",
        parent=top,
        critical=True
    )
    await build_event_checks(evaluator, events_parent, extracted)

    return evaluator.get_summary()