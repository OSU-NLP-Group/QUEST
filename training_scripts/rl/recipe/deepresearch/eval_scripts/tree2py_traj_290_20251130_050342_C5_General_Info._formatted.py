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
TASK_ID = "superbowl_lx_2026"
TASK_DESCRIPTION = (
    "I'm interested in attending Super Bowl LX in 2026. Please provide me with comprehensive planning information "
    "including: the exact event date, the name of the stadium hosting the game and its location (city and state), "
    "the stadium's complete street address with ZIP code, the stadium's standard seating capacity, which NFL team "
    "calls this stadium home, which television network will be the primary broadcaster in the United States, and who "
    "will be headlining the halftime show. For each major category of information (event/venue basics, stadium "
    "specifications, and media/entertainment), please provide reference URLs from official or reliable sources."
)

# Expected (ground-truth) values for Super Bowl LX (2026)
EXPECTED_EVENT_NAME = "Super Bowl LX"
EXPECTED_DATE = "February 8, 2026"
EXPECTED_STADIUM = "Levi's Stadium"
EXPECTED_CITY = "Santa Clara"
EXPECTED_STATE = "California"
EXPECTED_FULL_ADDRESS = "4900 Marie P DeBartolo Way, Santa Clara, CA 95054"
EXPECTED_CAPACITY = "68,500"
EXPECTED_HOME_TEAM = "San Francisco 49ers"
EXPECTED_US_BROADCASTER = "NBC"
EXPECTED_HALFTIME_HEADLINER = "Bad Bunny"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventVenueBasics(BaseModel):
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    stadium_name: Optional[str] = None
    stadium_city: Optional[str] = None
    stadium_state: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class StadiumSpecifications(BaseModel):
    full_address: Optional[str] = None
    seating_capacity: Optional[str] = None
    home_team: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class MediaEntertainment(BaseModel):
    us_broadcaster: Optional[str] = None
    halftime_headliner: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class SuperBowlPlanningExtraction(BaseModel):
    event_venue: Optional[EventVenueBasics] = None
    stadium_specs: Optional[StadiumSpecifications] = None
    media_ent: Optional[MediaEntertainment] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superbowl_plan() -> str:
    return """
    Extract the comprehensive planning information for Super Bowl LX (2026) exactly as stated in the answer.

    Return a JSON object with the following nested structure and fields:
    {
      "event_venue": {
        "event_name": string | null,
        "event_date": string | null,
        "stadium_name": string | null,
        "stadium_city": string | null,
        "stadium_state": string | null,
        "references": [url, ...]  // URLs provided by the answer that support the event/venue basics
      },
      "stadium_specs": {
        "full_address": string | null,        // include street, city, state, and ZIP
        "seating_capacity": string | null,    // use formatting as shown (e.g., "68,500")
        "home_team": string | null,
        "references": [url, ...]              // URLs provided by the answer that support stadium specifications
      },
      "media_ent": {
        "us_broadcaster": string | null,      // primary U.S. TV broadcaster
        "halftime_headliner": string | null,  // halftime headliner
        "references": [url, ...]              // URLs provided by the answer that support media/entertainment details
      }
    }

    Rules:
    - Extract only what the answer explicitly states. Do not infer or invent any information.
    - If a field is missing in the answer, set it to null.
    - For references arrays, include only valid URLs that are explicitly present in the answer (plain URLs or markdown links). If no references are provided for a category, return an empty array.
    - Keep the original formatting of strings (e.g., commas in numbers, apostrophes in names).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s if (s is not None and str(s).strip() != "") else "None"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_event_and_venue_checks(
    evaluator: Evaluator,
    parent_node,
    ev: Optional[EventVenueBasics],
) -> None:
    node = evaluator.add_parallel(
        id="Event_and_Venue_Basics",
        desc="Event/venue basics must match required Super Bowl LX details and include supporting references.",
        parent=parent_node,
        critical=True,
    )

    refs = ev.references if ev and ev.references else []

    # Event identification: Super Bowl LX
    event_leaf = evaluator.add_leaf(
        id="Event_Is_Super_Bowl_LX",
        desc="Identifies the event as Super Bowl LX (the 60th Super Bowl).",
        parent=node,
        critical=True,
    )
    claim_event = (
        f"The stated event '{_safe(ev.event_name if ev else None)}' is equivalent to 'Super Bowl LX' (the 60th Super Bowl). "
        f"Treat 'Super Bowl LX' and 'Super Bowl 60' as equivalent."
    )
    await evaluator.verify(
        claim=claim_event,
        node=event_leaf,
        sources=refs,
        additional_instruction="Consider roman numerals (LX) and numeric forms (60) equivalent. Focus on the event identity.",
    )

    # Event date: February 8, 2026
    date_leaf = evaluator.add_leaf(
        id="Event_Date_Is_Feb_8_2026",
        desc="States the event date as February 8, 2026.",
        parent=node,
        critical=True,
    )
    claim_date = (
        f"The stated event date '{_safe(ev.event_date if ev else None)}' equals February 8, 2026 "
        f"(accept minor formatting variants like 'Feb 8, 2026', '02/08/2026')."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=refs,
        additional_instruction="Allow common date format variations: 'February 8, 2026', 'Feb 8, 2026', '02/08/2026'.",
    )

    # Host stadium: Levi's Stadium
    stadium_leaf = evaluator.add_leaf(
        id="Host_Stadium_Is_Levis_Stadium",
        desc="States the host stadium name as Levi's Stadium.",
        parent=node,
        critical=True,
    )
    claim_stadium = (
        f"The stated host stadium '{_safe(ev.stadium_name if ev else None)}' equals 'Levi's Stadium' "
        f"(allow minor punctuation differences like missing apostrophe)."
    )
    await evaluator.verify(
        claim=claim_stadium,
        node=stadium_leaf,
        sources=refs,
        additional_instruction="Treat 'Levis Stadium' and 'Levi's Stadium' as equivalent if clearly the same venue.",
    )

    # Stadium location: Santa Clara, California
    location_leaf = evaluator.add_leaf(
        id="Stadium_Location_Is_Santa_Clara_CA",
        desc="States the stadium location (city, state) as Santa Clara, California.",
        parent=node,
        critical=True,
    )
    stated_loc = f"{_safe(ev.stadium_city if ev else None)}, {_safe(ev.stadium_state if ev else None)}"
    claim_location = (
        f"The stated stadium location '{stated_loc}' equals 'Santa Clara, California' "
        f"(accept 'Santa Clara, CA' as an equivalent)."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=refs,
        additional_instruction="Allow 'CA' abbreviation for California. Focus on city/state correctness.",
    )

    # References provided for event/venue basics
    refs_leaf = evaluator.add_custom_node(
        result=bool(refs),
        id="Event_and_Venue_References_Provided",
        desc="Provides at least one reference URL from official or reliable sources supporting the event/venue basics (event, date, stadium, location).",
        parent=node,
        critical=True,
    )


async def build_stadium_spec_checks(
    evaluator: Evaluator,
    parent_node,
    ss: Optional[StadiumSpecifications],
) -> None:
    node = evaluator.add_parallel(
        id="Stadium_Specifications",
        desc="Stadium specifications must match required values and include supporting references.",
        parent=parent_node,
        critical=True,
    )

    refs = ss.references if ss and ss.references else []

    # Full address
    addr_leaf = evaluator.add_leaf(
        id="Full_Address_Is_4900_Marie_P_DeBartolo_Way_Santa_Clara_CA_95054",
        desc="States the full street address including ZIP code as 4900 Marie P DeBartolo Way, Santa Clara, CA 95054.",
        parent=node,
        critical=True,
    )
    claim_addr = (
        f"The stated full address '{_safe(ss.full_address if ss else None)}' equals "
        f"'4900 Marie P DeBartolo Way, Santa Clara, CA 95054' (allow minor punctuation such as 'P.' vs 'P')."
    )
    await evaluator.verify(
        claim=claim_addr,
        node=addr_leaf,
        sources=refs,
        additional_instruction="Allow minor punctuation/spacing variants in street names (e.g., 'P.' vs 'P'). ZIP code must be 95054.",
    )

    # Seating capacity
    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity_Is_68500",
        desc="States the stadium's standard seating capacity as 68,500.",
        parent=node,
        critical=True,
    )
    claim_capacity = (
        f"The stated standard seating capacity '{_safe(ss.seating_capacity if ss else None)}' equals '68,500' "
        f"(accept numeric formatting variations like '68500')."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=refs,
        additional_instruction="Accept '68,500' and '68500' as equivalent formatting. Focus on standard capacity, not event-specific expansions.",
    )

    # Home team
    home_leaf = evaluator.add_leaf(
        id="Home_Team_Is_SF_49ers",
        desc="Identifies the stadium's home NFL team as the San Francisco 49ers.",
        parent=node,
        critical=True,
    )
    claim_home = (
        f"The stated home NFL team '{_safe(ss.home_team if ss else None)}' equals 'San Francisco 49ers' "
        f"(accept '49ers' or 'SF 49ers' if clearly referring to the same team)."
    )
    await evaluator.verify(
        claim=claim_home,
        node=home_leaf,
        sources=refs,
        additional_instruction="Allow synonyms: '49ers', 'SF 49ers', 'San Francisco 49ers' treated as the same team.",
    )

    # References provided for stadium specifications
    refs_leaf = evaluator.add_custom_node(
        result=bool(refs),
        id="Stadium_Specifications_References_Provided",
        desc="Provides at least one reference URL from official or reliable sources supporting the stadium specifications (address, capacity, home team).",
        parent=node,
        critical=True,
    )


async def build_media_ent_checks(
    evaluator: Evaluator,
    parent_node,
    me: Optional[MediaEntertainment],
) -> None:
    node = evaluator.add_parallel(
        id="Media_and_Entertainment",
        desc="Media/entertainment details must match required values and include supporting references.",
        parent=parent_node,
        critical=True,
    )

    refs = me.references if me and me.references else []

    # US broadcaster: NBC
    broadcaster_leaf = evaluator.add_leaf(
        id="Primary_US_Broadcaster_Is_NBC",
        desc="Identifies the primary U.S. television broadcaster as NBC.",
        parent=node,
        critical=True,
    )
    claim_broadcaster = (
        f"The stated primary U.S. broadcaster '{_safe(me.us_broadcaster if me else None)}' equals 'NBC' "
        f"(accept 'NBC network' or 'NBCUniversal' if clearly referring to NBC as the TV broadcaster)."
    )
    await evaluator.verify(
        claim=claim_broadcaster,
        node=broadcaster_leaf,
        sources=refs,
        additional_instruction="Focus on the primary U.S. television broadcaster; 'NBC' is required. Streaming (e.g., Peacock) is not a substitute for NBC.",
    )

    # Halftime headliner: Bad Bunny
    halftime_leaf = evaluator.add_leaf(
        id="Halftime_Headliner_Is_Bad_Bunny",
        desc="Identifies the halftime show headliner as Bad Bunny.",
        parent=node,
        critical=True,
    )
    claim_halftime = (
        f"The stated halftime headliner '{_safe(me.halftime_headliner if me else None)}' equals 'Bad Bunny' "
        f"(accept the legal name 'Benito Antonio Martínez Ocasio' as equivalent to 'Bad Bunny')."
    )
    await evaluator.verify(
        claim=claim_halftime,
        node=halftime_leaf,
        sources=refs,
        additional_instruction="Treat the stage name 'Bad Bunny' and legal name 'Benito Antonio Martínez Ocasio' as equivalent when clearly referring to the artist.",
    )

    # References provided for media/entertainment
    refs_leaf = evaluator.add_custom_node(
        result=bool(refs),
        id="Media_and_Entertainment_References_Provided",
        desc="Provides at least one reference URL from official or reliable sources supporting the media/entertainment details (broadcaster, halftime headliner).",
        parent=node,
        critical=True,
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
    Evaluate the answer for the Super Bowl LX (2026) planning guide.
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_superbowl_plan(),
        template_class=SuperBowlPlanningExtraction,
        extraction_name="superbowl_lx_planning_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_event_name": EXPECTED_EVENT_NAME,
        "expected_date": EXPECTED_DATE,
        "expected_stadium": EXPECTED_STADIUM,
        "expected_city_state": f"{EXPECTED_CITY}, {EXPECTED_STATE}",
        "expected_full_address": EXPECTED_FULL_ADDRESS,
        "expected_capacity": EXPECTED_CAPACITY,
        "expected_home_team": EXPECTED_HOME_TEAM,
        "expected_us_broadcaster": EXPECTED_US_BROADCASTER,
        "expected_halftime_headliner": EXPECTED_HALFTIME_HEADLINER,
    }, gt_type="expected_values")

    # Top-level critical node (as per rubric)
    top_node = evaluator.add_parallel(
        id="Super_Bowl_LX_Planning_Guide",
        desc="Comprehensive planning information for Super Bowl LX (2026) with required exact values and per-category references.",
        parent=root,
        critical=True,
    )

    # Build category verifications
    await build_event_and_venue_checks(evaluator, top_node, extracted.event_venue)
    await build_stadium_spec_checks(evaluator, top_node, extracted.stadium_specs)
    await build_media_ent_checks(evaluator, top_node, extracted.media_ent)

    # Return final summary
    return evaluator.get_summary()