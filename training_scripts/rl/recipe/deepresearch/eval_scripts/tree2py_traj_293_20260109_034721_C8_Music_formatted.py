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
TASK_ID = "us_music_festivals_2024"
TASK_DESCRIPTION = (
    "Identify four major music festivals that took place in the United States during 2024, where each festival occurred "
    "in a different state and meets the following specific criteria:\n\n"
    "Festival 1: A festival that occurred in California during the spring months (March, April, or May), featured at least "
    "one headliner who was nominated for Album of the Year at the 2024 Grammy Awards, and took place at an outdoor venue.\n\n"
    "Festival 2: A festival that occurred in Illinois during the summer months (June, July, or August), took place in an "
    "urban park in Chicago, and spanned exactly 4 days.\n\n"
    "Festival 3: A festival that occurred in Tennessee during June, took place at a farm or camping-style venue (not in an "
    "urban setting), and featured rock or alternative music headliners.\n\n"
    "Festival 4: A festival that occurred in Texas during the fall months (September, October, or November), took place over "
    "two separate weekends, and featured country, Americana, or roots music artists among its headliners.\n\n"
    "For each festival, provide: (1) the festival name, (2) the specific dates it occurred, (3) the exact venue or location name, "
    "(4) at least one headliner name, and (5) a reference URL that confirms this information."
)

SPRING_MONTHS = {"March", "April", "May"}
SUMMER_MONTHS = {"June", "July", "August"}
FALL_MONTHS = {"September", "October", "November"}

# Reference list of 2024 Grammy Album of the Year nominees (to assist simple verification)
GRAMMY_2024_AOTY_NOMINEES = {
    "Taylor Swift",
    "Olivia Rodrigo",
    "Miley Cyrus",
    "SZA",
    "Lana Del Rey",
    "boygenius",
    "Jon Batiste",
    "Janelle Monáe",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    name: Optional[str] = None
    dates_text: Optional[str] = None  # e.g., "April 12–14, 2024" or "June 13-16, 2024"
    venue_name: Optional[str] = None  # exact venue or location
    city: Optional[str] = None
    state: Optional[str] = None
    headliners: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class FestivalsExtraction(BaseModel):
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return (
        "Extract up to four festival entries described in the answer. For each festival, return the following fields:\n"
        "1) name: The festival name as written in the answer.\n"
        "2) dates_text: The specific 2024 dates mentioned (e.g., 'April 12–14, 2024'). If not specified, return null.\n"
        "3) venue_name: The exact venue/location name (e.g., 'Empire Polo Club', 'Grant Park'). If not specified, return null.\n"
        "4) city: The city name, if provided.\n"
        "5) state: The state name or standard two-letter abbreviation, if provided.\n"
        "6) headliners: A list of at least one headliner name mentioned for the festival. If none are mentioned, return an empty list.\n"
        "7) reference_urls: All URLs explicitly cited that corroborate this festival's information (the page(s) should ideally confirm name, dates, venue, and lineup/headliners). "
        "Include full URLs; if no URLs are provided for this festival, return an empty list.\n\n"
        "Return a JSON object with a single key 'festivals' that is an array of these festival objects. If more than four are mentioned, include only the first four. "
        "If fewer than four are mentioned, include the available ones."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _festival_or_placeholder(festivals: List[FestivalItem], idx: int) -> FestivalItem:
    return festivals[idx] if idx < len(festivals) else FestivalItem()


def _format_headliners_for_claim(headliners: List[str]) -> str:
    if not headliners:
        return "[]"
    return ", ".join([h.strip() for h in headliners if h and h.strip()])


def _reference_claim(f: FestivalItem) -> str:
    hl = _format_headliners_for_claim(f.headliners)
    return (
        f"The referenced page(s) explicitly corroborate for the 2024 edition: "
        f"(a) the festival name '{f.name}', (b) the specific dates '{f.dates_text}', "
        f"(c) the exact venue/location '{f.venue_name}', and (d) at least one headliner among [{hl}]. "
        f"The page(s) should be about the 2024 festival edition and the information should be clearly stated."
    )


# --------------------------------------------------------------------------- #
# Verification builders for each festival                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_festival_1(evaluator: Evaluator, parent_node, fest: FestivalItem) -> None:
    f_node = evaluator.add_parallel(
        id="Festival_1",
        desc="Festival 1 meets California + Spring 2024 + outdoor + 2024 Grammy AOTY-nominated headliner constraints and required fields.",
        parent=parent_node,
        critical=False
    )

    # Field existence checks (critical)
    evaluator.add_custom_node(
        result=bool(fest.name and fest.name.strip()),
        id="F1_Name_Provided",
        desc="Festival name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.dates_text and fest.dates_text.strip()),
        id="F1_Dates_Provided",
        desc="Specific festival dates are provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.venue_name and fest.venue_name.strip()),
        id="F1_Venue_Name_Provided",
        desc="Exact venue or location name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.headliners),
        id="F1_Headliner_Provided",
        desc="At least one headliner name is provided.",
        parent=f_node,
        critical=True
    )

    # Major/large-scale characterization
    n = evaluator.add_leaf(
        id="F1_Major_Objective",
        desc="At least one provided reference explicitly characterizes the festival as a major/large-scale festival (e.g., uses terms like “major”, “one of the largest”, “premier”, “flagship”).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="This festival is explicitly characterized as major/large-scale on at least one provided reference page.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Look for explicit descriptors such as 'major', 'one of the largest', 'premier', 'flagship', 'large-scale', or clearly implied massive attendance figures. The wording should be about the festival itself."
    )

    # State: California
    n = evaluator.add_leaf(
        id="F1_State_CA",
        desc="Festival occurred in California.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 edition of the festival took place in the U.S. state of California.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Confirm via location details on the page (city, venue, state). For example, Indio, CA or other California cities/venues."
    )

    # Timing: Spring 2024 (March/April/May)
    n = evaluator.add_leaf(
        id="F1_Timing_Spring_2024",
        desc="Festival occurred during March, April, or May of 2024.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="All official 2024 dates for the festival fall within March, April, or May 2024.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Use the 2024 dates shown on the page to determine the month(s). All festival days should be within March, April, or May 2024."
    )

    # Venue is outdoor
    n = evaluator.add_leaf(
        id="F1_Venue_Outdoor",
        desc="Festival took place at an outdoor venue.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 festival was held primarily outdoors (e.g., park, fields, fairgrounds, open-air venue).",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Accept parks, open-air fields, fairgrounds, polo clubs, etc. If the main stages and experience are outdoors, this passes."
    )

    # Headliner includes a 2024 Grammy AOTY nominee (simple verification allowed)
    n = evaluator.add_leaf(
        id="F1_Headliner_GrammyAOTY_Nominee_2024",
        desc="At least one listed headliner was nominated for Album of the Year at the 2024 Grammy Awards.",
        parent=f_node,
        critical=True
    )
    headliner_list = _format_headliners_for_claim(fest.headliners)
    await evaluator.verify(
        claim=f"At least one of the listed headliners [{headliner_list}] was nominated for Album of the Year at the 2024 Grammy Awards.",
        node=n,
        sources=None,
        additional_instruction=(
            "Use your general knowledge of the 2024 Grammy AOTY nominees to judge. The nominees include: "
            + ", ".join(sorted(GRAMMY_2024_AOTY_NOMINEES))
            + ". Consider reasonable name variants and case-insensitivity."
        )
    )

    # Reference URL corroborates name/dates/venue/headliner
    n = evaluator.add_leaf(
        id="F1_Reference_URL_Corroborates",
        desc="At least one reference URL is provided that corroborates the provided festival name, dates, venue/location, and at least one headliner (and supports the applicable constraints).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim=_reference_claim(fest),
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Prefer a single page that confirms all required fields for the 2024 edition. Minor formatting differences are acceptable."
    )


async def build_and_verify_festival_2(evaluator: Evaluator, parent_node, fest: FestivalItem) -> None:
    f_node = evaluator.add_parallel(
        id="Festival_2",
        desc="Festival 2 meets Illinois + Summer 2024 + urban park in Chicago + exactly 4 days constraints and required fields.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(fest.name and fest.name.strip()),
        id="F2_Name_Provided",
        desc="Festival name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.dates_text and fest.dates_text.strip()),
        id="F2_Dates_Provided",
        desc="Specific festival dates are provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.venue_name and fest.venue_name.strip()),
        id="F2_Venue_Name_Provided",
        desc="Exact venue or location name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.headliners),
        id="F2_Headliner_Provided",
        desc="At least one headliner name is provided.",
        parent=f_node,
        critical=True
    )

    n = evaluator.add_leaf(
        id="F2_Major_Objective",
        desc="At least one provided reference explicitly characterizes the festival as a major/large-scale festival (e.g., uses terms like “major”, “one of the largest”, “premier”, “flagship”).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="This festival is explicitly characterized as major/large-scale on at least one provided reference page.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Look for explicit descriptors or strong implications of large scale and prominence."
    )

    n = evaluator.add_leaf(
        id="F2_State_IL",
        desc="Festival occurred in Illinois.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 edition of the festival took place in Illinois, USA.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Confirm via location details on the page (city, venue, state). Chicago, IL qualifies."
    )

    n = evaluator.add_leaf(
        id="F2_Timing_Summer_2024",
        desc="Festival occurred during June, July, or August of 2024.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="All official 2024 dates for the festival fall within June, July, or August 2024.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Use the 2024 dates shown on the page to determine the month(s)."
    )

    n = evaluator.add_leaf(
        id="F2_Venue_UrbanPark_Chicago",
        desc="Festival took place in an urban park in Chicago.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 festival took place in an urban park located within the city of Chicago (e.g., Grant Park).",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="The venue must be an urban park within Chicago city limits."
    )

    n = evaluator.add_leaf(
        id="F2_Duration_Exactly_4_Days",
        desc="Festival spanned exactly 4 days.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 festival spanned exactly four (4) calendar days.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Count the number of days indicated by the dates or schedule on the page. Exactly four days are required."
    )

    n = evaluator.add_leaf(
        id="F2_Reference_URL_Corroborates",
        desc="At least one reference URL is provided that corroborates the provided festival name, dates, venue/location, and at least one headliner (and supports the applicable constraints).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim=_reference_claim(fest),
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Prefer a single page that confirms all required fields for the 2024 edition."
    )


async def build_and_verify_festival_3(evaluator: Evaluator, parent_node, fest: FestivalItem) -> None:
    f_node = evaluator.add_parallel(
        id="Festival_3",
        desc="Festival 3 meets Tennessee + June 2024 + farm/camping (non-urban) + rock/alternative headliner constraints and required fields.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(fest.name and fest.name.strip()),
        id="F3_Name_Provided",
        desc="Festival name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.dates_text and fest.dates_text.strip()),
        id="F3_Dates_Provided",
        desc="Specific festival dates are provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.venue_name and fest.venue_name.strip()),
        id="F3_Venue_Name_Provided",
        desc="Exact venue or location name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.headliners),
        id="F3_Headliner_Provided",
        desc="At least one headliner name is provided.",
        parent=f_node,
        critical=True
    )

    n = evaluator.add_leaf(
        id="F3_Major_Objective",
        desc="At least one provided reference explicitly characterizes the festival as a major/large-scale festival (e.g., uses terms like “major”, “one of the largest”, “premier”, “flagship”).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="This festival is explicitly characterized as major/large-scale on at least one provided reference page.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Look for explicit descriptors or strong implications of large scale and prominence."
    )

    n = evaluator.add_leaf(
        id="F3_State_TN",
        desc="Festival occurred in Tennessee.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 edition of the festival took place in Tennessee, USA.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Confirm via location details on the page (city, venue, state)."
    )

    n = evaluator.add_leaf(
        id="F3_Timing_June_2024",
        desc="Festival occurred in June 2024.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="All official 2024 dates for the festival fall within June 2024.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Use the 2024 dates shown on the page to confirm June."
    )

    n = evaluator.add_leaf(
        id="F3_Venue_FarmOrCamping_NonUrban",
        desc="Festival took place at a farm or camping-style venue (not an urban setting).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 festival took place at a farm or camping-style venue outside an urban setting.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Look for terms like farm, campground, camping, rural location (e.g., Great Stage Park near Manchester, TN), or similar."
    )

    n = evaluator.add_leaf(
        id="F3_Headliner_Genre_RockOrAlternative",
        desc="Festival featured rock or alternative music headliners.",
        parent=f_node,
        critical=True
    )
    headliner_list = _format_headliners_for_claim(fest.headliners)
    await evaluator.verify(
        claim=f"At least one of the listed headliners [{headliner_list}] is a rock or alternative artist.",
        node=n,
        sources=None,
        additional_instruction="Use general knowledge of artist genres; accept rock, alternative rock, indie rock, alt-pop, post-punk, etc. Minor classification nuances are acceptable."
    )

    n = evaluator.add_leaf(
        id="F3_Reference_URL_Corroborates",
        desc="At least one reference URL is provided that corroborates the provided festival name, dates, venue/location, and at least one headliner (and supports the applicable constraints).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim=_reference_claim(fest),
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Prefer a single page that confirms all required fields for the 2024 edition."
    )


async def build_and_verify_festival_4(evaluator: Evaluator, parent_node, fest: FestivalItem) -> None:
    f_node = evaluator.add_parallel(
        id="Festival_4",
        desc="Festival 4 meets Texas + Fall 2024 + two separate weekends + country/Americana/roots headliner constraints and required fields.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(fest.name and fest.name.strip()),
        id="F4_Name_Provided",
        desc="Festival name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.dates_text and fest.dates_text.strip()),
        id="F4_Dates_Provided",
        desc="Specific festival dates are provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.venue_name and fest.venue_name.strip()),
        id="F4_Venue_Name_Provided",
        desc="Exact venue or location name is provided.",
        parent=f_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest.headliners),
        id="F4_Headliner_Provided",
        desc="At least one headliner name is provided.",
        parent=f_node,
        critical=True
    )

    n = evaluator.add_leaf(
        id="F4_Major_Objective",
        desc="At least one provided reference explicitly characterizes the festival as a major/large-scale festival (e.g., uses terms like “major”, “one of the largest”, “premier”, “flagship”).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="This festival is explicitly characterized as major/large-scale on at least one provided reference page.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Look for explicit descriptors or strong implications of large scale and prominence."
    )

    n = evaluator.add_leaf(
        id="F4_State_TX",
        desc="Festival occurred in Texas.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 edition of the festival took place in Texas, USA.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Confirm via location details on the page (city, venue, state). Austin, TX qualifies."
    )

    n = evaluator.add_leaf(
        id="F4_Timing_Fall_2024",
        desc="Festival occurred during September, October, or November of 2024.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="All official 2024 dates for the festival fall within September, October, or November 2024.",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Use the 2024 dates shown on the page to determine the month(s)."
    )

    n = evaluator.add_leaf(
        id="F4_Two_Separate_Weekends",
        desc="Festival took place over two separate weekends.",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 festival took place over two separate weekends (e.g., Weekend 1 and Weekend 2).",
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Confirm the presence of two distinct weekends; they may be consecutive weeks but must be separate weekend runs."
    )

    n = evaluator.add_leaf(
        id="F4_Headliner_Genre_CountryAmericanaRoots",
        desc="Festival featured country, Americana, or roots music artists among its headliners.",
        parent=f_node,
        critical=True
    )
    headliner_list = _format_headliners_for_claim(fest.headliners)
    await evaluator.verify(
        claim=f"At least one of the listed headliners [{headliner_list}] is a country, Americana, or roots artist.",
        node=n,
        sources=None,
        additional_instruction="Use general knowledge of artist genres; accept country, Americana, roots, folk-country, etc. Minor classification nuances are acceptable."
    )

    n = evaluator.add_leaf(
        id="F4_Reference_URL_Corroborates",
        desc="At least one reference URL is provided that corroborates the provided festival name, dates, venue/location, and at least one headliner (and supports the applicable constraints).",
        parent=f_node,
        critical=True
    )
    await evaluator.verify(
        claim=_reference_claim(fest),
        node=n,
        sources=fest.reference_urls,
        additional_instruction="Prefer a single page that confirms all required fields for the 2024 edition."
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
        default_model=model
    )

    # Extract festival info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction"
    )

    # Normalize to exactly 4 positions (pad with empty placeholders if needed)
    festivals = extraction.festivals[:4]
    while len(festivals) < 4:
        festivals.append(FestivalItem())

    # Build and verify each festival per rubric
    await build_and_verify_festival_1(evaluator, root, festivals[0])
    await build_and_verify_festival_2(evaluator, root, festivals[1])
    await build_and_verify_festival_3(evaluator, root, festivals[2])
    await build_and_verify_festival_4(evaluator, root, festivals[3])

    return evaluator.get_summary()