import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "historic_nyc_concert_hall_1891"
TASK_DESCRIPTION = (
    "In 1891, a concert hall opened in New York City that would become one of the most acoustically renowned venues "
    "in the world. This concert hall's main auditorium has a seating capacity between 2,790 and 2,804 seats arranged "
    "across exactly five levels. The hall was designed by an architect who was also a musician—specifically, he played "
    "the cello—and at the request of the venue's benefactor, this architect prioritized acoustic excellence as the primary "
    "design principle. Identify this concert hall and provide the following information: (1) The full name of the concert "
    "hall, (2) The architect's full name, (3) Confirmation that the architect was a cellist, (4) Confirmation that the design "
    "prioritized acoustic excellence, and (5) Confirmation that the main auditorium has exactly five levels. Provide URL references "
    "supporting the architect's identity, the acoustic design priority, and the five-level configuration of the main auditorium."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class URLReferences(BaseModel):
    architect_identity_urls: List[str] = Field(default_factory=list)
    acoustic_design_urls: List[str] = Field(default_factory=list)
    five_level_config_urls: List[str] = Field(default_factory=list)


class ConcertHallInfo(BaseModel):
    hall_name: Optional[str] = None
    opening_year: Optional[str] = None
    city: Optional[str] = None

    architect_name: Optional[str] = None
    architect_cellist_confirmation: Optional[str] = None  # e.g., "yes", "no", or textual confirmation from the answer

    acoustic_priority_confirmation: Optional[str] = None  # textual confirmation (e.g., "yes", "stated")
    five_levels_confirmation: Optional[str] = None        # textual confirmation (e.g., "five tiers", "five levels")

    main_auditorium_capacity: Optional[str] = None        # e.g., "2,804", "about 2,800", "2,790–2,804"
    main_auditorium_level_count: Optional[str] = None     # e.g., "5"

    national_historic_landmark_status: Optional[str] = None  # e.g., "yes"
    still_operating_today: Optional[str] = None              # e.g., "yes"

    references: URLReferences = Field(default_factory=URLReferences)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concert_hall_info() -> str:
    return """
    Extract the key facts stated in the answer about the historic New York City concert hall (opened in 1891) and the requested citations.

    Return a JSON with the following fields (use null if any item is not explicitly stated in the answer):

    - hall_name: Full name of the concert hall (string).
    - opening_year: Opening year (string, e.g., "1891").
    - city: City of the venue (string, e.g., "New York City" or "NYC" or "Manhattan, New York").
    - architect_name: Full name of the architect (string).
    - architect_cellist_confirmation: Did the answer explicitly say the architect was a cellist or played the cello? Put the exact phrasing or "yes"/"no".
    - acoustic_priority_confirmation: Did the answer explicitly say acoustic excellence was prioritized in the design? Put the exact phrasing or "yes"/"no".
    - five_levels_confirmation: Did the answer explicitly say the main auditorium has exactly five levels (or tiers)? Put the exact phrasing or "yes"/"no".
    - main_auditorium_capacity: The capacity described for the main auditorium in the answer (e.g., "2,804", "about 2,800", "2,790–2,804").
    - main_auditorium_level_count: The number of levels/tiers stated for the main auditorium (string; e.g., "5").
    - national_historic_landmark_status: Did the answer say it's a National Historic Landmark? Put the exact phrasing or "yes"/"no".
    - still_operating_today: Did the answer say the venue is still operating today? Put the exact phrasing or "yes"/"no".

    - references: An object containing three URL lists (extract only URLs explicitly present in the answer text):
        - architect_identity_urls: URLs that support the architect’s identity (that this architect designed the hall).
        - acoustic_design_urls: URLs that support that the design prioritized acoustic excellence.
        - five_level_config_urls: URLs that support the main auditorium having exactly five levels.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer. If a category has no URLs, return an empty list for that category.
    - Do not invent or infer details not present in the answer. Use null for missing fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


def _union_urls(info: ConcertHallInfo) -> List[str]:
    return list({
        *(_safe_list(info.references.architect_identity_urls)),
        *(_safe_list(info.references.acoustic_design_urls)),
        *(_safe_list(info.references.five_level_config_urls)),
    })


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
    Evaluate the answer for the Historic NYC concert hall (1891) task.
    """
    # 1) Initialize evaluator
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

    # 2) Extract structured info from the answer
    extracted: ConcertHallInfo = await evaluator.extract(
        prompt=prompt_extract_concert_hall_info(),
        template_class=ConcertHallInfo,
        extraction_name="concert_hall_info",
    )

    # 3) Build the critical top-level node (as per rubric)
    top = evaluator.add_parallel(
        id="Historic_NYC_Concert_Hall_1891",
        desc="Identify the NYC concert hall that opened in 1891 and provide all required confirmations and citations per the question/constraints.",
        parent=root,
        critical=True,
    )

    hall_name = extracted.hall_name or "the identified concert hall"
    all_urls = _union_urls(extracted)

    # 4) Leaf: Concert Hall Full Name Provided (existence check)
    evaluator.add_custom_node(
        result=bool(extracted.hall_name and extracted.hall_name.strip()),
        id="Concert_Hall_Full_Name_Provided",
        desc="The answer provides the full name of the concert hall being identified.",
        parent=top,
        critical=True,
    )

    # 5) Leaf: Opening Year 1891 (verify with any provided URLs if available)
    node_open_1891 = evaluator.add_leaf(
        id="Opening_Year_1891",
        desc="The identified concert hall opened in 1891.",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{hall_name} opened in 1891.",
        node=node_open_1891,
        sources=all_urls if all_urls else None,
        additional_instruction="Check whether the page explicitly states the opening year as 1891. Allow phrasing like 'opened in 1891' or 'opened on May 5, 1891'.",
    )

    # 6) Leaf: NYC Location (verify with any provided URLs if available)
    node_nyc = evaluator.add_leaf(
        id="NYC_Location",
        desc="The identified concert hall is located in New York City.",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{hall_name} is located in New York City (NYC, Manhattan, New York).",
        node=node_nyc,
        sources=all_urls if all_urls else None,
        additional_instruction="Accept mentions like 'New York City', 'NYC', 'Manhattan, New York', or similar that clearly indicate NYC.",
    )

    # 7) Leaf: Main Auditorium Seating Capacity Range (2,790–2,804)
    node_capacity = evaluator.add_leaf(
        id="Main_Auditorium_Seating_Capacity_Range",
        desc="The main auditorium seating capacity is between 2,790 and 2,804 seats.",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The main auditorium of {hall_name} has a seating capacity between 2,790 and 2,804 seats (inclusive).",
        node=node_capacity,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Treat values like 'about 2,800', '2,804', or '2,790' as within the stated range. "
            "Verify that the auditorium capacity falls within 2,790 to 2,804."
        ),
    )

    # 8) Leaf: Architect identified as William Burnet Tuthill (from the answer content)
    node_architect_ident = evaluator.add_leaf(
        id="Architect_Identified_As_William_Burnet_Tuthill",
        desc="The architect is identified and the full name matches William Burnet Tuthill (William Burnet Tuthill).",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The architect of {hall_name} is William Burnet Tuthill.",
        node=node_architect_ident,
        sources=None,  # This checks consistency with the answer text itself.
        additional_instruction="Judge based on the answer content whether it identifies the architect as 'William Burnet Tuthill'. Minor formatting differences are acceptable.",
    )

    # 9) Leaf: Architect was a cellist (verify using any relevant URLs if present)
    node_cellist = evaluator.add_leaf(
        id="Architect_Was_A_Cellist",
        desc="The answer confirms the architect was a musician who played the cello (i.e., was a cellist).",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim="William Burnet Tuthill was a cellist (played the cello).",
        node=node_cellist,
        sources=all_urls if all_urls else None,
        additional_instruction="Accept statements like 'amateur cellist', 'played the cello', or 'cellist'.",
    )

    # 10) Leaf: Design prioritized acoustic excellence (verify preferably with acoustic URLs)
    node_acoustic = evaluator.add_leaf(
        id="Design_Prioritized_Acoustic_Excellence",
        desc="The answer confirms the design prioritized acoustic excellence as the primary design principle (at the benefactor’s request).",
        parent=top,
        critical=True,
    )
    acoustic_urls = _safe_list(extracted.references.acoustic_design_urls)
    await evaluator.verify(
        claim=f"The design of {hall_name} prioritized acoustic excellence as the primary design principle.",
        node=node_acoustic,
        sources=acoustic_urls if acoustic_urls else all_urls if all_urls else None,
        additional_instruction="Supportive phrasing includes 'acoustics were paramount', 'acoustic excellence was the primary concern', or similar.",
    )

    # 11) Leaf: Exactly five levels (verify preferably with five-level URLs)
    node_five_levels = evaluator.add_leaf(
        id="Main_Auditorium_Has_Exactly_Five_Levels",
        desc="The answer confirms the main auditorium has exactly five levels/tiers.",
        parent=top,
        critical=True,
    )
    five_urls = _safe_list(extracted.references.five_level_config_urls)
    await evaluator.verify(
        claim=f"The main auditorium of {hall_name} has exactly five levels (tiers).",
        node=node_five_levels,
        sources=five_urls if five_urls else all_urls if all_urls else None,
        additional_instruction="Accept synonyms like 'five tiers' or explicit listings of five named levels.",
    )

    # 12) Leaf: National Historic Landmark status (verify with any provided URLs)
    node_nhl = evaluator.add_leaf(
        id="National_Historic_Landmark_Status",
        desc="The answer states the venue is a National Historic Landmark (as required by constraints).",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{hall_name} is a National Historic Landmark.",
        node=node_nhl,
        sources=all_urls if all_urls else None,
        additional_instruction="Look for explicit statements like 'National Historic Landmark' or recognized equivalents.",
    )

    # 13) Leaf: Still operating today (verify with any provided URLs)
    node_operating = evaluator.add_leaf(
        id="Still_Operating_Today",
        desc="The answer states the venue is still operating today (as required by constraints).",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{hall_name} is still operating today.",
        node=node_operating,
        sources=all_urls if all_urls else None,
        additional_instruction="Accept evidence of ongoing events, programming, or phrasing indicating current operation.",
    )

    # 14) Required URL References Provided (critical parallel sub-node)
    url_parent = evaluator.add_parallel(
        id="Required_URL_References_Provided",
        desc="Provide URL references supporting (i) the architect’s identity, (ii) the acoustic design priority, and (iii) the five-level configuration (as requested in the question).",
        parent=top,
        critical=True,
    )

    # 14.a) Architect identity URL(s)
    node_architect_url = evaluator.add_leaf(
        id="URL_Supporting_Architect_Identity",
        desc="At least one URL is provided that supports the architect’s identity (William Burnet Tuthill as architect).",
        parent=url_parent,
        critical=True,
    )
    identity_urls = _safe_list(extracted.references.architect_identity_urls)
    if len(identity_urls) == 0:
        node_architect_url.score = 0.0
        node_architect_url.status = "failed"
    else:
        await evaluator.verify(
            claim=f"William Burnet Tuthill is the architect of {hall_name}.",
            node=node_architect_url,
            sources=identity_urls,
            additional_instruction="Look for phrases like 'designed by William Burnet Tuthill' or 'architect William Burnet Tuthill'.",
        )

    # 14.b) Acoustic design priority URL(s)
    node_acoustic_url = evaluator.add_leaf(
        id="URL_Supporting_Acoustic_Design_Priority",
        desc="At least one URL is provided that supports that acoustic excellence was the primary design priority.",
        parent=url_parent,
        critical=True,
    )
    if len(acoustic_urls) == 0:
        node_acoustic_url.score = 0.0
        node_acoustic_url.status = "failed"
    else:
        await evaluator.verify(
            claim=f"The design of {hall_name} prioritized acoustic excellence as the primary design principle.",
            node=node_acoustic_url,
            sources=acoustic_urls,
            additional_instruction="Supportive statements include 'acoustics were paramount', 'acoustic excellence was prioritized', etc.",
        )

    # 14.c) Five-level configuration URL(s)
    node_five_url = evaluator.add_leaf(
        id="URL_Supporting_Five_Level_Configuration",
        desc="At least one URL is provided that supports the main auditorium has exactly five levels/tiers.",
        parent=url_parent,
        critical=True,
    )
    if len(five_urls) == 0:
        node_five_url.score = 0.0
        node_five_url.status = "failed"
    else:
        await evaluator.verify(
            claim=f"The main auditorium of {hall_name} has exactly five levels (tiers).",
            node=node_five_url,
            sources=five_urls,
            additional_instruction="Accept phrasing like 'five tiers' or an explicit enumeration of five levels.",
        )

    # 15) Return final summary
    return evaluator.get_summary()