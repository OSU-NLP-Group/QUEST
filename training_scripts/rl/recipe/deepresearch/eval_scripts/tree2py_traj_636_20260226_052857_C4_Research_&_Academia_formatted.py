import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

TASK_ID = "conference_lunar_planetary_2026"
TASK_DESCRIPTION = "Identify the name and exact location (city and state) of the conference that meets all of the following criteria: (1) It is a space science or astronomy conference held in the United States in 2026; (2) The conference focuses on lunar and planetary science; (3) It takes place in March 2026; (4) The conference runs for exactly 5 consecutive days; (5) The conference location is in Texas; (6) The venue is at a combined hotel and convention center facility; (7) The conference is the 57th edition of this annual event series. Provide the official conference name and the city and state where it is held."


class ConferenceExtraction(BaseModel):
    conference_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    venue_name: Optional[str] = None
    dates_text: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days_text: Optional[str] = None
    edition_text: Optional[str] = None
    edition_number: Optional[str] = None
    type_text: Optional[str] = None
    focus_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_conference() -> str:
    return (
        "From the answer, extract the following details about the identified conference:\n"
        "- conference_name: The official name of the conference (e.g., 'Lunar and Planetary Science Conference')\n"
        "- city: The city where the conference is held\n"
        "- state: The U.S. state where the conference is held\n"
        "- country: The country (e.g., 'United States', 'USA') if mentioned\n"
        "- venue_name: The venue name (e.g., 'The Woodlands Waterway Marriott Hotel & Convention Center') if mentioned\n"
        "- dates_text: The dates as stated (e.g., 'March 16–20, 2026')\n"
        "- start_date: The starting date in text as provided in the answer\n"
        "- end_date: The ending date in text as provided in the answer\n"
        "- duration_days_text: The duration wording (e.g., '5 days') if mentioned\n"
        "- edition_text: The edition wording (e.g., '57th') if mentioned\n"
        "- edition_number: The edition number as text (e.g., '57') if present\n"
        "- type_text: The domain or type (e.g., 'space science', 'astronomy') as described\n"
        "- focus_text: A focus description (e.g., 'lunar and planetary science') if present\n"
        "- source_urls: All URLs cited in the answer that substantiate the claims about the conference\n"
        "Rules for URL extraction:\n"
        "• Extract only actual URLs found in the answer (including markdown links). Do not invent URLs.\n"
        "• If a URL is missing a protocol, prepend http://.\n"
        "If any field is not present in the answer, return null for that field (or an empty list for source_urls)."
    )


async def build_and_verify_conference_tree(
    evaluator: Evaluator,
    parent_node,
    info: ConferenceExtraction,
) -> None:
    conference_node = evaluator.add_parallel(
        id="Conference_Identification",
        desc="Correctly identify the space science conference that meets all specified criteria and provide the required information",
        parent=parent_node,
        critical=False,
    )

    name_provided_node = evaluator.add_custom_node(
        result=bool(info.conference_name and info.conference_name.strip()),
        id="Conference_Name_Provided",
        desc="The answer provides the official conference name",
        parent=conference_node,
        critical=True,
    )

    city_provided_node = evaluator.add_custom_node(
        result=bool(info.city and info.city.strip()),
        id="City_Provided",
        desc="The answer provides the city where the conference is held",
        parent=conference_node,
        critical=True,
    )

    state_provided_node = evaluator.add_custom_node(
        result=bool(info.state and info.state.strip()),
        id="State_Provided",
        desc="The answer provides the state where the conference is held",
        parent=conference_node,
        critical=True,
    )

    us_location_leaf = evaluator.add_leaf(
        id="US_Location",
        desc="The conference is held in the United States",
        parent=conference_node,
        critical=True,
    )
    us_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' is held in the United States."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_location_leaf,
        sources=info.source_urls,
        additional_instruction="Use the cited webpage(s) to confirm the country is the United States (USA). Mentions like 'TX' or 'Texas' should be recognized as US, but rely on the page evidence.",
    )

    type_leaf = evaluator.add_leaf(
        id="Conference_Type",
        desc="The conference is a space science or astronomy conference",
        parent=conference_node,
        critical=True,
    )
    type_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' is a space science or astronomy conference."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=info.source_urls,
        additional_instruction="Confirm from the page(s) that the event is within the domain of space science or astronomy.",
    )

    march_2026_leaf = evaluator.add_leaf(
        id="March_2026_Timing",
        desc="The conference takes place in March 2026",
        parent=conference_node,
        critical=True,
    )
    march_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' takes place in March 2026."
    )
    await evaluator.verify(
        claim=march_claim,
        node=march_2026_leaf,
        sources=info.source_urls,
        additional_instruction="Confirm that all official dates listed for the conference fall within March 2026.",
    )

    five_day_leaf = evaluator.add_leaf(
        id="Five_Day_Duration",
        desc="The conference has a duration of exactly 5 consecutive days",
        parent=conference_node,
        critical=True,
    )
    five_day_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' runs for exactly five consecutive days."
    )
    await evaluator.verify(
        claim=five_day_claim,
        node=five_day_leaf,
        sources=info.source_urls,
        additional_instruction="Use the start/end dates on the official page(s) to confirm a 5-day consecutive schedule (e.g., Monday–Friday).",
    )

    texas_leaf = evaluator.add_leaf(
        id="Texas_Location",
        desc="The conference is held in Texas",
        parent=conference_node,
        critical=True,
    )
    texas_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' is held in Texas."
    )
    await evaluator.verify(
        claim=texas_claim,
        node=texas_leaf,
        sources=info.source_urls,
        additional_instruction="Confirm the state is Texas (TX) from the cited page(s).",
    )

    focus_leaf = evaluator.add_leaf(
        id="Lunar_Planetary_Focus",
        desc="The conference focuses on lunar and planetary science",
        parent=conference_node,
        critical=True,
    )
    focus_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' focuses on lunar and planetary science."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=focus_leaf,
        sources=info.source_urls,
        additional_instruction="Look for explicit phrasing such as 'Lunar and Planetary Science' or equivalent; confirm the event's focus is planetary/lunar science.",
    )

    venue_leaf = evaluator.add_leaf(
        id="Hotel_Convention_Center_Venue",
        desc="The conference venue is at a combined hotel and convention center facility",
        parent=conference_node,
        critical=True,
    )
    venue_specific = info.venue_name or "the official venue"
    venue_claim = (
        f"The conference venue is a combined hotel and convention center facility: {venue_specific}."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=info.source_urls,
        additional_instruction="Verify that the named venue is both a hotel and a convention center (e.g., 'Marriott Hotel & Convention Center').",
    )

    edition_leaf = evaluator.add_leaf(
        id="57th_Edition",
        desc="The conference is the 57th edition of this annual event series",
        parent=conference_node,
        critical=True,
    )
    edition_claim = (
        f"The conference '{info.conference_name or 'the identified conference'}' is the 57th edition of this annual series."
    )
    await evaluator.verify(
        claim=edition_claim,
        node=edition_leaf,
        sources=info.source_urls,
        additional_instruction="Confirm references to '57th LPSC' or '57th edition' on the official page(s).",
    )


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

    extracted_info = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction",
    )

    evaluator.add_ground_truth(
        {
            "expected_example": {
                "conference_name": "Lunar and Planetary Science Conference (LPSC)",
                "edition": "57th",
                "month_year": "March 2026",
                "duration_days": "5",
                "city": "The Woodlands",
                "state": "Texas",
                "country": "United States",
                "venue": "The Woodlands Waterway Marriott Hotel & Convention Center",
            },
            "note": "Ground truth provided for reference; verification is based on the answer's cited sources.",
        },
        gt_type="ground_truth_conference",
    )

    await build_and_verify_conference_tree(evaluator, root, extracted_info)

    return evaluator.get_summary()