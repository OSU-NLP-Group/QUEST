import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "piano_concert"
TASK_DESCRIPTION = """
I'm thinking of attending a piano performance in the US or Canada, at least two months from now. I'd like the pianist to be someone who won a prize (1st–6th place) at the International Chopin Piano Competition from 2015 onward. Please help me find one eligible upcoming concert. For the pianist, provide their name, the year and prize they received at the Chopin Piano Competition, and a YouTube link showing their final-round performance at that competition. For the concert, include the date, city, venue, and a link to the official event page.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class PianistInfo(BaseModel):
    """Information about the pianist"""
    name: Optional[str] = Field(default=None, description="Name of the pianist")
    competition_year: Optional[str] = Field(default=None, description="Year they won at Chopin Competition")
    prize_placement: Optional[str] = Field(default=None, description="Prize placement (1st-6th)")
    youtube_url: Optional[str] = Field(default=None, description="YouTube link of final round performance")
    all_pianist_urls: Optional[List[str]] = Field(default_factory=list, description="All URLs related to the pianist (including non-YouTube)")



class ConcertInfo(BaseModel):
    """Information about the concert"""
    date: Optional[str] = Field(default=None, description="Concert date")
    city: Optional[str] = Field(default=None, description="Concert city")
    venue: Optional[str] = Field(default=None, description="Concert venue")
    event_page_url: Optional[str] = Field(default=None, description="Official event page URL")


def prompt_extract_pianist_info() -> str:
    """Extraction prompt for pianist information"""
    return """
    Extract information about the pianist from the answer.

    Look for:
    - name: The pianist's full name
    - competition_year: The year they won a prize at the International Chopin Piano Competition
    - prize_placement: Their placement (1st, 2nd, 3rd, 4th, 5th, or 6th place)
    - youtube_url: The main YouTube link provided for their final-round performance
    - all_pianist_urls: ALL URLs mentioned in relation to the pianist (including YouTube links, competition websites, pianist websites, etc.)

    Extract information exactly as it appears in the text.
    If any field is not mentioned, set it to null.
    For all_pianist_urls, include every URL that is mentioned in context of the pianist or their achievements.
    """


def prompt_extract_concert_info() -> str:
    """Extraction prompt for concert information"""
    return """
    Extract information about the concert from the answer.

    Look for:
    - date: The concert date (extract the full date as mentioned)
    - city: The city where the concert will take place
    - venue: The venue/hall name where the concert will be held
    - event_page_url: The main official event page URL provided

    Extract information exactly as it appears in the text.
    If any field is not mentioned, set it to null.
    """


async def verify_pianist_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        pianist_info: PianistInfo,
) -> None:
    """Verify all pianist-related information"""

    # Create pianist section node
    pianist_node = evaluator.add_parallel(
        id="pianist_verification",
        desc="Pianist information verification",
        parent=parent_node,
        critical=True  # Non-critical to allow partial credit
    )

    # Check pianist name exists
    name_exists_node = evaluator.add_custom_node(
        result=bool(pianist_info.name and pianist_info.name.strip() and pianist_info.competition_year and pianist_info.competition_year.strip() and pianist_info.prize_placement and pianist_info.prize_placement.strip() and pianist_info.youtube_url and pianist_info.youtube_url.strip()),
        id="pianist_name_exists",
        desc="All info provided",
        parent=pianist_node,
        critical=True
    )

    # Verify year is 2015 or later
    year_valid = False
    if pianist_info.competition_year:
        try:
            year = int(pianist_info.competition_year.strip())
            year_valid = year >= 2015
        except:
            pass

    year_range_node = evaluator.add_custom_node(
        result=year_valid,
        id="year_2015_or_later",
        desc=f"Competition year {pianist_info.competition_year} is 2015 or later",
        parent=pianist_node,
        critical=True
    )


    # Verify prize is 1st-6th place
    placement_valid_node = evaluator.add_leaf(
        id="prize_1st_to_6th",
        desc="Prize placement is between 1st and 6th place",
        parent=pianist_node,
        critical=True
    )

    # if pianist_info.prize_placement:
    claim = f"The prize placement '{pianist_info.prize_placement}' is one of: 1st place, 2nd place, 3rd place, 4th place, 5th place, or 6th place"
    await evaluator.verify(
        claim=claim,
        node=placement_valid_node,
        additional_instruction="Accept reasonable variations like 'first', 'second', etc."
    )

    # Verify prize placement is supported by URLs
    placement_supported_node = evaluator.add_leaf(
        id="placement_supported_by_url",
        desc="Prize placement claim is supported by provided URLs",
        parent=pianist_node,
        critical=True
    )

    # if pianist_info.prize_placement and pianist_info.all_pianist_urls:
    claim = f"{pianist_info.name} won {pianist_info.prize_placement} at the {pianist_info.competition_year} International Chopin Piano Competition"
    await evaluator.verify(
        claim=claim,
        node=placement_supported_node,
        sources=pianist_info.all_pianist_urls if pianist_info.all_pianist_urls else [],
        additional_instruction="Verify that the webpage contains information confirming the pianist's placement at the Chopin Competition"
    )


    # Verify YouTube URL is for final round performance
    youtube_final_round_node = evaluator.add_leaf(
        id="youtube_final_round",
        desc="YouTube link shows final-round Chopin Competition performance",
        parent=pianist_node,
        critical=True
    )

    # if pianist_info.youtube_url:
        # Check if URL actually shows the final round performance
    claim = f"The YouTube video shows {pianist_info.name}'s final-round performance at the {pianist_info.competition_year} International Chopin Piano Competition"
    await evaluator.verify(
        claim=claim,
        node=youtube_final_round_node,
        sources=pianist_info.youtube_url,
        additional_instruction="Verify that the video title, description, or content indicates this is a final round (or final stage) performance from the International Chopin Piano Competition of this person this year."
    )


async def verify_concert_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        concert_info: ConcertInfo,
) -> None:
    """Verify all concert-related information"""

    # Create concert section node
    concert_node = evaluator.add_parallel(
        id="concert_verification",
        desc="Concert information verification",
        parent=parent_node,
        critical=True  # Non-critical to allow partial credit
    )

    # Check info exists
    info_exists_node = evaluator.add_custom_node(
        result=bool(concert_info.date and concert_info.date.strip() and concert_info.city and concert_info.city.strip() and concert_info.venue and concert_info.venue.strip() and concert_info.event_page_url and concert_info.event_page_url.strip()),
        id="concert_info_exists",
        desc="Concert info is provided",
        parent=concert_node,
        critical=True
    )

    # Verify date is at least 2 months from now
    date_valid_node = evaluator.add_leaf(
        id="date_two_months_future",
        desc="Concert date is at least two months from now",
        parent=concert_node,
        critical=True
    )

    # if concert_info.date:
        # Note: We use relative language since we don't know the exact evaluation date
    today = datetime.utcnow().date()
    two_months_from_now = today + relativedelta(months=2)
    claim = (
        f"The concert date '{concert_info.date}' occurs on or after "
        f"{two_months_from_now.strftime('%B %d, %Y')}."
    )
    await evaluator.verify(
        claim=claim,
        node=date_valid_node,
    )

    # Verify city is in US or Canada
    city_location_node = evaluator.add_leaf(
        id="city_us_or_canada",
        desc="Concert city is located in the US or Canada",
        parent=concert_node,
        critical=True
    )

    # if concert_info.city:
    claim = f"The city '{concert_info.city}' is located in either the United States or Canada"
    await evaluator.verify(
        claim=claim,
        node=city_location_node,
        additional_instruction="Verify that the city is clearly in the US or Canada. Consider state/province abbreviations if provided."
    )


    # Verify event page shows the concert details
    event_page_valid_node = evaluator.add_leaf(
        id="event_page_shows_concert",
        desc="Event page confirms the concert details",
        parent=concert_node,
        critical=True
    )

    claim = f"The event page shows a concert on {concert_info.date} at {concert_info.venue} in {concert_info.city}"
    await evaluator.verify(
        claim=claim,
        node=event_page_valid_node,
        sources=concert_info.event_page_url,
        additional_instruction="Verify that the event page confirms the concert date, venue, and city."
    )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for piano concert task.

    Evaluates whether the answer provides:
    1. A pianist who won 1st-6th place at Chopin Competition from 2015 onward
    2. YouTube link of their final round performance
    3. An upcoming concert (2+ months away) in US/Canada
    4. Complete concert details with official event page
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract pianist information
    pianist_info = await evaluator.extract(
        prompt=prompt_extract_pianist_info(),
        template_class=PianistInfo,
        extraction_name="pianist_info",
    )

    # Extract concert information
    concert_info = await evaluator.extract(
        prompt=prompt_extract_concert_info(),
        template_class=ConcertInfo,
        extraction_name="concert_info",
    )

    # Build verification tree
    await verify_pianist_info(evaluator, root, pianist_info)
    await verify_concert_info(evaluator, root, concert_info)

    # Return evaluation results
    return evaluator.get_summary()
