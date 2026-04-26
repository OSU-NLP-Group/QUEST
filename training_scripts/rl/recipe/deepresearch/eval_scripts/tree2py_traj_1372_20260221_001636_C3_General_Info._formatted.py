import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "netflix_boxing_awards_2025"
TASK_DESCRIPTION = (
    "In December 2025, a major professional boxing match was streamed live on Netflix. "
    "This fight took place at a venue in Miami, Florida. Using publicly available information, "
    "identify the specific date (including the day of the month) when this boxing event occurred. "
    "After determining this date, identify a major entertainment industry awards ceremony that took place "
    "after this boxing event and before March 1, 2025, where awards for ensemble cast performances in films were presented. "
    "At this identified ceremony, determine which film won the award for Outstanding Performance by a Cast in a Motion Picture "
    "(or the equivalent top ensemble/cast category). What is the complete title of this winning film?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BoxingEventInfo(BaseModel):
    event_date: Optional[str] = None  # e.g., "December 14, 2025"
    netflix_streaming: Optional[str] = None  # e.g., "streamed live on Netflix" or "yes"/"no"
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None

    date_sources: List[str] = Field(default_factory=list)
    streaming_sources: List[str] = Field(default_factory=list)
    venue_sources: List[str] = Field(default_factory=list)


class CeremonyInfo(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None  # e.g., "February 24, 2025"
    ensemble_award_name: Optional[str] = None  # e.g., "Outstanding Performance by a Cast in a Motion Picture"

    identity_sources: List[str] = Field(default_factory=list)  # identity or general ceremony info pages
    date_sources: List[str] = Field(default_factory=list)      # sources that include the ceremony date
    ensemble_award_sources: List[str] = Field(default_factory=list)  # sources confirming ensemble award category


class WinnerInfo(BaseModel):
    film_title: Optional[str] = None  # complete official film title
    sources: List[str] = Field(default_factory=list)       # sources confirming the winner and category
    title_sources: List[str] = Field(default_factory=list) # sources specifically validating the full title


class FullExtraction(BaseModel):
    boxing_event: Optional[BoxingEventInfo] = None
    ceremony: Optional[CeremonyInfo] = None
    winner: Optional[WinnerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return (
        "Extract structured information from the answer for three parts: the December 2025 Netflix-streamed Miami boxing event, "
        "the awards ceremony, and the winning film.\n\n"
        "For the boxing event, extract:\n"
        "- event_date: the specific calendar date including day-of-month (e.g., 'December 14, 2025'); return null if not provided.\n"
        "- netflix_streaming: a short phrase indicating whether it was streamed live on Netflix (e.g., 'streamed live on Netflix'/'yes'/'no'); return null if not provided.\n"
        "- venue_name: the venue name if given (e.g., 'Kaseya Center'); return null if not provided.\n"
        "- venue_city: the city if given; return null if not provided.\n"
        "- venue_state: the state if given; return null if not provided.\n"
        "- date_sources: all URLs provided in the answer that support the specific boxing event date. If none are provided, return an empty list.\n"
        "- streaming_sources: all URLs provided that support Netflix live streaming of the event. If none, return an empty list.\n"
        "- venue_sources: all URLs provided that support the event being at a venue in Miami, Florida (including venue name/city/state). If none, return an empty list.\n\n"
        "For the awards ceremony, extract:\n"
        "- name: the ceremony name (e.g., 'Screen Actors Guild Awards'); return null if not provided.\n"
        "- date: the ceremony date (e.g., 'February 24, 2025'); return null if not provided.\n"
        "- ensemble_award_name: the name of the ceremony’s top film ensemble/cast award category (e.g., 'Outstanding Performance by a Cast in a Motion Picture'); return null if not provided.\n"
        "- identity_sources: all URLs that identify the ceremony and can help verify it exists; if none, return an empty list.\n"
        "- date_sources: URLs that include or allow verification of the ceremony date; if none, return an empty list.\n"
        "- ensemble_award_sources: URLs that confirm the ceremony presents a film ensemble/cast award category; if none, return an empty list.\n\n"
        "For the winning film, extract:\n"
        "- film_title: the complete official title of the film that won the ceremony’s top film ensemble/cast category; return null if not provided.\n"
        "- sources: URLs that support which film won the top film ensemble/cast category at the identified ceremony; if none, return an empty list.\n"
        "- title_sources: URLs that specifically confirm the complete official film title (can overlap with 'sources'); if none, return an empty list.\n\n"
        "Important URL rules:\n"
        "- Extract only actual URLs explicitly present in the answer text (including markdown links).\n"
        "- If a URL is missing 'http://' or 'https://', prepend 'http://'.\n"
        "- Ignore obviously invalid URLs. Return empty lists if no URLs were provided.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_day_of_month(date_str: Optional[str]) -> bool:
    """
    Returns True if the provided date string appears to include a day-of-month (1-31),
    rather than just month/year. Simple heuristic: presence of a standalone 1-31 number.
    """
    if not date_str or not isinstance(date_str, str):
        return False
    # Accept typical formats like "December 14, 2025" or "Dec. 14, 2025"
    match = re.search(r"\b([1-9]|[12][0-9]|3[01])\b", date_str)
    return bool(match)


def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Step builders                                                               #
# --------------------------------------------------------------------------- #
async def build_step1_boxing_event(
    evaluator: Evaluator,
    root_node,
    extraction: FullExtraction
) -> None:
    """
    Step 1: Determine the specific calendar date and ensure event matches constraints:
    - Date includes day-of-month
    - Occurs in December 2025
    - Streamed live on Netflix
    - Venue in Miami, Florida
    """
    step1 = evaluator.add_parallel(
        id="Step1_BoxingEventDateAndQualification",
        desc="Determine the specific calendar date (with day-of-month) for the boxing event described, and ensure the event matches the stated constraints.",
        parent=root_node,
        critical=True
    )

    be = extraction.boxing_event or BoxingEventInfo()
    date_includes_dom = _has_day_of_month(be.event_date)

    # Leaf: EventDateIncludesDayOfMonth (custom check)
    evaluator.add_custom_node(
        result=date_includes_dom,
        id="EventDateIncludesDayOfMonth",
        desc="The answer gives a specific calendar date for the boxing event including the day of the month (not only month/year).",
        parent=step1,
        critical=True
    )

    # Leaf: EventOccursInDecember2025
    occurs_dec_leaf = evaluator.add_leaf(
        id="EventOccursInDecember2025",
        desc="The boxing event tied to the provided date occurred in December 2025.",
        parent=step1,
        critical=True
    )
    await evaluator.verify(
        claim="The boxing event occurred in December 2025.",
        node=occurs_dec_leaf,
        sources=_clean_urls(be.date_sources),
        additional_instruction="Use the provided URLs to confirm the event date falls within December 2025."
    )

    # Leaf: EventStreamedLiveOnNetflix
    streamed_leaf = evaluator.add_leaf(
        id="EventStreamedLiveOnNetflix",
        desc="The boxing event tied to the provided date was streamed live on Netflix.",
        parent=step1,
        critical=True
    )
    await evaluator.verify(
        claim="This boxing event was streamed live on Netflix.",
        node=streamed_leaf,
        sources=_clean_urls(be.streaming_sources),
        additional_instruction="Judge based on the referenced webpages whether the event was broadcast live on Netflix."
    )

    # Leaf: EventVenueInMiamiFlorida
    venue_leaf = evaluator.add_leaf(
        id="EventVenueInMiamiFlorida",
        desc="The boxing event tied to the provided date took place at a venue in Miami, Florida.",
        parent=step1,
        critical=True
    )
    await evaluator.verify(
        claim="This boxing event took place at a venue in Miami, Florida.",
        node=venue_leaf,
        sources=_clean_urls(be.venue_sources),
        additional_instruction="Verify from the pages whether the venue is in Miami, Florida (city and state)."
    )


async def build_step2_ceremony_selection(
    evaluator: Evaluator,
    root_node,
    extraction: FullExtraction
) -> None:
    """
    Step 2: Identify an awards ceremony within the required time window and that presents film ensemble/cast awards.
    - Ceremony occurred after the boxing event date identified in Step 1
    - Ceremony occurred before March 1, 2025
    - Ceremony presents film ensemble/cast awards
    """
    step2 = evaluator.add_parallel(
        id="Step2_AwardsCeremonySelection",
        desc="Identify an entertainment industry awards ceremony that fits the required time window and presents film ensemble/cast awards.",
        parent=root_node,
        critical=True
    )

    be = extraction.boxing_event or BoxingEventInfo()
    cer = extraction.ceremony or CeremonyInfo()

    # Leaf: CeremonyAfterBoxingEventDate
    after_leaf = evaluator.add_leaf(
        id="CeremonyAfterBoxingEventDate",
        desc="The ceremony occurred after the boxing event date identified in Step 1.",
        parent=step2,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ceremony took place after the boxing event date ({be.event_date or 'unknown date'}).",
        node=after_leaf,
        sources=_clean_urls(cer.date_sources),
        additional_instruction="Using the ceremony date source(s), judge whether the ceremony date is chronologically after the boxing event date stated."
    )

    # Leaf: CeremonyBeforeMarch1_2025
    before_leaf = evaluator.add_leaf(
        id="CeremonyBeforeMarch1_2025",
        desc="The ceremony occurred before March 1, 2025.",
        parent=step2,
        critical=True
    )
    await evaluator.verify(
        claim="The ceremony occurred before March 1, 2025.",
        node=before_leaf,
        sources=_clean_urls(cer.date_sources),
        additional_instruction="Confirm the ceremony date is earlier than March 1, 2025."
    )

    # Leaf: CeremonyPresentsFilmEnsembleAwards
    presents_leaf = evaluator.add_leaf(
        id="CeremonyPresentsFilmEnsembleAwards",
        desc="The ceremony presents awards to film casts/ensembles (i.e., includes an ensemble/cast category for films).",
        parent=step2,
        critical=True
    )
    award_name = cer.ensemble_award_name or "a film ensemble/cast award category"
    await evaluator.verify(
        claim=f"The ceremony includes a film ensemble/cast award category such as '{award_name}'.",
        node=presents_leaf,
        sources=_clean_urls(cer.ensemble_award_sources),
        additional_instruction="Verify from official or reputable sources that this ceremony presents a film ensemble/cast award category."
    )


async def build_step3_ensemble_award_winner(
    evaluator: Evaluator,
    root_node,
    extraction: FullExtraction
) -> None:
    """
    Step 3: Determine the film that won the ceremony’s top ensemble/cast award and provide its complete official title.
    - Winner is for top ensemble/cast category (not an individual acting award)
    - Complete official film title provided
    """
    step3 = evaluator.add_parallel(
        id="Step3_EnsembleAwardWinnerFilmTitle",
        desc="Determine the film that won the ceremony’s top ensemble/cast award and provide its complete official title.",
        parent=root_node,
        critical=True
    )

    cer = extraction.ceremony or CeremonyInfo()
    win = extraction.winner or WinnerInfo()

    # Leaf: WinnerIsForTopEnsembleCastCategory
    winner_cat_leaf = evaluator.add_leaf(
        id="WinnerIsForTopEnsembleCastCategory",
        desc="The identified award is the ceremony’s top film ensemble/cast category (not an individual acting award).",
        parent=step3,
        critical=True
    )
    await evaluator.verify(
        claim=f"The film '{win.film_title or ''}' won the top ensemble/cast category '{cer.ensemble_award_name or ''}' at '{cer.name or ''}'.",
        node=winner_cat_leaf,
        sources=_clean_urls(win.sources),
        additional_instruction="Verify the claim refers to the ceremony’s top film ensemble/cast category, not an individual acting award."
    )

    # Leaf: CompleteOfficialFilmTitleProvided
    complete_title_leaf = evaluator.add_leaf(
        id="CompleteOfficialFilmTitleProvided",
        desc="The answer provides the complete official title of the winning film (full title, correctly spelled).",
        parent=step3,
        critical=True
    )
    await evaluator.verify(
        claim=f"The complete official title of the winning film is '{win.film_title or ''}'.",
        node=complete_title_leaf,
        sources=_clean_urls(win.title_sources or win.sources),
        additional_instruction="Confirm the film title matches the official title exactly (minor case differences acceptable)."
    )


async def build_step4_source_verification(
    evaluator: Evaluator,
    root_node,
    extraction: FullExtraction
) -> None:
    """
    Step 4: Provide reputable URL references that verify each required factual component.
    - Boxing event date
    - Netflix streaming
    - Miami venue
    - Ceremony identity and date
    - Ceremony has ensemble/cast film award
    - Winning film and category
    - Complete official film title
    """
    step4 = evaluator.add_parallel(
        id="Step4_SourceVerification",
        desc="Provide reputable URL references that verify each required factual component (boxing event details, ceremony timing/eligibility, and winning film/title/category).",
        parent=root_node,
        critical=True
    )

    be = extraction.boxing_event or BoxingEventInfo()
    cer = extraction.ceremony or CeremonyInfo()
    win = extraction.winner or WinnerInfo()

    # URLsVerifyBoxingEventDate
    urls_date_leaf = evaluator.add_leaf(
        id="URLsVerifyBoxingEventDate",
        desc="Reputable URL source(s) are provided that support the specific boxing event date (including day-of-month).",
        parent=step4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The boxing event occurred on {be.event_date or 'an explicit date'} (including the specific day-of-month).",
        node=urls_date_leaf,
        sources=_clean_urls(be.date_sources),
        additional_instruction="Use only the provided URLs. If no URLs are provided or they do not support the specific date including day-of-month, judge as not supported."
    )

    # URLsVerifyBoxingEventNetflixStreaming
    urls_stream_leaf = evaluator.add_leaf(
        id="URLsVerifyBoxingEventNetflixStreaming",
        desc="Reputable URL source(s) are provided that support that the boxing event was streamed live on Netflix.",
        parent=step4,
        critical=True
    )
    await evaluator.verify(
        claim="The boxing event was streamed live on Netflix.",
        node=urls_stream_leaf,
        sources=_clean_urls(be.streaming_sources),
        additional_instruction="Judge based strictly on the referenced webpages; do not infer. If no URLs are provided, consider it not supported."
    )

    # URLsVerifyBoxingEventMiamiVenue
    urls_venue_leaf = evaluator.add_leaf(
        id="URLsVerifyBoxingEventMiamiVenue",
        desc="Reputable URL source(s) are provided that support that the boxing event took place at a venue in Miami, Florida.",
        parent=step4,
        critical=True
    )
    await evaluator.verify(
        claim="The boxing event took place at a venue in Miami, Florida.",
        node=urls_venue_leaf,
        sources=_clean_urls(be.venue_sources),
        additional_instruction="Confirm the venue’s location is Miami, Florida. If no URLs are provided, consider it not supported."
    )

    # URLsVerifyAwardsCeremonyDateAndIdentity
    urls_cer_date_id_leaf = evaluator.add_leaf(
        id="URLsVerifyAwardsCeremonyDateAndIdentity",
        desc="Reputable URL source(s) are provided that support the awards ceremony’s identity and date.",
        parent=step4,
        critical=True
    )
    combined_cer_sources = _clean_urls((cer.identity_sources or []) + (cer.date_sources or []))
    await evaluator.verify(
        claim=f"The ceremony '{cer.name or 'the identified ceremony'}' took place on {cer.date or 'the specified date'}.",
        node=urls_cer_date_id_leaf,
        sources=combined_cer_sources,
        additional_instruction="From the URLs provided, verify both the ceremony identity and its date. If URLs are missing or do not confirm these, judge as not supported."
    )

    # URLsVerifyCeremonyHasEnsembleCastFilmAward
    urls_cer_has_ensemble_leaf = evaluator.add_leaf(
        id="URLsVerifyCeremonyHasEnsembleCastFilmAward",
        desc="Reputable URL source(s) are provided that support that the ceremony presents a film ensemble/cast award category.",
        parent=step4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ceremony presents a film ensemble/cast award category such as '{cer.ensemble_award_name or ''}'.",
        node=urls_cer_has_ensemble_leaf,
        sources=_clean_urls(cer.ensemble_award_sources),
        additional_instruction="Verify via the provided sources that the ceremony includes a film ensemble/cast category. If absent, judge as not supported."
    )

    # URLsVerifyWinningFilmAndCategory
    urls_winner_category_leaf = evaluator.add_leaf(
        id="URLsVerifyWinningFilmAndCategory",
        desc="Reputable URL source(s) are provided that support which film won the ceremony’s top film ensemble/cast award category.",
        parent=step4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The film '{win.film_title or ''}' won the top ensemble/cast category '{cer.ensemble_award_name or ''}' at '{cer.name or ''}'.",
        node=urls_winner_category_leaf,
        sources=_clean_urls(win.sources),
        additional_instruction="Use only the provided URLs to confirm the winning film and category. If URLs are missing or inconclusive, judge as not supported."
    )

    # URLsVerifyCompleteOfficialFilmTitle
    urls_full_title_leaf = evaluator.add_leaf(
        id="URLsVerifyCompleteOfficialFilmTitle",
        desc="Reputable URL source(s) are provided that support the complete official title of the winning film as stated in the answer.",
        parent=step4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The complete official title of the winning film is '{win.film_title or ''}'.",
        node=urls_full_title_leaf,
        sources=_clean_urls(win.title_sources or win.sources),
        additional_instruction="Confirm the film’s full official title from the provided URLs. If no URLs are provided, judge as not supported."
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
    Entry point for evaluating the agent's answer against the rubric using Mind2Web2 framework.
    Builds a sequential root with four critical steps and verifies each with source-grounded checks where applicable.
    """
    # Initialize evaluator (root created as non-critical by framework; children critical will gate scoring)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract all structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="structured_extraction"
    )

    # Build and verify Step 1
    await build_step1_boxing_event(evaluator, root, extraction)

    # Build and verify Step 2
    await build_step2_ceremony_selection(evaluator, root, extraction)

    # Build and verify Step 3
    await build_step3_ensemble_award_winner(evaluator, root, extraction)

    # Build and verify Step 4
    await build_step4_source_verification(evaluator, root, extraction)

    # Return structured result summary including the verification tree
    return evaluator.get_summary()