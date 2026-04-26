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
TASK_ID = "us_stadium_tour_2026_may_to_july"
TASK_DESCRIPTION = (
    "Identify a major music concert tour featuring stadium venues in the United States, with the tour beginning "
    "between May 1 and July 31, 2026. Provide the following information: the official tour name, the headlining "
    "artist(s), the date of the first stadium concert, the name of the stadium venue for the first concert, the city "
    "and state where the first venue is located, confirmation that the venue is a stadium (not an arena or theater), "
    "an official tour website or ticket purchase link, and the starting ticket price or price range (if publicly available)."
)

DATE_RANGE_START_TEXT = "May 1, 2026"
DATE_RANGE_END_TEXT = "July 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TourLinks(BaseModel):
    tour_website: Optional[str] = None
    ticket_link: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)
    first_venue_official_url: Optional[str] = None


class FirstConcert(BaseModel):
    date: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    is_stadium: Optional[bool] = None
    stadium_confirmation_text: Optional[str] = None
    stadium_confirmation_source_urls: List[str] = Field(default_factory=list)


class TourExtraction(BaseModel):
    tour_name: Optional[str] = None
    headlining_artists: List[str] = Field(default_factory=list)
    first_concert: Optional[FirstConcert] = None
    links: Optional[TourLinks] = None
    ticket_price_info: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_info() -> str:
    return """
    Extract the SINGLE main tour the answer focuses on (ignore any alternatives). Return the following fields:

    - tour_name: Official tour name as written (string; null if not provided).
    - headlining_artists: List of the headlining artist names exactly as written in the answer. If one artist, return a one-element array. If none, return [].
    - first_concert:
        - date: The date of the FIRST public concert of the tour (prefer the first U.S. stadium show if clearly stated as the opening show) exactly as written in the answer (e.g., "May 10, 2026" or "2026-05-10"). Return null if unknown.
        - venue_name: Name of the venue for that first concert, exactly as written (null if unknown).
        - city: City of the first concert venue (null if unknown).
        - state: U.S. state (full name or abbreviation) for the first concert venue (null if unknown).
        - country: Country for the first concert venue (null if unknown).
        - is_stadium: true if the answer explicitly says or clearly implies the venue is a stadium; false if it implies not a stadium; null if unclear.
        - stadium_confirmation_text: Any brief quote/phrase from the answer that indicates it is a "stadium" (e.g., "XYZ Stadium", "stadium tour", etc.); null if none.
        - stadium_confirmation_source_urls: URLs cited in the answer that directly help confirm the venue is a stadium ([], if none).
    - links:
        - tour_website: Official tour or artist page URL if provided (null if none).
        - ticket_link: Official ticket purchase URL if provided (Ticketmaster/AXS/Live Nation/SeatGeek or the artist's official store). Null if none.
        - schedule_urls: All URLs (array) that include announced dates/venues (official artist/promoter pages, or major ticketing platforms). [] if none.
        - first_venue_official_url: The official website URL of the first venue if provided. Null if none.
    - ticket_price_info: Starting ticket price or price range string exactly as written, if provided. If explicitly stated as "not available", "TBA", or similar, return that exact phrase. Null if not mentioned.

    Only extract URLs explicitly present in the answer (plain or markdown). Do not fabricate URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if not it:
            continue
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _collect_all_evidence_urls(ex: TourExtraction) -> List[str]:
    urls: List[str] = []
    if ex and ex.links:
        if ex.links.tour_website:
            urls.append(ex.links.tour_website)
        if ex.links.ticket_link:
            urls.append(ex.links.ticket_link)
        urls.extend(ex.links.schedule_urls or [])
        if ex.links.first_venue_official_url:
            urls.append(ex.links.first_venue_official_url)
    if ex and ex.first_concert:
        urls.extend(ex.first_concert.stadium_confirmation_source_urls or [])
    return _dedup_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: TourExtraction,
    answer_text: str,
) -> None:
    # Top-level container for this evaluation (non-critical to allow partial score on non-critical items)
    main = evaluator.add_parallel(
        id="stadium_concert_tour_information",
        desc="Information about a publicly announced U.S. stadium-venue music concert tour with first concert between May 1 and July 31, 2026.",
        parent=parent_node,
        critical=False,
    )

    all_urls = _collect_all_evidence_urls(extracted)
    first = extracted.first_concert or FirstConcert()
    links = extracted.links or TourLinks()

    # 1) Tour is a music concert tour (critical)
    tour_is_tour_node = evaluator.add_leaf(
        id="tour_is_music_concert_tour",
        desc="The identified event is a music concert tour (multiple scheduled concert dates).",
        parent=main,
        critical=True,
    )
    claim_is_tour = (
        f"The identified event{' named ' + extracted.tour_name if extracted and extracted.tour_name else ''} "
        f"is a music concert tour with multiple scheduled concert dates, as evidenced by the provided sources."
    )
    await evaluator.verify(
        claim=claim_is_tour,
        node=tour_is_tour_node,
        sources=all_urls,
        additional_instruction="Confirm that the sources show more than one concert date (i.e., a tour), not a single standalone event.",
    )

    # 2) Tour Name (critical group)
    tour_name_group = evaluator.add_parallel(
        id="tour_name_group",
        desc="The official tour name is provided and supported by sources.",
        parent=main,
        critical=True,
    )
    tour_name_provided = evaluator.add_custom_node(
        result=bool(extracted.tour_name and extracted.tour_name.strip()),
        id="tour_name_provided",
        desc="The official tour name is provided in the answer.",
        parent=tour_name_group,
        critical=True,
    )
    tour_name_supported = evaluator.add_leaf(
        id="tour_name_supported",
        desc="The official tour name is supported by authoritative sources.",
        parent=tour_name_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official tour name is '{extracted.tour_name}'.",
        node=tour_name_supported,
        sources=all_urls,
        additional_instruction="Check the official artist page, tour page, or major ticketing/promoter sites to confirm the exact/stylized tour name. Allow minor punctuation/casing variations.",
        extra_prerequisites=[tour_name_provided],
    )

    # 3) Headlining Artist(s) (critical group)
    headliner_group = evaluator.add_parallel(
        id="headlining_artist_group",
        desc="Headlining artist(s) are identified and supported by sources.",
        parent=main,
        critical=True,
    )
    headliner_provided = evaluator.add_custom_node(
        result=bool(extracted.headlining_artists),
        id="headlining_artist_provided",
        desc="The headlining artist(s) are provided.",
        parent=headliner_group,
        critical=True,
    )
    headliner_supported = evaluator.add_leaf(
        id="headlining_artist_supported",
        desc="The headlining artist(s) are accurately cited per sources.",
        parent=headliner_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The headlining artist(s) for the tour are: {extracted.headlining_artists}.",
        node=headliner_supported,
        sources=all_urls,
        additional_instruction="Verify that the listed headliner(s) match the official tour/artist or major ticketing sources. Allow minor name formatting differences.",
        extra_prerequisites=[headliner_provided],
    )

    # 4) Public announcement with verifiable schedule (critical)
    public_sched_group = evaluator.add_sequential(
        id="public_announcement_with_verifiable_schedule_group",
        desc="Tour publicly announced with verifiable scheduled dates/venues via authoritative sources.",
        parent=main,
        critical=True,
    )
    has_schedule_sources = evaluator.add_custom_node(
        result=bool(links.schedule_urls) or bool(links.tour_website) or bool(links.ticket_link),
        id="schedule_sources_provided",
        desc="At least one official/authoritative URL is provided for the tour or its schedule.",
        parent=public_sched_group,
        critical=True,
    )
    public_sched_supported = evaluator.add_leaf(
        id="public_announcement_with_verifiable_schedule",
        desc="The tour is publicly announced and has verifiable scheduled dates/venues via authoritative sources.",
        parent=public_sched_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The tour is publicly announced and has verifiable scheduled dates/venues on authoritative sources (official artist/promoter pages or major ticketing platforms).",
        node=public_sched_supported,
        sources=all_urls,
        additional_instruction="Look for schedules/lineups on official artist or promoter pages or on Ticketmaster, AXS, Live Nation, SeatGeek, or the stadium's official event page.",
    )

    # 5) Tour features stadium venues (critical)
    stadium_venues_node = evaluator.add_leaf(
        id="tour_features_stadium_venues",
        desc="Evidence indicates the tour features stadium venues (not arenas/theaters).",
        parent=main,
        critical=True,
    )
    await evaluator.verify(
        claim="The tour features concerts at stadium venues (not arenas/theaters), as indicated by the provided sources.",
        node=stadium_venues_node,
        sources=(first.stadium_confirmation_source_urls or links.schedule_urls or all_urls),
        additional_instruction="Accept evidence such as 'Stadium' in venue names, venue descriptions, or explicit 'stadium tour' labeling on official/ticketing pages.",
        # Making it depend on 'public_sched_supported' helps ensure evidence exists
        extra_prerequisites=[public_sched_supported],
    )

    # 6) First Concert Details (critical group)
    first_details = evaluator.add_parallel(
        id="first_concert_details",
        desc="First concert identified with date, venue, location; confirmed U.S. and stadium.",
        parent=main,
        critical=True,
    )

    # 6.a) First concert date checks (critical sequential)
    first_date_seq = evaluator.add_sequential(
        id="first_concert_date_checks",
        desc="First concert date provided, supported, and within required range.",
        parent=first_details,
        critical=True,
    )
    first_date_provided = evaluator.add_custom_node(
        result=bool(first.date and first.date.strip()),
        id="first_concert_date_provided",
        desc="First concert date is provided.",
        parent=first_date_seq,
        critical=True,
    )
    first_date_supported = evaluator.add_leaf(
        id="first_concert_date_supported",
        desc="The stated first concert date is supported by sources.",
        parent=first_date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The first concert date of the tour is {first.date}. If the schedule shows multiple dates, this is the opening/earliest public concert date.",
        node=first_date_supported,
        sources=(links.schedule_urls or all_urls),
        additional_instruction="Confirm the date corresponds to the earliest/first listed show (opening night) when possible. Allow minor date format differences.",
    )
    first_date_in_range = evaluator.add_leaf(
        id="first_concert_date_in_range",
        desc=f"The first concert date falls between {DATE_RANGE_START_TEXT} and {DATE_RANGE_END_TEXT} (inclusive).",
        parent=first_date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The date '{first.date}' falls between {DATE_RANGE_START_TEXT} and {DATE_RANGE_END_TEXT} (inclusive).",
        node=first_date_in_range,
        additional_instruction="Interpret common date formats. Consider timezones irrelevant; only the calendar date matters.",
    )

    # 6.b) First concert venue name (critical sequential)
    first_venue_seq = evaluator.add_sequential(
        id="first_concert_venue_name_checks",
        desc="First concert venue name provided and supported.",
        parent=first_details,
        critical=True,
    )
    first_venue_provided = evaluator.add_custom_node(
        result=bool(first.venue_name and first.venue_name.strip()),
        id="first_concert_venue_name_provided",
        desc="First concert venue name is provided.",
        parent=first_venue_seq,
        critical=True,
    )
    first_venue_supported = evaluator.add_leaf(
        id="first_concert_venue_name_supported",
        desc="First concert venue name is supported by sources.",
        parent=first_venue_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue for the first concert is '{first.venue_name}'.",
        node=first_venue_supported,
        sources=(links.schedule_urls or [links.first_venue_official_url] or all_urls),
        additional_instruction="Cross-check the first date's venue name on the tour schedule, ticketing page, or the venue's official event page.",
    )

    # 6.c) First concert city and state (critical sequential)
    first_loc_seq = evaluator.add_sequential(
        id="first_concert_city_state_checks",
        desc="First concert venue city and state provided and supported.",
        parent=first_details,
        critical=True,
    )
    city_state_provided = evaluator.add_custom_node(
        result=bool(first.city and first.city.strip() and first.state and first.state.strip()),
        id="first_concert_city_state_provided",
        desc="First concert venue city and state are provided.",
        parent=first_loc_seq,
        critical=True,
    )
    city_state_supported = evaluator.add_leaf(
        id="first_concert_city_state_supported",
        desc="First concert venue city and state are supported by sources.",
        parent=first_loc_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The first concert venue is located in {first.city}, {first.state}.",
        node=city_state_supported,
        sources=(links.schedule_urls or [links.first_venue_official_url] or all_urls),
        additional_instruction="Use the event listing or venue site to confirm city and state. Allow common abbreviations (e.g., CA for California).",
    )

    # 6.d) First concert is in United States (critical leaf)
    first_in_us = evaluator.add_leaf(
        id="first_concert_is_in_united_states",
        desc="The first concert takes place in the United States.",
        parent=first_details,
        critical=True,
    )
    claim_in_us = (
        f"The first concert venue "
        f"{'('+first.venue_name+') ' if first.venue_name else ''}"
        f"is located in the United States."
        f"{' The city/state listed are '+str(first.city)+', '+str(first.state)+'.' if first.city or first.state else ''}"
    )
    await evaluator.verify(
        claim=claim_in_us,
        node=first_in_us,
        sources=(links.schedule_urls or [links.first_venue_official_url] or all_urls),
        additional_instruction="Confirm the venue is in the USA. City/state presence usually implies U.S. location for U.S. states.",
    )

    # 6.e) First venue confirmed stadium (critical sequential)
    first_stadium_seq = evaluator.add_sequential(
        id="first_venue_confirmed_stadium_seq",
        desc="The first concert venue is confirmed to be a stadium.",
        parent=first_details,
        critical=True,
    )
    stadium_evidence_provided = evaluator.add_custom_node(
        result=bool(
            (first.is_stadium is True)
            or (first.stadium_confirmation_text and "stadium" in first.stadium_confirmation_text.lower())
            or (first.venue_name and "stadium" in first.venue_name.lower())
        ),
        id="first_venue_stadium_evidence_provided",
        desc="Some stadium indication is present (venue name/description implies stadium).",
        parent=first_stadium_seq,
        critical=True,
    )
    first_venue_stadium_supported = evaluator.add_leaf(
        id="first_venue_confirmed_stadium",
        desc="The first concert venue is a stadium (not an arena or theater), per authoritative sources.",
        parent=first_stadium_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{first.venue_name}' is a stadium (not an arena/theater).",
        node=first_venue_stadium_supported,
        sources=(first.stadium_confirmation_source_urls or [links.first_venue_official_url] or all_urls),
        additional_instruction="Use the venue's official page, Wikipedia infobox/lead, or credible sources to confirm 'stadium' designation.",
    )

    # 7) Official website or ticket purchase link (critical group)
    official_link_group = evaluator.add_parallel(
        id="official_website_or_ticket_link_group",
        desc="An official tour website or official ticket purchase link is provided and is official.",
        parent=main,
        critical=True,
    )
    official_link_provided = evaluator.add_custom_node(
        result=bool(links.tour_website or links.ticket_link),
        id="official_link_provided",
        desc="At least one: official tour website or official ticket purchase link is provided.",
        parent=official_link_group,
        critical=True,
    )
    official_link_is_official = evaluator.add_leaf(
        id="official_link_is_official",
        desc="The provided website/link is official (artist/promoter) or an official ticketing platform.",
        parent=official_link_group,
        critical=True,
    )
    provided_check_urls = _dedup_preserve_order(
        [links.tour_website or "", links.ticket_link or ""]
    )
    await evaluator.verify(
        claim="At least one of the provided links is an official tour/artist website or an official ticketing platform page for this tour.",
        node=official_link_is_official,
        sources=(provided_check_urls or all_urls),
        additional_instruction="Look for official artist domains or recognized ticketing platforms (Ticketmaster, AXS, Live Nation, SeatGeek) linked from official sources.",
        extra_prerequisites=[official_link_provided],
    )

    # 8) Official ticketing info publicly available (critical leaf)
    ticketing_available = evaluator.add_leaf(
        id="official_ticketing_info_publicly_available",
        desc="Publicly available ticket purchasing information exists through official channels.",
        parent=main,
        critical=True,
    )
    await evaluator.verify(
        claim="There exists a publicly accessible official ticket purchasing page for this tour (or for the first stadium date).",
        node=ticketing_available,
        sources=( [links.ticket_link] if links.ticket_link else all_urls ),
        additional_instruction="Prefer a direct official ticket purchase page for the tour/date. If not direct, the official site must route to an official ticketing platform.",
        extra_prerequisites=[official_link_provided],
    )

    # 9) Ticket price information (non-critical group)
    price_group = evaluator.add_parallel(
        id="ticket_price_information_group",
        desc="Starting ticket price or range is provided if publicly available; otherwise explicitly indicated as not publicly available.",
        parent=main,
        critical=False,
    )
    price_keywords = ["not available", "tba", "to be announced", "not yet available", "unavailable"]
    answer_lower = (answer_text or "").lower()
    explicitly_not_available = any(k in answer_lower for k in price_keywords)
    price_declared_or_na = evaluator.add_custom_node(
        result=bool(extracted.ticket_price_info and extracted.ticket_price_info.strip()) or explicitly_not_available,
        id="ticket_price_declared_or_not_available",
        desc="Ticket price provided OR the answer explicitly states it is not publicly available.",
        parent=price_group,
        critical=False,
    )
    price_supported = evaluator.add_leaf(
        id="ticket_price_supported_by_sources",
        desc="If a price is provided, it is supported by official/ticketing sources.",
        parent=price_group,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The starting ticket price or price range for the first stadium concert is '{extracted.ticket_price_info}'.",
        node=price_supported,
        sources=( [links.ticket_link] if links.ticket_link else all_urls ),
        additional_instruction="Verify only if a concrete price/value is provided. Allow ranges and dynamic pricing notes. If no price is provided, this claim should be judged as incorrect.",
        extra_prerequisites=[price_declared_or_na],
    )

    # Add helpful debug info
    evaluator.add_custom_info(
        info={
            "collected_evidence_urls": all_urls,
            "extracted_summary": {
                "tour_name": extracted.tour_name,
                "headlining_artists": extracted.headlining_artists,
                "first_concert": first.dict(),
                "links": links.dict(),
                "ticket_price_info": extracted.ticket_price_info,
            },
        },
        info_type="debug",
        info_name="extraction_and_evidence_summary",
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
    Evaluate an answer for the U.S. Stadium Tour 2026 (May–July start) task.
    """
    # Initialize evaluator with PARALLEL aggregation at root
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_tour_info(),
        template_class=TourExtraction,
        extraction_name="tour_extraction",
    )

    # Build verification tree
    await build_verification_tree(
        evaluator=evaluator,
        parent_node=root,
        extracted=extraction,
        answer_text=answer,
    )

    # Return standardized summary
    return evaluator.get_summary()