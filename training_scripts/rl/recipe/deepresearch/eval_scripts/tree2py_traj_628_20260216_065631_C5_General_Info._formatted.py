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
TASK_ID = "entertainment_year_end_2025_2026"
TASK_DESCRIPTION = (
    "For an entertainment industry year-end report covering major Q4 2025 and Q1 2026 releases and events, identify "
    "five items with details as specified in the rubric: (1) a live holiday theatrical show in NYC at a large-capacity "
    "venue that ran into January 2026; (2) a streaming series finale released on Dec 31, 2025 with runtime ≥2 hours and "
    "also screened in theaters; (3) a film released Dec 25, 2025 that became the distributor’s highest-grossing with "
    "budget $60–70M; (4) a dance competition Season 34 winner with a pro partner who won Season 33, finale in late "
    "Nov 2025; (5) a WWE wrestler active since 2007 who changed ring name in Jan 2026 and has exactly three WWE titles."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Item1HolidayShow(BaseModel):
    # A live holiday theatrical show in NYC running through new year with venue capacity >= 5900
    show_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None
    venue_city: Optional[str] = None
    run_notes: Optional[str] = None  # any statement about running into Jan 2026
    sources: List[str] = Field(default_factory=list)


class Item2StreamingFinale(BaseModel):
    # Streaming series finale on 2025-12-31, runtime >= 2h, also screened in theaters
    series_name: Optional[str] = None
    episode_title: Optional[str] = None
    release_date: Optional[str] = None  # expect a date string like 2025-12-31 or textual "December 31, 2025"
    runtime: Optional[str] = None       # e.g., "2h 3m" or "123 minutes"
    theatrical_confirmation: Optional[str] = None  # text note like "also screened in theaters"
    sources: List[str] = Field(default_factory=list)


class Item3FilmRelease(BaseModel):
    # Film released 2025-12-25, highest-grossing for distributor, budget $60–70M
    film_name: Optional[str] = None
    distributor_name: Optional[str] = None
    release_date: Optional[str] = None
    production_budget: Optional[str] = None   # keep as string
    worldwide_box_office: Optional[str] = None
    distributor_record_note: Optional[str] = None  # text like "highest-grossing for distributor"
    sources: List[str] = Field(default_factory=list)


class Item4CompetitionWinner(BaseModel):
    # Dance competition Season 34 winner; partner also won Season 33; finale late Nov 2025
    celebrity_winner: Optional[str] = None
    professional_partner: Optional[str] = None
    show_name: Optional[str] = None
    season_number: Optional[str] = None
    finale_date: Optional[str] = None
    partner_previous_win: Optional[str] = None  # text indicating partner won Season 33
    sources: List[str] = Field(default_factory=list)


class Item5ProfessionalWrestler(BaseModel):
    # WWE wrestler active since 2007; ring name change in Jan 2026; exactly 3 titles
    previous_ring_name: Optional[str] = None
    new_ring_name: Optional[str] = None
    change_date: Optional[str] = None
    championship_count: Optional[str] = None  # keep as string, we’ll verify "exactly three"
    tenure_start_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EntertainmentReportExtraction(BaseModel):
    item1: Optional[Item1HolidayShow] = None
    item2: Optional[Item2StreamingFinale] = None
    item3: Optional[Item3FilmRelease] = None
    item4: Optional[Item4CompetitionWinner] = None
    item5: Optional[Item5ProfessionalWrestler] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the five requested entertainment items from the answer. Only extract information explicitly present in the answer. 
    If a field is missing, set it to null. Also extract all source URLs explicitly mentioned in the answer for each item.

    Item 1 (Holiday Theatrical Show in NYC):
    - show_name
    - venue_name
    - venue_capacity (as written, e.g., "6,015", "approx. 6015", etc.)
    - venue_city (city name if specified)
    - run_notes (any text that indicates performances continued into January 2026)
    - sources (URLs list specific to this item)

    Item 2 (Streaming Series Finale on Dec 31, 2025 with runtime ≥2h and theatrical screenings):
    - series_name
    - episode_title
    - release_date (as written)
    - runtime (as written, e.g., "2h 10m", "130 minutes")
    - theatrical_confirmation (text that claims it was also screened in theaters, if present)
    - sources (URLs list specific to this item)

    Item 3 (Theatrical Film released Dec 25, 2025; highest-grossing for distributor; budget $60–70M):
    - film_name
    - distributor_name
    - release_date (as written)
    - production_budget (as written, e.g., "$65 million")
    - worldwide_box_office (as written)
    - distributor_record_note (text claiming it is the distributor's highest-grossing worldwide release)
    - sources (URLs list specific to this item)

    Item 4 (Dance competition Season 34 winner; partner also won Season 33; finale late Nov 2025):
    - celebrity_winner
    - professional_partner
    - show_name
    - season_number (as written)
    - finale_date (as written)
    - partner_previous_win (text confirming the pro partner was a winner in Season 33)
    - sources (URLs list specific to this item)

    Item 5 (WWE wrestler active since 2007; ring name change in Jan 2026; exactly three championships):
    - previous_ring_name
    - new_ring_name
    - change_date (as written)
    - championship_count (as written; should indicate exactly 3)
    - tenure_start_year (as written, should be 2007)
    - sources (URLs list specific to this item)

    Return a JSON object with fields: item1, item2, item3, item4, item5. Each field should be a nested object with the keys above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic normalization: keep only non-empty strings that look like URLs
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            u = u.strip()
            if u:
                if not (u.startswith("http://") or u.startswith("https://")):
                    u = "http://" + u
                cleaned.append(u)
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


# --------------------------------------------------------------------------- #
# Verification logic per item                                                 #
# --------------------------------------------------------------------------- #
async def verify_item1_holiday_show(evaluator: Evaluator, parent_node, data: Optional[Item1HolidayShow]) -> None:
    node = evaluator.add_parallel(
        id="item1_holiday_show",
        desc="A live holiday theatrical show in NYC running through the new year period at a venue with capacity ≥5,900",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(data.sources if data else [])

    # Gate: sources provided
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item1_sources_provided",
        desc="Item 1: At least one source URL provided in the answer",
        parent=node,
        critical=True
    )

    # Show Identification (critical)
    show_ident = evaluator.add_parallel(
        id="item1_show_identification",
        desc="Show identification details",
        parent=node,
        critical=True
    )

    # Show Name leaf
    show_name_leaf = evaluator.add_leaf(
        id="item1_show_name",
        desc="Correct show name provided",
        parent=show_ident,
        critical=True
    )
    show_name = data.show_name if data else ""
    await evaluator.verify(
        claim=f"A theatrical stage production titled '{show_name}' exists.",
        node=show_name_leaf,
        sources=sources,
        additional_instruction="Verify that this is a live theatrical show (e.g., stage/musical/dance) rather than a concert or sporting event."
    )

    # Holiday Theme leaf
    holiday_leaf = evaluator.add_leaf(
        id="item1_holiday_theme",
        desc="Show is holiday/Christmas-themed theatrical production",
        parent=show_ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{show_name}' is holiday-themed (e.g., Christmas or seasonal holiday production).",
        node=holiday_leaf,
        sources=sources,
        additional_instruction="Confirm that the production explicitly references Christmas/holiday themes."
    )

    # Venue details (critical)
    venue = evaluator.add_parallel(
        id="item1_venue_details",
        desc="Venue specifications",
        parent=node,
        critical=True
    )

    # Venue Name leaf: ensure show at venue
    venue_name_leaf = evaluator.add_leaf(
        id="item1_venue_name",
        desc="Correct venue name provided",
        parent=venue,
        critical=True
    )
    venue_name = data.venue_name if data else ""
    await evaluator.verify(
        claim=f"The show '{show_name}' was performed at '{venue_name}'.",
        node=venue_name_leaf,
        sources=sources,
        additional_instruction="Confirm the venue name is correct for this production."
    )

    # Venue Capacity leaf: ≥ 5,900
    venue_capacity_leaf = evaluator.add_leaf(
        id="item1_venue_capacity",
        desc="Venue seating capacity ≥5,900 seats provided",
        parent=venue,
        critical=True
    )
    cap_txt = data.venue_capacity if data else ""
    await evaluator.verify(
        claim=f"The seating capacity of '{venue_name}' is {cap_txt}, which is at least 5,900 seats.",
        node=venue_capacity_leaf,
        sources=sources,
        additional_instruction="Verify the venue's seating capacity and confirm it meets or exceeds 5,900. Prefer official sources or Wikipedia/venue pages."
    )

    # Venue Location leaf: NYC
    venue_city_leaf = evaluator.add_leaf(
        id="item1_venue_location",
        desc="Venue is located in New York City",
        parent=venue,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue_name}' is located in New York City.",
        node=venue_city_leaf,
        sources=sources,
        additional_instruction="Confirm the city is New York City (Manhattan/NYC boroughs acceptable)."
    )

    # Timing leaf (critical)
    timing_leaf = evaluator.add_leaf(
        id="item1_timing",
        desc="Show runs through new year period (performances continuing past December 31, 2025 into January 2026)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{show_name}' had performances that continued past December 31, 2025 into January 2026.",
        node=timing_leaf,
        sources=sources,
        additional_instruction="Look for schedules, press materials, or listings that show January 2026 performance dates."
    )


async def verify_item2_streaming_finale(evaluator: Evaluator, parent_node, data: Optional[Item2StreamingFinale]) -> None:
    node = evaluator.add_parallel(
        id="item2_streaming_finale",
        desc="A streaming series finale released December 31, 2025 with runtime ≥2 hours and theatrical screenings",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(data.sources if data else [])

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item2_sources_provided",
        desc="Item 2: At least one source URL provided in the answer",
        parent=node,
        critical=True
    )

    # Series identification (critical)
    series_ident = evaluator.add_parallel(
        id="item2_series_ident",
        desc="Series and episode identification",
        parent=node,
        critical=True
    )

    series_name = data.series_name if data else ""
    episode_title = data.episode_title if data else ""

    series_name_leaf = evaluator.add_leaf(
        id="item2_series_name",
        desc="Correct series name provided",
        parent=series_ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"A streaming television series named '{series_name}' exists.",
        node=series_name_leaf,
        sources=sources,
        additional_instruction="Confirm the title is a streaming series (TV/limited series) on a streaming platform."
    )

    episode_title_leaf = evaluator.add_leaf(
        id="item2_episode_title",
        desc="Correct finale episode title provided",
        parent=series_ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The finale episode title of '{series_name}' is '{episode_title}'.",
        node=episode_title_leaf,
        sources=sources,
        additional_instruction="Verify the episode title corresponds to the series finale."
    )

    # Release details (critical)
    release_details = evaluator.add_parallel(
        id="item2_release_details",
        desc="Release timing specifications",
        parent=node,
        critical=True
    )

    release_date_txt = data.release_date if data else ""
    release_date_leaf = evaluator.add_leaf(
        id="item2_release_date",
        desc="Released on December 31, 2025",
        parent=release_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The finale '{episode_title}' of '{series_name}' was released on December 31, 2025.",
        node=release_date_leaf,
        sources=sources,
        additional_instruction="Confirm the streaming release date is 2025-12-31 (allow time zone phrasing but must align with Dec 31, 2025)."
    )

    runtime_txt = data.runtime if data else ""
    runtime_leaf = evaluator.add_leaf(
        id="item2_runtime",
        desc="Exact runtime ≥2 hours provided",
        parent=release_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The runtime of the finale '{episode_title}' is {runtime_txt}, which is at least 2 hours (≥120 minutes).",
        node=runtime_leaf,
        sources=sources,
        additional_instruction="Check the listed runtime and ensure it is 120 minutes or more. Accept formats like '2h 0m', '2h 10m', '120 minutes', etc."
    )

    # Theatrical screening confirmation (critical)
    theatrical_leaf = evaluator.add_leaf(
        id="item2_theatrical",
        desc="Theatrical screening in movie theaters confirmed",
        parent=node,
        critical=True
    )
    theatrical_note = data.theatrical_confirmation if data else ""
    await evaluator.verify(
        claim=f"The finale '{episode_title}' of '{series_name}' was also screened in movie theaters.",
        node=theatrical_leaf,
        sources=sources,
        additional_instruction="Look for press releases, trade reports, or theater listings confirming cinema screenings."
    )


async def verify_item3_film_release(evaluator: Evaluator, parent_node, data: Optional[Item3FilmRelease]) -> None:
    node = evaluator.add_parallel(
        id="item3_film_release",
        desc="A theatrical film released December 25, 2025 that became highest-grossing for its distributor with budget $60-70M",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(data.sources if data else [])

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item3_sources_provided",
        desc="Item 3: At least one source URL provided in the answer",
        parent=node,
        critical=True
    )

    # Film & distributor identification (critical)
    ident = evaluator.add_parallel(
        id="item3_film_identification",
        desc="Film and distributor identification",
        parent=node,
        critical=True
    )

    film_name = data.film_name if data else ""
    distributor_name = data.distributor_name if data else ""

    film_name_leaf = evaluator.add_leaf(
        id="item3_film_name",
        desc="Correct film name provided",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"A theatrical film titled '{film_name}' exists.",
        node=film_name_leaf,
        sources=sources,
        additional_instruction="Confirm it is a feature film released theatrically."
    )

    distributor_leaf = evaluator.add_leaf(
        id="item3_distributor_name",
        desc="Correct distributor name (independent film company) provided",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The film '{film_name}' was distributed by '{distributor_name}', an independent film company.",
        node=distributor_leaf,
        sources=sources,
        additional_instruction="Confirm distributor identity and that it's an independent film company."
    )

    # Release date leaf (critical)
    release_date_leaf = evaluator.add_leaf(
        id="item3_release_date",
        desc="Released on Christmas Day 2025 (December 25, 2025)",
        parent=node,
        critical=True
    )
    rel_txt = data.release_date if data else ""
    await evaluator.verify(
        claim=f"The film '{film_name}' was released on December 25, 2025.",
        node=release_date_leaf,
        sources=sources,
        additional_instruction="Check theatrical release date; region can be U.S. or primary territory but must align with Dec 25, 2025."
    )

    # Financial metrics (critical)
    finance = evaluator.add_parallel(
        id="item3_financial_metrics",
        desc="Financial performance details",
        parent=node,
        critical=True
    )

    budget_txt = data.production_budget if data else ""
    budget_leaf = evaluator.add_leaf(
        id="item3_production_budget",
        desc="Production budget between $60-70 million provided",
        parent=finance,
        critical=True
    )
    await evaluator.verify(
        claim=f"The production budget of '{film_name}' is {budget_txt}, which falls between $60 million and $70 million.",
        node=budget_leaf,
        sources=sources,
        additional_instruction="Confirm the budget value and that it lies within the $60–70M range."
    )

    box_office_txt = data.worldwide_box_office if data else ""
    box_office_leaf = evaluator.add_leaf(
        id="item3_box_office",
        desc="Worldwide box office total provided",
        parent=finance,
        critical=True
    )
    await evaluator.verify(
        claim=f"The worldwide box office total for '{film_name}' is {box_office_txt}.",
        node=box_office_leaf,
        sources=sources,
        additional_instruction="Confirm a worldwide total amount (global gross)."
    )

    record_leaf = evaluator.add_leaf(
        id="item3_distributor_record",
        desc="Confirmed as highest-grossing worldwide release for the distributor",
        parent=finance,
        critical=True
    )
    note_txt = data.distributor_record_note if data else ""
    await evaluator.verify(
        claim=f"'{film_name}' became the highest-grossing worldwide release for the distributor '{distributor_name}'.",
        node=record_leaf,
        sources=sources,
        additional_instruction="Look for explicit statements that this title is the distributor's highest-grossing worldwide release to date."
    )


async def verify_item4_competition_winner(evaluator: Evaluator, parent_node, data: Optional[Item4CompetitionWinner]) -> None:
    node = evaluator.add_parallel(
        id="item4_competition_winner",
        desc="Dance competition show Season 34 winner whose professional partner won the previous season, finale in late November 2025",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(data.sources if data else [])

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item4_sources_provided",
        desc="Item 4: At least one source URL provided in the answer",
        parent=node,
        critical=True
    )

    # Winner identification (critical)
    win_ident = evaluator.add_parallel(
        id="item4_winner_identification",
        desc="Winner and partner identification",
        parent=node,
        critical=True
    )

    celebrity = data.celebrity_winner if data else ""
    partner = data.professional_partner if data else ""
    show_name = data.show_name if data else ""
    season_number = data.season_number if data else ""
    finale_date = data.finale_date if data else ""

    celeb_leaf = evaluator.add_leaf(
        id="item4_celebrity_winner",
        desc="Correct celebrity winner name provided",
        parent=win_ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Season 34 of '{show_name}', the celebrity winner was {celebrity}.",
        node=celeb_leaf,
        sources=sources,
        additional_instruction="Confirm the named celebrity was the Season 34 champion."
    )

    partner_leaf = evaluator.add_leaf(
        id="item4_professional_partner",
        desc="Correct professional partner name provided",
        parent=win_ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Season 34 of '{show_name}', the winner's professional dance partner was {partner}.",
        node=partner_leaf,
        sources=sources,
        additional_instruction="Confirm the professional partner's name associated with the Season 34 winning pair."
    )

    partner_prev_win_leaf = evaluator.add_leaf(
        id="item4_partner_previous_win",
        desc="Professional partner won the competition in Season 33",
        parent=win_ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The professional partner {partner} won Season 33 of '{show_name}'.",
        node=partner_prev_win_leaf,
        sources=sources,
        additional_instruction="Verify that the same pro won the previous season (Season 33)."
    )

    # Show details (critical)
    details = evaluator.add_parallel(
        id="item4_show_details",
        desc="Competition show details",
        parent=node,
        critical=True
    )

    show_name_leaf = evaluator.add_leaf(
        id="item4_show_name",
        desc="Correct competition show name provided",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The competition show is titled '{show_name}'.",
        node=show_name_leaf,
        sources=sources,
        additional_instruction="Confirm the program name (e.g., a dance competition series)."
    )

    season_leaf = evaluator.add_leaf(
        id="item4_season_number",
        desc="Season 34 specified",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The season in question is Season 34.",
        node=season_leaf,
        sources=sources,
        additional_instruction="Confirm that this is indeed Season 34 for the show."
    )

    finale_date_leaf = evaluator.add_leaf(
        id="item4_finale_date",
        desc="Finale date in late November 2025 provided",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Season 34 finale aired on {finale_date}, which falls in late November 2025.",
        node=finale_date_leaf,
        sources=sources,
        additional_instruction="Treat 'late November' as approximately Nov 20–30, 2025 inclusive; verify the finale date falls in that range."
    )


async def verify_item5_professional_wrestler(evaluator: Evaluator, parent_node, data: Optional[Item5ProfessionalWrestler]) -> None:
    node = evaluator.add_parallel(
        id="item5_professional_wrestler",
        desc="WWE wrestler active since 2007 who changed ring name in January 2026 and has won exactly 3 championships",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(data.sources if data else [])

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item5_sources_provided",
        desc="Item 5: At least one source URL provided in the answer",
        parent=node,
        critical=True
    )

    # Name change details (critical)
    name_change = evaluator.add_parallel(
        id="item5_name_change",
        desc="Ring name change details",
        parent=node,
        critical=True
    )

    prev_name = data.previous_ring_name if data else ""
    new_name = data.new_ring_name if data else ""
    change_date = data.change_date if data else ""

    prev_name_leaf = evaluator.add_leaf(
        id="item5_previous_name",
        desc="Previous ring name provided correctly",
        parent=name_change,
        critical=True
    )
    await evaluator.verify(
        claim=f"The wrestler's previous ring name was '{prev_name}'.",
        node=prev_name_leaf,
        sources=sources,
        additional_instruction="Ensure that this previous ring name refers to the same individual who later changed the name."
    )

    new_name_leaf = evaluator.add_leaf(
        id="item5_new_name",
        desc="New ring name provided correctly",
        parent=name_change,
        critical=True
    )
    await evaluator.verify(
        claim=f"The wrestler's new ring name is '{new_name}'.",
        node=new_name_leaf,
        sources=sources,
        additional_instruction="Verify that the new ring name is officially used by WWE."
    )

    change_date_leaf = evaluator.add_leaf(
        id="item5_change_date",
        desc="Name change occurred in January 2026",
        parent=name_change,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ring name change from '{prev_name}' to '{new_name}' occurred in January 2026.",
        node=change_date_leaf,
        sources=sources,
        additional_instruction="Look for official announcements, WWE.com profiles, or reliable sources showing the name change timing in Jan 2026."
    )

    # Career statistics (critical)
    career = evaluator.add_parallel(
        id="item5_career_statistics",
        desc="WWE career statistics",
        parent=node,
        critical=True
    )

    champ_count_txt = data.championship_count if data else ""
    champ_leaf = evaluator.add_leaf(
        id="item5_championship_count",
        desc="Exactly three championship titles won",
        parent=career,
        critical=True
    )
    await evaluator.verify(
        claim=f"The wrestler formerly known as '{prev_name}' and now as '{new_name}' has won exactly three WWE championship titles.",
        node=champ_leaf,
        sources=sources,
        additional_instruction="Consider recognized WWE titles across brands (including tag/team) during WWE tenure; the total must equal three."
    )

    tenure_year = data.tenure_start_year if data else ""
    tenure_leaf = evaluator.add_leaf(
        id="item5_tenure_start",
        desc="WWE tenure began in 2007",
        parent=career,
        critical=True
    )
    await evaluator.verify(
        claim=f"The wrestler has been actively competing in WWE since {tenure_year}, beginning in 2007.",
        node=tenure_leaf,
        sources=sources,
        additional_instruction="Confirm the WWE debut/tenure start year is 2007 (allow developmental/FCW/NXT as WWE-affiliated)."
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
    Evaluate an answer for the entertainment year-end report (Q4 2025 & Q1 2026) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root aggregates items in parallel for partial credit
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

    # Extract all items from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=EntertainmentReportExtraction,
        extraction_name="entertainment_report_extraction"
    )

    # Build verification tree according to rubric (with source gating per item)
    await verify_item1_holiday_show(evaluator, root, extraction.item1)
    await verify_item2_streaming_finale(evaluator, root, extraction.item2)
    await verify_item3_film_release(evaluator, root, extraction.item3)
    await verify_item4_competition_winner(evaluator, root, extraction.item4)
    await verify_item5_professional_wrestler(evaluator, root, extraction.item5)

    # Return summary
    return evaluator.get_summary()