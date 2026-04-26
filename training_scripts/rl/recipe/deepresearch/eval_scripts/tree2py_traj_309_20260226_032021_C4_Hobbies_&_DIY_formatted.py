import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_craft_shopping_2025"
TASK_DESCRIPTION = (
    "I'm planning to create DIY Christmas decorations this year and need to purchase craft supplies from major "
    "craft store chains in the United States. I'm considering shopping at Michaels, Hobby Lobby, and Joann Fabrics. "
    "Please provide a comprehensive shopping plan that includes: (1) Which of these three craft store chains are "
    "currently available for shopping as of December 2025, (2) The store hours (opening and closing times) for "
    "available stores on Thanksgiving Day (November 27, 2025), (3) The store hours for available stores on Black "
    "Friday (November 28, 2025), (4) The store hours for available stores on Christmas Eve (December 24, 2025), "
    "(5) Whether these stores are open or closed on Christmas Day (December 25, 2025), and (6) Information about any "
    "extended holiday sales or promotions at these stores during late November through early December 2025. Please "
    "provide specific opening and closing times where applicable, and cite your sources with URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreHours(BaseModel):
    thanksgiving_status: Optional[str] = None
    thanksgiving_sources: List[str] = Field(default_factory=list)

    black_friday_open_time: Optional[str] = None
    black_friday_close_time: Optional[str] = None
    black_friday_sources: List[str] = Field(default_factory=list)

    christmas_eve_open_time: Optional[str] = None
    christmas_eve_close_time: Optional[str] = None
    christmas_eve_sources: List[str] = Field(default_factory=list)

    christmas_day_status: Optional[str] = None
    christmas_day_sources: List[str] = Field(default_factory=list)


class PromotionInfo(BaseModel):
    has_promo: Optional[bool] = None
    promo_desc: Optional[str] = None
    promo_period_start: Optional[str] = None
    promo_period_end: Optional[str] = None
    promo_discount_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StoreInfo(BaseModel):
    available: Optional[str] = None  # e.g., "available", "open", "closed", "permanently closed"
    availability_sources: List[str] = Field(default_factory=list)
    hours: Optional[StoreHours] = None
    promotions: Optional[PromotionInfo] = None


class HolidayPlanExtraction(BaseModel):
    michaels: Optional[StoreInfo] = None
    hobby_lobby: Optional[StoreInfo] = None
    joann: Optional[StoreInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_plan() -> str:
    return """
    Extract the structured information the answer provides for the three U.S. craft store chains: Michaels, Hobby Lobby, and Joann Fabrics (JOANN).
    You must extract EXACTLY what the answer states, and include all URLs cited for each specific claim. If a field is missing, return null or an empty list accordingly.

    For each store, return an object with:
    - available: String describing whether the answer claims the store is available/open for shopping as of December 2025 (e.g., "available", "open", "operational", "unavailable", "permanently closed").
    - availability_sources: Array of all URLs the answer cites specifically for the availability/operational status claim.
    - hours: An object with:
        - thanksgiving_status: String describing open/closed or hours on Thanksgiving Day (Nov 27, 2025) as claimed in the answer (e.g., "closed", "open 9am-6pm", etc.).
        - thanksgiving_sources: Array of all URLs cited specifically for Thanksgiving status/hours.
        - black_friday_open_time: Opening time on Black Friday (Nov 28, 2025) as claimed in the answer (e.g., "7:00 a.m.").
        - black_friday_close_time: Closing time on Black Friday (Nov 28, 2025) as claimed in the answer (e.g., "9:00 p.m.").
        - black_friday_sources: Array of all URLs cited specifically for Black Friday hours.
        - christmas_eve_open_time: Opening time on Christmas Eve (Dec 24, 2025) as claimed in the answer.
        - christmas_eve_close_time: Closing time on Christmas Eve (Dec 24, 2025) as claimed in the answer.
        - christmas_eve_sources: Array of all URLs cited specifically for Christmas Eve hours.
        - christmas_day_status: String describing open/closed status on Christmas Day (Dec 25, 2025) as claimed in the answer (e.g., "closed").
        - christmas_day_sources: Array of all URLs cited specifically for Christmas Day status.
    - promotions: An object with:
        - has_promo: Boolean indicating whether the answer claims there are extended holiday sales/promotions in late November through early December 2025.
        - promo_desc: Free-text summary of the promotion details the answer claims (e.g., "extended Black Friday/Cyber Monday sale from Nov 21 through Dec 6, up to 70% off").
        - promo_period_start: Start date of the promotion period (if present in the answer).
        - promo_period_end: End date of the promotion period (if present in the answer).
        - promo_discount_text: Any discount text like "up to 70%".
        - sources: Array of all URLs cited specifically for the promotions claim.

    Return a JSON object with top-level keys:
    - michaels
    - hobby_lobby
    - joann

    Each key maps to the store object described above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def collect_all_urls(extracted: HolidayPlanExtraction) -> List[str]:
    all_urls: List[str] = []

    def add(urls: Optional[List[str]]):
        nonlocal all_urls
        all_urls.extend(safe_urls(urls))

    for store in [extracted.michaels, extracted.hobby_lobby, extracted.joann]:
        if not store:
            continue
        add(store.availability_sources)
        if store.hours:
            add(store.hours.thanksgiving_sources)
            add(store.hours.black_friday_sources)
            add(store.hours.christmas_eve_sources)
            add(store.hours.christmas_day_sources)
        if store.promotions:
            add(store.promotions.sources)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def has_any_url(extracted: HolidayPlanExtraction) -> bool:
    return len(collect_all_urls(extracted)) > 0


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def add_citations_with_urls_node(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    evaluator.add_custom_node(
        result=has_any_url(extracted),
        id="Citations_With_URLs",
        desc="Provides source citations including URLs for the factual claims it makes (availability, holiday hours/status, promotions).",
        parent=parent,
        critical=True
    )


def _add_sources_presence_check(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: List[str]
):
    return evaluator.add_custom_node(
        result=len(safe_urls(urls)) > 0,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )


async def build_store_availability_group(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    group = evaluator.add_parallel(
        id="Store_Availability_As_Of_Dec_2025",
        desc="Addresses whether each of the three named chains is available for shopping as of December 2025.",
        parent=parent,
        critical=True
    )

    # JOANN: must state closure by May 30, 2025
    joann_sources = safe_urls((extracted.joann.availability_sources if extracted.joann else []) or [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Joann_Unavailable_Due_To_Closure_sources_present",
        "URLs provided to support JOANN closure/unavailability claim.",
        joann_sources
    )
    joann_leaf = evaluator.add_leaf(
        id="Joann_Unavailable_Due_To_Closure",
        desc="States that Joann Fabrics is no longer operational because it permanently closed all stores by May 30, 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="JOANN (Joann Fabrics) permanently closed all stores by May 30, 2025, and is no longer operational as of December 2025.",
        node=joann_leaf,
        sources=joann_sources,
        additional_instruction="Verify strictly against the provided URLs: do they explicitly support that JOANN closed all stores by May 30, 2025 and was not operating as of Dec 2025?"
    )

    # Michaels availability
    michaels_sources = safe_urls((extracted.michaels.availability_sources if extracted.michaels else []) or [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Michaels_Availability_Addressed_sources_present",
        "URLs provided to support Michaels availability/operational status in December 2025.",
        michaels_sources
    )
    michaels_leaf = evaluator.add_leaf(
        id="Michaels_Availability_Addressed",
        desc="Explicitly states whether Michaels is available for shopping as of December 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Michaels is operating and available for shopping as of December 2025.",
        node=michaels_leaf,
        sources=michaels_sources,
        additional_instruction="Confirm that the sources clearly indicate Michaels stores are operating and open to customers in December 2025."
    )

    # Hobby Lobby availability
    hl_sources = safe_urls((extracted.hobby_lobby.availability_sources if extracted.hobby_lobby else []) or [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Hobby_Lobby_Availability_Addressed_sources_present",
        "URLs provided to support Hobby Lobby availability/operational status in December 2025.",
        hl_sources
    )
    hl_leaf = evaluator.add_leaf(
        id="Hobby_Lobby_Availability_Addressed",
        desc="Explicitly states whether Hobby Lobby is available for shopping as of December 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Hobby Lobby is operating and available for shopping as of December 2025.",
        node=hl_leaf,
        sources=hl_sources,
        additional_instruction="Confirm that the sources clearly indicate Hobby Lobby stores are operating and open to customers in December 2025."
    )


async def build_thanksgiving_group(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    group = evaluator.add_parallel(
        id="Thanksgiving_Day_Status_Nov_27_2025",
        desc="Provides Thanksgiving Day (Nov 27, 2025) open/closed status for each available store.",
        parent=parent,
        critical=True
    )

    # Michaels closed on Thanksgiving Day
    m_hours = extracted.michaels.hours if (extracted.michaels and extracted.michaels.hours) else None
    m_t_sources = safe_urls(m_hours.thanksgiving_sources if m_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Michaels_Thanksgiving_Closed_sources_present",
        "URLs provided to support Michaels Thanksgiving Day (Nov 27, 2025) closed status.",
        m_t_sources
    )
    m_t_leaf = evaluator.add_leaf(
        id="Michaels_Thanksgiving_Closed",
        desc="States that Michaels is closed on Thanksgiving Day (Nov 27, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Michaels is closed on Thanksgiving Day, November 27, 2025.",
        node=m_t_leaf,
        sources=m_t_sources,
        additional_instruction="Focus on official Michaels holiday hours pages or trustworthy sources explicitly stating the Thanksgiving (Nov 27, 2025) closure."
    )

    # Hobby Lobby closed on Thanksgiving Day
    h_hours = extracted.hobby_lobby.hours if (extracted.hobby_lobby and extracted.hobby_lobby.hours) else None
    h_t_sources = safe_urls(h_hours.thanksgiving_sources if h_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Hobby_Lobby_Thanksgiving_Closed_sources_present",
        "URLs provided to support Hobby Lobby Thanksgiving Day (Nov 27, 2025) closed status.",
        h_t_sources
    )
    h_t_leaf = evaluator.add_leaf(
        id="Hobby_Lobby_Thanksgiving_Closed",
        desc="States that Hobby Lobby is closed on Thanksgiving Day (Nov 27, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Hobby Lobby is closed on Thanksgiving Day, November 27, 2025.",
        node=h_t_leaf,
        sources=h_t_sources,
        additional_instruction="Focus on official Hobby Lobby holiday hours pages or trustworthy sources explicitly stating the Thanksgiving (Nov 27, 2025) closure."
    )


async def build_black_friday_group(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    group = evaluator.add_parallel(
        id="Black_Friday_Hours_Nov_28_2025",
        desc="Provides Black Friday (Nov 28, 2025) store hours for each available store, with specific opening and closing times where applicable.",
        parent=parent,
        critical=True
    )

    # Michaels opens at 7:00 a.m.
    m_hours = extracted.michaels.hours if (extracted.michaels and extracted.michaels.hours) else None
    m_bf_sources = safe_urls(m_hours.black_friday_sources if m_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Michaels_Black_Friday_Hours_sources_present",
        "URLs provided to support Michaels Black Friday hours (Nov 28, 2025).",
        m_bf_sources
    )
    m_bf_leaf = evaluator.add_leaf(
        id="Michaels_Black_Friday_Hours",
        desc="Provides Michaels Black Friday hours and specifically states Michaels opens at 7:00 a.m. on Nov 28, 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="On Black Friday, November 28, 2025, Michaels opens at 7:00 a.m.",
        node=m_bf_leaf,
        sources=m_bf_sources,
        additional_instruction="Look for explicit Black Friday 2025 opening time for Michaels; confirm that the opening time is 7:00 a.m."
    )

    # Hobby Lobby 8:00 a.m. to 9:00 p.m.
    h_hours = extracted.hobby_lobby.hours if (extracted.hobby_lobby and extracted.hobby_lobby.hours) else None
    h_bf_sources = safe_urls(h_hours.black_friday_sources if h_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Hobby_Lobby_Black_Friday_8am_to_9pm_sources_present",
        "URLs provided to support Hobby Lobby Black Friday hours (Nov 28, 2025).",
        h_bf_sources
    )
    h_bf_leaf = evaluator.add_leaf(
        id="Hobby_Lobby_Black_Friday_8am_to_9pm",
        desc="States that Hobby Lobby is open from 8:00 a.m. to 9:00 p.m. on Black Friday (Nov 28, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="On Black Friday, November 28, 2025, Hobby Lobby is open from 8:00 a.m. to 9:00 p.m.",
        node=h_bf_leaf,
        sources=h_bf_sources,
        additional_instruction="Confirm that the Black Friday 2025 hours for Hobby Lobby explicitly run from 8:00 a.m. opening to 9:00 p.m. closing."
    )


async def build_christmas_eve_group(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    group = evaluator.add_parallel(
        id="Christmas_Eve_Hours_Dec_24_2025",
        desc="Provides Christmas Eve (Dec 24, 2025) store hours for each available store.",
        parent=parent,
        critical=True
    )

    # Michaels 7:00 a.m. to 6:00 p.m.
    m_hours = extracted.michaels.hours if (extracted.michaels and extracted.michaels.hours) else None
    m_ce_sources = safe_urls(m_hours.christmas_eve_sources if m_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Michaels_Christmas_Eve_7am_to_6pm_sources_present",
        "URLs provided to support Michaels Christmas Eve (Dec 24, 2025) hours.",
        m_ce_sources
    )
    m_ce_leaf = evaluator.add_leaf(
        id="Michaels_Christmas_Eve_7am_to_6pm",
        desc="States that Michaels is open from 7:00 a.m. to 6:00 p.m. on Christmas Eve (Dec 24, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="On Christmas Eve, December 24, 2025, Michaels is open from 7:00 a.m. to 6:00 p.m.",
        node=m_ce_leaf,
        sources=m_ce_sources,
        additional_instruction="Verify the specific Christmas Eve 2025 hours for Michaels are 7:00 a.m. opening and 6:00 p.m. closing."
    )

    # Hobby Lobby 9:00 a.m. to 5:30 p.m.
    h_hours = extracted.hobby_lobby.hours if (extracted.hobby_lobby and extracted.hobby_lobby.hours) else None
    h_ce_sources = safe_urls(h_hours.christmas_eve_sources if h_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Hobby_Lobby_Christmas_Eve_9am_to_530pm_sources_present",
        "URLs provided to support Hobby Lobby Christmas Eve (Dec 24, 2025) hours.",
        h_ce_sources
    )
    h_ce_leaf = evaluator.add_leaf(
        id="Hobby_Lobby_Christmas_Eve_9am_to_530pm",
        desc="States that Hobby Lobby is open from 9:00 a.m. to 5:30 p.m. on Christmas Eve (Dec 24, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="On Christmas Eve, December 24, 2025, Hobby Lobby is open from 9:00 a.m. to 5:30 p.m.",
        node=h_ce_leaf,
        sources=h_ce_sources,
        additional_instruction="Verify the specific Christmas Eve 2025 hours for Hobby Lobby are 9:00 a.m. opening and 5:30 p.m. closing."
    )


async def build_christmas_day_group(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    group = evaluator.add_parallel(
        id="Christmas_Day_Status_Dec_25_2025",
        desc="Provides Christmas Day (Dec 25, 2025) open/closed status for each available store.",
        parent=parent,
        critical=True
    )

    # Michaels closed on Christmas Day
    m_hours = extracted.michaels.hours if (extracted.michaels and extracted.michaels.hours) else None
    m_cd_sources = safe_urls(m_hours.christmas_day_sources if m_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Michaels_Christmas_Day_Closed_sources_present",
        "URLs provided to support Michaels Christmas Day (Dec 25, 2025) closed status.",
        m_cd_sources
    )
    m_cd_leaf = evaluator.add_leaf(
        id="Michaels_Christmas_Day_Closed",
        desc="States that Michaels is closed on Christmas Day (Dec 25, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Michaels is closed on Christmas Day, December 25, 2025.",
        node=m_cd_leaf,
        sources=m_cd_sources,
        additional_instruction="Confirm that the sources explicitly state Michaels is closed on Dec 25, 2025."
    )

    # Hobby Lobby closed on Christmas Day
    h_hours = extracted.hobby_lobby.hours if (extracted.hobby_lobby and extracted.hobby_lobby.hours) else None
    h_cd_sources = safe_urls(h_hours.christmas_day_sources if h_hours else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Hobby_Lobby_Christmas_Day_Closed_sources_present",
        "URLs provided to support Hobby Lobby Christmas Day (Dec 25, 2025) closed status.",
        h_cd_sources
    )
    h_cd_leaf = evaluator.add_leaf(
        id="Hobby_Lobby_Christmas_Day_Closed",
        desc="States that Hobby Lobby is closed on Christmas Day (Dec 25, 2025).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Hobby Lobby is closed on Christmas Day, December 25, 2025.",
        node=h_cd_leaf,
        sources=h_cd_sources,
        additional_instruction="Confirm that the sources explicitly state Hobby Lobby is closed on Dec 25, 2025."
    )


async def build_promotions_group(evaluator: Evaluator, parent, extracted: HolidayPlanExtraction) -> None:
    group = evaluator.add_parallel(
        id="Extended_Holiday_Promotions_Late_Nov_To_Early_Dec_2025",
        desc="Provides information about extended holiday sales/promotions in late November through early December 2025 for available stores.",
        parent=parent,
        critical=True
    )

    # Michaels extended sale Nov 21 - Dec 6, up to 70% off
    m_promos = extracted.michaels.promotions if (extracted.michaels and extracted.michaels.promotions) else None
    m_p_sources = safe_urls(m_promos.sources if m_promos else [])
    _add_sources_presence_check(
        evaluator,
        group,
        "Michaels_Extended_Sale_Nov21_to_Dec6_UpTo70_sources_present",
        "URLs provided to support Michaels extended sale period and discount claims.",
        m_p_sources
    )
    m_p_leaf = evaluator.add_leaf(
        id="Michaels_Extended_Sale_Nov21_to_Dec6_UpTo70",
        desc="States that Michaels has an extended Black Friday/Cyber Monday sale from Nov 21 through Dec 6, 2025, with discounts up to 70% off.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Michaels offers an extended Black Friday/Cyber Monday sale running from November 21 through December 6, 2025, with discounts up to 70% off.",
        node=m_p_leaf,
        sources=m_p_sources,
        additional_instruction="Verify both the date range (Nov 21–Dec 6, 2025) and the 'up to 70% off' discount are explicitly supported by the sources."
    )

    # Hobby Lobby promotions addressed (or explicitly none)
    h_promos = extracted.hobby_lobby.promotions if (extracted.hobby_lobby and extracted.hobby_lobby.promotions) else None
    h_p_sources = safe_urls(h_promos.sources if h_promos else [])
    # For this criterion, require that the answer addressed Hobby Lobby promotions in the specified period and provided URLs.
    evaluator.add_custom_node(
        result=(h_promos is not None and ((h_promos.has_promo is True and len(h_p_sources) > 0) or (h_promos.has_promo is False and len(h_p_sources) > 0))),
        id="Hobby_Lobby_Promotions_Addressed",
        desc="Provides information about any extended holiday sales/promotions for Hobby Lobby during late Nov through early Dec 2025 (or explicitly states none found/applicable).",
        parent=group,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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

    # Extraction
    extracted: HolidayPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_holiday_plan(),
        template_class=HolidayPlanExtraction,
        extraction_name="holiday_plan_struct"
    )

    # Build rubric tree under a critical top-level node
    top = evaluator.add_parallel(
        id="Holiday_Craft_Shopping_Plan",
        desc="Evaluate completeness and accuracy of the holiday craft store shopping plan for the named US craft chains and dates, including sources.",
        parent=root,
        critical=True
    )

    # Top-level citations presence check
    await add_citations_with_urls_node(evaluator, top, extracted)

    # Store availability group
    await build_store_availability_group(evaluator, top, extracted)

    # Thanksgiving status group
    await build_thanksgiving_group(evaluator, top, extracted)

    # Black Friday hours group
    await build_black_friday_group(evaluator, top, extracted)

    # Christmas Eve hours group
    await build_christmas_eve_group(evaluator, top, extracted)

    # Christmas Day status group
    await build_christmas_day_group(evaluator, top, extracted)

    # Promotions group
    await build_promotions_group(evaluator, top, extracted)

    # Return summary
    return evaluator.get_summary()