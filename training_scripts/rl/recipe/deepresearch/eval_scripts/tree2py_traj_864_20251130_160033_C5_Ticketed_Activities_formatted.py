import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_show_tv_adaptation"
TASK_DESCRIPTION = (
    "Identify a Broadway show currently playing in New York City that is based on a popular streaming or television series. "
    "Provide the following information: (1) The name of the theatre where it performs, including the complete street address, "
    "city, and ZIP code; (2) The total runtime of the show, including intermission details if applicable; "
    "(3) The price of rush tickets and where they can be purchased; (4) Reference URLs to support all information provided."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowSelection(BaseModel):
    show_name: Optional[str] = None
    based_on_statement: Optional[str] = None  # e.g., "Based on the Netflix series X"
    based_on_urls: List[str] = Field(default_factory=list)
    currently_playing_urls: List[str] = Field(default_factory=list)
    nyc_location_urls: List[str] = Field(default_factory=list)


class TheatreInfo(BaseModel):
    theatre_name: Optional[str] = None
    theatre_name_urls: List[str] = Field(default_factory=list)

    street_address: Optional[str] = None
    street_address_urls: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    city_urls: List[str] = Field(default_factory=list)

    zip_code: Optional[str] = None
    zip_urls: List[str] = Field(default_factory=list)


class PerformanceInfo(BaseModel):
    runtime_total: Optional[str] = None  # e.g., "2h 30m", "150 minutes"
    runtime_urls: List[str] = Field(default_factory=list)

    intermission_detail: Optional[str] = None  # e.g., "One intermission (15 minutes)" or "No intermission"
    intermission_urls: List[str] = Field(default_factory=list)


class RushInfo(BaseModel):
    rush_price: Optional[str] = None  # e.g., "$39", "$40"
    rush_purchase_where: Optional[str] = None  # e.g., "TodayTix app", "Box office"
    rush_price_urls: List[str] = Field(default_factory=list)
    rush_where_urls: List[str] = Field(default_factory=list)


class ShowExtraction(BaseModel):
    selection: Optional[ShowSelection] = None
    theatre: Optional[TheatreInfo] = None
    performance: Optional[PerformanceInfo] = None
    rush: Optional[RushInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_show_data() -> str:
    return """
    You will extract structured information about ONE Broadway show described in the answer. If the answer mentions multiple shows, extract ONLY the first one that is clearly presented with supporting details.

    Extract the following fields (return null for any missing field; return [] for any missing URL list):
    selection:
      - show_name: The Broadway show name.
      - based_on_statement: A short phrase the answer uses to assert the show is based on a streaming/TV series (e.g., "Based on the Netflix series ..."). If not explicitly stated, return null.
      - based_on_urls: All URLs cited that specifically support the claim that the show is based on a streaming/TV series.
      - currently_playing_urls: All URLs cited that support it is currently playing ON BROADWAY.
      - nyc_location_urls: All URLs cited that support the show is located in New York City (NYC).

    theatre:
      - theatre_name: The venue where the show performs (e.g., "Gerald Schoenfeld Theatre").
      - theatre_name_urls: URLs that support the theatre name for this show.
      - street_address: Complete street address (e.g., "236 W 45th St").
      - street_address_urls: URLs that support the street address.
      - city: City (e.g., "New York" or "New York, NY").
      - city_urls: URLs that support the city.
      - zip_code: ZIP code (e.g., "10036").
      - zip_urls: URLs that support the ZIP code.

    performance:
      - runtime_total: The total runtime as stated (e.g., "2h 30m", "150 minutes").
      - runtime_urls: URLs that support the runtime.
      - intermission_detail: Intermission information as stated (e.g., "One intermission (15 minutes)" OR "No intermission").
      - intermission_urls: URLs that support the intermission detail.

    rush:
      - rush_price: The price stated for rush tickets (e.g., "$39", "$40").
      - rush_purchase_where: Where rush tickets can be purchased (e.g., "Box office", "TodayTix app", "digital rush on TodayTix").
      - rush_price_urls: URLs that support the rush ticket price.
      - rush_where_urls: URLs that support the purchase location/process.

    SPECIAL URL RULES:
    - Extract only URLs that are explicitly present in the answer (plain links or markdown).
    - Include complete URLs with protocol; if missing, prepend "http://".
    - Do not invent URLs.

    Return a JSON object that fits the provided template exactly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _urls_clean(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


def _mk_source_instruction(urls: List[str], base_instruction: str) -> str:
    cnt = len(urls)
    extra = (
        f"Reference URL count provided: {cnt}. "
        "If zero URLs are provided for this claim, you MUST judge the claim as NOT SUPPORTED (Incorrect). "
    )
    return extra + base_instruction


def _safe(s: Optional[str], fallback: str = "") -> str:
    return s if isinstance(s, str) and s.strip() != "" else fallback


# --------------------------------------------------------------------------- #
# Verification workflow                                                       #
# --------------------------------------------------------------------------- #
async def _verify_show_selection(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    """
    Build and verify the 'Show_Selection_Criteria' group:
      - Based on streaming/TV series + citation
      - Currently playing on Broadway + citation
      - Located in NYC + citation
    """
    sel = data.selection or ShowSelection()
    show_name = _safe(sel.show_name, "the show")

    group = evaluator.add_parallel(
        id="Show_Selection_Criteria",
        desc="Chosen show satisfies all selection constraints, each supported by at least one valid reference URL.",
        parent=parent_node,
        critical=True,
    )

    # 1) Based on streaming/TV series (with citation)
    node_based = evaluator.add_leaf(
        id="Based_On_Streaming_Or_TV_Series_With_Citation",
        desc="States that the show is based on/adapted from a streaming or television series AND provides a valid reference URL supporting this claim.",
        parent=group,
        critical=True,
    )
    based_urls = _urls_clean(sel.based_on_urls)
    claim_based = (
        f"This page supports that the Broadway show '{show_name}' is based on or adapted from "
        f"a streaming or television series."
    )
    await evaluator.verify(
        claim=claim_based,
        node=node_based,
        sources=based_urls if based_urls else None,
        additional_instruction=_mk_source_instruction(
            based_urls,
            "Accept clear phrasing like 'based on the [network/streaming] series', "
            "'adapted from the TV series', or 'inspired by the [TV/streaming] show'. "
            "Do NOT accept adaptations from films or novels for this criterion."
        ),
    )

    # 2) Currently playing on Broadway (with citation)
    node_current = evaluator.add_leaf(
        id="Currently_Playing_On_Broadway_With_Citation",
        desc="States that the show is currently playing on Broadway AND provides a valid reference URL supporting current Broadway status.",
        parent=group,
        critical=True,
    )
    cur_urls = _urls_clean(sel.currently_playing_urls)
    claim_current = (
        f"This page indicates that '{show_name}' is currently playing on Broadway (i.e., a Broadway production in NYC, not Off‑Broadway)."
    )
    await evaluator.verify(
        claim=claim_current,
        node=node_current,
        sources=cur_urls if cur_urls else None,
        additional_instruction=_mk_source_instruction(
            cur_urls,
            "The evidence should refer to Broadway status (e.g., 'on Broadway', 'now playing on Broadway', or listings clearly marked as Broadway). "
            "If the page is clearly outdated or indicates a closed run, do not treat it as 'currently playing'."
        ),
    )

    # 3) Located in NYC (with citation)
    node_nyc = evaluator.add_leaf(
        id="Located_In_New_York_City_With_Citation",
        desc="States that the show is located in New York City AND provides a valid reference URL supporting the NYC location.",
        parent=group,
        critical=True,
    )
    nyc_urls = _urls_clean(sel.nyc_location_urls)
    claim_nyc = f"This page shows that '{show_name}' is located in New York City, New York (NYC)."
    await evaluator.verify(
        claim=claim_nyc,
        node=node_nyc,
        sources=nyc_urls if nyc_urls else None,
        additional_instruction=_mk_source_instruction(
            nyc_urls,
            "The page should refer to New York City (e.g., 'New York, NY'). "
            "References to other cities should not pass."
        ),
    )


async def _verify_theatre_info(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    """
    Build and verify the 'Theatre_Venue_Information' group:
      - Theatre name + citation
      - Street address + citation
      - City + citation
      - ZIP + citation
    """
    sel = data.selection or ShowSelection()
    th = data.theatre or TheatreInfo()

    show_name = _safe(sel.show_name, "the show")
    theatre_name = _safe(th.theatre_name, "")
    street_address = _safe(th.street_address, "")
    city = _safe(th.city, "")
    zip_code = _safe(th.zip_code, "")

    group = evaluator.add_parallel(
        id="Theatre_Venue_Information",
        desc="Theatre name and complete address are provided, each supported by at least one valid reference URL.",
        parent=parent_node,
        critical=True,
    )

    # Theatre name with citation
    node_name = evaluator.add_leaf(
        id="Theatre_Name_With_Citation",
        desc="Provides the theatre name where the show performs AND provides a valid reference URL supporting the theatre name.",
        parent=group,
        critical=True,
    )
    urls_name = _urls_clean(th.theatre_name_urls)
    claim_name = f"This page indicates that '{show_name}' performs at the theatre named '{theatre_name}'."
    await evaluator.verify(
        claim=claim_name,
        node=node_name,
        sources=urls_name if urls_name else None,
        additional_instruction=_mk_source_instruction(
            urls_name,
            "Allow small variations like 'Theatre' vs 'Theater' or inclusion of a sponsor name, but the core theatre identity should match."
        ),
    )

    # Street address with citation
    node_addr = evaluator.add_leaf(
        id="Theatre_Street_Address_With_Citation",
        desc="Provides the complete street address of the theatre AND provides a valid reference URL supporting the street address.",
        parent=group,
        critical=True,
    )
    urls_addr = _urls_clean(th.street_address_urls)
    claim_addr = f"This page shows that the theatre's street address is '{street_address}'."
    await evaluator.verify(
        claim=claim_addr,
        node=node_addr,
        sources=urls_addr if urls_addr else None,
        additional_instruction=_mk_source_instruction(
            urls_addr,
            "Minor formatting differences like 'St' vs 'Street' or punctuation are acceptable."
        ),
    )

    # City with citation
    node_city = evaluator.add_leaf(
        id="Theatre_City_With_Citation",
        desc="Provides the theatre city AND provides a valid reference URL supporting the city.",
        parent=group,
        critical=True,
    )
    urls_city = _urls_clean(th.city_urls)
    claim_city = f"This page shows that the theatre's city is '{city}' (i.e., NYC)."
    await evaluator.verify(
        claim=claim_city,
        node=node_city,
        sources=urls_city if urls_city else None,
        additional_instruction=_mk_source_instruction(
            urls_city,
            "Accept forms like 'New York' or 'New York, NY'. It must correspond to New York City."
        ),
    )

    # ZIP with citation
    node_zip = evaluator.add_leaf(
        id="Theatre_ZIP_With_Citation",
        desc="Provides the theatre ZIP code AND provides a valid reference URL supporting the ZIP code.",
        parent=group,
        critical=True,
    )
    urls_zip = _urls_clean(th.zip_urls)
    claim_zip = f"This page shows that the theatre's ZIP code is '{zip_code}'."
    await evaluator.verify(
        claim=claim_zip,
        node=node_zip,
        sources=urls_zip if urls_zip else None,
        additional_instruction=_mk_source_instruction(
            urls_zip,
            "Accept 5‑digit ZIP (e.g., '10036') or ZIP+4 if shown on the page. The number must match."
        ),
    )


async def _verify_performance_details(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    """
    Build and verify the 'Performance_Details' group:
      - Total runtime + citation
      - Intermission details (or explicit 'no intermission') + citation
    """
    sel = data.selection or ShowSelection()
    perf = data.performance or PerformanceInfo()

    show_name = _safe(sel.show_name, "the show")
    runtime_total = _safe(perf.runtime_total, "")
    intermission_detail = _safe(perf.intermission_detail, "")

    group = evaluator.add_parallel(
        id="Performance_Details",
        desc="Runtime and intermission details are provided, each supported by at least one valid reference URL.",
        parent=parent_node,
        critical=True,
    )

    # Runtime with citation
    node_runtime = evaluator.add_leaf(
        id="Total_Runtime_With_Citation",
        desc="Provides the total runtime of the show AND provides a valid reference URL supporting the runtime.",
        parent=group,
        critical=True,
    )
    urls_runtime = _urls_clean(perf.runtime_urls)
    claim_runtime = f"This page shows that the total runtime of '{show_name}' is '{runtime_total}'."
    await evaluator.verify(
        claim=claim_runtime,
        node=node_runtime,
        sources=urls_runtime if urls_runtime else None,
        additional_instruction=_mk_source_instruction(
            urls_runtime,
            "Allow minor rounding or formatting differences (e.g., '2h 30m' vs '150 minutes'). The stated duration should essentially match."
        ),
    )

    # Intermission with citation
    node_intermission = evaluator.add_leaf(
        id="Intermission_Details_If_Applicable_With_Citation",
        desc="Provides intermission information (either intermission duration or explicitly states there is no intermission) AND provides a valid reference URL supporting the intermission detail.",
        parent=group,
        critical=True,
    )
    urls_intermission = _urls_clean(perf.intermission_urls)
    claim_intermission = (
        f"This page confirms the intermission detail for '{show_name}': '{intermission_detail}'. "
        "If it says 'no intermission', that explicitly means none."
    )
    await evaluator.verify(
        claim=claim_intermission,
        node=node_intermission,
        sources=urls_intermission if urls_intermission else None,
        additional_instruction=_mk_source_instruction(
            urls_intermission,
            "Accept equivalent phrasing such as 'no intermission', 'runs straight through', or a specific intermission count/duration."
        ),
    )


async def _verify_rush_info(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    """
    Build and verify the 'Rush_Ticketing_Information' group:
      - Rush ticket price + citation
      - Where rush tickets can be purchased + citation
    """
    sel = data.selection or ShowSelection()
    rush = data.rush or RushInfo()

    show_name = _safe(sel.show_name, "the show")
    rush_price = _safe(rush.rush_price, "")
    rush_where = _safe(rush.rush_purchase_where, "")

    group = evaluator.add_parallel(
        id="Rush_Ticketing_Information",
        desc="Rush ticket price and purchase location are provided, each supported by at least one valid reference URL.",
        parent=parent_node,
        critical=True,
    )

    # Rush price with citation
    node_rush_price = evaluator.add_leaf(
        id="Rush_Ticket_Price_With_Citation",
        desc="Provides the price of rush tickets AND provides a valid reference URL supporting the rush price.",
        parent=group,
        critical=True,
    )
    urls_rush_price = _urls_clean(rush.rush_price_urls)
    claim_rush_price = f"This page shows that rush tickets for '{show_name}' cost '{rush_price}'."
    await evaluator.verify(
        claim=claim_rush_price,
        node=node_rush_price,
        sources=urls_rush_price if urls_rush_price else None,
        additional_instruction=_mk_source_instruction(
            urls_rush_price,
            "The price must refer to RUSH (e.g., box office rush, general rush, digital rush) and not regular tickets. "
            "Accept a clearly stated single price or range if the answer states a range."
        ),
    )

    # Rush purchase location with citation
    node_rush_where = evaluator.add_leaf(
        id="Where_To_Purchase_Rush_Tickets_With_Citation",
        desc="Provides where rush tickets can be purchased (e.g., box office, website/app) AND provides a valid reference URL supporting the purchase location/process.",
        parent=group,
        critical=True,
    )
    urls_rush_where = _urls_clean(rush.rush_where_urls)
    claim_rush_where = (
        f"This page indicates where rush tickets for '{show_name}' can be obtained: '{rush_where}'."
    )
    await evaluator.verify(
        claim=claim_rush_where,
        node=node_rush_where,
        sources=urls_rush_where if urls_rush_where else None,
        additional_instruction=_mk_source_instruction(
            urls_rush_where,
            "Accept sources that explicitly indicate the rush sales channel or process (e.g., 'TodayTix app', 'in-person at the box office', "
            "'day-of digital rush on TodayTix')."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Broadway show (TV/streaming adaptation) research task.
    """
    # Initialize evaluator with a neutral root, then add our critical sequential main node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root placeholder; we add a critical sequential child as the real root
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

    # Extract all needed fields in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_show_data(),
        template_class=ShowExtraction,
        extraction_name="extracted_show_info",
    )

    # Add a main critical sequential node to mirror rubric's root
    main = evaluator.add_sequential(
        id="Broadway_Show_Research_Task",
        desc="Identify one currently-playing Broadway show in NYC adapted from a streaming/TV series and provide venue address, runtime (with intermission details if applicable), rush ticket info, and supporting URLs.",
        parent=root,
        critical=True,
    )

    # Optional: record a summary snippet into the final breakdown for reference
    evaluator.add_custom_info(
        info={
            "show_name": (extracted.selection.show_name if extracted and extracted.selection else None),
            "theatre_name": (extracted.theatre.theatre_name if extracted and extracted.theatre else None),
            "runtime_total": (extracted.performance.runtime_total if extracted and extracted.performance else None),
            "rush_price": (extracted.rush.rush_price if extracted and extracted.rush else None),
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    # Build and verify each rubric group under the main node
    await _verify_show_selection(evaluator, main, extracted)
    await _verify_theatre_info(evaluator, main, extracted)
    await _verify_performance_details(evaluator, main, extracted)
    await _verify_rush_info(evaluator, main, extracted)

    # Return consolidated evaluation summary
    return evaluator.get_summary()