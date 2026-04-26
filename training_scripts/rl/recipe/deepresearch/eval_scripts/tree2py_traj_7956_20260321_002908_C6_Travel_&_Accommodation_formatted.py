import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "trip_planning_phx_mke_orlando_may2026"
TASK_DESCRIPTION = (
    "A music festival enthusiast living in Phoenix, Arizona is planning to visit their friend in Milwaukee, "
    "Wisconsin (a city on Lake Michigan) in early 2026. After the Milwaukee visit, they want to attend a major music "
    "festival in Orlando, Florida in May 2026. They prefer budget airlines for cost savings and need accommodation "
    "near the event venue.\n\n"
    "Identify the following for their trip planning:\n\n"
    "1. Budget Airline Route: Which budget airline (low-cost carrier) operates direct flights from Phoenix Sky Harbor "
    "International Airport (PHX) to Milwaukee Mitchell International Airport (MKE)?\n\n"
    "2. Lake Michigan Verification: Confirm that Milwaukee is indeed located on Lake Michigan.\n\n"
    "3. Orlando Festival: What is the name and exact dates of the major music festival taking place at Camping World "
    "Stadium in Orlando, Florida in May 2026? Also provide the venue's seating capacity.\n\n"
    "4. Hotel Accommodations: Identify three different hotels rated 3-star or higher that are located near Camping "
    "World Stadium in downtown Orlando, suitable for staying during the festival.\n\n"
    "For each component, provide supporting reference URLs from reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BudgetAirlineInfo(BaseModel):
    airline_name: Optional[str] = None
    low_cost_sources: List[str] = Field(default_factory=list)
    route_sources: List[str] = Field(default_factory=list)


class LakeMichiganInfo(BaseModel):
    sources: List[str] = Field(default_factory=list)


class FestivalInfo(BaseModel):
    festival_name: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    name_dates_sources: List[str] = Field(default_factory=list)
    venue_capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)


class HotelItem(BaseModel):
    name: Optional[str] = None
    star_rating_text: Optional[str] = None  # e.g., "3-star hotel", "4-star"
    rating_sources: List[str] = Field(default_factory=list)     # URLs supporting 3+ star rating
    proximity_sources: List[str] = Field(default_factory=list)  # URLs supporting near stadium & downtown


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_budget_airline() -> str:
    return """
    Extract details for a budget airline operating direct PHX→MKE flights from the answer text.

    Return a JSON object with:
    - airline_name: the name of the identified budget/low-cost carrier (string)
    - low_cost_sources: an array of URLs cited in the answer that support the airline's classification as a low-cost/budget carrier
                        (e.g., airline's Wikipedia page indicating LCC, reputable aviation sites listing it as LCC)
    - route_sources: an array of URLs cited in the answer that specifically support that this airline operates a direct/nonstop route
                     from Phoenix Sky Harbor International Airport (PHX) to Milwaukee Mitchell International Airport (MKE)
                     (e.g., airline schedule page, route map, airport route listings, press releases)

    Rules:
    - Extract only what is explicitly present in the answer; do not invent or infer.
    - For URLs, extract the actual URL strings exactly as provided (including those in markdown links).
    - If any field is missing in the answer, set it to null (for strings) or [] (for arrays).
    """


def prompt_extract_lake_michigan_sources() -> str:
    return """
    Extract URLs that the answer cites to support the fact that Milwaukee, Wisconsin is located on Lake Michigan.

    Return a JSON object:
    - sources: array of all such URLs explicitly present in the answer (empty array if none)

    Rules:
    - Extract only URLs that are actually present in the answer.
    - Include Wikipedia or official/state/city geography pages if they were cited.
    """


def prompt_extract_festival() -> str:
    return """
    Extract details for the Orlando music festival at Camping World Stadium in May 2026 from the answer.

    Return a JSON object with:
    - festival_name: the name of the festival (string or null)
    - date_start: the exact start date as a string as written in the answer (e.g., "May 10, 2026") or null if missing
    - date_end: the exact end date as a string as written in the answer (e.g., "May 12, 2026") or null if missing
    - name_dates_sources: array of URLs that corroborate the festival name and dates at Camping World Stadium (empty if none)
    - venue_capacity: Camping World Stadium seating capacity as a numeric-looking string as written in the answer (e.g., "60,000")
                      If missing, null.
    - capacity_sources: array of URLs that corroborate the seating capacity (empty if none)

    Rules:
    - Extract only information explicitly present in the answer without alteration.
    - Dates may be in various formats (e.g., "May 10–12, 2026" or start/end separately). If the answer provides a single range,
      set date_start to the start and date_end to the end if clearly indicated; otherwise keep whatever exact strings are provided.
    - For URLs, extract exact strings as provided (including those inside markdown links).
    """


def prompt_extract_hotels() -> str:
    return """
    Extract up to three distinct hotel entries near Camping World Stadium in/near downtown Orlando from the answer.

    For each hotel, extract:
    - name: the hotel name (string or null)
    - star_rating_text: the star rating text as written (e.g., "3-star hotel", "4-star"; string or null)
    - rating_sources: array of URLs supporting the 3+ star rating (e.g., booking sites, hotel official pages if they list star category)
    - proximity_sources: array of URLs supporting proximity to Camping World Stadium and in/near downtown Orlando

    Return a JSON object:
    - hotels: array of up to three hotel objects in the order they appear in the answer. If fewer than 3 are present, return as many as available.

    Rules:
    - Extract only what is explicitly present in the answer; do not infer.
    - For URLs, extract actual URL strings exactly as provided (including markdown links).
    - If a field is missing for a hotel, set it to null (string fields) or [] (URL arrays).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_nonempty_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _looks_numeric(text: Optional[str]) -> bool:
    if not _has_nonempty_text(text):
        return False
    return bool(re.search(r"\d", text or ""))


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_budget_airline(
    evaluator: Evaluator,
    parent,
    info: BudgetAirlineInfo,
) -> None:
    node = evaluator.add_parallel(
        id="Budget_Airline_Route_PHX_to_MKE",
        desc="Identify a low-cost/budget carrier operating a direct/nonstop PHX→MKE route, with supporting evidence URLs.",
        parent=parent,
        critical=True,
    )

    # Airline name provided
    evaluator.add_custom_node(
        result=_has_nonempty_text(info.airline_name),
        id="Airline_Name",
        desc="Provide the airline name.",
        parent=node,
        critical=True,
    )

    # Low-cost carrier verification: source provided + supported by URL(s)
    lc_src_exist = evaluator.add_custom_node(
        result=_has_any_url(info.low_cost_sources),
        id="Low_Cost_Carrier_Source_Provided",
        desc="At least one URL is provided to support low-cost/budget classification.",
        parent=node,
        critical=True,
    )

    lc_verify = evaluator.add_leaf(
        id="Low_Cost_Carrier_Verification_With_URL",
        desc="Provide a reliable reference URL supporting that the identified airline is classified as a low-cost/budget carrier.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The airline '{info.airline_name or ''}' is a low-cost/budget carrier (LCC).",
        node=lc_verify,
        sources=info.low_cost_sources if _has_any_url(info.low_cost_sources) else None,
        additional_instruction="Use the provided sources to confirm that the airline is classified as a low-cost/budget carrier. "
                               "Accept credible references such as Wikipedia airline pages or reputable aviation sources that explicitly label it as low-cost.",
    )

    # Direct PHX→MKE route verification: source provided + supported by URL(s)
    dr_src_exist = evaluator.add_custom_node(
        result=_has_any_url(info.route_sources),
        id="Direct_Route_Source_Provided",
        desc="At least one URL is provided to support direct/nonstop PHX→MKE operation for the identified airline.",
        parent=node,
        critical=True,
    )

    dr_verify = evaluator.add_leaf(
        id="Direct_Flight_PHX_to_MKE_Verification_With_URL",
        desc="Provide a reliable reference URL supporting that the identified airline operates a direct/nonstop PHX→MKE route (route listing/airline page/airport route map, etc.).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The airline '{info.airline_name or ''}' operates a direct (nonstop) route from Phoenix Sky Harbor International Airport (PHX) to Milwaukee Mitchell International Airport (MKE).",
        node=dr_verify,
        sources=info.route_sources if _has_any_url(info.route_sources) else None,
        additional_instruction="The page(s) should clearly show the airline operates the PHX–MKE route as direct/nonstop. "
                               "Accept synonyms like 'nonstop'. Ensure the route specifically references the same airline.",
    )

    # Ensure dependency order (critical siblings auto-gate subsequent verifications)
    _ = lc_src_exist
    _ = dr_src_exist


async def verify_lake_michigan(
    evaluator: Evaluator,
    parent,
    info: LakeMichiganInfo,
) -> None:
    node = evaluator.add_parallel(
        id="Lake_Michigan_Verification",
        desc="Confirm Milwaukee is located on Lake Michigan, with supporting evidence.",
        parent=parent,
        critical=True,
    )

    # 1) The answer explicitly states it (content presence check)
    stated_node = evaluator.add_leaf(
        id="Milwaukee_On_Lake_Michigan",
        desc="Explicitly state that Milwaukee is located on Lake Michigan.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that Milwaukee is located on Lake Michigan (e.g., 'Milwaukee is on Lake Michigan', "
              "'on the shores of Lake Michigan', or equivalent wording).",
        node=stated_node,
        additional_instruction="Examine only the provided answer text (not external sources). Allow equivalent phrasings like "
                               "'on the shore of Lake Michigan' or 'on Lake Michigan'.",
    )

    # 2) Supporting URL(s) exist
    evaluator.add_custom_node(
        result=_has_any_url(info.sources),
        id="Lake_Michigan_Supporting_URL_Provided",
        desc="Provide at least one reliable reference URL confirming Milwaukee’s location on Lake Michigan.",
        parent=node,
        critical=True,
    )

    # 3) Fact supported by the provided URL(s)
    fact_node = evaluator.add_leaf(
        id="Supporting_URL_For_Lake_Michigan_Location",
        desc="Provide at least one reliable reference URL confirming Milwaukee’s location on Lake Michigan.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Milwaukee, Wisconsin is located on the shore of Lake Michigan.",
        node=fact_node,
        sources=info.sources if _has_any_url(info.sources) else None,
        additional_instruction="Confirm from the provided URL(s) that Milwaukee is on Lake Michigan. Accept explicit statements or official/city references.",
    )


async def verify_festival(
    evaluator: Evaluator,
    parent,
    info: FestivalInfo,
) -> None:
    node = evaluator.add_parallel(
        id="Orlando_Festival_Camping_World_Stadium_May_2026",
        desc="Identify the major music festival at Camping World Stadium in May 2026, give exact dates, and provide the stadium seating capacity, with sources.",
        parent=parent,
        critical=True,
    )

    # Festival name provided
    evaluator.add_custom_node(
        result=_has_nonempty_text(info.festival_name),
        id="Festival_Name",
        desc="Provide the festival name.",
        parent=node,
        critical=True,
    )

    # Exact dates provided and in May 2026
    dates_provided = evaluator.add_custom_node(
        result=_has_nonempty_text(info.date_start) and _has_nonempty_text(info.date_end),
        id="Festival_Dates_Provided",
        desc="Festival start and end dates are provided.",
        parent=node,
        critical=True,
    )

    dates_in_may = evaluator.add_leaf(
        id="Festival_Exact_Dates_May_2026",
        desc="Provide the exact start and end dates (must fall in May 2026).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The festival dates from '{info.date_start or ''}' to '{info.date_end or ''}' fall entirely within May 2026.",
        node=dates_in_may,
        additional_instruction="Judge using only the strings provided for start/end. Consider common date formats. "
                               "Both start and end must be in May 2026.",
    )

    # Name & dates supported by URL(s)
    evaluator.add_custom_node(
        result=_has_any_url(info.name_dates_sources),
        id="Festival_Name_Dates_Source_Provided",
        desc="At least one reliable URL is provided to corroborate festival name and exact dates at Camping World Stadium.",
        parent=node,
        critical=True,
    )
    name_dates_node = evaluator.add_leaf(
        id="Festival_Name_And_Dates_Supporting_URL",
        desc="Provide at least one reliable reference URL corroborating the festival name and exact dates.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The festival '{info.festival_name or ''}' will take place at Camping World Stadium in Orlando, Florida "
              f"from {info.date_start or ''} to {info.date_end or ''}.",
        node=name_dates_node,
        sources=info.name_dates_sources if _has_any_url(info.name_dates_sources) else None,
        additional_instruction="The page(s) should explicitly name the festival, confirm Camping World Stadium as the venue, "
                               "and show the exact dates matching those provided.",
    )

    # Venue seating capacity: numeric value + supported by URL(s)
    evaluator.add_custom_node(
        result=_looks_numeric(info.venue_capacity),
        id="Venue_Seating_Capacity",
        desc="Provide Camping World Stadium’s seating capacity as a numeric value.",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_any_url(info.capacity_sources),
        id="Venue_Capacity_Source_Provided",
        desc="At least one reliable URL is provided to corroborate Camping World Stadium seating capacity.",
        parent=node,
        critical=True,
    )
    capacity_node = evaluator.add_leaf(
        id="Venue_Capacity_Supporting_URL",
        desc="Provide at least one reliable reference URL corroborating Camping World Stadium seating capacity.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Camping World Stadium has a seating capacity of {info.venue_capacity or ''}.",
        node=capacity_node,
        sources=info.capacity_sources if _has_any_url(info.capacity_sources) else None,
        additional_instruction="Confirm the stated capacity from the provided URL(s). If multiple configurations exist, "
                               "the stated number must be directly supported (or clearly equivalent) by the source.",
    )

    _ = dates_provided  # ensure presence as a critical sibling precondition


async def verify_one_hotel(
    evaluator: Evaluator,
    parent,
    hotel: HotelItem,
    index: int,
) -> None:
    hid = f"Hotel_{index}"
    hnode = evaluator.add_parallel(
        id=hid,
        desc=f"{['First','Second','Third'][index-1]} qualifying hotel.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_nonempty_text(hotel.name),
        id=f"{hid}_Name",
        desc="Provide the hotel name.",
        parent=hnode,
        critical=True,
    )

    # Star rating 3+ evidence
    evaluator.add_custom_node(
        result=_has_any_url(hotel.rating_sources),
        id=f"{hid}_Star_Rating_Source_Provided",
        desc="At least one URL is provided that shows the hotel is 3-star or higher.",
        parent=hnode,
        critical=True,
    )
    rating_leaf = evaluator.add_leaf(
        id=f"{hid}_Star_Rating_3_Plus_With_URL",
        desc="Provide evidence via a reliable reference URL that the hotel is rated 3-star or higher.",
        parent=hnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or ''}' has a star rating of 3 stars or higher.",
        node=rating_leaf,
        sources=hotel.rating_sources if _has_any_url(hotel.rating_sources) else None,
        additional_instruction="Use the provided page(s) to verify the star category is at least 3. "
                               "Accept phrases like '3-star hotel', '4-star', etc. Ignore user review scores; focus on star-category where shown.",
    )

    # Proximity & downtown evidence
    evaluator.add_custom_node(
        result=_has_any_url(hotel.proximity_sources),
        id=f"{hid}_Proximity_Source_Provided",
        desc="At least one URL is provided showing the hotel is near Camping World Stadium and in/near downtown Orlando.",
        parent=hnode,
        critical=True,
    )
    prox_leaf = evaluator.add_leaf(
        id=f"{hid}_Near_Venue_And_Downtown_With_URL",
        desc="Provide evidence via a reliable reference URL that the hotel is near Camping World Stadium and is in/near downtown Orlando (as claimed).",
        parent=hnode,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or ''}' is located near Camping World Stadium and is in or near downtown Orlando, Florida.",
        node=prox_leaf,
        sources=hotel.proximity_sources if _has_any_url(hotel.proximity_sources) else None,
        additional_instruction="From the provided page(s), confirm proximity to Camping World Stadium (e.g., within ~2 miles or a short drive/walk) "
                               "and that the property is in or near Downtown Orlando.",
    )


async def verify_hotels(
    evaluator: Evaluator,
    parent,
    hotels_info: HotelsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_Accommodations_Near_Camping_World_Stadium",
        desc="Identify three different hotels rated 3-star or higher, near Camping World Stadium, in/near downtown Orlando; each hotel must have supporting URLs for rating and proximity/location.",
        parent=parent,
        critical=True,
    )

    # Take exactly three entries (pad with blanks if needed)
    hotels = list(hotels_info.hotels[:3])
    while len(hotels) < 3:
        hotels.append(HotelItem())

    # Verify each hotel block
    for i, h in enumerate(hotels, start=1):
        await verify_one_hotel(evaluator, node, h, i)

    # Distinctness check
    distinct_leaf = evaluator.add_leaf(
        id="Three_Different_Hotels",
        desc="Verify the three hotels are distinct properties (no duplicates).",
        parent=node,
        critical=True,
    )
    hnames = [h.name or "" for h in hotels]
    await evaluator.verify(
        claim=f"The three hotel names represent three different properties (no duplicates or alternate names for the same property): "
              f"'{hnames[0]}', '{hnames[1]}', '{hnames[2]}'.",
        node=distinct_leaf,
        additional_instruction="Consider minor formatting or brand suffix differences; if two names refer to the same physical property, treat as duplicates.",
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
    Evaluate an answer for the PHX→MKE + Orlando May 2026 festival trip planning task.
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

    # Top-level aggregator for the whole task
    trip_node = evaluator.add_parallel(
        id="Trip_Planning_Task",
        desc="Provide: (1) budget airline direct PHX→MKE route, (2) confirm Milwaukee is on Lake Michigan, (3) Orlando May-2026 festival at Camping World Stadium with dates + capacity, (4) three distinct 3-star+ hotels near the stadium in/near downtown Orlando; include supporting reference URLs as required.",
        parent=root,
        critical=False,  # keep non-critical to avoid forcing all descendants critical due to framework rule
    )

    # Run extractions (in parallel)
    budget_task = evaluator.extract(
        prompt=prompt_extract_budget_airline(),
        template_class=BudgetAirlineInfo,
        extraction_name="budget_airline_info",
    )
    lake_task = evaluator.extract(
        prompt=prompt_extract_lake_michigan_sources(),
        template_class=LakeMichiganInfo,
        extraction_name="lake_michigan_info",
    )
    festival_task = evaluator.extract(
        prompt=prompt_extract_festival(),
        template_class=FestivalInfo,
        extraction_name="festival_info",
    )
    hotels_task = evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_info",
    )

    budget_info, lake_info, fest_info, hotels_info = await asyncio.gather(
        budget_task, lake_task, festival_task, hotels_task
    )

    # Build verification subtrees
    await verify_budget_airline(evaluator, trip_node, budget_info)
    await verify_lake_michigan(evaluator, trip_node, lake_info)
    await verify_festival(evaluator, trip_node, fest_info)
    await verify_hotels(evaluator, trip_node, hotels_info)

    return evaluator.get_summary()