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
TASK_ID = "europe_trip_2026_planning"
TASK_DESCRIPTION = """
Plan a 2026 European research trip: select an eligible atmospheric science conference, an eclipse-totality country, and a MacBook with Apple education pricing, each with required reference URLs.
"""
ECLIPSE_DATE_STR = "August 12, 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # Free-form (e.g., "Vienna, Austria")
    country: Optional[str] = None   # Country if explicitly stated in the answer
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    field_or_theme: Optional[str] = None  # e.g., "atmospheric science", "meteorology"
    urls: List[str] = Field(default_factory=list)  # Conference detail URLs (dates page acceptable)


class EclipseInfo(BaseModel):
    country: Optional[str] = None  # Expected: Iceland, Spain, or Portugal
    urls: List[str] = Field(default_factory=list)  # Eclipse/path-of-totality reference URLs


class MacbookInfo(BaseModel):
    model: Optional[str] = None  # e.g., "MacBook Air 13-inch (M3)"
    education_price: Optional[str] = None  # String exactly as stated in the answer (e.g., "$899")
    regular_price: Optional[str] = None    # String as stated in the answer (e.g., "$999")
    education_urls: List[str] = Field(default_factory=list)  # Apple Education pricing page for this model
    regular_urls: List[str] = Field(default_factory=list)    # Optional: standard Apple product page URLs


class LogisticsInfo(BaseModel):
    time_gap_statement: Optional[str] = None  # Verbatim statement about gap or dates
    feasibility_statement: Optional[str] = None  # Verbatim feasibility statement (e.g., "enough time to travel")


class TripPlanExtraction(BaseModel):
    conference: Optional[ConferenceInfo] = None
    eclipse: Optional[EclipseInfo] = None
    macbook: Optional[MacbookInfo] = None
    logistics: Optional[LogisticsInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
Extract from the answer the single recommended option for each category (conference, eclipse site, and MacBook purchase), exactly as stated in the answer.

Return a JSON object with fields: conference, eclipse, macbook, logistics.

For "conference", extract:
- name: the conference name
- location: free-form location string if present (e.g., "Vienna, Austria"); else null
- country: the country explicitly mentioned for the conference, if present; else null
- start_date: event start date string exactly as written; else null
- end_date: event end date string exactly as written; else null
- field_or_theme: the field or theme (e.g., "atmospheric science", "meteorology", "climate", etc.) if explicitly indicated; else null
- urls: all conference-related reference URLs cited for details/dates (array; may be one or more). Extract only actual URLs present in the answer.

For "eclipse", extract:
- country: the selected European country to view the Aug 12, 2026 total solar eclipse (as stated)
- urls: all eclipse/path-of-totality reference URLs cited (array; may be one or more)

For "macbook", extract:
- model: specific MacBook model name
- education_price: education price string exactly as written (include currency symbol/format if present)
- regular_price: regular/standard retail price string exactly as written (include currency symbol/format if present)
- education_urls: Apple Education pricing URL(s) for the chosen model (array)
- regular_urls: Apple standard product page URL(s) for the chosen model if present (array; may be empty)

For "logistics", extract:
- time_gap_statement: the sentence/phrase where the answer explicitly states the time gap (e.g., 'X days/weeks') OR explicitly lists both the conference end date and 'August 12, 2026' enabling the reader to infer the gap; else null
- feasibility_statement: the sentence/phrase explicitly stating that the time gap is sufficient/feasible for travel logistics; else null

Rules:
- Extract only what the answer explicitly states. Do not infer or invent.
- For any missing item, set null (or empty array for URL lists).
- Keep all date strings and price strings exactly as written in the answer.
- Only include valid URLs that appear in the answer text (including markdown links).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nz(s: Optional[str]) -> str:
    return s if s else ""


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_conference(evaluator: Evaluator, parent_node, conf: Optional[ConferenceInfo]) -> None:
    node = evaluator.add_parallel(
        id="Conference_Selection",
        desc="Identify an atmospheric science conference in Europe meeting all stated date/verification constraints and provide a reference URL with details.",
        parent=parent_node,
        critical=True
    )

    # 1) Conference_Details_URL_Provided (existence check)
    details_url_provided = evaluator.add_custom_node(
        result=bool(conf and conf.urls and len(conf.urls) > 0),
        id="Conference_Details_URL_Provided",
        desc="Provide a reference URL that confirms conference details including dates.",
        parent=node,
        critical=True
    )

    # 2) Conference_Is_In_Europe (logical/world knowledge check)
    in_europe_leaf = evaluator.add_leaf(
        id="Conference_Is_In_Europe",
        desc="Conference location is in a European country.",
        parent=node,
        critical=True
    )
    loc = conf.country or conf.location or ""
    await evaluator.verify(
        claim=f"The conference location '{loc}' is in Europe.",
        node=in_europe_leaf,
        additional_instruction="Use your general world knowledge to judge whether the specified country/location is in Europe. If the location/country is missing or ambiguous, judge Incorrect."
    )

    # 3) Conference_Is_Atmospheric_Science (verify via URLs)
    atm_leaf = evaluator.add_leaf(
        id="Conference_Is_Atmospheric_Science",
        desc="Conference is an atmospheric science conference (or clearly within atmospheric science).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference '{nz(conf.name)}' is an atmospheric science conference or clearly within the atmospheric sciences domain (meteorology, climate, aerosols, atmospheric chemistry, etc.).",
        node=atm_leaf,
        sources=non_empty_urls(conf.urls),
        additional_instruction="Look for explicit mentions of atmospheric science, meteorology, climate, or related atmospheric topics on the referenced conference page."
    )

    # 4) Conference_Has_Verifiable_2026_Dates (verify via URLs)
    dates_leaf = evaluator.add_leaf(
        id="Conference_Has_Verifiable_2026_Dates",
        desc="Conference has publicly available and verifiable dates in 2026.",
        parent=node,
        critical=True
    )
    if conf and (conf.start_date or conf.end_date):
        claim_dates = f"The conference '{nz(conf.name)}' occurs in 2026 with dates '{nz(conf.start_date)}' to '{nz(conf.end_date)}'."
    else:
        claim_dates = f"The conference '{nz(conf.name)}' occurs in the year 2026."
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=non_empty_urls(conf.urls),
        additional_instruction="Confirm from the conference page that the event's dates are in calendar year 2026."
    )

    # 5) Conference_Ends_Before_Aug_12_2026 (verify via URLs + reasoning)
    ends_before_leaf = evaluator.add_leaf(
        id="Conference_Ends_Before_Aug_12_2026",
        desc="Conference end date is before August 12, 2026.",
        parent=node,
        critical=True
    )
    end_date_str = nz(conf.end_date if conf else None)
    await evaluator.verify(
        claim=f"The conference '{nz(conf.name)}' ends on '{end_date_str}', which is before {ECLIPSE_DATE_STR}.",
        node=ends_before_leaf,
        sources=non_empty_urls(conf.urls),
        additional_instruction=f"First, extract the conference end date from the provided URL(s). Then verify it's strictly earlier than {ECLIPSE_DATE_STR}. If the end date isn't shown or isn't earlier, judge Incorrect."
    )


async def verify_eclipse(evaluator: Evaluator, parent_node, eclipse: Optional[EclipseInfo]) -> None:
    node = evaluator.add_parallel(
        id="Eclipse_Observation_Site",
        desc="Select a European eclipse-viewing country within the Aug 12, 2026 path of totality and provide a reference URL.",
        parent=parent_node,
        critical=True
    )

    # 1) Eclipse_Information_URL_Provided (existence check)
    url_provided = evaluator.add_custom_node(
        result=bool(eclipse and eclipse.urls and len(eclipse.urls) > 0),
        id="Eclipse_Information_URL_Provided",
        desc="Provide a reference URL confirming eclipse visibility/path-of-totality information.",
        parent=node,
        critical=True
    )

    # 2) Eclipse_Country_Within_Totality_Path (verify via URLs)
    totality_leaf = evaluator.add_leaf(
        id="Eclipse_Country_Within_Totality_Path",
        desc="Selected country is within the path of totality for Aug 12, 2026 (Iceland, Spain, or Portugal).",
        parent=node,
        critical=True
    )
    country = nz(eclipse.country if eclipse else None)
    await evaluator.verify(
        claim=f"The selected country '{country}' lies within the path of totality for the total solar eclipse on {ECLIPSE_DATE_STR}.",
        node=totality_leaf,
        sources=non_empty_urls(eclipse.urls),
        additional_instruction="Accept only if the page clearly indicates that the path of totality crosses the named country. Note that for Aug 12, 2026 in Europe, acceptable countries include Iceland, Spain, and Portugal. If the country is different or unclear, judge Incorrect."
    )


async def verify_macbook(evaluator: Evaluator, parent_node, mac: Optional[MacbookInfo]) -> None:
    node = evaluator.add_parallel(
        id="Student_Equipment_Purchase",
        desc="Identify a MacBook model purchasable with Apple education pricing, show education vs regular price, and provide a pricing reference URL.",
        parent=parent_node,
        critical=True
    )

    # 1) MacBook_Model_Identified (existence check)
    model_node = evaluator.add_custom_node(
        result=bool(mac and mac.model and mac.model.strip()),
        id="MacBook_Model_Identified",
        desc="A specific MacBook model is identified.",
        parent=node,
        critical=True
    )

    # 2) Education_Pricing_URL_Provided (existence check)
    edu_url_node = evaluator.add_custom_node(
        result=bool(mac and mac.education_urls and len(mac.education_urls) > 0),
        id="Education_Pricing_URL_Provided",
        desc="Provide a reference URL confirming the Apple education pricing for the chosen MacBook model.",
        parent=node,
        critical=True
    )

    # 3) Qualifies_For_Apple_Education_Pricing (verify via education pricing URL)
    qualifies_leaf = evaluator.add_leaf(
        id="Qualifies_For_Apple_Education_Pricing",
        desc="The purchase qualifies for Apple's education pricing available to students/educators.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Apple offers education pricing for the MacBook model '{nz(mac.model if mac else None)}' to students and educators via the Apple Education Store.",
        node=qualifies_leaf,
        sources=non_empty_urls(mac.education_urls if mac else None),
        additional_instruction="Confirm from the education pricing page that the model is offered with education pricing for students/educators (e.g., Apple Education Store)."
    )

    # 4) Education_Price_Stated (verify via education pricing URL)
    edu_price_leaf = evaluator.add_leaf(
        id="Education_Price_Stated",
        desc="The education price for the selected MacBook model is stated.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Apple education price for '{nz(mac.model if mac else None)}' is '{nz(mac.education_price if mac else None)}'.",
        node=edu_price_leaf,
        sources=non_empty_urls(mac.education_urls if mac else None),
        additional_instruction="Verify that the education price amount (allowing minor currency/format variants, 'from' pricing) matches the claim for the specified model."
    )

    # 5) Regular_Price_Stated (presence in the answer)
    regular_price_leaf = evaluator.add_leaf(
        id="Regular_Price_Stated",
        desc="The regular retail price for the selected MacBook model is stated.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states the regular (non-education) retail price '{nz(mac.regular_price if mac else None)}' for the selected MacBook model.",
        node=regular_price_leaf,
        additional_instruction="Judge based on the answer text: pass if a numeric regular price is explicitly given (distinct from the education price), even if a URL for regular price is not provided."
    )

    # 6) Education_Price_Lower_Than_Regular (logical comparison)
    lower_than_leaf = evaluator.add_leaf(
        id="Education_Price_Lower_Than_Regular",
        desc="The education price is lower than the regular retail price for the same model.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The education price '{nz(mac.education_price if mac else None)}' is lower than the regular price '{nz(mac.regular_price if mac else None)}' for the same MacBook model.",
        node=lower_than_leaf,
        additional_instruction="Compare the numeric amounts while ignoring currency symbols and minor formatting. If either price is missing or not strictly lower, judge Incorrect."
    )


async def verify_logistics(evaluator: Evaluator, parent_node, logistics: Optional[LogisticsInfo], conf: Optional[ConferenceInfo]) -> None:
    node = evaluator.add_parallel(
        id="Trip_Logistics",
        desc="Satisfy the stated travel-logistics time constraint between the conference end date and the Aug 12, 2026 eclipse date.",
        parent=parent_node,
        critical=True
    )

    # 1) Time_Gap_Between_Conference_End_And_Eclipse_Stated (answer content check)
    gap_leaf = evaluator.add_leaf(
        id="Time_Gap_Between_Conference_End_And_Eclipse_Stated",
        desc="Answer explicitly states the time gap (e.g., number of days/weeks or the relevant dates) between the conference end date and August 12, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states the interval between the conference end date '{nz(conf.end_date if conf else None)}' and {ECLIPSE_DATE_STR}, either as a number of days/weeks or by explicitly listing both dates.",
        node=gap_leaf,
        additional_instruction="Evaluate only the answer text. Accept either (a) an explicit numeric gap (days/weeks) or (b) an explicit statement of both dates enabling the reader to infer the gap."
    )

    # 2) Logistics_Feasibility_Addressed (answer content check)
    feas_leaf = evaluator.add_leaf(
        id="Logistics_Feasibility_Addressed",
        desc="Answer explicitly addresses that the gap is sufficient for travel logistics (i.e., includes an explicit feasibility statement tied to the stated time gap).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the time gap is sufficient/feasible for travel logistics (e.g., mentions having enough time to travel/prepare between the conference end and the eclipse date).",
        node=feas_leaf,
        additional_instruction="Judge based on the answer text. Look for explicit feasibility phrasing tied to the stated time gap/dates (e.g., 'enough time', 'plenty of time', 'feasible to travel')."
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
    # Initialize evaluator with a parallel root (root is non-critical by framework design; children will be critical)
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

    # Extract structured trip plan info
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Add custom info for global constraints
    evaluator.add_custom_info(
        info={"eclipse_date": ECLIPSE_DATE_STR},
        info_type="global_constraints",
        info_name="event_constraints"
    )

    # Build and verify subtrees
    await verify_conference(evaluator, root, extracted.conference or ConferenceInfo())
    await verify_eclipse(evaluator, root, extracted.eclipse or EclipseInfo())
    await verify_macbook(evaluator, root, extracted.macbook or MacbookInfo())
    await verify_logistics(evaluator, root, extracted.logistics or LogisticsInfo(), extracted.conference or ConferenceInfo())

    # Return structured evaluation summary
    return evaluator.get_summary()