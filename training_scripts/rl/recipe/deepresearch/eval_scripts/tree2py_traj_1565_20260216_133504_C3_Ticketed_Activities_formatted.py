import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ballet_lincoln_center_2026"
TASK_DESCRIPTION = (
    "I am planning to attend a professional ballet performance in New York City during my visit in late winter/early spring 2026. "
    "I prefer to experience performances at Lincoln Center's premier ballet venues, specifically those with a seating capacity "
    "between 2,000 and 4,000 seats, as these provide an optimal viewing experience for dance.\n\n"
    "Please identify one professional ballet performance that meets the following criteria:\n"
    "- Must be a ballet performance by a professionally recognized ballet company\n"
    "- Must take place at Lincoln Center in New York City\n"
    "- The venue must have a seating capacity between 2,000 and 4,000 seats\n"
    "- Must have performances scheduled between February 1, 2026 and March 31, 2026\n\n"
    "For the performance you identify, please provide:\n"
    "1. The name of the ballet company\n"
    "2. The title of the ballet or program being performed\n"
    "3. The official name of the venue\n"
    "4. Confirmation that the venue is located at Lincoln Center\n"
    "5. The venue's seating capacity\n"
    "6. The complete physical address of the venue\n"
    "7. At least one specific performance date within the February 1 - March 31, 2026 timeframe\n"
    "8. Information about how to purchase tickets, including the official ticketing website\n"
    "9. URL references supporting all of the above information"
)

DATE_RANGE_TEXT = "between February 1, 2026 and March 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BalletCompanyInfo(BaseModel):
    company_name: Optional[str] = None
    professional_status_note: Optional[str] = None
    company_urls: List[str] = Field(default_factory=list)


class PerformanceDetails(BaseModel):
    performance_title: Optional[str] = None
    performance_urls: List[str] = Field(default_factory=list)
    schedule_dates: List[str] = Field(default_factory=list)  # Extract as strings (e.g., "March 5, 2026")
    schedule_urls: List[str] = Field(default_factory=list)


class VenueDetails(BaseModel):
    venue_name: Optional[str] = None
    lincoln_center_confirmation: Optional[str] = None  # e.g., "Yes", "Located at Lincoln Center"
    venue_urls: List[str] = Field(default_factory=list)
    venue_address: Optional[str] = None
    venue_capacity: Optional[str] = None  # Keep as a string to allow ranges or approx values
    capacity_urls: List[str] = Field(default_factory=list)


class TicketingDetails(BaseModel):
    purchase_method: Optional[str] = None  # e.g., "Online via lincolncenter.org", "Box office"
    ticketing_url: Optional[str] = None


class BalletPerformanceExtraction(BaseModel):
    company: Optional[BalletCompanyInfo] = None
    performance: Optional[PerformanceDetails] = None
    venue: Optional[VenueDetails] = None
    ticketing: Optional[TicketingDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ballet_performance() -> str:
    return """
    Extract information for exactly one professional ballet performance described in the answer that is intended to take place at Lincoln Center in NYC in February–March 2026. 
    If multiple performances are mentioned, choose the first one that has clear supporting details and URLs and appears to meet the requirements. 
    Return the following JSON fields:

    company:
      - company_name: Official name of the ballet company
      - professional_status_note: Any statement indicating professional recognition (e.g., "major company", "resident company", "internationally acclaimed")
      - company_urls: All URLs in the answer that refer to the company's official site or authoritative pages (can be zero or more)

    performance:
      - performance_title: Title of the ballet or program
      - performance_urls: All URLs that specifically describe the performance (e.g., Lincoln Center event page, company page for the program)
      - schedule_dates: A list of specific performance dates explicitly mentioned (strings, e.g., "March 5, 2026"; include at least one if available)
      - schedule_urls: All URLs that list dates/times for the performance (may overlap with performance_urls)

    venue:
      - venue_name: Official name of the venue (e.g., "David H. Koch Theater")
      - lincoln_center_confirmation: A short phrase from the answer confirming it's at Lincoln Center (e.g., "at Lincoln Center", "Lincoln Center campus") if present; else null
      - venue_urls: All URLs about the venue identification/location (e.g., venue page on Lincoln Center, official theater page)
      - venue_address: Full street address of the venue if provided (e.g., "20 Lincoln Center Plaza, New York, NY 10023")
      - venue_capacity: The seating capacity stated (keep as text, may be approximate or a range)
      - capacity_urls: All URLs that specifically reference seating capacity (if none, provide an empty list; do not invent)

    ticketing:
      - purchase_method: How tickets can be purchased (short note extracted from the answer, e.g., "online via venue's site")
      - ticketing_url: The primary official URL for purchasing tickets or finding ticket info (prefer Lincoln Center or the company's official ticketing page)

    Rules:
    - Extract ONLY what is explicitly present in the answer. Do not invent.
    - For any missing field, return null (for strings) or [] for lists.
    - Extract full URLs including protocol, and keep them as-is.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    merged.append(uu)
    return merged


def pick_first_date(perf: Optional[PerformanceDetails]) -> Optional[str]:
    if not perf or not perf.schedule_dates:
        return None
    # Prefer a clean date string; just return the first entry
    return perf.schedule_dates[0].strip() if perf.schedule_dates[0] else None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_performance_identification_and_schedule(
    evaluator: Evaluator,
    parent_node,
    data: BalletPerformanceExtraction,
) -> None:
    perf_main = evaluator.add_sequential(
        id="Performance_Identification_And_Schedule",
        desc="Identify the ballet performance, company, and verify the performance schedule",
        parent=parent_node,
        critical=True,
    )

    # Company and Performance Details (parallel, critical)
    comp_perf = evaluator.add_parallel(
        id="Company_And_Performance_Details",
        desc="Identify the ballet company and specific performance",
        parent=perf_main,
        critical=True,
    )

    # Company Information (parallel, critical)
    company_info = evaluator.add_parallel(
        id="Company_Information",
        desc="Verify the ballet company details",
        parent=comp_perf,
        critical=True,
    )

    company_name_present = bool(data.company and data.company.company_name and data.company.company_name.strip())
    evaluator.add_custom_node(
        result=company_name_present,
        id="Company_Name",
        desc="Provide the official name of the ballet company",
        parent=company_info,
        critical=True,
    )

    company_prof_leaf = evaluator.add_leaf(
        id="Company_Professional_Status",
        desc="Confirm the company is a professionally recognized ballet company",
        parent=company_info,
        critical=True,
    )
    company_sources = merge_urls(
        data.company.company_urls if data.company else [],
        data.performance.performance_urls if data.performance else [],
    )
    company_name = data.company.company_name if data.company and data.company.company_name else ""
    prof_claim = f"The ballet company '{company_name}' is a professionally recognized ballet company."
    await evaluator.verify(
        claim=prof_claim,
        node=company_prof_leaf,
        sources=company_sources,
        additional_instruction=(
            "Confirm via the provided webpages that this organization is a professional ballet company "
            "(e.g., major resident company, well-known company with professional seasons). "
            "Allow reasonable phrasing variations. If the pages are irrelevant or do not support professional status, mark as not supported."
        ),
    )

    # Performance Title (critical leaf under comp_perf)
    perf_title_leaf = evaluator.add_leaf(
        id="Performance_Title",
        desc="Provide the specific title of the ballet or program being performed",
        parent=comp_perf,
        critical=True,
    )
    perf_title = data.performance.performance_title if data.performance and data.performance.performance_title else ""
    perf_title_claim = f"The performance/program title is '{perf_title}' for the ballet company '{company_name}'."
    await evaluator.verify(
        claim=perf_title_claim,
        node=perf_title_leaf,
        sources=(data.performance.performance_urls if data.performance else []),
        additional_instruction=(
            "Check the event/program page(s) to confirm the title. "
            "Minor naming variants, punctuation, and capitalization differences are acceptable if clearly the same program."
        ),
    )

    # Performance URL Reference (critical leaf)
    perf_url_leaf = evaluator.add_leaf(
        id="Performance_URL_Reference",
        desc="Provide a URL reference confirming the performance details",
        parent=comp_perf,
        critical=True,
    )
    perf_url_claim = (
        f"The provided page(s) explicitly describe the performance '{perf_title}' by '{company_name}', "
        f"including key details (title/company)."
    )
    await evaluator.verify(
        claim=perf_url_claim,
        node=perf_url_leaf,
        sources=(data.performance.performance_urls if data.performance else []),
        additional_instruction=(
            "Verify that the cited URLs are directly about the performance (event/program page) "
            "and contain clear, explicit information matching the described title and company."
        ),
    )

    # Schedule Verification (parallel, critical)
    schedule_node = evaluator.add_parallel(
        id="Schedule_Verification",
        desc="Verify the performance schedule meets temporal requirements",
        parent=perf_main,
        critical=True,
    )

    # Specific Dates (critical leaf)
    specific_date_leaf = evaluator.add_leaf(
        id="Specific_Dates",
        desc="Provide at least one specific performance date",
        parent=schedule_node,
        critical=True,
    )
    first_date = pick_first_date(data.performance)
    date_company = company_name
    date_perf_title = perf_title
    date_claim = (
        f"There is a scheduled performance date for '{date_company}' performing '{date_perf_title}' on '{first_date}'."
        if first_date
        else "No specific date provided; this claim should be judged as not supported."
    )
    await evaluator.verify(
        claim=date_claim,
        node=specific_date_leaf,
        sources=(data.performance.schedule_urls if data.performance else []),
        additional_instruction=(
            "Check the schedule/date listing pages to confirm at least one explicit performance date. "
            "If the claim shows 'No specific date provided', mark as not supported."
        ),
    )

    # Date Range Compliance (critical leaf)
    range_leaf = evaluator.add_leaf(
        id="Date_Range_Compliance",
        desc=f"Confirm the performance date(s) fall {DATE_RANGE_TEXT}",
        parent=schedule_node,
        critical=True,
    )
    range_claim = (
        f"The performance date '{first_date}' is {DATE_RANGE_TEXT}."
        if first_date
        else f"The performance date is missing; therefore it does not comply with being {DATE_RANGE_TEXT}."
    )
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        sources=None,  # Logical check based on the extracted date text
        additional_instruction=(
            "Judge the claim purely on whether the provided date text falls between Feb 1, 2026 and Mar 31, 2026. "
            "If the date is missing, the claim should be marked incorrect."
        ),
    )

    # Schedule URL Reference (critical leaf)
    schedule_url_leaf = evaluator.add_leaf(
        id="Schedule_URL_Reference",
        desc="Provide a URL reference confirming the performance dates",
        parent=schedule_node,
        critical=True,
    )
    schedule_url_claim = (
        f"The provided page(s) list dates/times for '{date_company}' performing '{date_perf_title}' in Feb–Mar 2026."
    )
    await evaluator.verify(
        claim=schedule_url_claim,
        node=schedule_url_leaf,
        sources=(data.performance.schedule_urls if data.performance else []),
        additional_instruction=(
            "Verify that the URLs actually provide performance date listings (calendar/schedule) for this program in the specified period. "
            "If none of the pages provide such date listings, mark as not supported."
        ),
    )


async def build_venue_requirements(
    evaluator: Evaluator,
    parent_node,
    data: BalletPerformanceExtraction,
) -> None:
    venue_main = evaluator.add_sequential(
        id="Venue_Requirements",
        desc="Verify the venue meets all specified requirements",
        parent=parent_node,
        critical=True,
    )

    # Venue Identification (parallel, critical)
    venue_ident = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Identify and verify the venue details",
        parent=venue_main,
        critical=True,
    )

    # Venue Basic Info (parallel, critical)
    venue_basic = evaluator.add_parallel(
        id="Venue_Basic_Info",
        desc="Provide basic venue identification information",
        parent=venue_ident,
        critical=True,
    )

    venue_name_present = bool(data.venue and data.venue.venue_name and data.venue.venue_name.strip())
    evaluator.add_custom_node(
        result=venue_name_present,
        id="Venue_Name",
        desc="Provide the official name of the venue",
        parent=venue_basic,
        critical=True,
    )

    lincoln_center_leaf = evaluator.add_leaf(
        id="Lincoln_Center_Location",
        desc="Confirm the venue is located at Lincoln Center",
        parent=venue_basic,
        critical=True,
    )
    venue_name = data.venue.venue_name if data.venue and data.venue.venue_name else ""
    lc_claim = f"The venue '{venue_name}' is located at Lincoln Center in New York City."
    await evaluator.verify(
        claim=lc_claim,
        node=lincoln_center_leaf,
        sources=(data.venue.venue_urls if data.venue else []),
        additional_instruction=(
            "Use the provided venue pages to confirm that the venue is part of (or located at) Lincoln Center in NYC. "
            "Reasonable phrasing variants are acceptable (e.g., campus, part of Lincoln Center complex)."
        ),
    )

    venue_url_leaf = evaluator.add_leaf(
        id="Venue_URL_Reference",
        desc="Provide a URL reference for the venue information",
        parent=venue_basic,
        critical=True,
    )
    venue_url_claim = f"The provided page(s) are official or authoritative pages about the venue '{venue_name}' at Lincoln Center."
    await evaluator.verify(
        claim=venue_url_claim,
        node=venue_url_leaf,
        sources=(data.venue.venue_urls if data.venue else []),
        additional_instruction=(
            "Verify that the URLs are about the venue itself (official venue page, Lincoln Center venue info page, etc.). "
            "If URLs do not describe the venue, mark as not supported."
        ),
    )

    addr_leaf = evaluator.add_leaf(
        id="Venue_Address",
        desc="Provide the complete physical address of the venue",
        parent=venue_ident,
        critical=True,
    )
    venue_address = data.venue.venue_address if data.venue and data.venue.venue_address else ""
    addr_claim = f"The complete physical address of '{venue_name}' is '{venue_address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=(data.venue.venue_urls if data.venue else []),
        additional_instruction=(
            "Check the venue pages for postal/street address. Minor formatting variations (e.g., abbreviations) are acceptable "
            "as long as the address clearly matches."
        ),
    )

    # Venue Capacity Compliance (parallel, critical)
    capacity_node = evaluator.add_parallel(
        id="Venue_Capacity_Compliance",
        desc="Verify the venue's seating capacity meets requirements",
        parent=venue_main,
        critical=True,
    )

    capacity_range_leaf = evaluator.add_leaf(
        id="Capacity_Range",
        desc="Confirm the venue has a seating capacity between 2,000 and 4,000 seats",
        parent=capacity_node,
        critical=True,
    )
    cap_sources = (data.venue.capacity_urls if data.venue else [])
    if not cap_sources:
        # Fallback to venue URLs if capacity URLs not provided
        cap_sources = (data.venue.venue_urls if data.venue else [])
    capacity_range_claim = f"The venue '{venue_name}' has a seating capacity between 2,000 and 4,000 seats."
    await evaluator.verify(
        claim=capacity_range_claim,
        node=capacity_range_leaf,
        sources=cap_sources,
        additional_instruction=(
            "Check the cited pages for a seating capacity (or typical capacity) value and confirm it lies within 2,000–4,000. "
            "Allow approximate wording (e.g., 'about', ranges) as long as it clearly falls within this interval."
        ),
    )

    capacity_url_leaf = evaluator.add_leaf(
        id="Capacity_URL_Reference",
        desc="Provide a URL reference confirming the venue capacity",
        parent=capacity_node,
        critical=True,
    )
    capacity_text = data.venue.venue_capacity if data.venue and data.venue.venue_capacity else ""
    capacity_url_claim = (
        f"The provided page(s) report the seating capacity of '{venue_name}' as '{capacity_text}' (or equivalent)."
    )
    await evaluator.verify(
        claim=capacity_url_claim,
        node=capacity_url_leaf,
        sources=cap_sources,
        additional_instruction=(
            "Verify that the page(s) explicitly mention a seating capacity value for the venue. "
            "Minor numeric formatting differences (commas, approximations) are acceptable if equivalent."
        ),
    )


async def build_ticketing_information(
    evaluator: Evaluator,
    parent_node,
    data: BalletPerformanceExtraction,
) -> None:
    ticketing_node = evaluator.add_parallel(
        id="Ticketing_Information",
        desc="Provide information about how to purchase tickets",
        parent=parent_node,
        critical=False,
    )

    purchase_leaf = evaluator.add_leaf(
        id="Purchase_Method",
        desc="Describe how tickets can be purchased (online, box office, phone, etc.)",
        parent=ticketing_node,
        critical=False,
    )
    purchase_method = data.ticketing.purchase_method if data.ticketing and data.ticketing.purchase_method else ""
    purchase_claim = f"Tickets can be purchased via: {purchase_method}."
    await evaluator.verify(
        claim=purchase_claim,
        node=purchase_leaf,
        sources=(data.ticketing.ticketing_url if data.ticketing and data.ticketing.ticketing_url else None),
        additional_instruction=(
            "Check the provided ticketing URL (if any) to confirm the stated purchase method(s). "
            "If the method is not described or the URL is missing/irrelevant, mark as not supported."
        ),
    )

    ticketing_url_leaf = evaluator.add_leaf(
        id="Ticketing_Website",
        desc="Provide the URL for purchasing tickets or getting ticket information",
        parent=ticketing_node,
        critical=False,
    )
    ticketing_url_claim = (
        "This URL is the official page to purchase tickets or obtain ticketing information for the identified performance or venue."
    )
    await evaluator.verify(
        claim=ticketing_url_claim,
        node=ticketing_url_leaf,
        sources=(data.ticketing.ticketing_url if data.ticketing and data.ticketing.ticketing_url else None),
        additional_instruction=(
            "Confirm that the URL is an official ticketing page (e.g., shows tickets, buy, purchase, or box office info) "
            "for the specific performance or the venue hosting it."
        ),
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
    Evaluate an answer for the ballet performance at Lincoln Center (Feb–Mar 2026) task.
    """
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ballet_performance(),
        template_class=BalletPerformanceExtraction,
        extraction_name="ballet_performance_extraction",
    )

    # Build rubric verification tree
    # Root corresponds to overall task (non-critical, sequential)
    ballet_task_node = evaluator.add_sequential(
        id="Ballet_Performance_Task",
        desc="Identify a professional ballet performance in NYC at Lincoln Center between Feb–Mar 2026, at a venue with 2,000–4,000 seat capacity",
        parent=root,
        critical=False,
    )

    # Subtrees according to rubric JSON
    await build_performance_identification_and_schedule(evaluator, ballet_task_node, extracted)
    await build_venue_requirements(evaluator, ballet_task_node, extracted)
    await build_ticketing_information(evaluator, ballet_task_node, extracted)

    # Add custom info for the date range target to the summary
    evaluator.add_custom_info(
        {"target_date_range": DATE_RANGE_TEXT, "location": "Lincoln Center, NYC", "capacity_requirement": "2,000–4,000"},
        info_type="task_requirements",
    )

    return evaluator.get_summary()