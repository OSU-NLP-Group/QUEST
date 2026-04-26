import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "clt_sapporo_feb2026_itinerary"
TASK_DESCRIPTION = """I'm planning a 4-day winter trip from Charlotte, North Carolina to Sapporo, Japan to experience skiing and the Sapporo Snow Festival in February 2026. Please help me create a detailed itinerary that meets the following requirements:

Travel Requirements:
- Outbound Journey: Depart from Charlotte Douglas International Airport (CLT) between February 2-4, 2026, and arrive in Sapporo (or at New Chitose Airport) before February 7, 2026. Include information about an airport lounge available at CLT where I can relax before my departure, specifying the lounge name, its concourse location, and provide a reference URL.

- Ski Resort Days (Days 2 & 3): Select two different ski resorts near Sapporo for February 7-9, 2026. Each resort must meet ALL of these criteria:
  - Accessible within 90 minutes from Sapporo city center
  - 1-day adult lift pass price not exceeding ¥9,500 for the 2025-26 season
  - On-site ski/snowboard equipment rental services available
  - Operating during early February 2026
  
  For each resort, provide: the resort name, travel time from Sapporo, 1-day lift pass price, confirmation of rental availability, and reference URLs for all claims.

- Snow Festival Day (Day 4): Plan to attend the Sapporo Snow Festival on February 9 or 10, 2026. Confirm the festival dates for 2026, identify the main festival venue (Odori Site location), and explain how to access it via Sapporo's subway system, including which station provides access. Provide reference URLs.

- Return Journey: Depart from Sapporo (or New Chitose Airport) on or after February 11, 2026 (the last day of the Snow Festival) and return to Charlotte Douglas International Airport (CLT). Provide the return routing and reference URLs.

For all flight segments, provide routing information and reference URLs confirming the flight connections and schedules.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FlightSegment(BaseModel):
    origin: Optional[str] = None          # e.g., "CLT"
    destination: Optional[str] = None     # e.g., "CTS"
    dep_datetime: Optional[str] = None    # keep as string (e.g., "Feb 3, 2026 10:30")
    arr_datetime: Optional[str] = None
    url: Optional[str] = None             # URL confirming this segment/schedule


class TravelPlan(BaseModel):
    depart_airport: Optional[str] = None  # e.g., "CLT"
    depart_date: Optional[str] = None     # keep as text (e.g., "Feb 3, 2026")
    arrive_airport: Optional[str] = None  # e.g., "CTS" or "Sapporo"
    arrive_date: Optional[str] = None
    routing_summary: Optional[str] = None # e.g., "CLT → LAX → NRT → CTS"
    segments: List[FlightSegment] = Field(default_factory=list)
    schedule_urls: List[str] = Field(default_factory=list)


class LoungeInfo(BaseModel):
    name: Optional[str] = None
    concourse: Optional[str] = None
    url: Optional[str] = None


class OutboundExtraction(BaseModel):
    travel: TravelPlan = TravelPlan()
    lounge: LoungeInfo = LoungeInfo()


class ReturnExtraction(BaseModel):
    travel: TravelPlan = TravelPlan()


class SkiResortInfo(BaseModel):
    name: Optional[str] = None
    scheduled_date: Optional[str] = None  # the date assigned for that ski day in the itinerary
    access_time: Optional[str] = None     # as text (e.g., "70 minutes", "1.5 hours")
    access_urls: List[str] = Field(default_factory=list)
    lift_price: Optional[str] = None      # as text (e.g., "¥7,500")
    price_urls: List[str] = Field(default_factory=list)
    rentals_statement: Optional[str] = None
    rentals_urls: List[str] = Field(default_factory=list)
    operating_statement: Optional[str] = None
    operating_urls: List[str] = Field(default_factory=list)


class SkiResortsExtraction(BaseModel):
    day2: SkiResortInfo = SkiResortInfo()
    day3: SkiResortInfo = SkiResortInfo()


class SnowFestivalExtraction(BaseModel):
    attendance_day: Optional[str] = None      # e.g., "Feb 9, 2026"
    festival_start: Optional[str] = None      # e.g., "Feb 4, 2026"
    festival_end: Optional[str] = None        # e.g., "Feb 11, 2026"
    dates_urls: List[str] = Field(default_factory=list)
    main_venue: Optional[str] = None          # e.g., "Odori Site (Odori Park)"
    venue_urls: List[str] = Field(default_factory=list)
    subway_access: Optional[str] = None       # explanation text
    odori_station_convergence: Optional[str] = None  # statement text mentioning all 3 lines converge
    subway_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outbound() -> str:
    return """
    Extract the outbound travel information and CLT lounge details mentioned in the answer.

    Return the following JSON:
    {
      "travel": {
        "depart_airport": string or null,        // departure airport code or name explicitly shown (e.g., CLT)
        "depart_date": string or null,           // departure date as written in the answer
        "arrive_airport": string or null,        // arrival airport or city in the Sapporo area (e.g., CTS or Sapporo)
        "arrive_date": string or null,           // arrival date as written
        "routing_summary": string or null,       // textual routing summary (e.g., "CLT → ORD → HND → CTS")
        "segments": [                            // each explicitly mentioned segment, if any
          {
            "origin": string or null,
            "destination": string or null,
            "dep_datetime": string or null,
            "arr_datetime": string or null,
            "url": string or null                // a URL that confirms this segment/schedule
          }
        ],
        "schedule_urls": [string, ...]           // any other URLs provided that confirm the outbound schedules/routing
      },
      "lounge": {
        "name": string or null,                  // CLT lounge name
        "concourse": string or null,             // CLT concourse/location (e.g., Concourse D/E)
        "url": string or null                    // a reference URL confirming lounge name/location at CLT
      }
    }

    Rules:
    - Only extract information explicitly present in the answer text.
    - For URLs, include actual links shown in the answer (including markdown links).
    - Do not invent any information. If missing, set the field to null or empty list accordingly.
    """


def prompt_extract_return() -> str:
    return """
    Extract the return travel information mentioned in the answer.

    Return the following JSON:
    {
      "travel": {
        "depart_airport": string or null,        // departure airport in Sapporo area (e.g., CTS)
        "depart_date": string or null,           // departure date as written in the answer
        "arrive_airport": string or null,        // final destination airport (should be CLT)
        "arrive_date": string or null,           // arrival date as written
        "routing_summary": string or null,       // textual routing summary
        "segments": [
          {
            "origin": string or null,
            "destination": string or null,
            "dep_datetime": string or null,
            "arr_datetime": string or null,
            "url": string or null
          }
        ],
        "schedule_urls": [string, ...]           // URLs provided that confirm the return schedules/routing
      }
    }

    Rules:
    - Only extract what appears explicitly in the answer.
    - Ensure URLs are real links from the answer.
    - Use nulls/empty lists where data is not present.
    """


def prompt_extract_ski_resorts() -> str:
    return """
    Extract the two ski resorts used for Day 2 and Day 3 in the itinerary with all required constraints and sources.

    Return the following JSON:
    {
      "day2": {
        "name": string or null,
        "scheduled_date": string or null,           // the specific date assigned to Day 2 skiing
        "access_time": string or null,              // travel time from Sapporo city center as stated
        "access_urls": [string, ...],               // URLs supporting the stated access time from Sapporo
        "lift_price": string or null,               // stated 1-day adult lift pass price (2025-26 season)
        "price_urls": [string, ...],                // URLs supporting the stated lift price (2025-26 season)
        "rentals_statement": string or null,        // statement that rentals are available
        "rentals_urls": [string, ...],              // URLs supporting rentals availability
        "operating_statement": string or null,      // statement that the resort operates in early Feb 2026
        "operating_urls": [string, ...]             // URLs supporting operating season/dates (covering early Feb 2026)
      },
      "day3": {
        "name": string or null,
        "scheduled_date": string or null,
        "access_time": string or null,
        "access_urls": [string, ...],
        "lift_price": string or null,
        "price_urls": [string, ...],
        "rentals_statement": string or null,
        "rentals_urls": [string, ...],
        "operating_statement": string or null,
        "operating_urls": [string, ...]
      }
    }

    Rules:
    - Extract fields exactly as mentioned in the answer.
    - Only return URLs actually present in the answer.
    - If a field is not present, return null or empty list as appropriate.
    """


def prompt_extract_festival() -> str:
    return """
    Extract the Sapporo Snow Festival planning and reference details from the answer.

    Return the following JSON:
    {
      "attendance_day": string or null,           // the specific date planned for attending (e.g., "Feb 9, 2026" or "Feb 10, 2026")
      "festival_start": string or null,           // confirmed 2026 festival start date as written
      "festival_end": string or null,             // confirmed 2026 festival end date as written
      "dates_urls": [string, ...],                // URLs confirming the 2026 festival dates
      "main_venue": string or null,               // text identifying Odori Site (Odori Park) as the main venue
      "venue_urls": [string, ...],                // URLs confirming Odori Site/Odori Park as a main venue
      "subway_access": string or null,            // explanation on subway access to the Odori Site
      "odori_station_convergence": string or null,// statement indicating Odori Station and that all 3 lines converge there
      "subway_urls": [string, ...]                // URLs supporting subway access and the convergence claim
    }

    Rules:
    - Only extract what is explicitly present in the answer.
    - For URLs, include actual links present in the answer.
    - Use nulls/empty arrays when appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _collect_urls_from_travel(tp: TravelPlan) -> List[str]:
    urls: List[str] = []
    urls.extend([u for u in (tp.schedule_urls or []) if _nonempty(u)])
    for seg in tp.segments or []:
        if _nonempty(seg.url):
            urls.append(seg.url)  # type: ignore
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_itinerary_structure(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="itinerary_structure",
        desc="Provides a day-by-day itinerary with at least four labeled days (Day 1–Day 4) and assigns the required activities to the specified days (ski on Days 2 & 3; Snow Festival on Day 4).",
        parent=parent,
        critical=True
    )

    # Leaf: Days 1–4 present
    leaf_days = evaluator.add_leaf(
        id="days_1_to_4_present",
        desc="Includes an explicit Day 1, Day 2, Day 3, and Day 4 plan in the itinerary.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The itinerary explicitly includes plans labeled Day 1, Day 2, Day 3, and Day 4.",
        node=leaf_days,
        additional_instruction="Accept minor formatting variants like 'Day-1' or headings; the key is that 4 distinct labeled days are clearly present."
    )

    # Leaf: Days 2 and 3 are ski days (with resorts specified)
    leaf_ski = evaluator.add_leaf(
        id="days_2_and_3_are_ski_days",
        desc="Designates Days 2 and 3 as ski-resort days (with resorts specified).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Day 2 and Day 3 are designated as ski-resort days and each day specifies a resort.",
        node=leaf_ski,
        additional_instruction="Look for explicit references such as 'Day 2: Ski at <Resort Name>' and 'Day 3: Ski at <Resort Name>'."
    )

    # Leaf: Day 4 is Snow Festival day
    leaf_festival = evaluator.add_leaf(
        id="day_4_is_snow_festival_day",
        desc="Designates Day 4 as the Sapporo Snow Festival day.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Day 4 is scheduled for attending the Sapporo Snow Festival.",
        node=leaf_festival,
        additional_instruction="Look for explicit wording that Day 4 is for the Sapporo Snow Festival."
    )


async def verify_outbound(evaluator: Evaluator, parent, outbound: OutboundExtraction) -> None:
    node = evaluator.add_parallel(
        id="outbound_travel",
        desc="Outbound journey meets CLT departure window, Sapporo/CTS arrival deadline, provides routing, and includes CLT lounge details with citations; flight routing must be feasible per cited schedules.",
        parent=parent,
        critical=True
    )

    # Simple structure checks based on answer text
    leaf_dep_airport = evaluator.add_leaf(
        id="outbound_departure_airport",
        desc="Outbound flight departs from Charlotte Douglas International Airport (CLT).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The outbound routing departs from Charlotte Douglas International Airport (CLT).",
        node=leaf_dep_airport,
        additional_instruction="Check the answer text to confirm the outbound departure airport is CLT."
    )

    leaf_dep_window = evaluator.add_leaf(
        id="outbound_departure_date_window",
        desc="Outbound departure occurs between Feb 2–4, 2026 (inclusive).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The outbound departure occurs between February 2 and February 4, 2026 (inclusive).",
        node=leaf_dep_window,
        additional_instruction="Consider the explicit departure date given in the itinerary; minor timezone descriptions are acceptable as long as the stated departure date is within Feb 2–4, 2026."
    )

    leaf_arrival_deadline = evaluator.add_leaf(
        id="outbound_arrival_deadline",
        desc="Arrival is in the Sapporo area (Sapporo city or New Chitose Airport/CTS) before Feb 7, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Arrival to the Sapporo area (Sapporo city or New Chitose Airport, CTS) occurs before February 7, 2026.",
        node=leaf_arrival_deadline,
        additional_instruction="Focus on the stated arrival date in the itinerary; 'before Feb 7' includes arriving on Feb 6 or earlier."
    )

    leaf_routing_provided = evaluator.add_leaf(
        id="outbound_routing_provided",
        desc="Provides outbound flight routing information (sequence of airports/cities; connections if any).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides outbound flight routing details listing the sequence of airports or cities and any connections.",
        node=leaf_routing_provided,
        additional_instruction="Routing may be shown as airport codes (e.g., CLT → HND → CTS) or city names with carriers; either is acceptable."
    )

    # Citations presence
    outbound_urls = _collect_urls_from_travel(outbound.travel)
    evaluator.add_custom_node(
        result=len(outbound_urls) > 0,
        id="outbound_routing_citations",
        desc="Provides reference URL(s) confirming the outbound flight connections and schedules.",
        parent=node,
        critical=True
    )

    # Feasibility supported by cited schedules
    leaf_feasible = evaluator.add_leaf(
        id="outbound_routing_feasible",
        desc="The cited schedules/connections make the outbound routing feasible (i.e., connections exist and are chronologically workable).",
        parent=node,
        critical=True
    )
    dep_date_txt = outbound.travel.depart_date or "early Feb 2026"
    summary = outbound.travel.routing_summary or "the proposed outbound routing"
    await evaluator.verify(
        claim=f"The cited schedules confirm that {summary} from CLT to Sapporo/CTS is feasible on or around {dep_date_txt} and arrives before February 7, 2026.",
        node=leaf_feasible,
        sources=outbound_urls,
        additional_instruction="Verify that flights on the provided pages exist and the connection sequence is logically workable near the stated dates; allow reasonable timezone/date-line effects."
    )

    # CLT Lounge
    lounge_node = evaluator.add_parallel(
        id="clt_lounge",
        desc="Includes a CLT airport lounge with required details and supporting citation.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(outbound.lounge.name),
        id="clt_lounge_name",
        desc="Provides the lounge name.",
        parent=lounge_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(outbound.lounge.concourse),
        id="clt_lounge_concourse_location",
        desc="Specifies the lounge concourse/location within CLT.",
        parent=lounge_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(outbound.lounge.url),
        id="clt_lounge_reference_url",
        desc="Provides a reference URL supporting the lounge name and location at CLT.",
        parent=lounge_node,
        critical=True
    )


async def _verify_resort_for_day(
    evaluator: Evaluator,
    parent,
    info: SkiResortInfo,
    day_label: str,         # "Day 2" or "Day 3"
    id_prefix: str          # "resort_2" or "resort_3"
) -> None:
    node = evaluator.add_parallel(
        id=f"{'resort_day_2' if id_prefix=='resort_2' else 'resort_day_3'}",
        desc=f"{day_label} ski resort satisfies all required constraints and includes required fields + citations.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(info.name),
        id=f"{id_prefix}_name",
        desc=f"Provides the ski resort name for {day_label}.",
        parent=node,
        critical=True
    )

    leaf_date_window = evaluator.add_leaf(
        id=f"{id_prefix}_ski_date_in_window",
        desc=f"Assigns {day_label} skiing to a date within Feb 7–9, 2026.",
        parent=node,
        critical=True
    )
    stated_date = info.scheduled_date or "(date not clearly stated)"
    await evaluator.verify(
        claim=f"{day_label} skiing is scheduled on {stated_date}, which is within Feb 7–9, 2026.",
        node=leaf_date_window,
        additional_instruction="Judge based on the itinerary's stated date for this day; accept common date formats (e.g., 'Feb 7, 2026', 'February 7, 2026')."
    )

    # Access time (presence of URL + support)
    evaluator.add_custom_node(
        result=len(info.access_urls) > 0,
        id=f"{id_prefix}_access_reference_url",
        desc=f"Provides reference URL(s) supporting the stated travel time/access from Sapporo.",
        parent=node,
        critical=True
    )
    leaf_access = evaluator.add_leaf(
        id=f"{id_prefix}_access_time_limit",
        desc="Travel time from Sapporo city center to the resort is stated as ≤ 90 minutes.",
        parent=node,
        critical=True
    )
    resort_name = info.name or "the resort"
    await evaluator.verify(
        claim=f"The travel time from Sapporo city center to {resort_name} is 90 minutes or less, as supported by the cited source(s).",
        node=leaf_access,
        sources=info.access_urls,
        additional_instruction="Look for travel time guidance from Sapporo to the resort (by bus/train/car). If a range is provided, the upper bound must be ≤ 90 minutes."
    )

    # Lift price (presence of URL + support)
    evaluator.add_custom_node(
        result=len(info.price_urls) > 0,
        id=f"{id_prefix}_lift_price_reference_url",
        desc="Provides reference URL(s) supporting the stated lift pass price for the 2025–26 season.",
        parent=node,
        critical=True
    )
    leaf_price = evaluator.add_leaf(
        id=f"{id_prefix}_lift_price_limit",
        desc="States the 2025–26 season 1-day adult lift pass price is ≤ ¥9,500.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the 2025–26 season, a 1-day adult lift pass at {resort_name} costs ¥9,500 or less.",
        node=leaf_price,
        sources=info.price_urls,
        additional_instruction="Prefer official resort pricing pages; if JPY tax is separated, use total day pass price. Seasonal pricing must be for 2025–26."
    )

    # Rentals (presence of URL + support)
    evaluator.add_custom_node(
        result=len(info.rentals_urls) > 0,
        id=f"{id_prefix}_rentals_reference_url",
        desc="Provides reference URL(s) supporting rental availability.",
        parent=node,
        critical=True
    )
    leaf_rentals = evaluator.add_leaf(
        id=f"{id_prefix}_rentals_available",
        desc="States on-site ski/snowboard equipment rentals are available.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{resort_name} offers on-site ski/snowboard equipment rentals.",
        node=leaf_rentals,
        sources=info.rentals_urls,
        additional_instruction="Look for rental pages or resort service pages indicating availability of ski/snowboard equipment rentals at the resort."
    )

    # Operating season (presence of URL + support)
    evaluator.add_custom_node(
        result=len(info.operating_urls) > 0,
        id=f"{id_prefix}_operating_reference_url",
        desc="Provides reference URL(s) supporting the operating season/dates.",
        parent=node,
        critical=True
    )
    leaf_operating = evaluator.add_leaf(
        id=f"{id_prefix}_operating_early_feb",
        desc="States the resort is operating in early February 2026 (operating season/dates include early Feb 2026).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{resort_name} is operating in early February 2026 (its operating season includes early Feb 2026).",
        node=leaf_operating,
        sources=info.operating_urls,
        additional_instruction="Check resort season calendars/notices for 2025–26; early February should be within the open season."
    )


async def verify_ski_resorts(evaluator: Evaluator, parent, ski: SkiResortsExtraction) -> None:
    node = evaluator.add_parallel(
        id="ski_resorts_days_2_and_3",
        desc="Selects two different ski resorts near Sapporo for Days 2 & 3 (dates within Feb 7–9, 2026). Each resort must meet access/price/rental/operating constraints and provide URLs for each claim.",
        parent=parent,
        critical=True
    )

    # Distinct resorts
    name2 = (ski.day2.name or "").strip().lower()
    name3 = (ski.day3.name or "").strip().lower()
    evaluator.add_custom_node(
        result=bool(name2) and bool(name3) and (name2 != name3),
        id="two_resorts_distinct",
        desc="The two selected ski resorts are different (not the same resort repeated).",
        parent=node,
        critical=True
    )

    # Day 2 resort
    await _verify_resort_for_day(
        evaluator=evaluator,
        parent=node,
        info=ski.day2,
        day_label="Day 2",
        id_prefix="resort_2"
    )

    # Day 3 resort
    await _verify_resort_for_day(
        evaluator=evaluator,
        parent=node,
        info=ski.day3,
        day_label="Day 3",
        id_prefix="resort_3"
    )


async def verify_snow_festival(evaluator: Evaluator, parent, fest: SnowFestivalExtraction) -> None:
    node = evaluator.add_parallel(
        id="snow_festival_day_4",
        desc="Plans Snow Festival attendance on Feb 9 or 10, 2026; confirms 2026 festival dates; identifies main venue (Odori Site in Odori Park) and required subway access details; provides citations; ensures trip coincides with the festival dates.",
        parent=parent,
        critical=True
    )

    # Attendance day choice (Feb 9 or 10, 2026)
    leaf_att_date = evaluator.add_leaf(
        id="attendance_date_choice",
        desc="Schedules Snow Festival attendance on Feb 9 or Feb 10, 2026.",
        parent=node,
        critical=True
    )
    att_txt = fest.attendance_day or "(attendance date not clearly stated)"
    await evaluator.verify(
        claim=f"The itinerary schedules Sapporo Snow Festival attendance on {att_txt}, which is either February 9 or February 10, 2026.",
        node=leaf_att_date,
        additional_instruction="Look for an explicit Day 4 assignment on Feb 9 or Feb 10, 2026."
    )

    # Dates references present
    evaluator.add_custom_node(
        result=len(fest.dates_urls) > 0,
        id="festival_dates_reference_url",
        desc="Provides reference URL(s) confirming the 2026 Snow Festival dates.",
        parent=node,
        critical=True
    )

    # Dates confirmed by URLs
    leaf_dates = evaluator.add_leaf(
        id="festival_dates_confirmed",
        desc="Provides the confirmed 2026 Sapporo Snow Festival dates (start and end) in the answer.",
        parent=node,
        critical=True
    )
    start_txt = fest.festival_start or "the published start date in 2026"
    end_txt = fest.festival_end or "the published end date in 2026"
    await evaluator.verify(
        claim=f"The 2026 Sapporo Snow Festival is scheduled from {start_txt} to {end_txt}.",
        node=leaf_dates,
        sources=fest.dates_urls,
        additional_instruction="Confirm the official date range for the 2026 festival on the cited page(s)."
    )

    # Attendance date falls within confirmed dates
    leaf_within = evaluator.add_leaf(
        id="attendance_within_festival_dates",
        desc="The scheduled attendance date (Feb 9 or 10, 2026) falls within the confirmed 2026 festival date range (trip coincides with the festival).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The scheduled attendance date {att_txt} falls within the confirmed 2026 Sapporo Snow Festival dates of {start_txt} to {end_txt}.",
        node=leaf_within,
        additional_instruction="Judge against the date values provided in the answer (not external knowledge)."
    )

    # Main venue identified
    leaf_venue_ident = evaluator.add_leaf(
        id="main_venue_identified",
        desc="Identifies the main festival venue as the Odori Site located in Odori Park.",
        parent=node,
        critical=True
    )
    venue_txt = fest.main_venue or "(venue text not clearly stated)"
    await evaluator.verify(
        claim=f"The answer identifies the main festival venue as the Odori Site in Odori Park (stated as: {venue_txt}).",
        node=leaf_venue_ident,
        additional_instruction="Look for clear mention of 'Odori Site' and 'Odori Park' as the main venue."
    )

    # Venue supported by URL(s)
    leaf_venue_ref = evaluator.add_leaf(
        id="venue_reference_url",
        desc="Provides reference URL(s) confirming Odori Site/Odori Park as a main festival venue.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Odori Site in Odori Park is one of the main venues of the Sapporo Snow Festival.",
        node=leaf_venue_ref,
        sources=fest.venue_urls,
        additional_instruction="Use official festival or city tourism pages if provided."
    )

    # Subway access explained
    leaf_subway_expl = evaluator.add_leaf(
        id="subway_access_explained",
        desc="Explains how to access the Odori Site via Sapporo's subway system (mode/route-level description).",
        parent=node,
        critical=True
    )
    subway_txt = fest.subway_access or "(subway access not clearly explained)"
    await evaluator.verify(
        claim=f"The answer explains how to reach the Odori Site by Sapporo's subway (stated as: {subway_txt}).",
        node=leaf_subway_expl,
        additional_instruction="Look for mention of lines/stations or instructions using Sapporo's subway to reach Odori Park."
    )

    # Odori Station + three-line convergence identified in the answer text
    leaf_converge_ident = evaluator.add_leaf(
        id="odori_station_convergence_identified",
        desc="Specifically identifies Odori Station as the subway access station and includes the claim that all three Sapporo subway lines converge there.",
        parent=node,
        critical=True
    )
    converge_txt = fest.odori_station_convergence or "(no explicit station convergence statement located)"
    await evaluator.verify(
        claim=f"The itinerary states that Odori Station is the access station and that all three Sapporo subway lines converge there (stated as: {converge_txt}).",
        node=leaf_converge_ident,
        additional_instruction="Accept equivalent wording; the key is explicitly naming 'Odori Station' and mentioning that all 3 subway lines meet/converge there."
    )

    # Subway claims supported by URLs (including convergence)
    leaf_subway_refs = evaluator.add_leaf(
        id="subway_access_reference_url",
        desc="Provides reference URL(s) supporting the subway access and station claims (including the convergence claim).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Odori Station provides access to Odori Park (Odori Site) and all three Sapporo subway lines converge at Odori Station (Namboku, Tozai, and Toho lines).",
        node=leaf_subway_refs,
        sources=fest.subway_urls,
        additional_instruction="Look for an official subway map or city transport page confirming line convergence at Odori Station."
    )


async def verify_return(evaluator: Evaluator, parent, ret: ReturnExtraction) -> None:
    node = evaluator.add_parallel(
        id="return_travel",
        desc="Return journey departs Sapporo area on/after Feb 11, 2026 and returns to CLT, with routing + citations; flight routing must be feasible per cited schedules.",
        parent=parent,
        critical=True
    )

    # Return departs on/after Feb 11, 2026
    leaf_dep_after = evaluator.add_leaf(
        id="return_departure_date_constraint",
        desc="Return departure from Sapporo area (Sapporo city or New Chitose Airport/CTS) is on or after Feb 11, 2026.",
        parent=node,
        critical=True
    )
    ret_dep_txt = ret.travel.depart_date or "(return departure date not clearly stated)"
    await evaluator.verify(
        claim=f"The return departure from the Sapporo area is on or after February 11, 2026 (stated as: {ret_dep_txt}).",
        node=leaf_dep_after,
        additional_instruction="Judge based on the stated departure date in the itinerary."
    )

    # Final destination is CLT
    leaf_dest_clt = evaluator.add_leaf(
        id="return_destination_clt",
        desc="Final destination is Charlotte Douglas International Airport (CLT).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The final destination of the return routing is Charlotte Douglas International Airport (CLT).",
        node=leaf_dest_clt,
        additional_instruction="Look for explicit mention of CLT as the final airport."
    )

    # Routing provided
    leaf_routing_provided = evaluator.add_leaf(
        id="return_routing_provided",
        desc="Provides return flight routing information (sequence of airports/cities; connections if any).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides return flight routing details listing the sequence of airports or cities and any connections.",
        node=leaf_routing_provided,
        additional_instruction="Routing can be summarized (e.g., CTS → HND → DFW → CLT)."
    )

    # Citations presence
    return_urls = _collect_urls_from_travel(ret.travel)
    evaluator.add_custom_node(
        result=len(return_urls) > 0,
        id="return_routing_citations",
        desc="Provides reference URL(s) confirming the return flight connections and schedules.",
        parent=node,
        critical=True
    )

    # Feasibility via cited schedules
    leaf_return_feasible = evaluator.add_leaf(
        id="return_routing_feasible",
        desc="The cited schedules/connections make the return routing feasible (i.e., connections exist and are chronologically workable).",
        parent=node,
        critical=True
    )
    ret_sum = ret.travel.routing_summary or "the proposed return routing"
    await evaluator.verify(
        claim=f"The cited schedules confirm that {ret_sum} from Sapporo/CTS to CLT on or after February 11, 2026 is feasible with workable connections.",
        node=leaf_return_feasible,
        sources=return_urls,
        additional_instruction="Check that flights exist and the listed sequence is available near the stated dates; allow reasonable timezone differences."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract all major info (can be parallelized)
    outbound_task = evaluator.extract(
        prompt=prompt_extract_outbound(),
        template_class=OutboundExtraction,
        extraction_name="outbound_extraction"
    )
    return_task = evaluator.extract(
        prompt=prompt_extract_return(),
        template_class=ReturnExtraction,
        extraction_name="return_extraction"
    )
    ski_task = evaluator.extract(
        prompt=prompt_extract_ski_resorts(),
        template_class=SkiResortsExtraction,
        extraction_name="ski_resorts_extraction"
    )
    fest_task = evaluator.extract(
        prompt=prompt_extract_festival(),
        template_class=SnowFestivalExtraction,
        extraction_name="festival_extraction"
    )

    outbound, ret, ski, fest = await asyncio.gather(outbound_task, return_task, ski_task, fest_task)

    # Build and run verification tree
    await verify_itinerary_structure(evaluator, root)
    await verify_outbound(evaluator, root, outbound)
    await verify_ski_resorts(evaluator, root, ski)
    await verify_snow_festival(evaluator, root, fest)
    await verify_return(evaluator, root, ret)

    # Return structured summary
    return evaluator.get_summary()