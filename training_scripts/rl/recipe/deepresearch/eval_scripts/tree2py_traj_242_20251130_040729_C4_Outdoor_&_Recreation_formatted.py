import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "camping_reservations_2026"
TASK_DESCRIPTION = (
    "A family is planning a camping road trip for summer 2026 and needs to understand the reservation systems for multiple parks to ensure they can book sites as soon as they become available. "
    "For each of the following four park systems, provide the complete reservation timing information:\n\n"
    "1. Yosemite National Park (reservable campgrounds such as Upper Pines, Lower Pines, North Pines): How far in advance are camping reservations released, "
    "and on what specific day of the month and at what time (including time zone) do new reservation dates become available?\n\n"
    "2. California State Parks: How far in advance can camping reservations be made, and at what specific time each day (including time zone) do new reservation dates open?\n\n"
    "3. Acadia National Park (campgrounds such as Blackwoods, Seawall, Schoodic Woods): How far in advance are camping reservations released, "
    "and on what specific day of the month and at what time (including time zone) do new reservation dates become available?\n\n"
    "4. Great Smoky Mountains National Park (backcountry camping): How far in advance can backcountry camping permits be reserved, what is the fee per person per night, "
    "and what is the maximum fee per person?\n\nFor each park system, provide supporting reference URLs from official sources."
)


# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------
class YosemiteSection(BaseModel):
    advance_window: Optional[str] = None
    release_rule: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CaliforniaSection(BaseModel):
    advance_window: Optional[str] = None
    daily_release_time: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AcadiaSection(BaseModel):
    advance_window_with_percent: Optional[str] = None
    monthly_release_time: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class GSMNPSection(BaseModel):
    advance_window: Optional[str] = None
    fee_per_person_per_night: Optional[str] = None
    max_fee_per_person: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ReservationExtraction(BaseModel):
    yosemite: Optional[YosemiteSection] = None
    california_state_parks: Optional[CaliforniaSection] = None
    acadia: Optional[AcadiaSection] = None
    gsmnp_backcountry: Optional[GSMNPSection] = None


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_reservation_info() -> str:
    return """
    Extract the reservation timing/fee information for each of the four park systems from the answer text. 
    For each park system, return ONLY information explicitly present in the answer.
    
    Return a JSON object with the following structure and fields:
    {
      "yosemite": {
        "advance_window": string | null,                  // e.g., "5 months in advance"
        "release_rule": string | null,                    // e.g., "on the 15th of each month at 7:00 AM PT (10:00 AM ET)"
        "urls": string[]                                  // all URLs the answer cites for Yosemite
      },
      "california_state_parks": {
        "advance_window": string | null,                  // e.g., "6 months in advance of arrival date"
        "daily_release_time": string | null,              // e.g., "8:00 AM PT each day"
        "urls": string[]                                  // all URLs the answer cites for California State Parks
      },
      "acadia": {
        "advance_window_with_percent": string | null,     // e.g., "90% released 6 months in advance"
        "monthly_release_time": string | null,            // e.g., "on the first of each month at 10:00 AM ET"
        "urls": string[]                                  // all URLs the answer cites for Acadia
      },
      "gsmnp_backcountry": {
        "advance_window": string | null,                  // e.g., "30 days in advance of the first night"
        "fee_per_person_per_night": string | null,        // e.g., "$8 per person per night"
        "max_fee_per_person": string | null,              // e.g., "$40 per person per permit"
        "urls": string[]                                  // all URLs the answer cites for GSMNP backcountry permits/fees
      }
    }
    
    Important:
    - Extract only what is explicitly stated in the answer text.
    - For URLs, extract every URL associated with each park system (including markdown links); include only valid URLs.
    - If any field is not present in the answer, return null for that field (or an empty array for urls).
    """


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def is_official_url_for_park(url: str, park_key: str) -> bool:
    try:
        parsed = urlparse(url.strip().lower())
    except Exception:
        return False

    host = parsed.netloc
    path = parsed.path or ""

    if not host:
        return False

    # Allow common official hosts
    is_recreation = host.endswith("recreation.gov")
    is_nps = host.endswith("nps.gov")
    is_parks_ca = host.endswith("parks.ca.gov")
    is_reserve_ca = host.endswith("reservecalifornia.com")

    # Yosemite: NPS Yosemite pages or Recreation.gov
    if park_key == "yosemite":
        return (is_nps and ("yose" in path)) or is_recreation

    # California State Parks: Official CA state parks domains
    if park_key == "california_state_parks":
        return is_parks_ca or is_reserve_ca

    # Acadia: NPS Acadia pages or Recreation.gov
    if park_key == "acadia":
        return (is_nps and ("acad" in path)) or is_recreation

    # GSMNP: NPS Great Smoky Mountains pages or official NPS permit subdomains
    if park_key == "gsmnp_backcountry":
        return (is_nps and ("grsm" in path or "smokiespermits" in host)) or ("smokiespermits.nps.gov" in host)

    return False


def filter_official_urls(urls: List[str], park_key: str) -> List[str]:
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if not u2.startswith("http://") and not u2.startswith("https://"):
            u2 = "http://" + u2
        if is_official_url_for_park(u2, park_key):
            cleaned.append(u2)
    return cleaned


# ------------------------------------------------------------------------------
# Verification builders per park
# ------------------------------------------------------------------------------
async def verify_yosemite(evaluator: Evaluator, parent_node, extracted: ReservationExtraction) -> None:
    park_node = evaluator.add_parallel(
        id="YosemiteNationalPark",
        desc="Yosemite National Park reservable campground reservation release timing is correctly provided.",
        parent=parent_node,
        critical=True
    )

    urls = _safe_urls((extracted.yosemite or YosemiteSection()).urls)
    official = filter_official_urls(urls, "yosemite")

    evaluator.add_custom_node(
        result=len(official) >= 1,
        id="YosemiteOfficialReferenceURLs",
        desc="Provides at least one supporting reference URL from an official source for Yosemite reservation timing.",
        parent=park_node,
        critical=True
    )

    # States 5 months in advance
    node_adv = evaluator.add_leaf(
        id="YosemiteAdvanceWindow",
        desc="States Yosemite reservable campground reservations are released 5 months in advance.",
        parent=park_node,
        critical=True
    )
    claim_adv = (
        "The answer explicitly states that Yosemite reservable campground reservations "
        "are released 5 months in advance (e.g., '5 months', 'five months' before arrival)."
    )
    await evaluator.verify(
        claim=claim_adv,
        node=node_adv,
        additional_instruction="Check only the answer text. Accept variants like 'five months' or '5 months'."
    )

    # States specific release day/time/time zone
    node_time = evaluator.add_leaf(
        id="YosemiteReleaseDayTimeAndZone",
        desc="States Yosemite release occurs on the 15th of each month at 7:00 AM Pacific Time (10:00 AM Eastern Time), including time zone(s).",
        parent=park_node,
        critical=True
    )
    claim_time = (
        "The answer explicitly states that Yosemite releases new reservation dates on the 15th of each month at 7:00 AM Pacific Time, "
        "and includes the time zone (e.g., PT/PDT/PST or 'Pacific Time'). Mentioning the equivalent 10:00 AM Eastern Time is optional."
    )
    await evaluator.verify(
        claim=claim_time,
        node=node_time,
        additional_instruction=(
            "Focus on the answer text. Accept 'PT', 'PDT', 'PST', or 'Pacific Time'. "
            "The statement must clearly include the 15th of each month and 7:00 AM in Pacific Time."
        )
    )


async def verify_california(evaluator: Evaluator, parent_node, extracted: ReservationExtraction) -> None:
    park_node = evaluator.add_parallel(
        id="CaliforniaStateParks",
        desc="California State Parks camping reservation opening timing is correctly provided.",
        parent=parent_node,
        critical=True
    )

    urls = _safe_urls((extracted.california_state_parks or CaliforniaSection()).urls)
    official = filter_official_urls(urls, "california_state_parks")

    evaluator.add_custom_node(
        result=len(official) >= 1,
        id="CaliforniaOfficialReferenceURLs",
        desc="Provides at least one supporting reference URL from an official source for California State Parks reservation timing.",
        parent=park_node,
        critical=True
    )

    # States 6 months in advance
    node_adv = evaluator.add_leaf(
        id="CaliforniaAdvanceWindow",
        desc="States California State Parks camping reservations can be made 6 months in advance of the arrival date.",
        parent=park_node,
        critical=True
    )
    claim_adv = (
        "The answer explicitly states that California State Parks camping reservations can be made 6 months in advance of the arrival date."
    )
    await evaluator.verify(
        claim=claim_adv,
        node=node_adv,
        additional_instruction="Check only the answer text. Accept variants like 'six months', '6 months'."
    )

    # States daily release time at 8:00 AM PT
    node_time = evaluator.add_leaf(
        id="CaliforniaDailyReleaseTimeAndZone",
        desc="States new reservation dates open at 8:00 AM each day in Pacific Time (PST/PDT or PT), including time zone.",
        parent=park_node,
        critical=True
    )
    claim_time = (
        "The answer explicitly states that new reservation dates open at 8:00 AM each day in Pacific Time (PT/PST/PDT), and includes the time zone."
    )
    await evaluator.verify(
        claim=claim_time,
        node=node_time,
        additional_instruction="Focus on the answer text. Accept 'PT', 'PDT', 'PST', or 'Pacific Time'."
    )


async def verify_acadia(evaluator: Evaluator, parent_node, extracted: ReservationExtraction) -> None:
    park_node = evaluator.add_parallel(
        id="AcadiaNationalPark",
        desc="Acadia National Park campground reservation release timing is correctly provided.",
        parent=parent_node,
        critical=True
    )

    urls = _safe_urls((extracted.acadia or AcadiaSection()).urls)
    official = filter_official_urls(urls, "acadia")

    evaluator.add_custom_node(
        result=len(official) >= 1,
        id="AcadiaOfficialReferenceURLs",
        desc="Provides at least one supporting reference URL from an official source for Acadia reservation timing.",
        parent=park_node,
        critical=True
    )

    # States 90% released 6 months in advance
    node_adv = evaluator.add_leaf(
        id="AcadiaAdvanceWindowWithPercent",
        desc="States that 90% of Acadia campsites are released 6 months in advance.",
        parent=park_node,
        critical=True
    )
    claim_adv = (
        "The answer explicitly states that 90% of Acadia campsites are released 6 months in advance."
    )
    await evaluator.verify(
        claim=claim_adv,
        node=node_adv,
        additional_instruction="Check only the answer text. The statement must include both '90%' and '6 months'."
    )

    # States monthly release day/time/time zone
    node_time = evaluator.add_leaf(
        id="AcadiaReleaseDayTimeAndZone",
        desc="States Acadia release occurs on the first of each month at 10:00 AM Eastern Time, including time zone.",
        parent=park_node,
        critical=True
    )
    claim_time = (
        "The answer explicitly states that Acadia releases new reservation dates on the first of each month at 10:00 AM Eastern Time, "
        "and includes the time zone (e.g., ET/EDT/EST or 'Eastern Time')."
    )
    await evaluator.verify(
        claim=claim_time,
        node=node_time,
        additional_instruction="Focus on the answer text. Accept 'ET', 'EDT', 'EST', or 'Eastern Time'."
    )


async def verify_gsmnp(evaluator: Evaluator, parent_node, extracted: ReservationExtraction) -> None:
    park_node = evaluator.add_parallel(
        id="GreatSmokyMountainsBackcountry",
        desc="Great Smoky Mountains National Park backcountry permit reservation timing and fees are correctly provided.",
        parent=parent_node,
        critical=True
    )

    urls = _safe_urls((extracted.gsmnp_backcountry or GSMNPSection()).urls)
    official = filter_official_urls(urls, "gsmnp_backcountry")

    evaluator.add_custom_node(
        result=len(official) >= 1,
        id="GSMNPOfficialReferenceURLs",
        desc="Provides at least one supporting reference URL from an official source for GSMNP backcountry permits/fees.",
        parent=park_node,
        critical=True
    )

    # 30 days in advance
    node_adv = evaluator.add_leaf(
        id="GSMNPAdvanceWindow",
        desc="States backcountry permits can be reserved up to 30 days in advance of the first night of the trip.",
        parent=park_node,
        critical=True
    )
    claim_adv = (
        "The answer explicitly states that Great Smoky Mountains National Park backcountry permits can be reserved up to 30 days in advance of the first night of the trip."
    )
    await evaluator.verify(
        claim=claim_adv,
        node=node_adv,
        additional_instruction="Check only the answer text. Accept '30 days' or 'thirty days'."
    )

    # $8 per person per night
    node_fee = evaluator.add_leaf(
        id="GSMNPFeePerPersonPerNight",
        desc="States the fee is $8 per person per night.",
        parent=park_node,
        critical=True
    )
    claim_fee = "The answer explicitly states that the backcountry permit fee is $8 per person per night."
    await evaluator.verify(
        claim=claim_fee,
        node=node_fee,
        additional_instruction="Check only the answer text. Accept '$8', '8 dollars', or 'eight dollars'."
    )

    # Maximum $40 per person per permit
    node_max = evaluator.add_leaf(
        id="GSMNPMaxFeePerPerson",
        desc="States the maximum fee is $40 per person per permit.",
        parent=park_node,
        critical=True
    )
    claim_max = "The answer explicitly states that the maximum fee is $40 per person per permit."
    await evaluator.verify(
        claim=claim_max,
        node=node_max,
        additional_instruction="Check only the answer text. Accept '$40', '40 dollars', or 'forty dollars'."
    )


# ------------------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------------------
async def evaluate_answer(
    client: Any,
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_reservation_info(),
        template_class=ReservationExtraction,
        extraction_name="reservation_extraction"
    )

    # Record ground-truth-like expectations for transparency
    evaluator.add_ground_truth({
        "expected": {
            "yosemite": {
                "advance_window": "5 months in advance",
                "release": "15th of each month at 7:00 AM Pacific Time (10:00 AM Eastern Time)"
            },
            "california_state_parks": {
                "advance_window": "6 months in advance of arrival date",
                "release": "8:00 AM Pacific Time each day"
            },
            "acadia": {
                "advance_window_with_percent": "90% of campsites released 6 months in advance",
                "release": "first of each month at 10:00 AM Eastern Time"
            },
            "gsmnp_backcountry": {
                "advance_window": "30 days in advance of the first night",
                "fee_per_person_per_night": "$8 per person per night",
                "max_fee_per_person": "$40 per person per permit"
            }
        }
    })

    # Add an overall critical container to mirror rubric root
    compare_node = evaluator.add_parallel(
        id="ReservationInformationComparison",
        desc="Provides reservation timing/fee details for all four specified park systems, each with official supporting references.",
        parent=root,
        critical=True
    )

    # Run park verifications
    await verify_yosemite(evaluator, compare_node, extraction)
    await verify_california(evaluator, compare_node, extraction)
    await verify_acadia(evaluator, compare_node, extraction)
    await verify_gsmnp(evaluator, compare_node, extraction)

    # For debugging/analytics, store official URL counts
    yose_urls = _safe_urls((extraction.yosemite or YosemiteSection()).urls)
    ca_urls = _safe_urls((extraction.california_state_parks or CaliforniaSection()).urls)
    acad_urls = _safe_urls((extraction.acadia or AcadiaSection()).urls)
    gsmnp_urls = _safe_urls((extraction.gsmnp_backcountry or GSMNPSection()).urls)

    evaluator.add_custom_info(
        info={
            "yosemite_official_urls": filter_official_urls(yose_urls, "yosemite"),
            "california_official_urls": filter_official_urls(ca_urls, "california_state_parks"),
            "acadia_official_urls": filter_official_urls(acad_urls, "acadia"),
            "gsmnp_official_urls": filter_official_urls(gsmnp_urls, "gsmnp_backcountry"),
        },
        info_type="official_urls",
        info_name="official_urls_by_park"
    )

    return evaluator.get_summary()