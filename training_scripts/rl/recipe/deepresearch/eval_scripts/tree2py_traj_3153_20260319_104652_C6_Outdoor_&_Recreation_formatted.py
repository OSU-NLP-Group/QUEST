import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_winter_camping_mlk_2026"
TASK_DESCRIPTION = """
You are planning a 3-night winter camping trip to a California State Park over the Martin Luther King Jr. Day 2026 long weekend. You want to arrive on Saturday, January 17, 2026, and depart on Tuesday, January 20, 2026. Please provide the following information: (1) Select any California State Park that: (a) accepts camping reservations through the ReserveCalifornia system, (b) is open for camping in January, and (c) has established campsites (not dispersed camping only). (2) Calculate the exact date and time (including timezone) when reservations for your arrival date (January 17, 2026) will first become available on the ReserveCalifornia website. (3) Calculate the minimum total cost for this camping reservation, including: (a) the mandatory non-refundable reservation fee charged by California State Parks, (b) the estimated per-night camping fee for 3 nights (use the typical range if exact pricing varies by site), and (c) any day-use or vehicle entry fees if applicable. (4) Explain whether National Parks offer free entry on MLK Day 2026, and if not, state the current price of the America the Beautiful Annual Pass for U.S. residents. (5) Identify one major California airport near your selected state park that supports TSA PreCheck Touchless ID technology (available at select airports by Spring 2026). For each answer component, provide supporting reference URLs from your research.
"""

ARRIVAL_DATE_ISO = "2026-01-17"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkSelectionExtraction(BaseModel):
    park_name: Optional[str] = None
    campground_name: Optional[str] = None
    park_evidence_urls: List[str] = Field(default_factory=list)


class ReservationReleaseExtraction(BaseModel):
    release_date_iso: Optional[str] = None
    release_time_text: Optional[str] = None
    release_timezone_text: Optional[str] = None
    timing_urls: List[str] = Field(default_factory=list)


class CostExtraction(BaseModel):
    reservation_fee_amount: Optional[str] = None
    nightly_fee_exact: Optional[str] = None
    nightly_fee_low: Optional[str] = None
    nightly_fee_high: Optional[str] = None
    number_of_nights: Optional[str] = None
    day_use_or_vehicle_fees_text: Optional[str] = None
    day_use_total_amount: Optional[str] = None
    total_cost_text: Optional[str] = None
    total_cost_min: Optional[str] = None
    total_cost_max: Optional[str] = None
    cost_urls: List[str] = Field(default_factory=list)


class NPSPolicyExtraction(BaseModel):
    mlk_day_free_entry: Optional[str] = None  # expected values like "yes", "no", "free", "not free"
    annual_pass_price: Optional[str] = None
    nps_urls: List[str] = Field(default_factory=list)


class AirportExtraction(BaseModel):
    airport_name: Optional[str] = None
    airport_state: Optional[str] = None
    proximity_urls: List[str] = Field(default_factory=list)
    touchless_id_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_park_selection() -> str:
    return """
    From the answer, extract the selected California State Park and campground information, along with the URLs used as evidence for eligibility.
    Requirements the park/campground must satisfy:
    - It is part of the California State Parks system.
    - It handles camping reservations through ReserveCalifornia.
    - It is open for camping in January (including January 2026).
    - It offers established (developed) campsites (not dispersed camping only).

    Return:
    - park_name: Name of the California State Park explicitly stated in the answer.
    - campground_name: Name of the specific campground (if mentioned); else null.
    - park_evidence_urls: A list of all URLs cited in the answer that substantiate any of the above bullets (e.g., the official parks.ca.gov page, the ReserveCalifornia campground listing page, seasonality info, campground description showing developed sites, etc.). Extract only URLs that appear in the answer. Do not invent any URLs.
    """


def prompt_extract_reservation_release() -> str:
    return f"""
    The arrival date is {ARRIVAL_DATE_ISO} (YYYY-MM-DD). From the answer, extract the computed reservation release info and the supporting URLs about ReserveCalifornia booking rules.
    Return:
    - release_date_iso: The calendar date when reservations first become available for that arrival date, in YYYY-MM-DD. If the answer gives a natural-language date, convert it to ISO.
    - release_time_text: The stated release time as shown in the answer (e.g., "8:00 AM").
    - release_timezone_text: The stated Pacific timezone string or abbreviation as shown in the answer (e.g., "PST", "PDT", "Pacific Time").
    - timing_urls: URLs cited in the answer that confirm the booking window (e.g., "6 months in advance") AND that the release is at 8:00 AM Pacific Time. Extract only URLs actually present in the answer.
    """


def prompt_extract_costs() -> str:
    return """
    From the answer, extract the minimum total cost components and references for a 3-night camping reservation.
    Return:
    - reservation_fee_amount: The mandatory non-refundable California State Parks reservation fee as stated (e.g., "$8").
    - nightly_fee_exact: If the answer gives a single nightly camping price, return it (e.g., "$40"); else null.
    - nightly_fee_low: If the answer gives a nightly price range, the low end (e.g., "$35"); else null.
    - nightly_fee_high: If the answer gives a nightly price range, the high end (e.g., "$45"); else null.
    - number_of_nights: The number of nights used in the calculation (as a string, e.g., "3").
    - day_use_or_vehicle_fees_text: The answer's statement about day-use or vehicle entry fees (e.g., "None" or "$10 per vehicle per day"); return null if not addressed.
    - day_use_total_amount: If the answer computed a trip-total for day-use/vehicle fees, return that amount (e.g., "$30"); else null.
    - total_cost_text: The final total cost as stated in the answer; if a range is stated, keep the range text (e.g., "$113–$143"); if a single number, keep it (e.g., "$128").
    - total_cost_min: If a range is stated, return the lower bound as text (e.g., "$113"); else null.
    - total_cost_max: If a range is stated, return the upper bound as text (e.g., "$143"); else null.
    - cost_urls: All URLs cited in the answer that substantiate: the reservation fee amount, the nightly camping fee (range or park-specific), and any day-use/vehicle entry fees. Extract only URLs actually provided in the answer.
    """


def prompt_extract_nps_policy() -> str:
    return """
    From the answer, extract the National Park Service (NPS) MLK Day 2026 fee-free statement and the America the Beautiful Annual Pass price, with citations.
    Return:
    - mlk_day_free_entry: The answer's claim about MLK Day 2026 park entry (e.g., "free", "not free", "yes", "no").
    - annual_pass_price: The stated price of the America the Beautiful Annual Pass for U.S. residents (e.g., "$80").
    - nps_urls: URLs cited in the answer that substantiate the MLK Day policy and the annual pass price. Extract only URLs actually present in the answer.
    """


def prompt_extract_airport(park_name_placeholder: str) -> str:
    return f"""
    From the answer, extract one major California airport near the selected state park (e.g., near "{park_name_placeholder}") and citations for both proximity and TSA PreCheck Touchless ID support.
    Return:
    - airport_name: The airport's name/identifier as stated in the answer (e.g., "San Francisco International Airport (SFO)").
    - airport_state: The state for the airport as stated (e.g., "California", "CA"), if mentioned; else null.
    - proximity_urls: URLs cited in the answer to substantiate that this airport is near the selected park (e.g., an official page, or a mapping/directions link showing the relationship).
    - touchless_id_urls: URLs cited in the answer to substantiate that this airport supports TSA PreCheck Touchless ID / Digital ID (e.g., TSA press release or airport page). Extract only URLs actually present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    unique = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s and s not in unique:
                unique.append(s)
    return unique


def parse_money_first(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.findall(r'(\d+(?:\.\d{1,2})?)', s.replace(',', ''))
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def parse_int_first(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.findall(r'(\d+)', s)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def compute_six_months_prior(arrival_iso: str) -> Optional[str]:
    """
    Compute 6 months prior by simple calendar arithmetic.
    For the given task date 2026-01-17, this is straightforward => 2025-07-17.
    """
    try:
        y, m, d = [int(p) for p in arrival_iso.split('-')]
        m -= 6
        if m <= 0:
            y -= 1
            m += 12
        # Minimal day-of-month safety for typical scenario; here 17 is safe for all months we hit.
        return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        return None


def time_is_eight_am(s: Optional[str]) -> bool:
    if not s:
        return False
    st = s.strip().lower()
    patterns = ["8:00 am", "8am", "8 am", "08:00 am", "8 a.m", "8 a. m", "8:00 a.m"]
    return any(p in st for p in patterns)


def tz_is_pdt(s: Optional[str]) -> bool:
    if not s:
        return False
    st = s.strip().lower()
    return ("pdt" in st) or ("pacific daylight" in st)


def expected_pacific_tz_for_date(iso_date: str) -> str:
    """
    For July in the U.S. Pacific region, expect PDT.
    For January, expect PST. Only needed for the release date (which will be in July 2025).
    """
    try:
        _, month, _ = iso_date.split('-')
        m = int(month)
        # Rough rule: DST (PDT) is typically from Mar to early Nov; treat July as PDT.
        if 3 <= m <= 11:
            return "PDT"
        else:
            return "PST"
    except Exception:
        return "PDT"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_park_selection(evaluator: Evaluator, root, park: ParkSelectionExtraction) -> None:
    section = evaluator.add_parallel(
        id="1_park_selection",
        desc="Select a California State Park campground that meets all stated eligibility conditions and provide supporting URLs.",
        parent=root,
        critical=True
    )
    park_urls = clean_urls(park.park_evidence_urls)
    park_name_display = park.park_name or "the selected park"

    # park_is_ca_state_park
    node_ca = evaluator.add_leaf(
        id="park_is_ca_state_park",
        desc="Selected location is part of the California State Parks system.",
        parent=section,
        critical=True
    )
    claim_ca = f"{park_name_display} is a California State Park (i.e., part of the California Department of Parks and Recreation)."
    await evaluator.verify(
        claim=claim_ca,
        node=node_ca,
        sources=park_urls,
        additional_instruction="Confirm the park is part of the California State Parks system (parks.ca.gov or an authoritative ReserveCalifornia listing for a CA State Park)."
    )

    # uses_reservecalifornia
    node_rc = evaluator.add_leaf(
        id="uses_reservecalifornia",
        desc="Camping reservations for the selected park/campground are handled through ReserveCalifornia.",
        parent=section,
        critical=True
    )
    claim_rc = f"Camping reservations for {park_name_display} are handled via ReserveCalifornia."
    await evaluator.verify(
        claim=claim_rc,
        node=node_rc,
        sources=park_urls,
        additional_instruction="Look for ReserveCalifornia listing pages or explicit statements that reservations are made through ReserveCalifornia."
    )

    # open_for_camping_in_january
    node_jan = evaluator.add_leaf(
        id="open_for_camping_in_january",
        desc="Selected park/campground is open for camping during January 2026.",
        parent=section,
        critical=True
    )
    claim_jan = f"The campground at {park_name_display} is open for camping in January (including January 2026)."
    await evaluator.verify(
        claim=claim_jan,
        node=node_jan,
        sources=park_urls,
        additional_instruction="Evidence could be 'Year-round' operation, seasonal dates including January, or a ReserveCalifornia calendar showing availability in January."
    )

    # has_established_campsites_not_dispersed_only
    node_dev = evaluator.add_leaf(
        id="has_established_campsites_not_dispersed_only",
        desc="Selected option has established campsites (not dispersed camping only).",
        parent=section,
        critical=True
    )
    claim_dev = f"{park_name_display} offers established (developed) campsites, not dispersed camping only."
    await evaluator.verify(
        claim=claim_dev,
        node=node_dev,
        sources=park_urls,
        additional_instruction="Look for campground pages showing developed campsites, site amenities, or campground maps."
    )

    # park_component_citations
    node_cites = evaluator.add_custom_node(
        result=len(park_urls) > 0,
        id="park_component_citations",
        desc="Provides supporting reference URL(s) substantiating the above park eligibility claims.",
        parent=section,
        critical=True
    )


async def verify_reservation_release(evaluator: Evaluator, root, release: ReservationReleaseExtraction) -> None:
    section = evaluator.add_parallel(
        id="2_reservation_release_datetime",
        desc="Compute when reservations for arrival date 2026-01-17 first become available, including timezone, with citations.",
        parent=root,
        critical=True
    )

    expected_release_iso = compute_six_months_prior(ARRIVAL_DATE_ISO)  # Expected 2025-07-17
    # computes_correct_release_date
    node_date = evaluator.add_custom_node(
        result=(release.release_date_iso is not None and expected_release_iso is not None and release.release_date_iso.strip() == expected_release_iso),
        id="computes_correct_release_date",
        desc="Correctly calculates the reservation release calendar date from the arrival date using the stated rule (6 months / 180 days as specified in constraints).",
        parent=section,
        critical=True
    )

    # includes_correct_release_time_and_timezone
    # Check 8:00 AM and correct Pacific designation for July (PDT)
    expected_tz = expected_pacific_tz_for_date(expected_release_iso) if expected_release_iso else "PDT"
    time_ok = time_is_eight_am(release.release_time_text)
    tz_ok = tz_is_pdt((release.release_timezone_text or "") + " " + (release.release_time_text or "")) if expected_tz == "PDT" else True
    node_time_tz = evaluator.add_custom_node(
        result=(time_ok and tz_ok),
        id="includes_correct_release_time_and_timezone",
        desc="States the release time as 8:00 AM and specifies the correct Pacific timezone designation for that release date (PST vs PDT).",
        parent=section,
        critical=True
    )

    # reservation_timing_citations
    node_cite = evaluator.add_leaf(
        id="reservation_timing_citations",
        desc="Provides supporting reference URL(s) confirming the booking window and the 8:00 AM Pacific release time.",
        parent=section,
        critical=True
    )
    claim_cite = "California State Parks/ReserveCalifornia opens campsite reservations six months in advance, and the daily release time is 8:00 AM Pacific Time."
    # Important: if no URLs are provided, force failure via instruction
    timing_urls = clean_urls(release.timing_urls)
    await evaluator.verify(
        claim=claim_cite,
        node=node_cite,
        sources=timing_urls if timing_urls else None,
        additional_instruction="If no URL is provided, mark this as NOT SUPPORTED. Otherwise, verify the booking window and the 8:00 AM Pacific release time."
    )

    # Add expected info to summary for transparency
    evaluator.add_custom_info(
        info={
            "arrival_date_iso": ARRIVAL_DATE_ISO,
            "expected_release_date_iso": expected_release_iso,
            "expected_pacific_tz_for_release": expected_tz
        },
        info_type="reservation_release_ground_truth",
        info_name="expected_release_info"
    )


async def verify_costs(evaluator: Evaluator, root, cost: CostExtraction) -> None:
    section = evaluator.add_parallel(
        id="3_min_total_cost",
        desc="Compute the minimum total cost for the 3-night reservation including required fee components, with citations.",
        parent=root,
        critical=True
    )

    # includes_mandatory_reservation_fee ($8)
    res_fee_val = parse_money_first(cost.reservation_fee_amount)
    node_resfee = evaluator.add_custom_node(
        result=(res_fee_val is not None and abs(res_fee_val - 8.0) < 0.01),
        id="includes_mandatory_reservation_fee",
        desc="Includes the mandatory non-refundable CA State Parks reservation fee ($8) in the total.",
        parent=section,
        critical=True
    )

    # includes_3_nights_camping_fees
    nights_val = parse_int_first(cost.number_of_nights)
    has_nightly = bool((cost.nightly_fee_exact and parse_money_first(cost.nightly_fee_exact) is not None) or
                       (cost.nightly_fee_low and parse_money_first(cost.nightly_fee_low) is not None) or
                       (cost.nightly_fee_high and parse_money_first(cost.nightly_fee_high) is not None))
    node_3n = evaluator.add_custom_node(
        result=(nights_val == 3 and has_nightly),
        id="includes_3_nights_camping_fees",
        desc="Includes camping fees for exactly 3 nights (Jan 17–20, 2026) using the stated typical nightly range ($35–$45) or park-specific nightly price if provided.",
        parent=section,
        critical=True
    )

    # addresses_day_use_or_vehicle_fees_if_applicable
    addressed_day_use = (cost.day_use_or_vehicle_fees_text is not None) and (str(cost.day_use_or_vehicle_fees_text).strip() != "")
    node_dayuse = evaluator.add_custom_node(
        result=addressed_day_use,
        id="addresses_day_use_or_vehicle_fees_if_applicable",
        desc="States whether any day-use/vehicle entry fees apply (if applicable) and includes them in the total if applicable.",
        parent=section,
        critical=True
    )

    # total_cost_arithmetic_consistency (LLM reasoning using extracted components)
    node_arith = evaluator.add_leaf(
        id="total_cost_arithmetic_consistency",
        desc="Total cost (or range) is arithmetically consistent with the stated components (reservation fee + 3 nights + any applicable entry/day-use fees).",
        parent=section,
        critical=True
    )
    nightly_desc = cost.nightly_fee_exact or f"range {cost.nightly_fee_low}–{cost.nightly_fee_high}"
    day_use_desc = cost.day_use_or_vehicle_fees_text or "not mentioned"
    claim_arith = (
        f"Given a reservation fee {cost.reservation_fee_amount}, nightly camping fee {nightly_desc} for {cost.number_of_nights} nights, "
        f"and day-use/vehicle fees described as '{day_use_desc}' (trip total if stated: {cost.day_use_total_amount}), "
        f"the stated total '{cost.total_cost_text}' (min {cost.total_cost_min}, max {cost.total_cost_max}) is arithmetically consistent."
    )
    await evaluator.verify(
        claim=claim_arith,
        node=node_arith,
        additional_instruction="Judge whether the final total (or range) correctly reflects reservation fee + (nightly × nights) + any day-use/vehicle fees (if included). Allow reasonable rounding."
    )

    # cost_component_citations
    node_cost_cites = evaluator.add_leaf(
        id="cost_component_citations",
        desc="Provides supporting reference URL(s) substantiating the reservation fee, the camping-fee basis (range or park-specific pricing), and any included entry/day-use fees.",
        parent=section,
        critical=True
    )
    claim_cost_cites = (
        "The provided sources substantiate that the California State Parks reservation fee is $8 per reservation and substantiate the stated nightly campsite fee (range or specific) and any day-use/vehicle fees included."
    )
    cost_urls = clean_urls(cost.cost_urls)
    await evaluator.verify(
        claim=claim_cost_cites,
        node=node_cost_cites,
        sources=cost_urls if cost_urls else None,
        additional_instruction="If no URL is provided, mark this as NOT SUPPORTED. Otherwise, verify the cited pages support the $8 reservation fee and the nightly/day-use fee basis used."
    )


async def verify_nps_policy_and_pass(evaluator: Evaluator, root, nps: NPSPolicyExtraction) -> None:
    section = evaluator.add_parallel(
        id="4_national_parks_policy_and_pass",
        desc="Answer the MLK Day 2026 National Parks fee-free question and provide the America the Beautiful Annual Pass price with citations.",
        parent=root,
        critical=True
    )

    # mlk_day_not_fee_free (per rubric requirement)
    mlk_text = (nps.mlk_day_free_entry or "").strip().lower()
    states_not_free = any(x in mlk_text for x in ["not free", "no", "not fee-free", "fees apply", "paid"])
    node_mlk = evaluator.add_custom_node(
        result=states_not_free,
        id="mlk_day_not_fee_free",
        desc="States that National Parks do not offer free entry on MLK Day 2026 (per provided constraint).",
        parent=section,
        critical=True
    )

    # annual_pass_price ($80)
    pass_price_val = parse_money_first(nps.annual_pass_price)
    node_pass = evaluator.add_custom_node(
        result=(pass_price_val is not None and abs(pass_price_val - 80.0) < 0.01),
        id="annual_pass_price",
        desc="States the America the Beautiful Annual Pass price as $80 for U.S. residents (per provided constraint).",
        parent=section,
        critical=True
    )

    # nps_policy_citations (verify sources support the two claims)
    node_nps_cites = evaluator.add_leaf(
        id="nps_policy_citations",
        desc="Provides supporting reference URL(s) for the MLK Day policy and the annual pass pricing.",
        parent=section,
        critical=True
    )
    nps_urls = clean_urls(nps.nps_urls)
    claim_nps_cites = "The provided sources support that MLK Day 2026 is NOT fee-free at National Parks and that the America the Beautiful Annual Pass price is $80."
    await evaluator.verify(
        claim=claim_nps_cites,
        node=node_nps_cites,
        sources=nps_urls if nps_urls else None,
        additional_instruction="If no URL is provided, mark this as NOT SUPPORTED. Otherwise, check that the pages substantiate the MLK Day policy and the $80 pass price."
    )


async def verify_airport(evaluator: Evaluator, root, airport: AirportExtraction, park: ParkSelectionExtraction) -> None:
    section = evaluator.add_parallel(
        id="5_airport_with_touchless_id",
        desc="Identify one major California airport near the selected state park that supports TSA PreCheck Touchless ID and provide citations.",
        parent=root,
        critical=True
    )
    airport_name_disp = airport.airport_name or "the identified airport"
    park_disp = park.park_name or "the selected park"

    # airport_named
    node_named = evaluator.add_custom_node(
        result=bool(airport.airport_name and airport.airport_name.strip()),
        id="airport_named",
        desc="Names one specific major airport.",
        parent=section,
        critical=True
    )

    # airport_in_california
    node_in_ca = evaluator.add_leaf(
        id="airport_in_california",
        desc="The identified airport is located in California.",
        parent=section,
        critical=True
    )
    all_airport_urls = clean_urls(airport.proximity_urls + airport.touchless_id_urls)
    claim_in_ca = f"{airport_name_disp} is located in California."
    await evaluator.verify(
        claim=claim_in_ca,
        node=node_in_ca,
        sources=all_airport_urls if all_airport_urls else None,
        additional_instruction="If possible, confirm via the airport's official site or authoritative listings that the airport is in California."
    )

    # airport_is_near_selected_park
    node_near = evaluator.add_leaf(
        id="airport_is_near_selected_park",
        desc="The answer indicates the airport is near the selected park and provides at least one supporting URL that substantiates proximity (e.g., an official source or mapping/directions page showing the relationship).",
        parent=section,
        critical=True
    )
    claim_near = f"{airport_name_disp} is near {park_disp} (reasonable driving distance) as supported by the provided link(s)."
    prox_urls = clean_urls(airport.proximity_urls)
    await evaluator.verify(
        claim=claim_near,
        node=node_near,
        sources=prox_urls if prox_urls else None,
        additional_instruction="Prefer a mapping/directions page or other authoritative evidence indicating proximity. If no URL is provided, mark as NOT SUPPORTED."
    )

    # touchless_id_supported_at_airport
    node_touchless = evaluator.add_leaf(
        id="touchless_id_supported_at_airport",
        desc="Provides support that the identified airport supports TSA PreCheck Touchless ID technology (in the stated Spring 2026 context).",
        parent=section,
        critical=True
    )
    touch_urls = clean_urls(airport.touchless_id_urls)
    claim_touchless = f"{airport_name_disp} supports TSA PreCheck Touchless ID / Digital ID."
    await evaluator.verify(
        claim=claim_touchless,
        node=node_touchless,
        sources=touch_urls if touch_urls else None,
        additional_instruction="Accept synonymous phrasing like 'TSA PreCheck Digital ID' or 'Touchless ID' operated by TSA. If no URL is provided, mark as NOT SUPPORTED."
    )

    # touchless_id_airport_citations
    cites_ok = (len(touch_urls) > 0) and (len(prox_urls) > 0)
    node_air_cites = evaluator.add_custom_node(
        result=cites_ok,
        id="touchless_id_airport_citations",
        desc="Provides supporting reference URL(s) for Touchless ID availability and for the proximity claim.",
        parent=section,
        critical=True
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
    Evaluate an answer for the California State Park MLK 2026 winter camping planning task.
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
        default_model=model
    )

    # Extract all sections (in parallel)
    park_task = evaluator.extract(
        prompt=prompt_extract_park_selection(),
        template_class=ParkSelectionExtraction,
        extraction_name="park_selection"
    )
    release_task = evaluator.extract(
        prompt=prompt_extract_reservation_release(),
        template_class=ReservationReleaseExtraction,
        extraction_name="reservation_release"
    )
    cost_task = evaluator.extract(
        prompt=prompt_extract_costs(),
        template_class=CostExtraction,
        extraction_name="costs"
    )
    nps_task = evaluator.extract(
        prompt=prompt_extract_nps_policy(),
        template_class=NPSPolicyExtraction,
        extraction_name="nps_policy"
    )

    # Await first four
    park_ex, release_ex, cost_ex, nps_ex = await asyncio.gather(
        park_task, release_task, cost_task, nps_task
    )

    # Airport extraction depends on park name for prompt context
    airport_ex = await evaluator.extract(
        prompt=prompt_extract_airport(park_ex.park_name or "the selected park"),
        template_class=AirportExtraction,
        extraction_name="airport"
    )

    # Add some ground truth/context info
    evaluator.add_ground_truth({
        "arrival_date_iso": ARRIVAL_DATE_ISO,
        "expected_release_date_iso": compute_six_months_prior(ARRIVAL_DATE_ISO),
        "expected_release_pacific_tz": expected_pacific_tz_for_date(compute_six_months_prior(ARRIVAL_DATE_ISO) or "2025-07-17"),
        "notes": "ReserveCalifornia commonly releases sites 6 months in advance at 8:00 AM Pacific."
    }, gt_type="reservation_rule_context")

    # Build and verify each rubric branch
    await verify_park_selection(evaluator, root, park_ex)
    await verify_reservation_release(evaluator, root, release_ex)
    await verify_costs(evaluator, root, cost_ex)
    await verify_nps_policy_and_pass(evaluator, root, nps_ex)
    await verify_airport(evaluator, root, airport_ex, park_ex)

    # Return the full evaluation summary
    return evaluator.get_summary()