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
TASK_ID = "cruise_vacation_planning"
TASK_DESCRIPTION = (
    "I am planning a cruise vacation and gathering travel cost information. I need to identify a specific cruise ship and provide related travel expenses based on the following criteria:\n\n"
    "The cruise ship must meet ALL of these requirements:\n"
    "- Made its maiden voyage on November 20, 2025\n"
    "- Departs from Port Everglades in Fort Lauderdale, Florida\n"
    "- Offers both 4-night and 5-night itineraries to the Bahamas\n\n"
    "Please provide the following information:\n\n"
    "1. What is the name of this cruise ship?\n"
    "2. What is the passenger capacity of this ship at double occupancy?\n"
    "3. What is the official daily parking rate for standard vehicles at Port Everglades?\n"
    "4. What is the cost of the 2025 America the Beautiful Annual Pass for U.S. residents (excluding online processing fees)?\n"
    "5. What is the entrance fee per vehicle for Yosemite National Park that is valid for 7 days?\n\n"
    "All information must be supported with reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CruiseShipExtraction(BaseModel):
    # Ship identification
    ship_name: Optional[str] = None
    ship_name_urls: List[str] = Field(default_factory=list)

    maiden_voyage_date: Optional[str] = None
    maiden_voyage_urls: List[str] = Field(default_factory=list)

    departs_port_everglades_urls: List[str] = Field(default_factory=list)

    bahamas_itineraries_urls: List[str] = Field(default_factory=list)

    # Passenger capacity (double occupancy)
    passenger_capacity_double_occupancy: Optional[str] = None
    passenger_capacity_urls: List[str] = Field(default_factory=list)

    # Port Everglades parking rate
    port_everglades_daily_parking_rate: Optional[str] = None
    port_everglades_parking_urls: List[str] = Field(default_factory=list)

    # America the Beautiful Annual Pass (2025)
    america_beautiful_pass_2025_price: Optional[str] = None
    america_beautiful_pass_urls: List[str] = Field(default_factory=list)

    # Yosemite 7-day vehicle entrance fee
    yosemite_vehicle_7_day_fee: Optional[str] = None
    yosemite_fee_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following structured information exactly as presented in the answer. Do not invent any values, and only extract URLs that explicitly appear in the answer.

Required fields (return null for any missing value, and [] for any missing URL list):
- ship_name: The name of the cruise ship that the answer identifies as meeting all criteria.
- ship_name_urls: All URLs in the answer that refer to or are about this ship (e.g., official ship page, press release, cruise line page, itinerary page that clearly references the ship).
- maiden_voyage_date: The maiden voyage date mentioned for the identified ship (e.g., "November 20, 2025").
- maiden_voyage_urls: URLs that explicitly support the ship’s maiden voyage timing/date.
- departs_port_everglades_urls: URLs that explicitly support that the ship departs from Port Everglades (Fort Lauderdale, FL).
- bahamas_itineraries_urls: URLs that explicitly support that the ship offers both 4-night and 5-night itineraries to the Bahamas (pages can be separate for 4-night and 5-night; include all relevant).
- passenger_capacity_double_occupancy: The stated passenger capacity at double occupancy for the identified ship (as a concrete numeric value string like "3,250" or "3,250 passengers").
- passenger_capacity_urls: URL(s) supporting the double-occupancy capacity (prefer official/authoritative sources like cruise line spec pages).
- port_everglades_daily_parking_rate: The official daily parking rate for standard vehicles at Port Everglades (return the textual amount as written, e.g., "$20/day").
- port_everglades_parking_urls: Official Port Everglades or port authority URL(s) supporting the daily parking rate.
- america_beautiful_pass_2025_price: The 2025 America the Beautiful Annual Pass price for U.S. residents excluding online processing fees (as a textual price string, e.g., "$80").
- america_beautiful_pass_urls: Official authoritative URL(s) (e.g., NPS.gov or USGS.gov) that support the 2025 price.
- yosemite_vehicle_7_day_fee: The entrance fee per vehicle for Yosemite National Park valid for 7 days (as a textual price string).
- yosemite_fee_urls: Official NPS Yosemite (or NPS) URL(s) that support the 7-day private vehicle fee.

Important:
- For each URL field, include only actual URLs explicitly present in the answer (plain or markdown links).
- Do not add URL(s) that are not in the answer.
- Keep values as strings exactly as written in the answer (e.g., keep currency symbols, commas, etc.).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_ship_related_urls(ex: CruiseShipExtraction) -> List[str]:
    """Union of all ship-related URLs (name, maiden, departure, itins, capacity)."""
    urls = (
        ex.ship_name_urls
        + ex.maiden_voyage_urls
        + ex.departs_port_everglades_urls
        + ex.bahamas_itineraries_urls
        + ex.passenger_capacity_urls
    )
    seen = set()
    combined = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            combined.append(u)
    return combined


def _ins_if_no_sources(sources: List[str], base_instruction: str = "None") -> str:
    """
    If no sources were provided in the answer for this claim, explicitly instruct the judge to return Incorrect.
    """
    if not sources:
        return "No source URLs were provided in the answer for this claim; return Incorrect."
    return base_instruction


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: CruiseShipExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    All node criticalities and strategies follow the rubric. Each verification leaf is binary.
    """

    # Root-level aggregate node (critical, parallel)
    root_main = evaluator.add_parallel(
        id="cruise_vacation_planning",
        desc="Evaluate whether the answer identifies a cruise ship matching the required criteria and provides all requested travel cost information with supporting reference URLs from authoritative/official sources where applicable.",
        parent=evaluator.root,
        critical=True,
    )

    # Subtree: Cruise ship identification (critical, parallel)
    ship_ident = evaluator.add_parallel(
        id="cruise_ship_identification",
        desc="Identify a cruise ship that meets all specified criteria, with supporting URL(s).",
        parent=root_main,
        critical=True,
    )

    # Prepare common ship info
    ship_name = extracted.ship_name or ""
    # Ship-related URLs for general/backup usage
    all_ship_urls = _combine_ship_related_urls(extracted)

    # 1) Ship name provided (leaf)
    node_ship_name = evaluator.add_leaf(
        id="ship_name_provided",
        desc="Provides the cruise ship name (a specific ship is clearly identified) with a supporting reference URL.",
        parent=ship_ident,
        critical=True,
    )
    claim_ship_name = (
        f"At least one of the provided URLs is about the cruise ship named '{ship_name}'. "
        f"The ship name in the answer is clearly identified."
    )
    sources_ship_name = extracted.ship_name_urls or all_ship_urls
    add_ins_ship_name = _ins_if_no_sources(
        sources_ship_name,
        base_instruction="Allow minor naming variations (e.g., inclusion of cruise line name)."
    )
    # Verify (by URLs if provided; else simple with special instruction to fail)
    await evaluator.verify(
        claim=claim_ship_name,
        node=node_ship_name,
        sources=sources_ship_name if sources_ship_name else None,
        additional_instruction=add_ins_ship_name,
    )

    # 2) Maiden voyage date matches Nov 20, 2025 (leaf)
    node_maiden = evaluator.add_leaf(
        id="maiden_voyage_date_matches",
        desc="Shows (with a reference URL) that the identified ship’s maiden voyage date is November 20, 2025.",
        parent=ship_ident,
        critical=True,
    )
    claim_maiden = (
        f"The ship '{ship_name}' has its maiden voyage (inaugural/first sailing) on November 20, 2025."
    )
    sources_maiden = extracted.maiden_voyage_urls or all_ship_urls
    add_ins_maiden = _ins_if_no_sources(
        sources_maiden,
        base_instruction="Accept minor date format variants (e.g., Nov 20, 2025). "
                         "The page should explicitly indicate inaugural/maiden/first sailing equals November 20, 2025."
    )
    await evaluator.verify(
        claim=claim_maiden,
        node=node_maiden,
        sources=sources_maiden if sources_maiden else None,
        additional_instruction=add_ins_maiden,
    )

    # 3) Departs from Port Everglades (leaf)
    node_departs = evaluator.add_leaf(
        id="departs_port_everglades",
        desc="Shows (with a reference URL) that the identified ship departs from Port Everglades in Fort Lauderdale, Florida.",
        parent=ship_ident,
        critical=True,
    )
    claim_departs = (
        f"The ship '{ship_name}' departs from Port Everglades (Fort Lauderdale, Florida)."
    )
    sources_departs = extracted.departs_port_everglades_urls or all_ship_urls
    add_ins_departs = _ins_if_no_sources(
        sources_departs,
        base_instruction="Allow phrasing variants like 'Fort Lauderdale (Port Everglades)' or itinerary departure "
                         "ports that clearly indicate Port Everglades."
    )
    await evaluator.verify(
        claim=claim_departs,
        node=node_departs,
        sources=sources_departs if sources_departs else None,
        additional_instruction=add_ins_departs,
    )

    # 4) Offers both 4-night and 5-night Bahamas itineraries (leaf)
    node_bahamas = evaluator.add_leaf(
        id="offers_4_and_5_night_bahamas",
        desc="Shows (with a reference URL) that the identified ship offers both 4-night and 5-night itineraries to the Bahamas.",
        parent=ship_ident,
        critical=True,
    )
    claim_bahamas = (
        f"The ship '{ship_name}' offers both 4-night and 5-night itineraries to the Bahamas."
    )
    sources_bahamas = extracted.bahamas_itineraries_urls or all_ship_urls
    add_ins_bahamas = _ins_if_no_sources(
        sources_bahamas,
        base_instruction="It's acceptable if 4-night and 5-night Bahamas itineraries are shown on different pages; "
                         "the combined sources must support both durations to the Bahamas."
    )
    await evaluator.verify(
        claim=claim_bahamas,
        node=node_bahamas,
        sources=sources_bahamas if sources_bahamas else None,
        additional_instruction=add_ins_bahamas,
    )

    # ------------------------------------------------------------------ #
    # Additional critical leaves under the main planning node            #
    # ------------------------------------------------------------------ #

    # Passenger capacity at double occupancy (leaf)
    node_capacity = evaluator.add_leaf(
        id="passenger_capacity_double_occupancy",
        desc="Provides the ship’s passenger capacity at double occupancy as a specific numeric value and supports it with an official/authoritative source URL (e.g., cruise line, ship spec sheet, or similarly authoritative primary source).",
        parent=root_main,
        critical=True,
    )
    cap_val = extracted.passenger_capacity_double_occupancy or ""
    claim_capacity = (
        f"The passenger capacity at double occupancy for the ship '{ship_name}' is {cap_val}."
    )
    sources_capacity = extracted.passenger_capacity_urls or all_ship_urls
    add_ins_capacity = _ins_if_no_sources(
        sources_capacity,
        base_instruction="Verify that the page states the ship's capacity specifically at double occupancy. "
                         "Prefer official/authoritative sources (e.g., the cruise line's official site or spec sheet). "
                         "If the source is clearly not official/authoritative and does not provide credible specification data, return Incorrect. "
                         "Allow reasonable numeric formatting variants (commas, 'passengers')."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=sources_capacity if sources_capacity else None,
        additional_instruction=add_ins_capacity,
    )

    # Port Everglades daily parking rate for standard vehicles (leaf)
    node_parking = evaluator.add_leaf(
        id="port_everglades_daily_parking_rate",
        desc="Provides the official daily parking rate for standard vehicles at Port Everglades and supports it with an official Port Everglades (or port authority) URL.",
        parent=root_main,
        critical=True,
    )
    parking_val = extracted.port_everglades_daily_parking_rate or ""
    claim_parking = (
        f"The official daily parking rate for standard vehicles at Port Everglades is {parking_val}."
    )
    sources_parking = extracted.port_everglades_parking_urls
    add_ins_parking = _ins_if_no_sources(
        sources_parking,
        base_instruction="The source must be an official Port Everglades/Broward County site (e.g., broward.org/n or porteverglades). "
                         "If not official, return Incorrect. Accept minor rate formatting variants."
    )
    await evaluator.verify(
        claim=claim_parking,
        node=node_parking,
        sources=sources_parking if sources_parking else None,
        additional_instruction=add_ins_parking,
    )

    # America the Beautiful Annual Pass (2025) price excluding online processing fees (leaf)
    node_pass = evaluator.add_leaf(
        id="america_the_beautiful_pass_2025_price",
        desc="Provides the 2025 America the Beautiful Annual Pass base price for U.S. residents excluding online processing fees, supported by an official authoritative URL (e.g., NPS/USGS official page).",
        parent=root_main,
        critical=True,
    )
    pass_val = extracted.america_beautiful_pass_2025_price or ""
    claim_pass = (
        f"The 2025 America the Beautiful Annual Pass price for U.S. residents, excluding any online processing fees, is {pass_val}."
    )
    sources_pass = extracted.america_beautiful_pass_urls
    add_ins_pass = _ins_if_no_sources(
        sources_pass,
        base_instruction="The source must be an official NPS.gov or USGS.gov page (or equivalently authoritative federal site). "
                         "Confirm that the price is for 2025 or otherwise clearly the current price applicable in 2025; "
                         "exclude online processing fees. If the source is not official, return Incorrect."
    )
    await evaluator.verify(
        claim=claim_pass,
        node=node_pass,
        sources=sources_pass if sources_pass else None,
        additional_instruction=add_ins_pass,
    )

    # Yosemite 7-day per-vehicle entrance fee (leaf)
    node_yose = evaluator.add_leaf(
        id="yosemite_7_day_vehicle_fee",
        desc="Provides Yosemite National Park entrance fee per vehicle valid for 7 days, supported by an official NPS Yosemite (or NPS) URL.",
        parent=root_main,
        critical=True,
    )
    yose_val = extracted.yosemite_vehicle_7_day_fee or ""
    claim_yose = (
        f"The Yosemite National Park entrance fee per private vehicle valid for 7 days is {yose_val}."
    )
    sources_yose = extracted.yosemite_fee_urls
    add_ins_yose = _ins_if_no_sources(
        sources_yose,
        base_instruction="The source must be an official NPS Yosemite (nps.gov/yose) or NPS site. "
                         "Accept phrasing variants like 'Private Vehicle (non-commercial) 7-day pass'. "
                         "If the source is not an official NPS site, return Incorrect."
    )
    await evaluator.verify(
        claim=claim_yose,
        node=node_yose,
        sources=sources_yose if sources_yose else None,
        additional_instruction=add_ins_yose,
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the cruise vacation planning task and return a structured summary.
    """

    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root strategy; actual rubric root is a child node with critical=True
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

    # 1) Extract structured content from the answer
    extracted: CruiseShipExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=CruiseShipExtraction,
        extraction_name="cruise_vacation_planning_extraction",
    )

    # 2) Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()