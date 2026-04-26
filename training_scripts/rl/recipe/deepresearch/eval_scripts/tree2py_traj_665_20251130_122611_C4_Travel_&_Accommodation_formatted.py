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
TASK_ID = "so_cal_trip_yosemite_cruise"
TASK_DESCRIPTION = (
    "A family of 4 adults is planning a 10-day Southern California vacation that includes visiting Yosemite National Park "
    "on two separate occasions (requiring two separate entrance fees) and departing on a Disney cruise from San Diego. "
    "For their Yosemite visits, compare the total cost of purchasing two separate private vehicle entrance passes versus "
    "purchasing one America the Beautiful Annual Pass, and recommend which option is more cost-effective for this specific trip. "
    "Additionally, provide the following information for their cruise departure: (1) The street address of a Disney Cruise Line "
    "terminal in San Diego, (2) Whether long-term parking is available directly at the cruise terminal itself, (3) The name of at least one "
    "off-site parking facility that provides shuttle service to the cruise terminal, and (4) The approximate distance of that parking facility "
    "from the cruise terminal."
)

YOSEMITE_VEHICLE_FEE = 35  # $35 per vehicle per visit (7-day pass)
ANNUAL_PASS_PRICE = 80     # $80 America the Beautiful Annual Pass
EXPECTED_TWO_VISITS_TOTAL = YOSEMITE_VEHICLE_FEE * 2  # $70

ALLOWED_TERMINAL_ADDRESSES = [
    # Allow minor variations in formatting; verification will be lenient
    "1140 N. Harbor Drive, San Diego, CA 92101",  # B Street Pier
    "1000 N. Harbor Drive, San Diego, CA 92101",  # Broadway Pier
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class YosemitePassComparison(BaseModel):
    vehicle_pass_price_mentioned: Optional[str] = None
    vehicle_total_cost_two_visits: Optional[str] = None
    annual_pass_price_mentioned: Optional[str] = None
    annual_pass_total_cost: Optional[str] = None
    recommendation: Optional[str] = None
    recommendation_reason: Optional[str] = None
    yosemite_vehicle_sources: List[str] = Field(default_factory=list)
    yosemite_annual_sources: List[str] = Field(default_factory=list)


class CruiseDepartureInfo(BaseModel):
    terminal_address: Optional[str] = None
    terminal_pier_name: Optional[str] = None
    terminal_sources: List[str] = Field(default_factory=list)

    terminal_parking_availability: Optional[str] = None  # Expected values in answer: "yes"/"no"/"unknown"
    terminal_parking_sources: List[str] = Field(default_factory=list)

    offsite_parking_facility_name: Optional[str] = None
    offsite_parking_shuttle_service: Optional[str] = None  # Expected "yes"/"no"/"unknown"
    offsite_parking_sources: List[str] = Field(default_factory=list)

    offsite_parking_distance: Optional[str] = None  # e.g., "0.8 miles", "1.3 mi", or "not available"
    offsite_parking_distance_sources: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    yosemite: Optional[YosemitePassComparison] = None
    cruise: Optional[CruiseDepartureInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return (
        "Extract the specific details from the answer for two parts: Yosemite pass comparison and San Diego cruise departure logistics.\n\n"
        "Part A: Yosemite Pass Comparison\n"
        "- vehicle_pass_price_mentioned: The per-vehicle Yosemite entrance price explicitly used in the answer (string as written; e.g., \"$35\"). If missing, null.\n"
        "- vehicle_total_cost_two_visits: The total cost for two separate private vehicle passes (string as written; e.g., \"$70\"). If missing, null.\n"
        "- annual_pass_price_mentioned: The America the Beautiful Annual Pass price explicitly used in the answer (string; e.g., \"$80\"). If missing, null.\n"
        "- annual_pass_total_cost: The total cost explicitly stated for the annual pass (string; e.g., \"$80\"). If missing, null.\n"
        "- recommendation: The option recommended as more cost-effective for this specific trip (free text as written).\n"
        "- recommendation_reason: The supporting reasoning text as written.\n"
        "- yosemite_vehicle_sources: All URLs in the answer that support Yosemite vehicle fee info. Return only actual URLs; if none, return an empty list.\n"
        "- yosemite_annual_sources: All URLs in the answer that support annual pass price info. Return only actual URLs; if none, return an empty list.\n\n"
        "Part B: San Diego Disney Cruise Departure Logistics\n"
        "- terminal_address: The street address provided for the Disney Cruise Line terminal in San Diego (free text as written).\n"
        "- terminal_pier_name: The pier name if mentioned (e.g., \"B Street Pier\" or \"Broadway Pier\"). If not explicitly mentioned, null.\n"
        "- terminal_sources: All URLs cited regarding the terminal/address. Only extract actual URLs; if none, empty list.\n"
        "- terminal_parking_availability: Whether long-term parking is available directly at the terminal facilities, as stated in the answer (use 'yes', 'no', or 'unknown' based on the answer).\n"
        "- terminal_parking_sources: All URLs cited regarding terminal parking availability.\n"
        "- offsite_parking_facility_name: The name of at least one off-site parking facility mentioned.\n"
        "- offsite_parking_shuttle_service: Whether the answer claims the facility provides shuttle service to the cruise terminal (use 'yes', 'no', or 'unknown').\n"
        "- offsite_parking_sources: All URLs cited for the off-site facility/shuttle claim.\n"
        "- offsite_parking_distance: The approximate distance from the off-site facility to the cruise terminal as stated (string; e.g., \"0.8 miles\"). If the answer explicitly says the distance is not available, put \"not available\".\n"
        "- offsite_parking_distance_sources: URLs supporting the distance claim; if none, empty list.\n\n"
        "Rules:\n"
        "1. Extract only what is explicitly present in the answer. Do not invent.\n"
        "2. For URLs, extract the actual link strings (plain URLs or from markdown links). If the answer references a site without a URL, do not add one.\n"
        "3. Keep values as strings exactly as written in the answer; do not normalize numbers.\n"
        "4. If any field is missing in the answer, return null (or empty list for URL fields).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _text_indicates_unavailable(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(
        kw in t
        for kw in ["not available", "n/a", "unknown", "not stated", "not provided", "no info"]
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_yosemite_section(
    evaluator: Evaluator,
    parent_node,
    yosemite: Optional[YosemitePassComparison],
) -> None:
    """
    Build the Yosemite pass decision subtree:
    - pass_cost_comparison (parallel, critical) with three checks
    - pass_recommendation (leaf, critical)
    """
    yosemite_node = evaluator.add_sequential(
        id="yosemite_pass_decision",
        desc="Yosemite entrance fee cost comparison and cost-effective recommendation for this trip",
        parent=parent_node,
        critical=True,
    )

    # Sub-node: cost comparison checks (parallel, critical)
    compare_node = evaluator.add_parallel(
        id="pass_cost_comparison",
        desc="Correctly calculates the total cost of two separate Yosemite private vehicle entrance passes (using $35 per vehicle × 2 visits) and compares it to the cost of one America the Beautiful Annual Pass (using $80)",
        parent=yosemite_node,
        critical=True,
    )

    # 1) Vehicle price used $35
    vehicle_price_leaf = evaluator.add_leaf(
        id="vehicle_price_used_35",
        desc="Uses Yosemite private vehicle entrance fee of $35 per vehicle per visit",
        parent=compare_node,
        critical=True,
    )
    claim_vehicle_price = (
        "The answer explicitly uses the Yosemite private vehicle entrance fee of $35 per vehicle for each visit."
    )
    await evaluator.verify(
        claim=claim_vehicle_price,
        node=vehicle_price_leaf,
        sources=(yosemite.yosemite_vehicle_sources if yosemite else None),
        additional_instruction=(
            "If URLs are provided, confirm the stated price matches $35 on an official or credible source (e.g., NPS). "
            "If no URLs are provided, judge based on whether the answer itself states $35. Allow minor formatting variations."
        ),
    )

    # 2) Annual pass price used $80
    annual_price_leaf = evaluator.add_leaf(
        id="annual_price_used_80",
        desc="Uses America the Beautiful Annual Pass price of $80",
        parent=compare_node,
        critical=True,
    )
    claim_annual_price = (
        "The answer explicitly uses the America the Beautiful Annual Pass price of $80."
    )
    await evaluator.verify(
        claim=claim_annual_price,
        node=annual_price_leaf,
        sources=(yosemite.yosemite_annual_sources if yosemite else None),
        additional_instruction=(
            "If URLs are provided, confirm the stated annual pass price matches $80 from an official or credible source. "
            "If no URLs are provided, judge based on whether the answer itself states $80."
        ),
    )

    # 3) Vehicle total for two visits equals $70 (math check)
    vehicle_total_leaf = evaluator.add_leaf(
        id="vehicle_total_is_70",
        desc="Computes two separate vehicle passes total as $70 ($35 × 2 visits)",
        parent=compare_node,
        critical=True,
    )
    claim_vehicle_total = (
        "Based on the per-visit Yosemite vehicle entrance fee of $35, the correct total cost for two separate visits is $70, "
        "and the answer's computed total for two separate passes equals $70."
    )
    await evaluator.verify(
        claim=claim_vehicle_total,
        node=vehicle_total_leaf,
        sources=None,
        additional_instruction=(
            "Focus on the arithmetic and the answer's stated total. Minor formatting variations are acceptable."
        ),
    )

    # Recommendation leaf (critical) — gated by the sequential strategy
    recommend_leaf = evaluator.add_leaf(
        id="pass_recommendation",
        desc="Recommends the more cost-effective option based on the computed totals, with reasoning consistent with the comparison",
        parent=yosemite_node,
        critical=True,
    )
    claim_recommendation = (
        "Given two separate Yosemite visits requiring two entrance fees, the more cost-effective option for this specific trip "
        "is purchasing two separate private vehicle entrance passes ($70) rather than the $80 annual pass, and the answer recommends that option."
    )
    await evaluator.verify(
        claim=claim_recommendation,
        node=recommend_leaf,
        sources=None,
        additional_instruction=(
            "Check whether the answer explicitly recommends the cheaper option ($70 for two vehicle passes vs $80 annual pass) "
            "and that its reasoning aligns with the comparison."
        ),
    )


async def verify_cruise_section(
    evaluator: Evaluator,
    parent_node,
    cruise: Optional[CruiseDepartureInfo],
) -> None:
    """
    Build the San Diego cruise terminal and parking logistics subtree (parallel, critical).
    """
    cruise_node = evaluator.add_parallel(
        id="cruise_departure_info",
        desc="San Diego Disney Cruise Line departure terminal and parking logistics",
        parent=parent_node,
        critical=True,
    )

    # Terminal address leaf
    term_addr_leaf = evaluator.add_leaf(
        id="terminal_address",
        desc=(
            "Provides a valid street address for a Disney Cruise Line terminal in San Diego, consistent with the constraints: "
            "either B Street Pier (1140 N. Harbor Drive) or Broadway Pier (1000 N. Harbor Drive)."
        ),
        parent=cruise_node,
        critical=True,
    )
    provided_address = cruise.terminal_address if cruise else ""
    claim_term_addr = (
        f"The provided terminal address '{provided_address}' matches either '1140 N. Harbor Drive' (B Street Pier) or "
        f"'1000 N. Harbor Drive' (Broadway Pier) in San Diego."
    )
    await evaluator.verify(
        claim=claim_term_addr,
        node=term_addr_leaf,
        sources=(cruise.terminal_sources if cruise else None),
        additional_instruction=(
            "Allow minor formatting variations (e.g., 'Harbor Dr' vs 'Harbor Drive', missing ZIP). "
            "You are verifying whether the answer's stated address corresponds to one of the two permitted addresses."
        ),
    )

    # Terminal parking availability leaf
    term_parking_leaf = evaluator.add_leaf(
        id="terminal_parking_availability",
        desc="Correctly states whether long-term parking is available directly at the cruise terminal itself (per constraints: NOT available directly at the terminal facilities)",
        parent=cruise_node,
        critical=True,
    )
    claim_term_parking = (
        "The answer correctly states that long-term parking is NOT available directly at the cruise terminal itself (B Street or Broadway Pier)."
    )
    await evaluator.verify(
        claim=claim_term_parking,
        node=term_parking_leaf,
        sources=(cruise.terminal_parking_sources if cruise else None),
        additional_instruction=(
            "Check the answer's statement regarding long-term parking at the terminal. "
            "If URLs are provided, confirm they support the claim that long-term parking is not available directly at the terminal facilities."
        ),
    )

    # Off-site parking facility with shuttle leaf
    offsite_shuttle_leaf = evaluator.add_leaf(
        id="offsite_parking_facility_with_shuttle",
        desc="Identifies at least one off-site parking facility from the verified providers list and confirms it provides shuttle service to the cruise terminal",
        parent=cruise_node,
        critical=True,
    )
    facility_name = cruise.offsite_parking_facility_name if cruise else ""
    claim_offsite_shuttle = (
        f"The off-site parking facility '{facility_name}' provides shuttle service to the San Diego cruise terminal."
    )
    await evaluator.verify(
        claim=claim_offsite_shuttle,
        node=offsite_shuttle_leaf,
        sources=(cruise.offsite_parking_sources if cruise else None),
        additional_instruction=(
            "Confirm via the provided URLs (provider site or credible sources) that the named facility offers shuttle service to the cruise terminal "
            "(not just to the airport). If the URLs do not support shuttle-to-terminal, this should fail."
        ),
    )

    # Parking distance leaf
    distance_leaf = evaluator.add_leaf(
        id="parking_distance",
        desc="States the approximate distance of the identified off-site parking facility from the cruise terminal, OR explicitly states that an approximate distance was not available from the cited/available sources",
        parent=cruise_node,
        critical=True,
    )
    distance_text = cruise.offsite_parking_distance if cruise else None

    if _text_indicates_unavailable(distance_text):
        claim_distance = (
            "The answer explicitly states that an approximate distance from the off-site parking facility to the cruise terminal "
            "was not available from the cited or available sources."
        )
        sources_for_distance = (cruise.offsite_parking_distance_sources if cruise else None)
    else:
        dist_val = distance_text or ""
        claim_distance = (
            f"The approximate distance of the off-site parking facility '{facility_name}' from the San Diego cruise terminal "
            f"is {dist_val}."
        )
        # Prefer distance-specific sources; fall back to facility sources
        sources_for_distance = (
            cruise.offsite_parking_distance_sources if cruise and cruise.offsite_parking_distance_sources else
            (cruise.offsite_parking_sources if cruise else None)
        )

    await evaluator.verify(
        claim=claim_distance,
        node=distance_leaf,
        sources=sources_for_distance,
        additional_instruction=(
            "If a distance is provided, verify that the cited/available sources support it (allow minor rounding differences). "
            "If the answer states distance was not available from sources, verify that the answer explicitly made that statement."
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
    Evaluate the answer for the Southern California trip planning task, including Yosemite pass comparison and
    San Diego Disney Cruise Line terminal logistics.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent branches (Yosemite decision, Cruise logistics)
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Add ground truth / reference info
    evaluator.add_ground_truth({
        "yosemite_vehicle_fee_usd": YOSEMITE_VEHICLE_FEE,
        "annual_pass_price_usd": ANNUAL_PASS_PRICE,
        "two_visits_total_usd": EXPECTED_TWO_VISITS_TOTAL,
        "allowed_terminal_addresses": ALLOWED_TERMINAL_ADDRESSES,
        "terminal_long_term_parking_available": "no"
    }, gt_type="reference_values")

    # Build verification tree
    await verify_yosemite_section(evaluator, root, extracted.yosemite)
    await verify_cruise_section(evaluator, root, extracted.cruise)

    # Return evaluation summary
    return evaluator.get_summary()