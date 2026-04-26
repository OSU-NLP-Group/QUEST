import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "whistler_budget_trip_feb2026"
TASK_DESCRIPTION = """You are planning a ski trip to Whistler, British Columbia for mid-February 2026, departing from the Washington DC metropolitan area. You want to use budget airlines to minimize costs and are booking in early January 2026 (more than 28 days in advance).

Task:
1. Identify the complete budget airline routing from a Washington DC area airport to reach Whistler (via Vancouver, BC), specifying:
   - The departure airport in the DC area
   - The connecting city/hub
   - The destination airport for accessing Whistler
   - The budget airline that operates this route

2. Calculate the estimated per-person cost for lift tickets at Whistler Blackcomb for one day of skiing in mid-February 2026, taking into account:
   - The standard adult lift ticket price
   - The advance purchase discount available when booking 28+ days ahead
   - Provide the final estimated lift ticket cost in USD

3. Confirm that Whistler Blackcomb is operationally accessible during the February 2026 winter season.

Requirements:
- Use only budget airlines operating from DC area airports
- Apply advance purchase discount for lift tickets (28+ days booking window)
- Provide specific airport codes and airline names
- Show cost calculations with the discount applied
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RouteExtraction(BaseModel):
    departure_airport_name: Optional[str] = None
    departure_airport_code: Optional[str] = None
    connecting_city: Optional[str] = None
    connecting_airport_code: Optional[str] = None
    destination_airport_name: Optional[str] = None
    destination_airport_code: Optional[str] = None
    airline_name: Optional[str] = None
    route_sources: List[str] = Field(default_factory=list)


class LiftTicketExtraction(BaseModel):
    standard_price_cad: Optional[str] = None
    standard_price_usd: Optional[str] = None
    advance_discount_percent: Optional[str] = None
    final_price_usd: Optional[str] = None
    ticket_sources: List[str] = Field(default_factory=list)


class AccessibilityExtraction(BaseModel):
    claim_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_route() -> str:
    return """
    Extract the budget airline routing details exactly as stated in the answer.

    Required fields:
    - departure_airport_name: The full name of the departure airport in the DC area (e.g., "Baltimore/Washington International Thurgood Marshall")
    - departure_airport_code: The 3-letter IATA code (e.g., "BWI")
    - connecting_city: The name of the connecting city or hub (e.g., "Minneapolis/St. Paul")
    - connecting_airport_code: The 3-letter IATA code of the connecting airport if provided (e.g., "MSP")
    - destination_airport_name: The full name of the destination airport for accessing Whistler (e.g., "Vancouver International Airport")
    - destination_airport_code: The 3-letter IATA code (e.g., "YVR")
    - airline_name: The budget airline operating the route (e.g., "Sun Country Airlines")
    - route_sources: All URLs cited in the answer that support the airline/route/booking information (extract only actual URLs)

    Rules:
    - Do not invent any information. If a field is not present in the answer, set it to null (or empty list for URLs).
    - For URLs, extract the concrete URL(s) explicitly present in the answer (including markdown links).
    """


def prompt_extract_lift_tickets() -> str:
    return """
    Extract the Whistler Blackcomb lift ticket pricing details exactly as stated in the answer for a 1-day adult ticket in mid-February 2026.

    Required fields:
    - standard_price_cad: The standard (window) adult 1-day price mentioned in CAD, if present (e.g., "$351 CAD")
    - standard_price_usd: The standard (window) adult 1-day price mentioned in USD, if present (e.g., "$258 USD")
    - advance_discount_percent: The advance purchase discount percentage mentioned for booking 28+ days in advance (e.g., "30%")
    - final_price_usd: The final per-person USD price after discount as presented in the answer (e.g., "$180.60")
    - ticket_sources: All URLs cited for pricing/discount policy (extract only actual URLs)

    Notes:
    - Return the values as they appear in the answer (strings). Do not convert currencies yourself.
    - If a field is missing, set it to null (or empty list for URLs).
    """


def prompt_extract_accessibility() -> str:
    return """
    Extract the answer's confirmation that Whistler Blackcomb is open/operational during the February 2026 winter season.

    Required fields:
    - claim_text: A concise sentence from the answer that asserts operational accessibility in February 2026, if present.
    - sources: All URLs cited to support season operations or operating calendars/status.

    If not explicitly stated or sources are missing, set claim_text to null and sources to an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_non_empty(*vals: Optional[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _parse_amount(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # Extract the first number (handles commas and decimals)
    m = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?', s)
    if not m:
        return None
    whole = m.group(1).replace(",", "")
    frac = m.group(2) or ""
    try:
        return float(f"{whole}.{frac}" if frac else whole)
    except Exception:
        return None


def _parse_percent(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_route(
    evaluator: Evaluator,
    parent_node,
    route: RouteExtraction
) -> None:
    """
    Build and verify the 'Budget_Airline_Route' subtree (critical, parallel).
    """
    route_node = evaluator.add_parallel(
        id="Budget_Airline_Route",
        desc="Identify the complete budget airline routing from the DC area to Whistler via Vancouver",
        parent=parent_node,
        critical=True
    )

    # Departure airport verification (BWI)
    dep_leaf = evaluator.add_leaf(
        id="Departure_Airport",
        desc="Identify that Baltimore/Washington (BWI) is the departure airport in the DC area",
        parent=route_node,
        critical=True
    )
    dep_val = _first_non_empty(route.departure_airport_code, route.departure_airport_name)
    dep_claim = (
        f"The departure airport identified in the answer ('{dep_val}') matches the expected Baltimore/Washington International (BWI). "
        f"Treat 'BWI', 'Baltimore/Washington', and 'Baltimore/Washington International Thurgood Marshall' as equivalent (case-insensitive)."
    )
    await evaluator.verify(
        claim=dep_claim,
        node=dep_leaf,
        additional_instruction="This is a simple equivalence check between the identified airport and the expected BWI. Allow common synonyms and abbreviations."
    )

    # Connection hub verification (MSP)
    conn_leaf = evaluator.add_leaf(
        id="Connection_Hub",
        desc="Identify that Minneapolis/St. Paul (MSP) is the connecting city/hub",
        parent=route_node,
        critical=True
    )
    conn_val = _first_non_empty(route.connecting_airport_code, route.connecting_city)
    conn_claim = (
        f"The connecting hub identified in the answer ('{conn_val}') matches the expected Minneapolis/St. Paul (MSP). "
        f"Treat 'MSP', 'Minneapolis', 'Minneapolis–Saint Paul', and 'Minneapolis/St. Paul' as equivalent (case-insensitive)."
    )
    await evaluator.verify(
        claim=conn_claim,
        node=conn_leaf,
        additional_instruction="Simple match check; allow common variations of Minneapolis–Saint Paul naming."
    )

    # Destination airport verification (YVR)
    dest_leaf = evaluator.add_leaf(
        id="Destination_Airport",
        desc="Identify that Vancouver (YVR) is the destination airport for accessing Whistler",
        parent=route_node,
        critical=True
    )
    dest_val = _first_non_empty(route.destination_airport_code, route.destination_airport_name)
    dest_claim = (
        f"The destination airport identified in the answer ('{dest_val}') matches the expected Vancouver International (YVR). "
        f"Treat 'YVR' and 'Vancouver International Airport' as equivalent (case-insensitive)."
    )
    await evaluator.verify(
        claim=dest_claim,
        node=dest_leaf,
        additional_instruction="Simple equivalence check between identified destination and YVR; allow reasonable synonyms."
    )

    # Airline verification (Sun Country Airlines)
    airline_leaf = evaluator.add_leaf(
        id="Airline_Identification",
        desc="Identify that Sun Country Airlines is the budget airline operating this route",
        parent=route_node,
        critical=True
    )
    airline_val = (route.airline_name or "").strip()
    airline_claim = (
        f"The budget airline identified in the answer ('{airline_val}') matches the expected 'Sun Country Airlines'. "
        f"Treat 'Sun Country' and 'Sun Country Airlines' as equivalent (case-insensitive)."
    )
    await evaluator.verify(
        claim=airline_claim,
        node=airline_leaf,
        additional_instruction="Simple name match; ignore minor wording differences like 'Airlines' suffix."
    )


async def verify_lift_tickets(
    evaluator: Evaluator,
    parent_node,
    lift: LiftTicketExtraction
) -> None:
    """
    Build and verify the 'Lift_Ticket_Cost_Calculation' subtree (critical, sequential).
    """
    lt_node = evaluator.add_sequential(
        id="Lift_Ticket_Cost_Calculation",
        desc="Calculate the estimated per-person lift ticket cost with advance purchase discount applied",
        parent=parent_node,
        critical=True
    )

    # 1) Standard price identification (expected: 351 CAD or 258 USD)
    std_leaf = evaluator.add_leaf(
        id="Standard_Price_Identification",
        desc="Identify the standard adult lift ticket price for mid-February 2026 ($351 CAD or $258 USD)",
        parent=lt_node,
        critical=True
    )
    std_cad = (lift.standard_price_cad or "").strip()
    std_usd = (lift.standard_price_usd or "").strip()
    std_claim = (
        "According to the cited webpages, the standard (window) adult 1-day lift ticket price for mid-February 2026 at Whistler Blackcomb "
        "is 351 CAD (about 258 USD). "
        f"The answer states the standard price as CAD='{std_cad}' and/or USD='{std_usd}'. "
        "Confirm that the stated standard price matches either 351 CAD or 258 USD (allowing minor formatting/rounding)."
    )
    await evaluator.verify(
        claim=std_claim,
        node=std_leaf,
        sources=lift.ticket_sources,
        additional_instruction="Use the provided pricing source(s). Accept either 351 CAD or 258 USD (± small rounding)."
    )

    # 2) Advance purchase discount application (up to 30% off; final USD price computed)
    adv_leaf = evaluator.add_leaf(
        id="Advance_Purchase_Discount_Application",
        desc="Apply the advance purchase discount (up to 30% off) available when booking 28+ days in advance, and calculate the final discounted price in USD",
        parent=lt_node,
        critical=True
    )

    # Prepare numeric hints for the verifier in the claim (optional)
    std_usd_val = _parse_amount(lift.standard_price_usd) or 258.0  # If missing, reference expected USD
    disc_pct_val = _parse_percent(lift.advance_discount_percent)
    final_usd_val = _parse_amount(lift.final_price_usd)
    calc_example_text = ""
    if disc_pct_val is not None:
        try:
            expected_final = std_usd_val * (1.0 - min(max(disc_pct_val, 0.0), 30.0) / 100.0)
            calc_example_text = (
                f"For reference, using a standard price of ${std_usd_val:.2f} USD and a {disc_pct_val:.2f}% discount "
                f"(capped at 30%) yields about ${expected_final:.2f} USD."
            )
        except Exception:
            calc_example_text = ""

    adv_claim = (
        "Booking in early January for mid-February is more than 28 days in advance, so the advance purchase policy should apply. "
        "According to the cited webpages, Whistler Blackcomb offers up to 30% off for advance purchases (28+ days). "
        f"The answer indicates an advance discount percent as '{(lift.advance_discount_percent or '').strip()}' "
        f"and a final USD price after discount as '{(lift.final_price_usd or '').strip()}'. "
        "Verify BOTH of the following: "
        "(a) the cited webpages support an advance-purchase discount of up to 30% (28+ days), and "
        "(b) the final USD price stated is consistent with applying a discount ≤ 30% to the standard USD price (allowing minor rounding). "
        + calc_example_text
    )
    await evaluator.verify(
        claim=adv_claim,
        node=adv_leaf,
        sources=lift.ticket_sources,
        additional_instruction="Check the policy text (advance purchase up to 30% off) and whether the arithmetic is consistent with a discount ≤ 30% applied to the standard rate (~$258 USD). Allow small rounding differences."
    )


async def verify_accessibility(
    evaluator: Evaluator,
    parent_node,
    acc: AccessibilityExtraction
) -> None:
    """
    Add and verify the 'Winter_Accessibility_Confirmation' leaf (critical).
    """
    access_leaf = evaluator.add_leaf(
        id="Winter_Accessibility_Confirmation",
        desc="Confirm that Whistler Blackcomb is operationally accessible during the February 2026 winter season",
        parent=parent_node,
        critical=True
    )
    access_claim = (
        "Whistler Blackcomb is open and operational for skiing during February 2026 (i.e., during the 2025–26 winter season). "
        "Confirm that the cited official pages (e.g., operations calendar, status/alerts) support this."
    )
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=acc.sources,
        additional_instruction="Rely on official Whistler Blackcomb or Vail Resorts sources if available. If multiple official sources are provided, any one that clearly indicates February 2026 operations is sufficient."
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
    Evaluate an answer for the Whistler budget trip planning task (February 2026).
    """
    # Initialize evaluator
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

    # Parallel extraction of all needed info
    route_task = evaluator.extract(
        prompt=prompt_extract_route(),
        template_class=RouteExtraction,
        extraction_name="route_extraction"
    )
    ticket_task = evaluator.extract(
        prompt=prompt_extract_lift_tickets(),
        template_class=LiftTicketExtraction,
        extraction_name="lift_ticket_extraction"
    )
    access_task = evaluator.extract(
        prompt=prompt_extract_accessibility(),
        template_class=AccessibilityExtraction,
        extraction_name="accessibility_extraction"
    )

    route_info, lift_info, access_info = await asyncio.gather(route_task, ticket_task, access_task)

    # Add ground truth reference info (for transparency)
    evaluator.add_ground_truth({
        "expected_route": {
            "departure_airport_code": "BWI",
            "connecting_airport_code": "MSP",
            "destination_airport_code": "YVR",
            "airline_name": "Sun Country Airlines"
        },
        "expected_standard_price": {
            "cad": "351 CAD",
            "usd": "258 USD"
        },
        "advance_purchase_policy": "Up to 30% off when purchasing 28+ days in advance"
    }, gt_type="expected_trip_criteria")

    # Build Trip Planning Evaluation node (critical root for rubric)
    trip_root = evaluator.add_parallel(
        id="Trip_Planning_Evaluation",
        desc="Evaluate the complete winter ski trip plan from Washington DC area to Whistler for February 2026",
        parent=root,
        critical=True
    )

    # Verify subtrees/nodes as per rubric
    await verify_route(evaluator, trip_root, route_info)
    await verify_lift_tickets(evaluator, trip_root, lift_info)
    await verify_accessibility(evaluator, trip_root, access_info)

    # Optional: record computed numbers for debugging
    evaluator.add_custom_info(
        info={
            "extracted_route": route_info.dict(),
            "extracted_lift_ticket": lift_info.dict(),
            "extracted_accessibility": access_info.dict()
        },
        info_type="extraction_echo",
        info_name="parsed_answer_snapshot"
    )

    return evaluator.get_summary()