import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "sdzoo_trip_plan_2026"
TASK_DESCRIPTION = (
    "A family of four (2 adults and 2 children ages 8 and 10) is planning a 1-day outdoor recreation trip to visit the San Diego Zoo on March 5, 2026. "
    "They will fly on Allegiant Air and must choose between departing from Nashville International Airport (BNA) or Charlotte Douglas International Airport (CLT). "
    "For their trip planning, they need to: (1) Select the departure airport that offers the lowest parking cost by comparing the long-term/economy parking rates at BNA and CLT (considering rates effective March 1, 2026 for CLT). "
    "(2) Calculate airport parking costs for a 2-day trip (parking from March 4 evening through March 6 morning, requiring 2 full days of parking). "
    "(3) Determine Allegiant Air baggage fees for the family, where each family member will bring one free personal item and the family will check exactly 2 bags, paying the advance booking rates. "
    "(4) Calculate San Diego Zoo admission costs using the 1-Day Pass Any Day ticket prices for 2 adults (ages 12+) and 2 children (ages 3-11). "
    "(5) Include San Diego Zoo parking fee for their vehicle at the standard vehicle rate (in effect as of January 5, 2026). "
    "(6) Calculate the total trip cost by summing all components: airport parking + baggage fees + zoo admission + zoo parking. "
    "(7) Verify budget compliance by confirming that the total cost does not exceed $600. "
    "Provide the complete cost breakdown with each component itemized, identify the selected airport, show all calculations, and include reference URLs supporting each cost figure."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ParkingRatesExtraction(BaseModel):
    bna_rate_per_day: Optional[str] = None
    clt_rate_per_day: Optional[str] = None
    clt_rate_effective_date: Optional[str] = None
    bna_rate_urls: List[str] = Field(default_factory=list)
    clt_rate_urls: List[str] = Field(default_factory=list)


class SelectedAirportExtraction(BaseModel):
    selected_airport_code: Optional[str] = None  # Expect "BNA" or "CLT"
    selection_reason: Optional[str] = None


class AirportParkingCostExtraction(BaseModel):
    parking_duration_days: Optional[int] = None
    parking_total_cost: Optional[str] = None  # keep as string (e.g., "$28")


class BaggageExtraction(BaseModel):
    personal_items_count: Optional[int] = None
    checked_bags_count: Optional[int] = None
    checked_bag_first_rate: Optional[str] = None  # e.g., "$35"
    checked_bag_second_rate: Optional[str] = None  # e.g., "$45"
    checked_bags_total_cost: Optional[str] = None  # e.g., "$80"
    baggage_urls: List[str] = Field(default_factory=list)


class ZooAdmissionExtraction(BaseModel):
    adult_ticket_price: Optional[str] = None  # "$78"
    adult_count: Optional[int] = None  # expect 2
    adult_total_cost: Optional[str] = None
    child_ticket_price: Optional[str] = None  # "$68"
    child_count: Optional[int] = None  # expect 2
    child_total_cost: Optional[str] = None
    admission_urls: List[str] = Field(default_factory=list)


class ZooParkingExtraction(BaseModel):
    zoo_parking_fee: Optional[str] = None  # "$16"
    zoo_parking_urls: List[str] = Field(default_factory=list)


class TotalCostExtraction(BaseModel):
    total_cost: Optional[str] = None  # e.g., "$400"


class TripPlanExtraction(BaseModel):
    parking_rates: Optional[ParkingRatesExtraction] = None
    selected_airport: Optional[SelectedAirportExtraction] = None
    parking_costs: Optional[AirportParkingCostExtraction] = None
    baggage: Optional[BaggageExtraction] = None
    zoo_admission: Optional[ZooAdmissionExtraction] = None
    zoo_parking: Optional[ZooParkingExtraction] = None
    total: Optional[TotalCostExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
Extract the trip planning details from the answer and return a JSON object with the following structure. Only extract information explicitly present in the answer. Use null for any missing fields. Include all URLs mentioned.

{
  "parking_rates": {
    "bna_rate_per_day": "string or null (e.g., \"$16/day\" or \"$16\")",
    "clt_rate_per_day": "string or null (e.g., \"$14/day\" or \"$14\")",
    "clt_rate_effective_date": "string or null (e.g., \"March 1, 2026\")",
    "bna_rate_urls": ["list of URLs that support BNA parking rate, as quoted in the answer"],
    "clt_rate_urls": ["list of URLs that support CLT long-term parking rate/effective date, as quoted in the answer"]
  },
  "selected_airport": {
    "selected_airport_code": "string or null (BNA or CLT)",
    "selection_reason": "string or null (the explanation given)"
  },
  "parking_costs": {
    "parking_duration_days": "integer or null (e.g., 2)",
    "parking_total_cost": "string or null (e.g., \"$32\")"
  },
  "baggage": {
    "personal_items_count": "integer or null (should be 4 if explicitly stated)",
    "checked_bags_count": "integer or null (should be 2 if explicitly stated)",
    "checked_bag_first_rate": "string or null (e.g., \"$35\")",
    "checked_bag_second_rate": "string or null (e.g., \"$45\")",
    "checked_bags_total_cost": "string or null (e.g., \"$80\")",
    "baggage_urls": ["list of Allegiant baggage fee URLs cited"]
  },
  "zoo_admission": {
    "adult_ticket_price": "string or null (e.g., \"$78\")",
    "adult_count": "integer or null (e.g., 2)",
    "adult_total_cost": "string or null (e.g., \"$156\")",
    "child_ticket_price": "string or null (e.g., \"$68\")",
    "child_count": "integer or null (e.g., 2)",
    "child_total_cost": "string or null (e.g., \"$136\")",
    "admission_urls": ["list of San Diego Zoo ticket URLs cited"]
  },
  "zoo_parking": {
    "zoo_parking_fee": "string or null (e.g., \"$16\")",
    "zoo_parking_urls": ["list of URLs cited for San Diego Zoo parking fee"]
  },
  "total": {
    "total_cost": "string or null (sum of all components as presented)"
  }
}
Guidelines:
- For money fields, extract the exact string the answer used (including currency symbol if present).
- For URLs, include all URLs explicitly present in the answer (including markdown links).
- Do not infer or calculate values here; just extract what the answer stated.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_money_to_float(maybe_money: Optional[str]) -> Optional[float]:
    if not maybe_money:
        return None
    s = maybe_money.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def normalize_airport_code(code: Optional[str]) -> Optional[str]:
    return code.strip().upper() if code else None


def str_includes_ci(s: Optional[str], needle: str) -> bool:
    if not s:
        return False
    return needle.lower() in s.lower()


def safe_int(v: Optional[int]) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_airport_selection(evaluator: Evaluator, parent_node, data: TripPlanExtraction) -> None:
    node = evaluator.add_sequential(
        id="Airport_Selection",
        desc="Validates that the departure airport is correctly selected based on parking cost comparison",
        parent=parent_node,
        critical=True,
    )

    # Sub-node: Parking_Rates_Identified -> split into concrete leaves
    pri = evaluator.add_parallel(
        id="Parking_Rates_Identified",
        desc="Correctly identifies both BNA Nashville airport's Economy Lots parking rate as $16 per day and CLT Charlotte airport's Long Term parking rate as $14 per day (effective March 1, 2026)",
        parent=node,
        critical=True,
    )

    bna_rate_val = parse_money_to_float(data.parking_rates.bna_rate_per_day if data.parking_rates else None)
    clt_rate_val = parse_money_to_float(data.parking_rates.clt_rate_per_day if data.parking_rates else None)
    clt_eff_date = (data.parking_rates.clt_rate_effective_date if data.parking_rates else None) or ""

    evaluator.add_custom_node(
        result=(bna_rate_val == 16.0),
        id="BNA_Rate_Stated_Correct",
        desc="The answer states BNA economy/long-term daily parking rate as $16 per day",
        parent=pri,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(clt_rate_val == 14.0),
        id="CLT_Rate_Stated_Correct",
        desc="The answer states CLT long-term daily parking rate as $14 per day",
        parent=pri,
        critical=True,
    )

    evaluator.add_custom_node(
        result=str_includes_ci(clt_eff_date, "march 1, 2026"),
        id="CLT_Rate_EffectiveDate_Stated",
        desc="The answer states the CLT rate effective date as March 1, 2026",
        parent=pri,
        critical=True,
    )

    # Leaf: Cheaper_Airport_Selected
    cheaper_leaf = evaluator.add_custom_node(
        result=(lambda sel, bna, clt: (normalize_airport_code(sel) == ("CLT" if (bna is not None and clt is not None and clt < bna) else ("BNA" if (bna is not None and clt is not None and bna < clt) else normalize_airport_code(sel)))))
        (data.selected_airport.selected_airport_code if data.selected_airport else None, bna_rate_val, clt_rate_val),
        id="Cheaper_Airport_Selected",
        desc="Correctly identifies and selects the departure airport with the lower parking cost based on the comparison of rates",
        parent=node,
        critical=True,
    )

    # Sub-node: Airport_Reference -> verify URLs support stated rates
    ar = evaluator.add_parallel(
        id="Airport_Reference",
        desc="Provides valid reference URL(s) supporting the airport parking rate information",
        parent=node,
        critical=True,
    )

    # BNA source existence
    bna_src_exist = evaluator.add_custom_node(
        result=bool(data.parking_rates and data.parking_rates.bna_rate_urls),
        id="BNA_Parking_Source_Provided",
        desc="BNA parking source URL(s) provided",
        parent=ar,
        critical=True,
    )
    # BNA source supports $16/day
    bna_support_leaf = evaluator.add_leaf(
        id="BNA_Parking_Rate_Supported",
        desc="BNA economy/long-term daily parking rate is $16 per day (supported by source)",
        parent=ar,
        critical=True,
    )
    await evaluator.verify(
        claim="The daily rate for long-term/economy parking at Nashville International Airport (BNA) is $16 per day.",
        node=bna_support_leaf,
        sources=(data.parking_rates.bna_rate_urls if data.parking_rates else []),
        additional_instruction="Verify the BNA long-term/economy daily parking rate on the official airport parking page.",
        extra_prerequisites=[bna_src_exist],
    )

    # CLT source existence
    clt_src_exist = evaluator.add_custom_node(
        result=bool(data.parking_rates and data.parking_rates.clt_rate_urls),
        id="CLT_Parking_Source_Provided",
        desc="CLT parking source URL(s) provided",
        parent=ar,
        critical=True,
    )
    # CLT source supports $14/day effective Mar 1, 2026
    clt_support_leaf = evaluator.add_leaf(
        id="CLT_Parking_Rate_Supported",
        desc="CLT long-term daily parking rate is $14 per day, effective March 1, 2026 (supported by source)",
        parent=ar,
        critical=True,
    )
    await evaluator.verify(
        claim="At Charlotte Douglas International Airport (CLT), the long-term parking daily rate is $14 per day, effective March 1, 2026.",
        node=clt_support_leaf,
        sources=(data.parking_rates.clt_rate_urls if data.parking_rates else []),
        additional_instruction="Verify CLT official parking page(s) for long-term parking daily rate and the effective date March 1, 2026.",
        extra_prerequisites=[clt_src_exist],
    )


async def verify_airport_parking_cost(evaluator: Evaluator, parent_node, data: TripPlanExtraction) -> None:
    node = evaluator.add_sequential(
        id="Airport_Parking_Cost",
        desc="Validates the correct calculation of airport parking costs for the trip duration",
        parent=parent_node,
        critical=True,
    )

    # Parking duration: should be 2 days
    duration_ok = evaluator.add_custom_node(
        result=(safe_int(data.parking_costs.parking_duration_days if data.parking_costs else None) == 2),
        id="Parking_Duration",
        desc="Correctly calculates parking duration as 2 full days based on trip dates (March 4-6, 2026)",
        parent=node,
        critical=True,
    )

    # Parking calculation: total = selected airport daily rate * 2
    selected = normalize_airport_code(data.selected_airport.selected_airport_code if data.selected_airport else None)
    bna_rate_val = parse_money_to_float(data.parking_rates.bna_rate_per_day if data.parking_rates else None)
    clt_rate_val = parse_money_to_float(data.parking_rates.clt_rate_per_day if data.parking_rates else None)
    selected_rate = bna_rate_val if selected == "BNA" else (clt_rate_val if selected == "CLT" else None)
    expected_total = (selected_rate * 2) if (selected_rate is not None and safe_int(data.parking_costs.parking_duration_days if data.parking_costs else None) == 2) else None
    actual_total = parse_money_to_float(data.parking_costs.parking_total_cost if data.parking_costs else None)

    evaluator.add_custom_node(
        result=(expected_total is not None and actual_total is not None and abs(actual_total - expected_total) < 0.01),
        id="Parking_Calculation",
        desc="Correctly calculates total parking cost by multiplying the selected airport's daily rate by the number of days",
        parent=node,
        critical=True,
    )

    # Parking reference: verify selected airport rate supported again
    pr = evaluator.add_parallel(
        id="Parking_Reference",
        desc="Provides valid reference URL supporting the selected airport's parking rate",
        parent=node,
        critical=True,
    )

    if selected == "BNA":
        src_exist = evaluator.add_custom_node(
            result=bool(data.parking_rates and data.parking_rates.bna_rate_urls),
            id="Selected_Parking_Source_Provided",
            desc="Selected airport parking source URL(s) provided",
            parent=pr,
            critical=True,
        )
        leaf = evaluator.add_leaf(
            id="Selected_Parking_Rate_Supported",
            desc="Selected airport daily parking rate is supported by the provided URL(s)",
            parent=pr,
            critical=True,
        )
        await evaluator.verify(
            claim="The daily rate for long-term/economy parking at Nashville International Airport (BNA) is $16 per day.",
            node=leaf,
            sources=(data.parking_rates.bna_rate_urls if data.parking_rates else []),
            additional_instruction="Verify the BNA long-term/economy daily parking rate on the official airport parking page.",
            extra_prerequisites=[src_exist, duration_ok],
        )
    elif selected == "CLT":
        src_exist = evaluator.add_custom_node(
            result=bool(data.parking_rates and data.parking_rates.clt_rate_urls),
            id="Selected_Parking_Source_Provided",
            desc="Selected airport parking source URL(s) provided",
            parent=pr,
            critical=True,
        )
        leaf = evaluator.add_leaf(
            id="Selected_Parking_Rate_Supported",
            desc="Selected airport daily parking rate is supported by the provided URL(s)",
            parent=pr,
            critical=True,
        )
        await evaluator.verify(
            claim="At Charlotte Douglas International Airport (CLT), the long-term parking daily rate is $14 per day, effective March 1, 2026.",
            node=leaf,
            sources=(data.parking_rates.clt_rate_urls if data.parking_rates else []),
            additional_instruction="Verify CLT official parking page(s) for long-term parking daily rate and the effective date March 1, 2026.",
            extra_prerequisites=[src_exist, duration_ok],
        )
    else:
        # If no selected airport extracted, still add a failing existence check to reflect missing info
        evaluator.add_custom_node(
            result=False,
            id="Selected_Parking_Source_Provided",
            desc="Selected airport parking source URL(s) provided",
            parent=pr,
            critical=True,
        )


async def verify_baggage_fees(evaluator: Evaluator, parent_node, data: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Baggage_Fees",
        desc="Validates correct calculation of Allegiant Air baggage fees for the family",
        parent=parent_node,
        critical=True,
    )

    # Personal items: 4 free
    evaluator.add_custom_node(
        result=(safe_int(data.baggage.personal_items_count if data.baggage else None) == 4),
        id="Personal_Items",
        desc="Correctly accounts for 4 free personal items (one per family member)",
        parent=node,
        critical=True,
    )

    # Checked bags count: 2
    evaluator.add_custom_node(
        result=(safe_int(data.baggage.checked_bags_count if data.baggage else None) == 2),
        id="Checked_Bags_Count",
        desc="Specifies that the family will check exactly 2 bags total",
        parent=node,
        critical=True,  # Adjusted to True to satisfy critical parent constraint
    )

    # Checked bags cost -> split into two concrete leaves under a parallel aggregator
    cbc = evaluator.add_parallel(
        id="Checked_Bags_Cost",
        desc="Correctly calculates total checked bag fees using Allegiant's advance booking rates: $35 for the first checked bag and $45 for the second checked bag",
        parent=node,
        critical=True,
    )

    # Source existence
    baggage_src_exist = evaluator.add_custom_node(
        result=bool(data.baggage and data.baggage.baggage_urls),
        id="Baggage_Source_Provided",
        desc="Baggage fee source URL(s) provided",
        parent=cbc,
        critical=True,
    )

    # Source supports $35 first, $45 second (advance booking)
    rates_supported_leaf = evaluator.add_leaf(
        id="Baggage_Rates_Supported",
        desc="Allegiant advance booking checked baggage fees are $35 (first bag) and $45 (second bag)",
        parent=cbc,
        critical=True,
    )
    await evaluator.verify(
        claim="Allegiant Air's advance purchase checked baggage fees are $35 for the first checked bag and $45 for the second checked bag.",
        node=rates_supported_leaf,
        sources=(data.baggage.baggage_urls if data.baggage else []),
        additional_instruction="Verify Allegiant official baggage fee info or a current fee table that explicitly lists advance booking rates.",
        extra_prerequisites=[baggage_src_exist],
    )

    # Calculation correct: total = first + second
    first_rate = parse_money_to_float(data.baggage.checked_bag_first_rate if data.baggage else None)
    second_rate = parse_money_to_float(data.baggage.checked_bag_second_rate if data.baggage else None)
    cb_total = parse_money_to_float(data.baggage.checked_bags_total_cost if data.baggage else None)
    expected_cb_total = (first_rate + second_rate) if (first_rate is not None and second_rate is not None) else None

    evaluator.add_custom_node(
        result=(expected_cb_total is not None and cb_total is not None and abs(cb_total - expected_cb_total) < 0.01),
        id="Baggage_Total_Calculated_Correctly",
        desc="Checked bag total equals the sum of the first and second bag advance rates",
        parent=cbc,
        critical=True,
    )

    # General baggage reference node (valid reference URL supporting baggage fee info)
    br = evaluator.add_leaf(
        id="Baggage_Reference",
        desc="Provides valid reference URL supporting Allegiant baggage fee information",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page provides Allegiant baggage fee information, including checked bag pricing.",
        node=br,
        sources=(data.baggage.baggage_urls if data.baggage else []),
        additional_instruction="Confirm the page is relevant to Allegiant Airlines baggage fees and lists pricing details.",
    )


async def verify_zoo_admission(evaluator: Evaluator, parent_node, data: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Zoo_Admission",
        desc="Validates correct calculation of San Diego Zoo admission costs",
        parent=parent_node,
        critical=True,
    )

    # Adult Admission aggregator
    aa = evaluator.add_parallel(
        id="Adult_Admission",
        desc="Correctly calculates cost for 2 adult tickets at $78 each",
        parent=node,
        critical=True,
    )

    # Verify price supported by sources
    adm_src_exist = evaluator.add_custom_node(
        result=bool(data.zoo_admission and data.zoo_admission.admission_urls),
        id="Admission_Source_Provided",
        desc="Admission price source URL(s) provided",
        parent=aa,
        critical=True,
    )
    adult_price_supported = evaluator.add_leaf(
        id="Adult_Price_Supported",
        desc="San Diego Zoo 1-Day Pass Any Day adult ticket price is $78",
        parent=aa,
        critical=True,
    )
    await evaluator.verify(
        claim="The San Diego Zoo 1-Day Pass Any Day adult ticket price is $78.",
        node=adult_price_supported,
        sources=(data.zoo_admission.admission_urls if data.zoo_admission else []),
        additional_instruction="Confirm the '1-Day Pass Any Day' adult price is $78 on the official San Diego Zoo ticketing page.",
        extra_prerequisites=[adm_src_exist],
    )

    # Verify calculation 2 * 78
    adult_count = safe_int(data.zoo_admission.adult_count if data.zoo_admission else None)
    adult_price_val = parse_money_to_float(data.zoo_admission.adult_ticket_price if data.zoo_admission else None)
    adult_total_val = parse_money_to_float(data.zoo_admission.adult_total_cost if data.zoo_admission else None)
    expected_adult_total = (adult_count * adult_price_val) if (adult_count is not None and adult_price_val is not None) else None

    evaluator.add_custom_node(
        result=(adult_count == 2 and adult_price_val == 78.0 and expected_adult_total is not None and adult_total_val is not None and abs(adult_total_val - expected_adult_total) < 0.01),
        id="Adult_Total_Calculated_Correctly",
        desc="Adult total equals 2 × $78 and matches the stated amount",
        parent=aa,
        critical=True,
    )

    # Child Admission aggregator
    ca = evaluator.add_parallel(
        id="Child_Admission",
        desc="Correctly calculates cost for 2 child tickets (ages 8 and 10, both ages 3-11) at $68 each",
        parent=node,
        critical=True,
    )

    child_price_supported = evaluator.add_leaf(
        id="Child_Price_Supported",
        desc="San Diego Zoo 1-Day Pass Any Day child ticket price is $68",
        parent=ca,
        critical=True,
    )
    await evaluator.verify(
        claim="The San Diego Zoo 1-Day Pass Any Day child ticket price (ages 3–11) is $68.",
        node=child_price_supported,
        sources=(data.zoo_admission.admission_urls if data.zoo_admission else []),
        additional_instruction="Confirm the '1-Day Pass Any Day' child price is $68 on the official San Diego Zoo ticketing page.",
        extra_prerequisites=[adm_src_exist],
    )

    child_count = safe_int(data.zoo_admission.child_count if data.zoo_admission else None)
    child_price_val = parse_money_to_float(data.zoo_admission.child_ticket_price if data.zoo_admission else None)
    child_total_val = parse_money_to_float(data.zoo_admission.child_total_cost if data.zoo_admission else None)
    expected_child_total = (child_count * child_price_val) if (child_count is not None and child_price_val is not None) else None

    evaluator.add_custom_node(
        result=(child_count == 2 and child_price_val == 68.0 and expected_child_total is not None and child_total_val is not None and abs(child_total_val - expected_child_total) < 0.01),
        id="Child_Total_Calculated_Correctly",
        desc="Child total equals 2 × $68 and matches the stated amount",
        parent=ca,
        critical=True,
    )

    # Admission Reference (redundant but per rubric)
    adm_ref_leaf = evaluator.add_leaf(
        id="Admission_Reference",
        desc="Provides valid reference URL supporting San Diego Zoo admission prices",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page lists San Diego Zoo 1-Day Pass Any Day prices for adults ($78) and children ($68).",
        node=adm_ref_leaf,
        sources=(data.zoo_admission.admission_urls if data.zoo_admission else []),
        additional_instruction="Confirm that the ticketing/pricing page clearly shows the Any Day prices for adult and child.",
    )


async def verify_zoo_parking(evaluator: Evaluator, parent_node, data: TripPlanExtraction) -> None:
    node = evaluator.add_sequential(
        id="Zoo_Parking",
        desc="Validates that San Diego Zoo parking fee is included in the total cost",
        parent=parent_node,
        critical=True,
    )

    # Include $16 per vehicle
    zoo_parking_val = parse_money_to_float(data.zoo_parking.zoo_parking_fee if data.zoo_parking else None)
    evaluator.add_custom_node(
        result=(zoo_parking_val == 16.0),
        id="Parking_Fee_Included",
        desc="Includes the $16 per vehicle parking fee at San Diego Zoo (standard vehicle rate effective January 5, 2026)",
        parent=node,
        critical=True,
    )

    # Reference
    pr = evaluator.add_parallel(
        id="Zoo_Parking_Reference",
        desc="Provides valid reference URL supporting San Diego Zoo parking fee information",
        parent=node,
        critical=True,
    )

    zp_src_exist = evaluator.add_custom_node(
        result=bool(data.zoo_parking and data.zoo_parking.zoo_parking_urls),
        id="Zoo_Parking_Source_Provided",
        desc="Zoo parking fee source URL(s) provided",
        parent=pr,
        critical=True,
    )

    zp_leaf = evaluator.add_leaf(
        id="Zoo_Parking_Fee_Supported",
        desc="San Diego Zoo standard vehicle parking fee is $16 (supported by source)",
        parent=pr,
        critical=True,
    )
    await evaluator.verify(
        claim="The San Diego Zoo standard vehicle parking fee is $16.",
        node=zp_leaf,
        sources=(data.zoo_parking.zoo_parking_urls if data.zoo_parking else []),
        additional_instruction="Confirm the official parking fee information (effective January 5, 2026 if shown).",
        extra_prerequisites=[zp_src_exist],
    )


async def verify_total_budget(evaluator: Evaluator, parent_node, data: TripPlanExtraction) -> None:
    node = evaluator.add_sequential(
        id="Total_Budget",
        desc="Validates that all costs are correctly summed and the total stays within budget",
        parent=parent_node,
        critical=True,
    )

    # Compute expected totals from extracted components
    parking_total = parse_money_to_float(data.parking_costs.parking_total_cost if data.parking_costs else None)
    baggage_total = parse_money_to_float(data.baggage.checked_bags_total_cost if data.baggage else None)
    adult_total = parse_money_to_float(data.zoo_admission.adult_total_cost if data.zoo_admission else None)
    child_total = parse_money_to_float(data.zoo_admission.child_total_cost if data.zoo_admission else None)
    zoo_parking_fee = parse_money_to_float(data.zoo_parking.zoo_parking_fee if data.zoo_parking else None)
    declared_total = parse_money_to_float(data.total.total_cost if data.total else None)

    components_available = [x is not None for x in [parking_total, baggage_total, adult_total, child_total, zoo_parking_fee]]
    expected_sum = None
    if all(components_available):
        expected_sum = parking_total + baggage_total + adult_total + child_total + zoo_parking_fee

    evaluator.add_custom_node(
        result=(expected_sum is not None and declared_total is not None and abs(declared_total - expected_sum) < 0.01),
        id="All_Costs_Summed",
        desc="Correctly sums all cost components: airport parking + baggage fees + zoo admission + zoo parking",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(declared_total is not None and declared_total <= 600.0),
        id="Budget_Compliance",
        desc="Verifies that the calculated total trip cost does not exceed the $600 budget limit",
        parent=node,
        critical=True,
    )

    # Record computed info for transparency
    evaluator.add_custom_info(
        info={
            "computed_expected_sum": expected_sum,
            "declared_total": declared_total,
            "components": {
                "airport_parking_total": parking_total,
                "baggage_total": baggage_total,
                "zoo_adult_total": adult_total,
                "zoo_child_total": child_total,
                "zoo_parking_fee": zoo_parking_fee
            }
        },
        info_type="computation",
        info_name="budget_computation_details"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
        strategy=AggregationStrategy.PARALLEL,  # root aggregator
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
    extracted: TripPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Add ground truth expectations (per rubric) for transparency
    evaluator.add_ground_truth({
        "expected_bna_rate_per_day": "$16",
        "expected_clt_rate_per_day": "$14",
        "clt_rate_effective_date": "March 1, 2026",
        "expected_allegiant_checked_bag_rates_advance": {"first": "$35", "second": "$45"},
        "expected_zoo_adult_anyday": "$78",
        "expected_zoo_child_anyday": "$68",
        "expected_zoo_parking_fee": "$16",
        "parking_duration_days": 2,
        "budget_cap": "$600"
    }, gt_type="rubric_expectations")

    # Build Trip Planning critical aggregator
    trip_node = evaluator.add_parallel(
        id="Trip_Planning",
        desc="Validates the complete planning of a family outdoor recreation trip to San Diego Zoo, including airport selection, costs calculation, and budget compliance",
        parent=root,
        critical=True,
    )

    # Subtasks
    await verify_airport_selection(evaluator, trip_node, extracted)
    await verify_airport_parking_cost(evaluator, trip_node, extracted)
    await verify_baggage_fees(evaluator, trip_node, extracted)
    await verify_zoo_admission(evaluator, trip_node, extracted)
    await verify_zoo_parking(evaluator, trip_node, extracted)
    await verify_total_budget(evaluator, trip_node, extracted)

    return evaluator.get_summary()