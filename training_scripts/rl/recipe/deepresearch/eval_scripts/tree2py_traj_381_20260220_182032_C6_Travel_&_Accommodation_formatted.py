import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vacation_comparison_2026"
TASK_DESCRIPTION = """A family of four (2 adults and 2 children aged 8 and 11) living in Madison, Wisconsin is planning a 4-day summer vacation in 2026 and considering two destination options: (1) Wisconsin Dells or (2) Epic Universe at Universal Orlando Resort in Florida.

For each destination option, research and provide a detailed vacation budget comparison that includes:

For Wisconsin Dells:
- Hotel accommodation for 3 nights at a water park resort (specify hotel name, nightly rate, and whether water park admission is included)
- Water park admission costs (if not included with hotel)
- Ground transportation costs from Madison airport (MSN) to Wisconsin Dells
- Estimated meals and miscellaneous expenses

For Epic Universe/Universal Orlando:
- Roundtrip flights for the family from Madison (MSN) to Orlando (MCO), identifying at least two airline options and baggage fee policies
- Hotel accommodation for 3 nights (specify hotel name and nightly rate, note any special benefits like Early Park Admission for on-site hotels)
- Theme park tickets for the family (specify whether single-day Epic Universe only or multi-day park-to-park tickets)
- Ground transportation costs from Orlando airport to the hotel/Epic Universe area
- Estimated meals and miscellaneous expenses

Comparison Requirements:
- Provide itemized budget breakdowns for both destinations showing all cost components
- Calculate the total estimated cost for each vacation option
- State which destination is more affordable and by how much
- Identify at least two key differences or tradeoffs between the two options (such as travel time, variety of attractions, hotel amenities, etc.)

All pricing information must include reference URLs from your sources. Assume the travel dates are flexible within June-August 2026 to find reasonable pricing.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PricingScope(BaseModel):
    date_window_text: Optional[str] = None  # e.g., "late July 2026" or "June–August 2026"


class HotelInfoWD(BaseModel):
    name: Optional[str] = None
    nightly_rate: Optional[str] = None  # Prefer string to allow ranges/approx
    waterpark_included: Optional[str] = None  # e.g., "included", "not included", "partially included"
    source_urls: List[str] = Field(default_factory=list)


class AdmissionInfoWD(BaseModel):
    admission_cost: Optional[str] = None  # e.g., "$49 per person", "included"
    included_flag: Optional[str] = None  # explicit inclusion note if present
    source_urls: List[str] = Field(default_factory=list)


class TransportInfo(BaseModel):
    cost_estimate: Optional[str] = None  # e.g., "$250 rental car for 4 days", "$120 shuttle"
    source_urls: List[str] = Field(default_factory=list)


class MealsMiscInfo(BaseModel):
    total_estimate: Optional[str] = None  # e.g., "$400 for 4 days"
    source_urls: List[str] = Field(default_factory=list)


class ItemizedBudget(BaseModel):
    items: List[str] = Field(default_factory=list)  # Human-readable items like ["Hotel: $600", "Tickets: $200"]
    total_cost: Optional[str] = None  # e.g., "$1,550"


class WisconsinDellsBudget(BaseModel):
    hotel: Optional[HotelInfoWD] = None
    admission: Optional[AdmissionInfoWD] = None
    transport: Optional[TransportInfo] = None
    meals_misc: Optional[MealsMiscInfo] = None
    itemized: Optional[ItemizedBudget] = None


class AirlineOption(BaseModel):
    airline: Optional[str] = None
    baggage_policy: Optional[str] = None  # summarized baggage policy text (carry-on/checked)
    source_urls: List[str] = Field(default_factory=list)


class FlightInfoEU(BaseModel):
    total_cost_estimate: Optional[str] = None  # Family of 4 roundtrip total
    airline_options: List[AirlineOption] = Field(default_factory=list)
    price_source_urls: List[str] = Field(default_factory=list)  # Sources used for price estimate


class HotelInfoEU(BaseModel):
    name: Optional[str] = None
    nightly_rate: Optional[str] = None
    benefits_note: Optional[str] = None  # e.g., "Early Park Admission included"
    source_urls: List[str] = Field(default_factory=list)


class TicketInfoEU(BaseModel):
    ticket_type: Optional[str] = None  # e.g., "single-day Epic Universe", "2-day park-to-park"
    total_cost: Optional[str] = None  # Family of 4 total
    source_urls: List[str] = Field(default_factory=list)


class EpicUniverseBudget(BaseModel):
    flights: Optional[FlightInfoEU] = None
    hotel: Optional[HotelInfoEU] = None
    tickets: Optional[TicketInfoEU] = None
    transport: Optional[TransportInfo] = None
    meals_misc: Optional[MealsMiscInfo] = None
    itemized: Optional[ItemizedBudget] = None


class ComparisonInfo(BaseModel):
    cheaper_option: Optional[str] = None  # e.g., "Wisconsin Dells" or "Epic Universe/Universal Orlando"
    difference_amount: Optional[str] = None  # e.g., "$350"
    tradeoffs: List[str] = Field(default_factory=list)  # at least two differences/tradeoffs


class VacationComparisonExtraction(BaseModel):
    scope: Optional[PricingScope] = None
    wisconsin_dells: Optional[WisconsinDellsBudget] = None
    epic_universe: Optional[EpicUniverseBudget] = None
    comparison: Optional[ComparisonInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vacation_comparison() -> str:
    return """
    Extract structured information from the answer for a two-option vacation budget comparison (Wisconsin Dells vs. Epic Universe/Universal Orlando). Only extract details explicitly mentioned in the answer. Use null if a field is missing. Collect all cited URLs for each relevant part as lists.

    Required structure:
    {
      "scope": {
        "date_window_text": string | null
      },
      "wisconsin_dells": {
        "hotel": {
          "name": string | null,
          "nightly_rate": string | null,
          "waterpark_included": string | null,  // e.g., "included", "not included", "partially included"
          "source_urls": string[]               // hotel rate/amenity source URLs
        },
        "admission": {
          "admission_cost": string | null,      // if tickets needed
          "included_flag": string | null,       // if explicitly stated "included with hotel"
          "source_urls": string[]               // admission pricing source URLs
        },
        "transport": {
          "cost_estimate": string | null,       // MSN → Wisconsin Dells ground transport estimate
          "source_urls": string[]
        },
        "meals_misc": {
          "total_estimate": string | null,
          "source_urls": string[]               // basis for meals/misc estimate (per-diem or typical costs)
        },
        "itemized": {
          "items": string[],                    // itemized lines like "Hotel: $600", "Tickets: $200"
          "total_cost": string | null
        }
      },
      "epic_universe": {
        "flights": {
          "total_cost_estimate": string | null, // family of 4 roundtrip total
          "airline_options": [
            {
              "airline": string | null,
              "baggage_policy": string | null,  // carry-on & checked-bag allowances/fees summary
              "source_urls": string[]
            }
          ],
          "price_source_urls": string[]         // flight pricing source URLs for the estimate
        },
        "hotel": {
          "name": string | null,
          "nightly_rate": string | null,
          "benefits_note": string | null,       // e.g., Early Park Admission
          "source_urls": string[]
        },
        "tickets": {
          "ticket_type": string | null,         // "single-day Epic Universe" or "multi-day park-to-park"
          "total_cost": string | null,          // family of 4 total
          "source_urls": string[]
        },
        "transport": {
          "cost_estimate": string | null,       // MCO → hotel/Epic Universe area estimate
          "source_urls": string[]
        },
        "meals_misc": {
          "total_estimate": string | null,
          "source_urls": string[]
        },
        "itemized": {
          "items": string[],
          "total_cost": string | null
        }
      },
      "comparison": {
        "cheaper_option": string | null,
        "difference_amount": string | null,
        "tradeoffs": string[]                   // at least two differences/tradeoffs
      }
    }

    Special rules:
    - Extract only URLs actually present in the answer. If an element references a site but no URL is provided, leave its source_urls as an empty list.
    - When nightly rates or costs are ranges/estimates, extract the text as-is (e.g., "$180–$220/night").
    - If waterpark admission is included with the hotel, mark "waterpark_included" or "included_flag" accordingly and set "admission_cost" to null.
    - For flights, include at least two airline options if the answer presents two. Extract baggage policy summaries and URLs for each airline where cited.
    - For itemized budgets, list the items exactly as written in the answer and the total as text (e.g., "$2,150 total").
    - If the answer states a specific date window within June–August 2026 (e.g., "mid-July 2026"), record it as date_window_text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and len(u.strip()) > 0 for u in urls)


def combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if lst:
            for u in lst:
                if isinstance(u, str) and len(u.strip()) > 0:
                    combined.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in combined:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def interpret_included(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    if any(k in t for k in ["included", "includes", "with stay", "admission included", "waterpark included", "complimentary access"]):
        if any(k in t for k in ["not included", "no", "excluded", "separate"]):
            # contradictory; return None to force URL verification
            return None
        return True
    if any(k in t for k in ["not included", "no", "excluded", "separate ticket", "purchase tickets"]):
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_scope(evaluator: Evaluator, parent_node, scope: Optional[PricingScope]) -> None:
    scope_node = evaluator.add_leaf(
        id="Summer_2026_Pricing_Scope",
        desc="All priced components are scoped to travel dates within June–August 2026 (or the answer clearly states the specific date window within that range used to obtain pricing).",
        parent=parent_node,
        critical=True,
    )
    if scope and scope.date_window_text:
        claim = f"The answer explicitly scoping/pricing uses the date window '{scope.date_window_text}', and it falls within June–August 2026."
    else:
        claim = "The priced components in the answer are explicitly scoped to travel dates within June–August 2026 or clearly within that range."
    await evaluator.verify(
        claim=claim,
        node=scope_node,
        additional_instruction="Check the answer text to confirm the pricing date window is within June–August 2026, or that a clearly stated specific window within that range is used."
    )


async def verify_wisconsin_dells(evaluator: Evaluator, parent_node, wd: Optional[WisconsinDellsBudget]) -> None:
    wd_node = evaluator.add_parallel(
        id="Wisconsin_Dells_Budget",
        desc="Provide a complete, sourced, itemized 3-night Wisconsin Dells budget.",
        parent=parent_node,
        critical=True
    )

    # Hotel Accommodation group
    hotel_group = evaluator.add_parallel(
        id="WD_Hotel_Accommodation",
        desc="Hotel accommodation for 3 nights at a water park resort, including required details and a source URL.",
        parent=wd_node,
        critical=True
    )

    hotel = wd.hotel if wd else None
    hotel_urls = hotel.source_urls if hotel else []

    # WD_Hotel_Name
    node_hotel_name = evaluator.add_leaf(
        id="WD_Hotel_Name",
        desc="Specify the water park resort hotel name.",
        parent=hotel_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The selected Wisconsin Dells hotel is '{hotel.name if hotel and hotel.name else ''}'.",
        node=node_hotel_name,
        sources=hotel_urls if has_valid_urls(hotel_urls) else None,
        additional_instruction="Verify that the hotel's name matches the cited hotel page or related official listing when URLs are provided."
    )

    # WD_Hotel_Nightly_Rate
    node_hotel_rate = evaluator.add_leaf(
        id="WD_Hotel_Nightly_Rate",
        desc="Provide the nightly rate (or a clearly derived 3-night lodging cost) for the selected dates/window.",
        parent=hotel_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The nightly rate (or derived 3-night lodging cost) for the selected hotel is '{hotel.nightly_rate if hotel and hotel.nightly_rate else ''}'.",
        node=node_hotel_rate,
        sources=hotel_urls if has_valid_urls(hotel_urls) else None,
        additional_instruction="Confirm from the hotel's rate/booking page. Allow reasonable June–August 2026 flexibility and approximate pricing if clearly stated."
    )

    # WD_Waterpark_Included_Flag
    node_wp_included = evaluator.add_leaf(
        id="WD_Waterpark_Included_Flag",
        desc="State whether water park admission is included with the hotel stay.",
        parent=hotel_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Water park admission included with hotel: '{hotel.waterpark_included if hotel and hotel.waterpark_included else ''}'.",
        node=node_wp_included,
        sources=hotel_urls if has_valid_urls(hotel_urls) else None,
        additional_instruction="Verify the hotel's own policy page or resort information that explicitly states whether waterpark admission is included or not."
    )

    # WD_Hotel_Source_URL (existence check)
    evaluator.add_custom_node(
        result=has_valid_urls(hotel_urls),
        id="WD_Hotel_Source_URL",
        desc="Provide a reference URL supporting the hotel rate (and included-admission policy if claimed).",
        parent=hotel_group,
        critical=True
    )

    # WD_Waterpark_Admission_Costs_Conditional
    admission = wd.admission if wd else None
    admission_urls = admission.source_urls if admission else []
    included_bool = interpret_included(admission.included_flag if admission else None)
    node_admission_cond = evaluator.add_leaf(
        id="WD_Waterpark_Admission_Costs_Conditional",
        desc="If water park admission is not included with the hotel, provide admission pricing and a source URL. If included, explicitly mark it as included/no separate ticket purchase.",
        parent=wd_node,
        critical=True
    )
    if included_bool is True or interpret_included(hotel.waterpark_included if hotel else None) is True:
        claim_adm = "Water park admission is included with the hotel stay; no separate ticket purchase is required."
        adm_sources = combine_urls(hotel_urls, admission_urls)
    else:
        claim_adm = f"Water park admission pricing is '{admission.admission_cost if admission and admission.admission_cost else ''}' for the relevant visit."
        adm_sources = admission_urls
    await evaluator.verify(
        claim=claim_adm,
        node=node_admission_cond,
        sources=adm_sources if has_valid_urls(adm_sources) else (hotel_urls if has_valid_urls(hotel_urls) else None),
        additional_instruction="Confirm via resort/waterpark official ticketing or hotel policy pages whether admission is included or the stated pricing applies."
    )

    # WD_Ground_Transportation_Cost group
    transport = wd.transport if wd else None
    transport_urls = transport.source_urls if transport else []
    transport_group = evaluator.add_parallel(
        id="WD_Ground_Transportation_Cost",
        desc="Provide ground transportation cost from MSN to Wisconsin Dells with a reference URL.",
        parent=wd_node,
        critical=True
    )
    node_trans_cost = evaluator.add_leaf(
        id="WD_Transport_Cost_Estimate",
        desc="Provide an estimated transportation cost (e.g., rental car/shuttle/taxi) for the trip.",
        parent=transport_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The estimated ground transportation cost from Madison airport (MSN) to Wisconsin Dells is '{transport.cost_estimate if transport and transport.cost_estimate else ''}'.",
        node=node_trans_cost,
        sources=transport_urls if has_valid_urls(transport_urls) else None,
        additional_instruction="Check the cited transportation source (rental car aggregator, shuttle/taxi estimate, etc.) supporting the stated estimate."
    )
    evaluator.add_custom_node(
        result=has_valid_urls(transport_urls),
        id="WD_Transport_Source_URL",
        desc="Provide a reference URL supporting the transportation pricing (or a sourced basis used to compute the estimate).",
        parent=transport_group,
        critical=True
    )

    # WD_Meals_and_Misc group
    meals = wd.meals_misc if wd else None
    meals_urls = meals.source_urls if meals else []
    meals_group = evaluator.add_parallel(
        id="WD_Meals_and_Misc",
        desc="Provide an estimated meals and miscellaneous cost component with at least one reference URL supporting the basis for the estimate (e.g., per-diem guidance, typical meal costs, or similar).",
        parent=wd_node,
        critical=True
    )
    node_meals_cost = evaluator.add_leaf(
        id="WD_Meals_Misc_Cost_Estimate",
        desc="State the estimated meals and miscellaneous total for the trip.",
        parent=meals_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The estimated meals and miscellaneous total for the family is '{meals.total_estimate if meals and meals.total_estimate else ''}'.",
        node=node_meals_cost,
        sources=meals_urls if has_valid_urls(meals_urls) else None,
        additional_instruction="Verify per-diem or typical meal cost basis from the cited sources (family of four, 4-day trip)."
    )
    evaluator.add_custom_node(
        result=has_valid_urls(meals_urls),
        id="WD_Meals_Misc_Source_URL",
        desc="Provide a reference URL supporting the basis for the meals/misc estimate.",
        parent=meals_group,
        critical=True
    )

    # WD_Itemized_Budget_and_Total
    node_itemized = evaluator.add_leaf(
        id="WD_Itemized_Budget_and_Total",
        desc="Provide an itemized breakdown showing all cost components and a total estimated cost for Wisconsin Dells.",
        parent=wd_node,
        critical=True
    )
    items_preview = "; ".join(wd.itemized.items) if wd and wd.itemized and wd.itemized.items else ""
    claim_itemized = f"The answer contains an itemized breakdown for Wisconsin Dells (e.g., {items_preview}) and a total estimated cost '{wd.itemized.total_cost if wd and wd.itemized and wd.itemized.total_cost else ''}'."
    await evaluator.verify(
        claim=claim_itemized,
        node=node_itemized,
        additional_instruction="Check the answer text for an itemized list of WD costs and a single computed total."
    )


async def verify_epic_universe(evaluator: Evaluator, parent_node, eu: Optional[EpicUniverseBudget]) -> None:
    eu_node = evaluator.add_parallel(
        id="Epic_Universe_Budget",
        desc="Provide a complete, sourced, itemized 3-night Epic Universe/Universal Orlando budget.",
        parent=parent_node,
        critical=True
    )

    # Flights group
    flights_group = evaluator.add_parallel(
        id="EU_Roundtrip_Flights",
        desc="Roundtrip flights for a family of 4 from MSN to MCO with at least two airline options, baggage policies, and source URLs.",
        parent=eu_node,
        critical=True
    )
    flights = eu.flights if eu else None
    airline_opts = flights.airline_options if flights else []
    price_urls = flights.price_source_urls if flights else []

    # EU_Two_Airline_Options (existence/count check)
    evaluator.add_custom_node(
        result=(airline_opts is not None and len(airline_opts) >= 2 and all((opt.airline is not None and len(opt.airline.strip()) > 0) for opt in airline_opts[:2])),
        id="EU_Two_Airline_Options",
        desc="Identify at least two airline options for the MSN→MCO roundtrip itinerary.",
        parent=flights_group,
        critical=True
    )

    # EU_Flight_Total_Cost
    node_flight_total = evaluator.add_leaf(
        id="EU_Flight_Total_Cost",
        desc="Provide an estimated roundtrip flight cost for the family of 4 with enough detail to understand how it was computed.",
        parent=flights_group,
        critical=True
    )
    claim_flight_total = f"The estimated MSN↔MCO roundtrip flight total for the family of 4 is '{flights.total_cost_estimate if flights and flights.total_cost_estimate else ''}'."
    combined_flight_urls = combine_urls(price_urls, *[opt.source_urls for opt in airline_opts]) if flights else []
    await evaluator.verify(
        claim=claim_flight_total,
        node=node_flight_total,
        sources=combined_flight_urls if has_valid_urls(combined_flight_urls) else (price_urls if has_valid_urls(price_urls) else None),
        additional_instruction="Confirm flight pricing estimate from cited fare search/airline sources; allow typical price ranges within June–August 2026."
    )

    # EU_Baggage_Policies_For_Identified_Airlines
    node_baggage = evaluator.add_leaf(
        id="EU_Baggage_Policies_For_Identified_Airlines",
        desc="Provide baggage fee policies for the identified airline options (at minimum typical carry-on and checked-bag policy for each).",
        parent=flights_group,
        critical=True
    )
    bp_summary = []
    sources_baggage = []
    if airline_opts:
        for i, opt in enumerate(airline_opts[:2]):
            bp_summary.append(f"{opt.airline or ''}: {opt.baggage_policy or ''}")
            sources_baggage.extend(opt.source_urls or [])
    claim_baggage = " ; ".join(bp_summary) if bp_summary else "Baggage policies not specified."
    await evaluator.verify(
        claim=f"Baggage policies: {claim_baggage}",
        node=node_baggage,
        sources=sources_baggage if has_valid_urls(sources_baggage) else None,
        additional_instruction="Verify each airline's carry-on and checked baggage allowances/fees from official airline pages or credible sources provided."
    )

    # EU_Flights_and_Baggage_Source_URLs (existence check across price & policies)
    evaluator.add_custom_node(
        result=(has_valid_urls(price_urls) and all(has_valid_urls(opt.source_urls) for opt in airline_opts[:2])),
        id="EU_Flights_and_Baggage_Source_URLs",
        desc="Provide reference URL(s) supporting flight pricing and baggage fees/policies.",
        parent=flights_group,
        critical=True
    )

    # Hotel group
    hotel_group = evaluator.add_parallel(
        id="EU_Hotel_Accommodation",
        desc="Hotel accommodation for 3 nights with nightly rate, benefits note, and a source URL.",
        parent=eu_node,
        critical=True
    )
    hotel = eu.hotel if eu else None
    hotel_urls = hotel.source_urls if hotel else []

    node_eu_hotel_name = evaluator.add_leaf(
        id="EU_Hotel_Name",
        desc="Specify the hotel name (on-site or nearby).",
        parent=hotel_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The selected Orlando hotel is '{hotel.name if hotel and hotel.name else ''}'.",
        node=node_eu_hotel_name,
        sources=hotel_urls if has_valid_urls(hotel_urls) else None,
        additional_instruction="Verify the hotel's identity against the cited hotel page or official listing."
    )

    node_eu_hotel_rate = evaluator.add_leaf(
        id="EU_Hotel_Nightly_Rate",
        desc="Provide the nightly rate (or a clearly derived 3-night lodging cost) for the selected dates/window.",
        parent=hotel_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The nightly rate (or derived 3-night lodging cost) for the Orlando hotel is '{hotel.nightly_rate if hotel and hotel.nightly_rate else ''}'.",
        node=node_eu_hotel_rate,
        sources=hotel_urls if has_valid_urls(hotel_urls) else None,
        additional_instruction="Confirm from the hotel's rate/booking page; allow reasonable June–August 2026 flexibility and approximate pricing if clearly stated."
    )

    node_eu_hotel_benefits = evaluator.add_leaf(
        id="EU_Hotel_Benefits",
        desc="Note any special benefits like Early Park Admission for on-site hotels when applicable.",
        parent=hotel_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Hotel benefits noted: '{hotel.benefits_note if hotel and hotel.benefits_note else ''}'.",
        node=node_eu_hotel_benefits,
        sources=hotel_urls if has_valid_urls(hotel_urls) else None,
        additional_instruction="Verify benefits/policies (e.g., Early Park Admission) on the hotel's official page; if no benefits are claimed, the page should not state such benefits."
    )

    evaluator.add_custom_node(
        result=has_valid_urls(hotel_urls),
        id="EU_Hotel_Source_URL",
        desc="Provide a reference URL supporting the hotel rate and any stated benefits/policies where relevant.",
        parent=hotel_group,
        critical=True
    )

    # Theme Park Tickets group
    tickets_group = evaluator.add_parallel(
        id="EU_Theme_Park_Tickets",
        desc="Theme park ticket type and cost for the family with a source URL.",
        parent=eu_node,
        critical=True
    )
    tickets = eu.tickets if eu else None
    ticket_urls = tickets.source_urls if tickets else []

    node_ticket_type = evaluator.add_leaf(
        id="EU_Ticket_Type",
        desc="Specify whether tickets are single-day Epic Universe only or multi-day park-to-park tickets.",
        parent=tickets_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chosen ticket type is '{tickets.ticket_type if tickets and tickets.ticket_type else ''}'.",
        node=node_ticket_type,
        sources=ticket_urls if has_valid_urls(ticket_urls) else None,
        additional_instruction="Verify the product type on the cited ticketing page (Epic Universe-only day ticket vs multi-day park-to-park)."
    )

    node_ticket_total = evaluator.add_leaf(
        id="EU_Ticket_Total_Cost",
        desc="Provide total ticket cost for the family of 4 consistent with the specified ticket type.",
        parent=tickets_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total theme park ticket cost for the family of 4 is '{tickets.total_cost if tickets and tickets.total_cost else ''}'.",
        node=node_ticket_total,
        sources=ticket_urls if has_valid_urls(ticket_urls) else None,
        additional_instruction="Confirm ticket pricing from the cited ticketing page; allow June–August 2026 typical pricing ranges if clearly noted."
    )

    evaluator.add_custom_node(
        result=has_valid_urls(ticket_urls),
        id="EU_Ticket_Source_URL",
        desc="Provide a reference URL for ticket pricing.",
        parent=tickets_group,
        critical=True
    )

    # Ground Transportation group
    eu_transport = eu.transport if eu else None
    eu_trans_urls = eu_transport.source_urls if eu_transport else []
    eu_trans_group = evaluator.add_parallel(
        id="EU_Ground_Transportation_Cost",
        desc="Provide ground transportation cost from MCO to the hotel/Epic Universe area with a reference URL.",
        parent=eu_node,
        critical=True
    )
    node_eu_trans_cost = evaluator.add_leaf(
        id="EU_Transport_Cost_Estimate",
        desc="Provide an estimated transportation cost (e.g., shuttle/Uber/taxi).",
        parent=eu_trans_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The estimated ground transportation cost from MCO to the hotel/Epic Universe area is '{eu_transport.cost_estimate if eu_transport and eu_transport.cost_estimate else ''}'.",
        node=node_eu_trans_cost,
        sources=eu_trans_urls if has_valid_urls(eu_trans_urls) else None,
        additional_instruction="Check the cited transportation source for typical Uber/taxi/shuttle/pricing supporting the estimate."
    )
    evaluator.add_custom_node(
        result=has_valid_urls(eu_trans_urls),
        id="EU_Transport_Source_URL",
        desc="Provide a reference URL supporting the transportation pricing (or a sourced basis used to compute the estimate).",
        parent=eu_trans_group,
        critical=True
    )

    # Meals and Misc group
    eu_meals = eu.meals_misc if eu else None
    eu_meals_urls = eu_meals.source_urls if eu_meals else []
    eu_meals_group = evaluator.add_parallel(
        id="EU_Meals_and_Misc",
        desc="Provide an estimated meals and miscellaneous cost component with at least one reference URL supporting the basis for the estimate.",
        parent=eu_node,
        critical=True
    )
    node_eu_meals_cost = evaluator.add_leaf(
        id="EU_Meals_Misc_Cost_Estimate",
        desc="State the estimated meals and miscellaneous total for the trip.",
        parent=eu_meals_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The estimated meals and miscellaneous total for the family is '{eu_meals.total_estimate if eu_meals and eu_meals.total_estimate else ''}'.",
        node=node_eu_meals_cost,
        sources=eu_meals_urls if has_valid_urls(eu_meals_urls) else None,
        additional_instruction="Verify per-diem or typical meal cost basis from the cited sources (family of four, 4-day trip)."
    )
    evaluator.add_custom_node(
        result=has_valid_urls(eu_meals_urls),
        id="EU_Meals_Misc_Source_URL",
        desc="Provide a reference URL supporting the basis for the meals/misc estimate.",
        parent=eu_meals_group,
        critical=True
    )

    # EU_Itemized_Budget_and_Total
    node_eu_itemized = evaluator.add_leaf(
        id="EU_Itemized_Budget_and_Total",
        desc="Provide an itemized breakdown showing all cost components and a total estimated cost for Epic Universe/Universal Orlando.",
        parent=eu_node,
        critical=True
    )
    items_preview_eu = "; ".join(eu.itemized.items) if eu and eu.itemized and eu.itemized.items else ""
    claim_eu_itemized = f"The answer contains an itemized breakdown for Epic Universe/Universal Orlando (e.g., {items_preview_eu}) and a total estimated cost '{eu.itemized.total_cost if eu and eu.itemized and eu.itemized.total_cost else ''}'."
    await evaluator.verify(
        claim=claim_eu_itemized,
        node=node_eu_itemized,
        additional_instruction="Check the answer text for an itemized list of Epic Universe costs and a single computed total."
    )


async def verify_comparison(evaluator: Evaluator, parent_node, comp: Optional[ComparisonInfo]) -> None:
    comp_node = evaluator.add_parallel(
        id="Comparison_Summary",
        desc="Compare totals and summarize tradeoffs between the two options.",
        parent=parent_node,
        critical=True
    )

    node_affordable = evaluator.add_leaf(
        id="More_Affordable_Option",
        desc="State which destination option is more affordable based on the calculated totals.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies the more affordable option as '{comp.cheaper_option if comp and comp.cheaper_option else ''}' based on the computed totals.",
        node=node_affordable,
        additional_instruction="Verify consistency with the totals presented in the answer; the cheaper option should correspond to the lower total."
    )

    node_difference = evaluator.add_leaf(
        id="Affordability_Difference",
        desc="State by how much (dollar difference) the more affordable option is cheaper, consistent with the totals.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated affordability difference is '{comp.difference_amount if comp and comp.difference_amount else ''}', and it matches the difference between the two totals shown.",
        node=node_difference,
        additional_instruction="Check the arithmetic consistency using the totals stated in the answer."
    )

    node_tradeoffs = evaluator.add_leaf(
        id="Two_Tradeoffs",
        desc="Identify at least two key differences/tradeoffs between the options (e.g., travel time, variety of attractions, amenities).",
        parent=comp_node,
        critical=True
    )
    trades_preview = "; ".join(comp.tradeoffs) if comp and comp.tradeoffs else ""
    await evaluator.verify(
        claim=f"The answer lists at least two tradeoffs/differences (e.g., {trades_preview}).",
        node=node_tradeoffs,
        additional_instruction="Verify at least two distinct tradeoffs are stated in the answer."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2026 summer vacation budget comparison task.
    Builds a hierarchical verification tree with critical checks and evidence-backed verifications.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root orchestration; we'll add a critical planning node under it
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
        prompt=prompt_extract_vacation_comparison(),
        template_class=VacationComparisonExtraction,
        extraction_name="vacation_comparison_extraction"
    )

    # Build the top-level critical planning node (sequential as per rubric)
    planning_node = evaluator.add_sequential(
        id="Vacation_Comparison_Planning",
        desc="Compare Wisconsin Dells vs. Epic Universe/Universal Orlando for a family of 4 with itemized budgets, totals, sources, and a final comparison.",
        parent=root,
        critical=True
    )

    # 1) Summer 2026 Pricing Scope (critical leaf)
    await verify_scope(evaluator, planning_node, extracted.scope)

    # 2) Budgets For Both Destinations (critical parallel group)
    budgets_node = evaluator.add_parallel(
        id="Budgets_For_Both_Destinations",
        desc="Provide complete, sourced, itemized budgets (with totals) for both destination options.",
        parent=planning_node,
        critical=True
    )

    # 2a) Wisconsin Dells budget sub-tree
    await verify_wisconsin_dells(evaluator, budgets_node, extracted.wisconsin_dells)

    # 2b) Epic Universe/Universal Orlando budget sub-tree
    await verify_epic_universe(evaluator, budgets_node, extracted.epic_universe)

    # 3) Comparison Summary (critical parallel group)
    await verify_comparison(evaluator, planning_node, extracted.comparison)

    # Return final structured summary
    return evaluator.get_summary()