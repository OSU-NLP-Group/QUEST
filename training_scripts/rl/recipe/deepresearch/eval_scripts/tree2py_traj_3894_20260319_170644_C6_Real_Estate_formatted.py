import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lv_multifamily_investment"
TASK_DESCRIPTION = (
    "Identify a Class B multifamily investment property in the Las Vegas metropolitan area (Las Vegas, Henderson, or "
    "North Las Vegas) that meets the specified investment criteria. Provide address, units, age/class justification, "
    "occupancy, NOI, price/value, cap rate and calculation, DSCR with assumed loan terms (~6% rate, 75% LTV) and "
    "annual debt service, required 25% down payment, parking and zoning compliance, submarket vacancy (<8%), any "
    "renovation costs per unit, and include reference URLs supporting each major data point."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertySelectionExtraction(BaseModel):
    # Core identification
    property_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    submarket: Optional[str] = None

    # Physical and classification
    unit_count: Optional[str] = None
    year_built: Optional[str] = None
    property_class: Optional[str] = None

    # Operations
    occupancy_rate: Optional[str] = None
    noi: Optional[str] = None

    # Value/pricing
    purchase_price: Optional[str] = None
    assessed_value: Optional[str] = None
    cap_rate: Optional[str] = None

    # Financing and returns
    dscr: Optional[str] = None
    assumed_interest_rate: Optional[str] = None
    amortization_years: Optional[str] = None
    ltv: Optional[str] = None
    annual_debt_service: Optional[str] = None
    down_payment: Optional[str] = None

    # Parking and compliance
    parking_spaces: Optional[str] = None
    parking_ratio: Optional[str] = None

    # Market data
    submarket_vacancy_rate: Optional[str] = None

    # Renovations
    renovations_needed: Optional[str] = None  # e.g., "yes", "no", or description
    renovation_cost_per_unit: Optional[str] = None

    # Source URLs (categorical)
    source_urls_property: List[str] = Field(default_factory=list)    # listing/official property page(s)
    source_urls_noi: List[str] = Field(default_factory=list)
    source_urls_cap_rate: List[str] = Field(default_factory=list)
    source_urls_dscr: List[str] = Field(default_factory=list)        # e.g., market rate references, lender pages
    source_urls_equity: List[str] = Field(default_factory=list)      # e.g., price confirmation
    source_urls_occupancy: List[str] = Field(default_factory=list)
    source_urls_rent_roll: List[str] = Field(default_factory=list)
    source_urls_submarket: List[str] = Field(default_factory=list)   # market reports/statistics
    source_urls_condition: List[str] = Field(default_factory=list)   # renovation/condition discussion
    source_urls_parking: List[str] = Field(default_factory=list)     # property parking + local code references
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    Extract all requested data points for the proposed multifamily investment property mentioned in the answer.
    Return a single JSON object with the following fields:

    Core identification
    - property_name: the property's name if available (e.g., community name), else null
    - address: the complete street address if presented, else what is provided
    - city: city where the property is located
    - state: US state (expect NV)
    - zip_code: 5-digit ZIP if available, else null
    - submarket: the submarket/neighborhood area (e.g., "Henderson", "Spring Valley") if provided

    Physical and classification
    - unit_count: total number of residential units (keep as a string exactly as in the answer, e.g., "96", "96 units")
    - year_built: the year built or range/phrase shown in the answer
    - property_class: the class label if mentioned (e.g., "Class B")

    Operations
    - occupancy_rate: current occupancy rate string as shown (e.g., "92%", "approximately 95%")
    - noi: Net Operating Income per year string, as shown (e.g., "$1,250,000")

    Value/pricing
    - purchase_price: the price/value figure used for analysis (string)
    - assessed_value: assessed value if included (string or null)
    - cap_rate: cap rate value if provided (string, like "6.2%")

    Financing/returns
    - dscr: DSCR value if provided (string, like "1.30x")
    - assumed_interest_rate: assumed interest rate for DSCR calc (string, like "6%" or "6.0%")
    - amortization_years: assumed amortization term in years (string, e.g., "30")
    - ltv: assumed LTV ratio (string, like "75%")
    - annual_debt_service: annual debt service used in DSCR calc (string)
    - down_payment: the required down payment amount or percentage as provided (string)

    Parking and compliance
    - parking_spaces: total on-site parking space count if provided (string)
    - parking_ratio: the implied/claimed parking ratio if mentioned (e.g., "1.5 per unit")

    Market data
    - submarket_vacancy_rate: vacancy rate figure for the property's submarket (string)

    Renovations
    - renovations_needed: "yes"/"no" or a short phrase indicating need
    - renovation_cost_per_unit: per-unit renovation cost if needed (string like "$12,000")

    Source URLs (explicitly extract any URLs shown in the answer; do not invent)
    - source_urls_property: URLs supporting the property's identity (listing/official/community pages)
    - source_urls_noi: URLs supporting NOI
    - source_urls_cap_rate: URLs supporting cap rate or its inputs (NOI and price/value)
    - source_urls_dscr: URLs supporting DSCR analysis/assumptions (e.g., market loan rates around 6%, typical terms)
    - source_urls_equity: URLs supporting purchase price/value used for down payment calculation
    - source_urls_occupancy: URLs supporting occupancy rate
    - source_urls_rent_roll: URLs supporting unit mix/average rents if provided
    - source_urls_submarket: URLs supporting submarket vacancy rate
    - source_urls_condition: URLs supporting property condition/renovations
    - source_urls_parking: URLs supporting parking counts and local parking code/zoning references
    - other_urls: any remaining relevant references

    Rules:
    - Extract values exactly as written in the answer (do not normalize or round).
    - If a value is not present in the answer, return null for scalar fields or [] for URL arrays.
    - For URL fields, include only valid URLs explicitly present in the answer text (accept also markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u or not isinstance(u, str):
                continue
            key = u.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(key)
    return merged


def fmt_address(info: PropertySelectionExtraction) -> str:
    parts = [info.address or "", info.city or "", (info.state or "NV"), (info.zip_code or "")]
    return ", ".join([p for p in parts if p]).strip(", ").strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_property_identification(evaluator: Evaluator, parent, info: PropertySelectionExtraction) -> None:
    node = evaluator.add_parallel(
        id="Property_Identification",
        desc="Identify a specific property that meets basic classification and location requirements",
        parent=parent,
        critical=True
    )

    prop_urls = merge_urls(info.source_urls_property)

    # 1) Location Verification
    loc_leaf = evaluator.add_leaf(
        id="Location_Verification",
        desc="Verify the property is located within the Las Vegas metropolitan area (Las Vegas, Henderson, or North Las Vegas)",
        parent=node,
        critical=True
    )
    city = (info.city or "").strip()
    addr = fmt_address(info)
    loc_claim = (
        f"The property located at '{addr}' is in {city}, Nevada, and {city} is part of the Las Vegas metropolitan area "
        f"(acceptable cities: Las Vegas, Henderson, North Las Vegas)."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=prop_urls if prop_urls else None,
        additional_instruction="If the page shows the city as Las Vegas, Henderson, or North Las Vegas, accept as within the Las Vegas metro."
    )

    # 2) Property Class Verification (Class B)
    class_leaf = evaluator.add_leaf(
        id="Property_Class_Verification",
        desc="Verify the property is classified as Class B multifamily (typically 10-35 years old with moderate amenities)",
        parent=node,
        critical=True
    )
    yr_txt = (info.year_built or "").strip()
    class_txt = (info.property_class or "").strip()
    class_claim = (
        f"The property can be reasonably classified as 'Class B' based on the information provided: "
        f"year built '{yr_txt}' and amenities/positioning from the referenced page(s). "
        f"If the page explicitly labels it 'Class B', accept directly; otherwise, accept if the evidence supports "
        f"a typical Class B profile (roughly 10–35 years old with moderate amenities and professional management)."
    )
    await evaluator.verify(
        claim=class_claim,
        node=class_leaf,
        sources=prop_urls if prop_urls else None,
        additional_instruction="Use the year built and amenity description to judge Class B reasonableness if not explicitly labeled."
    )

    # 3) Unit Count Verification (50–100)
    units_leaf = evaluator.add_leaf(
        id="Unit_Count_Verification",
        desc="Verify the property contains between 50-100 residential units",
        parent=node,
        critical=True
    )
    units_txt = (info.unit_count or "").strip()
    units_claim = (
        f"The property has '{units_txt}' residential units, and this unit count is between 50 and 100 inclusive."
    )
    await evaluator.verify(
        claim=units_claim,
        node=units_leaf,
        sources=prop_urls if prop_urls else None,
        additional_instruction="Confirm the unit count on the referenced page. Treat minor formatting (e.g., '96 units') as acceptable."
    )

    # 4) Property Address Documentation
    addr_leaf = evaluator.add_leaf(
        id="Property_Address_Documentation",
        desc="Provide complete property address and basic identifying information",
        parent=node,
        critical=True
    )
    name_txt = (info.property_name or "").strip()
    addr_claim = (
        f"The property is identified as '{name_txt}' at the address '{addr}'. The referenced page(s) explicitly show this identification."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=prop_urls if prop_urls else None,
        additional_instruction="Accept if the page shows either the full address or a clear property/community identifier."
    )

    # 5) Reference URL presence/validity
    if prop_urls:
        ref_leaf = evaluator.add_leaf(
            id="Reference_URL",
            desc="Provide valid reference URL supporting property identification",
            parent=node,
            critical=True
        )
        ref_claim = (
            f"At least one of the provided URLs is an official or listing page that directly identifies the property "
            f"by its name and/or address (e.g., '{addr}')."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=prop_urls,
            additional_instruction="Look for explicit matches to the property name or street address on the page."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL",
            desc="Provide valid reference URL supporting property identification (no property URL provided)",
            parent=node,
            critical=True
        )


async def build_financial_analysis(evaluator: Evaluator, parent, info: PropertySelectionExtraction) -> None:
    fin_node = evaluator.add_parallel(
        id="Financial_Analysis",
        desc="Evaluate financial performance and investment metrics",
        parent=parent,
        critical=False
    )

    # ---- NOI Analysis ----
    noi_node = evaluator.add_parallel(
        id="Net_Operating_Income_Analysis",
        desc="Analyze and verify the property's Net Operating Income (NOI)",
        parent=fin_node,
        critical=False
    )
    noi_urls = merge_urls(info.source_urls_noi, info.source_urls_property)

    noi_doc_leaf = evaluator.add_leaf(
        id="NOI_Documentation",
        desc="Provide documented NOI figure with calculation methodology",
        parent=noi_node,
        critical=True
    )
    noi_txt = (info.noi or "").strip()
    noi_doc_claim = (
        f"The Net Operating Income (NOI) for the property is reported as '{noi_txt}' per year, and the referenced page(s) "
        f"either directly report NOI or provide enough detail (income and expenses) to derive it."
    )
    await evaluator.verify(
        claim=noi_doc_claim,
        node=noi_doc_leaf,
        sources=noi_urls if noi_urls else None,
        additional_instruction="Accept if the page lists NOI or provides gross income and operating expenses enabling calculation."
    )

    noi_break_leaf = evaluator.add_leaf(
        id="NOI_Components_Breakdown",
        desc="Break down gross rental income and operating expenses",
        parent=noi_node,
        critical=False
    )
    noi_break_claim = (
        "The answer provides a breakdown of NOI components (at least gross income and operating expenses) or describes "
        "how NOI was derived from them."
    )
    await evaluator.verify(
        claim=noi_break_claim,
        node=noi_break_leaf,
        sources=None,
        additional_instruction="Check in the answer text whether both income and operating expenses are mentioned for NOI derivation."
    )

    noi_ref_leaf = evaluator.add_leaf(
        id="NOI_Reference_URL",
        desc="Provide reference URL supporting NOI data",
        parent=noi_node,
        critical=True
    )
    noi_ref_claim = "The referenced page(s) explicitly support the NOI figure or provide inputs to calculate it."
    await evaluator.verify(
        claim=noi_ref_claim,
        node=noi_ref_leaf,
        sources=noi_urls if noi_urls else None,
        additional_instruction="Accept if the page shows NOI, T12 summary, or equivalent financials."
    )

    # ---- Cap Rate Analysis ----
    cap_node = evaluator.add_parallel(
        id="Capitalization_Rate_Analysis",
        desc="Calculate and verify the property's cap rate falls within acceptable range",
        parent=fin_node,
        critical=False
    )
    cap_urls = merge_urls(info.source_urls_cap_rate, info.source_urls_property)
    price_txt = (info.purchase_price or info.assessed_value or "").strip()
    cap_txt = (info.cap_rate or "").strip()

    cap_calc_leaf = evaluator.add_leaf(
        id="Cap_Rate_Calculation",
        desc="Calculate cap rate (NOI ÷ Property Value) and verify it falls within 5.5%-7.5%",
        parent=cap_node,
        critical=True
    )
    cap_calc_claim = (
        f"Using NOI '{noi_txt}' and property value/purchase price '{price_txt}', the cap rate should match the stated "
        f"cap rate '{cap_txt}' if provided and be within the 5.5%–7.5% range. Treat reasonable rounding as acceptable."
    )
    await evaluator.verify(
        claim=cap_calc_claim,
        node=cap_calc_leaf,
        sources=None,
        additional_instruction="Compute cap rate from the provided figures and check if it lies within 5.5%–7.5%."
    )

    market_comp_leaf = evaluator.add_leaf(
        id="Market_Comparison",
        desc="Compare calculated cap rate to similar Class B properties in the market",
        parent=cap_node,
        critical=False
    )
    market_comp_claim = (
        "The answer includes a brief market comparison, benchmarking the property's cap rate against similar Class B "
        "assets in the Las Vegas metro."
    )
    await evaluator.verify(
        claim=market_comp_claim,
        node=market_comp_leaf,
        sources=None,
        additional_instruction="Assess only whether the answer text includes a qualitative or quantitative comparison."
    )

    cap_ref_leaf = evaluator.add_leaf(
        id="Cap_Rate_Reference_URL",
        desc="Provide reference URL supporting cap rate analysis",
        parent=cap_node,
        critical=True
    )
    cap_ref_claim = (
        "At least one referenced page provides the cap rate directly or provides both NOI and price/value needed to compute it."
    )
    await evaluator.verify(
        claim=cap_ref_claim,
        node=cap_ref_leaf,
        sources=cap_urls if cap_urls else None,
        additional_instruction="Accept a page if it shows NOI and price/value, or explicitly states a cap rate."
    )

    # ---- DSCR Analysis ----
    dscr_node = evaluator.add_parallel(
        id="Debt_Service_Coverage_Analysis",
        desc="Evaluate financing feasibility through DSCR calculation",
        parent=fin_node,
        critical=False
    )
    dscr_urls = merge_urls(info.source_urls_dscr)
    dscr_txt = (info.dscr or "").strip()
    rate_txt = (info.assumed_interest_rate or "").strip()
    amort_txt = (info.amortization_years or "").strip()
    ltv_txt = (info.ltv or "").strip() or "75%"
    debt_service_txt = (info.annual_debt_service or "").strip()

    dscr_calc_leaf = evaluator.add_leaf(
        id="DSCR_Calculation",
        desc="Calculate DSCR (NOI ÷ Annual Debt Service) and verify it meets or exceeds 1.25",
        parent=dscr_node,
        critical=True
    )
    dscr_calc_claim = (
        f"Given NOI '{noi_txt}', annual debt service '{debt_service_txt}', and assumed loan terms around "
        f"'{rate_txt}' interest, '{amort_txt}' years amortization, and '{ltv_txt}' LTV (<=75%), the DSCR '{dscr_txt}' "
        f"is at least 1.25."
    )
    await evaluator.verify(
        claim=dscr_calc_claim,
        node=dscr_calc_leaf,
        sources=None,
        additional_instruction="Treat DSCR ≥ 1.25 as passing; perform arithmetic using the provided NOI and annual debt service."
    )

    loan_params_leaf = evaluator.add_leaf(
        id="Loan_Parameters",
        desc="Document assumed loan terms (interest rate, amortization period, LTV ratio)",
        parent=dscr_node,
        critical=False
    )
    loan_params_claim = (
        f"The answer explicitly documents assumed loan parameters: interest rate near '6%' (e.g., '{rate_txt}'), "
        f"an amortization period (e.g., '{amort_txt}' years), and LTV around '75%' (e.g., '{ltv_txt}')."
    )
    await evaluator.verify(
        claim=loan_params_claim,
        node=loan_params_leaf,
        sources=None,
        additional_instruction="Check the answer text for presence of all three assumptions."
    )

    dscr_ref_leaf = evaluator.add_leaf(
        id="DSCR_Reference_URL",
        desc="Provide reference URL supporting DSCR analysis and loan assumptions",
        parent=dscr_node,
        critical=True
    )
    dscr_ref_claim = (
        "At least one referenced page supports the DSCR analysis or the plausibility of the assumed loan parameters "
        "(e.g., shows current multifamily/commercial mortgage rates near 6% or typical terms)."
    )
    await evaluator.verify(
        claim=dscr_ref_claim,
        node=dscr_ref_leaf,
        sources=dscr_urls if dscr_urls else None,
        additional_instruction="Accept lender/market data pages indicating rates near ~6% and common terms."
    )

    # ---- Equity / Down Payment ----
    equity_node = evaluator.add_parallel(
        id="Equity_Investment_Analysis",
        desc="Calculate required equity investment based on LTV constraints",
        parent=fin_node,
        critical=False
    )
    eq_urls = merge_urls(info.source_urls_equity, info.source_urls_property)
    down_txt = (info.down_payment or "").strip()

    dp_leaf = evaluator.add_leaf(
        id="Down_Payment_Calculation",
        desc="Calculate minimum down payment (25% of purchase price) and verify affordability",
        parent=equity_node,
        critical=True
    )
    dp_claim = (
        f"With a purchase price/value of '{price_txt}', the minimum down payment is 25% of that amount. "
        f"The stated down payment in the answer is '{down_txt}', which is at least 25% of the purchase price/value."
    )
    await evaluator.verify(
        claim=dp_claim,
        node=dp_leaf,
        sources=None,
        additional_instruction="Compute 25% of the stated price/value and confirm the provided down payment meets or exceeds it."
    )

    tac_leaf = evaluator.add_leaf(
        id="Total_Acquisition_Cost",
        desc="Document total acquisition cost including closing costs and initial reserves",
        parent=equity_node,
        critical=False
    )
    tac_claim = "The answer documents total acquisition cost, including closing costs and initial reserves (if applicable)."
    await evaluator.verify(
        claim=tac_claim,
        node=tac_leaf,
        sources=None,
        additional_instruction="Check the answer text for mention of closing costs and initial reserves."
    )

    eq_ref_leaf = evaluator.add_leaf(
        id="Equity_Reference_URL",
        desc="Provide reference URL supporting equity analysis",
        parent=equity_node,
        critical=True
    )
    eq_ref_claim = "At least one referenced page supports the purchase price/value used for the down payment calculation."
    await evaluator.verify(
        claim=eq_ref_claim,
        node=eq_ref_leaf,
        sources=eq_urls if eq_urls else None,
        additional_instruction="Accept if the page clearly shows listing price or value."
    )


async def build_operational_performance(evaluator: Evaluator, parent, info: PropertySelectionExtraction) -> None:
    op_node = evaluator.add_parallel(
        id="Operational_Performance",
        desc="Evaluate current operational metrics and property management quality",
        parent=parent,
        critical=False
    )

    # Occupancy Analysis
    occ_node = evaluator.add_parallel(
        id="Occupancy_Analysis",
        desc="Analyze current occupancy rate and tenant retention",
        parent=op_node,
        critical=False
    )
    occ_urls = merge_urls(info.source_urls_occupancy, info.source_urls_property)
    occ_txt = (info.occupancy_rate or "").strip()

    occ_rate_leaf = evaluator.add_leaf(
        id="Current_Occupancy_Rate",
        desc="Verify current occupancy rate is at least 90%",
        parent=occ_node,
        critical=True
    )
    occ_rate_claim = f"The current occupancy rate is '{occ_txt}', which is at least 90%."
    await evaluator.verify(
        claim=occ_rate_claim,
        node=occ_rate_leaf,
        sources=occ_urls if occ_urls else None,
        additional_instruction="Confirm the occupancy rate on the referenced page; accept reasonable rounding."
    )

    occ_trend_leaf = evaluator.add_leaf(
        id="Occupancy_Trend",
        desc="Document occupancy trend over the past 12 months",
        parent=occ_node,
        critical=False
    )
    occ_trend_claim = "The answer documents an occupancy trend for approximately the past 12 months."
    await evaluator.verify(
        claim=occ_trend_claim,
        node=occ_trend_leaf,
        sources=None,
        additional_instruction="Look for narrative or numbers showing month-to-month or rolling occupancy."
    )

    occ_ref_leaf = evaluator.add_leaf(
        id="Occupancy_Reference_URL",
        desc="Provide reference URL supporting occupancy data",
        parent=occ_node,
        critical=True
    )
    occ_ref_claim = "At least one referenced page supports the stated current occupancy rate."
    await evaluator.verify(
        claim=occ_ref_claim,
        node=occ_ref_leaf,
        sources=occ_urls if occ_urls else None,
        additional_instruction="Accept if the page shows occupancy, preleased %, or similar indicator."
    )

    # Rent Roll Analysis (Non-critical grouping)
    rent_node = evaluator.add_parallel(
        id="Rent_Roll_Analysis",
        desc="Analyze rent roll and unit mix",
        parent=op_node,
        critical=False
    )
    rent_urls = merge_urls(info.source_urls_rent_roll, info.source_urls_property)

    unit_mix_leaf = evaluator.add_leaf(
        id="Unit_Mix_Documentation",
        desc="Document unit mix (number of 1BR, 2BR, 3BR units, etc.)",
        parent=rent_node,
        critical=False
    )
    unit_mix_claim = "The answer documents the unit mix (e.g., quantities of 1BR, 2BR, etc.)."
    await evaluator.verify(
        claim=unit_mix_claim,
        node=unit_mix_leaf,
        sources=None,
        additional_instruction="Check if the answer lists the counts by unit type."
    )

    avg_rent_leaf = evaluator.add_leaf(
        id="Average_Rent_Per_Unit",
        desc="Calculate and document average rent per unit type",
        parent=rent_node,
        critical=False
    )
    avg_rent_claim = "The answer provides typical/average rents per unit type or an overall average rent per unit."
    await evaluator.verify(
        claim=avg_rent_claim,
        node=avg_rent_leaf,
        sources=None,
        additional_instruction="Look for a dollar figure per unit or per plan."
    )

    rent_ref_leaf = evaluator.add_leaf(
        id="Rent_Reference_URL",
        desc="Provide reference URL supporting rent roll data",
        parent=rent_node,
        critical=False
    )
    rent_ref_claim = "At least one referenced page supports the rent roll or typical asking rents."
    await evaluator.verify(
        claim=rent_ref_claim,
        node=rent_ref_leaf,
        sources=rent_urls if rent_urls else None,
        additional_instruction="Accept if the page shows rents or a rent schedule."
    )


async def build_market_position(evaluator: Evaluator, parent, info: PropertySelectionExtraction) -> None:
    mp_node = evaluator.add_parallel(
        id="Market_Position_Analysis",
        desc="Evaluate property's position within its submarket",
        parent=parent,
        critical=False
    )

    # Submarket Fundamentals
    sub_node = evaluator.add_parallel(
        id="Submarket_Fundamentals",
        desc="Analyze submarket supply and demand dynamics",
        parent=mp_node,
        critical=False
    )
    sub_urls = merge_urls(info.source_urls_submarket)
    vac_txt = (info.submarket_vacancy_rate or "").strip()
    area_txt = (info.submarket or info.city or "the local submarket").strip()

    vac_leaf = evaluator.add_leaf(
        id="Vacancy_Rate_Analysis",
        desc="Verify submarket multifamily vacancy rate is below 8%",
        parent=sub_node,
        critical=True
    )
    vac_claim = f"The submarket vacancy rate for {area_txt} is '{vac_txt}' and is below 8%."
    await evaluator.verify(
        claim=vac_claim,
        node=vac_leaf,
        sources=sub_urls if sub_urls else None,
        additional_instruction="Verify the vacancy percentage on the referenced market report or data page."
    )

    comp_pos_leaf = evaluator.add_leaf(
        id="Competitive_Position",
        desc="Assess property's competitive position relative to comparable properties",
        parent=sub_node,
        critical=False
    )
    comp_pos_claim = "The answer discusses the property's competitive position relative to comparable assets in the submarket."
    await evaluator.verify(
        claim=comp_pos_claim,
        node=comp_pos_leaf,
        sources=None,
        additional_instruction="Look for qualitative commentary in the answer."
    )

    sub_ref_leaf = evaluator.add_leaf(
        id="Submarket_Reference_URL",
        desc="Provide reference URL supporting submarket data",
        parent=sub_node,
        critical=True
    )
    sub_ref_claim = "At least one referenced page supports the submarket data (e.g., vacancy rate)."
    await evaluator.verify(
        claim=sub_ref_claim,
        node=sub_ref_leaf,
        sources=sub_urls if sub_urls else None,
        additional_instruction="Accept reputable market reports, brokerage dashboards, or government data."
    )

    # Location Quality (non-critical)
    locq_node = evaluator.add_parallel(
        id="Location_Quality",
        desc="Evaluate neighborhood quality and access to amenities",
        parent=mp_node,
        critical=False
    )

    access_emp_leaf = evaluator.add_leaf(
        id="Access_To_Employment",
        desc="Document proximity to major employment centers",
        parent=locq_node,
        critical=False
    )
    access_emp_claim = "The answer mentions proximity or convenient access to major employment centers."
    await evaluator.verify(
        claim=access_emp_claim,
        node=access_emp_leaf,
        sources=None,
        additional_instruction="Check the answer text for a short statement about employment access."
    )

    trans_leaf = evaluator.add_leaf(
        id="Transportation_Access",
        desc="Document access to major transportation corridors",
        parent=locq_node,
        critical=False
    )
    trans_claim = "The answer mentions access to major transportation corridors or transit options."
    await evaluator.verify(
        claim=trans_claim,
        node=trans_leaf,
        sources=None,
        additional_instruction="Check the answer text for roads/highways/transit access statements."
    )


async def build_physical_property(evaluator: Evaluator, parent, info: PropertySelectionExtraction) -> None:
    phys_node = evaluator.add_parallel(
        id="Physical_Property_Assessment",
        desc="Evaluate physical condition and required improvements",
        parent=parent,
        critical=False
    )

    # Property Condition (non-critical)
    cond_node = evaluator.add_parallel(
        id="Property_Condition",
        desc="Assess overall property condition and required renovations",
        parent=phys_node,
        critical=False
    )
    cond_urls = merge_urls(info.source_urls_condition, info.source_urls_property)
    ren_need_txt = (info.renovations_needed or "").strip().lower()
    ren_cost_txt = (info.renovation_cost_per_unit or "").strip()

    ren_leaf = evaluator.add_leaf(
        id="Renovation_Needs_Assessment",
        desc="Identify required renovations and verify cost does not exceed $15,000 per unit",
        parent=cond_node,
        critical=True
    )
    ren_claim = (
        f"If renovations are needed ('{ren_need_txt}'), the per-unit renovation cost '{ren_cost_txt}' does not exceed $15,000; "
        f"if no renovations are needed, this criterion is satisfied."
    )
    await evaluator.verify(
        claim=ren_claim,
        node=ren_leaf,
        sources=None,
        additional_instruction="Check only the logic with the provided values; accept if 'no renovations' or cost ≤ $15k/unit."
    )

    def_maint_leaf = evaluator.add_leaf(
        id="Deferred_Maintenance",
        desc="Document any significant deferred maintenance items",
        parent=cond_node,
        critical=False
    )
    def_maint_claim = "The answer mentions any significant deferred maintenance items, if applicable."
    await evaluator.verify(
        claim=def_maint_claim,
        node=def_maint_leaf,
        sources=None,
        additional_instruction="Look for narrative in the answer regarding deferred maintenance."
    )

    cond_ref_leaf = evaluator.add_leaf(
        id="Condition_Reference_URL",
        desc="Provide reference URL supporting property condition assessment",
        parent=cond_node,
        critical=True
    )
    cond_ref_claim = "At least one referenced page supports the described property condition and/or renovation scope."
    await evaluator.verify(
        claim=cond_ref_claim,
        node=cond_ref_leaf,
        sources=cond_urls if cond_urls else None,
        additional_instruction="Accept if the page shows photos, descriptions, or scope of work relevant to condition."
    )

    # Parking Compliance (critical within physical assessment)
    park_node = evaluator.add_parallel(
        id="Parking_Compliance",
        desc="Verify adequate parking is provided per local requirements",
        parent=phys_node,
        critical=True
    )
    park_urls = merge_urls(info.source_urls_parking, info.source_urls_property)
    park_spaces_txt = (info.parking_spaces or "").strip()
    city_txt = (info.city or "").strip()

    park_count_leaf = evaluator.add_leaf(
        id="Parking_Space_Count",
        desc="Document number of parking spaces provided",
        parent=park_node,
        critical=True
    )
    park_count_claim = f"The property provides '{park_spaces_txt}' total parking spaces."
    await evaluator.verify(
        claim=park_count_claim,
        node=park_count_leaf,
        sources=park_urls if park_urls else None,
        additional_instruction="Accept if the page shows parking counts, site plan counts, or an explicit number."
    )

    park_ratio_leaf = evaluator.add_leaf(
        id="Parking_Ratio_Compliance",
        desc="Verify parking ratio meets local zoning requirements for multifamily properties",
        parent=park_node,
        critical=True
    )
    park_ratio_claim = (
        f"Based on the provided number of units '{(info.unit_count or '').strip()}' and parking spaces '{park_spaces_txt}', "
        f"the property's parking supply meets or exceeds the minimum multifamily parking requirement in {city_txt} "
        f"(using the referenced local zoning/code source)."
    )
    await evaluator.verify(
        claim=park_ratio_claim,
        node=park_ratio_leaf,
        sources=park_urls if park_urls else None,
        additional_instruction="Use both the property page (spaces/units) and the municipal code page(s) to judge compliance."
    )

    park_ref_leaf = evaluator.add_leaf(
        id="Parking_Reference_URL",
        desc="Provide reference URL supporting parking data",
        parent=park_node,
        critical=True
    )
    park_ref_claim = "At least one referenced page supports the property's parking details and/or the applicable local code requirement."
    await evaluator.verify(
        claim=park_ref_claim,
        node=park_ref_leaf,
        sources=park_urls if park_urls else None,
        additional_instruction="Accept if either the property page shows counts or the code page specifies required ratios."
    )


async def build_property_evaluation(evaluator: Evaluator, parent, info: PropertySelectionExtraction) -> None:
    pe_node = evaluator.add_parallel(
        id="Property_Evaluation",
        desc="Comprehensive evaluation of the identified property across financial, operational, and compliance dimensions",
        parent=parent,
        critical=False
    )

    await build_financial_analysis(evaluator, pe_node, info)
    await build_operational_performance(evaluator, pe_node, info)
    await build_market_position(evaluator, pe_node, info)
    await build_physical_property(evaluator, pe_node, info)


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Identify first, then evaluate
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

    # IMPORTANT: To allow partial credit across the big tree (and to avoid critical-parent constraint issues),
    # we keep the root node non-critical. We'll enforce criticality at meaningful subtrees/leaves.
    root.critical = False

    # Extract structured data from the answer
    prop_info = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=PropertySelectionExtraction,
        extraction_name="property_extraction"
    )

    # Phase 1: Property Identification (critical gate)
    await build_property_identification(evaluator, root, prop_info)

    # Phase 2: Property Evaluation (non-critical; will be skipped automatically if Phase 1 fails due to sequential root)
    await build_property_evaluation(evaluator, root, prop_info)

    # Return evaluation summary
    return evaluator.get_summary()