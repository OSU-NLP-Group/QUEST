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
TASK_ID = "real_estate_portfolio_2026"
TASK_DESCRIPTION = """
You are assisting a real estate investment firm that is diversifying its portfolio across three distinct markets in 2026. The firm has asked you to identify three specific investment-grade properties, each in a different market and each meeting detailed financing, location, operational, and market criteria.

Property 1 - Carolinas Commercial Property (SBA 504 Eligible):
Identify a commercial property located in a city served by Duke Energy Carolinas (such as Charlotte, Belmont, Anderson SC, Alamance NC, Archdale NC, Asheboro NC, or similar cities in the Duke Energy Carolinas service territory). The property must:
- Have a price or require a loan amount of at least $400,000 to qualify for a Bank of America SBA 504 loan
- Be suitable for at least 51% owner occupancy by a business (to meet SBA 504 requirements for existing buildings)
- Be commercial real estate (office, retail, industrial, or mixed-use with a commercial component)
- Have net operating income sufficient to support a Debt Service Coverage Ratio (DSCR) of at least 1.25x
- Be eligible for commercial property insurance with replacement cost coverage

Provide: complete street address, total square footage, current asking price or assessed value, and URL reference to the property listing or official source.

Property 2 - Cancun Vacation Rental:
Identify a vacation rental property in Cancun, Quintana Roo, Mexico. The property must:
- Be residential real estate suitable for short-term vacation rental operations (villa, condo, apartment, or house)
- Be currently operating as a vacation rental OR be furnished and ready for vacation rental use
- Demonstrate or have realistic potential to achieve a gross rental yield of at least 8% (based on market data for well-managed Cancun vacation rentals in 2026)
- Demonstrate or have realistic potential to achieve an average occupancy rate of at least 40% (based on Cancun vacation rental market benchmarks)
- Have financial analysis that accounts for property management fees of approximately 25-30% of rental income

Provide: property address or specific location within Cancun, property size (bedrooms/bathrooms or square footage), current asking price or estimated market value, and URL reference to the property listing or real estate source.

Property 3 - Midtown West Manhattan Retail Space:
Identify a retail space in Midtown West, Manhattan, New York, within reasonable proximity to the New York Road Runners (NYRR) headquarters at 156 West 56th Street. The property must:
- Be located in the Midtown West neighborhood
- Be designated or zoned as retail commercial space
- Be ground floor or street-level retail space
- Have an asking rental rate that can be expressed in dollars per square foot per year
- Have a rental rate that reflects current Midtown West market conditions (with consideration that Trophy Class A space commands approximately $120-125 per square foot in 2025, though rates vary by property class and specific location)

Provide: complete street address, total square footage of the retail space, annual rent or asking rent, rental rate per square foot per year, and URL reference to the retail space listing or leasing information.

For each property, you must provide complete documentation including all requested details and URL references that verify the property information and demonstrate that it meets the specified criteria.
"""

NYRR_HQ_ADDRESS = "156 West 56th Street, New York, NY 10019"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Property1Extraction(BaseModel):
    # Core deliverables
    address: Optional[str] = None
    total_square_footage: Optional[str] = None
    asking_price_or_assessed_value: Optional[str] = None
    listing_urls: List[str] = Field(default_factory=list)

    # Location/service area support
    duke_service_reference_urls: List[str] = Field(default_factory=list)

    # SBA/Use/type
    property_type: Optional[str] = None
    owner_occupancy_suitability_claim: Optional[str] = None  # e.g., "Suitable for 51% owner occupancy"

    # Financial/DSCR
    noi: Optional[str] = None
    annual_debt_service: Optional[str] = None
    dscr_claim: Optional[str] = None  # e.g., "DSCR 1.35x"

    # Insurance
    insurance_claim: Optional[str] = None  # e.g., "Eligible for replacement cost coverage"
    insurance_support_urls: List[str] = Field(default_factory=list)

    # Optional loan amount if price missing
    required_loan_amount: Optional[str] = None


class Property2Extraction(BaseModel):
    # Core deliverables
    address_or_location: Optional[str] = None
    size_bed_bath_or_sqft: Optional[str] = None
    asking_price_or_market_value: Optional[str] = None
    listing_urls: List[str] = Field(default_factory=list)

    # Suitability & readiness
    residential_type: Optional[str] = None  # condo/house/villa/apartment
    short_term_suitable_claim: Optional[str] = None
    operating_or_ready_claim: Optional[str] = None

    # Returns/benchmarks
    gross_yield_percent: Optional[str] = None  # e.g., "9%" or "0.09"
    annual_gross_rental_income: Optional[str] = None  # e.g., "$72,000"
    occupancy_percent: Optional[str] = None  # e.g., "45%"
    management_fee_percent: Optional[str] = None  # e.g., "28%"

    market_benchmark_urls: List[str] = Field(default_factory=list)


class Property3Extraction(BaseModel):
    # Core deliverables
    address: Optional[str] = None
    total_square_footage: Optional[str] = None
    annual_rent_or_asking_rent: Optional[str] = None
    rate_per_sf_per_year: Optional[str] = None
    listing_urls: List[str] = Field(default_factory=list)

    # Retail designation, floor level
    zoning_or_designation_claim: Optional[str] = None  # "retail", "commercial retail"
    ground_floor_or_street_level_claim: Optional[str] = None

    # Midtown West and proximity
    midtown_west_claim: Optional[str] = None
    proximity_claim: Optional[str] = None  # e.g., "0.4 miles from 156 W 56th"
    proximity_support_urls: List[str] = Field(default_factory=list)

    # Market context explanation
    market_context_explanation: Optional[str] = None  # narrative comparing to trophy rates


class RealEstateExtraction(BaseModel):
    property_1: Optional[Property1Extraction] = None
    property_2: Optional[Property2Extraction] = None
    property_3: Optional[Property3Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_real_estate() -> str:
    return """
    Extract structured details for three properties described in the answer. Return null for any field not present.
    
    For Property 1 (Carolinas Commercial, SBA 504 eligible), extract:
    - address: complete street address
    - total_square_footage: total SF (e.g., "12,345 SF")
    - asking_price_or_assessed_value: price/value string as shown (e.g., "$1,250,000")
    - listing_urls: all URLs to the listing or official source(s)
    - duke_service_reference_urls: URLs that show Duke Energy Carolinas service territory for the property's city
    - property_type: type of property (office/retail/industrial/mixed-use)
    - owner_occupancy_suitability_claim: verbatim text indicating suitability for ≥51% owner occupancy, if provided
    - noi: net operating income figure string, if provided
    - annual_debt_service: annual debt service figure string or basis, if provided
    - dscr_claim: explicit DSCR statement if present (e.g., "DSCR 1.30x")
    - insurance_claim: verbatim text regarding eligibility for commercial property insurance with replacement cost coverage, if provided
    - insurance_support_urls: URLs that support insurability/coverage suitability
    - required_loan_amount: required or contemplated loan amount if given
    
    For Property 2 (Cancun vacation rental), extract:
    - address_or_location: the address or specific location within Cancun
    - size_bed_bath_or_sqft: e.g., "2BR/2BA 1,100 SF"
    - asking_price_or_market_value: price or estimated value string
    - listing_urls: all URLs to listing/real estate sources
    - residential_type: condo/house/villa/apartment/etc.
    - short_term_suitable_claim: text showing suitability for STR
    - operating_or_ready_claim: text indicating currently operating as a vacation rental or furnished/ready
    - gross_yield_percent: stated or computed gross yield percent if given (e.g., "8.5%")
    - annual_gross_rental_income: annual gross income used or stated in the analysis
    - occupancy_percent: occupancy rate percent if provided (e.g., "45%")
    - management_fee_percent: stated management fee percent (aiming for 25–30%)
    - market_benchmark_urls: URLs that provide market benchmarks (ADR/occupancy/yield) for Cancun STR
    
    For Property 3 (Midtown West retail), extract:
    - address: complete street address
    - total_square_footage: total SF of the retail space
    - annual_rent_or_asking_rent: annual rent or asking rent string
    - rate_per_sf_per_year: explicit $/SF/YR if available
    - listing_urls: all listing/leasing URLs
    - zoning_or_designation_claim: text indicating retail use/designation/zoning
    - ground_floor_or_street_level_claim: text showing ground floor/street-level
    - midtown_west_claim: any text explicitly indicating Midtown West
    - proximity_claim: stated distance/time to 156 West 56th Street, if provided
    - proximity_support_urls: URLs (e.g., maps) used to support proximity
    - market_context_explanation: narrative explaining how the rate reflects current Midtown West market context considering trophy ~$120–125/SF (2025).
    
    Output JSON with keys:
    {
      "property_1": { ...fields above... },
      "property_2": { ...fields above... },
      "property_3": { ...fields above... }
    }
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
_money_re = re.compile(r"([-+]?\d{1,3}(?:[,]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(\s*(k|m|b|bn|million|billion))?", re.I)
_percent_re = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")

def parse_money_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.strip().lower()
    m = _money_re.search(t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    suffix = (m.group(3) or "").lower()
    if suffix in ("k",):
        num *= 1_000
    elif suffix in ("m", "million"):
        num *= 1_000_000
    elif suffix in ("b", "bn", "billion"):
        num *= 1_000_000_000
    return num


def parse_percent_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _percent_re.search(text)
    if m:
        return float(m.group(1))
    # Try decimal like "0.08"
    try:
        val = float(text.strip())
        # Heuristic: if <= 1.0, treat as decimal fraction
        if 0 < val <= 1.0:
            return val * 100.0
        return val
    except Exception:
        return None


def to_unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    out.append(u2)
    return out


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def infer_city_from_address(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    # Simple heuristic: split by comma and take the segment following the street
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 2:
        # Often index 1 is city
        return parts[1]
    return None


def compute_dscr(noi_text: Optional[str], debt_service_text: Optional[str], dscr_text: Optional[str]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    # Try explicit DSCR first
    dscr = parse_percent_to_float(dscr_text)  # may not be %; handle below
    if dscr is not None:
        # If DSCR is like "1.30%" it's nonsensical; usually it's "1.30x"
        # Here, percent parser would return 1.30; good enough. We'll ignore x
        return dscr, None, None
    # Try float from DSCR string like "1.35x"
    if dscr_text:
        m = re.search(r"(\d+(?:\.\d+)?)\s*x", dscr_text.lower())
        if m:
            try:
                return float(m.group(1)), None, None
            except Exception:
                pass
        try:
            maybe = float(dscr_text.strip())
            return maybe, None, None
        except Exception:
            pass
    # Compute from NOI and debt service
    noi = parse_money_to_float(noi_text)
    ds = parse_money_to_float(debt_service_text)
    if noi is not None and ds and ds > 0:
        return noi / ds, noi_text, debt_service_text
    return None, noi_text, debt_service_text


def compute_rate_per_sf_per_year(annual_rent_text: Optional[str], sqft_text: Optional[str]) -> Optional[float]:
    rent = parse_money_to_float(annual_rent_text)
    if not rent:
        return None
    # Extract numeric sqft
    if not sqft_text:
        return None
    # Find number in sqft
    m = _money_re.search(sqft_text.replace("SF", "").replace("sf", "").replace("sqft", ""))
    sf = None
    if m:
        try:
            sf = float(m.group(1).replace(",", ""))
        except Exception:
            sf = None
    if sf and sf > 0:
        return rent / sf
    return None


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_and_verify(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    sources: Optional[List[str] | str] = None,
    add_ins: Optional[str] = None,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins or "None",
    )
    return leaf


# --------------------------------------------------------------------------- #
# Property 1 verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_property_1(evaluator: Evaluator, root, p1: Optional[Property1Extraction]) -> None:
    prop_node = evaluator.add_parallel(
        id="property_1_carolinas_commercial",
        desc="Property 1 satisfies Duke Energy Carolinas service-area, SBA 504-related criteria, underwriting/insurance criteria, and required outputs.",
        parent=root,
        critical=False,
    )

    address = (p1.address if p1 else "") or ""
    city = infer_city_from_address(address) or ""
    list_urls = (p1.listing_urls if p1 else []) or []
    duke_urls = (p1.duke_service_reference_urls if p1 else []) or []
    all_loc_sources = to_unique_urls(list_urls, duke_urls)

    # p1_location_in_duke_service_area (critical)
    await add_and_verify(
        evaluator,
        "p1_location_in_duke_service_area",
        "Property is located in a city served by Duke Energy Carolinas (supported by the address/city and a cited Duke Energy Carolinas service-territory reference).",
        prop_node,
        True,
        claim=(f"The property at {address} is in {city} and {city or 'this city'} is served by Duke Energy Carolinas (not Duke Energy Progress)."),
        sources=all_loc_sources if all_loc_sources else None,
        add_ins="Use the provided Duke Energy Carolinas service-territory source(s) to confirm the city lies within Duke Energy Carolinas (DEC), not Duke Energy Progress. Minor address formatting differences are acceptable.",
    )

    # p1_min_price_or_loan_400k (critical)
    price_text = (p1.asking_price_or_assessed_value if p1 else None) or (p1.required_loan_amount if p1 else None) or ""
    price_val = parse_money_to_float(price_text)
    await add_and_verify(
        evaluator,
        "p1_min_price_or_loan_400k",
        "Asking price or required loan amount is at least $400,000.",
        prop_node,
        True,
        claim=(f"The property's asking price or required loan amount is at least $400,000. "
               f"{('It is ' + price_text) if price_text else ''}"),
        sources=list_urls if list_urls else None,
        add_ins="Confirm from the listing or provided documentation that the price/value shown is ≥ $400,000. If a range is shown, accept if the minimum is ≥ $400,000.",
    )

    # p1_owner_occupancy_51pct (critical)
    oo_claim = (p1.owner_occupancy_suitability_claim if p1 else None) or "The property is suitable for at least 51% owner occupancy by a business."
    await add_and_verify(
        evaluator,
        "p1_owner_occupancy_51pct",
        "Property is suitable for at least 51% owner occupancy by a business (SBA 504 requirement for existing buildings).",
        prop_node,
        True,
        claim=oo_claim,
        sources=list_urls if list_urls else None,
        add_ins="From the listing language or answer rationale, judge whether ≥51% owner occupancy is feasible (e.g., single-tenant building that can be owner-occupied, vacant or partially vacant space). If it's fully locked in a long-term NNN lease to a third party, it likely is not suitable.",
    )

    # p1_is_commercial_real_estate (critical)
    ptype = (p1.property_type if p1 else None) or "commercial real estate (office/retail/industrial or mixed-use with commercial component)"
    await add_and_verify(
        evaluator,
        "p1_is_commercial_real_estate",
        "Property is commercial real estate (office, retail, industrial, or mixed-use with a commercial component).",
        prop_node,
        True,
        claim=f"The property is {ptype}.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm the listing indicates commercial usage (office, retail, industrial, or commercial component in mixed-use).",
    )

    # p1_dscr_at_least_1_25 (critical) - logical/calculation check from answer
    dscr_val, used_noi_text, used_debt_text = compute_dscr(
        p1.noi if p1 else None,
        p1.annual_debt_service if p1 else None,
        p1.dscr_claim if p1 else None,
    )
    if dscr_val is not None:
        dscr_claim_text = f"Based on the provided figures, DSCR is approximately {dscr_val:.2f}, which is at least 1.25x."
    else:
        dscr_claim_text = "The property can achieve DSCR ≥ 1.25x based on the provided NOI and debt service assumptions."

    await add_and_verify(
        evaluator,
        "p1_dscr_at_least_1_25",
        "Provides support that the property can achieve DSCR ≥ 1.25x based on provided/estimated NOI and debt service (calculation or clearly stated DSCR claim with supporting figures).",
        prop_node,
        True,
        claim=dscr_claim_text,
        sources=None,  # Calculation/logical check from the answer itself
        add_ins="Use the figures in the answer (NOI and annual debt service) to compute DSCR = NOI / Debt Service. Accept minor rounding differences. If an explicit DSCR ≥ 1.25 is stated with figures, accept.",
    )

    # p1_replacement_cost_insurance_eligible (critical)
    ins_claim = (p1.insurance_claim if p1 else None) or "The property is eligible for commercial property insurance with replacement cost coverage."
    ins_sources = to_unique_urls(p1.insurance_support_urls if p1 else [], list_urls)
    await add_and_verify(
        evaluator,
        "p1_replacement_cost_insurance_eligible",
        "States that the property is eligible for commercial property insurance with replacement cost coverage (with a supporting citation or listing/official documentation indicating insurability/coverage suitability).",
        prop_node,
        True,
        claim=ins_claim,
        sources=ins_sources if ins_sources else None,
        add_ins="Look for language indicating insurability and suitability for replacement cost coverage in the provided sources (listing or insurance/official documentation). If no such indication exists, mark as not supported.",
    )

    # p1_required_outputs (critical group)
    req_node = evaluator.add_parallel(
        id="p1_required_outputs",
        desc="Required deliverables for Property 1 are provided.",
        parent=prop_node,
        critical=True,
    )

    await add_and_verify(
        evaluator,
        "p1_complete_street_address",
        "Provides complete street address.",
        req_node,
        True,
        claim=f"The complete street address for Property 1 is '{address}'.",
        sources=list_urls if list_urls else None,
        add_ins="Verify the address matches or reasonably corresponds to the listing source. Minor formatting differences are acceptable.",
    )

    sqft_text = (p1.total_square_footage if p1 else "") or ""
    await add_and_verify(
        evaluator,
        "p1_total_square_footage",
        "Provides total square footage.",
        req_node,
        True,
        claim=f"The total square footage for Property 1 is '{sqft_text}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm the total building area or SF figure appears on the listing/source. Minor unit/format variations acceptable.",
    )

    price_text_full = (p1.asking_price_or_assessed_value if p1 else None) or ""
    await add_and_verify(
        evaluator,
        "p1_asking_price_or_assessed_value",
        "Provides current asking price or assessed value.",
        req_node,
        True,
        claim=f"The current asking price or assessed value for Property 1 is '{price_text_full}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm that the stated price/value appears on the listing or official source.",
    )

    evaluator.add_custom_node(
        result=any(is_valid_url(u) for u in list_urls),
        id="p1_listing_or_official_url",
        desc="Provides a URL reference to the property listing or official source supporting the provided facts.",
        parent=req_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Property 2 verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_property_2(evaluator: Evaluator, root, p2: Optional[Property2Extraction]) -> None:
    prop_node = evaluator.add_parallel(
        id="property_2_cancun_vacation_rental",
        desc="Property 2 satisfies Cancun location, STR suitability/readiness, yield/occupancy targets, management-fee assumption, and required outputs.",
        parent=root,
        critical=False,
    )

    list_urls = (p2.listing_urls if p2 else []) or []
    bench_urls = (p2.market_benchmark_urls if p2 else []) or []
    all_urls = to_unique_urls(list_urls, bench_urls)
    addr_loc = (p2.address_or_location if p2 else "") or ""

    # p2_in_cancun_quintana_roo (critical)
    await add_and_verify(
        evaluator,
        "p2_in_cancun_quintana_roo",
        "Property is located in Cancun, Quintana Roo, Mexico.",
        prop_node,
        True,
        claim=f"The property is located in Cancun, Quintana Roo, Mexico. Location provided: '{addr_loc}'.",
        sources=list_urls if list_urls else None,
        add_ins="Verify from the listing/source that the property is indeed in Cancun, Quintana Roo. Neighborhood-level confirmation is acceptable.",
    )

    # p2_residential_short_term_suitable (critical)
    rtype = (p2.residential_type if p2 else None) or "residential property suitable for short-term vacation rental operations (villa/condo/apartment/house)"
    await add_and_verify(
        evaluator,
        "p2_residential_short_term_suitable",
        "Property is residential real estate suitable for short-term vacation rental operations (villa/condo/apartment/house).",
        prop_node,
        True,
        claim=f"The property is a {rtype} and is suitable for short-term vacation rentals.",
        sources=list_urls if list_urls else None,
        add_ins="Check listing language indicating residential type and suitability for vacation/short-term rental.",
    )

    # p2_operating_or_furnished_ready (critical)
    or_claim = (p2.operating_or_ready_claim if p2 else None) or "The property is currently operating as a vacation rental or is furnished and ready for vacation rental use."
    await add_and_verify(
        evaluator,
        "p2_operating_or_furnished_ready",
        "Property is currently operating as a vacation rental OR is furnished and ready for vacation rental use (supported by the listing/source).",
        prop_node,
        True,
        claim=or_claim,
        sources=list_urls if list_urls else None,
        add_ins="Look for phrases like 'currently operating as a vacation rental', 'Airbnb-ready', 'furnished', 'turn-key', or similar in the listing/source.",
    )

    # p2_gross_yield_at_least_8pct (critical) - calculation/logical
    yield_pct = parse_percent_to_float(p2.gross_yield_percent if p2 else None)
    ann_gross = parse_money_to_float(p2.annual_gross_rental_income if p2 else None)
    price_val = parse_money_to_float(p2.asking_price_or_market_value if p2 else None)
    calc_yield = None
    if ann_gross and price_val and price_val > 0:
        calc_yield = 100.0 * (ann_gross / price_val)
    if yield_pct is not None:
        yield_claim_text = f"The gross rental yield is approximately {yield_pct:.2f}% which is at least 8%."
    elif calc_yield is not None:
        yield_claim_text = f"Based on annual gross rental income {p2.annual_gross_rental_income} and price {p2.asking_price_or_market_value}, gross yield is about {calc_yield:.2f}%, which is at least 8%."
    else:
        yield_claim_text = "The property can realistically achieve a gross rental yield of at least 8% based on provided market data/assumptions."

    await add_and_verify(
        evaluator,
        "p2_gross_yield_at_least_8pct",
        "Demonstrates or argues realistic potential for gross rental yield ≥ 8% (includes the yield figure and basis; cites at least one market-data/benchmark source as support).",
        prop_node,
        True,
        claim=yield_claim_text,
        sources=None,  # Allow calculation/logical verification from the answer
        add_ins="Use numbers and rationale in the answer. If the answer cites market benchmarks, consider them. Minor rounding is acceptable.",
    )

    # p2_occupancy_at_least_40pct (critical)
    occ = parse_percent_to_float(p2.occupancy_percent if p2 else None)
    if occ is not None:
        occ_claim_text = f"The average occupancy rate is approximately {occ:.2f}%, which is at least 40%."
    else:
        occ_claim_text = "The property can achieve an average occupancy rate of at least 40% based on Cancun STR benchmarks."
    await add_and_verify(
        evaluator,
        "p2_occupancy_at_least_40pct",
        "Demonstrates or argues realistic potential for average occupancy ≥ 40% (includes the occupancy assumption/figure and cites a market benchmark/source).",
        prop_node,
        True,
        claim=occ_claim_text,
        sources=None,  # Calculation/assumption check from the answer
        add_ins="Judge based on the provided assumption/benchmark in the answer. Minor rounding acceptable.",
    )

    # p2_management_fee_25_30_accounted (critical)
    mgmt = parse_percent_to_float(p2.management_fee_percent if p2 else None)
    if mgmt is not None:
        mgmt_claim_text = f"The financial analysis accounts for a property management fee of approximately {mgmt:.2f}% of rental income, within the 25–30% range."
    else:
        mgmt_claim_text = "The financial analysis accounts for property management fees of approximately 25–30% of rental income."
    await add_and_verify(
        evaluator,
        "p2_management_fee_25_30_accounted",
        "Financial analysis accounts for property management fees of approximately 25–30% of rental income (states a fee in that range or shows it in the analysis).",
        prop_node,
        True,
        claim=mgmt_claim_text,
        sources=None,
        add_ins="Check that the analysis explicitly includes a management fee in the ~25–30% range.",
    )

    # p2_required_outputs (critical group)
    req_node = evaluator.add_parallel(
        id="p2_required_outputs",
        desc="Required deliverables for Property 2 are provided.",
        parent=prop_node,
        critical=True,
    )

    await add_and_verify(
        evaluator,
        "p2_address_or_specific_cancun_location",
        "Provides property address or specific location within Cancun.",
        req_node,
        True,
        claim=f"Property 2 location within Cancun: '{addr_loc}'.",
        sources=list_urls if list_urls else None,
        add_ins="Accept neighborhood/zone-level location if that is how the listing presents it.",
    )

    size_text = (p2.size_bed_bath_or_sqft if p2 else "") or ""
    await add_and_verify(
        evaluator,
        "p2_size_bed_bath_or_sqft",
        "Provides property size (bedrooms/bathrooms or square footage).",
        req_node,
        True,
        claim=f"Property 2 size is '{size_text}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm stated bedrooms/bathrooms or SF on the listing/source.",
    )

    price_text2 = (p2.asking_price_or_market_value if p2 else "") or ""
    await add_and_verify(
        evaluator,
        "p2_asking_price_or_market_value",
        "Provides current asking price or estimated market value.",
        req_node,
        True,
        claim=f"Property 2 price/market value is '{price_text2}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm price/value appears on the listing/source.",
    )

    evaluator.add_custom_node(
        result=any(is_valid_url(u) for u in list_urls),
        id="p2_listing_or_real_estate_source_url",
        desc="Provides a URL reference to the property listing or real estate source supporting the provided facts.",
        parent=req_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Property 3 verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_property_3(evaluator: Evaluator, root, p3: Optional[Property3Extraction]) -> None:
    prop_node = evaluator.add_parallel(
        id="property_3_midtown_west_retail",
        desc="Property 3 satisfies Midtown West location, proximity-to-NYRR requirement, retail designation, street-level requirement, rent expression/market-context requirement, and required outputs.",
        parent=root,
        critical=False,
    )

    list_urls = (p3.listing_urls if p3 else []) or []
    prox_urls = (p3.proximity_support_urls if p3 else []) or []
    all_prox_sources = to_unique_urls(list_urls, prox_urls)

    address = (p3.address if p3 else "") or ""
    sf_text = (p3.total_square_footage if p3 else "") or ""
    annual_rent_text = (p3.annual_rent_or_asking_rent if p3 else "") or ""
    rate_text = (p3.rate_per_sf_per_year if p3 else "") or ""

    # p3_in_midtown_west (critical)
    midtown_claim = (p3.midtown_west_claim if p3 else None) or "The property is located in Midtown West, Manhattan, New York."
    await add_and_verify(
        evaluator,
        "p3_in_midtown_west",
        "Property is located in the Midtown West neighborhood, Manhattan, New York.",
        prop_node,
        True,
        claim=midtown_claim,
        sources=list_urls if list_urls else None,
        add_ins="Confirm from the listing/address context that the property is in Midtown West. Minor neighborhood boundary interpretations acceptable.",
    )

    # p3_proximity_to_nyrr_hq_supported (critical)
    prox_claim = (p3.proximity_claim if p3 else None) or f"The property is within reasonable proximity to NYRR HQ at {NYRR_HQ_ADDRESS}."
    await add_and_verify(
        evaluator,
        "p3_proximity_to_nyrr_hq_supported",
        "Provides support for being within reasonable proximity to NYRR HQ at 156 West 56th Street (e.g., states a distance or walking time based on the two addresses, with a citation if available).",
        prop_node,
        True,
        claim=prox_claim,
        sources=all_prox_sources if all_prox_sources else None,
        add_ins="If a map/distance URL is provided, confirm. Otherwise, rely on the answer’s stated walking time/distance.",
    )

    # p3_designated_or_zoned_retail (critical)
    z_claim = (p3.zoning_or_designation_claim if p3 else None) or "The property is designated/marketed/zoned as retail commercial space."
    await add_and_verify(
        evaluator,
        "p3_designated_or_zoned_retail",
        "Property is designated/marketed/zoned as retail commercial space (supported by listing language or official documentation URL).",
        prop_node,
        True,
        claim=z_claim,
        sources=list_urls if list_urls else None,
        add_ins="Check listing/zoning language indicating retail usage.",
    )

    # p3_ground_floor_or_street_level (critical)
    gf_claim = (p3.ground_floor_or_street_level_claim if p3 else None) or "The retail space is ground floor or street-level."
    await add_and_verify(
        evaluator,
        "p3_ground_floor_or_street_level",
        "Property is ground floor or street-level retail space (supported by listing details).",
        prop_node,
        True,
        claim=gf_claim,
        sources=list_urls if list_urls else None,
        add_ins="Confirm from listing details/photos/floor info that the space is at street level/ground floor.",
    )

    # p3_rate_expressed_per_sf_per_year (critical)
    computed_rate = compute_rate_per_sf_per_year(annual_rent_text, sf_text)
    if rate_text:
        rate_claim = f"The asking rental rate is '{rate_text}' in $/SF/YR terms (either stated directly or derivable)."
    elif computed_rate is not None:
        rate_claim = (f"Based on the annual rent '{annual_rent_text}' and area '{sf_text}', "
                      f"the rental rate is approximately ${computed_rate:.2f} per SF per year.")
    else:
        rate_claim = "The asking rental rate can be expressed in $/SF/YR terms."

    await add_and_verify(
        evaluator,
        "p3_rate_expressed_per_sf_per_year",
        "Asking rental rate can be expressed in $/sf/year (either stated directly or derivable from annual rent and square footage, with the $/sf/year value provided).",
        prop_node,
        True,
        claim=rate_claim,
        sources=list_urls if list_urls else None,
        add_ins="If not stated directly, compute from annual rent ÷ square footage. Minor rounding differences acceptable.",
    )

    # p3_market_context_considers_trophy_reference (critical) - explanation in answer
    context_expl = (p3.market_context_explanation if p3 else None) or \
                   "The asking rental rate reflects current Midtown West conditions considering that Trophy Class A space is roughly $120–125/SF (2025), and this property’s class/location justifies the stated rate."
    await add_and_verify(
        evaluator,
        "p3_market_context_considers_trophy_reference",
        "Explains how the asking rental rate reflects current Midtown West market conditions while considering the provided Trophy Class A reference range (~$120–125/sf in 2025).",
        prop_node,
        True,
        claim=context_expl,
        sources=None,
        add_ins="Check that the answer provides a reasonable narrative tying the subject rate to market context and the trophy reference range.",
    )

    # p3_required_outputs (critical group)
    req_node = evaluator.add_parallel(
        id="p3_required_outputs",
        desc="Required deliverables for Property 3 are provided.",
        parent=prop_node,
        critical=True,
    )

    await add_and_verify(
        evaluator,
        "p3_complete_street_address",
        "Provides complete street address.",
        req_node,
        True,
        claim=f"The complete street address for Property 3 is '{address}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm the address appears on the listing/leasing source. Minor formatting differences acceptable.",
    )

    await add_and_verify(
        evaluator,
        "p3_total_square_footage",
        "Provides total square footage of the retail space.",
        req_node,
        True,
        claim=f"The total square footage for Property 3 is '{sf_text}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm SF appears on the listing/leasing source.",
    )

    await add_and_verify(
        evaluator,
        "p3_annual_rent_or_asking_rent",
        "Provides annual rent or asking rent amount.",
        req_node,
        True,
        claim=f"The annual rent or asking rent for Property 3 is '{annual_rent_text}'.",
        sources=list_urls if list_urls else None,
        add_ins="Confirm rent appears on the listing/leasing source.",
    )

    evaluator.add_custom_node(
        result=any(is_valid_url(u) for u in list_urls),
        id="p3_listing_or_leasing_info_url",
        desc="Provides a URL reference to the retail space listing or leasing information supporting the provided facts.",
        parent=req_node,
        critical=True,
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
        default_model=model,
    )

    # Extract structured information for all three properties
    extraction = await evaluator.extract(
        prompt=prompt_extract_real_estate(),
        template_class=RealEstateExtraction,
        extraction_name="real_estate_extraction",
    )

    # Ground truth/context info
    evaluator.add_ground_truth({
        "nyrr_hq_address": NYRR_HQ_ADDRESS,
        "notes": "City service territory check for Property 1 must reference Duke Energy Carolinas (DEC). Midtown West alignment and market context for Property 3 should consider trophy ~$120–125/SF (2025)."
    })

    # Verify each property subtree
    await verify_property_1(evaluator, root, extraction.property_1)
    await verify_property_2(evaluator, root, extraction.property_2)
    await verify_property_3(evaluator, root, extraction.property_3)

    # Return standardized summary
    return evaluator.get_summary()