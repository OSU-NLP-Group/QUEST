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
TASK_ID = "multi_city_str_multifamily_analysis"
TASK_DESCRIPTION = (
    "I am a real estate investor evaluating multi-family investment opportunities in three growing U.S. cities with different regulatory environments. "
    "For each of the following cities—Denver, Colorado; Austin, Texas; and Nashville, Tennessee—research and document the complete regulatory framework "
    "and investment feasibility criteria for operating a multi-family property (2-4 units) as a short-term rental. For each city, provide:\n\n"
    "1. Zoning Requirements: Identify at least two specific zoning designations that permit multi-family residential use (2-4 units), "
    "including the official zoning code designations and descriptions.\n\n"
    "2. Short-Term Rental Regulations: Document the city's short-term rental permit requirements, including permit types available; owner-occupancy requirements (if any); "
    "bedroom/unit limitations; annual cap on rental days (if applicable); minimum rental period requirements.\n\n"
    "3. Parking Requirements: Specify the minimum parking spaces required for multi-family residential properties with 2-4 units, "
    "including any variations based on proximity to public transit or location within specific districts.\n\n"
    "4. Occupancy Standards: Document the maximum occupancy limits per bedroom and per unit based on local code or state law.\n\n"
    "5. Safety and Building Code Requirements: Identify egress window requirements for bedrooms, including minimum dimensions for opening height, width, and net clear opening area.\n\n"
    "6. Financial Benchmarking: For each city, provide current median or average rental rates for 2-bedroom units; typical capitalization rates for multi-family investment properties; "
    "average cash-on-cash returns reported for rental investments.\n\n"
    "7. Flood Risk Assessment Tools: Provide the URL to access FEMA flood maps for properties in each city.\n\n"
    "8. Reference Documentation: For each requirement, provide direct URLs to official city zoning maps or GIS systems; city municipal codes for short-term rentals; "
    "building codes or residential standards; parking requirement ordinances; rental market data sources.\n\n"
    "Each city analysis must include at least 8 distinct reference URLs to official city government sources, state agency websites, or recognized real estate data providers. "
    "Present the findings in a structured format that allows comparison across all three cities to identify which location offers the most favorable regulatory environment "
    "and investment potential for a multi-family short-term rental property."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ReferenceEntry(BaseModel):
    url: Optional[str] = None
    category: Optional[str] = None  # e.g., zoning_map_gis, str_code, building_code, parking_ordinance, rental_market_data, investment_metric, fema_flood_maps


class ZoningDesignation(BaseModel):
    code: Optional[str] = None  # Official code designation (e.g., "RMF", "R2-A", etc.)
    description: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class STRRegulations(BaseModel):
    permit_types: List[str] = Field(default_factory=list)
    permit_types_sources: List[str] = Field(default_factory=list)

    owner_occupancy: Optional[str] = None  # Text describing requirement or "none"
    owner_occupancy_sources: List[str] = Field(default_factory=list)

    bedroom_unit_limits: Optional[str] = None
    bedroom_unit_limits_sources: List[str] = Field(default_factory=list)

    annual_day_cap: Optional[str] = None
    annual_day_cap_sources: List[str] = Field(default_factory=list)

    minimum_rental_period: Optional[str] = None
    minimum_rental_period_sources: List[str] = Field(default_factory=list)


class ParkingRequirements(BaseModel):
    base_minimum: Optional[str] = None
    base_minimum_sources: List[str] = Field(default_factory=list)

    variations: Optional[str] = None  # Transit or district-based variations (or "none")
    variations_sources: List[str] = Field(default_factory=list)


class OccupancyStandards(BaseModel):
    per_bedroom_limit: Optional[str] = None
    per_bedroom_sources: List[str] = Field(default_factory=list)

    per_unit_limit: Optional[str] = None
    per_unit_sources: List[str] = Field(default_factory=list)


class SafetyEgress(BaseModel):
    opening_height: Optional[str] = None
    opening_height_sources: List[str] = Field(default_factory=list)

    opening_width: Optional[str] = None
    opening_width_sources: List[str] = Field(default_factory=list)

    net_clear_opening_area: Optional[str] = None
    net_clear_opening_area_sources: List[str] = Field(default_factory=list)


class FinancialBenchmarking(BaseModel):
    two_br_rent: Optional[str] = None
    two_br_rent_sources: List[str] = Field(default_factory=list)

    cap_rate: Optional[str] = None
    cap_rate_sources: List[str] = Field(default_factory=list)

    cash_on_cash: Optional[str] = None
    cash_on_cash_sources: List[str] = Field(default_factory=list)

    data_timestamp_2024_2025: Optional[str] = None  # e.g., "As of Q3 2025" or "2024"
    data_timestamp_sources: List[str] = Field(default_factory=list)


class CityAnalysis(BaseModel):
    city_name: Optional[str] = None
    zoning: List[ZoningDesignation] = Field(default_factory=list)
    str_regulations: Optional[STRRegulations] = None
    parking: Optional[ParkingRequirements] = None
    occupancy: Optional[OccupancyStandards] = None
    safety_egress: Optional[SafetyEgress] = None
    financial: Optional[FinancialBenchmarking] = None
    fema_url: Optional[str] = None
    references: List[ReferenceEntry] = Field(default_factory=list)


class MultiCityExtraction(BaseModel):
    denver: Optional[CityAnalysis] = None
    austin: Optional[CityAnalysis] = None
    nashville: Optional[CityAnalysis] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_multi_city() -> str:
    return (
        "Extract, for each of the three cities (Denver, Colorado; Austin, Texas; Nashville, Tennessee), the structured information below exactly as presented "
        "in the answer and capture all cited URLs. Do not invent or infer facts not stated. Return null or empty lists when a field is missing.\n\n"
        "For each city, extract a CityAnalysis object with these fields:\n"
        "- city_name: The city name (e.g., 'Denver, CO').\n"
        "- zoning: An array of zoning designations. For each, include:\n"
        "  • code: Official zoning code designation string.\n"
        "  • description: A short description of the designation exactly as stated.\n"
        "  • source_urls: All URLs cited for this designation (zoning map/GIS or zoning code pages).\n"
        "- str_regulations: Short-term rental regulations:\n"
        "  • permit_types: List of license/permit types available by name.\n"
        "  • permit_types_sources: URLs cited for permit types.\n"
        "  • owner_occupancy: Text describing owner-occupancy/primary residence requirements, or 'none' if explicitly stated none.\n"
        "  • owner_occupancy_sources: URLs cited.\n"
        "  • bedroom_unit_limits: Text describing bedroom or unit limits for STRs, or 'none' if explicitly stated none.\n"
        "  • bedroom_unit_limits_sources: URLs cited.\n"
        "  • annual_day_cap: Text for any annual cap on STR rental days, or 'none' if explicitly stated none.\n"
        "  • annual_day_cap_sources: URLs cited.\n"
        "  • minimum_rental_period: Text for minimum rental period (e.g., '30 days'), or 'none' if explicitly stated none.\n"
        "  • minimum_rental_period_sources: URLs cited.\n"
        "- parking: Parking requirements:\n"
        "  • base_minimum: Minimum parking spaces applicable to 2–4 unit multifamily properties (text/rule), including scaling if stated.\n"
        "  • base_minimum_sources: URLs cited.\n"
        "  • variations: Variations by transit proximity or district (text), or 'none' if explicitly stated none.\n"
        "  • variations_sources: URLs cited.\n"
        "- occupancy: Occupancy standards:\n"
        "  • per_bedroom_limit: Max occupancy per bedroom (text), or rule in effect.\n"
        "  • per_bedroom_sources: URLs cited.\n"
        "  • per_unit_limit: Max occupancy per unit (text), or 'not specified' if explicitly noted.\n"
        "  • per_unit_sources: URLs cited.\n"
        "- safety_egress: Egress window requirements for bedrooms:\n"
        "  • opening_height: Minimum opening height (text/numeric with unit).\n"
        "  • opening_height_sources: URLs cited.\n"
        "  • opening_width: Minimum opening width.\n"
        "  • opening_width_sources: URLs cited.\n"
        "  • net_clear_opening_area: Minimum net clear opening area.\n"
        "  • net_clear_opening_area_sources: URLs cited.\n"
        "- financial: Financial benchmarking:\n"
        "  • two_br_rent: Current median/average rent for 2-bedroom units (text/figure).\n"
        "  • two_br_rent_sources: URLs cited (recognized providers like Zillow, Redfin, RentCafe, etc.).\n"
        "  • cap_rate: Typical multifamily cap rate(s) (text/figure).\n"
        "  • cap_rate_sources: URLs cited (recognized providers like CBRE, Marcus & Millichap, etc.).\n"
        "  • cash_on_cash: Average cash-on-cash return benchmarks (text/figure).\n"
        "  • cash_on_cash_sources: URLs cited.\n"
        "  • data_timestamp_2024_2025: Date or 'as of' statement indicating data/regulations are current in 2024–2025.\n"
        "  • data_timestamp_sources: URLs cited.\n"
        "- fema_url: URL to access FEMA flood maps usable for properties in the city.\n"
        "- references: Array of ReferenceEntry objects for all URLs used in the city analysis. For each:\n"
        "  • url: The full URL exactly as shown in the answer (plain or markdown).\n"
        "  • category: One of {zoning_map_gis, str_code, building_code, parking_ordinance, rental_market_data, investment_metric, fema_flood_maps} "
        "when identifiable from the answer. If not identifiable, set null.\n\n"
        "Special rules for URLs:\n"
        "- Extract only URLs explicitly present in the answer. Include full protocol (http/https). If missing, prepend http://.\n"
        "- Do not deduplicate; include all URLs as presented. If the same URL appears multiple times, include multiple entries in references.\n"
        "- If the answer references a source verbally without a URL, return null for that source.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        u_str = u.strip()
        if not u_str:
            continue
        if u_str not in seen:
            seen.add(u_str)
            result.append(u_str)
    return result


def _union_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    return _dedup_urls(combined)


def _extract_reference_urls(refs: List[ReferenceEntry]) -> List[str]:
    urls = []
    for r in refs:
        if r.url:
            urls.append(r.url)
    return _dedup_urls(urls)


def _categories_present(refs: List[ReferenceEntry]) -> set:
    cats = set()
    for r in refs:
        if r.category:
            cats.add(r.category.strip().lower())
    return cats


def _has_required_categories(refs: List[ReferenceEntry]) -> bool:
    cats = _categories_present(refs)
    required = {
        "zoning_map_gis",
        "str_code",
        "building_code",
        "parking_ordinance",
        "rental_market_data",
        "investment_metric",
        "fema_flood_maps",
    }
    return required.issubset(cats)


def _designation_list_str(desigs: List[ZoningDesignation]) -> str:
    parts = []
    for d in desigs:
        code = d.code or "unknown"
        desc = d.description or "unspecified"
        parts.append(f"{code}: {desc}")
    return "; ".join(parts) if parts else "none provided"


# --------------------------------------------------------------------------- #
# City verification logic                                                     #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city_id: str,
    city_desc: str,
    city: Optional[CityAnalysis],
) -> None:
    # Create city container node (critical)
    city_node = evaluator.add_parallel(
        id=city_id,
        desc=city_desc,
        parent=parent_node,
        critical=True,
    )

    # Defensive defaults
    city = city or CityAnalysis()

    # ------------------------ Zoning ------------------------------------- #
    zoning_node = evaluator.add_parallel(
        id=f"{city_id}_Zoning",
        desc="Zoning designations permitting 2–4 unit multifamily residential use.",
        parent=city_node,
        critical=True,
    )

    # Leaf 1: At least two designations provided
    zoning_count_ok = len(city.zoning) >= 2
    evaluator.add_custom_node(
        result=zoning_count_ok,
        id=f"{city_id}_At_Least_Two_Designations_Provided",
        desc="Provides ≥2 specific zoning designations that permit 2–4 unit multifamily residential use.",
        parent=zoning_node,
        critical=True,
    )

    # Leaf 2: Each designation has code, description, citation
    each_has_required_fields = all(
        (z.code and z.description and len(z.source_urls) > 0) for z in city.zoning
    )
    # Verification: Check sources are official and support the designation info
    zoning_sources_union = _union_sources(*[z.source_urls for z in city.zoning])
    zoning_leaf = evaluator.add_leaf(
        id=f"{city_id}_Each_Designation_Has_Code_Description_Citation",
        desc="For each zoning designation provided: includes official code designation, description, and a direct URL to an official zoning map/GIS or zoning code source.",
        parent=zoning_node,
        critical=True,
    )
    zoning_claim = (
        f"For the following zoning designations in {city.city_name or 'the city'}: {_designation_list_str(city.zoning)}, "
        f"the cited sources include official zoning map/GIS or zoning code pages that describe these designations."
    )
    await evaluator.verify(
        claim=zoning_claim,
        node=zoning_leaf,
        sources=zoning_sources_union,
        additional_instruction=(
            "Confirm that each provided URL is an official zoning map/GIS or zoning code page for the city and that it describes the listed codes. "
            "If any designation lacks code, description, or official citation, mark as incorrect."
        ),
    )

    # ------------------------ STR Regulations ---------------------------- #
    str_node = evaluator.add_parallel(
        id=f"{city_id}_STR_Regulations",
        desc="Short-term rental permit requirements and restrictions.",
        parent=city_node,
        critical=True,
    )

    # Permit types
    permit_leaf = evaluator.add_leaf(
        id=f"{city_id}_Permit_Types",
        desc="Documents STR permit/license types available (names/types as defined by the jurisdiction) with citation URL(s).",
        parent=str_node,
        critical=True,
    )
    permit_claim = (
        f"The city's STR permit/license types include: {', '.join(city.str_regulations.permit_types) if city.str_regulations and city.str_regulations.permit_types else 'none specified'}."
    )
    await evaluator.verify(
        claim=permit_claim,
        node=permit_leaf,
        sources=(city.str_regulations.permit_types_sources if city.str_regulations else []),
        additional_instruction="Verify the permit/license type names on the official municipal code, licensing pages, or STR regulation pages."
    )

    # Owner occupancy
    owner_leaf = evaluator.add_leaf(
        id=f"{city_id}_Owner_Occupancy",
        desc="States any owner-occupancy/primary-residence requirement (or explicitly states none) with citation URL(s).",
        parent=str_node,
        critical=True,
    )
    owner_req = (city.str_regulations.owner_occupancy if city.str_regulations else None) or "unspecified"
    owner_claim = f"The city has the following STR owner-occupancy/primary residence requirement: {owner_req}."
    await evaluator.verify(
        claim=owner_claim,
        node=owner_leaf,
        sources=(city.str_regulations.owner_occupancy_sources if city.str_regulations else []),
        additional_instruction="Check official STR regulations or municipal code to confirm the presence or absence of an owner-occupancy/primary residence requirement."
    )

    # Bedroom or unit limits
    bu_leaf = evaluator.add_leaf(
        id=f"{city_id}_Bedroom_or_Unit_Limits",
        desc="States any bedroom and/or unit limitations for STR operation (or explicitly states none) with citation URL(s).",
        parent=str_node,
        critical=True,
    )
    bu_text = (city.str_regulations.bedroom_unit_limits if city.str_regulations else None) or "unspecified"
    bu_claim = f"The city imposes the following bedroom/unit limitations for STRs: {bu_text}."
    await evaluator.verify(
        claim=bu_claim,
        node=bu_leaf,
        sources=(city.str_regulations.bedroom_unit_limits_sources if city.str_regulations else []),
        additional_instruction="Confirm any stated bedroom or unit limits for STRs on the official municipal code or STR program pages."
    )

    # Annual day cap
    cap_leaf = evaluator.add_leaf(
        id=f"{city_id}_Annual_Day_Cap",
        desc="States any annual cap on STR rental days (or explicitly states none) with citation URL(s).",
        parent=str_node,
        critical=True,
    )
    cap_text = (city.str_regulations.annual_day_cap if city.str_regulations else None) or "unspecified"
    cap_claim = f"The city annual cap on STR rental days is: {cap_text}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=(city.str_regulations.annual_day_cap_sources if city.str_regulations else []),
        additional_instruction="Confirm the presence or absence of an annual cap on STR days via official regulation pages."
    )

    # Minimum rental period
    min_leaf = evaluator.add_leaf(
        id=f"{city_id}_Minimum_Rental_Period",
        desc="States any minimum rental period requirement (or explicitly states none) with citation URL(s).",
        parent=str_node,
        critical=True,
    )
    min_text = (city.str_regulations.minimum_rental_period if city.str_regulations else None) or "unspecified"
    min_claim = f"The minimum rental period requirement for STRs is: {min_text}."
    await evaluator.verify(
        claim=min_claim,
        node=min_leaf,
        sources=(city.str_regulations.minimum_rental_period_sources if city.str_regulations else []),
        additional_instruction="Verify any minimum rental period requirement (e.g., 30 days) via official municipal code or STR regulation pages."
    )

    # ------------------------ Parking Requirements ----------------------- #
    park_node = evaluator.add_parallel(
        id=f"{city_id}_Parking_Requirements",
        desc="Minimum parking spaces required for 2–4 unit multifamily properties, including location/transit variations.",
        parent=city_node,
        critical=True,
    )

    base_leaf = evaluator.add_leaf(
        id=f"{city_id}_Base_Minimum_For_2_to_4_Units",
        desc="Specifies minimum parking requirement(s) applicable to 2–4 unit multifamily properties with citation URL(s).",
        parent=park_node,
        critical=True,
    )
    base_text = (city.parking.base_minimum if city.parking else None) or "unspecified"
    base_claim = f"The base minimum parking requirement for 2–4 unit multifamily properties is: {base_text}."
    await evaluator.verify(
        claim=base_claim,
        node=base_leaf,
        sources=(city.parking.base_minimum_sources if city.parking else []),
        additional_instruction="Confirm the minimum parking ratios or counts from official parking ordinance or development code pages."
    )

    var_leaf = evaluator.add_leaf(
        id=f"{city_id}_Transit_or_District_Variations",
        desc="Documents any parking requirement variations based on transit proximity or specific districts with citation URL(s).",
        parent=park_node,
        critical=True,
    )
    var_text = (city.parking.variations if city.parking else None) or "unspecified"
    var_claim = f"Parking requirement variations based on transit proximity or districts: {var_text}."
    await evaluator.verify(
        claim=var_claim,
        node=var_leaf,
        sources=(city.parking.variations_sources if city.parking else []),
        additional_instruction="Confirm any stated reductions or alternative standards in TOD districts or overlays on official ordinance/code pages."
    )

    # ------------------------ Occupancy Standards ------------------------ #
    occ_node = evaluator.add_parallel(
        id=f"{city_id}_Occupancy_Standards",
        desc="Maximum occupancy limits per bedroom and per unit.",
        parent=city_node,
        critical=True,
    )

    per_bed_leaf = evaluator.add_leaf(
        id=f"{city_id}_Per_Bedroom_Limit",
        desc="Documents maximum occupancy limit per bedroom with citation URL(s).",
        parent=occ_node,
        critical=True,
    )
    per_bed_text = (city.occupancy.per_bedroom_limit if city.occupancy else None) or "unspecified"
    per_bed_claim = f"The maximum occupancy limit per bedroom is: {per_bed_text}."
    await evaluator.verify(
        claim=per_bed_claim,
        node=per_bed_leaf,
        sources=(city.occupancy.per_bedroom_sources if city.occupancy else []),
        additional_instruction="Verify occupancy limits per bedroom from official housing code or state/local regulations."
    )

    per_unit_leaf = evaluator.add_leaf(
        id=f"{city_id}_Per_Unit_Limit",
        desc="Documents maximum occupancy limit per unit (or explicitly states not separately specified) with citation URL(s).",
        parent=occ_node,
        critical=True,
    )
    per_unit_text = (city.occupancy.per_unit_limit if city.occupancy else None) or "unspecified"
    per_unit_claim = f"The maximum occupancy limit per unit is: {per_unit_text}."
    await evaluator.verify(
        claim=per_unit_claim,
        node=per_unit_leaf,
        sources=(city.occupancy.per_unit_sources if city.occupancy else []),
        additional_instruction="Check the local code or standards for per-unit occupancy thresholds. If not separately specified, confirm that's stated."
    )

    # ------------------------ Safety / Egress ---------------------------- #
    safe_node = evaluator.add_parallel(
        id=f"{city_id}_Safety_Building_Code_Egress",
        desc="Egress window requirements for bedrooms: minimum opening height, width, and net clear opening area.",
        parent=city_node,
        critical=True,
    )

    h_leaf = evaluator.add_leaf(
        id=f"{city_id}_Opening_Height",
        desc="Provides minimum egress window opening height requirement with citation URL(s).",
        parent=safe_node,
        critical=True,
    )
    h_text = (city.safety_egress.opening_height if city.safety_egress else None) or "unspecified"
    h_claim = f"The minimum egress window opening height is: {h_text}."
    await evaluator.verify(
        claim=h_claim,
        node=h_leaf,
        sources=(city.safety_egress.opening_height_sources if city.safety_egress else []),
        additional_instruction="Verify bedroom egress height requirement on official building code or residential standards pages."
    )

    w_leaf = evaluator.add_leaf(
        id=f"{city_id}_Opening_Width",
        desc="Provides minimum egress window opening width requirement with citation URL(s).",
        parent=safe_node,
        critical=True,
    )
    w_text = (city.safety_egress.opening_width if city.safety_egress else None) or "unspecified"
    w_claim = f"The minimum egress window opening width is: {w_text}."
    await evaluator.verify(
        claim=w_claim,
        node=w_leaf,
        sources=(city.safety_egress.opening_width_sources if city.safety_egress else []),
        additional_instruction="Verify bedroom egress width requirement on official building code or residential standards pages."
    )

    a_leaf = evaluator.add_leaf(
        id=f"{city_id}_Net_Clear_Opening_Area",
        desc="Provides minimum net clear opening area requirement with citation URL(s).",
        parent=safe_node,
        critical=True,
    )
    a_text = (city.safety_egress.net_clear_opening_area if city.safety_egress else None) or "unspecified"
    a_claim = f"The minimum net clear opening area is: {a_text}."
    await evaluator.verify(
        claim=a_claim,
        node=a_leaf,
        sources=(city.safety_egress.net_clear_opening_area_sources if city.safety_egress else []),
        additional_instruction="Confirm the net clear opening area requirement via official code or standards."
    )

    # ------------------------ Financial Benchmarking --------------------- #
    fin_node = evaluator.add_parallel(
        id=f"{city_id}_Financial_Benchmarking",
        desc="Market metrics for investment feasibility.",
        parent=city_node,
        critical=True,
    )

    rent_leaf = evaluator.add_leaf(
        id=f"{city_id}_Two_BR_Rent",
        desc="Provides current median or average rental rate for 2-bedroom units with a direct data-source URL.",
        parent=fin_node,
        critical=True,
    )
    rent_text = (city.financial.two_br_rent if city.financial else None) or "unspecified"
    rent_claim = f"The current median/average rent for 2-bedroom units is: {rent_text}."
    await evaluator.verify(
        claim=rent_claim,
        node=rent_leaf,
        sources=(city.financial.two_br_rent_sources if city.financial else []),
        additional_instruction="Verify the rent figure from recognized real estate data providers (e.g., Zillow, Redfin, RentCafe) or reputable market reports."
    )

    cap_leaf = evaluator.add_leaf(
        id=f"{city_id}_Cap_Rate",
        desc="Provides typical capitalization rate(s) for multifamily investments with a direct data-source URL.",
        parent=fin_node,
        critical=True,
    )
    cap_text = (city.financial.cap_rate if city.financial else None) or "unspecified"
    cap_claim = f"Typical multifamily capitalization rate(s) for the city: {cap_text}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=(city.financial.cap_rate_sources if city.financial else []),
        additional_instruction="Confirm cap rate benchmarks from recognized investment market reports (e.g., CBRE, Marcus & Millichap)."
    )

    coc_leaf = evaluator.add_leaf(
        id=f"{city_id}_Cash_on_Cash",
        desc="Provides average cash-on-cash return benchmarks with a direct data-source URL.",
        parent=fin_node,
        critical=True,
    )
    coc_text = (city.financial.cash_on_cash if city.financial else None) or "unspecified"
    coc_claim = f"Average cash-on-cash return benchmarks: {coc_text}."
    await evaluator.verify(
        claim=coc_claim,
        node=coc_leaf,
        sources=(city.financial.cash_on_cash_sources if city.financial else []),
        additional_instruction="Confirm cash-on-cash benchmarks from credible investment data sources; reasonable ranges acceptable."
    )

    ts_leaf = evaluator.add_leaf(
        id=f"{city_id}_Data_Timestamp_2024_2025",
        desc="Indicates that regulatory and market data are current as of 2024–2025.",
        parent=fin_node,
        critical=True,
    )
    ts_text = (city.financial.data_timestamp_2024_2025 if city.financial else None) or "unspecified"
    ts_claim = f"The provided regulatory and market data are current as of 2024–2025: {ts_text}."
    ts_sources = (city.financial.data_timestamp_sources if city.financial else []) or _union_sources(
        *(city.financial.two_br_rent_sources if city.financial else []),
        *(city.financial.cap_rate_sources if city.financial else []),
        *(city.financial.cash_on_cash_sources if city.financial else []),
    )
    await evaluator.verify(
        claim=ts_claim,
        node=ts_leaf,
        sources=ts_sources,
        additional_instruction="Check the dates or 'as of' statements on the cited pages to ensure the data/regulations are current in 2024–2025."
    )

    # ------------------------ FEMA Flood Maps ---------------------------- #
    fema_node = evaluator.add_parallel(
        id=f"{city_id}_FEMA_Flood_Maps",
        desc="FEMA flood map access for the city.",
        parent=city_node,
        critical=True,
    )

    fema_leaf = evaluator.add_leaf(
        id=f"{city_id}_FEMA_URL",
        desc="Provides URL to access FEMA flood maps usable for properties in the city.",
        parent=fema_node,
        critical=True,
    )
    fema_url = city.fema_url or ""
    fema_claim = f"The following URL provides access to FEMA flood maps usable for properties in {city.city_name or 'the city'}: {fema_url}."
    await evaluator.verify(
        claim=fema_claim,
        node=fema_leaf,
        sources=fema_url,
        additional_instruction="Confirm that the URL leads to FEMA's map viewer or an official FEMA flood map resource (e.g., msc.fema.gov)."
    )

    # ------------------------ References and Source Quality -------------- #
    refs_node = evaluator.add_parallel(
        id=f"{city_id}_References_and_Source_Quality",
        desc="Reference documentation requirements for the city.",
        parent=city_node,
        critical=True,
    )

    # At least 8 distinct URLs
    distinct_urls = _extract_reference_urls(city.references)
    at_least_8 = len(distinct_urls) >= 8
    evaluator.add_custom_node(
        result=at_least_8,
        id=f"{city_id}_At_Least_8_Distinct_URLs",
        desc="Includes ≥8 distinct reference URLs for the city analysis.",
        parent=refs_node,
        critical=True,
    )

    # Required categories covered
    categories_covered = _has_required_categories(city.references)
    evaluator.add_custom_node(
        result=categories_covered,
        id=f"{city_id}_Required_Categories_Covered",
        desc="Includes direct URLs covering required categories: zoning map/GIS, STR municipal code/regulations, building/egress code or standards, parking ordinance, rental market data source(s), investment metric source(s), and FEMA flood maps.",
        parent=refs_node,
        critical=True,
    )

    # Allowed source types only
    allowed_leaf = evaluator.add_leaf(
        id=f"{city_id}_Allowed_Source_Types_Only",
        desc="All URLs are from official city government sites, state agencies, FEMA.gov, or recognized real estate data providers.",
        parent=refs_node,
        critical=True,
    )
    allowed_claim = (
        "All listed URLs for this city are from allowed source types: official city government sites, state agencies, FEMA.gov, "
        "or recognized real estate data providers (e.g., NAR, Zillow, Redfin, CBRE, Marcus & Millichap, RentCafe)."
    )
    await evaluator.verify(
        claim=allowed_claim,
        node=allowed_leaf,
        sources=distinct_urls,
        additional_instruction="Evaluate domains and page content to ensure each URL belongs to the allowed source types. If any URL is not from an allowed source, mark as incorrect."
    )


# --------------------------------------------------------------------------- #
# Cross-city comparison verification                                          #
# --------------------------------------------------------------------------- #
async def verify_cross_city_comparison(evaluator: Evaluator, parent_node) -> None:
    cc_node = evaluator.add_parallel(
        id="Cross_City_Comparison",
        desc="Structured comparison across all three cities and identification of the most favorable location.",
        parent=parent_node,
        critical=True,
    )

    # Comparable structure leaf
    cs_leaf = evaluator.add_leaf(
        id="Comparable_Structure",
        desc="Presents results in a structured, comparable format across Denver, Austin, and Nashville (e.g., consistent sections or a comparison table).",
        parent=cc_node,
        critical=True,
    )
    cs_claim = (
        "The answer presents Denver, Austin, and Nashville side-by-side in a comparable structure (consistent sections or a comparison table) "
        "sufficient to compare regulatory and investment feasibility."
    )
    await evaluator.verify(
        claim=cs_claim,
        node=cs_leaf,
        additional_instruction="Assess the answer formatting and section consistency across the three cities. Allow reasonable variations as long as direct comparison is feasible."
    )

    # Most favorable city selected with rationale leaf
    mf_leaf = evaluator.add_leaf(
        id="Most_Favorable_City_Selected_With_Rationale",
        desc="Explicitly identifies which city is most favorable for a 2–4 unit multifamily STR and explains why using the documented regulatory and financial factors.",
        parent=cc_node,
        critical=True,
    )
    mf_claim = (
        "The answer explicitly identifies which of Denver, Austin, or Nashville is most favorable for a 2–4 unit multifamily STR, "
        "and provides a clear rationale using documented regulatory and financial factors (e.g., STR rules, parking, occupancy, safety codes, rent, cap rate, cash-on-cash, flood risk)."
    )
    await evaluator.verify(
        claim=mf_claim,
        node=mf_leaf,
        additional_instruction="Confirm both the explicit selection of one city and the rationale referencing the documented factors. Generic or missing rationale should be judged incorrect."
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
    Evaluate the multi-city STR multifamily analysis answer using the Mind2Web2 evaluation framework.
    """
    # 1) Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across city analyses and comparison
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

    # 2) Top-level critical container (to mirror rubric root which is critical)
    top_node = evaluator.add_parallel(
        id="Multi_City_STR_Multifamily_Analysis",
        desc="Regulatory and investment feasibility analysis for operating a 2–4 unit multifamily property as a short-term rental in Denver, Austin, and Nashville, with citations and a cross-city comparison.",
        parent=root,
        critical=True,
    )

    # 3) Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_multi_city(),
        template_class=MultiCityExtraction,
        extraction_name="multi_city_structured",
    )

    # 4) Verify each city according to the rubric
    await verify_city(
        evaluator=evaluator,
        parent_node=top_node,
        city_id="Denver_CO",
        city_desc="All required regulatory + financial attributes for Denver, Colorado.",
        city=extraction.denver,
    )

    await verify_city(
        evaluator=evaluator,
        parent_node=top_node,
        city_id="Austin_TX",
        city_desc="All required regulatory + financial attributes for Austin, Texas.",
        city=extraction.austin,
    )

    await verify_city(
        evaluator=evaluator,
        parent_node=top_node,
        city_id="Nashville_TN",
        city_desc="All required regulatory + financial attributes for Nashville, Tennessee.",
        city=extraction.nashville,
    )

    # 5) Cross-city comparison checks
    await verify_cross_city_comparison(evaluator, top_node)

    # 6) Return structured evaluation summary
    return evaluator.get_summary()