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
TASK_ID = "cle_mmud1_mixed_use_property"
TASK_DESCRIPTION = (
    "Identify a specific property currently available for development or redevelopment within Cleveland, Ohio's "
    "Midtown Mixed-Use District MMUD-1 sub-area that could accommodate a mixed-use residential and retail project. "
    "Your answer must include: (1) Property Identification: Provide the street address of the property and confirm "
    "it is located within the MMUD-1 sub-area boundaries. (2) Development Program: Propose a development program that "
    "includes residential units (apartments or townhouses) as a primary use, includes ground-level retail and/or "
    "tenant amenity spaces, meets the MMUD-1 requirement that at least 60% of the ground floor area be designated as "
    "retail, day care, or similar tenant/resident amenity or service uses, and if any proposed uses require mixed-use "
    "status per MMUD-1 regulations (Schedule 344.04, note 2), ensures that other uses account for at least 50% of total "
    "building square footage. All uses must be permitted by right (P), conditional (C), or accessory (A) in MMUD-1 "
    "according to Schedule 344.04. (3) Height Compliance: Identify the height district designation (1-9) applicable to "
    "the property, state the corresponding maximum allowable building height, and confirm that your proposed building "
    "height complies with this limit. (4) Financing Structure: Provide a basic financing structure showing the proposed "
    "loan-to-value (LTV) ratio and confirm it falls within typical commercial real estate lending standards (65-80%). "
    "(5) References: Provide URL references to Cleveland zoning code sections confirming MMUD-1 requirements (Chapter 344), "
    "property listing or assessment record, height district map or documentation, and any other relevant regulatory sources. "
    "All facts, figures, and property information must be grounded in actual, verifiable sources with URL references provided."
)

# Constants for rules (used as GT/hard checks where appropriate)
GROUND_FLOOR_MIN_PCT = 60.0
MIXED_USE_OTHER_MIN_PCT = 50.0
LTV_MIN = 65.0
LTV_MAX = 80.0


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AreaBreakdownItem(BaseModel):
    category: Optional[str] = None  # e.g., "retail", "amenity", "residential"
    area_sqft: Optional[str] = None  # e.g., "12,000 sf"
    percent: Optional[str] = None  # e.g., "65%"


class UseEntry(BaseModel):
    name: Optional[str] = None  # e.g., "Residential (apartments)", "Retail", "Day care"
    classification: Optional[str] = None  # One of "P", "C", "A" if stated
    requires_mixed_use_status: Optional[bool] = None  # If the use requires mixed-use status per note 2


class PropertyInfo(BaseModel):
    address: Optional[str] = None
    mmud1_urls: List[str] = Field(default_factory=list)  # URLs confirming property is within MMUD-1
    property_reference_urls: List[str] = Field(default_factory=list)  # Listing or assessment links (not explicitly scored here)


class DevelopmentProgram(BaseModel):
    # Uses and permitted classification references
    uses: List[UseEntry] = Field(default_factory=list)
    schedule_344_urls: List[str] = Field(default_factory=list)

    # Ground floor composition and references
    ground_floor_breakdown: List[AreaBreakdownItem] = Field(default_factory=list)
    calculated_retail_amenity_pct: Optional[str] = None  # e.g., "62%"
    ground_floor_requirement_urls: List[str] = Field(default_factory=list)

    # Mixed-use (note 2) composition and references
    requires_mixed_use: Optional[bool] = None
    building_area_breakdown: List[AreaBreakdownItem] = Field(default_factory=list)
    calculated_other_uses_pct: Optional[str] = None  # e.g., "52%"
    mixed_use_note2_urls: List[str] = Field(default_factory=list)

    # Pedestrian-oriented design (non-critical in rubric, handled as non-critical root-level leaf)
    pedestrian_features: Optional[str] = None
    pedestrian_feature_urls: List[str] = Field(default_factory=list)


class HeightInfo(BaseModel):
    height_district: Optional[str] = None  # "1" - "9"
    max_height_ft: Optional[str] = None  # e.g., "115 ft"
    proposed_height_ft: Optional[str] = None  # e.g., "110 ft"
    height_reference_urls: List[str] = Field(default_factory=list)  # Height district map or code references


class FinancingInfo(BaseModel):
    loan_amount: Optional[str] = None  # e.g., "$20,000,000"
    property_value: Optional[str] = None  # e.g., "$30,000,000"
    stated_ltv_pct: Optional[str] = None  # e.g., "67%"
    ltv_reference_urls: List[str] = Field(default_factory=list)  # Links confirming typical LTV ranges
    current_rate_assumption: Optional[str] = None  # e.g., "6%"
    rate_reference_urls: List[str] = Field(default_factory=list)  # Links supporting ~6% as of Mar 2026


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_property_info() -> str:
    return """
    Extract the property identification information from the answer.

    Required fields:
    - address: The complete street address of the property (include number, street, and city if provided; at minimum the street address).
    - mmud1_urls: A list of URL(s) in the answer that specifically confirm the property's location within the MMUD-1 (MidTown Mixed-Use District, Euclid Corridor Development Sub-Area) boundaries. These may include official zoning maps, planning maps, or city resources. Only include URLs explicitly present in the answer.
    - property_reference_urls: A list of URL(s) to a property listing, assessor/parcel record, or similar site-specific documentation mentioned in the answer.

    If any field is missing, return null for the field or an empty list as appropriate.
    """


def prompt_extract_development_program() -> str:
    return """
    Extract the development program details and zoning use conformance evidence from the answer.

    Required fields:
    - uses: Array of objects. For each proposed use, include:
        • name: The use name (e.g., "Residential (apartments)", "Retail", "Day care", "Amenity", "Parking", etc.)
        • classification: The classification in MMUD-1, if stated, as one of "P" (permitted), "C" (conditional), "A" (accessory).
        • requires_mixed_use_status: true/false if the use requires mixed-use status per Schedule 344.04 note 2 (if the answer states this).
    - schedule_344_urls: URL(s) to Cleveland Zoning Code Chapter 344 Schedule 344.04 that the answer cites for permitted uses.

    - ground_floor_breakdown: Array with the ground floor area by use type. Each item:
        • category: e.g., "retail", "day care", "amenity", "residential lobby", "other"
        • area_sqft: area string if given (e.g., "8,000 sf"); null if not given
        • percent: percent string if given (e.g., "65%"); null if not given
    - calculated_retail_amenity_pct: If the answer provides it, the calculated percentage of the ground floor designated as retail/day care/amenity/service uses.

    - ground_floor_requirement_urls: URL(s) the answer cites confirming the 60% requirement for residential projects in MMUD-1 (can be within Chapter 344).

    - requires_mixed_use: If the answer states that some proposed uses require mixed-use status per note 2, return true; otherwise false or null.
    - building_area_breakdown: Array with the total building area by use type. Same format as ground floor breakdown.
    - calculated_other_uses_pct: If provided, the calculated percentage of total building area constituted by “other uses” (i.e., non-primary use) to satisfy note 2.
    - mixed_use_note2_urls: URL(s) to Schedule 344.04 (note 2) or equivalent confirming the 50% mixed-use requirement.

    - pedestrian_features: List or text describing pedestrian-oriented design elements if mentioned (e.g., ground-level retail frontages, zero setback, active uses, pedestrian amenities). If not present, null.
    - pedestrian_feature_urls: Any supporting URLs for pedestrian-oriented features, if present.

    Only extract URLs explicitly present in the answer text. If any field is missing, return null or an empty list as appropriate.
    """


def prompt_extract_height_info() -> str:
    return """
    Extract the height compliance details from the answer.

    Required fields:
    - height_district: The numeric height district (1-9) identified for the property location.
    - max_height_ft: The maximum allowable height for that district, in feet, if stated (e.g., "115 ft").
    - proposed_height_ft: The proposed building height, in feet, if stated (e.g., "110 ft").
    - height_reference_urls: URL(s) the answer cites that confirm the height district designation and/or the height limits (e.g., city height map, height code section).

    If any field is missing, return null or an empty list as appropriate.
    """


def prompt_extract_financing_info() -> str:
    return """
    Extract the financing structure details from the answer.

    Required fields:
    - loan_amount: The proposed loan amount (e.g., "$20,000,000" or "20M").
    - property_value: The total property value or stabilized value used for LTV (e.g., "$30,000,000").
    - stated_ltv_pct: The stated LTV ratio (e.g., "67%") if provided.
    - ltv_reference_urls: URL(s) cited by the answer that confirm typical commercial LTV standards (e.g., 65-80%).
    - current_rate_assumption: The interest rate assumption used, if mentioned (e.g., "6%").
    - rate_reference_urls: URL(s) supporting the current rate environment claim (approximately 6% as of March 2026).

    If any field is missing, return null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions: parsing and calculations                                  #
# --------------------------------------------------------------------------- #
def _first_number(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_percent(p: Optional[str]) -> Optional[float]:
    if p is None:
        return None
    s = p.strip().lower().replace("percent", "").replace("per cent", "")
    num = _first_number(s)
    if num is None:
        return None
    if "%" in s:
        return num
    # Treat small fractions as ratio (e.g., 0.6 -> 60%)
    return num * 100.0 if num <= 1.0 else num


def parse_money(m: Optional[str]) -> Optional[float]:
    if m is None:
        return None
    s = m.strip().lower().replace(",", "").replace("$", "").replace("usd", "").strip()
    mult = 1.0
    # Detect billions/millions/thousands
    if "billion" in s or s.endswith("b"):
        s = s.replace("billion", "").replace("b", "").strip()
        mult = 1_000_000_000.0
    elif "million" in s or s.endswith("m"):
        s = s.replace("million", "").replace("m", "").strip()
        mult = 1_000_000.0
    elif "thousand" in s or s.endswith("k"):
        s = s.replace("thousand", "").replace("k", "").strip()
        mult = 1_000.0
    num = _first_number(s)
    return None if num is None else num * mult


def parse_height_ft(h: Optional[str]) -> Optional[float]:
    if h is None:
        return None
    s = h.strip().lower().replace("feet", "ft")
    # If explicitly stories, skip conversion
    if "stor" in s:
        return None
    num = _first_number(s)
    return num


def sum_pct_for_categories(breakdown: List[AreaBreakdownItem], positive_keywords: List[str]) -> Optional[float]:
    # Try percent-based first
    total_pct = 0.0
    had_pct = False
    for item in breakdown:
        cat = (item.category or "").lower()
        if any(k in cat for k in positive_keywords):
            pct = parse_percent(item.percent) if item.percent else None
            if pct is not None:
                had_pct = True
                total_pct += pct
    if had_pct:
        return total_pct

    # Try area-based if percents absent
    total_area = 0.0
    pos_area = 0.0
    for item in breakdown:
        a = item.area_sqft or ""
        av = _first_number(a.replace(",", "")) if a else None
        if av is not None:
            total_area += av
            cat = (item.category or "").lower()
            if any(k in cat for k in positive_keywords):
                pos_area += av
    if total_area > 0:
        return (pos_area / total_area) * 100.0

    return None


def get_other_uses_pct(building_breakdown: List[AreaBreakdownItem], fallback_pct: Optional[str]) -> Optional[float]:
    # Prefer an explicitly calculated percentage if provided
    if fallback_pct:
        pv = parse_percent(fallback_pct)
        if pv is not None:
            return pv

    # Otherwise, attempt to infer "other uses" by matching category keywords that indicate "other"/"non-primary"
    # Here we simply look for categories that contain "other" in the label.
    total_area = 0.0
    other_area = 0.0
    have_any_area = False
    for item in building_breakdown:
        a = item.area_sqft or ""
        av = _first_number(a.replace(",", "")) if a else None
        if av is not None:
            have_any_area = True
            total_area += av
            cat = (item.category or "").lower()
            if "other" in cat or "non-primary" in cat or "secondary" in cat:
                other_area += av
    if have_any_area and total_area > 0:
        return (other_area / total_area) * 100.0

    # Try percent-based: sum items labeled "other"
    total_pct = 0.0
    other_pct = 0.0
    had_pct = False
    for item in building_breakdown:
        pct = parse_percent(item.percent) if item.percent else None
        if pct is not None:
            had_pct = True
            total_pct += pct
            cat = (item.category or "").lower()
            if "other" in cat or "non-primary" in cat or "secondary" in cat:
                other_pct += pct
    if had_pct and total_pct > 0:
        return other_pct  # Already in percent units

    return None


def format_use_classifications(uses: List[UseEntry]) -> str:
    parts = []
    for u in uses:
        name = (u.name or "").strip()
        cls = (u.classification or "").strip()
        if name:
            if cls:
                parts.append(f"{name} = {cls}")
            else:
                parts.append(f"{name} = (unspecified)")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_property_identification(
    evaluator: Evaluator,
    parent_node,
    prop: PropertyInfo,
):
    # property_identification (critical, sequential)
    prop_node = evaluator.add_sequential(
        id="property_identification",
        desc="Property is correctly identified as located within Cleveland, Ohio's MMUD-1 sub-area",
        parent=parent_node,
        critical=True,
    )

    # location_specification (critical, parallel)
    loc_spec = evaluator.add_parallel(
        id="location_specification",
        desc="Property address and location details are provided",
        parent=prop_node,
        critical=True,
    )

    # address_provision (critical leaf via custom presence)
    address_ok = bool(prop and prop.address and prop.address.strip())
    evaluator.add_custom_node(
        result=address_ok,
        id="address_provision",
        desc="Complete street address of the property is provided",
        parent=loc_spec,
        critical=True,
    )

    # mmud1_boundary_verification (critical leaf, verify by URLs)
    mmud_leaf = evaluator.add_leaf(
        id="mmud1_boundary_verification",
        desc="Property location is confirmed to be within MMUD-1 (Euclid Corridor Development Sub-Area) boundaries",
        parent=loc_spec,
        critical=True,
    )
    addr = prop.address or "the subject property"
    mmud_claim = f"The property at {addr} is within the Cleveland Midtown Mixed-Use District MMUD-1 sub-area boundaries."
    await evaluator.verify(
        claim=mmud_claim,
        node=mmud_leaf,
        sources=prop.mmud1_urls,
        additional_instruction="Use the provided map/code sources to confirm the parcel/address lies inside the MMUD-1 sub-area boundary.",
    )

    # location_reference (critical leaf -> presence of MMUD-1 confirming URL)
    evaluator.add_custom_node(
        result=bool(prop.mmud1_urls),
        id="location_reference",
        desc="Provide URL reference confirming the property's location within MMUD-1",
        parent=prop_node,
        critical=True,
    )


async def verify_development_program_compliance(
    evaluator: Evaluator,
    parent_node,
    dev: DevelopmentProgram,
):
    # IMPORTANT: JSON marks this node critical and includes a non-critical child (pedestrian_oriented_design).
    # The verification tree enforces: critical parent cannot have non-critical child.
    # We therefore attach only critical compliance checks here, and handle pedestrian features as a separate
    # non-critical node under the global root.
    dev_node = evaluator.add_parallel(
        id="development_program_compliance",
        desc="Proposed development program meets MMUD-1 use and composition requirements",
        parent=parent_node,
        critical=True,
    )

    # zoning_use_compliance (critical, parallel)
    zoning_node = evaluator.add_parallel(
        id="zoning_use_compliance",
        desc="Development program complies with MMUD-1 zoning use requirements including permitted uses, ground floor composition, and mixed-use definitions",
        parent=dev_node,
        critical=True,
    )

    # permitted_uses (critical, sequential)
    permitted_node = evaluator.add_sequential(
        id="permitted_uses",
        desc="All proposed uses are permitted in MMUD-1 according to Schedule 344.04",
        parent=zoning_node,
        critical=True,
    )

    # use_identification_and_classification (critical, parallel)
    use_ic_node = evaluator.add_parallel(
        id="use_identification_and_classification",
        desc="Proposed uses are identified and classified according to Schedule 344.04",
        parent=permitted_node,
        critical=True,
    )

    # use_listing (critical presence)
    evaluator.add_custom_node(
        result=bool(dev.uses),
        id="use_listing",
        desc="All proposed uses in the development program are explicitly listed",
        parent=use_ic_node,
        critical=True,
    )

    # use_classification (critical verification with Schedule 344.04 sources)
    use_class_leaf = evaluator.add_leaf(
        id="use_classification",
        desc="Each proposed use is classified as permitted by right (P), conditional (C), or accessory (A) per Schedule 344.04",
        parent=use_ic_node,
        critical=True,
    )
    pairs = format_use_classifications(dev.uses)
    use_class_claim = (
        "According to Cleveland Zoning Code Chapter 344, Schedule 344.04 (MMUD-1), "
        "the proposed uses are permitted with the following classifications (P/C/A): "
        f"{pairs}. Treat reasonable synonyms (e.g., 'dwelling units' ≈ residential) and confirm they map correctly."
    )
    await evaluator.verify(
        claim=use_class_claim,
        node=use_class_leaf,
        sources=dev.schedule_344_urls,
        additional_instruction="Confirm each listed use appears in Schedule 344.04 for MMUD-1 with the stated P/C/A classification.",
    )

    # use_reference (critical presence)
    evaluator.add_custom_node(
        result=bool(dev.schedule_344_urls),
        id="use_reference",
        desc="Provide URL reference to Cleveland zoning code Schedule 344.04 confirming permitted use classifications",
        parent=permitted_node,
        critical=True,
    )

    # residential_ground_floor_requirement (critical, sequential)
    gfr_node = evaluator.add_sequential(
        id="residential_ground_floor_requirement",
        desc="If the project includes residential units (apartment or townhouse), at least 60% of ground floor area is designated as retail, day care, or similar tenant/resident amenity or service",
        parent=zoning_node,
        critical=True,
    )

    # ground_floor_composition_analysis (critical, sequential)
    gfr_analysis = evaluator.add_sequential(
        id="ground_floor_composition_analysis",
        desc="Ground floor area composition is analyzed to verify 60% retail/amenity requirement",
        parent=gfr_node,
        critical=True,
    )

    # ground_floor_area_breakdown (critical presence)
    evaluator.add_custom_node(
        result=bool(dev.ground_floor_breakdown),
        id="ground_floor_area_breakdown",
        desc="Ground floor area is broken down by use type (retail, amenity, residential, other)",
        parent=gfr_analysis,
        critical=True,
    )

    # sixty_percent_calculation (critical custom check)
    retail_keywords = ["retail", "day care", "daycare", "amenity", "tenant amenity", "resident amenity", "service"]
    gf_pct = None
    # Prefer explicitly calculated value
    if dev.calculated_retail_amenity_pct:
        gf_pct = parse_percent(dev.calculated_retail_amenity_pct)
    if gf_pct is None:
        gf_pct = sum_pct_for_categories(dev.ground_floor_breakdown, retail_keywords)
    meets_60 = (gf_pct is not None) and (gf_pct >= GROUND_FLOOR_MIN_PCT)
    evaluator.add_custom_node(
        result=meets_60,
        id="sixty_percent_calculation",
        desc="Percentage of ground floor area designated as retail/day care/amenity is calculated and shown to be at least 60%",
        parent=gfr_analysis,
        critical=True,
    )

    # requirement_reference (critical verification using URLs)
    req_ref_leaf = evaluator.add_leaf(
        id="requirement_reference",
        desc="Provide URL reference to MMUD zoning code confirming the 60% ground floor requirement for residential projects",
        parent=gfr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="For residential projects in MMUD-1, at least 60% of the ground floor area must be designated as retail, day care, or similar tenant/resident amenity or service uses.",
        node=req_ref_leaf,
        sources=dev.ground_floor_requirement_urls,
        additional_instruction="Locate the specific clause or schedule note in Chapter 344 confirming this requirement.",
    )

    # mixed_use_project_definition (critical, sequential)
    mu_node = evaluator.add_sequential(
        id="mixed_use_project_definition",
        desc="If the project includes uses that require mixed-use status in MMUD-1 (per note 2 of Schedule 344.04), other uses account for at least 50% of total building square footage",
        parent=zoning_node,
        critical=True,
    )

    # mixed_use_composition_analysis (critical, sequential)
    mu_analysis = evaluator.add_sequential(
        id="mixed_use_composition_analysis",
        desc="Building composition is analyzed to verify 50% mixed-use requirement",
        parent=mu_node,
        critical=True,
    )

    # building_area_breakdown (critical presence or pass if not required)
    requires_mu = bool(dev.requires_mixed_use) or any(u.requires_mixed_use_status for u in dev.uses if u.requires_mixed_use_status is not None and u.requires_mixed_use_status)
    building_breakdown_ok = True
    if requires_mu:
        building_breakdown_ok = bool(dev.building_area_breakdown)
    evaluator.add_custom_node(
        result=building_breakdown_ok,
        id="building_area_breakdown",
        desc="Total building square footage is broken down by use type",
        parent=mu_analysis,
        critical=True,
    )

    # fifty_percent_calculation (critical; only enforced if mixed-use is required)
    other_pct = get_other_uses_pct(dev.building_area_breakdown, dev.calculated_other_uses_pct) if requires_mu else 100.0
    meets_50 = (other_pct is not None) and (other_pct >= MIXED_USE_OTHER_MIN_PCT)
    evaluator.add_custom_node(
        result=meets_50,
        id="fifty_percent_calculation",
        desc="Percentage of building area designated as 'other uses' (non-primary use) is calculated and shown to be at least 50%",
        parent=mu_analysis,
        critical=True,
    )

    # mixed_use_reference (critical verification using URLs)
    mu_ref_leaf = evaluator.add_leaf(
        id="mixed_use_reference",
        desc="Provide URL reference to Schedule 344.04 note 2 confirming the 50% mixed-use requirement",
        parent=mu_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Schedule 344.04 note 2 for MMUD-1 requires that if certain uses are permitted only in a mixed-use project, other uses must account for at least 50% of total building floor area.",
        node=mu_ref_leaf,
        sources=dev.mixed_use_note2_urls,
        additional_instruction="Verify the exact note text that establishes the 50% 'other uses' requirement in MMUD-1.",
    )


async def verify_height_compliance(
    evaluator: Evaluator,
    parent_node,
    prop: PropertyInfo,
    height: HeightInfo,
):
    height_node = evaluator.add_sequential(
        id="height_compliance",
        desc="Proposed building height complies with the applicable height district limitation for the property location",
        parent=parent_node,
        critical=True,
    )

    # height_district_analysis (critical, sequential)
    hd_analysis = evaluator.add_sequential(
        id="height_district_analysis",
        desc="Height district for the property is identified and maximum allowable height is determined",
        parent=height_node,
        critical=True,
    )

    # height_district_identification (critical verification by URLs)
    hd_ident_leaf = evaluator.add_leaf(
        id="height_district_identification",
        desc="The height district designation (1-9) for the property is correctly identified",
        parent=hd_analysis,
        critical=True,
    )
    addr = prop.address or "the subject property"
    hd = (height.height_district or "").strip()
    claim_hd = f"The property at {addr} is located in Height District {hd} in Cleveland."
    await evaluator.verify(
        claim=claim_hd,
        node=hd_ident_leaf,
        sources=height.height_reference_urls,
        additional_instruction="Use the provided map or code reference to confirm the numeric height district for this parcel/address.",
    )

    # maximum_height_determination (critical verification by URLs)
    max_h_leaf = evaluator.add_leaf(
        id="maximum_height_determination",
        desc="The maximum allowable height corresponding to the height district is correctly stated (District 1: 35', District 2: 60', District 3: 115', District 4: 175', District 5: 250', District 6: 600', District 7: 700', District 8: 800', District 9: 900')",
        parent=hd_analysis,
        critical=True,
    )
    max_ft = (height.max_height_ft or "").strip()
    claim_max = f"In Cleveland, Height District {hd} has a maximum allowable building height of {max_ft}."
    await evaluator.verify(
        claim=claim_max,
        node=max_h_leaf,
        sources=height.height_reference_urls,
        additional_instruction="Verify the numeric maximum height in feet corresponding to the stated Height District.",
    )

    # proposed_height_compliance (critical custom check)
    max_ft_val = parse_height_ft(height.max_height_ft)
    prop_ft_val = parse_height_ft(height.proposed_height_ft)
    height_ok = (max_ft_val is not None) and (prop_ft_val is not None) and (prop_ft_val <= max_ft_val + 1e-6)
    evaluator.add_custom_node(
        result=height_ok,
        id="proposed_height_compliance",
        desc="Proposed building height does not exceed the maximum allowable height for the identified height district",
        parent=height_node,
        critical=True,
    )

    # height_reference (critical presence)
    evaluator.add_custom_node(
        result=bool(height.height_reference_urls),
        id="height_reference",
        desc="Provide URL reference confirming the height district designation and maximum height limits",
        parent=height_node,
        critical=True,
    )


async def verify_financing_feasibility(
    evaluator: Evaluator,
    parent_node,
    fin: FinancingInfo,
):
    finance_node = evaluator.add_parallel(
        id="financing_feasibility",
        desc="Proposed financing structure is feasible and meets standard commercial real estate lending requirements",
        parent=parent_node,
        critical=True,
    )

    # loan_to_value_ratio (critical, sequential)
    ltv_node = evaluator.add_sequential(
        id="loan_to_value_ratio",
        desc="Proposed loan-to-value (LTV) ratio is within typical commercial real estate lending standards of 65-80%",
        parent=finance_node,
        critical=True,
    )

    # ltv_analysis (critical, sequential)
    ltv_analysis = evaluator.add_sequential(
        id="ltv_analysis",
        desc="LTV ratio is calculated and verified to be within acceptable range",
        parent=ltv_node,
        critical=True,
    )

    # ltv_calculation (critical custom check)
    loan_val = parse_money(fin.loan_amount)
    prop_val = parse_money(fin.property_value)
    stated_ltv = parse_percent(fin.stated_ltv_pct) if fin.stated_ltv_pct else None
    derived_ltv = (loan_val / prop_val * 100.0) if (loan_val is not None and prop_val and prop_val > 0) else None

    calc_ok = False
    if derived_ltv is not None and stated_ltv is not None:
        calc_ok = abs(derived_ltv - stated_ltv) <= 1.0  # within 1 percentage point
    elif derived_ltv is not None:
        # We can at least confirm a valid calculation exists
        calc_ok = True
    elif stated_ltv is not None:
        # Accept stated LTV if provided, even if inputs absent (we cannot compute)
        calc_ok = True

    evaluator.add_custom_node(
        result=calc_ok,
        id="ltv_calculation",
        desc="LTV ratio calculation is provided showing the ratio of loan amount to property value",
        parent=ltv_analysis,
        critical=True,
    )

    # ltv_range_verification (critical custom check)
    ltv_to_check = derived_ltv if derived_ltv is not None else stated_ltv
    in_range = (ltv_to_check is not None) and (LTV_MIN - 1e-6 <= ltv_to_check <= LTV_MAX + 1e-6)
    evaluator.add_custom_node(
        result=in_range,
        id="ltv_range_verification",
        desc="Calculated LTV ratio is confirmed to fall within the 65-80% range",
        parent=ltv_analysis,
        critical=True,
    )

    # ltv_reference (critical verification via URLs)
    ltv_ref_leaf = evaluator.add_leaf(
        id="ltv_reference",
        desc="Provide URL reference confirming typical commercial LTV ratio standards",
        parent=ltv_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Typical commercial real estate loan-to-value (LTV) standards are about {int(LTV_MIN)}% to {int(LTV_MAX)}%.",
        node=ltv_ref_leaf,
        sources=fin.ltv_reference_urls,
        additional_instruction="Confirm that mainstream references indicate typical CRE LTV standards around 65–80%.",
    )

    # NOTE: The rubric also contains a non-critical child 'current_rate_environment' under a critical parent.
    # This violates the framework constraint (critical parent cannot have non-critical child).
    # We therefore verify the rate environment as a separate non-critical node under the global root (see below).


async def verify_pedestrian_features(
    evaluator: Evaluator,
    root_node,
    dev: DevelopmentProgram,
):
    # Non-critical root-level leaf (moved out of critical parent to satisfy framework constraints)
    ped_leaf = evaluator.add_leaf(
        id="pedestrian_oriented_design",
        desc="Development plan includes pedestrian-oriented features such as ground-level retail, building placement near front property line, or pedestrian amenities",
        parent=root_node,
        critical=False,
    )
    features_text = dev.pedestrian_features or "The plan includes pedestrian-oriented features."
    await evaluator.verify(
        claim=features_text,
        node=ped_leaf,
        sources=dev.pedestrian_feature_urls if dev.pedestrian_feature_urls else None,
        additional_instruction="Verify the presence of pedestrian-oriented elements described in the answer using any provided URLs if available. Allow reasonable paraphrasing.",
    )


async def verify_rate_environment(
    evaluator: Evaluator,
    root_node,
    fin: FinancingInfo,
):
    # Non-critical root-level leaf (moved out of critical parent to satisfy framework constraints)
    rate_leaf = evaluator.add_leaf(
        id="current_rate_environment",
        desc="Financing assumptions reflect current mortgage rate environment (approximately 6% as of March 2026)",
        parent=root_node,
        critical=False,
    )
    rate_text = fin.current_rate_assumption or "Current CRE mortgage rates are around 6% as of March 2026."
    await evaluator.verify(
        claim=rate_text,
        node=rate_leaf,
        sources=fin.rate_reference_urls if fin.rate_reference_urls else None,
        additional_instruction="Verify that a ~6% mortgage rate assumption is reasonable for CRE loans around March 2026 from the provided source(s).",
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
    Evaluate the answer for the Cleveland MMUD-1 mixed-use property task using a verification tree.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation across sub-criteria
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

    # Create a critical task root node (child of global root). All rubric-critical sections attach here.
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify a viable mixed-use development property in Cleveland's Midtown Mixed-Use District (MMUD-1 sub-area) that satisfies all specified zoning, use composition, financing, and regulatory requirements",
        parent=root,
        critical=True,
    )

    # Record fixed rule thresholds in summary for transparency
    evaluator.add_ground_truth(
        {
            "ground_floor_min_pct": GROUND_FLOOR_MIN_PCT,
            "mixed_use_other_min_pct": MIXED_USE_OTHER_MIN_PCT,
            "ltv_range_pct": [LTV_MIN, LTV_MAX],
        },
        gt_type="rule_thresholds",
    )

    # Run extractions (can be parallelized)
    prop_task = evaluator.extract(
        prompt=prompt_extract_property_info(),
        template_class=PropertyInfo,
        extraction_name="property_info",
    )
    dev_task = evaluator.extract(
        prompt=prompt_extract_development_program(),
        template_class=DevelopmentProgram,
        extraction_name="development_program",
    )
    height_task = evaluator.extract(
        prompt=prompt_extract_height_info(),
        template_class=HeightInfo,
        extraction_name="height_info",
    )
    finance_task = evaluator.extract(
        prompt=prompt_extract_financing_info(),
        template_class=FinancingInfo,
        extraction_name="financing_info",
    )

    prop, dev, height, fin = await asyncio.gather(prop_task, dev_task, height_task, finance_task)

    # Build and run verification according to rubric
    await verify_property_identification(evaluator, task_root, prop)
    await verify_development_program_compliance(evaluator, task_root, dev)
    await verify_height_compliance(evaluator, task_root, prop, height)
    await verify_financing_feasibility(evaluator, task_root, fin)

    # Non-critical checks separated to satisfy critical-child constraint
    await verify_pedestrian_features(evaluator, root, dev)
    await verify_rate_environment(evaluator, root, fin)

    # Return structured evaluation summary
    return evaluator.get_summary()