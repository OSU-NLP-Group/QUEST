import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_sb840_office_to_residential_leed_gold"
TASK_DESCRIPTION = """
A real estate development firm is planning to convert an existing office building in Texas into a mixed-use residential development under the provisions of Texas Senate Bill 840 (SB 840), which became effective on September 1, 2025. The firm aims to achieve LEED Gold certification for the converted building and wants to take advantage of SB 840's streamlined approval process and regulatory benefits for qualifying conversions.

Identify a specific office building in a Texas municipality that meets all of the following requirements:

1. Jurisdictional Requirements: The building must be located in a Texas municipality with a population greater than 150,000 that is wholly or partly located in a county with a population greater than 300,000, making it subject to SB 840.

2. Building Eligibility: The building must have been constructed at least 5 years before 2026 (i.e., built in 2021 or earlier) and currently be used primarily for office purposes.

3. Zoning Compatibility: The building must be located in a zoning classification that allows office, commercial, retail, warehouse, or mixed-use development.

4. Exclusion Zones: The building must NOT be located within 1,000 feet of an existing heavy industrial use or development site, within 3,000 feet of an airport or military base, or in an area designated as a clear zone or accident potential zone.

5. Conversion Requirements: The conversion plan must allocate at least 65% of the building's total floor area AND at least 65% of each occupiable floor to residential use, as required for SB 840 conversion benefits.

6. Development Standards: The building's height, the proposed residential density, and the parking plan must comply with SB 840 standards:
   - Height: Must not exceed the greater of the equivalent commercial height limit or 45 feet
   - Density: Must not exceed the greater of the municipality's highest residential density or 36 units per acre
   - Parking: Must provide no more than 1 parking space per dwelling unit

7. LEED Gold Certification: The conversion project must be designed to achieve LEED Gold certification, requiring a score of 60-79 points on the LEED scorecard.

8. Permit Timing: The building permit application must be planned for submission on or after September 1, 2025.

Provide the name and address of the identified building, specify the Texas municipality and county where it is located, confirm the building's construction year and current use, identify its zoning classification, describe how it meets all SB 840 conversion requirements (including conversion percentage, height, density, and parking compliance), explain how it will achieve LEED Gold certification, and confirm it is not located in any exclusion zones. Include URL references supporting each verification.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    # Building identification
    building_name: Optional[str] = None
    address: Optional[str] = None
    municipality: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    constructed_year: Optional[str] = None
    current_primary_use: Optional[str] = None
    building_reference_urls: List[str] = Field(default_factory=list)

    # Geographic and legal eligibility (populations and statute references)
    municipality_population: Optional[str] = None
    municipality_population_urls: List[str] = Field(default_factory=list)
    county_population: Optional[str] = None
    county_population_urls: List[str] = Field(default_factory=list)
    sb840_applicability_urls: List[str] = Field(default_factory=list)

    # Zoning compatibility
    zoning_classification: Optional[str] = None
    zoning_permitted_uses: List[str] = Field(default_factory=list)
    zoning_urls: List[str] = Field(default_factory=list)

    # Exclusion zones
    heavy_industrial_distance_ft: Optional[str] = None
    airport_military_distance_ft: Optional[str] = None
    safety_zone_designation: Optional[str] = None  # e.g., "Not in clear zone/APZ"
    exclusion_urls: List[str] = Field(default_factory=list)

    # Conversion requirements (percentages)
    total_residential_percentage: Optional[str] = None
    per_floor_residential_percentage: Optional[str] = None
    conversion_urls: List[str] = Field(default_factory=list)

    # Development standards
    proposed_height_ft: Optional[str] = None
    height_spec_urls: List[str] = Field(default_factory=list)
    equivalent_commercial_height_limit_ft: Optional[str] = None
    height_limit_urls: List[str] = Field(default_factory=list)

    proposed_units_per_acre: Optional[str] = None
    highest_residential_density_upa: Optional[str] = None
    density_urls: List[str] = Field(default_factory=list)

    parking_spaces_per_unit: Optional[str] = None
    parking_urls: List[str] = Field(default_factory=list)

    development_standards_reference_urls: List[str] = Field(default_factory=list)

    # Permit timing
    permit_submission_date: Optional[str] = None
    permit_timing_urls: List[str] = Field(default_factory=list)

    # LEED
    leed_rating_system: Optional[str] = None
    leed_target_points: Optional[str] = None
    leed_reference_urls: List[str] = Field(default_factory=list)
    leed_documentation_completeness: Optional[str] = None
    leed_documentation_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project_info() -> str:
    return """
    Extract the key details for the proposed Texas SB 840 office-to-residential conversion project exactly as stated in the answer. For every category that cites sources, extract ALL URLs the answer provides as supporting references.

    Required fields:
    1) Building identification:
       - building_name
       - address
       - municipality (city)
       - county
       - state
       - constructed_year (4-digit year if present)
       - current_primary_use
       - building_reference_urls (list of URLs confirming name, address, construction year, or current use)

    2) Geographic & legal eligibility:
       - municipality_population (numeric or textual as given)
       - municipality_population_urls (list of URLs that report the municipality population)
       - county_population (numeric or textual as given)
       - county_population_urls (list of URLs that report the county population)
       - sb840_applicability_urls (list of URLs citing SB 840 scope/effective date/jurisdiction criteria)

    3) Zoning compatibility:
       - zoning_classification (exact designation as stated)
       - zoning_permitted_uses (list of specific permitted uses if provided in the answer)
       - zoning_urls (list of URLs proving classification and permitted uses)

    4) Exclusion zones:
       - heavy_industrial_distance_ft (as stated, e.g., "1,200 ft" or "over 1000 ft")
       - airport_military_distance_ft (as stated)
       - safety_zone_designation (e.g., "Not in clear zone/APZ" or "Outside APZ")
       - exclusion_urls (list of URLs/maps that substantiate these determinations)

    5) Conversion percentages:
       - total_residential_percentage (e.g., "70%" or "0.70")
       - per_floor_residential_percentage (e.g., "65%" or "0.65")
       - conversion_urls (list of URLs that confirm/convey these percentages or plan diagrams)

    6) Development standards:
       - proposed_height_ft (height in feet if available, else as stated)
       - height_spec_urls (list of URLs that show the actual/proposed height)
       - equivalent_commercial_height_limit_ft (e.g., "60 ft" or "45 ft" if provided)
       - height_limit_urls (list of URLs that show the equivalent commercial height limit or city standard)
       - proposed_units_per_acre
       - highest_residential_density_upa
       - density_urls (list of URLs that show the proposed density and/or highest allowed density)
       - parking_spaces_per_unit (e.g., "0.8", "1.0", "0.75 per DU", etc.)
       - parking_urls (list of URLs that show the proposed parking ratio)
       - development_standards_reference_urls (URLs that describe SB 840 standards for height/density/parking)

    7) Permit timing:
       - permit_submission_date (as stated in the answer)
       - permit_timing_urls (URLs indicating the planned application timing)

    8) LEED:
       - leed_rating_system (e.g., "LEED BD+C: Core and Shell")
       - leed_target_points (numeric or textual points, as stated)
       - leed_reference_urls (URLs that explain LEED Gold thresholds/requirements)
       - leed_documentation_completeness (verbatim/summary statement about prerequisites/documentation status)
       - leed_documentation_urls (URLs that show the LEED strategy or documentation plan)

    Rules:
    - Return null for any missing field, and [] for any missing URL lists.
    - Do not invent or infer values or URLs not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
def _parse_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        # Try direct numeric
        ss = s.strip().lower().replace(",", "").replace("points", "").replace("ft", "").replace("feet", "")
        # Handle ratio like "0.85:1"
        if ":" in ss:
            left = ss.split(":", 1)[0]
            return float(left)
        # Fallback to first float in string
        m = re.search(r"[-+]?\d*\.?\d+", ss)
        if m:
            return float(m.group(0))
        return None
    except Exception:
        return None


def _parse_percentage_to_0_100(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    ss = s.strip().lower().replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", ss)
    if not m:
        return None
    val = float(m.group(0))
    if "%" in ss:
        return val  # e.g., 65%
    # If expressed as 0.x, treat as fraction
    if val <= 1.5:
        return val * 100.0
    return val


def _parse_parking_ratio(s: Optional[str]) -> Optional[float]:
    # Expect "0.8", "0.8:1", "0.8 per unit", "1 space per dwelling unit"
    if not s:
        return None
    ss = s.strip().lower().replace(",", "")
    # ratio form x:1
    if ":" in ss:
        left = ss.split(":", 1)[0]
        try:
            return float(left)
        except Exception:
            pass
    # "1 space per dwelling unit" -> take first number
    m = re.search(r"[-+]?\d*\.?\d+", ss)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _is_texas(state: Optional[str]) -> bool:
    if not state:
        return False
    st = state.strip().lower()
    return st in {"texas", "tx"} or "texas" in st


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_geographic_and_legal_eligibility(evaluator: Evaluator, parent, data: ProjectExtraction):
    node = evaluator.add_parallel(
        id="Geographic_and_Legal_Eligibility",
        desc="Verification that the project location meets Texas SB 840 jurisdictional requirements",
        parent=parent,
        critical=True
    )

    # Municipality population > 150,000
    muni_pop_leaf = evaluator.add_leaf(
        id="Municipality_Population_Requirement",
        desc="The municipality where the building is located has a population greater than 150,000",
        parent=node,
        critical=True
    )
    muni_name = data.municipality or "the municipality"
    muni_claim = f"The population of {muni_name}, Texas is greater than 150,000."
    await evaluator.verify(
        claim=muni_claim,
        node=muni_pop_leaf,
        sources=data.municipality_population_urls,
        additional_instruction="Use the cited population source (e.g., Census, state demographer, or official city page). Approximate or latest estimates are acceptable if clearly above 150,000."
    )

    # County population > 300,000
    county_pop_leaf = evaluator.add_leaf(
        id="County_Population_Requirement",
        desc="The municipality is wholly or partly located in a county with population greater than 300,000",
        parent=node,
        critical=True
    )
    county_name = data.county or "the county"
    county_claim = f"The population of {county_name} County, Texas is greater than 300,000."
    await evaluator.verify(
        claim=county_claim,
        node=county_pop_leaf,
        sources=data.county_population_urls,
        additional_instruction="Verify from a reliable county population source (Census, state demographer, or official county site) that the population exceeds 300,000."
    )

    # Applicability confirmation (logical combination)
    muni_pop_val = _parse_float(data.municipality_population)
    county_pop_val = _parse_float(data.county_population)
    applicability_ok = (
        _is_texas(data.state)
        and muni_pop_val is not None
        and muni_pop_val > 150000
        and county_pop_val is not None
        and county_pop_val > 300000
    )
    evaluator.add_custom_node(
        result=applicability_ok,
        id="SB_840_Applicability_Confirmation",
        desc="The project is confirmed to be within SB 840's scope of application based on location and jurisdiction",
        parent=node,
        critical=True
    )

    # Reference URL for geographic/legal eligibility (statute or summary)
    ref_geo_leaf = evaluator.add_leaf(
        id="Reference_URL_Geographic",
        desc="URL reference supporting the geographic and legal eligibility verification",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Texas Senate Bill 840 applies to municipalities with population greater than 150,000 that are wholly or partly located in a county with population greater than 300,000, effective September 1, 2025.",
        node=ref_geo_leaf,
        sources=data.sb840_applicability_urls,
        additional_instruction="Verify the applicability thresholds and effective date directly on the statute text or authoritative summary."
    )


async def verify_building_identification_and_baseline(evaluator: Evaluator, parent, data: ProjectExtraction):
    node = evaluator.add_sequential(
        id="Building_Identification_and_Baseline_Eligibility",
        desc="Identification of a specific office building and verification it meets baseline conversion eligibility criteria",
        parent=parent,
        critical=True
    )

    # Building Identification (parallel)
    b_id = evaluator.add_parallel(
        id="Building_Identification",
        desc="A specific existing office building in the qualifying Texas municipality is identified",
        parent=node,
        critical=True
    )

    # Building name and location
    b_name_loc = evaluator.add_leaf(
        id="Building_Name_and_Location",
        desc="The building's name and address are provided",
        parent=b_id,
        critical=True
    )
    claim_name_addr = f"The building named '{data.building_name or ''}' is located at '{data.address or ''}' in {data.municipality or ''}, {data.county or ''} County, Texas."
    await evaluator.verify(
        claim=claim_name_addr,
        node=b_name_loc,
        sources=data.building_reference_urls,
        additional_instruction="The page should explicitly show the building name and address (city and state can be inferred if explicitly shown on the page)."
    )

    # Current primary use
    b_use = evaluator.add_leaf(
        id="Current_Primary_Use",
        desc="The building is currently used primarily for office, retail, or warehouse purposes",
        parent=b_id,
        critical=True
    )
    use_str = (data.current_primary_use or "").lower()
    claim_use = f"The current primary use of the building is '{data.current_primary_use or ''}', which is an office/retail/warehouse use."
    await evaluator.verify(
        claim=claim_use,
        node=b_use,
        sources=data.building_reference_urls,
        additional_instruction="Verify the current or most recent primary use; if multiple uses are listed, confirm that office use is primary or predominant."
    )

    # Building age verification (constructed year <= 2021)
    b_age = evaluator.add_leaf(
        id="Building_Age_Verification",
        desc="The building was constructed at least 5 years before the proposed conversion date (before 2021 for 2026 conversion)",
        parent=b_id,
        critical=True
    )
    year_txt = data.constructed_year or ""
    claim_year = f"The building was originally constructed in {year_txt}, which is 2021 or earlier."
    await evaluator.verify(
        claim=claim_year,
        node=b_age,
        sources=data.building_reference_urls,
        additional_instruction="Confirm the original construction year and ensure it is 2021 or earlier."
    )

    # Reference URL building
    b_ref = evaluator.add_leaf(
        id="Reference_URL_Building",
        desc="URL reference supporting the building identification and basic characteristics",
        parent=b_id,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these sources directly confirms the building's name, address, construction year, or primary use.",
        node=b_ref,
        sources=data.building_reference_urls,
        additional_instruction="Look for explicit details on property listings, assessor records, or official building documents."
    )

    # Zoning Compatibility (parallel)
    zone = evaluator.add_parallel(
        id="Zoning_Compatibility",
        desc="The building is located in a zoning classification that allows office, commercial, retail, warehouse, or mixed-use development",
        parent=node,
        critical=True
    )

    # Zoning classification
    zone_class = evaluator.add_leaf(
        id="Zoning_Classification",
        desc="The current zoning classification of the property is identified",
        parent=zone,
        critical=True
    )
    claim_zone = f"The property's current zoning classification is '{data.zoning_classification or ''}'."
    await evaluator.verify(
        claim=claim_zone,
        node=zone_class,
        sources=data.zoning_urls,
        additional_instruction="Confirm the exact zoning designation (e.g., 'CBD', 'MU-3', 'C-1', 'Form-based' etc.) from official zoning map/GIS or code."
    )

    # Permitted uses verification
    zone_use = evaluator.add_leaf(
        id="Permitted_Uses_Verification",
        desc="The zoning classification permits office, commercial, retail, warehouse, or mixed-use as an allowed use",
        parent=zone,
        critical=True
    )
    claim_zone_use = f"Under the '{data.zoning_classification or ''}' zoning, at least one of office, commercial, retail, warehouse, or mixed-use is permitted."
    await evaluator.verify(
        claim=claim_zone_use,
        node=zone_use,
        sources=data.zoning_urls,
        additional_instruction="Cite the section/table where permitted uses are listed; mixed-use or commercial categories that include office/retail are acceptable."
    )

    # Reference URL zoning
    zone_ref = evaluator.add_leaf(
        id="Reference_URL_Zoning",
        desc="URL reference supporting the zoning classification and permitted uses",
        parent=zone,
        critical=True
    )
    await evaluator.verify(
        claim="These zoning references support the zoning classification and its permitted uses for the subject property.",
        node=zone_ref,
        sources=data.zoning_urls,
        additional_instruction="Ensure the references are official (city code, GIS, or published zoning summary)."
    )

    # Exclusion Zones (parallel)
    excl = evaluator.add_parallel(
        id="Exclusion_Zone_Verification",
        desc="The building is NOT located in any of the exclusion zones specified by SB 840",
        parent=node,
        critical=True
    )

    excl_industrial = evaluator.add_leaf(
        id="Heavy_Industrial_Exclusion",
        desc="The building is not within 1,000 feet of an existing heavy industrial use or development site",
        parent=excl,
        critical=True
    )
    await evaluator.verify(
        claim="The building is at least 1,000 feet away from any existing heavy industrial use or development site.",
        node=excl_industrial,
        sources=data.exclusion_urls,
        additional_instruction="Use cited GIS maps, buffers, or official planning resources; if a map shows buffer radii, ensure the site lies outside 1,000 feet from heavy industrial."
    )

    excl_airport = evaluator.add_leaf(
        id="Airport_Military_Exclusion",
        desc="The building is not within 3,000 feet of an airport or military base",
        parent=excl,
        critical=True
    )
    await evaluator.verify(
        claim="The building is at least 3,000 feet from any airport or military base.",
        node=excl_airport,
        sources=data.exclusion_urls,
        additional_instruction="Check official FAA or local airport overlay maps and military installation maps; confirm the site lies outside a 3,000-foot radius."
    )

    excl_safety = evaluator.add_leaf(
        id="Safety_Zone_Exclusion",
        desc="The building is not in an area designated as a clear zone or accident potential zone",
        parent=excl,
        critical=True
    )
    await evaluator.verify(
        claim="The building is not located within a designated clear zone or accident potential zone (APZ).",
        node=excl_safety,
        sources=data.exclusion_urls,
        additional_instruction="Review airport AICUZ or similar military/airport safety zone maps; confirm site is outside any clear zone or APZ."
    )

    excl_ref = evaluator.add_leaf(
        id="Reference_URL_Exclusions",
        desc="URL reference supporting the exclusion zone verification",
        parent=excl,
        critical=True
    )
    await evaluator.verify(
        claim="These exclusion reference URLs substantiate the determinations regarding heavy industrial proximity, airport/military distance, and safety zones.",
        node=excl_ref,
        sources=data.exclusion_urls,
        additional_instruction="There should be credible mapping or official documents supporting each exclusion determination."
    )


async def verify_sb840_conversion_requirements(evaluator: Evaluator, parent, data: ProjectExtraction):
    node = evaluator.add_parallel(
        id="SB_840_Conversion_Requirements_Compliance",
        desc="Verification that the proposed conversion meets all Texas SB 840 regulatory requirements",
        parent=parent,
        critical=True
    )

    # Conversion percentages (parallel)
    conv = evaluator.add_parallel(
        id="Conversion_Percentage_Requirements",
        desc="The conversion plan meets the minimum residential percentage requirements",
        parent=node,
        critical=True
    )

    conv_total = evaluator.add_leaf(
        id="Total_Building_Residential_Percentage",
        desc="At least 65% of the building's total floor area will be converted to residential use",
        parent=conv,
        critical=True
    )
    claim_total_pct = f"At least 65% of the building's total floor area will be converted to residential use (stated: {data.total_residential_percentage or ''})."
    await evaluator.verify(
        claim=claim_total_pct,
        node=conv_total,
        sources=data.conversion_urls,
        additional_instruction="Verify the total building conversion percentage is at least 65% from the plan, pro forma, or design documents."
    )

    conv_floor = evaluator.add_leaf(
        id="Per_Floor_Residential_Percentage",
        desc="At least 65% of each occupiable floor's floor area will be converted to residential use",
        parent=conv,
        critical=True
    )
    claim_per_floor = f"At least 65% of each occupiable floor will be converted to residential use (stated: {data.per_floor_residential_percentage or ''})."
    await evaluator.verify(
        claim=claim_per_floor,
        node=conv_floor,
        sources=data.conversion_urls,
        additional_instruction="Confirm the per-floor conversion percentage is ≥ 65% for all occupiable floors; a design narrative or per-floor schedule should show this."
    )

    conv_ref = evaluator.add_leaf(
        id="Reference_URL_Conversion",
        desc="URL reference supporting the conversion percentage calculations and plans",
        parent=conv,
        critical=True
    )
    await evaluator.verify(
        claim="These sources substantiate the stated residential conversion percentages (total and per-floor).",
        node=conv_ref,
        sources=data.conversion_urls,
        additional_instruction="Look for explicit numeric conversion percentages or diagrams with areas designated as residential."
    )

    # Development standards (parallel)
    dev = evaluator.add_parallel(
        id="Development_Standards_Compliance",
        desc="The proposed development complies with SB 840 requirements for height, density, and parking",
        parent=node,
        critical=True
    )

    # Height compliance (parallel)
    h_node = evaluator.add_parallel(
        id="Height_Standard_Compliance",
        desc="The building height complies with SB 840 standards",
        parent=dev,
        critical=True
    )
    h_spec = evaluator.add_leaf(
        id="Building_Height_Specification",
        desc="The actual or proposed building height is specified",
        parent=h_node,
        critical=True
    )
    claim_h_spec = f"The building's actual or proposed height is specified as {data.proposed_height_ft or ''} feet (or equivalent)."
    await evaluator.verify(
        claim=claim_h_spec,
        node=h_spec,
        sources=(data.height_spec_urls or data.building_reference_urls),
        additional_instruction="Find an explicit numeric building height or story count convertible to feet; if only stories are given, ensure the page clearly states height/limits."
    )

    # Height limit logical compliance: proposed <= max(45, equivalent commercial limit)
    proposed_h = _parse_float(data.proposed_height_ft)
    eq_comm_h = _parse_float(data.equivalent_commercial_height_limit_ft)
    height_limit = max(45.0, eq_comm_h) if eq_comm_h is not None else 45.0
    evaluator.add_custom_node(
        result=(proposed_h is not None and proposed_h <= height_limit),
        id="Height_Limit_Verification",
        desc="The specified height complies with SB 840 standards (not more restrictive than 45 feet or equivalent commercial height)",
        parent=h_node,
        critical=True
    )

    # Density compliance (parallel)
    d_node = evaluator.add_parallel(
        id="Density_Standard_Compliance",
        desc="The proposed residential density complies with SB 840 standards",
        parent=dev,
        critical=True
    )
    d_spec = evaluator.add_leaf(
        id="Proposed_Unit_Density",
        desc="The proposed residential unit density (units per acre) is specified",
        parent=d_node,
        critical=True
    )
    claim_d_spec = f"The proposed residential density is specified as {data.proposed_units_per_acre or ''} units per acre."
    await evaluator.verify(
        claim=claim_d_spec,
        node=d_spec,
        sources=(data.density_urls or data.conversion_urls),
        additional_instruction="Find an explicit numeric density statement (units per acre) in the project plans, reports, or narrative."
    )

    # Density limit logical compliance: proposed <= max(36, municipality highest residential density)
    proposed_upa = _parse_float(data.proposed_units_per_acre)
    highest_res_upa = _parse_float(data.highest_residential_density_upa)
    density_limit = max(36.0, highest_res_upa) if highest_res_upa is not None else 36.0
    evaluator.add_custom_node(
        result=(proposed_upa is not None and proposed_upa <= density_limit),
        id="Density_Limit_Verification",
        desc="The proposed density complies with SB 840 standards (not exceeding the greater of municipality's highest residential density or 36 units per acre)",
        parent=d_node,
        critical=True
    )

    # Parking compliance (parallel)
    p_node = evaluator.add_parallel(
        id="Parking_Standard_Compliance",
        desc="The parking plan provides no more than one parking space per dwelling unit as permitted under SB 840",
        parent=dev,
        critical=True
    )
    p_spec = evaluator.add_leaf(
        id="Parking_Ratio_Specification",
        desc="The parking ratio (spaces per dwelling unit) is specified",
        parent=p_node,
        critical=True
    )
    claim_p_spec = f"The parking plan specifies {data.parking_spaces_per_unit or ''} spaces per dwelling unit (or equivalent)."
    await evaluator.verify(
        claim=claim_p_spec,
        node=p_spec,
        sources=(data.parking_urls or data.conversion_urls),
        additional_instruction="Look for an explicit parking ratio statement tied to dwelling units."
    )

    # Logical parking compliance: ratio <= 1.0
    p_ratio = _parse_parking_ratio(data.parking_spaces_per_unit)
    evaluator.add_custom_node(
        result=(p_ratio is not None and p_ratio <= 1.0),
        id="Parking_Limit_Verification",
        desc="The parking ratio does not exceed 1.0 spaces per dwelling unit",
        parent=p_node,
        critical=True
    )

    # Reference URL for dev standards (normative SB 840 standards)
    dev_ref = evaluator.add_leaf(
        id="Reference_URL_Development_Standards",
        desc="URL reference supporting the development standards compliance verification",
        parent=dev,
        critical=True
    )
    await evaluator.verify(
        claim="Texas SB 840 sets limits that are not more restrictive than: height 45 feet (or equivalent commercial height if greater), density the municipality's highest residential density (or 36 units/acre if greater), and parking no more than 1 space per dwelling unit.",
        node=dev_ref,
        sources=data.development_standards_reference_urls,
        additional_instruction="Confirm these exact SB 840 standard caps from the statute text or an authoritative summary."
    )

    # Permit timing (critical leaf)
    permit_leaf = evaluator.add_leaf(
        id="Permit_Timing_Compliance",
        desc="The building permit for conversion is submitted on or after September 1, 2025 (SB 840 effective date)",
        parent=node,
        critical=True
    )
    claim_permit = "The building permit application for the conversion is planned to be submitted on or after September 1, 2025."
    await evaluator.verify(
        claim=claim_permit,
        node=permit_leaf,
        sources=data.permit_timing_urls,
        additional_instruction="Look for an explicit schedule or commitment indicating application on/after 2025-09-01."
    )


async def verify_leed(evaluator: Evaluator, parent, data: ProjectExtraction):
    # Note: Set this parent node as NON-CRITICAL to allow non-critical children per framework rule.
    leed_node = evaluator.add_sequential(
        id="LEED_Gold_Certification_Achievement",
        desc="Verification that the project achieves or is designed to achieve LEED Gold certification",
        parent=parent,
        critical=False
    )

    # Rating system selection (critical leaf under non-critical parent is allowed)
    rating_leaf = evaluator.add_leaf(
        id="LEED_Rating_System_Selection",
        desc="The appropriate LEED rating system for the conversion project is identified (typically LEED BD+C: Core and Shell or similar)",
        parent=leed_node,
        critical=True
    )
    claim_rating = f"The project uses an appropriate LEED rating system for an office-to-residential conversion (e.g., BD+C: Core and Shell, New Construction), stated as '{data.leed_rating_system or ''}'."
    await evaluator.verify(
        claim=claim_rating,
        node=rating_leaf,
        sources=(data.leed_documentation_urls or data.leed_reference_urls),
        additional_instruction="Verify that the stated LEED rating system is suitable for major renovation/conversion projects to residential/mixed-use."
    )

    # Points achievement (parallel, non-critical to allow a non-critical child)
    points_node = evaluator.add_parallel(
        id="LEED_Gold_Point_Achievement",
        desc="The project achieves or is designed to achieve the point threshold for LEED Gold certification",
        parent=leed_node,
        critical=False
    )

    # Minimum (critical)
    min_pts_leaf = evaluator.add_leaf(
        id="Minimum_Point_Threshold",
        desc="The project achieves at least 60 points on the LEED scorecard",
        parent=points_node,
        critical=True
    )
    claim_min_pts = "The project targets at least 60 LEED points (the lower bound for Gold)."
    await evaluator.verify(
        claim=claim_min_pts,
        node=min_pts_leaf,
        sources=(data.leed_documentation_urls or data.leed_reference_urls),
        additional_instruction="Look for a stated target score, scorecard, or LEED strategy indicating ≥60 points."
    )

    # Maximum (non-critical)
    tgt_points = _parse_float(data.leed_target_points)
    evaluator.add_custom_node(
        result=(tgt_points is not None and tgt_points <= 79.0),
        id="Maximum_Point_Threshold",
        desc="The project achieves no more than 79 points (to qualify as Gold, not Platinum)",
        parent=points_node,
        critical=False
    )

    # Reference for LEED thresholds (critical)
    ref_leed_leaf = evaluator.add_leaf(
        id="Reference_URL_LEED",
        desc="URL reference supporting LEED Gold certification requirements and point thresholds",
        parent=points_node,
        critical=True
    )
    await evaluator.verify(
        claim="LEED Gold certification requires between 60 and 79 points on the applicable LEED rating system.",
        node=ref_leed_leaf,
        sources=data.leed_reference_urls,
        additional_instruction="Confirm the Gold threshold (60–79 points) from USGBC or other authoritative LEED documentation."
    )

    # Documentation completeness (non-critical)
    doc_leaf = evaluator.add_leaf(
        id="LEED_Documentation_Completeness",
        desc="All required LEED prerequisites and documentation are completed or planned",
        parent=leed_node,
        critical=False
    )
    claim_docs = "All required LEED prerequisites and documentation are either completed or have a clear plan to be completed for LEED Gold submission."
    await evaluator.verify(
        claim=claim_docs,
        node=doc_leaf,
        sources=data.leed_documentation_urls,
        additional_instruction="Verify a credible plan or confirmation that prerequisites will be met and documentation will be submitted."
    )


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
    """
    Evaluate an answer for the Texas SB 840 office-to-residential conversion project with LEED Gold target.
    Note: Root is initialized as SEQUENTIAL but non-critical to allow sections with mixed criticality per framework constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extracted: ProjectExtraction = await evaluator.extract(
        prompt=prompt_extract_project_info(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Build verification tree according to rubric
    # Adjusted criticality: Root kept non-critical to allow non-critical leaves under some sections (per framework rule).
    # Each major section retains critical sub-requirements reflecting the rubric's intent.
    await verify_geographic_and_legal_eligibility(evaluator, root, extracted)
    await verify_building_identification_and_baseline(evaluator, root, extracted)
    await verify_sb840_conversion_requirements(evaluator, root, extracted)
    await verify_leed(evaluator, root, extracted)

    return evaluator.get_summary()