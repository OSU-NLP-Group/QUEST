import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sg_premium_office_selection"
TASK_DESCRIPTION = """A Fortune 500 technology company is seeking to lease premium office space in Singapore for its new Asia-Pacific regional headquarters. The company requires a long-term lease (10+ years) and has specific requirements for the property.

Identify a Singapore office property that meets ALL of the following requirements:

1. Location: Must be located in Singapore's Central Business District (CBD) or Marina Bay area
2. Ownership: Must be owned by one of these major Singapore REITs: CapitaLand Integrated Commercial Trust (CICT), Mapletree Pan Asia Commercial Trust (MPACT), Keppel REIT, or Suntec REIT
3. Building Size:
   - Total gross floor area of at least 600,000 square feet
   - Average floor plates of at least 20,000 square feet
   - Minimum of 30 stories
4. Building Quality: Must be classified as Grade A or Premium Grade A office space
5. Tenant Profile: Must currently house or be suitable for Fortune 500 companies or major international corporations, particularly in banking, financial services, or technology sectors
6. Lease Terms: Must support long-term corporate leases of 10+ years duration
7. Design Features (preferred but not mandatory): Modern, efficient floor plates with column-free or minimal-column design
8. Operational Performance (preferred but not mandatory): Committed occupancy rate of at least 95%

For your answer, provide:
- The official building name
- Complete address including postal code
- The REIT owner
- Total gross floor area (in square feet)
- Average floor plate size (in square feet)
- Number of stories/floors
- Grade classification
- At least one major Fortune 500 or equivalent corporate tenant
- Current occupancy rate (if available)
- URL references supporting each piece of information
"""

ALLOWED_REITS = [
    "CapitaLand Integrated Commercial Trust",
    "CICT",
    "Mapletree Pan Asia Commercial Trust",
    "MPACT",
    "Keppel REIT",
    "Suntec REIT",
]

CBD_AREAS_HINT = [
    "CBD",
    "Central Business District",
    "Raffles Place",
    "Shenton Way",
    "Tanjong Pagar",
    "Downtown Core",
    "Marina Bay",
    "Marina Bay Financial Centre",
]

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class TenantInfo(BaseModel):
    name: Optional[str] = None
    sector: Optional[str] = None  # e.g., "Banking", "Financial Services", "Technology"
    urls: List[str] = Field(default_factory=list)


class PropertyExtraction(BaseModel):
    # Basic identification & ownership
    building_name: Optional[str] = None
    address: Optional[str] = None
    property_urls: List[str] = Field(default_factory=list)            # General property/brochure URL(s)
    reit_owner: Optional[str] = None
    reit_owner_urls: List[str] = Field(default_factory=list)          # Ownership confirmation from REIT site

    # Location
    location_area_label: Optional[str] = None                         # e.g., "CBD", "Marina Bay", "Raffles Place"
    location_urls: List[str] = Field(default_factory=list)

    # Size & specs
    gross_floor_area_sqft: Optional[str] = None                       # Keep as string; may include commas or units
    gfa_urls: List[str] = Field(default_factory=list)
    avg_floor_plate_sqft: Optional[str] = None
    floor_plate_urls: List[str] = Field(default_factory=list)
    floors: Optional[str] = None                                      # e.g., "45", "45 storeys"
    floors_urls: List[str] = Field(default_factory=list)

    # Grade
    grade: Optional[str] = None                                       # e.g., "Premium Grade A", "Grade A"
    grade_urls: List[str] = Field(default_factory=list)

    # Design features (optional)
    design_features_text: Optional[str] = None                        # e.g., "column-free", "minimal columns", "3.0m ceiling"
    design_urls: List[str] = Field(default_factory=list)

    # Tenants
    tenants: List[TenantInfo] = Field(default_factory=list)

    # Lease term support
    lease_term_urls: List[str] = Field(default_factory=list)

    # Operational performance (optional)
    occupancy_rate_percent: Optional[str] = None                      # e.g., "97%", "96.5%"
    occupancy_urls: List[str] = Field(default_factory=list)
    performance_history_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
Extract structured information for a single Singapore premium office property as presented in the answer. Return EXACTLY AND ONLY what is explicitly stated in the answer. Do not invent.
If multiple candidate properties are mentioned, extract the first one that is presented as the recommended/primary property.

Return a JSON object with these fields:
- building_name: Official building name (string)
- address: Full address including Singapore postal code if available (string)
- property_urls: List of URL(s) that identify the property (e.g., building page, brochure, landlord page)
- reit_owner: Name of the REIT owner (string as written in the answer; allow abbreviations like CICT/MPACT)
- reit_owner_urls: URL(s) from the REIT's official website that confirm ownership (if provided)
- location_area_label: Short label for area such as "CBD", "Marina Bay", "Raffles Place", "Shenton Way", "Downtown Core" (string)
- location_urls: URL(s) that support the location claim
- gross_floor_area_sqft: Total gross floor area as written in the answer; prefer square feet text if available, otherwise any unit (string)
- gfa_urls: URL(s) supporting the GFA figure
- avg_floor_plate_sqft: Average floor plate size as written (string, can be approximate or range)
- floor_plate_urls: URL(s) supporting floor plate
- floors: Number of floors/stories as written (string)
- floors_urls: URL(s) supporting floor count
- grade: Grade classification as written (e.g., "Premium Grade A", "Grade A") (string)
- grade_urls: URL(s) supporting grade
- design_features_text: Any mention of modern/efficient floor plates, column-free/minimal columns, ceiling height, etc. (string)
- design_urls: URL(s) supporting design features
- tenants: List of up to 3 major tenants; for each tenant include:
    - name: Tenant/company name (string)
    - sector: Sector/industry if mentioned (e.g., Banking, Financial Services, Technology) (string or null)
    - urls: URL(s) that name the tenant in this property (list)
- lease_term_urls: URL(s) suggesting property supports long-term corporate leases (ideally 10+ years) or showing long WALE/anchor leases
- occupancy_rate_percent: Committed occupancy rate as written (string, e.g., "97%", "96.5%") if available
- occupancy_urls: URL(s) supporting occupancy figure
- performance_history_urls: URL(s) that reflect stable operational performance over time (optional)

Rules:
- Extract only URLs explicitly present in the answer. If missing, use an empty list.
- Do not normalize numbers; keep them as written (e.g., "1.1 million sq ft", "120,000 sqm", "20,000–25,000 sq ft").
- If an item is missing in the answer, set the string to null and the list fields to [].
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def combine_sources(*args: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in args:
        for u in (lst or []):
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def first_tenant(prop: PropertyExtraction) -> Optional[TenantInfo]:
    for t in prop.tenants:
        if non_empty(t.name):
            return t
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_basic_info(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Property_Basic_Information",
        desc="Provide building identification and REIT ownership",
        parent=parent,
        critical=True,  # Critical group; all children must be critical
    )

    # Existence of at least one property info URL
    prop_url_exists = evaluator.add_custom_node(
        result=len(prop.property_urls) > 0,
        id="Property_Info_URL",
        desc="Provide URL reference confirming property identification",
        parent=node,
        critical=True
    )

    # Building name provided (existence)
    evaluator.add_custom_node(
        result=non_empty(prop.building_name),
        id="Building_Name_Provided",
        desc="Building name is provided in the answer",
        parent=node,
        critical=True
    )

    # Building name verification
    name_leaf = evaluator.add_leaf(
        id="Building_Name",
        desc="Provide the official building name",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property is officially named '{prop.building_name}'.",
        node=name_leaf,
        sources=combine_sources(prop.property_urls, prop.reit_owner_urls),
        additional_instruction="Verify that the cited page(s) clearly identify the property with this official building name. Allow minor punctuation/casing variants."
    )

    # Address provided (existence)
    evaluator.add_custom_node(
        result=non_empty(prop.address),
        id="Building_Address_Provided",
        desc="Building address is provided in the answer",
        parent=node,
        critical=True
    )

    # Address verification
    addr_leaf = evaluator.add_leaf(
        id="Building_Address",
        desc="Provide complete street address including postal code",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The address of '{prop.building_name}' is '{prop.address}'.",
        node=addr_leaf,
        sources=combine_sources(prop.property_urls, prop.reit_owner_urls),
        additional_instruction="Confirm the street address (ideally including postal code) matches the cited page(s). Allow minor formatting differences."
    )

    # REIT ownership URL must exist
    evaluator.add_custom_node(
        result=len(prop.reit_owner_urls) > 0,
        id="REIT_Ownership_URL",
        desc="Provide URL from REIT website confirming ownership",
        parent=node,
        critical=True
    )

    # REIT owner provided (existence)
    evaluator.add_custom_node(
        result=non_empty(prop.reit_owner),
        id="REIT_Owner_Provided",
        desc="REIT owner is provided in the answer",
        parent=node,
        critical=True
    )

    # REIT owner must be in allowed set (simple logical check)
    reit_owner_check = evaluator.add_leaf(
        id="REIT_Owner_Name",
        desc="State the owning REIT (must be CICT, MPACT, Keppel REIT, or Suntec REIT)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The owning REIT '{prop.reit_owner}' is one of: {ALLOWED_REITS}. Treat 'CICT' as CapitaLand Integrated Commercial Trust and 'MPACT' as Mapletree Pan Asia Commercial Trust.",
        node=reit_owner_check,
        additional_instruction="This is a simple logical membership check against the allowed list; ignore URL evidence here."
    )

    # Ownership supported by REIT website
    owner_supported = evaluator.add_leaf(
        id="REIT_Ownership_Supported",
        desc="Ownership is confirmed on REIT website",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{prop.reit_owner}' is the current owner/landlord of '{prop.building_name}'.",
        node=owner_supported,
        sources=prop.reit_owner_urls,
        additional_instruction="Confirm on the official REIT site that this property is owned by the named REIT (e.g., asset list, portfolio page, property microsite)."
    )


async def verify_location(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Location_Requirements",
        desc="Verify property location meets requirements",
        parent=parent,
        critical=True
    )

    # Location verification URL must exist
    evaluator.add_custom_node(
        result=len(prop.location_urls) > 0 or len(prop.property_urls) > 0,
        id="Location_Verification_URL",
        desc="Provide URL confirming prime district location",
        parent=node,
        critical=True
    )

    # Confirm Singapore location
    sg_leaf = evaluator.add_leaf(
        id="Singapore_Location",
        desc="Confirm property is located in Singapore",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property '{prop.building_name}' is located in Singapore.",
        node=sg_leaf,
        sources=combine_sources(prop.location_urls, prop.property_urls, prop.reit_owner_urls),
        additional_instruction="Verify that the page clearly indicates the property is in Singapore."
    )

    # Confirm CBD or Marina Bay
    cbd_leaf = evaluator.add_leaf(
        id="CBD_or_Marina_Bay",
        desc="Confirm property is in Central Business District or Marina Bay area",
        parent=node,
        critical=True
    )
    area_hint = prop.location_area_label or "CBD/Marina Bay"
    await evaluator.verify(
        claim=f"The property is located within Singapore's CBD (Downtown Core) or Marina Bay area. The area label in the answer is '{area_hint}'.",
        node=cbd_leaf,
        sources=combine_sources(prop.location_urls, prop.property_urls, prop.reit_owner_urls),
        additional_instruction="Accept well-known CBD sub-districts as CBD (e.g., Raffles Place, Shenton Way, Tanjong Pagar, Downtown Core). Also accept Marina Bay and Marina Bay Financial Centre as matching 'Marina Bay'."
    )


async def verify_size_requirements(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Building_Size_Requirements",
        desc="Verify all building size specifications are met",
        parent=parent,
        critical=True
    )

    # ---- Gross Floor Area ----
    gfa_node = evaluator.add_parallel(
        id="Gross_Floor_Area",
        desc="Verify total GFA meets minimum requirement",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(prop.gfa_urls) > 0,
        id="GFA_Reference_URL",
        desc="Provide URL reference confirming GFA",
        parent=gfa_node,
        critical=True
    )
    gfa_value_leaf = evaluator.add_leaf(
        id="GFA_Value",
        desc="Provide total gross floor area in square feet",
        parent=gfa_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total gross floor area (GFA) of '{prop.building_name}' is reported as '{prop.gross_floor_area_sqft}'.",
        node=gfa_value_leaf,
        sources=prop.gfa_urls,
        additional_instruction="Confirm the GFA figure on the page. If units are in sqm, it's acceptable as long as it's clearly convertible; do not require exact sqft if sqm is provided."
    )
    gfa_min_leaf = evaluator.add_leaf(
        id="GFA_Minimum_Met",
        desc="Confirm GFA is at least 600,000 square feet",
        parent=gfa_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The reported GFA '{prop.gross_floor_area_sqft}' is at least 600,000 square feet.",
        node=gfa_min_leaf,
        additional_instruction="Interpret values given in million sq ft or sqm. Use 1 sqm ≈ 10.7639 sq ft if needed. Allow reasonable rounding."
    )

    # ---- Average Floor Plate ----
    fp_node = evaluator.add_parallel(
        id="Average_Floor_Plate",
        desc="Verify average floor plate size meets minimum",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(prop.floor_plate_urls) > 0,
        id="Floor_Plate_Reference_URL",
        desc="Provide URL reference confirming floor plate size",
        parent=fp_node,
        critical=True
    )
    fp_value_leaf = evaluator.add_leaf(
        id="Floor_Plate_Value",
        desc="Provide average floor plate size in square feet",
        parent=fp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The average floor plate size of '{prop.building_name}' is reported as '{prop.avg_floor_plate_sqft}'.",
        node=fp_value_leaf,
        sources=prop.floor_plate_urls,
        additional_instruction="Accept ranges (e.g., '20,000–25,000 sq ft') or approximations; confirm the provided value/range is stated."
    )
    fp_min_leaf = evaluator.add_leaf(
        id="Floor_Plate_Minimum_Met",
        desc="Confirm average floor plates are at least 20,000 square feet",
        parent=fp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The reported average floor plate '{prop.avg_floor_plate_sqft}' meets or exceeds 20,000 square feet.",
        node=fp_min_leaf,
        additional_instruction="If a range is provided, treat the lower bound as the average for threshold comparison. Allow reasonable rounding."
    )

    # ---- Building Height / Floors ----
    height_node = evaluator.add_parallel(
        id="Building_Height",
        desc="Verify building height meets minimum stories",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(prop.floors_urls) > 0,
        id="Height_Reference_URL",
        desc="Provide URL reference confirming building height",
        parent=height_node,
        critical=True
    )
    floors_leaf = evaluator.add_leaf(
        id="Floor_Count",
        desc="Provide total number of stories/floors",
        parent=height_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The building '{prop.building_name}' has '{prop.floors}' floors (stories).",
        node=floors_leaf,
        sources=prop.floors_urls,
        additional_instruction="Accept 'storeys' spelling and minor wording variants. If a range is shown (e.g., including podium), focus on the office tower's total floors stated."
    )
    min_stories_leaf = evaluator.add_leaf(
        id="Minimum_Stories_Met",
        desc="Confirm building has at least 30 stories",
        parent=height_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The building has at least 30 stories, based on the reported floors value '{prop.floors}'.",
        node=min_stories_leaf,
        additional_instruction="If 'floors' includes non-office podium levels, still treat a total >=30 as meeting the requirement."
    )


async def verify_grade(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Building_Grade_Requirements",
        desc="Verify building quality and grade classification",
        parent=parent,
        critical=True
    )

    grade_node = evaluator.add_parallel(
        id="Grade_Classification",
        desc="Verify Grade A or Premium Grade A classification",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(prop.grade_urls) > 0,
        id="Grade_Reference_URL",
        desc="Provide URL confirming grade classification",
        parent=grade_node,
        critical=True
    )
    grade_designation_leaf = evaluator.add_leaf(
        id="Grade_Designation",
        desc="Provide the official grade classification",
        parent=grade_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The building '{prop.building_name}' is classified as '{prop.grade}'.",
        node=grade_designation_leaf,
        sources=prop.grade_urls,
        additional_instruction="Confirm the grade designation shown (e.g., Grade A, Premium Grade A, Prime Grade A). Allow minor naming variants like 'Prime Grade A' equivalent to 'Premium Grade A'."
    )
    premium_confirm_leaf = evaluator.add_leaf(
        id="Premium_Grade_Confirmation",
        desc="Confirm classification is Grade A or Premium Grade A",
        parent=grade_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The grade classification '{prop.grade}' indicates the building is Grade A or Premium Grade A.",
        node=premium_confirm_leaf,
        additional_instruction="Treat 'Prime Grade A', 'A+', or 'Premium Grade A' as acceptable equivalents within Grade A/Premium Grade A family."
    )


async def verify_design_features_optional(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Modern_Design_Features",
        desc="Verify modern design features (preferred but not mandatory)",
        parent=parent,
        critical=False
    )

    # Column-free/minimal columns (optional)
    col_free_leaf = evaluator.add_leaf(
        id="Column_Free_Design",
        desc="Confirm floor plates are column-free or have minimal columns",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The property features modern, efficient floor plates described as '{prop.design_features_text}', indicating column-free or minimal-column design.",
        node=col_free_leaf,
        sources=prop.design_urls,
        additional_instruction="Look for phrases like 'column-free', 'large, efficient floor plates', or 'minimal columns'. Minor wording differences are acceptable."
    )

    # Ceiling height (optional)
    ceiling_leaf = evaluator.add_leaf(
        id="Ceiling_Height",
        desc="Verify adequate ceiling height (typically 3 meters or more)",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The property has adequate ceiling height (typically around 3.0m or higher) as indicated by '{prop.design_features_text}'.",
        node=ceiling_leaf,
        sources=prop.design_urls,
        additional_instruction="Accept values around 3.0m or higher; look for references to generous/raised ceiling heights."
    )


async def verify_tenant_profile(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Tenant_Profile_Requirements",
        desc="Verify tenant profile meets requirements",
        parent=parent,
        critical=True
    )

    f500_node = evaluator.add_parallel(
        id="Fortune_500_Tenants",
        desc="Verify presence of Fortune 500 or equivalent major corporate tenants",
        parent=node,
        critical=True
    )

    # Ensure we have at least one URL for tenants (either tenant URLs or property URLs mentioning tenants)
    has_any_tenant_url = any(len(t.urls) > 0 for t in prop.tenants) or len(prop.property_urls) > 0
    evaluator.add_custom_node(
        result=has_any_tenant_url,
        id="Tenant_Info_URL",
        desc="Provide URL confirming tenant information",
        parent=f500_node,
        critical=True
    )

    # Major tenant presence & verification
    tenant = first_tenant(prop)
    tenant_name = tenant.name if tenant and tenant.name else ""
    major_tenant_leaf = evaluator.add_leaf(
        id="Major_Tenant_Names",
        desc="Identify at least one Fortune 500 company or equivalent major international corporation as tenant",
        parent=f500_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{tenant_name}' is a current or notable tenant in '{prop.building_name}'.",
        node=major_tenant_leaf,
        sources=combine_sources(tenant.urls if tenant else [], prop.property_urls, prop.reit_owner_urls),
        additional_instruction="Verify that the cited page(s) explicitly list or reference the tenant within this property. Allow reasonable time variation if clearly a prominent tenant."
    )

    # Optional: sector match (non-critical) – placed at root level to avoid critical-parent constraint
    # We'll add it as a separate non-critical node under root in the main function if sector is provided.


async def verify_lease_terms(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Lease_Term_Requirements",
        desc="Verify long-term lease capability",
        parent=parent,
        critical=True
    )

    ten_year_leaf = evaluator.add_leaf(
        id="Ten_Year_Lease_Support",
        desc="Confirm property can accommodate leases of 10+ years duration",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property '{prop.building_name}' supports long-term corporate leases of 10 years or more.",
        node=ten_year_leaf,
        sources=prop.lease_term_urls,
        additional_instruction="Evidence may include landlord policy statements, leasing brochures, anchor tenant long leases, or statements implying long-term corporate leasing capability (e.g., long WALE, renewal options)."
    )


async def verify_operational_performance_optional(evaluator: Evaluator, parent, prop: PropertyExtraction):
    node = evaluator.add_parallel(
        id="Operational_Performance",
        desc="Verify operational performance metrics (preferred but not mandatory)",
        parent=parent,
        critical=False
    )

    # Occupancy Rate (optional)
    occ_node = evaluator.add_parallel(
        id="Occupancy_Rate",
        desc="Verify high occupancy rate",
        parent=node,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(prop.occupancy_urls) > 0,
        id="Occupancy_Reference_URL",
        desc="Provide URL for occupancy data",
        parent=occ_node,
        critical=False
    )

    occ_value_leaf = evaluator.add_leaf(
        id="Current_Occupancy_Percentage",
        desc="Provide current committed occupancy rate",
        parent=occ_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The committed occupancy rate is reported as '{prop.occupancy_rate_percent}'.",
        node=occ_value_leaf,
        sources=prop.occupancy_urls,
        additional_instruction="Confirm the occupancy figure and allow minor rounding. If reported 'as of' a date, it's acceptable."
    )

    occ_thresh_leaf = evaluator.add_leaf(
        id="Occupancy_Threshold_Met",
        desc="Confirm occupancy rate is at least 95%",
        parent=occ_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The reported occupancy '{prop.occupancy_rate_percent}' is at least 95%.",
        node=occ_thresh_leaf,
        additional_instruction="Treat '>= 95%' as meeting the requirement; minor rounding acceptable."
    )

    # Performance history (optional)
    perf_hist_leaf = evaluator.add_leaf(
        id="Performance_History",
        desc="Verify stable operational track record",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The property '{prop.building_name}' demonstrates a stable operational performance track record (e.g., consistently high occupancy or stable rents) across reports.",
        node=perf_hist_leaf,
        sources=combine_sources(prop.performance_history_urls, prop.occupancy_urls, prop.reit_owner_urls),
        additional_instruction="Look for signals across time (e.g., multiple reporting periods) indicating stability; minor inference allowed if clearly supported."
    )


async def add_optional_sector_match(evaluator: Evaluator, parent, prop: PropertyExtraction):
    # Optional sector match (non-critical) – checks that at least one tenant is in BFSI or Tech
    tenant = first_tenant(prop)
    if not tenant or not non_empty(tenant.sector):
        # Still add a leaf but it will likely fail/skip; it's non-critical
        sector = ""
    else:
        sector = tenant.sector

    sector_leaf = evaluator.add_leaf(
        id="Tenant_Sector_Match",
        desc="Confirm tenants include banking, financial services, or technology sectors",
        parent=parent,
        critical=False
    )
    await evaluator.verify(
        claim=f"The tenant sector '{sector}' belongs to banking, financial services, or technology sectors.",
        node=sector_leaf,
        additional_instruction="Treat synonyms and close variants as matches (e.g., fintech, investment banking, software, cloud)."
    )


async def add_optional_long_term_evidence(evaluator: Evaluator, parent, prop: PropertyExtraction):
    # Optional evidence leaf for long-term leases
    lt_leaf = evaluator.add_leaf(
        id="Long_Term_Lease_Evidence",
        desc="Provide evidence of current or historical long-term leases",
        parent=parent,
        critical=False
    )
    await evaluator.verify(
        claim=f"There is evidence that '{prop.building_name}' has current or historical long-term (10+ years) leases with corporate tenants.",
        node=lt_leaf,
        sources=prop.lease_term_urls,
        additional_instruction="Look for mentions of long WALE, fixed long-term leases, anchor tenant long leases, or announced multi-year lease terms."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Singapore premium office property selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent major requirement groups
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

    # Extract structured property info from the answer
    prop: PropertyExtraction = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=PropertyExtraction,
        extraction_name="property_extraction",
    )

    # Record ground truth constraints for transparency
    evaluator.add_ground_truth({
        "allowed_reits": ALLOWED_REITS,
        "location_required": "Singapore CBD or Marina Bay",
        "min_gfa_sqft": "600,000",
        "min_avg_floor_plate_sqft": "20,000",
        "min_stories": "30",
        "required_grade": "Grade A or Premium Grade A",
        "required_lease_term": "10+ years",
        "preferred_occupancy_threshold": ">= 95%",
    }, gt_type="constraints")

    # Build verification subtrees
    await verify_basic_info(evaluator, root, prop)
    await verify_location(evaluator, root, prop)
    await verify_size_requirements(evaluator, root, prop)
    await verify_grade(evaluator, root, prop)

    # Optional design features (moved to root as non-critical to satisfy critical-parent constraint)
    await verify_design_features_optional(evaluator, root, prop)

    await verify_tenant_profile(evaluator, root, prop)
    await verify_lease_terms(evaluator, root, prop)

    # Optional: sector match and long-term lease evidence (non-critical)
    await add_optional_sector_match(evaluator, root, prop)
    await add_optional_long_term_evidence(evaluator, root, prop)

    # Optional operational performance
    await verify_operational_performance_optional(evaluator, root, prop)

    # Return standard summary
    return evaluator.get_summary()