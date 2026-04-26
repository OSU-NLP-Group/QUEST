import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_region_colocation"
TASK_DESCRIPTION = (
    "Identify 4 enterprise-grade colocation data center facilities, with one facility located in each of the following 4 U.S. geographic regions: Western, Eastern, Central/Midwest, and Southern. Each facility must meet all of the following technical and operational requirements:\n\n"
    "1. Tier Certification: Must hold Uptime Institute Tier III (concurrent maintainability) or Tier IV certification\n"
    "2. Power Capacity: Must offer wholesale colocation services with a minimum of 300 kW available power capacity\n"
    "3. Rack Power Density: Must support minimum 12 kW per rack average power density\n"
    "4. Redundancy: Must provide N+1 redundancy minimum for both power and cooling systems\n"
    "5. Cooling Standards: Must comply with ASHRAE thermal guidelines (Class A1 with 15-32°C recommended temperature range, or Class A2 with 10-35°C recommended temperature range)\n"
    "6. Physical Security: Must implement comprehensive physical security including biometric access, mantrap entry systems, and 24/7 security personnel with CCTV\n"
    "7. Security Certifications: Must hold valid ISO 27001 certification\n"
    "8. Audit Compliance: Must hold valid SOC 2 Type II certification\n"
    "9. Network Connectivity: Must be carrier-neutral, provide access to ≥3 carriers, and have meet-me room (MMR)\n"
    "10. Service Level Agreement: Must provide minimum 99.99% uptime SLA guarantee\n"
    "11. Fire Suppression: Must utilize clean agent fire suppression (FM-200, Novec 1230, or equivalent)\n"
    "12. Infrastructure: Must have raised floor with minimum load capacity of 150 lbf/ft² (7.2 kPa)\n"
    "13. Space Options: Must offer flexible wholesale colocation space options including cages, private suites, or dedicated data halls\n\n"
    "For each facility, provide the facility name, specific location (city and state), and reference URLs confirming that each requirement is met."
)

# --------------------------------------------------------------------------- #
# Region classification sets                                                  #
# --------------------------------------------------------------------------- #
WEST_STATES = {
    "WA", "OR", "CA", "NV", "AZ", "NM", "UT", "CO", "ID", "MT", "WY", "AK", "HI"
}
EAST_STATES = {
    "ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "DC", "VA"
}
CENTRAL_STATES = {
    "OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "KS", "NE", "SD", "ND"
}
SOUTH_STATES = {
    "NC", "SC", "GA", "FL", "AL", "MS", "LA", "AR", "OK", "TX", "TN", "KY", "WV"
}


def state_to_region(state_abbrev: Optional[str]) -> Optional[str]:
    if not state_abbrev:
        return None
    s = state_abbrev.strip().upper()
    if s in WEST_STATES:
        return "western"
    if s in EAST_STATES:
        return "eastern"
    if s in CENTRAL_STATES:
        return "central"
    if s in SOUTH_STATES:
        return "southern"
    return None


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    region_hint: Optional[str] = None  # if the answer explicitly tags a region (e.g., "Western")

    # URLs for location verification
    location_urls: List[str] = Field(default_factory=list)

    # URLs for each requirement
    tier_urls: List[str] = Field(default_factory=list)
    power_capacity_urls: List[str] = Field(default_factory=list)
    rack_density_urls: List[str] = Field(default_factory=list)
    power_redundancy_urls: List[str] = Field(default_factory=list)
    cooling_redundancy_urls: List[str] = Field(default_factory=list)
    ashrae_urls: List[str] = Field(default_factory=list)

    biometric_urls: List[str] = Field(default_factory=list)
    mantrap_urls: List[str] = Field(default_factory=list)
    security_24x7_urls: List[str] = Field(default_factory=list)

    iso27001_urls: List[str] = Field(default_factory=list)
    soc2_urls: List[str] = Field(default_factory=list)

    multiple_carriers_urls: List[str] = Field(default_factory=list)
    mmr_urls: List[str] = Field(default_factory=list)

    sla_urls: List[str] = Field(default_factory=list)
    fire_suppression_urls: List[str] = Field(default_factory=list)
    raised_floor_urls: List[str] = Field(default_factory=list)
    space_options_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract all colocation data center facilities mentioned in the answer. For each facility, extract:
    1) name: The facility name
    2) city: The city where the facility is located
    3) state: The U.S. state (use two-letter abbreviation if available; otherwise, the full state name)
    4) region_hint: If the answer explicitly labels the facility's region (e.g., "Western", "Eastern", "Central/Midwest", "Southern"), extract that label; otherwise null.
    5) location_urls: URLs that confirm the facility name and location (city and state)

    For each of the following requirements, extract ONLY the URLs in the answer that support the requirement for the specific facility. If no URL is provided in the answer for a requirement, return an empty list for that requirement:
    - tier_urls: URLs that confirm Uptime Institute Tier III or Tier IV certification
    - power_capacity_urls: URLs that confirm minimum 300 kW wholesale available capacity
    - rack_density_urls: URLs that confirm ≥12 kW per rack average power density
    - power_redundancy_urls: URLs that confirm N+1 (or higher) power redundancy
    - cooling_redundancy_urls: URLs that confirm N+1 (or higher) cooling redundancy
    - ashrae_urls: URLs that confirm ASHRAE Class A1 (15–32°C) or Class A2 (10–35°C) compliance
    - biometric_urls: URLs that confirm biometric access control (fingerprint/retinal/facial recognition)
    - mantrap_urls: URLs that confirm mantrap entry systems with interlocking doors
    - security_24x7_urls: URLs that confirm 24/7 on-site security personnel and CCTV surveillance
    - iso27001_urls: URLs that confirm ISO/IEC 27001 certification
    - soc2_urls: URLs that confirm SOC 2 Type II certification
    - multiple_carriers_urls: URLs that confirm ≥3 telecom/ISP carriers and carrier-neutral status
    - mmr_urls: URLs that confirm meet-me room (MMR) for interconnection/cross-connects
    - sla_urls: URLs that confirm minimum 99.99% uptime SLA
    - fire_suppression_urls: URLs that confirm clean agent fire suppression (FM-200, Novec 1230, or equivalent)
    - raised_floor_urls: URLs that confirm raised floor ≥150 lbf/ft² (≈7.2 kPa) load capacity
    - space_options_urls: URLs that confirm flexible wholesale colocation space (cages, suites, dedicated halls)

    IMPORTANT:
    - Extract only URLs explicitly present in the answer content. Do not invent or infer URLs.
    - If any field is missing for a facility, set it to null or empty list as appropriate.
    - Return a JSON object with a 'facilities' array (one object per facility found).
    """


# --------------------------------------------------------------------------- #
# Helper functions to allocate facilities to regions                          #
# --------------------------------------------------------------------------- #
def pick_facility_for_region(
    facilities: List[FacilityItem],
    desired_region: str,
    used_indices: set
) -> Tuple[Optional[FacilityItem], Optional[int]]:
    # 1) Try region_hint first
    for idx, fac in enumerate(facilities):
        if idx in used_indices:
            continue
        if fac.region_hint and fac.region_hint.strip().lower() in {desired_region, desired_region.replace("_", " ")}:
            return fac, idx

    # 2) Try by state mapping
    for idx, fac in enumerate(facilities):
        if idx in used_indices:
            continue
        reg = state_to_region(fac.state)
        if reg == desired_region:
            return fac, idx

    return None, None


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    return state.strip().upper()


# --------------------------------------------------------------------------- #
# Verification helper to build sequential requirement groups                  #
# --------------------------------------------------------------------------- #
async def add_requirement_sequential(
    evaluator: Evaluator,
    parent_node,
    id_prefix: str,
    group_desc: str,
    urls: List[str],
    check_id_suffix: str,
    check_desc: str,
    url_id_suffix: str,
    url_desc: str,
    claim: str,
    add_ins: str
):
    """
    Build a sequential group for one requirement:
      1) First, ensure URLs exist (critical)
      2) Then, verify the claim using those URLs (critical)
    """
    group_node = evaluator.add_sequential(
        id=id_prefix,
        desc=group_desc,
        parent=parent_node,
        critical=True
    )

    url_node = evaluator.add_custom_node(
        result=bool(urls) and len(urls) > 0,
        id=f"{id_prefix}_{url_id_suffix}",
        desc=url_desc,
        parent=group_node,
        critical=True
    )

    check_node = evaluator.add_leaf(
        id=f"{id_prefix}_{check_id_suffix}",
        desc=check_desc,
        parent=group_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=check_node,
        sources=urls,
        additional_instruction=add_ins
    )

    return group_node


# --------------------------------------------------------------------------- #
# Region subtree construction                                                 #
# --------------------------------------------------------------------------- #
async def verify_region_facility(
    evaluator: Evaluator,
    root_node,
    region_key: str,         # "west" | "east" | "central" | "south"
    region_title: str,       # "Western" | "Eastern" | "Central/Midwest" | "Southern"
    fac: Optional[FacilityItem]
):
    """
    Build the full verification subtree for one region's facility.
    The region node is critical: failing any requirement fails the region node.
    """
    region_node = evaluator.add_parallel(
        id=f"{region_key.capitalize()}_Region_Facility",
        desc=f"A colocation data center facility located in the {region_title} U.S. region that meets all specified requirements",
        parent=root_node,
        critical=True
    )

    # ---------------- Basic identification & location ------------------- #
    basic_node = evaluator.add_sequential(
        id=f"{region_key.capitalize()}_Facility_Basic",
        desc="Facility identification and location verification",
        parent=region_node,
        critical=True
    )

    name_exists = bool(fac and fac.name and fac.name.strip())
    location_exists = bool(fac and fac.city and fac.state and fac.city.strip() and fac.state.strip())

    evaluator.add_custom_node(
        result=name_exists,
        id=f"{region_key.capitalize()}_Facility_Info_Provided",
        desc="Facility name is provided",
        parent=basic_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=location_exists,
        id=f"{region_key.capitalize()}_Facility_Location_Provided",
        desc="Facility location (city and state) is provided",
        parent=basic_node,
        critical=True
    )

    loc_urls = fac.location_urls if fac else []
    evaluator.add_custom_node(
        result=bool(loc_urls),
        id=f"{region_key.capitalize()}_Facility_Location_URL",
        desc="Provide URL reference confirming the facility's city and state",
        parent=basic_node,
        critical=True
    )

    loc_claim = ""
    if fac and fac.name and fac.city and fac.state:
        loc_claim = f"The facility '{fac.name}' is located in {fac.city}, {fac.state}."
    else:
        loc_claim = "The facility is located at the stated city and state."
    loc_check_node = evaluator.add_leaf(
        id=f"{region_key.capitalize()}_Facility_Location_Check",
        desc="Verify the facility location (city, state) is correct",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_check_node,
        sources=loc_urls,
        additional_instruction="Confirm that the facility page or related official page explicitly states the city and state."
    )

    # Region mapping (custom rule based on state sets)
    state = normalize_state(fac.state if fac else None)
    mapped_region = state_to_region(state) if state else None
    evaluator.add_custom_node(
        result=(mapped_region == region_key),
        id=f"{region_key.capitalize()}_Region_Mapping",
        desc=f"Verify the state belongs to the {region_title} U.S. region",
        parent=basic_node,
        critical=True
    )

    # Helper facility label for claims
    facility_label = fac.name if (fac and fac.name) else "the facility"

    # ---------------- Tier Classification ------------------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_Tier_Classification",
        group_desc="Facility must hold Uptime Institute Tier III or Tier IV certification",
        urls=fac.tier_urls if fac else [],
        check_id_suffix="Level_Check",
        check_desc="Verify the facility is certified as Tier III (concurrent maintainability) or Tier IV",
        url_id_suffix="Tier_URL",
        url_desc="Provide URL reference confirming the Tier certification",
        claim=f"{facility_label} holds Uptime Institute Tier III or Tier IV certification.",
        add_ins="Look for explicit 'Uptime Institute Tier III' or 'Tier IV' certification on the provided URL. Equivalent phrasing like 'Tier 3/4' is acceptable."
    )

    # ---------------- Power Capacity ≥300 kW ---------------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_Power_Capacity",
        group_desc="Facility must offer wholesale colocation with minimum 300 kW power capacity available",
        urls=fac.power_capacity_urls if fac else [],
        check_id_suffix="Power_Minimum_Check",
        check_desc="Verify available power capacity meets or exceeds 300 kW threshold",
        url_id_suffix="Power_URL",
        url_desc="Provide URL reference confirming power capacity specifications",
        claim=f"{facility_label} offers wholesale colocation with at least 300 kW available power capacity.",
        add_ins="Accept numbers shown in kW or MW (e.g., 0.3 MW = 300 kW). The claim is satisfied if the documented capacity is ≥ 300 kW."
    )

    # ---------------- Rack Density ≥12 kW per rack ---------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_Rack_Density",
        group_desc="Facility must support minimum 12 kW per rack average power density",
        urls=fac.rack_density_urls if fac else [],
        check_id_suffix="Rack_Density_Check",
        check_desc="Verify rack power density meets or exceeds 12 kW per rack",
        url_id_suffix="Rack_Density_URL",
        url_desc="Provide URL reference confirming rack power density specifications",
        claim=f"{facility_label} supports a minimum average power density of 12 kW per rack.",
        add_ins="Look for statements such as '12 kW/rack', '≥12 kW per rack', or higher values (e.g., 15, 20 kW/rack)."
    )

    # ---------------- Redundancy (Power, Cooling) ----------------------- #
    red_node = evaluator.add_parallel(
        id=f"{region_key.capitalize()}_Redundancy",
        desc="Facility must provide N+1 redundancy minimum for power and cooling systems",
        parent=region_node,
        critical=True
    )

    # Power redundancy
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=red_node,
        id_prefix=f"{region_key.capitalize()}_Power_Redundancy",
        group_desc="Power systems must have N+1 redundancy configuration",
        urls=fac.power_redundancy_urls if fac else [],
        check_id_suffix="Power_Redundancy_Check",
        check_desc="Verify N+1 or higher redundancy for power infrastructure",
        url_id_suffix="Power_Redundancy_URL",
        url_desc="Provide URL reference confirming power redundancy configuration",
        claim=f"{facility_label} uses N+1 (or higher, e.g., 2N) redundancy for power infrastructure.",
        add_ins="Evidence could mention 'N+1', '2N', 'redundant UPS/generators', etc. Any ≥N+1 level satisfies the requirement."
    )

    # Cooling redundancy
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=red_node,
        id_prefix=f"{region_key.capitalize()}_Cooling_Redundancy",
        group_desc="Cooling systems must have N+1 redundancy configuration",
        urls=fac.cooling_redundancy_urls if fac else [],
        check_id_suffix="Cooling_Redundancy_Check",
        check_desc="Verify N+1 or higher redundancy for cooling infrastructure",
        url_id_suffix="Cooling_Redundancy_URL",
        url_desc="Provide URL reference confirming cooling redundancy configuration",
        claim=f"{facility_label} uses N+1 (or higher) redundancy for cooling infrastructure.",
        add_ins="Evidence may mention 'N+1 CRAC/CRAH units', 'redundant chillers', etc."
    )

    # ---------------- ASHRAE thermal guidelines ------------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_ASHRAE_Compliance",
        group_desc="Facility must comply with ASHRAE thermal guidelines (Class A1 or A2)",
        urls=fac.ashrae_urls if fac else [],
        check_id_suffix="ASHRAE_Check",
        check_desc="Verify compliance with ASHRAE Class A1 (15–32°C) or A2 (10–35°C) temperature ranges",
        url_id_suffix="ASHRAE_URL",
        url_desc="Provide URL reference confirming ASHRAE compliance",
        claim=f"{facility_label} complies with ASHRAE Class A1 (recommended 15–32°C) or Class A2 (recommended 10–35°C) thermal guidelines.",
        add_ins="The page should mention ASHRAE classes (A1/A2) or recommended ranges consistent with those classes."
    )

    # ---------------- Physical Security (Biometric, Mantrap, 24/7) ------ #
    phys_node = evaluator.add_parallel(
        id=f"{region_key.capitalize()}_Physical_Security",
        desc="Facility must implement comprehensive physical security measures",
        parent=region_node,
        critical=True
    )

    # Biometric access
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=phys_node,
        id_prefix=f"{region_key.capitalize()}_Biometric_Access",
        group_desc="Facility must have biometric access control systems",
        urls=fac.biometric_urls if fac else [],
        check_id_suffix="Biometric_Check",
        check_desc="Verify presence of biometric authentication (fingerprint, retinal, facial recognition, etc.)",
        url_id_suffix="Biometric_URL",
        url_desc="Provide URL reference confirming biometric access control",
        claim=f"{facility_label} implements biometric access control (e.g., fingerprint, retinal, facial recognition or similar).",
        add_ins="Look for explicit mention of biometric systems for access control."
    )

    # Mantrap entry system
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=phys_node,
        id_prefix=f"{region_key.capitalize()}_Mantrap_System",
        group_desc="Facility must have mantrap entry systems with interlocking doors",
        urls=fac.mantrap_urls if fac else [],
        check_id_suffix="Mantrap_Check",
        check_desc="Verify presence of mantrap/security vestibule with two interlocking doors",
        url_id_suffix="Mantrap_URL",
        url_desc="Provide URL reference confirming mantrap entry system",
        claim=f"{facility_label} uses a mantrap entry system with interlocking doors.",
        add_ins="The page should mention 'mantrap', 'two interlocking doors', or equivalent security vestibule."
    )

    # 24/7 security personnel & CCTV
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=phys_node,
        id_prefix=f"{region_key.capitalize()}_24x7_Security",
        group_desc="Facility must maintain 24/7 on-site security personnel and CCTV surveillance",
        urls=fac.security_24x7_urls if fac else [],
        check_id_suffix="24x7_Check",
        check_desc="Verify 24×7×365 on-site security staff and continuous video surveillance",
        url_id_suffix="24x7_URL",
        url_desc="Provide URL reference confirming 24/7 security operations",
        claim=f"{facility_label} maintains 24/7 on-site security personnel and continuous CCTV surveillance.",
        add_ins="Accept phrasing like '24x7', '24/7/365', 'on-site security around the clock', and 'CCTV'/'video surveillance'."
    )

    # ---------------- ISO 27001 certification --------------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_ISO27001_Certification",
        group_desc="Facility must hold valid ISO 27001 certification for information security management",
        urls=fac.iso27001_urls if fac else [],
        check_id_suffix="ISO27001_Check",
        check_desc="Verify current ISO/IEC 27001 certification status",
        url_id_suffix="ISO27001_URL",
        url_desc="Provide URL reference confirming ISO 27001 certification",
        claim=f"{facility_label} holds valid ISO/IEC 27001 certification for its information security management system.",
        add_ins="Evidence may be a certificate listing, compliance page, or accreditation. Verify that it is ISO/IEC 27001."
    )

    # ---------------- SOC 2 Type II certification ----------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_SOC2_Certification",
        group_desc="Facility must hold valid SOC 2 Type II certification",
        urls=fac.soc2_urls if fac else [],
        check_id_suffix="SOC2_Check",
        check_desc="Verify current SOC 2 Type II audit report and certification",
        url_id_suffix="SOC2_URL",
        url_desc="Provide URL reference confirming SOC 2 Type II certification",
        claim=f"{facility_label} holds valid SOC 2 Type II certification.",
        add_ins="Look for explicit 'SOC 2 Type II' wording (type II), not just Type I."
    )

    # ---------------- Carrier-neutral & MMR ----------------------------- #
    carrier_node = evaluator.add_parallel(
        id=f"{region_key.capitalize()}_Carrier_Neutral",
        desc="Facility must be carrier-neutral with meet-me room capabilities",
        parent=region_node,
        critical=True
    )

    # Multiple carriers (≥3) & carrier-neutral
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=carrier_node,
        id_prefix=f"{region_key.capitalize()}_Multiple_Carriers",
        group_desc="Facility must provide access to multiple telecommunications and ISP carriers",
        urls=fac.multiple_carriers_urls if fac else [],
        check_id_suffix="Carriers_Check",
        check_desc="Verify carrier-neutral status with access to 3 or more carriers/ISPs",
        url_id_suffix="Carriers_URL",
        url_desc="Provide URL reference confirming carrier-neutral connectivity options",
        claim=f"{facility_label} is carrier-neutral and provides access to at least 3 telecommunications/ISP carriers.",
        add_ins="Evidence may list carriers present or explicitly state 'carrier-neutral' with multiple carriers (≥3)."
    )

    # Meet-me room (MMR)
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=carrier_node,
        id_prefix=f"{region_key.capitalize()}_MMR_Availability",
        group_desc="Facility must provide meet-me room (MMR) for interconnection",
        urls=fac.mmr_urls if fac else [],
        check_id_suffix="MMR_Check",
        check_desc="Verify presence of meet-me room infrastructure for cross-connects",
        url_id_suffix="MMR_URL",
        url_desc="Provide URL reference confirming meet-me room availability",
        claim=f"{facility_label} provides a meet-me room (MMR) infrastructure for interconnection and cross-connects.",
        add_ins="Look for 'meet-me room', 'MMR', 'carrier interconnection room', or similar language."
    )

    # ---------------- SLA ≥99.99% -------------------------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_SLA_Uptime",
        group_desc="Facility must provide minimum 99.99% uptime SLA guarantee",
        urls=fac.sla_urls if fac else [],
        check_id_suffix="SLA_Check",
        check_desc="Verify uptime SLA is 99.99% or higher (maximum 52.56 minutes downtime per year)",
        url_id_suffix="SLA_URL",
        url_desc="Provide URL reference confirming SLA uptime guarantee",
        claim=f"{facility_label} provides a minimum 99.99% uptime SLA guarantee (≤ 52.56 minutes downtime per year).",
        add_ins="Accept '99.99%' or higher (e.g., 99.999%). If ≥99.99%, the requirement is satisfied."
    )

    # ---------------- Clean agent fire suppression ---------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_Fire_Suppression",
        group_desc="Facility must utilize clean agent fire suppression systems (e.g., FM-200) in IT areas",
        urls=fac.fire_suppression_urls if fac else [],
        check_id_suffix="Fire_System_Check",
        check_desc="Verify deployment of clean agent fire suppression (FM-200, Novec 1230, or equivalent) for IT equipment protection",
        url_id_suffix="Fire_System_URL",
        url_desc="Provide URL reference confirming fire suppression system specifications",
        claim=f"{facility_label} uses clean agent fire suppression systems (FM-200, Novec 1230, or equivalent) in IT equipment areas.",
        add_ins="Look for 'FM-200', 'Novec 1230', 'clean agent', or equivalent systems specifically applied to IT areas."
    )

    # ---------------- Raised floor capacity ----------------------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_Raised_Floor",
        group_desc="Facility must have raised floor with minimum load capacity of 150 lbf/ft²",
        urls=fac.raised_floor_urls if fac else [],
        check_id_suffix="Floor_Capacity_Check",
        check_desc="Verify raised floor load capacity meets or exceeds 150 lbf/ft² (7.2 kPa)",
        url_id_suffix="Floor_Capacity_URL",
        url_desc="Provide URL reference confirming raised floor specifications",
        claim=f"{facility_label} has raised floor with a minimum load capacity of 150 lbf/ft² (≈7.2 kPa).",
        add_ins="Accept phrasing showing ≥150 lbf/ft² or equivalent in kPa (around 7.2 kPa)."
    )

    # ---------------- Space options (cages, suites, halls) -------------- #
    await add_requirement_sequential(
        evaluator=evaluator,
        parent_node=region_node,
        id_prefix=f"{region_key.capitalize()}_Colocation_Options",
        group_desc="Facility must offer flexible colocation space options including cages and private suites",
        urls=fac.space_options_urls if fac else [],
        check_id_suffix="Space_Options_Check",
        check_desc="Verify availability of wholesale colocation spaces (cages, suites, or dedicated halls)",
        url_id_suffix="Space_Options_URL",
        url_desc="Provide URL reference confirming colocation space options",
        claim=f"{facility_label} offers flexible wholesale colocation space options including cages, private suites, or dedicated data halls.",
        add_ins="Evidence may list 'cages', 'private suites', 'data halls', 'wholesale spaces'."
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
) -> Dict:
    """
    Evaluate an answer for the multi-region colocation deployment task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Regions evaluated independently
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

    # Root node must be critical to enforce "all 4 regions must be satisfied".
    # Note: The framework enforces that critical parents can only have critical children.
    # We'll convert region nodes to critical children to satisfy this.
    root.critical = True

    # 1) Extract facilities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    facilities = extracted.facilities or []

    # 2) Allocate up to one facility to each required region (unique allocation)
    used_indices: set = set()

    allocations: Dict[str, Optional[FacilityItem]] = {}
    alloc_indices: Dict[str, Optional[int]] = {}

    for key in ["western", "eastern", "central", "southern"]:
        fac, idx = pick_facility_for_region(facilities, key, used_indices)
        allocations[key] = fac
        alloc_indices[key] = idx
        if idx is not None:
            used_indices.add(idx)

    # 3) Build verification trees for each region (children of root are critical)
    region_meta = [
        ("western", "Western"),
        ("eastern", "Eastern"),
        ("central", "Central/Midwest"),
        ("southern", "Southern"),
    ]

    for reg_key, reg_title in region_meta:
        # Create a critical child region container to comply with root critical constraint
        # We'll attach the whole region subtree under this container
        region_container = evaluator.add_parallel(
            id=f"{reg_key.capitalize()}_Region_Container",
            desc=f"{reg_title} region facility verification",
            parent=root,
            critical=True
        )

        fac = allocations.get(reg_key)
        # If no facility found for this region, create an empty placeholder to trigger failures in required checks
        if not fac:
            fac = FacilityItem()

        await verify_region_facility(
            evaluator=evaluator,
            root_node=region_container,
            region_key=reg_key,
            region_title=reg_title,
            fac=fac
        )

    # 4) Return standardized summary
    return evaluator.get_summary()