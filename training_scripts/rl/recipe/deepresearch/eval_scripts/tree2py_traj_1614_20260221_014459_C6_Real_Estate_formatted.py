import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "austin_sb840_buildings"
TASK_DESCRIPTION = """
A real estate development firm is planning to convert vacant commercial office buildings in downtown Austin, Texas into multifamily residential housing under the provisions of Texas Senate Bill 840 (SB 840), which became effective on September 1, 2025. The firm needs to identify four commercial office buildings that meet the following criteria:

1. Each building must be located in downtown Austin, Texas
2. Each building must currently be used as a commercial office building
3. Each building must have a vacancy rate of at least 50%
4. Each building must be eligible under SB 840, meaning:
   - The building must be in a zoning district that allows commercial, office, retail, warehouse, or mixed-use (not heavy industrial zoning such as IP, LI, MI, or PUD/PDA/R&D that permits heavy industrial uses)
   - The building must not be within 1,000 feet of an existing heavy industrial use
   - The building must not be within 3,000 feet of an airport or military base
   - The building must not be in a clear zone or accident protection zone

For each of the four buildings, provide:
- The specific street address
- Documentation of the current vacancy rate
- Verification of SB 840 eligibility (zoning information and confirmation that the building does not fall under any exclusion criteria)
- The maximum residential density allowed under Austin's SB 840 implementation
- The maximum building height allowed under Austin's SB 840 implementation for that specific property

Identify these four buildings with all required information and supporting URL references.
"""

# City of Austin SB 840 core implementation references (used in additional instructions)
SB840_MAX_DENSITY_UNITS_PER_ACRE = 54
SB840_HEIGHT_RULE_SUMMARY = "Under SB 840, the maximum residential building height at an eligible site is the greater of (a) the height permitted for commercial uses on the site, or (b) 45 feet."

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BuildingBasicInfo(BaseModel):
    building_name: Optional[str] = None
    address: Optional[str] = None
    current_use: Optional[str] = None
    vacancy_rate: Optional[str] = None
    basic_info_sources: List[str] = Field(default_factory=list)


class SB840Eligibility(BaseModel):
    zoning_district: Optional[str] = None
    eligibility_sources: List[str] = Field(default_factory=list)
    industrial_proximity_notes: Optional[str] = None
    airport_military_proximity_notes: Optional[str] = None


class DevelopmentStandards(BaseModel):
    max_density_allowed: Optional[str] = None
    proposed_density: Optional[str] = None
    max_height_allowed: Optional[str] = None
    development_sources: List[str] = Field(default_factory=list)


class BuildingExtractionItem(BaseModel):
    basic: Optional[BuildingBasicInfo] = None
    sb840: Optional[SB840Eligibility] = None
    development: Optional[DevelopmentStandards] = None


class BuildingsExtraction(BaseModel):
    buildings: List[BuildingExtractionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_buildings() -> str:
    return """
    Extract up to four commercial office buildings in downtown Austin, Texas as presented in the answer. For each building, return an object with three sections: 'basic', 'sb840', and 'development'.

    For each building:
    [basic]
    - building_name: Name of the building/property if provided; otherwise null
    - address: Specific street address in downtown Austin, Texas
    - current_use: How the building is currently used (e.g., "commercial office", "Class A office", "office tower", etc.)
    - vacancy_rate: The documented vacancy rate text as written (e.g., "52%", "about half empty", "50% vacant as of Q3 2025")
    - basic_info_sources: List of URLs provided in the answer that support the address/current use/vacancy rate

    [sb840]
    - zoning_district: The zoning district name/code shown in the answer or its sources (e.g., "CBD", "CS", "MU", "DMU", etc.)
    - eligibility_sources: List of URLs supporting zoning and/or eligibility checks (e.g., city zoning map or GIS, planning docs)
    - industrial_proximity_notes: Any text indicating proximity or buffers relative to heavy industrial uses (e.g., "no heavy industrial nearby", "LI zone > 1,000 ft away")
    - airport_military_proximity_notes: Any text indicating proximity or buffers relative to airports/military bases, or "clear/accident protection zones"

    [development]
    - max_density_allowed: The maximum residential density allowed at this property under Austin's SB 840 implementation (text as given; e.g., "54 units/acre")
    - proposed_density: If the answer states a proposed density, extract it (text as given); otherwise null
    - max_height_allowed: The maximum building height allowed at this property under SB 840 (text as given; e.g., "45 ft", "60 ft")
    - development_sources: List of URLs that support density/height/development standards as cited

    Rules:
    - Do not invent information. Extract only what appears in the answer.
    - If a requested field is not mentioned, return null for that field (or [] for a URL list).
    - Extract URLs in full (plain URL or markdown link target).
    - Return an array 'buildings' with at most 4 items, in the same order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_non_empty(*vals: Optional[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _limit_to_4_buildings(extraction: BuildingsExtraction) -> List[BuildingExtractionItem]:
    items = extraction.buildings[:4]
    # Pad if fewer than 4
    while len(items) < 4:
        items.append(BuildingExtractionItem(
            basic=BuildingBasicInfo(),
            sb840=SB840Eligibility(),
            development=DevelopmentStandards()
        ))
    # Ensure sub-objects exist
    for i, it in enumerate(items):
        if it.basic is None:
            it.basic = BuildingBasicInfo()
        if it.sb840 is None:
            it.sb840 = SB840Eligibility()
        if it.development is None:
            it.development = DevelopmentStandards()
    return items


def _downtown_additional_instruction() -> str:
    return (
        "Determine whether the address is in downtown Austin, Texas. "
        "Consider Downtown Austin to include the Central Business District and nearby recognized downtown subdistricts "
        "(e.g., Warehouse District, 2nd Street District, Rainey District, Seaholm area). "
        "Approximate downtown boundaries often cited include Lady Bird Lake to the south, the UT Capitol/MLK area to the north, "
        "IH-35 to the east, and Lamar Blvd to the west. Allow reasonable variations and common naming. "
        "If the source clearly states 'Downtown Austin' or equivalent, that is sufficient."
    )


def _zoning_additional_instruction() -> str:
    return (
        "Verify that the zoning district shown is a commercial/office/retail/warehouse/mixed-use type that qualifies under SB 840, "
        "and is NOT a heavy industrial zone. Disqualifying heavy industrial categories include IP, LI, MI, and PUD/PDA/R&D "
        "if they permit heavy industrial uses. Mixed-use and downtown/commercial districts (e.g., DMU, CBD, CS, MU) typically qualify."
    )


def _industrial_buffer_instruction() -> str:
    return (
        "Verify the property is NOT within 1,000 feet of any existing heavy industrial use. "
        "Accept supporting evidence from official maps, planning/GIS databases, or authoritative documents provided in the sources."
    )


def _airport_military_instruction() -> str:
    return (
        "Verify the property is NOT within 3,000 feet of an airport or military base and is NOT located in a clear zone or accident protection zone. "
        "Relevant Austin contexts include Austin-Bergstrom International Airport (ABIA) and Camp Mabry."
    )


def _density_additional_instruction() -> str:
    return (
        f"Under Austin's SB 840 implementation, the maximum residential density is {SB840_MAX_DENSITY_UNITS_PER_ACRE} units per acre. "
        "Verify that the stated or proposed density for this property does not exceed that maximum (interpret textual formats such as 'units/acre', "
        "or equivalent). If the source explicitly confirms the maximum as 54 u/ac, that is sufficient."
    )


def _height_additional_instruction() -> str:
    return (
        f"Verify the stated maximum building height complies with SB 840's rule: {SB840_HEIGHT_RULE_SUMMARY} "
        "Use the provided sources to confirm the property's allowed height and whether it satisfies the SB 840 standard."
    )


# --------------------------------------------------------------------------- #
# Verification logic per building                                             #
# --------------------------------------------------------------------------- #
async def verify_building(
    evaluator: Evaluator,
    parent_node,
    b: BuildingExtractionItem,
    idx: int,
) -> None:
    """
    Build the verification sub-tree for one building and execute all checks.
    """

    # Create the per-building sequential node
    building_node = evaluator.add_sequential(
        id=f"Building_{idx+1}",
        desc=[
            "First eligible building with complete information and compliance verification",
            "Second eligible building with complete information and compliance verification",
            "Third eligible building with complete information and compliance verification",
            "Fourth eligible building with complete information and compliance verification",
        ][idx],
        parent=parent_node,
        critical=False  # allow partial credit across buildings
    )

    # -------------------- Basic Information (critical, parallel) --------------------
    basic_node = evaluator.add_parallel(
        id=f"Building_{idx+1}_Basic_Information",
        desc=f"Basic identifying information and current status of the building #{idx+1}",
        parent=building_node,
        critical=True
    )

    # Basic info reference existence (critical custom)
    basic_ref_node = evaluator.add_custom_node(
        result=bool(b.basic.basic_info_sources),
        id=f"Building_{idx+1}_Basic_Info_Reference",
        desc="URL reference provided to verify basic building information",
        parent=basic_node,
        critical=True
    )

    # Address leaf
    addr_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Address",
        desc="Specific street address of the building in downtown Austin is provided",
        parent=basic_node,
        critical=True
    )
    addr_claim = (
        f"The property's street address is '{_first_non_empty(b.basic.address)}' and it is located in downtown Austin, Texas."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=b.basic.basic_info_sources,
        extra_prerequisites=[basic_ref_node],
        additional_instruction=_downtown_additional_instruction()
    )

    # Current use leaf
    use_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Current_Use",
        desc="Building is verified to be a commercial office building",
        parent=basic_node,
        critical=True
    )
    use_claim = (
        f"The building at '{_first_non_empty(b.basic.address)}' is currently used as a commercial office building."
    )
    await evaluator.verify(
        claim=use_claim,
        node=use_leaf,
        sources=b.basic.basic_info_sources,
        extra_prerequisites=[basic_ref_node],
        additional_instruction="Confirm that the property type is office (e.g., 'Class A office', 'office tower', 'commercial office')."
    )

    # Vacancy rate leaf
    vac_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Vacancy_Rate",
        desc="Building has a documented vacancy rate of at least 50%",
        parent=basic_node,
        critical=True
    )
    vac_str = _first_non_empty(b.basic.vacancy_rate)
    vac_claim = (
        f"The building has a documented vacancy rate of at least 50%. The source reports vacancy as: '{vac_str}'."
    )
    await evaluator.verify(
        claim=vac_claim,
        node=vac_leaf,
        sources=b.basic.basic_info_sources,
        extra_prerequisites=[basic_ref_node],
        additional_instruction="Interpret phrases like 'half empty' or numeric percentages (>= 50%). If the source explicitly shows 50% or more, consider the claim supported."
    )

    # -------------------- SB 840 Eligibility (critical, parallel) -------------------
    elig_node = evaluator.add_parallel(
        id=f"Building_{idx+1}_SB840_Eligibility",
        desc="Verification that the building meets all SB 840 eligibility requirements",
        parent=building_node,
        critical=True
    )

    # Eligibility reference existence (critical custom)
    elig_ref_node = evaluator.add_custom_node(
        result=bool(b.sb840.eligibility_sources),
        id=f"Building_{idx+1}_Eligibility_Reference",
        desc="URL reference provided to verify zoning and eligibility status",
        parent=elig_node,
        critical=True
    )

    # Eligible zoning leaf
    zoning_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Eligible_Zoning",
        desc="Building is in a commercial, office, retail, warehouse, or mixed-use zoning district that qualifies under SB 840",
        parent=elig_node,
        critical=True
    )
    zoning_claim = (
        f"The property's zoning district is '{_first_non_empty(b.sb840.zoning_district)}', which qualifies under SB 840 "
        "as a commercial/office/retail/warehouse/mixed-use type (not heavy industrial)."
    )
    await evaluator.verify(
        claim=zoning_claim,
        node=zoning_leaf,
        sources=b.sb840.eligibility_sources,
        extra_prerequisites=[elig_ref_node],
        additional_instruction=_zoning_additional_instruction()
    )

    # Exclusion zones (critical, parallel)
    exclude_node = evaluator.add_parallel(
        id=f"Building_{idx+1}_Exclusion_Zones",
        desc="Building is not subject to any SB 840 exclusion criteria",
        parent=elig_node,
        critical=True
    )

    # Not heavy industrial zoning leaf
    not_industrial_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Not_Heavy_Industrial",
        desc="Building is not zoned for heavy industrial uses (IP, LI, MI, or PUD/PDA/R&D permitting heavy industrial)",
        parent=exclude_node,
        critical=True
    )
    not_industrial_claim = (
        f"The zoning district '{_first_non_empty(b.sb840.zoning_district)}' is not a heavy industrial category (IP, LI, MI) "
        "and not a PUD/PDA/R&D that permits heavy industrial uses."
    )
    await evaluator.verify(
        claim=not_industrial_claim,
        node=not_industrial_leaf,
        sources=b.sb840.eligibility_sources,
        extra_prerequisites=[elig_ref_node],
        additional_instruction="Confirm that the zone is not IP/LI/MI and not a PUD/PDA/R&D allowing heavy industrial uses."
    )

    # Not within 1000 ft of heavy industrial uses leaf
    not_near_industrial_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Not_Near_Heavy_Industrial",
        desc="Building is not within 1,000 feet of an existing heavy industrial use",
        parent=exclude_node,
        critical=True
    )
    not_near_industrial_claim = (
        f"The property at '{_first_non_empty(b.basic.address)}' is not within 1,000 feet of any existing heavy industrial use."
    )
    await evaluator.verify(
        claim=not_near_industrial_claim,
        node=not_near_industrial_leaf,
        sources=b.sb840.eligibility_sources,
        extra_prerequisites=[elig_ref_node],
        additional_instruction=_industrial_buffer_instruction()
    )

    # Not near airport/military and not in clear/accident zones leaf
    not_near_airport_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Not_Near_Airport",
        desc="Building is not within 3,000 feet of an airport or military base, and not in a clear zone or accident protection zone",
        parent=exclude_node,
        critical=True
    )
    not_near_airport_claim = (
        f"The property at '{_first_non_empty(b.basic.address)}' is not within 3,000 feet of an airport or military base and "
        "is not in a clear zone or accident protection zone."
    )
    await evaluator.verify(
        claim=not_near_airport_claim,
        node=not_near_airport_leaf,
        sources=b.sb840.eligibility_sources,
        extra_prerequisites=[elig_ref_node],
        additional_instruction=_airport_military_instruction()
    )

    # -------------------- Development Standards (critical, parallel) --------------
    dev_node = evaluator.add_parallel(
        id=f"Building_{idx+1}_Development_Standards",
        desc="Proposed development parameters comply with Austin's SB 840 implementation standards",
        parent=building_node,
        critical=True
    )

    # Development reference existence (critical custom)
    dev_ref_node = evaluator.add_custom_node(
        result=bool(b.development.development_sources),
        id=f"Building_{idx+1}_Development_Reference",
        desc="URL reference provided to verify development standards or building specifications",
        parent=dev_node,
        critical=True
    )

    # Density compliance leaf
    density_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Density_Compliance",
        desc=f"Proposed density is stated and does not exceed Austin's maximum of {SB840_MAX_DENSITY_UNITS_PER_ACRE} units per acre under SB 840",
        parent=dev_node,
        critical=True
    )
    density_value = _first_non_empty(b.development.proposed_density, b.development.max_density_allowed)
    density_claim = (
        f"The stated/proposed residential density for this property is '{density_value}', and it does not exceed "
        f"{SB840_MAX_DENSITY_UNITS_PER_ACRE} units per acre under Austin's SB 840 implementation."
    )
    await evaluator.verify(
        claim=density_claim,
        node=density_leaf,
        sources=b.development.development_sources,
        extra_prerequisites=[dev_ref_node],
        additional_instruction=_density_additional_instruction()
    )

    # Height compliance leaf
    height_leaf = evaluator.add_leaf(
        id=f"Building_{idx+1}_Height_Compliance",
        desc="Proposed height is stated and complies with the greater of (a) height permitted for commercial uses on the site, or (b) 45 feet",
        parent=dev_node,
        critical=True
    )
    height_value = _first_non_empty(b.development.max_height_allowed)
    height_claim = (
        f"The maximum allowed residential building height for this property under SB 840 is '{height_value}', and it complies with "
        "the rule that the allowed height is the greater of 45 feet or the height permitted for commercial uses on the site."
    )
    await evaluator.verify(
        claim=height_claim,
        node=height_leaf,
        sources=b.development.development_sources,
        extra_prerequisites=[dev_ref_node],
        additional_instruction=_height_additional_instruction()
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
    Evaluate an answer for the Austin SB 840 eligible buildings task and return a structured result dictionary.
    """

    # Initialize evaluator with root parallel aggregation
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

    # Extract building information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="buildings_extraction"
    )

    # Normalize to exactly four buildings
    buildings = _limit_to_4_buildings(extracted)

    # Add a high-level parent node for the four buildings (parallel)
    four_node = evaluator.add_parallel(
        id="Four_Eligible_Buildings_Identified",
        desc="Identify four commercial office buildings in downtown Austin that meet all eligibility and development criteria for conversion to multifamily residential under SB 840",
        parent=root,
        critical=False
    )

    # Add ground truth info/context for SB 840 rules to the summary (informational only)
    evaluator.add_ground_truth({
        "sb840_max_density_units_per_acre": SB840_MAX_DENSITY_UNITS_PER_ACRE,
        "sb840_height_rule": SB840_HEIGHT_RULE_SUMMARY,
        "requirements": [
            "Downtown Austin location",
            "Current commercial office use",
            "Vacancy rate >= 50%",
            "Eligible zoning (commercial/office/retail/warehouse/mixed-use; NOT heavy industrial)",
            "Not within 1,000 ft of heavy industrial use",
            "Not within 3,000 ft of an airport or military base",
            "Not in a clear zone or accident protection zone",
            "Provide max density and max height under SB 840 for the property"
        ]
    })

    # Build and verify each building subtree
    for i in range(4):
        await verify_building(evaluator, four_node, buildings[i], i)

    # Return structured result
    return evaluator.get_summary()