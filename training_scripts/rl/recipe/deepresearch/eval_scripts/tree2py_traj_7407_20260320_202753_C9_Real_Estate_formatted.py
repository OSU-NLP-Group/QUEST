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
TASK_ID = "tx_office_transit_leed_ada"
TASK_DESCRIPTION = """
Identify a commercial office development project in Dallas, Houston, or Austin, Texas that meets ALL of the following requirements:

1. Location & Transit: The project must be located within 1/2 mile (2,640 feet) of a major public transit rail station (DART light rail in Dallas, Metro rail in Houston, or CapMetro rail in Austin). Specify the nearest station name and line.

2. Building Size: The total building must be at least 100,000 square feet, with at least 50,000 square feet dedicated to office space. Provide the total square footage and office component size.

3. LEED Certification: The project must be registered for or have achieved LEED certification at the Silver level or higher (50+ points). Provide the LEED certification level (Silver, Gold, or Platinum) and include a link to the project's LEED directory entry or official certification documentation.

4. Development Status: The project must be either currently under construction, completed within the last 2 years (since January 2024), or in advanced planning stages with publicly available documentation. Specify the current status and expected or actual completion date.

5. ADA Compliance: The building must include elevators (required for 3+ story buildings) and provide accessible parking at a minimum of 2% of total parking spaces (applicable when 500-1000 total spaces are provided).

6. Documentation: Provide a valid URL reference that verifies the project's existence, basic details, and specifications.

For your answer, provide:
- Project name and full address
- City location
- Nearest transit station (name and rail line) with distance
- Total building square footage and office space size
- Number of stories or height
- LEED certification level and status
- Current development status and timeline
- Developer name
- URL reference for LEED certification
- URL reference for project information
- Verification of elevator and accessible parking provisions
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectIdentity(BaseModel):
    project_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    developer_name: Optional[str] = None


class TransitInfo(BaseModel):
    nearest_station_name: Optional[str] = None
    rail_system: Optional[str] = None  # DART / Metro / CapMetro
    rail_line: Optional[str] = None
    distance_to_station: Optional[str] = None  # e.g., "0.3 miles", "1500 ft"
    transit_urls: List[str] = Field(default_factory=list)


class SizeInfo(BaseModel):
    total_sqft: Optional[str] = None
    office_sqft: Optional[str] = None
    stories: Optional[str] = None
    height: Optional[str] = None
    size_urls: List[str] = Field(default_factory=list)


class LEEDInfo(BaseModel):
    leed_status: Optional[str] = None  # "registered" or "certified" (or targeted)
    leed_level: Optional[str] = None   # "Silver", "Gold", "Platinum" (or targeted level)
    leed_urls: List[str] = Field(default_factory=list)


class StatusInfo(BaseModel):
    current_status: Optional[str] = None  # e.g., "under construction", "completed", "planning"
    completion_date: Optional[str] = None  # e.g., "Jan 2025", "2024-06", "Q4 2025"
    status_urls: List[str] = Field(default_factory=list)


class ADAInfo(BaseModel):
    elevators_present: Optional[str] = None  # "yes/no" or description
    elevator_car_dimensions: Optional[str] = None  # e.g., '51" x 68"' or '51 by 68 inches'
    total_parking_spaces: Optional[str] = None
    accessible_parking_spaces: Optional[str] = None
    van_accessible_spaces: Optional[str] = None
    accessible_restroom_stall: Optional[str] = None  # "yes/no" or details
    restroom_door_width: Optional[str] = None  # e.g., "32 inches"
    ada_urls: List[str] = Field(default_factory=list)


class RegulatoryInfo(BaseModel):
    zoning_allows_office: Optional[str] = None  # brief statement as claimed in answer
    zoning_urls: List[str] = Field(default_factory=list)
    code_height_compliance: Optional[str] = None  # statement or citation snippet
    code_urls: List[str] = Field(default_factory=list)


class Documentation(BaseModel):
    project_info_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class ProjectExtraction(BaseModel):
    identity: Optional[ProjectIdentity] = None
    transit: Optional[TransitInfo] = None
    size: Optional[SizeInfo] = None
    leed: Optional[LEEDInfo] = None
    status: Optional[StatusInfo] = None
    ada: Optional[ADAInfo] = None
    regulatory: Optional[RegulatoryInfo] = None
    docs: Optional[Documentation] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
    Extract the requested structured information for a single commercial office development project mentioned in the answer.
    Return all fields as strings exactly as stated in the answer when applicable. For any URL lists, include every URL explicitly cited in the answer that is relevant to the specific topic.

    Output JSON schema (null for any missing fields):

    {
      "identity": {
        "project_name": string | null,
        "address": string | null,
        "city": string | null,
        "state": string | null,
        "developer_name": string | null
      },
      "transit": {
        "nearest_station_name": string | null,
        "rail_system": string | null,             // e.g., DART, Metro, CapMetro
        "rail_line": string | null,
        "distance_to_station": string | null,     // e.g., "0.3 miles", "1500 ft"
        "transit_urls": string[]                  // URLs supporting station, line, or distance (maps/transit/project docs)
      },
      "size": {
        "total_sqft": string | null,              // keep the original formatting, e.g., "250,000 sq ft"
        "office_sqft": string | null,             // keep formatting
        "stories": string | null,                 // number of stories if available
        "height": string | null,                  // height text (e.g., "350 ft")
        "size_urls": string[]                     // URLs supporting sizes and stories/height
      },
      "leed": {
        "leed_status": string | null,             // "registered", "certified", or similar
        "leed_level": string | null,              // "Silver", "Gold", or "Platinum" (or "targeting Silver", etc.)
        "leed_urls": string[]                     // URLs to USGBC/GBCI LEED directory or official certification documentation
      },
      "status": {
        "current_status": string | null,          // "under construction", "completed", "planning", etc.
        "completion_date": string | null,         // expected or actual completion date as stated
        "status_urls": string[]                   // URLs supporting status and timeline
      },
      "ada": {
        "elevators_present": string | null,       // "yes", "no", or descriptive text indicating elevators included
        "elevator_car_dimensions": string | null, // e.g., '51" x 68"' or equivalent
        "total_parking_spaces": string | null,
        "accessible_parking_spaces": string | null,
        "van_accessible_spaces": string | null,
        "accessible_restroom_stall": string | null, // "yes", "no", or description
        "restroom_door_width": string | null,       // e.g., "32 inches"
        "ada_urls": string[]                        // URLs supporting elevator, parking ratios, restrooms, door widths
      },
      "regulatory": {
        "zoning_allows_office": string | null,    // short statement extracted from answer
        "zoning_urls": string[],                  // URLs to zoning map/code or official docs supporting office use
        "code_height_compliance": string | null,  // extracted statement supporting code/height compliance
        "code_urls": string[]                     // URLs to building code/city code or official compliance docs
      },
      "docs": {
        "project_info_urls": string[],            // Primary project information URLs verifying existence and specs
        "other_urls": string[]                    // Any other general URLs cited in the answer
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - Do not invent or infer information not stated.
    - For any URL list, include only valid URLs mentioned in the answer text (including markdown links).
    - If the answer mentions multiple projects, focus on the first that best matches the constraints; otherwise return nulls if unclear.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Keep only plausible URLs
    return [u for u in urls if isinstance(u, str) and len(u.strip()) >= 8 and "." in u]


def _combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in _normalize_list(lst or []):
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Extract first integer sequence from text (e.g., "250,000 sq ft" -> 250000)
    m = re.search(r"(\d[\d,\.]*)", text)
    if not m:
        return None
    try:
        num = m.group(1).replace(",", "")
        if "." in num:
            return int(float(num))
        return int(num)
    except Exception:
        return None


def _parse_inches_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Try patterns like 32", 32 inches, 32-in, 32 in
    m = re.search(r"(\d+(\.\d+)?)\s*(?:\"|inches|inch|in)\b", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _stories_int(stories_text: Optional[str]) -> Optional[int]:
    return _parse_int_from_text(stories_text)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_project_identity(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="project_identity_and_type",
        desc="Project is clearly identified and is a commercial office development in an allowed city",
        parent=root,
        critical=True
    )

    identity = ex.identity or ProjectIdentity()
    docs = ex.docs or Documentation()
    general_sources = _combine_sources(docs.project_info_urls, docs.other_urls)

    # 1) Name + address + city provided (existence check)
    provided = all([
        isinstance(identity.project_name, str) and identity.project_name.strip(),
        isinstance(identity.address, str) and identity.address.strip(),
        isinstance(identity.city, str) and identity.city.strip()
    ])
    evaluator.add_custom_node(
        result=provided,
        id="name_address_city_provided",
        desc="Provides project name, full address, and city",
        parent=parent,
        critical=True
    )

    # 2) Allowed city
    allowed_city_node = evaluator.add_leaf(
        id="allowed_city",
        desc="Project is located in Dallas, Houston, or Austin, Texas",
        parent=parent,
        critical=True
    )
    city = identity.city or ""
    state = identity.state or "Texas"
    claim_allowed_city = f"The project is located in {city}, {state}, and the city is one of: Dallas, Houston, Austin (Texas)."
    await evaluator.verify(
        claim=claim_allowed_city,
        node=allowed_city_node,
        sources=general_sources,
        additional_instruction="Confirm the city is explicitly Dallas, Houston, or Austin (Texas). If the source shows a suburb or a different city outside these three, mark as not supported."
    )

    # 3) Commercial office development
    office_dev_node = evaluator.add_leaf(
        id="commercial_office_development",
        desc="Project is a commercial office development (office is primary or a significant component)",
        parent=parent,
        critical=True
    )
    size = ex.size or SizeInfo()
    claim_office_dev = (
        f"The project '{identity.project_name or ''}' is a commercial office development, "
        f"with an office component (stated office size: {size.office_sqft or 'unknown'})."
    )
    office_sources = _combine_sources(size.size_urls, general_sources)
    await evaluator.verify(
        claim=claim_office_dev,
        node=office_dev_node,
        sources=office_sources,
        additional_instruction="Accept as commercial office development if the project is primarily office or includes a substantial office component (>= 50,000 sf). Marketing/official pages suffice."
    )


async def verify_transit(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="transit_proximity_requirement",
        desc="Project meets the location & transit proximity requirement",
        parent=root,
        critical=True
    )
    transit = ex.transit or TransitInfo()
    docs = ex.docs or Documentation()
    general_sources = _combine_sources(docs.project_info_urls, docs.other_urls)
    transit_sources = _combine_sources(transit.transit_urls, general_sources)

    # 1) Nearest station and line
    station_line_node = evaluator.add_leaf(
        id="nearest_station_and_line",
        desc="Identifies the nearest qualifying rail station and specifies the station name and rail line (DART/Metro/CapMetro as applicable)",
        parent=parent,
        critical=True
    )
    claim_station_line = (
        f"The nearest qualifying rail station is '{transit.nearest_station_name or 'UNKNOWN'}' on the "
        f"'{transit.rail_line or 'UNKNOWN LINE'}' line of {transit.rail_system or 'UNKNOWN SYSTEM'}."
    )
    await evaluator.verify(
        claim=claim_station_line,
        node=station_line_node,
        sources=transit_sources,
        additional_instruction="Qualifying systems: DART (Dallas), METRO rail (Houston), or CapMetro rail (Austin). Verify station name and line are correctly identified on the provided sources (project page, transit agency page, or map)."
    )

    # 2) Distance within 1/2 mile
    distance_node = evaluator.add_leaf(
        id="distance_within_half_mile",
        desc="States and supports that the project is within 1/2 mile (2,640 feet) of the qualifying rail station",
        parent=parent,
        critical=True
    )
    claim_distance = (
        f"The project is within 1/2 mile (2,640 feet) of the station '{transit.nearest_station_name or 'UNKNOWN'}'; "
        f"the stated distance is {transit.distance_to_station or 'unknown'}."
    )
    await evaluator.verify(
        claim=claim_distance,
        node=distance_node,
        sources=transit_sources,
        additional_instruction="If a precise distance is stated on provided sources or a map link supports it, accept. If unclear or appears greater than 0.5 miles, mark as not supported."
    )


async def verify_building_size(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="building_size_requirement",
        desc="Project meets minimum building size requirements",
        parent=root,
        critical=True
    )
    size = ex.size or SizeInfo()
    docs = ex.docs or Documentation()
    sources = _combine_sources(size.size_urls, docs.project_info_urls, docs.other_urls)

    # 1) Total GSF >= 100k
    total_node = evaluator.add_leaf(
        id="total_gsf_at_least_100k",
        desc="Provides total building square footage and verifies it is at least 100,000 square feet",
        parent=parent,
        critical=True
    )
    claim_total = f"The total building size is {size.total_sqft or 'unknown'}, which is at least 100,000 square feet."
    await evaluator.verify(
        claim=claim_total,
        node=total_node,
        sources=sources,
        additional_instruction="Confirm the stated total GSF on the provided source(s). If total is not provided or is < 100,000 sf, mark as not supported."
    )

    # 2) Office GSF >= 50k
    office_node = evaluator.add_leaf(
        id="office_gsf_at_least_50k",
        desc="Provides office component square footage and verifies it is at least 50,000 square feet",
        parent=parent,
        critical=True
    )
    claim_office = f"The office component is {size.office_sqft or 'unknown'}, which is at least 50,000 square feet."
    await evaluator.verify(
        claim=claim_office,
        node=office_node,
        sources=sources,
        additional_instruction="Confirm the stated office GSF on the provided source(s). If office area is not provided or is < 50,000 sf, mark as not supported."
    )

    # 3) Stories or height provided (existence)
    has_stories_or_height = bool((size.stories and size.stories.strip()) or (size.height and size.height.strip()))
    evaluator.add_custom_node(
        result=has_stories_or_height,
        id="stories_or_height_provided",
        desc="Provides number of stories or building height",
        parent=parent,
        critical=True
    )


async def verify_leed(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="leed_requirement",
        desc="Project is registered for or has achieved required LEED certification level and provides required LEED documentation",
        parent=root,
        critical=True
    )
    leed = ex.leed or LEEDInfo()
    docs = ex.docs or Documentation()
    sources = _combine_sources(leed.leed_urls, docs.project_info_urls, docs.other_urls)

    # 1) LEED registered or certified
    reg_cert_node = evaluator.add_leaf(
        id="leed_registered_or_certified",
        desc="States whether the project is LEED-registered or LEED-certified",
        parent=parent,
        critical=True
    )
    claim_reg_cert = f"The project is {leed.leed_status or 'unknown'} under LEED (registered or certified)."
    await evaluator.verify(
        claim=claim_reg_cert,
        node=reg_cert_node,
        sources=sources,
        additional_instruction="Verify on USGBC/GBCI or clearly official documentation. Accept 'registered' or 'certified'. If only marketing claims without credible source, mark as not supported."
    )

    # 2) LEED level Silver or higher
    level_node = evaluator.add_leaf(
        id="leed_level_silver_or_higher",
        desc="LEED level is Silver, Gold, or Platinum (Silver or higher) and is supported by documentation",
        parent=parent,
        critical=True
    )
    claim_level = f"The LEED level for the project is {leed.leed_level or 'unknown'}, which is Silver, Gold, or Platinum (Silver or higher), or an explicitly stated target at Silver+."
    await evaluator.verify(
        claim=claim_level,
        node=level_node,
        sources=sources,
        additional_instruction="Validate that the level is at least Silver (50+ points). If only 'registered' with no level or level below Silver, mark as not supported."
    )

    # 3) LEED directory or official URL present and relevant
    leed_url_node = evaluator.add_leaf(
        id="leed_directory_or_official_url",
        desc="Provides a URL to the project's LEED directory entry or official certification documentation",
        parent=parent,
        critical=True
    )
    claim_url = "At least one of the provided URLs is the project's official LEED directory entry or official LEED certification documentation."
    await evaluator.verify(
        claim=claim_url,
        node=leed_url_node,
        sources=leed.leed_urls or sources,
        additional_instruction="Prefer USGBC or GBCI links; official PDF certificates also acceptable. Marketing pages alone are insufficient for this node."
    )


async def verify_development_status(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="development_status_and_timeline_requirement",
        desc="Project status and completion timing meet the requirement and are reported",
        parent=root,
        critical=True
    )
    status = ex.status or StatusInfo()
    docs = ex.docs or Documentation()
    sources = _combine_sources(status.status_urls, docs.project_info_urls, docs.other_urls)

    # 1) Status in allowed set
    status_allowed = evaluator.add_leaf(
        id="status_in_allowed_set",
        desc="Project is (a) under construction, (b) completed since January 2024, or (c) in advanced planning with publicly available documentation",
        parent=parent,
        critical=True
    )
    claim_status = (
        f"The project status is '{status.current_status or 'unknown'}' with completion date '{status.completion_date or 'unknown'}'. "
        f"This satisfies one of: under construction; completed on/after January 1, 2024; or advanced planning with public documentation."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_allowed,
        sources=sources,
        additional_instruction="If 'completed', ensure the completion date is January 1, 2024 or later. If 'planning', require credible public documentation indicating advanced planning (e.g., site plan approval, permits)."
    )

    # 2) Completion date provided (and supported)
    completion_node = evaluator.add_leaf(
        id="completion_date_provided",
        desc="Provides expected or actual completion date",
        parent=parent,
        critical=True
    )
    claim_date = f"The expected or actual completion date for the project is stated as: {status.completion_date or 'unknown'}."
    await evaluator.verify(
        claim=claim_date,
        node=completion_node,
        sources=sources,
        additional_instruction="Verify that the completion date is explicitly present on a credible source. If absent, mark as not supported."
    )


async def verify_ada(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="ada_accessibility_requirements",
        desc="Project meets the specified ADA-related constraints",
        parent=root,
        critical=True
    )

    ada = ex.ada or ADAInfo()
    size = ex.size or SizeInfo()
    docs = ex.docs or Documentation()
    sources = _combine_sources(ada.ada_urls, size.size_urls, docs.project_info_urls, docs.other_urls)

    # 1) Elevators present if 3+ stories (or requirement satisfied)
    elevators_node = evaluator.add_leaf(
        id="elevators_present_if_3plus_stories",
        desc="If the building is 3+ stories, verifies elevators are included",
        parent=parent,
        critical=True
    )
    stories_n = _stories_int(size.stories)
    claim_elevators = (
        f"The building includes elevators (elevators stated: {ada.elevators_present or 'unknown'}). "
        f"If the building has 3 or more stories (stories stated: {size.stories or 'unknown'}), this is required for ADA compliance; "
        f"otherwise, the requirement is still considered satisfied if elevators are included or the building is <3 stories."
    )
    await evaluator.verify(
        claim=claim_elevators,
        node=elevators_node,
        sources=sources,
        additional_instruction="If sources indicate 3+ stories, require explicit evidence that elevators are included. If <3 stories, accept as satisfied even if elevators not explicitly stated."
    )

    # 2) Elevator car dimensions
    elevator_dims_node = evaluator.add_leaf(
        id="elevator_car_dimensions",
        desc="Provides evidence that elevator cars meet minimum ADA dimensions of 51 inches deep by 68 inches wide",
        parent=parent,
        critical=True
    )
    claim_dims = (
        f"The building's elevator car dimensions meet or exceed 51 inches deep by 68 inches wide "
        f"(stated: {ada.elevator_car_dimensions or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_dims,
        node=elevator_dims_node,
        sources=sources,
        additional_instruction="Accept specs from official design documents, code compliance sheets, or credible building specifications. If dimensions are not given, mark as not supported."
    )

    # 3) Accessible parking ratio if total 500–1000 spaces
    acc_parking_node = evaluator.add_leaf(
        id="accessible_parking_ratio_if_500_to_1000",
        desc="For parking 500–1000 total spaces, verifies accessible spaces are at least 2% of total (with evidence for applicability and ratio)",
        parent=parent,
        critical=True
    )
    total_spaces = _parse_int_from_text(ada.total_parking_spaces)
    accessible_spaces = _parse_int_from_text(ada.accessible_parking_spaces)
    claim_parking = (
        f"The project's total parking is {ada.total_parking_spaces or 'unknown'} and accessible parking is "
        f"{ada.accessible_parking_spaces or 'unknown'}. If the total parking is within 500–1000 inclusive, "
        f"accessible spaces must be at least 2% of total; otherwise this requirement is not applicable and considered satisfied."
    )
    await evaluator.verify(
        claim=claim_parking,
        node=acc_parking_node,
        sources=sources,
        additional_instruction="Check if total parking is explicitly between 500 and 1000. If so, verify accessible >= 2% of total. If not in range or total not specified, accept as not applicable only if sources do not show 500–1000; if ambiguous, mark as not supported."
    )

    # 4) Van-accessible ratio (>= 1 per 6 accessible)
    van_ratio_node = evaluator.add_leaf(
        id="van_accessible_ratio",
        desc="Van-accessible spaces are provided at a minimum of 1 per 6 accessible spaces",
        parent=parent,
        critical=True
    )
    claim_van = (
        f"The project provides van-accessible spaces at least 1 per 6 accessible spaces "
        f"(van-accessible: {ada.van_accessible_spaces or 'unknown'}, accessible: {ada.accessible_parking_spaces or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_van,
        node=van_ratio_node,
        sources=sources,
        additional_instruction="If accessible spaces count is known, verify at least 1 out of every 6 are van-accessible. If counts are not stated, mark as not supported."
    )

    # 5) Accessible restroom stall present
    acc_restroom_node = evaluator.add_leaf(
        id="accessible_restroom_stall",
        desc="At least one accessible restroom stall per facility is provided",
        parent=parent,
        critical=True
    )
    claim_restroom = (
        f"The project provides at least one accessible restroom stall per facility "
        f"(stated: {ada.accessible_restroom_stall or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_restroom,
        node=acc_restroom_node,
        sources=sources,
        additional_instruction="Look for restrooms/accessibility sections in official plans/spec sheets or credible building documentation."
    )

    # 6) Restroom door width >= 32 inches
    door_width_node = evaluator.add_leaf(
        id="restroom_door_width",
        desc="Accessible restroom doors are minimum 32 inches wide",
        parent=parent,
        critical=True
    )
    claim_door = f"Accessible restroom door widths meet or exceed 32 inches (stated: {ada.restroom_door_width or 'unknown'})."
    await evaluator.verify(
        claim=claim_door,
        node=door_width_node,
        sources=sources,
        additional_instruction="If door width is not stated in credible sources, mark as not supported."
    )


async def verify_regulatory(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="regulatory_compliance_constraints",
        desc="Project meets stated zoning/code-related constraints to the extent required by the constraints list",
        parent=root,
        critical=True
    )
    reg = ex.regulatory or RegulatoryInfo()
    docs = ex.docs or Documentation()
    zoning_sources = _combine_sources(reg.zoning_urls, docs.project_info_urls, docs.other_urls)
    code_sources = _combine_sources(reg.code_urls, docs.project_info_urls, docs.other_urls)

    # 1) Zoning allows office use
    zoning_node = evaluator.add_leaf(
        id="zoning_allows_office_use",
        desc="Provides documentation that local zoning permits commercial office use (or mixed-use including office) for the project site",
        parent=parent,
        critical=True
    )
    claim_zoning = (
        f"Local zoning permits commercial office use (or mixed-use including office) at the project site "
        f"(stated/extracted: {reg.zoning_allows_office or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_zoning,
        node=zoning_node,
        sources=zoning_sources,
        additional_instruction="Accept official zoning map/code pages or official planning approvals confirming office is allowed."
    )

    # 2) Code and height compliance supported
    code_node = evaluator.add_leaf(
        id="code_and_height_compliance_supported",
        desc="Provides support indicating compliance with Texas building codes and applicable local height restrictions",
        parent=parent,
        critical=True
    )
    claim_code = (
        f"The project is compliant with Texas building codes and applicable local height restrictions "
        f"(supporting statement: {reg.code_height_compliance or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_code,
        node=code_node,
        sources=code_sources,
        additional_instruction="Evidence can be official code references, permits, approvals, or authoritative compliance statements. Marketing claims without credible citation are insufficient."
    )


async def verify_documentation(evaluator: Evaluator, root, ex: ProjectExtraction):
    parent = evaluator.add_parallel(
        id="documentation_and_attribution_requirements",
        desc="Provides the required references and developer attribution",
        parent=root,
        critical=True
    )
    identity = ex.identity or ProjectIdentity()
    docs = ex.docs or Documentation()
    leed = ex.leed or LEEDInfo()
    size = ex.size or SizeInfo()
    transit = ex.transit or TransitInfo()
    status = ex.status or StatusInfo()
    ada = ex.ada or ADAInfo()
    reg = ex.regulatory or RegulatoryInfo()

    all_sources = _combine_sources(
        docs.project_info_urls, docs.other_urls,
        size.size_urls, transit.transit_urls, status.status_urls,
        ada.ada_urls, reg.zoning_urls, reg.code_urls, leed.leed_urls
    )

    # 1) Project info URL present and supports basic details
    info_url_node = evaluator.add_leaf(
        id="project_info_url",
        desc="Provides a valid URL reference verifying project existence and basic details/specifications",
        parent=parent,
        critical=True
    )
    claim_info_url = "At least one provided URL is a valid project information page that verifies the project's existence and basic specifications."
    await evaluator.verify(
        claim=claim_info_url,
        node=info_url_node,
        sources=docs.project_info_urls or all_sources,
        additional_instruction="Accept developer or official project pages, credible news releases, or authoritative listings that state key specs such as location, size, status."
    )

    # 2) Developer name provided (and supported)
    developer_node = evaluator.add_leaf(
        id="developer_name_provided",
        desc="Identifies the developer name",
        parent=parent,
        critical=True
    )
    claim_developer = f"The developer of the project is stated as: {identity.developer_name or 'unknown'}."
    await evaluator.verify(
        claim=claim_developer,
        node=developer_node,
        sources=all_sources,
        additional_instruction="Verify that the developer name is explicitly stated on credible sources."
    )

    # 3) Claims verifiable from sources (holistic check)
    claims_supported_node = evaluator.add_leaf(
        id="claims_verifiable_from_sources",
        desc="All critical claims are supported by public documentation or official sources (citations provided for required claims)",
        parent=parent,
        critical=True
    )
    claim_supported = (
        "The provided set of URLs collectively support the critical claims regarding location/city, transit proximity, "
        "building sizes, LEED status/level, development status/timeline, ADA provisions, and regulatory compliance."
    )
    await evaluator.verify(
        claim=claim_supported,
        node=claims_supported_node,
        sources=all_sources,
        additional_instruction="Consider whether each critical requirement in the task can be verified from at least one of the provided URLs. If any critical area lacks credible support, mark as not supported."
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
    """
    Evaluate an answer for the Texas office project transit/LEED/ADA task using the Mind2Web2 framework.
    """
    # Initialize evaluator (root is non-critical by design; child critical groups will gate the score)
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

    # Extract structured info
    extracted: ProjectExtraction = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Optional contextual info
    evaluator.add_ground_truth({
        "allowed_cities": ["Dallas", "Houston", "Austin"],
        "transit_systems": {
            "Dallas": "DART",
            "Houston": "METRO Rail",
            "Austin": "CapMetro Rail"
        },
        "size_thresholds": {"total_min_sf": 100000, "office_min_sf": 50000},
        "leed_min_level": "Silver",
        "status_requirement": "Under construction OR completed since 2024-01 OR advanced planning with public documentation",
        "ada_requirements": {
            "elevators_for_3plus_stories": True,
            "elevator_car_min_inches": {"depth": 51, "width": 68},
            "accessible_parking_ratio_if_500_1000": ">=2%",
            "van_accessible_ratio": ">= 1 per 6 accessible",
            "restroom_door_min_width_inches": 32
        }
    })

    # Build verification tree according to rubric
    await verify_project_identity(evaluator, root, extracted)
    await verify_transit(evaluator, root, extracted)
    await verify_building_size(evaluator, root, extracted)
    await verify_leed(evaluator, root, extracted)
    await verify_development_status(evaluator, root, extracted)
    await verify_ada(evaluator, root, extracted)
    await verify_regulatory(evaluator, root, extracted)
    await verify_documentation(evaluator, root, extracted)

    # Return standard evaluation summary
    return evaluator.get_summary()