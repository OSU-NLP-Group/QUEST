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
TASK_ID = "eclipse_2026_western_na_locations"
TASK_DESCRIPTION = """
For the total lunar eclipse occurring on March 3, 2026, identify 3-4 optimal viewing locations in western North America that meet professional astronomical observation standards. For each location, provide the following information with supporting reference URLs: (1) Location name with an authoritative reference URL, (2) Elevation in feet above sea level, (3) Dark sky certification status (e.g., International Dark Sky Park designation or equivalent), (4) Public accessibility status during the eclipse timeframe. Requirements: All locations must be situated in western North America (western United States or western Canada); all locations must have elevations between 5,000 and 8,000 feet above sea level to ensure optimal atmospheric observation conditions; all locations must have verified visibility of the totality phase of the eclipse (not merely partial eclipse visibility); each location should have an official dark sky designation such as International Dark Sky Park certification to minimize light pollution; all locations must be accessible to the public during the eclipse viewing hours (early morning of March 3, 2026); provide authoritative reference URLs for each piece of information to verify the data.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class LocationItem(BaseModel):
    name: Optional[str] = None
    region: Optional[str] = None  # e.g., "Utah, USA" or "British Columbia, Canada"
    core_url: Optional[str] = None  # authoritative page about the place

    elevation_str: Optional[str] = None  # keep as string as provided in answer (e.g., "7,200 ft (2,195 m)")
    elevation_source_urls: List[str] = Field(default_factory=list)

    dark_sky_status: Optional[str] = None  # e.g., "International Dark Sky Park"
    dark_sky_source_urls: List[str] = Field(default_factory=list)

    access_status: Optional[str] = None  # e.g., "Open 24 hours" / "Open to public before dawn"
    access_source_urls: List[str] = Field(default_factory=list)

    totality_source_urls: List[str] = Field(default_factory=list)  # sources to verify totality visibility


class LocationsExtraction(BaseModel):
    locations: List[LocationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_locations() -> str:
    return """
    You will extract up to 4 eclipse-viewing locations from the answer. Return a JSON object with a 'locations' array. For each location, extract the fields below EXACTLY as they appear in the answer without inventing anything. If a field is not present, return null (for strings) or an empty array (for URLs).

    For each location object, extract:
    - name: The location name as stated.
    - region: A human-readable region (state/province and country), e.g., "Utah, USA" or "British Columbia, Canada".
    - core_url: An authoritative reference URL about the place (official park/site/city, IDA/darksky.org, government or Wikipedia). If not provided in answer, return null.
    - elevation_str: Elevation text as given in the answer (e.g., "7,200 ft (2,195 m)" or "2200 m"). Keep the units/format as written.
    - elevation_source_urls: All URLs in the answer that specifically support the elevation data. Return [] if none.
    - dark_sky_status: The stated dark sky designation or equivalent (e.g., "International Dark Sky Park", "Dark Sky Community"). If not given, return null.
    - dark_sky_source_urls: All URLs that support the dark sky designation. Return [] if none.
    - access_status: Public accessibility status as stated in the answer for early-morning viewing (e.g., "Open 24 hours", "Open to public", "Visitor access permitted before dawn in March"). If not given, return null.
    - access_source_urls: All URLs that support the public access/visitor policy. Return [] if none.
    - totality_source_urls: All URLs that support that the total phase (totality) of the March 3, 2026 total lunar eclipse is visible from this location (or its city/park/region). Return [] if none.

    IMPORTANT:
    - Only include URLs explicitly present in the answer text. Do not infer or create new URLs.
    - If more than 4 locations are listed in the answer, include only the first 4 in order of appearance.
    - If fewer than 3 locations are provided in the answer, still return whatever is there.
    - Keep strings as free text; do NOT normalize or convert units.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
US_WESTERN_STATES_FULL = {
    "washington", "oregon", "california",
    "nevada", "idaho", "montana", "wyoming", "utah", "colorado",
    "arizona", "new mexico", "alaska", "hawaii"
}
US_WESTERN_STATE_CODES = {
    "wa", "or", "ca", "nv", "id", "mt", "wy", "ut", "co", "az", "nm", "ak", "hi"
}
CANADA_WEST_FULL = {
    "british columbia", "alberta", "saskatchewan",
    "yukon", "northwest territories", "nunavut"
}
COUNTRY_TOKENS = {"usa", "u.s.a", "united states", "canada"}


def is_western_north_america(region_text: Optional[str]) -> bool:
    if not region_text:
        return False
    t = region_text.strip().lower()
    # Quick country sanity
    if not any(ct in t for ct in COUNTRY_TOKENS):
        # Still allow strings like "Arizona" without country token
        pass

    # Check for US state full names or codes
    for s in US_WESTERN_STATES_FULL:
        if s in t:
            return True
    # Look for 2-letter codes as a standalone token or in parentheses
    for code in US_WESTERN_STATE_CODES:
        # token-boundary match
        if re.search(rf'(^|\W){code}(\W|$)', t):
            return True

    # Check for Canada western provinces/territories full names
    for p in CANADA_WEST_FULL:
        if p in t:
            return True

    return False


def parse_elevation_to_feet(elevation_str: Optional[str]) -> Optional[float]:
    """
    Best-effort parse of an elevation string to feet.
    Handles patterns like:
      "7,200 ft", "7200 feet", "2195 m", "2,195 meters", "6,500–7,000 ft"
    Returns the first numeric figure normalized to feet (float), or None if cannot parse.
    """
    if not elevation_str:
        return None

    s = elevation_str.lower()
    # Normalize various dashes
    s = s.replace("–", "-").replace("—", "-")
    # Capture number + optional unit right after
    # e.g., "7,200 ft", "2195 m", "6,500-7,000 ft"
    matches = list(re.finditer(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*(ft|feet|foot|m|meter|meters|metre|metres))?', s))
    if not matches:
        return None

    # Take the first numeric occurrence
    num_txt, unit = matches[0].group(1), matches[0].group(2)
    try:
        val = float(num_txt.replace(",", ""))
    except Exception:
        return None

    # If unit is meters-ish, convert
    if unit and unit.startswith('m'):
        return val * 3.28084
    # If explicitly feet or missing unit but "ft/feet" appears elsewhere, assume feet
    if unit in (None, 'ft', 'feet', 'foot'):
        # If no unit captured but "m" appears after this number, try to detect "m" elsewhere
        if unit is None and re.search(r'\b(m|meter|meters|metre|metres)\b', s):
            # Mixed units string; but first number has no unit while later shows meters.
            # Assume feet if "ft" appears anywhere
            if re.search(r'\b(ft|feet|foot)\b', s):
                return val
            # Otherwise likely meters
            return val * 3.28084
        return val

    return None


def unique_by_name_keep_order(items: List[LocationItem]) -> List[LocationItem]:
    seen = set()
    out: List[LocationItem] = []
    for item in items:
        key = (item.name or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_location_identification_nodes(
    evaluator: Evaluator,
    parent,
    locations: List[LocationItem],
) -> None:
    """
    Build and verify:
      - at least 3 and at most 4 distinct locations provided
      - for each location: core URL supports the named location
      - region is in western North America (custom check) and supported by core URL
      - totality visibility verified by cited sources
    """
    loc_root = evaluator.add_parallel(
        id="location_identification",
        desc="Core location data including identification of 3-4 suitable sites in western North America with verified eclipse visibility",
        parent=parent,
        critical=True,
    )

    # 1) Count check
    evaluator.add_custom_node(
        result=(3 <= len(locations) <= 4),
        id="locations_provided",
        desc="Three to four distinct viewing locations are identified and named",
        parent=loc_root,
        critical=True
    )

    # 2) Per-location checks
    for idx, loc in enumerate(locations):
        node_i = evaluator.add_parallel(
            id=f"loc_{idx}_ident",
            desc=f"Location #{idx+1} identification and totality verification",
            parent=loc_root,
            critical=True
        )

        # Existence of core fields
        evaluator.add_custom_node(
            result=bool(loc.name and loc.name.strip()) and bool(loc.core_url and loc.core_url.strip()),
            id=f"loc_{idx}_provided",
            desc="Location name and core reference URL are provided",
            parent=node_i,
            critical=True
        )

        # Reference URL corresponds to the named location
        ref_leaf = evaluator.add_leaf(
            id=f"loc_{idx}_reference_url_match",
            desc="Authoritative reference URL corresponds to the named location",
            parent=node_i,
            critical=True
        )
        if loc.name and loc.core_url:
            claim = f"This webpage is about the location named '{loc.name}'."
            await evaluator.verify(
                claim=claim,
                node=ref_leaf,
                sources=loc.core_url,
                additional_instruction="Focus on whether the page is about the named place (park, observatory, city site, IDA/DarkSky, NPS/state/municipal, or Wikipedia). Minor naming variations are acceptable."
            )

        # Region in western North America (custom)
        evaluator.add_custom_node(
            result=is_western_north_america(loc.region),
            id=f"loc_{idx}_in_western_na",
            desc="Location is situated in western North America (western United States or western Canada)",
            parent=node_i,
            critical=True
        )

        # Region supported by the core URL page (if possible)
        region_verify = evaluator.add_leaf(
            id=f"loc_{idx}_region_supported",
            desc="Core reference URL indicates the stated state/province and country for the location",
            parent=node_i,
            critical=True
        )
        if loc.region and loc.core_url and loc.name:
            claim = f"This page indicates that '{loc.name}' is located in {loc.region}."
            await evaluator.verify(
                claim=claim,
                node=region_verify,
                sources=loc.core_url,
                additional_instruction="Look for mentions of the state/province and country (e.g., Utah, USA; British Columbia, Canada) in the page content or headers."
            )

        # Totality visibility sources present
        evaluator.add_custom_node(
            result=(len(loc.totality_source_urls) > 0),
            id=f"loc_{idx}_totality_sources_present",
            desc="At least one source URL is provided to verify totality visibility",
            parent=node_i,
            critical=True
        )

        # Verify totality visibility for March 3, 2026
        totality_leaf = evaluator.add_leaf(
            id=f"loc_{idx}_totality_visibility",
            desc="Verified that totality phase of the 2026-03-03 lunar eclipse is visible from this location",
            parent=node_i,
            critical=True
        )
        if loc.name and loc.totality_source_urls:
            claim = f"The total phase (totality) of the March 3, 2026 total lunar eclipse is visible from {loc.name}."
            await evaluator.verify(
                claim=claim,
                node=totality_leaf,
                sources=loc.totality_source_urls,
                additional_instruction="Confirm the source explicitly refers to the 2026-03-03 total lunar eclipse and indicates totality visibility for the location or its immediate city/park/region. Do NOT accept sources mentioning only partial visibility."
            )


async def build_elevation_requirement_nodes(
    evaluator: Evaluator,
    parent,
    locations: List[LocationItem],
) -> None:
    elev_root = evaluator.add_parallel(
        id="elevation_requirements",
        desc="Elevation data and compliance verification for optimal atmospheric observation conditions",
        parent=parent,
        critical=True
    )

    for idx, loc in enumerate(locations):
        node_i = evaluator.add_parallel(
            id=f"loc_{idx}_elevation",
            desc=f"Location #{idx+1} elevation verification",
            parent=elev_root,
            critical=True
        )

        # Elevation stated
        evaluator.add_custom_node(
            result=bool(loc.elevation_str and loc.elevation_str.strip()),
            id=f"loc_{idx}_elevation_stated",
            desc="Elevation above sea level stated (as text) for this location",
            parent=node_i,
            critical=True
        )

        # Range compliance 5,000–8,000 ft
        feet_val = parse_elevation_to_feet(loc.elevation_str)
        in_range = feet_val is not None and 5000.0 <= feet_val <= 8000.0
        evaluator.add_custom_node(
            result=in_range,
            id=f"loc_{idx}_elevation_in_range",
            desc="Elevation is between 5,000 and 8,000 feet above sea level",
            parent=node_i,
            critical=True
        )

        # Elevation sources present
        evaluator.add_custom_node(
            result=(len(loc.elevation_source_urls) > 0),
            id=f"loc_{idx}_elevation_sources_present",
            desc="Reference URLs provided verifying the elevation",
            parent=node_i,
            critical=True
        )

        # Elevation value supported (approximate acceptance)
        elev_leaf = evaluator.add_leaf(
            id=f"loc_{idx}_elevation_supported",
            desc="Elevation value is supported by cited sources",
            parent=node_i,
            critical=True
        )
        if loc.name and loc.elevation_str and loc.elevation_source_urls:
            claim = f"The elevation of {loc.name} is approximately {loc.elevation_str} above sea level."
            await evaluator.verify(
                claim=claim,
                node=elev_leaf,
                sources=loc.elevation_source_urls,
                additional_instruction="Verify that the referenced page supports the stated elevation. Allow reasonable rounding or unit conversions (feet/meters). Small discrepancies (±200 ft) are acceptable."
            )

        # Record parsed elevation for debugging
        evaluator.add_custom_info(
            info={"location_index": idx, "name": loc.name, "elevation_str": loc.elevation_str, "parsed_feet": feet_val},
            info_type="debug",
            info_name=f"elevation_parse_{idx}"
        )


async def build_dark_sky_nodes(
    evaluator: Evaluator,
    parent,
    locations: List[LocationItem],
) -> None:
    ds_root = evaluator.add_parallel(
        id="dark_sky_status",
        desc="Dark sky certification or designation status for each location to verify minimal light pollution",
        parent=parent,
        critical=True
    )

    for idx, loc in enumerate(locations):
        node_i = evaluator.add_parallel(
            id=f"loc_{idx}_dark",
            desc=f"Location #{idx+1} dark sky designation verification",
            parent=ds_root,
            critical=True
        )

        evaluator.add_custom_node(
            result=bool(loc.dark_sky_status and loc.dark_sky_status.strip()),
            id=f"loc_{idx}_dark_sky_designation_stated",
            desc="Dark sky status is explicitly stated for this location",
            parent=node_i,
            critical=True
        )

        evaluator.add_custom_node(
            result=(len(loc.dark_sky_source_urls) > 0),
            id=f"loc_{idx}_dark_sky_sources_present",
            desc="Reference URLs provided verifying dark sky certification or designation",
            parent=node_i,
            critical=True
        )

        ds_leaf = evaluator.add_leaf(
            id=f"loc_{idx}_dark_sky_verified",
            desc="Dark sky certification/designation is supported by cited sources",
            parent=node_i,
            critical=True
        )
        if loc.name and loc.dark_sky_status and loc.dark_sky_source_urls:
            claim = f"{loc.name} has an official dark sky designation: {loc.dark_sky_status}."
            await evaluator.verify(
                claim=claim,
                node=ds_leaf,
                sources=loc.dark_sky_source_urls,
                additional_instruction="Prefer authoritative pages (e.g., darksky.org/IDA, official park/city/government). The page should explicitly mention the designation or recognition."
            )


async def build_accessibility_nodes(
    evaluator: Evaluator,
    parent,
    locations: List[LocationItem],
) -> None:
    acc_root = evaluator.add_parallel(
        id="public_accessibility",
        desc="Public access status and practical viewing feasibility during the eclipse timeframe",
        parent=parent,
        critical=True
    )

    for idx, loc in enumerate(locations):
        node_i = evaluator.add_parallel(
            id=f"loc_{idx}_access",
            desc=f"Location #{idx+1} public accessibility verification",
            parent=acc_root,
            critical=True
        )

        evaluator.add_custom_node(
            result=bool(loc.access_status and loc.access_status.strip()),
            id=f"loc_{idx}_access_status_stated",
            desc="Public accessibility status is clearly stated for this location",
            parent=node_i,
            critical=True
        )

        evaluator.add_custom_node(
            result=(len(loc.access_source_urls) > 0),
            id=f"loc_{idx}_access_sources_present",
            desc="Supporting reference URLs verifying public access or visitor information are provided",
            parent=node_i,
            critical=True
        )

        acc_leaf = evaluator.add_leaf(
            id=f"loc_{idx}_access_verified",
            desc="Public accessibility during eclipse viewing hours is supported by sources",
            parent=node_i,
            critical=True
        )
        if loc.name and loc.access_source_urls:
            claim = f"{loc.name} is publicly accessible to visitors during overnight or pre-dawn hours appropriate for viewing the lunar eclipse on March 3, 2026."
            await evaluator.verify(
                claim=claim,
                node=acc_leaf,
                sources=loc.access_source_urls,
                additional_instruction=(
                    "Verify that general access policies allow public presence overnight or before dawn (e.g., 24-hour access, "
                    "night-sky viewing allowed, or specific hours covering pre-dawn). If a page explicitly limits access to daylight "
                    "hours only, consider this claim unsupported. Seasonal closures in early March should also be considered."
                )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for optimal March 3, 2026 total lunar eclipse viewing locations in western North America.
    """
    # Initialize evaluator (root is non-critical by design; add a critical child node for the whole analysis)
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

    # Extract structured locations
    extracted = await evaluator.extract(
        prompt=prompt_extract_locations(),
        template_class=LocationsExtraction,
        extraction_name="locations_structured"
    )

    # Post-process: unique by name, keep order; limit to first 4
    unique_locs = unique_by_name_keep_order(extracted.locations)
    processed_locs = unique_locs[:4]

    # Add a top-level critical analysis node (maps to rubric 'optimal_viewing_locations_analysis')
    analysis_node = evaluator.add_parallel(
        id="optimal_viewing_locations_analysis",
        desc="Comprehensive identification and analysis of optimal viewing locations for the March 3, 2026 total lunar eclipse in western North America",
        parent=root,
        critical=True
    )

    # Record some debug/custom info
    evaluator.add_custom_info(
        info={
            "original_count": len(extracted.locations),
            "unique_count": len(unique_locs),
            "processed_count": len(processed_locs),
            "names": [loc.name for loc in processed_locs]
        },
        info_type="processing_summary",
        info_name="location_list_summary"
    )

    # Build verification subtrees as per rubric
    await build_location_identification_nodes(evaluator, analysis_node, processed_locs)
    await build_elevation_requirement_nodes(evaluator, analysis_node, processed_locs)
    await build_dark_sky_nodes(evaluator, analysis_node, processed_locs)
    await build_accessibility_nodes(evaluator, analysis_node, processed_locs)

    # Return the summary
    return evaluator.get_summary()