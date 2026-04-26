import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "spring_2026_lunar_eclipse_research"
TASK_DESCRIPTION = """
I am planning to observe a major astronomical event in Spring 2026 and want to combine it with educational activities and professional networking. Please help me research the following:

1. Astronomical Event: Identify the total lunar eclipse occurring in Spring 2026 (March-May) that is visible from the continental United States. Provide:
   - The exact date of the eclipse
   - Confirmation that it is a total lunar eclipse (blood moon)
   - The duration of totality (in minutes)
   - Precise UTC times for when totality begins, reaches maximum, and ends
   - Information about which regions of the US can observe it
   - Local timing for at least two different US time zones (e.g., Eastern and Pacific)
   - URL references from authoritative sources (NASA, timeanddate.com, or similar astronomy websites) for all eclipse information

2. Observatory Viewing Programs: Find THREE different public observatories in the United States that are offering special viewing programs or events specifically for this eclipse. The three observatories must be in three different US states. For each observatory, provide:
   - The official name of the observatory
   - The US state where it is located
   - The type of viewing program offered (e.g., Star Party, guided viewing session, special eclipse event)
   - The specific date and start time of the eclipse viewing program
   - Confirmation that the program is open to the general public (not members-only)
   - Pricing information if available
   - URL to the observatory's official website and specific program information

3. Professional Conference: Identify ONE major professional conference organized by IEEE or a recognized astronomy/aerospace organization that occurs within 2 weeks (before or after) of the eclipse date. Provide:
   - The official name of the conference
   - Confirmation that it is organized by IEEE or a major astronomy/aerospace organization
   - The conference dates (start and end dates)
   - The venue name and location (city and state)
   - URL to the official conference website

All information must be verifiable through the provided URLs. Please ensure that the observatories are in three different states and that at least one observatory is located in the western United States.
"""


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
US_STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

WEST_STATES = {
    "Alaska", "Hawaii", "Washington", "Oregon", "California", "Nevada", "Idaho",
    "Montana", "Wyoming", "Utah", "Colorado", "Arizona", "New Mexico"
}

def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if len(s) == 2 and s.upper() in US_STATE_ABBR_TO_NAME:
        return US_STATE_ABBR_TO_NAME[s.upper()]
    # Capitalize simple input; if already full name, try to map loosely
    for full in US_STATE_ABBR_TO_NAME.values():
        if s.lower() == full.lower():
            return full
    # Try title case guess
    return s.title()

def extract_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,3})", text.replace(",", ""))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------
class EclipseUTC(BaseModel):
    totality_start_utc: Optional[str] = None
    maximum_utc: Optional[str] = None
    totality_end_utc: Optional[str] = None

class LocalZoneTimes(BaseModel):
    totality_start: Optional[str] = None
    maximum: Optional[str] = None
    totality_end: Optional[str] = None

class EclipseExtraction(BaseModel):
    date: Optional[str] = None
    eclipse_type: Optional[str] = None
    duration_minutes: Optional[str] = None
    utc_times: Optional[EclipseUTC] = None

    # Local times for at least two zones
    eastern_time: Optional[LocalZoneTimes] = None
    pacific_time: Optional[LocalZoneTimes] = None

    # Visibility
    us_visibility_statement: Optional[str] = None
    us_regions: List[str] = Field(default_factory=list)

    # Sources
    event_source_urls: List[str] = Field(default_factory=list)
    visibility_source_urls: List[str] = Field(default_factory=list)
    duration_source_urls: List[str] = Field(default_factory=list)
    timing_source_urls: List[str] = Field(default_factory=list)

class ObservatoryProgram(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    program_type: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    is_public: Optional[bool] = None
    pricing: Optional[str] = None
    website_url: Optional[str] = None
    program_url: Optional[str] = None

class ObservatoriesExtraction(BaseModel):
    observatories: List[ObservatoryProgram] = Field(default_factory=list)

class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    organization: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    website_url: Optional[str] = None
    location_url: Optional[str] = None


# ------------------------------------------------------------------------------
# Extraction Prompts
# ------------------------------------------------------------------------------
def prompt_extract_eclipse() -> str:
    return """
    Extract information about the total lunar eclipse in Spring 2026 (between March and May 2026) as presented in the answer.

    Required fields:
    - date: The exact calendar date for the eclipse (string as shown in the answer).
    - eclipse_type: The type of eclipse (expect "total lunar eclipse" or similar).
    - duration_minutes: The total duration of totality in minutes (string exactly as in the answer, may include units).
    - utc_times: An object with:
        - totality_start_utc: UTC time when totality begins
        - maximum_utc: UTC time of maximum eclipse
        - totality_end_utc: UTC time when totality ends
    - eastern_time: An object with local times in Eastern Time zone:
        - totality_start
        - maximum
        - totality_end
    - pacific_time: An object with local times in Pacific Time zone:
        - totality_start
        - maximum
        - totality_end
    - us_visibility_statement: A concise sentence or statement about US visibility from the answer.
    - us_regions: An array of region names mentioned for visibility (e.g., ["western", "central", "eastern"]).
    - event_source_urls: Array of authoritative URLs cited for event identification (NASA, timeanddate.com, etc.)
    - visibility_source_urls: Array of authoritative URLs cited for US visibility info.
    - duration_source_urls: Array of authoritative URLs cited for totality duration.
    - timing_source_urls: Array of authoritative URLs cited for UTC/local timing.

    Rules:
    - Only extract values explicitly present in the answer.
    - If a field is missing, set it to null (for scalars) or [] (for arrays).
    - For URL fields, include only valid, fully-qualified URLs as shown in the answer text.
    """

def prompt_extract_observatories() -> str:
    return """
    Extract details of up to three public observatories in the US that, per the answer, offer a special viewing program for the specified 2026 total lunar eclipse.

    Return an object with an array 'observatories' of up to three items. For each observatory element, extract:
    - name: Official name of the observatory (string)
    - state: US state (as written in the answer; can be full name or 2-letter code)
    - program_type: Type of viewing program (Star Party, guided viewing, special eclipse event, etc.)
    - date: The specific date of the eclipse viewing program
    - start_time: The start time of the program (string)
    - is_public: true if open to the general public; false if members-only; null if unclear
    - pricing: Pricing details if provided (string; null if missing)
    - website_url: URL to the observatory’s official website (string URL)
    - program_url: URL to the specific program/event information page (string URL)

    Notes:
    - Extract only what appears in the answer. If a field is absent, set to null.
    - Keep URLs exactly as shown (full URLs).
    - Preserve at most three observatories in the listed order from the answer.
    """

def prompt_extract_conference() -> str:
    return """
    Extract information about one professional conference occurring within 2 weeks (before or after) of the eclipse date, and organized by IEEE or a major astronomy/aerospace organization, as presented in the answer.

    Fields:
    - name: Official conference name
    - organization: Organizer (e.g., IEEE, AAS, AIAA, etc.)
    - start_date: Conference start date
    - end_date: Conference end date
    - venue: Venue or convention center name
    - city: City where it is held
    - state: State (US) where it is held
    - website_url: URL to the official conference website
    - location_url: URL confirming the venue/location (may be the same as website_url)

    Rules:
    - Only extract values explicitly included in the answer.
    - For missing values, set null.
    - For URL fields, extract full URLs exactly as shown in the answer.
    """


# ------------------------------------------------------------------------------
# Verification helpers
# ------------------------------------------------------------------------------
async def verify_eclipse_event(evaluator: Evaluator, parent_node, eclipse: EclipseExtraction) -> None:
    # Create main eclipse node
    eclipse_node = evaluator.add_parallel(
        id="eclipse_event",
        desc="Identify the total lunar eclipse event occurring in Spring 2026 visible from the continental United States",
        parent=parent_node,
        critical=False
    )

    # Event identification group
    event_id_node = evaluator.add_parallel(
        id="event_identification",
        desc="Correctly identify the specific date and type of the total lunar eclipse",
        parent=eclipse_node,
        critical=False
    )

    # Source existence (critical gate)
    event_src_exists = evaluator.add_custom_node(
        result=bool(eclipse and eclipse.event_source_urls and len(eclipse.event_source_urls) > 0),
        id="source_url",
        desc="Provide URL reference from an authoritative source (NASA, timeanddate.com, or similar) documenting this eclipse",
        parent=event_id_node,
        critical=True
    )

    # Eclipse date (critical)
    date_leaf = evaluator.add_leaf(
        id="eclipse_date",
        desc="Provide the correct date of the total lunar eclipse (must be in March-May 2026)",
        parent=event_id_node,
        critical=True
    )
    date_claim = f"The date of the total lunar eclipse is '{eclipse.date}', and it occurs in March-May 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=eclipse.event_source_urls,
        additional_instruction="Verify explicitly that the page states this eclipse date and that it falls within March to May 2026."
    )

    # Eclipse type (critical)
    type_leaf = evaluator.add_leaf(
        id="eclipse_type",
        desc="Identify the event as a total lunar eclipse (blood moon)",
        parent=event_id_node,
        critical=True
    )
    type_claim = "This event is a total lunar eclipse (also commonly called a blood moon)."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=eclipse.event_source_urls,
        additional_instruction="Confirm that the authoritative source explicitly identifies the event as a total lunar eclipse."
    )

    # Visibility information group
    visibility_node = evaluator.add_parallel(
        id="visibility_information",
        desc="Provide visibility information for the continental United States",
        parent=eclipse_node,
        critical=False
    )

    # Source existence for visibility (critical)
    vis_src_exists = evaluator.add_custom_node(
        result=bool(eclipse and eclipse.visibility_source_urls and len(eclipse.visibility_source_urls) > 0),
        id="visibility_source_url",
        desc="Provide URL reference for US visibility information",
        parent=visibility_node,
        critical=True
    )

    # US visibility (critical)
    us_vis_leaf = evaluator.add_leaf(
        id="us_visibility",
        desc="Confirm that the eclipse is visible from at least part of the continental United States",
        parent=visibility_node,
        critical=True
    )
    us_vis_claim = "This eclipse is visible from at least part of the continental United States."
    await evaluator.verify(
        claim=us_vis_claim,
        node=us_vis_leaf,
        sources=eclipse.visibility_source_urls,
        additional_instruction="Look for a visibility map or text explicitly mentioning visibility from the contiguous U.S."
    )

    # Regional coverage (critical)
    regional_leaf = evaluator.add_leaf(
        id="regional_coverage",
        desc="Identify which regions of the US (western, central, eastern) can observe the eclipse",
        parent=visibility_node,
        critical=True
    )
    # Join listed regions (if any) into claim
    regions_listed = ", ".join(eclipse.us_regions) if eclipse and eclipse.us_regions else "some specified U.S. regions"
    regional_claim = f"The eclipse is observable in the following U.S. regions: {regions_listed}."
    await evaluator.verify(
        claim=regional_claim,
        node=regional_leaf,
        sources=eclipse.visibility_source_urls,
        additional_instruction="Allow reasonable synonyms (e.g., Midwest for central). The page should support the stated regional visibility."
    )

    # Totality duration group
    duration_node = evaluator.add_parallel(
        id="totality_duration",
        desc="Provide the duration of totality for the eclipse",
        parent=eclipse_node,
        critical=False
    )

    # Source existence for duration (critical)
    dur_src_exists = evaluator.add_custom_node(
        result=bool(eclipse and eclipse.duration_source_urls and len(eclipse.duration_source_urls) > 0),
        id="duration_source_url",
        desc="Provide URL reference documenting the totality duration",
        parent=duration_node,
        critical=True
    )

    # Duration value (critical)
    duration_leaf = evaluator.add_leaf(
        id="duration_value",
        desc="State the totality duration in minutes (must be between 30 minutes and 2 hours)",
        parent=duration_node,
        critical=True
    )
    duration_claim = f"The totality duration is {eclipse.duration_minutes} minutes."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=eclipse.duration_source_urls,
        additional_instruction="Verify the quoted totality duration in minutes on the page. If the page lists duration in h:mm, equivalently interpret in minutes."
    )

    # Additional custom numeric range check (critical)
    minutes_parsed = extract_first_int(eclipse.duration_minutes if eclipse else None)
    evaluator.add_custom_node(
        result=minutes_parsed is not None and 30 <= minutes_parsed <= 120,
        id="duration_range_check",
        desc="Totality duration falls between 30 and 120 minutes",
        parent=duration_node,
        critical=True
    )

    # Timing details group
    timing_node = evaluator.add_parallel(
        id="timing_details",
        desc="Provide precise timing information for the eclipse",
        parent=eclipse_node,
        critical=False
    )

    # UTC times subgroup
    utc_node = evaluator.add_parallel(
        id="utc_times",
        desc="Provide UTC times for key eclipse phases",
        parent=timing_node,
        critical=False
    )

    # Source existence for timing (critical)
    timing_src_exists = evaluator.add_custom_node(
        result=bool(eclipse and eclipse.timing_source_urls and len(eclipse.timing_source_urls) > 0),
        id="timing_source_url",
        desc="Provide URL reference for timing information",
        parent=utc_node,
        critical=True
    )

    # Individual UTC time leaves (critical)
    start_leaf = evaluator.add_leaf(
        id="totality_start",
        desc="Provide UTC time when totality begins",
        parent=utc_node,
        critical=True
    )
    start_claim = f"Totality begins at {eclipse.utc_times.totality_start_utc if eclipse and eclipse.utc_times else None} UTC."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=eclipse.timing_source_urls,
        additional_instruction="Verify the precise UTC start of totality from the authoritative page."
    )

    max_leaf = evaluator.add_leaf(
        id="maximum_eclipse",
        desc="Provide UTC time of maximum eclipse",
        parent=utc_node,
        critical=True
    )
    max_claim = f"Maximum eclipse occurs at {eclipse.utc_times.maximum_utc if eclipse and eclipse.utc_times else None} UTC."
    await evaluator.verify(
        claim=max_claim,
        node=max_leaf,
        sources=eclipse.timing_source_urls,
        additional_instruction="Verify the precise UTC time of maximum eclipse."
    )

    end_leaf = evaluator.add_leaf(
        id="totality_end",
        desc="Provide UTC time when totality ends",
        parent=utc_node,
        critical=True
    )
    end_claim = f"Totality ends at {eclipse.utc_times.totality_end_utc if eclipse and eclipse.utc_times else None} UTC."
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=eclipse.timing_source_urls,
        additional_instruction="Verify the precise UTC end of totality."
    )

    # Regional/local timing subgroup
    local_node = evaluator.add_parallel(
        id="regional_timing",
        desc="Provide local timing for at least two different US time zones",
        parent=timing_node,
        critical=False
    )

    # Eastern Time (critical)
    et_leaf = evaluator.add_leaf(
        id="eastern_time",
        desc="Provide timing for Eastern Time zone",
        parent=local_node,
        critical=True
    )
    et_start = eclipse.eastern_time.totality_start if eclipse and eclipse.eastern_time else None
    et_max = eclipse.eastern_time.maximum if eclipse and eclipse.eastern_time else None
    et_end = eclipse.eastern_time.totality_end if eclipse and eclipse.eastern_time else None
    et_claim = f"In Eastern Time, totality begins at {et_start}, reaches maximum at {et_max}, and ends at {et_end}."
    await evaluator.verify(
        claim=et_claim,
        node=et_leaf,
        sources=eclipse.timing_source_urls,
        additional_instruction="Confirm local Eastern Time values for totality start, maximum, and end as shown on the page. Allow ET/EDT notation."
    )

    # Pacific Time (critical)
    pt_leaf = evaluator.add_leaf(
        id="pacific_time",
        desc="Provide timing for Pacific Time zone",
        parent=local_node,
        critical=True
    )
    pt_start = eclipse.pacific_time.totality_start if eclipse and eclipse.pacific_time else None
    pt_max = eclipse.pacific_time.maximum if eclipse and eclipse.pacific_time else None
    pt_end = eclipse.pacific_time.totality_end if eclipse and eclipse.pacific_time else None
    pt_claim = f"In Pacific Time, totality begins at {pt_start}, reaches maximum at {pt_max}, and ends at {pt_end}."
    await evaluator.verify(
        claim=pt_claim,
        node=pt_leaf,
        sources=eclipse.timing_source_urls,
        additional_instruction="Confirm local Pacific Time values for totality start, maximum, and end as shown on the page. Allow PT/PDT notation."
    )


async def verify_single_observatory(
    evaluator: Evaluator,
    parent_node,
    obs: ObservatoryProgram,
    idx: int,
    prior_states: List[str],
    eclipse_date_for_context: Optional[str] = None
) -> None:
    oid = f"observatory_{idx+1}"

    obs_node = evaluator.add_parallel(
        id=oid,
        desc=f"{['First','Second','Third'][idx] if idx < 3 else f'Observatory #{idx+1}'} observatory with eclipse viewing program",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id=f"{oid}_identification",
        desc="Identify the observatory name and location",
        parent=obs_node,
        critical=False
    )

    # Identification URL existence (critical)
    ident_url_exists = evaluator.add_custom_node(
        result=bool(obs and obs.website_url and obs.website_url.strip()),
        id=f"{oid}_identification_url",
        desc="Provide URL to the observatory's official website",
        parent=ident_node,
        critical=True
    )

    # Name (critical)
    name_leaf = evaluator.add_leaf(
        id=f"{oid}_name",
        desc="Provide the official name of the observatory",
        parent=ident_node,
        critical=True
    )
    name_claim = f"The official name of the observatory is '{obs.name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=obs.website_url,
        additional_instruction="Confirm the organization/observatory name on the official site or the program page."
    )

    # State (critical; for obs 2 and 3 ensure difference)
    state_leaf = evaluator.add_leaf(
        id=f"{oid}_state",
        desc=("Provide the US state where the observatory is located"
              + (" (must be different from Observatory 1)" if idx == 1 else "")
              + (" (must be different from Observatories 1 and 2)" if idx == 2 else "")),
        parent=ident_node,
        critical=True
    )
    normalized_state = normalize_state_name(obs.state)
    prior_state_text = ", ".join(prior_states) if prior_states else ""
    if idx == 1:
        state_claim = f"The observatory is located in {normalized_state}, which is different from {prior_state_text}."
    elif idx == 2:
        state_claim = f"The observatory is located in {normalized_state}, which is different from {prior_state_text}."
    else:
        state_claim = f"The observatory is located in {normalized_state}."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=[u for u in [obs.program_url, obs.website_url] if u],
        additional_instruction="Verify the state indicated on the official site or event page. If prior states are provided, ensure this one is different."
    )

    # Programs group
    prog_node = evaluator.add_parallel(
        id=f"{oid}_programs",
        desc="Document the eclipse viewing programs offered",
        parent=obs_node,
        critical=False
    )

    # Program type (critical)
    ptype_leaf = evaluator.add_leaf(
        id=f"{oid}_program_type",
        desc="Identify the type of program offered (e.g., Star Party, guided viewing, special event)",
        parent=prog_node,
        critical=True
    )
    ptype_claim = f"The observatory offers a '{obs.program_type}' program specifically for the 2026 total lunar eclipse."
    await evaluator.verify(
        claim=ptype_claim,
        node=ptype_leaf,
        sources=obs.program_url or obs.website_url,
        additional_instruction="Confirm the program type terminology used on the official program page."
    )

    # Schedule subgroup
    sched_node = evaluator.add_parallel(
        id=f"{oid}_schedule",
        desc="Provide specific dates and times for the eclipse viewing program",
        parent=prog_node,
        critical=False
    )

    # Schedule URL existence (critical)
    sched_url_exists = evaluator.add_custom_node(
        result=bool(obs and obs.program_url and obs.program_url.strip()),
        id=f"{oid}_schedule_url",
        desc="Provide URL reference for the program schedule",
        parent=sched_node,
        critical=True
    )

    # Date (critical)
    date_leaf = evaluator.add_leaf(
        id=f"{oid}_date",
        desc="Confirm the program is scheduled on or around the eclipse date",
        parent=sched_node,
        critical=True
    )
    date_claim = f"The eclipse viewing program is scheduled on {obs.date}."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=obs.program_url,
        additional_instruction="Verify the program date on the official program page. It should correspond to the eclipse date or a closely related local date."
    )

    # Time (critical)
    time_leaf = evaluator.add_leaf(
        id=f"{oid}_time",
        desc="Provide the start time of the viewing program",
        parent=sched_node,
        critical=True
    )
    time_claim = f"The start time of the viewing program is {obs.start_time}."
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=obs.program_url,
        additional_instruction="Verify the announced start time on the program page."
    )

    # Public access subgroup
    access_node = evaluator.add_parallel(
        id=f"{oid}_public_access",
        desc="Confirm the program is open to the general public",
        parent=prog_node,
        critical=False
    )

    access_leaf = evaluator.add_leaf(
        id=f"{oid}_access_type",
        desc="Verify public can attend (not members-only)",
        parent=access_node,
        critical=True
    )
    access_statement = "open to the general public" if obs.is_public is True else "open to the public (not members-only)"
    access_claim = f"The eclipse viewing program is {access_statement}."
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=obs.program_url or obs.website_url,
        additional_instruction="Confirm the page indicates public access (e.g., 'open to the public', 'all ages welcome'). If it's members-only, the claim should be considered unsupported."
    )

    pricing_leaf = evaluator.add_leaf(
        id=f"{oid}_pricing",
        desc="Provide pricing information for the program",
        parent=access_node,
        critical=False
    )
    pricing_claim = f"The program pricing information includes: {obs.pricing}."
    await evaluator.verify(
        claim=pricing_claim,
        node=pricing_leaf,
        sources=obs.program_url or obs.website_url,
        additional_instruction="Verify that the specified price, fee, donation, or 'free' status appears on the official program page. If no pricing is provided in the answer, this will likely fail (non-critical)."
    )


async def verify_observatories(
    evaluator: Evaluator,
    parent_node,
    observatories: List[ObservatoryProgram],
    eclipse_date_for_context: Optional[str] = None
) -> None:
    obs_parent = evaluator.add_parallel(
        id="observatory_programs",
        desc="Identify three US public observatories offering special viewing programs during the eclipse",
        parent=parent_node,
        critical=False
    )

    # Ensure exactly 3 entries (pad with empty if fewer)
    obs_items = (observatories or [])[:3]
    while len(obs_items) < 3:
        obs_items.append(ObservatoryProgram())

    seen_states: List[str] = []

    for i, obs in enumerate(obs_items):
        await verify_single_observatory(evaluator, obs_parent, obs, i, prior_states=seen_states, eclipse_date_for_context=eclipse_date_for_context)
        # Update seen states
        st_norm = normalize_state_name(obs.state)
        if st_norm:
            seen_states.append(st_norm)

    # Geographic distribution: at least one in the western US (critical per rubric)
    has_west = any((s in WEST_STATES) for s in seen_states)
    evaluator.add_custom_node(
        result=has_west,
        id="geographic_distribution",
        desc="Verify that at least one of the three observatories is located in the western United States",
        parent=obs_parent,
        critical=True
    )


async def verify_conference(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceInfo,
    eclipse_date_for_context: Optional[str]
) -> None:
    conf_node = evaluator.add_parallel(
        id="professional_conference",
        desc="Identify a major IEEE or astronomy/aerospace professional conference occurring within 2 weeks of the eclipse",
        parent=parent_node,
        critical=False
    )

    # Identification group
    ident_node = evaluator.add_parallel(
        id="conference_identification",
        desc="Identify the conference name and organizing body",
        parent=conf_node,
        critical=False
    )

    # Identification URL existence (critical)
    ident_url_exists = evaluator.add_custom_node(
        result=bool(conf and conf.website_url and conf.website_url.strip()),
        id="conference_identification_url",
        desc="Provide URL to the official conference website",
        parent=ident_node,
        critical=True
    )

    # Name (critical)
    cname_leaf = evaluator.add_leaf(
        id="conference_name",
        desc="Provide the official name of the conference",
        parent=ident_node,
        critical=True
    )
    cname_claim = f"The official name of the conference is '{conf.name}'."
    await evaluator.verify(
        claim=cname_claim,
        node=cname_leaf,
        sources=conf.website_url,
        additional_instruction="Verify the exact official conference title as shown on the site."
    )

    # Organization (critical)
    corg_leaf = evaluator.add_leaf(
        id="conference_organization",
        desc="Confirm the conference is organized by IEEE or a major astronomy/aerospace organization",
        parent=ident_node,
        critical=True
    )
    org_claim = f"The conference is organized by {conf.organization}, which is IEEE or a major astronomy/aerospace organization."
    await evaluator.verify(
        claim=org_claim,
        node=corg_leaf,
        sources=conf.website_url,
        additional_instruction="Look for organizer logos or text (e.g., IEEE, AAS, AIAA, AGU). Confirm the named body is indeed the organizer."
    )

    # Timing group
    timing_node = evaluator.add_parallel(
        id="conference_timing",
        desc="Verify the conference dates are within 2 weeks before or after the eclipse",
        parent=conf_node,
        critical=False
    )

    # Start/end date (critical)
    cstart_leaf = evaluator.add_leaf(
        id="conference_start_date",
        desc="Provide the conference start date",
        parent=timing_node,
        critical=True
    )
    cstart_claim = f"The conference starts on {conf.start_date}."
    await evaluator.verify(
        claim=cstart_claim,
        node=cstart_leaf,
        sources=conf.website_url,
        additional_instruction="Verify the stated start date matches the conference site."
    )

    cend_leaf = evaluator.add_leaf(
        id="conference_end_date",
        desc="Provide the conference end date",
        parent=timing_node,
        critical=True
    )
    cend_claim = f"The conference ends on {conf.end_date}."
    await evaluator.verify(
        claim=cend_claim,
        node=cend_leaf,
        sources=conf.website_url,
        additional_instruction="Verify the stated end date matches the conference site."
    )

    # Proximity check (critical, logic-based)
    prox_leaf = evaluator.add_leaf(
        id="timing_proximity",
        desc="Verify the conference occurs within 14 days before or after the eclipse date",
        parent=timing_node,
        critical=True
    )
    prox_claim = f"The conference (from {conf.start_date} to {conf.end_date}) occurs within 14 days before or after the eclipse date {eclipse_date_for_context}."
    # Logical check; allow LLM to compute difference; we still pass the site URL for context of dates
    await evaluator.verify(
        claim=prox_claim,
        node=prox_leaf,
        sources=conf.website_url,
        additional_instruction="Use the provided dates to determine if the conference window is within ±14 days of the eclipse date. Minor date formatting variations are acceptable."
    )

    # Location group
    loc_node = evaluator.add_parallel(
        id="conference_location",
        desc="Provide the conference venue location",
        parent=conf_node,
        critical=False
    )

    # Location URL existence (critical)
    loc_url_exists = evaluator.add_custom_node(
        result=bool(conf and (conf.location_url or conf.website_url)),
        id="conference_location_url",
        desc="Provide URL reference confirming the conference location",
        parent=loc_node,
        critical=True
    )

    # Venue (critical)
    venue_leaf = evaluator.add_leaf(
        id="conference_venue",
        desc="Provide the name of the conference venue or center",
        parent=loc_node,
        critical=True
    )
    venue_claim = f"The conference venue is '{conf.venue}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=[conf.location_url or conf.website_url],
        additional_instruction="Verify the official venue name as stated on the site."
    )

    # City and State (critical)
    city_state_leaf = evaluator.add_leaf(
        id="conference_city_state",
        desc="Provide the city and state where the conference is held",
        parent=loc_node,
        critical=True
    )
    city_state_claim = f"The conference is held in {conf.city}, {conf.state}."
    await evaluator.verify(
        claim=city_state_claim,
        node=city_state_leaf,
        sources=[conf.location_url or conf.website_url],
        additional_instruction="Verify the city and state as shown on the site."
    )


# ------------------------------------------------------------------------------
# Main evaluation entry
# ------------------------------------------------------------------------------
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

    # Extract data in parallel
    eclipse_task = evaluator.extract(
        prompt=prompt_extract_eclipse(),
        template_class=EclipseExtraction,
        extraction_name="eclipse_info"
    )
    observatories_task = evaluator.extract(
        prompt=prompt_extract_observatories(),
        template_class=ObservatoriesExtraction,
        extraction_name="observatories_info"
    )
    conference_task = evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceInfo,
        extraction_name="conference_info"
    )

    eclipse_info, observatories_info, conference_info = await asyncio.gather(
        eclipse_task, observatories_task, conference_task
    )

    # Build and verify subtrees
    await verify_eclipse_event(evaluator, root, eclipse_info)
    await verify_observatories(evaluator, root, observatories_info.observatories if observatories_info else [], eclipse_date_for_context=eclipse_info.date if eclipse_info else None)
    await verify_conference(evaluator, root, conference_info, eclipse_date_for_context=eclipse_info.date if eclipse_info else None)

    # Custom info
    evaluator.add_custom_info(
        info={
            "west_states_considered": sorted(list(WEST_STATES)),
            "note": "A state is considered western if it belongs to the US Census West (including AK, HI)."
        },
        info_type="geography_assumptions",
        info_name="geography_assumptions"
    )

    return evaluator.get_summary()