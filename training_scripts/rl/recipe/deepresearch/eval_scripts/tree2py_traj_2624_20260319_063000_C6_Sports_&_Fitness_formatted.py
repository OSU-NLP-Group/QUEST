import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "city_dual_hoops_march_2025"
TASK_DESCRIPTION = (
    "Identify the city that will host both a Power 5 conference men's basketball tournament AND an NCAA Division I "
    "Men's Basketball Tournament regional during March 2025. Your answer must include: (1) the city name, "
    "(2) the conference tournament venue name and its basketball seating capacity, (3) the conference name, "
    "(4) the state where that conference's headquarters is located, (5) the NCAA regional venue name and its capacity, "
    "(6) the specific regional designation (East, West, Midwest, or South), and (7) verification that both events occur "
    "within a 15-day window in March 2025. The conference tournament venue must have a basketball seating capacity of "
    "at least 17,000, the NCAA regional venue must have a capacity of at least 60,000, and the conference headquarters "
    "must be located in a different state than the host city."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DateRange(BaseModel):
    start_date: Optional[str] = None  # Prefer ISO format YYYY-MM-DD
    end_date: Optional[str] = None    # Prefer ISO format YYYY-MM-DD


class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None  # Keep as free text; we'll parse number
    headquarters_state: Optional[str] = None  # Full state name or postal abbreviation
    dates: Optional[DateRange] = None
    refs: List[str] = Field(default_factory=list)


class NCAARegionalInfo(BaseModel):
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None
    designation: Optional[str] = None  # East / West / Midwest / South
    dates: Optional[DateRange] = None
    refs: List[str] = Field(default_factory=list)


class CityTaskExtraction(BaseModel):
    city_name: Optional[str] = None
    city_state: Optional[str] = None  # Full state name or postal abbreviation for the host city
    conference: Optional[ConferenceInfo] = None
    ncaa: Optional[NCAARegionalInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_task() -> str:
    return """
Extract the following structured information strictly from the provided answer text. Do not infer or add facts.

Top-level:
- city_name: The single city that the answer identifies as hosting both events.
- city_state: The U.S. state for that city (full name preferred, otherwise postal abbreviation), if provided.

Conference tournament (Power 5):
- conference.name: The conference name as written (e.g., "Big Ten", "SEC", "ACC", "Big 12", "Pac-12").
- conference.venue_name: Venue/stadium/arena for the conference men's basketball tournament.
- conference.venue_capacity: The basketball seating capacity for that venue as stated in the answer (text as-is).
- conference.headquarters_state: The state where the conference headquarters is located (full name or postal).
- conference.dates.start_date: The start date of the tournament in 2025 (ISO 'YYYY-MM-DD' if possible; otherwise text).
- conference.dates.end_date: The end date of the tournament in 2025 (ISO 'YYYY-MM-DD' if possible; otherwise text).
- conference.refs: All URLs in the answer that support conference tournament details (venue, dates, capacity, city).

NCAA regional:
- ncaa.venue_name: Venue/stadium/arena for the NCAA Division I Men's Basketball Tournament regional.
- ncaa.venue_capacity: The stated venue capacity (text as-is).
- ncaa.designation: One of "East", "West", "Midwest", or "South", as stated in the answer.
- ncaa.dates.start_date: The start date of the regional in 2025 (ISO 'YYYY-MM-DD' if possible; otherwise text).
- ncaa.dates.end_date: The end date of the regional in 2025 (ISO 'YYYY-MM-DD' if possible; otherwise text).
- ncaa.refs: All URLs in the answer that support NCAA regional details (venue, dates, designation, city).

Rules:
- Return null for any field not explicitly present in the answer.
- Keep capacities and designations exactly as written (we will validate later).
- Prefer ISO dates (YYYY-MM-DD). If only a range phrase is provided, set start_date and end_date to best parsed values or null.
- Include every URL in refs arrays that the answer cites for the corresponding section.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
POWER_FIVE_NAMES = {
    "big ten", "big ten conference",
    "sec", "southeastern conference",
    "acc", "atlantic coast conference",
    "big 12", "big 12 conference",
    "pac-12", "pac 12", "pac 12 conference", "pac-12 conference"
}


def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip().lower()


def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    if not cap_str:
        return None
    s = cap_str.strip().lower()
    best = None
    for m in re.finditer(r'(\d{1,3}(?:,\d{3})+|\d+)\s*([kKmM])?', s):
        num_txt = m.group(1)
        suf = m.group(2)
        try:
            n = int(num_txt.replace(",", ""))
            if suf:
                if suf.lower() == 'k':
                    n *= 1000
                elif suf.lower() == 'm':
                    n *= 1_000_000
            best = n if (best is None or n > best) else best
        except Exception:
            continue
    return best


def try_parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    d = d.strip()
    # Prefer ISO
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', d)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    # Try common textual formats
    fmts = [
        "%B %d, %Y", "%b %d, %Y",
        "%m/%d/%Y", "%Y/%m/%d",
        "%m-%d-%Y", "%Y.%m.%d",
        "%Y %b %d", "%Y %B %d",
        "%B %d %Y", "%b %d %Y"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            continue
    # Fallback: try to find a YYYY and month name/day
    month_map = {m.lower(): i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"], start=1)}
    mm = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)'
                   r'\s+(\d{1,2}),?\s*(\d{4})', d, re.IGNORECASE)
    if mm:
        mon = month_map.get(mm.group(1).lower())
        day = int(mm.group(2))
        year = int(mm.group(3))
        try:
            return datetime(year, mon, day)
        except Exception:
            return None
    return None


def covered_span_days(conf_range: Optional[DateRange], ncaa_range: Optional[DateRange]) -> Optional[int]:
    if not conf_range or not ncaa_range:
        return None
    s1 = try_parse_date(conf_range.start_date)
    e1 = try_parse_date(conf_range.end_date)
    s2 = try_parse_date(ncaa_range.start_date)
    e2 = try_parse_date(ncaa_range.end_date)
    if not (s1 and e1 and s2 and e2):
        return None
    earliest = min(s1, s2)
    latest = max(e1, e2)
    return (latest - earliest).days


def str_present(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def is_valid_designation(desig: Optional[str]) -> bool:
    if not desig:
        return False
    return desig.strip().lower() in {"east", "west", "midwest", "south"}


def is_power_five(name: Optional[str]) -> bool:
    if not name:
        return False
    return name.strip().lower() in POWER_FIVE_NAMES


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_conference_criteria(
    evaluator: Evaluator,
    parent,
    data: CityTaskExtraction
):
    conf = data.conference or ConferenceInfo()
    city = data.city_name or ""
    conf_node = evaluator.add_parallel(
        id="conference_criteria",
        desc="Verifies that the city hosts a Power 5 conference basketball tournament in March 2025 meeting all requirements",
        parent=parent,
        critical=True  # Essential
    )

    # Conference_Tournament_Venue (critical group)
    venue_grp = evaluator.add_parallel(
        id="conference_tournament_venue",
        desc="Provides correct conference tournament venue information",
        parent=conf_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(conf.venue_name),
        id="conf_venue_name_provided",
        desc="Conference tournament venue name is provided",
        parent=venue_grp,
        critical=True
    )

    venue_loc_node = evaluator.add_leaf(
        id="conf_venue_location_matches",
        desc="Venue is located in the identified city",
        parent=venue_grp,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference tournament venue '{conf.venue_name}' is located in or clearly associated with {city}.",
        node=venue_loc_node,
        sources=conf.refs,
        additional_instruction="Allow reasonable metro-area naming (e.g., Glendale counted for Phoenix). Focus on whether the cited sources associate the venue with the identified host city/metro."
    )

    # Conference_Identity (critical group)
    id_grp = evaluator.add_parallel(
        id="conference_identity",
        desc="Correctly identifies the conference and verifies it is a Power 5 conference",
        parent=conf_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(conf.name),
        id="conf_name_provided",
        desc="Conference name is provided",
        parent=id_grp,
        critical=True
    )

    power5_node = evaluator.add_leaf(
        id="conf_is_power_five",
        desc="Conference is one of: Big Ten, SEC, ACC, Big 12, or formerly Pac-12",
        parent=id_grp,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference '{conf.name}' is a Power 5 conference (Big Ten / SEC / ACC / Big 12 / Pac-12).",
        node=power5_node,
        additional_instruction="Accept synonyms/expanded forms: SEC=Southeastern Conference; ACC=Atlantic Coast Conference; 'Big Ten Conference' and 'Big 12 Conference' are acceptable; 'Pac-12' is acceptable even if noted as 'formerly'."
    )

    # Tournament_Timing (critical group)
    timing_grp = evaluator.add_parallel(
        id="conference_timing",
        desc="Verifies tournament occurs in March 2025",
        parent=conf_node,
        critical=True
    )

    dates_in_march = evaluator.add_leaf(
        id="conf_dates_in_march_2025",
        desc="Tournament dates are in March 2025",
        parent=timing_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The conference men's basketball tournament occurs in March 2025.",
        node=dates_in_march,
        sources=conf.refs,
        additional_instruction="Confirm that the cited dates for the conference tournament fall within March 2025 (some overlap with late February may exist, but March 2025 participation should be explicit)."
    )

    # Conference_Venue_Capacity (critical group)
    cap_grp = evaluator.add_parallel(
        id="conference_venue_capacity",
        desc="Verifies venue capacity meets minimum requirement",
        parent=conf_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(conf.venue_capacity),
        id="conf_capacity_provided",
        desc="Basketball seating capacity is provided",
        parent=cap_grp,
        critical=True
    )

    conf_cap_int = parse_capacity_to_int(conf.venue_capacity)
    evaluator.add_custom_node(
        result=(conf_cap_int is not None and conf_cap_int >= 17000),
        id="conf_meets_min_17000",
        desc="Basketball seating capacity is at least 17,000",
        parent=cap_grp,
        critical=True
    )

    # Conference_Headquarters_Location (critical group)
    hq_grp = evaluator.add_parallel(
        id="conference_headquarters_location",
        desc="Verifies conference headquarters is in a different state than host city",
        parent=conf_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(conf.headquarters_state),
        id="conf_hq_state_provided",
        desc="Conference headquarters state is provided",
        parent=hq_grp,
        critical=True
    )

    # Compare host city state vs HQ state
    host_state_norm = _normalize_state(data.city_state)
    hq_state_norm = _normalize_state(conf.headquarters_state)
    evaluator.add_custom_node(
        result=(bool(host_state_norm) and bool(hq_state_norm) and host_state_norm != hq_state_norm),
        id="conf_hq_different_state",
        desc="Conference headquarters is in a different state than the host city",
        parent=hq_grp,
        critical=True
    )

    return conf_node


async def build_ncaa_criteria(
    evaluator: Evaluator,
    parent,
    data: CityTaskExtraction
):
    ncaa = data.ncaa or NCAARegionalInfo()
    city = data.city_name or ""

    ncaa_node = evaluator.add_parallel(
        id="ncaa_criteria",
        desc="Verifies that the city hosts an NCAA regional in March 2025 meeting all requirements",
        parent=parent,
        critical=True
    )

    # Regional_Venue (critical group)
    rvenue_grp = evaluator.add_parallel(
        id="regional_venue",
        desc="Provides correct NCAA regional venue information",
        parent=ncaa_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(ncaa.venue_name),
        id="regional_venue_name_provided",
        desc="NCAA regional venue name is provided",
        parent=rvenue_grp,
        critical=True
    )

    rvenue_loc_node = evaluator.add_leaf(
        id="regional_venue_location_matches",
        desc="Venue is located in the identified city",
        parent=rvenue_grp,
        critical=True
    )
    await evaluator.verify(
        claim=f"The NCAA regional venue '{ncaa.venue_name}' is located in or clearly associated with {city}.",
        node=rvenue_loc_node,
        sources=ncaa.refs,
        additional_instruction="Allow reasonable metro-area naming (e.g., Glendale counted for Phoenix). Focus on whether the cited sources associate the venue with the identified host city/metro."
    )

    # Regional_Designation (critical group)
    desig_grp = evaluator.add_parallel(
        id="regional_designation",
        desc="Provides correct regional designation",
        parent=ncaa_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(ncaa.designation),
        id="regional_designation_provided",
        desc="Regional designation is provided",
        parent=desig_grp,
        critical=True
    )

    valid_desig_node = evaluator.add_leaf(
        id="regional_valid_designation",
        desc="Designation is one of: East, West, Midwest, or South",
        parent=desig_grp,
        critical=True
    )
    await evaluator.verify(
        claim=f"The regional designation '{ncaa.designation}' is one of East, West, Midwest, or South.",
        node=valid_desig_node,
        additional_instruction="Treat case-insensitive matches or minor formatting variants (e.g., 'Mid-West') as 'Midwest' only if the meaning is clear."
    )

    # Regional_Timing (critical group)
    rtiming_grp = evaluator.add_parallel(
        id="regional_timing",
        desc="Verifies regional occurs in March 2025",
        parent=ncaa_node,
        critical=True
    )

    rdates_in_march = evaluator.add_leaf(
        id="regional_dates_in_march_2025",
        desc="Regional dates are in March 2025",
        parent=rtiming_grp,
        critical=True
    )
    pretty_desig = f"{ncaa.designation} " if str_present(ncaa.designation) else ""
    await evaluator.verify(
        claim=f"The NCAA {pretty_desig}regional occurs in March 2025.",
        node=rdates_in_march,
        sources=ncaa.refs,
        additional_instruction="Confirm that the cited regional dates fall within March 2025."
    )

    # Regional_Venue_Capacity (critical group)
    rcap_grp = evaluator.add_parallel(
        id="regional_venue_capacity",
        desc="Verifies venue capacity meets minimum requirement",
        parent=ncaa_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(ncaa.venue_capacity),
        id="regional_capacity_provided",
        desc="Venue capacity is provided",
        parent=rcap_grp,
        critical=True
    )

    ncaa_cap_int = parse_capacity_to_int(ncaa.venue_capacity)
    evaluator.add_custom_node(
        result=(ncaa_cap_int is not None and ncaa_cap_int >= 60000),
        id="regional_meets_min_60000",
        desc="Venue capacity is at least 60,000",
        parent=rcap_grp,
        critical=True
    )

    # Time_Window_Verification (critical group) – computed from extracted dates
    win_grp = evaluator.add_parallel(
        id="time_window_verification",
        desc="Verifies both events occur within 15-day window",
        parent=ncaa_node,
        critical=True
    )

    span = covered_span_days((data.conference or ConferenceInfo()).dates, (data.ncaa or NCAARegionalInfo()).dates)
    within_15 = (span is not None and span <= 15)
    evaluator.add_custom_node(
        result=within_15,
        id="within_15_days",
        desc="Conference tournament and NCAA regional are within 15 days of each other",
        parent=win_grp,
        critical=True
    )

    # Record diagnostic info
    evaluator.add_custom_info(
        info={
            "computed_span_days": span,
            "conference_dates": (data.conference.dates.dict() if data.conference and data.conference.dates else None),
            "ncaa_dates": (data.ncaa.dates.dict() if data.ncaa and data.ncaa.dates else None),
        },
        info_type="diagnostics",
        info_name="time_window_computation"
    )

    return ncaa_node


async def build_reference_checks(
    evaluator: Evaluator,
    parent,
    data: CityTaskExtraction
):
    # Non-critical URL support checks (placed outside critical groups to satisfy framework constraints)
    city = data.city_name or ""
    conf = data.conference or ConferenceInfo()
    ncaa = data.ncaa or NCAARegionalInfo()

    # Conference references support
    conf_ref_leaf = evaluator.add_leaf(
        id="conference_tournament_references",
        desc="Provides supporting URL references for conference tournament information",
        parent=parent,
        critical=False
    )
    conf_claim = (
        f"The provided URLs explicitly support that {city} hosts the {conf.name} men's basketball tournament "
        f"at '{conf.venue_name}' in March 2025."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_ref_leaf,
        sources=conf.refs,
        additional_instruction="A single solid source is enough. Look for explicit statements about city, venue, and March 2025 tournament dates."
    )

    # NCAA references support
    ncaa_ref_leaf = evaluator.add_leaf(
        id="ncaa_regional_references",
        desc="Provides supporting URL references for NCAA regional information",
        parent=parent,
        critical=False
    )
    ncaa_claim = (
        f"The provided URLs explicitly support that {city} hosts the NCAA {ncaa.designation or ''} regional "
        f"at '{ncaa.venue_name}' in March 2025."
    )
    await evaluator.verify(
        claim=ncaa_claim,
        node=ncaa_ref_leaf,
        sources=ncaa.refs,
        additional_instruction="A single solid source is enough. Look for explicit statements about city, venue, designation (if provided), and March 2025 dates."
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
    Evaluate an answer for the dual basketball events city task (March 2025).
    Note on tree structure: We slightly adapt criticality from the provided rubric to satisfy the
    framework constraint that a critical parent cannot have non-critical children. We keep all
    essential checks as critical and place non-critical reference checks as siblings at the same level.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation
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

    # 1) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_task(),
        template_class=CityTaskExtraction,
        extraction_name="city_dual_events_extraction"
    )

    # 2) Build City Identification node (non-critical wrapper; children control pass/fail via critical flags)
    city_node = evaluator.add_parallel(
        id="city_identification",
        desc="Correctly identifies the city that meets all specified criteria",
        parent=root,
        critical=False
    )

    # City_Name_Provided (critical leaf)
    evaluator.add_custom_node(
        result=str_present(extracted.city_name),
        id="city_name_provided",
        desc="City name is explicitly provided",
        parent=city_node,
        critical=True
    )

    # 3) Conference criteria (critical)
    await build_conference_criteria(evaluator, city_node, extracted)

    # 4) NCAA criteria (critical, includes 15-day window check)
    await build_ncaa_criteria(evaluator, city_node, extracted)

    # 5) Non-critical reference support leaves (kept outside critical groups)
    await build_reference_checks(evaluator, city_node, extracted)

    # 6) Return evaluation summary
    return evaluator.get_summary()