import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ams106_houston_2026"
TASK_DESCRIPTION = (
    "Identify the major atmospheric science conference held in the United States in late January 2026 that satisfies ALL of the following criteria:\n\n"
    "1. The conference was the 106th edition of its series and was organized by a major American meteorological society\n"
    "2. The conference took place in Houston, Texas, at a convention center\n"
    "3. The conference lasted 5 consecutive days\n"
    "4. The conference included at least two co-located specialized conferences: one being the 27th Conference on Satellite Meteorology, Oceanography, and Climatology, and another being the 23rd Conference on Space Weather\n"
    "5. Within 35 days after the conference ended, a total lunar eclipse occurred with a totality duration of 58-59 minutes and an umbral magnitude exceeding 1.15, visible from western North America\n"
    "6. Within approximately one week after the conference ended, a major sudden stratospheric warming (SSW) event was confirmed, causing polar vortex disruption with stratospheric temperature increases of up to 70°F within days\n\n"
    "Provide the following information:\n"
    "- The full official conference name\n"
    "- The exact conference dates (start date and end date)\n"
    "- The specific venue name in Houston\n"
    "- The organizing society's full name\n"
    "- The date of the total lunar eclipse that occurred after the conference\n"
    "- Confirmation of the SSW event timing in early February 2026\n\n"
    "All information must be supported by reference URLs from reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceExtraction(BaseModel):
    # Identity and organizer
    full_official_name: Optional[str] = None
    series_edition_text: Optional[str] = None  # e.g., "106th Annual Meeting"
    organizing_society_full_name: Optional[str] = None

    # Dates and location
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    conference_city: Optional[str] = None
    conference_state: Optional[str] = None
    venue_name: Optional[str] = None

    # Co-located events
    satellite_conf_name: Optional[str] = None  # Expect something like "27th Conference on Satellite Meteorology, Oceanography, and Climatology"
    space_weather_conf_name: Optional[str] = None  # Expect "23rd Conference on Space Weather"

    # Astronomical phenomenon (lunar eclipse)
    eclipse_date: Optional[str] = None
    eclipse_totality_duration: Optional[str] = None  # e.g., "59 minutes"
    eclipse_umbral_magnitude: Optional[str] = None  # e.g., "1.19"
    eclipse_visibility_text: Optional[str] = None  # e.g., "visible from western North America"

    # Atmospheric phenomenon (SSW)
    ssw_confirmation_date: Optional[str] = None  # date the SSW was confirmed/declared
    ssw_effects_summary: Optional[str] = None  # text describing polar vortex disruption and temperature change

    # URL evidence buckets (as explicitly cited in the answer)
    conference_urls: List[str] = Field(default_factory=list)         # main conference info (dates/identity/location/program)
    organizer_urls: List[str] = Field(default_factory=list)          # organizer background/identity pages
    venue_urls: List[str] = Field(default_factory=list)              # venue official/info pages
    satellite_conf_urls: List[str] = Field(default_factory=list)     # pages proving the 27th Satellite Meteorology conf was co-located
    space_weather_conf_urls: List[str] = Field(default_factory=list) # pages proving the 23rd Space Weather conf was co-located
    eclipse_urls: List[str] = Field(default_factory=list)            # NASA / reliable eclipse info sources
    ssw_urls: List[str] = Field(default_factory=list)                # NOAA/ECMWF/etc. sources confirming SSW timing/effects
    extra_urls: List[str] = Field(default_factory=list)              # any other URLs provided in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    Extract the requested conference facts and their cited source URLs from the answer text.

    Required fields (use strings; do not fabricate data; if missing, return null or empty list):
    - full_official_name: The complete official conference name as written
    - series_edition_text: The edition/series wording as written (e.g., "106th Annual Meeting" or "106th")
    - organizing_society_full_name: The full name of the organizing society
    - start_date: The exact conference start date as written (e.g., "January 25, 2026" or "2026-01-25")
    - end_date: The exact conference end date as written
    - conference_city: The city (expect "Houston")
    - conference_state: The U.S. state (expect "Texas" or "TX")
    - venue_name: The specific named venue in Houston (e.g., "George R. Brown Convention Center")

    Co-located events:
    - satellite_conf_name: Name string for the "27th Conference on Satellite Meteorology, Oceanography, and Climatology" (if mentioned exactly or near-equivalent)
    - space_weather_conf_name: Name string for the "23rd Conference on Space Weather" (if mentioned exactly or near-equivalent)

    Astronomical phenomenon (lunar eclipse after the conference):
    - eclipse_date: The date of the total lunar eclipse after the conference
    - eclipse_totality_duration: The stated totality duration text (e.g., "58 minutes", "59 minutes")
    - eclipse_umbral_magnitude: The stated umbral magnitude value text (e.g., "1.18", "1.19")
    - eclipse_visibility_text: Short textual mention that it was visible from western North America (if present)

    Atmospheric phenomenon (SSW after the conference):
    - ssw_confirmation_date: The confirmation/announcement date in early February 2026 for the SSW event
    - ssw_effects_summary: Short textual description about polar vortex disruption and stratospheric temperature increases (up to ~70°F) within days

    URL evidence (only extract real URLs explicitly present in the answer; put each into the most appropriate bucket):
    - conference_urls: URLs that substantiate the main conference identity/dates/location/venue and overall program
    - organizer_urls: URLs that substantiate the identity/background of the organizing society and that it organizes the conference
    - venue_urls: URLs that substantiate the venue identity/type (e.g., official venue or city pages)
    - satellite_conf_urls: URLs that substantiate the presence of the 27th Conference on Satellite Meteorology, Oceanography, and Climatology as co-located
    - space_weather_conf_urls: URLs that substantiate the presence of the 23rd Conference on Space Weather as co-located
    - eclipse_urls: URLs that substantiate the eclipse date, totality duration, umbral magnitude, and visibility (e.g., NASA or timeanddate)
    - ssw_urls: URLs that substantiate the SSW timing in early Feb 2026, polar vortex disruption, and up to ~70°F stratospheric temperature increase
    - extra_urls: any other URLs cited in the answer that don't clearly belong to the above categories

    Important:
    - Do not invent URLs. Only extract URLs that are explicitly present in the answer text (including markdown links).
    - Keep all text fields exactly as written in the answer without normalization.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if lst:
            combined.extend(lst)
    return _dedup_urls(combined)


def _strip_ordinals(s: str) -> str:
    for suf in ["st", "nd", "rd", "th"]:
        s = s.replace(f" {suf},", ",").replace(f"{suf},", ",").replace(f"{suf} ", " ")
    return s


def parse_date_flexible(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    ss = s.strip()
    ss = ss.replace("–", "-").replace("—", "-")
    ss = ss.replace("Jan.", "Jan").replace("Feb.", "Feb").replace("Mar.", "Mar")
    ss = ss.replace("Apr.", "Apr").replace("Jun.", "Jun").replace("Jul.", "Jul")
    ss = ss.replace("Aug.", "Aug").replace("Sep.", "Sep").replace("Oct.", "Oct")
    ss = ss.replace("Nov.", "Nov").replace("Dec.", "Dec")
    ss = _strip_ordinals(ss)

    fmts = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y %B %d",
        "%Y %b %d",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(ss, fmt).date()
        except Exception:
            continue
    # Try handle like "February 2026" by assuming first day
    try:
        dt = datetime.strptime(ss, "%B %Y").date()
        return date(dt.year, dt.month, 1)
    except Exception:
        pass
    # Try handle like "Feb 2026"
    try:
        dt = datetime.strptime(ss, "%b %Y").date()
        return date(dt.year, dt.month, 1)
    except Exception:
        pass
    return None


def inclusive_day_span(start: Optional[str], end: Optional[str]) -> Optional[int]:
    ds = parse_date_flexible(start)
    de = parse_date_flexible(end)
    if not ds or not de:
        return None
    return (de - ds).days + 1


def days_between(d1: Optional[str], d2: Optional[str]) -> Optional[int]:
    dd1 = parse_date_flexible(d1)
    dd2 = parse_date_flexible(d2)
    if not dd1 or not dd2:
        return None
    return (dd2 - dd1).days


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_conference_requirements(evaluator: Evaluator, parent, data: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="conference_requirements",
        desc="Conference identity, organizer, location, dates/duration, and co-located conferences satisfy the stated constraints.",
        parent=parent,
        critical=True
    )

    # Presence: official name
    evaluator.add_custom_node(
        result=bool(data.full_official_name and data.full_official_name.strip()),
        id="conference_name_provided",
        desc="The full official conference name is provided.",
        parent=node,
        critical=True
    )

    # 106th edition supported
    edition_node = evaluator.add_leaf(
        id="conference_is_106th_edition",
        desc="The conference is the 106th edition of its series.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This conference is the 106th edition of its series (e.g., '106th Annual Meeting' or equivalent phrasing).",
        node=edition_node,
        sources=merge_urls(data.conference_urls, data.organizer_urls),
        additional_instruction="Consider exact or near-equivalent phrasing such as '106th Annual Meeting' as confirmation."
    )

    # Organizer full name provided
    evaluator.add_custom_node(
        result=bool(data.organizing_society_full_name and data.organizing_society_full_name.strip()),
        id="organizing_society_full_name_provided",
        desc="The organizing society's full name is provided.",
        parent=node,
        critical=True
    )

    # Organizer is a major American meteorological society and organizes the conference
    org_claim_node = evaluator.add_leaf(
        id="organized_by_american_meteorological_society_type_with_evidence",
        desc="Reliable source(s) support that the organizer is a U.S.-based professional society focused on meteorology/atmospheric science and that it organizes the conference.",
        parent=node,
        critical=True
    )
    org_name = data.organizing_society_full_name or "the organizing society"
    await evaluator.verify(
        claim=f"The conference is organized by {org_name}, a U.S.-based professional society focused on meteorology/atmospheric science.",
        node=org_claim_node,
        sources=merge_urls(data.organizer_urls, data.conference_urls),
        additional_instruction="Look for explicit organizer identity (e.g., 'American Meteorological Society') and its nature (U.S.-based meteorological society)."
    )

    # Held in Houston, Texas
    held_houston_node = evaluator.add_leaf(
        id="held_in_houston_texas",
        desc="The conference took place in Houston, Texas.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The conference took place in Houston, Texas, United States.",
        node=held_houston_node,
        sources=merge_urls(data.conference_urls, data.venue_urls),
        additional_instruction="Check the official event or venue pages for the city and state."
    )

    # Venue is a convention center
    venue_is_cc_node = evaluator.add_leaf(
        id="venue_is_convention_center",
        desc="The conference venue was a convention center.",
        parent=node,
        critical=True
    )
    venue_name = data.venue_name or "the venue"
    await evaluator.verify(
        claim=f"{venue_name} is a convention center.",
        node=venue_is_cc_node,
        sources=merge_urls(data.venue_urls, data.conference_urls),
        additional_instruction="Accept clear descriptions indicating it is a convention center."
    )

    # Venue name provided
    evaluator.add_custom_node(
        result=bool(data.venue_name and data.venue_name.strip()),
        id="venue_name_provided",
        desc="The specific venue name in Houston is provided.",
        parent=node,
        critical=True
    )

    # Start and End dates provided
    evaluator.add_custom_node(
        result=bool(data.start_date and data.start_date.strip()),
        id="start_date_provided",
        desc="The exact conference start date is provided.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(data.end_date and data.end_date.strip()),
        id="end_date_provided",
        desc="The exact conference end date is provided.",
        parent=node,
        critical=True
    )

    # Conference occurred in late January 2026 (evidence-based from conference URLs)
    late_jan_node = evaluator.add_leaf(
        id="conference_in_late_january_2026",
        desc="The conference occurred in late January 2026.",
        parent=node,
        critical=True
    )
    sd_txt = data.start_date or ""
    ed_txt = data.end_date or ""
    await evaluator.verify(
        claim=f"The conference ran from {sd_txt} to {ed_txt}, and those dates are in late January 2026.",
        node=late_jan_node,
        sources=merge_urls(data.conference_urls),
        additional_instruction="Treat 'late January' as approximately the period from the 20th to the 31st of January."
    )

    # Conference lasted 5 consecutive days (derived check using provided dates)
    span = inclusive_day_span(data.start_date, data.end_date)
    evaluator.add_custom_node(
        result=(span == 5),
        id="conference_lasted_5_consecutive_days",
        desc="The conference lasted 5 consecutive days.",
        parent=node,
        critical=True
    )

    # Co-located: 27th Conference on Satellite Meteorology, Oceanography, and Climatology
    sat27_node = evaluator.add_leaf(
        id="includes_27th_satellite_meteorology_oceanography_climatology",
        desc="The event included a co-located 27th Conference on Satellite Meteorology, Oceanography, and Climatology.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The conference program included the '27th Conference on Satellite Meteorology, Oceanography, and Climatology' as a co-located event.",
        node=sat27_node,
        sources=merge_urls(data.satellite_conf_urls, data.conference_urls),
        additional_instruction="Look for exact or near-exact naming indicating it is the 27th installment and is part of the same meeting."
    )

    # Co-located: 23rd Conference on Space Weather
    swx23_node = evaluator.add_leaf(
        id="includes_23rd_conference_on_space_weather",
        desc="The event included a co-located 23rd Conference on Space Weather.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The conference program included the '23rd Conference on Space Weather' as a co-located event.",
        node=swx23_node,
        sources=merge_urls(data.space_weather_conf_urls, data.conference_urls),
        additional_instruction="Look for exact or near-exact naming indicating it is the 23rd installment and is part of the same meeting."
    )


async def build_astronomical_phenomenon(evaluator: Evaluator, parent, data: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="astronomical_phenomenon",
        desc="Post-conference total lunar eclipse timing and characteristics are verified.",
        parent=parent,
        critical=True
    )

    # Eclipse date provided
    evaluator.add_custom_node(
        result=bool(data.eclipse_date and data.eclipse_date.strip()),
        id="eclipse_date_provided",
        desc="The date of the total lunar eclipse is provided.",
        parent=node,
        critical=True
    )

    # Eclipse occurred within 35 days after conference end (derived check)
    diff_days = days_between(data.end_date, data.eclipse_date)
    evaluator.add_custom_node(
        result=(diff_days is not None and 0 < diff_days <= 35),
        id="eclipse_within_35_days_after_conference_end",
        desc="The total lunar eclipse occurred within 35 days after the conference ended.",
        parent=node,
        critical=True
    )

    # Eclipse totality duration 58–59 minutes (evidence-based)
    eclipse_dur_node = evaluator.add_leaf(
        id="eclipse_totality_duration_58_59_minutes",
        desc="The eclipse totality duration was 58–59 minutes.",
        parent=node,
        critical=True
    )
    edate_txt = data.eclipse_date or "the eclipse date"
    await evaluator.verify(
        claim=f"The total lunar eclipse on {edate_txt} had a totality duration between 58 and 59 minutes.",
        node=eclipse_dur_node,
        sources=merge_urls(data.eclipse_urls),
        additional_instruction="Accept values like '58 minutes', '59 minutes', or approximate statements clearly within 58–59 minutes."
    )

    # Eclipse umbral magnitude exceeded 1.15 (evidence-based)
    eclipse_mag_node = evaluator.add_leaf(
        id="eclipse_umbral_magnitude_over_1_15",
        desc="The eclipse umbral magnitude exceeded 1.15.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The umbral magnitude of the total lunar eclipse on {edate_txt} exceeded 1.15.",
        node=eclipse_mag_node,
        sources=merge_urls(data.eclipse_urls),
        additional_instruction="Look for 'umbral magnitude' or 'Umag' values greater than 1.15."
    )

    # Eclipse visible from western North America (evidence-based)
    eclipse_vis_node = evaluator.add_leaf(
        id="eclipse_visible_from_western_north_america",
        desc="The eclipse was visible from western North America.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total lunar eclipse on {edate_txt} was visible from western North America.",
        node=eclipse_vis_node,
        sources=merge_urls(data.eclipse_urls),
        additional_instruction="Visibility maps or text explicitly mentioning western U.S./Canada or 'western North America' should count."
    )


async def build_atmospheric_phenomenon(evaluator: Evaluator, parent, data: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="atmospheric_phenomenon",
        desc="Post-conference sudden stratospheric warming (SSW) timing and characteristics are verified.",
        parent=parent,
        critical=True
    )

    # SSW timing provided (early Feb 2026)
    evaluator.add_custom_node(
        result=bool(data.ssw_confirmation_date and data.ssw_confirmation_date.strip()),
        id="ssw_timing_early_feb_2026_provided",
        desc="The answer provides the SSW event timing/confirmation in early February 2026.",
        parent=node,
        critical=True
    )

    # SSW occurred approximately one week after conference end (derived check)
    diff_days_ssw = days_between(data.end_date, data.ssw_confirmation_date)
    # Allow a small tolerance around 7 days (e.g., <= 8 days)
    evaluator.add_custom_node(
        result=(diff_days_ssw is not None and 0 < diff_days_ssw <= 8),
        id="ssw_within_approx_one_week_after_conference_end",
        desc="The SSW event confirmation occurred approximately one week after the conference ended.",
        parent=node,
        critical=True
    )

    # SSW polar vortex disruption (evidence-based)
    ssw_pv_node = evaluator.add_leaf(
        id="ssw_polar_vortex_disruption",
        desc="The SSW event caused polar vortex disruption.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The early February 2026 SSW event caused polar vortex disruption (e.g., displacement or split).",
        node=ssw_pv_node,
        sources=merge_urls(data.ssw_urls),
        additional_instruction="Accept wording indicating the polar vortex was disrupted, displaced, or split."
    )

    # SSW temperature increase up to ~70°F within days (evidence-based)
    ssw_temp_node = evaluator.add_leaf(
        id="ssw_temperature_increase_up_to_70f",
        desc="The SSW event caused stratospheric temperature increases of up to 70°F within days.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="During the early February 2026 SSW event, stratospheric temperatures increased by up to around 70°F within days.",
        node=ssw_temp_node,
        sources=merge_urls(data.ssw_urls),
        additional_instruction="Treat values near 70°F (about 39–40°C) as equivalent. Look for phrasing like 'up to 70°F' or 'around 40°C'."
    )


async def build_references_section(evaluator: Evaluator, parent, data: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="references",
        desc="All required information is supported by reference URLs from reliable sources.",
        parent=parent,
        critical=True
    )

    # Check that URLs are present for all required claim categories.
    urls_ok = all([
        len(data.conference_urls) > 0,
        len(data.organizer_urls) > 0,
        len(data.venue_urls) > 0,
        len(data.satellite_conf_urls) > 0,
        len(data.space_weather_conf_urls) > 0,
        len(data.eclipse_urls) > 0,
        len(data.ssw_urls) > 0
    ])

    evaluator.add_custom_node(
        result=urls_ok,
        id="urls_provided_for_all_required_claims",
        desc="Reference URLs are provided that support each required field and each stated constraint (conference identity/dates/venue/co-located conferences, eclipse details, and SSW timing/characteristics).",
        parent=node,
        critical=True
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
    # Initialize evaluator with a critical root as parallel aggregator
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
        default_model=model,
    )
    # Mark root as critical by wrapping in a critical child aggregator according to framework constraints
    # Instead, we set the root's children as critical and let root aggregate; to enforce global criticality, all sections are critical=True.

    # Extract structured fields from the answer
    extracted: ConferenceExtraction = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=ConferenceExtraction,
        extraction_name="extracted_conference_facts"
    )

    # Build verification subtrees
    await build_conference_requirements(evaluator, root, extracted)
    await build_astronomical_phenomenon(evaluator, root, extracted)
    await build_atmospheric_phenomenon(evaluator, root, extracted)
    await build_references_section(evaluator, root, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()