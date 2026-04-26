import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "us_conferences_h2_2025_posters"
TASK_DESCRIPTION = (
    "You are planning to review major academic conference opportunities in the United States during the second half of 2025 "
    "(July 1 - December 31, 2025) for potential poster presentations in Earth sciences, environmental sciences, "
    "atmospheric sciences, planetary sciences, neuroscience, or computational/AI sciences fields.\n\n"
    "Identify 3 to 4 major annual academic conferences that meet ALL of the following requirements:\n"
    "1. The conference must have taken place (or be scheduled to take place) in the United States between July 1, 2025 and December 31, 2025.\n"
    "2. The conference must be organized by or associated with a recognized professional scientific society or organization.\n"
    "3. The conference must be relevant to at least one of these fields: Earth sciences, environmental sciences, atmospheric sciences, "
    "planetary sciences, neuroscience, or AI/machine learning/computational sciences.\n"
    "4. The conference must explicitly accept poster presentations as a presentation format.\n"
    "5. The conferences you select must not have overlapping dates with each other.\n"
    "6. At least 2 different U.S. cities must be represented among your selected conferences.\n\n"
    "For each conference you identify, provide the following information with supporting URL references:\n"
    "- Full official conference name and the professional organization that organizes/sponsors it\n"
    "- Exact conference dates (start date and end date)\n"
    "- Location: specific host city, U.S. state, and venue name where the conference is held\n"
    "- Abstract submission deadline that was published for the conference\n"
    "- Poster presentation format requirements: poster size (dimensions) and orientation (landscape or portrait)\n"
    "- Reference URLs: official conference website and specific URLs documenting the dates, deadlines, venue, and presentation format requirements\n\n"
    "All information must be verifiable from official conference sources."
)

WINDOW_START = datetime(2025, 7, 1)
WINDOW_END = datetime(2025, 12, 31)


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class ConferenceURLs(BaseModel):
    official_site_urls: List[str] = Field(default_factory=list)
    organizer_urls: List[str] = Field(default_factory=list)
    dates_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    deadline_urls: List[str] = Field(default_factory=list)
    poster_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class ConferenceItem(BaseModel):
    name: Optional[str] = None
    organizer: Optional[str] = None
    field_or_domain: Optional[str] = None

    start_date: Optional[str] = None  # Prefer ISO YYYY-MM-DD
    end_date: Optional[str] = None    # Prefer ISO YYYY-MM-DD

    city: Optional[str] = None
    state: Optional[str] = None
    venue: Optional[str] = None

    abstract_deadline: Optional[str] = None  # Prefer ISO YYYY-MM-DD

    poster_size: Optional[str] = None        # e.g., "36x48 inches"
    poster_orientation: Optional[str] = None # "landscape" or "portrait"

    urls: ConferenceURLs = Field(default_factory=ConferenceURLs)


class ConferencesExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_conferences() -> str:
    return """
Extract the first up to 4 distinct major academic conferences listed in the answer that are held in the U.S. between 2025-07-01 and 2025-12-31.
For each conference, return a JSON object with the following fields (use null if missing):

- name: Full official conference name
- organizer: The professional scientific society or recognized organization that organizes/sponsors/associates with the conference
- field_or_domain: The primary scientific field(s) this conference covers
- start_date: Exact start date, ISO format YYYY-MM-DD if possible
- end_date: Exact end date, ISO format YYYY-MM-DD if possible
- city: U.S. host city (e.g., "San Francisco")
- state: U.S. state (e.g., "CA" or "California")
- venue: Specific venue name (e.g., "Moscone Center")
- abstract_deadline: The published abstract submission deadline (ISO format if possible)
- poster_size: The stated poster size/dimensions exactly as written (e.g., "36 x 48 inches" or "A0")
- poster_orientation: The stated orientation ("landscape" or "portrait") if given, otherwise null

- urls: An object grouping only official-source URLs explicitly provided in the answer:
  - official_site_urls: URLs to the official conference website (home/overview pages)
  - organizer_urls: URLs to the professional society/organization pages about the conference or the society itself
  - dates_urls: URLs that explicitly state the 2025 conference dates
  - location_urls: URLs that explicitly state the city/state and the venue
  - deadline_urls: URLs that explicitly state the abstract submission deadline
  - poster_urls: URLs that explicitly state that posters are accepted and state poster format requirements (size/orientation)
  - other_urls: any other official URLs mentioned for this conference

IMPORTANT:
- Only extract URLs that are explicitly in the answer, and prefer official conference/organizer domains over third-party sites.
- If some information is missing from the answer, set that field to null (do not invent).
- Return a top-level object with a 'conferences' array of 3–4 items if available (fewer if the answer provides fewer).
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    # Try common formats
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    # Try remove ordinal suffixes ("st", "nd", "rd", "th")
    import re
    s2 = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', s)
    if s2 != s:
        for f in fmts:
            try:
                return datetime.strptime(s2, f)
            except Exception:
                continue
    return None


def dates_within_window(start_date: Optional[str], end_date: Optional[str]) -> bool:
    s = _parse_date(start_date)
    e = _parse_date(end_date)
    if not s or not e:
        return False
    return (WINDOW_START <= s <= WINDOW_END) and (WINDOW_START <= e <= WINDOW_END) and (s <= e)


def intervals_non_overlapping(intervals: List[Tuple[datetime, datetime]]) -> bool:
    if not intervals:
        return False
    # Sort by start
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    for i in range(1, len(intervals_sorted)):
        prev_end = intervals_sorted[i - 1][1]
        cur_start = intervals_sorted[i][0]
        # Non-overlapping means previous end < current start (strictly no overlap)
        if prev_end >= cur_start:
            return False
    return True


def gather_all_urls(conf: ConferenceItem) -> List[str]:
    urls: List[str] = []
    urls.extend(conf.urls.official_site_urls)
    urls.extend(conf.urls.organizer_urls)
    urls.extend(conf.urls.dates_urls)
    urls.extend(conf.urls.location_urls)
    urls.extend(conf.urls.deadline_urls)
    urls.extend(conf.urls.poster_urls)
    urls.extend(conf.urls.other_urls)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def official_only_instruction(extra: Optional[str] = None) -> str:
    base = (
        "Only consider the claim supported if it is explicitly documented on an official conference or organizer "
        "webpage (e.g., the conference's official site or the professional society's site). "
        "If no source URLs are provided or the pages are non-official/irrelevant, mark as not supported. "
        "Ensure the information specifically corresponds to the 2025 edition of the event."
    )
    if extra:
        return base + " " + extra
    return base


# -----------------------------------------------------------------------------
# Per-conference verification
# -----------------------------------------------------------------------------
async def verify_single_conference(
    evaluator: Evaluator,
    parent,
    conf: ConferenceItem,
    index: int,
) -> Dict[str, Any]:
    """
    Build the per-conference verification subtree and run all leaf verifications.
    Returns a registry of created nodes for cross checks.
    """
    conf_idx = index + 1
    conf_node = evaluator.add_parallel(
        id=f"Conference_{conf_idx}",
        desc=f"Conference {conf_idx} (evaluate if provided)",
        parent=parent,
        critical=False,
    )

    # Prepare values and sources
    name = normalize_str(conf.name)
    organizer = normalize_str(conf.organizer)
    field_domain = normalize_str(conf.field_or_domain)
    start_date = normalize_str(conf.start_date)
    end_date = normalize_str(conf.end_date)
    city = normalize_str(conf.city)
    state = normalize_str(conf.state)
    venue = normalize_str(conf.venue)
    abs_deadline = normalize_str(conf.abstract_deadline)
    poster_size = normalize_str(conf.poster_size)
    poster_orientation = normalize_str(conf.poster_orientation)

    # Specialized source groups with sensible fallbacks
    urls_official = conf.urls.official_site_urls or []
    urls_org = conf.urls.organizer_urls or []
    urls_dates = conf.urls.dates_urls or urls_official
    urls_location = conf.urls.location_urls or urls_official
    urls_deadline = conf.urls.deadline_urls or urls_official
    urls_poster = conf.urls.poster_urls or urls_official
    urls_all = gather_all_urls(conf)

    critical_leaves = []

    # 1) Name and Organizer (critical)
    leaf_name_org = evaluator.add_leaf(
        id=f"Conf{conf_idx}_NameAndOrganizer",
        desc="Provides full official conference name AND the organizing/sponsoring/associated recognized professional scientific society/organization",
        parent=conf_node,
        critical=True,
    )
    claim_name_org = (
        f"The official conference name is '{name}', and it is organized/sponsored by or associated with '{organizer}', "
        f"which is a recognized professional scientific society/organization."
    )
    await evaluator.verify(
        claim=claim_name_org,
        node=leaf_name_org,
        sources=(urls_official + urls_org) if (urls_official or urls_org) else urls_all,
        additional_instruction=official_only_instruction(
            "Confirm both the name and the organizing society/organization from official pages."
        ),
    )
    critical_leaves.append(leaf_name_org)

    # 2) Major annual conference (critical)
    leaf_major_annual = evaluator.add_leaf(
        id=f"Conf{conf_idx}_MajorAnnualConference",
        desc="Conference is a major annual academic/scientific conference (not a workshop/symposium) per constraints",
        parent=conf_node,
        critical=True,
    )
    claim_major_annual = (
        "This event is a major annual academic or scientific conference (not a smaller workshop or single-track symposium) "
        "run by a professional society or equivalent organization."
    )
    await evaluator.verify(
        claim=claim_major_annual,
        node=leaf_major_annual,
        sources=(urls_official + urls_org) if (urls_official or urls_org) else urls_all,
        additional_instruction=official_only_instruction(
            "Look for explicit indications (e.g., 'annual meeting', 'annual conference') on official pages."
        ),
    )
    critical_leaves.append(leaf_major_annual)

    # 3) Field relevance (critical)
    leaf_field = evaluator.add_leaf(
        id=f"Conf{conf_idx}_FieldRelevance",
        desc="Conference is relevant to at least one specified field (Earth/environmental/atmospheric/planetary/neuroscience/AI-ML-computational)",
        parent=conf_node,
        critical=True,
    )
    claim_field = (
        "The conference is relevant to at least one of the following fields: Earth sciences, environmental sciences, "
        "atmospheric sciences, planetary sciences, neuroscience, or AI/machine learning/computational sciences."
    )
    await evaluator.verify(
        claim=claim_field,
        node=leaf_field,
        sources=urls_all if urls_all else None,
        additional_instruction=official_only_instruction(
            "Judge based on the conference scope, tracks, or society domain stated on official pages."
        ),
    )
    critical_leaves.append(leaf_field)

    # 4) Dates within window (critical)
    leaf_dates = evaluator.add_leaf(
        id=f"Conf{conf_idx}_DatesProvidedAndWithinWindow",
        desc="Provides exact start and end dates AND dates fall between July 1, 2025 and Dec 31, 2025",
        parent=conf_node,
        critical=True,
    )
    claim_dates = (
        f"The conference takes place from {start_date} to {end_date}, and these dates fall between July 1, 2025 and December 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=leaf_dates,
        sources=urls_dates if urls_dates else urls_all,
        additional_instruction=official_only_instruction(
            "Verify both exact start and end dates and ensure they belong to the 2025 edition and lie within the specified window."
        ),
    )
    critical_leaves.append(leaf_dates)

    # 5) U.S. location provided (critical)
    leaf_location = evaluator.add_leaf(
        id=f"Conf{conf_idx}_USLocationProvided",
        desc="Provides U.S. host city, U.S. state, and specific venue name",
        parent=conf_node,
        critical=True,
    )
    claim_location = (
        f"The conference is held at '{venue}' in {city}, {state}, USA."
    )
    await evaluator.verify(
        claim=claim_location,
        node=leaf_location,
        sources=urls_location if urls_location else urls_all,
        additional_instruction=official_only_instruction(
            "Confirm the city, state, and named venue from official pages (program, venue, or travel pages)."
        ),
    )
    critical_leaves.append(leaf_location)

    # 6) Abstract submission deadline (critical)
    leaf_deadline = evaluator.add_leaf(
        id=f"Conf{conf_idx}_AbstractDeadlineProvided",
        desc="Provides the published abstract submission deadline date",
        parent=conf_node,
        critical=True,
    )
    claim_deadline = f"The published abstract submission deadline is {abs_deadline}."
    await evaluator.verify(
        claim=claim_deadline,
        node=leaf_deadline,
        sources=urls_deadline if urls_deadline else urls_all,
        additional_instruction=official_only_instruction(
            "Verify the exact abstract submission deadline date for the 2025 edition (call for abstracts/submissions)."
        ),
    )
    critical_leaves.append(leaf_deadline)

    # 7) Poster acceptance (critical)
    leaf_poster_accept = evaluator.add_leaf(
        id=f"Conf{conf_idx}_PosterAcceptance",
        desc="Conference explicitly accepts poster presentations as a format",
        parent=conf_node,
        critical=True,
    )
    claim_poster_accept = "The conference explicitly accepts poster presentations as a presentation format."
    await evaluator.verify(
        claim=claim_poster_accept,
        node=leaf_poster_accept,
        sources=urls_poster if urls_poster else urls_all,
        additional_instruction=official_only_instruction(
            "Look for 'poster' as an accepted presentation format or submission type on official pages."
        ),
    )
    critical_leaves.append(leaf_poster_accept)

    # 8) Poster format requirements (critical)
    leaf_poster_format = evaluator.add_leaf(
        id=f"Conf{conf_idx}_PosterFormatRequirements",
        desc="Provides poster format requirements including poster size (dimensions) AND orientation (landscape/portrait)",
        parent=conf_node,
        critical=True,
    )
    claim_poster_format = (
        f"The poster format requirements specify size '{poster_size}' and orientation '{poster_orientation}'."
    )
    await evaluator.verify(
        claim=claim_poster_format,
        node=leaf_poster_format,
        sources=urls_poster if urls_poster else urls_all,
        additional_instruction=official_only_instruction(
            "Confirm both the poster size/dimensions and the orientation from official guidelines."
        ),
    )
    critical_leaves.append(leaf_poster_format)

    # 9) Official sources coverage (critical)
    leaf_official_sources = evaluator.add_leaf(
        id=f"Conf{conf_idx}_OfficialSourceURLs",
        desc="Provides official conference website URL(s) and official-source URL(s) that document the stated dates, location/venue, abstract deadline, and poster acceptance/format requirements",
        parent=conf_node,
        critical=True,
    )
    claim_official_sources = (
        "The provided URLs are official conference/organizer webpages and, collectively, they document the 2025 dates, "
        "the city/state/venue, the abstract deadline, and the poster acceptance and poster format requirements (size and orientation)."
    )
    await evaluator.verify(
        claim=claim_official_sources,
        node=leaf_official_sources,
        sources=urls_all if urls_all else None,
        additional_instruction=official_only_instruction(
            "If any of the required aspects lack an official-source URL, this should be marked as not supported."
        ),
    )
    critical_leaves.append(leaf_official_sources)

    return {
        "conf_node": conf_node,
        "critical_leaves": critical_leaves,
        "conf_data": {
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "city": city,
            "state": state,
            "venue": venue,
        },
    }


# -----------------------------------------------------------------------------
# Cross-conference verification helpers
# -----------------------------------------------------------------------------
def _selected_conferences(confs: List[ConferenceItem]) -> List[ConferenceItem]:
    # Keep only those with at least a name; take first up to 4
    filtered = [c for c in confs if normalize_str(c.name)]
    return filtered[:4]


def _count_3_or_4_distinct(selected: List[ConferenceItem]) -> bool:
    names = [normalize_str(c.name).lower() for c in selected if normalize_str(c.name)]
    distinct = list(dict.fromkeys(names))
    return len(distinct) in (3, 4)


def _non_overlapping_dates_for_selected(selected: List[ConferenceItem]) -> bool:
    intervals: List[Tuple[datetime, datetime]] = []
    for c in selected:
        s = _parse_date(c.start_date)
        e = _parse_date(c.end_date)
        if not s or not e:
            return False
        intervals.append((s, e))
    return intervals_non_overlapping(intervals)


def _at_least_two_us_cities(selected: List[ConferenceItem]) -> bool:
    cities = [normalize_str(c.city).lower() for c in selected if normalize_str(c.city)]
    distinct = set(cities)
    return len(distinct) >= 2


def _all_selected_valid_conferences(conference_results: List[Dict[str, Any]]) -> bool:
    """
    Determine if all selected conferences satisfy all their critical leaves.
    Uses aggregated_score of each conference node, which will be 1.0 only if all critical children passed.
    """
    if not conference_results:
        return False
    for res in conference_results:
        conf_node = res["conf_node"]
        # Trigger computation
        if conf_node.aggregated_score < 1.0:
            return False
    return True


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
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

    # Create a critical top-level aggregator to mirror the rubric's Root critical node
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Identify 3–4 major annual academic conferences in the U.S. between Jul 1–Dec 31, 2025, in the specified fields, "
             "each explicitly accepting posters, with required details and official-source URLs; selected conferences must be "
             "non-overlapping and span ≥2 U.S. cities.",
        parent=root,
        critical=True,
    )

    # Extract conferences mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction",
    )

    # Choose up to the first 4 with names (filtering/padding handled implicitly by checks)
    selected = _selected_conferences(extracted.conferences)

    # Build per-conference trees
    conference_results: List[Dict[str, Any]] = []
    for idx, conf in enumerate(selected):
        res = await verify_single_conference(evaluator, task_root, conf, idx)
        conference_results.append(res)

    # If fewer than 3 were provided, we still create placeholders (these won't pass checks)
    # Do not fabricate items; just note in custom info.
    evaluator.add_custom_info(
        {"selected_count": len(selected), "total_extracted": len(extracted.conferences)},
        info_type="selection_stats",
        info_name="selection_statistics",
    )

    # Cross-conference constraints (critical group)
    cross_node = evaluator.add_parallel(
        id="CrossConference_Requirements",
        desc="Set-level constraints across the selected conferences",
        parent=task_root,
        critical=True,
    )

    # 1) ProvidedConferenceCountIs3Or4Distinct
    count_ok = _count_3_or_4_distinct(selected)
    evaluator.add_custom_node(
        result=count_ok,
        id="ProvidedConferenceCountIs3Or4Distinct",
        desc="Exactly 3 or 4 distinct conferences are provided (not fewer, not more, no duplicates)",
        parent=cross_node,
        critical=True,
    )

    # 2) NonOverlappingDates
    non_overlap_ok = _non_overlapping_dates_for_selected(selected)
    evaluator.add_custom_node(
        result=non_overlap_ok,
        id="NonOverlappingDates",
        desc="No two provided conferences have overlapping dates",
        parent=cross_node,
        critical=True,
    )

    # 3) AtLeastTwoDifferentUSCities
    two_cities_ok = _at_least_two_us_cities(selected)
    evaluator.add_custom_node(
        result=two_cities_ok,
        id="AtLeastTwoDifferentUSCities",
        desc="Among the provided conferences, at least 2 different U.S. cities are represented",
        parent=cross_node,
        critical=True,
    )

    # 4) AllProvidedConferencesValidSelections
    all_valid_ok = _all_selected_valid_conferences(conference_results)
    evaluator.add_custom_node(
        result=all_valid_ok,
        id="AllProvidedConferencesValidSelections",
        desc="All provided conferences (the 3–4 listed) satisfy all per-conference critical requirements (i.e., every listed conference is a valid selection)",
        parent=cross_node,
        critical=True,
    )

    # Return final summary
    return evaluator.get_summary()