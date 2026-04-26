import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ml_conf_vienna_2024"
TASK_DESCRIPTION = (
    "A researcher is planning to attend a major international computer science conference focused on machine learning "
    "that took place in Vienna, Austria during July 2024. The conference must be recognized as a top-tier (A* or A-ranked) "
    "venue and must have lasted at least 5 consecutive days. What is the name and full official title of this conference, "
    "and provide the following details: (1) The exact dates the conference was held, (2) The specific venue name where it "
    "was hosted, (3) Whether the conference used double-blind peer review for paper submissions, (4) The page limit for "
    "full paper submissions, (5) Whether both oral and poster presentations were accepted, (6) Whether workshop sessions "
    "were included in the program, (7) Whether student registration discounts were offered, (8) Whether conference "
    "proceedings were published, (9) Whether author registration was required for accepted papers."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceIdentifiers(BaseModel):
    conference_name: Optional[str] = None  # Short/acronym name if present (e.g., ICML 2024)
    official_title: Optional[str] = None   # Full official title (e.g., The 41st International Conference on Machine Learning)
    short_name_acronym: Optional[str] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    venue_name: Optional[str] = None


class ConferenceDates(BaseModel):
    dates_text: Optional[str] = None              # Raw text dates range, if the answer provides it
    start_date_text: Optional[str] = None         # Exact start date string (e.g., July 21, 2024)
    end_date_text: Optional[str] = None           # Exact end date string (e.g., July 27, 2024)


class ConferencePolicies(BaseModel):
    double_blind_review_text: Optional[str] = None                # e.g., "double-blind", "single-blind"
    full_paper_page_limit_text: Optional[str] = None              # e.g., "8 pages excluding references"
    accepts_oral_and_poster_text: Optional[str] = None            # e.g., "both oral and poster presentations"
    includes_workshops_text: Optional[str] = None                 # e.g., "workshops included"
    student_registration_discounts_text: Optional[str] = None     # e.g., "student discounts offered"
    proceedings_published_text: Optional[str] = None              # e.g., "proceedings available/published"
    author_registration_required_text: Optional[str] = None       # e.g., "at least one author must register"


class ConferenceSources(BaseModel):
    all_source_urls: List[str] = Field(default_factory=list)
    official_site_urls: List[str] = Field(default_factory=list)
    ranking_source_urls: List[str] = Field(default_factory=list)
    review_policy_urls: List[str] = Field(default_factory=list)
    submission_policy_urls: List[str] = Field(default_factory=list)
    registration_urls: List[str] = Field(default_factory=list)
    proceedings_urls: List[str] = Field(default_factory=list)


class ConferenceExtraction(BaseModel):
    identifiers: Optional[ConferenceIdentifiers] = None
    dates: Optional[ConferenceDates] = None
    policies: Optional[ConferencePolicies] = None
    sources: Optional[ConferenceSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_info() -> str:
    return """
    Extract the conference information exactly as presented in the answer text. Do not invent any details.
    Return a JSON object with the following nested structure and fields:

    identifiers:
      - conference_name: The short or acronym name of the conference, if stated (e.g., "ICML 2024"); otherwise null.
      - official_title: The full official title of the conference exactly as quoted in the answer (e.g., "The 41st International Conference on Machine Learning"); if missing, null.
      - short_name_acronym: If the answer provides a commonly used acronym (e.g., "ICML"), extract it; otherwise null.
      - location_city: Extract the city (e.g., "Vienna"); otherwise null.
      - location_country: Extract the country (e.g., "Austria"); otherwise null.
      - venue_name: Extract the specific venue name (e.g., "Messe Wien Exhibition & Congress Center"); otherwise null.

    dates:
      - dates_text: If the answer provides a date range in text form (e.g., "July 21–27, 2024"), extract it exactly; otherwise null.
      - start_date_text: Extract the exact start date string as provided (e.g., "July 21, 2024" or "21 July 2024"). If only month/year are present and no exact day, return null.
      - end_date_text: Extract the exact end date string as provided (e.g., "July 27, 2024" or "27 July 2024"). If only month/year are present and no exact day, return null.

    policies:
      - double_blind_review_text: Extract the statement about review model if present (e.g., "double-blind"); otherwise null.
      - full_paper_page_limit_text: Extract the full-paper page limit text exactly (e.g., "8 pages excluding references"); otherwise null.
      - accepts_oral_and_poster_text: Extract the statement indicating both oral and poster presentations were accepted if present; otherwise null.
      - includes_workshops_text: Extract the statement indicating workshop sessions were included if present; otherwise null.
      - student_registration_discounts_text: Extract the statement indicating student registration discounts were offered if present; otherwise null.
      - proceedings_published_text: Extract the statement indicating proceedings were published or made available if present; otherwise null.
      - author_registration_required_text: Extract the statement indicating at least one author per accepted paper must register if present; otherwise null.

    sources:
      - all_source_urls: Extract ALL URLs mentioned in the answer, including official pages, CFP pages, registration pages, proceedings pages, and ranking pages. If none, return an empty list.
      - official_site_urls: Extract the official conference website URLs if present; otherwise return an empty list.
      - ranking_source_urls: Extract URLs that indicate top-tier rankings (A* or A) for the conference, if present; otherwise return an empty list.
      - review_policy_urls: Extract URLs that discuss peer review policies, if present; otherwise return an empty list.
      - submission_policy_urls: Extract URLs related to submissions, page limits, author guidelines, etc., if present; otherwise return an empty list.
      - registration_urls: Extract URLs related to registration policies/fees/discounts, if present; otherwise return an empty list.
      - proceedings_urls: Extract URLs for proceedings (e.g., publication venue pages), if present; otherwise return an empty list.

    IMPORTANT:
    - Only extract information explicitly present in the answer. If a field is not present, return null (for strings) or an empty list (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_non_empty(*vals: Optional[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _has_day_number(text: Optional[str]) -> bool:
    """Return True if the text contains a day-of-month number (1–31)."""
    if not text:
        return False
    return re.search(r"\b([1-9]|[12][0-9]|3[01])\b", text) is not None


def _parse_date_str_generic(text: Optional[str]) -> Optional[datetime]:
    """Try to parse a date string using several common patterns. Returns datetime or None."""
    if not text:
        return None
    t = text.strip()
    # Normalize dash variants
    t = t.replace("–", "-").replace("—", "-")
    # Common patterns
    patterns = [
        "%B %d, %Y",   # July 21, 2024
        "%b %d, %Y",   # Jul 21, 2024
        "%d %B %Y",    # 21 July 2024
        "%d %b %Y",    # 21 Jul 2024
        "%Y-%m-%d",    # 2024-07-21
        "%Y/%m/%d",    # 2024/07/21
        "%m/%d/%Y",    # 07/21/2024
        "%m-%d-%Y",    # 07-21-2024
    ]
    for p in patterns:
        try:
            return datetime.strptime(t, p)
        except Exception:
            pass
    return None


def _compute_duration_days(start_text: Optional[str], end_text: Optional[str]) -> Optional[int]:
    """Compute inclusive consecutive day count from start/end text. Returns None if parsing fails."""
    start_dt = _parse_date_str_generic(start_text)
    end_dt = _parse_date_str_generic(end_text)
    if start_dt and end_dt and end_dt >= start_dt:
        return (end_dt - start_dt).days + 1
    return None


def _sources_pref(
    extraction: ConferenceExtraction,
    field_specific: Optional[List[str]],
) -> List[str]:
    """Choose sources: prefer field-specific; else official; else all."""
    field_specific = field_specific or []
    official = (extraction.sources.official_site_urls if extraction.sources else []) or []
    all_urls = (extraction.sources.all_source_urls if extraction.sources else []) or []
    if field_specific:
        return field_specific
    if official:
        return official
    return all_urls


def _ranking_sources(extraction: ConferenceExtraction) -> List[str]:
    ranking = (extraction.sources.ranking_source_urls if extraction.sources else []) or []
    return ranking or _sources_pref(extraction, ranking)


def _policy_sources(extraction: ConferenceExtraction) -> List[str]:
    # Combine policy-related sources
    candidates = []
    if extraction.sources:
        candidates = (extraction.sources.submission_policy_urls or []) + (extraction.sources.review_policy_urls or [])
    return _sources_pref(extraction, candidates)


def _registration_sources(extraction: ConferenceExtraction) -> List[str]:
    candidates = []
    if extraction.sources:
        candidates = extraction.sources.registration_urls or []
    return _sources_pref(extraction, candidates)


def _proceedings_sources(extraction: ConferenceExtraction) -> List[str]:
    candidates = []
    if extraction.sources:
        candidates = extraction.sources.proceedings_urls or []
    return _sources_pref(extraction, candidates)


def _location_sources(extraction: ConferenceExtraction) -> List[str]:
    return _sources_pref(extraction, None)


def _dates_sources(extraction: ConferenceExtraction) -> List[str]:
    return _sources_pref(extraction, None)


def _page_limit_number(text: Optional[str]) -> Optional[int]:
    """Extract the first integer from page limit text."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_conference(
    evaluator: Evaluator,
    root_node,
    extraction: ConferenceExtraction,
) -> None:
    # Top-level critical parallel aggregator
    conf_root = evaluator.add_parallel(
        id="Conference_Response",
        desc="Evaluate whether the response identifies a conference meeting all stated constraints and provides all requested details.",
        parent=root_node,
        critical=True,
    )

    # 1) Conference Identification & Required Details (critical)
    ident_node = evaluator.add_parallel(
        id="Conference_Identification_And_Required_Details",
        desc="Conference is identified and required identifying details are provided.",
        parent=conf_root,
        critical=True,
    )

    # Full official title provided (existence check)
    official_title = extraction.identifiers.official_title if extraction.identifiers else None
    evaluator.add_custom_node(
        result=bool(official_title and official_title.strip()),
        id="Conference_Full_Official_Title_Provided",
        desc="Provides the name and full official title of the conference.",
        parent=ident_node,
        critical=True,
    )

    # Exact conference dates provided (require day numbers)
    start_text = extraction.dates.start_date_text if extraction.dates else None
    end_text = extraction.dates.end_date_text if extraction.dates else None
    exact_dates_provided = _has_day_number(start_text) and _has_day_number(end_text)
    evaluator.add_custom_node(
        result=exact_dates_provided,
        id="Exact_Conference_Dates_Provided",
        desc="Provides the exact conference dates (not only month/year).",
        parent=ident_node,
        critical=True,
    )

    # Venue name provided
    venue_name = extraction.identifiers.venue_name if extraction.identifiers else None
    evaluator.add_custom_node(
        result=bool(venue_name and venue_name.strip()),
        id="Venue_Name_Provided",
        desc="Provides the specific venue name where the conference was hosted.",
        parent=ident_node,
        critical=True,
    )

    # 2) Conference Eligibility Constraints (critical)
    elig_node = evaluator.add_parallel(
        id="Conference_Eligibility_Constraints",
        desc="Conference satisfies all eligibility constraints (tier, time, location, duration).",
        parent=conf_root,
        critical=True,
    )

    # Prepare common strings
    conf_display_name = _first_non_empty(
        extraction.identifiers.conference_name if extraction.identifiers else None,
        extraction.identifiers.short_name_acronym if extraction.identifiers else None,
        extraction.identifiers.official_title if extraction.identifiers else None,
    )

    # Top-tier A*/A-ranked
    top_tier_leaf = evaluator.add_leaf(
        id="Top_Tier_AStar_Or_A_Ranked",
        desc="Conference is recognized as top-tier (A* or A-ranked) in ML/AI (per the constraint definition).",
        parent=elig_node,
        critical=True,
    )
    tt_claim = (
        f"The conference '{conf_display_name}' is recognized as an A* or A-ranked (top-tier) venue in machine learning/AI."
        if conf_display_name else
        "The conference is recognized as an A* or A-ranked (top-tier) venue in machine learning/AI."
    )
    await evaluator.verify(
        claim=tt_claim,
        node=top_tier_leaf,
        sources=_ranking_sources(extraction),
        additional_instruction=(
            "Confirm via credible ranking sources (e.g., CORE). "
            "Any A* or A classification is acceptable; B or lower is not."
        ),
    )

    # Held in 2024
    held_2024_leaf = evaluator.add_leaf(
        id="Held_In_2024",
        desc="Conference took place in 2024.",
        parent=elig_node,
        critical=True,
    )
    held_2024_claim = (
        f"The conference '{conf_display_name}' took place in 2024."
        if conf_display_name else
        "The conference took place in 2024."
    )
    await evaluator.verify(
        claim=held_2024_claim,
        node=held_2024_leaf,
        sources=_dates_sources(extraction),
        additional_instruction="Check the official program/schedule page for the year.",
    )

    # Held in Vienna, Austria
    held_vienna_leaf = evaluator.add_leaf(
        id="Held_In_Vienna_Austria",
        desc="Conference was held in Vienna, Austria.",
        parent=elig_node,
        critical=True,
    )
    city = extraction.identifiers.location_city if extraction.identifiers else ""
    country = extraction.identifiers.location_country if extraction.identifiers else ""
    held_vienna_claim = (
        f"The conference was held in Vienna, Austria."
    )
    await evaluator.verify(
        claim=held_vienna_claim,
        node=held_vienna_leaf,
        sources=_location_sources(extraction),
        additional_instruction="Confirm the host city and country as Vienna, Austria on official pages.",
    )

    # Dates in July 2024
    july_leaf = evaluator.add_leaf(
        id="Dates_In_July_2024",
        desc="Conference dates fall in July 2024.",
        parent=elig_node,
        critical=True,
    )
    start_show = start_text or "start date"
    end_show = end_text or "end date"
    july_claim = f"The conference dates ({start_show} to {end_show}) fall in July 2024."
    await evaluator.verify(
        claim=july_claim,
        node=july_leaf,
        sources=_dates_sources(extraction),
        additional_instruction=(
            "Verify that both the start and end dates are within July 2024. "
            "If dates span outside July, this should be marked incorrect."
        ),
    )

    # At least 5 consecutive days
    duration_leaf = evaluator.add_leaf(
        id="At_Least_5_Consecutive_Days",
        desc="Conference spanned at least 5 consecutive days.",
        parent=elig_node,
        critical=True,
    )
    # Build a claim using extracted dates, and instruct to compute duration
    duration_claim = (
        f"The conference lasted at least 5 consecutive days, running from {start_show} to {end_show}."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=_dates_sources(extraction),
        additional_instruction=(
            "Use the provided start and end dates to infer total consecutive days (inclusive). "
            "Mark as correct only if the duration is >= 5 days per the official schedule."
        ),
    )

    # 3) Policy & Program Constraints (critical)
    policy_node = evaluator.add_parallel(
        id="Policy_And_Program_Constraints",
        desc="Conference satisfies the policy/program constraints and the answer states them.",
        parent=conf_root,
        critical=True,
    )

    # Double-blind peer review
    dbl_blind_leaf = evaluator.add_leaf(
        id="Uses_Double_Blind_Review",
        desc="States the review model and indicates the conference used double-blind peer review for paper submissions.",
        parent=policy_node,
        critical=True,
    )
    dbl_blind_claim = (
        "The conference used double-blind peer review for paper submissions."
    )
    await evaluator.verify(
        claim=dbl_blind_claim,
        node=dbl_blind_leaf,
        sources=_policy_sources(extraction),
        additional_instruction=(
            "Confirm via the CFP/Author Guidelines/Submission Policies pages that the review was double-blind."
        ),
    )

    # Full-paper page limit within 6–10
    page_limit_leaf = evaluator.add_leaf(
        id="Full_Paper_Page_Limit_Within_6_to_10",
        desc="States the full-paper page limit and the limit is within 6–10 pages (as constrained).",
        parent=policy_node,
        critical=True,
    )
    limit_text = extraction.policies.full_paper_page_limit_text if extraction.policies else None
    limit_num = _page_limit_number(limit_text)
    limit_phrase = limit_text or "the stated full-paper page limit"
    if limit_num is not None:
        pl_claim = f"The full-paper page limit was {limit_num} pages, which is within 6–10 pages."
    else:
        pl_claim = f"The full-paper page limit ({limit_phrase}) is within 6–10 pages for the main content."
    await evaluator.verify(
        claim=pl_claim,
        node=page_limit_leaf,
        sources=_policy_sources(extraction),
        additional_instruction=(
            "Check submission/author guideline pages. If references do not count, that is acceptable; "
            "still ensure the main content limit lies within 6–10 pages."
        ),
    )

    # Accepts both oral and poster presentations
    oral_poster_leaf = evaluator.add_leaf(
        id="Accepts_Both_Oral_And_Poster",
        desc="States presentation formats and indicates both oral and poster presentations were accepted.",
        parent=policy_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The conference accepted both oral and poster presentations.",
        node=oral_poster_leaf,
        sources=_policy_sources(extraction),
        additional_instruction="Confirm presentation formats via program/CFP pages.",
    )

    # Includes workshop sessions
    workshops_leaf = evaluator.add_leaf(
        id="Includes_Workshop_Sessions",
        desc="Indicates workshop sessions were included in the program.",
        parent=policy_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The conference program included workshop sessions.",
        node=workshops_leaf,
        sources=_policy_sources(extraction),
        additional_instruction="Confirm via the official program or workshops page.",
    )

    # Student registration discounts offered
    student_disc_leaf = evaluator.add_leaf(
        id="Offers_Student_Registration_Discounts",
        desc="Indicates student registration discounts were offered.",
        parent=policy_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Student registration discounts were offered.",
        node=student_disc_leaf,
        sources=_registration_sources(extraction),
        additional_instruction="Confirm via registration pages (fees/discounts).",
    )

    # Proceedings published or available
    proceedings_leaf = evaluator.add_leaf(
        id="Proceedings_Published_Or_Available",
        desc="Indicates conference proceedings were published or made available.",
        parent=policy_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Conference proceedings were published or made available (e.g., via PMLR or similar).",
        node=proceedings_leaf,
        sources=_proceedings_sources(extraction),
        additional_instruction="Confirm via proceedings/publications page.",
    )

    # Author registration required
    author_reg_leaf = evaluator.add_leaf(
        id="Author_Registration_Required",
        desc="Indicates at least one author per accepted paper was required to register.",
        parent=policy_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one author per accepted paper was required to register for the conference.",
        node=author_reg_leaf,
        sources=_registration_sources(extraction),
        additional_instruction="Confirm via registration policy or author instructions.",
    )

    # Record some computed info for transparency
    computed_days = _compute_duration_days(start_text, end_text)
    evaluator.add_custom_info(
        info={
            "conference_display_name": conf_display_name,
            "start_date_text": start_text,
            "end_date_text": end_text,
            "computed_duration_days_inclusive": computed_days,
            "venue_name": venue_name,
            "source_url_counts": {
                "all": len(extraction.sources.all_source_urls) if extraction.sources else 0,
                "official": len(extraction.sources.official_site_urls) if extraction.sources else 0,
                "ranking": len(extraction.sources.ranking_source_urls) if extraction.sources else 0,
                "review_policy": len(extraction.sources.review_policy_urls) if extraction.sources else 0,
                "submission_policy": len(extraction.sources.submission_policy_urls) if extraction.sources else 0,
                "registration": len(extraction.sources.registration_urls) if extraction.sources else 0,
                "proceedings": len(extraction.sources.proceedings_urls) if extraction.sources else 0,
            }
        },
        info_type="computed_info",
        info_name="conference_computed_info"
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the machine-learning conference in Vienna, July 2024 task.
    Returns a structured summary dict from the evaluator.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured conference info from the answer
    extraction: ConferenceExtraction = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_conference(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()