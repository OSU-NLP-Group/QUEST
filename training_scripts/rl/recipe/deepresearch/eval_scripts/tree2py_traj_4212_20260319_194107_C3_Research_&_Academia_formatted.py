import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "march_2026_lunar_eclipse_na"
TASK_DESCRIPTION = (
    "I'm planning to observe the total lunar eclipse occurring in March 2026 from North America. "
    "Please research this eclipse and provide the following information: (1) the exact date of the eclipse, "
    "(2) the time of maximum eclipse in UTC, (3) the duration of totality in minutes, "
    "(4) the region of North America with the best viewing conditions, "
    "(5) a direct link to NASA's official visibility map or documentation for this eclipse, "
    "and (6) when the next total lunar eclipse will be visible from North America after this one."
)

# Optional ground-truth hints (for summary/debug visibility only; not used for hard checks)
GROUND_TRUTH_HINTS = {
    "expected_date": "March 3, 2026",
    "expected_max_utc": "≈11:33 UTC",
    "expected_totality_start_utc": "≈11:04 UTC",
    "expected_totality_end_utc": "≈12:03 UTC",
    "expected_totality_duration": "≈58–60 minutes",
    "expected_best_region": "western United States (within North America)",
    "expected_next_total_lunar_eclipse_year_from_NA": "2029",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EclipseExtraction(BaseModel):
    # Identity and timing
    eclipse_type: Optional[str] = None
    eclipse_date: Optional[str] = None  # e.g., "March 3, 2026"
    maximum_eclipse_utc: Optional[str] = None  # e.g., "11:33 UTC"
    totality_start_utc: Optional[str] = None   # e.g., "11:04 UTC"
    totality_end_utc: Optional[str] = None     # e.g., "12:03 UTC"
    totality_duration_minutes: Optional[str] = None  # e.g., "59 minutes", "58–60 minutes"

    # Viewing region (North America)
    best_region_north_america: Optional[str] = None  # e.g., "western United States", "western North America"

    # Sources
    nasa_official_urls: List[str] = Field(default_factory=list)  # NASA links for the 2026-03-03 eclipse
    next_eclipse_year: Optional[str] = None                      # e.g., "2029"
    next_eclipse_sources: List[str] = Field(default_factory=list)  # Prefer NASA links supporting the next eclipse info

    # Any other URLs present in the answer (non-NASA allowed)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract key details for the March 2026 lunar eclipse as they are explicitly stated in the answer text.

    REQUIRED FIELDS:
    - eclipse_type: The eclipse type stated (e.g., "total lunar eclipse"). Return null if not stated.
    - eclipse_date: The calendar date for the eclipse as given (e.g., "March 3, 2026"). Return null if not stated.
    - maximum_eclipse_utc: The time of maximum eclipse in UTC as presented (e.g., "11:33 UTC"). Keep the format as-is; do NOT convert. Return null if not stated.
    - totality_start_utc: The start of totality time in UTC as presented (e.g., "11:04 UTC"). Return null if not stated.
    - totality_end_utc: The end of totality time in UTC as presented (e.g., "12:03 UTC"). Return null if not stated.
    - totality_duration_minutes: The duration of totality in minutes as presented (e.g., "59 minutes" or "58–60 minutes"). Return null if not stated.
    - best_region_north_america: The answer's stated best viewing region within North America (e.g., "western United States"). Return null if not stated.

    NASA LINKS FOR THIS ECLIPSE:
    - nasa_official_urls: Collect all URLs in the answer that are direct NASA official pages (domain contains "nasa.gov") that specifically discuss or map the March 3, 2026 lunar eclipse. Return an array of URLs (can be empty if none).
    
    NEXT TOTAL LUNAR ECLIPSE INFO (after the 2026 event):
    - next_eclipse_year: The next total lunar eclipse year visible from North America (as stated in the answer), e.g., "2029". Return null if not stated.
    - next_eclipse_sources: URLs cited in the answer that support the next-eclipse claim (prefer NASA if present). Return an array (can be empty).

    OTHER URLS:
    - all_urls: Extract all URLs present in the answer (any domain, including NASA). This is a catch-all list. Return an array (can be empty).

    IMPORTANT RULES:
    - Extract only what is explicitly present in the answer. Do not invent or infer values.
    - Keep times in the same textual format as the answer shows.
    - For URLs, include full links. If a URL is embedded in markdown, extract the actual URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_nasa(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and "nasa.gov" in u.lower()]


def _first_non_empty(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if primary else fallback


def _safe_str(s: Optional[str], default: str = "") -> str:
    return s or default


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_event_identity(
    evaluator: Evaluator,
    parent,
    info: EclipseExtraction,
    nasa_sources_for_core: List[str],
) -> None:
    """
    Event Identity: type + date. Critical node.
    """
    node = evaluator.add_parallel(
        id="event_identity",
        desc="Correctly identify the eclipse event",
        parent=parent,
        critical=True,
    )

    # Eclipse Type (verify against NASA page if available)
    n_type = evaluator.add_leaf(
        id="eclipse_type",
        desc="States the event is a total lunar eclipse",
        parent=node,
        critical=True,
    )
    type_claim = (
        "The March 2026 lunar eclipse is a total lunar eclipse. "
        "If the page references the specific 2026 Mar 03 eclipse, it must indicate it is total."
    )
    sources_for_type = nasa_sources_for_core if nasa_sources_for_core else None
    await evaluator.verify(
        claim=type_claim,
        node=n_type,
        sources=sources_for_type,
        additional_instruction="Rely on the NASA page to confirm the eclipse type is total. "
                               "Treat 'total lunar eclipse' wording or unambiguous equivalent as confirming."
    )

    # Eclipse Date (verify the date the answer provided, with NASA if possible)
    n_date = evaluator.add_leaf(
        id="eclipse_date",
        desc="States the eclipse date is March 3, 2026",
        parent=node,
        critical=True,
    )
    date_text = _safe_str(info.eclipse_date, "March 3, 2026")
    date_claim = (
        f"The date of the eclipse is {date_text}. "
        f"Confirm this is the UTC calendar date for the total lunar eclipse."
    )
    sources_for_date = nasa_sources_for_core if nasa_sources_for_core else None
    await evaluator.verify(
        claim=date_claim,
        node=n_date,
        sources=sources_for_date,
        additional_instruction="Prefer NASA page confirmation. Accept reasonable formatting variants like '2026 Mar 03'."
    )


async def verify_timing_utc(
    evaluator: Evaluator,
    parent,
    info: EclipseExtraction,
    nasa_sources_for_core: List[str],
    any_sources: List[str],
) -> None:
    """
    UTC timing details. Critical node.
    """
    node = evaluator.add_parallel(
        id="eclipse_timing_utc",
        desc="Provide required UTC timing details",
        parent=parent,
        critical=True,
    )

    sources = nasa_sources_for_core if nasa_sources_for_core else any_sources if any_sources else None

    # Maximum Eclipse
    n_max = evaluator.add_leaf(
        id="maximum_eclipse_utc",
        desc="Gives the time of maximum eclipse as approximately 11:33 UTC",
        parent=node,
        critical=True,
    )
    max_text = _safe_str(info.maximum_eclipse_utc, "11:33 UTC")
    max_claim = (
        f"The time of maximum eclipse is approximately {max_text} (UTC). "
        f"Treat 'approximately' as allowing a ±5 minute tolerance."
    )
    await evaluator.verify(
        claim=max_claim,
        node=n_max,
        sources=sources,
        additional_instruction="Accept values within about ±5 minutes of 11:33 UTC (e.g., 11:32–11:34). "
                               "Allow format variants such as '11:33 UT' or with seconds."
    )

    # Totality Start
    n_start = evaluator.add_leaf(
        id="totality_start_utc",
        desc="Gives the start of totality as approximately 11:04 UTC",
        parent=node,
        critical=True,
    )
    start_text = _safe_str(info.totality_start_utc, "11:04 UTC")
    start_claim = (
        f"The start of totality is approximately {start_text} (UTC). "
        f"Treat 'approximately' as allowing a ±5 minute tolerance."
    )
    await evaluator.verify(
        claim=start_claim,
        node=n_start,
        sources=sources,
        additional_instruction="Accept values within about ±5 minutes of 11:04 UTC."
    )

    # Totality End
    n_end = evaluator.add_leaf(
        id="totality_end_utc",
        desc="Gives the end of totality as approximately 12:03 UTC",
        parent=node,
        critical=True,
    )
    end_text = _safe_str(info.totality_end_utc, "12:03 UTC")
    end_claim = (
        f"The end of totality is approximately {end_text} (UTC). "
        f"Treat 'approximately' as allowing a ±5 minute tolerance."
    )
    await evaluator.verify(
        claim=end_claim,
        node=n_end,
        sources=sources,
        additional_instruction="Accept values within about ±5 minutes of 12:03 UTC."
    )

    # Totality Duration
    n_dur = evaluator.add_leaf(
        id="totality_duration_minutes",
        desc="Gives the duration of totality as approximately 58–60 minutes",
        parent=node,
        critical=True,
    )
    dur_text = _safe_str(info.totality_duration_minutes, "58–60 minutes")
    dur_claim = (
        f"The duration of totality is approximately {dur_text}. "
        f"Treat 'approximately' as allowing 58 to 60 minutes inclusive."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=n_dur,
        sources=sources,
        additional_instruction="Consider 58, 59, or 60 minutes (or 'about one hour') as a correct duration."
    )


async def verify_viewing_conditions(
    evaluator: Evaluator,
    parent,
    info: EclipseExtraction,
    nasa_sources_for_core: List[str],
    any_sources: List[str],
) -> None:
    """
    Best viewing region in North America. Critical node.
    """
    node = evaluator.add_parallel(
        id="viewing_conditions",
        desc="Provide best-viewing region information for North America",
        parent=parent,
        critical=True,
    )

    n_region = evaluator.add_leaf(
        id="best_region_north_america",
        desc="Identifies the western United States as the best viewing region (within North America)",
        parent=node,
        critical=True,
    )

    region_text = _safe_str(info.best_region_north_america, "western United States")
    sources = nasa_sources_for_core if nasa_sources_for_core else any_sources if any_sources else None
    region_claim = (
        f"Within North America, the best viewing region for the March 3, 2026 total lunar eclipse is described as "
        f"'{region_text}', which corresponds to the western United States / western North America where totality is visible."
    )
    await evaluator.verify(
        claim=region_claim,
        node=n_region,
        sources=sources,
        additional_instruction=(
            "Use the NASA visibility map/text to judge whether the western United States (or western North America) "
            "is well-placed for totality and thus a 'best' region. Accept reasonable phrasings such as "
            "'western United States', 'western North America', or 'western U.S.'."
        ),
    )


async def verify_nasa_documentation_link(
    evaluator: Evaluator,
    parent,
    info: EclipseExtraction,
) -> None:
    """
    NASA official documentation link for this eclipse. Critical node.
    """
    node = evaluator.add_parallel(
        id="nasa_documentation_link",
        desc="Provide NASA official documentation/visibility map link for this eclipse",
        parent=parent,
        critical=True,
    )

    nasa_urls = _filter_nasa(info.nasa_official_urls) or _filter_nasa(info.all_urls)

    # Existence check (critical precondition to avoid accidental passing without URLs)
    evaluator.add_custom_node(
        result=len(nasa_urls) > 0,
        id="nasa_link_present",
        desc="At least one NASA official (nasa.gov) URL is provided for the March 3, 2026 eclipse",
        parent=node,
        critical=True,
    )

    n_url = evaluator.add_leaf(
        id="nasa_official_url",
        desc="Provides a direct official NASA URL (nasa.gov or a NASA subdomain) to visibility map/documentation for the March 3, 2026 total lunar eclipse",
        parent=node,
        critical=True,
    )

    if nasa_urls:
        url_claim = (
            "This is an official NASA (nasa.gov) page that provides a visibility map and/or documentation for the "
            "March 3, 2026 total lunar eclipse."
        )
        await evaluator.verify(
            claim=url_claim,
            node=n_url,
            sources=nasa_urls,
            additional_instruction="Verify that the page is on a nasa.gov domain and explicitly addresses the 2026-03-03 lunar eclipse."
        )
    else:
        # Force fail if no NASA URL provided
        n_url.score = 0.0
        n_url.status = "failed"


async def verify_next_total_lunar_eclipse(
    evaluator: Evaluator,
    parent,
    info: EclipseExtraction,
) -> None:
    """
    Next total lunar eclipse visible from North America after the 2026 event. Critical node.
    """
    node = evaluator.add_parallel(
        id="next_total_lunar_eclipse_from_north_america",
        desc="Provide the next total lunar eclipse visible from North America after this event",
        parent=parent,
        critical=True,
    )

    n_next = evaluator.add_leaf(
        id="next_eclipse_year",
        desc="States the next total lunar eclipse visible from North America after March 3, 2026 occurs in 2029",
        parent=node,
        critical=True,
    )

    next_year_text = _safe_str(info.next_eclipse_year, "2029")

    # Prefer NASA links from next_eclipse_sources; if absent, allow other NASA links from nasa_official_urls as fallback
    nasa_next_sources = _filter_nasa(info.next_eclipse_sources) or _filter_nasa(info.nasa_official_urls)
    sources = nasa_next_sources if nasa_next_sources else None

    next_claim = (
        f"The next total lunar eclipse visible from North America after March 3, 2026 occurs in {next_year_text}."
    )
    if sources:
        await evaluator.verify(
            claim=next_claim,
            node=n_next,
            sources=sources,
            additional_instruction=(
                "Confirm that at least one total lunar eclipse in 2029 is visible from North America. "
                "It is acceptable if the page lists a specific 2029 date (e.g., '2029 Jan xx') and indicates visibility in North America. "
                "Do not accept penumbral or partial eclipses; it must be total."
            ),
        )
    else:
        # No supporting NASA URL(s); fail the verification
        n_next.score = 0.0
        n_next.status = "failed"


async def verify_nasa_source_verifiability(
    evaluator: Evaluator,
    parent,
    info: EclipseExtraction,
) -> None:
    """
    Ensure that key claims are supported by NASA sources. Critical node.
    """
    node = evaluator.add_parallel(
        id="nasa_source_verifiability",
        desc="Key claims are supported by NASA official sources as required",
        parent=parent,
        critical=True,
    )

    nasa_core = _filter_nasa(info.nasa_official_urls) or _filter_nasa(info.all_urls)
    nasa_next = _filter_nasa(info.next_eclipse_sources) or nasa_core

    # Core details (date + UTC timing/duration)
    n_core = evaluator.add_leaf(
        id="nasa_source_supports_core_eclipse_details",
        desc="Cites at least one official NASA source (nasa.gov or NASA subdomain) that supports the provided eclipse date and UTC timing/duration claims",
        parent=node,
        critical=True,
    )
    if nasa_core:
        core_claim = (
            "This NASA page supports the core details for the March 3, 2026 total lunar eclipse, including the date "
            "and UTC timing such as maximum eclipse near 11:33 UTC and a totality duration of roughly one hour (≈58–60 minutes)."
        )
        await evaluator.verify(
            claim=core_claim,
            node=n_core,
            sources=nasa_core,
            additional_instruction="Look for NASA-provided tables, maps, or text indicating the UTC times (start of totality, greatest eclipse, end of totality) and totality duration."
        )
    else:
        n_core.score = 0.0
        n_core.status = "failed"

    # Next eclipse claim supported by NASA
    n_next = evaluator.add_leaf(
        id="nasa_source_supports_next_eclipse_claim",
        desc="Cites at least one official NASA source (nasa.gov or NASA subdomain) that supports the stated next-eclipse-from-North-America timing (2029)",
        parent=node,
        critical=True,
    )
    if nasa_next:
        next_claim = (
            "This NASA page supports that the next total lunar eclipse visible from North America after the March 3, 2026 event occurs in 2029."
        )
        await evaluator.verify(
            claim=next_claim,
            node=n_next,
            sources=nasa_next,
            additional_instruction="Accept if the NASA source shows a 2029 total lunar eclipse with visibility in North America."
        )
    else:
        n_next.score = 0.0
        n_next.status = "failed"


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
    Evaluate an answer for the March 2026 total lunar eclipse task using the obj_task_eval framework.
    Returns a standardized summary dictionary.
    """
    # Initialize evaluator with a root (non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root combines independent checks
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseExtraction,
        extraction_name="eclipse_extraction",
    )

    # Record GT hints for transparency
    evaluator.add_ground_truth({"hints": GROUND_TRUTH_HINTS}, gt_type="reference_hints")

    # Build a critical main task node to mirror the rubric's Root
    main = evaluator.add_parallel(
        id="root_task",
        desc="Provide required facts about the March 2026 total lunar eclipse (North America) and include NASA-source support as required",
        parent=root,
        critical=True,
    )

    # Choose NASA sources for core verification where possible
    nasa_core_sources = _filter_nasa(extracted.nasa_official_urls) or _filter_nasa(extracted.all_urls)
    any_sources = extracted.all_urls

    # Subtrees following the rubric
    await verify_event_identity(evaluator, main, extracted, nasa_core_sources)
    await verify_timing_utc(evaluator, main, extracted, nasa_core_sources, any_sources)
    await verify_viewing_conditions(evaluator, main, extracted, nasa_core_sources, any_sources)
    await verify_nasa_documentation_link(evaluator, main, extracted)
    await verify_next_total_lunar_eclipse(evaluator, main, extracted)
    await verify_nasa_source_verifiability(evaluator, main, extracted)

    # Optional: add quick custom info snapshot
    evaluator.add_custom_info(
        info={
            "nasa_official_urls_extracted": extracted.nasa_official_urls,
            "next_eclipse_sources_extracted": extracted.next_eclipse_sources,
            "all_urls_extracted_count": len(extracted.all_urls or []),
        },
        info_type="extraction_stats",
        info_name="extraction_stats_summary",
    )

    return evaluator.get_summary()