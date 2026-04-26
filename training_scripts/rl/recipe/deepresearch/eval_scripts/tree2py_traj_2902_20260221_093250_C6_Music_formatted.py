import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "music_festival_indio_2025_gaga"
TASK_DESCRIPTION = """
Identify the major U.S. music festival that takes place in April 2025 in Indio, California, and features Lady Gaga as one of its headliners. Provide comprehensive details including: (1) the festival name, (2) the exact dates for both weekends of the festival, (3) the names of all three co-headlining acts performing alongside Lady Gaga, and (4) the name of the venue where the festival is held. Include URL references to support each piece of information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FestivalExtraction(BaseModel):
    # Core fields
    festival_name: Optional[str] = None
    weekend1_dates: Optional[str] = None
    weekend2_dates: Optional[str] = None
    co_headliners: List[str] = Field(default_factory=list)  # Exclude "Lady Gaga"
    venue_name: Optional[str] = None

    # Source URLs per field
    festival_name_sources: List[str] = Field(default_factory=list)
    weekend_dates_sources: List[str] = Field(default_factory=list)
    co_headliners_sources: List[str] = Field(default_factory=list)
    lady_gaga_sources: List[str] = Field(default_factory=list)
    venue_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_festival_details() -> str:
    return """
Extract the festival details as explicitly presented in the answer. Return a JSON with the following fields:

- festival_name: The name of the identified festival.
- weekend1_dates: The exact date range for weekend 1 as written in the answer (e.g., "April 11–13, 2025").
- weekend2_dates: The exact date range for weekend 2 as written in the answer (e.g., "April 18–20, 2025").
- co_headliners: An array with exactly three co-headlining acts performing alongside Lady Gaga. IMPORTANT: Do not include "Lady Gaga" in this list. If more than three names are given, return the three that are explicitly identified as co-headliners; otherwise, return the first three unique names excluding Lady Gaga.
- venue_name: The name of the venue where the festival is held.

Also extract URL citations explicitly present in the answer for each piece of information. For each URLs field below, include only valid, complete URLs (prepend http:// if missing):

- festival_name_sources: URLs supporting the festival name identification.
- weekend_dates_sources: URLs supporting both weekend date ranges (include all mentioned for dates).
- co_headliners_sources: URLs supporting the three co-headliners list.
- lady_gaga_sources: URLs showing that Lady Gaga is a headliner at this festival.
- venue_sources: URLs supporting the venue name.

Do not invent any information or URLs. If a requested field is missing in the answer, set it to null (or [] for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _is_lady_gaga(name: str) -> bool:
    n = _normalize_name(name)
    # Allow some common variants
    return "lady gaga" in n or n == "gaga"


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _dedup_urls(urls: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


RELIABLE_DOMAIN_KEYWORDS = [
    # Official / organizers / venue / ticketing
    "coachella.com",
    "goldenvoice.com",
    "empirepolo.com",
    "ticketmaster.com",
    "axs.com",
    "livenation.com",
    # Major reputable publications
    "billboard.com",
    "rollingstone.com",
    "variety.com",
    "pitchfork.com",
    "nytimes.com",
    "latimes.com",
    "theguardian.com",
    "guardian.com",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "consequence.net",
    "stereogum.com",
    "nme.com",
    "forbes.com",
    "time.com",
    "hollywoodreporter.com",
    "spin.com",
    "washingtonpost.com",
]


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _is_reliable_url(url: str) -> bool:
    dom = _domain_from_url(url)
    if not dom:
        return False
    return any(dom.endswith(k) or k in dom for k in RELIABLE_DOMAIN_KEYWORDS)


def _has_reliable_among(urls: List[str]) -> bool:
    return any(_is_reliable_url(u) for u in urls)


async def _verify_with_sources(
    evaluator: Evaluator,
    claim: str,
    node,
    sources: Optional[List[str]],
    additional_instruction: str,
    extra_prereqs: Optional[List[Any]] = None
) -> bool:
    # Enforce source-grounding: if no sources, mark failed without LLM verification
    srcs = _dedup_urls(sources or [])
    if len(srcs) == 0:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=srcs,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prereqs or []
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _build_task_tree(evaluator: Evaluator, extracted: FestivalExtraction) -> None:
    root = evaluator.root

    # Normalize and clean certain fields
    festival_name = (extracted.festival_name or "").strip()
    weekend1 = (extracted.weekend1_dates or "").strip()
    weekend2 = (extracted.weekend2_dates or "").strip()
    venue_name = (extracted.venue_name or "").strip()

    # Prepare co-headliners: exclude Lady Gaga, take first three unique
    co_heads_raw = [c.strip() for c in extracted.co_headliners if c and c.strip()]
    co_heads_filtered = [n for n in co_heads_raw if not _is_lady_gaga(n)]
    co_heads_unique = _dedup_preserve_order(co_heads_filtered)[:3]

    # Clean URLs
    festival_name_sources = _dedup_urls(extracted.festival_name_sources)
    weekend_dates_sources = _dedup_urls(extracted.weekend_dates_sources)
    co_headliners_sources = _dedup_urls(extracted.co_headliners_sources)
    lady_gaga_sources = _dedup_urls(extracted.lady_gaga_sources)
    venue_sources = _dedup_urls(extracted.venue_sources)

    # Add a top-level critical node as per rubric
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify the U.S. music festival matching all stated constraints and provide all requested details with reliable URL citations.",
        parent=root,
        critical=True
    )

    # -------------------- Required Details Provided (critical) --------------------
    provided_node = evaluator.add_parallel(
        id="Required_Details_Provided",
        desc="All requested output fields are provided.",
        parent=task_node,
        critical=True
    )

    # Festival name provided
    fest_name_provided_node = evaluator.add_custom_node(
        result=bool(festival_name),
        id="Festival_Name_Provided",
        desc="Festival name is provided.",
        parent=provided_node,
        critical=True
    )

    # Both weekends provided
    both_weekends_provided_node = evaluator.add_custom_node(
        result=bool(weekend1) and bool(weekend2),
        id="Both_Weekend_Dates_Provided",
        desc="Exact dates for both weekends are provided.",
        parent=provided_node,
        critical=True
    )

    # All three co-headliners provided (exclude Lady Gaga)
    all_three_co_heads_provided_node = evaluator.add_custom_node(
        result=len(co_heads_unique) >= 3,
        id="All_Three_Co_Headliners_Provided",
        desc="Names of all three co-headlining acts performing alongside Lady Gaga are provided.",
        parent=provided_node,
        critical=True
    )

    # Venue name provided
    venue_name_provided_node = evaluator.add_custom_node(
        result=bool(venue_name),
        id="Venue_Name_Provided",
        desc="Name of the venue where the festival is held is provided.",
        parent=provided_node,
        critical=True
    )

    # -------------------- Festival Matches All Constraints (critical) ------------
    constraints_node = evaluator.add_parallel(
        id="Festival_Matches_All_Constraints",
        desc="The identified festival satisfies every stated constraint.",
        parent=task_node,
        critical=True
    )

    # Timing: April 2025
    timing_leaf = evaluator.add_leaf(
        id="Timing_Check",
        desc="Festival takes place in April 2025.",
        parent=constraints_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim="The festival takes place in April 2025.",
        node=timing_leaf,
        sources=weekend_dates_sources,
        additional_instruction="Verify that the festival's scheduled dates are within April 2025.",
        extra_prereqs=[both_weekends_provided_node]
    )

    # Location: Indio, California
    location_leaf = evaluator.add_leaf(
        id="Location_Check",
        desc="Festival is located in Indio, California.",
        parent=constraints_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim="The festival is located in Indio, California.",
        node=location_leaf,
        sources=(venue_sources or festival_name_sources),
        additional_instruction="Confirm that the festival takes place in Indio, CA. Accept confirmation from official pages or reputable publications.",
        extra_prereqs=[fest_name_provided_node]
    )

    # Two weekends check
    two_weekends_leaf = evaluator.add_leaf(
        id="Two_Weekends_Check",
        desc="Festival takes place over two weekends.",
        parent=constraints_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim="The festival takes place over two weekends.",
        node=two_weekends_leaf,
        sources=weekend_dates_sources,
        additional_instruction="Look for language indicating two distinct weekends or a poster listing two weekend ranges.",
        extra_prereqs=[both_weekends_provided_node]
    )

    # Outdoor venue check
    outdoor_leaf = evaluator.add_leaf(
        id="Outdoor_Venue_Check",
        desc="Festival is held at an outdoor venue.",
        parent=constraints_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim=f"The venue '{venue_name}' is an outdoor venue.",
        node=outdoor_leaf,
        sources=venue_sources,
        additional_instruction="Verify from the venue/festival pages or reputable sources that this is an outdoor venue (e.g., outdoor grounds, polo club, open-air site).",
        extra_prereqs=[venue_name_provided_node]
    )

    # Lady Gaga headliner check
    gaga_leaf = evaluator.add_leaf(
        id="Lady_Gaga_Headliner_Check",
        desc="Lady Gaga is one of the festival headliners.",
        parent=constraints_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim="Lady Gaga is a headliner at this festival.",
        node=gaga_leaf,
        sources=(lady_gaga_sources or (festival_name_sources + co_headliners_sources)),
        additional_instruction="Verify a lineup announcement, poster, or reputable news indicating Lady Gaga is one of the headliners.",
        extra_prereqs=[fest_name_provided_node]
    )

    # Four total headliners including Lady Gaga
    four_headliners_leaf = evaluator.add_leaf(
        id="Four_Total_Headliners_Check",
        desc="Festival has a total of four headlining acts including Lady Gaga.",
        parent=constraints_node,
        critical=True
    )
    # Construct claim using the three co-headliners + Lady Gaga
    if len(co_heads_unique) >= 3:
        co1, co2, co3 = co_heads_unique[0], co_heads_unique[1], co_heads_unique[2]
        claim_four = f"The festival has four headliners: Lady Gaga, {co1}, {co2}, and {co3}."
    else:
        claim_four = "The festival has four headliners including Lady Gaga."
    await _verify_with_sources(
        evaluator,
        claim=claim_four,
        node=four_headliners_leaf,
        sources=(co_headliners_sources or (lady_gaga_sources + festival_name_sources)),
        additional_instruction="Confirm that the lineup lists exactly four headliners: Lady Gaga plus three others.",
        extra_prereqs=[all_three_co_heads_provided_node, fest_name_provided_node]
    )

    # -------------------- Citations & Reliability (critical) ---------------------
    cite_node = evaluator.add_parallel(
        id="Citations_And_Reliability",
        desc="All provided information is supported by URL references from reliable sources.",
        parent=task_node,
        critical=True
    )

    # Festival name has citation (and support)
    fest_name_cite_leaf = evaluator.add_leaf(
        id="Festival_Name_Has_Citation",
        desc="Festival name has at least one supporting URL reference.",
        parent=cite_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim=f"The festival is named '{festival_name}'.",
        node=fest_name_cite_leaf,
        sources=festival_name_sources,
        additional_instruction="Verify the stated festival name on the cited page(s).",
        extra_prereqs=[fest_name_provided_node]
    )

    # Weekend dates have citation (and support)
    dates_cite_leaf = evaluator.add_leaf(
        id="Weekend_Dates_Have_Citation",
        desc="Both weekend date ranges have at least one supporting URL reference.",
        parent=cite_node,
        critical=True
    )
    if weekend1 and weekend2:
        claim_dates = f"The festival takes place on {weekend1} and {weekend2}."
    else:
        claim_dates = "The festival takes place across two weekends with the specified dates."
    await _verify_with_sources(
        evaluator,
        claim=claim_dates,
        node=dates_cite_leaf,
        sources=weekend_dates_sources,
        additional_instruction="Confirm both weekend date ranges exactly as stated.",
        extra_prereqs=[both_weekends_provided_node]
    )

    # Co-headliners have citation (and support)
    co_heads_cite_leaf = evaluator.add_leaf(
        id="Co_Headliners_Have_Citation",
        desc="All three co-headliner names have at least one supporting URL reference.",
        parent=cite_node,
        critical=True
    )
    if len(co_heads_unique) >= 3:
        claim_co = f"The three co-headliners performing alongside Lady Gaga are {co_heads_unique[0]}, {co_heads_unique[1]}, and {co_heads_unique[2]}."
    else:
        claim_co = "The three co-headliners performing alongside Lady Gaga are correctly listed."
    await _verify_with_sources(
        evaluator,
        claim=claim_co,
        node=co_heads_cite_leaf,
        sources=co_headliners_sources,
        additional_instruction="Verify that the three named co-headliners are indeed listed as headliners alongside Lady Gaga.",
        extra_prereqs=[all_three_co_heads_provided_node]
    )

    # Venue name has citation (and support)
    venue_cite_leaf = evaluator.add_leaf(
        id="Venue_Name_Has_Citation",
        desc="Venue name has at least one supporting URL reference.",
        parent=cite_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        claim=f"The festival is held at '{venue_name}'.",
        node=venue_cite_leaf,
        sources=venue_sources,
        additional_instruction="Verify the venue name on official festival/venue pages or reputable publications.",
        extra_prereqs=[venue_name_provided_node]
    )

    # Reliable sources check (domain-based heuristic)
    reliability_ok = (
        _has_reliable_among(festival_name_sources) and
        _has_reliable_among(weekend_dates_sources) and
        _has_reliable_among(co_headliners_sources) and
        _has_reliable_among(venue_sources)
    )
    evaluator.add_custom_node(
        result=reliability_ok,
        id="Reliable_Sources_Check",
        desc="Provided URLs are from reliable sources (e.g., official festival/venue pages or reputable news/industry publications).",
        parent=cite_node,
        critical=True
    )

    # Record some custom info for debugging / transparency
    evaluator.add_custom_info(
        info={
            "festival_name": festival_name,
            "weekend1_dates": weekend1,
            "weekend2_dates": weekend2,
            "co_headliners": co_heads_unique,
            "venue_name": venue_name,
            "festival_name_sources": festival_name_sources,
            "weekend_dates_sources": weekend_dates_sources,
            "co_headliners_sources": co_headliners_sources,
            "lady_gaga_sources": lady_gaga_sources,
            "venue_sources": venue_sources,
            "reliable_sources_check": reliability_ok
        },
        info_type="extracted_details",
        info_name="extracted_details_overview"
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
    Evaluate an answer for the April 2025 Indio festival with Lady Gaga headlining.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured festival details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_festival_details(),
        template_class=FestivalExtraction,
        extraction_name="festival_details"
    )

    # Build verification tree and run checks
    await _build_task_tree(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()