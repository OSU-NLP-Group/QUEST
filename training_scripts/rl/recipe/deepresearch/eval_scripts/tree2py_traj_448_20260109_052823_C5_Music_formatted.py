import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "historic_music_venues"
TASK_DESCRIPTION = """
Identify four historic music venues in the United States that meet all of the following criteria:

1. The venue must be listed on the National Register of Historic Places.
2. The venue must be primarily designed and used for music performances or concerts.
3. The venue must have a seating capacity of at least 2,000.
4. The venue must be currently operational and hosting events.
5. Each of the four venues must be located in a different U.S. state.

For each venue, provide:
- The official venue name
- The U.S. state where it is located
- Confirmation that it is listed on the National Register of Historic Places
- The seating capacity
- Confirmation that it is currently operational
- The year it was added to the National Register of Historic Places
- A reference URL that supports this information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """Information for a single venue extracted from the answer."""
    official_name: Optional[str] = None
    state: Optional[str] = None
    nrhp_listed_confirmation: Optional[str] = None  # e.g., "Listed on NRHP", "Yes", etc.
    primary_function: Optional[str] = None          # textual confirmation/description of music venue usage
    capacity: Optional[str] = None                  # prefer strings; may include ranges or words like "about 2,100"
    operational_status_confirmation: Optional[str] = None  # e.g., "Operational", "Currently hosting events"
    designation_year: Optional[str] = None          # year string like "1998"; keep as string to be permissive
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """List of venues found in the answer."""
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four historic music venues described in the answer that are intended to meet the task criteria.
    For each venue, return a JSON object with the following fields:
    - official_name: The official venue name as stated in the answer (string).
    - state: The U.S. state where the venue is located (string, e.g., "New York", "California").
    - nrhp_listed_confirmation: A textual confirmation that the venue is listed on the National Register of Historic Places (string). If not explicitly stated, return null.
    - primary_function: A textual confirmation or description that the venue is primarily designed and used for music performances or concerts (string). If not clearly stated, return null.
    - capacity: The seating capacity as stated (string; can include ranges or approximations, e.g., "about 2,100"). If not stated, return null.
    - operational_status_confirmation: A textual confirmation the venue is currently operational and hosting events (string). If not clearly stated, return null.
    - designation_year: The year the venue was added to the National Register of Historic Places (string). If not stated, return null.
    - reference_urls: An array of URLs explicitly mentioned in the answer that support the venue’s information (including NRHP listing/NRHP year, capacity, and operational status). If none are provided, return an empty array.

    Return the results as:
    {
      "venues": [
        { ... up to 4 venues ... }
      ]
    }

    Rules:
    - Only extract information explicitly present in the answer text.
    - Extract up to the first four venues if more are provided; if fewer than four are provided, return as many as available.
    - For URLs, include only valid URLs explicitly provided in the answer (including markdown links). Do not infer or create URLs.
    - Use full URLs; if a URL is missing protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: List[str]) -> List[str]:
    """Filter to potentially valid URLs and deduplicate."""
    seen = set()
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        url = u.strip()
        if not url:
            continue
        # Prepend protocol if missing (simple heuristic)
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        # Basic validity heuristic
        if "." in url and len(url) > 8:
            if url not in seen:
                cleaned.append(url)
                seen.add(url)
    return cleaned


def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    root_parent,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single venue and execute checks in a robust order
    so that downstream verifications can rely on earlier critical preconditions.
    """
    idx = index + 1
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx}",
        desc=f"Venue {idx}: qualifying historic music venue with all required attributes",
        parent=root_parent,
        critical=False  # Allow partial credit per venue
    )

    # Normalize/clean URLs once
    sources_list = _clean_urls(venue.reference_urls or [])

    # 1) Identification: official name + state must be provided (critical)
    id_exists = _nonempty_str(venue.official_name) and _nonempty_str(venue.state)
    evaluator.add_custom_node(
        result=id_exists,
        id=f"venue_{idx}_identification",
        desc="Provides the official venue name and the U.S. state where it is located",
        parent=venue_node,
        critical=True
    )

    # 2) Reference URL exists (critical) — at least one valid URL must be provided
    ref_exists = len(sources_list) > 0
    ref_node = evaluator.add_custom_node(
        result=ref_exists,
        id=f"venue_{idx}_reference",
        desc=("Provides at least one valid reference URL that supports the required venue information "
              "(including NRHP listing and NRHP listing year, capacity, and operational status)"),
        parent=venue_node,
        critical=True
    )

    # 3) NRHP listing confirmation (critical, verified via sources)
    nrhp_node = evaluator.add_leaf(
        id=f"venue_{idx}_national_register",
        desc="Confirms the venue is listed on the National Register of Historic Places",
        parent=venue_node,
        critical=True
    )
    nrhp_claim = f"The venue '{venue.official_name or ''}' is listed on the National Register of Historic Places."
    await evaluator.verify(
        claim=nrhp_claim,
        node=nrhp_node,
        sources=sources_list,
        additional_instruction=(
            "Verify explicitly that the venue is on the National Register of Historic Places (NRHP). "
            "Accepted evidence includes phrases like 'National Register of Historic Places', 'NRHP', 'NRHP No.', "
            "official NRHP listing pages, or reliable sources clearly stating the listing."
        )
    )

    # 4) Primary function as music venue (critical, verified via sources)
    primary_node = evaluator.add_leaf(
        id=f"venue_{idx}_primary_function",
        desc="Confirms the venue is primarily designed and used for music performances or concerts",
        parent=venue_node,
        critical=True
    )
    primary_claim = (
        f"The venue '{venue.official_name or ''}' is primarily designed and used for music performances or concerts."
    )
    await evaluator.verify(
        claim=primary_claim,
        node=primary_node,
        sources=sources_list,
        additional_instruction=(
            "The source should indicate the venue is intended for music performances/concerts as its primary use. "
            "Descriptions such as 'concert hall', 'music venue', 'performance venue where concerts are regularly held' "
            "are acceptable. Mixed-use facilities are acceptable if the music/concert function is primary."
        )
    )

    # 5) Capacity checks (critical group)
    capacity_group = evaluator.add_parallel(
        id=f"venue_{idx}_capacity",
        desc="Provides seating capacity and confirms it is at least 2,000",
        parent=venue_node,
        critical=True
    )

    # 5a) Capacity provided (critical)
    cap_provided = evaluator.add_custom_node(
        result=_nonempty_str(venue.capacity),
        id=f"venue_{idx}_capacity_provided",
        desc="Seating capacity value is provided in the answer",
        parent=capacity_group,
        critical=True
    )

    # 5b) Capacity threshold verification (critical, verified via sources)
    cap_thresh_node = evaluator.add_leaf(
        id=f"venue_{idx}_capacity_at_least_2000",
        desc="Seating capacity is at least 2,000",
        parent=capacity_group,
        critical=True
    )
    cap_thresh_claim = (
        f"The seating capacity of the venue '{venue.official_name or ''}' is at least 2,000."
    )
    await evaluator.verify(
        claim=cap_thresh_claim,
        node=cap_thresh_node,
        sources=sources_list,
        additional_instruction=(
            "Check the venue's seated capacity reported on the page. Approximations like 'about 2,000' or "
            "'~2,100' should be accepted. If multiple capacities are listed (e.g., standing vs. seated), "
            "prefer seated capacity or the commonly reported capacity."
        )
    )

    # 6) Operational status (critical, verified via sources)
    operational_node = evaluator.add_leaf(
        id=f"venue_{idx}_operational",
        desc="Confirms the venue is currently operational and hosting events",
        parent=venue_node,
        critical=True
    )
    operational_claim = (
        f"The venue '{venue.official_name or ''}' is currently operational and hosting events."
    )
    await evaluator.verify(
        claim=operational_claim,
        node=operational_node,
        sources=sources_list,
        additional_instruction=(
            "Look for explicit evidence of the venue currently operating (e.g., 'open', 'operational', "
            "recent/upcoming events calendar, ticketing, or schedule pages). If the venue is permanently closed "
            "or not hosting events, mark as not supported."
        )
    )

    # 7) NRHP designation year (critical, verified via sources)
    year_node = evaluator.add_leaf(
        id=f"venue_{idx}_designation_year",
        desc="Provides the year the venue was added to the National Register of Historic Places (verifiable)",
        parent=venue_node,
        critical=True
    )
    year_text = venue.designation_year or ""
    year_claim = (
        f"The venue '{venue.official_name or ''}' was added to the National Register of Historic Places in {year_text}."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_node,
        sources=sources_list,
        additional_instruction=(
            "Verify the NRHP listing year (the year added to NRHP). Distinguish from build/renovation years. "
            "Accept synonyms like 'Added to NRHP', 'NRHP listing year', or authoritative registry references."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the historic music venues task.

    Returns:
        A standard summary dictionary produced by Evaluator.get_summary().
    """
    # Initialize evaluator (root non-critical to allow partial credit; distinct states leaf is critical)
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

    # Extract structured venues information
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Prepare the four venues (truncate or pad to exactly four)
    venues_list = list(extraction.venues or [])
    if len(venues_list) > 4:
        venues_list = venues_list[:4]
    while len(venues_list) < 4:
        venues_list.append(VenueItem())

    # Add a quick custom info block for debugging
    evaluator.add_custom_info(
        info={"extracted_count": len(extraction.venues or []), "used_count": len(venues_list)},
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build and run verifications for each venue
    for i, v in enumerate(venues_list):
        await verify_single_venue(evaluator, root, v, i)

    # Global check: distinct states across the four venues (critical at root level)
    distinct_states = { (v.state or "").strip() for v in venues_list if _nonempty_str(v.state) }
    all_states_provided = all(_nonempty_str(v.state) for v in venues_list)
    distinct_ok = all_states_provided and len(distinct_states) == 4

    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_states_across_venues",
        desc="Confirms the four venues are located in four different U.S. states (no state repeats across the set of four)",
        parent=root,
        critical=True
    )

    # Optionally record the set of states for transparency
    evaluator.add_custom_info(
        info={"states": [v.state or None for v in venues_list], "unique_states_count": len(distinct_states)},
        info_type="states_summary",
        info_name="venues_states"
    )

    # Return final structured summary
    return evaluator.get_summary()