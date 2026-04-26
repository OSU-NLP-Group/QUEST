import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "free_us_astronomy_facilities"
TASK_DESCRIPTION = (
    "Identify four space or astronomy facilities in the United States that are open to public visitors and offer free admission. "
    "For each facility, provide: 1) name and location (city and state), 2) current operating hours or open days, "
    "3) a description of the public telescope viewing or public tour program, and 4) an official website link. "
    "All four facilities must offer free admission and have an active program available to general visitors."
)

ORDINAL_DESCS = [
    "First space or astronomy facility meeting all requirements",
    "Second space or astronomy facility meeting all requirements",
    "Third space or astronomy facility meeting all requirements",
    "Fourth space or astronomy facility meeting all requirements",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    """One facility entry as extracted from the agent's answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Full name or 2-letter abbreviation
    hours: Optional[str] = None  # Any provided hours or open days statement
    public_program: Optional[str] = None  # Description of telescope viewing or public tour program
    website_url: Optional[str] = None  # Official website or visitor information page
    additional_urls: List[str] = Field(default_factory=list)  # Any extra URLs cited for this facility
    admission_free_note: Optional[str] = None  # Text snippet or claim about free admission, if present


class FacilitiesExtraction(BaseModel):
    """Top-level extraction of facilities list."""
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return (
        "From the answer, extract up to the first four space or astronomy facilities (U.S.-based) mentioned. "
        "For each facility, return the following fields:\n"
        "- name: The complete facility name.\n"
        "- city: The city name.\n"
        "- state: The U.S. state name or its two-letter abbreviation.\n"
        "- hours: The current operating hours or open days as stated in the answer.\n"
        "- public_program: A description in the answer of the public telescope viewing program or public tour program.\n"
        "- website_url: The official website URL or the official visitor information page URL (extract the actual URL string).\n"
        "- additional_urls: An array of any other URLs in the answer that relate to this facility.\n"
        "- admission_free_note: Any text snippet or statement in the answer indicating free admission/no entry fee.\n\n"
        "IMPORTANT:\n"
        "1) Extract only what is explicitly present in the answer; do not invent any fields.\n"
        "2) If a field is missing for a facility, set it to null (or empty array for additional_urls).\n"
        "3) Accept URLs in plain text or markdown link format, but extract the actual URL string.\n"
        "4) If more than four facilities are present, only include the first four.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(item: FacilityItem) -> List[str]:
    """Combine website_url and additional_urls into a single deduplicated list."""
    sources: List[str] = []
    if item.website_url and isinstance(item.website_url, str) and item.website_url.strip():
        sources.append(item.website_url.strip())
    for u in item.additional_urls:
        if isinstance(u, str) and u.strip():
            sources.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)
    return unique_sources


def _looks_like_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification logic per facility                                             #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    item: FacilityItem,
    idx: int,
) -> None:
    """
    Build and run verification checks for a single facility.

    Leaves inside the facility node are critical to ensure that failing any requirement
    causes the facility to fail (no partial credit within a single facility).
    """
    # Create a facility-level parallel node (non-critical to allow partial scoring across facilities)
    facility_node = evaluator.add_parallel(
        id=f"facility_{idx + 1}",
        desc=ORDINAL_DESCS[idx],
        parent=parent_node,
        critical=False,
    )

    # 1) Identification (existence check for name + city + state)
    name_ok = bool(item.name and item.name.strip())
    city_ok = bool(item.city and item.city.strip())
    state_ok = bool(item.state and item.state.strip())
    evaluator.add_custom_node(
        result=(name_ok and city_ok and state_ok),
        id=f"facility_{idx + 1}_identification",
        desc="Facility name and location (city and state) are provided",
        parent=facility_node,
        critical=True,
    )

    # Prepare common sources list for evidence-backed checks
    sources_list = _combine_sources(item)

    # 2) US location
    us_loc_leaf = evaluator.add_leaf(
        id=f"facility_{idx + 1}_us_location",
        desc="Facility is located in the United States",
        parent=facility_node,
        critical=True,
    )
    city = item.city or ""
    state = item.state or ""
    name = item.name or "the facility"
    us_loc_claim = (
        f"The facility '{name}' is located in {city}, {state} in the United States (USA). "
        f"If the website shows a U.S. city and state (e.g., 'City, CA'), that is sufficient to confirm U.S. location."
    )
    await evaluator.verify(
        claim=us_loc_claim,
        node=us_loc_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Confirm U.S. location using the official page content. "
            "If the page clearly states a U.S. city and state, count as supported even if 'United States' or 'USA' is not spelled out explicitly."
        ),
    )

    # 3) Admission is free
    free_leaf = evaluator.add_leaf(
        id=f"facility_{idx + 1}_admission",
        desc="Admission is free (no entry fee)",
        parent=facility_node,
        critical=True,
    )
    free_claim = (
        f"Admission to '{name}' is free (no entry fee required for general visitors). "
        f"Verify that the official site indicates free admission, no ticket cost, or equivalent phrasing."
    )
    await evaluator.verify(
        claim=free_claim,
        node=free_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Look for explicit phrases like 'free admission', 'no entry fee', or '$0' for entry. "
            "If the page lists an entry fee or tickets for general entry, mark as not supported. "
            "Fees for special events are acceptable only if base entry for general visitors is free."
        ),
    )

    # 4) Operating hours or open days specified
    hours_leaf = evaluator.add_leaf(
        id=f"facility_{idx + 1}_hours",
        desc="Current operating hours or days are specified",
        parent=facility_node,
        critical=True,
    )
    hours_claim = (
        f"The official website for '{name}' specifies operating hours or open days for public visitors."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Look for any clear statement of hours (e.g., 'Open daily 9am–5pm') or open days (e.g., 'Open Saturdays'). "
            "An events calendar or 'Public Night every Friday' counts as hours/open days if clearly indicating when visitors can attend."
        ),
    )

    # 5) Public telescope viewing OR public tour program available and described
    program_leaf = evaluator.add_leaf(
        id=f"facility_{idx + 1}_public_program",
        desc="Public telescope viewing or public tour program is available and described",
        parent=facility_node,
        critical=True,
    )
    program_claim = (
        f"'{name}' offers an active public telescope viewing program or public tour program available to general visitors, "
        f"as indicated by the official page (e.g., 'public night', 'star party', 'guided tours', or similar)."
    )
    await evaluator.verify(
        claim=program_claim,
        node=program_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Confirm a program open to the general public: examples include public telescope nights, star parties, scheduled tours, or open houses. "
            "Exclusive member-only programs or purely private/internal events do not satisfy this requirement."
        ),
    )

    # 6) Valid reference URL to official facility website
    reference_leaf = evaluator.add_leaf(
        id=f"facility_{idx + 1}_reference",
        desc="Valid reference URL to official facility website is provided",
        parent=facility_node,
        critical=True,
    )
    if _looks_like_url(item.website_url):
        ref_claim = (
            f"The provided URL is the official website or an official visitor information page for '{name}'."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=reference_leaf,
            sources=item.website_url,
            additional_instruction=(
                "Check that the domain and page content clearly belong to the facility (e.g., .edu, .gov, or the organization's official domain). "
                "Third-party listing pages (Yelp, TripAdvisor, general directories) are not official."
            ),
        )
    else:
        # No valid URL string in the answer: verify the presence requirement via answer context
        ref_claim = (
            f"The answer provides a valid official website URL for '{name}'."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=reference_leaf,
            sources=None,
            additional_instruction=(
                "If the answer does not include a valid URL for the official site or official visitor info page, mark this verification as incorrect."
            ),
        )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer against the rubric:
    Four U.S. space/astronomy facilities with free admission and an active public program, with required fields.
    """
    # Initialize evaluator (root is non-critical parallel to allow partial scoring across facilities)
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

    # Extract facilities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    # Prepare exactly four items (pad with empty placeholders if needed)
    facilities = list(extracted.facilities[:4])
    while len(facilities) < 4:
        facilities.append(FacilityItem())

    # Build facility subtrees
    verify_tasks = []
    for idx, item in enumerate(facilities):
        verify_tasks.append(verify_facility(evaluator, root, item, idx))

    # Run verifications (in parallel)
    await asyncio.gather(*verify_tasks, return_exceptions=True)

    # Optionally record custom info
    evaluator.add_custom_info(
        {"extracted_facility_count": len(extracted.facilities)},
        info_type="extraction_stats",
        info_name="facilities_count",
    )

    # Return summary
    return evaluator.get_summary()