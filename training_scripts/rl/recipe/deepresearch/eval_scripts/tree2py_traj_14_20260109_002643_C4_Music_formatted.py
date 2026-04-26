import asyncio
import logging
import re
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "billboard_top_venues_5001_10000_2025"
TASK_DESCRIPTION = (
    "Which venue ranked No.1 in Billboard's Top Venues chart for the 5,001-10,000 capacity category "
    "based on shows that took place between October 1, 2024, and September 30, 2025? Provide the venue's name, "
    "location (city and state/region), seating capacity, information about its ADA accessibility services, "
    "and a reference URL supporting your answer."
)

RANKING_PERIOD_PLAIN = "between October 1, 2024 and September 30, 2025"
RANKING_CATEGORY = "5,001-10,000"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state_region: Optional[str] = None
    seating_capacity: Optional[str] = None
    accessibility_summary: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_details() -> str:
    return (
        "Extract the key details provided in the answer about the requested Billboard Top Venues ranking.\n"
        "Return a JSON object with the following fields:\n"
        "- venue_name: The name of the venue identified as No.1 in the relevant Billboard Top Venues chart.\n"
        "- city: The venue's city as stated in the answer.\n"
        "- state_region: The venue's state, province, or region as stated in the answer.\n"
        "- seating_capacity: The seating capacity number or range exactly as presented in the answer.\n"
        "- accessibility_summary: A short snippet (1–2 sentences or key phrases) summarizing what the answer says "
        "about the venue's ADA accessibility (e.g., accessible seating, entrances, services). If nothing provided, return null.\n"
        "- reference_urls: An array of all URLs explicitly cited in the answer that support the venue identification and/or "
        "the Billboard ranking claim. These should be actual URLs present in the answer text. Do not invent URLs.\n"
        "- accessibility_urls: An array of URLs (if any) explicitly cited in the answer that describe the venue's accessibility/ADA services.\n"
        "- other_urls: Any additional venue-related URLs explicitly present in the answer not included above.\n\n"
        "Rules:\n"
        "1) Only extract values explicitly present in the answer; if missing, use null (for strings) or an empty array (for URL lists).\n"
        "2) For URL fields, include full valid URLs. If a URL is missing a protocol, prepend http://.\n"
        "3) Do not deduplicate; include all as listed. The order should follow the answer's order."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_http_url(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.match(r"^https?://", s.strip()))


def dedup_urls_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def collect_supported_urls(extracted: VenueExtraction) -> List[str]:
    all_urls = []
    if extracted.reference_urls:
        all_urls.extend(extracted.reference_urls)
    if extracted.accessibility_urls:
        all_urls.extend(extracted.accessibility_urls)
    if extracted.other_urls:
        all_urls.extend(extracted.other_urls)
    all_urls = [u for u in all_urls if is_http_url(u)]
    return dedup_urls_preserve_order(all_urls)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_venue(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    # Build a critical parallel node that corresponds to the rubric root
    task_node = evaluator.add_parallel(
        id="all_criteria",
        desc="Evaluate the identified venue against all required criteria",
        parent=parent_node,
        critical=True
    )

    # 1) Venue identification (existence check)
    venue_ident_node = evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="venue_identification",
        desc="The venue is identified by name",
        parent=task_node,
        critical=True
    )

    # 2) Reference URL existence (existence + basic validity)
    valid_ref_urls = [u for u in extracted.reference_urls if is_http_url(u)]
    ref_url_exists_node = evaluator.add_custom_node(
        result=len(valid_ref_urls) > 0,
        id="reference_url",
        desc="A valid reference URL supporting the venue identification and ranking is provided",
        parent=task_node,
        critical=True
    )

    # 3) Ranking verification (URL-supported)
    ranking_node = evaluator.add_leaf(
        id="ranking_verification",
        desc="The venue ranked No.1 in Billboard's Top Venues chart for the 5,001-10,000 capacity category (data from Oct 1, 2024 to Sept 30, 2025)",
        parent=task_node,
        critical=True
    )
    vname = extracted.venue_name or ""
    ranking_claim = (
        f"The venue '{vname}' ranked No. 1 in Billboard's Top Venues chart for the {RANKING_CATEGORY} capacity category "
        f"based on shows that took place {RANKING_PERIOD_PLAIN}."
    )
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_node,
        sources=valid_ref_urls if len(valid_ref_urls) > 0 else None,
        additional_instruction=(
            "Verify that the provided URL(s) explicitly support this exact claim. The claim is only correct if the page shows "
            "the venue at position No.1 within the 5,001-10,000 capacity category for the specified Boxscore year period "
            "(Oct 1, 2024 to Sept 30, 2025). Accept reasonable formatting variations like '5001-10000' or '5k-10k'. "
            "If no valid URL is provided or the page does not clearly show this, conclude it is not supported."
        )
    )

    # 4) Location verification (city + state/region) against any provided URLs
    location_node = evaluator.add_leaf(
        id="location_verification",
        desc="The venue's city and state/region are correctly identified",
        parent=task_node,
        critical=True
    )
    city = extracted.city or ""
    state_region = extracted.state_region or ""
    location_claim = f"The venue '{vname}' is located in {city}, {state_region}."
    combined_urls = collect_supported_urls(extracted)
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=combined_urls if len(combined_urls) > 0 else None,
        additional_instruction=(
            "Check the page(s) for the venue's location; allow common variants such as state abbreviations vs full names "
            "(e.g., 'CA' vs 'California'), and minor punctuation differences in city names (e.g., 'St.' vs 'Saint'). "
            "If no provided URL mentions the location, or if the claim conflicts with the page(s), mark as not supported."
        )
    )

    # 5) Capacity specification (existence check only per rubric)
    capacity_node = evaluator.add_custom_node(
        result=bool(extracted.seating_capacity and extracted.seating_capacity.strip()),
        id="capacity_specification",
        desc="The venue's seating capacity is provided",
        parent=task_node,
        critical=True
    )

    # 6) Accessibility services (verify presence of ADA services information)
    accessibility_node = evaluator.add_leaf(
        id="accessibility_services",
        desc="The venue provides ADA-compliant accessibility services and seating",
        parent=task_node,
        critical=True
    )
    accessibility_claim = (
        f"The venue '{vname}' provides ADA-compliant accessibility services and accessible seating for guests with disabilities."
    )
    # Prefer specific accessibility URLs; fall back to any other provided sources
    acc_urls = extracted.accessibility_urls if len(extracted.accessibility_urls) > 0 else combined_urls
    acc_urls = [u for u in acc_urls if is_http_url(u)]
    await evaluator.verify(
        claim=accessibility_claim,
        node=accessibility_node,
        sources=acc_urls if len(acc_urls) > 0 else None,
        additional_instruction=(
            "Confirm that the page explicitly mentions ADA or accessibility accommodations (e.g., accessible/ADA seating, "
            "wheelchair access, accessible entrances/elevators, companion seating, assistive services). "
            "If none of the provided URLs mention accessibility, mark as not supported."
        )
    )

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "venue_name": extracted.venue_name,
            "city": extracted.city,
            "state_region": extracted.state_region,
            "seating_capacity": extracted.seating_capacity,
            "accessibility_summary": extracted.accessibility_summary,
            "reference_urls": extracted.reference_urls,
            "accessibility_urls": extracted.accessibility_urls,
            "other_urls": extracted.other_urls,
            "all_urls_considered": combined_urls
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info"
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
    # Initialize evaluator with a wrapper root (framework root is always non-critical)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_details(),
        template_class=VenueExtraction,
        extraction_name="venue_details"
    )

    # Build verification tree and run verifications
    await verify_venue(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()