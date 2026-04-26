import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_family_venues"
TASK_DESCRIPTION = (
    "Identify three different family-friendly entertainment venues in California that each meet all of the "
    "following requirements: (1) The venue must be located in California, (2) The venue must have a capacity of at least "
    "500 guests, (3) The venue must offer free admission for children under age 3, (4) The venue must provide group "
    "discount tickets for groups of 10 or more people, (5) The venue must offer either an annual membership or season "
    "pass option, (6) The venue must be ADA compliant with wheelchair accessible seating, and (7) The venue must offer at least "
    "two different admission price tiers based on age (such as child and adult pricing). For each venue, provide the venue name, "
    "location, and URL references confirming each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Basic identification
    name: Optional[str] = None
    location: Optional[str] = None
    # If a main website URL is given in the answer, capture it for potential verification
    official_url: Optional[str] = None

    # Requirement-specific URLs cited in the answer (can repeat the same URL if it supports multiple requirements)
    location_urls: List[str] = Field(default_factory=list)            # Evidence for being in California
    capacity_urls: List[str] = Field(default_factory=list)            # Evidence for capacity >= 500
    free_under_3_urls: List[str] = Field(default_factory=list)        # Evidence for free admission under age 3
    group_discount_urls: List[str] = Field(default_factory=list)      # Evidence for group discounts for 10+
    membership_or_pass_urls: List[str] = Field(default_factory=list)  # Evidence for annual membership or season pass
    ada_urls: List[str] = Field(default_factory=list)                 # Evidence for ADA + wheelchair accessible seating
    age_tiers_urls: List[str] = Field(default_factory=list)           # Evidence for at least two age-based pricing tiers


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first three family-friendly entertainment venues presented in the answer that claim to meet the task constraints.
    For each venue, extract the following fields exactly as they appear in the answer:

    Required fields per venue:
    - name: The venue name (string; return null if not present)
    - location: The stated location string (e.g., "Los Angeles, CA" or "San Diego, California"; return null if not present)
    - official_url: A main/official website URL for the venue if the answer explicitly provides one (return null if not provided)

    For each requirement, extract the URLs (if any) that the answer provides as citations specifically supporting that requirement:
    - location_urls: URLs that support that the venue is located in California (allow city/state pages on the official site or trusted pages)
    - capacity_urls: URLs indicating the seating/guest capacity (we only need the URLs; do not infer numbers)
    - free_under_3_urls: URLs showing that children under age 3 are admitted free
    - group_discount_urls: URLs showing group discount tickets for groups of 10 or more
    - membership_or_pass_urls: URLs showing annual membership or a season pass option
    - ada_urls: URLs showing ADA compliance and wheelchair accessible seating
    - age_tiers_urls: URLs showing at least two admission price tiers based on age (e.g., child vs adult pricing)

    Rules:
    1) Only extract URLs that explicitly appear in the answer. Do not invent or infer any URLs.
    2) If a single URL supports multiple requirements, include that URL in each relevant URL list.
    3) If the answer provides a general sources list for a venue, include those URLs where appropriate across the requirement lists if they evidently support the requirement(s).
    4) If no URL is provided for a particular requirement, return an empty list for that requirement.
    5) Preserve the order of venues as they appear in the answer and only return the first three venues.

    Return a JSON object with a single key 'venues' that is an array of up to three venue objects following the above schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    if not s:
        return ""
    s2 = s.strip().lower()
    # remove punctuation and collapse whitespace
    s2 = re.sub(r"[^\w\s]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        key = u.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(u)
    return result


def _with_fallback_urls(primary: List[str], fallback_single: Optional[str]) -> List[str]:
    if primary and len(primary) > 0:
        return _dedup_urls(primary)
    if fallback_single:
        return [fallback_single]
    return []


def _venues_are_distinct(venues: List[VenueItem]) -> bool:
    # Require three non-empty names and that (name, location) pairs are unique after normalization.
    if len(venues) < 3:
        return False
    pairs = []
    for v in venues[:3]:
        if not v or not v.name:
            return False
        name_n = _normalize_str(v.name)
        loc_n = _normalize_str(v.location)
        pairs.append((name_n, loc_n))
    return len(set(pairs)) == 3


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_requirement_by_urls(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    critical: bool = True,
    additional_instruction: str = "None",
):
    """
    Add a single leaf verification node. If urls is empty, fail the node immediately since the rubric requires citations.
    Otherwise, verify the claim against the provided URLs.
    """
    if urls:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=additional_instruction
        )
        return leaf
    else:
        # No citations provided -> automatically fail this "with citation" requirement
        return evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (failed: no citation URLs provided in the answer)",
            parent=parent_node,
            critical=critical
        )


async def _verify_single_venue(evaluator: Evaluator, root_parent, venue: VenueItem, idx: int):
    """
    Build the Venue_i subtree and verify each requirement for a single venue.
    """
    vid = idx + 1
    venue_node = evaluator.add_parallel(
        id=f"Venue_{vid}",
        desc=f"Evaluation of the {vid}{'st' if vid==1 else ('nd' if vid==2 else 'rd')} venue against all requirements.",
        parent=root_parent,
        critical=False
    )

    # 1) Name provided (critical gate)
    name_ok = venue.name is not None and venue.name.strip() != ""
    evaluator.add_custom_node(
        result=name_ok,
        id=f"V{vid}_Name_Provided",
        desc="Provide the venue name.",
        parent=venue_node,
        critical=True
    )

    # Prepare common data
    venue_name = venue.name or f"Venue {vid}"
    # For each requirement, use requirement-specific URLs; if empty, fall back to official_url if present
    loc_urls = _with_fallback_urls(venue.location_urls, venue.official_url)
    cap_urls = _with_fallback_urls(venue.capacity_urls, venue.official_url)
    free_urls = _with_fallback_urls(venue.free_under_3_urls, venue.official_url)
    group_urls = _with_fallback_urls(venue.group_discount_urls, venue.official_url)
    memb_urls = _with_fallback_urls(venue.membership_or_pass_urls, venue.official_url)
    ada_urls = _with_fallback_urls(venue.ada_urls, venue.official_url)
    tiers_urls = _with_fallback_urls(venue.age_tiers_urls, venue.official_url)

    # 2) Located in California (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_Located_In_California_With_Citation",
        desc="Venue is located in California, supported by a URL reference.",
        claim=f"The venue '{venue_name}' is located in the U.S. state of California (CA).",
        urls=loc_urls,
        critical=True,
        additional_instruction=(
            "Verify that the page explicitly indicates the venue is in California. "
            "Accept evidence such as 'California', 'CA', or a California city. "
            "Ensure the page refers to the same venue."
        )
    )

    # 3) Capacity at least 500 (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_Capacity_At_Least_500_With_Citation",
        desc="Venue has seating/guest capacity ≥ 500, supported by a URL reference.",
        claim=f"The venue '{venue_name}' has a seating or guest capacity of at least 500.",
        urls=cap_urls,
        critical=True,
        additional_instruction=(
            "Confirm that seating/guest capacity is 500 or higher. Accept phrasing like '500-capacity', "
            "'1,000 seats', 'over 500', etc. If multiple spaces are listed, it's sufficient if any main area "
            "or overall capacity is ≥ 500."
        )
    )

    # 4) Free admission for children under 3 (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_Free_Under_3_With_Citation",
        desc="Venue offers free admission for children under age 3, supported by a URL reference.",
        claim=f"The venue '{venue_name}' offers free admission for children under age 3 (e.g., '2 and under' free).",
        urls=free_urls,
        critical=True,
        additional_instruction=(
            "Look for explicit language such as 'children under 3 free' or 'age 2 and under free'. "
            "These are equivalent to 'under age 3'."
        )
    )

    # 5) Group discount for 10+ (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_Group_Discount_10plus_With_Citation",
        desc="Venue provides group discount tickets for groups of 10 or more, supported by a URL reference.",
        claim=f"The venue '{venue_name}' provides group discount tickets for groups of at least 10 people.",
        urls=group_urls,
        critical=True,
        additional_instruction=(
            "Accept phrasing like 'groups of 10+' or 'ten or more'. "
            "If the minimum group size is greater than 10 (e.g., 15+), this does NOT meet the requirement."
        )
    )

    # 6) Membership or season pass (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_Membership_Or_Season_Pass_With_Citation",
        desc="Venue offers either an annual membership or a season pass option, supported by a URL reference.",
        claim=f"The venue '{venue_name}' offers either an annual membership or a season pass option.",
        urls=memb_urls,
        critical=True,
        additional_instruction=(
            "Any clearly described annual membership program or season pass product qualifies."
        )
    )

    # 7) ADA + wheelchair accessible seating (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_ADA_Wheelchair_Accessible_Seating_With_Citation",
        desc="Venue is ADA compliant with wheelchair accessible seating, supported by a URL reference.",
        claim=f"The venue '{venue_name}' is ADA compliant and provides wheelchair accessible seating.",
        urls=ada_urls,
        critical=True,
        additional_instruction=(
            "Look for 'ADA compliant', 'accessible seating', 'wheelchair accessible seating/sections', or similar. "
            "The evidence must clearly indicate wheelchair-accessible seating is available."
        )
    )

    # 8) At least two age-based price tiers (with citation)
    await _verify_requirement_by_urls(
        evaluator=evaluator,
        parent_node=venue_node,
        node_id=f"V{vid}_Age_Based_Price_Tiers_With_Citation",
        desc="Venue offers at least two admission price tiers based on age, supported by a URL reference.",
        claim=f"The venue '{venue_name}' offers at least two admission price tiers differentiated by age.",
        urls=tiers_urls,
        critical=True,
        additional_instruction=(
            "Confirm that there are at least two admission categories based on age (e.g., child and adult, "
            "adult and senior, child and senior). The categories must be explicitly age-based."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the California family-friendly venues task.
    """
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

    # Extract venues and their citations from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Keep only first three venues; pad with empty objects if fewer than three
    venues: List[VenueItem] = (extracted.venues or [])[:3]
    while len(venues) < 3:
        venues.append(VenueItem())

    # Add distinctness check (critical at the root level)
    evaluator.add_custom_node(
        result=_venues_are_distinct(venues),
        id="Venues_Are_Distinct",
        desc="The three provided venues are different entities (no duplicate venue names/locations).",
        parent=root,
        critical=True
    )

    # Build and verify subtrees for each venue
    for idx in range(3):
        await _verify_single_venue(evaluator, root, venues[idx], idx)

    # Return full evaluation summary
    return evaluator.get_summary()