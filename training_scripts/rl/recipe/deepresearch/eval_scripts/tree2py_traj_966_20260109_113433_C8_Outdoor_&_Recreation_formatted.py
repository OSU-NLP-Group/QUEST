import asyncio
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_lake_erie_rv"
TASK_DESCRIPTION = (
    "Identify 2 Ohio state park campgrounds located on Lake Erie shoreline that meet ALL of the following "
    "requirements for a large family RV camping trip:\n\n"
    "- At least 90 total campsites\n"
    "- At least 30 full hookup campsites (providing water, sewer, and electric connections)\n"
    "- Direct beach access to Lake Erie\n"
    "- ADA accessible campsites available\n"
    "- Boat launch facilities for campers\n"
    "- Playground facilities\n\n"
    "For each campground, provide its official name and a reference URL from the Ohio Department of Natural Resources "
    "or official state park website."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    official_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract campground entries mentioned in the answer.

    For each campground, extract:
    - official_name: the official campground or park name as written in the answer (string; null if not provided)
    - reference_urls: a list of all URLs given as references for that campground (include all URLs exactly as provided; do not invent)

    Return a JSON object:
    {
      "campgrounds": [
        {"official_name": "...", "reference_urls": ["...", "..."]},
        ...
      ]
    }

    Notes:
    - Do not filter by location or requirements here; just extract all campgrounds and their URLs mentioned by the answer.
    - Include every URL presented (plain, markdown, or embedded); keep them as a list.
    - If no URL is provided for an item, use an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_ohio_dnr_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        # Common official domains (past and current):
        # - ohiodnr.gov (primary)
        # - parks.ohiodnr.gov (legacy subdomain)
        # - stateparks.ohio.gov (legacy; sometimes redirects)
        # Any subdomain of ohiodnr.gov should be considered official.
        if host.endswith("ohiodnr.gov"):
            return True
        if host.endswith("stateparks.ohio.gov"):
            return True
        return False
    except Exception:
        return False


def filter_official_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if is_official_ohio_dnr_url(u)]


def _name_or_placeholder(item: CampgroundItem, idx: int) -> str:
    return item.official_name.strip() if item.official_name else f"campground #{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_campground_verification(
    evaluator: Evaluator,
    parent_node,
    item: CampgroundItem,
    index: int,
) -> None:
    """
    Build verification subtree for one campground.
    We structure it as:
      Campground_i (SEQUENTIAL, critical)
        - Reference_URL_i_present (custom, critical)
        - Reference_URL_i (officialness, critical)
        - Constraints_i (PARALLEL, critical)
             • Ohio_State_Park_i
             • Lake_Erie_Location_i
             • Minimum_Sites_i
             • Full_Hookup_Count_i
             • Beach_Access_i
             • ADA_Sites_i
             • Boat_Launch_i
             • Playground_i
    """
    cg_name = _name_or_placeholder(item, index)
    urls_all = item.reference_urls or []
    urls_official = filter_official_urls(urls_all)

    cg_node = evaluator.add_sequential(
        id=f"Campground_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} qualifying campground identification and verification",
        parent=parent_node,
        critical=True  # Parent 'Task_Completion' is critical ⇒ children must also be critical
    )

    # 1) Reference URL presence (existence)
    evaluator.add_custom_node(
        result=(len(urls_all) > 0),
        id=f"Reference_URL_{index + 1}_present",
        desc=f"Provides at least one reference URL for {'first' if index == 0 else 'second'} campground",
        parent=cg_node,
        critical=True
    )

    # 2) Officialness of provided URL(s)
    # Verify that at least one provided URL is an official Ohio DNR or Ohio State Parks website page
    official_url_leaf = evaluator.add_leaf(
        id=f"Reference_URL_{index + 1}",
        desc=f"Provides reference URL from official Ohio DNR or park website for {'first' if index == 0 else 'second'} campground",
        parent=cg_node,
        critical=True
    )
    claim_official = (
        f"At least one of these URLs is an official Ohio Department of Natural Resources (ODNR) or Ohio State Parks "
        f"webpage for the park or campground related to {cg_name}."
    )
    await evaluator.verify(
        claim=claim_official,
        node=official_url_leaf,
        sources=urls_all,
        additional_instruction=(
            "Accept as official if the domain is ohiodnr.gov (including subdomains like parks.ohiodnr.gov) "
            "or stateparks.ohio.gov. The page does not need to be a dedicated campground subpage; a park-level page "
            "is acceptable if it clearly pertains to the same park/campground."
        ),
    )

    # 3) Constraints (parallel group) - only meaningful after official URL confirmed (sequential gating ensures that)
    constraints_node = evaluator.add_parallel(
        id=f"Constraints_{index + 1}",
        desc=f"All required facilities and counts for {'first' if index == 0 else 'second'} campground",
        parent=cg_node,
        critical=True
    )

    # Each verification below uses only official URLs when available.
    sources_for_constraints = urls_official if urls_official else urls_all

    # 3.1 Ohio State Park system
    ohio_park_leaf = evaluator.add_leaf(
        id=f"Ohio_State_Park_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground is an Ohio state park",
        parent=constraints_node,
        critical=True
    )
    claim_ohio_park = (
        f"{cg_name} is part of the Ohio State Parks system (managed by ODNR)."
    )
    await evaluator.verify(
        claim=claim_ohio_park,
        node=ohio_park_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Confirm the page indicates the park/campground belongs to the Ohio State Parks (ODNR). "
            "Look for ODNR/Ohio State Parks branding or explicit references."
        ),
    )

    # 3.2 Lake Erie shoreline location
    erie_loc_leaf = evaluator.add_leaf(
        id=f"Lake_Erie_Location_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground is located on Lake Erie shoreline",
        parent=constraints_node,
        critical=True
    )
    claim_erie = f"{cg_name} is located on the Lake Erie shoreline (i.e., directly on Lake Erie)."
    await evaluator.verify(
        claim=claim_erie,
        node=erie_loc_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Look for explicit mentions of 'Lake Erie', shore access, coastal positioning, or maps indicating direct "
            "location along the Lake Erie shoreline."
        ),
    )

    # 3.3 Minimum total campsites (>= 90)
    min_sites_leaf = evaluator.add_leaf(
        id=f"Minimum_Sites_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground has at least 90 total campsites",
        parent=constraints_node,
        critical=True
    )
    claim_min_sites = (
        f"The campground at {cg_name} has at least 90 total campsites (counting all types)."
    )
    await evaluator.verify(
        claim=claim_min_sites,
        node=min_sites_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Use the page to find a total campsite count (or a sum of listed site types) and check it is >= 90. "
            "If the page lists multiple site categories (e.g., electric, full-hookup, non-electric), sum as needed."
        ),
    )

    # 3.4 Full hookup count (>= 30; water + sewer + electric)
    fhu_leaf = evaluator.add_leaf(
        id=f"Full_Hookup_Count_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground has at least 30 full hookup campsites with water, sewer, and electric",
        parent=constraints_node,
        critical=True
    )
    claim_fhu = (
        f"The campground at {cg_name} has at least 30 full-hookup campsites that include water, sewer, and electric."
    )
    await evaluator.verify(
        claim=claim_fhu,
        node=fhu_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Look for terms such as 'full hookup', 'full-service', 'FHU', and ensure it implies water, sewer, and electric. "
            "Verify that the count of such sites is at least 30."
        ),
    )

    # 3.5 Direct beach access to Lake Erie
    beach_leaf = evaluator.add_leaf(
        id=f"Beach_Access_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground has direct beach access to Lake Erie",
        parent=constraints_node,
        critical=True
    )
    claim_beach = f"The campground at {cg_name} offers direct beach access to Lake Erie."
    await evaluator.verify(
        claim=claim_beach,
        node=beach_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Confirm that a beach is directly accessible from the park/campground and that it is on Lake Erie."
        ),
    )

    # 3.6 ADA accessible campsites available
    ada_leaf = evaluator.add_leaf(
        id=f"ADA_Sites_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground has ADA accessible campsites available",
        parent=constraints_node,
        critical=True
    )
    claim_ada = f"ADA-accessible campsites are available at the {cg_name} campground."
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Look for 'ADA', 'accessible sites', or similar statements indicating dedicated accessible campsites."
        ),
    )

    # 3.7 Boat launch facilities
    boat_leaf = evaluator.add_leaf(
        id=f"Boat_Launch_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground has boat launch facilities",
        parent=constraints_node,
        critical=True
    )
    claim_boat = f"The {cg_name} area provides boat launch facilities (e.g., ramp or marina ramp) accessible to campers."
    await evaluator.verify(
        claim=claim_boat,
        node=boat_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Look for 'boat launch', 'launch ramp', 'ramp', or 'marina' facilities suitable for launching boats."
        ),
    )

    # 3.8 Playground facilities
    play_leaf = evaluator.add_leaf(
        id=f"Playground_{index + 1}",
        desc=f"{'First' if index == 0 else 'Second'} campground has playground facilities",
        parent=constraints_node,
        critical=True
    )
    claim_play = f"The {cg_name} campground or park provides a playground."
    await evaluator.verify(
        claim=claim_play,
        node=play_leaf,
        sources=sources_for_constraints,
        additional_instruction=(
            "Confirm the presence of a 'playground' (sometimes listed under amenities or day-use facilities)."
        ),
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
    Evaluate an answer for the Ohio Lake Erie state park campground task.
    """
    # Initialize evaluator with a generic root
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Add ground truth style requirements info (for context in summary)
    evaluator.add_ground_truth({
        "required_count": 2,
        "constraints": {
            "min_total_campsites": ">= 90",
            "min_full_hookup_campsites": ">= 30 (water, sewer, electric)",
            "direct_beach_access": "Lake Erie",
            "ADA_accessible_sites": True,
            "boat_launch": True,
            "playground": True,
            "location": "On Lake Erie shoreline"
        },
        "source_requirement": "Official Ohio DNR / Ohio State Parks URL"
    }, gt_type="task_requirements")

    # Build top-level task completion node (critical)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identifies 2 qualifying Ohio state park campgrounds on Lake Erie meeting all specified requirements",
        parent=root,
        critical=True
    )

    # Select up to the first two campgrounds from the answer; pad if fewer
    items = list(extracted.campgrounds) if extracted and extracted.campgrounds else []
    selected: List[CampgroundItem] = items[:2]
    while len(selected) < 2:
        selected.append(CampgroundItem())

    # Build verification for each campground
    for idx, item in enumerate(selected):
        await build_campground_verification(evaluator, task_node, item, idx)

    # Return summary
    return evaluator.get_summary()