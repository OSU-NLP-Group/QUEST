import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_state_parks_rv_beach_pets_socal"
TASK_DESCRIPTION = """
I am planning an RV camping trip to Southern California with my dog and need to find suitable state park campgrounds. Identify three California state park campgrounds located in either Orange County or San Diego County that meet all of the following requirements: (1) Allow pets (dogs) in campsites, (2) Offer RV campsites with full hookups (water, electric, and sewer), (3) Accommodate RVs that are at least 35 feet in length, (4) Provide direct access to beach areas within the park, and (5) Have hiking trails within the park boundaries. For each campground, provide the official park name, and include a reference URL to the official California State Parks website page or the ReserveCalifornia reservation page for that specific campground.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    official_name: Optional[str] = None
    county: Optional[str] = None  # As claimed in the answer (may be absent)
    reference_url: Optional[str] = None  # Prefer parks.ca.gov or reservecalifornia.com page for this facility
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs cited for this campground


class CampgroundExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    From the answer, extract up to three California State Park campgrounds that the answer claims satisfy the user's criteria.
    For each campground, extract:
    - official_name: The official park or campground name as written (e.g., "San Elijo State Beach", "South Carlsbad State Beach").
    - county: The county mentioned for the campground (e.g., "San Diego County" or "Orange County") if stated in the answer; otherwise null.
    - reference_url: A single primary URL explicitly cited in the answer that points to either:
        • the official California State Parks website page for that park/campground (parks.ca.gov), or
        • the ReserveCalifornia reservation page for that specific campground (reservecalifornia.com).
      If multiple such official URLs are provided, pick the most directly relevant single page for that campground.
    - additional_urls: Any other URLs cited in the answer that are also associated with this same campground (can be empty).
    
    Important:
    - Only include California State Parks facilities, not city/county/private campgrounds.
    - Only include campgrounds the answer actually listed. Do not invent.
    - Return no more than three items, in the same order as in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_official_domain(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    return ("parks.ca.gov" in u) or ("reservecalifornia.com" in u)


def _collect_official_urls(item: CampgroundItem) -> List[str]:
    urls: List[str] = []
    if item.reference_url:
        urls.append(item.reference_url)
    urls.extend(item.additional_urls or [])
    # Keep only official domains and remove duplicates while preserving order
    seen = set()
    filtered: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        if _is_official_domain(u) and u not in seen:
            seen.add(u)
            filtered.append(u)
    return filtered


def _first_or_none(seq: List[str]) -> Optional[str]:
    return seq[0] if seq else None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_one_campground(
    evaluator: Evaluator,
    parent_node,
    item: CampgroundItem,
    index: int,
    prior_names: List[str]
) -> None:
    """
    Build verification subtree for a single campground, following the rubric leaves:
    - CG{n}_Name_and_Location
    - CG{n}_Pet_Policy
    - CG{n}_Full_Hookups
    - CG{n}_RV_Length
    - CG{n}_Beach_Access
    - CG{n}_Hiking_Trails
    - CG{n}_Reference_URL
    """
    cg_idx = index + 1
    cg_node = evaluator.add_parallel(
        id=f"Campground_{cg_idx}",
        desc=[
            "First campground meets all requirements",
            "Second campground meets all requirements",
            "Third campground meets all requirements",
        ][index] if index < 3 else f"Campground #{cg_idx} meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # Optional gating: require name and at least one official URL (to prevent meaningless downstream checks)
    required_info = evaluator.add_custom_node(
        result=bool(item and item.official_name and _is_official_domain(item.reference_url)),
        id=f"CG{cg_idx}_Required_Info",
        desc=f"Campground #{cg_idx} has an official name and an official (parks.ca.gov or reservecalifornia.com) reference URL",
        parent=cg_node,
        critical=True,
    )

    official_urls = _collect_official_urls(item)
    primary_url = _first_or_none(official_urls)

    # 1) Name and Location (Orange or San Diego County)
    name_loc_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Name_and_Location",
        desc="Provides the official name of a California state park campground located in Orange County or San Diego County",
        parent=cg_node,
        critical=True,
    )
    name_str = item.official_name or ""
    prior_names_str = ", ".join([n for n in prior_names if n]) if prior_names else ""
    name_loc_claim = (
        f"The official page shows the campground named '{name_str}' and indicates that it is located in either "
        f"Orange County or San Diego County, California."
    )
    await evaluator.verify(
        claim=name_loc_claim,
        node=name_loc_leaf,
        sources=official_urls,
        additional_instruction=(
            "Accept if the page explicitly states the county as Orange or San Diego, or if it clearly belongs to a "
            "San Diego Coast District/Orange Coast District that implies the county. If the page makes it clear the "
            "park is in either Orange or San Diego County (e.g., via address, district header, or site metadata), pass. "
            "If it is in any other county, fail. "
            f"If applicable, also ensure this campground is a different park from [{prior_names_str}]."
        ),
    )

    # 2) Pet policy (dogs allowed in campsites)
    pets_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Pet_Policy",
        desc="Confirms that pets (dogs) are allowed in the campground",
        parent=cg_node,
        critical=True,
    )
    pets_claim = (
        f"Dogs (pets) are allowed in the campsites at {name_str}. "
        "It is acceptable if there are restrictions (e.g., on leash) or if dogs are not allowed on the beach itself."
    )
    await evaluator.verify(
        claim=pets_claim,
        node=pets_leaf,
        sources=official_urls,
        additional_instruction=(
            "The requirement is that dogs are allowed in the campsites (overnight camping areas). "
            "If the page states dogs are only allowed in day-use but NOT allowed in the campground/campsites, fail."
        ),
    )

    # 3) Full hookups (water, electric, and sewer)
    hookups_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Full_Hookups",
        desc="Confirms availability of RV sites with full hookups (water, electric, and sewer)",
        parent=cg_node,
        critical=True,
    )
    hookups_claim = (
        f"The campground {name_str} offers RV sites with full hookups, meaning sites that provide water, electric, "
        "and sewer connections."
    )
    await evaluator.verify(
        claim=hookups_claim,
        node=hookups_leaf,
        sources=official_urls,
        additional_instruction=(
            "Pass ONLY if the page clearly indicates 'full hookups' or the equivalent (water + electric + sewer at the site). "
            "If only water/electric are provided with a dump station (no sewer hookup at site), that is NOT full hookups."
        ),
    )

    # 4) RV length at least 35 feet
    length_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_RV_Length",
        desc="Confirms the campground accommodates RVs of at least 35 feet in length",
        parent=cg_node,
        critical=True,
    )
    length_claim = (
        f"The maximum allowed RV length at {name_str} is at least 35 feet (i.e., 35 ft or greater)."
    )
    await evaluator.verify(
        claim=length_claim,
        node=length_leaf,
        sources=official_urls,
        additional_instruction=(
            "Look for site length limits or maximum vehicle length. "
            "If any listed maximum is 35 ft or more, pass. If all maximums are below 35 ft, fail."
        ),
    )

    # 5) Direct beach access within the park
    beach_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Beach_Access",
        desc="Confirms the park provides direct beach access or is located on the coast",
        parent=cg_node,
        critical=True,
    )
    beach_claim = (
        f"The park {name_str} provides direct access to beach areas within the park boundaries (i.e., it is on the coast)."
    )
    await evaluator.verify(
        claim=beach_claim,
        node=beach_leaf,
        sources=official_urls,
        additional_instruction=(
            "Accept if the page indicates a state beach campground or clearly states beach access within the park. "
            "Nearby off-park beaches without direct in-park access do not satisfy the requirement."
        ),
    )

    # 6) Hiking trails within the park boundaries
    trails_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Hiking_Trails",
        desc="Confirms the park has hiking trails within the park boundaries",
        parent=cg_node,
        critical=True,
    )
    trails_claim = (
        f"The park {name_str} has hiking trails within the park boundaries."
    )
    await evaluator.verify(
        claim=trails_claim,
        node=trails_leaf,
        sources=official_urls,
        additional_instruction=(
            "Look for mentions of 'trails', 'hiking', named trails, or maps indicating trails within the park. "
            "If trails are only outside the park with no access from within, or no trails are indicated, fail."
        ),
    )

    # 7) Reference URL validity (official page for this campground)
    ref_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Reference_URL",
        desc="Provides a valid reference URL to the official California State Parks page or reservation page for this specific campground",
        parent=cg_node,
        critical=True,
    )
    ref_claim = (
        f"This webpage is the official California State Parks (parks.ca.gov) page or the ReserveCalifornia reservation "
        f"page for the specific campground '{name_str}'."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=primary_url if primary_url else item.reference_url,
        additional_instruction=(
            "Pass only if the domain is parks.ca.gov or reservecalifornia.com and the content clearly corresponds "
            "to this specific campground/park (not a generic or unrelated page)."
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
    Evaluate an answer for the SoCal California State Parks RV campground requirements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent evaluation of each campground
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

    # Extract the campground list from the answer
    extracted: CampgroundExtraction = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundExtraction,
        extraction_name="campground_extraction",
    )

    # Ensure exactly 3 slots (pad with empty if fewer)
    items: List[CampgroundItem] = list(extracted.campgrounds[:3])
    while len(items) < 3:
        items.append(CampgroundItem())

    # Build the verification tree per campground
    prior_names: List[str] = []
    for i, item in enumerate(items):
        await verify_one_campground(
            evaluator=evaluator,
            parent_node=root,
            item=item,
            index=i,
            prior_names=prior_names.copy(),
        )
        # Track names to encourage distinct campgrounds in subsequent checks
        if item and item.official_name:
            prior_names.append(item.official_name)

    return evaluator.get_summary()