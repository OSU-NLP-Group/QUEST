import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_state_parks_rv_accessible_3_campgrounds"
TASK_DESCRIPTION = (
    "You are planning a 2-week RV camping trip through California state parks during September 2026. "
    "Your travel party includes family members who require accessible facilities, and you are traveling in a 35-foot Class A motorhome. "
    "Identify 3 different California state park campgrounds that meet ALL of the following requirements: "
    "(1) The campground must be part of the California State Parks system and accept reservations through ReserveCalifornia; "
    "(2) The campground must accommodate RVs with a maximum length of at least 35 feet; "
    "(3) The campground must offer ADA-accessible campsites; "
    "(4) The campground must allow reservations to be made 6 months in advance; "
    "(5) Each campground must have an official California State Parks webpage or ReserveCalifornia listing that confirms these specifications. "
    "For each of the 3 campgrounds, provide: the official name of the campground, the specific maximum RV length accepted, confirmation that accessible campsites are available, "
    "and a reference URL from the official California State Parks website (parks.ca.gov) or ReserveCalifornia.com."
)

ALLOWED_OFFICIAL_DOMAINS = ["parks.ca.gov", "reservecalifornia.com"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    official_name: Optional[str] = None
    max_rv_length: Optional[str] = None  # keep as string to accept variants like "35 ft", "40 feet", etc.
    accessible_confirmation: Optional[str] = None  # any text indicating ADA-accessible campsites availability
    reservation_system: Optional[str] = None  # e.g., "ReserveCalifornia"
    reservation_window: Optional[str] = None  # e.g., "6 months"
    reference_urls: List[str] = Field(default_factory=list)  # URLs explicitly cited for this campground


class CampgroundList(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to three (3) campgrounds presented in the answer that are proposed for a California State Parks RV trip.

    For each campground, extract the following fields strictly from the answer text:
    - official_name: the official name of the specific campground (not just the park name), if provided; otherwise the park campground name.
    - max_rv_length: the stated maximum RV length the campground can accommodate (as written, including units such as "ft" or "feet"); if missing, return null.
    - accessible_confirmation: the text/snippet that indicates ADA-accessible/accessible campsites are available; if missing, return null.
    - reservation_system: the named reservation system (e.g., "ReserveCalifornia") as stated; if missing, return null.
    - reservation_window: the text indicating how far in advance reservations can be made (e.g., "6 months"); if missing, return null.
    - reference_urls: a list of all URLs explicitly cited for this campground in the answer. Include only valid full URLs. Prefer official sources (parks.ca.gov or reservecalifornia.com) if present. Do not invent URLs.

    Rules:
    - Do not infer or fabricate any values. Return null if a field is not explicitly present in the answer.
    - For reference_urls, include only URLs actually shown in the answer (plain links or markdown links).
    - Return exactly the fields specified above for each campground.
    - Return at most three campgrounds, in the same order as the answer presents them.

    Return a JSON object:
    {
      "campgrounds": [
        { ... up to 3 items ... }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    u = url.strip().lower()
    return any(domain in u for domain in ALLOWED_OFFICIAL_DOMAINS)


def filter_official_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if is_official_url(u)]


def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


# --------------------------------------------------------------------------- #
# Verification for one campground                                             #
# --------------------------------------------------------------------------- #
async def verify_campground(
    evaluator: Evaluator,
    parent_node,
    campground: CampgroundItem,
    index: int,
) -> None:
    """
    Build verification subtree for a single campground and run verifications.

    Leaves (all critical):
      - location: CA State Parks system
      - rv_length: accommodates >= 35 feet
      - reservation_system: ReserveCalifornia + 6 months in advance
      - accessible_sites: ADA-accessible campsites available
      - reference: at least one official URL from parks.ca.gov or reservecalifornia.com
    """
    label = ordinal(index)
    cg_name = campground.official_name or f"Campground #{index + 1}"

    # Parent node for this campground (non-critical to allow partial across the three)
    cg_node = evaluator.add_parallel(
        id=f"campground_{index+1}",
        desc=f"{label} campground meets all specified requirements",
        parent=parent_node,
        critical=False
    )

    # Compute official URLs first and add a critical custom node for domain validity
    official_urls = filter_official_urls(campground.reference_urls)
    has_valid_official_ref = len(official_urls) > 0

    reference_node = evaluator.add_custom_node(
        result=has_valid_official_ref,
        id=f"campground_{index+1}_reference",
        desc="Valid reference URL provided from California State Parks official website (parks.ca.gov) or ReserveCalifornia.com that confirms the specifications",
        parent=cg_node,
        critical=True
    )

    # Create leaf nodes (critical as per rubric)
    location_node = evaluator.add_leaf(
        id=f"campground_{index+1}_location",
        desc="Campground is located within the California State Parks system",
        parent=cg_node,
        critical=True
    )
    rvlen_node = evaluator.add_leaf(
        id=f"campground_{index+1}_rv_length",
        desc="Campground accommodates RVs with a maximum length of at least 35 feet",
        parent=cg_node,
        critical=True
    )
    reservation_node = evaluator.add_leaf(
        id=f"campground_{index+1}_reservation_system",
        desc="Campground accepts reservations through ReserveCalifornia and allows booking 6 months in advance",
        parent=cg_node,
        critical=True
    )
    accessible_node = evaluator.add_leaf(
        id=f"campground_{index+1}_accessible_sites",
        desc="Campground provides ADA-accessible campsites",
        parent=cg_node,
        critical=True
    )

    # Build claims
    location_claim = (
        f"The campground '{cg_name}' is part of and managed by the California State Parks system."
    )
    rvlen_claim = (
        f"The campground '{cg_name}' accommodates RVs with a maximum length of at least 35 feet."
    )
    reservation_claim = (
        f"Reservations for the campground '{cg_name}' are made via ReserveCalifornia and can be booked 6 months in advance."
    )
    accessible_claim = (
        f"The campground '{cg_name}' offers ADA-accessible (accessible) campsites."
    )

    # Additional instructions to guide the verifier
    add_ins_location = (
        "Confirm the page is an official California State Parks park/campground page (parks.ca.gov) "
        "or an official ReserveCalifornia listing for a California State Park. "
        "If the page is unrelated or not official, mark as not supported."
    )
    add_ins_rvlen = (
        "Verify from the official page(s) whether the campground allows RVs of at least 35 feet. "
        "Accept if the maximum length shown is 35 feet or greater (e.g., 36, 40, 45). "
        "If only smaller lengths (e.g., 31 or 32) are indicated, mark as not supported. "
        "If multiple campsite types exist, it suffices that at least some sites accommodate ≥35'."
    )
    add_ins_reservation = (
        "Verify that reservations are handled through ReserveCalifornia AND the booking window is 6 months in advance. "
        "Many official pages or ReserveCalifornia listings explicitly note a '6 months' booking window. "
        "If either ReserveCalifornia is not indicated or the 6-month window is not supported on the provided page(s), mark as not supported."
    )
    add_ins_accessible = (
        "Confirm that accessible (ADA) campsites are available. Phrases like 'accessible sites', 'ADA sites', "
        "'accessibility' for campsites should count as positive confirmation."
    )

    # Run verifications. Because reference_node is a critical sibling with final status (passed/failed),
    # the evaluator will auto-treat it as a prerequisite; if it failed, the leaves below will be skipped.
    # Only use official URLs for verifying core facts.
    claims_and_sources = [
        (location_claim, official_urls, location_node, add_ins_location),
        (rvlen_claim, official_urls, rvlen_node, add_ins_rvlen),
        (reservation_claim, official_urls, reservation_node, add_ins_reservation),
        (accessible_claim, official_urls, accessible_node, add_ins_accessible),
    ]
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the California State Parks RV+ADA campground task.
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
        default_model=model
    )

    # Record allowed domains for transparency
    evaluator.add_custom_info(
        info={"allowed_official_domains": ALLOWED_OFFICIAL_DOMAINS},
        info_type="policy",
        info_name="official_domain_policy"
    )

    # 1) Extract up to 3 campgrounds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundList,
        extraction_name="extracted_campgrounds"
    )

    # Ensure exactly 3 campgrounds (pad with empty items if fewer; truncate if more)
    campgrounds: List[CampgroundItem] = list(extracted.campgrounds or [])
    while len(campgrounds) < 3:
        campgrounds.append(CampgroundItem())
    if len(campgrounds) > 3:
        campgrounds = campgrounds[:3]

    # 2) Build verification for each campground
    for i in range(3):
        await verify_campground(evaluator, root, campgrounds[i], i)

    # 3) Return summary
    return evaluator.get_summary()