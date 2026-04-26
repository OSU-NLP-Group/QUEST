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
TASK_ID = "pnw_wilderness_permits_2026"
TASK_DESCRIPTION = (
    "Your outdoor recreation club is planning a Pacific Northwest wilderness backpacking trip for a group of 10 adults "
    "(all aged 18 or older) during July 2026. The group is considering two national parks in Washington State: Olympic "
    "National Park and Mount Rainier National Park. For each park, determine: (1) whether the park allows groups of 10 "
    "people to camp together at designated wilderness camping sites, (2) the total wilderness camping permit cost for "
    "all 10 adults for a 2-night trip, (3) what food storage equipment or infrastructure is provided or required at "
    "designated backcountry campsites, and (4) when advance reservations for summer wilderness camping permits become available."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkDetermination(BaseModel):
    # 1) Group size/site capacity determination
    group_camping_allowed_together: Optional[str] = None  # e.g., "Yes", "No", "Depends", short summary
    group_policy_details: Optional[str] = None            # free-text details about single-site capacity for 10 people
    group_policy_urls: List[str] = Field(default_factory=list)  # URLs cited for group/site capacity

    # 2) Total permit cost (for 10 adults, 2 nights)
    total_permit_cost_2_nights_10_adults_usd: Optional[str] = None  # e.g., "$160", "USD 180", "about $200"
    permit_fee_components: Optional[str] = None  # e.g., "$8/person/night + $6 reservation fee", any breakdown text
    permit_fee_urls: List[str] = Field(default_factory=list)        # URLs cited for fee schedule and any added fees

    # 3) Food storage equipment/infrastructure requirement
    food_storage_requirement: Optional[str] = None  # e.g., "Bear canisters required", "Bear wires provided in some camps"
    food_storage_urls: List[str] = Field(default_factory=list)      # URLs cited for food storage policy/infrastructure

    # 4) Reservation opening timing for summer permits
    reservation_open_timing: Optional[str] = None   # e.g., "March 15 at 7am PT", "6 months in advance", etc.
    reservation_urls: List[str] = Field(default_factory=list)       # URLs cited for reservation opening timing


class ParksExtraction(BaseModel):
    olympic: Optional[ParkDetermination] = None
    rainier: Optional[ParkDetermination] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks_determinations() -> str:
    return """
Extract the determinations the answer makes for each park (Olympic National Park and Mount Rainier National Park).
You must strictly extract only what is stated in the answer (do not invent anything). If a field is not stated, return null or an empty list.

For each park, extract:
1) group_camping_allowed_together: A short yes/no/depends or concise phrase indicating whether a single designated wilderness/backcountry campsite can host a 10-person group together.
2) group_policy_details: The exact or paraphrased details the answer gives about site capacity/group-size rules relevant to having 10 people at one designated campsite (include helpful phrases like 'max 6 per site' or 'group sites for 12' if present).
3) group_policy_urls: All URLs cited that support the group-size/site-capacity policy.

4) total_permit_cost_2_nights_10_adults_usd: The total cost the answer claims for a 2-night trip for all 10 adults. Extract as plain text (e.g., '$160', 'USD 200', 'approx. $180').
5) permit_fee_components: The fee structure text from the answer (e.g., '$8/person/night plus $6 reservation fee').
6) permit_fee_urls: All URLs cited that support the fee structure or any required permit/transaction fees.

7) food_storage_requirement: The answer’s statement about what food storage is provided/required at designated backcountry campsites (e.g., 'Bear canisters required', 'Bear wires provided', 'Food lockers at X sites', etc.).
8) food_storage_urls: All URLs cited that support the food storage requirement/provision.

9) reservation_open_timing: The answer’s statement about when advance reservations for summer (July) wilderness/backcountry permits become available (e.g., a fixed date/time or a rolling window such as '6 months in advance').
10) reservation_urls: All URLs cited that support the reservation opening timing.

Return the JSON with two top-level objects: 'olympic' and 'rainier'. For any missing information, use null for strings and [] for URL lists.

Important rules for URL extraction:
- Extract only URLs actually present in the answer text (including markdown links).
- If a URL is missing protocol, prepend 'http://'.
- Do not invent or infer URLs.
"""


# --------------------------------------------------------------------------- #
# Helpers for claim construction                                              #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s:
                # if protocol missing, prepend http:// as per extraction special rules
                if not (s.startswith("http://") or s.startswith("https://")):
                    s = "http://" + s
                out.append(s)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _build_group_size_claim(park_name: str, det: ParkDetermination) -> str:
    """
    Build a verifiable claim about whether a single designated wilderness/backcountry campsite
    can host a 10-person group together, grounded in the answer's extracted content.
    """
    primary = (det.group_camping_allowed_together or "").strip()
    details = (det.group_policy_details or "").strip()

    text = ""
    if primary:
        low = primary.lower()
        # Try to normalize into a clear, binary-styled statement when possible
        neg_tokens = ["not allowed", "no", "cannot", "can't", "prohibited", "do not allow", "does not allow"]
        pos_tokens = ["yes", "allowed", "permits", "can", "allowed at group sites", "group site"]

        if any(tok in low for tok in neg_tokens):
            text = f"{park_name} does not allow a single designated wilderness/backcountry campsite to host a group of 10 adults together."
        elif any(tok in low for tok in pos_tokens):
            text = f"{park_name} allows a group of 10 adults to camp together at a single designated wilderness/backcountry campsite (e.g., via a site with capacity ≥ 10 or a designated group site)."

    # Fall back to using the provided details as the core claim if needed
    if not text:
        if details:
            text = f"In {park_name}, regarding whether a single designated wilderness/backcountry campsite can host 10 adults together, the applicable policy is: {details}"
        else:
            # As a last resort, still phrase the requirement to be verified
            text = f"In {park_name}, verify whether a single designated wilderness/backcountry campsite can host a group of 10 adults together."

    return text


def _build_permit_cost_claim(park_name: str, det: ParkDetermination) -> str:
    total = (det.total_permit_cost_2_nights_10_adults_usd or "").strip()
    components = (det.permit_fee_components or "").strip()

    if total and components:
        return (
            f"For {park_name}, the total wilderness/backcountry permit cost for 10 adults for a 2-night trip is {total}, "
            f"based on this fee structure: {components}"
        )
    elif total:
        return (
            f"For {park_name}, the total wilderness/backcountry permit cost for 10 adults for a 2-night trip is {total}."
        )
    elif components:
        return (
            f"For {park_name}, the wilderness/backcountry permit fees for this scenario are described as: {components}. "
            f"These fees imply the total cost for 10 adults for 2 nights."
        )
    else:
        return (
            f"For {park_name}, determine and verify the total wilderness/backcountry permit cost for 10 adults for a "
            f"2-night trip based on the official fee structure."
        )


def _build_food_storage_claim(park_name: str, det: ParkDetermination) -> str:
    storage = (det.food_storage_requirement or "").strip()
    if storage:
        return f"At designated backcountry campsites in {park_name}, the food storage requirement/provision is: {storage}"
    else:
        return f"At designated backcountry campsites in {park_name}, verify the required/provided food storage method(s)."


def _build_reservation_timing_claim(park_name: str, det: ParkDetermination) -> str:
    timing = (det.reservation_open_timing or "").strip()
    if timing:
        return f"For {park_name}, advance reservations for summer (July) wilderness/backcountry permits become available: {timing}"
    else:
        return f"For {park_name}, verify when advance reservations for summer (July) wilderness/backcountry permits become available."


# --------------------------------------------------------------------------- #
# Verification per park                                                       #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park_id: str,
    park_name: str,
    det: Optional[ParkDetermination],
) -> None:
    """
    Build the verification subtree and run four critical checks for the given park.
    If det is None, create leaf nodes and they will likely fail due to missing support.
    """
    # Park-level node (parallel, non-critical)
    park_node_id = "olympic_national_park" if park_id == "olympic" else "mount_rainier_national_park"
    park_node_desc = f"Required determinations for {park_name}"
    park_node = evaluator.add_parallel(
        id=park_node_id,
        desc=park_node_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare data (safe fallbacks)
    det = det or ParkDetermination()

    # Leaf nodes
    # 1) Group size/site capacity (critical)
    group_leaf = evaluator.add_leaf(
        id=f"{park_id}_group_size",
        desc=f"Determine whether {park_name} allows a group of 10 adults to camp together at designated wilderness camping sites (i.e., a single site that accommodates the entire group together)",
        parent=park_node,
        critical=True,
    )

    # 2) Total permit cost (critical)
    cost_leaf = evaluator.add_leaf(
        id=f"{park_id}_permit_cost",
        desc=f"Compute the total wilderness camping permit cost for all 10 adults for a 2-night trip, using the park’s official per-person per-night fees plus any applicable permit/transaction fees stated by the park",
        parent=park_node,
        critical=True,
    )

    # 3) Food storage requirement/provision (critical)
    food_leaf = evaluator.add_leaf(
        id=f"{park_id}_food_storage",
        desc=f"State what food storage equipment/infrastructure is provided at designated backcountry campsites and/or what food storage campers are required to bring or use in {park_name}",
        parent=park_node,
        critical=True,
    )

    # 4) Reservation timing (critical)
    reserve_leaf = evaluator.add_leaf(
        id=f"{park_id}_reservation_timing",
        desc=f"State when advance reservations for summer (July) wilderness camping permits become available for {park_name}",
        parent=park_node,
        critical=True,
    )

    # Build claims and sources
    group_claim = _build_group_size_claim(park_name, det)
    group_sources = _clean_urls(det.group_policy_urls)
    group_instruction = (
        "Verify whether a single designated wilderness/backcountry campsite can host a 10-person group together. "
        "Rely on official NPS or Recreation.gov policy pages that specify per-site capacity or group-size rules. "
        "Accept 'not allowed' if maximum per site is < 10 or if policy requires splitting across multiple sites. "
        "Accept 'allowed' only if official sources show a single designated site (e.g., a group site) with capacity ≥ 10."
    )

    cost_claim = _build_permit_cost_claim(park_name, det)
    cost_sources = _clean_urls(det.permit_fee_urls)
    cost_instruction = (
        "Verify the total against the official fee structure. If the per-person per-night fee is stated, compute "
        "10 adults × 2 nights × per-person-per-night, plus any additional listed permit/transaction fees. "
        "Allow minor rounding differences. The cited sources must clearly support the fee structure used."
    )

    food_claim = _build_food_storage_claim(park_name, det)
    food_sources = _clean_urls(det.food_storage_urls)
    food_instruction = (
        "Verify the stated food storage requirement/provision (e.g., bear canisters required, bear wires/lockers provided). "
        "Look for explicit policy language for designated wilderness/backcountry campsites on official sources."
    )

    reserve_claim = _build_reservation_timing_claim(park_name, det)
    reserve_sources = _clean_urls(det.reservation_urls)
    reserve_instruction = (
        "Verify when advance reservations for summer (July) wilderness/backcountry permits become available. "
        "Accept a fixed date/time or a rolling window (e.g., '6 months in advance') if clearly stated on official sources."
    )

    # Run verifications (in parallel for this park)
    await evaluator.batch_verify([
        (group_claim, group_sources, group_leaf, group_instruction),
        (cost_claim, cost_sources, cost_leaf, cost_instruction),
        (food_claim, food_sources, food_leaf, food_instruction),
        (reserve_claim, reserve_sources, reserve_leaf, reserve_instruction),
    ])


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
    Evaluate the answer for the PNW wilderness permits task (Olympic and Mount Rainier) using the Mind2Web2 framework.
    Returns a standard evaluation summary dict.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parks evaluated independently
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

    # Extract structured information from the answer
    parks_info = await evaluator.extract(
        prompt=prompt_extract_parks_determinations(),
        template_class=ParksExtraction,
        extraction_name="parks_determinations"
    )

    # Add scenario context as custom info (optional)
    evaluator.add_custom_info(
        info={
            "scenario": "10 adults (18+), 2 nights, July 2026",
            "parks": ["Olympic National Park", "Mount Rainier National Park"],
            "required_items": [
                "group size/site capacity determination for single designated site with 10 people",
                "total permit cost for 10 adults for 2 nights",
                "food storage requirement/provision",
                "reservation open timing for summer permits"
            ]
        },
        info_type="scenario_context"
    )

    # Build verification subtrees for both parks in parallel
    await asyncio.gather(
        verify_park(
            evaluator=evaluator,
            parent_node=root,
            park_id="olympic",
            park_name="Olympic National Park",
            det=parks_info.olympic
        ),
        verify_park(
            evaluator=evaluator,
            parent_node=root,
            park_id="rainier",
            park_name="Mount Rainier National Park",
            det=parks_info.rainier
        ),
    )

    # Return evaluation summary
    return evaluator.get_summary()