import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "state_capitol_tours_3x"
TASK_DESCRIPTION = """You are planning an educational civic tour for a student group and need to identify three U.S. state capitol buildings that meet the following requirements:

1. Each capitol building must be at least 300 feet tall
2. Each capitol must offer free public tours available on weekdays (Monday through Friday)
3. Each capitol must be a currently operational state legislature building
4. Tour schedule information (specific times and frequency) must be publicly available

For each of the three capitol buildings you identify, provide:
- The official name of the capitol building
- The complete street address (including city and state)
- The building's height in feet
- The architectural style of the building
- The weekday tour schedule (specific times and frequency)
- Tour capacity limits (if any exist)
- Reservation requirements or walk-in policy
- At least one contact method for tour information (phone number or website URL)
- A reference URL documenting the building height
- A reference URL documenting the architectural style
- A reference URL documenting the tour schedule and availability
- A reference URL for general capitol identification and contact information

All information must be verifiable through publicly available sources, and each piece of information should be supported by appropriate URL references.
"""

# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class CapitolItem(BaseModel):
    official_name: Optional[str] = None
    address: Optional[str] = None
    height_feet: Optional[str] = None
    architectural_style: Optional[str] = None

    tour_schedule_text: Optional[str] = None  # Include specific times and frequency for weekdays
    capacity_limits_text: Optional[str] = None  # "not specified" if explicitly stated as not specified
    reservation_policy_text: Optional[str] = None  # e.g., "walk-in allowed", "reservation required", etc.

    contact_phone: Optional[str] = None
    contact_url: Optional[str] = None

    height_ref_url: Optional[str] = None
    architecture_ref_url: Optional[str] = None
    tour_schedule_ref_url: Optional[str] = None
    general_contact_ref_url: Optional[str] = None


class CapitolsExtraction(BaseModel):
    capitols: List[CapitolItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_capitols() -> str:
    return """
Extract all U.S. state capitol buildings mentioned in the answer. Return a JSON object with a field "capitols" that is an array of objects. For each capitol, extract the following exact fields from the answer text:

- official_name: The official name of the capitol building, as provided in the answer.
- address: The complete street address including city and state, as provided in the answer.
- height_feet: The height value in feet, as provided in the answer (keep as string, include the unit 'ft' or 'feet' if present).
- architectural_style: The architectural style as provided in the answer.

- tour_schedule_text: A concise summary of the weekday tour schedule including specific start times and frequency (e.g., "Weekdays at 10:00am, 12:00pm, 2:00pm" or "Mon–Fri hourly from 9am–3pm"). If the answer explicitly provides these details, include them verbatim. Do not invent any schedule times.
- capacity_limits_text: If the answer provides capacity limits for tours (e.g., "groups up to 30"), include the quoted information. If the answer explicitly states that capacity limits are not specified, set this field to "not specified". If the answer says nothing about capacity limits, set this field to null.
- reservation_policy_text: The reservation or walk‑in policy description as provided (e.g., "walk‑ins welcome", "advance reservation required via website"). If not provided, set null.

- contact_phone: A phone number for tour information if provided; else null.
- contact_url: A website URL for tour information if provided; else null.

- height_ref_url: A URL cited in the answer that documents the building height.
- architecture_ref_url: A URL cited in the answer that documents the architectural style.
- tour_schedule_ref_url: A URL cited in the answer that documents the tour schedule and availability.
- general_contact_ref_url: A URL cited in the answer for general capitol identification and/or contact information (official capitol page or similar).

Special URL rules:
- Extract only URLs explicitly present in the answer text. If a URL is missing a protocol, prepend http://
- Do not infer or synthesize URLs.

Important:
- If more than three capitols are present in the answer, extract them all. We will later consider only the first three for evaluation.
- Preserve text exactly as written for schedule/policy/style/height where possible.
"""


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def is_non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def is_valid_url(u: Optional[str]) -> bool:
    if not is_non_empty(u):
        return False
    us = u.strip().lower()
    return us.startswith("http://") or us.startswith("https://")


def distinct_first_three(items: List[CapitolItem]) -> bool:
    """Check at least 3 items and first three are distinct (by official_name or address)."""
    first_three = items[:3]
    if len(first_three) < 3:
        return False
    keys = []
    for it in first_three:
        name_key = (it.official_name or "").strip().lower()
        addr_key = (it.address or "").strip().lower()
        keys.append((name_key, addr_key))
    # If different by either name or address, count unique
    unique = set(keys)
    return len(unique) == 3


def _safe(s: Optional[str]) -> str:
    return s if s is not None else ""


# -----------------------------------------------------------------------------
# Verification per‑capitol
# -----------------------------------------------------------------------------
async def verify_capitol(
    evaluator: Evaluator,
    root_node,
    cap: CapitolItem,
    idx_one_based: int
) -> None:
    """
    Build sub-tree and run verifications for a single capitol.
    """
    # Parent node for this capitol (non-critical to allow partial credit across capitols)
    cap_node = evaluator.add_parallel(
        id=f"capitol_{idx_one_based}",
        desc=f"Capitol building #{idx_one_based} (evaluated independently for partial credit)",
        parent=root_node,
        critical=False
    )

    # ---------------------------- Identity -----------------------------------
    identity_node = evaluator.add_parallel(
        id=f"capitol_{idx_one_based}_identity",
        desc=f"Capitol #{idx_one_based} identification fields are provided",
        parent=cap_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(cap.official_name),
        id=f"capitol_{idx_one_based}_official_name",
        desc="Official name of the capitol building is provided",
        parent=identity_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(cap.address),
        id=f"capitol_{idx_one_based}_complete_address",
        desc="Complete street address is provided (including city and state)",
        parent=identity_node,
        critical=True
    )

    # ---------------------------- Required URLs presence ----------------------
    urls_node = evaluator.add_parallel(
        id=f"capitol_{idx_one_based}_required_reference_urls",
        desc=f"Capitol #{idx_one_based} required reference URLs are provided",
        parent=cap_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_url(cap.height_ref_url),
        id=f"capitol_{idx_one_based}_height_reference_url",
        desc=f"A publicly accessible URL is provided documenting Capitol #{idx_one_based} building height",
        parent=urls_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_url(cap.architecture_ref_url),
        id=f"capitol_{idx_one_based}_architecture_reference_url",
        desc=f"A publicly accessible URL is provided documenting Capitol #{idx_one_based} architectural style",
        parent=urls_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_url(cap.tour_schedule_ref_url),
        id=f"capitol_{idx_one_based}_tour_schedule_reference_url",
        desc=f"A publicly accessible URL is provided documenting Capitol #{idx_one_based} tour schedule and availability",
        parent=urls_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_url(cap.general_contact_ref_url),
        id=f"capitol_{idx_one_based}_general_id_contact_url",
        desc=f"A publicly accessible URL is provided for general capitol identification and contact information",
        parent=urls_node,
        critical=True
    )

    # ---------------------------- Eligibility ---------------------------------
    elig_node = evaluator.add_parallel(
        id=f"capitol_{idx_one_based}_eligibility",
        desc=f"Capitol #{idx_one_based} meets all eligibility constraints",
        parent=cap_node,
        critical=True
    )

    # Operational
    op_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_operational",
        desc=f"Capitol #{idx_one_based} is a currently operational state legislature building",
        parent=elig_node,
        critical=True
    )
    claim_operational = (
        f"The {_safe(cap.official_name)} is the current, operational state capitol building housing the state's "
        f"legislature (e.g., Senate/House)."
    )
    await evaluator.verify(
        claim=claim_operational,
        node=op_leaf,
        sources=[u for u in [_safe(cap.general_contact_ref_url), _safe(cap.tour_schedule_ref_url)] if is_valid_url(u)],
        additional_instruction="Verify the page indicates this building is the state's official, current capitol and houses the active legislature."
    )

    # Height >= 300 ft
    h300_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_height_at_least_300",
        desc=f"Capitol #{idx_one_based} building height is at least 300 feet",
        parent=elig_node,
        critical=True
    )
    claim_h300 = f"The {_safe(cap.official_name)} has a height of at least 300 feet."
    await evaluator.verify(
        claim=claim_h300,
        node=h300_leaf,
        sources=cap.height_ref_url,
        additional_instruction="Check the height value on the cited page and confirm it's >= 300 ft (allow rounding tolerance)."
    )

    # Tours free
    free_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_tours_free",
        desc=f"Capitol #{idx_one_based} offers free public tours",
        parent=elig_node,
        critical=True
    )
    claim_free = f"Public tours at {_safe(cap.official_name)} are free (no charge)."
    await evaluator.verify(
        claim=claim_free,
        node=free_leaf,
        sources=cap.tour_schedule_ref_url,
        additional_instruction="Accept terms like 'free', 'no cost', or 'complimentary' as equivalent."
    )

    # Tours on weekdays
    weekdays_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_tours_weekdays",
        desc=f"Capitol #{idx_one_based} tours are available on weekdays (Monday–Friday)",
        parent=elig_node,
        critical=True
    )
    claim_weekdays = f"Public tours at {_safe(cap.official_name)} are available Monday through Friday."
    await evaluator.verify(
        claim=claim_weekdays,
        node=weekdays_leaf,
        sources=cap.tour_schedule_ref_url,
        additional_instruction="Look for 'Weekdays', 'Monday–Friday', or equivalent wording indicating weekday availability."
    )

    # Walk-in or online reservation available (OR condition)
    access_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_walkin_or_online_reservation_available",
        desc=f"Capitol #{idx_one_based} allows walk-in visitors OR has an online reservation system (as stated in the answer and supported by provided tour info sources)",
        parent=elig_node,
        critical=True
    )
    claim_access = (
        f"The tour access policy for {_safe(cap.official_name)} allows either walk-in visitors or provides an online reservation system."
    )
    await evaluator.verify(
        claim=claim_access,
        node=access_leaf,
        sources=cap.tour_schedule_ref_url,
        additional_instruction="Accept either explicit 'walk-ins welcome' or a clear online booking/reservation system as satisfying this criterion."
    )

    # ---------------------------- Required Details ----------------------------
    details_node = evaluator.add_parallel(
        id=f"capitol_{idx_one_based}_required_details",
        desc=f"Capitol #{idx_one_based} required descriptive and tour details are provided",
        parent=cap_node,
        critical=True
    )

    # Height value in feet (verify value against height reference)
    hval_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_height_value_feet",
        desc=f"Capitol #{idx_one_based} building height value is provided in feet",
        parent=details_node,
        critical=True
    )
    claim_hval = f"The height of {_safe(cap.official_name)} is {_safe(cap.height_feet)} (in feet)."
    await evaluator.verify(
        claim=claim_hval,
        node=hval_leaf,
        sources=cap.height_ref_url,
        additional_instruction="Verify that the stated height value matches (or reasonably rounds to) the value on the cited height reference page."
    )

    # Architectural style (verify against architecture reference)
    style_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_architectural_style",
        desc=f"Capitol #{idx_one_based} architectural style is provided",
        parent=details_node,
        critical=True
    )
    claim_style = f"The architectural style of {_safe(cap.official_name)} is {_safe(cap.architectural_style)}."
    await evaluator.verify(
        claim=claim_style,
        node=style_leaf,
        sources=cap.architecture_ref_url,
        additional_instruction="Match the provided style (e.g., Neoclassical, Beaux-Arts). Allow minor naming variations or multiple styles."
    )

    # Weekday tour schedule includes times and frequency
    sched_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_weekday_tour_schedule_times_frequency",
        desc=f"Capitol #{idx_one_based} weekday tour schedule includes specific times and frequency",
        parent=details_node,
        critical=True
    )
    claim_sched = (
        f"The weekday tour schedule for {_safe(cap.official_name)} provides specific start times and/or a clear frequency: "
        f"{_safe(cap.tour_schedule_text)}"
    )
    await evaluator.verify(
        claim=claim_sched,
        node=sched_leaf,
        sources=cap.tour_schedule_ref_url,
        additional_instruction="Confirm that the cited page lists concrete tour times and/or an explicit frequency (e.g., hourly, at 10am/12pm/2pm)."
    )

    # Capacity limits if any (or explicitly 'not specified')
    cap_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_capacity_limits_if_any",
        desc=f"Capitol #{idx_one_based} tour capacity limits are provided if specified by sources; otherwise the answer explicitly states that capacity limits are not specified",
        parent=details_node,
        critical=True
    )
    # Build a claim that works for either case
    cap_text = (_safe(cap.capacity_limits_text)).strip().lower()
    if is_non_empty(cap.capacity_limits_text) and cap_text not in ("not specified", "not stated", "n/a"):
        claim_capacity = (
            f"The tour capacity limits for {_safe(cap.official_name)} are: {_safe(cap.capacity_limits_text)}."
        )
        add_ins_capacity = "Verify that the cited page contains this capacity limit information (or equivalent phrasing/numbers)."
    else:
        claim_capacity = (
            f"The cited tour page for {_safe(cap.official_name)} does not specify any explicit tour capacity limits."
        )
        add_ins_capacity = "If the cited page does not mention capacity limits, accept this claim as supported; do not require explicit confirmation."
    await evaluator.verify(
        claim=claim_capacity,
        node=cap_leaf,
        sources=cap.tour_schedule_ref_url,
        additional_instruction=add_ins_capacity
    )

    # Reservation or walk-in policy text (verify matches cited page)
    res_leaf = evaluator.add_leaf(
        id=f"capitol_{idx_one_based}_reservation_or_walkin_policy",
        desc=f"Capitol #{idx_one_based} reservation requirements and/or walk-in policy is stated",
        parent=details_node,
        critical=True
    )
    claim_res = f"The reservation/walk-in policy for {_safe(cap.official_name)} is: {_safe(cap.reservation_policy_text)}."
    await evaluator.verify(
        claim=claim_res,
        node=res_leaf,
        sources=cap.tour_schedule_ref_url,
        additional_instruction="Confirm that the policy (e.g., walk-ins welcome; reservations required) appears on the cited page. Allow paraphrasing."
    )

    # At least one contact method provided (existence)
    contact_ok = is_non_empty(cap.contact_phone) or is_non_empty(cap.contact_url)
    evaluator.add_custom_node(
        result=contact_ok,
        id=f"capitol_{idx_one_based}_contact_method",
        desc="At least one contact method for tour information (phone number or website URL) is provided",
        parent=details_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 'three U.S. state capitol buildings' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent evaluation of sub-criteria and capitols
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
        prompt=prompt_extract_capitols(),
        template_class=CapitolsExtraction,
        extraction_name="capitols_extraction"
    )

    # Record summary info
    total_found = len(extracted.capitols)
    evaluator.add_custom_info(
        {"total_capitols_mentioned": total_found},
        info_type="stats",
        info_name="extraction_stats"
    )

    # Root-level critical check: set count and distinctness (using first three if more provided)
    # For robustness (per guidelines), accept answers that list >=3 by focusing on first 3 and ensuring they are distinct.
    count_node_result = (total_found >= 3) and distinct_first_three(extracted.capitols)
    evaluator.add_custom_node(
        result=count_node_result,
        id="set_count_and_distinctness",
        desc="Exactly three distinct U.S. state capitol buildings are identified (no duplicates)",
        parent=root,
        critical=True
    )

    # Prepare the three items to evaluate (pad with empty items if fewer than 3)
    items: List[CapitolItem] = extracted.capitols[:3]
    while len(items) < 3:
        items.append(CapitolItem())

    # Build each capitol subtree and run verifications
    for i, cap in enumerate(items, start=1):
        await verify_capitol(evaluator, root, cap, idx_one_based=i)

    # Return the full evaluation summary
    return evaluator.get_summary()