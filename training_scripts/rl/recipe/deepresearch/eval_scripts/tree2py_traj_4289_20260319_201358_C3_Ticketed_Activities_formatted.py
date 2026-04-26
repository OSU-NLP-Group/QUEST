import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dwts_2026_midwest_capacity_ada"
TASK_DESCRIPTION = (
    "Among the venues hosting the Dancing With The Stars Live! 2026 Tour in February 2026, "
    "identify one venue located in a Midwest United States state (Indiana, Iowa, Illinois, "
    "Michigan, Wisconsin, Minnesota, Ohio, Missouri, Kansas, Nebraska, South Dakota, or North Dakota) "
    "that has a seating capacity between 2,600 and 2,800 seats inclusive. For this identified venue, provide: "
    "(1) The venue name and its location (city and state); "
    "(2) The exact seating capacity of the venue; "
    "(3) The minimum number of wheelchair-accessible spaces required for this venue under the 2010 ADA Standards Section 221.2.1.1, "
    "calculated using the formula: 6 + ⌈(Total Seats - 500) / 150⌉; "
    "(4) A statement of the ADA requirement that venues cannot charge higher prices for accessible seats than for comparable non-accessible seats in the same seating section; "
    "(5) A statement of the ADA requirement that accessible seats must be offered in all price categories available to the public. "
    "Provide reference URLs supporting: (a) the venue's inclusion in the DWTS 2026 February tour schedule, (b) the venue's seating capacity, "
    "and (c) the ADA standards for wheelchair space requirements and ticket pricing."
)

MIDWEST_STATES = {
    "Indiana", "Iowa", "Illinois", "Michigan", "Wisconsin", "Minnesota",
    "Ohio", "Missouri", "Kansas", "Nebraska", "South Dakota", "North Dakota"
}

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Venue identity
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # Schedule evidence (DWTS Feb 2026)
    tour_schedule_urls: List[str] = Field(default_factory=list)

    # Capacity claim and sources
    capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)

    # ADA formula statement (as stated in the answer) and sources (must be ada.gov or access-board.gov ideally)
    ada_formula_statement: Optional[str] = None
    ada_formula_source_urls: List[str] = Field(default_factory=list)

    # ADA computed result as stated in the answer (the minimum wheelchair-accessible spaces)
    ada_calculated_spaces: Optional[str] = None

    # Accessibility pricing/availability statements and sources (official ADA guidance)
    price_parity_statement: Optional[str] = None
    all_price_categories_statement: Optional[str] = None
    accessibility_pricing_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
Extract from the answer exactly ONE venue that the answer claims meets the task. If multiple venues are mentioned, select the first one that the answer uses for detailed info.

Return the following fields (use null for any missing; copy text exactly as in the answer; do not infer):
- venue_name: The venue name selected by the answer.
- city: The city of the selected venue (as stated).
- state: The U.S. state (full name) of the selected venue (as stated).
- tour_schedule_urls: An array of all URLs the answer cites to support that the Dancing With The Stars Live! 2026 Tour has a performance at this venue in February 2026. Include only explicit URLs from the answer.
- capacity: The seating capacity of the selected venue as stated in the answer (e.g., "2,700", "approximately 2,700", "2,700 seats").
- capacity_source_urls: An array of URLs the answer cites to support the venue's seating capacity.
- ada_formula_statement: The exact formula text (if any) the answer provides for computing wheelchair spaces under 2010 ADA Standards Section 221.2.1.1 (e.g., "6 + ceil((Total Seats - 500)/150)").
- ada_formula_source_urls: All URLs (ideally from ada.gov or access-board.gov) the answer cites for the formula/section 221.2.1.1.
- ada_calculated_spaces: The minimum number of wheelchair-accessible spaces the answer computed for this venue (as a string, e.g., "17").
- price_parity_statement: The exact statement (if any) the answer provides asserting that accessible seats cannot be priced higher than comparable non-accessible seats in the same section.
- all_price_categories_statement: The exact statement (if any) the answer provides asserting that accessible seats must be offered in all price categories available to the public.
- accessibility_pricing_source_urls: All URLs (ideally from ada.gov) the answer cites for ADA ticket pricing/availability guidance.

Rules:
- Extract only what is explicitly present in the answer.
- Keep URLs exactly as written; include full URLs.
- If multiple items are present, include them all in their arrays; otherwise, use an empty array.
"""


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def parse_first_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Remove commas and find the first integer-like sequence
    s_clean = s.replace(",", " ")
    m = re.search(r"(\d{3,6})", s_clean)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def compute_ada_min_spaces(total_seats: Optional[int]) -> Optional[int]:
    if total_seats is None:
        return None
    # 6 + ceil((Total Seats - 500)/150). When total seats <= 500, ceil of negative values should be 0 minimum effectively.
    excess = max(total_seats - 500, 0)
    return 6 + math.ceil(excess / 150)


def any_official_ada_url(urls: List[str]) -> bool:
    for u in urls:
        if isinstance(u, str):
            lu = u.lower()
            if "ada.gov" in lu or "access-board.gov" in lu:
                return True
    return False


def union_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, data: VenueExtraction, logger: logging.Logger):
    # Convenience locals
    vname = data.venue_name or ""
    city = data.city or ""
    state = data.state or ""
    schedule_urls = data.tour_schedule_urls or []
    capacity_text = data.capacity or ""
    cap_urls = data.capacity_source_urls or []
    ada_formula_stmt = data.ada_formula_statement or ""
    ada_formula_urls = data.ada_formula_source_urls or []
    ada_calc_spaces_text = data.ada_calculated_spaces or ""
    pricing_urls = data.accessibility_pricing_source_urls or []

    # Parse capacity and compute ADA requirement
    capacity_int = parse_first_int(capacity_text)
    required_wheelchair = compute_ada_min_spaces(capacity_int)

    # Record computed helper info
    evaluator.add_custom_info(
        {
            "parsed_capacity_int": capacity_int,
            "computed_min_wheelchair_spaces": required_wheelchair,
            "midwest_states_reference": sorted(list(MIDWEST_STATES)),
        },
        info_type="helper_info",
        info_name="computed_helper_info",
    )

    # ------------------------ VenueIdentification (parallel, critical) ------------------------
    node_venue = evaluator.add_parallel(
        id="VenueIdentification",
        desc="Identify a venue from the Dancing With The Stars Live! 2026 Tour schedule in February 2026 located in a Midwest state",
        parent=root,
        critical=True,
    )

    # Schedule URL presence (critical) - custom presence check
    evaluator.add_custom_node(
        result=len(schedule_urls) > 0,
        id="ScheduleSourceURL",
        desc="Provide a reference URL from the official DWTS tour website or authorized ticketing platform confirming the venue and date",
        parent=node_venue,
        critical=True,
    )

    # Tour schedule verification (critical) - verify by provided schedule URLs
    tour_sched_leaf = evaluator.add_leaf(
        id="TourScheduleVerification",
        desc="Verify the venue is listed on the official DWTS 2026 tour schedule for February 2026",
        parent=node_venue,
        critical=True,
    )
    tour_claim = (
        f"The Dancing With The Stars Live! 2026 Tour schedule includes a performance at '{vname}' "
        f"in {city}, {state} during February 2026."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_sched_leaf,
        sources=schedule_urls,
        additional_instruction=(
            "Verify that the page(s) clearly indicate Dancing with the Stars Live! (or reasonable variants) has a show at the stated venue "
            "in February 2026. Authorized ticketing platforms (e.g., Ticketmaster, AXS, Live Nation) or the official DWTS site are acceptable. "
            "Minor naming variations like 'Dancing with the Stars: Live!' are acceptable."
        ),
    )

    # Geographic location check (critical)
    geo_leaf = evaluator.add_leaf(
        id="GeographicLocation",
        desc="Verify the venue is located in a Midwest United States state (Indiana, Iowa, Illinois, Michigan, Wisconsin, Minnesota, Ohio, Missouri, Kansas, Nebraska, South Dakota, or North Dakota)",
        parent=node_venue,
        critical=True,
    )
    geo_claim = (
        f"The venue '{vname}' is located in {city}, {state}, and {state} is one of the Midwest states: "
        f"{', '.join(sorted(MIDWEST_STATES))}."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=union_urls(schedule_urls, cap_urls),
        additional_instruction=(
            "Use the provided page(s) to confirm the venue location (city/state). "
            "For the Midwest classification, use the given list as authoritative. "
            "If the page shows the venue is in the specified state and that state appears in the given list, consider this verification correct."
        ),
    )

    # ------------------------ CapacityVerification (parallel, critical) ------------------------
    node_capacity = evaluator.add_parallel(
        id="CapacityVerification",
        desc="Verify the venue's seating capacity falls within the specified range of 2,600 to 2,800 seats",
        parent=root,
        critical=True,
    )

    # Capacity source URL presence (critical)
    evaluator.add_custom_node(
        result=len(cap_urls) > 0,
        id="CapacitySourceURL",
        desc="Provide a reference URL from the venue's official website, Wikipedia, or venue information database confirming the seating capacity",
        parent=node_capacity,
        critical=True,
    )

    # Capacity value correctness (critical) - verify value text at provided URLs
    cap_value_leaf = evaluator.add_leaf(
        id="CapacityValue",
        desc="Provide the exact seating capacity of the venue",
        parent=node_capacity,
        critical=True,
    )
    cap_value_claim = (
        f"The seating capacity of the venue '{vname}' is stated as '{capacity_text}'."
    )
    await evaluator.verify(
        claim=cap_value_claim,
        node=cap_value_leaf,
        sources=cap_urls,
        additional_instruction=(
            "Confirm that the page(s) support the stated seating capacity (accept reasonable textual variants like 'approximately 2,700')."
        ),
    )

    # Capacity range check (critical) - ensure it's between 2600 and 2800 inclusive using the same sources
    cap_range_leaf = evaluator.add_leaf(
        id="CapacityRange",
        desc="Confirm the capacity is between 2,600 and 2,800 seats inclusive",
        parent=node_capacity,
        critical=True,
    )
    cap_range_claim = (
        "The venue's seating capacity is between 2,600 and 2,800 seats inclusive."
    )
    await evaluator.verify(
        claim=cap_range_claim,
        node=cap_range_leaf,
        sources=cap_urls,
        additional_instruction=(
            "From the page(s), determine the venue's capacity and check if it lies within [2600, 2800]. "
            "If the capacity shown is within the range (including endpoints), consider this correct."
        ),
    )

    # ------------------------ ADACalculation (parallel, critical) ------------------------
    node_ada = evaluator.add_parallel(
        id="ADACalculation",
        desc="Calculate the minimum number of wheelchair-accessible spaces required under 2010 ADA Standards Section 221.2.1.1",
        parent=root,
        critical=True,
    )

    # ADA source URL presence for formula (critical) - must be ada.gov or access-board.gov
    evaluator.add_custom_node(
        result=any_official_ada_url(ada_formula_urls),
        id="ADASourceURL",
        desc="Provide a reference URL to the official ADA standards (ada.gov or access-board.gov) documenting the calculation formula from Section 221.2.1.1",
        parent=node_ada,
        critical=True,
    )

    # FormulaApplication (critical) - verify formula statement via ADA sources
    formula_leaf = evaluator.add_leaf(
        id="FormulaApplication",
        desc="Apply the correct ADA formula: 6 + ⌈(Total Seats - 500) / 150⌉",
        parent=node_ada,
        critical=True,
    )
    # If the answer gave a formula statement, use it; otherwise assert the correct formula directly.
    formula_text_for_claim = (
        ada_formula_stmt.strip()
        if ada_formula_stmt.strip()
        else "6 + ceil((Total Seats - 500) / 150)"
    )
    formula_claim = (
        f"Under the 2010 ADA Standards Section 221.2.1.1 for assembly seating, "
        f"the required number of wheelchair spaces is computed as {formula_text_for_claim}."
    )
    await evaluator.verify(
        claim=formula_claim,
        node=formula_leaf,
        sources=ada_formula_urls,
        additional_instruction=(
            "Check the ADA standards text (ada.gov or access-board.gov) for Section 221.2.1.1. "
            "The correct calculation is 6 + ceil((Total Seats - 500)/150). "
            "Allow minor textual/notation variants (e.g., ⌈ ⌉ vs ceil())."
        ),
    )

    # CorrectResult (critical) - ensure the number stated in the answer equals the correct computation
    correct_result_leaf = evaluator.add_leaf(
        id="CorrectResult",
        desc="Provide the correct calculated number of required wheelchair-accessible spaces",
        parent=node_ada,
        critical=True,
    )

    # Build a simple-verify claim that the answer states the computed number
    # If we couldn't compute, still verify equality based on answer text vs computed (will likely fail)
    if required_wheelchair is not None:
        correct_res_claim = (
            f"The answer states that the minimum number of wheelchair-accessible spaces required "
            f"for a venue with {capacity_int} seats is {required_wheelchair}."
        )
    else:
        # Fall back (will probably fail if capacity is missing)
        correct_res_claim = (
            "The answer states the exact minimum number of required wheelchair-accessible spaces, "
            "computed per ADA Section 221.2.1.1."
        )

    await evaluator.verify(
        claim=correct_res_claim,
        node=correct_result_leaf,
        additional_instruction=(
            "Match the numeric value stated in the answer with the computed result using the ADA formula. "
            "Allow minor phrasing like 'at least N' as equivalent to N."
        ),
    )

    # ------------------------ AccessibilityRequirements (parallel, critical) ------------------------
    node_access = evaluator.add_parallel(
        id="AccessibilityRequirements",
        desc="State the ADA requirements for accessible seating pricing and availability",
        parent=root,
        critical=True,
    )

    # Accessibility guidance source presence (critical) - official ADA guidance expected
    evaluator.add_custom_node(
        result=any_official_ada_url(pricing_urls),
        id="AccessibilitySourceURL",
        desc="Provide a reference URL to official ADA guidance on ticket sales and pricing requirements",
        parent=node_access,
        critical=True,
    )

    # Price Parity requirement (critical)
    price_parity_leaf = evaluator.add_leaf(
        id="PriceParity",
        desc="State that venues cannot charge higher prices for accessible seats than for comparable non-accessible seats in the same seating section",
        parent=node_access,
        critical=True,
    )
    price_parity_claim = (
        "Venues cannot charge higher prices for accessible seats than for comparable non-accessible seats in the same seating section."
    )
    await evaluator.verify(
        claim=price_parity_claim,
        node=price_parity_leaf,
        sources=pricing_urls,
        additional_instruction=(
            "Verify using official ADA guidance (e.g., ADA.gov Ticket Sales guidance) that accessible seats cannot be priced higher "
            "than comparable non-accessible seats in the same section."
        ),
    )

    # Price Category Availability requirement (critical)
    price_category_leaf = evaluator.add_leaf(
        id="PriceCategoryAvailability",
        desc="State that accessible seats must be offered in all price categories available to the public",
        parent=node_access,
        critical=True,
    )
    price_category_claim = (
        "Accessible seats must be offered in all price categories available to the public."
    )
    await evaluator.verify(
        claim=price_category_claim,
        node=price_category_leaf,
        sources=pricing_urls,
        additional_instruction=(
            "Verify using official ADA guidance (e.g., ADA.gov Ticket Sales guidance) that accessible seats must be offered "
            "in all price categories available to the public."
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
    # Initialize evaluator with a sequential root to respect task dependency order
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info from the agent's answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, root, extraction, logger)

    # Return structured evaluation summary
    return evaluator.get_summary()