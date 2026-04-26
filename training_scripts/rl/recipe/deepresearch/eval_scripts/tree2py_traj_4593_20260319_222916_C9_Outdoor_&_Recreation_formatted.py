import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "co_front_range_wilderness_trip_2026"
TASK_DESCRIPTION = (
    "You are planning a 4-night wilderness backpacking trip from Bangor, Maine to Colorado's Front Range wilderness "
    "areas, departing on June 15, 2026, and camping June 16-19, 2026. Your trip must meet the specified requirements "
    "including a direct BGR-DEN flight, camping within 2 hours of Denver, at least three different wilderness areas, "
    "inclusion of at least one advance-permit-with-fee area, minimized total permit costs, and compliance with 100-foot "
    "distance-from-water rules. Provide flight info, per-night wilderness details, cost summary, and regulatory compliance, "
    "all with reference URLs."
)

NIGHT_DATES = ["2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19"]


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class FlightInfo(BaseModel):
    airline: Optional[str] = None
    route_statement: Optional[str] = None  # e.g., "non-stop direct BGR to DEN"
    ref_urls: List[str] = Field(default_factory=list)


class NightPlan(BaseModel):
    date: Optional[str] = None
    area_name: Optional[str] = None
    permit_type: Optional[str] = None  # e.g., "advance reservation required", "self-issued at trailhead", "no permit required"
    permit_cost: Optional[str] = None  # keep as string to maximize compatibility; may include "$", "free", etc.
    booking_method: Optional[str] = None  # e.g., "Recreation.gov", "self-issued at trailhead", "USFS office"
    advance_booking: Optional[str] = None  # any booking window or advance reservation details
    distance_regulation: Optional[str] = None  # e.g., "100 feet from water and trails"
    ref_urls: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    flight: Optional[FlightInfo] = None
    nights: List[NightPlan] = Field(default_factory=list)
    total_permit_cost: Optional[str] = None  # stated total in the answer (string is fine)
    compliance_statement: Optional[str] = None  # e.g., "We will camp 100+ feet from water and trails"
    regulatory_urls: List[str] = Field(default_factory=list)  # any global regulatory references


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_trip_plan() -> str:
    return """
    Extract structured information about the proposed Colorado Front Range wilderness backpacking trip plan.

    A) Flight Information:
    - flight.airline: The airline named as operating the direct, non-stop BGR→DEN route (string; null if missing).
    - flight.route_statement: The answer's statement confirming the direct, non-stop BGR→DEN service (string; null if missing).
    - flight.ref_urls: All URLs cited in the answer that are intended to confirm this direct service (array of URLs; [] if none).

    B) For each of the 4 nights (June 16, 17, 18, 19, 2026), extract a NightPlan object:
    - date: The date for that night in ISO format YYYY-MM-DD (e.g., 2026-06-16). If missing, set to null.
    - area_name: The wilderness area name for that night (string; null if missing).
    - permit_type: The stated permit requirement type (e.g., "advance reservation required", "self-issued at trailhead", "no permit required") (string; null if missing).
    - permit_cost: The stated cost/fee for any required permit or reservation for that night (string; use "free" or "$0" if free; null if missing).
    - booking_method: How or where to obtain the permit (e.g., "Recreation.gov", "self-issued at trailhead", "USFS office/website") (string; null if missing).
    - advance_booking: Any advance booking window/timeline details stated (string; null if missing).
    - distance_regulation: The stated camping distance regulation from water/trails (string; null if missing).
    - ref_urls: All URLs cited in the answer that support permits/regulations for that specific area (array of URLs; [] if none).

    C) Cost Summary:
    - total_permit_cost: The total permit/reservation cost for the entire trip as stated in the answer (string; null if missing).

    D) Regulatory Compliance:
    - compliance_statement: Any explicit statement that they will follow 100-foot-from-water/trails rules (string; null if missing).
    - regulatory_urls: Any general regulation URLs cited in the answer beyond per-night references (array of URLs; [] if none).

    IMPORTANT:
    - Return exactly 4 NightPlan entries if possible; if the answer lists more than 4 nights, extract only the first 4 that correspond to June 16–19, 2026. If fewer than 4 are present, return as many as available.
    - Extract only URLs explicitly present in the answer.
    - Do not invent or infer data; use null or empty lists where information is not provided.
    """


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _ensure_four_nights(nights: List[NightPlan]) -> List[NightPlan]:
    # Normalize to exactly 4 entries (pad with empty entries if needed)
    nights = nights[:4]
    while len(nights) < 4:
        nights.append(NightPlan())
    # Fill missing dates with expected ISO dates
    for i, n in enumerate(nights):
        if not n.date:
            n.date = NIGHT_DATES[i]
    return nights


def _union_urls(*url_lists: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for lst in url_lists:
        for u in lst or []:
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _parse_cost_to_float(cost_str: Optional[str]) -> Optional[float]:
    if cost_str is None:
        return None
    s = cost_str.strip().lower()
    if not s:
        return None
    if "free" in s or "$0" in s or "no fee" in s or "0.00" in s:
        return 0.0
    # Replace commas in numbers
    s = s.replace(",", "")
    # Prefer values with $ sign if present
    dollar_amounts = re.findall(r"\$\s*(-?\d+(?:\.\d+)?)", s)
    candidates: List[float] = []
    for amt in dollar_amounts:
        try:
            val = float(amt)
            if val >= 0:
                candidates.append(val)
        except:
            pass
    # If none with $, fall back to any numeric token that doesn't look like a year
    if not candidates:
        all_nums = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", s)
        for tok in all_nums:
            try:
                val = float(tok)
                # Heuristic: ignore 4-digit likely years
                if 1800 <= val <= 2200:
                    continue
                if val >= 0:
                    candidates.append(val)
            except:
                pass
    if not candidates:
        return None
    # Pick the smallest positive value by default (often per-night fee; if there are add-on fees, smallest may represent base permit)
    return min(candidates)


def _sum_costs(costs: List[Optional[str]]) -> Tuple[Optional[float], List[Optional[float]]]:
    parsed: List[Optional[float]] = [_parse_cost_to_float(c) for c in costs]
    if any(v is None for v in parsed):
        return None, parsed
    return float(sum(v for v in parsed if v is not None)), parsed


def _area_name(n: Optional[NightPlan]) -> str:
    return (n.area_name or "").strip()


def _has_any_urls(urls: List[str]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


# -----------------------------------------------------------------------------
# Verification Subtrees
# -----------------------------------------------------------------------------
async def verify_flight(evaluator: Evaluator, parent) -> None:
    flight: Optional[FlightInfo] = None
    # Find extraction
    # The extraction is recorded by evaluator; but we already have it in the calling scope.
    # So we design this function to accept needed info via evaluator.add_custom_info before call.
    # Instead, we'll access from custom info bag if provided; better, we pass in separately.
    # To keep interface stable with rest of code, we implement a thin wrapper in main that calls this with flight.
    raise NotImplementedError("Use verify_flight_with_data instead of verify_flight")


async def verify_flight_with_data(evaluator: Evaluator, parent, flight: Optional[FlightInfo]) -> None:
    flight_node = evaluator.add_parallel(
        id="Flight_Logistics",
        desc="Direct flight from Bangor International Airport to Denver International Airport",
        parent=parent,
        critical=True
    )

    urls = flight.ref_urls if flight else []

    # Direct (non-stop) route verification
    direct_leaf = evaluator.add_leaf(
        id="Direct_Flight_Route",
        desc="Flight must be non-stop service from BGR to DEN",
        parent=flight_node,
        critical=True
    )
    await evaluator.verify(
        claim="There is a direct, non-stop flight service from Bangor International Airport (BGR) to Denver International Airport (DEN).",
        node=direct_leaf,
        sources=urls,
        additional_instruction="Pass only if the provided page(s) explicitly confirm a non-stop/direct BGR–DEN route (seasonal or specific weekdays acceptable). If the page does not clearly confirm a non-stop service, mark as not supported."
    )

    # Airline identification
    airline_name = (flight.airline or "").strip()
    airline_leaf = evaluator.add_leaf(
        id="Airline_Identification",
        desc="Correct airline operating the direct BGR-DEN route must be identified",
        parent=flight_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airline operating the direct BGR–DEN route is '{airline_name}'.",
        node=airline_leaf,
        sources=urls,
        additional_instruction="Verify that the provided page(s) name the airline operating the direct/non-stop BGR–DEN service. Allow minor name variations (e.g., 'United Airlines' vs 'United')."
    )

    # Flight reference URL existence (critical)
    evaluator.add_custom_node(
        result=_has_any_urls(urls),
        id="Flight_Reference_URL",
        desc="URL reference confirming direct flight availability",
        parent=flight_node,
        critical=True
    )


async def verify_single_night(
    evaluator: Evaluator,
    root_parent,
    night: NightPlan,
    night_index: int,
    previous_night: Optional[NightPlan]
) -> None:
    # Night node (parallel, non-critical overall to allow partial credit per night)
    night_titles = [
        "First night camping requirements (June 16, 2026)",
        "Second night camping requirements (June 17, 2026)",
        "Third night camping requirements (June 18, 2026)",
        "Fourth night camping requirements (June 19, 2026)",
    ]
    night_node = evaluator.add_parallel(
        id=f"Night_{night_index+1}_Wilderness_Camping",
        desc=night_titles[night_index],
        parent=root_parent,
        critical=False
    )

    area_name = _area_name(night)
    urls = night.ref_urls if night else []

    # Wilderness_Area_Selection (critical)
    was_node = evaluator.add_parallel(
        id=f"Night_{night_index+1}_Wilderness_Area_Selection",
        desc="Appropriate wilderness area selection and accessibility",
        parent=night_node,
        critical=True
    )

    # Different_Area_Requirement where applicable (Night 2 and Night 3 per rubric)
    if night_index == 1 and previous_night:
        diff_leaf = evaluator.add_leaf(
            id="Night_2_Different_Area_Requirement",
            desc="Selected area must be different from Night 1 wilderness area",
            parent=was_node,
            critical=True
        )
        claim = f"The Night 2 wilderness area '{area_name}' is different from Night 1 area '{_area_name(previous_night)}'."
        await evaluator.verify(
            claim=claim,
            node=diff_leaf,
            additional_instruction="Judge based on the names referring to different wilderness areas; allow minor naming variations or abbreviations."
        )
    if night_index == 2:
        # Compare against Nights 1 and 2 if available
        prev1 = previous_night
        # We'll need night 1 name too; to keep simple, we let caller pass previous two in additional instruction
        # Here we rely on evaluator.add_custom_info to supply, but simpler: we won't include second previous explicitly in claim; we verify difference from both via instruction
        # Instead, we will add explicit claim with two comparisons combined in one sentence; LLM simple check is fine.
        # For that we need Night 1 info; we will attach via additional_instruction context in main call; but since not easily accessible here,
        # we will include only Night 1 if available via evaluator.add_custom_info; For robustness, we'll add a generic claim referencing both Nights 1 and 2 using placeholders from custom info that main supplies.
        pass  # handled below after we fetch from custom_info in main via parameter injection

    # Area_Accessibility
    access_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Area_Accessibility",
        desc="Selected area is accessible per requirement",
        parent=was_node,
        critical=True
    )
    if night_index == 0:
        access_claim = f"The wilderness area '{area_name}' is within approximately a 2-hour drive from Denver, Colorado."
        add_ins = "Use the provided official or authoritative page(s) to judge proximity; if the page explicitly states proximity to Denver or Front Range, accept. If no such info is available on the page, mark unsupported."
        access_sources = urls
    else:
        prev_name = _area_name(previous_night) if previous_night else "the previous night's area"
        access_claim = f"The wilderness area '{area_name}' is reasonably accessible by road from '{prev_name}' for a next-day relocation (typical Front Range driving distances)."
        add_ins = "Use the provided page(s) to infer practical road access between the two areas (not backcountry travel). If clearly infeasible or requires excessive travel beyond a reasonable day transfer, mark unsupported."
        access_sources = _union_urls(urls, previous_night.ref_urls if previous_night else [])
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=access_sources,
        additional_instruction=add_ins
    )

    # Permit_Season_Alignment
    season_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Permit_Season_Alignment",
        desc="If area requires seasonal permits, the night falls within permit period",
        parent=was_node,
        critical=True
    )
    # Extract date string for this night (YYYY-MM-DD)
    date_str = night.date or NIGHT_DATES[night_index]
    await evaluator.verify(
        claim=f"If the wilderness area '{area_name}' has a seasonal overnight permit requirement, that season includes {date_str}; otherwise, there is no seasonal overnight permit requirement.",
        node=season_leaf,
        sources=urls,
        additional_instruction="Pass if the page shows a seasonal permit window that includes the specified date OR the page indicates there is no seasonal overnight permit window. If unclear, mark not supported."
    )

    # Permit_Type_Identification (critical)
    pti_node = evaluator.add_parallel(
        id=f"Night_{night_index+1}_Permit_Type_Identification",
        desc="Correct permit type for the selected wilderness area is identified",
        parent=night_node,
        critical=True
    )
    # Permit Requirement Status
    prs_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Permit_Requirement_Status",
        desc="Correctly identifies whether overnight permit is required, self-issued, or not needed",
        parent=pti_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For '{area_name}', the overnight permit type is stated as '{(night.permit_type or '').strip()}'.",
        node=prs_leaf,
        sources=urls,
        additional_instruction="Verify whether the page supports the stated permit type (advance reservation required vs. self-issued at trailhead vs. not needed)."
    )
    # Permit Authority
    pa_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Permit_Authority",
        desc="Identifies the correct issuing authority (Recreation.gov, self-issued at trailhead, USFS, etc.)",
        parent=pti_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For '{area_name}', permits are obtained via '{(night.booking_method or '').strip()}', which reflects the issuing authority or booking channel.",
        node=pa_leaf,
        sources=urls,
        additional_instruction="Pass if the page supports the stated channel/authority (e.g., Recreation.gov, self-issued at trailhead, USFS office/site)."
    )

    # Permit_Cost_Accuracy (critical)
    pca_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Permit_Cost_Accuracy",
        desc="Accurate permit cost for the selected wilderness area",
        parent=night_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The overnight permit/reservation cost for '{area_name}' is stated as '{(night.permit_cost or '').strip()}', including any required reservation/processing fee.",
        node=pca_leaf,
        sources=urls,
        additional_instruction="Verify the fee on the reference page(s). If the page indicates 'free', '$0', or 'no fee', that is acceptable. If unclear or contradictory, mark not supported."
    )

    # Booking_Procedure (set non-critical to allow a non-critical child inside per rubric intent)
    bp_node = evaluator.add_parallel(
        id=f"Night_{night_index+1}_Booking_Procedure",
        desc="Correct procedure for obtaining the permit is described",
        parent=night_node,
        critical=False
    )
    # Booking_Method (critical child)
    bm_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Booking_Method",
        desc="Identifies whether permit is obtained online, at trailhead, or in-person",
        parent=bp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The permit for '{area_name}' is obtained '{(night.booking_method or '').strip()}' (e.g., online via Recreation.gov, self-issued at trailhead, in-person/USFS).",
        node=bm_leaf,
        sources=urls,
        additional_instruction="Verify the method on the page(s). Allow minor phrasing variations."
    )
    # Advance_Booking_Requirements (non-critical child)
    abr_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Advance_Booking_Requirements",
        desc="If advance booking is required, the booking window or timeline is correctly stated",
        parent=bp_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The stated advance booking/timeline details for '{area_name}' are correct: '{(night.advance_booking or '').strip()}'. If no advance booking is required, this statement may be minimal or N/A.",
        node=abr_leaf,
        sources=urls,
        additional_instruction="Pass if the page supports the stated booking window/timeline. If no advance booking is required and the answer does not claim such a window, this can still pass."
    )

    # Camping_Distance_Regulations (critical)
    cdr_leaf = evaluator.add_leaf(
        id=f"Night_{night_index+1}_Camping_Distance_Regulations",
        desc="Camping distance requirements from water sources are correctly stated",
        parent=night_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The camping distance regulation for '{area_name}' is correctly stated as '{(night.distance_regulation or '').strip()}', typically requiring at least 100 feet from lakes/streams/trails.",
        node=cdr_leaf,
        sources=urls,
        additional_instruction="Verify that the page includes a camping distance-from-water/trails rule consistent with the stated regulation."
    )

    # Night Reference URL existence (critical)
    evaluator.add_custom_node(
        result=_has_any_urls(urls),
        id=f"Night_{night_index+1}_Reference_URL",
        desc="URL reference supporting wilderness area permit and regulation information",
        parent=night_node,
        critical=True
    )


async def verify_costs_and_optimization(
    evaluator: Evaluator,
    parent,
    nights: List[NightPlan],
    stated_total: Optional[str]
) -> None:
    # Cost Optimization main node (critical overall)
    cost_node = evaluator.add_parallel(
        id="Cost_Optimization",
        desc="Total permit costs are minimized while meeting all requirements",
        parent=parent,
        critical=True
    )

    all_urls = _union_urls(*[n.ref_urls for n in nights if n])

    # At least one paid advance-permit area (critical)
    paid_leaf = evaluator.add_leaf(
        id="At_Least_One_Paid_Permit_Area",
        desc="Trip includes at least one wilderness area that requires advance overnight permits with fees",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one selected wilderness area requires an advance overnight permit that must be reserved ahead of time and includes a fee (not a free self-issued permit).",
        node=paid_leaf,
        sources=all_urls,
        additional_instruction="Pass if any referenced page clearly shows an advance, paid, overnight permitting system (e.g., Indian Peaks Wilderness)."
    )

    # Total cost calculation (critical)
    total_cost_leaf = evaluator.add_leaf(
        id="Total_Cost_Calculation",
        desc="Total permit costs are accurately calculated by summing all individual permit fees",
        parent=cost_node,
        critical=True
    )
    nightly_cost_strs = [(n.permit_cost or "").strip() for n in nights]
    summed, parsed_list = _sum_costs(nightly_cost_strs)

    parsed_display = [("None" if v is None else f"{v:.2f}") for v in parsed_list]
    claimed_total_val = _parse_cost_to_float(stated_total)
    claimed_display = "None" if claimed_total_val is None else f"{claimed_total_val:.2f}"
    sum_display = "None" if summed is None else f"{summed:.2f}"

    total_claim = (
        f"The stated total permit cost for the trip is '{(stated_total or '').strip()}'. "
        f"The parsed nightly costs are {parsed_display}, which sum to {sum_display}. "
        f"Therefore, the computed sum equals the stated total ({claimed_display})."
    )
    await evaluator.verify(
        claim=total_claim,
        node=total_cost_leaf,
        additional_instruction="Judge whether the arithmetic is correct based on the parsed numeric values. "
                              "If any nightly cost cannot be parsed, or the totals do not match, mark as incorrect."
    )

    # Cost minimization strategy (non-critical in rubric; however, parent is critical, which requires critical children. Adjust to critical to satisfy framework constraint.)
    minimize_leaf = evaluator.add_leaf(
        id="Cost_Minimization_Strategy",
        desc="Selection demonstrates cost minimization by prioritizing free or self-issued permits where appropriate",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan minimizes total permit/reservation costs by using free or self-issued permits where appropriate while still including at least one required advance paid-permit night.",
        node=minimize_leaf,
        sources=all_urls,
        additional_instruction="Pass if most selected nights are free/self-issued, as evidenced by the referenced pages; minor reservation fees on one night are acceptable."
    )


async def verify_regulatory_compliance(
    evaluator: Evaluator,
    parent,
    nights: List[NightPlan],
    compliance_urls: List[str],
    compliance_statement: Optional[str],
) -> None:
    reg_node = evaluator.add_parallel(
        id="Regulatory_Compliance_Summary",
        desc="Overall compliance with camping regulations across all nights",
        parent=parent,
        critical=True
    )

    all_urls = _union_urls(compliance_urls, *[n.ref_urls for n in nights if n])

    # Universal 100-foot rule (critical)
    rule_leaf = evaluator.add_leaf(
        id="Universal_Distance_Rule",
        desc="All campsites must maintain minimum 100-foot distance from water sources (lakes, streams, trails)",
        parent=reg_node,
        critical=True
    )
    rule_claim = (
        "The plan confirms compliance with the universal rule requiring camping at least 100 feet from lakes, "
        "streams, and trails, and the referenced pages corroborate this requirement."
    )
    await evaluator.verify(
        claim=rule_claim,
        node=rule_leaf,
        sources=all_urls,
        additional_instruction="Pass if at least one reference page explicitly mentions a 100-foot (or similar) distance rule for camping, and the plan's statement is consistent."
    )

    # Group size compliance (adjusted to critical to satisfy framework constraint)
    group_leaf = evaluator.add_leaf(
        id="Group_Size_Compliance",
        desc="If any selected wilderness area has group size limits, compliance is acknowledged",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan acknowledges and will comply with any posted group size limits where applicable for the selected wilderness areas.",
        node=group_leaf,
        sources=all_urls,
        additional_instruction="Pass if at least one page mentions group size limits and the plan indicates compliance or does not contradict such limits."
    )

    # Permit carrying requirement (adjusted to critical to satisfy framework constraint)
    carry_leaf = evaluator.add_leaf(
        id="Permit_Carrying_Requirement",
        desc="Understanding that wilderness permits must be carried at all times is demonstrated",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan understands that wilderness permits (when required) must be carried at all times while in the wilderness.",
        node=carry_leaf,
        sources=all_urls,
        additional_instruction="Pass if any referenced page states that permits must be carried/displayed and the plan does not contradict this."
    )


# -----------------------------------------------------------------------------
# Main Evaluation
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root is non-critical by default; strategy per rubric root parallel)
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
    trip: TripPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Normalize nights
    nights = _ensure_four_nights(trip.nights)

    # Record some helpful info
    evaluator.add_custom_info(
        info={
            "normalized_nights": [n.dict() for n in nights],
            "expected_dates": NIGHT_DATES
        },
        info_type="normalization",
        info_name="normalized_night_data"
    )

    # Build verification tree according to rubric (with minor criticality adjustments for framework consistency)

    # Flight logistics
    await verify_flight_with_data(evaluator, root, trip.flight or FlightInfo())

    # Nights 1-4
    prev_night: Optional[NightPlan] = None
    for i in range(4):
        await verify_single_night(evaluator, root, nights[i], i, prev_night)

        # Special handling for Night 3 "Different_Area_Requirement" vs both previous nights
        if i == 2:
            n1 = nights[0]
            n2 = nights[1]
            area3 = _area_name(nights[2])
            diff_three_leaf_parent = evaluator.find_node(f"Night_{i+1}_Wilderness_Area_Selection")
            if diff_three_leaf_parent is None:
                diff_three_leaf_parent = evaluator.add_parallel(
                    id=f"Night_{i+1}_Wilderness_Area_Selection_extra",
                    desc="Night 3 area differentiation check (extra)",
                    parent=evaluator.find_node(f"Night_{i+1}_Wilderness_Camping"),
                    critical=True
                )
            diff3_leaf = evaluator.add_leaf(
                id="Night_3_Different_Area_Requirement",
                desc="Selected area must be different from Night 1 and Night 2 wilderness areas",
                parent=diff_three_leaf_parent,
                critical=True
            )
            claim = (
                f"The Night 3 wilderness area '{area3}' is different from Night 1 area '{_area_name(n1)}' "
                f"and Night 2 area '{_area_name(n2)}'."
            )
            await evaluator.verify(
                claim=claim,
                node=diff3_leaf,
                additional_instruction="Judge whether these are different wilderness areas by name; allow minor variations/abbreviations."
            )

        prev_night = nights[i]

    # Cost optimization and totals
    await verify_costs_and_optimization(evaluator, root, nights, trip.total_permit_cost)

    # Regulatory compliance summary
    await verify_regulatory_compliance(
        evaluator,
        root,
        nights,
        trip.regulatory_urls if trip and trip.regulatory_urls else [],
        trip.compliance_statement or ""
    )

    # Return summary
    return evaluator.get_summary()