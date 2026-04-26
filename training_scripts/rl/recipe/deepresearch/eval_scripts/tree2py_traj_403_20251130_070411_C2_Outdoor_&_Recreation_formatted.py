import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "grand_canyon_backcountry_march2026"
TASK_DESCRIPTION = (
    "I'm planning a backcountry hiking trip to Grand Canyon National Park with 3 friends (4 people total) for 3 days "
    "and 2 nights, with camping below the rim. We want to start our trip on March 10, 2026. Based on current "
    "information as of late November 2025: (1) Which rim of the Grand Canyon should we plan to access for this trip "
    "date, and why? (2) When is the lottery application window (start date and deadline with time zone) to apply for "
    "a backcountry permit for this trip? (3) What is the total cost for the backcountry permit for our group? Please "
    "provide specific dates, costs, and reference URLs from official sources to support your answer."
)

# Ground truth and constants used for arithmetic checks
EXPECTED_RIM = "South Rim"
GROUP_SIZE = 4
NIGHTS = 2
APPLICATION_FEE_USD = 10.0
PPP_NIGHT_FEE_USD = 15.0
EXPECTED_TOTAL_USD = APPLICATION_FEE_USD + PPP_NIGHT_FEE_USD * GROUP_SIZE * NIGHTS  # 10 + 15*4*2 = 130


class TripPlanningExtraction(BaseModel):
    rim: Optional[str] = None
    rim_reasoning: Optional[str] = None

    application_start_date: Optional[str] = None
    application_deadline_with_timezone: Optional[str] = None

    application_fee: Optional[str] = None
    per_person_per_night_fee: Optional[str] = None
    reported_total_cost: Optional[str] = None

    urls: List[str] = Field(default_factory=list)


def prompt_extract_trip_planning() -> str:
    return (
        "Extract the specific trip-planning details stated in the answer for the Grand Canyon backcountry trip. "
        "Return a JSON object with the following fields:\n"
        "1. rim: The rim the answer says should be accessed for March 10, 2026 (e.g., 'South Rim', 'North Rim').\n"
        "2. rim_reasoning: The stated rationale for the rim choice (e.g., North Rim seasonal closure, South Rim open year‑round).\n"
        "3. application_start_date: The lottery application opening date relevant to March 2026 start dates.\n"
        "4. application_deadline_with_timezone: The lottery application deadline including the time and time zone.\n"
        "5. application_fee: The non‑refundable application fee amount used in the answer (as written in the answer).\n"
        "6. per_person_per_night_fee: The per‑person per‑night fee used in the answer for below‑rim camping (as written in the answer).\n"
        "7. reported_total_cost: The total backcountry permit cost for the group as reported in the answer.\n"
        "8. urls: Array of all reference URLs included in the answer text (extract actual URL strings; include http/https).\n"
        "If any field is missing in the answer, set it to null (or an empty array for urls). Do not invent information."
    )


def _parse_currency_to_float(text: Optional[str]) -> Optional[float]:
    """Parse the first monetary number in a string to float."""
    if not text:
        return None
    # Remove common currency symbols and commas
    cleaned = text.replace(",", "")
    # Find the first number (integer or decimal)
    match = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _is_official_domain(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        # Accept subdomains of nps.gov and recreation.gov
        return host.endswith("nps.gov") or host.endswith("recreation.gov")
    except Exception:
        return False


async def build_rim_accessibility_checks(
    evaluator: Evaluator,
    parent_node,
    data: TripPlanningExtraction,
) -> None:
    rim_node = evaluator.add_parallel(
        id="rim_accessibility",
        desc="Verify that the answer identifies the appropriate rim for March 10, 2026 and explains why, using the provided rim access constraints.",
        parent=parent_node,
        critical=False,
    )

    # Leaf: rim_identification (critical)
    rim_ident_leaf = evaluator.add_leaf(
        id="rim_identification",
        desc="Answer clearly states which rim should be used for the March 10, 2026 trip date, consistent with the provided closure/open constraints.",
        parent=rim_node,
        critical=True,
    )

    rim_str = data.rim or ""
    claim_rim = (
        f"The rim selected in the answer ('{rim_str}') is the South Rim for a March 10, 2026 start date."
    )
    await evaluator.verify(
        claim=claim_rim,
        node=rim_ident_leaf,
        additional_instruction=(
            "Check the answer text to confirm that the rim indicated is the South Rim for early‑March access. "
            "Allow minor naming variants like 'South Rim (Grand Canyon Village)'. "
            "Do not rely on your own knowledge; use the answer context to judge whether the rim choice is stated."
        ),
    )

    # Leaf: rim_reasoning (critical)
    rim_reason_leaf = evaluator.add_leaf(
        id="rim_reasoning",
        desc="Answer provides a rationale grounded in the provided constraints (e.g., North Rim closed until anticipated May 15, 2026; South Rim open year-round).",
        parent=rim_node,
        critical=True,
    )

    reason_str = data.rim_reasoning or ""
    claim_reason = (
        "The answer's stated rationale for the rim choice is grounded in the constraints that the North Rim is closed "
        "until approximately May 15, 2026 and the South Rim is open year‑round."
    )
    await evaluator.verify(
        claim=claim_reason,
        node=rim_reason_leaf,
        sources=data.urls,  # If provided, use official pages to support the closure/availability statements
        additional_instruction=(
            "Use the answer text to judge whether it mentions North Rim seasonal closure (until ~May 15) and South Rim year‑round access. "
            "If URLs are provided, confirm that this rationale aligns with those official sources."
        ),
    )


async def build_lottery_window_checks(
    evaluator: Evaluator,
    parent_node,
    data: TripPlanningExtraction,
) -> None:
    lot_node = evaluator.add_parallel(
        id="lottery_application_window",
        desc="Verify that the lottery application window for March 2026 start dates is correctly reported per the provided constraints.",
        parent=parent_node,
        critical=False,
    )

    # Leaf: application_start_date (critical)
    start_leaf = evaluator.add_leaf(
        id="application_start_date",
        desc="Answer states the correct lottery application opening date for March 2026 start dates.",
        parent=lot_node,
        critical=True,
    )
    start_str = data.application_start_date or ""
    start_claim = f"The lottery application opening date for March 2026 start dates is '{start_str}'."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=data.urls,
        additional_instruction=(
            "Verify the stated opening date against official NPS or Recreation.gov pages describing the Backcountry Permit Early Access Lottery. "
            "Match month/year and the specific date for March 2026 start dates."
        ),
    )

    # Leaf: application_deadline_with_timezone (critical)
    deadline_leaf = evaluator.add_leaf(
        id="application_deadline_with_timezone",
        desc="Answer states the correct lottery application deadline including the time and time zone (as required).",
        parent=lot_node,
        critical=True,
    )
    deadline_str = data.application_deadline_with_timezone or ""
    deadline_claim = (
        f"The lottery application deadline for March 2026 start dates, including time and time zone, is '{deadline_str}'."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=data.urls,
        additional_instruction=(
            "Check official pages (NPS or Recreation.gov) to confirm the deadline includes the time and time zone for the lottery window applicable to March 2026 start dates."
        ),
    )


async def build_permit_cost_checks(
    evaluator: Evaluator,
    parent_node,
    data: TripPlanningExtraction,
) -> None:
    cost_node = evaluator.add_parallel(
        id="permit_cost",
        desc="Verify that the total permit cost is correctly computed for 4 people, 2 nights, below-rim camping, using the provided fee structure constraints.",
        parent=parent_node,
        critical=False,
    )

    # Leaf: fee_components_used (critical)
    fee_components_leaf = evaluator.add_leaf(
        id="fee_components_used",
        desc="Answer uses the correct fee components from constraints: $10 non-refundable application fee and $15 per person per night for below-rim camping.",
        parent=cost_node,
        critical=True,
    )
    fee_claim = (
        "The answer uses a $10 non‑refundable application fee and a $15 per‑person per‑night fee for below‑rim camping."
    )
    await evaluator.verify(
        claim=fee_claim,
        node=fee_components_leaf,
        additional_instruction=(
            "Check the answer text for the exact fee components: $10 application fee and $15 per person per night for below‑rim camping."
        ),
    )

    # Leaf: group_nights_application (critical)
    group_apply_leaf = evaluator.add_leaf(
        id="group_nights_application",
        desc="Answer correctly applies trip parameters (4 people, 2 nights) to the per-person-per-night fee and shows or implies correct arithmetic.",
        parent=cost_node,
        critical=True,
    )
    arithmetic_claim = (
        f"Using $15 per person per night for {GROUP_SIZE} people and {NIGHTS} nights, the subtotal is ${PPP_NIGHT_FEE_USD * GROUP_SIZE * NIGHTS:.0f}; "
        f"adding the $10 application fee yields a total of ${EXPECTED_TOTAL_USD:.0f}, and the answer shows or implies this arithmetic."
    )
    await evaluator.verify(
        claim=arithmetic_claim,
        node=group_apply_leaf,
        additional_instruction=(
            "Confirm that the answer applies 4 people and 2 nights to the $15 per‑person per‑night fee (15×4×2=120) and adds $10, totaling $130. "
            "Minor formatting differences are acceptable."
        ),
    )

    # Leaf: total_cost_reported (critical) implemented as a custom arithmetic check
    reported_total = _parse_currency_to_float(data.reported_total_cost)
    totals_match = (reported_total is not None) and (abs(reported_total - EXPECTED_TOTAL_USD) < 0.01)
    evaluator.add_custom_node(
        result=totals_match,
        id="total_cost_reported",
        desc="Answer reports a total permit cost consistent with the fee structure and trip parameters.",
        parent=cost_node,
        critical=True,
    )


async def build_official_urls_checks(
    evaluator: Evaluator,
    parent_node,
    data: TripPlanningExtraction,
) -> None:
    urls_node = evaluator.add_parallel(
        id="official_reference_urls",
        desc="Verify that the answer provides official reference URLs as required.",
        parent=parent_node,
        critical=True,  # Critical aggregator; all children must also be critical
    )

    # Leaf: urls_provided (critical)
    evaluator.add_custom_node(
        result=bool(data.urls),
        id="urls_provided",
        desc="At least one reference URL is included in the answer.",
        parent=urls_node,
        critical=True,
    )

    # Leaf: urls_official_domains (critical)
    all_official = all(_is_official_domain(u) for u in data.urls) if data.urls else False
    evaluator.add_custom_node(
        result=all_official,
        id="urls_official_domains",
        desc="All provided reference URLs are from official nps.gov and/or recreation.gov domains (per constraints).",
        parent=urls_node,
        critical=True,
    )


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
    data = await evaluator.extract(
        prompt=prompt_extract_trip_planning(),
        template_class=TripPlanningExtraction,
        extraction_name="trip_planning_extraction",
    )

    # Ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_rim": EXPECTED_RIM,
        "group_size": GROUP_SIZE,
        "nights": NIGHTS,
        "fee_components": {
            "application_fee_usd": APPLICATION_FEE_USD,
            "per_person_per_night_fee_usd": PPP_NIGHT_FEE_USD,
        },
        "expected_total_usd": EXPECTED_TOTAL_USD,
    }, gt_type="expected_parameters")

    # Build subtrees
    await build_rim_accessibility_checks(evaluator, root, data)
    await build_lottery_window_checks(evaluator, root, data)
    await build_permit_cost_checks(evaluator, root, data)
    await build_official_urls_checks(evaluator, root, data)

    # Summary result
    return evaluator.get_summary()