import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wonderland_trail_2026_permits"
TASK_DESCRIPTION = (
    "For a group planning to backpack the complete 93-mile Wonderland Trail loop around Mount Rainier in Washington "
    "during the 2026 summer season, provide comprehensive permit information including: (1) the complete early-access "
    "lottery timeline with specific dates and times for when applications open, when they close, when results are "
    "announced, and when the general on-sale period begins; (2) the complete fee structure including both the "
    "non-refundable reservation fee amount and the per-person per-night recreation fee; and (3) the key trip planning "
    "constraints including the maximum daily mileage limit allowed on Recreation.gov for advance reservations, the "
    "maximum party size permitted for standard trailside camps, and the minimum number of days in advance that "
    "reservations must be made before the trip start date. All information must be specific to the 2026 season and "
    "supported by official sources from Recreation.gov or the National Park Service."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TimelineInfo(BaseModel):
    application_open: Optional[str] = None
    application_close: Optional[str] = None
    results_start: Optional[str] = None
    general_on_sale: Optional[str] = None


class FeeInfo(BaseModel):
    reservation_fee: Optional[str] = None
    recreation_fee_per_person_per_night: Optional[str] = None
    effective_start_note: Optional[str] = None
    youth_fee_exemption: Optional[str] = None


class ConstraintsInfo(BaseModel):
    max_daily_mileage_limit: Optional[str] = None
    standard_trailside_camp_max_party_size: Optional[str] = None
    group_camp_requirement: Optional[str] = None
    minimum_advance_reservation_window_days: Optional[str] = None
    reservable_date_range: Optional[str] = None


class WonderlandPermitExtraction(BaseModel):
    timeline: Optional[TimelineInfo] = None
    fees: Optional[FeeInfo] = None
    constraints: Optional[ConstraintsInfo] = None
    official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wonderland_permit_info() -> str:
    return """
    Extract structured permit information for Mount Rainier's Wonderland Trail for the 2026 season exactly as stated in the answer.

    Return a JSON object with the following fields:

    timeline:
      - application_open: The stated date/time when early-access lottery applications open (include timezone wording like "Pacific Time/PT" if present).
      - application_close: The stated date/time when early-access lottery applications close (include timezone wording if present).
      - results_start: The stated date/time when lottery result notifications start (include timezone wording if present).
      - general_on_sale: The stated date/time when remaining inventory goes on sale (include timezone wording if present).

    fees:
      - reservation_fee: The stated non-refundable reservation/application fee amount (e.g., "$6.00").
      - recreation_fee_per_person_per_night: The stated per-person per-night recreation fee amount (e.g., "$10 per person per night").
      - effective_start_note: Any stated note about when the recreation fee policy became effective (e.g., "starting with the February 2025 lottery").
      - youth_fee_exemption: The stated youth fee exemption text (e.g., "15 and under free").

    constraints:
      - max_daily_mileage_limit: The stated maximum daily mileage limit for advance reservations (e.g., "17.5 trail miles").
      - standard_trailside_camp_max_party_size: The stated maximum party size/tent count for standard trailside camps (e.g., "5 people and 3 tents").
      - group_camp_requirement: The stated rule for larger parties requiring group camps (e.g., "6–12 people must use designated group camps").
      - minimum_advance_reservation_window_days: The stated minimum advance window for reservations (e.g., "at least 2 days in advance").
      - reservable_date_range: The stated reservable date range (e.g., "June 1 to October 10").

    official_urls:
      - Extract ONLY official citation URLs explicitly mentioned in the answer from recreation.gov or nps.gov (National Park Service domains).
      - Valid formats include plain URLs or markdown links; extract actual URLs.
      - If a URL is missing a protocol, prepend "http://".
      - Exclude non-official URLs.
      - Deduplicate and return as a list.

    General rules:
      - Do not invent or infer values; if any required information is missing, return null for that field.
      - Preserve the exact wording as stated in the answer for each field (including currency symbols, hyphens, en dashes).
      - If multiple values are stated for a field, choose the most specific/fully specified one.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_official_urls(urls: List[str]) -> List[str]:
    """Return only recreation.gov or nps.gov URLs, normalized, deduplicated."""
    result: List[str] = []
    seen: set = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        # Ensure protocol
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        try:
            parsed = urlparse(u)
            host = (parsed.netloc or "").lower()
            if "recreation.gov" in host or "nps.gov" in host:
                if u not in seen:
                    result.append(u)
                    seen.add(u)
        except Exception:
            continue
    return result


def _add_failed_leaf(
    evaluator: Evaluator,
    leaf_id: str,
    desc: str,
    parent: VerificationNode,
    critical: bool = True,
) -> VerificationNode:
    """Add a failed leaf (used when the answer omitted the required info)."""
    return evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
        score=0.0,
        status="failed",
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_official_source_support(
    evaluator: Evaluator,
    parent: VerificationNode,
    official_urls: List[str],
) -> VerificationNode:
    """Official source support: ensure at least one valid Recreation.gov/NPS URL is provided."""
    node = evaluator.add_parallel(
        id="Official_Source_Support",
        desc="All claims are supported by official sources from Recreation.gov or the National Park Service (nps.gov).",
        parent=parent,
        critical=True,
    )

    filtered = filter_official_urls(official_urls)

    # Leaf: Uses_Official_URLs (existence check via custom node)
    evaluator.add_custom_node(
        result=len(filtered) > 0,
        id="Uses_Official_URLs",
        desc="Provides at least one valid citation URL from Recreation.gov and/or nps.gov that supports the provided permit information.",
        parent=node,
        critical=True,
    )

    # Record which URLs we will use uniformly across verifications
    evaluator.add_custom_info({"official_urls_used": filtered}, info_type="url_list", info_name="official_sources")

    return node


async def build_timeline_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    timeline: Optional[TimelineInfo],
    official_urls: List[str],
) -> VerificationNode:
    """Early Access Lottery Timeline 2026"""
    node = evaluator.add_parallel(
        id="Early_Access_Lottery_Timeline_2026",
        desc="Accurate dates and times for the 2026 early-access lottery application period, results notification start, and general on-sale opening.",
        parent=parent,
        critical=True,
    )

    urls = filter_official_urls(official_urls)

    # Application Opening
    if not timeline or not timeline.application_open:
        _add_failed_leaf(
            evaluator,
            "Application_Opening_Date_Time",
            "States that early-access lottery applications open on February 10, 2026 at 7:00 am Pacific Time.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Application_Opening_Date_Time",
            desc="States that early-access lottery applications open on February 10, 2026 at 7:00 am Pacific Time.",
            parent=node,
            critical=True,
        )
        claim = f"Early-access lottery applications open on {timeline.application_open}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify that the official Recreation.gov or NPS page for Mount Rainier Wonderland Trail permits "
                "shows this opening date/time for the 2026 season. Accept minor timezone phrasing variations "
                "like PT/PST/PDT as Pacific Time."
            ),
        )

    # Application Closing
    if not timeline or not timeline.application_close:
        _add_failed_leaf(
            evaluator,
            "Application_Closing_Date_Time",
            "States that early-access lottery applications close on March 3, 2026 at 7:00 pm Pacific Time.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Application_Closing_Date_Time",
            desc="States that early-access lottery applications close on March 3, 2026 at 7:00 pm Pacific Time.",
            parent=node,
            critical=True,
        )
        claim = f"Early-access lottery applications close on {timeline.application_close}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Check the official page to confirm the closing date/time for the 2026 lottery application window. "
                "Timezone wording may vary slightly; treat PT/PST/PDT as Pacific Time."
            ),
        )

    # Results Notification Start
    if not timeline or not timeline.results_start:
        _add_failed_leaf(
            evaluator,
            "Results_Notification_Start_Date_Time",
            "States that lottery participants are notified starting March 14, 2026 at 7:00 am Pacific Time.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Results_Notification_Start_Date_Time",
            desc="States that lottery participants are notified starting March 14, 2026 at 7:00 am Pacific Time.",
            parent=node,
            critical=True,
        )
        claim = f"Lottery participants are notified starting {timeline.results_start}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Confirm on an official source that result notifications start at the stated date/time for 2026."
            ),
        )

    # General On-Sale Opening
    if not timeline or not timeline.general_on_sale:
        _add_failed_leaf(
            evaluator,
            "General_On_Sale_Date_Time",
            "States that all remaining reservable inventory becomes available on Recreation.gov starting April 25, 2026 at 7:00 am Pacific Time.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="General_On_Sale_Date_Time",
            desc="States that all remaining reservable inventory becomes available on Recreation.gov starting April 25, 2026 at 7:00 am Pacific Time.",
            parent=node,
            critical=True,
        )
        claim = f"All remaining reservable inventory becomes available on Recreation.gov starting {timeline.general_on_sale}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify the on-sale opening date/time for remaining inventory for the 2026 season in official sources."
            ),
        )

    return node


async def build_fee_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    fees: Optional[FeeInfo],
    official_urls: List[str],
) -> VerificationNode:
    """Permit Fee Structure"""
    node = evaluator.add_parallel(
        id="Permit_Fee_Structure",
        desc="Complete and accurate fee information for reservations and wilderness camping recreation fees, including exemptions and effective-start note from constraints.",
        parent=parent,
        critical=True,
    )

    urls = filter_official_urls(official_urls)

    # Non-refundable reservation fee
    if not fees or not fees.reservation_fee:
        _add_failed_leaf(
            evaluator,
            "Non_Refundable_Reservation_Fee",
            "States that each early-access lottery application or permit reservation has a non-refundable $6.00 fee.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Non_Refundable_Reservation_Fee",
            desc="States that each early-access lottery application or permit reservation has a non-refundable $6.00 fee.",
            parent=node,
            critical=True,
        )
        claim = f"The non-refundable reservation/application fee is {fees.reservation_fee}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Check official Recreation.gov/NPS sources for the application/reservation fee amount for Wonderland Trail permits."
            ),
        )

    # Recreation fee per person per night
    if not fees or not fees.recreation_fee_per_person_per_night:
        _add_failed_leaf(
            evaluator,
            "Recreation_Fee_Per_Person_Per_Night",
            "States that the recreation fee is $10 per person per night for wilderness camping permits.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Recreation_Fee_Per_Person_Per_Night",
            desc="States that the recreation fee is $10 per person per night for wilderness camping permits.",
            parent=node,
            critical=True,
        )
        claim = f"The wilderness camping recreation fee is {fees.recreation_fee_per_person_per_night}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Confirm the per-person per-night recreation fee for Mount Rainier wilderness permits on official sources."
            ),
        )

    # Effective start note
    if not fees or not fees.effective_start_note:
        _add_failed_leaf(
            evaluator,
            "Recreation_Fee_Effective_Start_Note",
            "Includes the constraint note that the $10 per person per night recreation fee applies starting with the February 2025 lottery.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Recreation_Fee_Effective_Start_Note",
            desc="Includes the constraint note that the $10 per person per night recreation fee applies starting with the February 2025 lottery.",
            parent=node,
            critical=True,
        )
        claim = f"The stated effective-start note for the recreation fee policy is: {fees.effective_start_note}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify that official sources indicate the effective start timing for the $10 per person per night fee (e.g., starting with the February 2025 lottery)."
            ),
        )

    # Youth fee exemption
    if not fees or not fees.youth_fee_exemption:
        _add_failed_leaf(
            evaluator,
            "Youth_Fee_Exemption",
            "States that youth aged 15 and under camp for free (no recreation fee).",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Youth_Fee_Exemption",
            desc="States that youth aged 15 and under camp for free (no recreation fee).",
            parent=node,
            critical=True,
        )
        claim = f"The youth exemption policy is: {fees.youth_fee_exemption}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Confirm that official sources state that youth 15 and under camp for free (no recreation fee)."
            ),
        )

    return node


async def build_constraints_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    constraints: Optional[ConstraintsInfo],
    official_urls: List[str],
) -> VerificationNode:
    """Trip Planning Constraints"""
    node = evaluator.add_parallel(
        id="Trip_Planning_Constraints",
        desc="Key planning restrictions from constraints, including daily mileage limit, party size limits, group-camp requirement, advance-booking minimum, and reservable date range.",
        parent=parent,
        critical=True,
    )

    urls = filter_official_urls(official_urls)

    # Maximum daily mileage limit
    if not constraints or not constraints.max_daily_mileage_limit:
        _add_failed_leaf(
            evaluator,
            "Maximum_Daily_Mileage_Limit",
            "States that the maximum daily mileage between camps is limited to 17.5 trail miles on Recreation.gov for advance reservations.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Maximum_Daily_Mileage_Limit",
            desc="States that the maximum daily mileage between camps is limited to 17.5 trail miles on Recreation.gov for advance reservations.",
            parent=node,
            critical=True,
        )
        claim = f"The maximum daily mileage between camps for advance reservations is {constraints.max_daily_mileage_limit}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify that official sources indicate the daily mileage limit enforced by Recreation.gov for advance reservations."
            ),
        )

    # Standard trailside camp max party size
    if not constraints or not constraints.standard_trailside_camp_max_party_size:
        _add_failed_leaf(
            evaluator,
            "Standard_Trailsidе_Camp_Max_Party_Size",
            "States that the maximum party size is 5 people and 3 tents for standard trailside camps.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Standard_Trailsidе_Camp_Max_Party_Size",
            desc="States that the maximum party size is 5 people and 3 tents for standard trailside camps.",
            parent=node,
            critical=True,
        )
        claim = f"The maximum party size/tent limit for standard trailside camps is {constraints.standard_trailside_camp_max_party_size}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Confirm on official sources the maximum party size and tent count for standard trailside camps."
            ),
        )

    # Group camp requirement for larger parties
    if not constraints or not constraints.group_camp_requirement:
        _add_failed_leaf(
            evaluator,
            "Group_Camp_Requirement_For_Larger_Parties",
            "States that parties of 6–12 people must camp in designated group camps.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Group_Camp_Requirement_For_Larger_Parties",
            desc="States that parties of 6–12 people must camp in designated group camps.",
            parent=node,
            critical=True,
        )
        claim = f"The rule for larger parties is: {constraints.group_camp_requirement}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify official sources specify that parties in the 6–12 range must use designated group camps."
            ),
        )

    # Minimum advance reservation window
    if not constraints or not constraints.minimum_advance_reservation_window_days:
        _add_failed_leaf(
            evaluator,
            "Minimum_Advance_Reservation_Window",
            "States that reservations must be made at least 2 days in advance of the trip start date.",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Minimum_Advance_Reservation_Window",
            desc="States that reservations must be made at least 2 days in advance of the trip start date.",
            parent=node,
            critical=True,
        )
        claim = f"Reservations must be made at least {constraints.minimum_advance_reservation_window_days} in advance of the trip start date."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify the minimum advance window stated on official pages (e.g., 'at least 2 days in advance')."
            ),
        )

    # Reservable date range
    if not constraints or not constraints.reservable_date_range:
        _add_failed_leaf(
            evaluator,
            "Reservable_Date_Range",
            "States that camps may be reserved for dates between June 1 and October 10 (approximately the first federal holiday in October).",
            node,
            critical=True,
        )
    else:
        leaf = evaluator.add_leaf(
            id="Reservable_Date_Range",
            desc="States that camps may be reserved for dates between June 1 and October 10 (approximately the first federal holiday in October).",
            parent=node,
            critical=True,
        )
        claim = f"Camps may be reserved for dates between {constraints.reservable_date_range}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Confirm that official sources state the reservable date range for the Wonderland Trail season."
            ),
        )

    return node


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the provided answer for Wonderland Trail 2026 permit information.
    """

    # Initialize evaluator with a critical parallel root (as per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete and accurate information about Mount Rainier's Wonderland Trail wilderness permit system for the 2026 season, including lottery timeline, fee structure, trip-planning constraints, and official-source support.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Make the root node critical (the Evaluator creates a non-critical root by default, so add a critical child root-equivalent)
    # To adhere to rubric, we create a top-level critical container under Evaluator's root.
    rubric_root = evaluator.add_parallel(
        id="Wonderland_Trail_2026_Permit_Information",
        desc="Complete and accurate information about Mount Rainier's Wonderland Trail wilderness permit system for the 2026 season, including lottery timeline, fee structure, trip-planning constraints, and official-source support.",
        parent=root,
        critical=True,
    )

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_wonderland_permit_info(),
        template_class=WonderlandPermitExtraction,
        extraction_name="wonderland_permit_2026_extraction",
    )

    # Build sections
    official_node = await build_official_source_support(
        evaluator,
        rubric_root,
        extraction.official_urls if extraction and extraction.official_urls else [],
    )

    # Timeline
    await build_timeline_section(
        evaluator,
        rubric_root,
        extraction.timeline if extraction else None,
        extraction.official_urls if extraction else [],
    )

    # Fees
    await build_fee_section(
        evaluator,
        rubric_root,
        extraction.fees if extraction else None,
        extraction.official_urls if extraction else [],
    )

    # Constraints
    await build_constraints_section(
        evaluator,
        rubric_root,
        extraction.constraints if extraction else None,
        extraction.official_urls if extraction else [],
    )

    # Optionally record expected highlights (for debugging; does not affect scoring)
    evaluator.add_ground_truth({
        "notes": "Rubric expects 2026-specific details verified against official sources (Recreation.gov/NPS).",
        "expected_items_example": {
            "timeline": [
                "Applications open: Feb 10, 2026 7:00 am PT",
                "Applications close: March 3, 2026 7:00 pm PT",
                "Results notify: March 14, 2026 7:00 am PT",
                "General On-Sale: April 25, 2026 7:00 am PT"
            ],
            "fees": [
                "Reservation/application fee: $6.00 (non-refundable)",
                "Recreation fee: $10 per person per night",
                "Effective start note: applies starting with February 2025 lottery",
                "Youth 15 and under: free (no recreation fee)"
            ],
            "constraints": [
                "Max daily mileage limit for advance reservations: 17.5 trail miles",
                "Standard trailside camp max party size: 5 people and 3 tents",
                "Group camps required for parties of 6–12 people",
                "Minimum advance reservation window: at least 2 days before start date",
                "Reservable dates typically: June 1 to October 10"
            ]
        }
    }, gt_type="rubric_expectations")

    # Return the evaluator's structured summary
    return evaluator.get_summary()