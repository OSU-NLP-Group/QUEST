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
TASK_ID = "broadway_group_2026"
TASK_DESCRIPTION = """A community organization in New York City is planning a group outing to see a Broadway show during the week of February 21-28, 2026. The group consists of 18 adults, including 2 individuals who use wheelchairs and will need wheelchair-accessible seating. The organization wants to purchase tickets as a group to receive a discount and plans to buy the tickets in person at the theater's box office.

Provide the following information:

1. Event Details: Identify a specific Broadway show that has confirmed performances during February 21-28, 2026, and specify the theater venue where it is performed.

2. Accessibility Requirements: Confirm that the venue offers wheelchair-accessible seating and that the organization can purchase the 2 wheelchair spaces along with companion seats for other group members in a single transaction (per ADA requirements allowing up to 3 companion seats per wheelchair space).

3. Group Discount: Verify whether the 18-person group qualifies for a group ticket discount. Specify the minimum group size required for discount eligibility and the discount percentage or savings amount offered.

4. Box Office Information: Provide the box office operating hours for in-person ticket purchase, including both regular weekday hours and any extended hours on performance days.

5. Refund Policy: State the venue's ticket refund and cancellation policy, specifically addressing: (a) whether tickets are refundable if the organization cancels, and (b) what happens if the venue cancels the performance.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutingExtraction(BaseModel):
    # Event details
    show_name: Optional[str] = None
    venue_name: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)
    performance_dates_mentioned: List[str] = Field(default_factory=list)

    # Accessibility
    wheelchair_accessible_statement: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    companion_seat_policy: Optional[str] = None
    companion_seat_urls: List[str] = Field(default_factory=list)
    purchase_process: Optional[str] = None

    # Group discount
    min_group_size: Optional[str] = None
    discount_amount: Optional[str] = None
    group_discount_urls: List[str] = Field(default_factory=list)

    # Box office hours
    weekday_hours: Optional[str] = None
    event_day_hours: Optional[str] = None
    box_office_urls: List[str] = Field(default_factory=list)

    # Refund policy
    refundable_on_buyer_cancel: Optional[str] = None
    venue_cancellation_policy: Optional[str] = None
    exchange_policy: Optional[str] = None
    refund_policy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outing_info() -> str:
    return """
    Extract the following fields exactly as presented in the answer. Do not infer or invent information. If something is missing, set it to null (for strings) or an empty array (for lists).

    Event Details:
    - show_name: the specific Broadway show name identified.
    - venue_name: the Broadway theater venue where the show is performed.
    - schedule_urls: all URLs cited that confirm the performance schedule or show page (include any official show site, ticketing pages, listings, Playbill/IBDB/Broadway League, etc.).
    - performance_dates_mentioned: all dates listed or paraphrased for performances (e.g., "February 23, 2026", "Feb 25, 2026 7 PM"). If the answer states a range, include representative dates it mentions.

    Accessibility:
    - wheelchair_accessible_statement: the statement confirming wheelchair-accessible seating is available (copy the phrase like "wheelchair accessible seating available").
    - accessibility_urls: all URLs cited that discuss venue accessibility or wheelchair seating.
    - companion_seat_policy: the statement describing companion seat allowances adjacent to wheelchair spaces (e.g., "up to 3 companions").
    - companion_seat_urls: all URLs cited that describe companion seat policy or ADA seating.
    - purchase_process: the description of how wheelchair spaces and companion seats can be purchased together (e.g., "contact box office", "purchase in one transaction at box office").

    Group Discount:
    - min_group_size: the minimum number of tickets required for a group discount (e.g., "10+", "12", etc.).
    - discount_amount: the discount percentage or savings amount (e.g., "up to 15%", "savings vary", "10-25%").
    - group_discount_urls: all URLs cited that describe group sales, minimums, and discount details.

    Box Office Information:
    - weekday_hours: the stated weekday box office hours (e.g., "Mon–Fri 10am–6pm").
    - event_day_hours: the stated performance-day/curtain-day box office hours or extension policy (e.g., "open until curtain").
    - box_office_urls: all URLs cited for box office hours.

    Refund Policy:
    - refundable_on_buyer_cancel: the standard refund policy for buyer-initiated cancellations (e.g., "non-refundable", "all sales final", or the exact phrasing provided).
    - venue_cancellation_policy: what happens if the venue cancels a performance (e.g., "automatic full refund").
    - exchange_policy: any transfer or exchange options described.
    - refund_policy_urls: all URLs cited for refund/cancellation/exchange policy.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text. If the answer uses markdown links, extract the destination URLs.
    - Include the protocol; if missing, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(text: Optional[str]) -> bool:
    return bool(text) and isinstance(text, str) and text.strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and isinstance(urls, list) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_event_identification(evaluator: Evaluator, parent, ex: OutingExtraction) -> None:
    event_node = evaluator.add_parallel(
        id="Event_Identification",
        desc="Identify a Broadway show with confirmed performances during February 21-28, 2026",
        parent=parent,
        critical=True
    )

    # Show name existence (critical)
    evaluator.add_custom_node(
        result=_nonempty(ex.show_name),
        id="Show_Name",
        desc="Provide the name of a specific Broadway show",
        parent=event_node,
        critical=True
    )

    # Performance date verification (critical)
    perf_node = evaluator.add_parallel(
        id="Performance_Date_Verification",
        desc="Confirm the show has performances scheduled during February 21-28, 2026",
        parent=event_node,
        critical=True
    )

    # Date range check leaf (critical)
    date_check_leaf = evaluator.add_leaf(
        id="Date_Range_Check",
        desc="Verify at least one performance date falls within February 21-28, 2026",
        parent=perf_node,
        critical=True
    )
    claim_dates = f"At least one scheduled performance of '{ex.show_name or ''}' is between 2026-02-21 and 2026-02-28 (inclusive)."
    await evaluator.verify(
        claim=claim_dates,
        node=date_check_leaf,
        sources=ex.schedule_urls,
        additional_instruction="Check the performance calendar on the provided page(s) to verify at least one performance occurs during the specified week."
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=_has_urls(ex.schedule_urls),
        id="Performance_Reference_URL",
        desc="Provide URL source confirming the performance schedule",
        parent=perf_node,
        critical=True
    )

    # Venue verification (critical)
    venue_leaf = evaluator.add_leaf(
        id="Venue_Name",
        desc="Specify the Broadway theater venue where the show is performed",
        parent=event_node,
        critical=True
    )
    claim_venue = f"The venue for the show '{ex.show_name or ''}' is '{ex.venue_name or ''}'."
    await evaluator.verify(
        claim=claim_venue,
        node=venue_leaf,
        sources=ex.schedule_urls,
        additional_instruction="Verify that the referenced page(s) indicate this venue for the identified show."
    )


async def build_accessibility(evaluator: Evaluator, parent, ex: OutingExtraction) -> None:
    acc_node = evaluator.add_parallel(
        id="Accessibility_Compliance",
        desc="Verify wheelchair-accessible seating availability and companion seat purchase allowance",
        parent=parent,
        critical=True
    )

    # Wheelchair seating availability (critical)
    wc_node = evaluator.add_parallel(
        id="Wheelchair_Seating_Availability",
        desc="Confirm the venue offers wheelchair-accessible seating",
        parent=acc_node,
        critical=True
        )
    wc_leaf = evaluator.add_leaf(
        id="Accessibility_Confirmation",
        desc="State that wheelchair spaces are available at the venue",
        parent=wc_node,
        critical=True
    )
    claim_wc = f"The venue '{ex.venue_name or ''}' offers wheelchair-accessible seating."
    await evaluator.verify(
        claim=claim_wc,
        node=wc_leaf,
        sources=ex.accessibility_urls,
        additional_instruction="Verify the venue's accessibility page or policy mentions wheelchair-accessible seating or ADA seating."
    )
    evaluator.add_custom_node(
        result=_has_urls(ex.accessibility_urls),
        id="Accessibility_Reference_URL",
        desc="Provide URL source for accessibility information",
        parent=wc_node,
        critical=True
    )

    # Companion seat policy (critical)
    comp_node = evaluator.add_parallel(
        id="Companion_Seat_Policy",
        desc="Verify the venue allows purchasing companion seats adjacent to wheelchair spaces",
        parent=acc_node,
        critical=True
    )
    comp_allow_leaf = evaluator.add_leaf(
        id="Companion_Seat_Allowance",
        desc="Confirm patrons can purchase companion seats with wheelchair spaces (per ADA, up to 3 companion seats allowed)",
        parent=comp_node,
        critical=True
    )
    claim_comp = f"The venue '{ex.venue_name or ''}' allows adjacent companion seats to be purchased with wheelchair spaces (up to 3 per wheelchair space)."
    await evaluator.verify(
        claim=claim_comp,
        node=comp_allow_leaf,
        sources=ex.companion_seat_urls,
        additional_instruction="Verify that the policy or ADA seating information indicates companion seats adjacent to wheelchair spaces (up to 3 per space)."
    )

    # Although originally marked non-critical in the JSON, to satisfy framework constraints under a critical parent,
    # we make this a critical existence check.
    evaluator.add_custom_node(
        result=_nonempty(ex.purchase_process),
        id="Purchase_Process",
        desc="Describe how wheelchair spaces and companion seats can be purchased together",
        parent=comp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(ex.companion_seat_urls),
        id="Companion_Seat_Reference_URL",
        desc="Provide URL source for companion seat policy",
        parent=comp_node,
        critical=True
    )


async def build_group_discount(evaluator: Evaluator, parent, ex: OutingExtraction) -> None:
    gd_node = evaluator.add_parallel(
        id="Group_Discount_Eligibility",
        desc="Verify the group qualifies for a discount and specify discount details",
        parent=parent,
        critical=True
    )

    # Group size threshold (critical)
    threshold_node = evaluator.add_parallel(
        id="Group_Size_Threshold",
        desc="Identify the minimum group size required for discount eligibility",
        parent=gd_node,
        critical=True
    )

    min_leaf = evaluator.add_leaf(
        id="Minimum_Group_Size",
        desc="State the minimum number of tickets required (typically 10+ people based on gathered information)",
        parent=threshold_node,
        critical=True
    )
    claim_min = f"The minimum group size required for a group discount is {ex.min_group_size or ''}."
    await evaluator.verify(
        claim=claim_min,
        node=min_leaf,
        sources=ex.group_discount_urls,
        additional_instruction="Verify the group sales page specifies this minimum group size threshold."
    )

    qualify_leaf = evaluator.add_leaf(
        id="Group_Qualification",
        desc="Confirm the 18-person group meets or exceeds the minimum threshold",
        parent=threshold_node,
        critical=True
    )
    # Pure logical verification using the extracted minimum; allow minor phrasing.
    claim_qualify = f"An 18-person group meets or exceeds the minimum group size of {ex.min_group_size or ''}."
    await evaluator.verify(
        claim=claim_qualify,
        node=qualify_leaf,
        additional_instruction="Treat this as a simple logical comparison. If the minimum is given as a number or '10+', an 18-person group satisfies it."
    )

    # Discount percentage/amount (critical)
    disc_node = evaluator.add_parallel(
        id="Discount_Percentage",
        desc="Specify the discount percentage or savings amount for group purchases",
        parent=gd_node,
        critical=True
    )

    disc_leaf = evaluator.add_leaf(
        id="Discount_Amount",
        desc="Provide the percentage discount or dollar savings (typically 10-25% based on gathered information)",
        parent=disc_node,
        critical=True
    )
    claim_disc = f"The group discount is {ex.discount_amount or ''}."
    await evaluator.verify(
        claim=claim_disc,
        node=disc_leaf,
        sources=ex.group_discount_urls,
        additional_instruction="Verify that the group sales page states this discount or savings range/amount."
    )

    evaluator.add_custom_node(
        result=_has_urls(ex.group_discount_urls),
        id="Group_Discount_Reference_URL",
        desc="Provide URL source for group discount information",
        parent=disc_node,
        critical=True
    )


async def build_box_office(evaluator: Evaluator, parent, ex: OutingExtraction) -> None:
    bo_node = evaluator.add_parallel(
        id="Box_Office_Purchase_Details",
        desc="Provide box office hours for in-person ticket purchase",
        parent=parent,
        critical=True
    )

    hours_node = evaluator.add_parallel(
        id="Box_Office_Hours",
        desc="Specify the operating hours for in-person ticket purchase",
        parent=bo_node,
        critical=True
    )

    weekday_leaf = evaluator.add_leaf(
        id="Weekday_Hours",
        desc="Provide weekday box office hours (typically Monday-Friday, 10am-5pm or similar based on gathered information)",
        parent=hours_node,
        critical=True
    )
    claim_weekday = f"The weekday box office hours are: {ex.weekday_hours or ''}."
    await evaluator.verify(
        claim=claim_weekday,
        node=weekday_leaf,
        sources=ex.box_office_urls,
        additional_instruction="Verify that the box office hours page lists these weekday hours."
    )

    eventday_leaf = evaluator.add_leaf(
        id="Event_Day_Hours",
        desc="Indicate if hours are extended on performance days",
        parent=hours_node,
        critical=True
    )
    claim_eventday = f"On performance days, the box office hours are: {ex.event_day_hours or ''}."
    await evaluator.verify(
        claim=claim_eventday,
        node=eventday_leaf,
        sources=ex.box_office_urls,
        additional_instruction="Verify if the hours are extended on performance days (e.g., 'open until curtain')."
    )

    evaluator.add_custom_node(
        result=_has_urls(ex.box_office_urls),
        id="Box_Office_Reference_URL",
        desc="Provide URL source for box office hours",
        parent=hours_node,
        critical=True
    )


async def build_refund_policy(evaluator: Evaluator, parent, ex: OutingExtraction) -> None:
    rp_node = evaluator.add_parallel(
        id="Refund_Policy",
        desc="State the cancellation and refund policy for group tickets",
        parent=parent,
        critical=True
    )

    std_node = evaluator.add_parallel(
        id="Standard_Refund_Policy",
        desc="Describe the venue's standard refund policy for ticket purchases",
        parent=rp_node,
        critical=True
    )

    refund_leaf = evaluator.add_leaf(
        id="Refund_Availability",
        desc="State whether tickets are refundable under standard circumstances (typically non-refundable based on gathered information)",
        parent=std_node,
        critical=True
    )
    claim_refund = f"Under standard circumstances, tickets are: {ex.refundable_on_buyer_cancel or ''}."
    await evaluator.verify(
        claim=claim_refund,
        node=refund_leaf,
        sources=ex.refund_policy_urls,
        additional_instruction="Verify the policy page states whether standard ticket purchases are refundable (often 'all sales final' or 'non-refundable')."
    )

    venue_cancel_leaf = evaluator.add_leaf(
        id="Venue_Cancellation_Policy",
        desc="Confirm whether full refunds are provided if the venue cancels the event (typically yes, full automatic refund based on gathered information)",
        parent=std_node,
        critical=True
    )
    claim_venue_cancel = f"If the venue cancels a performance, the policy is: {ex.venue_cancellation_policy or ''}."
    await evaluator.verify(
        claim=claim_venue_cancel,
        node=venue_cancel_leaf,
        sources=ex.refund_policy_urls,
        additional_instruction="Verify that the policy indicates refunds or automatic refunds when the performance is canceled by the venue."
    )

    # Transfer / Exchange policy
    # Although the original rubric marked this as non-critical, to satisfy framework constraints under a critical parent,
    # we mark it critical here.
    xfer_node = evaluator.add_parallel(
        id="Transfer_Exchange_Policy",
        desc="Describe any ticket transfer or exchange options available",
        parent=rp_node,
        critical=True
    )

    exchange_leaf = evaluator.add_leaf(
        id="Exchange_Availability",
        desc="State if tickets can be exchanged for different performance dates",
        parent=xfer_node,
        critical=True
    )
    claim_exchange = f"Ticket exchange policy: {ex.exchange_policy or ''}."
    await evaluator.verify(
        claim=claim_exchange,
        node=exchange_leaf,
        sources=ex.refund_policy_urls,
        additional_instruction="Verify whether exchanges are allowed and under what conditions."
    )

    evaluator.add_custom_node(
        result=_has_urls(ex.refund_policy_urls),
        id="Refund_Exchange_Reference_URL",
        desc="Provide URL source for refund and exchange policy",
        parent=xfer_node,
        critical=True
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
    Evaluate an answer for the Broadway group outing requirements task.
    """
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
        default_model=model
    )

    # Extract structured information from the answer
    ex: OutingExtraction = await evaluator.extract(
        prompt=prompt_extract_outing_info(),
        template_class=OutingExtraction,
        extraction_name="outing_extraction"
    )

    # Build main rubric node (non-critical to allow partial credit across sections while children enforce critical checks)
    main = evaluator.add_parallel(
        id="Broadway_Group_Outing_Requirements",
        desc="Verify all requirements for a Broadway group outing including event identification, accessibility compliance, group discount eligibility, purchase details, and cancellation policy",
        parent=root,
        critical=False
    )

    # Build each section
    await build_event_identification(evaluator, main, ex)
    await build_accessibility(evaluator, main, ex)
    await build_group_discount(evaluator, main, ex)
    await build_box_office(evaluator, main, ex)
    await build_refund_policy(evaluator, main, ex)

    # Return evaluation summary
    return evaluator.get_summary()