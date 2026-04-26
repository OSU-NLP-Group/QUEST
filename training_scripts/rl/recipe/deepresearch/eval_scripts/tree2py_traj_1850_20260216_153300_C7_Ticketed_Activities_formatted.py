import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "orlando_theme_park_ticketing_requirements"
TASK_DESCRIPTION = """Identify a major theme park or entertainment venue located in Orlando, Florida that meets all of the following ticketing and visitor service requirements:

1. Offers standard adult admission tickets (for ages 10 and older) with clearly listed pricing
2. Offers child admission tickets (for ages 3-9) at a different price point than adult tickets
3. Has a free admission policy for children below a specific age or height threshold when accompanied by a paying adult
4. Publishes daily operating hours
5. Provides on-site parking with a separately charged parking fee
6. Offers a priority access option (such as express pass, fast pass, or quick queue) that allows visitors to skip regular lines or reduce wait times
7. Provides pricing information for the priority access option
8. Offers multi-day passes or annual pass options for unlimited or multiple visits
9. Allows tickets to be purchased online in advance of the visit date
10. Participates in a military discount program offering reduced admission rates for active or retired military personnel
11. Has an official website where all ticketing information can be verified

For the identified venue, provide the following specific information:
- Venue name and official website URL
- Standard adult single-day admission price
- Child single-day admission price
- Free admission threshold (age or height)
- Typical daily operating hours
- Standard parking fee
- Name and pricing of the priority access option
- Confirmation of multi-day pass availability
- Military discount program details and discount percentage
- Confirmation that online ticket purchase is available
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueTicketingExtraction(BaseModel):
    # Venue basics
    venue_name: Optional[str] = None
    official_website_url: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None

    # Ticketing: adult & child
    adult_single_day_price: Optional[str] = None
    adult_age_definition: Optional[str] = None  # e.g., "ages 10+"
    child_single_day_price: Optional[str] = None
    child_age_definition: Optional[str] = None  # e.g., "ages 3–9"

    # Senior category
    senior_age_threshold: Optional[str] = None  # e.g., "ages 60+" or "65+"
    senior_price: Optional[str] = None

    # Free admission threshold
    free_admission_threshold: Optional[str] = None  # age or height text, e.g., "2 and under" or "<= 115 cm"
    free_admission_condition: Optional[str] = None  # e.g., "when accompanied by a paying adult"

    # Operating hours
    daily_operating_hours: Optional[str] = None  # typical or example hours string

    # Parking
    parking_fee: Optional[str] = None            # e.g., "$30"
    parking_fee_unit: Optional[str] = None       # e.g., "per vehicle per day"
    parking_separate: Optional[str] = None       # e.g., "charged separately" or "included"

    # Priority access / skip-the-line
    priority_access_option_name: Optional[str] = None  # e.g., "Express Pass", "Quick Queue"
    priority_access_option_price: Optional[str] = None

    # Passes
    multi_day_or_annual_pass_available: Optional[str] = None  # free-form "Yes: Multi-Day ..."

    # Online purchase & discounts
    online_advance_purchase_available: Optional[str] = None  # "Yes" / details
    online_ticket_discount_statement: Optional[str] = None   # e.g., "Buy online and save vs gate"

    # Military discount
    military_discount_details: Optional[str] = None
    military_discount_percentage: Optional[str] = None       # e.g., "15%"

    # Group discount
    group_discount_details: Optional[str] = None
    group_discount_min_size: Optional[str] = None            # e.g., "10+"
    group_discount_percentage: Optional[str] = None          # e.g., "10%"

    # All URLs explicitly mentioned in the answer (besides/including official website)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
Extract the following fields from the answer text if they are explicitly present. Return null for any field that is not stated in the answer. Do not infer or invent information.

Required JSON fields to extract:

- venue_name: The official name of the venue.
- official_website_url: The venue’s official website URL (homepage or a clearly official domain).
- location_city: The city where the venue is located (e.g., "Orlando").
- location_state: The state (e.g., "Florida" or "FL").

Ticketing prices and definitions:
- adult_single_day_price: The standard adult single-day admission price string exactly as written (e.g., "$119", "from $115").
- adult_age_definition: The age definition for adult tickets exactly as written (e.g., "ages 10+", "ages 10 and up").
- child_single_day_price: The standard child single-day admission price string exactly as written.
- child_age_definition: The age definition for child tickets exactly as written (e.g., "ages 3–9").

Senior category:
- senior_age_threshold: The senior category age threshold exactly as written (e.g., "60+", "65 and over").
- senior_price: The listed price for seniors exactly as written.

Free admission:
- free_admission_threshold: The free-admission threshold (age or height) exactly as written (e.g., "2 and under", "<= 115 cm").
- free_admission_condition: Any condition text for the free-admission policy (e.g., "when accompanied by a paying adult").

Operating hours:
- daily_operating_hours: Typical or example hours text as provided (e.g., "9:00 AM–9:00 PM").

Parking:
- parking_fee: The standard parking fee string exactly as written (e.g., "$30").
- parking_fee_unit: The unit if stated (e.g., "per vehicle per day").
- parking_separate: Text indicating whether parking is charged separately or included (extract the text).

Priority access:
- priority_access_option_name: The name of any priority access/skip-the-line product (e.g., "Express Pass", "Quick Queue").
- priority_access_option_price: The price text for that option exactly as written.

Passes:
- multi_day_or_annual_pass_available: Text indicating that multi-day or annual passes are available (extract as-is if present).

Online purchase & discounts:
- online_advance_purchase_available: Text indicating tickets can be purchased online in advance (extract as-is).
- online_ticket_discount_statement: Statement showing online price is lower than gate price (extract any explicit comparison or phrase like "Save when you buy online").

Military discount:
- military_discount_details: Text describing the military discount program (extract as-is).
- military_discount_percentage: The percent discount or rate if given (e.g., "15%", "up to 30%").

Group discount:
- group_discount_details: Text describing group discounts (extract as-is).
- group_discount_min_size: The minimum group size threshold exactly as written (e.g., "10+").
- group_discount_percentage: The discount percentage for groups if given (e.g., "10%", "15%").

URLs:
- source_urls: Extract ALL URLs explicitly mentioned in the answer (including ticketing, hours, parking, military/group discount, and priority access pages). Include valid full URLs only. If a URL is missing a protocol, prepend "http://".

Return a single JSON object with all fields listed above. Do not add extraneous fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_non_empty_urls(urls: List[Optional[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def parse_first_amount(text: Optional[str]) -> Optional[float]:
    """
    Extract the first numeric amount (e.g., $109, 109.99) from a free-form string.
    Returns a float if found, else None.
    """
    if not text:
        return None
    # Find a number, optionally preceded by a dollar sign, with optional decimals
    m = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)(?:\.[0-9]{1,2})?", text)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        return float(num_str)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate a single answer for the Orlando theme park/entertainment venue ticketing task.
    Builds a critical parallel verification node to enforce all constraints must be satisfied.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted: VenueTicketingExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueTicketingExtraction,
        extraction_name="venue_ticketing_extraction",
    )

    # Combine URLs for verification (official website + all mentioned URLs)
    combined_sources = _unique_non_empty_urls(
        [extracted.official_website_url] + (extracted.source_urls or [])
        if extracted.source_urls is not None
        else [extracted.official_website_url]
    )

    # Create the top-level critical parallel node
    main_node = evaluator.add_parallel(
        id="venue_selection_and_required_outputs",
        desc=(
            "Identify one major theme park/entertainment venue in Orlando, Florida that satisfies all requirements "
            "from the proposed question AND the provided constraints list, and provide the requested details with an "
            "official website for verification."
        ),
        parent=root,
        critical=True,
    )

    # Prepare leaves and corresponding claims for batch verification
    claims_batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 1. Venue name (verify appears on official website)
    node_venue_name = evaluator.add_leaf(
        id="venue_name",
        desc="Provide the venue name.",
        parent=main_node,
        critical=True,
    )
    claim_venue_name = f"The venue's official name is '{extracted.venue_name}'." if extracted.venue_name else "The venue name is provided and appears on the official website."
    claims_batch.append(
        (
            claim_venue_name,
            combined_sources,
            node_venue_name,
            "Verify that the official website displays this venue name. Minor formatting/case variations are acceptable."
        )
    )

    # 2. Venue in Orlando, Florida
    node_location = evaluator.add_leaf(
        id="venue_in_orlando_florida",
        desc="Venue is located in Orlando, Florida.",
        parent=main_node,
        critical=True,
    )
    claims_batch.append(
        (
            "The venue is located in Orlando, Florida.",
            combined_sources,
            node_location,
            "Check the official website's address/contact/location page for 'Orlando, FL' or 'Orlando, Florida'. Minor formatting variants are acceptable."
        )
    )

    # 3. Official website URL (verify is official)
    node_official_site = evaluator.add_leaf(
        id="official_website_url",
        desc="Provide the venue's official website URL where ticketing information can be verified.",
        parent=main_node,
        critical=True,
    )
    official_url_claim = (
        f"The URL {extracted.official_website_url} is the official website of {extracted.venue_name}."
        if extracted.official_website_url and extracted.venue_name
        else "This URL is the venue's official website."
    )
    claims_batch.append(
        (
            official_url_claim,
            extracted.official_website_url or None,
            node_official_site,
            "Verify branding/copyright and official contact information indicate this is the venue's official website."
        )
    )

    # 4. Adult single-day price (ages 10+ or equivalent)
    node_adult_price = evaluator.add_leaf(
        id="adult_single_day_price_age_10_plus",
        desc="Provide a clearly listed standard adult single-day admission price for ages 10+ (or an explicitly equivalent adult definition used by the venue).",
        parent=main_node,
        critical=True,
    )
    claim_adult_price = (
        f"The standard adult single-day admission price is {extracted.adult_single_day_price}, and 'adult' is defined as {extracted.adult_age_definition}."
        if extracted.adult_single_day_price and extracted.adult_age_definition
        else f"The standard adult single-day admission price is {extracted.adult_single_day_price}."
        if extracted.adult_single_day_price
        else "The standard adult single-day admission price for adults (typically ages 10+) is clearly listed."
    )
    claims_batch.append(
        (
            claim_adult_price,
            combined_sources,
            node_adult_price,
            "Verify the ticketing/pricing page lists the adult single-day ticket price. 'From' or variable date-based pricing is acceptable."
        )
    )

    # 5. Child single-day price (ages 3–9 or equivalent)
    node_child_price = evaluator.add_leaf(
        id="child_single_day_price_age_3_9",
        desc="Provide a clearly listed child single-day admission price for ages 3–9 (or an explicitly equivalent child definition used by the venue).",
        parent=main_node,
        critical=True,
    )
    claim_child_price = (
        f"The standard child single-day admission price is {extracted.child_single_day_price}, and 'child' is defined as {extracted.child_age_definition}."
        if extracted.child_single_day_price and extracted.child_age_definition
        else f"The standard child single-day admission price is {extracted.child_single_day_price}."
        if extracted.child_single_day_price
        else "The standard child single-day admission price for children (typically ages 3–9) is clearly listed."
    )
    claims_batch.append(
        (
            claim_child_price,
            combined_sources,
            node_child_price,
            "Verify the ticketing/pricing page lists the child single-day ticket price. 'From' or variable date-based pricing is acceptable."
        )
    )

    # 6. Adult/child prices different (derived check – custom node; added AFTER batch verify to avoid precondition skips)
    # Placeholder; will compute after batch verifications

    # 7. Senior admission category with threshold and price
    node_senior = evaluator.add_leaf(
        id="senior_ticket_category_required_by_constraints",
        desc="Confirm the venue has a senior admission category with an age threshold in the 60–65+ range and that the senior category has its own listed price (distinct category as required by constraints).",
        parent=main_node,
        critical=True,
    )
    claim_senior = (
        f"There is a senior admission category with an age threshold {extracted.senior_age_threshold}, and it has a listed price {extracted.senior_price}."
        if extracted.senior_age_threshold and extracted.senior_price
        else "There is a senior admission category for ages 60–65+ with its own listed price."
    )
    claims_batch.append(
        (
            claim_senior,
            combined_sources,
            node_senior,
            "Verify that the site lists a distinct senior ticket category with an age threshold within 60–65+ and shows a price for it."
        )
    )

    # 8. Free admission threshold with condition and specific limits
    node_free = evaluator.add_leaf(
        id="free_admission_threshold",
        desc="State the free-admission threshold (age or height) for children, indicate it applies when accompanied by a paying adult, and verify the threshold is under 3 years OR under 115 cm (as required by constraints).",
        parent=main_node,
        critical=True,
    )
    claim_free = (
        f"Children {extracted.free_admission_threshold} are admitted free when {extracted.free_admission_condition}."
        if extracted.free_admission_threshold and extracted.free_admission_condition
        else f"Children {extracted.free_admission_threshold} are admitted free when accompanied by a paying adult."
        if extracted.free_admission_threshold
        else "There is a free-admission threshold for young children (under 3 years old) or a height threshold under 115 cm, when accompanied by a paying adult."
    )
    claims_batch.append(
        (
            claim_free,
            combined_sources,
            node_free,
            "Verify the free admission threshold is specified and is under 3 years OR under 115 cm, and that it's contingent upon being accompanied by a paying adult."
        )
    )

    # 9. Daily operating hours published and within typical range
    node_hours = evaluator.add_leaf(
        id="daily_operating_hours_provided",
        desc="Provide published daily operating hours (demonstrating they are publicly posted) and verify the stated typical hours fall within 9:00 AM–10:00 PM (as required by constraints).",
        parent=main_node,
        critical=True,
    )
    claim_hours = (
        f"The venue publishes daily operating hours; a typical example is '{extracted.daily_operating_hours}', which falls within 9:00 AM–10:00 PM."
        if extracted.daily_operating_hours
        else "The venue publishes daily operating hours and typical hours fall within 9:00 AM–10:00 PM."
    )
    claims_batch.append(
        (
            claim_hours,
            combined_sources,
            node_hours,
            "Verify that the website posts operating hours. If hours vary by date/season, a representative example within the 9:00 AM–10:00 PM span is acceptable."
        )
    )

    # 10. Parking separately charged
    node_parking_separate = evaluator.add_leaf(
        id="parking_fee_separately_charged",
        desc="Confirm on-site parking is available and that parking is charged separately from admission.",
        parent=main_node,
        critical=True,
    )
    claims_batch.append(
        (
            "On-site parking is available and it is charged separately from admission.",
            combined_sources,
            node_parking_separate,
            "Verify the parking policy states on-site parking with a separate fee (not included in admission)."
        )
    )

    # 11. Standard parking fee amount between $10 and $60
    node_parking_fee = evaluator.add_leaf(
        id="standard_parking_fee_amount",
        desc="Provide the standard parking fee amount (with a clearly stated unit such as per vehicle per day) and verify it is between $10 and $60 (as required by constraints).",
        parent=main_node,
        critical=True,
    )
    claim_parking_fee = (
        f"The standard parking fee is {extracted.parking_fee} {extracted.parking_fee_unit or ''}. The fee falls between $10 and $60."
        if extracted.parking_fee
        else "The standard parking fee for a standard vehicle per day is posted and falls between $10 and $60."
    )
    claims_batch.append(
        (
            claim_parking_fee,
            combined_sources,
            node_parking_fee,
            "Verify the standard (not preferred/premium) daily parking fee is posted and within $10–$60."
        )
    )

    # 12. Priority access option name (reduces waits/skips lines)
    node_priority_name = evaluator.add_leaf(
        id="priority_access_option_name",
        desc="Identify the priority access option (e.g., express/fast/quick queue) that reduces waits or skips regular lines.",
        parent=main_node,
        critical=True,
    )
    claim_priority_name = (
        f"The venue offers a priority access option called '{extracted.priority_access_option_name}' that reduces wait times or allows skipping regular lines."
        if extracted.priority_access_option_name
        else "The venue offers a priority access option that reduces waits or skips regular lines."
    )
    claims_batch.append(
        (
            claim_priority_name,
            combined_sources,
            node_priority_name,
            "Verify the product name (e.g., Express Pass, Quick Queue, FastPass) and that it grants expedited/priority access."
        )
    )

    # 13. Priority access option pricing
    node_priority_price = evaluator.add_leaf(
        id="priority_access_option_pricing",
        desc="Provide pricing information for the priority access option.",
        parent=main_node,
        critical=True,
    )
    claim_priority_price = (
        f"The {extracted.priority_access_option_name or 'priority access option'} pricing is {extracted.priority_access_option_price}."
        if extracted.priority_access_option_price
        else "Pricing information for the priority access option is provided (e.g., listed price or 'from' price)."
    )
    claims_batch.append(
        (
            claim_priority_price,
            combined_sources,
            node_priority_price,
            "Verify that the website lists the price for the priority access option; 'from' pricing or date-based pricing is acceptable."
        )
    )

    # 14. Multi-day or annual passes available
    node_passes = evaluator.add_leaf(
        id="multi_day_or_annual_pass_available",
        desc="Confirm availability of multi-day passes or annual pass options for multiple/unlimited visits.",
        parent=main_node,
        critical=True,
    )
    claims_batch.append(
        (
            "The venue offers multi-day passes or annual passes for multiple or unlimited visits.",
            combined_sources,
            node_passes,
            "Verify the presence of multi-day ticket options and/or annual/membership passes on the official site."
        )
    )

    # 15. Online advance purchase available
    node_online = evaluator.add_leaf(
        id="online_advance_purchase_available",
        desc="Confirm tickets can be purchased online in advance of the visit date.",
        parent=main_node,
        critical=True,
    )
    claims_batch.append(
        (
            "Tickets can be purchased online in advance of the visit date.",
            combined_sources,
            node_online,
            "Verify that the official website offers online purchasing before the visit day (e.g., 'Buy Tickets' online)."
        )
    )

    # 16. Online ticket discount lower than gate price
    node_online_discount = evaluator.add_leaf(
        id="online_ticket_discount_required_by_constraints",
        desc="Confirm that tickets purchased online are priced lower than gate prices (as required by constraints) and provide an example or explicit statement showing the comparison.",
        parent=main_node,
        critical=True,
    )
    claim_online_discount = (
        f"The venue states that online ticket prices are lower than gate prices. Example statement: {extracted.online_ticket_discount_statement}."
        if extracted.online_ticket_discount_statement
        else "The venue states or shows that buying tickets online is cheaper than the gate price."
    )
    claims_batch.append(
        (
            claim_online_discount,
            combined_sources,
            node_online_discount,
            "Look for an explicit statement such as 'save when you buy online' or a direct comparison showing lower online prices than gate."
        )
    )

    # 17. Military discount program with percentage between 10–50%
    node_military = evaluator.add_leaf(
        id="military_discount_program_details",
        desc="Confirm participation in a military discount program and provide the program details and discount percentage (or clearly stated reduced rate); verify the discount is within 10–50% (as required by constraints).",
        parent=main_node,
        critical=True,
    )
    claim_military = (
        f"The venue participates in a military discount program for active or retired military. The discount is {extracted.military_discount_percentage}; details: {extracted.military_discount_details}. The discount falls within 10%–50%."
        if extracted.military_discount_percentage and extracted.military_discount_details
        else "The venue offers a military discount program with a clearly stated percentage between 10% and 50%."
    )
    claims_batch.append(
        (
            claim_military,
            combined_sources,
            node_military,
            "Verify the official site describes a military discount program and that the discount rate is between 10% and 50% (inclusive)."
        )
    )

    # 18. Group discount for 10+ visitors, discount between 10–20%
    node_group = evaluator.add_leaf(
        id="group_discount_required_by_constraints",
        desc="Confirm the venue offers a group discount for groups of 10+ visitors and verify the discount is within 10–20% (as required by constraints).",
        parent=main_node,
        critical=True,
    )
    claim_group = (
        f"The venue offers a group discount for groups of {extracted.group_discount_min_size or '10+'} visitors. The discount is {extracted.group_discount_percentage}; details: {extracted.group_discount_details}. The discount falls within 10%–20%."
        if (extracted.group_discount_details or extracted.group_discount_percentage or extracted.group_discount_min_size)
        else "The venue offers a group discount for groups of 10+ visitors with a discount rate between 10% and 20%."
    )
    claims_batch.append(
        (
            claim_group,
            combined_sources,
            node_group,
            "Verify the group sales/discount page indicates a discount for 10 or more visitors and that the discount percent is within 10–20%."
        )
    )

    # Run all URL-backed verifications in parallel to avoid precondition short-circuiting due to critical siblings
    await evaluator.batch_verify(claims_batch)

    # Now add the derived logical check for different adult vs child prices
    # This is a critical custom node (binary, no URL needed as it's a cross-field logical requirement).
    adult_amt = parse_first_amount(extracted.adult_single_day_price)
    child_amt = parse_first_amount(extracted.child_single_day_price)
    prices_different = (adult_amt is not None and child_amt is not None and abs(adult_amt - child_amt) > 1e-9)

    evaluator.add_custom_node(
        result=prices_different,
        id="adult_child_prices_different",
        desc="Adult and child single-day admission prices are at different price points (not the same price).",
        parent=main_node,
        critical=True,
    )

    # Return summary
    return evaluator.get_summary()