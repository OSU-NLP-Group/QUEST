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
TASK_ID = "us_multi_day_group_entertainment_venues"
TASK_DESCRIPTION = """
A corporate event planning company is developing a comprehensive entertainment venue guide for clients planning multi-day group events across the United States. They need to identify four distinct types of ticketed entertainment venues, each in a different state and meeting specific capacity, accessibility, and pricing requirements to accommodate diverse group needs.

Find the following four venues:

Venue 1 - Broadway Theater (New York):
Identify a Broadway theater in Manhattan's Theater District with a seating capacity between 500 and 700 seats that provides wheelchair accessible seating, aisle transfer seats, and companion seating, and currently has performances scheduled with accessible seating options available.

Venue 2 - Outdoor Amphitheater (California):
Identify an outdoor amphitheater in California with a seating capacity between 5,000 and 10,000 that offers wheelchair accessible seating in multiple price ranges and has upcoming concerts or events with active ticket sales.

Venue 3 - Theme Park (Florida):
Identify a theme park in Florida where the lowest-priced single-day adult admission ticket (before taxes and fees) is under $100, offers group discounts for parties of 15 or fewer people, and is currently operational with published operating hours.

Venue 4 - Sports Stadium Suite (Texas):
Identify a sports stadium in Texas that offers luxury suite rentals accommodating between 20 and 30 guests, where suite rentals include event tickets and VIP parking passes, and suites are available for rental for upcoming scheduled events.

For each venue, provide:
1. The venue name
2. Specific location (city and state)
3. Seating capacity or suite capacity (as applicable)
4. Evidence of accessibility features (for Venues 1 and 2) or pricing/discount information (for Venues 3 and 4)
5. Reference URLs supporting each requirement
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BroadwayVenue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    capacity_text: Optional[str] = None
    accessibility_features_mentioned: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class AmphitheaterVenue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_text: Optional[str] = None
    accessibility_price_tiers_examples: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class ThemeParkVenue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    lowest_single_day_adult_price_text: Optional[str] = None
    group_discount_min_size_text: Optional[str] = None
    operating_hours_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class StadiumSuiteVenue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    suite_capacity_text: Optional[str] = None
    includes_tickets_parking_text: Optional[str] = None
    availability_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    broadway: Optional[BroadwayVenue] = None
    amphitheater: Optional[AmphitheaterVenue] = None
    theme_park: Optional[ThemeParkVenue] = None
    stadium_suite: Optional[StadiumSuiteVenue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract the four venues and their fields from the answer. Only extract information explicitly present in the answer. For each venue, also extract all reference URLs that the answer cites to support the facts.

Return a JSON object with four objects: broadway, amphitheater, theme_park, stadium_suite.

For each object, extract:

1) broadway (Broadway Theater in New York)
   - name: string
   - city: string
   - state: string (e.g., "NY" or "New York")
   - district: string if mentioned (e.g., "Theater District", "Midtown Manhattan")
   - capacity_text: string as written in the answer (e.g., "589 seats", "about 600")
   - accessibility_features_mentioned: array of strings (e.g., ["wheelchair accessible seating","aisle transfer seats","companion seating"])
   - reference_urls: array of URLs explicitly present in the answer that support location, capacity, accessibility, or scheduled/accessible seating

2) amphitheater (Outdoor Amphitheater in California)
   - name: string
   - city: string
   - state: string
   - capacity_text: string as written (e.g., "9,500 capacity")
   - accessibility_price_tiers_examples: array of strings quoted from the answer if it mentions ADA/accessible seating in multiple price ranges (e.g., ["ADA in Lawn", "ADA in Section 101"])
   - reference_urls: array of URLs from the answer that support location, capacity, accessibility price tiers, or upcoming events / ticket sales

3) theme_park (Theme Park in Florida)
   - name: string
   - city: string
   - state: string
   - lowest_single_day_adult_price_text: string as written for the lowest-priced single-day adult admission before taxes/fees
   - group_discount_min_size_text: string as written (e.g., "groups of 10+")
   - operating_hours_text: string snippet if operating hours or calendar is mentioned
   - reference_urls: array of URLs from the answer that support location, ticket pricing/type, group discount terms, and operating hours/operational status

4) stadium_suite (Sports Stadium Suites in Texas)
   - name: string
   - city: string
   - state: string
   - suite_capacity_text: string as written for suite capacity (e.g., "20-30 guests", "up to 24")
   - includes_tickets_parking_text: string as written if the answer mentions that suites include event tickets and VIP/premium parking passes
   - availability_text: string snippet if upcoming suite availability is mentioned
   - reference_urls: array of URLs from the answer that support location, suite rental offering, suite capacity, inclusions, and upcoming availability

Rules:
- Do not invent URLs. Only include URLs that actually appear in the answer.
- If a field is not present in the answer, set it to null or [] accordingly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _bool_present(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls or []) > 0


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_venue_1_broadway(evaluator: Evaluator, root_node, v: Optional[BroadwayVenue]) -> None:
    node = evaluator.add_parallel(
        id="venue_1_broadway_theater_ny",
        desc="Venue 1: Broadway theater in Manhattan's Theater District (New York) meeting capacity, accessibility, and schedule constraints",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = _bool_present(v.name) if v else False
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_name_provided",
        desc="Provide the venue name",
        parent=node,
        critical=True
    )
    loc_ok = (_bool_present(v.city) and _bool_present(v.state)) if v else False
    evaluator.add_custom_node(
        result=loc_ok,
        id="location_city_state_provided",
        desc="Provide the venue location including city and state",
        parent=node,
        critical=True
    )

    # URLs presence (critical prerequisite for all URL-supported verifications)
    urls = (v.reference_urls if v else []) or []
    urls_node = evaluator.add_custom_node(
        result=_urls_present(urls),
        id="reference_urls_support_requirements",
        desc="Provide reference URL(s) that support the venue’s location, capacity, accessibility features, and scheduled-performance/accessible-seating availability",
        parent=node,
        critical=True
    )

    # Location: in Manhattan's Theater District (critical)
    loc_leaf = evaluator.add_leaf(
        id="location_in_manhattan_theater_district",
        desc="Venue is located in Manhattan's Theater District in New York City",
        parent=node,
        critical=True
    )
    location_claim = f"The venue '{v.name if v and v.name else 'the theater'}' is located in Manhattan's Theater District in New York City."
    await evaluator.verify(
        claim=location_claim,
        node=loc_leaf,
        sources=urls,
        additional_instruction="Verify the theater is a Broadway theater in Manhattan's Theater District (Midtown Manhattan). Accept phrasing like 'in the Theater District' or addresses within the Theater District boundaries.",
        extra_prerequisites=[urls_node]
    )

    # Capacity between 500 and 700 (critical)
    cap_leaf = evaluator.add_leaf(
        id="capacity_between_500_and_700",
        desc="Venue seating capacity is between 500 and 700 seats (inclusive)",
        parent=node,
        critical=True
    )
    cap_claim = "The theater's seating capacity is between 500 and 700 seats (inclusive)."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=urls,
        additional_instruction="Check the venue's auditorium capacity. If a specific number like 580 or 650 is shown, that satisfies the range condition.",
        extra_prerequisites=[urls_node]
    )

    # Accessibility features (critical)
    acc_leaf = evaluator.add_leaf(
        id="accessibility_features_present",
        desc="Venue provides wheelchair accessible seating, aisle transfer seats, and companion seating",
        parent=node,
        critical=True
    )
    acc_claim = "The theater provides wheelchair accessible seating, aisle transfer seats (aka transfer arm seats), and companion seating."
    await evaluator.verify(
        claim=acc_claim,
        node=acc_leaf,
        sources=urls,
        additional_instruction="Look for ADA/accessibility pages or seating policies. Accept synonyms like 'transfer arm aisle seats' for aisle transfer seats; 'companion seats' for companion seating.",
        extra_prerequisites=[urls_node]
    )

    # Scheduled performances with accessible seating (critical)
    sched_leaf = evaluator.add_leaf(
        id="scheduled_performances_with_accessible_seating_available",
        desc="Venue has scheduled performances and accessible seating options are available",
        parent=node,
        critical=True
    )
    sched_claim = "There are scheduled performances at the theater and accessible seating options are available for purchase for those performances."
    await evaluator.verify(
        claim=sched_claim,
        node=sched_leaf,
        sources=urls,
        additional_instruction="Ticketing or event pages often indicate accessible seats or an 'Accessible' filter. Verify upcoming performances with accessible seating.",
        extra_prerequisites=[urls_node]
    )


async def verify_venue_2_amphitheater(evaluator: Evaluator, root_node, v: Optional[AmphitheaterVenue]) -> None:
    node = evaluator.add_parallel(
        id="venue_2_outdoor_amphitheater_ca",
        desc="Venue 2: Outdoor amphitheater in California meeting capacity, accessibility pricing-tier, and active ticket-sales constraints",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = _bool_present(v.name) if v else False
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_name_provided",
        desc="Provide the venue name",
        parent=node,
        critical=True
    )
    loc_provided_ok = (_bool_present(v.city) and _bool_present(v.state)) if v else False
    evaluator.add_custom_node(
        result=loc_provided_ok,
        id="location_city_ca_provided_and_valid",
        desc="Provide the venue location including city and state, and the state is California",
        parent=node,
        critical=True
    )

    # URLs presence prerequisite (critical)
    urls = (v.reference_urls if v else []) or []
    urls_node = evaluator.add_custom_node(
        result=_urls_present(urls),
        id="reference_urls_support_requirements",
        desc="Provide reference URL(s) that support the venue’s location, capacity, accessible seating/price tiers, and upcoming-events/active-ticket-sales status",
        parent=node,
        critical=True
    )

    # State validity: California (critical)
    state_leaf = evaluator.add_leaf(
        id="state_is_california_valid",
        desc="Venue is located in California (valid)",
        parent=node,
        critical=True
    )
    state_claim = f"The venue '{v.name if v and v.name else 'the amphitheater'}' is located in California."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=urls,
        additional_instruction="Verify the state is California (CA) on the official venue page or reputable sources.",
        extra_prerequisites=[urls_node]
    )

    # Capacity between 5,000 and 10,000 (critical)
    cap_leaf = evaluator.add_leaf(
        id="capacity_between_5000_and_10000",
        desc="Venue seating capacity is between 5,000 and 10,000 (inclusive)",
        parent=node,
        critical=True
    )
    cap_claim = "The amphitheater's seating capacity is between 5,000 and 10,000 (inclusive)."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=urls,
        additional_instruction="Use official venue specs or reputable sources that state capacity numbers. If a range like 9,500 appears, that satisfies the criterion.",
        extra_prerequisites=[urls_node]
    )

    # Accessible seating in at least two price tiers (critical)
    ada_tiers_leaf = evaluator.add_leaf(
        id="accessible_seating_two_price_tiers",
        desc="Venue offers ADA/wheelchair accessible seating in at least two different price tiers",
        parent=node,
        critical=True
    )
    ada_tiers_claim = "The venue offers ADA/wheelchair-accessible seating available in at least two different price tiers."
    await evaluator.verify(
        claim=ada_tiers_claim,
        node=ada_tiers_leaf,
        sources=urls,
        additional_instruction="Check ticketing pages or seat maps for accessible seats available in multiple price categories (e.g., reserved seating sections with different prices, lawn vs. reserved, etc.).",
        extra_prerequisites=[urls_node]
    )

    # Upcoming events with active ticket sales (critical)
    sales_leaf = evaluator.add_leaf(
        id="upcoming_events_with_active_ticket_sales",
        desc="Venue has upcoming events publicly listed and tickets are actively for sale for at least one upcoming event",
        parent=node,
        critical=True
    )
    sales_claim = "The amphitheater has upcoming events listed and tickets are actively on sale for at least one upcoming event."
    await evaluator.verify(
        claim=sales_claim,
        node=sales_leaf,
        sources=urls,
        additional_instruction="Look for event calendars and 'Buy Tickets' buttons for upcoming dates.",
        extra_prerequisites=[urls_node]
    )


async def verify_venue_3_theme_park(evaluator: Evaluator, root_node, v: Optional[ThemeParkVenue]) -> None:
    node = evaluator.add_parallel(
        id="venue_3_theme_park_fl",
        desc="Venue 3: Theme park in Florida meeting ticket-price, group-discount, and operational-hours constraints",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = _bool_present(v.name) if v else False
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_name_provided",
        desc="Provide the theme park name",
        parent=node,
        critical=True
    )
    loc_provided_ok = (_bool_present(v.city) and _bool_present(v.state)) if v else False
    evaluator.add_custom_node(
        result=loc_provided_ok,
        id="location_city_fl_provided_and_valid",
        desc="Provide the theme park location including city and state, and the state is Florida",
        parent=node,
        critical=True
    )

    # URLs presence prerequisite (critical)
    urls = (v.reference_urls if v else []) or []
    urls_node = evaluator.add_custom_node(
        result=_urls_present(urls),
        id="reference_urls_support_requirements",
        desc="Provide reference URL(s) that support the park’s location, ticket pricing/type, group discount terms, and operational status/operating hours",
        parent=node,
        critical=True
    )

    # State validity: Florida (critical)
    state_leaf = evaluator.add_leaf(
        id="state_is_florida_valid",
        desc="Theme park is located in Florida (valid)",
        parent=node,
        critical=True
    )
    state_claim = f"The theme park '{v.name if v and v.name else 'the park'}' is located in Florida."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=urls,
        additional_instruction="Verify state is Florida (FL) on the official site or reputable sources.",
        extra_prerequisites=[urls_node]
    )

    # Lowest-priced single-day adult ticket under $100 and type matches (critical)
    price_leaf = evaluator.add_leaf(
        id="single_day_adult_ticket_under_100_and_type_matches",
        desc="Lowest-priced single-day adult admission ticket is under $100 before taxes/fees, and the ticket is single-day, single-park general admission",
        parent=node,
        critical=True
    )
    price_claim = "The lowest-priced single-day, single-park adult general admission ticket is under $100 before taxes and fees."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=urls,
        additional_instruction="Ignore multi-day, multi-park, or seasonal passes. Confirm the cheapest single-day adult general admission price is < $100 before taxes/fees.",
        extra_prerequisites=[urls_node]
    )

    # Group discount for 15 or fewer (critical)
    group_leaf = evaluator.add_leaf(
        id="group_discount_for_15_or_fewer",
        desc="Theme park offers a group discount program that applies to groups with a minimum size requirement of 15 or fewer people",
        parent=node,
        critical=True
    )
    group_claim = "The theme park offers group discounts with a minimum group size requirement of 15 or fewer people."
    await evaluator.verify(
        claim=group_claim,
        node=group_leaf,
        sources=urls,
        additional_instruction="Check group sales pages. Accept minimums like 10+, 12+, or 15+ as satisfying '15 or fewer'.",
        extra_prerequisites=[urls_node]
    )

    # Operational with published hours (critical)
    hours_leaf = evaluator.add_leaf(
        id="operational_with_published_hours",
        desc="Theme park is currently operational and has published operating hours",
        parent=node,
        critical=True
    )
    hours_claim = "The theme park is currently operational and publishes operating hours."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=urls,
        additional_instruction="Look for calendars or hours of operation pages showing current operating hours.",
        extra_prerequisites=[urls_node]
    )


async def verify_venue_4_stadium_suite(evaluator: Evaluator, root_node, v: Optional[StadiumSuiteVenue]) -> None:
    node = evaluator.add_parallel(
        id="venue_4_sports_stadium_suites_tx",
        desc="Venue 4: Sports stadium in Texas offering luxury suite rentals meeting capacity, inclusions, and upcoming-availability constraints",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = _bool_present(v.name) if v else False
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_name_provided",
        desc="Provide the stadium name",
        parent=node,
        critical=True
    )
    loc_provided_ok = (_bool_present(v.city) and _bool_present(v.state)) if v else False
    evaluator.add_custom_node(
        result=loc_provided_ok,
        id="location_city_tx_provided_and_valid",
        desc="Provide the stadium location including city and state, and the state is Texas",
        parent=node,
        critical=True
    )

    # URLs presence prerequisite (critical)
    urls = (v.reference_urls if v else []) or []
    urls_node = evaluator.add_custom_node(
        result=_urls_present(urls),
        id="reference_urls_support_requirements",
        desc="Provide reference URL(s) that support the stadium’s location, suite rental offering, suite capacity, inclusions (tickets + VIP parking), and upcoming availability",
        parent=node,
        critical=True
    )

    # State validity: Texas (critical)
    state_leaf = evaluator.add_leaf(
        id="state_is_texas_valid",
        desc="Stadium is located in Texas (valid)",
        parent=node,
        critical=True
    )
    state_claim = f"The stadium '{v.name if v and v.name else 'the stadium'}' is located in Texas."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=urls,
        additional_instruction="Verify state is Texas (TX) on the official venue page or reputable sources.",
        extra_prerequisites=[urls_node]
    )

    # Suite rentals offered (critical)
    suites_offered_leaf = evaluator.add_leaf(
        id="suite_rentals_offered_to_public_or_corporate",
        desc="Stadium offers luxury suite rentals to the public and/or corporate clients",
        parent=node,
        critical=True
    )
    suites_offered_claim = "The stadium offers luxury suite rentals to the public and/or corporate clients."
    await evaluator.verify(
        claim=suites_offered_claim,
        node=suites_offered_leaf,
        sources=urls,
        additional_instruction="Look for 'suites' or 'premium suites' rental information indicating availability for purchase/lease.",
        extra_prerequisites=[urls_node]
    )

    # Suite capacity 20–30 (critical)
    suite_cap_leaf = evaluator.add_leaf(
        id="suite_capacity_between_20_and_30",
        desc="Luxury suite rental accommodates between 20 and 30 guests (inclusive) or has suite options within this range",
        parent=node,
        critical=True
    )
    suite_cap_claim = "At least one luxury suite option accommodates between 20 and 30 guests (inclusive)."
    await evaluator.verify(
        claim=suite_cap_claim,
        node=suite_cap_leaf,
        sources=urls,
        additional_instruction="Accept any suite type at this stadium whose listed capacity falls within 20–30 (e.g., 20, 22, 24, 26, 28, or 30).",
        extra_prerequisites=[urls_node]
    )

    # Includes tickets and VIP/premium parking (critical)
    includes_leaf = evaluator.add_leaf(
        id="suite_includes_tickets_and_vip_parking",
        desc="Suite rental includes event tickets and VIP/premium parking passes",
        parent=node,
        critical=True
    )
    includes_claim = "Suite rentals include event tickets and VIP or premium parking passes."
    await evaluator.verify(
        claim=includes_claim,
        node=includes_leaf,
        sources=urls,
        additional_instruction="Accept synonyms like 'premium parking', 'VIP parking', or 'parking passes included' together with tickets included.",
        extra_prerequisites=[urls_node]
    )

    # Suites available for upcoming scheduled events (critical)
    avail_leaf = evaluator.add_leaf(
        id="suites_available_for_upcoming_scheduled_events",
        desc="Suites are available to rent for upcoming scheduled events",
        parent=node,
        critical=True
    )
    avail_claim = "Suites are available to rent for upcoming scheduled events at the stadium."
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=urls,
        additional_instruction="Look for upcoming event calendars and messaging like 'Suites available', 'Inquire for upcoming games/events', or 'Request availability' tied to specific upcoming events.",
        extra_prerequisites=[urls_node]
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
    Evaluate an answer for the multi-venue entertainment guide task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate four venues independently
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Build verification subtrees for each venue (in parallel if desired)
    await verify_venue_1_broadway(evaluator, root, extracted.broadway if extracted else None)
    await verify_venue_2_amphitheater(evaluator, root, extracted.amphitheater if extracted else None)
    await verify_venue_3_theme_park(evaluator, root, extracted.theme_park if extracted else None)
    await verify_venue_4_stadium_suite(evaluator, root, extracted.stadium_suite if extracted else None)

    return evaluator.get_summary()