import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "complete_travel_itinerary_2026"
TASK_DESCRIPTION = """A US citizen is planning a complex international journey in 2026 with the following requirements:

Flight Requirements:
1. The traveler must fly from Melbourne, Australia to Istanbul, Turkey with a connection at Kuala Lumpur International Airport (KLIA).
2. The first flight leg (Melbourne to Kuala Lumpur) must be operated by Singapore Airlines.
3. The second flight leg (Kuala Lumpur to Istanbul) must be operated by Turkish Airlines.
4. Both airlines must be members of the same airline alliance to enable baggage through-checking on separate ticket bookings.
5. The traveler will book Economy Class tickets.
6. The connection time at KLIA must meet the airport's minimum connection time requirement of 60 minutes, and preferably should meet the recommended 2-hour buffer for international transfers.

Baggage Requirements:
7. Provide the checked baggage allowance for Turkish Airlines international Economy Class flights, including the number of bags permitted, maximum weight per bag, and maximum total dimensions per bag.
8. Confirm that the airline alliance membership enables baggage through-checking between Singapore Airlines and Turkish Airlines even when booked on separate tickets.

Accommodation Requirements:
9. After arriving in Istanbul, the traveler will later travel to Yellowstone National Park in the United States and wishes to stay at Old Faithful Inn.
10. Specify when reservations for Old Faithful Inn open (how many months in advance and on which day of each month).
11. Identify one available room type category at Old Faithful Inn.
12. Confirm the operating season for Old Faithful Inn to ensure the planned dates fall within the property's open period.

Visa and Entry Requirements:
13. Confirm the visa requirements for US citizens traveling to Portugal (specifically the Azores) for tourism stays under 90 days.

Additional Benefits:
14. Describe one Star Alliance benefit that applies to travelers with Gold status when flying on member airlines.

Provide a comprehensive travel plan that addresses all these requirements, including specific details about the flights, baggage policies, accommodation booking procedures, visa requirements, and alliance benefits. Include reference URLs for each major component to support your answer.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class FlightsExtraction(BaseModel):
    leg1_origin: Optional[str] = None  # e.g., MEL or "Melbourne"
    leg1_destination: Optional[str] = None  # e.g., KUL or "Kuala Lumpur"
    leg1_operator: Optional[str] = None  # e.g., "Singapore Airlines" or "SQ"

    leg2_origin: Optional[str] = None  # e.g., KUL
    leg2_destination: Optional[str] = None  # e.g., IST
    leg2_operator: Optional[str] = None  # e.g., "Turkish Airlines" or "TK"

    connection_airport: Optional[str] = None  # e.g., "KUL", "KLIA", or "Kuala Lumpur International Airport"
    cabin_class: Optional[str] = None  # e.g., "Economy"
    connection_time: Optional[str] = None  # free text like "1h 45m", "120 minutes", etc.

    flights_sources: List[str] = Field(default_factory=list)  # URLs supporting flights/connection claims


class BaggageExtraction(BaseModel):
    turkish_bags: Optional[str] = None  # e.g., "2 bags"
    turkish_weight_per_bag: Optional[str] = None  # e.g., "23 kg"
    turkish_max_dimensions: Optional[str] = None  # e.g., "158 cm total (L+W+H)"
    through_checking_statement: Optional[str] = None  # as stated in the answer about alliance-enabled through-checking
    baggage_sources: List[str] = Field(default_factory=list)  # URLs for baggage & through-checking


class AccommodationExtraction(BaseModel):
    property_name: Optional[str] = None  # expect "Old Faithful Inn"
    reservation_opening_months_in_advance: Optional[str] = None  # e.g., "13 months"
    reservation_opening_day_of_month: Optional[str] = None  # e.g., "5th"
    room_type: Optional[str] = None  # any valid room category at Old Faithful Inn
    operating_season: Optional[str] = None  # e.g., "late April/early May to early/mid-October"
    accommodation_sources: List[str] = Field(default_factory=list)  # URLs for lodging info


class VisaBenefitExtraction(BaseModel):
    portugal_visa_under_90_days_statement: Optional[str] = None  # the statement as provided in the answer
    star_alliance_gold_benefit: Optional[str] = None  # one benefit description (e.g., "lounge access")
    visa_benefit_sources: List[str] = Field(default_factory=list)  # URLs for visa rules and alliance benefits


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flights() -> str:
    return """
    Extract the proposed flight plan details and supporting sources from the answer.

    Return a JSON object with the following fields:
    - leg1_origin: The departure city or IATA code for the first leg (e.g., "Melbourne" or "MEL").
    - leg1_destination: The arrival city or IATA code for the first leg (e.g., "Kuala Lumpur" or "KUL").
    - leg1_operator: The operating carrier for the first leg (e.g., "Singapore Airlines" or "SQ").
    - leg2_origin: The departure city or IATA code for the second leg (e.g., "Kuala Lumpur" or "KUL").
    - leg2_destination: The arrival city or IATA code for the second leg (e.g., "Istanbul" or "IST").
    - leg2_operator: The operating carrier for the second leg (e.g., "Turkish Airlines" or "TK").
    - connection_airport: The connection airport name or IATA code (e.g., "Kuala Lumpur International Airport", "KLIA", or "KUL").
    - cabin_class: The cabin stated for the tickets (e.g., "Economy").
    - connection_time: The planned connection time description if provided (e.g., "1h 20m", "2 hours").
    - flights_sources: An array of all URLs provided in the answer that support the routing, operating carriers, cabin class, flight times, or KLIA transfer timing.

    If any field is missing from the answer, set it to null (or an empty array for flights_sources).
    """


def prompt_extract_baggage() -> str:
    return """
    Extract Turkish Airlines international Economy checked baggage details and any statement about alliance-enabled through-checking on separate tickets, along with supporting sources.

    Return a JSON object with:
    - turkish_bags: Number of checked bags allowed (as stated), e.g., "1 bag", "2 bags".
    - turkish_weight_per_bag: Maximum weight per checked bag (as stated), e.g., "23 kg".
    - turkish_max_dimensions: Maximum total linear dimensions per bag (as stated), e.g., "158 cm".
    - through_checking_statement: The exact statement from the answer about baggage through-checking between Singapore Airlines and Turkish Airlines on separate tickets (include limitations if stated).
    - baggage_sources: An array of all URLs cited in the answer that support Turkish Airlines baggage rules or the alliance through-checking policy.

    If something is not specified, return null (or an empty array for baggage_sources).
    """


def prompt_extract_accommodation() -> str:
    return """
    Extract the Old Faithful Inn lodging details and supporting sources from the answer.

    Return a JSON object with:
    - property_name: The name of the property (should be "Old Faithful Inn" if correctly specified).
    - reservation_opening_months_in_advance: How many months in advance reservations open (e.g., "13 months").
    - reservation_opening_day_of_month: Which day of the month reservations open (e.g., "5th").
    - room_type: One room type/category available at Old Faithful Inn (e.g., "Old House Room with Bath", "Standard Room", "Suite").
    - operating_season: The operating season/open period wording as stated in the answer (e.g., "late April/early May to early/mid-October").
    - accommodation_sources: An array of all URLs supporting the booking window, room types, and operating season for Old Faithful Inn.

    If something is not specified, return null (or an empty array for accommodation_sources).
    """


def prompt_extract_visa_benefit() -> str:
    return """
    Extract the visa requirement statement for US citizens visiting Portugal (including the Azores) for tourism under 90 days, and one Star Alliance Gold benefit, plus their sources.

    Return a JSON object with:
    - portugal_visa_under_90_days_statement: The exact statement about visa requirements for US citizens for stays under 90 days.
    - star_alliance_gold_benefit: One Gold-tier benefit as stated in the answer (e.g., "lounge access", "priority check-in", "extra baggage", "priority boarding").
    - visa_benefit_sources: An array of URLs that support these claims (e.g., official government sites for visa rules; staralliance.com for benefits).

    If something is not specified, return null (or an empty array for visa_benefit_sources).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_flights_section(evaluator: Evaluator, parent_node, flights: FlightsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Flights_MEL_to_IST_via_KLIA",
        desc="Flight plan satisfies routing, operating-carrier, cabin, and connection-time requirements.",
        parent=parent_node,
        critical=False
    )

    sources = flights.flights_sources or []

    # References existence (critical within this section)
    flights_refs = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Flights_References",
        desc="Provides at least one reference URL relevant to the flights/connection-time claims in this section.",
        parent=node,
        critical=True
    )

    # Required routing via KLIA
    routing_leaf = evaluator.add_leaf(
        id="Required_Routing_via_KLIA",
        desc="Specifies travel from Melbourne (MEL) to Istanbul (IST) with a connection at Kuala Lumpur International Airport (KUL/KLIA).",
        parent=node,
        critical=True
    )
    routing_claim = (
        "This page shows a valid flight itinerary or schedule that routes the traveler from Melbourne (MEL) to "
        "Istanbul (IST) with a connection at Kuala Lumpur International Airport (KUL/KLIA). "
        "The evidence should explicitly reflect MEL → KUL (or Kuala Lumpur/ KLIA) → IST or equivalent wording."
    )
    await evaluator.verify(
        claim=routing_claim,
        node=routing_leaf,
        sources=sources,
        additional_instruction="Allow minor variations in airport naming (e.g., 'Melbourne', 'MEL'; 'Kuala Lumpur', 'KUL', 'KLIA'; 'Istanbul', 'IST').",
        extra_prerequisites=[flights_refs],
    )

    # Leg 1 operated by Singapore Airlines
    leg1_leaf = evaluator.add_leaf(
        id="Leg1_Operated_by_Singapore_Airlines",
        desc="States the first leg (MEL→KUL) is operated by Singapore Airlines.",
        parent=node,
        critical=True
    )
    leg1_claim = (
        "This page shows that the Melbourne to Kuala Lumpur flight leg is operated by Singapore Airlines (SQ). "
        "Look for 'operated by Singapore Airlines' or carrier indicators confirming SQ as the operating carrier."
    )
    await evaluator.verify(
        claim=leg1_claim,
        node=leg1_leaf,
        sources=sources,
        additional_instruction="Accept codeshare displays if they explicitly say 'operated by Singapore Airlines'.",
        extra_prerequisites=[flights_refs],
    )

    # Leg 2 operated by Turkish Airlines
    leg2_leaf = evaluator.add_leaf(
        id="Leg2_Operated_by_Turkish_Airlines",
        desc="States the second leg (KUL→IST) is operated by Turkish Airlines.",
        parent=node,
        critical=True
    )
    leg2_claim = (
        "This page shows that the Kuala Lumpur to Istanbul flight leg is operated by Turkish Airlines (TK). "
        "Look for 'operated by Turkish Airlines' or carrier indicators confirming TK as the operating carrier."
    )
    await evaluator.verify(
        claim=leg2_claim,
        node=leg2_leaf,
        sources=sources,
        additional_instruction="Accept minor naming variants like 'Turkish' or code 'TK' if clearly indicating the operating carrier.",
        extra_prerequisites=[flights_refs],
    )

    # Economy class
    economy_leaf = evaluator.add_leaf(
        id="Economy_Class",
        desc="States that tickets are booked in Economy Class.",
        parent=node,
        critical=True
    )
    economy_claim = (
        "This page indicates that the selected flights are booked in Economy Class (also acceptable synonyms include 'Economy', 'Y', 'Main Cabin' where applicable)."
    )
    await evaluator.verify(
        claim=economy_claim,
        node=economy_leaf,
        sources=sources,
        additional_instruction="Focus on cabin indicators; 'Economy' wording on booking/schedule pages is sufficient.",
        extra_prerequisites=[flights_refs],
    )

    # Meets minimum connection time at KLIA (≥ 60 minutes)
    mct_leaf = evaluator.add_leaf(
        id="Meets_Minimum_Connection_Time_KLIA",
        desc="Connection time at KLIA is ≥ 60 minutes (minimum connection time requirement).",
        parent=node,
        critical=True
    )
    mct_claim = (
        "The planned connection time at Kuala Lumpur International Airport (KLIA/KUL) between the two flights is at least 60 minutes. "
        "Use the shown arrival and departure times on the page to confirm the layover duration meets or exceeds 60 minutes."
    )
    await evaluator.verify(
        claim=mct_claim,
        node=mct_leaf,
        sources=sources,
        additional_instruction="If multiple options are shown, at least one clearly meets ≥ 60 minutes.",
        extra_prerequisites=[flights_refs],
    )

    # Preferred buffer (≥ 2 hours) - non-critical
    buffer_leaf = evaluator.add_leaf(
        id="Meets_Preferred_2_Hour_Buffer",
        desc="Connection time at KLIA is ≥ 2 hours (preferred/recommended buffer).",
        parent=node,
        critical=False
    )
    buffer_claim = (
        "The planned connection time at Kuala Lumpur International Airport (KLIA/KUL) is at least 2 hours (120 minutes). "
        "Confirm by comparing the two flight times on the provided page(s)."
    )
    await evaluator.verify(
        claim=buffer_claim,
        node=buffer_leaf,
        sources=sources,
        additional_instruction="If only one option is shown and it's ≥ 120 minutes, consider this supported.",
        extra_prerequisites=[flights_refs],
    )


async def verify_baggage_section(evaluator: Evaluator, parent_node, baggage: BaggageExtraction) -> None:
    node = evaluator.add_parallel(
        id="Baggage_and_Through_Checking",
        desc="Provides Turkish Airlines Economy checked baggage allowance details and confirms alliance-based through-checking on separate tickets.",
        parent=parent_node,
        critical=False
    )

    sources = baggage.baggage_sources or []

    # References existence (critical)
    bag_refs = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Baggage_References",
        desc="Provides at least one reference URL relevant to the baggage and through-checking claims in this section.",
        parent=node,
        critical=True
    )

    # Turkish Economy checked baggage allowance
    baggage_leaf = evaluator.add_leaf(
        id="Turkish_Economy_Checked_Baggage_Allowance",
        desc="States Turkish Airlines international Economy checked baggage allowance including number of checked bags, max weight per bag, and max total dimensions per bag.",
        parent=node,
        critical=True
    )
    bags = baggage.turkish_bags or "the stated number of checked bags"
    weight = baggage.turkish_weight_per_bag or "the stated maximum weight per bag"
    dims = baggage.turkish_max_dimensions or "the stated maximum total dimensions per bag"
    baggage_claim = (
        f"For Turkish Airlines international Economy Class, the checked baggage allowance includes {bags}, with {weight}, "
        f"and maximum total dimensions per bag of {dims}. The page supports these specific values."
    )
    await evaluator.verify(
        claim=baggage_claim,
        node=baggage_leaf,
        sources=sources,
        additional_instruction=(
            "Use official Turkish Airlines baggage policy or route-calculator pages where possible. "
            "Both piece-concept (e.g., 2×23 kg, 158 cm) and weight-concept regimes exist; the claim is supported if the cited page "
            "explicitly matches the stated numbers and dimensions for the described international Economy context."
        ),
        extra_prerequisites=[bag_refs],
    )

    # Alliance through-checking on separate tickets + same alliance
    through_leaf = evaluator.add_leaf(
        id="Alliance_Through_Checking_on_Separate_Tickets",
        desc="Confirms that Singapore Airlines and Turkish Airlines are in the same alliance and that this enables/permits baggage through-checking even when booked on separate tickets (or states applicable limitations/conditions).",
        parent=node,
        critical=True
    )
    if baggage.through_checking_statement and baggage.through_checking_statement.strip():
        through_claim = baggage.through_checking_statement.strip()
    else:
        through_claim = (
            "Singapore Airlines and Turkish Airlines are both Star Alliance members, and the provided sources confirm whether baggage "
            "can be through-checked between them on separate tickets (including any applicable limitations or conditions)."
        )
    await evaluator.verify(
        claim=through_claim,
        node=through_leaf,
        sources=sources,
        additional_instruction=(
            "Verification requires BOTH: (1) each airline's Star Alliance membership; and (2) whether through-checking on separate tickets "
            "is permitted or not, per Star Alliance and/or airline policy. If sources indicate limitations or non-availability on separate tickets, "
            "the claim should reflect that accordingly."
        ),
        extra_prerequisites=[bag_refs],
    )


async def verify_accommodation_section(evaluator: Evaluator, parent_node, acc: AccommodationExtraction) -> None:
    node = evaluator.add_parallel(
        id="Accommodation_Old_Faithful_Inn",
        desc="Addresses Old Faithful Inn booking timing, room category, and operating season.",
        parent=parent_node,
        critical=False
    )

    sources = acc.accommodation_sources or []

    # References existence (critical)
    acc_refs = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Accommodation_References",
        desc="Provides at least one reference URL relevant to the accommodation claims in this section.",
        parent=node,
        critical=True
    )

    # Property specified
    prop_leaf = evaluator.add_leaf(
        id="Property_Specified",
        desc="States the lodging is Old Faithful Inn (Yellowstone National Park).",
        parent=node,
        critical=True
    )
    prop_name = acc.property_name or "Old Faithful Inn"
    prop_claim = f"{prop_name} is a lodging property located in Yellowstone National Park."
    await evaluator.verify(
        claim=prop_claim,
        node=prop_leaf,
        sources=sources,
        additional_instruction="Do not confuse with Old Faithful Snow Lodge; confirm specifically 'Old Faithful Inn'.",
        extra_prerequisites=[acc_refs],
    )

    # Reservation opening policy (months in advance + day of month)
    open_leaf = evaluator.add_leaf(
        id="Reservation_Opening_Policy",
        desc="Specifies when reservations open: how many months in advance and which day of each month.",
        parent=node,
        critical=True
    )
    months = acc.reservation_opening_months_in_advance or "the stated number of months"
    dom = acc.reservation_opening_day_of_month or "the stated day"
    opening_claim = (
        f"Reservations for Old Faithful Inn open {months} in advance and bookings are released on the {dom} of each month (rolling release)."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=open_leaf,
        sources=sources,
        additional_instruction="Check official Yellowstone Lodges/Xanterra booking policy pages for the rolling monthly release schedule.",
        extra_prerequisites=[acc_refs],
    )

    # One room type category
    room_leaf = evaluator.add_leaf(
        id="One_Room_Type_Category",
        desc="Identifies one available room type category at Old Faithful Inn.",
        parent=node,
        critical=True
    )
    room_type = acc.room_type or "a listed room type at Old Faithful Inn"
    room_claim = f"Old Faithful Inn offers a room type/category named '{room_type}'."
    await evaluator.verify(
        claim=room_claim,
        node=room_leaf,
        sources=sources,
        additional_instruction="Accept any legitimate room category listed for Old Faithful Inn (e.g., Old House rooms, Standard, Deluxe, Suite).",
        extra_prerequisites=[acc_refs],
    )

    # Operating season
    season_leaf = evaluator.add_leaf(
        id="Operating_Season_Stated",
        desc="States the operating season/open period for Old Faithful Inn.",
        parent=node,
        critical=True
    )
    season_text = acc.operating_season or "the operating season as described"
    season_claim = f"Old Faithful Inn operates seasonally and is open during {season_text}."
    await evaluator.verify(
        claim=season_claim,
        node=season_leaf,
        sources=sources,
        additional_instruction="Verify the typical seasonal opening and closing period from official pages.",
        extra_prerequisites=[acc_refs],
    )


async def verify_visa_benefit_section(evaluator: Evaluator, parent_node, vb: VisaBenefitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Visa_and_Alliance_Benefit",
        desc="Covers Portugal/Azores entry rules for US citizens and one Star Alliance Gold benefit.",
        parent=parent_node,
        critical=False
    )

    sources = vb.visa_benefit_sources or []

    # References existence (critical)
    vb_refs = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Visa_and_Benefit_References",
        desc="Provides at least one reference URL relevant to the visa/entry and Star Alliance Gold benefit claims in this section.",
        parent=node,
        critical=True
    )

    # Portugal/Azores visa requirement for US citizens under 90 days
    visa_leaf = evaluator.add_leaf(
        id="Portugal_Azores_Visa_Under_90_Days",
        desc="States visa requirements for US citizens visiting Portugal (Azores) for tourism stays under 90 days.",
        parent=node,
        critical=True
    )
    if vb.portugal_visa_under_90_days_statement and vb.portugal_visa_under_90_days_statement.strip():
        visa_claim = vb.portugal_visa_under_90_days_statement.strip()
    else:
        visa_claim = (
            "US citizens visiting Portugal, including the Azores, for tourism for up to 90 days within any 180-day period "
            "do not require a visa (standard Schengen short-stay rules; other entry requirements like ETIAS may apply)."
        )
    await evaluator.verify(
        claim=visa_claim,
        node=visa_leaf,
        sources=sources,
        additional_instruction="Prefer official government or embassy sources for Schengen/Portugal short-stay rules; accept mention of ETIAS if applicable.",
        extra_prerequisites=[vb_refs],
    )

    # One Star Alliance Gold benefit
    gold_leaf = evaluator.add_leaf(
        id="One_Star_Alliance_Gold_Benefit",
        desc="Describes one correct Star Alliance Gold benefit applicable when flying on Star Alliance member airlines.",
        parent=node,
        critical=True
    )
    benefit = vb.star_alliance_gold_benefit or "a Star Alliance Gold benefit (e.g., lounge access, priority check-in, extra baggage, priority boarding)"
    benefit_claim = f"Star Alliance Gold members are entitled to: {benefit} when flying on Star Alliance member airlines (subject to conditions)."
    await evaluator.verify(
        claim=benefit_claim,
        node=gold_leaf,
        sources=sources,
        additional_instruction="Use staralliance.com or member-airline benefit pages to confirm the named Gold benefit.",
        extra_prerequisites=[vb_refs],
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract all sections (in parallel)
    flights_task = evaluator.extract(
        prompt=prompt_extract_flights(),
        template_class=FlightsExtraction,
        extraction_name="flights_extraction",
    )
    baggage_task = evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageExtraction,
        extraction_name="baggage_extraction",
    )
    accommodation_task = evaluator.extract(
        prompt=prompt_extract_accommodation(),
        template_class=AccommodationExtraction,
        extraction_name="accommodation_extraction",
    )
    visa_benefit_task = evaluator.extract(
        prompt=prompt_extract_visa_benefit(),
        template_class=VisaBenefitExtraction,
        extraction_name="visa_benefit_extraction",
    )

    flights, baggage, accommodation, visa_benefit = await asyncio.gather(
        flights_task, baggage_task, accommodation_task, visa_benefit_task
    )

    # Build rubric root node (non-critical at root to allow mixed critical children under it)
    complete_node = evaluator.add_parallel(
        id="Complete_Travel_Itinerary",
        desc="Answer addresses all specified flight, baggage, accommodation, visa/entry, alliance benefit, and sourcing requirements.",
        parent=root,
        critical=False
    )

    # Verify each major component
    await verify_flights_section(evaluator, complete_node, flights)
    await verify_baggage_section(evaluator, complete_node, baggage)
    await verify_accommodation_section(evaluator, complete_node, accommodation)
    await verify_visa_benefit_section(evaluator, complete_node, visa_benefit)

    # Optional: add some custom info about sources count
    evaluator.add_custom_info(
        {
            "flights_sources_count": len(flights.flights_sources or []),
            "baggage_sources_count": len(baggage.baggage_sources or []),
            "accommodation_sources_count": len(accommodation.accommodation_sources or []),
            "visa_benefit_sources_count": len(visa_benefit.visa_benefit_sources or []),
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    return evaluator.get_summary()