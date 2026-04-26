import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "florida_package_verification_2026"
TASK_DESCRIPTION = (
    "A non-US resident family of 6 people (4 adults and 2 children aged 10 and 12) has booked a Florida vacation package for March 2026 that includes the following components: "
    "(1) A 4-night Carnival cruise departing from Port Canaveral with vehicle parking for 7 days at a stated rate of $20 per day plus tax, "
    "(2) 3-day Disney World Park Hopper Plus tickets that include water park access, "
    "(3) A 2-night stay at a Disney Deluxe Resort with Extended Evening Hours access, "
    "(4) An America the Beautiful Annual Pass for visiting Everglades National Park. "
    "Verify that each component of this package accurately meets the stated specifications, and identify all additional fees, requirements, and deadlines that apply specifically to this non-US resident family. "
    "Include in your answer: confirmation of which Carnival embarkation deadlines and requirements must be met, verification of what is included in the Park Hopper Plus tickets, confirmation of which Disney resort benefits the family qualifies for, "
    "the correct type and cost of the America the Beautiful Pass for non-residents, any additional fees that apply at Everglades National Park for this family, and Port Canaveral parking payment requirements. "
    "Your answer must provide specific details for each component with supporting source URLs."
)


# ---------------------------- Data Models ----------------------------------
class CruiseInfo(BaseModel):
    operator: Optional[str] = None
    departure_port: Optional[str] = None
    duration_nights: Optional[str] = None

    parking_rate: Optional[str] = None  # e.g., "$20 per day plus tax"
    parking_includes_arrival_departure: Optional[str] = None  # "yes"/"no" or text
    parking_payment_credit_only: Optional[str] = None  # "yes"/"no" or text
    parking_open_time: Optional[str] = None  # e.g., "10 AM"

    online_checkin_deadline: Optional[str] = None  # e.g., "midnight ET the day before sailing"
    checked_baggage_cutoff: Optional[str] = None  # e.g., "2 hours before published departure time"
    final_boarding_deadline: Optional[str] = None  # e.g., "Final Boarding time on boarding pass"

    # Source URLs for verification
    cruise_sources: List[str] = Field(default_factory=list)
    parking_sources: List[str] = Field(default_factory=list)
    embarkation_sources: List[str] = Field(default_factory=list)


class DisneyTicketsInfo(BaseModel):
    ticket_type: Optional[str] = None  # e.g., "Park Hopper Plus"
    ticket_days: Optional[str] = None  # e.g., "3 days"
    park_hopper_rules: Optional[str] = None  # notes about hopping rules
    park_hopper_plus_inclusions: List[str] = Field(default_factory=list)  # e.g., ["Blizzard Beach", ...]
    ticket_sources: List[str] = Field(default_factory=list)


class DisneyResortInfo(BaseModel):
    stay_nights: Optional[str] = None  # e.g., "2 nights"
    resort_category: Optional[str] = None  # e.g., "Disney Deluxe Resort"
    extended_evening_hours_eligibility_rule: Optional[str] = None  # rule text
    extended_evening_hours_definition: Optional[str] = None  # e.g., "2 hours after park closing on select nights"
    early_theme_park_entry_rule: Optional[str] = None  # e.g., "30 minutes early for all Disney Resort hotel guests daily"
    resort_sources: List[str] = Field(default_factory=list)


class NationalParkPassInfo(BaseModel):
    pass_type: Optional[str] = None  # e.g., "America the Beautiful Annual Pass"
    nonresident_cost: Optional[str] = None  # e.g., "$250"
    coverage_agencies: List[str] = Field(default_factory=list)  # e.g., ["NPS", "USFS", ...]
    everglades_nonresident_fee_rule: Optional[str] = None  # e.g., "non-US residents 16+ pay $100 unless admitted with pass"
    family_composition: Optional[str] = None  # e.g., "4 adults and 2 children aged 10 and 12"
    pass_sources: List[str] = Field(default_factory=list)
    everglades_sources: List[str] = Field(default_factory=list)


class FloridaPackageExtraction(BaseModel):
    cruise: Optional[CruiseInfo] = None
    disney_tickets: Optional[DisneyTicketsInfo] = None
    disney_resort: Optional[DisneyResortInfo] = None
    national_park_pass: Optional[NationalParkPassInfo] = None


# ---------------------------- Extraction Prompt ----------------------------
def prompt_extract_package() -> str:
    return """
    Extract all specific package details and all cited source URLs exactly as stated in the answer. Do not invent anything.
    Structure the JSON with these sections and fields:

    cruise:
      operator: The operator name stated for the cruise (e.g., "Carnival Cruise Line")
      departure_port: The stated departure port (e.g., "Port Canaveral")
      duration_nights: The stated cruise duration (e.g., "4 nights")
      parking_rate: The stated Port Canaveral parking rate wording (e.g., "$20 per day plus tax")
      parking_includes_arrival_departure: Does the rate include both arrival and departure days? Return "yes", "no", or null if not stated.
      parking_payment_credit_only: Are only major credit cards accepted (no cash)? Return "yes", "no", or null if not stated.
      parking_open_time: The stated opening time for Port Canaveral parking on embarkation day (e.g., "10 AM")
      online_checkin_deadline: The stated deadline for completing online check-in and Arrival Appointment selection (e.g., "midnight ET the day before sailing")
      checked_baggage_cutoff: The stated cutoff for checked baggage service (e.g., "2 hours before published departure time")
      final_boarding_deadline: The stated final boarding requirement (e.g., "printed Final Boarding time on boarding pass")
      cruise_sources: List of all URLs the answer cites specifically for cruise operator/port/duration confirmation
      parking_sources: List of all URLs the answer cites specifically for Port Canaveral parking rules/rates/payment/opening time
      embarkation_sources: List of all URLs the answer cites specifically for Carnival embarkation deadlines/requirements

    disney_tickets:
      ticket_type: The stated ticket type (e.g., "Park Hopper Plus")
      ticket_days: The stated number of days (e.g., "3")
      park_hopper_rules: The stated Park Hopper rules (e.g., "can visit multiple theme parks per day after entering the first park; subject to capacity")
      park_hopper_plus_inclusions: Array of all venues stated as included in Park Hopper Plus (e.g., Blizzard Beach, Typhoon Lagoon, ESPN Wide World of Sports Complex, Disney's Oak Trail Golf Course)
      ticket_sources: List of all URLs the answer cites for ticket rules/inclusions

    disney_resort:
      stay_nights: The stated resort stay duration (e.g., "2 nights")
      resort_category: The stated hotel category (e.g., "Disney Deluxe Resort")
      extended_evening_hours_eligibility_rule: The stated eligibility rule for Extended Evening Hours
      extended_evening_hours_definition: The stated definition/duration (e.g., "2 hours after park closing on select nights")
      early_theme_park_entry_rule: The stated Early Theme Park Entry rule (e.g., "30 minutes before regular opening for all Disney Resort hotel guests daily")
      resort_sources: List of all URLs the answer cites for resort category and benefits

    national_park_pass:
      pass_type: The stated pass type (e.g., "America the Beautiful Annual Pass")
      nonresident_cost: The stated pass cost for non-US residents (e.g., "$250") if mentioned
      coverage_agencies: Array of agency acronyms stated as covered by the pass (e.g., NPS, USFS, BLM, FWS, BOR, USACE)
      everglades_nonresident_fee_rule: The stated Everglades non-US resident fee rule (e.g., "non-US residents age 16+ must pay $100 unless admitted with an Annual Pass")
      family_composition: The stated family composition (should be "4 adults and 2 children aged 10 and 12")
      pass_sources: List of all URLs the answer cites for pass pricing/coverage/eligibility
      everglades_sources: List of all URLs the answer cites for Everglades fees/rules

    SPECIAL RULES FOR URL EXTRACTION:
    - Only include URLs explicitly present in the answer text (plain links or markdown).
    - Include complete URLs with protocol. If missing, prepend "http://".
    - If a section has no URLs cited, return an empty array for that section's sources.

    If any field is not stated, return null (or an empty array for URL lists).
    """


# ---------------------------- Helper Utils ---------------------------------
def coalesce_urls(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if isinstance(u, str) and u.strip():
                merged.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# ---------------------------- Verification Builders ------------------------
async def build_cruise_component(evaluator: Evaluator, parent_node, data: FloridaPackageExtraction) -> None:
    cruise = data.cruise or CruiseInfo()

    # Cruise Component (critical)
    cruise_node = evaluator.add_parallel(
        id="Cruise_Component",
        desc="Verify the Carnival cruise details, Port Canaveral parking terms, and Carnival embarkation deadlines/requirements.",
        parent=parent_node,
        critical=True,
    )

    # Cruise Core Specs (critical)
    core_node = evaluator.add_parallel(
        id="Cruise_Core_Specs",
        desc="Cruise matches stated operator/port/duration specifications and port satisfies the allowed-Florida-port constraint.",
        parent=cruise_node,
        critical=True,
    )

    # Operated by Carnival
    carnival_node = evaluator.add_leaf(
        id="Operated_By_Carnival",
        desc="Confirms the cruise is operated by Carnival Cruise Line.",
        parent=core_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cruise is operated by Carnival Cruise Line.",
        node=carnival_node,
        sources=coalesce_urls(cruise.cruise_sources),
        additional_instruction="Treat 'Carnival' and 'Carnival Cruise Line' as equivalent naming. Verify the operator per the cited source(s).",
    )

    # Departure Port: Port Canaveral and allowed port constraint
    port_node = evaluator.add_leaf(
        id="Departure_Port_PortCanaveral_And_Allowed",
        desc="Confirms the cruise departs from Port Canaveral, which is one of Carnival's allowed Florida ports (Miami, Port Canaveral, Tampa, Jacksonville).",
        parent=core_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This cruise departs from Port Canaveral, and Port Canaveral is one of Carnival's Florida homeports.",
        node=port_node,
        sources=coalesce_urls(cruise.cruise_sources, cruise.parking_sources),
        additional_instruction="Confirm both the departure port (Port Canaveral) for this sailing and that Port Canaveral is a Carnival Florida port.",
    )

    # Cruise duration 4 nights
    duration_node = evaluator.add_leaf(
        id="Cruise_Duration_4_Nights",
        desc="Confirms the cruise duration is 4 nights.",
        parent=core_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cruise duration is 4 nights.",
        node=duration_node,
        sources=coalesce_urls(cruise.cruise_sources),
        additional_instruction="Verify the sailing length on the cited source(s).",
    )

    # Parking requirements (critical)
    parking_node = evaluator.add_parallel(
        id="Port_Canaveral_Parking_Requirements",
        desc="Parking requirements for Port Canaveral per constraints.",
        parent=cruise_node,
        critical=True,
    )

    # Parking rate $20 per day plus tax
    rate_node = evaluator.add_leaf(
        id="Parking_Rate_20_Per_Day_Plus_Tax",
        desc="Confirms parking rate is $20.00 per day plus tax.",
        parent=parking_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Port Canaveral cruise parking is $20 per day plus tax.",
        node=rate_node,
        sources=coalesce_urls(cruise.parking_sources),
        additional_instruction="Verify the publicly stated on-site cruise parking rate at Port Canaveral.",
    )

    # Parking includes arrival and departure days
    include_days_node = evaluator.add_leaf(
        id="Parking_Day_Count_Includes_Arrival_And_Departure",
        desc="Confirms the rate includes both arrival and departure days.",
        parent=parking_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Port Canaveral parking day count/rate includes both the arrival day and the departure day.",
        node=include_days_node,
        sources=coalesce_urls(cruise.parking_sources),
        additional_instruction="Look for any policy statement indicating that parking charges apply to both arrival and departure days.",
    )

    # Payment: credit only, no cash
    payment_node = evaluator.add_leaf(
        id="Parking_Payment_Credit_Only_No_Cash",
        desc="Confirms only major credit cards are accepted and no cash is accepted.",
        parent=parking_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Port Canaveral cruise parking accepts major credit cards only and does not accept cash.",
        node=payment_node,
        sources=coalesce_urls(cruise.parking_sources),
        additional_instruction="Verify accepted payment methods and confirm cash is not accepted.",
    )

    # Parking open time 10 AM (set as critical to satisfy framework constraints)
    open_time_node = evaluator.add_leaf(
        id="Parking_Open_Time_10AM",
        desc="Confirms Port Canaveral parking opens at 10 AM for embarkation.",
        parent=parking_node,
        critical=True,
    )
    await evaluator.verify(
        claim="On embarkation day, Port Canaveral parking opens at 10 AM.",
        node=open_time_node,
        sources=coalesce_urls(cruise.parking_sources),
        additional_instruction="Verify the posted opening time for cruise parking on embarkation day.",
    )

    # Embarkation deadlines (critical)
    embark_node = evaluator.add_parallel(
        id="Carnival_Embarkation_Deadlines",
        desc="Embarkation deadlines and requirements per constraints.",
        parent=cruise_node,
        critical=True,
    )

    # Online check-in deadline
    checkin_node = evaluator.add_leaf(
        id="Online_Checkin_And_Arrival_Appt_Deadline",
        desc="Confirms online check-in and Arrival Appointment selection must be completed by midnight ET the day before sailing.",
        parent=embark_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Online check-in and Arrival Appointment selection must be completed by midnight ET the day before sailing.",
        node=checkin_node,
        sources=coalesce_urls(cruise.embarkation_sources),
        additional_instruction="Verify Carnival's official policy regarding the deadline for online check-in and arrival time selection.",
    )

    # Checked baggage cutoff
    baggage_node = evaluator.add_leaf(
        id="Checked_Baggage_Cutoff",
        desc="Confirms checked baggage service is only available until 2 hours before the ship's published departure time.",
        parent=embark_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Checked baggage service is available only until 2 hours before the ship's published departure time.",
        node=baggage_node,
        sources=coalesce_urls(cruise.embarkation_sources),
        additional_instruction="Verify Carnival's checked baggage cutoff window.",
    )

    # Final boarding deadline
    final_board_node = evaluator.add_leaf(
        id="Final_Boarding_Deadline",
        desc="Confirms all guests must be on board by the Final Boarding time printed on the boarding pass.",
        parent=embark_node,
        critical=True,
    )
    await evaluator.verify(
        claim="All guests must be on board by the Final Boarding time printed on the boarding pass.",
        node=final_board_node,
        sources=coalesce_urls(cruise.embarkation_sources),
        additional_instruction="Verify the official wording about final boarding requirements.",
    )

    # Supporting sources presence (critical existence check)
    cruise_sources_exist = len(coalesce_urls(cruise.cruise_sources, cruise.parking_sources, cruise.embarkation_sources)) > 0
    evaluator.add_custom_node(
        result=cruise_sources_exist,
        id="Cruise_Supporting_Sources",
        desc="Provides supporting source URL(s) for the cruise/parking/embarkation claims made.",
        parent=cruise_node,
        critical=True,
    )


async def build_disney_tickets_component(evaluator: Evaluator, parent_node, data: FloridaPackageExtraction) -> None:
    tickets = data.disney_tickets or DisneyTicketsInfo()

    tickets_node = evaluator.add_parallel(
        id="Disney_Tickets_Component",
        desc="Verify the 3-day Disney World Park Hopper Plus tickets and included access.",
        parent=parent_node,
        critical=True,
    )

    # Ticket duration 3 days
    duration_node = evaluator.add_leaf(
        id="Ticket_Duration_3_Days",
        desc="Confirms the tickets are valid for 3 days.",
        parent=tickets_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This ticket product is valid for 3 days.",
        node=duration_node,
        sources=coalesce_urls(tickets.ticket_sources),
        additional_instruction="Verify the ticket validity period (3 days) per the cited source(s).",
    )

    # Park Hopper rules
    hopper_rules_node = evaluator.add_leaf(
        id="Park_Hopper_Rules",
        desc="Confirms Park Hopper allows visiting multiple theme parks per day after entering the first park, subject to capacity limitations.",
        parent=tickets_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Park Hopper allows visiting multiple theme parks per day after entering the first park, subject to capacity limitations.",
        node=hopper_rules_node,
        sources=coalesce_urls(tickets.ticket_sources),
        additional_instruction="Verify official Disney Park Hopper rules on visiting multiple parks and capacity limits.",
    )

    # Park Hopper Plus inclusions
    inclusions_node = evaluator.add_leaf(
        id="Park_Hopper_Plus_Inclusions_All",
        desc="Confirms Park Hopper Plus includes admission to Blizzard Beach, Typhoon Lagoon, ESPN Wide World of Sports Complex, and Disney's Oak Trail Golf Course.",
        parent=tickets_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Park Hopper Plus includes admission to Disney's Blizzard Beach, Disney's Typhoon Lagoon, ESPN Wide World of Sports Complex, and Disney's Oak Trail Golf Course.",
        node=inclusions_node,
        sources=coalesce_urls(tickets.ticket_sources),
        additional_instruction="Verify the Park Hopper Plus inclusions list from official Disney sources.",
    )

    # Supporting sources presence (critical existence check)
    evaluator.add_custom_node(
        result=len(coalesce_urls(tickets.ticket_sources)) > 0,
        id="Disney_Tickets_Supporting_Sources",
        desc="Provides supporting source URL(s) for the Disney ticket claims made (Park Hopper and Park Hopper Plus).",
        parent=tickets_node,
        critical=True,
    )


async def build_disney_resort_component(evaluator: Evaluator, parent_node, data: FloridaPackageExtraction) -> None:
    resort = data.disney_resort or DisneyResortInfo()

    resort_node = evaluator.add_parallel(
        id="Disney_Resort_Component",
        desc="Verify Disney Deluxe Resort stay and the benefits the family qualifies for.",
        parent=parent_node,
        critical=True,
    )

    # Stay duration 2 nights
    stay_node = evaluator.add_leaf(
        id="Stay_Duration_2_Nights",
        desc="Confirms the stay duration is 2 nights.",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The resort stay is 2 nights.",
        node=stay_node,
        sources=coalesce_urls(resort.resort_sources),
        additional_instruction="Verify the stated stay duration using the cited source(s).",
    )

    # Hotel category is Disney Deluxe Resort
    category_node = evaluator.add_leaf(
        id="Hotel_Is_Disney_Deluxe_Resort",
        desc="Confirms the hotel category is Disney Deluxe Resort (as stated in the package).",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel is categorized as a Disney Deluxe Resort.",
        node=category_node,
        sources=coalesce_urls(resort.resort_sources),
        additional_instruction="Verify the hotel's classification as a Disney Deluxe Resort on official Disney pages.",
    )

    # Extended Evening Hours eligibility
    eligibility_node = evaluator.add_leaf(
        id="Extended_Evening_Hours_Eligibility",
        desc="Confirms Extended Evening Hours are available only to guests staying at Disney Deluxe Resorts, Disney Deluxe Villa Resorts, or other select hotels, and thus whether the family qualifies given the stated hotel type.",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Extended Evening Hours are offered to guests staying at Disney Deluxe Resorts, Disney Deluxe Villa Resorts, or other select hotels.",
        node=eligibility_node,
        sources=coalesce_urls(resort.resort_sources),
        additional_instruction="Verify the official eligibility rule for Extended Evening Hours; since the stated hotel is a Disney Deluxe Resort, this family qualifies.",
    )

    # Extended Evening Hours definition
    definition_node = evaluator.add_leaf(
        id="Extended_Evening_Hours_Definition",
        desc="Confirms Extended Evening Hours are 2 hours after park closing on select nights at select parks.",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Extended Evening Hours run for 2 hours after park closing on select nights at select parks.",
        node=definition_node,
        sources=coalesce_urls(resort.resort_sources),
        additional_instruction="Verify the official definition/duration of Extended Evening Hours.",
    )

    # Early Theme Park Entry (set critical True to satisfy framework constraints)
    early_entry_node = evaluator.add_leaf(
        id="Early_Theme_Park_Entry",
        desc="Confirms Early Theme Park Entry (30 minutes before regular opening) is available to all Disney Resort hotel guests every day.",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Early Theme Park Entry allows Disney Resort hotel guests to enter 30 minutes before regular opening, available daily.",
        node=early_entry_node,
        sources=coalesce_urls(resort.resort_sources),
        additional_instruction="Verify Disney's official Early Theme Park Entry policy.",
    )

    # Supporting sources presence (critical existence check)
    evaluator.add_custom_node(
        result=len(coalesce_urls(resort.resort_sources)) > 0,
        id="Disney_Resort_Supporting_Sources",
        desc="Provides supporting source URL(s) for the Disney resort benefit claims made (Extended Evening Hours / Early Entry).",
        parent=resort_node,
        critical=True,
    )


async def build_national_park_pass_component(evaluator: Evaluator, parent_node, data: FloridaPackageExtraction) -> None:
    npass = data.national_park_pass or NationalParkPassInfo()

    np_node = evaluator.add_parallel(
        id="National_Park_Pass_Component",
        desc="Verify America the Beautiful pass type/cost for non-residents and Everglades-specific additional fee rules for this family.",
        parent=parent_node,
        critical=True,
    )

    # Pass is Annual Pass
    pass_type_node = evaluator.add_leaf(
        id="Pass_Is_Annual_Pass",
        desc="Confirms the package includes an America the Beautiful Annual Pass.",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The package includes an America the Beautiful Annual Pass.",
        node=pass_type_node,
        sources=coalesce_urls(npass.pass_sources),
        additional_instruction="Verify the pass type using official federal recreation or NPS sources.",
    )

    # Nonresident cost (verify the stated value from the answer)
    cost_node = evaluator.add_leaf(
        id="Nonresident_Cost_250",
        desc="Confirms the annual pass cost for non-US residents is $250 (per constraints).",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The annual pass cost for non-US residents is $250.",
        node=cost_node,
        sources=coalesce_urls(npass.pass_sources),
        additional_instruction="Verify the stated non-resident cost for the America the Beautiful Annual Pass on official sources.",
    )

    # Coverage agencies
    coverage_node = evaluator.add_leaf(
        id="Pass_Coverage_Agencies",
        desc="Confirms the pass covers entrance fees at federal recreation lands managed by the listed agencies (NPS, USFS, BLM, FWS, BOR, USACE).",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The America the Beautiful Annual Pass covers entrance fees at federal recreation lands managed by NPS, USFS, BLM, FWS, BOR, and USACE.",
        node=coverage_node,
        sources=coalesce_urls(npass.pass_sources),
        additional_instruction="Verify the agencies covered by the pass from official sources.",
    )

    # Everglades nonresident fee rule
    ev_rule_node = evaluator.add_leaf(
        id="Everglades_Nonresident_Fee_Rule",
        desc="States the rule: at Everglades, non-US residents age 16+ must pay an additional $100 per person unless admitted with an Annual Pass.",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At Everglades National Park, non-US residents age 16+ must pay an additional $100 per person unless admitted with an Annual Pass.",
        node=ev_rule_node,
        sources=coalesce_urls(npass.everglades_sources),
        additional_instruction="Verify the stated Everglades nonresident fee rule using official park sources.",
    )

    # Applies age threshold to family (simple logic check)
    age_map_node = evaluator.add_leaf(
        id="Applies_Age_Threshold_To_Family",
        desc="Correctly identifies which family members are age 16+ versus under 16 based on the given family composition (4 adults; children age 10 and 12).",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In a family of 4 adults and two children aged 10 and 12, the members aged 16+ are the four adults; the two children are under 16.",
        node=age_map_node,
        sources=None,
        additional_instruction="This is a straightforward logical determination from the provided family ages; no web evidence is required.",
    )

    # Determines whether fee is owed given Annual Pass
    owes_fee_node = evaluator.add_leaf(
        id="Determines_Whether_Fee_Is_Owed_Given_Annual_Pass",
        desc="Determines and states whether the additional $100/person Everglades fee is owed or waived due to admission with an Annual Pass, consistent with the stated rule.",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Because admission is with an America the Beautiful Annual Pass, the additional $100 per non-US resident age 16+ fee at Everglades is waived.",
        node=owes_fee_node,
        sources=coalesce_urls(npass.pass_sources, npass.everglades_sources),
        additional_instruction="Use the official pass coverage and Everglades fee rule to determine whether the additional fee applies or is waived.",
    )

    # Supporting sources presence (critical existence check - require at least one pass source and one Everglades source)
    evaluator.add_custom_node(
        result=(len(coalesce_urls(npass.pass_sources)) > 0 and len(coalesce_urls(npass.everglades_sources)) > 0),
        id="National_Park_Supporting_Sources",
        desc="Provides supporting source URL(s) for pass pricing/coverage and the Everglades nonresident-fee rule claims made.",
        parent=np_node,
        critical=True,
    )


# ---------------------------- Main Evaluation ------------------------------
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
        strategy=AggregationStrategy.PARALLEL,  # Root parallel; allow independent components
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

    # IMPORTANT: Root set as non-critical to allow non-critical checks deeper if needed.
    # We'll create an explicit root node mirroring the rubric root but non-critical to satisfy framework constraints.
    pkg_root = evaluator.add_parallel(
        id="Package_Verification",
        desc="Verify each package component meets the stated specifications and identify applicable fees/requirements/deadlines for the non-US resident family, with supporting source URLs.",
        parent=root,
        critical=False,
    )

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_package(),
        template_class=FloridaPackageExtraction,
        extraction_name="package_extraction",
    )

    # Build components
    await build_cruise_component(evaluator, pkg_root, extraction)
    await build_disney_tickets_component(evaluator, pkg_root, extraction)
    await build_disney_resort_component(evaluator, pkg_root, extraction)
    await build_national_park_pass_component(evaluator, pkg_root, extraction)

    # Return standard summary
    return evaluator.get_summary()