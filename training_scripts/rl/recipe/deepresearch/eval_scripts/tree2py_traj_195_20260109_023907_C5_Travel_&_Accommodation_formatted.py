import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_international_travel_planning"
TASK_DESCRIPTION = (
    "You are planning an international trip from California to Thailand. Your departure date is March 15, 2026, "
    "and your US passport expires on August 20, 2026. You have mobility needs and will be traveling with a service dog.\n\n"
    "Complete the following planning tasks:\n\n"
    "1. Passport Validity: Determine whether your passport meets Thailand's validity requirements for entry. Provide the specific requirement and cite an official US government source.\n\n"
    "2. Accessible Hotel: Identify one specific hotel in the Los Angeles, San Francisco, or San Diego area that offers ADA-compliant accessible rooms with a roll-in shower. Verify and describe the accessibility features including: grab bars, shower seat, handheld showerhead, and door width compliance. Confirm the hotel's service animal policy. Provide the hotel website or booking platform URL as reference.\n\n"
    "3. Airport Wheelchair Assistance: Select your departure airport (LAX, SFO, or San Diego International) and provide the advance notice requirement for requesting TSA Cares wheelchair assistance. Include the TSA Cares contact information and cite the official TSA source."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PassportExtraction(BaseModel):
    requirement_statement: Optional[str] = None
    us_gov_source_urls: List[str] = Field(default_factory=list)
    meets_six_month_rule: Optional[str] = None  # 'yes'/'no'/'unknown' as expressed in the answer
    assumed_arrival_date: Optional[str] = None  # e.g., 'Mar 16, 2026' or 'mid-March 2026'
    conclusion_assessment: Optional[str] = None  # free text summary of their assessment


class HotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    metro_area: Optional[str] = None  # 'Los Angeles'|'San Francisco'|'San Diego' or variations
    hotel_city: Optional[str] = None  # city as mentioned in the answer (optional)
    url_main: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Accessibility feature claims as stated in the answer ('yes'/'no'/'unknown' or free text)
    roll_in_shower: Optional[str] = None
    grab_bars: Optional[str] = None
    shower_seat: Optional[str] = None
    handheld_showerhead: Optional[str] = None
    door_width_32_clear: Optional[str] = None

    # Service animal policy claims
    service_animal_policy: Optional[str] = None
    service_animal_no_fee: Optional[str] = None
    service_animal_policy_url: Optional[str] = None


class TSAExtraction(BaseModel):
    selected_airport_code: Optional[str] = None  # e.g., LAX/SFO/SAN
    selected_airport_name: Optional[str] = None  # e.g., 'Los Angeles International Airport'
    tsa_cares_advance_notice_text: Optional[str] = None  # text stating 72 hours/3 days
    tsa_cares_contact_info_text: Optional[str] = None  # e.g., phone/form mention as stated in the answer
    tsa_cares_urls: List[str] = Field(default_factory=list)
    airline_coordination_note_present: Optional[str] = None  # 'yes'/'no' or similar


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_passport_section() -> str:
    return """
    From the answer, extract the passport validity requirement for Thailand and the user's assessment:

    Return a JSON with:
    - requirement_statement: The exact requirement text as stated (e.g., "Passport must be valid at least 6 months beyond the date of entry").
    - us_gov_source_urls: A list of URLs the answer cites for this requirement that are official US government sources (e.g., travel.state.gov, state.gov, usembassy.gov). Only include URLs explicitly present in the answer; otherwise return an empty list.
    - meets_six_month_rule: 'yes' if the answer concludes the passport meets the six-month-beyond-entry requirement; 'no' if it concludes it does not meet; 'unknown' if the answer does not clearly conclude.
    - assumed_arrival_date: The arrival date or window assumed by the answer for the Thailand entry (e.g., 'March 16, 2026', 'mid-March 2026'). If not specified, return null.
    - conclusion_assessment: A short free-text summary (1–2 sentences) of the answer’s conclusion about whether the passport satisfies the requirement.

    Do not invent URLs or facts; only extract what appears in the answer.
    """


def prompt_extract_hotel_section() -> str:
    return """
    Extract the hotel information and accessibility claims from the answer. Return a JSON with:
    - hotel_name: The specific property name.
    - metro_area: Which of 'Los Angeles', 'San Francisco', or 'San Diego' the hotel is in (as stated). Accept reasonable variants like 'LA', 'SF', etc. If not clearly stated, return null.
    - hotel_city: The city named for the property, if explicitly provided (e.g., 'Santa Monica', 'San Diego'). If not provided, return null.
    - url_main: The primary hotel webpage or booking listing URL cited by the answer for this hotel (hotel brand site or a booking platform URL).
    - additional_urls: Any other URLs provided that pertain to the hotel's accessibility or policies; only include URLs explicitly present in the answer.
    - roll_in_shower: As stated by the answer for this hotel: 'yes', 'no', or 'unknown'.
    - grab_bars: 'yes', 'no', or 'unknown'.
    - shower_seat: 'yes', 'no', or 'unknown'.
    - handheld_showerhead: 'yes', 'no', or 'unknown'.
    - door_width_32_clear: 'yes', 'no', or 'unknown' regarding at least 32 inches of clear width.
    - service_animal_policy: The service animal policy as stated in the answer (short text), or null if not stated.
    - service_animal_no_fee: 'yes', 'no', or 'unknown' as stated in the answer about extra fees for service animals.
    - service_animal_policy_url: If a separate policy URL is cited for service animals, include it; else null.

    Only extract what is explicitly present in the answer; do not infer or add external knowledge.
    """


def prompt_extract_tsa_section() -> str:
    return """
    Extract the TSA Cares information and the selected airport from the answer. Return a JSON with:
    - selected_airport_code: The code chosen from LAX, SFO, or SAN if present; else null.
    - selected_airport_name: The full airport name if provided (e.g., 'Los Angeles International Airport'); else null.
    - tsa_cares_advance_notice_text: The text from the answer regarding how many hours/days in advance to contact TSA Cares (e.g., '72 hours', '3 days').
    - tsa_cares_contact_info_text: The contact info text as provided (e.g., phone number or "via TSA Cares page/form"); else null.
    - tsa_cares_urls: A list of TSA official URLs (tsa.gov) cited in the answer for TSA Cares; only include URLs explicitly present in the answer.
    - airline_coordination_note_present: 'yes' if the answer notes that wheelchair assistance at airports is coordinated through airlines; 'no' if the answer does not include such a note.

    Only extract content that appears in the answer; do not invent any URLs or details.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_yes_no(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = value.strip().lower()
    if any(k in s for k in ["yes", "true", "y"]):
        return "yes"
    if any(k in s for k in ["no", "false", "n"]):
        return "no"
    if "unknown" in s or "unsure" in s or "not sure" in s or "not specified" in s:
        return "unknown"
    # heuristic: contains "does not meet" => no; "meets" => yes
    if "does not meet" in s or "doesn't meet" in s or "not meet" in s or "insufficient" in s:
        return "no"
    if "meets" in s or "satisfies" in s or "sufficient" in s:
        return "yes"
    return None


def _merge_urls(*args: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in args:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                result.append(u)
                seen.add(u)
    return result


def _any_gov_url(urls: List[str]) -> bool:
    for u in urls:
        if ".gov" in u.lower():
            return True
    return False


def _any_domain(urls: List[str], domain: str) -> bool:
    d = domain.lower()
    for u in urls:
        if d in u.lower():
            return True
    return False


def _canonical_metro(area: Optional[str]) -> Optional[str]:
    if not area:
        return None
    s = area.strip().lower()
    if s in ["la", "los angeles", "los angeles area", "los angeles county", "greater los angeles"]:
        return "Los Angeles"
    if s in ["sf", "san francisco", "san francisco bay area", "bay area"]:
        return "San Francisco"
    if "san diego" in s:
        return "San Diego"
    return None


def _allowed_airport_selected(code: Optional[str], name: Optional[str]) -> bool:
    code_ok = (code or "").strip().upper() in {"LAX", "SFO", "SAN"}
    name_s = (name or "").strip().lower()
    name_ok = any(
        key in name_s
        for key in [
            "los angeles international",  # LAX
            "san francisco international",  # SFO
            "san diego international",  # SAN
        ]
    )
    return code_ok or name_ok


# --------------------------------------------------------------------------- #
# Verification tree construction: subtrees                                    #
# --------------------------------------------------------------------------- #
async def build_passport_subtree(
    evaluator: Evaluator,
    parent_node,
    passport: PassportExtraction,
) -> None:
    passport_node = evaluator.add_parallel(
        id="Passport_Validity_Assessment",
        desc="Determine whether the passport meets Thailand entry validity requirements and cite an official US government source.",
        parent=parent_node,
        critical=True,
    )

    gov_urls = passport.us_gov_source_urls or []

    # 1) Six_Month_Validity_Requirement
    six_month_leaf = evaluator.add_leaf(
        id="Six_Month_Validity_Requirement",
        desc="State the requirement that passports must be valid for at least 6 months beyond the date of arrival/entry to Thailand.",
        parent=passport_node,
        critical=True,
    )
    six_month_claim = (
        "Thailand requires that passports be valid for at least 6 months beyond the date of entry (arrival) to Thailand."
    )
    await evaluator.verify(
        claim=six_month_claim,
        node=six_month_leaf,
        sources=gov_urls,
        additional_instruction=(
            "Verify that the cited official US government source (e.g., travel.state.gov) explicitly states "
            "the six-month passport validity requirement for entry to Thailand. Accept close paraphrases "
            "like 'six months of validity from date of entry.'"
        ),
    )

    # 2) Validity_Assessment_With_Date_Assumption
    meets = _norm_yes_no(passport.meets_six_month_rule) or "unknown"
    assumed_arrival = passport.assumed_arrival_date or "around March 15, 2026"
    does_phrase = "does" if meets == "yes" else "does not"
    assessment_leaf = evaluator.add_leaf(
        id="Validity_Assessment_With_Date_Assumption",
        desc=(
            "Assess whether a passport expiring on Aug 20, 2026 satisfies the 6-month-beyond-arrival rule for the trip, "
            "explicitly stating the assumed arrival date/window (or providing a conditional conclusion if arrival date is not given)."
        ),
        parent=passport_node,
        critical=True,
    )
    assessment_claim = (
        f"With arrival {assumed_arrival}, a passport expiring on August 20, 2026 {does_phrase} meet the requirement "
        f"of being valid at least 6 months beyond entry to Thailand."
    )
    await evaluator.verify(
        claim=assessment_claim,
        node=assessment_leaf,
        additional_instruction=(
            "Compute six months beyond the assumed arrival date/window. For example, six months after March 15, 2026 is approximately "
            "September 15, 2026. Conclude whether an August 20, 2026 expiration satisfies or fails that requirement. "
            "Judge the correctness of the answer's stated conclusion."
        ),
    )

    # 3) Official_US_Gov_Source_Citation (custom existence/domain check)
    gov_citation_leaf = evaluator.add_custom_node(
        result=_any_gov_url(gov_urls),
        id="Official_US_Gov_Source_Citation",
        desc="Cite an official US government source confirming Thailand's passport validity requirement.",
        parent=passport_node,
        critical=True,
    )


async def build_hotel_subtree(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelExtraction,
) -> None:
    hotel_node = evaluator.add_parallel(
        id="Accessible_Hotel_Identification",
        desc="Identify one qualifying accessible hotel in the specified California metro areas and verify required accessibility and service animal policy with a URL reference.",
        parent=parent_node,
        critical=True,
    )

    # Prepare source URLs
    ref_urls = _merge_urls(
        [hotel.url_main] if hotel.url_main else [],
        hotel.additional_urls,
        [hotel.service_animal_policy_url] if hotel.service_animal_policy_url else [],
    )

    # 1) Hotel_In_Allowed_Area
    hotel_area_leaf = evaluator.add_leaf(
        id="Hotel_In_Allowed_Area",
        desc="Name one specific hotel in the Los Angeles, San Francisco, or San Diego area.",
        parent=hotel_node,
        critical=True,
    )
    canonical_area = _canonical_metro(hotel.metro_area) or "Los Angeles/San Francisco/San Diego"
    hotel_name = hotel.hotel_name or "the specified hotel"
    city_text = f" in {hotel.hotel_city}" if hotel.hotel_city else ""
    area_claim = (
        f"The hotel's page/listing for '{hotel_name}' shows the property is located in the {canonical_area} area{city_text} "
        f"(one of Los Angeles, San Francisco, or San Diego)."
    )
    await evaluator.verify(
        claim=area_claim,
        node=hotel_area_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Verify the property's location on the referenced page(s). Accept neighborhoods or nearby cities commonly considered within "
            "these metros (e.g., Santa Monica/Long Beach for LA; Oakland/San Mateo for SF Bay; La Jolla/Carlsbad for San Diego). "
            "If no credible location context is visible, mark as not supported."
        ),
    )

    # 2) Hotel_Reference_URL (exists)
    ref_url_leaf = evaluator.add_custom_node(
        result=bool(ref_urls),
        id="Hotel_Reference_URL",
        desc="Provide a hotel website or booking platform URL that supports the stated accessibility claims.",
        parent=hotel_node,
        critical=True,
    )

    # 3) Accessible_Room_Roll_In_Shower
    rollin_leaf = evaluator.add_leaf(
        id="Accessible_Room_Roll_In_Shower",
        desc="Verify the hotel offers an ADA-accessible room with a roll-in shower.",
        parent=hotel_node,
        critical=True,
    )
    rollin_claim = (
        f"The hotel '{hotel_name}' offers at least one ADA-accessible room that includes a roll-in shower."
    )
    await evaluator.verify(
        claim=rollin_claim,
        node=rollin_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Look for phrases like 'roll-in shower' or an accessible bathroom description. Accept brand accessibility pages or booking "
            "platform listings that explicitly state 'roll-in shower' for this property."
        ),
    )

    # 4) Bathroom_Grab_Bars
    grab_bars_leaf = evaluator.add_leaf(
        id="Bathroom_Grab_Bars",
        desc="Verify the roll-in shower/bathroom includes grab bars.",
        parent=hotel_node,
        critical=True,
    )
    grab_bars_claim = (
        f"The accessible bathroom at '{hotel_name}' includes grab bars."
    )
    await evaluator.verify(
        claim=grab_bars_claim,
        node=grab_bars_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Look for the terms 'grab bars' or 'handrails' in bathroom accessibility features. "
            "Accept if explicitly stated on hotel or booking pages for this property."
        ),
    )

    # 5) Bathroom_Shower_Seat
    seat_leaf = evaluator.add_leaf(
        id="Bathroom_Shower_Seat",
        desc="Verify the roll-in shower includes a shower seat (fold-down or fixed).",
        parent=hotel_node,
        critical=True,
    )
    seat_claim = (
        f"The roll-in shower at '{hotel_name}' includes a shower seat (fold-down, fixed, or provided on request)."
    )
    await evaluator.verify(
        claim=seat_claim,
        node=seat_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Look for 'shower seat', 'fold-down seat', 'portable shower chair available' in the accessibility details."
        ),
    )

    # 6) Bathroom_Handheld_Showerhead
    handheld_leaf = evaluator.add_leaf(
        id="Bathroom_Handheld_Showerhead",
        desc="Verify the roll-in shower includes a handheld showerhead.",
        parent=hotel_node,
        critical=True,
    )
    handheld_claim = (
        f"The accessible shower at '{hotel_name}' includes a handheld (hand) showerhead."
    )
    await evaluator.verify(
        claim=handheld_claim,
        node=handheld_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Look for 'handheld shower', 'hand shower', or 'detachable showerhead' in the room or accessibility description."
        ),
    )

    # 7) Door_Width_32_Inch_Clearance
    door_leaf = evaluator.add_leaf(
        id="Door_Width_32_Inch_Clearance",
        desc="Verify door width compliance by confirming at least 32 inches of clear width (per the stated constraint).",
        parent=hotel_node,
        critical=True,
    )
    door_claim = (
        f"The accessible room/entrance doorways at '{hotel_name}' provide at least 32 inches (0.81 m) of clear width."
    )
    await evaluator.verify(
        claim=door_claim,
        node=door_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Look for explicit mentions of '32-inch wide doorways' or '32 inches of clear width'. "
            "If not mentioned or unclear, mark as not supported."
        ),
    )

    # 8) Service_Animal_Policy_No_Extra_Fee
    svc_animal_leaf = evaluator.add_leaf(
        id="Service_Animal_Policy_No_Extra_Fee",
        desc="Confirm the hotel's service animal policy accommodates service animals and does not charge additional fees for them (per the stated constraint).",
        parent=hotel_node,
        critical=True,
    )
    svc_animal_claim = (
        f"The hotel's policy for '{hotel_name}' permits service animals and does not charge additional fees for service animals."
    )
    await evaluator.verify(
        claim=svc_animal_claim,
        node=svc_animal_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Accept explicit statements like 'Service animals allowed' and 'No fee for service animals' or equivalent brand policy language. "
            "If a page charges pet fees but exempts service animals from fees, that supports the claim."
        ),
    )


async def build_tsa_subtree(
    evaluator: Evaluator,
    parent_node,
    tsa: TSAExtraction,
) -> None:
    # Note: JSON marks 'Airline_Coordination_Note' as NON-CRITICAL. However, the framework requires
    # all children of a critical parent to be critical. We therefore set it critical=True to satisfy
    # framework constraints, while still verifying it as a minor note.
    tsa_node = evaluator.add_parallel(
        id="Airport_Wheelchair_Assistance",
        desc="Select a permitted departure airport and provide TSA Cares wheelchair assistance advance notice requirement, contact information, and official TSA citation.",
        parent=parent_node,
        critical=True,
    )

    tsa_urls = tsa.tsa_cares_urls or []

    # 1) Departure_Airport_Selection (custom check)
    airport_sel_leaf = evaluator.add_custom_node(
        result=_allowed_airport_selected(tsa.selected_airport_code, tsa.selected_airport_name),
        id="Departure_Airport_Selection",
        desc="Select one departure airport from LAX, SFO, or San Diego International.",
        parent=tsa_node,
        critical=True,
    )

    # 2) Advance_Notice_Requirement_72_Hours
    notice_leaf = evaluator.add_leaf(
        id="Advance_Notice_Requirement_72_Hours",
        desc="State the TSA Cares advance-notice requirement of at least 72 hours (3 days) for assistance requests.",
        parent=tsa_node,
        critical=True,
    )
    notice_claim = (
        "The TSA Cares program asks travelers to contact TSA Cares at least 72 hours (3 days) before travel for assistance."
    )
    await evaluator.verify(
        claim=notice_claim,
        node=notice_leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Verify on official TSA pages (tsa.gov) that TSA Cares recommends or requests contacting at least 72 hours prior to travel."
        ),
    )

    # 3) TSA_Cares_Contact_Information
    contact_leaf = evaluator.add_leaf(
        id="TSA_Cares_Contact_Information",
        desc="Provide TSA Cares contact information (e.g., phone and/or official TSA webpage contact pathway) without requiring any specific phone number unless cited from TSA.",
        parent=tsa_node,
        critical=True,
    )
    contact_claim = (
        "The referenced TSA page(s) provide contact information and/or a contact pathway for TSA Cares (such as a phone number or online form)."
    )
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Check that the page includes a phone number or clearly provides a link or instructions to request assistance from TSA Cares."
        ),
    )

    # 4) Official_TSA_Source_Citation (custom)
    tsa_source_leaf = evaluator.add_custom_node(
        result=_any_domain(tsa_urls, "tsa.gov"),
        id="Official_TSA_Source_Citation",
        desc="Cite an official TSA source that supports the advance notice requirement and/or contact information.",
        parent=tsa_node,
        critical=True,
    )

    # 5) Airline_Coordination_Note (set critical True for framework consistency)
    airline_note_leaf = evaluator.add_leaf(
        id="Airline_Coordination_Note",
        desc="Note that wheelchair assistance at airports is coordinated through airlines (per the stated constraint).",
        parent=tsa_node,
        critical=True,
    )
    airline_note_claim = (
        "The answer includes a note that wheelchair assistance at airports is coordinated through airlines."
    )
    await evaluator.verify(
        claim=airline_note_claim,
        node=airline_note_leaf,
        additional_instruction=(
            "Check the answer text to see whether it explicitly mentions that wheelchair assistance is arranged via airlines."
        ),
    )


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
) -> Dict:
    # Initialize evaluator with a neutral root; attach our task root under it.
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

    # Create task root (critical, as per rubric)
    planning_root = evaluator.add_parallel(
        id="Accessible_International_Travel_Planning",
        desc="Plan an accessible international trip from California to Thailand by addressing passport validity, accessible lodging, and airport wheelchair assistance per the question and constraints.",
        parent=root,
        critical=True,
    )

    # Extract sections concurrently
    passport_task = evaluator.extract(
        prompt=prompt_extract_passport_section(),
        template_class=PassportExtraction,
        extraction_name="passport_section",
    )
    hotel_task = evaluator.extract(
        prompt=prompt_extract_hotel_section(),
        template_class=HotelExtraction,
        extraction_name="hotel_section",
    )
    tsa_task = evaluator.extract(
        prompt=prompt_extract_tsa_section(),
        template_class=TSAExtraction,
        extraction_name="tsa_section",
    )

    passport_data, hotel_data, tsa_data = await asyncio.gather(passport_task, hotel_task, tsa_task)

    # Build subtrees (all critical under planning_root)
    await build_passport_subtree(evaluator, planning_root, passport_data)
    await build_hotel_subtree(evaluator, planning_root, hotel_data)
    await build_tsa_subtree(evaluator, planning_root, tsa_data)

    return evaluator.get_summary()