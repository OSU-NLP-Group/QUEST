import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bna_florida_feb2026"
TASK_DESCRIPTION = """
A family of four is planning a week-long vacation from Nashville International Airport (BNA) in mid-February 2026. Their trip will include visiting Walt Disney World and taking a short cruise from a Florida port. They need comprehensive information about their travel logistics.

Provide a complete travel plan that includes the following verified information:

1. Airport Services at BNA: Identify which premium airport lounge(s) are available at Nashville International Airport, the location of rest facilities, and where TSA PreCheck enrollment can be completed.

2. Airline and Baggage: Identify a budget airline that serves nonstop flights from BNA to Florida destinations, and provide their personal item size restrictions and checked baggage fee structure.

3. Disney Resort Accommodation: Identify a Disney World resort hotel that has Disney Skyliner access with direct connections to both EPCOT and Hollywood Studios.

4. Cruise Departure Port: Identify an accessible Florida cruise port where Royal Caribbean operates, including approximate distance or travel time from the Orlando area.

5. Travel Documentation: Confirm the current requirements for domestic air travel identification as of February 2026, including REAL ID status, any fees for alternative identification methods, and acceptance of digital IDs.

All information must be supported by reference URLs from the sources where this information was found.
"""


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class LoungeInfo(BaseModel):
    lounge_names: List[str] = Field(default_factory=list)
    lounge_urls: List[str] = Field(default_factory=list)
    mentions_centurion_not_available: Optional[bool] = None


class RestFacilityInfo(BaseModel):
    name: Optional[str] = None
    location_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TSAEnrollmentInfo(BaseModel):
    location_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BNAAirportServices(BaseModel):
    lounges: LoungeInfo = LoungeInfo()
    rest_facility: RestFacilityInfo = RestFacilityInfo()
    tsa_enrollment: TSAEnrollmentInfo = TSAEnrollmentInfo()


class AirlineInfo(BaseModel):
    airline_name: Optional[str] = None
    route_urls: List[str] = Field(default_factory=list)
    personal_item_size: Optional[str] = None
    personal_item_urls: List[str] = Field(default_factory=list)
    checked_bag_fee_text: Optional[str] = None
    checked_bag_urls: List[str] = Field(default_factory=list)

    # Optional merger context mentioned in rubric; extracted but not enforced as critical
    merger_mentioned: Optional[bool] = None
    merger_date: Optional[str] = None
    merger_urls: List[str] = Field(default_factory=list)


class ResortInfo(BaseModel):
    resort_name: Optional[str] = None
    resort_urls: List[str] = Field(default_factory=list)
    skyliner_epcot_urls: List[str] = Field(default_factory=list)
    skyliner_hs_urls: List[str] = Field(default_factory=list)


class CruisePortInfo(BaseModel):
    port_name: Optional[str] = None
    rc_urls: List[str] = Field(default_factory=list)  # Royal Caribbean operates at this port
    distance_or_time_text: Optional[str] = None
    distance_urls: List[str] = Field(default_factory=list)


class TravelDocsInfo(BaseModel):
    real_id_date_text: Optional[str] = None
    real_id_urls: List[str] = Field(default_factory=list)
    confirmid_fee_text: Optional[str] = None
    confirmid_urls: List[str] = Field(default_factory=list)
    digital_id_text: Optional[str] = None
    digital_id_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bna_services() -> str:
    return """
    Extract airport services at Nashville International Airport (BNA) from the answer.

    Return a JSON object with fields:
    - lounges:
        - lounge_names: list of premium lounge names explicitly stated (e.g., "Delta Sky Club", "American Airlines Admirals Club")
        - lounge_urls: list of URLs that support lounge availability at BNA
        - mentions_centurion_not_available: boolean, true only if the answer explicitly states that a Centurion Lounge is not available at BNA; otherwise false or null
    - rest_facility:
        - name: the rest facility name (e.g., "Minute Suites")
        - location_text: the location description provided (e.g., "Concourse D near/across from Gate D3")
        - urls: URLs supporting the rest facility location at BNA
    - tsa_enrollment:
        - location_text: the TSA PreCheck enrollment location description at BNA (e.g., "pre-security, first floor, north side near Welcome Desk")
        - urls: URLs supporting TSA PreCheck enrollment location at BNA

    Rules:
    - Extract only what appears in the answer text; do not invent.
    - URLs must be actual URLs present in the answer. If a section lacks URLs, return an empty list for that section.
    """


def prompt_extract_airline_info() -> str:
    return """
    Extract budget airline and baggage policy information relevant to nonstop BNA→Florida.

    Return a JSON object with fields:
    - airline_name: the identified airline (e.g., "Allegiant Air" or "Sun Country Airlines")
    - route_urls: URLs supporting that the airline operates nonstop flights from BNA to Florida destinations
    - personal_item_size: the personal item size restriction text (e.g., "18 x 14 x 8 inches")
    - personal_item_urls: URLs supporting the personal item size policy for the identified airline
    - checked_bag_fee_text: the checked baggage fee structure text (e.g., examples of fees by purchase channel)
    - checked_bag_urls: URLs supporting the checked baggage fees for the identified airline

    Optional (merger context):
    - merger_mentioned: boolean, true only if the answer mentions an Allegiant–Sun Country merger
    - merger_date: if merger_mentioned is true, the announcement date text from the answer (e.g., "January 11, 2026")
    - merger_urls: URLs supporting the merger announcement (if mentioned)

    Rules:
    - Extract only what appears in the answer text; do not invent.
    - URLs must be actual URLs present in the answer. If a field lacks URLs, return an empty list.
    """


def prompt_extract_resort_info() -> str:
    return """
    Extract the chosen Disney World resort and Skyliner connectivity references.

    Return a JSON object with fields:
    - resort_name: the selected resort name (e.g., "Disney's Pop Century Resort", "Disney's Caribbean Beach Resort")
    - resort_urls: URLs supporting that the resort has Disney Skyliner access
    - skyliner_epcot_urls: URLs supporting direct Skyliner connection from this resort to EPCOT (International Gateway)
    - skyliner_hs_urls: URLs supporting direct Skyliner connection from this resort to Disney's Hollywood Studios

    Rules:
    - Extract only what appears in the answer text; do not invent.
    - If separate URLs are not provided for EPCOT/HS connections, leave those arrays empty and rely on resort_urls.
    """


def prompt_extract_cruise_port_info() -> str:
    return """
    Extract cruise departure port information relevant to Royal Caribbean and Orlando proximity.

    Return a JSON object with fields:
    - port_name: the identified Florida cruise port (e.g., "Port Canaveral", "PortMiami", "Port Everglades", "Port of Tampa")
    - rc_urls: URLs supporting that Royal Caribbean operates from the identified port
    - distance_or_time_text: approximate distance or travel time from the Orlando area to the identified port (as stated in the answer)
    - distance_urls: URLs supporting the stated distance or travel time

    Rules:
    - Extract only what appears in the answer text; do not invent.
    - URLs must be actual URLs present in the answer. If a field lacks URLs, return an empty list.
    """


def prompt_extract_travel_docs_info() -> str:
    return """
    Extract domestic air travel identification requirements as of February 2026.

    Return a JSON object with fields:
    - real_id_date_text: the statement about REAL ID mandatory date (e.g., "REAL ID became mandatory on May 7, 2025")
    - real_id_urls: URLs supporting the REAL ID implementation date
    - confirmid_fee_text: the TSA ConfirmID fee statement (e.g., "$45 fee starting February 1, 2026; valid 10 days")
    - confirmid_urls: URLs supporting the TSA ConfirmID fee and validity
    - digital_id_text: the statement about Apple Digital ID acceptance (e.g., "Accepted at 250+ U.S. airports as of November 2025")
    - digital_id_urls: URLs supporting the digital ID acceptance statement

    Rules:
    - Extract only what appears in the answer text; do not invent.
    - URLs must be actual URLs present in the answer. If a field lacks URLs, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _list_or_fallback(primary: List[str], fallback: List[str]) -> List[str]:
    """Return primary if non-empty; otherwise fallback."""
    return primary if primary else fallback


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_airport_services(evaluator: Evaluator, parent_node, bna: BNAAirportServices) -> None:
    # Airport Services at BNA (critical, parallel)
    bna_node = evaluator.add_parallel(
        id="Airport_Services_at_BNA",
        desc="Accurate identification of three specific airport services available at Nashville International Airport",
        parent=parent_node,
        critical=True
    )

    # Premium Lounge Identification (critical, sequential)
    lounge_node = evaluator.add_sequential(
        id="Premium_Lounge_Identification",
        desc="Correctly identifies at least one premium lounge available at BNA and explicitly notes Centurion Lounge is NOT available",
        parent=bna_node,
        critical=True
    )

    # Existence of lounge URLs
    evaluator.add_custom_node(
        result=bool(bna.lounges.lounge_urls),
        id="Reference_URL_Lounges_Exist",
        desc="Provides at least one reference URL for BNA lounge information",
        parent=lounge_node,
        critical=True
    )

    # Lounge availability claim
    lounge_avail_leaf = evaluator.add_leaf(
        id="Lounge_Availability_Correct",
        desc="BNA has at least one premium lounge (Delta Sky Club or American Airlines Admirals Club)",
        parent=lounge_node,
        critical=True
    )
    lounge_claim = "Nashville International Airport (BNA) has at least one premium airline lounge such as Delta Sky Club or American Airlines Admirals Club."
    await evaluator.verify(
        claim=lounge_claim,
        node=lounge_avail_leaf,
        sources=bna.lounges.lounge_urls,
        additional_instruction="Verify that at least one of these lounges is present at BNA; accepting pages that list BNA's lounges or official lounge pages indicating location at BNA."
    )

    # Centurion Lounge not available (explicit statement verification with provided URLs)
    centurion_leaf = evaluator.add_leaf(
        id="Centurion_Not_Available",
        desc="BNA does not have an American Express Centurion Lounge",
        parent=lounge_node,
        critical=True
    )
    centurion_claim = "Nashville International Airport (BNA) does not have an American Express Centurion Lounge."
    await evaluator.verify(
        claim=centurion_claim,
        node=centurion_leaf,
        sources=bna.lounges.lounge_urls,
        additional_instruction="Support can come from official BNA lounge listings or American Express Centurion Lounge location list indicating no BNA location. If any provided source clearly implies absence, consider supported."
    )

    # Rest Facility Location (critical, sequential)
    rest_node = evaluator.add_sequential(
        id="Rest_Facility_Location",
        desc="Correctly identifies Minute Suites at Concourse D near/across from Gate D3",
        parent=bna_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(bna.rest_facility.urls),
        id="Reference_URL_Rest_Facility_Exist",
        desc="Provides at least one reference URL for Minute Suites location at BNA",
        parent=rest_node,
        critical=True
    )
    rest_leaf = evaluator.add_leaf(
        id="Minute_Suites_Location_Correct",
        desc="Minute Suites location at BNA is Concourse D near/across from Gate D3",
        parent=rest_node,
        critical=True
    )
    rest_claim = "Minute Suites at Nashville International Airport (BNA) are located in Concourse D near or across from Gate D3."
    await evaluator.verify(
        claim=rest_claim,
        node=rest_leaf,
        sources=bna.rest_facility.urls,
        additional_instruction="Accept minor wording variations like 'near Gate D3' or 'across from D3'. The page must clearly show Minute Suites location on Concourse D."
    )

    # TSA PreCheck Enrollment Location (critical, sequential)
    tsa_node = evaluator.add_sequential(
        id="TSA_PreCheck_Enrollment_Location",
        desc="Correctly identifies TSA PreCheck enrollment location at BNA as pre-security, 1st floor, north side near Welcome Desk",
        parent=bna_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(bna.tsa_enrollment.urls),
        id="Reference_URL_TSA_Enrollment_Exist",
        desc="Provides at least one reference URL for TSA PreCheck enrollment at BNA",
        parent=tsa_node,
        critical=True
    )
    tsa_leaf = evaluator.add_leaf(
        id="TSA_Enrollment_Location_Correct",
        desc="TSA PreCheck enrollment at BNA is pre-security, first floor, north side near the Welcome Desk",
        parent=tsa_node,
        critical=True
    )
    tsa_claim = "At BNA, TSA PreCheck enrollment is located pre-security on the first floor (north side) near the Welcome Desk."
    await evaluator.verify(
        claim=tsa_claim,
        node=tsa_leaf,
        sources=bna.tsa_enrollment.urls,
        additional_instruction="Focus on BNA's official directory or TSA enrollment partner pages that specify pre-security, first floor, north side near Welcome Desk."
    )


async def verify_airline_baggage(evaluator: Evaluator, parent_node, airline: AirlineInfo) -> None:
    # Airline and Baggage Information (critical, parallel)
    airline_node = evaluator.add_parallel(
        id="Airline_and_Baggage_Information",
        desc="Complete information about budget airline serving BNA-Florida route with baggage policies",
        parent=parent_node,
        critical=True
    )

    # Budget Airline Identification (critical, sequential)
    budget_node = evaluator.add_sequential(
        id="Budget_Airline_Identification",
        desc="Identifies Allegiant Air or Sun Country Airlines as serving BNA to Florida destinations",
        parent=airline_node,
        critical=True
    )
    # Name validity check (simple)
    name_leaf = evaluator.add_leaf(
        id="Airline_Name_One_Of",
        desc="Identified airline is either Allegiant Air or Sun Country Airlines",
        parent=budget_node,
        critical=True
    )
    name_claim = f"The identified airline '{airline.airline_name or ''}' is either Allegiant Air or Sun Country Airlines."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Allow minor name variants, e.g., 'Allegiant' or 'Sun Country'."
    )
    # Existence of route URLs
    evaluator.add_custom_node(
        result=bool(airline.route_urls),
        id="Reference_URL_Airline_Routes_Exist",
        desc="Provides at least one reference URL for airline route information",
        parent=budget_node,
        critical=True
    )
    # Route verification claim
    route_leaf = evaluator.add_leaf(
        id="Airline_Route_Verified",
        desc="The airline operates nonstop flights from BNA to Florida destinations",
        parent=budget_node,
        critical=True
    )
    route_claim = f"{airline.airline_name or 'The airline'} operates nonstop flights from Nashville (BNA) to at least one Florida destination."
    await evaluator.verify(
        claim=route_claim,
        node=route_leaf,
        sources=airline.route_urls,
        additional_instruction="Prefer official route maps, schedules, or airport pages indicating nonstop service BNA↔Florida for the identified airline."
    )

    # Personal Item Size Restrictions (critical, sequential)
    pi_node = evaluator.add_sequential(
        id="Personal_Item_Size_Restrictions",
        desc="Provides accurate personal item size restrictions for the identified airline",
        parent=airline_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(airline.personal_item_urls),
        id="Reference_URL_Personal_Item_Size_Exist",
        desc="Provides at least one reference URL for personal item size policy",
        parent=pi_node,
        critical=True
    )
    pi_leaf = evaluator.add_leaf(
        id="Personal_Item_Size_Correct",
        desc="Personal item size restriction is accurate for the identified airline",
        parent=pi_node,
        critical=True
    )
    pi_claim = f"For {airline.airline_name or 'the airline'}, the personal item size limit is {airline.personal_item_size or ''}."
    await evaluator.verify(
        claim=pi_claim,
        node=pi_leaf,
        sources=airline.personal_item_urls,
        additional_instruction="Common limits include around 18 x 14 x 8 inches. Verify exact policy text on the airline’s official baggage policy page."
    )

    # Checked Baggage Fee Structure (critical, sequential)
    cb_node = evaluator.add_sequential(
        id="Checked_Baggage_Fee_Structure",
        desc="Provides accurate checked bag fee information for the identified airline",
        parent=airline_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(airline.checked_bag_urls),
        id="Reference_URL_Checked_Bag_Fees_Exist",
        desc="Provides at least one reference URL for checked baggage fees",
        parent=cb_node,
        critical=True
    )
    cb_leaf = evaluator.add_leaf(
        id="Checked_Bag_Fees_Correct",
        desc="Checked baggage fee structure is accurate for the identified airline",
        parent=cb_node,
        critical=True
    )
    cb_claim = f"For {airline.airline_name or 'the airline'}, the checked baggage fees are described as: {airline.checked_bag_fee_text or ''}."
    await evaluator.verify(
        claim=cb_claim,
        node=cb_leaf,
        sources=airline.checked_bag_urls,
        additional_instruction="Verify fee ranges and conditions directly from the airline’s baggage fees page; accept minor variations in display (e.g., ranges by booking method)."
    )

    # Optional: Merger context — extracted but NOT enforced to avoid penalizing answers that omit it
    # (Omitted from verification tree to keep essential criteria intact.)


async def verify_resort(evaluator: Evaluator, parent_node, resort: ResortInfo) -> None:
    # Disney Resort Accommodation (critical, parallel)
    resort_node = evaluator.add_parallel(
        id="Disney_Resort_Accommodation",
        desc="Disney World resort selection with verified Skyliner access to both required theme parks",
        parent=parent_node,
        critical=True
    )

    # Resort Has Skyliner Access (critical, sequential)
    has_node = evaluator.add_sequential(
        id="Resort_Has_Skyliner_Access",
        desc="Identifies a Disney resort with Skyliner access (Pop Century, Art of Animation, Caribbean Beach, or Riviera)",
        parent=resort_node,
        critical=True
    )
    # Name validity check (simple)
    allowed_resorts = [
        "Disney's Pop Century Resort",
        "Disney’s Pop Century Resort",
        "Pop Century",
        "Disney's Art of Animation Resort",
        "Disney’s Art of Animation Resort",
        "Art of Animation",
        "Disney's Caribbean Beach Resort",
        "Disney’s Caribbean Beach Resort",
        "Caribbean Beach",
        "Disney's Riviera Resort",
        "Disney’s Riviera Resort",
        "Riviera"
    ]
    name_leaf = evaluator.add_leaf(
        id="Resort_Name_In_Allowed_List",
        desc="Chosen resort is one served by Disney Skyliner",
        parent=has_node,
        critical=True
    )
    name_claim = f"The chosen resort '{resort.resort_name or ''}' is one of Pop Century, Art of Animation, Caribbean Beach, or Disney's Riviera Resort."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Allow reasonable name variants and presence/absence of 'Disney's' prefix."
    )
    # Resort URLs existence
    evaluator.add_custom_node(
        result=bool(resort.resort_urls),
        id="Reference_URL_Skyliner_Resorts_Exist",
        desc="Provides at least one reference URL for Skyliner resort list/access",
        parent=has_node,
        critical=True
    )
    # Verify access by URLs
    has_leaf = evaluator.add_leaf(
        id="Skyliner_Access_Verified",
        desc="Resort is confirmed to have Disney Skyliner access",
        parent=has_node,
        critical=True
    )
    has_claim = f"{resort.resort_name or 'The resort'} has access to the Disney Skyliner transportation system."
    await evaluator.verify(
        claim=has_claim,
        node=has_leaf,
        sources=resort.resort_urls,
        additional_instruction="Prefer official Disney resort pages or Disney Skyliner pages confirming gondola access from the resort."
    )

    # Skyliner EPCOT Connection (critical, sequential)
    epcot_node = evaluator.add_sequential(
        id="Skyliner_EPCOT_Connection",
        desc="Confirms direct Skyliner connection to EPCOT from the identified resort",
        parent=resort_node,
        critical=True
    )
    epcot_sources = _list_or_fallback(resort.skyliner_epcot_urls, resort.resort_urls)
    evaluator.add_custom_node(
        result=bool(epcot_sources),
        id="Reference_URL_EPCOT_Connection_Exist",
        desc="Provides at least one reference URL for Skyliner–EPCOT connection",
        parent=epcot_node,
        critical=True
    )
    epcot_leaf = evaluator.add_leaf(
        id="EPCOT_Connection_Correct",
        desc="Skyliner provides direct connection to EPCOT (International Gateway) from the resort",
        parent=epcot_node,
        critical=True
    )
    epcot_claim = f"From {resort.resort_name or 'the resort'}, the Disney Skyliner provides a direct connection to EPCOT via the International Gateway station."
    await evaluator.verify(
        claim=epcot_claim,
        node=epcot_leaf,
        sources=epcot_sources,
        additional_instruction="Official Disney Skyliner maps or resort transport pages should indicate a direct EPCOT connection; minor wording differences are acceptable."
    )

    # Skyliner Hollywood Studios Connection (critical, sequential)
    hs_node = evaluator.add_sequential(
        id="Skyliner_Hollywood_Studios_Connection",
        desc="Confirms direct Skyliner connection to Hollywood Studios from the identified resort",
        parent=resort_node,
        critical=True
    )
    hs_sources = _list_or_fallback(resort.skyliner_hs_urls, resort.resort_urls)
    evaluator.add_custom_node(
        result=bool(hs_sources),
        id="Reference_URL_Hollywood_Studios_Connection_Exist",
        desc="Provides at least one reference URL for Skyliner–Hollywood Studios connection",
        parent=hs_node,
        critical=True
    )
    hs_leaf = evaluator.add_leaf(
        id="Hollywood_Studios_Connection_Correct",
        desc="Skyliner provides direct connection to Disney's Hollywood Studios from the resort",
        parent=hs_node,
        critical=True
    )
    hs_claim = f"From {resort.resort_name or 'the resort'}, the Disney Skyliner provides a direct connection to Disney's Hollywood Studios."
    await evaluator.verify(
        claim=hs_claim,
        node=hs_leaf,
        sources=hs_sources,
        additional_instruction="Official Skyliner maps or resort transport pages should indicate a direct Hollywood Studios connection; allow minor naming variants."
    )


async def verify_cruise_port(evaluator: Evaluator, parent_node, cruise: CruisePortInfo) -> None:
    # Cruise Departure Port Information (critical, parallel)
    cruise_node = evaluator.add_parallel(
        id="Cruise_Departure_Port_Information",
        desc="Identification of accessible Florida cruise port with distance information",
        parent=parent_node,
        critical=True
    )

    # Cruise Port Identification (critical, sequential)
    port_id_node = evaluator.add_sequential(
        id="Cruise_Port_Identification",
        desc="Correctly identifies a major Florida cruise port where Royal Caribbean operates",
        parent=cruise_node,
        critical=True
    )
    # Port name validity (simple)
    allowed_ports = [
        "Port Canaveral",
        "PortMiami",
        "Port of Miami",
        "Miami",
        "Port Everglades",
        "Fort Lauderdale",
        "Port of Tampa",
        "Port Tampa Bay",
        "Tampa"
    ]
    port_leaf = evaluator.add_leaf(
        id="Port_Name_One_Of",
        desc="Identified port is among major Florida ports (Canaveral, Miami, Fort Lauderdale, Tampa)",
        parent=port_id_node,
        critical=True
    )
    port_claim = f"The identified cruise port '{cruise.port_name or ''}' is one of Port Canaveral, Miami (PortMiami), Fort Lauderdale (Port Everglades), or Tampa."
    await evaluator.verify(
        claim=port_claim,
        node=port_leaf,
        additional_instruction="Allow common naming variants (e.g., 'PortMiami', 'Port Everglades', 'Port Tampa Bay')."
    )
    evaluator.add_custom_node(
        result=bool(cruise.rc_urls),
        id="Reference_URL_Cruise_Port_Exist",
        desc="Provides at least one reference URL for Royal Caribbean operations at the port",
        parent=port_id_node,
        critical=True
    )
    rc_leaf = evaluator.add_leaf(
        id="Royal_Caribbean_Operates_Verified",
        desc="Royal Caribbean operates from the identified port",
        parent=port_id_node,
        critical=True
    )
    rc_claim = f"Royal Caribbean operates cruises from {cruise.port_name or 'the identified port'}."
    await evaluator.verify(
        claim=rc_claim,
        node=rc_leaf,
        sources=cruise.rc_urls,
        additional_instruction="Prefer Royal Caribbean's official port pages or port authority cruise line listings."
    )

    # Port Distance from Orlando (critical, sequential)
    dist_node = evaluator.add_sequential(
        id="Port_Distance_from_Orlando",
        desc="Provides accurate distance or travel time from Orlando area to the identified cruise port",
        parent=cruise_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(cruise.distance_urls),
        id="Reference_URL_Port_Proximity_Exist",
        desc="Provides at least one reference URL for port distance/proximity information",
        parent=dist_node,
        critical=True
    )
    dist_leaf = evaluator.add_leaf(
        id="Port_Proximity_Correct",
        desc="Distance or travel time from Orlando area to the port is approximately as stated",
        parent=dist_node,
        critical=True
    )
    dist_claim = f"The distance or travel time from the Orlando area to {cruise.port_name or 'the port'} is approximately {cruise.distance_or_time_text or ''}."
    await evaluator.verify(
        claim=dist_claim,
        node=dist_leaf,
        sources=cruise.distance_urls,
        additional_instruction="Accept reasonable approximations; verify the order of magnitude (e.g., ~60 miles or ~45–60 minutes to Port Canaveral) using official or reputable travel information sources."
    )


async def verify_travel_docs(evaluator: Evaluator, parent_node, docs: TravelDocsInfo) -> None:
    # Travel Documentation Requirements (critical, parallel)
    docs_node = evaluator.add_parallel(
        id="Travel_Documentation_Requirements",
        desc="Complete information about domestic air travel identification requirements for February 2026",
        parent=parent_node,
        critical=True
    )

    # REAL ID Mandatory Status (critical, sequential)
    real_node = evaluator.add_sequential(
        id="REAL_ID_Mandatory_Status",
        desc="Correctly states REAL ID became mandatory for domestic air travel on May 7, 2025",
        parent=docs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(docs.real_id_urls),
        id="Reference_URL_REAL_ID_Exist",
        desc="Provides at least one reference URL for REAL ID implementation date",
        parent=real_node,
        critical=True
    )
    real_leaf = evaluator.add_leaf(
        id="REAL_ID_Date_Correct",
        desc="REAL ID mandatory date statement is correct",
        parent=real_node,
        critical=True
    )
    real_claim = f"The statement is that REAL ID became mandatory for domestic air travel on May 7, 2025."
    await evaluator.verify(
        claim=real_claim,
        node=real_leaf,
        sources=docs.real_id_urls,
        additional_instruction="Prefer DHS or TSA official announcements confirming the May 7, 2025 enforcement date."
    )

    # TSA ConfirmID Fee Information (critical, sequential)
    ci_node = evaluator.add_sequential(
        id="TSA_ConfirmID_Fee_Information",
        desc="Correctly states TSA charges $45 fee for TSA ConfirmID starting February 1, 2026 (10-day validity)",
        parent=docs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(docs.confirmid_urls),
        id="Reference_URL_ConfirmID_Fee_Exist",
        desc="Provides at least one reference URL for TSA ConfirmID fee",
        parent=ci_node,
        critical=True
    )
    ci_leaf = evaluator.add_leaf(
        id="ConfirmID_Fee_Correct",
        desc="TSA ConfirmID fee and validity statement is correct",
        parent=ci_node,
        critical=True
    )
    ci_claim = "TSA ConfirmID costs $45 starting February 1, 2026, and is valid for 10 days."
    await evaluator.verify(
        claim=ci_claim,
        node=ci_leaf,
        sources=docs.confirmid_urls,
        additional_instruction="Prefer TSA official page describing ConfirmID fees and validity; accept reputable airline/TSA partner pages explicitly stating these details."
    )

    # Digital ID Acceptance Status (critical, sequential)
    did_node = evaluator.add_sequential(
        id="Digital_ID_Acceptance_Status",
        desc="Confirms Apple Digital ID is accepted at 250+ U.S. airports as of November 2025",
        parent=docs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(docs.digital_id_urls),
        id="Reference_URL_Digital_ID_Exist",
        desc="Provides at least one reference URL for Digital ID acceptance",
        parent=did_node,
        critical=True
    )
    did_leaf = evaluator.add_leaf(
        id="Digital_ID_Acceptance_Correct",
        desc="Apple Digital ID acceptance statement is correct",
        parent=did_node,
        critical=True
    )
    did_claim = "Apple Digital ID is accepted at 250+ U.S. airports as of November 2025."
    await evaluator.verify(
        claim=did_claim,
        node=did_leaf,
        sources=docs.digital_id_urls,
        additional_instruction="Prefer TSA or Apple official announcements/summaries listing acceptance at 250+ airports around November 2025."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the travel plan answer for BNA → Florida trip in Feb 2026.

    Returns a structured summary with verification tree and final score.
    """
    # Initialize evaluator (root non-critical to allow flexible aggregation and avoid critical consistency issues)
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

    # 1) Extract structured info from the answer (can be done in parallel)
    bna_task = evaluator.extract(
        prompt=prompt_extract_bna_services(),
        template_class=BNAAirportServices,
        extraction_name="bna_airport_services"
    )
    airline_task = evaluator.extract(
        prompt=prompt_extract_airline_info(),
        template_class=AirlineInfo,
        extraction_name="airline_baggage_info"
    )
    resort_task = evaluator.extract(
        prompt=prompt_extract_resort_info(),
        template_class=ResortInfo,
        extraction_name="disney_resort_info"
    )
    cruise_task = evaluator.extract(
        prompt=prompt_extract_cruise_port_info(),
        template_class=CruisePortInfo,
        extraction_name="cruise_port_info"
    )
    docs_task = evaluator.extract(
        prompt=prompt_extract_travel_docs_info(),
        template_class=TravelDocsInfo,
        extraction_name="travel_docs_info"
    )

    bna, airline, resort, cruise, docs = await asyncio.gather(
        bna_task, airline_task, resort_task, cruise_task, docs_task
    )

    # 2) Build verification tree and run checks
    await verify_airport_services(evaluator, root, bna)
    await verify_airline_baggage(evaluator, root, airline)
    await verify_resort(evaluator, root, resort)
    await verify_cruise_port(evaluator, root, cruise)
    await verify_travel_docs(evaluator, root, docs)

    # 3) Return summary
    return evaluator.get_summary()