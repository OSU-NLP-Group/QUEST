import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "fl_gulf_coast_rv_full_hookup"
TASK_DESCRIPTION = (
    "A non-Florida resident RV owner is planning a camping trip along Florida's Gulf Coast and requires a state park "
    "with extensive RV facilities and full hookups. Based on publicly available Florida State Parks information as of "
    "November 2025, identify the state park that meets ALL of the following criteria: Located on Florida's Gulf Coast; "
    "Has at least 150 RV campsites; Provides full hookups (water, sewer, and electric service) at RV sites; Can "
    "accommodate RVs up to 45 feet in length. Once you identify this park, provide the following information for trip "
    "planning purposes: 1) The name of the state park, 2) Confirmation of its Gulf Coast location, 3) The exact number "
    "of RV campsites available, 4) Detailed verification of full hookup availability including water hookup availability, "
    "sewer hookup availability, and electric service specifications (amp options), 5) The maximum RV length that can be "
    "accommodated, 6) The advance booking window available for non-Florida residents, 7) The reservation fee amount "
    "charged per reservation, and 8) The nightly utility fee charged for RV sites."
)


# =========================
# Data Models (Extraction)
# =========================
class HookupInfo(BaseModel):
    water: Optional[str] = None  # Use strings to maximize extraction robustness
    sewer: Optional[str] = None
    electric: Optional[str] = None
    electric_amp_options: List[str] = Field(default_factory=list)


class ParkInfo(BaseModel):
    park_name: Optional[str] = None
    gulf_coast_location_confirmation: Optional[str] = None
    rv_campsite_count: Optional[str] = None
    max_rv_length: Optional[str] = None
    hookups: HookupInfo = Field(default_factory=HookupInfo)


class ReservationInfo(BaseModel):
    non_resident_advance_booking_window: Optional[str] = None
    reservation_fee_amount: Optional[str] = None
    nightly_utility_fee_amount: Optional[str] = None


class SourceBundle(BaseModel):
    park_urls: List[str] = Field(default_factory=list)     # Official park/camping pages related to the identified park
    policy_urls: List[str] = Field(default_factory=list)   # Statewide reservation/fees policy pages
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs cited in the answer


class ParkTripInfo(BaseModel):
    park: ParkInfo = Field(default_factory=ParkInfo)
    reservation: ReservationInfo = Field(default_factory=ReservationInfo)
    sources: SourceBundle = Field(default_factory=SourceBundle)


# =========================
# Extraction Prompt
# =========================
def prompt_extract_park_trip_info() -> str:
    return """
    Extract structured information about the identified Florida State Park and the requested RV/camping details from the answer.

    Required structured fields:

    park:
      - park_name: The official name of the Florida State Park identified.
      - gulf_coast_location_confirmation: Any text from the answer directly confirming the park is on Florida's Gulf Coast (e.g., "on the Gulf Coast", "located along the Gulf of Mexico in Florida"). If not explicitly stated, return null.
      - rv_campsite_count: The exact number of RV campsites claimed in the answer (e.g., "156", "about 156", "approximately 156"). Extract the text as-is; do not convert to number.
      - max_rv_length: The maximum RV length accommodated claimed in the answer (e.g., "45 feet", "45 ft"). Extract the text as-is.
      - hookups:
          - water: Any text in the answer asserting water hookups are available at RV sites (e.g., "water hookups", "potable water at sites"). If not present, return null.
          - sewer: Any text in the answer asserting sewer hookups are available at RV sites (e.g., "sewer hookups"). If not present, return null.
          - electric: Any text in the answer asserting electric hookups are available at RV sites (e.g., "electric hookups"). If not present, return null.
          - electric_amp_options: List all amp/service options stated in the answer for electric service at RV sites (e.g., ["50 amp", "30 amp"]). If not mentioned, return an empty list.

    reservation:
      - non_resident_advance_booking_window: The stated advance booking window for non-Florida residents (e.g., "10 months", "11 months", "up to 11 months"). Extract the text as-is.
      - reservation_fee_amount: The per-reservation reservation fee amount (e.g., "$6.70"). Extract the text as-is.
      - nightly_utility_fee_amount: The nightly utility fee for RV sites (e.g., "$7.00"). Extract the text as-is.

    sources:
      - park_urls: All URLs provided in the answer that point to official Florida State Parks pages for the identified park (general park page, camping page, etc.). Extract only URLs present in the answer text; include full URLs with protocol. If none are present, return an empty list.
      - policy_urls: All URLs provided in the answer that point to official Florida State Parks policy pages relevant to reservations and fees (e.g., reservation window, utility fees, reservation fee). Extract only URLs present; include full URLs with protocol. If none, return an empty list.
      - additional_urls: Any other URLs cited in the answer text. Extract only URLs present; include full URLs with protocol.

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent or infer any values.
    - If a required field is not present in the answer, set it to null (or empty list for arrays).
    - For URLs, accept plain URLs, markdown links, or embedded links; always return the actual URL string.
    """


# =========================
# Verification Helpers
# =========================
def _get_sources_for_park(info: ParkTripInfo) -> List[str]:
    urls = []
    urls.extend(info.sources.park_urls or [])
    # If park_urls are missing, also consider additional_urls for park claims
    urls.extend(info.sources.additional_urls or [])
    return urls


def _get_sources_for_policy(info: ParkTripInfo) -> List[str]:
    urls = []
    urls.extend(info.sources.policy_urls or [])
    # If policy_urls missing, include additional_urls as fallback
    urls.extend(info.sources.additional_urls or [])
    return urls


# =========================
# Verification Subtrees
# =========================
async def build_identify_qualifying_park(
    evaluator: Evaluator,
    parent_node,
    extracted: ParkTripInfo,
) -> None:
    """
    Build the 'Identify_Qualifying_Park' subtree with critical checks:
      - Park name provided
      - Gulf Coast location confirmed
      - RV campsite count: provided, exact number verification, and minimum threshold (>=150)
      - Full hookups: water, sewer, electric (with amp options stated)
      - Max RV length: provided and at least 45 ft
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Qualifying_Park",
        desc="A specific Florida State Park is identified and it satisfies all required eligibility criteria.",
        parent=parent_node,
        critical=True,
    )

    # Park name provided (existence)
    park_name_exists = bool(extracted.park.park_name and extracted.park.park_name.strip())
    evaluator.add_custom_node(
        result=park_name_exists,
        id="Park_Name_Provided",
        desc="The answer provides the name of the identified Florida State Park.",
        parent=identify_node,
        critical=True,
    )

    # Gulf Coast location confirmed (verify claim using park sources)
    gulf_node = evaluator.add_leaf(
        id="Gulf_Coast_Location_Confirmed",
        desc="The answer confirms the park is located on Florida's Gulf Coast.",
        parent=identify_node,
        critical=True,
    )
    gulf_claim = (
        f"The state park '{extracted.park.park_name or ''}' is located on Florida's Gulf Coast "
        f"(i.e., along the Gulf of Mexico coastline within Florida)."
    )
    await evaluator.verify(
        claim=gulf_claim,
        node=gulf_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction=(
            "Verify the park's location from the official Florida State Parks page(s). "
            "Confirmation should explicitly indicate Gulf Coast or Gulf of Mexico location in Florida."
        ),
    )

    # RV campsite checks grouped as sequential (ensure provided -> exact -> threshold)
    rv_checks = evaluator.add_sequential(
        id="RV_Campsite_Count_Checks",
        desc="RV campsite count is provided, exact number is verified, and it meets the 150 minimum.",
        parent=identify_node,
        critical=True,
    )

    # Provided
    rv_count_provided = bool(extracted.park.rv_campsite_count and extracted.park.rv_campsite_count.strip())
    evaluator.add_custom_node(
        result=rv_count_provided,
        id="RV_Campsite_Count_Provided",
        desc="The answer provides the number of RV campsites.",
        parent=rv_checks,
        critical=True,
    )

    # Exact number verification
    rv_exact_node = evaluator.add_leaf(
        id="RV_Campsite_Count_Exact_Verified",
        desc="The exact number of RV campsites is correct per official sources.",
        parent=rv_checks,
        critical=True,
    )
    rv_exact_claim = (
        f"The state park '{extracted.park.park_name or ''}' has exactly "
        f"{extracted.park.rv_campsite_count or ''} RV campsites."
    )
    await evaluator.verify(
        claim=rv_exact_claim,
        node=rv_exact_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction=(
            "Verify the stated exact number of RV campsites on the official park camping information page. "
            "Allow minor textual variants such as 'approximately' if the page uses that wording."
        ),
    )

    # Minimum threshold (>=150)
    rv_min_node = evaluator.add_leaf(
        id="RV_Campsite_Count_Meets_Minimum",
        desc="The answer provides the number of RV campsites and it is at least 150.",
        parent=rv_checks,
        critical=True,
    )
    rv_min_claim = (
        f"The state park '{extracted.park.park_name or ''}' provides at least 150 RV campsites."
    )
    await evaluator.verify(
        claim=rv_min_claim,
        node=rv_min_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction=(
            "Confirm that the official source(s) support having at least 150 RV campsites."
        ),
    )

    # Full hookups (parallel: water, sewer, electric with amps)
    hookups_node = evaluator.add_parallel(
        id="Full_Hookups_Provided",
        desc="The answer verifies that RV sites provide full hookups (water, sewer, and electric).",
        parent=identify_node,
        critical=True,
    )

    # Water
    water_node = evaluator.add_leaf(
        id="Water_Hookup_Confirmed",
        desc="Water hookup availability at RV sites is verified.",
        parent=hookups_node,
        critical=True,
    )
    water_claim = (
        f"Water hookups are available at RV campsites in '{extracted.park.park_name or ''}'."
    )
    await evaluator.verify(
        claim=water_claim,
        node=water_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction="Verify the RV site amenities include water hookups at the identified park.",
    )

    # Sewer
    sewer_node = evaluator.add_leaf(
        id="Sewer_Hookup_Confirmed",
        desc="Sewer hookup availability at RV sites is verified.",
        parent=hookups_node,
        critical=True,
    )
    sewer_claim = (
        f"Sewer hookups are available at RV campsites in '{extracted.park.park_name or ''}'."
    )
    await evaluator.verify(
        claim=sewer_claim,
        node=sewer_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction="Verify the RV site amenities include sewer hookups at the identified park.",
    )

    # Electric with amp options stated
    electric_node = evaluator.add_leaf(
        id="Electric_Service_And_Amp_Options_Stated",
        desc="Electric hookup availability is verified and the amp/service options are stated.",
        parent=hookups_node,
        critical=True,
    )
    amps_str = ", ".join(extracted.park.hookups.electric_amp_options) if extracted.park.hookups.electric_amp_options else ""
    electric_claim = (
        f"Electric hookups are available at RV campsites in '{extracted.park.park_name or ''}', "
        f"and the available amp/service options include: {amps_str}."
    )
    await evaluator.verify(
        claim=electric_claim,
        node=electric_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction=(
            "Verify that electric hookups exist at RV sites and confirm the amp options (e.g., 30 amp, 50 amp) "
            "as stated on the official park page. No specific amp values are required by the question, but the "
            "answer must state the options."
        ),
    )

    # Max RV length checks (sequential: provided -> at least 45 ft)
    max_len_checks = evaluator.add_sequential(
        id="Max_RV_Length_Checks",
        desc="Maximum RV length is provided and verified to be at least 45 feet.",
        parent=identify_node,
        critical=True,
    )

    # Provided
    max_len_provided = bool(extracted.park.max_rv_length and extracted.park.max_rv_length.strip())
    evaluator.add_custom_node(
        result=max_len_provided,
        id="Max_RV_Length_Provided",
        desc="The maximum RV length that can be accommodated is stated.",
        parent=max_len_checks,
        critical=True,
    )

    # At least 45 ft
    max_len_node = evaluator.add_leaf(
        id="Max_RV_Length_At_Least_45ft",
        desc="The maximum RV length that can be accommodated is stated and is at least 45 feet.",
        parent=max_len_checks,
        critical=True,
    )
    max_len_claim = (
        f"The maximum RV length accommodated at '{extracted.park.park_name or ''}' is at least 45 feet."
    )
    await evaluator.verify(
        claim=max_len_claim,
        node=max_len_node,
        sources=_get_sources_for_park(extracted),
        additional_instruction="Verify the maximum RV length from the official park camping page.",
    )


async def build_trip_planning_policies_and_fees(
    evaluator: Evaluator,
    parent_node,
    extracted: ParkTripInfo,
) -> None:
    """
    Build the 'Provide_Trip_Planning_Policies_And_Fees' subtree with critical checks:
      - Non-resident advance booking window
      - Reservation fee amount
      - Nightly utility fee amount
    """
    policies_node = evaluator.add_parallel(
        id="Provide_Trip_Planning_Policies_And_Fees",
        desc="The answer provides the requested reservation policy/fee details for non-Florida residents and RV utilities (as of Nov 2025).",
        parent=parent_node,
        critical=True,
    )

    # Non-resident advance booking window
    non_res_leaf = evaluator.add_leaf(
        id="Non_Resident_Advance_Booking_Window",
        desc="The advance booking window for non-Florida residents is stated and is correct per Florida State Parks public information (as of Nov 2025).",
        parent=policies_node,
        critical=True,
    )
    non_res_claim = (
        f"As of November 2025, non-Florida residents may reserve campsites up to "
        f"{extracted.reservation.non_resident_advance_booking_window or ''} in advance within Florida State Parks."
    )
    await evaluator.verify(
        claim=non_res_claim,
        node=non_res_leaf,
        sources=_get_sources_for_policy(extracted),
        additional_instruction=(
            "Verify the statewide reservation policy (advance window specifically for non-Florida residents) "
            "from official Florida State Parks reservation/policy pages."
        ),
    )

    # Reservation fee amount per reservation
    reservation_fee_leaf = evaluator.add_leaf(
        id="Reservation_Fee_Amount",
        desc="The per-reservation reservation fee amount is stated and is correct per Florida State Parks public information (as of Nov 2025).",
        parent=policies_node,
        critical=True,
    )
    reservation_fee_claim = (
        f"The per-reservation reservation fee charged by Florida State Parks is "
        f"{extracted.reservation.reservation_fee_amount or ''}."
    )
    await evaluator.verify(
        claim=reservation_fee_claim,
        node=reservation_fee_leaf,
        sources=_get_sources_for_policy(extracted),
        additional_instruction=(
            "Verify the reservation fee (per reservation, not per night) from official fees/policies pages."
        ),
    )

    # Nightly utility fee amount for RV sites
    utility_fee_leaf = evaluator.add_leaf(
        id="Nightly_Utility_Fee_Amount",
        desc="The nightly utility fee charged for RV sites is stated and is correct per Florida State Parks public information (as of Nov 2025).",
        parent=policies_node,
        critical=True,
    )
    utility_fee_claim = (
        f"The nightly utility fee charged for RV sites in Florida State Parks is "
        f"{extracted.reservation.nightly_utility_fee_amount or ''} per night."
    )
    await evaluator.verify(
        claim=utility_fee_claim,
        node=utility_fee_leaf,
        sources=_get_sources_for_policy(extracted),
        additional_instruction=(
            "Verify the nightly utility fee for RV sites (utilities surcharge per night) from official fees/policies pages."
        ),
    )


# =========================
# Main Evaluation Function
# =========================
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Entry point for evaluating the agent's answer for the Florida Gulf Coast RV full hookups task.
    Builds a hierarchical verification tree and returns a structured summary including the final score.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall sequence: identify qualifying park first, then policies/fees
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_park_trip_info(),
        template_class=ParkTripInfo,
        extraction_name="park_trip_info",
    )

    # Create a critical wrapper node to match rubric root requirement
    complete_info_node = evaluator.add_sequential(
        id="Complete_Florida_State_Park_Information",
        desc="Identify a Florida State Park that satisfies the stated RV constraints and provide the requested trip-planning details (based on public Florida State Parks information as of Nov 2025).",
        parent=root,
        critical=True,
    )

    # 1) Identify qualifying park and RV facility details
    await build_identify_qualifying_park(
        evaluator=evaluator,
        parent_node=complete_info_node,
        extracted=extracted_info,
    )

    # 2) Provide trip planning policies and fees
    await build_trip_planning_policies_and_fees(
        evaluator=evaluator,
        parent_node=complete_info_node,
        extracted=extracted_info,
    )

    # Return summary with verification tree and scores
    return evaluator.get_summary()