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
TASK_ID = "fl_outdoor_planning_2026"
TASK_DESCRIPTION = """
You are planning a multi-day outdoor recreation event in Florida for a group of up to 15 participants in Summer 2026. Your event requires three different types of facilities, and you need to identify specific locations that meet all the following requirements:

Facility 1 - Group Campground:
Find a Florida state park that offers group camping facilities with the following characteristics:
- Can accommodate your entire group of 15 people in a single group campsite
- Allows reservations to be made at least 6 months in advance of the arrival date
- Specify the minimum stay requirement for weekend reservations
- Describe the refund policy for cancellations made 30 or more days before arrival

Facility 2 - Day-Use Pavilion:
Find a park pavilion in Florida suitable for a group gathering with these requirements:
- Has seating capacity for at least 40 people
- Specify the rental fee per day
- Identify the deposit amount required at the time of reservation (if applicable)
- Confirm the minimum age requirement for the person making the reservation

Facility 3 - Wilderness Area:
Find a wilderness area or backcountry zone in Florida with these specifications:
- Requires advance permits for overnight camping
- Specify the maximum group size allowed per wilderness campsite type
- Indicate how far in advance wilderness permits can be reserved
- Provide the fee structure for wilderness overnight permits

Insurance Requirements:
Determine whether any of the three facilities you identified require liability insurance for group events. If insurance is required, specify:
- Which facility or facilities require it
- The minimum liability coverage amount (if specified)
- The deadline for submitting the certificate of insurance before the event

For each facility, provide the specific name and location, and include a reference URL from an official source (state park website, recreation.gov, or official government site) that confirms each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GroupCampground(BaseModel):
    park_name: Optional[str] = None
    location: Optional[str] = None
    capacity_single_group_site: Optional[str] = None  # e.g., "up to 25 people"
    booking_window_advance: Optional[str] = None      # e.g., "11 months", "180 days"
    weekend_min_stay: Optional[str] = None            # e.g., "2 nights"
    refund_policy_30_days: Optional[str] = None       # e.g., "Full refund minus fee if 30+ days"
    reference_urls: List[str] = Field(default_factory=list)


class DayUsePavilion(BaseModel):
    park_pavilion_name: Optional[str] = None
    location: Optional[str] = None
    seating_capacity: Optional[str] = None            # e.g., "50", "40-60"
    rental_fee_per_day: Optional[str] = None          # e.g., "$100/day"
    deposit_amount: Optional[str] = None              # e.g., "$100", "No deposit"
    min_reservation_age: Optional[str] = None         # e.g., "18", "21"
    reference_urls: List[str] = Field(default_factory=list)


class WildernessArea(BaseModel):
    area_name: Optional[str] = None
    location: Optional[str] = None
    overnight_permit_required: Optional[str] = None   # "Yes" / "No" or descriptive text
    max_group_size: Optional[str] = None              # e.g., "8 per campsite"
    reservation_window_advance: Optional[str] = None  # e.g., "up to 30 days in advance"
    permit_fee_structure: Optional[str] = None        # e.g., "$X per person per night"
    reference_urls: List[str] = Field(default_factory=list)


class InsuranceInfo(BaseModel):
    required: Optional[bool] = None
    applies_to: List[str] = Field(default_factory=list)  # list of facility names or labels it applies to
    minimum_coverage: Optional[str] = None               # e.g., "$1,000,000"
    submission_deadline: Optional[str] = None            # e.g., "30 days prior"
    reference_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facility1: Optional[GroupCampground] = None
    facility2: Optional[DayUsePavilion] = None
    facility3: Optional[WildernessArea] = None
    insurance: Optional[InsuranceInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract the structured information for three facilities and any insurance requirements as explicitly stated in the answer text.

    Return a JSON object with the following schema and keys. If any field is not present in the answer, set it to null (or an empty array for URL lists). Do NOT invent information. Extract values as strings exactly as written in the answer when possible.

    {
      "facility1": {
        "park_name": string | null,
        "location": string | null,
        "capacity_single_group_site": string | null,
        "booking_window_advance": string | null,
        "weekend_min_stay": string | null,
        "refund_policy_30_days": string | null,
        "reference_urls": string[]  // URLs cited in the answer that substantiate facility #1 info; include all if multiple, and only URLs explicitly present in the answer
      },
      "facility2": {
        "park_pavilion_name": string | null,
        "location": string | null,
        "seating_capacity": string | null,
        "rental_fee_per_day": string | null,
        "deposit_amount": string | null,
        "min_reservation_age": string | null,
        "reference_urls": string[]  // URLs cited in the answer that substantiate facility #2 info
      },
      "facility3": {
        "area_name": string | null,
        "location": string | null,
        "overnight_permit_required": string | null,
        "max_group_size": string | null,
        "reservation_window_advance": string | null,
        "permit_fee_structure": string | null,
        "reference_urls": string[]  // URLs cited in the answer that substantiate facility #3 info
      },
      "insurance": {
        "required": boolean | null,         // true if any identified facility requires liability insurance, false if explicitly stated no insurance required
        "applies_to": string[],             // names/labels of facilities that require insurance as stated in the answer text
        "minimum_coverage": string | null,  // e.g., "$1,000,000" if specified
        "submission_deadline": string | null, // e.g., "30 days prior to event" if specified
        "reference_urls": string[]          // Insurance policy/requirements URLs cited in the answer; can include facility policy pages, park system policies, or official government pages
      }
    }

    URL extraction special rules:
    - Only include URLs explicitly present in the answer text (plain URL or markdown link).
    - Prefer official sources (e.g., floridastateparks.org, recreation.gov, nps.gov, .gov domains, or official county/municipal gov sites) but still extract whatever URLs actually appear in the answer.
    - Do not fabricate URLs. If the answer mentions a source without a URL, omit it here.

    Important:
    - Keep all numeric values as strings (e.g., "11 months", "2 nights", "$100").
    - If deposit is not required and the answer says so, set deposit_amount to something like "No deposit" (as written).
    - For insurance.applies_to, use the facility names as written in the answer when possible (e.g., "Hillsborough River State Park Group Camp", "Large Pavilion at XYZ Park").
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _name_or_placeholder(name: Optional[str], default: str) -> str:
    return name.strip() if name else default


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_facility_1(evaluator: Evaluator, parent_node, f1: Optional[GroupCampground]) -> None:
    node = evaluator.add_parallel(
        id="facility_1_group_campground",
        desc="First facility: A state park with group camping that accommodates 15 people with advance reservations",
        parent=parent_node,
        critical=False
    )

    f1 = f1 or GroupCampground()

    # Critical: Reference URL(s) exist
    evaluator.add_custom_node(
        result=len(_urls(f1.reference_urls)) > 0,
        id="f1_reference_url",
        desc="Official website URL confirming the facility information",
        parent=node,
        critical=True
    )

    # Critical: Park name and that page covers group camping
    leaf = evaluator.add_leaf(
        id="f1_park_name",
        desc="Name of the Florida state park with group camping facilities",
        parent=node,
        critical=True
    )
    park_name = _name_or_placeholder(f1.park_name, "the identified Florida state park")
    await evaluator.verify(
        claim=f"The official page is about {park_name} in Florida and it includes information about group camping (group campground or group camp area).",
        node=leaf,
        sources=_urls(f1.reference_urls),
        additional_instruction="Allow minor name variations or formatting differences. Confirm that the page is an official park or government booking page and mentions group camping."
    )

    # Critical: Capacity supports 15 in a single group site
    leaf = evaluator.add_leaf(
        id="f1_capacity_verification",
        desc="Verification that the facility can accommodate 15 people in a single group site",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The group camping facility at {park_name} can accommodate at least 15 people in a single group campsite or group camp area that is reserved as one unit.",
        node=leaf,
        sources=_urls(f1.reference_urls),
        additional_instruction="Accept if the page states a capacity of 15 or more for a single group campsite/area (not by combining multiple separate individual campsites)."
    )

    # Critical: Booking window at least 6 months
    leaf = evaluator.add_leaf(
        id="f1_booking_window",
        desc="Confirmation that reservations can be made at least 6 months in advance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Reservations for the group campground at {park_name} can be made at least 6 months (180 days) before the arrival date.",
        node=leaf,
        sources=_urls(f1.reference_urls),
        additional_instruction="If the page states a reservation window of X months or Y days before arrival, consider this claim supported if X ≥ 6 months or Y ≥ 180 days."
    )

    # Non-critical: Minimum stay for weekends
    leaf = evaluator.add_leaf(
        id="f1_minimum_stay",
        desc="The minimum number of nights required for weekend reservations",
        parent=node,
        critical=False
    )
    if f1.weekend_min_stay:
        claim_text = f"The official page specifies that the minimum stay for weekend reservations is {f1.weekend_min_stay}."
    else:
        claim_text = "The official page specifies a minimum stay requirement for weekend reservations."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f1.reference_urls),
        additional_instruction="Look for language such as 'minimum stay', '2-night minimum on weekends', or similar policy details."
    )

    # Non-critical: Refund policy for cancellations 30+ days
    leaf = evaluator.add_leaf(
        id="f1_refund_policy_30days",
        desc="Refund policy for cancellations made 30 or more days before arrival",
        parent=node,
        critical=False
    )
    if f1.refund_policy_30_days:
        claim_text = f"For cancellations made 30 or more days before arrival, the refund policy is: {f1.refund_policy_30_days}."
    else:
        claim_text = "The official page describes a refund policy for cancellations made 30 or more days before arrival."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f1.reference_urls),
        additional_instruction="Verify the portion of the policy that applies specifically to cancellations ≥ 30 days before arrival."
    )


async def verify_facility_2(evaluator: Evaluator, parent_node, f2: Optional[DayUsePavilion]) -> None:
    node = evaluator.add_parallel(
        id="facility_2_pavilion",
        desc="Second facility: A park pavilion for day-use events seating at least 40 people",
        parent=parent_node,
        critical=False
    )

    f2 = f2 or DayUsePavilion()

    # Critical: Reference URL(s) exist
    evaluator.add_custom_node(
        result=len(_urls(f2.reference_urls)) > 0,
        id="f2_reference_url",
        desc="Official website URL confirming the pavilion information",
        parent=node,
        critical=True
    )

    # Critical: Pavilion park/name exists on page
    leaf = evaluator.add_leaf(
        id="f2_park_pavilion_name",
        desc="Name of the park and specific pavilion in Florida",
        parent=node,
        critical=True
    )
    pav_name = _name_or_placeholder(f2.park_pavilion_name, "the identified pavilion")
    await evaluator.verify(
        claim=f"The official page is about {pav_name} (or the pavilions at the specified Florida park) and clearly corresponds to the pavilion referenced in the answer.",
        node=leaf,
        sources=_urls(f2.reference_urls),
        additional_instruction="Allow minor naming variations (e.g., 'Large Pavilion' vs 'Main Pavilion') as long as it's clearly the same pavilion/park."
    )

    # Critical: Seating capacity at least 40
    leaf = evaluator.add_leaf(
        id="f2_seating_capacity",
        desc="Verification that the pavilion seats at least 40 people",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The pavilion at {pav_name} provides seating capacity for at least 40 people.",
        node=leaf,
        sources=_urls(f2.reference_urls),
        additional_instruction="Consider 'capacity', 'seating', or 'accommodates' language. Accept if the stated capacity is ≥ 40."
    )

    # Non-critical: Rental fee per day
    leaf = evaluator.add_leaf(
        id="f2_rental_fee",
        desc="Rental fee per day for the pavilion",
        parent=node,
        critical=False
    )
    if f2.rental_fee_per_day:
        claim_text = f"The rental fee per day for the pavilion is {f2.rental_fee_per_day}."
    else:
        claim_text = "The official page provides the daily rental fee for the pavilion."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f2.reference_urls),
        additional_instruction="Look for 'fee', 'rate', or similar pricing information for a single day use."
    )

    # Non-critical: Deposit requirement (if applicable)
    leaf = evaluator.add_leaf(
        id="f2_deposit_requirement",
        desc="Deposit amount required at time of reservation (if applicable)",
        parent=node,
        critical=False
    )
    if f2.deposit_amount:
        claim_text = f"The deposit amount required at the time of reservation is {f2.deposit_amount}."
    else:
        claim_text = "The official page indicates whether a deposit is required at the time of reservation (and, if so, how much)."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f2.reference_urls),
        additional_instruction="Accept either an explicit deposit amount or a clear statement that no deposit is required."
    )

    # Non-critical: Minimum age for reservation
    leaf = evaluator.add_leaf(
        id="f2_age_requirement",
        desc="Minimum age to make a facility reservation",
        parent=node,
        critical=False
    )
    if f2.min_reservation_age:
        claim_text = f"The minimum age to make a reservation for this pavilion/facility is {f2.min_reservation_age}."
    else:
        claim_text = "The official page specifies a minimum age requirement for the person making a reservation."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f2.reference_urls),
        additional_instruction="Look for policy language specifying 'must be 18 or older', '21 or older', or similar requirements."
    )


async def verify_facility_3(evaluator: Evaluator, parent_node, f3: Optional[WildernessArea]) -> None:
    node = evaluator.add_parallel(
        id="facility_3_wilderness",
        desc="Third facility: A wilderness area requiring advance permits for overnight camping",
        parent=parent_node,
        critical=False
    )

    f3 = f3 or WildernessArea()

    # Critical: Reference URL(s) exist
    evaluator.add_custom_node(
        result=len(_urls(f3.reference_urls)) > 0,
        id="f3_reference_url",
        desc="Official website URL confirming the wilderness permit information",
        parent=node,
        critical=True
    )

    # Critical: Area name exists on page
    leaf = evaluator.add_leaf(
        id="f3_area_name",
        desc="Name of the wilderness area or backcountry zone in Florida",
        parent=node,
        critical=True
    )
    area_name = _name_or_placeholder(f3.area_name, "the identified wilderness/backcountry area")
    await evaluator.verify(
        claim=f"The official page is for {area_name} in Florida (wilderness or backcountry area).",
        node=leaf,
        sources=_urls(f3.reference_urls),
        additional_instruction="Allow minor naming variations. Confirm the page clearly corresponds to the same area."
    )

    # Critical: Advance permits are required for overnight stays
    leaf = evaluator.add_leaf(
        id="f3_permit_requirement",
        desc="Confirmation that advance permits are required for overnight stays",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Advance permits are required for overnight camping in {area_name}.",
        node=leaf,
        sources=_urls(f3.reference_urls),
        additional_instruction="Look for language like 'permit required', 'backcountry permit', or 'overnight permit', and that it must be obtained in advance."
    )

    # Critical: Max group size per campsite type
    leaf = evaluator.add_leaf(
        id="f3_max_group_size",
        desc="Maximum group size allowed per wilderness campsite type",
        parent=node,
        critical=True
    )
    if f3.max_group_size:
        claim_text = f"The maximum group size allowed per wilderness campsite (or backcountry site) is {f3.max_group_size}."
    else:
        claim_text = "The official page specifies a maximum group size limit per wilderness/backcountry campsite."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f3.reference_urls),
        additional_instruction="Accept if the page states a numeric or descriptive limit for group size at a single campsite or zone."
    )

    # Critical: Reservation window (how far in advance)
    leaf = evaluator.add_leaf(
        id="f3_reservation_window",
        desc="How far in advance wilderness permits can be reserved",
        parent=node,
        critical=True
    )
    if f3.reservation_window_advance:
        claim_text = f"Wilderness overnight permits for {area_name} can be reserved {f3.reservation_window_advance}."
    else:
        claim_text = "The official page specifies how far in advance wilderness/backcountry permits can be reserved."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f3.reference_urls),
        additional_instruction="Look for reservation opening windows, e.g., 'up to X days/months in advance' or similar."
    )

    # Non-critical: Fee structure for overnight permits
    leaf = evaluator.add_leaf(
        id="f3_permit_fee",
        desc="Fee structure for wilderness overnight permits",
        parent=node,
        critical=False
    )
    if f3.permit_fee_structure:
        claim_text = f"The fee structure for the wilderness overnight permits is: {f3.permit_fee_structure}."
    else:
        claim_text = "The official page describes the fee structure for wilderness/backcountry overnight permits."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(f3.reference_urls),
        additional_instruction="Accept per-person, per-night, per-permit, or similar fee formulations as long as they match the page."
    )


async def verify_insurance(evaluator: Evaluator, parent_node, ins: Optional[InsuranceInfo]) -> None:
    node = evaluator.add_parallel(
        id="policy_insurance",
        desc="Insurance requirements: Determine if liability insurance is required for group events at any identified facility",
        parent=parent_node,
        critical=False
    )

    ins = ins or InsuranceInfo()

    # Critical: Reference URL(s) exist
    evaluator.add_custom_node(
        result=len(_urls(ins.reference_urls)) > 0,
        id="insurance_reference_url",
        desc="Official website URL confirming insurance requirements",
        parent=node,
        critical=True
    )

    # Critical: Identify which facilities require insurance (if any)
    leaf = evaluator.add_leaf(
        id="insurance_required",
        desc="Identification of which facilities require liability insurance for groups",
        parent=node,
        critical=True
    )
    if ins.required is True and ins.applies_to:
        applies = ", ".join(ins.applies_to)
        claim_text = f"Based on the official sources, liability insurance is required for group events for the following identified facilities: {applies}."
    elif ins.required is False:
        claim_text = "Based on the official sources, none of the identified facilities require liability insurance for group events."
    else:
        # Generic claim if answer is ambiguous but URLs are provided
        claim_text = "Based on the official sources, at least one identified facility may require liability insurance for group events, and the answer identifies which facility or facilities."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(ins.reference_urls),
        additional_instruction="Evaluate the official pages for explicit insurance/COI requirements related to group events, pavilion rentals, or group camping. Accept if the pages clearly indicate insurance is required for the named facilities. If the claim states none require insurance, accept only if the sources explicitly support that conclusion."
    )

    # Non-critical: Minimum coverage amount (if specified)
    leaf = evaluator.add_leaf(
        id="insurance_minimum_coverage",
        desc="Minimum liability coverage amount required (if applicable)",
        parent=node,
        critical=False
    )
    if ins.minimum_coverage:
        claim_text = f"The minimum liability coverage amount required is {ins.minimum_coverage}."
    else:
        claim_text = "The official page specifies the minimum liability coverage amount required (if applicable)."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(ins.reference_urls),
        additional_instruction="Check for policy text like '$1,000,000 liability', 'COI minimum coverage', etc."
    )

    # Non-critical: Submission deadline (if specified)
    leaf = evaluator.add_leaf(
        id="insurance_submission_deadline",
        desc="Deadline for submitting certificate of insurance before event",
        parent=node,
        critical=False
    )
    if ins.submission_deadline:
        claim_text = f"The deadline for submitting the certificate of insurance before the event is {ins.submission_deadline}."
    else:
        claim_text = "The official page specifies a deadline for submitting the certificate of insurance before the event."
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_urls(ins.reference_urls),
        additional_instruction="Look for phrasing such as 'submit COI X days before the event' or similar."
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
    Evaluate an answer for the Florida outdoor planning facilities and insurance requirements task.
    """
    # Initialize evaluator (root kept non-critical for compatibility with mixed critical children)
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Build verification subtrees
    await verify_facility_1(evaluator, root, extraction.facility1)
    await verify_facility_2(evaluator, root, extraction.facility2)
    await verify_facility_3(evaluator, root, extraction.facility3)
    await verify_insurance(evaluator, root, extraction.insurance)

    # Return aggregated summary
    return evaluator.get_summary()