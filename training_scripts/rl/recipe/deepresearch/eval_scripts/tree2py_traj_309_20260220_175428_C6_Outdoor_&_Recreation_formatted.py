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
TASK_ID = "green_ridge_trip_plan"
TASK_DESCRIPTION = (
    "A group of outdoor enthusiasts is planning a 2-night primitive camping trip at Green Ridge State Forest in Maryland "
    "for early March 2026. They want to understand all the essential requirements and logistics before their trip. Provide "
    "a comprehensive camping plan that includes: (1) the complete registration process and location where they must register "
    "before camping; (2) the exact nightly camping fee for primitive campsites; (3) a detailed description of what amenities "
    "are provided at each primitive campsite and what amenities are NOT available; (4) information about at least one specific "
    "named hiking trail in the forest, including its distance and difficulty rating; (5) the rules regarding firewood acquisition "
    "and use; and (6) the headquarters phone number for any questions. All information must be supported by reference URLs from "
    "official Maryland DNR sources or reputable outdoor recreation websites."
)

# Optional, helpful ground truths commonly expected for GRSF (for summary only; verification relies on sources)
GROUND_TRUTH_HINTS = {
    "expected_nightly_fee_text": "$10 per night",
    "expected_phone": "301-478-3124",
    "expected_included_amenities": ["picnic table", "fire ring"],
    "expected_trail_overview": "over 80 miles of trails"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RegistrationInfo(BaseModel):
    location_text: Optional[str] = None
    pre_occupancy_text: Optional[str] = None
    office_hours_text: Optional[str] = None
    kiosk_available_text: Optional[str] = None
    kiosk_payment_options_text: Optional[str] = None
    backpack_itinerary_text: Optional[str] = None
    backpack_hq_reg_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FeeInfo(BaseModel):
    nightly_fee_text: Optional[str] = None
    applies_to_primitive_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AmenitiesInfo(BaseModel):
    picnic_table_text: Optional[str] = None
    fire_ring_text: Optional[str] = None
    excluded_amenities_text: Optional[str] = None
    number_of_sites_text: Optional[str] = None
    year_round_availability_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TrailInfo(BaseModel):
    trail_name: Optional[str] = None
    trail_distance_text: Optional[str] = None
    trail_difficulty_text: Optional[str] = None
    trail_elevation_gain_text: Optional[str] = None
    trail_duration_text: Optional[str] = None
    trail_overview_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FirewoodInfo(BaseModel):
    sources_text: Optional[str] = None
    personal_prohibition_text: Optional[str] = None
    rationale_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ContactInfo(BaseModel):
    phone: Optional[str] = None
    address: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    registration: Optional[RegistrationInfo] = None
    fees: Optional[FeeInfo] = None
    amenities: Optional[AmenitiesInfo] = None
    trail: Optional[TrailInfo] = None
    firewood: Optional[FirewoodInfo] = None
    contact: Optional[ContactInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract from the answer all the specific information required for a primitive camping plan at Green Ridge State Forest (GRSF).
    You must only extract exactly what the answer text states, without inventing or inferring any missing details.

    Organize your extraction in the following structured JSON fields:

    registration:
      - location_text: Where registration must occur (e.g., "Green Ridge State Forest Headquarters")
      - pre_occupancy_text: Statement that registration is required before occupying the campsite
      - office_hours_text: The office hours text as stated (e.g., "7am–3pm daily")
      - kiosk_available_text: Statement that a self-registration kiosk is available when the office is closed
      - kiosk_payment_options_text: The kiosk payment options text (e.g., "cash or check only")
      - backpack_itinerary_text: If backpack camping is mentioned, the statement about submitting an itinerary with all camper names
      - backpack_hq_reg_text: If backpack camping is mentioned, statement that backpack camping also requires headquarters registration
      - sources: All URLs cited in the answer that support registration requirements

    fees:
      - nightly_fee_text: The exact nightly fee text for primitive campsites (e.g., "$10 per night")
      - applies_to_primitive_text: Statement clarifying the fee applies to primitive campsites
      - sources: All URLs cited that support fee information

    amenities:
      - picnic_table_text: Statement that each primitive campsite includes a picnic table
      - fire_ring_text: Statement that each primitive campsite includes a fire ring
      - excluded_amenities_text: Statement that there is no plumbing or other amenities (primitive camping)
      - number_of_sites_text: Statement of how many designated primitive campsites there are (e.g., "100 designated primitive campsites")
      - year_round_availability_text: Statement that sites are available year-round
      - sources: All URLs cited that support amenities information

    trail:
      - trail_name: The name of at least one specific trail in GRSF
      - trail_distance_text: The trail distance (as text, e.g., "5.3 miles")
      - trail_difficulty_text: The trail difficulty rating (e.g., "moderate")
      - trail_elevation_gain_text: Elevation gain if provided (as text)
      - trail_duration_text: Estimated time if provided (as text)
      - trail_overview_text: Statement that GRSF has over 80 miles of trails for various activities if provided
      - sources: All URLs cited that support trail information (official DNR or reputable outdoor sites like AllTrails)

    firewood:
      - sources_text: Statement that firewood must be purchased locally or gathered on-site
      - personal_prohibition_text: Statement that bringing personal/outside firewood is prohibited
      - rationale_text: Reason for the prohibition (e.g., preventing forest pest spread)
      - sources: All URLs cited that support firewood policy

    contact:
      - phone: The headquarters phone number (e.g., "301-478-3124")
      - address: The headquarters physical address, if provided
      - sources: All URLs cited that provide contact information

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only actual URLs appearing in the answer (plain or markdown). Do not invent URLs.
    - If a URL is missing a protocol, prepend "http://".
    - If no sources are provided for a section, return an empty list for that section's 'sources'.

    If any field is not mentioned in the answer, set it to null (for strings) or [] (for sources lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _source_list(sources: Optional[List[str]]) -> List[str]:
    return sources if sources else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_registration(evaluator: Evaluator, parent_node, info: RegistrationInfo) -> None:
    reg_node = evaluator.add_parallel(
        id="Registration_and_Permit_Requirements",
        desc="Correct identification of registration location, process, and requirements",
        parent=parent_node,
        critical=False
    )

    # Registration Location
    if _non_empty_text(info.location_text):
        node = evaluator.add_leaf(
            id="Registration_Location",
            desc="Identifies that registration must occur at Green Ridge State Forest Headquarters",
            parent=reg_node,
            critical=True
        )
        claim = "Primitive campsite registration must occur at the Green Ridge State Forest Headquarters."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the page states the registration location is the GRSF Headquarters. Allow minor wording variations."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Registration_Location",
            desc="Identifies that registration must occur at Green Ridge State Forest Headquarters",
            parent=reg_node,
            critical=True
        )

    # Pre-Occupancy Requirement
    if _non_empty_text(info.pre_occupancy_text):
        node = evaluator.add_leaf(
            id="Pre_Occupancy_Requirement",
            desc="States that registration is required before occupying the campsite",
            parent=reg_node,
            critical=True
        )
        claim = "Registration is required before occupying a campsite at Green Ridge State Forest."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify that campers must register prior to occupying or using a primitive campsite."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Pre_Occupancy_Requirement",
            desc="States that registration is required before occupying the campsite",
            parent=reg_node,
            critical=True
        )

    # Registration Methods (office vs. kiosk)
    methods_node = evaluator.add_parallel(
        id="Registration_Methods",
        desc="Describes available registration methods (office hours vs. self-registration kiosk)",
        parent=reg_node,
        critical=False
    )

    # Office hours method
    if _non_empty_text(info.office_hours_text):
        node = evaluator.add_leaf(
            id="Office_Hours_Method",
            desc="Mentions registration during office hours (7am-3pm daily)",
            parent=methods_node,
            critical=False
        )
        claim = f"Campers can register during office hours at headquarters (the answer states: {info.office_hours_text})."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the source indicates registration is available during posted office hours; do not require an exact hour match if the answer's wording is close."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Office_Hours_Method",
            desc="Mentions registration during office hours (7am-3pm daily)",
            parent=methods_node,
            critical=False
        )

    # Kiosk method
    if _non_empty_text(info.kiosk_available_text):
        node = evaluator.add_leaf(
            id="Kiosk_Method",
            desc="Mentions self-registration kiosk when office is closed",
            parent=methods_node,
            critical=False
        )
        claim = "A self-registration kiosk is available for campers when the office is closed."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Check that the source indicates a self-registration kiosk is available outside office hours."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Kiosk_Method",
            desc="Mentions self-registration kiosk when office is closed",
            parent=methods_node,
            critical=False
        )

    # Kiosk payment options
    if _non_empty_text(info.kiosk_payment_options_text):
        node = evaluator.add_leaf(
            id="Kiosk_Payment_Options",
            desc="States that kiosk accepts only cash or check",
            parent=methods_node,
            critical=False
        )
        claim = "The self-registration kiosk accepts only cash or check (cards are not accepted)."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify that kiosk payment is limited to cash or check; phrasing such as 'exact cash or check' should pass."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Kiosk_Payment_Options",
            desc="States that kiosk accepts only cash or check",
            parent=methods_node,
            critical=False
        )

    # Backpack camping additional requirements
    backpack_node = evaluator.add_parallel(
        id="Backpack_Camping_Additional_Requirements",
        desc="If backpack camping is mentioned, identifies additional requirements",
        parent=reg_node,
        critical=False
    )

    if _non_empty_text(info.backpack_itinerary_text):
        node = evaluator.add_leaf(
            id="Itinerary_Submission",
            desc="States that itinerary with all camper names must be submitted for backpack camping",
            parent=backpack_node,
            critical=False
        )
        claim = "Backpack camping requires submitting an itinerary that includes the names of all campers."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the source mentions an itinerary submission with camper names for backpack camping."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Itinerary_Submission",
            desc="States that itinerary with all camper names must be submitted for backpack camping",
            parent=backpack_node,
            critical=False
        )

    if _non_empty_text(info.backpack_hq_reg_text):
        node = evaluator.add_leaf(
            id="Headquarters_Registration",
            desc="States that backpack camping also requires headquarters registration",
            parent=backpack_node,
            critical=False
        )
        claim = "Backpack camping also requires registration at Green Ridge State Forest Headquarters."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify the source indicates backpack camping registration must occur at headquarters."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Headquarters_Registration",
            desc="States that backpack camping also requires headquarters registration",
            parent=backpack_node,
            critical=False
        )

    # Registration Reference URL (presence)
    evaluator.add_custom_node(
        result=len(_source_list(info.sources)) > 0,
        id="Registration_Reference_URL",
        desc="Provides valid reference URL for registration requirements",
        parent=reg_node,
        critical=True
    )


async def verify_fees(evaluator: Evaluator, parent_node, info: FeeInfo) -> None:
    fees_node = evaluator.add_parallel(
        id="Camping_Fees",
        desc="Correct identification of camping fees and payment requirements",
        parent=parent_node,
        critical=False
    )

    if _non_empty_text(info.nightly_fee_text):
        node = evaluator.add_leaf(
            id="Nightly_Fee_Amount",
            desc="States that the camping fee is $10 per night",
            parent=fees_node,
            critical=True
        )
        claim = f"The primitive campsite fee at Green Ridge State Forest is {info.nightly_fee_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the page states the nightly fee amount for primitive camping; allow $10 per night equivalently formatted."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Nightly_Fee_Amount",
            desc="States that the camping fee is $10 per night",
            parent=fees_node,
            critical=True
        )

    if _non_empty_text(info.applies_to_primitive_text):
        node = evaluator.add_leaf(
            id="Fee_Applies_To_Primitive_Sites",
            desc="Clarifies that the $10/night fee applies to primitive campsites",
            parent=fees_node,
            critical=True
        )
        claim = "The stated nightly fee applies specifically to primitive campsites."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the fee applies to primitive campsites (not developed/modern sites)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Fee_Applies_To_Primitive_Sites",
            desc="Clarifies that the $10/night fee applies to primitive campsites",
            parent=fees_node,
            critical=True
        )

    evaluator.add_custom_node(
        result=len(_source_list(info.sources)) > 0,
        id="Fee_Reference_URL",
        desc="Provides valid reference URL for fee information",
        parent=fees_node,
        critical=True
    )


async def verify_amenities(evaluator: Evaluator, parent_node, info: AmenitiesInfo) -> None:
    am_node = evaluator.add_parallel(
        id="Campsite_Amenities",
        desc="Accurate description of what amenities are and are not provided at primitive campsites",
        parent=parent_node,
        critical=False
    )

    included_node = evaluator.add_parallel(
        id="Included_Amenities",
        desc="Correctly identifies that campsites include a picnic table and fire ring",
        parent=am_node,
        critical=False
    )

    # Picnic table
    if _non_empty_text(info.picnic_table_text):
        node = evaluator.add_leaf(
            id="Picnic_Table",
            desc="Mentions picnic table is provided",
            parent=included_node,
            critical=True
        )
        claim = "Each designated primitive campsite includes a picnic table."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the amenities list includes a picnic table for primitive campsites."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Picnic_Table",
            desc="Mentions picnic table is provided",
            parent=included_node,
            critical=True
        )

    # Fire ring
    if _non_empty_text(info.fire_ring_text):
        node = evaluator.add_leaf(
            id="Fire_Ring",
            desc="Mentions fire ring is provided",
            parent=included_node,
            critical=True
        )
        claim = "Each designated primitive campsite includes a fire ring."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the amenities list includes a fire ring for primitive campsites."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Fire_Ring",
            desc="Mentions fire ring is provided",
            parent=included_node,
            critical=True
        )

    # Excluded amenities
    if _non_empty_text(info.excluded_amenities_text):
        node = evaluator.add_leaf(
            id="Excluded_Amenities",
            desc="States that no plumbing or other amenities are provided (primitive camping)",
            parent=am_node,
            critical=True
        )
        claim = "Primitive campsites have no plumbing or other developed amenities (e.g., no hookups)."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify the page indicates the sites are primitive with no plumbing/utilities or other modern amenities."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Excluded_Amenities",
            desc="States that no plumbing or other amenities are provided (primitive camping)",
            parent=am_node,
            critical=True
        )

    # Number of sites (non-critical)
    if _non_empty_text(info.number_of_sites_text):
        node = evaluator.add_leaf(
            id="Number_of_Sites",
            desc="Mentions that 100 designated primitive campsites are available",
            parent=am_node,
            critical=False
        )
        claim = f"Green Ridge State Forest has {info.number_of_sites_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the page states there are approximately 100 designated primitive campsites; allow equivalent phrasing."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Number_of_Sites",
            desc="Mentions that 100 designated primitive campsites are available",
            parent=am_node,
            critical=False
        )

    # Year-round availability (non-critical)
    if _non_empty_text(info.year_round_availability_text):
        node = evaluator.add_leaf(
            id="Year_Round_Availability",
            desc="States that sites are available year-round",
            parent=am_node,
            critical=False
        )
        claim = "Primitive campsites at Green Ridge State Forest are available year-round."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify that availability is year-round; seasonal closures wording should fail."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Year_Round_Availability",
            desc="States that sites are available year-round",
            parent=am_node,
            critical=False
        )

    evaluator.add_custom_node(
        result=len(_source_list(info.sources)) > 0,
        id="Amenities_Reference_URL",
        desc="Provides valid reference URL for campsite amenities information",
        parent=am_node,
        critical=True
    )


async def verify_trail(evaluator: Evaluator, parent_node, info: TrailInfo) -> None:
    trail_node = evaluator.add_parallel(
        id="Trail_Information",
        desc="Provides accurate information about at least one specific trail suitable for hiking",
        parent=parent_node,
        critical=False
    )

    # Trail name
    if _non_empty_text(info.trail_name):
        node = evaluator.add_leaf(
            id="Trail_Name",
            desc="Identifies a specific named trail within Green Ridge State Forest",
            parent=trail_node,
            critical=True
        )
        claim = f"The trail named '{info.trail_name}' is located within Green Ridge State Forest."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the trail is within GRSF; allow minor naming variants or abbreviations."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Trail_Name",
            desc="Identifies a specific named trail within Green Ridge State Forest",
            parent=trail_node,
            critical=True
        )

    # Trail characteristics
    char_node = evaluator.add_parallel(
        id="Trail_Characteristics",
        desc="Provides accurate trail characteristics (distance, difficulty, elevation gain, or time)",
        parent=trail_node,
        critical=False
    )

    if _non_empty_text(info.trail_distance_text):
        node = evaluator.add_leaf(
            id="Trail_Distance",
            desc="Provides the trail distance",
            parent=char_node,
            critical=True
        )
        claim = f"The trail distance is {info.trail_distance_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify the listed distance; allow rounding and minor format variations (e.g., miles vs mi)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Trail_Distance",
            desc="Provides the trail distance",
            parent=char_node,
            critical=True
        )

    if _non_empty_text(info.trail_difficulty_text):
        node = evaluator.add_leaf(
            id="Trail_Difficulty",
            desc="Provides the trail difficulty rating",
            parent=char_node,
            critical=True
        )
        claim = f"The trail difficulty rating is {info.trail_difficulty_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify difficulty rating (e.g., easy, moderate, hard); allow similar wording."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Trail_Difficulty",
            desc="Provides the trail difficulty rating",
            parent=char_node,
            critical=True
        )

    # Elevation gain (non-critical)
    if _non_empty_text(info.trail_elevation_gain_text):
        node = evaluator.add_leaf(
            id="Trail_Elevation_Gain",
            desc="Provides the trail elevation gain (if applicable)",
            parent=char_node,
            critical=False
        )
        claim = f"The trail elevation gain is {info.trail_elevation_gain_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify elevation gain; allow approximate or rounded values."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Trail_Elevation_Gain",
            desc="Provides the trail elevation gain (if applicable)",
            parent=char_node,
            critical=False
        )

    # Duration (non-critical)
    if _non_empty_text(info.trail_duration_text):
        node = evaluator.add_leaf(
            id="Trail_Duration",
            desc="Provides estimated time to complete the trail (if applicable)",
            parent=char_node,
            critical=False
        )
        claim = f"The estimated time to complete the trail is {info.trail_duration_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify duration/time estimate; allow typical pacing assumptions."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Trail_Duration",
            desc="Provides estimated time to complete the trail (if applicable)",
            parent=char_node,
            critical=False
        )

    # Trail system overview (non-critical)
    if _non_empty_text(info.trail_overview_text):
        node = evaluator.add_leaf(
            id="Trail_System_Overview",
            desc="Mentions that Green Ridge has over 80 miles of trails for various activities",
            parent=trail_node,
            critical=False
        )
        claim = "Green Ridge State Forest has over 80 miles of trails for various activities."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the page states ~80+ miles of trails; allow approximate language like 'over 80 miles'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Trail_System_Overview",
            desc="Mentions that Green Ridge has over 80 miles of trails for various activities",
            parent=trail_node,
            critical=False
        )

    evaluator.add_custom_node(
        result=len(_source_list(info.sources)) > 0,
        id="Trail_Reference_URL",
        desc="Provides valid reference URL for trail information",
        parent=trail_node,
        critical=True
    )


async def verify_firewood(evaluator: Evaluator, parent_node, info: FirewoodInfo) -> None:
    fw_node = evaluator.add_parallel(
        id="Firewood_Policy",
        desc="Correctly describes firewood acquisition rules and restrictions",
        parent=parent_node,
        critical=False
    )

    if _non_empty_text(info.sources_text):
        node = evaluator.add_leaf(
            id="Firewood_Sources",
            desc="States that firewood must be purchased locally or gathered on-site",
            parent=fw_node,
            critical=True
        )
        claim = "Firewood must be purchased locally or gathered on-site within Green Ridge State Forest."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm local purchase or on-site gathering is required; allow equivalent phrases like 'buy locally'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Firewood_Sources",
            desc="States that firewood must be purchased locally or gathered on-site",
            parent=fw_node,
            critical=True
        )

    if _non_empty_text(info.personal_prohibition_text):
        node = evaluator.add_leaf(
            id="Personal_Firewood_Prohibition",
            desc="States that personal firewood from other locations is prohibited",
            parent=fw_node,
            critical=True
        )
        claim = "Bringing personal/outside firewood from other locations is prohibited at Green Ridge State Forest."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify the prohibition on outside firewood to prevent pest spread."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Personal_Firewood_Prohibition",
            desc="States that personal firewood from other locations is prohibited",
            parent=fw_node,
            critical=True
        )

    if _non_empty_text(info.rationale_text):
        node = evaluator.add_leaf(
            id="Rationale_For_Prohibition",
            desc="Explains the reason for prohibition (preventing forest pest spread)",
            parent=fw_node,
            critical=False
        )
        claim = "The prohibition exists to prevent the spread of forest pests and diseases."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Confirm the rationale mentions preventing spread of invasive pests or diseases."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Rationale_For_Prohibition",
            desc="Explains the reason for prohibition (preventing forest pest spread)",
            parent=fw_node,
            critical=False
        )

    evaluator.add_custom_node(
        result=len(_source_list(info.sources)) > 0,
        id="Firewood_Reference_URL",
        desc="Provides valid reference URL for firewood policy",
        parent=fw_node,
        critical=True
    )


async def verify_contact(evaluator: Evaluator, parent_node, info: ContactInfo) -> None:
    contact_node = evaluator.add_parallel(
        id="Contact_Information",
        desc="Provides accurate headquarters contact information",
        parent=parent_node,
        critical=False
    )

    if _non_empty_text(info.phone):
        node = evaluator.add_leaf(
            id="Headquarters_Phone",
            desc="Provides the headquarters phone number (301-478-3124)",
            parent=contact_node,
            critical=True
        )
        claim = f"The Green Ridge State Forest Headquarters phone number is {info.phone}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify the headquarters phone number from the source; allow formatting variations like dashes or spaces."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Headquarters_Phone",
            desc="Provides the headquarters phone number (301-478-3124)",
            parent=contact_node,
            critical=True
        )

    if _non_empty_text(info.address):
        node = evaluator.add_leaf(
            id="Headquarters_Address",
            desc="Provides the headquarters physical address",
            parent=contact_node,
            critical=False
        )
        claim = f"The Green Ridge State Forest Headquarters address is {info.address}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=_source_list(info.sources),
            additional_instruction="Verify the headquarters physical address; allow minor formatting variations."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Headquarters_Address",
            desc="Provides the headquarters physical address",
            parent=contact_node,
            critical=False
        )

    evaluator.add_custom_node(
        result=len(_source_list(info.sources)) > 0,
        id="Contact_Reference_URL",
        desc="Provides valid reference URL for contact information",
        parent=contact_node,
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
) -> Dict[str, Any]:
    """
    Evaluate a comprehensive primitive camping plan for Green Ridge State Forest.
    Builds a verification tree and returns a structured summary.
    """
    # Initialize evaluator (root set as non-critical parallel to allow partial credit while gating via child critical leaves)
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

    # Extract structured plan data
    extracted_plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="grsf_plan_extraction"
    )

    # Add helpful ground-truth hints (for summary only)
    evaluator.add_ground_truth(GROUND_TRUTH_HINTS, gt_type="expected_hints")

    # Build top-level parent node (set to non-critical to comply with framework's critical-children constraint)
    plan_node = evaluator.add_parallel(
        id="Green_Ridge_Camping_Trip_Plan",
        desc="Evaluation of a comprehensive primitive camping trip plan for Green Ridge State Forest",
        parent=root,
        critical=False
    )

    # Run section verifications
    await verify_registration(evaluator, plan_node, extracted_plan.registration or RegistrationInfo())
    await verify_fees(evaluator, plan_node, extracted_plan.fees or FeeInfo())
    await verify_amenities(evaluator, plan_node, extracted_plan.amenities or AmenitiesInfo())
    await verify_trail(evaluator, plan_node, extracted_plan.trail or TrailInfo())
    await verify_firewood(evaluator, plan_node, extracted_plan.firewood or FirewoodInfo())
    await verify_contact(evaluator, plan_node, extracted_plan.contact or ContactInfo())

    # Return structured summary
    return evaluator.get_summary()