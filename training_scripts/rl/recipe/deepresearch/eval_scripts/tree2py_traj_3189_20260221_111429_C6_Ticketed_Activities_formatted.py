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
TASK_ID = "la_outdoor_amphitheater_capacity"
TASK_DESCRIPTION = """
Identify the outdoor amphitheater in Los Angeles County, California, that has the largest seating capacity among venues with at least 5,000 seats. For this venue, provide the following information:

1. Venue name and official seating capacity
2. Verification that it is an outdoor amphitheater (not an indoor arena or enclosed theater) located in Los Angeles County
3. Confirmation that this venue has the largest capacity among Los Angeles County outdoor amphitheaters with 5,000+ seats
4. Seating configuration details, including:
   - Evidence of multiple seating types (e.g., reserved seating and general admission/lawn seating)
   - Description of different seating sections or areas
5. ADA accessibility information, including:
   - Confirmation that accessible seating is available
   - Description of how accessible tickets can be purchased
6. Weather policy for outdoor events at this venue
7. Bag policy for venue entry (including size restrictions or clear bag requirements)
8. Camera/photography policy for attendees

All information must be verifiable through official venue sources or reputable third-party documentation, with reference URLs provided for each major category of information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured extraction of key venue details and source URLs from the agent's answer."""
    venue_name: Optional[str] = None
    official_capacity: Optional[str] = None

    # Source URLs per category (must be explicitly present in the answer)
    type_location_urls: List[str] = Field(default_factory=list)      # Venue type & location evidence
    capacity_urls: List[str] = Field(default_factory=list)           # Official/reputable capacity evidence
    largest_capacity_urls: List[str] = Field(default_factory=list)   # Sources supporting "largest capacity" claim
    seating_urls: List[str] = Field(default_factory=list)            # Seating types & sections evidence
    accessibility_urls: List[str] = Field(default_factory=list)      # ADA accessibility info evidence
    weather_urls: List[str] = Field(default_factory=list)            # Weather policy evidence
    bag_urls: List[str] = Field(default_factory=list)                # Bag policy evidence
    camera_urls: List[str] = Field(default_factory=list)             # Camera/photography policy evidence


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the required venue information and the explicit URL sources cited in the answer. Return the following fields:

    1. venue_name: The full name of the identified outdoor amphitheater in Los Angeles County.
    2. official_capacity: The official seating capacity value or description as stated in the answer (keep as text; do not convert to number).

    For each of the following categories, extract all explicit URLs mentioned in the answer that support the information. Only include actual URLs (plain or in markdown). If a category has no URLs in the answer, return an empty list for that category.

    3. type_location_urls: URLs that confirm the venue is an outdoor amphitheater and located in Los Angeles County, California.
    4. capacity_urls: URLs that verify the official capacity figure (prefer official venue pages or reputable documentation).
    5. largest_capacity_urls: URLs used to support the claim that this venue has the largest capacity among LA County outdoor amphitheaters with 5,000+ seats.
    6. seating_urls: URLs that describe seating types (e.g., reserved and lawn/general admission) and seating sections (orchestra, terrace, boxes, etc.).
    7. accessibility_urls: URLs that provide ADA accessibility information (accessible seating, wheelchair locations, companion seating, ticket purchase process).
    8. weather_urls: URLs stating the venue's weather policy for outdoor events.
    9. bag_urls: URLs stating the venue entry bag policy (size restrictions, clear bag requirements, prohibited items).
    10. camera_urls: URLs stating the camera/photography policy (personal vs. professional cameras, video recording).

    Rules:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - If any field is missing in the answer, return null (for strings) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _venue_label(ex: VenueExtraction) -> str:
    """Human-readable venue label used in claims."""
    return ex.venue_name or "the identified venue"

def _has_sources(urls: List[str]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_type_and_location(evaluator: Evaluator, parent_node, ex: VenueExtraction) -> None:
    """
    Build and verify the 'Venue_Type_and_Location' subtree.
    JSON intended this subtree as critical; we keep it critical to gate subsequent checks.
    """
    vt_node = evaluator.add_parallel(
        id="Venue_Type_and_Location",
        desc="Verify the venue is an outdoor amphitheater in Los Angeles County",
        parent=parent_node,
        critical=True
    )

    # Outdoor Amphitheater Verification (critical)
    amph_node = evaluator.add_parallel(
        id="Outdoor_Amphitheater_Verification",
        desc="Confirm the venue is an outdoor amphitheater (not an indoor arena or enclosed theater)",
        parent=vt_node,
        critical=True
    )

    # Open-Air configuration
    leaf_open_air = evaluator.add_leaf(
        id="Open_Air_Configuration",
        desc="Venue has open-air seating exposed to outdoor elements",
        parent=amph_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} has open-air seating exposed to outdoor elements (i.e., it is an outdoor amphitheater).",
        node=leaf_open_air,
        sources=ex.type_location_urls,
        additional_instruction="Look for explicit descriptions such as 'outdoor amphitheater', photos of uncovered seating, or phrases like 'under the stars'."
    )

    # Not an indoor arena
    leaf_not_indoor = evaluator.add_leaf(
        id="Not_Indoor_Arena",
        desc="Venue is not an enclosed indoor arena",
        parent=amph_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} is not an enclosed indoor arena; it is categorized as an outdoor amphitheater.",
        node=leaf_not_indoor,
        sources=ex.type_location_urls,
        additional_instruction="Confirm the venue's classification; explicit 'amphitheater' and outdoor characteristics should differentiate from indoor arenas."
    )

    # Not an enclosed theater
    leaf_not_theater = evaluator.add_leaf(
        id="Not_Enclosed_Theater",
        desc="Venue is not a fully enclosed theater building",
        parent=amph_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} is not a fully enclosed theater building and operates as an outdoor amphitheater.",
        node=leaf_not_theater,
        sources=ex.type_location_urls,
        additional_instruction="Use authoritative descriptions; outdoor seating and amphitheater designation indicates it's not an enclosed theater."
    )

    # LA County Location (critical)
    la_node = evaluator.add_parallel(
        id="Los_Angeles_County_Location",
        desc="Verify venue is located within Los Angeles County, California",
        parent=vt_node,
        critical=True
    )

    leaf_within_la = evaluator.add_leaf(
        id="Within_LA_County",
        desc="Venue address confirms Los Angeles County location",
        parent=la_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} is located within Los Angeles County, California.",
        node=leaf_within_la,
        sources=ex.type_location_urls,
        additional_instruction="Confirm via address or location details; the municipality should be within Los Angeles County."
    )

    leaf_in_california = evaluator.add_leaf(
        id="California_State",
        desc="Venue is located in California",
        parent=la_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} is located in the state of California.",
        node=leaf_in_california,
        sources=ex.type_location_urls,
        additional_instruction="Verify the state shown in the official address or description."
    )

    # Source existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(ex.type_location_urls),
        id="Venue_Type_Location_Source",
        desc="Provide URL reference confirming venue type and location",
        parent=vt_node,
        critical=True
    )


async def verify_capacity_requirements(evaluator: Evaluator, parent_node, ex: VenueExtraction) -> None:
    """
    Build and verify the 'Capacity_Requirements' subtree.
    JSON intended it critical sequential; we keep critical and sequential to gate 'largest capacity' after minimum threshold.
    """
    cap_node = evaluator.add_sequential(
        id="Capacity_Requirements",
        desc="Verify capacity meets minimum threshold and is the largest among qualifying venues",
        parent=parent_node,
        critical=True
    )

    # Minimum Capacity Threshold (critical, parallel)
    min_node = evaluator.add_parallel(
        id="Minimum_Capacity_Threshold",
        desc="Venue has minimum seating capacity of 5,000 people",
        parent=cap_node,
        critical=True
    )

    cap_text = ex.official_capacity or "at least 5,000"

    leaf_at_least_5000 = evaluator.add_leaf(
        id="Capacity_At_Least_5000",
        desc="Official stated capacity is 5,000 or greater",
        parent=min_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official seating capacity of {_venue_label(ex)} is {cap_text}, which is at least 5,000 people.",
        node=leaf_at_least_5000,
        sources=ex.capacity_urls,
        additional_instruction="Confirm a numeric capacity ≥5,000 or a textual description explicitly indicating ≥5,000 from official or reputable sources."
    )

    leaf_official_cap = evaluator.add_leaf(
        id="Official_Capacity_Figure",
        desc="Capacity figure is from official venue source or reputable documentation",
        parent=min_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated capacity for {_venue_label(ex)} is provided by an official venue source or reputable documentation.",
        node=leaf_official_cap,
        sources=ex.capacity_urls,
        additional_instruction="Consider official venue websites, ticketing partners (e.g., Ticketmaster/Live Nation) or widely recognized references (e.g., Wikipedia) as reputable."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.capacity_urls),
        id="Minimum_Capacity_Source",
        desc="Provide URL reference for capacity verification",
        parent=min_node,
        critical=True
    )

    # Largest Capacity Verification (critical, parallel) – evaluated only if min threshold passes
    largest_node = evaluator.add_parallel(
        id="Largest_Capacity_Verification",
        desc="Verify this venue has the largest capacity among LA County outdoor amphitheaters with capacity ≥5,000",
        parent=cap_node,
        critical=True
    )

    leaf_comparison_done = evaluator.add_leaf(
        id="Capacity_Comparison_Conducted",
        desc="Comparison made against other qualifying LA County outdoor amphitheaters",
        parent=largest_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources include capacity information for multiple Los Angeles County outdoor amphitheaters (≥5,000), enabling comparison.",
        node=leaf_comparison_done,
        sources=ex.largest_capacity_urls,
        additional_instruction="Look for sources that list multiple venues or explicitly compare capacities across LA County outdoor amphitheaters."
    )

    leaf_confirmed_largest = evaluator.add_leaf(
        id="Confirmed_Largest_Capacity",
        desc="Venue confirmed as having the largest capacity in the comparison group",
        parent=largest_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Among Los Angeles County outdoor amphitheaters with capacities of at least 5,000, {_venue_label(ex)} has the largest official capacity.",
        node=leaf_confirmed_largest,
        sources=ex.largest_capacity_urls,
        additional_instruction="Use authoritative or reputable lists/comparisons to confirm that this venue's capacity exceeds other LA County outdoor amphitheaters (≥5,000)."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.largest_capacity_urls),
        id="Largest_Capacity_Source",
        desc="Provide URL reference supporting largest capacity claim",
        parent=largest_node,
        critical=True
    )


async def verify_seating_configuration(evaluator: Evaluator, parent_node, ex: VenueExtraction) -> None:
    """
    Build and verify the 'Seating_Configuration' subtree.
    JSON marked parent as critical but included non-critical children; to satisfy framework constraints and allow partial credit,
    we set the parent to non-critical and retain critical checks for core evidence (reserved + GA/lawn).
    """
    seat_node = evaluator.add_parallel(
        id="Seating_Configuration",
        desc="Verify venue offers multiple seating types and configurations",
        parent=parent_node,
        critical=False
    )

    # Multiple Seating Types – keep critical checks inside for reserved + GA/lawn
    multi_types_node = evaluator.add_parallel(
        id="Multiple_Seating_Types",
        desc="Venue provides multiple distinct seating types or options",
        parent=seat_node,
        critical=False  # Parent non-critical to allow soft credit; children can be critical.
    )

    leaf_reserved = evaluator.add_leaf(
        id="Reserved_Seating_Available",
        desc="Venue offers reserved seating with assigned seats",
        parent=multi_types_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} offers reserved seating with assigned seats.",
        node=leaf_reserved,
        sources=ex.seating_urls,
        additional_instruction="Confirm explicit mentions like 'reserved seating', seat numbers, or assigned seats."
    )

    leaf_ga = evaluator.add_leaf(
        id="General_Admission_Available",
        desc="Venue offers general admission or lawn seating option",
        parent=multi_types_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} offers general admission or lawn seating.",
        node=leaf_ga,
        sources=ex.seating_urls,
        additional_instruction="Confirm mentions like 'general admission', 'GA', or 'lawn seating'."
    )

    leaf_types_documented = evaluator.add_leaf(
        id="Seating_Types_Documented",
        desc="Different seating types are clearly documented and described",
        parent=multi_types_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's seating types are clearly documented by official or reputable sources.",
        node=leaf_types_documented,
        sources=ex.seating_urls,
        additional_instruction="Look for explicit descriptions of seating categories and how they differ."
    )

    # Seating sections or areas (non-critical subitems)
    sections_node = evaluator.add_parallel(
        id="Seating_Sections_Identified",
        desc="Specific seating sections or areas are identified",
        parent=seat_node,
        critical=False
    )

    leaf_orchestra = evaluator.add_leaf(
        id="Lower_Orchestra_Section",
        desc="Venue has lower level or orchestra seating section",
        parent=sections_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} has a lower-level or orchestra seating section.",
        node=leaf_orchestra,
        sources=ex.seating_urls,
        additional_instruction="Look for seating maps or section names mentioning 'Orchestra', 'Lower Level', or equivalent."
    )

    leaf_upper_lawn = evaluator.add_leaf(
        id="Upper_Terrace_or_Lawn",
        desc="Venue has upper terrace, balcony, or lawn seating area",
        parent=sections_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} includes an upper terrace/balcony or lawn seating area.",
        node=leaf_upper_lawn,
        sources=ex.seating_urls,
        additional_instruction="Seating maps or descriptions should mention 'Terrace', 'Balcony', or 'Lawn'."
    )

    leaf_box_premium = evaluator.add_leaf(
        id="Box_or_Premium_Seating",
        desc="Venue offers box seats or premium seating options",
        parent=sections_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The venue {_venue_label(ex)} offers box seats or premium seating options.",
        node=leaf_box_premium,
        sources=ex.seating_urls,
        additional_instruction="Look for 'box seating', 'premium seats', 'VIP boxes', or similar language."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.seating_urls),
        id="Seating_Configuration_Source",
        desc="Provide URL reference for seating configuration details",
        parent=seat_node,
        critical=True
    )


async def verify_accessibility_compliance(evaluator: Evaluator, parent_node, ex: VenueExtraction) -> None:
    """
    Build and verify the 'Accessibility_Compliance' subtree.
    JSON marked parent as critical but included non-critical children; we set the parent to non-critical to allow partial credit.
    """
    acc_node = evaluator.add_parallel(
        id="Accessibility_Compliance",
        desc="Verify venue meets ADA accessibility requirements",
        parent=parent_node,
        critical=False
    )

    # ADA Accessible Seating – we keep children critical to emphasize core compliance evidence
    ada_node = evaluator.add_parallel(
        id="ADA_Accessible_Seating",
        desc="Venue provides ADA-compliant accessible seating",
        parent=acc_node,
        critical=True
    )

    leaf_accessible_designated = evaluator.add_leaf(
        id="Accessible_Seating_Designated",
        desc="Accessible seating areas are designated and available",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Accessible seating areas are designated and available at {_venue_label(ex)}.",
        node=leaf_accessible_designated,
        sources=ex.accessibility_urls,
        additional_instruction="Look for explicit language regarding ADA seating sections, designated areas, or similar."
    )

    leaf_wheelchair_locations = evaluator.add_leaf(
        id="Wheelchair_Accessible_Locations",
        desc="Wheelchair-accessible seating locations are provided",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wheelchair-accessible seating locations are provided at {_venue_label(ex)}.",
        node=leaf_wheelchair_locations,
        sources=ex.accessibility_urls,
        additional_instruction="Confirm mentions of wheelchair locations or ADA seating map indicators."
    )

    leaf_companion = evaluator.add_leaf(
        id="Companion_Seating",
        desc="Companion seating is available adjacent to accessible seats",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Companion seating is available adjacent to accessible seats at {_venue_label(ex)}.",
        node=leaf_companion,
        sources=ex.accessibility_urls,
        additional_instruction="Check for language indicating companion seats accompanying accessible seating."
    )

    # Accessible Ticket Purchase – emphasize critical purchase process evidence
    purchase_node = evaluator.add_parallel(
        id="Accessible_Ticket_Purchase",
        desc="Accessible seating tickets can be purchased through standard channels",
        parent=acc_node,
        critical=True
    )

    leaf_standard_channel = evaluator.add_leaf(
        id="Standard_Channel_Access",
        desc="Accessible tickets available via standard ticketing platforms (online/box office)",
        parent=purchase_node,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible seating tickets can be purchased via standard ticketing platforms (online, phone, or at the box office).",
        node=leaf_standard_channel,
        sources=ex.accessibility_urls,
        additional_instruction="Look for instructions that indicate accessible tickets can be bought through regular channels, not exclusively via special processes."
    )

    leaf_no_docs = evaluator.add_leaf(
        id="No_Special_Documentation_Required",
        desc="No disability documentation required to purchase accessible seating",
        parent=purchase_node,
        critical=True
    )
    await evaluator.verify(
        claim="No disability documentation is required to purchase accessible seating at this venue.",
        node=leaf_no_docs,
        sources=ex.accessibility_urls,
        additional_instruction="Confirm whether the policy states accessible tickets are available without requiring documentation. If documentation is required, this claim should fail."
    )

    leaf_same_price = evaluator.add_leaf(
        id="Same_Price_Policy",
        desc="Accessible seats priced comparably to regular seats in same section",
        parent=purchase_node,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible seats are priced comparably to regular seats in the same section.",
        node=leaf_same_price,
        sources=ex.accessibility_urls,
        additional_instruction="Look for explicit pricing policy statements; if unclear or absent, the claim should not pass."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.accessibility_urls),
        id="Accessibility_Source",
        desc="Provide URL reference for accessibility information",
        parent=acc_node,
        critical=True
    )


async def verify_operational_policies(evaluator: Evaluator, parent_node, ex: VenueExtraction) -> None:
    """
    Build and verify the 'Operational_Policies' subtree.
    JSON marked parent and some children as critical but mixed non-critical items; we set the parent non-critical to allow partial credit.
    """
    ops_node = evaluator.add_parallel(
        id="Operational_Policies",
        desc="Verify venue has publicly stated operational policies",
        parent=parent_node,
        critical=False
    )

    # Weather Policy (non-critical parent with mix of critical/non-critical leaves)
    weather_node = evaluator.add_parallel(
        id="Weather_Policy",
        desc="Venue has publicly stated weather policy for outdoor events",
        parent=ops_node,
        critical=False
    )

    leaf_rain_or_shine = evaluator.add_leaf(
        id="Rain_or_Shine_Policy",
        desc="Venue specifies rain or shine policy or weather contingency",
        parent=weather_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue specifies a 'rain or shine' policy or weather contingency for outdoor events.",
        node=leaf_rain_or_shine,
        sources=ex.weather_urls,
        additional_instruction="Look for explicit statements about events proceeding in rain or weather contingency procedures."
    )

    leaf_severe_weather = evaluator.add_leaf(
        id="Severe_Weather_Procedures",
        desc="Venue mentions procedures for severe weather conditions",
        parent=weather_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue mentions procedures for severe weather conditions (e.g., delays, shelter-in-place, or cancellations).",
        node=leaf_severe_weather,
        sources=ex.weather_urls,
        additional_instruction="Search for severe weather guidance or procedures if available."
    )

    leaf_refund_terms = evaluator.add_leaf(
        id="Refund_Cancellation_Terms",
        desc="Weather-related refund or cancellation terms are provided",
        parent=weather_node,
        critical=False
    )
    await evaluator.verify(
        claim="Weather-related refund or cancellation terms are provided by the venue.",
        node=leaf_refund_terms,
        sources=ex.weather_urls,
        additional_instruction="Look for refund/cancellation policy text related to weather."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.weather_urls),
        id="Weather_Policy_Source",
        desc="Provide URL reference for weather policy",
        parent=weather_node,
        critical=True
    )

    # Bag Policy
    bag_node = evaluator.add_parallel(
        id="Bag_Policy",
        desc="Venue has publicly stated bag policy for entry",
        parent=ops_node,
        critical=False
    )

    leaf_bag_size = evaluator.add_leaf(
        id="Bag_Size_Restrictions",
        desc="Bag size restrictions are specified",
        parent=bag_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's bag policy specifies size restrictions.",
        node=leaf_bag_size,
        sources=ex.bag_urls,
        additional_instruction="Look for maximum dimensions or specific size limits."
    )

    leaf_clear_bag = evaluator.add_leaf(
        id="Clear_Bag_Requirements",
        desc="Clear bag requirements (if any) are specified",
        parent=bag_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's bag policy specifies clear bag requirements (if applicable).",
        node=leaf_clear_bag,
        sources=ex.bag_urls,
        additional_instruction="Confirm mention of clear bag policy; if no clear bag policy, the claim should fail."
    )

    leaf_prohibited_items = evaluator.add_leaf(
        id="Prohibited_Items_Listed",
        desc="List of prohibited bag items is provided",
        parent=bag_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue provides a list of prohibited bag items.",
        node=leaf_prohibited_items,
        sources=ex.bag_urls,
        additional_instruction="Look for a prohibited items list within the bag/security policy page."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.bag_urls),
        id="Bag_Policy_Source",
        desc="Provide URL reference for bag policy",
        parent=bag_node,
        critical=True
    )

    # Camera Policy
    camera_node = evaluator.add_parallel(
        id="Camera_Policy",
        desc="Venue has publicly stated camera/photography policy",
        parent=ops_node,
        critical=False
    )

    leaf_personal_camera = evaluator.add_leaf(
        id="Personal_Camera_Allowance",
        desc="Policy specifies whether personal cameras/phones are allowed",
        parent=camera_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's policy specifies whether personal cameras or phones are allowed.",
        node=leaf_personal_camera,
        sources=ex.camera_urls,
        additional_instruction="Look for an explicit statement on personal photography devices."
    )

    leaf_professional_camera = evaluator.add_leaf(
        id="Professional_Camera_Restrictions",
        desc="Policy specifies restrictions on professional cameras with detachable lenses",
        parent=camera_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's policy specifies restrictions on professional cameras with detachable lenses.",
        node=leaf_professional_camera,
        sources=ex.camera_urls,
        additional_instruction="Look for language restricting professional gear, detachable lenses, tripods, etc."
    )

    leaf_video_recording = evaluator.add_leaf(
        id="Video_Recording_Policy",
        desc="Policy addresses video recording permissions or restrictions",
        parent=camera_node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's policy addresses permissions or restrictions for video recording.",
        node=leaf_video_recording,
        sources=ex.camera_urls,
        additional_instruction="Confirm whether video recording is permitted, restricted, or prohibited."
    )

    evaluator.add_custom_node(
        result=_has_sources(ex.camera_urls),
        id="Camera_Policy_Source",
        desc="Provide URL reference for camera policy",
        parent=camera_node,
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
    Evaluate an answer for the Los Angeles outdoor amphitheater capacity task.
    Note: To satisfy framework constraints (critical parent cannot have non-critical children),
    we set the root to non-critical sequential, while keeping essential subtrees critical.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential gating as the task logically depends on prior checks
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

    # Extract structured info from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree according to rubric (with adjusted criticality where necessary)
    # 1) Venue Type & Location (critical)
    await verify_type_and_location(evaluator, root, extracted)

    # 2) Capacity Requirements (critical, sequential)
    await verify_capacity_requirements(evaluator, root, extracted)

    # 3) Seating Configuration (non-critical with critical subchecks)
    await verify_seating_configuration(evaluator, root, extracted)

    # 4) Accessibility Compliance (non-critical with critical subchecks)
    await verify_accessibility_compliance(evaluator, root, extracted)

    # 5) Operational Policies (non-critical groups)
    await verify_operational_policies(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()