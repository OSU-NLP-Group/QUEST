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
TASK_ID = "boston_logan_hotel_requirements"
TASK_DESCRIPTION = """
I am planning a business trip to Boston and need to find a hotel near Boston Logan International Airport that meets the following requirements:
(1) Must be within 3 miles of Boston Logan Airport and provide airport shuttle service,
(2) Must have wheelchair-accessible rooms that meet ADA standards with doorways at least 32 inches wide,
(3) Must allow pets and clearly disclose the pet fee policy,
(4) Must offer on-site parking with disclosed rates,
(5) Must have a cancellation policy that allows free cancellation at least 24 hours before check-in,
(6) Must have an on-site fitness center,
(7) Must provide high-speed WiFi suitable for business use, and
(8) Must have meeting/conference room space available.
Please identify one hotel that meets all these requirements and provide verification for each requirement with supporting reference URLs.
"""

REQUIREMENTS_LIST = [
    "Identify exactly one specific hotel candidate.",
    "Within 3 miles of Boston Logan International Airport.",
    "Airport shuttle service provided.",
    "Wheelchair-accessible rooms meeting ADA standards.",
    "Accessible room doorway width at least 32 inches.",
    "Pets allowed.",
    "Pet fee policy disclosed.",
    "On-site parking available.",
    "Parking rates disclosed.",
    "Free cancellation allowed at least 24 hours before check-in.",
    "On-site fitness center available.",
    "High-speed WiFi suitable for business use.",
    "Meeting/conference room space available."
]

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class HotelCandidate(BaseModel):
    """Chosen hotel candidate extracted from the answer."""
    name: Optional[str] = None
    primary_url: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None


class ConstraintValues(BaseModel):
    """Human-readable values or descriptions extracted from the answer."""
    distance_miles: Optional[str] = None
    shuttle_desc: Optional[str] = None
    ada_rooms_desc: Optional[str] = None
    ada_door_width: Optional[str] = None
    pets_allowed_desc: Optional[str] = None
    pet_fee_policy_desc: Optional[str] = None
    parking_desc: Optional[str] = None
    parking_rates_desc: Optional[str] = None
    cancellation_policy_desc: Optional[str] = None
    fitness_center_desc: Optional[str] = None
    wifi_desc: Optional[str] = None
    meeting_space_desc: Optional[str] = None


class ConstraintURLs(BaseModel):
    """Per-constraint supporting URLs explicitly cited in the answer."""
    distance_sources: List[str] = Field(default_factory=list)
    shuttle_sources: List[str] = Field(default_factory=list)
    ada_room_sources: List[str] = Field(default_factory=list)
    ada_door_sources: List[str] = Field(default_factory=list)
    pet_sources: List[str] = Field(default_factory=list)
    pet_fee_sources: List[str] = Field(default_factory=list)
    parking_sources: List[str] = Field(default_factory=list)
    parking_rate_sources: List[str] = Field(default_factory=list)
    cancellation_sources: List[str] = Field(default_factory=list)
    fitness_sources: List[str] = Field(default_factory=list)
    wifi_sources: List[str] = Field(default_factory=list)
    meeting_sources: List[str] = Field(default_factory=list)


class HotelExtraction(BaseModel):
    """Combined extraction of the candidate hotel, values, and constraint-specific URLs."""
    candidate: Optional[HotelCandidate] = None
    values: ConstraintValues = ConstraintValues()
    sources: ConstraintURLs = ConstraintURLs()
    other_hotels_mentioned: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Your job is to extract the single chosen hotel candidate and all relevant verification URLs from the provided answer text.

    Extract exactly one chosen hotel candidate:
    - candidate.name: The hotel's official name the answer commits to (if multiple mentioned, pick the one the answer clearly selects; if unclear, pick the first one mentioned).
    - candidate.primary_url: The hotel's official website or a main authoritative page for the property (if available in the answer).
    - candidate.address: Street address of the hotel (if mentioned).
    - candidate.city: City of the hotel (if mentioned, likely Boston or nearby).

    Extract human-readable values (as text strings exactly as they appear in the answer; do not invent):
    - values.distance_miles: Distance to Boston Logan (e.g., "2.1 miles", "within 3 miles") if stated.
    - values.shuttle_desc: Shuttle service description if stated.
    - values.ada_rooms_desc: ADA/wheelchair accessible rooms description if stated.
    - values.ada_door_width: Doorway width stated (e.g., "32 inches") if stated.
    - values.pets_allowed_desc: Whether pets are allowed (text).
    - values.pet_fee_policy_desc: Pet fee policy details (amount/terms) if stated.
    - values.parking_desc: On-site parking description if stated.
    - values.parking_rates_desc: Parking rates details if stated.
    - values.cancellation_policy_desc: Cancellation policy statement if stated; include any time windows (e.g., "free cancellation 24 hours before check-in").
    - values.fitness_center_desc: Fitness center description if stated.
    - values.wifi_desc: WiFi description (e.g., "high-speed", "business-grade", "50 Mbps") if stated.
    - values.meeting_space_desc: Meeting/conference space description if stated.

    Extract per-constraint supporting URLs that the answer explicitly cites (do not invent or infer; only return URLs present in the answer):
    - sources.distance_sources: URLs that support distance to Boston Logan or location statements that explicitly mention distance.
    - sources.shuttle_sources: URLs that support airport shuttle service.
    - sources.ada_room_sources: URLs that support ADA/wheelchair-accessible rooms (ADA-compliant).
    - sources.ada_door_sources: URLs that explicitly mention accessible doorway width (inches).
    - sources.pet_sources: URLs that support pets being allowed.
    - sources.pet_fee_sources: URLs that disclose pet fee policy (amount and terms).
    - sources.parking_sources: URLs that support on-site parking availability.
    - sources.parking_rate_sources: URLs that disclose parking rates.
    - sources.cancellation_sources: URLs that describe the cancellation policy.
    - sources.fitness_sources: URLs that support fitness center availability.
    - sources.wifi_sources: URLs that support high-speed/business-grade WiFi or speed details (≥50 Mbps if specified).
    - sources.meeting_sources: URLs that support meeting/conference room availability.

    Additionally, list all hotel names mentioned anywhere in the answer (including the chosen one and any others):
    - other_hotels_mentioned: array of hotel names mentioned.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer (plain URLs or markdown links). Do not invent any URLs.
    - If any value or URL is not present in the answer, return null or an empty array for that field.
    - If more than one hotel is mentioned, your 'candidate' should be the single one the answer commits to or, if unclear, the first mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _sources_or_primary(specific_sources: List[str], primary_url: Optional[str]) -> List[str]:
    """Prefer constraint-specific sources; if none, fall back to primary_url if available."""
    if specific_sources and len(specific_sources) > 0:
        return specific_sources
    return [primary_url] if _non_empty(primary_url) else []


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def _build_identify_one_hotel(
    evaluator: Evaluator,
    parent: Any,
    extraction: HotelExtraction
) -> Any:
    """
    Build the 'Identify_One_Hotel' critical check as a small sequential group:
      - Hotel name provided (custom existence check)
      - Exactly one specific hotel identified (simple verification against the answer)
    """
    node = evaluator.add_sequential(
        id="Identify_One_Hotel",
        desc="Response identifies exactly one specific hotel near Boston Logan International Airport as the candidate.",
        parent=parent,
        critical=True
    )

    # Leaf 1: Hotel name provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty(extraction.candidate.name),
        id="Hotel_Name_Provided",
        desc="A specific hotel name is provided in the answer.",
        parent=node,
        critical=True
    )

    # Leaf 2: Exactly one specific hotel identified (judge the answer content)
    # The LLM judge focuses on whether the answer clearly commits to one candidate.
    exactly_one_leaf = evaluator.add_leaf(
        id="Exactly_One_Hotel_Identified",
        desc="The answer clearly identifies exactly one hotel as the chosen candidate.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly commits to exactly one hotel as the chosen candidate (mentions of other hotels as comparisons are acceptable but the final choice must be singular).",
        node=exactly_one_leaf,
        additional_instruction="Focus on the final commitment: even if multiple properties are discussed, the answer should clearly present a single chosen hotel."
    )

    return node


async def _build_verify_all_constraints(
    evaluator: Evaluator,
    parent: Any,
    extraction: HotelExtraction
) -> Any:
    """
    Build the 'Verify_All_Constraints' critical parallel group with one leaf per constraint.
    Each leaf uses verify() routed to verify_by_urls with provided or fallback sources.
    """
    hotel_name = extraction.candidate.name or "the hotel"

    node = evaluator.add_parallel(
        id="Verify_All_Constraints",
        desc="Candidate hotel satisfies each required constraint.",
        parent=parent,
        critical=True
    )

    claims_and_nodes: List[tuple] = []

    # Within 3 miles of Logan
    within_leaf = evaluator.add_leaf(
        id="Within_3_Miles_of_Logan",
        desc="Hotel is located within 3 miles of Boston Logan International Airport.",
        parent=node,
        critical=True
    )
    within_sources = _sources_or_primary(extraction.sources.distance_sources, extraction.candidate.primary_url)
    within_claim = f"The hotel '{hotel_name}' is located within 3 miles of Boston Logan International Airport (BOS)."
    claims_and_nodes.append((within_claim, within_sources, within_leaf,
                             "Look for explicit mentions like 'X miles to Logan', 'minutes to BOS', 'adjacent to airport'. If the page does not explicitly claim ≤3 miles, mark as not supported."))

    # Airport shuttle service
    shuttle_leaf = evaluator.add_leaf(
        id="Airport_Shuttle_Service",
        desc="Hotel provides airport shuttle service (free or paid).",
        parent=node,
        critical=True
    )
    shuttle_sources = _sources_or_primary(extraction.sources.shuttle_sources, extraction.candidate.primary_url)
    shuttle_claim = f"The hotel '{hotel_name}' provides an airport shuttle service to/from Boston Logan International Airport."
    claims_and_nodes.append((shuttle_claim, shuttle_sources, shuttle_leaf,
                             "Accept synonyms like 'airport shuttle', 'airport transportation', 'courtesy shuttle'. It may be free or paid; either satisfies the requirement."))

    # ADA wheelchair-accessible rooms
    ada_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Rooms_ADA",
        desc="Hotel has wheelchair-accessible rooms meeting ADA standards.",
        parent=node,
        critical=True
    )
    ada_sources = _sources_or_primary(extraction.sources.ada_room_sources, extraction.candidate.primary_url)
    ada_claim = f"The hotel '{hotel_name}' offers wheelchair-accessible rooms that meet ADA standards (ADA-compliant)."
    claims_and_nodes.append((ada_claim, ada_sources, ada_leaf,
                             "Look for 'ADA compliant', 'ADA accessible', or detailed accessibility features explicitly matching ADA requirements."))

    # Accessible doorway width ≥ 32 inches
    door_leaf = evaluator.add_leaf(
        id="Accessible_Doorway_Width_32_Inches",
        desc="Accessible rooms have doorways at least 32 inches wide.",
        parent=node,
        critical=True
    )
    door_sources = _sources_or_primary(extraction.sources.ada_door_sources, extraction.candidate.primary_url)
    door_claim = f"Accessible guestrooms at '{hotel_name}' have doorway widths of at least 32 inches."
    claims_and_nodes.append((door_claim, door_sources, door_leaf,
                             "Look for explicit doorway width information (e.g., 'doorway width (inches): 32'). If width is not explicitly stated, mark as not supported."))

    # Pets allowed
    pets_leaf = evaluator.add_leaf(
        id="Pets_Allowed",
        desc="Hotel allows pets.",
        parent=node,
        critical=True
    )
    pets_sources = _sources_or_primary(extraction.sources.pet_sources, extraction.candidate.primary_url)
    pets_claim = f"The hotel '{hotel_name}' allows pets."
    claims_and_nodes.append((pets_claim, pets_sources, pets_leaf,
                             "Accept any clear policy statement indicating pets are allowed (including service animals by default)."))

    # Pet fee policy disclosed
    pet_fee_leaf = evaluator.add_leaf(
        id="Pet_Fee_Policy_Disclosed",
        desc="Hotel clearly discloses pet fee policy (amount and terms).",
        parent=node,
        critical=True
    )
    pet_fee_sources = _sources_or_primary(extraction.sources.pet_fee_sources, extraction.candidate.primary_url)
    pet_fee_claim = f"The hotel '{hotel_name}' clearly discloses its pet fee policy, including fee amount and terms."
    claims_and_nodes.append((pet_fee_claim, pet_fee_sources, pet_fee_leaf,
                             "The page should include fee amount or specific terms (e.g., per-stay/per-night fee). Vague statements without amounts do not satisfy."))

    # On-site parking available
    parking_leaf = evaluator.add_leaf(
        id="On_Site_Parking_Available",
        desc="Hotel offers on-site parking.",
        parent=node,
        critical=True
    )
    parking_sources = _sources_or_primary(extraction.sources.parking_sources, extraction.candidate.primary_url)
    parking_claim = f"The hotel '{hotel_name}' offers on-site parking."
    claims_and_nodes.append((parking_claim, parking_sources, parking_leaf,
                             "Accept mentions of 'on-site parking', 'self-parking', or 'valet parking' available at the property."))

    # Parking rates disclosed
    parking_rate_leaf = evaluator.add_leaf(
        id="Parking_Rates_Disclosed",
        desc="Hotel discloses parking rates.",
        parent=node,
        critical=True
    )
    parking_rate_sources = _sources_or_primary(extraction.sources.parking_rate_sources, extraction.candidate.primary_url)
    parking_rate_claim = f"The hotel '{hotel_name}' discloses parking rates."
    claims_and_nodes.append((parking_rate_claim, parking_rate_sources, parking_rate_leaf,
                             "Look for specific dollar amounts or rate details per night/hour. General parking mention without pricing does not satisfy."))

    # Free cancellation ≥ 24h before check-in
    cancel_leaf = evaluator.add_leaf(
        id="Free_Cancellation_24h_Before_Checkin",
        desc="Hotel has a cancellation policy allowing free cancellation at least 24 hours before check-in.",
        parent=node,
        critical=True
    )
    cancel_sources = _sources_or_primary(extraction.sources.cancellation_sources, extraction.candidate.primary_url)
    cancel_claim = f"The hotel's cancellation policy allows free cancellation at least 24 hours before check-in."
    claims_and_nodes.append((cancel_claim, cancel_sources, cancel_leaf,
                             "Policy must explicitly allow free cancellation at least 24 hours (or more) prior to check-in; stricter policies do not satisfy."))

    # On-site fitness center
    fitness_leaf = evaluator.add_leaf(
        id="On_Site_Fitness_Center",
        desc="Hotel has an on-site fitness center.",
        parent=node,
        critical=True
    )
    fitness_sources = _sources_or_primary(extraction.sources.fitness_sources, extraction.candidate.primary_url)
    fitness_claim = f"The hotel '{hotel_name}' has an on-site fitness center."
    claims_and_nodes.append((fitness_claim, fitness_sources, fitness_leaf,
                             "Accept synonyms like 'gym', 'fitness center', 'health club' located on property."))

    # High-speed business WiFi
    wifi_leaf = evaluator.add_leaf(
        id="High_Speed_Business_WiFi",
        desc="Hotel provides high-speed WiFi suitable for business use (e.g., described as business-grade/high-speed or specifies a minimum such as ≥50 Mbps).",
        parent=node,
        critical=True
    )
    wifi_sources = _sources_or_primary(extraction.sources.wifi_sources, extraction.candidate.primary_url)
    wifi_claim = f"The hotel '{hotel_name}' provides high-speed WiFi suitable for business use."
    claims_and_nodes.append((wifi_claim, wifi_sources, wifi_leaf,
                             "Look for 'high-speed', 'business-grade', or a speed specification (e.g., 50 Mbps or higher). General 'WiFi available' without speed/business suitability does not satisfy."))

    # Meeting/conference room space
    meeting_leaf = evaluator.add_leaf(
        id="Meeting_Conference_Space",
        desc="Hotel has meeting/conference room space available.",
        parent=node,
        critical=True
    )
    meeting_sources = _sources_or_primary(extraction.sources.meeting_sources, extraction.candidate.primary_url)
    meeting_claim = f"The hotel '{hotel_name}' has meeting or conference room space available."
    claims_and_nodes.append((meeting_claim, meeting_sources, meeting_leaf,
                             "Accept terms like 'meeting rooms', 'conference space', 'event space', 'ballroom' that guests can reserve."))

    # Batch verify all parallel leaves under this node
    await evaluator.batch_verify(claims_and_nodes)

    return node


def _all_constraints_have_urls(extraction: HotelExtraction) -> bool:
    """
    Check whether the answer provides at least one supporting URL for each required constraint
    and at least one URL verifying the hotel identity (primary_url counts).
    """
    s = extraction.sources
    per_constraint_urls_present = all([
        len(s.distance_sources) > 0,
        len(s.shuttle_sources) > 0,
        len(s.ada_room_sources) > 0,
        len(s.ada_door_sources) > 0,
        len(s.pet_sources) > 0,
        len(s.pet_fee_sources) > 0,
        len(s.parking_sources) > 0,
        len(s.parking_rate_sources) > 0,
        len(s.cancellation_sources) > 0,
        len(s.fitness_sources) > 0,
        len(s.wifi_sources) > 0,
        len(s.meeting_sources) > 0,
    ])
    identity_has_url = _non_empty(extraction.candidate.primary_url)
    return per_constraint_urls_present and identity_has_url


async def _build_supporting_urls_check(
    evaluator: Evaluator,
    parent: Any,
    extraction: HotelExtraction
) -> Any:
    """
    Build the 'Provide_Supporting_Reference_URLs' critical leaf as a custom existence/coverage check.
    """
    # This is a single leaf under a critical sequential parent
    result = _all_constraints_have_urls(extraction)
    evaluator.add_custom_node(
        result=result,
        id="Provide_Supporting_Reference_URLs",
        desc="Response provides supporting reference URL(s) to verify the hotel identity and each required constraint above (not just uncited assertions).",
        parent=parent,
        critical=True
    )
    return parent


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
    Evaluate the answer for the Boston Logan hotel requirements task.
    """
    # Initialize evaluator with a neutral root, then attach critical sequential sub-root
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

    # Add a critical sequential node representing the overall hotel requirements evaluation
    hotel_req_root = evaluator.add_sequential(
        id="Hotel_Requirements",
        desc="Evaluate whether a single identified hotel satisfies all stated constraints and provides supporting reference URLs for verification.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction"
    )

    # Record ground truth task requirements for transparency
    evaluator.add_ground_truth({
        "requirements": REQUIREMENTS_LIST,
        "airport": "Boston Logan International Airport (BOS)"
    })

    # Build Identify-One-Hotel checks
    await _build_identify_one_hotel(evaluator, hotel_req_root, extraction)

    # Build Verify-All-Constraints checks
    await _build_verify_all_constraints(evaluator, hotel_req_root, extraction)

    # Build Supporting URLs existence/coverage check
    await _build_supporting_urls_check(evaluator, hotel_req_root, extraction)

    # Return summary with verification tree and aggregated score
    return evaluator.get_summary()