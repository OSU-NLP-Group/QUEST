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
TASK_ID = "den_airport_hotel_requirements"
TASK_DESCRIPTION = """
I need to find a hotel in the Denver Airport area that meets all of the following requirements for an upcoming business trip:

1. The hotel must be located in the Denver Airport area and serve Denver International Airport
2. The hotel must allow pets (specifically dogs)
3. The hotel must have wheelchair-accessible guest rooms
4. The hotel must provide wheelchair-accessible parking spaces
5. The hotel must have a swimming pool (indoor or outdoor)
6. The hotel must have a fitness center or gym facility
7. The hotel must provide airport shuttle service to/from Denver International Airport
8. The hotel must offer on-site parking (free or paid)
9. The hotel must have an on-site restaurant or dining facility
10. The hotel must have a business center with computer and printing services
11. The hotel must have meeting rooms or conference space available
12. The hotel must provide free WiFi to guests
13. The hotel must offer complimentary breakfast
14. The hotel must be a non-smoking property

Please provide the name of one hotel that meets all of these requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelCandidate(BaseModel):
    hotel_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AmenityChecklist(BaseModel):
    location: Optional[str] = None
    pet_friendly: Optional[str] = None
    wheelchair_accessible_rooms: Optional[str] = None
    accessible_parking: Optional[str] = None
    pool: Optional[str] = None
    fitness_center: Optional[str] = None
    airport_shuttle: Optional[str] = None
    parking: Optional[str] = None
    restaurant: Optional[str] = None
    business_center: Optional[str] = None
    meeting_rooms: Optional[str] = None
    wifi: Optional[str] = None
    breakfast: Optional[str] = None
    non_smoking: Optional[str] = None


class HotelExtraction(BaseModel):
    hotel: Optional[HotelCandidate] = None
    amenities: AmenityChecklist = Field(default_factory=AmenityChecklist)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return """
    Extract the single hotel candidate mentioned in the answer and all explicitly cited source URLs, along with any amenity statements the answer claims.

    Return a JSON object with:
    - hotel:
        - hotel_name: The hotel name provided (if multiple are mentioned, choose the primary/first one).
        - sources: An array of URLs explicitly present in the answer that are relevant to the hotel (official site, booking pages, aggregator listings, etc.). Do not invent any URLs.
    - amenities:
        For each amenity field below, extract the exact text snippet from the answer that claims the amenity. If not mentioned, set the field to null.
        Fields:
          - location
          - pet_friendly
          - wheelchair_accessible_rooms
          - accessible_parking
          - pool
          - fitness_center
          - airport_shuttle
          - parking
          - restaurant
          - business_center
          - meeting_rooms
          - wifi
          - breakfast
          - non_smoking

    Rules:
    - Extract only what is explicitly present in the answer.
    - For URLs, include full URLs (prepend http:// if missing protocol).
    - If the answer lists more than one URL, include them all in 'sources'.
    """


# --------------------------------------------------------------------------- #
# Helper: Amenity claims and instructions                                     #
# --------------------------------------------------------------------------- #
def build_amenity_claims(hotel_name: str) -> Dict[str, str]:
    # Creates precise claims for verification
    return {
        "location": f"The hotel '{hotel_name}' is located in the Denver Airport area and serves Denver International Airport (DEN/DIA).",
        "pet_friendly": f"The hotel '{hotel_name}' allows pets (dogs).",
        "wheelchair_accessible_rooms": f"The hotel '{hotel_name}' has wheelchair-accessible guest rooms (ADA-compliant accessible rooms).",
        "accessible_parking": f"The hotel '{hotel_name}' provides wheelchair-accessible parking spaces.",
        "pool": f"The hotel '{hotel_name}' has a swimming pool (indoor or outdoor).",
        "fitness_center": f"The hotel '{hotel_name}' has a fitness center or gym facility.",
        "airport_shuttle": f"The hotel '{hotel_name}' provides airport shuttle service to/from Denver International Airport (DEN).",
        "parking": f"The hotel '{hotel_name}' offers on-site parking.",
        "restaurant": f"The hotel '{hotel_name}' has an on-site restaurant or dining facility.",
        "business_center": f"The hotel '{hotel_name}' has a business center with computer and printing services.",
        "meeting_rooms": f"The hotel '{hotel_name}' has meeting rooms or conference space available.",
        "wifi": f"The hotel '{hotel_name}' provides free WiFi to guests.",
        "breakfast": f"The hotel '{hotel_name}' offers complimentary breakfast.",
        "non_smoking": f"The hotel '{hotel_name}' is a non-smoking property."
    }


def build_amenity_instructions(amenities: AmenityChecklist) -> Dict[str, str]:
    # Additional instructions per amenity to guide verification robustly
    return {
        "location": (
            "Confirm that the page explicitly ties the property to Denver International Airport (DEN/DIA). "
            "Accept phrases like 'near Denver International Airport', 'airport area', 'serves DEN', or listings explicitly for DEN."
        ),
        "pet_friendly": (
            "Look for 'pet-friendly', 'pets allowed', or policy pages that mention dogs. "
            "Reject service-animals-only policies unless they also allow pets/dogs."
        ),
        "wheelchair_accessible_rooms": (
            "Look for 'accessible rooms', 'ADA rooms', or similar phrasing that indicates wheelchair-accessible guest rooms."
        ),
        "accessible_parking": (
            "Look for 'accessible parking', 'ADA parking', or similar phrasing that indicates wheelchair-accessible parking spaces."
        ),
        "pool": (
            "Look for 'pool', 'indoor pool', or 'outdoor pool' in amenities or features."
        ),
        "fitness_center": (
            "Look for 'fitness center', 'gym', or 'health club' in amenities."
        ),
        "airport_shuttle": (
            "Look for 'airport shuttle' or 'shuttle to/from DEN'. "
            "Accept paid or complimentary shuttle, but it must be airport-specific."
        ),
        "parking": (
            "Look for 'on-site parking', 'self parking', or 'parking available' on property."
        ),
        "restaurant": (
            "Look for 'on-site restaurant', 'dining', 'bistro', or any on-property food service venue."
        ),
        "business_center": (
            "Look for 'business center' along with availability of computers and printers (or equivalent office services)."
        ),
        "meeting_rooms": (
            "Look for 'meeting rooms', 'event space', or 'conference facilities' available for booking."
        ),
        "wifi": (
            "It must be 'free WiFi' or 'complimentary wireless internet' for guests; paid-only WiFi does not count."
        ),
        "breakfast": (
            "Look for 'complimentary breakfast' or 'free breakfast'. "
            "If breakfast is only paid, do not count."
        ),
        "non_smoking": (
            "Look for 'non-smoking property', 'smoke-free hotel', or equivalent policy indicating no smoking in rooms/property."
        ),
    }


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hotel_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: HotelExtraction
) -> None:
    # Create a critical parallel node to enforce ALL requirements must pass
    requirements_node = evaluator.add_parallel(
        id="requirements_main",
        desc="Hotel in Denver Airport area meeting all specified amenity requirements",
        parent=parent_node,
        critical=True
    )

    # Existence checks (critical preconditions)
    hotel_name_ok = bool(extraction.hotel and extraction.hotel.hotel_name and extraction.hotel.hotel_name.strip())
    sources_ok = bool(extraction.hotel and extraction.hotel.sources and len(extraction.hotel.sources) > 0)

    evaluator.add_custom_node(
        result=hotel_name_ok,
        id="hotel_name_provided",
        desc="A hotel name is provided in the answer",
        parent=requirements_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=sources_ok,
        id="hotel_sources_provided",
        desc="At least one source URL is provided in the answer for the hotel",
        parent=requirements_node,
        critical=True
    )

    # Prepare claim and instructions per amenity
    hotel_name = extraction.hotel.hotel_name if extraction.hotel else ""
    sources = extraction.hotel.sources if extraction.hotel else []
    amenity_claims = build_amenity_claims(hotel_name)
    amenity_instructions = build_amenity_instructions(extraction.amenities)

    # For each amenity, add a critical leaf and verify with sources
    def add_and_verify_leaf(node_id: str, description: str, claim: str, add_ins: str) -> None:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=description,
            parent=requirements_node,
            critical=True
        )
        # The evaluator.verify will automatically short-circuit if preconditions failed
        # due to critical siblings (hotel_name_provided / hotel_sources_provided).
        asyncio.create_task(
            evaluator.verify(
                claim=claim,
                node=leaf,
                sources=sources,
                additional_instruction=add_ins
            )
        )

    # Create and schedule verifications. We will await them after creation for proper ordering.
    tasks: List[asyncio.Task] = []

    def schedule(node_id: str, desc: str, key: str):
        claim = amenity_claims[key]
        add_ins = amenity_instructions[key]
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=requirements_node,
            critical=True
        )
        tasks.append(asyncio.create_task(
            evaluator.verify(
                claim=claim,
                node=leaf,
                sources=sources,
                additional_instruction=add_ins
            )
        ))

    # Map rubric leaves
    schedule("location", "Hotel is located in the Denver Airport area (serves Denver International Airport)", "location")
    schedule("pet_friendly", "Hotel allows pets (specifically dogs)", "pet_friendly")
    schedule("wheelchair_accessible_rooms", "Hotel has wheelchair-accessible guest rooms", "wheelchair_accessible_rooms")
    schedule("accessible_parking", "Hotel provides wheelchair-accessible parking spaces", "accessible_parking")
    schedule("pool", "Hotel has an indoor or outdoor swimming pool", "pool")
    schedule("fitness_center", "Hotel has a fitness center or gym facility", "fitness_center")
    schedule("airport_shuttle", "Hotel provides airport shuttle service to/from Denver International Airport", "airport_shuttle")
    schedule("parking", "Hotel offers on-site parking (free or paid)", "parking")
    schedule("restaurant", "Hotel has an on-site restaurant or dining facility", "restaurant")
    schedule("business_center", "Hotel has a business center with computer and printing services", "business_center")
    schedule("meeting_rooms", "Hotel has meeting rooms or conference space available", "meeting_rooms")
    schedule("wifi", "Hotel provides free WiFi to guests", "wifi")
    schedule("breakfast", "Hotel offers complimentary breakfast", "breakfast")
    schedule("non_smoking", "Hotel is a non-smoking property", "non_smoking")

    # Await all verification tasks
    if tasks:
        await asyncio.gather(*tasks)


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
    Evaluate an answer for the Denver Airport hotel requirements task.
    """
    # Initialize evaluator (root node is always non-critical per framework)
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

    # Extract hotel candidate and amenity statements
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction"
    )

    # Record custom info: basic extracted snapshot
    evaluator.add_custom_info(
        info={
            "hotel_name": extraction.hotel.hotel_name if extraction.hotel else None,
            "num_sources": len(extraction.hotel.sources) if extraction.hotel else 0,
            "sources": extraction.hotel.sources if extraction.hotel else []
        },
        info_type="extraction_summary",
        info_name="hotel_extraction_summary"
    )

    # Build verification tree and run checks
    await verify_hotel_requirements(
        evaluator=evaluator,
        parent_node=root,
        extraction=extraction
    )

    # Return structured evaluation summary
    return evaluator.get_summary()