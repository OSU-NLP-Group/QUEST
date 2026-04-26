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
TASK_ID = "ush_partner_hotel_selection"
TASK_DESCRIPTION = """
I am planning a trip to Universal Studios Hollywood and would like to stay at a hotel that is conveniently located near the park. Identify one hotel that meets all of the following requirements:

1. The hotel must be an official Universal Studios Hollywood Partner Hotel
2. The hotel must be within walking distance (1 mile or less) from the Universal Studios Hollywood entrance
3. The hotel must offer complimentary shuttle service to Universal Studios Hollywood

Please provide the hotel's full name, complete street address, and a reference URL from the official Universal Studios Hollywood partner hotels listing.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelAnswerExtraction(BaseModel):
    """
    Structured extraction of the single hotel proposed by the answer.
    """
    hotel_name: Optional[str] = None
    street_address: Optional[str] = None
    partner_listing_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract the single hotel proposed in the answer that is intended to meet the Universal Studios Hollywood partner hotel requirements.

    Return a JSON object with the following fields:
    - hotel_name: The hotel's full official name as stated in the answer text.
    - street_address: The hotel's complete street address (including street number/name, city, state, and postal code) as provided in the answer text. If the answer provides multiple lines, combine them into a single string.
    - partner_listing_url: The URL to the official Universal Studios Hollywood Partner Hotels listing page for this specific hotel. It should be a URL hosted on the official Universal Studios Hollywood domain or subpages clearly labeled as "Partner Hotels" or "Official Hotels". If the answer does not provide such a URL explicitly, return null.
    - additional_urls: An array of any other URLs in the answer that relate to this hotel (e.g., hotel homepage, maps, travel pages). If none are present, return an empty array.

    IMPORTANT:
    - Extract only what is explicitly present in the answer; do not invent or infer any content.
    - If a field is missing, set it to null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
async def build_and_verify_hotel(
    evaluator: Evaluator,
    parent_node,
    extracted: HotelAnswerExtraction,
) -> None:
    """
    Build the verification tree for the hotel selection and run verifications.
    """
    # Create the top-level "Hotel_Selection" node as critical (as per rubric)
    hotel_node = evaluator.add_parallel(
        id="Hotel_Selection",
        desc="Identify one hotel that meets all specified requirements for staying near Universal Studios Hollywood",
        parent=parent_node,
        critical=True
    )

    # Existence / completeness check (critical)
    complete_info = (
        bool(extracted.hotel_name and extracted.hotel_name.strip()) and
        bool(extracted.street_address and extracted.street_address.strip()) and
        bool(extracted.partner_listing_url and extracted.partner_listing_url.strip())
    )
    evaluator.add_custom_node(
        result=complete_info,
        id="Complete_Information_Provided",
        desc="The answer includes the hotel's full name, complete street address, and a reference URL from the official Universal Studios Hollywood partner hotels listing",
        parent=hotel_node,
        critical=True
    )

    # Prepare common data
    hotel_name = extracted.hotel_name or ""
    partner_url = extracted.partner_listing_url or ""
    multi_sources_for_distance = [u for u in ([partner_url] + (extracted.additional_urls or [])) if u]

    # Leaf: Official Partner Status (critical)
    partner_status_node = evaluator.add_leaf(
        id="Official_Partner_Status",
        desc="The hotel is listed as an official Universal Studios Hollywood Partner Hotel on the official Universal Studios Hollywood partner hotels page",
        parent=hotel_node,
        critical=True
    )
    partner_status_claim = (
        f"The hotel '{hotel_name}' is listed as an official Universal Studios Hollywood Partner Hotel on this page."
    )
    await evaluator.verify(
        claim=partner_status_claim,
        node=partner_status_node,
        sources=partner_url if partner_url else None,
        additional_instruction=(
            "Verify that this page is part of the official Universal Studios Hollywood website and is a Partner Hotels listing. "
            "Confirm that the hotel's name appears as an official partner. Allow minor name variants (case differences, punctuation)."
        )
    )

    # Leaf: Walking Distance Requirement (critical)
    walking_node = evaluator.add_leaf(
        id="Walking_Distance_Requirement",
        desc="The hotel is located within walking distance (1 mile or less) from Universal Studios Hollywood entrance",
        parent=hotel_node,
        critical=True
    )
    walking_claim = (
        "This page indicates the hotel is within walking distance (1 mile or less) from the Universal Studios Hollywood entrance."
    )
    await evaluator.verify(
        claim=walking_claim,
        node=walking_node,
        sources=multi_sources_for_distance if multi_sources_for_distance else (partner_url if partner_url else None),
        additional_instruction=(
            "Look for phrases like 'walking distance', 'short walk', 'steps away', or an explicit distance ≤ 1 mile. "
            "If the page does not explicitly state walking distance or provides a distance greater than 1 mile, mark as not supported."
        )
    )

    # Leaf: Complimentary Shuttle Service (critical)
    shuttle_node = evaluator.add_leaf(
        id="Complimentary_Shuttle_Service",
        desc="The hotel offers complimentary shuttle service to Universal Studios Hollywood as stated in the official partner hotels listing",
        parent=hotel_node,
        critical=True
    )
    shuttle_claim = (
        f"This official partner hotels listing page states that '{hotel_name}' offers complimentary shuttle service to Universal Studios Hollywood."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_node,
        sources=partner_url if partner_url else None,
        additional_instruction=(
            "Check the page text for 'complimentary shuttle', 'free shuttle', or equivalent language indicating no cost. "
            "The shuttle must be to Universal Studios Hollywood; generic local shuttle without mention of Universal Studios Hollywood does not satisfy the requirement."
        )
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
    Evaluate an answer for the Universal Studios Hollywood partner hotel selection task.
    """
    # Initialize evaluator
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

    # Extract hotel info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelAnswerExtraction,
        extraction_name="hotel_info_extraction",
    )

    # Add a summary of task requirements (optional info)
    evaluator.add_custom_info(
        info={
            "requirements": [
                "Official Universal Studios Hollywood Partner Hotel",
                "Within walking distance (≤ 1 mile)",
                "Complimentary shuttle to Universal Studios Hollywood",
                "Provide hotel name, full street address, and official partner listing URL"
            ]
        },
        info_type="task_requirements"
    )

    # Build verification tree and run checks
    await build_and_verify_hotel(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()