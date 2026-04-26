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
TASK_ID = "sfo_hotel_shuttle_24_15"
TASK_DESCRIPTION = (
    "I am looking for a hotel near San Francisco International Airport (SFO) that offers a complimentary airport shuttle service. "
    "The shuttle must operate 24 hours per day and run at least every 15 minutes. "
    "Please provide the name of one hotel that meets these requirements, along with a reference URL from the hotel's official website "
    "or a reliable source confirming the shuttle service details."
)


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class HotelCandidateExtraction(BaseModel):
    """
    Extract a single hotel candidate from the answer, along with any cited reference URLs
    and the textual claims the answer made about shuttle properties (if present).
    """
    hotel_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    complimentary_claim: Optional[str] = None
    operates_24h_claim: Optional[str] = None
    frequency_claim: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_candidate() -> str:
    return (
        "From the provided answer, extract details for the FIRST hotel candidate mentioned that is intended to meet the SFO shuttle requirements.\n"
        "Return the following fields:\n"
        "1) hotel_name: The hotel's name exactly as written in the answer.\n"
        "2) reference_urls: All explicit URLs cited in the answer that are meant to support the shuttle service details for this hotel. "
        "   Include both official hotel pages and any third-party pages if they are mentioned. If none are present, return an empty list.\n"
        "3) complimentary_claim: The exact phrase or sentence in the answer that states the shuttle is complimentary/free (if present), else null.\n"
        "4) operates_24h_claim: The exact phrase or sentence in the answer that states the shuttle operates 24 hours per day (if present), else null.\n"
        "5) frequency_claim: The exact phrase or sentence in the answer that states the shuttle frequency (e.g., 'every 10 minutes', 'every 15 minutes') "
        "(if present), else null.\n"
        "If multiple hotels or multiple sets of URLs are provided in the answer, only extract the FIRST hotel's name and ALL URLs associated with that first hotel.\n"
        "Do not invent or infer any information not explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hotel_shuttle_requirements(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelCandidateExtraction,
) -> None:
    """
    Build the verification subtree for the hotel shuttle requirements and run verifications.
    The subtree follows the rubric:
      - Hotel_with_Required_Shuttle_Service (critical, parallel)
          • Hotel_Located_Near_SFO (critical leaf)
          • Complimentary_Airport_Shuttle (critical leaf)
          • Shuttle_Operates_24_Hours (critical leaf)
          • Shuttle_Frequency_At_Least_Every_15_Min (critical leaf)
          • Reference_URL_Provided_And_Reliable (critical leaf)
    """
    main_node = evaluator.add_parallel(
        id="Hotel_with_Required_Shuttle_Service",
        desc=(
            "Identify one hotel near San Francisco International Airport (SFO) that offers a complimentary airport shuttle "
            "meeting the specified operating schedule and provide a verifying reference URL."
        ),
        parent=parent_node,
        critical=True,
    )

    # Create leaf nodes
    located_node = evaluator.add_leaf(
        id="Hotel_Located_Near_SFO",
        desc="The hotel is located near San Francisco International Airport (SFO).",
        parent=main_node,
        critical=True,
    )
    complimentary_node = evaluator.add_leaf(
        id="Complimentary_Airport_Shuttle",
        desc="The hotel offers a complimentary (free) airport shuttle service to/from SFO.",
        parent=main_node,
        critical=True,
    )
    operates_24h_node = evaluator.add_leaf(
        id="Shuttle_Operates_24_Hours",
        desc="The airport shuttle operates 24 hours per day (every day of the week).",
        parent=main_node,
        critical=True,
    )
    frequency_node = evaluator.add_leaf(
        id="Shuttle_Frequency_At_Least_Every_15_Min",
        desc="The airport shuttle runs at least every 15 minutes (i.e., 15 minutes or more frequent) as required.",
        parent=main_node,
        critical=True,
    )
    ref_reliable_node = evaluator.add_leaf(
        id="Reference_URL_Provided_And_Reliable",
        desc="A reference URL from the hotel's official website or a reliable third-party source is provided and supports the shuttle details.",
        parent=main_node,
        critical=True,
    )

    # Build claims and sources
    hotel_name = hotel.hotel_name or ""
    sources = hotel.reference_urls if hotel.reference_urls else None

    # Claims
    claim_located = (
        f"The hotel '{hotel_name}' is located near San Francisco International Airport (SFO). "
        f"Acceptable evidence includes the page explicitly stating proximity to SFO, the area being described as 'near SFO', "
        f"or referencing 'airport area' or 'SFO' in the hotel's location description."
    )

    claim_complimentary = (
        f"The hotel '{hotel_name}' offers a complimentary (free) airport shuttle service to/from SFO."
    )

    claim_24h = (
        f"The hotel's airport shuttle for '{hotel_name}' operates 24 hours per day (24/7). "
        f"Synonyms such as '24/7', 'around the clock', or 'runs all day and night' should be acceptable."
    )

    claim_frequency = (
        "The hotel's airport shuttle runs at least every 15 minutes. "
        "Pass only if the page explicitly states a frequency that is 15 minutes or more frequent (e.g., every 15, 12, 10, or 5 minutes). "
        "If the stated interval ever exceeds 15 minutes (e.g., every 20 or 30 minutes, or ranges including 20), it should fail. "
        "If the shuttle is 'on-demand' without a guaranteed schedule, treat it as not meeting the 'every 15 minutes' requirement."
    )

    # Reliability: this check is about the URL itself being official/reputable.
    # We do not bundle schedule verification here (other leaves already check schedule details).
    claim_ref_reliable = (
        "At least one provided reference URL is valid and is from the hotel's official website or a reputable source. "
        "Official sources include the hotel's own domain or the brand's domain (e.g., marriott.com, hilton.com, hyatt.com, ihg.com, wyndhamhotels.com, sonesta.com, radissonhotels.com). "
        "Reputable third-party sources include well-known travel sites (e.g., booking.com, expedia.com) or the airport's official site (flysfo.com). "
        "If no URL is provided or the domain appears untrustworthy, this should fail."
    )

    # Additional instructions per check to guide the verifier
    add_ins_located = (
        "Use the webpage content to determine if the hotel explicitly indicates proximity to SFO. "
        "Direct mentions of 'near SFO', 'airport hotel', or location descriptions referencing SFO are sufficient."
    )

    add_ins_complimentary = (
        "Look for explicit terms like 'complimentary', 'free', or 'no charge' associated with the airport shuttle. "
        "If the page states a fee or surcharge for the shuttle, this should fail."
    )

    add_ins_24h = (
        "Accept '24 hours', '24/7', or equivalent language indicating around-the-clock operation. "
        "If specific limited hours are listed (e.g., 4am–12am), it should fail."
    )

    add_ins_frequency = (
        "Carefully check the stated frequency. Only pass if the maximum interval is 15 minutes or less. "
        "Ranges like 'every 10–15 minutes' should pass, but 'every 15–20 minutes' should fail."
    )

    add_ins_ref_reliable = (
        "Judge reliability by the URL's domain and the nature of the page. "
        "Official hotel or brand domains, airport official pages, or widely recognized travel sites should pass. "
        "Random blogs or unrecognized sites should fail. "
        "If no URL is provided in the answer, this should fail."
    )

    # Execute verifications in parallel for this subtree
    await evaluator.batch_verify(
        [
            (claim_located, sources, located_node, add_ins_located),
            (claim_complimentary, sources, complimentary_node, add_ins_complimentary),
            (claim_24h, sources, operates_24h_node, add_ins_24h),
            (claim_frequency, sources, frequency_node, add_ins_frequency),
            (claim_ref_reliable, sources, ref_reliable_node, add_ins_ref_reliable),
        ]
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate an answer for the SFO hotel shuttle requirement task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Single hotel verification with parallel sub-checks
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

    # Extraction step: parse the hotel candidate from the answer
    hotel = await evaluator.extract(
        prompt=prompt_extract_hotel_candidate(),
        template_class=HotelCandidateExtraction,
        extraction_name="hotel_candidate",
    )

    # Add ground truth criteria as meta info for transparency
    evaluator.add_ground_truth({
        "requirements": {
            "near_airport": "Hotel near San Francisco International Airport (SFO)",
            "complimentary_shuttle": "Shuttle is complimentary (free)",
            "operates_24h": "Shuttle operates 24 hours per day",
            "frequency": "Shuttle runs at least every 15 minutes",
            "reference_url": "Provide official hotel URL or reputable source URL supporting the shuttle details"
        }
    })

    # Build subtree and verify
    await verify_hotel_shuttle_requirements(evaluator, root, hotel)

    # Return evaluation summary
    return evaluator.get_summary()