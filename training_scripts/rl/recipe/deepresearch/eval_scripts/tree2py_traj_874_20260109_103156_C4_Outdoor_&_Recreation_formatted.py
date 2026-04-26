import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_coastal_rv_accessible_campground"
TASK_DESCRIPTION = (
    "Identify a California State Park campground that meets the following requirements for a planned RV camping trip: "
    "(1) The campground must be located on the California coast with oceanfront or beachfront access; "
    "(2) The campground must offer RV camping sites with full hookups (water, sewer, and electric); "
    "(3) The campground must accommodate RVs that are at least 30 feet in length; "
    "(4) The campground must have wheelchair-accessible campsites available; "
    "(5) The campground must provide wheelchair-accessible restroom and shower facilities; "
    "(6) Reservations must be available through ReserveCalifornia.com or by calling 1-800-444-7275; "
    "(7) The nightly fee for RV sites with full hookups should be between $60 and $75 per night; "
    "(8) The campground should allow pets in the camping area (dogs must be on leash). "
    "Provide the name of the campground and supporting reference URL(s) that verify it meets these requirements."
)

RESERVECALIFORNIA_PHONE = "1-800-444-7275"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundExtraction(BaseModel):
    """
    Extract the primary campground name and the reference URLs cited in the answer.
    If multiple campgrounds are mentioned, pick the first one that is actually recommended by the answer
    (or the first mentioned, if not clearly recommended).
    """
    campground_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campground() -> str:
    return """
    Extract the primary campground identified in the answer and the reference URLs used to support the claims.
    Requirements:
    - campground_name: The name of the specific campground that the answer recommends or focuses on.
      If multiple campgrounds are listed, choose the first recommended one (or the first one mentioned if no explicit recommendation).
      Return null if no campground is identified.
    - reference_urls: A list of all URLs cited as evidence for this campground's details (facilities, reservations, accessibility, etc.).
      Include any ReserveCalifornia page, California State Parks official page, or other relevant authoritative sources explicitly linked in the answer.
      Only include valid URLs actually present in the answer text. Do not invent URLs.

    Return a JSON object with:
    {
      "campground_name": string or null,
      "reference_urls": [url1, url2, ...]
    }
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_campground_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: CampgroundExtraction,
) -> None:
    """
    Build leaf checks according to the rubric and verify them against the provided reference URLs.
    """

    camp_name = extraction.campground_name or "the identified campground"
    sources = extraction.reference_urls or []

    # Create the main rubric node (set as non-critical to allow mixed critical/non-critical children)
    main_node = evaluator.add_parallel(
        id="California_Coastal_RV_Campground_Identification",
        desc="Evaluate whether the identified campground meets all specified requirements for coastal RV camping with accessibility features",
        parent=parent_node,
        critical=False
    )

    # Common additional instruction for all URL-based checks
    common_source_instruction = (
        "Use only the provided reference URLs as evidence. Do not rely on the answer text itself. "
        "If no valid URLs are provided for this verification, you must mark the claim as not supported (Incorrect). "
        "Allow reasonable synonyms and phrasing variants when matching the requirement."
    )

    # Prepare leaf nodes
    coastal_node = evaluator.add_leaf(
        id="Coastal_Oceanfront_Location",
        desc="The campground must be located on the California coast with oceanfront or beachfront access",
        parent=main_node,
        critical=True
    )
    full_hookups_node = evaluator.add_leaf(
        id="Full_Hookup_RV_Sites",
        desc="The campground must offer RV sites with full hookups (water, sewer, and electric)",
        parent=main_node,
        critical=True
    )
    rv_length_node = evaluator.add_leaf(
        id="RV_Length_Accommodation",
        desc="The campground must accommodate RVs of at least 30 feet in length",
        parent=main_node,
        critical=True
    )
    ada_sites_node = evaluator.add_leaf(
        id="ADA_Accessible_Campsites",
        desc="The campground must have wheelchair-accessible campsites available",
        parent=main_node,
        critical=True
    )
    accessible_restroom_shower_node = evaluator.add_leaf(
        id="Accessible_Restroom_Shower_Facilities",
        desc="The campground must provide wheelchair-accessible restroom and shower facilities",
        parent=main_node,
        critical=True
    )
    reservecal_node = evaluator.add_leaf(
        id="ReserveCalifornia_Booking",
        desc="The campground must accept reservations through ReserveCalifornia.com or 1-800-444-7275",
        parent=main_node,
        critical=True
    )
    fee_range_node = evaluator.add_leaf(
        id="RV_Hookup_Fee_Range",
        desc="The nightly fee for RV sites with full hookups must be between $60 and $75",
        parent=main_node,
        critical=False
    )
    pet_friendly_node = evaluator.add_leaf(
        id="Pet_Friendly_Campground",
        desc="The campground must allow pets in the camping area (on leash)",
        parent=main_node,
        critical=False
    )

    # Build claims and instructions
    claims_and_sources = [
        (
            f"The campground named '{camp_name}' is located on the California coast and has oceanfront or beachfront access to the Pacific Ocean.",
            sources,
            coastal_node,
            common_source_instruction + " Confirm explicit oceanfront/beachfront access or being directly on/adjacent to the beach."
        ),
        (
            f"The campground named '{camp_name}' offers RV campsites with full hookups, meaning water, sewer, and electric are available at the site.",
            sources,
            full_hookups_node,
            common_source_instruction + " Only pass if full hookups (water + sewer + electric) are explicitly available. Do not pass for partial hookups."
        ),
        (
            f"The campground named '{camp_name}' accommodates RVs that are at least 30 feet in length (i.e., the maximum allowed RV/trailer length is 30 feet or greater).",
            sources,
            rv_length_node,
            common_source_instruction + " Accept if the site or campground maximum length is 30 ft or greater for RVs/trailers/motorhomes."
        ),
        (
            f"The campground named '{camp_name}' has wheelchair-accessible (ADA) campsites available.",
            sources,
            ada_sites_node,
            common_source_instruction + " Look for terms like 'ADA site', 'accessible campsite(s)', or equivalent phrasing indicating accessible campsites."
        ),
        (
            f"The campground named '{camp_name}' provides wheelchair-accessible restroom and shower facilities.",
            sources,
            accessible_restroom_shower_node,
            common_source_instruction + " Both restrooms and showers should be accessible. If only one is accessible and the other is not mentioned, do not pass."
        ),
        (
            f"Reservations for the campground named '{camp_name}' are available through ReserveCalifornia.com or by calling {RESERVECALIFORNIA_PHONE}.",
            sources,
            reservecal_node,
            common_source_instruction + " Evidence can include a ReserveCalifornia page for the campground or explicit mention of ReserveCalifornia or the phone number."
        ),
        (
            f"The nightly fee for full-hookup RV sites at '{camp_name}' is between $60 and $75 per night.",
            sources,
            fee_range_node,
            common_source_instruction + " Allow reasonable seasonal variation; pass if a typical or listed full-hookup rate falls within $60–$75."
        ),
        (
            f"The campground named '{camp_name}' allows pets (dogs) in the camping area, with a leash requirement.",
            sources,
            pet_friendly_node,
            common_source_instruction + " It's okay if pets are restricted from beaches or trails; the claim concerns the camping area only."
        ),
    ]

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the California coastal RV campground with accessibility requirements task.
    """
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

    # Extract campground name and reference URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_campground(),
        template_class=CampgroundExtraction,
        extraction_name="campground_extraction",
    )

    # Record custom info for debugging
    evaluator.add_custom_info(
        info={
            "campground_name": extraction.campground_name,
            "reference_urls": extraction.reference_urls
        },
        info_type="extraction_overview",
        info_name="campground_overview"
    )

    # Build verification nodes according to rubric and verify
    await verify_campground_requirements(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()