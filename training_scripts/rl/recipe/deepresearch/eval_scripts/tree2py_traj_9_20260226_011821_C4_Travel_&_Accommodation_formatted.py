import asyncio
import logging
from typing import List, Optional, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hotel_pet_policy_houghton_lake"
TASK_DESCRIPTION = (
    "Identify the name of a lakefront hotel located in Houghton Lake, Michigan that has the following pet policy: "
    "allows a maximum of 2 pets per room, with each pet weighing up to 30 pounds, and charges exactly $25 per pet per night. "
    "Provide the hotel's name and a reference URL from an official or verified source."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_and_urls() -> str:
    return """
    From the answer, extract:
    - hotel_name: The name of the single hotel being recommended or identified that matches the task criteria.
    - reference_urls: All URLs provided in the answer that serve as references for the hotel (official hotel website or verified booking platform pages).

    Rules:
    1) If multiple hotels are mentioned, select the FIRST hotel that appears as the main recommendation and extract its name.
    2) Extract only URLs explicitly present in the answer. Include full URLs; if protocol is missing, prepend http://.
    3) Include both official hotel websites and verified booking platforms (e.g., booking.com, expedia.com, hotels.com, tripadvisor.com, agoda.com, priceline.com; or official chain domains like hilton.com, marriott.com, hyatt.com, ihg.com, wyndhamhotels.com, choicehotels.com, bestwestern.com, redroof.com, motel6.com).
    4) Do not invent any information. If hotel_name is not clearly given, set it to null. If no URLs are provided, return an empty list.

    Return a JSON object with:
    {
      "hotel_name": string or null,
      "reference_urls": array of strings (possibly empty)
    }
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_hotel_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: HotelExtraction,
) -> None:
    """
    Build the verification nodes under the critical Hotel Identification node and run verifications.
    """

    # Create the critical parent node (parallel aggregation, matches rubric)
    hotel_node = evaluator.add_parallel(
        id="Hotel_Identification",
        desc="Correctly identify a pet-friendly lakefront hotel in Houghton Lake, Michigan that meets all specified criteria",
        parent=parent_node,
        critical=True,  # All children must pass; this is a critical requirement
    )

    # Gather data
    hotel_name = (extracted.hotel_name or "").strip()
    sources_list = extracted.reference_urls[:]  # could be empty; verify_by_urls will fail if empty

    # 1) Reference URL: verify the provided URL(s) are valid and from official or verified sources.
    ref_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a valid reference URL from the hotel's official website or verified booking platform",
        parent=hotel_node,
        critical=True,
    )
    ref_claim = (
        f"The provided URL(s) are valid and belong to either the official website or a verified booking platform page "
        f"for the hotel '{hotel_name}' in Houghton Lake, Michigan."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=sources_list,  # verify against one or more URLs; fails if empty
        additional_instruction=(
            "Confirm the URL represents the actual hotel property page. "
            "Official domains include well-known hotel chains (e.g., hilton.com, marriott.com, hyatt.com, ihg.com, wyndhamhotels.com, choicehotels.com, bestwestern.com). "
            "Verified booking platforms include booking.com, expedia.com, hotels.com, agoda.com, priceline.com, tripadvisor.com. "
            "Reject generic blogs or non-official aggregator pages. The page content must correspond to the specific hotel in Houghton Lake, MI."
        ),
    )

    # All other verifications depend on having a valid reference URL.
    extra_prereqs = [ref_leaf]

    # 2) Hotel Name verification
    name_leaf = evaluator.add_leaf(
        id="Hotel_Name",
        desc="Provide the correct name of the hotel",
        parent=hotel_node,
        critical=True,
    )
    name_claim = f"The hotel's official name is '{hotel_name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify the property name as shown on the referenced page. "
            "Allow minor formatting variants (punctuation, inclusion of 'Hotel', 'Inn', 'Lodge', etc.) if they refer to the same property."
        ),
        extra_prerequisites=extra_prereqs,
    )

    # 3) Location verification (Houghton Lake, Michigan)
    loc_leaf = evaluator.add_leaf(
        id="Location_Verification",
        desc="Confirm the hotel is located in Houghton Lake, Michigan",
        parent=hotel_node,
        critical=True,
    )
    loc_claim = f"The hotel '{hotel_name}' is located in Houghton Lake, Michigan (MI)."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=sources_list,
        additional_instruction=(
            "Check the address or location details on the page. "
            "Accept 'Houghton Lake, MI' or 'Houghton Lake, Michigan'."
        ),
        extra_prerequisites=extra_prereqs,
    )

    # 4) Pet weight limit (up to 30 pounds per pet)
    weight_leaf = evaluator.add_leaf(
        id="Pet_Weight_Limit",
        desc="Verify the hotel allows pets up to 30 pounds each",
        parent=hotel_node,
        critical=True,
    )
    weight_claim = "The hotel's pet policy allows each pet to weigh up to 30 pounds (30 lbs)."
    await evaluator.verify(
        claim=weight_claim,
        node=weight_leaf,
        sources=sources_list,
        additional_instruction=(
            "Look for phrases such as 'up to 30 pounds', '30 lb limit', or 'pets up to 30 lbs'. "
            "If the weight limit is unspecified or differs from 30 lbs, this should fail."
        ),
        extra_prerequisites=extra_prereqs,
    )

    # 5) Maximum pets per room (2)
    max_pets_leaf = evaluator.add_leaf(
        id="Maximum_Pets",
        desc="Confirm the hotel allows a maximum of 2 pets per room",
        parent=hotel_node,
        critical=True,
    )
    max_pets_claim = "The hotel's pet policy allows a maximum of 2 pets per room."
    await evaluator.verify(
        claim=max_pets_claim,
        node=max_pets_leaf,
        sources=sources_list,
        additional_instruction=(
            "Look for 'maximum 2 pets', 'two pets per room', or 'up to 2 pets per room'. "
            "If it states 1 pet or more than 2, or if it is unclear, this should fail."
        ),
        extra_prerequisites=extra_prereqs,
    )

    # 6) Pet fee ($25 per pet per night)
    fee_leaf = evaluator.add_leaf(
        id="Pet_Fee",
        desc="Verify the pet fee is $25 per pet per night",
        parent=hotel_node,
        critical=True,
    )
    fee_claim = "The hotel's pet fee is exactly $25 per pet per night."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=sources_list,
        additional_instruction=(
            "The wording must clearly indicate $25 per pet per night. "
            "Accept '$25.00' and equivalent currency formatting. "
            "Reject 'per stay' or different amounts."
        ),
        extra_prerequisites=extra_prereqs,
    )

    # 7) Lakefront location
    lakefront_leaf = evaluator.add_leaf(
        id="Lakefront_Location",
        desc="Confirm the hotel is located on a lakefront/beachfront property",
        parent=hotel_node,
        critical=True,
    )
    lakefront_claim = "The property is lakefront/beachfront on Houghton Lake (i.e., waterfront or on the shores of Houghton Lake)."
    await evaluator.verify(
        claim=lakefront_claim,
        node=lakefront_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify terms like 'lakefront', 'waterfront', 'on Houghton Lake', 'on the shores of Houghton Lake', or 'private beach on Houghton Lake'. "
            "General proximity to the lake without explicit lakefront/beachfront should not count."
        ),
        extra_prerequisites=extra_prereqs,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Houghton Lake pet policy hotel identification task.
    """

    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; critical child node handles gating
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

    # Extract hotel name and reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_and_urls(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Optionally record custom info to help debugging
    evaluator.add_custom_info(
        info={"hotel_name": extracted.hotel_name, "reference_urls": extracted.reference_urls},
        info_type="extraction_summary",
        info_name="extracted_hotel_info",
    )

    # Build verification nodes and run verifications
    await build_and_verify_hotel_nodes(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()