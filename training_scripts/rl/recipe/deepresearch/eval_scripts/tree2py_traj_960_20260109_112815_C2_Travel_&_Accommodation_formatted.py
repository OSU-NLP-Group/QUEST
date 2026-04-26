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
TASK_ID = "grand_canyon_pets_south_rim"
TASK_DESCRIPTION = (
    "I am planning a trip to Grand Canyon National Park and need to bring my dog. "
    "Identify the lodge at the South Rim that officially allows pets (located within the park boundaries). "
    "Provide the following information: the name of the pet-friendly lodge, confirmation that it is located within Grand Canyon National Park at the South Rim, "
    "at least one official reservation phone number, a direct link to the lodge's official webpage or online reservation page, "
    "a reference URL from an official source (such as the National Park Service or authorized concessionaire website) that confirms the pet-friendly policy, "
    "and whether there is an additional fee for bringing pets."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LodgeInfo(BaseModel):
    lodge_name: Optional[str] = None
    official_website_url: Optional[str] = None
    policy_reference_url: Optional[str] = None
    reservation_phone_numbers: List[str] = Field(default_factory=list)
    pet_fee_info: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_lodge_info() -> str:
    return (
        "Extract the single pet-friendly lodge at the South Rim that is located within Grand Canyon National Park (inside park boundaries) from the answer.\n"
        "Return the following fields:\n"
        "1. lodge_name: The exact lodge name.\n"
        "2. official_website_url: A direct link to the lodge's official webpage or official online reservation page. Prefer authorized concessionaire domains (e.g., grandcanyonlodges.com by Xanterra) over third-party sites; NPS pages are acceptable if they directly represent official information.\n"
        "3. policy_reference_url: A reference URL from an official source explicitly confirming the lodge's pet-friendly policy (NPS or authorized concessionaire like grandcanyonlodges.com). Do not use third-party travel blogs or aggregators.\n"
        "4. reservation_phone_numbers: At least one official reservation or contact phone number for the lodge, as presented in the answer; include all numbers mentioned for reservations or lodging inquiries.\n"
        "5. pet_fee_info: A concise statement from the answer indicating whether there is an additional pet fee and any brief details (e.g., 'Yes, $100 per stay' or 'No additional fee').\n"
        "If any field is not explicitly present in the answer, set it to null (or empty array for reservation_phone_numbers).\n"
        "Extract only one lodge (the pet-friendly South Rim lodge within the park)."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_pet_friendly_lodge(
    evaluator: Evaluator,
    extracted: LodgeInfo,
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """

    # Root critical node for the rubric (under evaluator.root)
    pet_info_root = evaluator.add_parallel(
        id="Pet_Friendly_Lodge_Information",
        desc="Complete and accurate information about pet-friendly lodging at Grand Canyon National Park South Rim",
        parent=evaluator.root,
        critical=True,
    )

    # ----------------------- Lodge Identification ------------------------- #
    lodge_ident_node = evaluator.add_parallel(
        id="Lodge_Identification",
        desc="Correct identification of a lodge at South Rim that officially allows pets",
        parent=pet_info_root,
        critical=True,
    )

    # Lodge_Name (existence check)
    lodge_name_exists = bool(extracted.lodge_name and extracted.lodge_name.strip())
    evaluator.add_custom_node(
        result=lodge_name_exists,
        id="Lodge_Name",
        desc="The name of a lodge located at Grand Canyon South Rim that officially allows pets is provided",
        parent=lodge_ident_node,
        critical=True,
    )

    # In_Park_Location (verify by URL when available)
    in_park_node = evaluator.add_leaf(
        id="In_Park_Location",
        desc="Verification that the identified lodge is located within Grand Canyon National Park boundaries at the South Rim",
        parent=lodge_ident_node,
        critical=True,
    )
    # Build sources for location verification
    location_sources: List[str] = []
    if extracted.official_website_url:
        location_sources.append(extracted.official_website_url)
    if extracted.policy_reference_url and extracted.policy_reference_url not in location_sources:
        location_sources.append(extracted.policy_reference_url)

    if location_sources and lodge_name_exists:
        location_claim = (
            f"The lodge '{extracted.lodge_name}' is located within Grand Canyon National Park at the South Rim (inside park boundaries)."
        )
        await evaluator.verify(
            claim=location_claim,
            node=in_park_node,
            sources=location_sources,
            additional_instruction=(
                "Confirm that the lodge is inside Grand Canyon National Park boundaries and specifically at the South Rim area. "
                "Accept synonyms such as 'Grand Canyon Village' for South Rim. "
                "Do not accept locations outside park boundaries or on the West Rim (Hualapai lands)."
            ),
        )
    else:
        # Fail when no sources or no lodge name to verify
        in_park_node.score = 0.0
        in_park_node.status = "failed"

    # Reference_URL (verify pet-friendly policy from official source)
    ref_url_node = evaluator.add_leaf(
        id="Reference_URL",
        desc="A reference URL from an official source (NPS or authorized concessionaire website) confirming the lodge's pet-friendly policy",
        parent=lodge_ident_node,
        critical=True,
    )
    if extracted.policy_reference_url and lodge_name_exists:
        pet_policy_claim = (
            f"The official source confirms that the lodge '{extracted.lodge_name}' allows pets (pet-friendly policy present)."
        )
        await evaluator.verify(
            claim=pet_policy_claim,
            node=ref_url_node,
            sources=extracted.policy_reference_url,
            additional_instruction=(
                "Verify that the page belongs to the National Park Service (nps.gov) or an authorized concessionaire "
                "(e.g., Xanterra's grandcanyonlodges.com) and that it explicitly confirms pets are allowed at the lodge."
            ),
        )
    else:
        ref_url_node.score = 0.0
        ref_url_node.status = "failed"

    # ------------------- Reservation Contact Details ---------------------- #
    reservation_node = evaluator.add_parallel(
        id="Reservation_Contact_Details",
        desc="Official contact information for making reservations at the identified lodge",
        parent=pet_info_root,
        critical=True,
    )

    # Official_Website_Link (verify the link is official lodge or reservation page)
    official_link_node = evaluator.add_leaf(
        id="Official_Website_Link",
        desc="A direct link to the lodge's official webpage or online reservation page is provided",
        parent=reservation_node,
        critical=True,
    )
    if extracted.official_website_url and lodge_name_exists:
        official_link_claim = (
            f"This URL is the official webpage or official online reservation page for the lodge '{extracted.lodge_name}' at Grand Canyon South Rim."
        )
        await evaluator.verify(
            claim=official_link_claim,
            node=official_link_node,
            sources=extracted.official_website_url,
            additional_instruction=(
                "Confirm that the URL is an official source for the lodge (e.g., grandcanyonlodges.com or xanterra.com). "
                "Third-party travel aggregators or blogs are not considered official. "
                "The page should clearly be for the specified lodge or its official booking."
            ),
        )
    else:
        official_link_node.score = 0.0
        official_link_node.status = "failed"

    # Phone_Number (verify at least one official reservation phone number using official sources)
    phone_node = evaluator.add_leaf(
        id="Phone_Number",
        desc="At least one official reservation phone number for the lodge is provided",
        parent=reservation_node,
        critical=True,
    )
    primary_phone: Optional[str] = None
    for p in extracted.reservation_phone_numbers:
        if p and p.strip():
            primary_phone = p.strip()
            break

    if primary_phone and (extracted.official_website_url or extracted.policy_reference_url) and lodge_name_exists:
        phone_sources: List[str] = []
        if extracted.official_website_url:
            phone_sources.append(extracted.official_website_url)
        if extracted.policy_reference_url and extracted.policy_reference_url not in phone_sources:
            phone_sources.append(extracted.policy_reference_url)

        phone_claim = (
            f"The official page lists the reservation/contact phone number '{primary_phone}' for the lodge '{extracted.lodge_name}'."
        )
        await evaluator.verify(
            claim=phone_claim,
            node=phone_node,
            sources=phone_sources,
            additional_instruction=(
                "Check that the specified phone number appears on the official lodge webpage or official reservation page "
                "and is clearly indicated for reservations or lodging inquiries."
            ),
        )
    else:
        phone_node.score = 0.0
        phone_node.status = "failed"

    # ----------------------- Pet Fee Information -------------------------- #
    pet_fee_node = evaluator.add_leaf(
        id="Pet_Fee_Information",
        desc="Information about whether pets incur an additional fee at the identified lodge",
        parent=pet_info_root,
        critical=True,
    )
    if extracted.pet_fee_info and extracted.pet_fee_info.strip() and (extracted.policy_reference_url or extracted.official_website_url):
        fee_sources: List[str] = []
        if extracted.policy_reference_url:
            fee_sources.append(extracted.policy_reference_url)
        if extracted.official_website_url and extracted.official_website_url not in fee_sources:
            fee_sources.append(extracted.official_website_url)

        fee_claim = (
            f"The official policy indicates: {extracted.pet_fee_info.strip()} (regarding whether an additional pet fee is required)."
        )
        await evaluator.verify(
            claim=fee_claim,
            node=pet_fee_node,
            sources=fee_sources,
            additional_instruction=(
                "Focus on whether the lodge charges an additional fee for pets. The exact amount is helpful but not required "
                "as long as the presence or absence of a fee is clearly confirmed by the official source."
            ),
        )
    else:
        pet_fee_node.score = 0.0
        pet_fee_node.status = "failed"


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Grand Canyon pet-friendly lodge at South Rim task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation as parallel (rubric main node is parallel)
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

    # Extract lodge info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_lodge_info(),
        template_class=LodgeInfo,
        extraction_name="pet_friendly_lodge_info",
    )

    # Build tree and run verifications
    await verify_pet_friendly_lodge(evaluator, extracted_info)

    # Return structured summary
    return evaluator.get_summary()