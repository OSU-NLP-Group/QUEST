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
TASK_ID = "la_tv_show_taping_13yo"
TASK_DESCRIPTION = (
    "A family is visiting Los Angeles and wants to attend a live TV show taping. They have a 13-year-old child who "
    "will be attending with them. Identify a TV show that films in Los Angeles and allows 13-year-old audience members "
    "to attend. Provide the following information: (1) The name of the TV show, (2) The complete street address of the "
    "studio where it films, and (3) A phone number for ticket reservations or information."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ShowAttendanceExtraction(BaseModel):
    show_name: Optional[str] = None
    studio_address: Optional[str] = None
    phone_number: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_show_attendance() -> str:
    return (
        "You will extract a single TV show suitable for attending a live taping with a 13-year-old in Los Angeles County.\n"
        "- show_name: The exact name of the TV show identified in the answer.\n"
        "- studio_address: The complete street address (include building number/name if applicable, street, city, and state).\n"
        "- phone_number: A phone number listed for ticket reservations or audience information for this show. If multiple "
        "numbers are present, choose the one most clearly labeled for tickets/audience info. Keep original formatting from the answer.\n"
        "- reference_urls: All URLs in the answer that support the show's age policy, filming location, studio address, or ticket/contact information. "
        "Include every relevant URL mentioned in the answer. If none are present, return an empty array.\n\n"
        "If the answer mentions multiple shows, extract only the one that the answer positions as meeting the requirements (Los Angeles filming and "
        "allowing a 13-year-old). If you cannot determine which one, select the first that plausibly meets both requirements. "
        "If a required field is not present in the answer, set it to null (or [] for the URL list)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_show_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: ShowAttendanceExtraction
) -> None:
    """
    Build and verify the Show Identification subtree:
    - Show_Name_Provided (existence)
    - Eligibility_Criteria (Age + Location) using provided URLs
    - Show_Reference_URL (existence of at least one URL)
    """
    show_ident_node = evaluator.add_parallel(
        id="Show_Identification",
        desc="A specific TV show is identified and meets the stated eligibility and verification requirements",
        parent=parent_node,
        critical=True,
    )

    # Show name provided (existence)
    name_exists = bool(extracted.show_name and extracted.show_name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="Show_Name_Provided",
        desc="The name of the TV show is provided",
        parent=show_ident_node,
        critical=True
    )

    # At least one reference URL provided (existence)
    urls_present = bool(extracted.reference_urls and len(extracted.reference_urls) > 0)
    evaluator.add_custom_node(
        result=urls_present,
        id="Show_Reference_URL",
        desc="At least one verifiable reference URL is provided that confirms the show's age requirement and filming location",
        parent=show_ident_node,
        critical=True
    )

    # Eligibility criteria (Age + Location)
    eligibility_node = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="The selected show satisfies both the age and location requirements",
        parent=show_ident_node,
        critical=True
    )

    # Age requirement: 13-year-old can attend
    age_leaf = evaluator.add_leaf(
        id="Age_Requirement_Met",
        desc="The show's minimum age requirement is 13 years or lower (so a 13-year-old can attend)",
        parent=eligibility_node,
        critical=True
    )

    show_name_for_claim = extracted.show_name or ""
    age_claim = (
        f"A 13-year-old can attend a live taping of '{show_name_for_claim}'. "
        f"This means the minimum audience age requirement is 13 or lower (e.g., '13+', 'ages 12+ also OK'), "
        f"and any guardian/accompaniment conditions are acceptable."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm from the provided URLs that the show allows 13-year-old audience members. "
            "Accept phrasing like 'minimum age 13', '13+', or any minimum age ≤ 13. "
            "If the minimum age is 14 or older, this should fail."
        )
    )

    # Location requirement: Films in Los Angeles County, California
    location_leaf = evaluator.add_leaf(
        id="Location_Requirement_Met",
        desc="The show films in Los Angeles County, California",
        parent=eligibility_node,
        critical=True
    )

    location_claim = (
        f"'{show_name_for_claim}' films in Los Angeles County, California."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Verify that the filming location is in Los Angeles County. "
            "Accept cities/areas within Los Angeles County such as Los Angeles, Hollywood, Burbank, Glendale, Culver City, "
            "Universal City, Studio City, Santa Monica, West Hollywood, Inglewood, Manhattan Beach, etc. "
            "The evidence should imply or state that the show tapes/films in one of these LA County locations."
        )
    )


async def verify_address_and_phone(
    evaluator: Evaluator,
    parent_node,
    extracted: ShowAttendanceExtraction
) -> None:
    """
    Add and verify:
    - Studio_Address_Provided (complete address and corresponds to the filming studio)
    - Contact_Phone_Provided (phone for tickets or audience information)
    """
    # Studio address provided and correct
    addr_leaf = evaluator.add_leaf(
        id="Studio_Address_Provided",
        desc="A complete street address for the studio is provided, including building name or number, street name, city, and state",
        parent=parent_node,
        critical=True
    )

    address_text = extracted.studio_address or ""
    show_name_for_claim = extracted.show_name or ""
    addr_claim = (
        f"The studio address for '{show_name_for_claim}' is '{address_text}'. "
        f"This is a complete street address (number/name, street, city, state) for the studio where the show films."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm from the provided URLs that the address is indeed the filming studio location for the show. "
            "Check that the address string is complete (includes number/name, street, city, and state). "
            "Addresses like 'Burbank, CA' without street or number are incomplete and should fail."
        )
    )

    # Contact phone provided and relevant for ticket reservations/info
    phone_leaf = evaluator.add_leaf(
        id="Contact_Phone_Provided",
        desc="A valid/working phone number is provided that serves as a contact for ticket reservations or ticket information for the identified show",
        parent=parent_node,
        critical=True
    )

    phone_text = extracted.phone_number or ""
    phone_claim = (
        f"The phone number '{phone_text}' is provided as a contact for ticket reservations or audience information "
        f"for attending '{show_name_for_claim}' live tapings."
    )
    await evaluator.verify(
        claim=phone_claim,
        node=phone_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "From the provided URLs, verify that this phone number is presented as a contact for audience tickets, "
            "ticket reservations, or taping information for the specified show. "
            "General corporate phone numbers without any connection to tickets/audience info should fail."
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
    Evaluate an answer for the Los Angeles live TV show taping task with a 13-year-old attendee.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; main logic under a critical sequential child node
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
    extraction: ShowAttendanceExtraction = await evaluator.extract(
        prompt=prompt_extract_show_attendance(),
        template_class=ShowAttendanceExtraction,
        extraction_name="show_attendance_extraction"
    )

    # Build the main critical sequential node matching the rubric root
    main_node = evaluator.add_sequential(
        id="TV_Show_Attendance_Information",
        desc="Complete information for attending a Los Angeles County TV show taping with a 13-year-old",
        parent=root,
        critical=True
    )

    # 1) Show identification & eligibility subtree
    await verify_show_identification(evaluator, main_node, extraction)

    # 2) Studio address leaf and 3) Phone contact leaf
    await verify_address_and_phone(evaluator, main_node, extraction)

    # Return evaluation summary
    return evaluator.get_summary()