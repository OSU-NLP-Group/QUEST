import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_totality_nps_park"
TASK_DESCRIPTION = (
    "Based on information available as of April 8, 2024, identify the U.S. National Park (proper NPS designation, not National Monument, River, "
    "or other classification) that meets ALL of the following criteria:\n\n"
    "1. The park was located within the path of totality for the April 8, 2024 total solar eclipse\n"
    "2. The park experienced a totality duration of at least 3 minutes and 30 seconds during the eclipse\n"
    "3. The park does not charge an entrance fee for entry\n"
    "4. The park is located in a U.S. state that had at least one city served by Breeze Airways with service beginning before April 8, 2024\n"
    "5. According to published sources, this park was specifically identified as one of only two U.S. National Parks (proper designation) that were in "
    "the path of totality for the April 8, 2024 eclipse\n\n"
    "Additionally, provide the following information:\n"
    "- The name of the National Park\n"
    "- The specific city in that state served by Breeze Airways\n"
    "- The date when Breeze Airways service to that city began\n"
    "- Whether the America the Beautiful Annual Pass provides entrance benefits at this park\n"
    "- The exact duration of totality (in minutes and seconds) at this park during the April 8, 2024 eclipse"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    """Structured extraction of the single park selection and all requested fields."""
    park_name: Optional[str] = None
    park_state: Optional[str] = None
    designation: Optional[str] = None  # e.g., "National Park", "National Park & Preserve"
    totality_duration: Optional[str] = None  # e.g., "3 minutes 35 seconds" or "3:35"
    entrance_fee_statement: Optional[str] = None  # e.g., "No entrance fee" or description
    breeze_city: Optional[str] = None  # city in the same state
    breeze_service_start_date: Optional[str] = None  # e.g., "June 15, 2023"
    america_the_beautiful_pass_benefits: Optional[str] = None  # e.g., "Yes, provides entrance benefits" or "No entrance fee"

    # Source URLs explicitly cited in the answer for verification of each constraint/detail
    sources_designation: List[str] = Field(default_factory=list)
    sources_totality_path: List[str] = Field(default_factory=list)
    sources_totality_duration: List[str] = Field(default_factory=list)
    sources_entrance_fee: List[str] = Field(default_factory=list)
    sources_breeze_service: List[str] = Field(default_factory=list)  # service announcement or airport press release
    sources_two_parks_identified: List[str] = Field(default_factory=list)  # articles stating only two NPs in totality path


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return (
        "Extract the single U.S. National Park (proper NPS designation) the answer claims satisfies ALL constraints, "
        "and return the following fields exactly as presented in the answer text:\n\n"
        "Required fields:\n"
        "1. park_name: The full official name of the National Park selected.\n"
        "2. park_state: The U.S. state where the park is located (e.g., 'Arkansas', 'Ohio').\n"
        "3. designation: The official NPS unit designation for the selected park (e.g., 'National Park', 'National Park & Preserve').\n"
        "4. totality_duration: The exact duration of totality at the park during the April 8, 2024 eclipse, in minutes and seconds (e.g., '3 minutes 35 seconds' or '3:35').\n"
        "5. entrance_fee_statement: A statement indicating whether the park charges an entrance fee (e.g., 'No entrance fee').\n"
        "6. breeze_city: A city in the same state that the answer cites as being served by Breeze Airways.\n"
        "7. breeze_service_start_date: The date Breeze Airways service to that city began, as stated in the answer (e.g., 'June 15, 2023').\n"
        "8. america_the_beautiful_pass_benefits: Whether the America the Beautiful Annual Pass provides entrance benefits at this park.\n\n"
        "Source URL arrays (extract only actual URLs explicitly cited in the answer text; if none are present for a category, return an empty array):\n"
        "9. sources_designation: URLs supporting the park's official NPS designation as a 'National Park' (not monument/river/etc.).\n"
        "10. sources_totality_path: URLs supporting that the park was located within the April 8, 2024 path of totality.\n"
        "11. sources_totality_duration: URLs supporting the stated totality duration at the park.\n"
        "12. sources_entrance_fee: URLs supporting the 'no entrance fee' claim.\n"
        "13. sources_breeze_service: URLs supporting Breeze Airways service to the cited city and the start date.\n"
        "14. sources_two_parks_identified: URLs where published sources identify the selected park as one of only two properly-designated U.S. National Parks in the path of totality for April 8, 2024.\n\n"
        "Important rules:\n"
        "- If the answer lists multiple parks, extract the first park that the answer claims meets all constraints.\n"
        "- For any field not present in the answer, return null (for strings) or an empty array (for sources).\n"
        "- For URLs presented as markdown links, extract the actual target URL.\n"
        "- Do not invent or infer any values; only extract what appears in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_minutes_and_seconds(duration_text: Optional[str]) -> bool:
    """Check if the provided duration string includes both minutes and seconds in a reasonable format."""
    if not duration_text:
        return False
    s = duration_text.strip().lower()

    # Patterns: "3:35", "03:35", "3 min 35 sec", "3 minutes 35 seconds", "3 minutes, 35 seconds", "3m 35s"
    colon_pat = re.compile(r"^\s*\d{1,2}\s*:\s*\d{1,2}\s*$")
    words_pat = re.compile(
        r"(?P<min>\d{1,2})\s*(minutes?|mins?|m)\b.*?(?P<sec>\d{1,2})\s*(seconds?|secs?|s)\b"
    )
    short_pat = re.compile(r"(?P<min>\d{1,2})\s*m\s*(?P<sec>\d{1,2})\s*s")

    return bool(colon_pat.match(s) or words_pat.search(s) or short_pat.search(s))


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: ParkInfo) -> None:
    """
    Build the verification tree from the rubric and perform verifications.
    """

    # --------------------------------------------------------------------- #
    # Node 1: Park_Meets_All_Constraints (parallel, critical)               #
    # --------------------------------------------------------------------- #
    constraints_node = evaluator.add_parallel(
        id="Park_Meets_All_Constraints",
        desc="The identified park satisfies every eligibility constraint in the prompt.",
        parent=root_node,
        critical=True
    )

    # 1) Proper NPS National Park designation (leaf, critical)
    proper_designation_leaf = evaluator.add_leaf(
        id="Proper_NPS_National_Park_Designation",
        desc="Park is an NPS unit designated specifically as a U.S. National Park (not a National Monument, National River, etc.).",
        parent=constraints_node,
        critical=True
    )
    designation_claim = (
        f"{extracted.park_name or 'The selected park'} is designated by the National Park Service as a U.S. National Park "
        f"(proper designation; variations like 'National Park & Preserve' count as National Park), not a National Monument, National River, or other unit type."
    )
    await evaluator.verify(
        claim=designation_claim,
        node=proper_designation_leaf,
        sources=extracted.sources_designation,
        additional_instruction=(
            "Use official NPS or authoritative sources. Confirm that the unit type is 'National Park' (including 'National Park & Preserve') "
            "and not any other designation like National Monument, National River, National Historical Park, etc."
        ),
    )

    # 2) In April 8, 2024 totality path (leaf, critical)
    totality_path_leaf = evaluator.add_leaf(
        id="In_April_8_2024_Totality_Path",
        desc="Park was located within the path of totality for the April 8, 2024 total solar eclipse.",
        parent=constraints_node,
        critical=True
    )
    totality_path_claim = (
        f"{extracted.park_name or 'The selected park'} was located within the path of totality for the April 8, 2024 total solar eclipse."
    )
    await evaluator.verify(
        claim=totality_path_claim,
        node=totality_path_leaf,
        sources=extracted.sources_totality_path,
        additional_instruction=(
            "Confirm the park's location intersects the official path of totality for April 8, 2024. "
            "Authoritative eclipse maps or credible sources (NASA, state eclipse sites, etc.) are acceptable."
        ),
    )

    # 3) Totality duration minimum met (leaf, critical)
    totality_min_leaf = evaluator.add_leaf(
        id="Totality_Duration_Minimum_Met",
        desc="Park experienced totality duration of at least 3 minutes and 30 seconds during the April 8, 2024 eclipse.",
        parent=constraints_node,
        critical=True
    )
    totality_min_claim = (
        f"On April 8, 2024, the totality at {extracted.park_name or 'the selected park'} lasted at least 3 minutes and 30 seconds."
    )
    await evaluator.verify(
        claim=totality_min_claim,
        node=totality_min_leaf,
        sources=extracted.sources_totality_duration,
        additional_instruction=(
            "Use the provided sources to confirm the duration meets or exceeds 3 minutes and 30 seconds. "
            "Accept reasonable rounding."
        ),
    )

    # 4) No entrance fee (leaf, critical)
    no_fee_leaf = evaluator.add_leaf(
        id="No_Entrance_Fee",
        desc="Park does not charge an entrance fee for entry.",
        parent=constraints_node,
        critical=True
    )
    no_fee_claim = (
        f"{extracted.park_name or 'The selected park'} does not charge an entrance fee for entry."
    )
    await evaluator.verify(
        claim=no_fee_claim,
        node=no_fee_leaf,
        sources=extracted.sources_entrance_fee,
        additional_instruction=(
            "Confirm via official NPS fee page or credible sources that the park has no entrance fee. "
            "If there are other fees (e.g., camping), they do not count as entrance fees."
        ),
    )

    # 5) Breeze service in state before eclipse (leaf, critical)
    breeze_leaf = evaluator.add_leaf(
        id="Breeze_Service_In_State_Pre_Eclipse",
        desc="Park is located in a U.S. state that had at least one city served by Breeze Airways with service beginning before April 8, 2024.",
        parent=constraints_node,
        critical=True
    )
    breeze_city = extracted.breeze_city or "a cited city"
    breeze_state = extracted.park_state or "the park's state"
    breeze_date = extracted.breeze_service_start_date or "the cited start date"
    breeze_claim = (
        f"Breeze Airways began service to {breeze_city}, {breeze_state} on {breeze_date}, "
        f"which is before April 8, 2024."
    )
    await evaluator.verify(
        claim=breeze_claim,
        node=breeze_leaf,
        sources=extracted.sources_breeze_service,
        additional_instruction=(
            "Verify the service start date for the cited city in the same state as the park and confirm it is earlier than 2024-04-08. "
            "Official Breeze press releases, airport announcements, or credible news sources are acceptable."
        ),
    )

    # 6) Identified as one of only two National Parks in totality path (leaf, critical)
    only_two_leaf = evaluator.add_leaf(
        id="Identified_As_One_Of_Only_Two_National_Parks_In_Totality",
        desc="Published sources specifically identified this park as one of only two properly-designated U.S. National Parks that were in the April 8, 2024 path of totality.",
        parent=constraints_node,
        critical=True
    )
    only_two_claim = (
        f"Published sources specifically identify {extracted.park_name or 'the selected park'} as one of only two properly-designated U.S. National Parks "
        f"in the path of totality for the April 8, 2024 eclipse."
    )
    await evaluator.verify(
        claim=only_two_claim,
        node=only_two_leaf,
        sources=extracted.sources_two_parks_identified,
        additional_instruction=(
            "Confirm that the source explicitly states that only two U.S. National Parks (proper designation) were in the path of totality and "
            "that the selected park is one of them."
        ),
    )

    # --------------------------------------------------------------------- #
    # Node 2: Required_Output_Details_Provided (parallel, critical)         #
    # --------------------------------------------------------------------- #
    details_node = evaluator.add_parallel(
        id="Required_Output_Details_Provided",
        desc="Answer includes all additional requested information fields.",
        parent=root_node,
        critical=True
    )

    # Provide_Park_Name (leaf-like via custom boolean check)
    evaluator.add_custom_node(
        result=bool(extracted.park_name and extracted.park_name.strip()),
        id="Provide_Park_Name",
        desc="Answer provides the name of the National Park.",
        parent=details_node,
        critical=True
    )

    # Provide_Breeze_Destination_City
    evaluator.add_custom_node(
        result=bool(extracted.breeze_city and extracted.breeze_city.strip()),
        id="Provide_Breeze_Destination_City",
        desc="Answer provides a specific city in the park’s state that is/was served by Breeze Airways.",
        parent=details_node,
        critical=True
    )

    # Provide_Breeze_Service_Start_Date
    evaluator.add_custom_node(
        result=bool(extracted.breeze_service_start_date and extracted.breeze_service_start_date.strip()),
        id="Provide_Breeze_Service_Start_Date",
        desc="Answer provides the date when Breeze Airways service to that city began.",
        parent=details_node,
        critical=True
    )

    # Provide_America_The_Beautiful_Pass_Benefit_Status
    evaluator.add_custom_node(
        result=bool(extracted.america_the_beautiful_pass_benefits and extracted.america_the_beautiful_pass_benefits.strip()),
        id="Provide_America_The_Beautiful_Pass_Benefit_Status",
        desc="Answer states whether the America the Beautiful Annual Pass provides entrance benefits at this park.",
        parent=details_node,
        critical=True
    )

    # Provide_Exact_Totality_Duration (ensure minutes and seconds are present)
    evaluator.add_custom_node(
        result=has_minutes_and_seconds(extracted.totality_duration),
        id="Provide_Exact_Totality_Duration",
        desc="Answer states the exact totality duration at the park during the April 8, 2024 eclipse in minutes and seconds.",
        parent=details_node,
        critical=True
    )


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
    Evaluate an agent's answer for the eclipse totality National Park task.
    Returns a standardized summary dictionary including the verification tree.
    """

    # Initialize evaluator with a sequential, critical root as per rubric
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify the single U.S. National Park that satisfies all stated constraints and provide all requested associated details.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured info from the answer
    extracted_info: ParkInfo = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkInfo,
        extraction_name="park_selection_and_details"
    )

    # Add the raw extraction to custom info for transparency
    evaluator.add_custom_info(
        info=extracted_info.dict(),
        info_type="extraction",
        info_name="parsed_park_info"
    )

    # Build and execute verification tree according to rubric
    await build_verification_tree(evaluator, root, extracted_info)

    # Return unified summary with final aggregated score and tree
    return evaluator.get_summary()