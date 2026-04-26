import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "state_cdc_schedule_rejection_2026"
TASK_DESCRIPTION = (
    "On January 5, 2026, the CDC released an updated childhood immunization schedule that reduced the number of "
    "universally recommended vaccines from 17 to 11. How many states, along with Washington, DC, rejected this new "
    "CDC schedule, and which organization's vaccine guidance are these states following instead? Additionally, when "
    "did this alternative organization release its 2026 childhood immunization schedule?"
)

# Expected facts captured from the rubric for verification phrasing
EXPECTED_CDC_UPDATE_DATE = "January 5, 2026"
EXPECTED_CDC_UNIVERSAL_BEFORE = "17"
EXPECTED_CDC_UNIVERSAL_AFTER = "11"
EXPECTED_REJECTING_STATES_NUMBER = "23"
EXPECTED_ALT_ORG_FULL = "American Academy of Pediatrics"
EXPECTED_ALT_ORG_ABBR = "AAP"
EXPECTED_AAP_SCHEDULE_RELEASE_DATE = "January 28, 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ScheduleRejectionExtraction(BaseModel):
    # CDC update context
    cdc_update_date: Optional[str] = None
    cdc_universal_count_before: Optional[str] = None
    cdc_universal_count_after: Optional[str] = None
    cdc_reference_urls: List[str] = Field(default_factory=list)

    # States rejection info
    rejecting_states_number: Optional[str] = None
    includes_washington_dc: Optional[bool] = None

    # Alternative guidance
    alternative_org_name: Optional[str] = None
    alt_schedule_release_date: Optional[str] = None
    states_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schedule_rejection() -> str:
    return """
    Extract the specific information explicitly stated in the answer about the CDC’s January 2026 childhood immunization schedule update and the states’ response.

    Required fields:
    1) cdc_update_date: The exact date (as text) when the CDC released the updated childhood immunization recommendations (e.g., "January 5, 2026"). If not stated, null.
    2) cdc_universal_count_before: The number of universally recommended childhood vaccines before the update (e.g., "17"). If not stated, null.
    3) cdc_universal_count_after: The number of universally recommended childhood vaccines after the update (e.g., "11"). If not stated, null.
    4) cdc_reference_urls: All URLs cited in the answer that support the CDC update details. If none are provided, return an empty list.

    5) rejecting_states_number: The number of U.S. states that rejected the new CDC schedule (as a string integer, e.g., "23"). If not stated, null.
    6) includes_washington_dc: A boolean indicating whether the answer explicitly includes Washington, DC among the rejecting jurisdictions (true/false). If not stated, null.

    7) alternative_org_name: The name of the alternative organization these states will follow (e.g., "American Academy of Pediatrics" or "AAP"). If not stated, null.
    8) alt_schedule_release_date: The date (as text) the alternative organization released its 2026 childhood immunization schedule (e.g., "January 28, 2026"). If not stated, null.
    9) states_reference_urls: All URLs cited in the answer that support the claims about how many states rejected the CDC schedule, that they follow the alternative organization, and the alternative schedule’s release date. If none are provided, return an empty list.

    Notes:
    - Only extract what is explicitly present in the answer.
    - For URL fields, include only valid, explicit URLs mentioned (from plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_cdc_context_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: ScheduleRejectionExtraction
) -> None:
    """
    Build and verify the subtree for CDC schedule update context.
    """
    cdc_ctx = evaluator.add_parallel(
        id="CDC_Schedule_Update_Context",
        desc="Verifies key details about the CDC's January 5, 2026 childhood immunization schedule update",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence for CDC context (critical)
    evaluator.add_custom_node(
        result=bool(extracted.cdc_reference_urls),
        id="Reference_URL_CDC_Context",
        desc="Provides URL reference supporting CDC schedule update details",
        parent=cdc_ctx,
        critical=True
    )

    # Verify CDC update date claim via cited CDC context URLs (critical)
    date_leaf = evaluator.add_leaf(
        id="CDC_Update_Date",
        desc="Confirms the CDC released updated childhood immunization recommendations on January 5, 2026",
        parent=cdc_ctx,
        critical=True
    )
    claim_date = f"The CDC released updated childhood immunization recommendations on {EXPECTED_CDC_UPDATE_DATE}."
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=extracted.cdc_reference_urls,
        additional_instruction=(
            "Check the CDC source(s) for the update date of the childhood immunization schedule; "
            "accept reasonable date format variations that correspond to January 5, 2026."
        )
    )

    # Verify vaccine count reduction claim via cited CDC context URLs (critical)
    reduction_leaf = evaluator.add_leaf(
        id="Vaccine_Count_Reduction",
        desc="Confirms the CDC reduced universally recommended childhood vaccines from 17 to 11",
        parent=cdc_ctx,
        critical=True
    )
    claim_reduction = (
        f"In its January 2026 update, the CDC reduced the number of universally recommended childhood vaccines "
        f"from {EXPECTED_CDC_UNIVERSAL_BEFORE} to {EXPECTED_CDC_UNIVERSAL_AFTER}."
    )
    await evaluator.verify(
        claim=claim_reduction,
        node=reduction_leaf,
        sources=extracted.cdc_reference_urls,
        additional_instruction=(
            "Verify on the CDC source(s) that the number of universally recommended childhood vaccines "
            "was reduced from 17 to 11 in the 2026 update. Allow minor wording differences."
        )
    )


async def build_states_rejection_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: ScheduleRejectionExtraction
) -> None:
    """
    Build and verify the subtree for states rejecting the CDC schedule and following AAP.
    """
    states_node = evaluator.add_parallel(
        id="States_Rejection_Information",
        desc="Identifies states that rejected the CDC schedule and their alternative guidance",
        parent=parent_node,
        critical=True
    )

    # Number of rejecting states + DC (critical)
    num_leaf = evaluator.add_leaf(
        id="Number_of_Rejecting_States",
        desc="Confirms that 23 states plus Washington, DC rejected the new CDC vaccination schedule",
        parent=states_node,
        critical=True
    )
    claim_states = (
        f"{EXPECTED_REJECTING_STATES_NUMBER} U.S. states plus Washington, DC rejected the new CDC childhood "
        f"vaccination schedule released in January 2026."
    )
    await evaluator.verify(
        claim=claim_states,
        node=num_leaf,
        sources=extracted.states_reference_urls,
        additional_instruction=(
            "Verify that sources explicitly indicate that twenty-three (23) U.S. states, plus Washington, D.C., "
            "rejected the CDC's revised childhood vaccination schedule. Accept numeric or spelled-out variants."
        )
    )

    # Alternative guidance identification block (critical)
    alt_node = evaluator.add_parallel(
        id="Alternative_Guidance_Identification",
        desc="Identifies that rejecting states are following American Academy of Pediatrics (AAP) guidance instead",
        parent=states_node,
        critical=True
    )

    # Reference URL existence for states/AAP claims (critical)
    evaluator.add_custom_node(
        result=bool(extracted.states_reference_urls),
        id="Reference_URL_States_Response",
        desc="Provides URL reference supporting information about states rejecting CDC schedule and following AAP",
        parent=alt_node,
        critical=True
    )

    # AAP as alternative guidance (critical)
    aap_alt_leaf = evaluator.add_leaf(
        id="AAP_as_Alternative_Source",
        desc="Confirms states plan to follow vaccine guidance from the AAP rather than the CDC's revised schedule",
        parent=alt_node,
        critical=True
    )
    claim_alt_org = (
        f"These rejecting states are following vaccine guidance from the {EXPECTED_ALT_ORG_FULL} "
        f"({EXPECTED_ALT_ORG_ABBR}) instead of the CDC's revised schedule."
    )
    await evaluator.verify(
        claim=claim_alt_org,
        node=aap_alt_leaf,
        sources=extracted.states_reference_urls,
        additional_instruction=(
            "Confirm that the sources explicitly state the rejecting states will follow AAP guidance (American Academy of Pediatrics), "
            "also recognizing the abbreviation 'AAP'."
        )
    )

    # AAP 2026 schedule release date (critical)
    aap_date_leaf = evaluator.add_leaf(
        id="AAP_Schedule_Release_Date",
        desc="Confirms the AAP released its 2026 childhood immunization schedule on January 28, 2026",
        parent=alt_node,
        critical=True
    )
    claim_aap_date = (
        f"The {EXPECTED_ALT_ORG_FULL} ({EXPECTED_ALT_ORG_ABBR}) released its 2026 childhood immunization schedule on "
        f"{EXPECTED_AAP_SCHEDULE_RELEASE_DATE}."
    )
    await evaluator.verify(
        claim=claim_aap_date,
        node=aap_date_leaf,
        sources=extracted.states_reference_urls,
        additional_instruction=(
            "Verify on the provided sources (including any AAP page or announcements if cited) that the 2026 childhood "
            "immunization schedule was released on January 28, 2026. Allow standard date format variants like 'Jan 28, 2026'."
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
    Build the verification tree and run the evaluation for the CDC schedule rejection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # As specified by the rubric
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_schedule_rejection(),
        template_class=ScheduleRejectionExtraction,
        extraction_name="schedule_rejection_extraction"
    )

    # Optional: record expected facts for transparency
    evaluator.add_ground_truth({
        "expected_cdc_update_date": EXPECTED_CDC_UPDATE_DATE,
        "expected_cdc_universal_counts": {
            "before": EXPECTED_CDC_UNIVERSAL_BEFORE,
            "after": EXPECTED_CDC_UNIVERSAL_AFTER
        },
        "expected_rejecting_states_plus_dc": f"{EXPECTED_REJECTING_STATES_NUMBER} + DC",
        "expected_alternative_org": f"{EXPECTED_ALT_ORG_FULL} ({EXPECTED_ALT_ORG_ABBR})",
        "expected_aap_schedule_release_date": EXPECTED_AAP_SCHEDULE_RELEASE_DATE
    })

    # Build subtrees according to rubric
    await build_cdc_context_subtree(evaluator, root, extracted)
    await build_states_rejection_subtree(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()