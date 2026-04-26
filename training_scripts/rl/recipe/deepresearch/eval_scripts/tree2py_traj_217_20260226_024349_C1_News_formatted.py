import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "defense_bill_dec2025"
TASK_DESCRIPTION = (
    "What is the official name of the defense bill that was passed by the U.S. Senate in December 2025, "
    "which includes a provision restricting a percentage of the Defense Secretary's travel budget until "
    "specific video footage is provided to Congress?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BillInfo(BaseModel):
    """
    Structured extraction for the defense bill identification task.
    """
    official_name: Optional[str] = None
    bill_type_claim: Optional[str] = None
    senate_passage_time: Optional[str] = None
    travel_budget_restriction_percent: Optional[str] = None
    video_footage_description: Optional[str] = None

    # URLs explicitly cited in the answer
    bill_sources: List[str] = Field(default_factory=list)
    provision_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bill_info() -> str:
    return """
    Extract the information about the defense bill mentioned in the answer.

    Required fields:
    1) official_name: The official name of the defense bill (e.g., "National Defense Authorization Act for Fiscal Year 2026").
    2) bill_type_claim: How the answer characterizes the bill's type (e.g., "defense authorization bill", "defense policy bill").
    3) senate_passage_time: The time statement related to Senate passage (e.g., "December 2025" or a specific date).
    4) travel_budget_restriction_percent: The percentage of the Defense Secretary's travel budget that is restricted (e.g., "15%"). Extract exactly as stated in the answer.
    5) video_footage_description: The description of the specific video footage that must be provided to Congress (e.g., "unredacted body camera footage of X").
    6) bill_sources: All URLs the answer cites that support the bill identification (news articles, official pages, etc.).
    7) provision_sources: All URLs the answer cites that support the budget restriction and video footage condition provision.

    Rules:
    - Extract only what is explicitly stated in the answer text.
    - Return null for any missing field.
    - For URLs, extract only valid, complete URLs (include protocol).
    - If the answer mentions the source without a URL, do not fabricate a URL; leave the corresponding list empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combined_sources(info: BillInfo) -> List[str]:
    """
    Combine and de-duplicate bill and provision sources, preserving order.
    """
    seen = set()
    out: List[str] = []
    for u in (info.bill_sources or []) + (info.provision_sources or []):
        if isinstance(u, str) and u.strip() and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: BillInfo) -> None:
    """
    Build the verification tree and run all necessary verifications.
    """

    # Top-level critical node representing the rubric root
    root_crit = evaluator.add_parallel(
        id="Defense_Bill_Identification",
        desc=("Correctly identify the defense bill passed by the Senate in December 2025 that contains provisions "
              "restricting the Defense Secretary's travel budget pending video footage delivery to Congress"),
        parent=evaluator.root,
        critical=True
    )

    # 0) Official bill name presence (critical existence gate)
    name_present_node = evaluator.add_custom_node(
        result=bool(extracted.official_name and extracted.official_name.strip()),
        id="Bill_Official_Name_Provided",
        desc="The answer provides an official bill name",
        parent=root_crit,
        critical=True
    )

    # 0.1) Official bill name is supported by sources (critical)
    name_verified_node = evaluator.add_leaf(
        id="Bill_Official_Name_Verified",
        desc="The official bill name is correctly supported by cited sources",
        parent=root_crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the bill is '{extracted.official_name or ''}'.",
        node=name_verified_node,
        sources=extracted.bill_sources,
        additional_instruction=("Verify the exact official name of the bill on the provided webpages. "
                                "Minor stylistic variations (e.g., 'NDAA for FY2026' vs 'National Defense Authorization Act for Fiscal Year 2026') "
                                "should be accepted as equivalent if they clearly refer to the same bill.")
    )

    # 1) Bill type and timing (critical group)
    type_timing_node = evaluator.add_parallel(
        id="Bill_Type_And_Timing",
        desc="The identified bill must be a defense authorization or defense policy bill passed by the U.S. Senate in December 2025",
        parent=root_crit,
        critical=True
    )

    # 1.1) Bill type is defense authorization/policy (critical leaf)
    bill_type_leaf = evaluator.add_leaf(
        id="Bill_Is_Defense_Policy_Bill",
        desc="The bill is a defense authorization or defense policy bill",
        parent=type_timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The bill named '{extracted.official_name or ''}' is a defense authorization or defense policy bill."),
        node=bill_type_leaf,
        sources=extracted.bill_sources,
        additional_instruction=("Look for explicit descriptions indicating the bill is an annual defense authorization/policy bill "
                                "(e.g., references to the NDAA or defense policy legislation). Accept standard naming conventions.")
    )

    # 1.2) Senate passage in December 2025 (critical leaf)
    timing_leaf = evaluator.add_leaf(
        id="Bill_Senate_Passage_December_2025",
        desc="The bill was passed by the U.S. Senate in December 2025",
        parent=type_timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The bill named '{extracted.official_name or ''}' was passed by the U.S. Senate in December 2025."),
        node=timing_leaf,
        sources=combined_sources(extracted),
        additional_instruction=("Confirm the U.S. Senate passage took place in December 2025. "
                                "Use legislative records or credible reporting; article publication dates alone are insufficient unless they clearly state the Senate passage timing.")
    )

    # 2) Travel budget restriction (critical group)
    travel_restr_node = evaluator.add_parallel(
        id="Travel_Budget_Restriction",
        desc="The bill must include a provision that restricts a specific percentage of the Defense Secretary's travel budget",
        parent=root_crit,
        critical=True
    )

    # 2.0) Percentage presence (critical existence)
    percent_present_leaf = evaluator.add_custom_node(
        result=bool(extracted.travel_budget_restriction_percent and extracted.travel_budget_restriction_percent.strip()),
        id="Travel_Budget_Percent_Provided",
        desc="The answer specifies the percentage of the Defense Secretary's travel budget to be restricted",
        parent=travel_restr_node,
        critical=True
    )

    # 2.1) Percentage restriction supported by sources (critical)
    percent_supported_leaf = evaluator.add_leaf(
        id="Travel_Budget_Percent_Supported",
        desc="The bill includes a provision restricting the specified percentage of the Defense Secretary's travel budget",
        parent=travel_restr_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The bill named '{extracted.official_name or ''}' includes a provision that restricts "
               f"{extracted.travel_budget_restriction_percent or ''} of the Defense Secretary's travel budget."),
        node=percent_supported_leaf,
        sources=combined_sources(extracted),
        additional_instruction=("Confirm the text explicitly ties a specific percentage of the Defense Secretary's travel budget to a restriction/withholding. "
                                "The percentage must match what is claimed in the answer.")
    )

    # 3) Video footage condition (critical group)
    video_condition_node = evaluator.add_parallel(
        id="Video_Footage_Condition",
        desc="The travel budget restriction must be conditional upon the Defense Secretary providing specific video footage to Congress",
        parent=root_crit,
        critical=True
    )

    # 3.0) Video footage description presence (critical existence)
    footage_desc_present_leaf = evaluator.add_custom_node(
        result=bool(extracted.video_footage_description and extracted.video_footage_description.strip()),
        id="Video_Footage_Description_Provided",
        desc="The answer provides a description of the specific video footage to be provided to Congress",
        parent=video_condition_node,
        critical=True
    )

    # 3.1) Conditionality supported by sources (critical)
    condition_supported_leaf = evaluator.add_leaf(
        id="Video_Footage_Condition_Supported",
        desc="The budget restriction is conditional upon the Defense Secretary providing the specified video footage to Congress",
        parent=video_condition_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"In the bill named '{extracted.official_name or ''}', the restriction on the Defense Secretary's travel budget "
               f"is conditional upon the Defense Secretary providing {extracted.video_footage_description or ''} to Congress."),
        node=condition_supported_leaf,
        sources=combined_sources(extracted),
        additional_instruction=("Verify that the provision clearly links the budget restriction to the act of providing specific video footage to Congress. "
                                "Language such as 'withheld until', 'released upon', or equivalent conditional phrasing should be present and tied to the footage delivery.")
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
    Evaluate an answer for the defense bill identification task.
    """
    # Initialize evaluator and root
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

    # Extract structured bill info
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_bill_info(),
        template_class=BillInfo,
        extraction_name="bill_info"
    )

    # Build tree and run verifications
    await build_verification_tree(evaluator, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()