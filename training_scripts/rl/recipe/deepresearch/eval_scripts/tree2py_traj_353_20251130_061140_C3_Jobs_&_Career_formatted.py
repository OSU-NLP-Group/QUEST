import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "psu_coach_hire_2023"
TASK_DESCRIPTION = (
    "In March 2023, Penn State University announced the hiring of a new men's basketball head coach who came from "
    "Virginia Commonwealth University (VCU), where he had served as head coach. This hiring included a multi-year, "
    "multi-million dollar contract. Please identify this coach and provide the following verified information: "
    "(1) the coach's full name, (2) the exact date the hiring was announced, (3) the contract duration in years and "
    "the total contract value in dollars, (4) the years of his tenure as head coach at VCU, and (5) the name of the "
    "coach he replaced at Penn State."
)

# Ground truth constraints (for verification phrasing and guidance)
EXPECTED = {
    "coach_full_name": "Mike Rhoades",
    "announcement_date": "March 29, 2023",
    "contract_duration_years": "7",
    "contract_total_value": "$25.9 million",
    "vcu_tenure_start_year": "2017",
    "vcu_tenure_end_year": "2023",
    "replaced_coach_name": "Micah Shrewsberry",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HireDetails(BaseModel):
    coach_full_name: Optional[str] = None
    announcement_date: Optional[str] = None
    contract_duration_years: Optional[str] = None
    contract_total_value_dollars: Optional[str] = None
    vcu_tenure_start_year: Optional[str] = None
    vcu_tenure_end_year: Optional[str] = None
    replaced_coach_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hire_details() -> str:
    return (
        "Extract the details about the Penn State men's basketball head coach hire (March 2023, from VCU) as stated "
        "in the answer. Return a JSON object with the following fields:\n"
        "1) coach_full_name: The coach's full name exactly as provided in the answer.\n"
        "2) announcement_date: The hiring announcement date as stated in the answer (e.g., 'March 29, 2023').\n"
        "3) contract_duration_years: The contract duration in years as stated (accept forms like '7', '7 years', 'seven').\n"
        "4) contract_total_value_dollars: The total contract value in dollars as stated (accept '$25.9 million', "
        "'25.9M', '25,900,000 USD').\n"
        "5) vcu_tenure_start_year: The start year of his tenure as VCU head coach as stated (e.g., '2017').\n"
        "6) vcu_tenure_end_year: The end year of his tenure as VCU head coach as stated (e.g., '2023').\n"
        "7) replaced_coach_name: The name of the coach he replaced at Penn State as stated.\n"
        "8) source_urls: An array of all URLs explicitly cited in the answer that support this hire and its details. "
        "Include full URLs (plain or markdown). If no URLs are provided, return an empty array.\n"
        "Rules:\n"
        "- Extract ONLY what appears in the answer; do not invent values.\n"
        "- If an item is not mentioned, set it to null (or empty array for source_urls).\n"
        "- Keep dates and names as written; minor normalization is okay (e.g., trim whitespace)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_coach_identity(
    evaluator: Evaluator,
    parent_node,
    details: HireDetails,
) -> None:
    """
    Build and verify the 'identify_coach' subtree (critical, parallel).
    """
    identify_node = evaluator.add_parallel(
        id="identify_coach",
        desc="Correctly identify the coach associated with the described Penn State hire.",
        parent=parent_node,
        critical=True,
    )

    # Existence check for coach full name
    evaluator.add_custom_node(
        result=bool(details.coach_full_name and details.coach_full_name.strip()),
        id="coach_full_name_provided",
        desc="Provides the coach's full name.",
        parent=identify_node,
        critical=True,
    )

    # Verify the identified coach matches the hire context and expected identity
    coach_context_node = evaluator.add_leaf(
        id="coach_matches_hire_context",
        desc=(
            "The identified coach matches the described hire context: Penn State men's basketball head coach hired "
            "in March 2023 from VCU where he served as head coach."
        ),
        parent=identify_node,
        critical=True,
    )

    claim = (
        f"The coach identified in the answer ('{details.coach_full_name}') is {EXPECTED['coach_full_name']}, and he "
        f"was hired as the Penn State men's basketball head coach in March 2023 after serving as the head coach at "
        f"Virginia Commonwealth University (VCU)."
    )
    await evaluator.verify(
        claim=claim,
        node=coach_context_node,
        sources=details.source_urls,
        additional_instruction=(
            "Confirm both the identity (Mike Rhoades, allowing minor name variants) and the context: "
            "hire is for Penn State men's basketball, announcement occurred in March 2023, and he came from VCU where "
            "he served as head coach. Only pass if the provided URLs explicitly support all parts of the claim AND the "
            "answer-stated name refers to Mike Rhoades."
        ),
    )


async def verify_required_details(
    evaluator: Evaluator,
    parent_node,
    details: HireDetails,
) -> None:
    """
    Build and verify the 'required_details' subtree (critical, parallel).
    """
    req_node = evaluator.add_parallel(
        id="required_details",
        desc="Provides all required verified details about the hire.",
        parent=parent_node,
        critical=True,
    )

    # Announcement date
    node_announcement = evaluator.add_leaf(
        id="announcement_date",
        desc="States the hiring announcement date and it matches the constraint (March 29, 2023).",
        parent=req_node,
        critical=True,
    )
    claim_announcement = (
        f"The hiring announcement date stated in the answer ('{details.announcement_date}') is "
        f"{EXPECTED['announcement_date']}."
    )

    # Contract duration in years
    node_contract_years = evaluator.add_leaf(
        id="contract_duration_years",
        desc="States the contract duration in years and it matches the constraint (7 years).",
        parent=req_node,
        critical=True,
    )
    claim_contract_years = (
        f"The contract duration stated in the answer ('{details.contract_duration_years}') is "
        f"{EXPECTED['contract_duration_years']} years."
    )

    # Contract total value
    node_contract_value = evaluator.add_leaf(
        id="contract_total_value",
        desc="States the total contract value in dollars and it matches the constraint ($25.9 million).",
        parent=req_node,
        critical=True,
    )
    claim_contract_value = (
        f"The total contract value stated in the answer ('{details.contract_total_value_dollars}') is "
        f"{EXPECTED['contract_total_value']}."
    )

    # VCU tenure years
    node_vcu_years = evaluator.add_leaf(
        id="vcu_tenure_years",
        desc="States the years of tenure as VCU head coach and they match the constraint (2017 to 2023).",
        parent=req_node,
        critical=True,
    )
    stated_span = (
        f"{details.vcu_tenure_start_year} to {details.vcu_tenure_end_year}"
        if details.vcu_tenure_start_year or details.vcu_tenure_end_year
        else "None"
    )
    claim_vcu_years = (
        f"The years of tenure as VCU head coach stated in the answer ('{stated_span}') are "
        f"{EXPECTED['vcu_tenure_start_year']} to {EXPECTED['vcu_tenure_end_year']}."
    )

    # Replaced coach
    node_replaced = evaluator.add_leaf(
        id="replaced_coach",
        desc="States the name of the coach replaced at Penn State and it matches the constraint (Micah Shrewsberry).",
        parent=req_node,
        critical=True,
    )
    claim_replaced = (
        f"The coach he replaced at Penn State stated in the answer ('{details.replaced_coach_name}') is "
        f"{EXPECTED['replaced_coach_name']}."
    )

    # Batch verify all detail claims in parallel
    await evaluator.batch_verify(
        [
            (
                claim_announcement,
                details.source_urls,
                node_announcement,
                (
                    "Use official Penn State Athletics announcements or reputable news sources to confirm the "
                    "announcement date. Only pass if sources explicitly support March 29, 2023 and the "
                    "answer-stated date equals that value (allow minor formatting differences)."
                ),
            ),
            (
                claim_contract_years,
                details.source_urls,
                node_contract_years,
                (
                    "Confirm the contract duration is 7 years. Accept phrasing variants like '7-year', 'seven years'. "
                    "Only pass if sources explicitly support 7 years and the answer-stated duration equals 7 years."
                ),
            ),
            (
                claim_contract_value,
                details.source_urls,
                node_contract_value,
                (
                    "Confirm the total value is $25.9 million. Accept equivalent numeric representations "
                    "like 25.9M, $25,900,000, USD 25.9 million. Only pass if sources support semantic equivalence "
                    "to $25.9 million and the answer-stated value is equivalent."
                ),
            ),
            (
                claim_vcu_years,
                details.source_urls,
                node_vcu_years,
                (
                    "Confirm the VCU head coach tenure span is 2017 to 2023. Accept minor formatting variants "
                    "like '2017–23'. Only pass if sources support 2017–2023 and the answer-stated years match."
                ),
            ),
            (
                claim_replaced,
                details.source_urls,
                node_replaced,
                (
                    "Confirm the replaced coach at Penn State was Micah Shrewsberry. Accept minor name variants. "
                    "Only pass if sources explicitly support this and the answer-stated name matches."
                ),
            ),
        ]
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Penn State coach hire (March 2023 from VCU) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured details from the answer
    details = await evaluator.extract(
        prompt=prompt_extract_hire_details(),
        template_class=HireDetails,
        extraction_name="hire_details",
    )

    # Add ground truth info to summary for transparency
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "task_focus": "Penn State men's basketball head coach hire (March 2023) from VCU",
        },
        gt_type="ground_truth",
    )

    # Build the critical, sequential root child as per rubric
    hire_node = evaluator.add_sequential(
        id="coaching_hire_identification",
        desc="Identify the Penn State men's basketball head coach hire (March 2023, from VCU) and provide all required verified details.",
        parent=root,
        critical=True,
    )

    # Subtree 1: Identify coach
    await verify_coach_identity(evaluator, hire_node, details)

    # Subtree 2: Required details
    await verify_required_details(evaluator, hire_node, details)

    # Return final structured summary
    return evaluator.get_summary()