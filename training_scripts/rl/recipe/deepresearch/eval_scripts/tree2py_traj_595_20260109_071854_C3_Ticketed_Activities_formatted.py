import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hamilton_lottery_requirements"
TASK_DESCRIPTION = (
    "According to the official Hamilton Broadway lottery policies, what is the minimum age requirement for a patron "
    "to enter the digital lottery, and what specific type of identification document must be presented at the box "
    "office when claiming tickets if they win?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LotteryAgeExtraction(BaseModel):
    """Extraction of age requirement details and cited sources from the agent's answer."""
    stated_minimum_age: Optional[str] = None  # e.g., "18+", "18 or older", "must be 18"
    age_number: Optional[str] = None          # e.g., "18"
    age_policy_quote: Optional[str] = None    # direct quote or summary from the answer
    age_sources: List[str] = Field(default_factory=list)  # URLs cited in the answer relevant to age requirement


class LotteryIDExtraction(BaseModel):
    """Extraction of ID requirement details and cited sources from the agent's answer."""
    id_requirement_summary: Optional[str] = None  # summary of ID requirement as stated in the answer
    id_policy_quote: Optional[str] = None         # quote text from the answer
    id_sources: List[str] = Field(default_factory=list)  # URLs cited in the answer relevant to ID requirement


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_age_policy() -> str:
    return """
    From the provided answer, extract the information related to the minimum age requirement for entering the Hamilton Broadway digital lottery.

    Return a JSON object with the following fields:
    - stated_minimum_age: The exact phrasing used in the answer describing the minimum age (e.g., "18 or older", "must be 18 years of age", etc.). If not present, return null.
    - age_number: If the answer mentions a concrete number (e.g., 18), extract just the number as a string (e.g., "18"). Otherwise, return null.
    - age_policy_quote: A short quotation or paraphrase from the answer that encapsulates the age rule. If absent, return null.
    - age_sources: An array of all URLs mentioned in the answer that are cited specifically for the age requirement (include only URLs; if none are provided, return an empty list).

    Extract URLs exactly as they appear, including markdown links. Do not invent or infer any URLs.
    """


def prompt_extract_id_policy() -> str:
    return """
    From the provided answer, extract the information related to the identification requirement for claiming Hamilton Broadway lottery tickets at the box office.

    Return a JSON object with the following fields:
    - id_requirement_summary: A brief summary of the ID requirement as stated in the answer (e.g., "present a valid, non-expired photo ID", "name on the ID must match the lottery entry name").
    - id_policy_quote: A short quotation or paraphrase from the answer that encapsulates the ID rule(s). If absent, return null.
    - id_sources: An array of all URLs mentioned in the answer that are cited specifically for the ID requirement (include only URLs; if none are provided, return an empty list).

    Extract URLs exactly as they appear, including markdown links. Do not invent or infer any URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_official_urls(urls: List[str]) -> List[str]:
    """Return only official Hamilton Musical URLs."""
    result = []
    for u in urls:
        if isinstance(u, str) and "hamiltonmusical.com" in u.lower():
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_age_requirement(
    evaluator: Evaluator,
    parent_node,
    age_info: LotteryAgeExtraction,
) -> None:
    """
    Build and verify the 'Minimum_Age_Requirement' subtree:
    - Age_Citation: Verify the provided official Hamilton URL(s) explicitly support the age requirement.
      If no official URL is provided in the answer, mark this leaf as failed directly.
    - Age_Value: Verify the specific age requirement (18 years or older) using the official Hamilton URL(s).
    """

    # Create the 'Minimum_Age_Requirement' node (critical, parallel)
    age_group_node = evaluator.add_parallel(
        id="Minimum_Age_Requirement",
        desc="State and verify the minimum age requirement for entering the digital lottery.",
        parent=parent_node,
        critical=True,
    )

    official_age_urls = filter_official_urls(age_info.age_sources or [])

    # Leaf: Age_Citation
    if official_age_urls:
        age_citation_node = evaluator.add_leaf(
            id="Age_Citation",
            desc="Provide a hamiltonmusical.com URL (official Hamilton Broadway lottery documentation) that explicitly supports the stated age requirement.",
            parent=age_group_node,
            critical=True,
        )
        await evaluator.verify(
            claim="The official Hamilton Musical website explicitly states the minimum age to enter the Hamilton Broadway digital lottery is 18 years or older.",
            node=age_citation_node,
            sources=official_age_urls,
            additional_instruction=(
                "Confirm that the provided HamiltonMusical.com page clearly states that entrants must be 18 years old "
                "or older to enter the digital lottery. If the page does not contain this information, judge as not supported."
            ),
        )
    else:
        # No official URL provided in the answer -> mark citation leaf as failed
        evaluator.add_leaf(
            id="Age_Citation",
            desc="Provide a hamiltonmusical.com URL (official Hamilton Broadway lottery documentation) that explicitly supports the stated age requirement.",
            parent=age_group_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Leaf: Age_Value
    age_value_node = evaluator.add_leaf(
        id="Age_Value",
        desc="Minimum age to enter the Hamilton Broadway digital lottery is 18 years or older.",
        parent=age_group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The minimum age requirement to enter the Hamilton Broadway digital lottery is 18 years or older.",
        node=age_value_node,
        sources=official_age_urls if official_age_urls else None,
        additional_instruction=(
            "Verify this claim primarily against the provided official HamiltonMusical.com source(s). Accept equivalent "
            "wording such as 'must be 18 or older' or 'must be at least 18 years old'. If no official source is provided, "
            "you should still consider the claim unsupported."
        ),
    )


async def verify_id_requirement(
    evaluator: Evaluator,
    parent_node,
    id_info: LotteryIDExtraction,
) -> None:
    """
    Build and verify the 'ID_Requirement' subtree:
    - ID_Citation: Verify the provided official Hamilton URL(s) explicitly support the ID requirements.
      If no official URL is provided in the answer, mark this leaf as failed directly.
    - ID_Type_And_Validity: Verify the requirement to present a valid, non-expired photo ID.
    - ID_Name_Match: Verify the requirement that the name on the ID must match the lottery entry name.
    """

    # Create the 'ID_Requirement' node (critical, parallel)
    id_group_node = evaluator.add_parallel(
        id="ID_Requirement",
        desc="State and verify the identification requirement for claiming lottery tickets at the box office.",
        parent=parent_node,
        critical=True,
    )

    official_id_urls = filter_official_urls(id_info.id_sources or [])

    # Leaf: ID_Citation
    if official_id_urls:
        id_citation_node = evaluator.add_leaf(
            id="ID_Citation",
            desc="Provide a hamiltonmusical.com URL (official Hamilton Broadway lottery documentation) that explicitly supports the stated ID requirements.",
            parent=id_group_node,
            critical=True,
        )
        await evaluator.verify(
            claim=(
                "The official Hamilton Musical website explicitly states that winners must present a valid, non-expired "
                "photo ID and that the name on the ID must match the name used to enter the lottery."
            ),
            node=id_citation_node,
            sources=official_id_urls,
            additional_instruction=(
                "Verify that the provided HamiltonMusical.com page mentions both: (1) a valid, non-expired photo ID is required "
                "to claim tickets, and (2) the name on the photo ID must match the name used to enter the lottery. "
                "Both conditions must be explicitly supported."
            ),
        )
    else:
        # No official URL provided in the answer -> mark citation leaf as failed
        evaluator.add_leaf(
            id="ID_Citation",
            desc="Provide a hamiltonmusical.com URL (official Hamilton Broadway lottery documentation) that explicitly supports the stated ID requirements.",
            parent=id_group_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Leaf: ID_Type_And_Validity
    id_type_valid_node = evaluator.add_leaf(
        id="ID_Type_And_Validity",
        desc="Winners must present a valid, non-expired photo ID when claiming tickets.",
        parent=id_group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Winners must present a valid, non-expired photo ID when claiming Hamilton Broadway lottery tickets.",
        node=id_type_valid_node,
        sources=official_id_urls if official_id_urls else None,
        additional_instruction=(
            "Check the provided HamiltonMusical.com page(s) for wording that clearly indicates a photo ID is required and "
            "that it must be valid (non-expired). Accept reasonable variants like 'government-issued photo ID' if the page implies "
            "photo ID and validity."
        ),
    )

    # Leaf: ID_Name_Match
    id_name_match_node = evaluator.add_leaf(
        id="ID_Name_Match",
        desc="The name on the photo ID must exactly match the name used to enter the lottery.",
        parent=id_group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The name on the photo ID must match the name used to enter the Hamilton Broadway digital lottery.",
        node=id_name_match_node,
        sources=official_id_urls if official_id_urls else None,
        additional_instruction=(
            "Verify that the official HamiltonMusical.com page requires the same name on the photo ID as the name used to enter. "
            "Treat 'must match' and 'must be the same name' as equivalent to 'exact match'."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Hamilton Broadway lottery requirements task.
    Builds a hierarchical verification tree that checks:
      - Minimum age requirement (value and official citation).
      - ID requirements (photo ID validity and name match, plus official citation).
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

    # Top-level critical node as per rubric
    complete_node = evaluator.add_parallel(
        id="Complete_Lottery_Requirements",
        desc="Provide both the minimum age requirement to enter the Hamilton Broadway digital lottery and the identification requirement for claiming tickets, each supported by official hamiltonmusical.com documentation.",
        parent=root,
        critical=True,
    )

    # Extraction: Age and ID policy info from the answer
    age_info = await evaluator.extract(
        prompt=prompt_extract_age_policy(),
        template_class=LotteryAgeExtraction,
        extraction_name="age_policy_extraction",
    )

    id_info = await evaluator.extract(
        prompt=prompt_extract_id_policy(),
        template_class=LotteryIDExtraction,
        extraction_name="id_policy_extraction",
    )

    # Optional: Add ground truth info for transparency (non-essential)
    evaluator.add_ground_truth({
        "expected_minimum_age": "18 or older",
        "expected_id_requirements": [
            "Valid, non-expired photo ID required to claim tickets",
            "Name on ID must match the name used to enter the lottery",
        ],
        "source_domain_requirement": "hamiltonmusical.com",
    }, gt_type="ground_truth_policy_expectations")

    # Build and run verifications
    await verify_age_requirement(evaluator, complete_node, age_info)
    await verify_id_requirement(evaluator, complete_node, id_info)

    # Return structured result summary
    return evaluator.get_summary()