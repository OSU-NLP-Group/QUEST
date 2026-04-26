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
TASK_ID = "live_oak_tx_school_district"
TASK_DESCRIPTION = (
    "A family with school-age children is planning to move to Live Oak, Texas. "
    "Which school district serves the city of Live Oak, and what is the total geographic area in square miles that this district covers?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictExtraction(BaseModel):
    """Extracted district information from the agent's answer."""
    name: Optional[str] = None
    state: Optional[str] = None
    serves_city: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoverageExtraction(BaseModel):
    """Extracted coverage area (square miles) and related sources from the agent's answer."""
    area_sq_miles: Optional[str] = None  # Keep as string to allow ranges or units (e.g., "55 square miles", "≈55 sq mi")
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_district_info() -> str:
    return (
        "Extract the school district information claimed in the answer regarding Live Oak, Texas.\n"
        "Return a JSON object with the following fields:\n"
        "- name: The full name of the school district the answer identifies as serving Live Oak (e.g., 'Judson Independent School District' or 'Judson ISD'). If multiple are mentioned, choose the primary one the answer asserts serves Live Oak.\n"
        "- state: The state of the identified district if explicitly mentioned (e.g., 'Texas'); otherwise null.\n"
        "- serves_city: The city name the answer claims this district serves (e.g., 'Live Oak'); otherwise null.\n"
        "- sources: An array of all URLs explicitly provided in the answer that support the district identification and/or the claim that it serves Live Oak. Include plain URLs or URLs inside markdown links. If none are provided, return an empty array.\n"
        "Do not invent any information beyond what is present in the answer."
    )


def prompt_extract_coverage_info() -> str:
    return (
        "Extract the claimed total geographic coverage area of the identified school district, measured in square miles.\n"
        "Return a JSON object with the following fields:\n"
        "- area_sq_miles: The text of the area value as stated in the answer (e.g., '55 square miles', 'about 55 sq mi'). If not provided, return null.\n"
        "- sources: An array of all URLs explicitly provided in the answer that support the stated coverage area (e.g., district 'About' page, boundary map, TEA profile). If none are provided, return an empty array.\n"
        "Do not invent any information; only extract what is present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    district: DistrictExtraction,
    coverage: CoverageExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and execute verifications.
    Root-level node is added under evaluator.root with critical aggregation.
    """
    # Create top-level rubric node under root (critical, parallel aggregation)
    top_node = evaluator.add_parallel(
        id="School_District_Query_Response",
        desc="Evaluate whether the response correctly identifies a school district serving Live Oak, Texas and provides accurate geographic coverage information",
        parent=evaluator.root,
        critical=True,
    )

    # 1) District_Identification (existence check; critical)
    district_identification_node = evaluator.add_custom_node(
        result=bool(district.name and district.name.strip()),
        id="District_Identification",
        desc="The response identifies a specific school district by name",
        parent=top_node,
        critical=True,
    )

    # 2) District_Serves_Live_Oak (evidence-backed; critical)
    serves_leaf = evaluator.add_leaf(
        id="District_Serves_Live_Oak",
        desc="The identified school district actually serves the city of Live Oak, Texas according to verifiable official sources",
        parent=top_node,
        critical=True,
    )
    # Source-grounding policy: If no URLs were provided, fail this leaf immediately.
    if not district.sources:
        serves_leaf.score = 0.0
        serves_leaf.status = "failed"
    else:
        district_name = district.name or ""
        claim_serves = f"The school district '{district_name}' serves the city of Live Oak, Texas."
        await evaluator.verify(
            claim=claim_serves,
            node=serves_leaf,
            sources=district.sources,
            additional_instruction=(
                "Use the provided URLs to determine whether the district lists Live Oak as served, "
                "or if boundary/attendance maps explicitly include Live Oak. "
                "Accept phrases like 'communities served' that list Live Oak. Allow minor wording variations."
            ),
        )

    # 3) District_Located_In_Texas (evidence-backed; critical)
    located_leaf = evaluator.add_leaf(
        id="District_Located_In_Texas",
        desc="The identified school district is located in the state of Texas",
        parent=top_node,
        critical=True,
    )
    if not district.sources:
        located_leaf.score = 0.0
        located_leaf.status = "failed"
    else:
        district_name = district.name or ""
        claim_located = f"The school district '{district_name}' is located in Texas."
        await evaluator.verify(
            claim=claim_located,
            node=located_leaf,
            sources=district.sources,
            additional_instruction=(
                "Confirm the district is a Texas school district using the provided URLs (e.g., official district site, TEA profile). "
                "Accept common abbreviations like 'ISD' as indicative only if the page context clearly shows it's Texas."
            ),
        )

    # 4) Coverage_Area_Provided (existence check; critical)
    coverage_provided_node = evaluator.add_custom_node(
        result=bool(coverage.area_sq_miles and coverage.area_sq_miles.strip()),
        id="Coverage_Area_Provided",
        desc="The response provides the district's geographic coverage area in square miles",
        parent=top_node,
        critical=True,
    )

    # 5) Coverage_Area_Accuracy (evidence-backed; critical)
    coverage_accuracy_leaf = evaluator.add_leaf(
        id="Coverage_Area_Accuracy",
        desc="The stated coverage area in square miles matches official district boundary data for the identified district",
        parent=top_node,
        critical=True,
    )
    # Enforce source-grounding: must have URLs to verify coverage area.
    if not coverage.sources:
        coverage_accuracy_leaf.score = 0.0
        coverage_accuracy_leaf.status = "failed"
    else:
        district_name = district.name or "the identified district"
        area_text = coverage.area_sq_miles or ""
        claim_area = (
            f"The total geographic area covered by {district_name} is {area_text}, measured in square miles."
        )
        await evaluator.verify(
            claim=claim_area,
            node=coverage_accuracy_leaf,
            sources=coverage.sources,
            additional_instruction=(
                "Verify the square-mile figure against official sources (e.g., district 'About' page, boundary/attendance area documentation, TEA profile). "
                "Allow minor rounding differences and equivalent unit expressions (e.g., 'sq mi'). The claim should be explicitly supported."
            ),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Live Oak, Texas school district and coverage area task.
    Returns the standard evaluation summary dictionary produced by Evaluator.get_summary().
    """
    # Initialize evaluator (root is non-critical by design; we'll add a critical child node)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction: district info and coverage info
    district_info = await evaluator.extract(
        prompt=prompt_extract_district_info(),
        template_class=DistrictExtraction,
        extraction_name="district_info",
    )
    coverage_info = await evaluator.extract(
        prompt=prompt_extract_coverage_info(),
        template_class=CoverageExtraction,
        extraction_name="coverage_info",
    )

    # Build tree and run verifications according to rubric
    await build_and_verify_tree(evaluator, district_info, coverage_info)

    # Return structured evaluation summary
    return evaluator.get_summary()