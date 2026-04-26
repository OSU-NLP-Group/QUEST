import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_semiconductor_facility"
TASK_DESCRIPTION = (
    "Identify the name of the semiconductor manufacturing facility project in the United States that meets ALL of the "
    "following criteria: (1) The facility is located in the Licking County portion of the New Albany International "
    "Business Park in Ohio, (2) The groundbreaking ceremony for this facility took place in 2022, (3) The initial "
    "investment announcement for this project was for more than $20 billion, (4) As updated in early 2025, the first "
    "fabrication plant (Mod 1) is expected to begin operations between 2030 and 2031. Provide the official name of "
    "this facility project and the company operating it, along with reference URLs confirming each of the above criteria."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    facility_project_name: Optional[str] = None
    operating_company_name: Optional[str] = None

    # Location constraint
    location_urls: List[str] = Field(default_factory=list)

    # Groundbreaking constraint
    groundbreaking_year: Optional[str] = None
    groundbreaking_urls: List[str] = Field(default_factory=list)

    # Investment constraint
    initial_investment_phrase: Optional[str] = None
    investment_urls: List[str] = Field(default_factory=list)

    # Operations timeline constraint (early 2025 update)
    operations_timeline_text: Optional[str] = None
    operations_update_timeframe: Optional[str] = None
    operations_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return (
        "Extract the following fields strictly from the provided answer text. Do not invent or infer any values.\n"
        "Return a JSON object with these fields:\n"
        "1) facility_project_name: The official or recognized name of the semiconductor manufacturing facility project mentioned in the answer. If not explicitly provided, return null.\n"
        "2) operating_company_name: The company operating the facility project as mentioned in the answer. If not explicitly provided, return null.\n"
        "3) location_urls: An array of all explicit URLs in the answer that support the location being in the Licking County portion of the New Albany International Business Park in Ohio. If none are provided, return an empty array.\n"
        "4) groundbreaking_year: The year stated in the answer for the groundbreaking ceremony (e.g., '2022', 'September 2022'). If not stated, return null.\n"
        "5) groundbreaking_urls: An array of all explicit URLs in the answer that support the 2022 groundbreaking claim. If none are provided, return an empty array.\n"
        "6) initial_investment_phrase: The exact phrase from the answer describing the initial investment announcement amount for the project (e.g., '$20 billion', 'over $20B', 'more than $20 billion'). If not stated, return null.\n"
        "7) investment_urls: An array of all explicit URLs in the answer that support the initial investment amount being more than $20B. If none are provided, return an empty array.\n"
        "8) operations_timeline_text: The exact phrase the answer uses for the operations timeline of Mod 1 (e.g., 'between 2030 and 2031'). If not stated, return null.\n"
        "9) operations_update_timeframe: The phrase in the answer indicating that the timeline is from an early 2025 update (e.g., 'early 2025', 'January 2025', 'Q1 2025'). If not stated, return null.\n"
        "10) operations_urls: An array of all explicit URLs in the answer that support the early-2025 updated operations timeline for Mod 1 being 2030–2031. If none are provided, return an empty array.\n"
        "Notes:\n"
        "- Only include URLs explicitly present in the answer text (plain text or markdown link). Do not infer URLs.\n"
        "- For year and phrases, capture exactly what the answer states (free text string)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_project_and_operator_provided(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityExtraction,
) -> None:
    """
    Verify that the official facility project name and the operating company are provided.
    """
    node = evaluator.add_parallel(
        id="project_and_operator_provided",
        desc="Provide the official facility project name and the operating company",
        parent=parent_node,
        critical=True,
    )

    # Facility project name presence (critical)
    evaluator.add_custom_node(
        result=bool(ext.facility_project_name and ext.facility_project_name.strip()),
        id="facility_project_name",
        desc="Answer provides the official name of the facility project",
        parent=node,
        critical=True,
    )

    # Operating company name presence (critical)
    evaluator.add_custom_node(
        result=bool(ext.operating_company_name and ext.operating_company_name.strip()),
        id="operating_company_name",
        desc="Answer identifies the company operating the facility project",
        parent=node,
        critical=True,
    )


async def verify_constraints_with_sources(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityExtraction,
) -> None:
    """
    Verify each constraint and check that there is at least one supporting URL that confirms the claim.
    """
    constraints_node = evaluator.add_parallel(
        id="constraints_verified_with_sources",
        desc="Each stated constraint is satisfied and supported by at least one cited reference URL",
        parent=parent_node,
        critical=True,
    )

    # Location constraint
    loc_node = evaluator.add_parallel(
        id="location_constraint",
        desc="Facility is located in the Licking County portion of the New Albany International Business Park in Ohio, with supporting URL",
        parent=constraints_node,
        critical=True,
    )

    # Fact stated in the answer
    loc_fact_leaf = evaluator.add_leaf(
        id="location_fact_correct",
        desc="Answer states the facility location matches the specified Licking County portion of the New Albany International Business Park in Ohio",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the facility is located in the Licking County portion of the New Albany International Business Park in Ohio.",
        node=loc_fact_leaf,
        additional_instruction="Judge based solely on the provided answer text. Allow equivalent phrasing like 'in Licking County at New Albany International Business Park'.",
    )

    # Supported by provided URLs
    loc_support_leaf = evaluator.add_leaf(
        id="location_supporting_url",
        desc="At least one valid reference URL is provided that supports the stated location constraint",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The facility described is located in the Licking County portion of the New Albany International Business Park in Ohio.",
        node=loc_support_leaf,
        sources=ext.location_urls,
        additional_instruction="Verify strictly against the cited page(s). Accept minor wording variants such as 'New Albany International Business Park site in Licking County, Ohio'.",
    )

    # Groundbreaking constraint
    gb_node = evaluator.add_parallel(
        id="groundbreaking_constraint",
        desc="Groundbreaking ceremony occurred in 2022, with supporting URL",
        parent=constraints_node,
        critical=True,
    )

    gb_fact_leaf = evaluator.add_leaf(
        id="groundbreaking_fact_correct",
        desc="Answer states the groundbreaking ceremony took place in 2022",
        parent=gb_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the groundbreaking ceremony took place in 2022.",
        node=gb_fact_leaf,
        additional_instruction="Consider phrasing like 'September 2022' or 'late 2022' as satisfying the year 2022.",
    )

    gb_support_leaf = evaluator.add_leaf(
        id="groundbreaking_supporting_url",
        desc="At least one valid reference URL is provided that supports the 2022 groundbreaking claim",
        parent=gb_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The groundbreaking ceremony for the facility occurred in 2022.",
        node=gb_support_leaf,
        sources=ext.groundbreaking_urls,
        additional_instruction="Confirm via the provided page(s) that the groundbreaking took place in calendar year 2022. Accept explicit date references within 2022.",
    )

    # Investment constraint
    inv_node = evaluator.add_parallel(
        id="investment_constraint",
        desc="Initial investment announcement was for more than $20 billion, with supporting URL",
        parent=constraints_node,
        critical=True,
    )

    inv_fact_leaf = evaluator.add_leaf(
        id="investment_fact_correct",
        desc="Answer states the initial investment announcement exceeded $20 billion",
        parent=inv_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the initial investment announcement for this project was for more than $20 billion.",
        node=inv_fact_leaf,
        additional_instruction="Look for phrases such as 'more than $20 billion', 'over $20B', or equivalent wording. Focus on the answer text only.",
    )

    inv_support_leaf = evaluator.add_leaf(
        id="investment_supporting_url",
        desc="At least one valid reference URL is provided that supports the initial investment amount being > $20B",
        parent=inv_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The initial investment announcement for the project was for more than $20 billion.",
        node=inv_support_leaf,
        sources=ext.investment_urls,
        additional_instruction="Verify using the cited page(s). Accept synonymous phrasing indicating the initial announcement exceeded $20B (e.g., 'over $20 billion').",
    )

    # Operations timeline constraint
    ops_node = evaluator.add_parallel(
        id="operations_timeline_constraint",
        desc="As updated in early 2025, Mod 1 expected to begin operations between 2030 and 2031, with supporting URL",
        parent=constraints_node,
        critical=True,
    )

    ops_fact_leaf = evaluator.add_leaf(
        id="operations_timeline_fact_correct",
        desc="Answer states that (per an early-2025 update) Mod 1 is expected to begin operations between 2030 and 2031",
        parent=ops_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that, per an early 2025 update, the first fabrication plant (Mod 1) is expected to begin operations between 2030 and 2031.",
        node=ops_fact_leaf,
        additional_instruction="Answer text should indicate both the 'early 2025' update context and the 2030–2031 operations window (e.g., 'in 2030 or 2031').",
    )

    ops_support_leaf = evaluator.add_leaf(
        id="operations_timeline_supporting_url",
        desc="At least one valid reference URL is provided that supports the early-2025 updated 2030–2031 operations timeline for Mod 1",
        parent=ops_node,
        critical=True,
    )
    await evaluator.verify(
        claim="According to an early 2025 update, Mod 1 (the first fabrication plant) is expected to begin operations between 2030 and 2031.",
        node=ops_support_leaf,
        sources=ext.operations_urls,
        additional_instruction=(
            "Confirm that the cited page(s) are an early 2025 update (e.g., Jan–Mar 2025) and explicitly state the operations "
            "timeline for Mod 1 as 2030–2031 (allow wording such as 'operations start', 'production begins', 'start-up')."
        ),
    )


async def verify_facility_identification(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityExtraction,
) -> None:
    """
    Build the facility identification tree (sequential, critical).
    """
    fi_node = evaluator.add_sequential(
        id="facility_identification",
        desc="Identify the semiconductor facility project and provide required details and supporting evidence per the question constraints",
        parent=parent_node,
        critical=True,
    )

    # Step 1: Project and operator provided
    await verify_project_and_operator_provided(evaluator, fi_node, ext)

    # Step 2: Constraints verified with sources
    await verify_constraints_with_sources(evaluator, fi_node, ext)


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
    Evaluate an answer for the Ohio semiconductor facility identification task.
    """
    # Initialize evaluator (root node is non-critical by design)
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

    # Extract structured info from the answer
    ext: FacilityExtraction = await evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Optionally record simple custom info helpful for debugging
    evaluator.add_custom_info(
        info={
            "facility_project_name": ext.facility_project_name,
            "operating_company_name": ext.operating_company_name,
            "location_url_count": len(ext.location_urls),
            "groundbreaking_url_count": len(ext.groundbreaking_urls),
            "investment_url_count": len(ext.investment_urls),
            "operations_url_count": len(ext.operations_urls),
        },
        info_type="extraction_stats",
        info_name="extraction_stats",
    )

    # Build and run verification tree
    await verify_facility_identification(evaluator, root, ext)

    # Return structured summary
    return evaluator.get_summary()