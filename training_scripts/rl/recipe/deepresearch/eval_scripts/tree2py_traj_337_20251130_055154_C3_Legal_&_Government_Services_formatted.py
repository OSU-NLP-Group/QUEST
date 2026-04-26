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
TASK_ID = "ga_zoning_notice_requirements"
TASK_DESCRIPTION = (
    "Under Georgia state law, what are the specific advance notice publication requirements for local government "
    "public hearings on proposed zoning decisions? Your answer must identify: (1) the specific Georgia Code section "
    "that governs these requirements, (2) the minimum number of days prior to the hearing that notice must be "
    "published, (3) the maximum number of days prior to the hearing that notice may be published, and (4) the type of "
    "publication medium required. Provide reference URLs supporting each element of your answer."
)

# Ground truth references (for summary only)
GROUND_TRUTH = {
    "jurisdiction": "Georgia",
    "statute": "O.C.G.A. § 36-66-4 (Georgia Code § 36-66-4)",
    "minimum_days": "15",
    "maximum_days": "45",
    "publication_medium": "Newspaper of general circulation within the territorial boundaries of the local government",
    "notice_contents": "Include the time, place, and purpose of the hearing",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ZoningNoticeExtraction(BaseModel):
    # Jurisdiction
    state: Optional[str] = None
    state_urls: List[str] = Field(default_factory=list)

    # Statute
    statute: Optional[str] = None
    statute_urls: List[str] = Field(default_factory=list)

    # Timeframe
    minimum_days: Optional[str] = None
    minimum_days_urls: List[str] = Field(default_factory=list)
    maximum_days: Optional[str] = None
    maximum_days_urls: List[str] = Field(default_factory=list)

    # Publication requirements
    publication_medium: Optional[str] = None
    publication_medium_urls: List[str] = Field(default_factory=list)

    # Notice contents
    notice_contents: Optional[str] = None
    notice_contents_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return (
        "Extract the specific elements from the answer related to Georgia state law zoning notice requirements. "
        "Return a JSON object with the following fields:\n"
        "1) state: The state identified as the relevant jurisdiction (string; e.g., 'Georgia').\n"
        "2) state_urls: An array of URLs cited that support Georgia being the relevant jurisdiction for the stated requirements.\n"
        "3) statute: The governing statute citation as written in the answer (string; e.g., 'O.C.G.A. § 36-66-4' or 'Georgia Code § 36-66-4').\n"
        "4) statute_urls: An array of URLs cited that support the governing statute identification.\n"
        "5) minimum_days: The minimum number of days prior to the hearing that notice must be published, exactly as stated in the answer "
        "(string; keep numbers as strings, e.g., '15', or phrases like 'at least 15 days').\n"
        "6) minimum_days_urls: An array of URLs cited that support the minimum-days requirement.\n"
        "7) maximum_days: The maximum number of days prior to the hearing that notice may be published, exactly as stated in the answer "
        "(string; e.g., '45', or phrases like 'not more than 45 days').\n"
        "8) maximum_days_urls: An array of URLs cited that support the maximum-days requirement.\n"
        "9) publication_medium: The required publication medium/location description as stated in the answer "
        "(string; e.g., 'a newspaper of general circulation within the territorial boundaries of the local government').\n"
        "10) publication_medium_urls: An array of URLs cited that support the publication medium requirement.\n"
        "11) notice_contents: The stated required contents of the published notice (string; e.g., 'time, place, and purpose of the hearing').\n"
        "12) notice_contents_urls: An array of URLs cited that support the required notice contents.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer. Do not invent or infer.\n"
        "- For URL fields, include only actual URLs present in the answer (plain URLs or markdown links). If none are present, return an empty array.\n"
        "- If any field is missing, set it to null (for single-value fields) or an empty array (for URL arrays).\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_jurisdiction(evaluator: Evaluator, parent_node, info: ZoningNoticeExtraction) -> None:
    # Group node (critical)
    group = evaluator.add_parallel(
        id="jurisdiction_requirement",
        desc="Identify the correct state jurisdiction and provide supporting URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Leaf: State identification (must be Georgia)
    state_leaf = evaluator.add_leaf(
        id="state_identification",
        desc="Provide Georgia as the relevant state jurisdiction",
        parent=group,
        critical=True,
    )
    state_claim = f"The stated jurisdiction '{info.state}' is Georgia."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        additional_instruction=(
            "Judge whether the stated jurisdiction equals 'Georgia'. Allow minor formatting variations like 'State of Georgia'. "
            "If the value is missing, treat as incorrect."
        ),
    )

    # Custom existence check for URLs (ensure at least one supporting URL provided)
    evaluator.add_custom_node(
        result=bool(info.state_urls) and len(info.state_urls) > 0,
        id="state_reference_urls_provided",
        desc="At least one reference URL is provided supporting Georgia as the jurisdiction",
        parent=group,
        critical=True,
    )

    # Leaf: URLs support that Georgia is the relevant jurisdiction
    state_urls_leaf = evaluator.add_leaf(
        id="state_reference_urls",
        desc="Provide valid reference URL(s) supporting that Georgia is the relevant jurisdiction for the cited requirements",
        parent=group,
        critical=True,
    )
    urls_claim = "The cited requirements are governed by Georgia state law (Georgia/Georgia Code)."
    await evaluator.verify(
        claim=urls_claim,
        node=state_urls_leaf,
        sources=info.state_urls,
        additional_instruction=(
            "Use the provided URLs to confirm Georgia is the jurisdiction. Prefer pages explicitly referencing 'Georgia Code' or 'O.C.G.A.' "
            "or official Georgia government sources. If URLs are irrelevant to Georgia law, judge as not supported."
        ),
    )


async def verify_statute(evaluator: Evaluator, parent_node, info: ZoningNoticeExtraction) -> None:
    # Group node (critical)
    group = evaluator.add_parallel(
        id="statute_requirement",
        desc="Identify the governing Georgia Code section and provide supporting URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Statute identification (must correspond to O.C.G.A. § 36-66-4)
    statute_leaf = evaluator.add_leaf(
        id="statute_identification",
        desc="Provide Georgia Code Title 36, § 36-66-4 (or equivalent citation format) as the governing statute",
        parent=group,
        critical=True,
    )
    statute_claim = (
        f"The statute identified in the answer ('{info.statute}') refers to Georgia Code § 36-66-4 (O.C.G.A. § 36-66-4)."
    )
    await evaluator.verify(
        claim=statute_claim,
        node=statute_leaf,
        additional_instruction=(
            "Allow equivalent citation formats (e.g., 'Georgia Code 36-66-4', 'O.C.G.A. 36-66-4', '§ 36-66-4'). "
            "If the stated statute is missing or is a different section, judge as incorrect."
        ),
    )

    # Custom existence check for URLs
    evaluator.add_custom_node(
        result=bool(info.statute_urls) and len(info.statute_urls) > 0,
        id="statute_reference_urls_provided",
        desc="At least one reference URL is provided supporting the governing statute identification",
        parent=group,
        critical=True,
    )

    # Leaf: URLs support the statute identification
    statute_urls_leaf = evaluator.add_leaf(
        id="statute_reference_urls",
        desc="Provide valid reference URL(s) supporting the governing statute identification",
        parent=group,
        critical=True,
    )
    statute_urls_claim = (
        "The provided URL(s) explicitly identify O.C.G.A. § 36-66-4 (Georgia Code § 36-66-4) as the statute governing "
        "public hearing notice publication for zoning decisions."
    )
    await evaluator.verify(
        claim=statute_urls_claim,
        node=statute_urls_leaf,
        sources=info.statute_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly mention § 36-66-4 and that it governs public hearing notice publication "
            "for local government zoning decisions (Zoning Procedures Law)."
        ),
    )


async def verify_timeframe(evaluator: Evaluator, parent_node, info: ZoningNoticeExtraction) -> None:
    # Group node (critical)
    group = evaluator.add_parallel(
        id="notice_timeframe_requirements",
        desc="Extract both the minimum and maximum advance notice publication timing (days) with supporting URL(s) for each",
        parent=parent_node,
        critical=True,
    )

    # Minimum days requirement (must be 15)
    min_leaf = evaluator.add_leaf(
        id="minimum_days_requirement",
        desc="Provide 15 days as the minimum advance notice period",
        parent=group,
        critical=True,
    )
    min_claim = (
        f"The stated minimum advance notice period ('{info.minimum_days}') corresponds to 15 days prior to the hearing."
    )
    await evaluator.verify(
        claim=min_claim,
        node=min_leaf,
        additional_instruction=(
            "Interpret the extracted text and decide if it denotes 15 days (e.g., '15', 'at least 15 days'). "
            "If missing or not equal to 15, judge as incorrect."
        ),
    )

    # Custom existence check for min URLs
    evaluator.add_custom_node(
        result=bool(info.minimum_days_urls) and len(info.minimum_days_urls) > 0,
        id="minimum_days_reference_urls_provided",
        desc="At least one reference URL is provided supporting the minimum (15 days) notice requirement",
        parent=group,
        critical=True,
    )

    # Leaf: URLs support minimum
    min_urls_leaf = evaluator.add_leaf(
        id="minimum_days_reference_urls",
        desc="Provide valid reference URL(s) supporting the minimum (15 days) notice requirement",
        parent=group,
        critical=True,
    )
    min_urls_claim = "Under Georgia law (O.C.G.A. § 36-66-4), notice must be published not less than 15 days prior to the hearing."
    await evaluator.verify(
        claim=min_urls_claim,
        node=min_urls_leaf,
        sources=info.minimum_days_urls,
        additional_instruction=(
            "Look for 'not less than 15 days' (or equivalent language) on the provided pages. If absent, judge as not supported."
        ),
    )

    # Maximum days requirement (must be 45)
    max_leaf = evaluator.add_leaf(
        id="maximum_days_requirement",
        desc="Provide 45 days as the maximum advance notice period",
        parent=group,
        critical=True,
    )
    max_claim = (
        f"The stated maximum advance notice period ('{info.maximum_days}') corresponds to 45 days prior to the hearing."
    )
    await evaluator.verify(
        claim=max_claim,
        node=max_leaf,
        additional_instruction=(
            "Interpret the extracted text and decide if it denotes 45 days (e.g., '45', 'not more than 45 days'). "
            "If missing or not equal to 45, judge as incorrect."
        ),
    )

    # Custom existence check for max URLs
    evaluator.add_custom_node(
        result=bool(info.maximum_days_urls) and len(info.maximum_days_urls) > 0,
        id="maximum_days_reference_urls_provided",
        desc="At least one reference URL is provided supporting the maximum (45 days) notice requirement",
        parent=group,
        critical=True,
    )

    # Leaf: URLs support maximum
    max_urls_leaf = evaluator.add_leaf(
        id="maximum_days_reference_urls",
        desc="Provide valid reference URL(s) supporting the maximum (45 days) notice requirement",
        parent=group,
        critical=True,
    )
    max_urls_claim = "Under Georgia law (O.C.G.A. § 36-66-4), notice must be published not more than 45 days prior to the hearing."
    await evaluator.verify(
        claim=max_urls_claim,
        node=max_urls_leaf,
        sources=info.maximum_days_urls,
        additional_instruction=(
            "Look for 'not more than 45 days' (or equivalent language) on the provided pages. If absent, judge as not supported."
        ),
    )


async def verify_publication(evaluator: Evaluator, parent_node, info: ZoningNoticeExtraction) -> None:
    # Group node (critical)
    group = evaluator.add_parallel(
        id="publication_requirements",
        desc="Extract publication medium/location requirement and required notice contents, each with supporting URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Publication medium description
    medium_leaf = evaluator.add_leaf(
        id="publication_medium_description",
        desc="Specify that notice must be published in a newspaper of general circulation within the territorial boundaries of the local government",
        parent=group,
        critical=True,
    )
    medium_claim = (
        f"The stated publication medium ('{info.publication_medium}') corresponds to a newspaper of general circulation "
        "within the territorial boundaries of the local government."
    )
    await evaluator.verify(
        claim=medium_claim,
        node=medium_leaf,
        additional_instruction=(
            "Judge whether the text describes the required medium: a newspaper of general circulation within the local government's territorial boundaries. "
            "Minor wording variations are acceptable (e.g., 'newspaper of general circulation in the jurisdiction')."
        ),
    )

    # Custom existence check for medium URLs
    evaluator.add_custom_node(
        result=bool(info.publication_medium_urls) and len(info.publication_medium_urls) > 0,
        id="publication_medium_reference_urls_provided",
        desc="At least one reference URL is provided supporting the publication medium requirement",
        parent=group,
        critical=True,
    )

    # Publication medium reference URLs support
    medium_urls_leaf = evaluator.add_leaf(
        id="publication_medium_reference_urls",
        desc="Provide valid reference URL(s) supporting the publication medium requirement",
        parent=group,
        critical=True,
    )
    medium_urls_claim = (
        "Under Georgia law (O.C.G.A. § 36-66-4), the notice must be published in a newspaper of general circulation "
        "within the territorial boundaries of the local government."
    )
    await evaluator.verify(
        claim=medium_urls_claim,
        node=medium_urls_leaf,
        sources=info.publication_medium_urls,
        additional_instruction=(
            "Confirm the language requiring publication in a 'newspaper of general circulation' within the local government's territorial boundaries."
        ),
    )

    # Notice contents description
    contents_leaf = evaluator.add_leaf(
        id="notice_contents_description",
        desc="State that the published notice must include the time, place, and purpose of the hearing",
        parent=group,
        critical=True,
    )
    contents_claim = (
        f"The stated notice contents ('{info.notice_contents}') include the time, place, and purpose of the hearing."
    )
    await evaluator.verify(
        claim=contents_claim,
        node=contents_leaf,
        additional_instruction=(
            "Judge whether the text includes all three elements: time, place, and purpose of the hearing. "
            "Accept synonymous phrasing (e.g., 'date/time' for time). If any element is missing, judge as incorrect."
        ),
    )

    # Custom existence check for contents URLs
    evaluator.add_custom_node(
        result=bool(info.notice_contents_urls) and len(info.notice_contents_urls) > 0,
        id="notice_contents_reference_urls_provided",
        desc="At least one reference URL is provided supporting the notice contents requirement (time, place, and purpose)",
        parent=group,
        critical=True,
    )

    # Notice contents reference URLs support
    contents_urls_leaf = evaluator.add_leaf(
        id="notice_contents_reference_urls",
        desc="Provide valid reference URL(s) supporting the notice contents requirement (time, place, and purpose)",
        parent=group,
        critical=True,
    )
    contents_urls_claim = (
        "Under Georgia law (O.C.G.A. § 36-66-4), the published notice must include the time, place, and purpose of the hearing."
    )
    await evaluator.verify(
        claim=contents_urls_claim,
        node=contents_urls_leaf,
        sources=info.notice_contents_urls,
        additional_instruction=(
            "Verify that the statute or authoritative source explicitly lists 'time, place, and purpose' as required notice contents."
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
    Evaluate an answer for Georgia zoning notice publication requirements.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=ZoningNoticeExtraction,
        extraction_name="zoning_notice_requirements",
    )

    # Add ground truth for context in summary
    evaluator.add_ground_truth({"expected": GROUND_TRUTH}, gt_type="ground_truth")

    # Build task completion node (critical)
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Complete identification and extraction of Georgia's zoning ordinance public hearing notice requirements (including all stated constraints) with supporting reference URLs",
        parent=root,
        critical=True,
    )

    # Verify sub-requirements
    await verify_jurisdiction(evaluator, task_node, extracted)
    await verify_statute(evaluator, task_node, extracted)
    await verify_timeframe(evaluator, task_node, extracted)
    await verify_publication(evaluator, task_node, extracted)

    # Return structured summary
    return evaluator.get_summary()