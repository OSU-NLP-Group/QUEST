import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "zohran_mamdani_inauguration"
TASK_DESCRIPTION = (
    "Find information about Zohran Mamdani's inauguration as New York City mayor. "
    "Provide the date of the inauguration, the specific location where the ceremony took place, "
    "and the name of the person who administered the oath of office."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InaugurationExtraction(BaseModel):
    date: Optional[str] = None
    location: Optional[str] = None
    oath_administrator: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    oath_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_inauguration_info() -> str:
    return (
        "From the provided answer, extract the following inauguration details for Zohran Mamdani as "
        "New York City mayor:\n"
        "1) date: The exact claimed date of the inauguration.\n"
        "2) location: The specific place or venue where the ceremony took place.\n"
        "3) oath_administrator: The full name of the person who administered the oath of office.\n"
        "Additionally, extract URLs (sources) that the answer cites for each field:\n"
        "4) date_sources: All explicit URL(s) cited for the date claim.\n"
        "5) location_sources: All explicit URL(s) cited for the location claim.\n"
        "6) oath_sources: All explicit URL(s) cited for the oath administrator claim.\n"
        "Finally, also extract:\n"
        "7) general_sources: Any other explicit URL(s) in the answer not clearly tied to a specific field.\n\n"
        "Rules:\n"
        "- Only extract values exactly as stated in the answer; do not infer or invent.\n"
        "- For URLs, extract only explicit URLs appearing in the answer (including markdown links), and list them in the appropriate arrays.\n"
        "- If a field is not mentioned, set it to null. If a field has no URLs, keep its sources list empty.\n"
        "- If the answer provides only general sources without tying to specific fields, include those in general_sources."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def choose_sources(field_specific: List[str], general: List[str]) -> List[str]:
    """
    Prefer field-specific sources; if none, fall back to general sources.
    Deduplicate while preserving order.
    """
    base = field_specific if field_specific else general
    seen = set()
    result = []
    for u in base:
        if u and u not in seen:
            result.append(u)
            seen.add(u)
    return result


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_field_verification(
    evaluator: Evaluator,
    parent_node,
    field_id: str,
    field_container_desc: str,
    value: Optional[str],
    sources: List[str],
    claim_template: str,
    add_instruction: str,
) -> Dict[str, Any]:
    """
    Build a critical parallel sub-tree for one field:
      - existence check: value present and at least one source provided (field-specific or general)
      - support check: verify claim against sources
    Returns dict with handles to created nodes for optional downstream logic.
    """
    field_node = evaluator.add_parallel(
        id=field_id,
        desc=field_container_desc,
        parent=parent_node,
        critical=True,
    )

    has_value = bool(value and value.strip())
    has_sources = bool(sources and len(sources) > 0)

    existence_node = evaluator.add_custom_node(
        result=has_value and has_sources,
        id=f"{field_id}_provided",
        desc=f"{field_container_desc} - value and sources are provided",
        parent=field_node,
        critical=True,
    )

    support_leaf = evaluator.add_leaf(
        id=f"{field_id}_supported",
        desc=f"{field_container_desc} - claim is correct and supported by cited sources",
        parent=field_node,
        critical=True,
    )

    # Build claim text; if value is missing, use an empty placeholder to encourage failure
    value_text = value or ""
    claim = claim_template.format(value=value_text)

    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=sources if has_sources else None,
        additional_instruction=add_instruction,
    )

    return {
        "field_node": field_node,
        "existence_node": existence_node,
        "support_leaf": support_leaf,
    }


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for the Zohran Mamdani inauguration information task.
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

    # Extract structured details from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_inauguration_info(),
        template_class=InaugurationExtraction,
        extraction_name="inauguration_extraction",
    )

    # Build top-level container (non-critical to prevent cross-field auto precondition gating)
    main_node = evaluator.add_parallel(
        id="Zohran_Mamdani_Inauguration_Information",
        desc=(
            "Provide accurate information about Zohran Mamdani's inauguration as New York City mayor, "
            "including the date, location, and who administered the oath of office."
        ),
        parent=root,
        critical=False,
    )

    # Prepare sources for each field
    date_sources = choose_sources(extraction.date_sources, extraction.general_sources)
    location_sources = choose_sources(extraction.location_sources, extraction.general_sources)
    oath_sources = choose_sources(extraction.oath_sources, extraction.general_sources)

    # Build verification sub-trees for each field
    date_nodes = await add_field_verification(
        evaluator=evaluator,
        parent_node=main_node,
        field_id="Inauguration_Date",
        field_container_desc="Provide the correct date when Zohran Mamdani was inaugurated as New York City mayor.",
        value=extraction.date,
        sources=date_sources,
        claim_template="Zohran Mamdani was inaugurated as New York City mayor on '{value}'.",
        add_instruction=(
            "Verify strictly whether the provided webpage(s) explicitly support that Zohran Mamdani "
            "was inaugurated as New York City mayor on this date. If the page indicates he was not NYC mayor "
            "or does not mention such an inauguration, mark as not supported. Allow minor formatting variations in dates."
        ),
    )

    location_nodes = await add_field_verification(
        evaluator=evaluator,
        parent_node=main_node,
        field_id="Inauguration_Location",
        field_container_desc="Provide the correct location where the inauguration ceremony took place.",
        value=extraction.location,
        sources=location_sources,
        claim_template="The inauguration ceremony for Zohran Mamdani as New York City mayor took place at '{value}'.",
        add_instruction=(
            "Verify whether the webpage(s) clearly state the specific venue or place where "
            "Zohran Mamdani's New York City mayoral inauguration occurred. If the page is about a different person "
            "or role, or does not mention such a ceremony, mark as not supported. Allow minor phrasing variations for locations."
        ),
    )

    oath_nodes = await add_field_verification(
        evaluator=evaluator,
        parent_node=main_node,
        field_id="Oath_Administrator",
        field_container_desc="Identify the person who administered the oath of office to Zohran Mamdani.",
        value=extraction.oath_administrator,
        sources=oath_sources,
        claim_template="The oath of office for Zohran Mamdani as New York City mayor was administered by '{value}'.",
        add_instruction=(
            "Verify if the webpage(s) explicitly state the name of the person who administered the oath of office "
            "to Zohran Mamdani for his New York City mayoral inauguration. If the page indicates he did not hold this office "
            "or does not mention an oath administrator, mark as not supported. Allow minor name variants."
        ),
    )

    # Final critical gate: all three supported checks must pass to award full credit
    final_pass = (
        date_nodes["support_leaf"].status == "passed"
        and location_nodes["support_leaf"].status == "passed"
        and oath_nodes["support_leaf"].status == "passed"
    )

    evaluator.add_custom_node(
        result=final_pass,
        id="all_fields_verified",
        desc="All inauguration details (date, location, oath administrator) are correct and supported by cited sources",
        parent=root,
        critical=True,
    )

    # Return evaluation summary
    return evaluator.get_summary()