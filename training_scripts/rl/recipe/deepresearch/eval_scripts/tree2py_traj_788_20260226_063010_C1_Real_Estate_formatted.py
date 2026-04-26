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
TASK_ID = "gsk_biologics_flex_pa_township"
TASK_DESCRIPTION = (
    "In September 2025, GSK announced plans to construct a new biologics flex factory in Montgomery County, "
    "Pennsylvania, with construction planned to commence in 2026. What is the name of the township or municipality "
    "where this facility will be built?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    """
    Structured extraction of the GSK project details as presented in the agent's answer.
    """
    project_description: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    announcement_date: Optional[str] = None  # e.g., "September 2025"
    construction_commence_year: Optional[str] = None  # e.g., "2026"
    township_or_municipality: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return """
    Extract the details about the GSK biologics flex factory project referenced in the answer.

    Return a JSON object with the following fields:
    1. project_description: Brief description of the project as stated in the answer (e.g., "GSK biologics flex factory").
    2. county: County name if mentioned (e.g., "Montgomery County").
    3. state: State name if mentioned (e.g., "Pennsylvania").
    4. announcement_date: The month and year of the announcement as stated (e.g., "September 2025").
    5. construction_commence_year: The year construction is planned to commence, as stated (e.g., "2026").
    6. township_or_municipality: The township or municipality name where the facility will be built (e.g., "Upper Merion Township"). 
       Do not return county or state in this field; it must be the specific municipality/township/borough/city.
    7. source_urls: All URLs explicitly cited in the answer that support any of the above information (including the location). 
       Extract only valid URLs present in the answer text. Include multiple if there are several.

    Rules:
    - Do not invent information; only extract what's explicitly stated in the answer.
    - If a field is not present, set it to null (for strings) or [] (for lists).
    - For URLs, accept plain or markdown links; normalize them and include full protocol.
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: FacilityExtraction) -> None:
    """
    Construct and execute the verification tree for the GSK township identification rubric.
    """
    # Top-level rubric node (critical)
    main_node = evaluator.add_parallel(
        id="GSK_Facility_Township_Identification",
        desc="Correctly identifies the township or municipality where the referenced GSK biologics flex factory will be built.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare commonly used values from extraction
    township = (extracted.township_or_municipality or "").strip()
    county = (extracted.county or "").strip()
    state = (extracted.state or "").strip()
    announcement = (extracted.announcement_date or "").strip()
    start_year = (extracted.construction_commence_year or "").strip()
    sources_list = extracted.source_urls or []

    # Leaf 1: Project_Match (critical)
    project_match_node = evaluator.add_leaf(
        id="Project_Match",
        desc=("Answer clearly refers to the GSK-announced biologics flex factory project in Montgomery County, "
              "Pennsylvania, announced in September 2025 with construction planned to commence in 2026 (not a different project)."),
        parent=main_node,
        critical=True
    )

    # Build the project match claim using the rubric's specifics rather than relying on extracted variants
    project_match_claim = (
        "The provided sources explicitly describe GSK's biologics flex factory project in Montgomery County, Pennsylvania, "
        "that was announced in September 2025, with construction planned to commence in 2026. The sources are about this same project, "
        "not a different GSK facility."
    )
    await evaluator.verify(
        claim=project_match_claim,
        node=project_match_node,
        sources=sources_list,
        additional_instruction=(
            "Verify the page(s) explicitly mention: (1) GSK biologics flex factory; (2) Montgomery County, Pennsylvania; "
            "(3) announcement timing in September 2025; (4) construction planned to commence in 2026. "
            "Minor wording variations are acceptable, but the core facts must match the same project."
        )
    )

    # Leaf 2: Township_Or_Municipality_Provided (critical, existence check)
    municipality_provided_node = evaluator.add_custom_node(
        result=bool(township),
        id="Township_Or_Municipality_Provided",
        desc="Answer provides a specific township or municipality name (not merely the county or state).",
        parent=main_node,
        critical=True
    )

    # Leaf 3: Township_Or_Municipality_Correctness (critical)
    municipality_correct_node = evaluator.add_leaf(
        id="Township_Or_Municipality_Correctness",
        desc="The township/municipality named is the correct build location for that specific project.",
        parent=main_node,
        critical=True
    )

    # Construct correctness claim; include county/state context if available
    # Even if county/state were not extracted, the claim focuses on the municipality correctness for the same project.
    if county and state:
        municipality_claim = (
            f"The GSK biologics flex factory project described will be built in {township} in {county}, {state}."
        )
    else:
        municipality_claim = (
            f"The GSK biologics flex factory project described will be built in {township}."
        )

    await evaluator.verify(
        claim=municipality_claim,
        node=municipality_correct_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm that the cited source(s) explicitly identify the exact township/municipality (or borough/city) "
            "where the project will be built. Accept reasonable naming variants (e.g., 'Twp.' for 'Township', "
            "or municipality vs. borough/city distinctions) as long as the place is the same. "
            "If the sources do not name a specific municipality or they contradict the provided name, mark as not supported."
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
    Evaluate an agent's answer for the GSK biologics flex factory township identification task.
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return the standardized summary
    return evaluator.get_summary()