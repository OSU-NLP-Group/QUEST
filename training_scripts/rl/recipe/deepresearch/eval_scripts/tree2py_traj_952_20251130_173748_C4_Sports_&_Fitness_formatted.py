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
TASK_ID = "nfl_stadium_roof_grass"
TASK_DESCRIPTION = """
Identify the NFL stadium that has both a retractable roof and a natural grass playing surface. Provide the following information about this stadium:
(1) the stadium name,
(2) the NFL team that plays there,
(3) the city and state location,
(4) the standard seating capacity,
(5) confirmation that it has a retractable roof,
(6) confirmation that it uses natural grass, and
(7) at least one reference URL supporting your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumExtraction(BaseModel):
    stadium_name: Optional[str] = None
    team_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    roof_confirmation_text: Optional[str] = None  # exact phrase/sentence from the answer confirming retractable roof
    grass_confirmation_text: Optional[str] = None  # exact phrase/sentence from the answer confirming natural grass
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_info() -> str:
    return """
    From the provided answer, extract the details for a single NFL stadium that the answer claims has BOTH:
    - a retractable roof, and
    - a natural grass playing surface.

    Extract the following fields exactly as they appear in the answer:
    1) stadium_name: The stadium's name (e.g., "State Farm Stadium").
    2) team_name: The NFL team that plays there (e.g., "Arizona Cardinals").
    3) city: The city where the stadium is located (e.g., "Glendale").
    4) state: The state where the stadium is located (e.g., "Arizona").
    5) seating_capacity: The standard seating capacity as stated (string; keep formatting exactly as in the answer, e.g., "63,400").
    6) roof_confirmation_text: Copy the exact phrase or sentence from the answer that explicitly confirms the stadium has a retractable roof; if not explicitly stated, return null.
    7) grass_confirmation_text: Copy the exact phrase or sentence from the answer that explicitly confirms the playing surface is natural grass (not artificial turf); if not explicitly stated, return null.
    8) reference_urls: A list of all explicit URLs provided in the answer as references supporting the stadium identity and roof/surface claims. Extract only valid URLs explicitly present in the answer (including markdown links); if none are present, return an empty list.

    If any field is missing from the answer, set it to null (or empty list for reference_urls).
    Do not infer or fabricate any information beyond what is explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _mentions_retractable(text: Optional[str]) -> bool:
    if not _has_text(text):
        return False
    t = text.lower()
    return "retractable" in t and "roof" in t


def _mentions_natural_grass(text: Optional[str]) -> bool:
    if not _has_text(text):
        return False
    t = text.lower()
    return ("natural" in t and "grass" in t) or ("real grass" in t) or ("grass field" in t)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: StadiumExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Create top-level critical node under the (non-critical) evaluator root
    main = evaluator.add_parallel(
        id="NFL_Stadium_with_Retractable_Roof_and_Natural_Grass",
        desc="Answer identifies an NFL stadium that has a retractable roof and a natural grass playing surface, and provides all required requested details with supporting reference(s).",
        parent=evaluator.root,
        critical=True
    )

    # Reference URLs list
    sources_list = extraction.reference_urls if extraction.reference_urls else []

    # Required details (existence checks) - all critical under the critical parent
    stadium_name_node = evaluator.add_custom_node(
        result=_has_text(extraction.stadium_name),
        id="Stadium_Name",
        desc="Provides the stadium name.",
        parent=main,
        critical=True
    )

    team_name_node = evaluator.add_custom_node(
        result=_has_text(extraction.team_name),
        id="Team_Name",
        desc="Identifies the NFL team that plays at the stadium (home team/primary tenant).",
        parent=main,
        critical=True
    )

    location_node = evaluator.add_custom_node(
        result=_has_text(extraction.city) and _has_text(extraction.state),
        id="Location_City_and_State",
        desc="Provides the city and state where the stadium is located.",
        parent=main,
        critical=True
    )

    capacity_node = evaluator.add_custom_node(
        result=_has_text(extraction.seating_capacity),
        id="Standard_Seating_Capacity",
        desc="Provides the stadium’s standard seating capacity.",
        parent=main,
        critical=True
    )

    ref_url_node = evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="Reference_URL",
        desc="Provides at least one reference URL that supports the key claims (at minimum: stadium identity and roof/surface).",
        parent=main,
        critical=True
    )

    # Stadium Eligibility group (all defining constraints) - critical parallel
    eligibility = evaluator.add_parallel(
        id="Stadium_Eligibility",
        desc="The selected stadium satisfies all defining constraints (NFL stadium + retractable roof + natural grass).",
        parent=main,
        critical=True
    )

    # Explicit mentions in the answer (split into separate checks to avoid combining multiple verifications in one leaf)
    roof_explicit_node = evaluator.add_custom_node(
        result=_mentions_retractable(extraction.roof_confirmation_text),
        id="Has_Retractable_Roof_Explicit_Mention",
        desc="Answer explicitly confirms the stadium has a retractable roof system.",
        parent=eligibility,
        critical=True
    )

    grass_explicit_node = evaluator.add_custom_node(
        result=_mentions_natural_grass(extraction.grass_confirmation_text),
        id="Uses_Natural_Grass_Explicit_Mention",
        desc="Answer explicitly confirms the playing surface is natural grass (not artificial turf).",
        parent=eligibility,
        critical=True
    )

    # 1) Is NFL Stadium (i.e., an NFL team plays there)
    nfl_stadium_leaf = evaluator.add_leaf(
        id="Is_NFL_Stadium",
        desc="Selected venue is an NFL stadium (i.e., an NFL team plays there as a home stadium).",
        parent=eligibility,
        critical=True
    )
    team = extraction.team_name or "the team"
    stadium = extraction.stadium_name or "the stadium"
    nfl_claim = f"The NFL team {team} plays its home games at {stadium}."
    await evaluator.verify(
        claim=nfl_claim,
        node=nfl_stadium_leaf,
        sources=sources_list,
        additional_instruction="Verify on the provided reference(s) that the specified team is an NFL team and that the stadium is its home venue.",
        extra_prerequisites=[stadium_name_node, team_name_node, ref_url_node]
    )

    # 2) Retractable roof confirmation (accuracy via sources)
    roof_leaf = evaluator.add_leaf(
        id="Has_Retractable_Roof_Confirmation",
        desc="Answer explicitly confirms the stadium has a retractable roof system and the claim is correct.",
        parent=eligibility,
        critical=True
    )
    roof_claim = f"{stadium} has a retractable roof."
    await evaluator.verify(
        claim=roof_claim,
        node=roof_leaf,
        sources=sources_list,
        additional_instruction="Confirm that the stadium's roof is retractable (as opposed to fixed or open-air). Look for explicit wording like 'retractable roof'.",
        extra_prerequisites=[roof_explicit_node, ref_url_node, stadium_name_node]
    )

    # 3) Natural grass confirmation (accuracy via sources)
    grass_leaf = evaluator.add_leaf(
        id="Uses_Natural_Grass_Confirmation",
        desc="Answer explicitly confirms the playing surface is natural grass (not artificial turf) and the claim is correct.",
        parent=eligibility,
        critical=True
    )
    grass_claim = f"The playing surface at {stadium} is natural grass (not artificial turf)."
    await evaluator.verify(
        claim=grass_claim,
        node=grass_leaf,
        sources=sources_list,
        additional_instruction="Confirm the field surface is natural grass. Accept descriptions like 'natural grass', 'real grass', or a specific grass type (e.g., Bermuda).",
        extra_prerequisites=[grass_explicit_node, ref_url_node, stadium_name_node]
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
    Evaluate an answer for the NFL stadium with retractable roof and natural grass task.
    """

    # Initialize Evaluator (framework root is non-critical; we add our critical main node under it)
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
        default_model=model
    )

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadium_info(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction"
    )

    # Build and run verifications according to rubric
    await build_verification_tree(evaluator, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()