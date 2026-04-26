import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "studio_identification_2002_cod_msft69b"
TASK_DESCRIPTION = (
    "Identify the gaming studio that meets all of the following criteria: "
    "(1) The studio was founded in 2002 by exactly three people: Grant Collier, Jason West, and Vince Zampella; "
    "(2) The studio developed the first game in a major first-person shooter franchise, which was released on October 29, 2003; "
    "(3) The studio's parent company was acquired by Microsoft in a deal that was completed in October 2023; "
    "(4) The acquisition deal was valued at approximately $69 billion. Provide the name of the studio and a reference URL that confirms these details."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StudioExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    studio_name: Optional[str] = None
    founders: List[str] = Field(default_factory=list)
    founding_year: Optional[str] = None

    first_game_title: Optional[str] = None
    franchise_name: Optional[str] = None
    first_game_release_date: Optional[str] = None
    first_game_genre: Optional[str] = None

    parent_company: Optional[str] = None
    acquirer: Optional[str] = None
    acquisition_completion_month_year: Optional[str] = None
    deal_value: Optional[str] = None

    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_studio_info() -> str:
    return """
    Extract the studio identification details explicitly from the provided answer text. Return a single JSON object with the following fields:

    1. studio_name: The gaming studio's name identified as meeting the criteria.
    2. founders: An array of founder names listed in the answer (do not invent). If the answer lists exactly three founders, they should be: Grant Collier, Jason West, Vince Zampella.
    3. founding_year: The year the studio was founded (as a string, e.g., "2002").
    4. first_game_title: The title of the first game in the major franchise associated with the studio (e.g., "Call of Duty").
    5. franchise_name: The franchise name (e.g., "Call of Duty").
    6. first_game_release_date: The release date of that first game (e.g., "October 29, 2003").
    7. first_game_genre: The genre of the game (e.g., "first-person shooter").
    8. parent_company: The studio's parent company at the time of the Microsoft acquisition (e.g., "Activision" or "Activision Blizzard").
    9. acquirer: The company that acquired the parent company (e.g., "Microsoft").
    10. acquisition_completion_month_year: The month and year when the acquisition was completed (e.g., "October 2023").
    11. deal_value: The approximate total value of the acquisition deal (e.g., "$69 billion").
    12. reference_urls: An array of one or more URLs explicitly cited in the answer that support these details. Only include valid URLs present in the answer. If none are present, return an empty array.

    Rules:
    - Extract only what appears in the answer. If a field is not present, set it to null (or empty array for lists).
    - For URLs, extract the actual URL strings (from plain text or markdown links).
    - Do not add or infer information beyond what the answer provides.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str], fallback: str) -> str:
    return (name or "").strip() or fallback

def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst or []

def _safe_franchise(extracted: StudioExtraction) -> str:
    # Prefer extracted franchise_name; otherwise fall back to common expected franchise "Call of Duty"
    return _safe_name(extracted.franchise_name, "Call of Duty")

def _safe_first_game_title(extracted: StudioExtraction) -> str:
    # Prefer extracted first_game_title; otherwise fall back to franchise name
    fallback = _safe_franchise(extracted)
    return _safe_name(extracted.first_game_title, fallback)

def _safe_parent_company(extracted: StudioExtraction) -> str:
    # Prefer extracted parent company; otherwise use "Activision Blizzard" which commonly appears in acquisition coverage
    return _safe_name(extracted.parent_company, "Activision Blizzard")


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_studio_founding(
    evaluator: Evaluator,
    parent_node,
    extracted: StudioExtraction,
) -> None:
    studio_name = _safe_name(extracted.studio_name, "the identified studio")
    sources = _safe_list(extracted.reference_urls)

    founding_node = evaluator.add_parallel(
        id="Studio_Founding",
        desc="Verify the studio's founding year and founders",
        parent=parent_node,
        critical=True,
    )

    # Founded in 2002
    founded_2002_leaf = evaluator.add_leaf(
        id="Founded_2002",
        desc="The studio was founded in 2002",
        parent=founding_node,
        critical=True,
    )
    founded_claim = f"{studio_name} was founded in 2002."
    await evaluator.verify(
        claim=founded_claim,
        node=founded_2002_leaf,
        sources=sources,
        additional_instruction="Check the page for the studio's founding year and confirm it states 2002."
    )

    # Three founders group
    three_founders_node = evaluator.add_parallel(
        id="Three_Founders",
        desc="The studio had exactly three co-founders with the specified names",
        parent=founding_node,
        critical=True,
    )

    # Explicit exact-three founders claim
    founders_exact_leaf = evaluator.add_leaf(
        id="Founders_Exact_Three",
        desc="The studio was founded by exactly three co-founders: Grant Collier, Jason West, Vince Zampella",
        parent=three_founders_node,
        critical=True,
    )
    founders_exact_claim = (
        f"{studio_name} was founded by exactly three people: Grant Collier, Jason West, and Vince Zampella."
    )
    await evaluator.verify(
        claim=founders_exact_claim,
        node=founders_exact_leaf,
        sources=sources,
        additional_instruction="Confirm that the page explicitly indicates these three individuals as the full set of co-founders (no more, no fewer)."
    )

    # Individual founder confirmations
    founder_grant_leaf = evaluator.add_leaf(
        id="Founder_Grant_Collier",
        desc="Grant Collier was one of the co-founders",
        parent=three_founders_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Grant Collier was one of the co-founders of {studio_name}.",
        node=founder_grant_leaf,
        sources=sources,
        additional_instruction="Confirm the page lists Grant Collier as a co-founder."
    )

    founder_west_leaf = evaluator.add_leaf(
        id="Founder_Jason_West",
        desc="Jason West was one of the co-founders",
        parent=three_founders_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Jason West was one of the co-founders of {studio_name}.",
        node=founder_west_leaf,
        sources=sources,
        additional_instruction="Confirm the page lists Jason West as a co-founder."
    )

    founder_vince_leaf = evaluator.add_leaf(
        id="Founder_Vince_Zampella",
        desc="Vince Zampella was one of the co-founders",
        parent=three_founders_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Vince Zampella was one of the co-founders of {studio_name}.",
        node=founder_vince_leaf,
        sources=sources,
        additional_instruction="Confirm the page lists Vince Zampella as a co-founder."
    )


async def verify_first_game_details(
    evaluator: Evaluator,
    parent_node,
    extracted: StudioExtraction,
) -> None:
    studio_name = _safe_name(extracted.studio_name, "the identified studio")
    sources = _safe_list(extracted.reference_urls)
    franchise = _safe_franchise(extracted)
    first_game_title = _safe_first_game_title(extracted)

    game_node = evaluator.add_parallel(
        id="First_Game_Details",
        desc="Verify details about the studio's first game in a major franchise",
        parent=parent_node,
        critical=True,
    )

    # Release date: October 29, 2003
    release_leaf = evaluator.add_leaf(
        id="Release_Date",
        desc="The first game was released on October 29, 2003",
        parent=game_node,
        critical=True,
    )
    release_claim = f"The first {franchise} game ({first_game_title}) was released on October 29, 2003."
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=sources,
        additional_instruction="Verify the first installment's release date is October 29, 2003. Minor wording variations are acceptable."
    )

    # Major franchise: studio developed first game
    franchise_leaf = evaluator.add_leaf(
        id="Major_Franchise",
        desc="The game was the first in a major franchise",
        parent=game_node,
        critical=True,
    )
    franchise_claim = f"{studio_name} developed the first game in the {franchise} franchise."
    await evaluator.verify(
        claim=franchise_claim,
        node=franchise_leaf,
        sources=sources,
        additional_instruction="Confirm the page states the studio developed the first game of this franchise (e.g., Call of Duty (2003))."
    )

    # FPS genre
    fps_leaf = evaluator.add_leaf(
        id="FPS_Genre",
        desc="The game is a first-person shooter",
        parent=game_node,
        critical=True,
    )
    fps_claim = f"{first_game_title} (the first {franchise} game) is a first-person shooter."
    await evaluator.verify(
        claim=fps_claim,
        node=fps_leaf,
        sources=sources,
        additional_instruction="Confirm the page describes the game as a first-person shooter."
    )


async def verify_parent_acquisition(
    evaluator: Evaluator,
    parent_node,
    extracted: StudioExtraction,
) -> None:
    sources = _safe_list(extracted.reference_urls)
    parent_company = _safe_parent_company(extracted)

    acq_node = evaluator.add_parallel(
        id="Parent_Acquisition",
        desc="Verify details about the parent company acquisition",
        parent=parent_node,
        critical=True,
    )

    # Acquirer: Microsoft
    acquirer_leaf = evaluator.add_leaf(
        id="Acquirer_Microsoft",
        desc="The parent company was acquired by Microsoft",
        parent=acq_node,
        critical=True,
    )
    acquirer_claim = f"Microsoft acquired {parent_company}."
    await evaluator.verify(
        claim=acquirer_claim,
        node=acquirer_leaf,
        sources=sources,
        additional_instruction="Confirm the page states Microsoft acquired the parent company of the studio (commonly referenced as Activision Blizzard)."
    )

    # Completion date: October 2023
    completion_leaf = evaluator.add_leaf(
        id="Completion_October_2023",
        desc="The acquisition was completed in October 2023",
        parent=acq_node,
        critical=True,
    )
    completion_claim = "The acquisition was completed in October 2023."
    await evaluator.verify(
        claim=completion_claim,
        node=completion_leaf,
        sources=sources,
        additional_instruction="Confirm the page indicates the deal's completion in October 2023."
    )

    # Deal value: approximately $69 billion
    value_leaf = evaluator.add_leaf(
        id="Deal_Value",
        desc="The acquisition deal was valued at approximately $69 billion",
        parent=acq_node,
        critical=True,
    )
    value_claim = "The acquisition deal was valued at approximately $69 billion."
    await evaluator.verify(
        claim=value_claim,
        node=value_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the page states a deal value around $69B (e.g., $68.7B or ~$69B). Allow minor rounding or approximation."
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
) -> Dict:
    """
    Evaluate the agent's answer for the studio identification task.
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
        prompt=prompt_extract_studio_info(),
        template_class=StudioExtraction,
        extraction_name="studio_extraction",
    )

    # Build the main verification node (critical to enforce all criteria)
    studio_node = evaluator.add_parallel(
        id="Studio_Identification",
        desc="Identify the gaming studio that meets all specified criteria",
        parent=root,
        critical=True,
    )

    # Critical existence checks at the top level
    studio_name_ok = bool(_safe_name(extracted.studio_name, "").strip())
    urls_ok = len(_safe_list(extracted.reference_urls)) > 0

    evaluator.add_custom_node(
        result=studio_name_ok,
        id="Studio_Name_Provided",
        desc="Studio name is provided in the answer",
        parent=studio_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_ok,
        id="Reference_URL",
        desc="A reference URL is provided supporting the studio identification",
        parent=studio_node,
        critical=True,
    )

    # Subtree verifications
    await verify_studio_founding(evaluator, studio_node, extracted)
    await verify_first_game_details(evaluator, studio_node, extracted)
    await verify_parent_acquisition(evaluator, studio_node, extracted)

    # Optional: Add ground truth info to aid analysis (not used for scoring)
    evaluator.add_ground_truth({
        "expected_studio_example": "Infinity Ward",
        "expected_first_game": "Call of Duty (2003)",
        "expected_release_date": "October 29, 2003",
        "expected_acquirer": "Microsoft",
        "expected_completion": "October 2023",
        "expected_deal_value": "≈ $69B",
    }, gt_type="reference_expectations")

    # Return structured summary
    return evaluator.get_summary()