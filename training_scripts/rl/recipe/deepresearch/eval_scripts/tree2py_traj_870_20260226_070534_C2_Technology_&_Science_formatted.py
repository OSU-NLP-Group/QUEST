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
TASK_ID = "psplus_racing_jan2026"
TASK_DESCRIPTION = (
    "In January 2026, PlayStation Plus Essential tier added three new games to its monthly lineup. "
    "One of these games is a racing game. Identify this racing game and provide the following information: "
    "(1) the exact game title, (2) the PlayStation console platform(s) on which this game is available through "
    "PS Plus Essential, and (3) the specific date when this game became available to PlayStation Plus subscribers."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RacingGameInfo(BaseModel):
    title: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)
    availability_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_racing_game_info() -> str:
    return """
    From the provided answer, extract information about the single racing game that is part of the PlayStation Plus Essential
    Monthly Games lineup for January 2026. Return:
    - title: the exact game title for the racing game identified
    - platforms: a list of PlayStation console platform names (e.g., "PS5", "PlayStation 5", "PS4", "PlayStation 4") on which this game is available through PS Plus Essential in January 2026, as stated in the answer
    - availability_date: the specific calendar date when this game became available to PlayStation Plus Essential subscribers in January 2026 (the go-live date to add to library), exactly as stated in the answer
    - sources: an array of all URLs cited in the answer that support the identification of the game, its platforms, or the availability date. Extract actual URLs only (including those in markdown links).
    
    Important:
    - If the answer mentions multiple games, select only the one that is a racing game.
    - If a required field is not explicitly present in the answer, return null for it (or an empty list for platforms/sources).
    - Do not invent or infer URLs. Only include URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_nonempty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst and any((x or "").strip() for x in lst))


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, info: RacingGameInfo) -> None:
    """
    Build the verification tree according to the rubric and run the necessary verifications.
    """
    # Top-level critical sequential node mirroring the rubric's root
    main_node = evaluator.add_sequential(
        id="PS_Plus_Racing_Game_Info",
        desc="Complete and accurate information about the racing game added to PlayStation Plus Essential tier in January 2026",
        parent=parent_node,
        critical=True
    )

    # ------------------------ Game Identification ------------------------ #
    game_ident_node = evaluator.add_parallel(
        id="Game_Identification",
        desc="Correct identification of the racing game included in PS Plus Essential monthly games for January 2026",
        parent=main_node,
        critical=True
    )

    # Existence checks (critical gating)
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.title),
        id="Title_Provided",
        desc="The racing game title is provided in the answer",
        parent=game_ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_list(info.sources),
        id="Sources_Provided",
        desc="At least one supporting source URL is provided in the answer",
        parent=game_ident_node,
        critical=True
    )

    # Leaf: Game_Title (verification that the identified title is indeed part of PS Plus Essential Monthly Games in Jan 2026)
    node_game_title = evaluator.add_leaf(
        id="Game_Title",
        desc="The exact title of the racing game",
        parent=game_ident_node,
        critical=True
    )
    title_for_claim = info.title or ""
    claim_inclusion = (
        f"'{title_for_claim}' was included in the PlayStation Plus Essential 'Monthly Games' lineup for January 2026."
    )
    await evaluator.verify(
        claim=claim_inclusion,
        node=node_game_title,
        sources=info.sources,
        additional_instruction=(
            "Verify that the provided source(s) explicitly list the PlayStation Plus Essential Monthly Games "
            "for January 2026 and include this exact game title (allow minor punctuation or subtitle variations). "
            "If January 2026 is not shown or the title is not present, the claim is not supported."
        ),
    )

    # Leaf: Racing genre confirmation (explicitly ensure it is a racing game)
    node_genre = evaluator.add_leaf(
        id="Racing_Genre_Confirmed",
        desc="The identified game is a racing game",
        parent=game_ident_node,
        critical=True
    )
    claim_genre = f"The game '{title_for_claim}' is a racing video game (belongs to the racing genre)."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=info.sources,
        additional_instruction=(
            "Accept descriptors such as 'racing', 'racer', 'arcade racer', 'racing simulator', 'kart racing', etc. "
            "The sources should clearly indicate or imply that the game is in the racing genre."
        ),
    )

    # --------------------------- Game Details ---------------------------- #
    details_node = evaluator.add_parallel(
        id="Game_Details",
        desc="Availability details for the identified racing game on PlayStation Plus",
        parent=main_node,
        critical=True
    )

    # Existence checks (critical gating for details)
    evaluator.add_custom_node(
        result=_has_nonempty_list(info.platforms),
        id="Platforms_Provided",
        desc="The answer includes PlayStation platform(s) for the game",
        parent=details_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.availability_date),
        id="Availability_Date_Provided",
        desc="The answer provides the date when the game became available to PS Plus subscribers",
        parent=details_node,
        critical=True
    )

    # Leaf: Platform_Availability
    node_platforms = evaluator.add_leaf(
        id="Platform_Availability",
        desc="The PlayStation console platform(s) on which the game is available through PS Plus Essential",
        parent=details_node,
        critical=True
    )
    platforms_text = ", ".join([p.strip() for p in info.platforms if (p or "").strip()]) if info.platforms else ""
    claim_platforms = (
        f"On PlayStation Plus Essential in January 2026, the game '{title_for_claim}' is available on the following "
        f"platform(s): {platforms_text}."
    )
    await evaluator.verify(
        claim=claim_platforms,
        node=node_platforms,
        sources=info.sources,
        additional_instruction=(
            "Verify the platform(s) listed for this title specifically in the January 2026 PlayStation Plus Monthly Games "
            "post or other official PlayStation sources. Treat 'PS5' and 'PlayStation 5' as equivalent, and similarly for 'PS4' "
            "and 'PlayStation 4'. Consider the claim correct only if the set of platforms matches what the source indicates "
            "(allowing minor formatting differences)."
        ),
    )

    # Leaf: Availability_Date
    node_date = evaluator.add_leaf(
        id="Availability_Date",
        desc="The specific date when the game became available to PlayStation Plus Essential subscribers",
        parent=details_node,
        critical=True
    )
    date_text = info.availability_date or ""
    claim_date = (
        f"The January 2026 PlayStation Plus Essential Monthly Games, including '{title_for_claim}', became available "
        f"to claim on {date_text}."
    )
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=info.sources,
        additional_instruction=(
            "Check the go-live date when the January 2026 Monthly Games became available to add to the library "
            "for PS Plus Essential subscribers (not the announcement date). Accept equivalent date formats "
            "(e.g., 'January 6, 2026' vs '6 January 2026')."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the PS Plus January 2026 racing game task.
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
        default_model=model
    )

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_racing_game_info(),
        template_class=RacingGameInfo,
        extraction_name="racing_game_info"
    )

    # Build and run verifications according to rubric
    await build_verification_tree(evaluator, root, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()