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
TASK_ID = "tga2024_best_vr_ar_exclusive_oct2024"
TASK_DESCRIPTION = (
    "Identify the VR game that won the Best VR/AR Game award at The Game Awards 2024. "
    "This game must meet the following criteria: it must be exclusive to a specific VR headset platform "
    "(not available across all VR platforms), and it must have been released in October 2024. "
    "For the identified game, provide: (1) The game's title, (2) A reference URL confirming it won the Best VR/AR Game "
    "award at The Game Awards 2024, (3) The specific VR platform(s) to which the game is exclusive, "
    "(4) A reference URL confirming the platform exclusivity, (5) The game's release date (must be in October 2024), "
    "(6) A reference URL confirming the release date, (7) The name of the game's developer, "
    "(8) The title of another VR game previously developed by the same developer, (9) A reference URL for the developer's previous VR game, "
    "(10) The release month and year of the developer's previous VR game, and (11) A reference URL confirming the previous game's release date."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VRWinnerExtraction(BaseModel):
    # Main game identity and award
    game_title: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)  # URLs confirming Best VR/AR Game at TGA 2024

    # Platform exclusivity
    exclusive_platforms: List[str] = Field(default_factory=list)  # e.g., ["PS VR2"] or ["Meta Quest 3"]
    exclusivity_urls: List[str] = Field(default_factory=list)  # URLs confirming exclusivity

    # Release date (must be in October 2024)
    release_date: Optional[str] = None  # e.g., "October 10, 2024" or "Oct 2024"
    release_date_urls: List[str] = Field(default_factory=list)  # URLs confirming main game's release date

    # Developer information
    developer_name: Optional[str] = None
    developer_urls: List[str] = Field(default_factory=list)  # URLs confirming the developer for the identified game

    # Previous VR game by the same developer
    previous_game_title: Optional[str] = None
    previous_game_urls: List[str] = Field(default_factory=list)  # URLs documenting the previous VR game
    previous_game_release_month_year: Optional[str] = None  # e.g., "March 2022" or "Mar 2022"
    previous_game_release_date_urls: List[str] = Field(default_factory=list)  # URLs confirming previous game's release date


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vr_winner() -> str:
    return """
    Extract the following fields strictly from the provided answer text. Do not invent any values. If a value is missing, return null (for strings) or [] (for arrays).

    Required fields:
    - game_title: The exact title of the VR game identified as the winner.
    - award_urls: A list of URL(s) explicitly confirming that this game won the "Best VR/AR Game" at The Game Awards 2024. Extract only actual URLs mentioned in the answer.
    - exclusive_platforms: A list of the specific VR headset platform(s) (e.g., "PS VR2", "Meta Quest 3", "Apple Vision Pro") to which the game is exclusive as stated in the answer.
    - exclusivity_urls: A list of URL(s) confirming the game's platform exclusivity (e.g., official store page, developer/publisher announcement, or reliable source).
    - release_date: The game's release date as stated in the answer (string; allow formats like "October 10, 2024" or "Oct 2024").
    - release_date_urls: A list of URL(s) confirming the game's release date (store page, press release, or reliable source).
    - developer_name: The name of the game's developer.
    - developer_urls: A list of URL(s) confirming the developer of this identified game.
    - previous_game_title: The title of another VR game previously developed by the same developer.
    - previous_game_urls: A list of URL(s) documenting the previous VR game (developer site/store/reliable source).
    - previous_game_release_month_year: The release month and year of the developer’s previous VR game (e.g., "March 2022").
    - previous_game_release_date_urls: A list of URL(s) confirming the previous game's release month and year.

    Notes:
    - Only extract URLs that are explicitly present in the answer text. If none are present for a field, return an empty list for that URL field.
    - Preserve string values exactly as written in the answer; do not normalize.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _non_empty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst and isinstance(lst, list) and len(lst) > 0)


def _fmt_platforms(platforms: List[str]) -> str:
    if not platforms:
        return ""
    return ", ".join(p.strip() for p in platforms if _non_empty(p))


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_winner_core(
    evaluator: Evaluator,
    parent_node,
    extracted: VRWinnerExtraction
) -> None:
    """
    Winner_Game_Core_Identification (parallel, critical):
    - Game_Title_Provided (existence)
    - Won_Best_VR_AR_TGA_2024 (verify via award URLs)
    - Award_Confirmation_URL (existence of URL)
    """
    node = evaluator.add_parallel(
        id="Winner_Game_Core_Identification",
        desc="Provide the winner game’s identity and verify it won Best VR/AR Game at The Game Awards 2024.",
        parent=parent_node,
        critical=True
    )

    # 1) Title provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.game_title),
        id="Game_Title_Provided",
        desc="The game’s title is explicitly provided.",
        parent=node,
        critical=True
    )

    # 2) Award confirmation URL provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_list(extracted.award_urls),
        id="Award_Confirmation_URL",
        desc="A reference URL is provided confirming the game won Best VR/AR Game at The Game Awards 2024 (official TGA site or reliable source).",
        parent=node,
        critical=True
    )

    # 3) Won Best VR/AR Game at TGA 2024 (critical verification)
    won_leaf = evaluator.add_leaf(
        id="Won_Best_VR_AR_TGA_2024",
        desc="The provided game is confirmed to have won Best VR/AR Game at The Game Awards 2024 (Dec 12, 2024).",
        parent=node,
        critical=True
    )
    title = extracted.game_title or ""
    claim = f"The game '{title}' won the 'Best VR/AR Game' award at The Game Awards 2024."
    await evaluator.verify(
        claim=claim,
        node=won_leaf,
        sources=extracted.award_urls,
        additional_instruction=(
            "Verify that the provided URL(s) explicitly indicate that the named game is the Winner of 'Best VR/AR Game' "
            "for The Game Awards 2024. Allow minor wording variations like 'Best AR/VR' or capitalization differences. "
            "The event date was Dec 12, 2024; some sources may omit the exact date but must clearly refer to the 2024 awards."
        )
    )


async def build_platform_exclusivity(
    evaluator: Evaluator,
    parent_node,
    extracted: VRWinnerExtraction
) -> None:
    """
    Platform_Exclusivity (parallel, critical):
    - Exclusive_Platforms_Listed (existence)
    - Exclusivity_Confirmation_URL (existence)
    - Exclusivity_Condition_Met (verification via exclusivity URLs)
    """
    node = evaluator.add_parallel(
        id="Platform_Exclusivity",
        desc="Verify the game is exclusive to a specific VR headset platform and provide the exclusive platform(s).",
        parent=parent_node,
        critical=True
    )

    # 1) Exclusive platforms listed (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_list(extracted.exclusive_platforms),
        id="Exclusive_Platforms_Listed",
        desc="The specific VR platform(s)/headset ecosystem to which the game is exclusive are explicitly listed.",
        parent=node,
        critical=True
    )

    # 2) Exclusivity confirmation URL provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_list(extracted.exclusivity_urls),
        id="Exclusivity_Confirmation_URL",
        desc="A reference URL is provided (official store/developer statement/reliable source) confirming the platform exclusivity.",
        parent=node,
        critical=True
    )

    # 3) Exclusivity condition met (critical verification)
    exclusivity_leaf = evaluator.add_leaf(
        id="Exclusivity_Condition_Met",
        desc="The game is confirmed to be exclusive to a specific VR headset platform (i.e., not available across all VR platforms).",
        parent=node,
        critical=True
    )
    platforms_str = _fmt_platforms(extracted.exclusive_platforms)
    title = extracted.game_title or ""
    exclusivity_claim = (
        f"The VR game '{title}' is exclusive to the following platform(s): {platforms_str}. "
        f"This means it is only available on {platforms_str} and is not available on other major VR platforms."
    )
    await evaluator.verify(
        claim=exclusivity_claim,
        node=exclusivity_leaf,
        sources=extracted.exclusivity_urls,
        additional_instruction=(
            "Determine whether the cited source(s) substantiate that the game is exclusive to the specified platform(s). "
            "Accept phrasings like 'exclusive', 'only on', or 'only available for' the named headset/platform. "
            "If the source indicates availability on other major VR platforms (e.g., both PS VR2 and Meta Quest, or PCVR broadly), "
            "then this should not be considered exclusive. Timed exclusivity counts as exclusivity if clearly indicated."
        )
    )


async def build_release_date_oct2024(
    evaluator: Evaluator,
    parent_node,
    extracted: VRWinnerExtraction
) -> None:
    """
    Release_Date_October_2024 (parallel, critical):
    - Release_Date_Provided (existence)
    - Main_Game_Release_Date_URL (existence)
    - Release_In_October_2024 (verification via release date URLs)
    """
    node = evaluator.add_parallel(
        id="Release_Date_October_2024",
        desc="Verify the game’s release date and that it falls in October 2024.",
        parent=parent_node,
        critical=True
    )

    # 1) Release date provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.release_date),
        id="Release_Date_Provided",
        desc="A concrete release date for the game is provided (at minimum month and year; may include day).",
        parent=node,
        critical=True
    )

    # 2) Main game release date URL provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_list(extracted.release_date_urls),
        id="Main_Game_Release_Date_URL",
        desc="A reference URL is provided (official store/developer site/reliable source) confirming the game’s release date.",
        parent=node,
        critical=True
    )

    # 3) Release in October 2024 (critical verification)
    in_oct_leaf = evaluator.add_leaf(
        id="Release_In_October_2024",
        desc="The provided/verified release date is in October 2024.",
        parent=node,
        critical=True
    )
    title = extracted.game_title or ""
    claim = f"The game '{title}' was released in October 2024."
    await evaluator.verify(
        claim=claim,
        node=in_oct_leaf,
        sources=extracted.release_date_urls,
        additional_instruction=(
            "Check the source(s) for the game's release date. Confirm the date clearly falls within October 2024. "
            "Accept variations like 'Oct 2024' or a specific day in October 2024. If the source indicates a different month/year, mark as incorrect."
        )
    )


async def build_developer_and_prior(
    evaluator: Evaluator,
    parent_node,
    extracted: VRWinnerExtraction
) -> None:
    """
    Developer_And_Prior_VR_Game (parallel, critical):
    - Developer_Name_Provided (existence)
    - Developer_Confirmation_URL (existence)
    - Prior_VR_Game_By_Same_Developer (parallel, critical)
        - Previous_VR_Game_Title_Provided (existence)
        - Previous_VR_Game_URL (existence)
        - Previous_VR_Game_Release_Month_Year_Provided (existence)
        - Previous_VR_Game_Release_Date_URL (existence)
        - Previous_Game_Predates_Main_Game (verification via simple logic)
    """
    node = evaluator.add_parallel(
        id="Developer_And_Prior_VR_Game",
        desc="Identify the developer with verification and provide a prior VR game by the same developer with release month/year and sources.",
        parent=parent_node,
        critical=True
    )

    # 1) Developer name provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.developer_name),
        id="Developer_Name_Provided",
        desc="The name of the game’s developer is provided.",
        parent=node,
        critical=True
    )

    # 2) Developer confirmation URL provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_list(extracted.developer_urls),
        id="Developer_Confirmation_URL",
        desc="A reference URL is provided confirming the developer of the identified game (official developer/site/store/reliable source).",
        parent=node,
        critical=True
    )

    # 3) Prior VR game details (parallel, critical)
    prior_node = evaluator.add_parallel(
        id="Prior_VR_Game_By_Same_Developer",
        desc="Another VR game previously developed by the same developer is identified.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.previous_game_title),
        id="Previous_VR_Game_Title_Provided",
        desc="The title of the developer’s previous VR game is provided.",
        parent=prior_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_list(extracted.previous_game_urls),
        id="Previous_VR_Game_URL",
        desc="A reference URL is provided documenting the developer’s previous VR game (developer site/store/reliable source).",
        parent=prior_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.previous_game_release_month_year),
        id="Previous_VR_Game_Release_Month_Year_Provided",
        desc="The release month and year of the developer’s previous VR game are provided.",
        parent=prior_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_list(extracted.previous_game_release_date_urls),
        id="Previous_VR_Game_Release_Date_URL",
        desc="A reference URL is provided confirming the previous VR game’s release month and year (developer site/store/reliable source).",
        parent=prior_node,
        critical=True
    )

    # Previous game's release predates the main game's release (critical verification)
    predates_leaf = evaluator.add_leaf(
        id="Previous_Game_Predates_Main_Game",
        desc="The previous VR game’s release date is earlier than the identified winner game’s release date (i.e., it is truly prior).",
        parent=prior_node,
        critical=True
    )
    main_rel = extracted.release_date or ""
    prev_rel = extracted.previous_game_release_month_year or ""
    predates_claim = (
        f"Given the main game's release date '{main_rel}' and the previous VR game's release timeframe '{prev_rel}', "
        f"the previous game's release is earlier than the main game's release."
    )
    await evaluator.verify(
        claim=predates_claim,
        node=predates_leaf,
        additional_instruction=(
            "Compare the two dates using common calendar knowledge. If one date is only month-year (e.g., 'Mar 2022'), "
            "assume it corresponds to the first day of that month for comparison. "
            "Return Correct only if the previous game's release is strictly earlier than the main game's release."
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
    Evaluate an answer for the TGA 2024 Best VR/AR Game (exclusive + Oct 2024 release) task.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; actual flow control under the top critical sequential node
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

    # 2) Extract structured info from the answer
    extracted: VRWinnerExtraction = await evaluator.extract(
        prompt=prompt_extract_vr_winner(),
        template_class=VRWinnerExtraction,
        extraction_name="vr_game_identification"
    )

    # 3) Build verification tree according to rubric
    # Top-level node mirroring rubric root: sequential + critical
    top = evaluator.add_sequential(
        id="Complete_VR_Game_Identification",
        desc="Identify the Best VR/AR Game winner at The Game Awards 2024 that is platform-exclusive and released in Oct 2024; provide all required fields with verifiable sources; and provide a prior VR game by the same developer with release timing and sources.",
        parent=root,
        critical=True
    )

    # Child blocks in order (sequential dependency at top node):
    await build_winner_core(evaluator, top, extracted)
    await build_platform_exclusivity(evaluator, top, extracted)
    await build_release_date_oct2024(evaluator, top, extracted)
    await build_developer_and_prior(evaluator, top, extracted)

    # 4) Return structured summary
    return evaluator.get_summary()