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
TASK_ID = "esports_game_selection_2026"
TASK_DESCRIPTION = (
    "A gaming tournament organizer in Berlin is planning a multi-game esports event scheduled for March 20-22, 2026. "
    "They need to identify three games that meet the following technical and accessibility requirements:\n\n"
    "Requirements:\n"
    "1. Each game must be officially released on at least three of the following four current-generation platforms: "
    "PlayStation 5, Xbox Series X|S, Nintendo Switch 2, or PC (via Steam/Epic Games Store)\n"
    "2. Each game must support cross-platform multiplayer functionality between at least two different platform families "
    "(e.g., PlayStation-Xbox, Console-PC, etc.)\n"
    "3. Each game must have been officially released by March 15, 2026\n"
    "4. For games with PC versions, the recommended system requirements must specify no more than 16GB RAM\n"
    "5. The three selected games must collectively represent at least two distinct gaming genres\n\n"
    "Task: Identify three specific games that satisfy all five requirements listed above. For each game, provide:\n"
    "- Game title\n- Platform availability (specify which platforms)\n"
    "- Cross-platform multiplayer support details (specify which platform combinations)\n"
    "- Official release date\n- PC recommended RAM requirement\n- Gaming genre\n"
    "- Reference URLs supporting each claim"
)
RELEASE_DEADLINE = "2026-03-15"  # March 15, 2026


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GameItem(BaseModel):
    title: Optional[str] = None
    platforms: List[str] = Field(default_factory=list, description="Canonical platform names: PS5, Xbox Series X|S, Nintendo Switch 2, PC")
    crossplay: List[str] = Field(default_factory=list, description="List of cross-play pairs like 'PS5-Xbox', 'PS5-PC', 'Xbox-PC', 'Switch-PC', etc.")
    release_date: Optional[str] = None
    pc_recommended_ram: Optional[str] = None
    genre: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class GamesExtraction(BaseModel):
    games: List[GameItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_games() -> str:
    return """
    Extract exactly up to three games described in the answer that the author proposes for the tournament and return structured info for each. 
    If the answer lists more than three games, return only the first three in the order mentioned. If fewer than three are present, return as many as available.

    For each game, extract the following fields:
    - title: The game title as written.
    - platforms: A list of which of the following platforms the game is stated to be officially available on. Canonicalize using ONLY these exact tokens: 
      "PS5", "Xbox Series X|S", "Nintendo Switch 2", "PC".
      Examples of mapping:
        • "PlayStation 5" -> PS5
        • "Xbox Series X", "Xbox Series S", "Xbox Series" -> Xbox Series X|S
        • "Switch 2", "Nintendo Switch (2nd gen)", "Nintendo Switch 2" -> Nintendo Switch 2
        • "PC via Steam", "PC (Epic Games Store)", "Windows (Steam/Epic)" -> PC
      If a platform is not clearly stated, do not include it.
    - crossplay: A list of cross-platform multiplayer pairs across different platform families, using hyphen-separated family names. Allowed pairs include:
        "PS5-Xbox", "PS5-PC", "Xbox-PC", "Switch-PS5", "Switch-Xbox", "Switch-PC".
      Include all pairs the answer claims are supported. If the answer says "full crossplay across all platforms", list all applicable pairs across distinct families.
      If crossplay is not described, return an empty list.
    - release_date: The official release date (use the earliest official release across platforms if multiple are given). Use the date format from the answer.
    - pc_recommended_ram: The recommended RAM requirement for the PC version (e.g., "16 GB RAM", "8GB", "12 GB"). If no PC version or no recommended RAM is specified, set to null.
    - genre: The primary genre classification (e.g., "shooter", "sports", "fighting", "racing", "battle royale", "extraction"). Use the genre presented in the answer; if multiple are given, pick the primary one.
    - reference_urls: All URLs that the answer provides as sources for this game (official sites, store pages, platform availability pages, crossplay documentation, news/announcement pages, etc.). 
      Extract only actual URLs that appear in the answer (including markdown links). If none are provided, return an empty list.

    Important:
    - Do not invent information. Only extract what is explicitly stated in the answer.
    - Keep platforms canonicalized exactly as specified.
    - Keep at most three games in total.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_pc_platform(platforms: List[str]) -> bool:
    return any(p.strip().lower() == "pc" for p in platforms)


def union_urls(games: List[GameItem]) -> List[str]:
    seen = set()
    merged = []
    for g in games:
        for u in g.reference_urls:
            if isinstance(u, str) and u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_game(
    evaluator: Evaluator,
    parent_node,
    game: GameItem,
    game_index: int
) -> None:
    """
    Build the verification subtree for a single game and run verifications.
    """
    # Container for the game (parallel aggregation; criticality managed at leaf level)
    game_node = evaluator.add_parallel(
        id=f"game_{game_index}_compliance",
        desc=f"Evaluation of the game #{game_index + 1} compliance with all tournament requirements",
        parent=parent_node,
        critical=False
    )

    title = game.title or f"Game #{game_index + 1}"
    platforms_str = ", ".join(game.platforms) if game.platforms else "none listed"
    crossplay_str = ", ".join(game.crossplay) if game.crossplay else "none listed"
    urls = game.reference_urls if game.reference_urls else []

    # 1) Platform Requirements
    plat_leaf = evaluator.add_leaf(
        id=f"game_{game_index}_platform_requirements",
        desc="Game is officially released on at least three of PS5, Xbox Series X|S, Nintendo Switch 2, or PC; supported by sources",
        parent=game_node,
        critical=True
    )
    plat_claim = (
        f"The game '{title}' is officially available on at least three of the following platforms: "
        f"PlayStation 5, Xbox Series X|S, Nintendo Switch 2, and PC (via Steam or Epic Games Store). "
        f"The answer lists availability on: {platforms_str}."
    )
    plat_instruction = (
        "Verify using the provided URLs that the game is officially available on at least three of the specified platforms. "
        "Allow reasonable naming variants (e.g., 'PlayStation 5' ≈ PS5; 'Xbox Series X' or 'Series S' ≈ Xbox Series X|S; 'Switch 2' ≈ Nintendo Switch 2). "
        "For PC, prefer Steam or Epic Games Store pages; an official publisher PC page is acceptable if clearly indicating PC availability. "
        "If the answer provides no URLs supporting platform availability, or sources do not confirm at least three platforms, mark as Incorrect."
    )
    await evaluator.verify(
        claim=plat_claim,
        node=plat_leaf,
        sources=urls,
        additional_instruction=plat_instruction
    )

    # 2) Cross-Platform Support
    cps_leaf = evaluator.add_leaf(
        id=f"game_{game_index}_cross_platform_support",
        desc="Game supports cross-platform multiplayer across at least two different platform families; supported by sources",
        parent=game_node,
        critical=True
    )
    cps_claim = (
        f"The game '{title}' supports cross-platform multiplayer between at least two different platform families "
        f"(e.g., PlayStation-Xbox, PlayStation-PC, Xbox-PC, Switch-PS5, Switch-Xbox, or Switch-PC). "
        f"The answer lists cross-play pairs: {crossplay_str}."
    )
    cps_instruction = (
        "Verify that multiplayer cross-play (not just cross-progression) exists between at least one pair of different platform families. "
        "Accept explicit statements such as 'full crossplay across PlayStation, Xbox, and PC' or a publisher's support article listing cross-play platforms. "
        "If sources are absent or only indicate cross-progression/cross-save without multiplayer cross-play, mark as Incorrect."
    )
    await evaluator.verify(
        claim=cps_claim,
        node=cps_leaf,
        sources=urls,
        additional_instruction=cps_instruction
    )

    # 3) Release Timeline
    rel_leaf = evaluator.add_leaf(
        id=f"game_{game_index}_release_timeline",
        desc=f"Game was officially released on or before {RELEASE_DEADLINE}; supported by sources",
        parent=game_node,
        critical=True
    )
    rel_claim = (
        f"The game '{title}' was officially released on or before {RELEASE_DEADLINE}."
        " Use the earliest official release across platforms if multiple are present; disregard early-access betas."
    )
    rel_instruction = (
        f"Confirm using official pages, platform store pages, or reputable announcements that the official release date "
        f"(not early access) is on or before {RELEASE_DEADLINE}. If sources are missing or unclear, mark as Incorrect."
    )
    await evaluator.verify(
        claim=rel_claim,
        node=rel_leaf,
        sources=urls,
        additional_instruction=rel_instruction
    )

    # 4) PC System Requirements (recommended RAM <= 16GB) – only applies if PC version exists
    sys_leaf = evaluator.add_leaf(
        id=f"game_{game_index}_system_requirements",
        desc="PC version's recommended RAM is 16 GB or less, or requirement is N/A if no PC version; supported by sources",
        parent=game_node,
        critical=True
    )

    if has_pc_platform(game.platforms):
        sys_claim = (
            f"The PC version of '{title}' has a recommended RAM requirement of 16 GB or less. "
            f"The answer's stated recommended RAM is: {game.pc_recommended_ram or 'unspecified'}."
        )
        sys_instruction = (
            "Verify on official PC requirement sources (e.g., Steam, Epic Games Store, or official publisher pages) "
            "that the recommended RAM is 16 GB or less. If the recommended RAM is greater than 16 GB, "
            "or the sources do not provide a recommended RAM value, mark as Incorrect."
        )
    else:
        # Not applicable -> we still verify with clear instruction to consider it satisfied when no PC version
        sys_claim = (
            f"The requirement about PC recommended RAM is not applicable to '{title}' because the game is not listed as available on PC."
        )
        sys_instruction = (
            "If the provided platforms do not include PC, consider this check satisfied (Correct). "
            "If sources clearly indicate a PC version exists contrary to the answer's platforms, mark as Incorrect."
        )

    await evaluator.verify(
        claim=sys_claim,
        node=sys_leaf,
        sources=urls,
        additional_instruction=sys_instruction
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
    Evaluate an answer for the esports multi-platform game selection task.
    """
    # Initialize evaluator (root is non-critical to allow a mix of critical and non-critical children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluation of tournament game selection meeting all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract up to 3 games from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_games(),
        template_class=GamesExtraction,
        extraction_name="games_extraction"
    )

    games: List[GameItem] = extracted.games[:3] if extracted.games else []

    # Root child: Minimum_Game_Count (critical)
    min_count_node = evaluator.add_custom_node(
        result=(len(games) >= 3),
        id="minimum_game_count",
        desc="Verification that at least three games are provided in the answer (we evaluate the first three).",
        parent=root,
        critical=True
    )

    # Root child: Individual_Game_Evaluations (parallel, non-critical)
    indiv_parent = evaluator.add_parallel(
        id="individual_game_evaluations",
        desc="Assessment of whether each proposed game meets all technical and availability requirements",
        parent=root,
        critical=False
    )

    # Ensure we always build exactly 3 verification branches (pad with empty items if needed)
    while len(games) < 3:
        games.append(GameItem())

    # Verify each game
    for idx, game in enumerate(games[:3]):
        await verify_single_game(evaluator, indiv_parent, game, idx)

    # Root child: Genre_Diversity_Requirement (critical leaf)
    genres = [g.genre for g in games[:3]]
    titles = [g.title or f"Game #{i+1}" for i, g in enumerate(games[:3])]
    combined_urls = union_urls(games[:3])

    genre_leaf = evaluator.add_leaf(
        id="genre_diversity_requirement",
        desc="The three selected games collectively represent at least two distinct gaming genres; supported by sources",
        parent=root,
        critical=True
    )

    # Construct claim summarizing titles and genres
    trio_summary_parts = []
    for i in range(3):
        trio_summary_parts.append(f"'{titles[i]}' is categorized as '{genres[i] or 'unspecified'}'")
    trio_summary = "; ".join(trio_summary_parts)

    genre_claim = (
        f"Across the three selected games, there are at least two distinct genres. Specifically: {trio_summary}."
    )
    genre_instruction = (
        "Use the provided sources (official sites, platform store pages, publisher pages, or reputable listings) to confirm the listed genre for each game. "
        "Allow minor naming variations (e.g., 'first-person shooter' ≈ 'shooter'). "
        "If all three games fall under the same genre or if the sources cannot confirm at least two distinct genres, mark as Incorrect."
    )
    await evaluator.verify(
        claim=genre_claim,
        node=genre_leaf,
        sources=combined_urls,
        additional_instruction=genre_instruction
    )

    # Add contextual custom info
    evaluator.add_custom_info(
        info={
            "evaluation_deadline": RELEASE_DEADLINE,
            "games_extracted_count": len(extracted.games) if extracted.games else 0,
            "games_used_for_eval": [g.title for g in games[:3]]
        },
        info_type="context",
        info_name="evaluation_context"
    )

    return evaluator.get_summary()