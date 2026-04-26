import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ewc2026_esports_crossplay_schedule"
TASK_DESCRIPTION = """
Identify three different esports titles that meet ALL of the following criteria:

1. The game is featured in the official Esports World Cup 2026 tournament lineup (which includes 24 esports titles total)

2. The game supports cross-platform multiplayer gameplay between PC and at least one console platform (PlayStation 5, Xbox Series X/S, or Nintendo Switch) in 2026

3. The game's tournament at the Esports World Cup 2026 is scheduled to take place between July 6-19, 2026, which represents the first two weeks of the event (the full event runs from July 6 to August 23, 2026, in Riyadh, Saudi Arabia)

For each of the three games you identify, provide:
- The complete game title
- Confirmation that it appears in the EWC 2026 official lineup, with a supporting reference URL
- Confirmation of its cross-platform multiplayer support capabilities in 2026, with a supporting reference URL
- The specific tournament start and end dates for that game within the EWC 2026 schedule, with a supporting reference URL from official EWC sources or reputable esports databases
"""

# Window for the first two weeks of EWC 2026
EWC_2026_START = "2026-07-06"
EWC_2026_FIRST_TWO_WEEKS_END = "2026-07-19"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GameItem(BaseModel):
    title: Optional[str] = None
    lineup_urls: List[str] = Field(default_factory=list)
    crossplay_urls: List[str] = Field(default_factory=list)
    schedule_urls: List[str] = Field(default_factory=list)
    tournament_start_date: Optional[str] = None  # e.g., "July 10, 2026" or "2026-07-10"
    tournament_end_date: Optional[str] = None    # e.g., "July 12, 2026" or "2026-07-12"


class GamesExtraction(BaseModel):
    games: List[GameItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_games() -> str:
    return """
Extract up to THREE unique esports games from the provided answer that the agent claims satisfy ALL of the following:
(1) Included in the official Esports World Cup (EWC) 2026 lineup,
(2) Support cross-platform multiplayer between PC and at least one console (PS5, Xbox Series X/S, or Nintendo Switch) in 2026,
(3) The game's EWC 2026 tournament is scheduled during July 6–19, 2026 (first two weeks of the event), with specific start and end dates provided.

For each identified game, return an object with:
- title: The complete game title as presented in the answer (string).
- lineup_urls: An array of URL(s) that the answer cites to confirm the game is in the EWC 2026 official lineup. These should be URLs explicitly present in the answer (e.g., esportsworldcup.com, Liquipedia, Esports.gg, Dot Esports, official publisher announcements). If none are present, return an empty array.
- crossplay_urls: An array of URL(s) explicitly present in the answer that confirm cross-platform multiplayer between PC and at least one of PS5, Xbox Series X/S, or Nintendo Switch in 2026. If none are present, return an empty array.
- schedule_urls: An array of URL(s) explicitly present in the answer that provide the EWC 2026 tournament schedule for the specific game (official EWC site or reputable esports databases). If none are present, return an empty array.
- tournament_start_date: The specific start date for that game's EWC 2026 tournament as stated in the answer (string; keep the original format, e.g., "July 10, 2026" or "2026-07-10"). If not provided, return null.
- tournament_end_date: The specific end date for that game's EWC 2026 tournament as stated in the answer (string; keep the original format). If not provided, return null.

Rules:
- Only extract URLs that are explicitly present in the answer text (plain URLs or markdown links).
- Do not fabricate or infer URLs or dates.
- If the answer lists more than three games, include only the first three in order of appearance.
- If fewer than three are provided, include the ones available and leave missing fields as null or empty arrays as appropriate.

Return a JSON object with a single field:
{ "games": [ ... up to 3 GameItem objects ... ] }
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["first", "second", "third"][n] if 0 <= n < 3 else f"#{n+1}"


def non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification for one game                                                   #
# --------------------------------------------------------------------------- #
async def verify_single_game(
    evaluator: Evaluator,
    parent_node,
    game: GameItem,
    index: int,
) -> None:
    """
    Build verification subtree for one identified game.
    """
    game_idx = index + 1
    display_ord = ordinal(index)

    # Game-level container (parallel, non-critical to allow partial across games)
    game_node = evaluator.add_parallel(
        id=f"Game_{game_idx}",
        desc=f"Evaluation of the {display_ord} identified game",
        parent=parent_node,
        critical=False,
    )

    title_display = game.title if non_empty_str(game.title) else "(missing title)"

    # 1) EWC 2026 Lineup Verification (sequential, critical)
    lineup_node = evaluator.add_sequential(
        id=f"g{game_idx}_EWC_2026_Lineup_Verification",
        desc="Verification that the game is featured in the official Esports World Cup 2026 tournament lineup",
        parent=game_node,
        critical=True,
    )

    # 1.1 Claim: The game is in the EWC 2026 lineup (verify via provided URLs if any)
    lineup_claim_leaf = evaluator.add_leaf(
        id=f"g{game_idx}_Game_In_Official_Lineup",
        desc="The game is confirmed as one of the 24 esports titles in the official EWC 2026 tournament lineup",
        parent=lineup_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{title_display}' is included in the official Esports World Cup 2026 tournament lineup of 24 titles.",
        node=lineup_claim_leaf,
        sources=game.lineup_urls,  # may be empty; presence is checked next
        additional_instruction=(
            "Verify that the provided source(s) explicitly list this game as part of the EWC 2026 lineup. "
            "Prefer official EWC sources (esportsworldcup.com) or reputable esports databases (e.g., Liquipedia). "
            "Minor naming variations are acceptable if clearly the same title."
        ),
    )

    # 1.2 Presence of lineup reference URL(s)
    lineup_url_presence = evaluator.add_custom_node(
        result=bool(game.lineup_urls),
        id=f"g{game_idx}_EWC_Lineup_Reference_URL",
        desc="Reference URL from official EWC sources or reputable esports databases confirming lineup inclusion",
        parent=lineup_node,
        critical=True,
    )

    # 2) Cross-Platform Support Verification (sequential, critical)
    cross_node = evaluator.add_sequential(
        id=f"g{game_idx}_Cross_Platform_Support_Verification",
        desc="Verification that the game supports cross-platform multiplayer gameplay in 2026",
        parent=game_node,
        critical=True,
    )

    # 2.1 Claim: Cross-platform play (PC to at least one console)
    cross_claim_leaf = evaluator.add_leaf(
        id=f"g{game_idx}_Cross_Platform_Play_Confirmed",
        desc="The game supports cross-platform multiplayer between PC and at least one console platform (PS5, Xbox Series X/S, or Nintendo Switch) in 2026",
        parent=cross_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"In 2026, '{title_display}' supports cross-platform multiplayer that includes PC-to-console play with at least "
            "one of PlayStation 5, Xbox Series X/S, or Nintendo Switch."
        ),
        node=cross_claim_leaf,
        sources=game.crossplay_urls,  # may be empty; presence checked next
        additional_instruction=(
            "Confirm that cross-play includes PC and at least one console (PS5, Xbox Series X/S, or Nintendo Switch). "
            "Cross-progression or cross-gen alone does NOT count. "
            "Accept official support pages, FAQs, patch notes, or reputable coverage as evidence."
        ),
    )

    # 2.2 Presence of cross-platform reference URL(s)
    cross_url_presence = evaluator.add_custom_node(
        result=bool(game.crossplay_urls),
        id=f"g{game_idx}_Cross_Platform_Reference_URL",
        desc="Reference URL confirming the game's cross-platform multiplayer support capabilities in 2026",
        parent=cross_node,
        critical=True,
    )

    # 3) Tournament Schedule Specification (sequential, critical)
    sched_node = evaluator.add_sequential(
        id=f"g{game_idx}_Tournament_Schedule_Specification",
        desc="Specification of the game's tournament dates within the EWC 2026 schedule during the first two weeks",
        parent=game_node,
        critical=True,
    )

    # 3.1 Claim: Tournament occurs within July 6–19, 2026 window
    window_claim_leaf = evaluator.add_leaf(
        id=f"g{game_idx}_Tournament_In_July_6_19_Window",
        desc="The game's EWC 2026 tournament occurs between July 6-19, 2026, which is within the first two weeks of the event (July 6-August 23)",
        parent=sched_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The EWC 2026 tournament for '{title_display}' is scheduled entirely within July 6 to July 19, 2026 (inclusive)."
        ),
        node=window_claim_leaf,
        sources=game.schedule_urls,  # may be empty; presence checked below
        additional_instruction=(
            "Use official EWC schedule pages or reputable esports databases. "
            f"Confirm that all competitive days for this title's EWC 2026 tournament fall within {EWC_2026_START} to {EWC_2026_FIRST_TWO_WEEKS_END} inclusive. "
            "If qualifiers outside the event window are listed, focus on the EWC 2026 tournament stage scheduled in Riyadh."
        ),
    )

    # 3.2 Specific dates provided (presence check)
    dates_provided = evaluator.add_custom_node(
        result=non_empty_str(game.tournament_start_date) and non_empty_str(game.tournament_end_date),
        id=f"g{game_idx}_Specific_Dates_Provided",
        desc="Specific start and end dates for the game's EWC 2026 tournament are clearly stated",
        parent=sched_node,
        critical=True,
    )

    # 3.3 Presence of schedule reference URL(s)
    sched_url_presence = evaluator.add_custom_node(
        result=bool(game.schedule_urls),
        id=f"g{game_idx}_Tournament_Schedule_Reference_URL",
        desc="Reference URL from official EWC sources or esports databases providing the tournament schedule",
        parent=sched_node,
        critical=True,
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
    Evaluate an answer for the EWC 2026 esports titles + cross-platform + schedule task.
    Returns a structured evaluation summary dictionary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel across the three games
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

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_games(),
        template_class=GamesExtraction,
        extraction_name="games_extraction",
    )

    # Limit/pad to exactly 3 games
    games: List[GameItem] = list(extracted.games[:3])
    while len(games) < 3:
        games.append(GameItem())

    # Add GT/Context info (not used for scoring, for traceability)
    evaluator.add_ground_truth(
        {
            "ewc_2026_first_two_weeks_window": {
                "start_inclusive": EWC_2026_START,
                "end_inclusive": EWC_2026_FIRST_TWO_WEEKS_END,
            },
            "requirements": {
                "in_official_lineup": True,
                "crossplay_pc_to_console": True,
                "within_first_two_weeks": True,
            },
            "notes": "Sources should ideally include official EWC pages or reputable esports databases. Cross-play must include PC-to-console.",
        },
        gt_type="task_requirements",
    )

    # 2) Build verification tree per game
    verify_tasks = []
    for i in range(3):
        verify_tasks.append(verify_single_game(evaluator, root, games[i], i))
    await asyncio.gather(*verify_tasks)

    # 3) Return summary
    return evaluator.get_summary()