import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# ------------------------------------------------------------
# Task metadata
# ------------------------------------------------------------
TASK_ID = "ivy_league_fb_2025"
TASK_DESCRIPTION = (
    "Which team or teams won the 2025 Ivy League football championship? Provide their final conference records and "
    "overall records. Additionally, describe the outcome of The Game (Harvard vs Yale) played on November 22, 2025, "
    "including the final score and the location where it was played."
)


# ------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------
class ChampionTeam(BaseModel):
    team_name: Optional[str] = None
    conference_record: Optional[str] = None  # Keep as string to allow ranges or variants like "6–1"
    overall_record: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ChampionsExtraction(BaseModel):
    champions: List[ChampionTeam] = Field(default_factory=list)


class TheGameExtraction(BaseModel):
    # Score and team assignment
    harvard_score: Optional[str] = None
    yale_score: Optional[str] = None
    # Location details
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    # Date if mentioned
    date_text: Optional[str] = None  # e.g., "November 22, 2025"
    sources: List[str] = Field(default_factory=list)


# ------------------------------------------------------------
# Extraction prompts
# ------------------------------------------------------------
def prompt_extract_champions() -> str:
    return """
    From the provided answer text, extract ONLY the Ivy League champion team(s) (including co-champions if tied) for the 2025 football season.
    For each identified champion, extract the following fields:
    - team_name: The champion team name (e.g., "Harvard", "Yale", etc.)
    - conference_record: The final Ivy League conference record in W-L form (e.g., "6-1"). Keep exactly as written in the answer if present.
    - overall_record: The final overall record in W-L form (e.g., "9-1"). Keep exactly as written in the answer if present.
    - sources: All URLs (full links) cited in the answer that directly support the champion identification and/or the records. If the answer provides multiple supporting URLs (e.g., Ivy League official standings page, NCAA, ESPN, school sites, news reports), include them all. If no URL is given, return an empty array.

    Return a JSON object with a single field:
    {
      "champions": [
        { "team_name": ..., "conference_record": ..., "overall_record": ..., "sources": [...] },
        ...
      ]
    }

    Notes:
    - Do not infer team names or records. Only extract what's explicitly stated in the answer.
    - If a particular field for a champion is missing in the answer, set it to null (for strings) or [] for the sources list.
    - If the answer lists more teams than the champions, include only those explicitly claimed to be champions/co-champions.
    """


def prompt_extract_the_game() -> str:
    return """
    From the provided answer text, extract the details of The Game (Harvard vs Yale) for 2025, requested by the user:
    - harvard_score: The numeric points Harvard scored in that game (as a string; e.g., "24").
    - yale_score: The numeric points Yale scored in that game (as a string; e.g., "17").
    - venue: The venue/stadium name (e.g., "Yale Bowl", "Harvard Stadium"), if mentioned.
    - city: The city where the game was played (e.g., "New Haven", "Cambridge"), if mentioned.
    - state: The state where the game was played (e.g., "Connecticut", "Massachusetts"), if mentioned.
    - date_text: The date as mentioned in the answer, if any (e.g., "November 22, 2025"); if not mentioned, set to null.
    - sources: All URLs (full links) cited in the answer that directly support the game details (score and/or location). Include all provided links.

    Return a JSON object with these fields. If any field is not present in the answer, return null for strings or [] for the sources list.
    Do not invent or infer information not present in the answer text.
    """


# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    clean = []
    for u in urls:
        if not u:
            continue
        su = u.strip()
        if not su:
            continue
        if su not in seen:
            seen.add(su)
            clean.append(su)
    return clean


def champions_names_list(champs: ChampionsExtraction) -> List[str]:
    return [c.team_name.strip() for c in champs.champions if c.team_name and c.team_name.strip()]


def champions_sources_aggregate(champs: ChampionsExtraction) -> List[str]:
    all_urls: List[str] = []
    for c in champs.champions:
        all_urls.extend(c.sources or [])
    return unique_urls(all_urls)


def champions_conf_records_text(champs: ChampionsExtraction) -> str:
    parts = []
    for c in champs.champions:
        if c.team_name and c.conference_record:
            parts.append(f"{c.team_name}: {c.conference_record}")
    return "; ".join(parts) if parts else ""


def champions_overall_records_text(champs: ChampionsExtraction) -> str:
    parts = []
    for c in champs.champions:
        if c.team_name and c.overall_record:
            parts.append(f"{c.team_name}: {c.overall_record}")
    return "; ".join(parts) if parts else ""


# ------------------------------------------------------------
# Verification builders
# ------------------------------------------------------------
async def build_championship_section(
    evaluator: Evaluator,
    parent_node,
    champs: ChampionsExtraction,
) -> None:
    """
    Build and verify the 'Championship_Teams_and_Records' subtree.
    """
    node_main = evaluator.add_parallel(
        id="Championship_Teams_and_Records",
        desc="Correctly identifies the 2025 Ivy League football champion team(s) and provides their conference and overall records.",
        parent=parent_node,
        critical=True,
    )

    # Optional (but helpful) gate: champions present and sources provided (as a critical sibling).
    # This will cause other leaves to be skipped if absent (using framework's auto preconditions).
    champions_present = len(champs.champions) > 0 and len(champions_names_list(champs)) > 0
    champions_sources_ok = len(champions_sources_aggregate(champs)) > 0
    evaluator.add_custom_node(
        result=champions_present and champions_sources_ok,
        id="Champions_Data_And_Sources_Exist",
        desc="At least one champion team identified with at least one supporting source URL present in the answer.",
        parent=node_main,
        critical=True,
    )

    # 1) Champion_Teams_Correct
    leaf_champions_correct = evaluator.add_leaf(
        id="Champion_Teams_Correct",
        desc="Identifies the correct champion team(s) (including any co-champions tied for the best Ivy League conference record), consistent with the official 2025 Ivy League standings.",
        parent=node_main,
        critical=True,
    )
    champion_names = champions_names_list(champs)
    champion_list_text = ", ".join(champion_names) if champion_names else "None"
    claim_champions = (
        f"For the 2025 Ivy League football season, the champion team(s) were: {champion_list_text}. "
        f"If multiple teams are listed, they were co-champions tied for the best Ivy League conference record."
    )
    await evaluator.verify(
        claim=claim_champions,
        node=leaf_champions_correct,
        sources=champions_sources_aggregate(champs),
        additional_instruction=(
            "Verify the set of champion team(s) for 2025 using the provided URLs (e.g., Ivy League official sites, NCAA, "
            "trusted sports outlets). The claim should align with official 2025 Ivy League standings/champions. "
            "If there are co-champions, ensure all listed teams indeed share the title."
        ),
    )

    # 2) Champion_Conference_Records_Provided_and_Correct
    leaf_conf_records = evaluator.add_leaf(
        id="Champion_Conference_Records_Provided_and_Correct",
        desc="Provides the final Ivy League conference record (W-L) for each identified champion team, and each record is correct.",
        parent=node_main,
        critical=True,
    )
    conf_text = champions_conf_records_text(champs)
    claim_conf = (
        f"The final Ivy League conference records (W-L) for the identified champion team(s) are: {conf_text}."
        if conf_text
        else "No conference records were provided for the identified champion team(s)."
    )
    await evaluator.verify(
        claim=claim_conf,
        node=leaf_conf_records,
        sources=champions_sources_aggregate(champs),
        additional_instruction=(
            "Verify the final Ivy League conference records (league-only W-L) for the listed champion team(s). "
            "If multiple champions are listed, ensure each team's league record matches the evidence."
        ),
    )

    # 3) Champion_Overall_Records_Provided_and_Correct
    leaf_overall_records = evaluator.add_leaf(
        id="Champion_Overall_Records_Provided_and_Correct",
        desc="Provides the final overall record (W-L) for each identified champion team, and each record is correct.",
        parent=node_main,
        critical=True,
    )
    overall_text = champions_overall_records_text(champs)
    claim_overall = (
        f"The final overall records (W-L) for the identified champion team(s) are: {overall_text}."
        if overall_text
        else "No overall records were provided for the identified champion team(s)."
    )
    await evaluator.verify(
        claim=claim_overall,
        node=leaf_overall_records,
        sources=champions_sources_aggregate(champs),
        additional_instruction=(
            "Verify the final overall W-L records (including non-conference games) for the listed champion team(s) "
            "based on the provided webpages."
        ),
    )


async def build_the_game_section(
    evaluator: Evaluator,
    parent_node,
    game: TheGameExtraction,
) -> None:
    """
    Build and verify 'The_Game_Requested_Details' subtree for score and location.
    Note: The rubric includes an optional date mention node. Due to framework constraints on critical parent-child consistency,
    we keep the critical requested details (score/location) under this node and handle the optional date check separately
    (not affecting the main scoring).
    """
    node_main = evaluator.add_parallel(
        id="The_Game_Requested_Details",
        desc="Provides the requested details about The Game (Harvard vs Yale): final score and location.",
        parent=parent_node,
        critical=True,
    )

    # Gate: game sources exist (critical sibling precondition)
    game_sources_ok = len(unique_urls(game.sources)) > 0
    evaluator.add_custom_node(
        result=game_sources_ok,
        id="The_Game_Sources_Exist",
        desc="At least one source URL for The Game details is provided in the answer.",
        parent=node_main,
        critical=True,
    )

    # 1) Location correctness
    leaf_location = evaluator.add_leaf(
        id="Game_Location_Correct",
        desc="States the correct game location (venue and city/state) consistent with authoritative game records.",
        parent=node_main,
        critical=True,
    )
    venue = game.venue or ""
    city = game.city or ""
    state = game.state or ""
    claim_location = (
        f"The 2025 Harvard vs Yale football game ('The Game') was played at {venue}, {city}, {state}."
    )
    await evaluator.verify(
        claim=claim_location,
        node=leaf_location,
        sources=unique_urls(game.sources),
        additional_instruction=(
            "Check the venue and city/state on the provided game sources (official sites, NCAA/game recap pages, "
            "school athletics pages, or reputable outlets). Minor formatting variants are acceptable "
            "(e.g., 'New Haven, CT' vs 'New Haven, Connecticut')."
        ),
    )

    # 2) Final score with team assignment correctness
    leaf_score = evaluator.add_leaf(
        id="Game_Final_Score_Correct_With_Team_Assignment",
        desc="Provides the correct final score for The Game and makes clear which team scored which points (so the outcome is unambiguous).",
        parent=node_main,
        critical=True,
    )
    harvard = game.harvard_score or ""
    yale = game.yale_score or ""
    claim_score = (
        f"In the 2025 Harvard vs Yale game, Harvard scored {harvard} points and Yale scored {yale} points."
    )
    await evaluator.verify(
        claim=claim_score,
        node=leaf_score,
        sources=unique_urls(game.sources),
        additional_instruction=(
            "Verify the final score and team-point assignment using the provided sources. "
            "Ensure it's unambiguous which team scored which points. Allow minor numeric/textual formatting variations."
        ),
    )

    # Optional date mention (non-critical) – handled separately to avoid impacting main critical scoring.
    # We implement it as a separate non-critical subtree under root and use simple verification against the answer text.
    node_optional = evaluator.add_parallel(
        id="The_Game_Optional_Details",
        desc="Optional detail(s) for The Game (should not affect main scoring).",
        parent=parent_node,
        critical=False,
    )
    leaf_date_optional = evaluator.add_leaf(
        id="Game_Date_Mentioned_Optional",
        desc="Optionally mentions the game date (Nov 22, 2025) without contradicting it.",
        parent=node_optional,
        critical=False,
    )
    # The optional check targets the answer text itself (simple verification). It passes if either:
    # - The answer mentions Nov 22, 2025 as the date; OR
    # - The answer does not mention an incorrect/contradictory date.
    claim_date_optional = (
        "The answer either states that The Game (Harvard vs Yale) took place on November 22, 2025, "
        "or it does not mention any contradictory date."
    )
    await evaluator.verify(
        claim=claim_date_optional,
        node=leaf_date_optional,
        sources=None,  # simple verification against the answer text
        additional_instruction=(
            "Check the answer text itself: if it mentions a date, ensure it's 'November 22, 2025'. "
            "If the answer does not mention any date, this is acceptable as long as it doesn't provide a contradictory date."
        ),
    )


# ------------------------------------------------------------
# Main evaluation entrypoint
# ------------------------------------------------------------
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
    Evaluate an answer for the 2025 Ivy League football championship and The Game details.
    """
    # Initialize evaluator with a parallel root
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

    # Extract champions + The Game details concurrently
    champions_task = evaluator.extract(
        prompt=prompt_extract_champions(),
        template_class=ChampionsExtraction,
        extraction_name="champions_extraction",
    )
    game_task = evaluator.extract(
        prompt=prompt_extract_the_game(),
        template_class=TheGameExtraction,
        extraction_name="the_game_extraction",
    )
    champs, game = await asyncio.gather(champions_task, game_task)

    # Top-level critical container mirroring rubric "2025_Ivy_League_Football_Championship"
    # We create a critical parallel node to ensure failure if any essential part fails.
    rubric_root = evaluator.add_parallel(
        id="2025_Ivy_League_Football_Championship",
        desc="Evaluate whether the answer correctly identifies the 2025 Ivy League football champion team(s), provides required records, and reports The Game outcome details (final score and location).",
        parent=root,
        critical=True,
    )

    # Build the two main critical sections
    await build_championship_section(evaluator, rubric_root, champs)
    await build_the_game_section(evaluator, rubric_root, game)

    # Return the evaluator's summary
    return evaluator.get_summary()