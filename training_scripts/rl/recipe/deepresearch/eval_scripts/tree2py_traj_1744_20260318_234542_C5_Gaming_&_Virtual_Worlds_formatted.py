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
TASK_ID = "vr_games_showcase_coop_2026"
TASK_DESCRIPTION = (
    "I am planning to organize regular cooperative VR gaming sessions for my friend group throughout 2026, "
    "and I want to focus on newly released titles that were showcased at major VR gaming events. I'm particularly "
    "interested in games that were featured at the VR Games Showcase that took place on March 24, 2026.\n\n"
    "Please identify four (4) VR games that meet ALL of the following criteria:\n\n"
    "1. VR Games Showcase Feature: The game must have been featured, announced, or showcased during the VR Games "
    "Showcase event on March 24, 2026 (either in the main show or pre-show).\n"
    "2. Cooperative Multiplayer Support: The game must support cooperative multiplayer or team-based gameplay "
    "features, allowing multiple players to play together (not single-player only games).\n"
    "3. 2026 Release Window: The game must be scheduled for release or have already released at some point during "
    "the year 2026.\n"
    "4. Platform Availability: The game must be available or announced for at least one of the following VR platforms: "
    "Meta Quest (Quest 2, Quest 3, or Quest 3S), PlayStation VR2, or PC VR (SteamVR).\n"
    "5. Developer Attribution: The game's developer studio or development company must be publicly identified and "
    "verifiable from official sources.\n"
    "6. Commercial Availability: The game must be (or will be) commercially available for purchase as a full game "
    "product, not merely a free demo, tech preview, or prototype.\n\n"
    "For each game, please provide: the game title, the developer studio name, the supported VR platform(s), a brief "
    "description of the cooperative/multiplayer features, the 2026 release date or release window, confirmation it "
    "was featured at VRGS March 2026, and a reference URL to an official source."
)

EVENT_NAME = "VR Games Showcase March 2026"
EVENT_DATE = "March 24, 2026"
ALLOWED_PLATFORM_HINTS = [
    "Meta Quest", "Quest 2", "Quest 3", "Quest 3S", "PlayStation VR2", "PS VR2", "PC VR", "SteamVR"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GameItem(BaseModel):
    title: Optional[str] = None
    developer: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)
    coop_multiplayer_desc: Optional[str] = None
    release_2026: Optional[str] = None
    vrgs_feature_note: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class GamesExtraction(BaseModel):
    games: List[GameItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_games() -> str:
    return """
    Extract up to four (4) VR games from the answer. For each game, extract the following fields exactly as presented:
    - title: The game's title.
    - developer: The publicly identified developer studio/company (not the publisher, unless they are explicitly the developer).
    - platforms: A list of supported or announced VR platforms as stated (e.g., "Meta Quest 3", "Quest 2", "PS VR2", "SteamVR", "PC VR").
    - coop_multiplayer_desc: A short description of the cooperative or team-based multiplayer features, as described in the answer.
    - release_2026: The 2026 release date or release window mentioned (e.g., "Q2 2026", "late 2026", "2026").
    - vrgs_feature_note: Any text in the answer that states or implies the game was featured/announced/shown at the VR Games Showcase on March 24, 2026.
    - source_urls: A list of the actual URLs cited for this game in the answer. Include all official URLs mentioned, such as:
        • Event/showcase pages or official recaps
        • Official store pages (Meta/PlayStation/Steam)
        • Developer or publisher announcements, press releases, or blog posts
        • Official social media announcements with direct links
      IMPORTANT: Only include URLs that are explicitly present in the answer text. If none are present for a game, return an empty list.

    Rules:
    - Do not invent or infer information not explicitly present in the answer.
    - If a field is missing, set it to null (or an empty list for platforms/source_urls).
    - Preserve the exact phrasing provided in the answer.
    - Return a JSON object with a single field "games", which is an array of up to 4 game objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _pad_or_trim_games(games: List[GameItem], target: int = 4) -> List[GameItem]:
    trimmed = games[:target]
    while len(trimmed) < target:
        trimmed.append(GameItem())
    return trimmed


# --------------------------------------------------------------------------- #
# Verification for one game                                                   #
# --------------------------------------------------------------------------- #
async def verify_single_game(evaluator: Evaluator, parent_node, game: GameItem, game_idx: int) -> None:
    """
    Build verification sub-tree for a single game with the following leaf checks:
      - Reference presence (critical, custom)
      - VRGS feature at March 24, 2026 (critical, URL-grounded)
      - Multiplayer co-op support (critical, URL-grounded)
      - Release in 2026 (critical, URL-grounded)
      - Platform availability among allowed platforms (critical, URL-grounded)
      - Developer publicly identified (critical, URL-grounded)
      - Commercial availability as a full product (critical, URL-grounded)
    """
    game_n = game_idx + 1
    game_node = evaluator.add_parallel(
        id=f"Game_{game_n}",
        desc=f"Game #{game_n} meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Basic URLs presence check (critical)
    has_sources = bool(game.source_urls and len(game.source_urls) > 0)
    evaluator.add_custom_node(
        result=has_sources,
        id=f"Game_{game_n}_Reference",
        desc=f"Verify a valid reference URL is provided for this game",
        parent=game_node,
        critical=True
    )

    # Prepare shared strings
    title_str = game.title or "the game"
    platforms_str = ", ".join(game.platforms) if game.platforms else "unspecified platforms"
    coop_desc_str = game.coop_multiplayer_desc or "cooperative/multiplayer features"
    release_str = game.release_2026 or "2026"
    dev_str = game.developer or "the listed developer"

    # 1) VRGS feature verification
    vrgs_leaf = evaluator.add_leaf(
        id=f"Game_{game_n}_VRGS_Feature",
        desc=f"Verify the game was featured at {EVENT_NAME} ({EVENT_DATE})",
        parent=game_node,
        critical=True
    )
    vrgs_claim = (
        f"The title '{title_str}' was featured, announced, or showcased in the {EVENT_NAME} that occurred on {EVENT_DATE}. "
        "Mentions of pre-show, post-show, official recaps, or the event’s lineup are acceptable if they clearly tie this title to the event."
    )
    await evaluator.verify(
        claim=vrgs_claim,
        node=vrgs_leaf,
        sources=game.source_urls,
        additional_instruction=(
            f"Look for explicit mentions that the game appeared at '{EVENT_NAME}' on {EVENT_DATE}. "
            "Accept terms like 'featured', 'announced', 'revealed', or 'shown' during that show. "
            "If the URLs are unrelated, outdated, or do not mention the event, mark as not supported."
        ),
    )

    # 2) Multiplayer co-op verification
    coop_leaf = evaluator.add_leaf(
        id=f"Game_{game_n}_Multiplayer",
        desc=f"Verify the game supports cooperative multiplayer or team-based gameplay",
        parent=game_node,
        critical=True
    )
    coop_claim = (
        f"The title '{title_str}' supports cooperative multiplayer or team-based gameplay for multiple players. "
        f"Description from the answer: '{coop_desc_str}'."
    )
    await evaluator.verify(
        claim=coop_claim,
        node=coop_leaf,
        sources=game.source_urls,
        additional_instruction=(
            "Confirm that the game explicitly supports co-op or team-based multiplayer (2 or more players playing together in real time). "
            "Do not count asynchronous leaderboards, spectator modes, or single-player only experiences."
        ),
    )

    # 3) Release in 2026 verification
    release_leaf = evaluator.add_leaf(
        id=f"Game_{game_n}_Release_2026",
        desc=f"Verify the game is scheduled for release or released during 2026",
        parent=game_node,
        critical=True
    )
    release_claim = f"The title '{title_str}' is released or scheduled for release in 2026 (e.g., '{release_str}')."
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=game.source_urls,
        additional_instruction=(
            "Accept any explicit 2026 date or window like 'Q1 2026', 'late 2026', or '2026'. "
            "If only a non-2026 timeframe is provided or the date/window is missing, fail this check."
        ),
    )

    # 4) Platform availability verification (allowed platforms)
    platform_leaf = evaluator.add_leaf(
        id=f"Game_{game_n}_Platform",
        desc="Verify the game is available on Meta Quest, PlayStation VR2, or PC VR (SteamVR)",
        parent=game_node,
        critical=True
    )
    platform_claim = (
        f"The title '{title_str}' is announced for at least one of these VR platforms: "
        "Meta Quest (Quest 2/Quest 3/Quest 3S), PlayStation VR2, or PC VR (SteamVR). "
        f"Extracted platforms from the answer: {platforms_str}."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=game.source_urls,
        additional_instruction=(
            "Check the page(s) for platform mentions. Accept synonyms like 'PS VR2' for PlayStation VR2, 'SteamVR' or 'PC VR' for PC, "
            "and 'Meta Quest', 'Quest 2', 'Quest 3', or 'Quest 3S' for Quest. "
            "At least one platform from this allowed set must be clearly indicated."
        ),
    )

    # 5) Developer identification verification
    developer_leaf = evaluator.add_leaf(
        id=f"Game_{game_n}_Developer",
        desc="Verify the developer studio/company is publicly identified",
        parent=game_node,
        critical=True
    )
    developer_claim = (
        f"The publicly identified developer of '{title_str}' is '{dev_str}' (or an equivalent naming/branding of the same studio)."
    )
    await evaluator.verify(
        claim=developer_claim,
        node=developer_leaf,
        sources=game.source_urls,
        additional_instruction=(
            "Confirm the developer (not just publisher) is named on the official source(s). "
            "Minor naming variants or rebrandings are acceptable if they clearly refer to the same studio."
        ),
    )

    # 6) Commercial availability as full product
    commercial_leaf = evaluator.add_leaf(
        id=f"Game_{game_n}_Commercial",
        desc="Verify the game is (or will be) commercially available as a full product (not only a demo/prototype)",
        parent=game_node,
        critical=True
    )
    commercial_claim = (
        f"'{title_str}' is a full commercial game product that is sold (or will be sold), not only a free demo, tech preview, or prototype."
    )
    await evaluator.verify(
        claim=commercial_claim,
        node=commercial_leaf,
        sources=game.source_urls,
        additional_instruction=(
            "Look for store listings, pricing info, 'full release' wording, or official statements indicating a full commercial launch. "
            "If sources only reference a free demo, prototype, alpha, or tech preview with no commitment to a full paid product, fail this check."
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
    Evaluate an answer for the VR Games Showcase March 2026 co-op titles task.
    """
    # Initialize evaluator (root is non-critical to allow partial credit across games)
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

    # Extract up to 4 games from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_games(),
        template_class=GamesExtraction,
        extraction_name="games_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if needed)
    games = _pad_or_trim_games(extraction.games, 4)

    # Add contextual info for transparency
    evaluator.add_custom_info(
        info={
            "event_name": EVENT_NAME,
            "event_date": EVENT_DATE,
            "allowed_platform_hints": ALLOWED_PLATFORM_HINTS
        },
        info_type="task_context",
        info_name="vrgs_criteria_context"
    )

    # Build subtrees for each of the four games
    for idx in range(4):
        await verify_single_game(evaluator, root, games[idx], idx)

    # Return evaluation summary
    return evaluator.get_summary()