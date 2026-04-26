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
TASK_ID = "goty_2025_dual_awards"
TASK_DESCRIPTION = (
    "Research the video game that won Game of the Year at both The Game Awards 2025 (held December 11, 2025) "
    "and the IGN Awards 2025 (published December 21, 2025). Provide the following information about this game:\n\n"
    "1. The exact title of the game\n"
    "2. Its release date\n"
    "3. All platforms it is available on\n"
    "4. The name of the developer/publisher\n"
    "5. At least two other award categories it won at The Game Awards 2025 (beyond Game of the Year)\n"
    "6. At least two other award categories it won at the IGN Awards 2025 (beyond Best Game of 2025)\n"
    "7. Reference URLs that verify the award wins at both award shows\n\n"
    "Ensure all information is accurate and verifiable from official sources."
)

TGA_EVENT_DATE = "December 11, 2025"
IGN_PUBLICATION_DATE = "December 21, 2025"

AUTHORITATIVE_DOMAINS = [
    "thegameawards.com",
    "ign.com",
    "gamespot.com",
    "polygon.com",
    "eurogamer.net",
    "pcgamer.com",
    "gameinformer.com",
    "rockpapershotgun.com",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GameExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    title: Optional[str] = None
    release_date: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)
    developer: Optional[str] = None
    publisher: Optional[str] = None

    # Award categories beyond overall GOTY/Best Game
    tga_additional_awards: List[str] = Field(default_factory=list)
    ign_additional_awards: List[str] = Field(default_factory=list)

    # Reference URLs intended to verify award wins
    tga_urls: List[str] = Field(default_factory=list)
    ign_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_game_info() -> str:
    return (
        "Extract the single game and all requested details exactly as stated in the answer.\n"
        "Return a JSON object with the following fields:\n"
        "- title: exact game title (string)\n"
        "- release_date: official launch release date as given in the answer (string)\n"
        "- platforms: list of all platforms stated (strings, e.g., 'PC', 'PlayStation 5', 'Xbox Series X|S', 'Nintendo Switch')\n"
        "- developer: developer name if stated (string or null)\n"
        "- publisher: publisher name if stated (string or null)\n"
        "- tga_additional_awards: list of other categories the game WON at The Game Awards 2025, EXCLUDING 'Game of the Year' (strings)\n"
        "- ign_additional_awards: list of other categories the game WON at the IGN Awards 2025, EXCLUDING 'Best Game of 2025' (strings)\n"
        "- tga_urls: list of URL(s) provided in the answer that verify the Game of the Year win at The Game Awards 2025\n"
        "- ign_urls: list of URL(s) provided in the answer that verify the Best Game of 2025 win at the IGN Awards 2025\n\n"
        "Important:\n"
        "1) Only extract URLs explicitly present in the answer text (plain or markdown link). Do not invent URLs.\n"
        "2) For both tga_additional_awards and ign_additional_awards, include ONLY categories that the game WON (not nominations) beyond the overall GOTY/Best Game.\n"
        "3) If any item is missing, use null for strings or empty list for arrays.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_string(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""

def _urls_nonempty(urls: List[str]) -> bool:
    return isinstance(urls, list) and len([u for u in urls if _looks_like_url(u)]) > 0

def _looks_like_url(u: Optional[str]) -> bool:
    if not isinstance(u, str):
        return False
    x = u.strip().lower()
    return x.startswith("http://") or x.startswith("https://")

def _contains_authoritative_domain(urls: List[str]) -> bool:
    """Check if any URL is from an authoritative domain."""
    for u in urls:
        if not isinstance(u, str):
            continue
        low = u.lower()
        for dom in AUTHORITATIVE_DOMAINS:
            if dom in low:
                return True
    return False


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_game_title_checks(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    game_title node: Provide the exact title of the game.
    Implemented as a critical leaf existence check (title provided).
    """
    evaluator.add_custom_node(
        result=_non_empty_string(info.title),
        id="game_title",
        desc="Provide the exact title of the game.",
        parent=parent,
        critical=True
    )


async def build_award_verification(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    Verify award wins at both shows for the same game.
    Critical parallel node containing:
    - tga_goty_win: The game won TGA 2025 Game of the Year
    - ign_best_game_win: The game won IGN Awards 2025 Best Game of 2025
    - same_game_constraint: Answer consistently identifies the same title for both shows
    """
    node = evaluator.add_parallel(
        id="award_verification",
        desc="Verify award wins at both shows for the same game.",
        parent=parent,
        critical=True
    )

    # TGA GOTY verification
    tga_node = evaluator.add_leaf(
        id="tga_goty_win",
        desc=f"Shows that the named game won Game of the Year at The Game Awards 2025 ({TGA_EVENT_DATE}).",
        parent=node,
        critical=True
    )
    tga_claim = (
        f"The game titled '{info.title}' won 'Game of the Year' at The Game Awards 2025."
    )
    await evaluator.verify(
        claim=tga_claim,
        node=tga_node,
        sources=info.tga_urls,
        additional_instruction=(
            "Confirm that the page explicitly lists the game as the 'Game of the Year' winner for 2025. "
            "Ignore pages that only mention nominations or predictions. Minor title formatting differences are acceptable."
        )
    )

    # IGN Best Game verification
    ign_node = evaluator.add_leaf(
        id="ign_best_game_win",
        desc=f"Shows that the named game won IGN Awards 2025 'Best Game of 2025' ({IGN_PUBLICATION_DATE}).",
        parent=node,
        critical=True
    )
    ign_claim = (
        f"The game titled '{info.title}' won 'Best Game of 2025' in the IGN Awards 2025 (overall GOTY equivalent)."
    )
    await evaluator.verify(
        claim=ign_claim,
        node=ign_node,
        sources=info.ign_urls,
        additional_instruction=(
            "Confirm that the IGN Awards page explicitly names the game as 'Best Game of 2025'. "
            "Ignore articles about nominations or runner-ups. Minor title formatting differences are acceptable."
        )
    )

    # Same game constraint - internal consistency check based on the answer text
    same_game_node = evaluator.add_leaf(
        id="same_game_constraint",
        desc="The game identified as the winner is the same title for both award shows.",
        parent=node,
        critical=True
    )
    same_game_claim = (
        "In the provided answer text, the game identified as winning The Game Awards 2025 and the IGN Awards 2025 "
        "is the same single title (no mismatch between the two shows)."
    )
    await evaluator.verify(
        claim=same_game_claim,
        node=same_game_node,
        additional_instruction=(
            "Check the answer text for internal consistency: the same game title must be referenced for both shows. "
            "Allow minor formatting variations (punctuation, capitalization, subtitles)."
        )
    )


async def build_release_date_checks(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    release_date node: Provide the official launch release date, and it must be in 2025.
    Implemented as a single critical leaf verified logically (simple_verify).
    """
    release_leaf = evaluator.add_leaf(
        id="release_date",
        desc="Provide the official launch release date, and it must be in 2025.",
        parent=parent,
        critical=True
    )
    rd = info.release_date or ""
    release_claim = (
        f"The game's release date is stated as '{rd}', and the year indicated is 2025."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        additional_instruction=(
            "Judge based on the answer text whether the provided release date falls in 2025. "
            "If the date format is natural language (e.g., 'October 2025'), that is acceptable."
        )
    )


async def build_platform_checks(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    platform_availability node: List all platforms the game is available on.
    Implemented as critical existence check (at least one platform listed).
    """
    evaluator.add_custom_node(
        result=bool(info.platforms),
        id="platform_availability",
        desc="List all platforms the game is available on (console and PC platforms), without omitting any platform supported per authoritative sources.",
        parent=parent,
        critical=True
    )


async def build_devpub_checks(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    developer_or_publisher node: Correctly identify the developer and/or publisher.
    Implemented as critical existence check (developer or publisher provided).
    """
    evaluator.add_custom_node(
        result=_non_empty_string(info.developer) or _non_empty_string(info.publisher),
        id="developer_or_publisher",
        desc="Correctly identify the developer and/or publisher of the game.",
        parent=parent,
        critical=True
    )


async def build_additional_tga_awards(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    additional_tga_awards node: Identify at least two additional award categories (beyond GOTY) won at TGA 2025.
    Implemented as a critical parallel node with:
    - Count check (>=2)
    - URL-supported verification of the specific categories
    """
    node = evaluator.add_parallel(
        id="additional_tga_awards",
        desc="Identify at least two additional award categories (beyond Game of the Year) that the game won at The Game Awards 2025.",
        parent=parent,
        critical=True
    )

    # Count check
    count_leaf = evaluator.add_custom_node(
        result=len(info.tga_additional_awards) >= 2,
        id="tga_awards_at_least_two",
        desc="At least two additional TGA 2025 award categories are provided (excluding Game of the Year).",
        parent=node,
        critical=True
    )

    # URL-supported verification
    verify_leaf = evaluator.add_leaf(
        id="tga_awards_supported",
        desc="The listed TGA 2025 additional award categories are supported by the cited sources.",
        parent=node,
        critical=True
    )
    awards_str = ", ".join(info.tga_additional_awards) if info.tga_additional_awards else ""
    tga_awards_claim = (
        f"The game won the following additional categories at The Game Awards 2025 (excluding Game of the Year): {awards_str}."
    )
    await evaluator.verify(
        claim=tga_awards_claim,
        node=verify_leaf,
        sources=info.tga_urls,
        additional_instruction=(
            "Confirm that each listed category is a WIN for the named game at The Game Awards 2025. "
            "Reject nominations or non-winning mentions. Category title wording may have minor variations."
        )
    )


async def build_additional_ign_awards(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    additional_ign_awards node: Identify at least two additional award categories (beyond Best Game) won at IGN Awards 2025.
    Implemented as a critical parallel node with:
    - Count check (>=2)
    - URL-supported verification of the specific categories
    """
    node = evaluator.add_parallel(
        id="additional_ign_awards",
        desc="Identify at least two additional award categories (beyond Best Game of 2025) that the game won at the IGN Awards 2025.",
        parent=parent,
        critical=True
    )

    # Count check
    count_leaf = evaluator.add_custom_node(
        result=len(info.ign_additional_awards) >= 2,
        id="ign_awards_at_least_two",
        desc="At least two additional IGN Awards 2025 categories are provided (excluding Best Game of 2025).",
        parent=node,
        critical=True
    )

    # URL-supported verification
    verify_leaf = evaluator.add_leaf(
        id="ign_awards_supported",
        desc="The listed IGN Awards 2025 additional award categories are supported by the cited sources.",
        parent=node,
        critical=True
    )
    awards_str = ", ".join(info.ign_additional_awards) if info.ign_additional_awards else ""
    ign_awards_claim = (
        f"The game won the following additional categories at the IGN Awards 2025 (excluding Best Game of 2025): {awards_str}."
    )
    await evaluator.verify(
        claim=ign_awards_claim,
        node=verify_leaf,
        sources=info.ign_urls,
        additional_instruction=(
            "Confirm that each listed category is a WIN for the named game at the IGN Awards 2025. "
            "Reject nominations or non-winning mentions. Category title wording may have minor variations."
        )
    )


async def build_reference_urls(evaluator: Evaluator, parent, info: GameExtraction) -> None:
    """
    reference_urls node: Provide reference URLs that verify the award wins at both shows and are authoritative.
    Critical parallel node containing:
    - tga_reference_url: at least one valid URL provided for TGA win (basic validity + verification)
    - ign_reference_url: at least one valid URL provided for IGN win (basic validity + verification)
    - source_authoritativeness: subnode ensuring domains are authoritative for both sets
    """
    ref_node = evaluator.add_parallel(
        id="reference_urls",
        desc="Provide reference URLs that verify the award wins at both shows and are official or authoritative sources.",
        parent=parent,
        critical=True
    )

    # TGA URL provided (basic validity)
    tga_ref_exists = evaluator.add_custom_node(
        result=_urls_nonempty(info.tga_urls),
        id="tga_reference_url",
        desc="Provide a valid URL that supports The Game Awards 2025 Game of the Year win claim for the named game.",
        parent=ref_node,
        critical=True
    )
    # Also verify via content (redundant check but aligns with rubric wording)
    tga_ref_verify = evaluator.add_leaf(
        id="tga_reference_url_supports_win",
        desc="The provided TGA reference URL(s) explicitly support the GOTY win for the named game.",
        parent=ref_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm that '{info.title}' won Game of the Year at The Game Awards 2025.",
        node=tga_ref_verify,
        sources=info.tga_urls,
        additional_instruction=(
            "Ensure the page clearly shows 'Game of the Year' and the named game as winner at The Game Awards 2025."
        )
    )

    # IGN URL provided (basic validity)
    ign_ref_exists = evaluator.add_custom_node(
        result=_urls_nonempty(info.ign_urls),
        id="ign_reference_url",
        desc="Provide a valid URL that supports the IGN Awards 2025 Best Game of 2025 win claim for the named game.",
        parent=ref_node,
        critical=True
    )
    # Also verify via content
    ign_ref_verify = evaluator.add_leaf(
        id="ign_reference_url_supports_win",
        desc="The provided IGN reference URL(s) explicitly support the Best Game of 2025 win for the named game.",
        parent=ref_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm that '{info.title}' won 'Best Game of 2025' at the IGN Awards 2025.",
        node=ign_ref_verify,
        sources=info.ign_urls,
        additional_instruction=(
            "Ensure the page clearly shows 'Best Game of 2025' and the named game as winner at the IGN Awards 2025."
        )
    )

    # Authoritativeness checks: require authoritative domains for both sets
    authority_node = evaluator.add_parallel(
        id="source_authoritativeness",
        desc="The provided URLs are from official award sites or otherwise authoritative gaming news sources, and the cited pages contain the needed verification.",
        parent=ref_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_authoritative_domain(info.tga_urls),
        id="tga_source_authoritative",
        desc="TGA reference URLs include official or authoritative domains.",
        parent=authority_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_authoritative_domain(info.ign_urls),
        id="ign_source_authoritative",
        desc="IGN reference URLs include official or authoritative domains.",
        parent=authority_node,
        critical=True
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
    Evaluate an agent's answer for the dual GOTY 2025 task.
    Returns a structured summary with the verification tree and score.
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
    info = await evaluator.extract(
        prompt=prompt_extract_game_info(),
        template_class=GameExtraction,
        extraction_name="game_info"
    )

    # Build verification tree according to rubric
    await build_game_title_checks(evaluator, root, info)
    await build_award_verification(evaluator, root, info)
    await build_release_date_checks(evaluator, root, info)
    await build_platform_checks(evaluator, root, info)
    await build_devpub_checks(evaluator, root, info)
    await build_additional_tga_awards(evaluator, root, info)
    await build_additional_ign_awards(evaluator, root, info)
    await build_reference_urls(evaluator, root, info)

    # Return evaluation summary
    return evaluator.get_summary()