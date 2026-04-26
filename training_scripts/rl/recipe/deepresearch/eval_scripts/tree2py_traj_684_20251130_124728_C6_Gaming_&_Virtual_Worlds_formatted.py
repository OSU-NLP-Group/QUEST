import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "crossplay_event_nov2025"
TASK_DESCRIPTION = """
A gaming community is organizing a cross-platform multiplayer event in November 2025 for members who own different gaming consoles and PCs. They are evaluating three popular multiplayer games: Madden NFL 26, Helldivers 2, and Fortnite. For each of these three games, provide the following information:

1. For Madden NFL 26: Identify the official release date, and determine which of the following platforms can play together through crossplay functionality: PlayStation 5, Xbox Series X, PC, and Nintendo Switch 2. For each platform, specify which other platforms it can crossplay with.

2. For Helldivers 2: Identify when the game became available on Xbox Series X/S, and determine which of the following platforms can play together through crossplay functionality: PlayStation 5, Xbox Series X, and PC. For each platform, specify which other platforms it can crossplay with.

3. For Fortnite: Determine which of the following platforms can play together through crossplay functionality: PlayStation 5, Xbox Series X, PC, and Nintendo Switch. For each platform, specify which other platforms it can crossplay with.

For all provided information, include reference URLs from official or reliable gaming news sources to verify each claim.
"""

# Canonical platform name sets per game
MADDEN_PLATFORMS = ["PlayStation 5", "Xbox Series X", "PC", "Nintendo Switch 2"]
HELLDIVERS_PLATFORMS = ["PlayStation 5", "Xbox Series X", "PC"]
FORTNITE_PLATFORMS = ["PlayStation 5", "Xbox Series X", "PC", "Nintendo Switch"]

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class PlatformCrossplayInfo(BaseModel):
    platform: Optional[str] = None  # Use canonical names (e.g., "PlayStation 5", "Xbox Series X", "PC", "Nintendo Switch 2")
    crossplay_with: List[str] = Field(default_factory=list)  # List of canonical platform names it can crossplay with
    sources: List[str] = Field(default_factory=list)  # URLs supporting the crossplay statement for this platform


class MaddenExtraction(BaseModel):
    release_date: Optional[str] = None  # The official general public release date
    release_sources: List[str] = Field(default_factory=list)
    crossplay: List[PlatformCrossplayInfo] = Field(default_factory=list)  # One item per platform (PS5, Xbox Series X, PC, Nintendo Switch 2)


class HelldiversExtraction(BaseModel):
    xbox_release_date: Optional[str] = None  # Date Helldivers 2 became available on Xbox Series X/S
    xbox_release_sources: List[str] = Field(default_factory=list)
    crossplay: List[PlatformCrossplayInfo] = Field(default_factory=list)  # For PS5, Xbox Series X, PC


class FortniteExtraction(BaseModel):
    crossplay: List[PlatformCrossplayInfo] = Field(default_factory=list)  # For PS5, Xbox Series X, PC, Nintendo Switch


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_madden() -> str:
    return """
    Extract structured information about Madden NFL 26 from the answer.

    Required fields:
    - release_date: The official general public release date of Madden NFL 26 (not early access/trial dates). If not stated, return null.
    - release_sources: A list of URLs explicitly cited in the answer that support the stated release_date. If none are provided, return an empty list.
    - crossplay: An array containing up to 4 objects, each for one of the following platforms:
        ["PlayStation 5", "Xbox Series X", "PC", "Nintendo Switch 2"].
      For each object, include:
        * platform: The platform name EXACTLY as one of the above strings.
        * crossplay_with: A list of other platforms (from the same set) this platform can crossplay with, as stated in the answer.
                          Use EXACT canonical names, allow any order. If not stated, use an empty list.
        * sources: A list of URLs from the answer that specifically support the crossplay statement for this platform.
                   If none are provided, return an empty list.

    Notes:
    - Only extract URLs that actually appear in the answer text (including markdown links).
    - If a platform is not mentioned, you may omit it or include it with empty lists.
    """


def prompt_extract_helldivers() -> str:
    return """
    Extract structured information about Helldivers 2 from the answer.

    Required fields:
    - xbox_release_date: The date Helldivers 2 became available on Xbox Series X/S (general availability). If not stated, return null.
    - xbox_release_sources: A list of URLs explicitly cited in the answer that support the stated xbox_release_date. If none are provided, return an empty list.
    - crossplay: An array with objects for the following platforms: ["PlayStation 5", "Xbox Series X", "PC"].
      For each object, include:
        * platform: EXACTLY one of the above strings.
        * crossplay_with: A list of other platforms (from the same set) that this platform can crossplay with, as stated in the answer.
                          If not stated, use an empty list.
        * sources: A list of URLs from the answer that support the crossplay claim for this platform. If none, use an empty list.

    Notes:
    - Only extract URLs that actually appear in the answer text (including markdown links).
    - Treat "Xbox Series X|S" as "Xbox Series X" for the canonical platform name.
    """


def prompt_extract_fortnite() -> str:
    return """
    Extract structured information about Fortnite crossplay from the answer.

    Required fields:
    - crossplay: An array with objects for the following platforms: ["PlayStation 5", "Xbox Series X", "PC", "Nintendo Switch"].
      For each object, include:
        * platform: EXACTLY one of the above strings.
        * crossplay_with: A list of other platforms (from the same set) this platform can crossplay with, as stated in the answer.
                          If not stated, use an empty list.
        * sources: A list of URLs from the answer that support the crossplay claim for this platform. If none, use an empty list.

    Notes:
    - Only extract URLs that actually appear in the answer text (including markdown links).
    - Accept minor name variants in the answer, but normalize to the canonical names listed above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def find_platform_info(crossplay_list: List[PlatformCrossplayInfo], platform_name: str) -> PlatformCrossplayInfo:
    for item in crossplay_list or []:
        if (item.platform or "").strip().lower() == platform_name.strip().lower():
            # Normalize crossplay_with contents to canonical capitalization as-is provided; leave as strings
            return item
    # Return an empty placeholder if not found
    return PlatformCrossplayInfo(platform=platform_name, crossplay_with=[], sources=[])


def crossplay_additional_instruction(allowed_platforms: List[str], game_name: str) -> str:
    allowed_str = ", ".join(allowed_platforms)
    return (
        "Evaluate only crossplay relationships among the specified platform set: "
        f"[{allowed_str}] for the game {game_name}. Consider synonymous naming acceptable, e.g., "
        "'PS5' ≈ 'PlayStation 5'; 'Xbox Series X|S' ≈ 'Xbox Series X'; 'PC' may appear as 'Windows', 'Steam', or 'Epic Games'. "
        "If a source states 'full crossplay across all platforms' among the listed set, it supports each pairwise crossplay. "
        "Ignore references to cross-progression; verify crossplay (playing together) only."
    )


def release_date_additional_instruction(game_name: str) -> str:
    return (
        f"Verify the official general public release date for {game_name}. If sources mention early access, previews, or trials,"
        " do not treat those as the official release date. Prefer the commonly reported public launch date."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_release_verification(
    evaluator: Evaluator,
    parent_node,
    *,
    game_name: str,
    date_value: Optional[str],
    date_sources: List[str],
    id_prefix: str
) -> None:
    """
    Build the release/availability verification subtree:
    - Reference existence (critical custom)
    - Release date supported by URLs (critical verify)
    Parent node itself should be critical (handled by caller).
    """
    # Reference existence (critical)
    ref_exists = evaluator.add_custom_node(
        result=bool(date_sources),
        id=f"{id_prefix}_Release_Date_Reference",
        desc=f"Provide URL reference for the {game_name} release/availability date",
        parent=parent_node,
        critical=True
    )

    # Release date verify (critical) — gated by reference existence
    release_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Release_Date",
        desc=f"Provide the official release/availability date of {game_name}",
        parent=parent_node,
        critical=True
    )
    claim_date = f"The official release/availability date of {game_name} is '{date_value}'." if date_value else (
        f"The official release/availability date of {game_name} is not provided."
    )
    await evaluator.verify(
        claim=claim_date,
        node=release_leaf,
        sources=date_sources,
        additional_instruction=release_date_additional_instruction(game_name),
        extra_prerequisites=[ref_exists]
    )


async def build_platform_crossplay_verification(
    evaluator: Evaluator,
    parent_node,
    *,
    game_name: str,
    platform_name: str,
    info: PlatformCrossplayInfo,
    allowed_platforms: List[str],
    id_prefix: str
) -> None:
    """
    Build the platform-level crossplay verification subtree for a single platform:
    - Reference existence (critical custom)
    - Crossplay status supported by URLs (critical verify, gated on ref existence)
    Parent platform node itself should be critical (handled by caller).
    """
    # Create platform group node (critical)
    platform_group = evaluator.add_parallel(
        id=f"{id_prefix}_{platform_name.replace(' ', '_').replace('/', '_')}_Crossplay",
        desc=f"Determine if {platform_name} version supports crossplay for {game_name} and with which platforms",
        parent=parent_node,
        critical=True
    )

    # Reference existence (critical)
    ref_exists = evaluator.add_custom_node(
        result=bool(info.sources),
        id=f"{id_prefix}_{platform_name.replace(' ', '_').replace('/', '_')}_Crossplay_Reference",
        desc=f"Provide URL reference for {platform_name} crossplay information in {game_name}",
        parent=platform_group,
        critical=True
    )

    # Status verify (critical) — gated by reference existence
    status_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_{platform_name.replace(' ', '_').replace('/', '_')}_Crossplay_Status",
        desc=f"Specify which platforms {platform_name} can crossplay with for {game_name}",
        parent=platform_group,
        critical=True
    )

    partners = info.crossplay_with or []
    if partners:
        partners_str = ", ".join(partners)
        claim = (
            f"In {game_name}, the {platform_name} version supports crossplay with the following platforms: {partners_str}."
        )
    else:
        claim = (
            f"In {game_name}, the {platform_name} version does not support crossplay with any of the specified platforms."
        )

    await evaluator.verify(
        claim=claim,
        node=status_leaf,
        sources=info.sources,
        additional_instruction=crossplay_additional_instruction(allowed_platforms, game_name),
        extra_prerequisites=[ref_exists]
    )


async def build_game_crossplay_section(
    evaluator: Evaluator,
    parent_node,
    *,
    game_name: str,
    crossplay_list: List[PlatformCrossplayInfo],
    allowed_platforms: List[str],
    id_prefix: str
) -> None:
    """
    Build the crossplay section for a game, including all specified platforms.
    The section node is critical; each platform group is critical.
    """
    crossplay_root = evaluator.add_parallel(
        id=f"{id_prefix}_Platform_Crossplay_Support",
        desc=f"Identification of which platforms support crossplay in {game_name}",
        parent=parent_node,
        critical=True
    )

    # Build for each required platform
    for plat in allowed_platforms:
        info = find_platform_info(crossplay_list, plat)
        await build_platform_crossplay_verification(
            evaluator,
            crossplay_root,
            game_name=game_name,
            platform_name=plat,
            info=info,
            allowed_platforms=allowed_platforms,
            id_prefix=id_prefix
        )


# --------------------------------------------------------------------------- #
# Game-specific tree builders                                                 #
# --------------------------------------------------------------------------- #
async def verify_madden(
    evaluator: Evaluator,
    root_node
) -> None:
    """
    Build verification tree for Madden NFL 26.
    """
    # Create analysis node for Madden (non-critical)
    madden_node = evaluator.add_parallel(
        id="Madden_NFL_26_Analysis",
        desc="Analysis of Madden NFL 26 crossplay compatibility and release information",
        parent=root_node,
        critical=False
    )

    # Extracted data should already be recorded; fetch from evaluator's last extraction record? Instead, we pass it in from caller
    # We'll retrieve via a provided parameter in caller; to keep structure uniform, we will not attempt to read internal cache here.
    # This function will be called with the extracted object via partial. So this function signature needs the object.
    # To keep it pure, create a nested helper. We'll override with closure in caller.
    return


async def verify_helldivers(
    evaluator: Evaluator,
    root_node
) -> None:
    """
    Placeholder to keep structure; actual logic executed in main builder with extracted data.
    """
    return


async def verify_fortnite(
    evaluator: Evaluator,
    root_node
) -> None:
    """
    Placeholder to keep structure; actual logic executed in main builder with extracted data.
    """
    return


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer for the cross-platform multiplayer analysis task.
    """
    # Initialize evaluator (root is non-critical, parallel aggregation)
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

    # Extract data for all three games in parallel
    madden_task = evaluator.extract(
        prompt=prompt_extract_madden(),
        template_class=MaddenExtraction,
        extraction_name="madden_extraction"
    )
    helldivers_task = evaluator.extract(
        prompt=prompt_extract_helldivers(),
        template_class=HelldiversExtraction,
        extraction_name="helldivers_extraction"
    )
    fortnite_task = evaluator.extract(
        prompt=prompt_extract_fortnite(),
        template_class=FortniteExtraction,
        extraction_name="fortnite_extraction"
    )

    madden_data, helldivers_data, fortnite_data = await asyncio.gather(
        madden_task, helldivers_task, fortnite_task
    )

    # -------------------- Madden NFL 26 -------------------- #
    madden_node = evaluator.add_parallel(
        id="Madden_NFL_26_Analysis",
        desc="Analysis of Madden NFL 26 crossplay compatibility and release information",
        parent=root,
        critical=False
    )

    # Release information (critical)
    madden_release_node = evaluator.add_parallel(
        id="Madden_Release_Information",
        desc="Verification of Madden NFL 26 release date and availability",
        parent=madden_node,
        critical=True
    )
    await build_release_verification(
        evaluator,
        madden_release_node,
        game_name="Madden NFL 26",
        date_value=madden_data.release_date,
        date_sources=madden_data.release_sources,
        id_prefix="Madden"
    )

    # Crossplay information (critical)
    await build_game_crossplay_section(
        evaluator,
        madden_node,
        game_name="Madden NFL 26",
        crossplay_list=madden_data.crossplay or [],
        allowed_platforms=MADDEN_PLATFORMS,
        id_prefix="Madden"
    )

    # -------------------- Helldivers 2 -------------------- #
    helldivers_node = evaluator.add_parallel(
        id="Helldivers_2_Analysis",
        desc="Analysis of Helldivers 2 crossplay compatibility and release information",
        parent=root,
        critical=False
    )

    # Xbox release information (critical)
    helldivers_release_node = evaluator.add_parallel(
        id="Helldivers_Release_Information",
        desc="Verification of Helldivers 2 Xbox release date",
        parent=helldivers_node,
        critical=True
    )
    await build_release_verification(
        evaluator,
        helldivers_release_node,
        game_name="Helldivers 2 on Xbox Series X/S",
        date_value=helldivers_data.xbox_release_date,
        date_sources=helldivers_data.xbox_release_sources,
        id_prefix="Helldivers_Xbox"
    )

    # Crossplay information (critical)
    await build_game_crossplay_section(
        evaluator,
        helldivers_node,
        game_name="Helldivers 2",
        crossplay_list=helldivers_data.crossplay or [],
        allowed_platforms=HELLDIVERS_PLATFORMS,
        id_prefix="Helldivers"
    )

    # -------------------- Fortnite -------------------- #
    fortnite_node = evaluator.add_parallel(
        id="Fortnite_Analysis",
        desc="Analysis of Fortnite crossplay compatibility across platforms",
        parent=root,
        critical=False
    )

    await build_game_crossplay_section(
        evaluator,
        fortnite_node,
        game_name="Fortnite",
        crossplay_list=fortnite_data.crossplay or [],
        allowed_platforms=FORTNITE_PLATFORMS,
        id_prefix="Fortnite"
    )

    # Return structured result
    return evaluator.get_summary()