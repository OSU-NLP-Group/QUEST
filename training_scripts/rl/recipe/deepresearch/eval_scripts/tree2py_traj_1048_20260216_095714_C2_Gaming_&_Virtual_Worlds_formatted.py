import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "springdale_indie_game_studio_2026"
TASK_DESCRIPTION = """
Identify the indie game development studio based in Springdale, Arkansas that released its debut game in January 2026. Provide the names of the studio's two co-founders and the title of their debut game that was released on Steam for PC and Mac.
"""


class StudioInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    cofounders: List[str] = Field(default_factory=list)
    founding_timeline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GameInfo(BaseModel):
    title: Optional[str] = None
    release_date_text: Optional[str] = None  # Prefer a string like "January 2026"
    steam_url: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)  # e.g., ["Windows", "macOS"]
    sources: List[str] = Field(default_factory=list)


class StudioGameExtraction(BaseModel):
    studio: Optional[StudioInfo] = None
    game: Optional[GameInfo] = None


def prompt_extract_studio_and_game_info() -> str:
    return """
    Extract structured information about the Springdale, Arkansas-based indie game studio and its debut game from the answer text.

    Return a JSON object with two top-level fields: "studio" and "game".

    For "studio", extract:
    - name: The studio's name exactly as stated.
    - location: The location string associated with the studio (e.g., "Springdale, Arkansas").
    - description: The studio's self-description or tagline if present, especially any phrasing like "narrative-focused independent game studio in the Ozarks of Northwest Arkansas".
    - cofounders: An array of names of co-founders mentioned (include all names that are explicitly stated as co-founders; if more than two are listed, include them all).
    - founding_timeline: The founding timing phrasing or year if provided (e.g., "founded approximately five years before 2025", "founded in 2019/2020").
    - sources: All URLs in the answer that directly refer to the studio (official site, press, profiles). Include only valid URLs explicitly present in the answer.

    For "game", extract:
    - title: The title of the debut game.
    - release_date_text: The release date phrasing as stated (e.g., "January 2026").
    - steam_url: The Steam store page URL for the game (if provided).
    - platforms: A list of platforms/OS mentioned for the game (e.g., "Windows", "macOS", "PC", "Mac").
    - sources: All URLs in the answer that directly refer to the game (Steam page, official announcements, reviews). Include only valid URLs explicitly present in the answer.

    Rules:
    - Do not invent information. If any field is missing, set it to null (or an empty array for list fields).
    - For URLs, include only full valid URLs. If a URL lacks protocol, prepend "http://".
    - If multiple URLs are given, include all of them.
    """


def _sanitize_urls(urls: List[str]) -> List[str]:
    seen = set()
    clean: List[str] = []
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Basic validity check
        if "://" not in u:
            # Prepend protocol if looks like a domain/path
            if "." in u:
                u = "http://" + u
            else:
                continue
        if u not in seen:
            seen.add(u)
            clean.append(u)
    return clean


def _combine_sources(studio: Optional[StudioInfo], game: Optional[GameInfo]) -> List[str]:
    studio_sources = _sanitize_urls(studio.sources if studio else [])
    game_sources = _sanitize_urls(game.sources if game else [])
    steam = [game.steam_url] if (game and game.steam_url) else []
    return _sanitize_urls(studio_sources + game_sources + steam)


def _studio_sources(studio: Optional[StudioInfo]) -> List[str]:
    return _sanitize_urls(studio.sources if studio else [])


def _game_sources(game: Optional[GameInfo]) -> List[str]:
    steam = [game.steam_url] if (game and game.steam_url) else []
    return _sanitize_urls((game.sources if game else []) + steam)


async def verify_studio_information(evaluator: Evaluator, parent_node, extracted: StudioGameExtraction) -> None:
    studio_node = evaluator.add_parallel(
        id="Studio_Identification",
        desc="Correctly identify the game development studio based in Springdale, Arkansas that released its debut game in January 2026",
        parent=parent_node,
        critical=False
    )

    studio = extracted.studio
    game = extracted.game
    studio_name = (studio.name or "").strip()

    # Leaf: Studio Location
    loc_leaf = evaluator.add_leaf(
        id="Studio_Location",
        desc="The studio is located in Springdale, Arkansas",
        parent=studio_node,
        critical=True
    )
    location_claim = (
        f"The studio {studio_name} is based in Springdale, Arkansas."
        if studio_name else
        "The studio is based in Springdale, Arkansas."
    )
    await evaluator.verify(
        claim=location_claim,
        node=loc_leaf,
        sources=_studio_sources(studio) or _combine_sources(studio, game),
        additional_instruction="Verify the page states the studio is located in Springdale, AR (allow 'Springdale, Arkansas'). Mentions of the Ozarks/Northwest Arkansas region are supportive but must connect specifically to Springdale."
    )

    # Leaf: Studio Description
    desc_leaf = evaluator.add_leaf(
        id="Studio_Description",
        desc="The studio is described as a narrative-focused independent game studio in the Ozarks of Northwest Arkansas",
        parent=studio_node,
        critical=True
    )
    desc_claim = (
        f"The studio {studio_name} describes itself as a narrative-focused independent game studio in the Ozarks of Northwest Arkansas."
        if studio_name else
        "The studio describes itself as a narrative-focused independent game studio in the Ozarks of Northwest Arkansas."
    )
    await evaluator.verify(
        claim=desc_claim,
        node=desc_leaf,
        sources=_studio_sources(studio) or _combine_sources(studio, game),
        additional_instruction="Accept paraphrases like 'narrative-driven indie studio' and 'Ozark region of Northwest Arkansas'. The essence must match: narrative focus, independence, and Ozarks/NW Arkansas."
    )

    # Leaf: Studio Cofounders
    cofounders_leaf = evaluator.add_leaf(
        id="Studio_Cofounders",
        desc="Two co-founders of the studio are identified by name",
        parent=studio_node,
        critical=True
    )
    cofounders = studio.cofounders if studio and studio.cofounders else []
    # Use first two names if available
    cf1 = cofounders[0] if len(cofounders) > 0 else ""
    cf2 = cofounders[1] if len(cofounders) > 1 else ""
    cofounders_claim = (
        f"The studio {studio_name} was co-founded by {cf1} and {cf2}."
        if studio_name else
        f"The studio was co-founded by {cf1} and {cf2}."
    )
    await evaluator.verify(
        claim=cofounders_claim,
        node=cofounders_leaf,
        sources=_studio_sources(studio) or _combine_sources(studio, game),
        additional_instruction="Verify that the page explicitly names the studio's co-founders. Accept synonyms like 'founders' or 'co-founders'. Both names must be present."
    )

    # Leaf: Studio Founding Timeline
    founding_leaf = evaluator.add_leaf(
        id="Studio_Founding_Timeline",
        desc="The studio was founded approximately five years before 2025",
        parent=studio_node,
        critical=True
    )
    founding_claim = (
        f"The studio {studio_name} was founded approximately five years before 2025 (around 2019–2020)."
        if studio_name else
        "The studio was founded approximately five years before 2025 (around 2019–2020)."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=founding_leaf,
        sources=_studio_sources(studio) or _combine_sources(studio, game),
        additional_instruction="Accept phrasing indicating a founding around 2019 or 2020. 'Approximately five years before 2025' should be interpreted as circa 2019–2020."
    )


async def verify_game_information(evaluator: Evaluator, parent_node, extracted: StudioGameExtraction) -> None:
    game_node = evaluator.add_parallel(
        id="Debut_Game_Information",
        desc="Provide accurate information about the studio's debut game released in January 2026",
        parent=parent_node,
        critical=False
    )

    studio = extracted.studio
    game = extracted.game
    studio_name = (studio.name or "").strip()
    game_title = (game.title or "").strip()

    # Leaf: Game Title
    title_leaf = evaluator.add_leaf(
        id="Game_Title",
        desc="The debut game title is correctly identified",
        parent=game_node,
        critical=True
    )
    title_claim = (
        f"The studio {studio_name}'s debut game is titled '{game_title}'."
        if studio_name else
        f"The debut game is titled '{game_title}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=_game_sources(game),
        additional_instruction="Verify the title on the Steam page or official announcement matches exactly or nearly (allow minor punctuation/casing variants). If developer info is present, it should reference the named studio."
    )

    # Leaf: Game Release Date
    rel_leaf = evaluator.add_leaf(
        id="Game_Release_Date",
        desc="The debut game was released in January 2026",
        parent=game_node,
        critical=True
    )
    rel_claim = "The debut game was released in January 2026."
    await evaluator.verify(
        claim=rel_claim,
        node=rel_leaf,
        sources=_game_sources(game),
        additional_instruction="Check the Steam page or official sources for the release date. Accept timezone-related edge cases but the visible release date should be within January 2026."
    )

    # Leaf: Game Platform (Steam for PC and Mac)
    platform_leaf = evaluator.add_leaf(
        id="Game_Platform",
        desc="The debut game is available on Steam for PC and Mac platforms",
        parent=game_node,
        critical=True
    )
    platform_claim = "The debut game is available on Steam for Windows (PC) and macOS (Mac)."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=_game_sources(game),
        additional_instruction="Verify the Steam page indicates both Windows and macOS support (PC and Mac). Accept 'macOS' for Mac and 'Windows' for PC."
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_studio_and_game_info(),
        template_class=StudioGameExtraction,
        extraction_name="studio_game_extraction"
    )

    # Record some lightweight custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "studio_name": extracted.studio.name if extracted.studio else None,
            "game_title": extracted.game.title if extracted.game else None,
            "steam_url": extracted.game.steam_url if extracted.game else None
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    await verify_studio_information(evaluator, root, extracted)
    await verify_game_information(evaluator, root, extracted)

    return evaluator.get_summary()