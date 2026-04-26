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
TASK_ID = "tga2024_goty_four_criteria"
TASK_DESCRIPTION = """
From the six Game of the Year nominees at The Game Awards 2024, identify four games that meet the following distinct criteria:

1. One game developed by a studio located in Japan (provide studio name, city location in Japan, and parent company if applicable)

2. One game developed by a solo independent developer (provide developer name/alias and confirm it was created by a single person)

3. One game developed by a studio located in China (provide studio name and city location in China)

4. One game that is an expansion or DLC rather than a standalone original game (provide the name of the base game it expands and the expansion's release date in 2024)

For each of the four games, provide:
- The game's title
- Developer information (studio name or individual developer name)
- Geographic location of the developer (city and country)
- Verification that it was nominated for Game of the Year at The Game Awards 2024
- At least one reference URL confirming the developer information
- At least one reference URL confirming the GOTY nomination

Additionally, provide any relevant supplementary information such as awards won at TGA 2024, platforms, or other notable details about each game.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameEntry(BaseModel):
    # Common fields
    title: Optional[str] = None

    # Developer identification
    developer_name: Optional[str] = None  # studio name OR solo developer alias/name
    developer_location_city: Optional[str] = None
    developer_location_country: Optional[str] = None

    # Developer relationships/applicability
    parent_company: Optional[str] = None
    # True only if the answer explicitly states independent/no parent; False if answer states there is a parent; null if unknown
    parent_company_not_applicable: Optional[bool] = None

    # Criterion-specific flags
    is_solo_developer: Optional[bool] = None
    is_expansion_or_dlc: Optional[bool] = None

    # Expansion-specific metadata
    base_game_name: Optional[str] = None
    expansion_release_date: Optional[str] = None  # keep as string to maximize compatibility

    # Sources
    developer_info_urls: List[str] = Field(default_factory=list)
    goty_nominee_urls: List[str] = Field(default_factory=list)

    # Supplementary
    supplementary_info: Optional[str] = None


class GamesExtraction(BaseModel):
    japan: Optional[GameEntry] = None
    indie: Optional[GameEntry] = None
    china: Optional[GameEntry] = None
    expansion: Optional[GameEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_games() -> str:
    return """
    Extract structured information for exactly four games that the answer claims satisfy the four requested categories, using the answer text ONLY:

    You must produce a JSON object with four top-level objects: "japan", "indie", "china", and "expansion".
    For each object, extract the following fields (use null for any missing field). Do not invent information.

    Common fields for all four:
    - title: The game's title.
    - developer_name: Studio name (for studio categories) or solo developer name/alias (for indie category).
    - developer_location_city: City stated for the developer.
    - developer_location_country: Country stated for the developer.
    - developer_info_urls: An array of URLs that the answer uses as references to confirm developer information (location, organization, solo status, etc.). Only include explicit URLs present in the answer.
    - goty_nominee_urls: An array of URLs that the answer uses to confirm that the game was a Game of the Year nominee at The Game Awards 2024. Only include explicit URLs present in the answer.
    - supplementary_info: Any additional details mentioned (platforms, awards at TGA 2024, notable facts). If none, return null.

    Japan-specific (japan object):
    - parent_company: Parent company name if the answer explicitly provides one; otherwise null.
    - parent_company_not_applicable: boolean
        • true if the answer explicitly states the studio is independent or has no parent company.
        • false if the answer explicitly states a parent exists or implies subsidiary status.
        • null if not specified.

    Indie-specific (indie object):
    - is_solo_developer: boolean
        • true if the answer explicitly states the game was made by a single person (solo developer).
        • false if the answer explicitly states a team or multiple contributors.
        • null if not specified.

    China-specific (china object):
    # No extra booleans beyond common fields.

    Expansion-specific (expansion object):
    - is_expansion_or_dlc: boolean (true only if the answer explicitly says expansion/DLC; false if standalone; null if not specified).
    - base_game_name: The base game that this expansion/DLC extends (if provided).
    - expansion_release_date: The release date string given in the answer for the expansion/DLC (expected in 2024). Keep the exact text.

    IMPORTANT:
    - Only extract URLs that appear in the answer text. Accept both plain URLs and markdown-formatted links.
    - Do not normalize or deduplicate text fields; copy exactly as written.
    - For boolean fields, set true/false only if the answer makes it explicit; otherwise null.
    - Return null for any field if the answer does not state it explicitly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""

def _urls_nonempty(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate and filter empties
    seen = set()
    result = []
    for u in urls:
        if isinstance(u, str):
            u2 = u.strip()
            if u2 and u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result

def _country_matches(entry_country: Optional[str], expected_country: str) -> bool:
    if not _nonempty(entry_country):
        return False
    return expected_country.lower() in entry_country.strip().lower()

def _titles_distinct(*titles: Optional[str]) -> bool:
    t = [ti.strip().lower() for ti in titles if _nonempty(ti)]
    return len(t) == 4 and len(set(t)) == 4


# --------------------------------------------------------------------------- #
# Verification builders per category                                          #
# --------------------------------------------------------------------------- #
async def verify_japan_game(evaluator: Evaluator, parent_node, entry: Optional[GameEntry]) -> None:
    node = evaluator.add_parallel(
        id="game_from_japan",
        desc="One GOTY nominee developed by a studio located in Japan, with required developer fields and references.",
        parent=parent_node,
        critical=False  # allow supplementary info to be non-critical
    )

    title_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.title),
        id="japan_game_title",
        desc="Game title is provided.",
        parent=node,
        critical=True
    )

    goty_urls = _urls_nonempty(entry.goty_nominee_urls if entry else [])
    goty_ref_present = evaluator.add_custom_node(
        result=len(goty_urls) > 0,
        id="japan_goty_nominee_reference_url",
        desc="At least one reference URL confirming the GOTY nomination is provided.",
        parent=node,
        critical=True
    )

    goty_verify_leaf = evaluator.add_leaf(
        id="japan_goty_nominee_verification",
        desc="Game is verified to be a TGA 2024 Game of the Year nominee.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The game '{entry.title if entry else ''}' was nominated for Game of the Year at The Game Awards 2024.",
        node=goty_verify_leaf,
        sources=goty_urls,
        additional_instruction="The source must explicitly indicate the game was a 'Game of the Year' nominee at The Game Awards 2024 (a.k.a. TGA 2024). Accept synonymous phrasing like 'GOTY nominee at TGA 2024'."
    )

    studio_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_name),
        id="japan_studio_name",
        desc="Developer studio name is provided.",
        parent=node,
        critical=True
    )

    loc_is_japan = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_location_city) and _country_matches(entry.developer_location_country, "Japan"),
        id="japan_studio_location_city_country",
        desc="Developer geographic location is provided (city + country) and is in Japan.",
        parent=node,
        critical=True
    )

    parent_company_ok = evaluator.add_custom_node(
        result=entry is not None and (
            _nonempty(entry.parent_company) or (entry.parent_company_not_applicable is True)
        ),
        id="japan_parent_company_if_applicable",
        desc="Parent company is provided if applicable.",
        parent=node,
        critical=True
    )

    dev_urls = _urls_nonempty(entry.developer_info_urls if entry else [])
    dev_ref_present = evaluator.add_custom_node(
        result=len(dev_urls) > 0,
        id="japan_developer_reference_url",
        desc="At least one reference URL confirming the developer information is provided.",
        parent=node,
        critical=True
    )

    supp_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.supplementary_info),
        id="japan_supplementary_info",
        desc="Supplementary information is provided (e.g., awards won at TGA 2024, platforms, or other notable details).",
        parent=node,
        critical=False
    )


async def verify_indie_solo_game(evaluator: Evaluator, parent_node, entry: Optional[GameEntry]) -> None:
    node = evaluator.add_parallel(
        id="indie_solo_developer_game",
        desc="One GOTY nominee developed by a solo independent developer, with required developer fields and references.",
        parent=parent_node,
        critical=False
    )

    title_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.title),
        id="indie_game_title",
        desc="Game title is provided.",
        parent=node,
        critical=True
    )

    goty_urls = _urls_nonempty(entry.goty_nominee_urls if entry else [])
    goty_ref_present = evaluator.add_custom_node(
        result=len(goty_urls) > 0,
        id="indie_goty_nominee_reference_url",
        desc="At least one reference URL confirming the GOTY nomination is provided.",
        parent=node,
        critical=True
    )

    goty_verify_leaf = evaluator.add_leaf(
        id="indie_goty_nominee_verification",
        desc="Game is verified to be a TGA 2024 Game of the Year nominee.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The game '{entry.title if entry else ''}' was nominated for Game of the Year at The Game Awards 2024.",
        node=goty_verify_leaf,
        sources=goty_urls,
        additional_instruction="The source must explicitly indicate the game was a GOTY nominee at TGA 2024."
    )

    dev_name_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_name),
        id="indie_developer_name_or_alias",
        desc="Solo developer name/alias is provided.",
        parent=node,
        critical=True
    )

    loc_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_location_city) and _nonempty(entry.developer_location_country),
        id="indie_developer_location_city_country",
        desc="Developer geographic location is provided (city + country).",
        parent=node,
        critical=True
    )

    dev_urls = _urls_nonempty(entry.developer_info_urls if entry else [])
    dev_ref_present = evaluator.add_custom_node(
        result=len(dev_urls) > 0,
        id="indie_developer_reference_url",
        desc="At least one reference URL confirming the developer information is provided.",
        parent=node,
        critical=True
    )

    solo_leaf = evaluator.add_leaf(
        id="indie_solo_developer_confirmation",
        desc="It is confirmed the game was created by a single person (solo developer).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The game '{entry.title if entry else ''}' was created by a single person (solo developer) named '{entry.developer_name if entry else ''}'.",
        node=solo_leaf,
        sources=dev_urls,
        additional_instruction="The source should clearly indicate that this is a solo-developed game (phrases like 'solo developer', 'made by one person', or similar)."
    )

    supp_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.supplementary_info),
        id="indie_supplementary_info",
        desc="Supplementary information is provided (e.g., awards won at TGA 2024, platforms, or other notable details).",
        parent=node,
        critical=False
    )


async def verify_china_game(evaluator: Evaluator, parent_node, entry: Optional[GameEntry]) -> None:
    node = evaluator.add_parallel(
        id="game_from_china",
        desc="One GOTY nominee developed by a studio located in China, with required developer fields and references.",
        parent=parent_node,
        critical=False
    )

    title_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.title),
        id="china_game_title",
        desc="Game title is provided.",
        parent=node,
        critical=True
    )

    goty_urls = _urls_nonempty(entry.goty_nominee_urls if entry else [])
    goty_ref_present = evaluator.add_custom_node(
        result=len(goty_urls) > 0,
        id="china_goty_nominee_reference_url",
        desc="At least one reference URL confirming the GOTY nomination is provided.",
        parent=node,
        critical=True
    )

    goty_verify_leaf = evaluator.add_leaf(
        id="china_goty_nominee_verification",
        desc="Game is verified to be a TGA 2024 Game of the Year nominee.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The game '{entry.title if entry else ''}' was nominated for Game of the Year at The Game Awards 2024.",
        node=goty_verify_leaf,
        sources=goty_urls,
        additional_instruction="The source must explicitly indicate the game was a GOTY nominee at TGA 2024."
    )

    studio_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_name),
        id="china_studio_name",
        desc="Developer studio name is provided.",
        parent=node,
        critical=True
    )

    loc_is_china = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_location_city) and _country_matches(entry.developer_location_country, "China"),
        id="china_studio_location_city_country",
        desc="Developer geographic location is provided (city + country) and is in China.",
        parent=node,
        critical=True
    )

    dev_urls = _urls_nonempty(entry.developer_info_urls if entry else [])
    dev_ref_present = evaluator.add_custom_node(
        result=len(dev_urls) > 0,
        id="china_developer_reference_url",
        desc="At least one reference URL confirming the developer information is provided.",
        parent=node,
        critical=True
    )

    supp_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.supplementary_info),
        id="china_supplementary_info",
        desc="Supplementary information is provided (e.g., awards won at TGA 2024, platforms, or other notable details).",
        parent=node,
        critical=False
    )


async def verify_expansion_game(evaluator: Evaluator, parent_node, entry: Optional[GameEntry]) -> None:
    node = evaluator.add_parallel(
        id="expansion_dlc_game",
        desc="One GOTY nominee that is an expansion/DLC (not a standalone original), with base game + 2024 release date, and required references.",
        parent=parent_node,
        critical=False
    )

    title_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.title),
        id="expansion_title",
        desc="Expansion/DLC title is provided.",
        parent=node,
        critical=True
    )

    goty_urls = _urls_nonempty(entry.goty_nominee_urls if entry else [])
    goty_ref_present = evaluator.add_custom_node(
        result=len(goty_urls) > 0,
        id="expansion_goty_nominee_reference_url",
        desc="At least one reference URL confirming the GOTY nomination is provided.",
        parent=node,
        critical=True
    )

    goty_verify_leaf = evaluator.add_leaf(
        id="expansion_goty_nominee_verification",
        desc="Expansion/DLC is verified to be a TGA 2024 Game of the Year nominee.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The game '{entry.title if entry else ''}' was nominated for Game of the Year at The Game Awards 2024.",
        node=goty_verify_leaf,
        sources=goty_urls,
        additional_instruction="The source must explicitly indicate the game was a GOTY nominee at TGA 2024."
    )

    # DLC confirmation
    dev_urls = _urls_nonempty(entry.developer_info_urls if entry else [])
    dlc_leaf = evaluator.add_leaf(
        id="expansion_is_dlc_confirmation",
        desc="It is confirmed the nominee is an expansion or DLC rather than a standalone original game.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The title '{entry.title if entry else ''}' is an expansion or DLC (not a standalone original game).",
        node=dlc_leaf,
        sources=dev_urls if dev_urls else goty_urls,
        additional_instruction="The source should clearly describe the nominee as an 'expansion', 'DLC', or equivalent (e.g., 'expansion pack')."
    )

    base_game_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.base_game_name),
        id="expansion_base_game_name",
        desc="Name of the base game that the expansion/DLC expands is provided.",
        parent=node,
        critical=True
    )

    # Release date provided and is in 2024; verify against URLs if possible
    release_leaf = evaluator.add_leaf(
        id="expansion_release_date_2024",
        desc="Expansion/DLC release date is provided and is in 2024.",
        parent=node,
        critical=True
    )
    release_date_text = entry.expansion_release_date if entry else ""
    await evaluator.verify(
        claim=f"The expansion/DLC '{entry.title if entry else ''}' was released on {release_date_text} in 2024.",
        node=release_leaf,
        sources=dev_urls if dev_urls else goty_urls,
        additional_instruction="Confirm that the given date corresponds to the expansion/DLC and that the year is 2024. If the exact day/month varies by region, it's acceptable as long as the year 2024 is correct."
    )

    dev_name_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_name),
        id="expansion_developer_name",
        desc="Developer (studio or entity) name is provided.",
        parent=node,
        critical=True
    )

    loc_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.developer_location_city) and _nonempty(entry.developer_location_country),
        id="expansion_developer_location_city_country",
        desc="Developer geographic location is provided (city + country).",
        parent=node,
        critical=True
    )

    dev_ref_present = evaluator.add_custom_node(
        result=len(dev_urls) > 0,
        id="expansion_developer_reference_url",
        desc="At least one reference URL confirming the developer information is provided.",
        parent=node,
        critical=True
    )

    supp_present = evaluator.add_custom_node(
        result=entry is not None and _nonempty(entry.supplementary_info),
        id="expansion_supplementary_info",
        desc="Supplementary information is provided (e.g., awards won at TGA 2024, platforms, or other notable details).",
        parent=node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the TGA 2024 GOTY nominees with four distinct criteria.
    """
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

    # Root-level container node mirroring "task_completion" (set non-critical to allow supplementary leaves)
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Identify four distinct TGA 2024 Game of the Year nominees matching the four specified criteria, and provide required fields and required references for each.",
        parent=root,
        critical=False
    )

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_games(),
        template_class=GamesExtraction,
        extraction_name="games_extraction"
    )

    # 2) Distinctness check across all four games (critical at the task level)
    titles = (
        (extracted.japan.title if extracted and extracted.japan else None),
        (extracted.indie.title if extracted and extracted.indie else None),
        (extracted.china.title if extracted and extracted.china else None),
        (extracted.expansion.title if extracted and extracted.expansion else None),
    )
    evaluator.add_custom_node(
        result=_titles_distinct(*titles),
        id="games_are_distinct",
        desc="The four selected games are distinct (no game is used to satisfy more than one of the four criteria).",
        parent=task_node,
        critical=True
    )

    # 3) Per-category verifications
    await verify_japan_game(evaluator, task_node, extracted.japan if extracted else None)
    await verify_indie_solo_game(evaluator, task_node, extracted.indie if extracted else None)
    await verify_china_game(evaluator, task_node, extracted.china if extracted else None)
    await verify_expansion_game(evaluator, task_node, extracted.expansion if extracted else None)

    # Return final summary
    return evaluator.get_summary()