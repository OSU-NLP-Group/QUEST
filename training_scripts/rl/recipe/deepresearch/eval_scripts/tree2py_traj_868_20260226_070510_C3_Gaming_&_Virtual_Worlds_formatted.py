import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "indie_studio_iga_2025"
TASK_DESCRIPTION = (
    "Identify the name of the indie game development studio that meets all of the following criteria: "
    "(1) The studio developed a game that won Game of the Year at The Indie Game Awards 2025 ceremony, which took place on December 18, 2025; "
    "(2) This game was the studio's first released game (debut game); "
    "(3) The studio was founded between 2015 and 2022, inclusive; "
    "(4) The studio is located in France; and "
    "(5) The city where the studio is headquartered has a name that starts with the letter 'M'."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StudioIdentification(BaseModel):
    # Core entities
    studio_name: Optional[str] = None
    game_title: Optional[str] = None

    # Award info
    award_event: Optional[str] = None               # e.g., "The Indie Game Awards 2025"
    award_category: Optional[str] = None            # e.g., "Game of the Year"
    award_event_date: Optional[str] = None          # e.g., "December 18, 2025" or "2025-12-18"

    # Debut information
    is_debut_game: Optional[str] = None             # e.g., "yes", "true", "first game", or "no"

    # Founding and location
    founding_year: Optional[str] = None             # keep as string to be lenient
    country: Optional[str] = None
    city: Optional[str] = None                      # headquarters city

    # URLs for evidence (as explicitly included in the answer)
    award_urls: List[str] = Field(default_factory=list)     # sources about the award and event
    debut_urls: List[str] = Field(default_factory=list)     # sources confirming debut status
    founding_urls: List[str] = Field(default_factory=list)  # sources for founding year
    location_urls: List[str] = Field(default_factory=list)  # sources for country/city
    studio_urls: List[str] = Field(default_factory=list)    # official site, Steam page, Wikipedia, etc.
    general_urls: List[str] = Field(default_factory=list)   # any other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_studio_identification() -> str:
    return """
    Extract the key details about the identified indie game development studio and its award-winning game from the answer.

    Required fields:
    - studio_name: The full name of the studio.
    - game_title: The title of the game that supposedly won the award.
    - award_event: The name of the award event (e.g., "The Indie Game Awards 2025").
    - award_category: The award category that the game won (e.g., "Game of the Year").
    - award_event_date: The date of the ceremony as stated (any reasonable format is acceptable).
    - is_debut_game: Whether the award-winning game was the studio's first released game (free-form string; e.g., "yes", "first game", or "no").
    - founding_year: The founding year of the studio (as a string; do not coerce to number).
    - country: The country where the studio is located/headquartered.
    - city: The headquarters city of the studio.

    Also extract all URLs mentioned in the answer and categorize them (empty array if none for a category):
    - award_urls: URLs that support the award win and/or event details.
    - debut_urls: URLs supporting that the game is the studio's debut.
    - founding_urls: URLs about the founding year.
    - location_urls: URLs confirming the studio's country/city.
    - studio_urls: URLs to official/authoritative pages about the studio or the game (e.g., official site, Steam, Wikipedia).
    - general_urls: Any other URLs cited.

    IMPORTANT:
    - Extract only what is explicitly present in the answer.
    - Do not invent content. If any field is not present in the answer, return null for that field.
    - For URLs, extract only valid URLs explicitly present in the answer. Do not infer or fabricate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# Build and verify tree sections                                              #
# --------------------------------------------------------------------------- #
async def build_award_section(evaluator: Evaluator, parent_node, data: StudioIdentification) -> None:
    """
    Build and verify the 'Award_Winning_Game' section:
    - Studio developed the game.
    - The game won Game of the Year at The Indie Game Awards 2025.
    - The ceremony took place on December 18, 2025.
    """
    node = evaluator.add_parallel(
        id="Award_Winning_Game",
        desc="The studio developed a game that won Game of the Year at The Indie Game Awards 2025 ceremony (held December 18, 2025)",
        parent=parent_node,
        critical=True
    )

    # Existence and sources checks (critical siblings to gate verification)
    evaluator.add_custom_node(
        result=_nonempty(data.studio_name) and _nonempty(data.game_title),
        id="award_studio_and_game_present",
        desc="Studio name and game title are provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(data.award_urls) > 0 or len(data.general_urls) > 0,
        id="award_sources_provided",
        desc="Award-related sources are provided",
        parent=node,
        critical=True
    )

    # 1) Studio developed the game
    leaf_dev = evaluator.add_leaf(
        id="studio_developed_game",
        desc="The identified studio developed the identified game",
        parent=node,
        critical=True
    )
    dev_claim = f"The studio '{data.studio_name or ''}' developed the game '{data.game_title or ''}'."
    await evaluator.verify(
        claim=dev_claim,
        node=leaf_dev,
        sources=_merge_sources(data.studio_urls, data.award_urls, data.general_urls),
        additional_instruction="Verify developer relationship (not just publisher). Allow reasonable name variants."
    )

    # 2) Game won 'Game of the Year' at The Indie Game Awards 2025
    leaf_goty = evaluator.add_leaf(
        id="award_goty_2025",
        desc="The game won 'Game of the Year' at The Indie Game Awards 2025",
        parent=node,
        critical=True
    )
    goty_claim = (
        f"The game '{data.game_title or ''}' won the 'Game of the Year' award at The Indie Game Awards 2025."
    )
    await evaluator.verify(
        claim=goty_claim,
        node=leaf_goty,
        sources=_merge_sources(data.award_urls, data.general_urls),
        additional_instruction=(
            "Confirm the event name corresponds to 'The Indie Game Awards 2025' and the category is 'Game of the Year'. "
            "Allow minor formatting variations like capitalization or quotes."
        )
    )

    # 3) Ceremony date verification: December 18, 2025
    leaf_date = evaluator.add_leaf(
        id="award_ceremony_date_dec18",
        desc="The Indie Game Awards 2025 ceremony took place on December 18, 2025",
        parent=node,
        critical=True
    )
    date_claim = "The Indie Game Awards 2025 ceremony took place on December 18, 2025."
    await evaluator.verify(
        claim=date_claim,
        node=leaf_date,
        sources=_merge_sources(data.award_urls, data.general_urls),
        additional_instruction="Verify the specific ceremony date (December 18, 2025)."
    )


async def build_debut_section(evaluator: Evaluator, parent_node, data: StudioIdentification) -> None:
    """
    Build and verify the 'Debut_Game_Verification' section:
    - The award-winning game was the studio's first released game (debut).
    """
    node = evaluator.add_parallel(
        id="Debut_Game_Verification",
        desc="The award-winning game was the developer studio's first released game (debut game)",
        parent=parent_node,
        critical=True
    )

    # Existence and sources checks
    evaluator.add_custom_node(
        result=_nonempty(data.studio_name) and _nonempty(data.game_title),
        id="debut_studio_and_game_present",
        desc="Studio name and game title are provided in the answer (for debut verification)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(data.debut_urls) > 0 or len(data.studio_urls) > 0 or len(data.general_urls) > 0,
        id="debut_sources_provided",
        desc="Debut-related sources are provided",
        parent=node,
        critical=True
    )

    # Debut verification leaf
    leaf_debut = evaluator.add_leaf(
        id="game_is_debut",
        desc="The award-winning game is the studio's debut (first released) game",
        parent=node,
        critical=True
    )
    debut_claim = (
        f"The game '{data.game_title or ''}' was the first released (debut) game by the studio '{data.studio_name or ''}'."
    )
    await evaluator.verify(
        claim=debut_claim,
        node=leaf_debut,
        sources=_merge_sources(data.debut_urls, data.studio_urls, data.general_urls),
        additional_instruction=(
            "Confirm that this was the studio's first ever released game (a debut). "
            "Accept synonymous phrasing like 'first title', 'debut release', or 'first commercial release'."
        )
    )


async def build_founding_section(evaluator: Evaluator, parent_node, data: StudioIdentification) -> None:
    """
    Build and verify 'Studio_Founding_Date':
    - The studio was founded between 2015 and 2022 inclusive.
    """
    node = evaluator.add_parallel(
        id="Studio_Founding_Date",
        desc="The developer studio was founded between 2015 and 2022, inclusive",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(data.studio_name),
        id="founding_studio_present",
        desc="Studio name provided (for founding verification)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(data.founding_urls) > 0 or len(data.studio_urls) > 0 or len(data.general_urls) > 0,
        id="founding_sources_provided",
        desc="Founding-related sources are provided",
        parent=node,
        critical=True
    )

    leaf_range = evaluator.add_leaf(
        id="founded_between_2015_2022",
        desc="Studio founded between 2015 and 2022 inclusive",
        parent=node,
        critical=True
    )
    founding_claim = (
        f"The studio '{data.studio_name or ''}' was founded between 2015 and 2022, inclusive."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=leaf_range,
        sources=_merge_sources(data.founding_urls, data.studio_urls, data.general_urls),
        additional_instruction=(
            "Use the provided sources to determine the founding year and check whether it falls within the range 2015–2022 inclusive."
        )
    )


async def build_geographic_section(evaluator: Evaluator, parent_node, data: StudioIdentification) -> None:
    """
    Build and verify 'Geographic_Location' with two critical children:
    - Country_Location: located in France
    - City_Name_Constraint: headquarters city starts with 'M'
    """
    node = evaluator.add_parallel(
        id="Geographic_Location",
        desc="Verify the studio's geographic location meets the specified criteria",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(data.studio_name),
        id="geo_studio_present",
        desc="Studio name provided (for geographic verification)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(data.location_urls) > 0 or len(data.studio_urls) > 0 or len(data.general_urls) > 0,
        id="location_sources_provided",
        desc="Location-related sources are provided",
        parent=node,
        critical=True
    )

    # Country: France
    country_leaf = evaluator.add_leaf(
        id="Country_Location",
        desc="The developer studio is located in France",
        parent=node,
        critical=True
    )
    country_claim = f"The studio '{data.studio_name or ''}' is located in France."
    await evaluator.verify(
        claim=country_claim,
        node=country_leaf,
        sources=_merge_sources(data.location_urls, data.studio_urls, data.general_urls),
        additional_instruction="Confirm the primary location/headquarters country is France."
    )

    # City starts with 'M'
    city_leaf = evaluator.add_leaf(
        id="City_Name_Constraint",
        desc="The city where the studio is headquartered has a name starting with the letter 'M'",
        parent=node,
        critical=True
    )
    city_text = data.city if _nonempty(data.city) else "its headquarters city in France"
    city_claim = (
        f"The studio '{data.studio_name or ''}' is headquartered in {city_text}, and the city name starts with the letter 'M'."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=_merge_sources(data.location_urls, data.studio_urls, data.general_urls),
        additional_instruction=(
            "Identify the headquarters city from the sources and verify that the city's name begins with the letter 'M' "
            "(e.g., Marseille, Montpellier, Montreuil, Metz, etc.). Allow minor orthographic variants or hyphenations."
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
    Evaluate an answer for the indie studio identification task.
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_studio_identification(),
        template_class=StudioIdentification,
        extraction_name="studio_identification"
    )

    # Add a summary of the criteria as ground truth context (not the actual answer)
    evaluator.add_ground_truth({
        "criteria": {
            "award": "Game won 'Game of the Year' at The Indie Game Awards 2025 (ceremony on Dec 18, 2025)",
            "debut": "That game was the studio's first released (debut) game",
            "founding_range": "Studio founded between 2015 and 2022 inclusive",
            "country": "Studio located in France",
            "city_initial": "Headquarters city starts with 'M'"
        }
    }, gt_type="evaluation_criteria")

    # Build the rubric tree according to JSON (with detailed leaves)
    main = evaluator.add_parallel(
        id="Game_Studio_Identification",
        desc="Identify the name of the indie game development studio that meets all specified criteria",
        parent=root,
        critical=True
    )

    # Sections
    await build_award_section(evaluator, main, extracted)
    await build_debut_section(evaluator, main, extracted)
    await build_founding_section(evaluator, main, extracted)
    await build_geographic_section(evaluator, main, extracted)

    # Return final summary
    return evaluator.get_summary()