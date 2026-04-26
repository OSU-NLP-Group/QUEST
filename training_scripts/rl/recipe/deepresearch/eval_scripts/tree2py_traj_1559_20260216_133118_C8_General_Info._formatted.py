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
TASK_ID = "fifa_wc26_cities"
TASK_DESCRIPTION = (
    "The FIFA World Cup 2026 will be hosted across 16 cities in Canada, Mexico, and the United States, featuring 104 total matches. "
    "Among all venues, one U.S. stadium will host the maximum number of games at 9 matches, while another U.S. stadium will host 8 matches including a semifinal.\n\n"
    "Identify the following 4 specific FIFA World Cup 2026 host cities, providing complete information for each:\n\n"
    "1. The U.S. city whose stadium will host the most games (9 matches total)\n"
    "2. The U.S. city whose stadium will host 8 matches including a semifinal\n"
    "3. The northernmost host city among all 16 FIFA World Cup 2026 host cities\n"
    "4. The southernmost host city among all 16 FIFA World Cup 2026 host cities\n\n"
    "For each city, provide:\n"
    "- The city name\n"
    "- The official stadium name (as it will be branded during FIFA events)\n"
    "- The host country\n"
    "- The number of matches hosted (for cities 1 and 2)\n"
    "- Confirmation of geographic position (for cities 3 and 4)"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CityItem(BaseModel):
    city_name: Optional[str] = None
    stadium_official_name: Optional[str] = None
    country: Optional[str] = None
    matches_hosted: Optional[str] = None  # Keep as string to tolerate words like "eight"
    sources: List[str] = Field(default_factory=list)


class CitiesExtraction(BaseModel):
    most_games_us_city: Optional[CityItem] = None
    semifinal_us_city: Optional[CityItem] = None
    northernmost_city: Optional[CityItem] = None
    southernmost_city: Optional[CityItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wc26_cities() -> str:
    return """
Extract from the answer the four specific FIFA World Cup 2026 host cities requested. For each of the following categories, extract the fields exactly as written in the answer:

1) most_games_us_city:
   - city_name: The city identified as hosting the most games (U.S. city; 9 matches total)
   - stadium_official_name: The official FIFA-branded stadium name as it will be used during the tournament (non‑commercial name, e.g., “Dallas Stadium”)
   - country: The host country for this city
   - matches_hosted: The number of matches this stadium will host, as stated in the answer (e.g., "9" or "nine"); if not provided, set to null
   - sources: All URLs cited in the answer that support any of the above information; must be actual URLs present in the answer

2) semifinal_us_city:
   - city_name: The U.S. city whose stadium will host 8 matches including a semifinal
   - stadium_official_name: The official FIFA-branded stadium name (non‑commercial)
   - country: The host country for this city
   - matches_hosted: The number of matches this stadium will host, as stated in the answer (e.g., "8" or "eight"); if not provided, set to null
   - sources: All supporting URLs cited in the answer for this item

3) northernmost_city:
   - city_name: The northernmost city among all 16 host cities
   - stadium_official_name: The official FIFA-branded stadium name (non‑commercial)
   - country: The host country for this city
   - matches_hosted: If the answer explicitly lists matches for this city, extract it; otherwise set to null
   - sources: All supporting URLs cited in the answer for this item

4) southernmost_city:
   - city_name: The southernmost city among all 16 host cities
   - stadium_official_name: The official FIFA-branded stadium name (non‑commercial)
   - country: The host country for this city
   - matches_hosted: If the answer explicitly lists matches for this city, extract it; otherwise set to null
   - sources: All supporting URLs cited in the answer for this item

General rules:
- Do not invent or infer any data not present in the answer.
- If a field is missing, return null (or empty list for sources).
- For URL extraction, return only actual URLs present in the answer (plain or markdown); include http/https. If a URL lacks protocol, prepend http://.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _sources_present(sources: Optional[List[str]]) -> bool:
    return isinstance(sources, list) and len([u for u in sources if _has_text(u)]) > 0


# Common additional instruction strings
INS_NAME_VARIANTS = (
    "Allow minor naming variations: city vs metro area vs suburb (e.g., Dallas vs Arlington; New York/New Jersey), "
    "letter casing differences, and common abbreviations (USA vs United States)."
)
INS_FIFA_BRANDING = (
    "FIFA uses non‑commercial stadium names for the 2026 tournament (e.g., 'Dallas Stadium' for AT&T Stadium). "
    "Verify the official FIFA‑branded name as stated; accept equivalence where a source explicitly maps commercial to FIFA‑branded names."
)
INS_CITY_HOST = (
    "Verify that this city is one of the 16 host cities for the FIFA World Cup 2026. "
    + INS_NAME_VARIANTS
)
INS_COUNTRY_US = (
    "Verify that the host city is in the United States (USA / U.S. / United States of America are equivalent)."
)
INS_COUNT_MAX = (
    "Verify both: (1) the total match count, and (2) that this is the most of any 2026 host venue. "
    "Accept phrases like 'most matches of any city' or similar wording. "
    "Allow minor numeric formatting differences (e.g., 'nine' vs '9')."
)
INS_COUNT_ONLY = (
    "Verify the total number of 2026 World Cup matches hosted at this stadium. "
    "Allow minor numeric formatting differences (e.g., 'eight' vs '8')."
)
INS_SEMIFINAL = (
    "Verify that this stadium will host a semifinal match for the 2026 FIFA World Cup."
)
INS_NORTH = (
    "Verify that this city is the northernmost among all 16 2026 host cities. "
    "Accept if the source explicitly states 'northernmost host city'. "
    "If latitudes are shown, higher latitude indicates more north."
)
INS_SOUTH = (
    "Verify that this city is the southernmost among all 16 2026 host cities. "
    "Accept if the source explicitly states 'southernmost host city'. "
    "If latitudes are shown, lower latitude indicates more south."
)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_most_games_us_city(evaluator: Evaluator, parent_node, info: Optional[CityItem]) -> None:
    group = evaluator.add_parallel(
        id="most_games_us_city",
        desc="U.S. city whose stadium hosts the most World Cup 2026 games (9 matches)",
        parent=parent_node,
        critical=False
    )

    exists_node = evaluator.add_custom_node(
        result=(info is not None and _has_text(info.city_name) and _has_text(info.stadium_official_name) and _sources_present(info.sources)),
        id="most_games_info_exists",
        desc="Most-games city info present with at least one source",
        parent=group,
        critical=True
    )

    # City name verification
    city_leaf = evaluator.add_leaf(
        id="most_games_city_name",
        desc="Correct city name provided",
        parent=group,
        critical=True
    )
    city_claim = f"{info.city_name} is listed as a host city for the FIFA World Cup 2026." if info and info.city_name else "This city is listed as a host city for the FIFA World Cup 2026."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_CITY_HOST
    )

    # Stadium official FIFA-branded name verification
    stadium_leaf = evaluator.add_leaf(
        id="most_games_stadium_name",
        desc="Correct official FIFA-branded stadium name provided",
        parent=group,
        critical=True
    )
    stadium_claim = (
        f"The official FIFA-branded stadium name for {info.city_name} is '{info.stadium_official_name}'."
        if info and _has_text(info.city_name) and _has_text(info.stadium_official_name)
        else "The official FIFA-branded stadium name is correctly identified for this host city."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_FIFA_BRANDING
    )

    # Country verification (United States)
    country_leaf = evaluator.add_leaf(
        id="most_games_country",
        desc="Correct host country (United States) identified",
        parent=group,
        critical=True
    )
    country_claim = (
        f"The host city {info.city_name} is in the United States."
        if info and _has_text(info.city_name)
        else "This host city is in the United States."
    )
    await evaluator.verify(
        claim=country_claim,
        node=country_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_COUNTRY_US
    )

    # Matches count and maximum verification
    count_leaf = evaluator.add_leaf(
        id="most_games_count_verification",
        desc="Verified that stadium hosts exactly 9 games, the maximum among all venues",
        parent=group,
        critical=True
    )
    if info and _has_text(info.matches_hosted):
        count_claim = (
            f"The stadium in {info.city_name} will host exactly {info.matches_hosted} matches in the 2026 FIFA World Cup, "
            f"and this is the most of any host venue."
            if _has_text(info.city_name) else
            f"The stadium will host exactly {info.matches_hosted} matches in the 2026 FIFA World Cup, and this is the most of any host venue."
        )
    else:
        # If the answer did not state a number, default the verification to the rubric's 9 with 'most' wording
        count_claim = (
            f"The stadium in {info.city_name} will host exactly 9 matches in the 2026 FIFA World Cup, the most of any host venue."
            if info and _has_text(info.city_name) else
            "This stadium will host exactly 9 matches in the 2026 FIFA World Cup, the most of any host venue."
        )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_COUNT_MAX
    )


async def verify_semifinal_us_city(evaluator: Evaluator, parent_node, info: Optional[CityItem]) -> None:
    group = evaluator.add_parallel(
        id="semifinal_us_city",
        desc="U.S. city whose stadium hosts 8 matches including a semifinal",
        parent=parent_node,
        critical=False
    )

    exists_node = evaluator.add_custom_node(
        result=(info is not None and _has_text(info.city_name) and _has_text(info.stadium_official_name) and _sources_present(info.sources)),
        id="semifinal_info_exists",
        desc="Semifinal city info present with at least one source",
        parent=group,
        critical=True
    )

    # City name verification
    city_leaf = evaluator.add_leaf(
        id="semifinal_city_name",
        desc="Correct city name provided",
        parent=group,
        critical=True
    )
    city_claim = f"{info.city_name} is listed as a host city for the FIFA World Cup 2026." if info and info.city_name else "This city is listed as a host city for the FIFA World Cup 2026."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_CITY_HOST
    )

    # Stadium official name verification
    stadium_leaf = evaluator.add_leaf(
        id="semifinal_stadium_name",
        desc="Correct official FIFA-branded stadium name provided",
        parent=group,
        critical=True
    )
    stadium_claim = (
        f"The official FIFA-branded stadium name for {info.city_name} is '{info.stadium_official_name}'."
        if info and _has_text(info.city_name) and _has_text(info.stadium_official_name)
        else "The official FIFA-branded stadium name is correctly identified for this host city."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_FIFA_BRANDING
    )

    # Country verification (United States)
    country_leaf = evaluator.add_leaf(
        id="semifinal_country",
        desc="Correct host country (United States) identified",
        parent=group,
        critical=True
    )
    country_claim = (
        f"The host city {info.city_name} is in the United States."
        if info and _has_text(info.city_name)
        else "This host city is in the United States."
    )
    await evaluator.verify(
        claim=country_claim,
        node=country_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_COUNTRY_US
    )

    # Game count verification (8)
    count_leaf = evaluator.add_leaf(
        id="semifinal_game_count",
        desc="Verified that stadium hosts exactly 8 games",
        parent=group,
        critical=True
    )
    if info and _has_text(info.matches_hosted):
        count_claim = (
            f"The stadium in {info.city_name} will host exactly {info.matches_hosted} matches in the 2026 FIFA World Cup."
            if _has_text(info.city_name) else
            f"The stadium will host exactly {info.matches_hosted} matches in the 2026 FIFA World Cup."
        )
    else:
        count_claim = (
            f"The stadium in {info.city_name} will host exactly 8 matches in the 2026 FIFA World Cup."
            if info and _has_text(info.city_name) else
            "This stadium will host exactly 8 matches in the 2026 FIFA World Cup."
        )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_COUNT_ONLY
    )

    # Semifinal match verification
    semi_leaf = evaluator.add_leaf(
        id="semifinal_match_verification",
        desc="Verified that stadium hosts a semifinal match",
        parent=group,
        critical=True
    )
    semi_claim = (
        f"The stadium in {info.city_name} will host a semifinal match at the 2026 FIFA World Cup."
        if info and _has_text(info.city_name)
        else "This stadium will host a 2026 FIFA World Cup semifinal match."
    )
    await evaluator.verify(
        claim=semi_claim,
        node=semi_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_SEMIFINAL
    )


async def verify_northernmost_city(evaluator: Evaluator, parent_node, info: Optional[CityItem]) -> None:
    group = evaluator.add_parallel(
        id="northernmost_city",
        desc="The northernmost city among all 16 FIFA World Cup 2026 host cities",
        parent=parent_node,
        critical=False
    )

    exists_node = evaluator.add_custom_node(
        result=(info is not None and _has_text(info.city_name) and _has_text(info.stadium_official_name) and _sources_present(info.sources)),
        id="northernmost_info_exists",
        desc="Northernmost city info present with at least one source",
        parent=group,
        critical=True
    )

    # City name verification (is a host city)
    city_leaf = evaluator.add_leaf(
        id="northernmost_city_name",
        desc="Correct city name provided",
        parent=group,
        critical=True
    )
    city_claim = f"{info.city_name} is listed as a host city for the FIFA World Cup 2026." if info and info.city_name else "This city is listed as a host city for the FIFA World Cup 2026."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_CITY_HOST
    )

    # Stadium official name verification
    stadium_leaf = evaluator.add_leaf(
        id="northernmost_stadium_name",
        desc="Correct official stadium name provided",
        parent=group,
        critical=True
    )
    stadium_claim = (
        f"The official FIFA-branded stadium name for {info.city_name} is '{info.stadium_official_name}'."
        if info and _has_text(info.city_name) and _has_text(info.stadium_official_name)
        else "The official FIFA-branded stadium name is correctly identified for this host city."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_FIFA_BRANDING
    )

    # Country verification (whatever country is provided)
    country_leaf = evaluator.add_leaf(
        id="northernmost_country",
        desc="Correct host country identified",
        parent=group,
        critical=True
    )
    if info and _has_text(info.city_name) and _has_text(info.country):
        country_claim = f"The host city {info.city_name} is in {info.country}."
    elif info and _has_text(info.city_name):
        country_claim = f"The host city {info.city_name} is in its stated host country."
    else:
        country_claim = "This host city is correctly placed in its stated host country."
    await evaluator.verify(
        claim=country_claim,
        node=country_leaf,
        sources=(info.sources if info else []),
        additional_instruction="Verify the host country for this city. " + INS_NAME_VARIANTS
    )

    # Position verification (northernmost)
    pos_leaf = evaluator.add_leaf(
        id="northernmost_position_verified",
        desc="Verified as the northernmost among all 16 host cities",
        parent=group,
        critical=True
    )
    pos_claim = (
        f"{info.city_name} is the northernmost among all 16 host cities for the FIFA World Cup 2026."
        if info and _has_text(info.city_name)
        else "This city is the northernmost among all 16 host cities for the FIFA World Cup 2026."
    )
    await evaluator.verify(
        claim=pos_claim,
        node=pos_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_NORTH
    )


async def verify_southernmost_city(evaluator: Evaluator, parent_node, info: Optional[CityItem]) -> None:
    group = evaluator.add_parallel(
        id="southernmost_city",
        desc="The southernmost city among all 16 FIFA World Cup 2026 host cities",
        parent=parent_node,
        critical=False
    )

    exists_node = evaluator.add_custom_node(
        result=(info is not None and _has_text(info.city_name) and _has_text(info.stadium_official_name) and _sources_present(info.sources)),
        id="southernmost_info_exists",
        desc="Southernmost city info present with at least one source",
        parent=group,
        critical=True
    )

    # City name verification (is a host city)
    city_leaf = evaluator.add_leaf(
        id="southernmost_city_name",
        desc="Correct city name provided",
        parent=group,
        critical=True
    )
    city_claim = f"{info.city_name} is listed as a host city for the FIFA World Cup 2026." if info and info.city_name else "This city is listed as a host city for the FIFA World Cup 2026."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_CITY_HOST
    )

    # Stadium official name verification
    stadium_leaf = evaluator.add_leaf(
        id="southernmost_stadium_name",
        desc="Correct official FIFA-branded stadium name provided",
        parent=group,
        critical=True
    )
    stadium_claim = (
        f"The official FIFA-branded stadium name for {info.city_name} is '{info.stadium_official_name}'."
        if info and _has_text(info.city_name) and _has_text(info.stadium_official_name)
        else "The official FIFA-branded stadium name is correctly identified for this host city."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_FIFA_BRANDING
    )

    # Country verification (whatever country is provided)
    country_leaf = evaluator.add_leaf(
        id="southernmost_country",
        desc="Correct host country identified",
        parent=group,
        critical=True
    )
    if info and _has_text(info.city_name) and _has_text(info.country):
        country_claim = f"The host city {info.city_name} is in {info.country}."
    elif info and _has_text(info.city_name):
        country_claim = f"The host city {info.city_name} is in its stated host country."
    else:
        country_claim = "This host city is correctly placed in its stated host country."
    await evaluator.verify(
        claim=country_claim,
        node=country_leaf,
        sources=(info.sources if info else []),
        additional_instruction="Verify the host country for this city. " + INS_NAME_VARIANTS
    )

    # Position verification (southernmost)
    pos_leaf = evaluator.add_leaf(
        id="southernmost_position_verified",
        desc="Verified as the southernmost among all 16 host cities",
        parent=group,
        critical=True
    )
    pos_claim = (
        f"{info.city_name} is the southernmost among all 16 host cities for the FIFA World Cup 2026."
        if info and _has_text(info.city_name)
        else "This city is the southernmost among all 16 host cities for the FIFA World Cup 2026."
    )
    await evaluator.verify(
        claim=pos_claim,
        node=pos_leaf,
        sources=(info.sources if info else []),
        additional_instruction=INS_SOUTH
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the FIFA World Cup 2026 host cities task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: four independent sub-tasks
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

    # Extract the four city items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_wc26_cities(),
        template_class=CitiesExtraction,
        extraction_name="wc26_cities_extraction"
    )

    # Build verification subtrees
    await verify_most_games_us_city(evaluator, root, extracted.most_games_us_city)
    await verify_semifinal_us_city(evaluator, root, extracted.semifinal_us_city)
    await verify_northernmost_city(evaluator, root, extracted.northernmost_city)
    await verify_southernmost_city(evaluator, root, extracted.southernmost_city)

    return evaluator.get_summary()