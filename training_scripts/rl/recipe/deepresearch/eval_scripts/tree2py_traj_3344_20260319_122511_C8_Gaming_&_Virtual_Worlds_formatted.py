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
TASK_ID = "gaming_events_march_2026"
TASK_DESCRIPTION = """
I'm planning my gaming schedule for March 2026 and want to stay updated on major events and releases across multiple games. Please find detailed information about the following four gaming events or updates scheduled for March 2026:

1. Pokemon GO Community Day (March 2026):
   - What is the exact date and time (including time zone specification) of the Community Day event?
   - Which Pokemon is featured during this Community Day?
   - What exclusive move can the featured Pokemon's final evolution learn during this event?
   - What special bonus is active during the event related to egg hatching?

2. Pokemon GO Five-Star Raid Schedule (March 2026):
   - Which legendary Pokemon appear in Five-Star Raids from March 4-10, 2026?
   - Which legendary Pokemon appears in Five-Star Raids from March 11-17, 2026?
   - Which Pokemon is featured during the Raid Hour on Wednesday, March 18, 2026?
   - What time do Raid Hours occur (include local time specification)?

3. Fortnite Chapter 7 Season 2 Launch:
   - What is the launch date for Fortnite Chapter 7 Season 2?
   - What is the theme or name of Season 2?
   - Confirm the chapter and season numbers.

4. Call of Duty: Warzone Season 2 Reloaded Update:
   - What is the release date for the Season 2 Reloaded update?
   - What time does the update go live (provide at least two time zones)?
   - What is the name of the new battle royale mode being introduced?
   - What is the name of the new map for this mode?

For each event, please provide a reference URL from an official source or reputable gaming news website that confirms the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CommunityDayInfo(BaseModel):
    date: Optional[str] = None  # e.g., "Saturday, March 14, 2026"
    start_time: Optional[str] = None  # e.g., "2:00 PM"
    end_time: Optional[str] = None  # e.g., "5:00 PM"
    time_zone_spec: Optional[str] = None  # e.g., "local time"
    featured_pokemon: Optional[str] = None  # e.g., "Scorbunny"
    final_evolution: Optional[str] = None  # e.g., "Cinderace"
    exclusive_move: Optional[str] = None  # e.g., "Blast Burn"
    egg_bonus: Optional[str] = None  # e.g., "1/4 egg hatch distance"
    urls: List[str] = Field(default_factory=list)


class RaidScheduleInfo(BaseModel):
    raids_march_4_10: List[str] = Field(default_factory=list)  # e.g., ["Articuno", "Zapdos", "Moltres"]
    raids_march_11_17: List[str] = Field(default_factory=list)  # e.g., ["Zacian (Hero of Many Battles)"]
    raid_hour_featured_march_18: Optional[str] = None  # e.g., "Zamazenta"
    raid_hour_time_local: Optional[str] = None  # e.g., "6:00 PM to 7:00 PM local time"
    urls: List[str] = Field(default_factory=list)


class FortniteSeasonInfo(BaseModel):
    launch_date: Optional[str] = None  # e.g., "March 19, 2026"
    theme_name: Optional[str] = None  # e.g., "Showdown"
    chapter: Optional[str] = None  # e.g., "7"
    season: Optional[str] = None  # e.g., "2"
    urls: List[str] = Field(default_factory=list)


class WarzoneUpdateInfo(BaseModel):
    release_date: Optional[str] = None  # e.g., "March 11, 2026"
    release_time: Optional[str] = None  # e.g., "9 AM PT / 12 PM ET / 5 PM GMT"
    mode_name: Optional[str] = None  # e.g., "Black Ops Royale"
    map_name: Optional[str] = None  # e.g., "Avalon"
    urls: List[str] = Field(default_factory=list)


class MarchGamingExtraction(BaseModel):
    pokemon_go_community_day: Optional[CommunityDayInfo] = None
    pokemon_go_raid_schedule: Optional[RaidScheduleInfo] = None
    fortnite_season_launch: Optional[FortniteSeasonInfo] = None
    cod_warzone_update: Optional[WarzoneUpdateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_march_gaming_events() -> str:
    return """
Extract the requested details from the answer text for each of the following four topics. Return strictly and only the JSON fields specified.

1) Pokemon GO Community Day (March 2026) -> pokemon_go_community_day:
   - date: the stated calendar date, e.g., "Saturday, March 14, 2026"
   - start_time: e.g., "2:00 PM"
   - end_time: e.g., "5:00 PM"
   - time_zone_spec: explicitly capture any phrase like "local time" or a specific time zone
   - featured_pokemon: e.g., "Scorbunny"
   - final_evolution: e.g., "Cinderace"
   - exclusive_move: e.g., "Blast Burn"
   - egg_bonus: e.g., "1/4 egg hatch distance"
   - urls: array of reference URLs provided specifically for Community Day

2) Pokemon GO Five-Star Raids (March 2026) -> pokemon_go_raid_schedule:
   - raids_march_4_10: array of legendary names for Mar 4–10 (e.g., ["Articuno","Zapdos","Moltres"])
   - raids_march_11_17: array of legendary names for Mar 11–17 (e.g., ["Zacian (Hero of Many Battles)"])
   - raid_hour_featured_march_18: the Pokemon featured on Wed, Mar 18, 2026 (e.g., "Zamazenta")
   - raid_hour_time_local: the recurring Raid Hour time window text including local time (e.g., "6:00 PM to 7:00 PM local time")
   - urls: array of reference URLs provided specifically for the raid schedule/raid hour

3) Fortnite Chapter 7 Season 2 -> fortnite_season_launch:
   - launch_date: e.g., "March 19, 2026"
   - theme_name: e.g., "Showdown"
   - chapter: e.g., "7"
   - season: e.g., "2"
   - urls: array of reference URLs provided for Fortnite details

4) Call of Duty: Warzone Season 2 Reloaded -> cod_warzone_update:
   - release_date: e.g., "March 11, 2026"
   - release_time: e.g., "9 AM PT / 12 PM ET / 5 PM GMT"
   - mode_name: e.g., "Black Ops Royale"
   - map_name: e.g., "Avalon"
   - urls: array of reference URLs provided for Warzone details

Rules:
- Extract only what appears in the answer. If a field is not present, set it to null; if an array is missing, use an empty array.
- For URLs: extract only valid, complete URLs explicitly present in the answer (including those in markdown links).
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _urls_bullet_list(urls: List[str]) -> str:
    if not urls:
        return "(no URLs)"
    return "\n".join(f"- {u}" for u in urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_pokemon_go_community_day(evaluator: Evaluator, root_parent, info: Optional[CommunityDayInfo]) -> None:
    node = evaluator.add_parallel(
        id="pokemon_go_community_day",
        desc="Pokemon GO Community Day (March 2026) details",
        parent=root_parent,
        critical=False
    )

    urls = _ensure_list(info.urls if info else [])

    # References group
    refs = evaluator.add_parallel(
        id="community_day_references",
        desc="Provides reference URL(s) that confirm the Community Day information.",
        parent=node,
        critical=True
    )

    # URL provided
    evaluator.add_custom_node(
        result=len(urls) >= 1,
        id="community_day_url_provided",
        desc="Provides ≥1 reference URL for Community Day.",
        parent=refs,
        critical=True
    )

    # URL source quality
    leaf_source_quality = evaluator.add_leaf(
        id="community_day_url_source_meets_requirement",
        desc="Reference URL(s) are from an official source or a reputable gaming news website.",
        parent=refs,
        critical=True
    )
    quality_claim = (
        "Each of the following URLs is from an official Pokemon/Niantic source or a reputable gaming news website.\n"
        f"URLs:\n{_urls_bullet_list(urls)}"
    )
    await evaluator.verify(
        claim=quality_claim,
        node=leaf_source_quality,
        additional_instruction=(
            "Accept as official: pokemongolive.com, pokemon.com (Pokemon GO section), nianticlabs.com, "
            "support.pokemon.com. Accept as reputable news: IGN, GameSpot, Polygon, Eurogamer, Kotaku, The Verge, "
            "PC Gamer, VGC, etc. Judge based on domain reputation."
        )
    )

    # URL corroborates details
    leaf_refs_corroborate = evaluator.add_leaf(
        id="community_day_url_corroborates_details",
        desc="The reference URL(s) corroborate the stated Community Day details (date/time, featured Pokemon, exclusive move, egg-hatching bonus).",
        parent=refs,
        critical=True
    )
    combined_claim = (
        "The provided sources confirm all of the following for Pokemon GO Community Day (March 2026): "
        "1) Date/time: Saturday, March 14, 2026, from 2:00 PM to 5:00 PM local time. "
        "2) Featured Pokemon: Scorbunny. "
        "3) Exclusive move for the final evolution Cinderace: Blast Burn. "
        "4) Event bonus: 1/4 egg hatch distance."
    )
    await evaluator.verify(
        claim=combined_claim,
        node=leaf_refs_corroborate,
        sources=urls,
        additional_instruction="Ensure the webpages explicitly state or clearly imply each listed item."
    )

    # Individual detail checks (all critical)
    leaf_date_time = evaluator.add_leaf(
        id="community_day_date_time",
        desc="States Community Day occurs on Saturday, March 14, 2026, from 2:00 PM to 5:00 PM local time (includes time zone/local-time specification).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Pokemon GO Community Day in March 2026 occurs on Saturday, March 14, 2026, from 2:00 PM to 5:00 PM local time.",
        node=leaf_date_time,
        sources=urls,
        additional_instruction="Verify date and the full time window including 'local time' or equivalent local-time phrasing."
    )

    leaf_featured = evaluator.add_leaf(
        id="community_day_featured_pokemon",
        desc="Identifies Scorbunny as the featured Pokemon.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The featured Pokémon for the March 2026 Community Day is Scorbunny.",
        node=leaf_featured,
        sources=urls
    )

    leaf_move = evaluator.add_leaf(
        id="community_day_exclusive_move",
        desc="States the final evolution (Cinderace) can learn the exclusive move Blast Burn during the event.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="During the March 2026 Community Day event, Cinderace (the final evolution of Scorbunny) can learn the exclusive move Blast Burn.",
        node=leaf_move,
        sources=urls,
        additional_instruction="Allow phrasing variants like 'Cinderace will learn Blast Burn' or 'evolves to learn Blast Burn.'"
    )

    leaf_bonus = evaluator.add_leaf(
        id="community_day_egg_bonus",
        desc="States the event bonus includes 1/4 egg hatch distance.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The event bonus includes 1/4 egg hatch distance.",
        node=leaf_bonus,
        sources=urls
    )


async def verify_pokemon_go_raid_schedule(evaluator: Evaluator, root_parent, info: Optional[RaidScheduleInfo]) -> None:
    node = evaluator.add_parallel(
        id="pokemon_go_raid_schedule",
        desc="Pokemon GO Five-Star Raid schedule and Raid Hour details",
        parent=root_parent,
        critical=False
    )

    urls = _ensure_list(info.urls if info else [])

    # References group
    refs = evaluator.add_parallel(
        id="raid_schedule_references",
        desc="Provides reference URL(s) that confirm the raid schedule / raid hour information.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls) >= 1,
        id="raid_schedule_url_provided",
        desc="Provides ≥1 reference URL for the raid schedule/raid hour information.",
        parent=refs,
        critical=True
    )

    leaf_source_quality = evaluator.add_leaf(
        id="raid_schedule_url_source_meets_requirement",
        desc="Reference URL(s) are from an official source or a reputable gaming news website (as required by the prompt).",
        parent=refs,
        critical=True
    )
    quality_claim = (
        "Each of the following URLs is from an official Pokemon/Niantic source or a reputable gaming news website.\n"
        f"URLs:\n{_urls_bullet_list(urls)}"
    )
    await evaluator.verify(
        claim=quality_claim,
        node=leaf_source_quality,
        additional_instruction=(
            "Accept as official: pokemongolive.com, pokemon.com (Pokemon GO section), nianticlabs.com, "
            "support.pokemon.com. Accept as reputable news: IGN, GameSpot, Polygon, Eurogamer, Kotaku, The Verge, "
            "PC Gamer, VGC, etc. Judge based on domain reputation."
        )
    )

    leaf_refs_corroborate = evaluator.add_leaf(
        id="raid_schedule_url_corroborates_details",
        desc="The reference URL(s) corroborate the stated raid schedule/raid hour details.",
        parent=refs,
        critical=True
    )
    combined_claim = (
        "The provided sources confirm all of the following for Pokemon GO in March 2026: "
        "1) From March 4–10, 2026, Five-Star Raids feature Articuno, Zapdos, and Moltres. "
        "2) From March 11–17, 2026, Five-Star Raids feature Zacian (Hero of Many Battles). "
        "3) Starting March 18, 2026, the Five-Star Raid boss is Zamazenta, and it is featured during the Raid Hour on Wednesday, March 18, 2026. "
        "4) Raid Hours occur every Wednesday from 6:00 PM to 7:00 PM local time."
    )
    await evaluator.verify(
        claim=combined_claim,
        node=leaf_refs_corroborate,
        sources=urls,
        additional_instruction="Ensure each date window and boss is explicitly supported by the sources."
    )

    # Individual detail checks (all critical)
    leaf_4_10 = evaluator.add_leaf(
        id="five_star_raids_march_4_10",
        desc="Identifies Articuno, Zapdos, and Moltres as appearing in Five-Star Raids from March 4–10, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="From March 4–10, 2026, Pokemon GO Five-Star Raids feature Articuno, Zapdos, and Moltres.",
        node=leaf_4_10,
        sources=urls
    )

    leaf_11_17 = evaluator.add_leaf(
        id="five_star_raids_march_11_17",
        desc="Identifies Zacian (Hero of Many Battles) as appearing in Five-Star Raids from March 11–17, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="From March 11–17, 2026, Pokemon GO Five-Star Raids feature Zacian (Hero of Many Battles).",
        node=leaf_11_17,
        sources=urls
    )

    leaf_march_18 = evaluator.add_leaf(
        id="march_18_zamazenta_raid_boss_and_raid_hour",
        desc="Identifies Zamazenta as (a) the Five-Star Raid boss starting March 18, 2026 and (b) the Pokemon featured during the Raid Hour on Wednesday, March 18, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Starting March 18, 2026, the Five-Star Raid boss is Zamazenta, and Zamazenta is featured during the Raid Hour on Wednesday, March 18, 2026.",
        node=leaf_march_18,
        sources=urls
    )

    leaf_raid_hour_time = evaluator.add_leaf(
        id="raid_hour_time",
        desc="States Raid Hours occur every Wednesday from 6:00 PM to 7:00 PM local time (includes local time specification).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Pokemon GO Raid Hours occur every Wednesday from 6:00 PM to 7:00 PM local time.",
        node=leaf_raid_hour_time,
        sources=urls
    )


async def verify_fortnite_season_launch(evaluator: Evaluator, root_parent, info: Optional[FortniteSeasonInfo]) -> None:
    node = evaluator.add_parallel(
        id="fortnite_season_launch",
        desc="Fortnite Chapter 7 Season 2 launch details",
        parent=root_parent,
        critical=False
    )

    urls = _ensure_list(info.urls if info else [])

    refs = evaluator.add_parallel(
        id="fortnite_references",
        desc="Provides reference URL(s) that confirm the Fortnite season information.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls) >= 1,
        id="fortnite_url_provided",
        desc="Provides ≥1 reference URL for the Fortnite season information.",
        parent=refs,
        critical=True
    )

    leaf_source_quality = evaluator.add_leaf(
        id="fortnite_url_source_meets_requirement",
        desc="Reference URL(s) are from an official source or a reputable gaming news website (as required by the prompt).",
        parent=refs,
        critical=True
    )
    quality_claim = (
        "Each of the following URLs is from an official Epic/Fortnite source or a reputable gaming news website.\n"
        f"URLs:\n{_urls_bullet_list(urls)}"
    )
    await evaluator.verify(
        claim=quality_claim,
        node=leaf_source_quality,
        additional_instruction=(
            "Accept as official: epicgames.com, fortnite.com, news.fortnite.com. "
            "Accept as reputable news: IGN, GameSpot, Polygon, Eurogamer, Kotaku, The Verge, PC Gamer, VGC, etc."
        )
    )

    leaf_refs_corroborate = evaluator.add_leaf(
        id="fortnite_url_corroborates_details",
        desc="The reference URL(s) corroborate the stated Fortnite season details (launch date, theme/name, chapter/season numbering).",
        parent=refs,
        critical=True
    )
    combined_claim = (
        "The provided sources confirm all of the following for Fortnite Chapter 7 Season 2: "
        "1) Launch date: March 19, 2026. "
        "2) Season theme/name: 'Showdown' focusing on one-on-one rivalries. "
        "3) Chapter and season numbering: Chapter 7, Season 2."
    )
    await evaluator.verify(
        claim=combined_claim,
        node=leaf_refs_corroborate,
        sources=urls
    )

    leaf_launch_date = evaluator.add_leaf(
        id="fortnite_launch_date",
        desc="States Fortnite Chapter 7 Season 2 launches on March 19, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Fortnite Chapter 7 Season 2 launches on March 19, 2026.",
        node=leaf_launch_date,
        sources=urls
    )

    leaf_theme = evaluator.add_leaf(
        id="fortnite_theme_name",
        desc="States the Season 2 theme/name is 'Showdown' (focusing on one-on-one rivalries).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Season 2 theme/name is 'Showdown', focusing on one-on-one rivalries.",
        node=leaf_theme,
        sources=urls
    )

    leaf_numbers = evaluator.add_leaf(
        id="fortnite_chapter_season_numbers",
        desc="Confirms the chapter and season numbers as Chapter 7, Season 2.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="It is Chapter 7, Season 2.",
        node=leaf_numbers,
        sources=urls
    )


async def verify_warzone_update(evaluator: Evaluator, root_parent, info: Optional[WarzoneUpdateInfo]) -> None:
    node = evaluator.add_parallel(
        id="cod_warzone_update",
        desc="Call of Duty: Warzone Season 2 Reloaded update details",
        parent=root_parent,
        critical=False
    )

    urls = _ensure_list(info.urls if info else [])

    refs = evaluator.add_parallel(
        id="warzone_references",
        desc="Provides reference URL(s) that confirm the Warzone Season 2 Reloaded information.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls) >= 1,
        id="warzone_url_provided",
        desc="Provides ≥1 reference URL for the Warzone Season 2 Reloaded information.",
        parent=refs,
        critical=True
    )

    leaf_source_quality = evaluator.add_leaf(
        id="warzone_url_source_meets_requirement",
        desc="Reference URL(s) are from an official source or a reputable gaming news website (as required by the prompt).",
        parent=refs,
        critical=True
    )
    quality_claim = (
        "Each of the following URLs is from an official Call of Duty/Activision source or a reputable gaming news website.\n"
        f"URLs:\n{_urls_bullet_list(urls)}"
    )
    await evaluator.verify(
        claim=quality_claim,
        node=leaf_source_quality,
        additional_instruction=(
            "Accept as official: callofduty.com, activision.com, blog.activision.com. "
            "Accept as reputable news: IGN, GameSpot, Polygon, Eurogamer, Kotaku, The Verge, PC Gamer, VGC, etc."
        )
    )

    leaf_refs_corroborate = evaluator.add_leaf(
        id="warzone_url_corroborates_details",
        desc="The reference URL(s) corroborate the stated Warzone details (release date, release time with ≥2 time zones, mode name, map name).",
        parent=refs,
        critical=True
    )
    combined_claim = (
        "The provided sources confirm all of the following for Call of Duty: Warzone Season 2 Reloaded: "
        "1) Release date: March 11, 2026. "
        "2) Go-live time: 9 AM PT / 12 PM ET / 5 PM GMT (at least two time zones provided). "
        "3) New battle royale mode: Black Ops Royale. "
        "4) New map for this mode: Avalon."
    )
    await evaluator.verify(
        claim=combined_claim,
        node=leaf_refs_corroborate,
        sources=urls
    )

    # Individual detail checks (all critical)
    leaf_release_date = evaluator.add_leaf(
        id="warzone_reloaded_release_date",
        desc="States Warzone Season 2 Reloaded launches on March 11, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Warzone Season 2 Reloaded launches on March 11, 2026.",
        node=leaf_release_date,
        sources=urls
    )

    leaf_release_time = evaluator.add_leaf(
        id="warzone_reloaded_release_time",
        desc="States the go-live time is 9 AM PT / 12 PM ET / 5 PM GMT, and includes at least two time zones.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The go-live time for Warzone Season 2 Reloaded is 9 AM PT / 12 PM ET / 5 PM GMT (includes at least two time zones).",
        node=leaf_release_time,
        sources=urls,
        additional_instruction="Minor formatting variations (e.g., 9:00 a.m. PT) should still be accepted as equivalent."
    )

    leaf_mode = evaluator.add_leaf(
        id="warzone_new_mode",
        desc="Identifies the new battle royale mode as Black Ops Royale.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The new battle royale mode introduced with this update is called Black Ops Royale.",
        node=leaf_mode,
        sources=urls
    )

    leaf_map = evaluator.add_leaf(
        id="warzone_new_map",
        desc="Identifies the new map for this mode as Avalon.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The new map for Black Ops Royale is named Avalon.",
        node=leaf_map,
        sources=urls
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
    Evaluate an answer for the March 2026 gaming events task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall, four independent topics
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_march_gaming_events(),
        template_class=MarchGamingExtraction,
        extraction_name="gaming_events_march_2026"
    )

    # Add expected facts as ground truth context (for transparency)
    evaluator.add_ground_truth({
        "community_day_expected": {
            "date_time": "Saturday, March 14, 2026, 2:00 PM–5:00 PM local time",
            "featured_pokemon": "Scorbunny",
            "exclusive_move_final_evo": "Cinderace learns Blast Burn",
            "egg_bonus": "1/4 egg hatch distance"
        },
        "raid_schedule_expected": {
            "mar_4_10": ["Articuno", "Zapdos", "Moltres"],
            "mar_11_17": ["Zacian (Hero of Many Battles)"],
            "mar_18": "Zamazenta (boss and featured Raid Hour)",
            "raid_hour_time": "Every Wednesday 6:00 PM–7:00 PM local time"
        },
        "fortnite_expected": {
            "launch_date": "March 19, 2026",
            "theme_name": "Showdown",
            "chapter_season": "Chapter 7, Season 2"
        },
        "warzone_expected": {
            "release_date": "March 11, 2026",
            "release_time": "9 AM PT / 12 PM ET / 5 PM GMT",
            "mode_name": "Black Ops Royale",
            "map_name": "Avalon"
        }
    }, gt_type="expected_facts")

    # Build verification subtrees per topic
    await verify_pokemon_go_community_day(evaluator, root, extracted.pokemon_go_community_day)
    await verify_pokemon_go_raid_schedule(evaluator, root, extracted.pokemon_go_raid_schedule)
    await verify_fortnite_season_launch(evaluator, root, extracted.fortnite_season_launch)
    await verify_warzone_update(evaluator, root, extracted.cod_warzone_update)

    return evaluator.get_summary()