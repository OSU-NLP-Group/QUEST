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
TASK_ID = "ent_ref_2024_2026"
TASK_DESCRIPTION = """
You are compiling an entertainment industry reference guide for the 2024-2026 period. Identify the following five specific entertainment references:

1. Young HBO Actor: The youngest actor (by birth year) who has a lead role in an HBO fantasy or drama series that premiered in 2026. Provide the actor's full name and birth year.
2. NBC Parent-Child Acting Duo: A parent and child acting pair who both portrayed the same character (at different ages/time periods) in Season 2 of an NBC television series that premiered in January 2026. Provide both names and the series title.
3. Oscar Record Film: A 2016 film that received exactly 14 Academy Award nominations (tying the all-time record with "All About Eve" and "Titanic") and won exactly 6 Oscars. Provide the film title, number of nominations, and number of wins.
4. Netflix Final Season: A Netflix original series whose eighth and final season concluded with its last episode released specifically on December 31, 2025. Provide the series title, season number, and the final episode's release date.
5. CBS Newsmagazine Program: The long-running CBS newsmagazine program that has a standard runtime of 60 minutes and features hard-hitting investigative reports, newsmaker interviews, and in-depth profiles. Provide the program title and its typical runtime.

For each item, provide supporting reference URL(s) that verify the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Item1Extraction(BaseModel):
    actor_full_name: Optional[str] = None
    actor_birth_year: Optional[str] = None
    series_title: Optional[str] = None
    series_network: Optional[str] = None
    series_genre: Optional[str] = None
    premiere_year: Optional[str] = None
    lead_role_statement: Optional[str] = None
    youngest_among_leads_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Item2Extraction(BaseModel):
    parent_actor_name: Optional[str] = None
    child_actor_name: Optional[str] = None
    series_title: Optional[str] = None
    character_name: Optional[str] = None
    season_number: Optional[str] = None
    season2_premiere_date: Optional[str] = None  # Prefer explicit date, else month/year
    network: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Item3Extraction(BaseModel):
    film_title: Optional[str] = None
    release_year: Optional[str] = None
    academy_award_nominations: Optional[str] = None
    academy_award_wins: Optional[str] = None
    record_tie_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Item4Extraction(BaseModel):
    series_title: Optional[str] = None
    platform: Optional[str] = None
    is_netflix_original: Optional[str] = None  # "yes"/"true"/"original" etc.
    season_number: Optional[str] = None
    is_final_season: Optional[str] = None
    final_episode_release_date: Optional[str] = None  # Expect "December 31, 2025" or "Dec 31, 2025"
    sources: List[str] = Field(default_factory=list)


class Item5Extraction(BaseModel):
    program_title: Optional[str] = None
    network: Optional[str] = None  # Expect "CBS"
    format: Optional[str] = None   # Expect "newsmagazine"
    typical_runtime: Optional[str] = None  # Expect "60 minutes"
    features: Optional[str] = None  # Should include investigative reports, newsmaker interviews, in-depth profiles
    long_running_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EntertainmentExtraction(BaseModel):
    item1: Optional[Item1Extraction] = None
    item2: Optional[Item2Extraction] = None
    item3: Optional[Item3Extraction] = None
    item4: Optional[Item4Extraction] = None
    item5: Optional[Item5Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_entertainment() -> str:
    return """
    Extract structured information for FIVE items from the provided answer. Extract ONLY what is explicitly stated in the answer text. If any field is missing, return null for that field. For each item, also extract the list of reference URLs (sources) that the answer cites for that item. The 'sources' arrays must contain actual URLs presented in the answer (plain or markdown links). Do not invent URLs.

    Extract as JSON with the following structure:

    item1: {
      actor_full_name: string | null,
      actor_birth_year: string | null,
      series_title: string | null,
      series_network: string | null,
      series_genre: string | null,  // e.g., "fantasy", "drama", "fantasy drama"
      premiere_year: string | null, // year the series premiered
      lead_role_statement: string | null, // any text in the answer asserting lead role
      youngest_among_leads_statement: string | null, // any text asserting youngest among leads
      sources: string[] // URLs cited for item 1 only
    }

    item2: {
      parent_actor_name: string | null,
      child_actor_name: string | null,
      series_title: string | null,
      character_name: string | null,
      season_number: string | null,
      season2_premiere_date: string | null, // date string or "January 2026"
      network: string | null,
      sources: string[]
    }

    item3: {
      film_title: string | null,
      release_year: string | null,
      academy_award_nominations: string | null, // number as string if present
      academy_award_wins: string | null,        // number as string if present
      record_tie_statement: string | null,      // any text mentioning tie with All About Eve and Titanic
      sources: string[]
    }

    item4: {
      series_title: string | null,
      platform: string | null,
      is_netflix_original: string | null, // e.g., "Netflix original", "original"
      season_number: string | null,
      is_final_season: string | null,     // any text stating final season
      final_episode_release_date: string | null, // expected to be "December 31, 2025" or similar
      sources: string[]
    }

    item5: {
      program_title: string | null,
      network: string | null,
      format: string | null,         // "newsmagazine" etc.
      typical_runtime: string | null,// e.g., "60 minutes"
      features: string | null,       // text mentioning investigative reports, newsmaker interviews, in-depth profiles
      long_running_description: string | null, // text indicating long-running
      sources: string[]
    }

    Rules:
    - Keep all numeric values as strings (e.g., "2014", "14", "6", "60 minutes").
    - 'sources' arrays must include only the URLs explicitly shown in the answer for that item.
    - If the answer has fewer than five items or omits some fields, still return the JSON with nulls where missing.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if non_empty_str(u)]) > 0


# --------------------------------------------------------------------------- #
# Verification functions per item                                             #
# --------------------------------------------------------------------------- #
async def verify_item_1(evaluator: Evaluator, parent_node, data: Optional[Item1Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_1_Young_HBO_Actor_2026",
        desc="Youngest (by birth year) lead actor in an HBO fantasy/drama series that premiered in 2026; provide actor name and birth year with sources.",
        parent=parent_node,
        critical=False
    )

    name = data.actor_full_name if data else None
    byear = data.actor_birth_year if data else None
    series = data.series_title if data else None
    sources = data.sources if data else []

    # 1A: Actor full name provided (existence)
    evaluator.add_custom_node(
        result=non_empty_str(name),
        id="1A_Actor_full_name_provided",
        desc="Answer provides the actor’s full name.",
        parent=node,
        critical=True
    )

    # 1B: Actor birth year is stated and is 2014 (URL-verified)
    leaf_1b = evaluator.add_leaf(
        id="1B_Actor_birth_year_is_2014",
        desc="Actor birth year is stated and is 2014.",
        parent=node,
        critical=True
    )
    claim_1b = f"The actor {name or '[MISSING NAME]'} was born in 2014."
    await evaluator.verify(
        claim=claim_1b,
        node=leaf_1b,
        sources=sources,
        additional_instruction="Verify the person's birth year from reliable sources on the provided URLs. Accept if the source clearly states 2014."
    )

    # 1C: HBO fantasy/drama and premiered 2026 (single combined check as rubric requires)
    leaf_1c = evaluator.add_leaf(
        id="1C_Series_is_HBO_fantasy_or_drama_and_premiered_2026",
        desc="An HBO fantasy or drama series is identified and its premiere year is 2026.",
        parent=node,
        critical=True
    )
    claim_1c = f"The series titled '{series or '[MISSING SERIES]'}' is an HBO fantasy or drama series and it premiered in 2026."
    await evaluator.verify(
        claim=claim_1c,
        node=leaf_1c,
        sources=sources,
        additional_instruction="Check that the network is HBO (or HBO-branded streaming) and that the genre is fantasy or drama. Confirm the premiere year is 2026 (first release of the series)."
    )

    # 1D: Lead role
    leaf_1d = evaluator.add_leaf(
        id="1D_Lead_role",
        desc="Actor is a lead role in the identified series.",
        parent=node,
        critical=True
    )
    claim_1d = f"{name or '[MISSING NAME]'} has a lead role in the series '{series or '[MISSING SERIES]'}'."
    await evaluator.verify(
        claim=claim_1d,
        node=leaf_1d,
        sources=sources,
        additional_instruction="Confirm that the actor is among the lead/main cast (not merely recurring/guest)."
    )

    # 1E: Youngest by birth year among lead actors
    leaf_1e = evaluator.add_leaf(
        id="1E_Youngest_by_birth_year_among_lead_cast",
        desc="Chosen actor is the youngest by birth year among the series’ lead actors (latest birth year among leads).",
        parent=node,
        critical=True
    )
    claim_1e = f"Among the lead cast of '{series or '[MISSING SERIES]'}', {name or '[MISSING NAME]'} is the youngest by birth year."
    await evaluator.verify(
        claim=claim_1e,
        node=leaf_1e,
        sources=sources,
        additional_instruction="Use the provided URLs to compare lead actors' birth years. Consider the youngest as the one with the latest birth year. Allow minor ambiguities only if clearly resolved by sources."
    )

    # 1F: Supporting references exist for Item 1
    evaluator.add_custom_node(
        result=has_sources(sources),
        id="1F_Supporting_reference_URLs_cover_key_claims",
        desc="Provides reference URL(s) that substantively support Item 1’s key claims (actor identity, birth year=2014, series HBO+genre, premiere year=2026, lead role, and youngest comparison).",
        parent=node,
        critical=True
    )


async def verify_item_2(evaluator: Evaluator, parent_node, data: Optional[Item2Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_2_NBC_Parent_Child_Acting_Duo",
        desc="Parent-child acting pair portraying the same character at different ages/time periods in Season 2 of an NBC series whose Season 2 premiered in January 2026; include sources.",
        parent=parent_node,
        critical=False
    )

    parent_name = data.parent_actor_name if data else None
    child_name = data.child_actor_name if data else None
    series = data.series_title if data else None
    character = data.character_name if data else None
    season_number = data.season_number if data else None
    premiere_date = data.season2_premiere_date if data else None
    network = data.network if data else None
    sources = data.sources if data else []

    # 2A: Parent and child names provided
    evaluator.add_custom_node(
        result=non_empty_str(parent_name) and non_empty_str(child_name),
        id="2A_Parent_and_child_names_provided",
        desc="Answer provides both the parent actor’s name and the child actor’s name.",
        parent=node,
        critical=True
    )

    # 2B: Series title provided
    evaluator.add_custom_node(
        result=non_empty_str(series),
        id="2B_Series_title_provided",
        desc="Answer provides the series title.",
        parent=node,
        critical=True
    )

    # 2C: Parent-child relationship (URL-verified)
    leaf_2c = evaluator.add_leaf(
        id="2C_Parent_child_relationship",
        desc="The two named actors are a real parent-child pair.",
        parent=node,
        critical=True
    )
    claim_2c = f"{parent_name or '[MISSING PARENT]'} and {child_name or '[MISSING CHILD]'} are a real parent-child pair."
    await evaluator.verify(
        claim=claim_2c,
        node=leaf_2c,
        sources=sources,
        additional_instruction="Verify familial relationship: the first person is the parent of the second person (biological or legally recognized)."
    )

    # 2D: Same character in Season 2 at different ages/time periods
    leaf_2d = evaluator.add_leaf(
        id="2D_Same_character_different_ages_time_periods_in_season_2",
        desc="Both actors portrayed the same character at different ages/time periods in Season 2.",
        parent=node,
        critical=True
    )
    if non_empty_str(character):
        claim_2d = f"In Season 2 of '{series or '[MISSING SERIES]'}', {parent_name or '[MISSING PARENT]'} and {child_name or '[MISSING CHILD]'} both portrayed the same character '{character}', at different ages/time periods."
    else:
        claim_2d = f"In Season 2 of '{series or '[MISSING SERIES]'}', {parent_name or '[MISSING PARENT]'} and {child_name or '[MISSING CHILD]'} both portrayed the same character at different ages/time periods."
    await evaluator.verify(
        claim=claim_2d,
        node=leaf_2d,
        sources=sources,
        additional_instruction="Confirm both actors played the same role (e.g., young vs. adult versions) specifically in Season 2."
    )

    # 2E: Series is on NBC
    leaf_2e = evaluator.add_leaf(
        id="2E_Series_is_on_NBC",
        desc="The series is an NBC television series.",
        parent=node,
        critical=True
    )
    claim_2e = f"'{series or '[MISSING SERIES]'}' is an NBC television series (airs on NBC)."
    await evaluator.verify(
        claim=claim_2e,
        node=leaf_2e,
        sources=sources,
        additional_instruction="Confirm network is NBC (not a different network or streaming-only platform)."
    )

    # 2F: Season 2 premiered in January 2026
    leaf_2f = evaluator.add_leaf(
        id="2F_Season_2_premiered_in_January_2026",
        desc="Season 2 premiered in January 2026.",
        parent=node,
        critical=True
    )
    claim_2f = f"Season 2 of '{series or '[MISSING SERIES]'}' premiered in January 2026."
    await evaluator.verify(
        claim=claim_2f,
        node=leaf_2f,
        sources=sources,
        additional_instruction="Look for a Season 2 premiere date explicitly in January 2026 (any day Jan 1–31, 2026)."
    )

    # 2G: Supporting references exist
    evaluator.add_custom_node(
        result=has_sources(sources),
        id="2G_Supporting_reference_URLs_cover_key_claims",
        desc="Provides reference URL(s) that substantively support Item 2’s key claims (both actor names, parent-child relationship, same-character casting across ages/time, NBC network, Season 2, and Season 2 January 2026 premiere timing).",
        parent=node,
        critical=True
    )


async def verify_item_3(evaluator: Evaluator, parent_node, data: Optional[Item3Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_3_Oscar_Record_Film_2016",
        desc="A 2016 film with exactly 14 Academy Award nominations (tying the all-time record with All About Eve and Titanic) and exactly 6 wins; include sources.",
        parent=parent_node,
        critical=False
    )

    title = data.film_title if data else None
    year = data.release_year if data else None
    noms = data.academy_award_nominations if data else None
    wins = data.academy_award_wins if data else None
    sources = data.sources if data else []

    # 3A: Film title provided
    evaluator.add_custom_node(
        result=non_empty_str(title),
        id="3A_Film_title_provided",
        desc="Answer provides the film title.",
        parent=node,
        critical=True
    )

    # 3B: Release year is 2016 (URL-verified)
    leaf_3b = evaluator.add_leaf(
        id="3B_Release_year_2016",
        desc="Film release year is 2016.",
        parent=node,
        critical=True
    )
    claim_3b = f"The film '{title or '[MISSING TITLE]'}' was released in 2016."
    await evaluator.verify(
        claim=claim_3b,
        node=leaf_3b,
        sources=sources,
        additional_instruction="Verify original release year is 2016."
    )

    # 3C: Exactly 14 nominations
    leaf_3c = evaluator.add_leaf(
        id="3C_Exactly_14_nominations",
        desc="Film received exactly 14 Academy Award nominations.",
        parent=node,
        critical=True
    )
    claim_3c = f"The film '{title or '[MISSING TITLE]'}' received exactly 14 Academy Award nominations."
    await evaluator.verify(
        claim=claim_3c,
        node=leaf_3c,
        sources=sources,
        additional_instruction="Confirm the official or widely cited count of Academy Award nominations is exactly 14."
    )

    # 3D: Ties record with All About Eve and Titanic
    leaf_3d = evaluator.add_leaf(
        id="3D_Record_tie_with_All_About_Eve_and_Titanic",
        desc="Those 14 nominations tie the all-time record with 'All About Eve' and 'Titanic'.",
        parent=node,
        critical=True
    )
    claim_3d = f"The film '{title or '[MISSING TITLE]'}' with 14 nominations ties the all-time record with 'All About Eve' and 'Titanic'."
    await evaluator.verify(
        claim=claim_3d,
        node=leaf_3d,
        sources=sources,
        additional_instruction="Verify that 14 nominations tie the all-time record specifically matched by 'All About Eve' (1950) and 'Titanic' (1997)."
    )

    # 3E: Exactly 6 wins
    leaf_3e = evaluator.add_leaf(
        id="3E_Exactly_6_wins",
        desc="Film won exactly 6 Academy Awards.",
        parent=node,
        critical=True
    )
    claim_3e = f"The film '{title or '[MISSING TITLE]'}' won exactly 6 Academy Awards."
    await evaluator.verify(
        claim=claim_3e,
        node=leaf_3e,
        sources=sources,
        additional_instruction="Confirm the film's Oscar wins count is exactly 6."
    )

    # 3F: Answer explicitly reports both numbers (existence in answer)
    evaluator.add_custom_node(
        result=non_empty_str(noms) and non_empty_str(wins),
        id="3F_Nominations_and_wins_numbers_reported",
        desc="Answer explicitly reports both the number of nominations and the number of wins.",
        parent=node,
        critical=True
    )

    # 3G: Supporting references exist
    evaluator.add_custom_node(
        result=has_sources(sources),
        id="3G_Supporting_reference_URLs_cover_key_claims",
        desc="Provides reference URL(s) that substantively support Item 3’s key claims (2016 release, 14 nominations, record tie, and 6 wins).",
        parent=node,
        critical=True
    )


async def verify_item_4(evaluator: Evaluator, parent_node, data: Optional[Item4Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_4_Netflix_Eighth_Final_Season_Dec_31_2025",
        desc="A Netflix original series whose eighth and final season’s last episode was released on December 31, 2025; include sources.",
        parent=parent_node,
        critical=False
    )

    series = data.series_title if data else None
    platform = data.platform if data else None
    is_original = data.is_netflix_original if data else None
    season_number = data.season_number if data else None
    is_final = data.is_final_season if data else None
    final_date = data.final_episode_release_date if data else None
    sources = data.sources if data else []

    # 4A: Series title provided
    evaluator.add_custom_node(
        result=non_empty_str(series),
        id="4A_Series_title_provided",
        desc="Answer provides the series title.",
        parent=node,
        critical=True
    )

    # 4B: Netflix original
    leaf_4b = evaluator.add_leaf(
        id="4B_Netflix_original",
        desc="Series is a Netflix original series.",
        parent=node,
        critical=True
    )
    claim_4b = f"'{series or '[MISSING SERIES]'}' is a Netflix original series."
    await evaluator.verify(
        claim=claim_4b,
        node=leaf_4b,
        sources=sources,
        additional_instruction="Confirm the series is an original Netflix production (marketed as 'Netflix Original')."
    )

    # 4C: Eighth season
    leaf_4c = evaluator.add_leaf(
        id="4C_Eighth_season",
        desc="Referenced season is Season 8 (eighth season).",
        parent=node,
        critical=True
    )
    claim_4c = f"'{series or '[MISSING SERIES]'}' has an eighth season (Season 8)."
    await evaluator.verify(
        claim=claim_4c,
        node=leaf_4c,
        sources=sources,
        additional_instruction="Confirm the existence of Season 8 for the series."
    )

    # 4D: Eighth season is final
    leaf_4d = evaluator.add_leaf(
        id="4D_Eighth_season_is_final",
        desc="Season 8 is the final season.",
        parent=node,
        critical=True
    )
    claim_4d = f"Season 8 of '{series or '[MISSING SERIES]'}' is the final season."
    await evaluator.verify(
        claim=claim_4d,
        node=leaf_4d,
        sources=sources,
        additional_instruction="Check that Season 8 is officially the last/final season."
    )

    # 4E: Final episode release date is December 31, 2025
    leaf_4e = evaluator.add_leaf(
        id="4E_Final_episode_release_date_Dec_31_2025",
        desc="The last episode of the final season was released on December 31, 2025.",
        parent=node,
        critical=True
    )
    claim_4e = f"The last episode of the final season (Season 8) of '{series or '[MISSING SERIES]'}' was released on December 31, 2025."
    await evaluator.verify(
        claim=claim_4e,
        node=leaf_4e,
        sources=sources,
        additional_instruction="Confirm the exact release date of the final episode is December 31, 2025. Consider regional release timing; the date should explicitly match 31 Dec 2025."
    )

    # 4F: Answer explicitly reports season number and final release date
    evaluator.add_custom_node(
        result=non_empty_str(season_number) and non_empty_str(final_date),
        id="4F_Season_number_and_final_release_date_reported",
        desc="Answer explicitly reports the season number and the final episode release date.",
        parent=node,
        critical=True
    )

    # 4G: Supporting references exist
    evaluator.add_custom_node(
        result=has_sources(sources),
        id="4G_Supporting_reference_URLs_cover_key_claims",
        desc="Provides reference URL(s) that substantively support Item 4’s key claims (Netflix original status, Season 8, Season 8 being final, and final-episode release date of Dec 31, 2025).",
        parent=node,
        critical=True
    )


async def verify_item_5(evaluator: Evaluator, parent_node, data: Optional[Item5Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_5_CBS_Newsmagazine_Program",
        desc="Long-running CBS newsmagazine program with standard runtime of 60 minutes and described content elements; include sources.",
        parent=parent_node,
        critical=False
    )

    title = data.program_title if data else None
    network = data.network if data else None
    fmt = data.format if data else None
    runtime = data.typical_runtime if data else None
    features = data.features if data else None
    long_running = data.long_running_description if data else None
    sources = data.sources if data else []

    # 5A: Program title provided
    evaluator.add_custom_node(
        result=non_empty_str(title),
        id="5A_Program_title_provided",
        desc="Answer provides the program title.",
        parent=node,
        critical=True
    )

    # 5B: Airs on CBS
    leaf_5b = evaluator.add_leaf(
        id="5B_Airs_on_CBS",
        desc="Program airs on CBS.",
        parent=node,
        critical=True
    )
    claim_5b = f"The program '{title or '[MISSING TITLE]'}' airs on CBS."
    await evaluator.verify(
        claim=claim_5b,
        node=leaf_5b,
        sources=sources,
        additional_instruction="Confirm network affiliation is CBS."
    )

    # 5C: Newsmagazine format
    leaf_5c = evaluator.add_leaf(
        id="5C_Newsmagazine_format",
        desc="Program is a newsmagazine format.",
        parent=node,
        critical=True
    )
    claim_5c = f"'{title or '[MISSING TITLE]'}' is a newsmagazine program."
    await evaluator.verify(
        claim=claim_5c,
        node=leaf_5c,
        sources=sources,
        additional_instruction="Look for descriptors like 'newsmagazine', 'news magazine', or equivalent phrasing."
    )

    # 5D: Standard runtime 60 minutes
    leaf_5d = evaluator.add_leaf(
        id="5D_Standard_runtime_60_minutes",
        desc="Program’s typical/standard runtime is 60 minutes.",
        parent=node,
        critical=True
    )
    claim_5d = f"The typical runtime of '{title or '[MISSING TITLE]'}' is 60 minutes."
    await evaluator.verify(
        claim=claim_5d,
        node=leaf_5d,
        sources=sources,
        additional_instruction="Confirm standard episode length is 60 minutes (one hour)."
    )

    # 5E: Features investigative reports, newsmaker interviews, in-depth profiles
    leaf_5e = evaluator.add_leaf(
        id="5E_Features_investigative_reports_interviews_profiles",
        desc="Program features investigative reports, newsmaker interviews, and in-depth profiles.",
        parent=node,
        critical=True
    )
    claim_5e = f"'{title or '[MISSING TITLE]'}' features investigative reports, newsmaker interviews, and in-depth profiles."
    await evaluator.verify(
        claim=claim_5e,
        node=leaf_5e,
        sources=sources,
        additional_instruction="Accept close synonyms (e.g., 'investigative journalism', 'interviews with major figures', 'in-depth segments/profiles')."
    )

    # 5F: Long-running characterization
    leaf_5f = evaluator.add_leaf(
        id="5F_Long_running",
        desc="Program is accurately characterized as long-running.",
        parent=node,
        critical=True
    )
    claim_5f = f"'{title or '[MISSING TITLE]'}' is a long-running program."
    await evaluator.verify(
        claim=claim_5f,
        node=leaf_5f,
        sources=sources,
        additional_instruction="Verify that the program has aired for many years/decades and is commonly described as 'long-running'."
    )

    # 5G: Supporting references exist
    evaluator.add_custom_node(
        result=has_sources(sources),
        id="5G_Supporting_reference_URLs_cover_key_claims",
        desc="Provides reference URL(s) that substantively support Item 5’s key claims (CBS, newsmagazine format, 60-minute runtime, content features, and long-running status).",
        parent=node,
        critical=True
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
    Evaluate an answer for the entertainment references task and return a structured result.
    Note: The rubric's top-level node was marked as critical in the JSON, but the framework enforces that a critical parent
    cannot have non-critical children. Therefore, we keep the root as non-critical and create a non-critical container node.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level items evaluated independently
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_entertainment(),
        template_class=EntertainmentExtraction,
        extraction_name="entertainment_extraction"
    )

    # Optional: record constraints as ground truth context
    evaluator.add_ground_truth({
        "constraints_summary": {
            "item1": "Youngest-by-birth-year lead actor in an HBO fantasy/drama series premiering in 2026; birth year must be 2014.",
            "item2": "Parent-child duo playing same character at different ages in Season 2 of an NBC series; S2 premiered Jan 2026.",
            "item3": "2016 film with exactly 14 Oscar nominations (ties 'All About Eve' and 'Titanic') and exactly 6 wins.",
            "item4": "Netflix original; Season 8 is final; last episode released on Dec 31, 2025.",
            "item5": "Long-running CBS newsmagazine; standard runtime 60 minutes; features investigative reports, newsmaker interviews, in-depth profiles."
        }
    }, gt_type="rubric_constraints")

    # Container node mirroring rubric root (kept non-critical due to framework constraint)
    collection_node = evaluator.add_parallel(
        id="Entertainment_Reference_Collection",
        desc="Provide five entertainment references matching the question’s criteria, each with supporting reference URL(s).",
        parent=root,
        critical=False
    )

    # Verify each item
    await verify_item_1(evaluator, collection_node, extraction.item1)
    await verify_item_2(evaluator, collection_node, extraction.item2)
    await verify_item_3(evaluator, collection_node, extraction.item3)
    await verify_item_4(evaluator, collection_node, extraction.item4)
    await verify_item_5(evaluator, collection_node, extraction.item5)

    return evaluator.get_summary()