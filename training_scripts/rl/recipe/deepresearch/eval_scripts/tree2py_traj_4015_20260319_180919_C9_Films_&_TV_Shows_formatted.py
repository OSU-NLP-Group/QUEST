import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_2025_entertainment_evaluation"
TASK_DESCRIPTION = """
During the 2025 holiday season (late November through December), several notable entertainment programs and films were released or broadcast. Find the following five entertainment items, providing complete details for each:

Item 1: A Hallmark Channel original Christmas movie that premiered during the weekend of November 15-16, 2025. This movie must feature three male actors playing brothers who previously starred together in two earlier films in the same Hallmark Christmas trilogy (2023 and 2024). The movie must also feature the actress who plays the mother character returning from the previous films. Provide the movie title, exact premiere date, the names of all three male lead actors, the name of the actress playing the mother, and a reference URL confirming these details.

Item 2: A Hallmark Channel original Christmas movie that premiered on November 22, 2025, featuring a storyline centered around a specific NFL team. This movie must include cameo appearances from at least three real members or legends of that NFL team (such as current players, coaches, or former players). Provide the movie title, the NFL team featured, the female and male lead actors' names, the names and roles of at least three people from the actual team who made cameo appearances, and a reference URL confirming these details.

Item 3: The annual televised dog show competition that aired on NBC on Thanksgiving Day (November 27, 2025). Provide the complete official name of the show, confirm the broadcast time (12:00 to 2:00 p.m. local time), and identify the Best in Show winner including: the dog's name, breed, which group the winner came from (Herding, Hound, Sporting, Terrier, Toy, Non-sporting, or Working), the handler's full name, and the handler's home location (city and state). Provide a reference URL confirming these details.

Item 4: The Saturday Night Live Season 51 cast member who reached their 10th season milestone with the show during Season 51 (which began in October 2025). Identify this cast member's full name, confirm they are a Repertory Player (not a Featured Player), identify when they first joined as a cast member (the year), mention any previous role they had at SNL before becoming a cast member (if applicable), and provide at least two notable recurring characters or viral sketches they are known for. Provide a reference URL from NBC or a reliable entertainment source confirming these details.

Item 5: A theatrical film that was released in the United States in December 2025, directed by an Oscar and Emmy Award-winning filmmaker. The film must be a political comedy-drama featuring a female politician as the protagonist, with the character's name appearing in the film's title. Provide the film title, exact U.S. theatrical release date, the director's name, the actress who plays the title character, at least three other notable supporting cast members, and a reference URL from an official studio source or IMDb confirming these details.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class Item1Extraction(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    hallmark_channel_original: Optional[bool] = None
    brother_actors: List[str] = Field(default_factory=list)
    mother_actress: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CameoPerson(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None  # player / coach / legend / staff


class Item2Extraction(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    hallmark_channel_original: Optional[bool] = None
    nfl_team: Optional[str] = None
    female_lead: Optional[str] = None
    male_lead: Optional[str] = None
    team_cameos: List[CameoPerson] = Field(default_factory=list)  # at least three required
    reference_urls: List[str] = Field(default_factory=list)


class Item3Extraction(BaseModel):
    official_show_name: Optional[str] = None
    network: Optional[str] = None
    air_date: Optional[str] = None
    broadcast_time_window: Optional[str] = None  # e.g., "12:00 to 2:00 p.m. local time"
    best_in_show_dog_name: Optional[str] = None
    best_in_show_breed: Optional[str] = None
    best_in_show_group: Optional[str] = None
    handler_full_name: Optional[str] = None
    handler_home_location: Optional[str] = None  # city, state
    reference_urls: List[str] = Field(default_factory=list)


class Item4Extraction(BaseModel):
    cast_member_full_name: Optional[str] = None
    tenth_season_milestone_in_season_51: Optional[bool] = None
    repertory_player_status: Optional[bool] = None
    first_joined_year: Optional[str] = None
    previous_snl_role: Optional[str] = None  # if omitted or None, omission acceptable
    notable_works: List[str] = Field(default_factory=list)  # at least two
    reference_urls: List[str] = Field(default_factory=list)


class Item5Extraction(BaseModel):
    film_title: Optional[str] = None
    us_theatrical_release_date: Optional[str] = None
    director_name: Optional[str] = None
    director_is_oscar_and_emmy_winner: Optional[bool] = None
    genre_description: Optional[str] = None  # should imply political comedy-drama
    female_politician_protagonist: Optional[bool] = None
    protagonist_name_in_title: Optional[bool] = None
    title_character_actress: Optional[str] = None
    supporting_cast: List[str] = Field(default_factory=list)  # at least three
    reference_urls: List[str] = Field(default_factory=list)


class HolidayEntertainmentExtraction(BaseModel):
    item1: Optional[Item1Extraction] = None
    item2: Optional[Item2Extraction] = None
    item3: Optional[Item3Extraction] = None
    item4: Optional[Item4Extraction] = None
    item5: Optional[Item5Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_item1() -> str:
    return """
    Extract Item 1 details from the answer text.
    Required fields:
    - title: exact movie title.
    - premiere_date: the exact premiere date (e.g., "November 15, 2025" or "Nov 16, 2025").
    - hallmark_channel_original: true/false indicating whether it is a Hallmark Channel original Christmas movie.
    - brother_actors: list of the three male lead actors who play brothers (3 names).
    - mother_actress: the actress who plays the mother and returns from prior films.
    - reference_urls: all URLs explicitly provided that substantiate this item (include Hallmark, press releases, TV listings, IMDb pages, etc.).
    Rules:
    - Only extract what appears in the answer text literally.
    - If a field is not present, set it to null or an empty list.
    """


def prompt_extract_item2() -> str:
    return """
    Extract Item 2 details from the answer text.
    Required fields:
    - title
    - premiere_date
    - hallmark_channel_original: true/false
    - nfl_team: the specific NFL team featured.
    - female_lead: name of the female lead actor.
    - male_lead: name of the male lead actor.
    - team_cameos: array of cameo people with fields {name, role} where role is e.g., player/coach/legend; include at least three if present.
    - reference_urls: all URLs explicitly provided that substantiate this item.
    Rules:
    - Only extract what appears in the answer text literally.
    - If a field is not present, set it to null or an empty list.
    """


def prompt_extract_item3() -> str:
    return """
    Extract Item 3 details from the answer text.
    Required fields:
    - official_show_name
    - network
    - air_date
    - broadcast_time_window: should reflect "12:00 to 2:00 p.m. local time" if present.
    - best_in_show_dog_name
    - best_in_show_breed
    - best_in_show_group
    - handler_full_name
    - handler_home_location
    - reference_urls: all URLs explicitly provided that substantiate this item.
    Rules:
    - Only extract what appears in the answer text literally.
    - If a field is not present, set it to null or an empty list.
    """


def prompt_extract_item4() -> str:
    return """
    Extract Item 4 details from the answer text.
    Required fields:
    - cast_member_full_name
    - tenth_season_milestone_in_season_51: true/false
    - repertory_player_status: true/false for Repertory Player (not Featured) in Season 51.
    - first_joined_year
    - previous_snl_role: prior role at SNL before becoming a cast member, if explicitly mentioned; else null.
    - notable_works: array with at least two notable recurring characters or viral sketches if present.
    - reference_urls: include NBC or other reliable entertainment source URLs explicitly provided.
    Rules:
    - Only extract what appears in the answer text literally.
    - If a field is not present, set it to null or an empty list.
    """


def prompt_extract_item5() -> str:
    return """
    Extract Item 5 details from the answer text.
    Required fields:
    - film_title
    - us_theatrical_release_date
    - director_name
    - director_is_oscar_and_emmy_winner: true/false if the director has won both Oscar and Emmy awards.
    - genre_description: text that should imply political comedy-drama.
    - female_politician_protagonist: true/false
    - protagonist_name_in_title: true/false
    - title_character_actress
    - supporting_cast: list of at least three notable supporting cast members if present.
    - reference_urls: official studio source or IMDb URLs explicitly provided.
    Rules:
    - Only extract what appears in the answer text literally.
    - If a field is not present, set it to null or an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_n_nonempty(items: List[Optional[str]], n: int) -> List[str]:
    return [x for x in items if x and x.strip()][:n]


def ensure_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification builders per item                                              #
# --------------------------------------------------------------------------- #
async def build_item1_verification(evaluator: Evaluator, parent, data: Optional[Item1Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_1_Hallmark_Trilogy_Movie",
        desc="Item 1: Hallmark Channel original Christmas movie premiering the weekend of Nov 15–16, 2025, with three brothers actors returning and returning mother actress.",
        parent=parent,
        critical=False
    )

    title = data.title if data else ""
    premiere_date = data.premiere_date if data else ""
    brothers = data.brother_actors if (data and data.brother_actors) else []
    mother = data.mother_actress if data else ""
    urls = ensure_urls(data.reference_urls if data else [])

    # Movie_Title
    n = evaluator.add_leaf(id="Item1_Movie_Title", desc="Provide the movie title.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The referenced page is about a movie with the title '{title}'.",
        node=n,
        sources=urls,
        additional_instruction="Allow minor punctuation or capitalization variations when comparing the title."
    )

    # Premiere_Date (Nov 15 or Nov 16, 2025)
    n = evaluator.add_leaf(id="Item1_Premiere_Date", desc="Provide the exact premiere date, and it must fall on Nov 15 or Nov 16, 2025.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page states the exact premiere date as '{premiere_date}', and that date is either November 15, 2025 or November 16, 2025.",
        node=n,
        sources=urls,
        additional_instruction="Explicitly check that the stated date is one of Nov 15, 2025 or Nov 16, 2025."
    )

    # Hallmark_Channel_Original
    n = evaluator.add_leaf(id="Item1_Hallmark_Channel_Original", desc="Confirm the movie is a Hallmark Channel original Christmas movie.", parent=node, critical=True)
    await evaluator.verify(
        claim="The page confirms this is a Hallmark Channel original Christmas movie.",
        node=n,
        sources=urls
    )

    # Three_Brothers_Actors
    n = evaluator.add_leaf(id="Item1_Three_Brothers_Actors", desc="Provide the names of all three male lead actors who play brothers.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page lists three male lead actors who play brothers in this film: {', '.join(brothers)}.",
        node=n,
        sources=urls,
        additional_instruction="Confirm there are three distinct male actors and they are explicitly described as brothers in the film."
    )

    # Trilogy_Continuity
    n = evaluator.add_leaf(id="Item1_Trilogy_Continuity", desc="Confirm these three actors previously starred together in the two earlier films in the same Hallmark Christmas trilogy (the 2023 and 2024 films).", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page indicates that these three actors ({', '.join(brothers)}) previously starred together as brothers in the earlier two films of the same Hallmark Christmas trilogy released in 2023 and 2024.",
        node=n,
        sources=urls,
        additional_instruction="The support can be via mentions of 'trilogy', references to the 2023 and 2024 entries, or cast continuity notes."
    )

    # Mother_Actress_Returning
    n = evaluator.add_leaf(id="Item1_Mother_Actress_Returning", desc="Provide the name of the actress who plays the mother character and confirm she is returning from the previous films.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page confirms that {mother} plays the mother character and that she is returning from the previous films.",
        node=n,
        sources=urls
    )

    # Reference_URL overall support
    n = evaluator.add_leaf(id="Item1_Reference_URL", desc="Provide a reference URL that supports the title, premiere date, cast (three brothers + mother), and trilogy/returning-cast claims.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The reference page(s) substantiate the title '{title}', the premiere date '{premiere_date}', the three brother actors ({', '.join(brothers)}), the returning mother actress ({mother}), and the trilogy continuity details.",
        node=n,
        sources=urls,
        additional_instruction="If multiple URLs are provided, any combination that clearly supports all of these points is acceptable."
    )


async def build_item2_verification(evaluator: Evaluator, parent, data: Optional[Item2Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_2_Hallmark_NFL_Team_Movie",
        desc="Item 2: Hallmark original Christmas movie on Nov 22, 2025, centered on a specific NFL team, with ≥3 real team members/legends cameoing.",
        parent=parent,
        critical=False
    )

    title = data.title if data else ""
    premiere_date = data.premiere_date if data else ""
    team = data.nfl_team if data else ""
    female_lead = data.female_lead if data else ""
    male_lead = data.male_lead if data else ""
    cameos = data.team_cameos if (data and data.team_cameos) else []
    urls = ensure_urls(data.reference_urls if data else [])

    cameo_triplet = [c for c in cameos if c and c.name]
    cameo_triplet = cameo_triplet[:3]

    cameo_desc = "; ".join([f"{c.name} ({c.role})" if c.role else f"{c.name}" for c in cameo_triplet])

    # Movie_Title
    n = evaluator.add_leaf(id="Item2_Movie_Title", desc="Provide the movie title.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The referenced page is about a movie with the title '{title}'.",
        node=n,
        sources=urls,
        additional_instruction="Allow minor punctuation or capitalization variations."
    )

    # Premiere_Date exact Nov 22, 2025
    n = evaluator.add_leaf(id="Item2_Premiere_Date", desc="Confirm the premiere date is Nov 22, 2025.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page states the premiere date as '{premiere_date}', which is exactly November 22, 2025.",
        node=n,
        sources=urls
    )

    # Hallmark_Channel_Original
    n = evaluator.add_leaf(id="Item2_Hallmark_Channel_Original", desc="Confirm the movie is a Hallmark Channel original Christmas movie.", parent=node, critical=True)
    await evaluator.verify(
        claim="The page confirms this is a Hallmark Channel original Christmas movie.",
        node=n,
        sources=urls
    )

    # NFL_Team_Featured
    n = evaluator.add_leaf(id="Item2_NFL_Team_Featured", desc="Identify the NFL team the storyline is centered around.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page indicates the storyline is centered on the NFL team '{team}'.",
        node=n,
        sources=urls
    )

    # Lead_Actors
    n = evaluator.add_leaf(id="Item2_Lead_Actors", desc="Provide the female and male lead actors’ names.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page lists the leads as female lead '{female_lead}' and male lead '{male_lead}'.",
        node=n,
        sources=urls
    )

    # Team_Cameos_At_Least_Three
    n = evaluator.add_leaf(id="Item2_Team_Cameos_At_Least_Three", desc="Provide names and roles for at least three real members/legends of that NFL team who cameo.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page confirms that at least three real {team} members or legends make cameo appearances: {cameo_desc}.",
        node=n,
        sources=urls,
        additional_instruction="Confirm those named are real persons associated with the specified NFL team and that the page mentions their cameos."
    )

    # Reference_URL overall support
    n = evaluator.add_leaf(id="Item2_Reference_URL", desc="Provide a reference URL that supports the premiere date, featured team, leads, and the cameo appearances.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The reference page(s) substantiate the Nov 22, 2025 premiere date, the featured NFL team '{team}', the leads ({female_lead} and {male_lead}), and at least three cameos: {cameo_desc}.",
        node=n,
        sources=urls
    )


async def build_item3_verification(evaluator: Evaluator, parent, data: Optional[Item3Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_3_NBC_Thanksgiving_Dog_Show",
        desc="Item 3: NBC Thanksgiving Day dog show (Nov 27, 2025) with broadcast window and Best in Show details.",
        parent=parent,
        critical=False
    )

    show_name = data.official_show_name if data else ""
    network = data.network if data else ""
    air_date = data.air_date if data else ""
    time_win = data.broadcast_time_window if data else ""
    dog = data.best_in_show_dog_name if data else ""
    breed = data.best_in_show_breed if data else ""
    group = data.best_in_show_group if data else ""
    handler = data.handler_full_name if data else ""
    location = data.handler_home_location if data else ""
    urls = ensure_urls(data.reference_urls if data else [])

    # Official_Show_Name
    n = evaluator.add_leaf(id="Item3_Official_Show_Name", desc="Provide the complete official name of the dog show.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page gives the complete official name of the show as '{show_name}'.",
        node=n,
        sources=urls
    )

    # Network_And_Date (NBC on Nov 27, 2025)
    n = evaluator.add_leaf(id="Item3_Network_And_Date", desc="Confirm it aired on NBC on Thanksgiving Day, Nov 27, 2025.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page confirms the show aired on NBC on November 27, 2025 (Thanksgiving Day).",
        node=n,
        sources=urls
    )

    # Broadcast_Time_Window
    n = evaluator.add_leaf(id="Item3_Broadcast_Time_Window", desc="Confirm the broadcast time was 12:00 to 2:00 p.m. local time.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page confirms the broadcast time window was 12:00 to 2:00 p.m. local time (often described as '12 noon to 2 p.m.' in each time zone).",
        node=n,
        sources=urls,
        additional_instruction="If phrased per time zone, consider that equivalent to local time."
    )

    # Best_in_Show_Dog_Name
    n = evaluator.add_leaf(id="Item3_BIS_Dog_Name", desc="Provide the Best in Show winner dog’s name.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page identifies the Best in Show winner as a dog named '{dog}'.",
        node=n,
        sources=urls
    )

    # Best_in_Show_Breed
    n = evaluator.add_leaf(id="Item3_BIS_Breed", desc="Provide the Best in Show winner dog’s breed.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The Best in Show winner's breed was '{breed}'.",
        node=n,
        sources=urls
    )

    # Best_in_Show_Group
    n = evaluator.add_leaf(id="Item3_BIS_Group", desc="Provide which group the Best in Show winner came from (Herding, Hound, Sporting, Terrier, Toy, Non-sporting, or Working).", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The Best in Show winner came from the '{group}' Group.",
        node=n,
        sources=urls,
        additional_instruction="Ensure the group is one of: Herding, Hound, Sporting, Terrier, Toy, Non-sporting, Working."
    )

    # Handler_Name
    n = evaluator.add_leaf(id="Item3_Handler_Name", desc="Provide the handler’s full name.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The handler's full name was '{handler}'.",
        node=n,
        sources=urls
    )

    # Handler_Home_Location
    n = evaluator.add_leaf(id="Item3_Handler_Location", desc="Provide the handler’s home location (city and state).", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The handler's home location (city and state) was '{location}'.",
        node=n,
        sources=urls
    )

    # Reference_URL
    n = evaluator.add_leaf(id="Item3_Reference_URL", desc="Provide a reference URL supporting the show name, network/date/time, and the Best in Show winner details.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The reference page(s) substantiate the official show name '{show_name}', the NBC air date '{air_date}' with the 12:00–2:00 p.m. local time window, and the Best in Show details (dog '{dog}', breed '{breed}', group '{group}', handler '{handler}', location '{location}').",
        node=n,
        sources=urls
    )


async def build_item4_verification(evaluator: Evaluator, parent, data: Optional[Item4Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_4_SNL_Season51_10th_Season_Cast",
        desc="Item 4: SNL Season 51 cast member who reached 10th season; verify status, join-year, prior role (if any), and notable works.",
        parent=parent,
        critical=False
    )

    name = data.cast_member_full_name if data else ""
    tenth = data.tenth_season_milestone_in_season_51 if data else None
    repertory = data.repertory_player_status if data else None
    year = data.first_joined_year if data else ""
    prev_role = data.previous_snl_role if data else None
    works = data.notable_works if (data and data.notable_works) else []
    urls = ensure_urls(data.reference_urls if data else [])

    # Cast_Member_Full_Name
    n = evaluator.add_leaf(id="Item4_Cast_Member_Full_Name", desc="Provide the cast member’s full name.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page identifies a Saturday Night Live Season 51 cast member named '{name}'.",
        node=n,
        sources=urls
    )

    # Tenth_Season_Milestone_In_Season_51
    n = evaluator.add_leaf(id="Item4_Tenth_Season_Milestone_In_Season_51", desc="Confirm the identified cast member reached their 10th season milestone as a cast member during SNL Season 51.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page confirms that {name} reached their 10th season as a cast member during Season 51.",
        node=n,
        sources=urls
    )

    # Repertory_Player_Status
    n = evaluator.add_leaf(id="Item4_Repertory_Player_Status", desc="Confirm the person is a Repertory Player (not a Featured Player) in Season 51.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page indicates that {name} is a Repertory Player in Season 51 (not a Featured Player).",
        node=n,
        sources=urls
    )

    # First_Joined_Year
    n = evaluator.add_leaf(id="Item4_First_Joined_Year", desc="Provide the year they first joined as a cast member.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page confirms that {name} first joined SNL as a cast member in {year}.",
        node=n,
        sources=urls
    )

    # Previous_SNL_Role_If_Any
    if prev_role and prev_role.strip():
        n = evaluator.add_leaf(id="Item4_Previous_SNL_Role_If_Any", desc="If the cast member held a previous role at SNL before becoming a cast member, state what it was and confirm it is accurate.", parent=node, critical=True)
        await evaluator.verify(
            claim=f"Before becoming a cast member, {name} previously held the role '{prev_role}' at SNL.",
            node=n,
            sources=urls
        )
    else:
        # Omission acceptable per rubric; pass via custom node
        evaluator.add_custom_node(
            result=True,
            id="Item4_Previous_SNL_Role_If_Any",
            desc="No prior SNL role explicitly claimed; omission acceptable per rubric.",
            parent=node,
            critical=True
        )

    # At_Least_Two_Notable_Works
    display_works = first_n_nonempty(works, 2)
    n = evaluator.add_leaf(id="Item4_At_Least_Two_Notable_Works", desc="Provide at least two notable recurring characters or viral sketches the cast member is known for.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page identifies at least two notable recurring characters or viral sketches for {name}: {', '.join(display_works)}.",
        node=n,
        sources=urls,
        additional_instruction="Confirm these are indeed recurring characters or widely recognized sketches associated with this cast member."
    )

    # Reference_URL
    n = evaluator.add_leaf(id="Item4_Reference_URL", desc="Provide a reference URL from NBC or a reliable entertainment source supporting the milestone/status/join-year/prior-role (if any) and notable works.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The reference page(s) substantiate that {name} hit their 10th season during Season 51, is a Repertory Player, first joined in {year}, prior role if any ('{prev_role}' if provided), and notable works ({', '.join(display_works)}).",
        node=n,
        sources=urls,
        additional_instruction="Accept NBC.com, official press, or reliable trade/entertainment sources."
    )


async def build_item5_verification(evaluator: Evaluator, parent, data: Optional[Item5Extraction]) -> None:
    node = evaluator.add_parallel(
        id="Item_5_December2025_Political_ComedyDrama_Film",
        desc="Item 5: U.S. theatrical December 2025 film; director is Oscar & Emmy winner; political comedy-drama with female politician protagonist named in title; verify cast and sources.",
        parent=parent,
        critical=False
    )

    title = data.film_title if data else ""
    release_date = data.us_theatrical_release_date if data else ""
    director = data.director_name if data else ""
    osc_emmy = data.director_is_oscar_and_emmy_winner if data else None
    genre_text = data.genre_description if data else ""
    female_pol = data.female_politician_protagonist if data else None
    name_in_title = data.protagonist_name_in_title if data else None
    actress = data.title_character_actress if data else ""
    support = data.supporting_cast if (data and data.supporting_cast) else []
    urls = ensure_urls(data.reference_urls if data else [])

    support_three = first_n_nonempty(support, 3)

    # Film_Title
    n = evaluator.add_leaf(id="Item5_Film_Title", desc="Provide the film title.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The referenced page is about a film titled '{title}'.",
        node=n,
        sources=urls
    )

    # US_Theatrical_Release_Date
    n = evaluator.add_leaf(id="Item5_US_Theatrical_Release_Date", desc="Provide the exact U.S. theatrical release date, and it must be in December 2025.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The U.S. theatrical release date was '{release_date}', and that date is in December 2025.",
        node=n,
        sources=urls
    )

    # Director_And_Awards
    n = evaluator.add_leaf(id="Item5_Director_And_Awards", desc="Provide the director’s name and confirm the director is an Oscar and Emmy Award winner.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The film was directed by {director}, who has won both an Academy Award (Oscar) and an Emmy Award.",
        node=n,
        sources=urls,
        additional_instruction="Explicit confirmation of both Oscar and Emmy wins must be present or strongly implied."
    )

    # Genre_Political_ComedyDrama
    n = evaluator.add_leaf(id="Item5_Genre_Political_ComedyDrama", desc="Confirm the film is a political comedy-drama.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page indicates the film is a political comedy-drama. Supporting description: {genre_text}",
        node=n,
        sources=urls
    )

    # Female_Politician_Protagonist
    n = evaluator.add_leaf(id="Item5_Female_Politician_Protagonist", desc="Confirm the protagonist is a female politician.", parent=node, critical=True)
    await evaluator.verify(
        claim="The protagonist of the film is a female politician.",
        node=n,
        sources=urls
    )

    # Protagonist_Name_In_Title
    n = evaluator.add_leaf(id="Item5_Protagonist_Name_In_Title", desc="Confirm the protagonist character’s name appears in the film’s title.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The protagonist character's name appears in the film title '{title}'.",
        node=n,
        sources=urls
    )

    # Title_Character_Actress
    n = evaluator.add_leaf(id="Item5_Title_Character_Actress", desc="Provide the actress who plays the title character.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The title character is played by {actress}.",
        node=n,
        sources=urls
    )

    # Supporting_Cast_At_Least_Three
    n = evaluator.add_leaf(id="Item5_Supporting_Cast_At_Least_Three", desc="Provide at least three other notable supporting cast members’ names.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The page lists at least three notable supporting cast members in addition to the lead: {', '.join(support_three)}.",
        node=n,
        sources=urls,
        additional_instruction="Verify that at least three named people are part of the credited supporting cast."
    )

    # Reference_URL
    n = evaluator.add_leaf(id="Item5_Reference_URL", desc="Provide a reference URL from an official studio source or IMDb supporting the title, release date, director, premise/genre, and cast details.", parent=node, critical=True)
    await evaluator.verify(
        claim=f"The reference page(s) substantiate the film '{title}', the U.S. theatrical date '{release_date}', director '{director}' and their awards status, the political comedy-drama premise, and the cast including {actress} and supporting names like {', '.join(support_three)}.",
        node=n,
        sources=urls,
        additional_instruction="Prefer official studio sources or IMDb; strong trade sources acceptable if clearly authoritative."
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
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel per rubric
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

    # Concurrently extract each item to keep prompts focused and robust
    item1_fut = evaluator.extract(
        prompt=prompt_extract_item1(),
        template_class=Item1Extraction,
        extraction_name="item1_extraction",
    )
    item2_fut = evaluator.extract(
        prompt=prompt_extract_item2(),
        template_class=Item2Extraction,
        extraction_name="item2_extraction",
    )
    item3_fut = evaluator.extract(
        prompt=prompt_extract_item3(),
        template_class=Item3Extraction,
        extraction_name="item3_extraction",
    )
    item4_fut = evaluator.extract(
        prompt=prompt_extract_item4(),
        template_class=Item4Extraction,
        extraction_name="item4_extraction",
    )
    item5_fut = evaluator.extract(
        prompt=prompt_extract_item5(),
        template_class=Item5Extraction,
        extraction_name="item5_extraction",
    )

    item1, item2, item3, item4, item5 = await asyncio.gather(
        item1_fut, item2_fut, item3_fut, item4_fut, item5_fut
    )

    # Build verification subtrees for each item
    await build_item1_verification(evaluator, root, item1)
    await build_item2_verification(evaluator, root, item2)
    await build_item3_verification(evaluator, root, item3)
    await build_item4_verification(evaluator, root, item4)
    await build_item5_verification(evaluator, root, item5)

    # Finalize and return summary
    return evaluator.get_summary()