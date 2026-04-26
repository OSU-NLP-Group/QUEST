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
TASK_ID = "artist_superbowl_2026"
TASK_DESCRIPTION = """
Identify the music artist who achieved both of the following: (1) was Spotify's most-streamed artist globally for three consecutive years (2020, 2021, and 2022), and (2) released the first all-Spanish language album to reach #1 on the Billboard 200 chart. For this artist, provide the following information: What major sporting event halftime show are they scheduled to headline in February 2026, and on what date? What is the name and location (city and state) of the stadium where this performance will take place? Which NFL team uses this stadium as their home venue? In which specific town or city and territory was this artist born? What is the political relationship of that birthplace territory to the United States?
"""

# Ground truth references for validator context (not used as hard checks, but recorded)
GROUND_TRUTH = {
    "expected_artist": "Bad Bunny",
    "spotify_most_streamed_years": ["2020", "2021", "2022"],
    "first_spanish_number1_album": "El Último Tour del Mundo",
    "halftime_event": "Super Bowl LX halftime show",
    "halftime_date": "February 8, 2026",
    "stadium_name": "Levi's Stadium",
    "stadium_city": "Santa Clara",
    "stadium_state": "California",
    "stadium_home_team": "San Francisco 49ers",
    "birthplace_city": "Vega Baja",
    "birthplace_territory": "Puerto Rico",
    "territory_status": "U.S. territory (unincorporated territory)"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArtistCore(BaseModel):
    artist_name: Optional[str] = None
    spotify_years_claimed: List[str] = Field(default_factory=list)
    album_name_first_spanish_no1: Optional[str] = None
    spotify_sources: List[str] = Field(default_factory=list)
    billboard_sources: List[str] = Field(default_factory=list)


class HalftimeShowInfo(BaseModel):
    event_name: Optional[str] = None  # e.g., "Super Bowl LX halftime show"
    date: Optional[str] = None        # e.g., "February 8, 2026"
    sources: List[str] = Field(default_factory=list)


class StadiumInfo(BaseModel):
    name: Optional[str] = None        # e.g., "Levi's Stadium"
    city: Optional[str] = None        # e.g., "Santa Clara"
    state: Optional[str] = None       # e.g., "California"
    sources: List[str] = Field(default_factory=list)


class StadiumHomeInfo(BaseModel):
    team_name: Optional[str] = None   # e.g., "San Francisco 49ers"
    sources: List[str] = Field(default_factory=list)


class BirthplaceInfo(BaseModel):
    town_city: Optional[str] = None   # e.g., "Vega Baja"
    territory: Optional[str] = None   # e.g., "Puerto Rico"
    sources: List[str] = Field(default_factory=list)


class TerritoryStatusInfo(BaseModel):
    relationship_to_US: Optional[str] = None  # e.g., "a U.S. territory (unincorporated territory)"
    sources: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    artist_core: Optional[ArtistCore] = None
    halftime_show: Optional[HalftimeShowInfo] = None
    stadium: Optional[StadiumInfo] = None
    stadium_home: Optional[StadiumHomeInfo] = None
    birthplace: Optional[BirthplaceInfo] = None
    territory_status: Optional[TerritoryStatusInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_artist_core() -> str:
    return """
    Extract the core artist identification and achievement evidence from the answer.

    Return a JSON object with:
    - artist_name: The artist explicitly named in the answer.
    - spotify_years_claimed: An array of years (as strings) the answer claims the artist was Spotify's most-streamed globally.
    - album_name_first_spanish_no1: The album name, if the answer names the album it claims was the first all-Spanish-language album to reach #1 on the Billboard 200; else null.
    - spotify_sources: All URLs cited for the Spotify most-streamed claim.
    - billboard_sources: All URLs cited for the Billboard #1 all-Spanish album claim.

    If any field is not present in the answer, return null or an empty array as appropriate.
    """


def prompt_extract_halftime_show() -> str:
    return """
    Extract the halftime show event and date for February 2026.

    Return:
    - event_name: The named halftime show (e.g., "Super Bowl LX halftime show").
    - date: The stated date (e.g., "February 8, 2026").
    - sources: All URLs cited for this halftime show scheduling information.

    Only extract information explicitly present in the answer. If missing, set fields to null or [].
    """


def prompt_extract_stadium() -> str:
    return """
    Extract the stadium details where the halftime show will occur.

    Return:
    - name: Stadium name (e.g., "Levi's Stadium").
    - city: City (e.g., "Santa Clara").
    - state: State (e.g., "California").
    - sources: All URLs cited for the stadium/location information.

    Only extract data explicitly present in the answer.
    """


def prompt_extract_stadium_home() -> str:
    return """
    Extract the NFL team that uses the referenced stadium as their home venue.

    Return:
    - team_name: e.g., "San Francisco 49ers".
    - sources: All URLs cited for the team/stadium association.

    Only extract information explicitly present in the answer.
    """


def prompt_extract_birthplace() -> str:
    return """
    Extract the artist's birthplace town/city and territory.

    Return:
    - town_city: e.g., "Vega Baja".
    - territory: e.g., "Puerto Rico".
    - sources: All URLs cited for the birthplace information.

    Only extract information explicitly present in the answer.
    """


def prompt_extract_territory_status() -> str:
    return """
    Extract the stated political relationship of the birthplace territory to the United States.

    Return:
    - relationship_to_US: e.g., "a U.S. territory (unincorporated territory)".
    - sources: All URLs cited for this political status information.

    Only extract information explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_identify_artist(evaluator: Evaluator, parent_node, extracted: FullExtraction) -> None:
    """
    Build and verify the 'Identify_Artist' critical parallel node and its leaf checks.
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Artist",
        desc="Correctly identify the artist who meets both achievement constraints.",
        parent=parent_node,
        critical=True
    )

    # Artist name provided (existence check)
    artist_name = (extracted.artist_core.artist_name if extracted.artist_core else None)
    name_provided = bool(artist_name and artist_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Artist_Name_Provided",
        desc="Answer explicitly names the artist.",
        parent=identify_node,
        critical=True
    )

    # Spotify most-streamed globally 2020–2022
    spotify_leaf = evaluator.add_leaf(
        id="Spotify_Most_Streamed_2020_2022",
        desc="Artist was Spotify's most-streamed artist globally for three consecutive years: 2020, 2021, and 2022.",
        parent=identify_node,
        critical=True
    )
    spotify_sources = (extracted.artist_core.spotify_sources if extracted.artist_core else [])
    spotify_claim_artist = artist_name or "the named artist"
    spotify_claim = (
        f"{spotify_claim_artist} was Spotify's most-streamed artist globally for 2020, 2021, and 2022."
    )
    await evaluator.verify(
        claim=spotify_claim,
        node=spotify_leaf,
        sources=(spotify_sources if spotify_sources else None),
        additional_instruction=(
            "Confirm using official Spotify year-end or authoritative reports. "
            "Allow equivalent naming (e.g., stage name vs. full legal name). "
            "The judgment must affirm that the same artist was the global most‑streamed in 2020, 2021, and 2022 consecutively."
        )
    )

    # First all‑Spanish #1 Billboard 200 album
    billboard_leaf = evaluator.add_leaf(
        id="First_All_Spanish_Billboard200_Number1",
        desc="Artist released the first all-Spanish language album to reach #1 on the Billboard 200 chart.",
        parent=identify_node,
        critical=True
    )
    billboard_sources = (extracted.artist_core.billboard_sources if extracted.artist_core else [])
    album_name = (extracted.artist_core.album_name_first_spanish_no1 if extracted.artist_core else None)
    if album_name and album_name.strip():
        billboard_claim = (
            f"'{album_name}' by {spotify_claim_artist} was the first all-Spanish-language album to reach #1 on the Billboard 200 chart."
        )
    else:
        billboard_claim = (
            f"{spotify_claim_artist} released the first all-Spanish-language album to reach #1 on the Billboard 200 chart."
        )
    await evaluator.verify(
        claim=billboard_claim,
        node=billboard_leaf,
        sources=(billboard_sources if billboard_sources else None),
        additional_instruction=(
            "Use authoritative sources (Billboard, major press) to confirm the 'first all‑Spanish' #1 album. "
            "Minor naming variations are acceptable."
        )
    )


async def verify_requested_details(evaluator: Evaluator, parent_node, extracted: FullExtraction) -> None:
    """
    Build and verify the 'Provide_Requested_Details' critical parallel node and its leaf checks.
    """
    details_node = evaluator.add_parallel(
        id="Provide_Requested_Details",
        desc="Provide all requested details about the artist’s scheduled halftime show, venue, and birthplace/political status.",
        parent=parent_node,
        critical=True
    )

    # Halftime show event and date
    halftime_leaf = evaluator.add_leaf(
        id="Halftime_Show_Event_And_Date",
        desc="Identify the halftime show as the Super Bowl LX halftime show and give the date as February 8, 2026.",
        parent=details_node,
        critical=True
    )
    artist_name = (extracted.artist_core.artist_name if extracted.artist_core else "the named artist")
    halftime_sources = (extracted.halftime_show.sources if extracted.halftime_show else [])
    event_name = (extracted.halftime_show.event_name if extracted.halftime_show else None) or "Super Bowl LX halftime show"
    date_str = (extracted.halftime_show.date if extracted.halftime_show else None) or "February 8, 2026"
    halftime_claim = (
        f"{artist_name} is scheduled to headline the {event_name} on {date_str}."
    )
    await evaluator.verify(
        claim=halftime_claim,
        node=halftime_leaf,
        sources=(halftime_sources if halftime_sources else None),
        additional_instruction=(
            "Verify the scheduling announcement using authoritative sources (NFL, Apple Music Halftime Show announcements, major outlets). "
            "The event should be Super Bowl LX and the date February 8, 2026."
        )
    )

    # Stadium name and location
    stadium_leaf = evaluator.add_leaf(
        id="Stadium_Name_And_Location",
        desc="Name and location (city and state) of the stadium: Levi's Stadium in Santa Clara, California.",
        parent=details_node,
        critical=True
    )
    stadium_sources = (extracted.stadium.sources if extracted.stadium else [])
    stadium_name = (extracted.stadium.name if extracted.stadium else None) or "Levi's Stadium"
    stadium_city = (extracted.stadium.city if extracted.stadium else None) or "Santa Clara"
    stadium_state = (extracted.stadium.state if extracted.stadium else None) or "California"
    stadium_claim = (
        f"The performance will take place at {stadium_name} in {stadium_city}, {stadium_state}."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_leaf,
        sources=(stadium_sources if stadium_sources else None),
        additional_instruction=(
            "Confirm that Super Bowl LX is hosted at Levi's Stadium and that it is located in Santa Clara, California."
        )
    )

    # Stadium home NFL team
    home_team_leaf = evaluator.add_leaf(
        id="Stadium_Home_NFL_Team",
        desc="Identify the NFL home team for this stadium: the San Francisco 49ers.",
        parent=details_node,
        critical=True
    )
    home_sources = (extracted.stadium_home.sources if extracted.stadium_home else [])
    # If no direct home team sources, reuse stadium sources
    if not home_sources and stadium_sources:
        home_sources = stadium_sources
    team_name = (extracted.stadium_home.team_name if extracted.stadium_home else None) or "San Francisco 49ers"
    home_claim = (
        f"{stadium_name} is the home stadium of the {team_name}."
    )
    await evaluator.verify(
        claim=home_claim,
        node=home_team_leaf,
        sources=(home_sources if home_sources else None),
        additional_instruction=(
            "Confirm from authoritative sources (team/NFL/stadium pages) that Levi's Stadium is the home venue of the San Francisco 49ers."
        )
    )

    # Birthplace town and territory
    birthplace_leaf = evaluator.add_leaf(
        id="Birthplace_Town_And_Territory",
        desc="Provide the artist’s birthplace town/city and territory: Vega Baja, Puerto Rico.",
        parent=details_node,
        critical=True
    )
    birthplace_sources = (extracted.birthplace.sources if extracted.birthplace else [])
    town_city = (extracted.birthplace.town_city if extracted.birthplace else None) or "Vega Baja"
    territory = (extracted.birthplace.territory if extracted.birthplace else None) or "Puerto Rico"
    birthplace_claim = (
        f"{artist_name} was born in {town_city}, {territory}."
    )
    await evaluator.verify(
        claim=birthplace_claim,
        node=birthplace_leaf,
        sources=(birthplace_sources if birthplace_sources else None),
        additional_instruction=(
            "Allow more specific barrio/locality within the municipality (e.g., Almirante Sur in Vega Baja) to count as consistent with Vega Baja, Puerto Rico."
        )
    )

    # Birthplace territory political relationship
    territory_leaf = evaluator.add_leaf(
        id="Birthplace_Territory_Political_Relationship",
        desc="State the political relationship of Puerto Rico to the United States: a U.S. territory (specifically an unincorporated territory).",
        parent=details_node,
        critical=True
    )
    territory_sources = (extracted.territory_status.sources if extracted.territory_status else [])
    relationship_text = (extracted.territory_status.relationship_to_US if extracted.territory_status else None) or \
                        "a U.S. territory (specifically an unincorporated territory)"
    territory_claim = (
        f"Puerto Rico is {relationship_text} of the United States."
    )
    await evaluator.verify(
        claim=territory_claim,
        node=territory_leaf,
        sources=(territory_sources if territory_sources else None),
        additional_instruction=(
            "Accept equivalent formulations indicating Puerto Rico is a U.S. unincorporated territory/commonwealth."
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
    Evaluate an agent's answer for the 2026 Super Bowl halftime artist and related details task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Add ground truth info for transparency
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="expected_values")

    # Run extractions (can be done concurrently)
    artist_core_task = evaluator.extract(
        prompt=prompt_extract_artist_core(),
        template_class=ArtistCore,
        extraction_name="artist_core"
    )
    halftime_task = evaluator.extract(
        prompt=prompt_extract_halftime_show(),
        template_class=HalftimeShowInfo,
        extraction_name="halftime_show"
    )
    stadium_task = evaluator.extract(
        prompt=prompt_extract_stadium(),
        template_class=StadiumInfo,
        extraction_name="stadium_info"
    )
    stadium_home_task = evaluator.extract(
        prompt=prompt_extract_stadium_home(),
        template_class=StadiumHomeInfo,
        extraction_name="stadium_home_team"
    )
    birthplace_task = evaluator.extract(
        prompt=prompt_extract_birthplace(),
        template_class=BirthplaceInfo,
        extraction_name="birthplace"
    )
    territory_status_task = evaluator.extract(
        prompt=prompt_extract_territory_status(),
        template_class=TerritoryStatusInfo,
        extraction_name="territory_status"
    )

    artist_core, halftime_show, stadium, stadium_home, birthplace, territory_status = await asyncio.gather(
        artist_core_task, halftime_task, stadium_task, stadium_home_task, birthplace_task, territory_status_task
    )

    # Combine into one model for convenience (also recorded in summary via individual extractions above)
    extracted = FullExtraction(
        artist_core=artist_core,
        halftime_show=halftime_show,
        stadium=stadium,
        stadium_home=stadium_home,
        birthplace=birthplace,
        territory_status=territory_status
    )

    # Build "Complete_Task" critical sequential node (as per rubric)
    complete_task_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify the correct artist satisfying both achievement criteria and provide all requested performance, venue, and origin details.",
        parent=root,
        critical=True
    )

    # 1) Identify artist & achievements (critical parallel)
    await verify_identify_artist(evaluator, complete_task_node, extracted)

    # 2) Provide requested details (critical parallel)
    await verify_requested_details(evaluator, complete_task_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()