import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "artist_identification_country_crossover_2026"
TASK_DESCRIPTION = (
    "Identify the country music artist who meets ALL of the specified criteria as of February 20, 2026, and provide the requested information with supporting URL references:\n\n"
    "Biographical Requirements:\n"
    "- Born on December 4, 1984\n"
    "- From the Antioch neighborhood of Nashville, Tennessee\n"
    "- Legal name is Jason Bradley DeFord\n"
    "- Had a history of incarceration with felony convictions\n"
    "- Received a pardon from Tennessee Governor Bill Lee on December 18, 2025\n\n"
    "Career Timeline Requirements:\n"
    "- Began music career in 2003 with mixtapes\n"
    "- Started in the Southern hip-hop/rap genre\n"
    "- Transitioned to country music crossover\n"
    "- Made Grand Ole Opry debut in November 2021\n\n"
    "Chart Performance Requirements:\n"
    "- Achieved first country radio #1 with \"Son of a Sinner\" in January 2023\n"
    "- \"Son of a Sinner\" was certified 2× Platinum (2,000,000 units) by the RIAA\n"
    "- \"Need a Favor\" topped both the Mainstream Rock Airplay chart and the Country Airplay chart\n"
    "- Sold out Nashville's Bridgestone Arena on December 9, 2022\n\n"
    "Awards and Recognition Requirements:\n"
    "- Nominated for Best New Artist at the 66th Annual Grammy Awards (2024)\n"
    "- \"Save Me\" with Lainey Wilson was nominated for Best Country Duo/Group Performance at the 2024 Grammys\n"
    "- Won CMA New Artist of the Year at the 57th Annual CMA Awards (2023)\n"
    "- Won three Grammy awards at the 67th Annual Grammy Awards (2026)\n\n"
    "Discography Requirements:\n"
    "- Released an album titled \"Ballads of the Broken\" on September 17, 2021, containing 10 tracks\n"
    "- Released an album titled \"Whitsitt Chapel\" on June 2, 2023, with \"Need a Favor\" as the lead single\n"
    "- Released an album titled \"Beautifully Broken\" on October 11, 2024, containing 22 tracks\n\n"
    "Personal Life Requirements:\n"
    "- Married to Alisa DeFord (known as Bunnie XO) since 2016\n"
    "- Spouse hosts the \"Dumb Blonde\" podcast\n"
    "- Performed as musical guest on Saturday Night Live Season 50 in September 2024\n\n"
    "Required Response Format:\n"
    "Provide the artist's stage name and, for each requirement category listed above (Biographical, Career Timeline, Chart Performance, Awards and Recognition, Discography, and Personal Life), provide verification including specific facts and at least one supporting URL reference for each major requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArtistEvidenceExtraction(BaseModel):
    stage_name: Optional[str] = None

    # Biographical URLs
    birth_date_urls: List[str] = Field(default_factory=list)
    origin_antioch_urls: List[str] = Field(default_factory=list)
    legal_name_urls: List[str] = Field(default_factory=list)
    incarceration_felony_urls: List[str] = Field(default_factory=list)
    pardon_2025_urls: List[str] = Field(default_factory=list)

    # Career Timeline URLs
    career_2003_mixtapes_urls: List[str] = Field(default_factory=list)
    started_southern_hiphop_urls: List[str] = Field(default_factory=list)
    transitioned_country_crossover_urls: List[str] = Field(default_factory=list)
    opry_debut_nov_2021_urls: List[str] = Field(default_factory=list)

    # Chart Performance URLs
    first_country_radio_no1_urls: List[str] = Field(default_factory=list)
    riaa_son_of_a_sinner_2x_urls: List[str] = Field(default_factory=list)
    need_a_favor_dual_charts_urls: List[str] = Field(default_factory=list)
    bridgestone_sellout_2022_urls: List[str] = Field(default_factory=list)

    # Awards and Recognition URLs
    grammy_2024_bna_urls: List[str] = Field(default_factory=list)
    grammy_2024_save_me_duo_urls: List[str] = Field(default_factory=list)
    cma_2023_new_artist_win_urls: List[str] = Field(default_factory=list)
    grammy_2026_three_wins_urls: List[str] = Field(default_factory=list)

    # Discography URLs
    ballads_of_the_broken_urls: List[str] = Field(default_factory=list)
    whitsitt_chapel_release_urls: List[str] = Field(default_factory=list)
    whitsitt_chapel_lead_single_urls: List[str] = Field(default_factory=list)
    beautifully_broken_urls: List[str] = Field(default_factory=list)

    # Personal Life URLs
    marriage_since_2016_urls: List[str] = Field(default_factory=list)
    spouse_dumb_blonde_urls: List[str] = Field(default_factory=list)
    snl_season50_sept_2024_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_artist_evidence() -> str:
    return """
Extract the artist’s stage name and all supporting URL references the answer explicitly associates with each requirement. STRICTLY follow these rules:
- Only extract URLs that are explicitly present in the answer text (including markdown links). Do not invent any URLs.
- If a single URL is used to support multiple requirements, include it in each relevant list.
- If the answer provides a general “Sources/References” section, assign URLs to the appropriate requirement(s) based on the nearby context or labels in the answer. If ambiguous, duplicate a URL into all clearly relevant requirement lists; otherwise leave it out.
- Return null for stage_name if it is not provided.
- Return an empty array for any URL list if the answer provides no URLs for that requirement.

Return a JSON object with the following fields:
- stage_name

Biographical (URLs for each):
- birth_date_urls                                  # URLs supporting the birth date (Dec 4, 1984)
- origin_antioch_urls                              # URLs supporting origin as Antioch (Nashville, TN)
- legal_name_urls                                  # URLs supporting legal name: Jason Bradley DeFord
- incarceration_felony_urls                        # URLs supporting history of incarceration with felony convictions
- pardon_2025_urls                                 # URLs supporting the pardon by Tennessee Governor Bill Lee on Dec 18, 2025

Career Timeline (URLs for each):
- career_2003_mixtapes_urls                        # URLs supporting career began in 2003 with mixtapes
- started_southern_hiphop_urls                     # URLs supporting early genre: Southern hip-hop/rap
- transitioned_country_crossover_urls              # URLs supporting transition to country music crossover
- opry_debut_nov_2021_urls                         # URLs supporting Grand Ole Opry debut in Nov 2021

Chart Performance (URLs for each):
- first_country_radio_no1_urls                     # URLs supporting first country radio #1 with "Son of a Sinner" in Jan 2023
- riaa_son_of_a_sinner_2x_urls                     # URLs supporting "Son of a Sinner" certified 2× Platinum by RIAA
- need_a_favor_dual_charts_urls                    # URLs supporting "'Need a Favor' topped both Mainstream Rock Airplay and Country Airplay"
- bridgestone_sellout_2022_urls                    # URLs supporting sold out Nashville's Bridgestone Arena on Dec 9, 2022

Awards and Recognition (URLs for each):
- grammy_2024_bna_urls                             # URLs supporting nomination for Best New Artist at the 66th Grammys (2024)
- grammy_2024_save_me_duo_urls                     # URLs supporting "'Save Me' with Lainey Wilson" nominated for Best Country Duo/Group Performance (2024)
- cma_2023_new_artist_win_urls                     # URLs supporting CMA New Artist of the Year win (57th CMA Awards, 2023)
- grammy_2026_three_wins_urls                      # URLs supporting three Grammy wins at the 67th Annual Grammy Awards (2026)

Discography (URLs for each):
- ballads_of_the_broken_urls                       # URLs supporting release date (Sep 17, 2021) and 10 tracks for "Ballads of the Broken"
- whitsitt_chapel_release_urls                     # URLs supporting "Whitsitt Chapel" release date (June 2, 2023)
- whitsitt_chapel_lead_single_urls                 # URLs supporting "'Need a Favor' was the lead single from 'Whitsitt Chapel'"
- beautifully_broken_urls                          # URLs supporting "Beautifully Broken" release date (Oct 11, 2024) and 22 tracks

Personal Life (URLs for each):
- marriage_since_2016_urls                         # URLs supporting marriage to Alisa DeFord (Bunnie XO) since 2016
- spouse_dumb_blonde_urls                          # URLs supporting that the spouse hosts the "Dumb Blonde" podcast
- snl_season50_sept_2024_urls                      # URLs supporting musical guest appearance on SNL Season 50 in Sept 2024
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _artist_ref(stage_name: Optional[str]) -> str:
    """Return a robust reference to the artist for claim text."""
    if stage_name and stage_name.strip():
        # Include legal name alias for robust matching on source pages
        return f"{stage_name.strip()} (legal name: Jason Bradley DeFord)"
    return "the artist (legal name: Jason Bradley DeFord)"


def _presence_instruction() -> str:
    return (
        "Your task is to verify ONLY whether the provided answer text explicitly contains this statement "
        "or a clearly equivalent paraphrase. Case and minor wording variations are acceptable "
        "(e.g., 'Dec. 4, 1984' ~ 'December 4, 1984'). Focus on the answer content; do not use external knowledge."
    )


def _url_support_instruction() -> str:
    return (
        "Verify that the webpage(s) explicitly support the stated claim. Treat name variants reasonably "
        "(e.g., 'Jelly Roll' and 'Jason Bradley DeFord' refer to the same person). Accept common formatting variations "
        "for dates and titles. If the page is irrelevant or does not support the claim clearly, mark it as NOT supported."
    )


async def _add_fact_and_url_pair(
    evaluator: Evaluator,
    parent,
    id_base: str,
    requirement_desc: str,
    fact_leaf_id: str,
    fact_desc: str,
    fact_presence_claim: str,
    url_leaf_id: str,
    url_desc: str,
    url_support_claim: str,
    urls: List[str],
    critical: bool = True,
) -> None:
    """Add a requirement node with two leaves: presence-in-answer and URL support."""
    req_node = evaluator.add_parallel(
        id=id_base,
        desc=requirement_desc,
        parent=parent,
        critical=critical
    )

    # Fact presence in answer
    fact_node = evaluator.add_leaf(
        id=fact_leaf_id,
        desc=fact_desc,
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=fact_presence_claim,
        node=fact_node,
        sources=None,
        additional_instruction=_presence_instruction()
    )

    # URL support for the fact
    url_node = evaluator.add_leaf(
        id=url_leaf_id,
        desc=url_desc,
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=url_support_claim,
        node=url_node,
        sources=urls,
        additional_instruction=_url_support_instruction()
    )


# --------------------------------------------------------------------------- #
# Category builders                                                           #
# --------------------------------------------------------------------------- #
async def build_biographical_checks(evaluator: Evaluator, parent, data: ArtistEvidenceExtraction) -> None:
    bio_node = evaluator.add_parallel(
        id="Biographical_Requirements",
        desc="Biographical requirements (each must have a matching fact + URL).",
        parent=parent,
        critical=True
    )
    artist = _artist_ref(data.stage_name)

    # Birth Date
    await _add_fact_and_url_pair(
        evaluator, bio_node,
        id_base="Birth_Date",
        requirement_desc="Birth date verification (fact + URL).",
        fact_leaf_id="Birth_Date_Fact",
        fact_desc="States the artist was born on December 4, 1984.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist was born on December 4, 1984.",
        url_leaf_id="Birth_Date_URL",
        url_desc="Provides at least one supporting URL for the birth date claim.",
        url_support_claim=f"{artist} was born on December 4, 1984.",
        urls=data.birth_date_urls
    )

    # Origin Antioch (Nashville)
    await _add_fact_and_url_pair(
        evaluator, bio_node,
        id_base="Origin_Antioch_Nashville",
        requirement_desc="Origin verification (fact + URL).",
        fact_leaf_id="Origin_Fact",
        fact_desc="States the artist is from the Antioch neighborhood of Nashville, Tennessee.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist is from the Antioch neighborhood of Nashville, Tennessee.",
        url_leaf_id="Origin_URL",
        url_desc="Provides at least one supporting URL for the origin claim.",
        url_support_claim=f"{artist} is from the Antioch neighborhood of Nashville, Tennessee.",
        urls=data.origin_antioch_urls
    )

    # Legal Name
    await _add_fact_and_url_pair(
        evaluator, bio_node,
        id_base="Legal_Name",
        requirement_desc="Legal name verification (fact + URL).",
        fact_leaf_id="Legal_Name_Fact",
        fact_desc="States the artist’s legal name is Jason Bradley DeFord.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist’s legal name is Jason Bradley DeFord.",
        url_leaf_id="Legal_Name_URL",
        url_desc="Provides at least one supporting URL for the legal name claim.",
        url_support_claim="The artist’s legal name is Jason Bradley DeFord.",
        urls=data.legal_name_urls
    )

    # Felony/Incarceration History
    await _add_fact_and_url_pair(
        evaluator, bio_node,
        id_base="Felony_Incarceration_History",
        requirement_desc="Incarceration/felony history verification (fact + URL).",
        fact_leaf_id="Felony_Incarceration_Fact",
        fact_desc="States the artist had a history of incarceration with felony convictions.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist had a history of incarceration with felony convictions.",
        url_leaf_id="Felony_Incarceration_URL",
        url_desc="Provides at least one supporting URL for the incarceration/felony history claim.",
        url_support_claim=f"{artist} has a history of incarceration with felony convictions.",
        urls=data.incarceration_felony_urls
    )

    # Pardon on Dec 18, 2025
    await _add_fact_and_url_pair(
        evaluator, bio_node,
        id_base="Pardon_Dec_18_2025",
        requirement_desc="Pardon verification (fact + URL).",
        fact_leaf_id="Pardon_Fact",
        fact_desc="States the artist received a pardon from Tennessee Governor Bill Lee on December 18, 2025.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist received a pardon from Tennessee Governor Bill Lee on December 18, 2025.",
        url_leaf_id="Pardon_URL",
        url_desc="Provides at least one supporting URL for the pardon claim.",
        url_support_claim=f"{artist} received a pardon from Tennessee Governor Bill Lee on December 18, 2025.",
        urls=data.pardon_2025_urls
    )


async def build_career_timeline_checks(evaluator: Evaluator, parent, data: ArtistEvidenceExtraction) -> None:
    career_node = evaluator.add_parallel(
        id="Career_Timeline_Requirements",
        desc="Career timeline requirements (each must have a matching fact + URL).",
        parent=parent,
        critical=True
    )
    artist = _artist_ref(data.stage_name)

    # Began in 2003 with mixtapes
    await _add_fact_and_url_pair(
        evaluator, career_node,
        id_base="Career_Began_2003_Mixtapes",
        requirement_desc="Career start verification (fact + URL).",
        fact_leaf_id="Career_Began_Fact",
        fact_desc="States the artist began their music career in 2003 with mixtapes.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist began their music career in 2003 with mixtapes.",
        url_leaf_id="Career_Began_URL",
        url_desc="Provides at least one supporting URL for the 2003 mixtapes career-start claim.",
        url_support_claim=f"{artist} began their music career in 2003 by releasing mixtapes.",
        urls=data.career_2003_mixtapes_urls
    )

    # Started in Southern hip-hop/rap
    await _add_fact_and_url_pair(
        evaluator, career_node,
        id_base="Started_Southern_Hiphop_Rap",
        requirement_desc="Early genre verification (fact + URL).",
        fact_leaf_id="Early_Genre_Fact",
        fact_desc="States the artist started in the Southern hip-hop/rap genre.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist started in the Southern hip-hop/rap genre.",
        url_leaf_id="Early_Genre_URL",
        url_desc="Provides at least one supporting URL for the early-genre claim.",
        url_support_claim=f"{artist} started in the Southern hip-hop/rap genre.",
        urls=data.started_southern_hiphop_urls
    )

    # Transitioned to country crossover
    await _add_fact_and_url_pair(
        evaluator, career_node,
        id_base="Transitioned_To_Country_Crossover",
        requirement_desc="Genre transition verification (fact + URL).",
        fact_leaf_id="Transition_Fact",
        fact_desc="States the artist transitioned to country music crossover.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist transitioned to country music crossover.",
        url_leaf_id="Transition_URL",
        url_desc="Provides at least one supporting URL for the genre-transition claim.",
        url_support_claim=f"{artist} transitioned to a country music crossover artist.",
        urls=data.transitioned_country_crossover_urls
    )

    # Grand Ole Opry debut in Nov 2021
    await _add_fact_and_url_pair(
        evaluator, career_node,
        id_base="Grand_Ole_Opry_Debut_Nov_2021",
        requirement_desc="Opry debut verification (fact + URL).",
        fact_leaf_id="Opry_Debut_Fact",
        fact_desc="States the artist made their Grand Ole Opry debut in November 2021.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist made their Grand Ole Opry debut in November 2021.",
        url_leaf_id="Opry_Debut_URL",
        url_desc="Provides at least one supporting URL for the Opry debut claim.",
        url_support_claim=f"{artist} made their Grand Ole Opry debut in November 2021.",
        urls=data.opry_debut_nov_2021_urls
    )


async def build_chart_performance_checks(evaluator: Evaluator, parent, data: ArtistEvidenceExtraction) -> None:
    chart_node = evaluator.add_parallel(
        id="Chart_Performance_Requirements",
        desc="Chart performance requirements (each must have a matching fact + URL).",
        parent=parent,
        critical=True
    )
    artist = _artist_ref(data.stage_name)

    # First country radio #1 with "Son of a Sinner" (Jan 2023)
    await _add_fact_and_url_pair(
        evaluator, chart_node,
        id_base="First_Country_Radio_Number1_Son_Of_A_Sinner_Jan_2023",
        requirement_desc="First country radio #1 verification (fact + URL).",
        fact_leaf_id="First_Number1_Fact",
        fact_desc="States the artist’s first country radio #1 was 'Son of a Sinner' in January 2023.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist achieved their first country radio #1 with 'Son of a Sinner' in January 2023.",
        url_leaf_id="First_Number1_URL",
        url_desc="Provides at least one supporting URL for the first country radio #1 claim.",
        url_support_claim=f"{artist} achieved their first country radio #1 with 'Son of a Sinner' in January 2023.",
        urls=data.first_country_radio_no1_urls
    )

    # RIAA 2× Platinum for "Son of a Sinner"
    await _add_fact_and_url_pair(
        evaluator, chart_node,
        id_base="Son_Of_A_Sinner_RIAA_2x_Platinum_2M",
        requirement_desc="RIAA certification verification (fact + URL).",
        fact_leaf_id="RIAA_Fact",
        fact_desc="States 'Son of a Sinner' was certified 2× Platinum (2,000,000 units) by the RIAA.",
        fact_presence_claim="In the provided answer, it is explicitly stated that 'Son of a Sinner' was certified 2× Platinum (2,000,000 units) by the RIAA.",
        url_leaf_id="RIAA_URL",
        url_desc="Provides at least one supporting URL for the RIAA certification claim.",
        url_support_claim=f"The song 'Son of a Sinner' by {artist} was certified 2× Platinum by the RIAA.",
        urls=data.riaa_son_of_a_sinner_2x_urls
    )

    # "Need a Favor" topped both Mainstream Rock Airplay and Country Airplay charts
    await _add_fact_and_url_pair(
        evaluator, chart_node,
        id_base="Need_A_Favor_Topped_Rock_And_Country_Airplay",
        requirement_desc="Dual-chart topping verification (fact + URL).",
        fact_leaf_id="Need_A_Favor_Charts_Fact",
        fact_desc="States 'Need a Favor' topped both the Mainstream Rock Airplay chart and the Country Airplay chart.",
        fact_presence_claim="In the provided answer, it is explicitly stated that 'Need a Favor' topped both the Mainstream Rock Airplay chart and the Country Airplay chart.",
        url_leaf_id="Need_A_Favor_Charts_URL",
        url_desc="Provides at least one supporting URL for the dual-chart topping claim.",
        url_support_claim=f"'Need a Favor' by {artist} reached number one on both Billboard's Mainstream Rock Airplay chart and the Country Airplay chart.",
        urls=data.need_a_favor_dual_charts_urls
    )

    # Sold out Bridgestone Arena on Dec 9, 2022
    await _add_fact_and_url_pair(
        evaluator, chart_node,
        id_base="Sold_Out_Bridgestone_Arena_Dec_9_2022",
        requirement_desc="Bridgestone Arena sell-out verification (fact + URL).",
        fact_leaf_id="Bridgestone_Fact",
        fact_desc="States the artist sold out Nashville’s Bridgestone Arena on December 9, 2022.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist sold out Nashville’s Bridgestone Arena on December 9, 2022.",
        url_leaf_id="Bridgestone_URL",
        url_desc="Provides at least one supporting URL for the Bridgestone Arena sell-out claim.",
        url_support_claim=f"{artist} sold out Nashville’s Bridgestone Arena on December 9, 2022.",
        urls=data.bridgestone_sellout_2022_urls
    )


async def build_awards_checks(evaluator: Evaluator, parent, data: ArtistEvidenceExtraction) -> None:
    awards_node = evaluator.add_parallel(
        id="Awards_And_Recognition_Requirements",
        desc="Awards/recognition requirements (each must have a matching fact + URL).",
        parent=parent,
        critical=True
    )
    artist = _artist_ref(data.stage_name)

    # Grammy 2024 Best New Artist nomination
    await _add_fact_and_url_pair(
        evaluator, awards_node,
        id_base="Grammy_2024_Best_New_Artist_Nomination",
        requirement_desc="Grammy nomination verification (fact + URL).",
        fact_leaf_id="BNA_2024_Fact",
        fact_desc="States the artist was nominated for Best New Artist at the 66th Annual Grammy Awards (2024).",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist was nominated for Best New Artist at the 66th Annual Grammy Awards (2024).",
        url_leaf_id="BNA_2024_URL",
        url_desc="Provides at least one supporting URL for the Best New Artist nomination claim.",
        url_support_claim=f"{artist} was nominated for Best New Artist at the 66th Annual Grammy Awards (2024).",
        urls=data.grammy_2024_bna_urls
    )

    # "Save Me" with Lainey Wilson nomination (Best Country Duo/Group Performance, 2024)
    await _add_fact_and_url_pair(
        evaluator, awards_node,
        id_base="Grammy_2024_Save_Me_DuoGroup_Nomination",
        requirement_desc="Specific nomination verification (fact + URL).",
        fact_leaf_id="Save_Me_Nom_Fact",
        fact_desc="States 'Save Me' with Lainey Wilson was nominated for Best Country Duo/Group Performance at the 2024 Grammys.",
        fact_presence_claim="In the provided answer, it is explicitly stated that 'Save Me' with Lainey Wilson was nominated for Best Country Duo/Group Performance at the 2024 Grammys.",
        url_leaf_id="Save_Me_Nom_URL",
        url_desc="Provides at least one supporting URL for the 'Save Me' nomination claim.",
        url_support_claim=f"The collaboration 'Save Me' by {artist} with Lainey Wilson was nominated for Best Country Duo/Group Performance at the 2024 Grammy Awards.",
        urls=data.grammy_2024_save_me_duo_urls
    )

    # CMA 2023 New Artist of the Year win
    await _add_fact_and_url_pair(
        evaluator, awards_node,
        id_base="CMA_2023_New_Artist_Of_The_Year_WWin" if False else "CMA_2023_New_Artist_Of_The_Year_Win",
        requirement_desc="CMA win verification (fact + URL).",
        fact_leaf_id="CMA_Win_Fact",
        fact_desc="States the artist won CMA New Artist of the Year at the 57th Annual CMA Awards (2023).",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist won CMA New Artist of the Year at the 57th Annual CMA Awards (2023).",
        url_leaf_id="CMA_Win_URL",
        url_desc="Provides at least one supporting URL for the CMA win claim.",
        url_support_claim=f"{artist} won CMA New Artist of the Year at the 57th Annual CMA Awards (2023).",
        urls=data.cma_2023_new_artist_win_urls
    )

    # Grammy 2026 three wins (67th)
    await _add_fact_and_url_pair(
        evaluator, awards_node,
        id_base="Grammy_2026_Three_Wins",
        requirement_desc="2026 Grammy wins verification (fact + URL).",
        fact_leaf_id="Grammy_2026_Wins_Fact",
        fact_desc="States the artist won three Grammy awards at the 67th Annual Grammy Awards (2026).",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist won three Grammy awards at the 67th Annual Grammy Awards (2026).",
        url_leaf_id="Grammy_2026_Wins_URL",
        url_desc="Provides at least one supporting URL for the 2026 three-Grammys claim.",
        url_support_claim=f"{artist} won three Grammy awards at the 67th Annual Grammy Awards (2026).",
        urls=data.grammy_2026_three_wins_urls
    )


async def build_discography_checks(evaluator: Evaluator, parent, data: ArtistEvidenceExtraction) -> None:
    disco_node = evaluator.add_parallel(
        id="Discography_Requirements",
        desc="Discography requirements (each must have a matching fact + URL).",
        parent=parent,
        critical=True
    )
    artist = _artist_ref(data.stage_name)

    # Ballads of the Broken – Sep 17, 2021; 10 tracks
    await _add_fact_and_url_pair(
        evaluator, disco_node,
        id_base="Ballads_Of_The_Broken_Release_And_Tracks",
        requirement_desc="Album verification (fact + URL).",
        fact_leaf_id="Ballads_Fact",
        fact_desc="States the artist released 'Ballads of the Broken' on September 17, 2021, and it contains 10 tracks.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist released 'Ballads of the Broken' on September 17, 2021, and it contains 10 tracks.",
        url_leaf_id="Ballads_URL",
        url_desc="Provides at least one supporting URL for the 'Ballads of the Broken' release date and track count claim.",
        url_support_claim=f"{artist} released the album 'Ballads of the Broken' on September 17, 2021, and the album contains 10 tracks.",
        urls=data.ballads_of_the_broken_urls
    )

    # Whitsitt Chapel – release date June 2, 2023
    await _add_fact_and_url_pair(
        evaluator, disco_node,
        id_base="Whitsitt_Chapel_Release_Date",
        requirement_desc="Album release verification (fact + URL).",
        fact_leaf_id="Whitsitt_Release_Fact",
        fact_desc="States the artist released 'Whitsitt Chapel' on June 2, 2023.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist released 'Whitsitt Chapel' on June 2, 2023.",
        url_leaf_id="Whitsitt_Release_URL",
        url_desc="Provides at least one supporting URL for the 'Whitsitt Chapel' release date claim.",
        url_support_claim=f"{artist} released the album 'Whitsitt Chapel' on June 2, 2023.",
        urls=data.whitsitt_chapel_release_urls
    )

    # Whitsitt Chapel – lead single "Need a Favor"
    await _add_fact_and_url_pair(
        evaluator, disco_node,
        id_base="Whitsitt_Chapel_Lead_Single",
        requirement_desc="Lead single verification (fact + URL).",
        fact_leaf_id="Lead_Single_Fact",
        fact_desc="States 'Need a Favor' was the lead single from 'Whitsitt Chapel'.",
        fact_presence_claim="In the provided answer, it is explicitly stated that 'Need a Favor' was the lead single from 'Whitsitt Chapel'.",
        url_leaf_id="Lead_Single_URL",
        url_desc="Provides at least one supporting URL for the lead single claim.",
        url_support_claim=f"'Need a Favor' was the lead single from {artist}'s album 'Whitsitt Chapel'.",
        urls=data.whitsitt_chapel_lead_single_urls
    )

    # Beautifully Broken – Oct 11, 2024; 22 tracks
    await _add_fact_and_url_pair(
        evaluator, disco_node,
        id_base="Beautifully_Broken_Release_And_Tracks",
        requirement_desc="Album verification (fact + URL).",
        fact_leaf_id="Beautifully_Broken_Fact",
        fact_desc="States the artist released 'Beautifully Broken' on October 11, 2024, and it contains 22 tracks.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist released 'Beautifully Broken' on October 11, 2024, and it contains 22 tracks.",
        url_leaf_id="Beautifully_Broken_URL",
        url_desc="Provides at least one supporting URL for the 'Beautifully Broken' release date and track count claim.",
        url_support_claim=f"{artist} released the album 'Beautifully Broken' on October 11, 2024, and the album contains 22 tracks.",
        urls=data.beautifully_broken_urls
    )


async def build_personal_life_checks(evaluator: Evaluator, parent, data: ArtistEvidenceExtraction) -> None:
    personal_node = evaluator.add_parallel(
        id="Personal_Life_Requirements",
        desc="Personal life requirements (each must have a matching fact + URL).",
        parent=parent,
        critical=True
    )
    artist = _artist_ref(data.stage_name)

    # Marriage since 2016 to Alisa DeFord (Bunnie XO)
    await _add_fact_and_url_pair(
        evaluator, personal_node,
        id_base="Marriage_Since_2016",
        requirement_desc="Marriage verification (fact + URL).",
        fact_leaf_id="Marriage_Fact",
        fact_desc="States the artist has been married to Alisa DeFord (Bunnie XO) since 2016.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist has been married to Alisa DeFord (Bunnie XO) since 2016.",
        url_leaf_id="Marriage_URL",
        url_desc="Provides at least one supporting URL for the marriage claim.",
        url_support_claim=f"{artist} has been married to Alisa DeFord (known as Bunnie XO) since 2016.",
        urls=data.marriage_since_2016_urls
    )

    # Spouse hosts "Dumb Blonde" podcast
    await _add_fact_and_url_pair(
        evaluator, personal_node,
        id_base="Spouse_Hosts_Dumb_Blonde_Podcast",
        requirement_desc="Spouse podcast-hosting verification (fact + URL).",
        fact_leaf_id="Podcast_Fact",
        fact_desc="States the artist’s spouse hosts the 'Dumb Blonde' podcast.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist’s spouse hosts the 'Dumb Blonde' podcast.",
        url_leaf_id="Podcast_URL",
        url_desc="Provides at least one supporting URL for the 'Dumb Blonde' hosting claim.",
        url_support_claim=f"{artist}'s spouse, Alisa 'Bunnie XO' DeFord, hosts the 'Dumb Blonde' podcast.",
        urls=data.spouse_dumb_blonde_urls
    )

    # SNL Season 50 musical guest in September 2024
    await _add_fact_and_url_pair(
        evaluator, personal_node,
        id_base="SNL_Season_50_Sep_2024_Musical_Guest",
        requirement_desc="SNL appearance verification (fact + URL).",
        fact_leaf_id="SNL_Fact",
        fact_desc="States the artist performed as musical guest on Saturday Night Live Season 50 in September 2024.",
        fact_presence_claim="In the provided answer, it is explicitly stated that the artist performed as musical guest on Saturday Night Live Season 50 in September 2024.",
        url_leaf_id="SNL_URL",
        url_desc="Provides at least one supporting URL for the SNL appearance claim.",
        url_support_claim=f"{artist} performed as the musical guest on Saturday Night Live Season 50 in September 2024.",
        urls=data.snl_season50_sept_2024_urls
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
    Evaluate an answer for the Artist Identification task using the Mind2Web2 framework.
    """
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

    # Extract stage name and categorized URLs
    extraction = await evaluator.extract(
        prompt=prompt_extract_artist_evidence(),
        template_class=ArtistEvidenceExtraction,
        extraction_name="artist_evidence"
    )

    # Build the task node (critical)
    task_node = evaluator.add_sequential(
        id="Artist_Identification_Task",
        desc="Identify the artist that satisfies all listed constraints and provide the required supporting URLs.",
        parent=root,
        critical=True
    )

    # Provide artist stage name (critical existence check)
    stage_name_exists = bool(extraction.stage_name and extraction.stage_name.strip())
    evaluator.add_custom_node(
        result=stage_name_exists,
        id="Provide_Artist_Stage_Name",
        desc="Provide the artist’s stage name.",
        parent=task_node,
        critical=True
    )

    # Verify all requirements with sources (critical, parallel)
    verify_all_node = evaluator.add_parallel(
        id="Verify_All_Requirements_With_Sources",
        desc="For every listed requirement, provide (1) a matching fact claim and (2) at least one supporting URL.",
        parent=task_node,
        critical=True
    )

    # Build category subtrees (all critical)
    await build_biographical_checks(evaluator, verify_all_node, extraction)
    await build_career_timeline_checks(evaluator, verify_all_node, extraction)
    await build_chart_performance_checks(evaluator, verify_all_node, extraction)
    await build_awards_checks(evaluator, verify_all_node, extraction)
    await build_discography_checks(evaluator, verify_all_node, extraction)
    await build_personal_life_checks(evaluator, verify_all_node, extraction)

    # Optional: add expected facts (for debugging visibility)
    evaluator.add_custom_info(
        info={
            "expected_facts": {
                "biographical": {
                    "birth_date": "Born on December 4, 1984",
                    "origin": "From Antioch neighborhood of Nashville, Tennessee",
                    "legal_name": "Legal name: Jason Bradley DeFord",
                    "incarceration": "History of incarceration with felony convictions",
                    "pardon": "Pardoned by Tennessee Governor Bill Lee on December 18, 2025"
                },
                "career_timeline": {
                    "start_2003": "Began music career in 2003 with mixtapes",
                    "early_genre": "Started in the Southern hip-hop/rap genre",
                    "transition": "Transitioned to country music crossover",
                    "opry_debut": "Grand Ole Opry debut in November 2021"
                },
                "chart_performance": {
                    "first_no1": "First country radio #1 with 'Son of a Sinner' in January 2023",
                    "riaa": "'Son of a Sinner' certified 2× Platinum by RIAA",
                    "need_a_favor_dual": "'Need a Favor' topped both Mainstream Rock Airplay and Country Airplay",
                    "bridgestone": "Sold out Nashville's Bridgestone Arena on December 9, 2022"
                },
                "awards": {
                    "grammy_bna_2024": "Best New Artist nomination at 66th Grammys (2024)",
                    "save_me_nom": "'Save Me' with Lainey Wilson nominated for Best Country Duo/Group Performance (2024)",
                    "cma_2023_win": "CMA New Artist of the Year (57th CMA Awards, 2023)",
                    "grammy_2026_wins": "Three Grammy wins at the 67th Annual Grammy Awards (2026)"
                },
                "discography": {
                    "ballads": "'Ballads of the Broken' released Sep 17, 2021; 10 tracks",
                    "whitsitt_release": "'Whitsitt Chapel' released June 2, 2023",
                    "whitsitt_lead_single": "'Need a Favor' lead single from 'Whitsitt Chapel'",
                    "beautifully_broken": "'Beautifully Broken' released Oct 11, 2024; 22 tracks"
                },
                "personal_life": {
                    "marriage": "Married to Alisa DeFord (Bunnie XO) since 2016",
                    "spouse_podcast": "Spouse hosts 'Dumb Blonde' podcast",
                    "snl_appearance": "Musical guest on Saturday Night Live Season 50 in September 2024"
                }
            }
        },
        info_type="rubric_expectations"
    )

    return evaluator.get_summary()