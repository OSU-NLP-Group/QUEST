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
TASK_ID = "grammy_2026_major_winners"
TASK_DESCRIPTION = (
    "At the 68th Annual Grammy Awards held on February 1, 2026, several historic wins occurred across major "
    "categories. Identify the winners in the following four categories and provide comprehensive details about their "
    "winning albums:\n\n"
    "1. Album of the Year: Provide the album title, artist name, release date, track count, primary producers, and the "
    "historical significance of this win.\n\n"
    "2. Best New Artist: Provide the artist's name, their debut album title and its peak position on the UK Albums "
    "Chart, and details about their second album (title, release date, and UK chart peak).\n\n"
    "3. Best Rap Album: Provide the album title, artist name, release date, track count, whether it was a surprise "
    "release, the artist's total career Grammy wins after the 2026 ceremony, and whether the artist achieved a "
    "historic Grammy record.\n\n"
    "4. Best Pop Vocal Album: Provide the album title, artist name, and the total number of nominees in this category.\n\n"
    "For each category, include at least one reference URL that verifies the winner information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AlbumOfTheYearInfo(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None
    release_date: Optional[str] = None
    track_count: Optional[str] = None
    primary_producers: List[str] = Field(default_factory=list)
    historical_significance: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SecondAlbumDetails(BaseModel):
    second_album_title: Optional[str] = None
    second_album_release_date: Optional[str] = None
    second_album_uk_chart_peak: Optional[str] = None


class BestNewArtistInfo(BaseModel):
    artist_name: Optional[str] = None
    debut_album_title: Optional[str] = None
    debut_album_uk_chart_peak: Optional[str] = None
    second_album: Optional[SecondAlbumDetails] = None
    reference_urls: List[str] = Field(default_factory=list)


class BestRapAlbumInfo(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None
    release_date: Optional[str] = None
    track_count: Optional[str] = None
    release_type: Optional[str] = None  # e.g., "surprise release" or "announced in advance"
    artist_career_grammy_total: Optional[str] = None
    historic_achievement: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BestPopVocalAlbumInfo(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None
    total_nominees_in_category: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class WinnersExtraction(BaseModel):
    album_of_the_year: Optional[AlbumOfTheYearInfo] = None
    best_new_artist: Optional[BestNewArtistInfo] = None
    best_rap_album: Optional[BestRapAlbumInfo] = None
    best_pop_vocal_album: Optional[BestPopVocalAlbumInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_winners() -> str:
    return """
    You must extract structured information for four Grammy 2026 categories from the provided answer text.
    IMPORTANT:
    - Extract exactly what the answer provides; do not infer or invent.
    - For all URL fields, only include explicit URLs mentioned in the answer (plain or markdown link).
    - If an item is not present, set it to null (for strings) or [] (for lists).

    Return a single JSON object with the following top-level fields:
    - album_of_the_year: object or null
    - best_new_artist: object or null
    - best_rap_album: object or null
    - best_pop_vocal_album: object or null

    Each object should follow these schemas:

    album_of_the_year:
      - album_title: string or null
      - artist_name: string or null
      - release_date: string or null        # accept any reasonable date format as a string
      - track_count: string or null         # keep as string (e.g., "12", "10 (standard), 14 (deluxe)")
      - primary_producers: string[]         # names; can be empty
      - historical_significance: string or null
      - reference_urls: string[]            # at least one URL if provided; only URLs explicitly in answer

    best_new_artist:
      - artist_name: string or null
      - debut_album_title: string or null
      - debut_album_uk_chart_peak: string or null   # e.g., "1", "No. 2", "Top 10"
      - second_album: object or null with:
          - second_album_title: string or null
          - second_album_release_date: string or null
          - second_album_uk_chart_peak: string or null
      - reference_urls: string[]            # only URLs from the answer

    best_rap_album:
      - album_title: string or null
      - artist_name: string or null
      - release_date: string or null
      - track_count: string or null
      - release_type: string or null        # e.g., "surprise release", "announced in advance"
      - artist_career_grammy_total: string or null  # total wins after the 2026 ceremony
      - historic_achievement: string or null        # description (e.g., "most wins by a rapper")
      - reference_urls: string[]

    best_pop_vocal_album:
      - album_title: string or null
      - artist_name: string or null
      - total_nominees_in_category: string or null  # keep as string (e.g., "5")
      - reference_urls: string[]

    Ensure URLs are valid and include http:// or https:// (prepend http:// if missing).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _fmt_list_str(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_album_of_the_year(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AlbumOfTheYearInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Album_of_the_Year_Winner",
        desc="Information about the Album of the Year winner at the 2026 Grammy Awards",
        parent=parent_node,
        critical=False,
    )

    refs = info.reference_urls if info else []
    refs_present = evaluator.add_custom_node(
        result=bool(refs),
        id="aoty_reference_url",
        desc="A URL that verifies the Album of the Year winner information is provided",
        parent=node,
        critical=True,
    )

    # Album Title (critical)
    album_title_leaf = evaluator.add_leaf(
        id="aoty_album_title",
        desc="The title of the winning album",
        parent=node,
        critical=True,
    )
    album_title = (info.album_title if info else None) or ""
    await evaluator.verify(
        claim=f"At the 68th Annual Grammy Awards (February 1, 2026), the Album of the Year winning album was titled '{album_title}'.",
        node=album_title_leaf,
        sources=refs,
        additional_instruction="Verify explicitly that the album title listed is the Album of the Year winner for the 2026 Grammys.",
    )

    # Artist Name (critical)
    artist_leaf = evaluator.add_leaf(
        id="aoty_artist_name",
        desc="The name of the artist who won",
        parent=node,
        critical=True,
    )
    artist_name = (info.artist_name if info else None) or ""
    await evaluator.verify(
        claim=f"The artist who won Album of the Year at the 68th Annual Grammy Awards (2026) was {artist_name}.",
        node=artist_leaf,
        sources=refs,
        additional_instruction="Check the official winner listing for Album of the Year.",
    )

    # Release Date (critical)
    release_date_leaf = evaluator.add_leaf(
        id="aoty_release_date",
        desc="The release date of the album",
        parent=node,
        critical=True,
    )
    release_date = (info.release_date if info else None) or ""
    await evaluator.verify(
        claim=f"The album '{album_title}' was released on {release_date}.",
        node=release_date_leaf,
        sources=refs,
        additional_instruction="Accept reasonable date formats. Verify the album release date from the provided sources.",
    )

    # Track Count (non-critical)
    track_count_leaf = evaluator.add_leaf(
        id="aoty_track_count",
        desc="The number of tracks on the album",
        parent=node,
        critical=False,
    )
    track_count = (info.track_count if info else None) or ""
    await evaluator.verify(
        claim=f"The album '{album_title}' contains {track_count} tracks.",
        node=track_count_leaf,
        sources=refs,
        additional_instruction="If multiple editions exist, accept a clearly indicated standard edition track count if available.",
    )

    # Primary Producers (non-critical)
    producers_leaf = evaluator.add_leaf(
        id="aoty_primary_producers",
        desc="The primary producers or production team for the album",
        parent=node,
        critical=False,
    )
    producers_str = _fmt_list_str(info.primary_producers if info else [])
    await evaluator.verify(
        claim=f"The primary producers of '{album_title}' include: {producers_str}.",
        node=producers_leaf,
        sources=refs,
        additional_instruction="Allow reasonable variants (e.g., including executive producers or co-producers if the source frames them as primary).",
    )

    # Historical Significance (critical)
    significance_leaf = evaluator.add_leaf(
        id="aoty_historical_significance",
        desc="The historical significance of this win (e.g., first Spanish-language album to win)",
        parent=node,
        critical=True,
    )
    significance = (info.historical_significance if info else None) or ""
    await evaluator.verify(
        claim=f"This Album of the Year win was historically significant because: {significance}.",
        node=significance_leaf,
        sources=refs,
        additional_instruction="Verify explicit statements about historical 'firsts' or records associated with the AOTY win.",
    )


async def verify_best_new_artist(
    evaluator: Evaluator,
    parent_node,
    info: Optional[BestNewArtistInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Best_New_Artist_Winner",
        desc="Information about the Best New Artist winner at the 2026 Grammy Awards",
        parent=parent_node,
        critical=False,
    )

    refs = info.reference_urls if info else []
    refs_present = evaluator.add_custom_node(
        result=bool(refs),
        id="bna_reference_url",
        desc="A URL that verifies the Best New Artist winner information is provided",
        parent=node,
        critical=True,
    )

    # Artist Name (critical)
    artist_leaf = evaluator.add_leaf(
        id="bna_artist_name",
        desc="The name of the Best New Artist winner",
        parent=node,
        critical=True,
    )
    artist_name = (info.artist_name if info else None) or ""
    await evaluator.verify(
        claim=f"The Best New Artist winner at the 68th Annual Grammy Awards (2026) was {artist_name}.",
        node=artist_leaf,
        sources=refs,
        additional_instruction="Use official winner listings or reputable media coverage to confirm the winner.",
    )

    # Debut Album Title (non-critical)
    debut_title_leaf = evaluator.add_leaf(
        id="bna_debut_album_title",
        desc="The title of the artist's debut album",
        parent=node,
        critical=False,
    )
    debut_title = (info.debut_album_title if info else None) or ""
    await evaluator.verify(
        claim=f"{artist_name}'s debut album is titled '{debut_title}'.",
        node=debut_title_leaf,
        sources=refs,
        additional_instruction="Verify that this album is indeed described as the artist's debut album.",
    )

    # Debut Album UK Chart Peak (non-critical)
    debut_peak_leaf = evaluator.add_leaf(
        id="bna_debut_album_uk_peak",
        desc="The peak chart position of the debut album on the UK Albums Chart",
        parent=node,
        critical=False,
    )
    debut_peak = (info.debut_album_uk_chart_peak if info else None) or ""
    await evaluator.verify(
        claim=f"The debut album '{debut_title}' peaked at position {debut_peak} on the UK Albums Chart.",
        node=debut_peak_leaf,
        sources=refs,
        additional_instruction="Accept minor variations in formatting such as 'No. 1' vs '1'.",
    )

    # Second Album Details (non-critical group)
    second_group = evaluator.add_parallel(
        id="bna_second_album_details",
        desc="Information about the artist's second album",
        parent=node,
        critical=False,
    )
    second = info.second_album if info else None
    second_title = (second.second_album_title if second else None) or ""
    second_release = (second.second_album_release_date if second else None) or ""
    second_peak = (second.second_album_uk_chart_peak if second else None) or ""

    second_title_leaf = evaluator.add_leaf(
        id="bna_second_album_title",
        desc="The title of the second album",
        parent=second_group,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{artist_name}'s second album is titled '{second_title}'.",
        node=second_title_leaf,
        sources=refs,
        additional_instruction="Ensure the album is indeed the artist's second full-length studio album.",
    )

    second_release_leaf = evaluator.add_leaf(
        id="bna_second_album_release_date",
        desc="The release date of the second album",
        parent=second_group,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The release date of the second album '{second_title}' was {second_release}.",
        node=second_release_leaf,
        sources=refs,
        additional_instruction="Accept reasonable date formats.",
    )

    second_peak_leaf = evaluator.add_leaf(
        id="bna_second_album_uk_peak",
        desc="The peak chart position of the second album on the UK Albums Chart",
        parent=second_group,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The second album '{second_title}' peaked at position {second_peak} on the UK Albums Chart.",
        node=second_peak_leaf,
        sources=refs,
        additional_instruction="Accept minor formatting variations for chart peaks.",
    )


async def verify_best_rap_album(
    evaluator: Evaluator,
    parent_node,
    info: Optional[BestRapAlbumInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Best_Rap_Album_Winner",
        desc="Information about the Best Rap Album winner at the 2026 Grammy Awards",
        parent=parent_node,
        critical=False,
    )

    refs = info.reference_urls if info else []
    refs_present = evaluator.add_custom_node(
        result=bool(refs),
        id="rap_reference_url",
        desc="A URL that verifies the Best Rap Album winner information is provided",
        parent=node,
        critical=True,
    )

    album_title = (info.album_title if info else None) or ""
    artist_name = (info.artist_name if info else None) or ""
    release_date = (info.release_date if info else None) or ""
    track_count = (info.track_count if info else None) or ""
    release_type = (info.release_type if info else None) or ""
    total_grammys = (info.artist_career_grammy_total if info else None) or ""
    historic = (info.historic_achievement if info else None) or ""

    # Album Title (critical)
    rap_album_leaf = evaluator.add_leaf(
        id="rap_album_title",
        desc="The title of the winning rap album",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Best Rap Album winner at the 68th Annual Grammy Awards (2026) was '{album_title}'.",
        node=rap_album_leaf,
        sources=refs,
        additional_instruction="Verify from official winner lists or reputable coverage.",
    )

    # Artist Name (critical)
    rap_artist_leaf = evaluator.add_leaf(
        id="rap_artist_name",
        desc="The name of the artist who won Best Rap Album",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The artist who won Best Rap Album at the 68th Annual Grammy Awards (2026) was {artist_name}.",
        node=rap_artist_leaf,
        sources=refs,
        additional_instruction="Verify the named winner for the category.",
    )

    # Release Date (critical)
    rap_release_date_leaf = evaluator.add_leaf(
        id="rap_release_date",
        desc="The release date of the album",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The album '{album_title}' was released on {release_date}.",
        node=rap_release_date_leaf,
        sources=refs,
        additional_instruction="Check album metadata on credible sources.",
    )

    # Track Count (non-critical)
    rap_track_leaf = evaluator.add_leaf(
        id="rap_track_count",
        desc="The number of tracks on the album",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The album '{album_title}' contains {track_count} tracks.",
        node=rap_track_leaf,
        sources=refs,
        additional_instruction="If multiple editions exist, accept clearly identified standard edition track count.",
    )

    # Release Type (non-critical)
    rap_release_type_leaf = evaluator.add_leaf(
        id="rap_release_type",
        desc="Whether the album was a surprise release or announced in advance",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Reports characterize the release of '{album_title}' as: '{release_type}'.",
        node=rap_release_type_leaf,
        sources=refs,
        additional_instruction="Interpret 'surprise release' vs 'announced' based on the phrasing in the sources.",
    )

    # Artist career Grammy total after 2026 ceremony (critical)
    rap_total_grammys_leaf = evaluator.add_leaf(
        id="rap_artist_career_grammy_total",
        desc="The artist's total career Grammy wins after the 2026 ceremony",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Following the 2026 Grammys, {artist_name} has a career total of {total_grammys} Grammy wins.",
        node=rap_total_grammys_leaf,
        sources=refs,
        additional_instruction="Verify totals stated in coverage that includes the 2026 ceremony outcome.",
    )

    # Historic achievement (critical)
    rap_historic_leaf = evaluator.add_leaf(
        id="rap_historic_achievement",
        desc="Whether the artist achieved a historic Grammy record (e.g., most Grammy wins for a rapper)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As a result of the 2026 Grammys, {artist_name} achieved a historic record: {historic}.",
        node=rap_historic_leaf,
        sources=refs,
        additional_instruction="Verify explicit statements of records or 'firsts' connected to the 2026 awards.",
    )


async def verify_best_pop_vocal_album(
    evaluator: Evaluator,
    parent_node,
    info: Optional[BestPopVocalAlbumInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Best_Pop_Vocal_Album_Winner",
        desc="Information about the Best Pop Vocal Album winner at the 2026 Grammy Awards",
        parent=parent_node,
        critical=False,
    )

    refs = info.reference_urls if info else []
    refs_present = evaluator.add_custom_node(
        result=bool(refs),
        id="pop_reference_url",
        desc="A URL that verifies the Best Pop Vocal Album winner information is provided",
        parent=node,
        critical=True,
    )

    album_title = (info.album_title if info else None) or ""
    artist_name = (info.artist_name if info else None) or ""
    total_nominees = (info.total_nominees_in_category if info else None) or ""

    # Album Title (critical)
    pop_album_leaf = evaluator.add_leaf(
        id="pop_album_title",
        desc="The title of the winning pop vocal album",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Best Pop Vocal Album winner at the 68th Annual Grammy Awards (2026) was '{album_title}'.",
        node=pop_album_leaf,
        sources=refs,
        additional_instruction="Verify the category winner from official sources or reputable coverage.",
    )

    # Artist Name (critical)
    pop_artist_leaf = evaluator.add_leaf(
        id="pop_artist_name",
        desc="The name of the artist who won Best Pop Vocal Album",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The artist who won Best Pop Vocal Album at the 68th Annual Grammy Awards (2026) was {artist_name}.",
        node=pop_artist_leaf,
        sources=refs,
        additional_instruction="Confirm the winner's identity as listed for the category.",
    )

    # Total nominees (non-critical)
    pop_nominees_leaf = evaluator.add_leaf(
        id="pop_total_nominees",
        desc="The total number of nominees in the Best Pop Vocal Album category",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"There were {total_nominees} nominees in the Best Pop Vocal Album category at the 68th Annual Grammy Awards.",
        node=pop_nominees_leaf,
        sources=refs,
        additional_instruction="Count should match official nominee list size for this category in 2026.",
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
    """
    Evaluate an answer for Grammy 2026 major winners across four categories.
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_winners(),
        template_class=WinnersExtraction,
        extraction_name="grammy_2026_winners",
    )

    # Add top-level node for organization (optional)
    winners_root = evaluator.add_parallel(
        id="Grammy_2026_Major_Winners",
        desc="Comprehensive information about Grammy 2026 winners in four major categories",
        parent=root,
        critical=False,
    )

    # Build and verify each category sub-tree
    await verify_album_of_the_year(evaluator, winners_root, extraction.album_of_the_year)
    await verify_best_new_artist(evaluator, winners_root, extraction.best_new_artist)
    await verify_best_rap_album(evaluator, winners_root, extraction.best_rap_album)
    await verify_best_pop_vocal_album(evaluator, winners_root, extraction.best_pop_vocal_album)

    return evaluator.get_summary()