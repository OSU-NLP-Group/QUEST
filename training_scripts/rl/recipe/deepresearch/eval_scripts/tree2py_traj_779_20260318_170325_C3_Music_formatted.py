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
TASK_ID = "willie_2025_texas_songwriter"
TASK_DESCRIPTION = """
In 2025, Willie Nelson released an album featuring exclusively songs written by a specific Texas-born songwriter. Identify this album and provide the following information:

1. The title of Willie Nelson's 2025 album and its release date
2. The full name of the songwriter whose songs comprise this album
3. The name of the Texas hometown where this songwriter was raised
4. The Texas county in which this hometown is located
5. The name of the legendary female country artist whose backing band this songwriter joined as a member in the 1970s, and the name of that band

For each piece of information, provide a reference URL that supports your answer.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NelsonAlbumSongwriterExtraction(BaseModel):
    # Album
    album_title: Optional[str] = None
    album_release_date: Optional[str] = None
    album_urls: List[str] = Field(default_factory=list)

    # Songwriter
    songwriter_name: Optional[str] = None
    songwriter_urls: List[str] = Field(default_factory=list)

    # Hometown (Texas)
    hometown: Optional[str] = None
    hometown_urls: List[str] = Field(default_factory=list)

    # County (Texas)
    county: Optional[str] = None
    county_urls: List[str] = Field(default_factory=list)

    # Collaboration (1970s)
    collaborating_artist: Optional[str] = None
    band_name: Optional[str] = None
    collaboration_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_willie_2025_album_and_songwriter() -> str:
    return """
    Extract the specific information below from the provided answer text. Do NOT invent any information; only extract what is explicitly present in the answer. 
    If an item is missing, set the field to null (for strings) or an empty list (for URL lists).

    Your goal is to capture details about Willie Nelson's 2025 album that features exclusively songs by a single Texas-born songwriter, and biographical details about that songwriter.

    Required fields (return a single JSON object with exactly these keys):
    - album_title: The title of Willie Nelson's album released in 2025 that exclusively features songs by one songwriter.
    - album_release_date: The release date of that album as written in the answer (free-form string is fine).
    - album_urls: An array of all URLs in the answer that support the album identification and/or its release date (e.g., label page, press release, reputable news, Wikipedia, streaming listings).

    - songwriter_name: The full name of the specific Texas-born songwriter whose songs comprise the ENTIRE album.
    - songwriter_urls: An array of URLs in the answer that support that this songwriter wrote all the songs on this album.

    - hometown: The name of the Texas town (city/town/community) where the songwriter was raised (not necessarily the birthplace).
    - hometown_urls: An array of URLs that support the hometown identification.

    - county: The name of the Texas county where that hometown is located (e.g., "Travis County").
    - county_urls: An array of URLs that support the county location of the hometown.

    - collaborating_artist: The name of the legendary female country artist whose backing band the songwriter joined in the 1970s.
    - band_name: The name of that backing band.
    - collaboration_urls: An array of URLs that support the songwriter's membership in that band in the 1970s.

    URL extraction rules:
    - Extract only real URLs present in the answer (plain URLs or those inside markdown links).
    - If a URL is missing a scheme, prepend "http://".
    - Do not fabricate URLs.

    If multiple albums are mentioned, select the one that:
    1) is a Willie Nelson album released in 2025, and 
    2) consists exclusively of songs written by a single songwriter.
    If no such album is clearly identified in the answer, set fields to null/[] accordingly.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty_str(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip() != "")


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_album_subtree(evaluator: Evaluator, parent_node, data: NelsonAlbumSongwriterExtraction) -> None:
    album_node = evaluator.add_sequential(
        id="album_discovery",
        desc="Identify Willie Nelson's 2025 album that exclusively features songs by one songwriter",
        parent=parent_node,
        critical=False
    )

    # Basic info (critical)
    album_basic = evaluator.add_parallel(
        id="album_basic_info",
        desc="Provide the album's title and release information",
        parent=album_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.album_title),
        id="album_title",
        desc="Provide the title of Willie Nelson's 2025 album",
        parent=album_basic,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.album_release_date),
        id="album_release_date",
        desc="Provide the release date of the album",
        parent=album_basic,
        critical=True
    )

    # Album verification (critical)
    album_verif = evaluator.add_parallel(
        id="album_verification",
        desc="Provide reference URL supporting the album identification",
        parent=album_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.album_urls),
        id="album_urls_provided",
        desc="At least one reference URL is provided for the album details",
        parent=album_verif,
        critical=True
    )

    ref_album_leaf = evaluator.add_leaf(
        id="reference_url_album",
        desc="Provide at least one valid reference URL confirming the album details",
        parent=album_verif,
        critical=True
    )

    album_title = data.album_title or ""
    release_date = data.album_release_date or ""
    claim_album = (
        f"There exists a Willie Nelson album released in 2025 titled '{album_title}', "
        f"with a release date around '{release_date}'."
    )

    await evaluator.verify(
        claim=claim_album,
        node=ref_album_leaf,
        sources=data.album_urls,
        additional_instruction=(
            "Confirm that the provided page(s) indicate a Willie Nelson album with the specified title, "
            "released in 2025. Minor formatting differences in the date are acceptable as long as the "
            "release year is 2025 and the page clearly supports the album identification and release timing."
        )
    )


async def build_songwriter_subtree(evaluator: Evaluator, parent_node, data: NelsonAlbumSongwriterExtraction) -> None:
    songwriter_node = evaluator.add_sequential(
        id="songwriter_discovery",
        desc="Identify the songwriter whose songs comprise the album",
        parent=parent_node,
        critical=False
    )

    # Songwriter identity (critical)
    sw_identity = evaluator.add_parallel(
        id="songwriter_identity",
        desc="Provide the full name of the songwriter",
        parent=songwriter_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.songwriter_name),
        id="songwriter_name",
        desc="Provide the full name of the Texas-born songwriter who wrote all songs on the album",
        parent=sw_identity,
        critical=True
    )

    # Songwriter verification (critical)
    sw_verif = evaluator.add_parallel(
        id="songwriter_verification",
        desc="Provide reference URL confirming the songwriter",
        parent=songwriter_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.songwriter_urls) or _has_urls(data.album_urls),
        id="songwriter_urls_provided",
        desc="At least one reference URL is provided confirming the album's songwriter authorship",
        parent=sw_verif,
        critical=True
    )

    ref_sw_leaf = evaluator.add_leaf(
        id="reference_url_songwriter",
        desc="Provide at least one valid reference URL confirming this songwriter wrote the songs on this album",
        parent=sw_verif,
        critical=True
    )

    songwriter_name = data.songwriter_name or ""
    album_title = data.album_title or ""
    claim_sw = (
        f"All tracks on Willie Nelson's album '{album_title}' were written by {songwriter_name} "
        f"(i.e., the album consists exclusively of songs by this songwriter)."
    )
    sw_urls = (data.songwriter_urls or []) + (data.album_urls or [])

    await evaluator.verify(
        claim=claim_sw,
        node=ref_sw_leaf,
        sources=sw_urls,
        additional_instruction=(
            "Look for language such as 'all songs written by', 'entirely written by', "
            "'composed by (all tracks)', or equivalent statements confirming exclusive authorship."
        )
    )


async def build_hometown_subtree(evaluator: Evaluator, parent_node, data: NelsonAlbumSongwriterExtraction) -> None:
    hometown_node = evaluator.add_sequential(
        id="hometown_discovery",
        desc="Identify the Texas hometown where the songwriter was raised",
        parent=parent_node,
        critical=False
    )

    # Hometown identity (critical)
    ht_identity = evaluator.add_parallel(
        id="hometown_identity",
        desc="Provide the name of the hometown",
        parent=hometown_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.hometown),
        id="hometown_name",
        desc="Provide the name of the Texas town where the songwriter was raised",
        parent=ht_identity,
        critical=True
    )

    # Hometown verification (critical)
    ht_verif = evaluator.add_parallel(
        id="hometown_verification",
        desc="Provide reference URL confirming the hometown",
        parent=hometown_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.hometown_urls) or _has_urls(data.songwriter_urls),
        id="hometown_urls_provided",
        desc="At least one reference URL is provided confirming the songwriter's hometown",
        parent=ht_verif,
        critical=True
    )

    ref_ht_leaf = evaluator.add_leaf(
        id="reference_url_hometown",
        desc="Provide at least one valid reference URL confirming the songwriter's hometown",
        parent=ht_verif,
        critical=True
    )

    songwriter_name = data.songwriter_name or "the songwriter"
    hometown = data.hometown or ""
    claim_ht = f"{songwriter_name} was raised in {hometown}, Texas."
    ht_urls = (data.hometown_urls or []) + (data.songwriter_urls or [])

    await evaluator.verify(
        claim=claim_ht,
        node=ref_ht_leaf,
        sources=ht_urls,
        additional_instruction="Focus on 'raised in' or 'grew up in' wording; birthplace is not required for this check."
    )


async def build_county_subtree(evaluator: Evaluator, parent_node, data: NelsonAlbumSongwriterExtraction) -> None:
    county_node = evaluator.add_sequential(
        id="county_discovery",
        desc="Identify the Texas county where the hometown is located",
        parent=parent_node,
        critical=False
    )

    # County identity (critical)
    co_identity = evaluator.add_parallel(
        id="county_identity",
        desc="Provide the name of the county",
        parent=county_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.county),
        id="county_name",
        desc="Provide the name of the Texas county in which the hometown is located",
        parent=co_identity,
        critical=True
    )

    # County verification (critical)
    co_verif = evaluator.add_parallel(
        id="county_verification",
        desc="Provide reference URL confirming the county location",
        parent=county_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.county_urls) or _has_urls(data.hometown_urls),
        id="county_urls_provided",
        desc="At least one reference URL is provided confirming the county location of the hometown",
        parent=co_verif,
        critical=True
    )

    ref_co_leaf = evaluator.add_leaf(
        id="reference_url_county",
        desc="Provide at least one valid reference URL confirming the county location of the hometown",
        parent=co_verif,
        critical=True
    )

    hometown = data.hometown or ""
    county = data.county or ""
    claim_co = f"{hometown}, Texas is located in {county} County, Texas."
    co_urls = (data.county_urls or []) + (data.hometown_urls or [])

    await evaluator.verify(
        claim=claim_co,
        node=ref_co_leaf,
        sources=co_urls,
        additional_instruction="Confirm the municipality/unincorporated community lies within the specified Texas county."
    )


async def build_collaboration_subtree(evaluator: Evaluator, parent_node, data: NelsonAlbumSongwriterExtraction) -> None:
    collab_node = evaluator.add_sequential(
        id="collaboration_discovery",
        desc="Identify the songwriter's notable 1970s collaboration with a female country artist",
        parent=parent_node,
        critical=False
    )

    # Collaboration details (critical)
    collab_details = evaluator.add_parallel(
        id="collaboration_details",
        desc="Provide information about the collaboration",
        parent=collab_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.collaborating_artist),
        id="collaborating_artist",
        desc="Provide the name of the legendary female country artist whose backing band the songwriter joined in the 1970s",
        parent=collab_details,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.band_name),
        id="band_name",
        desc="Provide the name of the backing band that the songwriter was a member of",
        parent=collab_details,
        critical=True
    )

    # Collaboration verification (critical)
    collab_verif = evaluator.add_parallel(
        id="collaboration_verification",
        desc="Provide reference URL confirming the collaboration",
        parent=collab_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.collaboration_urls) or _has_urls(data.songwriter_urls),
        id="collaboration_urls_provided",
        desc="At least one reference URL is provided confirming the songwriter's membership in the specified band",
        parent=collab_verif,
        critical=True
    )

    ref_collab_leaf = evaluator.add_leaf(
        id="reference_url_collaboration",
        desc="Provide at least one valid reference URL confirming the songwriter's membership in the specified band",
        parent=collab_verif,
        critical=True
    )

    songwriter_name = data.songwriter_name or "the songwriter"
    artist = data.collaborating_artist or ""
    band = data.band_name or ""
    claim_collab = (
        f"{songwriter_name} was a member of the backing band '{band}' of the legendary female country artist '{artist}' "
        f"in the 1970s."
    )
    collab_urls = (data.collaboration_urls or []) + (data.songwriter_urls or [])

    await evaluator.verify(
        claim=claim_collab,
        node=ref_collab_leaf,
        sources=collab_urls,
        additional_instruction="Confirm band membership with this artist; timeframe should be in the 1970s (approximate is acceptable if clearly indicated)."
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
    Evaluate an answer for the Willie Nelson 2025 album + Texas songwriter task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,   # Follow the problem's sequential discovery
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_willie_2025_album_and_songwriter(),
        template_class=NelsonAlbumSongwriterExtraction,
        extraction_name="willie_2025_album_and_songwriter"
    )

    # Build verification subtrees (order matters due to sequential root)
    await build_album_subtree(evaluator, root, extracted)
    await build_songwriter_subtree(evaluator, root, extracted)
    await build_hometown_subtree(evaluator, root, extracted)
    await build_county_subtree(evaluator, root, extracted)
    await build_collaboration_subtree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()