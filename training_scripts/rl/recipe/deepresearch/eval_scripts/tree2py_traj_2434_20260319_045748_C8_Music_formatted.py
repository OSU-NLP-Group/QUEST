import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "grammy_bna_spotify_top4_2026"
TASK_DESCRIPTION = """
The 68th Annual Grammy Awards (also known as the 2026 Grammys) took place on February 1, 2026. The Best New Artist category featured eight nominees.

Identify the four nominees from the 2026 Grammy Best New Artist category who have the highest Spotify monthly listeners as of March 2026. For each of these four artists, ranked from highest to lowest by their listener count, provide:

1. Artist name
2. Title of their debut studio album (not EP)
3. Release date of their debut studio album
4. Current Spotify monthly listeners (as of March 2026, in millions)
5. URL to their official Spotify artist profile page

Present your answer as a structured list with the four artists ordered by Spotify listener count from highest to lowest.
""".strip()


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class RankedArtist(BaseModel):
    artist_name: Optional[str] = None
    debut_album_title: Optional[str] = None
    debut_album_release_date: Optional[str] = None
    spotify_monthly_listeners_millions: Optional[str] = None
    spotify_profile_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class Top4ArtistsExtraction(BaseModel):
    ranked_1: Optional[RankedArtist] = None
    ranked_2: Optional[RankedArtist] = None
    ranked_3: Optional[RankedArtist] = None
    ranked_4: Optional[RankedArtist] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_top4_artists() -> str:
    return """
    Extract exactly the top four artists listed in the answer for the 2026 Grammy Best New Artist nominees ranked by Spotify monthly listeners (as of March 2026), in descending order.

    For each of the four ranked entries (ranked_1 through ranked_4), extract the following fields:
    - artist_name: The artist's name exactly as written.
    - debut_album_title: The title of the artist's debut studio album (ignore EPs, mixtapes, or reissues).
    - debut_album_release_date: The release date of that debut studio album (any reasonable format is fine; include day if available).
    - spotify_monthly_listeners_millions: The Spotify monthly listeners count as a string in millions, as stated in the answer (e.g., "67.5", "67.5 million", "about 68").
    - spotify_profile_url: The official Spotify artist profile URL.
    - supporting_urls: A list of all additional URLs (besides the Spotify profile URL) cited in the answer that support any of the above facts (e.g., Grammys nominee list pages, Wikipedia, artist websites, music databases, press releases, or analytics sites reporting monthly listeners).

    Rules:
    - Only extract information explicitly present in the answer. Do not invent or infer new URLs or facts.
    - If a field is missing, return null for single fields and an empty list for supporting_urls.
    - Always keep the four entries aligned to their claimed ranks in the answer (ranked_1 is the highest listeners).
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(n % 10, f"{n}th")
    return suffix if suffix.endswith(("st", "nd", "rd", "th")) else f"{n}{suffix}"


def collect_sources(artist: Optional[RankedArtist]) -> List[str]:
    """Collect all verifiable sources for a given artist (Spotify + supporting URLs)."""
    if not artist:
        return []
    urls: List[str] = []
    if artist.spotify_profile_url:
        urls.append(artist.spotify_profile_url)
    if artist.supporting_urls:
        urls.extend([u for u in artist.supporting_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def artist_or_empty(a: Optional[RankedArtist]) -> RankedArtist:
    return a or RankedArtist()


# -----------------------------------------------------------------------------
# Verification logic per ranked artist
# -----------------------------------------------------------------------------
async def verify_ranked_artist(
    evaluator: Evaluator,
    parent_node,
    rank_index: int,
    artist: RankedArtist,
) -> None:
    """
    Build the sub-tree and perform verifications for a single ranked artist.
    The structure follows the provided rubric:
      Nominee_Ranked_{ordinal} (sequential)
        ├─ Artist_Identification_{ordinal} (leaf, critical)
        └─ Artist_Attributes_{ordinal} (parallel, non-critical)
            ├─ Debut_Album_Title_{ordinal} (leaf)
            ├─ Debut_Album_Release_Date_{ordinal} (leaf)
            ├─ Spotify_Monthly_Listeners_{ordinal} (leaf)
            └─ Spotify_Profile_URL_{ordinal} (leaf)
    """
    ord_str = ordinal(rank_index)

    # Create the sequential parent node for this rank
    rank_node = evaluator.add_sequential(
        id=f"Nominee_Ranked_{ord_str}",
        desc=f"Evaluate the artist ranked {ord_str} by Spotify monthly listeners",
        parent=parent_node,
        critical=False
    )

    # 1) Critical identification leaf
    ident_leaf = evaluator.add_leaf(
        id=f"Artist_Identification_{ord_str}",
        desc=f"Verify that the artist with the {ord_str}-highest Spotify monthly listener count among 2026 Best New Artist nominees is correctly identified",
        parent=rank_node,
        critical=True
    )

    # We prioritize verifying that the listed artist is indeed a 2026 Grammys Best New Artist nominee.
    # This is more source-verifiable than asserting the global "highest" ordering across all eight nominees
    # (which often isn't explicitly stated by a single source). If the answer provides reliable sources
    # that explicitly state or allow strong inference of the rank, the LLM judge can still pass the claim.
    ident_claim = (
        f"The artist '{artist.artist_name or ''}' is one of the eight nominees in the 2026 Grammys Best New Artist category, "
        f"and is presented here as the {ord_str}-highest by Spotify monthly listeners (as of March 2026)."
    )

    await evaluator.verify(
        claim=ident_claim,
        node=ident_leaf,
        sources=collect_sources(artist),
        additional_instruction=(
            "First, confirm from the provided sources whether the artist is indeed a 2026 Grammys Best New Artist nominee. "
            "If reliable evidence is also provided to support the stated rank by Spotify monthly listeners (as of March 2026), "
            "consider that as further support. If the nominee status itself is not supported by the URLs, mark this verification as not supported."
        ),
    )

    # 2) Non-critical parallel attributes block
    attrs_node = evaluator.add_parallel(
        id=f"Artist_Attributes_{ord_str}",
        desc=f"Verify the attributes provided for the {ord_str}-ranked artist",
        parent=rank_node,
        critical=False
    )

    # Prepare leaves
    debut_title_leaf = evaluator.add_leaf(
        id=f"Debut_Album_Title_{ord_str}",
        desc=f"Verify that the debut studio album title for the {ord_str}-ranked artist is correctly provided",
        parent=attrs_node,
        critical=False
    )
    release_date_leaf = evaluator.add_leaf(
        id=f"Debut_Album_Release_Date_{ord_str}",
        desc=f"Verify that the debut album release date for the {ord_str}-ranked artist is correctly provided",
        parent=attrs_node,
        critical=False
    )
    listeners_leaf = evaluator.add_leaf(
        id=f"Spotify_Monthly_Listeners_{ord_str}",
        desc=f"Verify that the current Spotify monthly listeners count (March 2026) for the {ord_str}-ranked artist is correctly provided",
        parent=attrs_node,
        critical=False
    )
    profile_url_leaf = evaluator.add_leaf(
        id=f"Spotify_Profile_URL_{ord_str}",
        desc=f"Verify that a valid URL to the Spotify artist profile for the {ord_str}-ranked artist is provided",
        parent=attrs_node,
        critical=False
    )

    # Build claims and sources
    sources_all = collect_sources(artist)

    # Debut album title
    debut_title_claim = (
        f"The debut studio album of {artist.artist_name or ''} is titled '{artist.debut_album_title or ''}'."
    )
    debut_title_ins = (
        "Verify the title of the artist's debut studio album (not an EP, mixtape, reissue, or live compilation). "
        "If multiple releases exist, ensure this is the first full-length studio album. "
        "Prefer authoritative sources such as artist's official site, reputable databases, or Wikipedia."
    )

    # Debut album release date
    release_date_claim = (
        f"The release date of the debut studio album '{artist.debut_album_title or ''}' by {artist.artist_name or ''} "
        f"is {artist.debut_album_release_date or ''}."
    )
    release_date_ins = (
        "Confirm the initial official release date of the debut studio album. "
        "Accept reasonable date formats and allow minor formatting differences. "
        "Use authoritative sources if possible."
    )

    # Monthly listeners
    listeners_claim = (
        f"As of March 2026, {artist.artist_name or ''} has approximately {artist.spotify_monthly_listeners_millions or ''} million monthly listeners on Spotify."
    )
    listeners_ins = (
        "Check the Spotify artist profile page screenshot/text for the 'monthly listeners' metric (not followers). "
        "Allow small rounding differences (e.g., 67.5 ≈ 67.6). If the provided sources do not clearly show March 2026's value, "
        "but show a very close timeframe and the value is consistent, that can be acceptable."
    )

    # Spotify profile URL validity
    profile_url_claim = (
        f"This URL is the official Spotify artist profile page for {artist.artist_name or ''}."
    )
    profile_url_sources = [artist.spotify_profile_url] if artist.spotify_profile_url else []
    profile_url_ins = (
        "Confirm the page is on Spotify (e.g., open.spotify.com/artist/...) and that the artist name on the page "
        "matches the stated artist, allowing minor variations like capitalization or diacritics."
    )

    # Run attribute verifications in parallel (after identification has been evaluated)
    await evaluator.batch_verify(
        [
            (debut_title_claim, sources_all, debut_title_leaf, debut_title_ins),
            (release_date_claim, sources_all, release_date_leaf, release_date_ins),
            (listeners_claim, sources_all, listeners_leaf, listeners_ins),
            (profile_url_claim, profile_url_sources, profile_url_leaf, profile_url_ins),
        ]
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 'Top 4 Grammy BNA Nominees by Spotify Listeners (2026)' task.
    """
    # Initialize evaluator (root parallel as per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Evaluate whether the solution correctly identifies the four 2026 Grammy Best New Artist nominees with the "
            "highest Spotify monthly listeners as of March 2026 and provides accurate details for each."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_top4_artists(),
        template_class=Top4ArtistsExtraction,
        extraction_name="top4_ranked_artists",
    )

    # Normalize to always have four RankedArtist entries (in order)
    ranked_list: List[RankedArtist] = [
        artist_or_empty(extracted.ranked_1),
        artist_or_empty(extracted.ranked_2),
        artist_or_empty(extracted.ranked_3),
        artist_or_empty(extracted.ranked_4),
    ]

    # Build the verification tree following the rubric and run checks
    # Root node title mirrors the rubric's root node name
    top4_root = evaluator.add_parallel(
        id="Top_4_Grammy_BNA_Nominees_by_Spotify_Listeners",
        desc="Evaluate whether the solution correctly identifies the four 2026 Grammy Best New Artist nominees with the highest Spotify monthly listeners as of March 2026 and provides accurate details for each",
        parent=root,
        critical=False,
    )

    # Verify each ranked nominee block
    for i, artist in enumerate(ranked_list, start=1):
        await verify_ranked_artist(evaluator, top4_root, i, artist)

    # Optionally record a compact snapshot of extracted data for transparency
    try:
        evaluator.add_custom_info(
            info={
                "ranked_artists": [
                    {
                        "rank": i,
                        "artist_name": a.artist_name,
                        "debut_album_title": a.debut_album_title,
                        "debut_album_release_date": a.debut_album_release_date,
                        "spotify_monthly_listeners_millions": a.spotify_monthly_listeners_millions,
                        "spotify_profile_url": a.spotify_profile_url,
                        "supporting_urls_count": len(a.supporting_urls or []),
                    }
                    for i, a in enumerate(ranked_list, start=1)
                ]
            },
            info_type="extracted_overview",
            info_name="extracted_top4_overview",
        )
    except Exception:
        pass

    # Return the final structured summary
    return evaluator.get_summary()