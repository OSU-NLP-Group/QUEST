import asyncio
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cure_perry_bamonte_albums_1990_2005"
TASK_DESCRIPTION = (
    "Perry Bamonte, the late guitarist and keyboardist for The Cure, was a full member of the band from 1990 to 2005, "
    "contributing guitar, keyboards, and six-string bass to several of their albums. Identify all studio albums The Cure "
    "released during this period that Perry Bamonte played on as a full band member. For each album, provide: the complete "
    "album title, the release year, and a reference URL to an official or reliable source (such as The Cure's official website, "
    "Wikipedia, or a major music database) that confirms Perry Bamonte's participation on that album. Note: Only include studio "
    "albums, not live albums or compilations (unless the compilation consists primarily of new studio recordings)."
)

EXPECTED_ALBUMS = [
    {"title": "Wish", "year": "1992"},
    {"title": "Wild Mood Swings", "year": "1996"},
    {"title": "Bloodflowers", "year": "2000"},
    {"title": "Acoustic Hits", "year": "2001"},
    {"title": "The Cure", "year": "2004"},
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlbumItem(BaseModel):
    title: Optional[str] = None
    release_year: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class AlbumExtraction(BaseModel):
    albums: List[AlbumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_albums() -> str:
    return """
    Extract all album entries that the answer provides for The Cure albums during 1990–2005 on which Perry Bamonte is said to have participated as a full band member.
    For each listed album in the answer, extract:
    - title: The album title exactly as written in the answer (string).
    - release_year: The release year the answer associates with that album (string; keep as-is, do not convert to number).
    - source_urls: All URLs cited in the answer that support Perry Bamonte's participation in that specific album (array of URLs). Include official or reliable sources if present (e.g., thecure.com, Wikipedia, Discogs, AllMusic, MusicBrainz, etc.). Only extract URLs explicitly present in the answer.

    Return a JSON object with:
    {
      "albums": [
        {"title": "...", "release_year": "...", "source_urls": ["...", "..."]},
        ...
      ]
    }

    Rules:
    - Only extract albums that the answer explicitly lists for this task.
    - Do not invent any URLs or years; if a field is missing, set it to null or [] for source_urls.
    - If the URL appears without a scheme, prepend http:// to make it a valid URL.
    - If multiple URLs are present for one album, include them all in source_urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)  # remove any parentheses content
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize_title(s))


def find_best_album_match(extracted: AlbumExtraction, expected_title: str) -> Optional[AlbumItem]:
    """Find the most likely matching album item from the extracted list for the expected title."""
    if not extracted or not extracted.albums:
        return None

    target = _normalize_title(expected_title)
    best_item: Optional[AlbumItem] = None
    best_score = 0.0

    # Prefer direct/substring matches; otherwise use similarity ratio
    for item in extracted.albums:
        if not item.title:
            continue
        cand = _normalize_title(item.title)
        if not cand:
            continue

        # Direct or substring match gets high score
        if cand == target or target in cand or cand in target:
            score = 1.0
        else:
            score = SequenceMatcher(None, cand, target).ratio()

        if score > best_score:
            best_score = score
            best_item = item

    # Threshold to avoid spurious matches
    if best_score >= 0.55:
        return best_item
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_expected_album(
    evaluator: Evaluator,
    parent_node,
    expected_title: str,
    expected_year: str,
    extracted_albums: AlbumExtraction,
) -> None:
    """
    Build verification nodes for a single expected album:
    - Existence + URL presence
    - Title match (logical)
    - Year provided (existence)
    - Year matches expected (logical)
    - Year supported by sources (URL-grounded)
    - Participation supported by sources (URL-grounded)
    """
    slug = _slugify(expected_title)
    album_node = evaluator.add_sequential(
        id=f"album_{slug}",
        desc=f"Provides the album {expected_title} with its correct release year ({expected_year}) and a reference URL",
        parent=parent_node,
        critical=False,
    )

    matched = find_best_album_match(extracted_albums, expected_title)
    has_url = bool(matched and matched.source_urls and len(matched.source_urls) > 0)

    # 1) Existence with URL(s) (Critical to gate subsequent checks)
    evaluator.add_custom_node(
        result=bool(matched) and has_url,
        id=f"album_{slug}_present",
        desc=f"Album '{expected_title}' is included in the answer with at least one reference URL",
        parent=album_node,
        critical=True,
    )

    # Group all detailed checks in parallel after existence
    checks_node = evaluator.add_parallel(
        id=f"album_{slug}_checks",
        desc=f"Detailed checks for '{expected_title}'",
        parent=album_node,
        critical=False,
    )

    provided_title = matched.title if matched and matched.title else ""
    provided_year = matched.release_year if matched and matched.release_year else ""
    urls = matched.source_urls if matched and matched.source_urls else []

    # 2) Title match (Critical)
    title_match_leaf = evaluator.add_leaf(
        id=f"album_{slug}_title_match",
        desc=f"Provided album title corresponds to '{expected_title}' by The Cure",
        parent=checks_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided album title '{provided_title}' refers to the same album as '{expected_title}' by The Cure.",
        node=title_match_leaf,
        additional_instruction=(
            "Allow minor variations like casing, punctuation, or parenthetical notes (e.g., remaster info). "
            "If the provided title clearly refers to The Cure's album with the expected title, consider it a match."
        ),
    )

    # 3) Year provided (Critical)
    year_provided_leaf = evaluator.add_custom_node(
        result=bool(provided_year and provided_year.strip()),
        id=f"album_{slug}_year_provided",
        desc=f"Release year is provided in the answer for '{expected_title}'",
        parent=checks_node,
        critical=True,
    )

    # 4) Year equals expected (Critical; logical consistency with the answer)
    year_match_leaf = evaluator.add_leaf(
        id=f"album_{slug}_year_match",
        desc=f"Provided release year matches the expected original release year ({expected_year}) for '{expected_title}'",
        parent=checks_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The answer claims the release year for '{expected_title}' is '{provided_year}', "
            f"which matches the correct original release year '{expected_year}'."
        ),
        node=year_match_leaf,
        additional_instruction=(
            "Judge logically whether the provided year equals the expected year. "
            "If the provided year is missing or different, this should fail."
        ),
    )

    # 5) Year supported by cited sources (Critical; URL-grounded)
    year_supported_leaf = evaluator.add_leaf(
        id=f"album_{slug}_year_supported",
        desc=f"The album '{expected_title}' has original release year {expected_year}, supported by the cited sources",
        parent=checks_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The album '{expected_title}' by The Cure was originally released in {expected_year}.",
        node=year_supported_leaf,
        sources=urls,
        additional_instruction=(
            "Use the cited page(s) to confirm the album's original release year. "
            "Ignore reissue/remaster dates; focus on the original release year. "
            "Only the year must match (exact day/month is not required)."
        ),
    )

    # 6) Participation supported by cited sources (Critical; URL-grounded)
    participation_leaf = evaluator.add_leaf(
        id=f"album_{slug}_perry_supported",
        desc=f"Perry Bamonte's participation on '{expected_title}' is supported by cited sources",
        parent=checks_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Perry Bamonte is credited as a band member (e.g., guitar, keyboards, six-string bass) on The Cure's album '{expected_title}'."
        ),
        node=participation_leaf,
        sources=urls,
        additional_instruction=(
            "Check the page's credits/personnel section for 'Perry Bamonte'. "
            "Accept roles like guitar, keyboards, six-string bass, baritone guitar, or similar. "
            "Explicit band membership credit also suffices."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator and root
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

    # Container node mirroring the rubric section
    albums_root = evaluator.add_parallel(
        id="albums_root",
        desc="Identifies all studio albums The Cure released that Perry Bamonte contributed to as a full member of the band",
        parent=root,
        critical=False,
    )

    # Extract all album entries from the answer
    extracted_albums = await evaluator.extract(
        prompt=prompt_extract_albums(),
        template_class=AlbumExtraction,
        extraction_name="album_list",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected_albums": EXPECTED_ALBUMS,
            "period": "1990–2005",
            "note": "Only studio albums included; 'Acoustic Hits' consists of newly recorded acoustic versions and is accepted here."
        },
        gt_type="ground_truth_expected_albums",
    )

    # Build verification for each expected album
    for item in EXPECTED_ALBUMS:
        await verify_expected_album(
            evaluator=evaluator,
            parent_node=albums_root,
            expected_title=item["title"],
            expected_year=item["year"],
            extracted_albums=extracted_albums,
        )

    # Return evaluation summary
    return evaluator.get_summary()