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
TASK_ID = "grammy_poty_2025"
TASK_DESCRIPTION = (
    "A music industry talent agency is conducting research on the 2025 Grammy Awards Producer of the Year (Non-Classical) nominees to identify potential collaboration opportunities for their artist roster. "
    "The agency needs comprehensive profiles of all five producers nominated in this category.\n\n"
    "For each of the five Producer of the Year (Non-Classical) nominees at the 2025 Grammy Awards, provide:\n\n"
    "1. The producer's professional name (as listed in the nomination)\n"
    "2. At least three different artists the producer worked with in 2024-2025\n"
    "3. At least three specific songs, albums, or projects the producer worked on that were released or gained recognition in 2024-2025\n"
    "4. The primary music genres or styles the producer specializes in\n"
    "5. Any notable achievements, awards, or recognition the producer received prior to or including 2025\n"
    "6. A reference URL confirming the producer's nomination for the 2025 Grammy Producer of the Year award\n\n"
    "All information must be verifiable through publicly available sources."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ProducerProfile(BaseModel):
    professional_name: Optional[str] = None
    nomination_url: Optional[str] = None
    artists: List[str] = Field(default_factory=list)
    works: List[str] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class ProducersExtraction(BaseModel):
    producers: List[ProducerProfile] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_producers() -> str:
    return """
    Extract up to five Producer of the Year (Non-Classical) nominees for the 2025 Grammy Awards as they appear in the provided answer text. If the answer includes more than five, only extract the first five mentioned. If fewer than five are present, still return a list with the available ones.

    For each nominee, extract the following fields:
    - professional_name: The producer’s professional name exactly as written in the answer (aim to match the nomination listing).
    - nomination_url: A URL explicitly included in the answer that confirms this person’s nomination for the 2025 Grammy Producer of the Year (Non-Classical). If multiple are listed, choose the single best/most relevant one; if none is present in the answer text, return null.
    - artists: A list of at least three different artists the producer worked with in 2024 or 2025 as stated in the answer. If fewer are provided in the answer, include the ones given. Do not invent any.
    - works: A list of at least three specific songs, albums, or projects the producer worked on that were released or recognized in 2024–2025 as stated in the answer. If fewer are provided, include the ones given. Do not invent any.
    - genres: A list of the primary music genres or styles the producer specializes in, as stated in the answer.
    - achievements: A list of notable achievements, awards, or recognition the producer received prior to or including 2025, as stated in the answer.
    - source_urls: A list of all additional URLs (besides the nomination URL) that the answer cites for this producer’s profile (e.g., interviews, credits pages, news articles, label pages, streaming or chart pages). Only include URLs explicitly present in the answer text. Do not invent any URLs.

    Important rules for URL extraction:
    - Only extract URLs that are explicitly present in the answer text. Do not create or infer any URLs.
    - Include full URLs. If a URL is missing the protocol, prepend http://.
    - nomination_url must be a single URL. source_urls is a list for any other supporting links tied to this producer in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_clean(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for it in items or []:
        if not it:
            continue
        val = it.strip()
        low = val.lower()
        if val and low not in seen:
            seen.add(low)
            result.append(val)
    return result


def _top_k(items: List[str], k: int) -> List[str]:
    return items[:k] if items else []


def _combine_sources(prod: ProducerProfile) -> List[str]:
    urls: List[str] = []
    if prod.nomination_url and prod.nomination_url.strip():
        urls.append(prod.nomination_url.strip())
    for u in prod.source_urls or []:
        if u and u.strip():
            urls.append(u.strip())
    # Dedup while preserving order
    return _unique_clean(urls)


# --------------------------------------------------------------------------- #
# Verification sub-tree for one producer                                      #
# --------------------------------------------------------------------------- #
async def verify_single_producer(
    evaluator: Evaluator,
    parent_node,
    producer: ProducerProfile,
    index: int,
) -> None:
    """
    Build verification nodes for one producer (nominee #index+1).
    The section-level nodes mirror the rubric's child names while splitting
    into atomic leaf checks under each section.
    """
    # Normalize fields
    name = (producer.professional_name or "").strip()
    artists = _unique_clean(producer.artists)
    works = _unique_clean(producer.works)
    genres = _unique_clean(producer.genres)
    achievements = _unique_clean(producer.achievements)
    nomination_url = (producer.nomination_url or "").strip()
    combined_sources = _combine_sources(producer)

    # Producer profile (parallel aggregator)
    producer_node = evaluator.add_parallel(
        id=f"producer_{index+1}_profile",
        desc=f"Profile for nominee #{index + 1} (one of the five nominees).",
        parent=parent_node,
        critical=False
    )

    # 1) Professional_Name (sequential): existence -> matches nomination page
    name_main = evaluator.add_sequential(
        id=f"producer_{index+1}_Professional_Name",
        desc="Provide the producer's professional name exactly as listed in the nomination.",
        parent=producer_node,
        critical=True
    )

    # 1.a existence
    evaluator.add_custom_node(
        result=bool(name),
        id=f"producer_{index+1}_name_provided",
        desc="Professional name is provided in the answer.",
        parent=name_main,
        critical=True
    )

    # 1.b matches on nomination page
    name_match_leaf = evaluator.add_leaf(
        id=f"producer_{index+1}_name_matches_nomination",
        desc=f"Nomination page lists the nominee as '{name}' in the 2025 Producer of the Year (Non-Classical) category.",
        parent=name_main,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"On the provided nomination page, the nominee for 'Producer of the Year, Non-Classical' in 2025 is '{name}', "
            f"or '{name}' is clearly listed among nominees in that category."
        ),
        node=name_match_leaf,
        sources=nomination_url,
        additional_instruction=(
            "Accept minor variations in capitalization or punctuation in the name. "
            "Ensure the page refers to the 2025 Grammys and the 'Producer of the Year, Non-Classical' category."
        ),
    )

    # 2) Nomination_Reference_URL (sequential): url provided -> confirms nomination
    nom_main = evaluator.add_sequential(
        id=f"producer_{index+1}_Nomination_Reference_URL",
        desc="Provide a reference URL that confirms this producer's nomination for the 2025 Grammy Award for Producer of the Year (Non-Classical).",
        parent=producer_node,
        critical=True
    )

    # 2.a URL provided
    evaluator.add_custom_node(
        result=bool(nomination_url),
        id=f"producer_{index+1}_nomination_url_provided",
        desc="Nomination reference URL is provided.",
        parent=nom_main,
        critical=True
    )

    # 2.b Confirmation on the URL
    nomination_confirm_leaf = evaluator.add_leaf(
        id=f"producer_{index+1}_nomination_confirmed",
        desc="Nomination URL confirms the 2025 Producer of the Year (Non-Classical) nomination for this producer.",
        parent=nom_main,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided page confirms that {name} is a nominee for the 2025 Grammy 'Producer of the Year, Non-Classical'."
        ),
        node=nomination_confirm_leaf,
        sources=nomination_url,
        additional_instruction="The page should explicitly show the nomination, the category, and that it is for the 2025 Grammys."
    )

    # 3) Artists_2024_2025 (sequential): count -> sources present -> verify top 3 artist collaborations
    artists_main = evaluator.add_sequential(
        id=f"producer_{index+1}_Artists_2024_2025",
        desc="Provide at least three different artists the producer worked with during 2024–2025.",
        parent=producer_node,
        critical=True
    )
    # 3.a At least 3 artists provided
    evaluator.add_custom_node(
        result=(len(artists) >= 3),
        id=f"producer_{index+1}_artists_count_gte3",
        desc="At least three different artists are listed.",
        parent=artists_main,
        critical=True
    )
    # 3.b Sources available to support artist collaborations
    evaluator.add_custom_node(
        result=(len(_combine_sources(producer)) > 0),
        id=f"producer_{index+1}_artists_sources_available",
        desc="At least one source URL is provided to support artist collaboration claims.",
        parent=artists_main,
        critical=True
    )
    # 3.c Verify first three artists
    for j, artist in enumerate(_top_k(artists, 3)):
        leaf = evaluator.add_leaf(
            id=f"producer_{index+1}_artist_{j+1}_supported",
            desc=f"Collaboration in 2024–2025 between {name} and {artist} is supported by sources.",
            parent=artists_main,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"{name} worked with {artist} during 2024 or 2025 in a production capacity "
                f"(e.g., produced, co-produced, executive produced)."
            ),
            node=leaf,
            sources=combined_sources,
            additional_instruction=(
                "Look for credits, articles, liner notes, or official pages indicating the producer worked with the artist. "
                "The timeframe must be 2024 or 2025; allow indirect evidence such as coverage of a 2024/2025 release credit."
            )
        )

    # 4) Works_2024_2025 (sequential): count -> sources present -> verify top 3 works
    works_main = evaluator.add_sequential(
        id=f"producer_{index+1}_Works_2024_2025",
        desc="Provide at least three specific songs, albums, or projects the producer worked on that were released or gained recognition in 2024–2025.",
        parent=producer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(works) >= 3),
        id=f"producer_{index+1}_works_count_gte3",
        desc="At least three specific works from 2024–2025 are listed.",
        parent=works_main,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(combined_sources) > 0),
        id=f"producer_{index+1}_works_sources_available",
        desc="At least one source URL is provided to support work/project claims.",
        parent=works_main,
        critical=True
    )
    for j, work in enumerate(_top_k(works, 3)):
        leaf = evaluator.add_leaf(
            id=f"producer_{index+1}_work_{j+1}_supported",
            desc=f"Work '{work}' in 2024–2025 with production involvement by {name} is supported by sources.",
            parent=works_main,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"{name} worked on '{work}' in 2024 or 2025 as a producer (including co-producer/executive producer), "
                f"or the work gained significant recognition in that period with {name} credited for production."
            ),
            node=leaf,
            sources=combined_sources,
            additional_instruction=(
                "Check release dates, credit pages, press coverage, or official listings to confirm both the timeframe (2024–2025) "
                "and the producer’s involvement."
            )
        )

    # 5) Primary_Genres_or_Styles (sequential): provided -> sources available -> verify up to 2 genres
    genres_main = evaluator.add_sequential(
        id=f"producer_{index+1}_Primary_Genres_or_Styles",
        desc="Identify the primary music genres or styles the producer specializes in.",
        parent=producer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(genres) > 0),
        id=f"producer_{index+1}_genres_present",
        desc="At least one primary genre/style is listed.",
        parent=genres_main,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(combined_sources) > 0),
        id=f"producer_{index+1}_genres_sources_available",
        desc="At least one source URL is provided to support genre/style claims.",
        parent=genres_main,
        critical=True
    )
    for j, genre in enumerate(_top_k(genres, 2)):
        leaf = evaluator.add_leaf(
            id=f"producer_{index+1}_genre_{j+1}_supported",
            desc=f"Genre '{genre}' is supported by sources as a primary or typical style for {name}.",
            parent=genres_main,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"{name} is commonly associated with or primarily specializes in the '{genre}' genre/style."
            ),
            node=leaf,
            sources=combined_sources,
            additional_instruction=(
                "Accept language indicating primary, typical, or notable specialization in the genre/style. "
                "Evidence may include interviews, artist bios, label pages, or credible media coverage."
            )
        )

    # 6) Notable_Achievements_Pre_or_Through_2025 (sequential): provided -> sources -> verify up to 2 achievements
    ach_main = evaluator.add_sequential(
        id=f"producer_{index+1}_Notable_Achievements_Pre_or_Through_2025",
        desc="Provide notable achievements, awards, or recognition received prior to or including 2025.",
        parent=producer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(achievements) > 0),
        id=f"producer_{index+1}_achievements_present",
        desc="At least one notable achievement/award/recognition is listed.",
        parent=ach_main,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(combined_sources) > 0),
        id=f"producer_{index+1}_achievements_sources_available",
        desc="At least one source URL is provided to support achievement/award claims.",
        parent=ach_main,
        critical=True
    )
    for j, ach in enumerate(_top_k(achievements, 2)):
        leaf = evaluator.add_leaf(
            id=f"producer_{index+1}_achievement_{j+1}_supported",
            desc=f"Achievement/award/recognition is supported by sources: {ach}",
            parent=ach_main,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"Before or in 2025, {name} received or achieved: {ach}."
            ),
            node=leaf,
            sources=combined_sources,
            additional_instruction=(
                "Verify that the described achievement/award/recognition is accurate and occurred on or before 2025. "
                "Allow reasonable title variations and recognize widely cited industry accolades."
            )
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an answer for the 2025 Grammy Producer of the Year (Non-Classical) nominees research task.
    """
    # Initialize evaluator (root is non-critical parallel to allow partial credit across nominees)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_producers(),
        template_class=ProducersExtraction,
        extraction_name="producers_extraction",
    )

    # Normalize number of producers to exactly 5 (pad with empty if fewer)
    producers: List[ProducerProfile] = list(extracted.producers or [])
    if len(producers) > 5:
        producers = producers[:5]
    while len(producers) < 5:
        producers.append(ProducerProfile())

    # Global critical check: public verifiability presence (URLs exist)
    # Require each producer to have at least one public URL: nomination_url or any source_urls
    global_verifiable = True
    for prod in producers:
        has_any_url = bool((prod.nomination_url or "").strip()) or any((u or "").strip() for u in (prod.source_urls or []))
        if not has_any_url:
            global_verifiable = False
            break

    evaluator.add_custom_node(
        result=global_verifiable,
        id="Public_Verifiability_All_Information",
        desc="All claims provided across the entire answer are verifiable via publicly available sources (basic URL presence per producer).",
        parent=root,
        critical=True,
    )

    # Build per-producer verification trees
    verify_tasks = []
    for i in range(5):
        verify_tasks.append(verify_single_producer(evaluator, root, producers[i], i))
    # Execute in sequence to respect sequential-gated children ordering within each producer
    for t in verify_tasks:
        await t

    # Return structured result
    return evaluator.get_summary()