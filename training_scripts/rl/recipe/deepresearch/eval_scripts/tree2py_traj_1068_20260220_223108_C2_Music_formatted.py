import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammy_2026_aoty_producers"
TASK_DESCRIPTION = (
    "Identify the album that won Album of the Year at the 2026 Grammy Awards (68th Annual Grammy Awards). "
    "Then, for that album, determine which three music producers are credited on the highest number of tracks. "
    "For each of these three producers, provide: (1) Their full name or professional name, "
    "(2) The exact number of tracks on the album where they are credited as a producer, "
    "and (3) A reference URL (such as Wikipedia, Apple Music, or an official music credits source) that verifies their production credits. "
    "List the three producers in descending order based on the number of tracks they produced, "
    "starting with the producer who worked on the most tracks."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlbumWinnerExtraction(BaseModel):
    """Album of the Year winner identification from the answer."""
    album_title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProducerExtractionItem(BaseModel):
    """A single producer entry extracted from the answer."""
    name: Optional[str] = None
    track_count: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProducersExtraction(BaseModel):
    """Top producers extracted from the answer."""
    producers: List[ProducerExtractionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_aoty_album() -> str:
    return (
        "From the provided answer, extract the album explicitly stated to have won Album of the Year at the 2026 Grammy Awards "
        "(the 68th Annual Grammy Awards). Return the following fields:\n"
        "- album_title: the album title exactly as stated in the answer\n"
        "- sources: an array of URL(s) cited in the answer that directly support the Album of the Year win claim for this album "
        "(e.g., Grammy.com, Wikipedia awards page, or other credible sources). Do not invent URLs; include only URLs explicitly present in the answer.\n"
        "If the answer does not state an album as the 2026 Album of the Year, set album_title to null. "
        "If no supporting URLs are present in the answer, return an empty list for sources."
    )


def prompt_extract_top_three_producers(album_title_hint: Optional[str]) -> str:
    hint_text = f"The referenced album is: {album_title_hint}." if album_title_hint else "No album hint is available."
    return (
        f"{hint_text}\n"
        "From the provided answer, extract the list of producers that the answer claims are the top three by number of tracks credited as producer on the identified album. "
        "Return a JSON object with a 'producers' array. Each element must include:\n"
        "- name: the producer’s full or professional name as stated in the answer\n"
        "- track_count: the exact integer count of tracks on the album that the answer claims the producer is credited as producer\n"
        "- urls: an array of URL(s) cited in the answer that verify this producer’s credit information for the album "
        "(e.g., Wikipedia, Apple Music, Tidal, official credits). Include only URLs explicitly present in the answer.\n"
        "Extract producers in the same order they are presented in the answer. If more than three producers are presented, extract all of them. "
        "If fewer are presented, extract those present. Do not infer or invent data."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def try_parse_int(value: Optional[str]) -> Optional[int]:
    """Extract the first integer found in a string; return None if not found."""
    if value is None:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def non_increasing_order(nums: List[Optional[int]]) -> bool:
    """Check non-increasing order allowing ties (a1 >= a2 >= a3). Return False if any is None."""
    if any(n is None for n in nums):
        return False
    return all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1))


def unique_nonempty_urls(url_lists: List[List[str]]) -> List[str]:
    """Unique union of URL lists, preserving only non-empty strings."""
    seen = set()
    result = []
    for urls in url_lists:
        for u in urls:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    result.append(u2)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_aoty_album_identification(
    evaluator: Evaluator,
    parent_node,
    album_extraction: AlbumWinnerExtraction
) -> None:
    """Build and verify the AOTY album identification subtree."""
    node = evaluator.add_parallel(
        id="AOTY_Album_Identification",
        desc="Correctly identify the Album of the Year winner at the 2026 Grammy Awards and provide verification.",
        parent=parent_node,
        critical=True
    )

    # Album stated (existence check)
    album_stated = evaluator.add_custom_node(
        result=bool(album_extraction.album_title and album_extraction.album_title.strip()),
        id="Album_Stated",
        desc="Response states a specific album (title) as the 2026 Grammy Awards (68th) Album of the Year winner.",
        parent=node,
        critical=True
    )

    # Verify winner by URL(s)
    winner_verify_leaf = evaluator.add_leaf(
        id="AOTY_Winner_Verified_By_URL",
        desc="Provides at least one reference URL that supports the AOTY winner claim.",
        parent=node,
        critical=True
    )

    album_title = album_extraction.album_title or ""
    aoty_sources = album_extraction.sources or []
    claim = (
        f"The album '{album_title}' won Album of the Year at the 2026 Grammy Awards (68th Annual Grammy Awards)."
    )
    await evaluator.verify(
        claim=claim,
        node=winner_verify_leaf,
        sources=aoty_sources,
        additional_instruction=(
            "Verify strictly using the provided URL(s). Accept credible sources such as Grammy.com, Wikipedia awards pages, "
            "or reliable news listings. The URL(s) must clearly state the Album of the Year winner for the 2026 Grammys (68th Annual)."
        ),
    )


async def build_producer_details_for_index(
    evaluator: Evaluator,
    parent_node,
    album_title: str,
    producer: ProducerExtractionItem,
    index: int
) -> None:
    """Build and verify the details subtree for a single producer."""
    details_node = evaluator.add_parallel(
        id=f"Producer_{index+1}_Details",
        desc=(
            f"{['First', 'Second', 'Third'][index]}-listed producer includes all required fields with verification."
            if index < 3 else f"Producer #{index+1} details"
        ),
        parent=parent_node,
        critical=True
    )

    # Name provided
    name_provided = evaluator.add_custom_node(
        result=bool(producer.name and producer.name.strip()),
        id=f"P{index+1}_Name_Provided",
        desc="Provides the producer’s full name or professional name.",
        parent=details_node,
        critical=True
    )

    # Exact track count provided (integer)
    count_int = try_parse_int(producer.track_count)
    count_provided = evaluator.add_custom_node(
        result=(count_int is not None),
        id=f"P{index+1}_Exact_Track_Count_Provided",
        desc="Provides an exact integer count of tracks on the album on which the producer is credited as producer.",
        parent=details_node,
        critical=True
    )

    # Credit verification URL provided
    urls_provided = evaluator.add_custom_node(
        result=bool(producer.urls and len(producer.urls) > 0),
        id=f"P{index+1}_Credit_Verification_URL",
        desc="Provides at least one reference URL that supports the producer credit information for the identified album.",
        parent=details_node,
        critical=True
    )

    # Track count matches source(s)
    count_matches_leaf = evaluator.add_leaf(
        id=f"P{index+1}_Track_Count_Matches_Source",
        desc="The stated track count for the producer matches what the cited source(s) support for producer credits on that album.",
        parent=details_node,
        critical=True
    )

    pname = producer.name or ""
    claim = (
        f"On the album '{album_title}', {pname} is credited as producer on exactly {count_int} tracks."
        if count_int is not None else f"On the album '{album_title}', {pname} is credited as producer on an exact number of tracks."
    )
    await evaluator.verify(
        claim=claim,
        node=count_matches_leaf,
        sources=producer.urls,
        additional_instruction=(
            "Use the provided URL(s) to confirm the number of tracks on which the named person is credited specifically as a producer. "
            "Treat 'producer' and 'co-producer' credits as producer credits. Do not count 'executive producer' or non-producer roles. "
            "If multiple versions of the album exist, use the main release as implied by the source."
        ),
    )


async def build_top_three_producers_section(
    evaluator: Evaluator,
    parent_node,
    album_title: str,
    producers_extraction: ProducersExtraction
) -> None:
    """Build and verify the top-three producers subtree."""
    node = evaluator.add_parallel(
        id="Top_3_Producers_By_Track_Credits",
        desc=(
            "Determine and present the three producers credited on the highest number of tracks on the identified album, "
            "with required details, verification, and correct ranking."
        ),
        parent=parent_node,
        critical=True
    )

    # Prepare the list and slice to the first 3 for detail verification
    producers_all = producers_extraction.producers or []
    producers_top3 = (producers_all[:3] if len(producers_all) >= 3 else producers_all + [ProducerExtractionItem()] * (3 - len(producers_all)))

    # Build producer detail nodes first (to serve as auto preconditions for other checks)
    for i, prod in enumerate(producers_top3[:3]):
        await build_producer_details_for_index(evaluator, node, album_title, prod, i)

    # Exactly three producers listed
    exactly_three = evaluator.add_custom_node(
        result=(len(producers_all) == 3),
        id="Exactly_Three_Producers_Listed",
        desc="Response lists exactly three producers as the top three by number of tracks credited as producer for the identified album.",
        parent=node,
        critical=True
    )

    # Descending order by track count (non-increasing: c1 >= c2 >= c3)
    desc_order_leaf = evaluator.add_leaf(
        id="Descending_Order_By_Track_Count",
        desc="The three producers are ordered in strictly non-increasing order by the stated track counts (highest first).",
        parent=node,
        critical=True
    )

    counts_int = [try_parse_int(p.track_count) for p in producers_top3[:3]]
    claim_order = (
        f"The stated integer track counts are in non-increasing order: {counts_int} (ties allowed). "
        f"Non-increasing means c1 >= c2 >= c3."
    )
    await evaluator.verify(
        claim=claim_order,
        node=desc_order_leaf,
        additional_instruction=(
            "Judge this as a pure logical check on the stated integers only. If any count is missing or not an integer, this should fail."
        ),
    )

    # Top three correctness: using all provided URLs from the top three producers
    top_three_correct_leaf = evaluator.add_leaf(
        id="Top_Three_Correctness",
        desc=(
            "The selected three producers are in fact the producers with the highest track-credit counts on the identified album, "
            "based on verifiable credits sources."
        ),
        parent=node,
        critical=True
    )

    aggregated_urls = unique_nonempty_urls([p.urls for p in producers_top3[:3]])
    names_counts_pairs = [
        (p.name or "", try_parse_int(p.track_count))
        for p in producers_top3[:3]
    ]
    readable_pairs = "; ".join([f"{n} - {c if c is not None else 'N/A'}" for n, c in names_counts_pairs])
    claim_top3 = (
        f"On the album '{album_title}', the top three producers by number of tracks credited are: {readable_pairs}. "
        f"No other producer credited as producer on this album has a higher track count than any of these three."
    )
    await evaluator.verify(
        claim=claim_top3,
        node=top_three_correct_leaf,
        sources=aggregated_urls,
        additional_instruction=(
            "Use the provided URL(s) to confirm per-track producer credits for the album and determine counts per producer. "
            "Treat 'producer' and 'co-producer' as producer credits. Exclude executive producer or non-producer roles. "
            "Confirm that no omitted producer has a higher track count."
        ),
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
    Evaluate an answer for the 2026 Grammy AOTY producers task.
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

    # Add a critical main task node under root to honor rubric criticality
    main_task = evaluator.add_sequential(
        id="Grammy_AOTY_Producers_Task",
        desc=(
            "Identify the Album of the Year winner at the 2026 Grammys (68th Annual) and the top three producers by track count "
            "with required details, verification, and correct ranking."
        ),
        parent=root,
        critical=True
    )

    # 1) Extract album identification
    album_extraction = await evaluator.extract(
        prompt=prompt_extract_aoty_album(),
        template_class=AlbumWinnerExtraction,
        extraction_name="aoty_album_extraction"
    )

    # Build album identification verification
    await build_aoty_album_identification(evaluator, main_task, album_extraction)

    # 2) Extract top producers from the answer
    album_title_hint = album_extraction.album_title
    producers_extraction = await evaluator.extract(
        prompt=prompt_extract_top_three_producers(album_title_hint),
        template_class=ProducersExtraction,
        extraction_name="top_producers_extraction"
    )

    # Build top three producers verification
    await build_top_three_producers_section(
        evaluator,
        main_task,
        album_extraction.album_title or "",
        producers_extraction
    )

    # Optionally record custom info
    evaluator.add_custom_info(
        info={
            "extracted_album_title": album_extraction.album_title,
            "album_sources_count": len(album_extraction.sources),
            "extracted_producers_count": len(producers_extraction.producers),
            "producer_names": [p.name for p in producers_extraction.producers[:3]]
        },
        info_type="extraction_summary"
    )

    return evaluator.get_summary()