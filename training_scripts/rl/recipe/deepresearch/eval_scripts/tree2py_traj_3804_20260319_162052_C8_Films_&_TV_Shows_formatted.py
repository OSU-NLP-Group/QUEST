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
TASK_ID = "streaming_2026_03_20"
TASK_DESCRIPTION = """
Identify two films or series that began streaming on major platforms on March 20, 2026, where each content item must meet ALL of the following requirements: (1) The streaming platform must be either Netflix or Peacock; (2) The streaming release date must be exactly March 20, 2026; (3) The content must have been released in theaters before becoming available for streaming; (4) The theatrical release date must be verifiable. For each of the two content items you identify, provide: the title of the film or series, the streaming platform (Netflix or Peacock), the streaming release date (must be March 20, 2026), the theatrical release date (must be before March 20, 2026), the runtime (in minutes for films) OR the number of episodes (for series), the director's name (for films) or creator's name (for series), the writer's or screenplay writer's name, and at least one principal cast member's name. Include reference URLs supporting each piece of information.
"""
TARGET_STREAM_DATE_TEXT = "March 20, 2026"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ContentItem(BaseModel):
    # Identity and basic classification
    title: Optional[str] = None
    content_type: Optional[str] = None  # "film", "movie", "series", "tv series", etc.

    # Required fields
    platform: Optional[str] = None  # Should be "Netflix" or "Peacock"
    streaming_release_date: Optional[str] = None  # Expected "March 20, 2026" or equivalent
    theatrical_release_date: Optional[str] = None  # Must be before Mar 20, 2026

    # Duration info (OR requirement)
    runtime_minutes: Optional[str] = None  # for films
    episodes_count: Optional[str] = None  # for series

    # Credits and cast
    director_or_creator: Optional[str] = None
    writer: Optional[str] = None
    main_cast: List[str] = Field(default_factory=list)

    # Optional extras (not required by task but present in rubric)
    genre: Optional[str] = None
    based_on_or_connection: Optional[str] = None
    production_company: Optional[str] = None

    # Per-field reference URLs
    platform_urls: List[str] = Field(default_factory=list)
    streaming_date_urls: List[str] = Field(default_factory=list)
    theatrical_date_urls: List[str] = Field(default_factory=list)
    runtime_or_episodes_urls: List[str] = Field(default_factory=list)
    director_or_creator_urls: List[str] = Field(default_factory=list)
    writer_urls: List[str] = Field(default_factory=list)
    cast_urls: List[str] = Field(default_factory=list)
    genre_urls: List[str] = Field(default_factory=list)
    connection_urls: List[str] = Field(default_factory=list)
    production_company_urls: List[str] = Field(default_factory=list)


class ContentExtraction(BaseModel):
    items: List[ContentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_content_items() -> str:
    return f"""
Extract up to TWO content items (films or series) from the answer. If more than two are mentioned, return only the first two. If fewer are mentioned, return whatever is available, and fill missing fields with null or empty arrays accordingly.

For EACH item, extract the following fields exactly as stated in the answer:

Identity and classification:
- title: The title of the film or series.
- content_type: One of "film"/"movie" or "series"/"tv series"/"limited series" etc. If unclear, infer from context; otherwise null.

Required verification targets (with URLs per field):
- platform: The streaming platform (expect "Netflix" or "Peacock").
- streaming_release_date: The streaming release date. Prefer exact textual form from the answer (e.g., "March 20, 2026" or "20 March 2026" or "2026-03-20").
- theatrical_release_date: The theatrical (cinema) release date for the content (must be before March 20, 2026).

Duration information (OR requirement):
- runtime_minutes: Runtime in minutes (for films). If not applicable, set null.
- episodes_count: Number of episodes (for series). If not applicable, set null.

Credits and cast:
- director_or_creator: Director (for films) or creator (for series) as stated.
- writer: Writer or screenplay writer.
- main_cast: A list with at least one principal cast member (if provided).

Optional extras:
- genre: If stated.
- based_on_or_connection: If explicitly stated that the content is based on prior work or connected to a prior IP, extract that description.
- production_company: Production company or studio.

Per-field reference URLs (as arrays of URLs). Extract only URLs explicitly present in the answer. Include plain URLs or markdown links, but return the actual URL strings:
- platform_urls
- streaming_date_urls
- theatrical_date_urls
- runtime_or_episodes_urls
- director_or_creator_urls
- writer_urls
- cast_urls
- genre_urls
- connection_urls
- production_company_urls

General rules:
- Do not invent or infer information that is not in the answer. If a field is missing, set it to null (or empty list for arrays).
- Normalize obvious malformed URLs. If a URL is missing a protocol, prepend "http://".
- The returned JSON must follow this Pydantic template: ContentExtraction with a field 'items' that is a list of ContentItem objects.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not _has_text(x):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _aggregate_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(lst or [])
    return _dedup(merged)


def _urls_provided_for_all_required_fields(item: ContentItem) -> bool:
    """Check that each critical field has at least one supporting URL."""
    url_sets = [
        item.platform_urls,
        item.streaming_date_urls,
        item.theatrical_date_urls,
        item.runtime_or_episodes_urls,
        item.director_or_creator_urls,
        item.writer_urls,
        item.cast_urls,
    ]
    # All required fields must have at least one URL
    return all(isinstance(us, list) and len(us) > 0 for us in url_sets)


# --------------------------------------------------------------------------- #
# Verification for a single content item                                      #
# --------------------------------------------------------------------------- #
async def verify_content_item(
    evaluator: Evaluator,
    parent_node,
    item: ContentItem,
    index: int,
) -> None:
    """
    Build the verification sub-tree for a single content item.
    The rubric tree expects a parallel node aggregating all checks.
    """
    # Item node: parallel aggregation
    item_node = evaluator.add_parallel(
        id="First_Content_Item" if index == 0 else "Second_Content_Item",
        desc="First streaming content item releasing on March 20, 2026" if index == 0
        else "Second streaming content item releasing on March 20, 2026",
        parent=parent_node,
        critical=False,  # Each item contributes partial credit independently
    )

    # Basic existence (title)
    evaluator.add_custom_node(
        result=_has_text(item.title),
        id=("first_item_title_exists" if index == 0 else "second_item_title_exists"),
        desc=f"{'First' if index == 0 else 'Second'} item: Title is provided",
        parent=item_node,
        critical=True,  # Gate other checks
    )

    # Reference URLs present (critical gating)
    evaluator.add_custom_node(
        result=_urls_provided_for_all_required_fields(item),
        id=("First_Item_Reference_URLs" if index == 0 else "Second_Item_Reference_URLs"),
        desc="Reference URLs supporting the information must be provided",
        parent=item_node,
        critical=True,
    )

    # 1) Streaming Platform (must be Netflix or Peacock)
    platform_leaf = evaluator.add_leaf(
        id=("First_Item_Streaming_Platform" if index == 0 else "Second_Item_Streaming_Platform"),
        desc="The streaming platform must be Netflix or Peacock",
        parent=item_node,
        critical=True,
    )
    platform_claim = (
        f"The streaming platform for '{item.title}' is either Netflix or Peacock, specifically it is '{item.platform}'."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=_aggregate_sources(item.platform_urls, item.streaming_date_urls),
        additional_instruction="Verify the title is associated with the stated platform (Netflix or Peacock). Allow minor phrasing variations like 'arrives on Netflix' or 'streams on Peacock'.",
    )

    # 2) Streaming Release Date (must be exactly March 20, 2026)
    stream_date_leaf = evaluator.add_leaf(
        id=("First_Item_Streaming_Release_Date" if index == 0 else "Second_Item_Streaming_Release_Date"),
        desc="The streaming release date must be March 20, 2026",
        parent=item_node,
        critical=True,
    )
    streaming_date_claim = (
        f"The streaming release date for '{item.title}' on {item.platform} is {TARGET_STREAM_DATE_TEXT}."
    )
    await evaluator.verify(
        claim=streaming_date_claim,
        node=stream_date_leaf,
        sources=item.streaming_date_urls,
        additional_instruction="Confirm the page explicitly states the streaming availability date as March 20, 2026 (accept equivalent formats like 2026-03-20 or 20 March 2026).",
    )

    # 3) Theatrical Release Date (verifiable and before March 20, 2026)
    theatrical_leaf = evaluator.add_leaf(
        id=("First_Item_Theatrical_Release_Date" if index == 0 else "Second_Item_Theatrical_Release_Date"),
        desc="The theatrical release date must be verifiable and occur before March 20, 2026",
        parent=item_node,
        critical=True,
    )
    theatrical_claim = (
        f"The theatrical release date of '{item.title}' was {item.theatrical_release_date}, "
        f"which is before {TARGET_STREAM_DATE_TEXT}."
    )
    await evaluator.verify(
        claim=theatrical_claim,
        node=theatrical_leaf,
        sources=item.theatrical_date_urls,
        additional_instruction="Verify the page lists the theatrical (cinema) release date. It must clearly be earlier than March 20, 2026.",
    )

    # 4) Runtime OR Episode Count (choose whichever provided)
    runtime_or_eps_leaf = evaluator.add_leaf(
        id=("First_Item_Runtime_or_Episode_Count" if index == 0 else "Second_Item_Runtime_or_Episode_Count"),
        desc="For films, provide the runtime in minutes; for series, provide the number of episodes",
        parent=item_node,
        critical=True,
    )
    if _has_text(item.runtime_minutes):
        duration_claim = f"The runtime of '{item.title}' is {item.runtime_minutes} minutes."
    elif _has_text(item.episodes_count):
        duration_claim = f"The series '{item.title}' has {item.episodes_count} episodes."
    else:
        # Construct a claim that is guaranteed to be judged false if nothing provided
        duration_claim = f"No valid runtime or episode count is provided for '{item.title}'."
    await evaluator.verify(
        claim=duration_claim,
        node=runtime_or_eps_leaf,
        sources=item.runtime_or_episodes_urls,
        additional_instruction="Verify runtime (for films) or total episode count (for series) as presented on the page.",
    )

    # 5) Director or Creator
    doc_leaf = evaluator.add_leaf(
        id=("First_Item_Director_or_Creator" if index == 0 else "Second_Item_Director_or_Creator"),
        desc="The director (for films) or creator (for series) must be specified",
        parent=item_node,
        critical=True,
    )
    if (item.content_type or "").lower() in ["film", "movie", "feature"]:
        doc_claim = f"The director of '{item.title}' is {item.director_or_creator}."
    elif (item.content_type or "").lower() in ["series", "tv series", "limited series", "miniseries"]:
        doc_claim = f"The creator of the series '{item.title}' is {item.director_or_creator}."
    else:
        # Fallback wording if content_type uncertain
        doc_claim = f"For '{item.title}', the director or creator is {item.director_or_creator}."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=item.director_or_creator_urls,
        additional_instruction="Confirm the page explicitly credits the named person as director (film) or creator (series).",
    )

    # 6) Writer
    writer_leaf = evaluator.add_leaf(
        id=("First_Item_Writer" if index == 0 else "Second_Item_Writer"),
        desc="The writer or screenplay writer must be specified",
        parent=item_node,
        critical=True,
    )
    writer_claim = f"The writer or screenplay writer of '{item.title}' is {item.writer}."
    await evaluator.verify(
        claim=writer_claim,
        node=writer_leaf,
        sources=item.writer_urls,
        additional_instruction="Verify the page credits the stated person as writer/screenwriter.",
    )

    # 7) Main cast (at least one)
    cast_leaf = evaluator.add_leaf(
        id=("First_Item_Main_Cast_Member" if index == 0 else "Second_Item_Main_Cast_Member"),
        desc="At least one principal cast member must be named",
        parent=item_node,
        critical=True,
    )
    cast_name = item.main_cast[0] if item.main_cast else None
    cast_claim = f"One principal cast member of '{item.title}' is {cast_name}."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=item.cast_urls,
        additional_instruction="Verify the listed person appears as a principal/lead cast member on the page.",
    )

    # 8) Genre (mark non-critical; the base task didn't strictly require genre)
    genre_leaf = evaluator.add_leaf(
        id=("First_Item_Genre" if index == 0 else "Second_Item_Genre"),
        desc="The genre must be identifiable",
        parent=item_node,
        critical=False,  # Adjusted to non-critical to align with task requirements
    )
    genre_claim = f"The genre of '{item.title}' is {item.genre}."
    await evaluator.verify(
        claim=genre_claim,
        node=genre_leaf,
        sources=item.genre_urls,
        additional_instruction="Verify that the page identifies the genre for the title.",
    )

    # 9) Connection to prior work (non-critical)
    conn_leaf = evaluator.add_leaf(
        id=("First_Item_Connection_to_Prior_Work" if index == 0 else "Second_Item_Connection_to_Prior_Work"),
        desc="The content must be based on or connected to prior work",
        parent=item_node,
        critical=False,  # Adjusted to non-critical to avoid over-penalizing
    )
    conn_claim = f"The content '{item.title}' is based on or connected to: {item.based_on_or_connection}."
    await evaluator.verify(
        claim=conn_claim,
        node=conn_leaf,
        sources=item.connection_urls,
        additional_instruction="Verify any stated 'based on', adaptation, sequel, reboot, or other connection to prior IP.",
    )

    # 10) Production company (non-critical)
    prod_leaf = evaluator.add_leaf(
        id=("First_Item_Production_Company" if index == 0 else "Second_Item_Production_Company"),
        desc="The production company or studio involvement must be verifiable",
        parent=item_node,
        critical=False,  # Adjusted to non-critical to align with task description focus
    )
    prod_claim = f"The production company or studio involved with '{item.title}' is {item.production_company}."
    await evaluator.verify(
        claim=prod_claim,
        node=prod_leaf,
        sources=item.production_company_urls,
        additional_instruction="Verify production company or studio involvement as listed on the page.",
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
    Entry point for evaluating answers for the March 20, 2026 streaming content task.
    """
    # Initialize evaluator with a parallel root (two items evaluated independently)
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

    # Extract structured content items from the answer
    extracted: ContentExtraction = await evaluator.extract(
        prompt=prompt_extract_content_items(),
        template_class=ContentExtraction,
        extraction_name="content_items_extraction",
    )

    # Keep only the first two items; pad with empty items if fewer than two
    items: List[ContentItem] = list(extracted.items[:2])
    while len(items) < 2:
        items.append(ContentItem())

    # Add light GT/context info
    evaluator.add_custom_info(
        info={"target_stream_date": TARGET_STREAM_DATE_TEXT, "allowed_platforms": ["Netflix", "Peacock"]},
        info_type="task_constraints",
        info_name="constraints",
    )

    # Build verification subtrees for each of the two items
    for idx in range(2):
        await verify_content_item(evaluator, root, items[idx], idx)

    # Return evaluation summary
    return evaluator.get_summary()