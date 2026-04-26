import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "album_grammy_66_aoty"
TASK_DESCRIPTION = (
    "What is the title of the album that won the Grammy Award for Album of the Year at the 66th Annual Grammy Awards (2024), "
    "was released in 2022, has exactly 13 tracks in its standard edition, and was released through a record label headquartered in New York?"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlbumExtraction(BaseModel):
    album_title: Optional[str] = None
    artist: Optional[str] = None
    release_year: Optional[str] = None
    standard_edition_track_count: Optional[str] = None
    label_name: Optional[str] = None
    label_hq: Optional[str] = None

    # URLs explicitly provided in the answer
    sources: List[str] = Field(default_factory=list)

    # Optional, category-specific sources if the answer distinguishes them
    grammy_sources: List[str] = Field(default_factory=list)
    release_year_sources: List[str] = Field(default_factory=list)
    tracklist_sources: List[str] = Field(default_factory=list)
    label_sources: List[str] = Field(default_factory=list)
    label_hq_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_album_info() -> str:
    return """
    Extract the album information explicitly stated in the answer.

    Required fields:
    - album_title: the album title as written in the answer (string)
    - artist: the primary artist name as written in the answer (string or null)
    - release_year: the release year for the album (string, e.g., "2022"; do not infer; null if not provided)
    - standard_edition_track_count: the number of tracks in the standard edition (string, exact text or digit; null if not provided)
    - label_name: the record label that released the album (string or null)
    - label_hq: the headquarters location for the label as stated (string or null)

    URLs explicitly provided in the answer:
    - sources: list of all URLs present in the answer if not categorized
    - grammy_sources: URLs that support the Grammy Album of the Year claim (list)
    - release_year_sources: URLs that support the album's release year (list)
    - tracklist_sources: URLs that support the standard edition track count (list)
    - label_sources: URLs that support the album-label association (list)
    - label_hq_sources: URLs that support the label HQ location (list)

    Rules:
    - Only extract what is explicitly present in the answer. Do not invent or infer.
    - If a field is not present, set it to null (or an empty array for URL lists).
    - For URLs: extract actual URLs (including markdown links). Ignore non-URL mentions.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    combined.append(u)
    return combined


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: AlbumExtraction) -> None:
    """
    Build the verification tree and perform checks according to the rubric.
    """
    # Top-level critical node aggregating all constraints
    album_node = evaluator.add_parallel(
        id="Album_Identification",
        desc=("Correctly identifies an album that won the Grammy Award for Album of the Year at the 66th Annual Grammy Awards (2024), "
              "was released in 2022, has exactly 13 tracks in its standard edition, and was released through a record label headquartered in New York"),
        parent=evaluator.root,
        critical=True,
    )

    # Existence / identification gate (critical)
    album_exists = evaluator.add_custom_node(
        result=_nonempty(extracted.album_title),
        id="Album_Exists",
        desc="An album title is provided in the answer",
        parent=album_node,
        critical=True
    )

    # 1) Grammy Award verification (critical)
    grammy_leaf = evaluator.add_leaf(
        id="Grammy_Award_Verification",
        desc="The identified album won the Grammy Award for Album of the Year at the 66th Annual Grammy Awards (2024 ceremony)",
        parent=album_node,
        critical=True
    )

    grammy_claim = f"The album '{extracted.album_title or ''}' won the Grammy Award for Album of the Year at the 66th Annual Grammy Awards (held in 2024)."
    grammy_sources = _combine_sources(extracted.grammy_sources, extracted.sources)
    await evaluator.verify(
        claim=grammy_claim,
        node=grammy_leaf,
        sources=grammy_sources if grammy_sources else None,
        additional_instruction="Verify the Album of the Year winner for the 66th Grammy Awards (2024). Allow minor variations in formatting of the album title."
    )

    # 2) Release year verification (critical)
    release_leaf = evaluator.add_leaf(
        id="Release_Year_Verification",
        desc="The identified album was released in 2022",
        parent=album_node,
        critical=True
    )

    release_claim = f"The album '{extracted.album_title or ''}' was released in 2022."
    release_sources = _combine_sources(extracted.release_year_sources, extracted.sources)
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=release_sources if release_sources else None,
        additional_instruction="Focus on the initial/original release year of the album. Deluxe or special reissues do not change the original release year."
    )

    # 3) Track count verification (critical)
    track_leaf = evaluator.add_leaf(
        id="Track_Count_Verification",
        desc="The standard edition of the identified album contains exactly 13 tracks",
        parent=album_node,
        critical=True
    )

    track_claim = f"The standard edition of the album '{extracted.album_title or ''}' contains exactly 13 tracks."
    track_sources = _combine_sources(extracted.tracklist_sources, extracted.sources)
    await evaluator.verify(
        claim=track_claim,
        node=track_leaf,
        sources=track_sources if track_sources else None,
        additional_instruction="Only consider the standard edition (ignore 3am editions, deluxe editions, or international versions) when counting tracks."
    )

    # 4) Label + Headquarters verification (split into two critical leaves under a critical parallel group)
    label_group = evaluator.add_parallel(
        id="Label_Headquarters_Verification",
        desc="The album was released through a record label with headquarters located in New York",
        parent=album_node,
        critical=True
    )

    # Gate for label info
    label_info_exists = evaluator.add_custom_node(
        result=_nonempty(extracted.label_name),
        id="Label_Info_Exists",
        desc="Record label name is provided in the answer",
        parent=label_group,
        critical=True
    )

    # 4a) Album-label association (critical)
    label_assoc_leaf = evaluator.add_leaf(
        id="Label_Association_Verification",
        desc="The album was released through the stated record label",
        parent=label_group,
        critical=True
    )

    label_assoc_claim = f"The album '{extracted.album_title or ''}' was released through the record label '{extracted.label_name or ''}'."
    label_assoc_sources = _combine_sources(extracted.label_sources, extracted.sources)
    await evaluator.verify(
        claim=label_assoc_claim,
        node=label_assoc_leaf,
        sources=label_assoc_sources if label_assoc_sources else None,
        additional_instruction="Confirm that the specified label released or issued the album (phrases like 'released by', 'through', or 'under' are acceptable)."
    )

    # 4b) Label headquarters location (critical)
    label_hq_leaf = evaluator.add_leaf(
        id="Label_HQ_Location_Verification",
        desc="The record label’s headquarters is located in New York",
        parent=label_group,
        critical=True
    )

    label_hq_claim = f"The record label '{extracted.label_name or ''}' has its headquarters located in New York."
    label_hq_sources = _combine_sources(extracted.label_hq_sources, extracted.sources)
    await evaluator.verify(
        claim=label_hq_claim,
        node=label_hq_leaf,
        sources=label_hq_sources if label_hq_sources else None,
        additional_instruction="Accept 'New York City', 'NYC', 'New York, NY', 'Manhattan, New York' as valid expressions of being headquartered in New York."
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
    Evaluate an answer for the album identification task (66th Grammys AOTY 2024 with constraints).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured album info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_album_info(),
        template_class=AlbumExtraction,
        extraction_name="album_extraction",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()