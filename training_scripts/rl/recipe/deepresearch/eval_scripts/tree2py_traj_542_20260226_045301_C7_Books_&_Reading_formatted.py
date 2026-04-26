import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_2025_winners"
TASK_DESCRIPTION = (
    "Identify the 2025 National Book Award winners in all five categories (Fiction, Nonfiction, Poetry, "
    "Translated Literature, and Young People's Literature) and provide the following information for each winner: "
    "author name, book title, and publisher. For the Translated Literature category winner, also include the translator's name."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class WinnerInfo(BaseModel):
    author: Optional[str] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    translator: Optional[str] = None  # Only required for Translated Literature
    sources: List[str] = Field(default_factory=list)


class NBAWinnersExtraction(BaseModel):
    fiction: Optional[WinnerInfo] = None
    nonfiction: Optional[WinnerInfo] = None
    poetry: Optional[WinnerInfo] = None
    translated_literature: Optional[WinnerInfo] = None
    young_peoples_literature: Optional[WinnerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winners() -> str:
    return """
    Extract the 2025 National Book Award winners as presented in the answer. You must extract exactly one winner for each of the five categories:
    - Fiction
    - Nonfiction
    - Poetry
    - Translated Literature
    - Young People's Literature

    For each category, extract the following fields:
    - author: the winner author's name.
    - title: the full book title.
    - publisher: the publisher or imprint (exactly as written in the answer).
    - translator: the translator's name (only for Translated Literature; otherwise return null).
    - sources: an array of all URLs that the answer cites for that category (official National Book Foundation preferred, but also accept reputable major outlets if present). Extract only actual URLs mentioned in the answer.

    Return a single JSON object with the structure:
    {
      "fiction": { "author": ..., "title": ..., "publisher": ..., "translator": null, "sources": [...] },
      "nonfiction": { "author": ..., "title": ..., "publisher": ..., "translator": null, "sources": [...] },
      "poetry": { "author": ..., "title": ..., "publisher": ..., "translator": null, "sources": [...] },
      "translated_literature": { "author": ..., "title": ..., "publisher": ..., "translator": ..., "sources": [...] },
      "young_peoples_literature": { "author": ..., "title": ..., "publisher": ..., "translator": null, "sources": [...] }
    }

    Rules:
    - Do not fabricate any information; return null for any field not provided in the answer.
    - The 'sources' array must only include actual URLs that appear in the answer (in plain form or markdown links).
    - Keep the exact strings as written in the answer (e.g., preserve capitalization and punctuation).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _required_fields_present(info: Optional[WinnerInfo], needs_translator: bool) -> bool:
    if info is None:
        return False
    if not (_is_nonempty(info.author) and _is_nonempty(info.title) and _is_nonempty(info.publisher)):
        return False
    if needs_translator and not _is_nonempty(info.translator):
        return False
    return True


def _citation_present(info: Optional[WinnerInfo]) -> bool:
    return bool(info and isinstance(info.sources, list) and len(info.sources) > 0)


# --------------------------------------------------------------------------- #
# Verification per-category                                                   #
# --------------------------------------------------------------------------- #
async def verify_category(
    evaluator: Evaluator,
    parent_node,
    cat_key: str,
    cat_label: str,
    needs_translator: bool,
    info: Optional[WinnerInfo],
) -> None:
    """
    Build verification nodes for a single category and run checks.
    """
    # Category aggregator (parallel, non-critical to allow partial credit across categories)
    cat_node = evaluator.add_parallel(
        id=f"{cat_key}_winner",
        desc=f"{cat_label} category winner information is provided and correct.",
        parent=parent_node,
        critical=False,
    )

    # 1) Required fields present (critical leaf)
    req_ok = _required_fields_present(info, needs_translator)
    evaluator.add_custom_node(
        result=req_ok,
        id=f"{cat_key}_required_fields",
        desc=f"Provides {cat_label} winner required fields ({'author/title/publisher/translator' if needs_translator else 'author/title/publisher'}).",
        parent=cat_node,
        critical=True,
    )

    # 2) Citation provided (at least one URL) (critical leaf)
    cit_ok = _citation_present(info)
    evaluator.add_custom_node(
        result=cit_ok,
        id=f"{cat_key}_citation_provided",
        desc=f"Includes at least one citation/link for the {cat_label} winner.",
        parent=cat_node,
        critical=True,
    )

    # 3) Correctness checks grouped (critical aggregator with separate leaf checks)
    correctness_group = evaluator.add_parallel(
        id=f"{cat_key}_correctness_group",
        desc=f"{cat_label} winner details match official sources or reputable reporting (author/title/publisher"
             f"{'/translator' if needs_translator else ''}).",
        parent=cat_node,
        critical=True,
    )

    # Prepare common claim parts and sources
    author = info.author if info else None
    title = info.title if info else None
    publisher = info.publisher if info else None
    translator = info.translator if info else None
    sources = info.sources if info else []

    # Author correctness
    author_leaf = evaluator.add_leaf(
        id=f"{cat_key}_author_correct",
        desc=f"{cat_label}: author matches sources",
        parent=correctness_group,
        critical=True,
    )
    author_claim = f"The 2025 National Book Award winner for {cat_label} lists the author as {author}."
    await evaluator.verify(
        claim=author_claim,
        node=author_leaf,
        sources=sources,
        additional_instruction=(
            "Verify on the cited page(s) that for the 2025 National Book Award in the specified category, "
            f"the author is listed as {author}. Allow minor formatting or name variants (middle initials, accents, casing)."
        ),
    )

    # Title correctness
    title_leaf = evaluator.add_leaf(
        id=f"{cat_key}_title_correct",
        desc=f"{cat_label}: title matches sources",
        parent=correctness_group,
        critical=True,
    )
    title_claim = f"The 2025 National Book Award winner for {cat_label} is titled '{title}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the winning book title for the 2025 National Book Award in this category matches exactly or clearly "
            "corresponds to the provided title (allowing for standard punctuation or subtitle variations)."
        ),
    )

    # Publisher correctness
    publisher_leaf = evaluator.add_leaf(
        id=f"{cat_key}_publisher_correct",
        desc=f"{cat_label}: publisher matches sources",
        parent=correctness_group,
        critical=True,
    )
    publisher_claim = (
        f"The 2025 National Book Award winner for {cat_label} was published by {publisher}."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        sources=sources,
        additional_instruction=(
            "Check the cited page(s) for the publisher or imprint. Treat an imprint as valid if it is a recognized imprint "
            "of the named publisher (and vice versa)."
        ),
    )

    # Translator correctness (only for Translated Literature)
    if needs_translator:
        translator_leaf = evaluator.add_leaf(
            id=f"{cat_key}_translator_correct",
            desc=f"{cat_label}: translator matches sources",
            parent=correctness_group,
            critical=True,
        )
        translator_claim = (
            f"The 2025 National Book Award winner for {cat_label} lists the translator as {translator}."
        )
        await evaluator.verify(
            claim=translator_claim,
            node=translator_leaf,
            sources=sources,
            additional_instruction=(
                "Verify on the cited page(s) that the translator for the 2025 winner in this category matches the provided name. "
                "Allow minor name variants (middle initials, diacritics, casing)."
            ),
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the 2025 National Book Award winners.
    """
    # Initialize evaluator with a parallel root to aggregate independent category checks
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

    # Extract structured winners info from the answer
    winners = await evaluator.extract(
        prompt=prompt_extract_winners(),
        template_class=NBAWinnersExtraction,
        extraction_name="nba_2025_winners",
    )

    # Top-level aggregator node (non-critical to allow partial credit if some categories are correct)
    winners_node = evaluator.add_parallel(
        id="nba_2025_winners_main",
        desc="Identify the 2025 National Book Award winners in the five categories with correct bibliographic info and citations.",
        parent=root,
        critical=False,
    )

    # Map categories: (key in extraction, human-readable label, needs_translator)
    categories: List[Tuple[str, str, bool]] = [
        ("fiction", "Fiction", False),
        ("nonfiction", "Nonfiction", False),
        ("poetry", "Poetry", False),
        ("translated_literature", "Translated Literature", True),
        ("young_peoples_literature", "Young People's Literature", False),
    ]

    # Verify each category
    for key, label, needs_trans in categories:
        info = getattr(winners, key)
        await verify_category(
            evaluator=evaluator,
            parent_node=winners_node,
            cat_key=key,
            cat_label=label,
            needs_translator=needs_trans,
            info=info,
        )

    # Return standardized summary
    return evaluator.get_summary()