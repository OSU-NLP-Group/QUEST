import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "lit_awards_2025_novels"
TASK_DESCRIPTION = (
    "Identify four novels from major 2025 literary award lists that satisfy the following specific criteria. "
    "For each novel, provide its title, author, publisher, publication date, and reference URLs confirming each piece of information:\n\n"
    "1. A novel that was a 2025 National Book Awards Fiction finalist and was published by Knopf\n"
    "2. A novel that was a 2025 National Book Awards Fiction finalist and was published by Farrar, Straus and Giroux (FSG)\n"
    "3. A novel that was on the 2025 Booker Prize shortlist and was published by Farrar, Straus and Giroux (FSG)\n"
    "4. A novel that was on the 2025 Booker Prize shortlist and was published by a Penguin Random House imprint (such as Hogarth, Riverhead, Knopf, or Doubleday)"
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class NovelItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    basic_urls: List[str] = Field(default_factory=list)      # confirms title/author
    publisher_urls: List[str] = Field(default_factory=list)  # confirms publisher
    date_urls: List[str] = Field(default_factory=list)       # confirms publication date
    award_urls: List[str] = Field(default_factory=list)      # confirms award status


class NovelsExtraction(BaseModel):
    novels: List[NovelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_novels() -> str:
    return """
    Extract up to four novel entries the answer provides to satisfy the following four slots IN THIS EXACT ORDER:
    1) A 2025 National Book Awards Fiction finalist published by Knopf
    2) A 2025 National Book Awards Fiction finalist published by Farrar, Straus and Giroux (FSG)
    3) A 2025 Booker Prize shortlisted novel published by Farrar, Straus and Giroux (FSG)
    4) A 2025 Booker Prize shortlisted novel published by a Penguin Random House (PRH) imprint (e.g., Hogarth, Riverhead, Knopf, Doubleday, Viking, etc.)

    For each novel, extract:
    - title: Exact book title as written in the answer
    - author: Exact author name(s) as written in the answer
    - publisher: Publisher or imprint as written in the answer (do not infer; keep the text given)
    - publication_date: The publication date string as written in the answer (any reasonable format)
    - basic_urls: URLs in the answer that directly confirm the title and author (e.g., publisher or retailer page for the book, author page, etc.)
    - publisher_urls: URLs in the answer that confirm the publisher/imprint for this book
    - date_urls: URLs in the answer that confirm the publication date for this book
    - award_urls: URLs in the answer that confirm the award status (NBA finalist or Booker shortlist)

    IMPORTANT URL RULES:
    - Only extract URLs explicitly present in the answer. Do not invent any.
    - Include full URLs. If protocol is missing, prepend http://
    - If no URL of a required type is present, return an empty list for that URL field.

    Return JSON with a single field "novels", which is an array of at most 4 objects in the specified order.
    If the answer provides fewer than 4 suitable novels, include as many as provided and set missing fields to null/[] as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _get_novel(novels: List[NovelItem], idx: int) -> NovelItem:
    return novels[idx] if idx < len(novels) else NovelItem()


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_basic_info_checks(
    evaluator: Evaluator,
    parent,  # novel_i node
    item: NovelItem,
    novel_idx: int,
) -> None:
    # Critical group: Basic identification info
    basic_node = evaluator.add_parallel(
        id=f"novel_{novel_idx}_basic_info",
        desc=f"Basic identification information for the {'first' if novel_idx==1 else 'second' if novel_idx==2 else 'third' if novel_idx==3 else 'fourth'} novel",
        parent=parent,
        critical=True,
    )

    # Title present (existence check)
    evaluator.add_custom_node(
        result=_nonempty(item.title),
        id=f"novel_{novel_idx}_title",
        desc="The novel's title is provided",
        parent=basic_node,
        critical=True,
    )

    # Author present (existence check)
    evaluator.add_custom_node(
        result=_nonempty(item.author),
        id=f"novel_{novel_idx}_author",
        desc="The novel's author is provided",
        parent=basic_node,
        critical=True,
    )

    # At least one URL confirming title/author present (existence check)
    evaluator.add_custom_node(
        result=bool(item.basic_urls),
        id=f"novel_{novel_idx}_basic_reference",
        desc="Reference URL confirming title and author",
        parent=basic_node,
        critical=True,
    )


async def _build_publisher_checks(
    evaluator: Evaluator,
    parent,  # novel_i node
    item: NovelItem,
    novel_idx: int,
    publisher_requirement: str,  # 'knopf' | 'fsg' | 'prh'
) -> None:
    pub_node = evaluator.add_parallel(
        id=f"novel_{novel_idx}_publisher",
        desc=f"Publisher verification for the {'first' if novel_idx==1 else 'second' if novel_idx==2 else 'third' if novel_idx==3 else 'fourth'} novel",
        parent=parent,
        critical=True,
    )

    # Existence of publisher reference URLs (gates verification)
    evaluator.add_custom_node(
        result=bool(item.publisher_urls),
        id=f"novel_{novel_idx}_publisher_reference",
        desc="Reference URL confirming publisher",
        parent=pub_node,
        critical=True,
    )

    # Publisher-specific verification leaf
    if publisher_requirement == "knopf":
        pub_leaf = evaluator.add_leaf(
            id=f"novel_{novel_idx}_publisher_is_knopf",
            desc="The publisher is Knopf (or Knopf Doubleday Publishing Group)",
            parent=pub_node,
            critical=True,
        )
        title = item.title or "the book"
        claim = (
            f"This page shows that the publisher of '{title}' is Knopf (Alfred A. Knopf), "
            f"which is part of the Knopf Doubleday Publishing Group."
        )
        await evaluator.verify(
            claim=claim,
            node=pub_leaf,
            sources=item.publisher_urls,
            additional_instruction=(
                "Accept reasonable variants such as 'Alfred A. Knopf', or mention of 'Knopf Doubleday Publishing Group' "
                "as confirmation that the book is published under Knopf. Minor formatting differences are acceptable."
            ),
        )
    elif publisher_requirement == "fsg":
        pub_leaf = evaluator.add_leaf(
            id=f"novel_{novel_idx}_publisher_is_fsg",
            desc="The publisher is Farrar, Straus and Giroux (or FSG or Macmillan Publishers)",
            parent=pub_node,
            critical=True,
        )
        title = item.title or "the book"
        claim = (
            f"This page shows that the publisher of '{title}' is Farrar, Straus and Giroux (FSG)."
        )
        await evaluator.verify(
            claim=claim,
            node=pub_leaf,
            sources=item.publisher_urls,
            additional_instruction=(
                "Accept reasonable variants such as 'Farrar, Straus & Giroux' or 'FSG'. "
                "If the page clearly associates the imprint (e.g., MCD, FSG Originals) with FSG, that is acceptable."
            ),
        )
    elif publisher_requirement == "prh":
        pub_leaf = evaluator.add_leaf(
            id=f"novel_{novel_idx}_publisher_is_prh",
            desc="The publisher is a Penguin Random House imprint (Hogarth, Riverhead, Knopf, Doubleday, or other PRH imprint)",
            parent=pub_node,
            critical=True,
        )
        title = item.title or "the book"
        claim = (
            f"This page shows that the publisher of '{title}' is an imprint under Penguin Random House (PRH), "
            f"such as Hogarth, Riverhead, Knopf, Doubleday, Viking, or another PRH imprint."
        )
        await evaluator.verify(
            claim=claim,
            node=pub_leaf,
            sources=item.publisher_urls,
            additional_instruction=(
                "Confirm that the named imprint belongs to Penguin Random House (PRH). "
                "Pages that explicitly note PRH or list the imprint under PRH are sufficient."
            ),
        )
    else:
        # Fallback: mark as failed leaf if unknown requirement
        pub_leaf = evaluator.add_leaf(
            id=f"novel_{novel_idx}_publisher_requirement_unknown",
            desc="Unknown publisher requirement",
            parent=pub_node,
            critical=True,
            score=0.0,
            status="failed",
        )


async def _build_pubdate_checks(
    evaluator: Evaluator,
    parent,  # novel_i node
    item: NovelItem,
    novel_idx: int,
) -> None:
    date_node = evaluator.add_parallel(
        id=f"novel_{novel_idx}_publication_date",
        desc=f"Publication date verification for the {'first' if novel_idx==1 else 'second' if novel_idx==2 else 'third' if novel_idx==3 else 'fourth'} novel",
        parent=parent,
        critical=True,
    )

    # Existence of date reference URL
    evaluator.add_custom_node(
        result=bool(item.date_urls),
        id=f"novel_{novel_idx}_date_reference",
        desc="Reference URL confirming publication date",
        parent=date_node,
        critical=True,
    )

    # Year 2025 check using the provided date URLs
    year_leaf = evaluator.add_leaf(
        id=f"novel_{novel_idx}_year_2025",
        desc="The publication year is 2025",
        parent=date_node,
        critical=True,
    )
    title = item.title or "the book"
    claim = (
        f"This page indicates that the publication date for '{title}' is in the year 2025 "
        f"(month/day may vary by market)."
    )
    await evaluator.verify(
        claim=claim,
        node=year_leaf,
        sources=item.date_urls,
        additional_instruction=(
            "Confirm that the publication date shown on the page falls in 2025. "
            "If multiple regions are shown, accept the U.S. publication date if available."
        ),
    )


async def _build_award_checks(
    evaluator: Evaluator,
    parent,  # novel_i node
    item: NovelItem,
    novel_idx: int,
    award_requirement: str,  # 'nba_finalist' | 'booker_shortlist'
) -> None:
    award_node = evaluator.add_parallel(
        id=f"novel_{novel_idx}_award_status",
        desc=f"{'Award finalist' if award_requirement=='nba_finalist' else 'Award shortlist'} status verification for the {'first' if novel_idx==1 else 'second' if novel_idx==2 else 'third' if novel_idx==3 else 'fourth'} novel",
        parent=parent,
        critical=True,
    )

    # Existence of award reference URL
    evaluator.add_custom_node(
        result=bool(item.award_urls),
        id=f"novel_{novel_idx}_{'nba' if award_requirement=='nba_finalist' else 'booker'}_reference",
        desc=(
            "Reference URL from National Book Foundation or authoritative source confirming finalist status"
            if award_requirement == "nba_finalist"
            else "Reference URL from Booker Prize official website or authoritative source confirming shortlist status"
        ),
        parent=award_node,
        critical=True,
    )

    if award_requirement == "nba_finalist":
        leaf = evaluator.add_leaf(
            id=f"novel_{novel_idx}_nba_finalist",
            desc="The novel is listed as a 2025 National Book Awards Fiction finalist",
            parent=award_node,
            critical=True,
        )
        title = item.title or "the book"
        author = item.author or ""
        claim = (
            f"This page lists '{title}'{(' by ' + author) if author else ''} as a 2025 National Book Awards "
            f"Fiction finalist."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=item.award_urls,
            additional_instruction=(
                "Prefer the National Book Foundation (nationalbook.org) finalists page for 2025 (Fiction). "
                "Credible press releases or major media coverage explicitly stating 2025 NBA Fiction finalist status are acceptable."
            ),
        )
    elif award_requirement == "booker_shortlist":
        leaf = evaluator.add_leaf(
            id=f"novel_{novel_idx}_booker_shortlist",
            desc="The novel is on the 2025 Booker Prize shortlist",
            parent=award_node,
            critical=True,
        )
        title = item.title or "the book"
        author = item.author or ""
        claim = (
            f"This page lists '{title}'{(' by ' + author) if author else ''} on the 2025 Booker Prize shortlist."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=item.award_urls,
            additional_instruction=(
                "Prefer the official Booker Prizes site (thebookerprizes.com). "
                "Credible press releases or major media coverage explicitly stating 2025 Booker shortlist status are acceptable."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"novel_{novel_idx}_award_requirement_unknown",
            desc="Unknown award requirement",
            parent=award_node,
            critical=True,
            score=0.0,
            status="failed",
        )


async def _build_novel_verification(
    evaluator: Evaluator,
    root,
    item: NovelItem,
    idx: int,
    publisher_requirement: str,  # 'knopf' | 'fsg' | 'prh'
    award_requirement: str,      # 'nba_finalist' | 'booker_shortlist'
) -> None:
    novel_node = evaluator.add_parallel(
        id=f"novel_{idx}",
        desc=(
            "First novel: A 2025 National Book Awards Fiction finalist published by Knopf" if idx == 1 else
            "Second novel: A 2025 National Book Awards Fiction finalist published by Farrar, Straus and Giroux (FSG)" if idx == 2 else
            "Third novel: A 2025 Booker Prize shortlisted novel published by Farrar, Straus and Giroux (FSG)" if idx == 3 else
            "Fourth novel: A 2025 Booker Prize shortlisted novel published by a Penguin Random House imprint (such as Hogarth, Riverhead, Knopf, or Doubleday)"
        ),
        parent=root,
        critical=False,
    )

    # Build sub-verifications
    await _build_basic_info_checks(evaluator, novel_node, item, idx)
    await _build_publisher_checks(evaluator, novel_node, item, idx, publisher_requirement)
    await _build_pubdate_checks(evaluator, novel_node, item, idx)
    await _build_award_checks(evaluator, novel_node, item, idx, award_requirement)


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As per rubric root
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

    # Extract structured novel info
    extracted = await evaluator.extract(
        prompt=prompt_extract_novels(),
        template_class=NovelsExtraction,
        extraction_name="novels_extraction",
    )

    # Normalize to exactly 4 items (pad with empty if fewer)
    novels: List[NovelItem] = list(extracted.novels[:4])
    while len(novels) < 4:
        novels.append(NovelItem())

    evaluator.add_custom_info(
        info={
            "requirements": [
                "1) 2025 NBA Fiction finalist + Knopf",
                "2) 2025 NBA Fiction finalist + FSG",
                "3) 2025 Booker shortlist + FSG",
                "4) 2025 Booker shortlist + PRH imprint",
            ],
            "note": "All factual checks are grounded in the provided URLs when available."
        },
        info_type="task_requirements",
    )

    # Build and run verification subtrees for the four novels
    await _build_novel_verification(
        evaluator, root, _get_novel(novels, 0), 1, publisher_requirement="knopf", award_requirement="nba_finalist"
    )
    await _build_novel_verification(
        evaluator, root, _get_novel(novels, 1), 2, publisher_requirement="fsg", award_requirement="nba_finalist"
    )
    await _build_novel_verification(
        evaluator, root, _get_novel(novels, 2), 3, publisher_requirement="fsg", award_requirement="booker_shortlist"
    )
    await _build_novel_verification(
        evaluator, root, _get_novel(novels, 3), 4, publisher_requirement="prh", award_requirement="booker_shortlist"
    )

    return evaluator.get_summary()