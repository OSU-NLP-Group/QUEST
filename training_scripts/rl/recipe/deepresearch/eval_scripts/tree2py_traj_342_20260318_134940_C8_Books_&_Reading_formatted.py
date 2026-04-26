import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_2025_award_novels"
TASK_DESCRIPTION = (
    "I am building a curated reading collection for my book club focused on critically acclaimed fiction from 2025. "
    "Identify four novels that each won a different major English-language literary award in 2025. The awards to consider include: "
    "Pulitzer Prize for Fiction, National Book Award for Fiction, Booker Prize, Women's Prize for Fiction, and Edgar Award for Best Novel. "
    "For each of the four novels, provide: (1) The complete title of the novel, (2) The full name of the author, "
    "(3) The publisher's name, (4) The specific award the novel won in 2025, and (5) A reference URL from an authoritative source "
    "(such as the official award website, major news outlets, or Publishers Weekly) confirming the award win. "
    "Requirements: All four novels must have won different awards (you may not select two novels from the same award); "
    "all novels must be 2025 award winners, not finalists or nominees; all information must be verifiable through the provided reference URLs."
)

INDEX_TO_PREFIX = ["First", "Second", "Third", "Fourth"]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class AwardedNovel(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    award: Optional[str] = None
    reference_url: Optional[str] = None


class NovelsExtraction(BaseModel):
    novels: List[AwardedNovel] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_novels() -> str:
    return """
    Extract all novels mentioned in the answer that the answer claims WON a major English-language literary award in 2025.
    For each such novel, extract the following fields exactly as stated in the answer:
    - title: Complete title of the novel.
    - author: Full name of the author.
    - publisher: Publisher's name (e.g., imprint/house).
    - award: The specific award the novel won (e.g., 'Pulitzer Prize for Fiction', 'National Book Award for Fiction', 'Booker Prize',
             'Women's Prize for Fiction', 'Edgar Award for Best Novel', etc.). Do not list 'finalist', 'shortlist', or 'nominee'.
    - reference_url: A single URL (pick the most authoritative if multiple are given) that explicitly confirms the award win.

    IMPORTANT RULES:
    - Only include items that the answer explicitly claims to be WINNERS in 2025 (exclude finalists/nominees/shortlists).
    - The reference_url must be an actual URL string present in the answer (markdown links acceptable, extract the URL).
    - If any field is missing for an item, set it to null.
    - Return a JSON object with one field:
        novels: an array of objects {title, author, publisher, award, reference_url}
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_award_name(name: str) -> str:
    """Normalize award name for deduplication (very lightweight)."""
    s = (name or "").lower().strip()
    # remove non-alphanumerics for rough normalization
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _is_url_like(s: Optional[str]) -> bool:
    if not _is_nonempty(s):
        return False
    s = s.strip()
    return s.startswith("http://") or s.startswith("https://")


def _build_award_confirmation_claim(novel: AwardedNovel) -> str:
    title = novel.title or ""
    author = novel.author or ""
    award = novel.award or ""
    return (
        f"The webpage confirms that in 2025, the {award} was awarded to the novel '{title}' by {author}. "
        f"The page should indicate a WIN (winner), not a finalist, shortlist, or nominee."
    )


def _build_title_author_presence_claim(novel: AwardedNovel) -> str:
    title = novel.title or ""
    author = novel.author or ""
    return (
        f"This page shows the winning work as '{title}' by {author}, or an equivalent minor-variant of these names/titles."
    )


def _build_publisher_presence_claim(novel: AwardedNovel) -> str:
    publisher = novel.publisher or ""
    title = novel.title or ""
    author = novel.author or ""
    # Some official award pages list the publisher/imprint; others may not.
    # We verify if the provided publisher appears on the referenced authoritative confirmation page.
    return (
        f"This page mentions that the novel '{title}' by {author} has the publisher (or imprint/house) '{publisher}', "
        f"or an equivalent naming variant for the publisher."
    )


# --------------------------------------------------------------------------- #
# Verification subroutine per novel                                           #
# --------------------------------------------------------------------------- #
async def verify_single_novel(
    evaluator: Evaluator,
    parent_node,
    novel: AwardedNovel,
    novel_index: int,
) -> None:
    """
    Build verification nodes for a single novel.
    Structure:
      - Parallel novel-level node (non-critical)
         - Five existence/format checks (critical) for title/author/publisher/award/reference URL
         - Award confirmation on the reference page (critical; URL-grounded)
         - Title+Author presence on the reference page (non-critical; URL-grounded)
         - Publisher presence on the reference page (non-critical; URL-grounded)
    """
    prefix = INDEX_TO_PREFIX[novel_index]
    novel_node = evaluator.add_parallel(
        id=f"{prefix}_Novel",
        desc=f"{prefix} award-winning novel from 2025",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_is_nonempty(novel.title),
        id=f"{prefix}_Novel_Title",
        desc=f"Complete title of the {prefix.lower()} novel is provided",
        parent=novel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(novel.author),
        id=f"{prefix}_Novel_Author",
        desc=f"Full name of the author of the {prefix.lower()} novel is provided",
        parent=novel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(novel.publisher),
        id=f"{prefix}_Novel_Publisher",
        desc=f"Publisher name for the {prefix.lower()} novel is provided",
        parent=novel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(novel.award),
        id=f"{prefix}_Novel_Award",
        desc=f"The specific major literary award won by the {prefix.lower()} novel in 2025 is provided",
        parent=novel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_url_like(novel.reference_url),
        id=f"{prefix}_Novel_Reference",
        desc=f"A reference URL from an authoritative source confirming the award win is provided",
        parent=novel_node,
        critical=True
    )

    # Critical verification: the reference page confirms WINNER status in 2025 for the given book/author/award
    award_supported_node = evaluator.add_leaf(
        id=f"{prefix}_Novel_Award_Supported",
        desc=f"The provided reference URL confirms the {prefix.lower()} novel's award win in 2025 (not a finalist/nominee)",
        parent=novel_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_award_confirmation_claim(novel),
        node=award_supported_node,
        sources=novel.reference_url,
        additional_instruction=(
            "Judge strictly: The page must clearly indicate that the book is the WINNER of that award in 2025. "
            "Mentions like 'finalist', 'nominee', 'shortlist', or 'longlist' should be considered NOT supported. "
            "Allow minor variants in title/author formatting (case, punctuation, middle initials)."
        ),
    )

    # Non-critical verification: title+author presence on the page
    ta_present_node = evaluator.add_leaf(
        id=f"{prefix}_Novel_TitleAuthor_On_Source",
        desc=f"The title and author for the {prefix.lower()} novel match the reference page",
        parent=novel_node,
        critical=False
    )
    await evaluator.verify(
        claim=_build_title_author_presence_claim(novel),
        node=ta_present_node,
        sources=novel.reference_url,
        additional_instruction=(
            "Verify the page lists the winning work with a reasonably matching title and author name. "
            "Minor variants (e.g., punctuation, subtitles, middle names/initials, casing) should be considered a match."
        ),
    )

    # Non-critical verification: publisher presence (many award pages list publisher; if not, this may fail without harming the whole novel group)
    publisher_present_node = evaluator.add_leaf(
        id=f"{prefix}_Novel_Publisher_On_Source",
        desc=f"The publisher for the {prefix.lower()} novel matches the reference page (if mentioned on that page)",
        parent=novel_node,
        critical=False
    )
    await evaluator.verify(
        claim=_build_publisher_presence_claim(novel),
        node=publisher_present_node,
        sources=novel.reference_url,
        additional_instruction=(
            "If the page mentions the publisher/imprint/house, check it matches the provided publisher (allowing reasonable imprint vs. house variants). "
            "If the page does NOT mention any publisher at all, consider this verification as NOT supported."
        ),
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'four_2025_award_novels' task and return a structured summary.
    """
    # Initialize evaluator (root is always non-critical per framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent groups
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

    # Extract structured items
    extracted = await evaluator.extract(
        prompt=prompt_extract_novels(),
        template_class=NovelsExtraction,
        extraction_name="novels_extraction"
    )

    # Choose first four; pad if fewer
    novels: List[AwardedNovel] = list(extracted.novels[:4])
    while len(novels) < 4:
        novels.append(AwardedNovel())

    # Build tree: separate a critical constraints section and a non-critical collection section.
    # Note: The provided JSON marks the whole collection as critical, but that would force all children to be critical
    # (disallowed by framework). We therefore put global constraints in a critical node and the collection itself
    # in a non-critical node to allow partial credit across the four novels.
    constraints_node = evaluator.add_parallel(
        id="Global_Constraints",
        desc="Global constraints checks for the four-novel collection",
        parent=root,
        critical=True
    )

    collection_node = evaluator.add_parallel(
        id="Four_Award_Winning_Novels_Collection",
        desc="A collection of four novels, each winning a different major literary award in 2025",
        parent=root,
        critical=False
    )

    # Award Diversity check (critical): All four awards provided and all different
    awards_list = [n.award.strip() for n in novels if _is_nonempty(n.award)]
    normalized = [_normalize_award_name(a) for a in awards_list]
    all_four_present = (len([n for n in novels if _is_nonempty(n.award)]) == 4)
    all_distinct = (len(set(normalized)) == 4) if all_four_present else False

    evaluator.add_custom_node(
        result=all_four_present and all_distinct,
        id="Award_Diversity",
        desc="All four books must have won different awards (no two books from the same award)",
        parent=constraints_node,
        critical=True
    )

    # Build per-novel verification groups
    for i in range(4):
        await verify_single_novel(evaluator, collection_node, novels[i], i)

    # Return evaluation summary
    return evaluator.get_summary()