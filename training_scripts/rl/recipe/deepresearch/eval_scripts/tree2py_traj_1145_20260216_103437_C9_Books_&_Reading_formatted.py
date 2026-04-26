import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lit_fiction_2024_awards"
TASK_DESCRIPTION = (
    "Identify at least three literary fiction novels that meet ALL of the following criteria: "
    "(1) The novel was published in the United States in 2024 (with a US publication date between January 1, 2024 and December 31, 2024); "
    "(2) The novel was published by one of the 'Big Five' publishing houses (Penguin Random House, HarperCollins, Simon & Schuster, Macmillan, or Hachette) or one of their established imprints; "
    "(3) The novel won at least one major literary award: the 2024 National Book Award for Fiction, the 2024 Booker Prize, the 2024 Women's Prize for Fiction, OR the 2025 Pulitzer Prize for Fiction; "
    "(4) The novel was shortlisted (not just longlisted) for at least two different major international literary prizes during the 2024-2025 award season; "
    "(5) The novel appeared on at least one authoritative 'Best Books of 2024' list from major publications such as The New York Times, Publishers Weekly, Literary Hub, or similar recognized literary outlets. "
    "For each qualifying novel, provide: the novel's title, author's full name, US publisher name (including specific imprint), exact US publication date, all major awards won, all major prizes for which it was shortlisted, and at least one 'Best of 2024' list on which it appeared."
)

ALLOWED_MAJOR_AWARDS = [
    "2024 National Book Award for Fiction",
    "2024 Booker Prize",
    "2024 Women's Prize for Fiction",
    "2025 Pulitzer Prize for Fiction",
]

BIG_FIVE_LIST = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Macmillan",
    "Hachette"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NovelItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    us_publisher_imprint: Optional[str] = None
    us_publication_date: Optional[str] = None
    publication_urls: List[str] = Field(default_factory=list)

    awards_won: List[str] = Field(default_factory=list)
    award_urls: List[str] = Field(default_factory=list)

    shortlists: List[str] = Field(default_factory=list)
    shortlist_urls: List[str] = Field(default_factory=list)

    best_of_lists: List[str] = Field(default_factory=list)
    best_of_urls: List[str] = Field(default_factory=list)


class NovelsExtraction(BaseModel):
    novels: List[NovelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_novels() -> str:
    return """
    Extract up to five literary fiction novels from the answer with the following fields for each novel:
    - title: The novel’s title as written in the answer.
    - author: The full name of the author as written in the answer.
    - us_publisher_imprint: The United States publisher name including the specific imprint, as written in the answer. If the answer provides publisher and imprint separately, combine them into a single string like "Riverhead Books (Penguin Random House)".
    - us_publication_date: The exact US publication date as provided in the answer (retain the exact formatting used).
    - publication_urls: All URLs in the answer that directly support publisher/imprint and/or US publication date information (e.g., publisher page, imprint page, official book page, or retailer page that lists the US pub date).
    - awards_won: All major awards (as written in the answer) that the novel won.
    - award_urls: All URLs in the answer that support the award(s) information (e.g., official award site, news announcement).
    - shortlists: All prize shortlists (as written in the answer) that the novel was shortlisted for (2024–2025 award season).
    - shortlist_urls: All URLs supporting those shortlist claims.
    - best_of_lists: The 'Best Books of 2024' lists on which the novel appeared (e.g., 'The New York Times 10 Best Books of 2024', 'Publishers Weekly Best Books 2024', 'Literary Hub: The Best Books of 2024').
    - best_of_urls: All URLs supporting those 'Best of 2024' list appearances.
    
    IMPORTANT:
    - Extract only information explicitly present in the answer. Do not infer or invent additional data.
    - Return null for any missing single-value field (e.g., title, author, us_publisher_imprint, us_publication_date).
    - For URL lists, include only valid URLs explicitly present (plain or markdown links). If none are provided, return an empty list.
    - Preserve the answer’s exact wording for names and titles; do not normalize.
    - Do not include more than 5 novels in total in the extracted array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[\u2018\u2019’]", "'", s)
    s = re.sub(r"[\u201c\u201d“”]", '"', s)
    s = re.sub(r"[^a-z0-9&/\\'\"+ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_allowed_award(awards: List[str]) -> Optional[Tuple[str, str]]:
    """
    Try to find a canonical allowed award from the provided awards list.
    Returns (matched_award_from_answer, canonical_allowed_award) or None if not found.
    """
    if not awards:
        return None
    normalized_allowed = [(_normalize_text(a), a) for a in ALLOWED_MAJOR_AWARDS]
    for a in awards:
        na = _normalize_text(a)
        for norm_allowed, canonical in normalized_allowed:
            # Loose containment match to allow small formatting variations
            if norm_allowed in na or na in norm_allowed:
                return a, canonical
            # Handle "women s" vs "women's"
            if "womens prize" in na and "women's prize" in canonical.lower():
                return a, canonical
    return None


def _unique_nonempty(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        k = x.strip()
        if not k:
            continue
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


# --------------------------------------------------------------------------- #
# Verification for a single novel                                             #
# --------------------------------------------------------------------------- #
async def verify_single_novel(
    evaluator: Evaluator,
    parent_node,
    novel: NovelItem,
    novel_index: int,
) -> None:
    """
    Build and verify the tree for a single novel (index-based). Follows the rubric structure
    while ensuring each concrete check is a leaf node with a binary outcome.
    Note: To satisfy framework constraints (critical parent cannot have non-critical child),
    all children under critical parents are set to critical=True, including 'references' checks.
    """
    display_idx = novel_index + 1
    novel_node = evaluator.add_parallel(
        id=f"novel_{display_idx}",
        desc=f"{['First', 'Second', 'Third', 'Fourth', 'Fifth'][novel_index] if novel_index < 5 else f'Novel #{display_idx}'} qualifying novel",
        parent=parent_node,
        critical=False  # keep novel-level non-critical; internal groups enforce critical criteria
    )

    # ---------------- Publication information (critical group) ---------------- #
    pub_node = evaluator.add_parallel(
        id=f"novel_{display_idx}_publication_info",
        desc=f"Publication information for novel #{display_idx}",
        parent=novel_node,
        critical=True
    )

    # Output fields existence (critical leaves as custom nodes)
    evaluator.add_custom_node(
        result=bool(novel.title and novel.title.strip()),
        id=f"novel_{display_idx}_output_title",
        desc="The answer provides the novel's title",
        parent=pub_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(novel.author and novel.author.strip()),
        id=f"novel_{display_idx}_output_author",
        desc="The answer provides the author's full name",
        parent=pub_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(novel.us_publisher_imprint and novel.us_publisher_imprint.strip()),
        id=f"novel_{display_idx}_output_publisher_imprint",
        desc="The answer provides the US publisher name including specific imprint",
        parent=pub_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(novel.us_publication_date and novel.us_publication_date.strip()),
        id=f"novel_{display_idx}_output_exact_date",
        desc="The answer provides the exact US publication date",
        parent=pub_node,
        critical=True
    )
    # References presence (critical to gate web-grounded checks)
    evaluator.add_custom_node(
        result=bool(novel.publication_urls and len(_unique_nonempty(novel.publication_urls)) > 0),
        id=f"novel_{display_idx}_pub_references",
        desc="URL references supporting publication details are provided",
        parent=pub_node,
        critical=True
    )

    # US publication date in 2024 (web-verified)
    us_date_leaf = evaluator.add_leaf(
        id=f"novel_{display_idx}_us_pub_date_2024",
        desc="The novel was published in the United States with a publication date between January 1, 2024 and December 31, 2024",
        parent=pub_node,
        critical=True
    )
    claim_us_date = (
        f"The US publication date for '{novel.title or 'the novel'}' is {novel.us_publication_date or '[missing date]'}, "
        f"and it falls between January 1, 2024 and December 31, 2024."
    )
    await evaluator.verify(
        claim=claim_us_date,
        node=us_date_leaf,
        sources=_unique_nonempty(novel.publication_urls),
        additional_instruction="Verify the US publication date on the cited page(s). Minor formatting differences are acceptable, but the date must be in calendar year 2024."
    )

    # Big Five publisher/imprint (web-verified)
    big_five_leaf = evaluator.add_leaf(
        id=f"novel_{display_idx}_big_five_publisher",
        desc="The novel was published by a Big Five publisher or one of their established imprints",
        parent=pub_node,
        critical=True
    )
    claim_big_five = (
        f"The US publisher/imprint for '{novel.title or 'the novel'}' is '{novel.us_publisher_imprint or '[missing publisher]'}', "
        f"and it belongs to one of the Big Five publishers (Penguin Random House, HarperCollins, Simon & Schuster, Macmillan, or Hachette)."
    )
    await evaluator.verify(
        claim=claim_big_five,
        node=big_five_leaf,
        sources=_unique_nonempty(novel.publication_urls),
        additional_instruction="Use the publisher/imprint or credible reference pages to verify the imprint's parent. Accept common abbreviations (e.g., PRH) and minor name variations."
    )

    # ---------------- Award achievements (critical group) ---------------- #
    awards_node = evaluator.add_parallel(
        id=f"novel_{display_idx}_awards",
        desc=f"Award achievements for novel #{display_idx}",
        parent=novel_node,
        critical=True
    )

    # Ensure awards provided
    evaluator.add_custom_node(
        result=bool(novel.awards_won and len(_unique_nonempty(novel.awards_won)) > 0),
        id=f"novel_{display_idx}_output_awards_won",
        desc="The answer provides all major awards won by the novel",
        parent=awards_node,
        critical=True
    )
    # Ensure at least two shortlists listed in the answer
    evaluator.add_custom_node(
        result=len(_unique_nonempty(novel.shortlists)) >= 2,
        id=f"novel_{display_idx}_output_shortlists",
        desc="The answer provides all major prizes for which the novel was shortlisted",
        parent=awards_node,
        critical=True
    )
    # Award references presence (critical to gate web checks)
    evaluator.add_custom_node(
        result=bool(novel.award_urls and len(_unique_nonempty(novel.award_urls)) > 0),
        id=f"novel_{display_idx}_award_references",
        desc="URL references supporting award information are provided",
        parent=awards_node,
        critical=True
    )

    # Verify at least one allowed major award win (web-verified)
    matched = _find_allowed_award(novel.awards_won or [])
    if matched:
        matched_from_answer, canonical = matched
        claim_award = f"'{novel.title or 'The novel'}' won the {canonical}."
    else:
        # Fallback generic claim referencing the allowed set; verification will look for any of them on provided URLs
        claim_award = (
            f"'{novel.title or 'The novel'}' won at least one of the following awards: "
            f"{'; '.join(ALLOWED_MAJOR_AWARDS)}."
        )
    major_award_leaf = evaluator.add_leaf(
        id=f"novel_{display_idx}_major_award_win",
        desc="The novel won at least one specified major literary award",
        parent=awards_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_award,
        node=major_award_leaf,
        sources=_unique_nonempty(novel.award_urls),
        additional_instruction="Check whether the cited page(s) explicitly state that the book is a winner of one of the specified awards."
    )

    # Verify that there are at least two different shortlistings (counting check on extracted data)
    # Implemented as a custom binary leaf (critical) to avoid multi-URL multi-fact entanglement in a single leaf.
    evaluator.add_custom_node(
        result=len({_normalize_text(s) for s in _unique_nonempty(novel.shortlists)}) >= 2,
        id=f"novel_{display_idx}_two_shortlists",
        desc="The novel was shortlisted for at least two different major international prizes during the 2024-2025 award season",
        parent=awards_node,
        critical=True
    )

    # ---------------- Recognition (critical group) ---------------- #
    rec_node = evaluator.add_parallel(
        id=f"novel_{display_idx}_recognition",
        desc=f"Critical recognition for novel #{display_idx}",
        parent=novel_node,
        critical=True
    )

    # Ensure best-of list(s) output
    evaluator.add_custom_node(
        result=bool(novel.best_of_lists and len(_unique_nonempty(novel.best_of_lists)) > 0),
        id=f"novel_{display_idx}_output_best_of_list",
        desc="The answer provides at least one specific 'Best of 2024' list on which the novel appeared",
        parent=rec_node,
        critical=True
    )
    # References presence (critical)
    evaluator.add_custom_node(
        result=bool(novel.best_of_urls and len(_unique_nonempty(novel.best_of_urls)) > 0),
        id=f"novel_{display_idx}_recognition_references",
        desc="URL references supporting recognition details are provided",
        parent=rec_node,
        critical=True
    )

    # Verify best-of-2024 list appearance (web-verified)
    best_name = (_unique_nonempty(novel.best_of_lists) or [None])[0]
    best_of_leaf = evaluator.add_leaf(
        id=f"novel_{display_idx}_best_of_2024_list",
        desc="The novel appeared on at least one authoritative 'Best Books of 2024' list from major publications",
        parent=rec_node,
        critical=True
    )
    claim_best = (
        f"'{novel.title or 'The novel'}' is listed on the '{best_name or 'Best Books of 2024'}' list."
    )
    await evaluator.verify(
        claim=claim_best,
        node=best_of_leaf,
        sources=_unique_nonempty(novel.best_of_urls),
        additional_instruction="Verify that the cited list page includes the book among the 'Best Books of 2024'. Minor title formatting differences are acceptable."
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
    Evaluate an answer for the 2024 literary fiction award-winning novels task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root combines three novel subtrees in parallel
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
    # IMPORTANT: keep root non-critical per framework constraint (critical parents must have all-critical children)

    # Record ground truth rule set for transparency
    evaluator.add_ground_truth({
        "allowed_major_awards": ALLOWED_MAJOR_AWARDS,
        "big_five_publishers": BIG_FIVE_LIST,
        "require_us_publication_year": 2024,
        "require_at_least_two_shortlists": True,
        "require_best_of_2024_list": True
    }, gt_type="rules")

    # Extract structured novels data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_novels(),
        template_class=NovelsExtraction,
        extraction_name="extracted_novels"
    )

    # Normalize and select first three novels (pad with empty entries if fewer)
    novels: List[NovelItem] = (extracted.novels or [])[:3]
    while len(novels) < 3:
        novels.append(NovelItem())

    # Build verification subtrees for three novels
    for i in range(3):
        await verify_single_novel(evaluator, root, novels[i], i)

    # Produce final summary
    return evaluator.get_summary()