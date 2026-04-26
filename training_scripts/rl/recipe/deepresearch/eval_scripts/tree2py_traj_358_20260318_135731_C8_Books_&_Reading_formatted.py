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
TASK_ID = "awards_2025_major_literary_winners"
TASK_DESCRIPTION = (
    "For the 2025 literary award season, identify the winning novels for the following four major awards: "
    "the Pulitzer Prize for Fiction, the National Book Award for Fiction, the PEN/Faulkner Award for Fiction, "
    "and the Booker Prize. For each winning novel, provide the following information: "
    "(1) The complete book title and author name, (2) The publisher (for the Booker Prize, provide the US publisher), "
    "(3) The parent company of the publisher (if the publisher is an imprint of a larger publishing house), "
    "(4) The page count of the book (hardcover edition), and (5) The exact date when the award winner was announced "
    "or the ceremony was held. Present your findings with supporting URL references for each piece of information."
)

# Expected constraints from rubric (used for "matches constraint" checks)
AWARD_CONFIG = {
    "pulitzer": {
        "display_name": "Pulitzer Prize for Fiction (2025)",
        "expected_title": "James",
        "expected_author": "Percival Everett",
        "expected_publisher": "Doubleday",
        "expected_parent_company": "Penguin Random House",
        "expected_page_count": "320",
        "award_date_str": "May 5, 2025",
        "publisher_label": "publisher",
    },
    "nba": {
        "display_name": "National Book Award for Fiction (2025)",
        "expected_title": "The True True Story of Raja the Gullible (and His Mother)",
        "expected_author": "Rabih Alameddine",
        "expected_publisher": "Grove Press",
        "expected_parent_company": None,  # Will verify whatever the answer claims with URLs
        "expected_page_count": "336",
        "award_date_str": "November 20, 2025",
        "publisher_label": "publisher",
    },
    "penfaulkner": {
        "display_name": "PEN/Faulkner Award for Fiction (2025)",
        "expected_title": "Small Rain",
        "expected_author": "Garth Greenwell",
        "expected_publisher": "Farrar, Straus and Giroux",
        "expected_parent_company": "Macmillan Publishers",
        "expected_page_count": "320",
        "award_date_str": "April 7, 2025",
        "publisher_label": "publisher",
    },
    "booker": {
        "display_name": "Booker Prize (2025)",
        "expected_title": "Flesh",
        "expected_author": "David Szalay",
        "expected_publisher": "Scribner",  # US publisher
        "expected_parent_company": "Simon & Schuster",
        "expected_page_count": "368",
        "award_date_str": "November 10, 2025",
        "publisher_label": "US publisher",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AttributeURLs(BaseModel):
    winner: List[str] = Field(default_factory=list)          # URLs supporting winner/title/author
    publisher: List[str] = Field(default_factory=list)       # URLs supporting publisher (or US publisher for Booker)
    parent_company: List[str] = Field(default_factory=list)  # URLs supporting parent company relationship
    page_count: List[str] = Field(default_factory=list)      # URLs supporting hardcover page count
    award_date: List[str] = Field(default_factory=list)      # URLs supporting the announcement/ceremony date
    other: List[str] = Field(default_factory=list)           # Any additional references the answer provides


class AwardItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None           # For Booker, this should be the US publisher
    parent_company: Optional[str] = None      # If none/independent, allow "independent"/"none"/"N/A"
    page_count: Optional[str] = None          # Keep as string to allow variants like "320 pages"
    award_date: Optional[str] = None          # E.g., "May 5, 2025"
    urls: AttributeURLs = Field(default_factory=AttributeURLs)


class AwardsExtraction(BaseModel):
    pulitzer: Optional[AwardItem] = None
    national_book_award: Optional[AwardItem] = None
    pen_faulkner: Optional[AwardItem] = None
    booker: Optional[AwardItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_awards() -> str:
    return """
Extract structured information reported in the answer for FOUR 2025 awards:
- pulitzer: Pulitzer Prize for Fiction (2025)
- national_book_award: National Book Award for Fiction (2025)
- pen_faulkner: PEN/Faulkner Award for Fiction (2025)
- booker: Booker Prize (2025)

For EACH award, extract these fields:
- title: Complete book title (string)
- author: Author name (string)
- publisher: The publisher. For Booker, extract the US publisher into this field.
- parent_company: The parent company of the publisher if it is an imprint; if clearly independent, return 'independent' or 'none'
- page_count: Hardcover edition page count (string; do NOT force a number; keep format like '320' or '320 pages')
- award_date: The exact date when the winner was announced or the ceremony was held (string)

Also extract URL references for each fact as arrays under a nested 'urls' object:
- urls.winner: URLs confirming the correct winner/title/author for the 2025 award
- urls.publisher: URLs confirming the publisher (US publisher for Booker)
- urls.parent_company: URLs confirming the parent company relation (or independence)
- urls.page_count: URLs confirming the hardcover page count
- urls.award_date: URLs confirming the exact announcement/ceremony date
- urls.other: Any additional relevant references in the answer

IMPORTANT:
- Only extract URLs explicitly present in the answer. Do not invent URLs.
- If a required field is not present in the answer, return null (or empty array for URL fields).
- Keep strings exactly as written in the answer (do not normalize).
- The 'publisher' for Booker must be the US publisher per the task.
Return a JSON object with keys: pulitzer, national_book_award, pen_faulkner, booker. Each key maps to the fields above.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_all_sources(item: Optional[AwardItem]) -> List[str]:
    if not item:
        return []
    urls = []
    urls.extend(item.urls.winner or [])
    urls.extend(item.urls.publisher or [])
    urls.extend(item.urls.parent_company or [])
    urls.extend(item.urls.page_count or [])
    urls.extend(item.urls.award_date or [])
    urls.extend(item.urls.other or [])
    # Deduplicate while preserving order
    seen = set()
    merged = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        if u not in seen:
            seen.add(u)
            merged.append(u)
    return merged


def _get_attr_sources(item: Optional[AwardItem], attr: str) -> List[str]:
    if not item:
        return []
    src = getattr(item.urls, attr, []) or []
    if src:
        return src
    # Fallback to merged award-level sources if attribute-specific sources missing
    return _merge_all_sources(item)


def _no_url_fail_instruction() -> str:
    return (
        "The answer did not provide any supporting URL(s) for this fact. For this evaluation, "
        "you must treat the claim as NOT SUPPORTED due to missing evidence."
    )


# --------------------------------------------------------------------------- #
# Verification per award                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_award(
    evaluator: Evaluator,
    parent_node,
    award_key: str,
    item: Optional[AwardItem],
    cfg: Dict[str, Optional[str]],
):
    """
    Build the verification subtree for one award item using the rubric leaves.
    Children are all critical leaves under a parallel award node.
    """
    display = cfg["display_name"]
    award_node = evaluator.add_parallel(
        id=f"{award_key}_item",
        desc=f"{display} winner and required attributes, supported by URLs",
        parent=parent_node,
        critical=False,  # per rubric: item is non-critical; leaves inside are critical
    )

    # Winner Title and Author matches constraint with URL
    winner_leaf = evaluator.add_leaf(
        id=f"{award_key}_winner_title_author",
        desc=f"Identifies the winner as '{cfg['expected_title']}' by {cfg['expected_author']}, with a supporting URL confirming the winner/title/author.",
        parent=award_node,
        critical=True,
    )
    winner_claim = f"The 2025 {display.replace(' (2025)', '')} was awarded to '{cfg['expected_title']}' by {cfg['expected_author']}."
    winner_sources = _get_attr_sources(item, "winner")
    await evaluator.verify(
        claim=winner_claim,
        node=winner_leaf,
        sources=winner_sources or None,
        additional_instruction=(
            "Verify the 2025 winner exactly (allowing minor casing/punctuation variants). "
            "The page must explicitly indicate the 2025 winner. "
            + (_no_url_fail_instruction() if not winner_sources else "")
        ),
    )

    # Publisher matches constraint with URL (Booker: US publisher)
    publisher_label = cfg.get("publisher_label", "publisher")
    publisher_leaf = evaluator.add_leaf(
        id=f"{award_key}_publisher",
        desc=f"States the {publisher_label} as {cfg['expected_publisher']}, with a supporting URL.",
        parent=award_node,
        critical=True,
    )
    pub_claim = (
        f"The {publisher_label} of '{cfg['expected_title']}' by {cfg['expected_author']}"
        f" is {cfg['expected_publisher']}."
    )
    pub_sources = _get_attr_sources(item, "publisher")
    await evaluator.verify(
        claim=pub_claim,
        node=publisher_leaf,
        sources=pub_sources or None,
        additional_instruction=(
            f"Confirm the {publisher_label} for the cited book. Accept imprint variants that clearly map to {cfg['expected_publisher']}. "
            + (_no_url_fail_instruction() if not pub_sources else "")
        ),
    )

    # Publisher Parent Company check
    parent_company_leaf = evaluator.add_leaf(
        id=f"{award_key}_parent_company",
        desc=(
            f"States {cfg['expected_publisher']} is part of {cfg['expected_parent_company']}, with a supporting URL."
            if cfg.get("expected_parent_company")
            else "Provides the parent company of the publisher (or states independent) with a supporting URL."
        ),
        parent=award_node,
        critical=True,
    )

    if cfg.get("expected_parent_company"):
        pc_claim = (
            f"{cfg['expected_publisher']} is an imprint of {cfg['expected_parent_company']}."
        )
    else:
        # For NBA: verify whatever the answer claims, but require a URL
        if item and item.publisher and item.parent_company:
            lc = item.parent_company.strip().lower()
            if lc in {"independent", "none", "n/a", "na", "no parent"}:
                pc_claim = (
                    f"{item.publisher} is an independent publisher (not part of a larger publishing house)."
                )
            else:
                pc_claim = f"{item.publisher} is an imprint of {item.parent_company}."
        else:
            pc_claim = (
                "The answer does not provide a verifiable parent company (or independence) claim for the publisher."
            )
    pc_sources = _get_attr_sources(item, "parent_company")
    await evaluator.verify(
        claim=pc_claim,
        node=parent_company_leaf,
        sources=pc_sources or None,
        additional_instruction=(
            "Verify the corporate relationship using reliable sources (e.g., publisher corporate pages, reputable articles). "
            "Allow minor naming variants (e.g., 'PRH' vs. 'Penguin Random House'). "
            + (_no_url_fail_instruction() if not pc_sources else "")
        ),
    )

    # Hardcover Page Count matches constraint with URL
    page_leaf = evaluator.add_leaf(
        id=f"{award_key}_hardcover_pages",
        desc=f"States hardcover page count is {cfg['expected_page_count']} pages, with a supporting URL.",
        parent=award_node,
        critical=True,
    )
    page_claim = (
        f"The hardcover edition of '{cfg['expected_title']}' by {cfg['expected_author']} has "
        f"{cfg['expected_page_count']} pages."
    )
    page_sources = _get_attr_sources(item, "page_count")
    await evaluator.verify(
        claim=page_claim,
        node=page_leaf,
        sources=page_sources or None,
        additional_instruction=(
            "Confirm the hardcover page count from credible sources (publisher catalog page, ISBN listing, librarian catalog, bookseller page). "
            "Allow minor formatting differences like '320' vs '320 pages'. "
            + (_no_url_fail_instruction() if not page_sources else "")
        ),
    )

    # Award Date matches constraint with URL
    date_leaf = evaluator.add_leaf(
        id=f"{award_key}_award_date",
        desc=f"States the award announcement/ceremony date is {cfg['award_date_str']}, with a supporting URL.",
        parent=award_node,
        critical=True,
    )
    # Phrase date type generically to cover either announcement or ceremony as allowed by task
    date_claim = (
        f"The {display} winner was announced or the official ceremony was held on {cfg['award_date_str']}."
    )
    date_sources = _get_attr_sources(item, "award_date")
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=date_sources or None,
        additional_instruction=(
            "The source must explicitly state the 2025 announcement or ceremony date; exact date matching is required "
            "(allowing format variants like 'Nov 10, 2025' vs 'November 10, 2025'). "
            + (_no_url_fail_instruction() if not date_sources else "")
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating a single answer against the 2025 major literary awards rubric.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # per rubric: root behaves as parallel aggregator
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

    # Extract structured data from the answer
    extraction: AwardsExtraction = await evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction",
    )

    # Record ground truth / expected constraints for transparency
    evaluator.add_ground_truth(
        {
            "expected": {
                "pulitzer": {
                    "title": AWARD_CONFIG["pulitzer"]["expected_title"],
                    "author": AWARD_CONFIG["pulitzer"]["expected_author"],
                    "publisher": AWARD_CONFIG["pulitzer"]["expected_publisher"],
                    "parent_company": AWARD_CONFIG["pulitzer"]["expected_parent_company"],
                    "page_count": AWARD_CONFIG["pulitzer"]["expected_page_count"],
                    "award_date": AWARD_CONFIG["pulitzer"]["award_date_str"],
                },
                "national_book_award": {
                    "title": AWARD_CONFIG["nba"]["expected_title"],
                    "author": AWARD_CONFIG["nba"]["expected_author"],
                    "publisher": AWARD_CONFIG["nba"]["expected_publisher"],
                    "parent_company": "Verify what answer claims (independent/imprint) with URLs",
                    "page_count": AWARD_CONFIG["nba"]["expected_page_count"],
                    "award_date": AWARD_CONFIG["nba"]["award_date_str"],
                },
                "pen_faulkner": {
                    "title": AWARD_CONFIG["penfaulkner"]["expected_title"],
                    "author": AWARD_CONFIG["penfaulkner"]["expected_author"],
                    "publisher": AWARD_CONFIG["penfaulkner"]["expected_publisher"],
                    "parent_company": AWARD_CONFIG["penfaulkner"]["expected_parent_company"],
                    "page_count": AWARD_CONFIG["penfaulkner"]["expected_page_count"],
                    "award_date": AWARD_CONFIG["penfaulkner"]["award_date_str"],
                },
                "booker": {
                    "title": AWARD_CONFIG["booker"]["expected_title"],
                    "author": AWARD_CONFIG["booker"]["expected_author"],
                    "us_publisher": AWARD_CONFIG["booker"]["expected_publisher"],
                    "parent_company": AWARD_CONFIG["booker"]["expected_parent_company"],
                    "page_count": AWARD_CONFIG["booker"]["expected_page_count"],
                    "award_date": AWARD_CONFIG["booker"]["award_date_str"],
                },
            }
        },
        gt_type="ground_truth_expected_constraints",
    )

    # Build award-level parent node (non-critical, parallel)
    awards_parent = evaluator.add_parallel(
        id="awards_2025",
        desc="Identify the winning novels for the four specified 2025 awards and provide required attributes; all required facts must be supported by URLs.",
        parent=root,
        critical=False,
    )

    # Map extraction to verification with expected constraints
    await verify_one_award(
        evaluator,
        awards_parent,
        "pulitzer",
        extraction.pulitzer,
        AWARD_CONFIG["pulitzer"],
    )
    await verify_one_award(
        evaluator,
        awards_parent,
        "nba",
        extraction.national_book_award,
        AWARD_CONFIG["nba"],
    )
    await verify_one_award(
        evaluator,
        awards_parent,
        "penfaulkner",
        extraction.pen_faulkner,
        AWARD_CONFIG["penfaulkner"],
    )
    await verify_one_award(
        evaluator,
        awards_parent,
        "booker",
        extraction.booker,
        AWARD_CONFIG["booker"],
    )

    return evaluator.get_summary()