import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_memoirs_bigfive_2017_2025"
TASK_DESCRIPTION = (
    "Identify four celebrity memoirs or autobiographies published between 2017 and 2025, "
    "where each memoir is published by a different member of the Big Five publishing houses. "
    "For each memoir, provide: (1) the celebrity author's name, (2) the complete book title, "
    "(3) the specific imprint name (not just the parent publisher), (4) verification that this "
    "imprint belongs to the stated Big Five publisher, (5) the exact publication month and year, "
    "(6) the ISBN-13 number, and (7) reference URLs supporting each piece of information. "
    "Additionally, at least one memoir must have a co-author (provide the co-author's name), "
    "and at least one memoir must have won or been a finalist for either the National Book Award "
    "or the Pulitzer Prize (specify the award, year, and category)."
)

BIG_FIVE_CANONICAL = {
    "penguin random house": "Penguin Random House",
    "macmillan": "Macmillan",
    "harpercollins": "HarperCollins",
    "hachette": "Hachette",
    "simon & schuster": "Simon & Schuster",
    "simon and schuster": "Simon & Schuster",
    "hachette book group": "Hachette",
    "macmillan publishers": "Macmillan",
    "harper collins": "HarperCollins",
}

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    award_type: Optional[str] = None
    award_year: Optional[str] = None
    award_category: Optional[str] = None
    award_status: Optional[str] = None  # Winner / Finalist
    award_urls: List[str] = Field(default_factory=list)


class MemoirItem(BaseModel):
    author: Optional[str] = None
    coauthor: Optional[str] = None
    title: Optional[str] = None
    imprint: Optional[str] = None
    parent_publisher: Optional[str] = None
    publication_date: Optional[str] = None  # Month and Year (as string)
    isbn13: Optional[str] = None

    title_urls: List[str] = Field(default_factory=list)
    publisher_urls: List[str] = Field(default_factory=list)
    pub_date_urls: List[str] = Field(default_factory=list)
    isbn_urls: List[str] = Field(default_factory=list)
    author_urls: List[str] = Field(default_factory=list)
    coauthor_urls: List[str] = Field(default_factory=list)

    award: Optional[AwardInfo] = None


class MemoirExtraction(BaseModel):
    memoirs: List[MemoirItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_memoirs() -> str:
    return """
    Extract up to four celebrity memoirs or autobiographies described in the answer text. For each memoir, return a JSON object with the following fields (use null if missing, and lists may be empty):
    - author: primary celebrity author's name (as written in the answer)
    - coauthor: co-author's name if explicitly mentioned (else null)
    - title: complete book title (as written)
    - imprint: the specific imprint name (not just the parent publisher)
    - parent_publisher: the Big Five parent publisher explicitly stated in the answer (one of: Penguin Random House, Macmillan, HarperCollins, Hachette, Simon & Schuster). If the parent publisher is not clearly stated, return null. DO NOT infer; extract explicitly.
    - publication_date: the exact publication month and year string as written in the answer (e.g., "May 2019", "October 2023")
    - isbn13: the ISBN-13 string exactly as written (hyphens allowed)
    - title_urls: list of URLs in the answer that support the title (e.g., retailer or publisher page showing the title)
    - publisher_urls: list of URLs in the answer that support the publisher/imprint information (e.g., publisher page showing the imprint; imprint page showing it belongs to the parent publisher)
    - pub_date_urls: list of URLs in the answer that support the publication month/year
    - isbn_urls: list of URLs in the answer that support the ISBN-13
    - author_urls: list of URLs in the answer that support the author's celebrity status (e.g., profile pages, reputable bios)
    - coauthor_urls: list of URLs in the answer that support the co-author credit for this memoir (if applicable)
    - award: an object with potential award information (null if none was provided in the answer):
        * award_type: "National Book Award" or "Pulitzer Prize" if explicitly stated (else null)
        * award_year: year string as written (else null)
        * award_category: category string as written (e.g., "Nonfiction", "Biography") (else null)
        * award_status: "Winner" or "Finalist" if explicitly stated (else null)
        * award_urls: list of URLs that support the award details for this memoir

    Return a JSON object with a single field:
    {
      "memoirs": [ ... up to 4 items exactly as defined above ... ]
    }

    IMPORTANT:
    - Extract only what is explicitly present in the answer (do not invent).
    - Include URLs exactly as shown in the answer; markdown links should be converted to their target URLs.
    - If the answer lists more than four memoirs, include the first four in the order they appear.
    - If the answer lists fewer than four memoirs, include what is available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_publisher_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower().replace(" and ", " & ")
    # Also try variant with & replaced by 'and'
    key_alt = name.strip().lower().replace("&", "and")
    if key in BIG_FIVE_CANONICAL:
        return BIG_FIVE_CANONICAL[key]
    if key_alt in BIG_FIVE_CANONICAL:
        return BIG_FIVE_CANONICAL[key_alt]
    # Try compact variants
    compact = re.sub(r"\s+", " ", key).strip()
    if compact in BIG_FIVE_CANONICAL:
        return BIG_FIVE_CANONICAL[compact]
    return None


def year_in_range(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    # Find a 4-digit year
    m = re.search(r"\b(20\d{2})\b", date_str)
    if not m:
        return False
    year = int(m.group(1))
    if year < 2017 or year > 2025:
        return False
    # Check month presence (name or numeric)
    ds_lower = date_str.lower()
    has_month_name = any(mn in ds_lower for mn in MONTH_NAMES)
    has_numeric_month = bool(re.search(r"\b(0?[1-9]|1[0-2])\b", date_str))
    return has_month_name or has_numeric_month


def is_isbn13_format(isbn: Optional[str]) -> bool:
    if not isbn:
        return False
    digits = re.sub(r"[^0-9Xx]", "", isbn)
    # Allow 'X' only for ISBN-10, but we need ISBN-13 → must be exactly 13 digits
    return bool(re.fullmatch(r"\d{13}", digits))


def distinct_publisher_flags(memoirs: List[MemoirItem]) -> List[bool]:
    """
    For each memoir, check if its parent publisher (normalized to Big Five canonical)
    is present and distinct from the other three memoirs' normalized parent publishers.
    Returns per-item flags (True if distinct valid Big Five publisher, False otherwise).
    """
    normalized = [normalize_publisher_name(m.parent_publisher) for m in memoirs]
    flags = [False] * len(memoirs)
    for i, pub in enumerate(normalized):
        if pub is None:
            flags[i] = False
            continue
        # Distinct from others
        others = [p for j, p in enumerate(normalized) if j != i and p is not None]
        if pub in others:
            flags[i] = False
        else:
            # Valid Big Five and distinct
            flags[i] = True
    return flags


def pick_first_with_coauthor(memoirs: List[MemoirItem]) -> Optional[Tuple[int, MemoirItem]]:
    for idx, m in enumerate(memoirs):
        if m.coauthor and m.coauthor.strip():
            return idx, m
    return None


def pick_first_with_award(memoirs: List[MemoirItem]) -> Optional[Tuple[int, MemoirItem]]:
    for idx, m in enumerate(memoirs):
        if m.award and (m.award.award_type or m.award.award_year or m.award.award_category or m.award.award_status):
            return idx, m
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_memoir(
    evaluator: Evaluator,
    parent_node,
    memoir: MemoirItem,
    idx: int,
    is_distinct_publisher: bool,
) -> None:
    """
    Build verification sub-tree for one memoir (index idx: 0..3).
    """
    memoir_node = evaluator.add_parallel(
        id=f"Memoir_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} celebrity memoir meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # ------------------------ M{i}_Book_Title ---------------------------- #
    title_group = evaluator.add_parallel(
        id=f"M{idx+1}_Book_Title",
        desc="Provide the complete book title",
        parent=memoir_node,
        critical=True
    )
    # Title provided (existence)
    evaluator.add_custom_node(
        result=bool(memoir.title and memoir.title.strip()),
        id=f"M{idx+1}_Title_Provided",
        desc="Verify that a complete book title is provided",
        parent=title_group,
        critical=True
    )
    # Title URL verification
    title_url_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_Title_URL",
        desc="Provide reference URL confirming the book title",
        parent=title_group,
        critical=True
    )
    title_claim = f"This webpage shows the book titled '{memoir.title or ''}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_url_leaf,
        sources=memoir.title_urls,
        additional_instruction="Verify that the page clearly displays the exact book title."
    )

    # -------------------- M{i}_Publisher_Verification -------------------- #
    pub_group = evaluator.add_parallel(
        id=f"M{idx+1}_Publisher_Verification",
        desc=f"Verify publisher and imprint information for Memoir {idx+1}",
        parent=memoir_node,
        critical=True
    )
    # Publisher URL existence (gate)
    evaluator.add_custom_node(
        result=bool(memoir.publisher_urls),
        id=f"M{idx+1}_Publisher_URL",
        desc="Provide reference URL confirming publisher and imprint information",
        parent=pub_group,
        critical=True
    )
    # Parent publisher verification
    parent_pub_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_Parent_Publisher",
        desc="Confirm the memoir is published by one of the Big Five publishers (Penguin Random House, Macmillan, HarperCollins, Hachette, or Simon & Schuster)",
        parent=pub_group,
        critical=True
    )
    parent_pub_claim = (
        f"This page shows that '{memoir.title or ''}' was published by {memoir.parent_publisher or ''}, "
        f"which is one of the Big Five publishers."
    )
    await evaluator.verify(
        claim=parent_pub_claim,
        node=parent_pub_leaf,
        sources=memoir.publisher_urls,
        additional_instruction="Confirm the page explicitly identifies the publisher as the stated parent publisher."
    )
    # Specific imprint verification
    imprint_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_Specific_Imprint",
        desc="Identify and verify the specific imprint name within the parent publisher",
        parent=pub_group,
        critical=True
    )
    imprint_claim = (
        f"This page shows that the imprint that published '{memoir.title or ''}' is '{memoir.imprint or ''}'."
    )
    await evaluator.verify(
        claim=imprint_claim,
        node=imprint_leaf,
        sources=memoir.publisher_urls,
        additional_instruction="The page should clearly mention the imprint name associated with this title."
    )
    # Imprint belongs to publisher verification
    imprint_belongs_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_Imprint_Belongs_To_Publisher",
        desc="Verify that the stated imprint is actually owned by the stated parent publisher",
        parent=pub_group,
        critical=True
    )
    imprint_belongs_claim = (
        f"The imprint '{memoir.imprint or ''}' is part of {memoir.parent_publisher or ''}."
    )
    await evaluator.verify(
        claim=imprint_belongs_claim,
        node=imprint_belongs_leaf,
        sources=memoir.publisher_urls,
        additional_instruction="The supporting page(s) should establish that the imprint is an imprint of the specified parent publisher."
    )

    # -------------------- M{i}_Publication_Details ----------------------- #
    pub_details_group = evaluator.add_parallel(
        id=f"M{idx+1}_Publication_Details",
        desc=f"Verify complete publication details for Memoir {idx+1}",
        parent=memoir_node,
        critical=True
    )
    # Publication Date subgroup
    pub_date_group = evaluator.add_parallel(
        id=f"M{idx+1}_Publication_Date",
        desc="Provide exact publication month and year (between 2017-2025)",
        parent=pub_details_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=year_in_range(memoir.publication_date),
        id=f"M{idx+1}_Pub_Date_Within_Range",
        desc="Confirm publication date falls within 2017-2025 range",
        parent=pub_date_group,
        critical=True
    )
    pub_date_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_Pub_Date_URL",
        desc="Provide reference URL confirming publication date",
        parent=pub_date_group,
        critical=True
    )
    pub_date_claim = f"The publication date for '{memoir.title or ''}' is '{memoir.publication_date or ''}'."
    await evaluator.verify(
        claim=pub_date_claim,
        node=pub_date_leaf,
        sources=memoir.pub_date_urls,
        additional_instruction="Confirm the page shows the publication date (month and year) exactly as stated."
    )
    # ISBN subgroup
    isbn_group = evaluator.add_parallel(
        id=f"M{idx+1}_ISBN",
        desc="Provide valid ISBN-13 number",
        parent=pub_details_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_isbn13_format(memoir.isbn13),
        id=f"M{idx+1}_ISBN_Format",
        desc="Verify ISBN is in correct 13-digit format",
        parent=isbn_group,
        critical=True
    )
    isbn_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_ISBN_URL",
        desc="Provide reference URL confirming ISBN number",
        parent=isbn_group,
        critical=True
    )
    isbn_claim = f"The ISBN-13 for '{memoir.title or ''}' is '{memoir.isbn13 or ''}'."
    await evaluator.verify(
        claim=isbn_claim,
        node=isbn_leaf,
        sources=memoir.isbn_urls,
        additional_instruction="Confirm the page shows the exact ISBN-13 value."
    )

    # ---------------------- M{i}_Author_Details -------------------------- #
    auth_group = evaluator.add_parallel(
        id=f"M{idx+1}_Author_Details",
        desc=f"Verify author information for Memoir {idx+1}",
        parent=memoir_node,
        critical=True
    )
    # Author URL existence gate
    evaluator.add_custom_node(
        result=bool(memoir.author_urls),
        id=f"M{idx+1}_Author_URL",
        desc="Provide reference URL confirming author's celebrity status",
        parent=auth_group,
        critical=True
    )
    celeb_leaf = evaluator.add_leaf(
        id=f"M{idx+1}_Celebrity_Author",
        desc="Confirm the primary author is a recognized celebrity (entertainer, athlete, public figure)",
        parent=auth_group,
        critical=True
    )
    celeb_claim = (
        f"The author {memoir.author or ''} is a recognized celebrity or public figure."
    )
    await evaluator.verify(
        claim=celeb_claim,
        node=celeb_leaf,
        sources=memoir.author_urls,
        additional_instruction="Use reputable bios/profiles to confirm celebrity/public figure status."
    )

    # ---------------------- M{i}_Different_Publisher --------------------- #
    evaluator.add_custom_node(
        result=is_distinct_publisher,
        id=f"M{idx+1}_Different_Publisher",
        desc="Verify this memoir is from a different Big Five publisher than the other three memoirs",
        parent=memoir_node,
        critical=True
    )


async def verify_global_coauthor(
    evaluator: Evaluator,
    parent_node,
    memoirs: List[MemoirItem]
) -> None:
    co_group = evaluator.add_parallel(
        id="Global_Coauthor_Check",
        desc="Verify that at least one of the four memoirs has a credited co-author",
        parent=parent_node,
        critical=True
    )

    pick = pick_first_with_coauthor(memoirs)
    co_present = pick is not None

    evaluator.add_custom_node(
        result=co_present,
        id="Coauthor_Present",
        desc="Confirm that at least one memoir identifies a co-author by name",
        parent=co_group,
        critical=True
    )

    co_leaf = evaluator.add_leaf(
        id="Coauthor_URL",
        desc="Provide reference URL confirming co-author credit for the applicable memoir",
        parent=co_group,
        critical=True
    )

    if pick:
        idx, m = pick
        co_claim = f"The book '{m.title or ''}' lists {m.coauthor or ''} as a co-author."
        await evaluator.verify(
            claim=co_claim,
            node=co_leaf,
            sources=m.coauthor_urls,
            additional_instruction="Confirm that the page explicitly credits the named co-author for this book."
        )
    else:
        # Even if missing, we still attempt verification; auto preconditions will skip due to Coauthor_Present failure
        await evaluator.verify(
            claim="At least one memoir includes a co-author credited on a reliable page.",
            node=co_leaf,
            sources=None,
            additional_instruction="None"
        )


async def verify_global_award(
    evaluator: Evaluator,
    parent_node,
    memoirs: List[MemoirItem]
) -> None:
    award_group = evaluator.add_parallel(
        id="Global_Award_Check",
        desc="Verify that at least one of the four memoirs won or was a finalist for National Book Award or Pulitzer Prize",
        parent=parent_node,
        critical=True
    )

    pick = pick_first_with_award(memoirs)
    # We will build leaves regardless; the evidence leaf will ground the claims
    # If missing award info, claims will likely fail, keeping the node strict.

    # Award Type
    award_type_leaf = evaluator.add_leaf(
        id="Award_Type",
        desc="Identify the specific award (National Book Award or Pulitzer Prize)",
        parent=award_group,
        critical=True
    )
    # Award Year
    award_year_leaf = evaluator.add_leaf(
        id="Award_Year",
        desc="Provide the year the award was given",
        parent=award_group,
        critical=True
    )
    # Award Category
    award_cat_leaf = evaluator.add_leaf(
        id="Award_Category",
        desc="Provide the award category (e.g., Nonfiction, Biography, Memoir)",
        parent=award_group,
        critical=True
    )
    # Award Status
    award_status_leaf = evaluator.add_leaf(
        id="Award_Status",
        desc="Specify whether the memoir won the award or was a finalist",
        parent=award_group,
        critical=True
    )
    # Award URL (evidence)
    award_url_leaf = evaluator.add_leaf(
        id="Award_URL",
        desc="Provide reference URL confirming award status",
        parent=award_group,
        critical=True
    )

    if pick:
        idx, m = pick
        aw = m.award or AwardInfo()
        # Claims grounded by award_urls where possible
        type_claim = f"The memoir '{m.title or ''}' is associated with the '{aw.award_type or ''}'."
        await evaluator.verify(
            claim=type_claim,
            node=award_type_leaf,
            sources=(aw.award_urls or []),
            additional_instruction="Confirm the page states the award type for this memoir (National Book Award or Pulitzer Prize)."
        )

        year_claim = f"The award year for '{m.title or ''}' is '{aw.award_year or ''}'."
        await evaluator.verify(
            claim=year_claim,
            node=award_year_leaf,
            sources=(aw.award_urls or []),
            additional_instruction="Confirm the page shows the award year for this memoir."
        )

        cat_claim = f"The award category for '{m.title or ''}' is '{aw.award_category or ''}'."
        await evaluator.verify(
            claim=cat_claim,
            node=award_cat_leaf,
            sources=(aw.award_urls or []),
            additional_instruction="Confirm the page shows the award category for this memoir."
        )

        status_claim = f"The memoir '{m.title or ''}' was a '{aw.award_status or ''}' for the {aw.award_type or ''}."
        await evaluator.verify(
            claim=status_claim,
            node=award_status_leaf,
            sources=(aw.award_urls or []),
            additional_instruction="Confirm the page states whether the memoir won or was a finalist."
        )

        # Evidence leaf verifying combined details
        evidence_claim = (
            f"The provided award page(s) confirm that '{m.title or ''}' "
            f"received status '{aw.award_status or ''}' for the {aw.award_type or ''} "
            f"in {aw.award_year or ''} (category: {aw.award_category or ''})."
        )
        await evaluator.verify(
            claim=evidence_claim,
            node=award_url_leaf,
            sources=(aw.award_urls or []),
            additional_instruction="The page(s) should substantiate award type, year, category, and status for this memoir."
        )
    else:
        # No award present; attempt generic claims but will fail/skip under critical grouping
        await evaluator.verify(
            claim="At least one memoir has a verified major literary award (National Book Award or Pulitzer Prize).",
            node=award_type_leaf,
            sources=None,
            additional_instruction="None"
        )
        await evaluator.verify(
            claim="The award year is provided and verified on a reliable page.",
            node=award_year_leaf,
            sources=None,
            additional_instruction="None"
        )
        await evaluator.verify(
            claim="The award category is provided and verified on a reliable page.",
            node=award_cat_leaf,
            sources=None,
            additional_instruction="None"
        )
        await evaluator.verify(
            claim="The award status (Winner or Finalist) is provided and verified on a reliable page.",
            node=award_status_leaf,
            sources=None,
            additional_instruction="None"
        )
        await evaluator.verify(
            claim="A reference page confirms the award details.",
            node=award_url_leaf,
            sources=None,
            additional_instruction="None"
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
    Evaluate an answer for the Celebrity Memoirs Big Five task.
    """
    # Initialize evaluator
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

    # Add a task-level node under root to mirror rubric structure
    task_node = evaluator.add_parallel(
        id="Celebrity_Memoirs_Task",
        desc="Find four celebrity memoirs published between 2017-2025, each from a different Big Five publisher, meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Extract memoirs
    extraction = await evaluator.extract(
        prompt=prompt_extract_memoirs(),
        template_class=MemoirExtraction,
        extraction_name="memoirs_extraction"
    )

    # Pad/Trim to four memoirs
    memoirs: List[MemoirItem] = list(extraction.memoirs[:4])
    while len(memoirs) < 4:
        memoirs.append(MemoirItem())

    # Publisher distinctness flags
    distinct_flags = distinct_publisher_flags(memoirs)

    # Add Big Five info to summary
    evaluator.add_ground_truth({
        "big_five_publishers": [
            "Penguin Random House",
            "Macmillan",
            "HarperCollins",
            "Hachette",
            "Simon & Schuster"
        ],
        "date_range": "2017–2025"
    }, gt_type="reference_info")

    # Verify each memoir
    for i in range(4):
        await verify_single_memoir(
            evaluator=evaluator,
            parent_node=task_node,
            memoir=memoirs[i],
            idx=i,
            is_distinct_publisher=distinct_flags[i],
        )

    # Global checks: co-author and award
    await verify_global_coauthor(evaluator, task_node, memoirs)
    await verify_global_award(evaluator, task_node, memoirs)

    # Return structured result
    return evaluator.get_summary()