import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_2024_publishers"
TASK_DESCRIPTION = (
    "Identify the winners of the 2024 National Book Awards across all five categories "
    "(Fiction, Nonfiction, Poetry, Translated Literature, and Young People's Literature). "
    "For each winning book, provide: (1) the complete book title, (2) the author's name "
    "(and translator name for Translated Literature), (3) the publisher or publishing imprint, "
    "and (4) whether that publisher is one of the Big Five publishing houses headquartered in New York City. "
    "The Big Five publishers are: Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, "
    "and Macmillan Publishers."
)

BIG_FIVE_LIST = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette Book Group",
    "Macmillan Publishers",
]

# Expected author and publisher information provided in rubric (used for verification)
EXPECTED_AUTHORS: Dict[str, str] = {
    "Fiction": "Percival Everett",
    "Nonfiction": "Jason De León",
    "Poetry": "Lena Khalaf Tuffaha",
    "Translated_Literature": "Yáng Shuāng-zǐ",
    "Young_Peoples_Literature": "Shifa Saltagi Safadi",
}

EXPECTED_PUBLISHERS: Dict[str, Optional[str]] = {
    "Fiction": "Doubleday",
    "Nonfiction": "Viking",
    "Poetry": "University of Akron Press",
    "Translated_Literature": None,  # Not specified in rubric; verify correctness via sources if available
    "Young_Peoples_Literature": "G.P. Putnam's Sons Books for Young Readers",
}

EXPECTED_TRANSLATOR: Optional[str] = "Lin King"  # For Translated Literature


# Optional mapping of common imprints to Big Five parents to assist classification (used in additional instruction)
KNOWN_IMPRINTS_BIG_FIVE: Dict[str, str] = {
    # Penguin Random House
    "doubleday": "Penguin Random House",
    "viking": "Penguin Random House",
    "g.p. putnam's sons": "Penguin Random House",
    "g.p. putnam's sons books for young readers": "Penguin Random House",
    "knopf": "Penguin Random House",
    "random house": "Penguin Random House",
    "riverhead": "Penguin Random House",
    "crown": "Penguin Random House",
    # Hachette Book Group
    "little, brown": "Hachette Book Group",
    "grand central": "Hachette Book Group",
    # Macmillan Publishers
    "farrar, straus and giroux": "Macmillan Publishers",
    "henry holt": "Macmillan Publishers",
    "st. martin's press": "Macmillan Publishers",
    "tor books": "Macmillan Publishers",
    # HarperCollins
    "harpercollins": "HarperCollins",
    "harper collins": "HarperCollins",
    "harper": "HarperCollins",
    "william morrow": "HarperCollins",
    "ecco": "HarperCollins",
    # Simon & Schuster
    "simon & schuster": "Simon & Schuster",
    "simon and schuster": "Simon & Schuster",
    "atria": "Simon & Schuster",
    "scribner": "Simon & Schuster",
    "gallery books": "Simon & Schuster",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CategoryEntry(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    translator: Optional[str] = None  # Only applicable to Translated Literature
    publisher_imprint: Optional[str] = None
    big_five_status: Optional[str] = None  # "yes" / "no" / "unknown" as claimed by the answer
    source_urls: List[str] = Field(default_factory=list)


class WinnersExtraction(BaseModel):
    fiction: Optional[CategoryEntry] = None
    nonfiction: Optional[CategoryEntry] = None
    poetry: Optional[CategoryEntry] = None
    translated_literature: Optional[CategoryEntry] = None
    young_people: Optional[CategoryEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winners() -> str:
    return (
        "Extract the 2024 National Book Awards winners information exactly as presented in the answer. "
        "Organize the result into five categories: fiction, nonfiction, poetry, translated_literature, and young_people. "
        "For each category, extract:\n"
        "1) title: the complete official book title (include subtitle if provided in the answer).\n"
        "2) author: the author name as stated in the answer.\n"
        "3) translator: ONLY for Translated Literature; return the translator name if provided; otherwise null.\n"
        "4) publisher_imprint: the publisher or publishing imprint as stated in the answer.\n"
        "5) big_five_status: the answer's claimed classification of whether the publisher/imprint belongs to the Big Five "
        "(use 'yes', 'no', or 'unknown'; do not infer if the answer does not explicitly state it).\n"
        "6) source_urls: all URLs explicitly associated with that category entry in the answer (official awards page, publisher page, etc.). "
        "Extract only valid URLs mentioned in the answer (plain URLs or markdown links). If none, return an empty list.\n"
        "Return null for any field/category that is not mentioned in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def classify_big_five_by_imprint(publisher_imprint: Optional[str]) -> Optional[bool]:
    """
    Attempt to classify whether the given publisher/imprint belongs to a Big Five house.
    Returns:
      True  -> belongs to Big Five
      False -> does not belong to Big Five
      None  -> unknown (not confidently recognized)
    """
    if not publisher_imprint:
        return None
    p = publisher_imprint.lower().strip()

    # Known true matches via imprint lookup
    for key, parent in KNOWN_IMPRINTS_BIG_FIVE.items():
        if key in p:
            if parent in BIG_FIVE_LIST:
                return True

    # Known negatives (heuristics)
    negatives = [
        "university press",
        "university of akron press",
        "graywolf press",
        "coffee house press",
        "milkweed editions",
        "duke university press",
        "algonquin books",
    ]
    for neg in negatives:
        if neg in p:
            return False

    return None


def covers_all_categories_check(extracted: WinnersExtraction) -> bool:
    """
    Check if the response includes entries for all five categories with essential fields.
    For Translated Literature, translator must be present as essential field too.
    """
    fic = extracted.fiction
    non = extracted.nonfiction
    poe = extracted.poetry
    tl = extracted.translated_literature
    ypl = extracted.young_people

    def has_title_author(entry: Optional[CategoryEntry]) -> bool:
        return bool(entry and normalize_str(entry.title) and normalize_str(entry.author))

    all_basic = all([
        has_title_author(fic),
        has_title_author(non),
        has_title_author(poe),
        has_title_author(ypl),
    ])

    tl_ok = tl is not None and normalize_str(tl.title) and normalize_str(tl.author) and normalize_str(tl.translator)

    return all_basic and tl_ok


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_category(
    evaluator: Evaluator,
    parent_node,
    category_id: str,           # e.g., "Fiction_Winner"
    leaf_prefix: str,           # e.g., "Fiction"
    entry: Optional[CategoryEntry],
    expected_author: Optional[str] = None,
    expected_publisher: Optional[str] = None,
    expected_translator: Optional[str] = None,
) -> None:
    """
    Build verification sub-tree for a specific category.
    The category node is non-critical for partial credit across categories; leaf checks are critical.
    """
    # Create category node
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=f"{leaf_prefix.replace('_', ' ')} category winner entry (scored independently for partial credit).",
        parent=parent_node,
        critical=False,
    )

    # Prepare values
    title = normalize_str(entry.title if entry else None)
    author = normalize_str(entry.author if entry else None)
    translator = normalize_str(entry.translator if entry else None)
    publisher = normalize_str(entry.publisher_imprint if entry else None)
    sources_list = entry.source_urls if entry else []

    # 1) Title correctness
    title_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Title",
        desc=f"Complete official book title for the {leaf_prefix.replace('_', ' ')} winner is correctly provided.",
        parent=cat_node,
        critical=True,
    )
    title_claim = (
        f"The official 2024 National Book Awards {leaf_prefix.replace('_', ' ')} winner has the title '{title}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Verify the book title against the provided sources if available (e.g., National Book Foundation winners page, publisher pages). "
            "Allow minor variations in punctuation or the presence/absence of a subtitle. If no sources are provided, rely on the answer context."
        ),
    )

    # 2) Author correctness - match expected author if provided in rubric
    author_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Author",
        desc=f"Author correctly identified as {expected_author}." if expected_author else "Author is correctly provided.",
        parent=cat_node,
        critical=True,
    )
    if expected_author:
        author_claim = (
            f"The author of the 2024 National Book Awards {leaf_prefix.replace('_', ' ')} winner is '{author}', "
            f"which matches the expected '{expected_author}'."
        )
    else:
        author_claim = (
            f"The author '{author}' for the 2024 National Book Awards {leaf_prefix.replace('_', ' ')} winner is correct."
        )
    await evaluator.verify(
        claim=author_claim,
        node=author_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Allow minor or reasonable variants (diacritics, middle initials, transliteration differences). "
            "Prefer evidence from the official awards page or publisher pages if provided."
        ),
    )

    # 3) Publisher / Imprint correctness
    pub_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Publisher_Imprint",
        desc=(
            f"Publisher or publishing imprint correctly identified as {expected_publisher}."
            if expected_publisher else
            "Publisher or publishing imprint for the winning book is provided and correct."
        ),
        parent=cat_node,
        critical=True,
    )
    if expected_publisher:
        pub_claim = (
            f"The publisher/imprint for the 2024 National Book Awards {leaf_prefix.replace('_', ' ')} winner "
            f"is '{publisher}', which matches the expected '{expected_publisher}'."
        )
    else:
        pub_claim = (
            f"The publisher/imprint '{publisher}' stated for the 2024 National Book Awards {leaf_prefix.replace('_', ' ')} "
            f"winner is correct."
        )
    await evaluator.verify(
        claim=pub_claim,
        node=pub_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Verify the publisher/imprint attribution using the provided sources if available. "
            "If no sources are provided, check internal consistency in the answer; allow minor naming variations."
        ),
    )

    # 4) Translator correctness (only for Translated Literature with expected translator)
    if expected_translator is not None:
        translator_node = evaluator.add_leaf(
            id=f"{leaf_prefix}_Translator",
            desc=f"Translator correctly identified as {expected_translator}.",
            parent=cat_node,
            critical=True,
        )
        translator_claim = (
            f"The translator of the 2024 National Book Awards {leaf_prefix.replace('_', ' ')} winner is '{translator}', "
            f"which matches the expected '{EXPECTED_TRANSLATOR}'."
        )
        await evaluator.verify(
            claim=translator_claim,
            node=translator_node,
            sources=sources_list if sources_list else None,
            additional_instruction=(
                "Allow minor or reasonable variants (diacritics, hyphenation, transliteration). "
                "Prefer evidence from the official awards page or publisher pages if provided."
            ),
        )

    # 5) Big Five status correctness
    bf_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Big_Five_Status",
        desc=(
            "Correctly determines whether the identified publisher/imprint belongs to one of the Big Five NYC-headquartered publishers (per provided list)."
        ),
        parent=cat_node,
        critical=True,
    )

    derived_bf = classify_big_five_by_imprint(publisher)
    if derived_bf is True:
        bf_claim = (
            f"The publisher/imprint '{publisher}' belongs to one of the Big Five publishers: {', '.join(BIG_FIVE_LIST)}."
        )
    elif derived_bf is False:
        bf_claim = (
            f"The publisher/imprint '{publisher}' does NOT belong to any of the Big Five publishers: {', '.join(BIG_FIVE_LIST)}."
        )
    else:
        # Unknown; ask verifier to decide using sources or general imprint relationships
        bf_claim = (
            f"It is correct to determine whether '{publisher}' belongs to one of the Big Five publishers: {', '.join(BIG_FIVE_LIST)}."
        )

    await evaluator.verify(
        claim=bf_claim,
        node=bf_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Use the provided Big Five list. If sources are available (publisher page or awards page), use them to determine imprint ownership. "
            "You may rely on common publishing knowledge about imprint-parent relationships when sources are absent. "
            "If the imprint clearly belongs to one of these parents (e.g., Doubleday/Viking/G.P. Putnam's → Penguin Random House), mark as Big Five; "
            "if it is a university press or an independent press (e.g., University of Akron Press), mark as not Big Five."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2024 National Book Awards winners and publisher analysis task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent scoring across categories with a critical coverage gate
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

    # Extract structured winners info
    winners = await evaluator.extract(
        prompt=prompt_extract_winners(),
        template_class=WinnersExtraction,
        extraction_name="nba_2024_winners",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_authors": EXPECTED_AUTHORS,
        "expected_publishers": EXPECTED_PUBLISHERS,
        "big_five_list": BIG_FIVE_LIST,
        "notes": "Titles are not specified in rubric; verification relies on sources if provided or internal consistency."
    })

    # Critical coverage check: ensure all five categories are present
    coverage_result = covers_all_categories_check(winners)
    evaluator.add_custom_node(
        result=coverage_result,
        id="Covers_All_Five_Categories",
        desc="Response includes entries for all five categories: Fiction, Nonfiction, Poetry, Translated Literature, and Young People's Literature.",
        parent=root,
        critical=True,
    )

    # Build verification trees per category (non-critical parents, critical leaves)
    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Fiction_Winner",
        leaf_prefix="Fiction",
        entry=winners.fiction,
        expected_author=EXPECTED_AUTHORS.get("Fiction"),
        expected_publisher=EXPECTED_PUBLISHERS.get("Fiction"),
        expected_translator=None,
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Nonfiction_Winner",
        leaf_prefix="Nonfiction",
        entry=winners.nonfiction,
        expected_author=EXPECTED_AUTHORS.get("Nonfiction"),
        expected_publisher=EXPECTED_PUBLISHERS.get("Nonfiction"),
        expected_translator=None,
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Poetry_Winner",
        leaf_prefix="Poetry",
        entry=winners.poetry,
        expected_author=EXPECTED_AUTHORS.get("Poetry"),
        expected_publisher=EXPECTED_PUBLISHERS.get("Poetry"),
        expected_translator=None,
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Translated_Literature_Winner",
        leaf_prefix="Translated_Literature",
        entry=winners.translated_literature,
        expected_author=EXPECTED_AUTHORS.get("Translated_Literature"),
        expected_publisher=EXPECTED_PUBLISHERS.get("Translated_Literature"),
        expected_translator=EXPECTED_TRANSLATOR,
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Young_Peoples_Literature_Winner",
        leaf_prefix="Young_Peoples_Literature",
        entry=winners.young_people,
        expected_author=EXPECTED_AUTHORS.get("Young_Peoples_Literature"),
        expected_publisher=EXPECTED_PUBLISHERS.get("Young_Peoples_Literature"),
        expected_translator=None,
    )

    # Return final structured summary
    return evaluator.get_summary()