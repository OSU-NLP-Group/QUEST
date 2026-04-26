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
TASK_ID = "memoir_2015_paperback_biblio"
TASK_DESCRIPTION = (
    "A prominent self-help author and motivational speaker, known for decades of teaching self-empowerment and writing over 40 bestselling books, "
    "published a memoir whose title is taken from a famous 1972 song lyric about achieving clarity of vision. Identify this memoir and provide complete "
    "bibliographic cataloging information for the paperback reprint edition published in 2015, including: "
    "(1) complete book title, (2) author's full name as it appears on the book, (3) publisher name, (4) exact publication date, (5) publication year, "
    "(6) ISBN-13, (7) ISBN-10 or ASIN, (8) page count, (9) format specification, (10) edition type, (11) publication language, "
    "(12) physical dimensions, and (13) item weight."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MemoirIdentification(BaseModel):
    title: Optional[str] = None
    author_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BibliographicInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    publication_year: Optional[str] = None
    isbn_13: Optional[str] = None
    isbn_10_or_asin: Optional[str] = None
    page_count: Optional[str] = None
    format: Optional[str] = None
    edition_type: Optional[str] = None
    language: Optional[str] = None
    dimensions: Optional[str] = None
    weight: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MemoirBibliographyExtraction(BaseModel):
    memoir: Optional[MemoirIdentification] = None
    bibliographic_2015_paperback_reprint: Optional[BibliographicInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_memoir_and_biblio() -> str:
    return """
    Identify the memoir described in the answer and extract the bibliographic details for the 2015 paperback reprint edition.
    
    Part 1: Memoir Identification
    - title: The memoir's title as stated in the answer.
    - author_name: The author's full name as given in the answer for this memoir.
    - sources: Collect all URLs cited in the answer that directly support the identity of the book (e.g., retailer pages, publisher pages, library catalogs, or other authoritative pages for this memoir).
    
    Part 2: Bibliographic (2015 Paperback Reprint Edition)
    Extract the following fields exactly as presented in the answer for the paperback reprint edition published in 2015:
    - title: Complete book title as presented for the 2015 paperback reprint edition.
    - author: Author's full name as credited on the book for this edition.
    - publisher: Publisher name.
    - publication_date: Exact publication date (the answer may format as MM/DD/YYYY; if a spelled-out month is shown in the cited page, the date should still correspond to the same day-month-year).
    - publication_year: The publication year for this paperback reprint edition (should be 2015).
    - isbn_13: The 13-digit ISBN for this paperback reprint edition (with or without hyphens).
    - isbn_10_or_asin: The ISBN-10 or ASIN for this paperback reprint edition.
    - page_count: The print length/page count for this edition.
    - format: Format specification for the edition (e.g., Paperback).
    - edition_type: The edition type (e.g., Reprint).
    - language: Publication language.
    - dimensions: Physical product dimensions for the print edition.
    - weight: Item weight for the print edition.
    - sources: Collect all URLs cited in the answer that support these bibliographic details for the 2015 paperback reprint edition. Do NOT invent URLs.
    
    IMPORTANT:
    - Only extract information explicitly present in the answer. If the answer doesn't provide a field, set it to null.
    - Only include URLs that appear in the answer text (plain links or markdown links). Ignore non-URL references.
    - Use string types for all values (e.g., '384 pages', '5.3 x 0.9 x 8 inches', '12 ounces', '10/13/2015', 'October 13, 2015', 'Hay House Inc.').
    - If multiple editions are mentioned, focus on the 2015 paperback reprint edition values.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_str(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def _merge_sources(*source_lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists, de-duplicate while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in source_lists:
        if not lst:
            continue
        for url in lst:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_identify_memoir(
    evaluator: Evaluator,
    parent_node,
    extraction: MemoirBibliographyExtraction
) -> None:
    """
    Build and run the identification verification leaf.
    """
    memoir = extraction.memoir or MemoirIdentification()
    memoir_title = _norm_str(memoir.title)
    memoir_author = _norm_str(memoir.author_name)
    all_sources = _merge_sources(memoir.sources, (extraction.bibliographic_2015_paperback_reprint or BibliographicInfo()).sources)

    identify_node = evaluator.add_leaf(
        id="identify_memoir",
        desc="Memoir is correctly identified based on the contextual clues (prominent self-help author; title from 1972 song lyric; memoir)",
        parent=parent_node,
        critical=True
    )

    # If title or author missing, fail immediately
    if not memoir_title or not memoir_author:
        identify_node.score = 0.0
        identify_node.status = "failed"
        return

    claim = (
        f"The identified memoir is titled '{memoir_title}' and authored by '{memoir_author}'. "
        f"Confirm this is a memoir by a prominent self-help author/motivational speaker."
    )

    add_ins = (
        "Use the provided URLs to confirm the book title/author and that it is indeed a memoir. "
        "You do not need to explicitly verify the 1972 lyric; just ensure the book identity is correct. "
        "Allow minor variations in author presentation (e.g., with/without 'Dr.', middle initials)."
    )

    await evaluator.verify(
        claim=claim,
        node=identify_node,
        sources=all_sources,
        additional_instruction=add_ins
    )


async def verify_bibliographic_block(
    evaluator: Evaluator,
    parent_node,
    extraction: MemoirBibliographyExtraction
) -> None:
    """
    Build the bibliographic info node and add all 13 leaves; verify them (mostly via URLs).
    """
    biblio = extraction.bibliographic_2015_paperback_reprint or BibliographicInfo()
    memoir = extraction.memoir or MemoirIdentification()
    all_sources = _merge_sources(biblio.sources, memoir.sources)

    biblio_node = evaluator.add_parallel(
        id="bibliographic_info_2015_paperback_reprint",
        desc="Provide complete bibliographic cataloging information for the paperback reprint edition published in 2015",
        parent=parent_node,
        critical=True
    )

    claims_to_verify: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    def add_field_leaf(field_id: str, desc: str, value: Optional[str], claim_text: str, additional_instruction: str) -> None:
        v = _norm_str(value)
        leaf = evaluator.add_leaf(
            id=field_id,
            desc=desc,
            parent=biblio_node,
            critical=True
        )
        if not v:
            leaf.score = 0.0
            leaf.status = "failed"
        else:
            # Fill in claim with the provided value when appropriate
            claim = claim_text.replace("{value}", v)
            claims_to_verify.append((claim, all_sources, leaf, additional_instruction))

    # 1) Title
    add_field_leaf(
        field_id="title",
        desc="Complete book title is provided exactly as it appears on the publication",
        value=biblio.title,
        claim_text="The complete book title for the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Match the exact title on the product/catalog/publisher page for the paperback (reprint) edition; allow trivial punctuation/case variants."
    )

    # 2) Author
    add_field_leaf(
        field_id="author",
        desc="Author's full name is provided as credited on the book",
        value=biblio.author,
        claim_text="The author's full name credited on the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Allow minor variants such as presence/absence of 'Dr.' or middle initials; focus on the credited author for the paperback reprint."
    )

    # 3) Publisher
    add_field_leaf(
        field_id="publisher",
        desc="Publisher name is accurate and complete",
        value=biblio.publisher,
        claim_text="The publisher of the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Allow minor corporate suffix variants (e.g., 'Inc', 'Inc.', 'Incorporated'); ensure it refers to the correct publisher for this paperback reprint."
    )

    # 4) Publication date (exact)
    add_field_leaf(
        field_id="publication_date",
        desc="Exact publication date is provided in month/day/year format",
        value=biblio.publication_date,
        claim_text="The exact publication date for the 2015 paperback reprint edition is {value}.",
        additional_instruction="Confirm the date matches the edition date on the page. If the page uses 'Month DD, YYYY', accept equivalence with the provided MM/DD/YYYY when normalized."
    )

    # 5) Publication year (should be 2015)
    # For this field, we ensure the claim reflects the provided value. The instruction enforces that correct year is 2015.
    pub_year_value = _norm_str(biblio.publication_year)
    pub_year_claim = "The publication year for the 2015 paperback reprint edition is '{value}'."
    add_field_leaf(
        field_id="publication_year",
        desc="Publication year is correctly specified as 2015 for the specified reprint edition",
        value=pub_year_value,
        claim_text=pub_year_claim,
        additional_instruction="This must be 2015 for the paperback reprint edition. Mark incorrect if the sources show a different year."
    )

    # 6) ISBN-13
    add_field_leaf(
        field_id="isbn_13",
        desc="ISBN-13 is provided in standard 13-digit format for the paperback reprint edition",
        value=biblio.isbn_13,
        claim_text="The ISBN-13 for the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Normalize hyphens/spaces when matching; digits must correspond to the ISBN-13 for the paperback reprint edition."
    )

    # 7) ISBN-10 or ASIN
    add_field_leaf(
        field_id="isbn_10_or_asin",
        desc="ISBN-10 or ASIN is provided in standard format",
        value=biblio.isbn_10_or_asin,
        claim_text="The ISBN-10 or ASIN for the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Match the ISBN-10 or ASIN from the authoritative page for the paperback reprint (2015). Allow hyphen/space normalization."
    )

    # 8) Page count
    add_field_leaf(
        field_id="page_count",
        desc="Page count accurately reflects the print length of the specified edition",
        value=biblio.page_count,
        claim_text="The page count (print length) of the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Confirm the print length/page count on the page matches the provided number (allow the word 'pages' to be present/absent)."
    )

    # 9) Format
    add_field_leaf(
        field_id="format",
        desc="Format specification accurately describes the physical format of the edition (paperback reprint edition)",
        value=biblio.format,
        claim_text="The format specification for the 2015 reprint edition is '{value}'.",
        additional_instruction="This should indicate 'Paperback' for the reprint edition. Mark incorrect if the listed format is not Paperback."
    )

    # 10) Edition type
    add_field_leaf(
        field_id="edition_type",
        desc="Edition type is correctly specified (reprint)",
        value=biblio.edition_type,
        claim_text="The edition type for the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="This should indicate 'Reprint' (or clear equivalent). If the page shows a different edition type, mark incorrect."
    )

    # 11) Language
    add_field_leaf(
        field_id="language",
        desc="Publication language is specified",
        value=biblio.language,
        claim_text="The publication language of the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Confirm the language shown on the page for the paperback reprint edition."
    )

    # 12) Dimensions
    add_field_leaf(
        field_id="dimensions",
        desc="Physical dimensions are provided for the print edition",
        value=biblio.dimensions,
        claim_text="The physical dimensions of the 2015 paperback reprint edition are '{value}'.",
        additional_instruction="Compare numeric dimensions and units; allow minor rounding/formatting differences, but values and units should align."
    )

    # 13) Weight
    add_field_leaf(
        field_id="weight",
        desc="Item weight is provided for the print edition",
        value=biblio.weight,
        claim_text="The item weight of the 2015 paperback reprint edition is '{value}'.",
        additional_instruction="Compare numeric weight and units; allow trivial unit formatting differences (e.g., 'oz' vs 'ounces')."
    )

    # Run URL-based verifications in parallel for all provided fields
    if claims_to_verify:
        await evaluator.batch_verify(claims_and_sources=claims_to_verify)


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
    Evaluate an answer for the memoir identification and 2015 paperback reprint bibliographic task.
    """
    # Initialize evaluator (root is non-critical by framework design). We'll add a critical main node under root.
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

    # Add a critical, sequential main node to reflect rubric's critical root requirement
    main_node = evaluator.add_sequential(
        id="main_evaluation",
        desc="Identify the memoir and verify complete 2015 paperback reprint bibliographic information",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_memoir_and_biblio(),
        template_class=MemoirBibliographyExtraction,
        extraction_name="memoir_bibliographic_extraction"
    )

    # 1) Identification (critical, first step)
    await verify_identify_memoir(
        evaluator=evaluator,
        parent_node=main_node,
        extraction=extraction
    )

    # 2) Bibliographic verification block (critical, parallel children)
    await verify_bibliographic_block(
        evaluator=evaluator,
        parent_node=main_node,
        extraction=extraction
    )

    # Return the structured result summary
    return evaluator.get_summary()