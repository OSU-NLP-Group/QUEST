import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nature_nov2025_space"
TASK_DESCRIPTION = (
    "Identify one original research article published in the main Nature journal during November 2025 that focuses on planetary science, astronomy, or space exploration. "
    "The article must meet the following requirements: (1) it must be an original research article published in the main Nature journal (not a Nature sub-journal such as Nature Communications or Nature Geoscience); "
    "(2) it must have been published in November 2025; (3) it must be an original research article, not a commentary, news & views, perspective, or review; "
    "(4) it must involve international collaboration with authors from at least two different countries; (5) at least one author must be affiliated with a U.S.-based research institution. "
    "Provide the following information: the full article title, complete list of all authors, all author affiliations including institution names and countries, the exact publication date, the DOI, and a direct URL to the article page on Nature's website. "
    "All information must be verifiable from the provided article URL."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Affiliation(BaseModel):
    """Affiliation entry for an author."""
    author: Optional[str] = None
    institution: Optional[str] = None
    country: Optional[str] = None
    full_text: Optional[str] = None


class ArticleExtraction(BaseModel):
    """Structured metadata for a single Nature article."""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    affiliations: List[Affiliation] = Field(default_factory=list)
    publication_date: Optional[str] = None  # Allow flexible formats like "12 November 2025" or "2025-11-12"
    doi: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None           # Expected to be "Nature" (main journal)
    article_type: Optional[str] = None      # e.g., "Article", "Letter", "Research Article"
    topics: List[str] = Field(default_factory=list)  # keywords/themes from answer if provided


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_article() -> str:
    return (
        "Extract exactly one article mentioned in the answer that aims to meet the requirements. If multiple candidates are provided, choose the first one that best matches. "
        "Return a JSON object containing these fields:\n"
        "1. title: The full article title as stated in the answer.\n"
        "2. authors: A complete list (array) of all authors as stated in the answer; preserve order if given.\n"
        "3. affiliations: An array of objects; each object must include:\n"
        "   - author: The author name this affiliation belongs to.\n"
        "   - institution: The institution name.\n"
        "   - country: The country of the institution.\n"
        "   - full_text: Optional raw text of the affiliation if provided.\n"
        "   Provide at least one affiliation per author if available in the answer; if missing for any author, include none for that author.\n"
        "4. publication_date: The exact publication date string for the article (e.g., '12 November 2025' or '2025-11-12').\n"
        "5. doi: The DOI string (e.g., '10.xxxx/xxxxx').\n"
        "6. url: A direct URL to the article page on nature.com (main Nature journal page for the article).\n"
        "7. journal: The journal name string (e.g., 'Nature').\n"
        "8. article_type: The article type (e.g., 'Article', 'Letter', 'Research Article'; must not be 'Commentary', 'News & Views', 'Perspective', or 'Review').\n"
        "9. topics: An array of keywords or phrases indicating the topical domain (e.g., planetary science, astronomy, space exploration) if present in the answer.\n"
        "If any field is not present in the answer, return null for that field (or an empty list for arrays). Follow the special URL extraction rules: extract only URLs explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def map_affiliations_by_author(affiliations: List[Affiliation]) -> Dict[str, List[Affiliation]]:
    mapping: Dict[str, List[Affiliation]] = {}
    for aff in affiliations:
        key = (aff.author or "").strip()
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(aff)
    return mapping


def is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def is_valid_doi_format(doi: Optional[str]) -> bool:
    if not is_nonempty_str(doi):
        return False
    # Basic DOI format check: starts with "10." followed by digits and a slash, with non-space suffix
    pattern = r"^10\.\d{4,9}/\S+$"
    return re.match(pattern, doi.strip()) is not None


def countries_from_affiliations(affiliations: List[Affiliation]) -> List[str]:
    countries: List[str] = []
    for aff in affiliations:
        c = (aff.country or "").strip()
        if c:
            countries.append(c)
    return list(dict.fromkeys(countries))  # preserve order, remove duplicates


def has_us_country(countries: List[str]) -> bool:
    us_aliases = {"united states", "united states of america", "usa", "u.s.a.", "u.s.", "us", "america"}
    return any((c or "").strip().lower() in us_aliases for c in countries)


def affiliations_claim_text(authors: List[str], affiliations: List[Affiliation]) -> str:
    # Build a concise listing of authors and their affiliation pairs to verify on the article page.
    by_author = map_affiliations_by_author(affiliations)
    segments: List[str] = []
    for author in authors:
        affs = by_author.get(author, [])
        if not affs:
            segments.append(f"{author}: (no affiliations provided)")
        else:
            parts = []
            for a in affs:
                inst = a.institution or ""
                country = a.country or ""
                parts.append(f"{inst}, {country}".strip(", ").strip())
            segments.append(f"{author}: " + "; ".join(parts))
    return "Authors and affiliations: " + " | ".join(segments)


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_article_page_url_group(
    evaluator: Evaluator,
    parent_node,
    extracted: ArticleExtraction
) -> Tuple[Optional[str], Optional[Any], Optional[Any]]:
    """
    Build the 'article_page_url' critical group:
    - url_provided (custom existence)
    - url_points_to_nature_article_page (verify by URL)
    Returns (url, url_provided_node, url_access_node) for use as dependencies.
    """
    group = evaluator.add_parallel(
        id="article_page_url",
        desc="A direct Nature article-page URL is provided and is accessible.",
        parent=parent_node,
        critical=True
    )

    # Existence check: URL provided
    url_provided_node = evaluator.add_custom_node(
        result=is_nonempty_str(extracted.url),
        id="url_provided",
        desc="Direct URL to the Nature article page is provided.",
        parent=group,
        critical=True
    )

    # Verify that the URL points to an accessible article page on nature.com
    url_access_node = evaluator.add_leaf(
        id="url_points_to_nature_article_page",
        desc="URL points to an accessible article page on nature.com.",
        parent=group,
        critical=True
    )
    url_to_check = extracted.url or ""
    await evaluator.verify(
        claim="This URL is an accessible article page hosted on nature.com (Nature main site).",
        node=url_access_node,
        sources=url_to_check,
        additional_instruction=(
            "Confirm the page is accessible and represents a Nature article page (has article metadata, DOI, title, authors). "
            "Do not accept search pages, landing pages, or Nature sub-journal homepages."
        )
    )

    return (extracted.url, url_provided_node, url_access_node)


async def build_article_constraints_group(
    evaluator: Evaluator,
    parent_node,
    article_url: Optional[str],
    prereqs: List[Any]
) -> None:
    """
    Build the 'article_constraints' critical group:
    - main_nature_journal
    - november_2025
    - original_research_article
    - topic_relevance
    """
    group = evaluator.add_parallel(
        id="article_constraints",
        desc="The chosen article satisfies venue, date, type, and topical-domain constraints.",
        parent=parent_node,
        critical=True
    )

    # Main Nature journal (not sub-journal)
    main_nature_node = evaluator.add_leaf(
        id="main_nature_journal",
        desc="Article is published in the main Nature journal (not a Nature sub-journal such as Nature Communications or Nature Geoscience).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="This article is published in the main 'Nature' journal and not in any Nature sub-journal.",
        node=main_nature_node,
        sources=article_url,
        additional_instruction=(
            "Check the journal branding and metadata on the page. Reject pages that indicate sub-journals like "
            "Nature Communications, Nature Geoscience, Nature Astronomy, Nature Physics, Nature Medicine, Scientific Reports, etc."
        ),
        extra_prerequisites=prereqs
    )

    # Publication date in November 2025
    nov2025_node = evaluator.add_leaf(
        id="november_2025",
        desc="Article publication date is in November 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="This article was published in November 2025.",
        node=nov2025_node,
        sources=article_url,
        additional_instruction="Verify the publication date month and year on the Nature article page.",
        extra_prerequisites=prereqs
    )

    # Original research article (not commentary, News & Views, perspective, or review)
    research_type_node = evaluator.add_leaf(
        id="original_research_article",
        desc="Article is an original research article (not commentary, News & Views, perspective, or review).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="This page represents an original research article, such as 'Article' or 'Letter', and is not a commentary, News & Views, perspective, or review.",
        node=research_type_node,
        sources=article_url,
        additional_instruction=(
            "On Nature, original research articles are typically labelled 'Article' or 'Letter'. "
            "Reject 'Commentary', 'News & Views', 'Perspective', 'Review', or editorial content."
        ),
        extra_prerequisites=prereqs
    )

    # Topic relevance: planetary science, astronomy, or space exploration
    topic_node = evaluator.add_leaf(
        id="topic_relevance",
        desc="Article topic is related to planetary science, astronomy, or space exploration.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The article focuses on planetary science, astronomy, or space exploration.",
        node=topic_node,
        sources=article_url,
        additional_instruction=(
            "Check the abstract, title, subject categories, or body text for clear relevance to planetary science, astronomy, "
            "or space exploration."
        ),
        extra_prerequisites=prereqs
    )


async def build_collaboration_constraints_group(
    evaluator: Evaluator,
    parent_node,
    article_url: Optional[str],
    extracted: ArticleExtraction,
    prereqs: List[Any]
) -> None:
    """
    Build the 'collaboration_constraints' critical group:
    - international_collaboration
    - us_institution_present
    """
    group = evaluator.add_parallel(
        id="collaboration_constraints",
        desc="The article satisfies international collaboration and U.S.-affiliation constraints.",
        parent=parent_node,
        critical=True
    )

    # International collaboration: at least two different countries in affiliations
    unique_countries = countries_from_affiliations(extracted.affiliations)
    intl_node = evaluator.add_leaf(
        id="international_collaboration",
        desc="Author affiliations include institutions from at least two different countries.",
        parent=group,
        critical=True
    )
    intl_claim = (
        f"The author affiliations span at least two different countries. Reported countries include: {', '.join(unique_countries)}."
        if unique_countries else
        "The author affiliations span at least two different countries."
    )
    await evaluator.verify(
        claim=intl_claim,
        node=intl_node,
        sources=article_url,
        additional_instruction=(
            "Inspect the affiliations section. Count distinct countries across all affiliations. "
            "Pass only if the set includes at least two unique countries."
        ),
        extra_prerequisites=prereqs
    )

    # At least one U.S.-based institution present
    us_node = evaluator.add_leaf(
        id="us_institution_present",
        desc="At least one author is affiliated with a U.S.-based research institution.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one author has an affiliation in the United States (USA).",
        node=us_node,
        sources=article_url,
        additional_instruction=(
            "Look for 'United States', 'USA', 'U.S.', or 'US' in affiliation country text or institutional location. "
            "Examples: NASA, JPL/Caltech, U.S. universities and laboratories."
        ),
        extra_prerequisites=prereqs
    )


async def build_required_metadata_group(
    evaluator: Evaluator,
    parent_node,
    article_url: Optional[str],
    extracted: ArticleExtraction,
    prereqs: List[Any]
) -> None:
    """
    Build the 'required_metadata_and_verifiability' critical group with leaf checks:
    - Full title provided + verifiable
    - Complete author list provided + verifiable
    - Exact publication date provided + verifiable
    - DOI provided + valid format + verifiable
    - Affiliations coverage and details provided + verifiable
    """
    group = evaluator.add_parallel(
        id="required_metadata_and_verifiability",
        desc="All required metadata fields are provided and each is verifiable from the Nature article-page URL.",
        parent=parent_node,
        critical=True
    )

    # Title existence
    title_provided = evaluator.add_custom_node(
        result=is_nonempty_str(extracted.title),
        id="full_title_provided",
        desc="Full article title is provided.",
        parent=group,
        critical=True
    )

    # Title verifiable from URL
    title_verify_node = evaluator.add_leaf(
        id="full_title_verifiable_from_url",
        desc="Provided article title matches/verifiable from the Nature article page.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The article title is '{(extracted.title or '').strip()}'.",
        node=title_verify_node,
        sources=article_url,
        additional_instruction=(
            "Match the full title shown on the Nature article page. Allow minor punctuation or casing differences."
        ),
        extra_prerequisites=prereqs + [title_provided]
    )

    # Author list provided
    authors_provided = evaluator.add_custom_node(
        result=bool(extracted.authors) and all(is_nonempty_str(a) for a in extracted.authors),
        id="complete_author_list_provided",
        desc="A complete list of all authors is provided.",
        parent=group,
        critical=True
    )

    # Author list verifiable
    authors_verify_node = evaluator.add_leaf(
        id="author_list_verifiable_from_url",
        desc="Provided author list matches/verifiable from the Nature article page.",
        parent=group,
        critical=True
    )
    author_list_text = ", ".join([a.strip() for a in extracted.authors]) if extracted.authors else ""
    await evaluator.verify(
        claim=f"The complete author list is: {author_list_text}.",
        node=authors_verify_node,
        sources=article_url,
        additional_instruction=(
            "Verify that each listed author appears on the article page. Allow minor variations such as middle initials, "
            "name ordering, accents, or punctuation."
        ),
        extra_prerequisites=prereqs + [authors_provided]
    )

    # Publication date provided
    pubdate_provided = evaluator.add_custom_node(
        result=is_nonempty_str(extracted.publication_date),
        id="exact_publication_date_provided",
        desc="Exact publication date (day, month, year) is provided.",
        parent=group,
        critical=True
    )

    # Publication date verifiable
    pubdate_verify_node = evaluator.add_leaf(
        id="publication_date_verifiable_from_url",
        desc="Provided publication date matches/verifiable from the Nature article page.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date is '{(extracted.publication_date or '').strip()}'.",
        node=pubdate_verify_node,
        sources=article_url,
        additional_instruction=(
            "Match the exact publication date string shown on the article page. Allow reasonable format variations "
            "like '12 November 2025' vs '2025-11-12'."
        ),
        extra_prerequisites=prereqs + [pubdate_provided]
    )

    # DOI provided (existence)
    doi_provided_node = evaluator.add_custom_node(
        result=is_nonempty_str(extracted.doi),
        id="doi_provided",
        desc="A DOI is provided.",
        parent=group,
        critical=True
    )

    # DOI valid format (custom, not LLM)
    doi_valid_node = evaluator.add_custom_node(
        result=is_valid_doi_format(extracted.doi),
        id="doi_valid_format",
        desc="Provided DOI is in a valid DOI format (e.g., 10.xxxx/xxxxx).",
        parent=group,
        critical=True
    )

    # DOI verifiable from URL
    doi_verify_node = evaluator.add_leaf(
        id="doi_verifiable_from_url",
        desc="Provided DOI matches/verifiable from the Nature article page.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The DOI for this article is '{(extracted.doi or '').strip()}'.",
        node=doi_verify_node,
        sources=article_url,
        additional_instruction="Confirm the DOI displayed on the Nature article page exactly matches.",
        extra_prerequisites=prereqs + [doi_provided_node, doi_valid_node]
    )

    # Affiliations group (critical parallel sub-node)
    aff_group = evaluator.add_parallel(
        id="affiliations_for_all_authors",
        desc="Affiliations for all authors are provided with required details and verifiable from the Nature article page.",
        parent=group,
        critical=True
    )

    # Coverage: every listed author has at least one affiliation provided
    by_author = map_affiliations_by_author(extracted.affiliations)
    coverage_all = bool(extracted.authors) and all(len(by_author.get(a, [])) > 0 for a in extracted.authors)
    evaluator.add_custom_node(
        result=coverage_all,
        id="affiliation_coverage_all_authors",
        desc="Every listed author has at least one affiliation provided.",
        parent=aff_group,
        critical=True
    )

    # Institution names present
    institutions_present = bool(extracted.affiliations) and all(is_nonempty_str(aff.institution) for aff in extracted.affiliations)
    evaluator.add_custom_node(
        result=institutions_present,
        id="institution_names_present",
        desc="Each provided affiliation includes an institution name.",
        parent=aff_group,
        critical=True
    )

    # Countries present
    countries_present_flag = bool(extracted.affiliations) and all(is_nonempty_str(aff.country) for aff in extracted.affiliations)
    evaluator.add_custom_node(
        result=countries_present_flag,
        id="countries_present",
        desc="Each provided affiliation includes a country.",
        parent=aff_group,
        critical=True
    )

    # Affiliations verifiable from URL
    aff_verify_node = evaluator.add_leaf(
        id="affiliations_verifiable_from_url",
        desc="Provided affiliations (including institution names and countries) match/verifiable from the Nature article page.",
        parent=aff_group,
        critical=True
    )
    aff_claim = affiliations_claim_text(extracted.authors, extracted.affiliations)
    await evaluator.verify(
        claim=aff_claim,
        node=aff_verify_node,
        sources=article_url,
        additional_instruction=(
            "For each author, verify that at least one listed institution-country pair appears in the affiliations on the Nature article page. "
            "Allow minor name/country formatting differences."
        ),
        extra_prerequisites=prereqs + [authors_provided]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Nature November 2025 planetary/astronomy/space article task.
    """
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
        default_model=model
    )

    # Extract structured article information from the answer
    extracted: ArticleExtraction = await evaluator.extract(
        prompt=prompt_extract_article(),
        template_class=ArticleExtraction,
        extraction_name="article_extraction"
    )

    # Build a critical top-level node representing the rubric root (since Evaluator's root is non-critical)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify one original research article in the main Nature journal published in November 2025 on planetary science/astronomy/space exploration, with international collaboration and at least one U.S.-affiliated author, and provide all required metadata and links verifiable from the article page.",
        parent=root,
        critical=True
    )

    # Article page URL group
    article_url, url_provided_node, url_access_node = await build_article_page_url_group(
        evaluator, task_root, extracted
    )

    prereqs = [node for node in [url_provided_node, url_access_node] if node is not None]

    # Article constraints group
    await build_article_constraints_group(
        evaluator, task_root, article_url, prereqs
    )

    # Collaboration constraints group
    await build_collaboration_constraints_group(
        evaluator, task_root, article_url, extracted, prereqs
    )

    # Required metadata and verifiability group
    await build_required_metadata_group(
        evaluator, task_root, article_url, extracted, prereqs
    )

    # Return evaluation summary
    return evaluator.get_summary()