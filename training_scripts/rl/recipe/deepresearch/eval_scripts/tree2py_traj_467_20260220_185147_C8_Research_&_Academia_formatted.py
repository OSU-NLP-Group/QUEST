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
TASK_ID = "cs_conf_2026_papers"
TASK_DESCRIPTION = (
    "Identify four distinct research papers that were accepted at major computer science conferences held in 2026. "
    "For each paper, provide the following information:\n\n"
    "1. The name of the conference where the paper was accepted\n"
    "2. The complete title of the paper\n"
    "3. The full name of the first author\n"
    "4. The institutional affiliation of at least one author (include the name of the institution and its country)\n"
    "5. The primary research area or topic that the paper addresses\n"
    "6. A direct URL to the paper or its official conference acceptance page (this can be a link to the conference "
    "proceedings, arXiv, or another verifiable source)\n\n"
    "Additionally, ensure that:\n"
    "- The institutional affiliations across all four papers represent at least two different countries\n"
    "- The four papers collectively cover different research areas within computer science (such as machine learning, "
    "human-computer interaction, computer vision, natural language processing, systems, theory, or other recognized subfields)"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperItem(BaseModel):
    conference_name: Optional[str] = None
    conference_year: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    first_author: Optional[str] = None
    affiliation_institution: Optional[str] = None
    affiliation_country: Optional[str] = None
    research_area: Optional[str] = None
    paper_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class PapersExtraction(BaseModel):
    papers: List[PaperItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
Extract up to four distinct research papers accepted at major computer science conferences held in 2026 as presented in the answer.

For each paper in the answer, extract the following fields exactly as stated:
- conference_name: Name of the conference where the paper was accepted (e.g., "NeurIPS", "ICML", "CHI", "ACL", "SIGCOMM", "STOC", etc.).
- conference_year: The conference year/edition mentioned (should be "2026" if provided).
- title: The complete title of the paper exactly as written.
- authors: An array of full author names in order, exactly as presented.
- first_author: The full name of the first author (if missing in the answer, return null).
- affiliation_institution: The name of at least one author's institution (exactly as written in the answer).
- affiliation_country: The country corresponding to the provided institution (exactly as written in the answer).
- research_area: The paper’s primary research area (e.g., "machine learning", "human–computer interaction", "computer vision", "natural language processing", "systems", "theory", etc.).
- paper_url: A direct URL to the paper or official acceptance/proceedings page. If the answer includes a URL, include it here exactly.
- additional_urls: Any other URLs cited in the answer for this paper (e.g., additional acceptance/proceedings links, OpenReview, lab pages). Provide all that are associated with this paper.
    
Rules:
- Only extract up to the first 4 papers described in the answer. If fewer are present, return fewer.
- Do not invent any missing fields; use null for missing scalar fields and empty arrays for missing list fields.
- Include only URLs explicitly present in the answer text (plain URLs or markdown links).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _collect_urls(p: PaperItem) -> List[str]:
    urls: List[str] = []
    if p.paper_url and p.paper_url.strip():
        urls.append(p.paper_url.strip())
    urls.extend([u.strip() for u in (p.additional_urls or []) if isinstance(u, str) and u.strip()])
    return urls


def _safe_first_author(p: PaperItem) -> Optional[str]:
    if p.first_author and p.first_author.strip():
        return p.first_author.strip()
    if p.authors:
        first = p.authors[0].strip()
        return first if first else None
    return None


# --------------------------------------------------------------------------- #
# Verification for a single paper                                             #
# --------------------------------------------------------------------------- #
async def verify_single_paper(
    evaluator: Evaluator,
    parent_node,
    paper: PaperItem,
    index: int,
) -> None:
    # Group node for this paper (parallel aggregation; allow partial credit inside a paper)
    paper_group = evaluator.add_parallel(
        id=f"Paper_{index + 1}",
        desc=f"Paper #{index + 1} metadata and eligibility checks",
        parent=parent_node,
        critical=False
    )

    urls_all = _collect_urls(paper)
    has_urls = len(urls_all) > 0

    # 1) Conference & Year (critical)
    if paper.conference_name and has_urls:
        node_conf_year = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Conference_And_Year",
            desc="Provides the name of a major computer science conference and indicates it is the 2026 edition (conference held in 2026).",
            parent=paper_group,
            critical=True
        )
        claim_conf_year = (
            f"This page shows that the paper is associated with the {paper.conference_name} 2026 conference (i.e., the conference was held in 2026)."
        )
        await evaluator.verify(
            claim=claim_conf_year,
            node=node_conf_year,
            sources=urls_all,
            additional_instruction="Verify that the conference name appears and that 2026 is indicated on the page (e.g., in the header, proceedings metadata, or event name)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Conference_And_Year",
            desc="Provides the name of a major computer science conference and indicates it is the 2026 edition (conference held in 2026).",
            parent=paper_group,
            critical=True
        )

    # 2) Acceptance Status (critical)
    if has_urls and paper.conference_name:
        node_accept = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Acceptance_Status",
            desc="Provides evidence the paper is an officially accepted main-conference paper (not a rejected submission or a workshop paper).",
            parent=paper_group,
            critical=True
        )
        claim_accept = (
            f"This page indicates that the paper was officially accepted as a main-conference paper at {paper.conference_name} 2026 (not a workshop or poster)."
        )
        await evaluator.verify(
            claim=claim_accept,
            node=node_accept,
            sources=urls_all,
            additional_instruction="Confirm the page (or one of the provided pages) explicitly shows acceptance in the main conference track (e.g., proceedings listing, research track). Do not accept workshop-only or poster-only entries."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Acceptance_Status",
            desc="Provides evidence the paper is an officially accepted main-conference paper (not a rejected submission or a workshop paper).",
            parent=paper_group,
            critical=True
        )

    # 3) Title (critical)
    if paper.title and paper.title.strip() and has_urls:
        node_title = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Title",
            desc="Provides the complete title of the paper.",
            parent=paper_group,
            critical=True
        )
        claim_title = f"The title of the paper on this page is exactly or equivalently: '{paper.title}'."
        await evaluator.verify(
            claim=claim_title,
            node=node_title,
            sources=urls_all,
            additional_instruction="Allow minor punctuation/case variations but the substantive title should match exactly."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Title",
            desc="Provides the complete title of the paper.",
            parent=paper_group,
            critical=True
        )

    # 4) Author list with full names (critical)
    if paper.authors and has_urls:
        node_authors = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Author_List_Full_Names",
            desc="Provides (or links to) a clearly identifiable list of all authors, with full names.",
            parent=paper_group,
            critical=True
        )
        claim_authors = f"The page lists the authors as: {paper.authors} (full names)."
        await evaluator.verify(
            claim=claim_authors,
            node=node_authors,
            sources=urls_all,
            additional_instruction="Verify the listed authors on the page match the provided list (allow minor name variants like middle initials)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Author_List_Full_Names",
            desc="Provides (or links to) a clearly identifiable list of all authors, with full names.",
            parent=paper_group,
            critical=True
        )

    # 5) First author (critical)
    fa = _safe_first_author(paper)
    if fa and has_urls:
        node_fa = evaluator.add_leaf(
            id=f"Paper_{index + 1}_First_Author",
            desc="Identifies the first author of the paper by full name.",
            parent=paper_group,
            critical=True
        )
        claim_fa = f"The first author of the paper is {fa}."
        await evaluator.verify(
            claim=claim_fa,
            node=node_fa,
            sources=urls_all,
            additional_instruction="Verify that the first listed author on the page is the stated person (allow minor name variants like middle initials)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_First_Author",
            desc="Identifies the first author of the paper by full name.",
            parent=paper_group,
            critical=True
        )

    # 6) Affiliation with country (critical)
    if paper.affiliation_institution and paper.affiliation_country and has_urls:
        node_aff = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Affiliation_With_Country",
            desc="Provides at least one author's institutional affiliation including institution name and country.",
            parent=paper_group,
            critical=True
        )
        claim_aff = (
            f"At least one author is affiliated with '{paper.affiliation_institution}' in '{paper.affiliation_country}'."
        )
        await evaluator.verify(
            claim=claim_aff,
            node=node_aff,
            sources=urls_all,
            additional_instruction="Check the page (or provided sources) for affiliation details; accept if any author is shown with the stated institution and country."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Affiliation_With_Country",
            desc="Provides at least one author's institutional affiliation including institution name and country.",
            parent=paper_group,
            critical=True
        )

    # 7) Abstract present (critical)
    if has_urls:
        node_abs = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Abstract_Present",
            desc="Provides (or links to) an abstract as part of a standard academic conference paper format.",
            parent=paper_group,
            critical=True
        )
        claim_abs = "This page provides an abstract summarizing the paper."
        await evaluator.verify(
            claim=claim_abs,
            node=node_abs,
            sources=urls_all,
            additional_instruction="Confirm that there is an 'Abstract' section or equivalent summary text on the page."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Abstract_Present",
            desc="Provides (or links to) an abstract as part of a standard academic conference paper format.",
            parent=paper_group,
            critical=True
        )

    # 8) Research area (critical)
    if paper.research_area and has_urls:
        node_area = evaluator.add_leaf(
            id=f"Paper_{index + 1}_Research_Area",
            desc="Identifies the primary research area/topic of the paper.",
            parent=paper_group,
            critical=True
        )
        claim_area = f"The paper's primary research area is '{paper.research_area}'."
        await evaluator.verify(
            claim=claim_area,
            node=node_area,
            sources=urls_all,
            additional_instruction="Verify that the paper clearly falls under the stated area (allow synonyms or closely related labels)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_Research_Area",
            desc="Identifies the primary research area/topic of the paper.",
            parent=paper_group,
            critical=True
        )

    # 9) Direct URL (critical)
    if paper.paper_url and paper.paper_url.strip():
        node_url = evaluator.add_leaf(
            id=f"Paper_{index + 1}_URL",
            desc="Provides a direct, verifiable URL to the paper or official acceptance/proceedings page.",
            parent=paper_group,
            critical=True
        )
        claim_url = "This URL is a direct page for the paper or its official acceptance/proceedings entry."
        await evaluator.verify(
            claim=claim_url,
            node=node_url,
            sources=paper.paper_url.strip(),
            additional_instruction="Verify that the URL resolves to a direct paper page or official acceptance/proceedings entry (e.g., conference site, arXiv, OpenReview)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Paper_{index + 1}_URL",
            desc="Provides a direct, verifiable URL to the paper or official acceptance/proceedings page.",
            parent=paper_group,
            critical=True
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

    # Extract up to four papers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction"
    )

    papers: List[PaperItem] = (extracted.papers or [])[:4]
    # Pad to exactly 4 items with empty placeholders if fewer provided
    while len(papers) < 4:
        papers.append(PaperItem())

    # Root node for the task
    root_node = evaluator.add_parallel(
        id="Find_4_Conference_Papers",
        desc="Identify four distinct research papers accepted at major computer science conferences held in 2026; provide required metadata for each; satisfy cross-paper diversity constraints.",
        parent=root,
        critical=False
    )

    # Add per-paper verification subtrees
    for i in range(4):
        await verify_single_paper(evaluator, root_node, papers[i], i)

    # Cross-paper validations (critical)
    # 1) Papers are distinct (no duplicate titles or URLs; require four non-empty titles and URLs to pass)
    titles = [p.title.strip() for p in papers if p.title and p.title.strip()]
    urls = [p.paper_url.strip() for p in papers if p.paper_url and p.paper_url.strip()]
    titles_unique = len(titles) == 4 and len(set(_normalize(t) for t in titles)) == 4
    urls_unique = len(urls) == 4 and len(set(_normalize(u) for u in urls)) == 4
    evaluator.add_custom_node(
        result=(titles_unique and urls_unique),
        id="Papers_Are_Distinct",
        desc="Confirms the four identified papers are distinct (e.g., not the same paper repeated; titles/URLs are not duplicates).",
        parent=root_node,
        critical=True
    )

    # 2) Geographic diversity: at least two distinct countries across the four papers
    countries = [p.affiliation_country.strip() for p in papers if p.affiliation_country and p.affiliation_country.strip()]
    country_set = set(_normalize(c) for c in countries)
    evaluator.add_custom_node(
        result=(len(country_set) >= 2),
        id="Geographic_Diversity",
        desc="Across the four papers, the provided affiliations include at least two distinct countries.",
        parent=root_node,
        critical=True
    )

    # 3) Research area diversity: all four primary areas must be different
    areas = [p.research_area.strip() for p in papers if p.research_area and p.research_area.strip()]
    areas_norm = [_normalize(a) for a in areas]
    evaluator.add_custom_node(
        result=(len(areas_norm) == 4 and len(set(areas_norm)) == 4),
        id="Research_Area_Diversity",
        desc="Across the four papers, the primary research areas are all different (no two papers share the same primary area).",
        parent=root_node,
        critical=True
    )

    # Optional: record a concise summary of extracted metadata for debugging
    summary_info = {
        "papers_extracted_count": len([p for p in papers if p.title]),
        "conference_names": [p.conference_name for p in papers],
        "conference_years": [p.conference_year for p in papers],
        "titles": [p.title for p in papers],
        "first_authors": [_safe_first_author(p) for p in papers],
        "affiliations": [{"institution": p.affiliation_institution, "country": p.affiliation_country} for p in papers],
        "research_areas": [p.research_area for p in papers],
        "paper_urls": [p.paper_url for p in papers],
    }
    evaluator.add_custom_info(summary_info, info_type="extraction_summary", info_name="extraction_summary")

    return evaluator.get_summary()