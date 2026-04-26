import asyncio
import logging
from typing import Any, Optional, List, Dict, Tuple, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs2024_multi_inst_papers"
TASK_DESCRIPTION = (
    "Find four distinct research papers that were published in major computer science conference proceedings in 2024 "
    "(specifically from NeurIPS 2024, ICML 2024, CVPR 2024, ACL 2024, EMNLP 2024, or AAAI 2024). Each paper must meet "
    "all of the following criteria:\n\n"
    "1. The paper must have at least 4 authors listed\n"
    "2. The authors must be affiliated with at least 3 different institutions (which can include universities, research laboratories, or companies)\n"
    "3. These 3 or more institutions must be located in at least 2 different countries\n"
    "4. The paper must be publicly accessible with a direct link to the full PDF (through sources such as arXiv, conference open access repositories, or institutional repositories)\n\n"
    "For each of the four papers, provide:\n"
    "- The paper title\n"
    "- The complete list of authors\n"
    "- The conference name and year\n"
    "- The affiliations of all authors, including the full institution names and their countries\n"
    "- A direct URL link to the publicly accessible full paper (PDF or open access version)"
)

ALLOWED_VENUES_SHORT = ["NeurIPS", "ICML", "CVPR", "ACL", "EMNLP", "AAAI"]
ALLOWED_VENUES_HINT = (
    "Allowed venues and synonyms include: "
    "NeurIPS (Neural Information Processing Systems; formerly NIPS), "
    "ICML (International Conference on Machine Learning; often via PMLR Proceedings of Machine Learning Research), "
    "CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition; CVF Open Access), "
    "ACL (Association for Computational Linguistics; ACL Anthology), "
    "EMNLP (Empirical Methods in Natural Language Processing; ACL Anthology), "
    "AAAI (AAAI Conference on Artificial Intelligence)."
)
AUTHORITATIVE_DOMAINS_HINT = (
    "Treat the following as authoritative bibliographic sources: "
    "dblp.org; scholar.google.com (Google Scholar); proceedings.mlr.press (PMLR); "
    "neurips.cc/paper_files and neurips.cc; openreview.net (for some NeurIPS entries); "
    "openaccess.thecvf.com (CVF Open Access); aclanthology.org (ACL/EMNLP); "
    "aaai.org or ojs.aaai.org (AAAI); official conference proceedings pages."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Affiliation(BaseModel):
    institution: Optional[str] = None
    country: Optional[str] = None


class AuthorInfo(BaseModel):
    name: Optional[str] = None
    affiliations: List[Affiliation] = Field(default_factory=list)


class PaperInfo(BaseModel):
    title: Optional[str] = None
    conference_name: Optional[str] = None
    conference_year: Optional[str] = None
    authors: List[AuthorInfo] = Field(default_factory=list)
    pdf_url: Optional[str] = None
    bib_sources: List[str] = Field(default_factory=list)


class PapersExtraction(BaseModel):
    papers: List[PaperInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return (
        "Extract up to four (4) distinct paper entries from the answer, preserving their order of appearance. "
        "If the answer includes more than four, keep only the first four. If fewer are present, return only those.\n\n"
        "For each paper, extract the following fields:\n"
        "1) title: The exact paper title text as given in the answer.\n"
        "2) conference_name: The conference name exactly as stated in the answer (e.g., 'NeurIPS', 'ICML', 'CVPR', 'ACL', 'EMNLP', 'AAAI').\n"
        "3) conference_year: The year string as it appears. If mentioned as '2024', include '2024'. Do not guess.\n"
        "4) authors: The complete list of authors provided. For each author, include:\n"
        "   - name: The author’s full name as presented.\n"
        "   - affiliations: A list of affiliations. For each affiliation, include:\n"
        "       * institution: Full institution/organization name\n"
        "       * country: Country name for that institution\n"
        "   If an author has multiple affiliations, include all as separate entries.\n"
        "5) pdf_url: A direct publicly accessible full-text URL (preferably a direct PDF link) as provided in the answer; "
        "   acceptable sources include arXiv PDF links, official conference open access PDF pages (e.g., CVF, ACL Anthology, PMLR), or institutional repositories. "
        "   If no such URL is explicitly present in the answer, set this to null. If a URL is missing http/https, prepend http://.\n"
        "6) bib_sources: All additional URLs in the answer that are intended to support bibliographic details "
        "(e.g., DBLP entry, Google Scholar, official conference proceedings page, ACL Anthology entry, CVF Open Access, PMLR page). "
        "Do not include the pdf_url again. If none, return an empty list.\n\n"
        "IMPORTANT:\n"
        "- Extract only what is explicitly present in the answer; do not invent or infer. If an item is missing, set it to null or an empty list as appropriate.\n"
        "- Keep author order exactly as provided.\n"
        "- Ensure URLs are validly formatted; if protocol is missing, prepend http://.\n"
        "- Return a JSON object with a top-level field 'papers' that is an array of paper objects."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _unique_nonempty(seq: List[Optional[str]]) -> Set[str]:
    return {(_norm(x)) for x in seq if x and _norm(x)}


def _paper_identity_key(p: PaperInfo) -> Tuple[str, str]:
    # Primary key: normalized title; Secondary: normalized pdf_url
    return (_norm(p.title), _norm(p.pdf_url))


def _author_names(p: PaperInfo) -> List[str]:
    return [a.name for a in p.authors if a and a.name]  # type: ignore


def _institutions_for_paper(p: PaperInfo) -> Set[str]:
    insts: Set[str] = set()
    for a in p.authors:
        for aff in a.affiliations:
            if aff.institution:
                insts.add(_norm(aff.institution))
    return {i for i in insts if i}


def _countries_for_paper(p: PaperInfo) -> Set[str]:
    countries: Set[str] = set()
    for a in p.authors:
        for aff in a.affiliations:
            if aff.country:
                countries.add(_norm(aff.country))
    return {c for c in countries if c}


def _all_authors_have_affil_with_country(p: PaperInfo) -> bool:
    # For each author, at least one affiliation has both institution and country
    if not p.authors:
        return False
    for a in p.authors:
        if not a.name:
            return False
        ok = False
        for aff in a.affiliations:
            if aff.institution and aff.country and _norm(aff.institution) and _norm(aff.country):
                ok = True
                break
        if not ok:
            return False
    return True


def _combined_urls(pdf_url: Optional[str], extra: List[str]) -> List[str]:
    urls: List[str] = []
    if pdf_url and _norm(pdf_url):
        urls.append(pdf_url)
    urls.extend([u for u in extra if u and _norm(u)])
    return urls


# --------------------------------------------------------------------------- #
# Verification per paper                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_paper(
    evaluator: Evaluator,
    parent_node,
    paper: PaperInfo,
    index: int,
) -> None:
    i = index + 1
    paper_node = evaluator.add_parallel(
        id=f"Paper_{i}",
        desc=f"Paper {i} meets all constraints and required fields are provided",
        parent=parent_node,
        critical=False  # Each paper contributes partial credit
    )

    # 1) Title provided
    title_provided = evaluator.add_custom_node(
        result=bool(paper.title and _norm(paper.title)),
        id=f"Paper_{i}_Title_Provided",
        desc="Paper title is provided",
        parent=paper_node,
        critical=True
    )

    # 2) Conference name and year provided (ensure 2024 explicitly present in the provided year)
    conf_year_provided = evaluator.add_custom_node(
        result=bool(paper.conference_name and _norm(paper.conference_name) and paper.conference_year and ("2024" in str(paper.conference_year))),
        id=f"Paper_{i}_Conference_Name_Year_Provided",
        desc="Conference name and year are explicitly provided in the answer",
        parent=paper_node,
        critical=True
    )

    # 3) Authors list provided
    author_names = _author_names(paper)
    authors_list_provided = evaluator.add_custom_node(
        result=bool(author_names and len(author_names) > 0),
        id=f"Paper_{i}_Authors_List_Provided",
        desc="Complete list of authors is provided",
        parent=paper_node,
        critical=True
    )

    # 4) Author count >= 4
    author_count_ok = evaluator.add_custom_node(
        result=bool(len(author_names) >= 4),
        id=f"Paper_{i}_Author_Count_At_Least_4",
        desc="Paper has at least 4 authors",
        parent=paper_node,
        critical=True
    )

    # 5) Affiliations with countries provided for all authors
    affil_with_countries_ok = evaluator.add_custom_node(
        result=_all_authors_have_affil_with_country(paper),
        id=f"Paper_{i}_Affiliations_With_Countries_Provided",
        desc="Affiliations for all authors are provided, including full institution names and institution countries",
        parent=paper_node,
        critical=True
    )

    # 6) At least 3 institutions across authors
    institutions = _institutions_for_paper(paper)
    at_least_3_insts = evaluator.add_custom_node(
        result=bool(len(institutions) >= 3),
        id=f"Paper_{i}_At_Least_3_Institutions",
        desc="Authors span at least 3 different institutions",
        parent=paper_node,
        critical=True
    )

    # 7) At least 2 countries represented
    countries = _countries_for_paper(paper)
    at_least_2_countries = evaluator.add_custom_node(
        result=bool(len(countries) >= 2),
        id=f"Paper_{i}_At_Least_2_Countries",
        desc="Those institutions are located in at least 2 different countries",
        parent=paper_node,
        critical=True
    )

    # 8) Public PDF link PROVIDED (existence gate) – added to improve clarity and gating
    pdf_link_provided = evaluator.add_custom_node(
        result=bool(paper.pdf_url and _norm(paper.pdf_url)),
        id=f"Paper_{i}_Public_PDF_Link_Provided",
        desc="A direct publicly accessible link to the full paper is provided (existence check)",
        parent=paper_node,
        critical=True
    )

    # 9) Public PDF link ACCESSIBLE (verification)
    pdf_access_leaf = evaluator.add_leaf(
        id=f"Paper_{i}_Public_PDF_Link_Provided_And_Accessible",
        desc="A direct publicly accessible link to the full paper (PDF or open-access version) is provided",
        parent=paper_node,
        critical=True
    )
    pdf_claim = (
        f"This URL provides the full text of the paper titled '{paper.title or ''}' in a publicly accessible form "
        f"(a direct PDF file or an open-access page with a working PDF download), without login or paywall."
    )

    # 10) Valid conference and 2024 proceedings paper
    valid_conf_leaf = evaluator.add_leaf(
        id=f"Paper_{i}_Valid_Conference_2024",
        desc="Paper is from NeurIPS/ICML/CVPR/ACL/EMNLP/AAAI and is a 2024 proceedings paper",
        parent=paper_node,
        critical=True
    )
    conf_claim = (
        f"The paper titled '{paper.title or ''}' was published as a proceedings paper at {paper.conference_name or ''} 2024, "
        f"and this venue is one of {', '.join(ALLOWED_VENUES_SHORT)}."
    )
    conf_sources = _combined_urls(paper.pdf_url, paper.bib_sources)

    # 11) Bibliographic info verifiable via authoritative source
    biblio_leaf = evaluator.add_leaf(
        id=f"Paper_{i}_Bibliographic_Info_Verifiable",
        desc="Bibliographic info (title/authors/affiliations/venue) is verifiable via at least one authoritative source (DBLP, Google Scholar, or official conference site)",
        parent=paper_node,
        critical=True
    )
    # Compose authors string (limit to avoid extremely long claims)
    shown_authors = [n for n in author_names][:10]
    biblio_claim = (
        f"At least one of the provided source URLs is an authoritative bibliographic page (e.g., DBLP, Google Scholar, or the official conference proceedings page) "
        f"for the paper titled '{paper.title or ''}', listing authors including {', '.join(shown_authors)}."
    )
    biblio_sources = paper.bib_sources if paper.bib_sources else conf_sources

    # Batch verify the three factual checks (they will automatically honor critical sibling preconditions)
    await evaluator.batch_verify([
        (pdf_claim, paper.pdf_url, pdf_access_leaf, "Verify the URL opens a full paper PDF or a page with an accessible PDF download; not just an abstract. It must be publicly accessible without login."),
        (conf_claim, conf_sources, valid_conf_leaf, f"Check the page indicates the paper is in {paper.conference_name or 'the specified venue'} in 2024. {ALLOWED_VENUES_HINT}"),
        (biblio_claim, biblio_sources, biblio_leaf, f"Confirm at least one URL is authoritative and lists the same title/authors. {AUTHORITATIVE_DOMAINS_HINT}"),
    ])


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator (root is non-critical to avoid strict child-critical constraint)
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

    # Extract structured data
    extracted: PapersExtraction = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction",
    )

    # Add custom info: allowed venues
    evaluator.add_custom_info(
        info={"allowed_venues": ALLOWED_VENUES_SHORT, "year_required": "2024"},
        info_type="constraints",
        info_name="global_constraints"
    )

    # Build Global Requirements node (critical)
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Global requirements applying across the entire set of returned papers",
        parent=root,
        critical=True
    )

    papers_list: List[PaperInfo] = extracted.papers or []
    first_four: List[PaperInfo] = papers_list[:4]

    # Four papers provided
    four_provided = evaluator.add_custom_node(
        result=bool(len(papers_list) >= 4),
        id="Four_Papers_Provided",
        desc="Answer provides four paper entries (Paper 1–Paper 4)",
        parent=global_node,
        critical=True
    )

    # Papers are distinct (use normalized titles and/or pdf urls)
    distinct_ok = False
    if len(first_four) == 4:
        keys = [_paper_identity_key(p) for p in first_four]
        # Filter keys where at least one of title or url is non-empty to avoid empty duplicates
        filtered_keys = [k for k in keys if k[0] or k[1]]
        distinct_ok = len(filtered_keys) == 4 and len(set(filtered_keys)) == 4
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Papers_Are_Distinct",
        desc="All four paper entries refer to distinct papers (e.g., different titles/identifiers)",
        parent=global_node,
        critical=True
    )

    # Ensure we have exactly 4 placeholders for downstream nodes
    while len(first_four) < 4:
        first_four.append(PaperInfo())

    # Create verification subtrees for each paper (parallel under root)
    # Using the same IDs as specified in rubric for clarity
    for idx in range(4):
        await verify_single_paper(evaluator, root, first_four[idx], idx)

    # Return summary
    return evaluator.get_summary()