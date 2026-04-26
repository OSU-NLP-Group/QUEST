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
TASK_ID = "lamniform_nt_aptian_2025"
TASK_DESCRIPTION = (
    "Identify a peer-reviewed scientific paper published between October 1, 2025 and November 30, 2025, "
    "that reports the discovery of lamniform shark fossils from the Northern Territory of Australia. "
    "The fossils must be dated to the Aptian age of the Cretaceous period (approximately 113-125 million years ago), "
    "and the primary specimens described must be vertebral centra (vertebrae). The paper must provide numerical body "
    "length estimates in meters for the extinct shark and must employ regression modeling techniques to estimate body "
    "size from fossil measurements. Additionally, the research must include publicly available data repositories "
    "(such as GitHub, Figshare, Zenodo, Dryad, or similar platforms) containing the research data or analysis code. "
    "Provide the following information about the paper: (1) the full title of the paper, (2) the complete author list, "
    "(3) the journal name and publication date, (4) the DOI (Digital Object Identifier), and (5) a URL to access the "
    "paper or its abstract."
)

DATE_WINDOW_START = "2025-10-01"
DATE_WINDOW_END = "2025-11-30"
DATE_WINDOW_READABLE = "between October 1, 2025 and November 30, 2025"

REPO_PLATFORMS = [
    "github.com", "figshare.com", "zenodo.org", "dryad.org", "datadryad.org", "osf.io",
    "bitbucket.org", "data.mendeley.com", "kaggle.com", "doi.org/10.5281/zenodo", "datahub.io"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_date: Optional[str] = None  # Keep as free text string from answer
    doi: Optional[str] = None               # Could be "10.xxxx/..." or full https://doi.org/...
    url: Optional[str] = None               # Main URL to paper or abstract
    source_urls: List[str] = Field(default_factory=list)      # All URLs mentioned relating to the paper
    repository_urls: List[str] = Field(default_factory=list)  # Subset of source URLs for data/code repositories


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
    Extract the metadata and cited URLs for the single scientific paper identified in the answer.

    Return a JSON object with the following fields:
    - title: The full title of the paper (string). If not provided, return null.
    - authors: An array of strings, the complete author list in order as presented in the answer. If missing, return an empty array.
    - journal: The journal name (string). If not provided, return null.
    - publication_date: The publication date mentioned in the answer (string as-is, e.g., 'October 15, 2025' or '2025-10-15'). If not provided, return null.
    - doi: The DOI string. Accept either a bare DOI (e.g., '10.1234/abc.def') or a full DOI URL (e.g., 'https://doi.org/10.1234/abc.def'). If not provided, return null.
    - url: A main URL to access the paper or its abstract (publisher page or similar). If multiple URLs are given, choose the main landing page URL. If not provided, return null.
    - source_urls: An array of all URLs cited in the answer that are relevant to the paper or its resources, including the DOI URL if present, publisher pages, journal pages, and any supplementary pages. Include only valid URLs. If none, return an empty array.
    - repository_urls: An array containing only the URLs in source_urls that point to public data/code repository platforms (e.g., GitHub, Figshare, Zenodo, Dryad, OSF, Bitbucket, Mendeley Data, Kaggle). If none were cited, return an empty array.

    Rules:
    1. Extract only what appears explicitly in the answer text. Do not invent fields.
    2. Always include full URLs (prepend 'http://' if protocol missing).
    3. Deduplicate URLs while preserving order of first appearance.
    4. For 'authors', return each author as a separate string.
    5. If the answer cites multiple repository URLs, include them all in repository_urls and also in source_urls.

    Respond strictly in JSON format.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_text(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not _is_nonempty_text(u):
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _doi_to_url(doi: Optional[str]) -> Optional[str]:
    if not _is_nonempty_text(doi):
        return None
    doi_str = str(doi).strip()
    if doi_str.lower().startswith("http://") or doi_str.lower().startswith("https://"):
        return doi_str
    # If it's a bare DOI, convert to resolver URL
    return f"https://doi.org/{doi_str}"


def build_all_sources(info: PaperInfo) -> List[str]:
    urls: List[str] = []
    if _is_nonempty_text(info.url):
        urls.append(info.url.strip())
    doi_url = _doi_to_url(info.doi)
    if _is_nonempty_text(doi_url):
        urls.append(doi_url.strip())
    if info.source_urls:
        urls.extend(info.source_urls)
    # Also include repository URLs (they are sources too)
    if info.repository_urls:
        urls.extend(info.repository_urls)
    return _dedup_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def add_required_fields_nodes(
    evaluator: Evaluator,
    parent_node,
    info: PaperInfo,
) -> Dict[str, Any]:
    """
    Add critical existence checks for all required bibliographic/access fields.
    Returns a dict of nodes for potential prerequisite usage.
    """
    req_root = evaluator.add_parallel(
        id="required_fields_provided",
        desc="The answer provides all required identifying/bibliographic/access information for the paper",
        parent=parent_node,
        critical=True
    )

    # (1) Title provided
    n_title = evaluator.add_custom_node(
        result=_is_nonempty_text(info.title),
        id="paper_title_provided",
        desc="The answer provides the full title of the paper",
        parent=req_root,
        critical=True
    )

    # (2) Complete author list provided (at least one)
    n_authors = evaluator.add_custom_node(
        result=(isinstance(info.authors, list) and len(info.authors) > 0 and all(_is_nonempty_text(a) for a in info.authors)),
        id="complete_author_list_provided",
        desc="The answer provides the complete author list",
        parent=req_root,
        critical=True
    )

    # (3) Journal name provided
    n_journal = evaluator.add_custom_node(
        result=_is_nonempty_text(info.journal),
        id="journal_name_provided",
        desc="The answer provides the journal name",
        parent=req_root,
        critical=True
    )

    # (4) Publication date provided
    n_pubdate = evaluator.add_custom_node(
        result=_is_nonempty_text(info.publication_date),
        id="publication_date_provided",
        desc="The answer provides the publication date",
        parent=req_root,
        critical=True
    )

    # (5) DOI provided
    n_doi = evaluator.add_custom_node(
        result=_is_nonempty_text(info.doi),
        id="doi_provided",
        desc="The answer provides the paper's DOI",
        parent=req_root,
        critical=True
    )

    # (6) URL provided
    n_url = evaluator.add_custom_node(
        result=_is_nonempty_text(info.url),
        id="url_provided",
        desc="The answer provides a URL to access the paper or its abstract",
        parent=req_root,
        critical=True
    )

    return {
        "req_root": req_root,
        "paper_title_provided": n_title,
        "complete_author_list_provided": n_authors,
        "journal_name_provided": n_journal,
        "publication_date_provided": n_pubdate,
        "doi_provided": n_doi,
        "url_provided": n_url,
    }


async def add_constraints_nodes(
    evaluator: Evaluator,
    parent_node,
    info: PaperInfo,
    prerequisite_nodes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add critical verification leaves for all scientific/discovery constraints.
    """
    constraints_root = evaluator.add_parallel(
        id="paper_constraints_satisfied",
        desc="The identified paper satisfies all scientific/discovery constraints stated in the question",
        parent=parent_node,
        critical=True
    )

    all_sources = build_all_sources(info)
    repo_sources = info.repository_urls if info.repository_urls else []

    # 1) Publication timeframe
    leaf_pubtime = evaluator.add_leaf(
        id="publication_timeframe",
        desc=f"The paper was published {DATE_WINDOW_READABLE}",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"This paper was published {DATE_WINDOW_READABLE}.",
        node=leaf_pubtime,
        sources=all_sources,
        additional_instruction=(
            f"Check the publication date on the article or DOI landing page. "
            f"The date must fall between {DATE_WINDOW_START} and {DATE_WINDOW_END} (inclusive). "
            "Accept 'Published online' or 'Publication date' within the window. "
            "If the visible date is outside this window, mark as not supported."
        ),
    )

    # 2) Peer-reviewed publication
    leaf_peer = evaluator.add_leaf(
        id="peer_reviewed_publication",
        desc="The paper is published in a peer-reviewed scientific journal",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The paper is published in a peer-reviewed scientific journal.",
        node=leaf_peer,
        sources=all_sources,
        additional_instruction=(
            "Use the journal/publisher page to check whether the venue is a peer-reviewed scientific journal. "
            "If the page or the journal's 'About' section indicates peer review, consider supported. "
            "Conference proceedings or preprints alone should not be considered peer-reviewed."
        ),
    )

    # 3) Geographic location: Northern Territory of Australia
    leaf_geo = evaluator.add_leaf(
        id="geographic_location",
        desc="The fossil discovery reported in the paper is from the Northern Territory of Australia",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The fossil discovery reported in the paper is from the Northern Territory of Australia.",
        node=leaf_geo,
        sources=all_sources,
        additional_instruction=(
            "Look for 'Northern Territory', 'NT', or locality names within the Northern Territory on the article page. "
            "General 'Australia' without specifying Northern Territory is insufficient."
        ),
    )

    # 4) Taxonomic identification: lamniform sharks
    leaf_taxon = evaluator.add_leaf(
        id="taxonomic_identification",
        desc="The fossils described in the paper are identified as lamniform sharks",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The fossils described in the paper are identified as lamniform sharks.",
        node=leaf_taxon,
        sources=all_sources,
        additional_instruction=(
            "Check for explicit mention of 'Lamniformes' or 'lamniform shark(s)' in the title, abstract, or main text."
        ),
    )

    # 5) Geological age: Aptian (Early Cretaceous, ~113–125 Ma)
    leaf_age = evaluator.add_leaf(
        id="geological_age",
        desc="The fossils are dated to the Aptian age of the Cretaceous period (approximately 113–125 million years ago)",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The fossils are dated to the Aptian age of the Cretaceous period (approximately 113–125 million years ago).",
        node=leaf_age,
        sources=all_sources,
        additional_instruction=(
            "Look for explicit mention of 'Aptian', 'Early Cretaceous', or the ~113–125 Ma window in stratigraphic context."
        ),
    )

    # 6) Specimen type: vertebral centra (vertebrae)
    leaf_specimen = evaluator.add_leaf(
        id="specimen_type",
        desc="The primary fossil specimens described are vertebral centra (vertebrae)",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The primary fossil specimens described are vertebral centra (vertebrae).",
        node=leaf_specimen,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the main specimens described are vertebral centra (vertebrae), not isolated teeth or other elements."
        ),
    )

    # 7) Quantitative size estimates: numerical body length estimates in meters
    leaf_size = evaluator.add_leaf(
        id="quantitative_size_estimates",
        desc="The paper provides numerical body length estimates in meters for the extinct shark",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The paper provides numerical body length estimates in meters for the extinct shark.",
        node=leaf_size,
        sources=all_sources,
        additional_instruction=(
            "Check for explicit body length numbers with unit 'm' (meters), e.g., '5.2 m', '6 m'. "
            "Ranges (e.g., '5–6 m') are acceptable if clearly in meters."
        ),
    )

    # 8) Regression methodology: regression modeling for body size from fossil measurements
    leaf_regression = evaluator.add_leaf(
        id="regression_methodology",
        desc="The study employs regression modeling techniques to estimate body size from fossil measurements",
        parent=constraints_root,
        critical=True
    )
    await evaluator.verify(
        claim="The study employs regression modeling techniques to estimate body size from fossil measurements.",
        node=leaf_regression,
        sources=all_sources,
        additional_instruction=(
            "Look for mentions of 'regression', 'linear regression', 'model', 'predictive model', or similar, "
            "used to relate vertebral or other measurements to body length."
        ),
    )

    # 9) Open data/code repository present
    leaf_repo = evaluator.add_leaf(
        id="open_data_repository",
        desc="The research includes publicly available data/code via a repository platform (e.g., GitHub, Figshare, Zenodo, Dryad, or similar)",
        parent=constraints_root,
        critical=True
    )
    # Prefer repository URLs; if none were extracted, still allow verification against all sources
    repo_verif_sources: List[str] = repo_sources if len(repo_sources) > 0 else all_sources
    await evaluator.verify(
        claim="The research includes publicly available data or analysis code via a repository platform (e.g., GitHub, Figshare, Zenodo, Dryad, OSF, Bitbucket, Mendeley Data).",
        node=leaf_repo,
        sources=repo_verif_sources,
        additional_instruction=(
            "Verify that at least one provided URL leads to a public repository page containing data or code related to this study. "
            "Platform examples include GitHub, Figshare, Zenodo, Dryad, OSF, Bitbucket, Mendeley Data, Kaggle."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the lamniform NT Aptian 2025 task.
    """
    # Initialize evaluator (root is non-critical by framework design; children critical will gate the score)
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

    # Extract structured info from the answer
    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperInfo,
        extraction_name="paper_info"
    )

    # Build top-level critical node to mirror rubric's critical root semantics
    main_node = evaluator.add_parallel(
        id="main_verification",
        desc="The identified paper meets all stated constraints and the answer provides all required bibliographic/access fields",
        parent=root,
        critical=True
    )

    # Required fields existence checks (critical)
    req_nodes = await add_required_fields_nodes(evaluator, main_node, paper_info)

    # Scientific/discovery constraints verifications (critical)
    await add_constraints_nodes(evaluator, main_node, paper_info, prerequisite_nodes=req_nodes)

    # Return evaluation summary
    return evaluator.get_summary()