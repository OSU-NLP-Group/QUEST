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
TASK_ID = "icml2025_cs_lg_top_tier_aff_arxiv_timing"
TASK_DESCRIPTION = """
Identify a research paper that was accepted at the International Conference on Machine Learning (ICML) 2025 and meets ALL of the following requirements:

1. At least one author must be affiliated with Stanford University, MIT (Massachusetts Institute of Technology), UC Berkeley, or Carnegie Mellon University at the time of the paper
2. The paper must have been published on arXiv in the cs.LG (Machine Learning) category
3. The paper must have been made available on arXiv before May 15, 2025 (the ICML 2025 full paper submission deadline)
4. The paper must be officially listed among the accepted papers for ICML 2025

For the paper you identify, provide the following information:
- Complete paper title
- Complete list of all authors with their institutional affiliations
- Direct arXiv URL for the paper
- Direct URL to the paper's page on the official ICML 2025 conference website or proceedings
- Total number of authors on the paper
- The specific top-tier institution(s) (from Stanford/MIT/Berkeley/CMU) that one or more of the authors are affiliated with
"""

CUTOFF_DATE_ISO = "2025-05-15"
ICML_OFFICIAL_DOMAINS = ("icml.cc", "proceedings.mlr.press")
TOP_TIER_FULL = [
    "Stanford University",
    "Massachusetts Institute of Technology",
    "MIT",
    "University of California, Berkeley",
    "UC Berkeley",
    "Carnegie Mellon University",
    "CMU",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Author(BaseModel):
    name: Optional[str] = None
    affiliations: List[str] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    title: Optional[str] = None
    arxiv_url: Optional[str] = None
    icml_url: Optional[str] = None
    authors: List[Author] = Field(default_factory=list)
    author_count: Optional[str] = None
    specific_institutions: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
    Extract the details of a single ICML 2025 accepted paper as presented in the answer. You must not invent any information.

    Return a JSON with the following fields:
    - title: The complete paper title exactly as given in the answer text.
    - arxiv_url: The direct arXiv URL for the paper (prefer the /abs/ URL if multiple are present).
    - icml_url: The direct URL to the paper’s official ICML 2025 page or PMLR proceedings page (if present).
    - authors: An array of objects, each with:
        - name: The author's full name exactly as shown in the answer
        - affiliations: An array of institution names for that author as stated in the answer
    - author_count: The total number of authors as stated in the answer (string form). If not explicitly stated, set to null.
    - specific_institutions: An array listing which of the following institutions are claimed in the answer to be represented by the authors:
        ["Stanford University", "Massachusetts Institute of Technology", "MIT",
         "University of California, Berkeley", "UC Berkeley",
         "Carnegie Mellon University", "CMU"].
      Use the same string variants as they appear in the answer. If none are stated, return an empty array.

    Apply URL extraction rules: only extract URLs that are explicitly present. If a field is missing in the answer, set it to null or an empty array as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def looks_like_arxiv_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    return ("arxiv.org" in u) and ("/abs/" in u or "/pdf/" in u)


def looks_like_icml_official_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(domain in u for domain in ICML_OFFICIAL_DOMAINS)


def format_authors_for_claim(authors: List[Author]) -> str:
    # Format: "Name (Aff1; Aff2); Name2 (Aff1)"
    entries = []
    for a in authors:
        name = a.name or ""
        affs = "; ".join([aff for aff in a.affiliations if aff]) if a.affiliations else ""
        if affs:
            entries.append(f"{name} ({affs})")
        else:
            entries.append(f"{name}")
    return "; ".join(entries)


def compute_expected_author_count(extracted: PaperExtraction) -> Optional[int]:
    # Prefer explicit author_count if it looks like a number; else use len(authors)
    if extracted.author_count:
        try:
            # Extract first integer occurrence
            import re
            m = re.search(r"\d+", extracted.author_count)
            if m:
                return int(m.group(0))
        except Exception:
            pass
    if extracted.authors:
        return len(extracted.authors)
    return None


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_urls_and_identity_nodes(
    evaluator: Evaluator,
    parent,
    info: PaperExtraction,
):
    """
    Critical: Ensure core identity items (title, arXiv URL, ICML URL) are provided and consistent with sources.
    Returns a dict of key nodes for dependency referencing.
    """
    url_ident_node = evaluator.add_parallel(
        id="urls_and_identity",
        desc="Core identity: title provided, arXiv URL and ICML URL provided and title matches on both pages",
        parent=parent,
        critical=True
    )

    # Existence/format checks (critical)
    title_provided_node = evaluator.add_custom_node(
        result=bool(info.title and info.title.strip()),
        id="paper_title_provided",
        desc="Paper title is provided (non-empty)",
        parent=url_ident_node,
        critical=True
    )

    arxiv_url_provided_node = evaluator.add_custom_node(
        result=looks_like_arxiv_url(info.arxiv_url),
        id="arxiv_url_provided_valid",
        desc="arXiv URL is provided and appears valid",
        parent=url_ident_node,
        critical=True
    )

    icml_url_provided_node = evaluator.add_custom_node(
        result=looks_like_icml_official_url(info.icml_url),
        id="icml_url_provided_valid",
        desc="ICML URL is provided and appears to be an official ICML 2025 or PMLR page",
        parent=url_ident_node,
        critical=True
    )

    # Title matches on arXiv
    title_match_arxiv = evaluator.add_leaf(
        id="title_match_arxiv",
        desc="The provided title matches the title shown on the arXiv page",
        parent=url_ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The paper title on this arXiv page is '{info.title or ''}'.",
        node=title_match_arxiv,
        sources=info.arxiv_url if info.arxiv_url else None,
        additional_instruction="Allow minor punctuation/case differences. Compare the full main title as shown on arXiv.",
        extra_prerequisites=[title_provided_node, arxiv_url_provided_node]
    )

    # Title matches on ICML official page (icml.cc or PMLR)
    title_match_icml = evaluator.add_leaf(
        id="title_match_icml",
        desc="The provided title matches the title shown on the official ICML 2025 page (icml.cc or PMLR)",
        parent=url_ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The paper title on this ICML 2025 official page is '{info.title or ''}'.",
        node=title_match_icml,
        sources=info.icml_url if info.icml_url else None,
        additional_instruction="The URL should be on icml.cc or proceedings.mlr.press (PMLR). Allow minor punctuation/case differences.",
        extra_prerequisites=[title_provided_node, icml_url_provided_node]
    )

    return {
        "title_provided": title_provided_node,
        "arxiv_url_valid": arxiv_url_provided_node,
        "icml_url_valid": icml_url_provided_node,
    }


async def build_core_constraints_nodes(
    evaluator: Evaluator,
    parent,
    info: PaperExtraction,
    prereq_nodes: Dict[str, Any]
):
    """
    Critical constraints:
    - At least one author from Stanford/MIT/UC Berkeley/CMU (top-tier)
    - arXiv category includes cs.LG
    - arXiv first submission date on/before 2025-05-15
    - ICML acceptance as official page for ICML 2025
    """
    core_node = evaluator.add_parallel(
        id="core_constraints",
        desc="All core constraints satisfied (top-tier affiliation, arXiv cs.LG, arXiv timing, listed as ICML 2025 accepted)",
        parent=parent,
        critical=True
    )

    # Top-tier affiliation present (check on either arXiv or ICML official page)
    top_tier_leaf = evaluator.add_leaf(
        id="top_tier_affiliation",
        desc="At least one author has affiliation with Stanford/MIT/UC Berkeley/CMU",
        parent=core_node,
        critical=True
    )
    top_tier_claim = (
        "Among the author affiliations shown on this page, at least one is one of the following institutions: "
        "Stanford University, Massachusetts Institute of Technology (MIT), University of California, Berkeley (UC Berkeley), or Carnegie Mellon University (CMU)."
    )
    sources_for_aff = [u for u in [info.icml_url, info.arxiv_url] if u]
    await evaluator.verify(
        claim=top_tier_claim,
        node=top_tier_leaf,
        sources=sources_for_aff if sources_for_aff else None,
        additional_instruction="Check the author affiliations on the provided pages. Accept common abbreviations (MIT for Massachusetts Institute of Technology; UC Berkeley for University of California, Berkeley; CMU for Carnegie Mellon University).",
        extra_prerequisites=[prereq_nodes.get("arxiv_url_valid"), prereq_nodes.get("icml_url_valid")]
    )

    # arXiv category includes cs.LG
    arxiv_category_leaf = evaluator.add_leaf(
        id="arxiv_category",
        desc="arXiv category includes cs.LG (Machine Learning)",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim="This arXiv paper is categorized under cs.LG (Machine Learning).",
        node=arxiv_category_leaf,
        sources=info.arxiv_url if info.arxiv_url else None,
        additional_instruction="Confirm that 'cs.LG' appears in the subject classifications on the arXiv page. It can be primary or secondary.",
        extra_prerequisites=[prereq_nodes.get("arxiv_url_valid")]
    )

    # arXiv timing: first submission (v1) on or before CUTOFF_DATE_ISO
    arxiv_timing_leaf = evaluator.add_leaf(
        id="arxiv_timing",
        desc=f"arXiv first submission date is on or before {CUTOFF_DATE_ISO}",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first arXiv submission (v1) date is on or before {CUTOFF_DATE_ISO}.",
        node=arxiv_timing_leaf,
        sources=info.arxiv_url if info.arxiv_url else None,
        additional_instruction="Check the 'Submission history' section. Use the v1 date. If equal to the cutoff date, consider it acceptable.",
        extra_prerequisites=[prereq_nodes.get("arxiv_url_valid")]
    )

    # ICML acceptance (official page)
    icml_accept_leaf = evaluator.add_leaf(
        id="icml_acceptance",
        desc="The paper is officially listed among ICML 2025 accepted papers (official ICML or PMLR page)",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is an official page for ICML 2025 (icml.cc or PMLR proceedings) that lists this paper as part of ICML 2025.",
        node=icml_accept_leaf,
        sources=info.icml_url if info.icml_url else None,
        additional_instruction="Verify that the domain is icml.cc or proceedings.mlr.press and the page clearly indicates ICML 2025 and the paper title/entry.",
        extra_prerequisites=[prereq_nodes.get("icml_url_valid")]
    )


async def build_authors_affiliations_nodes(
    evaluator: Evaluator,
    parent,
    info: PaperExtraction,
    prereq_nodes: Dict[str, Any]
):
    """
    Critical reporting: Provide complete authors list with affiliations and verify against sources.
    """
    aa_node = evaluator.add_parallel(
        id="authors_and_affiliations_group",
        desc="Authors and affiliations are provided and supported by sources",
        parent=parent,
        critical=True
    )

    # Existence checks (critical)
    authors_list_provided = evaluator.add_custom_node(
        result=bool(info.authors) and any(a.name for a in info.authors),
        id="authors_list_provided",
        desc="Authors list is provided",
        parent=aa_node,
        critical=True
    )

    affiliations_provided = evaluator.add_custom_node(
        result=bool(info.authors) and all((a.affiliations and any(aff.strip() for aff in a.affiliations)) for a in info.authors if (a and a.name)),
        id="affiliations_provided",
        desc="Each listed author has at least one institutional affiliation provided",
        parent=aa_node,
        critical=True
    )

    # Authors list supported by (arXiv and/or ICML)
    authors_supported_leaf = evaluator.add_leaf(
        id="authors_supported",
        desc="The provided full authors list matches the sources (allow minor ordering/format differences)",
        parent=aa_node,
        critical=True
    )
    author_names = [a.name for a in info.authors if a and a.name]
    sources_for_authors = [u for u in [info.icml_url, info.arxiv_url] if u]
    await evaluator.verify(
        claim=f"The complete author list for this paper matches the following names (order can differ, allow minor variants): {author_names}.",
        node=authors_supported_leaf,
        sources=sources_for_authors if sources_for_authors else None,
        additional_instruction="Compare the set of author names. Allow minor variants (middle initials, punctuation, name order differences).",
        extra_prerequisites=[authors_list_provided, prereq_nodes.get("arxiv_url_valid"), prereq_nodes.get("icml_url_valid")]
    )

    # Affiliations supported by sources
    affiliations_supported_leaf = evaluator.add_leaf(
        id="affiliations_supported",
        desc="The author–affiliation mapping is supported by the sources",
        parent=aa_node,
        critical=True
    )
    formatted_pairs = format_authors_for_claim(info.authors)
    await evaluator.verify(
        claim=f"The following author–affiliation information is correct for this paper: {formatted_pairs}",
        node=affiliations_supported_leaf,
        sources=sources_for_authors if sources_for_authors else None,
        additional_instruction=(
            "Verify that each author has at least one listed affiliation consistent with the sources. "
            "Allow minor formatting differences. If a source page does not explicitly list affiliations for some authors, "
            "use any affiliation info available on the provided official pages (icml.cc or PMLR and arXiv)."
        ),
        extra_prerequisites=[affiliations_provided, prereq_nodes.get("arxiv_url_valid"), prereq_nodes.get("icml_url_valid")]
    )


async def build_optional_info_nodes(
    evaluator: Evaluator,
    parent,
    info: PaperExtraction,
    prereq_nodes: Dict[str, Any]
):
    """
    Non-critical: author count and specific institutions list.
    """
    opt_node = evaluator.add_parallel(
        id="optional_info",
        desc="Optional reporting checks (non-critical)",
        parent=parent,
        critical=False
    )

    # Author count group (non-critical)
    author_count_group = evaluator.add_parallel(
        id="author_count_group",
        desc="Author count provided and accurate (non-critical)",
        parent=opt_node,
        critical=False
    )

    author_count_provided = evaluator.add_custom_node(
        result=bool(info.author_count and info.author_count.strip()),
        id="author_count_provided",
        desc="Author count is provided",
        parent=author_count_group,
        critical=False
    )

    author_count_accurate = evaluator.add_leaf(
        id="author_count_accurate",
        desc="Author count in the answer matches the number of authors shown on sources",
        parent=author_count_group,
        critical=False
    )
    expected_count = compute_expected_author_count(info)
    count_claim = (
        f"The paper has exactly {expected_count} authors."
        if expected_count is not None else
        "The number of authors can be determined from this page."
    )
    sources_for_count = [u for u in [info.icml_url, info.arxiv_url] if u]
    await evaluator.verify(
        claim=count_claim,
        node=author_count_accurate,
        sources=sources_for_count if sources_for_count else None,
        additional_instruction="Count the authors listed on the page. Allow that some pages show all authors. Verify the exact count if determinable.",
        extra_prerequisites=[prereq_nodes.get("arxiv_url_valid"), prereq_nodes.get("icml_url_valid")]
    )

    # Specific top-tier institutions group (non-critical)
    specific_inst_group = evaluator.add_parallel(
        id="specific_institution_group",
        desc="Specific top-tier institutions identified are correct (non-critical)",
        parent=opt_node,
        critical=False
    )

    specific_inst_provided = evaluator.add_custom_node(
        result=bool(info.specific_institutions),
        id="specific_institution_provided",
        desc="Specific top-tier institution(s) are provided",
        parent=specific_inst_group,
        critical=False
    )

    specific_inst_supported = evaluator.add_leaf(
        id="specific_institution_supported",
        desc="The listed specific top-tier institution(s) appear among the author affiliations on sources",
        parent=specific_inst_group,
        critical=False
    )
    inst_list = info.specific_institutions if info.specific_institutions else []
    inst_claim = f"The following top-tier institutions appear among the author affiliations on this page: {inst_list}."
    await evaluator.verify(
        claim=inst_claim,
        node=specific_inst_supported,
        sources=sources_for_count if sources_for_count else None,
        additional_instruction=(
            "Check that the listed institutions appear in affiliations. "
            "Accept common variants: MIT = Massachusetts Institute of Technology; "
            "UC Berkeley = University of California, Berkeley; CMU = Carnegie Mellon University."
        ),
        extra_prerequisites=[prereq_nodes.get("arxiv_url_valid"), prereq_nodes.get("icml_url_valid")]
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
    Evaluate an answer for the ICML 2025 accepted paper with cs.LG and top-tier affiliation constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root remains non-critical to allow mixed criticality children
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

    # Extract structured paper info
    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperExtraction,
        extraction_name="paper_info"
    )

    # Add a high-level non-leaf node mirroring rubric intent (root has parallel children)
    # 1) URLs & identity (critical)
    prereq_nodes = await build_urls_and_identity_nodes(evaluator, root, paper_info)

    # 2) Core constraints (critical)
    await build_core_constraints_nodes(evaluator, root, paper_info, prereq_nodes)

    # 3) Authors & affiliations (critical)
    await build_authors_affiliations_nodes(evaluator, root, paper_info, prereq_nodes)

    # 4) Optional reporting (non-critical)
    await build_optional_info_nodes(evaluator, root, paper_info, prereq_nodes)

    # Provide custom info about the constraint thresholds for transparency
    evaluator.add_custom_info(
        {
            "cutoff_date_iso": CUTOFF_DATE_ISO,
            "top_tier_institutions_allowed": TOP_TIER_FULL,
            "icml_official_domains": list(ICML_OFFICIAL_DOMAINS)
        },
        info_type="policy",
        info_name="evaluation_policy"
    )

    return evaluator.get_summary()