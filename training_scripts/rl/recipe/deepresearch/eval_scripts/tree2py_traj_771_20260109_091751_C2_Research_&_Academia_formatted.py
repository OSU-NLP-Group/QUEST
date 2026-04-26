import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "conf_paper_2024_asia_affil"
TASK_DESCRIPTION = (
    "I am looking for a recently published research paper on large language models, transformer architectures, or "
    "natural language processing from either the NeurIPS 2024 or ICML 2024 conference, where at least one of the "
    "authors is affiliated with a university or research institution located in Asia (including countries such as "
    "China, Japan, South Korea, Singapore, India, etc.).\n\n"
    "Please provide the following information:\n"
    "1. The paper title\n"
    "2. The authors' names\n"
    "3. The institutional affiliations of all authors\n"
    "4. A link to the paper from the official conference proceedings website (nips.cc for NeurIPS or icml.cc / proceedings.mlr.press for ICML)\n"
    "5. A reference URL that verifies the geographic location of the Asian institution(s)"
)

ASIAN_COUNTRY_HINT = (
    "Asia includes (non-exhaustive examples): China, Japan, South Korea, Singapore, India, Hong Kong, Taiwan, "
    "Pakistan, Bangladesh, Vietnam, Thailand, Malaysia, Indonesia, Philippines, Sri Lanka, Nepal, Mongolia, "
    "Kazakhstan, Saudi Arabia, United Arab Emirates, Qatar, Kuwait, Iran, Israel, Turkey and others in Asia."
)

ALLOWED_PROCEEDINGS_DOMAINS = [
    "nips.cc",
    "neurips.cc",
    "proceedings.neurips.cc",
    "icml.cc",
    "proceedings.mlr.press",
]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    proceedings_url: Optional[str] = None
    conference: Optional[str] = None  # "NeurIPS 2024" or "ICML 2024" if present in the answer
    location_verification_url: Optional[str] = None  # Official institution URL confirming Asia location


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper() -> str:
    return (
        "Extract details for exactly one selected paper as presented in the answer. If multiple papers are listed, "
        "extract the first one that appears, preferably one that fits the constraints (NeurIPS 2024 or ICML 2024). "
        "Return the following fields:\n"
        "- title: The exact paper title as written in the answer text.\n"
        "- authors: Array of author names exactly as listed.\n"
        "- affiliations: Array of institutional affiliations for the authors exactly as written in the answer text. "
        "If the answer groups or lists affiliations, capture them as individual institution strings.\n"
        "- proceedings_url: The URL to the official conference proceedings page for this paper "
        "(NeurIPS: nips.cc or proceedings.neurips.cc; ICML: icml.cc or proceedings.mlr.press). If missing, set to null.\n"
        "- conference: If the conference (NeurIPS 2024 or ICML 2024) is explicitly mentioned, include it, otherwise null.\n"
        "- location_verification_url: A URL to an official institution page that can verify the institution's geographic "
        "location (e.g., the university or institute's official site). If missing, set to null.\n"
        "Do not invent any information not present in the answer text. If any field is missing, set it to null or an empty array."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _domain_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def is_official_proceedings_url(url: Optional[str]) -> bool:
    if not url:
        return False
    dom = _domain_of(url)
    if not dom:
        return False
    return any(dom == allowed or dom.endswith("." + allowed) for allowed in ALLOWED_PROCEEDINGS_DOMAINS)


def guess_conference_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    dom = _domain_of(url) or ""
    if "neurips" in dom or "nips.cc" in dom:
        return "NeurIPS 2024"
    if "icml.cc" in dom or "mlr.press" in dom:
        return "ICML 2024"
    return None


def list_to_readable(items: List[str]) -> str:
    items = [s for s in (items or []) if s and s.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_paper_eligibility(
    evaluator: Evaluator,
    parent_node,
    paper: PaperInfo,
) -> Dict[str, Any]:
    """
    Build and verify 'Paper_Eligibility' subtree.

    Structure (critical parallel):
    - Official_Proceedings_Access_Criterion (custom existence + domain check)
    - Conference_Criterion (verify by proceedings_url)
    - Topic_Criterion (verify by proceedings_url)
    - Asian_Affiliation_Criterion (parallel critical)
        - Asian_Affiliation_Matches_Proceedings (verify by proceedings_url)
        - Asian_Institution_Located_In_Asia (verify by location_verification_url or proceedings_url as fallback)
    """
    elig_node = evaluator.add_parallel(
        id="Paper_Eligibility",
        desc="Chosen paper satisfies all eligibility constraints (conference, topic, Asian affiliation, and official proceedings availability).",
        parent=parent_node,
        critical=True,
    )

    # 1) Official proceedings access (domain) check
    official_ok = is_official_proceedings_url(paper.proceedings_url)
    official_node = evaluator.add_custom_node(
        result=bool(paper.proceedings_url) and official_ok,
        id="Official_Proceedings_Access_Criterion",
        desc="Paper is accessible via an official conference proceedings URL (e.g., proceedings.neurips.cc for NeurIPS; proceedings.mlr.press or icml.cc for ICML).",
        parent=elig_node,
        critical=True,
    )

    # 2) Conference criterion (NeurIPS 2024 or ICML 2024)
    conf_node = evaluator.add_leaf(
        id="Conference_Criterion",
        desc="Paper is from NeurIPS 2024 or ICML 2024 conference proceedings.",
        parent=elig_node,
        critical=True,
    )
    conf_hint = paper.conference or guess_conference_from_url(paper.proceedings_url) or "NeurIPS 2024 or ICML 2024"
    conf_claim = (
        f"This page is an official proceedings page for {conf_hint}. "
        f"It should clearly indicate the 2024 edition of NeurIPS or ICML."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_node,
        sources=paper.proceedings_url,
        additional_instruction="Look for explicit indicators: 'NeurIPS 2024', 'ICML 2024', 'Proceedings', official conference branding. "
                               "On PMLR pages, 'Proceedings of the International Conference on Machine Learning' along with year 2024 also qualifies.",
        extra_prerequisites=[official_node],
    )

    # 3) Topic criterion (LLMs/Transformers/NLP)
    topic_node = evaluator.add_leaf(
        id="Topic_Criterion",
        desc="Paper focuses on large language models, transformer architectures, natural language processing, or related deep learning topics.",
        parent=elig_node,
        critical=True,
    )
    topic_claim = (
        "This paper is about large language models (LLMs), Transformer architectures (e.g., self‑attention), "
        "or natural language processing (NLP)."
    )
    await evaluator.verify(
        claim=topic_claim,
        node=topic_node,
        sources=paper.proceedings_url,
        additional_instruction="Check the title, abstract, and keywords for mentions such as 'LLM', 'language model', "
                               "'Transformer', 'self-attention', 'NLP', 'text generation', or equivalent terms.",
        extra_prerequisites=[official_node],
    )

    # 4) Asian affiliation criterion - break into two concrete leaves under a parallel critical node
    asian_node = evaluator.add_parallel(
        id="Asian_Affiliation_Criterion",
        desc="At least one author is affiliated with a university or research institution located in Asia.",
        parent=elig_node,
        critical=True,
    )

    # 4a) At least one of the provided affiliations appears on the proceedings page
    aff_on_proc_node = evaluator.add_leaf(
        id="Asian_Affiliation_Matches_Proceedings",
        desc="At least one listed affiliation appears among the affiliations on the official proceedings page.",
        parent=asian_node,
        critical=True,
    )
    aff_list_text = list_to_readable(paper.affiliations)
    aff_claim = (
        f"At least one of these affiliations appears among the author affiliations on this proceedings page: {paper.affiliations}."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_on_proc_node,
        sources=paper.proceedings_url,
        additional_instruction="It's sufficient if one institution string (or a close variant/abbreviation) appears anywhere on the page. "
                               "Allow minor variations (e.g., abbreviations, department names attached).",
        extra_prerequisites=[official_node],
    )

    # 4b) The referenced institution page indicates it is located in Asia
    in_asia_node = evaluator.add_leaf(
        id="Asian_Institution_Located_In_Asia",
        desc="The referenced institution website indicates the institution is located in an Asian country.",
        parent=asian_node,
        critical=True,
    )
    loc_sources = [u for u in [paper.location_verification_url, paper.proceedings_url] if u] or None
    asia_claim = (
        f"The organization described on this page is a university or research institution based in an Asian country. "
        f"{ASIAN_COUNTRY_HINT}"
    )
    await evaluator.verify(
        claim=asia_claim,
        node=in_asia_node,
        sources=loc_sources,
        additional_instruction="Check address, contact, about pages, or footer information to infer the country/city. "
                               "Passing examples: explicit mention of an Asian country/city, or a country-specific top-level domain with "
                               "supporting content. If unclear, mark as not supported.",
    )

    return {
        "eligibility_node": elig_node,
        "official_node": official_node,
        "conference_node": conf_node,
        "topic_node": topic_node,
        "asian_node": asian_node,
    }


async def build_required_outputs(
    evaluator: Evaluator,
    parent_node,
    paper: PaperInfo,
    official_precond_node,
) -> Dict[str, Any]:
    """
    Build and verify 'Required_Outputs' subtree.

    Structure (critical parallel):
    - Paper_Title (verify by proceedings_url)
    - Author_Names (verify by proceedings_url)
    - All_Author_Affiliations (existence/custom)
    - Official_Proceedings_Paper_Link (existence + domain/custom)
    - Asian_Institution_Location_Verification_URL (existence/custom)
    """
    req_node = evaluator.add_parallel(
        id="Required_Outputs",
        desc="Provide all requested fields for the selected paper.",
        parent=parent_node,
        critical=True,
    )

    # Paper title verification
    title_node = evaluator.add_leaf(
        id="Paper_Title",
        desc="Provide the paper title.",
        parent=req_node,
        critical=True,
    )
    title_claim = f"The title on the official proceedings page matches: '{paper.title}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=paper.proceedings_url,
        additional_instruction="Allow minor formatting, punctuation, and casing differences. "
                               "Focus on substantive equivalence.",
        extra_prerequisites=[official_precond_node],
    )

    # Author names verification
    authors_node = evaluator.add_leaf(
        id="Author_Names",
        desc="Provide the authors' names.",
        parent=req_node,
        critical=True,
    )
    authors_claim = (
        f"The author list on the proceedings page matches the provided set of names (order not required): {paper.authors}."
    )
    await evaluator.verify(
        claim=authors_claim,
        node=authors_node,
        sources=paper.proceedings_url,
        additional_instruction="Match authors by names allowing minor variants (middle initials, accents, casing). "
                               "Order need not match; check set equivalence or containment.",
        extra_prerequisites=[official_precond_node],
    )

    # Affiliations existence (we do not strictly verify all are shown on the proceedings page due to variability across sites)
    affs_exist = bool(paper.affiliations) and any(s.strip() for s in paper.affiliations)
    affs_node = evaluator.add_custom_node(
        result=affs_exist,
        id="All_Author_Affiliations",
        desc="Provide the institutional affiliations of all authors as stated in the paper.",
        parent=req_node,
        critical=True,
    )

    # Official proceedings paper link provided (existence + official domain)
    official_ok = is_official_proceedings_url(paper.proceedings_url)
    official_link_node = evaluator.add_custom_node(
        result=bool(paper.proceedings_url) and official_ok,
        id="Official_Proceedings_Paper_Link",
        desc="Provide a link to the paper from the official conference proceedings website (nips/neurips or icml official proceedings domains).",
        parent=req_node,
        critical=True,
    )

    # Asian institution location verification URL provided (existence)
    asian_loc_url_node = evaluator.add_custom_node(
        result=bool(paper.location_verification_url) and paper.location_verification_url.strip() != "",
        id="Asian_Institution_Location_Verification_URL",
        desc="Provide an official university or institutional website URL that verifies the geographic location of the Asian institution(s).",
        parent=req_node,
        critical=True,
    )

    return {
        "required_node": req_node,
        "title_node": title_node,
        "authors_node": authors_node,
        "affs_node": affs_node,
        "official_link_node": official_link_node,
        "asian_loc_url_node": asian_loc_url_node,
    }


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the NeurIPS/ICML 2024 LLM/Transformer/NLP paper with Asian affiliation requirement.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll add our own critical sequential main node beneath
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

    # Extract structured paper info from the answer
    paper: PaperInfo = await evaluator.extract(
        prompt=prompt_extract_paper(),
        template_class=PaperInfo,
        extraction_name="paper_extraction",
    )

    # Main task node (critical sequential as per rubric)
    task_node = evaluator.add_sequential(
        id="Complete_Research_Task",
        desc="Identify one qualifying NeurIPS 2024 or ICML 2024 paper and provide all required details and verification links.",
        parent=root,
        critical=True,
    )

    # Build eligibility subtree first
    elig_info = await build_paper_eligibility(evaluator, task_node, paper)

    # Build required outputs subtree second (sequential dependency)
    await build_required_outputs(
        evaluator,
        task_node,
        paper,
        official_precond_node=elig_info["official_node"],
    )

    # Return the finalized evaluation summary
    return evaluator.get_summary()