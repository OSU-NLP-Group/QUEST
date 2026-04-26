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
TASK_ID = "faculty_topcs_4_unis"
TASK_DESCRIPTION = (
    "Identify one current faculty researcher from each of the following four universities: "
    "Massachusetts Institute of Technology (MIT), Stanford University, Carnegie Mellon University (CMU), "
    "and University of California, Berkeley (UC Berkeley). Each researcher must satisfy both of these criteria:\n\n"
    "1. Have an h-index of 15 or higher (as shown on their Google Scholar profile)\n"
    "2. Have published at least 3 papers at top-tier computer science conferences—specifically NeurIPS, ICML, ACL, CVPR, or ICCV—between 2021 and 2025 (inclusive)\n\n"
    "For each of the four researchers, provide:\n"
    "- Their full name\n"
    "- Their current h-index\n"
    "- A reference URL to either their Google Scholar profile or their official university faculty page\n"
    "- A brief list of their qualifying conference publications (title and year) from the specified conferences and time period"
)

ALLOWED_CONFERENCES = ["NeurIPS", "ICML", "ACL", "CVPR", "ICCV"]
YEAR_MIN = 2021
YEAR_MAX = 2025

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Publication(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Researcher(BaseModel):
    name: Optional[str] = None
    h_index: Optional[str] = None
    reference_url: Optional[str] = None  # Google Scholar profile OR official faculty page
    scholar_url: Optional[str] = None    # Explicit Scholar URL if separately cited
    publications: List[Publication] = Field(default_factory=list)


class ResearchersExtraction(BaseModel):
    mit: Optional[Researcher] = None
    stanford: Optional[Researcher] = None
    cmu: Optional[Researcher] = None
    berkeley: Optional[Researcher] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    return (
        "Extract exactly one researcher per university (MIT, Stanford, CMU, UC Berkeley) as presented in the answer. "
        "For each university, return a structured object with the following fields:\n"
        "- name: The full name of the researcher\n"
        "- h_index: The h-index value exactly as stated in the answer (string)\n"
        "- reference_url: A single URL to either the researcher's Google Scholar profile OR their official university faculty page (whichever the answer cites). "
        "If multiple are given, pick the primary one according to the answer's wording.\n"
        "- scholar_url: If the answer explicitly cites a Google Scholar profile URL, include it here; otherwise return null\n"
        "- publications: An array of publications explicitly listed in the answer that the author claims as qualifying. "
        "Each publication item must include:\n"
        "  * title: The paper title as written in the answer\n"
        "  * year: The year (string) as written; do not convert types\n"
        "  * venue: The conference name if provided (e.g., NeurIPS, ICML, ACL, CVPR, ICCV). If not stated, return null\n"
        "  * urls: All URLs associated with this specific publication as cited in the answer (e.g., proceedings page, OpenReview, Google Scholar item). "
        "If none provided, return an empty list.\n\n"
        "Important rules:\n"
        "1) Do not invent or infer any data. Only extract what is explicitly in the answer.\n"
        "2) If the answer mentions more than one researcher for a university, select the first clearly presented one.\n"
        "3) For publications, extract up to the first five items listed that the answer claims as qualifying; keep their titles/years exactly.\n"
        "4) URLs may appear as plain links or markdown links; extract the actual URLs.\n"
        "5) If any field is missing in the answer, set it to null (or empty list for arrays).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def researcher_sources(data: Researcher) -> List[str]:
    sources: List[str] = []
    if _nonempty(data.scholar_url):
        sources.append(data.scholar_url.strip())  # Prefer Scholar first for h-index checks
    if _nonempty(data.reference_url):
        sources.append(data.reference_url.strip())
    # Deduplicate while preserving order
    seen = set()
    uniq_sources = []
    for u in sources:
        if u not in seen:
            seen.add(u)
            uniq_sources.append(u)
    return uniq_sources


def publication_sources(pub: Publication, data: Researcher) -> List[str]:
    sources = list(pub.urls or [])
    # Fallback to researcher sources if publication has no URLs
    if not sources:
        sources = researcher_sources(data)
    # Deduplicate
    seen = set()
    uniq_sources = []
    for u in sources:
        if _nonempty(u) and u not in seen:
            seen.add(u)
            uniq_sources.append(u)
    return uniq_sources


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_publication(
    evaluator: Evaluator,
    parent_node,
    uni_key: str,
    pub: Publication,
    pub_index: int,
    researcher_name: Optional[str],
    data: Researcher,
) -> None:
    """
    Verify a single publication qualifies (title supported by sources, year in range, venue allowed, authored by researcher).
    """
    pub_title = pub.title or ""
    pub_year = pub.year or ""
    pub_venue = pub.venue or ""
    sources = publication_sources(pub, data)

    pub_node = evaluator.add_parallel(
        id=f"{uni_key}_pub_{pub_index}",
        desc=f"Publication #{pub_index + 1} qualifies (allowed venue and year, authored by the researcher)",
        parent=parent_node,
        critical=True,
    )

    # Title provided (critical existence)
    title_provided = _nonempty(pub.title)
    evaluator.add_custom_node(
        result=title_provided,
        id=f"{uni_key}_pub_{pub_index}_title_provided",
        desc=f"Publication #{pub_index + 1}: title is provided",
        parent=pub_node,
        critical=True,
    )

    # Title supported by sources
    title_supported_node = evaluator.add_leaf(
        id=f"{uni_key}_pub_{pub_index}_title_supported",
        desc=f"Publication #{pub_index + 1}: title appears on the cited source page(s)",
        parent=pub_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The paper titled '{pub_title}' appears on the provided page(s), allowing minor title variants.",
        node=title_supported_node,
        sources=sources,
        additional_instruction=(
            "Search the page(s) for a paper title that matches or closely matches the provided title "
            "(case-insensitive and allowing minor punctuation/spacing differences)."
        ),
    )

    # Year in range verification
    year_range_node = evaluator.add_leaf(
        id=f"{uni_key}_pub_{pub_index}_year_in_range",
        desc=f"Publication #{pub_index + 1}: publication year is between 2021 and 2025 (inclusive)",
        parent=pub_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This paper was published in {pub_year}, and the year is within 2021–2025 inclusive.",
        node=year_range_node,
        sources=sources,
        additional_instruction=(
            "Confirm the publication year on the page. The year must be 2021, 2022, 2023, 2024, or 2025. "
            "If the year is unclear or missing, treat the claim as not supported."
        ),
    )

    # Venue allowed verification
    venue_allowed_node = evaluator.add_leaf(
        id=f"{uni_key}_pub_{pub_index}_venue_allowed",
        desc=f"Publication #{pub_index + 1}: venue is one of NeurIPS, ICML, ACL, CVPR, or ICCV",
        parent=pub_node,
        critical=True,
    )
    venue_claim = (
        f"This paper was published at {pub_venue}, which is among NeurIPS, ICML, ACL, CVPR, or ICCV."
        if _nonempty(pub_venue)
        else "This paper was published at one of the following conferences: NeurIPS, ICML, ACL, CVPR, or ICCV."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_allowed_node,
        sources=sources,
        additional_instruction=(
            "Check whether the publication is in any of these conferences: "
            "NeurIPS (Conference on Neural Information Processing Systems), "
            "ICML (International Conference on Machine Learning), "
            "ACL (Association for Computational Linguistics conference), "
            "CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition), "
            "ICCV (International Conference on Computer Vision). "
            "Workshop tracks do not count unless clearly part of these main conferences."
        ),
    )

    # Authorship includes the researcher
    authored_by_node = evaluator.add_leaf(
        id=f"{uni_key}_pub_{pub_index}_authored_by",
        desc=f"Publication #{pub_index + 1}: authorship includes the named researcher",
        parent=pub_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{(researcher_name or '').strip()} is an author of this paper.",
        node=authored_by_node,
        sources=sources,
        additional_instruction=(
            "Verify that the named researcher appears among the authors. "
            "Allow minor name variants (middle initials, casing, abbreviated given names)."
        ),
    )


async def verify_university_researcher(
    evaluator: Evaluator,
    root_node,
    uni_key: str,
    uni_name: str,
    data: Optional[Researcher],
) -> None:
    """
    Build verification nodes for one university's researcher, covering credentials and publications.
    """
    data = data or Researcher()
    uni_node = evaluator.add_parallel(
        id=f"{uni_key}_researcher",
        desc=f"One researcher currently affiliated with {uni_name}",
        parent=root_node,
        critical=False,
    )

    # ----- Credentials group (critical) -----
    cred_node = evaluator.add_parallel(
        id=f"{uni_key}_credentials",
        desc=f"Researcher name, h-index of 15 or higher, and reference URL to Google Scholar profile or official {uni_name} faculty page provided",
        parent=uni_node,
        critical=True,
    )

    name_ok = _nonempty(data.name)
    h_ok = _nonempty(data.h_index)
    ref_ok = _nonempty(data.reference_url)

    # Required info provided (name + h-index + reference URL)
    evaluator.add_custom_node(
        result=(name_ok and h_ok and ref_ok),
        id=f"{uni_key}_credentials_provided",
        desc=f"{uni_name}: researcher name, h-index, and reference URL are provided",
        parent=cred_node,
        critical=True,
    )

    # Reference URL kind check (Scholar or official university faculty page)
    ref_kind_node = evaluator.add_leaf(
        id=f"{uni_key}_reference_url_kind",
        desc=f"{uni_name}: reference URL is either a Google Scholar profile or an official {uni_name} faculty page",
        parent=cred_node,
        critical=True,
    )
    uni_domain_hint = ""
    if "MIT" in uni_name:
        uni_domain_hint = "mit.edu"
    elif "Stanford" in uni_name:
        uni_domain_hint = "stanford.edu"
    elif "Carnegie Mellon" in uni_name or "CMU" in uni_name:
        uni_domain_hint = "cmu.edu"
    elif "Berkeley" in uni_name:
        uni_domain_hint = "berkeley.edu"

    await evaluator.verify(
        claim=f"This page is either a Google Scholar profile page or an official faculty page of {uni_name}.",
        node=ref_kind_node,
        sources=(data.reference_url or None),
        additional_instruction=(
            "Accept pages under scholar.google.com as Google Scholar profiles. "
            f"Accept pages under '*.{uni_domain_hint}' or '{uni_domain_hint}' as official university sites if they appear to be a faculty profile page. "
            "Use page content and URL to judge."
        ),
    )

    # Affiliation verification
    affiliation_node = evaluator.add_leaf(
        id=f"{uni_key}_affiliation_current",
        desc=f"{uni_name}: the named researcher is currently affiliated with {uni_name}",
        parent=cred_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{(data.name or '').strip()} is currently affiliated with {uni_name} as faculty/researcher.",
        node=affiliation_node,
        sources=researcher_sources(data),
        additional_instruction=(
            "Confirm that the page(s) indicate current affiliation (e.g., titles such as Assistant/Associate/Full Professor, "
            "or affiliation lines on Google Scholar). Prefer official faculty page if available."
        ),
    )

    # h-index threshold verification (>= 15) grounded by Scholar or provided sources
    hindex_node = evaluator.add_leaf(
        id=f"{uni_key}_hindex_threshold",
        desc=f"{uni_name}: h-index is 15 or higher per Google Scholar (as cited)",
        parent=cred_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The profile shows an h-index value of '{(data.h_index or '').strip()}', and this value is at least 15.",
        node=hindex_node,
        sources=researcher_sources(data),
        additional_instruction=(
            "Use the Google Scholar profile page to locate the 'h-index' metric. "
            "Confirm the h-index is a numeric value >= 15. If the page is not a Scholar profile or h-index is not shown, treat as not supported."
        ),
    )

    # ----- Publications group (critical) -----
    pubs_node = evaluator.add_parallel(
        id=f"{uni_key}_publications",
        desc="A list of at least 3 qualifying publications at top-tier CS conferences (NeurIPS, ICML, ACL, CVPR, or ICCV) "
             "published between 2021-2025 is provided, with publication titles and years included for each",
        parent=uni_node,
        critical=True,
    )

    # At least 3 items with title and year provided
    num_with_title_year = sum(1 for p in (data.publications or []) if _nonempty(p.title) and _nonempty(p.year))
    evaluator.add_custom_node(
        result=(num_with_title_year >= 3),
        id=f"{uni_key}_pubs_list_provided",
        desc=f"{uni_name}: at least 3 publications with titles and years are provided in the answer",
        parent=pubs_node,
        critical=True,
    )

    # Verify the first 3 publications individually
    pubs_to_check = (data.publications or [])[:3]
    # Pad to ensure we always create nodes (for clarity/debugging)
    while len(pubs_to_check) < 3:
        pubs_to_check.append(Publication())

    for idx, pub in enumerate(pubs_to_check):
        await verify_single_publication(
            evaluator=evaluator,
            parent_node=pubs_node,
            uni_key=uni_key,
            pub=pub,
            pub_index=idx,
            researcher_name=data.name,
            data=data,
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
    Evaluate an answer for the four-university faculty researcher criteria task.
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction",
    )

    # Build verification tree per university
    await verify_university_researcher(
        evaluator=evaluator,
        root_node=root,
        uni_key="mit",
        uni_name="Massachusetts Institute of Technology (MIT)",
        data=extracted.mit,
    )
    await verify_university_researcher(
        evaluator=evaluator,
        root_node=root,
        uni_key="stanford",
        uni_name="Stanford University",
        data=extracted.stanford,
    )
    await verify_university_researcher(
        evaluator=evaluator,
        root_node=root,
        uni_key="cmu",
        uni_name="Carnegie Mellon University (CMU)",
        data=extracted.cmu,
    )
    await verify_university_researcher(
        evaluator=evaluator,
        root_node=root,
        uni_key="berkeley",
        uni_name="University of California, Berkeley (UC Berkeley)",
        data=extracted.berkeley,
    )

    # Return standardized evaluation summary
    return evaluator.get_summary()