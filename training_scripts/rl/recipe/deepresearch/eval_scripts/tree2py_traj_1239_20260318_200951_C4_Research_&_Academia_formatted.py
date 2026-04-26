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
TASK_ID = "researcher_identification_aaai2026"
TASK_DESCRIPTION = """Identify a computer science faculty member at a research university in the United States who meets all of the following criteria:

1. The researcher must be currently employed as faculty at a research university located in the United States.

2. The researcher must be affiliated with a Computer Science department, or a closely related department such as Electrical Engineering and Computer Science or Computer Engineering.

3. The researcher must have a publicly accessible Google Scholar profile that displays their publications, citations, and h-index.

4. The researcher must have at least one paper accepted to the AAAI 2026 conference, which was held from January 20-27, 2026 in Singapore. This can be verified through their Google Scholar profile or the official AAAI 2026 proceedings.

5. According to their Google Scholar profile, the researcher must have an h-index of at least 30, indicating established research impact in their field.

6. According to their Google Scholar profile, the researcher must have at least 5,000 total citations.

7. The researcher must have at least one publication that includes an international co-author—that is, a co-author affiliated with an institution located outside the United States. This collaboration should be verifiable through author affiliations listed on published papers.

For your answer, provide the following information:
- The researcher's full name
- Their current university affiliation and department
- A link to their Google Scholar profile
- Their h-index and total citation count from Google Scholar
- A link to at least one AAAI 2026 paper they authored
- An example of a publication demonstrating international collaboration, including the international co-author's name and their affiliated institution
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AAAIPaper(BaseModel):
    title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    venue: Optional[str] = None


class InternationalCollaboration(BaseModel):
    title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    coauthor_name: Optional[str] = None
    coauthor_institution: Optional[str] = None
    coauthor_country: Optional[str] = None


class ResearcherExtraction(BaseModel):
    name: Optional[str] = None
    university: Optional[str] = None
    department: Optional[str] = None
    gs_profile_url: Optional[str] = None
    h_index: Optional[str] = None
    total_citations: Optional[str] = None

    affiliation_urls: List[str] = Field(default_factory=list)
    aaai_2026_papers: List[AAAIPaper] = Field(default_factory=list)
    international_collab_example: Optional[InternationalCollaboration] = None
    extra_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher_info() -> str:
    return """
    Extract the following fields exactly as presented in the answer. Do NOT invent or infer missing info.

    Required top-level fields:
    - name: Full name of the researcher (string)
    - university: Current university affiliation (string)
    - department: Department or unit (e.g., Computer Science, CSE, EECS) (string)
    - gs_profile_url: URL of the Google Scholar profile (string URL)
    - h_index: The h-index value stated in the answer (string; keep as provided, including any formatting)
    - total_citations: The total citation count stated in the answer (string; keep formatting as provided)

    Evidence URLs (explicitly present in the answer text):
    - affiliation_urls: List of URLs that support the person's affiliation (e.g., official university/department page, faculty profile page)
    - aaai_2026_papers: A list of objects for AAAI 2026 publications the researcher authored. For each:
        - title: The paper title (string; as written in the answer)
        - urls: List of URLs for that paper (e.g., AAAI proceedings page, DOI/publisher page, arXiv/Google Scholar entry referencing AAAI 2026)
        - venue: The venue string if mentioned (e.g., "AAAI 2026") (string or null)
    - international_collab_example: One publication that demonstrates international collaboration, if provided. It should include:
        - title: The publication title (string)
        - urls: List of URLs to the paper/publisher page showing author affiliations (not just an abstract if possible)
        - coauthor_name: The international co-author's name (string)
        - coauthor_institution: The international co-author's institution (string)
        - coauthor_country: The country of that institution if explicitly stated in the answer (string or null)
    - extra_urls: Any other URLs provided in the answer that are relevant support but don't fit the above buckets.

    Special rules for URLs:
    - Extract only actual URLs explicitly present in the answer (including markdown links). Do not infer or fabricate links.
    - Return empty lists when URLs are not provided in the answer.

    If any field is missing in the answer, set it to null (for strings) or [] (for URL lists).

    Return a single JSON object conforming to the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _present(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _combine_sources(*parts: Any) -> List[str]:
    """Combine strings and lists of URLs into a de-duplicated list (order-preserving)."""
    urls: List[str] = []
    seen = set()
    def add_one(u: Optional[str]):
        if not _present(u):
            return
        if u not in seen:
            urls.append(u)  # type: ignore
            seen.add(u)

    for p in parts:
        if p is None:
            continue
        if isinstance(p, str):
            add_one(p)
        elif isinstance(p, list):
            for u in p:
                add_one(u)
        elif isinstance(p, AAAIPaper):
            for u in p.urls:
                add_one(u)
        elif isinstance(p, InternationalCollaboration):
            for u in p.urls:
                add_one(u)
        else:
            # Try to iterate if it's a list-like of AAAIPaper
            try:
                for item in p:  # type: ignore
                    if isinstance(item, AAAIPaper):
                        for u in item.urls:
                            add_one(u)
                    elif isinstance(item, str):
                        add_one(item)
                    else:
                        pass
            except TypeError:
                pass
    return urls


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, ex: ResearcherExtraction) -> None:
    """
    Build the verification tree according to the rubric with proper leaf checks.
    Each rubric criterion is implemented as a critical sequential node with:
      - An existence/evidence gate (custom leaf)
      - A factual verification leaf grounded in URLs
    """

    # Create the rubric root under the main root to mirror the provided JSON
    rubric_root = evaluator.add_parallel(
        id="Researcher_Identification",
        desc="Identify a computer science faculty member at a U.S. research university who meets specified research profile criteria",
        parent=parent_node,
        critical=False
    )

    # Precompute common source pools
    sources_affil = _combine_sources(ex.affiliation_urls, ex.gs_profile_url, ex.extra_urls)
    aaai_urls_all: List[str] = []
    if ex.aaai_2026_papers:
        for p in ex.aaai_2026_papers:
            aaai_urls_all.extend(p.urls or [])
    sources_aaai = _combine_sources(aaai_urls_all, ex.gs_profile_url)
    sources_gs = _combine_sources(ex.gs_profile_url)
    sources_extra = _combine_sources(ex.extra_urls)

    # 1) US_University_Affiliation (critical)
    node_us = evaluator.add_sequential(
        id="US_University_Affiliation",
        desc="The researcher must be current faculty at a research university located in the United States",
        parent=rubric_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_present(ex.name) and _present(ex.university) and len(sources_affil) > 0),
        id="US_University_Affiliation_evidence",
        desc="Affiliation info and at least one supporting URL are provided",
        parent=node_us,
        critical=True
    )
    leaf_us_verify = evaluator.add_leaf(
        id="US_University_Affiliation_verify",
        desc="Verify the person is currently faculty at a U.S. research university",
        parent=node_us,
        critical=True
    )
    claim_us = f"{ex.name} is currently a faculty member at {ex.university}, which is a research university located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=leaf_us_verify,
        sources=sources_affil,
        additional_instruction="Use the provided URLs (e.g., university/department/faculty profile pages or the affiliation line on Google Scholar) to confirm two things: (1) the person currently holds a faculty position (Assistant/Associate/Full Professor, Teaching/Research Professor, Lecturer, etc.) at the institution; and (2) the institution is in the United States. If the page lists an address/country or is clearly a U.S. university, count as U.S. research university."
    )

    # 2) CS_Department_Affiliation (critical)
    node_cs = evaluator.add_sequential(
        id="CS_Department_Affiliation",
        desc="Affiliation with Computer Science or a closely related department (e.g., EECS, CSE, Computer Engineering)",
        parent=rubric_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_present(ex.department) and len(sources_affil) > 0),
        id="CS_Department_Affiliation_evidence",
        desc="Department info provided with at least one supporting URL",
        parent=node_cs,
        critical=True
    )
    leaf_cs_verify = evaluator.add_leaf(
        id="CS_Department_Affiliation_verify",
        desc="Verify department is CS or a closely related CS unit",
        parent=node_cs,
        critical=True
    )
    claim_cs = f"{ex.name} is affiliated with the department '{ex.department}', which is a Computer Science department or closely related CS unit."
    await evaluator.verify(
        claim=claim_cs,
        node=leaf_cs_verify,
        sources=sources_affil,
        additional_instruction="From the provided pages, determine whether the department or unit is Computer Science or a close variant (e.g., Computer Science and Engineering, EECS/Electrical Engineering and Computer Science, Computer Engineering, School of Computer Science, CSE). Allow reasonable naming variations."
    )

    # 3) Google_Scholar_Profile (critical)
    node_gs = evaluator.add_sequential(
        id="Google_Scholar_Profile",
        desc="The researcher has a publicly accessible Google Scholar profile showing publications, citations, and h-index",
        parent=rubric_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_present(ex.gs_profile_url),
        id="Google_Scholar_Profile_url_provided",
        desc="Google Scholar profile URL is provided",
        parent=node_gs,
        critical=True
    )
    leaf_gs_verify = evaluator.add_leaf(
        id="Google_Scholar_Profile_verify",
        desc="Verify the Google Scholar profile displays publications, citations, and h-index",
        parent=node_gs,
        critical=True
    )
    claim_gs = "This Google Scholar profile page presents the researcher's publications, total citations, and h-index."
    await evaluator.verify(
        claim=claim_gs,
        node=leaf_gs_verify,
        sources=sources_gs,
        additional_instruction="Open the Google Scholar profile and confirm that it displays (a) a list of publications, (b) the 'Citations' count, and (c) the 'h-index'. Slight UI variations are acceptable as long as those metrics are shown."
    )

    # 4) AAAI_2026_Publication (critical)
    node_aaai = evaluator.add_sequential(
        id="AAAI_2026_Publication",
        desc="The researcher has at least one AAAI 2026 paper (Jan 20–27, 2026, Singapore), verifiable via Scholar or AAAI proceedings",
        parent=rubric_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(sources_aaai) > 0),
        id="AAAI_2026_Publication_evidence",
        desc="At least one evidence URL is provided for AAAI 2026 (e.g., AAAI proceedings page or Google Scholar profile)",
        parent=node_aaai,
        critical=True
    )
    leaf_aaai_verify = evaluator.add_leaf(
        id="AAAI_2026_Publication_verify",
        desc="Verify at least one AAAI 2026 publication by the researcher",
        parent=node_aaai,
        critical=True
    )
    # Prepare helpful context in additional instruction
    aaai_titles = [p.title for p in ex.aaai_2026_papers if _present(p.title)] if ex.aaai_2026_papers else []
    titles_hint = f"Candidate titles from the answer: {aaai_titles}" if aaai_titles else "No specific AAAI titles provided in the answer."
    claim_aaai = f"{ex.name} has at least one paper accepted to AAAI 2026, held January 20–27, 2026 in Singapore."
    await evaluator.verify(
        claim=claim_aaai,
        node=leaf_aaai_verify,
        sources=sources_aaai,
        additional_instruction=(
            "Confirm that at least one provided URL supports that the person authored a paper in AAAI 2026. "
            "Evidence might appear on: (1) AAAI 2026 proceedings pages; or (2) the Google Scholar profile listing a paper explicitly tagged with 'AAAI 2026' or 'Proceedings of the AAAI Conference on Artificial Intelligence 2026'. "
            f"{titles_hint} Allow minor formatting/name variants."
        )
    )

    # 5) H_Index_Threshold (critical)
    node_hidx = evaluator.add_sequential(
        id="H_Index_Threshold",
        desc="h-index is at least 30 according to Google Scholar profile",
        parent=rubric_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(sources_gs) > 0),
        id="H_Index_Threshold_evidence",
        desc="Google Scholar profile URL available for h-index verification",
        parent=node_hidx,
        critical=True
    )
    leaf_hidx_verify = evaluator.add_leaf(
        id="H_Index_Threshold_verify",
        desc="Verify h-index >= 30 on Google Scholar profile",
        parent=node_hidx,
        critical=True
    )
    claim_hidx = "According to the Google Scholar profile page, the researcher's h-index is at least 30."
    await evaluator.verify(
        claim=claim_hidx,
        node=leaf_hidx_verify,
        sources=sources_gs,
        additional_instruction="Read the 'h-index' metric on the profile (not i10-index). If it is 30 or higher, pass. Allow minor rounding or display variations, but ensure the threshold is clearly met."
    )

    # 6) Citation_Count (critical)
    node_cite = evaluator.add_sequential(
        id="Citation_Count",
        desc="Total citations are at least 5,000 according to Google Scholar profile",
        parent=rubric_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(sources_gs) > 0),
        id="Citation_Count_evidence",
        desc="Google Scholar profile URL available for citation verification",
        parent=node_cite,
        critical=True
    )
    leaf_cite_verify = evaluator.add_leaf(
        id="Citation_Count_verify",
        desc="Verify total citations >= 5,000 on Google Scholar profile",
        parent=node_cite,
        critical=True
    )
    claim_cite = "According to the Google Scholar profile page, the researcher's total citations are at least 5,000."
    await evaluator.verify(
        claim=claim_cite,
        node=leaf_cite_verify,
        sources=sources_gs,
        additional_instruction="Check the 'Citations' total on the profile. If the total is 5,000 or more, pass. Allow reasonable rounding (e.g., 5k shown explicitly)."
    )

    # 7) International_Collaboration (critical)
    node_intl = evaluator.add_sequential(
        id="International_Collaboration",
        desc="At least one publication includes an international co-author (affiliated outside the United States)",
        parent=rubric_root,
        critical=True
    )
    intl = ex.international_collab_example
    intl_urls = _combine_sources(intl.urls if intl else [])
    evaluator.add_custom_node(
        result=(intl is not None and _present(intl.coauthor_name) and _present(intl.coauthor_institution) and len(intl_urls) > 0),
        id="International_Collaboration_evidence",
        desc="International-collaboration publication and its URL(s) plus co-author name and institution are provided",
        parent=node_intl,
        critical=True
    )
    leaf_intl_verify = evaluator.add_leaf(
        id="International_Collaboration_verify",
        desc="Verify publication shows a non-U.S. co-author affiliation",
        parent=node_intl,
        critical=True
    )
    pub_title = intl.title if intl and _present(intl.title) else "the referenced publication"
    co_name = intl.coauthor_name if intl and _present(intl.coauthor_name) else "the specified co-author"
    co_inst = intl.coauthor_institution if intl and _present(intl.coauthor_institution) else "the specified institution"
    claim_intl = (
        f"The publication '{pub_title}' lists {co_name} as a co-author affiliated with {co_inst}, "
        "and that institution is located outside the United States. The named researcher is also an author on this publication."
    )
    await evaluator.verify(
        claim=claim_intl,
        node=leaf_intl_verify,
        sources=intl_urls,
        additional_instruction="Use the publication/publisher page to confirm: (1) the researcher's authorship; (2) the named co-author appears; and (3) the co-author's institution is outside the U.S. (often indicated by the country, city, or institutional location). If the page explicitly lists a non-U.S. country or clearly non-U.S. institution, pass."
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
    Evaluate an answer for the 'Identify a CS faculty with AAAI 2026 and strong Scholar profile' task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researcher_info(),
        template_class=ResearcherExtraction,
        extraction_name="researcher_extraction"
    )

    # Build and run verification checks
    await build_verification_tree(evaluator, root, extracted)

    # Return final summary
    return evaluator.get_summary()