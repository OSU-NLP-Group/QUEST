import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bingbin_liu_lineage_2024"
TASK_DESCRIPTION = (
    "Starting with Bingbin Liu's 2024 PhD dissertation from Carnegie Mellon University's Machine Learning Department, "
    "perform the following multi-step investigation: (1) Identify the full title of Bingbin Liu's dissertation. "
    "(2) In Chapter 1 (Introduction) of this dissertation, locate the paper cited as [Cohen et al., 2021]. Identify the "
    "complete title of this cited paper and the conference/venue where it was published. (3) Identify the first author of "
    "the [Cohen et al., 2021] paper by full name. (4) Determine whether this first author also completed a PhD at Carnegie "
    "Mellon University. If so, identify the title and year of their PhD dissertation. (5) Compare the dissertation committees "
    "of both Bingbin Liu and the first author you identified. Find at least one faculty member who served on both dissertation "
    "committees. (6) For the overlapping committee member you identified, determine their current institutional affiliation "
    "as of 2024, including the specific institution name and their department or role. Provide all answers with supporting "
    "URL references from official sources."
)

# Ground truth expectations (used to build verification claims)
EXPECTED = {
    "dissertation_title": "Guiding Machine Learning Design With Insights From Simple Sandboxes",
    "dissertation_author": "Bingbin Liu",
    "dissertation_year": "2024",
    "dissertation_institution": "Carnegie Mellon University",
    "dissertation_department": "Machine Learning Department",

    "cited_paper_title": "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability",
    "cited_paper_venue": "ICLR 2021",
    "cited_paper_first_author": "Jeremy M. Cohen",

    "cohen_dissertation_title": "The Dynamics of Optimization in Deep Learning",
    "cohen_dissertation_year": "2024",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DissertationInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    institution: Optional[str] = None
    department: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CitedPaperInfo(BaseModel):
    title: Optional[str] = None
    venue: Optional[str] = None
    first_author: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CohenPhDInfo(BaseModel):
    dissertation_title: Optional[str] = None
    dissertation_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CommitteeInfo(BaseModel):
    liu_committee: List[str] = Field(default_factory=list)
    cohen_committee: List[str] = Field(default_factory=list)
    liu_committee_urls: List[str] = Field(default_factory=list)
    cohen_committee_urls: List[str] = Field(default_factory=list)
    overlapping_member: Optional[str] = None
    overlap_urls: List[str] = Field(default_factory=list)


class AffiliationInfo(BaseModel):
    institution_name: Optional[str] = None
    department_or_role: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class InvestigationExtraction(BaseModel):
    dissertation: Optional[DissertationInfo] = None
    cited_paper: Optional[CitedPaperInfo] = None
    cohen_phd: Optional[CohenPhDInfo] = None
    committees: Optional[CommitteeInfo] = None
    affiliation: Optional[AffiliationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_investigation() -> str:
    return """
    Extract the structured information explicitly presented in the answer for the following sections.
    Only extract values that are explicitly stated in the answer. For each section, also extract all supporting URLs
    provided in the answer (official or accepted sources). Return null for any field that is missing.

    1) Starting Dissertation (Bingbin Liu, CMU Machine Learning, 2024)
       - dissertation.title: full dissertation title
       - dissertation.author: full author name
       - dissertation.year: completion year
       - dissertation.institution: institution name
       - dissertation.department: department name
       - dissertation.urls: array of all supporting URLs provided for the dissertation (official or accepted sources)

    2) Cited Paper [Cohen et al., 2021] from Chapter 1 (Introduction)
       - cited_paper.title: full title of the cited paper
       - cited_paper.venue: conference/venue name (e.g., ICLR 2021)
       - cited_paper.first_author: full name of the first author
       - cited_paper.urls: array of all supporting URLs provided for the cited paper and/or citation location (official or accepted sources)

    3) First Author (Jeremy M. Cohen) CMU PhD details
       - cohen_phd.dissertation_title: full title of Jeremy M. Cohen's PhD dissertation
       - cohen_phd.dissertation_year: year of the dissertation
       - cohen_phd.urls: array of supporting URLs confirming CMU PhD and dissertation details (official or accepted sources)

    4) Dissertation Committees and Overlap
       - committees.liu_committee: array of committee member names for Bingbin Liu's dissertation
       - committees.liu_committee_urls: array of URLs that list Bingbin Liu's committee (official or accepted sources)
       - committees.cohen_committee: array of committee member names for Jeremy M. Cohen's dissertation
       - committees.cohen_committee_urls: array of URLs that list Jeremy M. Cohen's committee (official or accepted sources)
       - committees.overlapping_member: at least one overlapping committee member name (served on both committees), if identified
       - committees.overlap_urls: array of URLs that support the overlap/connection (official or accepted sources)

    5) Overlapping Member Current Affiliation (as of 2024)
       - affiliation.institution_name: current institution name (as of 2024)
       - affiliation.department_or_role: current department or role (as of 2024)
       - affiliation.urls: array of supporting URLs for affiliation (official or accepted sources)

    SPECIAL RULES FOR URL EXTRACTION:
    - The sources must be explicitly mentioned as URLs in the answer text. Extract actual URLs (including markdown link targets).
    - Include only valid URLs. If a URL is missing protocol, prepend http://.
    - Do not invent or infer URLs; return empty arrays when none are provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_OFFICIAL_SUFFIXES = [
    "cmu.edu",          # University site (and all subdomains)
    "ml.cmu.edu",       # Department site
    "cs.cmu.edu",
    "ece.cmu.edu",
    "lti.cs.cmu.edu",
    "kilthub.cmu.edu",  # CMU repository
    "library.cmu.edu",
    "openreview.net",   # Conference peer-review platform (ICLR)
    "iclr.cc",          # ICLR official
    "arxiv.org",        # Academic repository
    "scholar.google.com",  # Verified academic profile
    "proquest.com",     # Dissertation repository (accepted source)
]

def is_official_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(netloc.endswith(sfx) for sfx in ALLOWED_OFFICIAL_SUFFIXES)

def count_official_urls(urls: Optional[List[str]]) -> int:
    if not urls:
        return 0
    return sum(1 for u in urls if is_official_url(u))

def union_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_step1_dissertation(evaluator: Evaluator, parent_node, data: InvestigationExtraction) -> None:
    step_node = evaluator.add_parallel(
        id="Step1_Starting_Dissertation",
        desc="Identify the starting 2024 CMU Machine Learning PhD dissertation by Bingbin Liu",
        parent=parent_node,
        critical=False,
    )
    diss = data.dissertation or DissertationInfo()

    # Critical URL support existence (official/accepted source required)
    step1_url_support = evaluator.add_custom_node(
        result=count_official_urls(diss.urls) >= 1,
        id="Step1_URL_Support",
        desc="Provide at least one supporting URL for the starting dissertation from an official/accepted source",
        parent=step_node,
        critical=True,
    )

    # Title verification
    title_node = evaluator.add_leaf(
        id="Dissertation_Title",
        desc=f"The dissertation title is '{EXPECTED['dissertation_title']}'",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Bingbin Liu's PhD dissertation title is '{EXPECTED['dissertation_title']}'.",
        node=title_node,
        sources=diss.urls,
        additional_instruction="Verify the exact dissertation title using official CMU or repository pages. Minor punctuation/casing variations are acceptable."
    )

    # Author verification
    author_node = evaluator.add_leaf(
        id="Dissertation_Author",
        desc="The dissertation author is Bingbin Liu",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The author of the dissertation is Bingbin Liu.",
        node=author_node,
        sources=diss.urls,
        additional_instruction="Verify the dissertation author on official CMU or repository pages."
    )

    # Year verification
    year_node = evaluator.add_leaf(
        id="Dissertation_Year",
        desc="The dissertation completion year is 2024",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The dissertation completion year is 2024.",
        node=year_node,
        sources=diss.urls,
        additional_instruction="Verify the year on the official dissertation or program pages; '2024' should be stated."
    )

    # Institution verification
    inst_node = evaluator.add_leaf(
        id="Dissertation_Institution",
        desc="The institution is Carnegie Mellon University",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The dissertation was completed at Carnegie Mellon University.",
        node=inst_node,
        sources=diss.urls,
        additional_instruction="Verify the institution, looking for 'Carnegie Mellon University' on official sources."
    )

    # Department verification
    dept_node = evaluator.add_leaf(
        id="Dissertation_Department",
        desc="The department is Carnegie Mellon University's Machine Learning Department",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The dissertation is from CMU's Machine Learning Department.",
        node=dept_node,
        sources=diss.urls,
        additional_instruction="Verify the department (Machine Learning Department) from CMU official or repository pages."
    )


async def verify_step2_cited_paper(evaluator: Evaluator, parent_node, data: InvestigationExtraction) -> None:
    step_node = evaluator.add_parallel(
        id="Step2_Cited_Paper_From_Chapter1",
        desc="Identify the paper cited as [Cohen et al., 2021] in Chapter 1 (Introduction) of the starting dissertation",
        parent=parent_node,
        critical=False,
    )
    cp = data.cited_paper or CitedPaperInfo()

    # Critical URL support existence
    step2_url_support = evaluator.add_custom_node(
        result=count_official_urls(cp.urls) >= 1,
        id="Step2_URL_Support",
        desc="Provide at least one supporting URL for the cited-paper identification (citation location and/or paper record) from an official/accepted source",
        parent=step_node,
        critical=True,
    )

    # Paper title verification
    title_node = evaluator.add_leaf(
        id="Cited_Paper_Title",
        desc=f"The cited paper title is '{EXPECTED['cited_paper_title']}'",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The paper '[Cohen et al., 2021]' is titled '{EXPECTED['cited_paper_title']}'.",
        node=title_node,
        sources=cp.urls,
        additional_instruction="Use official records (ICLR/OpenReview/arXiv) to confirm the exact title. Minor casing/punctuation variations acceptable."
    )

    # Venue verification
    venue_node = evaluator.add_leaf(
        id="Cited_Paper_Venue",
        desc="The cited paper venue is ICLR 2021",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper '[Cohen et al., 2021]' was published at ICLR 2021 (International Conference on Learning Representations 2021).",
        node=venue_node,
        sources=cp.urls,
        additional_instruction="Confirm that the venue is ICLR 2021 via official conference or OpenReview records."
    )

    # First author verification
    fa_node = evaluator.add_leaf(
        id="Cited_Paper_First_Author",
        desc="The cited paper first author is Jeremy M. Cohen",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The first author of '[Cohen et al., 2021]' is Jeremy M. Cohen.",
        node=fa_node,
        sources=cp.urls,
        additional_instruction="Check the author order on official records; ensure Jeremy M. Cohen is listed first."
    )


async def verify_step3_cohen_phd(evaluator: Evaluator, parent_node, data: InvestigationExtraction) -> None:
    step_node = evaluator.add_sequential(
        id="Step3_First_Author_CMU_PhD",
        desc="Determine whether the first author completed a PhD at Carnegie Mellon University; if so, identify dissertation title and year",
        parent=parent_node,
        critical=False,
    )
    cphd = data.cohen_phd or CohenPhDInfo()

    # Critical: verification that Jeremy M. Cohen completed a CMU PhD
    completed_node = evaluator.add_leaf(
        id="Cohen_Completed_CMU_PhD",
        desc="Verify that Jeremy M. Cohen completed a PhD at Carnegie Mellon University",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Jeremy M. Cohen completed a PhD at Carnegie Mellon University.",
        node=completed_node,
        sources=cphd.urls,
        additional_instruction="Use official CMU departmental/repository pages or accepted sources to confirm CMU PhD completion."
    )

    # Non-critical: dissertation details (parallel)
    details_node = evaluator.add_parallel(
        id="Cohen_Dissertation_Details",
        desc="Provide Jeremy M. Cohen's PhD dissertation title and year",
        parent=step_node,
        critical=False,
    )

    # Critical URL support under details
    step3_url_support = evaluator.add_custom_node(
        result=count_official_urls(cphd.urls) >= 1,
        id="Step3_URL_Support",
        desc="Provide at least one supporting URL for Jeremy M. Cohen's PhD/dissertation details from an official/accepted source",
        parent=details_node,
        critical=True,
    )

    # Title verification
    cohen_title_node = evaluator.add_leaf(
        id="Cohen_Dissertation_Title",
        desc=f"Jeremy M. Cohen's dissertation title is '{EXPECTED['cohen_dissertation_title']}'",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Jeremy M. Cohen's PhD dissertation title is '{EXPECTED['cohen_dissertation_title']}'.",
        node=cohen_title_node,
        sources=cphd.urls,
        additional_instruction="Confirm the exact dissertation title on official CMU or repository pages."
    )

    # Year verification
    cohen_year_node = evaluator.add_leaf(
        id="Cohen_Dissertation_Year",
        desc=f"Jeremy M. Cohen's dissertation year is {EXPECTED['cohen_dissertation_year']}",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Jeremy M. Cohen's PhD dissertation year is {EXPECTED['cohen_dissertation_year']}.",
        node=cohen_year_node,
        sources=cphd.urls,
        additional_instruction="Confirm the dissertation year (2024) on official CMU or repository pages."
    )


async def verify_step4_overlap_and_affiliation(evaluator: Evaluator, parent_node, data: InvestigationExtraction) -> None:
    step_node = evaluator.add_sequential(
        id="Step4_Committee_Overlap_And_Affiliation",
        desc="Identify at least one overlapping committee member (or allowed connection per constraints) and provide their 2024 affiliation",
        parent=parent_node,
        critical=False,
    )
    committees = data.committees or CommitteeInfo()
    affiliation = data.affiliation or AffiliationInfo()

    # Overlap identification (parallel)
    overlap_node = evaluator.add_parallel(
        id="Overlap_Member_Identification",
        desc="Identify at least one faculty member connected to both dissertations as allowed by the constraints, with evidence",
        parent=step_node,
        critical=False,
    )

    # Overlapping member name (critical leaf)
    overlap_name_node = evaluator.add_leaf(
        id="Overlapping_Member_Name",
        desc="Name at least one overlapping/connected faculty member for both dissertations (as permitted by the constraints)",
        parent=overlap_node,
        critical=True,
    )
    combined_committee_urls = union_urls(committees.overlap_urls, committees.liu_committee_urls, committees.cohen_committee_urls)
    await evaluator.verify(
        claim=f"There is at least one overlapping dissertation committee member between Bingbin Liu and Jeremy M. Cohen: {committees.overlapping_member or ''}.",
        node=overlap_name_node,
        sources=combined_committee_urls,
        additional_instruction="Verify that the named faculty member served on both dissertation committees (or permitted connection). Use official university or repository pages."
    )

    # Overlap evidence URL (critical leaf to ensure evidence exists and supports connection)
    overlap_evidence_node = evaluator.add_leaf(
        id="Overlap_Evidence_URL",
        desc="Provide URL evidence supporting the claimed overlap/connection to both dissertations from an official/accepted source",
        parent=overlap_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided URLs confirm that the overlapping faculty member served on both Bingbin Liu's and Jeremy M. Cohen's dissertation committees.",
        node=overlap_evidence_node,
        sources=combined_committee_urls,
        additional_instruction="Confirm via official committee listings or dissertation documents for both individuals."
    )

    # Current affiliation (parallel)
    aff_node = evaluator.add_parallel(
        id="Overlapping_Member_Current_Affiliation_2024",
        desc="Provide the overlapping member's current institutional affiliation as of 2024",
        parent=step_node,
        critical=False,
    )

    # Institution name (critical leaf)
    aff_inst_node = evaluator.add_leaf(
        id="Affiliation_Institution_Name",
        desc="Provide the current institution name (as of 2024)",
        parent=aff_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of 2024, the overlapping committee member's institution is '{affiliation.institution_name or ''}'.",
        node=aff_inst_node,
        sources=affiliation.urls,
        additional_instruction="Use official institutional pages or verified academic profiles to confirm institution."
    )

    # Department/Role (critical leaf)
    aff_role_node = evaluator.add_leaf(
        id="Affiliation_Department_Or_Role",
        desc="Provide the department or role at that institution (as of 2024)",
        parent=aff_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of 2024, their department or role is '{affiliation.department_or_role or ''}'.",
        node=aff_role_node,
        sources=affiliation.urls,
        additional_instruction="Verify department/role on the official institutional page; minor title variants acceptable."
    )

    # Affiliation URL support (critical custom)
    aff_url_support_node = evaluator.add_custom_node(
        result=count_official_urls(affiliation.urls) >= 1,
        id="Affiliation_URL_Support",
        desc="Provide a supporting URL verifying the 2024 affiliation from an official/accepted source",
        parent=aff_node,
        critical=True,
    )


def check_global_url_requirement(data: InvestigationExtraction) -> bool:
    diss_urls = (data.dissertation.urls if data.dissertation else []) or []
    cp_urls = (data.cited_paper.urls if data.cited_paper else []) or []
    cphd_urls = (data.cohen_phd.urls if data.cohen_phd else []) or []
    overlap_urls = (data.committees.overlap_urls if data.committees else []) or []
    aff_urls = (data.affiliation.urls if data.affiliation else []) or []

    # Each required section must have at least one official/accepted URL
    sections_ok = (
        count_official_urls(diss_urls) >= 1 and
        count_official_urls(cp_urls) >= 1 and
        count_official_urls(cphd_urls) >= 1 and
        count_official_urls(overlap_urls) >= 1 and
        count_official_urls(aff_urls) >= 1
    )
    return sections_ok


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
) -> Dict[str, Any]:
    """
    Evaluate the multi-step research lineage investigation answer.
    """
    # Initialize evaluator with sequential aggregation at root per rubric
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
        default_model=model,
    )

    # Record ground truth expectations
    evaluator.add_ground_truth({
        "expected": EXPECTED
    }, gt_type="ground_truth")

    # Extract structured information from the answer
    extraction: InvestigationExtraction = await evaluator.extract(
        prompt=prompt_extract_investigation(),
        template_class=InvestigationExtraction,
        extraction_name="investigation_extraction",
    )

    # Build verification tree by steps
    await verify_step1_dissertation(evaluator, root, extraction)
    await verify_step2_cited_paper(evaluator, root, extraction)
    await verify_step3_cohen_phd(evaluator, root, extraction)
    await verify_step4_overlap_and_affiliation(evaluator, root, extraction)

    # Global URL requirement (critical): Ensure official/accepted URLs support all required claims
    global_ok = check_global_url_requirement(extraction)
    evaluator.add_custom_node(
        result=global_ok,
        id="Global_URL_Requirement",
        desc="All required claims are supported with URL references from official university websites, academic repositories, or verified academic profiles",
        parent=root,
        critical=True,
    )

    # Record custom info: URL statistics
    all_urls = union_urls(
        extraction.dissertation.urls if extraction.dissertation else [],
        extraction.cited_paper.urls if extraction.cited_paper else [],
        extraction.cohen_phd.urls if extraction.cohen_phd else [],
        extraction.committees.liu_committee_urls if extraction.committees else [],
        extraction.committees.cohen_committee_urls if extraction.committees else [],
        extraction.committees.overlap_urls if extraction.committees else [],
        extraction.affiliation.urls if extraction.affiliation else [],
    )
    evaluator.add_custom_info(
        info={
            "total_urls_collected": len(all_urls),
            "official_urls_count": count_official_urls(all_urls),
            "allowed_official_suffixes": ALLOWED_OFFICIAL_SUFFIXES,
            "all_urls": all_urls,
        },
        info_type="url_statistics",
    )

    # Return structured result
    return evaluator.get_summary()