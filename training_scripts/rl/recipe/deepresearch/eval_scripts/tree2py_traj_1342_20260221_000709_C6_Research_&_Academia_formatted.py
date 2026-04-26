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
TASK_ID = "3i_atlas_mnras_letters_eval"
TASK_DESCRIPTION = (
    "Identify a research paper about the interstellar object 3I/ATLAS (also known as C/2025 N1) that meets all of the "
    "following requirements: (1) The paper must be published in Monthly Notices of the Royal Astronomical Society: Letters "
    "(MNRAS Letters); (2) The paper must have been submitted in July 2025; (3) The paper must comply with MNRAS Letters' "
    "5-page limit; (4) The first author must be affiliated with a research institution located in California, United States; "
    "(5) The first author's institution must be a corporation, not a university; (6) The first author's institution must have "
    "been founded in the 1990s; (7) The paper must represent international collaboration with co-authors from at least 5 "
    "different countries; (8) At least one co-author must be affiliated with a university in Hawaii; (9) The Hawaiian university "
    "must have a dedicated institute or department specifically for astronomy research. Provide the paper title, first author name, "
    "and the first author's institutional affiliation."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperMetadata(BaseModel):
    paper_title: Optional[str] = None
    paper_urls: List[str] = Field(default_factory=list)
    journal_name: Optional[str] = None
    journal_urls: List[str] = Field(default_factory=list)
    submission_date_text: Optional[str] = None
    page_count_text: Optional[str] = None
    first_author_name: Optional[str] = None
    first_author_institution: Optional[str] = None


class AuthorInstitutionInfo(BaseModel):
    first_author_name: Optional[str] = None
    first_author_institution: Optional[str] = None
    institution_urls: List[str] = Field(default_factory=list)
    institution_location_state: Optional[str] = None
    institution_location_country: Optional[str] = None
    institution_type: Optional[str] = None
    institution_founding_year_text: Optional[str] = None


class CollaborationInfo(BaseModel):
    coauthor_countries: List[str] = Field(default_factory=list)
    hawaii_coauthor_present: Optional[bool] = None
    hawaiian_coauthor_university_name: Optional[str] = None
    hawaiian_university_urls: List[str] = Field(default_factory=list)
    astronomy_unit_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_metadata() -> str:
    return (
        "From the answer, extract the core paper metadata strictly as stated. Do not invent or infer missing items.\n"
        "Fields to extract:\n"
        "- paper_title: The exact title of the paper.\n"
        "- paper_urls: All URLs that point directly to the paper or its journal page and can confirm subject and metadata.\n"
        "- journal_name: The journal name (e.g., 'Monthly Notices of the Royal Astronomical Society: Letters').\n"
        "- journal_urls: URLs that specifically identify the journal venue and article page.\n"
        "- submission_date_text: Any text in the answer that states the submission or received date (e.g., 'Submitted July 2025').\n"
        "- page_count_text: Any page length indication as written (e.g., '5 pages', 'L45–L49').\n"
        "- first_author_name: The first author's full name as written.\n"
        "- first_author_institution: The first author's institution as written.\n"
        "Return null for any missing field. For URLs, include full URLs only. If the answer lists markdown links, extract the actual URLs."
    )


def prompt_extract_institution_info() -> str:
    return (
        "From the answer, extract information about the first author's institution strictly as written. Do not infer.\n"
        "Fields to extract:\n"
        "- first_author_name: The first author's name as written.\n"
        "- first_author_institution: The institution name.\n"
        "- institution_urls: URLs that confirm the institution's location, type (corporation vs university), and founding year.\n"
        "- institution_location_state: The U.S. state if stated (e.g., 'California').\n"
        "- institution_location_country: The country if stated (e.g., 'United States').\n"
        "- institution_type: As written in the answer (e.g., 'corporation', 'nonprofit', 'company', 'university').\n"
        "- institution_founding_year_text: Founding year information as written (e.g., 'Founded in 1997').\n"
        "Return null for any missing field. For URLs, include full URLs only."
    )


def prompt_extract_collaboration_info() -> str:
    return (
        "From the answer, extract the collaboration and Hawaiian institution details strictly as written. Do not infer.\n"
        "Fields to extract:\n"
        "- coauthor_countries: A list of country names mentioned for authors' affiliations.\n"
        "- hawaii_coauthor_present: Whether the answer states at least one co-author is affiliated with a university in Hawaii (true/false).\n"
        "- hawaiian_coauthor_university_name: The Hawaiian university name (e.g., 'University of Hawai‘i').\n"
        "- hawaiian_university_urls: URLs for the Hawaiian university or its astronomy institute/department.\n"
        "- astronomy_unit_name: The name of a dedicated astronomy institute or department (e.g., 'Institute for Astronomy').\n"
        "Return null for any missing field. For URLs, include full URLs only."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _combine_sources(*lists: List[str]) -> List[str]:
    out: List[str] = []
    for lst in lists:
        for u in lst:
            if u and u not in out:
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_paper_discovery_and_core_metadata(
    evaluator: Evaluator,
    parent_node,
    paper: PaperMetadata,
) -> None:
    node = evaluator.add_sequential(
        id="Paper_Discovery_and_Core_Metadata",
        desc="Identification of the paper and verification of its core research subject",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence first (to gate subsequent checks)
    paper_urls = _nonempty_urls(paper.paper_urls)
    ref_exists = len(paper_urls) > 0
    ref_node = evaluator.add_custom_node(
        result=ref_exists,
        id="Paper_Reference_URL",
        desc="URL reference to the paper confirming its subject matter and metadata",
        parent=node,
        critical=False,
    )

    # Subject verification: 3I/ATLAS (C/2025 N1)
    topic_leaf = evaluator.add_leaf(
        id="3I_ATLAS_Research_Topic",
        desc="The paper's primary subject is the interstellar object 3I/ATLAS (also known as C/2025 N1)",
        parent=node,
        critical=True,
    )
    claim = "This paper concerns the interstellar object 3I/ATLAS, also designated C/2025 N1."
    await evaluator.verify(
        claim=claim,
        node=topic_leaf,
        sources=paper_urls,
        additional_instruction=(
            "Confirm that the article page explicitly refers to '3I/ATLAS' or its designation 'C/2025 N1'. "
            "Minor naming variations are acceptable, including hyphens or slashes."
        ),
    )


async def build_journal_and_publication_compliance(
    evaluator: Evaluator,
    parent_node,
    paper: PaperMetadata,
) -> None:
    node = evaluator.add_sequential(
        id="Journal_and_Publication_Compliance",
        desc="Verification of journal publication requirements and submission timeline",
        parent=parent_node,
        critical=False,
    )

    # Journal reference URL existence first
    journal_sources = _nonempty_urls(paper.journal_urls) or _nonempty_urls(paper.paper_urls)
    journal_ref_node = evaluator.add_custom_node(
        result=len(journal_sources) > 0,
        id="Journal_Reference_URL",
        desc="URL reference confirming journal publication details and submission date",
        parent=node,
        critical=False,
    )

    # Venue requirements (critical group, parallel)
    venue_node = evaluator.add_parallel(
        id="Publication_Venue_Requirements",
        desc="The paper meets all publication venue and format requirements",
        parent=node,
        critical=True,
    )

    # MNRAS Letters venue
    mnras_leaf = evaluator.add_leaf(
        id="MNRAS_Letters_Journal",
        desc="The paper is published in Monthly Notices of the Royal Astronomical Society: Letters (MNRAS Letters)",
        parent=venue_node,
        critical=True,
    )
    mnras_claim = (
        "This article is published in 'Monthly Notices of the Royal Astronomical Society: Letters' (MNRAS Letters)."
    )

    # Page limit compliance (<=5 pages)
    page_leaf = evaluator.add_leaf(
        id="Five_Page_Limit_Compliance",
        desc="The paper complies with MNRAS Letters' maximum 5-page limit",
        parent=venue_node,
        critical=True,
    )
    page_claim = "The article's length is five pages or fewer, complying with the MNRAS Letters 5-page limit."

    # Submitted in July 2025
    submission_leaf = evaluator.add_leaf(
        id="July_2025_Submission",
        desc="The paper was submitted in July 2025",
        parent=venue_node,
        critical=True,
    )
    submission_claim = "The article shows a submission (or received) date in July 2025."

    # Batch verify all venue constraints
    await evaluator.batch_verify(
        [
            (
                mnras_claim,
                journal_sources,
                mnras_leaf,
                "Accept equivalent naming like 'MNRAS Letters' or 'Monthly Notices: Letters'. The venue must be the Letters section.",
            ),
            (
                page_claim,
                journal_sources,
                page_leaf,
                "Use the article page range or explicit 'pages' field to evaluate page count. If pages are like 'L45–L49', "
                "count inclusive pages and ensure the count is <= 5.",
            ),
            (
                submission_claim,
                journal_sources,
                submission_leaf,
                "On MNRAS pages, submission may be listed as 'Submitted' or 'Received'. Accept either label if clearly in July 2025.",
            ),
        ]
    )


async def build_first_author_institutional_verification(
    evaluator: Evaluator,
    parent_node,
    inst: AuthorInstitutionInfo,
) -> None:
    node = evaluator.add_sequential(
        id="First_Author_Institutional_Verification",
        desc="Verification of all requirements regarding the first author's institutional affiliation",
        parent=parent_node,
        critical=False,
    )

    inst_sources = _nonempty_urls(inst.institution_urls)

    # Institution reference URL existence first
    inst_ref_node = evaluator.add_custom_node(
        result=len(inst_sources) > 0,
        id="Institution_Reference_URL",
        desc="URL reference to the first author's institution confirming location, type, and founding date",
        parent=node,
        critical=False,
    )

    # Institution properties (critical group, parallel)
    props_node = evaluator.add_parallel(
        id="Institution_Properties",
        desc="The first author's institution meets all specified requirements",
        parent=node,
        critical=True,
    )

    # California location
    ca_leaf = evaluator.add_leaf(
        id="California_Location",
        desc="The first author's research institution is located in California, United States",
        parent=props_node,
        critical=True,
    )
    ca_claim = f"The institution '{inst.first_author_institution or 'the institution'}' is located in California, United States."

    # Corporation not university
    corp_leaf = evaluator.add_leaf(
        id="Corporation_Not_University",
        desc="The institution is a corporation and explicitly not a university or college",
        parent=props_node,
        critical=True,
    )
    corp_claim = (
        f"The institution '{inst.first_author_institution or 'the institution'}' is a corporation (e.g., company or nonprofit corporation) "
        "and is not a university or college."
    )

    # Founded in the 1990s
    founded_leaf = evaluator.add_leaf(
        id="Founded_1990s",
        desc="The institution was founded in the 1990s (1990-1999)",
        parent=props_node,
        critical=True,
    )
    founded_claim = f"The institution '{inst.first_author_institution or 'the institution'}' was founded between 1990 and 1999."

    await evaluator.batch_verify(
        [
            (
                ca_claim,
                inst_sources,
                ca_leaf,
                "Confirm the institution's location or headquarters is in California, USA. Accept official website or Wikipedia.",
            ),
            (
                corp_claim,
                inst_sources,
                corp_leaf,
                "Confirm the institution is organized as a corporation (including nonprofit corporation) and is not a university or college.",
            ),
            (
                founded_claim,
                inst_sources,
                founded_leaf,
                "Confirm the founding year lies within 1990–1999 inclusive.",
            ),
        ]
    )


async def build_multi_national_collaboration_verification(
    evaluator: Evaluator,
    parent_node,
    paper: PaperMetadata,
    collab: CollaborationInfo,
) -> None:
    node = evaluator.add_sequential(
        id="Multi_National_Collaboration_Verification",
        desc="Verification of the paper's international collaboration scope",
        parent=parent_node,
        critical=False,
    )

    paper_sources = _nonempty_urls(paper.paper_urls)

    # Collaboration reference URL existence first
    collab_ref_node = evaluator.add_custom_node(
        result=len(paper_sources) > 0,
        id="Collaboration_Reference_URL",
        desc="URL reference confirming the international scope and Hawaiian co-author participation",
        parent=node,
        critical=False,
    )

    # Requirements (critical group, parallel)
    req_node = evaluator.add_parallel(
        id="International_Collaboration_Requirements",
        desc="The paper demonstrates required international collaboration scope",
        parent=node,
        critical=True,
    )

    # Minimum five countries
    countries_leaf = evaluator.add_leaf(
        id="Minimum_Five_Countries",
        desc="Co-authors are affiliated with institutions from at least 5 different countries",
        parent=req_node,
        critical=True,
    )
    countries_claim = "The article lists co-author affiliations from at least five distinct countries."

    # Hawaii co-author present
    hawaii_leaf = evaluator.add_leaf(
        id="Hawaii_Co_Author_Present",
        desc="At least one co-author is affiliated with a university in Hawaii, United States",
        parent=req_node,
        critical=True,
    )
    hawaii_sources = _combine_sources(paper_sources, _nonempty_urls(collab.hawaiian_university_urls))
    hawaii_claim = (
        "At least one co-author's affiliation is a university in Hawaii (e.g., University of Hawai‘i)."
    )

    await evaluator.batch_verify(
        [
            (
                countries_claim,
                paper_sources,
                countries_leaf,
                "Use the affiliations list on the article page to count unique country names. The count must be >= 5.",
            ),
            (
                hawaii_claim,
                hawaii_sources,
                hawaii_leaf,
                "Confirm that at least one author affiliation includes a Hawaiian university such as the University of Hawai‘i "
                "(Manoa or Hilo). Check the article affiliations and, if provided, the university site.",
            ),
        ]
    )


async def build_hawaiian_institution_verification(
    evaluator: Evaluator,
    parent_node,
    collab: CollaborationInfo,
) -> None:
    node = evaluator.add_sequential(
        id="Hawaiian_Institution_Verification",
        desc="Verification that the Hawaiian co-author's university has dedicated astronomy research facilities",
        parent=parent_node,
        critical=False,
    )

    hawaii_sources = _nonempty_urls(collab.hawaiian_university_urls)

    # Hawaiian institution reference URL existence first
    hawaii_ref_node = evaluator.add_custom_node(
        result=len(hawaii_sources) > 0,
        id="Hawaiian_Institution_Reference_URL",
        desc="URL reference confirming the Hawaiian university's astronomy research facilities",
        parent=node,
        critical=False,
    )

    cap_node = evaluator.add_parallel(
        id="Astronomy_Research_Capability",
        desc="The Hawaiian institution has the required astronomy research infrastructure",
        parent=node,
        critical=True,
    )

    # University status
    uni_leaf = evaluator.add_leaf(
        id="University_Status",
        desc="The Hawaiian institution is a university",
        parent=cap_node,
        critical=True,
    )
    uni_name = collab.hawaiian_coauthor_university_name or "the Hawaiian institution"
    uni_claim = f"The institution '{uni_name}' is a university."

    # Dedicated astronomy unit
    astro_leaf = evaluator.add_leaf(
        id="Dedicated_Astronomy_Unit",
        desc="The university has a dedicated institute or department specifically for astronomy research",
        parent=cap_node,
        critical=True,
    )
    unit_name = collab.astronomy_unit_name or "a dedicated astronomy institute or department"
    astro_claim = (
        f"The university '{uni_name}' has {unit_name} specifically dedicated to astronomy research."
    )

    await evaluator.batch_verify(
        [
            (
                uni_claim,
                hawaii_sources,
                uni_leaf,
                "Confirm that the institution is indeed a university (e.g., University of Hawai‘i).",
            ),
            (
                astro_claim,
                hawaii_sources,
                astro_leaf,
                "Confirm there is a dedicated institute or department for astronomy (e.g., Institute for Astronomy).",
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for the 3I/ATLAS MNRAS Letters paper identification task.
    """
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

    # Extract structured information from the answer
    paper_meta = await evaluator.extract(
        prompt=prompt_extract_paper_metadata(),
        template_class=PaperMetadata,
        extraction_name="paper_metadata",
    )
    inst_info = await evaluator.extract(
        prompt=prompt_extract_institution_info(),
        template_class=AuthorInstitutionInfo,
        extraction_name="first_author_institution",
    )
    collab_info = await evaluator.extract(
        prompt=prompt_extract_collaboration_info(),
        template_class=CollaborationInfo,
        extraction_name="collaboration_info",
    )

    # Build verification tree according to rubric
    await build_paper_discovery_and_core_metadata(evaluator, root, paper_meta)
    await build_journal_and_publication_compliance(evaluator, root, paper_meta)
    await build_first_author_institutional_verification(evaluator, root, inst_info)
    await build_multi_national_collaboration_verification(evaluator, root, paper_meta, collab_info)
    await build_hawaiian_institution_verification(evaluator, root, collab_info)

    return evaluator.get_summary()