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
TASK_ID = "ai_ml_qs2025_top10_neurips_hindex100"
TASK_DESCRIPTION = (
    "Identify three artificial intelligence or machine learning researchers who meet ALL of the following criteria: "
    "(1) Each researcher must be a current faculty member at a university that is ranked in the top 10 for Computer Science & Information Systems "
    "in the QS World University Rankings by Subject 2025; (2) The university must be located in the United States; "
    "(3) Each researcher must have published at least one paper at the NeurIPS (Neural Information Processing Systems) conference; "
    "(4) Each researcher must have an h-index of at least 100 according to Google Scholar; "
    "(5) Each researcher must have a publicly accessible Google Scholar profile. "
    "For your answer, provide: the name of each researcher, their affiliated university, a reference URL to the QS World University Rankings by Subject 2025 "
    "for Computer Science & Information Systems, each researcher's Google Scholar profile URL, and a URL verifying each university's ranking position."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QSReference(BaseModel):
    """Global QS reference URL for CS & IS 2025 subject ranking."""
    qs_reference_url: Optional[str] = None


class ResearcherInfo(BaseModel):
    """Information for a single researcher, extracted from the answer."""
    name: Optional[str] = None
    university: Optional[str] = None
    scholar_url: Optional[str] = None
    university_ranking_url: Optional[str] = None  # URL verifying the university's ranking position
    faculty_url: Optional[str] = None            # Official university faculty directory or department profile page for the researcher
    field_url: Optional[str] = None              # Page referencing AI/ML field (e.g., lab page, profile)
    neurips_urls: List[str] = Field(default_factory=list)  # URLs evidencing NeurIPS publications (DBLP/NeurIPS pages etc.)


class ResearchersExtraction(BaseModel):
    """List of up to 3 researchers."""
    researchers: List[ResearcherInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_qs_reference() -> str:
    return (
        "Extract the single most relevant URL in the answer that corresponds to the QS World University Rankings by Subject 2025 "
        "for Computer Science & Information Systems. If multiple QS URLs are present, choose the one that directly references the 2025 subject ranking for "
        "Computer Science & Information Systems. If none is present, return null."
    )


def prompt_extract_researchers() -> str:
    return (
        "Extract up to three researchers described in the answer who purportedly meet the criteria. For each researcher, extract:\n"
        "1. name: Full name of the researcher.\n"
        "2. university: The affiliated university name.\n"
        "3. scholar_url: The publicly accessible Google Scholar profile URL for this researcher (must be a profile page, not a search results page).\n"
        "4. university_ranking_url: A URL that verifies the university's ranking position for QS 2025 Computer Science & Information Systems.\n"
        "5. faculty_url: The official university faculty directory or department profile page URL for the researcher (if provided). If not provided, return null.\n"
        "6. field_url: A URL indicating the researcher's field (e.g., personal homepage, lab page) that supports AI/ML research (if provided). If not provided, return null.\n"
        "7. neurips_urls: An array of URLs that verify the researcher has published at least one paper at NeurIPS/NIPS (e.g., DBLP or NeurIPS proceedings pages). If none are provided, return an empty array.\n"
        "Return a JSON object with a 'researchers' array of up to 3 items following this schema. Only extract information explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(*url_groups: Optional[List[Optional[str]]]) -> List[str]:
    """Flatten and deduplicate URLs, drop None/empty."""
    collected: List[str] = []
    for group in url_groups:
        if not group:
            continue
        for url in group:
            if url and isinstance(url, str) and url.strip():
                if url not in collected:
                    collected.append(url)
    return collected


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_researcher(
    evaluator: Evaluator,
    parent_node,
    researcher: ResearcherInfo,
    index: int,
    qs_ref_url: Optional[str],
) -> None:
    """
    Build and verify the tree for one researcher.
    """
    # Create sequential node for this researcher
    res_node = evaluator.add_sequential(
        id=f"researcher_{index+1}",
        desc=f"Researcher #{index+1} meeting all specified criteria",
        parent=parent_node,
        critical=False,  # allow partial credit across different researchers
    )

    # ---------------- University Requirements (Critical, Parallel) ---------------- #
    uni_node = evaluator.add_parallel(
        id=f"researcher_{index+1}_university_requirements",
        desc="Verify the researcher's affiliated university meets ranking and location requirements",
        parent=res_node,
        critical=True,
    )

    # Existence: University name provided
    evaluator.add_custom_node(
        result=bool(researcher.university and researcher.university.strip()),
        id=f"researcher_{index+1}_university_name_provided",
        desc="The name of the affiliated university is provided",
        parent=uni_node,
        critical=True,
    )

    # Existence: University ranking URL provided
    evaluator.add_custom_node(
        result=bool(researcher.university_ranking_url and researcher.university_ranking_url.strip()),
        id=f"researcher_{index+1}_university_ranking_url",
        desc="Provide a URL that verifies the university's ranking position",
        parent=uni_node,
        critical=True,
    )

    # Verify Top-10 in QS 2025 CS&IS
    top10_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_top10_qs_ranking_2025",
        desc="The university is ranked in the top 10 for CS & IS in QS World University Rankings by Subject 2025",
        parent=uni_node,
        critical=True,
    )
    top10_sources = _collect_sources(
        [qs_ref_url] if qs_ref_url else None,
        [researcher.university_ranking_url] if researcher.university_ranking_url else None,
    )
    claim_top10 = (
        f"The university '{researcher.university or ''}' is ranked within the top 10 for "
        "Computer Science & Information Systems in the QS World University Rankings by Subject 2025."
    )
    await evaluator.verify(
        claim=claim_top10,
        node=top10_leaf,
        sources=top10_sources if top10_sources else None,
        additional_instruction=(
            "Use the QS 2025 CS & IS subject ranking page or the provided ranking verification URL. "
            "Confirm the university appears with a rank from 1 to 10 for the 2025 CS & IS subject list. "
            "Minor naming variations of the university are acceptable."
        ),
    )

    # Verify US Location
    us_loc_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_us_location",
        desc="The university is located in the United States",
        parent=uni_node,
        critical=True,
    )
    us_sources = _collect_sources(
        [researcher.university_ranking_url] if researcher.university_ranking_url else None,
        [qs_ref_url] if qs_ref_url else None,
    )
    claim_us = f"The university '{researcher.university or ''}' is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=us_loc_leaf,
        sources=us_sources if us_sources else None,
        additional_instruction=(
            "Confirm the country listed for the university is 'United States' or 'USA'. "
            "QS pages often display institution location; department or university pages may also show location."
        ),
    )

    # ---------------- Researcher Requirements (Critical, Parallel) ---------------- #
    rr_node = evaluator.add_parallel(
        id=f"researcher_{index+1}_requirements",
        desc="Verify the researcher meets all individual qualification criteria",
        parent=res_node,
        critical=True,
    )

    # Existence: Researcher name provided
    evaluator.add_custom_node(
        result=bool(researcher.name and researcher.name.strip()),
        id=f"researcher_{index+1}_name_provided",
        desc="The name of the researcher is provided",
        parent=rr_node,
        critical=True,
    )

    # Google Scholar profile URL - verify that it's a publicly accessible Scholar profile
    scholar_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_google_scholar_profile_url",
        desc="Provide the researcher's publicly accessible Google Scholar profile URL",
        parent=rr_node,
        critical=True,
    )
    claim_scholar = (
        f"The URL provided is a publicly accessible Google Scholar profile for '{researcher.name or ''}'. "
        "It should be a 'citations' or 'user' profile page showing metrics like 'h-index'."
    )
    await evaluator.verify(
        claim=claim_scholar,
        node=scholar_leaf,
        sources=researcher.scholar_url if (researcher.scholar_url and researcher.scholar_url.strip()) else None,
        additional_instruction=(
            "Confirm the page is a Google Scholar profile (scholar.google.com/citations?user=...). "
            "It should display profile details and metrics (e.g., h-index)."
        ),
    )

    # Current faculty position - prefer official faculty directory/department page
    faculty_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_current_faculty_position",
        desc="The researcher holds a current faculty position at the university (verifiable through official university faculty directory)",
        parent=rr_node,
        critical=True,
    )
    fac_sources = _collect_sources(
        [researcher.faculty_url] if researcher.faculty_url else None,
        [researcher.scholar_url] if researcher.scholar_url else None,
    )
    claim_faculty = (
        f"'{researcher.name or ''}' currently holds a faculty position at '{researcher.university or ''}'. "
        "Titles such as Professor, Associate Professor, Assistant Professor are considered faculty."
    )
    await evaluator.verify(
        claim=claim_faculty,
        node=faculty_leaf,
        sources=fac_sources if fac_sources else None,
        additional_instruction=(
            "Prefer an official university directory/department page that lists the researcher as current faculty. "
            "If unavailable, corroborating information on the Scholar profile may be considered but should clearly indicate a current faculty role."
        ),
    )

    # AI/ML research field
    field_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_ai_ml_research_field",
        desc="The researcher's work is in artificial intelligence, machine learning, or related computer science subfields",
        parent=rr_node,
        critical=True,
    )
    field_sources = _collect_sources(
        [researcher.field_url] if researcher.field_url else None,
        [researcher.scholar_url] if researcher.scholar_url else None,
    )
    claim_field = (
        f"'{researcher.name or ''}' works in artificial intelligence, machine learning, or closely related subfields "
        "(e.g., deep learning, computer vision, natural language processing, reinforcement learning)."
    )
    await evaluator.verify(
        claim=claim_field,
        node=field_leaf,
        sources=field_sources if field_sources else None,
        additional_instruction=(
            "Look for keywords such as 'machine learning', 'artificial intelligence', 'deep learning', 'computer vision', "
            "'natural language processing', or 'reinforcement learning' in the provided sources."
        ),
    )

    # H-index threshold (>= 100) according to Google Scholar (all-time h-index acceptable)
    hindex_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_h_index_threshold",
        desc="The researcher has an h-index of at least 100 according to Google Scholar",
        parent=rr_node,
        critical=True,
    )
    claim_hindex = (
        f"According to the provided Google Scholar profile, '{researcher.name or ''}' has an h-index of at least 100 "
        "(consider the all-time h-index metric)."
    )
    await evaluator.verify(
        claim=claim_hindex,
        node=hindex_leaf,
        sources=researcher.scholar_url if (researcher.scholar_url and researcher.scholar_url.strip()) else None,
        additional_instruction=(
            "Check the metrics section on the Scholar profile and confirm the h-index (All) is >= 100. "
            "Minor rounding differences are acceptable."
        ),
    )

    # NeurIPS publication record (at least one)
    neurips_leaf = evaluator.add_leaf(
        id=f"researcher_{index+1}_neurips_publication_record",
        desc="The researcher has published at least one paper at NeurIPS (Neural Information Processing Systems) conference",
        parent=rr_node,
        critical=True,
    )
    neurips_sources = _collect_sources(
        researcher.neurips_urls if researcher.neurips_urls else None,
        [researcher.scholar_url] if researcher.scholar_url else None,
    )
    claim_neurips = (
        f"'{researcher.name or ''}' has at least one publication at the NeurIPS (also historically called NIPS) conference."
    )
    await evaluator.verify(
        claim=claim_neurips,
        node=neurips_leaf,
        sources=neurips_sources if neurips_sources else None,
        additional_instruction=(
            "Look for 'NeurIPS' or 'NIPS' in the publication venue or title. DBLP pages or NeurIPS proceedings pages "
            "are considered valid evidence."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the AI/ML researchers at QS 2025 top-10 US universities with NeurIPS and h-index criteria.
    """
    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent checks for QS reference and each researcher
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

    # ---------------- Extraction ---------------- #
    qs_ref_task = evaluator.extract(
        prompt=prompt_extract_qs_reference(),
        template_class=QSReference,
        extraction_name="qs_reference",
    )
    researchers_task = evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction",
    )
    qs_ref, researchers_extraction = await asyncio.gather(qs_ref_task, researchers_task)

    # Ensure we have exactly 3 researchers (pad with empty if fewer; take first 3 if more)
    extracted_list = researchers_extraction.researchers[:3]
    while len(extracted_list) < 3:
        extracted_list.append(ResearcherInfo())

    # ---------------- QS Reference Verification (Critical leaf) ---------------- #
    qs_leaf = evaluator.add_leaf(
        id="qs_2025_ranking_reference",
        desc="Provide a reference URL to the QS World University Rankings by Subject 2025 for Computer Science & Information Systems",
        parent=root,
        critical=True,
    )
    claim_qs = (
        "This URL corresponds to the QS World University Rankings by Subject 2025 page for Computer Science & Information Systems."
    )
    await evaluator.verify(
        claim=claim_qs,
        node=qs_leaf,
        sources=qs_ref.qs_reference_url if (qs_ref.qs_reference_url and qs_ref.qs_reference_url.strip()) else None,
        additional_instruction=(
            "Confirm the page is on an official QS domain (e.g., topuniversities.com) and specifically references the 2025 subject ranking for "
            "Computer Science & Information Systems."
        ),
    )

    # ---------------- Researcher Verifications ---------------- #
    for i, r in enumerate(extracted_list):
        await verify_researcher(
            evaluator=evaluator,
            parent_node=root,
            researcher=r,
            index=i,
            qs_ref_url=qs_ref.qs_reference_url,
        )

    # Return evaluation summary
    return evaluator.get_summary()