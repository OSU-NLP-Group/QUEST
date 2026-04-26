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
TASK_ID = "acl2025_researcher_eval"
TASK_DESCRIPTION = """
Identify a researcher who satisfies all of the following criteria:

1. The researcher must have published at least 2 papers at ACL 2025 (The 63rd Annual Meeting of the Association for Computational Linguistics, held in Vienna, Austria, July 27 - August 1, 2025). These papers can be from the main conference (long papers or short papers) or findings papers.

2. At the time of publication, the researcher must have been affiliated with a university located in the United States.

3. The researcher must have a Google Scholar h-index of at least 10.

4. The researcher must hold an academic position as either an Assistant Professor or Associate Professor (not a Full Professor, and not a graduate student or postdoc).

5. The researcher must be listed as the first author OR corresponding author on at least one of their ACL 2025 papers.

For your answer, provide:
- The researcher's full name
- Their university affiliation
- Their current h-index from Google Scholar
- A list of their ACL 2025 papers with titles and ACL Anthology URLs
- The URL to their Google Scholar profile
- The URL to their university faculty profile page

All information must be verifiable through the provided URLs.
"""

# Limit how many papers we verify in depth to keep the evaluation efficient but thorough
MAX_PAPERS_TO_VERIFY = 3


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ACLPaper(BaseModel):
    """Information about a single ACL 2025 paper as presented in the answer."""
    title: Optional[str] = None
    acl_anthology_url: Optional[str] = None
    venue_note: Optional[str] = None  # e.g., "ACL 2025" or "Findings of ACL 2025" if the answer states it


class ResearcherSolution(BaseModel):
    """All information the answer should provide for the researcher."""
    name: Optional[str] = None
    university: Optional[str] = None
    google_scholar_url: Optional[str] = None
    scholar_h_index: Optional[str] = None  # keep as string to be robust to formats
    faculty_profile_url: Optional[str] = None
    position_title: Optional[str] = None
    acl_papers: List[ACLPaper] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher_and_papers() -> str:
    return """
    Extract the following fields from the answer exactly as stated:

    researcher fields:
    - name: The researcher's full name.
    - university: The university affiliation stated in the answer.
    - google_scholar_url: The URL to the Google Scholar profile.
    - scholar_h_index: The h-index value stated in the answer (as shown on Google Scholar).
    - faculty_profile_url: The URL to the university faculty profile page.
    - position_title: The academic position title stated (e.g., Assistant Professor, Associate Professor).

    ACL 2025 papers:
    Return an array 'acl_papers', each with:
    - title: Paper title.
    - acl_anthology_url: ACL Anthology URL provided for the paper.
    - venue_note: Any venue text provided (e.g., "ACL 2025", "Findings of ACL 2025") as written in the answer.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer.
    - If any field is missing, set it to null (or empty list for arrays).
    - Include all ACL 2025 papers the answer lists. If the answer lists more than 3 papers, we will verify only the first 3.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x or ""

def _paper_subset(papers: List[ACLPaper]) -> List[ACLPaper]:
    return [p for p in papers if (p.title or p.acl_anthology_url)][:MAX_PAPERS_TO_VERIFY]

def _has_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_solution_completeness(
    evaluator: Evaluator,
    parent,
    sol: ResearcherSolution,
    papers_to_check: List[ACLPaper],
) -> None:
    node = evaluator.add_parallel(
        id="Solution_Completeness",
        desc="Verify all required information is provided in the solution",
        parent=parent,
        critical=True,
    )

    # Researcher_Identification
    evaluator.add_custom_node(
        result=_has_text(sol.name),
        id="Researcher_Identification",
        desc="The researcher's full name is clearly provided",
        parent=node,
        critical=True,
    )

    # Paper_List_With_References (existence checks per paper)
    papers_list_node = evaluator.add_parallel(
        id="Paper_List_With_References",
        desc="All ACL 2025 papers are listed with ACL Anthology URLs or DOIs",
        parent=node,
        critical=True,
    )
    for i, p in enumerate(papers_to_check):
        evaluator.add_custom_node(
            result=_has_text(p.title) and _has_text(p.acl_anthology_url),
            id=f"paper_{i}_listed_with_reference",
            desc=f"Paper #{i+1} listed with title and ACL Anthology URL",
            parent=papers_list_node,
            critical=True,
        )

    # Verification_Sources (required URLs presence)
    sources_node = evaluator.add_parallel(
        id="Verification_Sources",
        desc="URLs for Google Scholar profile, university affiliation page, and ACL Anthology entries are provided",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(sol.google_scholar_url),
        id="Scholar_URL_Provided",
        desc="Google Scholar profile URL is provided",
        parent=sources_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(sol.faculty_profile_url),
        id="Faculty_URL_Provided",
        desc="University faculty profile URL is provided",
        parent=sources_node,
        critical=True,
    )
    # At least 2 ACL Anthology URLs present
    evaluator.add_custom_node(
        result=sum(1 for p in sol.acl_papers if _has_text(p.acl_anthology_url)) >= 2,
        id="ACL_URLs_Provided_Min2",
        desc="At least 2 ACL Anthology URLs are provided",
        parent=sources_node,
        critical=True,
    )


async def build_acl_participation(
    evaluator: Evaluator,
    parent,
    sol: ResearcherSolution,
    papers_to_check: List[ACLPaper],
) -> None:
    node = evaluator.add_parallel(
        id="ACL_2025_Participation",
        desc="Verify the researcher has sufficient participation at ACL 2025",
        parent=parent,
        critical=True,
    )

    # Minimum_Paper_Count
    min_count = sum(1 for p in sol.acl_papers if _has_text(p.title) and _has_text(p.acl_anthology_url))
    evaluator.add_custom_node(
        result=min_count >= 2,
        id="Minimum_Paper_Count",
        desc="The researcher has at least 2 papers accepted at ACL 2025 (including main conference long papers, short papers, or findings papers)",
        parent=node,
        critical=True,
    )

    # Papers_In_ACL_Anthology (per-paper verification)
    anth_idx_node = evaluator.add_parallel(
        id="Papers_In_ACL_Anthology",
        desc="All claimed ACL 2025 papers are indexed in the ACL Anthology",
        parent=node,
        critical=True,
    )
    batch_1 = []
    for i, p in enumerate(papers_to_check):
        leaf = evaluator.add_leaf(
            id=f"paper_{i}_anthology_indexed",
            desc=f"Paper #{i+1} is an ACL Anthology entry and matches the claimed title",
            parent=anth_idx_node,
            critical=True,
        )
        claim = f"This page is an ACL Anthology entry and the title matches '{_safe_str(p.title)}'."
        src = p.acl_anthology_url if _has_text(p.acl_anthology_url) else None
        batch_1.append((
            claim,
            src,
            leaf,
            "Check the page header for 'ACL Anthology' and confirm the paper title matches or is equivalent (allow minor formatting or punctuation differences).",
        ))
    await evaluator.batch_verify(batch_1)

    # Paper_Metadata_Verifiable (title & authors visible on page)
    md_node = evaluator.add_parallel(
        id="Paper_Metadata_Verifiable",
        desc="Each paper has verifiable metadata including title, authors, and publication venue",
        parent=node,
        critical=True,
    )
    batch_2 = []
    for i, p in enumerate(papers_to_check):
        leaf = evaluator.add_leaf(
            id=f"paper_{i}_metadata_visible",
            desc=f"Paper #{i+1}: Title and author list are visible on the page",
            parent=md_node,
            critical=True,
        )
        claim = "The page shows the paper title and the list of authors."
        src = p.acl_anthology_url if _has_text(p.acl_anthology_url) else None
        batch_2.append((
            claim,
            src,
            leaf,
            "Inspect the page content (and screenshot) to confirm the title and author names are present.",
        ))
    await evaluator.batch_verify(batch_2)

    # ACL_2025_Venue_Confirmation (per paper venue confirmation)
    venue_node = evaluator.add_parallel(
        id="ACL_2025_Venue_Confirmation",
        desc="Papers are confirmed to be from ACL 2025 (Vienna, Austria, July 27 - August 1, 2025)",
        parent=node,
        critical=True,
    )
    batch_3 = []
    for i, p in enumerate(papers_to_check):
        leaf = evaluator.add_leaf(
            id=f"paper_{i}_venue_confirmed",
            desc=f"Paper #{i+1}: Venue is ACL 2025 (main conference or Findings of ACL 2025)",
            parent=venue_node,
            critical=True,
        )
        claim = "This paper is part of ACL 2025 or Findings of ACL 2025 (Vienna, Austria, July 27–August 1, 2025)."
        src = p.acl_anthology_url if _has_text(p.acl_anthology_url) else None
        batch_3.append((
            claim,
            src,
            leaf,
            "Confirm the venue mentions 'ACL 2025' or 'Findings of ACL 2025' and the year 2025; minor phrasing differences are acceptable.",
        ))
    await evaluator.batch_verify(batch_3)


async def build_us_affiliation(
    evaluator: Evaluator,
    parent,
    sol: ResearcherSolution,
    papers_to_check: List[ACLPaper],
) -> None:
    node = evaluator.add_parallel(
        id="US_University_Affiliation",
        desc="Verify the researcher's affiliation with a US university",
        parent=parent,
        critical=True,
    )

    # Aggregate sources: faculty page + ACL paper pages
    acl_urls = [p.acl_anthology_url for p in papers_to_check if _has_text(p.acl_anthology_url)]
    sources_combo = []
    if _has_text(sol.faculty_profile_url):
        sources_combo.append(sol.faculty_profile_url)
    sources_combo.extend(acl_urls)

    # Affiliated_At_Publication
    aff_pub_leaf = evaluator.add_leaf(
        id="Affiliated_At_Publication",
        desc="The researcher was affiliated with a US university at the time of ACL 2025 publication",
        parent=node,
        critical=True,
    )
    claim_aff = f"At the time of ACL 2025 publication, {_safe_str(sol.name)} was affiliated with {_safe_str(sol.university)} in the United States."
    await evaluator.verify(
        claim=claim_aff,
        node=aff_pub_leaf,
        sources=sources_combo,
        additional_instruction="Use evidence from the faculty profile and/or the paper author info to confirm US university affiliation around the 2025 period.",
    )

    # University_Verification
    uni_ver_leaf = evaluator.add_leaf(
        id="University_Verification",
        desc="The university affiliation can be verified through the paper's author list or institutional website",
        parent=node,
        critical=True,
    )
    claim_uni = f"{_safe_str(sol.name)} is affiliated with {_safe_str(sol.university)}."
    await evaluator.verify(
        claim=claim_uni,
        node=uni_ver_leaf,
        sources=sources_combo,
        additional_instruction="Confirm that the affiliation is explicitly stated on the faculty page or consistent with the paper author affiliation text.",
    )

    # US_Institution_Confirmed
    us_inst_leaf = evaluator.add_leaf(
        id="US_Institution_Confirmed",
        desc="The institution is confirmed to be located in the United States",
        parent=node,
        critical=True,
    )
    claim_us = f"The institution {_safe_str(sol.university)} is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=us_inst_leaf,
        sources=sol.faculty_profile_url,
        additional_instruction="Confirm the institution's location (address, .edu domain, or page content indicating US).",
    )


async def build_hindex_requirement(
    evaluator: Evaluator,
    parent,
    sol: ResearcherSolution,
) -> None:
    node = evaluator.add_parallel(
        id="H_Index_Requirement",
        desc="Verify the researcher meets the h-index threshold",
        parent=parent,
        critical=True,
    )

    # H_Index_Minimum
    hmin_leaf = evaluator.add_leaf(
        id="H_Index_Minimum",
        desc="The researcher has an h-index of at least 10 according to Google Scholar",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Google Scholar profile shows an h-index of at least 10.",
        node=hmin_leaf,
        sources=sol.google_scholar_url,
        additional_instruction="Check the 'h-index' shown on the Google Scholar profile; rounding is acceptable (e.g., 9.9 → 10 is NOT allowed, but allow reasonable parsing differences).",
    )

    # Google_Scholar_Profile_Verifiable
    gspv_leaf = evaluator.add_leaf(
        id="Google_Scholar_Profile_Verifiable",
        desc="The researcher has a verifiable Google Scholar profile with current h-index information",
        parent=node,
        critical=True,
    )
    claim_prof = f"This page is a Google Scholar profile for {_safe_str(sol.name)} and displays h-index information (e.g., '{_safe_str(sol.scholar_h_index)}')."
    await evaluator.verify(
        claim=claim_prof,
        node=gspv_leaf,
        sources=sol.google_scholar_url,
        additional_instruction="Confirm the page is a Google Scholar profile and that h-index is visible; allow minor name variations.",
    )


async def build_career_stage(
    evaluator: Evaluator,
    parent,
    sol: ResearcherSolution,
) -> None:
    node = evaluator.add_parallel(
        id="Career_Stage_Verification",
        desc="Verify the researcher's academic career stage",
        parent=parent,
        critical=True,
    )

    # Position_Title
    pos_leaf = evaluator.add_leaf(
        id="Position_Title",
        desc="The researcher holds a position as Assistant Professor or Associate Professor (not Full Professor or student)",
        parent=node,
        critical=True,
    )
    claim_pos = f"On the faculty profile page, {_safe_str(sol.name)} holds an academic position as Assistant Professor or Associate Professor (not Full Professor, student, or postdoc)."
    await evaluator.verify(
        claim=claim_pos,
        node=pos_leaf,
        sources=sol.faculty_profile_url,
        additional_instruction="Look for position labels like 'Assistant Professor' or 'Associate Professor'. If the page indicates 'Full Professor', 'Lecturer', 'PhD student', or 'Postdoc', this should fail.",
    )

    # Position_Source_Verification
    psv_leaf = evaluator.add_leaf(
        id="Position_Source_Verification",
        desc="The position title is verifiable from the university's faculty directory, department website, or professional profile",
        parent=node,
        critical=True,
    )
    claim_psv = "The stated position title is explicitly shown on the provided university faculty profile (or equivalent institutional page)."
    await evaluator.verify(
        claim=claim_psv,
        node=psv_leaf,
        sources=sol.faculty_profile_url,
        additional_instruction="Confirm the page explicitly shows the position title; generic bios without title do not suffice.",
    )


async def build_authorship_role(
    evaluator: Evaluator,
    parent,
    sol: ResearcherSolution,
    papers_to_check: List[ACLPaper],
) -> None:
    node = evaluator.add_parallel(
        id="Authorship_Role",
        desc="Verify significant authorship contribution",
        parent=parent,
        critical=True,
    )

    acl_urls = [p.acl_anthology_url for p in papers_to_check if _has_text(p.acl_anthology_url)]

    # First_Or_Corresponding_Author (at least one paper)
    foca_leaf = evaluator.add_leaf(
        id="First_Or_Corresponding_Author",
        desc="The researcher is listed as first author OR corresponding author on at least one of their ACL 2025 papers",
        parent=node,
        critical=True,
    )
    claim_foca = f"On at least one provided ACL 2025 paper page, {_safe_str(sol.name)} is shown as the first author or marked as the corresponding author."
    await evaluator.verify(
        claim=claim_foca,
        node=foca_leaf,
        sources=acl_urls if acl_urls else None,
        additional_instruction="Check the author order (first author) or corresponding author markings (star, label, or explicit text) on the paper page or PDF preview.",
    )

    # Authorship_Verifiable
    av_leaf = evaluator.add_leaf(
        id="Authorship_Verifiable",
        desc="The authorship position (first or corresponding) can be verified from the paper's author list or metadata",
        parent=node,
        critical=True,
    )
    claim_av = "The author list or metadata on the paper page allows verification of first/corresponding authorship."
    await evaluator.verify(
        claim=claim_av,
        node=av_leaf,
        sources=acl_urls if acl_urls else None,
        additional_instruction="Confirm that author order is visible and/or corresponding author is indicated; allow minor formatting differences.",
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
    Evaluate an answer for the ACL 2025 researcher identification task.
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

    # Extract structured data from the answer
    sol: ResearcherSolution = await evaluator.extract(
        prompt=prompt_extract_researcher_and_papers(),
        template_class=ResearcherSolution,
        extraction_name="researcher_solution",
    )

    # Select a manageable subset of papers for intensive verification
    papers_to_check = _paper_subset(sol.acl_papers)

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "papers_total_in_answer": len(sol.acl_papers),
            "papers_considered_for_verification": len(papers_to_check),
            "max_papers_to_verify": MAX_PAPERS_TO_VERIFY,
            "min_required_papers": 2,
        },
        info_type="papers_statistics",
        info_name="papers_statistics",
    )

    # Build a critical task root under the framework root
    task_root = evaluator.add_parallel(
        id="Task_Root",
        desc="Identify a researcher who meets all specified criteria for ACL 2025 publication and academic standing",
        parent=root,
        critical=True,
    )

    # Build subtrees according to rubric
    await build_solution_completeness(evaluator, task_root, sol, papers_to_check)
    await build_acl_participation(evaluator, task_root, sol, papers_to_check)
    await build_us_affiliation(evaluator, task_root, sol, papers_to_check)
    await build_hindex_requirement(evaluator, task_root, sol)
    await build_career_stage(evaluator, task_root, sol)
    await build_authorship_role(evaluator, task_root, sol, papers_to_check)

    # Return the structured evaluation summary
    return evaluator.get_summary()