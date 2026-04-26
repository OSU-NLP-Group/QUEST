import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chi2024_best_paper_canadian_assoc_prof"
TASK_DESCRIPTION = (
    "A researcher who is an associate professor at a Canadian university was among the authors who won a Best Paper Award "
    "at CHI 2024 (the ACM CHI Conference on Human Factors in Computing Systems held in May 2024 in Honolulu, Hawaii). "
    "Identify this researcher and provide: (1) the specific name of the department or school where they work at their university, "
    "and (2) the full name of at least one research lab or group they are affiliated with at that institution."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Information about the CHI 2024 Best Paper and sources."""
    title: Optional[str] = None
    award_sources: List[str] = Field(default_factory=list, description="URLs that explicitly support the Best Paper Award at CHI 2024")
    authorship_sources: List[str] = Field(default_factory=list, description="URLs that list the paper's authors (e.g., ACM DL, project page)")


class ResearcherInfo(BaseModel):
    """Information about the researcher and their institution."""
    name: Optional[str] = None
    university_name: Optional[str] = None
    university_urls: List[str] = Field(default_factory=list, description="URLs supporting the university affiliation")
    rank_title: Optional[str] = None
    rank_sources: List[str] = Field(default_factory=list, description="URLs supporting the academic rank (Associate Professor)")


class DepartmentInfo(BaseModel):
    """Department or school information for the researcher."""
    department_name: Optional[str] = None
    department_sources: List[str] = Field(default_factory=list, description="URLs supporting department/school assignment")


class LabInfo(BaseModel):
    """Research lab or group affiliations for the researcher."""
    lab_names: List[str] = Field(default_factory=list)
    lab_sources: List[str] = Field(default_factory=list, description="URLs supporting lab/group affiliation(s)")


class AnswerExtraction(BaseModel):
    """Top-level extraction structure to capture all necessary fields from the answer."""
    paper: Optional[PaperInfo] = None
    researcher: Optional[ResearcherInfo] = None
    department: Optional[DepartmentInfo] = None
    labs: Optional[LabInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return (
        "Extract, from the provided answer text, the identity of exactly one researcher who matches all the constraints. "
        "You must extract the following structured fields:\n"
        "1) paper:\n"
        "   - title: The title of the CHI 2024 Best Paper associated with the researcher (string)\n"
        "   - award_sources: A list of the exact URLs cited in the answer that explicitly confirm the paper won a Best Paper Award at CHI 2024\n"
        "   - authorship_sources: A list of the exact URLs cited in the answer that list the paper's authors (e.g., ACM Digital Library, project page)\n"
        "2) researcher:\n"
        "   - name: The full name of the identified researcher (string)\n"
        "   - university_name: The name of the Canadian university where they work (string)\n"
        "   - university_urls: A list of exact URLs cited in the answer that support the university affiliation\n"
        "   - rank_title: The academic rank as stated in the answer (should include 'Associate Professor' if correct)\n"
        "   - rank_sources: A list of exact URLs cited in the answer that support the academic rank\n"
        "3) department:\n"
        "   - department_name: The specific department or school name at their university (string)\n"
        "   - department_sources: A list of exact URLs cited in the answer that support the department/school info\n"
        "4) labs:\n"
        "   - lab_names: A list of full names of research labs or groups the researcher is affiliated with at that institution\n"
        "   - lab_sources: A list of exact URLs cited in the answer that support the lab/group affiliation(s)\n\n"
        "IMPORTANT:\n"
        "- Extract only information explicitly present in the answer text. Do not invent content.\n"
        "- For all URL lists, include only actual URLs explicitly provided in the answer (plain, markdown link targets, etc.). If none are provided for a field, return an empty list.\n"
        "- If any required string field is missing, set it to null.\n"
        "- Return a single JSON object matching the specified schema."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists into a unique, ordered list."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _safe(s: Optional[str]) -> str:
    return s or ""


def _first_lab_name(labs: Optional[LabInfo]) -> str:
    if labs and labs.lab_names:
        return labs.lab_names[0]
    return ""


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: AnswerExtraction) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    Root node created by evaluator.initialize is non-critical; we add a critical sequential node under it.
    """
    # Create top-level critical sequential node representing the entire task
    task_node = evaluator.add_sequential(
        id="ResearcherIdentificationTask",
        desc="Identify a specific researcher based on award and affiliation criteria, and provide their institutional affiliations.",
        parent=evaluator.root,
        critical=True,
    )

    # ---- 1) AwardVerification (Leaf, Critical) -------------------------- #
    award_node = evaluator.add_leaf(
        id="AwardVerification",
        desc="Confirm the relevant research paper received a Best Paper Award at CHI 2024.",
        parent=task_node,
        critical=True,
    )
    paper_title = _safe(extraction.paper.title if extraction.paper else None)
    award_sources = _merge_sources(extraction.paper.award_sources if extraction.paper else [])

    award_claim = (
        f"The paper titled '{paper_title}' received a Best Paper Award at CHI 2024."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_node,
        sources=award_sources,
        additional_instruction=(
            "Verify that the cited evidence explicitly indicates a 'Best Paper Award' at CHI 2024 (May 2024, Honolulu). "
            "Do NOT treat 'Honorable Mention' or other awards as equivalent. "
            "The award must be specifically from CHI 2024."
        ),
    )

    # ---- 2) ResearcherCriteria (Parallel, all children Critical) -------- #
    criteria_node = evaluator.add_parallel(
        id="ResearcherCriteria",
        desc="Confirm the identified researcher satisfies the authorship, rank, and Canadian university affiliation constraints.",
        parent=task_node,
        critical=True,
    )

    # 2.a PaperAuthorship (Leaf)
    authorship_node = evaluator.add_leaf(
        id="PaperAuthorship",
        desc="The identified person is listed as an author of the CHI 2024 Best Paper Award paper.",
        parent=criteria_node,
        critical=True,
    )
    researcher_name = _safe(extraction.researcher.name if extraction.researcher else None)
    authorship_sources = _merge_sources(
        extraction.paper.authorship_sources if extraction.paper else [],
        extraction.paper.award_sources if extraction.paper else [],
    )
    author_claim = (
        f"{researcher_name} is listed as an author of the paper '{paper_title}'."
    )
    await evaluator.verify(
        claim=author_claim,
        node=authorship_node,
        sources=authorship_sources,
        additional_instruction=(
            "Check the author list on credible sources (e.g., ACM DL, official project/paper pages). "
            "Allow minor name formatting variations (middle initials, casing)."
        ),
    )

    # 2.b AcademicRank (Leaf)
    rank_node = evaluator.add_leaf(
        id="AcademicRank",
        desc="The researcher holds the rank of Associate Professor at the time of the award.",
        parent=criteria_node,
        critical=True,
    )
    university_name = _safe(extraction.researcher.university_name if extraction.researcher else None)
    rank_title = _safe(extraction.researcher.rank_title if extraction.researcher else None)
    rank_sources = _merge_sources(extraction.researcher.rank_sources if extraction.researcher else [])
    rank_claim = (
        f"As of May 2024 (around the CHI 2024 timeframe), {researcher_name} is an Associate Professor at {university_name}."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_node,
        sources=rank_sources,
        additional_instruction=(
            "Accept credible institutional pages (e.g., university profile, department page) indicating Associate Professor. "
            "If the page shows Associate Professor around 2023–2025, consider it consistent with May 2024. "
            "Focus on rank being 'Associate Professor'."
        ),
    )

    # 2.c CanadianInstitution (Leaf)
    canadian_node = evaluator.add_leaf(
        id="CanadianInstitution",
        desc="The researcher's primary institutional affiliation is at a university located in Canada.",
        parent=criteria_node,
        critical=True,
    )
    university_urls = _merge_sources(extraction.researcher.university_urls if extraction.researcher else [])
    canada_claim = f"The university '{university_name}' is located in Canada."
    await evaluator.verify(
        claim=canada_claim,
        node=canadian_node,
        sources=university_urls,
        additional_instruction=(
            "Verify that the institution is a Canadian university (e.g., .ca domain, mention of Canada on official or reputable pages, "
            "or Wikipedia indicating location in Canada)."
        ),
    )

    # ---- 3) AffiliationDetails (Parallel, all children Critical) -------- #
    affiliation_node = evaluator.add_parallel(
        id="AffiliationDetails",
        desc="Provide the required institutional details for the identified researcher.",
        parent=task_node,
        critical=True,
    )

    # 3.a DepartmentName (Leaf)
    dept_node = evaluator.add_leaf(
        id="DepartmentName",
        desc="Provide the specific name of the department or school where the researcher works at their university.",
        parent=affiliation_node,
        critical=True,
    )
    department_name = _safe(extraction.department.department_name if extraction.department else None)
    department_sources = _merge_sources(extraction.department.department_sources if extraction.department else [])
    dept_claim = (
        f"{researcher_name} works in the department/school '{department_name}' at {university_name}."
    )
    await evaluator.verify(
        claim=dept_claim,
        node=dept_node,
        sources=department_sources,
        additional_instruction=(
            "Confirm that the cited page(s) explicitly state the researcher's department/school at the specified university. "
            "Allow reasonable naming variants (e.g., 'Department of Computer Science', 'School of Computing')."
        ),
    )

    # 3.b ResearchLabAffiliation (Leaf)
    lab_node = evaluator.add_leaf(
        id="ResearchLabAffiliation",
        desc="Provide the full name of at least one research lab or group the researcher is affiliated with at that institution.",
        parent=affiliation_node,
        critical=True,
    )
    lab_name = _safe(_first_lab_name(extraction.labs))
    lab_sources = _merge_sources(extraction.labs.lab_sources if extraction.labs else [])
    lab_claim = (
        f"{researcher_name} is affiliated with the research lab/group '{lab_name}' at {university_name}."
    )
    await evaluator.verify(
        claim=lab_claim,
        node=lab_node,
        sources=lab_sources,
        additional_instruction=(
            "Verify that the cited page(s) indicate the researcher's affiliation with the named lab/group. "
            "It should be part of or associated with the same institution."
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
) -> Dict:
    """
    Evaluate an answer for the CHI 2024 Best Paper Canadian Associate Professor identification task.
    """
    # Initialize evaluator; use SEQUENTIAL to reflect top-level flow
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract core structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=AnswerExtraction,
        extraction_name="core_extraction",
    )

    # Build verification tree and perform checks
    await build_verification_tree(evaluator, extraction)

    # Return structured summary
    return evaluator.get_summary()