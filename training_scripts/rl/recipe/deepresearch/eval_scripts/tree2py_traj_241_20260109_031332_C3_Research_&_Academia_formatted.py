import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "neurips24_best_paper_affiliation_faculty_publication"
TASK_DESCRIPTION = (
    "Identify the first author of the NeurIPS 2024 Best Paper Award-winning paper. "
    "Determine their primary university affiliation in China. Then, find one faculty member from the same university's computer science or related department who specializes in computer vision, machine learning, or artificial intelligence. "
    "Finally, identify one peer-reviewed publication by this faculty member from 2023-2025 that appeared in a top-tier conference or journal (such as CVPR, ICCV, NeurIPS, ICML, ICLR, AAAI, ECCV, ACM MM, SIGGRAPH, or equivalent), and provide the publication's title and venue."
)

TOP_TIER_VENUES = {
    "CVPR", "ICCV", "ECCV", "NeurIPS", "ICML", "ICLR", "AAAI", "ACM MM", "SIGGRAPH",
    "TPAMI", "IJCV", "JMLR"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NeurIPSBestPaperTaskExtraction(BaseModel):
    # Step 1: Best Paper
    best_paper_title: Optional[str] = None
    best_paper_urls: List[str] = Field(default_factory=list)

    # Step 2: First author
    first_author_name: Optional[str] = None
    first_author_sources: List[str] = Field(default_factory=list)

    # Step 3: Primary affiliation in China
    primary_university_name: Optional[str] = None
    primary_university_country: Optional[str] = None
    university_sources: List[str] = Field(default_factory=list)

    # Step 4: Faculty member from same university
    faculty_name: Optional[str] = None
    faculty_university_name: Optional[str] = None
    faculty_department: Optional[str] = None
    faculty_specializations: List[str] = Field(default_factory=list)
    faculty_profile_urls: List[str] = Field(default_factory=list)

    # Step 5: Publication by that faculty (2023–2025)
    publication_title: Optional[str] = None
    publication_venue: Optional[str] = None
    publication_year: Optional[str] = None
    publication_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information from the answer for the following items. Extract ONLY what is explicitly present in the answer; do not invent anything. If an item is missing, return null (for a single field) or an empty list (for URLs). Use full URLs where possible.

    1) best_paper_title: The exact title of the NeurIPS 2024 Best Paper Award-winning paper mentioned in the answer.
    2) best_paper_urls: All URLs provided in the answer that directly support the claim that the identified paper won the NeurIPS 2024 Best Paper Award (e.g., NeurIPS official awards page, the paper’s NeurIPS page, press releases).

    3) first_author_name: The name of the first author of that Best Paper as stated in the answer.
    4) first_author_sources: URLs in the answer that show the author list/order for the paper (e.g., paper page, OpenReview, arXiv, NeurIPS proceedings).

    5) primary_university_name: The primary university affiliation in China of the first author (as provided in the answer).
    6) primary_university_country: The country of that university (as indicated or implied in the answer; if not present, return null).
    7) university_sources: URLs in the answer that support the affiliation and/or the university’s location.

    8) faculty_name: The name of one faculty member from the same university (as the first author’s primary affiliation) in CS or a closely related department.
    9) faculty_university_name: The faculty’s university (as per the answer; should match the primary university above).
    10) faculty_department: The department or school the faculty belongs to (e.g., Computer Science, Artificial Intelligence, Data Science, ECE).
    11) faculty_specializations: A list of specialization keywords listed in the answer (e.g., computer vision, machine learning, artificial intelligence, deep learning).
    12) faculty_profile_urls: URLs in the answer that support the faculty’s affiliation, department, and specialization.

    13) publication_title: The title of one peer-reviewed publication (2023–2025) by the faculty member.
    14) publication_venue: The venue name (conference or journal).
    15) publication_year: The publication’s year (e.g., 2024). Extract as a string as it appears.
    16) publication_urls: URLs in the answer that show the publication’s title, authorship, venue, and year (e.g., IEEE/ACM/CVF pages, official conference proceedings, publisher pages, DBLP/ACM DL).

    Return a single JSON object with these exact field names. If any field is missing in the answer, set it to null or [] accordingly. Do not include any extra fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists and keep unique non-empty strings."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            u = (url or "").strip()
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def extract_year_int(year_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from a string, if possible."""
    if not year_str:
        return None
    m = re.search(r"(20\d{2})", year_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def safe_text(v: Optional[str]) -> str:
    return (v or "").strip()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_step1_best_paper(
    evaluator: Evaluator,
    parent_node,
    data: NeurIPSBestPaperTaskExtraction,
) -> None:
    step_node = evaluator.add_parallel(
        id="step1_best_paper_winner",
        desc="Identify the NeurIPS 2024 Best Paper Award-winning paper.",
        parent=parent_node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="best_paper_correct",
        desc="Correctly identify the NeurIPS 2024 Best Paper Award winner (must be Best Paper, not runner-up or honorable mention).",
        parent=step_node,
        critical=True,
    )

    claim = f"The paper titled '{safe_text(data.best_paper_title)}' won the NeurIPS 2024 Best Paper Award."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.best_paper_urls,
        additional_instruction=(
            "Focus strictly on the NeurIPS 2024 Best Paper Award (the top award). "
            "Do not accept honorable mentions, runners-up, or other awards. "
            "Prefer official NeurIPS announcements or authoritative sources."
        ),
    )


async def verify_step2_first_author(
    evaluator: Evaluator,
    parent_node,
    data: NeurIPSBestPaperTaskExtraction,
) -> None:
    step_node = evaluator.add_parallel(
        id="step2_first_author",
        desc="Identify the first author of the identified NeurIPS 2024 Best Paper.",
        parent=parent_node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="first_author_correct",
        desc="Provide the correct first author name for the identified NeurIPS 2024 Best Paper.",
        parent=step_node,
        critical=True,
    )

    combined_sources = merge_sources(data.first_author_sources, data.best_paper_urls)
    claim = (
        f"The first author of the paper '{safe_text(data.best_paper_title)}' is '{safe_text(data.first_author_name)}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=combined_sources,
        additional_instruction=(
            "Verify the author order shown on authoritative sources (e.g., NeurIPS proceedings page, OpenReview, arXiv). "
            "Treat the first listed author as the first author. "
            "Allow minor name variations (initials, middle names, casing)."
        ),
    )


async def verify_step3_primary_affiliation_china(
    evaluator: Evaluator,
    parent_node,
    data: NeurIPSBestPaperTaskExtraction,
) -> None:
    step_node = evaluator.add_parallel(
        id="step3_primary_affiliation_china",
        desc="Determine the first author's primary university affiliation in China.",
        parent=parent_node,
        critical=True,
    )

    # 3.1 Affiliation correctness
    leaf_affil = evaluator.add_leaf(
        id="primary_university_affiliation_correct",
        desc="Provide the correct primary university affiliation of the first author.",
        parent=step_node,
        critical=True,
    )

    affil_claim = (
        f"The primary university affiliation of '{safe_text(data.first_author_name)}' is '{safe_text(data.primary_university_name)}'."
    )
    affil_sources = merge_sources(data.university_sources, data.first_author_sources)
    await evaluator.verify(
        claim=affil_claim,
        node=leaf_affil,
        sources=affil_sources,
        additional_instruction=(
            "Use official profiles, paper affiliation lines, or university pages cited in the answer. "
            "If multiple affiliations are listed, focus on the one stated as primary in the answer. "
            "Ensure the affiliation is a university."
        ),
    )

    # 3.2 University is in China
    leaf_country = evaluator.add_leaf(
        id="university_in_china",
        desc="The identified university must be located in China.",
        parent=step_node,
        critical=True,
    )

    country_claim = (
        f"The university '{safe_text(data.primary_university_name)}' is located in China."
    )
    await evaluator.verify(
        claim=country_claim,
        node=leaf_country,
        sources=data.university_sources,
        additional_instruction=(
            "Check the university’s location on authoritative pages (official site or Wikipedia). "
            "Confirm it is in China. Minor naming variants are acceptable."
        ),
    )


async def verify_step4_faculty_member(
    evaluator: Evaluator,
    parent_node,
    data: NeurIPSBestPaperTaskExtraction,
) -> None:
    step_node = evaluator.add_parallel(
        id="step4_faculty_member",
        desc="Identify one qualifying faculty member from the same university in a CS/related department with relevant specialization.",
        parent=parent_node,
        critical=True,
    )

    # Existence of faculty name (custom leaf)
    evaluator.add_custom_node(
        result=bool(safe_text(data.faculty_name)),
        id="faculty_name_provided",
        desc="Provide the name of one faculty member.",
        parent=step_node,
        critical=True,
    )

    # Same university
    leaf_same_uni = evaluator.add_leaf(
        id="faculty_same_university",
        desc="The faculty member must be affiliated with the same university identified as the first author's primary affiliation.",
        parent=step_node,
        critical=True,
    )
    same_uni_claim = (
        f"The faculty member '{safe_text(data.faculty_name)}' is affiliated with '{safe_text(data.primary_university_name)}'."
    )
    await evaluator.verify(
        claim=same_uni_claim,
        node=leaf_same_uni,
        sources=data.faculty_profile_urls,
        additional_instruction=(
            "Use the faculty’s official profile or university directory cited in the answer to confirm they are affiliated with the same university."
        ),
    )

    # Relevant department
    leaf_dept = evaluator.add_leaf(
        id="faculty_department_relevant",
        desc="The faculty member must be in computer science or a closely related department (e.g., AI, data science).",
        parent=step_node,
        critical=True,
    )
    dept_claim = (
        f"The faculty member '{safe_text(data.faculty_name)}' belongs to a computer science or closely related department (e.g., AI, Data Science, ECE with AI/CS focus). Reported department: '{safe_text(data.faculty_department)}'."
    )
    await evaluator.verify(
        claim=dept_claim,
        node=leaf_dept,
        sources=data.faculty_profile_urls,
        additional_instruction=(
            "Check the department/unit on the faculty profile. Accept CS, AI, Data Science, EECS/ECE with AI/CS emphasis, Automation/Information science if clearly related."
        ),
    )

    # Relevant specialization
    leaf_spec = evaluator.add_leaf(
        id="faculty_specialization_relevant",
        desc="The faculty member must specialize in computer vision, machine learning, or artificial intelligence.",
        parent=step_node,
        critical=True,
    )
    spec_text = ", ".join([s for s in data.faculty_specializations if s]) or safe_text(data.faculty_department)
    spec_claim = (
        f"The faculty member '{safe_text(data.faculty_name)}' specializes in computer vision or machine learning or artificial intelligence. Reported specializations: {spec_text}."
    )
    spec_sources = merge_sources(data.faculty_profile_urls, data.publication_urls)
    await evaluator.verify(
        claim=spec_claim,
        node=leaf_spec,
        sources=spec_sources,
        additional_instruction=(
            "Verify research interests and publications shown on the profile or authoritative pages. "
            "Accept synonyms like deep learning, pattern recognition, machine vision, AI."
        ),
    )


async def verify_step5_faculty_publication(
    evaluator: Evaluator,
    parent_node,
    data: NeurIPSBestPaperTaskExtraction,
) -> None:
    step_node = evaluator.add_parallel(
        id="step5_faculty_publication",
        desc="Identify one qualifying 2023–2025 peer-reviewed top-tier publication by the faculty member and provide title and venue.",
        parent=parent_node,
        critical=True,
    )

    # Existence checks (custom leaves)
    evaluator.add_custom_node(
        result=bool(safe_text(data.publication_title)),
        id="publication_title_provided",
        desc="Provide the publication title.",
        parent=step_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(safe_text(data.publication_venue)),
        id="publication_venue_provided",
        desc="Provide the publication venue (conference or journal name).",
        parent=step_node,
        critical=True,
    )

    # Faculty is an author
    leaf_author = evaluator.add_leaf(
        id="faculty_is_author",
        desc="The identified faculty member must be an author of the publication.",
        parent=step_node,
        critical=True,
    )
    author_claim = (
        f"'{safe_text(data.faculty_name)}' is listed as an author of the publication titled '{safe_text(data.publication_title)}'."
    )
    await evaluator.verify(
        claim=author_claim,
        node=leaf_author,
        sources=data.publication_urls,
        additional_instruction=(
            "Check the authors list on authoritative publication pages (publisher site, official proceedings). "
            "Co-authorship is acceptable."
        ),
    )

    # Publication year in range 2023–2025
    leaf_year = evaluator.add_leaf(
        id="publication_year_in_range",
        desc="The publication year must be 2023, 2024, or 2025.",
        parent=step_node,
        critical=True,
    )
    year_int = extract_year_int(data.publication_year)
    year_text = str(year_int) if year_int is not None else safe_text(data.publication_year)
    year_claim = (
        f"The publication '{safe_text(data.publication_title)}' was published in {year_text}, which is within 2023–2025 inclusive."
    )
    await evaluator.verify(
        claim=year_claim,
        node=leaf_year,
        sources=data.publication_urls,
        additional_instruction=(
            "Use the publication page to confirm the year and judge whether it falls in 2023, 2024, or 2025."
        ),
    )

    # Peer-reviewed
    leaf_peer = evaluator.add_leaf(
        id="publication_peer_reviewed",
        desc="The publication must be peer-reviewed.",
        parent=step_node,
        critical=True,
    )
    peer_claim = (
        f"The publication '{safe_text(data.publication_title)}' at venue '{safe_text(data.publication_venue)}' is peer-reviewed."
    )
    await evaluator.verify(
        claim=peer_claim,
        node=leaf_peer,
        sources=data.publication_urls,
        additional_instruction=(
            "Top-tier conferences/journals listed are peer-reviewed; verify that the venue is a standard peer-reviewed event or journal."
        ),
    )

    # Venue top-tier
    leaf_toptier = evaluator.add_leaf(
        id="venue_top_tier",
        desc="The venue must be top-tier (e.g., CVPR, ICCV, NeurIPS, ICML, ICLR, AAAI, ECCV, ACM MM, SIGGRAPH, or an equivalent top-tier journal such as TPAMI, IJCV, JMLR).",
        parent=step_node,
        critical=True,
    )
    venue_claim = (
        f"The venue '{safe_text(data.publication_venue)}' for the publication '{safe_text(data.publication_title)}' is a top-tier AI/CV/ML publication venue."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=leaf_toptier,
        sources=data.publication_urls,
        additional_instruction=(
            "Judge whether the venue matches one of the commonly recognized top-tier venues: "
            "CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI, ACM MM, SIGGRAPH, TPAMI, IJCV, JMLR, or an equivalent top-tier journal. "
            "Allow standard naming variations (e.g., IEEE/CVF CVPR, Conference on Neural Information Processing Systems for NeurIPS)."
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=NeurIPSBestPaperTaskExtraction,
        extraction_name="structured_extraction",
    )

    # Optional custom info for reference
    evaluator.add_custom_info(
        {"top_tier_venues": sorted(list(TOP_TIER_VENUES))},
        info_type="reference",
        info_name="top_tier_venues_reference",
    )

    # Build verification tree per rubric
    await verify_step1_best_paper(evaluator, root, extracted)
    await verify_step2_first_author(evaluator, root, extracted)
    await verify_step3_primary_affiliation_china(evaluator, root, extracted)
    await verify_step4_faculty_member(evaluator, root, extracted)
    await verify_step5_faculty_publication(evaluator, root, extracted)

    return evaluator.get_summary()