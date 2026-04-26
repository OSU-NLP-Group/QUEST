import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "astro_phd_nasa_2026"
TASK_DESCRIPTION = (
    "Identify astronomy or astrophysics PhD programs at universities in the United States that meet ALL of the following criteria: "
    "(1) The university has an active NASA partnership agreement that was signed or renewed in 2026, "
    "(2) The institution is a member of a NASA Space Grant Consortium, "
    "(3) At least one faculty member from the astronomy/astrophysics department is presenting at an astrophysics conference in 2026, "
    "(4) The PhD program requires a minimum GPA of 3.0 and completion of at least 6 core graduate courses, "
    "(5) The PhD program requires at least one peer-reviewed publication before dissertation completion, "
    "(6) The institution offers or hosts postdoctoral fellowships in astronomy or astrophysics, "
    "(7) The institution has an affiliation with or operates an astronomical observatory, "
    "(8) The PhD program requires a comprehensive or qualifying examination by the end of the 2nd or 3rd year, "
    "(9) The PhD program has a typical completion time of 5-6 years, "
    "(10) The program offers research opportunities in observational astronomy, theoretical astrophysics, or computational astrophysics, "
    "(11) The program provides graduate student financial support through teaching assistantships, research assistantships, or fellowships, "
    "(12) The program acknowledges US citizenship requirements for Space Grant funded opportunities. "
    "For each qualifying institution, provide the university name, documentation confirming each of the 12 criteria above, and reference URLs supporting each criterion."
)

CUTOFF_YEAR = 2026
MAX_INSTITUTIONS_TO_EVAL = 1  # Evaluate only the first qualifying institution mentioned in the answer


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityEntry(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    identity_sources: List[str] = Field(default_factory=list)

    criterion1_nasa_partnership_2026_urls: List[str] = Field(default_factory=list)
    criterion2_space_grant_membership_urls: List[str] = Field(default_factory=list)
    criterion3_faculty_conference_2026_urls: List[str] = Field(default_factory=list)
    criterion4_min_gpa_3_0_urls: List[str] = Field(default_factory=list)
    criterion5_core_courses_6_plus_urls: List[str] = Field(default_factory=list)
    criterion6_peer_reviewed_publication_urls: List[str] = Field(default_factory=list)
    criterion7_postdoc_fellowships_urls: List[str] = Field(default_factory=list)
    criterion8_observatory_affiliation_urls: List[str] = Field(default_factory=list)
    criterion9_qualifying_exam_timing_urls: List[str] = Field(default_factory=list)
    criterion10_completion_time_5_6_years_urls: List[str] = Field(default_factory=list)
    criterion11_research_opportunities_urls: List[str] = Field(default_factory=list)
    criterion12_financial_support_urls: List[str] = Field(default_factory=list)
    criterion13_space_grant_citizenship_urls: List[str] = Field(default_factory=list)


class InstitutionsExtraction(BaseModel):
    institutions: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
    Extract up to the first 3 institutions (universities) mentioned in the answer that the respondent claims satisfy the task. For each institution, return:
    - university_name: The exact university name as stated in the answer.
    - program_name: The astronomy/astrophysics PhD program name (if stated).
    - identity_sources: A list of URL(s) explicitly cited in the answer intended to demonstrate BOTH that the institution is US-based and that it offers an astronomy/astrophysics PhD program (e.g., department/program page, university page). Include only URLs actually present in the answer.
    - criterion1_nasa_partnership_2026_urls: URL(s) cited to show the university has an active NASA partnership agreement signed or renewed in 2026 (e.g., NASA/University press releases, Space Act Agreement pages).
    - criterion2_space_grant_membership_urls: URL(s) cited to show membership in a NASA Space Grant Consortium (e.g., NASA Space Grant listing, state consortium page).
    - criterion3_faculty_conference_2026_urls: URL(s) cited to show at least one astronomy/astrophysics faculty member is presenting at an astrophysics conference in 2026 (e.g., AAS 2026 program, conference agenda, departmental news).
    - criterion4_min_gpa_3_0_urls: URL(s) cited to show the PhD program requires a minimum GPA of 3.0.
    - criterion5_core_courses_6_plus_urls: URL(s) cited to show the PhD program requires completion of at least 6 core graduate courses.
    - criterion6_peer_reviewed_publication_urls: URL(s) cited to show the PhD program requires at least one peer-reviewed publication before dissertation completion.
    - criterion7_postdoc_fellowships_urls: URL(s) cited to show the institution offers or hosts postdoctoral fellowships in astronomy/astrophysics (departmental postdoc pages; hosting national fellowships acceptable if clearly hosted).
    - criterion8_observatory_affiliation_urls: URL(s) cited to show the institution has an affiliation with or operates an astronomical observatory.
    - criterion9_qualifying_exam_timing_urls: URL(s) cited to show a comprehensive/qualifying exam is required by the end of the 2nd or 3rd year.
    - criterion10_completion_time_5_6_years_urls: URL(s) cited to show the typical PhD completion time is 5–6 years.
    - criterion11_research_opportunities_urls: URL(s) cited to show the program offers research opportunities in at least one of: observational astronomy, theoretical astrophysics, computational astrophysics.
    - criterion12_financial_support_urls: URL(s) cited to show graduate student financial support is provided via TA/RA/fellowships.
    - criterion13_space_grant_citizenship_urls: URL(s) cited to show the program acknowledges US citizenship requirements for Space Grant funded opportunities.

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer text. Do not invent or infer any URLs.
    - If a specific field lacks any URLs in the answer, return an empty list for that field (not null).
    - identity_sources should be URLs that best support both US location and existence of an astronomy/astrophysics PhD program at the named university.
    - Return a JSON object with an 'institutions' array of entries as specified above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification for one institution                                            #
# --------------------------------------------------------------------------- #
async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    inst: UniversityEntry,
    idx: int,
) -> None:
    """
    Build and verify the subtree for a single institution.
    All checks are critical to satisfy the "ALL constraints" requirement.
    """
    # Institution container (critical)
    inst_node = evaluator.add_parallel(
        id=f"inst_{idx}",
        desc="A single institution entry evaluated for identity plus satisfaction of each constraint (with supporting URL evidence).",
        parent=parent_node,
        critical=True,
    )

    # Existence / required info gate (critical): must have a name and identity sources
    has_required = (inst.university_name is not None and inst.university_name.strip() != "") and _has_urls(inst.identity_sources)
    evaluator.add_custom_node(
        result=has_required,
        id=f"inst_{idx}_required_info",
        desc=f"Institution #{idx + 1} has a university name and identity sources",
        parent=inst_node,
        critical=True,
    )

    uni = inst.university_name or ""

    # Institution identification leaf
    id_leaf = evaluator.add_leaf(
        id=f"inst_{idx}_institution_identification",
        desc="Provides the university name and indicates it is a US-based institution with an astronomy/astrophysics PhD program.",
        parent=inst_node,
        critical=True,
    )
    id_claim = (
        f"{uni} is a university in the United States and it offers a PhD program in astronomy or astrophysics."
    )
    # If no sources, fail the leaf immediately (source-grounding policy)
    if not _has_urls(inst.identity_sources):
        id_leaf.score = 0.0
        id_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim=id_claim,
            node=id_leaf,
            sources=_safe_urls(inst.identity_sources),
            additional_instruction=(
                "Verify BOTH: (1) the institution is US-based (e.g., address or statement), and "
                "(2) it offers a PhD program in astronomy or astrophysics (e.g., department/program page). "
                "Minor naming variations are acceptable (e.g., 'Astronomy PhD', 'Astrophysics PhD', or 'Astronomy & Astrophysics PhD'). "
                "Prefer official university/department pages."
            ),
        )

    # Prepare criterion leaves and their verification
    # Mapping: (node_suffix, description, claim_template, urls, additional_instruction)
    crit_items: List[Dict[str, Any]] = [
        dict(
            suffix="criterion_1_nasa_partnership_2026",
            desc="Provides evidence with supporting URL(s) that the university has an active NASA partnership agreement signed or renewed in 2026.",
            claim=f"{uni} has an active NASA partnership agreement that was signed or renewed in {CUTOFF_YEAR}.",
            urls=inst.criterion1_nasa_partnership_2026_urls,
            add_ins=(
                "Look for NASA partnership announcements, Space Act Agreements, or official press releases explicitly mentioning the year 2026. "
                "The page should clearly indicate that the agreement was signed or renewed in 2026."
            ),
        ),
        dict(
            suffix="criterion_2_space_grant_membership",
            desc="Provides evidence with supporting URL(s) that the institution is a member of a NASA Space Grant Consortium.",
            claim=f"{uni} is a member of a NASA Space Grant Consortium.",
            urls=inst.criterion2_space_grant_membership_urls,
            add_ins=(
                "Verify membership on official NASA Space Grant pages or state consortium websites; the institution name should appear on the member list."
            ),
        ),
        dict(
            suffix="criterion_3_faculty_conference_2026",
            desc="Provides evidence with supporting URL(s) that at least one astronomy/astrophysics faculty member is presenting at an astrophysics conference in 2026.",
            claim=f"At least one faculty member from {uni}'s astronomy or astrophysics department is presenting at an astrophysics conference in {CUTOFF_YEAR}.",
            urls=inst.criterion3_faculty_conference_2026_urls,
            add_ins=(
                "Accept conference programs, abstracts, agendas, or official news pages. "
                "The presenter should be identifiable as a faculty member (not just a student), and the event year must be 2026."
            ),
        ),
        dict(
            suffix="criterion_4_min_gpa_3_0",
            desc="Provides evidence with supporting URL(s) that the PhD program requires a minimum GPA of 3.0.",
            claim=f"The astronomy/astrophysics PhD program at {uni} requires a minimum GPA of 3.0.",
            urls=inst.criterion4_min_gpa_3_0_urls,
            add_ins=(
                "Check program admissions pages, graduate handbook, or university catalog for minimum GPA requirement of 3.0."
            ),
        ),
        dict(
            suffix="criterion_5_core_courses_6_plus",
            desc="Provides evidence with supporting URL(s) that the PhD program requires completion of at least 6 core graduate courses.",
            claim=f"The astronomy/astrophysics PhD program at {uni} requires completion of at least 6 core graduate courses.",
            urls=inst.criterion5_core_courses_6_plus_urls,
            add_ins=(
                "Look for curriculum or degree requirements indicating a minimum of 6 'core' courses (not just total credits). "
                "The page should explicitly list or quantify core course requirements."
            ),
        ),
        dict(
            suffix="criterion_6_peer_reviewed_publication",
            desc="Provides evidence with supporting URL(s) that the PhD program requires at least one peer-reviewed publication before dissertation completion.",
            claim=f"Before dissertation completion, the astronomy/astrophysics PhD program at {uni} requires at least one peer-reviewed publication.",
            urls=inst.criterion6_peer_reviewed_publication_urls,
            add_ins=(
                "Check handbooks or policies explicitly stating a requirement for at least one peer-reviewed paper prior to dissertation or graduation."
            ),
        ),
        dict(
            suffix="criterion_7_postdoc_fellowships",
            desc="Provides evidence with supporting URL(s) that the institution offers or hosts postdoctoral fellowships in astronomy/astrophysics.",
            claim=f"{uni} offers or hosts postdoctoral fellowships in astronomy or astrophysics.",
            urls=inst.criterion7_postdoc_fellowships_urls,
            add_ins=(
                "Departmental or institutional postdoc pages count. "
                "Hosting nationally competitive astrophysics fellowships (e.g., Hubble, NHFP) is acceptable if clearly hosted at the institution."
            ),
        ),
        dict(
            suffix="criterion_8_observatory_affiliation",
            desc="Provides evidence with supporting URL(s) that the institution has an affiliation with or operates an astronomical observatory.",
            claim=f"{uni} has an affiliation with or operates an astronomical observatory.",
            urls=inst.criterion8_observatory_affiliation_urls,
            add_ins=(
                "Look for official descriptions of an on-campus or partnered observatory; affiliation or operation should be explicit."
            ),
        ),
        dict(
            suffix="criterion_9_qualifying_exam_timing",
            desc="Provides evidence with supporting URL(s) that the PhD program requires a comprehensive/qualifying exam by the end of the 2nd or 3rd year.",
            claim=f"The astronomy/astrophysics PhD program at {uni} requires a comprehensive or qualifying examination by the end of the 2nd or 3rd year.",
            urls=inst.criterion9_qualifying_exam_timing_urls,
            add_ins=(
                "Check graduate handbook or program milestones for timing requirements (end of year 2 or 3)."
            ),
        ),
        dict(
            suffix="criterion_10_completion_time_5_6_years",
            desc="Provides evidence with supporting URL(s) that the PhD program has a typical completion time of 5–6 years.",
            claim=f"The typical time to complete the astronomy/astrophysics PhD at {uni} is 5 to 6 years.",
            urls=inst.criterion10_completion_time_5_6_years_urls,
            add_ins=(
                "Look for statements like 'typical', 'average', or 'expected' time to degree being 5–6 years."
            ),
        ),
        dict(
            suffix="criterion_11_research_opportunities",
            desc="Provides evidence with supporting URL(s) that the program offers research opportunities in at least one of: observational astronomy, theoretical astrophysics, computational astrophysics.",
            claim=f"The program at {uni} offers research opportunities in at least one of: observational astronomy, theoretical astrophysics, or computational astrophysics.",
            urls=inst.criterion11_research_opportunities_urls,
            add_ins=(
                "Faculty/research group pages, program overviews, or research descriptions that clearly indicate these areas are acceptable."
            ),
        ),
        dict(
            suffix="criterion_12_financial_support",
            desc="Provides evidence with supporting URL(s) that the program provides graduate student financial support through at least one of: TA, RA, fellowships.",
            claim=f"{uni}'s astronomy/astrophysics PhD program provides financial support via teaching assistantships, research assistantships, or fellowships.",
            urls=inst.criterion12_financial_support_urls,
            add_ins=(
                "Admissions/funding pages that explicitly mention TA/RA/fellowships for graduate students are acceptable."
            ),
        ),
        dict(
            suffix="criterion_13_space_grant_citizenship",
            desc="Provides evidence with supporting URL(s) that the program acknowledges US citizenship requirements for Space Grant-funded opportunities.",
            claim=f"The program at {uni} acknowledges US citizenship requirements for Space Grant funded opportunities.",
            urls=inst.criterion13_space_grant_citizenship_urls,
            add_ins=(
                "Look for notes on eligibility stating that Space Grant opportunities require US citizenship; an official page should mention this."
            ),
        ),
    ]

    # Create leaves and schedule verifications
    batch_items: List[tuple[str, List[str], Any, Optional[str]]] = []
    leaves_created: List[Any] = []

    for item in crit_items:
        leaf = evaluator.add_leaf(
            id=f"inst_{idx}_{item['suffix']}",
            desc=item["desc"],
            parent=inst_node,
            critical=True,
        )
        leaves_created.append(leaf)

        urls = _safe_urls(item["urls"])
        if not _has_urls(urls):
            # Enforce source-grounding: fail immediately if no supporting URLs were provided
            leaf.score = 0.0
            leaf.status = "failed"
        else:
            batch_items.append((item["claim"], urls, leaf, item["add_ins"]))

    # Run all URL-grounded verifications in parallel
    if batch_items:
        await evaluator.batch_verify(batch_items)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the astronomy/astrophysics PhD program NASA criteria task.
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

    # Extract institution entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
    )

    # Add a critical container under root to mirror rubric's critical root
    institutions_root = evaluator.add_parallel(
        id="institution_entries_root",
        desc="Provide at least one US-based university astronomy/astrophysics PhD program that satisfies all stated constraints; for each provided institution, include evidence with supporting URL(s) for each constraint.",
        parent=root,
        critical=True,
    )

    # Record meta information
    total_found = len(extracted.institutions)
    evaluator.add_custom_info(
        info={"max_institutions_evaluated": MAX_INSTITUTIONS_TO_EVAL, "institutions_found_in_answer": total_found},
        info_type="extraction_meta",
        info_name="extraction_meta",
    )

    # If no institutions found, add a failing placeholder node to reflect failure
    if total_found == 0:
        evaluator.add_custom_node(
            result=False,
            id="no_institution_provided",
            desc="No institution entry with required fields was found in the answer.",
            parent=institutions_root,
            critical=True,
        )
        return evaluator.get_summary()

    # Evaluate up to the first MAX_INSTITUTIONS_TO_EVAL institutions
    for idx, inst in enumerate(extracted.institutions[:MAX_INSTITUTIONS_TO_EVAL]):
        await verify_institution(evaluator, institutions_root, inst, idx)

    return evaluator.get_summary()