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
TASK_ID = "top_cs_phd_fall2026"
TASK_DESCRIPTION = """
Identify three distinct top-ranked computer science PhD programs in the United States that are accepting applications for Fall 2026 admission. For each program, provide the following information:

1. Institution Name: The name of the university offering the program
2. Ranking Verification: Confirmation that the program is ranked in the top 10 for computer science graduate programs according to at least one major ranking source (U.S. News & World Report, QS World University Rankings, or Times Higher Education World University Rankings), along with a reference URL
3. Application Deadline: The specific application deadline for Fall 2026 PhD admission, which must be in December 2025, along with a reference URL
4. GRE Policy: Confirmation that the program has waived the GRE requirement or made it optional for Fall 2026 PhD applicants, along with a reference URL
5. Recommendation Letters: The number of recommendation letters required for application, along with a reference URL
6. Annual Stipend: The annual stipend amount for PhD students in the 2025-2026 or 2026-2027 academic year, which must be at least $35,000, along with a reference URL
7. Funding Coverage: Confirmation that the program provides full funding including tuition, fees, and health insurance coverage, along with a reference URL
8. Guaranteed Funding Duration: The guaranteed funding duration for PhD students making satisfactory academic progress (typically 5 years), along with a reference URL
9. Course Requirements: A description of the graduate course requirements for PhD students, along with a reference URL
10. Qualifying Examination: Confirmation that the program has a qualifying or candidacy examination requirement, along with a reference URL

All three programs must be different institutions, and each must satisfy all the specified criteria. Provide reference URLs from official university websites or authoritative ranking sources to support each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    institution: Optional[str] = None

    # Ranking
    ranking_urls: List[str] = Field(default_factory=list)

    # Application requirements
    application_deadline: Optional[str] = None
    application_deadline_urls: List[str] = Field(default_factory=list)

    gre_policy_text: Optional[str] = None
    gre_policy_urls: List[str] = Field(default_factory=list)

    recommendation_letters_count: Optional[str] = None
    recommendation_urls: List[str] = Field(default_factory=list)

    # Funding package
    stipend_amount: Optional[str] = None
    stipend_urls: List[str] = Field(default_factory=list)

    full_funding_urls: List[str] = Field(default_factory=list)
    health_insurance_urls: List[str] = Field(default_factory=list)

    guaranteed_funding_duration: Optional[str] = None
    funding_duration_urls: List[str] = Field(default_factory=list)

    # Program structure
    course_requirements_summary: Optional[str] = None
    course_requirements_urls: List[str] = Field(default_factory=list)

    qualifying_exam_text: Optional[str] = None
    qualifying_exam_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract information for up to the first three distinct U.S. universities listed in the answer that offer a PhD program in Computer Science and are being proposed for Fall 2026 admission. For each program, return an object with the following fields. If any field is not explicitly provided in the answer, set it to null for strings or an empty list for arrays.

    For each program, extract:
    - institution: The university name (e.g., "Stanford University").
    - ranking_urls: A list of URL(s) from major sources confirming top-10 CS ranking (acceptable sources: U.S. News & World Report, QS, Times Higher Education). Extract only URLs explicitly present in the answer.
    - application_deadline: The concrete application deadline string for Fall 2026 PhD admission (must be in December 2025 if stated).
    - application_deadline_urls: URL(s) that state the application deadline for Fall 2026.
    - gre_policy_text: The stated GRE policy text for Fall 2026 PhD applicants (e.g., "optional", "not required", "waived").
    - gre_policy_urls: URL(s) that state the GRE policy.
    - recommendation_letters_count: The stated number of recommendation letters required (e.g., "3").
    - recommendation_urls: URL(s) that state the recommendation letter requirement.
    - stipend_amount: The annual stipend amount string for 2025–2026 or 2026–2027 (e.g., "$37,000 per year").
    - stipend_urls: URL(s) that state the stipend/assistantship amount.
    - full_funding_urls: URL(s) that confirm full funding coverage including tuition and mandatory fees.
    - health_insurance_urls: URL(s) that confirm health insurance coverage is included.
    - guaranteed_funding_duration: The stated guaranteed funding duration text (e.g., "5 years").
    - funding_duration_urls: URL(s) that state the guaranteed funding duration.
    - course_requirements_summary: A brief phrase/sentence describing the PhD CS coursework/credit requirements (if present).
    - course_requirements_urls: URL(s) that specify graduate course requirements for the CS PhD.
    - qualifying_exam_text: A brief phrase/sentence confirming the presence of a qualifying/candidacy exam requirement.
    - qualifying_exam_urls: URL(s) that confirm a qualifying or candidacy exam requirement.

    Rules:
    - Extract only URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
    - Keep institutions distinct; only the first occurrence of a repeated institution should be extracted.
    - Limit to the first three qualifying institutions mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def aggregate_program_urls(p: ProgramItem) -> List[str]:
    urls: List[str] = []
    urls.extend(p.application_deadline_urls or [])
    urls.extend(p.gre_policy_urls or [])
    urls.extend(p.recommendation_urls or [])
    urls.extend(p.stipend_urls or [])
    urls.extend(p.full_funding_urls or [])
    urls.extend(p.health_insurance_urls or [])
    urls.extend(p.funding_duration_urls or [])
    urls.extend(p.course_requirements_urls or [])
    urls.extend(p.qualifying_exam_urls or [])
    # ranking_urls typically used only for ranking checks
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Per-program verification builder                                            #
# --------------------------------------------------------------------------- #
async def verify_single_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    idx_one_based: int,
) -> None:
    display_inst = program.institution or f"Program #{idx_one_based}"

    program_node = evaluator.add_parallel(
        id=f"program_{idx_one_based}",
        desc=f"{display_inst} — qualifying computer science PhD program meeting all specified criteria",
        parent=parent_node,
        critical=False,  # allow partial credit per program
    )

    # -------------------- Program Identity & Ranking -------------------- #
    identity_node = evaluator.add_parallel(
        id=f"program_{idx_one_based}_identity",
        desc="Program identification and ranking verification",
        parent=program_node,
        critical=True,
    )

    # Institution name presence (critical)
    evaluator.add_custom_node(
        result=nonempty(program.institution),
        id=f"program_{idx_one_based}_institution_name",
        desc="The program must be from a specific U.S. institution offering a PhD in Computer Science (institution name provided)",
        parent=identity_node,
        critical=True,
    )

    # Ranking URLs presence (critical)
    evaluator.add_custom_node(
        result=has_urls(program.ranking_urls),
        id=f"program_{idx_one_based}_ranking_url_present",
        desc="URL reference confirming the top-10 ranking status is provided",
        parent=identity_node,
        critical=True,
    )

    # Top-10 ranking supported by ranking source(s) (critical)
    rank_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_top10_ranking_supported",
        desc="The program is ranked in the top 10 for CS graduate programs per at least one major source (U.S. News, QS, or THE)",
        parent=identity_node,
        critical=True,
    )
    rank_claim = (
        f"According to at least one major ranking source (U.S. News & World Report, QS, or Times Higher Education), "
        f"the computer science graduate program at {display_inst} is ranked within the top 10."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=program.ranking_urls,
        additional_instruction=(
            "Only accept rankings from: usnews.com (U.S. News), topuniversities.com (QS), or timeshighereducation.com (THE). "
            "If the page clearly shows the institution within the top 10 for Computer Science (graduate or equivalent), mark as supported. "
            "If it is a different field or not top 10, mark as not supported."
        ),
    )

    # -------------------- Application Requirements --------------------- #
    app_node = evaluator.add_parallel(
        id=f"program_{idx_one_based}_application_requirements",
        desc="Application process requirements for Fall 2026 admission",
        parent=program_node,
        critical=True,
    )

    # Deadline presence (critical)
    evaluator.add_custom_node(
        result=nonempty(program.application_deadline) and has_urls(program.application_deadline_urls),
        id=f"program_{idx_one_based}_application_deadline_present",
        desc="Application deadline (string) provided and at least one deadline URL cited",
        parent=app_node,
        critical=True,
    )

    # Deadline supported by URL(s) and in December 2025 for Fall 2026 (critical)
    deadline_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_application_deadline_supported",
        desc="Application deadline for Fall 2026 is in December 2025 and is supported by cited URL(s)",
        parent=app_node,
        critical=True,
    )
    deadline_val = program.application_deadline or "the stated date"
    deadline_claim = (
        f"The PhD in Computer Science application deadline for Fall 2026 at {display_inst} is {deadline_val}, "
        f"and this deadline falls in December 2025."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=program.application_deadline_urls,
        additional_instruction=(
            "Verify that the cited page(s) specify a Fall 2026 application deadline and that the date is in December 2025. "
            "Accept reasonable wording variants (e.g., 'Dec 15, 2025'). If the page lists multiple deadlines, ensure that the relevant PhD CS deadline is in December 2025."
        ),
    )

    # GRE policy presence (critical)
    evaluator.add_custom_node(
        result=nonempty(program.gre_policy_text) and has_urls(program.gre_policy_urls),
        id=f"program_{idx_one_based}_gre_policy_present",
        desc="GRE policy text provided and at least one GRE policy URL cited",
        parent=app_node,
        critical=True,
    )

    # GRE policy supported (critical)
    gre_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_gre_waived_or_optional_supported",
        desc="The program has waived the GRE requirement or made it optional for Fall 2026 PhD applicants (supported by URL)",
        parent=app_node,
        critical=True,
    )
    gre_claim = (
        f"For Fall 2026 PhD applicants in Computer Science at {display_inst}, the GRE requirement is waived or optional."
    )
    await evaluator.verify(
        claim=gre_claim,
        node=gre_leaf,
        sources=program.gre_policy_urls,
        additional_instruction=(
            "Confirm that GRE is not required (waived) or explicitly listed as optional/encouraged but not required for Fall 2026. "
            "If the page is for a different term or degree, or clearly requires GRE, mark as not supported."
        ),
    )

    # Recommendation letters presence (critical)
    evaluator.add_custom_node(
        result=nonempty(program.recommendation_letters_count) and has_urls(program.recommendation_urls),
        id=f"program_{idx_one_based}_recommendations_present",
        desc="Recommendation letter count provided and at least one URL cited",
        parent=app_node,
        critical=True,
    )

    # Recommendation letters supported (critical)
    rec_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_recommendations_supported",
        desc="The number of recommendation letters required is supported by cited URL(s)",
        parent=app_node,
        critical=True,
    )
    rec_claim = (
        f"The Computer Science PhD application at {display_inst} requires {program.recommendation_letters_count} recommendation letters."
    )
    await evaluator.verify(
        claim=rec_claim,
        node=rec_leaf,
        sources=program.recommendation_urls,
        additional_instruction="Accept equivalent wording such as 'three letters of recommendation'.",
    )

    # -------------------- Funding Package -------------------------------- #
    funding_node = evaluator.add_parallel(
        id=f"program_{idx_one_based}_funding_package",
        desc="Financial support and funding details for PhD students",
        parent=program_node,
        critical=True,
    )

    # Stipend presence (critical)
    evaluator.add_custom_node(
        result=nonempty(program.stipend_amount) and has_urls(program.stipend_urls),
        id=f"program_{idx_one_based}_stipend_present",
        desc="Annual stipend amount provided and stipend URL(s) cited",
        parent=funding_node,
        critical=True,
    )

    # Stipend supported and >= $35,000 (critical)
    stipend_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_stipend_supported",
        desc="Annual stipend for 2025–2026 or 2026–2027 is at least $35,000 (supported by URL)",
        parent=funding_node,
        critical=True,
    )
    stipend_claim = (
        f"For the 2025–2026 or 2026–2027 academic year, the standard annual stipend for CS PhD students at {display_inst} is at least $35,000."
    )
    await evaluator.verify(
        claim=stipend_claim,
        node=stipend_leaf,
        sources=program.stipend_urls,
        additional_instruction=(
            "If the page provides 9-month plus typical summer support amounts, consider the total 12-month support. "
            "If the page lists ranges or department-wide assistantship rates, accept as long as the annualized total is ≥ $35,000."
        ),
    )

    # Full funding presence (critical)
    evaluator.add_custom_node(
        result=has_urls(program.full_funding_urls),
        id=f"program_{idx_one_based}_full_funding_present",
        desc="At least one URL cited for full funding coverage (tuition and mandatory fees)",
        parent=funding_node,
        critical=True,
    )

    # Full funding supported (critical)
    fullfund_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_full_funding_supported",
        desc="The program provides full funding coverage including tuition and mandatory fees (supported by URL)",
        parent=funding_node,
        critical=True,
    )
    fullfund_claim = (
        f"The Computer Science PhD program at {display_inst} provides full funding that includes tuition and mandatory fees coverage."
    )
    await evaluator.verify(
        claim=fullfund_claim,
        node=fullfund_leaf,
        sources=program.full_funding_urls,
        additional_instruction="Accept language such as 'full tuition remission' and 'payment of mandatory/standard fees'.",
    )

    # Health insurance presence (critical)
    evaluator.add_custom_node(
        result=has_urls(program.health_insurance_urls),
        id=f"program_{idx_one_based}_health_insurance_present",
        desc="At least one URL cited confirming health insurance coverage is included",
        parent=funding_node,
        critical=True,
    )

    # Health insurance supported (critical)
    health_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_health_insurance_supported",
        desc="Funding package includes health insurance coverage (supported by URL)",
        parent=funding_node,
        critical=True,
    )
    health_claim = (
        f"The funding package for CS PhD students at {display_inst} includes health insurance coverage (or an equivalent premium subsidy)."
    )
    await evaluator.verify(
        claim=health_claim,
        node=health_leaf,
        sources=program.health_insurance_urls,
        additional_instruction="Accept explicit insurance coverage or an explicit premium subsidy that effectively covers health insurance.",
    )

    # -------------------- Program Structure ------------------------------ #
    structure_node = evaluator.add_parallel(
        id=f"program_{idx_one_based}_program_structure",
        desc="Core program requirements and structure",
        parent=program_node,
        critical=True,
    )

    # Course requirements presence (critical)
    evaluator.add_custom_node(
        result=has_urls(program.course_requirements_urls),
        id=f"program_{idx_one_based}_course_requirements_present",
        desc="At least one URL cited specifying graduate course requirements for the CS PhD",
        parent=structure_node,
        critical=True,
    )

    # Course requirements supported (critical)
    coursereq_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_course_requirements_supported",
        desc="The cited page specifies graduate course requirements for PhD students (supported by URL)",
        parent=structure_node,
        critical=True,
    )
    coursereq_claim = (
        f"The cited page(s) specify the graduate course (or credit) requirements for the Computer Science PhD program at {display_inst}."
    )
    await evaluator.verify(
        claim=coursereq_claim,
        node=coursereq_leaf,
        sources=program.course_requirements_urls,
        additional_instruction="Accept degree requirement pages, handbooks, or departmental policy pages that outline coursework/credit requirements.",
    )

    # Qualifying exam presence (critical)
    evaluator.add_custom_node(
        result=has_urls(program.qualifying_exam_urls),
        id=f"program_{idx_one_based}_qualifying_exam_present",
        desc="At least one URL cited confirming a qualifying/candidacy exam requirement",
        parent=structure_node,
        critical=True,
    )

    # Qualifying exam supported (critical)
    qual_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_qualifying_exam_supported",
        desc="The program has a qualifying or candidacy examination requirement (supported by URL)",
        parent=structure_node,
        critical=True,
    )
    qual_claim = (
        f"The Computer Science PhD program at {display_inst} includes a qualifying exam or candidacy examination requirement."
    )
    await evaluator.verify(
        claim=qual_claim,
        node=qual_leaf,
        sources=program.qualifying_exam_urls,
        additional_instruction="Accept equivalent terms like preliminary exam, candidacy exam, or qualifying milestone.",
    )

    # Guaranteed funding duration presence (critical)
    evaluator.add_custom_node(
        result=has_urls(program.funding_duration_urls),
        id=f"program_{idx_one_based}_funding_duration_present",
        desc="At least one URL cited confirming guaranteed funding duration",
        parent=structure_node,
        critical=True,
    )

    # Guaranteed funding duration supported (critical)
    gfd_leaf = evaluator.add_leaf(
        id=f"program_{idx_one_based}_funding_duration_supported",
        desc="Program guarantees funding for at least 5 years for PhD students in good standing (supported by URL)",
        parent=structure_node,
        critical=True,
    )
    gfd_claim = (
        f"The Computer Science PhD program at {display_inst} guarantees funding for at least 5 years for students in good academic standing."
    )
    await evaluator.verify(
        claim=gfd_claim,
        node=gfd_leaf,
        sources=program.funding_duration_urls,
        additional_instruction=(
            "Accept language such as 'guaranteed support for five years' or 'at least five years of funding'. "
            "If the duration is explicitly less than 5 years, mark as not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Distinct institutions verification                                          #
# --------------------------------------------------------------------------- #
def add_distinct_institutions_checks(
    evaluator: Evaluator,
    parent_node,
    institutions: List[Optional[str]],
) -> None:
    distinct_node = evaluator.add_parallel(
        id="institutions_distinct",
        desc="All three programs are from distinct institutions",
        parent=parent_node,
        critical=True,
    )

    # Pairwise checks as separate custom (leaf) nodes
    a, b, c = (institutions + [None, None, None])[:3]

    evaluator.add_custom_node(
        result=nonempty(a) and nonempty(b) and (a.strip().lower() != b.strip().lower()),
        id="distinct_1_2",
        desc="Program 1 and Program 2 are different institutions",
        parent=distinct_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(a) and nonempty(c) and (a.strip().lower() != c.strip().lower()),
        id="distinct_1_3",
        desc="Program 1 and Program 3 are different institutions",
        parent=distinct_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(b) and nonempty(c) and (b.strip().lower() != c.strip().lower()),
        id="distinct_2_3",
        desc="Program 2 and Program 3 are different institutions",
        parent=distinct_node,
        critical=True,
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
    # Initialize evaluator (root is non-critical to allow partial scoring; we enforce key constraints via critical children)
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

    # Extract structured program data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep only the first 3 programs; if fewer provided, pad with empty placeholders
    programs: List[ProgramItem] = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramItem())

    # Build program-specific verification subtrees
    for i in range(3):
        await verify_single_program(
            evaluator=evaluator,
            parent_node=root,
            program=programs[i],
            idx_one_based=i + 1,
        )

    # Global constraint: all three institutions must be distinct
    add_distinct_institutions_checks(
        evaluator=evaluator,
        parent_node=root,
        institutions=[p.institution for p in programs],
    )

    # Optional: record a compact summary of extracted institutions for debugging
    evaluator.add_custom_info(
        info={
            "institutions": [p.institution for p in programs],
            "ranking_url_counts": [len(p.ranking_urls) for p in programs],
            "deadline_url_counts": [len(p.application_deadline_urls) for p in programs],
            "gre_url_counts": [len(p.gre_policy_urls) for p in programs],
            "rec_url_counts": [len(p.recommendation_urls) for p in programs],
            "stipend_url_counts": [len(p.stipend_urls) for p in programs],
        },
        info_type="extraction_overview",
    )

    # Return final summary
    return evaluator.get_summary()