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
TASK_ID = "sec_universities_r1_law_aacsb_abet"
TASK_DESCRIPTION = """Identify three public universities that are current members of the Southeastern Conference (SEC) and meet ALL of the following criteria:

1. The university must have been founded (established or chartered) before 1850.
2. The university must have a total enrollment exceeding 40,000 students as of Fall 2024 or Fall 2025.
3. The university must hold R1 classification ("Very High Research Spending and Doctorate Production") in the Carnegie Classification of Institutions of Higher Education.
4. The university must have a law school ranked in the top 60 nationally according to the U.S. News & World Report 2025 Best Law Schools rankings.
5. The university must have an AACSB-accredited business school.
6. The university must have at least three ABET-accredited undergraduate engineering programs.

For each of the three universities identified, provide the following information:
- The university's official name
- The year the university was founded (established or chartered)
- The total student enrollment figure (specify whether Fall 2024 or Fall 2025)
- The law school's official name and its U.S. News ranking
- The business school's official name and confirmation of AACSB accreditation
- The names of at least three ABET-accredited undergraduate engineering programs at the university
- A reference URL for each piece of information provided
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    """Information for a single engineering program."""
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    abet_note: Optional[str] = None  # any note like "ABET accredited" as stated in the answer


class UniversityInfo(BaseModel):
    """All required info for a single university."""
    # Identification + SEC/public
    name: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)  # URLs that support identification/SEC/public status
    sec_membership_note: Optional[str] = None
    public_status_note: Optional[str] = None

    # Founding
    founding_year: Optional[str] = None
    founding_urls: List[str] = Field(default_factory=list)

    # Enrollment
    enrollment_figure: Optional[str] = None  # prefer strings, e.g., "41,500" or "about 42,000"
    enrollment_term: Optional[str] = None  # e.g., "Fall 2024" or "Fall 2025"
    enrollment_urls: List[str] = Field(default_factory=list)

    # R1 classification
    r1_status_note: Optional[str] = None
    r1_urls: List[str] = Field(default_factory=list)

    # Law school
    law_school_name: Optional[str] = None
    law_school_ranking: Optional[str] = None  # e.g., "#45", "No. 45", "Rank 45 (tie)"
    law_school_urls: List[str] = Field(default_factory=list)

    # Business school / AACSB
    business_school_name: Optional[str] = None
    aacsb_status_note: Optional[str] = None
    business_school_urls: List[str] = Field(default_factory=list)

    # ABET engineering programs
    engineering_programs: List[ProgramInfo] = Field(default_factory=list)
    engineering_urls: List[str] = Field(default_factory=list)  # general ABET confirmation URLs (e.g., ABET directory)


class SECUniversitiesExtraction(BaseModel):
    """Root extraction structure listing universities."""
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to five universities from the answer that the agent claims meet ALL specified SEC criteria. For each university, return a JSON object containing:

    1) name: Official university name exactly as presented in the answer (string or null).
    2) identification_urls: Array of URLs used for identification and to support SEC membership and public status. These may include the official SEC site, the university's official pages, Wikipedia, or credible news articles. If none are provided, return an empty array.
    3) sec_membership_note: If the answer text explicitly states SEC membership, capture the phrase; otherwise null.
    4) public_status_note: If the answer text explicitly states the university is public, capture the phrase; otherwise null.

    5) founding_year: The stated founding/established/charter year exactly as written in the answer; use strings (e.g., "1831"). If missing, null.
    6) founding_urls: Array of URLs the answer provides to support the founding year. If missing, empty array.

    7) enrollment_figure: The total enrollment figure exactly as stated (string, may include commas/words). If missing, null.
    8) enrollment_term: The term associated with the enrollment figure (either "Fall 2024" or "Fall 2025"), exactly as written. If missing, null.
    9) enrollment_urls: Array of URLs provided to support the enrollment figure (e.g., official facts page, IR office, IPEDS). If missing, empty array.

    10) r1_status_note: If the answer text explicitly mentions Carnegie R1 ("Very High Research Activity" or "Very High Research Spending and Doctorate Production"), capture the phrase; otherwise null.
    11) r1_urls: Array of URLs provided to support R1 classification (e.g., Carnegie Classification pages, university announcements). If missing, empty array.

    12) law_school_name: Official law school name as stated. If missing, null.
    13) law_school_ranking: The U.S. News & World Report 2025 Best Law Schools ranking string (e.g., "#45"). If missing, null.
    14) law_school_urls: Array of URLs provided to support the ranking. If missing, empty array.

    15) business_school_name: Official business school name as stated. If missing, null.
    16) aacsb_status_note: If the answer explicitly mentions AACSB accreditation, capture the phrase; otherwise null.
    17) business_school_urls: Array of URLs provided to support AACSB accreditation. If missing, empty array.

    18) engineering_programs: Array of at least three undergraduate engineering programs. For each program, include:
        - name: Program name exactly as presented (string or null).
        - urls: Array of URLs specifically associated with the program (ABET directory page, program page) or empty array.
        - abet_note: If the answer explicitly mentions ABET accreditation for this program, capture the phrase; otherwise null.
    19) engineering_urls: Array of general URLs confirming ABET-accredited programs at the university (e.g., ABET program search results for the university), or empty array.

    IMPORTANT:
    - Extract only what is explicitly present in the agent's answer. Do NOT invent information.
    - If any required field is missing, set it to null or empty array as appropriate.
    - Ensure all URLs are valid strings; include markdown-linked URLs by extracting the actual URL.
    - Preserve formatting for names and ranking strings exactly as written.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ensure_three_programs(uni: UniversityInfo) -> List[ProgramInfo]:
    """Return the first three engineering programs, padding with empty ProgramInfo if fewer are provided."""
    programs = list(uni.engineering_programs[:3])
    while len(programs) < 3:
        programs.append(ProgramInfo())
    return programs


def pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Choose primary if available, otherwise fallback."""
    return primary if primary else fallback


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    index: int,
) -> None:
    """
    Build the verification subtree for one university and run checks.
    """
    # ---- University aggregate node (non-critical to allow partial credit across universities) ----
    uni_node = evaluator.add_parallel(
        id=f"university_{index}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][index-1]} university meets all criteria with complete information",
        parent=parent_node,
        critical=False,
    )

    # ========== 1) Identification (SEC member + public) ==========
    ident_node = evaluator.add_parallel(
        id=f"university_{index}_identification",
        desc="University is correctly identified and is a current SEC member",
        parent=uni_node,
        critical=True,
    )

    # 1.1 Official name provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"university_{index}_name",
        desc="University's official name is provided",
        parent=ident_node,
        critical=True
    )

    # 1.2 SEC membership (verify by URLs)
    sec_member_leaf = evaluator.add_leaf(
        id=f"university_{index}_sec_membership",
        desc="University is confirmed as a current SEC member",
        parent=ident_node,
        critical=True
    )
    sec_claim = f"{uni.name or 'The university'} is a current member of the Southeastern Conference (SEC)."
    await evaluator.verify(
        claim=sec_claim,
        node=sec_member_leaf,
        sources=uni.identification_urls,
        additional_instruction=(
            "Verify that the provided webpage(s) explicitly indicate the university is a current SEC member. "
            "Accept synonyms like 'SEC', 'Southeastern Conference'. If the page lists SEC members and includes the university, that counts."
        ),
    )

    # 1.3 Public institution (verify by URLs)
    public_leaf = evaluator.add_leaf(
        id=f"university_{index}_public_status",
        desc="University is confirmed as a public institution",
        parent=ident_node,
        critical=True
    )
    public_claim = f"{uni.name or 'The university'} is a public institution (public university)."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=uni.identification_urls,
        additional_instruction=(
            "Confirm that the university is described as 'public', 'public research university', 'state university', or equivalent on the cited page(s)."
        ),
    )

    # 1.4 Identification/SEC membership reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.identification_urls),
        id=f"university_{index}_basic_info_urls",
        desc="Reference URL provided for university identification and SEC membership",
        parent=ident_node,
        critical=True
    )

    # ========== 2) Founding year (< 1850) ==========
    founding_node = evaluator.add_parallel(
        id=f"university_{index}_founding_criterion",
        desc="University's founding year is before 1850 with supporting evidence",
        parent=uni_node,
        critical=True
    )

    # 2.1 Founding year stated and before 1850 (verify by URLs)
    founding_leaf = evaluator.add_leaf(
        id=f"university_{index}_founding_year",
        desc="Founding year (establishment or charter date) is stated and is before 1850",
        parent=founding_node,
        critical=True
    )
    founding_claim = (
        f"{uni.name or 'The university'} was founded (established or chartered) in {uni.founding_year or 'UNKNOWN'}, "
        "and this founding year is before 1850."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=founding_leaf,
        sources=uni.founding_urls,
        additional_instruction=(
            "Confirm the founding year from the cited page(s). Accept 'chartered' or 'established' as founding. "
            "Also verify that the year is strictly earlier than 1850."
        ),
    )

    # 2.2 Founding reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.founding_urls),
        id=f"university_{index}_founding_url",
        desc="Reference URL provided confirming the founding year",
        parent=founding_node,
        critical=True
    )

    # ========== 3) Enrollment (> 40,000, Fall 2024/2025) ==========
    enroll_node = evaluator.add_parallel(
        id=f"university_{index}_enrollment_criterion",
        desc="University's enrollment exceeds 40,000 students with supporting evidence",
        parent=uni_node,
        critical=True
    )

    # 3.1 Enrollment figure (verify by URLs)
    enroll_fig_leaf = evaluator.add_leaf(
        id=f"university_{index}_enrollment_figure",
        desc="Total enrollment figure is stated and exceeds 40,000",
        parent=enroll_node,
        critical=True
    )
    enroll_fig_claim = (
        f"The total enrollment of {uni.name or 'the university'} is {uni.enrollment_figure or 'UNKNOWN'} "
        f"as of {uni.enrollment_term or 'UNKNOWN'}, and it exceeds 40,000 students."
    )
    await evaluator.verify(
        claim=enroll_fig_claim,
        node=enroll_fig_leaf,
        sources=uni.enrollment_urls,
        additional_instruction=(
            "Verify the enrollment figure and confirm that it exceeds 40,000. "
            "Allow modest rounding or approximate phrasing (e.g., 'about 41,000')."
        ),
    )

    # 3.2 Enrollment term specified (verify by URLs)
    enroll_term_leaf = evaluator.add_leaf(
        id=f"university_{index}_enrollment_term",
        desc="Enrollment term (Fall 2024 or Fall 2025) is specified",
        parent=enroll_node,
        critical=True
    )
    enroll_term_claim = (
        f"The enrollment figure for {uni.name or 'the university'} is reported for {uni.enrollment_term or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=enroll_term_claim,
        node=enroll_term_leaf,
        sources=uni.enrollment_urls,
        additional_instruction=(
            "Confirm that the cited page(s) associate the enrollment figure with Fall 2024 or Fall 2025."
        ),
    )

    # 3.3 Enrollment reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.enrollment_urls),
        id=f"university_{index}_enrollment_url",
        desc="Reference URL provided confirming the enrollment figure",
        parent=enroll_node,
        critical=True
    )

    # ========== 4) R1 classification ==========
    r1_node = evaluator.add_parallel(
        id=f"university_{index}_r1_criterion",
        desc="University holds R1 Carnegie classification with supporting evidence",
        parent=uni_node,
        critical=True
    )

    # 4.1 R1 classification status confirmed (verify by URLs)
    r1_status_leaf = evaluator.add_leaf(
        id=f"university_{index}_r1_status",
        desc="R1 classification status is confirmed",
        parent=r1_node,
        critical=True
    )
    r1_claim = (
        f"{uni.name or 'The university'} holds Carnegie R1 classification "
        "(Very High Research Activity or Very High Research Spending and Doctorate Production)."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_status_leaf,
        sources=uni.r1_urls,
        additional_instruction=(
            "Confirm the university is classified as R1. Accept either the legacy label 'Very High Research Activity' "
            "or the 2025 phrasing 'Very High Research Spending and Doctorate Production'."
        ),
    )

    # 4.2 R1 reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.r1_urls),
        id=f"university_{index}_r1_url",
        desc="Reference URL provided confirming R1 classification",
        parent=r1_node,
        critical=True
    )

    # ========== 5) Law school (top 60, USNWR 2025) ==========
    law_node = evaluator.add_parallel(
        id=f"university_{index}_law_school_criterion",
        desc="University has a law school ranked in top 60 with supporting evidence",
        parent=uni_node,
        critical=True
    )

    # 5.1 Law school name provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.law_school_name and uni.law_school_name.strip()),
        id=f"university_{index}_law_school_name",
        desc="Law school's official name is provided",
        parent=law_node,
        critical=True
    )

    # 5.2 Ranking stated and is 60 or better (verify by URLs)
    law_rank_leaf = evaluator.add_leaf(
        id=f"university_{index}_law_school_ranking",
        desc="U.S. News 2025 ranking is stated and is 60 or better",
        parent=law_node,
        critical=True
    )
    law_rank_claim = (
        f"The law school {uni.law_school_name or 'the law school'} has a U.S. News & World Report 2025 Best Law Schools "
        f"ranking of {uni.law_school_ranking or 'UNKNOWN'}, and that rank is within the top 60 nationally."
    )
    await evaluator.verify(
        claim=law_rank_claim,
        node=law_rank_leaf,
        sources=uni.law_school_urls,
        additional_instruction=(
            "Check the cited page(s) for the 2025 Best Law Schools ranking. Parse the numeric rank from strings like '#45' or 'No. 45 (tie)'. "
            "Confirm that the rank is 60 or better (i.e., numeric rank <= 60)."
        ),
    )

    # 5.3 Law school ranking reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.law_school_urls),
        id=f"university_{index}_law_school_url",
        desc="Reference URL provided confirming law school ranking",
        parent=law_node,
        critical=True
    )

    # ========== 6) Business school (AACSB) ==========
    biz_node = evaluator.add_parallel(
        id=f"university_{index}_business_school_criterion",
        desc="University has AACSB-accredited business school with supporting evidence",
        parent=uni_node,
        critical=True
    )

    # 6.1 Business school name provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.business_school_name and uni.business_school_name.strip()),
        id=f"university_{index}_business_school_name",
        desc="Business school's official name is provided",
        parent=biz_node,
        critical=True
    )

    # 6.2 AACSB accreditation confirmed (verify by URLs)
    aacsb_leaf = evaluator.add_leaf(
        id=f"university_{index}_aacsb_status",
        desc="AACSB accreditation is confirmed",
        parent=biz_node,
        critical=True
    )
    aacsb_claim = (
        f"The business school {uni.business_school_name or 'the business school'} is accredited by AACSB."
    )
    await evaluator.verify(
        claim=aacsb_claim,
        node=aacsb_leaf,
        sources=uni.business_school_urls,
        additional_instruction=(
            "Confirm that the cited page(s)—such as the AACSB official directory or the school's accreditation page—explicitly indicate AACSB accreditation."
        ),
    )

    # 6.3 Business school accreditation reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.business_school_urls),
        id=f"university_{index}_business_school_url",
        desc="Reference URL provided confirming AACSB accreditation",
        parent=biz_node,
        critical=True
    )

    # ========== 7) Engineering (>= three ABET-accredited programs) ==========
    eng_node = evaluator.add_parallel(
        id=f"university_{index}_engineering_criterion",
        desc="University has at least three ABET-accredited undergraduate engineering programs with supporting evidence",
        parent=uni_node,
        critical=True
    )

    # Ensure exactly three program checks
    programs = ensure_three_programs(uni)

    for j, prog in enumerate(programs, start=1):
        prog_node = evaluator.add_parallel(
            id=f"university_{index}_engineering_program_{j}",
            desc=f"{['First','Second','Third'][j-1]} ABET-accredited undergraduate engineering program is identified",
            parent=eng_node,
            critical=True
        )

        # Program name provided (existence)
        evaluator.add_custom_node(
            result=bool(prog.name and prog.name.strip()),
            id=f"university_{index}_eng_prog_{j}_name",
            desc="Program name is provided",
            parent=prog_node,
            critical=True
        )

        # ABET accreditation confirmed (verify by URLs)
        abet_leaf = evaluator.add_leaf(
            id=f"university_{index}_eng_prog_{j}_abet",
            desc="ABET accreditation is confirmed",
            parent=prog_node,
            critical=True
        )
        abet_sources = pick_sources(prog.urls, uni.engineering_urls)
        abet_claim = (
            f"The undergraduate engineering program '{prog.name or 'UNKNOWN PROGRAM'}' at {uni.name or 'the university'} is ABET-accredited."
        )
        await evaluator.verify(
            claim=abet_claim,
            node=abet_leaf,
            sources=abet_sources,
            additional_instruction=(
                "Use the ABET program search or official accreditation pages to confirm ABET accreditation. "
                "Accept accreditation under the Engineering Accreditation Commission (EAC) or appropriate ABET commission for undergraduate programs."
            ),
        )

    # Reference URL(s) provided to confirm ABET programs (existence)
    evaluator.add_custom_node(
        result=bool(uni.engineering_urls),
        id=f"university_{index}_engineering_url",
        desc="Reference URL provided confirming ABET accreditation of engineering programs",
        parent=eng_node,
        critical=True
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
    Evaluate an answer for the SEC universities criteria task.
    """
    # Initialize evaluator (root non-critical to allow partial credit across universities; parallel aggregation)
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
        default_model=model
    )

    # Extract structured universities info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=SECUniversitiesExtraction,
        extraction_name="sec_universities_extraction"
    )

    # Only evaluate the first three universities; pad if fewer
    universities = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityInfo())

    # Build verification subtrees for each of the three universities
    for i, uni in enumerate(universities, start=1):
        await verify_university(evaluator, root, uni, i)

    # Return structured summary
    return evaluator.get_summary()