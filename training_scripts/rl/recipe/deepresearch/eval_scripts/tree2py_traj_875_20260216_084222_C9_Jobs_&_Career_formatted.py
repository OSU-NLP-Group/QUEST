import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "career_services_universities_2026"
TASK_DESCRIPTION = (
    "Identify 3 universities in the United States that meet ALL of the following criteria based on publicly available "
    "information as of February 2026:\n\n"
    "1. The university must be listed in the Princeton Review's 'Best Career Services' rankings for either 2024 or 2025.\n\n"
    "2. The university must report career outcomes data that follows NACE (National Association of Colleges and Employers) "
    "First Destination Survey standards, including reporting either a Knowledge Rate or Career Outcomes Rate.\n\n"
    "3. The university must have undergraduate enrollment of fewer than 15,000 students.\n\n"
    "4. For each identified university, provide the following specific information:\n"
    "   - The university's rank or position in the Princeton Review Best Career Services rankings (if a specific numerical rank "
    "is provided), or confirmation of inclusion in the ranking list\n"
    "   - The specific Career Outcomes Rate or Knowledge Rate percentage reported by the university for their most recent "
    "graduating class\n"
    "   - The graduating class year for which the career outcomes data was reported\n"
    "   - The total undergraduate enrollment count\n"
    "   - One specific operational detail about their career services center (such as: number of career fairs held annually, "
    "career center staff size, student engagement percentage, or specific career services programs offered)\n\n"
    "For each piece of information, provide the URL reference where this information was found."
)

ALLOWED_PR_YEARS = ["2024", "2025"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None

    # Princeton Review Best Career Services
    princeton_review_year: Optional[str] = None  # e.g., "2024" or "2025"
    princeton_review_rank_or_inclusion: Optional[str] = None  # e.g., "No. 5" or "Included"
    princeton_review_urls: List[str] = Field(default_factory=list)

    # Career outcomes metric
    outcomes_metric_type: Optional[str] = None  # "Career Outcomes Rate" or "Knowledge Rate"
    outcomes_metric_value: Optional[str] = None  # e.g., "92%", "91.5%"
    outcomes_urls: List[str] = Field(default_factory=list)

    # Graduating class year
    graduating_class_year: Optional[str] = None  # e.g., "2024"
    graduating_class_year_urls: List[str] = Field(default_factory=list)

    # NACE compliance
    nace_compliance_statement: Optional[str] = None
    nace_urls: List[str] = Field(default_factory=list)

    # Undergraduate enrollment
    undergrad_enrollment: Optional[str] = None  # string to allow commas or ranges
    enrollment_urls: List[str] = Field(default_factory=list)

    # Operational detail about career center
    operational_detail: Optional[str] = None
    operational_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to the first 5 universities mentioned in the answer that the answer claims meet the task criteria. For each university, extract the following fields exactly as presented in the answer text (do not infer):

For each university (use array field 'universities'):
- university_name: The university's full name.

Princeton Review Best Career Services:
- princeton_review_year: The year of the Princeton Review Best Career Services ranking explicitly mentioned in the answer (e.g., "2024" or "2025"). If not explicitly stated, return null.
- princeton_review_rank_or_inclusion: The specific rank (e.g., "No. 3", "Ranked #5") or a plain confirmation of inclusion (e.g., "included", "listed"), as stated in the answer. If missing, return null.
- princeton_review_urls: All URLs provided in the answer that are cited to support the Princeton Review 'Best Career Services' ranking. Extract actual URLs only.

Career outcomes:
- outcomes_metric_type: The label of the reported metric in the answer, e.g., "Career Outcomes Rate" or "Knowledge Rate". If not explicitly named, but a percent is given, return null for this field.
- outcomes_metric_value: The percentage string for the reported Career Outcomes Rate or Knowledge Rate (e.g., "92%", "91.5%"), as presented in the answer. If not given, return null.
- outcomes_urls: All URLs for the career outcomes page(s) or report(s) cited in the answer.

Graduating class year:
- graduating_class_year: The graduating class year explicitly tied to the provided career outcomes data in the answer (e.g., "Class of 2024", "2023-2024", return the exact string). If not explicitly stated, return null.
- graduating_class_year_urls: URLs in the answer that specifically support the graduating class year. If none are provided separately, leave this array empty.

NACE compliance:
- nace_compliance_statement: Any phrase in the answer that indicates outcomes reporting follows NACE First Destination Survey standards (e.g., "in accordance with NACE FDS"). If none is present, return null.
- nace_urls: All URLs the answer cites for NACE methodology or standards compliance.

Undergraduate enrollment:
- undergrad_enrollment: The total undergraduate enrollment count exactly as written in the answer (e.g., "14,200", "about 14,000"). If not mentioned, return null.
- enrollment_urls: All URLs cited for the enrollment data.

Operational detail:
- operational_detail: One specific operational detail about the career services center that the answer provides (e.g., "hosts 8 career fairs per year", "staff of 25", "85% student engagement", "offers a dedicated micro-internship program"). If not present, return null.
- operational_urls: All URLs cited for this operational detail.

Rules:
- Extract only URLs that explicitly appear in the answer (including markdown links). Do not invent or infer any URLs.
- If a field is missing in the answer, set it to null (or empty list for URL arrays).
- Preserve percentage signs, punctuation, and formatting as in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _desc_for_university(i: int) -> str:
    if i == 0:
        return "First university meeting all specified career services criteria"
    if i == 1:
        return "Second university meeting all specified career services criteria"
    return "Third university meeting all specified career services criteria"


# --------------------------------------------------------------------------- #
# Verification for one university                                             #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, root_node, uni: UniversityItem, idx: int) -> None:
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=_desc_for_university(idx),
        parent=root_node,
        critical=False  # Overall, each university contributes partial credit independently
    )

    # -------------------- Princeton Review ranking -------------------- #
    pr_node = evaluator.add_parallel(
        id=f"u{idx+1}_princeton_review_ranking",
        desc="University is ranked in Princeton Review's 'Best Career Services' category for 2024 or 2025",
        parent=uni_node,
        critical=True
    )

    pr_urls = _sanitize_urls(uni.princeton_review_urls)
    # URL existence (critical)
    evaluator.add_custom_node(
        result=(len(pr_urls) > 0),
        id=f"u{idx+1}_pr_ranking_url",
        desc="URL reference for Princeton Review ranking",
        parent=pr_node,
        critical=True
    )

    # Verification leaf
    pr_verify_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_pr_ranking_verification",
        desc="Verification that the university appears in the Princeton Review Best Career Services rankings",
        parent=pr_node,
        critical=True
    )

    year_str = uni.princeton_review_year if uni.princeton_review_year else "2024 or 2025"
    rank_str = ""
    if uni.princeton_review_rank_or_inclusion and any(ch.isdigit() for ch in uni.princeton_review_rank_or_inclusion):
        rank_str = f" with a rank {uni.princeton_review_rank_or_inclusion}"
    pr_claim_name = uni.university_name if uni.university_name else "the university"
    pr_claim = (
        f"{pr_claim_name} appears in the Princeton Review 'Best Career Services' ranking for {year_str}{rank_str}."
    )
    await evaluator.verify(
        claim=pr_claim,
        node=pr_verify_leaf,
        sources=pr_urls,
        additional_instruction=(
            "Confirm that the page shows the institution listed under Princeton Review's 'Best Career Services' ranking "
            "for either 2024 or 2025. If a rank is provided in the claim, allow formats like 'No. X' or '#X'. "
            "Minor name variations are acceptable."
        )
    )

    # -------------------- Career outcomes rate -------------------- #
    outcomes_node = evaluator.add_parallel(
        id=f"u{idx+1}_career_outcomes_rate",
        desc="University reports a Career Outcomes Rate or Knowledge Rate for most recent graduating class",
        parent=uni_node,
        critical=True
    )

    outcomes_urls = _sanitize_urls(uni.outcomes_urls)
    evaluator.add_custom_node(
        result=(len(outcomes_urls) > 0),
        id=f"u{idx+1}_outcomes_rate_url",
        desc="URL reference for career outcomes data",
        parent=outcomes_node,
        critical=True
    )

    outcomes_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_outcomes_rate_value",
        desc="The reported Career Outcomes Rate or Knowledge Rate percentage is provided",
        parent=outcomes_node,
        critical=True
    )

    metric_type = uni.outcomes_metric_type if uni.outcomes_metric_type else "Career Outcomes Rate or Knowledge Rate"
    metric_value = uni.outcomes_metric_value if uni.outcomes_metric_value else ""
    outcomes_claim_name = uni.university_name if uni.university_name else "the university"
    outcomes_claim = (
        f"The university reports a {metric_type} of {metric_value} for its most recent graduating class."
        if metric_value else
        f"The university reports a {metric_type} for its most recent graduating class."
    )
    await evaluator.verify(
        claim=outcomes_claim,
        node=outcomes_leaf,
        sources=outcomes_urls,
        additional_instruction=(
            "Verify the page mentions either a 'Career Outcomes Rate' or 'Knowledge Rate' and matches the claimed percentage "
            "if provided. Accept reasonable formatting (e.g., 'Career outcome rate', 'Knowledge-rate', or presence/absence of '%'). "
            "If multiple years are shown, the value for the most recent graduating class is acceptable."
        )
    )

    # -------------------- Graduating class year -------------------- #
    class_year_node = evaluator.add_parallel(
        id=f"u{idx+1}_graduating_class_year",
        desc="The graduating class year for which career outcomes data was reported",
        parent=uni_node,
        critical=True
    )

    class_year_urls = _sanitize_urls(uni.graduating_class_year_urls)
    evaluator.add_custom_node(
        result=(len(class_year_urls) > 0),
        id=f"u{idx+1}_class_year_url",
        desc="URL reference for graduating class year information",
        parent=class_year_node,
        critical=True
    )

    class_year_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_class_year_verification",
        desc="The specific graduating class year is identified",
        parent=class_year_node,
        critical=True
    )

    class_year_text = uni.graduating_class_year if uni.graduating_class_year else ""
    cy_claim = (
        f"The career outcomes data cited is for the graduating class {class_year_text}."
        if class_year_text else
        "The page identifies a specific graduating class year for the cited outcomes data."
    )
    await evaluator.verify(
        claim=cy_claim,
        node=class_year_leaf,
        sources=class_year_urls,
        additional_instruction=(
            "Confirm that the page clearly states the graduating class year associated with the outcomes data "
            "(e.g., 'Class of 2024', '2023-2024'). Allow reasonable variations like 'graduates of 2024'. "
            "If multiple years appear, ensure one is explicitly tied to the cited outcomes."
        )
    )

    # -------------------- NACE compliance -------------------- #
    nace_node = evaluator.add_parallel(
        id=f"u{idx+1}_nace_compliance",
        desc="University reports career outcomes following NACE First Destination Survey standards",
        parent=uni_node,
        critical=True
    )

    nace_urls = _sanitize_urls(uni.nace_urls)
    evaluator.add_custom_node(
        result=(len(nace_urls) > 0),
        id=f"u{idx+1}_nace_url",
        desc="URL reference for NACE compliance documentation",
        parent=nace_node,
        critical=True
    )

    nace_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_nace_methodology",
        desc="Evidence that the university uses NACE-compliant methodology for outcomes reporting",
        parent=nace_node,
        critical=True
    )

    nace_claim = (
        "The career outcomes reporting follows NACE (National Association of Colleges and Employers) First Destination Survey standards."
    )
    await evaluator.verify(
        claim=nace_claim,
        node=nace_leaf,
        sources=nace_urls,
        additional_instruction=(
            "Look for mentions of 'NACE', 'National Association of Colleges and Employers', 'First Destination Survey', "
            "or 'FDS' indicating compliance, alignment, or adherence to NACE methodology/standards."
        )
    )

    # -------------------- Undergraduate enrollment (<15,000) -------------------- #
    enroll_node = evaluator.add_parallel(
        id=f"u{idx+1}_enrollment_size",
        desc="University has fewer than 15,000 undergraduate students",
        parent=uni_node,
        critical=True
    )

    enroll_urls = _sanitize_urls(uni.enrollment_urls)
    evaluator.add_custom_node(
        result=(len(enroll_urls) > 0),
        id=f"u{idx+1}_enrollment_url",
        desc="URL reference for enrollment data",
        parent=enroll_node,
        critical=True
    )

    enroll_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_enrollment_count",
        desc="The undergraduate enrollment count is verified to be under 15,000",
        parent=enroll_node,
        critical=True
    )

    enrollment_text = uni.untergrad_enrollment if hasattr(uni, "untergrad_enrollment") else None  # safeguard typo
    if enrollment_text is None:
        enrollment_text = uni.undergrad_enrollment

    enroll_name = uni.university_name if uni.university_name else "the university"
    if enrollment_text and str(enrollment_text).strip():
        enroll_claim = f"The undergraduate enrollment at {enroll_name} is {enrollment_text}, which is fewer than 15,000."
    else:
        enroll_claim = f"The undergraduate enrollment at {enroll_name} is fewer than 15,000 students."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=enroll_urls,
        additional_instruction=(
            "Verify that the cited page states an undergraduate enrollment count below 15,000. "
            "Allow minor formatting differences (commas, 'about', '~'). "
            "If multiple figures are shown (e.g., total vs undergraduate), focus on undergraduate count."
        )
    )

    # -------------------- Operational metrics -------------------- #
    oper_node = evaluator.add_parallel(
        id=f"u{idx+1}_operational_metrics",
        desc="University provides specific operational information about career center staffing, programs, or student engagement",
        parent=uni_node,
        critical=True
    )

    oper_urls = _sanitize_urls(uni.operational_urls)
    evaluator.add_custom_node(
        result=(len(oper_urls) > 0),
        id=f"u{idx+1}_operational_url",
        desc="URL reference for operational metrics",
        parent=oper_node,
        critical=True
    )

    oper_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_operational_data",
        desc="Specific quantitative or qualitative data about career center operations is publicly available",
        parent=oper_node,
        critical=True
    )

    oper_detail = uni.operational_detail if uni.operational_detail else ""
    oper_claim = (
        f"The university's career services center reports: {oper_detail}."
        if oper_detail else
        "The page provides at least one specific operational detail about the career services center (e.g., fairs per year, staff size, engagement %, or programs)."
    )
    await evaluator.verify(
        claim=oper_claim,
        node=oper_leaf,
        sources=oper_urls,
        additional_instruction=(
            "Confirm the page contains the specific operational detail cited (e.g., number of career fairs per year, staff count, "
            "student engagement percentage, or a named program). Accept reasonable synonymy and formatting."
        )
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Keep only first 3 universities; pad with empty entries if fewer
    universities: List[UniversityItem] = list(extracted.universities or [])[:3]
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Add a small info block for transparency
    evaluator.add_custom_info(
        info={
            "extracted_universities_count": len(extracted.universities or []),
            "used_universities_count": 3,
            "allowed_pr_years": ALLOWED_PR_YEARS
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build the verification tree for 3 universities
    tasks = []
    for i in range(3):
        tasks.append(verify_university(evaluator, root, universities[i], i))
    # Run all three in parallel to speed up (each university node aggregates independently)
    await asyncio.gather(*tasks)

    return evaluator.get_summary()