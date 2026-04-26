import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "top100_superintendent_positions"
TASK_DESCRIPTION = (
    "Identify three superintendent positions in school districts that rank among the 100 largest in the United States by student enrollment. "
    "Each position must meet all of the following requirements:\n"
    "1) District ranks among the 100 largest U.S. districts by enrollment; "
    "2) Base salary at least $200,000; "
    "3) Requires a master's degree or higher; "
    "4) Requires at least 3 years of prior administrative/leadership experience in education; "
    "5) Position is currently available or was filled within the past 12 months (since March 2025). "
    "Provide for each: the district name, current enrollment and a source confirming top-100 status, "
    "the base salary, and direct URL references for: (a) enrollment/ranking, (b) salary, (c) education requirement, "
    "(d) experience requirement, and (e) posting/announcement."
)

# Timeframe for "past 12 months (since March 2025)" given current_date: 2026-03-22
TIMEFRAME_START_ISO = "2025-03-01"
TIMEFRAME_END_ISO = "2026-03-22"


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    district_name: Optional[str] = None

    enrollment_text: Optional[str] = None
    ranking_source_urls: List[str] = Field(default_factory=list)

    salary_text: Optional[str] = None
    salary_source_urls: List[str] = Field(default_factory=list)

    education_requirement_text: Optional[str] = None
    education_source_urls: List[str] = Field(default_factory=list)

    experience_requirement_text: Optional[str] = None
    experience_source_urls: List[str] = Field(default_factory=list)

    contract_term_text: Optional[str] = None
    contract_source_urls: List[str] = Field(default_factory=list)

    benefits_list: List[str] = Field(default_factory=list)
    benefits_source_urls: List[str] = Field(default_factory=list)

    position_status_text: Optional[str] = None
    position_status_urls: List[str] = Field(default_factory=list)


class SuperintendentsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract up to three superintendent positions that the answer claims satisfy the task requirements.

For each position, extract the following fields exactly as presented in the answer text (do not invent or infer):
- district_name: The full official district name (e.g., "Clark County School District")
- enrollment_text: The current or recent total student enrollment figure or phrase from the answer (e.g., "320,000 students", "approx. 50,000")
- ranking_source_urls: An array of 1–5 URLs that document the district’s enrollment or explicit ranking/listing among large U.S. districts (NCES, Niche, Ballotpedia, Wikipedia with references, or official district/agency pages). Include only URLs explicitly present in the answer.
- salary_text: The documented superintendent base salary (or salary range) text from the answer (e.g., "$250,000 base", "range $230,000–$260,000")
- salary_source_urls: 1–5 URLs that document the superintendent salary (job posting on district site, board agenda/minutes/resolution, contract PDF, or official HR salary schedule)
- education_requirement_text: The phrase indicating a master’s degree or higher is required (e.g., "Master’s degree required", "Doctorate preferred/required")
- education_source_urls: 1–5 URLs documenting the educational requirement
- experience_requirement_text: The phrase indicating at least 3 years of prior administrative/leadership experience in education is required (e.g., "minimum 3 years administrative experience")
- experience_source_urls: 1–5 URLs documenting the experience requirement
- contract_term_text: The contract term text if mentioned (e.g., "three-year contract", "initial term of 4 years"); null if not mentioned
- contract_source_urls: 0–5 URLs documenting the contract term, if any
- benefits_list: List the distinct benefits explicitly mentioned for the position (choose only from: "health insurance", "retirement/pension", "professional development", "life insurance"). Include each at most once. If fewer than three are mentioned, include what’s present. If none, return an empty list.
- benefits_source_urls: 0–5 URLs documenting the benefits package
- position_status_text: Status phrase indicating that the position is currently open/posted or was filled within the past 12 months (since March 2025) (e.g., "posting open until filled", "appointed in Oct 2025"); null if not mentioned
- position_status_urls: 0–5 URLs to the job posting, board announcement/resolution, or reputable local news documenting the posting/appointment

Rules:
- Extract only information explicitly stated in the answer. Do not infer.
- For URL fields, include only valid URLs that are explicitly present in the answer. If a URL appears without protocol, prepend http://
- If a field is not present, set it to null (for strings) or [] (for lists).
- Return an object { "positions": PositionItem[] }. Keep the original phrasing for *_text fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_n_or_pad(items: List[PositionItem], n: int) -> List[PositionItem]:
    out = list(items[:n])
    while len(out) < n:
        out.append(PositionItem())
    return out


def _nz_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


# --------------------------------------------------------------------------- #
# Verification builder per position                                           #
# --------------------------------------------------------------------------- #
async def build_and_verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int,
) -> None:
    pos_num = idx + 1
    district_name = (pos.district_name or "").strip() or f"Position #{pos_num} District (unspecified)"

    # Position container (parallel, non-critical to allow partial across 3 positions)
    position_node = evaluator.add_parallel(
        id=f"Position_{pos_num}",
        desc=f"Position #{pos_num}: qualifying superintendent position with complete verified information",
        parent=parent_node,
        critical=False,
    )

    # ------------------ District Eligibility (Critical) ------------------ #
    district_elig = evaluator.add_sequential(
        id=f"District_Eligibility_{pos_num}",
        desc="Verification that the school district ranks among the 100 largest in the United States by student enrollment",
        parent=position_node,
        critical=True,
    )

    # 1) District identification (existence check)
    evaluator.add_custom_node(
        result=bool(pos.district_name and pos.district_name.strip()),
        id=f"District_Identification_{pos_num}",
        desc="The name of the school district is provided",
        parent=district_elig,
        critical=True,
    )

    # 2) Enrollment + Ranking evidence (Parallel under eligibility)
    enr_rank = evaluator.add_parallel(
        id=f"Enrollment_Ranking_{pos_num}",
        desc="Evidence confirms the district's student enrollment and its status among top 100 U.S. districts",
        parent=district_elig,
        critical=True,
    )

    # 2.a) Enrollment data is present in the provided source(s)
    enr_leaf = evaluator.add_leaf(
        id=f"Enrollment_Data_{pos_num}",
        desc="Current or recent enrollment numbers are provided for the district",
        parent=enr_rank,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"At least one provided enrollment/ranking URL contains the current or recent total student enrollment "
            f"for {district_name} (or includes a table/list entry for the district showing enrollment)."
        ),
        node=enr_leaf,
        sources=_nz_list(pos.ranking_source_urls),
        additional_instruction=(
            "Pass if the page provides a numeric enrollment figure or a table/list row for the named district. "
            "Accept synonyms like 'student enrollment', 'students enrolled', or 'K-12 enrollment'."
        ),
    )

    # 2.b) Top-100 largest confirmation from provided source(s)
    top100_leaf = evaluator.add_leaf(
        id=f"Ranking_Source_URL_{pos_num}",
        desc="URL reference to a source documenting the district's ranking or enrollment size (such as NCES, Niche, Ballotpedia, or official district data)",
        parent=enr_rank,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"At least one provided enrollment/ranking URL explicitly shows that {district_name} is among the 100 largest "
            "U.S. school districts by student enrollment, e.g., appears on a 'largest school districts' list or states a U.S. rank ≤ 100."
        ),
        node=top100_leaf,
        sources=_nz_list(pos.ranking_source_urls),
        additional_instruction=(
            "Rely on the page content. Accept if the page is a credible/official list or otherwise explicitly states the "
            "district is within the top 100 U.S. districts by enrollment. Do not rely on your own memory."
        ),
    )

    # ------------------ Compensation Requirements (Critical) ------------- #
    comp_req = evaluator.add_sequential(
        id=f"Compensation_Requirements_{pos_num}",
        desc="Verification that the superintendent position meets the minimum salary threshold of $200,000 annually",
        parent=position_node,
        critical=True,
    )

    # Base salary amount >= $200,000
    salary_amount_leaf = evaluator.add_leaf(
        id=f"Base_Salary_Amount_{pos_num}",
        desc="The documented base annual salary for the position is stated and meets or exceeds $200,000",
        parent=comp_req,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The superintendent position's base salary is at least $200,000 per year. "
            "If a range is shown, the minimum of the range must be ≥ $200,000."
        ),
        node=salary_amount_leaf,
        sources=_nz_list(pos.salary_source_urls),
        additional_instruction=(
            "Only count base or minimum base salary (exclude total compensation with allowances/stipends unless explicitly part of base). "
            "Accept official job postings, board docs/resolutions/minutes, contract PDFs, or official salary schedules."
        ),
    )

    # Salary documentation (Parallel, critical)
    salary_docs = evaluator.add_parallel(
        id=f"Salary_Documentation_{pos_num}",
        desc="The salary information is documented through official sources",
        parent=comp_req,
        critical=True,
    )

    # Source explicitly contains the superintendent salary
    salary_src_contains_leaf = evaluator.add_leaf(
        id=f"Salary_Source_Provided_{pos_num}",
        desc="A specific source for the salary information is identified (job posting, board documents, contract, or official salary schedule)",
        parent=salary_docs,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"At least one provided salary URL explicitly states the superintendent base salary (or a salary range) for {district_name}."
        ),
        node=salary_src_contains_leaf,
        sources=_nz_list(pos.salary_source_urls),
        additional_instruction="Pass only if the page itself contains the salary figure(s) or range for the superintendent role.",
    )

    # Source is official/authoritative in nature
    salary_src_url_leaf = evaluator.add_leaf(
        id=f"Salary_Source_URL_{pos_num}",
        desc="URL reference to the source documenting the superintendent's salary",
        parent=salary_docs,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one provided salary URL is an official or authoritative source, such as a district website page, "
            "board agenda/minutes/resolution, contract PDF, or official HR salary schedule."
        ),
        node=salary_src_url_leaf,
        sources=_nz_list(pos.salary_source_urls),
        additional_instruction=(
            "Prefer .k12, .gov, district domains, or official board packet/document links. "
            "Avoid generic third-party summaries without source documents. Pass if at least one URL is clearly official."
        ),
    )

    # ------------------ Qualification Requirements (Critical) ------------ #
    qual_req = evaluator.add_parallel(
        id=f"Qualification_Requirements_{pos_num}",
        desc="Verification that the position requires both a master's degree and at least 3 years of administrative experience",
        parent=position_node,
        critical=True,
    )

    # Education requirement (Sequential, critical)
    edu_req = evaluator.add_sequential(
        id=f"Education_Requirement_{pos_num}",
        desc="Documentation confirms the position requires a master's degree or higher from an accredited institution",
        parent=qual_req,
        critical=True,
    )

    edu_stated_leaf = evaluator.add_leaf(
        id=f"Education_Stated_{pos_num}",
        desc="The master's degree requirement is explicitly stated in the job requirements or qualifications",
        parent=edu_req,
        critical=True,
    )
    await evaluator.verify(
        claim="The position explicitly requires a master's degree or higher from an accredited institution.",
        node=edu_stated_leaf,
        sources=_nz_list(pos.education_source_urls),
        additional_instruction=(
            "Pass if the page clearly states a master's degree is required; a doctorate preferred/required is also acceptable. "
            "Look for 'Master's degree required', 'Master’s from an accredited institution', or similar."
        ),
    )

    edu_src_leaf = evaluator.add_leaf(
        id=f"Education_Source_URL_{pos_num}",
        desc="URL reference to the source documenting the educational requirement (job posting, district policy, or official announcement)",
        parent=edu_req,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one provided URL documents the master's-or-higher educational requirement for the superintendent position.",
        node=edu_src_leaf,
        sources=_nz_list(pos.education_source_urls),
        additional_instruction="Pass if the URL content shows the education requirement for the superintendent role.",
    )

    # Experience requirement (Sequential, critical)
    exp_req = evaluator.add_sequential(
        id=f"Experience_Requirement_{pos_num}",
        desc="Documentation confirms the position requires at least 3 years of prior administrative or leadership experience in education",
        parent=qual_req,
        critical=True,
    )

    exp_stated_leaf = evaluator.add_leaf(
        id=f"Experience_Stated_{pos_num}",
        desc="The administrative experience requirement of at least 3 years is explicitly stated in the job requirements or qualifications",
        parent=exp_req,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The position explicitly requires at least 3 years of prior administrative or leadership experience in education "
            "(e.g., principal, assistant principal, or district-level administrator)."
        ),
        node=exp_stated_leaf,
        sources=_nz_list(pos.experience_source_urls),
        additional_instruction=(
            "Accept phrasing like 'minimum of three (3) years administrative experience' or 'at least 3 years leadership experience in K-12 education'."
        ),
    )

    exp_src_leaf = evaluator.add_leaf(
        id=f"Experience_Source_URL_{pos_num}",
        desc="URL reference to the source documenting the experience requirement (job posting, district policy, or official announcement)",
        parent=exp_req,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one provided URL documents the 3+ years administrative/leadership experience requirement for the superintendent role.",
        node=exp_src_leaf,
        sources=_nz_list(pos.experience_source_urls),
        additional_instruction="Pass if the URL content shows the explicit '3 years' (or more) administrative/leadership experience requirement.",
    )

    # ------------------ Contract Term (Non-Critical) --------------------- #
    contract_node = evaluator.add_sequential(
        id=f"Contract_Term_{pos_num}",
        desc="Verification that the superintendent employment contract specifies a multi-year term (typically 3-5 years)",
        parent=position_node,
        critical=False,
    )

    contract_len_leaf = evaluator.add_leaf(
        id=f"Contract_Duration_Stated_{pos_num}",
        desc="The contract term length is documented and indicates a multi-year commitment",
        parent=contract_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The superintendent employment contract specifies a multi-year term (typically 3–5 years).",
        node=contract_len_leaf,
        sources=_nz_list(pos.contract_source_urls),
        additional_instruction=(
            "Pass for language like 'three-year contract', 'initial term of 4 years', or similar. "
            "Sources can be contract PDFs, board resolutions, or official announcements."
        ),
    )

    contract_src_leaf = evaluator.add_leaf(
        id=f"Contract_Source_URL_{pos_num}",
        desc="URL reference to the source documenting the contract term (contract document, board resolution, or official announcement)",
        parent=contract_node,
        critical=False,
    )
    await evaluator.verify(
        claim="At least one provided URL documents the superintendent contract term length.",
        node=contract_src_leaf,
        sources=_nz_list(pos.contract_source_urls),
        additional_instruction="Pass if the URL content includes the multi-year term detail.",
    )

    # ------------------ Benefits Package (Non-Critical) ------------------ #
    benefits_node = evaluator.add_sequential(
        id=f"Benefits_Package_{pos_num}",
        desc="Verification that the position includes a documented benefits package containing at least three of the specified categories",
        parent=position_node,
        critical=False,
    )

    benefits_listed_leaf = evaluator.add_leaf(
        id=f"Benefits_Listed_{pos_num}",
        desc="At least three benefit components from the specified categories are documented for the position",
        parent=benefits_node,
        critical=False,
    )
    await evaluator.verify(
        claim=(
            "The provided URL(s) document at least three of the following superintendent benefits: "
            "health insurance; retirement/pension contributions; professional development funding; life insurance."
        ),
        node=benefits_listed_leaf,
        sources=_nz_list(pos.benefits_source_urls),
        additional_instruction=(
            "Look for explicit mentions of at least three of the four categories. "
            "Accept official contract/benefits overview/HR documentation showing these items."
        ),
    )

    benefits_src_leaf = evaluator.add_leaf(
        id=f"Benefits_Source_URL_{pos_num}",
        desc="URL reference to the source documenting the benefits package (contract, district benefits overview, or official documentation)",
        parent=benefits_node,
        critical=False,
    )
    await evaluator.verify(
        claim="At least one provided URL documents the superintendent benefits package.",
        node=benefits_src_leaf,
        sources=_nz_list(pos.benefits_source_urls),
        additional_instruction="Pass if the URL content clearly describes the benefits.",
    )

    # ------------------ Position Status (Critical per task requirement) --- #
    status_node = evaluator.add_sequential(
        id=f"Position_Status_{pos_num}",
        desc="Verification of position availability or recent filling status",
        parent=position_node,
        critical=True,  # Elevated to critical to match task requirement
    )

    status_time_leaf = evaluator.add_leaf(
        id=f"Availability_Timeframe_{pos_num}",
        desc="The position is documented as currently available or filled within the past 12 months",
        parent=status_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The position is either currently available for application (open posting), "
            f"or was filled within the last 12 months between {TIMEFRAME_START_ISO} and {TIMEFRAME_END_ISO}."
        ),
        node=status_time_leaf,
        sources=_nz_list(pos.position_status_urls),
        additional_instruction=(
            "Pass if the job posting is open/active (e.g., 'open until filled'), or if an official board announcement/news reports "
            "the appointment within the timeframe. Use the page content/screenshot provided."
        ),
    )

    status_src_leaf = evaluator.add_leaf(
        id=f"Position_Source_URL_{pos_num}",
        desc="URL reference to the job posting, board announcement, or news article documenting the position",
        parent=status_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one provided URL is a job posting, official board announcement/resolution, or a reputable local news article about the appointment.",
        node=status_src_leaf,
        sources=_nz_list(pos.position_status_urls),
        additional_instruction="Pass if the URL content is clearly about the superintendent posting or appointment.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Note: We set root as non-critical to allow partial credit across the three positions
    root.critical = False

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=SuperintendentsExtraction,
        extraction_name="positions_extraction",
    )

    # Keep only first 3 positions; pad if fewer
    positions = _first_n_or_pad(extracted.positions if extracted and extracted.positions else [], 3)

    # Record timeframe used for judging Position Status
    evaluator.add_custom_info(
        info={
            "timeframe_start": TIMEFRAME_START_ISO,
            "timeframe_end": TIMEFRAME_END_ISO,
            "note": "Positions must be currently available or filled within this window."
        },
        info_type="timeframe",
        info_name="position_status_timeframe"
    )

    # Build and verify for each of three positions
    for i, pos in enumerate(positions):
        await build_and_verify_position(evaluator, root, pos, i)

    return evaluator.get_summary()