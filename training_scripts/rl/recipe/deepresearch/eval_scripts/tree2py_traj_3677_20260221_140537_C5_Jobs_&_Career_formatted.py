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
TASK_ID = "uva_10th_president_outcomes_2025"
TASK_DESCRIPTION = (
    "The University of Virginia appointed its 10th president in 2025. Before assuming this role, this individual "
    "served as dean of a business school at the same university. Your task is to: "
    "(1) Identify this university president and the business school where they previously served as dean; "
    "(2) Locate the most recent career outcomes or first destination report published by that business school; "
    "(3) Extract the following information from that report: the overall career outcomes rate or employment rate percentage, "
    "the graduating class year covered by the report, whether the report follows NACE (National Association of Colleges and Employers) standards, "
    "the knowledge rate or response rate percentage, and the total number of graduates in the cohort. Provide the specific metrics with supporting reference URLs. "
    "Verify whether the report mentions NACE compliance, the measurement timeline (such as six-month post-graduation), and whether outcomes are reported separately for different degree levels."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LeaderInfo(BaseModel):
    president_name: Optional[str] = None
    appointment_url: Optional[str] = None  # UVA announcement or similar
    previous_role_title: Optional[str] = None  # e.g., Dean
    business_school_name: Optional[str] = None  # e.g., Darden School of Business or McIntire School of Commerce
    previous_institution_url: Optional[str] = None  # Page confirming the dean role


class ReportInfo(BaseModel):
    report_url: Optional[str] = None  # Most recent career outcomes/first destination report page
    report_class_year: Optional[str] = None  # e.g., Class of 2024
    metrics_url: Optional[str] = None  # If metrics are on another dedicated page; otherwise same as report_url
    overall_outcomes_rate: Optional[str] = None  # e.g., "97%" or "97 percent"
    knowledge_rate: Optional[str] = None  # e.g., "92%"
    response_rate: Optional[str] = None  # e.g., "80%"
    total_graduates: Optional[str] = None  # number or string, e.g., "487"
    nace_compliance_mentioned: Optional[str] = None  # "yes"/"no"/"unsure" as claimed
    six_month_timeline_mentioned: Optional[str] = None  # "yes"/"no"/"unsure"
    bachelors_reported: Optional[str] = None  # "yes"/"no"/"unsure"
    masters_reported: Optional[str] = None  # "yes"/"no"/"unsure"
    school_college_breakdown: Optional[str] = None  # "yes"/"no"/"unsure"
    additional_metric_urls: List[str] = Field(default_factory=list)  # Any other URLs cited for metrics


class UVAOutcomesExtraction(BaseModel):
    leader: Optional[LeaderInfo] = None
    report: Optional[ReportInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract the requested information from the provided answer. Return a JSON with two top-level objects: "
        "'leader' and 'report'. Follow these field definitions exactly:\n"
        "leader:\n"
        "  - president_name: the full name of the person appointed as the University of Virginia's 10th president in 2025\n"
        "  - appointment_url: the URL cited that confirms this appointment (UVA announcement or similar); if absent in the answer, set null\n"
        "  - previous_role_title: the previous administrative role title (e.g., 'Dean')\n"
        "  - business_school_name: the name of the UVA business school where the person served as dean\n"
        "  - previous_institution_url: a URL cited that confirms the dean role at that business school (UVA page or authoritative source)\n"
        "\n"
        "report:\n"
        "  - report_url: URL to the most recent career outcomes or first destination report page published by the identified business school\n"
        "  - report_class_year: the graduating class year covered (e.g., 'Class of 2024' or '2023-2024'); if unclear in the answer, set null\n"
        "  - metrics_url: if metrics are on another page, provide that URL; otherwise set equal to report_url or null if not provided\n"
        "  - overall_outcomes_rate: the overall career outcomes rate or employment rate percentage (string, include % if present)\n"
        "  - knowledge_rate: the knowledge rate percentage (string) if provided; else null\n"
        "  - response_rate: the response rate percentage (string) if provided; else null\n"
        "  - total_graduates: the total number of graduates in the cohort as a string; if missing, set null\n"
        "  - nace_compliance_mentioned: 'yes' if the answer claims the report follows NACE standards; 'no' if it claims it does not; 'unsure' if not stated\n"
        "  - six_month_timeline_mentioned: 'yes' if the answer claims outcomes measured within six months post-graduation; 'no' if it claims otherwise; 'unsure' if not stated\n"
        "  - bachelors_reported: 'yes' if the answer claims bachelor's outcomes are reported separately; 'no' if not; 'unsure' if not stated\n"
        "  - masters_reported: 'yes' if the answer claims master's outcomes are reported separately; 'no' if not; 'unsure' if not stated\n"
        "  - school_college_breakdown: 'yes' if the answer claims outcomes broken down by school/college/program; 'no' if not; 'unsure' if not stated\n"
        "  - additional_metric_urls: list of any other URLs cited for metrics; if none, return empty list\n"
        "\n"
        "General rules:\n"
        "- Extract exactly what appears in the answer. Do not invent new information.\n"
        "- For URLs, extract the actual valid URLs that appear (plain URL or markdown link). If a URL is missing protocol, prepend 'http://'.\n"
        "- If a field is not mentioned in the answer, set it to null (or empty list for arrays).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(v: Optional[str]) -> str:
    return v or ""

def _non_empty_url(u: Optional[str]) -> Optional[str]:
    if u and isinstance(u, str) and u.strip():
        return u.strip()
    return None

def _unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if _non_empty_url(u) and _non_empty_url(u) not in seen:
            out.append(_non_empty_url(u))  # type: ignore
            seen.add(_non_empty_url(u))  # type: ignore
    return out

def _metric_sources(report: Optional[ReportInfo]) -> List[str]:
    if not report:
        return []
    base = []
    base.append(_non_empty_url(report.report_url))
    base.append(_non_empty_url(report.metrics_url))
    # add any additional URLs
    for u in (report.additional_metric_urls or []):
        base.append(_non_empty_url(u))
    return _unique_nonempty(base)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_leader_identification(evaluator: Evaluator, parent_node, data: UVAOutcomesExtraction) -> None:
    leader = data.leader or LeaderInfo()

    node = evaluator.add_parallel(
        id="leader_identification",
        desc="Correctly identify the university leader and their previous institution",
        parent=parent_node,
        critical=True
    )

    # Leaf: leader_name
    ln = evaluator.add_leaf(
        id="leader_name",
        desc="Provide the correct name of the current university president",
        parent=node,
        critical=True
    )
    claim_leader = f"The person appointed as the University of Virginia's 10th president in 2025 is {_safe_str(leader.president_name)}."
    leader_sources = _non_empty_url(leader.appointment_url) or _non_empty_url(leader.previous_institution_url)
    await evaluator.verify(
        claim=claim_leader,
        node=ln,
        sources=leader_sources,
        additional_instruction=(
            "Use the cited page to confirm the individual's identity and appointment. "
            "If the page is an official UVA announcement, it should explicitly mention the appointment and numbering (10th president) in 2025. "
            "Allow minor title wording variations."
        ),
    )

    # Leaf: previous_role
    pr = evaluator.add_leaf(
        id="previous_role",
        desc="Identify the leader's previous administrative role at the same or different institution",
        parent=node,
        critical=True
    )
    claim_prev_role = (
        f"Before assuming the presidency, {_safe_str(leader.president_name)} served as "
        f"{_safe_str(leader.previous_role_title)} of {_safe_str(leader.business_school_name)} at the University of Virginia."
    )
    await evaluator.verify(
        claim=claim_prev_role,
        node=pr,
        sources=_non_empty_url(leader.previous_institution_url),
        additional_instruction=(
            "Verify that the cited page confirms the person held the dean role at the named business school. "
            "The page should be authoritative (official school page or UVA domain) and clearly state the role."
        ),
    )

    # Leaf: previous_institution
    pi = evaluator.add_leaf(
        id="previous_institution",
        desc="Identify the institution where the leader held the previous role",
        parent=node,
        critical=True
    )
    claim_prev_inst = (
        f"The previous institution where {_safe_str(leader.president_name)} served as dean was the University of Virginia's "
        f"{_safe_str(leader.business_school_name)}."
    )
    await evaluator.verify(
        claim=claim_prev_inst,
        node=pi,
        sources=_non_empty_url(leader.previous_institution_url),
        additional_instruction=(
            "Confirm that the page indicates the business school is part of the University of Virginia and associates the person with that school."
        ),
    )

    # Leaf: previous_institution_url
    purl = evaluator.add_leaf(
        id="previous_institution_url",
        desc="Provide a reference URL confirming the leader's previous institutional affiliation",
        parent=node,
        critical=True
    )
    claim_prev_url = (
        f"This page confirms that {_safe_str(leader.president_name)} served as {_safe_str(leader.previous_role_title)} of "
        f"{_safe_str(leader.business_school_name)} at the University of Virginia."
    )
    await evaluator.verify(
        claim=claim_prev_url,
        node=purl,
        sources=_non_empty_url(leader.previous_institution_url),
        additional_instruction=(
            "Check the page content to ensure it is related to the UVA business school and explicitly ties the individual to the dean role."
        ),
    )


async def verify_report_access(evaluator: Evaluator, parent_node, data: UVAOutcomesExtraction) -> None:
    report = data.report or ReportInfo()
    school_name = data.leader.business_school_name if data.leader else None

    node = evaluator.add_parallel(
        id="career_outcomes_report_access",
        desc="Locate and access the career outcomes or first destination report from the identified institution",
        parent=parent_node,
        critical=True
    )

    # Custom existence check: report_url provided
    report_exists_custom = evaluator.add_custom_node(
        result=bool(_non_empty_url(report.report_url)),
        id="report_exists",
        desc="Confirm that the institution publishes career outcomes or first destination data",
        parent=node,
        critical=True
    )

    # Leaf: report_timeframe (graduating class year(s))
    rtf = evaluator.add_leaf(
        id="report_timeframe",
        desc="Identify the graduating class year(s) covered in the available report",
        parent=node,
        critical=True
    )
    claim_timeframe = f"The report covers the graduating class year(s) {_safe_str(report.report_class_year)}."
    await evaluator.verify(
        claim=claim_timeframe,
        node=rtf,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Confirm the class year(s) on the report page (e.g., 'Class of 2024', '2023-2024'). Allow reasonable variations and exact wording differences."
        ),
    )

    # Leaf: report_url correctness (is the report page)
    rurl = evaluator.add_leaf(
        id="report_url",
        desc="Provide the URL where the career outcomes report can be accessed",
        parent=node,
        critical=True
    )
    claim_report_page = (
        f"This page is the career outcomes or first destination report for {_safe_str(school_name)} at the University of Virginia."
    )
    await evaluator.verify(
        claim=claim_report_page,
        node=rurl,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Verify that the page is an official or authoritative report page containing career outcomes/first destination information "
            "for the identified business school."
        ),
    )


async def verify_nace_standards(evaluator: Evaluator, parent_node, data: UVAOutcomesExtraction) -> None:
    report = data.report or ReportInfo()
    node = evaluator.add_parallel(
        id="nace_standards_verification",
        desc="Verify that the report follows NACE standards for career outcomes reporting",
        parent=parent_node,
        critical=False
    )

    # Leaf: knowledge_rate_mentioned
    krm = evaluator.add_leaf(
        id="knowledge_rate_mentioned",
        desc="Confirm whether the report mentions or calculates Knowledge Rate",
        parent=node,
        critical=False
    )
    claim_krm = "The report page mentions or calculates the Knowledge Rate."
    await evaluator.verify(
        claim=claim_krm,
        node=krm,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Look for 'Knowledge Rate' specifically; synonyms are rare, but accept phrasing like 'knowledge rate %' or 'knowledge rate percentage'."
        ),
    )

    # Leaf: career_outcomes_rate_mentioned
    corm = evaluator.add_leaf(
        id="career_outcomes_rate_mentioned",
        desc="Confirm whether the report mentions or calculates Career Outcomes Rate",
        parent=node,
        critical=False
    )
    claim_corm = "The report page mentions the Career Outcomes Rate, Employment Rate, or Placement Rate."
    await evaluator.verify(
        claim=claim_corm,
        node=corm,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Accept equivalent terms like 'employment rate', 'career outcome rate', 'placement rate', or 'overall outcomes rate'."
        ),
    )

    # Leaf: six_month_timeline
    sixm = evaluator.add_leaf(
        id="six_month_timeline",
        desc="Verify that outcomes are measured within 6 months of graduation",
        parent=node,
        critical=False
    )
    claim_sixm = "The report states that outcomes are measured within six months after graduation (e.g., six months post-graduation)."
    await evaluator.verify(
        claim=claim_sixm,
        node=sixm,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Check for language about timing windows like 'six months after graduation' or similar wording about the measurement timeline."
        ),
    )

    # Leaf: nace_compliance_stated
    nace = evaluator.add_leaf(
        id="nace_compliance_stated",
        desc="Check if the report explicitly states compliance with NACE standards",
        parent=node,
        critical=False
    )
    claim_nace = "The report explicitly states that it follows or complies with NACE standards or guidelines."
    await evaluator.verify(
        claim=claim_nace,
        node=nace,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Look for explicit statements of 'NACE standards', 'NACE compliant', 'per NACE guidelines', or similar wording."
        ),
    )


async def verify_specific_metrics(evaluator: Evaluator, parent_node, data: UVAOutcomesExtraction) -> None:
    report = data.report or ReportInfo()
    metric_sources = _metric_sources(report)

    node = evaluator.add_parallel(
        id="specific_metrics_extraction",
        desc="Extract specific career outcomes metrics from the report",
        parent=parent_node,
        critical=True
    )

    # Leaf: overall_outcomes_rate (employment/career outcomes rate)
    oor = evaluator.add_leaf(
        id="overall_outcomes_rate",
        desc="Provide the overall career outcomes rate or employment rate percentage",
        parent=node,
        critical=True
    )
    claim_oor = f"The overall career outcomes or employment rate reported is {_safe_str(report.overall_outcomes_rate)}."
    await evaluator.verify(
        claim=claim_oor,
        node=oor,
        sources=metric_sources,
        additional_instruction=(
            "Verify the percentage for overall employment/career outcomes. Accept reasonable rounding (e.g., 96.7% ≈ 97%). "
            "If multiple metrics exist, prefer the headline overall outcomes rate."
        ),
    )

    # Leaf: class year covered (explicit metric confirmation)
    cy = evaluator.add_leaf(
        id="metrics_class_year",
        desc="Confirm the graduating class year covered by the report",
        parent=node,
        critical=True
    )
    claim_cy = f"The report covers the graduating class year {_safe_str(report.report_class_year)}."
    await evaluator.verify(
        claim=claim_cy,
        node=cy,
        sources=metric_sources,
        additional_instruction=(
            "Confirm the class year(s) indicated on the report or metrics page (e.g., 'Class of 2024'). Allow equivalent phrasing."
        ),
    )

    # Custom leaf: at least one of knowledge_rate or response_rate must be provided
    kor_exist = evaluator.add_custom_node(
        result=bool(_safe_str(report.knowledge_rate)) or bool(_safe_str(report.response_rate)),
        id="knowledge_or_response_exists",
        desc="Provide either the knowledge rate percentage or the response rate percentage (existence check)",
        parent=node,
        critical=True
    )

    # Leaf: knowledge_or_response_rate value (choose available)
    kor_val = evaluator.add_leaf(
        id="knowledge_or_response_rate",
        desc="Provide either the knowledge rate percentage or the response rate percentage",
        parent=node,
        critical=True
    )
    selected_label = "Knowledge Rate" if _safe_str(report.knowledge_rate) else "Response Rate"
    selected_value = _safe_str(report.knowledge_rate) if _safe_str(report.knowledge_rate) else _safe_str(report.response_rate)
    claim_kor = f"The report shows a {selected_label} of {selected_value}."
    await evaluator.verify(
        claim=claim_kor,
        node=kor_val,
        sources=metric_sources,
        additional_instruction=(
            "Verify the exact percentage for Knowledge Rate or Response Rate. Accept minor rounding and reasonable formatting variations."
        ),
    )

    # Leaf: total_graduates
    tg = evaluator.add_leaf(
        id="total_graduates",
        desc="Provide the total number of graduates in the reported cohort",
        parent=node,
        critical=True
    )
    claim_tg = f"The total number of graduates in the cohort reported is {_safe_str(report.total_graduates)}."
    await evaluator.verify(
        claim=claim_tg,
        node=tg,
        sources=metric_sources,
        additional_instruction=(
            "Verify the cohort size or number of graduates. Accept reasonable numeric formatting variations."
        ),
    )

    # Leaf: metrics_url (page contains metrics)
    mu = evaluator.add_leaf(
        id="metrics_url",
        desc="Provide URL reference for the extracted metrics",
        parent=node,
        critical=True
    )
    claim_mu = (
        "This page contains the business school's career outcomes metrics for the specified cohort (e.g., employment rate, knowledge rate, total graduates)."
    )
    metrics_page = _non_empty_url(report.metrics_url) or _non_empty_url(report.report_url)
    await evaluator.verify(
        claim=claim_mu,
        node=mu,
        sources=metrics_page,
        additional_instruction=(
            "Confirm that the page includes key metrics such as employment/career outcomes rate, knowledge/response rate, and cohort size."
        ),
    )


async def verify_degree_breakdown(evaluator: Evaluator, parent_node, data: UVAOutcomesExtraction) -> None:
    report = data.report or ReportInfo()
    node = evaluator.add_parallel(
        id="degree_level_breakdown",
        desc="Identify if career outcomes are reported separately by degree level",
        parent=parent_node,
        critical=False
    )

    # Leaf: bachelors_reported
    br = evaluator.add_leaf(
        id="bachelors_reported",
        desc="Confirm whether bachelor's degree outcomes are reported separately",
        parent=node,
        critical=False
    )
    claim_br = "The report includes separate outcomes for bachelor's (undergraduate) degree students."
    await evaluator.verify(
        claim=claim_br,
        node=br,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Look for sections labeled 'Undergraduate', 'Bachelor's', 'B.S.', 'BCom', or similar indicating separate reporting."
        ),
    )

    # Leaf: masters_reported
    mr = evaluator.add_leaf(
        id="masters_reported",
        desc="Confirm whether master's degree outcomes are reported separately",
        parent=node,
        critical=False
    )
    claim_mr = "The report includes separate outcomes for master's (graduate) degree students."
    await evaluator.verify(
        claim=claim_mr,
        node=mr,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Look for sections labeled 'Graduate', 'Master's', 'MBA', 'MS', or similar indicating separate reporting."
        ),
    )

    # Leaf: school_college_breakdown
    scb = evaluator.add_leaf(
        id="school_college_breakdown",
        desc="Identify if outcomes are broken down by school or college within the university",
        parent=node,
        critical=False
    )
    claim_scb = "The report breaks down outcomes by school, college, program, or major within the university/business school."
    await evaluator.verify(
        claim=claim_scb,
        node=scb,
        sources=_non_empty_url(report.report_url),
        additional_instruction=(
            "Check whether the report provides outcomes segmented by program/major (e.g., Commerce, MBA, MS) or by school/college."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for UVA president identification and business school career outcomes metrics.
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

    # Extract all necessary information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=UVAOutcomesExtraction,
        extraction_name="uva_outcomes_extraction",
    )

    # Build verification tree following rubric structure
    # 1) Leader identification
    await verify_leader_identification(evaluator, root, extracted)

    # 2) Report access
    await verify_report_access(evaluator, root, extracted)

    # 3) NACE standards verification
    await verify_nace_standards(evaluator, root, extracted)

    # 4) Specific metrics extraction
    await verify_specific_metrics(evaluator, root, extracted)

    # 5) Degree level breakdown
    await verify_degree_breakdown(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()