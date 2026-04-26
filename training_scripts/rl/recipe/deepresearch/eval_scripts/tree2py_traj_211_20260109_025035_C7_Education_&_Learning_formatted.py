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
TASK_ID = "online_mba_flex_afford_top_tier"
TASK_DESCRIPTION = (
    "A working professional with 6 years of experience is seeking an online MBA program that offers maximum "
    "flexibility and affordability while maintaining top-tier quality. They are looking for a program that meets ALL "
    "of the following requirements:\n\n"
    "1. Must be ranked in the top 10 of the US News & World Report Best Online MBA Programs 2025 rankings\n"
    "2. Total program cost (tuition and fees) must be under $70,000\n"
    "3. Must offer a completion pathway of 24 months or less\n"
    "4. Must accept applications on a rolling admissions basis (not limited to fixed deadlines only)\n"
    "5. Must waive GMAT/GRE test requirements for applicants with 5 or more years of professional work experience\n"
    "6. Must require at least 2 years of full-time professional work experience for admission\n"
    "7. Must hold AACSB (Association to Advance Collegiate Schools of Business) accreditation\n"
    "8. Must not require mandatory on-campus attendance (optional in-person experiences are acceptable)\n"
    "9. Must offer at least 5 different concentration or specialization options\n"
    "10. Must offer at least 3 start dates per year\n"
    "11. Must provide primarily asynchronous online instruction (allowing students to access lectures on their own schedule)\n"
    "12. Must require between 45 and 65 credit hours to complete the degree\n"
    "13. Must provide dedicated career coaching or career services specifically for online MBA students\n"
    "14. Must be taught primarily by full-time faculty members (not primarily adjunct instructors)\n\n"
    "Identify one online MBA program from a U.S. business school that satisfies all of these criteria. Provide the name "
    "of the university and business school, and for each criterion, provide the specific evidence (with reference URL) "
    "demonstrating how the program meets that requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CriterionData(BaseModel):
    """
    Per-criterion extraction data:
    - claim: Restated claim text from the answer for the specific requirement.
    - urls: Evidence URLs explicitly provided in the answer to support the claim.
    """
    claim: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    """
    Full extraction for the selected program and criterion-by-criterion evidence.
    """
    university_name: Optional[str] = None
    business_school_name: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None

    # 14 criteria, each with claim + urls
    c1_top10: CriterionData = Field(default_factory=CriterionData)
    c2_cost_under_70k: CriterionData = Field(default_factory=CriterionData)
    c3_duration_24_months_or_less: CriterionData = Field(default_factory=CriterionData)
    c4_rolling_admissions: CriterionData = Field(default_factory=CriterionData)
    c5_gmat_gre_waiver_5plus_years: CriterionData = Field(default_factory=CriterionData)
    c6_min_2yrs_work_experience: CriterionData = Field(default_factory=CriterionData)
    c7_aacsb_accreditation: CriterionData = Field(default_factory=CriterionData)
    c8_no_mandatory_on_campus: CriterionData = Field(default_factory=CriterionData)
    c9_at_least_5_specializations: CriterionData = Field(default_factory=CriterionData)
    c10_at_least_3_start_dates: CriterionData = Field(default_factory=CriterionData)
    c11_primarily_asynchronous: CriterionData = Field(default_factory=CriterionData)
    c12_credits_between_45_and_65: CriterionData = Field(default_factory=CriterionData)
    c13_dedicated_career_services_online_mba: CriterionData = Field(default_factory=CriterionData)
    c14_primarily_full_time_faculty: CriterionData = Field(default_factory=CriterionData)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_and_evidence() -> str:
    return """
    From the answer, extract a single selected online MBA program and criterion-by-criterion evidence.

    You must return a JSON object matching the following schema:

    {
      "university_name": string or null,
      "business_school_name": string or null,
      "program_name": string or null,
      "program_url": string or null,

      "c1_top10": {"claim": string or null, "urls": [url, ...]},
      "c2_cost_under_70k": {"claim": string or null, "urls": [url, ...]},
      "c3_duration_24_months_or_less": {"claim": string or null, "urls": [url, ...]},
      "c4_rolling_admissions": {"claim": string or null, "urls": [url, ...]},
      "c5_gmat_gre_waiver_5plus_years": {"claim": string or null, "urls": [url, ...]},
      "c6_min_2yrs_work_experience": {"claim": string or null, "urls": [url, ...]},
      "c7_aacsb_accreditation": {"claim": string or null, "urls": [url, ...]},
      "c8_no_mandatory_on_campus": {"claim": string or null, "urls": [url, ...]},
      "c9_at_least_5_specializations": {"claim": string or null, "urls": [url, ...]},
      "c10_at_least_3_start_dates": {"claim": string or null, "urls": [url, ...]},
      "c11_primarily_asynchronous": {"claim": string or null, "urls": [url, ...]},
      "c12_credits_between_45_and_65": {"claim": string or null, "urls": [url, ...]},
      "c13_dedicated_career_services_online_mba": {"claim": string or null, "urls": [url, ...]},
      "c14_primarily_full_time_faculty": {"claim": string or null, "urls": [url, ...]}
    }

    Extraction rules:
    - Do NOT invent any information. Only extract what is explicitly stated in the answer.
    - For 'urls': extract only actual URLs explicitly mentioned in the answer (plain URLs or markdown links). If no URL is provided, return an empty array.
    - For each criterion 'claim': restate the criterion in your own words based on what the answer says (e.g., "Program is ranked within the top 10 in US News 2025", "Total cost is $XX and under $70,000", "Completion pathway can be <= 24 months", "rolling admissions are accepted", etc.). If the answer does not make the claim, return null.
    - 'program_url' should be the official program page URL if provided; otherwise null.
    - If any field is missing in the answer, return null (or empty list for urls) for that field exactly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _program_label(extracted: ProgramExtraction) -> str:
    pn = (extracted.program_name or "").strip()
    bs = (extracted.business_school_name or "").strip()
    univ = (extracted.university_name or "").strip()
    base = pn if pn else "the selected online MBA program"
    if bs and univ:
        return f"{base} at {bs}, {univ}"
    if univ:
        return f"{base} at {univ}"
    if bs:
        return f"{base} at {bs}"
    return base


def _fallback_claim_top10(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} is ranked within the top 10 in the US News & World Report Best Online MBA Programs 2025."


def _fallback_claim_cost(extracted: ProgramExtraction) -> str:
    return f"The total program cost (tuition and fees) for {_program_label(extracted)} is under $70,000."


def _fallback_claim_duration(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} offers a completion pathway of 24 months or less."


def _fallback_claim_rolling(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} accepts applications on a rolling admissions basis."


def _fallback_claim_waiver(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} waives GMAT/GRE requirements for applicants with 5 or more years of professional work experience."


def _fallback_claim_work_exp(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} requires at least 2 years of full-time professional work experience for admission."


def _fallback_claim_aacsb(extracted: ProgramExtraction) -> str:
    return f"The business school for {_program_label(extracted)} holds AACSB accreditation."


def _fallback_claim_no_residency(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} does not require mandatory on-campus attendance; any in-person experiences are optional."


def _fallback_claim_specializations(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} offers at least 5 different concentration or specialization options."


def _fallback_claim_start_dates(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} offers at least 3 start dates per year."


def _fallback_claim_asynchronous(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} provides primarily asynchronous online instruction."


def _fallback_claim_credit_hours(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} requires between 45 and 65 credit hours to complete the degree."


def _fallback_claim_career_services(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} provides dedicated career coaching or career services specifically for online MBA students."


def _fallback_claim_full_time_faculty(extracted: ProgramExtraction) -> str:
    return f"{_program_label(extracted)} is taught primarily by full-time faculty members (not primarily adjunct instructors)."


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_program_identification_nodes(
    evaluator: Evaluator,
    root_node,
    extracted: ProgramExtraction,
) -> None:
    """
    Build and verify the 'program_identification' parallel node with 3 critical children.
    """
    pid_node = evaluator.add_parallel(
        id="program_identification",
        desc="Answer clearly identifies the selected program and its sponsoring institution",
        parent=root_node,
        critical=True,
    )

    # University name provided (existence check)
    evaluator.add_custom_node(
        result=bool((extracted.university_name or "").strip()),
        id="university_name_provided",
        desc="Provides the name of the university offering the program",
        parent=pid_node,
        critical=True,
    )

    # Business school name provided (existence check)
    evaluator.add_custom_node(
        result=bool((extracted.business_school_name or "").strip()),
        id="business_school_name_provided",
        desc="Provides the name of the business school offering the program",
        parent=pid_node,
        critical=True,
    )

    # Program identified as an online MBA (verification leaf, prefer program URL + other relevant URLs)
    online_leaf = evaluator.add_leaf(
        id="program_identified_as_online_mba",
        desc="Identifies the program as an online MBA program (program name/degree clearly stated)",
        parent=pid_node,
        critical=True,
    )

    # Sources for identification: program_url + some criterion URLs likely to assert "online MBA"
    sources = []
    if extracted.program_url and extracted.program_url.strip():
        sources.append(extracted.program_url.strip())
    sources.extend(_non_empty_urls(extracted.c11_primarily_asynchronous.urls))
    sources.extend(_non_empty_urls(extracted.c1_top10.urls))
    sources.extend(_non_empty_urls(extracted.c8_no_mandatory_on_campus.urls))
    # deduplicate while preserving order
    seen = set()
    sources_unique = []
    for u in sources:
        if u not in seen:
            seen.add(u)
            sources_unique.append(u)

    program_name = (extracted.program_name or "the selected program").strip()
    bs = (extracted.business_school_name or "").strip()
    univ = (extracted.university_name or "").strip()
    claim = f"{program_name} is an online MBA program offered by {bs + ', ' if bs else ''}{univ if univ else 'a U.S. university'}."

    await evaluator.verify(
        claim=claim,
        node=online_leaf,
        sources=sources_unique,
        additional_instruction=(
            "Verify that the program is explicitly an online MBA (distance/online delivery). "
            "Accept wording like 'Online MBA', 'MBA (online)', 'distance MBA', or similar. "
            "Prefer official program pages; US News ranking pages referencing the program are acceptable if they clearly "
            "name the program as an online MBA."
        ),
    )


async def verify_criterion_sequential(
    evaluator: Evaluator,
    parent_node,
    criterion_node_id: str,
    criterion_desc: str,
    meets_desc: str,
    evidence_desc: str,
    claim_text: str,
    urls: List[str],
    additional_instruction: str,
) -> None:
    """
    Create a sequential node with two critical children:
    1) meets_requirement - verify claim against provided URLs (preferred).
    2) evidence_url_provided - existence check for URLs.
    """
    crit_node = evaluator.add_sequential(
        id=criterion_node_id,
        desc=criterion_desc,
        parent=parent_node,
        critical=True,
    )

    # 1) Meets requirement (verification leaf using claim + URLs)
    meets_leaf = evaluator.add_leaf(
        id=f"{criterion_node_id}_meets_requirement",
        desc=meets_desc,
        parent=crit_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_text,
        node=meets_leaf,
        sources=urls,
        additional_instruction=additional_instruction,
    )

    # 2) Evidence URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(_non_empty_urls(urls)),
        id=f"{criterion_node_id}_evidence_url_provided",
        desc=evidence_desc,
        parent=crit_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main verification orchestration                                             #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ProgramExtraction) -> None:
    """
    Build the full verification tree under root: program identification + 14 criteria.
    """
    root = evaluator.root

    # Program identification
    await add_program_identification_nodes(evaluator, root, extracted)

    # Prepare claims and additional instructions for each criterion
    mappings = [
        {
            "id": "criterion_1_top_ranking",
            "desc": "Top-10 US News Best Online MBA Programs 2025 requirement and evidence",
            "meets_desc": "The program is ranked in the top 10 of the US News & World Report Best Online MBA Programs 2025 rankings",
            "evidence_desc": "Provides at least one reference URL that directly supports the program's top-10 ranking in the 2025 US News online MBA rankings",
            "criterion": extracted.c1_top10,
            "fallback": _fallback_claim_top10(extracted),
            "add_ins": (
                "Only accept evidence that explicitly references US News & World Report 2025 Best Online MBA Programs. "
                "The program must appear within the top 10 placements for 2025. Prefer US News pages; "
                "school pages referencing the ranking are acceptable if they clearly identify the 2025 online MBA top-10 status."
            ),
        },
        {
            "id": "criterion_2_affordable_tuition",
            "desc": "Total cost under $70,000 requirement and evidence",
            "meets_desc": "The total program cost (tuition and fees) is under $70,000",
            "evidence_desc": "Provides at least one reference URL that directly supports the stated total program cost (tuition and fees)",
            "criterion": extracted.c2_cost_under_70k,
            "fallback": _fallback_claim_cost(extracted),
            "add_ins": (
                "Confirm that the total program cost (tuition + required fees) is under $70,000. "
                "Prefer official program pages. If costs are per-credit, ensure reasonable total using the "
                "program's required credit hours when explicitly shown; otherwise require an explicit total stated under $70,000."
            ),
        },
        {
            "id": "criterion_3_completion_duration",
            "desc": "Completion within 24 months requirement and evidence",
            "meets_desc": "The program offers a completion pathway of 24 months or less",
            "evidence_desc": "Provides at least one reference URL that directly supports the completion timeframe/pathway (<= 24 months)",
            "criterion": extracted.c3_duration_24_months_or_less,
            "fallback": _fallback_claim_duration(extracted),
            "add_ins": (
                "Verify an explicit pathway phrased as 'finish in 24 months or less', 'as few as 24 months', "
                "or equivalent. Slight wording variations are acceptable if they clearly indicate <= 24 months."
            ),
        },
        {
            "id": "criterion_4_rolling_admissions",
            "desc": "Rolling admissions requirement and evidence",
            "meets_desc": "The program accepts applications on a rolling admissions basis (not fixed deadlines only)",
            "evidence_desc": "Provides at least one reference URL that directly supports rolling admissions",
            "criterion": extracted.c4_rolling_admissions,
            "fallback": _fallback_claim_rolling(extracted),
            "add_ins": (
                "Look for terms like 'rolling admissions', 'applications reviewed as they are received', "
                "'no fixed deadlines', or similar. The evidence must indicate rolling admissions specifically."
            ),
        },
        {
            "id": "criterion_5_test_waiver",
            "desc": "GMAT/GRE waiver for >=5 years experience requirement and evidence",
            "meets_desc": "The program waives GMAT/GRE requirements for applicants with 5 or more years of professional work experience",
            "evidence_desc": "Provides at least one reference URL that directly supports the GMAT/GRE waiver policy for applicants with >=5 years experience",
            "criterion": extracted.c5_gmat_gre_waiver_5plus_years,
            "fallback": _fallback_claim_waiver(extracted),
            "add_ins": (
                "Accept policy statements like 'GMAT/GRE waived for applicants with 5+ years of work experience', "
                "'test optional for experienced professionals', or equivalent wording. The threshold must be >=5 years."
            ),
        },
        {
            "id": "criterion_6_work_experience",
            "desc": "Minimum 2 years full-time experience requirement and evidence",
            "meets_desc": "The program requires at least 2 years of full-time professional work experience for admission",
            "evidence_desc": "Provides at least one reference URL that directly supports the minimum full-time work experience requirement (>=2 years)",
            "criterion": extracted.c6_min_2yrs_work_experience,
            "fallback": _fallback_claim_work_exp(extracted),
            "add_ins": (
                "Confirm that admission explicitly requires >= 2 years of full-time professional work experience. "
                "Equivalent phrases like 'minimum two years' are acceptable."
            ),
        },
        {
            "id": "criterion_7_aacsb_accreditation",
            "desc": "AACSB accreditation requirement and evidence",
            "meets_desc": "The business school/program holds AACSB accreditation",
            "evidence_desc": "Provides at least one reference URL that directly supports AACSB accreditation status (e.g., AACSB listing or school accreditation page)",
            "criterion": extracted.c7_aacsb_accreditation,
            "fallback": _fallback_claim_aacsb(extracted),
            "add_ins": (
                "Prefer AACSB official listing pages; school accreditation pages explicitly stating AACSB accreditation are acceptable."
            ),
        },
        {
            "id": "criterion_8_no_mandatory_residency",
            "desc": "No mandatory on-campus attendance requirement and evidence",
            "meets_desc": "The program does not require mandatory on-campus attendance (optional in-person experiences acceptable)",
            "evidence_desc": "Provides at least one reference URL that directly supports that on-campus attendance is not mandatory",
            "criterion": extracted.c8_no_mandatory_on_campus,
            "fallback": _fallback_claim_no_residency(extracted),
            "add_ins": (
                "Look for statements like 'no on-campus requirement', 'no required residencies', or 'in-person components are optional'."
            ),
        },
        {
            "id": "criterion_9_specializations",
            "desc": "At least 5 concentrations/specializations requirement and evidence",
            "meets_desc": "The program offers at least 5 different concentration or specialization options",
            "evidence_desc": "Provides at least one reference URL that directly supports the available concentrations/specializations (showing >=5 options)",
            "criterion": extracted.c9_at_least_5_specializations,
            "fallback": _fallback_claim_specializations(extracted),
            "add_ins": (
                "The evidence should list specializations/concentrations or indicate the count is at least five."
            ),
        },
        {
            "id": "criterion_10_start_dates",
            "desc": "At least 3 start dates per year requirement and evidence",
            "meets_desc": "The program offers at least 3 start dates per year",
            "evidence_desc": "Provides at least one reference URL that directly supports the number/frequency of start dates (>=3 per year)",
            "criterion": extracted.c10_at_least_3_start_dates,
            "fallback": _fallback_claim_start_dates(extracted),
            "add_ins": (
                "Verify language indicating three or more annual intakes/start dates (e.g., Fall/Spring/Summer or specific months)."
            ),
        },
        {
            "id": "criterion_11_asynchronous_format",
            "desc": "Primarily asynchronous instruction requirement and evidence",
            "meets_desc": "The program provides primarily asynchronous online instruction",
            "evidence_desc": "Provides at least one reference URL that directly supports the primarily asynchronous delivery format",
            "criterion": extracted.c11_primarily_asynchronous,
            "fallback": _fallback_claim_asynchronous(extracted),
            "add_ins": (
                "Accept wording like 'primarily asynchronous', 'mostly asynchronous', 'on-demand lectures'. "
                "Some synchronous sessions may exist, but the primary mode must be asynchronous."
            ),
        },
        {
            "id": "criterion_12_credit_hours",
            "desc": "Credit hours between 45 and 65 requirement and evidence",
            "meets_desc": "The program requires between 45 and 65 credit hours to complete the degree",
            "evidence_desc": "Provides at least one reference URL that directly supports the stated credit-hour requirement",
            "criterion": extracted.c12_credits_between_45_and_65,
            "fallback": _fallback_claim_credit_hours(extracted),
            "add_ins": (
                "Confirm explicit credit-hour requirements within the inclusive range 45–65 credits."
            ),
        },
        {
            "id": "criterion_13_career_services",
            "desc": "Dedicated online-MBA career services requirement and evidence",
            "meets_desc": "The program provides dedicated career coaching or career services specifically for online MBA students",
            "evidence_desc": "Provides at least one reference URL that directly supports dedicated career services/coaching for online MBA students",
            "criterion": extracted.c13_dedicated_career_services_online_mba,
            "fallback": _fallback_claim_career_services(extracted),
            "add_ins": (
                "Look for dedicated career resources explicitly aimed at online MBA students (e.g., 'online MBA career coach', "
                "'career services for online MBA'). General school-wide services are acceptable if they explicitly include online MBA students."
            ),
        },
        {
            "id": "criterion_14_full_time_faculty",
            "desc": "Primarily full-time faculty requirement and evidence",
            "meets_desc": "The program is taught primarily by full-time faculty members (not primarily adjunct instructors)",
            "evidence_desc": "Provides at least one reference URL that directly supports the faculty composition claim (primarily full-time)",
            "criterion": extracted.c14_primarily_full_time_faculty,
            "fallback": _fallback_claim_full_time_faculty(extracted),
            "add_ins": (
                "Evidence should indicate that instruction is primarily delivered by full-time faculty. "
                "Statements like 'courses taught by full-time faculty' or 'majority full-time instructors' are acceptable."
            ),
        },
    ]

    # Build each criterion with sequential children
    for m in mappings:
        crit_data: CriterionData = m["criterion"]
        claim_text = (crit_data.claim or "").strip() or m["fallback"]
        urls = _non_empty_urls(crit_data.urls)
        await verify_criterion_sequential(
            evaluator=evaluator,
            parent_node=root,
            criterion_node_id=m["id"],
            criterion_desc=m["desc"],
            meets_desc=m["meets_desc"],
            evidence_desc=m["evidence_desc"],
            claim_text=claim_text,
            urls=urls,
            additional_instruction=m["add_ins"],
        )

    # Optional: add custom info to summary for debugging
    evaluator.add_custom_info(
        info={
            "program": {
                "university_name": extracted.university_name,
                "business_school_name": extracted.business_school_name,
                "program_name": extracted.program_name,
                "program_url": extracted.program_url,
            },
            "evidence_url_counts": {
                "c1": len(_non_empty_urls(extracted.c1_top10.urls)),
                "c2": len(_non_empty_urls(extracted.c2_cost_under_70k.urls)),
                "c3": len(_non_empty_urls(extracted.c3_duration_24_months_or_less.urls)),
                "c4": len(_non_empty_urls(extracted.c4_rolling_admissions.urls)),
                "c5": len(_non_empty_urls(extracted.c5_gmat_gre_waiver_5plus_years.urls)),
                "c6": len(_non_empty_urls(extracted.c6_min_2yrs_work_experience.urls)),
                "c7": len(_non_empty_urls(extracted.c7_aacsb_accreditation.urls)),
                "c8": len(_non_empty_urls(extracted.c8_no_mandatory_on_campus.urls)),
                "c9": len(_non_empty_urls(extracted.c9_at_least_5_specializations.urls)),
                "c10": len(_non_empty_urls(extracted.c10_at_least_3_start_dates.urls)),
                "c11": len(_non_empty_urls(extracted.c11_primarily_asynchronous.urls)),
                "c12": len(_non_empty_urls(extracted.c12_credits_between_45_and_65.urls)),
                "c13": len(_non_empty_urls(extracted.c13_dedicated_career_services_online_mba.urls)),
                "c14": len(_non_empty_urls(extracted.c14_primarily_full_time_faculty.urls)),
            }
        },
        info_type="debug",
        info_name="extraction_summary",
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
    Evaluate the answer for the Online MBA selection task:
    - Extract program identification and per-criterion claims + evidence URLs.
    - Build verification tree: program identification (parallel) + 14 criteria (sequential).
    - Verify claims against provided URLs using the Mind2Web2 LLM-as-a-Judge framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel root node (children independent)
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

    # 1) Extraction
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program_and_evidence(),
        template_class=ProgramExtraction,
        extraction_name="program_and_evidence",
    )

    # 2) Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # 3) Return summary
    return evaluator.get_summary()