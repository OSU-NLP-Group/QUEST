import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_public_univ_grad_cs_2026"
TASK_DESCRIPTION = """
I am an international student planning to pursue a graduate degree in Computer Science starting in Fall 2026. I am specifically interested in studying at a large public research university in Florida that has both strong research activity and a highly-ranked computer science program.

Please identify one public university in Florida that meets ALL of the following requirements:

1. Institutional Classification: Must have R1 Carnegie Classification status (Very High Research Activity)
2. Program Quality: The graduate Computer Science program must be ranked in the top 50 by U.S. News & World Report 2025 rankings
3. Academic Calendar: Uses a semester system (not quarter or trimester)
4. Affordability: In-state graduate tuition for the 2025-26 academic year must be under $14,000 per year
5. Campus Size: Total student enrollment between 40,000 and 70,000 students
6. Faculty Information: Computer Science department provides publicly available faculty qualification information
7. Admission Accessibility: Minimum GPA requirement of 3.0 or lower for graduate CS admission
8. Application Timeline: Must accept applications for Fall 2026 admission with a deadline for domestic applicants of May 1, 2026 or later

For your answer, provide the university name and include reference URLs that verify each of these requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    proposed_university: Optional[str] = None
    all_universities_mentioned: List[str] = Field(default_factory=list)

    # Evidence URLs per constraint (must be explicitly present in the answer)
    location_urls: List[str] = Field(default_factory=list)
    public_status_urls: List[str] = Field(default_factory=list)
    r1_urls: List[str] = Field(default_factory=list)
    grad_cs_urls: List[str] = Field(default_factory=list)
    usnews_urls: List[str] = Field(default_factory=list)
    calendar_urls: List[str] = Field(default_factory=list)
    tuition_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    faculty_qual_urls: List[str] = Field(default_factory=list)
    gpa_urls: List[str] = Field(default_factory=list)
    app_term_urls: List[str] = Field(default_factory=list)
    deadline_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_and_urls() -> str:
    return """
    Your task is to extract structured information from the answer that identifies a single Florida public university and the evidence URLs provided for each required constraint.

    Extract the following fields:
    - proposed_university: The single specific university the answer recommends as the match. If multiple are proposed, still extract the primary one the answer presents as the match. If ambiguous, set to null.
    - all_universities_mentioned: List all university names mentioned anywhere in the answer (including the proposed one, if present).

    For each constraint below, extract the explicit URL(s) that the answer cites as evidence. Only include URLs that actually appear in the answer text (including markdown links). Do not invent or infer any URLs.

    - location_urls: URLs that show the university is located in Florida.
    - public_status_urls: URLs that show the university is a public (state-funded) institution.
    - r1_urls: URLs that show the university has Carnegie Classification R1 (Very High Research Activity). Prefer authoritative sources (e.g., Carnegie Classification site) or official university statements clearly indicating R1 status.
    - grad_cs_urls: URLs that show the university offers graduate Computer Science degree programs (MS and/or PhD) in CS.
    - usnews_urls: URLs that show the graduate CS program is ranked in the top 50 by U.S. News & World Report 2025. Accept the official US News page or an authoritative announcement/news post that explicitly cites the 2025 US News ranking and the top-50 standing for graduate CS.
    - calendar_urls: URLs that show the university uses a semester academic calendar (not quarter or trimester).
    - tuition_urls: URLs that show in-state graduate tuition for the 2025–26 academic year and allow verifying that the annual total is under $14,000 (tuition only; fees excluded).
    - enrollment_urls: URLs that show total student enrollment (headcount) to verify it's between 40,000 and 70,000. Prefer official facts/statistics/IR pages or other authoritative profiles stating totals and a date/year.
    - faculty_qual_urls: URLs on the CS department site that publicly show faculty qualifications (e.g., degrees, bios, CVs).
    - gpa_urls: URLs that state the minimum GPA requirement for graduate CS admission is 3.0 or lower (program page or graduate school policy page that applies to CS).
    - app_term_urls: URLs that show applications are accepted for Fall 2026 entry term (university/department admissions page or application portal page that explicitly references Fall 2026).
    - deadline_urls: URLs that show the domestic (U.S.) application deadline for Fall 2026 is May 1, 2026 or later (page must explicitly indicate date and that it applies to domestic applicants).

    Rules:
    - Extract only URLs explicitly present in the answer text. If none are provided for a field, return an empty list for that field.
    - Ensure the 'all_universities_mentioned' includes every university named anywhere in the answer text (deduplicate later logic will handle uniqueness).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_names(names: List[str]) -> List[str]:
    return [n.strip().lower() for n in names if isinstance(n, str) and n.strip()]


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


async def _add_constraint_with_url_and_verify(
    evaluator: Evaluator,
    parent,
    *,
    base_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    additional_instruction: str,
    critical: bool = True,
) -> None:
    """
    Create a sequential container for a constraint with:
    - Critical leaf: URLs provided
    - Critical leaf: Verification supported by those URLs
    """
    container = evaluator.add_sequential(
        id=base_id,
        desc=desc,
        parent=parent,
        critical=critical
    )

    # Leaf 1: URLs provided (critical)
    evaluator.add_custom_node(
        result=_urls_present(urls),
        id=f"{base_id}_urls_present",
        desc=f"URLs are provided in the answer for: {desc}",
        parent=container,
        critical=True
    )

    # Leaf 2: Verification against provided URLs (critical)
    verify_leaf = evaluator.add_leaf(
        id=f"{base_id}_supported",
        desc=f"Claim is supported by the provided URL(s): {desc}",
        parent=container,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification logic tree construction                                        #
# --------------------------------------------------------------------------- #
async def verify_qualifying_university(
    evaluator: Evaluator,
    root,
    extracted: UniversityExtraction
) -> None:
    """
    Build the verification tree according to the rubric.
    """
    # Qualifying_University (critical, sequential)
    qual_node = evaluator.add_sequential(
        id="Qualifying_University",
        desc="Identify exactly one Florida public university that meets all stated constraints for Fall 2026 graduate Computer Science study, and provide reference URL(s) that verify each constraint.",
        parent=root,
        critical=True
    )

    # 1) Single_University_Provided (critical custom leaf)
    proposed = extracted.proposed_university or ""
    mentioned = list(extracted.all_universities_mentioned or [])
    names_set = set(_norm_names(mentioned + ([proposed] if proposed else [])))
    exactly_one_univ = (len(names_set) == 1) and bool(proposed.strip())

    evaluator.add_custom_node(
        result=exactly_one_univ,
        id="Single_University_Provided",
        desc="Answer identifies exactly one specific university (by name) as the proposed match.",
        parent=qual_node,
        critical=True
    )

    # 2) Meets_All_Constraints_With_Evidence (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="Meets_All_Constraints_With_Evidence",
        desc="The identified university satisfies every listed constraint, and the answer includes URL evidence for each constraint.",
        parent=qual_node,
        critical=True
    )

    univ = proposed.strip() if proposed else "the university"

    # Constraint: Located in Florida
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Located_In_Florida_With_URL",
        desc="University is located in the state of Florida AND answer provides at least one URL verifying this.",
        claim=f"{univ} is located in the state of Florida.",
        urls=extracted.location_urls,
        additional_instruction="Accept official university pages, state profiles, or credible sources (e.g., Wikipedia official infobox) that clearly state the university is in Florida."
    )

    # Constraint: Public university
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Public_University_With_URL",
        desc="University is a public (state-funded) institution AND answer provides at least one URL verifying this.",
        claim=f"{univ} is a public, state-funded university.",
        urls=extracted.public_status_urls,
        additional_instruction="Look for explicit phrasing like 'public university' on official sites or authoritative profiles. Avoid relying on inference."
    )

    # Constraint: R1 Carnegie Classification
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="R1_Carnegie_VeryHigh_With_URL",
        desc="University has R1 Carnegie Classification status (Very High Research Activity), with authoritative evidence.",
        claim=f"{univ} has Carnegie Classification R1 (Very high research activity).",
        urls=extracted.r1_urls,
        additional_instruction="Prefer the Carnegie Classification official website or an official university page/news that explicitly states 'R1: Doctoral Universities – Very high research activity'. If the page also explains the basis, that's fine; otherwise explicit R1 labeling suffices."
    )

    # Constraint: Offers graduate CS program
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Offers_Graduate_CS_With_URL",
        desc="University offers a graduate Computer Science degree program (MS and/or PhD) AND answer provides at least one URL verifying this.",
        claim=f"{univ} offers graduate Computer Science degree programs (MS and/or PhD) within Computer Science.",
        urls=extracted.grad_cs_urls,
        additional_instruction="Verify on the CS department, college, or graduate catalog pages that a graduate degree (MS and/or PhD) in Computer Science exists."
    )

    # Constraint: US News Top 50 in 2025 for graduate CS
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="USNews_Top50_2025_With_URL",
        desc="Graduate Computer Science program is ranked in the top 50 by U.S. News & World Report 2025 AND answer provides at least one URL verifying this.",
        claim=f"The graduate Computer Science program at {univ} is ranked within the top 50 by U.S. News & World Report for the 2025 rankings.",
        urls=extracted.usnews_urls,
        additional_instruction="Accept the official U.S. News page (if accessible) or an official university/department announcement explicitly citing 'U.S. News & World Report 2025' and 'top 50' for graduate CS. If the year or field is not explicit, do NOT accept."
    )

    # Constraint: Semester system
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Semester_System_With_URL",
        desc="University uses a semester academic calendar system AND answer provides at least one URL verifying this.",
        claim=f"{univ} uses a semester academic calendar (not quarter or trimester).",
        urls=extracted.calendar_urls,
        additional_instruction="Verify on the academic calendar, registrar, or catalog pages that the institution operates on semesters."
    )

    # Constraint: Tuition under $14,000 for 2025–26 (in-state graduate, per year)
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Tuition_Under_14000_2025_26_With_URL",
        desc="In-state graduate tuition for the 2025–26 academic year is under $14,000/year AND answer provides at least one URL verifying the amount and year.",
        claim=f"In-state graduate tuition for academic year 2025–26 at {univ} is under $14,000 per academic year (tuition only, excluding fees).",
        urls=extracted.tuition_urls,
        additional_instruction="The page must reference 2025–26 rates or an official table labeled for 2025–26. If only per-credit rates are shown, perform a reasonable full-time annualization (e.g., typical graduate load across two semesters) to judge whether it is under $14,000. Exclude mandatory fees from the calculation."
    )

    # Constraint: Enrollment between 40,000 and 70,000
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Enrollment_40k_70k_With_URL",
        desc="Total student enrollment is between 40,000 and 70,000 AND answer provides at least one URL verifying the enrollment figure and date/source.",
        claim=f"{univ}'s total student enrollment is between 40,000 and 70,000 students.",
        urls=extracted.enrollment_urls,
        additional_instruction="Use official facts/IR/statistics pages or other authoritative profiles that clearly state total (headcount) enrollment with a date or academic year. Accept if the stated total lies within the range."
    )

    # Constraint: Faculty qualifications publicly available
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Faculty_Qualifications_Public_With_URL",
        desc="CS department provides publicly available information about faculty qualifications AND answer provides at least one URL verifying availability (e.g., faculty pages/CVs/bios).",
        claim=f"The Computer Science department at {univ} provides publicly available information about faculty qualifications (e.g., degrees, bios, CVs).",
        urls=extracted.faculty_qual_urls,
        additional_instruction="Faculty directory pages showing degrees, bios, or links to CVs count as public qualification information. Verify that such pages are accessible."
    )

    # Constraint: Minimum GPA requirement ≤ 3.0
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Minimum_GPA_Le_3_0_With_URL",
        desc="Minimum GPA requirement for graduate CS admission is 3.0 or lower AND answer provides at least one URL verifying this requirement.",
        claim=f"The minimum GPA requirement for graduate Computer Science admission at {univ} is 3.0 or lower.",
        urls=extracted.gpa_urls,
        additional_instruction="Accept program pages or graduate school policy pages that apply to CS. Wording like 'minimum 3.0 (B) GPA' or '3.0 in the last 60 credits' qualifies as ≤ 3.0."
    )

    # Constraint: Accepts Fall 2026 applications
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Accepts_Fall_2026_Applications_With_URL",
        desc="University/program accepts applications for Fall 2026 admission AND answer provides at least one URL verifying Fall 2026 is an available entry term.",
        claim=f"{univ} (or the CS graduate program) accepts applications for the Fall 2026 entry term.",
        urls=extracted.app_term_urls,
        additional_instruction="The page must explicitly reference 'Fall 2026' as an available intake/entry term for applications (program or central graduate admissions)."
    )

    # Constraint: Domestic deadline May 1, 2026 or later
    await _add_constraint_with_url_and_verify(
        evaluator,
        constraints_node,
        base_id="Domestic_Deadline_May1OrLater_2026_With_URL",
        desc="Domestic applicant deadline for Fall 2026 is May 1, 2026 or later AND answer provides at least one URL verifying the deadline date and applicant type (domestic).",
        claim=f"For domestic applicants, the Fall 2026 application deadline at {univ} is May 1, 2026 or later.",
        urls=extracted.deadline_urls,
        additional_instruction="The page must explicitly reference Fall 2026 and a domestic (U.S.) applicant deadline that is ≥ May 1, 2026. If only international deadlines are shown, do not accept."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating the agent's answer against the rubric.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level container
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_and_urls(),
        template_class=UniversityExtraction,
        extraction_name="university_and_urls_extraction"
    )

    # Add some helpful context to the final report
    uniq_mentioned = list(set(_norm_names(extracted.all_universities_mentioned)))
    evaluator.add_custom_info(
        info={
            "proposed_university": extracted.proposed_university,
            "mentioned_university_count": len(uniq_mentioned),
            "constraints_url_counts": {
                "location": len(extracted.location_urls),
                "public_status": len(extracted.public_status_urls),
                "r1": len(extracted.r1_urls),
                "grad_cs": len(extracted.grad_cs_urls),
                "usnews": len(extracted.usnews_urls),
                "calendar": len(extracted.calendar_urls),
                "tuition_2025_26": len(extracted.tuition_urls),
                "enrollment": len(extracted.enrollment_urls),
                "faculty_qualifications": len(extracted.faculty_qual_urls),
                "gpa": len(extracted.gpa_urls),
                "fall_2026_entry": len(extracted.app_term_urls),
                "domestic_deadline_2026": len(extracted.deadline_urls),
            }
        },
        info_type="debug_summary"
    )

    # 2) Build verification tree and run checks
    await verify_qualifying_university(evaluator, root, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()