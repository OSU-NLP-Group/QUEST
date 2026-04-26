import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fellowship_eligibility_2026"
TASK_DESCRIPTION = """
Dr. Sarah Chen is a postdoctoral researcher planning to apply for fellowship funding in 2026. She needs to determine which of the following postdoctoral fellowship programs she is eligible to apply for: the Marie Curie MSCA Postdoctoral Fellowship (with a host institution in Germany) and the NIH F32 Postdoctoral Fellowship.

Dr. Chen's profile:
- Successfully defended her PhD thesis in molecular biology in August 2022 at Stanford University
- Formally awarded her PhD degree in December 2022
- U.S. permanent resident
- Has been conducting postdoctoral research at the Max Planck Institute in Munich, Germany since January 2023
- Current date: March 2026

Based on the official eligibility requirements for each fellowship program, determine:
1. Is Dr. Chen eligible to apply for the Marie Curie MSCA Postdoctoral Fellowship (2026 call, assuming a September 2026 deadline and a German host institution for a European Fellowship)?
2. Is Dr. Chen eligible to apply for the NIH F32 Postdoctoral Fellowship (for an application to be submitted in 2026)?

For each fellowship, provide:
- A clear YES or NO determination of eligibility
- The specific eligibility criteria that determine her eligibility or ineligibility
- Reference URL(s) supporting your determination
""".strip()


# --------------------------------------------------------------------------- #
# Profile and date assumptions for computation                                #
# --------------------------------------------------------------------------- #
PHD_DEFENSE_DATE = date(2022, 8, 15)   # Successfully defended in Aug 2022
PHD_AWARD_DATE = date(2022, 12, 15)    # Awarded in Dec 2022
IS_US_PERMANENT_RESIDENT = True
GERMANY_RESIDENCE_START = date(2023, 1, 1)  # Postdoc in Germany since Jan 2023

# Application timing assumptions
MSCA_DEADLINE = date(2026, 9, 30)      # Assume September 2026 deadline (end of month for safety)
F32_APPLICATION_DATE = date(2026, 6, 1)  # Assume a 2026 submission date


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def years_between(d1: date, d2: date) -> float:
    return abs((d2 - d1).days) / 365.25


def months_between(d1: date, d2: date) -> float:
    return abs((d2 - d1).days) / 30.44


def overlap_days(start1: date, end1: date, start2: date, end2: date) -> int:
    start = max(start1, start2)
    end = min(end1, end2)
    if end < start:
        return 0
    return (end - start).days + 1


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FellowshipAssessment(BaseModel):
    eligibility: Optional[str] = None  # Expect values like "YES" or "NO" (case-insensitive)
    criteria_summary: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class EligibilityExtraction(BaseModel):
    msca: Optional[FellowshipAssessment] = None
    f32: Optional[FellowshipAssessment] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eligibility() -> str:
    return """
    Extract, for each of the following two fellowships, the final eligibility determination provided in the answer, a short criteria summary, and the list of reference URLs cited.

    Programs to extract:
    1) msca: Marie Curie MSCA Postdoctoral Fellowship (European Fellowship with German host, 2026 call)
    2) f32: NIH F32 Postdoctoral Fellowship

    For each program, return:
    - eligibility: The final YES/NO determination exactly as stated (normalize to just "YES" or "NO" if possible)
    - criteria_summary: A brief textual summary of the specific criteria that were mentioned to justify the decision
    - reference_urls: An array of all URLs cited in the answer that support the eligibility reasoning for that program

    If any field is missing, set it to null (for strings) or an empty array (for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification helpers for MSCA and F32                                       #
# --------------------------------------------------------------------------- #
def compute_msca_checks() -> Dict[str, Any]:
    # PhD timing: Must hold a PhD (or have defended) by the application deadline
    phd_timing_ok = (PHD_AWARD_DATE <= MSCA_DEADLINE) or (PHD_DEFENSE_DATE <= MSCA_DEADLINE)

    # Research experience limit: ≤ 8 years from PhD award by deadline
    rexp_years = years_between(PHD_AWARD_DATE, MSCA_DEADLINE)
    rexp_ok = rexp_years <= 8.0

    # Mobility rule: Not have resided or had main activity in host country > 12 months in the 36 months before deadline
    window_days = int(365.25 * 3)  # approx 36 months
    window_start = MSCA_DEADLINE - timedelta(days=window_days)
    window_end = MSCA_DEADLINE
    overlap = overlap_days(GERMANY_RESIDENCE_START, MSCA_DEADLINE, window_start, window_end)
    months_in_germany_last36 = overlap / 30.44
    mobility_ok = months_in_germany_last36 <= 12.0

    return {
        "phd_timing_ok": phd_timing_ok,
        "research_experience_years": rexp_years,
        "research_experience_ok": rexp_ok,
        "mobility_months_last36": months_in_germany_last36,
        "mobility_ok": mobility_ok
    }


def compute_f32_checks() -> Dict[str, Any]:
    # Citizenship/permanent residency requirement
    citizenship_ok = IS_US_PERMANENT_RESIDENT is True

    # Doctoral degree requirement: must have received a doctoral degree
    doctoral_degree_ok = PHD_AWARD_DATE <= F32_APPLICATION_DATE

    # Career stage window: within 0-5 years post-PhD completion (as per rubric)
    years_post_phd = years_between(PHD_AWARD_DATE, F32_APPLICATION_DATE)
    career_stage_ok = years_post_phd <= 5.0

    return {
        "citizenship_ok": citizenship_ok,
        "doctoral_degree_ok": doctoral_degree_ok,
        "years_post_phd": years_post_phd,
        "career_stage_ok": career_stage_ok
    }


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def build_msca_branch(evaluator: Evaluator, parent_node, extracted: Optional[FellowshipAssessment]) -> None:
    msca_node = evaluator.add_parallel(
        id="Marie_Curie_MSCA_Eligibility",
        desc="Determine eligibility for Marie Curie MSCA Postdoctoral Fellowship based on PhD timing, research experience, and mobility requirements",
        parent=parent_node,
        critical=False
    )

    checks = compute_msca_checks()

    # Leaf: PhD timing requirement met
    evaluator.add_custom_node(
        result=checks["phd_timing_ok"],
        id="MSCA_PhD_Timing_Requirement",
        desc="Verify that the researcher either holds a PhD degree or has successfully defended their doctoral thesis by the application deadline",
        parent=msca_node,
        critical=True
    )

    # Leaf: Research experience limit (≤ 8 years from PhD award by deadline)
    evaluator.add_custom_node(
        result=checks["research_experience_ok"],
        id="MSCA_Research_Experience_Limit",
        desc="Verify that the researcher has no more than 8 years of research experience from PhD award date, excluding career breaks and time outside research",
        parent=msca_node,
        critical=True
    )

    # Leaf: Mobility rule compliance for German host institution (European Fellowship)
    evaluator.add_custom_node(
        result=checks["mobility_ok"],
        id="MSCA_Mobility_Rule_Compliance",
        desc="Verify that the researcher has not resided or carried out main activity in the intended host country for more than 12 months in the 36 months before the deadline",
        parent=msca_node,
        critical=True
    )

    # Record helpful computed info
    evaluator.add_custom_info(
        {
            "msca_deadline": MSCA_DEADLINE.isoformat(),
            "phd_award_date": PHD_AWARD_DATE.isoformat(),
            "phd_defense_date": PHD_DEFENSE_DATE.isoformat(),
            "research_experience_years_as_of_deadline": round(checks["research_experience_years"], 2),
            "germany_residence_start": GERMANY_RESIDENCE_START.isoformat(),
            "months_in_germany_within_last_36": round(checks["mobility_months_last36"], 2),
            "extracted_reference_urls": (extracted.reference_urls if extracted else [])
        },
        info_type="msca_computation",
        info_name="msca_derived_metrics"
    )


async def build_f32_branch(evaluator: Evaluator, parent_node, extracted: Optional[FellowshipAssessment]) -> None:
    f32_node = evaluator.add_parallel(
        id="NIH_F32_Eligibility",
        desc="Determine eligibility for NIH F32 Postdoctoral Fellowship based on citizenship, doctoral degree, and career stage requirements",
        parent=parent_node,
        critical=False
    )

    checks = compute_f32_checks()

    # Leaf: Citizenship/permanent residency requirement
    evaluator.add_custom_node(
        result=checks["citizenship_ok"],
        id="F32_Citizenship_Requirement",
        desc="Verify that the researcher is a U.S. citizen or permanent resident",
        parent=f32_node,
        critical=True
    )

    # Leaf: Doctoral degree requirement
    evaluator.add_custom_node(
        result=checks["doctoral_degree_ok"],
        id="F32_Doctoral_Degree_Requirement",
        desc="Verify that the researcher has received a doctoral degree (PhD, MD, DO, DC, DDS, or equivalent)",
        parent=f32_node,
        critical=True
    )

    # Leaf: Career stage window (0–5 years post-PhD per rubric)
    evaluator.add_custom_node(
        result=checks["career_stage_ok"],
        id="F32_Career_Stage_Window",
        desc="Verify that the researcher is within 0-5 years post-PhD completion (optimal eligibility window)",
        parent=f32_node,
        critical=True
    )

    # Record helpful computed info
    evaluator.add_custom_info(
        {
            "f32_application_date": F32_APPLICATION_DATE.isoformat(),
            "phd_award_date": PHD_AWARD_DATE.isoformat(),
            "years_post_phd_as_of_application": round(checks["years_post_phd"], 2),
            "citizenship_or_pr": IS_US_PERMANENT_RESIDENT,
            "extracted_reference_urls": (extracted.reference_urls if extracted else [])
        },
        info_type="f32_computation",
        info_name="f32_derived_metrics"
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
        default_model=model
    )

    # Extract the answer's reported determinations and URLs (for record-keeping)
    extracted = await evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility_extraction"
    )

    # Ground-truth profile and timing context (as provided by task)
    evaluator.add_ground_truth({
        "profile": {
            "phd_defense_date": PHD_DEFENSE_DATE.isoformat(),
            "phd_award_date": PHD_AWARD_DATE.isoformat(),
            "us_permanent_resident": IS_US_PERMANENT_RESIDENT,
            "germany_residence_start": GERMANY_RESIDENCE_START.isoformat(),
            "current_date_context": "2026-03-01 to 2026-03-31"
        },
        "applications": {
            "msca_deadline_assumed": MSCA_DEADLINE.isoformat(),
            "f32_application_date_assumed": F32_APPLICATION_DATE.isoformat()
        }
    }, gt_type="task_profile_and_assumptions")

    # Build the rubric tree as specified (root parallel -> two parallel branches)
    top = evaluator.add_parallel(
        id="Fellowship_Eligibility_Assessment",
        desc="Evaluate which postdoctoral fellowship programs the researcher is eligible to apply for based on their specific profile and circumstances",
        parent=root,
        critical=False
    )

    # Build MSCA branch
    await build_msca_branch(
        evaluator=evaluator,
        parent_node=top,
        extracted=(extracted.msca if extracted else None)
    )

    # Build F32 branch
    await build_f32_branch(
        evaluator=evaluator,
        parent_node=top,
        extracted=(extracted.f32 if extracted else None)
    )

    # Return structured result
    return evaluator.get_summary()