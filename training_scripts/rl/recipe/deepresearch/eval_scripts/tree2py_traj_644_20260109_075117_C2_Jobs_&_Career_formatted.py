import asyncio
import logging
import re
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "texas_pe_ce_renewal"
TASK_DESCRIPTION = (
    "A Professional Engineer licensed in Texas completed the following continuing education activities during their most recent "
    "annual renewal period: 8 hours of technical engineering courses (Advanced Structural Design), 1 hour of Texas Engineering "
    "Practice Act and Board Rules, 4 hours of self-directed study in geotechnical engineering (documented with dates, hours "
    "claimed, topic goals, resources used, and outcome summary), and 3 hours of project management for engineers. According to "
    "the Texas Board of Professional Engineers requirements, determine whether this PE has satisfied all continuing education "
    "requirements for license renewal. Specifically verify: (1) Whether the total Professional Development Hours (PDH) meet the "
    "required minimum of 15 hours annually, (2) Whether the requirement for at least 1 hour in Ethics and/or the Texas Engineering "
    "Practice Act/Board Rules is satisfied, (3) Whether self-directed study hours (if any) are within the permitted limit of 5 "
    "hours and properly documented as required (with date(s), hours claimed, topic goals, resources used, and outcome summary). "
    "Provide a clear determination of compliance or non-compliance for license renewal eligibility."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SelfStudyDocumentation(BaseModel):
    has_dates: Optional[bool] = None
    has_hours_claimed: Optional[bool] = None
    has_topic_goals: Optional[bool] = None
    has_resources_used: Optional[bool] = None
    has_outcome_summary: Optional[bool] = None

    # Optional snippets captured from the answer (helpful for auditing)
    dates_text: Optional[str] = None
    hours_text: Optional[str] = None
    topic_goals_text: Optional[str] = None
    resources_text: Optional[str] = None
    outcome_summary_text: Optional[str] = None


class CEExtraction(BaseModel):
    technical_hours: Optional[str] = None
    ethics_or_act_rules_hours: Optional[str] = None
    self_study_hours: Optional[str] = None
    project_management_hours: Optional[str] = None
    total_hours_claimed: Optional[str] = None
    self_study_documentation: Optional[SelfStudyDocumentation] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ce() -> str:
    return """
    Extract the continuing education (CE/PDH) details the answer claims for the most recent Texas PE renewal period.

    Return a JSON with the fields:
    - technical_hours: numeric hours as a string for technical engineering coursework (e.g., "8"). If not stated, null.
    - ethics_or_act_rules_hours: numeric hours as a string for Ethics and/or Texas Engineering Practice Act/Board Rules (e.g., "1"). If not stated, null.
    - self_study_hours: numeric hours as a string for self-directed study (e.g., "4"). If not stated, null.
    - project_management_hours: numeric hours as a string for project/engineering management (e.g., "3"). If not stated, null.
    - total_hours_claimed: if the answer explicitly gives a total PDH, extract it as a numeric string; otherwise null.

    - self_study_documentation: an object describing whether the answer explicitly documents the required elements for self-directed study. Set booleans to true only if the element is explicitly present or clearly stated in the answer. Include brief text snippets when available.
        - has_dates: whether date(s) are provided for the self-study
        - has_hours_claimed: whether hours claimed are stated for the self-study
        - has_topic_goals: whether topic goals/objectives are provided
        - has_resources_used: whether resources used are provided
        - has_outcome_summary: whether an outcome summary (what was learned/achieved) is provided

        - dates_text: short snippet showing the dates if present, else null
        - hours_text: short snippet showing the hours if present, else null
        - topic_goals_text: short snippet showing topic goals/objectives if present, else null
        - resources_text: short snippet showing resources if present, else null
        - outcome_summary_text: short snippet showing the outcome summary if present, else null

    Rules:
    - Always output numbers as simple numerals in strings if possible (e.g., "8", "1", "4", "3").
    - If any field is not explicitly present, return null for that field (or false for the boolean flags).
    - Do not infer or invent information.
    """


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
_WORD_NUMS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0,
    "nineteen": 19.0, "twenty": 20.0
}


def parse_hours_str(s: Optional[str]) -> float:
    """
    Parse a string to extract a number of hours. Handles common formats like "8", "8.0", "8 hours",
    and simple word-numbers like "eight hours". If a range appears, uses the first number found.
    Returns 0.0 if no number is found or input is None.
    """
    if not s:
        return 0.0
    s_low = s.strip().lower()

    # Try numeric first
    m = re.search(r"(\d+(?:\.\d+)?)", s_low)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # Try word numbers
    for w, v in _WORD_NUMS.items():
        if re.search(rf"\b{re.escape(w)}\b", s_low):
            return float(v)

    return 0.0


def bool_true(v: Optional[bool]) -> bool:
    return bool(v is True)


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: CEExtraction) -> None:
    """
    Build the rubric verification tree and perform the checks using computed booleans.
    All checks here are deterministic (no external URLs), so we use custom nodes.
    """

    # Compute numeric hours
    tech = parse_hours_str(extraction.technical_hours)
    ethics = parse_hours_str(extraction.ethics_or_act_rules_hours)
    self_study = parse_hours_str(extraction.self_study_hours)
    proj_mgmt = parse_hours_str(extraction.project_management_hours)

    computed_total = tech + ethics + self_study + proj_mgmt

    # Self-study documentation checks (apply only if self-study is claimed > 0)
    doc = extraction.self_study_documentation or SelfStudyDocumentation()
    doc_all_present = all([
        bool_true(doc.has_dates),
        bool_true(doc.has_hours_claimed),
        bool_true(doc.has_topic_goals),
        bool_true(doc.has_resources_used),
        bool_true(doc.has_outcome_summary),
    ])

    # Add a top-level aggregator node representing the rubric root (critical)
    main_node = evaluator.add_parallel(
        id="Texas_PE_License_Renewal_Eligibility",
        desc="Determine whether the PE satisfies Texas continuing education requirements for renewal based on total PDH, ethics/Act-Rules hours, and self-study limits/documentation.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Minimum total PDH >= 15
    evaluator.add_custom_node(
        result=(computed_total >= 15.0),
        id="Minimum_15_PDH_Total",
        desc="Total continuing education completed during the renewal period is at least 15 PDH.",
        parent=main_node,
        critical=True
    )

    # 2) Minimum 1 hour in Ethics and/or Act/Board Rules
    evaluator.add_custom_node(
        result=(ethics >= 1.0),
        id="Minimum_1_Hour_Ethics_or_ActRules",
        desc="At least 1 PDH is in Ethics and/or the Texas Engineering Practice Act/Board Rules during the renewal period.",
        parent=main_node,
        critical=True
    )

    # 3) Self-directed study compliance (parallel aggregator with two critical children)
    self_study_parent = evaluator.add_parallel(
        id="Self_Study_Compliance",
        desc="Self-directed study (if claimed) complies with Texas limits and documentation requirements.",
        parent=main_node,
        critical=True
    )

    # 3.a) Self-directed study within 5 PDH (pass if not claimed or <= 5)
    evaluator.add_custom_node(
        result=(self_study <= 5.0),
        id="Self_Study_Within_5_Hours",
        desc="Claimed self-directed study hours do not exceed 5 PDH.",
        parent=self_study_parent,
        critical=True
    )

    # 3.b) Proper documentation for self-directed study (pass if not claimed; else require all five elements)
    proper_doc_pass = True if self_study <= 0.0 else doc_all_present
    evaluator.add_custom_node(
        result=proper_doc_pass,
        id="Self_Study_Properly_Documented",
        desc="Self-directed study is documented with date(s), hours claimed, topic goals, resources used, and an outcome summary.",
        parent=self_study_parent,
        critical=True
    )

    # Record auxiliary info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_hours": {
                "technical_hours_raw": extraction.technical_hours,
                "ethics_or_act_rules_hours_raw": extraction.ethics_or_act_rules_hours,
                "self_study_hours_raw": extraction.self_study_hours,
                "project_management_hours_raw": extraction.project_management_hours,
                "total_hours_claimed_raw": extraction.total_hours_claimed,
            },
            "parsed_hours_numeric": {
                "technical": tech,
                "ethics_or_act_rules": ethics,
                "self_study": self_study,
                "project_management": proj_mgmt,
                "computed_total": computed_total,
            },
            "self_study_documentation": {
                "has_dates": bool_true(doc.has_dates),
                "has_hours_claimed": bool_true(doc.has_hours_claimed),
                "has_topic_goals": bool_true(doc.has_topic_goals),
                "has_resources_used": bool_true(doc.has_resources_used),
                "has_outcome_summary": bool_true(doc.has_outcome_summary),
                "snippets": {
                    "dates_text": doc.dates_text,
                    "hours_text": doc.hours_text,
                    "topic_goals_text": doc.topic_goals_text,
                    "resources_text": doc.resources_text,
                    "outcome_summary_text": doc.outcome_summary_text,
                }
            }
        },
        info_type="parsed_inputs",
        info_name="computed_inputs_summary"
    )

    # Provide a final determination summary (complementary to the tree result)
    unmet: List[str] = []
    if computed_total < 15.0:
        unmet.append("Total PDH < 15")
    if ethics < 1.0:
        unmet.append("Ethics/Act-Rules hours < 1")
    if self_study > 5.0:
        unmet.append("Self-directed study hours exceed 5")
    if self_study > 0.0 and not doc_all_present:
        unmet.append("Self-directed study documentation incomplete")

    determination = "Compliant for renewal" if len(unmet) == 0 else "Non-compliant for renewal"
    evaluator.add_custom_info(
        info={
            "determination": determination,
            "unmet_requirements": unmet,
            "notes": "This determination mirrors the verification tree outcome; all critical checks must pass."
        },
        info_type="determination"
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
    Evaluate the answer for Texas PE CE renewal compliance.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured CE information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ce(),
        template_class=CEExtraction,
        extraction_name="ce_extraction"
    )

    # Add baseline ground-truth requirement summary for context
    evaluator.add_ground_truth({
        "requirements": {
            "min_total_pdh": 15,
            "min_ethics_or_act_rules": 1,
            "max_self_study_pdh": 5,
            "self_study_doc_required_fields": [
                "dates", "hours_claimed", "topic_goals", "resources_used", "outcome_summary"
            ]
        }
    }, gt_type="texas_tbpe_requirements")

    # Build the rubric tree and perform checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()