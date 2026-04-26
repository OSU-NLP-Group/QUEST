import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_multicriteria"
TASK_DESCRIPTION = (
    "Identify a public school superintendent in the United States who meets ALL of the following criteria: "
    "(1) Currently serves as superintendent of a school district located in a U.S. state that experienced public school "
    "enrollment decline between fall 2019 and fall 2023; "
    "(2) Was appointed or began serving as superintendent between January 1, 2023, and December 31, 2024; "
    "(3) Leads a district with at least one high school that has participated in state football championship games between 2020 and 2025; "
    "(4) The district maintains a graduation rate of at least 80%; "
    "(5) The district offers Advanced Placement (AP) courses; "
    "(6) The district offers dual enrollment programs. "
    "Provide the superintendent's full name, the school district name, and the state."
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    superintendent_full_name: Optional[str] = None
    district_name: Optional[str] = None
    state: Optional[str] = None

    # Appointment timeframe evidence
    appointment_date_text: Optional[str] = None
    appointment_urls: List[str] = Field(default_factory=list)

    # State enrollment decline evidence
    enrollment_decline_urls: List[str] = Field(default_factory=list)

    # Athletics (state football championship participation)
    high_school_names: List[str] = Field(default_factory=list)
    athletics_urls: List[str] = Field(default_factory=list)

    # Graduation rate (>= 80%)
    graduation_rate_text: Optional[str] = None
    graduation_rate_urls: List[str] = Field(default_factory=list)

    # AP courses offering
    ap_info_text: Optional[str] = None
    ap_urls: List[str] = Field(default_factory=list)

    # Dual enrollment offering
    dual_enrollment_info_text: Optional[str] = None
    dual_enrollment_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return """
    You must extract information ONLY from the provided answer text. Do not invent or infer anything not explicitly present in the answer. 
    If an item is missing in the answer, return null for that field (or an empty list for list fields).

    Extract a single superintendent candidate that the answer proposes to meet the criteria. If the answer mentions multiple, extract only the FIRST complete candidate.

    Return a JSON object with the following fields:

    1) superintendent_full_name: The full name of the superintendent (string).
    2) district_name: The official name of the school district (string).
    3) state: The U.S. state of the district (string, e.g., "Texas", "Ohio", "New York").
    
    Appointment timeframe evidence:
    4) appointment_date_text: The appointment/start date text as provided (string; can be a date like "July 1, 2023" or "January 2024"; do NOT normalize).
    5) appointment_urls: List of URLs explicitly cited in the answer for the appointment/start date or current superintendent status (list of strings; if none provided, return an empty list).

    State enrollment decline evidence:
    6) enrollment_decline_urls: List of URLs explicitly cited in the answer that support that the state experienced public school enrollment decline between fall 2019 and fall 2023.

    Athletics (state football championship participation 2020-2025):
    7) high_school_names: List of one or more high school names in the district that the answer claims participated in a state football championship game between 2020 and 2025 inclusive.
    8) athletics_urls: List of URLs explicitly cited in the answer to support the championship participation.

    Graduation rate (>= 80%):
    9) graduation_rate_text: The graduation rate described in the answer (string; may include a percent or phrase like "82% in 2023").
    10) graduation_rate_urls: List of URLs explicitly cited in the answer for the graduation rate.

    AP offering:
    11) ap_info_text: The text stating AP courses are offered (string; do not invent).
    12) ap_urls: List of URLs explicitly cited in the answer that support AP course offerings (district or high school pages are acceptable).

    Dual enrollment offering:
    13) dual_enrollment_info_text: The text stating dual enrollment programs are offered (string; do not invent).
    14) dual_enrollment_urls: List of URLs explicitly cited in the answer that support dual enrollment offering.

    IMPORTANT URL RULES:
    - Extract only actual URLs explicitly present in the answer. If a URL is missing protocol, prepend http://
    - Accept both plain URLs and markdown links; extract the actual URL target.
    - If the answer references a source without a URL, do not invent one; return an empty list for that field.

    If multiple URLs are provided, include all of them in the corresponding list (do not deduplicate).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_string(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip())


def _non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _pick_first(items: List[str]) -> str:
    return items[0] if items else ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extracted: SuperintendentExtraction,
    parent_root,
) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    All verification leaves are binary and critical under the overall node to enforce ALL criteria.
    """
    # Create an overall critical node (since evaluator root is non-critical by design)
    overall = evaluator.add_parallel(
        id="overall",
        desc="Identify a superintendent meeting all specified criteria",
        parent=parent_root,
        critical=True,
    )

    # 1) Superintendent Identity (critical leaf)
    evaluator.add_custom_node(
        result=_non_empty_string(extracted.superintendent_full_name),
        id="Superintendent_Identity",
        desc="The superintendent is correctly identified with full name",
        parent=overall,
        critical=True
    )

    # 2) District Identity (critical leaf)
    evaluator.add_custom_node(
        result=_non_empty_string(extracted.district_name),
        id="District_Identity",
        desc="The school district name is correctly provided",
        parent=overall,
        critical=True
    )

    # 3) State Location (critical leaf)
    evaluator.add_custom_node(
        result=_non_empty_string(extracted.state),
        id="State_Location",
        desc="The state where the district is located is correctly identified",
        parent=overall,
        critical=True
    )

    # 4) Geographic Compliance: state enrollment decline between fall 2019 and fall 2023
    geo_node = evaluator.add_parallel(
        id="Geographic_Compliance",
        desc="District is in a state with enrollment decline",
        parent=overall,
        critical=True
    )

    # 4.a) Enrollment Decline Verification (critical leaf with URL verification)
    decline_verify_leaf = evaluator.add_leaf(
        id="Enrollment_Decline_Verification",
        desc="The state experienced public school enrollment decline between fall 2019 and fall 2023",
        parent=geo_node,
        critical=True
    )
    decline_state = extracted.state or "the stated state"
    decline_claim = (
        f"The U.S. state of {decline_state} experienced a net decline in public school enrollment "
        f"between fall 2019 and fall 2023."
    )
    await evaluator.verify(
        claim=decline_claim,
        node=decline_verify_leaf,
        sources=extracted.enrollment_decline_urls,
        additional_instruction=(
            "Verify using the provided sources whether statewide public school enrollment decreased from 2019 to 2023. "
            "Accept state-level official statistics or reputable reporting; minor wording differences are okay. "
            "If sources are irrelevant, invalid, or do not support the decline, judge as not supported."
        )
    )

    # 4.b) Enrollment Decline Reference present (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.enrollment_decline_urls),
        id="Enrollment_Decline_Reference",
        desc="Provides URL reference supporting enrollment decline data",
        parent=geo_node,
        critical=True
    )

    # 5) Appointment Timeframe (Jan 1, 2023 – Dec 31, 2024)
    appt_node = evaluator.add_parallel(
        id="Appointment_Timeframe",
        desc="Superintendent appointment was between January 2023 and December 2024",
        parent=overall,
        critical=True
    )

    # 5.a) Appointment Date within timeframe (critical, verify by URLs)
    appt_leaf = evaluator.add_leaf(
        id="Appointment_Date",
        desc="The appointment or start date falls within the specified timeframe",
        parent=appt_node,
        critical=True
    )
    sup_name = extracted.superintendent_full_name or "the identified superintendent"
    district = extracted.district_name or "the identified district"
    appt_claim = (
        f"{sup_name} was appointed or began serving as superintendent of {district} between "
        f"January 1, 2023 and December 31, 2024."
    )
    await evaluator.verify(
        claim=appt_claim,
        node=appt_leaf,
        sources=extracted.appointment_urls,
        additional_instruction=(
            "Confirm from the provided source(s) that the appointment/start date for this superintendent falls in 2023 or 2024 (inclusive). "
            "Titles like 'interim' are acceptable if the role is superintendent. If the source contradicts or lacks such date evidence, judge as not supported."
        )
    )

    # 5.b) Appointment Reference present (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.appointment_urls),
        id="Appointment_Reference",
        desc="Provides URL reference for appointment information",
        parent=appt_node,
        critical=True
    )

    # 6) Athletic Program: state football championship participation (sequential)
    athletic_node = evaluator.add_sequential(
        id="Athletic_Program",
        desc="District has high school(s) with football championship participation",
        parent=overall,
        critical=True
    )

    # 6.a) High School identified (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.high_school_names) if isinstance(extracted.high_school_names, list) else False,
        id="High_School_Identification",
        desc="At least one high school in the district is identified",
        parent=athletic_node,
        critical=True
    )

    # 6.b) Championship participation 2020–2025 (critical, verify by URLs)
    champ_leaf = evaluator.add_leaf(
        id="Championship_Participation",
        desc="The high school participated in state football championship competition between 2020-2025",
        parent=athletic_node,
        critical=True
    )
    hs = _pick_first(extracted.high_school_names) if isinstance(extracted.high_school_names, list) else ""
    champ_claim = (
        f"The high school '{hs}' from {district} participated in a state football championship game between 2020 and 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=champ_claim,
        node=champ_leaf,
        sources=extracted.athletics_urls,
        additional_instruction=(
            "Check if the provided source(s) indicate that the named high school reached and played in a state football championship game "
            "(final/title game) at any point from 2020 through 2025 inclusive. Synonyms like 'state title game' or 'state final' count. "
            "Participation as runner-up in the final is acceptable. If only semifinals or earlier rounds are shown, judge as not supported."
        )
    )

    # 6.c) Athletic reference present (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.athletics_urls),
        id="Athletic_Reference",
        desc="Provides URL reference for championship participation",
        parent=athletic_node,
        critical=True
    )

    # 7) Graduation Rate (>= 80%)
    grad_node = evaluator.add_parallel(
        id="Graduation_Rate",
        desc="District maintains required graduation rate",
        parent=overall,
        critical=True
    )

    # 7.a) Rate threshold (critical, verify by URLs)
    rate_leaf = evaluator.add_leaf(
        id="Rate_Threshold",
        desc="Four-year graduation rate is at least 80%",
        parent=grad_node,
        critical=True
    )
    rate_claim = (
        f"The school district {district} has a four-year high school graduation rate of at least 80% (based on the latest reported year in the source)."
    )
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=extracted.graduation_rate_urls,
        additional_instruction=(
            "Use the provided source(s) to verify the district-level four-year graduation rate is >= 80%. "
            "If multiple years are shown, accept any clearly reported year with >= 80%. If only school-level data is present, "
            "accept if it clearly indicates the district overall meets >= 80%. If unclear or below 80%, judge as not supported."
        )
    )

    # 7.b) Graduation reference present (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.graduation_rate_urls),
        id="Graduation_Reference",
        desc="Provides URL reference for graduation rate data",
        parent=grad_node,
        critical=True
    )

    # 8) AP Program
    ap_node = evaluator.add_parallel(
        id="AP_Program",
        desc="District offers Advanced Placement courses",
        parent=overall,
        critical=True
    )

    # 8.a) AP availability (critical, verify by URLs)
    ap_leaf = evaluator.add_leaf(
        id="AP_Availability",
        desc="AP courses are offered in the district",
        parent=ap_node,
        critical=True
    )
    ap_claim = (
        f"The school district {district} offers Advanced Placement (AP) courses (either district-wide or at least at one of its high schools)."
    )
    await evaluator.verify(
        claim=ap_claim,
        node=ap_leaf,
        sources=extracted.ap_urls,
        additional_instruction=(
            "Accept official district pages, high school course catalogs, or other reputable references showing AP courses offered "
            "for schools within the district. If the sources do not clearly indicate AP is offered, judge as not supported."
        )
    )

    # 8.b) AP reference present (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.ap_urls),
        id="AP_Reference",
        desc="Provides URL reference for AP program information",
        parent=ap_node,
        critical=True
    )

    # 9) Dual Enrollment
    dual_node = evaluator.add_parallel(
        id="Dual_Enrollment",
        desc="District offers dual enrollment programs",
        parent=overall,
        critical=True
    )

    # 9.a) Dual enrollment availability (critical, verify by URLs)
    dual_leaf = evaluator.add_leaf(
        id="Dual_Enrollment_Availability",
        desc="Dual enrollment programs are offered",
        parent=dual_node,
        critical=True
    )
    dual_claim = (
        f"The school district {district} offers dual enrollment programs (e.g., partnerships with colleges for students to earn college credit)."
    )
    await evaluator.verify(
        claim=dual_claim,
        node=dual_leaf,
        sources=extracted.dual_enrollment_urls,
        additional_instruction=(
            "Accept official district or school pages, or reputable sources that clearly indicate dual enrollment or early college "
            "programs are offered to district students. If unclear or absent, judge as not supported."
        )
    )

    # 9.b) Dual enrollment reference present (critical)
    evaluator.add_custom_node(
        result=_non_empty_urls(extracted.dual_enrollment_urls),
        id="Dual_Enrollment_Reference",
        desc="Provides URL reference for dual enrollment program",
        parent=dual_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the superintendent multi-criteria task.
    Returns a structured summary with a verification tree and final score.
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
        default_model=model
    )

    # Extraction
    extracted: SuperintendentExtraction = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction"
    )

    # Optional: add concise ground-truth-like requirement mirror for context
    evaluator.add_ground_truth({
        "requirements": {
            "state_enrollment_decline_window": "Fall 2019 to Fall 2023",
            "appointment_timeframe": "2023-01-01 to 2024-12-31",
            "football_championship_window": "2020 to 2025 inclusive",
            "min_graduation_rate": ">= 80%",
            "ap_offered": True,
            "dual_enrollment_offered": True
        }
    }, gt_type="criteria_spec")

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted, root)

    # Return structured result
    return evaluator.get_summary()