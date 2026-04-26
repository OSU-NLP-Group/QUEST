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
TASK_ID = "fcc_outage_reporting_requirements"
TASK_DESCRIPTION = (
    "A major U.S. wireless carrier experiences a network outage that lasts 2 hours, potentially affects "
    "1.5 million user-minutes of telephony service, and impacts 911 calling capabilities. According to FCC regulations, "
    "what are the specific reporting requirements and timelines that this carrier must follow? Your answer must include: "
    "(1) The minimum duration threshold (in minutes) for an outage to be reportable to the FCC, "
    "(2) The minimum user impact threshold (in user-minutes) for an outage to be reportable, "
    "(3) The maximum time allowed (in hours or days) to submit the initial outage report to the FCC after discovering the outage, "
    "(4) The maximum time allowed (in days) to submit the final outage report to the FCC after discovering the outage, "
    "(5) The maximum time allowed (in minutes) to notify affected Public Safety Answering Points (PSAPs) after discovering a 911-affecting outage, "
    "(6) The required frequency (in hours) for providing follow-up notifications to PSAPs after the initial notification, "
    "(7) The name of the electronic system through which outage reports must be submitted to the FCC, "
    "(8) An explanation of what status indicator affected customers would see on their phones during this outage and what it means."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FCCOutageRequirementsExtraction(BaseModel):
    # Extract exactly what the answer claims for each required item
    duration_threshold_minutes: Optional[str] = None
    user_impact_threshold_user_minutes: Optional[str] = None
    initial_report_timeline: Optional[str] = None  # e.g., "72 hours", "3 days"
    final_report_timeline: Optional[str] = None    # e.g., "30 days"
    psap_notification_timeline_minutes: Optional[str] = None  # e.g., "30 minutes"
    followup_notification_frequency_hours: Optional[str] = None  # e.g., "2 hours"
    reporting_system_name: Optional[str] = None  # e.g., "NORS"
    sos_mode_explanation: Optional[str] = None

    # Any URLs the answer cites (for reference; may be empty)
    sources_general: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract from the answer the specific values or names it provides for each of the following fields. Return the values exactly as stated in the answer (use strings; do not normalize units).
    Also extract all URLs explicitly cited in the answer as 'sources_general'.

    Fields to extract:
    - duration_threshold_minutes: the minimum outage duration threshold (in minutes) that makes an outage reportable to the FCC (e.g., "30 minutes", ">= 30 minutes").
    - user_impact_threshold_user_minutes: the minimum user impact threshold (in user-minutes) that makes an outage reportable (e.g., "900,000 user-minutes", "0.9 million user-minutes").
    - initial_report_timeline: the maximum time allowed to submit the initial outage report after discovery (e.g., "72 hours", "3 calendar days").
    - final_report_timeline: the maximum time allowed to submit the final outage report after discovery (e.g., "30 days").
    - psap_notification_timeline_minutes: the maximum time allowed to notify affected PSAPs after discovering a 911-affecting outage (e.g., "30 minutes").
    - followup_notification_frequency_hours: the frequency for follow-up notifications to PSAPs after the initial notification (e.g., "every 2 hours", "2 hours").
    - reporting_system_name: the name of the electronic system through which outage reports must be submitted to the FCC (e.g., "Network Outage Reporting System", "NORS").
    - sos_mode_explanation: the explanation given for the 'SOS' or 'SOS Only' indicator on phones during an outage.

    Additionally:
    - sources_general: a list of all URLs explicitly mentioned in the answer (include full URLs; do not invent any).
    If any field is not stated in the answer, set it to null. If there are no URLs, return an empty list for sources_general.
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
def _common_additional_instruction() -> str:
    return (
        "Judge against the provided answer text. Pass only if the answer explicitly includes this exact regulatory "
        "requirement using an equivalent value or wording. Allow reasonable equivalents: "
        "— 72 hours == 3 calendar days; "
        "— 900,000 user-minutes == 0.9 million user-minutes; "
        "— '>=' or 'at least' wording; "
        "— 'every two hours' == 'every 2 hours'. "
        "Ignore formatting or casing differences. Do not require citations; focus on whether the answer states the requirement."
    )


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    # Initialize evaluator (root is parallel as per rubric)
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

    # Extraction (record what the answer explicitly states)
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=FCCOutageRequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # Build leaf nodes according to rubric (all critical)
    # 1) Minimum duration threshold (30 minutes)
    node_duration = evaluator.add_leaf(
        id="duration_threshold",
        desc="Answer correctly identifies that the outage must last at least 30 minutes to trigger FCC reporting requirements",
        parent=root,
        critical=True,
    )

    # 2) Minimum user impact threshold (900,000 user-minutes)
    node_user_impact = evaluator.add_leaf(
        id="user_impact_threshold",
        desc="Answer correctly identifies that the outage must potentially affect at least 900,000 user-minutes of telephony service",
        parent=root,
        critical=True,
    )

    # 3) Initial report within 72 hours (3 days) of discovery
    node_initial = evaluator.add_leaf(
        id="initial_report_timeline",
        desc="Answer correctly identifies that the initial outage report must be submitted within 72 hours (or 3 calendar days) after discovering the outage",
        parent=root,
        critical=True,
    )

    # 4) Final report within 30 days of discovery
    node_final = evaluator.add_leaf(
        id="final_report_timeline",
        desc="Answer correctly identifies that the final outage report must be submitted within 30 days after discovering the outage",
        parent=root,
        critical=True,
    )

    # 5) PSAP notification within 30 minutes of discovering 911-affecting outage
    node_psap_notify = evaluator.add_leaf(
        id="psap_notification_timeline",
        desc="Answer correctly identifies that affected PSAPs must be notified within 30 minutes of discovering a 911-affecting outage",
        parent=root,
        critical=True,
    )

    # 6) Follow-up notifications to PSAPs every 2 hours after initial
    node_psap_followup = evaluator.add_leaf(
        id="followup_notification_frequency",
        desc="Answer correctly identifies that follow-up notifications to PSAPs must be provided every 2 hours after the initial notification",
        parent=root,
        critical=True,
    )

    # 7) Reporting system name is NORS
    node_reporting_system = evaluator.add_leaf(
        id="reporting_system",
        desc="Answer correctly identifies that reports must be submitted through the FCC's Network Outage Reporting System (NORS)",
        parent=root,
        critical=True,
    )

    # 8) SOS mode explanation on phones
    node_sos = evaluator.add_leaf(
        id="sos_mode_explanation",
        desc="Answer correctly explains what SOS mode means on phones during the outage (not connected to home network but can make emergency calls)",
        parent=root,
        critical=True,
    )

    # Prepare claims focused on what the ANSWER states (simple verification against the answer text)
    claims_and_sources = [
        (
            # Duration ≥ 30 minutes
            "The answer explicitly states that the minimum duration threshold for a reportable FCC outage is 30 minutes (i.e., outages lasting at least 30 minutes are reportable).",
            None,
            node_duration,
            _common_additional_instruction(),
        ),
        (
            # User-minutes ≥ 900,000
            "The answer explicitly states that the minimum user impact threshold for FCC reportability is 900,000 user-minutes (e.g., 0.9 million user-minutes).",
            None,
            node_user_impact,
            _common_additional_instruction(),
        ),
        (
            # Initial report within 72 hours / 3 days
            "The answer correctly states that the initial outage report to the FCC must be submitted within 72 hours (3 calendar days) after discovery of the outage.",
            None,
            node_initial,
            _common_additional_instruction(),
        ),
        (
            # Final report within 30 days
            "The answer correctly states that the final outage report to the FCC must be submitted within 30 days after discovery of the outage.",
            None,
            node_final,
            _common_additional_instruction(),
        ),
        (
            # PSAP notify within 30 minutes
            "The answer correctly states that affected PSAPs must be notified within 30 minutes of discovering a 911-affecting outage.",
            None,
            node_psap_notify,
            _common_additional_instruction(),
        ),
        (
            # PSAP follow-ups every 2 hours
            "The answer correctly states that follow-up notifications to PSAPs must be provided every 2 hours after the initial notification.",
            None,
            node_psap_followup,
            _common_additional_instruction(),
        ),
        (
            # Reporting system: NORS
            "The answer correctly identifies that outage reports must be filed through the FCC's Network Outage Reporting System (NORS).",
            None,
            node_reporting_system,
            _common_additional_instruction(),
        ),
        (
            # SOS explanation
            "The answer explains that 'SOS' or 'SOS Only' on the phone means the device is not connected to its carrier's network for normal service but can still place emergency calls (e.g., 911), potentially via any available network or satellite.",
            None,
            node_sos,
            "Judge whether this explanation is clearly present in the answer. Allow equivalent phrasing conveying the same meaning.",
        ),
    ]

    # Run verifications in parallel
    await evaluator.batch_verify(claims_and_sources)

    # Optionally add ground truth info for transparency (normative expectations)
    evaluator.add_ground_truth({
        "expected_values": {
            "duration_threshold_minutes": "30 minutes",
            "user_impact_threshold_user_minutes": "900,000 user-minutes",
            "initial_report_timeline": "72 hours (3 calendar days) after discovery",
            "final_report_timeline": "30 days after discovery",
            "psap_notification_timeline_minutes": "30 minutes after discovery for 911-affecting outages",
            "followup_notification_frequency_hours": "every 2 hours after initial notification",
            "reporting_system_name": "NORS (Network Outage Reporting System)",
            "sos_mode_explanation": "Phone shows 'SOS'/'SOS Only' meaning no normal carrier service but emergency calls are still possible",
        }
    })

    # Return evaluation summary
    return evaluator.get_summary()