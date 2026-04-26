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
TASK_ID = "fcc_nors_wireless_requirements"
TASK_DESCRIPTION = (
    "What are the complete FCC Network Outage Reporting System (NORS) requirements for wireless telecommunications "
    "carriers in the United States when they experience network outages? Your answer must include: (1) all threshold "
    "criteria that determine whether an outage is reportable, including specific numerical thresholds; (2) all timeline "
    "requirements for submitting notifications and reports to the FCC; (3) the methodology for calculating potentially "
    "affected users in switch failures; (4) all requirements for notifying 911 and 988 special facilities, including "
    "timelines, methods, and required content elements; and (5) a description of the material information elements that "
    "must be included in special facility notifications."
)

# Ground-truth style reference list for the special-facility material information elements.
REQUIRED_MATERIAL_INFO_ELEMENTS = [
    "unique identifier",
    "contact information",
    "provider name",
    "incident date/time",
    "affected service types",
    "geographic area",
    "impact statement",
    "restoration estimate",
    "cause",
    "notification type",
]

# Synonym map used to judge element coverage (best‑effort, not exhaustive).
REQUIRED_MATERIAL_INFO_SYNONYMS = {
    "unique identifier": ["unique identifier", "incident id", "ticket id", "tracking id", "reference id", "case id"],
    "contact information": ["contact", "point of contact", "poc", "contact info", "contact information", "phone", "email"],
    "provider name": ["provider name", "company name", "reporting entity", "carrier", "service provider"],
    "incident date/time": ["incident date", "incident time", "start time", "start date", "outage start", "date/time"],
    "affected service types": ["affected service types", "services affected", "service type", "voice", "sms", "text", "data", "paging"],
    "geographic area": ["geographic area", "location", "jurisdiction", "county", "counties", "state", "states", "psap area"],
    "impact statement": ["impact statement", "impact", "extent of impact", "customer impact", "functional impact"],
    "restoration estimate": ["restoration estimate", "estimated time to repair", "etr", "estimated restoration time", "eta"],
    "cause": ["cause", "root cause", "known cause", "cause of outage"],
    "notification type": ["notification type", "initial", "update", "final", "type of notification"],
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ClaimItem(BaseModel):
    present: Optional[bool] = None
    value: Optional[str] = None
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MaterialInfoExtraction(BaseModel):
    elements: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class NORSExtraction(BaseModel):
    # Threshold criteria
    duration_threshold: Optional[ClaimItem] = None
    msc_failure_criterion: Optional[ClaimItem] = None
    user_minutes_threshold: Optional[ClaimItem] = None
    oc3_minutes_threshold: Optional[ClaimItem] = None
    facility_911_criterion: Optional[ClaimItem] = None
    facility_988_criterion: Optional[ClaimItem] = None

    # NORS timelines
    nors_notification_timeline: Optional[ClaimItem] = None
    initial_report_timeline: Optional[ClaimItem] = None
    final_report_timeline: Optional[ClaimItem] = None

    # Calculation methodology
    user_calc_methodology: Optional[ClaimItem] = None

    # 911/988 special facility notifications
    notify_911_timeline: Optional[ClaimItem] = None
    notify_988_timeline: Optional[ClaimItem] = None
    followup_timeline: Optional[ClaimItem] = None
    notification_methods: Optional[ClaimItem] = None

    # Material information elements required in special facility notifications
    material_info_requirements: Optional[MaterialInfoExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nors() -> str:
    return """
Extract, from the provided answer text ONLY, structured information for each of the following FCC NORS requirements
as presented or claimed by the answer. For each item, fill:
- present: true/false (whether the answer explicitly and clearly states this requirement)
- value: for items that include a numeric/time threshold, capture the exact phrase used in the answer (e.g., "30 minutes", "900,000 user-minutes", "667 OC3-minutes", "120 minutes", "72 hours", "3 calendar days", "30 days", "2 hours").
- text: for text-based requirements (e.g., MSC criterion, notification methods), capture a concise paraphrase of what the answer claims.
- urls: all URLs that the answer explicitly cites as supporting this specific item. If the answer provides a general "Sources" section or shared references, duplicate those URLs into each relevant item's urls array. Only include actual URLs present in the answer text.

Also extract the list of "material information elements" for special facility notifications exactly as provided/enumerated in the answer, under material_info_requirements.elements
and the related urls in material_info_requirements.urls.

Items to extract (JSON keys -> expectation):
- duration_threshold -> expectation: at least 30 minutes duration for reportable outages.
- msc_failure_criterion -> expectation: MSC/switching office failures lasting ≥30 minutes are reportable regardless of user impact.
- user_minutes_threshold -> expectation: ≥ 900,000 user-minutes of telephony or paging service.
- oc3_minutes_threshold -> expectation: ≥ 667 OC3-minutes (DS3/OC-3 minutes).
- facility_911_criterion -> expectation: any outage potentially affecting a 911 special facility triggers reporting requirements.
- facility_988_criterion -> expectation: any outage potentially affecting a 988 special facility triggers reporting requirements.

- nors_notification_timeline -> expectation: NORS Notification within 120 minutes of discovering a reportable outage.
- initial_report_timeline -> expectation: Initial Communications Outage Report within 72 hours (i.e., within 3 calendar days) of discovery.
- final_report_timeline -> expectation: Final Communications Outage Report within 30 days of discovery.

- user_calc_methodology -> expectation: For switch/MSC failures, potentially affected users = disabled macro cell sites × (total users ÷ total macro cell sites).

- notify_911_timeline -> expectation: notify affected 911 special facilities as soon as possible but no later than 30 minutes after discovery.
- notify_988_timeline -> expectation: notify affected 988 special facilities as soon as possible but no later than 30 minutes after discovery.
- followup_timeline -> expectation: first follow-up notification to the special facility must be sent no later than 2 hours after the initial contact.
- notification_methods -> expectation: notify by telephone AND in writing via electronic means (e.g., email, web portal), unless mutually agreed otherwise.

- material_info_requirements -> expectation: the answer's list of elements such as unique identifier, contact info, provider name, incident date/time, affected service types, geographic area, impact statement, restoration estimate, cause, notification type.

Rules:
- Only extract what is explicitly in the answer. Do not infer or add content.
- If a URL is missing from the answer, leave urls as [].
- If an item is not mentioned, set present to false and value/text to null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_present(item: Optional[ClaimItem]) -> bool:
    return bool(item and item.present)


def _safe_urls(item: Optional[ClaimItem]) -> List[str]:
    return list(item.urls) if (item and item.urls) else []


def _safe_text(item: Optional[ClaimItem]) -> str:
    if not item:
        return ""
    return (item.value or item.text or "").strip()


def _covers_required_material_elements(elements: List[str]) -> bool:
    """
    Check whether the answer's enumerated elements cover all required categories,
    allowing for common synonyms/variants.
    """
    elements_lc = [e.lower() for e in elements]
    covered = 0
    for req, synonyms in REQUIRED_MATERIAL_INFO_SYNONYMS.items():
        found = False
        for e in elements_lc:
            if any(syn in e for syn in synonyms):
                found = True
                break
        if found:
            covered += 1
        else:
            return False
    return covered == len(REQUIRED_MATERIAL_INFO_SYNONYMS)


async def _add_exists_and_support_nodes(
    evaluator: Evaluator,
    parent,
    group_id: str,
    group_desc: str,
    present: bool,
    urls: List[str],
    normative_claim: str,
    add_ins: str,
) -> None:
    """
    Create a sequential group node per requirement:
      1) Critical existence+source presence check (custom node).
      2) Critical support check by cited URLs.
    """
    group_node = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=False,  # Non-critical at group level to allow partial credit across items
    )

    exists_node = evaluator.add_custom_node(
        result=(present and len(urls) > 0),
        id=f"{group_id}_exists",
        desc="The answer includes this requirement and cites at least one source URL",
        parent=group_node,
        critical=True,
    )

    supported_node = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc=f"{group_desc} — supported by the cited source(s)",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim=normative_claim,
        node=supported_node,
        sources=urls,
        additional_instruction=add_ins,
    )


def _additional_instruction_for(key: str) -> str:
    """
    Additional instructions to guide the verifier per item (synonyms, acceptance rules).
    """
    base = "Judge as supported only if the cited page(s) explicitly confirm this requirement. Minor paraphrases are acceptable."
    mapping = {
        "Duration_Threshold_Specified": base + " Treat '>= 30 minutes', '30 minutes or longer', or 'half an hour' as equivalent to 'at least 30 minutes'.",
        "MSC_Failure_Criterion": base + " Accept synonyms such as 'Mobile Switching Center', 'MSC', or 'switching office/MTSO'. Must be independent of user impact and for durations ≥30 minutes.",
        "User_Minute_Threshold": base + " Treat 'user-minutes', 'subscriber-minutes', or 'customer-minutes' as equivalent. Threshold must be at least 900,000.",
        "OC3_Minute_Threshold": base + " Treat 'OC-3 minutes' and 'DS3 minutes' as equivalent terms. Threshold must be at least 667.",
        "911_Facility_Criterion": base + " Accept references to '911 special facility' including PSAPs (Public Safety Answering Points).",
        "988_Facility_Criterion": base + " Accept references to '988 special facility' including the 988 Suicide & Crisis Lifeline. ",
        "NORS_Notification_Timeline": base + " This is the NORS Notification (not the Initial Report). Timeline must be within 120 minutes of discovery.",
        "Initial_Report_Timeline": base + " Treat 'within 72 hours' and 'within 3 calendar days' as equivalent.",
        "Final_Report_Timeline": base + " 'Within 30 days' of discovery is required.",
        "User_Calculation_Methodology": base + " Confirm the formula: disabled macro cell sites × (total users ÷ total macro cell sites). Minor phrasing differences allowed.",
        "911_Notification_Timeline": base + " Treat 'as soon as possible but no later than 30 minutes' as the correct standard.",
        "988_Notification_Timeline": base + " Treat 'as soon as possible but no later than 30 minutes' as the correct standard.",
        "Followup_Notification_Timeline": base + " Confirm 'first follow-up' must be no later than 2 hours after initial contact.",
        "Notification_Method_Requirements": base + " Confirm both telephone AND written (electronic) notification are required unless mutually agreed otherwise (email/web portal acceptable).",
        "Material_Information_Requirements": base + " Verify that the page lists these elements as required for special facility notifications; accept close synonyms.",
    }
    return mapping.get(key, base)


# --------------------------------------------------------------------------- #
# Main verification logic                                                     #
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
    Entry point for evaluating an answer to the FCC NORS (wireless) requirements task.
    """
    # Initialize evaluator (use parallel aggregation at the root to allow partial credit across all items)
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

    # Add a logical top-level node reflecting the rubric root
    top = evaluator.add_parallel(
        id="Complete_FCC_NORS_Wireless_Requirements",
        desc="The answer comprehensively identifies all FCC NORS reporting and notification requirements for wireless carriers experiencing network outages, covering threshold criteria, timelines, calculation methodologies, and special facility procedures.",
        parent=root,
        critical=False,
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nors(),
        template_class=NORSExtraction,
        extraction_name="nors_extraction",
    )

    # Record normative reference information
    evaluator.add_ground_truth(
        {
            "thresholds": {
                "duration_minutes_min": 30,
                "msc_failure_regardless_user_impact": True,
                "user_minutes_min": 900_000,
                "oc3_minutes_min": 667,
                "facility_911_triggers_reporting": True,
                "facility_988_triggers_reporting": True,
            },
            "timelines": {
                "nors_notification_minutes": 120,
                "initial_report_within": "72 hours (3 calendar days)",
                "final_report_within_days": 30,
                "notify_911_988_no_later_than_minutes": 30,
                "first_follow_up_no_later_than_hours": 2,
            },
            "calc_method": "disabled macro cell sites × (total users ÷ total macro cell sites)",
            "notification_methods": "telephone + written (electronic) unless mutually agreed otherwise",
            "special_facility_required_elements": REQUIRED_MATERIAL_INFO_ELEMENTS,
        },
        gt_type="normative_reference",
    )

    # Build verification groups (Sequential per item: existence+sources -> support)
    # 1) Duration threshold (>= 30 minutes)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="Duration_Threshold_Specified",
        group_desc="The answer specifies that reportable outages must last at least 30 minutes in duration",
        present=_safe_present(extracted.duration_threshold),
        urls=_safe_urls(extracted.duration_threshold),
        normative_claim="Under FCC Part 4/NORS, a reportable outage includes those lasting at least 30 minutes in duration.",
        add_ins=_additional_instruction_for("Duration_Threshold_Specified"),
    )

    # 2) MSC failure criterion (>= 30 minutes regardless of user impact)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="MSC_Failure_Criterion",
        group_desc="The answer describes that Mobile Switching Center (MSC) failures lasting at least 30 minutes must be reported regardless of user impact",
        present=_safe_present(extracted.msc_failure_criterion),
        urls=_safe_urls(extracted.msc_failure_criterion),
        normative_claim=(
            "Under FCC Part 4/NORS, a Mobile Switching Center (MSC) or switching office failure lasting at least 30 minutes "
            "is reportable regardless of the number of users impacted."
        ),
        add_ins=_additional_instruction_for("MSC_Failure_Criterion"),
    )

    # 3) User-minute threshold (>= 900,000 user-minutes)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="User_Minute_Threshold",
        group_desc="The answer specifies that outages potentially affecting at least 900,000 user-minutes of telephony or paging service must be reported",
        present=_safe_present(extracted.user_minutes_threshold),
        urls=_safe_urls(extracted.user_minutes_threshold),
        normative_claim=(
            "Under FCC Part 4/NORS, an outage is reportable when it potentially affects at least 900,000 user-minutes "
            "of telephony or paging service."
        ),
        add_ins=_additional_instruction_for("User_Minute_Threshold"),
    )

    # 4) OC3-minute threshold (>= 667 OC3-minutes)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="OC3_Minute_Threshold",
        group_desc="The answer specifies that outages affecting at least 667 OC3-minutes must be reported",
        present=_safe_present(extracted.oc3_minutes_threshold),
        urls=_safe_urls(extracted.oc3_minutes_threshold),
        normative_claim=(
            "Under FCC Part 4/NORS, an outage is reportable when it results in at least 667 OC-3 (DS3) minutes of "
            "unavailability."
        ),
        add_ins=_additional_instruction_for("OC3_Minute_Threshold"),
    )

    # 5) 911 facility criterion
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="911_Facility_Criterion",
        group_desc="The answer describes that any outage potentially affecting a 911 special facility triggers reporting requirements",
        present=_safe_present(extracted.facility_911_criterion),
        urls=_safe_urls(extracted.facility_911_criterion),
        normative_claim=(
            "Under FCC Part 4/NORS, any outage that potentially affects a 911 special facility (e.g., a PSAP) is "
            "reportable and triggers applicable reporting/notification requirements."
        ),
        add_ins=_additional_instruction_for("911_Facility_Criterion"),
    )

    # 6) 988 facility criterion
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="988_Facility_Criterion",
        group_desc="The answer describes that any outage potentially affecting a 988 special facility triggers reporting requirements",
        present=_safe_present(extracted.facility_988_criterion),
        urls=_safe_urls(extracted.facility_988_criterion),
        normative_claim=(
            "Under FCC Part 4/NORS, any outage that potentially affects a 988 special facility (the 988 Suicide & Crisis "
            "Lifeline) is reportable and triggers applicable reporting/notification requirements."
        ),
        add_ins=_additional_instruction_for("988_Facility_Criterion"),
    )

    # 7) NORS Notification within 120 minutes
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="NORS_Notification_Timeline",
        group_desc="The answer specifies that wireless providers must submit a NORS Notification within 120 minutes of discovering a reportable outage",
        present=_safe_present(extracted.nors_notification_timeline),
        urls=_safe_urls(extracted.nors_notification_timeline),
        normative_claim="Under FCC Part 4/NORS, a NORS Notification must be filed within 120 minutes of discovering a reportable outage.",
        add_ins=_additional_instruction_for("NORS_Notification_Timeline"),
    )

    # 8) Initial report within 72 hours (3 calendar days)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="Initial_Report_Timeline",
        group_desc="The answer specifies that wireless providers must submit an Initial Communications Outage Report within 72 hours (or 3 calendar days) of discovering the outage",
        present=_safe_present(extracted.initial_report_timeline),
        urls=_safe_urls(extracted.initial_report_timeline),
        normative_claim=(
            "Under FCC Part 4/NORS, the Initial Communications Outage Report must be submitted within 72 hours (i.e., "
            "within 3 calendar days) of discovery."
        ),
        add_ins=_additional_instruction_for("Initial_Report_Timeline"),
    )

    # 9) Final report within 30 days
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="Final_Report_Timeline",
        group_desc="The answer specifies that wireless providers must submit a Final Communications Outage Report within 30 days of discovering the outage",
        present=_safe_present(extracted.final_report_timeline),
        urls=_safe_urls(extracted.final_report_timeline),
        normative_claim="Under FCC Part 4/NORS, the Final Communications Outage Report must be submitted within 30 days of discovery.",
        add_ins=_additional_instruction_for("Final_Report_Timeline"),
    )

    # 10) User calculation methodology for switch/MSC failures
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="User_Calculation_Methodology",
        group_desc="The answer explains the methodology for calculating potentially affected users in switch failures",
        present=_safe_present(extracted.user_calc_methodology),
        urls=_safe_urls(extracted.user_calc_methodology),
        normative_claim=(
            "Under FCC Part 4/NORS, for switch or MSC failures the number of potentially affected users is calculated as: "
            "disabled macro cell sites × (total users ÷ total macro cell sites)."
        ),
        add_ins=_additional_instruction_for("User_Calculation_Methodology"),
    )

    # 11) 911 notification timeline (ASAP but no later than 30 minutes)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="911_Notification_Timeline",
        group_desc="The answer specifies that providers must notify affected 911 special facilities as soon as possible but no later than 30 minutes after discovering the outage",
        present=_safe_present(extracted.notify_911_timeline),
        urls=_safe_urls(extracted.notify_911_timeline),
        normative_claim=(
            "Affected 911 special facilities must be notified as soon as possible, but no later than 30 minutes after "
            "discovery of the outage."
        ),
        add_ins=_additional_instruction_for("911_Notification_Timeline"),
    )

    # 12) 988 notification timeline (ASAP but no later than 30 minutes)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="988_Notification_Timeline",
        group_desc="The answer specifies that providers must notify affected 988 special facilities as soon as possible but no later than 30 minutes after discovering the outage",
        present=_safe_present(extracted.notify_988_timeline),
        urls=_safe_urls(extracted.notify_988_timeline),
        normative_claim=(
            "Affected 988 special facilities must be notified as soon as possible, but no later than 30 minutes after "
            "discovery of the outage."
        ),
        add_ins=_additional_instruction_for("988_Notification_Timeline"),
    )

    # 13) First follow-up notification no later than 2 hours after initial contact
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="Followup_Notification_Timeline",
        group_desc="The answer specifies that the first follow-up notification to 911/988 facilities must be sent no later than 2 hours after initial contact",
        present=_safe_present(extracted.followup_timeline),
        urls=_safe_urls(extracted.followup_timeline),
        normative_claim="The first follow-up notification to the 911/988 special facility must be sent no later than 2 hours after the initial contact.",
        add_ins=_additional_instruction_for("Followup_Notification_Timeline"),
    )

    # 14) Notification method requirements (telephone AND written electronic)
    await _add_exists_and_support_nodes(
        evaluator=evaluator,
        parent=top,
        group_id="Notification_Method_Requirements",
        group_desc="The answer describes that notifications to 911/988 facilities must be transmitted by telephone and in writing via electronic means (unless mutually agreed otherwise)",
        present=_safe_present(extracted.notification_methods),
        urls=_safe_urls(extracted.notification_methods),
        normative_claim=(
            "Notifications to 911/988 special facilities must be transmitted by telephone and in writing by electronic "
            "means (e.g., email or web portal), unless different methods are mutually agreed."
        ),
        add_ins=_additional_instruction_for("Notification_Method_Requirements"),
    )

    # 15) Material information elements required in special facility notifications
    # Build a specialized sequential group with an extra coverage check.
    mat_group = evaluator.add_sequential(
        id="Material_Information_Requirements",
        desc="The answer describes the material information elements that must be included in special facility notifications",
        parent=top,
        critical=False,
    )

    mat_elements = []
    mat_urls: List[str] = []
    if extracted.material_info_requirements:
        mat_elements = list(extracted.material_info_requirements.elements or [])
        mat_urls = list(extracted.material_info_requirements.urls or [])

    # 15.1 Existence + at least one source URL
    mat_exists = evaluator.add_custom_node(
        result=(len(mat_elements) > 0 and len(mat_urls) > 0),
        id="Material_Information_Requirements_exists",
        desc="The answer enumerates material information elements and cites at least one source URL",
        parent=mat_group,
        critical=True,
    )

    # 15.2 Coverage of the required elements (best-effort string match using synonyms)
    coverage_ok = _covers_required_material_elements(mat_elements)
    mat_covered = evaluator.add_custom_node(
        result=coverage_ok,
        id="Material_Information_Requirements_elements_covered",
        desc="The answer's list covers the required material elements (unique ID, contact info, provider name, incident date/time, affected services, geography, impact, restoration estimate, cause, notification type)",
        parent=mat_group,
        critical=True,
    )

    # 15.3 Supported by the cited sources
    mat_supported = evaluator.add_leaf(
        id="Material_Information_Requirements_supported",
        desc="Material information element requirements are supported by the cited source(s)",
        parent=mat_group,
        critical=True,
    )
    required_list_str = "; ".join(REQUIRED_MATERIAL_INFO_ELEMENTS)
    mat_claim = (
        "FCC rules/guidance require that special facility notifications include at least the following material "
        f"information elements: {required_list_str}."
    )
    await evaluator.verify(
        claim=mat_claim,
        node=mat_supported,
        sources=mat_urls,
        additional_instruction=_additional_instruction_for("Material_Information_Requirements"),
    )

    # Optionally record custom info for debugging/traceability
    evaluator.add_custom_info(
        {
            "extracted_material_elements": mat_elements,
            "material_elements_coverage": "pass" if coverage_ok else "fail",
        },
        info_type="analysis",
        info_name="material_elements_check",
    )

    return evaluator.get_summary()