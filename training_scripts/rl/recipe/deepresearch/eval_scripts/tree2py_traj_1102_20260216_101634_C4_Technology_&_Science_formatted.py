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
TASK_ID = "fcc_nors_verizon_2026"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon Wireless experienced a major network outage that began around 12:30 PM Eastern Time "
    "and lasted approximately 10 hours, affecting over 1.5 million customers across major U.S. cities including New York, "
    "Washington D.C., Chicago, Boston, and Atlanta. The outage was caused by a software issue and left many customers' phones in SOS-only mode.\n\n"
    "Based on FCC regulations, analyze the Network Outage Reporting System (NORS) requirements that apply to this incident. Your analysis should include:\n\n"
    "1. Verification that this outage met the FCC's threshold criteria for mandatory NORS reporting\n"
    "2. The timeframe within which Verizon must submit the initial notification to FCC NORS after discovering the outage\n"
    "3. The timeframe within which Verizon must submit the Initial Communications Outage Report\n"
    "4. The timeframe within which Verizon must submit the Final Communications Outage Report\n"
    "5. If 911 or 988 special facilities were potentially affected, the notification requirements for those facilities\n"
    "6. A reference URL to the official FCC regulation or NORS page documenting these wireless carrier reporting requirements\n"
    "7. A reference URL to a credible source documenting the January 14, 2026 Verizon outage details"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FCCRequirementsExtraction(BaseModel):
    # Core NORS reporting requirements extracted from the answer
    initial_notification_deadline: Optional[str] = None            # e.g., "within 120 minutes" or "within 2 hours"
    initial_report_deadline: Optional[str] = None                  # e.g., "within 72 hours" or "within 3 calendar days"
    final_report_deadline: Optional[str] = None                    # e.g., "within 30 days"
    special_facility_notification: Optional[str] = None            # e.g., "notify within 30 minutes; first follow-up within 2 hours"

    # References
    fcc_regulation_url: Optional[str] = None                       # Official FCC regulation or NORS page
    outage_reference_url: Optional[str] = None                     # Credible source documenting Verizon outage details

    # Threshold reasoning/explanation
    threshold_met_reasoning: Optional[str] = None                  # Explanation that outage met NORS thresholds (duration/user-minutes/etc.)

    # Outage detail values (optional, if answer provides them)
    outage_start_time: Optional[str] = None                        # e.g., "around 12:30 PM ET"
    outage_duration: Optional[str] = None                          # e.g., "approximately 10 hours"
    affected_customers: Optional[str] = None                       # e.g., "over 1.5 million customers"
    outage_cause: Optional[str] = None                             # e.g., "software issue"
    affected_cities: List[str] = Field(default_factory=list)       # e.g., ["New York", "Washington D.C.", "Chicago", "Boston", "Atlanta"]


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the FCC NORS reporting analysis details as stated in the answer. Return a JSON object with the following fields:

    1) initial_notification_deadline: The stated timeframe within which wireless carriers must submit the initial NORS notification after discovering a reportable outage (e.g., "within 120 minutes", "within 2 hours"). If not present, return null.
    2) initial_report_deadline: The stated timeframe within which the Initial Communications Outage Report must be submitted after discovering the outage (e.g., "within 72 hours", "within 3 calendar days"). If not present, return null.
    3) final_report_deadline: The stated timeframe within which the Final Communications Outage Report must be submitted after discovering the outage (e.g., "within 30 days"). If not present, return null.
    4) special_facility_notification: If the answer mentions 911 or 988 special facilities notification requirements, extract the stated requirement text (e.g., "notify within 30 minutes; first follow-up within 2 hours"). If not present, return null.

    5) fcc_regulation_url: The URL to the official FCC regulation or FCC NORS page documenting wireless carrier reporting requirements (e.g., a 47 CFR § 4.9 page or an FCC NORS program page on fcc.gov). Extract only URLs explicitly present in the answer. If absent, return null.
    6) outage_reference_url: A URL to a credible source documenting the January 14, 2026 Verizon outage details. Extract only URLs explicitly present in the answer. If absent, return null.

    7) threshold_met_reasoning: Extract the answer's explanation for why this outage met FCC mandatory NORS reporting thresholds (e.g., based on duration ≥ 30 minutes, ≥ 900,000 user-minutes, MSC impacts, or 911/988 impacts). If absent, return null.

    Also attempt to extract the key outage details if provided in the answer:
    8) outage_start_time: The stated start time (e.g., "around 12:30 PM Eastern Time"). If absent, return null.
    9) outage_duration: The stated duration (e.g., "approximately 10 hours"). If absent, return null.
    10) affected_customers: The stated number of affected customers (e.g., "over 1.5 million"). If absent, return null.
    11) outage_cause: The stated cause (e.g., "software issue"). If absent, return null.
    12) affected_cities: A list of city names mentioned (e.g., ["New York", "Washington D.C.", "Chicago", "Boston", "Atlanta"]). If absent, return an empty array.

    IMPORTANT:
    - Extract only information explicitly present in the answer. Do not invent or infer.
    - For URLs, extract the full URL. If missing protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: FCCRequirementsExtraction) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Root analysis node
    analysis_node = evaluator.add_parallel(
        id="FCC_NORS_Reporting_Analysis",
        desc="Analysis of FCC NORS reporting requirements applicable to the January 14, 2026 Verizon wireless network outage",
        parent=evaluator.root,
        critical=False  # Allow partial credit overall
    )

    # ------------------------ FCC Regulation Reference --------------------- #
    fcc_ref_node = evaluator.add_parallel(
        id="FCC_Regulation_Reference",
        desc="Provision of a URL reference to the official FCC regulation (47 CFR § 4.9) or FCC NORS page that documents wireless carrier reporting requirements",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.fcc_regulation_url and extracted.fcc_regulation_url.strip()),
        id="FCC_Regulation_Reference_provided",
        desc="The answer provides an FCC regulation or NORS URL",
        parent=fcc_ref_node,
        critical=True
    )
    fcc_ref_leaf = evaluator.add_leaf(
        id="FCC_Regulation_Reference_official",
        desc="The provided URL is an official FCC page documenting NORS reporting requirements (47 CFR §4.9 or FCC NORS program page)",
        parent=fcc_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official FCC page that documents wireless carrier reporting requirements for NORS or 47 CFR § 4.9.",
        node=fcc_ref_leaf,
        sources=extracted.fcc_regulation_url,
        additional_instruction="Accept official pages on fcc.gov, including 47 CFR § 4.* regulation pages or official NORS program guidance pages."
    )

    # ------------------------ Outage Details Reference --------------------- #
    outage_ref_node = evaluator.add_parallel(
        id="Outage_Details_Reference",
        desc="Provision of a URL reference to a credible source documenting the January 14, 2026 Verizon outage details including start time, duration, affected users, and cause",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.outage_reference_url and extracted.outage_reference_url.strip()),
        id="Outage_Details_Reference_provided",
        desc="The answer provides a credible outage details URL",
        parent=outage_ref_node,
        critical=True
    )

    # Date and start time
    outage_dt_leaf = evaluator.add_leaf(
        id="Outage_Details_DateTime",
        desc="Outage page documents the date (January 14, 2026) and start time around 12:30 PM Eastern Time",
        parent=outage_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page reports that Verizon experienced a network outage on January 14, 2026 that began around 12:30 PM Eastern Time.",
        node=outage_dt_leaf,
        sources=extracted.outage_reference_url,
        additional_instruction="Allow minor variations in phrasing and minute-level rounding (e.g., 12:25–12:35 PM ET)."
    )

    # Duration
    outage_duration_leaf = evaluator.add_leaf(
        id="Outage_Details_Duration",
        desc="Outage page documents that the outage lasted approximately 10 hours",
        parent=outage_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page reports the outage lasted approximately 10 hours.",
        node=outage_duration_leaf,
        sources=extracted.outage_reference_url,
        additional_instruction="Allow moderate rounding around the 10-hour figure."
    )

    # Affected users
    outage_users_leaf = evaluator.add_leaf(
        id="Outage_Details_AffectedUsers",
        desc="Outage page documents that over 1.5 million customers were affected",
        parent=outage_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page reports that over 1.5 million customers were affected.",
        node=outage_users_leaf,
        sources=extracted.outage_reference_url,
        additional_instruction="The claim should be clearly supported; allow minor numeric variation if the source states 'more than 1.5 million'."
    )

    # Cause
    outage_cause_leaf = evaluator.add_leaf(
        id="Outage_Details_Cause",
        desc="Outage page documents that the cause was a software issue",
        parent=outage_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page reports the outage was caused by a software issue.",
        node=outage_cause_leaf,
        sources=extracted.outage_reference_url,
        additional_instruction="Accept phrases like 'software fault', 'software bug', or 'software-related issue'."
    )

    # ------------------------ Threshold Met -------------------------------- #
    threshold_node = evaluator.add_parallel(
        id="Threshold_Met",
        desc="Verification that the outage met FCC NORS reporting thresholds based on duration ≥ 30 minutes or ≥ 900,000 user-minutes",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.threshold_met_reasoning and extracted.threshold_met_reasoning.strip()),
        id="Threshold_Met_provided",
        desc="The answer explains why the outage met FCC thresholds",
        parent=threshold_node,
        critical=True
    )
    threshold_leaf = evaluator.add_leaf(
        id="Threshold_Met_correct",
        desc="The outage met FCC NORS reporting thresholds",
        parent=threshold_node,
        critical=True
    )
    # Logical verification using the context (answer + task) and relying on verified references as prerequisites
    await evaluator.verify(
        claim=("Given the reported outage lasted approximately 10 hours (well over 30 minutes) and affected over 1.5 million customers, "
               "the incident met FCC NORS mandatory reporting thresholds (e.g., duration ≥ 30 minutes or ≥ 900,000 user‑minutes)."),
        node=threshold_leaf,
        sources=None,  # Logical check; gated by prerequisites
        additional_instruction="Use simple logical reasoning based on the stated duration and affected users to determine threshold eligibility.",
        extra_prerequisites=[fcc_ref_node, outage_ref_node]
    )

    # ------------------------ Initial Notification Deadline ---------------- #
    init_notify_node = evaluator.add_parallel(
        id="Initial_Notification_Deadline",
        desc="Identification of the FCC requirement that wireless carriers must submit initial NORS notification within 120 minutes of discovering a reportable outage",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.initial_notification_deadline and extracted.initial_notification_deadline.strip()),
        id="Initial_Notification_Deadline_provided",
        desc="The answer provides the initial NORS notification timeframe",
        parent=init_notify_node,
        critical=True
    )
    init_notify_leaf = evaluator.add_leaf(
        id="Initial_Notification_Deadline_accurate",
        desc="Initial NORS notification must be submitted within 120 minutes (2 hours) of discovery",
        parent=init_notify_node,
        critical=True
    )
    await evaluator.verify(
        claim="Wireless carriers must submit the initial NORS notification within 120 minutes (2 hours) of discovering a reportable outage.",
        node=init_notify_leaf,
        sources=extracted.fcc_regulation_url,
        additional_instruction="Validate against FCC rules; allow 'within two hours' phrasing as equivalent."
    )

    # ------------------------ Initial Report Deadline ---------------------- #
    init_report_node = evaluator.add_parallel(
        id="Initial_Report_Deadline",
        desc="Identification of the FCC requirement that an Initial Communications Outage Report must be submitted within 72 hours (3 calendar days) after discovering the outage",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.initial_report_deadline and extracted.initial_report_deadline.strip()),
        id="Initial_Report_Deadline_provided",
        desc="The answer provides the Initial Communications Outage Report timeframe",
        parent=init_report_node,
        critical=True
    )
    init_report_leaf = evaluator.add_leaf(
        id="Initial_Report_Deadline_accurate",
        desc="Initial Communications Outage Report must be submitted within 72 hours (3 calendar days) after discovery",
        parent=init_report_node,
        critical=True
    )
    await evaluator.verify(
        claim="An Initial Communications Outage Report must be submitted within 72 hours (3 calendar days) after discovering the outage.",
        node=init_report_leaf,
        sources=extracted.fcc_regulation_url,
        additional_instruction="Confirm the timeframe on the FCC regulation or NORS guidance page."
    )

    # ------------------------ Final Report Deadline ------------------------ #
    final_report_node = evaluator.add_parallel(
        id="Final_Report_Deadline",
        desc="Identification of the FCC requirement that a Final Communications Outage Report must be submitted within 30 days after discovering the outage",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.final_report_deadline and extracted.final_report_deadline.strip()),
        id="Final_Report_Deadline_provided",
        desc="The answer provides the Final Communications Outage Report timeframe",
        parent=final_report_node,
        critical=True
    )
    final_report_leaf = evaluator.add_leaf(
        id="Final_Report_Deadline_accurate",
        desc="Final Communications Outage Report must be submitted within 30 days after discovery",
        parent=final_report_node,
        critical=True
    )
    await evaluator.verify(
        claim="A Final Communications Outage Report must be submitted within 30 days after discovering the outage.",
        node=final_report_leaf,
        sources=extracted.fcc_regulation_url,
        additional_instruction="Confirm the Final report deadline on the FCC regulation or NORS guidance page."
    )

    # ------------------------ Special Facility Notification ---------------- #
    special_fac_node = evaluator.add_parallel(
        id="Special_Facility_Notification",
        desc="Identification of the FCC requirement that if 911 or 988 special facilities are potentially affected, providers must notify the facility within 30 minutes of discovery, with first follow-up within 2 hours",
        parent=analysis_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.special_facility_notification and extracted.special_facility_notification.strip()),
        id="Special_Facility_Notification_provided",
        desc="The answer provides the 911/988 special facility notification requirements",
        parent=special_fac_node,
        critical=True
    )
    special_fac_leaf = evaluator.add_leaf(
        id="Special_Facility_Notification_accurate",
        desc="If 911 or 988 are potentially affected, notify within 30 minutes of discovery; first follow-up within 2 hours",
        parent=special_fac_node,
        critical=True
    )
    await evaluator.verify(
        claim="If 911 or 988 special facilities are potentially affected, providers must notify the facility within 30 minutes of discovery, and provide the first follow-up within 2 hours.",
        node=special_fac_leaf,
        sources=extracted.fcc_regulation_url,
        additional_instruction="Verify this requirement on the official FCC NORS guidance or the corresponding rule (47 CFR § 4.*)."
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
    Evaluate an answer for FCC NORS reporting requirements related to the January 14, 2026 Verizon outage.
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

    # Extract structured details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=FCCRequirementsExtraction,
        extraction_name="fcc_nors_requirements"
    )

    # Add canonical ground truth info for timelines (for reference in summary)
    evaluator.add_ground_truth({
        "canonical_timelines": {
            "initial_notification": "within 120 minutes (2 hours)",
            "initial_report": "within 72 hours (3 calendar days)",
            "final_report": "within 30 days",
            "special_facility": "notify within 30 minutes; first follow-up within 2 hours"
        },
        "notes": "Ground truth timelines are commonly referenced in FCC NORS guidance and 47 CFR § 4.*."
    })

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()