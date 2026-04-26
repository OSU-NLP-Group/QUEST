import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_wireless_outage_jan_2026"
TASK_DESCRIPTION = (
    "In January 2026, one of the major US wireless telecommunications carriers experienced the largest nationwide "
    "service outage of the year, affecting over 500,000 users according to outage monitoring platforms. Provide a "
    "comprehensive incident report that documents the following specific aspects of this service disruption:\n\n"
    "1. The exact date (specific day in January 2026) when the outage began\n"
    "2. The name of the wireless carrier that experienced the outage\n"
    "3. The technical cause category (specify if it was a hardware failure, software issue, cyberattack, natural disaster, or other)\n"
    "4. The specific network infrastructure component or system that failed\n"
    "5. The approximate duration of the service disruption for most users\n"
    "6. The approximate number or range of outage reports logged on Downdetector at peak\n"
    "7. Any geographic location (city or state) mentioned as the source or center of the technical failure\n"
    "8. The monetary compensation amount offered to affected customers by the carrier\n"
    "9. The types of wireless services that were disrupted (specify voice calls, SMS/text, mobile data, or combinations)\n"
    "10. Whether other major wireless carriers experienced simultaneous minor service disruptions\n"
    "11. When the carrier announced full service restoration (provide date/time)\n"
    "12. Whether the carrier publicly attributed the cause to internal technical issues or external factors\n"
    "13. Any regulatory body or FCC response mentioned in connection with the incident\n"
    "14. How this outage's severity was characterized in comparison to other recent carrier outages\n\n"
    "For each detail, provide the specific factual information and include supporting reference URLs that verify the information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DetailWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OutageIncidentExtraction(BaseModel):
    outage_date: Optional[DetailWithSources] = None
    carrier_identification: Optional[DetailWithSources] = None
    cause_category: Optional[DetailWithSources] = None
    failed_infrastructure: Optional[DetailWithSources] = None
    disruption_duration: Optional[DetailWithSources] = None
    report_volume: Optional[DetailWithSources] = None
    geographic_location: Optional[DetailWithSources] = None
    compensation_amount: Optional[DetailWithSources] = None
    affected_service_types: Optional[DetailWithSources] = None
    other_carriers: Optional[DetailWithSources] = None
    restoration_announcement: Optional[DetailWithSources] = None
    attribution: Optional[DetailWithSources] = None
    regulatory_response: Optional[DetailWithSources] = None
    severity_comparison: Optional[DetailWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_incident_report() -> str:
    return """
    Extract the incident details for the January 2026 major US wireless carrier outage from the provided answer.
    For each of the following 14 items, return a JSON object with exactly two fields:
    - value: the factual content as stated in the answer (string). If a list is mentioned (e.g., multiple service types), combine into a single string (comma-separated).
    - sources: an array of all supporting reference URLs cited in the answer for that item. Include only actual URLs (plain or markdown), no prose. If none are provided, return an empty array.

    Items to extract:
    1) outage_date: The specific date (day in January 2026) when the outage began.
    2) carrier_identification: The name of the major US wireless carrier that experienced the outage.
    3) cause_category: Categorize the technical cause strictly as one of: "hardware failure", "software issue", "cyberattack", "natural disaster", or "other". If the answer uses synonyms (e.g., "routing error", "database bug", "fiber cut", "DDoS"), map them to the closest category.
    4) failed_infrastructure: The specific network infrastructure component or system that failed (e.g., core router, authentication server/HSS, IMS/VoLTE subsystem, DNS, fiber backhaul).
    5) disruption_duration: Approximate duration (e.g., "about 6 hours", "several hours", "half a day").
    6) report_volume: Approximate number or range of outage reports at peak on Downdetector (e.g., "over 500,000", "around 520k").
    7) geographic_location: City or state mentioned as the source or center of the technical failure (if any).
    8) compensation_amount: Monetary credit/compensation offered to affected customers (e.g., "$5 bill credit").
    9) affected_service_types: Service types disrupted (e.g., "voice calls, SMS, mobile data"; combine into a single string).
    10) other_carriers: Whether other major carriers experienced simultaneous minor disruptions; summarize as "yes" or "no" (optionally add brief detail in the same value string).
    11) restoration_announcement: Date/time when full service restoration was announced.
    12) attribution: Whether the carrier attributed the cause to internal technical issues or external factors; keep the value concise (e.g., "internal systems", "external vendor", "cyberattack").
    13) regulatory_response: Any regulatory body or FCC response mentioned (e.g., "FCC investigation opened", "NORS report filed").
    14) severity_comparison: How this outage’s severity was characterized compared to other recent outages (e.g., "largest of 2026", "worst since 2020").

    STRICT RULES:
    - Do not invent information; extract only what the answer explicitly states.
    - For each item, include all URLs cited in the answer that support the item in the 'sources' array.
    - If any item is missing in the answer, set its 'value' to null and 'sources' to [] for that item.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_str(val: Optional[str]) -> str:
    return val if isinstance(val, str) else ""


def yes_no_from_text(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    v = val.strip().lower()
    if any(x in v for x in ["yes", "true", "yep", "affirmative"]):
        return True
    if any(x in v for x in ["no", "false", "nope", "negative"]):
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_detail(
    evaluator: Evaluator,
    parent_node,
    field_id: str,
    field_desc: str,
    detail: Optional[DetailWithSources],
    claim_builder: Callable[[Optional[str]], str],
    additional_instruction: str,
    critical: bool,
) -> None:
    """
    Create a sequential node for one incident detail:
    - Existence check (value AND at least one source)
    - URL-backed verification of the claim
    """
    seq_node = evaluator.add_sequential(
        id=field_id,
        desc=field_desc,
        parent=parent_node,
        critical=critical
    )

    value_present = bool(detail and detail.value and detail.value.strip())
    sources_present = bool(detail and detail.sources and len(detail.sources) > 0)

    # Existence check (precondition)
    evaluator.add_custom_node(
        result=(value_present and sources_present),
        id=f"{field_id}_provided",
        desc=f"{field_desc} - value and sources provided",
        parent=seq_node,
        critical=critical  # Children of a critical node must be critical if parent is critical
    )

    # Source-supported verification
    verify_node = evaluator.add_leaf(
        id=f"{field_id}_supported",
        desc=field_desc,
        parent=seq_node,
        critical=critical
    )

    claim_text = claim_builder(detail.value if detail else None)
    sources_list = detail.sources if (detail and detail.sources) else []

    await evaluator.verify(
        claim=claim_text,
        node=verify_node,
        sources=sources_list,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Incident-specific verification orchestration                                #
# --------------------------------------------------------------------------- #
async def build_incident_verifications(
    evaluator: Evaluator,
    root_node,
    extracted: OutageIncidentExtraction
) -> None:
    incident_node = evaluator.add_parallel(
        id="incident_documentation",
        desc="Answer provides comprehensive factual documentation of the January 2026 major US wireless carrier service outage",
        parent=root_node,
        critical=False
    )

    # 1) Outage Date (CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="outage_date",
        field_desc="Answer provides the specific date (day and month) in January 2026 when the outage occurred",
        detail=extracted.outage_date,
        claim_builder=lambda v: f"The outage began on {safe_str(v)} (in January 2026).",
        additional_instruction="Verify the onset date/time falls within January 2026. Allow timezone differences, but confirm the start date using the provided sources.",
        critical=True
    )

    # 2) Carrier Identification (CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="carrier_identification",
        field_desc="Answer identifies which major US wireless carrier experienced the outage",
        detail=extracted.carrier_identification,
        claim_builder=lambda v: f"The carrier that experienced the outage was {safe_str(v)}.",
        additional_instruction="Confirm the named entity is a major US wireless carrier and is the one reported to have experienced the January 2026 nationwide outage.",
        critical=True
    )

    # 3) Cause Category (CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="cause_category",
        field_desc="Answer correctly categorizes the technical cause as hardware failure, software issue, cyberattack, natural disaster, or other type",
        detail=extracted.cause_category,
        claim_builder=lambda v: f"The technical cause category was '{safe_str(v)}'.",
        additional_instruction=(
            "Map synonyms to categories: routing/database/firmware/DNS/IMS/VoLTE bugs/issues -> software issue; "
            "fiber cut/hardware malfunction/power equipment failure -> hardware failure; "
            "DDoS/intrusion/ransomware -> cyberattack; "
            "storm/hurricane/earthquake/wildfire/flood -> natural disaster; "
            "If none fit clearly -> other. Verify the category assignment is explicitly supported by sources."
        ),
        critical=True
    )

    # 4) Failed Infrastructure (CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="failed_infrastructure",
        field_desc="Answer identifies the specific network infrastructure component or system that failed",
        detail=extracted.failed_infrastructure,
        claim_builder=lambda v: f"The failed infrastructure component or system was {safe_str(v)}.",
        additional_instruction=(
            "Verify that sources explicitly mention the specific component/system (e.g., core router, HLR/HSS, IMS/VoLTE, DNS, fiber backhaul, radio base stations, authentication server)."
        ),
        critical=True
    )

    # 5) Disruption Duration (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="disruption_duration",
        field_desc="Answer provides the approximate duration or time period of the service disruption",
        detail=extracted.disruption_duration,
        claim_builder=lambda v: f"The service disruption lasted approximately {safe_str(v)} for most users.",
        additional_instruction="Accept reasonable approximations or ranges derived from start/end times stated by sources.",
        critical=False
    )

    # 6) Report Volume (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="report_volume",
        field_desc="Answer provides an approximate number or range for outage reports submitted to monitoring services",
        detail=extracted.report_volume,
        claim_builder=lambda v: f"At peak, Downdetector recorded approximately {safe_str(v)} outage reports for the carrier.",
        additional_instruction="Ensure the number/range is specifically tied to Downdetector or similar monitoring platform. Allow approximate wording like 'around' or 'over'.",
        critical=False
    )

    # 7) Geographic Location (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="geographic_location",
        field_desc="Answer provides a geographic location (city or state) associated with the technical failure source",
        detail=extracted.geographic_location,
        claim_builder=lambda v: f"The source or center of the technical failure was associated with {safe_str(v)}.",
        additional_instruction="Confirm that sources cite a city/state or specific locality relevant as the origin or center of the issue, if any.",
        critical=False
    )

    # 8) Compensation Amount (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="compensation_amount",
        field_desc="Answer specifies the monetary value of compensation or credit offered to affected customers",
        detail=extracted.compensation_amount,
        claim_builder=lambda v: f"The carrier offered affected customers compensation/credit of {safe_str(v)}.",
        additional_instruction="Verify the amount via official carrier statements or credible reporting.",
        critical=False
    )

    # 9) Affected Service Types (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="affected_service_types",
        field_desc="Answer specifies which service types were disrupted (voice, text, data, or combination)",
        detail=extracted.affected_service_types,
        claim_builder=lambda v: f"The following wireless services were disrupted: {safe_str(v)}.",
        additional_instruction="Allow synonyms and common phrasing: voice=calling/VoLTE; text=SMS; data=mobile data/internet.",
        critical=False
    )

    # 10) Other Carriers (NON-CRITICAL)
    def other_carriers_claim(v: Optional[str]) -> str:
        flag = yes_no_from_text(v)
        if flag is True:
            return "Other major carriers experienced simultaneous minor service disruptions."
        if flag is False:
            return "No other major carriers experienced simultaneous minor service disruptions."
        # If ambiguous, state as the provided text but still verify
        return f"Statement regarding other carriers: {safe_str(v)}."

    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="other_carriers",
        field_desc="Answer indicates whether other wireless carriers experienced related or simultaneous issues",
        detail=extracted.other_carriers,
        claim_builder=other_carriers_claim,
        additional_instruction="Check sources for mentions of other carriers (e.g., Verizon, AT&T, T-Mobile, etc.) reporting concurrent minor issues.",
        critical=False
    )

    # 11) Restoration Announcement (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="restoration_announcement",
        field_desc="Answer provides timing information about when service restoration was announced",
        detail=extracted.restoration_announcement,
        claim_builder=lambda v: f"The carrier announced full service restoration on {safe_str(v)}.",
        additional_instruction="Verify the announcement timestamp/date; accept reasonable timezone differences.",
        critical=False
    )

    # 12) Attribution (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="attribution",
        field_desc="Answer specifies whether the carrier attributed the cause to internal systems or external factors",
        detail=extracted.attribution,
        claim_builder=lambda v: f"The carrier publicly attributed the cause to {safe_str(v)}.",
        additional_instruction="Confirm attribution statements (internal systems vs external vendor vs cyberattack vs natural disaster).",
        critical=False
    )

    # 13) Regulatory Response (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="regulatory_response",
        field_desc="Answer indicates any regulatory body or FCC response mentioned in connection with the outage",
        detail=extracted.regulatory_response,
        claim_builder=lambda v: f"Regulatory/FCC response mentioned: {safe_str(v)}.",
        additional_instruction="Look for mentions of the FCC, NORS filings, investigations, or formal inquiries.",
        critical=False
    )

    # 14) Severity Comparison (NON-CRITICAL)
    await verify_detail(
        evaluator=evaluator,
        parent_node=incident_node,
        field_id="severity_comparison",
        field_desc="Answer provides information characterizing this outage's severity compared to other recent outages",
        detail=extracted.severity_comparison,
        claim_builder=lambda v: f"This outage was characterized as {safe_str(v)} compared to other recent carrier outages.",
        additional_instruction="Verify phrases like 'largest of the year', 'one of the worst', or comparisons to prior notable outages using the provided sources.",
        critical=False
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
    Evaluate an answer for the January 2026 US wireless outage incident report.
    """
    # Initialize evaluator (root parallel aggregation)
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

    # Extract incident details from the answer
    extracted_incident = await evaluator.extract(
        prompt=prompt_extract_incident_report(),
        template_class=OutageIncidentExtraction,
        extraction_name="incident_report_extraction",
    )

    # Optional: Record simple custom info about extraction completeness
    def count_provided(detail: Optional[DetailWithSources]) -> int:
        return int(bool(detail and detail.value)) + int(bool(detail and detail.sources))

    completeness = {
        "outage_date_fields_present": count_provided(extracted_incident.outage_date),
        "carrier_identification_fields_present": count_provided(extracted_incident.carrier_identification),
        "cause_category_fields_present": count_provided(extracted_incident.cause_category),
        "failed_infrastructure_fields_present": count_provided(extracted_incident.failed_infrastructure),
        "disruption_duration_fields_present": count_provided(extracted_incident.disruption_duration),
        "report_volume_fields_present": count_provided(extracted_incident.report_volume),
        "geographic_location_fields_present": count_provided(extracted_incident.geographic_location),
        "compensation_amount_fields_present": count_provided(extracted_incident.compensation_amount),
        "affected_service_types_fields_present": count_provided(extracted_incident.affected_service_types),
        "other_carriers_fields_present": count_provided(extracted_incident.other_carriers),
        "restoration_announcement_fields_present": count_provided(extracted_incident.restoration_announcement),
        "attribution_fields_present": count_provided(extracted_incident.attribution),
        "regulatory_response_fields_present": count_provided(extracted_incident.regulatory_response),
        "severity_comparison_fields_present": count_provided(extracted_incident.severity_comparison),
    }
    evaluator.add_custom_info(completeness, info_type="extraction_completeness")

    # Build verification tree and run checks
    await build_incident_verifications(evaluator, root, extracted_incident)

    # Return structured summary
    return evaluator.get_summary()