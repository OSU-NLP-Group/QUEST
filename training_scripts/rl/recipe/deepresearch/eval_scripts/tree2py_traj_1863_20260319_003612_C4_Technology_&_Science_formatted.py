import asyncio
import logging
from typing import Optional, List, Dict, Any

from urllib.parse import urlparse
from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_nors_requirements"
TASK_DESCRIPTION = (
    "A wireless telecommunications provider has experienced a network outage that lasted 45 minutes and potentially affected "
    "1.2 million user minutes of telephony service. As a compliance officer, you need to identify the complete set of FCC "
    "Network Outage Reporting System (NORS) reporting requirements applicable to this situation. Provide the following "
    "information with official FCC source references: (1) What is the minimum outage duration required for this outage to "
    "be reportable under FCC rules? (2) What is the minimum user impact threshold (in user minutes) that makes this outage "
    "reportable? (3) Within how many minutes of discovering the reportable outage must the wireless provider submit the "
    "initial NORS notification? (4) Within how many calendar days of discovering the outage must the provider submit the "
    "initial outage report? (5) Within how many days of discovering the outage must the provider submit the final report? "
    "(6) What is the confidentiality status of data submitted to NORS? All timing requirements, thresholds, and procedures "
    "must be supported by official FCC documentation."
)

# Ground truth (expected rules under FCC Part 4 / NORS for wireless providers)
EXPECTED_MIN_OUTAGE_DURATION = "30 minutes"           # At least 30 minutes
EXPECTED_USER_IMPACT_THRESHOLD = "900,000 user minutes"  # At least 900,000 user minutes
EXPECTED_INITIAL_NOTIFICATION_MINUTES = "120 minutes"  # Notification within 120 minutes of discovery
EXPECTED_INITIAL_REPORT_DAYS = "3 calendar days"      # Initial report within 3 calendar days of discovery
EXPECTED_FINAL_REPORT_DAYS = "30 days"                # Final report within 30 days of discovery
EXPECTED_CONFIDENTIALITY_STATUS = "presumed confidential"  # NORS submissions are presumed confidential


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def is_fcc_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return False
        return netloc == "fcc.gov" or netloc.endswith(".fcc.gov")
    except Exception:
        return False


def dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def filter_fcc_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if is_fcc_url(u)]


def coalesce_sources(primary: List[str], fallback: List[str]) -> List[str]:
    # Prefer primary FCC URLs; if none, fallback to FCC URLs from global references
    primary_fcc = filter_fcc_urls(primary or [])
    if primary_fcc:
        return dedupe_preserve_order(primary_fcc)
    return dedupe_preserve_order(filter_fcc_urls(fallback or []))


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NORSRequirementsExtraction(BaseModel):
    # Core answers (as stated in the agent's answer; keep as strings to be robust)
    min_outage_duration: Optional[str] = None
    user_impact_threshold: Optional[str] = None
    initial_notification_timeline: Optional[str] = None
    initial_report_timeline: Optional[str] = None
    final_report_timeline: Optional[str] = None
    confidentiality_status: Optional[str] = None

    # Item-level source URLs mentioned in the answer (ideally official FCC)
    duration_source_urls: List[str] = Field(default_factory=list)
    user_impact_source_urls: List[str] = Field(default_factory=list)
    notification_source_urls: List[str] = Field(default_factory=list)
    initial_report_source_urls: List[str] = Field(default_factory=list)
    final_report_source_urls: List[str] = Field(default_factory=list)
    confidentiality_source_urls: List[str] = Field(default_factory=list)

    # Any general reference URLs the answer cites (may include duplicates/mixtures)
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nors_requirements() -> str:
    return """
    Extract the FCC NORS reporting requirements for the scenario described, strictly from the provided answer.

    You must extract the following fields exactly as the answer states them (do NOT infer):
    - min_outage_duration: The minimum outage duration that makes an outage reportable under FCC rules (e.g., "30 minutes", "at least 30 minutes").
    - user_impact_threshold: The minimum user impact threshold (in user minutes) that makes an outage reportable (e.g., "900,000 user minutes", "≥ 900,000 user minutes").
    - initial_notification_timeline: How long (in minutes) from discovery a provider has to submit the initial NORS notification (e.g., "120 minutes").
    - initial_report_timeline: Within how many calendar days from discovery the provider must submit the initial outage report (e.g., "3 calendar days").
    - final_report_timeline: Within how many days from discovery the provider must submit the final report (e.g., "30 days").
    - confidentiality_status: The confidentiality status of data submitted to NORS (e.g., "presumed confidential", "confidential").

    Also extract any URLs that the answer cites as support for each specific item (only include URLs explicitly present in the answer):
    - duration_source_urls
    - user_impact_source_urls
    - notification_source_urls
    - initial_report_source_urls
    - final_report_source_urls
    - confidentiality_source_urls

    Additionally, extract any general cited URLs into:
    - reference_urls

    Special URL rules:
    - Extract only valid URLs actually present in the answer (plain or markdown). Do NOT invent or infer URLs.
    - Include full URLs with protocol. If missing protocol, prepend "http://".
    - Official FCC sources are on the fcc.gov domain (including subdomains like docs.fcc.gov, ecfsapi.fcc.gov, www.fcc.gov). Still extract any non-FCC URLs if present, but do not add any URLs not present in the answer.

    If any field is missing in the answer text, set it to null (or empty list for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_requirement_item(
    evaluator: Evaluator,
    parent_node,
    *,
    base_id: str,
    item_desc: str,
    extracted_value: Optional[str],
    expected_value: str,
    item_urls: List[str],
    global_reference_urls: List[str],
    support_claim_text: str,
    value_equivalence_guidance: str,
) -> None:
    """
    Create a critical parallel node for one requirement with:
    - value_present (custom)
    - value_correct (simple_verify)
    - sources_present (custom)
    - source_support (verify_by_urls)

    All leaves are critical to enforce correctness and source support.
    """
    node = evaluator.add_parallel(
        id=base_id,
        desc=item_desc,
        parent=parent_node,
        critical=True
    )

    # 1) Check the value is actually stated in the answer
    value_present = evaluator.add_custom_node(
        result=bool(extracted_value and str(extracted_value).strip()),
        id=f"{base_id}_value_present",
        desc=f"{item_desc} - value is explicitly stated in the answer",
        parent=node,
        critical=True
    )

    # 2) Verify the stated value matches the expected value
    value_correct_leaf = evaluator.add_leaf(
        id=f"{base_id}_value_correct",
        desc=f"{item_desc} - stated value matches expected rule",
        parent=node,
        critical=True
    )
    # Formulate a concise claim that references the answer's stated value and the expected standard
    stated = extracted_value or ""
    claim_value = (
        f"The answer states this value as '{stated}'. Verify that this is equivalent to the correct FCC requirement: "
        f"'{expected_value}'."
    )
    await evaluator.verify(
        claim=claim_value,
        node=value_correct_leaf,
        additional_instruction=(
            "Judge only the equivalence between what the answer states and the expected requirement. "
            f"Guidance: {value_equivalence_guidance}. Allow formatting and wording variations "
            "(e.g., 'thirty minutes', '>= 30 minutes', '3 days', '3 calendar days', etc.)."
        )
    )

    # 3) Ensure we have official FCC URLs to verify against
    sources_to_use = coalesce_sources(item_urls, global_reference_urls)
    sources_present = evaluator.add_custom_node(
        result=len(sources_to_use) > 0,
        id=f"{base_id}_sources_present",
        desc=f"{item_desc} - official FCC source URL(s) are provided",
        parent=node,
        critical=True
    )

    # 4) Verify the claim against the FCC sources
    source_support_leaf = evaluator.add_leaf(
        id=f"{base_id}_source_support",
        desc=f"{item_desc} - FCC source(s) explicitly support this requirement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=support_claim_text,
        node=source_support_leaf,
        sources=sources_to_use,
        additional_instruction=(
            "Use ONLY the provided webpage(s). The claim must be explicitly stated or clearly implied on an official FCC "
            "page (fcc.gov domain, including subdomains such as docs.fcc.gov). Allow reasonable wording differences "
            "but ensure the requirement is unambiguous. If the pages are irrelevant or do not contain the requirement, "
            "mark as not supported."
        ),
        extra_prerequisites=[sources_present, value_present]
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for FCC NORS reporting requirements for a wireless provider outage.
    """
    # Initialize evaluator/root
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

    # Create top-level critical requirements node (since all sub-requirements are mandatory)
    reqs_node = evaluator.add_parallel(
        id="FCC_NORS_Reporting_Requirements",
        desc="Complete identification of FCC NORS reporting requirements for wireless providers experiencing a reportable network outage",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nors_requirements(),
        template_class=NORSRequirementsExtraction,
        extraction_name="nors_requirements_extraction"
    )

    # Record ground truth/expected values for transparency
    evaluator.add_ground_truth(
        {
            "expected_min_outage_duration": EXPECTED_MIN_OUTAGE_DURATION,
            "expected_user_impact_threshold": EXPECTED_USER_IMPACT_THRESHOLD,
            "expected_initial_notification_timeline": EXPECTED_INITIAL_NOTIFICATION_MINUTES,
            "expected_initial_report_timeline": EXPECTED_INITIAL_REPORT_DAYS,
            "expected_final_report_timeline": EXPECTED_FINAL_REPORT_DAYS,
            "expected_confidentiality_status": EXPECTED_CONFIDENTIALITY_STATUS,
        },
        gt_type="expected_requirements"
    )

    # Consolidate all reference URLs mentioned in the answer (for global checks and fallback)
    all_references_union = dedupe_preserve_order(
        (extracted.reference_urls or [])
        + (extracted.duration_source_urls or [])
        + (extracted.user_impact_source_urls or [])
        + (extracted.notification_source_urls or [])
        + (extracted.initial_report_source_urls or [])
        + (extracted.final_report_source_urls or [])
        + (extracted.confidentiality_source_urls or [])
    )
    all_references_fcc = filter_fcc_urls(all_references_union)

    # Build per-requirement verifications (each as a critical parallel group)

    # 1) Minimum Outage Duration (≥ 30 minutes)
    await verify_requirement_item(
        evaluator,
        reqs_node,
        base_id="Minimum_Outage_Duration",
        item_desc="Minimum outage duration threshold (at least 30 minutes)",
        extracted_value=extracted.min_outage_duration,
        expected_value=EXPECTED_MIN_OUTAGE_DURATION,
        item_urls=extracted.duration_source_urls or [],
        global_reference_urls=all_references_fcc,
        support_claim_text=(
            "Under FCC Part 4 (NORS) rules for wireless providers, an outage is reportable only if it lasts at least "
            "30 minutes (i.e., a minimum duration of 30 minutes)."
        ),
        value_equivalence_guidance="Treat '30 minutes', 'thirty minutes', '>= 30 minutes', or 'at least 30 minutes' as equivalent."
    )

    # 2) User Impact Threshold (≥ 900,000 user minutes)
    await verify_requirement_item(
        evaluator,
        reqs_node,
        base_id="User_Impact_Threshold",
        item_desc="Minimum user impact threshold (at least 900,000 user minutes)",
        extracted_value=extracted.user_impact_threshold,
        expected_value=EXPECTED_USER_IMPACT_THRESHOLD,
        item_urls=extracted.user_impact_source_urls or [],
        global_reference_urls=all_references_fcc,
        support_claim_text=(
            "Under FCC Part 4 (NORS) rules for wireless providers, an outage is reportable only if it potentially affects "
            "at least 900,000 user minutes of telephony service."
        ),
        value_equivalence_guidance="Treat '900,000 user minutes', '≥ 900,000 user minutes', or 'at least 900,000 user minutes' as equivalent."
    )

    # 3) Initial Notification Timeline (within 120 minutes of discovery)
    await verify_requirement_item(
        evaluator,
        reqs_node,
        base_id="Initial_Notification_Timeline",
        item_desc="Initial NORS notification timeline (within 120 minutes of discovery)",
        extracted_value=extracted.initial_notification_timeline,
        expected_value=EXPECTED_INITIAL_NOTIFICATION_MINUTES,
        item_urls=extracted.notification_source_urls or [],
        global_reference_urls=all_references_fcc,
        support_claim_text=(
            "Service providers (including wireless) experiencing a reportable outage must submit an initial NORS "
            "notification within 120 minutes of discovering the outage."
        ),
        value_equivalence_guidance="Treat '120 minutes', 'within 120 minutes', or 'no later than 120 minutes' as equivalent."
    )

    # 4) Initial Report Timeline (within 3 calendar days of discovery)
    await verify_requirement_item(
        evaluator,
        reqs_node,
        base_id="Initial_Report_Timeline",
        item_desc="Initial outage report timeline (within 3 calendar days of discovery)",
        extracted_value=extracted.initial_report_timeline,
        expected_value=EXPECTED_INITIAL_REPORT_DAYS,
        item_urls=extracted.initial_report_source_urls or [],
        global_reference_urls=all_references_fcc,
        support_claim_text=(
            "Service providers must file an initial NORS outage report within 3 calendar days of discovering the outage."
        ),
        value_equivalence_guidance="Treat '3 calendar days', 'within three calendar days', or 'within 3 days (calendar)' as equivalent."
    )

    # 5) Final Report Timeline (within 30 days of discovery)
    await verify_requirement_item(
        evaluator,
        reqs_node,
        base_id="Final_Report_Timeline",
        item_desc="Final outage report timeline (within 30 days of discovery)",
        extracted_value=extracted.final_report_timeline,
        expected_value=EXPECTED_FINAL_REPORT_DAYS,
        item_urls=extracted.final_report_source_urls or [],
        global_reference_urls=all_references_fcc,
        support_claim_text=(
            "Service providers must submit a final NORS outage report within 30 days of discovering the outage."
        ),
        value_equivalence_guidance="Treat '30 days', 'within thirty days', or 'within 30 days' as equivalent."
    )

    # 6) NORS Data Confidentiality (presumed confidential)
    await verify_requirement_item(
        evaluator,
        reqs_node,
        base_id="NORS_Data_Confidentiality",
        item_desc="Confidentiality of NORS data (presumed confidential)",
        extracted_value=extracted.confidentiality_status,
        expected_value=EXPECTED_CONFIDENTIALITY_STATUS,
        item_urls=extracted.confidentiality_source_urls or [],
        global_reference_urls=all_references_fcc,
        support_claim_text=(
            "Information submitted to the FCC's Network Outage Reporting System (NORS) is presumed confidential."
        ),
        value_equivalence_guidance="Treat 'presumed confidential', 'confidential by default', or equivalent phrasing as equivalent."
    )

    # 7) Global Reference URLs validation (official FCC sources provided)
    # Build a dedicated critical parallel node to enforce URL provenance quality
    refs_node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Provides valid reference URLs from official FCC sources (fcc.gov domain) that support the identified requirements",
        parent=reqs_node,
        critical=True
    )

    # 7.1) Overall FCC references present
    refs_present_leaf = evaluator.add_custom_node(
        result=len(all_references_fcc) > 0,
        id="Reference_URLs_present",
        desc="At least one official FCC reference URL (fcc.gov) is provided in the answer",
        parent=refs_node,
        critical=True
    )

    # 7.2) All provided reference URLs are from FCC domain
    # Note: Evaluate only the explicit 'reference_urls' field for this check to ensure claimed references are official FCC sources.
    all_global_refs_are_fcc = all(is_fcc_url(u) for u in (extracted.reference_urls or [])) if (extracted.reference_urls or []) else False
    refs_official_leaf = evaluator.add_custom_node(
        result=all_global_refs_are_fcc,
        id="Reference_URLs_official_fcc",
        desc="All cited reference URLs listed under 'reference_urls' are on the official FCC domain (fcc.gov)",
        parent=refs_node,
        critical=True
    )

    # 7.3) Each of the six requirement items has at least one FCC source URL (direct or via global fallback)
    per_item_sources_ok = all([
        len(coalesce_sources(extracted.duration_source_urls or [], all_references_fcc)) > 0,
        len(coalesce_sources(extracted.user_impact_source_urls or [], all_references_fcc)) > 0,
        len(coalesce_sources(extracted.notification_source_urls or [], all_references_fcc)) > 0,
        len(coalesce_sources(extracted.initial_report_source_urls or [], all_references_fcc)) > 0,
        len(coalesce_sources(extracted.final_report_source_urls or [], all_references_fcc)) > 0,
        len(coalesce_sources(extracted.confidentiality_source_urls or [], all_references_fcc)) > 0,
    ])
    refs_per_item_leaf = evaluator.add_custom_node(
        result=per_item_sources_ok,
        id="Reference_URLs_each_requirement_supported",
        desc="Each requirement is supported by at least one official FCC URL (either item-specific or from global references)",
        parent=refs_node,
        critical=True
    )

    # Return structured evaluation summary
    return evaluator.get_summary()