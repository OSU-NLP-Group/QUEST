import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "ms_patch_tuesday_nov2025_zero_day"
TASK_DESCRIPTION = (
    "Identify the zero-day vulnerability that was actively exploited in Microsoft's November 2025 Patch Tuesday "
    "security update release. Provide the following information: (1) the CVE identifier, (2) the CVSS score and "
    "Microsoft's severity rating, and (3) the technical cause of the vulnerability and confirmation of its active "
    "exploitation status."
)

# Ground truth / required values for verification
REQUIRED_VALUES = {
    "cve_id": "CVE-2025-62215",
    "vulnerability_type": "Windows Kernel Elevation of Privilege",
    "cvss_score": "7.0",
    "ms_severity": "Important",
    "technical_cause": "race condition in the Windows Kernel",
    "privilege_impact": "elevation of privileges to SYSTEM level",
    "active_exploitation_required": True
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ZeroDayExtraction(BaseModel):
    """Structured extraction of zero-day info from the agent's answer."""
    cve_id: Optional[str] = None
    vulnerability_type: Optional[str] = None
    cvss_score: Optional[str] = None
    ms_severity: Optional[str] = None
    technical_cause: Optional[str] = None
    privilege_impact: Optional[str] = None
    exploitation_status: Optional[str] = None  # Text as stated, e.g., "actively exploited in the wild"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_zero_day() -> str:
    return (
        "Extract the specific details about the zero-day vulnerability described in the answer for Microsoft's "
        "November 2025 Patch Tuesday.\n"
        "Return a JSON object with the following fields, extracted exactly as written in the answer:\n"
        "1. cve_id: The CVE identifier string (e.g., 'CVE-2025-62215'). If not present, return null.\n"
        "2. vulnerability_type: The stated vulnerability type (e.g., 'Windows Kernel Elevation of Privilege'). "
        "If not present, return null.\n"
        "3. cvss_score: The CVSS score as a string exactly as written (e.g., '7.0' or '7.0/10'). If not present, return null.\n"
        "4. ms_severity: Microsoft's severity rating string (e.g., 'Important', 'Critical'). If not present, return null.\n"
        "5. technical_cause: The technical cause as described (e.g., 'race condition in the Windows Kernel'). If not present, return null.\n"
        "6. privilege_impact: The described privilege impact (e.g., 'elevation of privileges to SYSTEM level'). If not present, return null.\n"
        "7. exploitation_status: The text confirming active exploitation if stated (e.g., 'actively exploited', 'in the wild'). "
        "If not present, return null.\n"
        "Do not invent or normalize any values; copy them verbatim from the answer where possible."
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_zero_day_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: ZeroDayExtraction,
) -> None:
    """
    Build the verification tree for the zero-day information checks and run verifications.
    All checks are critical under the 'Zero_Day_Information' node as specified.
    """
    # Parent critical parallel node
    zd_node = evaluator.add_parallel(
        id="Zero_Day_Information",
        desc="Verify required information about the zero-day vulnerability in Microsoft's November 2025 Patch Tuesday release per the given constraints.",
        parent=parent_node,
        critical=True,
    )

    # Create leaf nodes (critical) per rubric
    node_cve = evaluator.add_leaf(
        id="CVE_Identifier",
        desc="Answer provides the CVE identifier and it matches the required value (CVE-2025-62215).",
        parent=zd_node,
        critical=True,
    )
    node_type = evaluator.add_leaf(
        id="Vulnerability_Type",
        desc="Answer identifies the vulnerability type as Windows Kernel Elevation of Privilege.",
        parent=zd_node,
        critical=True,
    )
    node_cvss = evaluator.add_leaf(
        id="CVSS_Score",
        desc="Answer states the CVSS score and it matches the required value (7.0).",
        parent=zd_node,
        critical=True,
    )
    node_severity = evaluator.add_leaf(
        id="Microsoft_Severity_Rating",
        desc="Answer states Microsoft's severity rating and it matches the required value ('Important', not 'Critical').",
        parent=zd_node,
        critical=True,
    )
    node_active = evaluator.add_leaf(
        id="Active_Exploitation_Status",
        desc="Answer explicitly confirms the vulnerability was actively exploited in the wild.",
        parent=zd_node,
        critical=True,
    )
    node_cause = evaluator.add_leaf(
        id="Technical_Cause",
        desc="Answer identifies the technical cause as a race condition in the Windows Kernel.",
        parent=zd_node,
        critical=True,
    )
    node_priv = evaluator.add_leaf(
        id="Privilege_Impact",
        desc="Answer states the vulnerability allows elevation of privileges to SYSTEM level.",
        parent=zd_node,
        critical=True,
    )

    # Prepare claims with additional instructions
    cve_str = extracted.cve_id or ""
    vuln_type_str = extracted.vulnerability_type or ""
    cvss_str = extracted.cvss_score or ""
    severity_str = extracted.ms_severity or ""
    exploitation_str = extracted.exploitation_status or ""
    cause_str = extracted.technical_cause or ""
    priv_str = extracted.privilege_impact or ""

    claims_and_sources: List[tuple[str, Optional[List[str] | str], Any, Optional[str]]] = [
        (
            f"The CVE identifier '{cve_str}' matches 'CVE-2025-62215' (ignore case and minor formatting; numeric part must be identical).",
            None,
            node_cve,
            "Focus on string identity for the CVE ID. Case-insensitive; whitespace and minor formatting may be ignored, "
            "but the numeric component must exactly match 2025-62215. If the answer lacks a CVE ID, mark incorrect."
        ),
        (
            f"The stated vulnerability type '{vuln_type_str}' corresponds to 'Windows Kernel Elevation of Privilege' (EoP).",
            None,
            node_type,
            "Accept reasonable variants such as 'Windows kernel privilege escalation', 'Windows kernel EoP'. "
            "It must clearly indicate Windows Kernel and Elevation of Privilege."
        ),
        (
            f"The CVSS score '{cvss_str}' equals 7.0 (allow '7' vs '7.0' and formatting like '/10').",
            None,
            node_cvss,
            "Judge numerical equality to 7.0. Minor formatting differences such as '7', '7.0/10', or 'CVSS: 7.0' are acceptable. "
            "If the answer does not state any CVSS value, mark incorrect."
        ),
        (
            f"The Microsoft's severity rating '{severity_str}' is 'Important' and not 'Critical'.",
            None,
            node_severity,
            "Verify that the stated Microsoft severity equals 'Important' explicitly, and is not 'Critical'. "
            "If 'Critical' appears as the rating, this must fail."
        ),
        (
            f"The exploitation status '{exploitation_str}' explicitly indicates the vulnerability was actively exploited in the wild.",
            None,
            node_active,
            "Look for explicit active exploitation language: 'actively exploited', 'under active attack', 'exploitation observed', "
            "'in the wild'. If the text does not clearly confirm active exploitation, mark incorrect."
        ),
        (
            f"The technical cause '{cause_str}' identifies a race condition in the Windows Kernel (TOCTOU/race condition).",
            None,
            node_cause,
            "Accept synonyms like 'race condition', 'TOCTOU', 'time-of-check-to-time-of-use' in the Windows Kernel context."
        ),
        (
            f"The described privilege impact '{priv_str}' indicates elevation of privileges to SYSTEM level.",
            None,
            node_priv,
            "Accept variants such as 'SYSTEM privileges', 'NT AUTHORITY\\SYSTEM'. Must clearly indicate SYSTEM-level privilege escalation."
        ),
    ]

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Entry point for evaluating the agent's answer for the November 2025 Patch Tuesday zero-day vulnerability.
    """
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

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_zero_day(),
        template_class=ZeroDayExtraction,
        extraction_name="zero_day_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "required_values": REQUIRED_VALUES,
        "context": "Microsoft November 2025 Patch Tuesday zero-day vulnerability metadata requirements"
    })

    # Build and run verification
    await build_zero_day_verification(evaluator, root, extracted_info)

    # Return summary
    return evaluator.get_summary()