import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ios_26_2_exploited_webkit_vulns"
TASK_DESCRIPTION = """
In December 2025, Apple released iOS 26.2, which addressed multiple security vulnerabilities. Apple publicly acknowledged that two specific WebKit vulnerabilities fixed in this update had been actively exploited in sophisticated attacks against targeted individuals using versions of iOS before iOS 26.

Please provide the following information about iOS 26.2 and these two exploited WebKit vulnerabilities:

1. The exact release date of iOS 26.2 (in the format: Month Day, Year)
2. The official Apple Support URL for the security content document of iOS 26.2
3. For each of the two WebKit vulnerabilities that were actively exploited:
   - The CVE identifier
   - The technical type of vulnerability (e.g., use-after-free, memory corruption, buffer overflow)
   - The organization(s) or entity that reported the vulnerability to Apple

Note: Both vulnerabilities should be WebKit-related and must have been confirmed by Apple as actively exploited in attacks. The information should be sourced from Apple's official security documentation.
"""

EXPECTED_RELEASE_DATE = "December 12, 2025"
EXPECTED_SECURITY_DOC_URL = "https://support.apple.com/en-us/125884"

EXPECTED_VULN1 = {
    "cve": "CVE-2025-43529",
    "type": "use-after-free",
    "reporters": ["Google Threat Analysis Group"],
}
EXPECTED_VULN2 = {
    "cve": "CVE-2025-14174",
    "type": "memory corruption",
    "reporters": ["Apple", "Google Threat Analysis Group"],
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WebKitVuln(BaseModel):
    cve_id: Optional[str] = None
    vuln_type: Optional[str] = None
    reporters: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class IOS262Extraction(BaseModel):
    release_date: Optional[str] = None
    security_content_url: Optional[str] = None
    vulnerabilities: List[WebKitVuln] = Field(default_factory=list)
    joint_exploited_together: Optional[bool] = None
    joint_statement: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the requested information about iOS 26.2 and the two actively exploited WebKit vulnerabilities exactly as presented in the answer.

    Return a JSON object with:
    - release_date: The exact release date string for iOS 26.2 as stated in the answer (e.g., "December 12, 2025"). If missing, null.
    - security_content_url: The official Apple Support URL for the iOS 26.2 security content document included in the answer. If multiple URLs are given, choose the one explicitly labeled as Apple's official security content for iOS 26.2. If missing, null.
    - vulnerabilities: An array (up to two items) for the WebKit vulnerabilities that the answer explicitly says were actively exploited. For each item:
        • cve_id: CVE identifier string (e.g., "CVE-2025-12345"), or null if not provided.
        • vuln_type: Technical type string exactly as in the answer (e.g., "use-after-free", "memory corruption"), or null.
        • reporters: Array of reporter names exactly as in the answer (e.g., ["Google Threat Analysis Group"]), empty array if none provided.
        • sources: Array of URLs cited in the answer for this CVE (prefer Apple's official documentation links if present), empty array if none provided.
    - joint_exploited_together: true/false if the answer explicitly states these two CVEs were exploited together, else null.
    - joint_statement: The exact sentence or short phrase from the answer supporting that joint exploitation claim (if present), else null.

    STRICT RULES:
    1) Only extract what is explicitly present in the answer; do not invent or infer.
    2) For URLs, only include actual URLs that appear in the answer.
    3) Keep strings exactly as written in the answer (do not normalize case or punctuation).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


def _equals_case_sensitive(a: Optional[str], b: str) -> bool:
    return _normalize(a) == b


def _month_day_year_format_ok(s: Optional[str]) -> bool:
    if not s:
        return False
    return re.fullmatch(r"[A-Z][a-z]+ [0-9]{1,2}, [0-9]{4}", s.strip()) is not None


def _find_vuln_by_cve(vulns: List[WebKitVuln], target_cve: str) -> Optional[WebKitVuln]:
    target = target_cve.strip().upper()
    for v in vulns:
        if _normalize(v.cve_id).upper() == target:
            return v
    return None


def _combine_sources(primary: Optional[str], extra: List[str]) -> List[str]:
    urls = []
    if primary:
        urls.append(primary)
    for u in extra:
        if u and isinstance(u, str):
            urls.append(u)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_release_info(evaluator: Evaluator, parent_node, extracted: IOS262Extraction):
    rel_node = evaluator.add_parallel(
        id="iOS_26.2_Release_Information",
        desc="Provide iOS 26.2 release information (must match constraints)",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Release_Date_Exact (critical) – exact match and correct format
    release_date_ok = _equals_case_sensitive(extracted.release_date, EXPECTED_RELEASE_DATE) and _month_day_year_format_ok(extracted.release_date)
    evaluator.add_custom_node(
        result=release_date_ok,
        id="Release_Date_Exact",
        desc='Release date is exactly "December 12, 2025" and presented in Month Day, Year format',
        parent=rel_node,
        critical=True,
    )

    # Leaf: Security_Content_URL_Exact (critical) – exact URL match
    url_ok = _equals_case_sensitive(extracted.security_content_url, EXPECTED_SECURITY_DOC_URL)
    evaluator.add_custom_node(
        result=url_ok,
        id="Security_Content_URL_Exact",
        desc=f"Security content document URL is exactly {EXPECTED_SECURITY_DOC_URL}",
        parent=rel_node,
        critical=True,
    )


async def verify_one_vuln(
    evaluator: Evaluator,
    parent_node,
    node_id_prefix: str,
    node_desc: str,
    extracted_vuln: Optional[WebKitVuln],
    expected: Dict[str, Any],
    apple_url: str,
):
    v_node = evaluator.add_parallel(
        id=node_id_prefix,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )

    # CVE_ID exact (critical) – checks the answer actually provided the correct CVE id
    cve_ok = _equals_case_sensitive(extracted_vuln.cve_id if extracted_vuln else None, expected["cve"])
    evaluator.add_custom_node(
        result=cve_ok,
        id=f"{node_id_prefix}_CVE_ID",
        desc=f"CVE identifier is exactly {expected['cve']}",
        parent=v_node,
        critical=True,
    )

    # Build sources for URL-based verifications
    vuln_sources = extracted_vuln.sources if extracted_vuln and extracted_vuln.sources else []
    sources = _combine_sources(apple_url, vuln_sources)

    # Prepare leaves
    webkit_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_WebKit_Related",
        desc="Vulnerability is identified as WebKit-related in Apple's security documentation",
        parent=v_node,
        critical=True,
    )
    exploited_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Actively_Exploited_Confirmed",
        desc="Apple's security documentation states the issue was actively exploited in attacks",
        parent=v_node,
        critical=True,
    )
    type_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Vulnerability_Type_Exact",
        desc=f"Technical type is {expected['type']}",
        parent=v_node,
        critical=True,
    )
    reporter_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Reporter_Exact",
        desc=f"Reported to Apple by {', '.join(expected['reporters'])}",
        parent=v_node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"On Apple's iOS 26.2 security content page, CVE {expected['cve']} is a WebKit vulnerability (i.e., explicitly listed under WebKit).",
            sources,
            webkit_leaf,
            "Allow reasonable phrasing variations (e.g., the entry is under a 'WebKit' section). Focus on whether the CVE is clearly associated with WebKit.",
        ),
        (
            f"Apple's iOS 26.2 security content page states that CVE {expected['cve']} was actively exploited (e.g., language like 'Apple is aware of a report that this issue may have been exploited').",
            sources,
            exploited_leaf,
            "Accept common Apple phrasing indicating exploitation in the wild.",
        ),
        (
            f"On Apple's iOS 26.2 security content page, CVE {expected['cve']} is described as a {expected['type']} vulnerability (allow hyphenation or spacing variants like 'use after free').",
            sources,
            type_leaf,
            "Treat minor wording variants as equivalent (e.g., 'use-after-free' vs 'use after free').",
        ),
        (
            f"On Apple's iOS 26.2 security content page, CVE {expected['cve']} is credited as reported by {', '.join(expected['reporters'])}.",
            sources,
            reporter_leaf,
            "Allow 'Google TAG' as an alias for 'Google Threat Analysis Group'. For multiple reporters, ensure all are listed.",
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_joint_constraint(evaluator: Evaluator, parent_node, apple_url: str):
    joint_leaf = evaluator.add_leaf(
        id="Joint_Constraint_Exploited_Together",
        desc="States that both CVE-2025-43529 and CVE-2025-14174 were exploited together (as per constraints)",
        parent=parent_node,
        critical=True,
    )
    claim = (
        "Apple's iOS 26.2 security content page indicates that CVE-2025-43529 and CVE-2025-14174 were exploited together "
        "(e.g., mentioned as being used in the same attacks, in combination, or as part of the same exploit chain)."
    )
    await evaluator.verify(
        claim=claim,
        node=joint_leaf,
        sources=apple_url,
        additional_instruction="Accept phrasing such as 'in combination', 'in the same attack chain', or similar language that clearly implies the two were exploited together.",
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
    model: str = "o4-mini",
) -> Dict:
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

    # Extract structured info from the answer
    extracted: IOS262Extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=IOS262Extraction,
        extraction_name="ios_26_2_extraction",
    )

    # Add ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_release_date": EXPECTED_RELEASE_DATE,
            "expected_security_doc_url": EXPECTED_SECURITY_DOC_URL,
            "expected_vulnerabilities": [
                {
                    "cve": EXPECTED_VULN1["cve"],
                    "type": EXPECTED_VULN1["type"],
                    "reporters": EXPECTED_VULN1["reporters"],
                },
                {
                    "cve": EXPECTED_VULN2["cve"],
                    "type": EXPECTED_VULN2["type"],
                    "reporters": EXPECTED_VULN2["reporters"],
                },
            ],
        },
        gt_type="expected_values",
    )

    # Build main investigation node (critical)
    main_node = evaluator.add_parallel(
        id="iOS_26.2_Exploited_Vulnerabilities_Investigation",
        desc="Provide iOS 26.2 release info and details for the two Apple-confirmed, actively exploited WebKit vulnerabilities from Apple's official security documentation, matching the stated constraints",
        parent=root,
        critical=True,
    )

    # 1) Release information checks
    await verify_release_info(evaluator, main_node, extracted)

    # 2) Two exploited WebKit vulnerabilities
    two_vulns_node = evaluator.add_parallel(
        id="Two_Exploited_WebKit_Vulnerabilities",
        desc="Provide required details for the two WebKit vulnerabilities Apple confirms were actively exploited (must match constraints)",
        parent=main_node,
        critical=True,
    )

    # Resolve Apple's URL to use for verification even if answer omitted it
    apple_url = extracted.security_content_url if _normalize(extracted.security_content_url) else EXPECTED_SECURITY_DOC_URL

    # Map extracted vulnerabilities by CVE
    v1_extracted = _find_vuln_by_cve(extracted.vulnerabilities, EXPECTED_VULN1["cve"]) if extracted and extracted.vulnerabilities else None
    v2_extracted = _find_vuln_by_cve(extracted.vulnerabilities, EXPECTED_VULN2["cve"]) if extracted and extracted.vulnerabilities else None

    # Vulnerability 1 subtree
    await verify_one_vuln(
        evaluator=evaluator,
        parent_node=two_vulns_node,
        node_id_prefix="Exploited_WebKit_Vulnerability_1",
        node_desc="Details for the first exploited WebKit vulnerability (must match constraints)",
        extracted_vuln=v1_extracted,
        expected=EXPECTED_VULN1,
        apple_url=apple_url,
    )

    # Vulnerability 2 subtree
    await verify_one_vuln(
        evaluator=evaluator,
        parent_node=two_vulns_node,
        node_id_prefix="Exploited_WebKit_Vulnerability_2",
        node_desc="Details for the second exploited WebKit vulnerability (must match constraints)",
        extracted_vuln=v2_extracted,
        expected=EXPECTED_VULN2,
        apple_url=apple_url,
    )

    # Joint constraint (critical)
    await verify_joint_constraint(evaluator, two_vulns_node, apple_url)

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "apple_url_used_for_verification": apple_url,
            "extracted_release_date": extracted.release_date,
            "extracted_security_content_url": extracted.security_content_url,
            "extracted_vuln_cves": [v.cve_id for v in extracted.vulnerabilities] if extracted and extracted.vulnerabilities else [],
        },
        info_type="debug",
        info_name="extraction_debug_info",
    )

    return evaluator.get_summary()