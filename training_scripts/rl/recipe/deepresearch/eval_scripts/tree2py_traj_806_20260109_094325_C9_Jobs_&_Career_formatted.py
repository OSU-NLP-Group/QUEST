import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_state_licensure_guide"
TASK_DESCRIPTION = """
A career development organization is creating a comprehensive licensure maintenance guide for professionals considering practice in multiple states. They need detailed, verified information about continuing education requirements, professional development benefits, and licensure standards for four specific profession-state combinations.

Research and provide complete information for each of the following:

1. Licensed Professional Clinical Counselors (LPCC) in California
2. Licensed Professional Counselors (LPC) in Texas
3. Licensed Clinical Social Workers (LCSW) in Louisiana
4. Certified Public Accountants (CPA) in Ohio

For each profession-state combination, identify and document:

- The specific license type and full name of the state licensing board
- The method or official website URL for verifying active licenses
- Total continuing education (CE) hours required per renewal cycle
- The duration of the renewal cycle (e.g., annual, biennial, triennial)
- All mandatory CE topics with their specific hour requirements (such as ethics, cultural competency, law, clinical practice, etc.)
- Any special requirements for first-time license renewal if they differ from standard renewal
- The maximum amount of tax-free educational assistance employers can provide per employee per year under federal law (IRS Section 127)
- Professional association recommendations for CE hours (such as NASW standards for social workers)
- Whether the state participates in an interstate licensure compact for that profession
- Any minimum annual CE hour requirements if the renewal cycle exceeds one year
- Requirements for CE provider approval or accreditation

Provide reference URLs from official state licensing board websites or federal government sources to support your findings.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MandatoryCEItem(BaseModel):
    topic: Optional[str] = None
    hours: Optional[str] = None
    notes: Optional[str] = None


class GlobalFederalInfo(BaseModel):
    irs_127_amount: Optional[str] = None
    irs_127_source_urls: List[str] = Field(default_factory=list)
    employer_assistance_examples: Optional[str] = None
    other_federal_urls: List[str] = Field(default_factory=list)


class ProfessionStateInfo(BaseModel):
    license_type: Optional[str] = None
    licensing_board_full_name: Optional[str] = None
    license_verification_url: Optional[str] = None

    total_ce_hours: Optional[str] = None
    renewal_cycle: Optional[str] = None

    mandatory_ce_items: List[MandatoryCEItem] = Field(default_factory=list)

    first_time_renewal_special: Optional[str] = None
    annual_minimum_ce: Optional[str] = None  # e.g., "20 hours/year" or "No annual minimum"

    ce_provider_approval: Optional[str] = None

    association_recommendation: Optional[str] = None
    association_source_urls: List[str] = Field(default_factory=list)

    interstate_compact_status: Optional[str] = None
    compact_source_urls: List[str] = Field(default_factory=list)

    official_reference_urls: List[str] = Field(default_factory=list)
    verification_support_urls: List[str] = Field(default_factory=list)


class LicensureGuideExtraction(BaseModel):
    global_federal: Optional[GlobalFederalInfo] = None
    ca_lpcc: Optional[ProfessionStateInfo] = None
    tx_lpc: Optional[ProfessionStateInfo] = None
    la_lcsw: Optional[ProfessionStateInfo] = None
    oh_cpa: Optional[ProfessionStateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract structured information from the answer for the specified profession–state combinations and the federal educational assistance context. Follow these rules:
- Extract only what is explicitly stated in the answer.
- Use null for any missing fields.
- For URL fields, extract only actual URLs (full URLs preferred).
- Keep hours and cycles as strings when possible (e.g., "36", "24 hours", "every 2 years", "annual", "triennial").

Return a JSON object matching this schema:

{
  "global_federal": {
    "irs_127_amount": string|null,                           // e.g., "$5,250" or "5250"
    "irs_127_source_urls": string[],                         // Federal .gov sources if present (e.g., irs.gov)
    "employer_assistance_examples": string|null,             // A brief sentence from the answer about employer tuition/educational assistance scope
    "other_federal_urls": string[]                           // Any other federal URLs referenced
  },
  "ca_lpcc": {
    "license_type": string|null,                             // e.g., "LPCC"
    "licensing_board_full_name": string|null,                // e.g., "California Board of Behavioral Sciences"
    "license_verification_url": string|null,                 // Official search/verification page URL
    "total_ce_hours": string|null,                           // e.g., "36"
    "renewal_cycle": string|null,                            // e.g., "every 2 years", "biennial"
    "mandatory_ce_items": [                                  // If the answer lists mandatory topics
      {"topic": string|null, "hours": string|null, "notes": string|null}
    ],
    "first_time_renewal_special": string|null,               // Specific differences or "no difference"
    "annual_minimum_ce": string|null,                        // e.g., "No annual minimum", "18 hours/year"
    "ce_provider_approval": string|null,                     // e.g., "Board-approved, NBCC, or APA providers"
    "association_recommendation": string|null,               // e.g., ACA/NBCC guidance if provided
    "association_source_urls": string[],                     // URLs for association recommendations
    "interstate_compact_status": string|null,                // e.g., "Not in Counseling Compact", or "In Counseling Compact"
    "compact_source_urls": string[],                         // Compact or official references
    "official_reference_urls": string[],                     // State board/official references for CE and renewal
    "verification_support_urls": string[]                    // Any additional official support URLs
  },
  "tx_lpc": { /* same fields as ca_lpcc */ },
  "la_lcsw": { /* same fields, adapted to LCSW */ },
  "oh_cpa": { /* same fields, adapted to CPA/CPE */ }
}

Special guidance:
- For California LPCC, if present, extract:
  • total_ce_hours as "36"
  • renewal_cycle as "every 2 years" or similar
  • include law and ethics requirement if listed (e.g., "6 hours")
- For Texas LPC, if present, extract:
  • total_ce_hours as "24"
  • renewal_cycle as "every 2 years"
  • include ethics "6 hours" and cultural diversity/competency "3 hours" if listed
- For Louisiana LCSW, if present, extract:
  • total_ce_hours as "20"
  • renewal_cycle as "annual"
  • include "10 hours" clinical social work if listed
- For Ohio CPA, if present, extract:
  • total_ce_hours as "120"
  • renewal_cycle as "every 3 years" or "triennial"
  • annual minimum such as "20 hours/year" if stated

For the federal section:
- Extract the IRS Section 127 maximum tax-free educational assistance amount (e.g., "$5,250") exactly as stated.
- Extract federal source URLs (prefer .gov domains such as irs.gov or congress.gov).
- Extract any mention that employers may offer tuition reimbursement/educational assistance for professional development, certification, and continuing education.

If multiple values are mentioned, extract the first, most prominent value as presented by the answer.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip() != ""]


def _combine_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for urls in url_lists:
        for u in urls:
            if u and u not in seen:
                ordered.append(u)
                seen.add(u)
    return ordered


def _has_gov_url(urls: List[str]) -> bool:
    urls = _safe_list(urls)
    for u in urls:
        if ".gov" in u:
            return True
    return False


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------------------------------------------------------- #
# Verification building blocks                                                #
# --------------------------------------------------------------------------- #
async def verify_global_federal(evaluator: Evaluator, parent_node, global_info: Optional[GlobalFederalInfo]) -> None:
    # Group node for the federal information (make it non-critical parent to allow child critical leaves)
    node = evaluator.add_parallel(
        id="Global_Federal_Ed_Assistance",
        desc="Federal educational assistance information applicable across the guide",
        parent=parent_node,
        critical=False
    )

    # Existence helpers
    irs_amount_str = (global_info.irs_127_amount if global_info else None) or ""
    irs_urls = _safe_list(global_info.irs_127_source_urls if global_info else [])

    # 1) IRS Section 127 maximum amount should be $5,250 — two checks: (a) answer states it, (b) supported by a federal URL
    amount_answer_leaf = evaluator.add_leaf(
        id="IRS_127_Max_Amount_Answer",
        desc="Answer states the IRS Section 127 maximum tax-free educational assistance amount as $5,250 per employee per year",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the answer, the IRS Section 127 maximum tax-free educational assistance is $5,250 per employee per year.",
        node=amount_answer_leaf,
        additional_instruction="Look for an explicit number 5,250 or $5,250 regarding IRS Section 127."
    )

    # Existence of a federal source URL (critical)
    fed_url_exists = evaluator.add_custom_node(
        result=_has_gov_url(irs_urls),
        id="IRS_127_Federal_Source_URL_Exists",
        desc="At least one federal (.gov) source URL is provided for IRS Section 127 amount",
        parent=node,
        critical=True
    )

    amount_supported_leaf = evaluator.add_leaf(
        id="IRS_127_Max_Amount_Supported",
        desc="A federal (.gov) source supports the IRS Section 127 maximum as $5,250 per employee per year",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The maximum tax-free educational assistance under Internal Revenue Code Section 127 is $5,250 per employee per year.",
        node=amount_supported_leaf,
        sources=irs_urls,
        additional_instruction="Confirm the $5,250 limit from a federal (.gov) source, preferably irs.gov.",
        extra_prerequisites=[fed_url_exists]
    )

    # 2) Employer assistance examples — non-critical
    employer_leaf = evaluator.add_leaf(
        id="Employer_Assistance_Examples",
        desc="Answer notes employers may offer tuition reimbursement/educational assistance covering PD, certification, and CE",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer states that employers may offer tuition reimbursement or educational assistance covering professional development, certification costs, and continuing education.",
        node=employer_leaf,
        additional_instruction="Allow reasonable synonyms for 'tuition reimbursement' and 'educational assistance'."
    )


# Expected constants derived from rubric constraints
EXPECTED = {
    "CA_LPCC": {
        "license_type": "LPCC",
        "board_name": "California Board of Behavioral Sciences",
        "total_ce": "36",
        "cycle": "every 2 years",
        "mandatory_requirements": [
            {"desc": "at least 6 hours in law and ethics", "keywords": ["law", "ethics"], "hours": "6"}
        ],
        "annual_minimum_expected": None,  # May be none or not required by rule; answer must address whether it exists
    },
    "TX_LPC": {
        "license_type": "LPC",
        "board_name": "Texas Behavioral Health Executive Council",
        "total_ce": "24",
        "cycle": "every 2 years",
        "mandatory_requirements": [
            {"desc": "at least 6 hours in ethics", "keywords": ["ethics"], "hours": "6"},
            {"desc": "at least 3 hours in cultural diversity or cultural competency", "keywords": ["cultural"], "hours": "3"}
        ],
        "annual_minimum_expected": None,
    },
    "LA_LCSW": {
        "license_type": "LCSW",
        "board_name": "Louisiana State Board of Social Work Examiners",
        "total_ce": "20",
        "cycle": "annual",
        "mandatory_requirements": [
            {"desc": "at least 10 hours in clinical social work", "keywords": ["clinical social work"], "hours": "10"}
        ],
        # No annual minimum node for LA (annual cycle)
    },
    "OH_CPA": {
        "license_type": "CPA",
        "board_name": "Accountancy Board of Ohio",
        "total_ce": "120",
        "cycle": "every 3 years",
        "annual_minimum": "20",  # hours per year
        # For mandatory topics: if any; we just verify that the answer lists any mandatory topic with hours and support it
    }
}


async def _verify_license_type_and_board(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    expected_license_type: str,
    expected_board_name: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_License_Type_And_Board",
        desc=f"Identifies the license type and provides the full name of the state licensing board for {profession_label}",
        parent=parent,
        critical=True
    )

    # Existence (critical)
    exists = evaluator.add_custom_node(
        result=_nonempty(info.license_type) and _nonempty(info.licensing_board_full_name),
        id=f"{node_id_prefix}_License_Type_And_Board_Exists",
        desc="License type and state licensing board name are present in the answer",
        parent=node,
        critical=True
    )

    # Answer states expected license type (critical)
    lt_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_License_Type_Answer",
        desc=f"Answer identifies the license type as {expected_license_type}",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the answer, the license type is {expected_license_type}.",
        node=lt_leaf,
        additional_instruction="Allow minor variations like pluralization or inclusion of full title."
    )

    # Answer names the board (critical)
    board_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Board_Name_Answer",
        desc=f"Answer provides the full name of the licensing board (e.g., {expected_board_name} or commonly used official variant)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies the licensing board as {expected_board_name} or an official variant (abbreviation allowed).",
        node=board_leaf,
        additional_instruction="Accept common official variants or abbreviations, e.g., 'BBS' for California."
    )

    # Supported by official references (critical)
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    ref_exists = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Board_Refs_Exist",
        desc="Official/board reference URL(s) are provided",
        parent=node,
        critical=True
    )
    board_supported = evaluator.add_leaf(
        id=f"{node_id_prefix}_Board_Supported",
        desc="Official references support the board being responsible for this license",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The {expected_board_name} (or relevant state authority) is the official regulator for the {expected_license_type} license.",
        node=board_supported,
        sources=refs,
        additional_instruction="Check the regulator identity from the official board/authority site.",
        extra_prerequisites=[ref_exists, exists]
    )


async def _verify_license_verification_url(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_License_Verification_URL",
        desc=f"Provides the official method/URL to verify an active license for {profession_label}",
        parent=parent,
        critical=True
    )
    url_present = evaluator.add_custom_node(
        result=_nonempty(info.license_verification_url),
        id=f"{node_id_prefix}_License_Verification_URL_Exists",
        desc="License verification URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_License_Verification_URL_Supported",
        desc="Provided URL is an official license verification/search portal",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official license verification or search portal for the relevant state licensing authority.",
        node=leaf,
        sources=(info.license_verification_url or None),
        additional_instruction="Look for terms like 'verify', 'license lookup', or 'search licensee'; confirm it's official.",
        extra_prerequisites=[url_present]
    )


async def _verify_total_ce(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    expected_hours: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Total_CE_Hours",
        desc=f"States total CE/CPE hours required per renewal cycle for {profession_label}",
        parent=parent,
        critical=True
    )

    # Answer declares expected hours
    answer_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Total_CE_Hours_Answer",
        desc=f"Answer states total CE/CPE hours as {expected_hours}",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the answer, the total continuing education hours required per renewal cycle are {expected_hours}.",
        node=answer_leaf,
        additional_instruction="Match the numeric total; allow small phrasing variations."
    )

    # Supported by official references
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Total_CE_Hours_Refs_Exist",
        desc="Official reference URL(s) for CE hours exist",
        parent=node,
        critical=True
    )
    supported_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Total_CE_Hours_Supported",
        desc="Official references support the stated total CE/CPE hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total continuing education hours required per renewal cycle are {expected_hours}.",
        node=supported_leaf,
        sources=refs,
        additional_instruction="Confirm the numeric total on official board or rules page.",
        extra_prerequisites=[refs_exist]
    )


async def _verify_renewal_cycle(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    expected_cycle_phrase: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Renewal_Cycle_Duration",
        desc=f"States renewal cycle duration for {profession_label}",
        parent=parent,
        critical=True
    )

    answer_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Renewal_Cycle_Answer",
        desc=f"Answer states renewal cycle as {expected_cycle_phrase}",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the answer, the renewal cycle is {expected_cycle_phrase}.",
        node=answer_leaf,
        additional_instruction="Accept synonymous phrasing, e.g., biennial for every 2 years, triennial for every 3 years."
    )

    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Renewal_Cycle_Refs_Exist",
        desc="Official reference URL(s) for renewal cycle exist",
        parent=node,
        critical=True
    )
    supported_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Renewal_Cycle_Supported",
        desc="Official references support the renewal cycle duration",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The renewal cycle is {expected_cycle_phrase}.",
        node=supported_leaf,
        sources=refs,
        additional_instruction="Confirm cycle duration terminology from official rules/board site.",
        extra_prerequisites=[refs_exist]
    )


async def _verify_mandatory_topics_minima(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    minima: List[Dict[str, Any]],
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Mandatory_CE_Topics_And_Hours",
        desc=f"Lists mandatory CE topics with specific hour minima for {profession_label}",
        parent=parent,
        critical=True
    )

    # One pair of leaves for each minimum requirement stated in rubric
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Mandatory_Topics_Refs_Exist",
        desc="Official reference URL(s) for mandatory topic requirements exist",
        parent=node,
        critical=True
    )

    for idx, req in enumerate(minima):
        desc = req["desc"]
        hours = req.get("hours")
        # Answer mentions this minimum requirement
        leaf_ans = evaluator.add_leaf(
            id=f"{node_id_prefix}_Mandatory_Min_{idx}_Answer",
            desc=f"Answer includes mandatory CE requirement: {desc}",
            parent=node,
            critical=True
        )
        claim_ans = f"According to the answer, {desc} is required."
        if hours:
            claim_ans = f"According to the answer, {desc} is required (i.e., {hours} hours)."
        await evaluator.verify(
            claim=claim_ans,
            node=leaf_ans,
            additional_instruction="Allow reasonable synonymous topic labels; focus on hours and intent."
        )

        # Supported by official sources
        leaf_sup = evaluator.add_leaf(
            id=f"{node_id_prefix}_Mandatory_Min_{idx}_Supported",
            desc=f"Official references support mandatory CE requirement: {desc}",
            parent=node,
            critical=True
        )
        claim_sup = f"The official rules require {desc}."
        await evaluator.verify(
            claim=claim_sup,
            node=leaf_sup,
            sources=refs,
            additional_instruction="Confirm the stated minimum topic-hours are required by official rules.",
            extra_prerequisites=[refs_exist]
        )


async def _verify_first_time_renewal(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_First_Time_Renewal_Special_Requirements",
        desc=f"States whether first-time renewal requirements differ from standard renewal for {profession_label}, with citation",
        parent=parent,
        critical=True
    )

    # Answer addresses this
    leaf_ans = evaluator.add_leaf(
        id=f"{node_id_prefix}_First_Time_Renewal_Answer",
        desc="Answer explicitly addresses whether first-time renewal differs from standard renewal",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states whether first-time renewal requirements differ from standard renewal (or that there is no difference).",
        node=leaf_ans,
        additional_instruction="Look for a clear statement like 'no difference' or describe the distinct rules."
    )

    # Supported by official references
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_First_Time_Renewal_Refs_Exist",
        desc="Official reference URL(s) for first-time renewal exist",
        parent=node,
        critical=True
    )

    detail = info.first_time_renewal_special or ""
    if _lower(detail) in ("none", "no difference", "no special requirements"):
        claim_sup = "Official sources indicate first-time renewal requirements do not differ from standard renewal."
    else:
        # If provided, use the description verbatim; otherwise state generically
        claim_sup = f"Official sources indicate special first-time renewal requirements as described: {detail}" if _nonempty(detail) else "Official sources indicate whether special first-time renewal requirements apply."

    leaf_sup = evaluator.add_leaf(
        id=f"{node_id_prefix}_First_Time_Renewal_Supported",
        desc="Official references support the stated first-time renewal requirement position",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_sup,
        node=leaf_sup,
        sources=refs,
        additional_instruction="Confirm the presence/absence of distinct first-time renewal requirements.",
        extra_prerequisites=[refs_exist]
    )


async def _verify_annual_minimum_if_multiyear(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Annual_Minimum_If_MultiYear_Cycle",
        desc=f"If renewal exceeds one year, states whether a minimum annual CE applies for {profession_label}",
        parent=parent,
        critical=True
    )

    # Answer addresses this requirement
    leaf_ans = evaluator.add_leaf(
        id=f"{node_id_prefix}_Annual_Minimum_Answer",
        desc="Answer states whether a minimum annual CE requirement applies (and amount) or explicitly states no annual minimum",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states whether a minimum annual CE requirement applies and includes the amount if applicable, or explicitly states there is no annual minimum.",
        node=leaf_ans,
        additional_instruction="Look for phrasing like 'no annual minimum' or 'X hours per year'."
    )

    # Supported by official references
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Annual_Minimum_Refs_Exist",
        desc="Official reference URL(s) for annual minimum CE exist",
        parent=node,
        critical=True
    )
    detail = info.annual_minimum_ce or ""
    if _nonempty(detail):
        claim_sup = f"Official sources indicate the annual minimum CE requirement as: {detail}"
    else:
        claim_sup = "Official sources indicate whether there is a minimum annual CE requirement."

    leaf_sup = evaluator.add_leaf(
        id=f"{node_id_prefix}_Annual_Minimum_Supported",
        desc="Official references support the stated annual minimum position",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_sup,
        node=leaf_sup,
        sources=refs,
        additional_instruction="Confirm whether a specific per-year minimum exists.",
        extra_prerequisites=[refs_exist]
    )


async def _verify_annual_minimum_ohio(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    expected_min_per_year: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Annual_Minimum_CE",
        desc=f"States the minimum annual CPE requirement for Ohio CPA",
        parent=parent,
        critical=True
    )

    leaf_ans = evaluator.add_leaf(
        id=f"{node_id_prefix}_Annual_Minimum_CE_Answer",
        desc=f"Answer states at least {expected_min_per_year} hours per year minimum",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the answer, the minimum annual CPE requirement is {expected_min_per_year} hours per year.",
        node=leaf_ans,
        additional_instruction="Accept equivalent phrasing like '20 per year' or 'at least 20 each year'."
    )

    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Annual_Minimum_CE_Refs_Exist",
        desc="Official reference URL(s) for annual minimum CPE exist",
        parent=node,
        critical=True
    )
    leaf_sup = evaluator.add_leaf(
        id=f"{node_id_prefix}_Annual_Minimum_CE_Supported",
        desc="Official references support the minimum annual CPE requirement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum annual CPE requirement is {expected_min_per_year} hours per year.",
        node=leaf_sup,
        sources=refs,
        additional_instruction="Confirm the per-year minimum on Accountancy Board official pages or rules.",
        extra_prerequisites=[refs_exist]
    )


async def _verify_provider_approval(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_CE_Provider_Approval",
        desc=f"Describes CE/CPE provider approval or accreditation requirements for {profession_label}",
        parent=parent,
        critical=True
    )

    # Answer addresses provider approval
    leaf_ans = evaluator.add_leaf(
        id=f"{node_id_prefix}_CE_Provider_Approval_Answer",
        desc="Answer describes provider approval/accreditation requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer describes CE/CPE provider approval or accreditation requirements (e.g., board-approved or accredited providers).",
        node=leaf_ans,
        additional_instruction="Look for mention of board approval, accredited organizations (NBCC, APA, AICPA/NASBA), or state-recognized providers."
    )

    # Supported by official references
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_CE_Provider_Approval_Refs_Exist",
        desc="Official reference URL(s) for provider approval exist",
        parent=node,
        critical=True
    )
    leaf_sup = evaluator.add_leaf(
        id=f"{node_id_prefix}_CE_Provider_Approval_Supported",
        desc="Official references support provider approval/accreditation requirements",
        parent=node,
        critical=True
    )
    detail = info.ce_provider_approval or "CE/CPE must be from approved/accredited providers."
    await evaluator.verify(
        claim=f"Official sources indicate: {detail}",
        node=leaf_sup,
        sources=refs,
        additional_instruction="Verify provider approval or accreditation details on official pages.",
        extra_prerequisites=[refs_exist]
    )


async def _verify_association_recommendation(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo,
    expected_statement: Optional[str] = None
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Association_CE_Recommendation",
        desc=f"Provides professional association recommendation/standard for CE/CPE hours for {profession_label}",
        parent=parent,
        critical=True
    )

    # Answer addresses recommendation
    leaf_ans = evaluator.add_leaf(
        id=f"{node_id_prefix}_Association_CE_Recommendation_Answer",
        desc="Answer provides a professional association recommendation with hours (or clear statement) and cites",
        parent=node,
        critical=True
    )
    if expected_statement:
        claim_ans = f"According to the answer, {expected_statement}."
    else:
        claim_ans = "According to the answer, a professional association recommendation or standard for CE/CPE hours is provided for this profession."
    await evaluator.verify(
        claim=claim_ans,
        node=leaf_ans,
        additional_instruction="Look for association names such as NASW (social work), ACA/NBCC (counselors), AICPA or state CPA societies."
    )

    urls = _safe_list(info.association_source_urls)
    urls_exist = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{node_id_prefix}_Association_CE_Recommendation_URLs_Exist",
        desc="Association source URL(s) are provided",
        parent=node,
        critical=True
    )

    leaf_sup = evaluator.add_leaf(
        id=f"{node_id_prefix}_Association_CE_Recommendation_Supported",
        desc="Association source supports the stated recommendation/standard",
        parent=node,
        critical=True
    )
    claim_sup = expected_statement if expected_statement else (info.association_recommendation or "The stated association recommendation applies.")
    await evaluator.verify(
        claim=claim_sup,
        node=leaf_sup,
        sources=urls,
        additional_instruction="Verify the recommendation from the association site.",
        extra_prerequisites=[urls_exist]
    )


async def _verify_compact_status(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Interstate_Compact_Status",
        desc=f"States whether the state participates in an interstate licensure compact for {profession_label}",
        parent=parent,
        critical=True
    )

    # Answer states compact participation status (yes/no)
    leaf_ans = evaluator.add_leaf(
        id=f"{node_id_prefix}_Compact_Status_Answer",
        desc="Answer states whether the state participates in a relevant interstate licensure compact",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states whether the state participates in a relevant interstate licensure compact for this profession.",
        node=leaf_ans,
        additional_instruction="Look for compact names such as Counseling Compact, Social Work Licensure Compact, etc."
    )

    urls = _safe_list(info.compact_source_urls)
    urls_exist = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{node_id_prefix}_Compact_Status_URLs_Exist",
        desc="Compact or official reference URL(s) are provided",
        parent=node,
        critical=True
    )

    detail = info.interstate_compact_status or "The state's participation status in a relevant compact."
    leaf_sup = evaluator.add_leaf(
        id=f"{node_id_prefix}_Compact_Status_Supported",
        desc="Compact or official source supports the stated participation status",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Official/compact sources support: {detail}",
        node=leaf_sup,
        sources=urls,
        additional_instruction="Confirm participation status on the compact's official site or a state official page.",
        extra_prerequisites=[urls_exist]
    )


async def _verify_official_references_present(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    profession_label: str,
    info: ProfessionStateInfo
) -> None:
    node = evaluator.add_parallel(
        id=f"{node_id_prefix}_Official_Reference_URLs",
        desc=f"Provides official reference URL(s) supporting board identity, verification method, and CE/renewal requirements for {profession_label}",
        parent=parent,
        critical=True
    )
    refs = _safe_list(info.official_reference_urls)
    exist_leaf = evaluator.add_custom_node(
        result=len(refs) > 0,
        id=f"{node_id_prefix}_Official_Refs_Exist",
        desc="At least one official reference URL is provided",
        parent=node,
        critical=True
    )

    # Also verify the answer claims references are official sources
    verify_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_Official_Refs_Are_Official",
        desc="Provided reference URLs are official licensing board or state sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided reference URLs are official licensing board or state authority sources for CE/renewal/verification information.",
        node=verify_leaf,
        sources=refs,
        additional_instruction="Look for clear official indicators (e.g., .gov domain, board name). For Louisiana, a .org domain for the official board is acceptable.",
        extra_prerequisites=[exist_leaf]
    )


async def verify_ca_lpcc(evaluator: Evaluator, parent, info: ProfessionStateInfo) -> None:
    node = evaluator.add_parallel(
        id="California_LPCC",
        desc="California — Licensed Professional Clinical Counselor (LPCC)",
        parent=parent,
        critical=False
    )
    expected = EXPECTED["CA_LPCC"]
    await _verify_license_type_and_board(evaluator, node, "CA", "California LPCC", expected["license_type"], expected["board_name"], info)
    await _verify_license_verification_url(evaluator, node, "CA", "California LPCC", info)
    await _verify_total_ce(evaluator, node, "CA", "California LPCC", expected["total_ce"], info)
    await _verify_renewal_cycle(evaluator, node, "CA", "California LPCC", expected["cycle"], info)
    await _verify_mandatory_topics_minima(evaluator, node, "CA", "California LPCC", expected["mandatory_requirements"], info)
    await _verify_first_time_renewal(evaluator, node, "CA", "California LPCC", info)
    await _verify_annual_minimum_if_multiyear(evaluator, node, "CA", "California LPCC", info)
    await _verify_provider_approval(evaluator, node, "CA", "California LPCC", info)
    # Association recommendation (no fixed expected statement for CA)
    await _verify_association_recommendation(evaluator, node, "CA", "California LPCC", info, expected_statement=None)
    await _verify_compact_status(evaluator, node, "CA", "California LPCC", info)
    await _verify_official_references_present(evaluator, node, "CA", "California LPCC", info)


async def verify_tx_lpc(evaluator: Evaluator, parent, info: ProfessionStateInfo) -> None:
    node = evaluator.add_parallel(
        id="Texas_LPC",
        desc="Texas — Licensed Professional Counselor (LPC)",
        parent=parent,
        critical=False
    )
    expected = EXPECTED["TX_LPC"]
    await _verify_license_type_and_board(evaluator, node, "TX", "Texas LPC", expected["license_type"], expected["board_name"], info)
    await _verify_license_verification_url(evaluator, node, "TX", "Texas LPC", info)
    await _verify_total_ce(evaluator, node, "TX", "Texas LPC", expected["total_ce"], info)
    await _verify_renewal_cycle(evaluator, node, "TX", "Texas LPC", expected["cycle"], info)
    await _verify_mandatory_topics_minima(evaluator, node, "TX", "Texas LPC", expected["mandatory_requirements"], info)
    await _verify_first_time_renewal(evaluator, node, "TX", "Texas LPC", info)
    await _verify_annual_minimum_if_multiyear(evaluator, node, "TX", "Texas LPC", info)
    await _verify_provider_approval(evaluator, node, "TX", "Texas LPC", info)
    await _verify_association_recommendation(evaluator, node, "TX", "Texas LPC", info, expected_statement=None)
    await _verify_compact_status(evaluator, node, "TX", "Texas LPC", info)
    await _verify_official_references_present(evaluator, node, "TX", "Texas LPC", info)


async def verify_la_lcsw(evaluator: Evaluator, parent, info: ProfessionStateInfo) -> None:
    node = evaluator.add_parallel(
        id="Louisiana_LCSW",
        desc="Louisiana — Licensed Clinical Social Worker (LCSW)",
        parent=parent,
        critical=False
    )
    expected = EXPECTED["LA_LCSW"]
    await _verify_license_type_and_board(evaluator, node, "LA", "Louisiana LCSW", expected["license_type"], expected["board_name"], info)
    await _verify_license_verification_url(evaluator, node, "LA", "Louisiana LCSW", info)
    await _verify_total_ce(evaluator, node, "LA", "Louisiana LCSW", expected["total_ce"], info)
    await _verify_renewal_cycle(evaluator, node, "LA", "Louisiana LCSW", expected["cycle"], info)
    await _verify_mandatory_topics_minima(evaluator, node, "LA", "Louisiana LCSW", expected["mandatory_requirements"], info)
    await _verify_first_time_renewal(evaluator, node, "LA", "Louisiana LCSW", info)
    await _verify_provider_approval(evaluator, node, "LA", "Louisiana LCSW", info)
    # Association recommendation: rubric notes NASW recommendation of 48 hours every 2 years
    await _verify_association_recommendation(
        evaluator, node, "LA", "Louisiana LCSW", info,
        expected_statement="the NASW recommendation for social workers is 48 hours of continuing education every 2 years"
    )
    await _verify_compact_status(evaluator, node, "LA", "Louisiana LCSW", info)
    await _verify_official_references_present(evaluator, node, "LA", "Louisiana LCSW", info)


async def verify_oh_cpa(evaluator: Evaluator, parent, info: ProfessionStateInfo) -> None:
    node = evaluator.add_parallel(
        id="Ohio_CPA",
        desc="Ohio — Certified Public Accountant (CPA)",
        parent=parent,
        critical=False
    )
    expected = EXPECTED["OH_CPA"]
    await _verify_license_type_and_board(evaluator, node, "OH", "Ohio CPA", expected["license_type"], expected["board_name"], info)
    await _verify_license_verification_url(evaluator, node, "OH", "Ohio CPA", info)
    await _verify_total_ce(evaluator, node, "OH", "Ohio CPA", expected["total_ce"], info)
    await _verify_renewal_cycle(evaluator, node, "OH", "Ohio CPA", expected["cycle"], info)
    await _verify_annual_minimum_ohio(evaluator, node, "OH", expected["annual_minimum"], info)

    # Mandatory topics (if any) — generic verification pair
    mand_node = evaluator.add_parallel(
        id="OH_Mandatory_CE_Topics_And_Hours",
        desc="Lists mandatory CE/CPE topics with specific hour requirements per official rules (if any)",
        parent=node,
        critical=True
    )
    # Answer claims mandatory topics with hours (or explicitly none)
    leaf_ans = evaluator.add_leaf(
        id="OH_Mandatory_CE_Topics_Answer",
        desc="Answer specifies mandatory CE/CPE topics with hours (if any exist) or explicitly states none",
        parent=mand_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer specifies mandatory CE/CPE topics with specific hours for Ohio CPA (if any exist), or explicitly states there are no such mandatory topic-hour requirements.",
        node=leaf_ans,
        additional_instruction="Look for topics like ethics, professional standards, etc., with hours; or a clear 'none specified' statement."
    )
    refs = _combine_sources(info.official_reference_urls, info.verification_support_urls)
    refs_exist = evaluator.add_custom_node(
        result=len(refs) > 0,
        id="OH_Mandatory_CE_Topics_Refs_Exist",
        desc="Official reference URL(s) for Ohio CPA mandatory topics exist",
        parent=mand_node,
        critical=True
    )
    leaf_sup = evaluator.add_leaf(
        id="OH_Mandatory_CE_Topics_Supported",
        desc="Official references support the stated mandatory topic-hour requirements (or confirm none)",
        parent=mand_node,
        critical=True
    )
    claim_sup = (info.mandatory_ce_items and len(info.mandatory_ce_items) > 0)
    if claim_sup:
        claim_text = "Official sources support the stated mandatory CE/CPE topic-hour requirements for Ohio CPA."
    else:
        claim_text = "Official sources indicate that there are no specific mandatory topic-hour requirements for Ohio CPA beyond general totals."
    await evaluator.verify(
        claim=claim_text,
        node=leaf_sup,
        sources=refs,
        additional_instruction="Check Accountancy Board of Ohio pages or rules for specific mandatory topics (e.g., ethics/PSR).",
        extra_prerequisites=[refs_exist]
    )

    await _verify_first_time_renewal(evaluator, node, "OH", "Ohio CPA", info)
    await _verify_provider_approval(evaluator, node, "OH", "Ohio CPA", info)
    await _verify_association_recommendation(evaluator, node, "OH", "Ohio CPA", info, expected_statement=None)
    await _verify_compact_status(evaluator, node, "OH", "Ohio CPA", info)
    await _verify_official_references_present(evaluator, node, "OH", "Ohio CPA", info)


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
    Evaluate a single answer for the multi-state licensure guide task and return a structured result dictionary.
    """
    # Initialize evaluator (Root as parallel, non-critical to allow mixed critical children)
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=LicensureGuideExtraction,
        extraction_name="licensure_guide_extraction"
    )

    # Optional: record expected constraints as ground truth to aid analysis (not used for scoring by the framework)
    evaluator.add_ground_truth({
        "expected_constraints": {
            "federal_irs_127": "$5,250",
            "CA_LPCC": {"total_ce": "36", "cycle": "every 2 years", "mandatory_min": ["6 hours law and ethics"]},
            "TX_LPC": {"total_ce": "24", "cycle": "every 2 years", "mandatory_min": ["6 hours ethics", "3 hours cultural diversity/competency"]},
            "LA_LCSW": {"total_ce": "20", "cycle": "annual", "mandatory_min": ["10 hours clinical social work"]},
            "OH_CPA": {"total_ce": "120", "cycle": "every 3 years", "annual_min": "20"}
        }
    }, gt_type="ground_truth")

    # Build tree according to rubric

    # Global Federal Educational Assistance
    await verify_global_federal(evaluator, root, extraction.global_federal or GlobalFederalInfo())

    # California — LPCC
    await verify_ca_lpcc(evaluator, root, extraction.ca_lpcc or ProfessionStateInfo())

    # Texas — LPC
    await verify_tx_lpc(evaluator, root, extraction.tx_lpc or ProfessionStateInfo())

    # Louisiana — LCSW
    await verify_la_lcsw(evaluator, root, extraction.la_lcsw or ProfessionStateInfo())

    # Ohio — CPA
    await verify_oh_cpa(evaluator, root, extraction.oh_cpa or ProfessionStateInfo())

    # Return evaluation summary
    return evaluator.get_summary()