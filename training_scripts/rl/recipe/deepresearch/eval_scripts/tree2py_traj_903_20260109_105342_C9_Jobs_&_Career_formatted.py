import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nursing_eligibility_multi_state_2026"
TASK_DESCRIPTION = (
    "A healthcare staffing agency is establishing standardized eligibility criteria for four specialized nursing positions across different U.S. states. "
    "Each position requires specific ANCC certifications and compliance with state regulations. Based on current ANCC certification requirements (effective January 2026), "
    "Nurse Licensure Compact (NLC) rules, and state-specific regulations, identify the complete set of mandatory requirements that a nurse must meet to be eligible for each position:\n\n"
    "Position 1: Medical-Surgical Staff Nurse in Florida (NLC compact state) — ANCC MEDSURG-BC\n"
    "Position 2: Family Nurse Practitioner in Pennsylvania (NLC implemented July 7, 2025) — ANCC FNP-BC\n"
    "Position 3: Cardiac Vascular Staff Nurse in New York (non-compact) — ANCC CV-BC\n"
    "Position 4: Adult-Gerontology Primary Care NP in California with prescriptive authority — ANCC AGPCNP-BC\n\n"
    "For each position, capture: (a) License requirements; (b) ANCC eligibility criteria; (c) State CE for maintenance; "
    "(d) APRN-specific requirements if applicable; (e) Documentation/verification; include numeric values/timeframes and authoritative source URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    label: Optional[str] = None
    detail: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NLCRequirements(BaseModel):
    status: Optional[str] = None  # e.g., "Florida is an NLC compact state (fully implemented)."
    implementation_date: Optional[str] = None  # e.g., for Pennsylvania: "July 7, 2025"
    primary_residency_required: Optional[bool] = None
    uniform_licensure_required: Optional[bool] = None
    multistate_privilege_scope: Optional[str] = None  # e.g., "Allows practice in all compact states"
    urls: List[str] = Field(default_factory=list)


class LicenseRequirements(BaseModel):
    license_to_practice: Optional[str] = None  # e.g., "Hold FL RN license or NLC multistate privilege"
    nlc: Optional[NLCRequirements] = None
    urls: List[str] = Field(default_factory=list)


class ANCCEligibility(BaseModel):
    practice_hours: Optional[str] = None  # e.g., "2,000 hours in last 3 years"
    rn_experience: Optional[str] = None  # e.g., "2 years full-time equivalent RN experience"
    ce_hours: Optional[str] = None  # e.g., "30 contact hours in specialty in last 3 years"
    degree_requirement: Optional[str] = None  # e.g., "Bachelor's or higher (if applicable per ANCC page)"
    active_rn_license: Optional[str] = None  # e.g., "Current, active RN license in a U.S. state/territory"
    program_accreditation: Optional[str] = None  # e.g., "Program accredited by CCNE/ACEN/NLN CNEA (for APRN)"
    supervised_clinical_hours: Optional[str] = None  # e.g., "≥500 faculty-supervised clinical hours"
    exam_timeline: Optional[str] = None  # e.g., "Pass APRN exam within 5 years of degree conferral (effective 1/1/2026)"
    urls: List[str] = Field(default_factory=list)


class StateCEMaintenance(BaseModel):
    items: List[RequirementItem] = Field(default_factory=list)


class DocumentationRequirements(BaseModel):
    practice_hours_verification: Optional[str] = None  # e.g., "Supervisor letter on organizational letterhead"
    other_required_docs: Optional[str] = None  # e.g., transcripts, certification verification, CE certificates, etc.
    urls: List[str] = Field(default_factory=list)


class PositionData(BaseModel):
    license: Optional[LicenseRequirements] = None
    ancc: Optional[ANCCEligibility] = None
    state_ce: Optional[StateCEMaintenance] = None
    aprn_applicable: Optional[bool] = None  # For RN roles, should be False; for NP roles, True
    aprn_note: Optional[str] = None  # Explanation (e.g., "Not applicable to staff RN roles")
    documentation: Optional[DocumentationRequirements] = None
    sources: List[str] = Field(default_factory=list)  # All authoritative URLs cited for this position


class AllPositionsExtraction(BaseModel):
    position1: Optional[PositionData] = None
    position2: Optional[PositionData] = None
    position3: Optional[PositionData] = None
    position4: Optional[PositionData] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract from the answer the complete set of mandatory requirements for each of the four positions, organized into the schema below. You must return exact numeric values and timeframes as stated in the answer, and include the specific authoritative URLs the answer cites for each item.

GENERAL INSTRUCTIONS:
- Capture exactly what the answer claims, without adding anything not present in the answer.
- For numeric requirements, include the numbers/timeframes verbatim (e.g., "2,000 hours within the last 3 years", "30 contact hours/3 years", "500 supervised clinical hours", "within 5 years of degree conferral", "4 hours every 3 years", "4 hours every 4 years", "30 hours every 2 years").
- For Nurse Licensure Compact (NLC) details, extract: compact status (and implementation date if mentioned), whether declaring a primary state of residency is required, whether uniform licensure requirements must be met, and the scope of multistate privilege.
- For ANCC eligibility, include practice hours, RN experience, CE hours, degree requirement (do not assume BSN unless explicitly stated), active RN license requirement, APRN program accreditation (CCNE/ACEN/NLN CNEA), supervised clinical hours (e.g., 500 minimum), and exam timeline policy (e.g., pass within 5 years; effective date if mentioned).
- For state CE maintenance, list each requirement as a separate item with detail and URLs (e.g., FL RN renewal CE totals and timeframes; PA RN/APRN CE; NY general CE and infection control; CA 30 hours/2 years).
- For documentation/verification, include practice-hour verification method (e.g., supervisor letter on organizational letterhead) if the answer mentions it, and any other mandatory documents (e.g., transcripts, CE certificates, certification verification). Provide the cited URLs.
- For APRN applicability, set aprn_applicable True for NP roles and False for staff RN roles. Include aprn_note clarifying applicability.

SCHEMA TO FILL (JSON):
{
  "position1": {
    "license": {
      "license_to_practice": string or null,
      "nlc": {
        "status": string or null,
        "implementation_date": string or null,
        "primary_residency_required": boolean or null,
        "uniform_licensure_required": boolean or null,
        "multistate_privilege_scope": string or null,
        "urls": [url, ...]
      } or null,
      "urls": [url, ...]
    },
    "ancc": {
      "practice_hours": string or null,
      "rn_experience": string or null,
      "ce_hours": string or null,
      "degree_requirement": string or null,
      "active_rn_license": string or null,
      "program_accreditation": string or null,
      "supervised_clinical_hours": string or null,
      "exam_timeline": string or null,
      "urls": [url, ...]
    },
    "state_ce": {
      "items": [
        {"label": string or null, "detail": string or null, "urls": [url, ...]},
        ...
      ]
    },
    "aprn_applicable": boolean or null,
    "aprn_note": string or null,
    "documentation": {
      "practice_hours_verification": string or null,
      "other_required_docs": string or null,
      "urls": [url, ...]
    },
    "sources": [url, ...]
  },
  "position2": { ... same fields ... },
  "position3": { ... same fields ... },
  "position4": { ... same fields ... }
}

Return null for any missing field. Only include URLs that are explicitly present in the answer (plain or markdown). Ensure all arrays are present even if empty.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists, deduplicate while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url or not isinstance(url, str):
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _first_nonempty_detail(items: Optional[List[RequirementItem]]) -> Optional[str]:
    if not items:
        return None
    for it in items:
        if it and it.detail and it.detail.strip():
            return it.detail.strip()
    return None


def _collect_item_urls(items: Optional[List[RequirementItem]]) -> List[str]:
    if not items:
        return []
    urls: List[str] = []
    for it in items:
        urls.extend(it.urls or [])
    return _merge_sources(urls)


async def _add_verification_leaf(
    evaluator: Evaluator,
    *,
    parent,
    leaf_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]] = None,
    critical: bool = True,
    add_ins: Optional[str] = None
):
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources or None,
        additional_instruction=add_ins or "None"
    )


# --------------------------------------------------------------------------- #
# Position verifiers                                                          #
# --------------------------------------------------------------------------- #
async def verify_position_1(evaluator: Evaluator, parent, pos: Optional[PositionData]):
    # Parent node for Position 1
    pos_node = evaluator.add_parallel(
        id="position_1_florida_medsurg",
        desc="Position 1: Medical-Surgical Staff Nurse in Florida (MEDSURG-BC) — mandatory requirements by category.",
        parent=parent,
        critical=False
    )

    pos = pos or PositionData()
    lic = pos.license or LicenseRequirements()
    nlc = lic.nlc or NLCRequirements()
    ancc = pos.ancc or ANCCEligibility()
    ce = pos.state_ce or StateCEMaintenance()
    doc = pos.documentation or DocumentationRequirements()
    pos_sources = pos.sources or []

    # License requirements node
    lic_node = evaluator.add_parallel(
        id="pos1_license_requirements",
        desc="License requirements for practicing as an RN in Florida, including NLC applicability and compact-eligibility criteria.",
        parent=pos_node,
        critical=True
    )
    lic_sources = _merge_sources(lic.urls, nlc.urls, pos_sources)

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos1_fl_compact_status",
        desc="Correctly indicates Florida is a fully implemented NLC compact state.",
        claim="Florida is a fully implemented Nurse Licensure Compact (NLC) state.",
        sources=lic_sources,
        critical=True,
        add_ins="Use official NCSBN or Florida Board of Nursing pages to verify Florida's NLC compact status."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos1_license_to_practice_in_fl",
        desc="States what license status is required to legally practice as an RN in Florida.",
        claim="To practice as an RN in Florida, a nurse must hold either a Florida RN license or a valid NLC multistate license that authorizes practice in Florida.",
        sources=lic_sources,
        critical=True,
        add_ins="Confirm practice authority via FL Board of Nursing and/or NCSBN NLC guidance. Do not require Florida to be the primary residence if using NLC; only that multistate privilege allows practice in Florida."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos1_compact_declares_primary_residency",
        desc="NLC: includes requirement to declare a primary state of residency.",
        claim="Under the NLC, nurses must declare a primary state of legal residency to be eligible for a multistate license.",
        sources=lic_sources,
        critical=True,
        add_ins="Verify using NCSBN NLC rules (primary state of legal residence requirement)."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos1_compact_meets_uniform_licensure_requirements",
        desc="NLC: includes uniform licensure requirements.",
        claim="Under the NLC, nurses must meet the Compact's uniform licensure requirements to obtain a multistate license.",
        sources=lic_sources,
        critical=True,
        add_ins="Verify using NCSBN's Uniform Licensure Requirements documentation."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos1_compact_multistate_privilege_scope",
        desc="NLC: correctly states scope of multistate privilege.",
        claim="An NLC multistate license grants the privilege to practice in all compact states, including Florida.",
        sources=lic_sources,
        critical=True,
        add_ins="Use NCSBN NLC overview showing that a multistate license confers practice privilege in other compact states."
    )

    # ANCC eligibility node
    ancc_node = evaluator.add_parallel(
        id="pos1_ancc_eligibility",
        desc="ANCC MEDSURG-BC eligibility criteria.",
        parent=pos_node,
        critical=True
    )
    ancc_sources = _merge_sources(ancc.urls, pos_sources)

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos1_ancc_practice_hours",
        desc="Includes required clinical practice hours and timeframe.",
        claim="ANCC MEDSURG-BC eligibility requires at least 2,000 hours of clinical practice in medical-surgical nursing within the last 3 years.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the official ANCC MEDSURG-BC eligibility page. Accept equivalent phrasing and exact numeric/timeframe match."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos1_ancc_rn_experience",
        desc="Includes RN experience requirement.",
        claim="ANCC MEDSURG-BC eligibility requires 2 years of full-time equivalent RN experience.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the official ANCC MEDSURG-BC eligibility page."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos1_ancc_ce",
        desc="Includes continuing education contact-hour requirement and timeframe.",
        claim="ANCC MEDSURG-BC eligibility requires 30 contact hours of continuing education in medical-surgical nursing within the last 3 years.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the ANCC MEDSURG-BC page; numeric value and timeframe must match."
    )

    degree_claim_text = ancc.degree_requirement or "The ANCC MEDSURG-BC education degree requirement is as specified by ANCC's official eligibility criteria."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos1_ancc_degree_requirement",
        desc="States the education degree requirement for MEDSURG-BC as required by ANCC.",
        claim=f"ANCC MEDSURG-BC education requirement: {degree_claim_text}",
        sources=ancc_sources,
        critical=True,
        add_ins="Do not assume a BSN unless ANCC explicitly requires it. Verify the exact degree requirement text on ANCC's eligibility page."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos1_ancc_active_rn_license",
        desc="Includes requirement of an active RN license.",
        claim="ANCC MEDSURG-BC eligibility requires a current, active RN license in a U.S. state or territory.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC's eligibility page."
    )

    # State CE maintenance node
    ce_node = evaluator.add_parallel(
        id="pos1_state_ce_maintenance",
        desc="Florida state-specific CE requirements for RN license maintenance.",
        parent=pos_node,
        critical=True
    )
    ce_desc = _first_nonempty_detail(ce.items)
    ce_urls = _merge_sources(_collect_item_urls(ce.items), pos_sources)
    ce_claim_text = (
        f"Florida RN license renewal CE requirements: {ce_desc}"
        if ce_desc else "Florida RN license renewal requires continuing education with specified numeric hours and renewal timeframes."
    )
    await _add_verification_leaf(
        evaluator,
        parent=ce_node,
        leaf_id="pos1_fl_ce_numeric_and_timeframe",
        desc="Provides Florida RN CE requirements with numeric values and renewal timeframe.",
        claim=ce_claim_text,
        sources=ce_urls,
        critical=True,
        add_ins="Verify using the Florida Board of Nursing or official state sources. Confirm numeric hours and renewal interval."
    )

    # APRN applicability node
    aprn_node = evaluator.add_parallel(
        id="pos1_aprn_applicability",
        desc="APRN-specific requirements handling (if applicable).",
        parent=pos_node,
        critical=True
    )
    aprn_note = pos.aprn_note or "For a staff RN role, APRN-specific requirements (program accreditation, supervised hours, exam timeline) are not applicable."
    await _add_verification_leaf(
        evaluator,
        parent=aprn_node,
        leaf_id="pos1_aprn_not_applicable",
        desc="Correctly indicates APRN requirements are not applicable for a staff RN position.",
        claim=aprn_note,
        sources=None,
        critical=True,
        add_ins="Judge using the task context and the answer: the role is a staff RN (not an APRN), so APRN-specific requirements are not applicable."
    )

    # Documentation/verification node
    doc_node = evaluator.add_parallel(
        id="pos1_documentation_verification",
        desc="Documentation/verification requirements used to validate eligibility.",
        parent=pos_node,
        critical=True
    )
    doc_sources = _merge_sources(doc.urls, ancc_sources, pos_sources)

    phv_text = doc.practice_hours_verification or "Practice hours, when required by ANCC specialty certifications, must be verified by a supervisor letter on organizational letterhead."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos1_practice_hours_verification",
        desc="Includes required verification method for practice hours.",
        claim=phv_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify on ANCC application/handbook/eligibility documentation that practice hours are verified via supervisor letter on organizational letterhead (when practice hours are required)."
    )

    other_docs_text = doc.other_required_docs or "Other mandatory documentation (e.g., transcripts, certification verification, CE certificates) is required as specified by ANCC or the state board."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos1_other_required_docs",
        desc="Lists other mandatory documentation needed for eligibility validation.",
        claim=other_docs_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify against ANCC/state/NLC official pages referenced in the answer."
    )

    # Sources leaf
    await _add_verification_leaf(
        evaluator,
        parent=pos_node,
        leaf_id="pos1_sources",
        desc="Provides authoritative source URLs supporting Position 1 requirements.",
        claim="The cited URLs provide official information supporting the Florida RN licensure/NLC rules, ANCC MEDSURG-BC eligibility, Florida CE maintenance, and required documentation.",
        sources=_merge_sources(pos_sources, lic_sources, ancc_sources, ce_urls, doc_sources),
        critical=True,
        add_ins="At least one of the URLs should be official (ANCC, NCSBN, state board/government) and substantively support the requirements."
    )


async def verify_position_2(evaluator: Evaluator, parent, pos: Optional[PositionData]):
    pos_node = evaluator.add_parallel(
        id="position_2_pennsylvania_fnp",
        desc="Position 2: Family Nurse Practitioner in Pennsylvania (FNP-BC) — mandatory requirements by category.",
        parent=parent,
        critical=False
    )

    pos = pos or PositionData()
    lic = pos.license or LicenseRequirements()
    nlc = lic.nlc or NLCRequirements()
    ancc = pos.ancc or ANCCEligibility()
    ce = pos.state_ce or StateCEMaintenance()
    doc = pos.documentation or DocumentationRequirements()
    pos_sources = pos.sources or []

    # License requirements
    lic_node = evaluator.add_parallel(
        id="pos2_license_requirements",
        desc="License requirements for legally practicing as an FNP in Pennsylvania, including NLC applicability.",
        parent=pos_node,
        critical=True
    )
    lic_sources = _merge_sources(lic.urls, nlc.urls, pos_sources)

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos2_pa_compact_status_and_date",
        desc="Correctly indicates PA is NLC with implementation date.",
        claim="Pennsylvania participates in the Nurse Licensure Compact; implementation effective July 7, 2025.",
        sources=lic_sources,
        critical=True,
        add_ins="Verify using NCSBN or the Pennsylvania State Board of Nursing official sources."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos2_compact_declares_primary_residency",
        desc="NLC: primary state of residency required.",
        claim="Under the NLC, nurses must declare a primary state of legal residency to obtain a multistate license.",
        sources=lic_sources,
        critical=True,
        add_ins="Use NCSBN NLC rules to verify."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos2_compact_meets_uniform_licensure_requirements",
        desc="NLC: uniform licensure requirements must be met.",
        claim="Under the NLC, nurses must meet Uniform Licensure Requirements to be eligible for a multistate license.",
        sources=lic_sources,
        critical=True,
        add_ins="Use NCSBN official documentation of ULR."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos2_compact_multistate_privilege_scope",
        desc="NLC: multistate license allows practice in compact states.",
        claim="An NLC multistate license permits practice in all compact states.",
        sources=lic_sources,
        critical=True,
        add_ins="Use NCSBN NLC overview pages."
    )

    license_to_practice_pa = lic.license_to_practice or "Pennsylvania requires appropriate RN licensure and APRN/CRNP authorization to practice as an FNP, consistent with state board rules."
    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos2_license_to_practice_as_aprn_in_pa",
        desc="States Pennsylvania requirements to practice as an APRN/FNP.",
        claim=license_to_practice_pa,
        sources=lic_sources,
        critical=True,
        add_ins="Verify with Pennsylvania State Board of Nursing APRN/CRNP (FNP) authorization/recognition requirements. Include any numeric/timeframes if stated."
    )

    # ANCC eligibility
    ancc_node = evaluator.add_parallel(
        id="pos2_ancc_eligibility",
        desc="ANCC FNP-BC eligibility criteria.",
        parent=pos_node,
        critical=True
    )
    ancc_sources = _merge_sources(ancc.urls, pos_sources)

    program_accr = ancc.program_accreditation or "Graduation from an accredited NP program (e.g., CCNE, ACEN, or NLN CNEA)."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos2_program_accreditation",
        desc="Includes NP program accreditation requirement.",
        claim=program_accr,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC FNP-BC eligibility that the NP program must be accredited by CCNE, ACEN, or NLN CNEA (or equivalent language)."
    )

    supervised_hours = ancc.supervised_clinical_hours or "At least 500 faculty-supervised clinical hours in the FNP role/population focus are required."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos2_supervised_clinical_hours",
        desc="Includes minimum supervised clinical hours (500).",
        claim=supervised_hours,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC FNP-BC eligibility that ≥500 supervised clinical hours are required."
    )

    exam_timeline = ancc.exam_timeline or "APRN candidates must pass the ANCC certification exam within 5 years of degree conferral (policy effective January 1, 2026)."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos2_exam_timeline_policy",
        desc="Includes exam timeline policy (5 years; effective 1/1/2026).",
        claim=exam_timeline,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC policy pages about the 5-year exam pass requirement effective Jan 1, 2026."
    )

    degree_req = ancc.degree_requirement or "The ANCC FNP-BC degree requirement is as specified by ANCC's eligibility criteria (do not assume degree level without source)."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos2_degree_requirement",
        desc="States ANCC degree requirement for FNP-BC (supported by sources).",
        claim=degree_req,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify degree requirement exactly as stated on ANCC eligibility page; do not infer a specific degree level not supported by ANCC."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos2_ancc_active_rn_license",
        desc="Includes requirement of an active RN license.",
        claim="ANCC FNP-BC eligibility requires a current, active RN license in a U.S. state or territory.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the ANCC FNP-BC eligibility page."
    )

    # State CE maintenance
    ce_node = evaluator.add_parallel(
        id="pos2_state_ce_maintenance",
        desc="PA continuing education requirements for maintaining RN/APRN authorization for FNP practice.",
        parent=pos_node,
        critical=True
    )
    ce_desc = _first_nonempty_detail(ce.items)
    ce_urls = _merge_sources(_collect_item_urls(ce.items), pos_sources)
    ce_claim_text = ce_desc or "Pennsylvania requires continuing education for RN/APRN maintenance; the answer provides numeric values/timeframes."
    await _add_verification_leaf(
        evaluator,
        parent=ce_node,
        leaf_id="pos2_pa_ce_numeric_and_timeframe",
        desc="Provides PA CE requirements with specific numeric values and renewal timeframes.",
        claim=ce_claim_text,
        sources=ce_urls,
        critical=True,
        add_ins="Verify using Pennsylvania State Board of Nursing (official) sources."
    )

    # Documentation/verification
    doc_node = evaluator.add_parallel(
        id="pos2_documentation_verification",
        desc="Documentation/verification requirements (ANCC/state/NLC as applicable).",
        parent=pos_node,
        critical=True
    )
    doc_sources = _merge_sources(doc.urls, ancc_sources, pos_sources)

    phv_text = doc.practice_hours_verification or "If any practice-hour verification is required, it must be confirmed by a supervisor letter on organizational letterhead."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos2_practice_hours_verification_if_applicable",
        desc="Includes practice hours verification method if applicable.",
        claim=phv_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify against ANCC's documentation rules; this applies when practice hours are part of the eligibility requirements."
    )

    other_docs_text = doc.other_required_docs or "Other mandatory documentation (e.g., transcripts, national certification, CE certificates) is required as specified by ANCC/state board."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos2_other_required_docs",
        desc="Lists other mandatory documentation needed for eligibility validation.",
        claim=other_docs_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify using ANCC and/or Pennsylvania State Board official pages."
    )

    # Sources leaf
    await _add_verification_leaf(
        evaluator,
        parent=pos_node,
        leaf_id="pos2_sources",
        desc="Provides authoritative source URLs supporting Position 2 requirements.",
        claim="The cited URLs provide official information supporting Pennsylvania FNP licensure/authorization, NLC rules, ANCC FNP-BC eligibility, CE maintenance, and required documentation.",
        sources=_merge_sources(pos_sources, lic_sources, ancc_sources, ce_urls, doc_sources),
        critical=True,
        add_ins="At least one URL should be official (ANCC, NCSBN, PA Board of Nursing/government) and substantively support the requirements."
    )


async def verify_position_3(evaluator: Evaluator, parent, pos: Optional[PositionData]):
    pos_node = evaluator.add_parallel(
        id="position_3_newyork_cv",
        desc="Position 3: Cardiac Vascular Staff Nurse in New York (CV-BC) — mandatory requirements by category.",
        parent=parent,
        critical=False
    )

    pos = pos or PositionData()
    lic = pos.license or LicenseRequirements()
    nlc = lic.nlc or NLCRequirements()
    ancc = pos.ancc or ANCCEligibility()
    ce = pos.state_ce or StateCEMaintenance()
    doc = pos.documentation or DocumentationRequirements()
    pos_sources = pos.sources or []

    # License requirements
    lic_node = evaluator.add_parallel(
        id="pos3_license_requirements",
        desc="License requirements for practicing as an RN in New York (non-compact).",
        parent=pos_node,
        critical=True
    )
    lic_sources = _merge_sources(lic.urls, nlc.urls, pos_sources)

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos3_ny_non_compact_status",
        desc="Correctly indicates New York is not an NLC compact state.",
        claim="New York is not a Nurse Licensure Compact (NLC) state.",
        sources=lic_sources,
        critical=True,
        add_ins="Verify using NCSBN compact status listings and/or New York State Board pages."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos3_ny_license_to_practice",
        desc="States NY RN licensure requirement to practice.",
        claim="To practice as an RN in New York, the nurse must meet New York RN licensure requirements and hold authorization to practice in New York.",
        sources=lic_sources,
        critical=True,
        add_ins="Verify using New York State Education Department/Board of Nursing official pages."
    )

    # ANCC eligibility (CV-BC)
    ancc_node = evaluator.add_parallel(
        id="pos3_ancc_eligibility",
        desc="ANCC CV-BC eligibility criteria.",
        parent=pos_node,
        critical=True
    )
    ancc_sources = _merge_sources(ancc.urls, pos_sources)

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos3_ancc_practice_hours",
        desc="Includes practice hours and timeframe for CV-BC.",
        claim="ANCC Cardiac Vascular Nursing (CV-BC) eligibility requires at least 2,000 hours of clinical practice in the specialty within the last 3 years.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the ANCC CV-BC eligibility page."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos3_ancc_rn_experience",
        desc="Includes RN experience requirement for CV-BC.",
        claim="ANCC CV-BC eligibility requires 2 years of full-time equivalent RN experience.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the ANCC CV-BC eligibility page."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos3_ancc_ce",
        desc="Includes CE requirement and timeframe for CV-BC.",
        claim="ANCC CV-BC eligibility requires 30 contact hours of continuing education in the specialty within the last 3 years.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on the ANCC CV-BC eligibility page."
    )

    degree_req = ancc.degree_requirement or "The ANCC CV-BC education degree requirement is as specified by ANCC's eligibility criteria."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos3_ancc_degree_requirement",
        desc="States the education degree requirement for CV-BC.",
        claim=degree_req,
        sources=ancc_sources,
        critical=True,
        add_ins="Do not assume a specific degree level unless ANCC explicitly requires it. Verify on the ANCC CV-BC eligibility page."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos3_ancc_active_rn_license",
        desc="Includes the requirement of an active RN license.",
        claim="ANCC CV-BC eligibility requires a current, active RN license in a U.S. state or territory.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC CV-BC eligibility."
    )

    # State CE maintenance (NY)
    ce_node = evaluator.add_parallel(
        id="pos3_state_ce_maintenance",
        desc="New York continuing education requirements for RN license maintenance.",
        parent=pos_node,
        critical=True
    )
    ce_urls = _merge_sources(_collect_item_urls(ce.items), pos_sources)

    await _add_verification_leaf(
        evaluator,
        parent=ce_node,
        leaf_id="pos3_ny_ce_4_hours_3_years",
        desc="Includes NY requirement: 4 contact hours every 3 years for RNs.",
        claim="New York requires 4 contact hours of continuing education every 3 years for RNs.",
        sources=ce_urls,
        critical=True,
        add_ins="Verify using New York State official CE requirements for RNs with numeric/timeframe match."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ce_node,
        leaf_id="pos3_ny_infection_control_4_hours_4_years",
        desc="Includes NY infection control training: 4 hours every 4 years.",
        claim="New York requires 4 hours of infection control training every 4 years for RNs.",
        sources=ce_urls,
        critical=True,
        add_ins="Verify using New York State official infection control training requirement pages."
    )

    # APRN applicability
    aprn_node = evaluator.add_parallel(
        id="pos3_aprn_applicability",
        desc="APRN-specific requirements handling (if applicable).",
        parent=pos_node,
        critical=True
    )
    aprn_note = pos.aprn_note or "For a staff RN position, APRN-specific requirements are not applicable."
    await _add_verification_leaf(
        evaluator,
        parent=aprn_node,
        leaf_id="pos3_aprn_not_applicable",
        desc="Correctly indicates APRN requirements are not applicable for a staff RN position.",
        claim=aprn_note,
        sources=None,
        critical=True,
        add_ins="Judge using the task context and the answer: this is a staff RN role, not APRN."
    )

    # Documentation/verification
    doc_node = evaluator.add_parallel(
        id="pos3_documentation_verification",
        desc="Documentation/verification requirements (ANCC/state as applicable).",
        parent=pos_node,
        critical=True
    )
    doc_sources = _merge_sources(doc.urls, ancc_sources, pos_sources)

    phv_text = doc.practice_hours_verification or "Practice hours for ANCC specialty certifications, when required, must be verified via supervisor letter on organizational letterhead."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos3_practice_hours_verification",
        desc="Includes practice hours verification method.",
        claim=phv_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify on ANCC documentation guidance."
    )

    other_docs_text = doc.other_required_docs or "Other mandatory documentation (e.g., transcripts, CE certificates) is required per ANCC/state rules."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos3_other_required_docs",
        desc="Lists other mandatory documentation.",
        claim=other_docs_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify using ANCC/state official pages."
    )

    # Sources leaf
    await _add_verification_leaf(
        evaluator,
        parent=pos_node,
        leaf_id="pos3_sources",
        desc="Provides authoritative source URLs supporting Position 3 requirements.",
        claim="The cited URLs provide official information supporting New York RN licensure, ANCC CV-BC eligibility, NY CE maintenance (including infection control), and documentation requirements.",
        sources=_merge_sources(pos_sources, lic_sources, ancc_sources, ce_urls, doc_sources),
        critical=True,
        add_ins="At least one URL should be official (ANCC, state board/government) and substantively support the requirements."
    )


async def verify_position_4(evaluator: Evaluator, parent, pos: Optional[PositionData]):
    pos_node = evaluator.add_parallel(
        id="position_4_california_agpcnp",
        desc="Position 4: Adult-Gerontology Primary Care NP in California with prescriptive authority (AGPCNP-BC) — mandatory requirements by category.",
        parent=parent,
        critical=False
    )

    pos = pos or PositionData()
    lic = pos.license or LicenseRequirements()
    nlc = lic.nlc or NLCRequirements()
    ancc = pos.ancc or ANCCEligibility()
    ce = pos.state_ce or StateCEMaintenance()
    doc = pos.documentation or DocumentationRequirements()
    pos_sources = pos.sources or []

    # License requirements
    lic_node = evaluator.add_parallel(
        id="pos4_license_requirements",
        desc="License requirements for practicing as an NP in California and prescriptive authority prerequisites.",
        parent=pos_node,
        critical=True
    )
    lic_sources = _merge_sources(lic.urls, nlc.urls, pos_sources)

    ca_nlc_status = nlc.status or "California does not participate in the NLC; NLC multistate privilege is not applicable for practicing in California."
    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos4_ca_compact_applicability",
        desc="Correctly states whether NLC multistate privilege is applicable in California.",
        claim=ca_nlc_status,
        sources=lic_sources,
        critical=True,
        add_ins="Verify using NCSBN compact status and California Board of Registered Nursing (BRN) official sources."
    )

    ca_license_to_practice = lic.license_to_practice or "California requires California RN licensure and NP recognition/approval (APRN) to practice as an NP in the state."
    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos4_ca_license_to_practice_as_aprn",
        desc="States what California requires to practice as an NP.",
        claim=ca_license_to_practice,
        sources=lic_sources,
        critical=True,
        add_ins="Verify using CA BRN official pages for RN licensure and NP recognition/approval."
    )

    await _add_verification_leaf(
        evaluator,
        parent=lic_node,
        leaf_id="pos4_ca_prescriptive_authority_pharmacology",
        desc="Includes CA prescriptive authority pharmacology CE requirement (additional 3 contact hours).",
        claim="California NP prescriptive authority requires an additional 3 contact hours in pharmacology (furnishing requirements).",
        sources=lic_sources,
        critical=True,
        add_ins="Verify using CA BRN official prescriptive authority/furnishing requirements pages."
    )

    # ANCC eligibility (AGPCNP-BC)
    ancc_node = evaluator.add_parallel(
        id="pos4_ancc_eligibility",
        desc="ANCC AGPCNP-BC eligibility criteria.",
        parent=pos_node,
        critical=True
    )
    ancc_sources = _merge_sources(ancc.urls, pos_sources)

    program_accr = ancc.program_accreditation or "Graduation from a program accredited by CCNE, ACEN, or NLN CNEA."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos4_program_accreditation",
        desc="Includes NP program accreditation requirement.",
        claim=program_accr,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC AGPCNP-BC eligibility."
    )

    supervised_hours = ancc.supervised_clinical_hours or "At least 500 faculty-supervised clinical hours in the AGPCNP role/population focus are required."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos4_supervised_clinical_hours",
        desc="Includes minimum supervised clinical hours (500).",
        claim=supervised_hours,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC AGPCNP-BC eligibility."
    )

    exam_timeline = ancc.exam_timeline or "APRN candidates must pass the ANCC certification exam within 5 years of degree conferral (policy effective January 1, 2026)."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos4_exam_timeline_policy",
        desc="Includes exam timeline requirement (5 years; effective 1/1/2026).",
        claim=exam_timeline,
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC policy pages."
    )

    degree_req = ancc.degree_requirement or "The ANCC AGPCNP-BC degree requirement is as specified by ANCC's eligibility criteria."
    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos4_degree_requirement",
        desc="States ANCC degree requirement for AGPCNP-BC (supported by sources).",
        claim=degree_req,
        sources=ancc_sources,
        critical=True,
        add_ins="Do not assume a specific degree level unless ANCC explicitly requires it. Verify on ANCC eligibility page."
    )

    await _add_verification_leaf(
        evaluator,
        parent=ancc_node,
        leaf_id="pos4_ancc_active_rn_license",
        desc="Includes requirement of an active RN license.",
        claim="ANCC AGPCNP-BC eligibility requires a current, active RN license in a U.S. state or territory.",
        sources=ancc_sources,
        critical=True,
        add_ins="Verify on ANCC AGPCNP-BC eligibility."
    )

    # State CE maintenance (California)
    ce_node = evaluator.add_parallel(
        id="pos4_state_ce_maintenance",
        desc="California continuing education requirements for license maintenance.",
        parent=pos_node,
        critical=True
    )
    ce_desc = _first_nonempty_detail(ce.items)
    ce_urls = _merge_sources(_collect_item_urls(ce.items), pos_sources)
    await _add_verification_leaf(
        evaluator,
        parent=ce_node,
        leaf_id="pos4_ca_ce_30_hours_2_years",
        desc="Includes CA requirement: 30 hours of CE every 2 years for RN/NP renewal.",
        claim="California requires 30 hours of continuing education every 2 years for RN/NP license renewal.",
        sources=ce_urls,
        critical=True,
        add_ins="Verify using CA BRN official CE renewal requirements."
    )

    # Documentation/verification
    doc_node = evaluator.add_parallel(
        id="pos4_documentation_verification",
        desc="Documentation/verification requirements (ANCC/state/prescriptive authority as applicable).",
        parent=pos_node,
        critical=True
    )
    doc_sources = _merge_sources(doc.urls, ancc_sources, pos_sources)

    phv_text = doc.practice_hours_verification or "When practice-hour verification is required (for relevant certifications), it must be attested by a supervisor letter on organizational letterhead."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos4_practice_hours_verification_if_applicable",
        desc="Includes practice-hour verification method if applicable.",
        claim=phv_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify using ANCC official documentation requirements."
    )

    other_docs_text = doc.other_required_docs or "Other mandatory documentation (e.g., transcripts, CE certificates, national certification verification) is required per ANCC/state rules."
    await _add_verification_leaf(
        evaluator,
        parent=doc_node,
        leaf_id="pos4_other_required_docs",
        desc="Lists other mandatory documentation needed for eligibility validation.",
        claim=other_docs_text,
        sources=doc_sources,
        critical=True,
        add_ins="Verify using ANCC and CA BRN official pages."
    )

    # Sources leaf
    await _add_verification_leaf(
        evaluator,
        parent=pos_node,
        leaf_id="pos4_sources",
        desc="Provides authoritative source URLs supporting Position 4 requirements.",
        claim="The cited URLs provide official information supporting California NP licensure/recognition and prescriptive authority, ANCC AGPCNP-BC eligibility, CE maintenance, and documentation requirements.",
        sources=_merge_sources(pos_sources, lic_sources, ancc_sources, ce_urls, doc_sources),
        critical=True,
        add_ins="At least one URL should be official (ANCC, NCSBN, CA BRN/government) and substantively support the requirements."
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
    Evaluate an answer for the multi-state nursing eligibility task (Jan 2026 rules).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel to evaluate positions independently
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

    # Extract structured information once for all positions
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=AllPositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Build tree according to rubric (root parallel)
    # Position 1
    await verify_position_1(evaluator, root, extracted.position1)

    # Position 2
    await verify_position_2(evaluator, root, extracted.position2)

    # Position 3
    await verify_position_3(evaluator, root, extracted.position3)

    # Position 4
    await verify_position_4(evaluator, root, extracted.position4)

    # Return structured summary
    return evaluator.get_summary()