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
TASK_ID = "wv_prof_services_compliance"
TASK_DESCRIPTION = """
You are assisting an entrepreneur who is planning to establish four separate professional service businesses in West Virginia. The entrepreneur needs comprehensive information about the regulatory compliance requirements for each business type.

For each of the following four professional services, research and provide the complete regulatory compliance information:

1. Engineering services (Professional Engineers)
2. Architectural services (Architects)
3. Accounting services (Certified Public Accountants)
4. Clinical social work services (Licensed Social Workers)

For each professional service business, provide the following information:

A. Chapter 30 Regulation Status:
- Determine whether this profession is regulated under West Virginia Code Chapter 30 as a professional service
- If yes, identify the specific West Virginia state licensing board responsible for regulating this profession
- Include the official website URL or official contact information for the licensing board

B. Secretary of State Registration Requirements:
- Confirm whether a Verification of Eligibility (Form VOE) from the state licensing board is required to be submitted with the Secretary of State business registration
- Note: Form VOE must be signed by an authorized representative of the licensing board

C. Business Registration Certificate:
- Confirm whether a Business Registration Certificate from the West Virginia State Tax Department is required before engaging in business activity
- Note: Separate certificates are required for each business location

D. Annual Report Filing:
- Confirm whether an annual report must be filed with the West Virginia Secretary of State
- Provide the annual report filing fee amount
- Provide the annual report filing deadline period (the range of dates during which the report must be filed each year)
- Note any consequences of failure to file by the deadline

All information must be sourced from official West Virginia government websites or official documentation. Provide reference URLs for each piece of information to ensure verifiability.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SectionA(BaseModel):
    chapter30_status: Optional[str] = None
    licensing_board_name: Optional[str] = None
    licensing_board_contact_or_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionB(BaseModel):
    voe_required: Optional[str] = None
    voe_signature_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionC(BaseModel):
    brc_required: Optional[str] = None
    brc_separate_per_name_and_location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionD(BaseModel):
    annual_report_required: Optional[str] = None
    annual_report_fee: Optional[str] = None
    annual_report_deadline_period: Optional[str] = None
    annual_report_noncompliance_consequences: Optional[str] = None
    online_filing_portal: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProfessionCompliance(BaseModel):
    profession_label: Optional[str] = None
    section_a: Optional[SectionA] = None
    section_b: Optional[SectionB] = None
    section_c: Optional[SectionC] = None
    section_d: Optional[SectionD] = None


class WVComplianceExtraction(BaseModel):
    engineering: Optional[ProfessionCompliance] = None
    architecture: Optional[ProfessionCompliance] = None
    accounting: Optional[ProfessionCompliance] = None
    social_work: Optional[ProfessionCompliance] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_compliance() -> str:
    return """
    Extract the compliance information for the following four professional services in West Virginia exactly as presented in the answer. For each profession, extract Sections A–D with fields and supporting source URLs (official WV government or official licensing board documentation only).

    PROFESSIONS:
    - engineering: Professional Engineers (engineering services)
    - architecture: Architects (architectural services)
    - accounting: Certified Public Accountants (accounting services)
    - social_work: Licensed Social Workers (clinical social work services)

    For each profession, capture the following:

    Section A: Chapter 30 Regulation Status
    - chapter30_status: Whether regulated under WV Code Chapter 30 as a professional service (text as given; e.g., "Yes", "No")
    - licensing_board_name: The WV state licensing board name (exact text)
    - licensing_board_contact_or_url: The official website URL or official contact information (prefer a full URL if present; otherwise the exact contact info text)
    - sources: An array of URLs that support the section A information (official WV government domains like sos.wv.gov, tax.wv.gov, business4.wv.gov, or official WV board/commission sites)

    Section B: Secretary of State Registration Requirements
    - voe_required: Whether a Verification of Eligibility (Form VOE) must be submitted with the SOS registration (text as given)
    - voe_signature_requirement: Statement that VOE must be signed by an authorized representative of the licensing board (text as given)
    - sources: An array of URLs that support Section B (WV SOS or official documentation)

    Section C: Business Registration Certificate
    - brc_required: Whether a WV State Tax Department Business Registration Certificate is required before engaging in business activity (text as given)
    - brc_separate_per_name_and_location: Statement that separate certificates are required for each business name and location (text as given)
    - sources: An array of URLs that support Section C (WV Tax/Revenue or official documentation)

    Section D: Annual Report Filing
    - annual_report_required: Whether an annual report must be filed with the WV Secretary of State (text as given)
    - annual_report_fee: The annual report filing fee amount (text as given)
    - annual_report_deadline_period: The annual report filing deadline period/range (text as given)
    - annual_report_noncompliance_consequences: Consequences of failure to file by the deadline (text as given)
    - online_filing_portal: Statement that online annual report filing is available via the West Virginia One Stop Business Portal (business4.wv.gov) (text as given)
    - sources: An array of URLs that support Section D (WV SOS, business4.wv.gov, or official documentation)

    IMPORTANT:
    - Extract values exactly as they appear in the answer; do not invent or infer.
    - For URLs, extract only valid complete URLs. If missing protocol, prepend "http://".
    - If any required field is missing in the answer, return null for that field.
    - For each section (A/B/C/D), include all cited URLs in 'sources' for that subsection.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _text_has_affirmative(val: Optional[str]) -> bool:
    """Heuristic to treat free-form text as affirmative."""
    if not val:
        return False
    s = val.strip().lower()
    affirmative_keywords = ["yes", "required", "must", "mandatory", "needed", "shall"]
    negative_keywords = ["no", "not required", "optional", "not needed"]
    if any(k in s for k in affirmative_keywords) and not any(k in s for k in negative_keywords):
        return True
    if any(k in s for k in negative_keywords):
        return False
    # Default to False when ambiguous
    return False


def _ensure_list(lst: Optional[List[str]]) -> List[str]:
    return lst or []


def _is_url(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification logic per business                                             #
# --------------------------------------------------------------------------- #
async def verify_business(
    evaluator: Evaluator,
    parent_root,
    business_idx: int,
    business_desc: str,
    data: Optional[ProfessionCompliance],
) -> None:
    """
    Build and execute verification nodes for a single professional service business.
    """
    # Parent business node (parallel, non-critical to allow partial credit)
    biz_node = evaluator.add_parallel(
        id=f"business_{business_idx}",
        desc=business_desc,
        parent=parent_root,
        critical=False,
    )

    # If no data extracted at all, add existence gating nodes so all children will be skipped by preconditions
    has_any_data = data is not None
    evaluator.add_custom_node(
        result=has_any_data,
        id=f"b{business_idx}_extraction_present",
        desc="Extraction for this business exists in the answer",
        parent=biz_node,
        critical=True,
    )

    # Section shortcuts
    A = data.section_a if data else SectionA()
    B = data.section_b if data else SectionB()
    C = data.section_c if data else SectionC()
    D = data.section_d if data else SectionD()

    # ------------------------------- A. Chapter 30 ------------------------------- #
    # Existence gates
    evaluator.add_custom_node(
        result=bool(A.chapter30_status),
        id=f"b{business_idx}_chapter30_status_exists",
        desc="Section A: Chapter 30 status value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(A.licensing_board_name),
        id=f"b{business_idx}_licensing_board_exists",
        desc="Section A: Licensing board name exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(A.licensing_board_contact_or_url),
        id=f"b{business_idx}_board_contact_or_url_exists",
        desc="Section A: Licensing board official URL or contact exists",
        parent=biz_node,
        critical=True,
    )

    # Leaf: Chapter 30 status (verify with Section A sources)
    ch30_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_chapter30_status",
        desc="State whether the profession is regulated under WV Code Chapter 30 as a professional service",
        parent=biz_node,
        critical=True,
    )
    ch30_claim = (
        "This profession is regulated under West Virginia Code Chapter 30 as a professional service."
        if _text_has_affirmative(A.chapter30_status)
        else "This profession is not regulated under West Virginia Code Chapter 30 as a professional service."
    )
    await evaluator.verify(
        claim=ch30_claim,
        node=ch30_leaf,
        sources=_ensure_list(A.sources),
        additional_instruction="Verify whether the page explicitly confirms the Chapter 30 regulation status for the profession. Prefer official WV government or board pages.",
    )

    # Leaf: Licensing board name
    board_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_licensing_board",
        desc="Identify the specific WV state licensing board responsible for regulating this profession",
        parent=biz_node,
        critical=True,
    )
    board_claim = f"The licensing board responsible for regulating this profession is '{A.licensing_board_name}'."
    await evaluator.verify(
        claim=board_claim,
        node=board_leaf,
        sources=_ensure_list(A.sources),
        additional_instruction="Confirm the regulating licensing board name on official WV government or official board pages.",
    )

    # Leaf: Licensing board official website URL or contact
    board_url_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_board_official_url_or_contact",
        desc="Provide the licensing board’s official website URL or official contact information",
        parent=biz_node,
        critical=True,
    )
    # Choose verification source: direct contact URL if provided, otherwise Section A sources
    contact_source = [A.licensing_board_contact_or_url] if _is_url(A.licensing_board_contact_or_url) else _ensure_list(A.sources)
    board_url_claim = f"The provided website or contact information is the official site/contact for '{A.licensing_board_name}'."
    await evaluator.verify(
        claim=board_url_claim,
        node=board_url_leaf,
        sources=contact_source,
        additional_instruction="Verify this is the official licensing board website or official contact page.",
    )

    # ------------------------------- B. VOE ------------------------------------- #
    evaluator.add_custom_node(
        result=bool(B.voe_required),
        id=f"b{business_idx}_voe_required_value_exists",
        desc="Section B: VOE required value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(B.voe_signature_requirement),
        id=f"b{business_idx}_voe_signature_value_exists",
        desc="Section B: VOE signature requirement value exists",
        parent=biz_node,
        critical=True,
    )
    # Applicability gate: signature requirement only applies when VOE is required
    evaluator.add_custom_node(
        result=_text_has_affirmative(B.voe_required),
        id=f"b{business_idx}_voe_applicable",
        desc="Section B: VOE is required (signature requirement applicable)",
        parent=biz_node,
        critical=True,
    )

    # Leaf: VOE required with SOS registration
    voe_req_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_voe_required_with_sos_registration",
        desc="State whether a Verification of Eligibility (Form VOE) from the licensing board is required to be submitted with Secretary of State registration",
        parent=biz_node,
        critical=True,
    )
    voe_req_claim = (
        "A Verification of Eligibility (Form VOE) from the licensing board is required to be submitted with the Secretary of State registration."
        if _text_has_affirmative(B.voe_required)
        else "A Verification of Eligibility (Form VOE) from the licensing board is not required to be submitted with the Secretary of State registration."
    )
    await evaluator.verify(
        claim=voe_req_claim,
        node=voe_req_leaf,
        sources=_ensure_list(B.sources),
        additional_instruction="Verify the VOE requirement on WV SOS or official documentation.",
    )

    # Leaf: VOE signature requirement
    voe_sig_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_voe_signature_requirement",
        desc="State that the VOE must be signed by an authorized representative of the licensing board (if VOE is required)",
        parent=biz_node,
        critical=True,
    )
    voe_sig_claim = "Form VOE must be signed by an authorized representative of the licensing board."
    await evaluator.verify(
        claim=voe_sig_claim,
        node=voe_sig_leaf,
        sources=_ensure_list(B.sources),
        additional_instruction="Confirm signature requirement language on WV SOS or official documentation. If VOE is not required, this may not apply.",
    )

    # ------------------------------- C. BRC ------------------------------------- #
    evaluator.add_custom_node(
        result=bool(C.brc_required),
        id=f"b{business_idx}_brc_required_value_exists",
        desc="Section C: BRC required value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(C.brc_separate_per_name_and_location),
        id=f"b{business_idx}_brc_separate_value_exists",
        desc="Section C: Separate certificates per name/location value exists",
        parent=biz_node,
        critical=True,
    )

    # Leaf: Business Registration Certificate required
    brc_req_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_business_registration_certificate_required",
        desc="Confirm whether a WV State Tax Department Business Registration Certificate is required before engaging in business activity",
        parent=biz_node,
        critical=True,
    )
    brc_req_claim = (
        "A Business Registration Certificate from the West Virginia State Tax Department is required before engaging in business activity."
        if _text_has_affirmative(C.brc_required)
        else "A Business Registration Certificate from the West Virginia State Tax Department is not required before engaging in business activity."
    )
    await evaluator.verify(
        claim=brc_req_claim,
        node=brc_req_leaf,
        sources=_ensure_list(C.sources),
        additional_instruction="Verify requirement on WV Tax/Revenue official website or documentation.",
    )

    # Leaf: Separate certificates per name and location
    brc_sep_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_brc_separate_per_name_and_location",
        desc="State that separate Business Registration Certificates are required for each business name and location",
        parent=biz_node,
        critical=True,
    )
    brc_sep_claim = "Separate Business Registration Certificates are required for each business name and location."
    await evaluator.verify(
        claim=brc_sep_claim,
        node=brc_sep_leaf,
        sources=_ensure_list(C.sources),
        additional_instruction="Confirm the rule about separate certificates on WV Tax/Revenue official documentation.",
    )

    # ------------------------------- D. Annual Report --------------------------- #
    evaluator.add_custom_node(
        result=bool(D.annual_report_required),
        id=f"b{business_idx}_annual_report_required_value_exists",
        desc="Section D: Annual report required value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(D.annual_report_fee),
        id=f"b{business_idx}_annual_report_fee_value_exists",
        desc="Section D: Annual report fee value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(D.annual_report_deadline_period),
        id=f"b{business_idx}_annual_report_deadline_value_exists",
        desc="Section D: Annual report deadline period value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(D.annual_report_noncompliance_consequences),
        id=f"b{business_idx}_annual_report_consequences_value_exists",
        desc="Section D: Annual report noncompliance consequences value exists",
        parent=biz_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(D.online_filing_portal),
        id=f"b{business_idx}_online_portal_value_exists",
        desc="Section D: Online filing portal statement exists",
        parent=biz_node,
        critical=True,
    )

    # Leaf: Annual report required
    ar_req_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_annual_report_required",
        desc="Confirm whether an annual report must be filed with the WV Secretary of State",
        parent=biz_node,
        critical=True,
    )
    ar_req_claim = (
        "An annual report must be filed with the West Virginia Secretary of State."
        if _text_has_affirmative(D.annual_report_required)
        else "An annual report does not need to be filed with the West Virginia Secretary of State."
    )
    await evaluator.verify(
        claim=ar_req_claim,
        node=ar_req_leaf,
        sources=_ensure_list(D.sources),
        additional_instruction="Verify the requirement on WV SOS or official documentation.",
    )

    # Leaf: Annual report fee
    ar_fee_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_annual_report_fee",
        desc="Provide the annual report filing fee amount",
        parent=biz_node,
        critical=True,
    )
    ar_fee_claim = f"The annual report filing fee amount is '{D.annual_report_fee}'."
    await evaluator.verify(
        claim=ar_fee_claim,
        node=ar_fee_leaf,
        sources=_ensure_list(D.sources),
        additional_instruction="Confirm the exact fee amount on WV SOS or One Stop Portal pages.",
    )

    # Leaf: Annual report deadline period
    ar_deadline_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_annual_report_deadline_period",
        desc="Provide the annual report filing deadline period (range of dates each year)",
        parent=biz_node,
        critical=True,
    )
    ar_deadline_claim = f"The annual report filing deadline period each year is '{D.annual_report_deadline_period}'."
    await evaluator.verify(
        claim=ar_deadline_claim,
        node=ar_deadline_leaf,
        sources=_ensure_list(D.sources),
        additional_instruction="Confirm the deadline window on WV SOS official pages.",
    )

    # Leaf: Annual report noncompliance consequences
    ar_conseq_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_annual_report_noncompliance_consequences",
        desc="Note consequences of failure to file the annual report by the deadline",
        parent=biz_node,
        critical=True,
    )
    ar_conseq_claim = f"Failure to file the annual report by the deadline results in: '{D.annual_report_noncompliance_consequences}'."
    await evaluator.verify(
        claim=ar_conseq_claim,
        node=ar_conseq_leaf,
        sources=_ensure_list(D.sources),
        additional_instruction="Verify consequences (e.g., penalties, administrative dissolution) on WV SOS official documentation.",
    )

    # Leaf: Online annual report filing portal statement
    ar_portal_leaf = evaluator.add_leaf(
        id=f"b{business_idx}_annual_report_online_filing_portal",
        desc="State that online annual report filing is available via the West Virginia One Stop Business Portal (business4.wv.gov)",
        parent=biz_node,
        critical=True,
    )
    ar_portal_claim = "Online annual report filing is available via the West Virginia One Stop Business Portal (business4.wv.gov)."
    await evaluator.verify(
        claim=ar_portal_claim,
        node=ar_portal_leaf,
        sources=_ensure_list(D.sources),
        additional_instruction="Confirm the availability of online filing on business4.wv.gov or WV SOS official pages.",
    )

    # ------------------------------- Official sources/citations --------------- #
    # Single custom leaf to ensure sources for all sections are provided
    sources_ok = (
        len(_ensure_list(A.sources)) > 0 and
        len(_ensure_list(B.sources)) > 0 and
        len(_ensure_list(C.sources)) > 0 and
        len(_ensure_list(D.sources)) > 0
    )
    evaluator.add_custom_node(
        result=sources_ok,
        id=f"b{business_idx}_official_sources_and_citations",
        desc="Provide official WV government/official documentation source URL(s) supporting each major subsection (A: Chapter 30/board; B: VOE; C: Business Registration Certificate; D: Annual report details)",
        parent=biz_node,
        critical=True,
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
    Evaluate an answer for WV professional services regulatory compliance requirements.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent businesses evaluated in parallel
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

    # IMPORTANT: Set root as non-critical to allow partial credit across businesses
    root.critical = False

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=WVComplianceExtraction,
        extraction_name="wv_prof_services_compliance",
    )

    # Add a quick summary info of presence
    evaluator.add_custom_info(
        info={
            "engineering_present": extracted.engineering is not None,
            "architecture_present": extracted.architecture is not None,
            "accounting_present": extracted.accounting is not None,
            "social_work_present": extracted.social_work is not None,
        },
        info_type="extraction_presence",
        info_name="profession_presence"
    )

    # Build and run verifications for four businesses
    await verify_business(
        evaluator=evaluator,
        parent_root=root,
        business_idx=1,
        business_desc="Engineering services (Professional Engineers) - compliance requirements",
        data=extracted.engineering,
    )
    await verify_business(
        evaluator=evaluator,
        parent_root=root,
        business_idx=2,
        business_desc="Architectural services (Architects) - compliance requirements",
        data=extracted.architecture,
    )
    await verify_business(
        evaluator=evaluator,
        parent_root=root,
        business_idx=3,
        business_desc="Accounting services (Certified Public Accountants) - compliance requirements",
        data=extracted.accounting,
    )
    await verify_business(
        evaluator=evaluator,
        parent_root=root,
        business_idx=4,
        business_desc="Clinical social work services (Licensed Social Workers) - compliance requirements",
        data=extracted.social_work,
    )

    # Return structured evaluation summary
    return evaluator.get_summary()