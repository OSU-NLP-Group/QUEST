import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from urllib.parse import urlparse

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "multi_state_charity_registration_2026"
TASK_DESCRIPTION = """A nonprofit organization based in Texas is planning to expand its fundraising activities to solicit donations from residents in California, Florida, and New York starting in 2026. The organization expects to receive approximately $75,000 in total charitable contributions during its first fiscal year of multi-state operations, with fundraising activities managed entirely by paid staff members.

For each of the three states (California, Florida, and New York), provide a comprehensive analysis that includes:

1. Registration Obligation: Identify whether registration is required, what triggers the registration requirement, and any applicable deadlines.
2. Fee Structure: Specify the initial registration fee amount and, where applicable, the basis for fee calculation. If a small charity exemption exists, identify the eligibility criteria and whether this organization qualifies.
3. Required Documentation: List the key documents required for initial registration.
4. Annual Compliance: Identify annual renewal or reporting requirements, including filing deadlines and any state-specific forms.
5. Enforcement: Note any penalties for late filing or non-compliance.
6. Supporting References: For each state's requirements, provide valid URL references from official government sources that support your analysis.

Additionally, provide context about multi-state charitable registration by identifying: (a) the total number of U.S. states (plus D.C.) that currently require charitable solicitation registration, and (b) whether any standardized multi-state registration tools exist to simplify compliance.
"""


# --------------------------- Data Models ---------------------------

class CaliforniaDetails(BaseModel):
    registration_trigger: Optional[str] = None
    deadline: Optional[str] = None
    initial_fee_amount: Optional[str] = None
    payment_methods: List[str] = Field(default_factory=list)
    required_documents: List[str] = Field(default_factory=list)
    annual_compliance: Optional[str] = None
    enforcement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class FloridaDetails(BaseModel):
    registration_trigger: Optional[str] = None
    fee_range: Optional[str] = None
    fee_basis: Optional[str] = None
    small_charity_exemption_criteria: Optional[str] = None
    small_charity_exemption_form: Optional[str] = None
    small_charity_qualification_statement: Optional[str] = None
    required_documents: List[str] = Field(default_factory=list)
    annual_renewal_statement: Optional[str] = None
    late_fee_statement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class NewYorkDetails(BaseModel):
    registration_requirement: Optional[str] = None
    fee_structure_statement: Optional[str] = None
    required_documents: List[str] = Field(default_factory=list)
    annual_char500_statement: Optional[str] = None
    annual_deadline_statement: Optional[str] = None
    signatories_count_statement: Optional[str] = None
    signatory_roles_statement: Optional[str] = None
    enforcement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class MultiStateContext(BaseModel):
    total_states_plus_dc_requiring_registration: Optional[str] = None
    unified_registration_statement: Optional[str] = None


class MultiStateExtraction(BaseModel):
    california: Optional[CaliforniaDetails] = None
    florida: Optional[FloridaDetails] = None
    new_york: Optional[NewYorkDetails] = None
    context: Optional[MultiStateContext] = None


# --------------------------- Extraction Prompt ---------------------------

def prompt_extract_multi_state() -> str:
    return """
Extract structured information for California, Florida, New York, and multi-state context from the answer. Only extract what is explicitly stated in the answer. Do not invent or infer.

For each state (California, Florida, New York), extract:
- registration_trigger: The condition that triggers initial registration (e.g., "first receipt of charitable assets" or "soliciting in or from the state").
- deadline: Any stated initial registration deadline (e.g., "within 30 days of trigger").
- initial_fee_amount (CA only, if stated): The initial registration fee amount (e.g., "$50").
- payment_methods (CA): List of payment methods mentioned for paying the fee (e.g., ["credit card", "ACH"]).
- fee_range (FL): The stated fee range (e.g., "$10-$400").
- fee_basis (FL): Basis for fee calculation (e.g., "contributions in immediately preceding fiscal year excluding government grants").
- small_charity_exemption_criteria (FL): The criteria for the small charity exemption (e.g., "<$50,000 and fundraising by unpaid personnel").
- small_charity_exemption_form (FL): The form used to claim the exemption (e.g., "FDACS-10110").
- small_charity_qualification_statement (FL): Whether the answer says the described organization qualifies or does not qualify for the FL small charity exemption (return a short phrase like "qualifies", "does not qualify", or "unspecified").
- required_documents: Key documents listed for initial registration (return list of document names/phrases).
- annual_compliance (CA): Statement of annual renewal/reporting obligations.
- annual_renewal_statement (FL): Statement that annual renewal is required (if present).
- annual_char500_statement (NY): Statement that annual CHAR500 filings/financial reports are required (if present).
- annual_deadline_statement (NY): Stated deadline for annual filings (e.g., "4.5 months after fiscal year end").
- signatories_count_statement (NY): Statement on number of electronic signatories required (e.g., "two electronic signatories").
- signatory_roles_statement (NY): Statement of required signatory roles (e.g., "president/authorized officer and CFO/treasurer/person with fiscal responsibility").
- enforcement: Penalties/late fees/non-compliance consequences (state-specific).
- references: Official government URLs mentioned for the state. Only include URLs explicitly present in the answer.

Also extract multi-state context:
- total_states_plus_dc_requiring_registration: The stated total number of states plus District of Columbia that require registration (e.g., "41 + DC").
- unified_registration_statement: Statement about existence of standardized multi-state tools (e.g., "URS exists; state-specific requirements vary").

Organize the JSON as:
{
  "california": {...},
  "florida": {...},
  "new_york": {...},
  "context": {...}
}
If any field is not present in the answer, set it to null (strings) or [] (lists).
"""


# --------------------------- Helper Functions ---------------------------

def _is_official_url(url: str, allowed_domains: List[str]) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        for dom in allowed_domains:
            if dom.startswith(".") and netloc.endswith(dom):
                return True
            if dom in netloc:
                return True
        return False
    except Exception:
        return False


def _has_official_references(urls: List[str], allowed_domains: List[str]) -> bool:
    if not urls:
        return False
    return any(_is_official_url(u, allowed_domains) for u in urls)


def _require_official_sources_instruction(state_name: str) -> str:
    return (
        f"This claim must be supported by official {state_name} government sources. "
        f"If the answer fails to provide valid official URLs or if the provided webpages do not support the claim, mark the claim as incorrect."
    )


# --------------------------- Verification Builders ---------------------------

async def verify_california(evaluator: Evaluator, parent_node, ca: Optional[CaliforniaDetails]) -> None:
    ca_node = evaluator.add_parallel(
        id="California",
        desc="California requirements (registration, fees, documents, compliance, enforcement) and supporting official references.",
        parent=parent_node,
        critical=True
    )

    ca_refs = (ca.references if ca else []) if ca is not None else []
    # Supporting references (official CA)
    ca_support_node = evaluator.add_custom_node(
        result=_has_official_references(ca_refs, allowed_domains=["oag.ca.gov", ".ca.gov"]),
        id="CA_Supporting_References",
        desc="Provides official California government URL reference(s) supporting the CA analysis (registration/fees/docs at minimum).",
        parent=ca_node,
        critical=True
    )

    # Registration Obligation
    reg_node = evaluator.add_parallel(
        id="CA_Registration_Obligation",
        desc="Whether CA registration is required for the described organization; what triggers it; and any deadlines.",
        parent=ca_node,
        critical=True
    )

    # Trigger
    trigger_leaf = evaluator.add_leaf(
        id="CA_Trigger_Receiving_Charitable_Assets",
        desc="States that CA registration is triggered by first receiving charitable assets (donations/property/grants/other contributions).",
        parent=reg_node,
        critical=True
    )
    trigger_claim = (
        "In California, initial registration with the Attorney General's Registry of Charitable Trusts is triggered by the first receipt of charitable assets, including donations, property, grants, or other contributions."
    )
    await evaluator.verify(
        claim=trigger_claim,
        node=trigger_leaf,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    # Deadline
    deadline_leaf = evaluator.add_leaf(
        id="CA_Deadline_30_Days",
        desc="States that CA registration must occur within 30 days after the trigger.",
        parent=reg_node,
        critical=True
    )
    deadline_claim = "In California, the initial registration must be completed within 30 days after the organization first receives charitable assets."
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    # Fee Structure
    fee_node = evaluator.add_parallel(
        id="CA_Fee_Structure",
        desc="CA initial registration fee and any stated calculation basis; include payment method info if provided.",
        parent=ca_node,
        critical=True
    )

    # Initial fee $50
    fee_leaf = evaluator.add_leaf(
        id="CA_Initial_Fee_50",
        desc="Identifies CA initial registration fee amount as $50.",
        parent=fee_node,
        critical=True
    )
    fee_claim = "California's initial charitable registration fee is $50."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    # Payment methods
    pm_leaf = evaluator.add_leaf(
        id="CA_Payment_Methods_CreditCard_ACH",
        desc="Identifies that CA fee is payable online via credit card or ACH.",
        parent=fee_node,
        critical=True
    )
    pm_claim = "California allows the registration fee to be paid online by credit card or ACH (e-check)."
    await evaluator.verify(
        claim=pm_claim,
        node=pm_leaf,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    # Required Documentation
    docs_node = evaluator.add_parallel(
        id="CA_Required_Documentation",
        desc="Key documents required for CA initial registration.",
        parent=ca_node,
        critical=True
    )

    doc1 = evaluator.add_leaf(
        id="CA_Founding_Documents",
        desc="Includes founding documents (e.g., certified Articles of Incorporation if incorporated).",
        parent=docs_node,
        critical=True
    )
    doc1_claim = "California's initial registration requires founding documents such as certified Articles of Incorporation (if incorporated)."
    await evaluator.verify(
        claim=doc1_claim,
        node=doc1,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    doc2 = evaluator.add_leaf(
        id="CA_Current_Bylaws",
        desc="Includes current bylaws.",
        parent=docs_node,
        critical=True
    )
    doc2_claim = "California's initial registration requires a copy of the organization's current bylaws."
    await evaluator.verify(
        claim=doc2_claim,
        node=doc2,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    doc3 = evaluator.add_leaf(
        id="CA_IRS_Determination_Letter",
        desc="Includes IRS determination letter.",
        parent=docs_node,
        critical=True
    )
    doc3_claim = "California's initial registration requires the IRS determination letter (if applicable)."
    await evaluator.verify(
        claim=doc3_claim,
        node=doc3,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    doc4 = evaluator.add_leaf(
        id="CA_IRS_Exemption_Application",
        desc="Includes IRS exemption application (Form 1023/1023-EZ/1024).",
        parent=docs_node,
        critical=True
    )
    doc4_claim = "California's initial registration requires the IRS exemption application (Form 1023, 1023-EZ, or 1024), as applicable."
    await evaluator.verify(
        claim=doc4_claim,
        node=doc4,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    # Annual Compliance
    annual_leaf = evaluator.add_leaf(
        id="CA_Annual_Compliance",
        desc="Describes CA annual renewal/reporting obligations, deadlines, and any state-specific forms (as applicable).",
        parent=ca_node,
        critical=True
    )
    annual_claim = "California requires annual renewal/reporting for charities, typically via the RRF-1 (or equivalent) with financial statements."
    await evaluator.verify(
        claim=annual_claim,
        node=annual_leaf,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )

    # Enforcement
    enf_leaf = evaluator.add_leaf(
        id="CA_Enforcement",
        desc="Notes CA penalties or consequences for late filing or non-compliance.",
        parent=ca_node,
        critical=True
    )
    enf_claim = "California imposes penalties or consequences for late filing or non-compliance, including potential late fees and delinquency/suspension of registration."
    await evaluator.verify(
        claim=enf_claim,
        node=enf_leaf,
        sources=ca_refs,
        additional_instruction=_require_official_sources_instruction("California"),
        extra_prerequisites=[ca_support_node]
    )


async def verify_florida(evaluator: Evaluator, parent_node, fl: Optional[FloridaDetails]) -> None:
    fl_node = evaluator.add_parallel(
        id="Florida",
        desc="Florida requirements (registration, fees including exemption, documents, compliance, enforcement) and supporting official references.",
        parent=parent_node,
        critical=True
    )

    fl_refs = (fl.references if fl else []) if fl is not None else []
    fl_support_node = evaluator.add_custom_node(
        result=_has_official_references(fl_refs, allowed_domains=["fdacs.gov", "flrules.org", "leg.state.fl.us", "myflorida.com"]),
        id="FL_Supporting_References",
        desc="Provides official Florida government URL reference(s) supporting the FL analysis (registration/fees/exemption/enforcement at minimum).",
        parent=fl_node,
        critical=True
    )

    # Registration Obligation
    fl_reg = evaluator.add_parallel(
        id="FL_Registration_Obligation",
        desc="Whether FL registration is required for the described organization and what triggers it; include applicable initial timing/deadlines if stated.",
        parent=fl_node,
        critical=True
    )
    fl_trigger_leaf = evaluator.add_leaf(
        id="FL_Trigger_Soliciting_In_Or_From_FL",
        desc="States that registration is required for those soliciting donations in or from Florida.",
        parent=fl_reg,
        critical=True
    )
    fl_trigger_claim = "Florida requires charitable organizations that solicit donations in or from Florida to register."
    await evaluator.verify(
        claim=fl_trigger_claim,
        node=fl_trigger_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    # Fee Structure and Exemption
    fl_fee = evaluator.add_parallel(
        id="FL_Fee_Structure_And_Exemption",
        desc="FL fee structure including the small charity exemption criteria/benefits and whether the described organization qualifies.",
        parent=fl_node,
        critical=True
    )

    fee_range_leaf = evaluator.add_leaf(
        id="FL_Standard_Fee_Range_10_400",
        desc="Identifies that FL registration fees range from $10 to $400.",
        parent=fl_fee,
        critical=True
    )
    fee_range_claim = "Florida charitable registration fees range from $10 to $400."
    await evaluator.verify(
        claim=fee_range_claim,
        node=fee_range_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    fee_basis_leaf = evaluator.add_leaf(
        id="FL_Fee_Basis_Preceding_FY_Contributions",
        desc="Identifies that FL fees are based on contributions in the immediately preceding fiscal year (excluding government grants).",
        parent=fl_fee,
        critical=True
    )
    fee_basis_claim = "Florida calculates registration fees based on contributions in the immediately preceding fiscal year, excluding government grants."
    await evaluator.verify(
        claim=fee_basis_claim,
        node=fee_basis_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    exem_criteria_leaf = evaluator.add_leaf(
        id="FL_Small_Charity_Exemption_Criteria",
        desc="Identifies the FL small charity exemption eligibility criteria: < $50,000 contributions in the immediately preceding fiscal year AND fundraising carried on by unpaid personnel.",
        parent=fl_fee,
        critical=True
    )
    exem_criteria_claim = "Florida's small charity exemption applies only if the organization received less than $50,000 in contributions in the immediately preceding fiscal year and fundraising is carried on solely by unpaid personnel."
    await evaluator.verify(
        claim=exem_criteria_claim,
        node=exem_criteria_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    exem_form_leaf = evaluator.add_leaf(
        id="FL_Small_Charity_Exemption_No_Fee_And_Form",
        desc="States that qualifying small charities can file Form FDACS-10110 and pay no registration fee.",
        parent=fl_fee,
        critical=True
    )
    exem_form_claim = "Qualifying small charities in Florida can file Form FDACS-10110 and pay no registration fee."
    await evaluator.verify(
        claim=exem_form_claim,
        node=exem_form_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    # Qualification determination for this org (logic check using scenario)
    qual_leaf = evaluator.add_leaf(
        id="FL_Exemption_Qualification_For_This_Org",
        desc="Correctly determines whether the described organization qualifies for the FL small charity exemption given the scenario (paid staff; expected contributions).",
        parent=fl_fee,
        critical=True
    )
    qual_claim = (
        "Given the scenario (expected ~$75,000 in contributions during the first fiscal year and fundraising managed entirely by paid staff), "
        "the organization does NOT qualify for Florida's small charity exemption."
    )
    await evaluator.verify(
        claim=qual_claim,
        node=qual_leaf,
        sources=None,
        additional_instruction=(
            "Use the scenario facts to judge the determination. The correct outcome is 'does not qualify' because expected contributions exceed $50,000 "
            "and fundraising is conducted by paid personnel."
        )
    )

    # Required Documentation
    fl_docs_leaf = evaluator.add_leaf(
        id="FL_Required_Documentation",
        desc="Lists key documents required for initial Florida registration (without inventing unsupported specifics).",
        parent=fl_node,
        critical=True
    )
    fl_docs_claim = "Florida's official registration process lists key documents required for initial registration (application and supporting organizational/financial documentation)."
    await evaluator.verify(
        claim=fl_docs_claim,
        node=fl_docs_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    # Annual Compliance
    fl_annual = evaluator.add_parallel(
        id="FL_Annual_Compliance",
        desc="Identifies Florida annual renewal/reporting requirements and any deadlines/forms (at minimum: states that annual renewal is required).",
        parent=fl_node,
        critical=True
    )
    fl_annual_leaf = evaluator.add_leaf(
        id="FL_Annual_Renewal_Required",
        desc="States that Florida requires annual renewal.",
        parent=fl_annual,
        critical=True
    )
    fl_annual_claim = "Florida requires annual renewal for charitable solicitation registration."
    await evaluator.verify(
        claim=fl_annual_claim,
        node=fl_annual_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )

    # Enforcement
    fl_enf = evaluator.add_parallel(
        id="FL_Enforcement",
        desc="Florida penalties for late filing/non-compliance.",
        parent=fl_node,
        critical=True
    )
    fl_enf_leaf = evaluator.add_leaf(
        id="FL_Late_Filing_Fee_25_Per_Month",
        desc="States the FL late filing fee is $25 for each month or part of a month after the renewal due date.",
        parent=fl_enf,
        critical=True
    )
    fl_enf_claim = "Florida assesses a $25 late filing fee for each month or part of a month after the renewal due date."
    await evaluator.verify(
        claim=fl_enf_claim,
        node=fl_enf_leaf,
        sources=fl_refs,
        additional_instruction=_require_official_sources_instruction("Florida"),
        extra_prerequisites=[fl_support_node]
    )


async def verify_new_york(evaluator: Evaluator, parent_node, ny: Optional[NewYorkDetails]) -> None:
    ny_node = evaluator.add_parallel(
        id="New_York",
        desc="New York requirements (registration, fees, documents, annual filing, signatories, enforcement) and supporting official references.",
        parent=parent_node,
        critical=True
    )

    ny_refs = (ny.references if ny else []) if ny is not None else []
    ny_support_node = evaluator.add_custom_node(
        result=_has_official_references(ny_refs, allowed_domains=["ag.ny.gov", "charitiesnys.com"]),
        id="NY_Supporting_References",
        desc="Provides official New York government URL reference(s) supporting the NY analysis (registration/CHAR500 deadline/signatories at minimum).",
        parent=ny_node,
        critical=True
    )

    # Registration Obligation
    ny_reg = evaluator.add_parallel(
        id="NY_Registration_Obligation",
        desc="Whether NY registration is required for the described organization and what triggers it; include any applicable deadlines if stated.",
        parent=ny_node,
        critical=True
    )
    ny_reg_leaf = evaluator.add_leaf(
        id="NY_Registration_Required_For_Charities_Operating_In_NY",
        desc="States that NY requires charitable organizations operating in NY to register.",
        parent=ny_reg,
        critical=True
    )
    ny_reg_claim = "New York requires charitable organizations operating in New York to register with the Charities Bureau."
    await evaluator.verify(
        claim=ny_reg_claim,
        node=ny_reg_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    # Fee Structure
    ny_fee_leaf = evaluator.add_leaf(
        id="NY_Fee_Structure",
        desc="Identifies NY initial registration fee amount and basis (or explicitly states if no fee / not specified).",
        parent=ny_node,
        critical=True
    )
    ny_fee_claim = (
        ny.fee_structure_statement
        if (ny and ny.fee_structure_statement)
        else "New York's official registration materials either specify an initial registration fee or indicate fees are assessed via annual filings; the answer should correctly reflect the official position."
    )
    await evaluator.verify(
        claim=ny_fee_claim,
        node=ny_fee_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    # Required Documentation
    ny_docs_leaf = evaluator.add_leaf(
        id="NY_Required_Documentation",
        desc="Lists key documents required for NY initial registration (without hard-coding unsupported document lists).",
        parent=ny_node,
        critical=True
    )
    ny_docs_list_text = ", ".join(ny.required_documents) if (ny and ny.required_documents) else "key organizational/financial documents"
    ny_docs_claim = f"New York's initial registration requires {ny_docs_list_text}."
    await evaluator.verify(
        claim=ny_docs_claim,
        node=ny_docs_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    # Annual Compliance (CHAR500)
    ny_annual = evaluator.add_parallel(
        id="NY_Annual_Compliance_CHAR500",
        desc="NY annual filing requirements and deadlines.",
        parent=ny_node,
        critical=True
    )
    ny_char500_leaf = evaluator.add_leaf(
        id="NY_Annual_Form_CHAR500",
        desc="Identifies that NY requires annual CHAR500 filings/financial reports.",
        parent=ny_annual,
        critical=True
    )
    ny_char500_claim = "New York requires annual filings using Form CHAR500 with accompanying financial reports."
    await evaluator.verify(
        claim=ny_char500_claim,
        node=ny_char500_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    ny_deadline_leaf = evaluator.add_leaf(
        id="NY_Deadline_4_5_Months_After_FYE",
        desc="States that CHAR500 is due 4.5 months after the fiscal year end.",
        parent=ny_annual,
        critical=True
    )
    ny_deadline_claim = "In New York, the annual CHAR500 filing is due 4.5 months after the organization's fiscal year end."
    await evaluator.verify(
        claim=ny_deadline_claim,
        node=ny_deadline_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    # Signatory Requirements
    ny_sign = evaluator.add_parallel(
        id="NY_Signatory_Requirements",
        desc="NY electronic signatory requirements for registration filings.",
        parent=ny_node,
        critical=True
    )
    ny_two_sign_leaf = evaluator.add_leaf(
        id="NY_Two_Electronic_Signatories",
        desc="States that NY registration requires two electronic signatories.",
        parent=ny_sign,
        critical=True
    )
    ny_two_sign_claim = "New York's registration filings require two electronic signatories."
    await evaluator.verify(
        claim=ny_two_sign_claim,
        node=ny_two_sign_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    ny_roles_leaf = evaluator.add_leaf(
        id="NY_Signatory_Roles",
        desc="Specifies the required signatory roles: (1) president/authorized officer and (2) CFO/treasurer/person with fiscal responsibility.",
        parent=ny_sign,
        critical=True
    )
    ny_roles_claim = "The required New York signatories are (1) the president or another authorized officer and (2) the CFO/treasurer or person with fiscal responsibility."
    await evaluator.verify(
        claim=ny_roles_claim,
        node=ny_roles_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )

    # Enforcement
    ny_enf_leaf = evaluator.add_leaf(
        id="NY_Enforcement",
        desc="Notes NY penalties or consequences for late filing or non-compliance.",
        parent=ny_node,
        critical=True
    )
    ny_enf_claim = "New York imposes penalties or consequences for late filing or non-compliance (e.g., fines, delinquency status, or other enforcement actions)."
    await evaluator.verify(
        claim=ny_enf_claim,
        node=ny_enf_leaf,
        sources=ny_refs,
        additional_instruction=_require_official_sources_instruction("New York"),
        extra_prerequisites=[ny_support_node]
    )


async def verify_multi_state_context(evaluator: Evaluator, parent_node, ctx: Optional[MultiStateContext]) -> None:
    ctx_node = evaluator.add_parallel(
        id="Multi_State_Registration_Context",
        desc="General context requested: how many jurisdictions require registration and whether standardized multi-state tools exist.",
        parent=parent_node,
        critical=True
    )

    total_leaf = evaluator.add_leaf(
        id="Total_States_Plus_DC_Requiring_Registration",
        desc="Identifies the total as 41 states plus the District of Columbia requiring some form of charitable solicitation registration.",
        parent=ctx_node,
        critical=True
    )
    total_claim = "41 states plus the District of Columbia require some form of charitable solicitation registration."
    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=None,
        additional_instruction="Judge whether the answer correctly states the total jurisdictions requiring registration. Accept minor phrasing variants."
    )

    urs_leaf = evaluator.add_leaf(
        id="Unified_Registration_Statement_Exists",
        desc="Identifies the Unified Registration Statement (URS) as a standardized tool and notes that state-specific requirements still vary.",
        parent=ctx_node,
        critical=True
    )
    urs_claim = "The Unified Registration Statement (URS) exists as a standardized multi-state tool, but state-specific requirements still vary."
    await evaluator.verify(
        claim=urs_claim,
        node=urs_leaf,
        sources=None,
        additional_instruction="Judge whether the answer correctly identifies the existence of URS and mentions that requirements vary by state."
    )


# --------------------------- Main Evaluation Function ---------------------------

async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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

    # Create the top-level critical node representing the rubric root
    ms_root = evaluator.add_parallel(
        id="Multi_State_Charitable_Registration_Requirements",
        desc="Analysis of charitable solicitation registration/compliance for CA, FL, NY plus multi-state context, using the provided scenario and constraints.",
        parent=root,
        critical=True
    )

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_multi_state(),
        template_class=MultiStateExtraction,
        extraction_name="multi_state_details",
    )

    # Verify each state's requirements and the multi-state context
    await verify_california(evaluator, ms_root, extraction.california)
    await verify_florida(evaluator, ms_root, extraction.florida)
    await verify_new_york(evaluator, ms_root, extraction.new_york)
    await verify_multi_state_context(evaluator, ms_root, extraction.context)

    return evaluator.get_summary()