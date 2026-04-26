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
TASK_ID = "la_engineering_llc_regulatory"
TASK_DESCRIPTION = (
    "A civil engineering professional plans to establish a new professional engineering consulting firm in Louisiana "
    "that will provide engineering design and consulting services. The firm will operate as a limited liability company (LLC), "
    "employ 3-5 professional engineers and support staff, and intends to pursue federal government contracts including potential "
    "U.S. Department of Defense projects. The firm's name will be \"Bayou Engineering Solutions, LLC\" and will have a physical "
    "office location in Baton Rouge, Louisiana.\n\n"
    "Question: What are the complete legal and regulatory requirements for establishing this firm, including all necessary "
    "registrations, licenses, permits, fees, required documentation, designated timelines, insurance requirements, and compliance "
    "obligations at the federal, state, and local levels that must be satisfied before the firm can legally offer professional "
    "engineering services and bid on federal government contracts?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Requirement(BaseModel):
    mentioned: Optional[bool] = None
    details: Optional[str] = None
    timeline: Optional[str] = None
    fee: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SOSLLCFormation(BaseModel):
    lapels_name_waiver_before_sos: Optional[Requirement] = None
    sos_register_and_certificate: Optional[Requirement] = None


class LapelsFirmLicensure(BaseModel):
    apply_within_30_days: Optional[Requirement] = None
    fee: Optional[Requirement] = None
    include_sos_certificate_copy: Optional[Requirement] = None
    designate_la_licensed_pe_responsible_party: Optional[Requirement] = None
    no_practice_before_license: Optional[Requirement] = None


class LDRRegistration(BaseModel):
    register_with_latap: Optional[Requirement] = None


class IRSEIN(BaseModel):
    obtain_ein: Optional[Requirement] = None


class FederalSAMRegistration(BaseModel):
    register_in_sam: Optional[Requirement] = None
    uei_assigned: Optional[Requirement] = None


class NAICSIdentification(BaseModel):
    identify_appropriate_naics: Optional[Requirement] = None
    engineering_services_under_naics_54: Optional[Requirement] = None


class SBAStandards(BaseModel):
    meet_sba_size_standard: Optional[Requirement] = None
    receipts_threshold_7_5m: Optional[Requirement] = None


class FARCompliance(BaseModel):
    comply_with_far: Optional[Requirement] = None


class WorkersComp(BaseModel):
    workers_comp_required_1plus_employees: Optional[Requirement] = None


class LouisianaUnemployment(BaseModel):
    register_ui_tax_account_if_employing: Optional[Requirement] = None


class LocalCompliance(BaseModel):
    comply_with_zoning: Optional[Requirement] = None
    occupational_license_may_be_required: Optional[Requirement] = None


class DoDCMMC(BaseModel):
    meet_applicable_cmmc_level: Optional[Requirement] = None


class EnvironmentalLPDES(BaseModel):
    lpdes_if_discharging_pollutants: Optional[Requirement] = None


class FirmSetupExtraction(BaseModel):
    firm_name: Optional[str] = None
    office_city: Optional[str] = None
    office_state: Optional[str] = None
    employees_count_desc: Optional[str] = None
    sos_formation: Optional[SOSLLCFormation] = None
    lapels_firm: Optional[LapelsFirmLicensure] = None
    ldr: Optional[LDRRegistration] = None
    ein: Optional[IRSEIN] = None
    sam: Optional[FederalSAMRegistration] = None
    naics: Optional[NAICSIdentification] = None
    sba: Optional[SBAStandards] = None
    far: Optional[FARCompliance] = None
    workers_comp: Optional[WorkersComp] = None
    la_unemployment: Optional[LouisianaUnemployment] = None
    local: Optional[LocalCompliance] = None
    dod_cmmc: Optional[DoDCMMC] = None
    lpdes: Optional[EnvironmentalLPDES] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return (
        "Extract, from the provided answer text only, the presence, details, timelines, fees, and source URLs for each of the "
        "following regulatory/registration requirements relevant to establishing a Louisiana engineering LLC named \"Bayou Engineering Solutions, LLC\" "
        "in Baton Rouge, Louisiana (with 3–5 employees and pursuing federal/DoD contracts). For each item, return:\n"
        "- mentioned: true/false (whether the answer explicitly mentions this requirement)\n"
        "- details: brief text as stated in the answer\n"
        "- timeline: any deadline or timing requirement mentioned\n"
        "- fee: the fee amount(s) or structure if mentioned\n"
        "- sources: array of URL(s) explicitly cited in the answer for this item (extract actual URLs only; if none, return an empty array)\n\n"
        "Items to extract:\n"
        "1) sos_formation.lapels_name_waiver_before_sos: LAPELS waiver/name usage approval prior to SOS processing when 'engineer/engineering' appears in the firm name.\n"
        "2) sos_formation.sos_register_and_certificate: SOS registration and obtaining Certificate of Organization.\n"
        "3) lapels_firm.apply_within_30_days: Apply for LAPELS firm licensure within 30 days of SOS certificate issuance.\n"
        "4) lapels_firm.fee: LAPELS firm licensure fee (e.g., $165 single-service or $330 dual-service) if stated.\n"
        "5) lapels_firm.include_sos_certificate_copy: Include copy of SOS Certificate with LAPELS application.\n"
        "6) lapels_firm.designate_la_licensed_pe_responsible_party: Designate a LA-licensed PE as responsible party (full-time employee or LLC member/manager).\n"
        "7) lapels_firm.no_practice_before_license: Do not offer/perform engineering services until LAPELS issues firm license.\n"
        "8) ldr.register_with_latap: Register with Louisiana Department of Revenue via LaTAP.\n"
        "9) ein.obtain_ein: Obtain IRS Employer Identification Number (EIN).\n"
        "10) sam.register_in_sam: Register in SAM.gov to be eligible for federal contracts.\n"
        "11) sam.uei_assigned: Ensure SAM yields UEI (12-character identifier).\n"
        "12) naics.identify_appropriate_naics: Identify appropriate NAICS code(s) for services (include codes mentioned in details).\n"
        "13) naics.engineering_services_under_naics_54: Recognize engineering services fall under NAICS 54 (as stated in the answer).\n"
        "14) sba.meet_sba_size_standard: Meet SBA size standards for the applicable NAICS.\n"
        "15) sba.receipts_threshold_7_5m: Statement that most professional services qualify as small if average receipts < $7.5M (as claimed in the answer).\n"
        "16) far.comply_with_far: Comply with FAR requirements for federal procurement.\n"
        "17) workers_comp.workers_comp_required_1plus_employees: Maintain workers’ comp insurance if 1+ employees.\n"
        "18) la_unemployment.register_ui_tax_account_if_employing: Register for LA unemployment insurance tax account if employing workers.\n"
        "19) local.comply_with_zoning: Comply with Baton Rouge zoning for commercial operations.\n"
        "20) local.occupational_license_may_be_required: Occupational license may be required by East Baton Rouge Parish.\n"
        "21) dod_cmmc.meet_applicable_cmmc_level: For DoD contracts, meet applicable CMMC level.\n"
        "22) lpdes.lpdes_if_discharging_pollutants: Obtain LPDES permit if facility discharges pollutants to state waters.\n\n"
        "Also extract basic firm context:\n"
        "- firm_name\n"
        "- office_city\n"
        "- office_state\n"
        "- employees_count_desc (e.g., '3-5 employees')\n\n"
        "Return a single JSON object matching the FirmSetupExtraction schema."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _mentioned_and_has_sources(item: Optional[Requirement]) -> bool:
    return bool(item and item.mentioned and (item.sources is not None) and (len(item.sources) > 0))


def _sources(item: Optional[Requirement]) -> List[str]:
    return item.sources if (item and item.sources) else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_sos_llc_formation(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_sequential(
        id="Louisiana_SOS_LLC_Formation",
        desc="Louisiana Secretary of State (SOS) LLC formation requirements",
        parent=parent_node,
        critical=True
    )

    # 1. LAPELS name waiver before SOS
    waiver_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.sos_formation.lapels_name_waiver_before_sos if ex.sos_formation else None),
        id="LAPELS_Name_Waiver_Before_SOS_exists",
        desc="LAPELS name waiver/approval requirement is mentioned with sources",
        parent=node,
        critical=True
    )
    waiver_verify = evaluator.add_leaf(
        id="LAPELS_Name_Waiver_Before_SOS",
        desc="If the firm name contains 'engineer' or 'engineering', submit the LAPELS Waiver Request Form before SOS processes the name/filing",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Louisiana LAPELS requires name usage approval/waiver before SOS processes filings when a firm name contains 'engineer' or 'engineering'.",
        node=waiver_verify,
        sources=_sources(ex.sos_formation.lapels_name_waiver_before_sos if ex.sos_formation else None),
        additional_instruction="Verify on the cited page(s) that LAPELS requires prior name usage approval for company names containing 'engineer' or 'engineering'."
    )

    # 2. SOS register and obtain certificate
    sos_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.sos_formation.sos_register_and_certificate if ex.sos_formation else None),
        id="SOS_Register_And_Certificate_exists",
        desc="SOS registration and certificate requirement mentioned with sources",
        parent=node,
        critical=True
    )
    sos_verify = evaluator.add_leaf(
        id="SOS_Register_And_Certificate",
        desc="Register the LLC with the Louisiana Secretary of State and obtain the Certificate of Organization",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A Louisiana LLC must register with the Louisiana Secretary of State and obtain a Certificate of Organization (or equivalent) to be formed.",
        node=sos_verify,
        sources=_sources(ex.sos_formation.sos_register_and_certificate if ex.sos_formation else None),
        additional_instruction="Check the cited Secretary of State or official guidance page confirming registration and certificate issuance for LLCs."
    )


async def build_lapels_firm_licensure(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="LAPELS_Firm_Licensure",
        desc="LAPELS firm licensure requirements to legally offer professional engineering services",
        parent=parent_node,
        critical=True
    )

    # Apply within 30 days
    apply_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.lapels_firm.apply_within_30_days if ex.lapels_firm else None),
        id="LAPELS_Apply_Within_30_Days_exists",
        desc="Application within 30 days mentioned with sources",
        parent=node,
        critical=True
    )
    apply_verify = evaluator.add_leaf(
        id="LAPELS_Apply_Within_30_Days",
        desc="Submit the Application for Firm Licensure to LAPELS within 30 days of SOS certificate issuance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Engineering firms in Louisiana must submit the LAPELS firm licensure application within approximately 30 days of SOS certificate issuance.",
        node=apply_verify,
        sources=_sources(ex.lapels_firm.apply_within_30_days if ex.lapels_firm else None),
        additional_instruction="Verify the stated 30-day application timing requirement on LAPELS guidance or application materials."
    )

    # Fee
    fee_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.lapels_firm.fee if ex.lapels_firm else None),
        id="LAPELS_Fee_exists",
        desc="LAPELS fee mentioned with sources",
        parent=node,
        critical=True
    )
    fee_verify = evaluator.add_leaf(
        id="LAPELS_Fee",
        desc="Pay the LAPELS application fee: $165 (single-service engineering firm) or $330 (dual-service firm)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The LAPELS firm license application fee is around $165 for single-service engineering firms or $330 for dual-service firms.",
        node=fee_verify,
        sources=_sources(ex.lapels_firm.fee if ex.lapels_firm else None),
        additional_instruction="Confirm the fee amounts on the official LAPELS fee schedule or application instructions. Allow minor wording variations."
    )

    # Include SOS certificate copy
    cert_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.lapels_firm.include_sos_certificate_copy if ex.lapels_firm else None),
        id="LAPELS_Include_SOS_Certificate_Copy_exists",
        desc="Including SOS certificate copy mentioned with sources",
        parent=node,
        critical=True
    )
    cert_verify = evaluator.add_leaf(
        id="LAPELS_Include_SOS_Certificate_Copy",
        desc="Include a copy of the SOS Certificate of Organization with the LAPELS firm licensure application",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="LAPELS firm licensure applications require a copy of the SOS Certificate of Organization (or formation documentation).",
        node=cert_verify,
        sources=_sources(ex.lapels_firm.include_sos_certificate_copy if ex.lapels_firm else None),
        additional_instruction="Verify in the LAPELS application checklist or instructions that the SOS certificate copy is required."
    )

    # Designate LA licensed PE
    resp_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.lapels_firm.designate_la_licensed_pe_responsible_party if ex.lapels_firm else None),
        id="Designate_LA_Licensed_PE_Responsible_Party_exists",
        desc="Designating LA-licensed PE responsible party mentioned with sources",
        parent=node,
        critical=True
    )
    resp_verify = evaluator.add_leaf(
        id="Designate_LA_Licensed_PE_Responsible_Party",
        desc="Designate a Louisiana-licensed professional engineer as the responsible party who is a full-time employee or an LLC member/manager",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Louisiana requires firms to designate a Louisiana-licensed professional engineer as the responsible party, typically a full-time employee or an LLC member/manager.",
        node=resp_verify,
        sources=_sources(ex.lapels_firm.designate_la_licensed_pe_responsible_party if ex.lapels_firm else None),
        additional_instruction="Check LAPELS rules describing 'responsible charge' and permitted employment/ownership relationships."
    )

    # No practice before license
    nop_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.lapels_firm.no_practice_before_license if ex.lapels_firm else None),
        id="No_Practice_Before_LAPELS_License_exists",
        desc="No practice before firm license issuance mentioned with sources",
        parent=node,
        critical=True
    )
    nop_verify = evaluator.add_leaf(
        id="No_Practice_Before_LAPELS_License",
        desc="Do not offer professional engineering services until LAPELS issues the firm license and confirmation letter",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="An engineering firm may not offer or practice engineering services in Louisiana until LAPELS issues the firm license (with confirmation).",
        node=nop_verify,
        sources=_sources(ex.lapels_firm.no_practice_before_license if ex.lapels_firm else None),
        additional_instruction="Verify LAPELS guidance prohibiting practice until firm licensure is granted."
    )


async def build_ldr_registration(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Louisiana_Department_of_Revenue_Registration",
        desc="Louisiana Department of Revenue (LDR) registration requirement",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.ldr.register_with_latap if ex.ldr else None),
        id="Register_With_LDR_via_LaTAP_exists",
        desc="LDR registration via LaTAP mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="Register_With_LDR_via_LaTAP",
        desc="Register with LDR via the LaTAP system after SOS registration",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Businesses in Louisiana should register with the Louisiana Department of Revenue via the LaTAP system.",
        node=verify_node,
        sources=_sources(ex.ldr.register_with_latap if ex.ldr else None),
        additional_instruction="Confirm LaTAP is the standard online system for LDR registration per official LDR guidance."
    )


async def build_ein(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Federal_EIN",
        desc="IRS Employer Identification Number requirement",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.ein.obtain_ein if ex.ein else None),
        id="Obtain_EIN_exists",
        desc="EIN requirement mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="Obtain_EIN",
        desc="Obtain a Federal Employer Identification Number (EIN) from the IRS",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A business must obtain a federal Employer Identification Number (EIN) from the IRS for tax and federal registration purposes.",
        node=verify_node,
        sources=_sources(ex.ein.obtain_ein if ex.ein else None),
        additional_instruction="Verify on IRS pages that businesses use EINs for identification in tax and federal systems."
    )


async def build_sam(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Federal_SAM_Registration",
        desc="System for Award Management (SAM.gov) registration requirements for federal contracting eligibility",
        parent=parent_node,
        critical=True
    )

    # Register in SAM
    reg_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.sam.register_in_sam if ex.sam else None),
        id="Register_in_SAM_exists",
        desc="SAM registration mentioned with sources",
        parent=node,
        critical=True
    )
    reg_verify = evaluator.add_leaf(
        id="Register_in_SAM",
        desc="Register in SAM.gov to be eligible to bid on federal government contracts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Organizations must register in SAM.gov to be eligible to bid on federal government contracts.",
        node=reg_verify,
        sources=_sources(ex.sam.register_in_sam if ex.sam else None),
        additional_instruction="Confirm on SAM.gov or official procurement pages that SAM registration is required for bidding."
    )

    # UEI
    uei_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.sam.uei_assigned if ex.sam else None),
        id="UEI_Assigned_exists",
        desc="UEI assignment mentioned with sources",
        parent=node,
        critical=True
    )
    uei_verify = evaluator.add_leaf(
        id="UEI_Assigned",
        desc="Ensure SAM registration results in a 12-character alphanumeric Unique Entity Identifier (UEI)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="SAM registration provides a 12-character alphanumeric Unique Entity Identifier (UEI) for each entity.",
        node=uei_verify,
        sources=_sources(ex.sam.uei_assigned if ex.sam else None),
        additional_instruction="Verify the UEI concept and format on official SAM or GSA guidance."
    )


async def build_naics(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="NAICS_Identification",
        desc="NAICS identification requirements",
        parent=parent_node,
        critical=True
    )

    # Identify appropriate NAICS
    id_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.naics.identify_appropriate_naics if ex.naics else None),
        id="Identify_Appropriate_NAICS_exists",
        desc="Identify appropriate NAICS codes mentioned with sources",
        parent=node,
        critical=True
    )
    id_verify = evaluator.add_leaf(
        id="Identify_Appropriate_NAICS",
        desc="Identify appropriate NAICS code(s) for the firm’s services",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Businesses should identify applicable NAICS code(s) that reflect their services.",
        node=id_verify,
        sources=_sources(ex.naics.identify_appropriate_naics if ex.naics else None),
        additional_instruction="Verify general guidance on selecting NAICS codes for professional engineering services."
    )

    # Engineering services under NAICS 54
    sec_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.naics.engineering_services_under_naics_54 if ex.naics else None),
        id="Engineering_Services_Under_NAICS_54_exists",
        desc="NAICS 54 classification mentioned with sources",
        parent=node,
        critical=True
    )
    sec_verify = evaluator.add_leaf(
        id="Engineering_Services_Under_NAICS_54",
        desc="Recognize that engineering services fall under NAICS 54 (as stated in constraints)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Engineering services fall within NAICS Sector 54: Professional, Scientific, and Technical Services.",
        node=sec_verify,
        sources=_sources(ex.naics.engineering_services_under_naics_54 if ex.naics else None),
        additional_instruction="Verify NAICS sector classification for engineering services (e.g., 541330 within Sector 54)."
    )


async def build_sba(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="SBA_Size_Standards",
        desc="SBA small business size standards requirements",
        parent=parent_node,
        critical=True
    )

    meet_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.sba.meet_sba_size_standard if ex.sba else None),
        id="Meet_SBA_Size_Standard_exists",
        desc="Meeting SBA size standard mentioned with sources",
        parent=node,
        critical=True
    )
    meet_verify = evaluator.add_leaf(
        id="Meet_SBA_Size_Standard",
        desc="Meet SBA small business size standards for the applicable NAICS code(s)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Eligibility as a small business depends on meeting SBA size standards for the applicable NAICS code(s).",
        node=meet_verify,
        sources=_sources(ex.sba.meet_sba_size_standard if ex.sba else None),
        additional_instruction="Verify SBA guidance on size standards tied to NAICS codes."
    )

    thr_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.sba.receipts_threshold_7_5m if ex.sba else None),
        id="Receipts_Threshold_7_5M_exists",
        desc="Receipts threshold claim mentioned with sources",
        parent=node,
        critical=True
    )
    thr_verify = evaluator.add_leaf(
        id="Receipts_Threshold_7_5M",
        desc="Apply the stated constraint that most professional services qualify as small if average annual receipts are under $7.5 million",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Most professional services qualify as small if average annual receipts are under about $7.5 million, per the cited source(s).",
        node=thr_verify,
        sources=_sources(ex.sba.receipts_threshold_7_5m if ex.sba else None),
        additional_instruction="Verify the specific receipts threshold claim against the cited SBA or official pages; allow reasonable phrasing differences."
    )


async def build_far(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="FAR_Compliance",
        desc="Federal Acquisition Regulation (FAR) compliance requirement",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.far.comply_with_far if ex.far else None),
        id="Comply_With_FAR_exists",
        desc="FAR compliance mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="Comply_With_FAR",
        desc="Comply with FAR requirements applicable to federal procurement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Contractors must comply with applicable Federal Acquisition Regulation (FAR) requirements to participate in federal procurement.",
        node=verify_node,
        sources=_sources(ex.far.comply_with_far if ex.far else None),
        additional_instruction="Verify general FAR applicability and requirements per official FAR resources."
    )


async def build_workers_comp(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Louisiana_Workers_Comp",
        desc="Workers’ compensation insurance requirement (Louisiana)",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.workers_comp.workers_comp_required_1plus_employees if ex.workers_comp else None),
        id="Workers_Comp_Required_1Plus_Employees_exists",
        desc="Workers’ compensation requirement mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="Workers_Comp_Required_1Plus_Employees",
        desc="Maintain workers’ compensation insurance if the firm has 1+ employees (as stated in constraints)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Louisiana requires employers with one or more employees to maintain workers’ compensation insurance.",
        node=verify_node,
        sources=_sources(ex.workers_comp.workers_comp_required_1plus_employees if ex.workers_comp else None),
        additional_instruction="Verify Louisiana workers’ compensation coverage requirements on official state resources."
    )


async def build_la_unemployment(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Louisiana_Unemployment_Insurance",
        desc="Louisiana unemployment insurance tax account requirement",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.la_unemployment.register_ui_tax_account_if_employing if ex.la_unemployment else None),
        id="Register_UI_Tax_Account_If_Employing_exists",
        desc="UI tax account registration mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="Register_UI_Tax_Account_If_Employing",
        desc="Register for a Louisiana unemployment insurance tax account if employing workers",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Employers in Louisiana must register for an unemployment insurance tax account if they employ workers.",
        node=verify_node,
        sources=_sources(ex.la_unemployment.register_ui_tax_account_if_employing if ex.la_unemployment else None),
        additional_instruction="Verify employer UI tax account registration requirement via Louisiana Workforce Commission or related official pages."
    )


async def build_local_compliance(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Local_Baton_Rouge_Compliance",
        desc="Local Baton Rouge / East Baton Rouge Parish compliance requirements",
        parent=parent_node,
        critical=False  # Set non-critical to allow partial credit and comply with framework constraints
    )

    # Zoning compliance (critical under local node)
    zone_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.local.comply_with_zoning if ex.local else None),
        id="Comply_With_Zoning_exists",
        desc="Zoning compliance mentioned with sources",
        parent=node,
        critical=True
    )
    zone_verify = evaluator.add_leaf(
        id="Comply_With_Zoning",
        desc="Comply with local Baton Rouge zoning regulations for commercial operations",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Local zoning regulations in Baton Rouge must be complied with for commercial office operations.",
        node=zone_verify,
        sources=_sources(ex.local.comply_with_zoning if ex.local else None),
        additional_instruction="Verify local zoning compliance requirements from Baton Rouge or East Baton Rouge Parish resources."
    )

    # Occupational license may be required (non-critical leaf)
    occ_exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.local.occupational_license_may_be_required if ex.local else None),
        id="Occupational_License_May_Be_Required_exists",
        desc="Occupational license note mentioned with sources",
        parent=node,
        critical=False
    )
    occ_verify = evaluator.add_leaf(
        id="Occupational_License_May_Be_Required",
        desc="Note that an occupational license may be required from East Baton Rouge Parish (depending on local rules)",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="An occupational license may be required by East Baton Rouge Parish for business operations, depending on local rules.",
        node=occ_verify,
        sources=_sources(ex.local.occupational_license_may_be_required if ex.local else None),
        additional_instruction="Verify any occupational license requirements or business license rules for East Baton Rouge Parish."
    )


async def build_dod_cmmc(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="DoD_Cybersecurity_CMMC",
        desc="DoD-specific cybersecurity requirement (conditional on pursuing DoD contracts)",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.dod_cmmc.meet_applicable_cmmc_level if ex.dod_cmmc else None),
        id="Meet_Applicable_CMMC_Level_exists",
        desc="CMMC requirement mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="Meet_Applicable_CMMC_Level",
        desc="For DoD contracts, meet the applicable CMMC level requirement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="DoD contracts require compliance with the applicable Cybersecurity Maturity Model Certification (CMMC) level.",
        node=verify_node,
        sources=_sources(ex.dod_cmmc.meet_applicable_cmmc_level if ex.dod_cmmc else None),
        additional_instruction="Verify on official DoD/CMMC program resources that applicable CMMC level compliance is required."
    )


async def build_lpdes(evaluator: Evaluator, parent_node, ex: FirmSetupExtraction) -> None:
    node = evaluator.add_parallel(
        id="Environmental_LPDES_If_Applicable",
        desc="Environmental permitting requirement (conditional)",
        parent=parent_node,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_mentioned_and_has_sources(ex.lpdes.lpdes_if_discharging_pollutants if ex.lpdes else None),
        id="LPDES_If_Discharging_Pollutants_exists",
        desc="LPDES conditional requirement mentioned with sources",
        parent=node,
        critical=True
    )
    verify_node = evaluator.add_leaf(
        id="LPDES_If_Discharging_Pollutants",
        desc="Obtain an LPDES permit if the facility discharges pollutants into state waters (per LAC 33:Chapter IX)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="An LPDES permit is required if a facility discharges pollutants into state waters, per Louisiana environmental regulations.",
        node=verify_node,
        sources=_sources(ex.lpdes.lpdes_if_discharging_pollutants if ex.lpdes else None),
        additional_instruction="Verify LPDES permitting requirement under Louisiana Administrative Code (LAC 33) or official DEQ resources."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Louisiana engineering LLC regulatory requirements task.
    """
    # Initialize evaluator with a parallel root (non-critical to allow partial credit across categories)
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

    # Extract structured requirements and sources from the answer
    extracted: FirmSetupExtraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=FirmSetupExtraction,
        extraction_name="requirements_extraction"
    )

    # Record firm context info for summary
    evaluator.add_custom_info(
        info={
            "firm_name": extracted.firm_name or "Bayou Engineering Solutions, LLC",
            "office_city": extracted.office_city or "Baton Rouge",
            "office_state": extracted.office_state or "Louisiana",
            "employees_count_desc": extracted.employees_count_desc or "3-5 employees",
            "intent": "Federal contracts including potential DoD projects"
        },
        info_type="firm_context",
        info_name="firm_profile"
    )

    # Build verification tree per rubric
    await build_sos_llc_formation(evaluator, root, extracted)
    await build_lapels_firm_licensure(evaluator, root, extracted)
    await build_ldr_registration(evaluator, root, extracted)
    await build_ein(evaluator, root, extracted)
    await build_sam(evaluator, root, extracted)
    await build_naics(evaluator, root, extracted)
    await build_sba(evaluator, root, extracted)
    await build_far(evaluator, root, extracted)
    await build_workers_comp(evaluator, root, extracted)
    await build_la_unemployment(evaluator, root, extracted)
    await build_local_compliance(evaluator, root, extracted)
    await build_dod_cmmc(evaluator, root, extracted)
    await build_lpdes(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()