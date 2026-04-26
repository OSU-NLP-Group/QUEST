import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "fda_nmibc_biologic_2024"
TASK_DESCRIPTION = """
In 2024, the FDA approved a biologic drug for the treatment of BCG-unresponsive non-muscle invasive bladder cancer (NMIBC) with carcinoma in situ (CIS). This drug is administered intravesically in combination with BCG and received FDA Breakthrough Therapy Designation. The drug's manufacturer is headquartered in California, and the approval was based on a clinical trial that reported a Complete Response Rate.

Identify this drug and provide the following comprehensive information:

1. Drug Identification:
   - Brand name and generic name
   - Exact FDA approval date (month, day, year)
   - FDA reference URL confirming the approval

2. Manufacturer Information:
   - Full legal name of the manufacturer/sponsor company
   - Complete headquarters address (street address, city, state, ZIP code)
   - URL confirming the headquarters location

3. Regulatory Pathway:
   - Confirmation that the drug received Breakthrough Therapy Designation
   - URL confirming the Breakthrough Therapy Designation
   - Confirmation that the BLA submission included all required Form FDA 356h components (applicant information, product/manufacturing information, pre-clinical studies, clinical studies)
   - URL reference for BLA submission requirements

4. Clinical Evidence:
   - Name or NCT number of the pivotal clinical trial
   - Trial design characteristics (phase, whether open-label, single-arm or controlled, and whether multicenter)
   - URL reference for the clinical trial
   - Complete Response Rate percentage with 95% confidence interval
   - Number of patients in the efficacy population
   - Duration of Response range (minimum to maximum months observed)
   - Percentage of responders with Duration of Response ≥12 months
   - URL reference for the efficacy data
   - How Complete Response was defined in the trial

All information must be supported by verifiable URLs from authoritative sources (FDA, manufacturer, clinical trial registries, or peer-reviewed publications).
"""


# --------------------------- Data Models ---------------------------------- #
class DrugIdentification(BaseModel):
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    approval_date: Optional[str] = None
    fda_approval_url: Optional[str] = None

    biologic_urls: List[str] = Field(default_factory=list)
    indication_urls: List[str] = Field(default_factory=list)
    administration_urls: List[str] = Field(default_factory=list)
    combination_urls: List[str] = Field(default_factory=list)


class ManufacturerInfo(BaseModel):
    company_name: Optional[str] = None
    headquarters_address: Optional[str] = None
    headquarters_url: Optional[str] = None
    extra_hq_urls: List[str] = Field(default_factory=list)


class RegulatoryPathway(BaseModel):
    breakthrough_url: Optional[str] = None
    submission_urls: List[str] = Field(default_factory=list)
    bla_requirements_url: Optional[str] = None


class ClinicalEvidence(BaseModel):
    trial_id_or_name: Optional[str] = None
    trial_url: Optional[str] = None

    trial_phase: Optional[str] = None
    open_label_status: Optional[str] = None
    arms_structure: Optional[str] = None
    multicenter_status: Optional[str] = None

    efficacy_url: Optional[str] = None
    cr_rate_with_95ci: Optional[str] = None
    efficacy_population_n: Optional[str] = None
    dor_range_months: Optional[str] = None
    dor_ge_12_months_percentage: Optional[str] = None
    cr_definition: Optional[str] = None


class NMIBCApprovalExtraction(BaseModel):
    drug_identification: Optional[DrugIdentification] = None
    manufacturer_information: Optional[ManufacturerInfo] = None
    regulatory_pathway: Optional[RegulatoryPathway] = None
    clinical_evidence: Optional[ClinicalEvidence] = None


# ------------------------ Extraction Prompt ------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the requested structured information from the answer. Return exactly the fields defined by the JSON template.

    1) drug_identification:
       - brand_name: brand name of the drug
       - generic_name: generic name of the drug
       - approval_date: FDA approval date string exactly as stated (e.g., "April 22, 2024")
       - fda_approval_url: a single authoritative FDA URL confirming approval/indication/date
       - biologic_urls: all URLs in the answer that support that the product is a biologic/BLA
       - indication_urls: all URLs that support the indication (BCG-unresponsive NMIBC with CIS)
       - administration_urls: all URLs that support intravesical administration
       - combination_urls: all URLs that support use in combination with BCG

    2) manufacturer_information:
       - company_name: full legal name
       - headquarters_address: complete HQ address (street, city, state, ZIP)
       - headquarters_url: one authoritative URL confirming HQ location
       - extra_hq_urls: any additional URLs supporting HQ address

    3) regulatory_pathway:
       - breakthrough_url: one authoritative URL confirming Breakthrough Therapy Designation
       - submission_urls: URLs supporting that approval was via BLA (not NDA)
       - bla_requirements_url: authoritative URL describing BLA submission/Form FDA 356h requirements

    4) clinical_evidence:
       - trial_id_or_name: pivotal trial name or NCT number that supported approval
       - trial_url: authoritative URL for the clinical trial (e.g., ClinicalTrials.gov or peer-reviewed publication)
       - trial_phase: trial phase string (e.g., Phase 2/3)
       - open_label_status: whether open-label (e.g., "open-label" or "not open-label")
       - arms_structure: whether single-arm or controlled
       - multicenter_status: whether multicenter
       - efficacy_url: authoritative URL supporting efficacy outcomes
       - cr_rate_with_95ci: Complete Response Rate with 95% CI exactly as stated in the answer
       - efficacy_population_n: number of patients in the efficacy population exactly as stated
       - dor_range_months: Duration of Response range (min–max months) exactly as stated
       - dor_ge_12_months_percentage: percentage of responders with DOR ≥12 months
       - cr_definition: how Complete Response was defined

    Rules:
    - Extract only what appears in the answer; if missing, use null for missing single fields and [] for missing lists.
    - Include full URLs with protocol; ignore malformed URLs.
    """


# ------------------------ Helper Utilities -------------------------------- #
def _merge_sources(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for a in args:
        if a is None:
            continue
        if isinstance(a, list):
            for u in a:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
        elif isinstance(a, str):
            if a.strip():
                urls.append(a.strip())
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ------------------------ Verification Builders --------------------------- #
async def build_drug_identification(
    evaluator: Evaluator,
    parent,
    di: Optional[DrugIdentification],
) -> None:
    node = evaluator.add_parallel(
        id="drug_identification",
        desc="Identify the correct FDA-approved drug and provide the requested naming and approval-reference details",
        parent=parent,
        critical=True,
    )

    # Existence checks for key fields/URLs
    fda_url_provided = di is not None and di.fda_approval_url is not None and di.fda_approval_url.strip() != ""
    evaluator.add_custom_node(
        result=fda_url_provided,
        id="fda_approval_url_exists",
        desc="FDA approval URL is provided",
        parent=node,
        critical=True,
    )

    names_provided = di is not None and (di.brand_name or di.generic_name)
    evaluator.add_custom_node(
        result=bool(names_provided),
        id="brand_generic_exist",
        desc="Brand or generic name is provided",
        parent=node,
        critical=True,
    )

    # biologic confirmation
    biologic_leaf = evaluator.add_leaf(
        id="biologic_confirmation",
        desc="Confirms the product is a biologic (i.e., FDA-regulated as a biologic; consistent with BLA-based approval)",
        parent=node,
        critical=True,
    )
    bio_sources = _merge_sources(di.fda_approval_url if di else None, di.biologic_urls if di else None, di.subset_urls if hasattr(di, "subset_urls") else None)
    claim_biologic = f"The product {di.generic_name or ''} is a biologic and its approval is under a Biologics License Application (BLA)."
    await evaluator.verify(
        claim=claim_biologic,
        node=biologic_leaf,
        sources=bio_sources,
        additional_instruction="Verify on FDA or equivalent authoritative page that the product is regulated as a biologic and approved under a BLA (not an NDA).",
    )

    # indication: BCG-unresponsive NMIBC with CIS
    indication_leaf = evaluator.add_leaf(
        id="indication_bcg_unresponsive_nmibc_with_cis",
        desc="Confirms the drug is approved for BCG-unresponsive non-muscle invasive bladder cancer (NMIBC) with carcinoma in situ (CIS)",
        parent=node,
        critical=True,
    )
    ind_sources = _merge_sources(di.fda_approval_url if di else None, di.indication_urls if di else None)
    claim_indication = f"The drug {di.brand_name or ''} ({di.generic_name or ''}) is FDA-approved for BCG-unresponsive NMIBC with carcinoma in situ (CIS)."
    await evaluator.verify(
        claim=claim_indication,
        node=indication_leaf,
        sources=ind_sources,
        additional_instruction="Confirm that the indication explicitly mentions BCG-unresponsive NMIBC with CIS on FDA or label/manufacturer sources.",
    )

    # intravesical administration
    intravesical_leaf = evaluator.add_leaf(
        id="intravesical_administration",
        desc="Confirms the drug is administered intravesically",
        parent=node,
        critical=True,
    )
    admin_sources = _merge_sources(di.administration_urls if di else None, di.fda_approval_url if di else None)
    claim_admin = f"The drug {di.brand_name or ''} is administered intravesically."
    await evaluator.verify(
        claim=claim_admin,
        node=intravesical_leaf,
        sources=admin_sources,
        additional_instruction="Confirm route of administration is intravesical on authoritative sources (e.g., labeling, FDA announcement, manufacturer site).",
    )

    # combination with BCG
    combo_leaf = evaluator.add_leaf(
        id="combination_with_bcg",
        desc="Confirms the drug is approved/used in combination with BCG",
        parent=node,
        critical=True,
    )
    combo_sources = _merge_sources(di.combination_urls if di else None, di.fda_approval_url if di else None)
    claim_combo = f"The drug {di.brand_name or ''} is used in combination with BCG for the indicated NMIBC population."
    await evaluator.verify(
        claim=claim_combo,
        node=combo_leaf,
        sources=combo_sources,
        additional_instruction="Verify that BCG co-administration is part of the regimen on FDA or authoritative sources.",
    )

    # brand and generic names
    names_leaf = evaluator.add_leaf(
        id="brand_and_generic_names",
        desc="Provides both the brand name and the generic name of the drug",
        parent=node,
        critical=True,
    )
    names_sources = _merge_sources(di.fda_approval_url if di else None, di.indication_urls if di else None, di.biologic_urls if di else None)
    claim_names = f"The drug's brand name is '{di.brand_name or ''}' and the generic name is '{di.generic_name or ''}'."
    await evaluator.verify(
        claim=claim_names,
        node=names_leaf,
        sources=names_sources,
        additional_instruction="Check that both the brand and generic names match authoritative sources (FDA/label/manufacturer).",
    )

    # exact FDA approval date in 2024
    date_leaf = evaluator.add_leaf(
        id="exact_fda_approval_date_2024",
        desc="Provides the exact FDA approval date (month, day, year) and the date is in calendar year 2024",
        parent=node,
        critical=True,
    )
    date_sources = _merge_sources(di.fda_approval_url if di else None)
    date_val = di.approval_date or ""
    claim_date = f"The FDA approval date for {di.brand_name or ''} ({di.generic_name or ''}) was {date_val}, and it occurred in 2024."
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=date_sources,
        additional_instruction="Confirm the exact calendar date and ensure the year is 2024 on the FDA approval page.",
    )

    # FDA approval URL itself
    fda_url_leaf = evaluator.add_leaf(
        id="fda_approval_url",
        desc="Provides an authoritative FDA URL that confirms the approval (supporting approval date and indication)",
        parent=node,
        critical=True,
    )
    fda_sources = _merge_sources(di.fda_approval_url if di else None)
    claim_fda_url = f"This FDA webpage confirms the approval and indication for {di.brand_name or ''} ({di.generic_name or ''})."
    await evaluator.verify(
        claim=claim_fda_url,
        node=fda_url_leaf,
        sources=fda_sources,
        additional_instruction="Verify that the provided URL is an FDA or accessdata.fda.gov page explicitly confirming approval and indication.",
    )


async def build_manufacturer_information(
    evaluator: Evaluator,
    parent,
    mi: Optional[ManufacturerInfo],
) -> None:
    node = evaluator.add_parallel(
        id="manufacturer_information",
        desc="Provide manufacturer/sponsor identity and headquarters details",
        parent=parent,
        critical=True,
    )

    hq_url_provided = mi is not None and mi.headquarters_url is not None and mi.headquarters_url.strip() != ""
    evaluator.add_custom_node(
        result=hq_url_provided,
        id="headquarters_url_exists",
        desc="Headquarters confirmation URL is provided",
        parent=node,
        critical=True,
    )

    # manufacturer legal name
    name_leaf = evaluator.add_leaf(
        id="manufacturer_legal_name",
        desc="Provides the full legal name of the manufacturer/sponsor company",
        parent=node,
        critical=True,
    )
    name_sources = _merge_sources(mi.headquarters_url if mi else None, mi.extra_hq_urls if mi else None)
    claim_company = f"The full legal name of the manufacturer/sponsor is '{mi.company_name or ''}'."
    await evaluator.verify(
        claim=claim_company,
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Confirm the company's legal name on an authoritative page (company site, SEC filing, FDA page).",
    )

    # complete headquarters address
    address_leaf = evaluator.add_leaf(
        id="headquarters_address_complete",
        desc="Provides the complete headquarters address (street, city, state, ZIP code)",
        parent=node,
        critical=True,
    )
    addr_sources = _merge_sources(mi.headquarters_url if mi else None, mi.extra_hq_urls if mi else None)
    claim_addr = f"The company's headquarters address is '{mi.headquarters_address or ''}', including street, city, state, and ZIP code."
    await evaluator.verify(
        claim=claim_addr,
        node=address_leaf,
        sources=addr_sources,
        additional_instruction="Verify the address exactly and ensure it includes street address, city, state, and ZIP code on an authoritative page.",
    )

    # headquarters in California
    ca_leaf = evaluator.add_leaf(
        id="headquarters_in_california",
        desc="Confirms the manufacturer headquarters is located in California",
        parent=node,
        critical=True,
    )
    ca_sources = _merge_sources(mi.headquarters_url if mi else None, mi.extra_hq_urls if mi else None)
    claim_ca = "The headquarters address is located in California (state of CA)."
    await evaluator.verify(
        claim=claim_ca,
        node=ca_leaf,
        sources=ca_sources,
        additional_instruction="Confirm that the HQ address indicates California or CA. Minor formatting variations are acceptable.",
    )

    # headquarters address URL validity
    hq_url_leaf = evaluator.add_leaf(
        id="headquarters_address_url",
        desc="Provides an authoritative URL confirming the headquarters location/address",
        parent=node,
        critical=True,
    )
    hq_sources = _merge_sources(mi.headquarters_url if mi else None)
    claim_hq_url = "This webpage confirms the company's headquarters location/address."
    await evaluator.verify(
        claim=claim_hq_url,
        node=hq_url_leaf,
        sources=hq_sources,
        additional_instruction="Verify that the provided page is authoritative and explicitly lists the HQ address.",
    )


async def build_regulatory_pathway(
    evaluator: Evaluator,
    parent,
    rp: Optional[RegulatoryPathway],
    di: Optional[DrugIdentification],
) -> None:
    node = evaluator.add_parallel(
        id="regulatory_pathway",
        desc="Provide required regulatory pathway/designation details and supporting URLs",
        parent=parent,
        critical=True,
    )

    # Existence checks for critical URLs
    btd_url_provided = rp is not None and rp.breakthrough_url is not None and rp.breakthrough_url.strip() != ""
    evaluator.add_custom_node(
        result=btd_url_provided,
        id="breakthrough_url_exists",
        desc="Breakthrough Therapy Designation URL is provided",
        parent=node,
        critical=True,
    )
    bla_req_url_provided = rp is not None and rp.bla_requirements_url is not None and rp.bla_requirements_url.strip() != ""
    evaluator.add_custom_node(
        result=bla_req_url_provided,
        id="bla_requirements_url_exists",
        desc="BLA/Form FDA 356h requirements URL is provided",
        parent=node,
        critical=True,
    )

    # Breakthrough Therapy Designation confirmation
    btd_leaf = evaluator.add_leaf(
        id="breakthrough_therapy_designation_confirmation",
        desc="Confirms the drug received FDA Breakthrough Therapy Designation",
        parent=node,
        critical=True,
    )
    btd_sources = _merge_sources(rp.breakthrough_url if rp else None, di.fda_approval_url if di else None)
    claim_btd = "The drug received FDA Breakthrough Therapy Designation."
    await evaluator.verify(
        claim=claim_btd,
        node=btd_leaf,
        sources=btd_sources,
        additional_instruction="Verify that the page explicitly states Breakthrough Therapy Designation for the drug.",
    )

    # Breakthrough designation URL validity
    btd_url_leaf = evaluator.add_leaf(
        id="breakthrough_therapy_designation_url",
        desc="Provides an authoritative URL confirming Breakthrough Therapy Designation",
        parent=node,
        critical=True,
    )
    btd_url_sources = _merge_sources(rp.breakthrough_url if rp else None)
    claim_btd_url = "This webpage explicitly confirms FDA Breakthrough Therapy Designation for the drug."
    await evaluator.verify(
        claim=claim_btd_url,
        node=btd_url_leaf,
        sources=btd_url_sources,
        additional_instruction="Confirm the page is authoritative (FDA, manufacturer press release, or peer-reviewed source).",
    )

    # Approval based on BLA (not NDA)
    bla_leaf = evaluator.add_leaf(
        id="submission_type_bla_not_nda",
        desc="Confirms the approval was based on a Biologics License Application (BLA), not an NDA",
        parent=node,
        critical=True,
    )
    bla_sources = _merge_sources(di.fda_approval_url if di else None, rp.submission_urls if rp else None)
    claim_bla = "The approval was based on a Biologics License Application (BLA), not an NDA."
    await evaluator.verify(
        claim=claim_bla,
        node=bla_leaf,
        sources=bla_sources,
        additional_instruction="Verify the submission type indicated on FDA or authoritative sources; it should be a BLA.",
    )

    # Form FDA 356h components were included
    form_leaf = evaluator.add_leaf(
        id="form_356h_required_components_included_confirmation",
        desc="Confirms the BLA submission included Form FDA 356h components covering: applicant information; product/manufacturing information; pre-clinical studies; clinical studies",
        parent=node,
        critical=True,
    )
    form_sources = _merge_sources(rp.bla_requirements_url if rp else None)
    claim_form = "The BLA submission included Form FDA 356h components covering applicant information, product/manufacturing information, pre-clinical studies, and clinical studies."
    await evaluator.verify(
        claim=claim_form,
        node=form_leaf,
        sources=form_sources,
        additional_instruction="Use FDA requirements documentation to confirm these components are required and included in BLA submissions; exact phrasing variations are acceptable.",
    )

    # Requirements URL validity
    req_leaf = evaluator.add_leaf(
        id="bla_submission_requirements_url",
        desc="Provides an authoritative URL reference describing BLA submission/Form FDA 356h requirements",
        parent=node,
        critical=True,
    )
    req_sources = _merge_sources(rp.bla_requirements_url if rp else None)
    claim_req = "This page describes BLA submission/Form FDA 356h requirements, including the required components."
    await evaluator.verify(
        claim=claim_req,
        node=req_leaf,
        sources=req_sources,
        additional_instruction="Confirm the page is FDA or equivalent authoritative documentation about BLA and Form FDA 356h requirements.",
    )


async def build_clinical_evidence(
    evaluator: Evaluator,
    parent,
    ce: Optional[ClinicalEvidence],
) -> None:
    node = evaluator.add_parallel(
        id="clinical_evidence",
        desc="Provide pivotal trial identification, design, and required efficacy outcomes with URLs",
        parent=parent,
        critical=True,
    )

    # Existence checks for critical URLs
    trial_url_provided = ce is not None and ce.trial_url is not None and ce.trial_url.strip() != ""
    evaluator.add_custom_node(
        result=trial_url_provided,
        id="clinical_trial_url_exists",
        desc="Clinical trial URL is provided",
        parent=node,
        critical=True,
    )
    efficacy_url_provided = ce is not None and ce.efficacy_url is not None and ce.efficacy_url.strip() != ""
    evaluator.add_custom_node(
        result=efficacy_url_provided,
        id="efficacy_data_url_exists",
        desc="Efficacy data URL is provided",
        parent=node,
        critical=True,
    )

    # Pivotal trial identifier
    trial_id_leaf = evaluator.add_leaf(
        id="pivotal_trial_identifier",
        desc="Provides the pivotal clinical trial name or NCT number that supported approval",
        parent=node,
        critical=True,
    )
    trial_sources = _merge_sources(ce.trial_url if ce else None)
    claim_trial_id = f"The pivotal trial supporting approval is identified as '{ce.trial_id_or_name or ''}'."
    await evaluator.verify(
        claim=claim_trial_id,
        node=trial_id_leaf,
        sources=trial_sources,
        additional_instruction="Verify the trial identifier (name or NCT number) on ClinicalTrials.gov or a peer-reviewed publication.",
    )

    # BCG-unresponsive population
    pop_leaf = evaluator.add_leaf(
        id="trial_population_bcg_unresponsive",
        desc="Confirms the pivotal trial enrolled patients with BCG-unresponsive disease",
        parent=node,
        critical=True,
    )
    claim_pop = "The pivotal trial enrolled patients with BCG-unresponsive NMIBC."
    await evaluator.verify(
        claim=claim_pop,
        node=pop_leaf,
        sources=trial_sources,
        additional_instruction="Confirm the inclusion criteria mention BCG-unresponsive disease.",
    )

    # CR as primary endpoint
    cr_primary_leaf = evaluator.add_leaf(
        id="complete_response_primary_endpoint",
        desc="Confirms Complete Response Rate was reported as a primary efficacy endpoint in the pivotal trial",
        parent=node,
        critical=True,
    )
    cr_primary_sources = _merge_sources(ce.trial_url if ce else None, ce.efficacy_url if ce else None)
    claim_cr_primary = "Complete Response Rate was a primary efficacy endpoint in the pivotal trial."
    await evaluator.verify(
        claim=claim_cr_primary,
        node=cr_primary_leaf,
        sources=cr_primary_sources,
        additional_instruction="Verify on trial registry or publication that CR rate was a primary endpoint.",
    )

    # Clinical trial URL validity
    trial_url_leaf = evaluator.add_leaf(
        id="clinical_trial_url",
        desc="Provides an authoritative URL for the clinical trial (e.g., ClinicalTrials.gov or peer-reviewed publication)",
        parent=node,
        critical=True,
    )
    claim_trial_url = "This URL is an authoritative page for the clinical trial (ClinicalTrials.gov or peer-reviewed publication)."
    await evaluator.verify(
        claim=claim_trial_url,
        node=trial_url_leaf,
        sources=trial_sources,
        additional_instruction="Confirm that the page is a recognized registry (e.g., clinicaltrials.gov) or a peer-reviewed article.",
    )

    # Trial design characteristics
    phase_leaf = evaluator.add_leaf(
        id="trial_phase",
        desc="Provides the trial phase",
        parent=node,
        critical=True,
    )
    claim_phase = f"The trial phase was {ce.trial_phase or ''}."
    await evaluator.verify(
        claim=claim_phase,
        node=phase_leaf,
        sources=trial_sources,
        additional_instruction="Verify the trial phase on the authoritative trial page.",
    )

    open_label_leaf = evaluator.add_leaf(
        id="trial_open_label_status",
        desc="States whether the trial was open-label",
        parent=node,
        critical=True,
    )
    open_label_str = (ce.open_label_status or "").strip().lower()
    if open_label_str:
        is_open_label = "open-label" in open_label_str
    else:
        is_open_label = True  # default to positive phrasing to force verification failure if unsupported
    claim_open_label = "The trial was open-label." if is_open_label else "The trial was not open-label."
    await evaluator.verify(
        claim=claim_open_label,
        node=open_label_leaf,
        sources=trial_sources,
        additional_instruction="Verify whether the design indicates open-label or masked.",
    )

    arms_leaf = evaluator.add_leaf(
        id="trial_arms_structure",
        desc="States whether the trial was single-arm or controlled",
        parent=node,
        critical=True,
    )
    arms_str = (ce.arms_structure or "").strip().lower()
    claim_arms = "The trial was single-arm." if "single" in arms_str else "The trial was controlled (had comparative arms)."
    await evaluator.verify(
        claim=claim_arms,
        node=arms_leaf,
        sources=trial_sources,
        additional_instruction="Verify arm structure (single-arm vs controlled) per the trial description.",
    )

    multicenter_leaf = evaluator.add_leaf(
        id="trial_multicenter_status",
        desc="States whether the trial was multicenter",
        parent=node,
        critical=True,
    )
    multi_str = (ce.multicenter_status or "").strip().lower()
    claim_multi = "The trial was multicenter." if "multi" in multi_str else "The trial was single-center."
    await evaluator.verify(
        claim=claim_multi,
        node=multicenter_leaf,
        sources=trial_sources,
        additional_instruction="Verify whether multiple centers participated in the trial.",
    )

    # Efficacy outcomes
    cr_rate_leaf = evaluator.add_leaf(
        id="cr_rate_with_95ci",
        desc="Provides the Complete Response Rate percentage together with its 95% confidence interval",
        parent=node,
        critical=True,
    )
    claim_cr_rate = f"The Complete Response Rate was reported as {ce.cr_rate_with_95ci or ''}."
    await evaluator.verify(
        claim=claim_cr_rate,
        node=cr_rate_leaf,
        sources=_merge_sources(ce.efficacy_url if ce else None),
        additional_instruction="Confirm CR rate percentage and 95% CI exactly (minor rounding acceptable) on the efficacy source.",
    )

    n_leaf = evaluator.add_leaf(
        id="efficacy_population_n",
        desc="Provides the number of patients in the efficacy population",
        parent=node,
        critical=True,
    )
    claim_n = f"The number of patients in the efficacy population was {ce.efficacy_population_n or ''}."
    await evaluator.verify(
        claim=claim_n,
        node=n_leaf,
        sources=_merge_sources(ce.efficacy_url if ce else None),
        additional_instruction="Verify the efficacy population size on the efficacy source.",
    )

    dor_range_leaf = evaluator.add_leaf(
        id="duration_of_response_range_months",
        desc="Provides the Duration of Response range (minimum to maximum months observed)",
        parent=node,
        critical=True,
    )
    claim_dor_range = f"The Duration of Response ranged from {ce.dor_range_months or ''} months."
    await evaluator.verify(
        claim=claim_dor_range,
        node=dor_range_leaf,
        sources=_merge_sources(ce.efficacy_url if ce else None),
        additional_instruction="Verify DoR range (minimum to maximum months) on the efficacy source.",
    )

    dor_ge_12_leaf = evaluator.add_leaf(
        id="dor_ge_12_months_percentage",
        desc="Provides the percentage of responders with Duration of Response ≥12 months",
        parent=node,
        critical=True,
    )
    claim_dor_ge_12 = f"The percentage of responders with Duration of Response ≥12 months was {ce.dor_ge_12_months_percentage or ''}."
    await evaluator.verify(
        claim=claim_dor_ge_12,
        node=dor_ge_12_leaf,
        sources=_merge_sources(ce.efficacy_url if ce else None),
        additional_instruction="Verify the proportion of CR responders with DoR ≥12 months on the efficacy source.",
    )

    cr_def_leaf = evaluator.add_leaf(
        id="complete_response_definition",
        desc="Describes how Complete Response was defined in the pivotal trial",
        parent=node,
        critical=True,
    )
    claim_cr_def = f"Complete Response was defined as {ce.cr_definition or ''}."
    await evaluator.verify(
        claim=claim_cr_def,
        node=cr_def_leaf,
        sources=_merge_sources(ce.efficacy_url if ce else None, ce.trial_url if ce else None),
        additional_instruction="Verify the CR definition per the trial protocol/publication or efficacy write-up.",
    )

    efficacy_url_leaf = evaluator.add_leaf(
        id="efficacy_data_url",
        desc="Provides an authoritative URL supporting the reported efficacy and duration-of-response data",
        parent=node,
        critical=True,
    )
    claim_efficacy_url = "This webpage provides authoritative efficacy outcomes including CR rate and duration-of-response data."
    await evaluator.verify(
        claim=claim_efficacy_url,
        node=efficacy_url_leaf,
        sources=_merge_sources(ce.efficacy_url if ce else None),
        additional_instruction="Confirm the page is authoritative (FDA labeling, FDA review, manufacturer data, or peer-reviewed publication).",
    )


# ------------------------ Main Evaluation --------------------------------- #
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

    # Create a critical task root under the evaluator root (since initialize() root is non-critical)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify the 2024 FDA-approved biologic meeting the NMIBC criteria and provide the requested drug, manufacturer, regulatory, and clinical evidence details with authoritative URLs",
        parent=root,
        critical=True,
    )

    # Extract all structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=NMIBCApprovalExtraction,
        extraction_name="nmibc_biologic_extraction",
    )

    # Build verification subtrees
    await build_drug_identification(evaluator, task_root, extracted.drug_identification)
    await build_manufacturer_information(evaluator, task_root, extracted.manufacturer_information)
    await build_regulatory_pathway(evaluator, task_root, extracted.regulatory_pathway, extracted.drug_identification)
    await build_clinical_evidence(evaluator, task_root, extracted.clinical_evidence)

    return evaluator.get_summary()