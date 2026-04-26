import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "novo_pap_2025"
TASK_DESCRIPTION = """
I am researching Novo Nordisk's Patient Assistance Program (PAP) to help a family member who may need financial assistance for a weight-loss medication. Please provide the following comprehensive information:

1. Identify which Novo Nordisk GLP-1 medication was FDA-approved specifically for chronic weight management (not for diabetes treatment) with an approval date between January 1, 2020 and December 31, 2022. Include a reference URL from either FDA.gov or Drugs.com that confirms this medication's FDA approval for weight management.

2. State the exact FDA approval date (month, day, and year) for this medication's weight management indication.

3. According to the FDA prescribing information for this medication, list all weight-related comorbid conditions that qualify an adult with a BMI between 27 and less than 30 kg/m² (overweight category) to receive this medication.

4. Describe all four key eligibility requirements for Novo Nordisk's Patient Assistance Program in 2025, specifically addressing: (a) citizenship or residency status requirements, (b) the household income limit expressed as a percentage of the federal poverty level, (c) insurance coverage requirements, and (d) restrictions regarding enrollment in other government assistance programs.

5. List at least three types of documents that Novo Nordisk accepts as proof of income when applying for the Patient Assistance Program.

6. Provide the specific annual income threshold (at 400% of the federal poverty level) that qualifies a household of exactly 3 people for the PAP in 2025.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MedicationApprovalExtraction(BaseModel):
    medication_name: Optional[str] = None
    approval_date: Optional[str] = None
    approval_sources: List[str] = Field(default_factory=list)


class LabelComorbiditiesExtraction(BaseModel):
    label_sources: List[str] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)


class PAPEligibilityExtraction(BaseModel):
    citizenship_residency_requirement: Optional[str] = None
    income_limit_percent_fpl: Optional[str] = None
    insurance_coverage_requirement: Optional[str] = None
    government_assistance_restrictions: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProofOfIncomeExtraction(BaseModel):
    documents: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FPLThresholdExtraction(BaseModel):
    threshold_amount: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_medication_approval() -> str:
    return """
    Identify the Novo Nordisk GLP-1 medication in the answer that is FDA-approved specifically for chronic weight management (not for diabetes treatment) with an approval date between January 1, 2020 and December 31, 2022.
    Extract:
    - medication_name: the medication's brand name (e.g., Wegovy).
    - approval_date: the exact FDA approval date for the chronic weight management indication (month, day, year as presented in the answer; if multiple dates are mentioned, choose the one explicitly tied to weight management approval).
    - approval_sources: all URLs in the answer that confirm the FDA approval for weight management. Only include URLs from FDA.gov or Drugs.com. If no such URLs are provided, return an empty list.
    """


def prompt_extract_label_comorbidities() -> str:
    return """
    From the answer, extract the list of weight-related comorbid conditions quoted from the FDA prescribing information (or FDA-hosted label) that qualify an adult with a BMI ≥27 and <30 kg/m² for the medication.
    Extract:
    - comorbidities: array of conditions exactly as stated in the answer (e.g., hypertension, type 2 diabetes mellitus, dyslipidemia).
    - label_sources: URLs cited for the FDA prescribing information/label. Prefer FDA-hosted pages (e.g., accessdata.fda.gov). If none are provided, return an empty list.
    """


def prompt_extract_pap_eligibility() -> str:
    return """
    Extract the four key eligibility requirements for Novo Nordisk's Patient Assistance Program (PAP) for 2025 as stated in the answer, along with official citation URLs.
    Extract:
    - citizenship_residency_requirement: e.g., US citizen or legal resident, exactly as described in the answer.
    - income_limit_percent_fpl: the household income limit expressed as a percentage of FPL (e.g., 400% FPL) as stated in the answer.
    - insurance_coverage_requirement: description of insurance requirement for PAP eligibility (e.g., uninsured, Medicare allowed, private/commercial insurance not eligible), as stated in the answer.
    - government_assistance_restrictions: restrictions related to enrollment in other government assistance programs (e.g., Medicaid, VA benefits, LIS/Extra Help), as stated in the answer.
    - sources: all URLs cited for these requirements, limited to official Novo Nordisk PAP pages (e.g., NovoCare, Novo Nordisk US). If none are provided, return an empty list.
    """


def prompt_extract_proof_of_income() -> str:
    return """
    Extract at least three types of acceptable proof-of-income documents that Novo Nordisk accepts for PAP applications, as provided in the answer.
    Extract:
    - documents: array listing each proof-of-income document type mentioned (e.g., recent pay stubs, W-2, tax return, Social Security award letter).
    - sources: official citation URLs (e.g., NovoCare PAP pages). If none are provided, return an empty list.
    """


def prompt_extract_fpl_threshold() -> str:
    return """
    Extract the specific annual income threshold at 400% of the federal poverty level for a household size of exactly 3 people (for 2025), as stated in the answer.
    Extract:
    - threshold_amount: the dollar amount string as presented (e.g., $106,600).
    - sources: official poverty guideline/FPL citation URLs (prefer HHS/ASPE). If none are provided, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _filter_domains(urls: List[str], allowed_domains: List[str]) -> List[str]:
    allowed = []
    for u in urls:
        low = u.lower()
        if any(domain in low for domain in allowed_domains):
            allowed.append(u)
    return allowed


def _parse_date_str(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    ds = date_str.strip()
    fmts = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]
    for f in fmts:
        try:
            return datetime.strptime(ds, f)
        except Exception:
            continue
    return None


def _is_in_range(dt: Optional[datetime], start: datetime, end: datetime) -> bool:
    if dt is None:
        return False
    return start <= dt <= end


def _currency_to_int(amount_str: Optional[str]) -> Optional[int]:
    if not amount_str:
        return None
    s = "".join(ch for ch in amount_str if ch.isdigit())
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_medication_research(
    evaluator: Evaluator,
    parent_node,
    med_info: MedicationApprovalExtraction,
    comorb_info: LabelComorbiditiesExtraction,
) -> None:
    med_node = evaluator.add_sequential(
        id="MedicationResearch",
        desc="Identify the correct Novo Nordisk GLP-1 weight-management medication (2020–2022) and extract required FDA-label details.",
        parent=parent_node,
        critical=True
    )

    # IdentifyMedicationWithAllowedApprovalSource - existence gate
    allowed_approval_sources = _filter_domains(_ensure_list(med_info.approval_sources), ["fda.gov", "drugs.com"])
    has_med = bool(med_info.medication_name and med_info.medication_name.strip())
    has_allowed_src = len(allowed_approval_sources) > 0

    evaluator.add_custom_node(
        result=has_med and has_allowed_src,
        id="IdentifyMedicationWithAllowedApprovalSource_existence",
        desc="Medication identified with at least one allowed approval source (FDA.gov or Drugs.com)",
        parent=med_node,
        critical=True
    )

    # IdentifyMedicationWithAllowedApprovalSource - verification leaf
    leaf_med_ident = evaluator.add_leaf(
        id="IdentifyMedicationWithAllowedApprovalSource",
        desc="Identifies the Novo Nordisk GLP-1 medication approved for chronic weight management with an allowed source.",
        parent=med_node,
        critical=True
    )

    claim_med = (
        f"The Novo Nordisk GLP-1 medication '{med_info.medication_name or ''}' is FDA-approved specifically for chronic weight management (not for diabetes)."
    )
    await evaluator.verify(
        claim=claim_med,
        node=leaf_med_ident,
        sources=allowed_approval_sources,
        additional_instruction="Confirm on FDA.gov or Drugs.com that the medication is approved for chronic weight management (obesity/overweight). Ignore pages focused solely on diabetes indications."
    )

    # ProvideExactFDAApprovalDate - range check custom node
    start_dt = datetime(2020, 1, 1)
    end_dt = datetime(2022, 12, 31)
    parsed_dt = _parse_date_str(med_info.approval_date)
    evaluator.add_custom_node(
        result=_is_in_range(parsed_dt, start_dt, end_dt),
        id="ProvideExactFDAApprovalDate_range",
        desc="Approval date falls between 2020-01-01 and 2022-12-31",
        parent=med_node,
        critical=True
    )

    # ProvideExactFDAApprovalDate - verification leaf
    leaf_date = evaluator.add_leaf(
        id="ProvideExactFDAApprovalDate",
        desc="States the exact FDA approval date for the weight-management indication with an allowed source.",
        parent=med_node,
        critical=True
    )
    claim_date = (
        f"The FDA approval date for the weight-management indication of '{med_info.medication_name or ''}' is {med_info.approval_date or ''}."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=allowed_approval_sources,
        additional_instruction="Verify that the date corresponds to the approval for the weight-management indication. Allow typical date formatting variations but ensure the month, day, and year match."
    )

    # ListAllEligibleComorbiditiesFromFDALabel - sub-parallel group
    comorb_node = evaluator.add_parallel(
        id="ListAllEligibleComorbiditiesFromFDALabel",
        desc="Lists all weight-related comorbidities qualifying adults with BMI ≥27 and <30 kg/m² per the FDA label.",
        parent=med_node,
        critical=True
    )

    label_sources_allowed = _filter_domains(_ensure_list(comorb_info.label_sources), ["fda.gov"])
    has_comorb_list = len(comorb_info.comorbidities) > 0
    evaluator.add_custom_node(
        result=has_comorb_list and len(label_sources_allowed) > 0,
        id="Comorbidities_existence",
        desc="Comorbidities listed with at least one FDA label/source URL",
        parent=comorb_node,
        critical=True
    )

    # Verify presence of key examples typically cited on FDA labels for weight-management indication
    # 1) Hypertension
    leaf_htn = evaluator.add_leaf(
        id="Comorbidity_hypertension_present",
        desc="FDA label includes hypertension as a qualifying weight-related comorbidity.",
        parent=comorb_node,
        critical=True
    )
    claim_htn = (
        "According to the FDA prescribing information for the medication, hypertension is listed as a qualifying weight-related comorbidity for patients with BMI ≥27 and <30 kg/m²."
    )
    await evaluator.verify(
        claim=claim_htn,
        node=leaf_htn,
        sources=label_sources_allowed,
        additional_instruction="Look for the indication statement describing use in adults with BMI ≥27 and <30 with at least one weight-related comorbidity. Confirm that 'hypertension' appears as an example."
    )

    # 2) Type 2 diabetes mellitus
    leaf_t2dm = evaluator.add_leaf(
        id="Comorbidity_t2dm_present",
        desc="FDA label includes type 2 diabetes mellitus as a qualifying weight-related comorbidity.",
        parent=comorb_node,
        critical=True
    )
    claim_t2dm = (
        "According to the FDA prescribing information for the medication, type 2 diabetes mellitus is listed as a qualifying weight-related comorbidity for patients with BMI ≥27 and <30 kg/m²."
    )
    await evaluator.verify(
        claim=claim_t2dm,
        node=leaf_t2dm,
        sources=label_sources_allowed,
        additional_instruction="Confirm that 'type 2 diabetes mellitus' (or equivalent phrasing) appears in the label as an example of weight-related comorbidities."
    )

    # 3) Dyslipidemia
    leaf_dys = evaluator.add_leaf(
        id="Comorbidity_dyslipidemia_present",
        desc="FDA label includes dyslipidemia as a qualifying weight-related comorbidity.",
        parent=comorb_node,
        critical=True
    )
    claim_dys = (
        "According to the FDA prescribing information for the medication, dyslipidemia is listed as a qualifying weight-related comorbidity for patients with BMI ≥27 and <30 kg/m²."
    )
    await evaluator.verify(
        claim=claim_dys,
        node=leaf_dys,
        sources=label_sources_allowed,
        additional_instruction="Confirm that 'dyslipidemia' appears in the label as an example of weight-related comorbidities."
    )

    # 4) Accuracy of the provided list
    leaf_list_acc = evaluator.add_leaf(
        id="Comorbidities_list_accuracy",
        desc="The provided comorbidities list matches the FDA label's examples (hypertension, type 2 diabetes mellitus, dyslipidemia).",
        parent=comorb_node,
        critical=True
    )
    comorb_str = ", ".join(comorb_info.comorbidities) if comorb_info.comorbidities else ""
    claim_list = (
        f"The answer lists the following comorbidities: {comorb_str}. This matches the examples shown in the FDA label (hypertension, type 2 diabetes mellitus, and dyslipidemia) for BMI ≥27 and <30 kg/m² eligibility."
    )
    await evaluator.verify(
        claim=claim_list,
        node=leaf_list_acc,
        sources=label_sources_allowed,
        additional_instruction="Check that the listed comorbidities from the answer correspond to those examples in the FDA label for the BMI ≥27 and <30 kg/m² eligibility criteria."
    )

    # Record some debug info
    evaluator.add_custom_info(
        info={"allowed_approval_sources": allowed_approval_sources, "label_sources_allowed": label_sources_allowed},
        info_type="debug",
        info_name="medication_sources_info"
    )


async def verify_pap_research(
    evaluator: Evaluator,
    parent_node,
    pap_info: PAPEligibilityExtraction,
    proof_info: ProofOfIncomeExtraction,
    fpl_info: FPLThresholdExtraction,
) -> None:
    pap_node = evaluator.add_parallel(
        id="PAPResearch2025",
        desc="Provide Novo Nordisk PAP eligibility requirements (2025), proof-of-income documents, and the 400% FPL threshold for a household of 3.",
        parent=parent_node,
        critical=True
    )

    # Eligibility requirements
    elig_node = evaluator.add_parallel(
        id="PAPKeyEligibilityRequirements",
        desc="Describes the four key PAP eligibility requirements for 2025 with official citations.",
        parent=pap_node,
        critical=True
    )

    pap_sources_allowed = _filter_domains(_ensure_list(pap_info.sources), ["novocare", "novonordisk"])
    has_pap_src = len(pap_sources_allowed) > 0

    # Citizenship/residency requirement
    evaluator.add_custom_node(
        result=bool(pap_info.citizenship_residency_requirement and pap_info.citizenship_residency_requirement.strip()) and has_pap_src,
        id="CitizenshipOrResidencyRequirement_existence",
        desc="Citizenship/residency requirement provided with official PAP source",
        parent=elig_node,
        critical=True
    )
    leaf_cit = evaluator.add_leaf(
        id="CitizenshipOrResidencyRequirement",
        desc="States PAP citizenship/residency eligibility requirement for 2025 with an official citation.",
        parent=elig_node,
        critical=True
    )
    claim_cit = (
        f"For the 2025 Novo Nordisk PAP, the citizenship/residency requirement states: {pap_info.citizenship_residency_requirement or ''}."
    )
    await evaluator.verify(
        claim=claim_cit,
        node=leaf_cit,
        sources=pap_sources_allowed,
        additional_instruction="Confirm the eligibility statement on an official NovoCare/Novo Nordisk PAP page. Allow typical phrasing variants (e.g., 'U.S. citizen' vs 'citizen of the United States')."
    )

    # Income limit %FPL
    evaluator.add_custom_node(
        result=bool(pap_info.income_limit_percent_fpl and pap_info.income_limit_percent_fpl.strip()) and has_pap_src,
        id="HouseholdIncomeLimitPercentFPL_existence",
        desc="Income limit %FPL provided with official PAP source",
        parent=elig_node,
        critical=True
    )
    leaf_fpl_pct = evaluator.add_leaf(
        id="HouseholdIncomeLimitPercentFPL",
        desc="States the PAP household income limit (400% FPL) for 2025 with official citation.",
        parent=elig_node,
        critical=True
    )
    claim_fpl_pct = (
        f"For the 2025 Novo Nordisk PAP, the household income limit is {pap_info.income_limit_percent_fpl or ''} of the federal poverty level."
    )
    await evaluator.verify(
        claim=claim_fpl_pct,
        node=leaf_fpl_pct,
        sources=pap_sources_allowed,
        additional_instruction="Verify that the PAP states a 400% FPL income limit for eligibility in 2025."
    )

    # Insurance coverage requirement
    evaluator.add_custom_node(
        result=bool(pap_info.insurance_coverage_requirement and pap_info.insurance_coverage_requirement.strip()) and has_pap_src,
        id="InsuranceCoverageRequirement_existence",
        desc="Insurance coverage requirement provided with official PAP source",
        parent=elig_node,
        critical=True
    )
    leaf_ins = evaluator.add_leaf(
        id="InsuranceCoverageRequirement",
        desc="States the PAP insurance coverage requirement for 2025 with official citation.",
        parent=elig_node,
        critical=True
    )
    claim_ins = (
        f"For the 2025 Novo Nordisk PAP, the insurance coverage requirement is described as: {pap_info.insurance_coverage_requirement or ''}."
    )
    await evaluator.verify(
        claim=claim_ins,
        node=leaf_ins,
        sources=pap_sources_allowed,
        additional_instruction="Confirm the insurance requirement (e.g., uninsured or certain coverage types; private/commercial might be ineligible) on an official PAP page."
    )

    # Government assistance restrictions
    evaluator.add_custom_node(
        result=bool(pap_info.government_assistance_restrictions and pap_info.government_assistance_restrictions.strip()) and has_pap_src,
        id="GovernmentAssistanceProgramRestrictions_existence",
        desc="Government assistance restrictions provided with official PAP source",
        parent=elig_node,
        critical=True
    )
    leaf_gov = evaluator.add_leaf(
        id="GovernmentAssistanceProgramRestrictions",
        desc="States PAP restrictions regarding other government assistance programs with official citation.",
        parent=elig_node,
        critical=True
    )
    claim_gov = (
        f"For the 2025 Novo Nordisk PAP, the restrictions regarding government assistance programs are: {pap_info.government_assistance_restrictions or ''}."
    )
    await evaluator.verify(
        claim=claim_gov,
        node=leaf_gov,
        sources=pap_sources_allowed,
        additional_instruction="Confirm whether enrollment/eligibility in programs like Medicaid, VA benefits, LIS/Extra Help restricts PAP eligibility, as stated on the official PAP page."
    )

    # Proof of income documents
    proof_sources_allowed = _filter_domains(_ensure_list(proof_info.sources), ["novocare", "novonordisk"])
    has_three_docs = len(proof_info.documents) >= 3 and all(doc.strip() for doc in proof_info.documents)
    evaluator.add_custom_node(
        result=has_three_docs and len(proof_sources_allowed) > 0,
        id="ProofOfIncomeDocuments_existence",
        desc="At least three acceptable proof-of-income documents provided with official source",
        parent=pap_node,
        critical=True
    )
    leaf_proof = evaluator.add_leaf(
        id="ProofOfIncomeDocuments",
        desc="Lists at least three acceptable proof-of-income documents with official citation.",
        parent=pap_node,
        critical=True
    )
    docs_str = ", ".join(proof_info.documents) if proof_info.documents else ""
    claim_proof = (
        f"Novo Nordisk PAP accepts the following as proof of income: {docs_str}."
    )
    await evaluator.verify(
        claim=claim_proof,
        node=leaf_proof,
        sources=proof_sources_allowed,
        additional_instruction="Verify on the official NovoCare/Novo Nordisk PAP page that these document types are accepted as proof of income."
    )

    # 400% FPL threshold for household of 3
    fpl_sources_allowed = _filter_domains(_ensure_list(fpl_info.sources), ["aspe.hhs.gov", "hhs.gov"])
    threshold_int = _currency_to_int(fpl_info.threshold_amount)
    evaluator.add_custom_node(
        result=(threshold_int == 106600) and len(fpl_sources_allowed) > 0,
        id="HouseholdOf3IncomeThresholdAt400PercentFPL_existence",
        desc="Threshold amount equals $106,600 and an official FPL source is provided",
        parent=pap_node,
        critical=True
    )
    leaf_threshold = evaluator.add_leaf(
        id="HouseholdOf3IncomeThresholdAt400PercentFPL",
        desc="Provides the 2025 400% FPL threshold for a household of 3 ($106,600) with official FPL source.",
        parent=pap_node,
        critical=True
    )
    claim_threshold = (
        "In 2025, 400% of the federal poverty level for a household of 3 is $106,600."
    )
    await evaluator.verify(
        claim=claim_threshold,
        node=leaf_threshold,
        sources=fpl_sources_allowed,
        additional_instruction="Use the official HHS/ASPE poverty guidelines. You may compute 400% from the base FPL if the page provides the base value. Confirm the 400% amount equals $106,600 for 3-person households in the 48 contiguous states and D.C."
    )

    # Record some debug info
    evaluator.add_custom_info(
        info={
            "pap_sources_allowed": pap_sources_allowed,
            "proof_sources_allowed": proof_sources_allowed,
            "fpl_sources_allowed": fpl_sources_allowed
        },
        info_type="debug",
        info_name="pap_sources_info"
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

    # Top-level critical wrapper node (as per rubric)
    top_node = evaluator.add_parallel(
        id="PatientAssistanceProgramResearch",
        desc="Provide all requested medication and Novo Nordisk PAP (2025) information with appropriate citations.",
        parent=root,
        critical=True
    )

    # Run extractions (can be parallelized)
    med_task = evaluator.extract(
        prompt=prompt_extract_medication_approval(),
        template_class=MedicationApprovalExtraction,
        extraction_name="medication_approval"
    )
    comorb_task = evaluator.extract(
        prompt=prompt_extract_label_comorbidities(),
        template_class=LabelComorbiditiesExtraction,
        extraction_name="label_comorbidities"
    )
    pap_task = evaluator.extract(
        prompt=prompt_extract_pap_eligibility(),
        template_class=PAPEligibilityExtraction,
        extraction_name="pap_eligibility_2025"
    )
    proof_task = evaluator.extract(
        prompt=prompt_extract_proof_of_income(),
        template_class=ProofOfIncomeExtraction,
        extraction_name="pap_proof_of_income_2025"
    )
    fpl_task = evaluator.extract(
        prompt=prompt_extract_fpl_threshold(),
        template_class=FPLThresholdExtraction,
        extraction_name="fpl_threshold_2025"
    )

    med_info, comorb_info, pap_info, proof_info, fpl_info = await asyncio.gather(
        med_task, comorb_task, pap_task, proof_task, fpl_task
    )

    # Build verification branches
    await verify_medication_research(evaluator, top_node, med_info, comorb_info)
    await verify_pap_research(evaluator, top_node, pap_info, proof_info, fpl_info)

    return evaluator.get_summary()