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
TASK_ID = "first_nonprofit_gene_therapy_dec2025"
TASK_DESCRIPTION = (
    "Identify the first gene therapy from a non-profit organization that received FDA approval in December 2025. "
    "For this gene therapy, provide the following information with supporting reference URLs: "
    "(1) the therapy name (both brand name and generic name), "
    "(2) the rare disease indication for which it was approved, "
    "(3) confirmation that this is the first FDA-approved gene therapy for this specific indication, "
    "(4) the name of the non-profit organization that developed it, with evidence confirming its non-profit status, "
    "(5) the specific medical center or institution where the clinical trials were conducted, "
    "(6) the exact date of FDA approval, and "
    "(7) the complete patient eligibility criteria, including minimum age requirements and conditions regarding HLA-matched donor availability."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NamePair(BaseModel):
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class IndicationInfo(BaseModel):
    indication: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ApprovalDateInfo(BaseModel):
    approval_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DeveloperInfo(BaseModel):
    org_name: Optional[str] = None
    org_urls: List[str] = Field(default_factory=list)
    nonprofit_status_urls: List[str] = Field(default_factory=list)


class ClaimInfo(BaseModel):
    claim: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TrialCenterInfo(BaseModel):
    center_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ApprovalAnnouncementInfo(BaseModel):
    urls: List[str] = Field(default_factory=list)


class EligibilityInfo(BaseModel):
    min_age: Optional[str] = None
    hla_donor_condition: Optional[str] = None
    full_criteria_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TherapyExtraction(BaseModel):
    therapy_names: Optional[NamePair] = None
    approved_indication: Optional[IndicationInfo] = None
    approval_date: Optional[ApprovalDateInfo] = None
    developer: Optional[DeveloperInfo] = None
    first_nonprofit_claim: Optional[ClaimInfo] = None
    autologous_hsc: Optional[ClaimInfo] = None
    lentiviral_vector: Optional[ClaimInfo] = None
    orphan_designation: Optional[ClaimInfo] = None
    trial_center: Optional[TrialCenterInfo] = None
    approval_announcement_trial_location: Optional[ApprovalAnnouncementInfo] = None
    first_for_indication_claim: Optional[ClaimInfo] = None
    patient_eligibility: Optional[EligibilityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_therapy() -> str:
    return """
    Extract the complete set of information for a single gene therapy that the answer claims is the first therapy from a non-profit organization with FDA approval in December 2025.

    You must extract exactly and only what is explicitly present in the answer. Do not infer or invent any information.

    Return a JSON object with the following fields and nested structures:

    1) therapy_names:
       - brand_name: string or null
       - generic_name: string or null
       - urls: array of URLs mentioned in the answer that support the therapy names (brand↔generic mapping). If none, return empty array.

    2) approved_indication:
       - indication: string or null (rare disease indication approved by FDA)
       - urls: array of URLs that support the indication. If none, empty array.

    3) approval_date:
       - approval_date: string or null (the exact FDA approval date as stated in the answer; e.g., "December 14, 2025")
       - urls: array of URLs that support the approval date. If none, empty array.

    4) developer:
       - org_name: string or null (developer/sponsor organization)
       - org_urls: array of URLs about the organization (e.g., official site, announcement). If none, empty array.
       - nonprofit_status_urls: array of URLs that specifically indicate or prove the organization is non-profit (e.g., 501(c)(3), non-profit designation). If none, empty array.

    5) first_nonprofit_claim:
       - claim: string or null (the answer's sentence or phrasing asserting this is the first FDA-approved gene therapy from a non-profit organization)
       - urls: array of URLs provided to support that claim. If none, empty array.

    6) autologous_hsc:
       - claim: string or null (the answer's statement confirming autologous hematopoietic stem cell-based therapy)
       - urls: array of URLs supporting autologous HSC-based nature. If none, empty array.

    7) lentiviral_vector:
       - claim: string or null (the answer's statement confirming use of a lentiviral vector)
       - urls: array of URLs supporting lentiviral vector usage. If none, empty array.

    8) orphan_designation:
       - claim: string or null (the answer's statement confirming orphan drug designation for the therapy/indication)
       - urls: array of URLs supporting orphan drug designation. If none, empty array.

    9) trial_center:
       - center_name: string or null (specific named medical center/institution where clinical trials were conducted)
       - urls: array of URLs supporting the trial center. If none, empty array.

    10) approval_announcement_trial_location:
       - urls: array of URLs (ideally FDA approval announcement or equivalent official communication) that include clinical trial location details. If none, empty array.

    11) first_for_indication_claim:
       - claim: string or null (the answer's statement asserting this is the first FDA-approved gene therapy for the specified indication)
       - urls: array of URLs supporting this "first for indication" claim. If none, empty array.

    12) patient_eligibility:
       - min_age: string or null (minimum age requirement as stated in the answer)
       - hla_donor_condition: string or null (conditions regarding HLA-matched donor availability as stated in the answer)
       - full_criteria_text: string or null (the complete or summarized eligibility criteria from the answer)
       - urls: array of URLs supporting patient eligibility criteria (label, FDA doc, etc.). If none, empty array.

    SPECIAL RULES:
    - Extract only URLs explicitly present in the answer (including plain text links or markdown links). Do not infer or fabricate URLs.
    - If a required field is not found, return null (for strings) or empty array (for URLs).
    - Do not include duplicate URLs; keep unique URLs only.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            u = (url or "").strip()
            if u and u not in combined:
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_therapy_identification(
    evaluator: Evaluator,
    parent_node,
    ex: TherapyExtraction,
) -> None:
    """
    Build 'Therapy_Identification' subtree with two critical leaves:
    - Therapy_Names_Brand_And_Generic_With_URL
    - Approved_Indication_With_URL
    """
    node = evaluator.add_parallel(
        id="Therapy_Identification",
        desc="Identify the therapy and its approved indication, with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Therapy_Names_Brand_And_Generic_With_URL
    names_leaf = evaluator.add_leaf(
        id="Therapy_Names_Brand_And_Generic_With_URL",
        desc="Provide the therapy name including both brand name and generic name, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    brand = _safe(ex.therapy_names.brand_name if ex.therapy_names else None)
    generic = _safe(ex.therapy_names.generic_name if ex.therapy_names else None)
    name_sources = ex.therapy_names.urls if ex.therapy_names else []
    name_claim = f"The therapy brand name is '{brand}' and the generic name is '{generic}'."
    await evaluator.verify(
        claim=name_claim,
        node=names_leaf,
        sources=name_sources,
        additional_instruction=(
            "Verify that the provided sources explicitly confirm the therapy's brand name and its corresponding generic name. "
            "Allow reasonable naming variants and capitalization differences, but the mapping must be clear."
        )
    )

    # Leaf: Approved_Indication_With_URL
    indication_leaf = evaluator.add_leaf(
        id="Approved_Indication_With_URL",
        desc="Provide the FDA-approved rare disease indication for the therapy, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    indication_text = _safe(ex.approved_indication.indication if ex.approved_indication else None)
    indication_sources = ex.approved_indication.urls if ex.approved_indication else []
    indication_claim = f"This therapy is FDA-approved for the indication: {indication_text}."
    await evaluator.verify(
        claim=indication_claim,
        node=indication_leaf,
        sources=indication_sources,
        additional_instruction=(
            "Confirm that the cited sources state the FDA-approved indication for the therapy. Prefer official labeling, FDA communications, or authoritative sources."
        )
    )


async def build_constraints_and_details(
    evaluator: Evaluator,
    parent_node,
    ex: TherapyExtraction,
) -> None:
    """
    Build 'Constraint_and_Detail_Verification' subtree with multiple critical leaves.
    """
    node = evaluator.add_parallel(
        id="Constraint_and_Detail_Verification",
        desc="Verify all constraints and supply remaining required details, each with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # FDA Approval Date in December 2025
    approval_leaf = evaluator.add_leaf(
        id="FDA_Approval_Exact_Date_In_December_2025_With_URL",
        desc="Provide the exact FDA approval date (day month year) and verify from the cited source that it falls in December 2025, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    approval_date_str = _safe(ex.approval_date.approval_date if ex.approval_date else None)
    approval_sources = ex.approval_date.urls if ex.approval_date else []
    approval_claim = f"The FDA approval date was {approval_date_str}, and it falls in December 2025."
    await evaluator.verify(
        claim=approval_claim,
        node=approval_leaf,
        sources=approval_sources,
        additional_instruction=(
            "Verify the exact approval date from the provided source(s). Ensure the month is December and the year is 2025."
        )
    )

    # Nonprofit developer organization
    nonprofit_leaf = evaluator.add_leaf(
        id="Nonprofit_Developer_Name_And_Nonprofit_Status_Evidence_With_URL",
        desc="Provide the developer/sponsor organization name and evidence it is a non-profit organization (not a pharmaceutical company), with supporting reference URL(s).",
        parent=node,
        critical=True
    )
    org_name = _safe(ex.developer.org_name if ex.developer else None)
    nonprofit_sources = _combine_sources(
        ex.developer.org_urls if ex.developer else [],
        ex.developer.nonprofit_status_urls if ex.developer else []
    )
    nonprofit_claim = f"The developer organization is '{org_name}', and it is a non-profit organization."
    await evaluator.verify(
        claim=nonprofit_claim,
        node=nonprofit_leaf,
        sources=nonprofit_sources,
        additional_instruction=(
            "Confirm from the cited sources that the organization is a non-profit (e.g., 501(c)(3) status or explicit 'non-profit' designation). "
            "It should not be a for-profit pharmaceutical company."
        )
    )

    # First FDA-approved gene therapy from a non-profit organization
    first_nonprofit_leaf = evaluator.add_leaf(
        id="First_Nonprofit_Gene_Therapy_Claim_With_URL",
        desc="Provide a supported claim (with URL) that this is the first FDA-approved gene therapy from a non-profit organization, as required by the question.",
        parent=node,
        critical=True
    )
    first_nonprofit_claim_text = _safe(ex.first_nonprofit_claim.claim if ex.first_nonprofit_claim else None)
    first_nonprofit_sources = ex.first_nonprofit_claim.urls if ex.first_nonprofit_claim else []
    first_nonprofit_claim = (
        first_nonprofit_claim_text or "This therapy is the first FDA-approved gene therapy from a non-profit organization."
    )
    await evaluator.verify(
        claim=first_nonprofit_claim,
        node=first_nonprofit_leaf,
        sources=first_nonprofit_sources,
        additional_instruction=(
            "Confirm that the cited sources support the 'first from a non-profit' claim. "
            "The source should explicitly position this therapy as the first such FDA-approved gene therapy from a non-profit developer."
        )
    )

    # Autologous HSC-based therapy
    hsc_leaf = evaluator.add_leaf(
        id="Autologous_HSC_Based_With_URL",
        desc="Confirm the therapy is an autologous hematopoietic stem cell-based therapy, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    hsc_sources = ex.autologous_hsc.urls if ex.autologous_hsc else []
    hsc_claim = "This therapy is an autologous hematopoietic stem cell-based gene therapy."
    await evaluator.verify(
        claim=hsc_claim,
        node=hsc_leaf,
        sources=hsc_sources,
        additional_instruction=(
            "Check for language such as 'autologous hematopoietic stem cells', 'autologous HSC', 'ex vivo transduced autologous CD34+ cells', "
            "or equivalent phrasing indicating autologous HSC-based nature."
        )
    )

    # Lentiviral vector usage
    lenti_leaf = evaluator.add_leaf(
        id="Lentiviral_Vector_Used_With_URL",
        desc="Confirm the therapy uses a lentiviral vector for gene delivery, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    lenti_sources = ex.lentiviral_vector.urls if ex.lentiviral_vector else []
    lenti_claim = "This therapy uses a lentiviral vector for gene delivery."
    await evaluator.verify(
        claim=lenti_claim,
        node=lenti_leaf,
        sources=lenti_sources,
        additional_instruction=(
            "Look for explicit mention of 'lentiviral vector', 'LVV', or similar terms in authoritative documents."
        )
    )

    # Orphan drug designation
    orphan_leaf = evaluator.add_leaf(
        id="Orphan_Drug_Designation_For_Indication_With_URL",
        desc="Confirm the therapy/indication has orphan drug designation for a rare disease, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    orphan_sources = ex.orphan_designation.urls if ex.orphan_designation else []
    orphan_claim = "The therapy or the indication has received orphan drug designation for a rare disease."
    await evaluator.verify(
        claim=orphan_claim,
        node=orphan_leaf,
        sources=orphan_sources,
        additional_instruction=(
            "Verify orphan drug designation from FDA, EMA, or other authoritative regulatory sources."
        )
    )

    # Clinical trial center
    trial_center_leaf = evaluator.add_leaf(
        id="Clinical_Trial_Medical_Center_Name_With_URL",
        desc="Provide the specific named medical center/institution where the clinical trials were conducted, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    center_name = _safe(ex.trial_center.center_name if ex.trial_center else None)
    trial_center_sources = ex.trial_center.urls if ex.trial_center else []
    trial_center_claim = f"Clinical trials for this therapy were conducted at '{center_name}'."
    await evaluator.verify(
        claim=trial_center_claim,
        node=trial_center_leaf,
        sources=trial_center_sources,
        additional_instruction=(
            "Confirm that the cited sources explicitly name the clinical trial site(s)/medical center(s) for the therapy."
        )
    )

    # Approval announcement includes trial location
    announcement_leaf = evaluator.add_leaf(
        id="Approval_Announcement_Includes_Trial_Location_With_URL",
        desc="Provide a reference URL to an FDA approval announcement (or equivalent official approval communication) that includes clinical trial location information.",
        parent=node,
        critical=True
    )
    announcement_sources = ex.approval_announcement_trial_location.urls if ex.approval_announcement_trial_location else []
    announcement_claim = "The FDA approval announcement (or equivalent official communication) includes clinical trial location information."
    await evaluator.verify(
        claim=announcement_claim,
        node=announcement_leaf,
        sources=announcement_sources,
        additional_instruction=(
            "Prefer official FDA news releases, approval letters, labeling documents, or credible official communications that include trial location details."
        )
    )

    # First gene therapy for the indication
    first_indication_leaf = evaluator.add_leaf(
        id="First_Gene_Therapy_For_Indication_Claim_With_URL",
        desc="Provide a supported claim (with URL) that this is the first FDA-approved gene therapy for the specified indication.",
        parent=node,
        critical=True
    )
    first_indication_claim_text = _safe(ex.first_for_indication_claim.claim if ex.first_for_indication_claim else None)
    first_indication_sources = ex.first_for_indication_claim.urls if ex.first_for_indication_claim else []
    indication_text = _safe(ex.approved_indication.indication if ex.approved_indication else None)
    first_indication_claim = (
        first_indication_claim_text
        or f"This is the first FDA-approved gene therapy for the indication '{indication_text}'."
    )
    await evaluator.verify(
        claim=first_indication_claim,
        node=first_indication_leaf,
        sources=first_indication_sources,
        additional_instruction=(
            "Confirm that the cited sources explicitly state this therapy is the first FDA-approved gene therapy for the specific indication."
        )
    )

    # Patient eligibility criteria: min age and HLA donor conditions
    eligibility_leaf = evaluator.add_leaf(
        id="Patient_Eligibility_Criteria_Including_Min_Age_And_HLA_Donor_Condition_With_URL",
        desc="Provide the patient eligibility criteria as stated in an authoritative source (e.g., FDA label/approval documentation), explicitly including (a) minimum age requirement(s) and (b) conditions regarding HLA-matched donor availability, with at least one supporting reference URL.",
        parent=node,
        critical=True
    )
    min_age = _safe(ex.patient_eligibility.min_age if ex.patient_eligibility else None)
    hla_cond = _safe(ex.patient_eligibility.hla_donor_condition if ex.patient_eligibility else None)
    eligibility_sources = ex.patient_eligibility.urls if ex.patient_eligibility else []
    eligibility_claim = (
        f"The patient eligibility criteria include a minimum age requirement of '{min_age}' and the following conditions "
        f"regarding HLA-matched donor availability: '{hla_cond}'."
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_leaf,
        sources=eligibility_sources,
        additional_instruction=(
            "Use authoritative sources such as FDA labels, approval letters, or official healthcare guidance. "
            "Ensure both minimum age and HLA-matched donor conditions are explicitly present."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    """
    Evaluate an answer for the 'first non-profit gene therapy FDA approval in December 2025' task.
    """
    # Initialize evaluator with a neutral root
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

    # Extract structured information from the answer
    ex: TherapyExtraction = await evaluator.extract(
        prompt=prompt_extract_therapy(),
        template_class=TherapyExtraction,
        extraction_name="therapy_extraction",
    )

    # Build main rubric subtree (critical sequential)
    main_node = evaluator.add_sequential(
        id="Gene_Therapy_Complete_Identification",
        desc="Identify the first gene therapy from a non-profit organization that received FDA approval in December 2025, and provide all required details with supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Subtree: Therapy Identification (critical parallel)
    await build_therapy_identification(evaluator, main_node, ex)

    # Subtree: Constraints and Details (critical parallel)
    await build_constraints_and_details(evaluator, main_node, ex)

    # Return final summary
    return evaluator.get_summary()