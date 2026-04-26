import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "anktiva_2024_fda_il15"
TASK_DESCRIPTION = (
    "In April 2024, the FDA approved the first-in-class IL-15 receptor agonist for the treatment of "
    "BCG-unresponsive non-muscle invasive bladder cancer with carcinoma in situ. Provide comprehensive information "
    "about this drug including: (1) The brand name and generic name of the drug, (2) The drug classification, "
    "(3) The name of the pharmaceutical company that manufactures it and the city and state where the company's "
    "headquarters is located, (4) The exact FDA approval date and whether it received any special FDA designations, "
    "(5) The precise medical indication for which it was approved, (6) The clinical trial identifier (NCT number) "
    "and name of the pivotal trial that led to its approval, (7) The number of evaluable patients in the pivotal "
    "trial and the complete response rate achieved, (8) The route of administration and the maximum duration of "
    "maintenance therapy. For each piece of information, provide a reference URL from an authoritative source that "
    "supports your answer."
)

# Ground truth expectations (used for value checks against the answer content)
EXPECTED = {
    "brand_name": "ANKTIVA",
    "generic_name": "nogapendekin alfa inbakicept-pmln",
    "drug_class_variants": [
        "il-15 receptor agonist",
        "il 15 receptor agonist",
        "il-15 superagonist",
        "il 15 superagonist",
        "interleukin-15 receptor agonist",
        "interleukin 15 receptor agonist",
    ],
    "manufacturer": ["immunitybio, inc.", "immunitybio"],  # allow both with/without ", Inc."
    "hq_city_state": ["culver city, california", "culver city, ca"],
    "approval_date": "april 22, 2024",
    "breakthrough_designation": True,
    "indication_variants": [
        "bcg-unresponsive non-muscle invasive bladder cancer with carcinoma in situ",
        "bcg unresponsive non muscle invasive bladder cancer with carcinoma in situ",
        "bcg-unresponsive nmibc with cis",
        "bcg unresponsive nmibc with cis",
    ],
    "trial_identifier": "nct03022825",
    "trial_name": ["quilt-3.032", "quilt 3.032", "quilt 3.032 study", "quilt-3.032 study"],
    "evaluable_patients": "77",
    "complete_response_rate": ["62%", "62 percent", "0.62"],
    "administration_route_variants": [
        "intravesical in combination with bcg",
        "intravesical with bcg",
        "intravesically with bcg",
        "intravesical administration with bcg",
    ],
    "max_maintenance_duration_variants": [
        "37 months",
        "thirty-seven months",
        "up to 37 months",
        "maximum of 37 months",
    ],
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DrugExtraction(BaseModel):
    # Drug identification
    brand_name: Optional[str] = None
    brand_url: Optional[str] = None
    generic_name: Optional[str] = None
    generic_url: Optional[str] = None
    drug_class: Optional[str] = None
    drug_class_url: Optional[str] = None

    # Manufacturer and HQ
    manufacturer_name: Optional[str] = None
    manufacturer_url: Optional[str] = None
    hq_city_state: Optional[str] = None
    hq_url: Optional[str] = None

    # FDA approval details
    approval_date: Optional[str] = None
    approval_url: Optional[str] = None
    breakthrough_designation: Optional[str] = None
    breakthrough_url: Optional[str] = None
    indication: Optional[str] = None
    indication_url: Optional[str] = None

    # Clinical trial details
    trial_identifier: Optional[str] = None
    trial_identifier_url: Optional[str] = None
    trial_name: Optional[str] = None
    trial_name_url: Optional[str] = None
    evaluable_patients: Optional[str] = None
    evaluable_patients_url: Optional[str] = None
    complete_response_rate: Optional[str] = None
    cr_rate_url: Optional[str] = None

    # Treatment specifications
    administration_route: Optional[str] = None
    administration_url: Optional[str] = None
    max_maintenance_duration: Optional[str] = None
    maintenance_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return """
    Extract the requested information exactly as stated in the provided answer text. For each requested attribute, also extract one explicit reference URL that the answer cites for that attribute. If multiple URLs are cited for an attribute, select the most directly supportive and authoritative (e.g., FDA, official label, ClinicalTrials.gov, peer-reviewed journal) single URL. If any attribute or its URL is missing, set it to null.

    Fields to extract (use these exact JSON keys):
    - brand_name: The brand (proprietary) name of the drug (e.g., "ANKTIVA").
    - brand_url: A single URL cited to support the brand name.
    - generic_name: The generic (nonproprietary) name (e.g., "nogapendekin alfa inbakicept-pmln").
    - generic_url: A single URL cited to support the generic name.
    - drug_class: The drug classification (e.g., "IL-15 receptor agonist" or "IL-15 superagonist").
    - drug_class_url: A single URL cited to support the drug class.

    - manufacturer_name: The name of the manufacturer (e.g., "ImmunityBio, Inc.").
    - manufacturer_url: A single URL cited to support the manufacturer of the drug.
    - hq_city_state: Headquarters city and state of the manufacturer as a single string (e.g., "Culver City, California").
    - hq_url: A single URL cited to support the HQ location.

    - approval_date: The exact FDA approval date for this indication (e.g., "April 22, 2024").
    - approval_url: A single URL cited to support the approval date.
    - breakthrough_designation: Whether the drug has FDA Breakthrough Therapy designation (e.g., "Breakthrough Therapy").
    - breakthrough_url: A single URL cited to support the designation status.
    - indication: The precise medical indication the FDA approved (e.g., "BCG-unresponsive non-muscle invasive bladder cancer with carcinoma in situ").
    - indication_url: A single URL cited to support the indication.

    - trial_identifier: The pivotal clinical trial identifier (e.g., "NCT03022825").
    - trial_identifier_url: A single URL cited to support the trial identifier.
    - trial_name: The pivotal trial name (e.g., "QUILT-3.032").
    - trial_name_url: A single URL cited to support the trial name.
    - evaluable_patients: The number of evaluable patients in the pivotal analysis (as provided in the answer, keep as string).
    - evaluable_patients_url: A single URL cited to support the number of evaluable patients.
    - complete_response_rate: The complete response (CR) rate achieved (keep formatting like "62%").
    - cr_rate_url: A single URL cited to support the CR rate.

    - administration_route: The route of administration and context (e.g., "intravesical in combination with BCG").
    - administration_url: A single URL cited to support the administration route.
    - max_maintenance_duration: The maximum duration of maintenance therapy (e.g., "37 months").
    - maintenance_url: A single URL cited to support the maintenance duration.

    Special URL rules:
    - Extract only explicit URLs from the answer. If only a site is mentioned without a URL, set the corresponding URL field to null.
    - If a URL is missing "http://" or "https://", prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper functions for normalization and checks                               #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def _contains_variant(value: Optional[str], variants: List[str]) -> bool:
    v = _normalize_text(value)
    for cand in variants:
        if _normalize_text(cand) in v:
            return True
    return False


def _equals_any(value: Optional[str], variants: List[str]) -> bool:
    v = _normalize_text(value)
    return any(v == _normalize_text(c) for c in variants)


def _has_digits(value: Optional[str], digits: str) -> bool:
    if not value:
        return False
    return digits in re.sub(r"[^\d]", "", value)


def _contains_number_or_percent(value: Optional[str], targets: List[str]) -> bool:
    """Allow formats like '62%', '62 percent', '0.62' etc."""
    if not value:
        return False
    v = _normalize_text(value)
    for t in targets:
        tn = _normalize_text(t)
        if tn.endswith("%"):
            num = re.sub(r"[^0-9]", "", tn)
            if num and num in re.sub(r"[^0-9]", "", v):
                return True
        else:
            if tn in v:
                return True
    return False


def _nonempty_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_value_and_reference_checks(
    evaluator: Evaluator,
    parent,
    id_base: str,
    wrapper_description: str,
    value_check_desc: str,
    value_check_result: bool,
    source_url: Optional[str],
    source_presence_desc: str,
    support_leaf_desc: str,
    support_claim: str,
    support_additional_instruction: str,
) -> None:
    """
    Create a sequential wrapper node holding three critical leaves:
    1) Value-correct custom check
    2) Source URL provided custom check
    3) Verification by URL that the claim is supported by the source
    """
    wrapper = evaluator.add_sequential(
        id=id_base,
        desc=wrapper_description,
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=value_check_result,
        id=f"{id_base}_value_correct",
        desc=value_check_desc,
        parent=wrapper,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_url(source_url),
        id=f"{id_base}_source_provided",
        desc=source_presence_desc,
        parent=wrapper,
        critical=True,
    )

    support_node = evaluator.add_leaf(
        id=f"{id_base}_supported_by_source",
        desc=support_leaf_desc,
        parent=wrapper,
        critical=True,
    )
    # This verify will auto-skip if previous custom nodes failed (due to preconditions)
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=source_url,
        additional_instruction=support_additional_instruction,
    )


# ---------------------------- Category verifications ----------------------- #
async def verify_drug_identification(evaluator: Evaluator, parent, ex: DrugExtraction) -> None:
    cat = evaluator.add_parallel(
        id="DrugIdentification",
        desc="Correct drug identification details, each supported by an authoritative reference URL.",
        parent=parent,
        critical=True,
    )

    # Brand name: must be ANKTIVA
    brand_value_ok = _equals_any(ex.brand_name, [EXPECTED["brand_name"]])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="BrandNameWithReference",
        wrapper_description="Provides brand name and authoritative reference.",
        value_check_desc=f"Answer provides brand name as '{EXPECTED['brand_name']}'",
        value_check_result=brand_value_ok,
        source_url=ex.brand_url,
        source_presence_desc="Brand name reference URL is provided",
        support_leaf_desc="Brand name is supported by the cited source",
        support_claim=f"The brand (proprietary) name of the drug is {EXPECTED['brand_name']}.",
        support_additional_instruction="Verify that the page explicitly mentions the brand name for the approved drug. Allow case-insensitive matching.",
    )

    # Generic name
    generic_value_ok = _equals_any(ex.generic_name, [EXPECTED["generic_name"]])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="GenericNameWithReference",
        wrapper_description="Provides generic name and authoritative reference.",
        value_check_desc=f"Answer provides generic name as '{EXPECTED['generic_name']}'",
        value_check_result=generic_value_ok,
        source_url=ex.generic_url,
        source_presence_desc="Generic name reference URL is provided",
        support_leaf_desc="Generic name is supported by the cited source",
        support_claim=f"The generic (nonproprietary) name of ANKTIVA is {EXPECTED['generic_name']}.",
        support_additional_instruction="Verify that the page explicitly states the generic (nonproprietary) name corresponding to ANKTIVA. Allow minor formatting and case variations.",
    )

    # Drug class (IL-15 receptor agonist / IL-15 superagonist)
    drug_class_ok = _contains_variant(ex.drug_class, EXPECTED["drug_class_variants"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="DrugClassWithReference",
        wrapper_description="States drug class and authoritative reference.",
        value_check_desc="Answer states drug class as IL-15 receptor agonist or IL-15 superagonist",
        value_check_result=drug_class_ok,
        source_url=ex.drug_class_url,
        source_presence_desc="Drug class reference URL is provided",
        support_leaf_desc="Drug class is supported by the cited source",
        support_claim="The drug ANKTIVA (nogapendekin alfa inbakicept-pmln) is an IL-15 receptor agonist (also described as an IL-15 superagonist).",
        support_additional_instruction="Verify that the page describes the drug as an IL-15 receptor agonist or IL-15 superagonist. Allow synonyms and minor variations.",
    )


async def verify_manufacturer_information(evaluator: Evaluator, parent, ex: DrugExtraction) -> None:
    cat = evaluator.add_parallel(
        id="ManufacturerInformation",
        desc="Manufacturer and headquarters information, each supported by an authoritative reference URL.",
        parent=parent,
        critical=True,
    )

    # Manufacturer name: ImmunityBio, Inc. (allow 'ImmunityBio')
    manuf_ok = _equals_any(ex.manufacturer_name, EXPECTED["manufacturer"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="ManufacturerNameWithReference",
        wrapper_description="Identifies manufacturer and authoritative reference.",
        value_check_desc="Answer identifies manufacturer as ImmunityBio (ImmunityBio, Inc.)",
        value_check_result=manuf_ok,
        source_url=ex.manufacturer_url,
        source_presence_desc="Manufacturer reference URL is provided",
        support_leaf_desc="Manufacturer is supported by the cited source",
        support_claim="ANKTIVA (nogapendekin alfa inbakicept-pmln) is manufactured by ImmunityBio, Inc.",
        support_additional_instruction="Verify that the page explicitly identifies ImmunityBio as the manufacturer of ANKTIVA (nogapendekin alfa inbakicept-pmln). Allow minor naming variants (with or without ', Inc.').",
    )

    # HQ location: Culver City, California (allow 'CA')
    hq_ok = _contains_variant(ex.hq_city_state, EXPECTED["hq_city_state"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="HeadquartersLocationWithReference",
        wrapper_description="States headquarters location and authoritative reference.",
        value_check_desc="Answer states headquarters location as Culver City, California",
        value_check_result=hq_ok,
        source_url=ex.hq_url,
        source_presence_desc="Headquarters location reference URL is provided",
        support_leaf_desc="Headquarters location is supported by the cited source",
        support_claim="ImmunityBio is headquartered in Culver City, California.",
        support_additional_instruction="Verify that the page states the company's headquarters is in Culver City, California (Culver City, CA acceptable).",
    )


async def verify_fda_approval_details(evaluator: Evaluator, parent, ex: DrugExtraction) -> None:
    cat = evaluator.add_parallel(
        id="FDAApprovalDetails",
        desc="FDA approval details (date, designation, indication), each supported by an authoritative reference URL.",
        parent=parent,
        critical=True,
    )

    # Approval date: April 22, 2024
    date_ok = _normalize_text(ex.approval_date) == EXPECTED["approval_date"]
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="ApprovalDateWithReference",
        wrapper_description="States the exact FDA approval date and authoritative reference.",
        value_check_desc="Answer states the FDA approval date as April 22, 2024",
        value_check_result=date_ok,
        source_url=ex.approval_url,
        source_presence_desc="Approval date reference URL is provided",
        support_leaf_desc="Approval date is supported by the cited source",
        support_claim="The FDA approved ANKTIVA on April 22, 2024.",
        support_additional_instruction="Verify that the page explicitly states the FDA approval date for ANKTIVA as April 22, 2024.",
    )

    # Breakthrough Therapy designation
    # Accept if the extracted text mentions 'breakthrough' (case-insensitive)
    breakthrough_ok = "breakthrough" in _normalize_text(ex.breakthrough_designation or "")
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="BreakthroughDesignationWithReference",
        wrapper_description="States Breakthrough Therapy designation and authoritative reference.",
        value_check_desc="Answer states the drug received FDA Breakthrough Therapy designation",
        value_check_result=breakthrough_ok,
        source_url=ex.breakthrough_url,
        source_presence_desc="Designation reference URL is provided",
        support_leaf_desc="Breakthrough designation is supported by the cited source",
        support_claim="ANKTIVA received FDA Breakthrough Therapy designation.",
        support_additional_instruction="Verify that the page explicitly states the drug received FDA Breakthrough Therapy designation.",
    )

    # Indication: BCG-unresponsive NMIBC with CIS
    indication_ok = _contains_variant(ex.indication, EXPECTED["indication_variants"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="IndicationWithReference",
        wrapper_description="States approved indication and authoritative reference.",
        value_check_desc="Answer states the approved indication as BCG-unresponsive NMIBC with carcinoma in situ (CIS)",
        value_check_result=indication_ok,
        source_url=ex.indication_url,
        source_presence_desc="Indication reference URL is provided",
        support_leaf_desc="Approved indication is supported by the cited source",
        support_claim="ANKTIVA was approved for BCG-unresponsive non-muscle invasive bladder cancer with carcinoma in situ (CIS).",
        support_additional_instruction="Verify that the page explicitly states the FDA-approved indication includes BCG-unresponsive NMIBC with CIS.",
    )


async def verify_clinical_trial_information(evaluator: Evaluator, parent, ex: DrugExtraction) -> None:
    cat = evaluator.add_parallel(
        id="ClinicalTrialInformation",
        desc="Pivotal trial details and outcomes, each supported by an authoritative reference URL.",
        parent=parent,
        critical=True,
    )

    # Trial Identifier: NCT03022825
    nct_ok = _normalize_text(ex.trial_identifier) == EXPECTED["trial_identifier"]
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="TrialIdentifierWithReference",
        wrapper_description="Provides pivotal clinical trial identifier and authoritative reference.",
        value_check_desc="Answer provides pivotal trial identifier as NCT03022825",
        value_check_result=nct_ok,
        source_url=ex.trial_identifier_url,
        source_presence_desc="Trial identifier reference URL is provided",
        support_leaf_desc="Trial identifier is supported by the cited source",
        support_claim="The pivotal clinical trial that supported approval has identifier NCT03022825.",
        support_additional_instruction="Verify that the page shows NCT03022825 as the identifier of the pivotal trial underpinning the approval.",
    )

    # Trial Name: QUILT-3.032 (allow minor variants)
    trial_name_ok = _contains_variant(ex.trial_name, EXPECTED["trial_name"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="TrialNameWithReference",
        wrapper_description="Provides pivotal trial name and authoritative reference.",
        value_check_desc="Answer provides pivotal trial name as QUILT-3.032",
        value_check_result=trial_name_ok,
        source_url=ex.trial_name_url,
        source_presence_desc="Trial name reference URL is provided",
        support_leaf_desc="Trial name is supported by the cited source",
        support_claim="The pivotal clinical trial name is QUILT-3.032.",
        support_additional_instruction="Verify that the page explicitly names the pivotal trial as QUILT-3.032 (minor formatting variations acceptable).",
    )

    # Evaluable patients: 77
    pts_ok = _has_digits(ex.evaluable_patients, re.sub(r"[^\d]", "", EXPECTED["evaluable_patients"]))
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="EvaluablePatientsWithReference",
        wrapper_description="States number of evaluable patients and authoritative reference.",
        value_check_desc="Answer states the number of evaluable patients as 77",
        value_check_result=pts_ok,
        source_url=ex.evaluable_patients_url,
        source_presence_desc="Evaluable patients reference URL is provided",
        support_leaf_desc="Number of evaluable patients is supported by the cited source",
        support_claim="In the pivotal analysis supporting approval, there were 77 evaluable patients.",
        support_additional_instruction="Verify that the page states there were 77 evaluable patients in the pivotal analysis supporting approval.",
    )

    # Complete Response (CR) rate: 62%
    cr_ok = _contains_number_or_percent(ex.complete_response_rate, EXPECTED["complete_response_rate"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="CompleteResponseRateWithReference",
        wrapper_description="States complete response rate and authoritative reference.",
        value_check_desc="Answer states the complete response rate as 62%",
        value_check_result=cr_ok,
        source_url=ex.cr_rate_url,
        source_presence_desc="CR rate reference URL is provided",
        support_leaf_desc="Complete response rate is supported by the cited source",
        support_claim="The complete response (CR) rate achieved in the pivotal analysis was 62%.",
        support_additional_instruction="Verify that the page reports a 62% complete response rate for the pivotal analysis or CIS cohort referenced for approval.",
    )


async def verify_treatment_specifications(evaluator: Evaluator, parent, ex: DrugExtraction) -> None:
    cat = evaluator.add_parallel(
        id="TreatmentSpecifications",
        desc="Administration route and maximum maintenance duration, each supported by an authoritative reference URL.",
        parent=parent,
        critical=True,
    )

    # Administration: intravesical in combination with BCG
    admin_ok = _contains_variant(ex.administration_route, EXPECTED["administration_route_variants"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="AdministrationRouteWithReference",
        wrapper_description="States administration route and authoritative reference.",
        value_check_desc="Answer states the administration route as intravesical in combination with BCG",
        value_check_result=admin_ok,
        source_url=ex.administration_url,
        source_presence_desc="Administration route reference URL is provided",
        support_leaf_desc="Administration route is supported by the cited source",
        support_claim="ANKTIVA is administered intravesically in combination with BCG.",
        support_additional_instruction="Verify that the page explicitly states intravesical administration with BCG.",
    )

    # Maintenance duration: 37 months (max)
    duration_ok = _contains_variant(ex.max_maintenance_duration, EXPECTED["max_maintenance_duration_variants"])
    await _build_value_and_reference_checks(
        evaluator=evaluator,
        parent=cat,
        id_base="MaximumMaintenanceDurationWithReference",
        wrapper_description="States maximum duration of maintenance therapy and authoritative reference.",
        value_check_desc="Answer states the maximum duration of maintenance therapy as 37 months",
        value_check_result=duration_ok,
        source_url=ex.maintenance_url,
        source_presence_desc="Maintenance duration reference URL is provided",
        support_leaf_desc="Maximum maintenance duration is supported by the cited source",
        support_claim="The maximum recommended duration of maintenance therapy is 37 months.",
        support_additional_instruction="Verify that the page states a maximum maintenance duration of 37 months (allow phrasing like 'up to 37 months').",
    )


# --------------------------------------------------------------------------- #
# Top-level verification assembly                                             #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, ex: DrugExtraction) -> None:
    # Top-level critical node (matches rubric's AnktivaInformation)
    top = evaluator.add_parallel(
        id="AnktivaInformation",
        desc="Complete and accurate information (with authoritative reference URLs for each requested attribute) about the first-in-class IL-15 receptor agonist approved by FDA in April 2024 for BCG-unresponsive NMIBC with CIS.",
        parent=evaluator.root,
        critical=True,
    )

    # Subsections
    await verify_drug_identification(evaluator, top, ex)
    await verify_manufacturer_information(evaluator, top, ex)
    await verify_fda_approval_details(evaluator, top, ex)
    await verify_clinical_trial_information(evaluator, top, ex)
    await verify_treatment_specifications(evaluator, top, ex)


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
    """
    Evaluate an answer for the ANKTIVA (IL-15 receptor agonist) FDA April 2024 approval information task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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
    ex: DrugExtraction = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugExtraction,
        extraction_name="drug_info_extraction",
    )

    # Record ground truth for reference in the summary
    evaluator.add_ground_truth(
        {
            "expected": {
                "brand_name": EXPECTED["brand_name"],
                "generic_name": EXPECTED["generic_name"],
                "drug_class_examples": list(set(EXPECTED["drug_class_variants"])),
                "manufacturer": EXPECTED["manufacturer"],
                "hq_city_state": EXPECTED["hq_city_state"],
                "approval_date": EXPECTED["approval_date"],
                "breakthrough_designation": EXPECTED["breakthrough_designation"],
                "indication_examples": list(set(EXPECTED["indication_variants"])),
                "trial_identifier": EXPECTED["trial_identifier"],
                "trial_name_examples": list(set(EXPECTED["trial_name"])),
                "evaluable_patients": EXPECTED["evaluable_patients"],
                "complete_response_rate_examples": list(set(EXPECTED["complete_response_rate"])),
                "administration_route_examples": list(set(EXPECTED["administration_route_variants"])),
                "max_maintenance_duration_examples": list(set(EXPECTED["max_maintenance_duration_variants"])),
            }
        },
        gt_type="ground_truth",
    )

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, ex)

    # Return evaluation summary
    return evaluator.get_summary()