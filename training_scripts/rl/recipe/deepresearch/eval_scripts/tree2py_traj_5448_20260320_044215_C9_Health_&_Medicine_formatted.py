import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_2025_orphan_breakthrough_first"
TASK_DESCRIPTION = """
Identify a drug that was approved by the FDA in 2025 and meets all of the following criteria:

1. The drug received FDA Breakthrough Therapy designation
2. The drug received FDA Orphan Drug designation
3. The drug is the first FDA-approved treatment for its specific indication
4. The target disease affects fewer than 200,000 people in the United States

For the identified drug, provide:
- The trade (brand) name
- The generic name
- The specific disease or condition it treats
- The pharmaceutical company that manufactures/sponsors it
- The FDA approval date in 2025
- Documentation confirming it received both Breakthrough Therapy and Orphan Drug designations
- Evidence that it is the first approved treatment for its indication
- Prevalence data showing the disease affects fewer than 200,000 people in the U.S.

Additionally, if available, include:
- Any other FDA designations it received (e.g., Priority Review, Rare Pediatric Disease designation, Fast Track)
- The type of approval (full approval or accelerated approval)
- Clinical trial information including ClinicalTrials.gov NCT numbers if applicable
- Key disease characteristics and affected patient populations
- Primary efficacy outcomes from clinical trials

All information must be supported by verifiable URLs from official sources such as FDA.gov, pharmaceutical company press releases, medical journals, or other credible medical information sources.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DrugExtraction(BaseModel):
    # Core identity
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    indication: Optional[str] = None
    company: Optional[str] = None

    # FDA approval (2025)
    approval_date: Optional[str] = None
    approval_urls: List[str] = Field(default_factory=list)
    novel_drug_urls: List[str] = Field(default_factory=list)  # CDER Novel Drugs 2025 or equivalent

    # Designations (required)
    btd_urls: List[str] = Field(default_factory=list)  # Breakthrough Therapy designation sources
    odd_urls: List[str] = Field(default_factory=list)  # Orphan Drug designation sources

    # First treatment status (required)
    first_treatment_urls: List[str] = Field(default_factory=list)

    # Prevalence (<200k in U.S.) (required)
    prevalence_statement: Optional[str] = None
    prevalence_number: Optional[str] = None  # keep as string for robustness
    prevalence_urls: List[str] = Field(default_factory=list)

    # Additional FDA designations (optional)
    additional_designations: List[str] = Field(default_factory=list)  # e.g., ["Priority Review", "Fast Track"]
    additional_designations_urls: List[str] = Field(default_factory=list)

    # Approval type and basis (optional)
    approval_type: Optional[str] = None  # e.g., "Accelerated approval" or "Traditional approval"
    approval_type_urls: List[str] = Field(default_factory=list)
    efficacy_endpoints: List[str] = Field(default_factory=list)
    efficacy_urls: List[str] = Field(default_factory=list)

    # Clinical trials (optional)
    nct_numbers: List[str] = Field(default_factory=list)
    trial_phase: Optional[str] = None
    clinical_trials_urls: List[str] = Field(default_factory=list)

    # Therapeutic area (optional)
    therapeutic_area: Optional[str] = None
    disease_category: Optional[str] = None

    # Disease info (optional but useful)
    disease_info_urls: List[str] = Field(default_factory=list)

    # Manufacturer / naming sources
    manufacturer_urls: List[str] = Field(default_factory=list)
    name_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return """
Extract the following information exactly as stated in the answer. Do not infer or invent missing items. If an item is missing, set null (for strings) or an empty list (for arrays).

Core identity:
- brand_name: FDA-approved trade/brand name of the drug
- generic_name: generic (nonproprietary) name
- indication: the specific disease/condition
- company: sponsor/manufacturer (organization name)

FDA approval (2025):
- approval_date: the explicit FDA approval date string as given in the answer
- approval_urls: all URLs cited that confirm FDA approval (FDA.gov preferred, may include company press release if cited)
- novel_drug_urls: URLs showing the drug is novel (e.g., CDER Novel Drugs 2025 list) if cited

Designations (required):
- btd_urls: URLs confirming Breakthrough Therapy designation (FDA or company press release)
- odd_urls: URLs confirming Orphan Drug designation (FDA or credible sources)

First treatment status (required):
- first_treatment_urls: URLs confirming first-in-class or first FDA-approved treatment for this indication

Prevalence (<200k in U.S.) (required):
- prevalence_statement: the sentence or phrase about U.S. prevalence
- prevalence_number: the numeric/statistical prevalence string if provided (keep as text, e.g., "fewer than 200,000")
- prevalence_urls: URLs that provide the prevalence information

Additional FDA designations (optional):
- additional_designations: a list of other designations explicitly claimed in the answer (e.g., "Priority Review", "Fast Track", "Rare Pediatric Disease")
- additional_designations_urls: URLs that support these additional designations

Approval type and basis (optional):
- approval_type: "Accelerated approval", "Traditional approval" (full), or similar if stated
- approval_type_urls: URLs that confirm approval type/classification
- efficacy_endpoints: list of primary efficacy endpoints/outcomes as stated in answer
- efficacy_urls: URLs supporting efficacy outcomes or FDA review summaries

Clinical trials (optional):
- nct_numbers: list of NCT identifiers mentioned (e.g., "NCT01234567")
- trial_phase: pivotal trial phase if provided (e.g., "Phase 2", "Phase 2/3")
- clinical_trials_urls: URLs to ClinicalTrials.gov or trial publications

Therapeutic area (optional):
- therapeutic_area: high-level area (e.g., "Oncology", "Neurology")
- disease_category: more specific category if stated

Disease info (optional):
- disease_info_urls: URLs providing disease characteristics (pathophysiology, population, symptoms)

Manufacturer / Naming sources:
- manufacturer_urls: URLs confirming sponsor/manufacturer role (company site or press release)
- name_sources: URLs that show trade and/or generic names (FDA or company)

SPECIAL RULES FOR URL FIELDS:
- Only include URLs explicitly present in the answer text.
- Include complete URLs (with http/https).
- If the answer mentions a source but does not include a URL, do not fabricate one; leave the URL list empty.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    result.append(uu)
    return result


def has_designation(desigs: List[str], key: str) -> bool:
    if not desigs:
        return False
    key_l = key.lower()
    return any(key_l in (d or "").lower() for d in desigs)


def drug_display(brand: Optional[str], generic: Optional[str]) -> str:
    if brand and generic:
        return f"{brand} ({generic})"
    return brand or generic or "the drug"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_fda_approval(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="FDA_2025_Approval",
        desc="Verify the drug received FDA approval in calendar year 2025",
        parent=parent,
        critical=True
    )

    # Approval date in 2025
    leaf_date = evaluator.add_leaf(
        id="Approval_Date_2025",
        desc="The FDA approval date falls between January 1, 2025 and December 31, 2025",
        parent=node,
        critical=True
    )
    claim_date = (
        f"FDA approved {drug_display(info.brand_name, info.generic_name)} for "
        f"{info.indication or 'its indication'} on {info.approval_date or 'an approval date'} "
        f"and this approval date is within calendar year 2025."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=combine_sources(info.approval_urls, info.manufacturer_urls),
        additional_instruction="Confirm the FDA approval date shown on the cited page is in 2025 (inclusive)."
    )

    # Novel drug status (novel drug classification)
    leaf_novel = evaluator.add_leaf(
        id="Novel_Drug_Status",
        desc="The drug is classified as a novel drug (never before approved or marketed in the U.S.)",
        parent=node,
        critical=True
    )
    claim_novel = (
        f"{drug_display(info.brand_name, info.generic_name)} is classified as a novel drug "
        f"(i.e., first-time FDA approval for its active moiety) in 2025."
    )
    await evaluator.verify(
        claim=claim_novel,
        node=leaf_novel,
        sources=combine_sources(info.novel_drug_urls, info.approval_urls),
        additional_instruction="Prefer FDA CDER 'Novel Drugs' 2025 or FDA documents that explicitly state 'novel drug'."
    )

    # FDA approval documentation URL support
    leaf_doc = evaluator.add_leaf(
        id="FDA_Approval_Documentation_URL",
        desc="Provide valid URL from FDA.gov or official FDA announcement confirming the 2025 approval",
        parent=node,
        critical=True
    )
    claim_doc = (
        f"The provided page(s) confirm the 2025 FDA approval of {drug_display(info.brand_name, info.generic_name)} "
        f"for {info.indication or 'its indicated use'}."
    )
    await evaluator.verify(
        claim=claim_doc,
        node=leaf_doc,
        sources=info.approval_urls,
        additional_instruction="At least one URL should be an FDA.gov page (e.g., Drugs@FDA, press release, approval letter) confirming the 2025 approval."
    )


async def build_breakthrough(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Breakthrough_Therapy_Designation",
        desc="Verify the drug received FDA Breakthrough Therapy designation",
        parent=parent,
        critical=True
    )

    # BTD status confirmed
    leaf_status = evaluator.add_leaf(
        id="BTD_Status_Confirmed",
        desc="The drug officially received Breakthrough Therapy designation from FDA",
        parent=node,
        critical=True
    )
    claim_btd = (
        f"FDA granted Breakthrough Therapy designation to {drug_display(info.brand_name, info.generic_name)} "
        f"for {info.indication or 'its indication'}."
    )
    await evaluator.verify(
        claim=claim_btd,
        node=leaf_status,
        sources=info.btd_urls,
        additional_instruction="Look for explicit mention of 'Breakthrough Therapy designation' granted by FDA."
    )

    # BTD criteria justification (critical aggregate)
    crit_node = evaluator.add_parallel(
        id="BTD_Criteria_Justification",
        desc="Documentation shows the drug met BTD criteria: treats serious condition with preliminary clinical evidence of substantial improvement",
        parent=node,
        critical=True
    )

    # Serious condition treatment
    leaf_serious = evaluator.add_leaf(
        id="Serious_Condition_Treatment",
        desc="The drug treats a serious or life-threatening disease or condition",
        parent=crit_node,
        critical=True
    )
    claim_serious = (
        f"The target disease/condition ({info.indication or 'the indication'}) is serious or life-threatening."
    )
    await evaluator.verify(
        claim=claim_serious,
        node=leaf_serious,
        sources=combine_sources(info.btd_urls, info.disease_info_urls, info.approval_urls),
        additional_instruction="The page should characterize the disease as serious or life-threatening or of substantial morbidity/mortality."
    )

    # Substantial improvement evidence
    leaf_improve = evaluator.add_leaf(
        id="Substantial_Improvement_Evidence",
        desc="Preliminary clinical evidence indicates substantial improvement over existing therapies",
        parent=crit_node,
        critical=True
    )
    claim_improve = (
        "There is preliminary clinical evidence indicating substantial improvement over available therapies, "
        "supporting Breakthrough Therapy designation."
    )
    await evaluator.verify(
        claim=claim_improve,
        node=leaf_improve,
        sources=combine_sources(info.btd_urls, info.efficacy_urls, info.approval_urls),
        additional_instruction="Look for wording that explains why BTD was granted (e.g., substantial improvement vs existing therapy)."
    )

    # BTD documentation URL
    leaf_btd_url = evaluator.add_leaf(
        id="BTD_Documentation_URL",
        desc="Provide valid URL confirming Breakthrough Therapy designation",
        parent=node,
        critical=True
    )
    claim_btd_url = "The provided page(s) explicitly confirm FDA Breakthrough Therapy designation."
    await evaluator.verify(
        claim=claim_btd_url,
        node=leaf_btd_url,
        sources=info.btd_urls,
        additional_instruction="Prefer FDA pages; company press releases acceptable if they explicitly state BTD from FDA."
    )


async def build_orphan(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Orphan_Drug_Designation",
        desc="Verify the drug received FDA Orphan Drug designation",
        parent=parent,
        critical=True
    )

    # ODD status confirmed
    leaf_status = evaluator.add_leaf(
        id="ODD_Status_Confirmed",
        desc="The drug officially received Orphan Drug designation from FDA",
        parent=node,
        critical=True
    )
    claim_odd = (
        f"FDA granted Orphan Drug designation to {drug_display(info.brand_name, info.generic_name)} "
        f"for {info.indication or 'its indication'}."
    )
    await evaluator.verify(
        claim=claim_odd,
        node=leaf_status,
        sources=info.odd_urls,
        additional_instruction="Look for explicit statement of 'Orphan Drug designation' granted by FDA."
    )

    # Rare disease qualification (aggregate; both children set critical to satisfy framework constraints)
    rare_node = evaluator.add_parallel(
        id="Rare_Disease_Qualification",
        desc="The target disease meets orphan disease criteria (affects fewer than 200,000 people in the U.S.)",
        parent=node,
        critical=True
    )

    # Prevalence under 200,000
    leaf_prev = evaluator.add_leaf(
        id="US_Prevalence_Under_200000",
        desc="Documented evidence shows U.S. prevalence is below 200,000 affected individuals",
        parent=rare_node,
        critical=True
    )
    claim_prev = (
        f"The disease/condition {info.indication or ''} affects fewer than 200,000 people in the United States."
    )
    await evaluator.verify(
        claim=claim_prev,
        node=leaf_prev,
        sources=combine_sources(info.prevalence_urls, info.odd_urls),
        additional_instruction="The cited page should explicitly state U.S. prevalence under 200,000 or equivalent wording."
    )

    # Prevalence data source credibility (promoted to critical to meet engine constraints)
    leaf_prev_src = evaluator.add_leaf(
        id="Prevalence_Data_Source",
        desc="Prevalence data is from credible medical literature, FDA documents, or disease registry",
        parent=rare_node,
        critical=True
    )
    claim_prev_src = (
        "The prevalence information comes from a credible source such as FDA, NIH, CDC, a recognized disease registry, "
        "or peer‑reviewed medical literature and indicates fewer than 200,000 affected in the U.S."
    )
    await evaluator.verify(
        claim=claim_prev_src,
        node=leaf_prev_src,
        sources=info.prevalence_urls,
        additional_instruction="Assess credibility (domain, publisher) and confirm the content presents prevalence figures."
    )

    # ODD documentation URL
    leaf_odd_url = evaluator.add_leaf(
        id="ODD_Documentation_URL",
        desc="Provide valid URL confirming Orphan Drug designation",
        parent=node,
        critical=True
    )
    claim_odd_url = "The provided page(s) explicitly confirm FDA Orphan Drug designation."
    await evaluator.verify(
        claim=claim_odd_url,
        node=leaf_odd_url,
        sources=info.odd_urls,
        additional_instruction="Prefer FDA pages; company press releases acceptable if they explicitly state ODD from FDA."
    )


async def build_first_status(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="First_Treatment_Status",
        desc="Verify the drug is the first-in-class or first FDA-approved treatment for its specific indication",
        parent=parent,
        critical=True
    )

    # First approved indication
    leaf_first = evaluator.add_leaf(
        id="First_Approved_Indication",
        desc="The drug is documented as the first FDA-approved treatment for this specific disease/indication",
        parent=node,
        critical=True
    )
    claim_first = (
        f"{drug_display(info.brand_name, info.generic_name)} is the first FDA‑approved treatment "
        f"for {info.indication or 'this indication'}."
    )
    await evaluator.verify(
        claim=claim_first,
        node=leaf_first,
        sources=combine_sources(info.first_treatment_urls, info.approval_urls),
        additional_instruction="Look for explicit wording such as 'first FDA-approved treatment' or 'first-in-class'."
    )

    # Novel mechanism documentation (promoted to critical to meet engine constraints)
    leaf_mech = evaluator.add_leaf(
        id="Novel_Mechanism_Documentation",
        desc="If first-in-class, documentation describes the novel mechanism of action or therapeutic approach",
        parent=node,
        critical=True
    )
    claim_mech = (
        "Documentation describes a novel mechanism of action or therapeutic approach consistent with a first-in-class therapy."
    )
    await evaluator.verify(
        claim=claim_mech,
        node=leaf_mech,
        sources=combine_sources(info.first_treatment_urls, info.manufacturer_urls, info.approval_urls),
        additional_instruction="Confirm the page explains novelty of mechanism or first-in-class status."
    )

    # First treatment URL support
    leaf_url = evaluator.add_leaf(
        id="First_Treatment_URL",
        desc="Provide valid URL confirming first-in-class or first-approved status",
        parent=node,
        critical=True
    )
    claim_first_url = "The provided page(s) confirm the first-approved or first-in-class status."
    await evaluator.verify(
        claim=claim_first_url,
        node=leaf_url,
        sources=info.first_treatment_urls,
        additional_instruction="Prefer FDA pages or credible press releases; wording should clearly state first status."
    )


async def build_additional_designations(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Additional_FDA_Designations",
        desc="Document any additional FDA designations beyond Breakthrough Therapy and Orphan Drug",
        parent=parent,
        critical=False
    )

    # Helper to add a claimed gate + verification leaf, causing skip if not claimed
    async def add_optional_designation(pair_id: str, human_desc: str, keyword: str):
        gate = evaluator.add_custom_node(
            result=has_designation(info.additional_designations, keyword),
            id=f"{pair_id}_claimed",
            desc=f"{human_desc} is claimed in the answer",
            parent=node,
            critical=True  # critical to act as prerequisite; verification leaf will be skipped if False
        )
        leaf = evaluator.add_leaf(
            id=pair_id,
            desc=f"If applicable, the drug received {human_desc} designation",
            parent=node,
            critical=False
        )
        claim = f"The drug received {human_desc} designation."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=info.additional_designations_urls,
            additional_instruction=f"Confirm that the page explicitly states '{human_desc}'. If the prerequisite node failed, this check will be skipped."
        )

    await add_optional_designation("Priority_Review", "Priority Review", "priority review")
    await add_optional_designation("Rare_Pediatric_Disease", "Rare Pediatric Disease", "rare pediatric")
    await add_optional_designation("Fast_Track", "Fast Track", "fast track")

    # Any additional designation URL (if any were claimed)
    leaf_urls = evaluator.add_leaf(
        id="Additional_Designations_URL",
        desc="Provide URL documenting any additional designations",
        parent=node,
        critical=False
    )
    claim_urls = "The provided page(s) document any additional FDA designations beyond BTD and ODD."
    await evaluator.verify(
        claim=claim_urls,
        node=leaf_urls,
        sources=info.additional_designations_urls,
        additional_instruction="Pass if any page supports any additionally claimed designation(s)."
    )


async def build_approval_type(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Approval_Type_Documentation",
        desc="Verify and document the specific type of FDA approval received",
        parent=parent,
        critical=False  # Adjusted to be non-critical (optional per task)
    )

    # Approval classification (optional)
    gate_type = evaluator.add_custom_node(
        result=bool(info.approval_type),
        id="Approval_Type_Provided",
        desc="Approval type is provided in the answer",
        parent=node,
        critical=True
    )
    leaf_class = evaluator.add_leaf(
        id="Approval_Classification",
        desc="The approval is classified as either Full Approval or Accelerated Approval",
        parent=node,
        critical=False
    )
    claim_class = (
        f"The FDA approval type/classification for {drug_display(info.brand_name, info.generic_name)} is "
        f"{info.approval_type or 'unspecified'}."
    )
    await evaluator.verify(
        claim=claim_class,
        node=leaf_class,
        sources=combine_sources(info.approval_type_urls, info.approval_urls),
        additional_instruction="Confirm whether the approval was 'Accelerated approval' or a traditional/full approval. If gate failed, this will be skipped."
    )

    # Approval basis (optional aggregate)
    basis_node = evaluator.add_parallel(
        id="Approval_Basis",
        desc="Documentation describes the clinical evidence basis for approval",
        parent=node,
        critical=False
    )

    leaf_trial_data = evaluator.add_leaf(
        id="Clinical_Trial_Data",
        desc="Clinical trial results supporting the approval are referenced",
        parent=basis_node,
        critical=False
    )
    claim_trial_data = (
        "The cited page(s) reference clinical trial data supporting approval, such as pivotal study results or FDA review documents."
    )
    await evaluator.verify(
        claim=claim_trial_data,
        node=leaf_trial_data,
        sources=combine_sources(info.efficacy_urls, info.clinical_trials_urls, info.approval_urls),
        additional_instruction="Look for summary results, study identifiers, or FDA multi-discipline review references."
    )

    leaf_endpoints = evaluator.add_leaf(
        id="Efficacy_Endpoints",
        desc="Primary efficacy endpoints or outcome measures are documented",
        parent=basis_node,
        critical=False
    )
    claim_endpoints = (
        f"The primary efficacy endpoint(s) or outcome measures "
        f"{(' — ' + '; '.join(info.efficacy_endpoints)) if info.efficacy_endpoints else ''} are documented on the cited page(s)."
    )
    await evaluator.verify(
        claim=claim_endpoints,
        node=leaf_endpoints,
        sources=combine_sources(info.efficacy_urls, info.approval_urls),
        additional_instruction="Check that endpoints/outcomes are explicitly described."
    )


async def build_manufacturer(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Manufacturer_Sponsor_Information",
        desc="Verify identifiable pharmaceutical company sponsor/manufacturer information",
        parent=parent,
        critical=True
    )

    leaf_company = evaluator.add_leaf(
        id="Company_Name",
        desc="The pharmaceutical company name (sponsor/manufacturer) is clearly identified",
        parent=node,
        critical=True
    )
    claim_company = (
        f"The sponsor/manufacturer of {drug_display(info.brand_name, info.generic_name)} is {info.company or 'unspecified company'}."
    )
    await evaluator.verify(
        claim=claim_company,
        node=leaf_company,
        sources=combine_sources(info.manufacturer_urls, info.approval_urls),
        additional_instruction="The page should clearly name the sponsor/manufacturer."
    )

    # Promote to critical to satisfy framework: all children under critical parent must be critical
    leaf_company_type = evaluator.add_leaf(
        id="Company_Type",
        desc="Company type is identifiable (e.g., established pharmaceutical company, biotechnology company, etc.)",
        parent=node,
        critical=True
    )
    claim_type = (
        f"{info.company or 'The company'} is a pharmaceutical or biotechnology company (or equivalent manufacturer/sponsor)."
    )
    await evaluator.verify(
        claim=claim_type,
        node=leaf_company_type,
        sources=info.manufacturer_urls,
        additional_instruction="Use the company's website or press release to verify its nature as a pharma/biotech entity."
    )

    leaf_manu_url = evaluator.add_leaf(
        id="Manufacturer_URL",
        desc="Provide URL from company website or press release confirming their role as sponsor/manufacturer",
        parent=node,
        critical=True
    )
    claim_manu_url = "The cited page(s) from the company or press releases confirm their role as sponsor/manufacturer."
    await evaluator.verify(
        claim=claim_manu_url,
        node=leaf_manu_url,
        sources=info.manufacturer_urls,
        additional_instruction="Should explicitly state sponsorship/manufacturing of the FDA-approved drug."
    )


async def build_naming(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Drug_Naming_Information",
        desc="Verify complete drug naming information including trade and generic names",
        parent=parent,
        critical=True
    )

    leaf_trade = evaluator.add_leaf(
        id="Trade_Name",
        desc="The FDA-approved trade (brand) name is documented",
        parent=node,
        critical=True
    )
    claim_trade = f"The FDA-approved trade (brand) name is {info.brand_name or 'unspecified'}."
    await evaluator.verify(
        claim=claim_trade,
        node=leaf_trade,
        sources=combine_sources(info.name_sources, info.approval_urls),
        additional_instruction="The cited page should show the official brand/trade name used at approval."
    )

    leaf_generic = evaluator.add_leaf(
        id="Generic_Name",
        desc="The generic (chemical/non-proprietary) name is documented",
        parent=node,
        critical=True
    )
    claim_generic = f"The generic (nonproprietary) name is {info.generic_name or 'unspecified'}."
    await evaluator.verify(
        claim=claim_generic,
        node=leaf_generic,
        sources=combine_sources(info.name_sources, info.approval_urls),
        additional_instruction="The cited page should show the generic name."
    )

    leaf_format = evaluator.add_leaf(
        id="Name_Format_Accuracy",
        desc="Trade name and generic name follow proper pharmaceutical naming conventions",
        parent=node,
        critical=True  # promoted to meet engine constraints
    )
    claim_format = (
        "The trade (brand) name and generic (nonproprietary) name are presented in standard pharmaceutical naming format, "
        "with the brand as a distinct proper noun and the generic as a nonproprietary name."
    )
    await evaluator.verify(
        claim=claim_format,
        node=leaf_format,
        sources=combine_sources(info.name_sources, info.approval_urls),
        additional_instruction="Check that both trade and generic names are clearly identified and formatted appropriately."
    )


async def build_therapeutic_area(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Therapeutic_Area_Classification",
        desc="Verify the drug's therapeutic area classification",
        parent=parent,
        critical=False
    )

    leaf_area = evaluator.add_leaf(
        id="Primary_Therapeutic_Area",
        desc="The primary therapeutic area is identified (e.g., Neurology, Oncology, Rare Genetic Disorders, etc.)",
        parent=node,
        critical=False
    )
    claim_area = (
        f"The primary therapeutic area is {info.therapeutic_area or 'unspecified'}."
    )
    await evaluator.verify(
        claim=claim_area,
        node=leaf_area,
        sources=combine_sources(info.disease_info_urls, info.approval_urls),
        additional_instruction="Verify that the page indicates the high-level therapeutic area."
    )

    leaf_category = evaluator.add_leaf(
        id="Disease_Category",
        desc="The specific disease category within the therapeutic area is documented",
        parent=node,
        critical=False
    )
    claim_category = f"The disease category within the therapeutic area is {info.disease_category or 'unspecified'}."
    await evaluator.verify(
        claim=claim_category,
        node=leaf_category,
        sources=combine_sources(info.disease_info_urls, info.approval_urls),
        additional_instruction="Verify that the page states a more specific disease category where applicable."
    )


async def build_disease_characteristics(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Disease_Characteristics_Documentation",
        desc="Document key clinical characteristics of the target disease",
        parent=parent,
        critical=False  # adjusted to optional
    )

    desc_node = evaluator.add_parallel(
        id="Disease_Description",
        desc="A clear clinical description of the disease/condition is provided",
        parent=node,
        critical=False
    )

    leaf_path = evaluator.add_leaf(
        id="Pathophysiology",
        desc="Basic disease mechanism or pathophysiology is described",
        parent=desc_node,
        critical=False
    )
    claim_path = "The cited page(s) describe the disease pathophysiology or underlying mechanism."
    await evaluator.verify(
        claim=claim_path,
        node=leaf_path,
        sources=info.disease_info_urls,
        additional_instruction="Look for a brief mechanism/etiology description."
    )

    leaf_pop = evaluator.add_leaf(
        id="Affected_Populations",
        desc="Patient populations affected (age groups, demographics) are documented",
        parent=desc_node,
        critical=False
    )
    claim_pop = "The cited page(s) describe the affected patient populations (e.g., age groups, demographics)."
    await evaluator.verify(
        claim=claim_pop,
        node=leaf_pop,
        sources=info.disease_info_urls,
        additional_instruction="Look for statements about who is affected."
    )

    leaf_manifest = evaluator.add_leaf(
        id="Clinical_Manifestations",
        desc="Key symptoms or clinical manifestations of the disease are documented",
        parent=node,
        critical=False
    )
    claim_manifest = "The cited page(s) describe key symptoms or clinical manifestations of the disease."
    await evaluator.verify(
        claim=claim_manifest,
        node=leaf_manifest,
        sources=info.disease_info_urls,
        additional_instruction="Look for signs/symptoms descriptions."
    )

    leaf_disease_url = evaluator.add_leaf(
        id="Disease_Characteristics_URL",
        desc="Provide URL with detailed disease information from medical literature or health organization",
        parent=node,
        critical=False
    )
    claim_disease_url = "The provided page(s) are credible medical sources providing disease information."
    await evaluator.verify(
        claim=claim_disease_url,
        node=leaf_disease_url,
        sources=info.disease_info_urls,
        additional_instruction="Prefer NIH, CDC, FDA, major hospitals, or peer‑reviewed journals."
    )


async def build_clinical_evidence(evaluator: Evaluator, parent, info: DrugExtraction) -> None:
    node = evaluator.add_parallel(
        id="Clinical_Evidence_Documentation",
        desc="Document the clinical evidence supporting the drug approval",
        parent=parent,
        critical=False
    )

    # Clinical trial registration
    reg_node = evaluator.add_parallel(
        id="Clinical_Trial_Registration",
        desc="If applicable, clinical trials are registered on ClinicalTrials.gov with NCT numbers",
        parent=node,
        critical=False
    )

    gate_nct = evaluator.add_custom_node(
        result=bool(info.nct_numbers),
        id="NCT_Present",
        desc="NCT number(s) are provided in the answer",
        parent=reg_node,
        critical=True
    )

    leaf_nct_fmt = evaluator.add_leaf(
        id="NCT_Number_Format",
        desc="NCT numbers follow the proper format: 'NCT' followed by 8 digits",
        parent=reg_node,
        critical=False
    )
    claim_nct_fmt = f"The NCT identifiers {info.nct_numbers or []} follow the 'NCT' + 8 digits format."
    await evaluator.verify(
        claim=claim_nct_fmt,
        node=leaf_nct_fmt,
        sources=None,
        additional_instruction="This is a format check; confirm that each NCT matches regex ^NCT\\d{8}$. If the prerequisite failed, this will be skipped."
    )

    leaf_phase = evaluator.add_leaf(
        id="Trial_Phase",
        desc="The phase of the pivotal trial(s) is documented (Phase 1, 2, 3, or combination)",
        parent=reg_node,
        critical=False
    )
    claim_phase = f"The pivotal trial phase is {info.trial_phase or 'unspecified'} and is documented on the cited page(s)."
    await evaluator.verify(
        claim=claim_phase,
        node=leaf_phase,
        sources=combine_sources(info.clinical_trials_urls, info.approval_urls),
        additional_instruction="Confirm that the page explicitly states the trial phase. If the NCT prerequisite failed, this may be skipped."
    )

    # Efficacy data
    eff_node = evaluator.add_parallel(
        id="Efficacy_Data",
        desc="Summary efficacy data or clinical outcomes are referenced",
        parent=node,
        critical=False
    )

    leaf_primary = evaluator.add_leaf(
        id="Primary_Outcome",
        desc="The primary efficacy outcome or endpoint is described",
        parent=eff_node,
        critical=False
    )
    claim_primary = (
        f"The primary efficacy outcome/endpoint "
        f"{(' — ' + '; '.join(info.efficacy_endpoints)) if info.efficacy_endpoints else ''} "
        f"is described on the cited page(s)."
    )
    await evaluator.verify(
        claim=claim_primary,
        node=leaf_primary,
        sources=combine_sources(info.efficacy_urls, info.approval_urls),
        additional_instruction="Look for 'primary endpoint', 'primary outcome', or equivalent phrasing."
    )

    leaf_benefit = evaluator.add_leaf(
        id="Patient_Benefit",
        desc="Documented patient benefit or clinical improvement is described",
        parent=eff_node,
        critical=False
    )
    claim_benefit = "The cited page(s) describe patient benefit or clinical improvement achieved with the drug."
    await evaluator.verify(
        claim=claim_benefit,
        node=leaf_benefit,
        sources=combine_sources(info.efficacy_urls, info.approval_urls),
        additional_instruction="Look for efficacy results that indicate meaningful benefit."
    )

    # Any clinical evidence URL
    leaf_ce_url = evaluator.add_leaf(
        id="Clinical_Evidence_URL",
        desc="Provide URL to clinical trial information, study results, or FDA review documents",
        parent=node,
        critical=False
    )
    claim_ce_url = "The provided page(s) include clinical trial information, results, or FDA review documents."
    await evaluator.verify(
        claim=claim_ce_url,
        node=leaf_ce_url,
        sources=combine_sources(info.efficacy_urls, info.clinical_trials_urls, info.approval_urls),
        additional_instruction="Any credible sources including ClinicalTrials.gov, journals, or FDA reviews are acceptable."
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
    # Initialize evaluator (root set to PARALLEL and non-critical to allow partial credit on optional sections)
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

    # Create root task node (non-critical to allow optional sections to be partial)
    root_task = evaluator.add_parallel(
        id="Drug_Identification_Task",
        desc="Verify that the identified drug meets all specified FDA regulatory, disease-related, and clinical criteria",
        parent=root,
        critical=False  # Adjusted to avoid framework constraint; critical children will still gate failures
    )

    # Extract structured information from the answer
    info: DrugExtraction = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugExtraction,
        extraction_name="drug_info"
    )

    # Build verification subtrees
    await build_fda_approval(evaluator, root_task, info)
    await build_breakthrough(evaluator, root_task, info)
    await build_orphan(evaluator, root_task, info)
    await build_first_status(evaluator, root_task, info)
    await build_additional_designations(evaluator, root_task, info)
    await build_approval_type(evaluator, root_task, info)
    await build_manufacturer(evaluator, root_task, info)
    await build_naming(evaluator, root_task, info)
    await build_therapeutic_area(evaluator, root_task, info)
    await build_disease_characteristics(evaluator, root_task, info)
    await build_clinical_evidence(evaluator, root_task, info)

    # Add a compact summary of extracted key fields for debugging/traceability
    evaluator.add_custom_info(
        info={
            "brand_name": info.brand_name,
            "generic_name": info.generic_name,
            "indication": info.indication,
            "company": info.company,
            "approval_date": info.approval_date,
            "urls_counts": {
                "approval_urls": len(info.approval_urls),
                "btd_urls": len(info.btd_urls),
                "odd_urls": len(info.odd_urls),
                "first_treatment_urls": len(info.first_treatment_urls),
                "prevalence_urls": len(info.prevalence_urls),
                "manufacturer_urls": len(info.manufacturer_urls),
            },
        },
        info_type="extracted_summary",
        info_name="extracted_core_summary"
    )

    # Return standard summary
    return evaluator.get_summary()