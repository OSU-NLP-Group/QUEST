import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nci_center_and_drug_2024"
TASK_DESCRIPTION = """
Identify an NCI-designated comprehensive cancer center in the United States that meets all of the following criteria:
(1) The center must have been both founded and became an NCI-designated comprehensive cancer center in 1987;
(2) The center must consist of exactly three consortium institutions;
(3) Among the three consortium institutions, one must be a university or medical school;
(4) Among the three consortium institutions, one must be a hospital system founded in 1866;
(5) Among the three consortium institutions, one must be a medical center founded in 1921;
(6) The cancer center must have enrolled more than 2,000 patients in clinical trials during 2024;
(7) The cancer center must have affiliated cancer centers at both Cleveland Clinic and University Hospitals.

Additionally, identify an immunotherapy drug that meets all of the following criteria:
(1) The drug must have been FDA-approved in April 2024;
(2) The drug must be classified as an IL-15 receptor agonist;
(3) The drug must be the first-in-class drug of its mechanism for its indication;
(4) The drug must be approved for BCG-unresponsive non-muscle invasive bladder cancer (NMIBC);
(5) The drug must be administered in combination with Bacillus Calmette-Guérin (BCG);
(6) The drug must activate both natural killer cells and T cells.

Provide the name of the cancer center and the brand name of the immunotherapy drug, along with supporting reference URLs for all key facts.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #

class ConsortiumMember(BaseModel):
    name: Optional[str] = None
    kind: Optional[str] = None  # e.g., "university", "medical school", "hospital system", "medical center"
    founding_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CancerCenterExtraction(BaseModel):
    name: Optional[str] = None

    # NCI status and years
    nci_status_text: Optional[str] = None  # e.g., "NCI-designated Comprehensive Cancer Center"
    founded_year: Optional[str] = None
    nci_designation_year: Optional[str] = None
    nci_reference_urls: List[str] = Field(default_factory=list)

    # Consortium structure
    consortium_members: List[ConsortiumMember] = Field(default_factory=list)
    consortium_structure_reference_urls: List[str] = Field(default_factory=list)

    # Enrollment in 2024
    enrollment_2024_text: Optional[str] = None  # e.g., "over 2,000 patients in 2024"
    enrollment_2024_number: Optional[str] = None  # raw number if present, else textual descriptor
    enrollment_reference_urls: List[str] = Field(default_factory=list)

    # Partner affiliations
    partner_affiliation_cleveland_clinic: Optional[str] = None
    partner_affiliation_university_hospitals: Optional[str] = None
    partner_reference_urls: List[str] = Field(default_factory=list)


class DrugExtraction(BaseModel):
    brand_name: Optional[str] = None

    # FDA approval
    fda_approval_date_text: Optional[str] = None  # e.g., "April 2024"
    approval_reference_urls: List[str] = Field(default_factory=list)

    # Classification & first-in-class
    il15_receptor_agonist_text: Optional[str] = None
    first_in_class_text: Optional[str] = None
    classification_reference_urls: List[str] = Field(default_factory=list)

    # Indication & combination
    bcg_unresponsive_nmibc_text: Optional[str] = None
    combination_with_bcg_text: Optional[str] = None
    indication_reference_urls: List[str] = Field(default_factory=list)

    # Mechanism (activates NK and T cells)
    activates_nk_cells_text: Optional[str] = None
    activates_t_cells_text: Optional[str] = None
    mechanism_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #

def prompt_extract_cancer_center() -> str:
    return """
    From the answer, extract the single NCI-designated comprehensive cancer center and its key facts with all supporting URLs exactly as provided.

    Required fields (strings preferred; use null if missing):
    - name: The cancer center name.
    - nci_status_text: Phrase indicating NCI comprehensive designation (e.g., "NCI-designated Comprehensive Cancer Center").
    - founded_year: Founding year of the cancer center (e.g., "1987").
    - nci_designation_year: The year it became an NCI-designated comprehensive cancer center (e.g., "1987").
    - nci_reference_urls: ALL URLs cited that support the NCI status and years (list).

    Consortium structure:
    - consortium_members: EXACTLY three members in the order presented in the answer (truncate if more; pad with empty if fewer).
      Each member requires:
        - name
        - kind (one of: "university", "medical school", "hospital system", "medical center", or a reasonable synonym)
        - founding_year (if provided in the answer)
        - reference_urls: ALL URLs for the member (list)
    - consortium_structure_reference_urls: ALL URLs that explicitly support the "three-institution consortium" structure (list).

    Clinical trial enrollment:
    - enrollment_2024_text: The phrase or sentence about 2024 enrollment (e.g., "over 2,000 patients in 2024").
    - enrollment_2024_number: The numeric value if present (e.g., "2000+", "2,345"), else null.
    - enrollment_reference_urls: ALL URLs that support the 2024 enrollment statement (list).

    Partner affiliations:
    - partner_affiliation_cleveland_clinic: Text describing affiliation with Cleveland Clinic.
    - partner_affiliation_university_hospitals: Text describing affiliation with University Hospitals.
    - partner_reference_urls: ALL URLs that together verify both affiliations (list).

    SPECIAL URL RULES:
    - Extract only actual URLs present in the answer (including markdown links).
    - Do not invent or fetch external URLs.
    - If a URL is missing protocol, prepend http:// as needed.
    """.strip()


def prompt_extract_drug() -> str:
    return """
    From the answer, extract the immunotherapy drug and its key facts with all supporting URLs exactly as provided.

    Required fields (strings preferred; use null if missing):
    - brand_name: Drug brand name.

    FDA approval:
    - fda_approval_date_text: Text indicating the approval timing (e.g., "April 2024").
    - approval_reference_urls: ALL URLs that support the approval timing (list).

    Classification & first-in-class:
    - il15_receptor_agonist_text: Text indicating it is an IL-15 receptor agonist (or superagonist targeting IL-15R).
    - first_in_class_text: Text indicating first-in-class status for its mechanism/indication.
    - classification_reference_urls: ALL URLs supporting classification & first-in-class (list).

    Indication & combination:
    - bcg_unresponsive_nmibc_text: Text indicating indication for BCG-unresponsive non–muscle invasive bladder cancer (NMIBC).
    - combination_with_bcg_text: Text indicating administration with Bacillus Calmette-Guérin (BCG).
    - indication_reference_urls: ALL URLs supporting indication and combination (list).

    Mechanism of action:
    - activates_nk_cells_text: Text indicating activation of natural killer (NK) cells.
    - activates_t_cells_text: Text indicating activation of T cells.
    - mechanism_reference_urls: ALL URLs supporting activation of both NK and T cells (list).

    SPECIAL URL RULES:
    - Extract only actual URLs present in the answer (including markdown links).
    - Do not invent or fetch external URLs.
    - If a URL is missing protocol, prepend http:// as needed.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #

def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Remove empty/whitespace entries
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _find_member_by_kind(members: List[ConsortiumMember], keywords: List[str]) -> Optional[ConsortiumMember]:
    """
    Find first member whose 'kind' or 'name' suggests one of the keywords.
    """
    kws = [k.lower() for k in keywords]
    for m in members:
        if not m:
            continue
        kind = (m.kind or "").lower()
        name = (m.name or "").lower()
        if any(kw in kind for kw in kws) or any(kw in name for kw in kws):
            return m
    return None


def _find_member_by_kind_and_year(members: List[ConsortiumMember], kind_keywords: List[str], year_hint: str) -> Optional[ConsortiumMember]:
    """
    Find a member by kind + founding year hint.
    """
    for m in members:
        if not m:
            continue
        kind = (m.kind or "").lower()
        name = (m.name or "").lower()
        fy = (m.founding_year or "")
        if (any(kw in kind for kw in kind_keywords) or any(kw in name for kw in kind_keywords)) and year_hint in fy:
            return m
    # If exact year match not found, fallback to kind only to still evaluate
    return _find_member_by_kind(members, kind_keywords)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #

async def verify_cancer_center(evaluator: Evaluator, parent: VerificationNode, cc: CancerCenterExtraction) -> None:
    """
    Build and verify the cancer center subtree.
    """
    cc_node = evaluator.add_parallel(
        id="cancer_center_identification",
        desc="Correct identification and verification of the NCI-designated comprehensive cancer center meeting all specified criteria",
        parent=parent,
        critical=False  # Allow partial at top-level within cancer center block
    )

    # 1) NCI designation & years
    nci_node = evaluator.add_parallel(
        id="nci_designation_verification",
        desc="Verification of NCI comprehensive cancer center designation and founding year",
        parent=cc_node,
        critical=True
    )

    # Presence of references for NCI status and years (critical prerequisite)
    evaluator.add_custom_node(
        result=len(_safe_urls(cc.nci_reference_urls)) > 0,
        id="nci_designation_reference",
        desc="Provided reference URL verifying NCI designation and founding year",
        parent=nci_node,
        critical=True
    )

    # NCI comprehensive status
    nci_status_leaf = evaluator.add_leaf(
        id="nci_comprehensive_status",
        desc="The center is designated as an NCI comprehensive cancer center",
        parent=nci_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The center '{cc.name or ''}' is designated as an NCI comprehensive cancer center.",
        node=nci_status_leaf,
        sources=_safe_urls(cc.nci_reference_urls),
        additional_instruction="Accept equivalent phrasings (e.g., 'Comprehensive Cancer Center designated by NCI'). The page(s) must explicitly show this designation."
    )

    # Founded and designated in 1987 (split into two critical leaves under a critical parent)
    founded_designated_parent = evaluator.add_parallel(
        id="founded_and_designated_1987",
        desc="The center was both founded and became NCI-designated in 1987",
        parent=nci_node,
        critical=True
    )

    founded_leaf = evaluator.add_leaf(
        id="founded_1987",
        desc="The center was founded in 1987",
        parent=founded_designated_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The center '{cc.name or ''}' was founded in 1987.",
        node=founded_leaf,
        sources=_safe_urls(cc.nci_reference_urls),
        additional_instruction="The source must clearly indicate the founding year is 1987. Accept phrasing like 'established in 1987'."
    )

    designated_leaf = evaluator.add_leaf(
        id="designated_1987",
        desc="The center became NCI-designated in 1987",
        parent=founded_designated_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The center '{cc.name or ''}' became an NCI-designated (comprehensive) cancer center in 1987.",
        node=designated_leaf,
        sources=_safe_urls(cc.nci_reference_urls),
        additional_instruction="The source must clearly indicate the NCI designation year is 1987. Accept phrasing like 'received NCI comprehensive designation in 1987'."
    )

    # 2) Consortium structure (exactly three institutions)
    consortium_node = evaluator.add_parallel(
        id="consortium_structure_verification",
        desc="Verification that the cancer center consists of exactly three consortium institutions",
        parent=cc_node,
        critical=True
    )

    # Presence of structure reference
    evaluator.add_custom_node(
        result=len(_safe_urls(cc.consortium_structure_reference_urls)) > 0,
        id="consortium_structure_reference",
        desc="Provided reference URL verifying the three-institution consortium structure",
        parent=consortium_node,
        critical=True
    )

    # Explicit three-institution claim
    consortium_leaf = evaluator.add_leaf(
        id="three_consortium_institutions",
        desc="The cancer center consists of exactly three consortium institutions",
        parent=consortium_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cancer center '{cc.name or ''}' consists of exactly three consortium institutions.",
        node=consortium_leaf,
        sources=_safe_urls(cc.consortium_structure_reference_urls),
        additional_instruction="The supporting source should explicitly say there are three institutions (e.g., 'a consortium of X, Y, and Z')."
    )

    # 3) Consortium members with required characteristics
    members_node = evaluator.add_parallel(
        id="consortium_member_identification",
        desc="Identification and verification of all three consortium members with their required characteristics",
        parent=cc_node,
        critical=True
    )

    # Select members by category/year
    univ_member = _find_member_by_kind(cc.consortium_members, ["university", "medical school", "school of medicine"])
    hosp_1866_member = _find_member_by_kind_and_year(cc.consortium_members, ["hospital", "hospital system", "health system"], "1866")
    medcenter_1921_member = _find_member_by_kind_and_year(cc.consortium_members, ["medical center", "clinic"], "1921")

    # University/Medical School
    univ_node = evaluator.add_parallel(
        id="first_consortium_member_university",
        desc="First consortium member identification and verification as university/medical school",
        parent=members_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(univ_member and len(_safe_urls(univ_member.reference_urls)) > 0),
        id="university_reference",
        desc="Provided reference URL verifying the university/medical school status",
        parent=univ_node,
        critical=True
    )
    univ_leaf = evaluator.add_leaf(
        id="university_medical_school_status",
        desc="The first consortium member is a university or medical school",
        parent=univ_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The consortium member '{(univ_member.name if univ_member else '')}' is a university or a medical school.",
        node=univ_leaf,
        sources=_safe_urls(univ_member.reference_urls if univ_member else []),
        additional_instruction="Accept synonyms like 'School of Medicine' or 'University'. The page should clearly classify the institution."
    )

    # Hospital system founded in 1866
    hosp_node = evaluator.add_parallel(
        id="second_consortium_member_hospital_1866",
        desc="Second consortium member identification and verification as hospital founded in 1866",
        parent=members_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hosp_1866_member and len(_safe_urls(hosp_1866_member.reference_urls)) > 0),
        id="hospital_1866_reference",
        desc="Provided reference URL verifying the hospital founding year of 1866",
        parent=hosp_node,
        critical=True
    )
    hosp_leaf = evaluator.add_leaf(
        id="hospital_founded_1866",
        desc="The second consortium member is a hospital system founded in 1866",
        parent=hosp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The consortium member '{(hosp_1866_member.name if hosp_1866_member else '')}' is a hospital system founded in 1866.",
        node=hosp_leaf,
        sources=_safe_urls(hosp_1866_member.reference_urls if hosp_1866_member else []),
        additional_instruction="The source should indicate this is a hospital system (or health system) and clearly show a founding year of 1866."
    )

    # Medical center founded in 1921
    med_node = evaluator.add_parallel(
        id="third_consortium_member_medical_center_1921",
        desc="Third consortium member identification and verification as medical center founded in 1921",
        parent=members_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(medcenter_1921_member and len(_safe_urls(medcenter_1921_member.reference_urls)) > 0),
        id="medical_center_1921_reference",
        desc="Provided reference URL verifying the medical center founding year of 1921",
        parent=med_node,
        critical=True
    )
    med_leaf = evaluator.add_leaf(
        id="medical_center_founded_1921",
        desc="The third consortium member is a medical center founded in 1921",
        parent=med_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The consortium member '{(medcenter_1921_member.name if medcenter_1921_member else '')}' is a medical center founded in 1921.",
        node=med_leaf,
        sources=_safe_urls(medcenter_1921_member.reference_urls if medcenter_1921_member else []),
        additional_instruction="Accept 'clinic' if it is widely recognized as a medical center. The page should clearly show a founding year of 1921."
    )

    # 4) Enrollment in 2024 > 2,000
    enroll_node = evaluator.add_parallel(
        id="clinical_trial_enrollment_verification",
        desc="Verification of clinical trial enrollment exceeding 2,000 patients in 2024",
        parent=cc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(cc.enrollment_reference_urls)) > 0,
        id="enrollment_data_reference",
        desc="Provided reference URL verifying the 2024 clinical trial enrollment data",
        parent=enroll_node,
        critical=True
    )
    enroll_leaf = evaluator.add_leaf(
        id="enrollment_exceeds_2000_in_2024",
        desc="The cancer center enrolled more than 2,000 patients in clinical trials during 2024",
        parent=enroll_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cancer center '{cc.name or ''}' enrolled more than 2,000 patients in clinical trials during 2024.",
        node=enroll_leaf,
        sources=_safe_urls(cc.enrollment_reference_urls),
        additional_instruction="Accept wording like 'over 2,000', 'more than two thousand', or any clear equivalent for 2024 totals."
    )

    # 5) Partner cancer centers affiliations
    partners_node = evaluator.add_parallel(
        id="partner_cancer_centers_verification",
        desc="Verification of affiliated cancer centers at Cleveland Clinic and University Hospitals",
        parent=cc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(cc.partner_reference_urls)) > 0,
        id="partner_centers_reference",
        desc="Provided reference URL verifying both Cleveland Clinic and University Hospitals affiliations",
        parent=partners_node,
        critical=True
    )

    cle_leaf = evaluator.add_leaf(
        id="cleveland_clinic_affiliation",
        desc="The cancer center has an affiliated cancer center at Cleveland Clinic",
        parent=partners_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cancer center '{cc.name or ''}' has an affiliated cancer center at Cleveland Clinic.",
        node=cle_leaf,
        sources=_safe_urls(cc.partner_reference_urls),
        additional_instruction="The source should explicitly link the cancer center with Cleveland Clinic as an affiliated or partner site."
    )

    uh_leaf = evaluator.add_leaf(
        id="university_hospitals_affiliation",
        desc="The cancer center has an affiliated cancer center at University Hospitals",
        parent=partners_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cancer center '{cc.name or ''}' has an affiliated cancer center at University Hospitals.",
        node=uh_leaf,
        sources=_safe_urls(cc.partner_reference_urls),
        additional_instruction="The source should explicitly link the cancer center with University Hospitals as an affiliated or partner site."
    )


async def verify_drug(evaluator: Evaluator, parent: VerificationNode, drug: DrugExtraction) -> None:
    """
    Build and verify the drug subtree.
    """
    drug_node = evaluator.add_parallel(
        id="immunotherapy_drug_identification",
        desc="Correct identification and verification of the immunotherapy drug meeting all specified criteria",
        parent=parent,
        critical=False  # Allow partial at top-level within drug block
    )

    # 1) FDA approval in April 2024
    approval_node = evaluator.add_parallel(
        id="fda_approval_verification",
        desc="Verification of FDA approval in April 2024",
        parent=drug_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(drug.approval_reference_urls)) > 0,
        id="fda_approval_reference",
        desc="Provided reference URL verifying FDA approval date in April 2024",
        parent=approval_node,
        critical=True
    )
    approval_leaf = evaluator.add_leaf(
        id="approved_april_2024",
        desc="The drug was FDA-approved in April 2024",
        parent=approval_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' was FDA-approved in April 2024.",
        node=approval_leaf,
        sources=_safe_urls(drug.approval_reference_urls),
        additional_instruction="Accept exact date strings (e.g., April 22, 2024). The page must clearly indicate April 2024 as the approval time."
    )

    # 2) Classification: IL-15 receptor agonist & first-in-class
    class_node = evaluator.add_parallel(
        id="drug_classification_mechanism_verification",
        desc="Verification of drug classification as IL-15 receptor agonist and first-in-class status",
        parent=drug_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(drug.classification_reference_urls)) > 0,
        id="classification_reference",
        desc="Provided reference URL verifying IL-15 receptor agonist classification and first-in-class status",
        parent=class_node,
        critical=True
    )

    il15_leaf = evaluator.add_leaf(
        id="il15_receptor_agonist",
        desc="The drug is classified as an IL-15 receptor agonist",
        parent=class_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' is classified as an IL-15 receptor agonist.",
        node=il15_leaf,
        sources=_safe_urls(drug.classification_reference_urls),
        additional_instruction="Accept 'IL-15 receptor superagonist' or 'IL-15R agonist' as equivalent language if it clearly implies IL-15 receptor agonism."
    )

    fic_leaf = evaluator.add_leaf(
        id="first_in_class_status",
        desc="The drug is the first-in-class of its mechanism for its indication",
        parent=class_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' is first-in-class for its mechanism for its indicated use.",
        node=fic_leaf,
        sources=_safe_urls(drug.classification_reference_urls),
        additional_instruction="Look for wording like 'first-in-class', 'first of its kind', or 'first approved agent of this mechanism' for the indicated disease."
    )

    # 3) Indication & combination with BCG
    indication_node = evaluator.add_parallel(
        id="drug_indication_verification",
        desc="Verification of drug indication for BCG-unresponsive NMIBC and combination with BCG",
        parent=drug_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(drug.indication_reference_urls)) > 0,
        id="indication_reference",
        desc="Provided reference URL verifying the BCG-unresponsive NMIBC indication and BCG combination",
        parent=indication_node,
        critical=True
    )

    nmibc_leaf = evaluator.add_leaf(
        id="bcg_unresponsive_nmibc_indication",
        desc="The drug is approved for BCG-unresponsive non-muscle invasive bladder cancer (NMIBC)",
        parent=indication_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' is approved for BCG-unresponsive non–muscle invasive bladder cancer (NMIBC).",
        node=nmibc_leaf,
        sources=_safe_urls(drug.indication_reference_urls),
        additional_instruction="Accept clear synonyms or expansions (e.g., 'BCG-unresponsive NMIBC including CIS or papillary disease')."
    )

    combo_leaf = evaluator.add_leaf(
        id="combination_with_bcg",
        desc="The drug is administered in combination with Bacillus Calmette-Guérin (BCG)",
        parent=indication_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' is administered in combination with Bacillus Calmette-Guérin (BCG).",
        node=combo_leaf,
        sources=_safe_urls(drug.indication_reference_urls),
        additional_instruction="The page should state that the drug is given with BCG (e.g., 'in combination with BCG')."
    )

    # 4) Mechanism of action: activates NK and T cells
    mech_node = evaluator.add_parallel(
        id="mechanism_of_action_verification",
        desc="Verification that the drug activates both natural killer cells and T cells",
        parent=drug_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(drug.mechanism_reference_urls)) > 0,
        id="mechanism_reference",
        desc="Provided reference URL verifying activation of natural killer cells and T cells",
        parent=mech_node,
        critical=True
    )

    nk_leaf = evaluator.add_leaf(
        id="activates_natural_killer_cells",
        desc="The drug activates natural killer cells",
        parent=mech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' activates natural killer (NK) cells.",
        node=nk_leaf,
        sources=_safe_urls(drug.mechanism_reference_urls),
        additional_instruction="Accept clear verbs like 'activates', 'stimulates', or 'expands'. Must explicitly mention NK cells."
    )

    t_leaf = evaluator.add_leaf(
        id="activates_t_cells",
        desc="The drug activates T cells",
        parent=mech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The drug '{drug.brand_name or ''}' activates T cells.",
        node=t_leaf,
        sources=_safe_urls(drug.mechanism_reference_urls),
        additional_instruction="Accept clear verbs like 'activates', 'stimulates', or 'expands'. Must explicitly mention T cells."
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
    Evaluate an answer for the NCI center & drug task.

    Returns:
        A structured summary containing the verification tree and scores.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates two major parts in parallel
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

    # Extract cancer center and drug info in parallel
    cancer_center_task = evaluator.extract(
        prompt=prompt_extract_cancer_center(),
        template_class=CancerCenterExtraction,
        extraction_name="cancer_center"
    )
    drug_task = evaluator.extract(
        prompt=prompt_extract_drug(),
        template_class=DrugExtraction,
        extraction_name="drug_info"
    )

    cancer_center_extraction, drug_extraction = await asyncio.gather(cancer_center_task, drug_task)

    # Root is critical in rubric, but framework root is non-critical; enforce both major branches as critical subtrees
    # Cancer center subtree (critical to pass as per rubric, but partial credit allowed within)
    await verify_cancer_center(evaluator, root, cancer_center_extraction)

    # Drug subtree (critical to pass as per rubric, but partial credit allowed within)
    await verify_drug(evaluator, root, drug_extraction)

    return evaluator.get_summary()