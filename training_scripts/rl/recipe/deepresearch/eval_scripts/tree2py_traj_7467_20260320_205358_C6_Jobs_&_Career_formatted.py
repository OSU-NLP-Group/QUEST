import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "pa_superintendent_loe_eval"
TASK_DESCRIPTION = """
A school board in Pennsylvania is evaluating whether a candidate meets the state requirements for the Superintendent Letter of Eligibility certification. The candidate profile is as follows:

- Holds a Bachelor of Science in Elementary Education from a regionally accredited university
- Holds a Master of Education (M.Ed.) in Educational Leadership from a regionally accredited university
- Has 3 years of classroom teaching experience in Pennsylvania public schools with a valid Pennsylvania teaching certificate
- Has served as an elementary school principal for 4 years in Pennsylvania
- Currently holds a valid Pennsylvania principal certification
- Does not hold a doctoral degree

Based on Pennsylvania Department of Education requirements, verify whether this candidate meets all the mandatory requirements for obtaining a Superintendent Letter of Eligibility in Pennsylvania. Your response must include:

1. Verification of educational qualifications (bachelor's degree, master's degree, and completion of required preparation programs)
2. Verification of teaching experience requirements
3. Verification of administrative and supervisory experience requirements (total years of school experience and years in supervisory capacity)
4. Verification of certification requirements (teaching certificate, principal certification, and eligibility for superintendent certification)
5. URL references documenting each category of Pennsylvania's requirements

For each requirement category, clearly state whether the candidate meets the requirement and provide the specific Pennsylvania Department of Education criteria that support your determination.
"""


# -----------------------------------------------------------------------------
# Data models for structured extraction
# -----------------------------------------------------------------------------
class EducationSection(BaseModel):
    determination_text: Optional[str] = None
    determination_yes: Optional[bool] = None  # Whether the answer says the education category is met
    pde_criteria_text: Optional[str] = None   # The PDE rule/requirement text or paraphrase the answer used
    urls: List[str] = Field(default_factory=list)

    # Sub-requirements (what the answer explicitly verified)
    bachelors_statement: Optional[str] = None
    masters_statement: Optional[str] = None
    prep_program_statement: Optional[str] = None


class TeachingSection(BaseModel):
    determination_text: Optional[str] = None
    determination_yes: Optional[bool] = None
    pde_criteria_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

    classroom_experience_statement: Optional[str] = None  # e.g., "3 years of teaching with valid PA cert"


class AdminSection(BaseModel):
    determination_text: Optional[str] = None
    determination_yes: Optional[bool] = None
    pde_criteria_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

    total_experience_statement: Optional[str] = None      # e.g., "7 years total K-12 experience"
    supervisory_experience_statement: Optional[str] = None  # e.g., "4 years as principal"
    supervisory_role_statement: Optional[str] = None        # e.g., "Principal counts as supervisory"


class CertificationSection(BaseModel):
    determination_text: Optional[str] = None
    determination_yes: Optional[bool] = None
    pde_criteria_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

    teaching_cert_statement: Optional[str] = None
    principal_cert_statement: Optional[str] = None


class OverallSection(BaseModel):
    overall_conclusion_text: Optional[str] = None
    overall_meets_all: Optional[bool] = None  # Whether the answer concludes the candidate meets ALL mandatory requirements


class RequirementsExtraction(BaseModel):
    education: Optional[EducationSection] = None
    teaching: Optional[TeachingSection] = None
    admin: Optional[AdminSection] = None
    certification: Optional[CertificationSection] = None
    overall: Optional[OverallSection] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_sections() -> str:
    return """
    Extract the following structured information from the answer text. Only extract what is explicitly present in the answer. Do not invent or infer.

    For each category (education, teaching, admin, certification):
    - determination_text: The explicit sentence/phrase where the answer states whether the category is met/not met.
    - determination_yes: true if the answer explicitly concludes the category is met; false if explicitly not met; null if not explicitly concluded.
    - pde_criteria_text: The specific PDE requirement language or a clear paraphrase quoted/used by the answer for this category (e.g., degree/program requirements; experience years; certification requirements).
    - urls: All URLs cited in the answer for that category (PDE/PA sources or other official sources used to justify that category).
    
    Education sub-fields (if present in the answer):
    - bachelors_statement: The sentence/phrase showing the bachelor's degree verification.
    - masters_statement: The sentence/phrase showing the master's degree verification.
    - prep_program_statement: The sentence/phrase showing the superintendent preparation program requirement/status (e.g., completed/not completed/required).

    Teaching sub-field:
    - classroom_experience_statement: The sentence/phrase verifying the candidate’s classroom teaching experience (e.g., "3 years in PA with valid PA teaching certificate").

    Admin sub-fields:
    - total_experience_statement: The sentence/phrase verifying total K-12 school experience (e.g., "7 years").
    - supervisory_experience_statement: The sentence/phrase verifying supervisory/administrative years (e.g., "4 years as principal").
    - supervisory_role_statement: The sentence/phrase indicating that the role (e.g., principal) qualifies as supervisory/administrative for eligibility.

    Certification sub-fields:
    - teaching_cert_statement: The sentence/phrase verifying a valid PA (or acceptable) teaching certificate.
    - principal_cert_statement: The sentence/phrase verifying a valid PA (or acceptable) principal/administrative certification.

    Overall:
    - overall_conclusion_text: The explicit final yes/no conclusion sentence about whether the candidate meets all mandatory requirements.
    - overall_meets_all: true if the answer explicitly says the candidate meets all mandatory requirements now; false if explicitly says they do NOT (or are missing a mandatory requirement); null if no explicit overall determination.

    Return a JSON object with:
    {
      "education": { ... },
      "teaching": { ... },
      "admin": { ... },
      "certification": { ... },
      "overall": { ... }
    }

    Notes:
    - For any field not explicitly present in the answer, set it to null (or [] for urls).
    - For URLs, extract actual URLs (including those embedded in markdown).
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _expected_profile_outcomes() -> Dict[str, Any]:
    """
    Compute expected outcomes based on the candidate profile in the task description.
    Assumptions from the provided profile:
      - Bachelor's degree: yes
      - Master's degree: yes
      - No doctoral degree
      - Superintendent preparation program: not stated; assume not completed yet
      - Teaching experience: 3 years classroom in PA with valid PA cert (yes)
      - Principal: 4 years (supervisory) (yes)
      - Total K-12 experience: 7 years (>= 6) (yes)
      - Certifications: valid PA teaching cert and valid PA principal cert (both yes)
    Conclusion: All categories met EXCEPT education's preparation-program completion,
    so overall all mandatory requirements are NOT yet fully met.
    """
    return {
        "bachelors_met": True,
        "masters_met": True,
        "prep_program_completed": False,  # not provided; assume not completed yet
        "teaching_experience_met": True,  # 3 years classroom in PA with valid cert
        "total_experience_met": True,     # 7 >= 6
        "supervisory_experience_met": True,  # 4 >= 3
        "qualifying_role_met": True,      # principal counts
        "teaching_cert_met": True,
        "principal_cert_met": True,
        # Derived category expectations
        "education_category_met": False,  # missing prep program completion and no doctoral equivalency
        "teaching_category_met": True,
        "admin_category_met": True,
        "cert_category_met": True,
        "overall_meets_all": False,       # education not fully met
    }


async def _verify_criteria_with_urls(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent_node,
    urls: Optional[List[str]],
    criteria_text: Optional[str],
    category_name: str,
) -> None:
    """
    Verify that the PDE criteria text the answer provided is actually supported by the cited URLs.
    If missing text or URLs, fail this critical leaf.
    """
    if not _non_empty(criteria_text) or not _has_urls(urls):
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=True
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True
    )
    claim = (
        f"According to the cited source(s), the Pennsylvania requirements relevant to the {category_name} for the "
        f"Superintendent Letter of Eligibility are accurately described as: {criteria_text}"
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Judge whether the claim is explicitly supported by the provided webpage(s). "
            "Prioritize Pennsylvania Department of Education (PDE) or Pennsylvania state sources. "
            "Minor paraphrasing is acceptable if the requirement is clearly present on the page."
        )
    )


async def _verify_category_urls_present_and_relevant(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent_node,
    urls: Optional[List[str]],
    category_name: str,
) -> None:
    """
    Verify that at least one URL is provided and that it documents Pennsylvania/PDE requirements
    relevant to this category for the Superintendent LOE.
    """
    if not _has_urls(urls):
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=True
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True
    )
    claim = (
        f"This page documents Pennsylvania (PDE or PA state) requirements relevant to the {category_name} "
        f"for the Superintendent Letter of Eligibility (or superintendent certification)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Pass if at least one cited page is an official Pennsylvania Department of Education or PA state page "
            "that states requirements relevant to this category. If none of the pages are relevant or official, fail."
        )
    )


# -----------------------------------------------------------------------------
# Category verification builders
# -----------------------------------------------------------------------------
async def build_education_section(
    evaluator: Evaluator,
    parent,
    edu: Optional[EducationSection],
    expected: Dict[str, Any]
) -> None:
    node = evaluator.add_parallel(
        id="educational_qualifications",
        desc="Verify educational qualifications (bachelor's degree, master's degree, and required preparation program) and include the required write-up elements and URLs.",
        parent=parent,
        critical=True
    )

    # Leaf: bachelor's degree requirement (presence + correctness relative to profile)
    leaf_bach = evaluator.add_leaf(
        id="bachelors_degree_requirement",
        desc="Candidate holds a bachelor's degree in education or a related field from a regionally accredited institution.",
        parent=node,
        critical=True
    )
    bach_claim = (
        "The answer explicitly verifies that the candidate holds a bachelor's degree from a regionally accredited "
        "institution, as stated in the task's candidate profile."
    )
    await evaluator.verify(
        claim=bach_claim,
        node=leaf_bach,
        additional_instruction="Fail if the answer does not clearly state this verification."
    )

    # Leaf: master's degree requirement (presence + correctness relative to profile)
    leaf_mast = evaluator.add_leaf(
        id="masters_degree_requirement",
        desc="Candidate holds a master's degree in educational leadership/administration or a related field from a regionally accredited institution.",
        parent=node,
        critical=True
    )
    mast_claim = (
        "The answer explicitly verifies that the candidate holds a master's degree from a regionally accredited "
        "institution, as stated in the task's candidate profile."
    )
    await evaluator.verify(
        claim=mast_claim,
        node=leaf_mast,
        additional_instruction="Fail if the answer does not clearly state this verification."
    )

    # Leaf: superintendent preparation program requirement (presence + clear determination)
    leaf_prep = evaluator.add_leaf(
        id="superintendent_preparation_program_requirement",
        desc="Candidate has completed or is eligible to complete the Pennsylvania-approved superintendent certification/preparation program required to obtain the Superintendent Letter of Eligibility.",
        parent=node,
        critical=True
    )
    # We require the answer to explicitly address the preparation-program requirement and give a clear status.
    # We also expect, given the task profile (no doctoral degree and no mention of program completion),
    # that the correct determination is that this requirement is NOT yet met unless the answer provides clear evidence of completion.
    prep_claim = (
        "The answer explicitly addresses the Pennsylvania-approved superintendent preparation program requirement for the "
        "Letter of Eligibility and clearly determines the candidate's current status (e.g., completed/not completed/pending). "
        "Given the candidate profile in the task (no doctoral degree and no stated completion of a superintendent preparation "
        "program), the correct determination is that this requirement is not yet met unless the answer presents clear evidence "
        "of program completion."
    )
    await evaluator.verify(
        claim=prep_claim,
        node=leaf_prep,
        additional_instruction=(
            "Pass if the answer clearly flags this as a mandatory requirement and concludes it is NOT yet satisfied "
            "unless they explicitly show completion; fail if ignored or incorrectly claimed as met without justification."
        )
    )

    # Leaf: education category determination (explicit + correct overall logic)
    leaf_det = evaluator.add_leaf(
        id="education_category_determination",
        desc="Explicitly state whether the candidate meets the educational qualifications category requirements.",
        parent=node,
        critical=True
    )
    edu_expected = "NOT MET" if not expected["education_category_met"] else "MET"
    det_claim = (
        f"The answer explicitly states the overall determination for the educational qualifications category. "
        f"Based on the task's candidate profile and PDE requirements (master's degree plus completion of a PDE-approved "
        f"superintendent preparation program or an approved doctoral equivalent), the correct overall determination is: {edu_expected}."
    )
    await evaluator.verify(
        claim=det_claim,
        node=leaf_det,
        additional_instruction=(
            "Accept phrasing like 'not met', 'unmet', 'pending completion', etc., as NOT MET. "
            "Fail if the answer lacks a clear category determination or clearly contradicts the correct outcome."
        )
    )

    # Leaf: education category PDE criteria (must be supported by URLs)
    await _verify_criteria_with_urls(
        evaluator=evaluator,
        node_id="education_category_pde_criteria",
        desc="Provide the specific PDE criteria (rule/regulation language or clearly identified requirement statements) used to justify the education-category determination.",
        parent_node=node,
        urls=(edu.urls if edu else []),
        criteria_text=(edu.pde_criteria_text if edu else None),
        category_name="education and preparation-program"
    )

    # Leaf: education category URLs (must include PDE/PA requirement documentation)
    await _verify_category_urls_present_and_relevant(
        evaluator=evaluator,
        node_id="education_category_urls",
        desc="Provide URL reference(s) documenting Pennsylvania/PDE educational and preparation-program requirements for superintendent/Letter of Eligibility.",
        parent_node=node,
        urls=(edu.urls if edu else []),
        category_name="education/preparation-program"
    )


async def build_teaching_section(
    evaluator: Evaluator,
    parent,
    teach: Optional[TeachingSection],
    expected: Dict[str, Any]
) -> None:
    node = evaluator.add_parallel(
        id="teaching_experience",
        desc="Verify teaching experience requirements and include the required write-up elements and URLs.",
        parent=parent,
        critical=True
    )

    # Leaf: classroom teaching experience requirement (presence + correctness relative to profile)
    leaf_cls = evaluator.add_leaf(
        id="classroom_teaching_experience_requirement",
        desc="Candidate meets Pennsylvania's minimum classroom (instructional) teaching experience requirement for superintendent/Letter of Eligibility eligibility (per PDE criteria).",
        parent=node,
        critical=True
    )
    cls_claim = (
        "The answer explicitly verifies the candidate's classroom teaching background as stated in the task profile "
        "(3 years in Pennsylvania with a valid PA teaching certificate) and uses it appropriately when assessing eligibility."
    )
    await evaluator.verify(
        claim=cls_claim,
        node=leaf_cls,
        additional_instruction="Fail if the answer omits or misstates the teaching experience from the task profile."
    )

    # Leaf: teaching category determination
    leaf_det = evaluator.add_leaf(
        id="teaching_category_determination",
        desc="Explicitly state whether the candidate meets the teaching experience category requirements.",
        parent=node,
        critical=True
    )
    teach_expected = "MET" if expected["teaching_category_met"] else "NOT MET"
    teach_det_claim = (
        f"The answer explicitly states the category determination for teaching/ instructional experience. "
        f"Given the task profile (3 years of classroom teaching in PA with a valid certificate), the correct determination is: {teach_expected}. "
        f"Accept formulations that treat classroom experience as contributing to the total-experience rule, as long as the category is clearly addressed."
    )
    await evaluator.verify(
        claim=teach_det_claim,
        node=leaf_det,
        additional_instruction="Fail if no clear category determination is provided."
    )

    # Leaf: teaching category PDE criteria (supported by URLs)
    await _verify_criteria_with_urls(
        evaluator=evaluator,
        node_id="teaching_category_pde_criteria",
        desc="Provide the specific PDE criteria used to justify the teaching-category determination.",
        parent_node=node,
        urls=(teach.urls if teach else []),
        criteria_text=(teach.pde_criteria_text if teach else None),
        category_name="experience/teaching"
    )

    # Leaf: teaching category URLs
    await _verify_category_urls_present_and_relevant(
        evaluator=evaluator,
        node_id="teaching_category_urls",
        desc="Provide URL reference(s) documenting Pennsylvania/PDE teaching experience requirements relevant to superintendent/Letter of Eligibility.",
        parent_node=node,
        urls=(teach.urls if teach else []),
        category_name="experience/teaching"
    )


async def build_admin_section(
    evaluator: Evaluator,
    parent,
    adm: Optional[AdminSection],
    expected: Dict[str, Any]
) -> None:
    node = evaluator.add_parallel(
        id="administrative_and_supervisory_experience",
        desc="Verify total school experience and supervisory/administrative experience requirements and include the required write-up elements and URLs.",
        parent=parent,
        critical=True
    )

    # Leaf: total school experience requirement (presence + correctness relative to profile)
    leaf_total = evaluator.add_leaf(
        id="total_school_experience_requirement",
        desc="Candidate has at least 6 years of satisfactory school experience in K-12 public or private schools (per Pennsylvania requirement).",
        parent=node,
        critical=True
    )
    total_claim = (
        "The answer explicitly verifies the candidate's total K-12 school experience based on the task profile "
        "(3 years classroom + 4 years principal = 7 years), and recognizes this as meeting the 'at least 6 years' requirement."
    )
    await evaluator.verify(
        claim=total_claim,
        node=leaf_total,
        additional_instruction="Fail if the answer omits total-experience verification or miscalculates the years."
    )

    # Leaf: supervisory/administrative experience requirement (presence + correctness relative to profile)
    leaf_superv = evaluator.add_leaf(
        id="supervisory_experience_requirement",
        desc="Candidate has at least 3 years of the required experience in a supervisory or administrative capacity (per Pennsylvania requirement).",
        parent=node,
        critical=True
    )
    superv_claim = (
        "The answer explicitly verifies that the candidate has at least 3 years in a supervisory/administrative capacity, "
        "based on the task profile (4 years as principal)."
    )
    await evaluator.verify(
        claim=superv_claim,
        node=leaf_superv,
        additional_instruction="Fail if the answer omits supervisory/administrative years or states them incorrectly."
    )

    # Leaf: qualifying supervisory role requirement (presence + correctness relative to profile)
    leaf_role = evaluator.add_leaf(
        id="qualifying_supervisory_role_requirement",
        desc="Candidate's supervisory/administrative role(s) are among those qualifying under Pennsylvania/PDE criteria (e.g., principal or other listed qualifying roles).",
        parent=node,
        critical=True
    )
    role_claim = (
        "The answer explicitly identifies the elementary school principal role as a qualifying supervisory/administrative "
        "role for Superintendent LOE eligibility under Pennsylvania criteria."
    )
    await evaluator.verify(
        claim=role_claim,
        node=leaf_role,
        additional_instruction="Fail if the answer does not make clear that principal is counted as supervisory/administrative."
    )

    # Leaf: admin category determination
    leaf_det = evaluator.add_leaf(
        id="admin_category_determination",
        desc="Explicitly state whether the candidate meets the administrative/supervisory experience category requirements.",
        parent=node,
        critical=True
    )
    adm_expected = "MET" if expected["admin_category_met"] else "NOT MET"
    adm_det_claim = (
        f"The answer explicitly provides the category determination for administrative/supervisory experience. "
        f"Given the task profile (7 total years with 4 in a supervisory/administrative role), the correct determination is: {adm_expected}."
    )
    await evaluator.verify(
        claim=adm_det_claim,
        node=leaf_det,
        additional_instruction="Fail if no clear category determination is provided."
    )

    # Leaf: admin category PDE criteria (supported by URLs)
    await _verify_criteria_with_urls(
        evaluator=evaluator,
        node_id="admin_category_pde_criteria",
        desc="Provide the specific PDE criteria used to justify the administrative/supervisory-category determination.",
        parent_node=node,
        urls=(adm.urls if adm else []),
        criteria_text=(adm.pde_criteria_text if adm else None),
        category_name="experience (total and supervisory)"
    )

    # Leaf: admin category URLs
    await _verify_category_urls_present_and_relevant(
        evaluator=evaluator,
        node_id="admin_category_urls",
        desc="Provide URL reference(s) documenting Pennsylvania/PDE total (6-year) and supervisory (3-year) experience requirements for superintendent/Letter of Eligibility.",
        parent_node=node,
        urls=(adm.urls if adm else []),
        category_name="experience (total/supervisory)"
    )


async def build_certification_section(
    evaluator: Evaluator,
    parent,
    cert: Optional[CertificationSection],
    expected: Dict[str, Any]
) -> None:
    node = evaluator.add_parallel(
        id="certification_requirements",
        desc="Verify certification/licensure requirements and include the required write-up elements and URLs.",
        parent=parent,
        critical=True
    )

    # Leaf: teaching certificate requirement (presence + correctness relative to profile)
    leaf_tcert = evaluator.add_leaf(
        id="teaching_certificate_requirement",
        desc="Candidate holds or has held a valid teaching certificate/license in Pennsylvania or another state (as allowed by Pennsylvania/PDE requirement).",
        parent=node,
        critical=True
    )
    tcert_claim = (
        "The answer explicitly verifies that the candidate holds a valid Pennsylvania teaching certificate, "
        "as stated in the task's candidate profile."
    )
    await evaluator.verify(
        claim=tcert_claim,
        node=leaf_tcert,
        additional_instruction="Fail if the answer does not clearly state this verification."
    )

    # Leaf: principal/admin certification requirement (presence + correctness relative to profile)
    leaf_pcert = evaluator.add_leaf(
        id="principal_or_admin_certification_requirement",
        desc="Candidate holds or has held principal certification or administrative certification in Pennsylvania or another state (as allowed by Pennsylvania/PDE requirement).",
        parent=node,
        critical=True
    )
    pcert_claim = (
        "The answer explicitly verifies that the candidate holds a valid Pennsylvania principal certification, "
        "as stated in the task's candidate profile."
    )
    await evaluator.verify(
        claim=pcert_claim,
        node=leaf_pcert,
        additional_instruction="Fail if the answer does not clearly state this verification."
    )

    # Leaf: certification category determination
    leaf_det = evaluator.add_leaf(
        id="certification_category_determination",
        desc="Explicitly state whether the candidate meets the certification requirements category.",
        parent=node,
        critical=True
    )
    cert_expected = "MET" if expected["cert_category_met"] else "NOT MET"
    cert_det_claim = (
        f"The answer explicitly provides the category determination for certifications. "
        f"Given the task profile (valid PA teaching certificate and valid PA principal certification), the correct determination is: {cert_expected}."
    )
    await evaluator.verify(
        claim=cert_det_claim,
        node=leaf_det,
        additional_instruction="Fail if no clear category determination is provided."
    )

    # Leaf: certification category PDE criteria (supported by URLs)
    await _verify_criteria_with_urls(
        evaluator=evaluator,
        node_id="certification_category_pde_criteria",
        desc="Provide the specific PDE criteria used to justify the certification-category determination (including what certifications are required/acceptable).",
        parent_node=node,
        urls=(cert.urls if cert else []),
        criteria_text=(cert.pde_criteria_text if cert else None),
        category_name="certification/licensure"
    )

    # Leaf: certification category URLs
    await _verify_category_urls_present_and_relevant(
        evaluator=evaluator,
        node_id="certification_category_urls",
        desc="Provide URL reference(s) documenting Pennsylvania/PDE certification/licensure requirements and the Letter of Eligibility process.",
        parent_node=node,
        urls=(cert.urls if cert else []),
        category_name="certification/licensure"
    )


async def build_overall_conclusion(
    evaluator: Evaluator,
    parent,
    overall: Optional[OverallSection],
    expected: Dict[str, Any]
) -> None:
    # Single critical leaf per rubric
    leaf = evaluator.add_leaf(
        id="overall_conclusion",
        desc="Provide an overall yes/no conclusion on whether the candidate meets all mandatory requirements for the Superintendent Letter of Eligibility (or identify which mandatory requirement(s) are not met).",
        parent=parent,
        critical=True
    )
    expected_text = "DOES NOT meet all mandatory requirements" if not expected["overall_meets_all"] else "MEETS all mandatory requirements"
    claim = (
        f"The answer provides a clear overall yes/no conclusion. Based on the task's candidate profile and Pennsylvania "
        f"requirements (including the approved superintendent preparation program requirement), the correct overall conclusion is: "
        f"{expected_text}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=(
            "Accept explicit statements such as 'does not meet', 'not yet eligible', or 'pending completion of the required "
            "superintendent preparation program' as indicating DOES NOT meet. Fail if the answer has no explicit overall conclusion "
            "or clearly contradicts the correct outcome."
        )
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer against Pennsylvania Superintendent Letter of Eligibility requirements,
    according to the provided rubric tree.
    """
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_sections(),
        template_class=RequirementsExtraction,
        extraction_name="structured_sections"
    )

    # Compute expected outcomes from the given candidate profile in the task description
    expected = _expected_profile_outcomes()

    # Add ground truth summary for transparency
    evaluator.add_ground_truth({
        "candidate_profile_summary": {
            "bachelor_degree": True,
            "master_degree": True,
            "doctoral_degree": False,
            "teaching_years": 3,
            "principal_years": 4,
            "total_years": 7,
            "pa_teaching_cert": True,
            "pa_principal_cert": True,
            "superintendent_prep_program_completed": False
        },
        "expected_outcomes": expected
    }, gt_type="expected_logic_from_profile")

    # Build/verify each rubric section
    await build_education_section(evaluator, root, extraction.education, expected)
    await build_teaching_section(evaluator, root, extraction.teaching, expected)
    await build_admin_section(evaluator, root, extraction.admin, expected)
    await build_certification_section(evaluator, root, extraction.certification, expected)
    await build_overall_conclusion(evaluator, root, extraction.overall, expected)

    return evaluator.get_summary()