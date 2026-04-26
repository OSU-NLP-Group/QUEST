import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "specialized_accreditations_requirements"
TASK_DESCRIPTION = """
A mid-sized public university is planning to expand its academic offerings and seek specialized programmatic accreditations for four new professional programs: business administration, engineering, nursing, and social work. The university's accreditation planning committee needs a comprehensive report identifying the specific mandatory requirements they must meet for each accreditation body.

For each of the following specialized accreditations, identify the key mandatory requirements that the university must satisfy:

1. AACSB International (Association to Advance Collegiate Schools of Business) - for business programs
2. ABET (Accreditation Board for Engineering and Technology) - for engineering programs
3. CCNE (Commission on Collegiate Nursing Education) - for nursing programs
4. CSWE (Council on Social Work Education) - for social work programs

For each accreditation type, your response must include:
- The specific curriculum or credit hour requirements (where applicable)
- Faculty credential requirements
- Program quality or learning outcome standards
- Any other critical mandatory requirements specified by the accrediting body
- A supporting URL reference from the official accrediting organization's website or recognized higher education sources for each stated requirement

Your answer should provide the university's planning committee with actionable, verifiable information about what standards they must meet to successfully achieve each of these four specialized accreditations.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    """
    One concrete requirement mentioned in the answer, with its own supporting URLs.
    Extract EXACTLY what the answer claims (verbatim or close paraphrase) and the URLs cited for this requirement.
    """
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AACSBExtraction(BaseModel):
    aol: Optional[RequirementItem] = None
    strategic_standards: Optional[RequirementItem] = None
    faculty_credentials: Optional[RequirementItem] = None
    quality_standards: Optional[RequirementItem] = None


class ABETExtraction(BaseModel):
    math_science_credits: Optional[RequirementItem] = None
    engineering_topics_credits: Optional[RequirementItem] = None
    design_component: Optional[RequirementItem] = None
    program_name_requirement: Optional[RequirementItem] = None


class CCNEExtraction(BaseModel):
    essentials_compliance: Optional[RequirementItem] = None
    faculty_qualifications: Optional[RequirementItem] = None
    student_outcomes: Optional[RequirementItem] = None
    degree_levels: Optional[RequirementItem] = None


class CSWEExtraction(BaseModel):
    epas_compliance: Optional[RequirementItem] = None
    competency_based: Optional[RequirementItem] = None
    faculty_credentials: Optional[RequirementItem] = None
    degree_levels: Optional[RequirementItem] = None


class AccreditationExtraction(BaseModel):
    aacsb: Optional[AACSBExtraction] = None
    abet: Optional[ABETExtraction] = None
    ccne: Optional[CCNEExtraction] = None
    cswe: Optional[CSWEExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_accreditation_requirements() -> str:
    return """
    You must extract, from the given answer text, the concrete requirement statements claimed for each accrediting body and the supporting URLs that the answer cites for those specific requirements.

    OUTPUT RULES:
    - Extract EXACTLY what the answer claims for each requirement (verbatim or close paraphrase), do not invent or normalize.
    - For each requirement, also extract the list of URLs that the answer explicitly associates with that requirement. If the answer provides one shared list of sources for multiple requirements for the same accreditor, repeat those URLs for each applicable requirement.
    - If a requirement is not mentioned, set its 'statement' to null and 'urls' to an empty list.
    - Only include URLs actually present in the answer; do not infer or fabricate URLs.

    TARGET FIELDS:
    1) aacsb:
       - aol: Assurance of Learning requirement text and URLs (e.g., implementation/documentation of AoL for learning outcomes)
       - strategic_standards: strategic management/innovation standards claim and URLs
       - faculty_credentials: faculty qualification requirements claim and URLs
       - quality_standards: quality in teaching/research/curricula/student learning outcomes claim and URLs

    2) abet:
       - math_science_credits: curriculum math/basic science requirement as stated (may be numeric credit hours OR phrased like “one year”); include URLs
       - engineering_topics_credits: curriculum engineering topics requirement as stated; include URLs
       - design_component: major design experience/component requirement; include URLs
       - program_name_requirement: requirement regarding including the word "engineering" or naming constraints as claimed; include URLs

    3) ccne:
       - essentials_compliance: claim that programs must comply with “The Essentials ...” standards; include URLs
       - faculty_qualifications: faculty credentials requirement; include URLs
       - student_outcomes: student success/learning outcomes requirement; include URLs
       - degree_levels: which degree levels CCNE accredits (e.g., BSN/MSN/DNP); include URLs

    4) cswe:
       - epas_compliance: claim that programs must meet EPAS; include URLs
       - competency_based: competency-based education approach requirement; include URLs
       - faculty_credentials: faculty credential requirements for social work; include URLs
       - degree_levels: which degree levels CSWE accredits (BSW/MSW); include URLs

    IMPORTANT:
    - Do not add any requirement text that is not explicitly stated in the answer.
    - If the answer mentions the requirement but without a dedicated URL, return an empty list for urls for that requirement.

    Return a single JSON object matching the AccreditationExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _build_additional_instruction(accreditor: str, topic_hint: str) -> str:
    base = (
        "Verify whether the provided URLs explicitly support that the quoted statement is a mandatory requirement "
        f"for {accreditor} accreditation. Accept reasonable wording variations and synonyms. "
        "Focus on explicit standards/criteria pages, handbooks, or official accreditation policies; "
        "recognized higher-education reference sites are acceptable if they accurately cite the accreditor's standards. "
        "If the page only provides general advice or marketing language without stating an accreditation requirement, treat as not supported."
    )
    extras = ""
    if accreditor.upper() == "ABET" and "credit" in topic_hint.lower():
        extras = (
            " For ABET curriculum credit requirements, treat phrases like 'one year of mathematics and basic sciences' "
            "as equivalent to common semester-credit-hour minima (e.g., around 30 credits) where appropriate; "
            "the key is that an explicit minimum breadth/amount is mandated by ABET."
        )
    return base + extras


async def verify_requirement_with_urls(
    evaluator: Evaluator,
    parent,
    *,
    accreditor: str,
    req_node_id: str,
    req_node_desc: str,
    url_leaf_id: str,
    url_leaf_desc: str,
    supported_leaf_id: str,
    item: Optional[RequirementItem],
    topic_hint: str,
    critical: bool = True,
) -> None:
    """
    Build a sequential node for a single requirement:
      1) Check URL presence (critical) – fail here will skip the support check.
      2) Verify that the requirement statement is supported by at least one provided URL (critical).
    """
    # Always create the requirement node
    req_node = evaluator.add_sequential(
        id=req_node_id,
        desc=req_node_desc,
        parent=parent,
        critical=critical
    )

    statement = (item.statement.strip() if (item and item.statement) else "") or ""
    urls = (item.urls if item else []) or []

    # 1) URL presence check (critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=url_leaf_id,
        desc=url_leaf_desc,
        parent=req_node,
        critical=True
    )

    # 2) Claim support by URLs (critical)
    supported_leaf = evaluator.add_leaf(
        id=supported_leaf_id,
        desc=f"Claim is supported by the provided URL(s) for {accreditor}: {topic_hint}",
        parent=req_node,
        critical=True
    )

    # Build claim tying to the answer text so we judge what the answer asserted
    # and verify it against the provided sources.
    quoted = statement if statement else "[NO STATEMENT PROVIDED IN ANSWER]"
    claim = (
        f"The answer claims the following requirement for {accreditor} accreditation:\n"
        f"\"{quoted}\"\n\n"
        f"Determine if this is an actual mandatory accreditation requirement of {accreditor}."
    )

    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=urls,
        additional_instruction=_build_additional_instruction(accreditor, topic_hint)
    )


# --------------------------------------------------------------------------- #
# Accreditation-specific verification builders                                #
# --------------------------------------------------------------------------- #
async def build_aacsb_checks(evaluator: Evaluator, root, data: Optional[AACSBExtraction]) -> None:
    grp = evaluator.add_parallel(
        id="AACSB_Requirements",
        desc="Complete and accurate requirements for AACSB business program accreditation",
        parent=root,
        critical=False
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="AACSB",
        req_node_id="AACSB_AoL_Requirement",
        req_node_desc="Formal Assurance of Learning (AoL) processes must be implemented and documented",
        url_leaf_id="AACSB_AoL_URL",
        url_leaf_desc="URL reference supporting the AoL requirement",
        supported_leaf_id="AACSB_AoL_Supported",
        item=data.aol if data else None,
        topic_hint="Assurance of Learning (AoL) requirement"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="AACSB",
        req_node_id="AACSB_Strategic_Standards",
        req_node_desc="Programs must demonstrate strategic management and innovation through defined standards",
        url_leaf_id="AACSB_Strategic_URL",
        url_leaf_desc="URL reference supporting strategic management standards",
        supported_leaf_id="AACSB_Strategic_Supported",
        item=data.strategic_standards if data else None,
        topic_hint="Strategic management and innovation standards"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="AACSB",
        req_node_id="AACSB_Faculty_Credentials",
        req_node_desc="Faculty must meet AACSB qualification requirements",
        url_leaf_id="AACSB_Faculty_URL",
        url_leaf_desc="URL reference supporting faculty credential requirements",
        supported_leaf_id="AACSB_Faculty_Supported",
        item=data.faculty_credentials if data else None,
        topic_hint="Faculty qualification/credentialing requirements"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="AACSB",
        req_node_id="AACSB_Quality_Standards",
        req_node_desc="Programs must demonstrate quality in teaching, research, curricula, and student learning outcomes",
        url_leaf_id="AACSB_Quality_URL",
        url_leaf_desc="URL reference supporting quality standards",
        supported_leaf_id="AACSB_Quality_Supported",
        item=data.quality_standards if data else None,
        topic_hint="Program quality and learning outcomes standards"
    )


async def build_abet_checks(evaluator: Evaluator, root, data: Optional[ABETExtraction]) -> None:
    grp = evaluator.add_parallel(
        id="ABET_Requirements",
        desc="Complete and accurate requirements for ABET engineering program accreditation",
        parent=root,
        critical=False
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="ABET",
        req_node_id="ABET_Math_Science_Credits",
        req_node_desc="Curriculum must include at least a specified minimum of mathematics and basic science",
        url_leaf_id="ABET_Math_URL",
        url_leaf_desc="URL reference supporting math and science requirement",
        supported_leaf_id="ABET_Math_Science_Supported",
        item=data.math_science_credits if data else None,
        topic_hint="Minimum mathematics and basic sciences requirement (e.g., 'one year' or numeric credits)"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="ABET",
        req_node_id="ABET_Engineering_Credits",
        req_node_desc="Curriculum must include a minimum in engineering topics",
        url_leaf_id="ABET_Engineering_URL",
        url_leaf_desc="URL reference supporting engineering topics requirement",
        supported_leaf_id="ABET_Engineering_Topics_Supported",
        item=data.engineering_topics_credits if data else None,
        topic_hint="Minimum engineering topics requirement (e.g., 'one and one-half years' or numeric credits)"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="ABET",
        req_node_id="ABET_Design_Component",
        req_node_desc="Curriculum must include a major design component/experience",
        url_leaf_id="ABET_Design_URL",
        url_leaf_desc="URL reference supporting design component requirement",
        supported_leaf_id="ABET_Design_Component_Supported",
        item=data.design_component if data else None,
        topic_hint="Major design experience/component (often capstone) requirement"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="ABET",
        req_node_id="ABET_Program_Name",
        req_node_desc="Program name requirement (e.g., inclusion of 'engineering' as claimed in the answer)",
        url_leaf_id="ABET_Name_URL",
        url_leaf_desc="URL reference supporting program naming requirement",
        supported_leaf_id="ABET_Program_Name_Supported",
        item=data.program_name_requirement if data else None,
        topic_hint="Program naming constraints (e.g., contains 'engineering' or other ABET naming constraints)"
    )


async def build_ccne_checks(evaluator: Evaluator, root, data: Optional[CCNEExtraction]) -> None:
    grp = evaluator.add_parallel(
        id="CCNE_Requirements",
        desc="Complete and accurate requirements for CCNE nursing program accreditation",
        parent=root,
        critical=False
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CCNE",
        req_node_id="CCNE_Essentials_Compliance",
        req_node_desc="Programs must comply with The Essentials standards, as claimed in the answer",
        url_leaf_id="CCNE_Essentials_URL",
        url_leaf_desc="URL reference supporting Essentials compliance requirement",
        supported_leaf_id="CCNE_Essentials_Supported",
        item=data.essentials_compliance if data else None,
        topic_hint="Compliance with 'The Essentials' standards (e.g., AACN Essentials referenced in CCNE standards)"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CCNE",
        req_node_id="CCNE_Faculty_Qualifications",
        req_node_desc="Programs must demonstrate qualified faculty credentials",
        url_leaf_id="CCNE_Faculty_URL",
        url_leaf_desc="URL reference supporting faculty qualification requirements",
        supported_leaf_id="CCNE_Faculty_Supported",
        item=data.faculty_qualifications if data else None,
        topic_hint="Faculty qualification/credential requirements"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CCNE",
        req_node_id="CCNE_Student_Outcomes",
        req_node_desc="Programs must demonstrate student success and learning outcomes",
        url_leaf_id="CCNE_Outcomes_URL",
        url_leaf_desc="URL reference supporting student outcomes requirements",
        supported_leaf_id="CCNE_Outcomes_Supported",
        item=data.student_outcomes if data else None,
        topic_hint="Student success, assessment, and learning outcomes standards"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CCNE",
        req_node_id="CCNE_Degree_Levels",
        req_node_desc="CCNE accreditation applies to specified nursing degree levels (as claimed in the answer)",
        url_leaf_id="CCNE_Levels_URL",
        url_leaf_desc="URL reference supporting degree level specifications",
        supported_leaf_id="CCNE_Degree_Levels_Supported",
        item=data.degree_levels if data else None,
        topic_hint="Degree levels covered (e.g., baccalaureate, master's, practice-focused doctoral)"
    )


async def build_cswe_checks(evaluator: Evaluator, root, data: Optional[CSWEExtraction]) -> None:
    grp = evaluator.add_parallel(
        id="CSWE_Requirements",
        desc="Complete and accurate requirements for CSWE social work program accreditation",
        parent=root,
        critical=False
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CSWE",
        req_node_id="CSWE_EPAS_Compliance",
        req_node_desc="Programs must meet Educational Policy and Accreditation Standards (EPAS)",
        url_leaf_id="CSWE_EPAS_URL",
        url_leaf_desc="URL reference supporting EPAS requirements",
        supported_leaf_id="CSWE_EPAS_Supported",
        item=data.epas_compliance if data else None,
        topic_hint="EPAS (Educational Policy and Accreditation Standards) compliance"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CSWE",
        req_node_id="CSWE_Competency_Based",
        req_node_desc="Programs must demonstrate a competency-based education approach",
        url_leaf_id="CSWE_Competency_URL",
        url_leaf_desc="URL reference supporting competency-based education requirement",
        supported_leaf_id="CSWE_Competency_Supported",
        item=data.competency_based if data else None,
        topic_hint="Competency-based education approach"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CSWE",
        req_node_id="CSWE_Faculty_Credentials",
        req_node_desc="Faculty must have appropriate credentials for social work education",
        url_leaf_id="CSWE_Faculty_URL",
        url_leaf_desc="URL reference supporting faculty credential requirements",
        supported_leaf_id="CSWE_Faculty_Supported",
        item=data.faculty_credentials if data else None,
        topic_hint="Faculty credential requirements for social work programs"
    )

    await verify_requirement_with_urls(
        evaluator,
        grp,
        accreditor="CSWE",
        req_node_id="CSWE_Degree_Levels",
        req_node_desc="CSWE standards apply to specified social work degree levels (as claimed in the answer)",
        url_leaf_id="CSWE_Levels_URL",
        url_leaf_desc="URL reference supporting degree level specifications",
        supported_leaf_id="CSWE_Degree_Levels_Supported",
        item=data.degree_levels if data else None,
        topic_hint="Degree levels covered (e.g., BSW and MSW)"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate an answer for specialized accreditation requirements across AACSB, ABET, CCNE, and CSWE.
    """
    # Initialize evaluator (root is non-critical by framework design; we allow partial credit across accreditors)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Accreditors evaluated independently
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

    # Extract structured accreditation requirements from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_accreditation_requirements(),
        template_class=AccreditationExtraction,
        extraction_name="accreditation_requirements"
    )

    # Build verification tree per accreditor
    await build_aacsb_checks(evaluator, root, extracted.aacsb)
    await build_abet_checks(evaluator, root, extracted.abet)
    await build_ccne_checks(evaluator, root, extracted.ccne)
    await build_cswe_checks(evaluator, root, extracted.cswe)

    # Return the evaluation summary
    return evaluator.get_summary()