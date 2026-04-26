import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "assistant_fb_coach_quals"
TASK_DESCRIPTION = (
    "Identify the complete set of qualifications required to become an assistant football coach at a Division I NCAA "
    "college or university. Your answer must include three main qualification categories: (1) Educational Requirements "
    "- specify the minimum degree level required and provide a list of acceptable academic fields of study for this degree; "
    "(2) Certification Requirements - identify the required professional coaching certification mandated by the NCAA, "
    "including the specific name of the certification test and details about how it is administered; and (3) Experience "
    "Requirements - describe the typical prior coaching experience needed, including the entry-level coaching roles that "
    "serve as prerequisites and information about the typical duration or years of experience required before qualifying "
    "for an assistant coach position. For each qualification category, you must provide supporting reference URLs from "
    "credible sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EducationInfo(BaseModel):
    min_degree_level: Optional[str] = None
    acceptable_fields: List[str] = Field(default_factory=list)
    masters_preferred_mentioned: Optional[bool] = None
    masters_preferred_note: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CertificationInfo(BaseModel):
    test_name: Optional[str] = None
    administration_details: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class ExperienceInfo(BaseModel):
    entry_roles: List[str] = Field(default_factory=list)
    years_required: Optional[str] = None
    advancement_path_mentioned: Optional[bool] = None
    advancement_path_note: Optional[str] = None
    drivers_license_required: Optional[bool] = None
    playing_experience_preferred: Optional[bool] = None
    rules_compliance_knowledge: Optional[bool] = None
    winning_record_head_coach: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class QualificationsExtraction(BaseModel):
    education: Optional[EducationInfo] = None
    certification: Optional[CertificationInfo] = None
    experience: Optional[ExperienceInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qualifications() -> str:
    return """
    Extract structured information about the qualifications to become an assistant football coach at an NCAA Division I institution
    as presented in the answer. Organize into three categories with the following fields:

    education:
      - min_degree_level: The minimum degree level explicitly stated (e.g., "bachelor's", "BA/BS", etc.). If not stated, null.
      - acceptable_fields: A list of academic fields the answer says are acceptable or appropriate (e.g., sports management, kinesiology, exercise science, physical education, sports administration, health science, business administration, or related). Extract as many as present.
      - masters_preferred_mentioned: true/false if the answer notes that many universities prefer a master's degree (but not universally required for assistant roles). If uncertain, set to null.
      - masters_preferred_note: The sentence/phrase used in the answer about master's preference, if present; else null.
      - urls: A list of all URLs the answer provides that specifically support EDUCATIONAL requirements. Extract only explicit URLs.

    certification:
      - test_name: The specific name of the NCAA-required certification test (e.g., "NCAA Coaches Certification (Recruiting) Test" or an equivalent NCAA recruiting certification test). If not stated, null.
      - administration_details: A list of concrete details from the answer about how the test is administered (proctored/unproctored, online/in-person, by whom, timing/availability). Empty list if none provided.
      - urls: A list of all URLs the answer provides that specifically support CERTIFICATION requirements. Extract only explicit URLs.

    experience:
      - entry_roles: A list of the typical entry-level prerequisite roles (e.g., Graduate Assistant (GA), Quality Control assistant, analyst, student assistant). Extract all listed.
      - years_required: The typical duration/years of prior coaching experience mentioned (exact value or a range as text, e.g., "2–4 years"). If absent, null.
      - advancement_path_mentioned: true/false if the answer mentions a typical path like GA/entry-level assistant → position coach → coordinator → head coach. If uncertain, null.
      - advancement_path_note: The sentence/phrase describing the advancement path if present; else null.
      - drivers_license_required: true/false if the answer states a valid driver's license is required for recruiting. If uncertain, null.
      - playing_experience_preferred: true/false if the answer states collegiate playing experience is preferred but not mandatory. If uncertain, null.
      - rules_compliance_knowledge: true/false if the answer notes head coaches/coordinators must show strong NCAA rules compliance knowledge. If uncertain, null.
      - winning_record_head_coach: true/false if the answer notes head coach candidates generally need a winning record. If uncertain, null.
      - urls: A list of all URLs the answer provides that specifically support EXPERIENCE requirements. Extract only explicit URLs.

    IMPORTANT:
    - Extract only what the answer explicitly states. Do not infer or add new information.
    - For URLs, include only explicit URLs mentioned in the answer (plain or markdown).
    - Use null for missing single-value fields; use empty lists for missing list fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_n(items: List[str], n: int = 4) -> List[str]:
    return [x for x in items if isinstance(x, str) and x.strip()][:n]


def _join_list(items: List[str], sep: str = ", ") -> str:
    items = [x.strip() for x in items if isinstance(x, str) and x.strip()]
    return sep.join(items)


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_education(evaluator: Evaluator, parent_node, edu: Optional[EducationInfo]) -> None:
    # Category node (non-critical to allow a mix of mandatory and optional children)
    education_node = evaluator.add_parallel(
        id="Educational_Requirements",
        desc="Educational requirements and acceptable fields of study, with supporting URLs.",
        parent=parent_node,
        critical=False,
    )

    # --- Sub-node: Content stated in the answer (critical group) ---
    edu_content = evaluator.add_parallel(
        id="edu_content",
        desc="Education content stated in the answer",
        parent=education_node,
        critical=True,
    )

    # Degree requirement (standalone sub-group to avoid over-gating)
    degree_req = evaluator.add_parallel(
        id="edu_degree_requirement",
        desc="Minimum degree requirement content",
        parent=edu_content,
        critical=True,
    )

    # Leaf: Minimum degree is bachelor's (answer states)
    min_deg_leaf = evaluator.add_leaf(
        id="Minimum_Degree_Level_Bachelors",
        desc="States that a bachelor's degree is the minimum required degree level for college-level coaching positions (or explicitly for the target role).",
        parent=degree_req,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a bachelor's degree (e.g., BA/BS) is the minimum required degree level for assistant college football coaching roles or college-level coaching positions.",
        node=min_deg_leaf,
        additional_instruction="Accept reasonable synonyms such as 'bachelor's degree', 'BA', 'BS', or 'undergraduate degree' as equivalent for the minimum requirement."
    )

    # Fields requirement (own sub-group; includes existence and support-by-URL in sources group)
    fields_req = evaluator.add_parallel(
        id="edu_fields_requirement",
        desc="Acceptable academic fields of study content",
        parent=edu_content,
        critical=True,
    )

    fields_list = edu.acceptable_fields if edu else []
    fields_exist = evaluator.add_custom_node(
        result=len(fields_list) >= 3,
        id="Acceptable_Fields_Count_At_Least_3",
        desc="Answer provides at least 3 acceptable academic field examples.",
        parent=fields_req,
        critical=True,
    )

    fields_leaf = evaluator.add_leaf(
        id="Acceptable_Fields_of_Study_Examples",
        desc="Provides a list of acceptable/appropriate academic fields for the degree and includes at least 3 example fields consistent with the constraints (e.g., sports management, kinesiology, exercise science, physical education, sports administration, health science, business administration, or related disciplines).",
        parent=fields_req,
        critical=True,
    )
    three_or_more = _first_n(fields_list, 6)
    fields_claim = f"The answer lists acceptable academic fields including: {', '.join(three_or_more)}; there are at least three such examples."
    await evaluator.verify(
        claim=fields_claim,
        node=fields_leaf,
        additional_instruction="Judge only whether the answer itself lists at least three plausible academic fields relevant to sports/athletics coaching domains. Do not require URL evidence for this leaf."
    )

    # --- Sub-node: Optional educational notes (non-critical) ---
    edu_optional = evaluator.add_parallel(
        id="edu_optional",
        desc="Optional educational notes (non-critical)",
        parent=education_node,
        critical=False,
    )

    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree_Preferred",
        desc="Notes that many universities prefer a master's degree (especially for coordinator/head roles) without claiming it is universally mandatory for assistant coach roles.",
        parent=edu_optional,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer notes that many universities prefer a master's degree (particularly for coordinator/head roles) but it is not universally mandatory for assistant coach roles.",
        node=masters_leaf,
        additional_instruction="Pass if the answer explicitly indicates preference rather than a strict universal requirement."
    )

    # --- Sub-node: Source support and credibility (critical group) ---
    edu_support = evaluator.add_parallel(
        id="edu_sources_support",
        desc="Educational requirements supported by credible sources",
        parent=education_node,
        critical=True,
    )

    edu_urls_present = evaluator.add_custom_node(
        result=_has_urls(edu.urls if edu else []),
        id="Educational_Source_URLs_Provided",
        desc="At least one educational supporting reference URL is provided in the answer.",
        parent=edu_support,
        critical=True,
    )

    # Verify min degree claim is supported by cited sources
    min_degree_supported = evaluator.add_leaf(
        id="Minimum_Degree_Level_Supported_By_Sources",
        desc="Bachelor's minimum degree requirement is supported by cited educational sources.",
        parent=edu_support,
        critical=True,
    )
    await evaluator.verify(
        claim="A bachelor's degree is cited as the minimum degree requirement for college-level assistant coaching roles (or college coaching roles) on this webpage.",
        node=min_degree_supported,
        sources=(edu.urls if edu else []),
        additional_instruction="Look for language indicating a bachelor's degree is required or minimum for assistant coach or college-level coaching roles. Accept .edu job postings or NCAA/credible org pages."
    )

    # Verify acceptable fields are supported by the cited sources
    fields_supported = evaluator.add_leaf(
        id="Acceptable_Fields_Supported_By_Sources",
        desc="Acceptable degree fields (e.g., sports management, kinesiology, exercise science, physical education, sports administration, health science, business) are supported by cited educational sources.",
        parent=edu_support,
        critical=True,
    )
    example_fields = ", ".join(_first_n(fields_list, 5)) if fields_list else "sports management, kinesiology, exercise science"
    await evaluator.verify(
        claim=f"The webpage supports that appropriate degree fields for aspiring college-level football coaches include fields such as {example_fields} or closely related disciplines.",
        node=fields_supported,
        sources=(edu.urls if edu else []),
        additional_instruction="Accept evidence where the page lists relevant fields for sports/athletics coaching or staff roles; allow near-equivalents."
    )

    # Credibility check for at least one educational source
    edu_sources_credible = evaluator.add_leaf(
        id="Educational_Sources_URLs_Credible",
        desc="Provides ≥1 supporting reference URL for the educational requirements; sources are credible/non-user-generated (e.g., NCAA, .edu, government, recognized professional organizations; not forums/social media).",
        parent=edu_support,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is a credible, non-user-generated source (e.g., NCAA, .edu, .gov, recognized professional organization) and it discusses educational qualifications/requirements relevant to college coaching.",
        node=edu_sources_credible,
        sources=(edu.urls if edu else []),
        additional_instruction="Pass if at least one URL meets the credibility bar and is relevant to educational requirements."
    )


async def verify_certification(evaluator: Evaluator, parent_node, cert: Optional[CertificationInfo]) -> None:
    # Category node
    cert_node = evaluator.add_parallel(
        id="Certification_Requirements",
        desc="NCAA-mandated coaching certification requirement, including test name, administration details, and supporting URLs.",
        parent=parent_node,
        critical=False,
    )

    # Content stated in answer (critical group)
    cert_content = evaluator.add_parallel(
        id="cert_content",
        desc="Certification content stated in the answer",
        parent=cert_node,
        critical=True,
    )

    # Test name mentioned in the answer
    test_name_leaf = evaluator.add_leaf(
        id="Required_Certification_Test_Name",
        desc="Identifies the required NCAA certification as the NCAA Coaches Certification (Recruiting) Test (or an equivalently named NCAA recruiting certification test).",
        parent=cert_content,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies the required NCAA certification as the NCAA Coaches Certification (Recruiting) Test or an equivalently named NCAA recruiting certification test.",
        node=test_name_leaf,
        additional_instruction="Accept synonyms/variants like 'NCAA Recruiting Test', 'NCAA Coaches Recruiting Certification', or 'CRC exam' provided it clearly refers to the NCAA recruiting certification for coaches."
    )

    # Administration details mentioned in the answer (existence + specific detail supported)
    admin_req = evaluator.add_parallel(
        id="cert_admin_requirement",
        desc="Administration details about the NCAA certification test",
        parent=cert_content,
        critical=True,
    )

    admin_detail_present = evaluator.add_custom_node(
        result=bool(cert and cert.administration_details and len(cert.administration_details) > 0),
        id="Certification_Admin_Detail_Provided",
        desc="Answer provides at least one concrete detail about how the certification test is administered.",
        parent=admin_req,
        critical=True,
    )

    admin_detail_text = (cert.administration_details[0] if cert and cert.administration_details else "").strip()
    admin_details_leaf = evaluator.add_leaf(
        id="Certification_Test_Administration_Details",
        desc="Provides at least one concrete detail about how the certification test is administered (e.g., proctored vs. unproctored, online vs. in-person, who proctors/administers, timing/availability).",
        parent=admin_req,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The webpage confirms this administration detail about the NCAA recruiting certification test: {admin_detail_text}",
        node=admin_details_leaf,
        sources=(cert.urls if cert else []),
        additional_instruction="Look for explicit administration details such as proctoring by campus compliance, online delivery, exam availability windows, portals, or testing arrangements. The detail must match the claim text."
    )

    # Source support and credibility (critical group)
    cert_support = evaluator.add_parallel(
        id="cert_sources_support",
        desc="Certification requirements supported by credible sources",
        parent=cert_node,
        critical=True,
    )

    cert_urls_present = evaluator.add_custom_node(
        result=_has_urls(cert.urls if cert else []),
        id="Certification_Source_URLs_Provided",
        desc="At least one certification supporting reference URL is provided in the answer.",
        parent=cert_support,
        critical=True,
    )

    test_name_supported = evaluator.add_leaf(
        id="Required_Certification_Test_Name_Supported",
        desc="NCAA recruiting certification test requirement is supported by cited sources.",
        parent=cert_support,
        critical=True,
    )
    await evaluator.verify(
        claim="The webpage states or clearly implies that NCAA requires coaches to pass a recruiting certification test (e.g., NCAA Coaches Certification Recruiting Test) to engage in recruiting activities.",
        node=test_name_supported,
        sources=(cert.urls if cert else []),
        additional_instruction="Prefer NCAA or institutional compliance pages; accept credible secondary sources describing the NCAA recruiting certification requirement."
    )

    cert_sources_credible = evaluator.add_leaf(
        id="Certification_Sources_URLs_Credible",
        desc="Provides ≥1 supporting reference URL for the certification requirement and administration details; sources are credible/non-user-generated.",
        parent=cert_support,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is a credible, non-user-generated source (e.g., NCAA, .edu, .gov, recognized professional org) and discusses the NCAA recruiting certification requirement and/or test administration details.",
        node=cert_sources_credible,
        sources=(cert.urls if cert else []),
        additional_instruction="Pass if at least one URL meets the credibility bar and is relevant to the NCAA recruiting certification."
    )


async def verify_experience(evaluator: Evaluator, parent_node, exp: Optional[ExperienceInfo]) -> None:
    # Category node
    exp_node = evaluator.add_parallel(
        id="Experience_Requirements",
        desc="Typical prior coaching experience expectations, including prerequisite roles and years/duration, with supporting URLs.",
        parent=parent_node,
        critical=False,
    )

    # Content stated in the answer (critical group)
    exp_content = evaluator.add_parallel(
        id="exp_content",
        desc="Experience content stated in the answer",
        parent=exp_node,
        critical=True,
    )

    # Prerequisite entry-level roles (simple verify)
    roles_group = evaluator.add_parallel(
        id="exp_roles_requirement",
        desc="Prerequisite entry-level roles content",
        parent=exp_content,
        critical=True,
    )

    roles_leaf = evaluator.add_leaf(
        id="Prerequisite_Entry_Level_Roles",
        desc="Identifies typical entry-level prerequisite roles for the pathway (e.g., Graduate Assistant (GA) or entry-level assistant roles) as preparation for an assistant coach position.",
        parent=roles_group,
        critical=True,
    )
    roles_list = exp.entry_roles if exp else []
    roles_text = _join_list(_first_n(roles_list, 5))
    await evaluator.verify(
        claim=f"The answer identifies typical entry-level prerequisite roles such as {roles_text} for progressing toward an assistant coach position.",
        node=roles_leaf,
        additional_instruction="Pass if the answer explicitly lists GA or other entry-level assistant roles or their equivalents as stepping-stone positions."
    )

    # Typical years/duration (existence + content claim)
    years_group = evaluator.add_parallel(
        id="exp_years_requirement",
        desc="Typical years/duration of experience content",
        parent=exp_content,
        critical=True,
    )

    years_present = evaluator.add_custom_node(
        result=bool(exp and isinstance(exp.years_required, str) and exp.years_required.strip()),
        id="Typical_Years_or_Duration_Present",
        desc="Answer provides a typical/approximate duration in years (or a range) of prior coaching experience.",
        parent=years_group,
        critical=True,
    )

    years_leaf = evaluator.add_leaf(
        id="Typical_Years_or_Duration",
        desc="Provides a typical/approximate duration in years (or a range) of prior coaching experience before qualifying for an assistant coach position (acknowledging variance).",
        parent=years_group,
        critical=True,
    )
    yrs_text = (exp.years_required or "").strip() if exp else ""
    await evaluator.verify(
        claim=f"The answer states that typical prior coaching experience before qualifying for an assistant coach position is approximately {yrs_text}.",
        node=years_leaf,
        additional_instruction="Pass if the answer provides a plausible range or approximate number of years; do not demand exact uniformity across all schools."
    )

    # Advancement path (simple verify)
    path_group = evaluator.add_parallel(
        id="exp_path_requirement",
        desc="Typical coaching advancement path content",
        parent=exp_content,
        critical=True,
    )

    path_leaf = evaluator.add_leaf(
        id="Typical_Advancement_Path",
        desc="Describes the typical advancement path as stated in constraints: GA/entry-level assistant → position coach → coordinator → head coach.",
        parent=path_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer describes a typical coaching advancement path: GA or entry-level assistant → position coach → coordinator → head coach.",
        node=path_leaf,
        additional_instruction="Allow minor wording variations; the sequence should preserve the general ladder."
    )

    # Driver's license requirement (simple verify)
    drivers_group = evaluator.add_parallel(
        id="exp_driver_requirement",
        desc="Driver's license requirement content",
        parent=exp_content,
        critical=True,
    )
    drivers_leaf = evaluator.add_leaf(
        id="Drivers_License_For_Recruiting",
        desc="States that a valid driver's license is required for recruiting activities.",
        parent=drivers_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a valid driver's license is required for recruiting activities.",
        node=drivers_leaf,
        additional_instruction="Pass if the answer explicitly mentions a valid driver's license as a requirement tied to recruiting/travel."
    )

    # Optional notes (non-critical)
    exp_optional = evaluator.add_parallel(
        id="exp_optional",
        desc="Optional experience-related notes (non-critical)",
        parent=exp_node,
        critical=False,
    )

    playing_leaf = evaluator.add_leaf(
        id="Collegiate_Playing_Experience_Preferred",
        desc="Mentions that collegiate playing experience is often preferred though not universally required.",
        parent=exp_optional,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer notes that collegiate playing experience is often preferred but not universally required.",
        node=playing_leaf,
        additional_instruction="Minor variations allowed, as long as it indicates preference not strict requirement."
    )

    rules_leaf = evaluator.add_leaf(
        id="NCAA_Rules_Compliance_Knowledge",
        desc="Notes that head coaches/coordinators must demonstrate strong working knowledge of NCAA rules and compliance (as stated in constraints).",
        parent=exp_optional,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer indicates that head coaches or coordinators must demonstrate strong working knowledge of NCAA rules and compliance.",
        node=rules_leaf,
        additional_instruction="Pass if the answer explicitly mentions a strong working knowledge of NCAA rules for higher coaching roles."
    )

    winning_leaf = evaluator.add_leaf(
        id="Winning_Record_For_Head_Coach_Candidates",
        desc="Notes that head coach candidates are generally expected to demonstrate a winning record in previous coaching roles (as stated in constraints).",
        parent=exp_optional,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer notes that head coach candidates are generally expected to show a winning record in prior coaching roles.",
        node=winning_leaf,
        additional_instruction="Pass for reasonable paraphrases that communicate the same idea."
    )

    # Source support and credibility (critical group)
    exp_support = evaluator.add_parallel(
        id="exp_sources_support",
        desc="Experience requirements supported by credible sources",
        parent=exp_node,
        critical=True,
    )

    exp_urls_present = evaluator.add_custom_node(
        result=_has_urls(exp.urls if exp else []),
        id="Experience_Source_URLs_Provided",
        desc="At least one experience supporting reference URL is provided in the answer.",
        parent=exp_support,
        critical=True,
    )

    roles_supported = evaluator.add_leaf(
        id="Prerequisite_Entry_Level_Roles_Supported",
        desc="Prerequisite entry-level roles are supported by cited experience sources.",
        parent=exp_support,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The webpage supports that common stepping-stone roles before an assistant coach position include roles such as {roles_text}.",
        node=roles_supported,
        sources=(exp.urls if exp else []),
        additional_instruction="Look for GA, quality control, analyst, or similar roles listed as typical entry-level steps in job postings or official pages (.edu preferred)."
    )

    years_supported = evaluator.add_leaf(
        id="Typical_Years_or_Duration_Supported",
        desc="Typical years/duration of prior experience is supported by cited experience sources.",
        parent=exp_support,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The webpage indicates that typical prior coaching experience required before an assistant coach role is about {yrs_text}.",
        node=years_supported,
        sources=(exp.urls if exp else []),
        additional_instruction="Evidence can come from representative .edu job postings specifying years of experience for assistant football coach roles."
    )

    path_supported = evaluator.add_leaf(
        id="Typical_Advancement_Path_Supported",
        desc="Typical advancement path is supported by cited experience sources.",
        parent=exp_support,
        critical=True,
    )
    await evaluator.verify(
        claim="The webpage supports a typical coaching advancement path: GA or entry-level assistant → position coach → coordinator → head coach.",
        node=path_supported,
        sources=(exp.urls if exp else []),
        additional_instruction="Allow reasonable equivalents (e.g., analyst/quality control at entry level) as long as the ladder is preserved."
    )

    drivers_supported = evaluator.add_leaf(
        id="Drivers_License_For_Recruiting_Supported",
        desc="Driver's license requirement is supported by cited experience sources.",
        parent=exp_support,
        critical=True,
    )
    await evaluator.verify(
        claim="The webpage indicates that a valid driver's license is required for coaching recruiting activities or travel.",
        node=drivers_supported,
        sources=(exp.urls if exp else []),
        additional_instruction="Job postings or official HR pages often list this as a requirement; such evidence should pass."
    )

    exp_sources_credible = evaluator.add_leaf(
        id="Experience_Sources_URLs_Credible",
        desc="Provides ≥1 supporting reference URL for the experience requirements; sources are credible/non-user-generated.",
        parent=exp_support,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is a credible, non-user-generated source (e.g., .edu HR/job posting, NCAA, .gov, recognized organization) and discusses experience requirements relevant to assistant college football coaching roles.",
        node=exp_sources_credible,
        sources=(exp.urls if exp else []),
        additional_instruction="Pass if at least one URL meets the credibility bar and is relevant to experience requirements."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator with a parallel root
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

    # Extract structured info from the answer
    extracted: QualificationsExtraction = await evaluator.extract(
        prompt=prompt_extract_qualifications(),
        template_class=QualificationsExtraction,
        extraction_name="qualifications_extraction",
    )

    # Optionally record minimal GT or evaluation notes (not actual ground truth here)
    evaluator.add_custom_info(
        info={
            "note": "Evaluation script checks three categories: education, certification, experience; validates answer content and support via cited credible URLs."
        },
        info_type="evaluation_notes",
        info_name="eval_notes",
    )

    # Build and run verification for each category
    await verify_education(evaluator, root, extracted.education)
    await verify_certification(evaluator, root, extracted.certification)
    await verify_experience(evaluator, root, extracted.experience)

    # Return standardized summary
    return evaluator.get_summary()