import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "csun_career_director_credentials"
TASK_DESCRIPTION = (
    "I am researching best practices in university career services leadership and would like to examine the educational "
    "qualifications of career center directors at major public universities. Please identify the current Director of the "
    "Career Center at California State University, Northridge (CSUN) and provide a comprehensive verification of their "
    "educational credentials. Specifically: (1) Provide the director's full name, official title, and a reference URL from "
    "an official CSUN webpage confirming their current position. (2) Verify whether this director holds a doctoral degree "
    "(EdD or PhD). If so, identify the specific field of study or major for the doctoral degree and the full name of the "
    "university that awarded the doctoral degree. (3) Additionally, verify that this director holds a master's degree relevant "
    "to career counseling, student affairs, or a related field. Provide the field of study for the master's degree and the full "
    "name of the university that granted the master's degree. (4) Provide a reference URL that confirms the director's complete "
    "educational background, including both their doctoral and master's degrees. All information must be verifiable through official "
    "university websites, professional profiles, or other authoritative public sources."
)

# ----------------------------- Data Models ----------------------------- #
class DirectorInfo(BaseModel):
    name: Optional[str] = None
    official_title: Optional[str] = None
    csun_role_url: Optional[str] = None


class EducationInfo(BaseModel):
    masters_field: Optional[str] = None
    masters_university: Optional[str] = None
    doctoral_degree_type: Optional[str] = None  # e.g., "PhD", "EdD"
    doctoral_field: Optional[str] = None
    doctoral_university: Optional[str] = None
    education_background_urls: List[str] = Field(default_factory=list)


class BachelorInfo(BaseModel):
    bachelors_field: Optional[str] = None
    bachelors_university: Optional[str] = None
    bachelors_source_urls: List[str] = Field(default_factory=list)


class ExperienceInfo(BaseModel):
    experience_summary: Optional[str] = None
    experience_source_urls: List[str] = Field(default_factory=list)


class CredentialInfo(BaseModel):
    ncc_claimed: Optional[bool] = None
    ncc_source_urls: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    director: Optional[DirectorInfo] = None
    education: Optional[EducationInfo] = None
    bachelors: Optional[BachelorInfo] = None
    experience: Optional[ExperienceInfo] = None
    credentials: Optional[CredentialInfo] = None


# ----------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_director_and_credentials() -> str:
    return """
    Extract all requested information from the provided answer text. Return null for any item not explicitly stated.

    1) Director identification (official CSUN confirmation is required):
       - director.name: Full name of the current Director of the CSUN Career Center
       - director.official_title: Official title as shown on CSUN webpages (e.g., "Director, Career Center", "Interim Director")
       - director.csun_role_url: A URL to an official CSUN webpage that confirms the person and the title (must be a csun.edu domain URL)
    
    2) Education (use authoritative URLs; these may be official university sites, CSUN profile pages, or authoritative professional profiles):
       - education.masters_field: Field/major of the master's degree
       - education.masters_university: Full name of the university that granted the master's degree
       - education.doctoral_degree_type: If a doctoral degree is mentioned, specify "PhD" or "EdD"; else return null
       - education.doctoral_field: Field/major of the doctoral degree (if present)
       - education.doctoral_university: Full name of the university that granted the doctoral degree (if present)
       - education.education_background_urls: List of one or more URLs that explicitly confirm the director's educational background (must include the master's degree, and doctoral degree if present). Extract only URLs that appear in the answer.

    3) Bachelor's degree (if mentioned; optional but helpful):
       - bachelors.bachelors_field: Field/major of the bachelor's degree
       - bachelors.bachelors_university: University that granted the bachelor's degree
       - bachelors.bachelors_source_urls: All URLs in the answer that confirm the bachelor's degree

    4) Experience requirement (authoritative verification required):
       - experience.experience_summary: Summary phrase or data that indicates at least 5 years of progressively responsible experience in higher education, career services, student affairs, or a related field (e.g., "10+ years")
       - experience.experience_source_urls: URLs in the answer that confirm the experience requirement from authoritative sources

    5) NCC credential (only if mentioned):
       - credentials.ncc_claimed: true if the answer claims the director holds NCC; false if explicitly claimed that they do not hold NCC; null if not mentioned
       - credentials.ncc_source_urls: URLs that verify NCC credential status (if claimed)

    URL extraction rules:
    - Extract only valid URLs that are explicitly present in the answer (plain URLs or markdown links).
    - Do not invent or infer any URLs.
    - If a URL is missing protocol, prepend http://
    """


# ----------------------------- Helper Functions ----------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _combine_sources(primary: List[str], fallback: List[str]) -> List[str]:
    s = _non_empty_urls(primary)
    if not s:
        s = _non_empty_urls(fallback)
    return s


# ----------------------------- Verification Builders ----------------------------- #
async def build_director_identification(evaluator: Evaluator, root, data: FullExtraction) -> None:
    node = evaluator.add_parallel(
        id="DirectorIdentification",
        desc="Correctly identify the current Director of the Career Center at CSUN and cite an official CSUN page confirming the role",
        parent=root,
        critical=True
    )

    director = data.director or DirectorInfo()

    # Existence check for CSUN official role URL
    role_url = _safe_str(director.csun_role_url)
    role_url_exists = bool(role_url)
    prereq_role_url = evaluator.add_custom_node(
        result=role_url_exists,
        id="csun_role_url_provided",
        desc="An official CSUN role confirmation URL is provided (csun.edu domain)",
        parent=node,
        critical=True
    )

    # DirectorName leaf
    leaf_name = evaluator.add_leaf(
        id="DirectorName",
        desc="Provide the director's full name",
        parent=node,
        critical=True
    )
    name_claim = f"The CSUN webpage confirms that the person named '{_safe_str(director.name)}' is listed on the page."
    await evaluator.verify(
        claim=name_claim,
        node=leaf_name,
        sources=role_url,
        additional_instruction="Verify that the person's name appears on this official CSUN page. Allow minor variations (e.g., middle initials).",
        extra_prerequisites=[prereq_role_url]
    )

    # OfficialTitle leaf
    leaf_title = evaluator.add_leaf(
        id="OfficialTitle",
        desc="Provide the director's official title",
        parent=node,
        critical=True
    )
    title_claim = (
        f"The official CSUN page shows that '{_safe_str(director.name)}' holds the title '{_safe_str(director.official_title)}', "
        "which corresponds to serving as the Director of the CSUN Career Center (allow synonyms like 'Director, Career Center', 'Interim Director', or 'Executive Director' if clearly the head of the center)."
    )
    await evaluator.verify(
        claim=title_claim,
        node=leaf_title,
        sources=role_url,
        additional_instruction="Confirm the title corresponds to serving as the head/director of the CSUN Career Center; allow reasonable synonyms.",
        extra_prerequisites=[prereq_role_url]
    )

    # CSUNRoleConfirmationURL leaf
    leaf_role_url = evaluator.add_leaf(
        id="CSUNRoleConfirmationURL",
        desc="Provide a reference URL from an official CSUN webpage that confirms the person and title as the current director",
        parent=node,
        critical=True
    )
    role_url_claim = (
        f"This webpage is an official CSUN page (csun.edu domain) that confirms '{_safe_str(director.name)}' "
        f"is the current Director of the CSUN Career Center with title '{_safe_str(director.official_title)}'."
    )
    await evaluator.verify(
        claim=role_url_claim,
        node=leaf_role_url,
        sources=role_url,
        additional_instruction="Confirm the page is on csun.edu and explicitly states the person's role as director of the CSUN Career Center.",
        extra_prerequisites=[prereq_role_url]
    )


async def build_education_and_credentials(evaluator: Evaluator, root, data: FullExtraction) -> None:
    qual_node = evaluator.add_parallel(
        id="QualificationsAndCredentialsVerification",
        desc="Verify the director's education and constraint-required qualifications with authoritative sources",
        parent=root,
        critical=False  # Parent allows mixture; critical children gate failure
    )

    # EducationVerification subtree (critical)
    edu_node = evaluator.add_parallel(
        id="EducationVerification",
        desc="Verify the director's master's degree (required) and doctoral status (optional) and provide a source confirming education history",
        parent=qual_node,
        critical=True
    )

    edu = data.education or EducationInfo()
    edu_urls = _non_empty_urls(edu.education_background_urls)

    # Existence check for education background URLs (critical for education verification)
    edu_urls_provided = evaluator.add_custom_node(
        result=bool(edu_urls),
        id="EducationBackgroundURLProvided",
        desc="At least one authoritative URL is provided that confirms the director's educational background",
        parent=edu_node,
        critical=True
    )

    # MastersDegreeRequirement subtree (critical)
    masters_node = evaluator.add_parallel(
        id="MastersDegreeRequirement",
        desc="Verify the director holds a master's degree relevant to the requested fields; provide field/major and awarding university",
        parent=edu_node,
        critical=True
    )

    # MastersFieldOfStudy leaf
    masters_field_leaf = evaluator.add_leaf(
        id="MastersFieldOfStudy",
        desc="Provide the master's field of study/major as stated in sources",
        parent=masters_node,
        critical=True
    )
    masters_field_claim = f"The director's master's degree field/major is '{_safe_str(edu.masters_field)}' as shown by the provided authoritative source(s)."
    await evaluator.verify(
        claim=masters_field_claim,
        node=masters_field_leaf,
        sources=edu_urls,
        additional_instruction="Confirm the master's field/major exactly or via reasonable equivalence on the source(s).",
        extra_prerequisites=[edu_urls_provided]
    )

    # MastersGrantingInstitution leaf
    masters_univ_leaf = evaluator.add_leaf(
        id="MastersGrantingInstitution",
        desc="Provide the full name of the university that granted the master's degree",
        parent=masters_node,
        critical=True
    )
    masters_univ_claim = f"The director's master's degree was awarded by '{_safe_str(edu.masters_university)}' according to the authoritative source(s)."
    await evaluator.verify(
        claim=masters_univ_claim,
        node=masters_univ_leaf,
        sources=edu_urls,
        additional_instruction="Confirm the awarding university name matches the source; allow minor name variants (e.g., abbreviation vs. full name).",
        extra_prerequisites=[edu_urls_provided]
    )

    # MastersRelevanceToAllowedFields leaf (logical check)
    masters_relevance_leaf = evaluator.add_leaf(
        id="MastersRelevanceToAllowedFields",
        desc="Confirm the master's field is within counseling, student affairs, higher education administration, psychology, or a related field",
        parent=masters_node,
        critical=True
    )
    masters_relevance_claim = (
        f"The master's degree field '{_safe_str(edu.masters_field)}' is relevant to career counseling, student affairs, "
        "higher education administration, psychology, or a closely related field."
    )
    await evaluator.verify(
        claim=masters_relevance_claim,
        node=masters_relevance_leaf,
        additional_instruction=(
            "Judge relevance logically. Consider synonymous or closely related programs as relevant, e.g., Counselor Education, Counseling Psychology, "
            "Higher Education, Student Affairs Administration, College Counseling, Educational Psychology."
        ),
        extra_prerequisites=[masters_field_leaf]
    )

    # EducationBackgroundSourceURL verification leaf (critical)
    edu_bg_verify_leaf = evaluator.add_leaf(
        id="EducationBackgroundSourceURL",
        desc="Provide an authoritative URL that confirms the director's educational background, including the master's degree and any doctoral degree if present",
        parent=edu_node,
        critical=True
    )
    edu_bg_claim = (
        "These provided authoritative URL(s) explicitly confirm the director's educational background, including the master's degree, "
        "and the doctoral degree if one is present."
    )
    await evaluator.verify(
        claim=edu_bg_claim,
        node=edu_bg_verify_leaf,
        sources=edu_urls,
        additional_instruction="The URL(s) must clearly confirm the master's degree. If a doctoral degree is present in the sources, it should also be confirmed.",
        extra_prerequisites=[edu_urls_provided]
    )

    # DoctoralDegreeStatusAndDetails subtree (optional, non-critical)
    doctoral_node = evaluator.add_parallel(
        id="DoctoralDegreeStatusAndDetails",
        desc="Verify whether the director holds an EdD or PhD; if yes, provide doctoral field/major and awarding university (supported by sources)",
        parent=qual_node,
        critical=False
    )

    # Doctoral presence leaf
    doctoral_presence_leaf = evaluator.add_leaf(
        id="DoctoralDegreePresence",
        desc="Verify whether the director holds a doctoral degree (EdD or PhD)",
        parent=doctoral_node,
        critical=False
    )
    if _safe_str(edu.doctoral_degree_type):
        doc_presence_claim = (
            f"The director holds a doctoral degree of type '{_safe_str(edu.doctoral_degree_type)}' according to the provided authoritative source(s)."
        )
    else:
        doc_presence_claim = (
            "According to the provided authoritative source(s), the director does not appear to hold a doctoral degree (EdD or PhD)."
        )
    await evaluator.verify(
        claim=doc_presence_claim,
        node=doctoral_presence_leaf,
        sources=edu_urls,
        additional_instruction="Confirm presence or absence. If present, the type must be clearly indicated; if absent, sources should not list any doctoral degree.",
        extra_prerequisites=[edu_urls_provided]
    )

    # Doctoral field leaf (depends on presence)
    doctoral_field_leaf = evaluator.add_leaf(
        id="DoctoralFieldOfStudy",
        desc="If a doctoral degree is present, provide the doctoral field/major as stated in sources",
        parent=doctoral_node,
        critical=False
    )
    doctoral_field_claim = f"The doctoral degree field/major is '{_safe_str(edu.doctoral_field)}' according to the source(s)."
    await evaluator.verify(
        claim=doctoral_field_claim,
        node=doctoral_field_leaf,
        sources=edu_urls,
        additional_instruction="Only applicable if a doctoral degree is present; otherwise should be skipped.",
        extra_prerequisites=[doctoral_presence_leaf]
    )

    # Doctoral awarding university leaf (depends on presence)
    doctoral_univ_leaf = evaluator.add_leaf(
        id="DoctoralGrantingInstitution",
        desc="If a doctoral degree is present, provide the full name of the university that granted the doctoral degree",
        parent=doctoral_node,
        critical=False
    )
    doctoral_univ_claim = f"The doctoral degree was awarded by '{_safe_str(edu.doctoral_university)}' according to the source(s)."
    await evaluator.verify(
        claim=doctoral_univ_claim,
        node=doctoral_univ_leaf,
        sources=edu_urls,
        additional_instruction="Only applicable if a doctoral degree is present; allow minor university name variants.",
        extra_prerequisites=[doctoral_presence_leaf]
    )

    # Bachelor's evidence (optional, non-critical)
    bachelors_node = evaluator.add_parallel(
        id="BachelorsDegreeEvidence",
        desc="Provide evidence of the director's bachelor's degree (field/major and awarding institution)",
        parent=qual_node,
        critical=False
    )
    bachelors = data.bachelors or BachelorInfo()
    bachelors_sources = _combine_sources(bachelors.bachelors_source_urls, edu_urls)

    bachelors_field_leaf = evaluator.add_leaf(
        id="BachelorsFieldOfStudy",
        desc="Bachelor's degree field/major is supported by source(s)",
        parent=bachelors_node,
        critical=False
    )
    bachelors_field_claim = f"The bachelor's degree field/major is '{_safe_str(bachelors.bachelors_field)}' according to the source(s)."
    await evaluator.verify(
        claim=bachelors_field_claim,
        node=bachelors_field_leaf,
        sources=bachelors_sources,
        additional_instruction="Confirm the bachelor's field/major if provided; otherwise this may fail due to lack of evidence."
    )

    bachelors_univ_leaf = evaluator.add_leaf(
        id="BachelorsGrantingInstitution",
        desc="Bachelor's degree awarding institution is supported by source(s)",
        parent=bachelors_node,
        critical=False
    )
    bachelors_univ_claim = f"The bachelor's degree was awarded by '{_safe_str(bachelors.bachelors_university)}' according to the source(s)."
    await evaluator.verify(
        claim=bachelors_univ_claim,
        node=bachelors_univ_leaf,
        sources=bachelors_sources,
        additional_instruction="Confirm the bachelor's awarding institution; allow minor name variants."
    )

    # Experience requirement (critical within qualifications)
    exp_leaf_parent = evaluator.add_parallel(
        id="ExperienceRequirementGroup",
        desc="Evidence of at least 5 years of progressively responsible experience in relevant fields",
        parent=qual_node,
        critical=True
    )
    exp = data.experience or ExperienceInfo()
    exp_urls = _non_empty_urls(exp.experience_source_urls)

    exp_urls_provided = evaluator.add_custom_node(
        result=bool(exp_urls),
        id="ExperienceURLsProvided",
        desc="Authoritative experience URL(s) provided",
        parent=exp_leaf_parent,
        critical=True
    )

    exp_leaf = evaluator.add_leaf(
        id="ExperienceRequirement",
        desc="Provide evidence (with an authoritative URL) that the director has at least 5 years of progressively responsible experience in relevant fields",
        parent=exp_leaf_parent,
        critical=True
    )
    exp_claim = (
        "These authoritative source(s) confirm the director has at least 5 years of progressively responsible experience "
        "in higher education, career services, student affairs, or a related field."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=exp_urls,
        additional_instruction=(
            "Look for tenure/years of experience, progressive roles, and relevant domains (higher education, student affairs, career services). "
            "If multiple roles cumulatively exceed 5 years, that satisfies the requirement."
        ),
        extra_prerequisites=[exp_urls_provided]
    )

    # NCC credential verification (only if claimed; non-critical)
    ncc_node = evaluator.add_parallel(
        id="NCCCredentialIfClaimed",
        desc="If the response claims the director holds the NCC credential, verify NCC status and master's major in counseling",
        parent=qual_node,
        critical=False
    )
    cred = data.credentials or CredentialInfo()
    ncc_urls = _non_empty_urls(cred.ncc_source_urls)

    ncc_present_leaf = evaluator.add_leaf(
        id="NCCCredentialStatus",
        desc="Verify NCC credential status if claimed",
        parent=ncc_node,
        critical=False
    )
    if cred.ncc_claimed:
        ncc_claim = "The director holds the NCC (National Certified Counselor) credential as verified by the provided authoritative source(s)."
    else:
        ncc_claim = "The director does not claim or does not hold the NCC credential per the provided sources."
    await evaluator.verify(
        claim=ncc_claim,
        node=ncc_present_leaf,
        sources=ncc_urls if cred.ncc_claimed else None,
        additional_instruction=(
            "If NCC is claimed, the source(s) must explicitly show NCC status (e.g., NBCC registry or official profile). "
            "If not claimed, this can pass without URLs."
        )
    )

    ncc_counseling_leaf = evaluator.add_leaf(
        id="NCCMastersCounselingCheck",
        desc="If NCC is claimed, confirm the master's degree major is in counseling",
        parent=ncc_node,
        critical=False
    )
    ncc_counseling_claim = f"The master's degree field '{_safe_str(edu.masters_field)}' is in counseling (or an equivalent counseling program), satisfying NCC-related constraints."
    await evaluator.verify(
        claim=ncc_counseling_claim,
        node=ncc_counseling_leaf,
        additional_instruction=(
            "Accept Counselor Education, Clinical Mental Health Counseling, Counseling Psychology, School Counseling, or similar counseling programs."
        ),
        extra_prerequisites=[ncc_present_leaf, masters_field_leaf]
    )


# ----------------------------- Main Evaluation Entry ----------------------------- #
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
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extracted: FullExtraction = await evaluator.extract(
        prompt=prompt_extract_director_and_credentials(),
        template_class=FullExtraction,
        extraction_name="director_and_credentials"
    )

    # Build verification tree
    await build_director_identification(evaluator, root, extracted)
    await build_education_and_credentials(evaluator, root, extracted)

    return evaluator.get_summary()