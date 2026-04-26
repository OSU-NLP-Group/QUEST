import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_div1_athletics_admin_job"
TASK_DESCRIPTION = (
    "You are exploring career opportunities in collegiate athletic administration in North Carolina. "
    "Identify one current athletic administration position (such as Assistant Athletic Director, Director of Compliance, "
    "or Athletic Operations Coordinator) at a Division I NCAA institution located in North Carolina. For the identified position, "
    "provide the following information: (1) Institution name and verification that it is an NCAA Division I institution, "
    "(2) Complete position title and URL link to the official job posting, (3) Minimum education requirement (degree level required), "
    "(4) Minimum experience requirement, including the number of years and type of experience required (e.g., athletics administration, NCAA compliance, etc.), "
    "(5) Evidence that the position requires knowledge of NCAA rules, regulations, or compliance, and (6) Salary or compensation information (if disclosed in the posting) "
    "or comparable salary range for similar positions. The position must be an administrative role within the athletic department, not a coaching position. "
    "All information must be verified with appropriate URL references."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JobPostingExtraction(BaseModel):
    # Identification and primary posting
    posting_url: Optional[str] = None
    position_title: Optional[str] = None
    institution_name: Optional[str] = None

    # Location and DI verification sources
    location_text: Optional[str] = None   # e.g., "Durham, NC"
    location_source_urls: List[str] = Field(default_factory=list)
    di_source_urls: List[str] = Field(default_factory=list)

    # Requirements and qualifications
    minimum_education_text: Optional[str] = None
    min_education_urls: List[str] = Field(default_factory=list)

    min_experience_years_text: Optional[str] = None   # e.g., "2 years", "1-3 years", "three (3) years"
    min_experience_domain_text: Optional[str] = None  # e.g., "athletics administration", "NCAA compliance"
    min_experience_urls: List[str] = Field(default_factory=list)

    rules_knowledge_text: Optional[str] = None
    rules_knowledge_urls: List[str] = Field(default_factory=list)

    # Compensation
    compensation_text: Optional[str] = None           # either explicit salary in the posting or a comparable range
    compensation_urls: List[str] = Field(default_factory=list)

    # Evidence the posting is current/open
    posting_current_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_job_posting() -> str:
    return """
    From the answer, extract exactly one identified athletic administration (non-coaching) job posting at an NCAA Division I institution in North Carolina, plus all required details and citation URLs. Extract only what is explicitly present in the answer text.

    Return a JSON object with the following fields:
    - posting_url: string or null. The direct URL to the official job posting (institutional HR, official athletics site, or the institution's ATS such as Workday/PeopleAdmin/Taleo/iCIMS/BrassRing). Do NOT use third-party aggregator sites unless they are the institution's official ATS.
    - position_title: string or null. The complete position title as shown on the posting.
    - institution_name: string or null. The institution's name as shown on the posting or institutional source.

    - location_text: string or null. The location as stated (e.g., city, state).
    - location_source_urls: array of strings. URLs explicitly in the answer that substantiate the institution is in North Carolina. Use the posting URL if it explicitly states NC; otherwise, use institutional sources.

    - di_source_urls: array of strings. URLs explicitly in the answer that substantiate the institution is NCAA Division I (e.g., official athletics site, NCAA website, or recognized conference site).

    - minimum_education_text: string or null. The minimum education requirement text exactly as presented in the answer (e.g., "Bachelor's degree required").
    - min_education_urls: array of strings. URLs explicitly in the answer supporting the minimum education requirement.

    - min_experience_years_text: string or null. The required minimum number of years of experience as text (e.g., "2 years", "one to three (1-3) years"). If a range is given, keep it as in the answer.
    - min_experience_domain_text: string or null. The required domain/type of experience (e.g., "athletics administration", "NCAA compliance", "college athletics").
    - min_experience_urls: array of strings. URLs explicitly in the answer supporting the minimum experience requirement.

    - rules_knowledge_text: string or null. The exact or paraphrased phrase from the answer indicating the posting requires knowledge of NCAA rules, regulations, or compliance.
    - rules_knowledge_urls: array of strings. URLs explicitly in the answer that support the requirement of NCAA rules knowledge.

    - compensation_text: string or null. Either the specific salary/compensation disclosed in the posting OR a comparable salary range for similar positions (as provided in the answer). Keep the text as provided in the answer.
    - compensation_urls: array of strings. URLs explicitly in the answer that support the salary/compensation or the comparable salary range. Use official postings or institutional sources only.

    - posting_current_urls: array of strings. URLs explicitly in the answer that indicate the posting is current/open/active (often the same as the posting_url).

    SPECIAL RULES:
    - Only extract URLs that are explicitly present in the answer. If the answer mentions a source but gives no URL, do not fabricate a URL; leave the field empty or the url list empty.
    - If a field is not mentioned in the answer, set it to null (or an empty array for lists).
    - Do not include commentary. Only extract the values exactly as in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(primary: Optional[str], extras: List[str]) -> List[str]:
    urls: List[str] = []
    if primary and primary.strip():
        urls.append(primary.strip())
    for u in extras:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_identify_qualifying_posting(
    evaluator: Evaluator,
    parent_node,
    info: JobPostingExtraction
) -> None:
    """
    Build and verify the 'identify_qualifying_posting' subtree.
    """
    node = evaluator.add_parallel(
        id="identify_qualifying_posting",
        desc="Identify a qualifying current job posting at a North Carolina NCAA Division I institution (administrative, not coaching)",
        parent=parent_node,
        critical=True
    )

    # 1) official_job_posting_url
    leaf_official_url = evaluator.add_leaf(
        id="official_job_posting_url",
        desc="Provide a direct URL to the official job posting (official posting page or institutional HR/athletics posting page)",
        parent=node,
        critical=True
    )
    title_part = f" titled '{info.position_title}'" if info.position_title else ""
    inst_part = f" at '{info.institution_name}'" if info.institution_name else ""
    claim_official = (
        f"This URL is an official institutional job posting page{title_part}{inst_part}, "
        "not a third-party aggregator. It is hosted by the institution (or its official applicant tracking system)."
    )
    await evaluator.verify(
        claim=claim_official,
        node=leaf_official_url,
        sources=info.posting_url,
        additional_instruction=(
            "Accept institutional HR domains and official ATS platforms (e.g., Workday, PeopleAdmin, iCIMS, Taleo, BrassRing) "
            "if they clearly host the institution's postings. Reject news articles or general aggregator boards."
        )
    )

    # 2) posting_is_current
    leaf_current = evaluator.add_leaf(
        id="posting_is_current",
        desc="Verify from the posting page that the position is current/open/active (e.g., not marked closed/filled/expired; date/status shown on the posting)",
        parent=node,
        critical=True
    )
    today = datetime.utcnow().date().isoformat()
    claim_current = (
        f"As of {today}, this job posting is currently open/active (not closed/filled/expired). "
        "There should be signals like an Apply button, 'Open until filled', active status, or a recent posting/open date, and no explicit 'Closed/Expired'."
    )
    current_sources = _combine_sources(info.posting_url, info.posting_current_urls)
    await evaluator.verify(
        claim=claim_current,
        node=leaf_current,
        sources=current_sources or info.posting_url,
        additional_instruction="Prefer explicit language on the page indicating open/active status; absence of 'expired/closed' and presence of 'Apply' typically indicate current."
    )

    # 3) institution_name_with_citation
    leaf_institution = evaluator.add_leaf(
        id="institution_name_with_citation",
        desc="Provide the institution name with a citation URL from the official posting or an institutional source",
        parent=node,
        critical=True
    )
    name_for_claim = info.institution_name or "the institution"
    claim_inst = f"The official posting or institutional source clearly shows the institution name as '{name_for_claim}'."
    await evaluator.verify(
        claim=claim_inst,
        node=leaf_institution,
        sources=info.posting_url,
        additional_instruction="Look for employer, institution, or athletics department name on the posting."
    )

    # 4) institution_in_north_carolina_with_citation
    leaf_nc = evaluator.add_leaf(
        id="institution_in_north_carolina_with_citation",
        desc="Verify the institution is located in North Carolina with a citation URL from an institutional source (or the official posting if it states location)",
        parent=node,
        critical=True
    )
    claim_nc = f"The institution {name_for_claim} is located in North Carolina (NC)."
    nc_sources = _combine_sources(info.posting_url, info.location_source_urls)
    await evaluator.verify(
        claim=claim_nc,
        node=leaf_nc,
        sources=nc_sources or info.posting_url,
        additional_instruction="Accept official institutional pages or the job posting if it explicitly shows a North Carolina city/state."
    )

    # 5) division_i_verification_with_citation
    leaf_di = evaluator.add_leaf(
        id="division_i_verification_with_citation",
        desc="Verify the institution is NCAA Division I with a citation URL from an authoritative institutional/official athletics or NCAA/recognized conference source",
        parent=node,
        critical=True
    )
    claim_di = f"The institution {name_for_claim} competes in NCAA Division I."
    await evaluator.verify(
        claim=claim_di,
        node=leaf_di,
        sources=info.di_source_urls if info.di_source_urls else None,
        additional_instruction="Prefer NCAA.org pages, the institution's official athletics website, or recognized conference websites explicitly stating Division I membership."
    )

    # 6) position_title_with_citation
    leaf_title = evaluator.add_leaf(
        id="position_title_with_citation",
        desc="Provide the complete position title as shown on the official posting, with a citation URL to the posting",
        parent=node,
        critical=True
    )
    title_for_claim = info.position_title or "the position"
    claim_title = f"The complete position title on the official posting is '{title_for_claim}'."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=info.posting_url,
        additional_instruction="Match the posting's displayed job title exactly or with minor acceptable variations (case, punctuation)."
    )

    # 7) administrative_not_coaching_with_citation
    leaf_admin = evaluator.add_leaf(
        id="administrative_not_coaching_with_citation",
        desc="Verify the role is an athletic-department administrative position and not a coaching position, with a citation URL to the posting text",
        parent=node,
        critical=True
    )
    claim_admin = (
        "This role is an athletic-department administrative position (e.g., administration, operations, compliance, academics) "
        "and is NOT a coaching position."
    )
    await evaluator.verify(
        claim=claim_admin,
        node=leaf_admin,
        sources=info.posting_url,
        additional_instruction=(
            "Check job family/duties. If the role is titled or described as 'Coach' (Head/Assistant) or includes primary coaching responsibilities, "
            "it is coaching and should be rejected. Administrative examples: compliance, operations, academics, business/finance, development, marketing."
        )
    )


async def verify_required_details(
    evaluator: Evaluator,
    parent_node,
    info: JobPostingExtraction
) -> None:
    """
    Build and verify the 'extract_required_posting_details' subtree.
    """
    node = evaluator.add_parallel(
        id="extract_required_posting_details",
        desc="Extract and report required minimum qualifications and compliance/compensation details for the identified position, with acceptable citations",
        parent=parent_node,
        critical=True
    )

    # 1) minimum_education_bachelors_with_citation
    leaf_edu = evaluator.add_leaf(
        id="minimum_education_bachelors_with_citation",
        desc="State the minimum education requirement and confirm it requires at least a Bachelor's degree, with a citation URL to the official posting/institutional source",
        parent=node,
        critical=True
    )
    claim_edu = (
        "The minimum education requirement for this position is at least a Bachelor's degree (Bachelor's or higher is required; "
        "phrases like 'Master's preferred' still imply Bachelor's required)."
    )
    edu_sources = _combine_sources(info.posting_url, info.min_education_urls)
    await evaluator.verify(
        claim=claim_edu,
        node=leaf_edu,
        sources=edu_sources or info.posting_url,
        additional_instruction="Reject if the page only lists degrees as 'preferred' without any degree required; accept Bachelor's, Master's, JD, PhD, etc., as satisfying 'at least a Bachelor's'."
    )

    # 2) minimum_experience_years_and_type_with_citation
    leaf_exp = evaluator.add_leaf(
        id="minimum_experience_years_and_type_with_citation",
        desc="State the minimum experience requirement including (a) number of years and (b) type/domain (athletics administration, NCAA compliance, or related field), with a citation URL to the official posting/institutional source",
        parent=node,
        critical=True
    )
    years_text = info.min_experience_years_text or "a specified minimum number of years"
    domain_text = info.min_experience_domain_text or "a specified relevant domain"
    claim_exp = (
        f"The posting states a minimum experience requirement including both the number of years (e.g., '{years_text}') "
        f"and the type/domain (e.g., '{domain_text}'), and these are minimum requirements (not merely preferred)."
    )
    exp_sources = _combine_sources(info.posting_url, info.min_experience_urls)
    await evaluator.verify(
        claim=claim_exp,
        node=leaf_exp,
        sources=exp_sources or info.posting_url,
        additional_instruction="Look for 'minimum' or 'required' experience language, not just 'preferred'. Accept reasonable variations (e.g., '1-3 years')."
    )

    # 3) explicit_ncaa_rules_knowledge_with_citation
    leaf_rules = evaluator.add_leaf(
        id="explicit_ncaa_rules_knowledge_with_citation",
        desc="Provide evidence that the posting explicitly requires knowledge of NCAA rules/regulations/compliance, with a citation URL to the official posting/institutional source",
        parent=node,
        critical=True
    )
    claim_rules = "The posting explicitly requires knowledge of NCAA rules, regulations, and/or compliance."
    rules_sources = _combine_sources(info.posting_url, info.rules_knowledge_urls)
    await evaluator.verify(
        claim=claim_rules,
        node=leaf_rules,
        sources=rules_sources or info.posting_url,
        additional_instruction="Look for phrases like 'knowledge of NCAA rules', 'NCAA compliance', 'adherence to NCAA regulations', etc."
    )

    # 4) compensation_or_comparable_with_citation
    leaf_comp = evaluator.add_leaf(
        id="compensation_or_comparable_with_citation",
        desc="Provide salary/compensation if disclosed in the posting OR provide a comparable salary range for similar positions, with citation URL(s) from official job postings or institutional sources",
        parent=node,
        critical=True
    )
    comp_text = info.compensation_text or "a stated salary/compensation or a comparable salary range for similar positions"
    claim_comp = (
        f"The provided sources support the following salary/compensation information for this role or comparable positions: {comp_text}"
    )
    comp_sources = _combine_sources(info.posting_url, info.compensation_urls)
    await evaluator.verify(
        claim=claim_comp,
        node=leaf_comp,
        sources=comp_sources if comp_sources else None,
        additional_instruction=(
            "If the exact posting discloses salary, verify that amount. If not, verify a comparable salary range using official sources or official postings. "
            "Reject if sources are non-official or do not substantiate the claimed figure/range."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the NC Division I athletics administration job task.
    """
    # Initialize evaluator (root is non-critical by framework design; we will add critical child nodes)
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

    # Extract structured info about the single job posting from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_job_posting(),
        template_class=JobPostingExtraction,
        extraction_name="job_posting_extraction"
    )

    # Build and verify tree per rubric
    await verify_identify_qualifying_posting(evaluator, root, extracted_info)
    await verify_required_details(evaluator, root, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()