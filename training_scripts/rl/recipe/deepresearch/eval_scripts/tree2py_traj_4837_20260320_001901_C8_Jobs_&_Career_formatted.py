import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entry_level_ed_multi_institutions"
TASK_DESCRIPTION = """
A recent graduate with a Bachelor's degree in Education (GPA 2.8) and no current teaching certification is seeking entry-level employment opportunities in the education sector across multiple states. The candidate is willing to pursue state certification through alternative certification programs if available at the hiring institution.

Identify one appropriate entry-level position at each of the following four educational institutions that matches the candidate's current qualifications or for which the candidate could qualify through the institution's programs:

1. A Texas school district: Either Katy Independent School District (Katy ISD) or Fort Bend Independent School District (Fort Bend ISD)
2. Carroll County Public Schools in Maryland
3. Dallas College (including positions available at the Mountain View campus)
4. Wake County Public Schools in North Carolina

For each identified position, provide:
- The exact position title as listed in the official job posting
- The direct URL to the official job posting on the institution's career website
- The minimum educational requirement specified for the position
- The certification or licensure requirements, including whether the position requires immediate certification or if the institution offers alternative certification pathways for uncertified candidates with bachelor's degrees
- Any additional relevant details such as salary information (for the Maryland position), campus location (for the Dallas College position), or specific program requirements

All positions must be currently available or regularly posted positions that the candidate with a bachelor's degree but no certification can apply for or become eligible for through institutional pathways.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    institution: Optional[str] = None  # e.g., "Katy ISD", "Fort Bend ISD", "Carroll County Public Schools"
    position_title: Optional[str] = None  # official job title as listed
    job_url: Optional[str] = None  # direct posting URL on official careers site
    min_education: Optional[str] = None  # quoted text or summary of minimum education requirement
    certification_requirement: Optional[str] = None  # quoted text or summary of certification/licensure requirement
    certification_pathways: Optional[str] = None  # any notes indicating ACP/residency/alternative pathways/willingness to sponsor
    salary_info: Optional[str] = None  # salary or hourly rate (esp. for MD)
    position_type: Optional[str] = None  # adjunct/full-time/part-time/staff/non-credit/etc. (esp. for Dallas College)
    campus_location: Optional[str] = None  # e.g., "Mountain View campus" (Dallas College)
    details: Optional[str] = None  # additional relevant details (duties, dept, grade level, etc.)
    extra_urls: List[str] = Field(default_factory=list)  # any additional official URLs explicitly cited in the answer


class JobSearchExtraction(BaseModel):
    texas_position: Optional[PositionItem] = None
    maryland_position: Optional[PositionItem] = None  # Carroll County Public Schools (MD)
    dallas_position: Optional[PositionItem] = None  # Dallas College
    nc_position: Optional[PositionItem] = None  # Wake County Public Schools (NC)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract the four institution-specific positions exactly as presented in the answer.
    For each of the following, return a JSON object with the specified fields. If a field is not present in the answer, set it to null (or [] for arrays).
    
    Institutions and required object keys:
    - texas_position:
        institution: The district selected (must be either "Katy Independent School District (Katy ISD)" or "Fort Bend Independent School District (Fort Bend ISD)"; use the short common form like "Katy ISD" or "Fort Bend ISD" if present)
        position_title: Official job title as listed in the posting
        job_url: Direct link to the official careers/job posting page for this position (not general homepages or third-party aggregators unless it is the district’s official ATS instance)
        min_education: The minimum educational requirement as stated
        certification_requirement: The certification/licensure requirement status as stated (e.g., "Texas teacher certification required", "ACP candidates accepted", "no certificate required")
        certification_pathways: Any mention of alternative certification (ACP), intern/probationary permits, residency, or institutional sponsorship pathways
        details: Any additional relevant details mentioned in the answer (duties, department, level/subject, program info)
        salary_info: If any pay information was provided for this Texas role (optional)
        position_type: Employment type if mentioned (optional)
        campus_location: Not applicable typically; set null unless specifically stated
        extra_urls: Any other official URLs cited in the answer for this same Texas role
        
    - maryland_position (Carroll County Public Schools):
        institution: Should be "Carroll County Public Schools" (or common abbreviation if used in the answer)
        position_title
        job_url
        min_education
        certification_requirement
        certification_pathways
        salary_info: Salary/hourly rate if provided in the answer (important for MD)
        details
        position_type
        campus_location
        extra_urls
        
    - dallas_position (Dallas College; Mountain View campus or system-wide are both acceptable):
        institution: Should be "Dallas College" (accept abbreviations if used)
        position_title
        job_url
        min_education
        certification_requirement
        certification_pathways
        position_type: e.g., "Staff", "Adjunct/Faculty", "Full-time", "Part-time", "Non-credit (CE)", "Temporary", etc.
        campus_location: campus/site if specified, e.g., "Mountain View campus"
        details
        salary_info
        extra_urls
        
    - nc_position (Wake County Public Schools in North Carolina):
        institution: Should be "Wake County Public Schools" (or "WCPSS")
        position_title
        job_url
        min_education
        certification_requirement
        certification_pathways: e.g., mention of "Residency License" or alternative pathway acceptance
        details
        salary_info
        position_type
        campus_location
        extra_urls

    IMPORTANT:
    - Only extract URLs explicitly present in the answer; do not invent.
    - Keep strings as they appear in the answer; do not normalize or rephrase.
    - If a URL is missing protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _host(url: Optional[str]) -> str:
    if not _non_empty(url):
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _texas_institution_ok(inst: Optional[str], url: Optional[str]) -> bool:
    """Accepts Katy ISD or Fort Bend ISD; tolerate common variants; also allow inference from URL host if present."""
    if not _non_empty(inst) and not _non_empty(url):
        return False
    inst_l = (inst or "").lower()
    host = _host(url)
    allowed_name_fragments = [
        "katy isd", "katy independent school district", "katyisd",
        "fort bend isd", "fort bend independent school district", "fbisd", "fortbendisd"
    ]
    if any(frag in inst_l for frag in allowed_name_fragments if frag.strip()):
        return True
    allowed_hosts = ["katy.tedk12.com", "fortbendisd.tedk12.com", "www.fortbendisd.com", "katyisd.org",
                     "fortbendisd.com", "fbisd.tedk12.com", "katyisd.tedk12.com", "applitrack.com"]
    if any(h in host for h in allowed_hosts):
        return True
    return False


def _is_ccps_md(inst: Optional[str], url: Optional[str]) -> bool:
    """Carroll County Public Schools (Maryland)"""
    inst_l = (inst or "").lower()
    host = _host(url)
    if "carroll county public schools" in inst_l or inst_l.strip() in {"ccps", "ccps - maryland", "ccps (md)"}:
        return True
    if any(k in host for k in ["carrollk12.org", "applitrack.com/carrollk12", "applitrack.com", "frontlineeducation.com"]):
        return True
    return False


def _is_dallas_college(inst: Optional[str], url: Optional[str]) -> bool:
    inst_l = (inst or "").lower()
    host = _host(url)
    if "dallas college" in inst_l or inst_l.strip() in {"dcccd", "dallascolllege"}:
        return True
    if any(k in host for k in ["dallascollege.edu", "dcccd.edu", "myworkdayjobs.com"]):
        return True
    return False


def _is_wcpss(inst: Optional[str], url: Optional[str]) -> bool:
    inst_l = (inst or "").lower()
    host = _host(url)
    if "wake county public schools" in inst_l or "wcpss" in inst_l:
        return True
    if any(k in host for k in ["wcpss.net", "applitrack.com/wcpss", "applitrack.com"]):
        return True
    return False


def _merge_urls(primary: Optional[str], extra: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    if _non_empty(primary):
        urls.append(primary)  # type: ignore
    if extra:
        for u in extra:
            if _non_empty(u):
                urls.append(u)
    return urls[:10]  # soft cap


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_texas(evaluator: Evaluator, root) -> None:
    group = evaluator.add_parallel(
        id="Texas_Institution_Position",
        desc="Identification and verification of an appropriate entry-level position at either Katy ISD or Fort Bend ISD in Texas",
        parent=root,
        critical=False
    )

    data: JobSearchExtraction = next(
        (item["extraction"] for item in evaluator._extraction_results if "extraction" in item), None
    )  # not used; we will access via evaluator._extraction_results recorded object
    # Safer: find last extraction of JobSearchExtraction
    extracted: Optional[JobSearchExtraction] = None
    for rec in evaluator._extraction_results:
        try:
            maybe = rec.get("positions_extraction")  # won't exist
        except Exception:
            pass
    # We have structured access via our saved handle in evaluate_answer; pass directly instead of inferring here.


async def _verify_texas(evaluator: Evaluator, parent, item: Optional[PositionItem]) -> None:
    group = evaluator.add_parallel(
        id="Texas_Institution_Position",
        desc="Identification and verification of an appropriate entry-level position at either Katy ISD or Fort Bend ISD in Texas",
        parent=parent,
        critical=False
    )

    # 1) Position identified with official title at allowed district (critical)
    exists_ok = _non_empty(item.position_title if item else None) and _texas_institution_ok(
        item.institution if item else None, item.job_url if item else None
    )
    evaluator.add_custom_node(
        result=exists_ok,
        id="Texas_Position_Identified",
        desc="A specific position at Katy ISD or Fort Bend ISD has been identified with its official title",
        parent=group,
        critical=True
    )

    # 2) URL is official careers/job posting (critical)
    url_node = evaluator.add_leaf(
        id="Texas_Position_URL",
        desc="A valid reference URL from the institution's official careers website or job board is provided",
        parent=group,
        critical=True
    )
    url_claim = f"The provided URL for the Texas position is '{(item.job_url if item else None)}', and it is an official careers or job posting page for Katy ISD or Fort Bend ISD."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=item.job_url if item and _non_empty(item.job_url) else None,
        additional_instruction=(
            "Pass only if the webpage clearly indicates it is hosted by Katy ISD or Fort Bend ISD (including their official ATS like Frontline/AppliTrack or TEDK12) "
            "with explicit district branding and a job posting structure. "
            "If the URL is missing, broken, or from non-official aggregators, mark as incorrect."
        )
    )

    # 3) Minimum education requirement is <= bachelor's (critical)
    edu_node = evaluator.add_leaf(
        id="Texas_Bachelor_Degree_Match",
        desc="The position's minimum education requirement matches or is below a bachelor's degree level",
        parent=group,
        critical=True
    )
    edu_claim = (
        "The minimum required education stated in this posting is at most a Bachelor's degree "
        "(Bachelor's, Associate's, or High School acceptable; Master's or higher would not qualify). "
        "Phrases like 'Bachelor's degree or higher' or 'Master's preferred' still satisfy this check."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=item.job_url if item else None,
        additional_instruction="Look for sections labeled Minimum Qualifications, Education, or Requirements."
    )

    # 4) Certification pathway suitability (critical)
    cert_node = evaluator.add_leaf(
        id="Texas_Certification_Pathway",
        desc="The position either does not require immediate Texas teaching certification OR the institution offers an alternative certification pathway for candidates with bachelor's degrees",
        parent=group,
        critical=True
    )
    cert_claim = (
        "One of the following is explicitly supported by the posting: "
        "(a) immediate Texas teaching certification is not required for this role; OR "
        "(b) candidates in an Alternative Certification Program (ACP), intern/probationary certification, emergency permit, "
        "residency/teacher residency, or district-supported alternative pathway are accepted."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_node,
        sources=item.job_url if item else None,
        additional_instruction="Accept synonyms: 'ACP accepted', 'intern/probationary', 'emergency permit', 'teacher resident', 'alternative certification'."
    )

    # 5) Additional details provided (non-critical, existence in answer)
    details_ok = _non_empty(item.details if item else None)
    evaluator.add_custom_node(
        result=details_ok,
        id="Texas_Position_Details",
        desc="Additional relevant information about the position such as job duties, department, or grade level is provided",
        parent=group,
        critical=False
    )


async def _verify_maryland(evaluator: Evaluator, parent, item: Optional[PositionItem]) -> None:
    group = evaluator.add_parallel(
        id="Maryland_Institution_Position",
        desc="Identification and verification of an appropriate entry-level position at Carroll County Public Schools in Maryland",
        parent=parent,
        critical=False
    )

    # 1) Position identified (critical)
    exists_ok = _non_empty(item.position_title if item else None) and _is_ccps_md(
        item.institution if item else None, item.job_url if item else None
    )
    evaluator.add_custom_node(
        result=exists_ok,
        id="Maryland_Position_Identified",
        desc="A specific position at Carroll County Public Schools has been identified with its official title",
        parent=group,
        critical=True
    )

    # 2) URL official (critical)
    url_node = evaluator.add_leaf(
        id="Maryland_Position_URL",
        desc="A valid reference URL from Carroll County Public Schools' official careers website or job board is provided",
        parent=group,
        critical=True
    )
    url_claim = f"The provided URL for the Maryland (Carroll County Public Schools) position is '{(item.job_url if item else None)}', and it is an official CCPS careers/job posting page."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=item.job_url if item and _non_empty(item.job_url) else None,
        additional_instruction="Accept official CCPS AppliTrack/Frontline or district-hosted postings that clearly show 'Carroll County Public Schools' branding."
    )

    # 3) Education requirement <= bachelor's (critical)
    edu_node = evaluator.add_leaf(
        id="Maryland_Education_Requirement",
        desc="The position's minimum education requirement matches or is below a bachelor's degree level",
        parent=group,
        critical=True
    )
    edu_claim = (
        "The minimum required education for this CCPS position is at most a Bachelor's degree. "
        "If it says Bachelor's or higher, that's acceptable. Master's required would fail."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=item.job_url if item else None,
        additional_instruction="Check 'Minimum Qualifications' or similar section."
    )

    # 4) Certification status identified (critical)
    cert_node = evaluator.add_leaf(
        id="Maryland_Certification_Status",
        desc="The position's Maryland certification requirement is clearly identified (whether immediate certification is required or if substitute/alternative pathways exist)",
        parent=group,
        critical=True
    )
    cert_claim = (
        "The posting clearly states Maryland certification status: either a certificate is required immediately "
        "or that the role is exempt/does not require immediate certification (e.g., substitute/paraeducator/assistant) "
        "or permits alternative pathways (conditional/provisional/waiver)."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_node,
        sources=item.job_url if item else None,
        additional_instruction="Look for 'certificate required', 'conditional certificate', 'substitute', 'paraeducator', or explicit mention of non-licensed roles."
    )

    # 5) Compensation provided (non-critical, provided in answer)
    comp_ok = _non_empty(item.salary_info if item else None)
    evaluator.add_custom_node(
        result=comp_ok,
        id="Maryland_Compensation_Info",
        desc="Salary or hourly rate information for the position is provided",
        parent=group,
        critical=False
    )


async def _verify_dallas(evaluator: Evaluator, parent, item: Optional[PositionItem]) -> None:
    group = evaluator.add_parallel(
        id="Dallas_College_Position",
        desc="Identification and verification of an appropriate position at Dallas College (specifically Mountain View campus or system-wide)",
        parent=parent,
        critical=False
    )

    # 1) Position identified (critical)
    exists_ok = _non_empty(item.position_title if item else None) and _is_dallas_college(
        item.institution if item else None, item.job_url if item else None
    )
    evaluator.add_custom_node(
        result=exists_ok,
        id="Dallas_Position_Identified",
        desc="A specific position at Dallas College has been identified with its official title",
        parent=group,
        critical=True
    )

    # 2) URL official (critical)
    url_node = evaluator.add_leaf(
        id="Dallas_Position_URL",
        desc="A valid reference URL from Dallas College's official careers website is provided",
        parent=group,
        critical=True
    )
    url_claim = f"The provided URL for the Dallas College position is '{(item.job_url if item else None)}', and it is an official Dallas College careers/job posting page."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=item.job_url if item and _non_empty(item.job_url) else None,
        additional_instruction="Accept DallasCollege.edu or its official ATS (e.g., Workday myworkdayjobs.com/DallasCollege) that displays Dallas College branding."
    )

    # 3) Degree requirement clearly stated and matches bachelor's level (critical)
    edu_node = evaluator.add_leaf(
        id="Dallas_Degree_Requirement",
        desc="The position's minimum degree requirement is clearly stated and matches the candidate's qualification level (bachelor's degree)",
        parent=group,
        critical=True
    )
    edu_claim = (
        "The posting clearly states the minimum degree requirement and it is at most a Bachelor's degree "
        "(Bachelor's acceptable; higher-only such as Master's required would fail)."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=item.job_url if item else None,
        additional_instruction="If it says 'Bachelor's required' or 'Bachelor's or higher', pass. If 'Master's required', fail."
    )

    # 4) Position type identified (critical)
    type_node = evaluator.add_leaf(
        id="Dallas_Position_Type",
        desc="The position type (adjunct, full-time, non-credit, support staff, etc.) is clearly identified",
        parent=group,
        critical=True
    )
    type_claim = (
        "The posting clearly indicates the employment category or position type (e.g., Staff, Faculty, Adjunct, Full-time, Part-time, Temporary, Non-credit/CE)."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=item.job_url if item else None,
        additional_instruction="Look for labels like 'Job Type', 'Role Type', 'Employee Type', 'Work Type', 'Faculty/Staff', or 'Adjunct'."
    )

    # 5) Campus location provided (non-critical, provided in answer)
    campus_ok = _non_empty(item.campus_location if item else None)
    evaluator.add_custom_node(
        result=campus_ok,
        id="Dallas_Campus_Location",
        desc="The specific campus or campuses where the position is located is provided",
        parent=group,
        critical=False
    )


async def _verify_nc(evaluator: Evaluator, parent, item: Optional[PositionItem]) -> None:
    group = evaluator.add_parallel(
        id="North_Carolina_Institution_Position",
        desc="Identification and verification of an appropriate entry-level position at Wake County Public Schools in North Carolina",
        parent=parent,
        critical=False
    )

    # 1) Position identified (critical)
    exists_ok = _non_empty(item.position_title if item else None) and _is_wcpss(
        item.institution if item else None, item.job_url if item else None
    )
    evaluator.add_custom_node(
        result=exists_ok,
        id="NC_Position_Identified",
        desc="A specific position at Wake County Public Schools has been identified with its official title",
        parent=group,
        critical=True
    )

    # 2) URL official (critical)
    url_node = evaluator.add_leaf(
        id="NC_Position_URL",
        desc="A valid reference URL from Wake County Public Schools' official careers website or job board is provided",
        parent=group,
        critical=True
    )
    url_claim = f"The provided URL for the Wake County Public Schools position is '{(item.job_url if item else None)}', and it is an official WCPSS careers/job posting page."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=item.job_url if item and _non_empty(item.job_url) else None,
        additional_instruction="Accept wcpss.net or official WCPSS AppliTrack/Frontline postings with WCPSS branding."
    )

    # 3) Education requirement <= bachelor's (critical)
    edu_node = evaluator.add_leaf(
        id="NC_Bachelor_Requirement",
        desc="The position's minimum education requirement matches or is below a bachelor's degree level",
        parent=group,
        critical=True
    )
    edu_claim = (
        "The minimum required education for this WCPSS position is at most a Bachelor's degree (Bachelor's or lower acceptable). "
        "If Master's or higher is required, fail."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=item.job_url if item else None,
        additional_instruction="Check 'Education/Requirements/Qualifications' on the posting."
    )

    # 4) GPA/license condition (critical)
    gpa_node = evaluator.add_leaf(
        id="NC_GPA_Requirement",
        desc="If the position requires a teaching license, the minimum GPA requirement (typically 2.7 for NC) is verified, OR the position does not require immediate licensure",
        parent=group,
        critical=True
    )
    nc_urls = _merge_urls(item.job_url if item else None, item.extra_urls if item else [])
    gpa_claim = (
        "One of the following is supported by the posting (or its directly linked official licensure info): "
        "(a) the job does not require an immediate North Carolina teaching license; OR "
        "(b) if a NC teaching license is required, the minimum GPA requirement (commonly 2.7 for NC residency license eligibility) "
        "is explicitly stated or linked."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_node,
        sources=nc_urls if nc_urls else None,
        additional_instruction="Look for 'Residency License', 'Lateral Entry (legacy)', 'Licensure requirements', or explicit GPA (e.g., 2.7). If the role is non-licensed (assistant, substitute), treat as satisfying condition (a)."
    )

    # 5) License pathway clarity (critical)
    license_node = evaluator.add_leaf(
        id="NC_License_Pathway",
        desc="The position's North Carolina teaching license requirement status is clearly identified (whether immediate license is required or if alternative pathways exist)",
        parent=group,
        critical=True
    )
    license_claim = (
        "The posting clearly communicates licensure status: either requires an immediate NC license, "
        "or specifies acceptance of alternative pathways (e.g., Residency License) or indicates the role is non-licensed."
    )
    await evaluator.verify(
        claim=license_claim,
        node=license_node,
        sources=item.job_url if item else None,
        additional_instruction="Keywords: 'Residency License', 'licensure required/not required', 'non-licensed', 'teacher assistant/paraeducator/substitute'."
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
    Evaluate an answer for the multi-institution entry-level education jobs task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel across institutions
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

    # Extract structured positions
    extraction: JobSearchExtraction = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=JobSearchExtraction,
        extraction_name="positions_extraction"
    )

    # Build verification tree per institution
    # Texas (Katy ISD or Fort Bend ISD)
    await _verify_texas(evaluator, root, extraction.texas_position or PositionItem())

    # Maryland (Carroll County Public Schools)
    await _verify_maryland(evaluator, root, extraction.maryland_position or PositionItem())

    # Dallas College
    await _verify_dallas(evaluator, root, extraction.dallas_position or PositionItem())

    # North Carolina (Wake County Public Schools)
    await _verify_nc(evaluator, root, extraction.nc_position or PositionItem())

    # Return summary
    return evaluator.get_summary()