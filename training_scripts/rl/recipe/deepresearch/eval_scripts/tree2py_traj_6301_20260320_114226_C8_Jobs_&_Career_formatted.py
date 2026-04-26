import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task identifiers
# -----------------------------------------------------------------------------
TASK_ID = "nc_school_district_jobs_comparison"
TASK_DESCRIPTION = """
I am planning a career in education in North Carolina and want to compare employment opportunities across different school districts. Please research and provide comprehensive employment information for four different North Carolina public school districts.

For each district, provide:

1. District Identification: The district name and a link to its official employment/careers website

2. Certification and Qualification Requirements: 
   - For at least one district: Teacher certification requirements (degree, licensure, alternative pathways)
   - For at least one district: School administrator requirements (degree, experience)
   - For at least one district: Coaching position requirements (certifications like NFHS, CPR/AED)

3. Salary Information: 
   - Teacher or administrator salary schedule information
   - Local supplement percentages (if offered beyond state base pay)
   - Any experience-based salary increments

4. Application Process:
   - The online application system used (e.g., Applitrack, Frontline Recruitment)
   - Link to the job application portal
   - Required application materials (resume, cover letter, transcripts, etc.)
   - Typical posting duration for vacancies

5. Benefits and Additional Information:
   - Employee benefits offered (health insurance, holidays, life insurance, etc.)
   - Contact information for HR/employment department
   - Any unique hiring practices or timelines

Each district must be a different North Carolina public school district. Provide valid, accessible URLs for all referenced information.
"""


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class TextWithSources(BaseModel):
    summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SalaryInfo(BaseModel):
    schedule_desc: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)
    local_supplement_info: Optional[str] = None
    local_supplement_sources: List[str] = Field(default_factory=list)
    experience_increments_info: Optional[str] = None
    experience_increments_sources: List[str] = Field(default_factory=list)


class ApplicationInfo(BaseModel):
    system_name: Optional[str] = None
    portal_url: Optional[str] = None
    materials: List[str] = Field(default_factory=list)
    materials_sources: List[str] = Field(default_factory=list)
    posting_duration: Optional[str] = None
    posting_sources: List[str] = Field(default_factory=list)


class BenefitsHRInfo(BaseModel):
    benefits_summary: Optional[str] = None
    benefits_sources: List[str] = Field(default_factory=list)
    hr_contact: Optional[str] = None
    hr_contact_sources: List[str] = Field(default_factory=list)


class DistrictInfo(BaseModel):
    name: Optional[str] = None
    employment_url: Optional[str] = None

    teacher_cert: TextWithSources = Field(default_factory=TextWithSources)
    admin_requirements: TextWithSources = Field(default_factory=TextWithSources)
    coaching_requirements: TextWithSources = Field(default_factory=TextWithSources)

    salary: SalaryInfo = Field(default_factory=SalaryInfo)
    application: ApplicationInfo = Field(default_factory=ApplicationInfo)
    benefits_hr: BenefitsHRInfo = Field(default_factory=BenefitsHRInfo)
    unique_practices_timeline: TextWithSources = Field(default_factory=TextWithSources)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictInfo] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_districts() -> str:
    return """
Extract all North Carolina public school districts discussed in the answer and their employment-related details.

GENERAL INSTRUCTIONS:
- Extract ALL the districts mentioned in the answer (not just four), preserving their order of appearance.
- For each district, capture the specific requested fields when they are explicitly present in the answer.
- For every informational field that cites a source, extract the actual URL(s) from the answer text; do not invent URLs.
- If a field is not mentioned, set its value to null (or an empty list where applicable).
- Prefer official district pages (e.g., *.k12.nc.us, *.org, or district official domains). If the answer cites a dedicated PDF, keep that URL.
- Keep portal_url separate from general employment_url.

Return JSON with:
{
  "districts": [
    {
      "name": string|null,
      "employment_url": string|null,

      "teacher_cert": {
        "summary": string|null,
        "sources": string[]   // URLs that support teacher certification requirements
      },
      "admin_requirements": {
        "summary": string|null,
        "sources": string[]   // URLs that support school administrator qualification requirements
      },
      "coaching_requirements": {
        "summary": string|null,
        "sources": string[]   // URLs that support coaching position requirements (e.g., NFHS, CPR/AED)
      },

      "salary": {
        "schedule_desc": string|null,
        "schedule_urls": string[],               // URLs for salary schedule documents/pages
        "local_supplement_info": string|null,    // Include % details if provided
        "local_supplement_sources": string[],
        "experience_increments_info": string|null,
        "experience_increments_sources": string[]
      },

      "application": {
        "system_name": string|null,              // e.g., Frontline Recruitment, AppliTrack, etc.
        "portal_url": string|null,               // Link to job application portal
        "materials": string[],                   // Items like resume, cover letter, transcripts, certifications, etc.
        "materials_sources": string[],           // URLs where materials are listed/required
        "posting_duration": string|null,         // Typical posting duration/policy text if given
        "posting_sources": string[]
      },

      "benefits_hr": {
        "benefits_summary": string|null,         // High-level summary (health, dental, life, holidays...)
        "benefits_sources": string[],
        "hr_contact": string|null,               // HR contact info (email/phone/address) if provided
        "hr_contact_sources": string[]
      },

      "unique_practices_timeline": {
        "summary": string|null,                  // Any unique hiring practices/timelines
        "sources": string[]
      }
    }
  ]
}

SPECIAL RULES FOR URLS:
- Only extract URLs explicitly present in the answer (plain or markdown links).
- Include complete URLs with http:// or https://.
- If a field references a source but no URL is present in the answer, leave sources empty.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _is_valid_url(u: Optional[str]) -> bool:
    u = _norm(u)
    return u.startswith("http://") or u.startswith("https://")


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        u2 = _norm(u)
        if not _is_valid_url(u2):
            continue
        if u2 not in seen:
            seen.add(u2)
            result.append(u2)
    return result


def _combine_sources(*url_lists: List[str]) -> List[str]:
    all_urls: List[str] = []
    for lst in url_lists:
        all_urls.extend(lst or [])
    return _dedup_urls(all_urls)


def _first_nonempty_text_and_sources(*items: Tuple[Optional[str], List[str]]) -> Tuple[Optional[str], List[str]]:
    for text, src in items:
        if _norm(text) and len(_dedup_urls(src)) > 0:
            return text, _dedup_urls(src)
    return None, []


def _normalize_dist_name(name: Optional[str]) -> str:
    # Lowercase, strip, remove common suffixes like "public schools", "schools", punctuation
    import re
    s = _norm(name).lower()
    s = re.sub(r'[\.\,\-\_]', ' ', s)
    s = re.sub(r'\b(public )?schools?\b', '', s)
    s = re.sub(r'\bdistrict\b', '', s)
    s = re.sub(r'\bnc\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _unique_names(names: List[Optional[str]]) -> int:
    return len({_normalize_dist_name(n) for n in names if _norm(n)})


def _select_first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []


# -----------------------------------------------------------------------------
# Coverage verifications (teacher/admin/coaching) across all districts
# -----------------------------------------------------------------------------
async def add_coverage_verification(
    evaluator: Evaluator,
    parent_node,
    node_base_id: str,
    description: str,
    candidate_urls: List[str],
    claim_text: str,
    add_ins: str
) -> None:
    """
    Build a sequential coverage node:
      1) existence of at least one supporting URL,
      2) verification supported by any of the provided URLs (multi-URL verification).
    """
    seq = evaluator.add_sequential(
        id=node_base_id,
        desc=description,
        parent=parent_node,
        critical=True
    )

    exists = len(_dedup_urls(candidate_urls)) > 0
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_base_id}_exists",
        desc=f"{description} — at least one cited source URL is provided in the answer.",
        parent=seq,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{node_base_id}_supported",
        desc=f"{description} — the cited source(s) explicitly support the requirement(s).",
        parent=seq,
        critical=True,
    )

    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_dedup_urls(candidate_urls),
        additional_instruction=add_ins
    )


# -----------------------------------------------------------------------------
# Per-district verification
# -----------------------------------------------------------------------------
async def verify_one_district(evaluator: Evaluator, parent_node, d: DistrictInfo, idx: int) -> None:
    """
    Build verification subtree for a single district (index starting at 0).
    """
    did = idx + 1
    dnode = evaluator.add_parallel(
        id=f"district_{did}",
        desc=f"District {did} employment information.",
        parent=parent_node,
        critical=False
    )

    # 1) Identification (sequential: presence -> verify official careers page)
    ident = evaluator.add_sequential(
        id=f"d{did}_identification",
        desc="Provides the district name and an official district employment/careers page URL (valid/accessible).",
        parent=dnode,
        critical=True
    )
    ident_present = bool(_norm(d.name)) and _is_valid_url(d.employment_url)
    evaluator.add_custom_node(
        result=ident_present,
        id=f"d{did}_identification_presence",
        desc="District name present and an employment/careers URL provided.",
        parent=ident,
        critical=True
    )
    ident_leaf = evaluator.add_leaf(
        id=f"d{did}_identification_page",
        desc="The employment/careers URL is an official page for this NC public school district.",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page is an official employment/careers page for '{_norm(d.name)}', a North Carolina public school district.",
        node=ident_leaf,
        sources=d.employment_url,
        additional_instruction="Accept district employment, careers, jobs, HR, or recruitment pages. Treat name variants (with/without 'Public Schools' or 'Schools') as acceptable. Consider '.k12.nc.us' or clearly official district domains as strong signals."
    )

    # 2) Qualification requirements (choose any one category with sources)
    qual = evaluator.add_sequential(
        id=f"d{did}_qualification_requirements",
        desc="Provides certification/qualification requirements for at least one relevant position type with supporting source URL(s).",
        parent=dnode,
        critical=True
    )
    # pick first available category with text + sources
    chosen_text, chosen_sources = _first_nonempty_text_and_sources(
        (d.teacher_cert.summary, d.teacher_cert.sources),
        (d.admin_requirements.summary, d.admin_requirements.sources),
        (d.coaching_requirements.summary, d.coaching_requirements.sources),
    )
    qual_exists = bool(_norm(chosen_text)) and len(_dedup_urls(chosen_sources)) > 0
    evaluator.add_custom_node(
        result=qual_exists,
        id=f"d{did}_qualification_presence",
        desc="At least one of teacher/admin/coaching requirements is provided with source URL(s).",
        parent=qual,
        critical=True
    )
    qual_leaf = evaluator.add_leaf(
        id=f"d{did}_qualification_supported",
        desc="The chosen requirements are supported by the cited source(s).",
        parent=qual,
        critical=True
    )
    await evaluator.verify(
        claim="This page states explicit certification/qualification requirements for at least one role (teacher, school administrator, or coach).",
        node=qual_leaf,
        sources=_dedup_urls(chosen_sources),
        additional_instruction="The page should clearly mention degree/licensure or alternative pathways for teachers, or degree/experience for administrators, or required certifications (e.g., NFHS, CPR/AED) for coaching roles."
    )

    # 3) Salary information (parallel): salary schedule, local supplement, experience increments
    salary = evaluator.add_parallel(
        id=f"d{did}_salary_information",
        desc="Provides required salary information for the district with supporting source URL(s).",
        parent=dnode,
        critical=True
    )
    # 3.1 salary schedule
    sched_seq = evaluator.add_sequential(
        id=f"d{did}_salary_schedule_seq",
        desc="Includes teacher or administrator salary schedule information with a valid/accessible URL.",
        parent=salary,
        critical=True
    )
    sched_exists = len(_dedup_urls(d.salary.schedule_urls)) > 0
    evaluator.add_custom_node(
        result=sched_exists,
        id=f"d{did}_salary_schedule_sources_present",
        desc="At least one salary schedule URL is provided.",
        parent=sched_seq,
        critical=True
    )
    sched_leaf = evaluator.add_leaf(
        id=f"d{did}_salary_schedule",
        desc="The cited page(s) provide a teacher or administrator salary schedule/scale.",
        parent=sched_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page provides a teacher or administrator salary schedule (or salary scale/chart) for the district.",
        node=sched_leaf,
        sources=_dedup_urls(d.salary.schedule_urls),
        additional_instruction="Accept salary schedule PDFs or web pages; look for tables or grids with steps/lanes, or explicitly labeled 'salary schedule'."
    )

    # 3.2 local supplement
    supp_seq = evaluator.add_sequential(
        id=f"d{did}_local_supplement_seq",
        desc="Provides local supplement percentage(s) if offered/published, with a source URL.",
        parent=salary,
        critical=True
    )
    supp_exists = bool(_norm(d.salary.local_supplement_info)) and len(_dedup_urls(d.salary.local_supplement_sources)) > 0
    evaluator.add_custom_node(
        result=supp_exists,
        id=f"d{did}_local_supplement_sources_present",
        desc="Local supplement information and source URL(s) are provided.",
        parent=supp_seq,
        critical=True
    )
    supp_leaf = evaluator.add_leaf(
        id=f"d{did}_local_supplement",
        desc="The cited page(s) state the district's local salary supplement percentage(s) or policy.",
        parent=supp_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page states the district's local salary supplement percentage(s) or policy beyond state base pay.",
        node=supp_leaf,
        sources=_dedup_urls(d.salary.local_supplement_sources),
        additional_instruction="Look for 'local supplement' language, percent amounts by position/teacher level, or equivalent terminology on the page/PDF."
    )

    # 3.3 experience increments
    exp_seq = evaluator.add_sequential(
        id=f"d{did}_experience_increments_seq",
        desc="Explains experience-based salary increments/steps (how experience affects pay) with a source URL.",
        parent=salary,
        critical=True
    )
    exp_exists = bool(_norm(d.salary.experience_increments_info)) and len(_dedup_urls(d.salary.experience_increments_sources)) > 0
    evaluator.add_custom_node(
        result=exp_exists,
        id=f"d{did}_experience_increments_sources_present",
        desc="Experience-based increment information and source URL(s) are provided.",
        parent=exp_seq,
        critical=True
    )
    exp_leaf = evaluator.add_leaf(
        id=f"d{did}_experience_increments",
        desc="The cited page(s) explain salary steps or experience-based increments.",
        parent=exp_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page explains how experience affects pay (salary steps/increments).",
        node=exp_leaf,
        sources=_dedup_urls(d.salary.experience_increments_sources),
        additional_instruction="Look for step tables by years of experience or text explaining increments based on service years."
    )

    # 4) Application process (parallel)
    app = evaluator.add_parallel(
        id=f"d{did}_application_process",
        desc="Provides required application process details with supporting source URL(s).",
        parent=dnode,
        critical=True
    )
    # 4.1 application system
    app_sys_seq = evaluator.add_sequential(
        id=f"d{did}_application_system_seq",
        desc="Names the online application system used by the district.",
        parent=app,
        critical=True
    )
    app_sys_sources = _dedup_urls([d.application.portal_url] if _is_valid_url(d.application.portal_url) else ([] if not _is_valid_url(d.employment_url) else [d.employment_url]))
    app_sys_exists = bool(_norm(d.application.system_name)) and len(app_sys_sources) > 0
    evaluator.add_custom_node(
        result=app_sys_exists,
        id=f"d{did}_application_system_present",
        desc="Application system name present with at least one supporting URL.",
        parent=app_sys_seq,
        critical=True
    )
    app_sys_leaf = evaluator.add_leaf(
        id=f"d{did}_application_system",
        desc="The cited page shows the district uses the named online application system.",
        parent=app_sys_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page shows the district uses '{_norm(d.application.system_name)}' (e.g., Frontline/AppliTrack/NEOGOV) as its online application system.",
        node=app_sys_leaf,
        sources=app_sys_sources,
        additional_instruction="Accept clear indicators such as vendor branding, platform name, or explicit statement that this is the district's applicant system."
    )

    # 4.2 application portal link
    portal_seq = evaluator.add_sequential(
        id=f"d{did}_application_portal_link_seq",
        desc="Provides a valid/accessible link to the job application portal.",
        parent=app,
        critical=True
    )
    portal_exists = _is_valid_url(d.application.portal_url)
    evaluator.add_custom_node(
        result=portal_exists,
        id=f"d{did}_application_portal_link_present",
        desc="An application portal URL is provided.",
        parent=portal_seq,
        critical=True
    )
    portal_leaf = evaluator.add_leaf(
        id=f"d{did}_application_portal_link",
        desc="The URL is the district's job application portal page.",
        parent=portal_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page is the job application portal for '{_norm(d.name)}' (shows job postings and/or apply functionality).",
        node=portal_leaf,
        sources=d.application.portal_url,
        additional_instruction="Look for listings of openings or an 'Apply' workflow on a recruiting platform. Minor name variations are acceptable."
    )

    # 4.3 application materials
    mats_seq = evaluator.add_sequential(
        id=f"d{did}_application_materials_seq",
        desc="Lists required application materials as stated by the district.",
        parent=app,
        critical=True
    )
    mats_sources = _combine_sources(d.application.materials_sources, [d.application.portal_url] if _is_valid_url(d.application.portal_url) else [])
    mats_exists = len(d.application.materials) > 0 and len(mats_sources) > 0
    evaluator.add_custom_node(
        result=mats_exists,
        id=f"d{did}_application_materials_present",
        desc="Application materials are listed with at least one supporting URL.",
        parent=mats_seq,
        critical=True
    )
    mats_leaf = evaluator.add_leaf(
        id=f"d{did}_application_materials",
        desc="The cited page lists required application materials.",
        parent=mats_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page lists required application materials for applicants (e.g., resume, cover letter, transcripts, certifications).",
        node=mats_leaf,
        sources=mats_sources,
        additional_instruction="Accept a bullet list, table, or section describing required documents/materials for application submission."
    )

    # 4.4 posting duration
    post_seq = evaluator.add_sequential(
        id=f"d{did}_posting_duration_seq",
        desc="Gives the typical posting duration for vacancies (and/or district policy) with a source URL.",
        parent=app,
        critical=True
    )
    post_exists = bool(_norm(d.application.posting_duration)) and len(_dedup_urls(d.application.posting_sources)) > 0
    evaluator.add_custom_node(
        result=post_exists,
        id=f"d{did}_posting_duration_present",
        desc="Posting duration/policy provided with at least one supporting URL.",
        parent=post_seq,
        critical=True
    )
    post_leaf = evaluator.add_leaf(
        id=f"d{did}_posting_duration",
        desc="The cited page states typical posting duration or policy (e.g., remain open until filled).",
        parent=post_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page states the typical posting duration for vacancies or the district's policy on how long postings remain open.",
        node=post_leaf,
        sources=_dedup_urls(d.application.posting_sources),
        additional_instruction="Accept language like 'Open until filled', specific day ranges, or policies governing posting timelines."
    )

    # 5) Benefits and HR (parallel)
    ben = evaluator.add_parallel(
        id=f"d{did}_benefits_and_hr",
        desc="Provides benefits information and HR contact details with supporting source URL(s).",
        parent=dnode,
        critical=True
    )
    # 5.1 benefits summary
    ben_seq = evaluator.add_sequential(
        id=f"d{did}_benefits_summary_seq",
        desc="Summarizes employee benefits offered with a source URL.",
        parent=ben,
        critical=True
    )
    ben_exists = bool(_norm(d.benefits_hr.benefits_summary)) and len(_dedup_urls(d.benefits_hr.benefits_sources)) > 0
    evaluator.add_custom_node(
        result=ben_exists,
        id=f"d{did}_benefits_summary_present",
        desc="Benefits summary text and source URL(s) are provided.",
        parent=ben_seq,
        critical=True
    )
    ben_leaf = evaluator.add_leaf(
        id=f"d{did}_benefits_summary",
        desc="The cited page describes employee benefits offered by the district.",
        parent=ben_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page describes the district's employee benefits (e.g., health, dental, life insurance, holidays, retirement).",
        node=ben_leaf,
        sources=_dedup_urls(d.benefits_hr.benefits_sources),
        additional_instruction="Accept district HR/benefits pages or official documents summarizing benefits offerings."
    )

    # 5.2 HR contact
    hr_seq = evaluator.add_sequential(
        id=f"d{did}_hr_contact_seq",
        desc="Provides HR/employment department contact information with a source URL.",
        parent=ben,
        critical=True
    )
    hr_exists = bool(_norm(d.benefits_hr.hr_contact)) and len(_dedup_urls(d.benefits_hr.hr_contact_sources)) > 0
    evaluator.add_custom_node(
        result=hr_exists,
        id=f"d{did}_hr_contact_present",
        desc="HR contact info and source URL(s) are provided.",
        parent=hr_seq,
        critical=True
    )
    hr_leaf = evaluator.add_leaf(
        id=f"d{did}_hr_contact",
        desc="The cited page shows HR/employment department contact info.",
        parent=hr_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page provides contact information for the district's HR or employment department (e.g., email, phone, office address).",
        node=hr_leaf,
        sources=_dedup_urls(d.benefits_hr.hr_contact_sources),
        additional_instruction="Accept explicit HR contact info. Minor formatting variations acceptable."
    )

    # 6) Unique hiring practices/timeline (sequential)
    uniq_seq = evaluator.add_sequential(
        id=f"d{did}_unique_practices_timeline_seq",
        desc="Describes any unique hiring practices or timelines mentioned by the district, with a supporting URL if referenced.",
        parent=dnode,
        critical=True
    )
    uniq_exists = bool(_norm(d.unique_practices_timeline.summary)) and len(_dedup_urls(d.unique_practices_timeline.sources)) > 0
    evaluator.add_custom_node(
        result=uniq_exists,
        id=f"d{did}_unique_practices_timeline_present",
        desc="Unique hiring practices/timelines provided with supporting URL(s).",
        parent=uniq_seq,
        critical=True
    )
    uniq_leaf = evaluator.add_leaf(
        id=f"d{did}_unique_practices_timeline",
        desc="The cited page describes a unique hiring practice or timeline for the district.",
        parent=uniq_seq,
        critical=True
    )
    await evaluator.verify(
        claim="This page describes unique hiring practices or timelines for the district (e.g., early contracts, job fairs, decision windows).",
        node=uniq_leaf,
        sources=_dedup_urls(d.unique_practices_timeline.sources),
        additional_instruction="Look for explicit notes about timing or distinct processes the district follows for hiring."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
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
    Evaluate an answer for the NC school district employment comparison task.
    """

    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root as parallel to allow partial credit aggregation
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

    # 1) Extract all districts and details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Ensure we have a list and select the first 4 for per-district checks
    all_districts: List[DistrictInfo] = extracted.districts or []
    first_four: List[DistrictInfo] = _select_first_k(all_districts, 4)
    # Pad with empty entries to always have 4 nodes for structure
    while len(first_four) < 4:
        first_four.append(DistrictInfo())

    # 2) District set requirements (critical)
    set_req = evaluator.add_parallel(
        id="district_set_requirements",
        desc="The response selects the correct number and type of districts.",
        parent=root,
        critical=True
    )

    # 2.1 Exactly four districts were provided (as per the answer, not just we sliced to 4)
    total_unique_in_answer = _unique_names([d.name for d in all_districts])
    evaluator.add_custom_node(
        result=(total_unique_in_answer == 4),
        id="exactly_four_districts",
        desc="Provides information for exactly four school districts.",
        parent=set_req,
        critical=True
    )

    # 2.2 All four are NC public school districts (verify each of the first four against its employment URL)
    nc_all = evaluator.add_parallel(
        id="all_are_nc_public_school_districts",
        desc="All four are North Carolina public school districts.",
        parent=set_req,
        critical=True
    )
    # Build leaves and batch verify
    claims_sources_nodes: List[Tuple[str, Optional[str], Any, Optional[str]]] = []
    for i, d in enumerate(first_four):
        leaf = evaluator.add_leaf(
            id=f"d{i+1}_is_nc_public_district",
            desc=f"District {i+1} is a North Carolina public school district (verified by the cited page).",
            parent=nc_all,
            critical=True
        )
        claim = f"This page belongs to the official website of a North Carolina public school district named '{_norm(d.name)}' (or a close variant) and is relevant to employment/careers."
        add_ins = "Accept official district sites, especially *.k12.nc.us or clearly official district domains. Look for references to 'North Carolina' or NC context."
        claims_sources_nodes.append((claim, d.employment_url if _is_valid_url(d.employment_url) else None, leaf, add_ins))

    # Parallel verification
    await evaluator.batch_verify(claims_sources_nodes)

    # 2.3 Districts are distinct (first four must be different)
    distinct_first_four = _unique_names([d.name for d in first_four]) == 4
    evaluator.add_custom_node(
        result=distinct_first_four,
        id="districts_are_distinct",
        desc="All four districts are different (no duplicates).",
        parent=set_req,
        critical=True
    )

    # 3) Certification type coverage across districts (critical)
    cov = evaluator.add_parallel(
        id="certification_type_coverage",
        desc="Across the four districts, the response includes each required qualification/certification category at least once.",
        parent=root,
        critical=True
    )

    # Collect candidate URLs for each category
    teacher_urls = _dedup_urls([u for d in first_four for u in (d.teacher_cert.sources or [])])
    admin_urls = _dedup_urls([u for d in first_four for u in (d.admin_requirements.sources or [])])
    coach_urls = _dedup_urls([u for d in first_four for u in (d.coaching_requirements.sources or [])])

    await add_coverage_verification(
        evaluator,
        cov,
        node_base_id="teacher_cert_coverage",
        description="At least one district includes teacher certification requirements (degree, licensure, and any alternative pathways).",
        candidate_urls=teacher_urls,
        claim_text="This page states teacher certification requirements for a North Carolina public school district, including degree, state licensure, or alternative pathways.",
        add_ins="Look for explicit references to teacher licensure (e.g., NC DPI license), degree requirements, and/or alternative certification pathways."
    )
    await add_coverage_verification(
        evaluator,
        cov,
        node_base_id="admin_cert_coverage",
        description="At least one district includes school administrator requirements (degree and required experience).",
        candidate_urls=admin_urls,
        claim_text="This page states qualification requirements for school administrators (e.g., principals/assistant principals), including degree and required experience.",
        add_ins="Look for administrator job postings/HR pages mentioning degree requirements and experience expectations for administrators."
    )
    await add_coverage_verification(
        evaluator,
        cov,
        node_base_id="coaching_cert_coverage",
        description="At least one district includes coaching position requirements (e.g., required safety/training certifications).",
        candidate_urls=coach_urls,
        claim_text="This page states coaching position requirements, such as mandatory safety/training certifications (e.g., NFHS, CPR/AED).",
        add_ins="Accept district athletics/HR/coaching handbook pages that explicitly list required certifications/training for coaches."
    )

    # 4) Per-district verification subtrees (non-critical parent nodes, but contain critical checks)
    for i, d in enumerate(first_four):
        await verify_one_district(evaluator, root, d, i)

    # Record a compact custom info block
    evaluator.add_custom_info(
        info={
            "all_extracted_district_count": len(all_districts),
            "unique_names_in_answer": total_unique_in_answer,
            "evaluated_districts": [(_norm(d.name), d.employment_url) for d in first_four],
        },
        info_type="evaluation_metadata",
        info_name="evaluation_overview"
    )

    # 5) Return standard summary
    return evaluator.get_summary()