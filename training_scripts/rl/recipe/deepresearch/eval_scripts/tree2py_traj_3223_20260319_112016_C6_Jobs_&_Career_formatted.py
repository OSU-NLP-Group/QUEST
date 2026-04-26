import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "faculty_opps_benefits_three_universities"
TASK_DESCRIPTION = """A candidate with a recently completed doctoral degree in Education is evaluating early-career academic opportunities and has particular interest in institutions that offer strong employee benefits, specifically retirement plans and tuition assistance. They have identified Belmont University, Clemson University, and the University of Virginia as potential employers.

For each of these three institutions, research and provide:

1. Whether the institution currently has any faculty fellowship programs or tenure-track positions suitable for recent doctoral graduates in education or related fields
2. The educational requirements (doctoral degree acceptance and timing)
3. What application materials are typically required for faculty positions (CV, teaching statement, research statement, letters of recommendation, etc.)
4. What retirement benefits the institution offers to faculty (type of plan, employer contribution percentage if applicable)
5. What tuition assistance benefits are available (specify whether benefits are for employees only, or also available for dependent children)
6. Where to find and access current job postings and how to apply

Provide specific details with URL references for all information.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    program_name: Optional[str] = None
    description: Optional[str] = None
    field_suitability: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PositionsSection(BaseModel):
    items: List[PositionItem] = Field(default_factory=list)


class EducationSection(BaseModel):
    doctoral_required: Optional[str] = None
    completion_timing: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ApplicationMaterialsSection(BaseModel):
    cv_required: Optional[str] = None
    statements_required: List[str] = Field(default_factory=list)
    other_materials: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class RetirementSection(BaseModel):
    plan_type: Optional[str] = None
    employer_contribution: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TuitionSection(BaseModel):
    employee_coverage: Optional[str] = None
    dependent_coverage: Optional[str] = None
    benefit_amount: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PortalSection(BaseModel):
    portal_location: Optional[str] = None
    application_process: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Institution(BaseModel):
    institution_name: Optional[str] = None
    positions: PositionsSection = Field(default_factory=PositionsSection)
    education: EducationSection = Field(default_factory=EducationSection)
    materials: ApplicationMaterialsSection = Field(default_factory=ApplicationMaterialsSection)
    retirement: RetirementSection = Field(default_factory=RetirementSection)
    tuition: TuitionSection = Field(default_factory=TuitionSection)
    portal: PortalSection = Field(default_factory=PortalSection)


class InstitutionsExtraction(BaseModel):
    belmont: Institution = Field(default_factory=Institution)
    clemson: Institution = Field(default_factory=Institution)
    uva: Institution = Field(default_factory=Institution)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
Extract structured information for EACH of the following institutions based ONLY on the provided answer text: Belmont University, Clemson University, and the University of Virginia.

For each institution, extract the following sections and fields:

- positions:
  - items: an array; include up to 3 concrete position/fellowship/hiring pathway entries specific to this institution (if any).
    For each item:
      • program_name: exact title of the position or program (e.g., "Assistant Professor of Education", "Postdoctoral Fellow", "Teaching Fellowship") as stated in the answer
      • description: one-sentence summary of the role or key duties/requirements as stated
      • field_suitability: short note indicating the relevant field (e.g., "Education", "Social Sciences", or similar) as stated
      • urls: all URLs in the answer directly referencing the listing or program information page(s)

- education:
  • doctoral_required: the exact phrasing related to doctoral/terminal degree requirement (e.g., "PhD required", "EdD required/terminal degree", "PhD by start date acceptable"), as stated
  • completion_timing: timing language if present (e.g., "by start date", "by time of appointment", "at application")
  • urls: URLs that substantiate these requirements

- materials:
  • cv_required: exact phrasing indicating CV/résumé requirement if present
  • statements_required: array of any required statements (e.g., "teaching statement", "research statement", "diversity statement", "teaching philosophy")
  • other_materials: array of other required items (e.g., "cover letter", "letters of recommendation", "references", "transcripts", "writing sample")
  • urls: URLs that substantiate application material requirements

- retirement:
  • plan_type: exact plan type language if present (e.g., "403(b)", "401(a)", "state pension/defined benefit", "ORP")
  • employer_contribution: employer contribution wording or percentage if present (e.g., "8% employer contribution"), else null
  • urls: URLs for faculty retirement info

- tuition:
  • employee_coverage: verbatim statement indicating whether tuition benefits apply to employees (e.g., "eligible employees receive..."; if not stated, null)
  • dependent_coverage: verbatim statement indicating whether benefits extend to dependent children (explicit yes/no with details if available; else null)
  • benefit_amount: the percentage/amount/scope as stated (e.g., "100% tuition remission for up to 6 credits/term"), else null
  • urls: URLs for tuition benefit info

- portal:
  • portal_location: the named portal/site/system where jobs are posted (e.g., "Workday", "Interfolio", "University HR jobs site", "PeopleAdmin")
  • application_process: short summary describing how to search and apply (as stated)
  • urls: URLs for the job portal or application system

Return a JSON object with three top-level keys: "belmont", "clemson", and "uva". For any field not explicitly present in the answer text, return null (for strings) or an empty list (for arrays). Only include URLs that are explicitly present in the answer text (plain URLs or in markdown).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(u: Optional[str]) -> bool:
    if not u or not isinstance(u, str):
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not _is_valid_url(s):
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _any_urls_present(urls: Optional[List[str]]) -> bool:
    return len(_clean_urls(urls)) > 0


def _gather_position_urls(positions: PositionsSection) -> List[str]:
    agg: List[str] = []
    for it in positions.items:
        agg.extend(it.urls or [])
    return _clean_urls(agg)


def _pick_primary_position(positions: PositionsSection) -> Optional[PositionItem]:
    # Prefer first item that has both a name/description and at least one URL
    for it in positions.items:
        if (it.program_name or it.description) and _any_urls_present(it.urls):
            return it
    # Then any item that has at least a name/description
    for it in positions.items:
        if it.program_name or it.description:
            return it
    # Otherwise first item if any
    return positions.items[0] if positions.items else None


def _first_non_empty(*url_lists: List[str]) -> List[str]:
    for lst in url_lists:
        cleaned = _clean_urls(lst)
        if cleaned:
            return cleaned
    return []


def _says_no(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    negatives = ["no", "not", "ineligible", "does not", "isn't", "aren't", "excluded"]
    return any(n in t for n in negatives)


# --------------------------------------------------------------------------- #
# Section verification builders                                                #
# --------------------------------------------------------------------------- #
async def build_positions_section(
    evaluator: Evaluator,
    parent,
    inst_name: str,
    inst_prefix: str,
    positions: PositionsSection,
    position_desc_for_institution: Optional[str] = None
) -> None:
    # Parent node (parallel)
    pos_node = evaluator.add_parallel(
        id=f"{inst_prefix}_position_availability",
        desc=position_desc_for_institution or "Faculty positions or fellowship programs for recent doctoral graduates are identified",
        parent=parent,
        critical=False
    )

    pos_urls = _gather_position_urls(positions)
    primary = _pick_primary_position(positions)
    # Common sources for this section
    sources = pos_urls

    # program_exists (critical, verify with URLs if possible)
    leaf_program_exists = evaluator.add_leaf(
        id=f"{inst_prefix}_program_exists",
        desc="At least one faculty position, fellowship program, or hiring pathway for recent PhDs is identified",
        parent=pos_node,
        critical=True
    )
    if primary and primary.program_name:
        claim = f"The provided page(s) show a real listing or program at {inst_name} named '{primary.program_name}', which is a faculty position, fellowship, or an early-career hiring pathway appropriate for recent PhD graduates."
    else:
        claim = f"The provided page(s) show at least one faculty position, fellowship, or early-career hiring pathway at {inst_name} suitable for recent PhD graduates."
    await evaluator.verify(
        claim=claim,
        node=leaf_program_exists,
        sources=sources,
        additional_instruction="Look for explicit job or fellowship listings or official faculty hiring info indicating opportunities suitable for recent PhDs (e.g., Assistant Professor, Postdoctoral Fellow, or similar)."
    )

    # field_appropriate (critical)
    leaf_field = evaluator.add_leaf(
        id=f"{inst_prefix}_field_appropriate",
        desc="Position or program is suitable for education or related social science fields",
        parent=pos_node,
        critical=True
    )
    if primary and primary.field_suitability:
        claim_f = f"The page(s) indicate that the position/program is in '{primary.field_suitability}' or otherwise suitable for Education or related social science fields at {inst_name}."
    else:
        claim_f = f"The page(s) indicate that at least one listed position/program at {inst_name} is within Education or a closely related social science field."
    await evaluator.verify(
        claim=claim_f,
        node=leaf_field,
        sources=sources,
        additional_instruction="Accept if the page references a School/College of Education or education-related disciplines (e.g., curriculum & instruction, learning sciences, education policy). General faculty hiring pages that include Education positions also qualify."
    )

    # position_details (critical)
    leaf_details = evaluator.add_leaf(
        id=f"{inst_prefix}_position_details",
        desc="Specific program/position name and basic description provided",
        parent=pos_node,
        critical=True
    )
    if primary and primary.program_name and primary.description:
        claim_d = f"The page(s) provide a specific position/program name '{primary.program_name}' and include a brief description of the role or requirements."
    else:
        claim_d = "The page(s) provide at least one specific position/program name and a basic description (duties/qualifications/summary) for that listing."
    await evaluator.verify(
        claim=claim_d,
        node=leaf_details,
        sources=sources,
        additional_instruction="A short description can appear in the overview, responsibilities, or qualifications section."
    )

    # url_positions (critical - custom existence check for URL presence in the answer)
    evaluator.add_custom_node(
        result=_any_urls_present(pos_urls),
        id=f"{inst_prefix}_url_positions",
        desc="URL provided for position listing or program information",
        parent=pos_node,
        critical=True
    )


async def build_education_section(
    evaluator: Evaluator,
    parent,
    inst_name: str,
    inst_prefix: str,
    edu: EducationSection,
    fallback_sources: List[str]
) -> None:
    edu_node = evaluator.add_parallel(
        id=f"{inst_prefix}_education_requirements",
        desc="Educational qualification requirements are clearly documented",
        parent=parent,
        critical=False
    )
    edu_urls = _clean_urls(edu.urls)
    sources = _first_non_empty(edu_urls, fallback_sources)

    # doctoral_required (critical)
    leaf_doc = evaluator.add_leaf(
        id=f"{inst_prefix}_doctoral_required",
        desc="Doctoral or terminal degree requirement is confirmed",
        parent=edu_node,
        critical=True
    )
    if edu.doctoral_required:
        claim = f"The page(s) state that a doctoral or terminal degree requirement/acceptance applies for relevant faculty roles at {inst_name}: '{edu.doctoral_required}'."
    else:
        claim = f"The page(s) state that a doctoral or terminal degree is required or acceptable for relevant faculty roles at {inst_name}."
    await evaluator.verify(
        claim=claim,
        node=leaf_doc,
        sources=sources,
        additional_instruction="Accept phrasing such as PhD/EdD required, terminal degree required, or similar."
    )

    # completion_timing (critical)
    leaf_timing = evaluator.add_leaf(
        id=f"{inst_prefix}_completion_timing",
        desc="Information about whether degree must be completed at application or by start date is provided",
        parent=edu_node,
        critical=True
    )
    if edu.completion_timing:
        claim_t = f"The page(s) explain timing for doctoral degree completion (e.g., by start date or at application) at {inst_name}: '{edu.completion_timing}'."
    else:
        claim_t = "The page(s) indicate whether the doctoral degree must be completed by the start date or at the time of application."
    await evaluator.verify(
        claim=claim_t,
        node=leaf_timing,
        sources=sources,
        additional_instruction="Look for statements like 'PhD in hand by start date' or 'completed by time of appointment'."
    )

    # url_education (critical, custom existence)
    evaluator.add_custom_node(
        result=_any_urls_present(edu_urls),
        id=f"{inst_prefix}_url_education",
        desc="URL reference for educational requirements",
        parent=edu_node,
        critical=True
    )


async def build_materials_section(
    evaluator: Evaluator,
    parent,
    inst_name: str,
    inst_prefix: str,
    mats: ApplicationMaterialsSection,
    fallback_sources: List[str]
) -> None:
    mats_node = evaluator.add_parallel(
        id=f"{inst_prefix}_application_materials",
        desc="Required application materials are documented",
        parent=parent,
        critical=False
    )
    mats_urls = _clean_urls(mats.urls)
    sources = _first_non_empty(mats_urls, fallback_sources)

    # cv_required (critical)
    leaf_cv = evaluator.add_leaf(
        id=f"{inst_prefix}_cv_required",
        desc="CV or curriculum vitae is confirmed as required",
        parent=mats_node,
        critical=True
    )
    if mats.cv_required:
        claim_cv = f"The page(s) state that a Curriculum Vitae (CV) is required for applications at {inst_name}: '{mats.cv_required}'."
    else:
        claim_cv = "The application materials section indicates that a Curriculum Vitae (CV) or résumé is required."
    await evaluator.verify(
        claim=claim_cv,
        node=leaf_cv,
        sources=sources,
        additional_instruction="Accept 'CV', 'curriculum vitae', or 'resume' for this check; faculty typically uses CV."
    )

    # statements_required (critical)
    leaf_stmt = evaluator.add_leaf(
        id=f"{inst_prefix}_statements_required",
        desc="Teaching statement, research statement, or similar documents are identified as required",
        parent=mats_node,
        critical=True
    )
    if mats.statements_required:
        claim_stmt = f"The page(s) indicate at least one required statement for applications at {inst_name}: {mats.statements_required}."
    else:
        claim_stmt = "The application requires at least one formal statement such as a teaching statement, research statement, teaching philosophy, or diversity statement."
    await evaluator.verify(
        claim=claim_stmt,
        node=leaf_stmt,
        sources=sources,
        additional_instruction="Any one among teaching statement, research statement, teaching philosophy, or diversity statement satisfies this."
    )

    # other_materials (non-critical) - verify presence of additional required items beyond CV/statements
    leaf_other = evaluator.add_leaf(
        id=f"{inst_prefix}_other_materials",
        desc="Other required materials (letters, cover letter, essays, etc.) are listed",
        parent=mats_node,
        critical=False
    )
    if mats.other_materials:
        claim_other = f"The application also requires other materials beyond CV/statements at {inst_name}, such as: {mats.other_materials}."
    else:
        claim_other = "The application requires at least one other material beyond CV and statements (e.g., cover letter, letters of recommendation, references, transcripts, writing sample)."
    await evaluator.verify(
        claim=claim_other,
        node=leaf_other,
        sources=sources,
        additional_instruction="If any such additional item is listed on the page(s), pass."
    )

    # url_application (critical, custom existence)
    evaluator.add_custom_node(
        result=_any_urls_present(mats_urls),
        id=f"{inst_prefix}_url_application",
        desc="URL reference for application requirements",
        parent=mats_node,
        critical=True
    )


async def build_retirement_section(
    evaluator: Evaluator,
    parent,
    inst_name: str,
    inst_prefix: str,
    ret: RetirementSection
) -> None:
    ret_node = evaluator.add_parallel(
        id=f"{inst_prefix}_retirement_benefits",
        desc="Retirement benefit offerings are documented with specifics",
        parent=parent,
        critical=False
    )
    ret_urls = _clean_urls(ret.urls)
    sources = ret_urls

    # plan_type (critical)
    leaf_plan = evaluator.add_leaf(
        id=f"{inst_prefix}_plan_type",
        desc="Type of retirement plan offered (403b, pension, defined benefit, etc.) is specified",
        parent=ret_node,
        critical=True
    )
    if ret.plan_type:
        claim_plan = f"The page(s) indicate the faculty retirement plan type at {inst_name} is '{ret.plan_type}'."
    else:
        claim_plan = f"The page(s) specify the type of faculty retirement plan offered at {inst_name} (e.g., 403(b), 401(a), ORP, or state pension/defined benefit)."
    await evaluator.verify(
        claim=claim_plan,
        node=leaf_plan,
        sources=sources,
        additional_instruction="Accept any clear plan type description specific to faculty."
    )

    # employer_contribution (non-critical presence)
    evaluator.add_custom_node(
        result=bool(ret.employer_contribution and ret.employer_contribution.strip()),
        id=f"{inst_prefix}_employer_contribution",
        desc="Employer contribution amount or percentage is provided if applicable",
        parent=ret_node,
        critical=False
    )

    # url_retirement (critical, custom existence)
    evaluator.add_custom_node(
        result=_any_urls_present(ret_urls),
        id=f"{inst_prefix}_url_retirement",
        desc="URL reference for retirement benefits information",
        parent=ret_node,
        critical=True
    )


async def build_tuition_section(
    evaluator: Evaluator,
    parent,
    inst_name: str,
    inst_prefix: str,
    tui: TuitionSection
) -> None:
    tui_node = evaluator.add_parallel(
        id=f"{inst_prefix}_tuition_benefits",
        desc="Tuition assistance benefits are documented with coverage details",
        parent=parent,
        critical=False
    )
    tui_urls = _clean_urls(tui.urls)
    sources = tui_urls

    # employee_coverage (critical)
    leaf_emp = evaluator.add_leaf(
        id=f"{inst_prefix}_employee_coverage",
        desc="Confirmation that tuition benefits are available for employees",
        parent=tui_node,
        critical=True
    )
    if _says_no(tui.employee_coverage):
        claim_emp = f"The page(s) indicate that tuition assistance benefits are NOT available to employees at {inst_name}."
    else:
        # default to positive claim if ambiguous
        if tui.employee_coverage:
            claim_emp = f"The page(s) indicate that tuition assistance benefits are available to employees at {inst_name}: '{tui.employee_coverage}'."
        else:
            claim_emp = f"The page(s) indicate that tuition assistance benefits are available to employees at {inst_name}."
    await evaluator.verify(
        claim=claim_emp,
        node=leaf_emp,
        sources=sources,
        additional_instruction="Look for explicit tuition remission/assistance eligibility for employees; if page clearly excludes employees, mark as not available."
    )

    # dependent_coverage (critical)
    leaf_dep = evaluator.add_leaf(
        id=f"{inst_prefix}_dependent_coverage",
        desc="Clear statement of whether tuition benefits extend to dependent children (yes/no with details)",
        parent=tui_node,
        critical=True
    )
    if _says_no(tui.dependent_coverage):
        claim_dep = f"The page(s) indicate that tuition assistance benefits do NOT extend to dependent children at {inst_name}."
    else:
        if tui.dependent_coverage:
            claim_dep = f"The page(s) indicate that tuition assistance benefits extend to dependent children at {inst_name}: '{tui.dependent_coverage}'."
        else:
            claim_dep = f"The page(s) indicate whether tuition assistance benefits extend to dependent children at {inst_name}."
    await evaluator.verify(
        claim=claim_dep,
        node=leaf_dep,
        sources=sources,
        additional_instruction="Accept if the page clearly indicates coverage for dependents/children; if it clearly excludes dependents, that also satisfies clarity."
    )

    # benefit_amount (non-critical presence)
    evaluator.add_custom_node(
        result=bool(tui.benefit_amount and tui.benefit_amount.strip()),
        id=f"{inst_prefix}_benefit_amount",
        desc="Amount, percentage, or scope of tuition benefit is provided",
        parent=tui_node,
        critical=False
    )

    # url_tuition (critical, custom existence)
    evaluator.add_custom_node(
        result=_any_urls_present(tui_urls),
        id=f"{inst_prefix}_url_tuition",
        desc="URL reference for tuition benefit information",
        parent=tui_node,
        critical=True
    )


async def build_portal_section(
    evaluator: Evaluator,
    parent,
    inst_name: str,
    inst_prefix: str,
    portal: PortalSection
) -> None:
    portal_node = evaluator.add_parallel(
        id=f"{inst_prefix}_application_portal",
        desc="Information about where and how to apply is provided",
        parent=parent,
        critical=False
    )
    p_urls = _clean_urls(portal.urls)
    sources = p_urls

    # portal_location (critical)
    leaf_loc = evaluator.add_leaf(
        id=f"{inst_prefix}_portal_location",
        desc="The job portal, website, or system where positions are posted is identified",
        parent=portal_node,
        critical=True
    )
    if portal.portal_location:
        claim_loc = f"The page(s) identify the official job portal/system for {inst_name} as '{portal.portal_location}'."
    else:
        claim_loc = f"The page(s) identify the official job portal/system for {inst_name}."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=sources,
        additional_instruction="Accept named systems such as Workday, Interfolio, PeopleAdmin, or an official university HR jobs site."
    )

    # application_process (critical)
    leaf_proc = evaluator.add_leaf(
        id=f"{inst_prefix}_application_process",
        desc="How to search for and apply to positions is explained",
        parent=portal_node,
        critical=True
    )
    if portal.application_process:
        claim_proc = f"The page(s) explain how to search and apply for positions at {inst_name}: '{portal.application_process}'."
    else:
        claim_proc = "The page(s) provide clear instructions on how to search for positions and submit an application."
    await evaluator.verify(
        claim=claim_proc,
        node=leaf_proc,
        sources=sources,
        additional_instruction="Look for steps such as searching the portal, selecting a posting, and submitting materials."
    )

    # url_portal (critical, custom existence)
    evaluator.add_custom_node(
        result=_any_urls_present(p_urls),
        id=f"{inst_prefix}_url_portal",
        desc="URL for job portal or application system",
        parent=portal_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Institution wrapper                                                          #
# --------------------------------------------------------------------------- #
async def verify_institution(
    evaluator: Evaluator,
    root,
    inst_key: str,
    inst_name: str,
    inst: Institution
) -> None:
    # Parent node for the institution (parallel, non-critical)
    inst_prefix = f"{inst_key}"
    inst_node = evaluator.add_parallel(
        id=f"institution_{inst_prefix}",
        desc=f"Complete information gathered for {inst_name}",
        parent=root,
        critical=False
    )

    # Build sections
    # Positions (slightly different wording per institution; optional custom desc)
    pos_desc = None
    if inst_key == "clemson":
        pos_desc = "Faculty positions or fellowship programs for recent doctoral graduates are identified (or general availability/hiring information is provided)"
    elif inst_key == "uva":
        pos_desc = "Faculty positions or hiring information for recent PhDs is provided"

    await build_positions_section(
        evaluator=evaluator,
        parent=inst_node,
        inst_name=inst_name,
        inst_prefix=f"{inst_prefix}",
        positions=inst.positions,
        position_desc_for_institution=pos_desc
    )

    # Fallback sources for edu/materials: use position URLs if section URLs missing
    fallback_sources = _gather_position_urls(inst.positions)

    await build_education_section(
        evaluator=evaluator,
        parent=inst_node,
        inst_name=inst_name,
        inst_prefix=f"{inst_prefix}",
        edu=inst.education,
        fallback_sources=fallback_sources
    )

    await build_materials_section(
        evaluator=evaluator,
        parent=inst_node,
        inst_name=inst_name,
        inst_prefix=f"{inst_prefix}",
        mats=inst.materials,
        fallback_sources=fallback_sources
    )

    await build_retirement_section(
        evaluator=evaluator,
        parent=inst_node,
        inst_name=inst_name,
        inst_prefix=f"{inst_prefix}",
        ret=inst.retirement
    )

    await build_tuition_section(
        evaluator=evaluator,
        parent=inst_node,
        inst_name=inst_name,
        inst_prefix=f"{inst_prefix}",
        tui=inst.tuition
    )

    await build_portal_section(
        evaluator=evaluator,
        parent=inst_node,
        inst_name=inst_name,
        inst_prefix=f"{inst_prefix}",
        portal=inst.portal
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                  #
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
    Evaluate an answer for faculty opportunities and benefits across Belmont, Clemson, and UVA.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Research and compare faculty employment opportunities and benefits at three specified universities for a recent doctoral graduate",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # 1) Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
        additional_instruction="Extract only what is explicitly present in the answer text. Return null for missing strings and empty arrays for missing lists."
    )

    # Add high-level context info (optional)
    evaluator.add_custom_info(
        info={
            "institutions": ["Belmont University", "Clemson University", "University of Virginia"],
            "task_focus": ["positions/fellowships", "education requirements", "application materials", "retirement benefits", "tuition benefits", "application portal"]
        },
        info_type="task_context"
    )

    # 2) Build verification tree per institution
    await verify_institution(
        evaluator=evaluator,
        root=root,
        inst_key="belmont",
        inst_name="Belmont University",
        inst=extraction.belmont
    )

    await verify_institution(
        evaluator=evaluator,
        root=root,
        inst_key="clemson",
        inst_name="Clemson University",
        inst=extraction.clemson
    )

    await verify_institution(
        evaluator=evaluator,
        root=root,
        inst_key="uva",
        inst_name="University of Virginia",
        inst=extraction.uva
    )

    # 3) Return summary
    return evaluator.get_summary()