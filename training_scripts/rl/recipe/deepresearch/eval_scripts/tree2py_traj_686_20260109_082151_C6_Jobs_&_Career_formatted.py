import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "cfo_f500_healthcare_2024_2025"
TASK_DESCRIPTION = (
    "Identify a Chief Financial Officer (CFO) of a top 10 Fortune 500 healthcare company "
    "(ranked in 2024 or 2025) who was appointed to their current CFO role in 2024 or 2025, "
    "and who meets all education/experience/governance constraints. Provide the name of the CFO, "
    "the current company, and reference URLs verifying each constraint."
)

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class CompanyConstraintURLs(BaseModel):
    fortune_rank_year: Optional[str] = None  # "2024" or "2025" if explicitly stated
    fortune_rank_urls: List[str] = Field(default_factory=list)
    nyse_listed_urls: List[str] = Field(default_factory=list)
    board_independence_urls: List[str] = Field(default_factory=list)


class EducationInfo(BaseModel):
    bachelors_degree: Optional[str] = None  # e.g., "B.S. in Accounting"
    bachelors_field: Optional[str] = None   # e.g., "Accounting"
    bachelors_urls: List[str] = Field(default_factory=list)
    mba_degree: Optional[str] = None        # e.g., "MBA from Harvard Business School"
    mba_urls: List[str] = Field(default_factory=list)


class ExperienceInfo(BaseModel):
    total_experience_text: Optional[str] = None  # e.g., "Over 25 years in finance leadership"
    total_experience_urls: List[str] = Field(default_factory=list)
    prior_public_cfo_text: Optional[str] = None  # e.g., "Previously CFO at XYZ (NYSE: XYZ)"
    prior_public_cfo_urls: List[str] = Field(default_factory=list)
    big4_or_md_text: Optional[str] = None        # e.g., "Former Partner at KPMG" or "Managing Director at Goldman Sachs"
    big4_or_md_urls: List[str] = Field(default_factory=list)


class AppointmentInfo(BaseModel):
    appointment_date: Optional[str] = None   # e.g., "March 4, 2025"
    appointment_year: Optional[str] = None   # "2024" or "2025"
    appointment_urls: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    cfo_name: Optional[str] = None
    company_name: Optional[str] = None
    company: CompanyConstraintURLs = Field(default_factory=CompanyConstraintURLs)
    education: EducationInfo = Field(default_factory=EducationInfo)
    experience: ExperienceInfo = Field(default_factory=ExperienceInfo)
    appointment: AppointmentInfo = Field(default_factory=AppointmentInfo)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate() -> str:
    return """
    Extract the single CFO candidate and their current company from the answer, along with URLs the answer cites to verify each specific constraint. 
    You must not invent any information or URLs; only extract what appears in the answer.

    Required top-level fields:
    - cfo_name: the person's full name who is the CFO candidate
    - company_name: the current company where this person is serving as CFO

    Appointment (current CFO role):
    - appointment:
        - appointment_date: exact phrasing for the appointment/effective date if provided (string)
        - appointment_year: the 4-digit year of appointment if explicitly present (prefer 2024 or 2025). Use a 4-digit string if present; otherwise null.
        - appointment_urls: list of URLs explicitly cited that verify the appointment timing (do not fabricate)

    Education:
    - education:
        - bachelors_degree: the bachelor's degree text as stated (e.g., "B.S. in Accounting", "BBA in Finance"), if present
        - bachelors_field: the field/major for the bachelor's (e.g., Accounting, Finance, Business Administration), if explicitly stated
        - bachelors_urls: list of URLs that verify the bachelor's degree/field (explicit URLs only)
        - mba_degree: the MBA text as stated (e.g., "MBA from Wharton"), if present
        - mba_urls: list of URLs that verify the MBA (explicit URLs only)

    Experience:
    - experience:
        - total_experience_text: the text indicating total years of finance-related experience (e.g., "more than 20 years", "over two decades"), if provided
        - total_experience_urls: list of URLs verifying the total experience claim (explicit URLs only)
        - prior_public_cfo_text: the text describing previously serving as CFO of another publicly traded company (before the current role), if provided
        - prior_public_cfo_urls: list of URLs verifying the prior public-company CFO service (explicit URLs only)
        - big4_or_md_text: the text describing either (A) being a partner at Deloitte/PwC/EY/KPMG, or (B) holding a Managing Director (or equivalent senior finance leadership) role at a major financial institution (e.g., Goldman Sachs, JPMorgan, Morgan Stanley, Bank of America, Citi), if provided
        - big4_or_md_urls: list of URLs verifying the qualifying Big Four partner or major financial institution MD-level (or equivalent) role (explicit URLs only)

    Company constraints:
    - company:
        - fortune_rank_year: the specific year "2024" or "2025" if explicitly mentioned for the Fortune 500 healthcare ranking; otherwise null
        - fortune_rank_urls: list of URLs that verify the company is ranked among the top 10 Fortune 500 healthcare companies in 2024 or 2025
        - nyse_listed_urls: list of URLs that verify the company is listed on the NYSE
        - board_independence_urls: list of URLs that verify that a majority (>50%) of the board are independent directors (as per NYSE independence concept)

    URL extraction rules:
    - Extract only valid URLs that appear in the answer content. Do not invent or infer any URLs.
    - Accept plain URLs or Markdown links; extract the actual URL string.
    - If a URL is missing the protocol, prepend "http://" as needed.
    - If the answer does not provide a URL for a particular item, return an empty list for that URL field.

    If any specific field is not present in the answer, set it to null (for single values) or [] (for URL lists).
    """

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order; keep only non-empty strings.
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _safe_cfo_name(data: CandidateExtraction) -> str:
    return data.cfo_name.strip() if _non_empty(data.cfo_name) else "the CFO"


def _safe_company_name(data: CandidateExtraction) -> str:
    return data.company_name.strip() if _non_empty(data.company_name) else "the company"


async def _add_ref_and_support_group(
    evaluator: Evaluator,
    *,
    parent,
    group_id: str,
    group_desc: str,
    ref_urls: List[str],
    support_leaf_desc: str,
    claim: str,
    additional_instruction: str,
    critical: bool = True,
) -> None:
    """
    Create a critical parallel group that requires: (1) at least one reference URL provided; (2) claim is supported by provided URLs.
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=critical,
    )

    # 1) Reference provided (critical)
    evaluator.add_custom_node(
        result=len(ref_urls) > 0,
        id=f"{group_id}_refs_provided",
        desc=f"At least one reference URL is provided for '{group_id}'",
        parent=group_node,
        critical=True,
    )

    # 2) Support by URLs (critical)
    support_leaf = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc=support_leaf_desc,
        parent=group_node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=ref_urls,
        additional_instruction=additional_instruction,
    )

# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_required_outputs_section(evaluator: Evaluator, root, data: CandidateExtraction) -> None:
    """
    Root child: required_outputs_present (critical).
    Split into two binary custom leaves for clarity: cfo_name_present and company_name_present.
    """
    required_node = evaluator.add_parallel(
        id="required_outputs_present",
        desc="Answer provides the CFO name and current company name",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty(data.cfo_name),
        id="cfo_name_present",
        desc="CFO name is provided",
        parent=required_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty(data.company_name),
        id="company_name_present",
        desc="Current company name is provided",
        parent=required_node,
        critical=True,
    )


async def build_company_constraints_section(evaluator: Evaluator, root, data: CandidateExtraction) -> None:
    """
    Root child: company_constraints (critical, parallel)
    Contains three critical sub-groups, each enforcing references + support:
      - fortune_500_top10_healthcare_rank_with_reference
      - nyse_listed_with_reference
      - board_majority_independent_with_reference
    """
    company_constraints = evaluator.add_parallel(
        id="company_constraints",
        desc="Current company satisfies company-level constraints (with verifying references)",
        parent=root,
        critical=True,
    )

    company = _safe_company_name(data)

    # Fortune 500 top-10 healthcare in 2024 or 2025
    fortune_urls = _norm_urls(data.company.fortune_rank_urls)
    year_str = data.company.fortune_rank_year.strip() if _non_empty(data.company.fortune_rank_year) else "2024 or 2025"
    await _add_ref_and_support_group(
        evaluator,
        parent=company_constraints,
        group_id="fortune_500_top10_healthcare_rank_with_reference",
        group_desc="Company is ranked among the top 10 Fortune 500 healthcare companies in 2024 or 2025 AND references are provided",
        ref_urls=fortune_urls,
        support_leaf_desc="Top-10 Fortune 500 healthcare ranking is supported by the cited sources",
        claim=f"{company} is ranked among the top 10 Fortune 500 healthcare companies in {year_str}.",
        additional_instruction=(
            "Verify that the provided source(s) explicitly indicate the company is within the top 10 for the "
            "Fortune 500 'Health Care' industry subset in 2024 or 2025. Accept synonyms like 'Health Care' or "
            "'Healthcare'. The page should clearly show ranking/top-10 status for the specified year or explicitly "
            "list a top-10 list where the company appears."
        ),
        critical=True,
    )

    # NYSE listed
    nyse_urls = _norm_urls(data.company.nyse_listed_urls)
    await _add_ref_and_support_group(
        evaluator,
        parent=company_constraints,
        group_id="nyse_listed_with_reference",
        group_desc="Company is listed on the NYSE AND references are provided",
        ref_urls=nyse_urls,
        support_leaf_desc="NYSE listing is supported by the cited sources",
        claim=f"{company} is listed on the New York Stock Exchange (NYSE).",
        additional_instruction=(
            "Confirm that the company is listed on NYSE. Accept notations like 'NYSE: TICKER' or an official NYSE "
            "listing page or company IR/SEC page stating NYSE listing. Do not accept NASDAQ-only listings."
        ),
        critical=True,
    )

    # Board majority independent
    board_urls = _norm_urls(data.company.board_independence_urls)
    await _add_ref_and_support_group(
        evaluator,
        parent=company_constraints,
        group_id="board_majority_independent_with_reference",
        group_desc="Company board has a majority (>50%) independent directors AND references are provided",
        ref_urls=board_urls,
        support_leaf_desc="Board majority independence is supported by the cited sources",
        claim=f"A majority (more than 50%) of {company}'s board of directors are independent.",
        additional_instruction=(
            "Look for explicit statements such as 'a majority of our directors are independent' or data/numbers that "
            "clearly show >50% of directors meet NYSE independence standards. Proxy statements, governance pages, or "
            "corporate governance guidelines are acceptable sources."
        ),
        critical=True,
    )


async def build_appointment_section(evaluator: Evaluator, root, data: CandidateExtraction) -> None:
    """
    Root child: cfo_appointment_constraint (critical, parallel)
    Child group: appointed_2024_or_2025_with_reference (critical parallel with reference+support)
    """
    appoint_root = evaluator.add_parallel(
        id="cfo_appointment_constraint",
        desc="CFO appointment timing constraint is satisfied (with verifying reference)",
        parent=root,
        critical=True,
    )

    cfo = _safe_cfo_name(data)
    company = _safe_company_name(data)

    year_for_claim = data.appointment.appointment_year.strip() if _non_empty(data.appointment.appointment_year) else "2024 or 2025"
    appoint_urls = _norm_urls(data.appointment.appointment_urls)

    await _add_ref_and_support_group(
        evaluator,
        parent=appoint_root,
        group_id="appointed_2024_or_2025_with_reference",
        group_desc="CFO was appointed to the current CFO role in 2024 or 2025 AND references are provided",
        ref_urls=appoint_urls,
        support_leaf_desc="Appointment year (2024/2025) to the current CFO role is supported by cited sources",
        claim=f"{cfo} was appointed as Chief Financial Officer (CFO) of {company} in {year_for_claim}.",
        additional_instruction=(
            "Verify that the source clearly indicates the appointment to the CFO role (or promotion/transition into the "
            "CFO position) occurred in 2024 or 2025. Accept wording like 'appointed', 'named', 'joined as CFO', or "
            "'effective [date]' if the effective date falls in 2024/2025."
        ),
        critical=True,
    )


async def build_education_section(evaluator: Evaluator, root, data: CandidateExtraction) -> None:
    """
    Root child: education_constraints (critical, parallel)
    - bachelors_in_required_field_with_reference (critical group: refs+support)
    - mba_with_reference (critical group: refs+support)
    """
    edu_root = evaluator.add_parallel(
        id="education_constraints",
        desc="CFO education constraints are satisfied (with verifying references)",
        parent=root,
        critical=True,
    )

    cfo = _safe_cfo_name(data)

    # Bachelor's in required field
    bachelor_urls = _norm_urls(data.education.bachelors_urls)
    # Build a precise claim if we have a field, otherwise the generic requirement
    if _non_empty(data.education.bachelors_field):
        field_name = data.education.bachelors_field.strip()
        bachelor_claim = f"{cfo} holds a bachelor's degree in {field_name}."
    else:
        bachelor_claim = f"{cfo} holds a bachelor's degree in accounting, finance, or business administration."

    await _add_ref_and_support_group(
        evaluator,
        parent=edu_root,
        group_id="bachelors_in_required_field_with_reference",
        group_desc="CFO holds a bachelor's in accounting, finance, or business administration AND references are provided",
        ref_urls=bachelor_urls,
        support_leaf_desc="Bachelor's degree in a required field is supported by cited sources",
        claim=bachelor_claim,
        additional_instruction=(
            "Confirm that the bachelor's field is one of: accounting (or accountancy), finance (or financial management), "
            "or business administration/management (including BBA/BSBA). The source must clearly state the field or a clear synonym."
        ),
        critical=True,
    )

    # MBA
    mba_urls = _norm_urls(data.education.mba_urls)
    await _add_ref_and_support_group(
        evaluator,
        parent=edu_root,
        group_id="mba_with_reference",
        group_desc="CFO holds an MBA degree AND references are provided",
        ref_urls=mba_urls,
        support_leaf_desc="MBA degree is supported by cited sources",
        claim=f"{cfo} holds an MBA (Master of Business Administration) degree.",
        additional_instruction=(
            "Accept synonyms like 'Master of Business Administration' or program names that clearly denote an MBA."
        ),
        critical=True,
    )


async def build_experience_section(evaluator: Evaluator, root, data: CandidateExtraction) -> None:
    """
    Root child: experience_constraints (critical, parallel)
    - more_than_20_years_finance_experience_with_reference (critical group: refs+support)
    - prior_public_company_cfo_with_reference (critical group: refs+support)
    - big_four_partner_or_major_fininst_md_with_reference (critical group: refs+support)
    """
    exp_root = evaluator.add_parallel(
        id="experience_constraints",
        desc="CFO professional experience constraints are satisfied (with verifying references)",
        parent=root,
        critical=True,
    )

    cfo = _safe_cfo_name(data)
    company = _safe_company_name(data)

    # >20 years finance experience
    exp_urls = _norm_urls(data.experience.total_experience_urls)
    await _add_ref_and_support_group(
        evaluator,
        parent=exp_root,
        group_id="more_than_20_years_finance_experience_with_reference",
        group_desc="CFO has >20 years finance-related experience AND references are provided",
        ref_urls=exp_urls,
        support_leaf_desc=">20 years finance experience is supported by cited sources",
        claim=f"{cfo} has more than 20 years of total professional experience in finance-related roles.",
        additional_instruction=(
            "Look for explicit phrasing like 'more than 20 years', 'over two decades', '25 years', etc., clearly referring to "
            "finance-related roles (e.g., accounting, corporate finance, investment banking, financial leadership)."
        ),
        critical=True,
    )

    # Prior CFO of another publicly traded company
    prior_cfo_urls = _norm_urls(data.experience.prior_public_cfo_urls)
    await _add_ref_and_support_group(
        evaluator,
        parent=exp_root,
        group_id="prior_public_company_cfo_with_reference",
        group_desc="CFO previously served as CFO of another publicly traded company AND references are provided",
        ref_urls=prior_cfo_urls,
        support_leaf_desc="Prior publicly traded company CFO service is supported by cited sources",
        claim=f"Before the current appointment at {company}, {cfo} served as CFO of another publicly traded company.",
        additional_instruction=(
            "The source must indicate a prior role as Chief Financial Officer at another company and that the company is/was publicly traded "
            "(e.g., shows exchange/ticker like NYSE/Nasdaq, or explicitly states it is a public company)."
        ),
        critical=True,
    )

    # Big Four Partner OR MD (or equivalent) at major financial institution
    big4_md_urls = _norm_urls(data.experience.big4_or_md_urls)
    await _add_ref_and_support_group(
        evaluator,
        parent=exp_root,
        group_id="big_four_partner_or_major_fininst_md_with_reference",
        group_desc="CFO has qualifying Big Four partner OR major financial institution MD-level role AND references are provided",
        ref_urls=big4_md_urls,
        support_leaf_desc="Big Four partner or MD-level (or equivalent) major financial institution experience is supported by cited sources",
        claim=(
            f"{cfo} has either (A) been a partner at Deloitte, PricewaterhouseCoopers (PwC), Ernst & Young (EY), or KPMG, "
            "or (B) held a Managing Director-level (or equivalent senior finance leadership) role at a major financial institution."
        ),
        additional_instruction=(
            "Accept any of the Big Four partner titles. For the financial institution path, the role should be at Managing Director "
            "level or a clearly equivalent senior finance leadership rank at a major institution (e.g., Goldman Sachs, JPMorgan, Morgan Stanley, "
            "Bank of America, Citi)."
        ),
        critical=True,
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
    """
    Evaluate an answer against the CFO Fortune 500 healthcare task rubric using the Mind2Web2 framework.
    Returns a structured summary including the verification tree and final score.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates all critical children in parallel
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_candidate(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction",
    )

    # Build verification tree according to rubric
    await build_required_outputs_section(evaluator, root, extraction)
    await build_company_constraints_section(evaluator, root, extraction)
    await build_appointment_section(evaluator, root, extraction)
    await build_education_section(evaluator, root, extraction)
    await build_experience_section(evaluator, root, extraction)

    # Return final summary
    return evaluator.get_summary()