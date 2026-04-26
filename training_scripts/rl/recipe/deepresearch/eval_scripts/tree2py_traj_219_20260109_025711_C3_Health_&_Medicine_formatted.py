import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "btd_rare_disease_ceo_edu_bio_dept"
TASK_DESCRIPTION = (
    "A biotechnology company received FDA breakthrough therapy designation for a treatment targeting a rare disease "
    "(affecting fewer than 200,000 people in the United States) between January 1, 2023, and January 31, 2025. "
    "The company's CEO or co-founder holds a PhD degree and completed their undergraduate degree in biology, "
    "biological sciences, or a closely related life sciences field at one university before December 31, 2005, and then "
    "obtained their PhD degree from a different university. Identify this therapy and provide the following information: "
    "(1) The name of the therapy that received the breakthrough therapy designation, (2) The name of the biotechnology company, "
    "(3) The name of the CEO or co-founder who fits the educational background described, (4) The name of the university where "
    "this person completed their undergraduate degree, (5) The specific year when the biology department or school of biological "
    "sciences was formally established as a separate organizational unit at that undergraduate university. "
    "Provide reference URLs to verify each piece of information."
)

BTD_START_DATE = "2023-01-01"
BTD_END_DATE = "2025-01-31"

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TherapyInfo(BaseModel):
    therapy_name: Optional[str] = None
    disease_name: Optional[str] = None
    btd_urls: List[str] = Field(default_factory=list)                 # URLs supporting BTD and timing
    disease_urls: List[str] = Field(default_factory=list)             # URLs supporting <200k US prevalence


class CompanyInfo(BaseModel):
    company_name: Optional[str] = None
    company_therapy_urls: List[str] = Field(default_factory=list)     # URLs linking company to therapy/BTD


class LeaderEduInfo(BaseModel):
    leader_name: Optional[str] = None
    leader_role: Optional[str] = None                                 # e.g., CEO or co-founder
    leader_profile_urls: List[str] = Field(default_factory=list)      # Company page, press, LinkedIn, bio
    phd_university: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)                 # URLs that support PhD and institution
    undergrad_university: Optional[str] = None
    undergrad_field: Optional[str] = None
    undergrad_completion_year: Optional[str] = None                   # e.g., "2003"
    undergrad_urls: List[str] = Field(default_factory=list)           # URLs that support UG school, field, and timing


class DeptInfo(BaseModel):
    dept_established_year: Optional[str] = None                       # Four-digit year of establishment as separate unit
    dept_urls: List[str] = Field(default_factory=list)                # URLs from the UG university or credible sources


class FullExtraction(BaseModel):
    therapy: Optional[TherapyInfo] = None
    company: Optional[CompanyInfo] = None
    leader: Optional[LeaderEduInfo] = None
    dept: Optional[DeptInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract a single, self-consistent set of information that satisfies the task constraints if present in the answer. 
    Return JSON with the following structure and rules:

    {
      "therapy": {
        "therapy_name": string|null,
        "disease_name": string|null,
        "btd_urls": [url, ...],             // URLs explicitly cited in the answer supporting FDA Breakthrough Therapy Designation and its date
        "disease_urls": [url, ...]          // URLs explicitly cited that support "fewer than 200,000 in the US" for the targeted disease
      },
      "company": {
        "company_name": string|null,
        "company_therapy_urls": [url, ...]  // URLs linking the company to the therapy and BTD (press releases, FDA pages, credible media)
      },
      "leader": {
        "leader_name": string|null,         // CEO or co-founder name
        "leader_role": string|null,         // e.g., "CEO" or "co-founder"
        "leader_profile_urls": [url, ...],  // URLs verifying leader identity/role and possibly education
        "phd_university": string|null,
        "phd_urls": [url, ...],             // URLs verifying PhD and PhD university
        "undergrad_university": string|null,
        "undergrad_field": string|null,     // biology, biological sciences, biochemistry, molecular biology, etc.
        "undergrad_completion_year": string|null, // 4-digit year if given; else null
        "undergrad_urls": [url, ...]        // URLs verifying UG institution, field, and completion timing
      },
      "dept": {
        "dept_established_year": string|null, // four-digit year of when biology department/school was formally established as a separate unit
        "dept_urls": [url, ...]               // URLs from the UG university (or authoritative source) verifying the founding as separate unit and the year
      }
    }

    Rules:
    - Extract only what is explicitly stated in the provided answer, exactly as written.
    - For URL fields, extract only valid URLs mentioned in the answer. Include all relevant ones.
    - If a field is not present in the answer, set it to null (for strings) or [] (for URL lists).
    - Prefer a single coherent example if the answer mentions multiple candidates.
    - For fields like "undergrad_completion_year" and "dept_established_year", prefer 4-digit years if available.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # Skip malformed URLs; we don't add protocols automatically to avoid fabricating sources
            continue
        cleaned.append(u)
    # deduplicate while preserving order
    seen = set()
    deduped = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _combine_sources(*url_lists: Optional[List[str]]) -> Optional[List[str]]:
    combined: List[str] = []
    for lst in url_lists:
        combined.extend(_filter_urls(lst))
    combined = list(dict.fromkeys(combined))  # dedupe
    return combined if combined else None


def _safe(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def _parse_year(year_text: Optional[str]) -> Optional[int]:
    if not year_text:
        return None
    # Try to find a 4-digit year in the text
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", year_text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    try:
        if len(year_text) == 4:
            return int(year_text)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, data: FullExtraction, logger: logging.Logger) -> None:
    therapy = data.therapy or TherapyInfo()
    company = data.company or CompanyInfo()
    leader = data.leader or LeaderEduInfo()
    dept = data.dept or DeptInfo()

    therapy_name = _safe(therapy.therapy_name)
    disease_name = _safe(therapy.disease_name)
    company_name = _safe(company.company_name)
    leader_name = _safe(leader.leader_name)
    leader_role = _safe(leader.leader_role)
    ug_univ = _safe(leader.undergrad_university)
    ug_field = _safe(leader.undergrad_field)
    ug_year_text = _safe(leader.undergrad_completion_year)
    phd_univ = _safe(leader.phd_university)
    dept_year_text = _safe(dept.dept_established_year)

    btd_urls = _filter_urls(therapy.btd_urls)
    disease_urls = _filter_urls(therapy.disease_urls)
    company_therapy_urls = _filter_urls(company.company_therapy_urls)
    leader_profile_urls = _filter_urls(leader.leader_profile_urls)
    phd_urls = _filter_urls(leader.phd_urls)
    ug_urls = _filter_urls(leader.undergrad_urls)
    dept_urls = _filter_urls(dept.dept_urls)

    # Root-level groups (all critical as per rubric)
    therapy_node = evaluator.add_parallel(
        id="therapy_requirements",
        desc="Therapy identification and eligibility constraints (BTD + time window + rare disease threshold).",
        parent=root_node,
        critical=True
    )

    company_node = evaluator.add_parallel(
        id="company_requirements",
        desc="Company identification and linkage to the therapy.",
        parent=root_node,
        critical=True
    )

    leader_node = evaluator.add_parallel(
        id="leader_identity_and_degree",
        desc="CEO/co-founder identification and PhD requirement.",
        parent=root_node,
        critical=True
    )

    edu_node = evaluator.add_parallel(
        id="education_constraints",
        desc="Undergraduate/PhD educational-path constraints for the identified CEO/co-founder.",
        parent=root_node,
        critical=True
    )

    dept_node = evaluator.add_parallel(
        id="biology_department_founding_year",
        desc="Undergraduate university biology department/school establishment constraint and the required founding-year output.",
        parent=root_node,
        critical=True
    )

    # 1) Therapy requirements
    evaluator.add_custom_node(
        result=bool(therapy_name),
        id="therapy_name_provided",
        desc="Provides the name of the therapy that received FDA BTD.",
        parent=therapy_node,
        critical=True
    )

    btd_within_window_leaf = evaluator.add_leaf(
        id="btd_within_window",
        desc="Therapy is documented as having received FDA Breakthrough Therapy Designation between 2023-01-01 and 2025-01-31 (inclusive).",
        parent=therapy_node,
        critical=True
    )
    btd_claim = (
        f"The therapy '{therapy_name or 'the therapy'}' received FDA Breakthrough Therapy Designation "
        f"between {BTD_START_DATE} and {BTD_END_DATE} (inclusive)."
    )
    await evaluator.verify(
        claim=btd_claim,
        node=btd_within_window_leaf,
        sources=_combine_sources(btd_urls),
        additional_instruction=(
            "Verify that the cited page(s) explicitly state Breakthrough Therapy Designation and the date of the designation. "
            "Accept if the date shown falls within 2023-01-01 to 2025-01-31. If multiple dates are present, focus on the date when BTD was granted."
        )
    )

    rare_leaf = evaluator.add_leaf(
        id="rare_disease_under_200k",
        desc="Therapy targets a rare disease affecting fewer than 200,000 people in the United States (as stated/verified by an appropriate public source).",
        parent=therapy_node,
        critical=True
    )
    rare_claim = (
        f"The disease targeted by this therapy ({disease_name or 'the target disease'}) affects fewer than 200,000 people in the United States."
    )
    await evaluator.verify(
        claim=rare_claim,
        node=rare_leaf,
        sources=_combine_sources(disease_urls),
        additional_instruction=(
            "Check that the page explicitly states U.S. prevalence fewer than 200,000 (or an equivalent statement). "
            "If the page defines 'rare disease' as <200,000 and explicitly states the disease meets that definition in the U.S., that is acceptable."
        )
    )

    # 2) Company requirements
    evaluator.add_custom_node(
        result=bool(company_name),
        id="company_name_provided",
        desc="Provides the name of the biotechnology company developing the therapy.",
        parent=company_node,
        critical=True
    )

    company_develops_leaf = evaluator.add_leaf(
        id="company_develops_therapy",
        desc="Company is verifiably the developer/sponsor of the identified therapy that received BTD.",
        parent=company_node,
        critical=True
    )
    company_develops_claim = (
        f"{company_name or 'The company'} is the developer or sponsor of the therapy '{therapy_name or 'the therapy'}' that received FDA Breakthrough Therapy Designation."
    )
    await evaluator.verify(
        claim=company_develops_claim,
        node=company_develops_leaf,
        sources=_combine_sources(company_therapy_urls, btd_urls),
        additional_instruction=(
            "Confirm that the cited source(s) tie the named company to the named therapy, "
            "and that this therapy is the one with Breakthrough Therapy Designation."
        )
    )

    # 3) Leader identity and degree
    leader_identity_leaf = evaluator.add_leaf(
        id="leader_name_and_role",
        desc="Provides the name of the CEO or co-founder and verifies they are CEO or co-founder of the identified company.",
        parent=leader_node,
        critical=True
    )
    leader_identity_claim = (
        f"{leader_name or 'The identified person'} is the CEO or a co-founder of {company_name or 'the identified company'}."
    )
    await evaluator.verify(
        claim=leader_identity_claim,
        node=leader_identity_leaf,
        sources=_combine_sources(leader_profile_urls, company_therapy_urls, btd_urls),
        additional_instruction=(
            "Verify that the person is either CEO or co-founder of the specified company from credible pages (company team page, press release, professional bio, authoritative media)."
        )
    )

    leader_phd_leaf = evaluator.add_leaf(
        id="leader_has_phd",
        desc="Verifies the identified CEO/co-founder holds a PhD degree.",
        parent=leader_node,
        critical=True
    )
    leader_phd_claim = f"{leader_name or 'The identified person'} holds a PhD (doctoral) degree."
    await evaluator.verify(
        claim=leader_phd_claim,
        node=leader_phd_leaf,
        sources=_combine_sources(leader_profile_urls, phd_urls),
        additional_instruction=(
            "Look for explicit mention of a PhD (or equivalent doctoral degree such as DPhil or Dr.). "
            "Titles like 'Ph.D.' or 'Doctorate' count; ensure it refers to an earned doctoral degree."
        )
    )

    # 4) Education constraints
    ug_provided_node = evaluator.add_custom_node(
        result=bool(ug_univ),
        id="undergrad_university_provided",
        desc="Provides the name of the university where the CEO/co-founder completed their undergraduate degree.",
        parent=edu_node,
        critical=True
    )

    ug_field_leaf = evaluator.add_leaf(
        id="undergrad_field_life_sciences",
        desc="Verifies the CEO/co-founder’s undergraduate degree field is biology, biological sciences, or a closely related life sciences field.",
        parent=edu_node,
        critical=True
    )
    ug_field_claim = (
        f"{leader_name or 'The identified person'} completed an undergraduate degree in a life sciences field "
        f"(e.g., biology/biological sciences or closely related), specifically '{ug_field or 'a life sciences field'}'."
    )
    await evaluator.verify(
        claim=ug_field_claim,
        node=ug_field_leaf,
        sources=_combine_sources(leader_profile_urls, ug_urls),
        additional_instruction=(
            "Accept life-science fields such as Biology, Biological Sciences, Biochemistry, Molecular Biology, Microbiology, Genetics, Physiology, "
            "Cell Biology, Neuroscience (if clearly life-science), or similar. The page must clearly indicate the undergraduate major/field."
        )
    )

    ug_before_2006_leaf = evaluator.add_leaf(
        id="undergrad_completed_before_2005_12_31",
        desc="Verifies the CEO/co-founder completed the undergraduate degree before 2005-12-31.",
        parent=edu_node,
        critical=True
    )
    ug_before_2006_claim = (
        f"{leader_name or 'The identified person'} completed their undergraduate degree before 2005-12-31."
    )
    await evaluator.verify(
        claim=ug_before_2006_claim,
        node=ug_before_2006_leaf,
        sources=_combine_sources(leader_profile_urls, ug_urls),
        additional_instruction=(
            "Look for completion or graduation year explicitly earlier than 2006 (e.g., 2005 or earlier). "
            "If month/day is shown, ensure completion date is on or before 2005-12-31."
        )
    )

    phd_diff_leaf = evaluator.add_leaf(
        id="phd_university_different",
        desc="Verifies the CEO/co-founder’s PhD was obtained from a different university than their undergraduate institution.",
        parent=edu_node,
        critical=True
    )
    phd_diff_claim = (
        f"The undergraduate university ({ug_univ or 'UG university unknown'}) and the PhD university ({phd_univ or 'PhD university unknown'}) are different universities."
    )
    await evaluator.verify(
        claim=phd_diff_claim,
        node=phd_diff_leaf,
        additional_instruction=(
            "Judge whether the two university names refer to different institutions. "
            "Treat clearly different named campuses (e.g., UC Berkeley vs UC San Diego) as different universities. "
            "Ignore minor variations like 'University of' vs 'Univ.' in names."
        )
    )

    # 5) Biology department founding year at UG university
    dept_established_leaf = evaluator.add_leaf(
        id="dept_formally_established_separate_unit",
        desc="Verifies the undergraduate university has a biology department or school of biological sciences that was formally established as a separate organizational unit.",
        parent=dept_node,
        critical=True
    )
    dept_established_claim = (
        f"The biology department or a school of biological sciences at {ug_univ or 'the undergraduate university'} "
        f"was formally established as a separate organizational unit."
    )
    await evaluator.verify(
        claim=dept_established_claim,
        node=dept_established_leaf,
        sources=_combine_sources(dept_urls),
        additional_instruction=(
            "Look for history pages that say 'Department of Biology established/founded in YEAR' or "
            "'School of Biological Sciences established in YEAR' or wording that indicates it became its own separate department/school."
        )
    )

    # Existence + sanity check for the specific establishment year
    dept_year = _parse_year(dept_year_text)
    current_year_limit = 2026  # safe upper bound based on current_date in meta
    dept_year_ok = dept_year is not None and (1800 <= dept_year <= current_year_limit)
    evaluator.add_custom_node(
        result=dept_year_ok,
        id="dept_establishment_year_provided",
        desc="Provides the specific year the biology department/school of biological sciences was formally established as a separate organizational unit at the undergraduate university.",
        parent=dept_node,
        critical=True
    )

    # 6) Citations presence check (as a single binary check)
    # Requirements: References exist to verify each required piece of information.
    has_btd_src = bool(btd_urls)
    has_rare_src = bool(disease_urls)
    has_company_src = bool(company_therapy_urls or btd_urls)
    has_leader_role_src = bool(leader_profile_urls)
    has_phd_src = bool(phd_urls or leader_profile_urls)
    has_ug_src = bool(ug_urls or leader_profile_urls)
    has_dept_src = bool(dept_urls)

    # For PhD-vs-UG difference, ensure we have sources covering both sides (UG and PhD), allowing leader profile to serve either if it contains both.
    has_phd_vs_ug_src = (bool(phd_urls or leader_profile_urls) and bool(ug_urls or leader_profile_urls))

    citations_ok = all([
        has_btd_src,
        has_rare_src,
        has_company_src,
        has_leader_role_src,
        has_phd_src,
        has_ug_src,
        has_phd_vs_ug_src,
        has_dept_src
    ])

    citations_node = evaluator.add_custom_node(
        result=citations_ok,
        id="citations",
        desc="Provides reference URLs sufficient to verify each required piece of information (therapy BTD/date window, rare disease threshold claim, company identity/link to therapy, CEO/co-founder role, PhD degree, undergraduate institution/field/completion timing, PhD institution difference, and biology department establishment year).",
        parent=root_node,
        critical=True
    )

    # Log detailed citation diagnostics
    evaluator.add_custom_info(
        info={
            "btd_urls_count": len(btd_urls),
            "disease_urls_count": len(disease_urls),
            "company_therapy_urls_count": len(company_therapy_urls),
            "leader_profile_urls_count": len(leader_profile_urls),
            "phd_urls_count": len(phd_urls),
            "undergrad_urls_count": len(ug_urls),
            "dept_urls_count": len(dept_urls),
            "citations_passed": citations_ok
        },
        info_type="citation_coverage"
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,  # Root node desc will use this
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Make the root critical to follow rubric; all children must be critical as well
    # Note: Evaluator.initialize creates a non-critical root by default in framework,
    # but we still treat all sub-criteria as critical and aggregate gating will enforce correctness.
    # We also add a small custom node to reflect the rubric's root-level criticality as an existence placeholder.
    # However, the verification framework computes parent aggregation automatically; no explicit root custom node needed.

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="extracted_info"
    )

    # Build tree and run verifications
    await build_verification_tree(evaluator, root, extracted, logger)

    # Return summary
    return evaluator.get_summary()