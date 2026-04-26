import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "oh_high_school_ad_2026_2_positions"
TASK_DESCRIPTION = """
Identify two distinct high school athletic director positions in different Ohio public school districts for the 2026-2027 school year that meet all of the following requirements: 
(1) The position must be for a high school athletic director overseeing grades 9-12 or equivalent; 
(2) The position must require or accept an Ohio administrative license (such as a professional principal license) OR a valid Ohio teaching license; 
(3) The position must require a current Ohio Pupil Activity Permit (PAP); 
(4) The position must require FBI and BCI background checks; 
(5) The position must require at least a bachelor's degree; 
(6) The position must require or prefer a minimum of 3 years of coaching or athletic administration experience; 
(7) The position must require compliance with Ohio High School Athletic Association (OHSAA) rules and regulations; 
(8) The advertised salary or minimum salary range must meet or exceed $70,000 annually; 
(9) The position must be scheduled to begin in the 2026-2027 school year (August 2026 or later); 
(10) The two positions must be from different school districts. 
For each position, provide the school district name, a brief description of the position including key qualifications, and reference URLs supporting each requirement.
"""


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class TextWithSources(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PositionItem(BaseModel):
    # Basic identifiers
    district_name: Optional[str] = None
    district_urls: List[str] = Field(default_factory=list)  # Primary district site and/or job posting URL(s)

    # Requirement-specific evidence (text + dedicated url list; reuse posting URL where applicable)
    position_level: TextWithSources = Field(default_factory=TextWithSources)          # HS AD (grades 9–12 or equivalent)
    start_date: TextWithSources = Field(default_factory=TextWithSources)              # 2026–2027 school year; Aug 2026+
    license_admin_or_teaching: TextWithSources = Field(default_factory=TextWithSources)  # OH admin license OR OH teaching license (require or accept)
    pupil_activity_permit: TextWithSources = Field(default_factory=TextWithSources)   # Current Ohio PAP required
    bachelors_degree: TextWithSources = Field(default_factory=TextWithSources)        # Bachelor's degree required
    experience_3yrs: TextWithSources = Field(default_factory=TextWithSources)         # ≥3 years coaching/athletic admin required or preferred
    background_checks: TextWithSources = Field(default_factory=TextWithSources)       # FBI and BCI checks required
    ohsaa: TextWithSources = Field(default_factory=TextWithSources)                   # OHSAA rules compliance required
    salary: TextWithSources = Field(default_factory=TextWithSources)                  # Salary ≥ $70,000 annually


class TwoPositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_positions() -> str:
    return """
Extract up to TWO distinct high school Athletic Director positions from the answer. Focus on Ohio public school districts only. 
If more than two are present, keep the first two. If fewer than two are present, include what is available.

For each position, extract:
- district_name: The exact Ohio public school district name, or the specific Ohio public high school name within a district (if the posting is school-specific).
- district_urls: Array of URLs for the district and/or the specific official job posting page(s). Prefer official district HR/posting links. Include full URLs.

For each of the following requirement categories, extract:
- text: The exact quoted or paraphrased snippet from the answer describing the requirement.
- urls: Array of URLs that support this specific requirement. 
  If one official posting URL supports multiple requirements, include it in multiple url arrays accordingly.

Requirement categories to extract (each as an object with {text, urls}):
- position_level: Confirms the role is a HIGH SCHOOL Athletic Director (grades 9–12 or equivalent).
- start_date: Confirms start is in the 2026–2027 school year (August 2026 or later) or phrasing like “2026-27 school year”.
- license_admin_or_teaching: Confirms the posting REQUIRES or ACCEPTS an Ohio administrative license (e.g., principal) OR a valid Ohio teaching license.
- pupil_activity_permit: Confirms a current Ohio Pupil Activity Permit (PAP) is required.
- bachelors_degree: Confirms at least a bachelor’s degree is required.
- experience_3yrs: Confirms a minimum of 3 years coaching OR athletic administration experience is required or preferred.
- background_checks: Confirms FBI and BCI background checks are required.
- ohsaa: Confirms compliance with OHSAA rules/regulations is required.
- salary: Confirms the advertised salary or MINIMUM of range is at least $70,000 per year.

SPECIAL INSTRUCTIONS:
- Only extract URLs explicitly present in the answer.
- If a single official posting URL supports multiple requirements, REUSE it in each requirement’s urls array.
- Ensure all URLs are full and valid; include protocol (https://).
- If some field is missing in the answer, set text to null and urls to [] for that field.

Return JSON with the following structure:

{
  "positions": [
    {
      "district_name": "...",
      "district_urls": ["...", "..."],
      "position_level": {"text": "...", "urls": ["..."]},
      "start_date": {"text": "...", "urls": ["..."]},
      "license_admin_or_teaching": {"text": "...", "urls": ["..."]},
      "pupil_activity_permit": {"text": "...", "urls": ["..."]},
      "bachelors_degree": {"text": "...", "urls": ["..."]},
      "experience_3yrs": {"text": "...", "urls": ["..."]},
      "background_checks": {"text": "...", "urls": ["..."]},
      "ohsaa": {"text": "...", "urls": ["..."]},
      "salary": {"text": "...", "urls": ["..."]}
    }
  ]
}
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def merge_sources(*maybe_lists: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for lst in maybe_lists:
        if lst:
            urls.extend([u.strip() for u in lst if isinstance(u, str) and u.strip()])
    return _uniq_preserve(urls)


def safe_text(t: Optional[str]) -> str:
    return t or ""


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def _verify_basic_info(
    evaluator: Evaluator,
    parent: VerificationNode,
    pos: PositionItem,
    pos_idx: int,
) -> None:
    """
    Build and verify the 'basic info' subtree:
      - position_{i}_school_district_url (existence, critical)
      - position_{i}_school_district (URL-grounded verification, critical)
      - position_{i}_position_level_url (existence, critical)
      - position_{i}_position_level (URL-grounded verification, critical)
      - position_{i}_start_date_url (existence, critical)
      - position_{i}_start_date (URL-grounded verification, critical)
    """
    pi = pos_idx + 1
    node = evaluator.add_parallel(
        id=f"position_{pi}_basic_info",
        desc=f"Basic position information for position #{pi}",
        parent=parent,
        critical=True,
    )

    # 1) School district URL presence (critical gate for district verification)
    sd_sources = merge_sources(pos.district_urls)
    evaluator.add_custom_node(
        result=len(sd_sources) > 0,
        id=f"position_{pi}_school_district_url",
        desc=f"Provides URL reference for the school district or job posting (position #{pi})",
        parent=node,
        critical=True,
    )

    # 2) School district identification (Ohio public school district or HS within district)
    sd_leaf = evaluator.add_leaf(
        id=f"position_{pi}_school_district",
        desc=f"Identifies the specific Ohio school district or school name (position #{pi})",
        parent=node,
        critical=True,
    )
    district_name = safe_text(pos.district_name)
    await evaluator.verify(
        claim=(
            f"The provided page(s) are an official district or job posting page for '{district_name}' "
            f"in Ohio (a public school district or a public high school within that district)."
        ),
        node=sd_leaf,
        sources=sd_sources,
        additional_instruction=(
            "Verify that the page clearly belongs to the stated Ohio district or a public high school in that district. "
            "Allow reasonable variants or abbreviations of the district name. "
            "Look for address/city/state, logos, HR/recruitment branding, or other signals indicating the Ohio district."
        ),
    )

    # 3) Position level URL presence
    level_sources = merge_sources(pos.position_level.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(level_sources) > 0,
        id=f"position_{pi}_position_level_url",
        desc=f"Provides URL reference confirming the position level (position #{pi})",
        parent=node,
        critical=True,
    )

    # 4) Position level HS AD (grades 9–12)
    level_leaf = evaluator.add_leaf(
        id=f"position_{pi}_position_level",
        desc=f"Confirms the position is for a high school athletic director (grades 9–12 or equivalent) (position #{pi})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This job posting is for a High School Athletic Director (i.e., it oversees the high school level "
            "— typically grades 9–12 — or explicitly references the high school)."
        ),
        node=level_leaf,
        sources=level_sources,
        additional_instruction=(
            "Accept synonyms such as 'Director of Athletics' or 'Athletics Director' if the posting clearly pertains to the HIGH SCHOOL level. "
            "Accept mentions of 'HS', 'grades 9–12', 'secondary school', 'upper school', or the specific high school name. "
            "If only district-wide K–12 is mentioned without explicit HS AD responsibility, do not mark correct."
        ),
    )

    # 5) Start date URL presence
    start_sources = merge_sources(pos.start_date.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(start_sources) > 0,
        id=f"position_{pi}_start_date_url",
        desc=f"Provides URL reference for the start date (position #{pi})",
        parent=node,
        critical=True,
    )

    # 6) Start date is 2026–2027 SY (Aug 2026 or later)
    start_leaf = evaluator.add_leaf(
        id=f"position_{pi}_start_date",
        desc=f"Verifies the position starts in the 2026-2027 school year (August 2026 or later) (position #{pi})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The posting indicates the position begins in the 2026–2027 school year "
            "(e.g., '2026-27 SY') or specifies a start date in August 2026 or later."
        ),
        node=start_leaf,
        sources=start_sources,
        additional_instruction=(
            "Accept phrases like '2026–27 school year', 'beginning August 2026', 'start date 8/2026', or later. "
            "References to start prior to August 2026 should be considered incorrect."
        ),
    )


async def _verify_qualifications(
    evaluator: Evaluator,
    parent: VerificationNode,
    pos: PositionItem,
    pos_idx: int,
) -> None:
    """
    Build and verify the 'qualifications' subtree, consisting of:
      - licensure (admin OR teaching license) + URLs
      - PAP + URLs
      - bachelor's degree + URLs
      - 3+ years experience + URLs
      - FBI & BCI background checks + URLs
      - OHSAA compliance + URLs
    All nodes here are critical.
    """
    pi = pos_idx + 1
    q_node = evaluator.add_parallel(
        id=f"position_{pi}_qualifications",
        desc=f"Qualification requirements for position #{pi}",
        parent=parent,
        critical=True,
    )

    # 1) Licensure cluster
    lic_node = evaluator.add_parallel(
        id=f"position_{pi}_licensure",
        desc=f"Licensure requirements verification (position #{pi})",
        parent=q_node,
        critical=True,
    )

    lic_sources = merge_sources(pos.license_admin_or_teaching.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(lic_sources) > 0,
        id=f"position_{pi}_admin_or_teaching_license_url",
        desc=f"Provides URL reference for licensure requirement (position #{pi})",
        parent=lic_node,
        critical=True,
    )

    lic_leaf = evaluator.add_leaf(
        id=f"position_{pi}_admin_or_teaching_license",
        desc=f"Confirms the position requires or accepts Ohio administrative license OR valid Ohio teaching license (position #{pi})",
        parent=lic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The posting requires or accepts an Ohio administrative license (e.g., principal/assistant principal/superintendent) "
            "OR a valid Ohio teaching license."
        ),
        node=lic_leaf,
        sources=lic_sources,
        additional_instruction=(
            "It is sufficient if either is required or explicitly accepted. "
            "Look for 'Ohio administrative license', 'principal license', 'Ohio teaching license', or similar language."
        ),
    )

    pap_sources = merge_sources(pos.pupil_activity_permit.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(pap_sources) > 0,
        id=f"position_{pi}_pupil_activity_permit_url",
        desc=f"Provides URL reference for PAP requirement (position #{pi})",
        parent=lic_node,
        critical=True,
    )

    pap_leaf = evaluator.add_leaf(
        id=f"position_{pi}_pupil_activity_permit",
        desc=f"Confirms the position requires a current Ohio Pupil Activity Permit (PAP) (position #{pi})",
        parent=lic_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting requires a current Ohio Pupil Activity Permit (PAP).",
        node=pap_leaf,
        sources=pap_sources,
        additional_instruction="Look for 'Pupil Activity Permit' or 'PAP' explicitly required.",
    )

    # 2) Education & Experience cluster
    edu_node = evaluator.add_parallel(
        id=f"position_{pi}_education_experience",
        desc=f"Education and experience requirements (position #{pi})",
        parent=q_node,
        critical=True,
    )

    deg_sources = merge_sources(pos.bachelors_degree.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(deg_sources) > 0,
        id=f"position_{pi}_bachelors_degree_url",
        desc=f"Provides URL reference for education requirement (position #{pi})",
        parent=edu_node,
        critical=True,
    )

    deg_leaf = evaluator.add_leaf(
        id=f"position_{pi}_bachelors_degree",
        desc=f"Confirms the position requires at least a bachelor's degree (position #{pi})",
        parent=edu_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting requires at least a Bachelor's degree (Master's may be preferred).",
        node=deg_leaf,
        sources=deg_sources,
        additional_instruction="Reject if it only prefers Bachelor's but does not require any degree.",
    )

    exp_sources = merge_sources(pos.experience_3yrs.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(exp_sources) > 0,
        id=f"position_{pi}_coaching_experience_url",
        desc=f"Provides URL reference for experience requirement (position #{pi})",
        parent=edu_node,
        critical=True,
    )

    exp_leaf = evaluator.add_leaf(
        id=f"position_{pi}_coaching_experience",
        desc=f"Confirms minimum 3 years coaching or athletic administration experience is required or preferred (position #{pi})",
        parent=edu_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The posting requires or prefers at least 3 years of coaching or athletic administration experience "
            "(e.g., 'minimum three years', '3–5 years')."
        ),
        node=exp_leaf,
        sources=exp_sources,
        additional_instruction="Accept both 'required' and 'preferred' language as meeting the minimum experience condition.",
    )

    # 3) Compliance cluster
    comp_node = evaluator.add_parallel(
        id=f"position_{pi}_compliance",
        desc=f"Compliance and background check requirements (position #{pi})",
        parent=q_node,
        critical=True,
    )

    bg_sources = merge_sources(pos.background_checks.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(bg_sources) > 0,
        id=f"position_{pi}_background_checks_url",
        desc=f"Provides URL reference for background check requirement (position #{pi})",
        parent=comp_node,
        critical=True,
    )

    bg_leaf = evaluator.add_leaf(
        id=f"position_{pi}_background_checks",
        desc=f"Confirms the position requires FBI and BCI background checks (position #{pi})",
        parent=comp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting requires both FBI and BCI background checks.",
        node=bg_leaf,
        sources=bg_sources,
        additional_instruction="Look for 'FBI and BCI' or 'BCI/FBI' phrasing specifically; both must be required.",
    )

    ohsaa_sources = merge_sources(pos.ohsaa.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(ohsaa_sources) > 0,
        id=f"position_{pi}_ohsaa_compliance_url",
        desc=f"Provides URL reference for OHSAA compliance requirement (position #{pi})",
        parent=comp_node,
        critical=True,
    )

    ohsaa_leaf = evaluator.add_leaf(
        id=f"position_{pi}_ohsaa_compliance",
        desc=f"Confirms the position requires compliance with OHSAA rules/regulations (position #{pi})",
        parent=comp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting requires compliance with Ohio High School Athletic Association (OHSAA) rules and regulations.",
        node=ohsaa_leaf,
        sources=ohsaa_sources,
        additional_instruction="Accept language like 'knowledge of and adherence to OHSAA rules' or 'OHSAA compliance required'.",
    )


async def _verify_salary_timeline(
    evaluator: Evaluator,
    parent: VerificationNode,
    pos: PositionItem,
    pos_idx: int,
) -> None:
    """
    Build and verify the 'salary & timeline' subtree (salary threshold only here, since start date handled in basic info):
      - position_{i}_salary_threshold_url (existence)
      - position_{i}_salary_threshold (URL-grounded verification ≥ $70,000)
    """
    pi = pos_idx + 1
    node = evaluator.add_parallel(
        id=f"position_{pi}_salary_timeline",
        desc=f"Salary and application timeline for position #{pi}",
        parent=parent,
        critical=True,
    )

    sal_sources = merge_sources(pos.salary.urls, pos.district_urls)
    evaluator.add_custom_node(
        result=len(sal_sources) > 0,
        id=f"position_{pi}_salary_threshold_url",
        desc=f"Provides URL reference for salary information (position #{pi})",
        parent=node,
        critical=True,
    )

    sal_leaf = evaluator.add_leaf(
        id=f"position_{pi}_salary_threshold",
        desc=f"Verifies advertised salary/minimum range meets or exceeds $70,000 annually (position #{pi})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The advertised salary (or the minimum of the salary range) is at least $70,000 per year.",
        node=sal_leaf,
        sources=sal_sources,
        additional_instruction=(
            "Consider numeric ranges (e.g., '$70,000–$85,000') or 'starting at $70,000'. "
            "If hourly/daily rates are shown, only accept if the posting also clearly states an annualized value ≥ $70,000."
        ),
    )


async def verify_single_position(
    evaluator: Evaluator,
    parent: VerificationNode,
    pos: PositionItem,
    pos_idx: int,
) -> None:
    """
    Build the full subtree for one position under 'position_{i}' parallel node.
    """
    pi = pos_idx + 1
    pnode = evaluator.add_parallel(
        id=f"position_{pi}",
        desc=f"Position #{pi} verification (must meet all requirements)",
        parent=parent,
        critical=False,  # allow partial scoring across positions
    )

    await _verify_basic_info(evaluator, pnode, pos, pos_idx)
    await _verify_qualifications(evaluator, pnode, pos, pos_idx)
    await _verify_salary_timeline(evaluator, pnode, pos, pos_idx)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Ohio High School Athletic Director (2026-27) two-position task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # positions verified independently
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

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=TwoPositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Normalize to exactly two positions (pad with empty placeholders if needed)
    positions: List[PositionItem] = list(extracted.positions[:2])
    while len(positions) < 2:
        positions.append(PositionItem())

    pos1, pos2 = positions[0], positions[1]

    # Build the rubric tree following the JSON hierarchy (root parallel)
    # 1) Position 1 subtree
    await verify_single_position(evaluator, root, pos1, 0)

    # 2) Position 2 subtree
    await verify_single_position(evaluator, root, pos2, 1)

    # 3) Cross-position constraint: Different school districts (critical gate at root)
    #    - First ensure both district names are present (existence custom node)
    names_present = bool(safe_text(pos1.district_name).strip()) and bool(safe_text(pos2.district_name).strip())
    names_exist_node = evaluator.add_custom_node(
        result=names_present,
        id="district_names_provided",
        desc="Both positions include district names",
        parent=root,
        critical=True,
    )

    #    - Then verify difference as a separate simple (non-URL) leaf node
    diff_leaf = evaluator.add_leaf(
        id="distinct_districts",
        desc="The two positions are from different Ohio school districts",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The district for position 1 ('{safe_text(pos1.district_name)}') "
            f"and the district for position 2 ('{safe_text(pos2.district_name)}') are different organizations."
        ),
        node=diff_leaf,
        sources=None,  # logical/textual check only
        additional_instruction=(
            "Determine if the two names refer to the same district. "
            "Use case-insensitive comparison and ignore trivial punctuation/abbreviation differences. "
            "If they are the same or refer to the same district, mark incorrect."
        ),
        extra_prerequisites=[names_exist_node],
    )

    # Record custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "position_1_district": pos1.district_name,
            "position_2_district": pos2.district_name,
            "position_1_primary_urls": pos1.district_urls,
            "position_2_primary_urls": pos2.district_urls,
        },
        info_type="extraction_summary",
        info_name="extraction_overview",
    )

    return evaluator.get_summary()