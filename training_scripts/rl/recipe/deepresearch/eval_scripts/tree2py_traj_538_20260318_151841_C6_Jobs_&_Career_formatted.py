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
TASK_ID = "tx_superintendents_qual_2026"
TASK_DESCRIPTION = """
Identify two current superintendents of Texas school districts who, as of March 18, 2026, each meet ALL of the following career and position requirements:

1. Currently serving as the superintendent of a Texas Independent School District
2. Their district enrolls at least 50,000 students
3. They previously held a position as Deputy Superintendent or Assistant Superintendent before becoming a superintendent
4. They previously served as a campus principal (at any level) for at least 3 years
5. They have been serving as superintendent of their current Texas school district for at least 7 consecutive years as of March 18, 2026

For each superintendent, provide their full name, current district name, and documented evidence of meeting each requirement with reference URLs.
"""

AS_OF_DATE = "March 18, 2026"
TENURE_MIN_START_DEADLINE = "March 18, 2019"
ENROLLMENT_THRESHOLD = 50000


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RoleEntry(BaseModel):
    title: Optional[str] = None
    org_or_district: Optional[str] = None
    campus_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    approx_years: Optional[str] = None


class SuperintendentItem(BaseModel):
    full_name: Optional[str] = None
    district_name: Optional[str] = None

    # URLs to verify identity/current title as superintendent (district bio page, news releases, board docs, etc.)
    current_title_urls: List[str] = Field(default_factory=list)

    # URLs that show the district is a Texas ISD (district/about page, TEA profile, etc.)
    isd_urls: List[str] = Field(default_factory=list)

    # Enrollment information and sources
    enrollment_count_text: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    # Principal experience details and sources
    principal_experiences: List[RoleEntry] = Field(default_factory=list)
    principal_urls: List[str] = Field(default_factory=list)

    # Deputy/Assistant Superintendent experience details and sources
    deputy_experiences: List[RoleEntry] = Field(default_factory=list)
    deputy_urls: List[str] = Field(default_factory=list)

    # Superintendent appointment date and tenure sources
    appointment_date_text: Optional[str] = None
    tenure_urls: List[str] = Field(default_factory=list)


class SuperintendentsExtraction(BaseModel):
    superintendents: List[SuperintendentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendents() -> str:
    return f"""
Extract up to TWO superintendent profiles exactly as presented in the answer. If the answer mentions more than two, extract only the first two. If it mentions fewer than two, still return an array with whatever is present (pad any missing fields with null or empty lists).

For each superintendent, extract these fields:
- full_name: The person's full name as given
- district_name: The current district name as given
- current_title_urls: All URLs cited that confirm the person is the current Superintendent of that district (e.g., district superintendent page, official announcements, board documents, credible news)
- isd_urls: URLs that indicate the district is a Texas Independent School District (ISD) or otherwise confirm the district is in Texas and is an ISD (e.g., district About page, TEA profile)
- enrollment_count_text: The specific student enrollment count if provided in the answer (as text, do not coerce to number). If a range or approximate phrase is given, include that phrase.
- enrollment_urls: All URLs that support the district's student enrollment (prefer official sources such as TEA, district fast facts, CAFR, or reputable reports)
- principal_experiences: A list of prior campus principal roles. For each role, extract:
  - title (e.g., Elementary Principal, High School Principal)
  - org_or_district (district name if available)
  - campus_name (school name)
  - start_date (as text if provided)
  - end_date (as text if provided)
  - approx_years (as text if provided; can be a number in text or phrasing like "3 years")
- principal_urls: All URLs that document principal experience (may overlap with bio/CV pages)
- deputy_experiences: A list of prior roles as Deputy Superintendent or Assistant Superintendent. For each role, extract:
  - title (e.g., Deputy Superintendent, Assistant Superintendent of Curriculum)
  - org_or_district (district name)
  - start_date (as text if provided)
  - end_date (as text if provided)
  - approx_years (as text if provided)
- deputy_urls: All URLs that document deputy/assistant superintendent experience
- appointment_date_text: The appointment date or start date for their current superintendent role as text (e.g., "July 1, 2018")
- tenure_urls: All URLs that document the appointment date and/or ongoing tenure as superintendent (board minutes, contract approvals/extensions, district bios indicating 'since YEAR', credible news)

Rules:
- Extract only what appears in the provided answer; do not invent information.
- Include only valid URLs that appear in the answer (full URLs; if missing protocol, prepend http://).
- If any field is missing from the answer, set it to null (for strings) or an empty list (for arrays).
- Do not perform calculations; just extract text for dates/years and roles as given.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_list(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and u.strip()]


def _merge_urls(*lists: List[str]) -> List[str]:
    merged = []
    seen = set()
    for lst in lists:
        for u in lst:
            if isinstance(u, str):
                v = u.strip()
                if v and v not in seen:
                    seen.add(v)
                    merged.append(v)
    return merged


def _roles_summary(roles: List[RoleEntry], max_items: int = 4) -> str:
    parts = []
    for r in roles[:max_items]:
        title = r.title or "Unknown title"
        campus = r.campus_name or r.org_or_district or "Unknown campus/district"
        start = r.start_date or "Unknown start"
        end = r.end_date or "Unknown end"
        yrs = f" (~{r.approx_years})" if r.approx_years else ""
        parts.append(f"{title} at {campus} ({start} – {end}){yrs}")
    return "; ".join(parts) if parts else "No specific roles listed"


def _label_for_index(idx: int) -> str:
    return "First" if idx == 0 else "Second"


# --------------------------------------------------------------------------- #
# Verification logic per superintendent                                       #
# --------------------------------------------------------------------------- #
async def verify_superintendent(
    evaluator: Evaluator,
    parent_node,
    sup: SuperintendentItem,
    idx: int,
) -> None:
    label = _label_for_index(idx)

    # Container for this superintendent
    sup_node = evaluator.add_parallel(
        id=f"{label}_Superintendent",
        desc=f"{label} qualifying superintendent meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # Career/position requirements (critical: must all pass for this person)
    career_node = evaluator.add_parallel(
        id=f"Career_Path_Requirements_{label}",
        desc=f"Verification of all career path and position requirements for the {label.lower()} superintendent",
        parent=sup_node,
        critical=True,
    )

    # ---------- Current Position Verification (critical) ----------
    current_pos_node = evaluator.add_parallel(
        id=f"Current_Position_Verification_{label}",
        desc=f"Verify current position as superintendent of a Texas ISD with required enrollment",
        parent=career_node,
        critical=True,
    )

    # Superintendent Identity
    identity_leaf = evaluator.add_leaf(
        id=f"Superintendent_Identity_{label}",
        desc="Provide full name and current district name of the superintendent",
        parent=current_pos_node,
        critical=True,
    )
    identity_claim = (
        f"The person named '{sup.full_name or ''}' is associated with the school district '{sup.district_name or ''}'. "
        "The provided source(s) should clearly connect this individual's name with that district."
    )
    identity_sources = _merge_urls(sup.current_title_urls, sup.tenure_urls, sup.isd_urls)
    await evaluator.verify(
        claim=identity_claim,
        node=identity_leaf,
        sources=identity_sources,
        additional_instruction="Confirm the individual's identity and the associated district. Allow minor name formatting variations.",
    )

    # Current Title Confirmation as of AS_OF_DATE
    current_title_leaf = evaluator.add_leaf(
        id=f"Current_Title_Confirmation_{label}",
        desc=f"Confirm the individual currently holds the title of Superintendent as of {AS_OF_DATE}",
        parent=current_pos_node,
        critical=True,
    )
    current_title_claim = (
        f"As of {AS_OF_DATE}, {sup.full_name or 'the individual'} is the current Superintendent of {sup.district_name or 'the district'}."
    )
    current_title_sources = _merge_urls(sup.current_title_urls, sup.tenure_urls)
    await evaluator.verify(
        claim=current_title_claim,
        node=current_title_leaf,
        sources=current_title_sources,
        additional_instruction=(
            f"Verify recency: pages should imply the person is the current superintendent on or after {AS_OF_DATE}. "
            f"Accept official district pages, TEA profiles, recent board documents, or credible news indicating ongoing service."
        ),
    )

    # Texas ISD Confirmation
    isd_leaf = evaluator.add_leaf(
        id=f"Texas_ISD_Confirmation_{label}",
        desc="Confirm the district is a Texas Independent School District",
        parent=current_pos_node,
        critical=True,
    )
    isd_claim = (
        f"'{sup.district_name or 'The district'}' is a Texas Independent School District (ISD) located in Texas."
    )
    isd_sources = _merge_urls(sup.isd_urls, sup.current_title_urls, sup.enrollment_urls)
    await evaluator.verify(
        claim=isd_claim,
        node=isd_leaf,
        sources=isd_sources,
        additional_instruction="Look for 'Independent School District' and Texas location on official or TEA pages.",
    )

    # District Enrollment Requirement (critical, sequential)
    enrollment_node = evaluator.add_sequential(
        id=f"District_Enrollment_Requirement_{label}",
        desc="Verify the district enrolls at least 50,000 students",
        parent=current_pos_node,
        critical=True,
    )

    # Reference URL existence first (to gate downstream checks)
    enrollment_url_exists = evaluator.add_custom_node(
        result=len(_nonempty_list(sup.enrollment_urls)) > 0,
        id=f"Enrollment_Reference_URL_{label}",
        desc="Provide reference URL verifying district enrollment",
        parent=enrollment_node,
        critical=True,
    )

    # Enrollment Count verification (at least 50,000)
    enrollment_leaf = evaluator.add_leaf(
        id=f"Enrollment_Count_{label}",
        desc="Provide the documented student enrollment count",
        parent=enrollment_node,
        critical=True,
    )
    if sup.enrollment_count_text:
        enrollment_claim = (
            f"The student enrollment for {sup.district_name or 'the district'} is reported as "
            f"'{sup.enrollment_count_text}', and it is at least {ENROLLMENT_THRESHOLD} students."
        )
    else:
        enrollment_claim = (
            f"The student enrollment for {sup.district_name or 'the district'} is at least {ENROLLMENT_THRESHOLD} students."
        )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_leaf,
        sources=sup.enrollment_urls,
        additional_instruction=(
            f"Use the source(s) to confirm an enrollment figure of {ENROLLMENT_THRESHOLD}+ students. "
            "Accept official district/TEA 'fast facts' or credible reports. "
            "If multiple years are shown, prefer the most recent credible number."
        ),
    )

    # ---------- Principal Experience Verification (critical, sequential) ----------
    principal_node = evaluator.add_sequential(
        id=f"Principal_Experience_Verification_{label}",
        desc="Verify the superintendent served as a campus principal for at least 3 years",
        parent=career_node,
        critical=True,
    )

    # Reference URL existence first
    principal_url_exists = evaluator.add_custom_node(
        result=len(_nonempty_list(sup.principal_urls)) > 0,
        id=f"Principal_Reference_URL_{label}",
        desc="Provide reference URL(s) documenting principal experience",
        parent=principal_node,
        critical=True,
    )

    principal_leaf = evaluator.add_leaf(
        id=f"Principal_Positions_Documentation_{label}",
        desc="Document specific principal position(s) held, campus name(s), and approximate years of service totaling at least 3 years",
        parent=principal_node,
        critical=True,
    )
    principal_roles_text = _roles_summary(sup.principal_experiences)
    principal_claim = (
        f"{sup.full_name or 'The individual'} previously served as a campus principal for at least 3 years. "
        f"Examples of documented roles: {principal_roles_text}"
    )
    await evaluator.verify(
        claim=principal_claim,
        node=principal_leaf,
        sources=sup.principal_urls,
        additional_instruction=(
            "Confirm that the combined documented principal roles amount to 3 or more years. "
            "Allow summing across multiple principal assignments where durations are indicated."
        ),
    )

    # ---------- Deputy/Assistant Superintendent Experience (critical, sequential) ----------
    deputy_node = evaluator.add_sequential(
        id=f"Deputy_Or_Assistant_Superintendent_Experience_{label}",
        desc="Verify the superintendent previously held a Deputy Superintendent or Assistant Superintendent position",
        parent=career_node,
        critical=True,
    )

    deputy_url_exists = evaluator.add_custom_node(
        result=len(_nonempty_list(sup.deputy_urls)) > 0,
        id=f"Deputy_Assistant_Reference_URL_{label}",
        desc="Provide reference URL(s) documenting Deputy/Assistant Superintendent experience",
        parent=deputy_node,
        critical=True,
    )

    deputy_leaf = evaluator.add_leaf(
        id=f"Deputy_Assistant_Position_Documentation_{label}",
        desc="Document the specific Deputy Superintendent or Assistant Superintendent position(s) held, district name(s), and time period",
        parent=deputy_node,
        critical=True,
    )
    deputy_roles_text = _roles_summary(sup.deputy_experiences)
    deputy_claim = (
        f"{sup.full_name or 'The individual'} previously served as Deputy Superintendent or Assistant Superintendent. "
        f"Examples of documented roles: {deputy_roles_text}"
    )
    await evaluator.verify(
        claim=deputy_claim,
        node=deputy_leaf,
        sources=sup.deputy_urls,
        additional_instruction=(
            "Explicitly confirm that at least one prior role includes 'Deputy Superintendent' or 'Assistant Superintendent' "
            "in the title or equivalent responsibilities at the district level."
        ),
    )

    # ---------- Current Superintendent Tenure Verification (critical, sequential) ----------
    tenure_node = evaluator.add_sequential(
        id=f"Current_Superintendent_Tenure_Verification_{label}",
        desc=f"Verify the superintendent has served in their current district superintendent role for at least 7 years as of {AS_OF_DATE}",
        parent=career_node,
        critical=True,
    )

    tenure_url_exists = evaluator.add_custom_node(
        result=len(_nonempty_list(sup.tenure_urls)) > 0,
        id=f"Tenure_Reference_URL_{label}",
        desc="Provide reference URL(s) documenting appointment date and tenure",
        parent=tenure_node,
        critical=True,
    )

    tenure_leaf = evaluator.add_leaf(
        id=f"Appointment_Date_And_Duration_{label}",
        desc=f"Document the superintendent's appointment date (must be on or before {TENURE_MIN_START_DEADLINE}) and verify continuous service through {AS_OF_DATE}",
        parent=tenure_node,
        critical=True,
    )
    tenure_claim = (
        f"{sup.full_name or 'The individual'} was appointed as Superintendent of {sup.district_name or 'the district'} "
        f"on or before {TENURE_MIN_START_DEADLINE} (e.g., '{sup.appointment_date_text or 'start date not specified in answer'}') "
        f"and has served continuously through {AS_OF_DATE}, totaling at least 7 consecutive years."
    )
    tenure_sources = _merge_urls(sup.tenure_urls, sup.current_title_urls)
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=tenure_sources,
        additional_instruction=(
            f"Confirm appointment/start date is on or before {TENURE_MIN_START_DEADLINE} and that service is continuous "
            f"through {AS_OF_DATE} (e.g., bio stating 'since 2018–present', contract renewals, or current superintendent pages)."
        ),
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two superintendents evaluated independently
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

    # Extract structured superintendent info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendents(),
        template_class=SuperintendentsExtraction,
        extraction_name="superintendents_extraction",
    )

    # Normalize to exactly two entries (pad with empty if needed)
    items = list(extracted.superintendents[:2])
    while len(items) < 2:
        items.append(SuperintendentItem())

    # Add context info
    evaluator.add_custom_info(
        info={"as_of_date": AS_OF_DATE, "tenure_min_start_deadline": TENURE_MIN_START_DEADLINE,
              "enrollment_threshold": ENROLLMENT_THRESHOLD},
        info_type="context",
        info_name="evaluation_context",
    )

    # Build verification subtrees for two superintendents
    await verify_superintendent(evaluator, root, items[0], 0)
    await verify_superintendent(evaluator, root, items[1], 1)

    # Return structured summary
    return evaluator.get_summary()