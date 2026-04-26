import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_ad_2022_2024_one"
TASK_DESCRIPTION = (
    "Identify one NCAA Division I university athletic director who was appointed to their current position between "
    "January 1, 2022 and December 31, 2024. For this athletic director, provide the following information with "
    "supporting documentation:\n"
    "1) Current institution name and the exact date (or month and year) of their appointment as athletic director, "
    "with an official URL confirming this appointment.\n"
    "2) The title and institution of the position they held immediately before becoming athletic director at their "
    "current institution. The previous position must be one of: Deputy Athletic Director, Senior Associate Athletic "
    "Director, Associate Athletic Director, or Athletic Director at another institution. Include start and end dates "
    "(at least month and year) for this previous position, showing at least 2 years of service.\n"
    "3) Confirmation that the person holds a master's degree, including the degree type or field of study, with a URL "
    "documenting their educational credentials.\n"
    "4) The title and institution of at least one position the person held prior to their immediate previous position, "
    "in athletic administration or collegiate athletics (e.g., Assistant AD, Associate AD, director-level role in "
    "athletics, Graduate Assistant, or college coach), with a URL documenting this earlier role.\n"
    "All information must be supported by official university websites, athletic department pages, or credible news sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RoleInfo(BaseModel):
    title: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None  # Accept free text like "Aug 2018" or "2018-08-01"
    end_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    person_name: Optional[str] = None  # If the answer specifies/duplicates the person’s name for this role


class CurrentAppointmentInfo(BaseModel):
    institution: Optional[str] = None
    appointment_date: Optional[str] = None  # Accept "Month YYYY" or full date
    urls: List[str] = Field(default_factory=list)  # Official announcement/roster/press that confirms appointment
    ncaa_urls: List[str] = Field(default_factory=list)  # Optional URLs showing NCAA Division I status
    person_name: Optional[str] = None


class DegreeInfo(BaseModel):
    has_masters: Optional[bool] = None  # Optional; extractor may set true/false; None if unclear
    degree_type_or_field: Optional[str] = None  # e.g., "M.S. in Sports Administration"
    institution: Optional[str] = None  # Granting institution, if mentioned
    urls: List[str] = Field(default_factory=list)  # URLs documenting the master's degree
    person_name: Optional[str] = None


class AthleticDirectorProfile(BaseModel):
    person_name: Optional[str] = None
    current: Optional[CurrentAppointmentInfo] = None
    previous: Optional[RoleInfo] = None
    earlier: Optional[RoleInfo] = None
    masters: Optional[DegreeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profile() -> str:
    return """
    You must extract details about exactly one NCAA Division I university athletic director (AD) described in the answer.
    If multiple ADs are mentioned, extract only the first one that appears.

    Extract the following JSON fields:

    person_name: The full name of the athletic director.

    current: Information about the current AD appointment.
      - institution: The current institution (university) where the person serves as Athletic Director.
      - appointment_date: The appointment date (full date or at least Month YYYY).
      - urls: An array of URL(s) in the answer that confirm the appointment and role; include official announcements or roster/leadership pages, or credible news.
      - ncaa_urls: If provided, URLs that explicitly show the institution is an NCAA Division I member (e.g., NCAA page, official athletics page).
      - person_name: If the answer repeats the person's name in this section, capture it (else null).

    previous: The position held immediately before the current AD appointment.
      - title: The job title of the immediate previous position (e.g., Deputy AD, Senior Associate AD, Associate AD, or AD at another school).
      - institution: The institution where this previous position was held.
      - start_date: The start date (at least Month YYYY).
      - end_date: The end date (at least Month YYYY).
      - urls: An array of URL(s) that document this previous position and its dates.
      - person_name: If the answer repeats the person's name in this section, capture it (else null).

    masters: Information about the master's degree.
      - has_masters: true if the answer explicitly confirms the person holds a master's degree; false if explicitly says they do not; null if not clear.
      - degree_type_or_field: The degree type or field (e.g., "M.S. in Kinesiology", "Master of Business Administration").
      - institution: The granting institution (if mentioned).
      - urls: URL(s) in the answer documenting the master's degree.
      - person_name: If the answer repeats the person's name in this section, capture it (else null).

    earlier: At least one role prior to the immediate previous position, in athletic administration or collegiate athletics.
      - title: The job title for this earlier role (e.g., Assistant AD, Associate AD, Director-level role in athletics, Graduate Assistant, or college coach).
      - institution: The institution/employer of this earlier role.
      - start_date: The start date (if present).
      - end_date: The end date (if present).
      - urls: URL(s) documenting this earlier role.
      - person_name: If the answer repeats the person's name in this section, capture it (else null).

    Rules for URL extraction:
    - Extract only URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
    - Include full URLs. If a URL lacks protocol, prepend http://
    - If a field is missing in the answer, set it to null (for strings) or [] (for arrays).

    Return a single JSON object matching the specified schema exactly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _uniq_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if not it:
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _collect_all_urls(profile: AthleticDirectorProfile) -> List[str]:
    urls: List[str] = []
    if profile.current:
        urls.extend(profile.current.urls or [])
        urls.extend(profile.current.ncaa_urls or [])
    if profile.previous:
        urls.extend(profile.previous.urls or [])
    if profile.masters:
        urls.extend(profile.masters.urls or [])
    if profile.earlier:
        urls.extend(profile.earlier.urls or [])
    return _uniq_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_entity_consistency(evaluator: Evaluator, parent, profile: AthleticDirectorProfile) -> None:
    node = evaluator.add_leaf(
        id="entity_consistency",
        desc="All provided details (current role, prior roles, education) refer to the same athletic director.",
        parent=parent,
        critical=True,
    )
    claimed_name = profile.person_name or (
        (profile.current.person_name if profile.current else None)
        or (profile.previous.person_name if profile.previous else None)
        or (profile.masters.person_name if profile.masters else None)
        or (profile.earlier.person_name if profile.earlier else None)
        or ""
    )

    claim = (
        f"All described roles and educational details in the answer consistently refer to the same person "
        f"named '{claimed_name}'. There is no mixing of different individuals."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=None,
        additional_instruction="Base your judgment on the answer content and its cited context; ensure there is no identity mix-up across sections."
    )


async def verify_current_appointment(evaluator: Evaluator, parent, profile: AthleticDirectorProfile) -> None:
    cur = profile.current or CurrentAppointmentInfo()

    cur_node = evaluator.add_parallel(
        id="current_ad_and_appointment",
        desc="Provide and verify the person's current NCAA Division I AD appointment (within the required date range) with documentation.",
        parent=parent,
        critical=True,
    )

    # Current institution name provided (existence)
    evaluator.add_custom_node(
        result=bool(cur.institution and cur.institution.strip()),
        id="current_institution_name_provided",
        desc="Provide the current institution name.",
        parent=cur_node,
        critical=True,
    )

    # Institution is NCAA Division I
    node_div1 = evaluator.add_leaf(
        id="institution_is_ncaa_division_i",
        desc="The current institution is an NCAA Division I university.",
        parent=cur_node,
        critical=True,
    )
    div_sources = _uniq_preserve_order((cur.urls or []) + (cur.ncaa_urls or []))
    inst_name = cur.institution or ""
    await evaluator.verify(
        claim=f"The institution '{inst_name}' is an NCAA Division I university.",
        node=node_div1,
        sources=div_sources,
        additional_instruction="Verify only if the provided page(s) explicitly indicate NCAA Division I (e.g., 'NCAA Division I', 'D-I', membership in a Division I conference). If no URL is provided, mark as Incorrect."
    )

    # Appointment date provided (existence)
    evaluator.add_custom_node(
        result=bool(cur.appointment_date and cur.appointment_date.strip()),
        id="appointment_date_provided",
        desc="Provide the exact appointment date or at least month and year for the current AD appointment.",
        parent=cur_node,
        critical=True,
    )

    # Appointment date in range
    node_range = evaluator.add_leaf(
        id="appointment_date_in_range",
        desc="The appointment date falls between January 1, 2022 and December 31, 2024 (inclusive).",
        parent=cur_node,
        critical=True,
    )
    appt_date = cur.appointment_date or ""
    await evaluator.verify(
        claim=f"The appointment date '{appt_date}' is between 2022-01-01 and 2024-12-31 inclusive.",
        node=node_range,
        sources=None,
        additional_instruction="Interpret common date formats, including 'Month YYYY'. If only month/year is given, consider any day in that month; then decide if it lies in the inclusive range."
    )

    # Appointment documented with URL (support)
    node_doc = evaluator.add_leaf(
        id="appointment_documented_with_url",
        desc="Provide at least one URL that confirms the current AD role and the appointment date (or month/year).",
        parent=cur_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The cited page(s) confirm that {profile.person_name or inst_name} was appointed as Athletic Director at '{inst_name}' on '{appt_date}' (or in that month/year).",
        node=node_doc,
        sources=cur.urls or [],
        additional_instruction="Confirm both the AD role and the appointment timing. If no URL is provided, mark Incorrect."
    )


async def verify_previous_position(evaluator: Evaluator, parent, profile: AthleticDirectorProfile) -> None:
    prev = profile.previous or RoleInfo()
    cur = profile.current or CurrentAppointmentInfo()

    prev_node = evaluator.add_parallel(
        id="immediate_previous_position",
        desc="Provide and verify the role held immediately before the current AD role, including allowed title type, dates, and >=2 years duration, with documentation.",
        parent=parent,
        critical=True,
    )

    # Immediate previous position claim (must be immediately prior)
    node_immediate = evaluator.add_leaf(
        id="previous_position_is_immediately_prior",
        desc="The identified previous position is the position held immediately before the current AD appointment.",
        parent=prev_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The previous role '{prev.title or ''}' at '{prev.institution or ''}' was the immediate role held before the "
            f"current AD appointment at '{cur.institution or ''}' on '{cur.appointment_date or ''}', as indicated by the timelines."
        ),
        node=node_immediate,
        sources=_uniq_preserve_order((prev.urls or []) + (cur.urls or [])),
        additional_instruction="Accept if explicitly stated or if the end date of the previous role aligns directly before (or overlaps up to) the AD appointment start date."
    )

    # Title in allowed set
    node_allowed = evaluator.add_leaf(
        id="previous_title_in_allowed_set",
        desc="Previous position title is in the allowed set.",
        parent=prev_node,
        critical=True,
    )
    allowed_set_str = "Deputy Athletic Director; Senior Associate Athletic Director; Associate Athletic Director; Athletic Director (at another institution)"
    await evaluator.verify(
        claim=(
            f"The previous title '{prev.title or ''}' is one of the allowed titles: {allowed_set_str}."
        ),
        node=node_allowed,
        sources=None,
        additional_instruction="Allow minor variations/synonyms such as 'Deputy Director of Athletics', 'Sr. Associate AD', etc."
    )

    # Previous institution provided
    evaluator.add_custom_node(
        result=bool(prev.institution and prev.institution.strip()),
        id="previous_institution_provided",
        desc="Provide the institution name where the immediate previous position was held.",
        parent=prev_node,
        critical=True,
    )

    # Start and end dates provided
    evaluator.add_custom_node(
        result=bool(prev.start_date and prev.start_date.strip() and prev.end_date and prev.end_date.strip()),
        id="previous_position_start_end_dates_provided",
        desc="Provide start and end dates (at least month and year) for the immediate previous position.",
        parent=prev_node,
        critical=True,
    )

    # Duration at least 2 years
    node_duration = evaluator.add_leaf(
        id="previous_position_duration_at_least_2_years",
        desc="The provided start/end dates demonstrate at least 2 years in the immediate previous position.",
        parent=prev_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"From '{prev.start_date or ''}' to '{prev.end_date or ''}' is a duration of at least 2 years."
        ),
        node=node_duration,
        sources=None,
        additional_instruction="Interpret dates with month/year granularity as needed. If only months/years are provided, assume day 1 for start and last day for end, and determine if duration >= 24 months."
    )

    # Documented with URL(s)
    node_prev_doc = evaluator.add_leaf(
        id="previous_position_documented_with_url",
        desc="Provide at least one URL documenting the immediate previous position and its dates.",
        parent=prev_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The cited page(s) document that {profile.person_name or ''} held the role '{prev.title or ''}' at "
            f"'{prev.institution or ''}' from '{prev.start_date or ''}' to '{prev.end_date or ''}'."
        ),
        node=node_prev_doc,
        sources=prev.urls or [],
        additional_instruction="If no URL is provided, mark Incorrect. If dates are partially shown across multiple official pages, that is acceptable when combined they imply the range."
    )


async def verify_masters_degree(evaluator: Evaluator, parent, profile: AthleticDirectorProfile) -> None:
    deg = profile.masters or DegreeInfo()

    deg_node = evaluator.add_parallel(
        id="masters_degree",
        desc="Confirm the person holds a master's degree and document it.",
        parent=parent,
        critical=True,
    )

    node_confirm = evaluator.add_leaf(
        id="masters_degree_confirmed",
        desc="Confirm the person holds a master's degree.",
        parent=deg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{profile.person_name or ''} holds a master's degree.",
        node=node_confirm,
        sources=deg.urls or [],
        additional_instruction="Confirm explicitly from the cited page(s). If no URL is provided, mark Incorrect."
    )

    evaluator.add_custom_node(
        result=bool(deg.degree_type_or_field and deg.degree_type_or_field.strip()),
        id="masters_degree_type_or_field_provided",
        desc="Provide the degree type or field of study for the master's degree.",
        parent=deg_node,
        critical=True,
    )

    node_doc = evaluator.add_leaf(
        id="masters_degree_documented_with_url",
        desc="Provide at least one URL documenting the master's degree credential.",
        parent=deg_node,
        critical=True,
    )
    deg_field = deg.degree_type_or_field or "master's degree"
    inst = deg.institution or ""
    await evaluator.verify(
        claim=f"The cited page(s) document that {profile.person_name or ''} earned a {deg_field}{(' from ' + inst) if inst else ''}.",
        node=node_doc,
        sources=deg.urls or [],
        additional_instruction="The page should clearly indicate a master's degree credential. If no URL is provided, mark Incorrect."
    )


async def verify_earlier_career_role(evaluator: Evaluator, parent, profile: AthleticDirectorProfile) -> None:
    prev = profile.previous or RoleInfo()
    early = profile.earlier or RoleInfo()

    early_node = evaluator.add_parallel(
        id="earlier_career_role",
        desc="Provide at least one role held prior to the immediate previous position, in athletic administration or collegiate athletics, with documentation.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(early.title and early.title.strip() and early.institution and early.institution.strip()),
        id="earlier_role_title_and_institution_provided",
        desc="Provide the title and institution for at least one such earlier role.",
        parent=early_node,
        critical=True,
    )

    node_domain = evaluator.add_leaf(
        id="earlier_role_in_athletics_domain",
        desc="Earlier role is in athletic administration or collegiate athletics domain.",
        parent=early_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The role '{early.title or ''}' at '{early.institution or ''}' is an athletics administration or collegiate athletics role "
            f"(e.g., Assistant AD, Associate AD, director-level athletics role, Graduate Assistant, or college coach)."
        ),
        node=node_domain,
        sources=early.urls or [],
        additional_instruction="Rely on the cited page(s) to judge domain. If unclear or non-athletics, mark Incorrect. If no URL is provided, mark Incorrect."
    )

    node_prior = evaluator.add_leaf(
        id="earlier_role_is_prior_to_immediate_previous",
        desc="The earlier role occurred before the immediate previous position.",
        parent=early_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The earlier role '{early.title or ''}' at '{early.institution or ''}' occurred before the immediate previous position "
            f"that started on '{prev.start_date or ''}'."
        ),
        node=node_prior,
        sources=_uniq_preserve_order((early.urls or []) + (prev.urls or [])),
        additional_instruction="Accept if dates or explicit textual ordering show this role predates the immediate previous one."
    )

    node_doc = evaluator.add_leaf(
        id="earlier_role_documented_with_url",
        desc="Provide at least one URL documenting the earlier career role.",
        parent=early_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The cited page(s) document that {profile.person_name or ''} held the role '{early.title or ''}' at '{early.institution or ''}'.",
        node=node_doc,
        sources=early.urls or [],
        additional_instruction="If no URL is provided, mark Incorrect."
    )


async def verify_source_type_compliance(evaluator: Evaluator, parent, profile: AthleticDirectorProfile) -> None:
    urls = _collect_all_urls(profile)

    src_node = evaluator.add_parallel(
        id="source_type_compliance",
        desc="All provided URLs are from official university websites, official athletic department pages, or credible news sources.",
        parent=parent,
        critical=True,
    )

    # Create a leaf for each URL to independently validate source type compliance
    for idx, url in enumerate(urls):
        leaf = evaluator.add_leaf(
            id=f"source_ok_{idx+1}",
            desc=f"URL {idx+1} source type is acceptable",
            parent=src_node,
            critical=True,
        )
        claim = (
            "This page is either (a) an official university or athletics department website/page (e.g., *.edu, official athletics subdomains), "
            "or (b) a credible news outlet. Pages like Wikipedia, LinkedIn, or personal blogs are not acceptable."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=url,
            additional_instruction="Judge by domain/branding and page content. If ambiguous or clearly not official/credible, mark Incorrect."
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
    # Initialize evaluator (root is a non-critical container)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level criteria are independent checks
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

    # Extract structured profile
    profile = await evaluator.extract(
        prompt=prompt_extract_profile(),
        template_class=AthleticDirectorProfile,
        extraction_name="ad_profile",
    )

    # Add a critical top-level node mirroring the rubric "Root"
    task_node = evaluator.add_parallel(
        id="task_root",
        desc="Identify one NCAA Division I AD appointed between Jan 1, 2022 and Dec 31, 2024 and verify prior role, education, and earlier-career details with documentation.",
        parent=root,
        critical=True,
    )

    # Build verification subtrees
    await verify_entity_consistency(evaluator, task_node, profile)
    await verify_current_appointment(evaluator, task_node, profile)
    await verify_previous_position(evaluator, task_node, profile)
    await verify_masters_degree(evaluator, task_node, profile)
    await verify_earlier_career_role(evaluator, task_node, profile)
    await verify_source_type_compliance(evaluator, task_node, profile)

    # Record all URLs for transparency
    evaluator.add_custom_info(
        info={
            "all_urls": _collect_all_urls(profile),
            "counts": {
                "current_urls": len(profile.current.urls) if profile.current and profile.current.urls else 0,
                "current_ncaa_urls": len(profile.current.ncaa_urls) if profile.current and profile.current.ncaa_urls else 0,
                "previous_urls": len(profile.previous.urls) if profile.previous and profile.previous.urls else 0,
                "masters_urls": len(profile.masters.urls) if profile.masters and profile.masters.urls else 0,
                "earlier_urls": len(profile.earlier.urls) if profile.earlier and profile.earlier.urls else 0,
            },
        },
        info_type="url_inventory",
        info_name="url_inventory",
    )

    return evaluator.get_summary()