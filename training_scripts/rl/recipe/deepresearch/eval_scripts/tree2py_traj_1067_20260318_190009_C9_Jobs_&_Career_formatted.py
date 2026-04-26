import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "leadership_appointments_2026"
TASK_DESCRIPTION = """
In early 2026, several prominent higher education leadership appointments were announced across the United States, reflecting diverse career pathways into senior administrative and coaching positions. Your task is to identify four specific appointments that represent distinct career transition patterns:

1. A University President from Dean Background
2. An Athletic Director from Deputy/Associate AD Background
3. A Head Football Coach Returning to a Former Institution
4. A University President with Significant Corporate Experience

For each appointment, all provided information must be verifiable through official university announcements, major news outlets, or the individuals' professional profiles.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PresidentFromDeanInfo(BaseModel):
    # Identity (current role)
    name: Optional[str] = None
    university: Optional[str] = None
    appointment_date: Optional[str] = None  # Keep as string to be flexible
    appointment_url: Optional[str] = None

    # Previous dean position
    prev_title: Optional[str] = None
    prev_institution: Optional[str] = None
    prev_years: Optional[str] = None
    prev_url: Optional[str] = None

    # Optional extra sources
    extra_urls: List[str] = Field(default_factory=list)


class ADFromDeputyInfo(BaseModel):
    # Identity (current role)
    name: Optional[str] = None
    university: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_url: Optional[str] = None

    # Previous deputy/associate AD position
    prev_title: Optional[str] = None  # e.g., "Deputy AD", "Senior Associate AD"
    prev_institution: Optional[str] = None
    prev_years: Optional[str] = None
    prev_url: Optional[str] = None

    # Optional URLs to support Division I check (membership page, etc.)
    division_urls: List[str] = Field(default_factory=list)
    extra_urls: List[str] = Field(default_factory=list)


class CoachReturningInfo(BaseModel):
    # Identity (current head coach role)
    name: Optional[str] = None
    university: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_url: Optional[str] = None

    # Most recent head coaching position (before returning)
    recent_head_coach_institution: Optional[str] = None
    recent_head_coach_years: Optional[str] = None
    recent_head_coach_url: Optional[str] = None

    # Previous tenure at current institution (returning history)
    prior_role_at_current: Optional[str] = None
    prior_role_years_at_current: Optional[str] = None
    prior_role_url: Optional[str] = None

    extra_urls: List[str] = Field(default_factory=list)


class PresidentFromCorporateInfo(BaseModel):
    # Identity (current role)
    name: Optional[str] = None
    university: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_url: Optional[str] = None

    # Most recent academic position before president
    prev_academic_title: Optional[str] = None
    prev_academic_institution: Optional[str] = None
    prev_academic_years: Optional[str] = None
    prev_academic_url: Optional[str] = None

    # Corporate background
    corporate_company: Optional[str] = None
    corporate_role: Optional[str] = None
    corporate_years: Optional[str] = None  # e.g., "2005–2018", "12 years"
    corporate_url: Optional[str] = None

    extra_urls: List[str] = Field(default_factory=list)


class AppointmentsExtraction(BaseModel):
    president_from_dean: Optional[PresidentFromDeanInfo] = None
    ad_from_deputy: Optional[ADFromDeputyInfo] = None
    coach_returning: Optional[CoachReturningInfo] = None
    president_from_corporate: Optional[PresidentFromCorporateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_appointments() -> str:
    return """
    Extract four appointments from the answer, matching the categories below. Return null for any category that isn't clearly provided in the answer.

    GENERAL RULES:
    - Extract exactly what appears in the answer; do not invent or infer new details.
    - Use strings for dates and years (e.g., "January 25, 2026", "2018–2023", "2019 to 2024").
    - Include full valid URLs (with http/https) that are explicitly present in the answer text.
    - Where a single URL is requested to confirm multiple facts, use the one the answer cites for that purpose. If multiple are present, pick the most authoritative (official university first, then major news, then profiles).
    - If an item is missing in the answer, set it to null (or an empty list for arrays).

    1) president_from_dean:
       - name: full name of the appointed president
       - university: the institution where they were appointed as president
       - appointment_date: announcement or effective date (string as shown)
       - appointment_url: URL confirming the presidential appointment
       - prev_title: exact prior dean title (e.g., "Dean of the College of Engineering")
       - prev_institution: institution where the dean role was held
       - prev_years: service years in the dean role (string as shown)
       - prev_url: URL confirming the dean role
       - extra_urls: any other URLs in the answer that are relevant to this item

    2) ad_from_deputy:
       - name: full name of the appointed athletic director
       - university: institution where they were appointed AD
       - appointment_date: announcement or effective date (string)
       - appointment_url: URL confirming the AD appointment
       - prev_title: exact prior deputy/associate title
       - prev_institution: institution of the prior deputy/associate role
       - prev_years: years in the deputy/associate role (string)
       - prev_url: URL confirming the prior deputy/associate role
       - division_urls: any URLs provided in the answer that help confirm NCAA Division I status (membership pages, athletics sites, etc.)
       - extra_urls: any other URLs relevant to this item

    3) coach_returning:
       - name: full name of the head football coach
       - university: institution where they were appointed head coach
       - appointment_date: announcement or effective date (string)
       - appointment_url: URL confirming the head coaching appointment
       - recent_head_coach_institution: most recent head coaching job (institution)
       - recent_head_coach_years: years in that head coaching role (string)
       - recent_head_coach_url: URL confirming the most recent head coaching role
       - prior_role_at_current: specific role previously held at the same (current) institution (e.g., "Offensive Coordinator")
       - prior_role_years_at_current: years of that previous role at current institution (string)
       - prior_role_url: URL documenting the previous tenure at the current institution
       - extra_urls: any other URLs relevant to this item

    4) president_from_corporate:
       - name: full name of the appointed president
       - university: institution where they were appointed president
       - appointment_date: announcement or effective date (string)
       - appointment_url: URL confirming the presidential appointment
       - prev_academic_title: most recent academic title held before president (e.g., Provost, Dean)
       - prev_academic_institution: institution of that prior academic position
       - prev_academic_years: years in that prior academic role (string)
       - prev_academic_url: URL confirming the prior academic position
       - corporate_company: company/business where they worked
       - corporate_role: role/position held in the corporate sector
       - corporate_years: duration of corporate experience (string; should reflect substantial time, ~10+ years)
       - corporate_url: URL documenting the corporate background
       - extra_urls: any other URLs relevant to this item
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*urls_or_lists: Any) -> List[str]:
    """Flatten and combine URL strings and lists into a single clean list."""
    combined: List[str] = []
    for item in urls_or_lists:
        if not item:
            continue
        if isinstance(item, str):
            if item.strip():
                combined.append(item.strip())
        elif isinstance(item, list):
            for u in item:
                if isinstance(u, str) and u.strip():
                    combined.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_president_from_dean(
    evaluator: Evaluator,
    parent,
    data: Optional[PresidentFromDeanInfo],
) -> None:
    node = evaluator.add_parallel(
        id="President_from_Dean",
        desc="Identify a university president appointed in 2026 who previously served as a dean at a different institution",
        parent=parent,
        critical=False,
    )

    # Identity group (critical)
    identity = evaluator.add_parallel(
        id="Dean_President_Identity",
        desc="Correctly identify the individual's name and the university where they were appointed as president",
        parent=node,
        critical=True,
    )

    # Leaves: Name
    leaf = evaluator.add_leaf(
        id="Dean_President_Name",
        desc="Provide the correct full name of the appointed president",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage confirms that {getattr(data, 'name', '')} was appointed as president of {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Verify that the announcement explicitly names the person as the next/appointed/19th/etc. president.",
    )

    # Institution
    leaf = evaluator.add_leaf(
        id="Dean_President_Institution",
        desc="Identify the correct university where the appointment as president was made",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The appointment is at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Accept if the page clearly states the presidency is at the specified university.",
    )

    # Effective/Announcement Date
    leaf = evaluator.add_leaf(
        id="Dean_President_Effective_Date",
        desc="Provide the effective date or announcement date of the presidential appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The presidential appointment was announced or effective on {getattr(data, 'appointment_date', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Allow exact date matches or clearly equivalent phrasing on the page (e.g., 'announced on', 'effective on').",
    )

    # Reference URL validity/support
    leaf = evaluator.add_leaf(
        id="Dean_President_Reference_URL",
        desc="Provide a valid reference URL confirming the presidential appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This is an official or credible announcement confirming that {getattr(data, 'name', '')} was appointed president of {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Accept official university announcements or major news outlets.",
    )

    # Previous position (critical)
    prev = evaluator.add_parallel(
        id="Dean_Previous_Position",
        desc="Correctly identify the dean position held immediately before the presidential appointment",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Dean_Previous_Title",
        desc="Specify the exact dean title held (e.g., Dean of Law School, Dean of College)",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Before becoming president at {getattr(data, 'university', '')}, {getattr(data, 'name', '')} served as {getattr(data, 'prev_title', '')} at {getattr(data, 'prev_institution', '')} during {getattr(data, 'prev_years', '')}.",
        node=leaf,
        sources=combine_sources(getattr(data, 'prev_url', None), getattr(data, 'appointment_url', None)),
        additional_instruction="The page(s) should clearly refer to a dean role at the specified institution with the given title.",
    )

    leaf = evaluator.add_leaf(
        id="Dean_Previous_Institution",
        desc="Identify the institution where the dean position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The dean position was held at {getattr(data, 'prev_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_url', None),
        additional_instruction="Confirm the institution of the dean role.",
    )

    leaf = evaluator.add_leaf(
        id="Dean_Previous_Years",
        desc="Provide the time period during which the dean position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The dean role was held during {getattr(data, 'prev_years', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_url', None),
        additional_instruction="Allow common date span formats such as '2018–2024' or '2018 to 2024' or 'since 2019'.",
    )

    leaf = evaluator.add_leaf(
        id="Dean_Previous_Reference_URL",
        desc="Provide a valid reference URL confirming the previous dean position",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents that {getattr(data, 'name', '')} served as {getattr(data, 'prev_title', '')} at {getattr(data, 'prev_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_url', None),
        additional_instruction="Accept official institution pages, biographies, or major news confirming the dean role.",
    )

    # Career verification (critical)
    ver = evaluator.add_parallel(
        id="Dean_Career_Verification",
        desc="Verify that the career transition follows the dean-to-president pattern",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Dean_Different_Institution",
        desc="Confirm that the dean position was at a different institution than the presidential appointment",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior dean institution ('{getattr(data, 'prev_institution', '')}') is different from the presidential institution ('{getattr(data, 'university', '')}').",
        node=leaf,
        additional_instruction="This is a simple logical/name comparison; consider them different if they refer to different institutions (case-insensitive, ignore minor punctuation).",
    )

    leaf = evaluator.add_leaf(
        id="Dean_2026_Timeline",
        desc="Confirm the presidential appointment was announced or became effective in 2026",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim="The presidential appointment was announced or became effective in 2026.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Pass if the page clearly indicates a 2026 announcement or effective date.",
    )


async def build_ad_from_deputy(
    evaluator: Evaluator,
    parent,
    data: Optional[ADFromDeputyInfo],
) -> None:
    node = evaluator.add_parallel(
        id="AD_from_Deputy",
        desc="Identify an athletic director appointed in 2026 who previously served as a deputy or associate athletic director at a different institution",
        parent=parent,
        critical=False,
    )

    # Identity (critical)
    identity = evaluator.add_parallel(
        id="Deputy_AD_Identity",
        desc="Correctly identify the individual's name and the university where they were appointed as athletic director",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Deputy_AD_Name",
        desc="Provide the correct full name of the appointed athletic director",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page announces {getattr(data, 'name', '')} as athletic director at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Confirm the announcement explicitly names the person as athletic director.",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_AD_Institution",
        desc="Identify the correct university where the appointment as athletic director was made",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The athletic director appointment is at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="The page should make it clear which university appointed the AD.",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_AD_Announcement_Date",
        desc="Provide the announcement date or effective date of the athletic director appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The athletic director appointment was announced or effective on {getattr(data, 'appointment_date', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Allow standard date expressions and clearly equivalent phrasings.",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_AD_Reference_URL",
        desc="Provide a valid reference URL confirming the athletic director appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This is a credible announcement confirming {getattr(data, 'name', '')} as athletic director at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Prefer official athletics or university sites; major news outlets acceptable.",
    )

    # Previous deputy/associate position (critical)
    prev = evaluator.add_parallel(
        id="Deputy_Previous_Position",
        desc="Correctly identify the deputy or associate AD position held immediately before the athletic director appointment",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Deputy_Previous_Title",
        desc="Specify the exact deputy/associate title held (e.g., Deputy AD, Associate AD, Senior Associate AD)",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Immediately prior to the AD appointment at {getattr(data, 'university', '')}, {getattr(data, 'name', '')} served as {getattr(data, 'prev_title', '')} at {getattr(data, 'prev_institution', '')} during {getattr(data, 'prev_years', '')}.",
        node=leaf,
        sources=combine_sources(getattr(data, 'prev_url', None), getattr(data, 'appointment_url', None)),
        additional_instruction="The page(s) should identify a deputy/associate AD role explicitly.",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_Previous_Institution",
        desc="Identify the institution where the deputy/associate position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The deputy/associate AD role was at {getattr(data, 'prev_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_url', None),
        additional_instruction="Confirm the institution for the deputy/associate role.",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_Previous_Years",
        desc="Provide the time period during which the deputy/associate position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The deputy/associate AD role was held during {getattr(data, 'prev_years', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_url', None),
        additional_instruction="Allow common range formats (e.g., '2019–2024').",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_Previous_Reference_URL",
        desc="Provide a valid reference URL confirming the previous deputy/associate position",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents that {getattr(data, 'name', '')} served as {getattr(data, 'prev_title', '')} at {getattr(data, 'prev_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_url', None),
        additional_instruction="Prefer official athletics/university pages or major news.",
    )

    # Career verification (critical)
    ver = evaluator.add_parallel(
        id="Deputy_Career_Verification",
        desc="Verify that the career transition follows the deputy/associate-to-AD pattern",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Deputy_Different_Institution",
        desc="Confirm that the deputy/associate position was at a different institution than the AD appointment",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior deputy/associate institution ('{getattr(data, 'prev_institution', '')}') is different from the AD appointment institution ('{getattr(data, 'university', '')}').",
        node=leaf,
        additional_instruction="Treat differently named universities as different (case-insensitive, ignore punctuation).",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_2026_Timeline",
        desc="Confirm the AD appointment was announced or became effective in 2026",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim="The athletic director appointment was announced or became effective in 2026.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Pass if the page clearly indicates a 2026 date.",
    )

    leaf = evaluator.add_leaf(
        id="Deputy_Division_I",
        desc="Confirm that both institutions (previous and current) are NCAA Division I",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Both {getattr(data, 'university', '')} and {getattr(data, 'prev_institution', '')} compete in NCAA Division I.",
        node=leaf,
        sources=combine_sources(
            getattr(data, 'appointment_url', None),
            getattr(data, 'prev_url', None),
            getattr(data, 'division_urls', []),
            getattr(data, 'extra_urls', []),
        ),
        additional_instruction="Accept if pages explicitly state NCAA Division I membership or membership in a known Division I conference.",
    )


async def build_coach_returning(
    evaluator: Evaluator,
    parent,
    data: Optional[CoachReturningInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Coach_Returning",
        desc="Identify a head football coach appointed in 2026 who returned to an institution where they previously worked in a coaching capacity",
        parent=parent,
        critical=False,
    )

    # Identity (critical)
    identity = evaluator.add_parallel(
        id="Returning_Coach_Identity",
        desc="Correctly identify the coach's name and the university where they were appointed as head coach",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Returning_Coach_Name",
        desc="Provide the correct full name of the appointed head football coach",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page confirms that {getattr(data, 'name', '')} was appointed head football coach at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="The page should explicitly name the person as the head football coach.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Coach_Institution",
        desc="Identify the correct university where the head coaching appointment was made",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The head coaching appointment is at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Confirm the institution of the head coaching appointment.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Coach_Date",
        desc="Provide the announcement date or effective date of the head coaching appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The head coaching appointment was announced or effective on {getattr(data, 'appointment_date', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Allow exact/explainable date phrasing on the page.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Coach_Reference_URL",
        desc="Provide a valid reference URL confirming the head coaching appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This is an official or credible announcement confirming {getattr(data, 'name', '')} as head football coach at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Prefer official athletics/university pages; major news acceptable.",
    )

    # Previous (most recent) head coaching position (critical)
    prev = evaluator.add_parallel(
        id="Returning_Previous_Position",
        desc="Correctly identify the most recent head coaching position held immediately before returning",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Returning_Previous_Title",
        desc="Specify the head coaching title at the previous institution",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Immediately before this appointment, {getattr(data, 'name', '')} served as head football coach at {getattr(data, 'recent_head_coach_institution', '')} during {getattr(data, 'recent_head_coach_years', '')}.",
        node=leaf,
        sources=getattr(data, 'recent_head_coach_url', None),
        additional_instruction="The cited page should clearly indicate the head coaching role and the institution.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Previous_Institution",
        desc="Identify the institution where the previous head coaching position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior head coaching position was at {getattr(data, 'recent_head_coach_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'recent_head_coach_url', None),
        additional_instruction="Confirm the prior head coaching institution.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Previous_Years",
        desc="Provide the time period during which the previous head coaching position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior head coaching role was held during {getattr(data, 'recent_head_coach_years', '')}.",
        node=leaf,
        sources=getattr(data, 'recent_head_coach_url', None),
        additional_instruction="Allow common date span formats; be flexible about en-dash vs hyphen.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Previous_Reference_URL",
        desc="Provide a valid reference URL confirming the previous head coaching position",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents {getattr(data, 'name', '')}'s prior head coaching position at {getattr(data, 'recent_head_coach_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'recent_head_coach_url', None),
        additional_instruction="Prefer official athletics team pages or major news outlets.",
    )

    # Returning history verification (critical)
    ver = evaluator.add_parallel(
        id="Returning_History_Verification",
        desc="Verify that the coach previously worked at the institution to which they returned",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Returning_Prior_Role",
        desc="Identify the specific coaching role held during the previous tenure at the current institution",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{getattr(data, 'name', '')} previously worked at {getattr(data, 'university', '')} as {getattr(data, 'prior_role_at_current', '')}.",
        node=leaf,
        sources=getattr(data, 'prior_role_url', None),
        additional_instruction="The page should clearly indicate the prior role at the same institution (e.g., OC, DC, position coach).",
    )

    leaf = evaluator.add_leaf(
        id="Returning_Prior_Years",
        desc="Provide the time period during which the coach previously worked at the current institution",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior role at {getattr(data, 'university', '')} was held during {getattr(data, 'prior_role_years_at_current', '')}.",
        node=leaf,
        sources=getattr(data, 'prior_role_url', None),
        additional_instruction="Allow standard formats for the date span.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_2026_Timeline",
        desc="Confirm the head coaching appointment was announced or became effective in 2026",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim="The head coaching appointment was announced or became effective in 2026.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Pass if the page dates clearly show 2026.",
    )

    leaf = evaluator.add_leaf(
        id="Returning_History_Reference_URL",
        desc="Provide a valid reference URL documenting the previous tenure at the current institution",
        parent=ver,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents {getattr(data, 'name', '')}'s previous coaching tenure at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'prior_role_url', None),
        additional_instruction="Prefer official athletics or university sites; major news acceptable.",
    )


async def build_president_from_corporate(
    evaluator: Evaluator,
    parent,
    data: Optional[PresidentFromCorporateInfo],
) -> None:
    node = evaluator.add_parallel(
        id="President_from_Corporate",
        desc="Identify a university president appointed in 2026 who had significant corporate/business experience before transitioning to higher education",
        parent=parent,
        critical=False,
    )

    # Identity (critical)
    identity = evaluator.add_parallel(
        id="Corporate_President_Identity",
        desc="Correctly identify the individual's name and the university where they were appointed as president",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Corporate_President_Name",
        desc="Provide the correct full name of the appointed president",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page confirms that {getattr(data, 'name', '')} was appointed president of {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Look for explicit naming of the appointee as president.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_President_Institution",
        desc="Identify the correct university where the appointment as president was made",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The presidency is at {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Confirm the institution on the announcement page.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_President_Date",
        desc="Provide the effective date or announcement date of the presidential appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The presidential appointment was announced or effective on {getattr(data, 'appointment_date', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Allow exact or clearly equivalent phrasing.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_President_Reference_URL",
        desc="Provide a valid reference URL confirming the presidential appointment",
        parent=identity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This is an official or credible page confirming {getattr(data, 'name', '')}'s appointment as president of {getattr(data, 'university', '')}.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Prefer official university announcements or major news outlets.",
    )

    # Previous academic position (critical)
    prev = evaluator.add_parallel(
        id="Corporate_Previous_Position",
        desc="Correctly identify the academic position held immediately before the presidential appointment",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Previous_Title",
        desc="Specify the exact academic title held before becoming president (e.g., Dean, Provost)",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Before the presidency at {getattr(data, 'university', '')}, {getattr(data, 'name', '')} served as {getattr(data, 'prev_academic_title', '')} at {getattr(data, 'prev_academic_institution', '')} during {getattr(data, 'prev_academic_years', '')}.",
        node=leaf,
        sources=combine_sources(getattr(data, 'prev_academic_url', None), getattr(data, 'appointment_url', None)),
        additional_instruction="Confirm the title, institution, and service period of the prior academic role.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Previous_Institution",
        desc="Identify the institution where the previous academic position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior academic role was at {getattr(data, 'prev_academic_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_academic_url', None),
        additional_instruction="Confirm the institution name for the previous academic role.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Previous_Years",
        desc="Provide the time period during which the previous academic position was held",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The prior academic role was held during {getattr(data, 'prev_academic_years', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_academic_url', None),
        additional_instruction="Allow common span formats and approximate language if clearly stated.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Previous_Reference_URL",
        desc="Provide a valid reference URL confirming the previous academic position",
        parent=prev,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents {getattr(data, 'name', '')}'s prior academic role at {getattr(data, 'prev_academic_institution', '')}.",
        node=leaf,
        sources=getattr(data, 'prev_academic_url', None),
        additional_instruction="Prefer official university pages or major news.",
    )

    # Corporate background (critical)
    corp = evaluator.add_parallel(
        id="Corporate_Background_Verification",
        desc="Verify the individual's significant corporate/business experience prior to higher education",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Company_Name",
        desc="Identify the specific corporation or business where the individual worked",
        parent=corp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{getattr(data, 'name', '')} worked at {getattr(data, 'corporate_company', '')}.",
        node=leaf,
        sources=combine_sources(getattr(data, 'corporate_url', None), getattr(data, 'appointment_url', None)),
        additional_instruction="The page(s) should explicitly name the company/business.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Role",
        desc="Specify the corporate role or position held",
        parent=corp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The corporate role held at {getattr(data, 'corporate_company', '')} was {getattr(data, 'corporate_role', '')}.",
        node=leaf,
        sources=getattr(data, 'corporate_url', None),
        additional_instruction="Accept equivalent role titles if clearly the same position (case-insensitive).",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Years",
        desc="Provide the duration of corporate experience (must be substantial, typically 10+ years)",
        parent=corp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The duration of corporate experience is at least about 10 years (the provided duration is '{getattr(data, 'corporate_years', '')}').",
        node=leaf,
        sources=getattr(data, 'corporate_url', None),
        additional_instruction="Consider the years span or explicit 'X years' statements; pass if it's ≥ ~10 years.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_2026_Timeline",
        desc="Confirm the presidential appointment was announced or became effective in 2026",
        parent=corp,
        critical=True,
    )
    await evaluator.verify(
        claim="The presidential appointment was announced or became effective in 2026.",
        node=leaf,
        sources=getattr(data, 'appointment_url', None),
        additional_instruction="Pass if the page clearly indicates a 2026 date.",
    )

    leaf = evaluator.add_leaf(
        id="Corporate_Background_Reference_URL",
        desc="Provide a valid reference URL documenting the corporate background",
        parent=corp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page credibly documents {getattr(data, 'name', '')}'s corporate background at {getattr(data, 'corporate_company', '')}.",
        node=leaf,
        sources=getattr(data, 'corporate_url', None),
        additional_instruction="Prefer official company bios, SEC filings, reputable news, or professional profiles with clear evidence.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2026 Leadership Appointments task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
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

    # Root (as per rubric)
    top = evaluator.add_parallel(
        id="2026_Leadership_Appointments",
        desc="Identify four specific higher education leadership appointments made in 2026, each meeting distinct career transition criteria",
        parent=root,
        critical=False,
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_appointments(),
        template_class=AppointmentsExtraction,
        extraction_name="appointments_extraction",
    )

    # Build and verify each category subtree
    await build_president_from_dean(
        evaluator,
        parent=top,
        data=extracted.president_from_dean or PresidentFromDeanInfo(),
    )
    await build_ad_from_deputy(
        evaluator,
        parent=top,
        data=extracted.ad_from_deputy or ADFromDeputyInfo(),
    )
    await build_coach_returning(
        evaluator,
        parent=top,
        data=extracted.coach_returning or CoachReturningInfo(),
    )
    await build_president_from_corporate(
        evaluator,
        parent=top,
        data=extracted.president_from_corporate or PresidentFromCorporateInfo(),
    )

    return evaluator.get_summary()