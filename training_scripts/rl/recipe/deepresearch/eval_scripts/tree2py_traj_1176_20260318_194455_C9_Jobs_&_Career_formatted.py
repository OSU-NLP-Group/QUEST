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
TASK_ID = "bigten_ad_hire_2025_2026"
TASK_DESCRIPTION = (
    "Between July 2025 and February 2026, a major Big Ten Conference university appointed a new athletic director who "
    "had previously served as a deputy athletic director (or equivalent executive position such as Executive Senior "
    "Associate AD or Deputy AD/COO) at a different Power Five conference institution. This individual had held their "
    "previous deputy AD role for at least three consecutive years before accepting the Big Ten position. The previous "
    "institution was a member of one of the other Power Five conferences (SEC, ACC, Big 12, or former Pac-12), not the "
    "Big Ten. Identify the full name of this athletic director and provide the following information: (1) The name of "
    "the Big Ten university where they were appointed as athletic director, (2) The month and year of their appointment "
    "announcement, (3) The name of their previous institution where they served as deputy athletic director, (4) The "
    "conference affiliation of their previous institution, (5) The approximate duration (in years) they served in the "
    "deputy AD role at their previous institution. All information must be supported by verifiable references from "
    "official university announcements, athletic department websites, or credible news sources."
)

WINDOW_START = "July 2025"
WINDOW_END = "February 2026"
_ALLOWED_PREV_CONFS = {"SEC", "ACC", "Big 12", "Pac-12", "Former Pac-12", "Pac 12", "former Pac-12", "former Pac 12"}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SourceSet(BaseModel):
    # URLs supporting the appointment announcement and role at Big Ten school
    appointment_urls: List[str] = Field(default_factory=list)
    # URLs supporting the previous role (deputy AD or equivalent) and tenure at the previous institution
    previous_role_urls: List[str] = Field(default_factory=list)
    # URLs supporting the conference membership of the previous institution (SEC/ACC/Big 12/former Pac-12)
    previous_conference_urls: List[str] = Field(default_factory=list)
    # URLs supporting Big Ten membership of the hiring institution (e.g., Big Ten official site, school athletics site)
    big_ten_membership_urls: List[str] = Field(default_factory=list)
    # Any other relevant citations the answer provided
    other_urls: List[str] = Field(default_factory=list)


class ADExtraction(BaseModel):
    # Required output fields
    full_name: Optional[str] = None
    hiring_university: Optional[str] = None
    appointment_month_year: Optional[str] = None  # e.g., "September 2025"
    previous_institution: Optional[str] = None
    previous_conference: Optional[str] = None
    deputy_tenure_approx_years: Optional[str] = None  # e.g., "3", "3+","about 4", "approximately 5"

    # Helpful auxiliary fields for constraint checks
    current_role_title: Optional[str] = None  # e.g., "Vice President and Director of Athletics"
    previous_role_title: Optional[str] = None  # e.g., "Deputy AD/COO", "Executive Senior Associate AD"
    deputy_tenure_start: Optional[str] = None  # e.g., "August 2021"
    deputy_tenure_end: Optional[str] = None    # e.g., "September 2025" or "2025"

    # Citations
    sources: SourceSet = Field(default_factory=SourceSet)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ad_info() -> str:
    return """
Extract the following fields for the Big Ten athletic director described in the answer. Only extract information explicitly present in the answer. If an item is not present, set it to null (or an empty list for URL arrays).

Required output fields:
- full_name: Full name of the athletic director.
- hiring_university: The Big Ten university that appointed the person as athletic director.
- appointment_month_year: The month and year of the appointment announcement (e.g., "September 2025").
- previous_institution: The name of the previous institution where the person served as deputy athletic director (or equivalent).
- previous_conference: The conference affiliation of the previous institution (e.g., "SEC", "ACC", "Big 12", "Pac-12", or "former Pac-12").
- deputy_tenure_approx_years: The approximate duration in years served in the deputy AD (or equivalent) role at the previous institution (e.g., "3", "about 4").

Helpful auxiliary fields (if present in the answer):
- current_role_title: The title of the new role at the Big Ten university (e.g., "Director of Athletics", "Vice President and Director of Athletics").
- previous_role_title: The title of the previous role (e.g., "Deputy Athletic Director", "Executive Senior Associate AD", "Deputy AD/COO").
- deputy_tenure_start: Start month/year or year of the deputy role at the previous institution (e.g., "August 2021").
- deputy_tenure_end: End month/year or year of the deputy role (or the month/year immediately before the Big Ten appointment), if stated.

Citations (URLs only; include any provided in the answer; leave empty if none):
- sources.appointment_urls: URLs to official announcements or credible reports confirming the appointment and its date.
- sources.previous_role_urls: URLs confirming the person’s previous deputy AD (or equivalent) role and tenure at the previous institution.
- sources.previous_conference_urls: URLs confirming the previous institution’s conference membership (SEC/ACC/Big 12/former Pac-12).
- sources.big_ten_membership_urls: URLs confirming the hiring university’s Big Ten membership (e.g., Big Ten official members page or the school’s athletics site).
- sources.other_urls: Any other relevant citations provided in the answer.

Return a single JSON object matching the ADExtraction schema exactly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _combine_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if u not in seen and _non_empty(u):
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ADExtraction) -> None:
    # Top-level container (critical) mirroring the rubric root
    task_node = evaluator.add_parallel(
        id="Athletic_Director_Identification_Task",
        desc="Identify an athletic director appointed at a Big Ten university (July 2025–Feb 2026) who previously served ≥3 consecutive years as a deputy AD (or equivalent) at a different, non-Big Ten Power Five institution; provide details with verifiable sources.",
        parent=evaluator.root,
        critical=True,
    )

    # Pre-check: existence of key citation groups (critical; used as automatic gate via critical-sibling rule)
    appt_sources_exist = evaluator.add_custom_node(
        result=_any_urls(extracted.sources.appointment_urls),
        id="Appointment_Announcement_Sources_Provided",
        desc="At least one appointment announcement or credible report URL is provided.",
        parent=task_node,
        critical=True,
    )
    prev_role_sources_exist = evaluator.add_custom_node(
        result=_any_urls(extracted.sources.previous_role_urls),
        id="Previous_Role_Sources_Provided",
        desc="At least one URL supports the previous deputy AD (or equivalent) role and tenure.",
        parent=task_node,
        critical=True,
    )
    prev_conf_sources_exist = evaluator.add_custom_node(
        result=_any_urls(extracted.sources.previous_conference_urls),
        id="Previous_Conference_Sources_Provided",
        desc="At least one URL supports the previous institution’s conference membership.",
        parent=task_node,
        critical=True,
    )
    bigten_membership_sources_exist = evaluator.add_custom_node(
        result=_any_urls(extracted.sources.big_ten_membership_urls),
        id="Big_Ten_Membership_Sources_Provided",
        desc="At least one URL supports the hiring university’s Big Ten membership as of 2026.",
        parent=task_node,
        critical=True,
    )

    # 1) Required Answer Fields Present (critical, parallel)
    req_node = evaluator.add_parallel(
        id="Required_Answer_Fields_Present",
        desc="Response provides all requested output fields for the identified athletic director.",
        parent=task_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(extracted.full_name),
        id="Full_Name_Provided",
        desc="Provides the full name of the athletic director.",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(extracted.hiring_university),
        id="Big_Ten_University_Provided",
        desc="Provides the name of the Big Ten university where they were appointed athletic director.",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(extracted.appointment_month_year),
        id="Appointment_Month_Year_Provided",
        desc="Provides the month and year of the appointment announcement.",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(extracted.previous_institution),
        id="Previous_Institution_Provided",
        desc="Provides the name of the previous institution where they served as deputy athletic director (or equivalent).",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(extracted.previous_conference),
        id="Previous_Institution_Conference_Provided",
        desc="Provides the conference affiliation of the previous institution.",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(extracted.deputy_tenure_approx_years) or _non_empty(extracted.deputy_tenure_start),
        id="Deputy_AD_Tenure_Duration_Provided",
        desc="Provides the approximate duration (in years) served in the deputy AD (or equivalent) role at the previous institution.",
        parent=req_node,
        critical=True,
    )

    # 2) Constraint Verification (critical, parallel)
    cons_node = evaluator.add_parallel(
        id="Constraint_Verification",
        desc="The identified candidate satisfies all stated constraints.",
        parent=task_node,
        critical=True,
    )

    # 2.1 Hiring institution is Big Ten as of 2026
    bigten_leaf = evaluator.add_leaf(
        id="Hiring_Institution_Is_Big_Ten_As_Of_2026",
        desc="Confirms the hiring institution is a current Big Ten Conference member as of 2026.",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{extracted.hiring_university} is a member of the Big Ten Conference as of 2026.",
        node=bigten_leaf,
        sources=extracted.sources.big_ten_membership_urls,
        additional_instruction="Accept Big Ten official members list or the hiring university's official athletics site that clearly indicates Big Ten membership.",
    )

    # 2.2 Appointment announced within window (July 1, 2025 to Feb 28, 2026)
    appt_window_leaf = evaluator.add_leaf(
        id="Appointment_Announced_Within_Window",
        desc="Confirms the appointment announcement date is between July 1, 2025 and February 28, 2026 (inclusive).",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official appointment announcement for {extracted.full_name} at {extracted.hiring_university} "
              f"was published in {extracted.appointment_month_year}, which falls between July 1, 2025 and February 28, 2026 (inclusive).",
        node=appt_window_leaf,
        sources=extracted.sources.appointment_urls,
        additional_instruction="Verify the page's stated or metadata publication date or the announcement's date stamp; confirm it lies in the inclusive window.",
    )

    # 2.3 Current role is AD or equivalent
    role_equiv_leaf = evaluator.add_leaf(
        id="Current_Role_Is_AD_Equivalent",
        desc="Confirms the person’s new role is titled Director of Athletics / Athletic Director or an equivalent title.",
        parent=cons_node,
        critical=True,
    )
    # Avoid over-relying on extracted title; let the page speak
    await evaluator.verify(
        claim=f"The appointment at {extracted.hiring_university} is to an athletic director (or equivalently titled) position.",
        node=role_equiv_leaf,
        sources=extracted.sources.appointment_urls,
        additional_instruction="Pass if the page clearly states they are 'Director of Athletics', 'Athletic Director', or an explicitly equivalent AD leadership title.",
    )

    # 2.4 Previous role is deputy AD equivalent (at a Division I university)
    prev_role_leaf = evaluator.add_leaf(
        id="Previous_Role_Is_Deputy_AD_Equivalent_At_D1",
        desc="Confirms the person previously held a deputy athletic director role or equivalent executive-level position at a Division I university.",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Before joining {extracted.hiring_university}, {extracted.full_name} held a deputy athletic director (or equivalent executive) role at {extracted.previous_institution}.",
        node=prev_role_leaf,
        sources=extracted.sources.previous_role_urls,
        additional_instruction="Treat titles such as 'Deputy Athletic Director', 'Executive Senior Associate AD', or 'Deputy AD/COO' as deputy-AD-equivalent executive positions.",
    )

    # 2.5 Previous institution differs from hiring institution
    diff_inst_leaf = evaluator.add_leaf(
        id="Previous_Institution_Different_From_Hiring_Institution",
        desc="Confirms the previous institution is not the same as the Big Ten hiring institution.",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The previous institution ({extracted.previous_institution}) is not the same as the hiring institution ({extracted.hiring_university}).",
        node=diff_inst_leaf,
        sources=None,  # logical check based on extracted values
        additional_instruction="Only check logical inequality of the two institution names; minor stylistic differences (e.g., 'University of X' vs 'X University') still count as the same if they refer to the same institution.",
    )

    # 2.6 Previous institution conference eligible (SEC/ACC/Big 12/former Pac-12, not Big Ten)
    prev_conf_leaf = evaluator.add_leaf(
        id="Previous_Institution_Conference_Eligible",
        desc="Confirms the previous institution was in SEC, ACC, Big 12, or former Pac-12 (i.e., not Big Ten).",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{extracted.previous_institution} is/was a member of the {extracted.previous_conference} conference, "
              f"which is one of SEC, ACC, Big 12, or former Pac-12 (and not Big Ten).",
        node=prev_conf_leaf,
        sources=extracted.sources.previous_conference_urls,
        additional_instruction="Pass if the page clearly ties the institution to SEC, ACC, Big 12, or (former) Pac-12. Fail if it suggests Big Ten membership.",
    )

    # 2.7 Deputy AD tenure at least 3 consecutive years
    tenure_claim_parts = []
    if _non_empty(extracted.deputy_tenure_start):
        tenure_claim_parts.append(f"from {extracted.deputy_tenure_start}")
    if _non_empty(extracted.deputy_tenure_end):
        tenure_claim_parts.append(f"through {extracted.deputy_tenure_end}")
    if _non_empty(extracted.deputy_tenure_approx_years):
        tenure_claim_parts.append(f"for about {extracted.deputy_tenure_approx_years} years")
    tenure_phrase = ", ".join(tenure_claim_parts) if tenure_claim_parts else "for at least three consecutive years"

    tenure_leaf = evaluator.add_leaf(
        id="Deputy_AD_Tenure_At_Least_3_Consecutive_Years",
        desc="Confirms the person served as deputy AD (or equivalent) for at least 3 consecutive years at the previous institution.",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At {extracted.previous_institution}, {extracted.full_name} served in a deputy athletic director "
              f"(or equivalent) role {tenure_phrase}, which amounts to at least three consecutive years before the Big Ten AD appointment.",
        node=tenure_leaf,
        sources=_combine_sources(extracted.sources.previous_role_urls, extracted.sources.appointment_urls),
        additional_instruction="The page can show start and end dates, or language like 'served since 20XX' and 'until 20YY', or an explicitly stated multi-year tenure of 3+ years.",
    )

    # 2.8 Official announcement source included
    official_leaf = evaluator.add_leaf(
        id="Official_Announcement_Source_Included",
        desc="Includes an official university announcement (or equivalent official documentation) supporting the hiring/announcement and its date.",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page is an official university or athletics department announcement (or an equivalent official documentation) "
              f"confirming that {extracted.full_name} was appointed AD at {extracted.hiring_university}, and it states the announcement date.",
        node=official_leaf,
        sources=extracted.sources.appointment_urls,
        additional_instruction="Prioritize .edu domains, official athletics department sites, or formal press releases. Credible major news outlets are acceptable if clearly confirming the appointment and date.",
    )

    # 3) Source Verifiability (critical, parallel)
    src_node = evaluator.add_parallel(
        id="Source_Verifiability",
        desc="All provided facts are supported by verifiable references from allowed source types.",
        parent=task_node,
        critical=True,
    )

    # 3.1 Citations cover all required fields (implement as a parallel group of field-specific leaves)
    cover_req_fields = evaluator.add_parallel(
        id="Citations_Cover_All_Required_Output_Fields",
        desc="Each required output field is supported by at least one citation.",
        parent=src_node,
        critical=True,
    )

    # Name supported
    name_supported_leaf = evaluator.add_leaf(
        id="Citation_Supports_Name",
        desc="Citation supports the full name of the athletic director.",
        parent=cover_req_fields,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page confirms the appointee's name is {extracted.full_name}.",
        node=name_supported_leaf,
        sources=_combine_sources(extracted.sources.appointment_urls, extracted.sources.other_urls),
        additional_instruction="The page should clearly state the person's full name matching the provided name.",
    )

    # Hiring university supported
    hiring_supported_leaf = evaluator.add_leaf(
        id="Citation_Supports_Hiring_University",
        desc="Citation supports the hiring university field.",
        parent=cover_req_fields,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page confirms that {extracted.hiring_university} appointed {extracted.full_name} as its athletic director.",
        node=hiring_supported_leaf,
        sources=extracted.sources.appointment_urls,
        additional_instruction="The page should clearly state the hiring institution and the appointment.",
    )

    # Appointment month/year supported
    appt_my_supported_leaf = evaluator.add_leaf(
        id="Citation_Supports_Appointment_Month_Year",
        desc="Citation supports the stated appointment month/year.",
        parent=cover_req_fields,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page shows the appointment announcement occurred in {extracted.appointment_month_year}.",
        node=appt_my_supported_leaf,
        sources=extracted.sources.appointment_urls,
        additional_instruction="Use the article dateline, byline, or page metadata to confirm the month and year.",
    )

    # Previous institution supported
    prev_inst_supported_leaf = evaluator.add_leaf(
        id="Citation_Supports_Previous_Institution",
        desc="Citation supports the previous institution field.",
        parent=cover_req_fields,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page states that, before joining {extracted.hiring_university}, {extracted.full_name} was at {extracted.previous_institution}.",
        node=prev_inst_supported_leaf,
        sources=_combine_sources(extracted.sources.appointment_urls, extracted.sources.previous_role_urls),
        additional_instruction="The page should explicitly mention the person's previous institution.",
    )

    # Previous conference supported
    prev_conf_supported_leaf = evaluator.add_leaf(
        id="Citation_Supports_Previous_Conference",
        desc="Citation supports the previous institution's conference affiliation.",
        parent=cover_req_fields,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page confirms {extracted.previous_institution} is/was a member of the {extracted.previous_conference} conference.",
        node=prev_conf_supported_leaf,
        sources=extracted.sources.previous_conference_urls,
        additional_instruction="The page should clearly connect the institution to SEC, ACC, Big 12, or (former) Pac-12.",
    )

    # Deputy tenure duration supported
    tenure_supported_leaf = evaluator.add_leaf(
        id="Citation_Supports_Tenure_Duration",
        desc="Citation supports the approximate tenure duration in the deputy AD role.",
        parent=cover_req_fields,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page provides tenure information indicating about {extracted.deputy_tenure_approx_years} years "
              f"in a deputy AD (or equivalent) role at {extracted.previous_institution}, or provides start/end dates consistent with that duration.",
        node=tenure_supported_leaf,
        sources=_combine_sources(extracted.sources.previous_role_urls, extracted.sources.appointment_urls),
        additional_instruction="Either an explicit 'X years' is acceptable or a clear timeline that implies the same.",
    )

    # 3.2 Citations cover eligibility-only claims (parallel subgroup)
    cover_elig = evaluator.add_parallel(
        id="Citations_Cover_Eligibility_Only_Claims",
        desc="Eligibility-only claims are supported by at least one citation.",
        parent=src_node,
        critical=True,
    )

    # Big Ten membership citation
    elig_bigten_leaf = evaluator.add_leaf(
        id="Eligibility_BigTen_Membership_Cited",
        desc="Citation supports Big Ten membership of hiring institution.",
        parent=cover_elig,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page confirms that {extracted.hiring_university} is a Big Ten Conference member (as of 2026).",
        node=elig_bigten_leaf,
        sources=extracted.sources.big_ten_membership_urls,
        additional_instruction="Big Ten official members listing or the university's athletics site stating 'Big Ten' are acceptable.",
    )

    # Deputy-AD-equivalent citation
    elig_deputy_equiv_leaf = evaluator.add_leaf(
        id="Eligibility_Deputy_EQ_Cited",
        desc="Citation supports that previous role is deputy AD or equivalent.",
        parent=cover_elig,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page shows that {extracted.full_name} held a deputy athletic director or equivalent executive role at {extracted.previous_institution}.",
        node=elig_deputy_equiv_leaf,
        sources=extracted.sources.previous_role_urls,
        additional_instruction="Titles like 'Executive Senior Associate AD', 'Deputy AD/COO' count as deputy-AD-equivalent.",
    )

    # Appointment window citation (re-affirm eligibility window support)
    elig_window_leaf = evaluator.add_leaf(
        id="Eligibility_Window_Cited",
        desc="Citation supports the appointment announcement date needed to evaluate the time window.",
        parent=cover_elig,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page provides the appointment announcement date in {extracted.appointment_month_year}, enabling verification against the July 2025–February 2026 window.",
        node=elig_window_leaf,
        sources=extracted.sources.appointment_urls,
        additional_instruction="The page should clearly present a date stamp or publication date aligned with the claimed month/year.",
    )

    # 3.3 All citations from allowed source types (single leaf)
    # We evaluate domain/source type acceptability via a simple logical verification on the provided URLs.
    all_urls = _combine_sources(
        extracted.sources.appointment_urls,
        extracted.sources.previous_role_urls,
        extracted.sources.previous_conference_urls,
        extracted.sources.big_ten_membership_urls,
        extracted.sources.other_urls,
    )
    allowed_src_leaf = evaluator.add_leaf(
        id="Citations_From_Allowed_Source_Types",
        desc="All citations come from allowed source types: official university announcements, athletic department websites, or credible news sources.",
        parent=src_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"All of these URLs are from allowed source types (official university or athletics department websites, or "
            f"widely recognized credible news organizations), and are acceptable for verifying the claims:\n"
            f"{all_urls}"
        ),
        node=allowed_src_leaf,
        sources=None,
        additional_instruction=(
            "Allowed: .edu domains (including athletics subdomains), official athletics department sites, Big Ten official site, "
            "and major reputable news outlets (e.g., AP, Reuters, ESPN, national newspapers). "
            "Disallow obvious personal blogs, anonymous aggregators, or low-credibility sites."
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
    # Initialize evaluator and root
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

    # Extract structured information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ad_info(),
        template_class=ADExtraction,
        extraction_name="ad_extraction",
    )

    # Record additional custom info (optional): the evaluation window for transparency
    evaluator.add_custom_info(
        info={"window_start": WINDOW_START, "window_end": WINDOW_END, "allowed_previous_conferences": sorted(_ALLOWED_PREV_CONFS)},
        info_type="constraints_window",
        info_name="constraints_window",
    )

    # Build verification tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return the final structured summary
    return evaluator.get_summary()