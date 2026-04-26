import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "academic_leader_identification"
TASK_DESCRIPTION = """
Identify the full name of the academic leader who satisfies ALL of the following career trajectory criteria:

Educational Background:
- Earned a Juris Doctor (JD) degree from a Big Ten university law school
- Earned an additional graduate degree (MA or MS) in a social science field from the same Big Ten university
- Earned an undergraduate degree from a university located in Washington, D.C.

Early Career:
- Served as a law clerk for a U.S. Supreme Court Justice
- Was born or raised in upstate New York

Law School Administrative Career:
- Served as associate dean for academic affairs at a Big Ten university law school
- Served as dean of a law school at a private university in the southeastern United States
- Served as dean of a law school at a private university in the Midwest
- Total tenure as a law school dean across all institutions spans at least 10 years

University Leadership:
- Was appointed chancellor/president of a private R1 research university located in New York State in 2013

Conference Leadership:
- Served as chair of the Atlantic Coast Conference (ACC) Board of Directors during 2019-2021
- Chaired the ACC commissioner search process in 2020

Career Transition:
- Announced their departure from the chancellor/president position in 2025
- Was appointed president of a public Big Ten university in Michigan in 2026
- The new university holds both R1 Carnegie classification and AAU (Association of American Universities) membership

Provide the person's full name and comprehensive career documentation with URL references for each major career milestone.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DegreeInfo(BaseModel):
    university: Optional[str] = None
    school_or_college: Optional[str] = None  # e.g., Law School
    degree: Optional[str] = None             # e.g., "JD", "MA", "MS"
    field: Optional[str] = None              # For MA/MS (e.g., Economics, Political Science)
    year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    # Optional extra supporting links (e.g., for Big Ten membership verification)
    support_urls: List[str] = Field(default_factory=list)


class UndergradInfo(BaseModel):
    university: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ClerkshipInfo(BaseModel):
    justice_name: Optional[str] = None
    term_or_years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class OriginInfo(BaseModel):
    origin_summary: Optional[str] = None  # e.g., "born in Rochester, New York" or "raised in upstate New York"
    urls: List[str] = Field(default_factory=list)


class AssocDeanInfo(BaseModel):
    university: Optional[str] = None
    law_school: Optional[str] = None
    role_title: Optional[str] = None  # e.g., "Associate Dean for Academic Affairs"
    years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    support_urls: List[str] = Field(default_factory=list)  # e.g., Big Ten membership links


class DeanRoleInfo(BaseModel):
    university: Optional[str] = None
    law_school: Optional[str] = None
    state: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    support_urls: List[str] = Field(default_factory=list)  # e.g., private status, location pages


class LeadershipAppointmentInfo(BaseModel):
    university: Optional[str] = None
    role: Optional[str] = None  # chancellor or president
    year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    support_urls: List[str] = Field(default_factory=list)  # e.g., R1 classification, private status, NY location


class ACCLeadershipInfo(BaseModel):
    board_chair_years: Optional[str] = None  # "2019–2021"
    urls_board_chair: List[str] = Field(default_factory=list)
    commissioner_search_year: Optional[str] = None  # "2020"
    urls_commissioner_search: List[str] = Field(default_factory=list)


class CareerTransitionInfo(BaseModel):
    departure_year: Optional[str] = None  # "2025"
    urls_departure: List[str] = Field(default_factory=list)

    new_university: Optional[str] = None
    appointment_year: Optional[str] = None  # "2026"
    urls_appointment: List[str] = Field(default_factory=list)
    support_urls_big_ten: List[str] = Field(default_factory=list)
    support_urls_r1: List[str] = Field(default_factory=list)
    support_urls_aau: List[str] = Field(default_factory=list)
    support_urls_public: List[str] = Field(default_factory=list)
    support_urls_location: List[str] = Field(default_factory=list)  # e.g., Michigan location page


class PersonProfile(BaseModel):
    full_name: Optional[str] = None

    jd: Optional[DegreeInfo] = None
    ma_ms: Optional[DegreeInfo] = None
    undergrad: Optional[UndergradInfo] = None

    clerkship: Optional[ClerkshipInfo] = None
    origin: Optional[OriginInfo] = None

    assoc_dean_big_ten: Optional[AssocDeanInfo] = None

    dean_private_southeast: Optional[DeanRoleInfo] = None
    dean_private_midwest: Optional[DeanRoleInfo] = None

    leadership_2013_private_r1_ny: Optional[LeadershipAppointmentInfo] = None

    acc_leadership: Optional[ACCLeadershipInfo] = None

    career_transition: Optional[CareerTransitionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_person_profile() -> str:
    return """
    Extract the structured profile of the identified academic leader strictly from the provided answer text. Do not infer or invent missing information. Return null for any unknown field. For each milestone, extract any URLs explicitly cited in the answer to support that milestone.

    Required JSON fields and meanings:
    - full_name: the person’s full name.

    - jd: {
        university: university awarding the JD,
        school_or_college: the specific law school name if given,
        degree: should be "JD" (as stated),
        field: if mentioned (often null for JD),
        year: four-digit year if present,
        urls: list of URLs that support the JD information,
        support_urls: list of URLs (if any are cited) that support Big Ten membership of the JD university
      }

    - ma_ms: {
        university: awarding university (should be the same as JD university for this task),
        school_or_college: if specified,
        degree: "MA" or "MS" exactly as stated,
        field: the social science field (e.g., Economics, Political Science, Sociology, etc.),
        year: four-digit year if present,
        urls: list of URLs supporting the MA/MS info,
        support_urls: list of URLs supporting Big Ten membership of this university (if cited)
      }

    - undergrad: {
        university: name of the undergraduate university,
        degree: e.g., "BA" or "BS" if given,
        field: major if given,
        city: city of the university if given,
        state: state or district (e.g., "Washington, D.C.") if given,
        urls: list of URLs supporting the undergraduate info
      }

    - clerkship: {
        justice_name: the name of the U.S. Supreme Court Justice the person clerked for,
        term_or_years: the clerkship term/years if mentioned,
        urls: list of URLs supporting the clerkship
      }

    - origin: {
        origin_summary: brief phrase indicating birth/raising in upstate New York (e.g., "born in Rochester, New York"),
        urls: list of URLs supporting origin
      }

    - assoc_dean_big_ten: {
        university: Big Ten university where the person served as associate dean for academic affairs,
        law_school: the law school name,
        role_title: exact/approximate title,
        years: timespan if given,
        urls: list of URLs supporting this associate dean role,
        support_urls: URLs supporting Big Ten membership of this university (if cited)
      }

    - dean_private_southeast: {
        university: private university in the southeastern United States,
        law_school: law school name,
        state: state where university is located,
        start_year: four-digit start year if given,
        end_year: four-digit end year if given,
        urls: list of URLs supporting this deanship,
        support_urls: URLs supporting private status and/or location if cited
      }

    - dean_private_midwest: {
        university: private university in the Midwest,
        law_school: law school name,
        state: state where university is located,
        start_year: four-digit start year if given,
        end_year: four-digit end year if given,
        urls: list of URLs supporting this deanship,
        support_urls: URLs supporting private status and/or location if cited
      }

    - leadership_2013_private_r1_ny: {
        university: private R1 research university located in New York State,
        role: chancellor or president,
        year: the appointment year (should be 2013),
        urls: list of URLs supporting this appointment,
        support_urls: URLs supporting private status, R1 classification, and New York location (if cited)
      }

    - acc_leadership: {
        board_chair_years: the years served as chair of the ACC Board of Directors (e.g., "2019–2021"),
        urls_board_chair: list of URLs supporting this,
        commissioner_search_year: "2020" if stated,
        urls_commissioner_search: list of URLs supporting chairing the ACC commissioner search
      }

    - career_transition: {
        departure_year: year when departure from the chancellor/president role was announced (should be 2025),
        urls_departure: URLs supporting the departure announcement,
        new_university: name of the public Big Ten university in Michigan,
        appointment_year: year of appointment (should be 2026),
        urls_appointment: URLs supporting the 2026 appointment,
        support_urls_big_ten: URLs supporting Big Ten membership of the new university (if cited),
        support_urls_r1: URLs supporting R1 status (if cited),
        support_urls_aau: URLs supporting AAU membership (if cited),
        support_urls_public: URLs supporting public status (if cited),
        support_urls_location: URLs supporting Michigan location (if cited)
      }

    Extraction rules:
    - Return only URLs explicitly cited in the answer text. If none are cited for a field, return an empty list for that field's URLs.
    - Preserve the exact strings for names, universities, and titles as shown in the answer whenever possible.
    - If a field is not present in the answer, set it to null and for URL lists return [].
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    result.append(uu)
    return result


def _add_ins_for_urls(base_instruction: str, urls: List[str]) -> str:
    if urls and len(urls) > 0:
        return base_instruction
    # Enforce source-grounding: if no URLs provided, instruct to mark as unsupported
    return base_instruction + "\nIMPORTANT: No URLs were provided for this verification. You must judge this claim as NOT SUPPORTED."


def _safe(s: Optional[str]) -> str:
    return s if (s is not None and str(s).strip() != "") else "UNKNOWN"


def _tenure_years_sum(*spans: tuple[Optional[str], Optional[str]]) -> Optional[int]:
    """
    Best-effort computation of total tenure in years using start/end years (four digits).
    If any span has missing or malformed years, it's skipped. Returns None if nothing computable.
    """
    total = 0
    found_any = False
    for start, end in spans:
        if not start or not end:
            continue
        m1 = re.search(r"\b(19|20)\d{2}\b", start)
        m2 = re.search(r"\b(19|20)\d{2}\b", end)
        if not m1 or not m2:
            continue
        ys, ye = int(m1.group(0)), int(m2.group(0))
        if ye >= ys:
            total += (ye - ys)
            found_any = True
    return total if found_any else None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_educational_background(evaluator: Evaluator, parent, profile: PersonProfile) -> None:
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Educational background criteria are satisfied, with URL evidence for each.",
        parent=parent,
        critical=True,
    )

    person = _safe(profile.full_name)

    # JD from Big Ten law school
    jd_leaf = evaluator.add_leaf(
        id="JD_BigTen_Law_School_With_URL",
        desc="Person earned a JD from a Big Ten university law school, supported by a URL reference.",
        parent=edu_node,
        critical=True,
    )
    jd = profile.jd or DegreeInfo()
    jd_urls = _dedup_urls(jd.urls, jd.support_urls)
    jd_claim = (
        f"{person} earned a Juris Doctor (JD) from {_safe(jd.university)} {_safe(jd.school_or_college)}; "
        f"{_safe(jd.university)} is a member of the Big Ten Conference."
    )
    await evaluator.verify(
        claim=jd_claim,
        node=jd_leaf,
        sources=jd_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify both the JD credential at the named law school and that the awarding university is a Big Ten member.",
            jd_urls
        ),
    )

    # MA/MS social science from the same Big Ten university
    ma_leaf = evaluator.add_leaf(
        id="MA_MS_SocialScience_Same_BigTen_With_URL",
        desc="Person earned an additional MA/MS in a social science field from the same Big Ten university as the JD, supported by a URL reference.",
        parent=edu_node,
        critical=True,
    )
    ma = profile.ma_ms or DegreeInfo()
    ma_urls = _dedup_urls(ma.urls, ma.support_urls, jd.urls, jd.support_urls)
    ma_claim = (
        f"{person} earned an {_safe(ma.degree)} in {_safe(ma.field)} from {_safe(ma.university)} "
        f"(the same university that awarded the JD: {_safe(jd.university)}), and that university is a Big Ten member."
    )
    await evaluator.verify(
        claim=ma_claim,
        node=ma_leaf,
        sources=ma_urls,
        additional_instruction=_add_ins_for_urls(
            "Confirm that the MA/MS is in a social science field and that it is from the same Big Ten university as the JD.",
            ma_urls
        ),
    )

    # Undergraduate degree from a university located in Washington, D.C.
    ug_leaf = evaluator.add_leaf(
        id="Undergraduate_DC_With_URL",
        desc="Person earned an undergraduate degree from a university located in Washington, D.C., supported by a URL reference.",
        parent=edu_node,
        critical=True,
    )
    ug = profile.undergrad or UndergradInfo()
    ug_urls = _dedup_urls(ug.urls)
    ug_claim = (
        f"{person} earned an undergraduate degree from {_safe(ug.university)}, which is located in Washington, D.C."
    )
    await evaluator.verify(
        claim=ug_claim,
        node=ug_leaf,
        sources=ug_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify the undergraduate credential and that the university is located in Washington, D.C.",
            ug_urls
        ),
    )


async def verify_early_career(evaluator: Evaluator, parent, profile: PersonProfile) -> None:
    ec_node = evaluator.add_parallel(
        id="Early_Career",
        desc="Early-career and origin criteria are satisfied, with URL evidence for each.",
        parent=parent,
        critical=True,
    )

    person = _safe(profile.full_name)

    # U.S. Supreme Court clerkship
    clerk_leaf = evaluator.add_leaf(
        id="Supreme_Court_Clerkship_With_URL",
        desc="Person served as a law clerk for a U.S. Supreme Court Justice, supported by a URL reference.",
        parent=ec_node,
        critical=True,
    )
    clerk = profile.clerkship or ClerkshipInfo()
    clerk_urls = _dedup_urls(clerk.urls)
    clerk_claim = (
        f"{person} served as a law clerk for U.S. Supreme Court Justice {_safe(clerk.justice_name)}."
    )
    await evaluator.verify(
        claim=clerk_claim,
        node=clerk_leaf,
        sources=clerk_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify the Supreme Court clerkship (justice's name and role) using the provided sources.",
            clerk_urls
        ),
    )

    # Upstate New York origin (born or raised)
    origin_leaf = evaluator.add_leaf(
        id="Upstate_New_York_Origin_With_URL",
        desc="Person was born or raised in upstate New York, supported by a URL reference.",
        parent=ec_node,
        critical=True,
    )
    origin = profile.origin or OriginInfo()
    origin_urls = _dedup_urls(origin.urls)
    origin_claim = (
        f"{person} was born or raised in upstate New York ({_safe(origin.origin_summary)})."
    )
    await evaluator.verify(
        claim=origin_claim,
        node=origin_leaf,
        sources=origin_urls,
        additional_instruction=_add_ins_for_urls(
            "Treat 'upstate New York' as regions north of New York City, e.g., Rochester, Syracuse, Buffalo, Albany, etc. Confirm from the provided sources.",
            origin_urls
        ),
    )


async def verify_admin_career(evaluator: Evaluator, parent, profile: PersonProfile) -> None:
    admin_node = evaluator.add_parallel(
        id="Law_School_Administrative_Career",
        desc="Law school administrative career criteria are satisfied, with URL evidence for each.",
        parent=parent,
        critical=True,
    )

    person = _safe(profile.full_name)

    # Associate Dean for Academic Affairs at a Big Ten university law school
    assoc_leaf = evaluator.add_leaf(
        id="Associate_Dean_Academic_Affairs_BigTen_With_URL",
        desc="Person served as associate dean for academic affairs at a Big Ten university law school, supported by a URL reference.",
        parent=admin_node,
        critical=True,
    )
    assoc = profile.assoc_dean_big_ten or AssocDeanInfo()
    assoc_urls = _dedup_urls(assoc.urls, assoc.support_urls)
    assoc_claim = (
        f"{person} served as {_safe(assoc.role_title)} at {_safe(assoc.law_school)} of {_safe(assoc.university)}, "
        f"which is a Big Ten university."
    )
    await evaluator.verify(
        claim=assoc_claim,
        node=assoc_leaf,
        sources=assoc_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify both the associate dean for academic affairs role and that the university is a Big Ten member.",
            assoc_urls
        ),
    )

    # Dean at a private university in the Southeastern U.S.
    dean_se_leaf = evaluator.add_leaf(
        id="Dean_Private_Southeast_With_URL",
        desc="Person served as dean of a law school at a private university in the southeastern United States, supported by a URL reference.",
        parent=admin_node,
        critical=True,
    )
    dse = profile.dean_private_southeast or DeanRoleInfo()
    dse_urls = _dedup_urls(dse.urls, dse.support_urls)
    dse_claim = (
        f"{person} served as dean of {_safe(dse.law_school)} at {_safe(dse.university)}, a private university in the state of {_safe(dse.state)}, "
        f"which is part of the southeastern United States."
    )
    await evaluator.verify(
        claim=dse_claim,
        node=dean_se_leaf,
        sources=dse_urls,
        additional_instruction=_add_ins_for_urls(
            "Confirm that the university is private and located in a southeastern U.S. state (e.g., AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV). "
            "Also confirm the deanship role.",
            dse_urls
        ),
    )

    # Dean at a private university in the Midwest
    dean_mw_leaf = evaluator.add_leaf(
        id="Dean_Private_Midwest_With_URL",
        desc="Person served as dean of a law school at a private university in the Midwest, supported by a URL reference.",
        parent=admin_node,
        critical=True,
    )
    dmw = profile.dean_private_midwest or DeanRoleInfo()
    dmw_urls = _dedup_urls(dmw.urls, dmw.support_urls)
    dmw_claim = (
        f"{person} served as dean of {_safe(dmw.law_school)} at {_safe(dmw.university)}, a private university in the state of {_safe(dmw.state)}, "
        f"which is part of the U.S. Midwest."
    )
    await evaluator.verify(
        claim=dmw_claim,
        node=dean_mw_leaf,
        sources=dmw_urls,
        additional_instruction=_add_ins_for_urls(
            "Confirm that the university is private and located in the Midwest (e.g., IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI). "
            "Also confirm the deanship role.",
            dmw_urls
        ),
    )

    # Total dean tenure at least 10 years (across institutions)
    tenure_leaf = evaluator.add_leaf(
        id="Total_Dean_Tenure_At_Least_10_Years_With_URLs",
        desc="Total tenure as a law school dean across all institutions spans at least 10 years, supported by URL references sufficient to verify the duration.",
        parent=admin_node,
        critical=True,
    )
    # Compute best-effort years from extracted values (for claim clarity only)
    total_years = _tenure_years_sum(
        (dse.start_year, dse.end_year),
        (dmw.start_year, dmw.end_year),
    )
    tenure_text = f" (computed total: {total_years} years)" if total_years is not None else ""
    tenure_urls = _dedup_urls(dse.urls, dse.support_urls, dmw.urls, dmw.support_urls)
    tenure_claim = (
        f"{person}'s combined service as a law school dean at {_safe(dse.law_school)} ({_safe(dse.start_year)}–{_safe(dse.end_year)}) "
        f"and {_safe(dmw.law_school)} ({_safe(dmw.start_year)}–{_safe(dmw.end_year)}) spans at least 10 years.{tenure_text}"
    )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=tenure_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify the start and end years of each deanship and confirm that the combined tenure is at least 10 years.",
            tenure_urls
        ),
    )


async def verify_university_leadership(evaluator: Evaluator, parent, profile: PersonProfile) -> None:
    ul_node = evaluator.add_parallel(
        id="University_Leadership",
        desc="University leadership criterion is satisfied with URL evidence.",
        parent=parent,
        critical=True,
    )

    person = _safe(profile.full_name)

    # Appointed in 2013 as chancellor/president of a private R1 research university in New York State
    ul_leaf = evaluator.add_leaf(
        id="Appointed_2013_Private_R1_NY_Chancellor_President_With_URL",
        desc="Person was appointed chancellor/president of a private R1 research university located in New York State in 2013, supported by a URL reference.",
        parent=ul_node,
        critical=True,
    )
    lead = profile.leadership_2013_private_r1_ny or LeadershipAppointmentInfo()
    ul_urls = _dedup_urls(lead.urls, lead.support_urls)
    ul_claim = (
        f"In 2013, {person} was appointed {_safe(lead.role)} of {_safe(lead.university)}, "
        f"a private university in New York State with R1 Carnegie classification."
    )
    await evaluator.verify(
        claim=ul_claim,
        node=ul_leaf,
        sources=ul_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify the 2013 appointment year, the role (chancellor/president), private status, New York State location, and R1 classification.",
            ul_urls
        ),
    )


async def verify_conference_leadership(evaluator: Evaluator, parent, profile: PersonProfile) -> None:
    conf_node = evaluator.add_parallel(
        id="Conference_Leadership",
        desc="Conference leadership criteria are satisfied, with URL evidence for each.",
        parent=parent,
        critical=True,
    )

    person = _safe(profile.full_name)
    acc = profile.acc_leadership or ACCLeadershipInfo()

    # ACC Board Chair 2019–2021
    acc_chair_leaf = evaluator.add_leaf(
        id="ACC_Board_Chair_2019_2021_With_URL",
        desc="Person served as chair of the ACC Board of Directors during 2019–2021, supported by a URL reference.",
        parent=conf_node,
        critical=True,
    )
    chair_urls = _dedup_urls(acc.urls_board_chair)
    acc_chair_claim = (
        f"{person} served as chair of the Atlantic Coast Conference (ACC) Board of Directors during 2019–2021."
    )
    await evaluator.verify(
        claim=acc_chair_claim,
        node=acc_chair_leaf,
        sources=chair_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify that the person served as ACC Board of Directors chair spanning 2019, 2020, and 2021.",
            chair_urls
        ),
    )

    # Chaired ACC commissioner search 2020
    acc_search_leaf = evaluator.add_leaf(
        id="Chaired_ACC_Commissioner_Search_2020_With_URL",
        desc="Person chaired the ACC commissioner search process in 2020, supported by a URL reference.",
        parent=conf_node,
        critical=True,
    )
    search_urls = _dedup_urls(acc.urls_commissioner_search)
    acc_search_claim = (
        f"In 2020, {person} chaired the ACC commissioner search process."
    )
    await evaluator.verify(
        claim=acc_search_claim,
        node=acc_search_leaf,
        sources=search_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify that in 2020 the person chaired the ACC commissioner search.",
            search_urls
        ),
    )


async def verify_career_transition(evaluator: Evaluator, parent, profile: PersonProfile) -> None:
    ct_node = evaluator.add_parallel(
        id="Career_Transition",
        desc="Career transition criteria are satisfied, with URL evidence for each.",
        parent=parent,
        critical=True,
    )

    person = _safe(profile.full_name)
    ct = profile.career_transition or CareerTransitionInfo()

    # Departure announced 2025
    dep_leaf = evaluator.add_leaf(
        id="Departure_Announced_2025_With_URL",
        desc="Person announced their departure from the chancellor/president position in 2025, supported by a URL reference.",
        parent=ct_node,
        critical=True,
    )
    dep_urls = _dedup_urls(ct.urls_departure)
    dep_claim = (
        f"In 2025, {person} announced departure from their chancellor/president position."
    )
    await evaluator.verify(
        claim=dep_claim,
        node=dep_leaf,
        sources=dep_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify that an official announcement or reputable source states the departure in 2025.",
            dep_urls
        ),
    )

    # Appointed 2026 president of a public Big Ten university in Michigan that is both R1 and AAU
    app_leaf = evaluator.add_leaf(
        id="Appointed_2026_Public_BigTen_MI_R1_AAU_With_URL",
        desc="Person was appointed president in 2026 of a public Big Ten university in Michigan that is both R1 and AAU, supported by a URL reference.",
        parent=ct_node,
        critical=True,
    )
    app_urls = _dedup_urls(
        ct.urls_appointment,
        ct.support_urls_public,
        ct.support_urls_big_ten,
        ct.support_urls_r1,
        ct.support_urls_aau,
        ct.support_urls_location,
    )
    app_claim = (
        f"In 2026, {person} was appointed president of {_safe(ct.new_university)}, "
        f"a public Big Ten university located in Michigan that holds both R1 classification and AAU membership."
    )
    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=app_urls,
        additional_instruction=_add_ins_for_urls(
            "Verify (1) appointment year 2026 and role as president, (2) the university is public, "
            "(3) the university is in the Big Ten, (4) the university is located in Michigan, "
            "(5) the university is R1, and (6) the university is an AAU member. Accept multiple URLs jointly supporting these points.",
            app_urls
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
    """
    Evaluate an answer for the academic leader identification and documentation task.
    """
    # Initialize evaluator
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

    # Extract structured profile from the answer
    profile: PersonProfile = await evaluator.extract(
        prompt=prompt_extract_person_profile(),
        template_class=PersonProfile,
        extraction_name="person_profile",
    )

    # Build the rubric tree per JSON specification
    person_node = evaluator.add_parallel(
        id="Person_Identification",
        desc="Identify the academic leader who satisfies all stated criteria and provide URL-backed documentation for each major milestone.",
        parent=root,
        critical=True,
    )

    # Provide full name (existence check)
    name_exists = bool(profile.full_name and str(profile.full_name).strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="Provide_Full_Name",
        desc="Answer provides the person’s full name.",
        parent=person_node,
        critical=True,
    )

    # Subtree verifications
    await verify_educational_background(evaluator, person_node, profile)
    await verify_early_career(evaluator, person_node, profile)
    await verify_admin_career(evaluator, person_node, profile)
    await verify_university_leadership(evaluator, person_node, profile)
    await verify_conference_leadership(evaluator, person_node, profile)
    await verify_career_transition(evaluator, person_node, profile)

    # Optionally record some computed helper info (not required)
    total_years = _tenure_years_sum(
        ((profile.dean_private_southeast.start_year if profile.dean_private_southeast else None),
         (profile.dean_private_southeast.end_year if profile.dean_private_southeast else None)),
        ((profile.dean_private_midwest.start_year if profile.dean_private_midwest else None),
         (profile.dean_private_midwest.end_year if profile.dean_private_midwest else None)),
    )
    evaluator.add_custom_info(
        info={
            "extracted_full_name": profile.full_name,
            "computed_total_dean_years": total_years,
        },
        info_type="computed_summary",
    )

    return evaluator.get_summary()