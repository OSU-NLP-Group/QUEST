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
TASK_ID = "higher_ed_leaders_march_2026"
TASK_DESCRIPTION = """
Identify three current higher education leaders (as of March 2026) who each meet the following distinct sets of criteria. Provide the full name of each leader and supporting URLs that verify each criterion.

Leader A must meet ALL of the following criteria:
1. Served as dean of a law school for approximately 6-8 years
2. As of March 2026, currently serving as chancellor of a Big Ten university
3. Was announced as the next president of an Ivy League university in January 2026
4. Earned a PhD from MIT (Massachusetts Institute of Technology)
5. Served as a visiting professor at an Ivy League law school at some point in their career

Leader B must meet ALL of the following criteria:
1. As of March 2026, currently serving as president of a public university in Ohio
2. Earned bachelor's, master's, and doctorate degrees all from the same university
3. Served as dean at a different institution before joining their current university
4. Has a contract that extends beyond 2028
5. Is a native of Ohio

Leader C must meet ALL of the following criteria:
1. As of March 2026, currently serving as president of a community college campus
2. Earned an associate degree from the same institution where they are currently serving as president
3. Previously served as a K-12 principal for more than 10 years
4. Previously served as chief of staff at a different higher education institution before their current presidency
5. Received a distinguished alumni award from their current employing institution

For each leader, provide:
- The leader's full name
- The name of their current institution
- Reference URLs that verify each of the required criteria
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class LeaderALawDeanship(BaseModel):
    law_school_name: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    approx_duration_years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LeaderABigTenChancellorship(BaseModel):
    institution: Optional[str] = None
    chancellorship_urls: List[str] = Field(default_factory=list)
    big_ten_urls: List[str] = Field(default_factory=list)


class LeaderAIvyLeagueAppointment(BaseModel):
    institution: Optional[str] = None
    announcement_month_year: Optional[str] = None  # e.g., "January 2026"
    urls: List[str] = Field(default_factory=list)
    ivy_league_urls: List[str] = Field(default_factory=list)


class LeaderAMitPhD(BaseModel):
    institution: Optional[str] = None  # Expect "MIT" or "Massachusetts Institute of Technology"
    urls: List[str] = Field(default_factory=list)


class LeaderAVisitingProf(BaseModel):
    law_school: Optional[str] = None  # e.g., "Harvard Law School"
    parent_university: Optional[str] = None  # e.g., "Harvard University"
    urls: List[str] = Field(default_factory=list)
    ivy_league_urls: List[str] = Field(default_factory=list)


class LeaderAInfo(BaseModel):
    name: Optional[str] = None
    current_institution: Optional[str] = None
    law_deanship: LeaderALawDeanship = Field(default_factory=LeaderALawDeanship)
    big_ten_chancellorship: LeaderABigTenChancellorship = Field(default_factory=LeaderABigTenChancellorship)
    ivy_appointment: LeaderAIvyLeagueAppointment = Field(default_factory=LeaderAIvyLeagueAppointment)
    mit_phd: LeaderAMitPhD = Field(default_factory=LeaderAMitPhD)
    visiting_prof: LeaderAVisitingProf = Field(default_factory=LeaderAVisitingProf)


class LeaderBOhioPresidency(BaseModel):
    university: Optional[str] = None
    presidency_urls: List[str] = Field(default_factory=list)
    ohio_public_urls: List[str] = Field(default_factory=list)


class LeaderBDegreesSameInstitution(BaseModel):
    institution: Optional[str] = None
    degrees_list: List[str] = Field(default_factory=list)  # bachelor's, master's, doctorate
    urls: List[str] = Field(default_factory=list)


class LeaderBPreviousDeanship(BaseModel):
    institution: Optional[str] = None
    role_title: Optional[str] = None  # e.g., "Dean"
    chronology_note: Optional[str] = None  # e.g., "before joining X University"
    urls: List[str] = Field(default_factory=list)


class LeaderBContract(BaseModel):
    end_year: Optional[str] = None  # e.g., "2030"
    end_date: Optional[str] = None  # e.g., "June 30, 2030"
    urls: List[str] = Field(default_factory=list)


class LeaderBOhioNative(BaseModel):
    birthplace: Optional[str] = None  # e.g., "Cleveland, Ohio"
    urls: List[str] = Field(default_factory=list)


class LeaderBInfo(BaseModel):
    name: Optional[str] = None
    current_institution: Optional[str] = None
    ohio_presidency: LeaderBOhioPresidency = Field(default_factory=LeaderBOhioPresidency)
    same_institution_degrees: LeaderBDegreesSameInstitution = Field(default_factory=LeaderBDegreesSameInstitution)
    previous_deanship: LeaderBPreviousDeanship = Field(default_factory=LeaderBPreviousDeanship)
    contract_extension: LeaderBContract = Field(default_factory=LeaderBContract)
    ohio_native: LeaderBOhioNative = Field(default_factory=LeaderBOhioNative)


class LeaderCCommunityCollegePresidency(BaseModel):
    college: Optional[str] = None  # The community college system or institution
    campus: Optional[str] = None   # Specific campus if applicable
    presidency_urls: List[str] = Field(default_factory=list)


class LeaderCAssociateDegreeSameInstitution(BaseModel):
    institution: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., "A.A." or "A.S."
    urls: List[str] = Field(default_factory=list)


class LeaderCK12Principal(BaseModel):
    role_desc: Optional[str] = None
    duration_years: Optional[str] = None  # e.g., "12 years"
    urls: List[str] = Field(default_factory=list)


class LeaderCChiefOfStaff(BaseModel):
    institution: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LeaderCDistinguishedAlumni(BaseModel):
    award_name: Optional[str] = None
    award_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LeaderCInfo(BaseModel):
    name: Optional[str] = None
    current_institution: Optional[str] = None
    community_college_presidency: LeaderCCommunityCollegePresidency = Field(default_factory=LeaderCCommunityCollegePresidency)
    alumni_status: LeaderCAssociateDegreeSameInstitution = Field(default_factory=LeaderCAssociateDegreeSameInstitution)
    k12_principal: LeaderCK12Principal = Field(default_factory=LeaderCK12Principal)
    chief_of_staff: LeaderCChiefOfStaff = Field(default_factory=LeaderCChiefOfStaff)
    distinguished_alumni: LeaderCDistinguishedAlumni = Field(default_factory=LeaderCDistinguishedAlumni)


class LeadersExtraction(BaseModel):
    leader_a: LeaderAInfo = Field(default_factory=LeaderAInfo)
    leader_b: LeaderBInfo = Field(default_factory=LeaderBInfo)
    leader_c: LeaderCInfo = Field(default_factory=LeaderCInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_leaders() -> str:
    return """
    Extract structured information for three leaders (Leader A, Leader B, Leader C) exactly as presented in the answer. 
    You must strictly extract only what the answer explicitly states, including the URLs it cites. 
    If an item is not present, set it to null (for strings) or [] (for lists). 
    
    General rules for URL extraction:
    - Extract only actual URLs explicitly present in the answer (plain or markdown links).
    - Do not fabricate or infer URLs. If no URL is provided for a criterion, return an empty list.

    For each leader, extract:

    Leader A:
    - name: full name
    - current_institution: current employing institution
    - law_deanship:
        - law_school_name
        - start_year (if present)
        - end_year (if present)
        - approx_duration_years (e.g., "7 years", if mentioned)
        - urls: list of URLs confirming deanship and/or duration
    - big_ten_chancellorship:
        - institution (the Big Ten university)
        - chancellorship_urls: URLs confirming current chancellorship
        - big_ten_urls: URLs confirming the institution is in the Big Ten
    - ivy_appointment:
        - institution (Ivy League university)
        - announcement_month_year (e.g., "January 2026" if stated)
        - urls: URLs confirming the announcement
        - ivy_league_urls: URLs confirming Ivy League status of the institution
    - mit_phd:
        - institution (should be "MIT" or "Massachusetts Institute of Technology" if stated)
        - urls: URLs confirming the PhD from MIT
    - visiting_prof:
        - law_school (e.g., "Harvard Law School")
        - parent_university (e.g., "Harvard University" if available)
        - urls: URLs confirming the visiting professorship
        - ivy_league_urls: URLs confirming the institution is Ivy League

    Leader B:
    - name
    - current_institution
    - ohio_presidency:
        - university
        - presidency_urls: URLs confirming the current presidency
        - ohio_public_urls: URLs confirming the university is located in Ohio and is a public institution
    - same_institution_degrees:
        - institution (the single institution that awarded bachelor's, master's, and doctorate)
        - degrees_list: list of degree levels mentioned (e.g., ["bachelor's", "master's", "doctorate"])
        - urls: URLs confirming all degrees are from the same institution
    - previous_deanship:
        - institution (where served as a dean previously)
        - role_title (e.g., "Dean of ...")
        - chronology_note (e.g., "before joining X University", if present)
        - urls: URLs confirming the previous deanship (and ideally chronology)
    - contract_extension:
        - end_year (if present)
        - end_date (if present)
        - urls: URLs confirming the contract runs beyond 2028
    - ohio_native:
        - birthplace (string mentioning Ohio if present)
        - urls: URLs confirming they are a native of Ohio

    Leader C:
    - name
    - current_institution
    - community_college_presidency:
        - college (community college or district)
        - campus (if specified)
        - presidency_urls: URLs confirming the current campus presidency
    - alumni_status:
        - institution (should match current institution if stated)
        - degree_type (e.g., "A.A.", "A.S.", or "associate degree")
        - urls: URLs confirming the associate degree from the same institution
    - k12_principal:
        - role_desc (e.g., "K-12 principal")
        - duration_years (e.g., "12 years", if present)
        - urls: URLs confirming service as K-12 principal and duration > 10 years
    - chief_of_staff:
        - institution (different higher ed institution)
        - urls: URLs confirming a chief of staff role at a different institution
    - distinguished_alumni:
        - award_name (e.g., "Distinguished Alumni Award")
        - award_year (if present)
        - urls: URLs confirming a distinguished alumni award from the current employing institution
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_leader_a(evaluator: Evaluator, parent) -> None:
    data = evaluator.find_node("root")  # just to avoid linter; not used
    # Pull extraction stored earlier
    # We rely on the recorded extraction; get the latest extraction result from evaluator._extraction_results
    # But we don't have public accessor; instead, pass the extracted object through closure.
    pass


# To avoid accessing private evaluator fields, we implement verification functions that accept the extracted objects.


async def verify_leader_a_with_data(evaluator: Evaluator, parent_node, a: LeaderAInfo) -> None:
    leader_node = evaluator.add_parallel(
        id="leader_a",
        desc="Identify the leader who meets all criteria for Leader A",
        parent=parent_node,
        critical=False
    )

    # 1) Law school deanship 6-8 years
    law_node = evaluator.add_parallel(
        id="leader_a_law_school_deanship",
        desc="Served as dean of a law school for approximately 6-8 years",
        parent=leader_node,
        critical=True
    )
    # 1.a Duration in range 6-8
    leaf = evaluator.add_leaf(
        id="leader_a_deanship_duration",
        desc="The deanship tenure was between 6 and 8 years",
        parent=law_node,
        critical=True
    )
    claim = (
        f"According to the cited sources, {a.name or 'the leader'} served as dean of "
        f"{a.law_deanship.law_school_name or 'a law school'} for approximately 6 to 8 years (inclusive). "
        f"If start/end years or dates are shown, compute the tenure and confirm it lies between 6 and 8 years; "
        f"if the text explicitly states 'about X years', accept values within 6–8."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.law_deanship.urls,
        additional_instruction="Focus on dean tenure length; small rounding (e.g., 7.5 years) still qualifies as 6–8."
    )
    # 1.b Verification of deanship and its duration presence
    leaf = evaluator.add_leaf(
        id="leader_a_deanship_verification",
        desc="Provide a URL confirming the law school deanship and its duration",
        parent=law_node,
        critical=True
    )
    claim = (
        f"The provided source(s) explicitly confirm that {a.name or 'the leader'} served as dean of "
        f"{a.law_deanship.law_school_name or 'a law school'} and include the service dates or duration."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.law_deanship.urls,
        additional_instruction="The page should clearly indicate the dean role and either specific years (start/end) or a stated duration."
    )

    # 2) Big Ten chancellorship (current as of March 2026)
    bigten_node = evaluator.add_parallel(
        id="leader_a_big_ten_chancellorship",
        desc="As of March 2026, currently serving as chancellor of a Big Ten university",
        parent=leader_node,
        critical=True
    )
    # 2.a Institution is Big Ten
    leaf = evaluator.add_leaf(
        id="leader_a_chancellor_institution_type",
        desc="The institution is confirmed as a Big Ten university",
        parent=bigten_node,
        critical=True
    )
    inst = a.big_ten_chancellorship.institution or a.current_institution or "the institution"
    claim = f"{inst} is a member of the Big Ten Conference (i.e., a Big Ten university)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_combine_sources(a.big_ten_chancellorship.big_ten_urls, a.big_ten_chancellorship.chancellorship_urls),
        additional_instruction="Accept official Big Ten listings or credible sources that list Big Ten member universities."
    )
    # 2.b Actively serving as of March 2026
    leaf = evaluator.add_leaf(
        id="leader_a_chancellor_status",
        desc="The individual is actively serving in this role as of March 2026",
        parent=bigten_node,
        critical=True
    )
    claim = f"As of March 2026, {a.name or 'the leader'} is serving as chancellor of {inst}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.big_ten_chancellorship.chancellorship_urls,
        additional_instruction="Prefer official institution pages or credible news/biography pages indicating current status around March 2026."
    )
    # 2.c Provide URL confirming current chancellorship
    leaf = evaluator.add_leaf(
        id="leader_a_chancellorship_verification",
        desc="Provide a URL confirming the current chancellorship at a Big Ten university",
        parent=bigten_node,
        critical=True
    )
    claim = f"The provided source confirms that {a.name or 'the leader'} holds the title 'Chancellor' at {inst}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.big_ten_chancellorship.chancellorship_urls,
        additional_instruction="The page must clearly show the chancellor title for the named individual."
    )

    # 3) Announced as next president of an Ivy League university in Jan 2026
    ivy_app_node = evaluator.add_parallel(
        id="leader_a_ivy_league_appointment",
        desc="Was announced as the next president of an Ivy League university in January 2026",
        parent=leader_node,
        critical=True
    )
    ivy_inst = a.ivy_appointment.institution or "the Ivy League university"
    # 3.a Announcement timing in Jan 2026
    leaf = evaluator.add_leaf(
        id="leader_a_announcement_timing",
        desc="The announcement occurred in January 2026",
        parent=ivy_app_node,
        critical=True
    )
    claim = (
        f"In January 2026, {a.name or 'the leader'} was announced as the next president of {ivy_inst}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.ivy_appointment.urls,
        additional_instruction="Look for publication date or explicit mention of 'January 2026'."
    )
    # 3.b Institution is Ivy League
    leaf = evaluator.add_leaf(
        id="leader_a_ivy_league_status",
        desc="The institution is confirmed as an Ivy League university",
        parent=ivy_app_node,
        critical=True
    )
    claim = f"{ivy_inst} is an Ivy League university."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_combine_sources(a.ivy_appointment.ivy_league_urls, a.ivy_appointment.urls),
        additional_instruction="Accept official Ivy League listings or widely recognized sources that list Ivy League members."
    )
    # 3.c URL confirming the Jan 2026 appointment announcement
    leaf = evaluator.add_leaf(
        id="leader_a_appointment_verification",
        desc="Provide a URL confirming the January 2026 announcement of the Ivy League presidency",
        parent=ivy_app_node,
        critical=True
    )
    claim = f"The source confirms that {a.name or 'the leader'} was announced in January 2026 as the next president of {ivy_inst}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.ivy_appointment.urls,
        additional_instruction="The item must clearly be an announcement naming the person as 'next president' with a January 2026 date."
    )

    # 4) MIT PhD
    mit_node = evaluator.add_parallel(
        id="leader_a_mit_phd",
        desc="Earned a PhD from MIT",
        parent=leader_node,
        critical=True
    )
    # 4.a PhD is from MIT
    leaf = evaluator.add_leaf(
        id="leader_a_phd_institution",
        desc="The PhD was earned from MIT (Massachusetts Institute of Technology)",
        parent=mit_node,
        critical=True
    )
    claim = f"{a.name or 'the leader'} earned a PhD from the Massachusetts Institute of Technology (MIT)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.mit_phd.urls,
        additional_instruction="The page should explicitly state that the person earned a PhD from MIT."
    )
    # 4.b URL confirming MIT PhD
    leaf = evaluator.add_leaf(
        id="leader_a_phd_verification",
        desc="Provide a URL confirming the MIT PhD",
        parent=mit_node,
        critical=True
    )
    claim = f"The provided source confirms that {a.name or 'the leader'} holds a PhD from MIT."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.mit_phd.urls,
        additional_instruction="Accept official bios, CVs, or credible news/press pages listing education."
    )

    # 5) Visiting professor at an Ivy League law school
    visit_node = evaluator.add_parallel(
        id="leader_a_visiting_professor",
        desc="Served as a visiting professor at an Ivy League law school",
        parent=leader_node,
        critical=True
    )
    # 5.a Law school is part of Ivy League institution
    leaf = evaluator.add_leaf(
        id="leader_a_visiting_ivy_status",
        desc="The law school is confirmed as part of an Ivy League institution",
        parent=visit_node,
        critical=True
    )
    law_school = a.visiting_prof.law_school or "the law school"
    parent_uni = a.visiting_prof.parent_university or "the parent university"
    claim = f"{law_school} is part of an Ivy League institution (e.g., {parent_uni})."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_combine_sources(a.visiting_prof.ivy_league_urls, a.visiting_prof.urls),
        additional_instruction="It's sufficient if the law school's parent university is an Ivy League institution."
    )
    # 5.b URL confirming visiting professorship
    leaf = evaluator.add_leaf(
        id="leader_a_visiting_verification",
        desc="Provide a URL confirming the visiting professorship at an Ivy League law school",
        parent=visit_node,
        critical=True
    )
    claim = f"The source confirms that {a.name or 'the leader'} served as a visiting professor at {law_school}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=a.visiting_prof.urls,
        additional_instruction="Look for titles such as 'Visiting Professor', 'Visiting Scholar', or similar at the law school."
    )


async def verify_leader_b_with_data(evaluator: Evaluator, parent_node, b: LeaderBInfo) -> None:
    leader_node = evaluator.add_parallel(
        id="leader_b",
        desc="Identify the leader who meets all criteria for Leader B",
        parent=parent_node,
        critical=False
    )

    # 1) Ohio public university presidency (current as of March 2026)
    ohio_pres_node = evaluator.add_parallel(
        id="leader_b_ohio_presidency",
        desc="As of March 2026, currently serving as president of a public university in Ohio",
        parent=leader_node,
        critical=True
    )
    uni = b.ohio_presidency.university or b.current_institution or "the university"
    # 1.a Holds a university presidency as of March 2026
    leaf = evaluator.add_leaf(
        id="leader_b_current_position",
        desc="The individual holds a university presidency as of March 2026",
        parent=ohio_pres_node,
        critical=True
    )
    claim = f"As of March 2026, {b.name or 'the leader'} is the president of {uni}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.ohio_presidency.presidency_urls,
        additional_instruction="Prefer official university leadership pages or recent press releases around March 2026."
    )
    # 1.b University is in Ohio
    leaf = evaluator.add_leaf(
        id="leader_b_ohio_location",
        desc="The university is located in Ohio",
        parent=ohio_pres_node,
        critical=True
    )
    claim = f"{uni} is located in the U.S. state of Ohio."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_combine_sources(b.ohio_presidency.ohio_public_urls, b.ohio_presidency.presidency_urls),
        additional_instruction="Any credible source that lists the university's location is acceptable."
    )
    # 1.c Public institution
    leaf = evaluator.add_leaf(
        id="leader_b_public_institution",
        desc="The university is a public institution",
        parent=ohio_pres_node,
        critical=True
    )
    claim = f"{uni} is a public university (not a private institution)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_combine_sources(b.ohio_presidency.ohio_public_urls, b.ohio_presidency.presidency_urls),
        additional_instruction="Accept official classifications or credible references specifying 'public'."
    )
    # 1.d URL confirming current Ohio public university presidency
    leaf = evaluator.add_leaf(
        id="leader_b_presidency_verification",
        desc="Provide a URL confirming the current Ohio public university presidency",
        parent=ohio_pres_node,
        critical=True
    )
    claim = f"The source confirms that {b.name or 'the leader'} is the president of {uni} (an Ohio public university)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_combine_sources(b.ohio_presidency.presidency_urls, b.ohio_presidency.ohio_public_urls),
        additional_instruction="Prefer the official university site; otherwise credible news is acceptable."
    )

    # 2) Degrees all from the same institution (bachelor's, master's, doctorate)
    deg_node = evaluator.add_parallel(
        id="leader_b_same_institution_degrees",
        desc="Earned bachelor's, master's, and doctorate degrees all from the same university",
        parent=leader_node,
        critical=True
    )
    # 2.a All three levels from one institution
    leaf = evaluator.add_leaf(
        id="leader_b_three_degrees",
        desc="Earned all three degree levels (bachelor's, master's, doctorate) from one institution",
        parent=deg_node,
        critical=True
    )
    deg_inst = b.same_institution_degrees.institution or "the same institution"
    claim = (
        f"{b.name or 'the leader'} earned a bachelor's, a master's, and a doctorate degree all from {deg_inst}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.same_institution_degrees.urls,
        additional_instruction="The page should list all three degrees and show they were awarded by the same institution."
    )
    # 2.b URL confirming same-institution degrees
    leaf = evaluator.add_leaf(
        id="leader_b_degrees_verification",
        desc="Provide a URL confirming all three degrees were from the same institution",
        parent=deg_node,
        critical=True
    )
    claim = f"The source confirms that all three degrees were awarded by {deg_inst}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.same_institution_degrees.urls,
        additional_instruction="Official bios or CVs are best; credible press is acceptable."
    )

    # 3) Served as dean at a different institution before current university
    prev_dean_node = evaluator.add_parallel(
        id="leader_b_previous_deanship",
        desc="Served as dean at a different institution before joining current university",
        parent=leader_node,
        critical=True
    )
    prev_inst = b.previous_deanship.institution or "another institution"
    # 3.a Held a dean position
    leaf = evaluator.add_leaf(
        id="leader_b_dean_role",
        desc="Held a dean position at another institution",
        parent=prev_dean_node,
        critical=True
    )
    claim = f"{b.name or 'the leader'} served in a dean role at {prev_inst}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.previous_deanship.urls,
        additional_instruction="Look for titles including 'Dean' at a prior institution."
    )
    # 3.b Occurred before the current presidency
    leaf = evaluator.add_leaf(
        id="leader_b_prior_to_current",
        desc="This deanship occurred before assuming the current presidency",
        parent=prev_dean_node,
        critical=True
    )
    claim = (
        f"The deanship at {prev_inst} occurred prior to {b.name or 'the leader'} assuming the presidency at "
        f"{b.current_institution or 'the current university'}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.previous_deanship.urls,
        additional_instruction="The source should imply or state chronology (e.g., 'before joining X, they served as dean at Y')."
    )
    # 3.c URL confirming previous deanship
    leaf = evaluator.add_leaf(
        id="leader_b_deanship_verification",
        desc="Provide a URL confirming the previous deanship at a different institution",
        parent=prev_dean_node,
        critical=True
    )
    claim = f"The source confirms a previous deanship at {prev_inst}, which is a different institution than the current one."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.previous_deanship.urls,
        additional_instruction="The page should clearly tie the person to a dean role at another institution."
    )

    # 4) Contract extends beyond 2028
    contract_node = evaluator.add_parallel(
        id="leader_b_contract_extension",
        desc="Has a contract extending beyond 2028",
        parent=leader_node,
        critical=True
    )
    # 4.a Contract end in 2029 or later
    leaf = evaluator.add_leaf(
        id="leader_b_contract_year",
        desc="The contract end date is in 2029 or later",
        parent=contract_node,
        critical=True
    )
    claim = (
        f"{b.name or 'the leader'}'s employment contract as president of {b.current_institution or 'the university'} "
        f"runs through at least 2029 (i.e., ends in 2029 or later)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.contract_extension.urls,
        additional_instruction="Confirm the explicit contract end date/year; renewed/extended terms count if they go beyond 2028."
    )
    # 4.b URL confirming contract beyond 2028
    leaf = evaluator.add_leaf(
        id="leader_b_contract_verification",
        desc="Provide a URL confirming the contract extends beyond 2028",
        parent=contract_node,
        critical=True
    )
    claim = f"The source confirms that the president's contract extends beyond 2028."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.contract_extension.urls,
        additional_instruction="Look for board approvals, contract announcements, or official statements showing term end after 2028."
    )

    # 5) Native of Ohio
    native_node = evaluator.add_parallel(
        id="leader_b_ohio_native",
        desc="Is a native of Ohio",
        parent=leader_node,
        critical=True
    )
    # 5.a Born or raised in Ohio
    leaf = evaluator.add_leaf(
        id="leader_b_birthplace",
        desc="Was born or raised in Ohio",
        parent=native_node,
        critical=True
    )
    claim = f"{b.name or 'the leader'} is a native of Ohio (born and/or raised in Ohio)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.ohio_native.urls,
        additional_instruction="Any credible bio statement indicating Ohio origin suffices."
    )
    # 5.b URL confirming Ohio as place of origin
    leaf = evaluator.add_leaf(
        id="leader_b_native_verification",
        desc="Provide a URL confirming Ohio as place of origin",
        parent=native_node,
        critical=True
    )
    claim = f"The source confirms that {b.name or 'the leader'} is from Ohio."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=b.ohio_native.urls,
        additional_instruction="Look for birthplace or 'Ohio native' phrasing."
    )


async def verify_leader_c_with_data(evaluator: Evaluator, parent_node, c: LeaderCInfo) -> None:
    leader_node = evaluator.add_parallel(
        id="leader_c",
        desc="Identify the leader who meets all criteria for Leader C",
        parent=parent_node,
        critical=False
    )

    # 1) Community college campus president (current as of March 2026)
    cc_pres_node = evaluator.add_parallel(
        id="leader_c_community_college_presidency",
        desc="As of March 2026, currently serving as president of a community college campus",
        parent=leader_node,
        critical=True
    )
    # 1.a Holds campus presidency at a community college
    leaf = evaluator.add_leaf(
        id="leader_c_current_role",
        desc="Holds a campus presidency at a community college",
        parent=cc_pres_node,
        critical=True
    )
    claim = (
        f"{c.name or 'the leader'} holds a presidency at a community college campus "
        f"({c.community_college_presidency.college or 'the community college'}"
        f"{' - ' + c.community_college_presidency.campus if c.community_college_presidency.campus else ''})."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.community_college_presidency.presidency_urls,
        additional_instruction="The page should clearly indicate a campus presidency at a community college."
    )
    # 1.b Active as of March 2026
    leaf = evaluator.add_leaf(
        id="leader_c_position_timing",
        desc="This position is active as of March 2026",
        parent=cc_pres_node,
        critical=True
    )
    claim = f"As of March 2026, {c.name or 'the leader'} is actively serving as president of the community college campus."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.community_college_presidency.presidency_urls,
        additional_instruction="Prefer recently updated official pages; confirm 'current' status around March 2026."
    )
    # 1.c URL confirming current community college campus presidency
    leaf = evaluator.add_leaf(
        id="leader_c_presidency_verification",
        desc="Provide a URL confirming the current community college campus presidency",
        parent=cc_pres_node,
        critical=True
    )
    claim = f"The source confirms {c.name or 'the leader'} is the current president of the named community college campus."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.community_college_presidency.presidency_urls,
        additional_instruction="Official institution pages preferred."
    )

    # 2) Associate degree from the same institution (current employer)
    alum_node = evaluator.add_parallel(
        id="leader_c_alumni_status",
        desc="Earned an associate degree from the same institution where currently serving as president",
        parent=leader_node,
        critical=True
    )
    # 2.a Earned an associate degree
    leaf = evaluator.add_leaf(
        id="leader_c_associate_degree",
        desc="Earned an associate degree (A.A. or A.S.)",
        parent=alum_node,
        critical=True
    )
    claim = f"{c.name or 'the leader'} earned an associate degree (e.g., AA or AS)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.alumni_status.urls,
        additional_instruction="The page should explicitly reference an associate degree."
    )
    # 2.b Degree from same institution as current presidency
    leaf = evaluator.add_leaf(
        id="leader_c_same_institution",
        desc="The degree was from the current employing institution",
        parent=alum_node,
        critical=True
    )
    claim = (
        f"The associate degree was awarded by {c.alumni_status.institution or 'the same institution'}, "
        f"which is the same institution where {c.name or 'the leader'} now serves as president."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.alumni_status.urls,
        additional_instruction="It should indicate the degree institution matches the leader’s current institution."
    )
    # 2.c URL confirming associate degree from current institution
    leaf = evaluator.add_leaf(
        id="leader_c_alumni_verification",
        desc="Provide a URL confirming the associate degree from the current institution",
        parent=alum_node,
        critical=True
    )
    claim = "The source confirms the associate degree was earned from the current employing institution."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.alumni_status.urls,
        additional_instruction="Official bios, alumni profiles, or institutional honors pages are acceptable."
    )

    # 3) K-12 principal > 10 years
    k12_node = evaluator.add_parallel(
        id="leader_c_k12_principal",
        desc="Previously served as a K-12 principal for more than 10 years",
        parent=leader_node,
        critical=True
    )
    # 3.a Served as principal in K-12
    leaf = evaluator.add_leaf(
        id="leader_c_principal_role",
        desc="Served as a principal in K-12 education",
        parent=k12_node,
        critical=True
    )
    claim = f"{c.name or 'the leader'} served as a principal in K-12 education."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.k12_principal.urls,
        additional_instruction="Look for 'principal' title in K-12 settings."
    )
    # 3.b Tenure exceeded 10 years
    leaf = evaluator.add_leaf(
        id="leader_c_principal_duration",
        desc="The principal tenure exceeded 10 years",
        parent=k12_node,
        critical=True
    )
    claim = f"The duration of {c.name or 'the leader'}'s K-12 principal service exceeded 10 years."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.k12_principal.urls,
        additional_instruction="If a range or multiple schools are listed, sum the years where clearly indicated."
    )
    # 3.c URL confirming >10 years as principal
    leaf = evaluator.add_leaf(
        id="leader_c_principal_verification",
        desc="Provide a URL confirming K-12 principal service for more than 10 years",
        parent=k12_node,
        critical=True
    )
    claim = "The source confirms more than a decade of service as a K-12 principal."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.k12_principal.urls,
        additional_instruction="Look for text indicating 11+ years, 'more than 10 years', or specific dates totaling >10."
    )

    # 4) Chief of staff at a different higher ed institution (before current presidency)
    cos_node = evaluator.add_parallel(
        id="leader_c_chief_of_staff",
        desc="Previously served as chief of staff at a different higher education institution",
        parent=leader_node,
        critical=True
    )
    # 4.a Held a chief of staff position in higher education
    leaf = evaluator.add_leaf(
        id="leader_c_cos_position",
        desc="Held a chief of staff position in higher education",
        parent=cos_node,
        critical=True
    )
    claim = f"{c.name or 'the leader'} held a 'chief of staff' role at a higher education institution."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.chief_of_staff.urls,
        additional_instruction="Titles like 'Chief of Staff to the President' at a university/college qualify."
    )
    # 4.b At a different institution than current
    leaf = evaluator.add_leaf(
        id="leader_c_cos_different_institution",
        desc="This position was at a different institution from the current one",
        parent=cos_node,
        critical=True
    )
    claim = (
        f"The 'chief of staff' position was at {c.chief_of_staff.institution or 'another institution'}, "
        f"which is a different institution than {c.current_institution or 'the current employer'}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.chief_of_staff.urls,
        additional_instruction="The source should make the institution explicit; verify it differs from the current institution."
    )
    # 4.c URL confirming the chief of staff position
    leaf = evaluator.add_leaf(
        id="leader_c_cos_verification",
        desc="Provide a URL confirming the chief of staff position at another higher education institution",
        parent=cos_node,
        critical=True
    )
    claim = "The source confirms the individual's prior chief of staff role at a different higher education institution."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.chief_of_staff.urls,
        additional_instruction="Official bios or HR/announcement pages preferred."
    )

    # 5) Distinguished alumni award from current employing institution
    award_node = evaluator.add_parallel(
        id="leader_c_distinguished_alumni",
        desc="Received a distinguished alumni award from the current employing institution",
        parent=leader_node,
        critical=True
    )
    # 5.a Received distinguished/notable alumni recognition
    leaf = evaluator.add_leaf(
        id="leader_c_alumni_award",
        desc="Received a distinguished or notable alumni recognition",
        parent=award_node,
        critical=True
    )
    claim = f"{c.name or 'the leader'} received a distinguished (or notable) alumni award."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.distinguished_alumni.urls,
        additional_instruction="Variations like 'Outstanding Alumnus/Alumna' are acceptable."
    )
    # 5.b From current employing institution
    leaf = evaluator.add_leaf(
        id="leader_c_award_from_current",
        desc="The award was from the institution where currently employed",
        parent=award_node,
        critical=True
    )
    claim = (
        f"The distinguished alumni award was conferred by {c.current_institution or 'the current employing institution'}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.distinguished_alumni.urls,
        additional_instruction="The page should identify the awarding institution and it should match the current employer."
    )
    # 5.c URL confirming the award from current institution
    leaf = evaluator.add_leaf(
        id="leader_c_award_verification",
        desc="Provide a URL confirming the distinguished alumni award from the current institution",
        parent=award_node,
        critical=True
    )
    claim = "The source confirms a distinguished alumni award from the same institution where the leader is currently employed."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=c.distinguished_alumni.urls,
        additional_instruction="Official alumni relations or institutional news pages are acceptable."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the higher education leaders task.
    Note: The provided rubric marks the root as critical, but the framework enforces that
    all children of a critical node must also be critical. Since leader-level nodes are
    non-critical (to allow partial credit across leaders), we set the root as non-critical.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Leaders evaluated independently
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

    # 1) Extract structured info
    extracted: LeadersExtraction = await evaluator.extract(
        prompt=prompt_extract_leaders(),
        template_class=LeadersExtraction,
        extraction_name="leaders_extraction",
    )

    # 2) Build verification tree following rubric
    # Leader A subtree
    await verify_leader_a_with_data(evaluator, root, extracted.leader_a)

    # Leader B subtree
    await verify_leader_b_with_data(evaluator, root, extracted.leader_b)

    # Leader C subtree
    await verify_leader_c_with_data(evaluator, root, extracted.leader_c)

    # 3) Return summary
    return evaluator.get_summary()