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
TASK_ID = "edu_ath_leadership_4roles_2026"
TASK_DESCRIPTION = """I am researching career paths in educational and athletic leadership in the United States. Please help me identify four specific individuals currently holding or recently transitioning between senior leadership positions that meet the following criteria:

Position 1: A current superintendent (as of February 2026) of a school district that serves more than 60,000 students, and who holds a doctoral degree (Ed.D. or Ph.D.). Please provide the superintendent's full name, the school district name, confirmation that the district serves over 60,000 students, and confirmation of their doctoral degree.

Position 2: A superintendent of a school district in Texas who announced their retirement in 2025 or later. Please provide the superintendent's full name, the Texas school district name, and confirmation that the retirement was announced in 2025 or later.

Position 3: A current athletic director (as of February 2026) at an Ivy League university who is female. Please provide the athletic director's full name, the Ivy League university name, confirmation that the university is a member of the Ivy League, and confirmation that the athletic director is female.

Position 4: A special teams coordinator who moved from the University of Iowa to Michigan State University, with this move being announced in December 2025. Please provide the coach's full name, confirmation of their previous position at Iowa, confirmation of their current position at Michigan State, and confirmation that the move was announced in December 2025.

For each position, please provide supporting URL references that confirm the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Position1(BaseModel):
    full_name: Optional[str] = None
    district_name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)        # confirms identity/current position
    enrollment_urls: List[str] = Field(default_factory=list)      # confirms enrollment > 60k
    doctoral_degree: Optional[str] = None                         # e.g., "Ed.D.", "Ph.D.", "Doctor of Education"
    degree_urls: List[str] = Field(default_factory=list)          # confirms doctoral degree


class Position2(BaseModel):
    full_name: Optional[str] = None
    district_name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)        # confirms identity
    location_urls: List[str] = Field(default_factory=list)        # confirms Texas
    retirement_urls: List[str] = Field(default_factory=list)      # confirms retirement announcement timing
    announcement_date: Optional[str] = None                       # e.g., "January 2026" (if provided)


class Position3(BaseModel):
    full_name: Optional[str] = None
    university_name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)        # confirms identity/current AD position
    institution_urls: List[str] = Field(default_factory=list)     # confirms Ivy League membership
    gender: Optional[str] = None                                  # e.g., "female"
    gender_urls: List[str] = Field(default_factory=list)          # confirms gender (pronouns/bio, etc.)


class Position4(BaseModel):
    full_name: Optional[str] = None
    msu_current_title: Optional[str] = None                       # should include "special teams coordinator"
    msu_urls: List[str] = Field(default_factory=list)             # confirms current position at MSU
    iowa_urls: List[str] = Field(default_factory=list)            # confirms previous position at Iowa
    announcement_urls: List[str] = Field(default_factory=list)    # confirms move announcement timing (Dec 2025)


class AllPositionsExtraction(BaseModel):
    position1: Optional[Position1] = None
    position2: Optional[Position2] = None
    position3: Optional[Position3] = None
    position4: Optional[Position4] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract structured information for four positions from the answer. Return a JSON object with keys:
- position1
- position2
- position3
- position4

Each position should be an object with the following fields (return null for any missing string field and [] for any missing URL list):

position1 (current superintendent as of Feb 2026; district >60,000 students; has doctoral degree):
- full_name: string | null
- district_name: string | null
- identity_urls: array of URLs confirming the superintendent identity and current status (as of around Feb 2026)
- enrollment_urls: array of URLs confirming the district serves >60,000 students
- doctoral_degree: string | null (e.g., "Ed.D.", "Ph.D.", "Doctor of Education", "Doctor of Philosophy")
- degree_urls: array of URLs confirming the doctoral degree

position2 (Texas superintendent who announced retirement in 2025 or later):
- full_name: string | null
- district_name: string | null
- identity_urls: array of URLs confirming the superintendent identity (may be the district site or news)
- location_urls: array of URLs confirming the district is located in Texas
- retirement_urls: array of URLs confirming that retirement was announced in 2025 or later (include credible news or official statements)
- announcement_date: string | null (e.g., "January 2026") if mentioned

position3 (current female athletic director at an Ivy League university as of Feb 2026):
- full_name: string | null
- university_name: string | null
- identity_urls: array of URLs confirming identity and current AD role (as of around Feb 2026)
- institution_urls: array of URLs confirming the university is in the Ivy League (e.g., Ivy League official site, Wikipedia)
- gender: string | null (e.g., "female", "woman", "she/her")
- gender_urls: array of URLs confirming that the AD is female (could be the same bio page if pronouns are present)

position4 (special teams coordinator moved from University of Iowa to Michigan State; move announced Dec 2025):
- full_name: string | null
- msu_current_title: string | null (title at Michigan State; should include "special teams coordinator" or equivalent)
- msu_urls: array of URLs confirming current MSU role (e.g., press release, bio page)
- iowa_urls: array of URLs confirming previous Iowa employment and role
- announcement_urls: array of URLs confirming the move announcement in December 2025

GENERAL RULES:
- Extract exactly what appears in the answer.
- Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
- If a field is not present, set it to null (for strings) or [] (for arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _combined_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not _nonempty(u):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _missing_sources_instruction(base_instruction: str, urls: List[str]) -> str:
    if urls and len(urls) > 0:
        return base_instruction
    # Strongly instruct the judge to fail when no URLs are provided for web-grounded facts
    return base_instruction + "\nIMPORTANT: No URL evidence is provided in the answer for this claim. Treat the claim as unsupported and judge it Incorrect."


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_position_1(evaluator: Evaluator, parent_node, p: Optional[Position1]) -> None:
    pos_node = evaluator.add_parallel(
        id="Position_1",
        desc="Identify a current superintendent (as of February 2026) of a school district serving more than 60,000 students who holds a doctoral degree",
        parent=parent_node,
        critical=False
    )

    # Identity group
    identity_node = evaluator.add_parallel(
        id="Position_1_Identity",
        desc="Verify the identity of the superintendent and school district",
        parent=pos_node,
        critical=True
    )

    # Full name exists (critical custom)
    evaluator.add_custom_node(
        result=_nonempty(p.full_name) if p else False,
        id="Position_1_Full_Name",
        desc="Provide the superintendent's full name",
        parent=identity_node,
        critical=True
    )

    # District name exists (critical custom)
    evaluator.add_custom_node(
        result=_nonempty(p.district_name) if p else False,
        id="Position_1_District_Name",
        desc="Provide the school district name",
        parent=identity_node,
        critical=True
    )

    # Current status leaf (as of Feb 2026)
    current_status_leaf = evaluator.add_leaf(
        id="Position_1_Current_Status",
        desc="Verify the person is currently serving as superintendent as of February 2026",
        parent=identity_node,
        critical=True
    )

    # Identity URL verification leaf
    identity_url_leaf = evaluator.add_leaf(
        id="Position_1_Identity_URL",
        desc="URL reference confirming the superintendent's identity and current position",
        parent=identity_node,
        critical=True
    )

    # District size group
    size_node = evaluator.add_parallel(
        id="Position_1_District_Size",
        desc="Verify the school district serves more than 60,000 students",
        parent=pos_node,
        critical=True
    )

    # Student count verification leaf
    student_count_leaf = evaluator.add_leaf(
        id="Position_1_Student_Count",
        desc="Confirm the district serves more than 60,000 students",
        parent=size_node,
        critical=True
    )

    # District URL existence (critical)
    evaluator.add_custom_node(
        result=bool(p and p.enrollment_urls and len(p.enrollment_urls) > 0),
        id="Position_1_District_URL",
        desc="URL reference confirming the district enrollment size",
        parent=size_node,
        critical=True
    )

    # Doctoral degree group
    degree_node = evaluator.add_parallel(
        id="Position_1_Doctoral_Degree",
        desc="Verify the superintendent holds a doctoral degree",
        parent=pos_node,
        critical=True
    )

    # Degree confirmation verification leaf
    degree_confirm_leaf = evaluator.add_leaf(
        id="Position_1_Degree_Confirmation",
        desc="Confirm the superintendent holds an Ed.D. or Ph.D. degree",
        parent=degree_node,
        critical=True
    )

    # Education URL existence (critical)
    evaluator.add_custom_node(
        result=bool(p and p.degree_urls and len(p.degree_urls) > 0),
        id="Position_1_Education_URL",
        desc="URL reference confirming the doctoral degree",
        parent=degree_node,
        critical=True
    )

    # Prepare claims and sources
    if not p:
        p = Position1()  # fallback safe

    claims: List[tuple[str, List[str], Any, str]] = []

    # Current status claim
    cs_claim = f"As of February 2026, {p.full_name or ''} is serving as superintendent of {p.district_name or ''}."
    cs_urls = _combined_urls(p.identity_urls)
    cs_ins = _missing_sources_instruction(
        "Verify recency and role from the cited pages. Accept if the page indicates current service around late 2025 to early 2026 (e.g., 'present', 'current', an updated bio, or a dated press release) and explicitly shows the superintendent role.",
        cs_urls
    )
    claims.append((cs_claim, cs_urls, current_status_leaf, cs_ins))

    # Identity URL claim
    id_claim = f"{p.full_name or ''} is the superintendent of {p.district_name or ''}."
    id_urls = _combined_urls(p.identity_urls)
    id_ins = _missing_sources_instruction(
        "At least one cited page should explicitly state that this person is the superintendent of the named district.",
        id_urls
    )
    claims.append((id_claim, id_urls, identity_url_leaf, id_ins))

    # Student count claim
    sc_claim = f"The {p.district_name or 'district'} school district serves more than 60,000 students."
    sc_urls = _combined_urls(p.enrollment_urls)
    sc_ins = _missing_sources_instruction(
        "Confirm the enrollment exceeds 60,000. Accept approximate phrasing such as 'over 60,000', 'more than 60k', or explicit numbers > 60,000.",
        sc_urls
    )
    claims.append((sc_claim, sc_urls, student_count_leaf, sc_ins))

    # Doctoral degree claim
    dd_claim = f"{p.full_name or ''} holds a doctoral degree (Ed.D. or Ph.D.)."
    dd_urls = _combined_urls(p.degree_urls, p.identity_urls)
    dd_ins = _missing_sources_instruction(
        "Confirm a doctoral credential, accepting 'Ed.D.', 'Doctor of Education', 'Ph.D.', or 'Doctor of Philosophy'.",
        dd_urls
    )
    claims.append((dd_claim, dd_urls, degree_confirm_leaf, dd_ins))

    await evaluator.batch_verify(claims)


async def verify_position_2(evaluator: Evaluator, parent_node, p: Optional[Position2]) -> None:
    pos_node = evaluator.add_parallel(
        id="Position_2",
        desc="Identify a superintendent of a school district in Texas who announced retirement in 2025 or later",
        parent=parent_node,
        critical=False
    )

    # Identity group
    identity_node = evaluator.add_parallel(
        id="Position_2_Identity",
        desc="Verify the identity of the superintendent and school district",
        parent=pos_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(p.full_name) if p else False,
        id="Position_2_Full_Name",
        desc="Provide the superintendent's full name",
        parent=identity_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(p.district_name) if p else False,
        id="Position_2_District_Name",
        desc="Provide the school district name",
        parent=identity_node,
        critical=True
    )

    identity_url_leaf = evaluator.add_leaf(
        id="Position_2_Identity_URL",
        desc="URL reference confirming the superintendent's identity",
        parent=identity_node,
        critical=True
    )

    # Texas location group
    location_node = evaluator.add_parallel(
        id="Position_2_Texas_Location",
        desc="Verify the school district is located in Texas",
        parent=pos_node,
        critical=True
    )

    state_confirm_leaf = evaluator.add_leaf(
        id="Position_2_State_Confirmation",
        desc="Confirm the district is located in Texas",
        parent=location_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and p.location_urls and len(p.location_urls) > 0),
        id="Position_2_Location_URL",
        desc="URL reference confirming Texas location",
        parent=location_node,
        critical=True
    )

    # Retirement announcement group
    retirement_node = evaluator.add_parallel(
        id="Position_2_Retirement_Announcement",
        desc="Verify the retirement announcement timing",
        parent=pos_node,
        critical=True
    )

    announce_date_leaf = evaluator.add_leaf(
        id="Position_2_Announcement_Date",
        desc="Confirm retirement was announced in 2025 or later",
        parent=retirement_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and p.retirement_urls and len(p.retirement_urls) > 0),
        id="Position_2_Retirement_URL",
        desc="URL reference confirming the retirement announcement",
        parent=retirement_node,
        critical=True
    )

    if not p:
        p = Position2()

    claims: List[tuple[str, List[str], Any, str]] = []

    # Identity claim
    id_claim = f"{p.full_name or ''} is (or was) the superintendent of {p.district_name or ''}."
    id_urls = _combined_urls(p.identity_urls)
    id_ins = _missing_sources_instruction(
        "At least one cited page should clearly associate the person with the superintendent role for the named district.",
        id_urls
    )
    claims.append((id_claim, id_urls, identity_url_leaf, id_ins))

    # Texas location claim
    loc_claim = f"The {p.district_name or 'district'} school district is located in Texas."
    loc_urls = _combined_urls(p.location_urls)
    loc_ins = _missing_sources_instruction(
        "Confirm that the district is in Texas. Accept official district pages, state education listings, Wikipedia, or credible sources.",
        loc_urls
    )
    claims.append((loc_claim, loc_urls, state_confirm_leaf, loc_ins))

    # Retirement announcement timing claim
    ra_claim = f"The retirement of {p.full_name or ''} from {p.district_name or 'the district'} was announced in 2025 or later."
    ra_urls = _combined_urls(p.retirement_urls)
    ra_ins = _missing_sources_instruction(
        "Check page or article dates. Accept if the announcement is dated 2025 or later; phrasing like 'will retire in 2026' with a 2025/2026 publication date qualifies.",
        ra_urls
    )
    claims.append((ra_claim, ra_urls, announce_date_leaf, ra_ins))

    await evaluator.batch_verify(claims)


async def verify_position_3(evaluator: Evaluator, parent_node, p: Optional[Position3]) -> None:
    pos_node = evaluator.add_parallel(
        id="Position_3",
        desc="Identify a current athletic director (as of February 2026) at an Ivy League university who is female",
        parent=parent_node,
        critical=False
    )

    # Identity group
    identity_node = evaluator.add_parallel(
        id="Position_3_Identity",
        desc="Verify the identity of the athletic director and university",
        parent=pos_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(p.full_name) if p else False,
        id="Position_3_Full_Name",
        desc="Provide the athletic director's full name",
        parent=identity_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(p.university_name) if p else False,
        id="Position_3_University_Name",
        desc="Provide the university name",
        parent=identity_node,
        critical=True
    )

    current_status_leaf = evaluator.add_leaf(
        id="Position_3_Current_Status",
        desc="Verify the person is currently serving as athletic director as of February 2026",
        parent=identity_node,
        critical=True
    )

    identity_url_leaf = evaluator.add_leaf(
        id="Position_3_Identity_URL",
        desc="URL reference confirming the athletic director's identity and current position",
        parent=identity_node,
        critical=True
    )

    # Ivy League group
    ivy_node = evaluator.add_parallel(
        id="Position_3_Ivy_League",
        desc="Verify the university is a member of the Ivy League",
        parent=pos_node,
        critical=True
    )

    ivy_confirm_leaf = evaluator.add_leaf(
        id="Position_3_Ivy_League_Confirmation",
        desc="Confirm the university is a member of the Ivy League",
        parent=ivy_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and p.institution_urls and len(p.institution_urls) > 0),
        id="Position_3_Institution_URL",
        desc="URL reference confirming Ivy League membership",
        parent=ivy_node,
        critical=True
    )

    # Gender group
    gender_node = evaluator.add_parallel(
        id="Position_3_Gender",
        desc="Verify the athletic director is female",
        parent=pos_node,
        critical=True
    )

    gender_confirm_leaf = evaluator.add_leaf(
        id="Position_3_Gender_Confirmation",
        desc="Confirm the athletic director is female",
        parent=gender_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and (p.gender_urls and len(p.gender_urls) > 0 or p.identity_urls and len(p.identity_urls) > 0)),
        id="Position_3_Gender_URL",
        desc="URL reference confirming gender",
        parent=gender_node,
        critical=True
    )

    if not p:
        p = Position3()

    claims: List[tuple[str, List[str], Any, str]] = []

    # Current status claim (AD as of Feb 2026)
    cs_claim = f"As of February 2026, {p.full_name or ''} is the athletic director at {p.university_name or ''}."
    cs_urls = _combined_urls(p.identity_urls)
    cs_ins = _missing_sources_instruction(
        "Confirm that the person is the athletic director around early 2026 (e.g., official bio, updated roster, or dated press release).",
        cs_urls
    )
    claims.append((cs_claim, cs_urls, current_status_leaf, cs_ins))

    # Identity URL claim (AD role)
    id_claim = f"{p.full_name or ''} is the athletic director at {p.university_name or ''}."
    id_urls = _combined_urls(p.identity_urls)
    id_ins = _missing_sources_instruction(
        "At least one cited page should clearly state the person is the athletic director at the university.",
        id_urls
    )
    claims.append((id_claim, id_urls, identity_url_leaf, id_ins))

    # Ivy League membership claim
    ivy_claim = f"{p.university_name or 'The university'} is a member of the Ivy League."
    ivy_urls = _combined_urls(p.institution_urls)
    ivy_ins = _missing_sources_instruction(
        "Confirm Ivy League membership. Official Ivy League site and reliable encyclopedic sources are acceptable.",
        ivy_urls
    )
    claims.append((ivy_claim, ivy_urls, ivy_confirm_leaf, ivy_ins))

    # Gender confirmation claim
    gender_text = (p.gender or "").lower()
    gender_claim = f"{p.full_name or ''} is female."
    gender_urls = _combined_urls(p.gender_urls, p.identity_urls)
    gender_ins = _missing_sources_instruction(
        "Confirm that the athletic director is female. Accept pronoun cues ('she/her'), explicit statements ('woman', 'female'), or authoritative biographical references.",
        gender_urls
    )
    claims.append((gender_claim, gender_urls, gender_confirm_leaf, gender_ins))

    await evaluator.batch_verify(claims)


async def verify_position_4(evaluator: Evaluator, parent_node, p: Optional[Position4]) -> None:
    pos_node = evaluator.add_parallel(
        id="Position_4",
        desc="Identify a special teams coordinator who moved from University of Iowa to Michigan State University with the move announced in December 2025",
        parent=parent_node,
        critical=False
    )

    # Identity/current role group
    identity_node = evaluator.add_parallel(
        id="Position_4_Identity",
        desc="Verify the identity and current position of the coach",
        parent=pos_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(p.full_name) if p else False,
        id="Position_4_Full_Name",
        desc="Provide the coach's full name",
        parent=identity_node,
        critical=True
    )

    current_title_leaf = evaluator.add_leaf(
        id="Position_4_Current_Title",
        desc="Verify the current position includes special teams coordinator role",
        parent=identity_node,
        critical=True
    )

    identity_url_leaf = evaluator.add_leaf(
        id="Position_4_Identity_URL",
        desc="URL reference confirming the coach's identity and current position",
        parent=identity_node,
        critical=True
    )

    # Previous position (Iowa) group
    prev_node = evaluator.add_parallel(
        id="Position_4_Previous_Position",
        desc="Verify previous employment at University of Iowa",
        parent=pos_node,
        critical=True
    )

    prev_institution_leaf = evaluator.add_leaf(
        id="Position_4_Previous_Institution",
        desc="Confirm previous employment at University of Iowa",
        parent=prev_node,
        critical=True
    )

    prev_role_leaf = evaluator.add_leaf(
        id="Position_4_Previous_Role",
        desc="Confirm the role at Iowa included special teams coordinator",
        parent=prev_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and p.iowa_urls and len(p.iowa_urls) > 0),
        id="Position_4_Iowa_URL",
        desc="URL reference confirming Iowa employment",
        parent=prev_node,
        critical=True
    )

    # Current position (MSU) group
    current_node = evaluator.add_parallel(
        id="Position_4_Current_Position",
        desc="Verify current employment at Michigan State University",
        parent=pos_node,
        critical=True
    )

    current_institution_leaf = evaluator.add_leaf(
        id="Position_4_Current_Institution",
        desc="Confirm current employment at Michigan State University",
        parent=current_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and p.msu_urls and len(p.msu_urls) > 0),
        id="Position_4_MSU_URL",
        desc="URL reference confirming Michigan State employment",
        parent=current_node,
        critical=True
    )

    # Move timing group
    timing_node = evaluator.add_parallel(
        id="Position_4_Move_Timing",
        desc="Verify the move was announced in December 2025",
        parent=pos_node,
        critical=True
    )

    move_announce_leaf = evaluator.add_leaf(
        id="Position_4_Move_Announcement",
        desc="Confirm the move was announced in December 2025",
        parent=timing_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(p and p.announcement_urls and len(p.announcement_urls) > 0),
        id="Position_4_Timing_URL",
        desc="URL reference confirming the announcement timing",
        parent=timing_node,
        critical=True
    )

    if not p:
        p = Position4()

    claims: List[tuple[str, List[str], Any, str]] = []

    # Current title at MSU (includes special teams coordinator)
    ct_claim = f"At Michigan State University, {p.full_name or ''} holds a special teams coordinator role (or equivalent wording)."
    ct_urls = _combined_urls(p.msu_urls)
    ct_ins = _missing_sources_instruction(
        "Confirm MSU role includes 'special teams coordinator'. Accept combined titles like 'special teams coordinator and [other role]'.",
        ct_urls
    )
    claims.append((ct_claim, ct_urls, current_title_leaf, ct_ins))

    # Identity/current position (redundant check to ensure support)
    id_claim = f"{p.full_name or ''} is on the Michigan State University football staff in a special teams coordinator capacity."
    id_urls = _combined_urls(p.msu_urls)
    id_ins = _missing_sources_instruction(
        "At least one cited page should clearly show the coach on MSU staff with special teams responsibilities.",
        id_urls
    )
    claims.append((id_claim, id_urls, identity_url_leaf, id_ins))

    # Previous institution (Iowa)
    pi_claim = f"Before joining Michigan State, {p.full_name or ''} worked at the University of Iowa."
    pi_urls = _combined_urls(p.iowa_urls)
    pi_ins = _missing_sources_instruction(
        "Confirm prior Iowa employment from official bios or credible reporting.",
        pi_urls
    )
    claims.append((pi_claim, pi_urls, prev_institution_leaf, pi_ins))

    # Previous role (Iowa special teams)
    pr_claim = f"At Iowa, {p.full_name or ''}'s role included special teams coordinator duties."
    pr_urls = _combined_urls(p.iowa_urls)
    pr_ins = _missing_sources_instruction(
        "Confirm the Iowa role included special teams coordinator/coach responsibilities (accept equivalent phrasing).",
        pr_urls
    )
    claims.append((pr_claim, pr_urls, prev_role_leaf, pr_ins))

    # Current institution at MSU
    ci_claim = f"{p.full_name or ''} is employed by Michigan State University."
    ci_urls = _combined_urls(p.msu_urls)
    ci_ins = _missing_sources_instruction(
        "Confirm the coach is on MSU staff (bio, roster, or press release).",
        ci_urls
    )
    claims.append((ci_claim, ci_urls, current_institution_leaf, ci_ins))

    # Move announcement timing (Dec 2025)
    mt_claim = f"The move of {p.full_name or ''} from Iowa to Michigan State was announced in December 2025."
    mt_urls = _combined_urls(p.announcement_urls, p.msu_urls)
    mt_ins = _missing_sources_instruction(
        "Confirm the hire/move announcement is dated in December 2025. The page date must show December 2025.",
        mt_urls
    )
    claims.append((mt_claim, mt_urls, move_announce_leaf, mt_ins))

    await evaluator.batch_verify(claims)


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel root to evaluate four independent positions
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract all positions data
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=AllPositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Build the four position subtrees per rubric
    await verify_position_1(evaluator, root, extracted.position1 if extracted else None)
    await verify_position_2(evaluator, root, extracted.position2 if extracted else None)
    await verify_position_3(evaluator, root, extracted.position3 if extracted else None)
    await verify_position_4(evaluator, root, extracted.position4 if extracted else None)

    return evaluator.get_summary()