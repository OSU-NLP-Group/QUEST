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
TASK_ID = "p5_ad_vp_athletics_leader"
TASK_DESCRIPTION = """
Identify a current Vice President for Intercollegiate Athletics or Director of Athletics at an NCAA Division I Power Five conference institution who meets all of the following criteria:

Educational Background:
- Holds a doctoral degree (PhD or EdD) in sport management, educational leadership, or a closely related field
- Holds a master's degree in athletic administration, sport management, or a related field
- Holds a bachelor's degree from an accredited four-year institution
- All degrees must be from regionally accredited institutions

Career Requirements:
- Served as Athletic Director at at least one NCAA Division I institution immediately prior to their current position
- Held that previous Athletic Director position for a minimum of 2 years
- Has worked at a minimum of 3 different higher education institutions during their athletics administration career
- Held at least one assistant or associate athletic director position earlier in their career
- Career demonstrates progressive advancement through multiple administrative levels in collegiate athletics

Current Position:
- Currently holds the title of Vice President for Intercollegiate Athletics or Director of Athletics
- Current institution must be a member of a Power Five conference (Big Ten, SEC, ACC, Big 12, or Pac-12)
- Began current position between July 2022 and December 2025

Athletic Background:
- Has experience as a former collegiate student-athlete

Provide the individual's full name, current institution, current title, and reference URLs supporting each major qualification category.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Degree(BaseModel):
    level: Optional[str] = None          # e.g., "PhD", "EdD", "Doctor of Education", "MS", "MEd", "BA", "BS"
    field: Optional[str] = None          # e.g., "Sport Management", "Educational Leadership", etc.
    institution: Optional[str] = None
    year: Optional[str] = None           # flexible string (e.g., "2014", "May 2014")
    sources: List[str] = Field(default_factory=list)


class CareerEntry(BaseModel):
    title: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None     # e.g., "July 2020", "2019"
    end_date: Optional[str] = None       # e.g., "June 2022", "Present"
    sources: List[str] = Field(default_factory=list)


class AthleticBackground(BaseModel):
    sport: Optional[str] = None          # e.g., "Football", "Track & Field"
    school: Optional[str] = None         # e.g., "XYZ University"
    details: Optional[str] = None        # any descriptive details
    sources: List[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    # Identity and current role
    name: Optional[str] = None
    current_institution: Optional[str] = None
    current_title: Optional[str] = None
    current_start_date: Optional[str] = None   # e.g., "August 2023"
    conference: Optional[str] = None           # e.g., "SEC", "Big Ten", etc.
    current_position_sources: List[str] = Field(default_factory=list)

    # Education
    degrees: List[Degree] = Field(default_factory=list)
    educational_sources: List[str] = Field(default_factory=list)

    # Career
    career_history: List[CareerEntry] = Field(default_factory=list)
    career_sources: List[str] = Field(default_factory=list)

    # Student-athlete background
    athletic_background: Optional[AthleticBackground] = None
    athletic_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profile() -> str:
    return """
    From the provided answer, extract a single individual's profile who is identified as a current Vice President for Intercollegiate Athletics or Director of Athletics at a Power Five institution. Return the following fields:

    1) Identity and current role (if stated):
       - name: Full name of the individual
       - current_institution: The current institution (university) where the person serves
       - current_title: The current job title
       - current_start_date: The month/year (or year) they began the current role (e.g., "July 2023")
       - conference: The athletic conference of the current institution (e.g., "SEC", "Big Ten", "ACC", "Big 12", or "Pac-12")
       - current_position_sources: URLs directly supporting the current role, title, start date, and conference membership

    2) Education:
       - degrees: an array (reverse-chronological if known). For each, include:
            * level (e.g., "PhD", "EdD", "Doctor of Education", "MS", "MEd", "MBA", "MA", "BA", "BS")
            * field (e.g., "Sport Management", "Educational Leadership", etc.)
            * institution (the awarding institution)
            * year (if available)
            * sources: URLs supporting this specific degree
       - educational_sources: URLs that the answer cites as general support for the educational background

    3) Career history (reverse-chronological, including the current role if stated):
       - career_history: an array of positions. For each, include:
            * title
            * institution
            * start_date (month/year or year if available)
            * end_date (month/year, year, or "Present")
            * sources: URLs supporting this specific position and dates
       - career_sources: URLs cited in the answer that generally support career history

    4) Athletic background:
       - athletic_background:
            * sport (e.g., "basketball", "track & field") if mentioned
            * school (where they competed)
            * details (any brief descriptive context)
            * sources: URLs specifically supporting collegiate student-athlete experience
       - athletic_sources: additional URLs cited to support athletic background (if any)

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. Do not invent URLs.
    - If any field is missing, set it to null (or an empty list for arrays).
    - Normalize URLs: include protocol (http:// or https://). If missing, prepend http://.
    - Keep free-form date strings as-is (do not try to reformat them).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _collect_all_urls(profile: CandidateProfile) -> List[str]:
    urls: List[str] = []
    urls.extend(profile.current_position_sources or [])
    urls.extend(profile.educational_sources or [])
    urls.extend(profile.career_sources or [])
    urls.extend(profile.athletic_sources or [])
    for d in profile.degrees or []:
        urls.extend(d.sources or [])
    for c in profile.career_history or []:
        urls.extend(c.sources or [])
    if profile.athletic_background:
        urls.extend(profile.athletic_background.sources or [])
    return _dedup_urls(urls)


def _degree_sources(profile: CandidateProfile, predicate=None) -> List[str]:
    urls: List[str] = []
    for d in profile.degrees or []:
        if predicate is None or predicate(d):
            urls.extend(d.sources or [])
    urls.extend(profile.educational_sources or [])
    return _dedup_urls(urls)


def _career_sources(profile: CandidateProfile, entries: Optional[List[CareerEntry]] = None) -> List[str]:
    urls: List[str] = []
    for c in (entries if entries is not None else (profile.career_history or [])):
        urls.extend(c.sources or [])
    urls.extend(profile.career_sources or [])
    return _dedup_urls(urls)


def _current_sources(profile: CandidateProfile) -> List[str]:
    # Include current-position sources and, if needed, also the "current" entry in career_history
    urls: List[str] = []
    urls.extend(profile.current_position_sources or [])
    # Try to include the entry that matches the current role (if available)
    for c in profile.career_history or []:
        if c.institution and profile.current_institution and c.institution.strip().lower() == profile.current_institution.strip().lower():
            if profile.current_title and c.title and c.title.strip().lower() == profile.current_title.strip().lower():
                urls.extend(c.sources or [])
    return _dedup_urls(urls)


def _athletic_sources(profile: CandidateProfile) -> List[str]:
    urls: List[str] = []
    if profile.athletic_background:
        urls.extend(profile.athletic_background.sources or [])
    urls.extend(profile.athletic_sources or [])
    return _dedup_urls(urls)


def _is_doctoral(level: Optional[str]) -> bool:
    if not level:
        return False
    s = level.lower()
    return ("phd" in s) or ("ph.d" in s) or ("doctor" in s) or ("ed.d" in s) or ("edd" in s)


def _is_master(level: Optional[str]) -> bool:
    if not level:
        return False
    s = level.lower()
    return ("master" in s) or ("m.s" in s) or ("ms" in s) or ("m.a" in s) or ("ma" in s) or ("mba" in s) or ("m.ed" in s) or ("med" in s)


def _is_bachelor(level: Optional[str]) -> bool:
    if not level:
        return False
    s = level.lower()
    return ("bachelor" in s) or ("b.s" in s) or ("bs" in s) or ("b.a" in s) or ("ba" in s)


def _find_first(predicate, items: List[Any]) -> Optional[Any]:
    for it in items or []:
        if predicate(it):
            return it
    return None


def _normalize_str(s: Optional[str]) -> str:
    return s or ""


def _pick_previous_position(profile: CandidateProfile) -> Optional[CareerEntry]:
    """
    Try to identify the immediate previous role before the current one.
    Heuristic: assume career_history is reverse-chronological. Find the current entry (matching current_institution+current_title if possible),
    then pick the next entry. Otherwise, pick the first non-current-looking entry with a finite end_date (not present).
    """
    ch = profile.career_history or []
    if not ch:
        return None

    # Try exact current match first
    cur_inst = (_normalize_str(profile.current_institution)).strip().lower()
    cur_title = (_normalize_str(profile.current_title)).strip().lower()

    cur_idx = None
    for i, e in enumerate(ch):
        ei = (_normalize_str(e.institution)).strip().lower()
        et = (_normalize_str(e.title)).strip().lower()
        if ei and cur_inst and ei == cur_inst:
            # If title also matches, best match
            if cur_title and et and et == cur_title:
                cur_idx = i
                break

    if cur_idx is not None:
        if cur_idx + 1 < len(ch):
            return ch[cur_idx + 1]

    # Fallback: return the first entry that is clearly not "Present"
    for e in ch:
        if e.end_date and ("present" in e.end_date.strip().lower() or "current" in e.end_date.strip().lower()):
            continue
        # Avoid returning an entry that seems to be the current one (institution+title)
        ei = (_normalize_str(e.institution)).strip().lower()
        et = (_normalize_str(e.title)).strip().lower()
        if not (ei == cur_inst and et == cur_title):
            return e

    # Last resort: if none above, try the second item if exists
    return ch[1] if len(ch) > 1 else None


def _contains_asst_assoc_ad(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    # Look for assistant/associate athletic director phrases
    return ("assistant athletic director" in t) or ("associate athletic director" in t) or ("asst. athletic director" in t) or ("assoc. athletic director" in t)


def _is_ad_role(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    return ("director of athletics" in t) or ("athletic director" in t) or ("athletics director" in t) or ("ad," in t) or (t.strip() == "ad")


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, profile: CandidateProfile) -> None:
    # Top-level critical node (parallel aggregation)
    ad_node = evaluator.add_parallel(
        id="Athletic_Director_Identification",
        desc="Evaluate whether the identified individual meets all specified criteria for a Vice President for Intercollegiate Athletics position at a Power Five Division I institution",
        parent=root,
        critical=True
    )

    # ------------------------- Educational Background --------------------- #
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Verify that the individual holds all required academic degrees from accredited institutions",
        parent=ad_node,
        critical=True
    )

    # Degree matches
    doctoral = _find_first(lambda d: _is_doctoral(d.level), profile.degrees or [])
    masters = _find_first(lambda d: _is_master(d.level), profile.degrees or [])
    bachelors = _find_first(lambda d: _is_bachelor(d.level), profile.degrees or [])

    # Doctoral_Degree
    doc_leaf = evaluator.add_leaf(
        id="Doctoral_Degree",
        desc="The individual holds a PhD or EdD in sport management, educational leadership, or closely related field",
        parent=edu_node,
        critical=True
    )
    doc_sources = _degree_sources(profile, predicate=lambda d: _is_doctoral(d.level)) or _collect_all_urls(profile)
    doc_field = _normalize_str(doctoral.field if doctoral else None)
    doc_inst = _normalize_str(doctoral.institution if doctoral else None)
    doc_level = _normalize_str(doctoral.level if doctoral else None)
    doc_claim = (
        f"The individual holds a doctoral degree (PhD or EdD) in sport management, educational leadership, or a closely related field. "
        f"Their doctoral credential is described as '{doc_level}' in '{doc_field}' from '{doc_inst}'."
    )
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=doc_sources,
        additional_instruction=(
            "Accept clear equivalents of EdD/PhD and closely related fields such as higher education administration, "
            "education administration, sport administration, sport studies, kinesiology (sport administration/management concentration), "
            "or educational leadership. The sources must explicitly support the doctoral credential and field."
        )
    )

    # Masters_Degree
    ms_leaf = evaluator.add_leaf(
        id="Masters_Degree",
        desc="The individual holds a master's degree in athletic administration, sport management, or related field",
        parent=edu_node,
        critical=True
    )
    ms_sources = _degree_sources(profile, predicate=lambda d: _is_master(d.level)) or _collect_all_urls(profile)
    ms_field = _normalize_str(masters.field if masters else None)
    ms_inst = _normalize_str(masters.institution if masters else None)
    ms_level = _normalize_str(masters.level if masters else None)
    ms_claim = (
        f"The individual holds a master's degree in athletic administration, sport management, or a related field. "
        f"Their master's credential is described as '{ms_level}' in '{ms_field}' from '{ms_inst}'."
    )
    await evaluator.verify(
        claim=ms_claim,
        node=ms_leaf,
        sources=ms_sources,
        additional_instruction=(
            "Treat master's-level degrees in sport management/administration, athletic administration, sports administration, "
            "or clearly related fields (e.g., higher education administration) as satisfying this requirement. "
            "The sources should explicitly support the master's credential and field."
        )
    )

    # Bachelors_Degree
    ba_leaf = evaluator.add_leaf(
        id="Bachelors_Degree",
        desc="The individual holds a bachelor's degree from an accredited four-year institution",
        parent=edu_node,
        critical=True
    )
    ba_sources = _degree_sources(profile, predicate=lambda d: _is_bachelor(d.level)) or _collect_all_urls(profile)
    ba_field = _normalize_str(bachelors.field if bachelors else None)
    ba_inst = _normalize_str(bachelors.institution if bachelors else None)
    ba_level = _normalize_str(bachelors.level if bachelors else None)
    ba_claim = (
        f"The individual holds a bachelor's degree from a four-year accredited institution. "
        f"The bachelor's credential is described as '{ba_level}' in '{ba_field}' from '{ba_inst}'."
    )
    await evaluator.verify(
        claim=ba_claim,
        node=ba_leaf,
        sources=ba_sources,
        additional_instruction=(
            "Verify that the person has a bachelor's degree (field can vary). The page(s) should clearly state the bachelor's degree and the awarding institution."
        )
    )

    # Degree_Accreditation
    acc_leaf = evaluator.add_leaf(
        id="Degree_Accreditation",
        desc="All degrees (bachelor's, master's, and doctoral) were earned from regionally accredited institutions",
        parent=edu_node,
        critical=True
    )
    awarding_insts = [d.institution for d in (profile.degrees or []) if d.institution]
    awarding_insts_str = ", ".join(sorted(set(awarding_insts))) if awarding_insts else "N/A"
    acc_sources = _degree_sources(profile, predicate=None) or _collect_all_urls(profile)
    acc_claim = (
        f"All degree-granting institutions for the individual's bachelor's, master's, and doctoral degrees "
        f"are regionally accredited U.S. institutions. Confirm regional accreditation for each awarding institution: {awarding_insts_str}."
    )
    await evaluator.verify(
        claim=acc_claim,
        node=acc_leaf,
        sources=acc_sources,
        additional_instruction=(
            "Use explicit evidence from the provided sources that the awarding institutions are regionally accredited "
            "(e.g., accreditation pages or mentions of regional accreditors like HLC, SACSCOC, MSCHE, NECHE, NWCCU, WSCUC). "
            "If accreditation cannot be confirmed for any awarding institution, mark this claim as not supported."
        )
    )

    # ------------------------- Career Requirements ------------------------ #
    career_node = evaluator.add_parallel(
        id="Career_Requirements",
        desc="Verify that the individual's career history meets all progression and experience requirements",
        parent=ad_node,
        critical=True
    )

    previous = _pick_previous_position(profile)
    prev_inst = _normalize_str(previous.institution if previous else None)
    prev_title = _normalize_str(previous.title if previous else None)
    prev_start = _normalize_str(previous.start_date if previous else None)
    prev_end = _normalize_str(previous.end_date if previous else None)
    prev_sources = _career_sources(profile, entries=[previous] if previous else None) or _career_sources(profile)

    # Previous_AD_Position
    prev_ad_leaf = evaluator.add_leaf(
        id="Previous_AD_Position",
        desc="The individual served as Athletic Director at at least one NCAA Division I institution immediately before their current role",
        parent=career_node,
        critical=True
    )
    prev_ad_claim = (
        f"Immediately before the current role, the individual served as an Athletic Director (e.g., 'Director of Athletics' or equivalent) "
        f"at '{prev_inst}', which is an NCAA Division I institution. Their title in that role was '{prev_title}'."
    )
    await evaluator.verify(
        claim=prev_ad_claim,
        node=prev_ad_leaf,
        sources=prev_sources,
        additional_instruction=(
            "Focus on the immediate prior role (the job held directly before the current position). "
            "The title must clearly indicate an Athletic Director role (e.g., 'Director of Athletics', 'Athletics Director', 'Athletic Director'). "
            "Also verify that the institution is NCAA Division I. If either the AD title or Division I status cannot be confirmed, mark as not supported."
        )
    )

    # AD_Tenure_Duration
    tenure_leaf = evaluator.add_leaf(
        id="AD_Tenure_Duration",
        desc="The previous Athletic Director position was held for a minimum of 2 years",
        parent=career_node,
        critical=True
    )
    tenure_claim = (
        f"In the immediate prior role as AD at '{prev_inst}', the individual served from '{prev_start}' to '{prev_end}', "
        f"which is at least two years (>= 24 months) in duration."
    )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=prev_sources,
        additional_instruction=(
            "Compute an approximate duration from the stated start and end dates. "
            "If only years are given (e.g., 2020–2022), treat that as approximately 2+ years if inclusive. "
            "If the duration is clearly less than 24 months, mark as not supported."
        )
    )

    # Multiple_Institutions
    multi_inst_leaf = evaluator.add_leaf(
        id="Multiple_Institutions",
        desc="The individual has worked at a minimum of 3 different higher education institutions during their athletics administration career",
        parent=career_node,
        critical=True
    )
    unique_insts = sorted({(_normalize_str(c.institution)).strip() for c in (profile.career_history or []) if _normalize_str(c.institution).strip()})
    inst_list_str = ", ".join(unique_insts) if unique_insts else "N/A"
    multi_claim = (
        f"The individual's athletics administration career includes work at at least 3 distinct higher education institutions. "
        f"Institutions listed: {inst_list_str}."
    )
    await evaluator.verify(
        claim=multi_claim,
        node=multi_inst_leaf,
        sources=_career_sources(profile),
        additional_instruction=(
            "Cross-check the career entries and confirm that at least three distinct higher education institutions are represented. "
            "If three or more unique institutions cannot be verified from the provided sources, mark as not supported."
        )
    )

    # Assistant_AD_Experience
    asst_entries = [c for c in (profile.career_history or []) if _contains_asst_assoc_ad(c.title)]
    asst_example = asst_entries[0] if asst_entries else None
    asst_title = _normalize_str(asst_example.title if asst_example else None)
    asst_inst = _normalize_str(asst_example.institution if asst_example else None)
    asst_leaf = evaluator.add_leaf(
        id="Assistant_AD_Experience",
        desc="The individual held at least one assistant or associate athletic director position earlier in their career",
        parent=career_node,
        critical=True
    )
    asst_claim = (
        f"The individual previously held at least one assistant or associate athletic director position "
        f"(e.g., '{asst_title}' at '{asst_inst}')."
    )
    await evaluator.verify(
        claim=asst_claim,
        node=asst_leaf,
        sources=_career_sources(profile, entries=asst_entries) or _career_sources(profile),
        additional_instruction=(
            "Look for titles explicitly containing 'Assistant Athletic Director' or 'Associate Athletic Director' (including abbreviations like 'Asst.' or 'Assoc.'). "
            "If no such role can be confirmed from the sources, mark as not supported."
        )
    )

    # Progressive_Career_Path
    prog_leaf = evaluator.add_leaf(
        id="Progressive_Career_Path",
        desc="The individual's career demonstrates progressive advancement through multiple administrative levels in collegiate athletics",
        parent=career_node,
        critical=True
    )
    prog_claim = (
        "The individual's career shows progressive advancement through multiple administrative levels in collegiate athletics "
        "(e.g., assistant/associate AD roles leading to AD/VP-level leadership across institutions)."
    )
    await evaluator.verify(
        claim=prog_claim,
        node=prog_leaf,
        sources=_career_sources(profile),
        additional_instruction=(
            "Evaluate the sequence of roles in the career history to confirm advancement from earlier/lower-tier administrative roles "
            "to higher leadership roles (e.g., AD or VP for Athletics). The evidence should support an upward trajectory."
        )
    )

    # --------------------------- Current Position ------------------------- #
    curr_node = evaluator.add_parallel(
        id="Current_Position",
        desc="Verify that the individual's current role meets all specified requirements",
        parent=ad_node,
        critical=True
    )

    # Current_Title
    curr_title_leaf = evaluator.add_leaf(
        id="Current_Title",
        desc="The individual currently holds the title of Vice President for Intercollegiate Athletics or Director of Athletics at an NCAA Division I institution",
        parent=curr_node,
        critical=True
    )
    curr_sources = _current_sources(profile) or _collect_all_urls(profile)
    curr_claim = (
        f"The individual currently serves as '{_normalize_str(profile.current_title)}' at '{_normalize_str(profile.current_institution)}'. "
        f"This title qualifies as 'Vice President for Intercollegiate Athletics' or 'Director of Athletics' (including common variants like 'Athletic Director' or 'Athletics Director')."
    )
    await evaluator.verify(
        claim=curr_claim,
        node=curr_title_leaf,
        sources=curr_sources,
        additional_instruction=(
            "Treat title variants and combined titles equivalently (e.g., 'Vice President & Director of Athletics', 'Athletic Director', 'Athletics Director'). "
            "The source must clearly support the current title and institution."
        )
    )

    # Power_Five_Conference
    p5_leaf = evaluator.add_leaf(
        id="Power_Five_Conference",
        desc="The individual's current institution is a member of a Power Five conference (Big Ten, SEC, ACC, Big 12, or Pac-12)",
        parent=curr_node,
        critical=True
    )
    inst = _normalize_str(profile.current_institution)
    conf = _normalize_str(profile.conference)
    p5_claim = (
        f"The current institution '{inst}' competes in a Power Five conference (Big Ten, SEC, ACC, Big 12, or Pac-12). "
        f"The conference referenced for this institution is '{conf}'."
    )
    await evaluator.verify(
        claim=p5_claim,
        node=p5_leaf,
        sources=curr_sources,
        additional_instruction=(
            "Confirm from the provided pages that the institution is in one of Big Ten, SEC, ACC, Big 12, or Pac-12. "
            "Official school athletics sites, official conference sites, or authoritative profiles/press releases are acceptable."
        )
    )

    # Starting_Date_Current_Role
    start_leaf = evaluator.add_leaf(
        id="Starting_Date_Current_Role",
        desc="The individual began their current position between July 2022 and December 2025",
        parent=curr_node,
        critical=True
    )
    start = _normalize_str(profile.current_start_date)
    start_claim = (
        f"The individual began the current position in '{start}', and that start date falls between July 2022 and December 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=curr_sources,
        additional_instruction=(
            "Use the stated start date on the provided sources. Verify the date is within the inclusive window: from July 1, 2022 to December 31, 2025. "
            "If the start date is outside this window or cannot be confirmed, mark as not supported."
        )
    )

    # ------------------------- Athletic Background ------------------------ #
    ath_node = evaluator.add_parallel(
        id="Athletic_Background",
        desc="Verify that the individual has the required athletic playing experience",
        parent=ad_node,
        critical=True
    )

    sa_leaf = evaluator.add_leaf(
        id="Student_Athlete_Background",
        desc="The individual has experience as a former collegiate student-athlete",
        parent=ath_node,
        critical=True
    )
    sport = _normalize_str(profile.athletic_background.sport if profile.athletic_background else None)
    school = _normalize_str(profile.athletic_background.school if profile.athletic_background else None)
    sa_claim = (
        f"The individual is a former collegiate student-athlete. Reported sport: '{sport}'. Reported school/team: '{school}'."
    )
    await evaluator.verify(
        claim=sa_claim,
        node=sa_leaf,
        sources=_athletic_sources(profile) or _collect_all_urls(profile),
        additional_instruction=(
            "The evidence should explicitly state that the individual competed as a student-athlete at the collegiate level (e.g., roster bio, official biography, media guide). "
            "If no explicit confirmation, mark as not supported."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Power Five Athletics Leader task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Parallel aggregation at top-level
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

    # 1) Extract structured data from the answer
    profile = await evaluator.extract(
        prompt=prompt_extract_profile(),
        template_class=CandidateProfile,
        extraction_name="candidate_profile",
    )

    # 2) Build verification tree and run checks
    await build_verification_tree(evaluator, root, profile)

    # 3) Return evaluation summary
    return evaluator.get_summary()