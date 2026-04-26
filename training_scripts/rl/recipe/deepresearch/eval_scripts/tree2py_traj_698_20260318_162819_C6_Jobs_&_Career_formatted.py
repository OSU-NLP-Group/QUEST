import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_universities_2026"
TASK_DESCRIPTION = """
Identify FOUR different universities that are current members of the Big Ten Conference (as of March 2026) and provide comprehensive information about each institution's athletic programs and graduate educational opportunities. The four universities must be four distinct institutions, and you must provide the following information for EACH university:

A. Institution Identification and Conference Membership
   - The official name of the university
   - The primary campus location (city and state)
   - Verification that the university is a current Big Ten Conference member (with reference URL from bigten.org or the university's official athletic website)
   - The official athletic department website URL

B. Football Program Verification
   - Confirmation that the university fields a varsity football team competing in NCAA Division I
   - The name and official title of the current head football coach
   - Reference URL from the official athletic website showing the current football coaching staff

C. Graduate Program in Sports/Athletic Field
   - Whether the university offers a graduate degree program (Master's or Doctorate) specifically in Sports Management, Sport Management, Athletic Administration, Athletic Training, or a directly equivalent field
   - If yes: the exact official program name
   - If yes: the college or school that houses the program (e.g., "College of Education," "School of Kinesiology")
   - If yes: the official program website URL

D. Graduate Assistantship Opportunities
   - Whether the university's athletic department offers graduate assistantship positions (Yes/No)
   - Reference URL or evidence from the university's athletic department, graduate school, or employment pages confirming the availability (or lack) of graduate assistantships in athletics

E. Educational Requirements for Athletic Staff
   - The typical minimum educational requirement (Bachelor's degree, Master's degree, or other) for assistant coaching or athletic administrative positions at this university, based on publicly available job postings, career pages, or HR policy documents
   - Reference URL supporting this educational requirement

Important Requirements:
- All four universities must be different Big Ten Conference member institutions
- You may NOT use Iowa or Michigan State as two of your four universities
- All information must be current as of March 2026
- All reference URLs must link to official university, Big Ten Conference, or NCAA sources
- Each piece of information must be verifiable through the provided URLs
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityEntry(BaseModel):
    # A. Identity and Membership
    official_name: Optional[str] = None
    location_text: Optional[str] = None  # As written (e.g., "Columbus, Ohio")
    campus_city: Optional[str] = None
    campus_state: Optional[str] = None
    athletics_website_url: Optional[str] = None
    membership_urls: List[str] = Field(default_factory=list)  # bigten.org or official athletics/university pages

    # B. Football
    football_head_coach_name: Optional[str] = None
    football_head_coach_title: Optional[str] = None  # e.g., "Head Football Coach"
    coaching_staff_url: Optional[str] = None  # official athletics site page for staff/head coach

    # C. Graduate sports-related program
    grad_program_exists: Optional[str] = None  # "yes" or "no"
    grad_program_name: Optional[str] = None
    grad_program_college_school: Optional[str] = None
    grad_program_url: Optional[str] = None

    # D. Graduate assistantships (athletics)
    ga_yes_no: Optional[str] = None  # "yes" or "no"
    ga_evidence_url: Optional[str] = None

    # E. Minimum education requirement for athletics roles
    min_edu_requirement: Optional[str] = None  # e.g., "Bachelor's degree", "Master's degree"
    min_edu_requirement_url: Optional[str] = None  # official job posting/HR/careers/policy URL


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract the universities and fields exactly as they are presented in the answer text. Each "university entry" corresponds to one distinct Big Ten member institution that the answer discusses.

For each university entry you find in the answer (in order of appearance), extract the following fields:

A. Identity & Membership
- official_name: The official university name (string).
- location_text: The primary campus location as a single string if provided (e.g., "Columbus, Ohio").
- campus_city: City name if explicitly provided; otherwise null.
- campus_state: State name or postal abbreviation if explicitly provided; otherwise null.
- athletics_website_url: The official athletics department main website URL for this university (if given; otherwise null).
- membership_urls: A list of one or more URLs cited in the answer that directly support Big Ten membership. Prefer urls from bigten.org or the university's official (athletics/university) domain. If none are provided, return an empty list.

B. Football
- football_head_coach_name: The name of the current head football coach as stated in the answer (if present).
- football_head_coach_title: The exact title as stated (e.g., "Head Football Coach") (if present).
- coaching_staff_url: A URL (from the official athletics site) that shows the current football coaching staff or head coach information.

C. Graduate Sports/Athletics Program
- grad_program_exists: "yes" or "no" exactly, according to the answer. If unclear in the answer, set to null.
- grad_program_name: If grad_program_exists is "yes", provide the exact official program name as written in the answer; else null.
- grad_program_college_school: If "yes", provide the name of the housing college/school; else null.
- grad_program_url: If "yes", provide the official program website URL; else null.

D. Graduate Assistantships in Athletics
- ga_yes_no: "yes" or "no" exactly, according to the answer. If unclear in the answer, set to null.
- ga_evidence_url: A URL from an official university/athletics/HR/graduate-school page that supports the claim about GA opportunities (availability or non-availability). If not provided, set null.

E. Educational Requirements (Athletics roles)
- min_edu_requirement: The typical minimum educational requirement stated for assistant coaching or athletic administrative roles (e.g., "Bachelor's degree", "Master's degree"). If given as text in the answer, extract it exactly.
- min_edu_requirement_url: A supporting official URL (job posting, HR/careers/policy page) cited for the requirement.

General instructions:
- Return a JSON object with a top-level array field "universities".
- Include ALL universities mentioned in the answer (do not filter by Iowa or Michigan State yourself). If the answer includes more than four, include all; do not alter the content. We will later consider only the first four entries for per-university checks.
- Do not invent or infer data. If a field isn't explicitly present, set it to null (or [] for lists).
- For URL fields, only extract URLs explicitly present in the answer text. Do not fabricate URLs. Extract full URLs including protocols. Ignore obviously malformed URLs.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _norm_lower(s: Optional[str]) -> str:
    return _norm(s).casefold()


def _is_nonempty_url(url: Optional[str]) -> bool:
    u = _norm(url)
    return u.startswith("http://") or u.startswith("https://")


def _extract_domain(url: Optional[str]) -> Optional[str]:
    if not _is_nonempty_url(url):
        return None
    try:
        return urlparse(url).hostname
    except Exception:
        return None


def _yes_no_norm(s: Optional[str]) -> Optional[str]:
    v = _norm_lower(s)
    if v in {"yes", "y"}:
        return "yes"
    if v in {"no", "n"}:
        return "no"
    return None


def _has_location(uni: UniversityEntry) -> bool:
    # Accept either both city and state, or a location_text that appears to contain both (comma-separated).
    if _norm(uni.campus_city) and _norm(uni.campus_state):
        return True
    lt = _norm(uni.location_text)
    if lt and ("," in lt):
        parts = [p.strip() for p in lt.split(",")]
        # minimally ensure two tokens
        return len(parts) >= 2 and all(bool(p) for p in parts[:2])
    return False


def _is_banned_university(name: Optional[str]) -> bool:
    """
    Exclude 'Iowa' (University of Iowa) and 'Michigan State' (Michigan State University).
    We'll match on official_name text only, not location.
    """
    n = _norm_lower(name)
    if not n:
        return False
    if "michigan state" in n:
        return True
    # Ban "university of iowa" or "iowa" standing alone (but not "iowa state")
    tokens = re.findall(r"[a-z]+", n)
    if "iowa" in tokens and "state" not in tokens:
        # likely "University of Iowa" or "Iowa"
        return True
    if "university" in tokens and "of" in tokens and "iowa" in tokens:
        return True
    return False


def _is_provided_university_entry(uni: UniversityEntry) -> bool:
    # Consider an entry present if it has at least an official name, athletics site URL, membership URL, or staff URL.
    if _norm(uni.official_name):
        return True
    if _is_nonempty_url(uni.athletics_website_url):
        return True
    if any(_is_nonempty_url(u) for u in uni.membership_urls):
        return True
    if _is_nonempty_url(uni.coaching_staff_url):
        return True
    return False


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _is_nonempty_url(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityEntry, idx: int) -> None:
    # University container (non-critical; partial credit per university)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"University entry #{idx+1}.",
        parent=parent_node,
        critical=False
    )

    # A. Identity & Membership (critical)
    a_node = evaluator.add_parallel(
        id=f"u{idx+1}_A_identity_membership",
        desc="Institution identification and Big Ten membership verification.",
        parent=uni_node,
        critical=True
    )
    # A1. Official Name provided
    evaluator.add_custom_node(
        result=bool(_norm(uni.official_name)),
        id=f"u{idx+1}_official_name_provided",
        desc="Provides the university's official name.",
        parent=a_node,
        critical=True
    )
    # A2. Primary campus location (city and state)
    evaluator.add_custom_node(
        result=_has_location(uni),
        id=f"u{idx+1}_primary_campus_location",
        desc="Provides the primary campus location (city and state).",
        parent=a_node,
        critical=True
    )
    # A3. Athletics website URL provided
    evaluator.add_custom_node(
        result=_is_nonempty_url(uni.athletics_website_url),
        id=f"u{idx+1}_athletics_website_url_provided",
        desc="Provides the official athletic department website URL.",
        parent=a_node,
        critical=True
    )
    # A4. Big Ten membership verification via allowed official citation
    a4_node = evaluator.add_leaf(
        id=f"u{idx+1}_big_ten_membership_supported",
        desc="Verifies current Big Ten membership as of March 2026 with official source(s).",
        parent=a_node,
        critical=True
    )
    membership_sources = list(uni.membership_urls or [])
    # Fallback to athletics site if no specific membership URL was provided
    if not membership_sources and _is_nonempty_url(uni.athletics_website_url):
        membership_sources = [uni.athletics_website_url]
    claim_membership = f"{_norm(uni.official_name) or 'This university'} is a current member of the Big Ten Conference as of March 2026."
    await evaluator.verify(
        claim=claim_membership,
        node=a4_node,
        sources=membership_sources,
        additional_instruction=(
            "Only accept if the provided page(s) are official: bigten.org or an official university/athletics domain. "
            "Look for explicit indications of Big Ten membership (e.g., school listed on bigten.org, school site stating Big Ten affiliation). "
            "Reject third‑party or unofficial sources."
        )
    )

    # B. Football Program (critical)
    b_node = evaluator.add_parallel(
        id=f"u{idx+1}_B_football",
        desc="Football program verification.",
        parent=uni_node,
        critical=True
    )
    # B0. Coaching staff URL provided
    evaluator.add_custom_node(
        result=_is_nonempty_url(uni.coaching_staff_url),
        id=f"u{idx+1}_coaching_staff_url_provided",
        desc="Provides an official athletics URL showing the current football coaching staff/head coach.",
        parent=b_node,
        critical=True
    )
    # B1. Division I football confirmation (via official sources)
    b1_node = evaluator.add_leaf(
        id=f"u{idx+1}_division1_football_confirmed",
        desc="Confirms varsity football team competes in NCAA Division I (FBS).",
        parent=b_node,
        critical=True
    )
    div_sources = _dedup_urls([
        uni.coaching_staff_url,
        uni.athletics_website_url,
        *(uni.membership_urls or []),
    ])
    await evaluator.verify(
        claim="This university's varsity football team competes in NCAA Division I (FBS).",
        node=b1_node,
        sources=div_sources,
        additional_instruction=(
            "It is valid to infer Division I (FBS) from confirmed Big Ten membership because the Big Ten is an NCAA Division I FBS conference. "
            "Prefer explicit statements on official pages (athletics site, bigten.org, NCAA) if present."
        )
    )
    # B2. Head coach name provided
    evaluator.add_custom_node(
        result=bool(_norm(uni.football_head_coach_name)),
        id=f"u{idx+1}_head_coach_name_provided",
        desc="Provides the current head football coach's name.",
        parent=b_node,
        critical=True
    )
    # B3. Head coach official title provided
    evaluator.add_custom_node(
        result=bool(_norm(uni.football_head_coach_title)),
        id=f"u{idx+1}_head_coach_title_provided",
        desc="Provides the current head football coach's official title.",
        parent=b_node,
        critical=True
    )
    # B4. Coach/name/title supported by official staff page
    b4_node = evaluator.add_leaf(
        id=f"u{idx+1}_coach_name_title_supported",
        desc="Head coach name and title are supported by the official athletics staff page.",
        parent=b_node,
        critical=True
    )
    coach_name = _norm(uni.football_head_coach_name)
    coach_title = _norm(uni.football_head_coach_title)
    await evaluator.verify(
        claim=f"The current head football coach is {coach_name} with title '{coach_title}'.",
        node=b4_node,
        sources=uni.coaching_staff_url if _is_nonempty_url(uni.coaching_staff_url) else None,
        additional_instruction=(
            "Use only the official athletics site page. Allow reasonable title variants like 'Head Coach' or 'Head Football Coach', "
            "and accept 'interim' labels as fulfilling 'current head coach'."
        )
    )

    # C. Graduate sports-related program (critical)
    c_node = evaluator.add_parallel(
        id=f"u{idx+1}_C_grad_program",
        desc="Graduate program in specified sports/athletics-related fields.",
        parent=uni_node,
        critical=True
    )
    exists_norm = _yes_no_norm(uni.grad_program_exists)
    # C1. States whether relevant graduate program exists (Yes/No)
    evaluator.add_custom_node(
        result=exists_norm in {"yes", "no"},
        id=f"u{idx+1}_grad_program_exists_stated",
        desc="States whether the university offers a relevant graduate degree (Yes/No).",
        parent=c_node,
        critical=True
    )
    # C2. If yes, provides exact official program name
    evaluator.add_custom_node(
        result=(exists_norm == "no") or (exists_norm == "yes" and bool(_norm(uni.grad_program_name))),
        id=f"u{idx+1}_grad_program_name_if_yes",
        desc="If YES, provides the exact official program name; otherwise N/A.",
        parent=c_node,
        critical=True
    )
    # C3. If yes, provides housing college or school
    evaluator.add_custom_node(
        result=(exists_norm == "no") or (exists_norm == "yes" and bool(_norm(uni.grad_program_college_school))),
        id=f"u{idx+1}_grad_program_college_if_yes",
        desc="If YES, provides the housing college/school; otherwise N/A.",
        parent=c_node,
        critical=True
    )
    # C4. If yes, provides official program website URL (and verify content)
    # Implement as one verification leaf: must be provided if YES and page should support the claim.
    if exists_norm == "yes":
        c4_node = evaluator.add_leaf(
            id=f"u{idx+1}_grad_program_url_supported",
            desc="If YES, the official program website URL is provided and supports the graduate program claim.",
            parent=c_node,
            critical=True
        )
        prog_claim = (
            f"This is an official program page for a graduate (Master's or Doctorate) program in the sports/athletics field "
            f"(e.g., Sport(s) Management, Sport Administration, Athletic Administration, Athletic Training, or a direct equivalent) "
            f"at {_norm(uni.official_name)}."
        )
        await evaluator.verify(
            claim=prog_claim,
            node=c4_node,
            sources=uni.grad_program_url if _is_nonempty_url(uni.grad_program_url) else None,
            additional_instruction=(
                "Only accept official university/college/school pages. "
                "Allow direct equivalents like 'Sport Administration', 'Sport and Entertainment Management', "
                "'Kinesiology with Sport Management concentration', or 'Athletic Training (MS/MA/PhD)'. "
                "The page should clearly indicate graduate-level study (Master's or Doctorate)."
            )
        )
    else:
        # If NO, we still need to ensure URL presence isn't required. We won't add the verification leaf.
        pass

    # D. Graduate Assistantships (critical)
    d_node = evaluator.add_parallel(
        id=f"u{idx+1}_D_grad_assistantships",
        desc="Graduate assistantship availability in athletics with evidence.",
        parent=uni_node,
        critical=True
    )
    # D1. States Yes/No
    evaluator.add_custom_node(
        result=_yes_no_norm(uni.ga_yes_no) in {"yes", "no"},
        id=f"u{idx+1}_ga_yes_no_stated",
        desc="States whether athletics offers graduate assistantship positions (Yes/No).",
        parent=d_node,
        critical=True
    )
    # D2. Evidence URL provided
    evaluator.add_custom_node(
        result=_is_nonempty_url(uni.ga_evidence_url),
        id=f"u{idx+1}_ga_evidence_url_provided",
        desc="Provides an official URL supporting the GA availability/non-availability claim.",
        parent=d_node,
        critical=True
    )
    # D3. Claim supported by the evidence URL
    d3_node = evaluator.add_leaf(
        id=f"u{idx+1}_ga_claim_supported",
        desc="Graduate assistantship Yes/No claim is supported by the official evidence URL.",
        parent=d_node,
        critical=True
    )
    ga_norm = _yes_no_norm(uni.ga_yes_no)
    ga_claim = (
        f"The athletic department (or relevant official unit) offers graduate assistantship positions."
        if ga_norm == "yes"
        else f"The athletic department (or relevant official unit) does not offer graduate assistantship positions."
    )
    await evaluator.verify(
        claim=ga_claim,
        node=d3_node,
        sources=uni.ga_evidence_url if _is_nonempty_url(uni.ga_evidence_url) else None,
        additional_instruction=(
            "Only accept official university/athletics/HR/graduate-school pages. "
            "For YES: look for terms like 'Graduate Assistant', 'GA positions', 'Athletics Graduate Assistant'. "
            "For NO: the page must clearly indicate unavailability or explicit policy stating no GA roles in athletics. "
            "Ambiguous or unrelated pages should not be accepted."
        )
    )

    # E. Educational requirements (critical)
    e_node = evaluator.add_parallel(
        id=f"u{idx+1}_E_edu_requirements",
        desc="Typical minimum educational requirement with supporting evidence.",
        parent=uni_node,
        critical=True
    )
    # E1. Requirement stated
    evaluator.add_custom_node(
        result=bool(_norm(uni.min_edu_requirement)),
        id=f"u{idx+1}_min_edu_req_stated",
        desc="States the typical minimum educational requirement for assistant coaching or athletics admin positions.",
        parent=e_node,
        critical=True
    )
    # E2. Supporting URL provided
    evaluator.add_custom_node(
        result=_is_nonempty_url(uni.min_edu_requirement_url),
        id=f"u{idx+1}_min_edu_req_url_provided",
        desc="Provides an official supporting URL (job posting, careers, HR/policy).",
        parent=e_node,
        critical=True
    )
    # E3. Requirement supported by the URL
    e3_node = evaluator.add_leaf(
        id=f"u{idx+1}_min_edu_req_supported",
        desc="Minimum education requirement is supported by the official URL.",
        parent=e_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The typical minimum educational requirement for assistant coaching or athletics administrative roles is: '{_norm(uni.min_edu_requirement)}'.",
        node=e3_node,
        sources=uni.min_edu_requirement_url if _is_nonempty_url(uni.min_edu_requirement_url) else None,
        additional_instruction=(
            "Accept only official pages (HR, careers, university job postings, or policy documents). "
            "Look for explicit 'required' language (e.g., 'Bachelor's degree required'). "
            "If it only states 'preferred' and not 'required', do not accept as meeting a minimum requirement."
        )
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
    # Initialize evaluator; use sequential at root so failing global checks can skip per-university checks
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract all university entries mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Prepare lists for set-level checks
    all_entries: List[UniversityEntry] = list(extracted.universities or [])
    provided_entries: List[UniversityEntry] = [u for u in all_entries if _is_provided_university_entry(u)]

    # ---------------------- Global Set Requirements ----------------------- #
    global_node = evaluator.add_parallel(
        id="global_set_requirements",
        desc="Requirements that apply to the set of four universities.",
        parent=root,
        critical=True
    )

    # 1) Exactly four universities provided
    exactly_four_leaf = evaluator.add_custom_node(
        result=(len(provided_entries) == 4),
        id="provides_exactly_four_universities",
        desc="Provides exactly four university entries.",
        parent=global_node,
        critical=True
    )

    # 2) All four universities are distinct
    keys_for_distinct: List[str] = []
    for i, u in enumerate(provided_entries):
        key = _norm_lower(u.official_name)
        if not key:
            # Fallback to domain keys to help detect duplicates when name is missing
            dom = _extract_domain(u.athletics_website_url) or (_extract_domain(u.membership_urls[0]) if u.membership_urls else None)
            key = (dom or f"missing_name_{i}").casefold()
        keys_for_distinct.append(key)
    all_distinct = len(set(keys_for_distinct)) == len(provided_entries) == 4
    evaluator.add_custom_node(
        result=all_distinct,
        id="all_four_universities_are_distinct",
        desc="All four universities are distinct institutions (no duplicates).",
        parent=global_node,
        critical=True
    )

    # 3) Excludes Iowa and Michigan State
    excludes_banned = all(not _is_banned_university(u.official_name) for u in provided_entries)
    evaluator.add_custom_node(
        result=excludes_banned,
        id="excludes_iowa_and_michigan_state",
        desc="None of the four universities is Iowa or Michigan State.",
        parent=global_node,
        critical=True
    )

    # 4) Information stated as current as of March 2026
    current_leaf = evaluator.add_leaf(
        id="stated_current_as_of_march_2026",
        desc="The response presents the information as current as of March 2026.",
        parent=global_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the information is current as of March 2026.",
        node=current_leaf,
        additional_instruction="Search the answer text for an explicit statement such as 'current as of March 2026' or equivalent phrasing."
    )

    # ---------------------- Per-University Evaluations -------------------- #
    uni_set_node = evaluator.add_parallel(
        id="universities_1_to_4",
        desc="Evaluate each of the four university entries (partial credit across entries allowed).",
        parent=root,
        critical=False
    )

    # Select first four entries (pad with empty entries if fewer than 4, per standard policy)
    selected: List[UniversityEntry] = provided_entries[:4]
    while len(selected) < 4:
        selected.append(UniversityEntry())

    # Build per-university verification subtrees
    for i in range(4):
        await verify_university(evaluator, uni_set_node, selected[i], i)

    # Return evaluation summary
    return evaluator.get_summary()